#include "flashvad_native.h"

#include <Accelerate/Accelerate.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

static void layer_norm(
    const float *input,
    const float *weight,
    const float *bias,
    size_t count,
    float *output
) {
    float mean = 0.0f;
    for (size_t index = 0; index < count; ++index) {
        mean += input[index];
    }
    mean /= (float)count;

    float variance = 0.0f;
    for (size_t index = 0; index < count; ++index) {
        const float centered = input[index] - mean;
        variance += centered * centered;
    }
    const float scale = 1.0f / sqrtf(variance / (float)count + 1.0e-5f);
    for (size_t index = 0; index < count; ++index) {
        output[index] = (input[index] - mean) * scale * weight[index] + bias[index];
    }
}

static void affine(
    const float *weight,
    const float *bias,
    int rows,
    int columns,
    const float *input,
    float *output
) {
    memcpy(output, bias, (size_t)rows * sizeof(float));
    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        rows,
        columns,
        1.0f,
        weight,
        columns,
        input,
        1,
        1.0f,
        output,
        1
    );
}

static float sigmoid_scalar(float value) {
    return 1.0f / (1.0f + expf(-value));
}

static void silu_in_place(float *values, size_t count) {
    for (size_t index = 0; index < count; ++index) {
        values[index] *= sigmoid_scalar(values[index]);
    }
}

size_t flashvad_state_size(void) {
    return sizeof(FlashVadState);
}

int flashvad_init(FlashVadState *state) {
    if (state == NULL) {
        return -1;
    }
    memset(state, 0, sizeof(*state));
    state->dft_setup = (void *)vDSP_DFT_zrop_CreateSetup(
        NULL,
        FV_N_FFT,
        vDSP_DFT_FORWARD
    );
    return state->dft_setup == NULL ? -1 : 0;
}

void flashvad_reset(FlashVadState *state) {
    if (state == NULL) {
        return;
    }
    void *setup = state->dft_setup;
    memset(state, 0, sizeof(*state));
    state->dft_setup = setup;
}

void flashvad_destroy(FlashVadState *state) {
    if (state == NULL) {
        return;
    }
    if (state->dft_setup != NULL) {
        vDSP_DFT_DestroySetup((vDSP_DFT_Setup)state->dft_setup);
    }
    memset(state, 0, sizeof(*state));
}

int flashvad_extract_features(
    FlashVadState *state,
    const float samples[FV_HOP_SAMPLES],
    float output[FV_FEATURE_DIM]
) {
    if (state == NULL || samples == NULL || output == NULL || state->dft_setup == NULL) {
        return -1;
    }

    memcpy(state->frame, state->history, sizeof(state->history));
    memcpy(
        state->frame + FV_HISTORY_SAMPLES,
        samples,
        FV_HOP_SAMPLES * sizeof(float)
    );

    memset(state->fft_even, 0, sizeof(state->fft_even));
    memset(state->fft_odd, 0, sizeof(state->fft_odd));
    for (size_t index = 0; index < FV_FRAME_SAMPLES; ++index) {
        const float windowed = state->frame[index] * fv_window[index];
        if ((index & 1U) == 0U) {
            state->fft_even[index / 2] = windowed;
        } else {
            state->fft_odd[index / 2] = windowed;
        }
    }
    vDSP_DFT_Execute(
        (vDSP_DFT_Setup)state->dft_setup,
        state->fft_even,
        state->fft_odd,
        state->fft_real,
        state->fft_imag
    );

    /* Accelerate's forward real DFT uses a factor of two. */
    state->power[0] = 0.25f * state->fft_real[0] * state->fft_real[0];
    for (size_t index = 1; index < FV_N_FFT / 2; ++index) {
        const float real = state->fft_real[index];
        const float imaginary = state->fft_imag[index];
        state->power[index] = 0.25f * (real * real + imaginary * imaginary);
    }
    state->power[FV_N_FFT / 2] =
        0.25f * state->fft_imag[0] * state->fft_imag[0];

    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        FV_N_MELS,
        FV_POWER_BINS,
        1.0f,
        fv_mel_filterbank,
        FV_POWER_BINS,
        state->power,
        1,
        0.0f,
        state->feature,
        1
    );
    float mel_mean = 0.0f;
    for (size_t index = 0; index < FV_N_MELS; ++index) {
        const float energy = fmaxf(state->feature[index], 1.0e-8f);
        state->feature[index] = logf(energy);
        mel_mean += state->feature[index];
    }
    mel_mean /= (float)FV_N_MELS;
    for (size_t index = 0; index < FV_N_MELS; ++index) {
        state->feature[index] -= mel_mean;
    }

    float square_sum = 0.0f;
    float arithmetic_sum = 0.0f;
    float log_power_sum = 0.0f;
    size_t crossings = 0;
    for (size_t index = 0; index < FV_FRAME_SAMPLES; ++index) {
        square_sum += state->frame[index] * state->frame[index];
        if (index > 0 && state->frame[index] * state->frame[index - 1] < 0.0f) {
            ++crossings;
        }
    }
    for (size_t index = 0; index < FV_POWER_BINS; ++index) {
        arithmetic_sum += state->power[index];
        state->log_power[index] = logf(fmaxf(state->power[index], 1.0e-10f));
        log_power_sum += state->log_power[index];
    }

    const float mean_square = fmaxf(square_sum / (float)FV_FRAME_SAMPLES, 1.0e-10f);
    const float arithmetic =
        fmaxf(arithmetic_sum / (float)FV_POWER_BINS, 1.0e-10f);
    const float mean_log_power = log_power_sum / (float)FV_POWER_BINS;
    state->feature[FV_N_MELS] = 0.5f * logf(mean_square);
    state->feature[FV_N_MELS + 1] = mean_log_power - logf(arithmetic);
    state->feature[FV_N_MELS + 2] =
        (float)crossings / (float)(FV_FRAME_SAMPLES - 1);

    memcpy(output, state->feature, FV_FEATURE_DIM * sizeof(float));
    memcpy(
        state->history,
        state->frame + FV_HOP_SAMPLES,
        FV_HISTORY_SAMPLES * sizeof(float)
    );
    return 0;
}

float flashvad_model_step(
    FlashVadState *state,
    const float feature[FV_FEATURE_DIM]
) {
    if (state == NULL || feature == NULL) {
        return NAN;
    }
    layer_norm(
        feature,
        fv_input_norm_weight,
        fv_input_norm_bias,
        FV_FEATURE_DIM,
        state->feature
    );
    affine(
        fv_input_projection_weight,
        fv_input_projection_bias,
        FV_HIDDEN_DIM,
        FV_FEATURE_DIM,
        state->feature,
        state->hidden
    );
    silu_in_place(state->hidden, FV_HIDDEN_DIM);

    size_t cache_offset = 0;
    for (size_t block = 0; block < FV_BLOCK_COUNT; ++block) {
        const size_t dilation = (size_t)fv_block_dilations[block];
        const size_t cache_size = dilation * (FV_KERNEL_SIZE - 1);
        float *block_cache = state->caches + cache_offset;
        memcpy(state->residual, state->hidden, sizeof(state->hidden));

        for (size_t channel = 0; channel < FV_HIDDEN_DIM; ++channel) {
            float *channel_cache = block_cache + channel * cache_size;
            const float *kernel = fv_depthwise_weight
                + (block * FV_HIDDEN_DIM + channel) * FV_KERNEL_SIZE;
            state->projected[channel] =
                kernel[0] * channel_cache[0]
                + kernel[1] * channel_cache[dilation]
                + kernel[2] * state->hidden[channel];
            memmove(
                channel_cache,
                channel_cache + 1,
                (cache_size - 1) * sizeof(float)
            );
            channel_cache[cache_size - 1] = state->hidden[channel];
        }

        affine(
            fv_pointwise_weight + block * FV_HIDDEN_DIM * FV_HIDDEN_DIM,
            fv_pointwise_bias + block * FV_HIDDEN_DIM,
            FV_HIDDEN_DIM,
            FV_HIDDEN_DIM,
            state->projected,
            state->hidden
        );
        layer_norm(
            state->hidden,
            fv_block_norm_weight + block * FV_HIDDEN_DIM,
            fv_block_norm_bias + block * FV_HIDDEN_DIM,
            FV_HIDDEN_DIM,
            state->projected
        );
        silu_in_place(state->projected, FV_HIDDEN_DIM);
        for (size_t channel = 0; channel < FV_HIDDEN_DIM; ++channel) {
            state->hidden[channel] =
                state->residual[channel] + state->projected[channel];
        }
        cache_offset += FV_HIDDEN_DIM * cache_size;
    }

    affine(
        fv_gru_weight_ih,
        fv_gru_bias_ih,
        3 * FV_RECURRENT_DIM,
        FV_HIDDEN_DIM,
        state->hidden,
        state->input_gates
    );
    affine(
        fv_gru_weight_hh,
        fv_gru_bias_hh,
        3 * FV_RECURRENT_DIM,
        FV_RECURRENT_DIM,
        state->recurrent,
        state->recurrent_gates
    );
    for (size_t index = 0; index < FV_RECURRENT_DIM; ++index) {
        const float reset = sigmoid_scalar(
            state->input_gates[index] + state->recurrent_gates[index]
        );
        const float update = sigmoid_scalar(
            state->input_gates[FV_RECURRENT_DIM + index]
            + state->recurrent_gates[FV_RECURRENT_DIM + index]
        );
        const float candidate = tanhf(
            state->input_gates[2 * FV_RECURRENT_DIM + index]
            + reset * state->recurrent_gates[2 * FV_RECURRENT_DIM + index]
        );
        state->recurrent[index] =
            (1.0f - update) * candidate + update * state->recurrent[index];
    }

    layer_norm(
        state->recurrent,
        fv_output_norm_weight,
        fv_output_norm_bias,
        FV_RECURRENT_DIM,
        state->normalized
    );
    const float logit = cblas_sdot(
        FV_RECURRENT_DIM,
        fv_speech_head_weight,
        1,
        state->normalized,
        1
    ) + fv_speech_head_bias[0];
    return sigmoid_scalar(logit);
}

float flashvad_process_hop(
    FlashVadState *state,
    const float samples[FV_HOP_SAMPLES]
) {
    if (flashvad_extract_features(state, samples, state->feature) != 0) {
        return NAN;
    }
    return flashvad_model_step(state, state->feature);
}

size_t flashvad_push(
    FlashVadState *state,
    const float *samples,
    size_t sample_count,
    float *probabilities,
    size_t output_capacity
) {
    if (state == NULL || (sample_count > 0 && samples == NULL)) {
        return 0;
    }
    if (sample_count > SIZE_MAX - state->pending_count) {
        return FLASHVAD_INSUFFICIENT_OUTPUT;
    }
    const size_t required =
        (state->pending_count + sample_count) / FV_HOP_SAMPLES;
    if (output_capacity < required || (required > 0 && probabilities == NULL)) {
        return FLASHVAD_INSUFFICIENT_OUTPUT;
    }
    size_t consumed = 0;
    size_t emitted = 0;
    while (consumed < sample_count) {
        const size_t available = FV_HOP_SAMPLES - state->pending_count;
        const size_t remaining = sample_count - consumed;
        const size_t take = available < remaining ? available : remaining;
        memcpy(
            state->pending + state->pending_count,
            samples + consumed,
            take * sizeof(float)
        );
        state->pending_count += take;
        consumed += take;
        if (state->pending_count == FV_HOP_SAMPLES) {
            probabilities[emitted++] =
                flashvad_process_hop(state, state->pending);
            state->pending_count = 0;
        }
    }
    return emitted;
}
