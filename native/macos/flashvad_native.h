#ifndef FLASHVAD_NATIVE_H
#define FLASHVAD_NATIVE_H

#include <stddef.h>

#include "flashvad_weights.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    void *dft_setup;
    float history[FV_HISTORY_SAMPLES];
    float recurrent[FV_RECURRENT_DIM];
    float caches[FV_TOTAL_CACHE_FLOATS];
    float pending[FV_HOP_SAMPLES];
    size_t pending_count;

    _Alignas(16) float frame[FV_FRAME_SAMPLES];
    _Alignas(16) float fft_even[FV_N_FFT / 2];
    _Alignas(16) float fft_odd[FV_N_FFT / 2];
    _Alignas(16) float fft_real[FV_N_FFT / 2];
    _Alignas(16) float fft_imag[FV_N_FFT / 2];
    _Alignas(16) float power[FV_POWER_BINS];
    _Alignas(16) float log_power[FV_POWER_BINS];
    _Alignas(16) float feature[FV_FEATURE_DIM];
    _Alignas(16) float hidden[FV_HIDDEN_DIM];
    _Alignas(16) float residual[FV_HIDDEN_DIM];
    _Alignas(16) float projected[FV_HIDDEN_DIM];
    _Alignas(16) float normalized[FV_RECURRENT_DIM];
    _Alignas(16) float input_gates[3 * FV_RECURRENT_DIM];
    _Alignas(16) float recurrent_gates[3 * FV_RECURRENT_DIM];
} FlashVadState;

size_t flashvad_state_size(void);

/*
 * Initializes one independent call stream. Model weights live in read-only
 * process memory; only the small state above is per call.
 */
int flashvad_init(FlashVadState *state);
void flashvad_reset(FlashVadState *state);
void flashvad_destroy(FlashVadState *state);

/*
 * Low-level parity APIs. Feature extraction updates the audio history; model
 * inference updates convolution and recurrent state.
 */
int flashvad_extract_features(
    FlashVadState *state,
    const float samples[FV_HOP_SAMPLES],
    float output[FV_FEATURE_DIM]
);
float flashvad_model_step(
    FlashVadState *state,
    const float feature[FV_FEATURE_DIM]
);

/* Process exactly one 10 ms hop and return a speech probability. */
float flashvad_process_hop(
    FlashVadState *state,
    const float samples[FV_HOP_SAMPLES]
);

/*
 * Accept arbitrary chunk sizes and return the number of emitted probabilities.
 * The call is rejected without consuming input when output_capacity is smaller
 * than floor((pending samples + sample_count) / FV_HOP_SAMPLES).
 */
#define FLASHVAD_INSUFFICIENT_OUTPUT ((size_t)-1)
size_t flashvad_push(
    FlashVadState *state,
    const float *samples,
    size_t sample_count,
    float *probabilities,
    size_t output_capacity
);

#ifdef __cplusplus
}
#endif

#endif
