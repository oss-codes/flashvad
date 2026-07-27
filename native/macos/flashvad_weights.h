/*
 * Generated model parameters. Their licence follows the source checkpoint
 * and is not automatically covered by the FlashVAD repository source licence.
 */
#ifndef FLASHVAD_WEIGHTS_H
#define FLASHVAD_WEIGHTS_H

#define FV_SAMPLE_RATE 16000
#define FV_FRAME_SAMPLES 400
#define FV_HOP_SAMPLES 160
#define FV_HISTORY_SAMPLES 240
#define FV_N_FFT 512
#define FV_POWER_BINS 257
#define FV_N_MELS 40
#define FV_FEATURE_DIM 43
#define FV_HIDDEN_DIM 64
#define FV_RECURRENT_DIM 64
#define FV_KERNEL_SIZE 3
#define FV_BLOCK_COUNT 4
#define FV_TOTAL_CACHE_FLOATS 1920

extern const int fv_block_dilations[FV_BLOCK_COUNT];
extern const float fv_window[FV_FRAME_SAMPLES];
extern const float fv_mel_filterbank[FV_N_MELS * FV_POWER_BINS];
extern const float fv_input_norm_weight[FV_FEATURE_DIM];
extern const float fv_input_norm_bias[FV_FEATURE_DIM];
extern const float fv_input_projection_weight[FV_HIDDEN_DIM * FV_FEATURE_DIM];
extern const float fv_input_projection_bias[FV_HIDDEN_DIM];
extern const float fv_depthwise_weight[FV_BLOCK_COUNT * FV_HIDDEN_DIM * FV_KERNEL_SIZE];
extern const float fv_pointwise_weight[FV_BLOCK_COUNT * FV_HIDDEN_DIM * FV_HIDDEN_DIM];
extern const float fv_pointwise_bias[FV_BLOCK_COUNT * FV_HIDDEN_DIM];
extern const float fv_block_norm_weight[FV_BLOCK_COUNT * FV_HIDDEN_DIM];
extern const float fv_block_norm_bias[FV_BLOCK_COUNT * FV_HIDDEN_DIM];
extern const float fv_gru_weight_ih[3 * FV_RECURRENT_DIM * FV_HIDDEN_DIM];
extern const float fv_gru_weight_hh[3 * FV_RECURRENT_DIM * FV_RECURRENT_DIM];
extern const float fv_gru_bias_ih[3 * FV_RECURRENT_DIM];
extern const float fv_gru_bias_hh[3 * FV_RECURRENT_DIM];
extern const float fv_output_norm_weight[FV_RECURRENT_DIM];
extern const float fv_output_norm_bias[FV_RECURRENT_DIM];
extern const float fv_speech_head_weight[FV_RECURRENT_DIM];
extern const float fv_speech_head_bias[1];

#endif
