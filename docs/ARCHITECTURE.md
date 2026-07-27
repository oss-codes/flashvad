# Architecture

## Latency contract

FlashVAD emits one causal probability every 10 ms. It never observes future audio. With the
default 25 ms analysis window, the first frames are left-padded and a decision is available as soon
as the first 10 ms hop arrives.

Model compute latency and algorithmic decision latency are measured separately:

- **compute latency:** feature extraction plus one neural step;
- **onset latency:** frame collection plus the configured onset confirmation;
- **offset latency:** configured stop confirmation;
- **turn latency:** VAD offset plus semantic endpoint policy.

## Frontend

Default input is 16 kHz mono float PCM. The frontend computes:

- 40 area-normalized log-mel bands;
- log RMS energy;
- log spectral flatness;
- zero-crossing rate.

Pitch is intentionally absent from v1. Pitch is useful but must not become a hard requirement:
whispered speech and unvoiced phonemes can be weakly periodic. A learned periodicity feature can be
added only if an ablation proves it improves the real-call benchmark.

## Encoder

The projected feature sequence passes through four causal residual depthwise-separable temporal
blocks at dilations 1, 2, 4, and 8. A single 64-unit GRU provides longer conversational memory. All
streaming convolution caches and recurrent state are explicit, and tests require streaming output
to match offline output.

The primary head predicts speech. The auxiliary head reserves logits for speech, music, and
non-speech vocal events. It enables multi-task training when suitable labels become available.

## Why this is original

The implementation uses common published building blocks such as mel features, depthwise temporal
convolution and a GRU, but the feature set, layer geometry, state API, training loss and detector are
implemented independently. No external VAD weights or source files are incorporated.

## Size target

The default model must stay below 250K parameters. The stronger target is 50K–150K parameters and
under 250 KB after int8 quantization. Accuracy and boundary latency are release gates; size alone is
not.
