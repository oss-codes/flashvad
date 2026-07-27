---
license: cc-by-4.0
library_name: flashvad
language:
  - ar
  - en
  - gu
  - hi
  - kn
  - pa
  - ta
  - te
  - ur
tags:
  - audio
  - voice-activity-detection
  - vad
  - streaming
  - onnx
  - pytorch
  - telephony
  - webrtc
---

# FlashVAD v0.1 model card

## Summary

FlashVAD v0.1 is a 46,170-parameter causal streaming voice-activity detector.
It consumes 16 kHz mono audio, produces one speech probability every 10 ms,
and keeps independent convolutional, recurrent, feature, and detector state
per call.

**Author:** Himanshu Maurya

This checkpoint is an **alpha research preview** for integration testing,
browser demonstrations, and shadow evaluation. It is not approved for
production use and is not validated as a general multilingual or India/GCC
call model.

## Artifacts

| Artifact | SHA-256 |
|---|---|
| `flashvad-v0.1.pt` | `ca9e35475518466b2a1f2e89b4953cd1e26e3d8c513cdcf265ab319e74e2b288` |
| `flashvad-stream.onnx` | `9a88e34bf3118d60e25a16cb622cb394e2f3ab71445b0aa5957df6f1d5f1b6ba` |
| `config.json` | `0b1ad372808f7c67cea5a1ca4b41a817714c9a0b0cf49b3baa56fe8d5f64ad2b` |
| `detector-calibration.json` | `b5d000e0406d81fbd87a9e66194a877fa8433a76783faac8275c7969c43051b4` |

The ONNX graph accepts precomputed 43-dimensional causal features. Use
`src/flashvad/features.py` or
`report-site/src/lib/vad-features.mjs`; it is not a raw-waveform graph.

The public ONNX file is self-contained and stripped of exporter stack traces,
local paths, and private build metadata.

## Download and source

Download the complete model repository:

```bash
hf download oss-codes/flashvad --local-dir flashvad-model
```

Source code and runtime integrations are published separately at
[`oss-codes/flashvad`](https://github.com/oss-codes/flashvad).

The ONNX graph does not accept raw waveform audio. It expects the causal
43-dimensional features described below, with independent feature and model
state for every call.

## Intended use

Appropriate current uses:

- research and architecture evaluation;
- functional integration with browser, LiveKit, Pipecat, SIP, or PSTN stacks;
- latency and concurrency measurement;
- shadow-mode comparison on consented, labelled call audio.

Do not use this checkpoint as the sole basis for:

- emergency, medical, legal, financial, or safety-critical decisions;
- call recording consent or compliance decisions;
- a production multilingual accuracy claim;
- semantic end-of-turn detection.

Reset all feature, model, resampler, and detector state when a call ends or an
audio discontinuity occurs.

## Architecture

- 25 ms causal analysis frame and 10 ms hop;
- 40 log-mel bands plus energy, zero-crossing rate, and spectral flatness;
- four causal depthwise temporal blocks with dilations 1, 2, 4, and 8;
- one 64-unit GRU;
- speech and auxiliary event heads;
- 184,680 bytes of FP32 parameters.

## Training inputs and provenance limit

Historical training notes associated with the retained checkpoint report:

- 558 derived clips from nine FLEURS configurations: Arabic, English,
  Gujarati, Hindi, Kannada, Punjabi, Tamil, Telugu, and Urdu;
- 288 AMI meeting clips with meeting-family-disjoint train/validation splits;
- 64 MUSAN noise clips;
- weak frame targets from the official Silero VAD model.

FLEURS, AMI, and MUSAN attribution is in `NOTICE`; dataset audio is not
distributed here.

Those counts and corpus names are not embedded as complete provenance in the
public checkpoint. The exact retained training manifests, their digests, source
revisions, and teacher-output digest were not preserved, so the checkpoint's
training run is **not bit-for-bit reproducible** from the public tree. This is a
provenance limitation, not evidence of broader accuracy. Future release
candidates must preserve those records before training begins.

## Evaluation

The retained checkpoint was repeatedly inspected on TEN VAD's public
30-recording set while research candidates were compared. At the retained
detector policy, the descriptive results are:

- 26,243 frames;
- ROC-AUC: 0.882;
- raw F1 at the configured threshold: 0.886;
- hysteresis-decision F1: 0.889;
- false-alarm rate: 26.3%;
- miss rate: 13.0%.

Because the public set influenced research decisions, this is an exploratory
external-set result—not an untouched test, independent benchmark, or
production generalization estimate. Language is recorded as `und`, and codec,
channel, device, and SNR are unknown.

The machine-readable report is
`benchmarks/flashvad-v0.1/ten-public-evaluation.json`. Its false-alarm rate is
too high for production.

## Known limitations

- Quiet speech, music, TTS leakage, echo, television, laughter, singing, and
  overlapping speakers may cause misses or false triggers.
- Read speech and meetings do not cover real carrier, device, packet-loss, and
  room conditions.
- Language presence in training does not prove per-language performance.
- Linear 8-to-16 kHz conversion prioritizes causal speed, not audio fidelity.
- VAD cannot determine whether a speaker has semantically completed a turn.

A production candidate needs consented, human-labelled, speaker-disjoint calls
with predeclared per-slice gates and a test set untouched until final
evaluation.

## Licences

Repository source code is MIT-licensed. The retained model artifacts are
separately available under CC BY 4.0; see
`MODEL_LICENSE.md` and `NOTICE`. Third-party datasets, models, and benchmark
materials retain their own terms.
