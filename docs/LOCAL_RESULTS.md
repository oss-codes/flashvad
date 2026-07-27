# FlashVAD v0.1 validation record

Date: 2026-07-27

This document records the retained clean-release artifacts, the pinned
same-machine speed comparison, and the rejected multilingual candidate.

## Release classification

**Software status:** alpha, integration-ready for evaluation.

**Checkpoint status:** research preview, not production-ready.

The runtime, packaging, and adapters can be tested independently of the model's
unmet accuracy gate.

## Measured machine

- Apple M4 Pro;
- macOS 26.5.2;
- Python 3.12.3;
- one calling thread;
- warm streaming state.

Each machine-readable runtime artifact records its own digest, protocol,
runtime/toolchain, and timestamp.

## FlashVAD runtime paths

| Path | Measured scope | Median |
|---|---|---:|
| Embedded Apple Accelerate | C frontend + model | 11.417 µs |
| Native Python call stream | conversion + C frontend + model + detector | 13.208 µs |
| ONNX Python call stream | conversion + Python frontend + model + detector | 89.333 µs |
| ONNX Runtime CPU | model step only | 58.875 µs |
| PyTorch reference | frontend + model | 857.729 µs |

Authoritative artifacts:

- `benchmarks/flashvad-v0.1/native-runtime-m4-pro.json`;
- `benchmarks/flashvad-v0.1/call-runtime-m4-pro.json`;
- `benchmarks/flashvad-v0.1/onnx-runtime-m4-pro.json`;
- `benchmarks/flashvad-v0.1/pytorch-runtime-m4-pro.json`.

The native path initializes in 3.542 µs median on the recorded run. Its
benchmark executable is 267,392 bytes and dynamic library is 266,680 bytes.
It uses no Python, Torch, or ONNX parser in the hot path.

Runtime numbers from different scopes must not be presented as an accuracy or
product ranking.

## Pinned external runtime snapshot

The retained external run used one thread, 1,000 warmup calls, 10,000 measured
calls, all-zero input, and each model's official warm streaming state. The
compute advantage is the competitor real-time factor divided by FlashVAD
native real-time factor, which normalizes Silero's 32 ms hop.

| Runtime | Median call | Hop | RTF | FlashVAD native compute advantage | Scope |
|---|---:|---:|---:|---:|---|
| FlashVAD native | 11.333 µs | 10 ms | 0.001133 | 1× | frontend + model |
| Silero ONNX | 128.166 µs | 32 ms | 0.004005 | 3.53× | model + recurrent state |
| TEN native | 161.041 µs | 10 ms | 0.016104 | 14.2× | frontend + model + decision |
| FireRed streaming ONNX | 511.959 µs | 10 ms | 0.051196 | 45.2× | model + cache |

The FireRed and Silero figures exclude their external postprocessing; FireRed
also excludes its frontend. These exclusions make the comparison conservative
for FlashVAD but still prevent an apples-to-apples accuracy or endpoint claim.
Exact revisions, hashes, initialization, tails, and license notes are in
`benchmarks/flashvad-v0.1/external-runtime-m4-pro.json`.

## Telephone ingress

The retained NumPy ingress snapshot reports 4.792 µs median for either PCMU or
PCMA decode plus causal 8-to-16 kHz conversion of a 20 ms packet. This is
transport preprocessing, not model inference. See
`benchmarks/flashvad-v0.1/telephony-ingress-m4-pro.json`.

The implementation is packet-boundary invariant and resets interpolation and
VAD state at call boundaries.

## Browser

The static Astro report uses dark mode by default. It:

- lazy-hydrates the interactive islands;
- loads the ONNX/WASM runtime only when the user starts a test;
- supports root and repository-subpath asset URLs;
- processes microphone and uploaded audio locally;
- uses 10 ms transferable AudioWorklet chunks;
- exposes the retained model licence and attribution beside the downloadable
  browser artifact.

Browser timing includes ONNX Runtime Web and is shown live in the playground.
It is not the native Accelerate timing.

## LiveKit, Pipecat, and calls

- `FlashVadLiveKit` implements the current LiveKit Agents `VAD`/`VADStream`
  contract and resamples mono input to exact 10 ms, 16 kHz hops.
- `FlashVadPipecatAnalyzer` implements Pipecat `VADAnalyzer` for 8 or 16 kHz
  mono PCM16.
- both retain per-call state while sharing one process-level ONNX owner;
- PCMU, PCMA, PCM16, resampling, reset, and framework event paths have focused
  tests.

## Exploratory public-set result

The retained checkpoint's descriptive result on TEN VAD's public 30-recording
set is:

| Metric | Result |
|---|---:|
| Frames | 26,243 |
| ROC-AUC | 0.882 |
| Raw F1 | 0.886 |
| Hysteresis-decision F1 | 0.889 |
| False-alarm rate | 26.3% |
| Miss rate | 13.0% |
| Segment recall | 91.3% |
| Premature-end rate | 27.0% |

This set was consulted across research candidates. The result is exploratory,
not an untouched test. Language, channel, codec, device, and SNR are not known,
so it cannot validate India, GCC, multilingual, or production-call accuracy.

## Multilingual Mac training sweep

The 2026-07-27 run trained eight configurations for 16 epochs each on Apple
MPS: 128 measured epochs in total. Training used 1,056 items and validation
used 264 items across 12 FLEURS configurations plus MUSAN hard negatives.
Content hashes prove that training, validation, and noise audio did not overlap.

The selected `focal1-conservative-seed-301` candidate reached:

| Development metric | Result |
|---|---:|
| Frames | 105,600 |
| ROC-AUC | 0.968 |
| Raw F1 | 0.895 |
| Detector F1 | 0.914 |
| False-alarm rate | 5.47% |
| Miss rate | 8.45% |
| Segment recall | 94.2% |
| False triggers | 67 |

TEN, Silero, and FireRed public data were forbidden from candidate selection.
After the model and detector were frozen, the candidate was evaluated
descriptively on TEN's 30-recording public set. It reduced false alarms to
12.35%, but reached only 0.849 ROC-AUC, 0.822 detector F1, and a 27.36% miss
rate. The retained checkpoint remains the default because its cross-domain
ROC-AUC and detector F1 are higher. This is a rejection of an automatic
promotion, not a claim that the retained checkpoint is production-ready.

The compact machine-readable record is
`benchmarks/flashvad-multilingual-alpha/training-and-evaluation.json`.

Segment recall and false-trigger counts use a permissive any-overlap rule: a
predicted and reference segment match when they share at least one 10 ms frame.
Frame F1, false-alarm rate, and miss rate do not depend on this segment rule.

## Release decision

The repository can be presented as a working alpha integration and runtime
project after all automated and manual checks pass. The retained checkpoint
must continue to be labelled a research preview.

A production model release remains blocked until a new consented,
human-labelled, speaker-disjoint call test set is evaluated exactly once after
model and threshold freeze.
