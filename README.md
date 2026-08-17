# FlashVAD

FlashVAD is a small causal voice-activity detector and integration toolkit for
low-latency voice calls. It emits one speech probability every 10 ms and
provides:

- a 46,170-parameter streaming model;
- a shared-session ONNX runtime with independent per-call state;
- an embedded Apple Accelerate runtime for macOS;
- browser microphone and file testing through ONNX Runtime Web;
- LiveKit Agents and Pipecat adapters;
- PCMU, PCMA, and PCM16 telephone ingress;
- reproducible concurrent-call and Colab CPU/CUDA benchmarks.

Created by **Himanshu Maurya** and published through the
[`oss-codes`](https://github.com/oss-codes) organization.

[Demo](https://flash.oss.codes) ·
[PyPI](https://pypi.org/project/flashvad/) ·
[npm](https://www.npmjs.com/package/flashvad) ·
[Hugging Face](https://huggingface.co/oss-codes/flashvad)

> **Status: alpha research preview.** The software paths are testable, but the
> retained checkpoint is not approved for production. Its public-set false-alarm
> rate is 26.3%, and multilingual India/GCC call accuracy has not been validated.

FlashVAD is acoustic VAD, not semantic end-of-turn detection. Voice agents
should combine its fast speech/silence evidence with ASR stability, semantic
completion, interruption state, and a timeout policy.

## Install

| Package | Install | Contents |
| --- | --- | --- |
| [PyPI `flashvad`](https://pypi.org/project/flashvad/) | `pip install flashvad` | Python runtime, telephony adapters, LiveKit and Pipecat integrations, CLI |
| [npm `flashvad`](https://www.npmjs.com/package/flashvad) | `npm install flashvad onnxruntime-web` | Browser ONNX/WASM runtime, feature frontend, detector |
| [Hugging Face](https://huggingface.co/oss-codes/flashvad) | — | Model artifact and card, CC BY 4.0 |

Both packages are pre-releases (`0.1.0a1` and `0.1.0-alpha.1`) and carry the
false-alarm rate described above.

```bash
pip install flashvad
```

For the `flashvad` command on its own PATH, without a project environment:

```bash
uv tool install flashvad     # or: pipx install flashvad
```

There is no Homebrew formula. ONNX Runtime ships wheels and no source
distribution, which Homebrew's Python formula path cannot consume, so a tap
would have to pin platform-specific wheels that `uv tool` and `pipx` already
resolve correctly.

Install only the integrations you use:

```bash
pip install "flashvad[livekit]"
pip install "flashvad[pipecat]"
pip install "flashvad[export]"
```

Training and data preparation need a source checkout:

```bash
git clone https://github.com/oss-codes/flashvad.git
cd flashvad
pip install ".[data,training]"
```

For repository development:

```bash
uv sync --all-extras
uv run pytest
uv run ruff check .
```

The runtime-only installation does not install or import PyTorch. The wheel
contains the self-contained ONNX graph, its metadata and licence, plus the
generated Accelerate weights used by the macOS path.

## Run the retained checkpoint

```bash
uv run flashvad benchmark \
  --checkpoint models/flashvad-v0.1/flashvad-v0.1.pt

uv run flashvad benchmark-onnx \
  --model models/flashvad-v0.1/flashvad-stream.onnx
```

Create one cached process-level ONNX owner and one small stream per call:

```python
from flashvad.runtime import OnnxStreamingVadModel

model = OnnxStreamingVadModel.load_bundled(threads=1)
call = model.new_stream()
probabilities, events = call.push(audio_float32_16khz)
call.reset()
```

Never load an ONNX session inside the per-packet or per-call hot path.

The bundled loader verifies the ONNX and metadata digests, validates the tensor
contract, and rejects unavailable execution providers or silent provider
fallback. An explicitly requested CUDA provider therefore cannot quietly run on
CPU and produce a misleading benchmark.

## Telephone audio

The telephone adapter decodes 8 kHz G.711 and causally converts it to the
model's 16 kHz stream:

```python
from flashvad.telephony import TelephonyVadStream

# Fastest packaged path on macOS. Use load_onnx("pcmu") elsewhere.
call = TelephonyVadStream.load_native("pcmu")
probabilities, events = call.push(payload_bytes)
```

RTP sequence handling, jitter buffering, authentication, and packet-loss
concealment remain the media server or SBC's responsibility.

Measure the complete packaged call path, including codec ingress, causal
features, inference, and detection:

```bash
python scripts/benchmark_call_scenarios.py \
  --provider CPUExecutionProvider \
  --calls 32 \
  --hops 500 \
  --output artifacts/call-scenarios.json
```

The benchmark reports queue-delay and end-to-end p50, p95, and p99 values for
PCMU, PCMA, 8 kHz PCM16, and direct 16 kHz float32 audio. It intentionally does
not claim to measure RTP jitter, packet loss, network I/O, or production
accuracy.

## LiveKit Agents

Load the model in worker prewarm and reuse the VAD owner:

```python
from flashvad.integrations.livekit import FlashVadLiveKit

vad = FlashVadLiveKit.load(threads=1)

# Pass `vad` to AgentSession or another LiveKit Agents voice pipeline.
```

The adapter implements LiveKit's current `VAD`/`VADStream` contract, accepts
mono input rates such as 48 kHz, resamples to 16 kHz, emits inference/start/end
events, and resets all call state on flush.

On macOS, use an already-built embedded library to remove ONNX/Python inference
from the per-hop path:

```python
vad = FlashVadLiveKit.load_native()
```

## Pipecat

```python
from flashvad.integrations.pipecat import FlashVadPipecatAnalyzer

vad_analyzer = FlashVadPipecatAnalyzer()

# Provide `vad_analyzer` to the Pipecat transport or VAD processor.
```

The Pipecat adapter implements `VADAnalyzer`, accepts 8 or 16 kHz mono PCM16,
and processes exact 10 ms frames with persistent model state.

For the embedded macOS path:

```python
vad_analyzer = FlashVadPipecatAnalyzer.load_native()
```

## Browser package

```bash
npm install flashvad onnxruntime-web
```

```js
import { FlashVad } from "flashvad";
import modelUrl from "flashvad/model?url";

const ort = await import("onnxruntime-web/wasm");
const vad = await FlashVad.create({ ort, model: modelUrl, sampleRate: 48_000 });
const { probabilities, events } = await vad.push(float32Samples);
```

`onnxruntime-web` is a peer dependency, so an application already running ORT
keeps one copy. The package source is `packages/web`, which also holds the
feature frontend the demo site imports, so the published package and the demo
cannot drift apart numerically. `speech_start` is back-dated by the detector's
pre-roll, so keep a short ring buffer and seek to `event.frame` rather than
acting from the event's arrival time. See
[`packages/web/README.md`](packages/web/README.md).

## Browser demo

```bash
cd report-site
npm ci
npm run dev
```

The dark-mode report includes a local microphone/file playground. Audio,
probabilities, and filenames remain in the browser. The heavy inference runtime
is lazy-loaded only when the playground becomes visible, and the site supports
both root and repository-subpath hosting. The official Astro Cloudflare adapter
and Wrangler configuration are included for a local Workers-compatible build;
nothing is deployed automatically.

## macOS native runtime

On Apple Silicon, `load_native()` compiles the packaged generated weights once
with Apple Accelerate into `~/Library/Caches/flashvad`, then reuses the
content-addressed library. No checkpoint, repository clone, Python model, ONNX
parser, or compile step remains in the per-call path.

To export and benchmark a different checkpoint:

```bash
uv run flashvad benchmark-native \
  --checkpoint models/flashvad-v0.1/flashvad-v0.1.pt \
  --output artifacts/native
```

The native source is included in built wheels. The benchmark writes a
machine-readable `benchmark.json` containing the checkpoint digest, machine,
toolchain, protocol, initialization, and warm tail timings.

## Colab training and provider benchmark

Use Colab for licensed multilingual training data or for measuring batched GPU
throughput. It is not required for macOS native inference, browser inference,
or single-call CPU operation. The complete handoff is in
[`docs/COLAB.md`](docs/COLAB.md).

On a GPU runtime, compare CPU, normal CUDA, and CUDA I/O binding before choosing
a provider:

```bash
python scripts/benchmark_colab_onnx.py --batches 1 8 32 128
```

The retained T4 artifact shows that CPU was faster for a single stream, while
CUDA I/O binding improved amortized model throughput at larger ready batches.
Do not delay a live call just to fill a batch. Preserve independent recurrent
state for every call and include scheduler queue delay in the deployment
decision. The saved T4 result and its limitations are documented in
[`benchmarks/README.md`](benchmarks/README.md).

## Reproducible speed snapshot

On the retained Apple M4 Pro run, FlashVAD native used 3.53× less
audio-normalized warm compute than Silero ONNX, 14.2× less than TEN native,
and 45.2× less than FireRed streaming ONNX. Exact upstream revisions, binary
and model hashes, scope differences, and aggregate timing statistics are recorded in
[`benchmarks/flashvad-v0.1/external-runtime-m4-pro.json`](benchmarks/flashvad-v0.1/external-runtime-m4-pro.json).
These are compute comparisons, not accuracy or end-of-turn rankings.

To reproduce them, use `scripts/benchmark_official_vads.py` with official
checkouts and artifacts. External code or weights are never vendored.

## Accuracy and claims

The retained model was repeatedly consulted on TEN VAD's public 30-recording
set while research candidates were compared. Its 0.882 ROC-AUC, 0.889
hysteresis F1, 26.3% false-alarm rate, and 13.0% miss rate are therefore
**exploratory external-set results**, not an untouched test or production
generalization estimate.

Read [the model card](MODEL_CARD.md), [validation record](docs/LOCAL_RESULTS.md),
and [data requirements](docs/DATA.md) before making an accuracy claim. A
production candidate still needs:

- consented, human-labelled India/GCC and target-customer calls;
- speaker-disjoint training, calibration, and untouched test splits;
- predeclared false-alarm, miss, boundary, and short-utterance gates;
- per-language, codec, device, SNR, and noise reporting;
- shadow-traffic validation.

For research iteration, the repository includes attributed FLEURS and MUSAN
preparation plus an eight-trial, multi-seed Mac sweep. It preserves upstream
train/development splits and forbids TEN, Silero, and FireRed public benchmark
sets from candidate selection. A trial must reach at least 0.85 detector F1,
at most 15% false alarms, and at most 20% misses on the development split. See
[the reproducible training workflow](docs/DATA.md#reproducible-mac-training-sweep).

The completed 128-epoch run selected a candidate with 0.914 development
detector F1 and 5.47% false alarms. It was **not** promoted: on the descriptive
TEN check it lowered false alarms to 12.35% but regressed to 0.822 detector F1
and a 27.36% miss rate. The retained checkpoint remains the balanced default.
See the
[machine-readable candidate record](benchmarks/flashvad-multilingual-alpha/training-and-evaluation.json).

## Licences

Repository source code is MIT-licensed. The retained model artifacts are
separately available under CC BY 4.0 from
[Hugging Face](https://huggingface.co/oss-codes/flashvad) and are also bundled
in both published packages; see
[`models/flashvad-v0.1/MODEL_LICENSE.md`](models/flashvad-v0.1/MODEL_LICENSE.md)
and [`NOTICE`](NOTICE). Third-party datasets, models, and benchmarks retain
their own terms.
