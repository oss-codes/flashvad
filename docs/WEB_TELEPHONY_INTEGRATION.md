# Web and telephone integration

Date: 2026-07-26

FlashVAD should be instantiated once per process and keep only the small
recurrent/cache state per call. Do not create an ONNX session for every caller.
The VAD emits acoustic evidence; turn completion remains a separate policy.

## Browser voice calls

```text
microphone
  -> AudioWorklet (10 ms transferable chunks)
  -> streaming resampler to 16 kHz
  -> 43 causal features
  -> persistent ONNX Runtime Web / WASM session
  -> per-call recurrent state
  -> 0.80/0.50 hysteresis
  -> speech events + pre-roll
```

The local report implements this pipeline in
`report-site/src/components/VadPlayground.tsx`. Audio never leaves the browser.
The worklet now forwards 10 ms capture chunks, avoiding the roughly 43 ms
buffering delay caused by a fixed 2,048-sample block at 48 kHz.

Recommended production behavior:

- create one browser inference session, then reset only recurrent and feature
  state between calls;
- lazy-load the WebAssembly-only ONNX Runtime entry point when the user opens
  the tester instead of adding it to the initial page payload;
- keep 200-300 ms of pre-roll outside the model and attach it when speech starts;
- transfer typed-array buffers from the AudioWorklet instead of cloning them;
- keep WASM SIMD as the default for this tiny batch-one model;
- feature-detect WebNN/WebGPU only as optional experiments because dispatch and
  initialization can cost more than a 46K-parameter CPU model;
- send speech and pre-roll upstream only when bandwidth savings matter. If the
  server needs independent auditing or diarization, send the full call.

[MDN AudioWorklet](https://developer.mozilla.org/en-US/docs/Web/API/AudioWorklet)
documents that audio processing runs on a separate Web Audio thread.
[WebNN](https://www.w3.org/TR/webnn/) remains a developing web standard, so it
must not be the only runtime.

## SIP, RTP, and PSTN calls

```text
carrier / SBC
  -> RTP jitter and sequence handling
  -> PCMU (PT 0) or PCMA (PT 8), 8 kHz mono
  -> exact G.711 decode
  -> causal 8 kHz -> 16 kHz conversion
  -> persistent native Accelerate or ONNX session
  -> per-call VAD state
  -> ASR stream
  -> semantic endpoint / interruption policy
```

[RFC 3551](https://datatracker.ietf.org/doc/html/rfc3551) assigns RTP payload
type 0 to PCMU and 8 to PCMA at 8 kHz. The project provides exact G.711 decoding
and a chunk-invariant causal resampler in `flashvad.telephony`.

```python
from flashvad.config import DetectorConfig
from flashvad.runtime import OnnxStreamingVadModel
from flashvad.telephony import TelephonyVadStream

model = OnnxStreamingVadModel.load_bundled()
call = TelephonyVadStream(
    model.new_stream(DetectorConfig(
        start_threshold=0.80,
        stop_threshold=0.50,
        start_frames=3,
        stop_frames=4,
        pre_roll_frames=3,
    )),
    codec="pcmu",
)

# One decoded RTP or media-stream payload at a time.
probabilities, events = call.push(payload_bytes)
```

For a Twilio bidirectional Media Stream, validate `X-Twilio-Signature`, decode
the base64 `media.payload`, and pass the raw `audio/x-mulaw`, 8 kHz, mono bytes
to `call.push`. Twilio's official
[Media Streams documentation](https://www.twilio.com/docs/voice/media-streams)
and [WebSocket message schema](https://www.twilio.com/docs/voice/media-streams/websocket-messages)
define that transport.

The adapter deliberately does not parse RTP headers or implement a jitter
buffer. Those belong at the SBC/media-server boundary where packet sequence,
timestamps, late packets, and packet-loss concealment are visible. Reset the
VAD state when a call or media stream ends.

## LiveKit Agents

Install the optional adapter and load it during worker prewarm:

```bash
pip install ".[livekit]"
```

```python
from flashvad.integrations.livekit import FlashVadLiveKit

vad = FlashVadLiveKit.load(threads=1)
```

`FlashVadLiveKit` implements LiveKit Agents' `VAD` and `VADStream` contracts.
It accepts mono LiveKit audio frames, resamples once to 16 kHz, emits exact
10 ms inference events, produces start/end events, retains bounded pre-roll,
and treats `flush()` as a hard state boundary.

Create the `FlashVadLiveKit` owner once per worker. Calling `stream()` creates
only the per-call audio, feature, model, and detector state.

On macOS, the packaged generated weights are compiled into a content-addressed
Accelerate library on first use and reused from the user cache:

```python
vad = FlashVadLiveKit.load_native()
```

## Pipecat

```bash
pip install ".[pipecat]"
```

```python
from flashvad.integrations.pipecat import FlashVadPipecatAnalyzer

vad_analyzer = FlashVadPipecatAnalyzer()
```

The analyzer implements Pipecat's `VADAnalyzer` interface for 8 or 16 kHz mono
PCM16. Its default policy uses the retained 0.80 start confidence, 30 ms start
confirmation, 40 ms stop confirmation, and no separate volume gate. Pass an
explicit `VADParams` if the surrounding pipeline needs a different operating
point.

```python
native_vad_analyzer = FlashVadPipecatAnalyzer.load_native()
```

## Concurrency and batching

For a single call, batching frames adds avoidable decision latency. For many
simultaneous calls:

1. share one immutable ONNX session per worker;
2. retain independent feature, convolution, recurrent, and detector state per
   call;
3. schedule ready 10 ms frames across calls;
4. optionally micro-batch only frames that arrive in the same scheduler tick;
5. measure p50, p95, and p99 queue delay as well as model compute.

On macOS the embedded Accelerate path is already about 11.42 microseconds per
10 ms hop, so per-call native invocation is usually simpler than waiting for a
batch. At that speed, transport, resampling, ASR, and network jitter dominate.

The full Python-owned native call stream measures 13.208 microseconds median,
including input conversion and detector logic. Load the process owner during
worker startup or prewarm; never compile the library or create an ONNX session
inside a call.

## VAD is not endpointing

Use the acoustic VAD for immediate speech start, barge-in, and cheap silence
evidence. End the user turn only after combining:

- VAD silence duration;
- ASR partial stability;
- punctuation or semantic completion;
- whether the user is interrupting the agent;
- a maximum wait deadline.

The [two-pass endpointing paper](https://arxiv.org/abs/2401.08916) supports a
fast first-pass detector plus a delayed arbitrator. LiveKit similarly documents
a separate [semantic turn detector](https://docs.livekit.io/agents/logic/turns/turn-detector/).

## Market map

| Option | Best fit | Main tradeoff |
|---|---|---|
| WebRTC VAD | Small deterministic C baseline at 8/16 kHz | Classic statistical model; weaker difficult-noise/domain adaptation |
| FlashVAD | Local macOS/web/PSTN research and controllable training | Current India/GCC accuracy gate is still open |
| Silero VAD | Mature MIT ONNX/JIT local deployment at 8/16 kHz | Validate on the exact call domain and operating point |
| TEN VAD | Small 10/16 ms ONNX/WASM reference | Inspect license restrictions and validate on the exact call domain |
| FireRedVAD | Multilingual teacher and server/GPU reference | Public call-domain comparison data is still needed |
| Picovoice Cobra | Supported commercial on-device/browser SDK | AccessKey and commercial service dependency |
| Deepgram endpointing | Cloud ASR-coupled pause/utterance events | Network, vendor cost, and transcript-service coupling |
| LiveKit turn detector | Semantic end-of-turn above a VAD | Heavier than frame VAD and not a replacement for barge-in detection |

Primary product references:

- [Silero VAD](https://github.com/snakers4/silero-vad)
- [TEN VAD](https://github.com/TEN-framework/ten-vad)
- [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)
- [WebRTC VAD source](https://webrtc.googlesource.com/src/+/refs/heads/main/common_audio/vad/)
- [Picovoice Cobra](https://picovoice.ai/docs/cobra/)
- [Deepgram endpointing](https://developers.deepgram.com/docs/endpointing)

These alternatives should be compared on the same audio, hardware, frame
stride, and operating-point constraints. Vendor-reported accuracy is
directional evidence, not proof of superiority for India/GCC calls.

The pinned M4 Pro runtime snapshot in
`benchmarks/flashvad-v0.1/external-runtime-m4-pro.json` reports 3.53× lower
audio-normalized warm compute than Silero ONNX, 14.2× lower than TEN native,
and 45.2× lower than FireRed streaming ONNX on its declared scopes. Those are
compute results only—not accuracy, endpoint latency, or production rankings.
