# flashvad

Causal streaming voice activity detector for browsers and real-time voice agents.

FlashVAD reads audio in 10 ms hops and emits a speech probability plus stable
`speech_start` / `speech_end` events, for barge-in, interruption handling,
silence timers, and deciding when to send audio to speech-to-text.

**Alpha research preview.** The retained checkpoint reached 0.882 ROC-AUC and
0.889 detector F1 on TEN VAD's public test set, but at a 26.3% false-alarm rate,
which is too high for production. That public set was consulted during
development, so those results are exploratory rather than a clean final test.
See the [model card](https://github.com/oss-codes/flashvad/blob/main/MODEL_CARD.md).

## Install

```bash
npm install flashvad onnxruntime-web
```

`onnxruntime-web` is a peer dependency: this package does not bundle it, so an
app already running ORT for another model keeps a single copy.

## Use

```js
import { FlashVad } from "flashvad";
import modelUrl from "flashvad/model?url";

const ort = await import("onnxruntime-web/wasm");
ort.env.wasm.numThreads = 1;

const vad = await FlashVad.create({
  ort,
  model: modelUrl,
  sampleRate: audioContext.sampleRate, // resampled to 16 kHz when it differs
});

const { probabilities, events } = await vad.push(float32Samples);
for (const event of events) {
  console.log(event.kind, event.timeSeconds);
}

vad.reset(); // clears per-call state; the session and weights are untouched
```

Push whatever chunk size your capture path produces. Samples that do not
complete a 10 ms hop are retained until the next call.

Create the session once per page and one `FlashVad` per call. Never construct an
`InferenceSession` inside the per-packet path.

## Reading `speech_start`

`speech_start` is deliberately back-dated by `preRollFrames` so the reported
frame sits *before* the onset the detector needed several frames to confirm.
Acting only from the moment the event arrives will clip the start of the word,
so keep a short ring buffer of recent audio and seek back to `event.frame`.

## Tuning

```js
const vad = await FlashVad.create({
  ort,
  model: modelUrl,
  detector: { startThreshold: 0.8, stopThreshold: 0.5, startFrames: 3, stopFrames: 4 },
});
```

Defaults are the calibrated operating point from the released checkpoint. The
right threshold depends on what you do with the events, and the costs are not
symmetric:

- **barge-in** punishes false starts — a false onset cuts the agent off mid-sentence
- **STT gating** punishes missed onsets — clipped leading phonemes are unrecoverable
- **end-of-turn** punishes late offsets — they become felt latency

## Scope

FlashVAD does fast acoustic speech detection only. No transcription, no language
identification, no speaker identification, no semantic end-of-turn detection.
With stereo audio, channel assignment provides "user" and "agent" labels;
FlashVAD does not identify the speaker.

## Licence

Code is MIT. The model artifact in `model/` is CC BY 4.0, see
`model/MODEL_LICENSE.md`.
