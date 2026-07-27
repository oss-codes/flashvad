# FlashVAD report and browser demo

This package builds the static FlashVAD benchmark report with Astro. The
microphone tester and model comparison are focused React islands; the retained
streaming ONNX model runs entirely in the browser with ONNX Runtime Web.

## Local preview

Requires Node.js 22.13 or newer.

```bash
npm ci
npm test
npm run dev
```

The microphone and file playground executes FlashVAD locally. Uploaded audio
can be played against its synchronized speech timeline. For a stereo call, the
demo treats the left channel as the user and the right channel as the AI by
default, with a control to swap those roles. Mono or mixed audio is shown only
as speech/silence because VAD alone cannot reliably identify the speaker.

Silero VAD, TEN VAD, and FireRedVAD appear only as clearly labelled reference
previews based on official released artifacts and the same-machine measurements
recorded in the repository.

The visual system follows the shared OSS Codes terminal theme used by
`oss-codes/voice-cost` and `oss-codes/website`: Geist, square data surfaces,
and the lime signal palette.

## Cloudflare Workers preview

The official Astro Cloudflare adapter is configured for Workers. The report
page remains prerendered, so Cloudflare serves the generated HTML and hashed
assets directly while the Worker runtime is available for future server routes.

```bash
npm run preview:cloudflare
```

This builds the project and starts the adapter-provided local Cloudflare
preview. It does not deploy. No Cloudflare account or credentials are required
for the local preview.
