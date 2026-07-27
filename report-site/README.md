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

The microphone and file playground executes FlashVAD locally. Silero VAD, TEN
VAD, and FireRedVAD appear only as clearly labelled reference previews based on
official released artifacts and the same-machine measurements recorded in the
repository.

The visual system follows the shared OSS Codes terminal theme used by
`oss-codes/voice-cost` and `oss-codes/website`: Geist, square data surfaces,
and the lime signal palette.

This package contains no deployment adapter or hosting configuration.
`npm run dev` and `npm run preview` are local-only commands.
