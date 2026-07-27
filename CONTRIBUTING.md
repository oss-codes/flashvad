# Contributing to FlashVAD

FlashVAD welcomes reproducible improvements to the model, runtime, data
pipeline, evaluation, and browser demo.

## Development setup

```bash
uv sync --all-extras
uv run ruff check .
uv run pytest

cd report-site
npm ci
npm run lint
npm test
```

On non-macOS systems, the Apple Accelerate parity tests are skipped. Changes to
the native runtime should also be tested on Apple Silicon.

## Evidence requirements

- Keep VAD and semantic end-of-turn claims separate.
- Report the threshold-selection split and keep frozen test thresholds fixed.
- Include false-alarm and miss rates, not only ROC-AUC or F1.
- Break down multilingual claims by language, region, channel, codec, device,
  noise condition, and code-switching where data permits.
- Add provenance and licence information for every dataset or model artifact.
- Do not commit raw call audio, credentials, generated build output, or
  third-party model weights without explicit redistribution rights.

## Pull requests

Keep each pull request focused, explain the measured tradeoff, and include the
commands used for verification. Performance changes should state the machine,
runtime, hop size, warmup, iteration count, and whether frontend cost is
included.
