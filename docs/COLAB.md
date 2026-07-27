# Colab training handoff

Colab is useful after the real speech/noise manifests exist. It is not needed for architecture,
streaming correctness, export, or Mac latency work.

## Recommended runtime

- L4, A100, or better GPU;
- high-RAM runtime for large manifests;
- persistent checkpoints in Google Drive or an object store;
- source audio kept outside the git repository.

## Inputs expected

```text
corpus/
  train.jsonl
  valid.jsonl
  test.jsonl
  audio/...
```

Every manifest must follow [DATA.md](DATA.md). Before uploading, validate consent, commercial-use
rights, retention policy and whether derivatives/checkpoints may be distributed.

## Reproducible run

```bash
git clone <project-repository>
cd <project-repository>
pip install ".[dev,export,training]"
python scripts/run_training_sweep.py \
  --sweep configs/sweeps/multilingual-mac.json \
  --region-profile configs/regions/india_gcc.json \
  --train-manifest /content/corpus/train.jsonl \
  --valid-manifest /content/corpus/valid.jsonl \
  --noise-manifest /content/corpus/noise.jsonl \
  --output /content/drive/MyDrive/flashvad/india-gcc
```

The sweep selects CUDA automatically and records the training-code, runtime,
region-profile, manifest, and audio-content fingerprints alongside every
checkpoint digest. Keep the complete output directory in persistent storage.

## Scaling sequence

1. Debug on 1–5 hours and overfit a tiny subset.
2. Train on 50–100 hours across all priority languages.
3. Inspect per-language and per-domain failure slices.
4. Add hard negatives from consented calls.
5. Scale only after the benchmark demonstrates that more data improves frozen-test performance.

Large training is not a substitute for labels. The strongest release gate remains a manually
adjudicated real-call test set covering India, GCC, 8 kHz PSTN, VoIP, noise, code-switching and
agent-TTS echo.
