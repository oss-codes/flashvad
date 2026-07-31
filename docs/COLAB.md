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

## Inference provider benchmark

FlashVAD is a small, stateful model. A T4 can be slower than a CPU for one call because every
10 ms hop launches a small graph and must return a speech probability to the detector. Keep the
CPU execution provider as the production default until the complete call path is faster on the
target machine.

On a Colab GPU runtime, run the provider matrix before selecting CUDA:

```bash
python scripts/benchmark_colab_onnx.py --batches 1 8 32 128
```

The script installs a CUDA 12.8-compatible ONNX Runtime GPU wheel, verifies the pinned model
digest, and compares these paths:

- CPU execution with NumPy inputs;
- CUDA execution with the normal NumPy API, including state transfers;
- CUDA I/O binding with recurrent state kept on the GPU and only the speech logit returned.

The report is model-only. It excludes the NumPy frontend and call scheduling, so confirm the
winning provider end to end:

The recorded Tesla T4 run in `benchmarks/flashvad-v0.1/onnx-provider-colab-t4.json` used the
release model digest, ONNX Runtime 1.26.0, 50 warmup steps, and 200 measured steps:

| Batch | CPU median call | CUDA I/O median call | CPU amortized/stream | CUDA amortized/stream |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 155.31 us | 651.31 us | 155.31 us | 651.31 us |
| 8 | 537.65 us | 695.31 us | 67.21 us | 86.91 us |
| 32 | 1,929.27 us | 677.90 us | 60.29 us | 21.18 us |
| 128 | 6,164.23 us | 693.82 us | 48.16 us | 5.42 us |

Use CPU for one call or a small number of calls. CUDA becomes a throughput optimization only
when roughly 32 or more streams are ready in the same scheduler tick. Do not delay a call to fill
a batch: run the streams that are already ready, preserve every stream's recurrent state, and
record the queue-delay p50, p95, and p99. The batch-32 CUDA I/O-binding p99 was 7.60 ms in this
short run, so the median speedup is not enough by itself for a production decision.
The benchmark code now also rejects provider fallback and compares the full logit trace. Rerun
the current script and replace the saved matrix before publishing performance claims.

```bash
python scripts/benchmark_call_scenarios.py \
  --provider CUDAExecutionProvider \
  --calls 32 \
  --hops 500 \
  --scenario float32-16k
```

CUDA is opt-in and requires `onnxruntime-gpu` with versions compatible with the runtime CUDA and
cuDNN libraries:

```python
from flashvad.runtime import OnnxStreamingVadModel

owner = OnnxStreamingVadModel.load_bundled(
    providers=[
        (
            "CUDAExecutionProvider",
            {"do_copy_in_default_stream": "1"},
        )
    ]
)
```

For a single live call, prefer the faster end-to-end result, which may remain CPU. For many
simultaneous calls, the ONNX graph supports a dynamic batch axis. Same-tick micro-batches can
improve throughput, but they require a scheduler that preserves per-call recurrent state and
measures the added queue delay. The current per-call adapter does not silently batch calls.
