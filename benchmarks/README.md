# Benchmark artifacts

`flashvad-v0.1/` contains the retained, machine-readable evidence used by the
documentation and report:

- `ten-public-evaluation.json`: exploratory frame, segment, boundary, and
  item-bootstrap results on TEN's repeatedly consulted public 30-file set;
- `native-runtime-m4-pro.json`: embedded Apple Accelerate runtime;
- `onnx-runtime-m4-pro.json`: one-thread ONNX Runtime model-step measurement;
- `pytorch-runtime-m4-pro.json`: one-thread PyTorch reference runtime;
- `onnx-provider-colab-t4.json`: model-only CPU/CUDA provider and batch-size matrix on a
  Colab Tesla T4, including output parity and amortized throughput;
- `telephony-ingress-m4-pro.json`: PCMU/PCMA decode plus causal resampling;
- `call-runtime-m4-pro.json`: real Python per-call native and ONNX stream wrappers;

Run `python scripts/benchmark_call_scenarios.py --calls 8 --hops 100` to measure
deterministic round-robin concurrent-call queue delay and end-to-end per-hop
latency for PCMU, PCMA, 8 kHz little-endian PCM16 telephony, and direct 16 kHz
float32 audio after warmup. The report records the artifact digest, machine,
runtime, and whether the causal frontend is included. RTP jitter, packet loss,
network I/O, and transport policy remain outside the model adapter.

The saved Colab matrix predates the benchmark's strict-fallback and full-trace parity hardening.
Its requested and active providers, final-output parity, model digest, and timing fields were
captured successfully, but rerun the current script before using the numbers in release claims.
- `external-runtime-m4-pro.json`: pinned same-machine Silero, TEN, FireRed,
  and FlashVAD runtime snapshot.

`flashvad-multilingual-alpha/training-and-evaluation.json` records the completed
eight-trial, 128-epoch multilingual Mac sweep and the decision not to promote
its selected checkpoint after the descriptive TEN check exposed a
cross-domain F1 and miss-rate regression.

The external-model harness is `scripts/benchmark_official_vads.py`. Its retained
artifact records exact source revisions, artifact hashes, machine/runtime
metadata, and a declared protocol. Its compute-advantage field uses
audio-normalized real-time factor because Silero's hop is 32 ms while the other
measured paths use 10 ms.

To reconstruct the descriptive TEN manifest from an independently obtained
checkout of the recorded revision:

```bash
python scripts/convert_scv_manifest.py \
  --input /path/to/ten-vad/testset \
  --output data/ten-public/manifest.jsonl \
  --relative-paths \
  --condition ten-public-testset \
  --expected-items 30 \
  --source-repository https://github.com/TEN-framework/ten-vad \
  --source-revision 22a3bcd4509d0faaa8eef4881e8af5f39c178950
```

The converter verifies mono 16 kHz PCM16 input and writes file hashes plus a
manifest digest to `manifest.provenance.json`. TEN's repository has additional
Agora licence conditions, and the test-set README does not grant a separate
redistribution licence for the constituent audio. Do not vendor the test set;
review its terms before use.

All retained timings are warm local FlashVAD measurements on the Apple M4 Pro
described in `docs/LOCAL_RESULTS.md`. Different scopes mean raw call latency is
not an accuracy, endpoint-latency, or product ranking.
