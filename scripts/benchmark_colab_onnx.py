#!/usr/bin/env python3
"""Benchmark FlashVAD ONNX execution providers on a Colab GPU runtime.

The benchmark keeps model state on the GPU for the I/O-binding path and
compares it with the normal NumPy API. It is intentionally model-only: the
NumPy feature extractor and call scheduler are not included.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_COMMIT = "09c079ce75724367dc5791979165e369f5b8b379"
DEFAULT_MODEL_SHA256 = "9a88e34bf3118d60e25a16cb622cb394e2f3ab71445b0aa5957df6f1d5f1b6ba"
DEFAULT_ORT_GPU = "1.26.0"


def _install_onnxruntime_gpu(version: str) -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "--yes", "onnxruntime"],
        check=False,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--upgrade",
            "--force-reinstall",
            "--no-deps",
            f"onnxruntime-gpu=={version}",
        ],
        check=True,
    )


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * q))]


def _summary(samples_us: list[float], batch: int) -> dict[str, float | int]:
    return {
        "batch": batch,
        "median_call_us": statistics.median(samples_us),
        "p95_call_us": _percentile(samples_us, 0.95),
        "p99_call_us": _percentile(samples_us, 0.99),
        "amortized_median_us_per_stream": statistics.median(samples_us) / batch,
    }


def _shape(shape: list[Any], batch: int) -> tuple[int, ...]:
    return tuple(value if isinstance(value, int) else batch for value in shape)


def _session_options(ort: Any, threads: int = 1) -> Any:
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return options


def _require_provider(session: Any, expected: str) -> None:
    active = session.get_providers()
    if not active or active[0] != expected:
        raise RuntimeError(
            f"requested {expected} but ONNX Runtime activated {active}; refusing fallback"
        )
    session.disable_fallback()


def _numpy_benchmark(
    ort: Any,
    model: Path,
    provider: str,
    batch: int,
    iterations: int,
    warmup: int,
) -> tuple[dict[str, Any], np.ndarray]:
    session = ort.InferenceSession(
        str(model),
        sess_options=_session_options(ort),
        providers=[provider],
    )
    _require_provider(session, provider)
    inputs = {
        item.name: np.zeros(_shape(item.shape, batch), dtype=np.float32)
        for item in session.get_inputs()
    }
    state_outputs = {
        output.name: output.name.removeprefix("next_")
        for output in session.get_outputs()
        if output.name.startswith("next_")
    }

    def step() -> np.ndarray:
        outputs = session.run(None, inputs)
        for metadata, value in zip(session.get_outputs(), outputs, strict=True):
            state_name = state_outputs.get(metadata.name)
            if state_name is not None:
                inputs[state_name] = value
        named_outputs = {
            metadata.name: value
            for metadata, value in zip(session.get_outputs(), outputs, strict=True)
        }
        return named_outputs["speech_logits"]

    for _ in range(warmup):
        step()
    samples_us: list[float] = []
    logits = np.empty((batch, 1), dtype=np.float32)
    trace = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        logits = step()
        samples_us.append((time.perf_counter_ns() - started) / 1_000)
        trace.append(logits.copy())
    return (
        {
            "path": "numpy",
            "requested_provider": provider,
            "session_providers": session.get_providers(),
            **_summary(samples_us, batch),
        },
        np.stack(trace),
    )


def _cuda_iobinding_benchmark(
    ort: Any,
    model: Path,
    batch: int,
    iterations: int,
    warmup: int,
) -> tuple[dict[str, Any], np.ndarray]:
    provider_options = {"do_copy_in_default_stream": "1"}
    session = ort.InferenceSession(
        str(model),
        sess_options=_session_options(ort),
        providers=[("CUDAExecutionProvider", provider_options)],
    )
    _require_provider(session, "CUDAExecutionProvider")
    input_shapes = {item.name: _shape(item.shape, batch) for item in session.get_inputs()}
    output_shapes = {item.name: _shape(item.shape, batch) for item in session.get_outputs()}
    state_names = [name for name in input_shapes if name != "feature"]
    state_outputs = {
        output.name: output.name.removeprefix("next_")
        for output in session.get_outputs()
        if output.name.startswith("next_")
    }
    feature_host = np.zeros(input_shapes["feature"], dtype=np.float32)
    feature_value = ort.OrtValue.ortvalue_from_numpy(feature_host, "cuda", 0)
    state_values = [
        {
            name: ort.OrtValue.ortvalue_from_numpy(
                np.zeros(input_shapes[name], dtype=np.float32), "cuda", 0
            )
            for name in state_names
        }
        for _ in range(2)
    ]
    logit_host = [np.zeros(output_shapes["speech_logits"], dtype=np.float32) for _ in range(2)]
    logit_values = [ort.OrtValue.ortvalue_from_numpy(value) for value in logit_host]
    bindings = []
    for active in (0, 1):
        binding = session.io_binding()
        binding.bind_ortvalue_input("feature", feature_value)
        for name in state_names:
            binding.bind_ortvalue_input(name, state_values[active][name])
        binding.bind_ortvalue_output("speech_logits", logit_values[active])
        for output_name, state_name in state_outputs.items():
            binding.bind_ortvalue_output(output_name, state_values[1 - active][state_name])
        bindings.append(binding)

    active = 0

    def step() -> None:
        nonlocal active
        feature_value.update_inplace(feature_host)
        session.run_with_iobinding(bindings[active])
        active = 1 - active

    for _ in range(warmup):
        step()
    samples_us: list[float] = []
    trace = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        step()
        samples_us.append((time.perf_counter_ns() - started) / 1_000)
        trace.append(logit_host[1 - active].copy())
    return (
        {
            "path": "cuda_iobinding_state_on_device_logit_on_cpu",
            "requested_provider": "CUDAExecutionProvider",
            "session_providers": session.get_providers(),
            **_summary(samples_us, batch),
        },
        np.stack(trace),
    )


def _verify_parity(
    result: dict[str, Any],
    candidate: np.ndarray,
    reference: np.ndarray,
) -> None:
    if not np.isfinite(candidate).all():
        raise RuntimeError(f"non-finite output from {result['path']}")
    difference = float(np.max(np.abs(candidate - reference)))
    if not np.allclose(candidate, reference, rtol=1e-4, atol=1e-5):
        raise RuntimeError(
            f"{result['path']} output differs from CPU by {difference:.8g}"
        )
    result["matches_cpu"] = True
    result["max_abs_diff_vs_cpu"] = difference
    result["logit_trace_sha256"] = hashlib.sha256(
        np.asarray(candidate, dtype="<f4").tobytes()
    ).hexdigest()
    result["final_logits_sha256"] = hashlib.sha256(
        np.asarray(candidate[-1], dtype="<f4").tobytes()
    ).hexdigest()


def _download_model(commit: str, destination: Path) -> None:
    url = (
        "https://raw.githubusercontent.com/oss-codes/flashvad/"
        f"{commit}/models/flashvad-v0.1/flashvad-stream.onnx"
    )
    urllib.request.urlretrieve(url, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", default=DEFAULT_COMMIT)
    parser.add_argument("--model-sha256", default=DEFAULT_MODEL_SHA256)
    parser.add_argument("--ort-gpu", default=DEFAULT_ORT_GPU)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128])
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0 or any(batch < 1 for batch in args.batches):
        parser.error("iterations and batches must be positive; warmup must be non-negative")

    if not args.skip_install:
        _install_onnxruntime_gpu(args.ort_gpu)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA runtime is required; select a Colab GPU runtime")
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            "CUDAExecutionProvider is unavailable; check ONNX Runtime, CUDA, and cuDNN versions"
        )

    with tempfile.TemporaryDirectory(prefix="flashvad-colab-") as directory:
        model = Path(directory) / "flashvad-stream.onnx"
        _download_model(args.commit, model)
        digest = hashlib.sha256(model.read_bytes()).hexdigest()
        if digest != args.model_sha256:
            raise RuntimeError(f"model digest mismatch: expected {args.model_sha256}, got {digest}")

        results = []
        for batch in args.batches:
            cpu_result, cpu_logits = _numpy_benchmark(
                ort,
                model,
                "CPUExecutionProvider",
                batch,
                args.iterations,
                args.warmup,
            )
            _verify_parity(cpu_result, cpu_logits, cpu_logits)
            results.append(cpu_result)
            cuda_result, cuda_logits = _numpy_benchmark(
                ort,
                model,
                "CUDAExecutionProvider",
                batch,
                args.iterations,
                args.warmup,
            )
            _verify_parity(cuda_result, cuda_logits, cpu_logits)
            results.append(cuda_result)
            binding_result, binding_logits = _cuda_iobinding_benchmark(
                ort,
                model,
                batch,
                args.iterations,
                args.warmup,
            )
            _verify_parity(binding_result, binding_logits, cpu_logits)
            results.append(binding_result)

        report = {
            "schema": "flashvad-colab-provider-benchmark-v1",
            "artifact": {"commit": args.commit, "sha256": digest, "bytes": model.stat().st_size},
            "hardware": {
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
            },
            "runtime": {
                "onnxruntime": ort.__version__,
                "providers": ort.get_available_providers(),
                "iterations": args.iterations,
                "warmup": args.warmup,
                "scope": "ONNX model step only; frontend and scheduling excluded",
            },
            "results": results,
        }
        print("FLASHVAD_COLAB_BENCHMARK=" + json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
