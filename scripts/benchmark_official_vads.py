#!/usr/bin/env python3
"""Same-machine latency harness for external VAD reference implementations.

External code and weights are deliberately not vendored. Pass paths to official
checkouts/artifacts so every result can record the exact upstream revision.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def percentile(values: list[float], quantile: float) -> float:
    return sorted(values)[round((len(values) - 1) * quantile)]


def latency_report(
    timings: list[float],
    *,
    hop_ms: float,
    creation_us: float,
    artifact: Path,
    source_revision: str,
    scope: str,
) -> dict[str, float | int | str]:
    median = statistics.median(timings)
    return {
        "artifact": artifact.name,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "source_revision": source_revision,
        "artifact_bytes": artifact.stat().st_size,
        "scope": scope,
        "hop_ms": hop_ms,
        "creation_us": creation_us,
        "median_us": median,
        "p95_us": percentile(timings, 0.95),
        "p99_us": percentile(timings, 0.99),
        "realtime_factor": median / (hop_ms * 1_000),
    }


def git_revision(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if len(revision) != 40:
        raise ValueError(f"unexpected git revision from {repository}")
    return revision


def benchmark_ten(
    repository: Path,
    library_path: Path,
    iterations: int,
    warmup: int,
    hop_samples: int,
) -> dict[str, float | int | str]:
    library = ctypes.CDLL(str(library_path))
    library.ten_vad_create.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_size_t,
        ctypes.c_float,
    ]
    library.ten_vad_create.restype = ctypes.c_int
    library.ten_vad_process.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int),
    ]
    library.ten_vad_process.restype = ctypes.c_int
    library.ten_vad_destroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    handle = ctypes.c_void_p()
    started = time.perf_counter_ns()
    result = library.ten_vad_create(
        ctypes.byref(handle),
        hop_samples,
        ctypes.c_float(0.5),
    )
    creation_us = (time.perf_counter_ns() - started) / 1_000
    if result != 0:
        raise RuntimeError("TEN VAD initialization failed")
    frame = np.zeros(hop_samples, dtype=np.int16)
    frame_pointer = frame.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
    probability = ctypes.c_float()
    flag = ctypes.c_int()
    try:
        for _ in range(warmup):
            result = library.ten_vad_process(
                handle,
                frame_pointer,
                hop_samples,
                ctypes.byref(probability),
                ctypes.byref(flag),
            )
            if result != 0:
                raise RuntimeError(
                    f"TEN VAD warmup inference failed with return code {result}"
                )
        timings = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            result = library.ten_vad_process(
                handle,
                frame_pointer,
                hop_samples,
                ctypes.byref(probability),
                ctypes.byref(flag),
            )
            elapsed_us = (time.perf_counter_ns() - started) / 1_000
            if result != 0:
                raise RuntimeError(
                    f"TEN VAD measured inference failed with return code {result}"
                )
            timings.append(elapsed_us)
    finally:
        library.ten_vad_destroy(ctypes.byref(handle))
    return latency_report(
        timings,
        hop_ms=hop_samples / 16,
        creation_us=creation_us,
        artifact=library_path,
        source_revision=git_revision(repository),
        scope="official native frontend, recurrent model, and decision",
    )


def benchmark_silero(
    repository: Path,
    model_path: Path,
    iterations: int,
    warmup: int,
) -> dict[str, float | int | str]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    started = time.perf_counter_ns()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    creation_us = (time.perf_counter_ns() - started) / 1_000
    inputs = {
        "input": np.zeros((1, 576), dtype=np.float32),
        "state": np.zeros((2, 1, 128), dtype=np.float32),
        "sr": np.array(16_000, dtype=np.int64),
    }
    for _ in range(warmup):
        outputs = session.run(None, inputs)
        inputs["state"] = outputs[1]
    timings = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        outputs = session.run(None, inputs)
        timings.append((time.perf_counter_ns() - started) / 1_000)
        inputs["state"] = outputs[1]
    return latency_report(
        timings,
        hop_ms=32,
        creation_us=creation_us,
        artifact=model_path,
        source_revision=git_revision(repository),
        scope="official ONNX model step with recurrent state; postprocessing excluded",
    )


def benchmark_firered(
    repository: Path,
    model_path: Path,
    iterations: int,
    warmup: int,
) -> dict[str, float | int | str]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    started = time.perf_counter_ns()
    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    creation_us = (time.perf_counter_ns() - started) / 1_000
    inputs = {
        "feat": np.zeros((1, 1, 80), dtype=np.float32),
        "caches_in": np.zeros((8, 1, 128, 19), dtype=np.float32),
    }
    for _ in range(warmup):
        outputs = session.run(None, inputs)
        inputs["caches_in"] = outputs[1]
    timings = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        outputs = session.run(None, inputs)
        timings.append((time.perf_counter_ns() - started) / 1_000)
        inputs["caches_in"] = outputs[1]
    return latency_report(
        timings,
        hop_ms=10,
        creation_us=creation_us,
        artifact=model_path,
        source_revision=git_revision(repository),
        scope="official streaming ONNX model step with cache; frontend and postprocessing excluded",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-benchmark", type=Path)
    parser.add_argument("--native-metadata", type=Path)
    parser.add_argument("--ten-repository", type=Path)
    parser.add_argument("--ten-library", type=Path)
    parser.add_argument("--ten-hop", type=int, default=160)
    parser.add_argument("--silero-repository", type=Path)
    parser.add_argument("--silero-model", type=Path)
    parser.add_argument("--firered-repository", type=Path)
    parser.add_argument("--firered-model", type=Path)
    parser.add_argument("--iterations", type=int, default=5_000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.native_metadata and not args.native_benchmark:
        parser.error("--native-metadata requires --native-benchmark")

    results: dict[str, object] = {
        "schema": "flashvad-external-runtime-benchmark-v1",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "clock": "perf_counter_ns",
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "warning": (
            "Different model hops and runtime stacks make this a development "
            "snapshot, not an accuracy or product ranking."
        ),
        "comparison_basis": (
            "single-thread warm-state compute normalized by audio hop duration; "
            "all-zero input; no endpointing-latency or accuracy inference"
        ),
    }
    if args.native_benchmark:
        completed = subprocess.run(
            [str(args.native_benchmark)],
            check=True,
            capture_output=True,
            text=True,
        )
        native = json.loads(completed.stdout)
        native.update(
            {
                "artifact": args.native_benchmark.name,
                "artifact_sha256": hashlib.sha256(
                    args.native_benchmark.read_bytes()
                ).hexdigest(),
                "artifact_bytes": args.native_benchmark.stat().st_size,
                "scope": "embedded native frontend and recurrent model; detector excluded",
                "license": "MIT code; CC-BY-4.0 retained model weights",
            }
        )
        if args.native_metadata:
            native["build_metadata"] = json.loads(
                args.native_metadata.read_text(encoding="utf-8")
            )
        results["flashvad_native"] = native
    if bool(args.ten_repository) != bool(args.ten_library):
        parser.error("TEN requires both --ten-repository and --ten-library")
    if args.ten_library and args.ten_repository:
        ten = benchmark_ten(
            args.ten_repository,
            args.ten_library,
            args.iterations,
            args.warmup,
            args.ten_hop,
        )
        ten.update(
            {
                "repository": "https://github.com/TEN-framework/ten-vad",
                "license": "Apache-2.0 with additional competitive-use conditions",
            }
        )
        results["ten"] = ten
    if bool(args.silero_repository) != bool(args.silero_model):
        parser.error("Silero requires both --silero-repository and --silero-model")
    if args.silero_model and args.silero_repository:
        silero = benchmark_silero(
            args.silero_repository,
            args.silero_model,
            args.iterations,
            args.warmup,
        )
        silero.update(
            {
                "repository": "https://github.com/snakers4/silero-vad",
                "license": "MIT",
            }
        )
        results["silero"] = silero
    if bool(args.firered_repository) != bool(args.firered_model):
        parser.error("FireRed requires both --firered-repository and --firered-model")
    if args.firered_repository and args.firered_model:
        firered = benchmark_firered(
            args.firered_repository,
            args.firered_model,
            args.iterations,
            args.warmup,
        )
        firered.update(
            {
                "repository": "https://github.com/FireRedTeam/FireRedVAD",
                "license": "Apache-2.0",
            }
        )
        results["firered_streaming"] = firered

    native = results.get("flashvad_native")
    if isinstance(native, dict):
        native_rtf = float(native["realtime_factor"])
        for key in ("ten", "silero", "firered_streaming"):
            candidate = results.get(key)
            if isinstance(candidate, dict):
                candidate["flashvad_native_compute_advantage"] = (
                    float(candidate["realtime_factor"]) / native_rtf
                )

    serialized = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
