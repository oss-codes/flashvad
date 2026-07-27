#!/usr/bin/env python3
"""Measure the actual per-call Python stream wrappers used by integrations."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from flashvad.native_runtime import MacosNativeVadModel
from flashvad.runtime import OnnxStreamingVadModel


def percentile(values: list[float], quantile: float) -> float:
    return sorted(values)[round((len(values) - 1) * quantile)]


def benchmark(stream: Any, iterations: int, warmup: int) -> dict[str, float | int]:
    hop = np.zeros(160, dtype=np.float32)
    for _ in range(warmup):
        stream.push(hop)
    timings: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        stream.push(hop)
        timings.append((time.perf_counter_ns() - started) / 1_000)
    median = statistics.median(timings)
    return {
        "median_us": median,
        "p95_us": percentile(timings, 0.95),
        "p99_us": percentile(timings, 0.99),
        "realtime_factor": median / 10_000,
        "iterations": iterations,
        "warmup": warmup,
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--iterations", type=int, default=5_000)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("iterations must be positive and warmup must be non-negative")

    platform_info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        platform_info["chip"] = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    results: dict[str, object] = {
        "schema": "flashvad-call-runtime-benchmark-v1",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "protocol": {
            "clock": "perf_counter_ns",
            "audio": "one all-zero 10 ms float32 hop per push",
            "scope": "Python call-stream wrapper including frontend, model, and detector",
            "threads": 1,
        },
        "platform": platform_info,
        "onnx": {
            "artifact": args.onnx.name,
            "artifact_sha256": digest(args.onnx),
            **benchmark(
                OnnxStreamingVadModel(args.onnx, threads=1).new_stream(),
                args.iterations,
                args.warmup,
            ),
        },
    }
    if args.native_library is not None:
        results["native"] = {
            "artifact": args.native_library.name,
            "artifact_sha256": digest(args.native_library),
            **benchmark(
                MacosNativeVadModel(args.native_library).new_stream(),
                args.iterations,
                args.warmup,
            ),
        }

    serialized = json.dumps(results, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")


if __name__ == "__main__":
    main()
