#!/usr/bin/env python3
"""Benchmark G.711 decode plus causal 8 kHz to 16 kHz conversion."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from flashvad.telephony import (
    CausalLinearResampler8To16,
    decode_g711_alaw,
    decode_g711_ulaw,
)


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values), percentile))


def benchmark_codec(
    codec: str,
    *,
    packet_samples: int,
    iterations: int,
    warmup: int,
) -> dict[str, float | int | str]:
    decoder = decode_g711_ulaw if codec == "pcmu" else decode_g711_alaw
    payload = bytes((index * 37 + 11) % 256 for index in range(packet_samples))
    resampler = CausalLinearResampler8To16()

    for _ in range(warmup):
        resampler.push(decoder(payload))
    timings: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        resampler.push(decoder(payload))
        timings.append((time.perf_counter_ns() - started) / 1_000)

    packet_ms = packet_samples / 8_000 * 1_000
    median = float(statistics.median(timings))
    return {
        "codec": codec,
        "packet_samples": packet_samples,
        "packet_ms": packet_ms,
        "median_us": median,
        "p95_us": _percentile(timings, 95),
        "p99_us": _percentile(timings, 99),
        "realtime_factor": median / (packet_ms * 1_000),
        "iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-ms", type=int, default=20, choices=(10, 20, 30))
    parser.add_argument("--iterations", type=int, default=20_000)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup must be non-negative")

    packet_samples = args.packet_ms * 8
    report = {
        "runtime": "python-numpy-telephone-ingress",
        "input_sample_rate": 8_000,
        "output_sample_rate": 16_000,
        "results": [
            benchmark_codec(
                codec,
                packet_samples=packet_samples,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            for codec in ("pcmu", "pcma")
        ],
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
