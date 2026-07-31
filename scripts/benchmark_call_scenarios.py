#!/usr/bin/env python3
"""Deterministic concurrent-call latency benchmark for codec ingress scenarios."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from flashvad.runtime import OnnxStreamingVadModel
from flashvad.telephony import TelephonyVadStream

_PAYLOAD_DESCRIPTIONS = {
    "pcmu": "80-byte 8 kHz PCMU silence (0xFF)",
    "pcma": "80-byte 8 kHz PCMA silence (0xD5)",
    "pcm16": "160-byte 8 kHz little-endian PCM16 silence",
    "float32-16k": "160-sample 16 kHz float32 silence",
}


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _scenario_stream(owner: OnnxStreamingVadModel, scenario: str) -> tuple[Any, Any]:
    if scenario == "pcmu":
        return TelephonyVadStream(owner.new_stream(), scenario), bytes([0xFF] * 80)
    if scenario == "pcma":
        return TelephonyVadStream(owner.new_stream(), scenario), bytes([0xD5] * 80)
    if scenario == "pcm16":
        payload = np.zeros(80, dtype="<i2").tobytes()
        return TelephonyVadStream(owner.new_stream(), scenario), payload
    if scenario == "float32-16k":
        return owner.new_stream(), np.zeros(160, dtype=np.float32)
    raise ValueError("scenario must be pcmu, pcma, pcm16, or float32-16k")


def benchmark_scenario(
    owner: OnnxStreamingVadModel,
    scenario: str,
    calls: int,
    hops: int,
    warmup_hops: int = 0,
) -> dict[str, float | int | str]:
    streams = [_scenario_stream(owner, scenario) for _ in range(calls)]
    for _ in range(warmup_hops):
        for stream, payload in streams:
            stream.push(payload)
    queue_delays: list[float] = []
    end_to_end: list[float] = []
    # Each round is one deterministic scheduler tick: all calls become ready together.
    for _ in range(hops):
        arrival = time.perf_counter_ns()
        for stream, payload in streams:
            started = time.perf_counter_ns()
            stream.push(payload)
            finished = time.perf_counter_ns()
            queue_delays.append((started - arrival) / 1_000.0)
            end_to_end.append((finished - arrival) / 1_000.0)
    return {
        "scenario": scenario,
        "payload": _PAYLOAD_DESCRIPTIONS[scenario],
        "calls": calls,
        "hops": hops,
        "warmup_hops": warmup_hops,
        "queue_delay_p50_us": statistics.median(queue_delays),
        "queue_delay_p95_us": percentile(queue_delays, 95),
        "queue_delay_p99_us": percentile(queue_delays, 99),
        "end_to_end_p50_us": statistics.median(end_to_end),
        "end_to_end_p95_us": percentile(end_to_end, 95),
        "end_to_end_p99_us": percentile(end_to_end, 99),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path)
    parser.add_argument("--calls", type=int, default=8)
    parser.add_argument("--hops", type=int, default=100)
    parser.add_argument("--warmup-hops", type=int, default=25)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument(
        "--scenario",
        choices=("pcmu", "pcma", "pcm16", "float32-16k"),
        action="append",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.calls < 1 or args.hops < 1 or args.warmup_hops < 0:
        parser.error("calls and hops must be positive; warmup hops must be non-negative")
    owner = (
        OnnxStreamingVadModel(args.onnx, providers=[args.provider])
        if args.onnx
        else OnnxStreamingVadModel.load_bundled(providers=[args.provider])
    )
    scenarios = args.scenario or ["pcmu", "pcma", "pcm16", "float32-16k"]
    model_digest = hashlib.sha256(owner.path.read_bytes()).hexdigest()
    report = {
        "schema": "flashvad-call-scenarios-benchmark-v1",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "artifact": {
            "path": str(owner.path),
            "sha256": model_digest,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "runtime": {
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "providers": owner.session.get_providers(),
        },
        "protocol": {
            "clock": "perf_counter_ns",
            "scheduler": "deterministic round-robin",
            "hop_ms": 10,
            "scope": "codec ingress, causal frontend, ONNX model, and detector",
            "transport_excluded": "RTP jitter, packet loss, and network I/O",
        },
        "results": [
            benchmark_scenario(
                owner,
                scenario,
                args.calls,
                args.hops,
                args.warmup_hops,
            )
            for scenario in scenarios
        ],
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
