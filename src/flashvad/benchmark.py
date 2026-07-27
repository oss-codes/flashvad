from __future__ import annotations

import hashlib
import json
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .config import ProjectConfig


def _platform_metadata() -> dict[str, str]:
    metadata = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        metadata["macos"] = platform.mac_ver()[0]
        completed = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            text=True,
            capture_output=True,
        )
        metadata["chip"] = completed.stdout.strip()
    return metadata


def _artifact_metadata(path: Path) -> dict[str, object]:
    return {
        "artifact": path.name,
        "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "platform": _platform_metadata(),
    }


def load_checkpoint(path: str | Path) -> tuple[ProjectConfig, Any]:
    from .checkpoint import load_checkpoint_data
    from .model import FlashVad

    checkpoint = load_checkpoint_data(path)
    config = ProjectConfig.from_dict(checkpoint["config"])
    model = FlashVad(config.model)
    model.load_state_dict(checkpoint["model"])
    return config, model.eval()


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * percent))
    return ordered[index]


def benchmark_checkpoint(
    checkpoint_path: str | Path,
    iterations: int = 5_000,
    warmup: int = 500,
    threads: int = 1,
) -> dict[str, object]:
    import torch

    from .features import CausalFeatureExtractor, StreamingFeatureExtractor

    torch.set_num_threads(threads)
    with torch.inference_mode():
        config, model = load_checkpoint(checkpoint_path)
        frontend = CausalFeatureExtractor(config.feature)
        feature = torch.zeros(1, config.model.feature_dim)
        state = model.initial_state(1, "cpu")

        for _ in range(warmup):
            _, state = model.stream_step(feature, state)
        state = model.initial_state(1, "cpu")
        timings: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            _, state = model.stream_step(feature, state)
            timings.append((time.perf_counter_ns() - started) / 1_000)

        hop_audio = torch.zeros(config.feature.hop_samples)
        frame_timings: list[float] = []
        stream = StreamingFeatureExtractor(frontend)
        for _ in range(warmup):
            stream.push(hop_audio)
        stream.reset()
        for _ in range(iterations):
            started = time.perf_counter_ns()
            stream.push(hop_audio)
            frame_timings.append((time.perf_counter_ns() - started) / 1_000)

    parameter_bytes_fp32 = model.parameter_count * 4
    result: dict[str, float | int] = {
        "parameters": model.parameter_count,
        "fp32_parameter_bytes": parameter_bytes_fp32,
        "estimated_int8_parameter_bytes": model.parameter_count,
        "model_step_median_us": statistics.median(timings),
        "model_step_p95_us": percentile(timings, 0.95),
        "model_step_p99_us": percentile(timings, 0.99),
        "frontend_median_us": statistics.median(frame_timings),
        "frontend_p95_us": percentile(frame_timings, 0.95),
        "frontend_p99_us": percentile(frame_timings, 0.99),
        "iterations": iterations,
        "warmup": warmup,
        "threads": threads,
    }
    result["combined_median_us"] = float(result["model_step_median_us"]) + float(
        result["frontend_median_us"]
    )
    report = {
        "schema": "flashvad-pytorch-benchmark-v1",
        **_artifact_metadata(Path(checkpoint_path)),
        "runtime": {
            "torch": torch.__version__,
            "audio": "all-zero 10 ms hops",
        },
        **result,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def benchmark_onnx(
    model_path: str | Path,
    iterations: int = 5_000,
    warmup: int = 500,
    threads: int = 1,
) -> dict[str, object]:
    import numpy as np
    import onnxruntime as ort

    path = Path(model_path)
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    inputs: dict[str, np.ndarray] = {}
    for item in session.get_inputs():
        shape = [1 if isinstance(value, str) else value for value in item.shape]
        inputs[item.name] = np.zeros(shape, dtype=np.float32)

    state_names = {
        output.name: output.name.removeprefix("next_")
        for output in session.get_outputs()
        if output.name.startswith("next_")
    }

    def update_state(outputs: list[np.ndarray]) -> None:
        for metadata, value in zip(session.get_outputs(), outputs, strict=True):
            destination = state_names.get(metadata.name)
            if destination:
                inputs[destination] = value

    for _ in range(warmup):
        update_state(session.run(None, inputs))
    timings: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        outputs = session.run(None, inputs)
        timings.append((time.perf_counter_ns() - started) / 1_000)
        update_state(outputs)
    result: dict[str, float | int | object] = {
        "schema": "flashvad-onnx-benchmark-v1",
        **_artifact_metadata(path),
        "runtime": {
            "onnxruntime": ort.__version__,
            "provider": "CPUExecutionProvider",
            "audio": "precomputed all-zero feature frames; frontend excluded",
        },
        "onnx_bytes": path.stat().st_size,
        "step_median_us": statistics.median(timings),
        "step_p95_us": percentile(timings, 0.95),
        "step_p99_us": percentile(timings, 0.99),
        "iterations": iterations,
        "warmup": warmup,
        "threads": threads,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result
