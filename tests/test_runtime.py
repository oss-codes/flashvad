import time

import numpy as np
import pytest

from flashvad.config import DetectorConfig


def test_onnx_stream_api_is_available_with_export_dependencies() -> None:
    runtime = pytest.importorskip("flashvad.runtime")
    # Export validation is covered by the integration run. Keep the unit test
    # independent of a generated model artifact.
    assert DetectorConfig().start_threshold > DetectorConfig().stop_threshold
    assert runtime.OnnxStreamingVadModel.new_stream


def test_real_onnx_per_call_state_allocation_is_sub_millisecond() -> None:
    runtime = pytest.importorskip("flashvad.runtime")
    owner = runtime.OnnxStreamingVadModel.load_bundled()
    timings = []
    for _ in range(1_000):
        started = time.perf_counter_ns()
        owner.new_stream()
        timings.append((time.perf_counter_ns() - started) / 1_000)
    assert float(np.percentile(timings, 99)) < 1_000


def test_bundled_onnx_model_runs_without_a_repository_model_path() -> None:
    runtime = pytest.importorskip("flashvad.runtime")

    owner = runtime.OnnxStreamingVadModel.load_bundled()
    probabilities, events = owner.new_stream().push(
        np.zeros(160, dtype=np.float32)
    )

    assert probabilities.shape == (1,)
    assert 0.0 <= float(probabilities[0]) <= 1.0
    assert isinstance(events, list)
    assert runtime.OnnxStreamingVadModel.load_bundled() is owner
