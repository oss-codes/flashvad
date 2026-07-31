from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_script() -> object:
    path = Path(__file__).parents[1] / "scripts" / "benchmark_colab_onnx.py"
    spec = importlib.util.spec_from_file_location("flashvad_benchmark_colab_onnx", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


def test_summary_labels_amortized_throughput_separately() -> None:
    summary = SCRIPT._summary([80.0, 100.0, 120.0], batch=4)

    assert summary["median_call_us"] == 100.0
    assert summary["amortized_median_us_per_stream"] == 25.0
    assert "median_per_stream_us" not in summary


def test_parity_records_digest_and_rejects_invalid_outputs() -> None:
    reference = np.asarray([[0.25], [0.75]], dtype=np.float32)
    result = {"path": "test"}

    SCRIPT._verify_parity(result, reference.copy(), reference)

    assert result["matches_cpu"] is True
    assert result["max_abs_diff_vs_cpu"] == 0.0
    assert len(result["final_logits_sha256"]) == 64

    with pytest.raises(RuntimeError, match="non-finite"):
        SCRIPT._verify_parity(
            {"path": "nan"},
            np.asarray([[np.nan]], dtype=np.float32),
            np.asarray([[0.0]], dtype=np.float32),
        )
    with pytest.raises(RuntimeError, match="differs from CPU"):
        SCRIPT._verify_parity(
            {"path": "wrong"},
            np.asarray([[1.0]], dtype=np.float32),
            np.asarray([[0.0]], dtype=np.float32),
        )


def test_provider_check_rejects_fallback_and_disables_future_fallback() -> None:
    class Session:
        def __init__(self, providers: list[str]) -> None:
            self.providers = providers
            self.fallback_disabled = False

        def get_providers(self) -> list[str]:
            return self.providers

        def disable_fallback(self) -> None:
            self.fallback_disabled = True

    cuda = Session(["CUDAExecutionProvider", "CPUExecutionProvider"])
    SCRIPT._require_provider(cuda, "CUDAExecutionProvider")
    assert cuda.fallback_disabled is True

    with pytest.raises(RuntimeError, match="refusing fallback"):
        SCRIPT._require_provider(Session(["CPUExecutionProvider"]), "CUDAExecutionProvider")
