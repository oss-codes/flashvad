from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script() -> object:
    path = Path(__file__).parents[1] / "scripts" / "benchmark_call_scenarios.py"
    spec = importlib.util.spec_from_file_location("flashvad_benchmark_call_scenarios", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


class _Stream:
    def push(self, payload: object) -> None:
        assert payload is not None


class _Owner:
    def new_stream(self) -> _Stream:
        return _Stream()


def test_percentile_and_scenario_report_have_expected_shape() -> None:
    assert SCRIPT.percentile([1.0, 2.0, 3.0], 50) == 2.0
    result = SCRIPT.benchmark_scenario(_Owner(), "pcmu", calls=2, hops=3)
    assert result["calls"] == 2
    assert result["hops"] == 3
    assert result["scenario"] == "pcmu"
    assert result["queue_delay_p99_us"] >= 0
    assert result["end_to_end_p99_us"] >= result["queue_delay_p99_us"]


def test_scenarios_accept_all_supported_payload_shapes() -> None:
    owner = _Owner()
    for scenario in ("pcmu", "pcma", "pcm16", "float32-16k"):
        result = SCRIPT.benchmark_scenario(owner, scenario, calls=1, hops=1)
        assert result["scenario"] == scenario
