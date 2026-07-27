from pathlib import Path

import numpy as np
import pytest

from flashvad.config import DetectorConfig
from flashvad.evaluation import (
    EvaluationRecord,
    _select_detector_result,
    bootstrap_confidence_intervals,
    boundary_metrics,
    calibrate_detector,
    detector_decisions,
    evaluation_report,
)
from flashvad.manifest import ManifestItem
from flashvad.metrics import average_precision, roc_auc


def _record(probabilities: list[float], labels: list[int]) -> EvaluationRecord:
    return EvaluationRecord(
        ManifestItem(Path("unused.wav"), 16_000, "en", "test", ()),
        np.asarray(probabilities, dtype=np.float64),
        np.asarray(labels, dtype=np.int8),
    )


def _bootstrap_records() -> list[EvaluationRecord]:
    return [
        _record([0.95, 0.10], [1, 0]),
        _record([0.80, 0.20, 0.70], [1, 0, 1]),
        _record([0.40, 0.60, 0.30, 0.90], [0, 1, 0, 1]),
    ]


def test_item_bootstrap_is_deterministic_and_reports_fixed_threshold_metrics() -> None:
    records = _bootstrap_records()
    first = bootstrap_confidence_intervals(records, threshold=0.5, iterations=200, seed=17)
    second = bootstrap_confidence_intervals(records, threshold=0.5, iterations=200, seed=17)

    assert first == second
    assert set(first) == {
        "roc_auc",
        "pr_auc",
        "f1",
        "false_alarm_rate",
        "miss_rate",
    }
    for interval in first.values():
        assert set(interval) == {"lower", "upper"}
        assert interval["lower"] <= interval["upper"]


def test_evaluation_report_includes_item_bootstrap_metadata_and_can_skip_it() -> None:
    records = _bootstrap_records()
    without_bootstrap = evaluation_report(records, threshold=0.5, hop_ms=10.0)
    assert "confidence_intervals" not in without_bootstrap

    report = evaluation_report(
        records,
        threshold=0.5,
        hop_ms=10.0,
        bootstrap_iterations=25,
        bootstrap_seed=23,
    )
    confidence_intervals = report["confidence_intervals"]
    assert confidence_intervals["method"] == "item_bootstrap"
    assert confidence_intervals["confidence_level"] == 0.95
    assert confidence_intervals["iterations"] == 25
    assert confidence_intervals["seed"] == 23
    assert confidence_intervals["threshold"] == 0.5
    assert confidence_intervals["threshold_policy"] == "fixed_supplied_threshold"
    assert set(confidence_intervals["metrics"]) == {
        "roc_auc",
        "pr_auc",
        "f1",
        "false_alarm_rate",
        "miss_rate",
    }


def test_bootstrap_report_covers_fixed_production_detector() -> None:
    detector = DetectorConfig(0.5, 0.3, 1, 1, 0)
    report = evaluation_report(
        _bootstrap_records(),
        threshold=0.5,
        hop_ms=10.0,
        detector_config=detector,
        bootstrap_iterations=25,
        bootstrap_seed=5,
    )

    intervals = report["confidence_intervals"]["production_detector_metrics"]
    assert set(intervals) == {
        "roc_auc",
        "pr_auc",
        "f1",
        "false_alarm_rate",
        "miss_rate",
    }


@pytest.mark.parametrize("iterations", [-1, 1.5, True])
def test_item_bootstrap_rejects_invalid_iteration_counts(iterations: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        bootstrap_confidence_intervals(_bootstrap_records(), 0.5, iterations, 0)


def test_auc_metrics_are_exact_for_separable_predictions() -> None:
    probabilities = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([0, 0, 1, 1])
    assert roc_auc(probabilities, labels) == 1.0
    assert average_precision(probabilities, labels) == 1.0


def test_roc_auc_handles_tied_scores() -> None:
    probabilities = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 1, 0, 0])
    assert roc_auc(probabilities, labels) == 0.5
    assert average_precision(probabilities, labels) == 0.5


def test_boundary_metrics_report_signed_errors() -> None:
    labels = np.array([0, 0, 1, 1, 1, 0, 0, 1, 1, 0])
    probabilities = np.array([0, 0, 0, 1, 1, 0, 0, 1, 0, 0], dtype=float)
    metrics = boundary_metrics(probabilities, labels, threshold=0.5, hop_ms=10)
    assert metrics["matched_segments"] == 2
    assert metrics["onset_error_p50_ms"] == 5.0
    assert metrics["offset_error_p50_ms"] == -5.0


def test_detector_decisions_match_streaming_hysteresis() -> None:
    probabilities = np.array([0.1, 0.7, 0.8, 0.5, 0.2, 0.2, 0.2, 0.8, 0.1])
    config = DetectorConfig(
        start_threshold=0.6,
        stop_threshold=0.3,
        start_frames=2,
        stop_frames=3,
        pre_roll_frames=1,
    )

    decisions = detector_decisions(probabilities, config, hop_ms=10)

    np.testing.assert_array_equal(decisions, [1, 1, 1, 1, 0, 0, 0, 0, 0])


def test_detector_calibration_respects_false_alarm_ceiling() -> None:
    probabilities = [np.array([0.6, 0.9, 0.9, 0.6, 0.6])]
    labels = [np.array([0, 1, 1, 1, 1])]
    permissive = DetectorConfig(0.5, 0.4, 1, 1, 0)
    conservative = DetectorConfig(0.8, 0.7, 1, 1, 0)

    selected, metrics = calibrate_detector(
        probabilities,
        labels,
        10,
        max_false_alarm_rate=0.2,
        candidates=(permissive, conservative),
    )

    assert selected == conservative
    assert metrics["false_alarm_rate"] == 0.0


def test_detector_calibration_prefers_fewer_triggers_within_f1_tolerance() -> None:
    fragmented = DetectorConfig(0.7, 0.6, 1, 8, 3)
    stable = DetectorConfig(0.6, 0.5, 3, 8, 2)
    selected, metrics = _select_detector_result(
        [
            (
                fragmented,
                {
                    "f1": 0.9141,
                    "false_alarm_rate": 0.045,
                    "miss_rate": 0.097,
                    "false_triggers": 129,
                },
            ),
            (
                stable,
                {
                    "f1": 0.9138,
                    "false_alarm_rate": 0.055,
                    "miss_rate": 0.085,
                    "false_triggers": 67,
                },
            ),
        ],
        max_false_alarm_rate=0.15,
    )

    assert selected == stable
    assert metrics["false_triggers"] == 67


def test_detector_calibration_can_choose_zero_preroll_and_short_release() -> None:
    probabilities = [np.array([0.0, 0.0, 0.95, 0.95, 0.0, 0.0])]
    labels = [np.array([0, 0, 1, 1, 0, 0])]

    config, metrics = calibrate_detector(
        probabilities,
        labels,
        hop_ms=10.0,
        max_false_alarm_rate=0.0,
    )

    assert config.pre_roll_frames == 0
    assert config.stop_frames == 2
    assert metrics["f1"] == 1.0
    assert metrics["false_alarm_rate"] == 0.0


def test_default_detector_calibration_grid_is_valid() -> None:
    selected, _ = calibrate_detector(
        [np.array([0.1, 0.9, 0.9, 0.1])],
        [np.array([0, 1, 1, 0])],
        10,
    )

    selected.validate()
