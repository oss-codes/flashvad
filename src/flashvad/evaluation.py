from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .audio import read_audio
from .benchmark import load_checkpoint
from .checkpoint import load_checkpoint_data
from .config import DetectorConfig
from .detector import HysteresisDetector
from .features import CausalFeatureExtractor
from .manifest import ManifestItem, frame_labels, load_manifest
from .metrics import average_precision, best_f1_threshold, binary_metrics, roc_auc


@dataclass(frozen=True)
class EvaluationRecord:
    item: ManifestItem
    probabilities: np.ndarray
    labels: np.ndarray


def _json_safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _binary_runs(values: np.ndarray) -> list[tuple[int, int]]:
    binary = np.asarray(values, dtype=bool).reshape(-1)
    padded = np.pad(binary.astype(np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def detector_decisions(
    probabilities: np.ndarray,
    config: DetectorConfig,
    hop_ms: float,
) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(scores)):
        raise ValueError("probabilities contain non-finite values")
    if hop_ms <= 0:
        raise ValueError("hop_ms must be positive")

    detector = HysteresisDetector(config, hop_ms / 1_000)
    decisions = np.zeros(scores.size, dtype=np.int8)
    active_start: int | None = None
    for score in scores:
        for event in detector.update(float(score)):
            if event.kind == "speech_start":
                active_start = event.frame
            elif active_start is not None:
                decisions[active_start : event.frame] = 1
                active_start = None
    if active_start is not None:
        decisions[active_start:] = 1
    return decisions


def _detector_segment_metrics(
    decision_sequences: Iterable[np.ndarray],
    label_sequences: Iterable[np.ndarray],
    hop_ms: float,
) -> dict[str, float | int]:
    predictions = [np.asarray(value).reshape(-1) for value in decision_sequences]
    labels = [np.asarray(value).reshape(-1) for value in label_sequences]
    reference_segments = 0
    predicted_segments = 0
    matched_segments = 0
    matched_short_segments = 0
    short_segments = 0
    for prediction, label in zip(predictions, labels, strict=True):
        predicted = _binary_runs(prediction)
        reference = _binary_runs(label == 1)
        candidates: list[tuple[float, int, int]] = []
        for reference_index, (reference_start, reference_end) in enumerate(reference):
            for predicted_index, (predicted_start, predicted_end) in enumerate(predicted):
                intersection = max(
                    0,
                    min(reference_end, predicted_end)
                    - max(reference_start, predicted_start),
                )
                if intersection == 0:
                    continue
                union = max(reference_end, predicted_end) - min(
                    reference_start,
                    predicted_start,
                )
                candidates.append(
                    (intersection / max(union, 1), reference_index, predicted_index)
                )
        matched_reference: set[int] = set()
        matched_prediction: set[int] = set()
        for _, reference_index, predicted_index in sorted(candidates, reverse=True):
            if (
                reference_index in matched_reference
                or predicted_index in matched_prediction
            ):
                continue
            matched_reference.add(reference_index)
            matched_prediction.add(predicted_index)

        short_reference = {
            index
            for index, (start, end) in enumerate(reference)
            if (end - start) * hop_ms <= 300.0
        }
        reference_segments += len(reference)
        predicted_segments += len(predicted)
        matched_segments += len(matched_reference)
        short_segments += len(short_reference)
        matched_short_segments += len(short_reference & matched_reference)

    return {
        "reference_segments": reference_segments,
        "predicted_segments": predicted_segments,
        "matched_segments": matched_segments,
        "segment_recall": matched_segments / max(reference_segments, 1),
        "short_utterance_recall": matched_short_segments / max(short_segments, 1),
        "false_triggers": predicted_segments - matched_segments,
    }


def _select_detector_result(
    results: list[tuple[DetectorConfig, dict[str, float | int]]],
    max_false_alarm_rate: float,
    *,
    f1_tolerance: float = 0.001,
) -> tuple[DetectorConfig, dict[str, float | int]]:
    eligible = [
        result
        for result in results
        if float(result[1]["false_alarm_rate"]) <= max_false_alarm_rate
    ]
    if eligible:
        best_f1 = max(float(result[1]["f1"]) for result in eligible)
        near_best = [
            result
            for result in eligible
            if float(result[1]["f1"]) >= best_f1 - f1_tolerance
        ]
        return min(
            near_best,
            key=lambda result: (
                int(result[1]["false_triggers"]),
                -float(result[1]["f1"]),
                float(result[1]["false_alarm_rate"]),
                float(result[1]["miss_rate"]),
            ),
        )
    return min(
        results,
        key=lambda result: (
            float(result[1]["false_alarm_rate"]),
            -float(result[1]["f1"]),
            float(result[1]["miss_rate"]),
        ),
    )


def calibrate_detector(
    probability_sequences: Iterable[np.ndarray],
    label_sequences: Iterable[np.ndarray],
    hop_ms: float,
    *,
    max_false_alarm_rate: float = 0.2,
    candidates: Iterable[DetectorConfig] | None = None,
) -> tuple[DetectorConfig, dict[str, float | int]]:
    probabilities = [np.asarray(value).reshape(-1) for value in probability_sequences]
    labels = [np.asarray(value).reshape(-1) for value in label_sequences]
    if len(probabilities) != len(labels) or not probabilities:
        raise ValueError("probability and label sequences must have equal non-zero length")
    if any(
        probability.size != label.size
        for probability, label in zip(probabilities, labels, strict=True)
    ):
        raise ValueError("each probability sequence must match its label sequence")
    if not 0.0 <= max_false_alarm_rate <= 1.0:
        raise ValueError("max_false_alarm_rate must be between zero and one")

    search_space = (
        list(candidates)
        if candidates is not None
        else [
            DetectorConfig(
                start_threshold=start_percent / 100,
                stop_threshold=stop_percent / 100,
                start_frames=start_frames,
                stop_frames=stop_frames,
                pre_roll_frames=pre_roll_frames,
            )
            for start_percent in range(30, 91, 10)
            for stop_percent in range(20, start_percent, 10)
            for start_frames in (1, 2, 3)
            for stop_frames in (2, 4, 8)
            for pre_roll_frames in (0, 2, 3)
        ]
    )
    if not search_space:
        raise ValueError("detector calibration requires at least one candidate")

    target = np.concatenate(labels)
    results: list[tuple[DetectorConfig, dict[str, float | int]]] = []
    for candidate in search_space:
        decisions = [
            detector_decisions(probability, candidate, hop_ms)
            for probability in probabilities
        ]
        metrics: dict[str, float | int] = {
            **binary_metrics(np.concatenate(decisions), target, 0.5),
            **_detector_segment_metrics(decisions, labels, hop_ms),
        }
        results.append((candidate, metrics))

    return _select_detector_result(
        results,
        max_false_alarm_rate,
    )


def boundary_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    hop_ms: float,
) -> dict[str, float | int]:
    predicted = _binary_runs(np.asarray(probabilities) >= threshold)
    reference = _binary_runs(np.asarray(labels) == 1)
    candidates: list[tuple[float, int, int]] = []
    for reference_index, (reference_start, reference_end) in enumerate(reference):
        for predicted_index, (predicted_start, predicted_end) in enumerate(predicted):
            intersection = max(
                0,
                min(reference_end, predicted_end) - max(reference_start, predicted_start),
            )
            if intersection == 0:
                continue
            union = max(reference_end, predicted_end) - min(reference_start, predicted_start)
            candidates.append((intersection / max(union, 1), reference_index, predicted_index))

    matched_reference: set[int] = set()
    matched_predicted: set[int] = set()
    onset_errors: list[float] = []
    offset_errors: list[float] = []
    for _, reference_index, predicted_index in sorted(candidates, reverse=True):
        if reference_index in matched_reference or predicted_index in matched_predicted:
            continue
        matched_reference.add(reference_index)
        matched_predicted.add(predicted_index)
        reference_start, reference_end = reference[reference_index]
        predicted_start, predicted_end = predicted[predicted_index]
        onset_errors.append((predicted_start - reference_start) * hop_ms)
        offset_errors.append((predicted_end - reference_end) * hop_ms)

    short_reference = {
        index
        for index, (start, end) in enumerate(reference)
        if (end - start) * hop_ms <= 300.0
    }
    negative_hours = float(np.sum(np.asarray(labels) == 0) * hop_ms / 3_600_000)
    false_triggers = len(predicted) - len(matched_predicted)

    def percentile(values: list[float], quantile: float) -> float:
        return float(np.percentile(values, quantile)) if values else float("nan")

    return {
        "reference_segments": len(reference),
        "predicted_segments": len(predicted),
        "matched_segments": len(matched_reference),
        "segment_recall": len(matched_reference) / max(len(reference), 1),
        "short_utterance_recall": (
            len(short_reference & matched_reference) / max(len(short_reference), 1)
        ),
        "onset_error_p50_ms": percentile(onset_errors, 50),
        "onset_error_p95_ms": percentile(onset_errors, 95),
        "offset_error_p50_ms": percentile(offset_errors, 50),
        "offset_error_p95_ms": percentile(offset_errors, 95),
        "premature_end_rate_100ms": (
            float(np.mean(np.asarray(offset_errors) < -100.0)) if offset_errors else float("nan")
        ),
        "false_triggers": false_triggers,
        "false_triggers_per_noise_hour": false_triggers / max(negative_hours, 1e-12),
    }


def _frame_report(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    return {
        "frames": int(labels.size),
        "speech_fraction": float(np.mean(labels)),
        "threshold": threshold,
        "roc_auc": roc_auc(probabilities, labels),
        "pr_auc": average_precision(probabilities, labels),
        **binary_metrics(probabilities, labels, threshold),
    }


def _combine(records: Iterable[EvaluationRecord]) -> tuple[np.ndarray, np.ndarray]:
    selected = list(records)
    return (
        np.concatenate([record.probabilities for record in selected]),
        np.concatenate([record.labels for record in selected]),
    )


_BOOTSTRAP_METRICS = (
    "roc_auc",
    "pr_auc",
    "f1",
    "false_alarm_rate",
    "miss_rate",
)


def _validate_bootstrap_iterations(iterations: int) -> int:
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)):
        raise ValueError("bootstrap_iterations must be a non-negative integer")
    if iterations < 0:
        raise ValueError("bootstrap_iterations must be a non-negative integer")
    return int(iterations)


def bootstrap_confidence_intervals(
    records: Iterable[EvaluationRecord],
    threshold: float,
    iterations: int = 0,
    seed: int | None = 0,
) -> dict[str, dict[str, float]]:
    """Estimate fixed-threshold metrics with an item-level percentile bootstrap.

    Each replicate samples complete :class:`EvaluationRecord` objects with
    replacement. Frames within a sampled item remain together, preserving
    within-call dependence while the aggregate metrics remain frame-weighted.
    The supplied threshold is used for every replicate; no threshold is
    selected or calibrated from a bootstrap sample.
    """
    iterations = _validate_bootstrap_iterations(iterations)
    if iterations == 0:
        return {}

    selected_records = list(records)
    if not selected_records:
        raise ValueError("bootstrap requires at least one evaluation record")

    generator = np.random.default_rng(seed)
    samples = {metric: np.empty(iterations, dtype=np.float64) for metric in _BOOTSTRAP_METRICS}
    for index in range(iterations):
        selected_indices = generator.integers(
            0,
            len(selected_records),
            size=len(selected_records),
        )
        sampled_records = [selected_records[int(item)] for item in selected_indices]
        probabilities, labels = _combine(sampled_records)
        metrics = _frame_report(probabilities, labels, threshold)
        for metric in _BOOTSTRAP_METRICS:
            samples[metric][index] = float(metrics[metric])

    intervals: dict[str, dict[str, float]] = {}
    for metric, values in samples.items():
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            lower, upper = np.percentile(finite_values, (2.5, 97.5))
        else:
            lower = upper = float("nan")
        intervals[metric] = {"lower": float(lower), "upper": float(upper)}
    return intervals


def evaluation_report(
    records: list[EvaluationRecord],
    threshold: float,
    hop_ms: float,
    detector_config: DetectorConfig | None = None,
    *,
    bootstrap_iterations: int = 0,
    bootstrap_seed: int | None = 0,
) -> dict[str, object]:
    bootstrap_iterations = _validate_bootstrap_iterations(bootstrap_iterations)
    probabilities, labels = _combine(records)
    separator = np.zeros(1, dtype=np.float32)
    boundary_probabilities = np.concatenate(
        [
            value
            for record in records
            for value in (record.probabilities, separator)
        ]
    )
    boundary_labels = np.concatenate(
        [
            value
            for record in records
            for value in (record.labels, separator)
        ]
    )
    oracle_threshold, oracle_metrics = best_f1_threshold(probabilities, labels)
    groups: dict[str, dict[str, dict[str, float | int]]] = {}
    for field in ("language", "domain", "channel", "codec", "device", "condition"):
        grouped: dict[str, dict[str, float | int]] = {}
        values = sorted({str(getattr(record.item, field)) for record in records})
        for value in values:
            group_probabilities, group_labels = _combine(
                record for record in records if str(getattr(record.item, field)) == value
            )
            grouped[value] = _frame_report(group_probabilities, group_labels, threshold)
        groups[field] = grouped
    snr_groups: dict[str, list[EvaluationRecord]] = {}
    for record in records:
        if record.item.snr_db is None:
            bucket = "unknown"
        elif record.item.snr_db < 0:
            bucket = "<0dB"
        elif record.item.snr_db < 5:
            bucket = "0-5dB"
        elif record.item.snr_db < 15:
            bucket = "5-15dB"
        else:
            bucket = ">=15dB"
        snr_groups.setdefault(bucket, []).append(record)
    groups["snr"] = {}
    for bucket, bucket_records in sorted(snr_groups.items()):
        group_probabilities, group_labels = _combine(bucket_records)
        groups["snr"][bucket] = _frame_report(
            group_probabilities,
            group_labels,
            threshold,
        )
    report: dict[str, object] = {
        "aggregate": {
            **_frame_report(probabilities, labels, threshold),
            **boundary_metrics(
                boundary_probabilities,
                boundary_labels,
                threshold,
                hop_ms,
            ),
        },
        "oracle_test_threshold": {
            "warning": "Diagnostic only. Never select a release threshold on the test set.",
            "threshold": oracle_threshold,
            **oracle_metrics,
        },
        "groups": groups,
    }
    if bootstrap_iterations > 0:
        report["confidence_intervals"] = {
            "method": "item_bootstrap",
            "confidence_level": 0.95,
            "iterations": bootstrap_iterations,
            "seed": bootstrap_seed,
            "threshold": threshold,
            "threshold_policy": "fixed_supplied_threshold",
            "metrics": bootstrap_confidence_intervals(
                records,
                threshold,
                bootstrap_iterations,
                bootstrap_seed,
            ),
        }
    if detector_config is not None:
        decision_records = [
            EvaluationRecord(
                record.item,
                detector_decisions(record.probabilities, detector_config, hop_ms),
                record.labels,
            )
            for record in records
        ]
        decisions, decision_labels = _combine(decision_records)
        boundary_decisions = np.concatenate(
            [
                value
                for record in decision_records
                for value in (record.probabilities, separator)
            ]
        )
        report["production_detector"] = {
            "detector": asdict(detector_config),
            "frames": int(decision_labels.size),
            "speech_fraction": float(np.mean(decision_labels)),
            **binary_metrics(decisions, decision_labels, 0.5),
            **boundary_metrics(
                boundary_decisions,
                boundary_labels,
                0.5,
                hop_ms,
            ),
        }
        if bootstrap_iterations > 0:
            confidence_intervals = report["confidence_intervals"]
            assert isinstance(confidence_intervals, dict)
            confidence_intervals["production_detector_metrics"] = (
                bootstrap_confidence_intervals(
                    decision_records,
                    0.5,
                    bootstrap_iterations,
                    bootstrap_seed,
                )
            )
    return report


@torch.inference_mode()
def evaluate_checkpoint(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_path: str | Path | None = None,
    threshold: float | None = None,
    *,
    bootstrap_iterations: int = 0,
    bootstrap_seed: int | None = 0,
) -> dict[str, object]:
    checkpoint = load_checkpoint_data(checkpoint_path)
    config, model = load_checkpoint(checkpoint_path)
    frontend = CausalFeatureExtractor(config.feature)
    release_threshold = (
        float(threshold)
        if threshold is not None
        else float(checkpoint.get("threshold", config.detector.start_threshold))
    )
    records: list[EvaluationRecord] = []
    for item in load_manifest(manifest_path):
        audio = read_audio(item.audio, config.feature.sample_rate)
        features = frontend(torch.from_numpy(audio))
        probabilities = torch.sigmoid(model(features)["speech_logits"]).squeeze(0).numpy()
        labels = frame_labels(item.segments, probabilities.size, config.feature)
        records.append(EvaluationRecord(item, probabilities, labels))

    report = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "manifest": str(Path(manifest_path).resolve()),
        "items": len(records),
        "report": evaluation_report(
            records,
            release_threshold,
            config.feature.hop_ms,
            config.detector,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        ),
    }
    serialized = json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n"
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return report
