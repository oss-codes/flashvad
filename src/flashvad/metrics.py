from __future__ import annotations

import numpy as np


def _validated_inputs(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    raw_target = np.asarray(labels).reshape(-1)
    if not np.all((raw_target == 0) | (raw_target == 1)):
        raise ValueError("labels must be binary")
    target = raw_target.astype(np.int64)
    if probability.size != target.size:
        raise ValueError("probabilities and labels must have equal length")
    if probability.size == 0:
        raise ValueError("metrics require at least one frame")
    if not np.all(np.isfinite(probability)):
        raise ValueError("probabilities contain non-finite values")
    return probability, target


def binary_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    probability, target = _validated_inputs(probabilities, labels)
    prediction = probability >= threshold
    positive = target == 1
    negative = ~positive

    true_positive = int(np.sum(prediction & positive))
    false_positive = int(np.sum(prediction & negative))
    false_negative = int(np.sum(~prediction & positive))
    true_negative = int(np.sum(~prediction & negative))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": (true_positive + true_negative) / max(target.size, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarm_rate": false_positive / max(int(np.sum(negative)), 1),
        "miss_rate": false_negative / max(int(np.sum(positive)), 1),
    }


def roc_auc(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probability, target = _validated_inputs(probabilities, labels)
    positive_count = int(np.sum(target == 1))
    negative_count = int(np.sum(target == 0))
    if positive_count == 0 or negative_count == 0:
        return float("nan")

    order = np.argsort(probability, kind="mergesort")
    sorted_probability = probability[order]
    ranks = np.empty(probability.size, dtype=np.float64)
    cursor = 0
    while cursor < probability.size:
        end = cursor + 1
        while end < probability.size and sorted_probability[end] == sorted_probability[cursor]:
            end += 1
        average_rank = 0.5 * ((cursor + 1) + end)
        ranks[order[cursor:end]] = average_rank
        cursor = end
    positive_rank_sum = float(np.sum(ranks[target == 1]))
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)


def average_precision(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probability, target = _validated_inputs(probabilities, labels)
    positive_count = int(np.sum(target == 1))
    if positive_count == 0:
        return float("nan")
    order = np.argsort(-probability, kind="mergesort")
    sorted_probability = probability[order]
    sorted_target = target[order]
    cumulative_positive = 0
    cumulative_count = 0
    score = 0.0
    cursor = 0
    while cursor < target.size:
        end = cursor + 1
        while end < target.size and sorted_probability[end] == sorted_probability[cursor]:
            end += 1
        group_positive = int(np.sum(sorted_target[cursor:end]))
        cumulative_positive += group_positive
        cumulative_count = end
        precision = cumulative_positive / cumulative_count
        score += group_positive / positive_count * precision
        cursor = end
    return float(score)


def best_f1_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    candidates = thresholds if thresholds is not None else np.linspace(0.05, 0.95, 91)
    results = [
        (float(threshold), binary_metrics(probabilities, labels, float(threshold)))
        for threshold in candidates
    ]
    return max(results, key=lambda item: item[1]["f1"])
