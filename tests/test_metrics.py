import numpy as np
import pytest

from flashvad.metrics import best_f1_threshold, binary_metrics


def test_binary_metrics_and_threshold_search() -> None:
    probabilities = np.array([0.01, 0.2, 0.7, 0.9])
    labels = np.array([0, 0, 1, 1])
    metrics = binary_metrics(probabilities, labels, 0.5)
    threshold, best = best_f1_threshold(probabilities, labels)

    assert metrics["f1"] == 1.0
    assert best["f1"] == 1.0
    assert 0.2 < threshold <= 0.7


def test_binary_metrics_reject_soft_labels_instead_of_truncating_them() -> None:
    with pytest.raises(ValueError, match="labels must be binary"):
        binary_metrics(
            np.array([0.2, 0.8]),
            np.array([0.1, 0.9]),
            0.5,
        )
