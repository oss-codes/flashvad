import numpy as np

from flashvad.teacher import (
    blend_teacher_probabilities,
    interpolate_probabilities,
)


def test_teacher_probabilities_align_to_ten_millisecond_frames() -> None:
    probabilities = interpolate_probabilities(
        np.array([0.2, 0.8], dtype=np.float32),
        source_hop_ms=32,
        target_hop_ms=10,
        target_frames=7,
    )

    assert probabilities.shape == (7,)
    assert probabilities[0] == np.float32(0.2)
    assert probabilities[-1] == np.float32(0.8)
    assert np.all(np.diff(probabilities) >= 0)


def test_uncertain_teacher_does_not_override_weak_labels() -> None:
    labels = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    teacher = np.array([0.5, 0.5, 0.95, 0.05], dtype=np.float32)

    blended = blend_teacher_probabilities(labels, teacher, weight=0.8)

    np.testing.assert_allclose(blended[:2], labels[:2])
    assert blended[2] > 0.65
    assert blended[3] < 0.35
