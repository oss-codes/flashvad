import pytest

from flashvad.config import DetectorConfig
from flashvad.streaming import HysteresisDetector


def test_hysteresis_emits_stable_boundaries() -> None:
    detector = HysteresisDetector(
        DetectorConfig(
            start_threshold=0.6,
            stop_threshold=0.3,
            start_frames=2,
            stop_frames=3,
            pre_roll_frames=1,
        ),
        hop_seconds=0.01,
    )
    probabilities = [0.1, 0.7, 0.8, 0.4, 0.2, 0.5, 0.2, 0.1, 0.1]
    events = [event for probability in probabilities for event in detector.update(probability)]

    assert [event.kind for event in events] == ["speech_start", "speech_end"]
    assert events[0].frame == 0
    assert events[1].frame == 6


def test_mid_speech_dip_does_not_end_turn() -> None:
    detector = HysteresisDetector(DetectorConfig(stop_frames=4), hop_seconds=0.01)
    probabilities = [0.8, 0.8, 0.1, 0.1, 0.7, 0.8]
    events = [event for probability in probabilities for event in detector.update(probability)]
    assert [event.kind for event in events] == ["speech_start"]


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan"), float("inf")])
def test_hysteresis_rejects_invalid_probabilities_without_advancing(
    probability: float,
) -> None:
    detector = HysteresisDetector(DetectorConfig(), hop_seconds=0.01)

    with pytest.raises(ValueError):
        detector.update(probability)

    assert detector.frame_index == -1
