import numpy as np

from flashvad.config import FeatureConfig
from flashvad.manifest import Segment, frame_auxiliary_labels


def test_auxiliary_labels_separate_speech_music_and_vocal_events() -> None:
    config = FeatureConfig()
    segments = (
        Segment(0.0, 0.02, "speech"),
        Segment(0.02, 0.04, "music"),
        Segment(0.04, 0.06, "laughter"),
    )
    labels = frame_auxiliary_labels(segments, 6, config)
    np.testing.assert_array_equal(
        labels,
        np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=np.float32,
        ),
    )
