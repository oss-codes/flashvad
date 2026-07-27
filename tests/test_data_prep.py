import numpy as np

from flashvad.data_prep import (
    compose_call,
    fixed_negative_excerpt,
    speech_regions,
    trim_speech,
)


def test_speech_regions_keep_real_pause_boundaries() -> None:
    sample_rate = 1_000
    tone = np.sin(np.linspace(0, 30, 400, dtype=np.float32)) * 0.3
    audio = np.concatenate(
        (
            np.zeros(300, dtype=np.float32),
            tone,
            np.zeros(150, dtype=np.float32),
            tone,
            np.zeros(300, dtype=np.float32),
        )
    )

    regions = speech_regions(audio, sample_rate, merge_gap_ms=80, pad_ms=0)

    assert len(regions) == 2
    assert regions[0][1] <= regions[1][0]


def test_trim_speech_removes_quiet_edges() -> None:
    sample_rate = 1_000
    audio = np.concatenate(
        (
            np.zeros(500, dtype=np.float32),
            np.full(800, 0.25, dtype=np.float32),
            np.zeros(500, dtype=np.float32),
        )
    )

    trimmed = trim_speech(audio, sample_rate, pad_ms=0)

    assert 790 <= trimmed.size <= 830
    assert np.max(np.abs(trimmed)) == 0.25


def test_compose_call_is_deterministic_and_labels_inserted_speech() -> None:
    speech = np.sin(np.linspace(0, 80, 32_000, dtype=np.float32)) * 0.2

    first = compose_call(speech, 16_000, np.random.default_rng(7))
    second = compose_call(speech, 16_000, np.random.default_rng(7))

    np.testing.assert_array_equal(first.audio, second.audio)
    assert first.audio.shape == (64_000,)
    assert first.segments
    assert all(0.0 <= start < end <= 4.0 for start, end in first.segments)


def test_compose_call_does_not_label_long_source_pauses_as_speech() -> None:
    sample_rate = 1_000
    tone = np.sin(np.linspace(0, 40, 400, dtype=np.float32)) * 0.3
    speech = np.concatenate(
        (
            np.zeros(300, dtype=np.float32),
            tone,
            np.zeros(500, dtype=np.float32),
            tone,
            np.zeros(300, dtype=np.float32),
        )
    )

    prepared = compose_call(
        speech,
        sample_rate,
        np.random.default_rng(17),
    )

    assert prepared.segments
    assert max(end - start for start, end in prepared.segments) <= 0.5


def test_compose_call_can_generate_noise_only_hard_negatives() -> None:
    prepared = compose_call(
        np.ones(16_000, dtype=np.float32),
        16_000,
        np.random.default_rng(9),
        hard_negative=True,
    )

    assert prepared.segments == ()
    assert np.any(prepared.audio)
    assert np.sqrt(np.mean(prepared.audio * prepared.audio)) >= 0.03


def test_fixed_negative_excerpt_is_deterministic_and_call_sized() -> None:
    source = np.linspace(-0.2, 0.2, 8_000, dtype=np.float32)

    first = fixed_negative_excerpt(
        source,
        16_000,
        np.random.default_rng(19),
        duration_seconds=4.0,
    )
    second = fixed_negative_excerpt(
        source,
        16_000,
        np.random.default_rng(19),
        duration_seconds=4.0,
    )

    assert first.shape == (64_000,)
    np.testing.assert_array_equal(first, second)
    assert 0.05 <= np.sqrt(np.mean(first * first)) <= 0.25
