import numpy as np

from flashvad.audio import a_law_roundtrip, telephone_roundtrip


def test_a_law_roundtrip_is_finite_and_bounded() -> None:
    audio = np.linspace(-1, 1, 2_000, dtype=np.float32)
    output = a_law_roundtrip(audio)
    assert output.dtype == np.float32
    assert np.all(np.isfinite(output))
    assert float(np.max(np.abs(output))) <= 1.0


def test_telephone_roundtrip_preserves_length_for_both_codecs() -> None:
    audio = np.zeros(1_603, dtype=np.float32)
    for codec in ("mu-law", "a-law"):
        assert telephone_roundtrip(audio, 16_000, codec).shape == audio.shape
