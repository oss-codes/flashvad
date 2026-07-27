import numpy as np
import pytest
import torch

from flashvad.config import FeatureConfig
from flashvad.features import CausalFeatureExtractor, StreamingFeatureExtractor
from flashvad.numpy_features import NumpyCausalFeatureExtractor, NumpyStreamingFeatureExtractor


def test_streaming_features_match_offline_features() -> None:
    torch.manual_seed(4)
    config = FeatureConfig()
    frontend = CausalFeatureExtractor(config)
    audio = torch.randn(config.hop_samples * 13) * 0.05
    expected = frontend(audio).squeeze(0)

    stream = StreamingFeatureExtractor(frontend)
    chunks = (31, 257, 19, 503, 41, 809, 1_000)
    cursor = 0
    outputs = []
    for size in chunks:
        if cursor >= audio.numel():
            break
        outputs.append(stream.push(audio[cursor : cursor + size]))
        cursor += size
    outputs.append(stream.push(audio[cursor:]))
    actual = torch.cat([output for output in outputs if output.numel()], dim=0)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


def test_feature_dimension_is_explicit() -> None:
    config = FeatureConfig(n_mels=32)
    frontend = CausalFeatureExtractor(config)
    output = frontend(torch.zeros(config.hop_samples * 2))
    assert output.shape == (1, 2, 35)


def test_numpy_streaming_features_match_the_training_frontend() -> None:
    torch.manual_seed(8)
    config = FeatureConfig()
    audio = torch.randn(config.hop_samples * 13) * 0.05
    expected = CausalFeatureExtractor(config)(audio).squeeze(0).numpy()
    stream = NumpyStreamingFeatureExtractor(NumpyCausalFeatureExtractor(config))
    chunks = (31, 257, 19, 503, 41, 809, 1_000)
    cursor = 0
    outputs = []
    for size in chunks:
        if cursor >= audio.numel():
            break
        outputs.append(stream.push(audio[cursor : cursor + size].numpy()))
        cursor += size
    outputs.append(stream.push(audio[cursor:].numpy()))
    actual = np.concatenate([output for output in outputs if output.size], axis=0)

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "audio",
    [
        torch.zeros(1, 160),
        torch.tensor([0.0, float("nan")]),
        torch.tensor([0.0, float("inf")]),
    ],
)
def test_streaming_features_reject_invalid_audio(audio: torch.Tensor) -> None:
    stream = StreamingFeatureExtractor(CausalFeatureExtractor(FeatureConfig()))

    with pytest.raises(ValueError):
        stream.push(audio)
