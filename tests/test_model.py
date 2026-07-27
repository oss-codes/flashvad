import torch

from flashvad.config import ModelConfig
from flashvad.model import FlashVad


def test_streaming_model_matches_offline_model() -> None:
    torch.manual_seed(9)
    config = ModelConfig(dropout=0.0)
    model = FlashVad(config).eval()
    features = torch.randn(2, 21, config.feature_dim)
    expected = model(features)["speech_logits"]

    state = model.initial_state(features.shape[0], "cpu")
    outputs = []
    for frame in features.unbind(dim=1):
        result, state = model.stream_step(frame, state)
        outputs.append(result["speech_logits"].squeeze(1))
    actual = torch.stack(outputs, dim=1)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_default_model_meets_parameter_budget() -> None:
    model = FlashVad()
    assert model.parameter_count < 250_000
    assert model.parameter_count > 25_000
