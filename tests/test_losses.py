import pytest
import torch
import torch.nn.functional as F

from flashvad.config import ProjectConfig
from flashvad.losses import binary_focal_loss_with_logits


def test_zero_gamma_matches_weighted_binary_cross_entropy() -> None:
    logits = torch.tensor([[1.5, -0.5, 0.25]])
    targets = torch.tensor([[1.0, 0.0, 0.4]])
    positive_weight = torch.tensor(1.7)

    actual = binary_focal_loss_with_logits(
        logits,
        targets,
        gamma=0.0,
        positive_weight=positive_weight,
    )
    expected = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
    )

    torch.testing.assert_close(actual, expected)


def test_focal_loss_concentrates_on_hard_predictions() -> None:
    easy = binary_focal_loss_with_logits(
        torch.tensor([4.0, -4.0]),
        torch.tensor([1.0, 0.0]),
        gamma=2.0,
    )
    hard = binary_focal_loss_with_logits(
        torch.tensor([-4.0, 4.0]),
        torch.tensor([1.0, 0.0]),
        gamma=2.0,
    )

    assert hard > easy * 1_000


def test_project_config_rejects_negative_focal_gamma() -> None:
    with pytest.raises(ValueError, match="focal_gamma"):
        ProjectConfig.from_dict({"training": {"focal_gamma": -0.1}})
