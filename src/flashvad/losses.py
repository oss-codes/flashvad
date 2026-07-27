from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def binary_focal_loss_with_logits(
    logits: Tensor,
    targets: Tensor,
    *,
    gamma: float,
    positive_weight: Tensor | None = None,
) -> Tensor:
    """Binary cross entropy that can focus training on hard frames.

    ``gamma=0`` is exactly weighted binary cross entropy. Soft teacher targets
    are supported by computing the probability assigned to the target mixture.
    """
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have the same shape")
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=positive_weight,
        reduction="none",
    )
    if gamma == 0:
        return cross_entropy.mean()
    probabilities = torch.sigmoid(logits)
    target_probability = targets * probabilities + (1.0 - targets) * (
        1.0 - probabilities
    )
    return (cross_entropy * (1.0 - target_probability).pow(gamma)).mean()
