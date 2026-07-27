from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


def load_checkpoint_data(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a FlashVAD checkpoint without enabling arbitrary pickle execution."""
    checkpoint = torch.load(
        Path(path),
        map_location=map_location,
        weights_only=True,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    if not isinstance(checkpoint.get("config"), Mapping):
        raise ValueError("checkpoint is missing a valid config mapping")
    if not isinstance(checkpoint.get("model"), Mapping):
        raise ValueError("checkpoint is missing a valid model state mapping")
    return dict(checkpoint)
