from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler

from .manifest import ManifestItem


@dataclass(frozen=True)
class RegionProfile:
    name: str
    language_weights: dict[str, float]
    domain_weights: dict[str, float]
    unknown_language_weight: float = 0.2
    unknown_domain_weight: float = 0.5

    @classmethod
    def load(cls, path: str | Path) -> RegionProfile:
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls(
            name=str(raw["name"]),
            language_weights={
                str(key): float(value) for key, value in raw["language_weights"].items()
            },
            domain_weights={str(key): float(value) for key, value in raw["domain_weights"].items()},
            unknown_language_weight=float(raw.get("unknown_language_weight", 0.2)),
            unknown_domain_weight=float(raw.get("unknown_domain_weight", 0.5)),
        )

    def language_weight(self, language: str) -> float:
        if language in self.language_weights:
            return self.language_weights[language]
        base = language.split("-", 1)[0]
        return self.language_weights.get(base, self.unknown_language_weight)

    def domain_weight(self, domain: str) -> float:
        return self.domain_weights.get(domain, self.unknown_domain_weight)


def balanced_item_weights(
    items: list[ManifestItem],
    profile: RegionProfile,
) -> torch.Tensor:
    """Balance corpus volume while preserving explicit regional priorities."""

    counts = Counter(item.language for item in items)
    weights = [
        profile.language_weight(item.language)
        * profile.domain_weight(item.domain)
        / counts[item.language]
        for item in items
    ]
    return torch.tensor(weights, dtype=torch.double)


def region_sampler(
    items: list[ManifestItem],
    profile: RegionProfile,
    seed: int,
) -> WeightedRandomSampler:
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        balanced_item_weights(items, profile),
        num_samples=len(items),
        replacement=True,
        generator=generator,
    )
