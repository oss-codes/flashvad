from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16_000
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    n_fft: int = 512
    n_mels: int = 40
    f_min: float = 50.0
    f_max: float = 7_600.0

    @property
    def frame_samples(self) -> int:
        return round(self.sample_rate * self.frame_ms / 1_000)

    @property
    def hop_samples(self) -> int:
        return round(self.sample_rate * self.hop_ms / 1_000)

    @property
    def feature_dim(self) -> int:
        return self.n_mels + 3

    def validate(self) -> None:
        if self.sample_rate not in (8_000, 16_000):
            raise ValueError("sample_rate must be 8000 or 16000")
        if self.frame_samples <= self.hop_samples:
            raise ValueError("frame must be longer than hop")
        if self.n_fft < self.frame_samples:
            raise ValueError("n_fft must be at least frame_samples")
        if not 0 <= self.f_min < self.f_max <= self.sample_rate / 2:
            raise ValueError("mel frequency bounds are invalid")


@dataclass(frozen=True)
class ModelConfig:
    feature_dim: int = 43
    hidden_dim: int = 64
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4)
    recurrent_dim: int = 56
    dropout: float = 0.08

    def validate(self) -> None:
        if self.kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        if any(dilation < 1 for dilation in self.dilations):
            raise ValueError("dilations must be positive")
        if self.hidden_dim < 8 or self.recurrent_dim < 8:
            raise ValueError("model dimensions are too small")


@dataclass(frozen=True)
class TrainingConfig:
    chunk_seconds: float = 4.0
    batch_size: int = 16
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    positive_weight: float = 1.4
    focal_gamma: float = 0.0
    boundary_weight: float = 0.15
    auxiliary_weight: float = 0.10
    gradient_clip: float = 5.0
    detector_max_false_alarm_rate: float = 0.2


@dataclass(frozen=True)
class DetectorConfig:
    start_threshold: float = 0.62
    stop_threshold: float = 0.36
    start_frames: int = 2
    stop_frames: int = 10
    pre_roll_frames: int = 3

    def validate(self) -> None:
        if not 0 < self.stop_threshold < self.start_threshold < 1:
            raise ValueError("expected 0 < stop_threshold < start_threshold < 1")
        if self.start_frames < 1 or self.stop_frames < 1 or self.pre_roll_frames < 0:
            raise ValueError("frame counts must be non-negative")


@dataclass(frozen=True)
class ProjectConfig:
    seed: int = 1337
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    def validate(self) -> None:
        self.feature.validate()
        self.model.validate()
        self.detector.validate()
        if not 0.0 <= self.training.detector_max_false_alarm_rate <= 1.0:
            raise ValueError("detector_max_false_alarm_rate must be between zero and one")
        if self.training.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative")
        if self.model.feature_dim != self.feature.feature_dim:
            raise ValueError(
                f"model feature_dim={self.model.feature_dim} does not match "
                f"frontend feature_dim={self.feature.feature_dim}"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ProjectConfig:
        model_raw = dict(raw.get("model", {}))
        if "dilations" in model_raw:
            model_raw["dilations"] = tuple(model_raw["dilations"])
        config = cls(
            seed=int(raw.get("seed", 1337)),
            feature=FeatureConfig(**raw.get("feature", {})),
            model=ModelConfig(**model_raw),
            training=TrainingConfig(**raw.get("training", {})),
            detector=DetectorConfig(**raw.get("detector", {})),
        )
        config.validate()
        return config

    @classmethod
    def load(cls, path: str | Path) -> ProjectConfig:
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)
            handle.write("\n")
