from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .audio import read_audio
from .augment import augment_audio
from .config import FeatureConfig
from .teacher import blend_teacher_probabilities


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    label: str = "speech"


@dataclass(frozen=True)
class ManifestItem:
    audio: Path
    sample_rate: int
    language: str
    domain: str
    segments: tuple[Segment, ...]
    channel: str = "unknown"
    codec: str = "unknown"
    device: str = "unknown"
    condition: str = "unknown"
    snr_db: float | None = None
    teacher_probabilities: Path | None = None
    teacher_weight: float = 1.0
    teacher_confidence_weighting: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any], root: Path) -> ManifestItem:
        audio_path = Path(raw["audio"])
        if not audio_path.is_absolute():
            audio_path = (root / audio_path).resolve()
        teacher_path = (
            Path(raw["teacher_probabilities"])
            if raw.get("teacher_probabilities")
            else None
        )
        if teacher_path is not None and not teacher_path.is_absolute():
            teacher_path = (root / teacher_path).resolve()
        segments = tuple(Segment(**segment) for segment in raw.get("segments", []))
        teacher_weight = float(raw.get("teacher_weight", 1.0))
        if not 0.0 <= teacher_weight <= 1.0:
            raise ValueError("teacher_weight must be between 0 and 1")
        return cls(
            audio=audio_path,
            sample_rate=int(raw.get("sample_rate", 16_000)),
            language=str(raw.get("language", "und")),
            domain=str(raw.get("domain", "unknown")),
            segments=segments,
            channel=str(raw.get("channel", "unknown")),
            codec=str(raw.get("codec", "unknown")),
            device=str(raw.get("device", "unknown")),
            condition=str(raw.get("condition", "unknown")),
            snr_db=(float(raw["snr_db"]) if raw.get("snr_db") is not None else None),
            teacher_probabilities=teacher_path,
            teacher_weight=teacher_weight,
            teacher_confidence_weighting=bool(
                raw.get("teacher_confidence_weighting", True)
            ),
        )


def rebase_manifest_record(
    record: dict[str, Any],
    source_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Rebase manifest file references without changing their targets."""
    rebased = dict(record)
    source = Path(source_root).resolve()
    destination = Path(output_root).resolve()
    for key in ("audio", "teacher_probabilities"):
        raw = rebased.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        if not path.is_absolute():
            path = (source / path).resolve()
        rebased[key] = os.path.relpath(path, destination)
    return rebased


def load_manifest(path: str | Path) -> list[ManifestItem]:
    manifest_path = Path(path).resolve()
    items: list[ManifestItem] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                items.append(ManifestItem.from_dict(json.loads(line), manifest_path.parent))
            except Exception as exc:
                raise ValueError(f"invalid manifest line {line_number}: {exc}") from exc
    if not items:
        raise ValueError(f"manifest is empty: {manifest_path}")
    return items


def frame_labels(
    segments: tuple[Segment, ...],
    num_frames: int,
    config: FeatureConfig,
    offset_seconds: float = 0.0,
) -> np.ndarray:
    frame_ends = (np.arange(num_frames) + 1) * config.hop_ms / 1_000 + offset_seconds
    labels = np.zeros(num_frames, dtype=np.float32)
    for segment in segments:
        if segment.label != "speech":
            continue
        labels[(frame_ends > segment.start) & (frame_ends <= segment.end)] = 1.0
    return labels


def frame_auxiliary_labels(
    segments: tuple[Segment, ...],
    num_frames: int,
    config: FeatureConfig,
    offset_seconds: float = 0.0,
) -> np.ndarray:
    frame_ends = (np.arange(num_frames) + 1) * config.hop_ms / 1_000 + offset_seconds
    labels = np.zeros((num_frames, 3), dtype=np.float32)
    class_index = {
        "speech": 0,
        "music": 1,
        "singing": 2,
        "other_vocal": 2,
        "laughter": 2,
        "cough": 2,
    }
    for segment in segments:
        index = class_index.get(segment.label)
        if index is None:
            continue
        labels[
            (frame_ends > segment.start) & (frame_ends <= segment.end),
            index,
        ] = 1.0
    return labels


class VadDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    def __init__(
        self,
        manifest: str | Path,
        feature_config: FeatureConfig,
        chunk_seconds: float,
        augment: bool = False,
        seed: int = 1337,
        noise_manifest: str | Path | None = None,
    ) -> None:
        self.items = load_manifest(manifest)
        self.noise_items = load_manifest(noise_manifest) if noise_manifest else []
        self.feature_config = feature_config
        self.chunk_samples = round(chunk_seconds * feature_config.sample_rate)
        self.augment = augment
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        item = self.items[index]
        audio = read_audio(item.audio, self.feature_config.sample_rate)
        rng = np.random.default_rng(np.random.SeedSequence((self.seed, self.epoch, index)))
        max_start = max(0, audio.size - self.chunk_samples)
        start = int(rng.integers(0, max_start + 1)) if max_start else 0
        chunk = audio[start : start + self.chunk_samples]
        if chunk.size < self.chunk_samples:
            chunk = np.pad(chunk, (0, self.chunk_samples - chunk.size))
        noise = None
        if self.noise_items:
            noise_item = self.noise_items[int(rng.integers(0, len(self.noise_items)))]
            noise_audio = read_audio(noise_item.audio, self.feature_config.sample_rate)
            noise_max_start = max(0, noise_audio.size - self.chunk_samples)
            noise_start = (
                int(rng.integers(0, noise_max_start + 1)) if noise_max_start else 0
            )
            noise = noise_audio[noise_start : noise_start + self.chunk_samples]
            if noise.size < self.chunk_samples:
                noise = np.pad(noise, (0, self.chunk_samples - noise.size))
        if self.augment:
            chunk = augment_audio(
                chunk,
                self.feature_config.sample_rate,
                rng,
                noise=noise,
            )
        num_frames = int(np.ceil(chunk.size / self.feature_config.hop_samples))
        labels = frame_labels(
            item.segments,
            num_frames,
            self.feature_config,
            offset_seconds=start / self.feature_config.sample_rate,
        )
        if item.teacher_probabilities is not None:
            teacher = np.asarray(
                np.load(item.teacher_probabilities),
                dtype=np.float32,
            ).reshape(-1)
            start_frame = round(
                start / self.feature_config.sample_rate
                / (self.feature_config.hop_ms / 1_000)
            )
            teacher = teacher[start_frame : start_frame + num_frames]
            if teacher.size < num_frames:
                teacher = np.pad(teacher, (0, num_frames - teacher.size))
            teacher = np.clip(teacher, 0.0, 1.0)
            if item.teacher_confidence_weighting:
                labels = blend_teacher_probabilities(
                    labels,
                    teacher,
                    weight=item.teacher_weight,
                )
            else:
                labels = (
                    (1.0 - item.teacher_weight) * labels
                    + item.teacher_weight * teacher
                ).astype(np.float32)
        auxiliary = frame_auxiliary_labels(
            item.segments,
            num_frames,
            self.feature_config,
            offset_seconds=start / self.feature_config.sample_rate,
        )
        return (
            torch.from_numpy(chunk.copy()),
            torch.from_numpy(labels),
            torch.from_numpy(auxiliary),
        )
