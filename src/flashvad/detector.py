"""Lightweight streaming detector policy shared by every runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from .config import DetectorConfig


@dataclass(frozen=True)
class VadEvent:
    kind: Literal["speech_start", "speech_end"]
    frame: int
    time_seconds: float
    probability: float


class HysteresisDetector:
    def __init__(self, config: DetectorConfig, hop_seconds: float) -> None:
        config.validate()
        self.config = config
        self.hop_seconds = hop_seconds
        self.reset()

    def reset(self) -> None:
        self.active = False
        self.frame_index = -1
        self.above_count = 0
        self.below_count = 0

    def update(self, probability: float) -> list[VadEvent]:
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be finite and between 0 and 1")
        self.frame_index += 1
        events: list[VadEvent] = []
        if not self.active:
            self.above_count = (
                self.above_count + 1 if probability >= self.config.start_threshold else 0
            )
            if self.above_count >= self.config.start_frames:
                start_frame = max(
                    0,
                    self.frame_index
                    - self.config.start_frames
                    + 1
                    - self.config.pre_roll_frames,
                )
                self.active = True
                self.below_count = 0
                events.append(
                    VadEvent(
                        "speech_start",
                        start_frame,
                        start_frame * self.hop_seconds,
                        probability,
                    )
                )
        else:
            self.below_count = (
                self.below_count + 1 if probability <= self.config.stop_threshold else 0
            )
            if self.below_count >= self.config.stop_frames:
                end_frame = self.frame_index - self.config.stop_frames + 1
                self.active = False
                self.above_count = 0
                events.append(
                    VadEvent(
                        "speech_end",
                        end_frame,
                        end_frame * self.hop_seconds,
                        probability,
                    )
                )
        return events


__all__ = ["HysteresisDetector", "VadEvent"]
