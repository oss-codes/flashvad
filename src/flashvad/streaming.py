from __future__ import annotations

import torch
from torch import Tensor

from .config import DetectorConfig
from .detector import HysteresisDetector, VadEvent
from .features import CausalFeatureExtractor, StreamingFeatureExtractor
from .model import FlashVad, StreamingModelState


class StreamingVadEngine:
    def __init__(
        self,
        frontend: CausalFeatureExtractor,
        model: FlashVad,
        detector_config: DetectorConfig,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.frontend = StreamingFeatureExtractor(frontend, self.device)
        self.model = model.to(self.device).eval()
        self.detector = HysteresisDetector(
            detector_config,
            frontend.config.hop_ms / 1_000,
        )
        self.model_state: StreamingModelState
        self.reset()

    def reset(self) -> None:
        self.frontend.reset()
        self.detector.reset()
        self.model_state = self.model.initial_state(1, self.device)

    @torch.inference_mode()
    def push(self, audio: Tensor) -> tuple[Tensor, list[VadEvent]]:
        features = self.frontend.push(audio)
        probabilities: list[Tensor] = []
        events: list[VadEvent] = []
        for feature in features:
            outputs, self.model_state = self.model.stream_step(
                feature.unsqueeze(0), self.model_state
            )
            probability = torch.sigmoid(outputs["speech_logits"]).squeeze()
            probabilities.append(probability)
            events.extend(self.detector.update(float(probability)))
        if not probabilities:
            return torch.empty(0, device=self.device), events
        return torch.stack(probabilities), events
