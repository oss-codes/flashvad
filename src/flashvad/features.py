from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import FeatureConfig


def _hz_to_mel(frequency: Tensor) -> Tensor:
    return 2595.0 * torch.log10(1.0 + frequency / 700.0)


def _mel_to_hz(mel: Tensor) -> Tensor:
    return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)


def build_mel_filterbank(config: FeatureConfig) -> Tensor:
    config.validate()
    frequencies = torch.linspace(0, config.sample_rate / 2, config.n_fft // 2 + 1)
    mel_min = _hz_to_mel(torch.tensor(config.f_min))
    mel_max = _hz_to_mel(torch.tensor(config.f_max))
    mel_points = torch.linspace(mel_min, mel_max, config.n_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    lower = hz_points[:-2, None]
    center = hz_points[1:-1, None]
    upper = hz_points[2:, None]
    rising = (frequencies - lower) / (center - lower).clamp_min(1e-8)
    falling = (upper - frequencies) / (upper - center).clamp_min(1e-8)
    filters = torch.minimum(rising, falling).clamp(0.0, 1.0)

    # Area normalization prevents wider high-frequency filters from dominating.
    enorm = 2.0 / (upper.squeeze(1) - lower.squeeze(1)).clamp_min(1e-8)
    return filters * enorm[:, None]


class CausalFeatureExtractor(nn.Module):
    """Causal log-mel frontend with three inexpensive robustness features.

    Output features are 40 log-mel bands by default, followed by log RMS,
    log spectral flatness, and zero-crossing rate. Frames are left-padded so
    frame N only observes samples available by the end of hop N.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        super().__init__()
        self.config = config or FeatureConfig()
        self.config.validate()
        self.register_buffer("window", torch.hann_window(self.config.frame_samples))
        self.register_buffer("mel_filterbank", build_mel_filterbank(self.config))

    def frames(self, audio: Tensor) -> Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.ndim != 2:
            raise ValueError("audio must have shape [samples] or [batch, samples]")
        left = self.config.frame_samples - self.config.hop_samples
        padded = F.pad(audio, (left, 0))
        if padded.shape[-1] < self.config.frame_samples:
            padded = F.pad(padded, (0, self.config.frame_samples - padded.shape[-1]))
        remainder = (padded.shape[-1] - self.config.frame_samples) % self.config.hop_samples
        if remainder:
            padded = F.pad(padded, (0, self.config.hop_samples - remainder))
        return padded.unfold(-1, self.config.frame_samples, self.config.hop_samples)

    def from_frames(self, frames: Tensor) -> Tensor:
        if frames.shape[-1] != self.config.frame_samples:
            raise ValueError(
                f"expected {self.config.frame_samples} samples per frame, got {frames.shape[-1]}"
            )
        windowed = frames * self.window
        spectrum = torch.fft.rfft(windowed, n=self.config.n_fft, dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()

        mel_energy = torch.matmul(power, self.mel_filterbank.transpose(0, 1))
        log_mel = torch.log(mel_energy.clamp_min(1e-8))
        log_mel = log_mel - log_mel.mean(dim=-1, keepdim=True)

        rms = torch.sqrt(frames.square().mean(dim=-1).clamp_min(1e-10))
        log_rms = torch.log(rms).unsqueeze(-1)
        geometric = torch.exp(torch.log(power.clamp_min(1e-10)).mean(dim=-1))
        arithmetic = power.mean(dim=-1).clamp_min(1e-10)
        log_flatness = torch.log((geometric / arithmetic).clamp_min(1e-10)).unsqueeze(-1)
        zcr = (frames[..., 1:] * frames[..., :-1] < 0).float().mean(dim=-1, keepdim=True)
        return torch.cat((log_mel, log_rms, log_flatness, zcr), dim=-1)

    def forward(self, audio: Tensor) -> Tensor:
        return self.from_frames(self.frames(audio.float()))


@dataclass
class StreamingFeatureState:
    history: Tensor
    pending: Tensor


class StreamingFeatureExtractor:
    """Arbitrary-chunk streaming wrapper exactly matching the offline frontend."""

    def __init__(
        self, frontend: CausalFeatureExtractor, device: torch.device | str = "cpu"
    ) -> None:
        self.frontend = frontend.to(device)
        self.device = torch.device(device)
        self.reset()

    def reset(self) -> None:
        history_size = self.frontend.config.frame_samples - self.frontend.config.hop_samples
        self.state = StreamingFeatureState(
            history=torch.zeros(history_size, device=self.device),
            pending=torch.zeros(0, device=self.device),
        )

    @torch.inference_mode()
    def push(self, audio: Tensor) -> Tensor:
        if audio.ndim != 1:
            raise ValueError("audio must be one-dimensional")
        incoming = audio.to(self.device, dtype=torch.float32).flatten()
        if incoming.numel() and not torch.isfinite(incoming).all():
            raise ValueError("audio samples must be finite")
        pending = torch.cat((self.state.pending, incoming))
        outputs: list[Tensor] = []
        hop = self.frontend.config.hop_samples
        while pending.numel() >= hop:
            current_hop = pending[:hop]
            pending = pending[hop:]
            frame = torch.cat((self.state.history, current_hop))
            features = self.frontend.from_frames(frame.view(1, 1, -1)).squeeze(0).squeeze(0)
            outputs.append(features)
            history_size = self.state.history.numel()
            self.state.history = frame[-history_size:]
        self.state.pending = pending
        if not outputs:
            return torch.empty(0, self.frontend.config.feature_dim, device=self.device)
        return torch.stack(outputs)
