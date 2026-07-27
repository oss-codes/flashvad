"""NumPy causal frontend for lightweight ONNX inference."""

from __future__ import annotations

import numpy as np

from .config import FeatureConfig


def _build_mel_filterbank(config: FeatureConfig) -> np.ndarray:
    frequencies = np.linspace(
        0,
        config.sample_rate / 2,
        config.n_fft // 2 + 1,
        dtype=np.float32,
    )
    mel_min = np.float32(2595.0) * np.log10(
        np.float32(1.0) + np.float32(config.f_min) / np.float32(700.0)
    )
    mel_max = np.float32(2595.0) * np.log10(
        np.float32(1.0) + np.float32(config.f_max) / np.float32(700.0)
    )
    mel_points = np.linspace(
        mel_min,
        mel_max,
        config.n_mels + 2,
        dtype=np.float32,
    )
    hz_points = np.float32(700.0) * (
        np.power(np.float32(10.0), mel_points / np.float32(2595.0))
        - np.float32(1.0)
    )
    lower = hz_points[:-2, None]
    center = hz_points[1:-1, None]
    upper = hz_points[2:, None]
    rising = (frequencies - lower) / np.maximum(
        center - lower,
        np.float32(1e-8),
    )
    falling = (upper - frequencies) / np.maximum(
        upper - center,
        np.float32(1e-8),
    )
    filters = np.maximum(
        np.float32(0),
        np.minimum(np.minimum(rising, falling), np.float32(1)),
    )
    normalization = np.float32(2.0) / np.maximum(
        upper[:, 0] - lower[:, 0],
        np.float32(1e-8),
    )
    return np.asarray(filters * normalization[:, None], dtype=np.float32)


class NumpyCausalFeatureExtractor:
    """Numerically aligned frontend without importing PyTorch."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()
        self.config.validate()
        indices = np.arange(self.config.frame_samples, dtype=np.float32)
        self.window = np.float32(0.5) - np.float32(0.5) * np.cos(
            np.float32(2 * np.pi) * indices / np.float32(self.config.frame_samples)
        )
        self.mel_filterbank = _build_mel_filterbank(self.config)

    def from_frame(self, frame: object) -> np.ndarray:
        samples = np.asarray(frame, dtype=np.float32)
        if samples.shape != (self.config.frame_samples,):
            raise ValueError(
                f"expected {self.config.frame_samples} samples per frame, "
                f"got {samples.shape}"
            )
        spectrum = np.fft.rfft(samples * self.window, n=self.config.n_fft)
        power = np.asarray(
            spectrum.real * spectrum.real + spectrum.imag * spectrum.imag,
            dtype=np.float32,
        )
        mel_energy = self.mel_filterbank @ power
        log_mel = np.log(np.maximum(mel_energy, np.float32(1e-8)))
        log_mel -= log_mel.mean(dtype=np.float32)

        rms = np.sqrt(
            np.maximum(
                np.mean(samples * samples, dtype=np.float32),
                np.float32(1e-10),
            )
        )
        geometric = np.exp(
            np.mean(
                np.log(np.maximum(power, np.float32(1e-10))),
                dtype=np.float32,
            )
        )
        arithmetic = np.maximum(
            np.mean(power, dtype=np.float32),
            np.float32(1e-10),
        )
        flatness = np.log(
            np.maximum(geometric / arithmetic, np.float32(1e-10))
        )
        zcr = np.mean(samples[1:] * samples[:-1] < 0, dtype=np.float32)
        return np.concatenate(
            (
                log_mel,
                np.asarray([np.log(rms), flatness, zcr], dtype=np.float32),
            )
        ).astype(np.float32, copy=False)


class NumpyStreamingFeatureExtractor:
    """Arbitrary-chunk streaming wrapper for the NumPy frontend."""

    def __init__(self, frontend: NumpyCausalFeatureExtractor) -> None:
        self.frontend = frontend
        self.reset()

    def reset(self) -> None:
        history_size = (
            self.frontend.config.frame_samples - self.frontend.config.hop_samples
        )
        self.history = np.zeros(history_size, dtype=np.float32)
        self.pending = np.empty(0, dtype=np.float32)

    def push(self, audio: object) -> np.ndarray:
        try:
            incoming = np.asarray(audio, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError("audio must be a one-dimensional numeric sequence") from exc
        if incoming.ndim != 1:
            raise ValueError("audio must be one-dimensional")
        if incoming.size and not np.isfinite(incoming).all():
            raise ValueError("audio samples must be finite")
        pending = np.concatenate((self.pending, incoming))
        outputs: list[np.ndarray] = []
        hop = self.frontend.config.hop_samples
        offset = 0
        while pending.size - offset >= hop:
            current_hop = pending[offset : offset + hop]
            frame = np.concatenate((self.history, current_hop))
            outputs.append(self.frontend.from_frame(frame))
            self.history = frame[-self.history.size :].copy()
            offset += hop
        self.pending = pending[offset:].copy()
        if not outputs:
            return np.empty(
                (0, self.frontend.config.feature_dim),
                dtype=np.float32,
            )
        return np.stack(outputs)


__all__ = ["NumpyCausalFeatureExtractor", "NumpyStreamingFeatureExtractor"]
