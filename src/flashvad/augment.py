from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .audio import telephone_roundtrip


@dataclass(frozen=True)
class AugmentConfig:
    gain_db_min: float = -12.0
    gain_db_max: float = 6.0
    noise_snr_db_min: float = -8.0
    noise_snr_db_max: float = 25.0
    telephone_probability: float = 0.35
    polarity_probability: float = 0.5
    clipping_probability: float = 0.15


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if noise.size < signal.size:
        repeats = int(np.ceil(signal.size / max(noise.size, 1)))
        noise = np.tile(noise, repeats)
    noise = noise[: signal.size]
    signal_rms = np.sqrt(np.mean(signal * signal) + 1e-10)
    noise_rms = np.sqrt(np.mean(noise * noise) + 1e-10)
    scale = signal_rms / (10 ** (snr_db / 20.0) * noise_rms + 1e-10)
    return np.clip(signal + noise * scale, -1.0, 1.0).astype(np.float32)


def augment_audio(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    config: AugmentConfig | None = None,
    noise: np.ndarray | None = None,
) -> np.ndarray:
    cfg = config or AugmentConfig()
    gain_db = rng.uniform(cfg.gain_db_min, cfg.gain_db_max)
    output = audio * (10 ** (gain_db / 20.0))
    if rng.random() < cfg.polarity_probability:
        output = -output
    if rng.random() < cfg.telephone_probability:
        codec = "mu-law" if rng.random() < 0.5 else "a-law"
        output = telephone_roundtrip(output, sample_rate, codec=codec)
    if rng.random() < cfg.clipping_probability:
        limit = float(rng.uniform(0.25, 0.9))
        output = np.clip(output, -limit, limit) / limit
    if noise is not None:
        snr_db = rng.uniform(cfg.noise_snr_db_min, cfg.noise_snr_db_max)
        output = mix_at_snr(output, noise, float(snr_db))
    return np.clip(output, -1.0, 1.0).astype(np.float32)
