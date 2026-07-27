from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def read_audio(path: str | Path, target_sample_rate: int) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sample_rate != target_sample_rate:
        divisor = math.gcd(sample_rate, target_sample_rate)
        mono = resample_poly(mono, target_sample_rate // divisor, sample_rate // divisor)
    return np.asarray(mono, dtype=np.float32)


def peak_normalize(audio: np.ndarray, peak: float = 0.98) -> np.ndarray:
    maximum = float(np.max(np.abs(audio))) if audio.size else 0.0
    if maximum <= 1e-8:
        return audio.astype(np.float32, copy=True)
    return (audio * (peak / maximum)).astype(np.float32)


def mu_law_roundtrip(audio: np.ndarray, levels: int = 256) -> np.ndarray:
    mu = float(levels - 1)
    clipped = np.clip(audio, -1.0, 1.0)
    encoded = np.sign(clipped) * np.log1p(mu * np.abs(clipped)) / np.log1p(mu)
    quantized = np.round((encoded + 1.0) * 0.5 * mu) / mu * 2.0 - 1.0
    decoded = np.sign(quantized) * np.expm1(np.abs(quantized) * np.log1p(mu)) / mu
    return decoded.astype(np.float32)


def a_law_roundtrip(audio: np.ndarray, levels: int = 256) -> np.ndarray:
    """Approximate a G.711 A-law encode/quantize/decode round trip."""
    a = 87.6
    clipped = np.clip(audio, -1.0, 1.0)
    magnitude = np.abs(clipped)
    denominator = 1.0 + np.log(a)
    encoded_magnitude = np.where(
        magnitude < 1.0 / a,
        a * magnitude / denominator,
        (1.0 + np.log(a * np.maximum(magnitude, 1.0 / a))) / denominator,
    )
    encoded = np.sign(clipped) * encoded_magnitude
    quantized = np.round((encoded + 1.0) * 0.5 * (levels - 1))
    quantized = quantized / (levels - 1) * 2.0 - 1.0
    quantized_magnitude = np.abs(quantized)
    threshold = 1.0 / denominator
    decoded_magnitude = np.where(
        quantized_magnitude < threshold,
        quantized_magnitude * denominator / a,
        np.exp(quantized_magnitude * denominator - 1.0) / a,
    )
    return (np.sign(quantized) * decoded_magnitude).astype(np.float32)


def telephone_roundtrip(
    audio: np.ndarray,
    sample_rate: int,
    codec: str = "mu-law",
) -> np.ndarray:
    if codec not in {"mu-law", "a-law"}:
        raise ValueError("codec must be 'mu-law' or 'a-law'")
    roundtrip = mu_law_roundtrip if codec == "mu-law" else a_law_roundtrip
    if sample_rate == 8_000:
        return roundtrip(audio)
    down = resample_poly(audio, 1, 2)
    coded = roundtrip(down)
    restored = resample_poly(coded, 2, 1)
    if restored.size < audio.size:
        restored = np.pad(restored, (0, audio.size - restored.size))
    return restored[: audio.size].astype(np.float32)
