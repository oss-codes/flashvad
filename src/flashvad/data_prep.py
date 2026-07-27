from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter


@dataclass(frozen=True)
class PreparedCall:
    audio: np.ndarray
    segments: tuple[tuple[float, float], ...]


def _frame_log_rms(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
) -> tuple[np.ndarray, int, int]:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        return np.empty(0, dtype=np.float32), 0, 0

    frame_samples = max(1, round(sample_rate * frame_ms / 1_000))
    hop_samples = max(1, round(sample_rate * hop_ms / 1_000))
    if samples.size < frame_samples:
        samples = np.pad(samples, (0, frame_samples - samples.size))
    frame_count = 1 + int(np.ceil((samples.size - frame_samples) / hop_samples))
    required = (frame_count - 1) * hop_samples + frame_samples
    padded = np.pad(samples, (0, max(0, required - samples.size)))
    frames = np.lib.stride_tricks.sliding_window_view(padded, frame_samples)[::hop_samples]
    log_rms = np.log(np.sqrt(np.mean(frames * frames, axis=1) + 1e-10))
    return log_rms, frame_samples, hop_samples


def _runs(values: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(values, dtype=np.int8), (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def speech_regions(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    merge_gap_ms: float = 80.0,
    min_speech_ms: float = 80.0,
    pad_ms: float = 30.0,
) -> tuple[tuple[int, int], ...]:
    """Return weak, energy-derived speech regions while preserving real pauses."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    log_rms, frame_samples, hop_samples = _frame_log_rms(
        samples,
        sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
    )
    if log_rms.size == 0:
        return ()

    floor = float(np.percentile(log_rms, 10))
    speech_level = float(np.percentile(log_rms, 90))
    dynamic_range = speech_level - floor
    if dynamic_range < 0.25:
        if float(np.sqrt(np.mean(samples * samples) + 1e-10)) <= 1e-4:
            return ()
        return ((0, samples.size),)

    threshold = floor + max(0.4, 0.28 * dynamic_range)
    active = log_rms >= threshold
    maximum_gap_frames = round(merge_gap_ms / hop_ms)
    for start, end in _runs(~active):
        if (
            start > 0
            and end < active.size
            and end - start <= maximum_gap_frames
        ):
            active[start:end] = True

    minimum_frames = max(1, round(min_speech_ms / hop_ms))
    for start, end in _runs(active):
        if end - start < minimum_frames:
            active[start:end] = False

    pad_samples = round(sample_rate * pad_ms / 1_000)
    regions: list[tuple[int, int]] = []
    for start_frame, end_frame in _runs(active):
        start = max(0, start_frame * hop_samples - pad_samples)
        end = min(
            samples.size,
            (end_frame - 1) * hop_samples + frame_samples + pad_samples,
        )
        if regions and start <= regions[-1][1]:
            regions[-1] = (regions[-1][0], max(regions[-1][1], end))
        else:
            regions.append((start, end))
    return tuple(regions)


def trim_speech(
    audio: np.ndarray,
    sample_rate: int,
    *,
    frame_ms: float = 20.0,
    hop_ms: float = 10.0,
    pad_ms: float = 50.0,
) -> np.ndarray:
    """Remove leading/trailing room tone without segmenting pauses inside speech."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    regions = speech_regions(
        samples,
        sample_rate,
        frame_ms=frame_ms,
        hop_ms=hop_ms,
        pad_ms=pad_ms,
    )
    if not regions:
        return samples.copy()
    return samples[regions[0][0] : regions[-1][1]].copy()


def _background_noise(
    rng: np.random.Generator,
    samples: int,
    sample_rate: int,
) -> np.ndarray:
    white = rng.normal(0.0, 1.0, samples).astype(np.float32)
    colored = lfilter([1.0], [1.0, -0.92], white).astype(np.float32)
    colored /= np.sqrt(np.mean(colored * colored) + 1e-10)
    frequency = float(rng.choice((50.0, 60.0, 100.0, 120.0)))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    timeline = np.arange(samples, dtype=np.float32) / sample_rate
    hum = np.sin(2.0 * np.pi * frequency * timeline + phase).astype(np.float32)
    tonal_frequency = float(rng.uniform(180.0, 2_400.0))
    modulation = 0.6 + 0.4 * np.sin(
        2.0 * np.pi * float(rng.uniform(0.3, 4.0)) * timeline
    )
    tonal = (
        modulation
        * (
            np.sin(2.0 * np.pi * tonal_frequency * timeline + phase)
            + 0.35 * np.sin(4.0 * np.pi * tonal_frequency * timeline + phase)
        )
    ).astype(np.float32)
    impulses = np.zeros(samples, dtype=np.float32)
    impulse_count = max(1, round(samples / sample_rate * rng.uniform(1.0, 8.0)))
    impulse_locations = rng.integers(0, samples, impulse_count)
    impulses[impulse_locations] = rng.uniform(-1.0, 1.0, impulse_count)
    impulses = lfilter([1.0, 0.6, 0.3], [1.0, -0.75], impulses).astype(np.float32)
    weights = rng.dirichlet((3.0, 1.0, 1.0, 1.0)).astype(np.float32)
    noise = (
        weights[0] * colored
        + weights[1] * hum
        + weights[2] * tonal
        + weights[3] * impulses
    )
    noise /= np.sqrt(np.mean(noise * noise) + 1e-10)
    return noise.astype(np.float32)


def fixed_negative_excerpt(
    audio: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    duration_seconds: float = 4.0,
) -> np.ndarray:
    """Crop or repeat an external non-speech recording at a realistic call level."""
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    target_samples = round(duration_seconds * sample_rate)
    if target_samples < 1:
        raise ValueError("duration and sample rate must produce at least one sample")
    if samples.size == 0:
        raise ValueError("negative audio cannot be empty")
    if samples.size < target_samples:
        samples = np.tile(samples, int(np.ceil(target_samples / samples.size)))
    maximum_start = samples.size - target_samples
    start = int(rng.integers(0, maximum_start + 1)) if maximum_start else 0
    excerpt = samples[start : start + target_samples].copy()
    source_rms = float(np.sqrt(np.mean(excerpt * excerpt) + 1e-10))
    if source_rms <= 1e-4:
        return np.zeros(target_samples, dtype=np.float32)
    target_rms = 10.0 ** (float(rng.uniform(-24.0, -14.0)) / 20.0)
    return np.clip(excerpt * (target_rms / source_rms), -0.98, 0.98).astype(
        np.float32
    )


def compose_call(
    speech: np.ndarray,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    duration_seconds: float = 4.0,
    hard_negative: bool = False,
) -> PreparedCall:
    """Create a fixed call-like clip from real speech with exact inserted boundaries."""
    total_samples = round(duration_seconds * sample_rate)
    output = np.zeros(total_samples, dtype=np.float32)
    noise = _background_noise(rng, total_samples, sample_rate)
    level_range = (-28.0, -12.0) if hard_negative else (-50.0, -26.0)
    noise_level = 10.0 ** (float(rng.uniform(*level_range)) / 20.0)
    output += noise * noise_level
    if hard_negative:
        return PreparedCall(np.clip(output, -0.98, 0.98), ())

    source = np.asarray(speech, dtype=np.float32).reshape(-1)
    regions = speech_regions(source, sample_rate)
    if not regions:
        return PreparedCall(np.clip(output, -0.98, 0.98), ())
    voiced = np.concatenate([source[start:end] for start, end in regions])
    source_rms = np.sqrt(np.mean(voiced * voiced) + 1e-10)
    target_rms = 10.0 ** (float(rng.uniform(-25.0, -14.0)) / 20.0)
    source = np.clip(source * (target_rms / source_rms), -0.95, 0.95)

    segments: list[tuple[float, float]] = []
    cursor = round(float(rng.uniform(0.15, 0.55)) * sample_rate)
    region_index = int(rng.integers(0, len(regions)))
    while cursor < total_samples - round(0.25 * sample_rate):
        region_start, region_end = regions[region_index % len(regions)]
        region_index += 1
        region_length = region_end - region_start
        if rng.random() < 0.3:
            desired = round(float(rng.uniform(0.08, 0.3)) * sample_rate)
        else:
            desired = round(float(rng.uniform(0.3, 1.2)) * sample_rate)
        available = total_samples - cursor - round(0.12 * sample_rate)
        length = min(desired, region_length, available)
        if length < round(0.06 * sample_rate):
            break

        maximum_offset = max(0, region_length - length)
        offset = int(rng.integers(0, maximum_offset + 1)) if maximum_offset else 0
        excerpt = source[
            region_start + offset : region_start + offset + length
        ].copy()
        fade = min(round(0.01 * sample_rate), length // 2)
        if fade:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            excerpt[:fade] *= ramp
            excerpt[-fade:] *= ramp[::-1]
        output[cursor : cursor + length] += excerpt
        segments.append((cursor / sample_rate, (cursor + length) / sample_rate))
        cursor += length + round(float(rng.uniform(0.16, 0.58)) * sample_rate)

    return PreparedCall(
        np.clip(output, -0.98, 0.98).astype(np.float32),
        tuple(segments),
    )
