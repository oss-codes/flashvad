from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000
DURATION = 4.0
LANGUAGES = ("hi", "ta", "ar", "en", "ur", "ml")


def voiced_signal(rng: np.random.Generator, samples: int, fundamental: float) -> np.ndarray:
    time = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
    phase_jitter = np.cumsum(rng.normal(0.0, 0.002, samples)).astype(np.float32)
    signal = sum(
        (1.0 / harmonic) * np.sin(2 * np.pi * fundamental * harmonic * time + phase_jitter)
        for harmonic in range(1, 6)
    )
    envelope = (
        np.maximum(
            np.sin(np.linspace(0, np.pi, samples, dtype=np.float32)),
            0.0,
        )
        ** 0.4
    )
    return (0.25 * signal * envelope).astype(np.float32)


def make_item(index: int, destination: Path) -> dict[str, object]:
    rng = np.random.default_rng(10_000 + index)
    total = round(DURATION * SAMPLE_RATE)
    audio = rng.normal(0.0, 0.004 + 0.004 * (index % 3), total).astype(np.float32)
    segments: list[dict[str, object]] = []
    cursor = float(rng.uniform(0.25, 0.55))
    while cursor < DURATION - 0.5:
        length = float(rng.uniform(0.25, 0.9))
        end = min(cursor + length, DURATION - 0.15)
        start_sample = round(cursor * SAMPLE_RATE)
        end_sample = round(end * SAMPLE_RATE)
        fundamental = float(rng.uniform(85, 280))
        audio[start_sample:end_sample] += voiced_signal(rng, end_sample - start_sample, fundamental)
        segments.append({"start": cursor, "end": end, "label": "speech"})
        cursor = end + float(rng.uniform(0.15, 0.55))
    audio = np.clip(audio, -0.98, 0.98)
    wav_path = destination / f"sample-{index:03d}.wav"
    sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
    return {
        "audio": wav_path.name,
        "sample_rate": SAMPLE_RATE,
        "language": LANGUAGES[index % len(LANGUAGES)],
        "domain": "synthetic-smoke-only",
        "segments": segments,
    }


def main() -> None:
    destination = Path("data/smoke")
    destination.mkdir(parents=True, exist_ok=True)
    items = [make_item(index, destination) for index in range(48)]
    for name, subset in (("train", items[:40]), ("valid", items[40:])):
        with (destination / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for item in subset:
                handle.write(json.dumps(item, separators=(",", ":")) + "\n")
    print(f"created {len(items)} files in {destination}")


if __name__ == "__main__":
    main()
