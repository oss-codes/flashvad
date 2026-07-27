#!/usr/bin/env python3
"""Convert comma-separated segment labels into the FlashVAD manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import wave
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scv_speech_segments(path: Path) -> list[dict[str, float | str]]:
    fields = path.read_text(encoding="utf-8").strip().split(",")
    values = fields[1:]
    if len(values) % 3:
        raise ValueError(f"invalid SCV triplets: {path}")
    segments = []
    for start, end, label in zip(values[::3], values[1::3], values[2::3], strict=True):
        if int(label) == 1:
            segments.append(
                {
                    "start": float(start),
                    "end": float(end),
                    "label": "speech",
                }
            )
    return segments


def validate_wav(path: Path) -> dict[str, int]:
    with wave.open(str(path), "rb") as source:
        metadata = {
            "channels": source.getnchannels(),
            "sample_width_bytes": source.getsampwidth(),
            "sample_rate": source.getframerate(),
            "frames": source.getnframes(),
        }
    if metadata["channels"] != 1:
        raise ValueError(f"expected mono WAV: {path}")
    if metadata["sample_width_bytes"] != 2:
        raise ValueError(f"expected signed 16-bit PCM WAV: {path}")
    if metadata["sample_rate"] != 16_000:
        raise ValueError(f"expected 16 kHz WAV: {path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--language", default="und")
    parser.add_argument("--domain", default="public-benchmark")
    parser.add_argument("--condition", default="scv")
    parser.add_argument("--source-repository")
    parser.add_argument("--source-revision")
    parser.add_argument("--expected-items", type=int)
    parser.add_argument(
        "--relative-paths",
        action="store_true",
        help="write audio paths relative to the output manifest",
    )
    args = parser.parse_args()

    records = []
    sources = []
    for audio in sorted(args.input.glob("*.wav")):
        labels = audio.with_suffix(".scv")
        if not labels.exists():
            raise FileNotFoundError(labels)
        wav_metadata = validate_wav(audio)
        audio_path = (
            os.path.relpath(audio.resolve(), args.output.parent.resolve())
            if args.relative_paths
            else str(audio.resolve())
        )
        records.append(
            {
                "audio": audio_path,
                "sample_rate": 16_000,
                "language": args.language,
                "domain": args.domain,
                "condition": args.condition,
                "segments": scv_speech_segments(labels),
            }
        )
        sources.append(
            {
                "audio": audio.name,
                "audio_sha256": sha256(audio),
                "labels": labels.name,
                "labels_sha256": sha256(labels),
                **wav_metadata,
            }
        )
    if not records:
        raise ValueError(f"no WAV files found in {args.input}")
    if args.expected_items is not None and len(records) != args.expected_items:
        raise ValueError(
            f"expected {args.expected_items} WAV/SCV pairs, found {len(records)}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = "".join(json.dumps(record) + "\n" for record in records)
    args.output.write_text(serialized, encoding="utf-8")
    provenance = {
        "schema": "flashvad-scv-conversion-v1",
        "source_repository": args.source_repository,
        "source_revision": args.source_revision,
        "source_directory": str(args.input.resolve()),
        "items": len(records),
        "manifest_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "label_policy": "speech segments are SCV triplets whose integer label is 1",
        "sources": sources,
    }
    provenance_path = args.output.with_suffix(".provenance.json")
    provenance_path.write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
