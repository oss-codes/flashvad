#!/usr/bin/env python3
"""Attach aligned Silero soft labels to a training/development manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from flashvad.audio import read_audio
from flashvad.teacher import SileroOnnxTeacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--probabilities-dir", required=True, type=Path)
    parser.add_argument("--teacher-weight", type=float, default=0.7)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    args = parser.parse_args()
    if not 0.0 <= args.teacher_weight <= 1.0:
        parser.error("--teacher-weight must be between zero and one")

    input_manifest = args.input_manifest.resolve()
    output_manifest = args.output_manifest.resolve()
    probabilities_dir = args.probabilities_dir.resolve()
    probabilities_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    teacher = SileroOnnxTeacher(args.model.resolve())

    records: list[dict[str, object]] = []
    exact_negative_items = 0
    with input_manifest.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            audio_path = Path(str(record["audio"]))
            if not audio_path.is_absolute():
                audio_path = (input_manifest.parent / audio_path).resolve()
            if not audio_path.exists():
                raise FileNotFoundError(f"manifest line {line_number}: {audio_path}")
            if str(record.get("condition", "")).endswith("hard-negative"):
                exact_negative_items += 1
                records.append(record)
                continue
            probabilities = teacher.predict(
                read_audio(audio_path, args.sample_rate),
                sample_rate=args.sample_rate,
            )
            identity = hashlib.sha256(str(audio_path).encode()).hexdigest()[:16]
            destination = probabilities_dir / f"{audio_path.stem}-{identity}.npy"
            np.save(destination, probabilities)
            record["teacher_probabilities"] = os.path.relpath(
                destination,
                output_manifest.parent,
            )
            record["teacher_weight"] = args.teacher_weight
            records.append(record)

    output_manifest.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    provenance = {
        "teacher": "Silero VAD ONNX",
        "teacher_model": str(args.model.resolve()),
        "teacher_model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "input_manifest": str(input_manifest),
        "teacher_weight": args.teacher_weight,
        "items": len(records),
        "teacher_labelled_items": len(records) - exact_negative_items,
        "exact_negative_items": exact_negative_items,
        "note": "Training/development supervision only. Never generated for frozen tests.",
    }
    output_manifest.with_suffix(".teacher.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    main()
