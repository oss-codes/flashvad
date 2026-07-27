#!/usr/bin/env python3
"""Attach equal-weight Silero and FireRed soft labels to a VAD manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from flashvad.audio import read_audio
from flashvad.teacher import FireRedOnnxTeacher, SileroOnnxTeacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--silero-model", required=True, type=Path)
    parser.add_argument("--firered-model", required=True, type=Path)
    parser.add_argument("--firered-cmvn", required=True, type=Path)
    parser.add_argument("--input-manifest", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--probabilities-dir", required=True, type=Path)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    args = parser.parse_args()

    input_manifest = args.input_manifest.resolve()
    output_manifest = args.output_manifest.resolve()
    probabilities_dir = args.probabilities_dir.resolve()
    probabilities_dir.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    silero = SileroOnnxTeacher(args.silero_model.resolve())
    firered = FireRedOnnxTeacher(
        args.firered_model.resolve(),
        args.firered_cmvn.resolve(),
    )

    records: list[dict[str, object]] = []
    exact_negative_items = 0
    for line_number, line in enumerate(
        input_manifest.read_text(encoding="utf-8").splitlines(),
        1,
    ):
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
            record.pop("teacher_probabilities", None)
            record.pop("teacher_weight", None)
            records.append(record)
            continue
        audio = read_audio(audio_path, args.sample_rate)
        silero_probabilities = silero.predict(
            audio,
            sample_rate=args.sample_rate,
        )
        firered_probabilities = firered.predict(
            audio,
            sample_rate=args.sample_rate,
        )
        if silero_probabilities.shape != firered_probabilities.shape:
            raise ValueError(f"teacher frame mismatch for {audio_path}")
        probabilities = 0.5 * silero_probabilities + 0.5 * firered_probabilities
        identity = hashlib.sha256(str(audio_path).encode()).hexdigest()[:16]
        destination = probabilities_dir / f"{audio_path.stem}-{identity}.npy"
        np.save(destination, probabilities.astype(np.float32))
        record["teacher_probabilities"] = os.path.relpath(
            destination,
            output_manifest.parent,
        )
        records.append(record)

    output_manifest.write_text(
        "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
        encoding="utf-8",
    )
    provenance = {
        "teachers": [
            {
                "name": "Silero VAD ONNX",
                "model": str(args.silero_model.resolve()),
                "sha256": hashlib.sha256(args.silero_model.read_bytes()).hexdigest(),
                "weight": 0.5,
            },
            {
                "name": "FireRedVAD non-streaming ONNX",
                "model": str(args.firered_model.resolve()),
                "sha256": hashlib.sha256(args.firered_model.read_bytes()).hexdigest(),
                "cmvn": str(args.firered_cmvn.resolve()),
                "cmvn_sha256": hashlib.sha256(args.firered_cmvn.read_bytes()).hexdigest(),
                "weight": 0.5,
            },
        ],
        "input_manifest": str(input_manifest),
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
