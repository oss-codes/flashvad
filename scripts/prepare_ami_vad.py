#!/usr/bin/env python3
"""Prepare speaker-disjoint AMI meeting clips with Silero soft supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import time
import urllib.parse
import urllib.request
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import soundfile as sf

from flashvad.audio import read_audio
from flashvad.teacher import SileroOnnxTeacher

MIRROR = "FluidInference/ami-corpus-mirror"
MIRROR_PAGE = f"https://huggingface.co/datasets/{MIRROR}"
MIRROR_API = f"https://huggingface.co/api/datasets/{MIRROR}"
CANONICAL_PAGE = "https://groups.inf.ed.ac.uk/ami/corpus/"
SAMPLE_RATE = 16_000
SYSTEM_CA_BUNDLE = Path("/etc/ssl/cert.pem")
MEETINGS = {
    "train": (
        "sdm/IS1009a.Mix-Headset.wav",
        "sdm/ES2004a.Mix-Headset.wav",
    ),
    "valid": ("sdm/TS3003a.Mix-Headset.wav",),
}


def _request_bytes(url: str, attempts: int = 6) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "flashvad-data/0.1"})
    context = ssl.create_default_context(
        cafile=str(SYSTEM_CA_BUNDLE) if SYSTEM_CA_BUNDLE.exists() else None
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=180, context=context) as response:
                return response.read()
        except HTTPError as exc:
            if attempt + 1 == attempts:
                raise
            retry_after = float(exc.headers.get("Retry-After", 0) or 0)
            time.sleep(min(45.0, max(retry_after, 2.0**attempt)))
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(min(45.0, 2.0**attempt))
    raise AssertionError("unreachable")


def _request_json(url: str) -> dict[str, object]:
    return json.loads(_request_bytes(url))


def _source_path(output: Path, meeting: str, revision: str) -> Path:
    destination = output / "sources" / Path(meeting).name
    if destination.exists():
        return destination
    url = (
        f"{MIRROR_PAGE}/resolve/{revision}/"
        f"{urllib.parse.quote(meeting, safe='/')}"
    )
    payload = _request_bytes(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".wav.part")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return destination


def _clip_starts(
    audio_samples: int,
    clip_samples: int,
    clips_per_meeting: int,
) -> np.ndarray:
    maximum_start = max(0, audio_samples - clip_samples)
    count = min(clips_per_meeting, max(1, audio_samples // clip_samples))
    return np.linspace(0, maximum_start, count, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/ami-vad"))
    parser.add_argument("--clips-per-meeting", type=int, default=96)
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    args = parser.parse_args()
    if not 1 <= args.clips_per_meeting <= 500:
        parser.error("--clips-per-meeting must be between 1 and 500")
    if args.clip_seconds <= 0:
        parser.error("--clip-seconds must be positive")
    if not 0.0 <= args.teacher_weight <= 1.0:
        parser.error("--teacher-weight must be between zero and one")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata = _request_json(MIRROR_API)
    revision = metadata.get("sha")
    if not isinstance(revision, str):
        raise ValueError("AMI mirror returned no revision")
    model_path = args.model.resolve()
    teacher = SileroOnnxTeacher(model_path)
    clip_samples = round(args.clip_seconds * SAMPLE_RATE)
    manifests: dict[str, list[dict[str, object]]] = {"train": [], "valid": []}
    sources: list[dict[str, object]] = []

    for split, meetings in MEETINGS.items():
        for meeting in meetings:
            source = _source_path(output, meeting, revision)
            audio = read_audio(source, SAMPLE_RATE)
            starts = _clip_starts(
                audio.size,
                clip_samples,
                args.clips_per_meeting,
            )
            meeting_id = source.name.removesuffix(".Mix-Headset.wav")
            for index, start in enumerate(starts):
                clip = audio[int(start) : int(start) + clip_samples]
                if clip.size < clip_samples:
                    clip = np.pad(clip, (0, clip_samples - clip.size))
                clip_destination = (
                    output / "clips" / split / f"{meeting_id}-{index:03d}.wav"
                )
                probability_destination = (
                    output / "teachers" / split / f"{meeting_id}-{index:03d}.npy"
                )
                clip_destination.parent.mkdir(parents=True, exist_ok=True)
                probability_destination.parent.mkdir(parents=True, exist_ok=True)
                sf.write(
                    clip_destination,
                    clip,
                    SAMPLE_RATE,
                    subtype="PCM_16",
                )
                probabilities = teacher.predict(
                    clip,
                    sample_rate=SAMPLE_RATE,
                )
                np.save(probability_destination, probabilities)
                manifests[split].append(
                    {
                        "audio": os.path.relpath(clip_destination, output),
                        "sample_rate": SAMPLE_RATE,
                        "language": "en",
                        "domain": "ami-meeting",
                        "channel": "mixed-headset",
                        "condition": "real-meeting-teacher-labelled",
                        "teacher_probabilities": os.path.relpath(
                            probability_destination,
                            output,
                        ),
                        "teacher_weight": args.teacher_weight,
                        "teacher_confidence_weighting": False,
                        "segments": [],
                    }
                )
            sources.append(
                {
                    "meeting": meeting,
                    "partition": split,
                    "local_source": str(source.relative_to(output)),
                    "clips": len(starts),
                    "duration_seconds": audio.size / SAMPLE_RATE,
                }
            )

    for split, records in manifests.items():
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
            encoding="utf-8",
        )
    provenance = {
        "dataset": "AMI Meeting Corpus",
        "canonical_page": CANONICAL_PAGE,
        "license": "CC-BY-4.0",
        "download_mirror": MIRROR_PAGE,
        "mirror_revision": revision,
        "teacher": "Silero VAD ONNX",
        "teacher_model": str(model_path),
        "teacher_model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "teacher_weight": args.teacher_weight,
        "clip_seconds": args.clip_seconds,
        "purpose": (
            "Real conversational teacher-student training and validation; "
            "not a release benchmark."
        ),
        "split_policy": "Meeting families are disjoint across train and validation.",
        "sources": sources,
    }
    (output / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "train": len(manifests["train"]),
                "valid": len(manifests["valid"]),
                "revision": revision,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
