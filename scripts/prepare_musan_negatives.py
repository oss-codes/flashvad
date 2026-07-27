#!/usr/bin/env python3
"""Prepare a small, attributed MUSAN real-noise hard-negative corpus."""

from __future__ import annotations

import argparse
import io
import json
import math
import ssl
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from flashvad.data_prep import fixed_negative_excerpt

MIRROR = "FluidInference/musan"
MIRROR_PAGE = f"https://huggingface.co/datasets/{MIRROR}"
MIRROR_API = f"https://huggingface.co/api/datasets/{MIRROR}"
CANONICAL_PAGE = "https://www.openslr.org/17/"
PAPER = "https://arxiv.org/abs/1510.08484"
SAMPLE_RATE = 16_000
SYSTEM_CA_BUNDLE = Path("/etc/ssl/cert.pem")
NOISE_DIRECTORIES = ("noise/free-sound", "noise/sound-bible")


def _request_bytes(url: str, attempts: int = 6) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "flashvad-data/0.1"})
    context = ssl.create_default_context(
        cafile=str(SYSTEM_CA_BUNDLE) if SYSTEM_CA_BUNDLE.exists() else None
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=60, context=context) as response:
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


def _request_json(url: str) -> object:
    return json.loads(_request_bytes(url))


def _available_noise_files() -> tuple[str, list[dict[str, object]]]:
    metadata = _request_json(MIRROR_API)
    if not isinstance(metadata, dict) or not isinstance(metadata.get("sha"), str):
        raise ValueError("MUSAN mirror returned invalid metadata")
    revision = metadata["sha"]
    files: list[dict[str, object]] = []
    for directory in NOISE_DIRECTORIES:
        path = urllib.parse.quote(directory, safe="/")
        listing = _request_json(
            f"{MIRROR_API}/tree/main/{path}?recursive=true&limit=1000"
        )
        if not isinstance(listing, list):
            raise ValueError(f"MUSAN mirror returned invalid tree for {directory}")
        files.extend(
            item
            for item in listing
            if isinstance(item, dict)
            and str(item.get("path", "")).endswith(".wav")
            and 3_200 <= int(item.get("size", 0)) <= 4_000_000
        )
    return revision, files


def _decode_audio(payload: bytes) -> np.ndarray:
    audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        divisor = math.gcd(sample_rate, SAMPLE_RATE)
        mono = resample_poly(
            mono,
            SAMPLE_RATE // divisor,
            sample_rate // divisor,
        )
    return np.asarray(mono, dtype=np.float32)


def _prepare_file(
    source: dict[str, object],
    revision: str,
    output: Path,
    seed: int,
    split: str,
) -> dict[str, object]:
    source_path = str(source["path"])
    source_url = (
        f"{MIRROR_PAGE}/resolve/{revision}/"
        f"{urllib.parse.quote(source_path, safe='/')}"
    )
    audio = _decode_audio(_request_bytes(source_url))
    identity = sum((index + 1) * ord(character) for index, character in enumerate(source_path))
    excerpt = fixed_negative_excerpt(
        audio,
        SAMPLE_RATE,
        np.random.default_rng(np.random.SeedSequence((seed, identity))),
    )
    stem = Path(source_path).stem
    destination = output / "clips" / split / f"{stem}.wav"
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, excerpt, SAMPLE_RATE, subtype="PCM_16")
    return {
        "manifest": {
            "audio": str(destination.relative_to(output)),
            "sample_rate": SAMPLE_RATE,
            "language": "zxx",
            "domain": "musan",
            "channel": "wideband",
            "condition": "real-noise-hard-negative",
            "segments": [],
        },
        "source": {
            "path": source_path,
            "size_bytes": int(source["size"]),
            "derived_clip": str(destination.relative_to(output)),
            "partition": split,
        },
        "split": split,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/musan-negatives"))
    parser.add_argument("--items", type=int, default=64)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260726)
    args = parser.parse_args()
    if not 1 <= args.items <= 200:
        parser.error("--items must be between 1 and 200")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    revision, candidates = _available_noise_files()
    if len(candidates) < args.items:
        raise ValueError(f"MUSAN mirror exposes only {len(candidates)} suitable noise files")
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(candidates), args.items, replace=False)
    selected = [candidates[int(index)] for index in indices]
    valid_count = max(1, round(len(selected) * 0.2))
    assignments = [
        (source, "valid" if index < valid_count else "train")
        for index, source in enumerate(selected)
    ]

    prepared: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _prepare_file,
                source,
                revision,
                output,
                args.seed,
                split,
            )
            for source, split in assignments
        ]
        for future in as_completed(futures):
            prepared.append(future.result())

    for split in ("train", "valid"):
        records = sorted(
            (item for item in prepared if item["split"] == split),
            key=lambda item: str(item["manifest"]["audio"]),
        )
        (output / f"{split}.jsonl").write_text(
            "".join(
                json.dumps(item["manifest"], separators=(",", ":")) + "\n"
                for item in records
            ),
            encoding="utf-8",
        )

    provenance = {
        "dataset": "MUSAN",
        "canonical_page": CANONICAL_PAGE,
        "paper": PAPER,
        "license": "CC-BY-4.0",
        "download_mirror": MIRROR_PAGE,
        "mirror_revision": revision,
        "seed": args.seed,
        "items": args.items,
        "purpose": "Real non-speech hard negatives; not an accuracy benchmark.",
        "sources": sorted(
            (item["source"] for item in prepared),
            key=lambda source: str(source["path"]),
        ),
    }
    (output / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "train": sum(item["split"] == "train" for item in prepared),
                "valid": sum(item["split"] == "valid" for item in prepared),
                "revision": revision,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
