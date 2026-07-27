#!/usr/bin/env python3
"""Build a small, attributed India/GCC VAD corpus from streamed FLEURS clips.

FLEURS is speech-only, so this script creates call-like mixtures with deterministic
speech insertion boundaries. These are weak training labels, not a release benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import ssl
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice
from pathlib import Path
from urllib.error import HTTPError

import numpy as np
import soundfile as sf

from flashvad.audio import read_audio
from flashvad.data_prep import compose_call

DATASET = "google/fleurs"
DATASET_PAGE = "https://huggingface.co/datasets/google/fleurs"
DATASET_API = "https://huggingface.co/api/datasets/google/fleurs"
ROWS_API = "https://datasets-server.huggingface.co/rows"
SAMPLE_RATE = 16_000
DEFAULT_LANGUAGES = (
    "ar_eg",
    "bn_in",
    "en_us",
    "gu_in",
    "hi_in",
    "kn_in",
    "ml_in",
    "mr_in",
    "pa_in",
    "ta_in",
    "te_in",
    "ur_pk",
)
SYSTEM_CA_BUNDLE = Path("/etc/ssl/cert.pem")


def _seed_component(value: str) -> int:
    return int.from_bytes(
        hashlib.sha256(value.encode("utf-8")).digest()[:8],
        "little",
    )


def _request_bytes(
    url: str,
    attempts: int = 6,
    timeout_seconds: float = 180,
) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "flashvad-data/0.1"})
    context = ssl.create_default_context(
        cafile=str(SYSTEM_CA_BUNDLE) if SYSTEM_CA_BUNDLE.exists() else None
    )
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout_seconds,
                context=context,
            ) as response:
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


def _request_json(
    url: str,
    attempts: int = 6,
    timeout_seconds: float = 180,
) -> dict[str, object]:
    return json.loads(_request_bytes(url, attempts, timeout_seconds))


def _rows(
    language: str,
    split: str,
    count: int,
    revision: str,
) -> list[dict[str, object]]:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": language,
            "split": split,
            "offset": 0,
            "length": count,
        }
    )
    payload = _request_json(
        f"{ROWS_API}?{query}",
        attempts=3,
        timeout_seconds=30,
    )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"FLEURS returned no {split} rows for {language}")
    revision_marker = f"/--/{revision}/--/"
    if any(revision_marker not in _audio_url(row) for row in rows):
        raise ValueError(
            "FLEURS rows API is not serving the requested dataset revision; "
            "retry with --backend datasets"
        )
    return rows


def _cached_rows(
    language: str,
    split: str,
    count: int,
    metadata: Path,
    revision: str,
) -> list[dict[str, object]]:
    destination = metadata / (
        f"{language}-{split}-{count}-{revision[:12]}-rows.json"
    )
    if destination.exists():
        cached = json.loads(destination.read_text(encoding="utf-8"))
        if (
            isinstance(cached, dict)
            and cached.get("dataset_revision") == revision
            and isinstance(cached.get("rows"), list)
            and cached["rows"]
        ):
            return cached["rows"]
    rows = _rows(language, split, count, revision)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"dataset_revision": revision, "rows": rows}),
        encoding="utf-8",
    )
    return rows


def _audio_url(row: dict[str, object]) -> str:
    values = row["row"]
    if not isinstance(values, dict):
        raise ValueError("invalid FLEURS row")
    audio = values["audio"]
    if not isinstance(audio, list) or not audio or not isinstance(audio[0], dict):
        raise ValueError("FLEURS row has no decoded audio asset")
    source = audio[0].get("src")
    if not isinstance(source, str):
        raise ValueError("FLEURS audio asset has no URL")
    return source


def _download_row(
    language: str,
    split: str,
    row: dict[str, object],
    sources: Path,
    revision: str,
) -> tuple[str, str, int, Path]:
    row_index = int(row["row_idx"])
    destination = (
        sources
        / revision[:12]
        / split
        / language
        / f"{row_index:05d}.wav"
    )
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".wav.part")
        temporary.write_bytes(_request_bytes(_audio_url(row)))
        temporary.replace(destination)
    return language, split, row_index, destination


def _streamed_sources(
    language: str,
    split: str,
    count: int,
    metadata: Path,
    sources: Path,
    output: Path,
    revision: str,
) -> list[tuple[str, str, int, Path]]:
    cache = metadata / (
        f"{language}-{split}-{count}-{revision[:12]}-datasets.json"
    )
    if cache.exists():
        cached = json.loads(cache.read_text(encoding="utf-8"))
        if (
            isinstance(cached, dict)
            and cached.get("dataset_revision") == revision
            and isinstance(cached.get("items"), list)
        ):
            restored = [
                (
                    language,
                    split,
                    int(entry["row_index"]),
                    output / str(entry["local_source"]),
                )
                for entry in cached["items"]
            ]
            if (
                len(restored) == count
                and all(path.is_file() for _, _, _, path in restored)
            ):
                return restored

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "the direct FLEURS backend requires `uv sync --extra data`"
        ) from exc

    dataset = load_dataset(
        DATASET,
        language,
        split=split,
        streaming=True,
        revision=revision,
    )
    dataset = dataset.decode(False)
    fetched: list[tuple[str, str, int, Path]] = []
    entries: list[dict[str, object]] = []
    for position, row in enumerate(islice(dataset, count)):
        audio = row.get("audio")
        if not isinstance(audio, dict):
            raise ValueError("streamed FLEURS row has no raw audio mapping")
        raw_path = audio.get("path")
        payload = audio.get("bytes")
        if payload is None and isinstance(raw_path, str):
            if raw_path.startswith(("https://", "http://")):
                payload = _request_bytes(raw_path)
            else:
                local_path = Path(raw_path)
                if local_path.is_file():
                    payload = local_path.read_bytes()
        if not isinstance(payload, bytes):
            raise ValueError("streamed FLEURS row has no accessible audio bytes")

        row_index = position
        upstream_id = row.get("id")
        suffix = Path(str(raw_path or "")).suffix.lower()
        if suffix not in {".wav", ".flac", ".ogg"}:
            suffix = ".wav"
        destination = (
            sources
            / revision[:12]
            / split
            / language
            / f"{row_index:05d}{suffix}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(payload)
        temporary.replace(destination)
        fetched.append((language, split, row_index, destination))
        entries.append(
            {
                "row_index": row_index,
                "upstream_id": upstream_id,
                "upstream_path": str(raw_path or ""),
                "local_source": str(destination.relative_to(output)),
            }
        )
    if len(fetched) != count:
        raise ValueError(
            f"FLEURS returned {len(fetched)} of {count} requested {split} rows "
            f"for {language}"
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(
        json.dumps(
            {"dataset_revision": revision, "items": entries},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return fetched


def _rows_api_sources(
    language: str,
    split: str,
    count: int,
    metadata: Path,
    sources: Path,
    revision: str,
) -> list[tuple[str, str, int, Path]]:
    rows = _cached_rows(language, split, count, metadata, revision)
    return [
        _download_row(language, split, row, sources, revision)
        for row in rows
    ]


def _hybrid_sources(
    language: str,
    split: str,
    count: int,
    metadata: Path,
    sources: Path,
    output: Path,
    revision: str,
) -> list[tuple[str, str, int, Path]]:
    if split == "train":
        return _streamed_sources(
            language,
            split,
            count,
            metadata,
            sources,
            output,
            revision,
        )
    try:
        return _rows_api_sources(
            language,
            split,
            count,
            metadata,
            sources,
            revision,
        )
    except Exception as exc:
        print(
            f"rows API unavailable for {language}/{split}: "
            f"{type(exc).__name__}: {exc}; falling back to revision-pinned streaming",
            flush=True,
        )
        return _streamed_sources(
            language,
            split,
            count,
            metadata,
            sources,
            output,
            revision,
        )


def _manifest_record(
    path: Path,
    root: Path,
    language: str,
    segments: tuple[tuple[float, float], ...],
    *,
    condition: str,
) -> dict[str, object]:
    return {
        "audio": str(path.relative_to(root)),
        "sample_rate": SAMPLE_RATE,
        "language": language.replace("_", "-"),
        "domain": "fleurs-derived-call-simulation",
        "channel": "mixed-wideband-telephony-augmentation",
        "condition": condition,
        "segments": [
            {"start": start, "end": end, "label": "speech"}
            for start, end in segments
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/fleurs-vad"))
    parser.add_argument("--train-samples-per-language", type=int, default=64)
    parser.add_argument("--valid-samples-per-language", type=int, default=16)
    parser.add_argument("--languages", nargs="+", default=list(DEFAULT_LANGUAGES))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument(
        "--backend",
        choices=("hybrid", "datasets", "rows-api"),
        default="hybrid",
        help=(
            "hybrid uses revision-pinned streaming for train and the lighter "
            "rows API for validation"
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.train_samples_per_language <= 100:
        parser.error("--train-samples-per-language must be between 1 and 100")
    if not 1 <= args.valid_samples_per_language <= 100:
        parser.error("--valid-samples-per-language must be between 1 and 100")

    output = args.output.resolve()
    sources = output / "sources"
    metadata = output / "metadata"
    clips = output / "clips"
    output.mkdir(parents=True, exist_ok=True)
    dataset_metadata = _request_json(DATASET_API)
    card_data = dataset_metadata.get("cardData")
    raw_license = card_data.get("license") if isinstance(card_data, dict) else None
    license_ids = (
        [str(value).lower() for value in raw_license]
        if isinstance(raw_license, list)
        else [str(raw_license).lower()]
    )
    if license_ids != ["cc-by-4.0"]:
        raise ValueError(f"unexpected FLEURS license metadata: {raw_license!r}")
    license_id = license_ids[0]

    revision = dataset_metadata.get("sha")
    if not isinstance(revision, str) or not revision:
        raise ValueError("FLEURS dataset metadata has no revision")

    source_groups: list[list[tuple[str, str, int, Path]]] = []
    requested_splits = (
        ("train", args.train_samples_per_language),
        ("validation", args.valid_samples_per_language),
    )
    with ThreadPoolExecutor(
        max_workers=min(2, args.workers, len(args.languages))
    ) as executor:
        futures = {
            executor.submit(
                (
                    _streamed_sources
                    if args.backend == "datasets"
                    else _hybrid_sources
                    if args.backend == "hybrid"
                    else _rows_api_sources
                ),
                *(
                    (
                        language,
                        split,
                        count,
                        metadata,
                        sources,
                        output,
                        revision,
                    )
                    if args.backend in {"datasets", "hybrid"}
                    else (language, split, count, metadata, sources, revision)
                ),
            ): (language, split)
            for language in args.languages
            for split, count in requested_splits
        }
        for future in as_completed(futures):
            source_groups.append(future.result())
    fetched = [
        item
        for group in source_groups
        for item in group
    ]

    manifests: dict[str, list[dict[str, object]]] = {"train": [], "valid": []}
    attributions: list[dict[str, object]] = []
    for language, source_split, row_index, source in sorted(fetched):
        split = "train" if source_split == "train" else "valid"
        source_audio = read_audio(source, SAMPLE_RATE)
        seed_parts = (
            args.seed,
            _seed_component(language),
            _seed_component(source_split),
            row_index,
        )
        speech_clip = compose_call(
            source_audio,
            SAMPLE_RATE,
            np.random.default_rng(np.random.SeedSequence((*seed_parts, 0))),
        )
        destination = clips / split / f"{language}-{row_index:05d}-speech.wav"
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, speech_clip.audio, SAMPLE_RATE, subtype="PCM_16")
        manifests[split].append(
            _manifest_record(
                destination,
                output,
                language,
                speech_clip.segments,
                condition="weak-boundary-call-simulation",
            )
        )
        attributions.append(
            {
                "language": language,
                "source_split": source_split,
                "source_row": row_index,
                "local_source": str(source.relative_to(output)),
                "derived_clip": str(destination.relative_to(output)),
                "partition": split,
                "kind": "speech",
            }
        )
        if row_index % 4 == 3:
            negative_clip = compose_call(
                source_audio,
                SAMPLE_RATE,
                np.random.default_rng(np.random.SeedSequence((*seed_parts, 1))),
                hard_negative=True,
            )
            negative_destination = (
                clips / split / f"{language}-{row_index:05d}-negative.wav"
            )
            sf.write(
                negative_destination,
                negative_clip.audio,
                SAMPLE_RATE,
                subtype="PCM_16",
            )
            manifests[split].append(
                _manifest_record(
                    negative_destination,
                    output,
                    language,
                    (),
                    condition="synthetic-hard-negative",
                )
            )
            attributions.append(
                {
                    "language": language,
                    "source_split": source_split,
                    "source_row": row_index,
                    "local_source": str(source.relative_to(output)),
                    "derived_clip": str(negative_destination.relative_to(output)),
                    "partition": split,
                    "kind": "hard-negative",
                }
            )

    for split, records in manifests.items():
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
            encoding="utf-8",
        )

    source_backends = {
        language: {
            split: (
                "datasets"
                if any(
                    metadata.glob(
                        f"{language}-{split}-{count}-{revision[:12]}-datasets.json"
                    )
                )
                else "rows-api"
            )
            for split, count in requested_splits
        }
        for language in args.languages
    }
    provenance = {
        "dataset": DATASET,
        "dataset_page": DATASET_PAGE,
        "dataset_revision": dataset_metadata.get("sha"),
        "download_backend": args.backend,
        "source_backends": source_backends,
        "license": str(license_id),
        "source_splits": {
            "train": "train",
            "valid": "validation",
        },
        "purpose": "Weakly labelled training/development data; never a release benchmark.",
        "seed": args.seed,
        "seed_derivation": "sha256-language-split-v1",
        "train_samples_per_language": args.train_samples_per_language,
        "valid_samples_per_language": args.valid_samples_per_language,
        "languages": sorted(args.languages),
        "sources": attributions,
    }
    (output / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "train_items": len(manifests["train"]),
                "valid_items": len(manifests["valid"]),
                "languages": len(args.languages),
                "dataset_revision": provenance["dataset_revision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
