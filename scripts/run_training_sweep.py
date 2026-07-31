#!/usr/bin/env python3
"""Run a leakage-safe multi-seed FlashVAD training sweep on one development split."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flashvad.checkpoint import load_checkpoint_data
from flashvad.config import ProjectConfig
from flashvad.manifest import validate_manifest_group_separation
from flashvad.train import select_training_device, train


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_tree_sha256(repository: Path) -> str:
    files = sorted((repository / "src" / "flashvad").rglob("*.py"))
    files.append(Path(__file__).resolve())
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(repository)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_json_safe(report), indent=2) + "\n",
        encoding="utf-8",
    )


def _training_input_digest(
    *,
    config_sha256: str,
    train_sha256: str,
    validation_sha256: str,
    noise_sha256: str | None,
    region_profile_sha256: str | None,
    runner_sha256: str,
    device: str,
    training_code_sha256: str,
    runtime_sha256: str,
    train_audio_fingerprint: str,
    validation_audio_fingerprint: str,
    noise_audio_fingerprint: str | None,
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "config": config_sha256,
                "train": train_sha256,
                "validation": validation_sha256,
                "noise": noise_sha256,
                "region_profile": region_profile_sha256,
                "runner": runner_sha256,
                "device": device,
                "training_code": training_code_sha256,
                "runtime": runtime_sha256,
                "train_audio": train_audio_fingerprint,
                "validation_audio": validation_audio_fingerprint,
                "noise_audio": noise_audio_fingerprint,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _manifest_audio_identity(path: Path) -> dict[str, Any]:
    root = path.resolve().parent
    seen: dict[str, Path] = {}
    duplicates: list[tuple[Path, Path]] = []
    count = 0
    with path.resolve().open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            audio = Path(str(raw["audio"]))
            if not audio.is_absolute():
                audio = (root / audio).resolve()
            if not audio.is_file():
                raise ValueError(
                    f"manifest {path} line {line_number} audio is missing: {audio}"
                )
            audio_sha256 = _sha256(audio)
            if audio_sha256 in seen:
                duplicates.append((seen[audio_sha256], audio))
            else:
                seen[audio_sha256] = audio
            count += 1
    if count == 0:
        raise ValueError(f"manifest is empty: {path}")
    if duplicates:
        first, second = duplicates[0]
        raise ValueError(
            f"manifest contains duplicate audio content: {first} and {second}"
        )
    fingerprint = hashlib.sha256(
        "\n".join(sorted(seen)).encode()
    ).hexdigest()
    return {
        "items": count,
        "unique_audio_sha256": sorted(seen),
        "fingerprint_sha256": fingerprint,
    }


def _validate_manifest_separation(
    train_manifest: Path,
    validation_manifest: Path,
    noise_manifest: Path | None = None,
) -> dict[str, dict[str, Any]]:
    validate_manifest_group_separation(
        {
            "train": train_manifest,
            "validation": validation_manifest,
        }
    )
    train_identity = _manifest_audio_identity(train_manifest)
    validation_identity = _manifest_audio_identity(validation_manifest)
    overlap = set(train_identity["unique_audio_sha256"]) & set(
        validation_identity["unique_audio_sha256"]
    )
    if overlap:
        raise ValueError(
            "train and validation manifests overlap by audio content "
            f"({len(overlap)} file(s))"
        )
    result = {
        "train": {
            "items": train_identity["items"],
            "fingerprint_sha256": train_identity["fingerprint_sha256"],
        },
        "validation": {
            "items": validation_identity["items"],
            "fingerprint_sha256": validation_identity["fingerprint_sha256"],
        },
    }
    if noise_manifest is not None:
        noise_identity = _manifest_audio_identity(noise_manifest)
        validation_noise_overlap = set(
            validation_identity["unique_audio_sha256"]
        ) & set(noise_identity["unique_audio_sha256"])
        if validation_noise_overlap:
            raise ValueError(
                "noise and validation manifests overlap by audio content "
                f"({len(validation_noise_overlap)} file(s))"
            )
        result["noise"] = {
            "items": noise_identity["items"],
            "fingerprint_sha256": noise_identity["fingerprint_sha256"],
        }
    return result


def _reject_forbidden_validation_data(
    validation_manifest: Path,
    identifiers: list[str],
) -> None:
    normalized = sorted(
        {
            identifier.strip().lower()
            for identifier in identifiers
            if identifier.strip()
        }
    )
    if not normalized:
        return
    resolved = validation_manifest.resolve()
    path_text = resolved.as_posix().lower()
    for identifier in normalized:
        if identifier in path_text:
            raise ValueError(
                "validation manifest appears to use a forbidden public evaluation "
                f"set ({identifier!r} matched its path)"
            )
    with resolved.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"validation manifest line {line_number} is invalid JSON"
                ) from exc
            searchable = json.dumps(record, sort_keys=True).lower()
            for identifier in normalized:
                if identifier in searchable:
                    raise ValueError(
                        "validation manifest appears to use a forbidden public "
                        f"evaluation set ({identifier!r} matched line {line_number})"
                    )


def _selection_key(result: dict[str, Any]) -> tuple[float, float, float, float]:
    detector = result["detector_metrics"]
    frame = result["frame_metrics"]
    return (
        float(detector["f1"]),
        -float(detector["false_alarm_rate"]),
        -float(detector["miss_rate"]),
        float(frame["f1"]),
    )


def _select_trial(
    completed: list[dict[str, Any]],
    *,
    min_detector_f1: float,
    max_false_alarm_rate: float,
    max_miss_rate: float,
) -> dict[str, Any] | None:
    eligible = [
        result
        for result in completed
        if (
            float(result["detector_metrics"]["f1"]) >= min_detector_f1
            and float(result["detector_metrics"]["false_alarm_rate"])
            <= max_false_alarm_rate
            and float(result["detector_metrics"]["miss_rate"]) <= max_miss_rate
        )
    ]
    return max(eligible, key=_selection_key) if eligible else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--valid-manifest", type=Path, required=True)
    parser.add_argument("--noise-manifest", type=Path)
    parser.add_argument("--region-profile", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()

    sweep_path = args.sweep.resolve()
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    base_path = (sweep_path.parent / sweep["base_config"]).resolve()
    base_raw = json.loads(base_path.read_text(encoding="utf-8"))
    ProjectConfig.from_dict(base_raw)
    trials = sweep.get("trials")
    if not isinstance(trials, list) or not trials:
        raise ValueError("sweep must contain at least one trial")
    selection = sweep.get("selection", {})
    forbidden_identifiers = selection.get("forbidden_evaluation_identifiers", [])
    if not isinstance(forbidden_identifiers, list) or not all(
        isinstance(item, str) for item in forbidden_identifiers
    ):
        raise ValueError(
            "selection.forbidden_evaluation_identifiers must be a list of strings"
        )
    _reject_forbidden_validation_data(
        args.valid_manifest,
        forbidden_identifiers,
    )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "sweep-results.json"
    resolved_device = str(select_training_device(args.device))
    runner_sha256 = _sha256(Path(__file__).resolve())
    repository = Path(__file__).resolve().parents[1]
    training_code_sha256 = _source_tree_sha256(repository)
    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "platform": platform.platform(),
        "device": resolved_device,
    }
    runtime_sha256 = hashlib.sha256(
        json.dumps(runtime, sort_keys=True).encode()
    ).hexdigest()
    manifest_identity = _validate_manifest_separation(
        args.train_manifest.resolve(),
        args.valid_manifest.resolve(),
        args.noise_manifest.resolve() if args.noise_manifest else None,
    )
    report: dict[str, Any] = {
        "schema": "flashvad-training-sweep-v1",
        "name": sweep["name"],
        "started_at_utc": datetime.now(UTC).isoformat(),
        "sweep_sha256": _sha256(sweep_path),
        "base_config_sha256": _sha256(base_path),
        "runner_sha256": runner_sha256,
        "training_code_sha256": training_code_sha256,
        "runtime": runtime,
        "runtime_sha256": runtime_sha256,
        "manifests": {
            "train": {
                "path": str(args.train_manifest.resolve()),
                "sha256": _sha256(args.train_manifest.resolve()),
            },
            "validation": {
                "path": str(args.valid_manifest.resolve()),
                "sha256": _sha256(args.valid_manifest.resolve()),
            },
            "noise": (
                {
                    "path": str(args.noise_manifest.resolve()),
                    "sha256": _sha256(args.noise_manifest.resolve()),
                }
                if args.noise_manifest
                else None
            ),
        },
        "region_profile": (
            {
                "path": str(args.region_profile.resolve()),
                "sha256": _sha256(args.region_profile.resolve()),
            }
            if args.region_profile
            else None
        ),
        "audio_identity": manifest_identity,
        "selection": selection,
        "results": [],
    }

    for trial in trials:
        trial_id = str(trial["id"])
        trial_output = output / trial_id
        checkpoint_path = trial_output / "best.pt"
        config_path = trial_output / "requested-config.json"
        input_marker = trial_output / "training-input.sha256"
        trial_output.mkdir(parents=True, exist_ok=True)
        config = ProjectConfig.from_dict(_merge(base_raw, trial.get("overrides", {})))
        config.save(config_path)
        input_digest = _training_input_digest(
            config_sha256=_sha256(config_path),
            train_sha256=report["manifests"]["train"]["sha256"],
            validation_sha256=report["manifests"]["validation"]["sha256"],
            noise_sha256=(
                report["manifests"]["noise"]["sha256"]
                if report["manifests"]["noise"]
                else None
            ),
            region_profile_sha256=(
                report["region_profile"]["sha256"]
                if report["region_profile"]
                else None
            ),
            runner_sha256=runner_sha256,
            device=resolved_device,
            training_code_sha256=training_code_sha256,
            runtime_sha256=runtime_sha256,
            train_audio_fingerprint=report["audio_identity"]["train"][
                "fingerprint_sha256"
            ],
            validation_audio_fingerprint=report["audio_identity"]["validation"][
                "fingerprint_sha256"
            ],
            noise_audio_fingerprint=(
                report["audio_identity"]["noise"]["fingerprint_sha256"]
                if "noise" in report["audio_identity"]
                else None
            ),
        )
        started = time.perf_counter()
        try:
            cached_digest = (
                input_marker.read_text(encoding="utf-8").strip()
                if input_marker.exists()
                else None
            )
            if args.rerun or not checkpoint_path.exists() or cached_digest != input_digest:
                checkpoint_path = train(
                    config_path,
                    args.train_manifest,
                    args.valid_manifest,
                    trial_output,
                    args.device,
                    args.region_profile,
                    args.noise_manifest,
                )
                input_marker.write_text(input_digest + "\n", encoding="utf-8")
            checkpoint = load_checkpoint_data(checkpoint_path, map_location="cpu")
            result = {
                "id": trial_id,
                "status": "completed",
                "elapsed_seconds": time.perf_counter() - started,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "requested_config_sha256": _sha256(config_path),
                "seed": config.seed,
                "frame_metrics": checkpoint["metrics"],
                "detector_metrics": checkpoint["detector_metrics"],
            }
        except Exception as exc:
            result = {
                "id": trial_id,
                "status": "failed",
                "elapsed_seconds": time.perf_counter() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        report["results"].append(result)
        _write_report(report_path, report)

    completed = [
        result for result in report["results"] if result["status"] == "completed"
    ]
    if not completed:
        raise RuntimeError(f"all sweep trials failed; see {report_path}")
    selection = sweep["selection"]
    selected = _select_trial(
        completed,
        min_detector_f1=float(selection["min_detector_f1"]),
        max_false_alarm_rate=float(selection["max_false_alarm_rate"]),
        max_miss_rate=float(selection["max_miss_rate"]),
    )
    report["selected_trial"] = selected["id"] if selected else None
    report["selection_gate_met"] = selected is not None
    report["completed_at_utc"] = datetime.now(UTC).isoformat()
    _write_report(report_path, report)
    if selected is None:
        raise RuntimeError(
            "no training trial met every configured validation gate; "
            f"see {report_path}"
        )
    print(json.dumps({"report": str(report_path), "selected_trial": selected["id"]}))


if __name__ == "__main__":
    main()
