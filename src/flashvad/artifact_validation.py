"""Integrity and schema checks for the bundled inference artifacts."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Any

from .config import ProjectConfig

RELEASE_MANIFEST_VERSION = "flashvad-v0.1"
_EXPECTED = {
    "flashvad-stream.onnx": "9a88e34bf3118d60e25a16cb622cb394e2f3ab71445b0aa5957df6f1d5f1b6ba",
    "flashvad-stream.json": "5052faab08f4a50c9b628b7a57e1747d368b3e7d4cedd3c92da8e3095ba3ea65",
}
_SIDECAR_KEYS = {"feature", "model", "detector", "mode", "note"}
_FEATURE_KEYS = {"sample_rate", "frame_ms", "hop_ms", "n_fft", "n_mels", "f_min", "f_max"}
_MODEL_KEYS = {
    "feature_dim", "hidden_dim", "kernel_size", "dilations", "recurrent_dim", "dropout"
}
_DETECTOR_KEYS = {
    "start_threshold", "stop_threshold", "start_frames", "stop_frames", "pre_roll_frames"
}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"bundled sidecar {name} must be an object")
    return value


def _exact_keys(value: dict[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"bundled sidecar {name} keys do not match release schema")


def validate_sidecar_metadata(path: str | Path) -> None:
    """Validate a FlashVAD release sidecar without trusting its model digest."""
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid bundled sidecar: {path}") from exc
    root = _mapping(payload, "root")
    _exact_keys(root, _SIDECAR_KEYS, "root")
    feature = _mapping(root["feature"], "feature")
    model = _mapping(root["model"], "model")
    detector = _mapping(root["detector"], "detector")
    _exact_keys(feature, _FEATURE_KEYS, "feature")
    _exact_keys(model, _MODEL_KEYS, "model")
    _exact_keys(detector, _DETECTOR_KEYS, "detector")
    if root["mode"] != "streaming" or not isinstance(root["note"], str):
        raise ValueError("bundled sidecar mode/note are invalid")
    integer_fields = {
        "sample_rate",
        "n_fft",
        "n_mels",
        "feature_dim",
        "hidden_dim",
        "kernel_size",
        "recurrent_dim",
        "start_frames",
        "stop_frames",
        "pre_roll_frames",
    }
    numeric = (
        *_FEATURE_KEYS,
        *(_MODEL_KEYS - {"dilations"}),
        *_DETECTOR_KEYS,
    )

    def value_for(key: str) -> Any:
        group = "feature" if key in _FEATURE_KEYS else "model" if key in _MODEL_KEYS else "detector"
        return root[group][key]

    for key in numeric:
        value = value_for(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"bundled sidecar {key} must be numeric")
        if not math.isfinite(float(value)):
            raise ValueError(f"bundled sidecar {key} must be finite")
        if key in integer_fields and not isinstance(value, int):
            raise ValueError(f"bundled sidecar {key} must be an integer")
    if (
        not isinstance(model["dilations"], list)
        or not model["dilations"]
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in model["dilations"]
        )
    ):
        raise ValueError("bundled sidecar dilations are invalid")
    ProjectConfig.from_dict(root)


def validate_bundled_artifacts(model_path: str | Path) -> Path:
    """Validate a release artifact and return its canonical ONNX path."""
    model = Path(model_path).resolve()
    sidecar = model.with_suffix(".json")
    for path in (model, sidecar):
        expected = _EXPECTED.get(path.name)
        if expected is None or not path.is_file():
            raise RuntimeError(f"bundled artifact is missing or unrecognized: {path}")
        actual = _digest(path)
        if not hmac.compare_digest(actual, expected):
            raise RuntimeError(f"bundled artifact integrity check failed: {path.name}")
    validate_sidecar_metadata(sidecar)
    return model


__all__ = ["validate_bundled_artifacts", "validate_sidecar_metadata"]
