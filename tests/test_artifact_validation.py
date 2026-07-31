from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from flashvad.artifact_validation import (
    validate_bundled_artifacts,
    validate_sidecar_metadata,
)

ROOT = Path(__file__).parents[1] / "models" / "flashvad-v0.1"


def _copy_release(tmp_path: Path) -> Path:
    model = tmp_path / "flashvad-stream.onnx"
    shutil.copy2(ROOT / model.name, model)
    shutil.copy2(ROOT / "flashvad-stream.json", model.with_suffix(".json"))
    return model


def test_bundled_release_artifacts_validate() -> None:
    assert validate_bundled_artifacts(ROOT / "flashvad-stream.onnx") == (
        ROOT / "flashvad-stream.onnx"
    ).resolve()


def test_artifact_validation_rejects_tampered_model(tmp_path: Path) -> None:
    model = _copy_release(tmp_path)
    model.write_bytes(model.read_bytes() + b"tamper")
    with pytest.raises(RuntimeError, match="integrity check failed"):
        validate_bundled_artifacts(model)


def test_artifact_validation_rechecks_a_previously_valid_path(tmp_path: Path) -> None:
    model = _copy_release(tmp_path)
    validate_bundled_artifacts(model)
    model.write_bytes(model.read_bytes() + b"tamper-after-validation")

    with pytest.raises(RuntimeError, match="integrity check failed"):
        validate_bundled_artifacts(model)


def test_artifact_validation_rejects_tampered_sidecar(tmp_path: Path) -> None:
    model = _copy_release(tmp_path)
    sidecar = model.with_suffix(".json")
    payload = json.loads(sidecar.read_text())
    payload["mode"] = "offline"
    sidecar.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="integrity check failed"):
        validate_bundled_artifacts(model)


def test_sidecar_schema_rejects_invalid_values_without_digest_bypass(tmp_path: Path) -> None:
    sidecar = tmp_path / "flashvad-stream.json"
    payload = json.loads((ROOT / sidecar.name).read_text())
    payload["model"]["feature_dim"] = True
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="feature_dim must be numeric"):
        validate_sidecar_metadata(sidecar)
