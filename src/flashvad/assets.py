"""Paths to FlashVAD's bundled inference artifacts."""

from __future__ import annotations

from pathlib import Path

from .artifact_validation import validate_bundled_artifacts


def _bundled_model_dir() -> Path:
    packaged = Path(__file__).resolve().parent / "_models"
    if packaged.is_dir():
        return packaged
    repository = Path(__file__).resolve().parents[2] / "models" / "flashvad-v0.1"
    if repository.is_dir():
        return repository
    raise RuntimeError("FlashVAD's bundled ONNX model is missing from this installation")


def bundled_model_path() -> Path:
    """Return the self-contained streaming ONNX model shipped with FlashVAD."""

    path = _bundled_model_dir() / "flashvad-stream.onnx"
    if not path.is_file():
        raise RuntimeError("FlashVAD's bundled ONNX model is missing from this installation")
    return validate_bundled_artifacts(path)


__all__ = ["bundled_model_path"]
