from __future__ import annotations

import importlib.util
import wave
from pathlib import Path
from types import ModuleType

import pytest


def _load_script() -> ModuleType:
    path = Path("scripts/convert_scv_manifest.py")
    spec = importlib.util.spec_from_file_location("flashvad_convert_scv_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


converter = _load_script()


def _write_wav(path: Path, *, sample_rate: int = 16_000) -> None:
    with wave.open(str(path), "wb") as destination:
        destination.setnchannels(1)
        destination.setsampwidth(2)
        destination.setframerate(sample_rate)
        destination.writeframes(b"\0\0" * 160)


def test_scv_converter_validates_audio_and_extracts_speech(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    labels = tmp_path / "sample.scv"
    _write_wav(audio)
    labels.write_text(
        "sample,0.000,0.100,0,0.100,0.500,1,0.500,0.600,0",
        encoding="utf-8",
    )

    assert converter.validate_wav(audio) == {
        "channels": 1,
        "sample_width_bytes": 2,
        "sample_rate": 16_000,
        "frames": 160,
    }
    assert converter.scv_speech_segments(labels) == [
        {"start": 0.1, "end": 0.5, "label": "speech"}
    ]
    assert len(converter.sha256(audio)) == 64


def test_scv_converter_rejects_non_16khz_audio(tmp_path: Path) -> None:
    audio = tmp_path / "sample.wav"
    _write_wav(audio, sample_rate=8_000)

    with pytest.raises(ValueError, match="16 kHz"):
        converter.validate_wav(audio)
