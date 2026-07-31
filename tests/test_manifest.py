import json
from pathlib import Path

import numpy as np
import soundfile as sf

from flashvad.config import FeatureConfig
from flashvad.manifest import (
    ManifestItem,
    VadDataset,
    frame_labels,
    load_manifest,
    rebase_manifest_record,
    validate_manifest_group_separation,
)


def test_manifest_item_preserves_the_original_positional_teacher_fields() -> None:
    teacher = Path("teacher.npy")
    item = ManifestItem(
        Path("sample.wav"),
        16_000,
        "hi",
        "telephone",
        (),
        "mono",
        "pcmu",
        "mobile",
        "noise",
        12.0,
        teacher,
        0.4,
        False,
    )

    assert item.teacher_probabilities == teacher
    assert item.teacher_weight == 0.4
    assert item.teacher_confidence_weighting is False
    assert item.speaker_id is None


def test_manifest_resolves_relative_audio_and_labels_frames(tmp_path) -> None:
    wav = tmp_path / "sample.wav"
    sf.write(wav, np.zeros(16_000, dtype=np.float32), 16_000)
    manifest = tmp_path / "items.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio": "sample.wav",
                "sample_rate": 16_000,
                "language": "hi",
                "domain": "telephone",
                "segments": [{"start": 0.02, "end": 0.05, "label": "speech"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    item = load_manifest(manifest)[0]
    labels = frame_labels(item.segments, 8, FeatureConfig())

    assert item.audio == wav
    assert item.language == "hi"
    assert labels.tolist() == [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_manifest_preserves_optional_group_identity_fields(tmp_path) -> None:
    manifest = tmp_path / "items.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio": "sample.wav",
                "segments": [],
                "speaker_id": "speaker-7",
                "call_id": "call-2",
                "session_id": "session-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    item = load_manifest(manifest)[0]

    assert item.speaker_id == "speaker-7"
    assert item.call_id == "call-2"
    assert item.session_id == "session-1"


def test_manifest_group_validation_reports_cross_split_identity(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    record = {"audio": "train.wav", "segments": [], "speaker_id": "speaker-7"}
    train.write_text(json.dumps(record) + "\n", encoding="utf-8")
    test.write_text(
        json.dumps({**record, "audio": "test.wav"}) + "\n", encoding="utf-8"
    )

    try:
        validate_manifest_group_separation({"train": train, "test": test})
    except ValueError as exc:
        message = str(exc)
        assert "speaker_id='speaker-7'" in message
        assert "train" in message and "test" in message
    else:
        raise AssertionError("cross-split speaker overlap must be rejected")


def test_manifest_group_validation_allows_missing_identity_fields(tmp_path) -> None:
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    train.write_text(json.dumps({"audio": "train.wav", "segments": []}) + "\n")
    test.write_text(json.dumps({"audio": "test.wav", "segments": []}) + "\n")

    result = validate_manifest_group_separation({"train": train, "test": test})

    assert result["checked_fields"] == ["speaker_id", "call_id", "session_id"]


def test_rebase_manifest_record_preserves_audio_and_teacher_targets(tmp_path) -> None:
    source_root = tmp_path / "source"
    output_root = tmp_path / "merged"
    source_root.mkdir()
    output_root.mkdir()
    record = {
        "audio": "clips/item.wav",
        "teacher_probabilities": "teachers/item.npy",
        "segments": [],
    }

    rebased = rebase_manifest_record(record, source_root, output_root)

    assert (output_root / rebased["audio"]).resolve() == (
        source_root / "clips/item.wav"
    ).resolve()
    assert (output_root / rebased["teacher_probabilities"]).resolve() == (
        source_root / "teachers/item.npy"
    ).resolve()


def test_unlabeled_target_audio_can_use_direct_teacher_probabilities(tmp_path) -> None:
    wav = tmp_path / "meeting.wav"
    sf.write(wav, np.zeros(16_000, dtype=np.float32), 16_000)
    teacher = tmp_path / "meeting.npy"
    np.save(teacher, np.full(100, 0.4, dtype=np.float32))
    manifest = tmp_path / "items.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio": "meeting.wav",
                "teacher_probabilities": "meeting.npy",
                "teacher_weight": 1.0,
                "teacher_confidence_weighting": False,
                "segments": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _, labels, _ = VadDataset(manifest, FeatureConfig(), 1.0)[0]

    np.testing.assert_allclose(labels.numpy(), 0.4)
