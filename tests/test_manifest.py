import json

import numpy as np
import soundfile as sf

from flashvad.config import FeatureConfig
from flashvad.manifest import (
    VadDataset,
    frame_labels,
    load_manifest,
    rebase_manifest_record,
)


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
