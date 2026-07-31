from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import datasets


def _load_script(name: str, filename: str) -> ModuleType:
    path = Path("scripts") / filename
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fleurs = _load_script("flashvad_prepare_fleurs_test", "prepare_fleurs_vad.py")
sweep_runner = _load_script("flashvad_training_sweep_test", "run_training_sweep.py")


def test_fleurs_cache_keeps_upstream_splits_separate(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, str, int]] = []

    def fake_rows(
        language: str,
        split: str,
        count: int,
        revision: str,
    ) -> list[dict[str, object]]:
        calls.append((language, split, count))
        return [{"row_idx": 1, "row": {"split": split}}]

    monkeypatch.setattr(fleurs, "_rows", fake_rows)

    train = fleurs._cached_rows("hi_in", "train", 1, tmp_path, "revision-a")
    valid = fleurs._cached_rows(
        "hi_in",
        "validation",
        1,
        tmp_path,
        "revision-a",
    )

    assert train != valid
    assert calls == [("hi_in", "train", 1), ("hi_in", "validation", 1)]
    assert (tmp_path / "hi_in-train-1-revision-a-rows.json").exists()
    assert (tmp_path / "hi_in-validation-1-revision-a-rows.json").exists()

    fleurs._cached_rows("hi_in", "train", 1, tmp_path, "revision-b")

    assert calls[-1] == ("hi_in", "train", 1)
    assert len(calls) == 3


def test_fleurs_streamed_sources_use_unique_row_positions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeDataset(list):
        def decode(self, _enabled: bool) -> FakeDataset:
            return self

    fake = FakeDataset(
        [
            {"id": 196, "audio": {"path": "first.wav", "bytes": b"first"}},
            {"id": 196, "audio": {"path": "second.wav", "bytes": b"second"}},
        ]
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: fake)

    records = fleurs._streamed_sources(
        "en_us",
        "train",
        2,
        tmp_path / "metadata",
        tmp_path / "sources",
        tmp_path,
        "revision-a",
    )

    assert [record[2] for record in records] == [0, 1]
    assert records[0][3] != records[1][3]
    assert "revision-a" in records[0][3].parts
    assert records[0][3].read_bytes() == b"first"
    assert records[1][3].read_bytes() == b"second"


def test_fleurs_hybrid_falls_back_to_revision_pinned_streaming(
    tmp_path: Path,
    monkeypatch,
) -> None:
    expected = [("bn_in", "validation", 0, tmp_path / "fallback.wav")]

    def rows_unavailable(*_args, **_kwargs):
        raise RuntimeError("rows API unavailable")

    def streamed(*args, **_kwargs):
        assert args[1] == "validation"
        assert args[-1] == "revision-a"
        return expected

    monkeypatch.setattr(fleurs, "_rows_api_sources", rows_unavailable)
    monkeypatch.setattr(fleurs, "_streamed_sources", streamed)

    result = fleurs._hybrid_sources(
        "bn_in",
        "validation",
        1,
        tmp_path / "metadata",
        tmp_path / "sources",
        tmp_path,
        "revision-a",
    )

    assert result == expected


def test_fleurs_seed_components_do_not_collide_for_default_languages() -> None:
    components = {
        language: fleurs._seed_component(language)
        for language in fleurs.DEFAULT_LANGUAGES
    }

    assert len(set(components.values())) == len(components)
    assert components["kn_in"] != components["ml_in"]


def test_sweep_overrides_nested_settings_without_mutating_base() -> None:
    base = {"seed": 1, "training": {"focal_gamma": 0.0, "epochs": 8}}

    merged = sweep_runner._merge(
        base,
        {"seed": 2, "training": {"focal_gamma": 1.0}},
    )

    assert merged == {
        "seed": 2,
        "training": {"focal_gamma": 1.0, "epochs": 8},
    }
    assert base["training"]["focal_gamma"] == 0.0


def test_sweep_selection_prioritizes_detector_f1_then_operating_cost() -> None:
    first = {
        "detector_metrics": {"f1": 0.9, "false_alarm_rate": 0.1, "miss_rate": 0.2},
        "frame_metrics": {"f1": 0.95},
    }
    second = {
        "detector_metrics": {"f1": 0.9, "false_alarm_rate": 0.08, "miss_rate": 0.25},
        "frame_metrics": {"f1": 0.96},
    }

    assert sweep_runner._selection_key(second) > sweep_runner._selection_key(first)


def test_sweep_cache_key_includes_the_region_profile() -> None:
    first = sweep_runner._training_input_digest(
        config_sha256="config",
        train_sha256="train",
        validation_sha256="valid",
        noise_sha256="noise",
        region_profile_sha256="region-a",
        runner_sha256="runner",
        device="mps",
        training_code_sha256="code",
        runtime_sha256="runtime",
        train_audio_fingerprint="train-audio",
        validation_audio_fingerprint="valid-audio",
        noise_audio_fingerprint="noise-audio",
    )
    second = sweep_runner._training_input_digest(
        config_sha256="config",
        train_sha256="train",
        validation_sha256="valid",
        noise_sha256="noise",
        region_profile_sha256="region-b",
        runner_sha256="runner",
        device="mps",
        training_code_sha256="code",
        runtime_sha256="runtime",
        train_audio_fingerprint="train-audio",
        validation_audio_fingerprint="valid-audio",
        noise_audio_fingerprint="noise-audio",
    )

    assert first != second


def test_sweep_does_not_select_a_trial_that_misses_the_false_alarm_gate() -> None:
    completed = [
        {
            "detector_metrics": {
                "f1": 0.99,
                "false_alarm_rate": 0.16,
                "miss_rate": 0.01,
            },
            "frame_metrics": {"f1": 0.99},
        }
    ]

    assert (
        sweep_runner._select_trial(
            completed,
            min_detector_f1=0.85,
            max_false_alarm_rate=0.15,
            max_miss_rate=0.2,
        )
        is None
    )


def test_sweep_rejects_degenerate_all_silence_trial() -> None:
    completed = [
        {
            "detector_metrics": {
                "f1": 0.0,
                "false_alarm_rate": 0.0,
                "miss_rate": 1.0,
            },
            "frame_metrics": {"f1": 0.0},
        }
    ]

    assert (
        sweep_runner._select_trial(
            completed,
            min_detector_f1=0.85,
            max_false_alarm_rate=0.15,
            max_miss_rate=0.2,
        )
        is None
    )


def test_sweep_rejects_train_validation_audio_overlap(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"same-audio")
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    record = json.dumps({"audio": "clip.wav"}) + "\n"
    train.write_text(record, encoding="utf-8")
    valid.write_text(record, encoding="utf-8")

    try:
        sweep_runner._validate_manifest_separation(train, valid)
    except ValueError as exc:
        assert "overlap by audio content" in str(exc)
    else:
        raise AssertionError("overlapping manifests must be rejected")


def test_sweep_rejects_noise_validation_audio_overlap(tmp_path: Path) -> None:
    train_audio = tmp_path / "train.wav"
    shared_audio = tmp_path / "shared.wav"
    train_audio.write_bytes(b"train-audio")
    shared_audio.write_bytes(b"shared-audio")
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    noise = tmp_path / "noise.jsonl"
    train.write_text(json.dumps({"audio": "train.wav"}) + "\n", encoding="utf-8")
    shared_record = json.dumps({"audio": "shared.wav"}) + "\n"
    valid.write_text(shared_record, encoding="utf-8")
    noise.write_text(shared_record, encoding="utf-8")

    try:
        sweep_runner._validate_manifest_separation(train, valid, noise)
    except ValueError as exc:
        assert "noise and validation" in str(exc)
    else:
        raise AssertionError("validation audio cannot be used as training noise")


def test_sweep_rejects_train_validation_speaker_overlap(tmp_path: Path) -> None:
    train_audio = tmp_path / "train.wav"
    valid_audio = tmp_path / "valid.wav"
    train_audio.write_bytes(b"different-train-audio")
    valid_audio.write_bytes(b"different-validation-audio")
    train = tmp_path / "train.jsonl"
    valid = tmp_path / "valid.jsonl"
    train.write_text(
        json.dumps({"audio": "train.wav", "speaker_id": "speaker-7"}) + "\n",
        encoding="utf-8",
    )
    valid.write_text(
        json.dumps({"audio": "valid.wav", "speaker_id": "speaker-7"}) + "\n",
        encoding="utf-8",
    )

    try:
        sweep_runner._validate_manifest_separation(train, valid)
    except ValueError as exc:
        assert "speaker_id='speaker-7'" in str(exc)
    else:
        raise AssertionError("speaker identities cannot cross training splits")


def test_committed_sweep_forbids_public_competitor_sets() -> None:
    sweep = json.loads(
        Path("configs/sweeps/multilingual-mac.json").read_text(encoding="utf-8")
    )

    forbidden = " ".join(sweep["selection"]["forbidden_evaluation_sets"])
    assert "TEN public test set" in forbidden
    assert sweep["selection"]["min_detector_f1"] >= 0.85
    assert sweep["selection"]["max_false_alarm_rate"] <= 0.15
    assert sweep["selection"]["max_miss_rate"] <= 0.2
    assert len(sweep["trials"]) >= 3


def test_sweep_rejects_forbidden_public_validation_data(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"audio")
    manifest = tmp_path / "validation.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audio": "clip.wav",
                "condition": "ten-public-testset",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    try:
        sweep_runner._reject_forbidden_validation_data(
            manifest,
            ["ten-public", "silero", "firered"],
        )
    except ValueError as exc:
        assert "forbidden public evaluation set" in str(exc)
    else:
        raise AssertionError("public evaluation data must be rejected from selection")


def test_default_fleurs_languages_cover_available_indian_languages() -> None:
    assert {
        "bn_in",
        "gu_in",
        "hi_in",
        "kn_in",
        "ml_in",
        "mr_in",
        "pa_in",
        "ta_in",
        "te_in",
        "ur_pk",
    }.issubset(fleurs.DEFAULT_LANGUAGES)


def test_region_profile_does_not_downweight_language_free_real_negatives() -> None:
    profile = json.loads(
        Path("configs/regions/india_gcc.json").read_text(encoding="utf-8")
    )

    assert profile["language_weights"]["zxx"] >= 1.0
    assert profile["domain_weights"]["musan"] >= 1.0
