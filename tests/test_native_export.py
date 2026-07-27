from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from flashvad.benchmark import load_checkpoint
from flashvad.features import CausalFeatureExtractor
from flashvad.native_export import _native_source_dir, build_macos_native, export_macos_native
from flashvad.native_runtime import MacosNativeVadModel


def _test_checkpoint() -> Path:
    return Path(
        os.environ.get(
            "FLASHVAD_TEST_CHECKPOINT",
            "models/flashvad-v0.1/flashvad-v0.1.pt",
        )
    )


def test_native_sources_are_available() -> None:
    source = _native_source_dir()
    assert (source / "flashvad_native.c").is_file()
    assert (source / "flashvad_native.h").is_file()
    assert (source / "benchmark.c").is_file()
    assert (source / "flashvad_weights.c").is_file()
    assert (source / "flashvad_weights.h").is_file()


@pytest.mark.skipif(platform.system() != "Darwin", reason="requires Apple Accelerate")
def test_bundled_native_runtime_builds_once_and_runs() -> None:
    owner = MacosNativeVadModel.load_bundled()
    probabilities, events = owner.new_stream().push(
        np.zeros(160, dtype=np.float32)
    )

    assert probabilities.shape == (1,)
    assert 0.0 <= float(probabilities[0]) <= 1.0
    assert isinstance(events, list)
    assert MacosNativeVadModel.load_bundled() is owner


@pytest.mark.skipif(platform.system() != "Darwin", reason="requires Apple Accelerate")
def test_native_runtime_matches_pytorch(tmp_path: Path) -> None:
    checkpoint = _test_checkpoint()
    if not checkpoint.exists():
        pytest.skip("smoke checkpoint is not present")

    generated = export_macos_native(checkpoint, tmp_path / "generated")
    products = build_macos_native(generated, tmp_path / "build")
    library = ctypes.CDLL(str(products["library"]))

    library.flashvad_state_size.argtypes = []
    library.flashvad_state_size.restype = ctypes.c_size_t
    library.flashvad_init.argtypes = [ctypes.c_void_p]
    library.flashvad_init.restype = ctypes.c_int
    library.flashvad_destroy.argtypes = [ctypes.c_void_p]
    library.flashvad_reset.argtypes = [ctypes.c_void_p]
    library.flashvad_extract_features.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
    ]
    library.flashvad_model_step.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
    ]
    library.flashvad_model_step.restype = ctypes.c_float
    library.flashvad_push.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    library.flashvad_push.restype = ctypes.c_size_t

    state = ctypes.create_string_buffer(library.flashvad_state_size())
    assert library.flashvad_init(state) == 0
    try:
        config, model = load_checkpoint(checkpoint)
        frontend = CausalFeatureExtractor(config.feature)
        torch.manual_seed(42)
        audio = torch.randn(config.feature.hop_samples * 1_000) * 0.05
        expected_features = frontend(audio).squeeze(0)
        model_state = model.initial_state(1, "cpu")

        native_probabilities = []
        torch_probabilities = []
        for hop_index, hop in enumerate(audio.split(config.feature.hop_samples)):
            hop_array = np.ascontiguousarray(hop.numpy(), dtype=np.float32)
            native_feature = np.empty(config.feature.feature_dim, dtype=np.float32)
            result = library.flashvad_extract_features(
                state,
                hop_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                native_feature.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
            assert result == 0
            np.testing.assert_allclose(
                native_feature,
                expected_features[hop_index].numpy(),
                rtol=2e-4,
                atol=2e-5,
            )
            native_probabilities.append(
                library.flashvad_model_step(
                    state,
                    native_feature.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                )
            )

            with torch.inference_mode():
                outputs, model_state = model.stream_step(
                    expected_features[hop_index].unsqueeze(0),
                    model_state,
                )
            torch_probabilities.append(
                float(torch.sigmoid(outputs["speech_logits"]).squeeze())
            )

        np.testing.assert_allclose(
            native_probabilities,
            torch_probabilities,
            rtol=2e-4,
            atol=2e-5,
        )
        assert np.max(
            np.abs(
                np.asarray(native_probabilities)
                - np.asarray(torch_probabilities)
            )
        ) < 2e-4

        library.flashvad_reset(state)
        pushed_probabilities = []
        cursor = 0
        for chunk_size in (17, 511, 3, 1_024, 99, 2_346):
            if cursor >= audio.numel():
                break
            chunk = np.ascontiguousarray(
                audio[cursor : cursor + chunk_size].numpy(),
                dtype=np.float32,
            )
            cursor += chunk.size
            capacity = (chunk.size + config.feature.hop_samples - 1) // (
                config.feature.hop_samples
            ) + 1
            output = np.empty(capacity, dtype=np.float32)
            emitted = library.flashvad_push(
                state,
                chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                chunk.size,
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                capacity,
            )
            pushed_probabilities.extend(output[:emitted])
        if cursor < audio.numel():
            chunk = np.ascontiguousarray(audio[cursor:].numpy(), dtype=np.float32)
            capacity = chunk.size // config.feature.hop_samples + 2
            output = np.empty(capacity, dtype=np.float32)
            emitted = library.flashvad_push(
                state,
                chunk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                chunk.size,
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                capacity,
            )
            pushed_probabilities.extend(output[:emitted])
        np.testing.assert_allclose(
            pushed_probabilities,
            native_probabilities,
            rtol=2e-4,
            atol=2e-5,
        )

        native_owner = MacosNativeVadModel(products["library"], config.detector)
        with native_owner.new_stream() as native_stream:
            wrapped_probabilities, wrapped_events = native_stream.push(audio.numpy())
        np.testing.assert_allclose(
            wrapped_probabilities,
            native_probabilities,
            rtol=2e-4,
            atol=2e-5,
        )
        assert isinstance(wrapped_events, list)

        library.flashvad_reset(state)
        one_hop = np.zeros(config.feature.hop_samples, dtype=np.float32)
        rejected = library.flashvad_push(
            state,
            one_hop.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            one_hop.size,
            None,
            0,
        )
        assert rejected == ctypes.c_size_t(-1).value
    finally:
        library.flashvad_destroy(state)


@pytest.mark.skipif(platform.system() != "Darwin", reason="requires Apple Accelerate")
def test_native_benchmark_emits_json(tmp_path: Path) -> None:
    checkpoint = _test_checkpoint()
    if not checkpoint.exists():
        pytest.skip("smoke checkpoint is not present")
    generated = export_macos_native(checkpoint, tmp_path / "generated")
    products = build_macos_native(generated, tmp_path / "build")
    completed = subprocess.run(
        [str(products["benchmark"])],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"runtime": "accelerate-embedded"' in completed.stdout
