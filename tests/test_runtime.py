import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from flashvad.config import DetectorConfig


def test_onnx_stream_api_is_available_with_export_dependencies() -> None:
    runtime = pytest.importorskip("flashvad.runtime")
    # Export validation is covered by the integration run. Keep the unit test
    # independent of a generated model artifact.
    assert DetectorConfig().start_threshold > DetectorConfig().stop_threshold
    assert runtime.OnnxStreamingVadModel.new_stream


def test_real_onnx_per_call_state_allocation_is_sub_millisecond() -> None:
    runtime = pytest.importorskip("flashvad.runtime")
    owner = runtime.OnnxStreamingVadModel.load_bundled()
    for _ in range(100):
        owner.new_stream()
    clock = time.thread_time_ns
    timings = []
    for _ in range(1_000):
        started = clock()
        owner.new_stream()
        timings.append((clock() - started) / 1_000)
    assert float(np.percentile(timings, 99)) < 1_000


def test_bundled_onnx_model_runs_without_a_repository_model_path() -> None:
    runtime = pytest.importorskip("flashvad.runtime")

    owner = runtime.OnnxStreamingVadModel.load_bundled()
    probabilities, events = owner.new_stream().push(
        np.zeros(160, dtype=np.float32)
    )

    assert probabilities.shape == (1,)
    assert 0.0 <= float(probabilities[0]) <= 1.0
    assert isinstance(events, list)
    assert runtime.OnnxStreamingVadModel.load_bundled() is owner


def test_onnx_runtime_defaults_to_cpu_execution_provider() -> None:
    runtime = pytest.importorskip("flashvad.runtime")
    owner = runtime.OnnxStreamingVadModel.load_bundled()

    assert owner.execution_provider == "CPUExecutionProvider"
    assert owner.device_type == "cpu"


def test_onnx_runtime_rejects_an_unavailable_execution_provider() -> None:
    runtime = pytest.importorskip("flashvad.runtime")

    with pytest.raises(ValueError, match="execution providers are unavailable"):
        runtime.OnnxStreamingVadModel.load_bundled(
            providers=["DefinitelyMissingExecutionProvider"]
        )


def test_onnx_runtime_rejects_an_empty_provider_list() -> None:
    runtime = pytest.importorskip("flashvad.runtime")

    with pytest.raises(ValueError, match="at least one"):
        runtime.OnnxStreamingVadModel.load_bundled(providers=[])


def test_onnx_runtime_rejects_silent_provider_fallback() -> None:
    runtime = pytest.importorskip("flashvad.runtime")

    class FallbackSession:
        def get_providers(self) -> list[str]:
            return ["CPUExecutionProvider"]

        def disable_fallback(self) -> None:
            raise AssertionError("fallback must be rejected before it is disabled")

    with pytest.raises(RuntimeError, match="refusing silent fallback"):
        runtime._require_active_provider(FallbackSession(), "CUDAExecutionProvider")


def test_model_constructor_enforces_the_requested_provider(monkeypatch) -> None:
    runtime = pytest.importorskip("flashvad.runtime")
    calls = []
    require_active_provider = runtime._require_active_provider

    def record_provider(session, expected: str) -> None:
        calls.append(expected)
        require_active_provider(session, expected)

    monkeypatch.setattr(runtime, "_require_active_provider", record_provider)
    owner = runtime.OnnxStreamingVadModel.load_bundled(
        providers=["CPUExecutionProvider"]
    )

    assert owner.execution_provider == "CPUExecutionProvider"
    assert calls == ["CPUExecutionProvider"]


def test_explicit_cpu_provider_uses_a_separate_owner() -> None:
    runtime = pytest.importorskip("flashvad.runtime")
    cached = runtime.OnnxStreamingVadModel.load_bundled()
    explicit = runtime.OnnxStreamingVadModel.load_bundled(
        providers=["CPUExecutionProvider"]
    )

    assert explicit is not cached
    assert explicit.execution_provider == "CPUExecutionProvider"


def test_cuda_stream_matches_cpu_when_cuda_provider_is_available() -> None:
    runtime = pytest.importorskip("flashvad.runtime")
    if "CUDAExecutionProvider" not in runtime.ort.get_available_providers():
        pytest.skip("onnxruntime-gpu CUDAExecutionProvider is unavailable")
    audio = np.random.default_rng(14).normal(0, 0.05, 160 * 16).astype(np.float32)
    cpu = runtime.OnnxStreamingVadModel.load_bundled(
        providers=["CPUExecutionProvider"]
    )
    cuda = runtime.OnnxStreamingVadModel.load_bundled(
        providers=[
            (
                "CUDAExecutionProvider",
                {"do_copy_in_default_stream": "1"},
            )
        ]
    )

    expected, _ = cpu.new_stream().push(audio)
    stream = cuda.new_stream()
    actual, _ = stream.push(audio)
    stream.reset()
    replay, _ = stream.push(audio)

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(replay, actual, rtol=0.0, atol=1e-7)
    assert all(isinstance(value, runtime.ort.OrtValue) for value in stream.state.values())


def test_load_bundled_keeps_legacy_subclass_constructor_compatible() -> None:
    runtime = pytest.importorskip("flashvad.runtime")

    class LegacyOwner(runtime.OnnxStreamingVadModel):
        def __init__(self, model_path: str | Path, threads: int = 1) -> None:
            super().__init__(model_path, threads=threads)

    owner = LegacyOwner.load_bundled()

    assert owner.execution_provider == "CPUExecutionProvider"


def test_runtime_handles_reordered_outputs_and_anonymous_dynamic_dimensions(
    tmp_path,
) -> None:
    onnx = pytest.importorskip("onnx")
    runtime = pytest.importorskip("flashvad.runtime")
    source = Path("models/flashvad-v0.1/flashvad-stream.onnx")
    model_path = tmp_path / source.name
    sidecar_path = model_path.with_suffix(".json")
    model = onnx.load(source)
    outputs = list(model.graph.output)
    model.graph.ClearField("output")
    model.graph.output.extend(reversed(outputs))
    speech_logits = next(output for output in model.graph.output if output.name == "speech_logits")
    speech_logits.type.tensor_type.shape.dim[0].ClearField("dim_param")
    onnx.save(model, model_path)
    shutil.copy2(source.with_suffix(".json"), sidecar_path)

    owner = runtime.OnnxStreamingVadModel(model_path)
    probabilities, _ = owner.new_stream().push(np.zeros(160, dtype=np.float32))

    assert probabilities.shape == (1,)
    assert np.isfinite(probabilities).all()


def test_onnx_stream_reset_and_chunking_are_deterministic() -> None:
    runtime = pytest.importorskip("flashvad.runtime")
    owner = runtime.OnnxStreamingVadModel.load_bundled()
    audio = np.random.default_rng(4).normal(0, 0.05, 160 * 8).astype(np.float32)
    baseline, baseline_events = owner.new_stream().push(audio)
    stream = owner.new_stream()
    chunks = [
        stream.push(audio[:37])[0],
        stream.push(audio[37:511])[0],
        stream.push(audio[511:])[0],
    ]
    chunked = np.concatenate(chunks)
    assert np.array_equal(chunked, baseline)
    stream.reset()
    replay, replay_events = stream.push(audio)
    assert np.array_equal(replay, baseline)
    assert replay_events == baseline_events


def test_onnx_streams_have_independent_state() -> None:
    runtime = pytest.importorskip("flashvad.runtime")
    owner = runtime.OnnxStreamingVadModel.load_bundled()
    audio = np.random.default_rng(9).normal(0, 0.05, 160 * 4).astype(np.float32)
    first, second = owner.new_stream(), owner.new_stream()
    a1 = first.push(audio[:320])[0]
    b = second.push(audio)[0]
    a2 = first.push(audio[320:])[0]
    independent = np.concatenate((a1, a2))
    assert np.array_equal(independent, b)


def test_bundled_onnx_probabilities_preserve_the_release_reference() -> None:
    runtime = pytest.importorskip("flashvad.runtime")
    audio = np.random.default_rng(4).normal(0, 0.05, 160 * 8).astype(np.float32)
    probabilities, _ = runtime.OnnxStreamingVadModel.load_bundled().new_stream().push(audio)
    expected = np.asarray(
        [
            0.14926424622535706,
            0.1480158567428589,
            0.15785996615886688,
            0.1314583122730255,
            0.10600105673074722,
            0.08741875737905502,
            0.07559503614902496,
            0.06994945555925369,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(probabilities, expected, rtol=0.0, atol=1e-7)


def test_shared_onnx_owner_supports_parallel_independent_streams() -> None:
    runtime = pytest.importorskip("flashvad.runtime")
    owner = runtime.OnnxStreamingVadModel.load_bundled()
    audio = np.random.default_rng(11).normal(0, 0.05, 160 * 16).astype(np.float32)
    expected, _ = owner.new_stream().push(audio)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: owner.new_stream().push(audio)[0], range(16)))

    for probabilities in results:
        np.testing.assert_array_equal(probabilities, expected)


def test_explicit_onnx_model_rejects_a_mismatched_sidecar_contract(tmp_path) -> None:
    runtime = pytest.importorskip("flashvad.runtime")
    source = Path("models/flashvad-v0.1/flashvad-stream.onnx")
    model = tmp_path / source.name
    sidecar = model.with_suffix(".json")
    shutil.copy2(source, model)
    payload = json.loads(source.with_suffix(".json").read_text())
    payload["model"]["recurrent_dim"] = 32
    sidecar.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="tensor contract"):
        runtime.OnnxStreamingVadModel(model)
