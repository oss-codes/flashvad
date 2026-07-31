from __future__ import annotations

import json
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .artifact_validation import validate_sidecar_metadata
from .assets import bundled_model_path
from .config import DetectorConfig, FeatureConfig
from .detector import HysteresisDetector, VadEvent
from .numpy_features import NumpyCausalFeatureExtractor, NumpyStreamingFeatureExtractor

ProviderOption = str | int | bool
ProviderSpec = str | tuple[str, dict[str, ProviderOption]]
_CUDA_PROVIDERS = {"CUDAExecutionProvider", "TensorrtExecutionProvider"}


def _provider_name(provider: ProviderSpec) -> str:
    return provider if isinstance(provider, str) else provider[0]


def _validated_providers(providers: Sequence[ProviderSpec] | None) -> list[ProviderSpec]:
    requested = list(("CPUExecutionProvider",) if providers is None else providers)
    if not requested:
        raise ValueError("at least one ONNX Runtime execution provider is required")
    available = set(ort.get_available_providers())
    missing = [
        _provider_name(provider)
        for provider in requested
        if _provider_name(provider) not in available
    ]
    if missing:
        raise ValueError(
            "requested ONNX Runtime execution providers are unavailable: "
            f"{missing}; available={sorted(available)}"
        )
    return requested


def _require_active_provider(session: ort.InferenceSession, expected: str) -> None:
    active = session.get_providers()
    if not active or active[0] != expected:
        raise RuntimeError(
            f"ONNX Runtime requested {expected} but activated {active}; refusing silent fallback"
        )
    session.disable_fallback()


def _shape_signature(shape: list[object]) -> tuple[int | None, ...]:
    return tuple(value if isinstance(value, int) else None for value in shape)


def _resolved_shape(shape: list[object], batch: int = 1) -> tuple[int, ...]:
    return tuple(value if isinstance(value, int) else batch for value in shape)


def _validate_tensor_contract(
    session: ort.InferenceSession,
    metadata: dict[str, object],
) -> None:
    feature = metadata["feature"]
    model = metadata["model"]
    if not isinstance(feature, dict) or not isinstance(model, dict):
        raise ValueError("ONNX sidecar tensor contract is invalid")
    feature_dim = int(model["feature_dim"])
    hidden_dim = int(model["hidden_dim"])
    recurrent_dim = int(model["recurrent_dim"])
    kernel_size = int(model["kernel_size"])
    dilations = [int(value) for value in model["dilations"]]
    expected_inputs: dict[str, tuple[int | None, ...]] = {
        "feature": (None, 1, feature_dim),
        "recurrent": (1, None, recurrent_dim),
    }
    expected_outputs: dict[str, tuple[int | None, ...]] = {
        "speech_logits": (None, 1),
        "next_recurrent": (1, None, recurrent_dim),
    }
    for index, dilation in enumerate(dilations):
        cache_shape = (None, hidden_dim, dilation * (kernel_size - 1))
        expected_inputs[f"cache_{index}"] = cache_shape
        expected_outputs[f"next_cache_{index}"] = cache_shape

    def actual(items: list[ort.NodeArg]) -> dict[str, tuple[int | None, ...]]:
        if any(item.type != "tensor(float)" for item in items):
            raise ValueError("ONNX tensor contract requires float32 tensors")
        return {item.name: _shape_signature(item.shape) for item in items}

    actual_inputs = actual(session.get_inputs())
    actual_outputs = actual(session.get_outputs())
    if actual_inputs != expected_inputs or actual_outputs != expected_outputs:
        raise ValueError(
            "ONNX tensor contract does not match sidecar metadata: "
            f"inputs={actual_inputs}, outputs={actual_outputs}"
        )


class OnnxStreamingVadModel:
    """Process-level model owner.

    Construct this once when a worker starts. Calls created with ``new_stream``
    share the immutable ONNX session and own only small feature/recurrent state.
    """

    def __init__(
        self,
        model_path: str | Path,
        threads: int = 1,
        providers: Sequence[ProviderSpec] | None = None,
    ) -> None:
        self.path = Path(model_path)
        sidecar_path = self.path.with_suffix(".json")
        validate_sidecar_metadata(sidecar_path)
        with sidecar_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        requested_providers = _validated_providers(providers)
        self.session = ort.InferenceSession(
            str(self.path),
            sess_options=options,
            providers=requested_providers,
        )
        expected_provider = _provider_name(requested_providers[0])
        _require_active_provider(self.session, expected_provider)
        self.execution_provider = expected_provider
        self.device_type = "cuda" if self.execution_provider in _CUDA_PROVIDERS else "cpu"
        _validate_tensor_contract(self.session, metadata)
        self.feature_config = FeatureConfig(**metadata["feature"])
        self.detector_config = DetectorConfig(**metadata.get("detector", {}))
        self.detector_config.validate()
        self.frontend = NumpyCausalFeatureExtractor(self.feature_config)
        self.input_shapes = {item.name: item.shape for item in self.session.get_inputs()}
        self.state_output_names = {
            output.name: output.name.removeprefix("next_")
            for output in self.session.get_outputs()
            if output.name.startswith("next_")
        }
        self._state_input_names = [name for name in self.input_shapes if name != "feature"]
        self._output_shapes = {
            output.name: output.shape
            for output in self.session.get_outputs()
        }

    @classmethod
    def load_bundled(
        cls,
        *,
        threads: int = 1,
        providers: Sequence[ProviderSpec] | None = None,
    ) -> OnnxStreamingVadModel:
        """Load the packaged streaming model once for process-level reuse."""

        if cls is OnnxStreamingVadModel:
            if providers is None:
                return _cached_bundled_model(threads)
            return cls(bundled_model_path(), threads=threads, providers=providers)
        if providers is None:
            return cls(bundled_model_path(), threads=threads)
        return cls(bundled_model_path(), threads=threads, providers=providers)

    def new_stream(
        self,
        detector_config: DetectorConfig | None = None,
    ) -> OnnxVadStream:
        return OnnxVadStream(self, detector_config or self.detector_config)


class OnnxVadStream:
    def __init__(self, owner: OnnxStreamingVadModel, detector_config: DetectorConfig) -> None:
        self.owner = owner
        self.features = NumpyStreamingFeatureExtractor(owner.frontend)
        self.detector = HysteresisDetector(
            detector_config,
            owner.feature_config.hop_ms / 1_000,
        )
        self._initialize_buffers()
        self.reset()

    def _current_state(self) -> dict[str, np.ndarray] | dict[str, ort.OrtValue]:
        if self.owner.device_type == "cpu":
            return self._state_buffers[self._active_state]
        return self._state_values[self._active_state]

    def _initialize_buffers(self) -> None:
        initial_state = {
            name: np.zeros(_resolved_shape(shape), dtype=np.float32)
            for name, shape in self.owner.input_shapes.items()
            if name != "feature"
        }
        self._state_buffers = [
            initial_state,
            {name: value.copy() for name, value in initial_state.items()},
        ]
        self._feature_buffer = np.zeros(
            (1, 1, self.owner.feature_config.feature_dim),
            dtype=np.float32,
        )
        self._logit_buffers = [
            np.zeros(_resolved_shape(self.owner._output_shapes["speech_logits"]), dtype=np.float32),
            np.zeros(_resolved_shape(self.owner._output_shapes["speech_logits"]), dtype=np.float32),
        ]
        self._feature_value = ort.OrtValue.ortvalue_from_numpy(
            self._feature_buffer,
            self.owner.device_type,
            0,
        )
        self._state_values = [
            {
                name: ort.OrtValue.ortvalue_from_numpy(value, self.owner.device_type, 0)
                for name, value in states.items()
            }
            for states in self._state_buffers
        ]
        self._logit_values = [
            ort.OrtValue.ortvalue_from_numpy(value, "cpu", 0) for value in self._logit_buffers
        ]
        self._bindings = []
        for active in (0, 1):
            binding = self.owner.session.io_binding()
            binding.bind_ortvalue_input("feature", self._feature_value)
            for name in self.owner._state_input_names:
                binding.bind_ortvalue_input(name, self._state_values[active][name])
            binding.bind_ortvalue_output("speech_logits", self._logit_values[active])
            for output_name, state_name in self.owner.state_output_names.items():
                binding.bind_ortvalue_output(
                    output_name,
                    self._state_values[1 - active][state_name],
                )
            self._bindings.append(binding)

    def reset(self) -> None:
        self.features.reset()
        self.detector.reset()
        self._feature_buffer.fill(0.0)
        for states in self._state_buffers:
            for value in states.values():
                value.fill(0.0)
        for value in self._logit_buffers:
            value.fill(0.0)
        if self.owner.device_type != "cpu":
            self._feature_value.update_inplace(self._feature_buffer)
            for states, values in zip(self._state_buffers, self._state_values, strict=True):
                for name, value in states.items():
                    values[name].update_inplace(value)
        self._active_state = 0
        self.state = self._current_state()

    def _step(self, feature: np.ndarray) -> float:
        np.copyto(self._feature_buffer[0, 0], feature)
        if self.owner.device_type != "cpu":
            self._feature_value.update_inplace(self._feature_buffer)
        self.owner.session.run_with_iobinding(self._bindings[self._active_state])
        probability = float(1.0 / (1.0 + np.exp(-self._logit_buffers[self._active_state].item())))
        self._active_state = 1 - self._active_state
        self.state = self._current_state()
        return probability

    def push(self, audio: object) -> tuple[np.ndarray, list[VadEvent]]:
        features = self.features.push(audio)
        probabilities: list[float] = []
        events: list[VadEvent] = []
        for feature in features:
            probability = self._step(feature)
            probabilities.append(probability)
            events.extend(self.detector.update(probability))
        return np.asarray(probabilities, dtype=np.float32), events


@lru_cache(maxsize=4)
def _cached_bundled_model(threads: int) -> OnnxStreamingVadModel:
    return OnnxStreamingVadModel(bundled_model_path(), threads=threads)
