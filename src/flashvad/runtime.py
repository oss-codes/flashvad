from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .assets import bundled_model_path
from .config import DetectorConfig, FeatureConfig
from .detector import HysteresisDetector, VadEvent
from .numpy_features import NumpyCausalFeatureExtractor, NumpyStreamingFeatureExtractor


class OnnxStreamingVadModel:
    """Process-level model owner.

    Construct this once when a worker starts. Calls created with ``new_stream``
    share the immutable ONNX session and own only small feature/recurrent state.
    """

    def __init__(self, model_path: str | Path, threads: int = 1) -> None:
        self.path = Path(model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(self.path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        with self.path.with_suffix(".json").open(encoding="utf-8") as handle:
            metadata = json.load(handle)
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

    @classmethod
    def load_bundled(cls, *, threads: int = 1) -> OnnxStreamingVadModel:
        """Load the packaged streaming model once for process-level reuse."""

        if cls is OnnxStreamingVadModel:
            return _cached_bundled_model(threads)
        return cls(bundled_model_path(), threads=threads)

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
        self.reset()

    def reset(self) -> None:
        self.features.reset()
        self.detector.reset()
        self.state = {
            name: np.zeros(
                [1 if isinstance(dimension, str) else dimension for dimension in shape],
                dtype=np.float32,
            )
            for name, shape in self.owner.input_shapes.items()
            if name != "feature"
        }

    def _step(self, feature: np.ndarray) -> float:
        inputs = {
            "feature": feature.reshape(1, 1, -1).astype(np.float32, copy=False),
            **self.state,
        }
        outputs = self.owner.session.run(None, inputs)
        probability = float(1.0 / (1.0 + np.exp(-outputs[0].item())))
        self.state = {
            self.owner.state_output_names[metadata.name]: value
            for metadata, value in zip(
                self.owner.session.get_outputs(),
                outputs,
                strict=True,
            )
            if metadata.name in self.owner.state_output_names
        }
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
