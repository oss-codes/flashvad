"""FlashVAD public API."""

from importlib import import_module
from typing import Any

from .assets import bundled_model_path
from .config import DetectorConfig, FeatureConfig, ModelConfig, ProjectConfig, TrainingConfig
from .detector import VadEvent

_LAZY_IMPORTS = {
    "CausalFeatureExtractor": (".features", "CausalFeatureExtractor"),
    "StreamingFeatureExtractor": (".features", "StreamingFeatureExtractor"),
    "FlashVad": (".model", "FlashVad"),
    "StreamingModelState": (".model", "StreamingModelState"),
    "StreamingVadEngine": (".streaming", "StreamingVadEngine"),
}

__all__ = [
    "CausalFeatureExtractor",
    "DetectorConfig",
    "FeatureConfig",
    "ModelConfig",
    "FlashVad",
    "ProjectConfig",
    "StreamingFeatureExtractor",
    "StreamingModelState",
    "StreamingVadEngine",
    "TrainingConfig",
    "VadEvent",
    "bundled_model_path",
]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})
