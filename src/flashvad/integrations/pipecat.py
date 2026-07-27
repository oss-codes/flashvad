from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams, VADState
except ImportError as exc:  # pragma: no cover - exercised without the optional extra
    raise ImportError(
        "Pipecat support requires `pip install 'flashvad[pipecat]'`"
    ) from exc

from ..native_runtime import MacosNativeVadModel
from ..runtime import OnnxStreamingVadModel
from ..telephony import CausalLinearResampler8To16

VadOwner = OnnxStreamingVadModel | MacosNativeVadModel


class FlashVadPipecatAnalyzer(VADAnalyzer):
    """Pipecat analyzer using 10 ms PCM16 frames and persistent model state."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        native_library_path: str | Path | None = None,
        owner: VadOwner | None = None,
        sample_rate: int | None = None,
        params: VADParams | None = None,
        threads: int = 1,
    ) -> None:
        provided = sum(
            value is not None
            for value in (model_path, native_library_path, owner)
        )
        if provided > 1:
            raise ValueError(
                "provide at most one of model_path, native_library_path, or owner"
            )
        if sample_rate is not None and sample_rate not in (8_000, 16_000):
            raise ValueError("FlashVAD Pipecat adapter supports 8 kHz or 16 kHz mono PCM")
        default_params = VADParams(
            confidence=0.8,
            start_secs=0.03,
            stop_secs=0.04,
            min_volume=0.0,
        )
        super().__init__(sample_rate=sample_rate, params=params or default_params)
        if owner is not None:
            self._owner = owner
        elif native_library_path is not None:
            self._owner = MacosNativeVadModel(native_library_path)
        elif model_path is not None:
            self._owner = OnnxStreamingVadModel(model_path, threads=threads)
        else:
            self._owner = OnnxStreamingVadModel.load_bundled(threads=threads)
        self._stream = self._owner.new_stream()
        self._resampler = CausalLinearResampler8To16()
        self._closed = False

    @classmethod
    def load_native(
        cls,
        library_path: str | Path | None = None,
        *,
        sample_rate: int | None = None,
        params: VADParams | None = None,
    ) -> FlashVadPipecatAnalyzer:
        """Load the cached bundled Accelerate runtime, or an explicit library."""

        owner = (
            MacosNativeVadModel(library_path)
            if library_path is not None
            else MacosNativeVadModel.load_bundled()
        )
        return cls(owner=owner, sample_rate=sample_rate, params=params)

    def set_sample_rate(self, sample_rate: int) -> None:
        effective_sample_rate = self._init_sample_rate or sample_rate
        if effective_sample_rate not in (8_000, 16_000):
            raise ValueError("FlashVAD Pipecat adapter supports 8 kHz or 16 kHz mono PCM")
        super().set_sample_rate(sample_rate)
        self.reset()

    def num_frames_required(self) -> int:
        if self.sample_rate not in (8_000, 16_000):
            return 160
        return self.sample_rate // 100

    def reset(self) -> None:
        self._stream.reset()
        self._resampler.reset()
        self._vad_buffer = b""
        self._vad_state = VADState.QUIET
        self._vad_starting_count = 0
        self._vad_stopping_count = 0
        self._prev_volume = 0

    def voice_confidence(self, buffer: bytes) -> float:
        samples = np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
        samples *= np.float32(1.0 / 32768.0)
        if self.sample_rate == 8_000:
            samples = self._resampler.push(samples)
        probabilities, _ = self._stream.push(samples)
        if probabilities.size != 1:
            raise RuntimeError("FlashVAD Pipecat adapter expected one 10 ms frame")
        return float(probabilities[0])

    async def cleanup(self) -> None:
        if self._closed:
            return
        self.reset()
        await super().cleanup()
        self._executor.shutdown(wait=True, cancel_futures=True)
        self._closed = True


__all__ = ["FlashVadPipecatAnalyzer"]
