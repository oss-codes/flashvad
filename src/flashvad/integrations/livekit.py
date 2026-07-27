from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np

try:
    from livekit import agents, rtc
    from livekit.agents.utils.audio import AudioByteStream
except ImportError as exc:  # pragma: no cover - exercised without the optional extra
    raise ImportError(
        "LiveKit support requires `pip install 'flashvad[livekit]'`"
    ) from exc

from ..native_runtime import MacosNativeVadModel, MacosNativeVadStream
from ..runtime import OnnxStreamingVadModel, OnnxVadStream

VadOwner = OnnxStreamingVadModel | MacosNativeVadModel
VadStream = OnnxVadStream | MacosNativeVadStream


class FlashVadLiveKit(agents.vad.VAD):
    """LiveKit VAD that shares one process-level owner across per-call streams."""

    def __init__(
        self,
        owner: VadOwner,
        *,
        prefix_padding_duration: float = 0.3,
        max_buffered_speech: float = 60.0,
    ) -> None:
        if prefix_padding_duration < 0:
            raise ValueError("prefix_padding_duration must be non-negative")
        if max_buffered_speech <= 0:
            raise ValueError("max_buffered_speech must be positive")
        super().__init__(capabilities=agents.vad.VADCapabilities(update_interval=0.01))
        self._owner = owner
        self._prefix_padding_duration = prefix_padding_duration
        self._max_buffered_speech = max_buffered_speech

    @classmethod
    def load(
        cls,
        model_path: str | Path | None = None,
        *,
        threads: int = 1,
        prefix_padding_duration: float = 0.3,
        max_buffered_speech: float = 60.0,
    ) -> FlashVadLiveKit:
        """Load once during LiveKit worker prewarm, not once per call."""
        return cls(
            (
                OnnxStreamingVadModel(model_path, threads=threads)
                if model_path is not None
                else OnnxStreamingVadModel.load_bundled(threads=threads)
            ),
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
        )

    @classmethod
    def load_native(
        cls,
        library_path: str | Path | None = None,
        *,
        prefix_padding_duration: float = 0.3,
        max_buffered_speech: float = 60.0,
    ) -> FlashVadLiveKit:
        """Use the embedded Accelerate runtime for lowest measured macOS latency."""
        return cls(
            (
                MacosNativeVadModel(library_path)
                if library_path is not None
                else MacosNativeVadModel.load_bundled()
            ),
            prefix_padding_duration=prefix_padding_duration,
            max_buffered_speech=max_buffered_speech,
        )

    @property
    def model(self) -> str:
        return "flashvad-v0.1"

    @property
    def provider(self) -> str:
        if isinstance(self._owner, MacosNativeVadModel):
            return "FlashVAD Accelerate"
        return "FlashVAD ONNX"

    def stream(self) -> FlashVadLiveKitStream:
        return FlashVadLiveKitStream(
            self,
            self._owner.new_stream(),
            prefix_padding_duration=self._prefix_padding_duration,
            max_buffered_speech=self._max_buffered_speech,
        )


class FlashVadLiveKitStream(agents.vad.VADStream):
    _SAMPLE_RATE = 16_000
    _HOP_SAMPLES = 160

    def __init__(
        self,
        vad: FlashVadLiveKit,
        stream: VadStream,
        *,
        prefix_padding_duration: float,
        max_buffered_speech: float,
    ) -> None:
        self._stream = stream
        self._prefix_hops = round(prefix_padding_duration * 100)
        self._max_speech_hops = round(max_buffered_speech * 100)
        super().__init__(vad)

    def _reset_runtime(self) -> None:
        self._stream.reset()
        self._input_sample_rate = 0
        self._resampler: rtc.AudioResampler | None = None
        self._chunker = AudioByteStream(
            sample_rate=self._SAMPLE_RATE,
            num_channels=1,
            samples_per_channel=self._HOP_SAMPLES,
        )
        self._prefix: deque[rtc.AudioFrame] = deque(maxlen=self._prefix_hops or 1)
        self._speech_frames: list[rtc.AudioFrame] = []
        self._speaking = False
        self._speech_duration = 0.0
        self._silence_duration = 0.0
        self._sample_index = 0

    def _combined_speech(self) -> list[rtc.AudioFrame]:
        if not self._speech_frames:
            return []
        return [rtc.combine_audio_frames(self._speech_frames)]

    def _process_hop(self, frame: rtc.AudioFrame) -> None:
        started = time.perf_counter()
        audio = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)
        audio *= np.float32(1.0 / 32768.0)
        probabilities, vad_events = self._stream.push(audio)
        inference_duration = time.perf_counter() - started
        if probabilities.size != 1:
            raise RuntimeError("FlashVAD LiveKit adapter expected one probability per 10 ms hop")

        probability = float(probabilities[0])
        hop_seconds = self._HOP_SAMPLES / self._SAMPLE_RATE
        self._sample_index += self._HOP_SAMPLES
        if self._speaking:
            self._speech_duration += hop_seconds
            if len(self._speech_frames) < self._max_speech_hops:
                self._speech_frames.append(frame)
        else:
            self._silence_duration += hop_seconds
            if self._prefix_hops:
                self._prefix.append(frame)

        self._event_ch.send_nowait(
            agents.vad.VADEvent(
                type=agents.vad.VADEventType.INFERENCE_DONE,
                samples_index=self._sample_index,
                timestamp=self._sample_index / self._SAMPLE_RATE,
                speech_duration=self._speech_duration,
                silence_duration=self._silence_duration,
                frames=[frame],
                probability=probability,
                inference_duration=inference_duration,
                speaking=self._speaking,
                raw_accumulated_speech=(
                    self._stream.detector.above_count * hop_seconds
                ),
                raw_accumulated_silence=(
                    self._stream.detector.below_count * hop_seconds
                ),
            )
        )

        for event in vad_events:
            if event.kind == "speech_start":
                self._speaking = True
                self._speech_duration = 0.0
                self._silence_duration = 0.0
                self._speech_frames = list(self._prefix)
                self._event_ch.send_nowait(
                    agents.vad.VADEvent(
                        type=agents.vad.VADEventType.START_OF_SPEECH,
                        samples_index=self._sample_index,
                        timestamp=self._sample_index / self._SAMPLE_RATE,
                        speech_duration=0.0,
                        silence_duration=0.0,
                        frames=self._combined_speech(),
                        speaking=True,
                    )
                )
            elif event.kind == "speech_end":
                self._speaking = False
                self._event_ch.send_nowait(
                    agents.vad.VADEvent(
                        type=agents.vad.VADEventType.END_OF_SPEECH,
                        samples_index=self._sample_index,
                        timestamp=self._sample_index / self._SAMPLE_RATE,
                        speech_duration=self._speech_duration,
                        silence_duration=self._silence_duration,
                        frames=self._combined_speech(),
                        speaking=False,
                    )
                )
                self._speech_duration = 0.0
                self._silence_duration = 0.0
                self._prefix.clear()
                if self._prefix_hops:
                    self._prefix.append(frame)
                self._speech_frames = []

    async def _main_task(self) -> None:
        self._reset_runtime()
        async for item in self._input_ch:
            if isinstance(item, self._FlushSentinel):
                self._reset_runtime()
                continue
            if not isinstance(item, rtc.AudioFrame):
                continue
            if item.num_channels != 1:
                raise ValueError("FlashVAD LiveKit adapter accepts mono audio only")

            if not self._input_sample_rate:
                self._input_sample_rate = item.sample_rate
                if item.sample_rate != self._SAMPLE_RATE:
                    self._resampler = rtc.AudioResampler(
                        input_rate=item.sample_rate,
                        output_rate=self._SAMPLE_RATE,
                        num_channels=1,
                        quality=rtc.AudioResamplerQuality.QUICK,
                    )
            elif self._input_sample_rate != item.sample_rate:
                raise ValueError("LiveKit audio sample rate changed within one stream")

            inference_frames = (
                self._resampler.push(item) if self._resampler is not None else [item]
            )
            for inference_frame in inference_frames:
                for hop in self._chunker.push(inference_frame.data):
                    self._process_hop(hop)


__all__ = ["FlashVadLiveKit", "FlashVadLiveKitStream"]
