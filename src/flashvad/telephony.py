"""Low-level adapters for 8 kHz telephone audio.

Telephone gateways commonly deliver one of the G.711 companded codecs (PCMU
or PCMA), or signed little-endian PCM16.  The decoder functions in this
module deliberately operate on byte payloads and return ordinary normalized
``float32`` samples.  ``CausalLinearResampler8To16`` then performs the small
amount of stateful interpolation needed to feed a 16 kHz streaming VAD.

No codec or resampling dependency is needed at runtime; the implementation is
just NumPy and keeps all state local to a stream.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .config import DetectorConfig

_NORMALIZATION = np.float32(1.0 / 32768.0)


def _payload_bytes(payload: object) -> bytes:
    """Return a stable bytes copy of a one-shot bytes-like payload."""

    try:
        return memoryview(payload).tobytes()
    except (TypeError, ValueError) as exc:
        raise TypeError("telephone payload must be bytes-like") from exc


def _payload_uint8(payload: object) -> np.ndarray:
    return np.frombuffer(_payload_bytes(payload), dtype=np.uint8)


def _expand_g711_ulaw(codes: np.ndarray) -> np.ndarray:
    codes = codes.astype(np.int32)
    inverted = np.bitwise_xor(codes, 0xFF)
    magnitude = ((inverted & 0x0F) << 3) + 0x84
    magnitude <<= (inverted & 0x70) >> 4
    pcm = np.where((inverted & 0x80) != 0, 0x84 - magnitude, magnitude - 0x84)
    return (pcm.astype(np.float32) * _NORMALIZATION).astype(np.float32, copy=False)


def _expand_g711_alaw(codes: np.ndarray) -> np.ndarray:
    codes = np.bitwise_xor(codes, np.uint8(0x55)).astype(np.int32)
    segment = (codes & 0x70) >> 4
    low = (codes & 0x0F) << 4

    # Avoid evaluating a negative shift for segment zero while retaining a
    # fully vectorized expansion for every other segment.
    magnitude = low + 8
    nonzero_segment = segment != 0
    magnitude[nonzero_segment] = (
        (low[nonzero_segment] + 0x108) << (segment[nonzero_segment] - 1)
    )
    pcm = np.where((codes & 0x80) != 0, magnitude, -magnitude)
    return (pcm.astype(np.float32) * _NORMALIZATION).astype(np.float32, copy=False)


_G711_ULAW_TABLE = _expand_g711_ulaw(np.arange(256, dtype=np.uint8))
_G711_ALAW_TABLE = _expand_g711_alaw(np.arange(256, dtype=np.uint8))


def decode_g711_ulaw(payload: object) -> np.ndarray:
    """Decode a PCMU (G.711 mu-law) payload to normalized ``float32``.

    The integer expansion follows the ITU-T G.711 bit layout exactly.  A
    16-bit signed PCM range is normalized by 32768, so every possible
    codeword maps to a finite value in approximately ``[-0.982, 0.982]``.
    """

    return _G711_ULAW_TABLE[_payload_uint8(payload)]


def decode_g711_alaw(payload: object) -> np.ndarray:
    """Decode a PCMA (G.711 A-law) payload to normalized ``float32``."""

    return _G711_ALAW_TABLE[_payload_uint8(payload)]


def decode_pcm16(payload: object) -> np.ndarray:
    """Decode signed little-endian PCM16 telephone samples.

    PCM16 payloads must contain complete two-byte samples.  The returned
    values use the same ``[-1, 1)`` normalization as the G.711 decoders.
    """

    raw = _payload_bytes(payload)
    if len(raw) % 2:
        raise ValueError("PCM16 payload length must be an even number of bytes")
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    return (pcm * _NORMALIZATION).astype(np.float32, copy=False)


class CausalLinearResampler8To16:
    """Stateful causal 2x linear interpolation for 8 kHz samples.

    Every input sample produces exactly two output samples.  For the first
    sample the previous value is edge-held, yielding ``[x0, x0]``.  Each later
    sample ``x`` yields ``[(previous + x) / 2, x]``.  Keeping the previous
    input sample makes this result independent of packet/chunk boundaries.
    """

    def __init__(self) -> None:
        self._previous: float | None = None

    def reset(self) -> None:
        """Forget the previous sample so the next push starts a new call."""

        self._previous = None

    def push(self, audio: object) -> np.ndarray:
        """Interpolate a one-dimensional sample chunk into ``float32``."""

        try:
            samples = np.asarray(audio, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise TypeError("audio must be a one-dimensional numeric sequence") from exc
        if samples.ndim != 1:
            raise ValueError("audio must be one-dimensional")
        if samples.size and not np.isfinite(samples).all():
            raise ValueError("audio samples must be finite")
        if samples.size == 0:
            return np.empty(0, dtype=np.float32)

        previous = np.empty(samples.size, dtype=np.float32)
        previous[0] = samples[0] if self._previous is None else self._previous
        if samples.size > 1:
            previous[1:] = samples[:-1]
        midpoints = (previous + samples) * np.float32(0.5)

        output = np.empty(samples.size * 2, dtype=np.float32)
        output[0::2] = midpoints
        output[1::2] = samples
        self._previous = float(samples[-1])
        return output


def _normalize_codec(codec: object) -> str:
    if not isinstance(codec, str):
        raise ValueError("codec must be one of PCMU, PCMA, or PCM16")
    key = "".join(character for character in codec.strip().upper() if character.isalnum())
    aliases = {
        "PCMU": "PCMU",
        "ULAW": "PCMU",
        "MULAW": "PCMU",
        "G711ULAW": "PCMU",
        "PCMA": "PCMA",
        "ALAW": "PCMA",
        "G711ALAW": "PCMA",
        "PCM16": "PCM16",
        "L16": "PCM16",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError("codec must be one of PCMU, PCMA, or PCM16") from exc


class TelephonyVadStream:
    """Decode telephone packets and feed normalized 16 kHz audio downstream."""

    def __init__(self, downstream: Any, codec: str) -> None:
        push = getattr(downstream, "push", None)
        if not callable(push):
            raise TypeError("downstream must expose a callable push(audio) method")
        self.downstream = downstream
        self.codec = _normalize_codec(codec)
        self.resampler = CausalLinearResampler8To16()

    @classmethod
    def load_onnx(
        cls,
        codec: str,
        model_path: str | Path | None = None,
        *,
        threads: int = 1,
        detector_config: DetectorConfig | None = None,
    ) -> TelephonyVadStream:
        """Create a call stream from one cached bundled or explicit ONNX owner."""

        from .runtime import OnnxStreamingVadModel

        owner = (
            OnnxStreamingVadModel(model_path, threads=threads)
            if model_path is not None
            else OnnxStreamingVadModel.load_bundled(threads=threads)
        )
        return cls(owner.new_stream(detector_config), codec)

    @classmethod
    def load_native(
        cls,
        codec: str,
        library_path: str | Path | None = None,
        *,
        detector_config: DetectorConfig | None = None,
    ) -> TelephonyVadStream:
        """Create a call stream from the cached bundled macOS Accelerate owner."""

        from .native_runtime import MacosNativeVadModel

        owner = (
            MacosNativeVadModel(library_path)
            if library_path is not None
            else MacosNativeVadModel.load_bundled()
        )
        return cls(owner.new_stream(detector_config), codec)

    def reset(self) -> None:
        """Reset interpolation state and reset the downstream stream if supported."""

        self.resampler.reset()
        reset = getattr(self.downstream, "reset", None)
        if reset is not None:
            if not callable(reset):
                raise TypeError("downstream reset attribute must be callable")
            reset()

    def push(self, payload: object) -> Any:
        """Decode one packet, forward 16 kHz audio, and return downstream's result."""

        if self.codec == "PCMU":
            decoded = decode_g711_ulaw(payload)
        elif self.codec == "PCMA":
            decoded = decode_g711_alaw(payload)
        else:
            decoded = decode_pcm16(payload)
        return self.downstream.push(self.resampler.push(decoded))


__all__ = [
    "CausalLinearResampler8To16",
    "TelephonyVadStream",
    "decode_g711_alaw",
    "decode_g711_ulaw",
    "decode_pcm16",
]
