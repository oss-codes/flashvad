from __future__ import annotations

import platform

import numpy as np
import pytest

from flashvad.telephony import (
    CausalLinearResampler8To16,
    TelephonyVadStream,
    decode_g711_alaw,
    decode_g711_ulaw,
    decode_pcm16,
)


def test_g711_ulaw_decodes_canonical_codewords_to_normalized_float32() -> None:
    payload = bytes([0x00, 0x01, 0x7F, 0x80, 0xFF, 0xD5])
    expected = np.asarray(
        [-32124, -31100, 0, 32124, 0, 716],
        dtype=np.float32,
    ) / np.float32(32768.0)

    decoded = decode_g711_ulaw(payload)

    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, expected)
    assert np.isfinite(decoded).all()


def test_g711_alaw_decodes_canonical_codewords_to_normalized_float32() -> None:
    payload = bytes([0x00, 0x01, 0x55, 0xD5, 0xFF, 0x80])
    expected = np.asarray(
        [-5504, -5248, -8, 8, 848, 5504],
        dtype=np.float32,
    ) / np.float32(32768.0)

    decoded = decode_g711_alaw(payload)

    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(decoded, expected)
    assert np.isfinite(decoded).all()


def test_causal_resampler_is_chunk_invariant_and_has_a_causal_edge() -> None:
    samples = np.asarray([-1.0, 0.0, 0.75, 1.0, -0.25], dtype=np.float32)
    expected = np.asarray(
        [-1.0, -1.0, -0.5, 0.0, 0.375, 0.75, 0.875, 1.0, 0.375, -0.25],
        dtype=np.float32,
    )

    whole = CausalLinearResampler8To16().push(samples)
    streamed_resampler = CausalLinearResampler8To16()
    streamed = np.concatenate(
        [
            streamed_resampler.push(samples[:1]),
            streamed_resampler.push(samples[1:3]),
            streamed_resampler.push(samples[3:]),
        ]
    )

    assert whole.dtype == np.float32
    np.testing.assert_array_equal(whole, expected)
    np.testing.assert_array_equal(streamed, whole)


def test_resampler_reset_starts_a_new_causal_edge() -> None:
    resampler = CausalLinearResampler8To16()

    resampler.push(np.asarray([1.0], dtype=np.float32))
    resampler.push(np.asarray([3.0], dtype=np.float32))
    resampler.reset()

    np.testing.assert_array_equal(
        resampler.push(np.asarray([-2.0], dtype=np.float32)),
        np.asarray([-2.0, -2.0], dtype=np.float32),
    )


def test_pcm16_requires_complete_little_endian_samples() -> None:
    payload = np.asarray([-32768, -1, 0, 32767], dtype="<i2").tobytes()

    decoded = decode_pcm16(payload)

    assert decoded.dtype == np.float32
    np.testing.assert_array_equal(
        decoded,
        np.asarray([-1.0, -1 / 32768, 0.0, 32767 / 32768], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="even"):
        decode_pcm16(b"\x00")


def test_decoders_reject_non_bytes_payloads() -> None:
    with pytest.raises(TypeError, match="bytes-like"):
        decode_g711_ulaw([0, 1, 2])


class _FakeDownstream:
    def __init__(self) -> None:
        self.received: list[np.ndarray] = []
        self.reset_count = 0

    def push(self, audio: np.ndarray) -> tuple[str, int]:
        self.received.append(audio.copy())
        return ("accepted", len(audio))

    def reset(self) -> None:
        self.reset_count += 1


@pytest.mark.parametrize(
    ("codec", "payload", "decoded"),
    [
        (
            "PCMU",
            bytes([0x80, 0xD5]),
            np.asarray([32124, 716], dtype=np.float32) / np.float32(32768),
        ),
        (
            "PCMA",
            bytes([0xD5, 0xFF]),
            np.asarray([8, 848], dtype=np.float32) / np.float32(32768),
        ),
        (
            "PCM16",
            np.asarray([1000, -2000], dtype="<i2").tobytes(),
            np.asarray([1000, -2000], dtype=np.float32) / np.float32(32768),
        ),
    ],
)
def test_telephony_stream_forwards_2x_audio_and_returns_downstream_result(
    codec: str,
    payload: bytes,
    decoded: np.ndarray,
) -> None:
    downstream = _FakeDownstream()
    stream = TelephonyVadStream(downstream, codec)

    result = stream.push(payload)

    expected = np.empty(decoded.size * 2, dtype=np.float32)
    expected[0] = decoded[0]
    expected[1] = decoded[0]
    expected[2::2] = (decoded[:-1] + decoded[1:]) * np.float32(0.5)
    expected[3::2] = decoded[1:]
    assert result == ("accepted", decoded.size * 2)
    assert len(downstream.received) == 1
    assert downstream.received[0].dtype == np.float32
    np.testing.assert_array_equal(downstream.received[0], expected)


def test_telephony_stream_keeps_interpolation_state_across_packets_and_reset() -> None:
    downstream = _FakeDownstream()
    stream = TelephonyVadStream(downstream, "pcm16")
    first = np.asarray([1000], dtype="<i2").tobytes()
    second = np.asarray([3000], dtype="<i2").tobytes()

    stream.push(first)
    stream.push(second)
    np.testing.assert_array_equal(
        downstream.received[1],
        np.asarray([2000, 3000], dtype=np.float32) / np.float32(32768),
    )

    stream.reset()
    assert downstream.reset_count == 1
    stream.push(np.asarray([-1000], dtype="<i2").tobytes())
    np.testing.assert_array_equal(
        downstream.received[-1],
        np.asarray([-1000, -1000], dtype=np.float32) / np.float32(32768),
    )


def test_telephony_stream_validates_codec_and_downstream() -> None:
    with pytest.raises(ValueError, match="codec"):
        TelephonyVadStream(_FakeDownstream(), "opus")
    with pytest.raises(TypeError, match="push"):
        TelephonyVadStream(object(), "PCMU")


def test_telephony_stream_rejects_malformed_pcm16_before_forwarding() -> None:
    downstream = _FakeDownstream()
    stream = TelephonyVadStream(downstream, "PCM16")

    with pytest.raises(ValueError, match="even"):
        stream.push(b"\x01")
    assert downstream.received == []


def test_telephony_stream_loads_the_bundled_onnx_model() -> None:
    stream = TelephonyVadStream.load_onnx("pcmu")

    probabilities, events = stream.push(bytes([0xFF] * 80))

    assert probabilities.shape == (1,)
    assert 0.0 <= float(probabilities[0]) <= 1.0
    assert isinstance(events, list)


@pytest.mark.skipif(platform.system() != "Darwin", reason="requires Apple Accelerate")
def test_telephony_stream_loads_the_bundled_native_model() -> None:
    stream = TelephonyVadStream.load_native("pcmu")

    probabilities, events = stream.push(bytes([0xFF] * 80))

    assert probabilities.shape == (1,)
    assert 0.0 <= float(probabilities[0]) <= 1.0
    assert isinstance(events, list)
