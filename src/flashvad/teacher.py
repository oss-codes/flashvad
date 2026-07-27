from __future__ import annotations

from pathlib import Path

import numpy as np


def blend_teacher_probabilities(
    labels: np.ndarray,
    teacher_probabilities: np.ndarray,
    *,
    weight: float,
) -> np.ndarray:
    """Blend confident teacher predictions while retaining uncertain weak labels."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("teacher weight must be between zero and one")
    targets = np.asarray(labels, dtype=np.float32)
    teacher = np.asarray(teacher_probabilities, dtype=np.float32)
    if targets.shape != teacher.shape:
        raise ValueError("labels and teacher probabilities must have the same shape")
    teacher = np.clip(teacher, 0.0, 1.0)
    confidence = np.abs(teacher - 0.5) * 2.0
    effective_weight = weight * confidence
    return (
        (1.0 - effective_weight) * targets + effective_weight * teacher
    ).astype(np.float32)


def interpolate_probabilities(
    probabilities: np.ndarray,
    *,
    source_hop_ms: float,
    target_hop_ms: float,
    target_frames: int,
) -> np.ndarray:
    source = np.asarray(probabilities, dtype=np.float32).reshape(-1)
    if source.size == 0:
        raise ValueError("teacher probabilities cannot be empty")
    if source_hop_ms <= 0 or target_hop_ms <= 0 or target_frames < 1:
        raise ValueError("hop sizes and target frame count must be positive")
    source_times = (np.arange(source.size) + 1) * source_hop_ms
    target_times = (np.arange(target_frames) + 1) * target_hop_ms
    return np.interp(
        target_times,
        source_times,
        source,
        left=float(source[0]),
        right=float(source[-1]),
    ).astype(np.float32)


class SileroOnnxTeacher:
    """Minimal stateful adapter for official Silero 8/16 kHz ONNX models."""

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

    def predict(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int = 16_000,
        target_hop_ms: float = 10.0,
    ) -> np.ndarray:
        if sample_rate not in (8_000, 16_000):
            raise ValueError("Silero teacher supports 8000 or 16000 Hz")
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise ValueError("audio cannot be empty")

        frame_samples = 512 if sample_rate == 16_000 else 256
        context_samples = 64 if sample_rate == 16_000 else 32
        target_frames = int(
            np.ceil(samples.size / (sample_rate * target_hop_ms / 1_000))
        )
        remainder = samples.size % frame_samples
        if remainder:
            samples = np.pad(samples, (0, frame_samples - remainder))

        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, context_samples), dtype=np.float32)
        probabilities: list[float] = []
        for start in range(0, samples.size, frame_samples):
            chunk = samples[start : start + frame_samples][None, :]
            model_input = np.concatenate((context, chunk), axis=1)
            output, state = self.session.run(
                None,
                {
                    "input": model_input,
                    "state": state,
                    "sr": np.array(sample_rate, dtype=np.int64),
                },
            )
            probabilities.append(float(np.asarray(output).reshape(-1)[0]))
            context = model_input[:, -context_samples:]

        return interpolate_probabilities(
            np.asarray(probabilities, dtype=np.float32),
            source_hop_ms=frame_samples / sample_rate * 1_000,
            target_hop_ms=target_hop_ms,
            target_frames=target_frames,
        )


class FireRedOnnxTeacher:
    """Adapter for the official FireRedVAD non-streaming ONNX teacher."""

    def __init__(
        self,
        model_path: str | Path,
        cmvn_path: str | Path,
    ) -> None:
        import kaldi_native_fbank as knf
        import kaldiio
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        stats = np.asarray(kaldiio.load_mat(str(cmvn_path)), dtype=np.float64)
        if stats.shape != (2, 81):
            raise ValueError(f"expected FireRed CMVN shape (2, 81), got {stats.shape}")
        count = float(stats[0, -1])
        if count < 1:
            raise ValueError("FireRed CMVN has an invalid sample count")
        self.means = (stats[0, :-1] / count).astype(np.float32)
        variance = stats[1, :-1] / count - self.means.astype(np.float64) ** 2
        self.inverse_std = (1.0 / np.sqrt(np.maximum(variance, 1e-20))).astype(
            np.float32
        )
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = 16_000
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.dither = 0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = 80
        opts.mel_opts.debug_mel = False
        self.fbank_options = opts

    def predict(
        self,
        audio: np.ndarray,
        *,
        sample_rate: int = 16_000,
        target_hop_ms: float = 10.0,
    ) -> np.ndarray:
        import kaldi_native_fbank as knf

        if sample_rate != 16_000:
            raise ValueError("FireRed teacher supports 16000 Hz")
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            raise ValueError("audio cannot be empty")
        pcm = np.clip(np.rint(samples * 32_768.0), -32_768, 32_767).astype(
            np.int16
        )
        fbank = knf.OnlineFbank(self.fbank_options)
        fbank.accept_waveform(sample_rate, pcm.tolist())
        frames = [fbank.get_frame(index) for index in range(fbank.num_frames_ready)]
        target_frames = int(
            np.ceil(samples.size / (sample_rate * target_hop_ms / 1_000))
        )
        if not frames:
            return np.zeros(target_frames, dtype=np.float32)
        features = np.vstack(frames).astype(np.float32)
        features = (features - self.means) * self.inverse_std
        probabilities = self.session.run(
            None,
            {"feat": features[None, :, :]},
        )[0].reshape(-1)
        return interpolate_probabilities(
            probabilities,
            source_hop_ms=10.0,
            target_hop_ms=target_hop_ms,
            target_frames=target_frames,
        )
