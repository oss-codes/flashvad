from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .benchmark import load_checkpoint
from .features import CausalFeatureExtractor


def _native_source_dir() -> Path:
    packaged = Path(__file__).resolve().parent / "_native" / "macos"
    if packaged.is_dir():
        return packaged
    repository = Path(__file__).resolve().parents[2] / "native" / "macos"
    if repository.is_dir():
        return repository
    raise RuntimeError("FlashVAD macOS native sources are missing from this installation")


def _first_line(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.splitlines()[0].strip()


def _c_float(value: float) -> str:
    if not np.isfinite(value):
        raise ValueError("native export does not support non-finite weights")
    return f"{float(np.float32(value)):.9e}f"


def _c_array(name: str, values: np.ndarray, *, storage: str = "float") -> str:
    flattened = np.asarray(values).reshape(-1)
    rows: list[str] = []
    width = 8
    for start in range(0, flattened.size, width):
        chunk = flattened[start : start + width]
        if storage == "int":
            rows.append("    " + ", ".join(str(int(value)) for value in chunk))
        else:
            rows.append("    " + ", ".join(_c_float(float(value)) for value in chunk))
    body = ",\n".join(rows)
    return f"const {storage} {name}[{flattened.size}] = {{\n{body}\n}};\n"


def _tensor(model, name: str) -> np.ndarray:
    return model.state_dict()[name].detach().cpu().contiguous().numpy()


def export_macos_native(
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> Path:
    checkpoint = Path(checkpoint_path)
    output = Path(output_dir)
    config, model = load_checkpoint(checkpoint)
    feature = config.feature
    architecture = config.model

    expected = {
        "sample_rate": 16_000,
        "frame_samples": 400,
        "hop_samples": 160,
        "n_fft": 512,
        "n_mels": 40,
        "feature_dim": 43,
        "kernel_size": 3,
    }
    actual = {
        "sample_rate": feature.sample_rate,
        "frame_samples": feature.frame_samples,
        "hop_samples": feature.hop_samples,
        "n_fft": feature.n_fft,
        "n_mels": feature.n_mels,
        "feature_dim": feature.feature_dim,
        "kernel_size": architecture.kernel_size,
    }
    mismatches = {
        key: (actual[key], wanted)
        for key, wanted in expected.items()
        if actual[key] != wanted
    }
    if mismatches:
        raise ValueError(f"unsupported native configuration: {mismatches}")
    if len(architecture.dilations) < 1:
        raise ValueError("native export requires at least one temporal block")

    frontend = CausalFeatureExtractor(feature)
    cache_sizes = [
        dilation * (architecture.kernel_size - 1)
        for dilation in architecture.dilations
    ]
    total_cache_floats = architecture.hidden_dim * sum(cache_sizes)
    output.mkdir(parents=True, exist_ok=True)

    header = f"""\
/*
 * Generated model parameters. Their licence follows the source checkpoint
 * and is not automatically covered by the FlashVAD repository source licence.
 */
#ifndef FLASHVAD_WEIGHTS_H
#define FLASHVAD_WEIGHTS_H

#define FV_SAMPLE_RATE {feature.sample_rate}
#define FV_FRAME_SAMPLES {feature.frame_samples}
#define FV_HOP_SAMPLES {feature.hop_samples}
#define FV_HISTORY_SAMPLES {feature.frame_samples - feature.hop_samples}
#define FV_N_FFT {feature.n_fft}
#define FV_POWER_BINS {feature.n_fft // 2 + 1}
#define FV_N_MELS {feature.n_mels}
#define FV_FEATURE_DIM {feature.feature_dim}
#define FV_HIDDEN_DIM {architecture.hidden_dim}
#define FV_RECURRENT_DIM {architecture.recurrent_dim}
#define FV_KERNEL_SIZE {architecture.kernel_size}
#define FV_BLOCK_COUNT {len(architecture.dilations)}
#define FV_TOTAL_CACHE_FLOATS {total_cache_floats}

extern const int fv_block_dilations[FV_BLOCK_COUNT];
extern const float fv_window[FV_FRAME_SAMPLES];
extern const float fv_mel_filterbank[FV_N_MELS * FV_POWER_BINS];
extern const float fv_input_norm_weight[FV_FEATURE_DIM];
extern const float fv_input_norm_bias[FV_FEATURE_DIM];
extern const float fv_input_projection_weight[FV_HIDDEN_DIM * FV_FEATURE_DIM];
extern const float fv_input_projection_bias[FV_HIDDEN_DIM];
extern const float fv_depthwise_weight[FV_BLOCK_COUNT * FV_HIDDEN_DIM * FV_KERNEL_SIZE];
extern const float fv_pointwise_weight[FV_BLOCK_COUNT * FV_HIDDEN_DIM * FV_HIDDEN_DIM];
extern const float fv_pointwise_bias[FV_BLOCK_COUNT * FV_HIDDEN_DIM];
extern const float fv_block_norm_weight[FV_BLOCK_COUNT * FV_HIDDEN_DIM];
extern const float fv_block_norm_bias[FV_BLOCK_COUNT * FV_HIDDEN_DIM];
extern const float fv_gru_weight_ih[3 * FV_RECURRENT_DIM * FV_HIDDEN_DIM];
extern const float fv_gru_weight_hh[3 * FV_RECURRENT_DIM * FV_RECURRENT_DIM];
extern const float fv_gru_bias_ih[3 * FV_RECURRENT_DIM];
extern const float fv_gru_bias_hh[3 * FV_RECURRENT_DIM];
extern const float fv_output_norm_weight[FV_RECURRENT_DIM];
extern const float fv_output_norm_bias[FV_RECURRENT_DIM];
extern const float fv_speech_head_weight[FV_RECURRENT_DIM];
extern const float fv_speech_head_bias[1];

#endif
"""
    (output / "flashvad_weights.h").write_text(header, encoding="utf-8")

    depthwise = np.concatenate(
        [_tensor(model, f"blocks.{index}.depthwise.weight") for index in range(len(cache_sizes))]
    )
    pointwise_weight = np.concatenate(
        [
            _tensor(model, f"blocks.{index}.pointwise.weight")
            for index in range(len(cache_sizes))
        ]
    )
    pointwise_bias = np.concatenate(
        [_tensor(model, f"blocks.{index}.pointwise.bias") for index in range(len(cache_sizes))]
    )
    block_norm_weight = np.concatenate(
        [_tensor(model, f"blocks.{index}.norm.weight") for index in range(len(cache_sizes))]
    )
    block_norm_bias = np.concatenate(
        [_tensor(model, f"blocks.{index}.norm.bias") for index in range(len(cache_sizes))]
    )
    arrays = [
        _c_array(
            "fv_block_dilations",
            np.asarray(architecture.dilations),
            storage="int",
        ),
        _c_array("fv_window", frontend.window.detach().cpu().numpy()),
        _c_array(
            "fv_mel_filterbank",
            frontend.mel_filterbank.detach().cpu().numpy(),
        ),
        _c_array("fv_input_norm_weight", _tensor(model, "input_norm.weight")),
        _c_array("fv_input_norm_bias", _tensor(model, "input_norm.bias")),
        _c_array(
            "fv_input_projection_weight",
            _tensor(model, "input_projection.weight"),
        ),
        _c_array(
            "fv_input_projection_bias",
            _tensor(model, "input_projection.bias"),
        ),
        _c_array("fv_depthwise_weight", depthwise),
        _c_array("fv_pointwise_weight", pointwise_weight),
        _c_array("fv_pointwise_bias", pointwise_bias),
        _c_array("fv_block_norm_weight", block_norm_weight),
        _c_array("fv_block_norm_bias", block_norm_bias),
        _c_array("fv_gru_weight_ih", _tensor(model, "recurrent.weight_ih_l0")),
        _c_array("fv_gru_weight_hh", _tensor(model, "recurrent.weight_hh_l0")),
        _c_array("fv_gru_bias_ih", _tensor(model, "recurrent.bias_ih_l0")),
        _c_array("fv_gru_bias_hh", _tensor(model, "recurrent.bias_hh_l0")),
        _c_array("fv_output_norm_weight", _tensor(model, "output_norm.weight")),
        _c_array("fv_output_norm_bias", _tensor(model, "output_norm.bias")),
        _c_array("fv_speech_head_weight", _tensor(model, "speech_head.weight")),
        _c_array("fv_speech_head_bias", _tensor(model, "speech_head.bias")),
    ]
    source = (
        "/* Generated model parameters; see the source checkpoint licence. */\n"
        '#include "flashvad_weights.h"\n\n'
        + "\n".join(arrays)
    )
    (output / "flashvad_weights.c").write_text(source, encoding="utf-8")

    metadata = {
        "format": "flashvad-accelerate-v1",
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "feature": asdict(feature),
        "model": asdict(architecture),
        "total_cache_floats": total_cache_floats,
    }
    (output / "flashvad_native.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def build_macos_native(
    weights_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Path]:
    if platform.system() != "Darwin":
        raise RuntimeError("the Accelerate runtime can only be built on macOS")
    compiler = shutil.which("clang")
    if compiler is None:
        raise RuntimeError("clang is required")

    weights = Path(weights_dir)
    output = Path(output_dir)
    native = _native_source_dir()
    output.mkdir(parents=True, exist_ok=True)
    common = [
        compiler,
        "-std=c11",
        "-O3",
        "-mcpu=native",
        "-DNDEBUG",
        "-DACCELERATE_NEW_LAPACK",
        f"-I{native}",
        f"-I{weights}",
        str(native / "flashvad_native.c"),
        str(weights / "flashvad_weights.c"),
        "-framework",
        "Accelerate",
    ]
    library = output / "libflashvad_native.dylib"
    subprocess.run(
        [*common, "-dynamiclib", "-o", str(library)],
        check=True,
    )
    benchmark = output / "flashvad_benchmark"
    subprocess.run(
        [*common, str(native / "benchmark.c"), "-o", str(benchmark)],
        check=True,
    )
    return {"library": library, "benchmark": benchmark}


def benchmark_macos_native(
    checkpoint_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    output = Path(output_dir)
    weights = export_macos_native(checkpoint_path, output / "generated")
    products = build_macos_native(weights, output / "build")
    completed = subprocess.run(
        [str(products["benchmark"])],
        check=True,
        text=True,
        capture_output=True,
    )
    result: dict[str, object] = json.loads(completed.stdout)
    checkpoint = Path(checkpoint_path)
    compiler = shutil.which("clang")
    if compiler is None:
        raise RuntimeError("clang is required")
    result = {
        "schema": "flashvad-native-benchmark-v1",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "macos": platform.mac_ver()[0],
            "machine": platform.machine(),
            "chip": _first_line(["sysctl", "-n", "machdep.cpu.brand_string"]),
        },
        "toolchain": {
            "python": platform.python_version(),
            "clang": _first_line([compiler, "--version"]),
            "build_flags": [
                "-std=c11",
                "-O3",
                "-mcpu=native",
                "-DNDEBUG",
                "-DACCELERATE_NEW_LAPACK",
            ],
        },
        "protocol": {
            "clock": "mach_continuous_time",
            "warmup_hops": 1_000,
            "initialization_iterations": 500,
            "measurement_iterations": result["iterations"],
            "audio": "all-zero 10 ms hops",
            "threads": 1,
        },
        **result,
    }
    result["binary_bytes"] = products["benchmark"].stat().st_size
    result["library_bytes"] = products["library"].stat().st_size
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (output / "benchmark.json").write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return result
