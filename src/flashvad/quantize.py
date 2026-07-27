from __future__ import annotations

from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_dynamic_int8(source: str | Path, destination: str | Path) -> Path:
    source_path = Path(source)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        str(source_path),
        str(output),
        weight_type=QuantType.QInt8,
    )
    metadata = source_path.with_suffix(".json")
    if metadata.exists():
        output.with_suffix(".json").write_bytes(metadata.read_bytes())
    return output
