from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper
from onnx.external_data_helper import uses_external_data

from flashvad.export import _sanitize_onnx_metadata


def test_sanitize_onnx_metadata_removes_private_exporter_details(tmp_path: Path) -> None:
    node = helper.make_node("Relu", ["input"], ["output"], name="relu")
    node.doc_string = "/Users/example/private/project/export.py"
    node.metadata_props.add(
        key="pkg.torch.onnx.stack_trace",
        value="/Users/example/private/project/export.py:42",
    )
    graph = helper.make_graph(
        [node],
        "private.module.ExportWrapper",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
    )
    graph.doc_string = "private.module.ExportWrapper"
    model = helper.make_model(graph)
    model.doc_string = "private build"
    model.metadata_props.add(key="source", value="/Users/example/private")
    destination = tmp_path / "model.onnx"
    onnx.save(model, destination)

    _sanitize_onnx_metadata(destination)

    sanitized = onnx.load(destination)
    assert sanitized.doc_string == ""
    assert len(sanitized.metadata_props) == 0
    assert sanitized.graph.doc_string == ""
    assert sanitized.graph.node[0].doc_string == ""
    assert len(sanitized.graph.node[0].metadata_props) == 0
    assert not any(uses_external_data(tensor) for tensor in sanitized.graph.initializer)
    serialized = destination.read_bytes()
    assert b"/Users/" not in serialized
    assert b"private.module" not in serialized
