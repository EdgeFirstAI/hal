# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Python binding coverage for edgefirst.decoder.infer_ultralytics_schema.

The Rust classifier itself is covered exhaustively by
crates/decoder/src/infer.rs and crates/decoder-capi/src/infer.rs; this file
only exercises the PyO3 boundary: argument marshalling and the
InferError -> ValueError mapping.
"""

import json
from pathlib import Path

import edgefirst.decoder as ef
import pytest

FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "crates"
    / "decoder"
    / "testdata"
    / "infer"
    / "yolov8n.signals.json"
)


def _load_fixture_call_args(path: Path):
    """Maps a captured Task-0 fixture's JSON onto the binding's call
    signature: (source, inputs, outputs, metadata)."""
    data = json.loads(path.read_text())

    def tensor_arg(t):
        q = t["quantization"]
        if q is None:
            return (t["name"], t["shape"], t["dtype"])
        return (t["name"], t["shape"], t["dtype"], (q["scale"], q["zero_point"]))

    inputs = [tensor_arg(t) for t in data["inputs"]]
    outputs = [tensor_arg(t) for t in data["outputs"]]
    return data["source"], inputs, outputs, data["metadata"]


def test_infer_ultralytics_schema_yolov8n():
    source, inputs, outputs, metadata = _load_fixture_call_args(FIXTURE)

    result = ef.infer_ultralytics_schema(source, inputs, outputs, metadata)

    assert len(result.labels) == 80
    assert result.schema["decoder_version"] == "yolov8"
    assert isinstance(result.description, str)
    assert result.description

    # Named fields and positional unpacking must stay interchangeable.
    schema, labels, description = result
    assert (schema, labels, description) == (
        result.schema,
        result.labels,
        result.description,
    )


def test_infer_ultralytics_schema_rejects_empty_metadata():
    source, inputs, outputs, _ = _load_fixture_call_args(FIXTURE)

    with pytest.raises(ValueError, match="no Ultralytics signature"):
        ef.infer_ultralytics_schema(source, inputs, outputs, {})


def test_infer_ultralytics_schema_labels_are_index_ordered():
    """`len(labels) == 80` alone passes on a shuffled or wrongly-keyed
    `names` dict; the order is the whole contract, since `decode()` returns
    class indices into this list."""
    source, inputs, outputs, metadata = _load_fixture_call_args(FIXTURE)
    labels = ef.infer_ultralytics_schema(source, inputs, outputs, metadata).labels
    assert labels[0] == "person"
    assert labels[1] == "bicycle"
    assert labels[79] == "toothbrush"


def test_inferred_schema_builds_a_decoder():
    """The point of the API: the JSON it returns must be something this same
    package accepts back as a decoder configuration. Nothing else in the
    Python suite closes that loop."""
    source, inputs, outputs, metadata = _load_fixture_call_args(FIXTURE)
    schema = ef.infer_ultralytics_schema(source, inputs, outputs, metadata).schema
    decoder = ef.Decoder(schema, score_threshold=0.25, iou_threshold=0.45)
    assert decoder is not None


def test_inferred_segmentation_schema_carries_protos():
    source, inputs, outputs, metadata = _load_fixture_call_args(
        FIXTURE.with_name("yolov8n-seg.signals.json")
    )
    schema, labels, description = ef.infer_ultralytics_schema(
        source, inputs, outputs, metadata
    )
    assert len(labels) == 80
    assert [o.get("type") for o in schema["outputs"]] == ["detection", "protos"]
    assert "segment" in description
    assert ef.Decoder(schema) is not None


def test_quantized_output_tuple_is_accepted():
    """No captured fixture is quantized at the boundary -- every Ultralytics
    export, int8 included, exposes float32 I/O -- so the 4-tuple quantization
    slot has no coverage from fixtures alone."""
    metadata = {
        "names": "{0: 'person', 1: 'bicycle'}",
        "task": "detect",
        "end2end": "False",
    }
    inputs = [("images", [1, 640, 640, 3], "float32")]
    outputs = [("output0", [1, 6, 8400], "int8", ([0.02], [-5]))]
    schema = ef.infer_ultralytics_schema("tflite", inputs, outputs, metadata).schema
    quant = schema["outputs"][0]["quantization"]
    # Scales cross as f32, so a Python float does not survive exactly.
    assert quant["scale"] == pytest.approx([0.02], rel=1e-6)
    assert quant["zero_point"] == [-5]
    assert quant["dtype"] == "int8"


def test_bare_and_explicit_none_quantization_agree():
    """A 3-tuple and a 4-tuple with `None` both mean unquantized. The binding
    distinguishes them by tuple arity alone, so the equivalence is worth
    pinning."""
    metadata = {"names": "{0: 'a', 1: 'b'}", "task": "detect", "end2end": "False"}
    inputs = [("images", [1, 3, 640, 640], "float32")]
    bare = ef.infer_ultralytics_schema(
        "onnx", inputs, [("output0", [1, 6, 8400], "float32")], metadata
    )
    explicit = ef.infer_ultralytics_schema(
        "onnx", inputs, [("output0", [1, 6, 8400], "float32", None)], metadata
    )
    assert bare == explicit


def test_input_tensors_accept_the_quantization_slot():
    """`inputs` and `outputs` take the same tensor-spec shape, so a caller
    can build both from one runtime description."""
    metadata = {"names": "{0: 'a', 1: 'b'}", "task": "detect", "end2end": "False"}
    outputs = [("output0", [1, 6, 8400], "float32")]
    three = ef.infer_ultralytics_schema(
        "onnx", [("images", [1, 3, 640, 640], "float32")], outputs, metadata
    )
    four = ef.infer_ultralytics_schema(
        "onnx", [("images", [1, 3, 640, 640], "float32", None)], outputs, metadata
    )
    assert three == four


def test_per_channel_quantization_rejected():
    metadata = {"names": "{0: 'a', 1: 'b'}", "task": "detect", "end2end": "False"}
    with pytest.raises(ValueError, match="per-channel"):
        ef.infer_ultralytics_schema(
            "tflite",
            [("images", [1, 640, 640, 3], "float32")],
            [("output0", [1, 6, 8400], "int8", ([0.02, 0.03], [-5, -7]))],
            metadata,
        )


def test_other_source_is_refused_not_guessed():
    metadata = {"names": "{0: 'a', 1: 'b'}", "task": "detect", "end2end": "False"}
    with pytest.raises(ValueError, match="box coordinate convention"):
        ef.infer_ultralytics_schema(
            "other",
            [("images", [1, 3, 640, 640], "float32")],
            [("output0", [1, 6, 8400], "float32")],
            metadata,
        )


def test_unknown_dtype_and_source_strings_are_rejected():
    metadata = {"names": "{0: 'a', 1: 'b'}", "task": "detect", "end2end": "False"}
    outputs = [("output0", [1, 6, 8400], "float32")]
    with pytest.raises(ValueError, match="unknown dtype"):
        ef.infer_ultralytics_schema(
            "onnx", [("images", [1, 3, 640, 640], "float64")], outputs, metadata
        )
    with pytest.raises(ValueError, match="unknown source"):
        ef.infer_ultralytics_schema(
            "onnxruntime", [("images", [1, 3, 640, 640], "float32")], outputs, metadata
        )


def test_dynamic_shape_dimensions_are_refused_by_name():
    """A model exported with `dynamic=True` reports a symbolic axis (ONNX)
    or -1 (TFLite). Both are refused with a ValueError naming the tensor and
    axis, rather than a TypeError/OverflowError out of argument conversion."""
    metadata = {"names": "{0: 'a', 1: 'b'}", "task": "detect", "end2end": "False"}
    outputs = [("output0", [1, 6, 8400], "float32")]
    for dynamic in (-1, "batch"):
        with pytest.raises(ValueError, match="dynamic dimension"):
            ef.infer_ultralytics_schema(
                "onnx",
                [("images", [dynamic, 3, 640, 640], "float32")],
                outputs,
                metadata,
            )


def test_batched_input_is_refused():
    metadata = {"names": "{0: 'a', 1: 'b'}", "task": "detect", "end2end": "False"}
    with pytest.raises(ValueError, match="batch 4"):
        ef.infer_ultralytics_schema(
            "onnx",
            [("images", [4, 3, 640, 640], "float32")],
            [("output0", [1, 6, 8400], "float32")],
            metadata,
        )


def test_end_to_end_schema_names_the_detections_type():
    source, inputs, outputs, metadata = _load_fixture_call_args(
        FIXTURE.with_name("yolo26n.signals.json")
    )
    schema = ef.infer_ultralytics_schema(source, inputs, outputs, metadata).schema
    assert schema["decoder_version"] == "yolo26"
    assert schema["outputs"][0]["type"] == "detections"
    assert "nms" not in schema


def test_pre_nms_schema_pins_class_aware_nms():
    source, inputs, outputs, metadata = _load_fixture_call_args(FIXTURE)
    schema = ef.infer_ultralytics_schema(source, inputs, outputs, metadata).schema
    assert schema["nms"] == "class_aware"
