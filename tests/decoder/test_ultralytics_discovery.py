# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""On-demand discovery integration test over real Ultralytics exports.

This repository ships no Ultralytics code and installs none. Export the
models yourself with your own Ultralytics installation and point
EF_ULTRALYTICS_EXPORTS at the directory (see TESTING.md); anything missing
is skipped rather than failed.

Deselected by default (see pyproject addopts). Run:
    source venv/bin/activate
    maturin develop -m crates/python-decoder/Cargo.toml
    EF_ULTRALYTICS_EXPORTS=/path/to/exports \
      pytest tests/decoder/test_ultralytics_discovery.py -m ultralytics -v
"""

import os
from pathlib import Path

import pytest

from .ultralytics_signals import signals_for

EXPORTS = Path(
    os.environ.get(
        "EF_ULTRALYTICS_EXPORTS",
        Path(__file__).resolve().parents[2] / ".ultralytics-exports",
    )
)
pytestmark = pytest.mark.ultralytics


def _spec(t):
    """Maps one captured tensor signal onto the binding's tensor spec.

    `inputs` and `outputs` take the same 3-or-4-tuple shape, so one mapper
    serves both.
    """
    q = t["quantization"]
    return (
        t["name"],
        t["shape"],
        t["dtype"],
        None if q is None else (q["scale"], q["zero_point"]),
    )


def _infer(model: Path):
    from edgefirst.decoder import infer_ultralytics_schema

    sig = signals_for(model)
    return infer_ultralytics_schema(
        sig["source"],
        [_spec(t) for t in sig["inputs"]],
        [_spec(t) for t in sig["outputs"]],
        sig["metadata"],
    )


SUPPORTED = [
    # (file, expect_version, expect_outputs, expect_seg)
    ("yolov8n.onnx", "yolov8", 1, False),
    ("yolo11n.onnx", "yolov8", 1, False),
    ("yolo26n.onnx", "yolo26", 1, False),
    ("yolov8n-seg.onnx", "yolov8", 2, True),
    ("yolo11n-seg.onnx", "yolov8", 2, True),
    ("yolo26n-seg.onnx", "yolo26", 2, True),
    ("yolov8n_float32.tflite", "yolov8", 1, False),
    ("yolov8n_int8.tflite", "yolov8", 1, False),
    ("yolov8n-seg_float32.tflite", "yolov8", 2, True),
    ("yolov8n-seg_int8.tflite", "yolov8", 2, True),
    ("yolo26n_float32.tflite", "yolo26", 1, False),
]


@pytest.mark.parametrize(
    "name,version,n_outputs,seg", SUPPORTED, ids=[s[0] for s in SUPPORTED]
)
def test_supported_family_discovers(name, version, n_outputs, seg):
    model = EXPORTS / name
    if not model.exists():
        pytest.skip(f"{name} not found in {EXPORTS} — see TESTING.md")
    schema, labels, description = _infer(model)
    assert len(labels) == 80
    assert schema["decoder_version"] == version
    assert len(schema["outputs"]) == n_outputs
    if seg:
        assert any(o.get("type") == "protos" for o in schema["outputs"])

    # "detections" (plural) is the schema vocabulary's fully-decoded
    # post-NMS type, which is what a YOLO26 end-to-end head emits;
    # "detection" is the anchor-grid output that still needs decoding.
    det_type = "detections" if version == "yolo26" else "detection"
    det = next(o for o in schema["outputs"] if o.get("type") == det_type)

    # Ultralytics TFLite exports emit [0,1] boxes, ONNX emits pixel-space.
    assert det["normalized"] is name.endswith(".tflite")

    # The schema must pin Ultralytics' own class-aware NMS for pre-NMS
    # heads, and add none for end-to-end heads that ran it in-graph.
    assert schema.get("nms") == (None if version == "yolo26" else "class_aware")
    assert "Ultralytics" in description


UNSUPPORTED = ["yolo11n-pose.onnx", "yolo11n-cls.onnx", "yolo11n-obb.onnx"]


@pytest.mark.parametrize("name", UNSUPPORTED)
def test_unsupported_family_rejected_gracefully(name):
    model = EXPORTS / name
    if not model.exists():
        pytest.skip(f"{name} not found in {EXPORTS} — see TESTING.md")
    with pytest.raises(ValueError) as e:
        _infer(model)
    # Graceful: the message names the problem, never a panic/crash.
    msg = str(e.value).lower()
    assert "task" in msg or "layout" in msg or "ultralytics" in msg


def test_non_ultralytics_metadata_rejected():
    from edgefirst.decoder import infer_ultralytics_schema

    # `match=` is load-bearing: a bare `pytest.raises(ValueError)` here used
    # to pass on a tuple-arity error from argument marshalling, never
    # reaching the check this test is named for.
    with pytest.raises(ValueError, match="no Ultralytics signature"):
        infer_ultralytics_schema(
            "onnx",
            [("images", [1, 3, 640, 640], "float32", None)],
            [("output", [1, 1000], "float32", None)],
            {},
        )
