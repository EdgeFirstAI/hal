"""Tests for scripts/per_scale_decode_reference.py helpers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


def _import_ref():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root / "scripts"))
    try:
        return importlib.import_module("per_scale_decode_reference")
    finally:
        sys.path.pop(0)


def _child(name: str, stride: int, shape, dshape=None, quant=None):
    out = {
        "name": name,
        "stride": stride,
        "shape": list(shape),
    }
    if dshape is not None:
        out["dshape"] = dshape
    if quant is not None:
        out["quantization"] = {
            "scale": quant[0],
            "zero_point": quant[1],
            "dtype": "int8",
        }
    return out


def test_validate_stride_sets_ok():
    mod = _import_ref()
    boxes = {8: {}, 16: {}}
    scores = {8: {}, 16: {}}
    mod._validate_stride_sets(boxes, scores, {})
    mod._validate_stride_sets(boxes, scores, {8: {}, 16: {}})


def test_validate_stride_sets_mismatch_raises():
    mod = _import_ref()
    with pytest.raises(AssertionError, match="Stride mismatch"):
        mod._validate_stride_sets({8: {}}, {16: {}}, {})
    with pytest.raises(AssertionError, match="mask_coefs"):
        mod._validate_stride_sets({8: {}}, {8: {}}, {16: {}})


def test_build_fpn_level_with_and_without_mc():
    mod = _import_ref()
    bx = _child(
        "boxes_0",
        8,
        (1, 80, 80, 64),
        dshape=[{"batch": 1}, {"height": 80}, {"width": 80}, {"num_features": 64}],
        quant=(0.1, -1),
    )
    sc = _child(
        "scores_0",
        8,
        (1, 80, 80, 80),
        dshape=[{"batch": 1}, {"height": 80}, {"width": 80}, {"num_classes": 80}],
        quant=(0.2, 0),
    )
    mc = _child(
        "mc_0",
        8,
        (1, 80, 80, 32),
        dshape=[{"batch": 1}, {"height": 80}, {"width": 80}, {"num_features": 32}],
        quant=(0.3, 1),
    )
    scores_meta = {"activation_required": "sigmoid"}

    level = mod._build_fpn_level(8, bx, sc, mc, scores_meta)
    assert level["stride"] == 8.0
    assert level["h"] == 80 and level["w"] == 80
    assert level["reg_max"] == 16
    assert level["nc"] == 80
    assert level["nm"] == 32
    assert level["mc_shape"] == (1, 80, 80, 32)
    assert level["score_activation"] == "sigmoid"

    det_only = mod._build_fpn_level(8, bx, sc, None, scores_meta)
    assert det_only["nm"] == 0
    assert det_only["mc_shape"] is None
    assert det_only["mc_quant"] is None


def test_per_scale_layout_from_metadata():
    mod = _import_ref()
    metadata = {
        "outputs": [
            {
                "type": "boxes",
                "encoding": "dfl",
                "outputs": [
                    _child(
                        "b8",
                        8,
                        (1, 2, 2, 64),
                        dshape=[
                            {"batch": 1},
                            {"height": 2},
                            {"width": 2},
                            {"num_features": 64},
                        ],
                        quant=(0.1, 0),
                    ),
                    _child(
                        "b16",
                        16,
                        (1, 1, 1, 64),
                        dshape=[
                            {"batch": 1},
                            {"height": 1},
                            {"width": 1},
                            {"num_features": 64},
                        ],
                        quant=(0.1, 0),
                    ),
                ],
            },
            {
                "type": "scores",
                "activation_required": "sigmoid",
                "outputs": [
                    _child(
                        "s8",
                        8,
                        (1, 2, 2, 3),
                        dshape=[
                            {"batch": 1},
                            {"height": 2},
                            {"width": 2},
                            {"num_classes": 3},
                        ],
                        quant=(0.2, 0),
                    ),
                    _child(
                        "s16",
                        16,
                        (1, 1, 1, 3),
                        dshape=[
                            {"batch": 1},
                            {"height": 1},
                            {"width": 1},
                            {"num_classes": 3},
                        ],
                        quant=(0.2, 0),
                    ),
                ],
            },
        ]
    }
    layout = mod.PerScaleLayout(metadata)
    assert layout.box_encoding == "dfl"
    assert len(layout.fpn_levels) == 2
    assert layout.fpn_levels[0]["stride"] == 8.0
    assert layout.total_anchors == 2 * 2 + 1 * 1
    assert layout.nc == 3
    assert layout.protos_shape is None


def test_nms_class_agnostic_empty_and_basic():
    mod = _import_ref()
    boxes = np.zeros((0, 4), dtype=np.float32)
    scores = np.zeros((0, 2), dtype=np.float32)
    out_boxes, out_scores, out_classes, indices = mod.nms_class_agnostic(
        boxes, scores, score_threshold=0.5
    )
    assert out_boxes.shape == (0, 4)
    assert out_scores.shape == (0,)
    assert out_classes.shape == (0,)
    assert indices.shape == (0,)

    boxes = np.array(
        [
            [10.0, 10.0, 20.0, 20.0],
            [11.0, 11.0, 20.0, 20.0],
            [100.0, 100.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )
    scores = np.array(
        [
            [0.9, 0.1],
            [0.85, 0.05],
            [0.1, 0.8],
        ],
        dtype=np.float32,
    )
    out_boxes, out_scores, out_classes, indices = mod.nms_class_agnostic(
        boxes,
        scores,
        iou_threshold=0.5,
        score_threshold=0.5,
        max_detections=10,
    )
    assert len(out_boxes) >= 1
    assert len(out_boxes) == len(out_scores) == len(out_classes) == len(indices)
