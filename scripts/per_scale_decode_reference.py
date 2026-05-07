#!/usr/bin/env python3
"""
Reference implementation: per-scale YOLOv8 segmentation decoder.

PURPOSE
-------
This script is a pure Python / NumPy reference for the per-scale TFLite
output decoding pipeline.  It demonstrates every algorithmic step that the
HAL's native Rust ``Decoder`` must implement to support models produced by
the EdgeFirst tflite-converter with ``quantization_split`` enabled.

The tflite-converter splits each logical output (boxes, scores, mask_coefs)
along the FPN scale axis so that each spatial scale gets its own
quantization parameters.  This dramatically improves int8 accuracy — see
the benchmarks below — but requires a new decode path in the HAL that
reassembles per-scale tensors before (or interleaved with) DFL + NMS.

BENCHMARK RESULTS  (coco128, imx8mp VX delegate)
--------------------------------------------------
  yolov8n-seg (encoding: dfl):
    Standard TFLite (monolithic):  Seg mAP@0.5 = 47.16   Box mAP@0.5 = 48.40
    Per-scale TFLite (this path):  Seg mAP@0.5 = 52.38   Box mAP@0.5 = 56.10
    ONNX FP32 reference:           Seg mAP@0.5 = 51.60   Box mAP@0.5 = 57.96

  yolo11n-seg (encoding: dfl):
    Per-scale TFLite (this path):  Seg mAP@0.5 = 50.03   Box mAP@0.5 = 53.65

  yolo26n-seg (encoding: ltrb):
    Standard TFLite (monolithic):  Seg mAP@0.5 = 28.60   Box mAP@0.5 = (collapsed)
    Per-scale TFLite (this path):  Seg mAP@0.5 = 44.60   Box mAP@0.5 = 47.35
    ONNX FP32 reference:           Seg mAP@0.5 = 60.98

  Per-scale quantization dramatically improves int8 accuracy across all
  model families.  For yolov8/yolo11 it essentially eliminates the
  degradation; for yolo26 it recovers +16 pp from the catastrophic
  monolithic collapse.

METADATA FORMAT  (edgefirst.json schema v2)
--------------------------------------------
The per-scale model's ``edgefirst.json`` declares hierarchical outputs.
Each logical output (``boxes``, ``scores``, ``mask_coefs``) contains an
``outputs`` array of per-scale children.  The ``protos`` output has no
children — it's a single physical tensor.

Two box encodings are supported:

**DFL encoding** (yolov8, yolo11) — ``"encoding": "dfl"``::

    {
      "name": "boxes", "type": "boxes",
      "decoder": "ultralytics", "encoding": "dfl", "normalized": true,
      "shape": [1, 4, 8400],
      "outputs": [
        {
          "name": "boxes_0", "type": "boxes", "stride": 8,
          "shape": [1, 80, 80, 64],   # NHWC: 4 * reg_max = 64
          "quantization": {"scale": 0.157, "zero_point": -42, "dtype": "int8"}
        }, ...
      ]
    }

  Box channels = ``4 * reg_max`` (typically 64).  Each side (L,T,R,B) is
  a distribution over ``reg_max`` bins.  Decode: softmax → weighted sum →
  dist2bbox.

**LTRB encoding** (yolo26) — ``"encoding": "ltrb"``::

    {
      "name": "boxes", "type": "boxes",
      "decoder": "ultralytics", "encoding": "ltrb", "normalized": true,
      "shape": [1, 4, 8400],
      "outputs": [
        {
          "name": "boxes_0", "type": "boxes", "stride": 8,
          "shape": [1, 80, 80, 4],    # NHWC: direct LTRB distances
          "quantization": {"scale": 0.039, "zero_point": -114, "dtype": "int8"}
        }, ...
      ]
    }

  Box channels = 4.  The DFL decode already happened inside the model
  graph (yolo26 architecture does DFL internally).  The 4 channels are
  direct LTRB distances in grid units.  Decode: just dist2bbox (no
  softmax step).

PIPELINE STAGES  (cross-reference with HAL crates)
----------------------------------------------------
Each section below is annotated with the HAL Rust module that already
implements the corresponding functionality for Hailo (flat or merged) paths.
The per-scale TFLite path differs primarily in:

  1. Physical tensor binding uses shape-matching (like TFLite ``get_tensor``)
     rather than HailoRT's named output API.
  2. Dequantization happens here (TFLite returns raw int8), whereas HailoRT
     can be asked for ``FormatType::FLOAT32`` dequant on-device.
  3. All 10 physical tensors arrive at once (no streaming); the decoder
     must sort/join children by stride before DFL decode.

Stage reference::

  ┌───────────────────────────────────────────────────────────────────┐
  │  Stage              │ HAL module                │ This function   │
  ├─────────────────────┼───────────────────────────┼─────────────────┤
  │  1. Bind tensors    │ merge.rs (shape_map)      │ bind_outputs()  │
  │  2. Dequantize      │ float.rs (dequantize_cpu) │ dequantize()    │
  │  3a. DFL decode     │ dfl.rs (decode_dfl_level) │ dfl_decode*()   │
  │  3b. LTRB dist2bbox │ dfl.rs (dist2bbox only)   │ ltrb_decode*()  │
  │  4. Sigmoid scores  │ postprocess.rs            │ sigmoid()       │
  │  5. Concat scales   │ merge.rs (PerScale)       │ concat in main  │
  │  6. NMS             │ yolo.rs (nms_*)           │ nms()           │
  │  7. Mask decode     │ decoder/mod.rs            │ decode_masks()  │
  └───────────────────────────────────────────────────────────────────┘

  Stage 3 branches on ``encoding``: "dfl" runs softmax + weighted sum
  before dist2bbox (yolov8/yolo11); "ltrb" skips straight to dist2bbox
  (yolo26).  All other stages are identical.

USAGE
-----
This script can be run standalone against any per-scale TFLite model with
embedded edgefirst.json metadata::

    python per_scale_decode_reference.py model.tflite image.jpg

It prints per-detection output and optionally visualizes results.  More
importantly, it is intended to be read as documentation — each function
maps to a HAL stage and is annotated accordingly.

IMPLEMENTATION NOTES FOR THE HAL AGENT
---------------------------------------
1. The ``Decoder::decode_proto`` path currently rejects per-scale models
   because ``find_output_by_shape([1, 4, 8400])`` fails — there is no
   monolithic boxes tensor.  The fix is to detect per-scale children in
   the schema v2 ``LogicalOutput`` and route through ``merge.rs``'s
   ``PerScale`` path (which already exists for Hailo) before the
   downstream float-path decode.

2. The ``merge.rs`` ``PerScale`` strategy already handles DFL decode via
   ``DflConfig``.  The missing piece is that ``decode_proto`` does not
   invoke the merge program when the decoder was constructed from v2
   metadata with per-scale children.  This is the primary integration gap.

3. Sigmoid must be applied to score tensors when ``activation_required:
   sigmoid`` is declared on the physical child.  The current merge path
   skips this (see ``merge.rs`` line 30: "Activation flags on floats —
   not yet supported").  This is required for per-scale TFLite models
   where the NPU does NOT apply sigmoid.

4. The validator currently falls back to numpy NMS when it detects
   per-scale children.  Once the HAL implements the pipeline below,
   the validator should detect HAL support and route through
   ``process_yolo_hal`` for the full GPU-accelerated path.

5. ``normalized: true`` on the logical ``boxes`` output refers to the
   original ONNX model (tf_wrapper normalization).  After per-scale DFL
   decode, boxes are in pixel coordinates (the split happens before
   the normalization post-processing).  The HAL should treat per-scale
   DFL-decoded boxes as pixel-coordinate, not normalized.
"""

from __future__ import annotations

import json
import os
import sys
import zipfile
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Stage 0: Metadata parsing
# ═══════════════════════════════════════════════════════════════════════
# HAL equivalent: schema.rs  (SchemaV2, LogicalOutput, PhysicalOutput)
#
# Parse the edgefirst.json embedded in the TFLite ZIP trailer and
# extract the hierarchical output layout.  Each logical output is
# either a direct physical tensor (no children) or a virtual parent
# with per-scale PhysicalOutput children.


def load_metadata(tflite_path: str) -> dict:
    """Extract edgefirst.json from the TFLite ZIP trailer.

    The EdgeFirst tflite-converter appends a ZIP archive containing
    ``edgefirst.json`` and ``labels.txt`` to the end of the TFLite
    flatbuffer.  TFLite runtime ignores the trailer bytes, so the model
    loads normally.

    HAL equivalent: the Rust decoder's ``new_from_metadata`` path, which
    deserializes ``SchemaV2`` via serde.
    """
    with zipfile.ZipFile(tflite_path) as zf:
        with zf.open("edgefirst.json") as f:
            return json.loads(f.read().decode("utf-8"))


class PerScaleLayout:
    """Parsed per-scale output layout, analogous to HAL's ``DflConfig`` +
    ``DecodeProgram::PerScale``.

    Attributes
    ----------
    box_encoding : str
        ``"dfl"`` for yolov8/yolo11 (softmax + weighted sum → dist2bbox)
        or ``"ltrb"`` for yolo26 (direct LTRB distances → dist2bbox only).
    fpn_levels : List[dict]
        One entry per FPN stride, sorted ascending, each containing:
        - ``stride`` (float): FPN spatial stride (8, 16, 32)
        - ``h``, ``w`` (int): spatial dimensions at this level
        - ``reg_max`` (int): DFL bin count (typically 16; 1 for LTRB)
        - ``nc`` (int): number of classes
        - ``nm`` (int): mask coefficient channels (0 if detection-only)
        - ``box_shape`` (tuple): physical box tensor shape
        - ``score_shape`` (tuple): physical score tensor shape
        - ``mc_shape`` (tuple or None): physical mask_coef tensor shape
        - ``box_quant``, ``score_quant``, ``mc_quant``:
          (scale, zero_point) tuples for dequantization
        - ``score_activation`` (str or None): e.g. "sigmoid"
    total_anchors : int
        Sum of H*W across all FPN levels (e.g. 8400 for 640×640 input).
    nc : int
        Number of classes.
    protos_shape : tuple or None
        Shape of the protos tensor, or None for detection-only models.
    protos_quant : tuple or None
        Quantization parameters for the protos tensor.
    """

    def __init__(self, metadata: dict):
        outputs_by_type: Dict[str, dict] = {}
        for logical in metadata["outputs"]:
            ltype = logical.get("type")
            if ltype:
                outputs_by_type[ltype] = logical

        boxes_meta = outputs_by_type["boxes"]
        scores_meta = outputs_by_type["scores"]
        mc_meta = outputs_by_type.get("mask_coefs")
        protos_meta = outputs_by_type.get("protos")

        # Box encoding: "dfl" (yolov8/yolo11) or "ltrb" (yolo26).
        # HAL equivalent: merge.rs checks LogicalOutput.encoding to
        # decide whether to run DFL decode or plain dist2bbox.
        self.box_encoding: str = boxes_meta.get("encoding", "dfl")

        # Index per-scale children by stride for correct pairing.
        # HAL equivalent: merge.rs plan_per_scale() sorts children by
        # stride ascending and validates stride-set equality across
        # boxes/scores/mask_coefs.
        box_by_stride = _index_by_stride(boxes_meta.get("outputs", []))
        score_by_stride = _index_by_stride(scores_meta.get("outputs", []))
        mc_by_stride = (
            _index_by_stride(mc_meta.get("outputs", []))
            if mc_meta else {}
        )

        assert set(box_by_stride) == set(score_by_stride), (
            f"Stride mismatch: boxes={set(box_by_stride)}, "
            f"scores={set(score_by_stride)}"
        )
        if mc_by_stride:
            assert set(mc_by_stride) == set(box_by_stride), (
                f"Stride mismatch: mask_coefs={set(mc_by_stride)}, "
                f"boxes={set(box_by_stride)}"
            )

        self.fpn_levels: List[dict] = []
        self.nc = 0

        for stride in sorted(box_by_stride):
            bx = box_by_stride[stride]
            sc = score_by_stride[stride]

            bx_ds = _dshape_dict(bx.get("dshape", []))
            sc_ds = _dshape_dict(sc.get("dshape", []))

            h = bx_ds.get("height", bx["shape"][1])
            w = bx_ds.get("width", bx["shape"][2])
            nc = sc_ds.get("num_classes",
                           sc_ds.get("num_features", sc["shape"][-1]))
            reg_max = bx_ds.get("num_features", bx["shape"][-1]) // 4
            self.nc = nc

            level = {
                "stride": float(stride),
                "h": int(h),
                "w": int(w),
                "reg_max": int(reg_max),
                "nc": int(nc),
                "box_shape": tuple(bx["shape"]),
                "score_shape": tuple(sc["shape"]),
                "box_quant": _extract_quant(bx),
                "score_quant": _extract_quant(sc),
                "score_activation": sc.get("activation_required",
                                           scores_meta.get(
                                               "activation_required")),
            }

            if stride in mc_by_stride:
                mc = mc_by_stride[stride]
                mc_ds = _dshape_dict(mc.get("dshape", []))
                level["nm"] = mc_ds.get("num_features", mc["shape"][-1])
                level["mc_shape"] = tuple(mc["shape"])
                level["mc_quant"] = _extract_quant(mc)
            else:
                level["nm"] = 0
                level["mc_shape"] = None
                level["mc_quant"] = None

            self.fpn_levels.append(level)

        self.total_anchors = sum(
            lvl["h"] * lvl["w"] for lvl in self.fpn_levels)

        self.protos_shape = (
            tuple(protos_meta["shape"]) if protos_meta else None)
        self.protos_quant = (
            _extract_quant(protos_meta) if protos_meta else None)


def _index_by_stride(children: List[dict]) -> Dict[int, dict]:
    """Group per-scale children by their ``stride`` field."""
    out: Dict[int, dict] = {}
    for child in children:
        stride = child.get("stride")
        assert stride is not None, (
            f"Per-scale child {child.get('name')!r} missing 'stride'")
        assert stride not in out, (
            f"Duplicate stride {stride} in {child.get('name')!r}")
        out[int(stride)] = child
    return out


def _dshape_dict(dshape: Sequence[dict]) -> dict:
    """Convert a dshape ordered-array into a flat ``{name: size}`` dict.

    HAL equivalent: ``configs::DimName`` parsing in ``schema.rs``.
    """
    out: dict = {}
    for entry in dshape:
        for k, v in entry.items():
            out[k] = v
    return out


def _extract_quant(output: dict) -> Optional[Tuple[float, int]]:
    """Extract ``(scale, zero_point)`` from a physical output's metadata.

    HAL equivalent: ``schema::Quantization`` → ``QuantTuple``.
    """
    q = output.get("quantization")
    if q is None:
        return None
    if isinstance(q, dict):
        return (q["scale"], q.get("zero_point", 0))
    if isinstance(q, (list, tuple)) and len(q) == 2:
        return (q[0], q[1])
    return None


# ═══════════════════════════════════════════════════════════════════════
# Stage 1: Tensor binding
# ═══════════════════════════════════════════════════════════════════════
# HAL equivalent: merge.rs execute_merge() → shape_map lookup
#
# Match each per-scale child's declared shape to the physical TFLite
# output tensors.  Unlike HailoRT (which uses named tensor APIs), TFLite
# outputs are identified by shape matching.  All 10 per-scale shapes are
# unique, so this is unambiguous.


def bind_outputs(
    layout: PerScaleLayout,
    output_details: list,
) -> Dict[Tuple[int, ...], int]:
    """Build a shape→index map from TFLite output details.

    Returns a dict mapping physical tensor shapes to their integer index
    in the TFLite output tensor list.

    HAL equivalent: ``merge.rs`` uses ``shape_map`` built from the
    input ``TensorDyn`` slice; here we do the same from TFLite's
    ``get_output_details()``.
    """
    shape_map: Dict[Tuple[int, ...], int] = {}
    for i, detail in enumerate(output_details):
        key = tuple(detail["shape"].tolist())
        assert key not in shape_map, (
            f"Duplicate output shape {key} at indices "
            f"{shape_map[key]} and {i}"
        )
        shape_map[key] = i
    return shape_map


# ═══════════════════════════════════════════════════════════════════════
# Stage 2: Dequantization
# ═══════════════════════════════════════════════════════════════════════
# HAL equivalent: float.rs dequantize_cpu_chunked()
#
# Per-tensor affine dequantization:  real = scale * (q - zero_point)
#
# Each per-scale child has its own (scale, zero_point) — this is the
# whole point of per-scale splitting.  The monolithic model has a single
# scale across all 8400 anchors, which is too coarse for the wide
# dynamic range across FPN levels.


def dequantize(
    tensor: np.ndarray,
    scale: float,
    zero_point: int,
) -> np.ndarray:
    """Per-tensor affine dequantization.

    HAL equivalent: ``float.rs::dequantize_cpu_chunked`` with per-tensor
    scalar parameters.  The HAL version uses SIMD (NEON on ARM) for the
    inner loop; this reference uses NumPy broadcasting.

    Parameters
    ----------
    tensor : np.ndarray
        Quantized int8/uint8/int16 tensor.
    scale : float
        Quantization scale factor.
    zero_point : int
        Quantization zero point offset.

    Returns
    -------
    np.ndarray
        Float32 tensor with recovered dynamic range.
    """
    return (tensor.astype(np.float32) - np.float32(zero_point)) * np.float32(scale)


# ═══════════════════════════════════════════════════════════════════════
# Stage 3: Box decode — DFL or LTRB
# ═══════════════════════════════════════════════════════════════════════
# HAL equivalent: dfl.rs decode_dfl_level() / dist2bbox
#
# Two box encoding schemes exist:
#
# **DFL** (yolov8, yolo11) — ``encoding: "dfl"``
#   The model outputs raw logits over ``reg_max`` bins per box side.
#   Decode: softmax → weighted sum → LTRB distances → dist2bbox.
#   Box tensor shape: (H, W, 4 * reg_max) — typically (H, W, 64).
#
# **LTRB** (yolo26) — ``encoding: "ltrb"``
#   The model already decoded DFL internally (yolo26 architecture
#   embeds the softmax+weighted-sum in its graph).  The 4 channels
#   are direct LTRB distances in grid units.  Decode: dist2bbox only.
#   Box tensor shape: (H, W, 4).
#
# Both encodings share the same dist2bbox step:
#
#   xc = (gx + (dist_right - dist_left) / 2) * stride
#   yc = (gy + (dist_bottom - dist_top) / 2) * stride
#   w  = (dist_left + dist_right) * stride
#   h  = (dist_top + dist_bottom) * stride
#
# Output: (H*W, 4) array of xcycwh boxes in pixel coordinates.


def _stable_softmax(x: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable softmax: subtract max before exp.

    HAL equivalent: ``dfl.rs::softmax_inplace`` — the Rust version
    operates in-place on a scratch buffer for each anchor×side.
    """
    x_max = x.max(axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / e.sum(axis=axis, keepdims=True)


def make_anchor_grid(h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    """Precompute anchor centre coordinates in grid units.

    Returns ``(grid_x, grid_y)`` each of shape ``(H, W)``.  The +0.5
    offset matches the Ultralytics ``make_anchors()`` convention —
    anchors sit at grid-cell centres.

    HAL equivalent: ``dfl.rs::make_anchor_grid`` — returns flat
    ``Vec<f32>`` of length ``H*W``.
    """
    gx = np.arange(w, dtype=np.float32) + 0.5
    gy = np.arange(h, dtype=np.float32) + 0.5
    return np.meshgrid(gx, gy)


def dfl_decode_level(
    bbox_tensor: np.ndarray,
    h: int,
    w: int,
    stride: float,
    reg_max: int,
) -> np.ndarray:
    """Decode one FPN level's DFL boxes to xcycwh pixel coordinates.

    HAL equivalent: ``dfl.rs::decode_dfl_level``

    Parameters
    ----------
    bbox_tensor : np.ndarray
        Float32 tensor of shape ``(H, W, 4 * reg_max)`` — already
        dequantized.
    h, w : int
        Spatial dimensions of this FPN level.
    stride : float
        Spatial stride (e.g. 8.0, 16.0, 32.0).
    reg_max : int
        Number of DFL distribution bins (typically 16).

    Returns
    -------
    np.ndarray
        Shape ``(H*W, 4)`` with columns ``[xc, yc, width, height]`` in
        pixel coordinates of the model input (e.g. 640×640).
    """
    # bins = [0, 1, 2, ..., reg_max - 1]
    bins = np.arange(reg_max, dtype=np.float32)

    # Reshape to (H, W, 4, reg_max) and apply softmax along the
    # distribution axis to get probabilities per bin.
    raw = bbox_tensor.reshape(h, w, 4, reg_max)
    probs = _stable_softmax(raw, axis=-1)

    # Weighted sum: expected distance in grid units per side.
    dist = (probs * bins).sum(axis=-1)  # (H, W, 4) — LTRB grid units

    # Unpack the four distances.
    d_left = dist[:, :, 0]
    d_top = dist[:, :, 1]
    d_right = dist[:, :, 2]
    d_bottom = dist[:, :, 3]

    # Anchor grid in grid units (centres of each cell).
    grid_x, grid_y = make_anchor_grid(h, w)

    # dist2bbox: LTRB distances → xcycwh in pixel coordinates.
    # The multiplication order matters for floating-point equivalence
    # with the Rust implementation.
    xc = (grid_x + (d_right - d_left) / 2.0) * stride
    yc = (grid_y + (d_bottom - d_top) / 2.0) * stride
    bw = (d_left + d_right) * stride
    bh = (d_top + d_bottom) * stride

    boxes = np.stack([xc, yc, bw, bh], axis=-1)  # (H, W, 4)
    return boxes.reshape(-1, 4)                    # (H*W, 4)


def dfl_decode_level_with_intermediates(
    box_q: np.ndarray,
    q_scale: float,
    q_zp: int,
    h: int,
    w: int,
    stride: int,
    reg_max: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """DFL decode for one FPN level, returning per-stage intermediates.

    Same algorithm as :func:`dfl_decode_level` but exposes the
    dequantized tensor and the post-softmax-weighted-sum ``ltrb`` array
    so callers can capture them as fixture data. Used only by the
    fixture generator; production callers use :func:`dfl_decode_level`.

    Returns
    -------
    (dequant, ltrb, xywh)
        - ``dequant`` : ``(1, h, w, 4*reg_max) float32``
        - ``ltrb``    : ``(h*w, 4) float32`` — per-side weighted sum
        - ``xywh``    : ``(h*w, 4) float32`` — pixel-coordinate boxes
    """
    dequant = (box_q.astype(np.float32) - float(q_zp)) * float(q_scale)
    reshaped = dequant.reshape(h * w, 4, reg_max)
    softmax = _stable_softmax(reshaped, axis=2)
    arange = np.arange(reg_max, dtype=np.float32)
    ltrb = (softmax * arange).sum(axis=2).astype(np.float32)

    grid_x = np.tile(np.arange(w, dtype=np.float32), h) + 0.5
    grid_y = np.repeat(np.arange(h, dtype=np.float32), w) + 0.5
    l, t, r, b = ltrb[:, 0], ltrb[:, 1], ltrb[:, 2], ltrb[:, 3]
    xc = (grid_x + (r - l) / 2.0) * float(stride)
    yc = (grid_y + (b - t) / 2.0) * float(stride)
    bw = (l + r) * float(stride)
    bh = (t + b) * float(stride)
    xywh = np.stack([xc, yc, bw, bh], axis=1).astype(np.float32)
    return dequant, ltrb, xywh


def ltrb_decode_level(
    bbox_tensor: np.ndarray,
    h: int,
    w: int,
    stride: float,
) -> np.ndarray:
    """Decode one FPN level's LTRB boxes to xcycwh pixel coordinates.

    Used for yolo26 models where DFL decode already happened inside
    the model graph.  The 4 channels are direct LTRB distances in
    grid units — we only need the dist2bbox conversion.

    HAL equivalent: the dist2bbox portion of ``dfl.rs::decode_dfl_level``
    (skip the softmax + weighted-sum, go straight to coordinate transform).

    Parameters
    ----------
    bbox_tensor : np.ndarray
        Float32 tensor of shape ``(H, W, 4)`` — already dequantized.
        Channels are ``[left, top, right, bottom]`` distances in grid units.
    h, w : int
        Spatial dimensions of this FPN level.
    stride : float
        Spatial stride (e.g. 8.0, 16.0, 32.0).

    Returns
    -------
    np.ndarray
        Shape ``(H*W, 4)`` with columns ``[xc, yc, width, height]`` in
        pixel coordinates of the model input (e.g. 640×640).
    """
    ltrb = bbox_tensor.reshape(h, w, 4)
    d_left = ltrb[:, :, 0]
    d_top = ltrb[:, :, 1]
    d_right = ltrb[:, :, 2]
    d_bottom = ltrb[:, :, 3]

    grid_x, grid_y = make_anchor_grid(h, w)

    xc = (grid_x + (d_right - d_left) / 2.0) * stride
    yc = (grid_y + (d_bottom - d_top) / 2.0) * stride
    bw = (d_left + d_right) * stride
    bh = (d_top + d_bottom) * stride

    boxes = np.stack([xc, yc, bw, bh], axis=-1)  # (H, W, 4)
    return boxes.reshape(-1, 4)                    # (H*W, 4)


def ltrb_decode_level_with_intermediates(
    box_q: np.ndarray,
    q_scale: float,
    q_zp: int,
    h: int,
    w: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """LTRB decode for one FPN level, returning per-stage intermediates.

    Same algorithm as :func:`ltrb_decode_level` but exposes the
    dequantized tensor for fixture capture.

    Returns
    -------
    (dequant, xywh)
        - ``dequant`` : ``(1, h, w, 4) float32``
        - ``xywh``    : ``(h*w, 4) float32`` — pixel-coordinate boxes
    """
    dequant = (box_q.astype(np.float32) - float(q_zp)) * float(q_scale)
    flat = dequant.reshape(h * w, 4)
    grid_x = np.tile(np.arange(w, dtype=np.float32), h) + 0.5
    grid_y = np.repeat(np.arange(h, dtype=np.float32), w) + 0.5
    l, t, r, b = flat[:, 0], flat[:, 1], flat[:, 2], flat[:, 3]
    xc = (grid_x + (r - l) / 2.0) * float(stride)
    yc = (grid_y + (b - t) / 2.0) * float(stride)
    bw = (l + r) * float(stride)
    bh = (t + b) * float(stride)
    xywh = np.stack([xc, yc, bw, bh], axis=1).astype(np.float32)
    return dequant, xywh


# ═══════════════════════════════════════════════════════════════════════
# Stage 4: Sigmoid activation for scores
# ═══════════════════════════════════════════════════════════════════════
# HAL equivalent: postprocess.rs (sigmoid is applied during NMS filtering)
#
# The TFLite NPU does NOT apply sigmoid to score outputs — the metadata
# declares ``activation_required: sigmoid`` on each score child.  The
# HAL merge path (merge.rs line 30) notes this is "not yet supported".
#
# Implementation priority: this activation is REQUIRED for correctness.
# Without sigmoid, raw logits are passed to NMS, resulting in wrong
# score-based filtering.


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Element-wise sigmoid activation.

    Uses the numerically stable form: for large negative values, compute
    via ``exp(x) / (1 + exp(x))`` to avoid overflow.

    HAL equivalent: typically fused into the NMS score threshold check.
    For the reference implementation we apply it as a separate step.
    """
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x))
    ).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Stage 5: Scale concatenation
# ═══════════════════════════════════════════════════════════════════════
# HAL equivalent: merge.rs execute_per_scale()
#
# After per-level box decode (DFL or LTRB), sigmoid (scores), and
# dequantization (mask_coefs), concatenate all FPN levels along the
# anchor axis:
#
#   boxes:      (80*80 + 40*40 + 20*20, 4)      = (8400, 4)  xcycwh pixels
#   scores:     (80*80 + 40*40 + 20*20, NC)      = (8400, 80) probabilities
#   mask_coefs: (80*80 + 40*40 + 20*20, NM)      = (8400, 32) coefficients
#
# This is the point where the per-scale path rejoins the monolithic
# decode pipeline.  The downstream NMS and mask decode stages are
# identical regardless of whether inputs were per-scale or monolithic.


def decode_all_scales(
    outputs: List[np.ndarray],
    layout: PerScaleLayout,
    shape_map: Dict[Tuple[int, ...], int],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run stages 2–5: dequantize → box decode → sigmoid → concat.

    This is the core per-scale reassembly function.  In the HAL, this
    maps to ``DecodeProgram::execute()`` with ``PerScale`` merge recipes.

    The box decode step (stage 3) branches on ``layout.box_encoding``:
    - ``"dfl"``: softmax + weighted sum → dist2bbox  (yolov8, yolo11)
    - ``"ltrb"``: dist2bbox only                      (yolo26)

    Parameters
    ----------
    outputs : List[np.ndarray]
        Raw TFLite output tensors, indexed by physical output position.
    layout : PerScaleLayout
        Parsed per-scale metadata.
    shape_map : Dict[Tuple[int, ...], int]
        Shape→index binding from ``bind_outputs()``.

    Returns
    -------
    boxes : np.ndarray
        (total_anchors, 4) xcycwh in pixel coordinates.
    scores : np.ndarray
        (total_anchors, NC) class probabilities in [0, 1].
    mask_coefs : np.ndarray or None
        (total_anchors, NM) mask coefficients, or None.
    """
    all_boxes = []
    all_scores = []
    all_mc = []

    for level in layout.fpn_levels:
        stride = level["stride"]
        h, w = level["h"], level["w"]

        # ── Stage 2: Dequantize boxes ────────────────────────────────
        box_idx = shape_map[level["box_shape"]]
        box_raw = outputs[box_idx]
        box_f32 = _dequantize_nhwc(box_raw, level["box_quant"])

        # ── Stage 3: Box decode → xcycwh pixels ─────────────────────
        if layout.box_encoding == "ltrb":
            # yolo26: 4 channels are direct LTRB distances
            level_boxes = ltrb_decode_level(box_f32, h, w, stride)
        else:
            # yolov8/yolo11: 4*reg_max channels need DFL decode
            level_boxes = dfl_decode_level(
                box_f32, h, w, stride, level["reg_max"])
        all_boxes.append(level_boxes)

        # ── Stage 2+4: Dequantize scores + sigmoid ──────────────────
        score_idx = shape_map[level["score_shape"]]
        score_raw = outputs[score_idx]
        score_f32 = _dequantize_nhwc(score_raw, level["score_quant"])

        if level.get("score_activation") == "sigmoid":
            score_f32 = sigmoid(score_f32)

        all_scores.append(score_f32.reshape(-1, score_f32.shape[-1]))

        # ── Stage 2: Dequantize mask coefficients ────────────────────
        if level["mc_shape"] is not None:
            mc_idx = shape_map[level["mc_shape"]]
            mc_raw = outputs[mc_idx]
            mc_f32 = _dequantize_nhwc(mc_raw, level["mc_quant"])
            all_mc.append(mc_f32.reshape(-1, mc_f32.shape[-1]))

    # ── Stage 5: Concatenate across scales ───────────────────────────
    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    mask_coefs = np.concatenate(all_mc, axis=0) if all_mc else None

    return boxes, scores, mask_coefs


def _dequantize_nhwc(
    tensor: np.ndarray,
    quant: Optional[Tuple[float, int]],
) -> np.ndarray:
    """Dequantize and remove batch dimension from an NHWC tensor.

    Parameters
    ----------
    tensor : np.ndarray
        Shape (1, H, W, C) with int8/uint8 dtype, or float32.
    quant : (scale, zero_point) or None

    Returns
    -------
    np.ndarray
        Shape (H, W, C) float32.
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    if quant is not None and np.issubdtype(tensor.dtype, np.integer):
        return dequantize(tensor, quant[0], quant[1])
    return tensor.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Stage 6: Non-Maximum Suppression (NMS)
# ═══════════════════════════════════════════════════════════════════════
# HAL equivalent: yolo.rs nms_class_agnostic() / nms_class_aware()
#
# Standard IoU-based NMS.  The HAL implements both class-agnostic
# (suppress across all classes) and class-aware (suppress within each
# class) modes.  For this reference we implement class-agnostic NMS,
# which is the simpler variant and the EdgeFirst validator default.
#
# Input boxes are xcycwh pixels; NMS operates on xyxy so we convert.


def xcycwh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (N, 4) xcycwh → xyxy."""
    xy = boxes[:, :2]
    wh = boxes[:, 2:4]
    half_wh = wh / 2.0
    return np.concatenate([xy - half_wh, xy + half_wh], axis=1)


def nms_class_agnostic(
    boxes_xcycwh: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.7,
    score_threshold: float = 0.001,
    max_detections: int = 300,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Class-agnostic NMS on per-class score matrix.

    HAL equivalent: ``yolo.rs::nms_class_agnostic``

    Parameters
    ----------
    boxes_xcycwh : (N, 4) xcycwh pixel coordinates
    scores : (N, NC) class probabilities
    iou_threshold : float
    score_threshold : float
    max_detections : int

    Returns
    -------
    boxes : (M, 4) xyxy normalized to [0, 1]  ← NOT normalized here,
            that's the caller's job
    scores : (M,) best class scores
    classes : (M,) best class indices
    indices : (M,) original anchor indices (for mask coefficient selection)
    """
    # Best class per anchor
    class_ids = scores.argmax(axis=1)
    best_scores = scores[np.arange(len(scores)), class_ids]

    # Score threshold filter
    mask = best_scores > score_threshold
    if not mask.any():
        empty = np.empty((0, 4), dtype=np.float32)
        return empty, np.empty(0), np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)

    filtered_indices = np.where(mask)[0]
    filtered_boxes = xcycwh_to_xyxy(boxes_xcycwh[filtered_indices])
    filtered_scores = best_scores[filtered_indices]
    filtered_classes = class_ids[filtered_indices]

    # Sort by score descending
    order = filtered_scores.argsort()[::-1]
    if max_detections > 0:
        order = order[:max_detections * 3]  # generous pre-filter

    keep = []
    x1 = filtered_boxes[:, 0]
    y1 = filtered_boxes[:, 1]
    x2 = filtered_boxes[:, 2]
    y2 = filtered_boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    suppressed = np.zeros(len(filtered_boxes), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        if len(keep) >= max_detections:
            break

        # Compute IoU with all remaining
        xx1 = np.maximum(x1[idx], x1)
        yy1 = np.maximum(y1[idx], y1)
        xx2 = np.minimum(x2[idx], x2)
        yy2 = np.minimum(y2[idx], y2)
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[idx] + areas - inter
        iou = np.where(union > 0, inter / union, 0.0)
        suppressed |= iou > iou_threshold
        suppressed[idx] = False  # don't suppress the keeper

    keep = np.array(keep, dtype=np.intp)
    return (
        filtered_boxes[keep],
        filtered_scores[keep],
        filtered_classes[keep],
        filtered_indices[keep],
    )


# ═══════════════════════════════════════════════════════════════════════
# Stage 7: Mask decode (instance segmentation)
# ═══════════════════════════════════════════════════════════════════════
# HAL equivalent: decoder/mod.rs (proto mask materialization path)
#
# YOLOv8-seg produces a ``protos`` tensor of shape (1, H_p, W_p, C_p)
# — typically (1, 160, 160, 32) — which is a spatial basis.  Each
# detection's mask is decoded as:
#
#   mask_logits = mask_coefficients @ protos.reshape(C_p, H_p * W_p)
#   mask_logits = mask_logits.reshape(H_p, W_p)
#   mask = sigmoid(mask_logits) > 0.5   (or > 0.0 on raw logits)
#
# The HAL's GPU path uses OpenGL shaders for the matmul + crop; this
# reference uses NumPy.


def decode_masks(
    mask_coefs: np.ndarray,
    protos_raw: np.ndarray,
    protos_quant: Optional[Tuple[float, int]],
    boxes_xyxy: np.ndarray,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Decode instance masks from prototype coefficients.

    HAL equivalent: the GPU mask materialization pipeline, which
    computes ``coefficients @ protos`` via OpenGL compute shaders and
    crops to bounding boxes.

    Parameters
    ----------
    mask_coefs : (M, NM) float32 mask coefficients (post-NMS)
    protos_raw : np.ndarray
        Raw protos tensor, shape (1, H_p, W_p, NM), possibly quantized.
    protos_quant : (scale, zero_point) or None
    boxes_xyxy : (M, 4) bounding boxes in pixel coordinates
    img_w, img_h : model input dimensions (e.g. 640, 640)

    Returns
    -------
    np.ndarray
        Binary masks of shape (M, H_p, W_p), thresholded at logit > 0.
    """
    if mask_coefs is None or mask_coefs.shape[0] == 0:
        return np.empty((0, 0, 0), dtype=np.float32)

    # Dequantize protos
    protos = _dequantize_nhwc(protos_raw, protos_quant)

    # protos is (H_p, W_p, C) — transpose to (C, H_p * W_p)
    h_p, w_p, c = protos.shape
    protos_flat = protos.reshape(-1, c).T  # (C, H_p * W_p)

    # Matrix multiply: (M, C) @ (C, H_p * W_p) → (M, H_p * W_p)
    mask_logits = mask_coefs @ protos_flat
    mask_logits = mask_logits.reshape(-1, h_p, w_p)

    # Crop to bounding box regions and threshold.
    # Scale boxes from pixel coords to proto resolution.
    scale_x = w_p / img_w
    scale_y = h_p / img_h

    for i in range(mask_logits.shape[0]):
        x1 = max(0, int(boxes_xyxy[i, 0] * scale_x))
        y1 = max(0, int(boxes_xyxy[i, 1] * scale_y))
        x2 = min(w_p, int(boxes_xyxy[i, 2] * scale_x + 0.5))
        y2 = min(h_p, int(boxes_xyxy[i, 3] * scale_y + 0.5))

        # Zero out everything outside the box
        crop_mask = np.zeros((h_p, w_p), dtype=np.float32)
        if x2 > x1 and y2 > y1:
            crop_mask[y1:y2, x1:x2] = 1.0
        mask_logits[i] *= crop_mask

    # Threshold on logits (> 0 corresponds to sigmoid > 0.5)
    return (mask_logits > 0.0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
# Full pipeline: end-to-end per-scale decode
# ═══════════════════════════════════════════════════════════════════════

def decode_per_scale(
    outputs: List[np.ndarray],
    layout: PerScaleLayout,
    shape_map: Dict[Tuple[int, ...], int],
    img_w: int = 640,
    img_h: int = 640,
    iou_threshold: float = 0.7,
    score_threshold: float = 0.001,
    max_detections: int = 300,
) -> dict:
    """Full per-scale decode pipeline: stages 2–7.

    This is the top-level function that the HAL's ``Decoder::decode_proto``
    should implement natively.  It runs the complete chain:

      dequantize → DFL decode → sigmoid → concat → NMS → mask decode

    Parameters
    ----------
    outputs : List[np.ndarray]
        Raw TFLite output tensors.
    layout : PerScaleLayout
        Parsed metadata.
    shape_map : Dict[Tuple[int, ...], int]
        Physical tensor shape→index binding.
    img_w, img_h : int
        Model input dimensions.
    iou_threshold, score_threshold : float
        NMS parameters.
    max_detections : int
        Maximum number of detections to return.

    Returns
    -------
    dict with keys:
        - ``boxes_xyxy``: (M, 4) xyxy in pixel coordinates
        - ``scores``: (M,) detection scores
        - ``classes``: (M,) class indices
        - ``masks``: (M, H_p, W_p) binary masks or empty array
        - ``num_detections``: int
    """
    # Stages 2–5: per-scale dequant + DFL + sigmoid + concat
    boxes_xcycwh, scores, mask_coefs = decode_all_scales(
        outputs, layout, shape_map)

    # Stage 6: NMS
    boxes_xyxy, det_scores, det_classes, det_indices = nms_class_agnostic(
        boxes_xcycwh, scores,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
        max_detections=max_detections,
    )

    # Stage 7: Mask decode (if segmentation model)
    masks = np.empty((0, 0, 0), dtype=np.float32)
    if mask_coefs is not None and layout.protos_shape is not None:
        det_mask_coefs = mask_coefs[det_indices] if len(det_indices) > 0 else mask_coefs[:0]
        protos_idx = shape_map[layout.protos_shape]
        masks = decode_masks(
            det_mask_coefs,
            outputs[protos_idx],
            layout.protos_quant,
            boxes_xyxy,
            img_w, img_h,
        )

    return {
        "boxes_xyxy": boxes_xyxy,
        "scores": det_scores,
        "classes": det_classes,
        "masks": masks,
        "num_detections": len(det_scores),
    }


# ═══════════════════════════════════════════════════════════════════════
# CLI driver
# ═══════════════════════════════════════════════════════════════════════

def main():
    """Run per-scale decode on a TFLite model with a test image.

    Usage::

        python per_scale_decode_reference.py model.tflite [image.jpg]

    If no image is provided, runs with a synthetic random input to
    verify the pipeline executes without error.
    """
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(positional) < 1:
        print(f"Usage: {sys.argv[0]} model.tflite [image.jpg] [--json-out]")
        sys.exit(1)
    model_path = positional[0]
    image_path = positional[1] if len(positional) > 1 else None
    json_out = "--json-out" in sys.argv[1:]

    # When emitting JSON we must keep stdout machine-parseable; route
    # diagnostic prints to stderr.
    log = (lambda *a, **kw: print(*a, file=sys.stderr, **kw)) if json_out else print

    # Load metadata
    metadata = load_metadata(model_path)
    layout = PerScaleLayout(metadata)

    log(f"Model: {model_path}")
    log(f"Box encoding: {layout.box_encoding}")
    log(f"FPN levels: {len(layout.fpn_levels)}")
    for lvl in layout.fpn_levels:
        log(f"  stride={lvl['stride']:2.0f}  "
            f"spatial={lvl['h']}×{lvl['w']}  "
            f"box_ch={lvl['box_shape'][-1]}  "
            f"nc={lvl['nc']}  nm={lvl['nm']}")
    log(f"Total anchors: {layout.total_anchors}")
    log(f"Protos shape: {layout.protos_shape}")
    log()

    # Load TFLite model
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError:
            print("ERROR: tflite_runtime or tensorflow required.")
            sys.exit(1)

    interp = Interpreter(model_path=model_path)
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    # Build shape map
    shape_map = bind_outputs(layout, output_details)

    # Get input shape
    input_shape = input_details[0]["shape"]
    _, in_h, in_w, in_c = input_shape
    input_dtype = input_details[0]["dtype"]

    log(f"Input: {input_shape} {input_dtype.__name__}")
    log(f"Outputs: {len(output_details)}")
    for d in output_details:
        log(f"  {str(tuple(d['shape'])):30s}  {np.dtype(d['dtype']).name}")
    log()

    # Prepare input image
    if image_path:
        from PIL import Image
        img = Image.open(image_path).convert("RGB")
        img = img.resize((in_w, in_h))
        img_np = np.array(img, dtype=np.uint8)
    else:
        log("No image provided — using synthetic input for smoke test.")
        img_np = np.random.randint(0, 255, (in_h, in_w, in_c), dtype=np.uint8)

    # Quantize input if needed
    input_data = img_np[np.newaxis]
    if input_dtype != np.uint8:
        input_data = input_data.astype(input_dtype)

    # Run inference
    interp.set_tensor(input_details[0]["index"], input_data)
    interp.invoke()
    outputs = [interp.get_tensor(d["index"]) for d in output_details]

    # Run decode
    result = decode_per_scale(
        outputs, layout, shape_map,
        img_w=in_w, img_h=in_h,
        iou_threshold=0.7,
        score_threshold=0.25,
        max_detections=300,
    )

    if json_out:
        json.dump({
            "boxes_xyxy": result["boxes_xyxy"].tolist(),
            "scores": result["scores"].tolist(),
            "classes": [int(c) for c in result["classes"]],
            "num_detections": result["num_detections"],
        }, sys.stdout)
        return

    # Print results
    n = result["num_detections"]
    print(f"Detections: {n}")
    if n > 0:
        labels = metadata.get("dataset", {}).get("classes", [])
        for i in range(min(n, 20)):
            cls = int(result["classes"][i])
            label = labels[cls] if cls < len(labels) else f"class_{cls}"
            score = result["scores"][i]
            box = result["boxes_xyxy"][i]
            print(f"  [{i:3d}] {label:20s}  score={score:.3f}  "
                  f"box=[{box[0]:.1f}, {box[1]:.1f}, "
                  f"{box[2]:.1f}, {box[3]:.1f}]")
        if n > 20:
            print(f"  ... and {n - 20} more")

        if result["masks"].size > 0:
            print(f"Masks: {result['masks'].shape}")


if __name__ == "__main__":
    main()
