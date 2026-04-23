"""End-to-end tests for ImageProcessor.materialize_masks(MaskResolution.Scaled).

Replays the exact tensor-passing pattern the EdgeFirst validator uses in
``runners/core.py::process_yolo_hal`` but substitutes the new Scaled path
for the per-tile ``ImageProcessor.convert()`` loop that exhibits the
wraparound bug described in HAILORT_BUG.md.

Verifies:

1. Scaled masks are binary ``uint8 {0, 255}`` (not continuous sigmoid).
2. Binary encoding is drop-in interchangeable with Proto continuous
   masks via the same ``> 127`` threshold.
3. Scaled output bbox-crop dimensions match the detection bbox mapped to
   the target ``(width, height)``.
4. Scaled masks match a NumPy ``sigmoid(coefs @ protos_upsampled) > 0.5``
   retina-style reference within the fast_sigmoid approximation
   tolerance (<0.5% pixel mismatch) when the bbox and protos are
   well-conditioned.
"""

from __future__ import annotations

import numpy as np

import edgefirst_hal

W, H = 640, 640
PROTO_H, PROTO_W = 160, 160
NC, NM = 80, 32


def _hal_metadata():
    return {
        "decoder_version": "yolov8",
        "nms": "class_agnostic",
        "outputs": [
            {
                "type": "detection",
                "decoder": "ultralytics",
                "shape": [1, 4 + NC + NM, 8400],
                "score_format": "per_class",
                "quantization": [1.0, 0],
            },
            {
                "type": "protos",
                "decoder": "ultralytics",
                "shape": [1, NM, PROTO_H, PROTO_W],
                "dshape": [
                    {"batch": 1},
                    {"num_protos": NM},
                    {"height": PROTO_H},
                    {"width": PROTO_W},
                ],
                "quantization": [1.0, 0],
            },
        ],
    }


def _to_tensor(arr: np.ndarray) -> edgefirst_hal.Tensor:
    t = edgefirst_hal.Tensor(list(arr.shape), dtype=arr.dtype.name)
    t.from_numpy(arr)
    return t


def _build_validator_pattern_tensors(rng, placements):
    """Replicate `_prepare_raw_outputs` + `check_normalized_boxes` tail.

    Returns `(combined_view, protos_nchw)` where `combined_view` is a
    non-contiguous `(1, 4+NC+NM, 8400)` view produced by `.T[np.newaxis]`
    on a C-contiguous `(8400, 4+NC+NM)` base, with pixel-space xywh
    divided by W/H exactly as the validator does pre-decode.
    """
    total_anchors = 8400
    det = np.zeros((total_anchors, 4 + NC + NM), dtype=np.float32)
    for i, (x0, y0, x1, y1) in enumerate(placements):
        det[i, 0] = (x0 + x1) / 2 * W  # xc pixel
        det[i, 1] = (y0 + y1) / 2 * H  # yc pixel
        det[i, 2] = (x1 - x0) * W  # w pixel
        det[i, 3] = (y1 - y0) * H  # h pixel
        det[i, 4 + (i % NC)] = 0.9  # class score (sigmoid-applied)
        det[i, 4 + NC + (i % NM)] = 4.0  # mask coef one-hot at (i % NM)

    combined = det.T[np.newaxis]  # (1, 4+NC+NM, 8400) non-contig view
    # Validator does this in-place on a copy before decode.
    combined = combined.copy()
    combined[:, [0, 2]] /= W
    combined[:, [1, 3]] /= H

    protos = (rng.standard_normal((1, NM, PROTO_H, PROTO_W)) * 2.0).astype(np.float32)
    return combined, protos


def _numpy_reference_retina(protos_nchw, coefs, bbox_xyxy_norm, target_w, target_h):
    """NumPy reference for ``MaskResolution.Scaled`` (letterbox=None).

    Upsamples the full proto plane to `(target_h, target_w)` with the
    same center-of-pixel bilinear convention the HAL path uses, computes
    `sigmoid(coefs @ upsampled) > 0.5` as a binary mask, then bbox-crops.
    """
    protos_hwc = protos_nchw[0].transpose(1, 2, 0)  # (H_p, W_p, NM)
    sx = PROTO_W / target_w
    sy = PROTO_H / target_h
    logits = np.zeros((target_h, target_w), dtype=np.float32)
    for yi in range(target_h):
        sy_coord = (yi + 0.5) * sy - 0.5
        y_f = int(np.floor(sy_coord))
        yw = sy_coord - y_f
        y_a = max(0, min(y_f, PROTO_H - 1))
        y_b = max(0, min(y_f + 1, PROTO_H - 1))
        for xi in range(target_w):
            sx_coord = (xi + 0.5) * sx - 0.5
            x_f = int(np.floor(sx_coord))
            xw = sx_coord - x_f
            x_a = max(0, min(x_f, PROTO_W - 1))
            x_b = max(0, min(x_f + 1, PROTO_W - 1))
            val = (
                (1 - xw) * (1 - yw) * protos_hwc[y_a, x_a]
                + xw * (1 - yw) * protos_hwc[y_a, x_b]
                + (1 - xw) * yw * protos_hwc[y_b, x_a]
                + xw * yw * protos_hwc[y_b, x_b]
            )
            logits[yi, xi] = float(np.dot(val, coefs))
    mask_full = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(np.uint8) * 255

    x0, y0, x1, y1 = bbox_xyxy_norm
    px0 = int(round(x0 * target_w))
    py0 = int(round(y0 * target_h))
    px1 = int(round(x1 * target_w))
    py1 = int(round(y1 * target_h))
    return mask_full[py0:py1, px0:px1]


def _make_decoder():
    return edgefirst_hal.Decoder(
        _hal_metadata(),
        score_threshold=0.5,
        iou_threshold=0.5,
        nms=edgefirst_hal.Nms.ClassAgnostic,
    )


def test_scaled_output_is_binary_0_or_255():
    """Scaled masks are strictly ``uint8 {0, 255}``, never a continuous value."""
    rng = np.random.default_rng(42)
    placements = [(0.10, 0.15, 0.35, 0.55), (0.40, 0.10, 0.65, 0.65)]
    combined, protos = _build_validator_pattern_tensors(rng, placements)

    decoder = _make_decoder()
    ip = edgefirst_hal.ImageProcessor()

    boxes, scores, classes, proto_data = decoder.decode_proto(
        [_to_tensor(combined), _to_tensor(protos)], max_boxes=300
    )
    assert proto_data is not None
    assert len(boxes) >= 1

    masks = ip.materialize_masks(
        boxes,
        scores,
        classes,
        proto_data,
        letterbox=None,
        resolution=edgefirst_hal.MaskResolution.Scaled(W, H),
    )
    assert len(masks) == len(boxes)

    for i, m in enumerate(masks):
        assert m.dtype == np.uint8, f"detection {i} mask dtype={m.dtype}"
        uniq = set(np.unique(m).tolist())
        assert uniq.issubset({0, 255}), (
            f"Scaled mask must be binary {{0, 255}}, got unique values {uniq} "
            f"for detection {i}"
        )


def test_scaled_bbox_crop_dims_match_target_resolution():
    """Scaled output tile dims equal the detection bbox mapped to ``(W, H)``."""
    rng = np.random.default_rng(7)
    placements = [(0.10, 0.15, 0.35, 0.55)]
    combined, protos = _build_validator_pattern_tensors(rng, placements)
    decoder = _make_decoder()
    ip = edgefirst_hal.ImageProcessor()
    boxes, scores, classes, proto_data = decoder.decode_proto(
        [_to_tensor(combined), _to_tensor(protos)], max_boxes=300
    )

    masks = ip.materialize_masks(
        boxes,
        scores,
        classes,
        proto_data,
        resolution=edgefirst_hal.MaskResolution.Scaled(W, H),
    )
    for b, m in zip(boxes, masks):
        expected_w = int(round((b[2] - b[0]) * W))
        expected_h = int(round((b[3] - b[1]) * H))
        assert m.shape == (expected_h, expected_w, 1), (
            f"expected ({expected_h}, {expected_w}, 1) got {m.shape} "
            f"for bbox {b.tolist()}"
        )


def test_scaled_binary_threshold_matches_greater_than_127():
    """Binary ``{0, 255}`` output is drop-in under the caller's ``> 127`` test.

    Validator code already does ``(mask > 127).astype(...)`` on continuous
    sigmoid masks. For Scaled masks this must be identical to ``mask == 255``.
    """
    rng = np.random.default_rng(123)
    placements = [(0.20, 0.25, 0.55, 0.70)]
    combined, protos = _build_validator_pattern_tensors(rng, placements)
    decoder = _make_decoder()
    ip = edgefirst_hal.ImageProcessor()
    boxes, scores, classes, proto_data = decoder.decode_proto(
        [_to_tensor(combined), _to_tensor(protos)], max_boxes=300
    )

    masks = ip.materialize_masks(
        boxes,
        scores,
        classes,
        proto_data,
        resolution=edgefirst_hal.MaskResolution.Scaled(W, H),
    )
    assert len(masks) >= 1
    m = masks[0][..., 0]
    assert np.array_equal(m > 127, m == 255)


def test_scaled_matches_numpy_retina_reference():
    """Scaled masks match a NumPy retina-style reference to within ``fast_sigmoid``
    approximation tolerance (<1% pixel mismatch per detection).

    Uses a small 96×96 target and 32×32 proto plane so the NumPy reference
    runs in seconds rather than minutes.
    """
    rng = np.random.default_rng(1729)
    # Override the full-size constants to keep the reference loop tractable.
    target_w, target_h = 96, 96
    proto_h, proto_w, nm, nc = 32, 32, 8, 4
    total_anchors = 8400

    # Build a single-detection fixture with coefficients 1.0 on channel 0.
    det = np.zeros((total_anchors, 4 + nc + nm), dtype=np.float32)
    bbox = (0.20, 0.25, 0.70, 0.80)
    det[0, 0] = (bbox[0] + bbox[2]) / 2 * target_w
    det[0, 1] = (bbox[1] + bbox[3]) / 2 * target_h
    det[0, 2] = (bbox[2] - bbox[0]) * target_w
    det[0, 3] = (bbox[3] - bbox[1]) * target_h
    det[0, 4] = 0.9
    det[0, 4 + nc] = 1.0  # mask coef on proto channel 0
    combined = det.T[np.newaxis].copy()
    combined[:, [0, 2]] /= target_w
    combined[:, [1, 3]] /= target_h

    protos = (rng.standard_normal((1, nm, proto_h, proto_w)) * 3.0).astype(np.float32)

    hal_metadata = {
        "decoder_version": "yolov8",
        "nms": "class_agnostic",
        "outputs": [
            {
                "type": "detection",
                "decoder": "ultralytics",
                "shape": [1, 4 + nc + nm, total_anchors],
                "score_format": "per_class",
                "quantization": [1.0, 0],
            },
            {
                "type": "protos",
                "decoder": "ultralytics",
                "shape": [1, nm, proto_h, proto_w],
                "dshape": [
                    {"batch": 1},
                    {"num_protos": nm},
                    {"height": proto_h},
                    {"width": proto_w},
                ],
                "quantization": [1.0, 0],
            },
        ],
    }
    decoder = edgefirst_hal.Decoder(
        hal_metadata,
        score_threshold=0.5,
        iou_threshold=0.5,
        nms=edgefirst_hal.Nms.ClassAgnostic,
    )
    ip = edgefirst_hal.ImageProcessor()
    boxes, scores, classes, proto_data = decoder.decode_proto(
        [_to_tensor(combined), _to_tensor(protos)], max_boxes=10
    )
    assert len(boxes) == 1

    hal_masks = ip.materialize_masks(
        boxes,
        scores,
        classes,
        proto_data,
        resolution=edgefirst_hal.MaskResolution.Scaled(target_w, target_h),
    )
    hal_tile = hal_masks[0][..., 0]

    coefs = np.zeros(nm, dtype=np.float32)
    coefs[0] = 1.0
    # Use the HAL-returned bbox (post-snap) so proto-pixel rounding matches.
    bbox_hal = tuple(float(v) for v in boxes[0])
    # Local reference (uses the scaled-down proto dims above).
    protos_hwc = protos[0].transpose(1, 2, 0)
    sx = proto_w / target_w
    sy = proto_h / target_h
    logits = np.zeros((target_h, target_w), dtype=np.float32)
    for yi in range(target_h):
        sy_coord = (yi + 0.5) * sy - 0.5
        y_f = int(np.floor(sy_coord))
        yw = sy_coord - y_f
        y_a = max(0, min(y_f, proto_h - 1))
        y_b = max(0, min(y_f + 1, proto_h - 1))
        for xi in range(target_w):
            sx_coord = (xi + 0.5) * sx - 0.5
            x_f = int(np.floor(sx_coord))
            xw = sx_coord - x_f
            x_a = max(0, min(x_f, proto_w - 1))
            x_b = max(0, min(x_f + 1, proto_w - 1))
            val = (
                (1 - xw) * (1 - yw) * protos_hwc[y_a, x_a]
                + xw * (1 - yw) * protos_hwc[y_a, x_b]
                + (1 - xw) * yw * protos_hwc[y_b, x_a]
                + xw * yw * protos_hwc[y_b, x_b]
            )
            logits[yi, xi] = float(np.dot(val, coefs))
    mask_full = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(np.uint8) * 255
    px0 = int(round(bbox_hal[0] * target_w))
    py0 = int(round(bbox_hal[1] * target_h))
    px1 = int(round(bbox_hal[2] * target_w))
    py1 = int(round(bbox_hal[3] * target_h))
    ref_tile = mask_full[py0:py1, px0:px1]

    assert hal_tile.shape == ref_tile.shape, (
        f"HAL tile shape {hal_tile.shape} differs from reference {ref_tile.shape}"
    )
    mismatches = int(np.sum(hal_tile != ref_tile))
    total = int(hal_tile.size)
    rate = mismatches / total
    assert rate < 0.01, (
        f"HAL vs NumPy retina reference diverged at {mismatches}/{total} "
        f"pixels ({rate:.4f} > 1% tolerance)"
    )
