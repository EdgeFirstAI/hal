"""Python-level reproducer for HAILORT_BUG.md.

Mirrors the exact tensor-passing pattern that
`edgefirst-validator`'s `process_yolo_hal` uses on the Hailo path:

  1. Build per-FPN-scale parts shaped `(H*W, 4 + nc + nm)` and
     `np.concatenate(axis=0)` into `(total_anchors, 4 + nc + nm)`
     C-contiguous.
  2. Apply `.T[np.newaxis]` to get a **non-contiguous view** of shape
     `(1, 4 + nc + nm, total_anchors)` — the same memory layout the
     validator's `_prepare_raw_outputs` returns.
  3. Apply the in-place `x[:, [0, 2]] /= width` normalization in the
     same 3-dim-fancy-index style `check_normalized_boxes` uses.
  4. Instantiate `edgefirst_hal.Tensor(shape, dtype).from_numpy(view)`
     and pass to `decoder.decode(...)` exactly as the validator does.
  5. Check each surviving detection's mask was rendered from the
     mask_coef row at its own anchor index.

Encodes an obvious high/low signal in mask_coefs + protos so any
mask-to-detection index swap would flip the rendered mean by
~250 u8 values.
"""

from __future__ import annotations

import numpy as np
import edgefirst_hal


def _build_combined_detection_tensor(
    per_scale_parts, normalize_boxes, model_input=(640, 640)
):
    """Replicate `_prepare_raw_outputs` tail + `check_normalized_boxes`.

    * `per_scale_parts`: list of `(H*W, 4 + nc + nm)` float32 arrays.
    * `normalize_boxes`: when True, divide xc/yc/w/h by model input size
      (validator does this for pixel-space boxes from Hailo DFL decode).
    """
    width, height = model_input
    detection = np.concatenate(per_scale_parts, axis=0)  # (A, F)
    detection = detection.T[np.newaxis]  # (1, F, A) view
    if normalize_boxes:
        # This is exactly what `check_normalized_boxes` does on a
        # (1, 4, A) slice when the boundary fraction is < 0.80.
        detection[:, [0, 2]] /= width
        detection[:, [1, 3]] /= height
    return detection


def test_hailort_validator_pattern_preserves_mask_detection_pairing():
    """Full Python-stack replay of the validator's pattern.

    Three FPN levels (strides 8/16/32) at 640×640, reg_max post-decode
    so boxes are already xcycwh (we build xywh directly). Ten anchors
    carry unique mask_coef one-hot signatures; protos are per-channel
    distinct so a correctly-paired mask sigmoid-saturates (→ 255) and
    any mis-indexed lookup renders as sigmoid(0) ≈ 0.5 (→ 128).
    """
    nc = 80
    nm = 32
    feat = 4 + nc + nm  # 116
    width = height = 640
    # Cheap grids — we don't need every anchor, just enough to put 10
    # above threshold in three different FPN blocks.
    fpn = [(80, 80, 8), (40, 40, 16), (20, 20, 32)]

    # Build per-FPN-scale chunks. Most anchors are zero; ten specific
    # anchors get a valid detection + one-hot mask_coef at channel K.
    parts = []
    # Target anchors: (part_idx, local_row, k)
    targets = []
    target_count = 10
    for idx in range(target_count):
        part_idx = idx % 3
        h, w, stride = fpn[part_idx]
        n_anchors = h * w
        local_row = (idx * 237) % n_anchors
        k = idx  # one-hot channel equals anchor index in [0, NM=32)
        targets.append((part_idx, local_row, k, stride))

    for p, (h, w, stride) in enumerate(fpn):
        n_anchors = h * w
        chunk = np.zeros((n_anchors, feat), dtype=np.float32)
        # Fill each target anchor
        for part_idx, local_row, k, s in targets:
            if part_idx != p:
                continue
            # Place a non-overlapping box centred at a grid-derived
            # position in pixel coords. Width/height small (~20px) so
            # NMS doesn't suppress across targets.
            row = local_row // w
            col = local_row % w
            xc = (col + 0.5) * stride
            yc = (row + 0.5) * stride
            chunk[local_row, 0] = xc
            chunk[local_row, 1] = yc
            chunk[local_row, 2] = 20.0  # width 20 px
            chunk[local_row, 3] = 20.0  # height 20 px
            chunk[local_row, 4] = 0.95  # class 0 score (pre-sigmoided)
            # One-hot mask_coefs at channel k, value +10 (strong)
            chunk[local_row, 4 + nc + k] = 10.0
        parts.append(chunk)

    detection = _build_combined_detection_tensor(
        parts, normalize_boxes=True, model_input=(width, height)
    )
    assert detection.shape == (1, feat, 80 * 80 + 40 * 40 + 20 * 20)
    # Sanity: the view is NOT C-contiguous — that's the validator's
    # pattern and the one we specifically want to test.
    assert not detection.flags["C_CONTIGUOUS"], (
        "test setup bug: combined detection tensor must be "
        ".T[np.newaxis] non-contig to replicate validator"
    )

    # Protos NCHW (1, 32, 160, 160). Channel k all-`5.0 + 0.1*k`.
    protos = np.zeros((1, nm, 160, 160), dtype=np.float32)
    for k in range(nm):
        protos[0, k] = 5.0 + 0.1 * k

    # Construct decoder via the dict API (validator's `init_decoder`
    # path at runners/core.py:164).
    hal_metadata = {
        "decoder_version": "yolov8",
        "nms": "class_agnostic",
        "outputs": [
            {
                "type": "detection",
                "decoder": "ultralytics",
                "shape": [1, feat, detection.shape[2]],
                "score_format": "per_class",
                "quantization": [1.0, 0],
            },
            {
                "type": "protos",
                "decoder": "ultralytics",
                "shape": [1, nm, 160, 160],
                "dshape": [
                    {"batch": 1},
                    {"num_protos": nm},
                    {"height": 160},
                    {"width": 160},
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

    # Pass tensors via from_numpy exactly like validator's
    # process_yolo_hal at runners/core.py:795-804.
    det_tensor = edgefirst_hal.Tensor(list(detection.shape), dtype="float32")
    det_tensor.from_numpy(detection)
    proto_tensor = edgefirst_hal.Tensor(list(protos.shape), dtype="float32")
    proto_tensor.from_numpy(protos)

    boxes, scores, classes, masks = decoder.decode(
        [det_tensor, proto_tensor], max_boxes=100
    )
    assert len(boxes) == target_count, (
        f"expected {target_count} detections, got {len(boxes)}"
    )
    assert len(masks) == target_count

    # For each surviving detection, verify its rendered mask was
    # looked up at the CORRECT anchor index. Correct pairings render
    # to ≈255; any misalignment would render to sigmoid(0) ≈ 128.
    for i, (b, m) in enumerate(zip(boxes, masks)):
        mean = float(np.mean(m))
        assert mean > 240.0, (
            f"detection {i} box={b} mean mask value {mean} indicates "
            f"mask-coef row was looked up at the WRONG anchor index "
            f"(expected ~255 for correct one-hot → per-channel-unique "
            f"proto lookup; got {mean} which suggests "
            f"sigmoid(0)≈128 from a zeroed mask_coefs row, "
            f"i.e. mask-detection misalignment)"
        )


def test_hailort_validator_pattern_with_tensor_cache_across_frames():
    """Replicate `_TENSOR_CACHE` reuse across multiple frames.

    The validator's `process_yolo_hal` re-uses the same `Tensor`
    across iterations via the module-level `_TENSOR_CACHE` dict,
    calling `tensor.from_numpy(output)` each frame. If `from_numpy`
    leaks stale data or fails to fully overwrite the buffer when
    the source view is non-contiguous, the decode would pair each
    frame's detections against some mix of stale + current
    mask_coefs — manifesting as incorrect mask rendering.
    """
    nc = 80
    nm = 32
    feat = 4 + nc + nm
    width = height = 640
    fpn = [(80, 80, 8), (40, 40, 16), (20, 20, 32)]
    total_anchors = sum(h * w for h, w, _ in fpn)

    hal_metadata = {
        "decoder_version": "yolov8",
        "nms": "class_agnostic",
        "outputs": [
            {
                "type": "detection",
                "decoder": "ultralytics",
                "shape": [1, feat, total_anchors],
                "score_format": "per_class",
                "quantization": [1.0, 0],
            },
            {
                "type": "protos",
                "decoder": "ultralytics",
                "shape": [1, nm, 160, 160],
                "dshape": [
                    {"batch": 1},
                    {"num_protos": nm},
                    {"height": 160},
                    {"width": 160},
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
    # Cache-style reuse: allocate once, from_numpy each frame.
    det_tensor = edgefirst_hal.Tensor([1, feat, total_anchors], dtype="float32")
    proto_tensor = edgefirst_hal.Tensor([1, nm, 160, 160], dtype="float32")

    protos = np.zeros((1, nm, 160, 160), dtype=np.float32)
    for k in range(nm):
        protos[0, k] = 5.0 + 0.1 * k
    proto_tensor.from_numpy(protos)

    # Iterate 5 frames, each with DIFFERENT target anchors. If a stale
    # buffer leaks, a later frame's detection would pick up an earlier
    # frame's mask_coefs → mismatched mask mean vs expected.
    rng_sequence = [0xAAAA, 0xBBBB, 0xCCCC, 0xDDDD, 0xEEEE]
    # Pre-compute the decoded centre for every anchor so we can enforce a
    # spatial separation when sampling. Without this, two stride-8 neighbours
    # could both be picked, their 20-px boxes overlap above IoU 0.5, and NMS
    # would suppress one — making the stale-buffer assertion flaky for
    # reasons unrelated to what this test is probing.
    anchor_centres = []
    offset = 0
    for h, w, stride in fpn:
        for local_row in range(h * w):
            row, col = divmod(local_row, w)
            xc = (col + 0.5) * stride
            yc = (row + 0.5) * stride
            anchor_centres.append((offset + local_row, xc, yc))
        offset += h * w

    for frame_idx, seed in enumerate(rng_sequence):
        rng = np.random.default_rng(seed)
        # Pick 10 spatially separated anchors (centres ≥ 20 px apart in both
        # axes — i.e. non-overlapping box footprints) + random one-hot k.
        order = rng.permutation(len(anchor_centres))
        targets = []
        selected_centres = []
        for candidate_idx in order:
            idx, xc, yc = anchor_centres[int(candidate_idx)]
            if any(
                abs(xc - pxc) < 20.0 and abs(yc - pyc) < 20.0
                for pxc, pyc in selected_centres
            ):
                continue
            k = int(rng.integers(0, nm))
            targets.append((idx, k))
            selected_centres.append((xc, yc))
            if len(targets) == 10:
                break
        assert len(targets) == 10, (
            f"frame {frame_idx}: failed to sample 10 well-separated anchors"
        )

        # Build per-scale parts.
        parts = []
        offset = 0
        for p, (h, w, stride) in enumerate(fpn):
            n_anchors = h * w
            chunk = np.zeros((n_anchors, feat), dtype=np.float32)
            for tgt_idx, k in targets:
                if not (offset <= tgt_idx < offset + n_anchors):
                    continue
                local_row = tgt_idx - offset
                row = local_row // w
                col = local_row % w
                chunk[local_row, 0] = (col + 0.5) * stride
                chunk[local_row, 1] = (row + 0.5) * stride
                chunk[local_row, 2] = 20.0
                chunk[local_row, 3] = 20.0
                chunk[local_row, 4] = 0.95
                chunk[local_row, 4 + nc + k] = 10.0
            parts.append(chunk)
            offset += n_anchors

        detection = _build_combined_detection_tensor(
            parts, normalize_boxes=True, model_input=(width, height)
        )
        det_tensor.from_numpy(detection)
        boxes, scores, classes, masks = decoder.decode(
            [det_tensor, proto_tensor], max_boxes=100
        )
        assert len(boxes) == 10, (
            f"frame {frame_idx}: expected 10 detections, got {len(boxes)}"
        )
        for i, (b, m) in enumerate(zip(boxes, masks)):
            mean = float(np.mean(m))
            assert mean > 240.0, (
                f"frame {frame_idx} detection {i}: mean mask value "
                f"{mean} < 240 → likely stale-buffer leak in "
                f"Tensor.from_numpy cross-frame reuse"
            )


def _numpy_reference_mask(mask_coefs, protos, bbox_norm, width, height):
    """NumPy reference for HAL's YoloSegDet mask rendering.

    * `mask_coefs`: shape `(nm,)` — anchor's mask_coefs row.
    * `protos`: shape `(nm, H_p, W_p)` NCHW — mask prototypes.
    * `bbox_norm`: `(x1, y1, x2, y2)` normalised to [0, 1].

    Returns the ROI-cropped mask as uint8 `(h_roi, w_roi)`.
    """
    H_p, W_p = protos.shape[1], protos.shape[2]
    x1 = int(max(0, bbox_norm[0] * W_p))
    y1 = int(max(0, bbox_norm[1] * H_p))
    x2 = int(np.ceil(bbox_norm[2] * W_p))
    y2 = int(np.ceil(bbox_norm[3] * H_p))
    x2 = max(x2, x1 + 1)
    y2 = max(y2, y1 + 1)
    # protos NHWC after HAL's internal protos_to_hwc: (H, W, nm)
    protos_hwc = protos.transpose(1, 2, 0)
    cropped = protos_hwc[y1:y2, x1:x2, :]  # (h, w, nm)
    mask_f = cropped @ mask_coefs  # (h, w)
    return (1.0 / (1.0 + np.exp(-mask_f)) * 255).round().astype(np.uint8)


def test_hailort_validator_pattern_dense_realistic_mask_coefs_match_numpy_reference():
    """Compare HAL's rendered masks against a NumPy reference using
    realistic dense mask_coefs (not one-hot). Mirrors the actual
    YOLOv8-seg runtime data distribution.

    If HAL's mask rendering is correctly paired with each detection,
    the per-pixel mask values should match the reference within
    rounding tolerance (± 2 u8).
    """
    nc = 80
    nm = 32
    feat = 4 + nc + nm
    width = height = 640
    fpn = [(80, 80, 8), (40, 40, 16), (20, 20, 32)]
    total_anchors = sum(h * w for h, w, _ in fpn)

    rng = np.random.default_rng(0x5E6D)

    # Place 12 non-overlapping detections at random anchors, each
    # with dense Gaussian-like mask_coefs and realistic protos.
    targets = []
    used = set()
    while len(targets) < 12:
        a = int(rng.integers(0, total_anchors))
        if a in used:
            continue
        used.add(a)
        targets.append(a)

    # Dense protos: 32 proto channels each containing a smooth
    # gradient pattern (realistic of segmentation proto outputs).
    protos = np.zeros((1, nm, 160, 160), dtype=np.float32)
    yy, xx = np.meshgrid(
        np.linspace(-1, 1, 160), np.linspace(-1, 1, 160), indexing="ij"
    )
    for k in range(nm):
        phase = k * 0.3
        protos[0, k] = np.sin(3 * xx + phase) * np.cos(2 * yy - phase)

    parts = []
    offset = 0
    anchor_mask_coefs = {}  # anchor_idx → (nm,) mask_coefs
    anchor_bbox_pix = {}  # anchor_idx → (x1, y1, x2, y2) in pixels
    for p, (h, w, stride) in enumerate(fpn):
        n_anchors = h * w
        chunk = np.zeros((n_anchors, feat), dtype=np.float32)
        for tgt in targets:
            if not (offset <= tgt < offset + n_anchors):
                continue
            local_row = tgt - offset
            row = local_row // w
            col = local_row % w
            xc_px = (col + 0.5) * stride
            yc_px = (row + 0.5) * stride
            w_px = 80.0
            h_px = 80.0
            chunk[local_row, 0] = xc_px
            chunk[local_row, 1] = yc_px
            chunk[local_row, 2] = w_px
            chunk[local_row, 3] = h_px
            chunk[local_row, 4] = 0.9
            # Dense realistic mask_coefs.
            coefs = rng.standard_normal(nm).astype(np.float32) * 0.5
            chunk[local_row, 4 + nc : 4 + nc + nm] = coefs
            anchor_mask_coefs[tgt] = coefs
            anchor_bbox_pix[tgt] = (
                xc_px - w_px / 2,
                yc_px - h_px / 2,
                xc_px + w_px / 2,
                yc_px + h_px / 2,
            )
        parts.append(chunk)
        offset += n_anchors

    detection = _build_combined_detection_tensor(
        parts, normalize_boxes=True, model_input=(width, height)
    )

    hal_metadata = {
        "decoder_version": "yolov8",
        "nms": "class_agnostic",
        "outputs": [
            {
                "type": "detection",
                "decoder": "ultralytics",
                "shape": [1, feat, total_anchors],
                "score_format": "per_class",
                "quantization": [1.0, 0],
            },
            {
                "type": "protos",
                "decoder": "ultralytics",
                "shape": [1, nm, 160, 160],
                "dshape": [
                    {"batch": 1},
                    {"num_protos": nm},
                    {"height": 160},
                    {"width": 160},
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
    det_tensor = edgefirst_hal.Tensor(list(detection.shape), dtype="float32")
    det_tensor.from_numpy(detection)
    proto_tensor = edgefirst_hal.Tensor(list(protos.shape), dtype="float32")
    proto_tensor.from_numpy(protos)
    boxes, scores, classes, masks = decoder.decode(
        [det_tensor, proto_tensor], max_boxes=100
    )
    assert len(boxes) == len(targets)

    # For each HAL detection, find the matching anchor from the
    # original setup (by bbox centre proximity in normalised coords),
    # then compare HAL's rendered mask to a NumPy reference that uses
    # that anchor's mask_coefs directly.
    matched = 0
    for b, m in zip(boxes, masks):
        cx = (b[0] + b[2]) * 0.5
        cy = (b[1] + b[3]) * 0.5
        # Find the target anchor whose normalised bbox centre is
        # closest to this HAL detection's centre.
        best_tgt = None
        best_d2 = 1e9
        for tgt, (x1p, y1p, x2p, y2p) in anchor_bbox_pix.items():
            ncx = (x1p + x2p) / (2 * width)
            ncy = (y1p + y2p) / (2 * height)
            d2 = (ncx - cx) ** 2 + (ncy - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_tgt = tgt
        assert best_d2 < 1e-3, (
            f"HAL detection centre ({cx:.3f}, {cy:.3f}) has no "
            f"nearby target anchor; best d²={best_d2}"
        )
        ref = _numpy_reference_mask(
            anchor_mask_coefs[best_tgt],
            protos[0],
            (b[0], b[1], b[2], b[3]),
            width,
            height,
        )
        hal_mask = m.squeeze()  # (h, w, 1) → (h, w)
        assert hal_mask.shape == ref.shape, (
            f"shape mismatch: HAL={hal_mask.shape}, ref={ref.shape}"
        )
        # Allow ± 2 u8 for rounding differences.
        diff = np.abs(hal_mask.astype(np.int32) - ref.astype(np.int32))
        max_diff = int(diff.max())
        assert max_diff <= 2, (
            f"detection at ({cx:.3f}, {cy:.3f}) mask diverges from "
            f"NumPy reference by up to {max_diff} u8 — HAL used the "
            f"WRONG mask_coefs row (indicating mask-detection "
            f"misalignment per HAILORT_BUG.md hypothesis)"
        )
        matched += 1
    assert matched == len(targets)


def test_hailort_validator_pattern_contiguous_control():
    """Control: same data but passed as a C-contiguous copy.

    If the non-contig test fails and this one passes, the bug is in
    `Tensor.from_numpy`'s non-contiguous copy path. If both pass, the
    HAL decoder is not mis-pairing masks under validator-style input.
    """
    nc = 80
    nm = 32
    feat = 4 + nc + nm
    fpn = [(80, 80, 8), (40, 40, 16), (20, 20, 32)]
    parts = []
    targets = []
    for idx in range(10):
        part_idx = idx % 3
        h, w, stride = fpn[part_idx]
        local_row = (idx * 237) % (h * w)
        targets.append((part_idx, local_row, idx, stride))
    for p, (h, w, stride) in enumerate(fpn):
        chunk = np.zeros((h * w, feat), dtype=np.float32)
        for part_idx, local_row, k, s in targets:
            if part_idx != p:
                continue
            row = local_row // w
            col = local_row % w
            chunk[local_row, 0] = (col + 0.5) * stride
            chunk[local_row, 1] = (row + 0.5) * stride
            chunk[local_row, 2] = 20.0
            chunk[local_row, 3] = 20.0
            chunk[local_row, 4] = 0.95
            chunk[local_row, 4 + nc + k] = 10.0
        parts.append(chunk)

    detection_view = _build_combined_detection_tensor(parts, normalize_boxes=True)
    # Force a fresh C-contig copy.
    detection = np.ascontiguousarray(detection_view)
    assert detection.flags["C_CONTIGUOUS"]

    protos = np.zeros((1, nm, 160, 160), dtype=np.float32)
    for k in range(nm):
        protos[0, k] = 5.0 + 0.1 * k

    hal_metadata = {
        "decoder_version": "yolov8",
        "nms": "class_agnostic",
        "outputs": [
            {
                "type": "detection",
                "decoder": "ultralytics",
                "shape": [1, feat, detection.shape[2]],
                "score_format": "per_class",
                "quantization": [1.0, 0],
            },
            {
                "type": "protos",
                "decoder": "ultralytics",
                "shape": [1, nm, 160, 160],
                "dshape": [
                    {"batch": 1},
                    {"num_protos": nm},
                    {"height": 160},
                    {"width": 160},
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
    det_tensor = edgefirst_hal.Tensor(list(detection.shape), dtype="float32")
    det_tensor.from_numpy(detection)
    proto_tensor = edgefirst_hal.Tensor(list(protos.shape), dtype="float32")
    proto_tensor.from_numpy(protos)
    boxes, scores, classes, masks = decoder.decode(
        [det_tensor, proto_tensor], max_boxes=100
    )
    assert len(boxes) == 10
    for i, (b, m) in enumerate(zip(boxes, masks)):
        mean = float(np.mean(m))
        assert mean > 240.0, f"C-contig control FAILED detection {i}: mean={mean}"
