#!/usr/bin/env python3
"""Validate a decoder reference fixture against a FP32 ONNX reference.

PURPOSE
-------
A safetensors fixture under ``testdata/decoder/`` records the
``decoded.*`` output of the NumPy reference decoder running over the
TFLite per-scale (int8) model. That fixture is trusted by the HAL
parity tests as the oracle.

This script independently validates that oracle: it runs an
architecturally-equivalent FP32 ONNX export of the same model on the
same image, applies a plain class-agnostic NMS, and compares its
detections (boxes, classes, scores, masks) against the fixture's
``decoded.*``. Quantization noise and small NMS-ordering differences
are expected; the script reports IoU statistics so you can decide
whether the agreement is good enough to trust the fixture.

USAGE
-----
::

    python scripts/decoder_validate_against_onnx.py \\
        testdata/decoder/yolo26n-seg.safetensors \\
        ~/software/hailo-converter/workdir/t-2686/yolo26n-seg-t-2686.onnx \\
        --image testdata/zidane.jpg \\
        --out-dir /tmp/yolo26_compare

The output directory receives ``fixture.png``, ``onnx.png``, and
``side_by_side.png`` overlays plus the textual stats are printed to
stdout.

ASSUMPTIONS
-----------
The ONNX model must be a standard Ultralytics-style YOLO segmentation
graph with:

- ``output0`` shape ``(1, 4 + NC + NM, num_anchors)`` — channel-major
  layout: 4 normalized xywh + ``NC`` post-sigmoid class scores + ``NM``
  raw mask coefficients. Box coords are in fraction of input width /
  height.
- ``output1`` shape ``(1, NM, H_p, W_p)`` — prototype masks (fp32).

The fixture's input image dtype must be uint8 NHWC. The ONNX input
must be NCHW float32 normalized to ``[0, 1]``.

REPRODUCIBILITY
---------------
Pin onnxruntime / pillow / numpy / safetensors. Same image plus same
ONNX produces deterministic detections.

::

    pip install 'onnx>=1.16' 'onnxruntime>=1.18' 'pillow>=10' \\
        'safetensors>=0.4' 'numpy>=1.26,<2'
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import safetensors
from PIL import Image, ImageDraw, ImageFont


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert (xc, yc, w, h) → (x1, y1, x2, y2). Vectorized."""
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out


def box_iou_pairwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU on xyxy boxes — shape ``(N, 4)`` × ``(M, 4)`` → ``(N, M)``."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def nms_class_agnostic(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    iou_th: float,
    score_th: float,
    max_det: int,
) -> np.ndarray:
    """Plain class-agnostic NMS. Returns indices into the original arrays."""
    keep_mask = scores >= score_th
    boxes = boxes_xyxy[keep_mask]
    scs = scores[keep_mask]
    cls = classes[keep_mask]
    idx_in_orig = np.nonzero(keep_mask)[0]
    order = np.argsort(-scs)
    boxes = boxes[order]
    scs = scs[order]
    cls = cls[order]
    idx_in_orig = idx_in_orig[order]

    keep: list[int] = []
    while len(boxes) and len(keep) < max_det:
        keep.append(int(idx_in_orig[0]))
        if len(boxes) == 1:
            break
        ious = box_iou_pairwise(boxes[0:1], boxes[1:])[0]
        survivors = ious < iou_th
        boxes = boxes[1:][survivors]
        scs = scs[1:][survivors]
        cls = cls[1:][survivors]
        idx_in_orig = idx_in_orig[1:][survivors]
    return np.asarray(keep, dtype=np.int64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def decode_mask(
    mc: np.ndarray,
    protos_flat: np.ndarray,
    box_xyxy: np.ndarray,
    h_p: int,
    w_p: int,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """Build a single binary mask: ``sigmoid(mc @ protos)`` cropped to box, threshold > 0.5."""
    logits = sigmoid(mc @ protos_flat).reshape(h_p, w_p)
    sx = w_p / img_w
    sy = h_p / img_h
    x1 = max(0, int(box_xyxy[0] * sx))
    y1 = max(0, int(box_xyxy[1] * sy))
    x2 = min(w_p, int(box_xyxy[2] * sx + 0.5))
    y2 = min(h_p, int(box_xyxy[3] * sy + 0.5))
    crop = np.zeros((h_p, w_p), dtype=np.float32)
    if x2 > x1 and y2 > y1:
        crop[y1:y2, x1:x2] = 1.0
    return ((logits * crop) > 0.5).astype(np.uint8)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


# ----------------------------------------------------------------------
# Detection matching
# ----------------------------------------------------------------------


def greedy_match(
    boxes_a: np.ndarray, boxes_b: np.ndarray, min_iou: float = 0.3
) -> list[tuple[int, int, float]]:
    """Pair boxes_a → boxes_b greedily by descending IoU. Returns ``[(i, j, iou), ...]``."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return []
    iou = box_iou_pairwise(boxes_a, boxes_b)
    pairs: list[tuple[int, int, float]] = []
    used_b: set[int] = set()
    for i in np.argsort(-iou.max(axis=1)):
        j_order = np.argsort(-iou[i])
        for j in j_order:
            if int(j) in used_b:
                continue
            if iou[i, j] < min_iou:
                break
            pairs.append((int(i), int(j), float(iou[i, j])))
            used_b.add(int(j))
            break
    return pairs


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def render(
    image_arr: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    classes: np.ndarray,
    label: str,
    out_path: Path,
    top_k: int = 20,
) -> Path:
    img = Image.fromarray(image_arr).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except Exception:
        font = ImageFont.load_default()
    order = np.argsort(-scores)[:top_k]
    for k, i in enumerate(order):
        b = boxes[i]
        color = "yellow" if k == 0 else "lime"
        draw.rectangle(b.tolist(), outline=color, width=2)
        draw.text(
            (b[0] + 2, b[1] + 2),
            f"{int(classes[i])}:{scores[i]:.2f}",
            fill=color,
            font=font,
        )
    draw.text((8, 8), label, fill="white", font=font)
    img.save(out_path)
    return out_path


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare a decoder fixture's decoded.* against a FP32 ONNX reference."
    )
    p.add_argument("fixture", type=Path, help="Path to a .safetensors fixture.")
    p.add_argument("onnx", type=Path, help="Path to the FP32 ONNX model.")
    p.add_argument(
        "--image", type=Path, default=None,
        help="Source image. Default: testdata/zidane.jpg next to the fixture root.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("/tmp/decoder_validate_onnx"),
        help="Directory for rendered overlays.",
    )
    p.add_argument(
        "--score-threshold", type=float, default=0.001,
        help="ONNX-side NMS score threshold. Default matches fixture default.",
    )
    p.add_argument(
        "--iou-threshold", type=float, default=0.7,
        help="ONNX-side NMS IoU threshold.",
    )
    p.add_argument(
        "--max-detections", type=int, default=300,
        help="ONNX-side NMS max-detection cap.",
    )
    p.add_argument(
        "--match-iou", type=float, default=0.3,
        help="Greedy fixture↔ONNX detection-pair matching IoU floor.",
    )
    p.add_argument(
        "--top-k-render", type=int, default=20,
        help="Per overlay, render only the top-K boxes for readability.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.fixture.exists():
        print(f"ERROR: fixture not found: {args.fixture}", file=sys.stderr)
        return 1
    if not args.onnx.exists():
        print(f"ERROR: ONNX not found: {args.onnx}", file=sys.stderr)
        return 1

    image_path = args.image
    if image_path is None:
        # Default: assume the fixture is under <repo>/testdata/decoder/, image at <repo>/testdata/zidane.jpg
        image_path = args.fixture.resolve().parents[1] / "zidane.jpg"
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------------
    # Load fixture
    # ----------------------------------------------------------------------
    print(f"Loading fixture: {args.fixture}")
    with safetensors.safe_open(str(args.fixture), framework="numpy") as f:
        fix_boxes = f.get_tensor("decoded.boxes_xyxy")
        fix_scores = f.get_tensor("decoded.scores")
        fix_classes = f.get_tensor("decoded.classes")
        fix_masks = f.get_tensor("decoded.masks") if "decoded.masks" in f.keys() else None
        fix_image = f.get_tensor("input.image")[0]  # drop batch
    print(
        f"  Fixture: {len(fix_boxes)} detections, "
        f"mask shape {None if fix_masks is None else fix_masks.shape}"
    )

    # ----------------------------------------------------------------------
    # Run ONNX
    # ----------------------------------------------------------------------
    print(f"Running ONNX: {args.onnx}")
    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    in_shape = sess.get_inputs()[0].shape  # e.g. [1, 3, 640, 640]
    img_h = int(in_shape[2])
    img_w = int(in_shape[3])
    img = Image.open(image_path).convert("RGB").resize((img_w, img_h))
    x = np.array(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    out0, out1 = sess.run(None, {in_name: x})  # (1, 4+NC+NM, A), (1, NM, Hp, Wp)

    arr = out0[0].T  # (A, 4+NC+NM)
    nm = out1.shape[1]
    nc = arr.shape[1] - 4 - nm
    boxes_xywh_norm = arr[:, :4]
    class_scores = arr[:, 4 : 4 + nc]
    mask_coefs = arr[:, 4 + nc :]
    best_cls = np.argmax(class_scores, axis=1)
    best_scr = class_scores[np.arange(len(best_cls)), best_cls]
    boxes_xywh_px = boxes_xywh_norm * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    boxes_xyxy_px = xywh_to_xyxy(boxes_xywh_px)

    keep_idx = nms_class_agnostic(
        boxes_xyxy_px,
        best_scr,
        best_cls,
        args.iou_threshold,
        args.score_threshold,
        args.max_detections,
    )
    onnx_boxes = boxes_xyxy_px[keep_idx]
    onnx_scores = best_scr[keep_idx]
    onnx_classes = best_cls[keep_idx]
    onnx_mc = mask_coefs[keep_idx]
    protos = out1[0]                              # (NM, Hp, Wp)
    protos_flat = protos.reshape(nm, -1)          # (NM, Hp*Wp)
    print(f"  ONNX  : {len(onnx_boxes)} detections")

    # ----------------------------------------------------------------------
    # Match + report
    # ----------------------------------------------------------------------
    pairs = greedy_match(fix_boxes, onnx_boxes, min_iou=args.match_iou)
    print(f"\nMatched {len(pairs)} fixture↔ONNX pairs (IoU ≥ {args.match_iou})")

    if pairs:
        ious = np.array([p[2] for p in pairs])
        class_match = sum(1 for i, j, _ in pairs if int(fix_classes[i]) == int(onnx_classes[j]))
        fs = np.array([fix_scores[i] for i, _, _ in pairs])
        os_ = np.array([onnx_scores[j] for _, j, _ in pairs])
        print(f"  Box IoU      : mean={ious.mean():.3f} median={np.median(ious):.3f} min={ious.min():.3f}")
        print(f"  Class match  : {class_match}/{len(pairs)} ({100*class_match/len(pairs):.1f}%)")
        print(f"  Score corr   : {np.corrcoef(fs, os_)[0, 1]:.3f}  fix_mean={fs.mean():.3f}  onnx_mean={os_.mean():.3f}")

        if fix_masks is not None:
            h_p, w_p = fix_masks.shape[1], fix_masks.shape[2]
            mask_ious = []
            for i, j, _ in pairs:
                onnx_mask = decode_mask(onnx_mc[j], protos_flat, onnx_boxes[j], h_p, w_p, img_w, img_h)
                fix_mask = (fix_masks[i] > 0).astype(np.uint8)
                mask_ious.append(mask_iou(fix_mask, onnx_mask))
            mask_ious = np.array(mask_ious)
            print(f"  Mask IoU     : mean={mask_ious.mean():.3f} median={np.median(mask_ious):.3f} min={mask_ious.min():.3f}")
    else:
        print("  (no matches above the threshold; comparison failed)")

    # ----------------------------------------------------------------------
    # Top-K dump
    # ----------------------------------------------------------------------
    print("\nTop 10 fixture detections:")
    for i in range(min(10, len(fix_boxes))):
        print(
            f"  [{i}] cls={int(fix_classes[i]):2d}  score={fix_scores[i]:.3f}  "
            f"box=[{fix_boxes[i,0]:6.1f}, {fix_boxes[i,1]:6.1f}, "
            f"{fix_boxes[i,2]:6.1f}, {fix_boxes[i,3]:6.1f}]"
        )
    print("\nTop 10 ONNX detections:")
    for i in range(min(10, len(onnx_boxes))):
        print(
            f"  [{i}] cls={int(onnx_classes[i]):2d}  score={onnx_scores[i]:.3f}  "
            f"box=[{onnx_boxes[i,0]:6.1f}, {onnx_boxes[i,1]:6.1f}, "
            f"{onnx_boxes[i,2]:6.1f}, {onnx_boxes[i,3]:6.1f}]"
        )

    # ----------------------------------------------------------------------
    # Render overlays
    # ----------------------------------------------------------------------
    print("\nRendering overlays...")
    fix_png = render(
        fix_image, fix_boxes, fix_scores, fix_classes,
        f"FIXTURE (TFLite int8 + NumPy ref): {len(fix_boxes)} det",
        args.out_dir / "fixture.png", top_k=args.top_k_render,
    )
    onnx_png = render(
        fix_image, onnx_boxes, onnx_scores, onnx_classes,
        f"ONNX (FP32): {len(onnx_boxes)} det",
        args.out_dir / "onnx.png", top_k=args.top_k_render,
    )

    fix_img = Image.open(fix_png)
    onnx_img = Image.open(onnx_png)
    combo = Image.new("RGB", (fix_img.width + onnx_img.width + 8,
                              max(fix_img.height, onnx_img.height)), "black")
    combo.paste(fix_img, (0, 0))
    combo.paste(onnx_img, (fix_img.width + 8, 0))
    combo_path = args.out_dir / "side_by_side.png"
    combo.save(combo_path)

    print(f"  fixture overlay: {fix_png}")
    print(f"  onnx    overlay: {onnx_png}")
    print(f"  side-by-side   : {combo_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
