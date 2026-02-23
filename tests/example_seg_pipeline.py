#!/usr/bin/env python3
"""Full segmentation pipeline comparison: HAL (OpenGL) vs OpenCV.

Loads zidane.jpg, runs YOLOv8n-seg INT8 TFLite inference, decodes outputs,
and renders colored instance masks at 640x640. Both pipelines produce a
PNG output for visual inspection and are benchmarked end-to-end.

Pipeline stages:
  1. Image load
  2. Letterbox resize to 640x640
  3. TFLite INT8 inference
  4. Output decoding (NMS + mask computation)
  5. Colored mask rendering at 640x640

Usage:
    # Run both pipelines, save outputs for visual inspection:
    python tests/example_seg_pipeline.py

    # Benchmark mode (N iterations, skip visual output):
    python tests/example_seg_pipeline.py --bench -n 50

    # HAL only:
    python tests/example_seg_pipeline.py --hal-only

    # OpenCV only:
    python tests/example_seg_pipeline.py --opencv-only
"""

import argparse
import io
import json
import os
import statistics
import time
import zipfile

import cv2
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAL_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_IMAGE = os.path.join(HAL_ROOT, "testdata", "zidane.jpg")
DEFAULT_MODEL = os.path.expanduser(
    "~/software/edgefirst-studio-ultralytics/workdir/test-matrix/"
    "yolov8n-seg-segment/yolov8n-seg_int8.tflite"
)
OUTPUT_DIR = os.path.join(HAL_ROOT, "testdata")

# COCO class names (80 classes)
COCO_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# 20 distinct instance colors (BGR for OpenCV, A=128 for semi-transparency)
INSTANCE_COLORS_BGR = [
    (56, 56, 255),  # red
    (151, 157, 255),  # pink
    (31, 112, 255),  # orange
    (29, 178, 255),  # yellow
    (49, 210, 207),  # lime
    (10, 249, 72),  # green
    (23, 204, 146),  # teal
    (134, 219, 61),  # cyan
    (52, 147, 26),  # forest
    (187, 212, 0),  # turquoise
    (168, 153, 44),  # blue-grey
    (255, 194, 0),  # sky
    (147, 69, 52),  # navy
    (255, 115, 100),  # lavender
    (236, 24, 0),  # royal
    (255, 56, 132),  # purple
    (133, 0, 82),  # wine
    (255, 56, 203),  # magenta
    (200, 149, 255),  # rose
    (199, 55, 255),  # hot pink
]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------
def extract_metadata(tflite_path):
    """Extract edgefirst.json metadata from TFLite model."""
    with open(tflite_path, "rb") as f:
        data = f.read()
    eocd = data.rfind(b"PK\x05\x06")
    if eocd < 0:
        raise ValueError(f"No zip metadata in {tflite_path}")
    pk = data.rfind(b"PK\x03\x04", 0, eocd)
    zf = zipfile.ZipFile(io.BytesIO(data[pk:]))
    for name in zf.namelist():
        if name.endswith(".json"):
            return json.loads(zf.read(name))
    raise ValueError(f"No JSON metadata in {tflite_path}")


def load_tflite(model_path):
    """Load TFLite interpreter."""
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite
    interp = tflite.Interpreter(model_path)
    interp.allocate_tensors()
    return interp


def run_tflite(interp, input_data):
    """Run TFLite inference, return list of output arrays."""
    inp = interp.get_input_details()[0]
    interp.set_tensor(inp["index"], input_data)
    interp.invoke()
    return [interp.get_tensor(d["index"]) for d in interp.get_output_details()]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def report(name, times):
    """Print timing statistics."""
    mean = statistics.mean(times)
    median = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    mn, mx = min(times), max(times)
    print(
        f"  {name:<35s}  mean={mean:7.2f}ms  median={median:7.2f}ms  "
        f"p95={p95:7.2f}ms  min={mn:7.2f}ms  max={mx:7.2f}ms"
    )


def render_masks_on_image(
    image_bgr,
    boxes,
    scores,
    classes,
    masks,
    mask_format="full",
    target_size=640,
    alpha=0.5,
):
    """Render colored per-instance masks onto a BGR image.

    Args:
        image_bgr: (H, W, 3) uint8 BGR image (letterboxed)
        boxes: (N, 4) normalized xyxy [0,1]
        scores: (N,)
        classes: (N,)
        masks: list of mask arrays (see mask_format)
        mask_format: "full" — (160, 160) float32 sigmoid at proto resolution
                     "cropped" — (H, W, 1) uint8 bbox-cropped masks from HAL
        target_size: render size (640)
        alpha: mask opacity

    Returns:
        (target_size, target_size, 3) uint8 BGR image
    """
    canvas = image_bgr.copy()

    for i in range(len(boxes)):
        x1 = int(boxes[i, 0] * target_size)
        y1 = int(boxes[i, 1] * target_size)
        x2 = int(boxes[i, 2] * target_size)
        y2 = int(boxes[i, 3] * target_size)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(target_size, x2), min(target_size, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        # Get binary mask in bbox region
        if mask_format == "full" and i < len(masks):
            mask_full = cv2.resize(
                masks[i], (target_size, target_size), interpolation=cv2.INTER_LINEAR
            )
            mask_bin = (mask_full[y1:y2, x1:x2] > 0.5).astype(np.float32)
        elif mask_format == "cropped" and i < len(masks):
            m = masks[i]
            if m.ndim == 3:
                m = m[:, :, 0]
            # Resize bbox-cropped mask to bbox pixel size
            mask_resized = cv2.resize(
                m.astype(np.float32), (bw, bh), interpolation=cv2.INTER_LINEAR
            )
            mask_bin = (mask_resized > 127.5).astype(np.float32)
        else:
            continue

        if mask_bin.sum() == 0:
            continue

        # Per-instance color (not per-class)
        color = INSTANCE_COLORS_BGR[i % len(INSTANCE_COLORS_BGR)]
        overlay = canvas[y1:y2, x1:x2].astype(np.float32)
        mask_3d = mask_bin[:, :, None]
        color_f = np.array(color, dtype=np.float32)
        blended = overlay * (1 - mask_3d * alpha) + color_f * mask_3d * alpha
        canvas[y1:y2, x1:x2] = blended.clip(0, 255).astype(np.uint8)

        # Draw box and label
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        name = (
            COCO_NAMES[int(classes[i])]
            if int(classes[i]) < len(COCO_NAMES)
            else str(int(classes[i]))
        )
        label = f"{name} {scores[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
        cv2.putText(
            canvas,
            label,
            (x1, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return canvas


# ---------------------------------------------------------------------------
# OpenCV pipeline — decode
# ---------------------------------------------------------------------------
def opencv_letterbox(image, new_shape=640, color=114):
    """Letterbox resize maintaining aspect ratio."""
    h, w = image.shape[:2]
    scale = min(new_shape / w, new_shape / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    pad_x, pad_y = (new_shape - nw) // 2, (new_shape - nh) // 2
    canvas[pad_y : pad_y + nh, pad_x : pad_x + nw] = resized
    return canvas, scale, pad_x, pad_y


def _nms_numpy(boxes, scores, iou_threshold):
    """Simple class-agnostic NMS on normalized xyxy boxes."""
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / (area_i + area_r - inter + 1e-7)
        order = rest[iou <= iou_threshold]
    return np.array(keep)


def opencv_decode_seg(outputs, metadata, score_thresh=0.25, iou_thresh=0.45):
    """Decode YOLOv8-seg INT8 TFLite outputs using numpy.

    Returns (boxes_norm, scores, classes, masks_160x160_float).
    Boxes are normalized [0,1] in xyxy format.
    Masks are continuous sigmoid values at proto resolution (160x160).
    """
    out_map = {}
    for i, spec in enumerate(metadata["outputs"]):
        out_map[spec["type"]] = (outputs[i], spec.get("quantization"))

    def deq(arr, qparams):
        if qparams and arr.dtype != np.float32:
            s, zp = qparams
            return (arr.astype(np.float32) - zp) * s
        return arr.astype(np.float32)

    scores_raw = deq(*out_map["scores"])[0].T  # (8400, 80)
    boxes_raw = deq(*out_map["boxes"])[0].T  # (8400, 4) xcycwh
    mc_raw = deq(*out_map["mask_coefficients"])[0].T  # (8400, 32)
    protos_raw = deq(*out_map["protos"])  # (1, 160, 160, 32)

    # xcycwh → xyxy (normalized [0,1])
    cx, cy, w, h = boxes_raw[:, 0], boxes_raw[:, 1], boxes_raw[:, 2], boxes_raw[:, 3]
    boxes_raw = np.column_stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    # Score filter
    max_scores = scores_raw.max(axis=1)
    filt = max_scores >= score_thresh
    scores_f = max_scores[filt]
    classes_f = scores_raw[filt].argmax(axis=1)
    boxes_f = boxes_raw[filt]
    mc_f = mc_raw[filt]

    if len(scores_f) == 0:
        return np.zeros((0, 4), np.float32), np.array([]), np.array([]), []

    indices = _nms_numpy(boxes_f, scores_f, iou_thresh)
    if len(indices) == 0:
        return np.zeros((0, 4), np.float32), np.array([]), np.array([]), []

    boxes_out = boxes_f[indices]
    scores_out = scores_f[indices]
    classes_out = classes_f[indices].astype(np.uintp)
    mc_out = mc_f[indices]

    # Masks: (N, 32) @ (32, 160*160) → sigmoid → (N, 160, 160)
    protos = protos_raw[0]  # (160, 160, 32)
    protos_flat = protos.reshape(-1, 32).T  # (32, 25600)
    masks_sig = sigmoid(mc_out @ protos_flat).reshape(-1, 160, 160)

    return boxes_out, scores_out, classes_out, masks_sig


def run_opencv_pipeline(image_path, metadata, interp, save_path=None):
    """Full OpenCV pipeline: load → letterbox → infer → decode → render.

    Returns (bgr_640x640, boxes, scores, classes, elapsed_ms_dict).
    """
    timings = {}

    t0 = time.perf_counter()
    img = cv2.imread(image_path)
    timings["load"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    lb, scale, px, py = opencv_letterbox(img, 640)
    input_data = np.expand_dims(lb, 0).astype(np.uint8)[..., ::-1].copy()
    timings["letterbox"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    outputs = run_tflite(interp, input_data)
    timings["inference"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    boxes, scores, classes, masks = opencv_decode_seg(outputs, metadata)
    timings["decode"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    bgr = render_masks_on_image(lb, boxes, scores, classes, masks, mask_format="full")
    timings["render"] = (time.perf_counter() - t0) * 1000

    timings["total"] = sum(timings.values())

    if save_path:
        cv2.imwrite(save_path, bgr)
        print(f"  Saved: {save_path}")

    return bgr, boxes, scores, classes, timings


# ---------------------------------------------------------------------------
# HAL pipeline
# ---------------------------------------------------------------------------
def run_hal_pipeline(image_path, metadata, interp, processor, save_path=None):
    """Full HAL pipeline: load → letterbox → infer → decode → render.

    Uses HAL for load + letterbox + decode, Python for mask rendering
    (per-instance colors). For benchmarking the fused OpenGL path, see
    run_hal_pipeline_bench().

    Returns (bgr_640x640, boxes, scores, classes, elapsed_ms_dict).
    """
    from edgefirst_hal import TensorImage, FourCC, Rect, Decoder

    timings = {}

    # 1. Load
    t0 = time.perf_counter()
    src = TensorImage.load(image_path, fourcc=FourCC.RGB)
    timings["load"] = (time.perf_counter() - t0) * 1000

    # 2. Letterbox
    t0 = time.perf_counter()
    scale = min(640 / src.width, 640 / src.height)
    nw, nh = int(src.width * scale), int(src.height * scale)
    px, py = (640 - nw) // 2, (640 - nh) // 2

    dst_rgb = TensorImage(640, 640, FourCC.RGB)
    processor.convert(
        src,
        dst_rgb,
        dst_crop=Rect(px, py, nw, nh),
        dst_color=[114, 114, 114, 255],
    )

    input_data = np.zeros((1, 640, 640, 3), dtype=np.uint8)
    dst_rgb.normalize_to_numpy(input_data[0])
    timings["letterbox"] = (time.perf_counter() - t0) * 1000

    # 3. Inference
    t0 = time.perf_counter()
    outputs = run_tflite(interp, input_data)
    timings["inference"] = (time.perf_counter() - t0) * 1000

    # 4. Decode (HAL Rust decoder — NMS + mask computation)
    t0 = time.perf_counter()
    decoder = Decoder(metadata, score_threshold=0.25, iou_threshold=0.45)
    boxes, scores, classes, masks = decoder.decode(outputs)
    timings["decode"] = (time.perf_counter() - t0) * 1000

    # 5. Render (Python — per-instance colors)
    t0 = time.perf_counter()
    # Get letterboxed image as BGR for rendering
    rgb_np = np.zeros((640, 640, 3), dtype=np.uint8)
    dst_rgb.normalize_to_numpy(rgb_np)
    bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

    bgr = render_masks_on_image(
        bgr_np, boxes, scores, classes, masks, mask_format="cropped"
    )
    timings["render"] = (time.perf_counter() - t0) * 1000

    timings["total"] = sum(timings.values())

    if save_path:
        cv2.imwrite(save_path, bgr)
        print(f"  Saved: {save_path}")

    return bgr, boxes, scores, classes, timings


def run_hal_pipeline_bench(image_path, metadata, interp, processor):
    """HAL pipeline using fused OpenGL decode_and_render for benchmarking.

    Returns elapsed_ms_dict (no visual output — uses HAL's built-in colors).
    """
    from edgefirst_hal import TensorImage, FourCC, Rect, Decoder

    timings = {}

    t0 = time.perf_counter()
    src = TensorImage.load(image_path, fourcc=FourCC.RGB)
    timings["load"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    scale = min(640 / src.width, 640 / src.height)
    nw, nh = int(src.width * scale), int(src.height * scale)
    px, py = (640 - nw) // 2, (640 - nh) // 2
    dst_rgb = TensorImage(640, 640, FourCC.RGB)
    processor.convert(
        src, dst_rgb, dst_crop=Rect(px, py, nw, nh), dst_color=[114, 114, 114, 255]
    )
    input_data = np.zeros((1, 640, 640, 3), dtype=np.uint8)
    dst_rgb.normalize_to_numpy(input_data[0])
    timings["letterbox"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    outputs = run_tflite(interp, input_data)
    timings["inference"] = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    render_dst = TensorImage(640, 640, FourCC.RGBA)
    processor.convert(dst_rgb, render_dst)
    decoder = Decoder(metadata, score_threshold=0.25, iou_threshold=0.45)
    decoder.decode_and_render(outputs, processor, render_dst)
    timings["decode+render"] = (time.perf_counter() - t0) * 1000

    timings["total"] = sum(timings.values())
    return timings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Input image path")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="YOLOv8n-seg INT8 TFLite model"
    )
    parser.add_argument(
        "--bench", action="store_true", help="Run benchmark mode (multiple iterations)"
    )
    parser.add_argument(
        "-n",
        "--iterations",
        type=int,
        default=50,
        help="Benchmark iterations (default: 50)",
    )
    parser.add_argument(
        "--warmup", type=int, default=5, help="Warmup iterations (default: 5)"
    )
    parser.add_argument("--hal-only", action="store_true", help="Only run HAL pipeline")
    parser.add_argument(
        "--opencv-only", action="store_true", help="Only run OpenCV pipeline"
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR, help="Directory for output images"
    )
    args = parser.parse_args()

    run_hal = not args.opencv_only
    run_opencv = not args.hal_only

    print(f"Image:  {args.image}")
    print(f"Model:  {args.model}")
    print()

    metadata = extract_metadata(args.model)
    interp = load_tflite(args.model)

    hal_processor = None
    if run_hal:
        from edgefirst_hal import ImageProcessor, probe_egl_displays

        displays = probe_egl_displays()
        gpu = displays[0].kind if displays else None
        print(f"EGL displays: {[str(d.kind) for d in displays]}")
        hal_processor = ImageProcessor(egl_display=gpu)

    if not args.bench:
        # --- Single-run mode: save outputs for visual inspection ---
        def print_detections(boxes, scores, classes, timings):
            print(f"  Detections: {len(boxes)}")
            for i in range(len(boxes)):
                name = (
                    COCO_NAMES[int(classes[i])]
                    if int(classes[i]) < len(COCO_NAMES)
                    else str(int(classes[i]))
                )
                print(
                    f"    {name}: {scores[i]:.3f}  "
                    f"box=[{boxes[i, 0]:.3f}, {boxes[i, 1]:.3f}, "
                    f"{boxes[i, 2]:.3f}, {boxes[i, 3]:.3f}]"
                )
            print("  Timings:")
            for stage, ms in timings.items():
                print(f"    {stage:<20s} {ms:7.2f}ms")

        if run_opencv:
            print("\n--- OpenCV Pipeline ---")
            out = os.path.join(args.output_dir, "output_opencv_seg.png")
            _, boxes, scores, classes, t = run_opencv_pipeline(
                args.image, metadata, interp, save_path=out
            )
            print_detections(boxes, scores, classes, t)

        if run_hal:
            print("\n--- HAL Pipeline ---")
            out = os.path.join(args.output_dir, "output_hal_seg.png")
            _, boxes, scores, classes, t = run_hal_pipeline(
                args.image, metadata, interp, hal_processor, save_path=out
            )
            print_detections(boxes, scores, classes, t)

    else:
        # --- Benchmark mode ---
        print(f"\nBenchmark: {args.iterations} iterations (warmup: {args.warmup})")
        print(f"{'=' * 100}")

        if run_opencv:
            print("\n--- OpenCV Pipeline ---")
            for _ in range(args.warmup):
                run_opencv_pipeline(args.image, metadata, interp)
            all_t = {}
            for _ in range(args.iterations):
                _, _, _, _, t = run_opencv_pipeline(args.image, metadata, interp)
                for k, v in t.items():
                    all_t.setdefault(k, []).append(v)
            for stage, times in all_t.items():
                report(f"opencv/{stage}", times)

        if run_hal:
            # Benchmark HAL decode() + Python render (apples-to-apples)
            print("\n--- HAL Pipeline (decode + Python render) ---")
            for _ in range(args.warmup):
                run_hal_pipeline(args.image, metadata, interp, hal_processor)
            all_t = {}
            for _ in range(args.iterations):
                _, _, _, _, t = run_hal_pipeline(
                    args.image, metadata, interp, hal_processor
                )
                for k, v in t.items():
                    all_t.setdefault(k, []).append(v)
            for stage, times in all_t.items():
                report(f"hal/{stage}", times)

            # Benchmark HAL fused decode_and_render (OpenGL)
            print("\n--- HAL Pipeline (fused OpenGL decode+render) ---")
            for _ in range(args.warmup):
                run_hal_pipeline_bench(args.image, metadata, interp, hal_processor)
            all_t = {}
            for _ in range(args.iterations):
                t = run_hal_pipeline_bench(args.image, metadata, interp, hal_processor)
                for k, v in t.items():
                    all_t.setdefault(k, []).append(v)
            for stage, times in all_t.items():
                report(f"hal-fused/{stage}", times)

        print(f"\n{'=' * 100}")


if __name__ == "__main__":
    main()
