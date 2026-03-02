#!/usr/bin/env python3
"""Benchmark: decode_and_render() fused path vs decode() + render_to_image() 2-step path.

Compares per-frame timing for YOLOv8n-seg INT8 TFLite outputs using the HAL
decoder and image processor. Uses real TFLite model inference outputs when the
model is available, otherwise falls back to synthetic int8 data matching the
model output shapes.

Usage:
    python tests/bench_decode_render.py [--iterations N] [--warmup N] [--json results.json]
"""

import argparse
import io
import json
import os
import platform
import socket
import statistics
import time
import zipfile

import numpy as np

from edgefirst_hal import (
    Decoder,
    ImageProcessor,
    TensorImage,
    FourCC,
    probe_egl_displays,
)

# Default model path (yolov8n-seg INT8 TFLite from the test matrix)
DEFAULT_MODEL = os.path.expanduser(
    "~/software/edgefirst-studio-ultralytics/workdir/test-matrix/"
    "yolov8n-seg-segment/yolov8n-seg_int8.tflite"
)

# Model output shapes and quantization params for synthetic fallback
SYNTHETIC_OUTPUTS = [
    {"shape": (1, 80, 8400), "dtype": np.int8},  # scores
    {"shape": (1, 160, 160, 32), "dtype": np.int8},  # protos
    {"shape": (1, 4, 8400), "dtype": np.int8},  # boxes
    {"shape": (1, 32, 8400), "dtype": np.int8},  # mask_coefficients
]


def extract_metadata(tflite_path):
    """Extract edgefirst.json metadata from TFLite model."""
    with open(tflite_path, "rb") as f:
        data = f.read()

    eocd_offset = data.rfind(b"PK\x05\x06")
    if eocd_offset < 0:
        raise ValueError(f"No zip metadata found in {tflite_path}")

    pk_offset = data.rfind(b"PK\x03\x04", 0, eocd_offset)
    zf = zipfile.ZipFile(io.BytesIO(data[pk_offset:]))

    for name in zf.namelist():
        if name.endswith(".json"):
            return json.loads(zf.read(name))

    raise ValueError(f"No JSON metadata found in {tflite_path}")


def get_model_outputs(tflite_path):
    """Run TFLite inference to get realistic model outputs."""
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        try:
            import tensorflow.lite as tflite
        except ImportError:
            return None

    interp = tflite.Interpreter(tflite_path)
    interp.allocate_tensors()

    # Feed random uint8 input
    input_details = interp.get_input_details()
    input_shape = input_details[0]["shape"]
    input_data = np.random.randint(
        0, 255, size=input_shape, dtype=input_details[0]["dtype"]
    )
    interp.set_tensor(input_details[0]["index"], input_data)
    interp.invoke()

    outputs = []
    for d in interp.get_output_details():
        outputs.append(interp.get_tensor(d["index"]))

    return outputs


def generate_synthetic_outputs():
    """Generate synthetic int8 outputs matching yolov8n-seg shape.

    Box values are constrained so that dequantized coordinates stay within
    [0, 1] (the decoder rejects un-normalized boxes).  With the fallback
    quantization params (scale=0.004, zp=-125), the dequantized value is
    (int8 + 125) * 0.004.  Boxes are in cxcywh format, so xmax = cx + w/2
    can exceed 1.0 if cx and w are both large.  Constraining box int8 values
    to [-125, 0] keeps all dequantized coordinates in [0, 0.5], ensuring
    xyxy stays within [0, 1].
    """
    results = []
    for spec in SYNTHETIC_OUTPUTS:
        if spec is SYNTHETIC_OUTPUTS[2]:  # boxes tensor
            arr = np.random.randint(
                -125, 0, size=spec["shape"], dtype=spec["dtype"]
            )
        else:
            arr = np.random.randint(
                -128, 127, size=spec["shape"], dtype=spec["dtype"]
            )
        results.append(arr)
    return results


def get_target_info(gpu_display):
    """Collect target identification for cross-target comparison."""
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "egl_display": "none",
        "egl_description": "",
    }
    try:
        displays = probe_egl_displays()
        if displays:
            info["egl_display"] = str(displays[0].kind)
            info["egl_description"] = displays[0].description
    except Exception:
        pass  # defaults already set
    return info


def compute_stats(times):
    """Compute timing statistics from a list of per-iteration times (ms)."""
    sorted_times = sorted(times)
    return {
        "mean": round(statistics.mean(times), 4),
        "median": round(statistics.median(times), 4),
        "p95": round(sorted_times[int(len(sorted_times) * 0.95)], 4),
        "min": round(min(times), 4),
        "max": round(max(times), 4),
        "times": [round(t, 4) for t in times],
    }


def bench_old_path(decoder, processor, dst, outputs, n_iter):
    """Benchmark: decode() + render_to_image() (2-step path)."""
    times = []
    n_dets = 0
    for _ in range(n_iter):
        t0 = time.perf_counter()
        boxes, scores, classes, masks = decoder.decode(outputs)
        processor.render_to_image(
            dst, bbox=boxes, scores=scores, classes=classes, seg=masks
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)  # ms
        n_dets = len(boxes)
    return times, n_dets


def bench_fused_path(decoder, processor, dst, outputs, n_iter):
    """Benchmark: decode_and_render() (fused path)."""
    times = []
    n_dets = 0
    for _ in range(n_iter):
        t0 = time.perf_counter()
        boxes, scores, classes = decoder.decode_and_render(outputs, processor, dst)
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)  # ms
        n_dets = len(boxes)
    return times, n_dets


def bench_mask_path(
    decoder, processor, outputs, n_iter, output_width=640, output_height=640
):
    """Benchmark: decode_and_render_masks() (GPU mask readback path)."""
    times = []
    n_dets = 0
    for _ in range(n_iter):
        t0 = time.perf_counter()
        boxes, scores, classes, masks = decoder.decode_and_render_masks(
            outputs, processor, output_width=output_width, output_height=output_height
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)  # ms
        n_dets = len(boxes)
    return times, n_dets


def bench_mask_atlas_path(
    decoder, processor, outputs, n_iter, output_width=640, output_height=640
):
    """Benchmark: decode_and_render_mask_atlas() (GPU atlas readback path)."""
    times = []
    n_dets = 0
    for _ in range(n_iter):
        t0 = time.perf_counter()
        boxes, scores, classes, atlas, metadata = decoder.decode_and_render_mask_atlas(
            outputs, processor, output_width=output_width, output_height=output_height
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed * 1000)  # ms
        n_dets = len(boxes)
    return times, n_dets


def verify_atlas_correctness(
    decoder, processor, outputs, output_width=640, output_height=640
):
    """Verify atlas path produces identical masks to per-detection path.

    Runs both decode_and_render_masks() and decode_and_render_mask_atlas(),
    then compares per-detection bbox crops from the atlas against the
    reference per-detection masks. They should be identical or within +-1
    due to floating-point rounding.
    """
    # Get per-detection reference masks
    boxes_ref, scores_ref, classes_ref, masks_ref = decoder.decode_and_render_masks(
        outputs, processor, output_width=output_width, output_height=output_height
    )

    # Get atlas masks
    boxes_atl, scores_atl, classes_atl, atlas, metadata = (
        decoder.decode_and_render_mask_atlas(
            outputs, processor, output_width=output_width, output_height=output_height
        )
    )

    n_ref = len(boxes_ref)
    n_atl = len(boxes_atl)

    if n_ref != n_atl:
        print(
            f"  FAIL: detection count mismatch: "
            f"per-detection={n_ref}, atlas={n_atl}"
        )
        return False

    if n_ref == 0:
        print("  Atlas correctness: 0 detections (nothing to compare)")
        return True

    max_diff = 0
    mismatched = 0

    for i in range(n_ref):
        x, y, w, h = metadata[i]

        # Extract crop from atlas: atlas shape is [N, H, W]
        atlas_crop = atlas[i, y : y + h, x : x + w]

        # Reference mask shape is (H, W, 1) — squeeze the last dim
        ref_crop = masks_ref[i][:, :, 0]

        if atlas_crop.shape != ref_crop.shape:
            print(
                f"  FAIL: detection {i} shape mismatch: "
                f"atlas_crop={atlas_crop.shape}, ref={ref_crop.shape}"
            )
            mismatched += 1
            continue

        diff = np.abs(atlas_crop.astype(np.int16) - ref_crop.astype(np.int16))
        det_max_diff = int(diff.max())
        max_diff = max(max_diff, det_max_diff)

        if det_max_diff > 1:
            mismatched += 1
            print(
                f"  WARNING: detection {i} max pixel diff = {det_max_diff} "
                f"(crop {w}x{h} at ({x},{y}))"
            )

    ok = mismatched == 0
    status = "PASS" if ok else "FAIL"
    print(
        f"  Atlas correctness: {n_ref} detections, "
        f"max pixel diff = {max_diff} [{status}]"
    )
    return ok


def has_set_int8_interpolation(processor):
    """Check if the processor supports set_int8_interpolation."""
    return hasattr(processor, "set_int8_interpolation")


def report(name, times):
    """Print timing statistics."""
    mean = statistics.mean(times)
    median = statistics.median(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    mn = min(times)
    mx = max(times)
    print(
        f"  {name:<30s}  mean={mean:7.2f}ms  median={median:7.2f}ms  "
        f"p95={p95:7.2f}ms  min={mn:7.2f}ms  max={mx:7.2f}ms"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Path to yolov8n-seg INT8 TFLite model"
    )
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=200,
        help="Number of timed iterations (default: 200)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force synthetic outputs (skip TFLite inference)",
    )
    parser.add_argument(
        "--interpolation",
        choices=["all", "bilinear", "nearest", "twopass"],
        default=None,
        help="Test specific int8 interpolation mode(s) for fused path",
    )
    parser.add_argument(
        "--json",
        metavar="PATH",
        default=None,
        help="Write benchmark results to a JSON file for cross-target comparison",
    )
    parser.add_argument(
        "--verify-atlas",
        action="store_true",
        help="Run atlas correctness verification before benchmarking",
    )
    args = parser.parse_args()

    # --- GPU / EGL display diagnostics ---
    print("GPU rendering status:")
    try:
        displays = probe_egl_displays()
        if displays:
            for d in displays:
                print(f"  {str(d.kind):40s} {d.description}")
            gpu_display = displays[0].kind
            print(f"  -> Using: {gpu_display} (highest priority)")
        else:
            print("  No EGL displays found — rendering will use CPU fallback")
            gpu_display = None
    except RuntimeError as e:
        print(f"  EGL probe failed: {e} — rendering will use CPU fallback")
        gpu_display = None

    # --- Collect target info and prepare results dict ---
    target_info = get_target_info(gpu_display)
    results = {}

    # --- Setup decoder config ---
    if os.path.exists(args.model) and not args.synthetic:
        print(f"Using model: {args.model}")
        metadata = extract_metadata(args.model)
        outputs = get_model_outputs(args.model)
        if outputs is None:
            print("  TFLite runtime not available, using synthetic outputs")
            outputs = generate_synthetic_outputs()
        else:
            print(f"  Got {len(outputs)} real model outputs")
    else:
        if not args.synthetic:
            print(f"Model not found: {args.model}, using synthetic outputs")
        else:
            print("Using synthetic outputs (--synthetic)")
        metadata = extract_metadata(args.model) if os.path.exists(args.model) else None
        outputs = generate_synthetic_outputs()

    if metadata is None:
        # Hardcoded fallback metadata for yolov8n-seg
        metadata = {
            "outputs": [
                {
                    "type": "scores",
                    "decoder": "ultralytics",
                    "shape": [1, 80, 8400],
                    "decode": True,
                    "score_format": "per_class",
                    "quantization": [0.00390625, -128],
                },
                {
                    "type": "protos",
                    "decoder": "ultralytics",
                    "shape": [1, 160, 160, 32],
                    "decode": True,
                    "quantization": [0.030935298651456833, -119],
                },
                {
                    "type": "boxes",
                    "decoder": "ultralytics",
                    "shape": [1, 4, 8400],
                    "decode": True,
                    "normalized": True,
                    "quantization": [0.0040979208424687386, -125],
                },
                {
                    "type": "mask_coefficients",
                    "decoder": "ultralytics",
                    "shape": [1, 32, 8400],
                    "decode": True,
                    "quantization": [0.025969834998250008, 30],
                },
            ]
        }

    # --- Create processor and image ---
    processor = ImageProcessor(egl_display=gpu_display)
    dst = TensorImage(640, 640, FourCC.RGBA)

    print(f"\nOutput shapes: {[o.shape for o in outputs]}")
    print(f"Output dtypes: {[o.dtype for o in outputs]}")
    print(f"Iterations: {args.iterations} (warmup: {args.warmup})")

    # Determine interpolation modes to test
    interp_modes = []
    if args.interpolation and has_set_int8_interpolation(processor):
        if args.interpolation == "all":
            interp_modes = ["bilinear", "nearest", "twopass"]
        else:
            interp_modes = [args.interpolation]
    elif has_set_int8_interpolation(processor):
        # Default: test all modes when the API is available
        interp_modes = ["bilinear", "nearest", "twopass"]

    # Test at multiple score thresholds to vary detection count
    thresholds = [0.25, 0.10, 0.01]

    for thresh in thresholds:
        decoder = Decoder(metadata, score_threshold=thresh, iou_threshold=0.45)
        thresh_key = str(thresh)
        thresh_results = {}

        # --- Warmup ---
        bench_old_path(decoder, processor, dst, outputs, args.warmup)
        bench_fused_path(decoder, processor, dst, outputs, args.warmup)
        bench_mask_path(decoder, processor, outputs, args.warmup)
        try:
            bench_mask_atlas_path(decoder, processor, outputs, args.warmup)
        except RuntimeError:
            pass  # atlas may fail if n_detections * height > GL_MAX_TEXTURE_SIZE

        # --- Benchmark 2-step path ---
        old_times, old_dets = bench_old_path(
            decoder, processor, dst, outputs, args.iterations
        )

        thresh_results["n_detections"] = old_dets
        thresh_results["decode_render_to_image"] = compute_stats(old_times)

        print(f"\n  score_threshold={thresh} ({old_dets} detections):")
        print(f"  {'─' * 95}")
        report("decode() + render_to_image()", old_times)

        if interp_modes:
            # Benchmark fused path for each interpolation mode
            for mode in interp_modes:
                processor.set_int8_interpolation(mode)
                # Warmup for this mode
                bench_fused_path(decoder, processor, dst, outputs, args.warmup)
                fused_times, fused_dets = bench_fused_path(
                    decoder, processor, dst, outputs, args.iterations
                )
                report(f"fused [int8 {mode}]", fused_times)
                speedup = statistics.mean(old_times) / statistics.mean(fused_times)
                saved = statistics.mean(old_times) - statistics.mean(fused_times)
                print(f"    -> {speedup:.2f}x ({saved:+.2f}ms vs 2-step)")
                thresh_results[f"decode_and_render_{mode}"] = compute_stats(fused_times)
        else:
            # No interpolation API — benchmark single fused path
            fused_times, fused_dets = bench_fused_path(
                decoder, processor, dst, outputs, args.iterations
            )
            report("decode_and_render() [fused]", fused_times)
            speedup = statistics.mean(old_times) / statistics.mean(fused_times)
            saved = statistics.mean(old_times) - statistics.mean(fused_times)
            print(f"  Speedup: {speedup:.2f}x ({saved:+.2f}ms per frame)")
            thresh_results["decode_and_render"] = compute_stats(fused_times)

        # --- Benchmark mask readback path ---
        mask_times, mask_dets = bench_mask_path(
            decoder, processor, outputs, args.iterations
        )
        report("decode_and_render_masks()", mask_times)
        mask_speedup = statistics.mean(old_times) / statistics.mean(mask_times)
        mask_saved = statistics.mean(old_times) - statistics.mean(mask_times)
        print(f"    -> {mask_speedup:.2f}x ({mask_saved:+.2f}ms vs 2-step)")
        thresh_results["decode_and_render_masks"] = compute_stats(mask_times)

        # --- Atlas correctness verification (if requested) ---
        if args.verify_atlas:
            print(f"\n  Atlas correctness verification (threshold={thresh}):")
            try:
                verify_atlas_correctness(decoder, processor, outputs)
            except RuntimeError as e:
                print(f"  SKIP: atlas verification failed ({e})")

        # --- Benchmark mask atlas path ---
        try:
            atlas_times, atlas_dets = bench_mask_atlas_path(
                decoder, processor, outputs, args.iterations
            )
            report("decode_and_render_mask_atlas()", atlas_times)
            atlas_speedup = statistics.mean(old_times) / statistics.mean(atlas_times)
            atlas_saved = statistics.mean(old_times) - statistics.mean(atlas_times)
            print(f"    -> {atlas_speedup:.2f}x ({atlas_saved:+.2f}ms vs 2-step)")
            if statistics.mean(mask_times) > 0:
                atlas_vs_mask = (
                    statistics.mean(mask_times) / statistics.mean(atlas_times)
                )
                atlas_mask_saved = (
                    statistics.mean(mask_times) - statistics.mean(atlas_times)
                )
                print(
                    f"    -> {atlas_vs_mask:.2f}x "
                    f"({atlas_mask_saved:+.2f}ms vs per-detection masks)"
                )
            thresh_results["decode_and_render_mask_atlas"] = compute_stats(atlas_times)
        except RuntimeError as e:
            # Atlas may fail if detection count * output_height exceeds
            # GL_MAX_TEXTURE_SIZE (e.g. 100 detections * 640 = 64000 > 16384).
            print(
                f"  decode_and_render_mask_atlas()  SKIPPED "
                f"({old_dets} detections exceed GPU atlas capacity)"
            )

        print(f"  {'─' * 95}")

        results[thresh_key] = thresh_results

    # --- Write JSON output if requested ---
    if args.json:
        output = {
            "target": target_info,
            "results": results,
        }
        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to: {args.json}")


if __name__ == "__main__":
    main()
