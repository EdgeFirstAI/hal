#!/usr/bin/env python3
"""Isolated profiling harness for decode+render paths.

Runs a single path (fused or 2-step) in a tight loop with no timing overhead,
designed for use with `perf record`. Setup (model load, TFLite inference,
EGL init) happens before the hot loop to keep profiles clean.

Usage:
    # Profile fused path:
    perf record -F 997 --call-graph dwarf -o perf-fused.data -- \
        python tests/profile_decode_render.py fused

    # Profile 2-step path:
    perf record -F 997 --call-graph dwarf -o perf-2step.data -- \
        python tests/profile_decode_render.py 2step

    # Analyze:
    perf report -i perf-fused.data
    perf report -i perf-2step.data
"""

import argparse
import json
import os
import sys
import time
import zipfile
import io

import numpy as np

from edgefirst_hal import (
    Decoder,
    ImageProcessor,
    Tensor,
    PixelFormat,
    probe_egl_displays,
)

DEFAULT_MODEL = os.path.expanduser(
    "~/software/edgefirst-studio-ultralytics/workdir/test-matrix/"
    "yolov8n-seg-segment/yolov8n-seg_int8.tflite"
)


def extract_metadata(tflite_path):
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
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        import tensorflow.lite as tflite

    interp = tflite.Interpreter(tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    interp.set_tensor(
        inp["index"],
        np.random.randint(0, 255, size=inp["shape"], dtype=inp["dtype"]),
    )
    interp.invoke()
    return [interp.get_tensor(d["index"]) for d in interp.get_output_details()]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "path", choices=["fused", "2step"], help="Which path to profile"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=5000,
        help="Iterations in hot loop (default: 5000)",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.01,
        help="Score threshold (lower = more detections)",
    )
    args = parser.parse_args()

    # --- Setup (outside profiled region) ---
    print(f"Setting up: model={args.model}, path={args.path}", file=sys.stderr)

    displays = probe_egl_displays()
    gpu_display = displays[0].kind if displays else None
    print(f"EGL: {[str(d.kind) for d in displays]}", file=sys.stderr)

    metadata = extract_metadata(args.model)
    outputs = get_model_outputs(args.model)
    print(f"Outputs: {[o.shape for o in outputs]}", file=sys.stderr)

    processor = ImageProcessor(egl_display=gpu_display)
    dst = Tensor.image(640, 640, PixelFormat.Rgba)
    decoder = Decoder(metadata, score_threshold=args.threshold, iou_threshold=0.45)

    # --- Warmup ---
    print(f"Warming up ({args.warmup} iters)...", file=sys.stderr)
    for _ in range(args.warmup):
        if args.path == "fused":
            processor.draw_masks_fused(decoder, outputs, dst)
        else:
            boxes, scores, classes, masks = decoder.decode(outputs)
            processor.draw_masks(
                dst, bbox=boxes, scores=scores, classes=classes, seg=masks
            )

    # --- Hot loop (this is what perf will sample) ---
    print(f"Profiling {args.path} for {args.iterations} iterations...", file=sys.stderr)
    t0 = time.perf_counter()

    if args.path == "fused":
        for _ in range(args.iterations):
            processor.draw_masks_fused(decoder, outputs, dst)
    else:
        for _ in range(args.iterations):
            boxes, scores, classes, masks = decoder.decode(outputs)
            processor.draw_masks(
                dst, bbox=boxes, scores=scores, classes=classes, seg=masks
            )

    elapsed = time.perf_counter() - t0
    per_iter = elapsed / args.iterations * 1000
    print(
        f"Done: {elapsed:.2f}s total, {per_iter:.3f}ms/iter ({args.iterations} iters)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
