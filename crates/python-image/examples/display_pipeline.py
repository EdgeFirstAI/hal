#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Display-pipeline building blocks from Python, checked on the device.

The pattern: the GPU renders the camera frame and the detection boxes into a
hardware-only (``access="none"``) DMA-BUF canvas whose fd goes to the
compositor, a CPU-written overlay lives in a ``access="write"`` image, and
the decoder turns model outputs into boxes. On Linux both images come from
the uncached CMA heap where available, so the compositor's per-frame mapping
costs no CPU cache maintenance; the HAL refuses any CPU read of them.

Each step prints ``PASS``/``FAIL``; the decode step also times the decoder.

Usage:
    python display_pipeline.py [capture_dir] [--iters N]

``capture_dir`` holds a model's raw outputs (``out<i>.bin``, ``meta.json``
with each output's shape, dtype and quantization) and its ``edgefirst.json``.
Without it the decode step is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
from edgefirst.decoder import Decoder
from edgefirst.image import ImageProcessor
from edgefirst.tensor import PixelFormat, Tensor, TensorMemory

W, H = 640, 480
FAILED = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")
    if not ok:
        FAILED.append(name)


def exporter(fd: int) -> str | None:
    """The heap a DMA-BUF fd came from, as /proc/self/fdinfo names it."""
    try:
        with open(f"/proc/self/fdinfo/{fd}") as f:
            for line in f:
                if line.startswith("exp_name:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def denied(fn, *args) -> bool:
    """Whether ``fn(*args)`` raises ``PermissionError``."""
    try:
        fn(*args)
    except PermissionError:
        return True
    return False


def fill(img: Tensor, rgba: tuple[int, int, int, int]) -> None:
    with img.map("write") as m:
        np.asarray(m.numpy())[...] = np.array(rgba, dtype=np.uint8)


def pixels(img: Tensor) -> np.ndarray:
    """The image as an (H, W, 4) array; the map's strides skip row padding."""
    with img.map("read") as m:
        return np.array(np.asarray(m.numpy()))


def box_arrays(xyxy):
    return (
        np.array([xyxy], dtype=np.float32),
        np.array([0.9], dtype=np.float32),
        np.array([0], dtype=np.uintp),
    )


def check_box_style(proc: ImageProcessor, label: str) -> None:
    """Box line width and colour alpha, read back from a CPU-readable image."""
    bg = proc.create_image(W, H, PixelFormat.Rgba, access="readwrite")
    fill(bg, (100, 100, 100, 255))
    dst = proc.create_image(W, H, PixelFormat.Rgba, access="readwrite")
    boxes, scores, classes = box_arrays([0.25, 0.25, 0.75, 0.75])

    proc.set_class_colors([(255, 0, 0, 255)])
    proc.set_box_line_width(5)
    proc.draw_decoded_masks(dst, boxes, scores, classes, background=bg)
    row = pixels(dst)[H // 2]
    red = np.flatnonzero((row[:, 0] == 255) & (row[:, 1] == 0))
    left = red[red < W // 2]
    check(
        f"{label}: box line width 5",
        left.size == 5,
        f"{left.size} px at x={left.tolist()}",
    )

    proc.set_class_colors([(255, 0, 0, 128)])
    proc.draw_decoded_masks(dst, boxes, scores, classes, background=bg)
    px = pixels(dst)[H // 2, left[0] if left.size else 0]
    want = round(100 + (255 - 100) * 128 / 255)
    ok = abs(int(px[0]) - want) <= 2 and abs(int(px[1]) - 50) <= 2
    check(f"{label}: box alpha 128 blends", ok, f"pixel {px[:3].tolist()}, red ~{want}")

    proc.set_class_colors([(255, 0, 0, 255)])
    proc.set_box_line_width(3)
    try:
        proc.set_box_line_width(0)
        check(f"{label}: width 0 rejected", False)
    except ValueError:
        check(f"{label}: width 0 rejected", True)


def check_display_images() -> None:
    """A GL-drawn hardware-only canvas with its fd exported, and a CPU-written
    overlay, both uncached; CPU reads of either are refused.

    Converts and draws use separate processors, as the demo does: on the
    i.MX 95's Mali GPU a draw leaves state behind that corrupts the next
    convert on the same processor.
    """
    proc, draw, readback_proc = ImageProcessor(), ImageProcessor(), ImageProcessor()
    camera = proc.create_image(W, H, PixelFormat.Rgba, access="readwrite")
    fill(camera, (30, 60, 90, 255))
    frame = proc.create_image(W, H, PixelFormat.Rgba, access="none")
    canvas = draw.create_image(W, H, PixelFormat.Rgba, access="none")
    uncached = os.path.exists("/dev/dma_heap/linux,cma-uncached")
    for name, img in (("frame", frame), ("canvas", canvas)):
        check(
            f"{name} declared none",
            img.declared_cpu_access == "none",
            img.declared_cpu_access,
        )
        check(
            f"{name} is a DMA-BUF", img.memory == TensorMemory.DMABUF, str(img.memory)
        )

    proc.convert(camera, frame)
    boxes, scores, classes = box_arrays([0.2, 0.2, 0.6, 0.6])
    draw.set_class_colors([(255, 96, 0, 160)])
    draw.set_box_line_width(2)
    draw.draw_decoded_masks(canvas, boxes, scores, classes, background=frame)
    check("GL draws into the none canvas", True)

    fd = canvas.fd
    try:
        name = exporter(fd)
        if uncached:
            check(
                "canvas fd is uncached CMA",
                name is not None and name.endswith("cma-uncached"),
                str(name),
            )
        check("canvas fd exported", fd >= 0, f"fd {fd}, stride {canvas.row_stride}")
    finally:
        os.close(fd)

    readback = readback_proc.create_image(W, H, PixelFormat.Rgba, access="readwrite")
    readback_proc.convert(canvas, readback)
    px = pixels(readback)
    check(
        "GL reads the canvas back",
        px[H // 2, W // 2, :3].tolist() == [30, 60, 90],
        f"centre {px[H // 2, W // 2, :3].tolist()}",
    )

    overlay = proc.create_image(W, H, PixelFormat.Rgba, access="write")
    check(
        "overlay declared write",
        overlay.declared_cpu_access == "write",
        overlay.declared_cpu_access,
    )
    fill(overlay, (10, 20, 30, 255))
    fd = overlay.fd
    try:
        if uncached:
            name = exporter(fd)
            check(
                "overlay fd is uncached CMA",
                name is not None and name.endswith("cma-uncached"),
                str(name),
            )
    finally:
        os.close(fd)
    readback_proc.convert(overlay, readback)
    check(
        "GL samples the CPU-written overlay",
        pixels(readback)[0, 0, :3].tolist() == [10, 20, 30],
    )

    check("read map of the none canvas refused", denied(canvas.map, "read"))
    check("write map of the none canvas refused", denied(canvas.map, "write"))
    check("read map of the write overlay refused", denied(overlay.map, "read"))


def check_cpu_refusals() -> None:
    """The CPU backend refuses work that would read a none or write image."""
    os.environ["EDGEFIRST_FORCE_BACKEND"] = "cpu"
    try:
        cpu = ImageProcessor()
    finally:
        del os.environ["EDGEFIRST_FORCE_BACKEND"]
    for access in ("write", "none"):
        dst = Tensor.image(W, H, PixelFormat.Rgba, TensorMemory.DMABUF, access=access)
        boxes, scores, classes = box_arrays([0.2, 0.2, 0.6, 0.6])
        check(
            f"CPU draw onto a {access} image refused",
            denied(cpu.draw_decoded_masks, dst, boxes, scores, classes),
        )
        dst2 = Tensor.image(
            W, H, PixelFormat.Rgba, TensorMemory.MEM, access="readwrite"
        )
        check(
            f"CPU convert from a {access} image refused",
            denied(cpu.convert, dst, dst2),
        )


def load_outputs(capture: str) -> list[Tensor]:
    with open(os.path.join(capture, "meta.json")) as f:
        meta = json.load(f)
    outputs = []
    for m in meta:
        data = np.fromfile(
            os.path.join(capture, f"out{int(m['index'])}.bin"), dtype=m["dtype"]
        )
        t = Tensor(m["shape"], dtype=m["dtype"])
        with t.map("write") as view:
            np.asarray(view.numpy())[...] = data.reshape(m["shape"])
        scale, zero_point = m["quantization"]
        t.set_quantization_per_tensor(scale, zero_point)
        outputs.append(t)
    return outputs


def time_decode(capture: str, iters: int) -> None:
    with open(os.path.join(capture, "edgefirst.json")) as f:
        decoder = Decoder(json.load(f), score_threshold=0.25, iou_threshold=0.7)
    outputs = load_outputs(capture)
    boxes = decoder.decode(outputs)[0]
    check("decoder finds boxes", len(boxes) > 0, f"{len(boxes)} boxes")
    for _ in range(10):
        decoder.decode(outputs)
    times = []
    cpu0 = time.process_time()
    for _ in range(iters):
        t0 = time.perf_counter()
        decoder.decode(outputs)
        times.append((time.perf_counter() - t0) * 1e6)
    cpu = (time.process_time() - cpu0) / iters * 1e6
    print(
        f"TIME  python decode: median {statistics.median(times):.1f} us, "
        f"mean {statistics.fmean(times):.1f} us, cpu {cpu:.1f} us/iter (n={iters})"
    )


def positive_int(text: str) -> int:
    """An argparse type: an integer of at least 1."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("capture", nargs="?")
    parser.add_argument("--iters", type=positive_int, default=1000)
    args = parser.parse_args()
    capture = os.path.realpath(args.capture) if args.capture else None
    if capture and not os.path.isdir(capture):
        parser.error(f"capture_dir {args.capture!r} is not a directory")

    proc = ImageProcessor()
    check_box_style(proc, "default backend")
    check_display_images()
    os.environ["EDGEFIRST_FORCE_BACKEND"] = "cpu"
    try:
        cpu = ImageProcessor()
    finally:
        del os.environ["EDGEFIRST_FORCE_BACKEND"]
    check_box_style(cpu, "cpu backend")
    check_cpu_refusals()
    if capture:
        time_decode(capture, args.iters)

    print(f"{'FAILED: ' + ', '.join(FAILED) if FAILED else 'all checks passed'}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
