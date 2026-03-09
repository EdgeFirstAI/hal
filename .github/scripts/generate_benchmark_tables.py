#!/usr/bin/env python3
"""Generate BENCHMARKS.md table content from collected JSON data."""

import argparse
import json
from pathlib import Path


def get_bench_dir(args=None):
    parser = argparse.ArgumentParser(description="Generate benchmark tables.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "benchmarks",
        help="Directory containing benchmark JSON data (default: <repo>/benchmarks/)",
    )
    parsed = parser.parse_args(args)
    return parsed.data_dir


BENCH_DIR: Path = None  # set in main


def load_json(platform: str, name: str) -> dict | None:
    path = BENCH_DIR / platform / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)

def get_bench(data: dict | None, name: str) -> float | None:
    if data is None:
        return None
    for b in data["benchmarks"]:
        if b["name"] == name:
            return b["median_us"]
    return None

def fmt(val: float | None) -> str:
    """Format microseconds as ms string for table."""
    if val is None:
        return "—"
    if val < 10:
        return f"{val:.1f} us"
    if val < 1000:
        return f"{val:.0f} us"
    if val < 100000:
        return f"{val/1000:.1f} ms"
    return f"{val/1000:.0f} ms"


# ============================================================================
# Tensor / Buffer Infrastructure
# ============================================================================

def gen_alloc_table():
    print("\n#### Allocation Latency\n")
    print("Measures `Tensor::new()` latency for each buffer type and resolution.\n")
    print("| Platform | Buffer | 720p (3.5 MB) | 1080p (7.9 MB) | 4K (31.6 MB) |")
    print("|----------|--------|---------------|-----------------|---------------|")

    configs = [
        ("imx8mp-frdm", ["mem", "shm", "dma"]),
        ("imx95-frdm", ["mem", "shm", "dma"]),
        ("raspberrypi", ["mem", "shm", "dma"]),
        ("x86-desktop", ["mem", "shm"]),
    ]

    for platform, bufs in configs:
        data = load_json(platform, "tensor")
        for buf in bufs:
            vals = [get_bench(data, f"alloc/{buf}/u8/{r}") for r in ["720p", "1080p", "4K"]]
            if any(v is not None for v in vals):
                buf_label = buf.upper()
                print(f"| {platform} | {buf_label} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} |")

def gen_map_table():
    print("\n#### Map/Unmap Latency\n")
    print("Measures `tensor.map()` round-trip latency.\n")
    print("| Platform | Buffer | 720p | 1080p | 4K |")
    print("|----------|--------|------|-------|-----|")

    configs = [
        ("imx8mp-frdm", ["mem", "shm", "dma"]),
        ("imx95-frdm", ["mem", "shm", "dma"]),
        ("raspberrypi", ["mem", "shm", "dma"]),
        ("x86-desktop", ["mem", "shm"]),
    ]

    for platform, bufs in configs:
        data = load_json(platform, "tensor")
        for buf in bufs:
            vals = [get_bench(data, f"map/{buf}/u8/{r}") for r in ["720p", "1080p", "4K"]]
            if any(v is not None and v > 0 for v in vals):
                buf_label = buf.upper()
                print(f"| {platform} | {buf_label} | {fmt(vals[0])} | {fmt(vals[1])} | {fmt(vals[2])} |")


# ============================================================================
# Letterbox Pipeline
# ============================================================================

def gen_letterbox_1080p():
    print("\n**1080p → 640×640:**\n")
    print("| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | YUYV→8BPi | NV12→RGBA | VYUY→RGBA |")
    print("|----------|---------|--------|-----------|----------|-----------|-----------|-----------|")

    configs = [
        ("imx8mp-frdm", "G2D", "DMA", "pipeline-g2d"),
        ("imx8mp-frdm", "GL", "DMA", "pipeline-opengl"),
        ("imx8mp-frdm", "CPU", "Heap", "pipeline-cpu"),
        ("imx95-frdm", "G2D", "DMA", "pipeline-g2d"),
        ("imx95-frdm", "GL", "DMA", "pipeline-opengl"),
        ("imx95-frdm", "CPU", "Heap", "pipeline-cpu"),
        ("raspberrypi", "GL", "DMA", "pipeline-opengl"),
        ("raspberrypi", "CPU", "Heap", "pipeline-cpu"),
        ("x86-desktop", "GL", "PBO", "pipeline-opengl"),
        ("x86-desktop", "CPU", "Heap", "pipeline-cpu"),
    ]

    for platform, compute, buf, jf in configs:
        data = load_json(platform, jf)
        yuyv_rgba = get_bench(data, "letterbox/1920x1080/YUYV->640x640/RGBA")
        yuyv_rgb = get_bench(data, "letterbox/1920x1080/YUYV->640x640/RGB")
        yuyv_8bpi = get_bench(data, "letterbox/1920x1080/YUYV->640x640/8BPi")
        nv12_rgba = get_bench(data, "letterbox/1920x1080/NV12->640x640/RGBA")
        vyuy_rgba = get_bench(data, "letterbox/1920x1080/VYUY->640x640/RGBA")
        print(f"| {platform} | {compute} | {buf} | {fmt(yuyv_rgba)} | {fmt(yuyv_rgb)} | {fmt(yuyv_8bpi)} | {fmt(nv12_rgba)} | {fmt(vyuy_rgba)} |")


def gen_letterbox_4k():
    print("\n**4K → 640×640:**\n")
    print("| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | NV12→RGBA |")
    print("|----------|---------|--------|-----------|----------|-----------|")

    configs = [
        ("imx8mp-frdm", "G2D", "DMA", "pipeline-g2d"),
        ("imx8mp-frdm", "GL", "DMA", "pipeline-opengl"),
        ("imx8mp-frdm", "CPU", "Heap", "pipeline-cpu"),
        ("imx95-frdm", "G2D", "DMA", "pipeline-g2d"),
        ("imx95-frdm", "GL", "DMA", "pipeline-opengl"),
        ("imx95-frdm", "CPU", "Heap", "pipeline-cpu"),
        ("raspberrypi", "GL", "DMA", "pipeline-opengl"),
        ("raspberrypi", "CPU", "Heap", "pipeline-cpu"),
        ("x86-desktop", "GL", "PBO", "pipeline-opengl"),
        ("x86-desktop", "CPU", "Heap", "pipeline-cpu"),
    ]

    for platform, compute, buf, jf in configs:
        data = load_json(platform, jf)
        yuyv_rgba = get_bench(data, "letterbox/3840x2160/YUYV->640x640/RGBA")
        yuyv_rgb = get_bench(data, "letterbox/3840x2160/YUYV->640x640/RGB")
        nv12_rgba = get_bench(data, "letterbox/3840x2160/NV12->640x640/RGBA")
        print(f"| {platform} | {compute} | {buf} | {fmt(yuyv_rgba)} | {fmt(yuyv_rgb)} | {fmt(nv12_rgba)} |")


# ============================================================================
# Format Conversion
# ============================================================================

def gen_convert_table():
    print("\n**1080p → 1080p:**\n")
    print("| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | NV12→RGBA | RGB→RGBA | RGBA→BGRA | RGBA→GREY |")
    print("|----------|---------|--------|-----------|----------|-----------|----------|-----------|-----------|")

    configs = [
        ("imx8mp-frdm", "G2D", "DMA", "pipeline-g2d"),
        ("imx8mp-frdm", "GL", "DMA", "pipeline-opengl"),
        ("imx8mp-frdm", "CPU", "Heap", "pipeline-cpu"),
        ("imx95-frdm", "G2D", "DMA", "pipeline-g2d"),
        ("imx95-frdm", "GL", "DMA", "pipeline-opengl"),
        ("imx95-frdm", "CPU", "Heap", "pipeline-cpu"),
        ("raspberrypi", "GL", "DMA", "pipeline-opengl"),
        ("raspberrypi", "CPU", "Heap", "pipeline-cpu"),
        ("x86-desktop", "GL", "PBO", "pipeline-opengl"),
        ("x86-desktop", "CPU", "Heap", "pipeline-cpu"),
    ]

    for platform, compute, buf, jf in configs:
        data = load_json(platform, jf)
        yuyv_rgba = get_bench(data, "convert/1920x1080/YUYV->RGBA")
        yuyv_rgb = get_bench(data, "convert/1920x1080/YUYV->RGB")
        nv12_rgba = get_bench(data, "convert/1920x1080/NV12->RGBA")
        rgb_rgba = get_bench(data, "convert/1920x1080/RGB->RGBA")
        rgba_bgra = get_bench(data, "convert/1920x1080/RGBA->BGRA")
        rgba_grey = get_bench(data, "convert/1920x1080/RGBA->GREY")
        print(f"| {platform} | {compute} | {buf} | {fmt(yuyv_rgba)} | {fmt(yuyv_rgb)} | {fmt(nv12_rgba)} | {fmt(rgb_rgba)} | {fmt(rgba_bgra)} | {fmt(rgba_grey)} |")


# ============================================================================
# Decoder
# ============================================================================

def gen_decoder_detect_table():
    print("\n**YOLOv8 Detection (84×8400, 80 classes):**\n")
    print("| Platform | Data Type | Decode + NMS | Decode Only | NMS Only | Dequantize |")
    print("|----------|-----------|-------------|-------------|----------|------------|")

    for platform in ["imx8mp-frdm", "imx95-frdm", "raspberrypi", "x86-desktop"]:
        data = load_json(platform, "decoder")
        if data is None:
            continue
        # Quant row
        det_q = get_bench(data, "decoder/yolo/quant")
        dec = get_bench(data, "decoder/quant/decode_boxes")
        nms = get_bench(data, "decoder/quant/nms")
        deq = get_bench(data, "decoder/dequantize/i8")
        print(f"| {platform} | i8 (quant) | {fmt(det_q)} | {fmt(dec)} | {fmt(nms)} | {fmt(deq)} |")
        # f32 row (only some platforms)
        det_f = get_bench(data, "decoder/yolo/f32")
        if det_f is not None:
            print(f"| {platform} | f32 | {fmt(det_f)} | — | — | — |")


def gen_decoder_seg_table():
    print("\n**YOLOv8 Segmentation (detection + mask coefficients):**\n")
    print("| Platform | Data Type | Full Segdet | Masks Only |")
    print("|----------|-----------|------------|------------|")

    for platform in ["imx8mp-frdm", "imx95-frdm", "raspberrypi", "x86-desktop"]:
        data = load_json(platform, "decoder")
        if data is None:
            continue
        segdet = get_bench(data, "decoder/yolo_segdet/quant")
        masks = get_bench(data, "decoder/masks/f32")
        if segdet is not None or masks is not None:
            print(f"| {platform} | i8 (quant) | {fmt(segdet)} | {fmt(masks)} |")


# ============================================================================
# Mask Rendering
# ============================================================================

def gen_mask_table():
    print("\n**640×640 RGBA destination, ~2 detections (YOLOv8n-seg):**\n")
    print("| Platform | Compute | Buffer | draw_masks (pre-decoded) | draw_masks_proto (fused) | decode_masks_atlas | hybrid_materialize_and_draw |")
    print("|----------|---------|--------|------------------------|------------------------|--------------------|---------------------------|")

    configs = [
        ("imx8mp-frdm", "GL", "DMA", "mask-opengl"),
        ("imx8mp-frdm", "CPU", "Heap", "mask-cpu"),
        ("imx95-frdm", "GL", "DMA", "mask-opengl"),
        ("imx95-frdm", "CPU", "Heap", "mask-cpu"),
        ("raspberrypi", "GL", "DMA", "mask-opengl"),
        ("raspberrypi", "CPU", "Heap", "mask-cpu"),
        ("x86-desktop", "GL", "PBO", "mask-opengl"),
        ("x86-desktop", "CPU", "Heap", "mask-cpu"),
    ]

    for platform, compute, buf, jf in configs:
        data = load_json(platform, jf)
        dm = get_bench(data, "draw_masks")
        dmp = get_bench(data, "draw_masks_proto")
        dma = get_bench(data, "decode_masks_atlas")
        hybrid = get_bench(data, "hybrid_materialize_and_draw")
        print(f"| {platform} | {compute} | {buf} | {fmt(dm)} | {fmt(dmp)} | {fmt(dma)} | {fmt(hybrid)} |")


def gen_hybrid_comparison_table():
    """Generate hybrid path comparison table (CPU materialize + GL overlay vs fused GPU)."""
    print("\n**Hybrid Path Comparison (CPU materialize + GL overlay vs fused GPU):**\n")
    print("The hybrid path decodes masks on CPU (`materialize_segmentations`) then overlays via GL (`draw_masks`). This is faster than fused GPU `draw_masks_proto` on all tested platforms. The auto-selection in `ImageProcessor::draw_masks_proto()` now prefers the hybrid path when both CPU and OpenGL backends are available.\n")
    print("| Platform | Full GPU (GL draw_masks_proto) | Hybrid (GL) | Speedup | Auto draw_masks_proto |")
    print("|----------|-------------------------------|-------------|---------|----------------------|")

    for platform in ["imx8mp-frdm", "imx95-frdm", "raspberrypi", "x86-desktop"]:
        gl_data = load_json(platform, "mask-opengl")
        if gl_data is None:
            continue
        gpu_fused = get_bench(gl_data, "draw_masks_proto")
        hybrid = get_bench(gl_data, "hybrid_materialize_and_draw")
        # Auto path — load mask-auto if available, else use hybrid as proxy
        auto_data = load_json(platform, "mask-auto")
        auto_val = get_bench(auto_data, "draw_masks_proto") if auto_data else hybrid
        if gpu_fused is not None and hybrid is not None and hybrid > 0:
            speedup = gpu_fused / hybrid
            print(f"| {platform} | {fmt(gpu_fused)} | {fmt(hybrid)} | **{speedup:.1f}×** | {fmt(auto_val)} |")


# ============================================================================
# Decode Cost (from mask benchmark)
# ============================================================================

def gen_decode_cost_table():
    print("\n**Mask Decode Cost (CPU-only, measured in mask_benchmark):**\n")
    print("| Platform | Proto Decode (NMS+coefficients) | Full Materialize (NMS+coefficients+pixels) |")
    print("|----------|-------------------------------|-------------------------------------------|")

    for platform in ["imx8mp-frdm", "imx95-frdm", "raspberrypi", "x86-desktop"]:
        data = load_json(platform, "mask-cpu")
        if data is None:
            continue
        proto = get_bench(data, "decode_masks/proto")
        materialize = get_bench(data, "decode_masks/materialize")
        print(f"| {platform} | {fmt(proto)} | {fmt(materialize)} |")


if __name__ == "__main__":
    BENCH_DIR = get_bench_dir()

    print("### Buffer Infrastructure\n")
    gen_alloc_table()
    gen_map_table()

    print("\n### Image Preprocessing: Letterbox Pipeline (Camera → Model Input)\n")
    gen_letterbox_1080p()
    gen_letterbox_4k()

    print("\n### Format Conversion (Same Size, No Resize)\n")
    gen_convert_table()

    print("\n### Decoder Post-Processing\n")
    gen_decoder_detect_table()
    gen_decoder_seg_table()

    print("\n### Mask Rendering\n")
    gen_mask_table()
    gen_hybrid_comparison_table()
    gen_decode_cost_table()
