#!/usr/bin/env python3
"""Generate BENCHMARKS.md table content from collected JSON data."""

from __future__ import annotations

from pathlib import Path

from benchmark_common import (
    MASK_CONFIGS,
    PIPELINE_CONFIGS,
    PLATFORMS,
    TENSOR_CONFIGS,
    fmt_us,
    get_bench,
    load_json,
    parse_data_dir,
)


# ============================================================================
# Tensor / Buffer Infrastructure
# ============================================================================


def gen_alloc_table(data_dir: Path):
    print("\n#### Allocation Latency\n")
    print("Measures `Tensor::new()` latency for each buffer type and resolution.\n")
    print("| Platform | Buffer | 720p (3.5 MB) | 1080p (7.9 MB) | 4K (31.6 MB) |")
    print("|----------|--------|---------------|-----------------|---------------|")

    for platform, bufs in TENSOR_CONFIGS:
        data = load_json(data_dir, platform, "tensor")
        for buf in bufs:
            vals = [get_bench(data, f"alloc/{buf}/u8/{r}") for r in ["720p", "1080p", "4K"]]
            if any(v is not None for v in vals):
                buf_label = buf.upper()
                print(f"| {platform} | {buf_label} | {fmt_us(vals[0])} | {fmt_us(vals[1])} | {fmt_us(vals[2])} |")


def gen_map_table(data_dir: Path):
    print("\n#### Map/Unmap Latency\n")
    print("Measures `tensor.map()` round-trip latency.\n")
    print("| Platform | Buffer | 720p | 1080p | 4K |")
    print("|----------|--------|------|-------|-----|")

    for platform, bufs in TENSOR_CONFIGS:
        data = load_json(data_dir, platform, "tensor")
        for buf in bufs:
            vals = [get_bench(data, f"map/{buf}/u8/{r}") for r in ["720p", "1080p", "4K"]]
            if any(v is not None and v > 0 for v in vals):
                buf_label = buf.upper()
                print(f"| {platform} | {buf_label} | {fmt_us(vals[0])} | {fmt_us(vals[1])} | {fmt_us(vals[2])} |")


# ============================================================================
# Letterbox Pipeline
# ============================================================================


def gen_letterbox_1080p(data_dir: Path):
    print("\n**1080p → 640×640:**\n")
    print("| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | YUYV→8BPi | NV12→RGBA | VYUY→RGBA |")
    print("|----------|---------|--------|-----------|----------|-----------|-----------|-----------|")

    for platform, compute, buf, jf in PIPELINE_CONFIGS:
        data = load_json(data_dir, platform, jf)
        yuyv_rgba = get_bench(data, "letterbox/1920x1080/YUYV->640x640/RGBA")
        yuyv_rgb = get_bench(data, "letterbox/1920x1080/YUYV->640x640/RGB")
        yuyv_8bpi = get_bench(data, "letterbox/1920x1080/YUYV->640x640/8BPi")
        nv12_rgba = get_bench(data, "letterbox/1920x1080/NV12->640x640/RGBA")
        vyuy_rgba = get_bench(data, "letterbox/1920x1080/VYUY->640x640/RGBA")
        print(f"| {platform} | {compute} | {buf} | {fmt_us(yuyv_rgba)} | {fmt_us(yuyv_rgb)} | {fmt_us(yuyv_8bpi)} | {fmt_us(nv12_rgba)} | {fmt_us(vyuy_rgba)} |")


def gen_letterbox_4k(data_dir: Path):
    print("\n**4K → 640×640:**\n")
    print("| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | NV12→RGBA |")
    print("|----------|---------|--------|-----------|----------|-----------|")

    for platform, compute, buf, jf in PIPELINE_CONFIGS:
        data = load_json(data_dir, platform, jf)
        yuyv_rgba = get_bench(data, "letterbox/3840x2160/YUYV->640x640/RGBA")
        yuyv_rgb = get_bench(data, "letterbox/3840x2160/YUYV->640x640/RGB")
        nv12_rgba = get_bench(data, "letterbox/3840x2160/NV12->640x640/RGBA")
        print(f"| {platform} | {compute} | {buf} | {fmt_us(yuyv_rgba)} | {fmt_us(yuyv_rgb)} | {fmt_us(nv12_rgba)} |")


# ============================================================================
# Format Conversion
# ============================================================================


def gen_convert_table(data_dir: Path):
    print("\n**1080p → 1080p:**\n")
    print("| Platform | Compute | Buffer | YUYV→RGBA | YUYV→RGB | NV12→RGBA | RGB→RGBA | RGBA→BGRA | RGBA→GREY |")
    print("|----------|---------|--------|-----------|----------|-----------|----------|-----------|-----------|")

    for platform, compute, buf, jf in PIPELINE_CONFIGS:
        data = load_json(data_dir, platform, jf)
        yuyv_rgba = get_bench(data, "convert/1920x1080/YUYV->RGBA")
        yuyv_rgb = get_bench(data, "convert/1920x1080/YUYV->RGB")
        nv12_rgba = get_bench(data, "convert/1920x1080/NV12->RGBA")
        rgb_rgba = get_bench(data, "convert/1920x1080/RGB->RGBA")
        rgba_bgra = get_bench(data, "convert/1920x1080/RGBA->BGRA")
        rgba_grey = get_bench(data, "convert/1920x1080/RGBA->GREY")
        print(f"| {platform} | {compute} | {buf} | {fmt_us(yuyv_rgba)} | {fmt_us(yuyv_rgb)} | {fmt_us(nv12_rgba)} | {fmt_us(rgb_rgba)} | {fmt_us(rgba_bgra)} | {fmt_us(rgba_grey)} |")


# ============================================================================
# Decoder
# ============================================================================


def gen_decoder_detect_table(data_dir: Path):
    print("\n**YOLOv8 Detection (84×8400, 80 classes):**\n")
    print("| Platform | Data Type | Decode + NMS | Decode Only | NMS Only | Dequantize |")
    print("|----------|-----------|-------------|-------------|----------|------------|")

    for platform in PLATFORMS:
        data = load_json(data_dir, platform, "decoder")
        if data is None:
            continue
        det_q = get_bench(data, "decoder/yolo/quant")
        dec = get_bench(data, "decoder/quant/decode_boxes")
        nms = get_bench(data, "decoder/quant/nms")
        deq = get_bench(data, "decoder/dequantize/i8")
        print(f"| {platform} | i8 (quant) | {fmt_us(det_q)} | {fmt_us(dec)} | {fmt_us(nms)} | {fmt_us(deq)} |")
        det_f = get_bench(data, "decoder/yolo/f32")
        if det_f is not None:
            print(f"| {platform} | f32 | {fmt_us(det_f)} | — | — | — |")


def gen_decoder_seg_table(data_dir: Path):
    print("\n**YOLOv8 Segmentation (mask coefficient → pixel decode):**\n")
    print("| Platform | Data Type | Masks Decode |")
    print("|----------|-----------|-------------|")

    for platform in PLATFORMS:
        data = load_json(data_dir, platform, "decoder")
        if data is None:
            continue
        masks_i8 = get_bench(data, "decoder/masks/i8")
        masks_f32 = get_bench(data, "decoder/masks/f32")
        if masks_i8 is not None:
            print(f"| {platform} | i8 (quant) | {fmt_us(masks_i8)} |")
        if masks_f32 is not None:
            print(f"| {platform} | f32 | {fmt_us(masks_f32)} |")


# ============================================================================
# Mask Rendering
# ============================================================================


def gen_mask_table(data_dir: Path):
    print("\n**640×640 RGBA destination, ~2 detections (YOLOv8n-seg):**\n")
    print("| Platform | Compute | Buffer | draw_decoded_masks (pre-decoded) | draw_proto_masks (fused) | hybrid_materialize_and_draw |")
    print("|----------|---------|--------|-------------------------------|------------------------|---------------------------|")

    for platform, compute, buf, jf in MASK_CONFIGS:
        data = load_json(data_dir, platform, jf)
        dm = get_bench(data, "draw_decoded_masks")
        dmp = get_bench(data, "draw_proto_masks")
        hybrid = get_bench(data, "hybrid_materialize_and_draw")
        print(f"| {platform} | {compute} | {buf} | {fmt_us(dm)} | {fmt_us(dmp)} | {fmt_us(hybrid)} |")


def gen_hybrid_comparison_table(data_dir: Path):
    """Generate hybrid path comparison table (CPU materialize + GL overlay vs fused GPU)."""
    print("\n**Hybrid Path Comparison (CPU materialize + GL overlay vs fused GPU):**\n")
    print("The hybrid path decodes masks on CPU (`materialize_segmentations`) then overlays via GL (`draw_decoded_masks`). This is faster than fused GPU `draw_proto_masks` on all tested platforms. The auto-selection in `ImageProcessor::draw_proto_masks()` now prefers the hybrid path when both CPU and OpenGL backends are available.\n")
    print("| Platform | Full GPU (GL draw_proto_masks) | Hybrid (GL) | Speedup | Auto draw_proto_masks |")
    print("|----------|-------------------------------|-------------|---------|----------------------|")

    for platform in PLATFORMS:
        gl_data = load_json(data_dir, platform, "mask-opengl")
        if gl_data is None:
            continue
        gpu_fused = get_bench(gl_data, "draw_proto_masks")
        hybrid = get_bench(gl_data, "hybrid_materialize_and_draw")
        auto_data = load_json(data_dir, platform, "mask-auto")
        auto_val = get_bench(auto_data, "draw_proto_masks") if auto_data else hybrid
        if gpu_fused is not None and hybrid is not None and hybrid > 0:
            speedup = gpu_fused / hybrid
            print(f"| {platform} | {fmt_us(gpu_fused)} | {fmt_us(hybrid)} | **{speedup:.1f}×** | {fmt_us(auto_val)} |")


# ============================================================================
# Decode Cost (from mask benchmark)
# ============================================================================


def gen_decode_cost_table(data_dir: Path):
    print("\n**Mask Decode Cost (CPU-only, measured in mask_benchmark):**\n")
    print("| Platform | Proto Decode (NMS+coefficients) | Full Materialize (NMS+coefficients+pixels) |")
    print("|----------|-------------------------------|-------------------------------------------|")

    for platform in PLATFORMS:
        data = load_json(data_dir, platform, "mask-cpu")
        if data is None:
            continue
        proto = get_bench(data, "decode_masks/proto")
        materialize = get_bench(data, "decode_masks/materialize")
        print(f"| {platform} | {fmt_us(proto)} | {fmt_us(materialize)} |")


if __name__ == "__main__":
    data_dir = parse_data_dir("Generate benchmark tables.")

    print("### Buffer Infrastructure\n")
    gen_alloc_table(data_dir)
    gen_map_table(data_dir)

    print("\n### Image Preprocessing: Letterbox Pipeline (Camera → Model Input)\n")
    gen_letterbox_1080p(data_dir)
    gen_letterbox_4k(data_dir)

    print("\n### Format Conversion (Same Size, No Resize)\n")
    gen_convert_table(data_dir)

    print("\n### Decoder Post-Processing\n")
    gen_decoder_detect_table(data_dir)
    gen_decoder_seg_table(data_dir)

    print("\n### Mask Rendering\n")
    gen_mask_table(data_dir)
    gen_hybrid_comparison_table(data_dir)
    gen_decode_cost_table(data_dir)
