#!/usr/bin/env python3
"""Extract benchmark results from JSON files and print summary tables."""

from __future__ import annotations

from pathlib import Path

from benchmark_common import (
    ALL_BACKENDS,
    MASK_CONFIGS,
    PIPELINE_CONFIGS,
    PLATFORMS,
    fmt_ms,
    fmt_us,
    get_bench,
    load_json,
    parse_data_dir,
    print_section,
)


def tensor_results(data_dir: Path):
    print_section("TENSOR BENCHMARK RESULTS")

    for platform in PLATFORMS:
        data = load_json(data_dir, platform, "tensor")
        if data is None:
            print(f"  {platform}: NO DATA")
            continue
        print(f"\n  {platform}:")

        # Allocation
        print("    Allocation:")
        for buf in ["mem", "shm", "dma", "pbo"]:
            vals = []
            for res in ["360p", "720p", "1080p", "4K"]:
                v = get_bench(data, f"alloc/{buf}/u8/{res}")
                vals.append(fmt_us(v))
            if any(v != "—" for v in vals):
                print(f"      {buf:4s}: {vals[0]:>10s}  {vals[1]:>10s}  {vals[2]:>10s}  {vals[3]:>10s}")

        # Map/Unmap
        print("    Map/Unmap:")
        for buf in ["mem", "shm", "dma", "pbo"]:
            vals = []
            for res in ["360p", "720p", "1080p", "4K"]:
                v = get_bench(data, f"map/{buf}/u8/{res}")
                vals.append(fmt_us(v))
            if any(v != "—" for v in vals):
                print(f"      {buf:4s}: {vals[0]:>10s}  {vals[1]:>10s}  {vals[2]:>10s}  {vals[3]:>10s}")


def decoder_results(data_dir: Path):
    print_section("DECODER BENCHMARK RESULTS")

    for platform in PLATFORMS:
        data = load_json(data_dir, platform, "decoder")
        if data is None:
            print(f"  {platform}: NO DATA")
            continue
        print(f"\n  {platform}:")

        for b in data["benchmarks"]:
            print(f"    {b['name']:50s} {fmt_ms(b['median_us']):>10s}")


def pipeline_results(data_dir: Path):
    print_section("PIPELINE BENCHMARK RESULTS")

    for platform in PLATFORMS:
        print(f"\n  {platform}:")
        for backend in ALL_BACKENDS:
            data = load_json(data_dir, platform, f"pipeline-{backend}")
            if data is None:
                continue
            print(f"    [{backend}]:")
            for b in data["benchmarks"]:
                print(f"      {b['name']:55s} {fmt_ms(b['median_us']):>10s}")


def image_results(data_dir: Path):
    print_section("IMAGE BENCHMARK RESULTS")

    for platform in PLATFORMS:
        print(f"\n  {platform}:")
        for backend in ALL_BACKENDS:
            data = load_json(data_dir, platform, f"image-{backend}")
            if data is None:
                continue
            print(f"    [{backend}]:")
            for b in data["benchmarks"]:
                print(f"      {b['name']:55s} {fmt_ms(b['median_us']):>10s}")


def mask_results(data_dir: Path):
    print_section("MASK BENCHMARK RESULTS")

    for platform in PLATFORMS:
        print(f"\n  {platform}:")
        for backend in ["auto", "cpu", "opengl"]:
            data = load_json(data_dir, platform, f"mask-{backend}")
            if data is None:
                continue
            print(f"    [{backend}]:")
            for b in data["benchmarks"]:
                print(f"      {b['name']:55s} {fmt_ms(b['median_us']):>10s}")


def letterbox_table(data_dir: Path):
    """Print the letterbox pipeline table for BENCHMARKS.md."""
    print_section("LETTERBOX PIPELINE TABLE (for BENCHMARKS.md)")

    # 1080p -> 640x640
    print("  1080p → 640×640:")
    print(f"  {'Platform':<16} {'Compute':<8} {'Buffer':<6} {'YUYV→RGBA':>10} {'YUYV→RGB':>10} {'YUYV→8BPi':>10} {'NV12→RGBA':>10}")
    print(f"  {'-' * 16} {'-' * 8} {'-' * 6} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    for platform, compute, buf, json_file in PIPELINE_CONFIGS:
        data = load_json(data_dir, platform, json_file)
        if data is None:
            continue

        yuyv_rgba = get_bench(data, "letterbox/1920x1080/YUYV->640x640/RGBA")
        yuyv_rgb = get_bench(data, "letterbox/1920x1080/YUYV->640x640/RGB")
        yuyv_8bpi = get_bench(data, "letterbox/1920x1080/YUYV->640x640/8BPi")
        nv12_rgba = get_bench(data, "letterbox/1920x1080/NV12->640x640/RGBA")

        print(f"  {platform:<16} {compute:<8} {buf:<6} {fmt_ms(yuyv_rgba):>10} {fmt_ms(yuyv_rgb):>10} {fmt_ms(yuyv_8bpi):>10} {fmt_ms(nv12_rgba):>10}")


def mask_table(data_dir: Path):
    """Print the mask rendering table for BENCHMARKS.md."""
    print_section("MASK RENDERING TABLE (for BENCHMARKS.md)")

    print(f"  {'Platform':<16} {'Compute':<8} {'Buffer':<6} {'draw_masks':>12} {'draw_masks_proto':>18}")
    print(f"  {'-' * 16} {'-' * 8} {'-' * 6} {'-' * 12} {'-' * 18}")

    for platform, compute, buf, json_file in MASK_CONFIGS:
        data = load_json(data_dir, platform, json_file)
        if data is None:
            continue
        dm = get_bench(data, "draw_masks")
        dmp = get_bench(data, "draw_masks_proto")
        print(f"  {platform:<16} {compute:<8} {buf:<6} {fmt_ms(dm):>12} {fmt_ms(dmp):>18}")


def decoder_table(data_dir: Path):
    """Print decoder table for BENCHMARKS.md."""
    print_section("DECODER TABLE (for BENCHMARKS.md)")

    print(f"  {'Platform':<16} {'detect_i8':>10} {'detect_f32':>11} {'masks_i8':>10} {'masks_f32':>10}")
    print(f"  {'-' * 16} {'-' * 10} {'-' * 11} {'-' * 10} {'-' * 10}")

    for platform in PLATFORMS:
        data = load_json(data_dir, platform, "decoder")
        if data is None:
            continue
        detect_i8 = get_bench(data, "decoder/yolo/quant")
        detect_f32 = get_bench(data, "decoder/yolo/f32")
        masks_i8 = get_bench(data, "decoder/masks/i8")
        masks_f32 = get_bench(data, "decoder/masks/f32")
        print(f"  {platform:<16} {fmt_ms(detect_i8):>10} {fmt_ms(detect_f32):>11} {fmt_ms(masks_i8):>10} {fmt_ms(masks_f32):>10}")


if __name__ == "__main__":
    data_dir = parse_data_dir("Extract benchmark results.")

    tensor_results(data_dir)
    decoder_results(data_dir)
    decoder_table(data_dir)
    letterbox_table(data_dir)
    mask_table(data_dir)
    pipeline_results(data_dir)
    image_results(data_dir)
    mask_results(data_dir)
