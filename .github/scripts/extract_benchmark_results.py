#!/usr/bin/env python3
"""Extract benchmark results from JSON files and print summary tables."""

import argparse
import json
from pathlib import Path


def get_bench_dir(args=None):
    parser = argparse.ArgumentParser(description="Extract benchmark results.")
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


def get_bench(data: dict, name: str) -> float | None:
    """Get median_us for a benchmark by name."""
    if data is None:
        return None
    for b in data["benchmarks"]:
        if b["name"] == name:
            return b["median_us"]
    return None


def fmt_us(val: float | None) -> str:
    """Format microseconds as human-readable string."""
    if val is None:
        return "—"
    if val < 1:
        return f"{val:.1f} us"
    if val < 1000:
        return f"{val:.0f} us"
    return f"{val / 1000:.1f} ms"


def fmt_ms(val: float | None) -> str:
    """Format microseconds as milliseconds."""
    if val is None:
        return "—"
    return f"{val / 1000:.1f} ms"


PLATFORMS = ["x86-desktop", "imx8mp-frdm", "imx95-frdm", "raspberrypi"]

def print_section(title: str):
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def tensor_results():
    print_section("TENSOR BENCHMARK RESULTS")

    for platform in PLATFORMS:
        data = load_json(platform, "tensor")
        if data is None:
            print(f"  {platform}: NO DATA")
            continue
        print(f"\n  {platform}:")

        # Allocation
        print(f"    Allocation:")
        for buf in ["mem", "shm", "dma", "pbo"]:
            vals = []
            for res in ["480p", "720p", "1080p", "4K"]:
                v = get_bench(data, f"alloc/{buf}/u8/{res}")
                vals.append(fmt_us(v))
            if any(v != "—" for v in vals):
                print(f"      {buf:4s}: {vals[0]:>10s}  {vals[1]:>10s}  {vals[2]:>10s}  {vals[3]:>10s}")

        # Map/Unmap
        print(f"    Map/Unmap:")
        for buf in ["mem", "shm", "dma", "pbo"]:
            vals = []
            for res in ["480p", "720p", "1080p", "4K"]:
                v = get_bench(data, f"map/{buf}/u8/{res}")
                vals.append(fmt_us(v))
            if any(v != "—" for v in vals):
                print(f"      {buf:4s}: {vals[0]:>10s}  {vals[1]:>10s}  {vals[2]:>10s}  {vals[3]:>10s}")


def decoder_results():
    print_section("DECODER BENCHMARK RESULTS")

    for platform in PLATFORMS:
        data = load_json(platform, "decoder")
        if data is None:
            print(f"  {platform}: NO DATA")
            continue
        print(f"\n  {platform}:")

        for b in data["benchmarks"]:
            print(f"    {b['name']:50s} {fmt_ms(b['median_us']):>10s}")


def pipeline_results():
    print_section("PIPELINE BENCHMARK RESULTS")

    for platform in PLATFORMS:
        print(f"\n  {platform}:")
        for backend in ["auto", "cpu", "g2d", "opengl"]:
            data = load_json(platform, f"pipeline-{backend}")
            if data is None:
                continue
            print(f"    [{backend}]:")
            for b in data["benchmarks"]:
                print(f"      {b['name']:55s} {fmt_ms(b['median_us']):>10s}")


def image_results():
    print_section("IMAGE BENCHMARK RESULTS")

    for platform in PLATFORMS:
        print(f"\n  {platform}:")
        for backend in ["auto", "cpu", "g2d", "opengl"]:
            data = load_json(platform, f"image-{backend}")
            if data is None:
                continue
            print(f"    [{backend}]:")
            for b in data["benchmarks"]:
                print(f"      {b['name']:55s} {fmt_ms(b['median_us']):>10s}")


def mask_results():
    print_section("MASK BENCHMARK RESULTS")

    for platform in PLATFORMS:
        print(f"\n  {platform}:")
        for backend in ["auto", "cpu", "opengl"]:
            data = load_json(platform, f"mask-{backend}")
            if data is None:
                continue
            print(f"    [{backend}]:")
            for b in data["benchmarks"]:
                print(f"      {b['name']:55s} {fmt_ms(b['median_us']):>10s}")


def letterbox_table():
    """Print the letterbox pipeline table for BENCHMARKS.md."""
    print_section("LETTERBOX PIPELINE TABLE (for BENCHMARKS.md)")

    # 1080p -> 640x640
    print("  1080p → 640×640:")
    print(f"  {'Platform':<16} {'Compute':<8} {'Buffer':<6} {'YUYV→RGBA':>10} {'YUYV→RGB':>10} {'YUYV→8BPi':>10} {'NV12→RGBA':>10}")
    print(f"  {'-'*16} {'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    configs = [
        # (platform, compute, buffer, json_file)
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

    for platform, compute, buf, json_file in configs:
        data = load_json(platform, json_file)
        if data is None:
            continue

        # Find letterbox benchmarks matching 1080p->640x640
        yuyv_rgba = get_bench(data, "letterbox/cpu/1920x1080/YUYV->640x640/RGBA") or \
                    get_bench(data, "letterbox/g2d/1920x1080/YUYV->640x640/RGBA") or \
                    get_bench(data, "letterbox/opengl/1920x1080/YUYV->640x640/RGBA")
        yuyv_rgb = get_bench(data, "letterbox/cpu/1920x1080/YUYV->640x640/RGB") or \
                   get_bench(data, "letterbox/g2d/1920x1080/YUYV->640x640/RGB") or \
                   get_bench(data, "letterbox/opengl/1920x1080/YUYV->640x640/RGB")
        yuyv_8bpi = get_bench(data, "letterbox/cpu/1920x1080/YUYV->640x640/8BPi") or \
                    get_bench(data, "letterbox/g2d/1920x1080/YUYV->640x640/8BPi") or \
                    get_bench(data, "letterbox/opengl/1920x1080/YUYV->640x640/8BPi")
        nv12_rgba = get_bench(data, "letterbox/cpu/1920x1080/NV12->640x640/RGBA") or \
                    get_bench(data, "letterbox/g2d/1920x1080/NV12->640x640/RGBA") or \
                    get_bench(data, "letterbox/opengl/1920x1080/NV12->640x640/RGBA")

        print(f"  {platform:<16} {compute:<8} {buf:<6} {fmt_ms(yuyv_rgba):>10} {fmt_ms(yuyv_rgb):>10} {fmt_ms(yuyv_8bpi):>10} {fmt_ms(nv12_rgba):>10}")


def mask_table():
    """Print the mask rendering table for BENCHMARKS.md."""
    print_section("MASK RENDERING TABLE (for BENCHMARKS.md)")

    print(f"  {'Platform':<16} {'Compute':<8} {'Buffer':<6} {'draw_masks':>12} {'draw_masks_proto':>18} {'decode_atlas':>14}")
    print(f"  {'-'*16} {'-'*8} {'-'*6} {'-'*12} {'-'*18} {'-'*14}")

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

    for platform, compute, buf, json_file in configs:
        data = load_json(platform, json_file)
        if data is None:
            continue
        dm = get_bench(data, "draw_masks")
        dmp = get_bench(data, "draw_masks_proto")
        dma = get_bench(data, "decode_masks_atlas")
        print(f"  {platform:<16} {compute:<8} {buf:<6} {fmt_ms(dm):>12} {fmt_ms(dmp):>18} {fmt_ms(dma):>14}")


def decoder_table():
    """Print decoder table for BENCHMARKS.md."""
    print_section("DECODER TABLE (for BENCHMARKS.md)")

    print(f"  {'Platform':<16} {'detect_i8':>10} {'detect_f32':>11} {'segdet_i8':>10} {'masks_f32':>10}")
    print(f"  {'-'*16} {'-'*10} {'-'*11} {'-'*10} {'-'*10}")

    for platform in PLATFORMS:
        data = load_json(platform, "decoder")
        if data is None:
            continue
        detect_i8 = get_bench(data, "detect_i8")
        detect_f32 = get_bench(data, "detect_f32")
        segdet_i8 = get_bench(data, "segdet_i8")
        masks_f32 = get_bench(data, "masks_f32")
        print(f"  {platform:<16} {fmt_ms(detect_i8):>10} {fmt_ms(detect_f32):>11} {fmt_ms(segdet_i8):>10} {fmt_ms(masks_f32):>10}")


if __name__ == "__main__":
    BENCH_DIR = get_bench_dir()

    tensor_results()
    decoder_results()
    decoder_table()
    letterbox_table()
    mask_table()
    pipeline_results()
    image_results()
    mask_results()
