"""Shared utilities, constants, and configs for benchmark scripts."""

import argparse
import json
from pathlib import Path


def parse_data_dir(description: str, args=None) -> Path:
    """Parse --data-dir argument and return the benchmark data directory."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "benchmarks",
        help="Directory containing benchmark JSON data (default: <repo>/benchmarks/)",
    )
    parsed = parser.parse_args(args)
    return parsed.data_dir


def load_json(data_dir: Path, platform: str, name: str) -> dict | None:
    """Load a benchmark JSON file, returning None if it doesn't exist."""
    path = data_dir / platform / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_bench(data: dict | None, name: str) -> float | None:
    """Extract median_us for a benchmark by name."""
    if data is None:
        return None
    for b in data["benchmarks"]:
        if b["name"] == name:
            return b["median_us"]
    return None


def fmt_us(val: float | None) -> str:
    """Format microseconds with auto-scaling (us/ms)."""
    if val is None:
        return "—"
    if val < 10:
        return f"{val:.1f} us"
    if val < 1000:
        return f"{val:.0f} us"
    if val < 100000:
        return f"{val / 1000:.1f} ms"
    return f"{val / 1000:.0f} ms"


def fmt_ms(val: float | None) -> str:
    """Format microseconds as milliseconds."""
    if val is None:
        return "—"
    return f"{val / 1000:.1f} ms"


def print_section(title: str):
    """Print a section header for the extract script."""
    print(f"\n{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}\n")


# Consistent platform ordering
PLATFORMS = ["imx8mp-frdm", "imx95-frdm", "raspberrypi", "x86-desktop"]

# All compute backends for iteration
ALL_BACKENDS = ["auto", "cpu", "g2d", "opengl"]

# Pipeline configs: (platform, compute_label, buffer_label, json_file)
PIPELINE_CONFIGS = [
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

# Mask configs: (platform, compute_label, buffer_label, json_file)
MASK_CONFIGS = [
    ("imx8mp-frdm", "GL", "DMA", "mask-opengl"),
    ("imx8mp-frdm", "CPU", "Heap", "mask-cpu"),
    ("imx95-frdm", "GL", "DMA", "mask-opengl"),
    ("imx95-frdm", "CPU", "Heap", "mask-cpu"),
    ("raspberrypi", "GL", "DMA", "mask-opengl"),
    ("raspberrypi", "CPU", "Heap", "mask-cpu"),
    ("x86-desktop", "GL", "PBO", "mask-opengl"),
    ("x86-desktop", "CPU", "Heap", "mask-cpu"),
]

# Tensor configs: (platform, buffer_types)
TENSOR_CONFIGS = [
    ("imx8mp-frdm", ["mem", "shm", "dma"]),
    ("imx95-frdm", ["mem", "shm", "dma"]),
    ("raspberrypi", ["mem", "shm", "dma"]),
    ("x86-desktop", ["mem", "shm"]),
]
