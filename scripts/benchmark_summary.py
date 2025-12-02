#!/usr/bin/env python3
"""Generate markdown summary from benchmark results.

Parses benchmark JSON output and generates GitHub Actions step summary.

Usage:
    python scripts/benchmark_summary.py benchmark-results/benchmark.json
    python scripts/benchmark_summary.py benchmark-results/benchmark.json --output summary.md
"""

import argparse
import json
import os
import sys


def generate_summary(benchmark_file: str) -> str:
    """Generate markdown summary from benchmark JSON.

    Args:
        benchmark_file: Path to benchmark JSON file

    Returns:
        Markdown formatted summary string
    """
    lines = [
        "## Benchmark Results (NXP i.MX8M Plus)",
        ""
    ]

    if not os.path.exists(benchmark_file):
        lines.append("No benchmark data available")
        return "\n".join(lines)

    with open(benchmark_file) as f:
        data = json.load(f)

    benchmarks = data.get('benchmarks', {})
    if not benchmarks:
        lines.append("No benchmark data available")
        return "\n".join(lines)

    lines.extend([
        "| Metric | Value | Unit |",
        "|--------|-------|------|"
    ])

    for name, result in benchmarks.items():
        if isinstance(result, dict):
            mean = result.get('mean', 'N/A')
            unit = result.get('unit', 'N/A')
            if isinstance(mean, (int, float)):
                lines.append(f"| {name} | {mean:.2f} | {unit} |")
            else:
                lines.append(f"| {name} | {mean} | {unit} |")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Generate markdown summary from benchmark results.'
    )
    parser.add_argument(
        'benchmark_file',
        help='Path to benchmark JSON file'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file (default: stdout)'
    )

    args = parser.parse_args()

    summary = generate_summary(args.benchmark_file)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(summary)
    else:
        print(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
