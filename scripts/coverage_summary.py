#!/usr/bin/env python3
"""
Parse coverage files and generate a markdown summary.

This script parses coverage reports from Rust (LCOV) and Python (Cobertura XML)
and outputs a formatted markdown summary.

Usage:
    # Default: looks for coverage files in target/
    python scripts/coverage_summary.py

    # Custom build directory
    python scripts/coverage_summary.py --build-dir /path/to/build

    # Output to file instead of stdout
    python scripts/coverage_summary.py --output coverage-report.md

    # GitHub Actions mode (outputs to GITHUB_STEP_SUMMARY)
    python scripts/coverage_summary.py --github-actions

Environment Variables (for CI):
    RUST_COVERAGE_PATHS: Comma-separated paths to Rust LCOV files
    PYTHON_COVERAGE_PATHS: Comma-separated paths to Python coverage XML files
"""

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def parse_rust_lcov(lcov_paths: str) -> dict:
    """Parse LCOV format for Rust coverage.

    Args:
        lcov_paths: Comma-separated paths to LCOV files

    Returns:
        Dictionary mapping file paths to coverage data
    """
    coverage = defaultdict(lambda: {'lines': 0, 'covered': 0})
    for lcov_path in lcov_paths.split(','):
        lcov_path = lcov_path.strip()
        if not lcov_path or not os.path.exists(lcov_path):
            continue
        try:
            current_file = None
            with open(lcov_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('SF:'):
                        path = line[3:]
                        # Normalize path to crates/...
                        match = re.search(r'crates/[^/]+/.*', path)
                        current_file = match.group(0) if match else path
                    elif line.startswith('DA:') and current_file:
                        parts = line[3:].split(',')
                        if len(parts) >= 2:
                            coverage[current_file]['lines'] += 1
                            if int(parts[1]) > 0:
                                coverage[current_file]['covered'] += 1
                    elif line == 'end_of_record':
                        current_file = None
        except Exception as e:
            print(f"Warning: Error processing {lcov_path}: {e}", file=sys.stderr)
    return dict(coverage)


def parse_python_cobertura(xml_paths: str, package_prefix: str = "edgefirst_hal") -> dict:
    """Parse Cobertura XML coverage format for Python.

    Args:
        xml_paths: Comma-separated paths to coverage XML files
        package_prefix: The package name to use for normalizing paths

    Returns:
        Dictionary mapping file paths to coverage data
    """
    coverage = {}
    for xml_path in xml_paths.split(','):
        xml_path = xml_path.strip()
        if not xml_path or not os.path.exists(xml_path):
            continue
        try:
            tree = ET.parse(xml_path)
            for cls in tree.getroot().findall('.//class'):
                filename = cls.get('filename', '')
                if not filename:
                    continue
                # Skip test files
                if '/tests/' in filename or filename.startswith('tests/'):
                    continue
                # Skip root __init__.py
                if filename == '__init__.py':
                    continue
                lines = covered = 0
                for line in cls.findall('.//line'):
                    lines += 1
                    if int(line.get('hits', 0)) > 0:
                        covered += 1
                if lines > 0:
                    if filename in coverage:
                        coverage[filename]['lines'] = max(coverage[filename]['lines'], lines)
                        coverage[filename]['covered'] = max(coverage[filename]['covered'], covered)
                    else:
                        coverage[filename] = {'lines': lines, 'covered': covered}
        except ET.ParseError as e:
            print(f"Warning: Failed to parse {xml_path}: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: Error processing {xml_path}: {e}", file=sys.stderr)
    return coverage


def format_table(coverage: dict, emoji: str, title: str) -> str:
    """Format coverage data as a markdown table.

    Args:
        coverage: Dictionary mapping file paths to coverage data
        emoji: Emoji to display in the title
        title: Section title

    Returns:
        Formatted markdown string
    """
    if not coverage:
        return f"## {emoji} {title}\n\nNo coverage data available.\n"

    lines = [
        f"## {emoji} {title}\n",
        "| File | Lines | Covered | Coverage |",
        "|------|-------|---------|----------|"
    ]

    total_lines = total_covered = 0
    for path in sorted(coverage.keys()):
        data = coverage[path]
        l, c = data['lines'], data['covered']
        total_lines += l
        total_covered += c
        pct = (c * 100 // l) if l > 0 else 0
        lines.append(f"| `{path}` | {l} | {c} | {pct}% |")

    if total_lines > 0:
        total_pct = (total_covered * 100 // total_lines)
        lines.append(f"| **Total** | **{total_lines}** | **{total_covered}** | **{total_pct}%** |")

    return "\n".join(lines) + "\n"


def generate_summary(rust_paths: str, python_paths: str,
                     github_actions: bool = False) -> str:
    """Generate the complete coverage summary.

    Args:
        rust_paths: Comma-separated paths to Rust coverage files
        python_paths: Comma-separated paths to Python coverage files
        github_actions: Whether to include GitHub Actions specific formatting

    Returns:
        Complete markdown summary
    """
    sections = []

    if github_actions:
        sections.append("# 📊 HAL Test & Coverage Summary\n")
    else:
        sections.append("# Coverage Summary\n")

    # Parse all coverage files
    rust_coverage = parse_rust_lcov(rust_paths)
    python_coverage = parse_python_cobertura(python_paths)

    # Generate tables
    sections.append(format_table(rust_coverage, "🦀", "Rust Coverage"))
    sections.append(format_table(python_coverage, "🐍", "Python Coverage"))

    # Overall summary
    total_lines = sum(d['lines'] for d in rust_coverage.values())
    total_lines += sum(d['lines'] for d in python_coverage.values())

    total_covered = sum(d['covered'] for d in rust_coverage.values())
    total_covered += sum(d['covered'] for d in python_coverage.values())

    if total_lines > 0:
        overall_pct = (total_covered * 100 // total_lines)
        sections.append(f"## 📈 Overall Coverage: **{overall_pct}%** ({total_covered}/{total_lines} lines)\n")

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description='Generate coverage summary from Rust and Python coverage reports.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--build-dir', '-b',
        default='target',
        help='Build directory containing coverage files (default: target)'
    )
    parser.add_argument(
        '--output', '-o',
        help='Output file (default: stdout)'
    )
    parser.add_argument(
        '--github-actions', '-g',
        action='store_true',
        help='GitHub Actions mode (uses env vars for paths)'
    )
    parser.add_argument(
        '--rust-coverage',
        help='Override Rust coverage file path(s), comma-separated'
    )
    parser.add_argument(
        '--python-coverage',
        help='Override Python coverage file path(s), comma-separated'
    )

    args = parser.parse_args()

    # Determine coverage file paths
    if args.github_actions:
        # Use environment variables in GitHub Actions
        rust_paths = args.rust_coverage or os.environ.get('RUST_COVERAGE_PATHS', '')
        python_paths = args.python_coverage or os.environ.get('PYTHON_COVERAGE_PATHS', '')
    else:
        # Use default paths in build directory
        build_dir = Path(args.build_dir)
        rust_paths = args.rust_coverage or str(build_dir / 'rust-coverage.lcov')
        python_paths = args.python_coverage or str(build_dir / 'python-coverage.xml')

    # Generate summary
    summary = generate_summary(rust_paths, python_paths, args.github_actions)

    # Output
    if args.output:
        with open(args.output, 'w') as f:
            f.write(summary)
        print(f"Coverage summary written to {args.output}", file=sys.stderr)
    else:
        print(summary)


if __name__ == '__main__':
    main()
