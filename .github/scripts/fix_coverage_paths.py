#!/usr/bin/env python3
"""Fix coverage.py XML report paths for SonarCloud compatibility.

The issue: coverage.py/pytest-cov/slipcover generate relative file paths
that may not match the repository structure expected by SonarCloud.

This script post-processes the Cobertura XML to:
1. Set <source> to the current directory (.)
2. Ensure filenames have the correct package prefix
3. Make paths resolvable by SonarCloud

Known repo-relative roots (``tests/``, ``crates/``, ``scripts/``,
``.github/``) are left unchanged. Only bare or otherwise ambiguous
filenames are prefixed with ``source_dir`` (default ``tests``).

Usage:
    python .github/scripts/fix_coverage_paths.py coverage.xml tests
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Paths that are already repo-relative and must not be rewritten.
KNOWN_ROOTS = ("tests/", "crates/", "scripts/", ".github/")


def normalize_coverage_filename(filename: str, source_dir: str = "tests") -> str:
    """Return a repo-relative coverage filename for SonarCloud.

    Args:
        filename: Cobertura ``class`` filename attribute.
        source_dir: Prefix applied only to bare/ambiguous names.

    Returns:
        Normalized path that should resolve under the repo root.
    """
    if filename.startswith("./"):
        filename = filename[2:]

    if filename.startswith(source_dir + "/"):
        return filename

    if any(filename.startswith(root) for root in KNOWN_ROOTS):
        return filename

    return f"{source_dir}/{filename}"


def fix_coverage_xml(xml_file: str, source_dir: str = "tests") -> None:
    """Fix file paths in coverage XML report.

    Args:
        xml_file: Path to the coverage XML file to fix
        source_dir: The source directory prefix for bare Python filenames
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    fixed_count = 0

    # Sonar expects <source> to be the root directory where files are located
    sources = root.findall(".//sources/source")
    if sources:
        for source_elem in sources:
            if source_elem.text:
                source_elem.text = "."

    for class_elem in root.findall(".//class"):
        filename = class_elem.get("filename")
        if not filename:
            continue
        normalized = normalize_coverage_filename(filename, source_dir)
        if normalized != filename:
            fixed_count += 1
        class_elem.set("filename", normalized)

    tree.write(xml_file, encoding="utf-8", xml_declaration=True)

    print(f"Fixed {fixed_count} file paths in {xml_file}")
    print("  Set <source> tag to: .")
    if fixed_count > 0:
        print(f"  Prefixed {fixed_count} filenames with '{source_dir}/'")


def main() -> int:
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: fix_coverage_paths.py <xml_file> [source_dir]")
        print("  xml_file: Path to coverage XML report")
        print("  source_dir: Source directory prefix (default: tests)")
        return 1

    xml_file = sys.argv[1]
    source_dir = sys.argv[2] if len(sys.argv) > 2 else "tests"

    if not Path(xml_file).exists():
        print(f"ERROR: File not found: {xml_file}")
        return 1

    try:
        fix_coverage_xml(xml_file, source_dir)
        return 0
    except Exception as e:
        print(f"ERROR: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
