#!/usr/bin/env python3
"""Fix coverage.py XML report paths for SonarCloud compatibility.

The issue: coverage.py/pytest-cov/slipcover generate relative file paths
that may not match the repository structure expected by SonarCloud.

This script post-processes the Cobertura XML to:
1. Set <source> to the current directory (.)
2. Ensure filenames have the correct package prefix
3. Make paths resolvable by SonarCloud

Usage:
    python scripts/fix_coverage_paths.py coverage.xml tests
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def fix_coverage_xml(xml_file: str, source_dir: str = "tests") -> None:
    """Fix file paths in coverage XML report.

    Args:
        xml_file: Path to the coverage XML file to fix
        source_dir: The source directory prefix for Python test files
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Track how many files were fixed
    fixed_count = 0

    # Fix the <source> tags to use current directory (.)
    # Sonar expects <source> to be the root directory where files are located
    sources = root.findall(".//sources/source")
    if sources:
        for source_elem in sources:
            source_path = source_elem.text
            if source_path:
                # Set source to current directory
                source_elem.text = "."

    # Find all class elements and fix their filename attributes
    for class_elem in root.findall(".//class"):
        filename = class_elem.get("filename")
        if filename:
            # Normalize path - remove any leading ./
            if filename.startswith('./'):
                filename = filename[2:]
            # Ensure the path starts with the expected prefix
            if not filename.startswith(source_dir + "/"):
                # Check if it should be prefixed
                if not filename.startswith("tests/") and not filename.startswith("crates/"):
                    class_elem.set("filename", f"{source_dir}/{filename}")
                    fixed_count += 1
            else:
                class_elem.set("filename", filename)

    # Write the fixed XML back
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
