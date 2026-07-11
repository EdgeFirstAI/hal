"""Tests for .github/scripts/fix_coverage_paths.py."""

from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / ".github" / "scripts" / "fix_coverage_paths.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fix_coverage_paths", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fix_mod():
    return _load_module()


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("scripts/decoder_generate_fixture.py", "scripts/decoder_generate_fixture.py"),
        (
            "scripts/per_scale_decode_reference.py",
            "scripts/per_scale_decode_reference.py",
        ),
        ("foo.py", "tests/foo.py"),
        ("tests/python/test_foo.py", "tests/python/test_foo.py"),
        ("crates/python/src/lib.rs", "crates/python/src/lib.rs"),
        (
            ".github/scripts/fix_coverage_paths.py",
            ".github/scripts/fix_coverage_paths.py",
        ),
        (
            "./scripts/decoder_generate_fixture.py",
            "scripts/decoder_generate_fixture.py",
        ),
        ("./foo.py", "tests/foo.py"),
    ],
)
def test_normalize_coverage_filename(fix_mod, filename, expected):
    assert fix_mod.normalize_coverage_filename(filename, "tests") == expected


def test_fix_coverage_xml_preserves_scripts(fix_mod, tmp_path):
    xml_path = tmp_path / "coverage.xml"
    xml_path.write_text(
        """<?xml version="1.0" ?>
<coverage>
  <sources>
    <source>/tmp/build</source>
  </sources>
  <packages>
    <package name="scripts">
      <classes>
        <class filename="scripts/decoder_generate_fixture.py" name="decoder_generate_fixture"/>
        <class filename="scripts/per_scale_decode_reference.py" name="per_scale_decode_reference"/>
        <class filename="foo.py" name="foo"/>
        <class filename="tests/python/test_foo.py" name="test_foo"/>
        <class filename="crates/python/src/x.py" name="x"/>
      </classes>
    </package>
  </packages>
</coverage>
""",
        encoding="utf-8",
    )

    fix_mod.fix_coverage_xml(str(xml_path), "tests")

    root = ET.parse(xml_path).getroot()
    sources = [s.text for s in root.findall(".//sources/source")]
    assert sources == ["."]

    filenames = [c.get("filename") for c in root.findall(".//class")]
    assert filenames == [
        "scripts/decoder_generate_fixture.py",
        "scripts/per_scale_decode_reference.py",
        "tests/foo.py",
        "tests/python/test_foo.py",
        "crates/python/src/x.py",
    ]
