# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Tests for the test-warning audit: every `WARNING:` line is found, in logs and JUnit, and grouped."""

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github" / "scripts" / "test_warnings.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("test_warnings", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tw = _load()

JUNIT = """<?xml version="1.0"?>
<testsuites><testsuite name="s">
  <testcase name="a"><system-err>WARNING: SKIPPED: a - no usable DMA heap
</system-err></testcase>
  <testcase name="b"><system-out>plain output</system-out>
    <system-err>WARNING: SKIPPED: a - no usable DMA heap</system-err></testcase>
</testsuite></testsuites>
"""


def test_logs_and_junit_are_both_read_and_grouped(tmp_path):
    (tmp_path / "board").mkdir()
    (tmp_path / "board" / "image.log").write_text(
        "running 2 tests\nWARNING: SKIPPED: b - G2D not available\ntest b ... ok\n"
    )
    (tmp_path / "test-results.xml").write_text(JUNIT)
    counts = tw.collect([str(tmp_path)])
    assert counts == {
        "SKIPPED: a - no usable DMA heap": 2,
        "SKIPPED: b - G2D not available": 1,
    }


def test_no_warnings_reports_none(tmp_path):
    (tmp_path / "x.log").write_text("test result: ok\n")
    assert "None." in tw.markdown(tw.collect([str(tmp_path)]))


def test_annotations_are_capped(tmp_path):
    counts = tw.Counter({f"SKIPPED: t{i}": 1 for i in range(tw.ANNOTATION_CAP + 3)})
    lines = tw.annotations(counts)
    assert len(lines) == tw.ANNOTATION_CAP + 1
    assert "3 more distinct" in lines[-1]


def test_unparsable_junit_and_missing_paths_are_ignored(tmp_path):
    (tmp_path / "broken.xml").write_text("<testsuites>")
    assert tw.collect([str(tmp_path), str(tmp_path / "missing.log")]) == {}


def test_table_escapes_pipes():
    table = tw.markdown(tw.Counter({"SKIPPED: a|b": 1}))
    assert "a\\|b" in table
