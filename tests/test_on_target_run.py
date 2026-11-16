# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Tests for what the board-side runner reports, scripts/on-target-run.sh.

A board's result is read from three files: `summary.txt` by
scripts/on-target-test.sh, `summary.json` by the fleet roll-up, and
`warnings.txt` by whoever audits the skips. These tests run the runner over a
stand-in bundle whose programs are shell stubs and pin what those readers
depend on: the summary is written on a board with no Python interpreter, it is
valid JSON whatever characters its fields carry, and every skipped test counts
once.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="the runner is a POSIX shell script for boards"
)

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "on-target-run.sh"
BASH = shutil.which("bash") or "bash"

MANIFEST = {
    "commit": "60208bd0",
    "features": "edgefirst-image/opengl",
    "built": 'by "ci" on a\\b, FILTER=none',
}

LIBTEST_BINARY = """\
if [[ $# -gt 1 ]]; then
  echo 'running 0 tests'
  echo
  echo 'test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 3 filtered out'
  exit 0
fi
echo 'running 3 tests'
echo 'WARNING: SKIPPED: fake::needs_gpu - no render node'
echo 'test result: ok. 3 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out'
"""

CODEC_C_PROGRAM = """\
echo 'SKIP: no H.264 encoder on this board'
echo 'test_codec_api: ok'
"""

# Stands in for a board with no usable interpreter: every call fails, as a
# missing python3 does.
NO_PYTHON = "exit 127\n"

# Stands in for a board whose pytest run skipped five tests, four of them
# sharing a location and reason, which pytest reports on one line.
PYTEST_WITH_GROUPED_SKIPS = """\
[[ "$1" == -c ]] && exit 0
if [[ "$1" == -m && "$2" == pytest ]]; then
  echo 'ssss.s                                                         [100%]'
  echo '==================== short test summary info ===================='
  echo 'SKIPPED [4] tests/codec/test_codec_api.py:41: no H.264 encoder here'
  echo 'SKIPPED [1] tests/codec/test_codec_api.py:77: no MJPEG decoder'
  echo '1 passed, 5 skipped in 0.01s'
  exit 0
fi
exit 97
"""


def _stub(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\n" + body, newline="\n")
    path.chmod(0o755)


def _run(tmp_path, python3, env=None, with_wheels=True):
    """Run the runner over a stand-in bundle; return (process, results dir)."""
    bundle = tmp_path / "bundle"
    (bundle / "scripts").mkdir(parents=True)
    shutil.copy(RUNNER, bundle / "scripts" / "on-target-run.sh")
    _stub(bundle / "scripts" / "dma-heap-setup.sh", "exit 0\n")
    _stub(bundle / "scripts" / "check-single-home.sh", "echo '  G1   PASS'\n")
    _stub(bundle / "bin" / "edgefirst_fake-0123456789abcdef", LIBTEST_BINARY)
    _stub(bundle / "target" / "debug" / "test_two_library_user", "exit 0\n")
    _stub(bundle / "target" / "debug" / "test_codec_api", CODEC_C_PROGRAM)
    if with_wheels:
        (bundle / "python").mkdir()
    (bundle / "MANIFEST").write_text(
        "".join(f"{k}={v}\n" for k, v in MANIFEST.items()), newline="\n"
    )
    tools = tmp_path / "tools"
    _stub(tools / "python3", python3)

    results = tmp_path / "results"
    run_env = {
        **os.environ,
        "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
        "FILTER": "",
        "HAL_ONTARGET_REQUIRE_GPU": "0",
        "HAL_BOARD_ENTRY": "imx95-evk",
        "RUNNER_NAME": "imx95evk-02",
        **(env or {}),
    }
    proc = subprocess.run(
        [
            BASH,
            (bundle / "scripts" / "on-target-run.sh").as_posix(),
            results.as_posix(),
        ],
        env=run_env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return proc, results


def _summary_txt(results):
    result, _, detail = (
        (results / "summary.txt").read_text().rstrip("\n").partition("|")
    )
    return result, detail


def _warnings(results):
    """warnings.txt as {warning: count}."""
    counts = {}
    for line in (results / "warnings.txt").read_text().splitlines():
        count, warning = line.split(None, 1)
        counts[warning] = int(count)
    return counts


def test_a_board_without_python_still_reports_its_result(tmp_path):
    proc, results = _run(tmp_path, NO_PYTHON)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert _summary_txt(results)[0] == "PASS"
    summary = json.loads((results / "summary.json").read_text())
    assert summary["result"] == "PASS"
    assert summary["detail"] == _summary_txt(results)[1]
    assert summary["entry"] == "imx95-evk"
    assert summary["runner"] == "imx95evk-02"
    assert summary["bundle"] == MANIFEST
    # One skip each from the test binary, the C program and the Python tests
    # the board could not run.
    assert summary["tests"] == {"passed": 3, "failed": 0, "ignored": 1, "skipped": 3}
    skipped_python = [w for w in _warnings(results) if "Python codec tests" in w]
    assert len(skipped_python) == 1, _warnings(results)


def test_a_bundle_without_wheels_skips_the_python_tests(tmp_path):
    proc, results = _run(tmp_path, NO_PYTHON, with_wheels=False)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    summary = json.loads((results / "summary.json").read_text())
    assert summary["tests"]["skipped"] == 3
    assert any("PYTHON_WHEELS" in w for w in _warnings(results))


def test_grouped_pytest_skips_count_once_per_test(tmp_path):
    proc, results = _run(tmp_path, PYTEST_WITH_GROUPED_SKIPS)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    warnings = _warnings(results)
    assert (
        warnings["SKIPPED: tests/codec/test_codec_api.py:41: no H.264 encoder here"]
        == 4
    )
    assert warnings["SKIPPED: tests/codec/test_codec_api.py:77: no MJPEG decoder"] == 1
    summary = json.loads((results / "summary.json").read_text())
    # The binary's skip, the C program's, and pytest's five.
    assert summary["tests"]["skipped"] == 7
    assert "7 tests skipped" in summary["detail"]


def test_summary_json_survives_quotes_and_backslashes(tmp_path):
    proc, results = _run(
        tmp_path,
        NO_PYTHON,
        env={
            "FILTER": 'no "such" test\\name',
            "HAL_BOARD_ENTRY": 'imx95-frdm+"ara240"',
            "RUNNER_NAME": "lab\\imx95\tfrdm",
        },
    )

    # A filter that matches no test name is NO-TESTS, and its detail quotes
    # the filter.
    assert proc.returncode == 1, proc.stdout + proc.stderr
    result, detail = _summary_txt(results)
    assert result == "NO-TESTS"
    assert '"such"' in detail
    summary = json.loads((results / "summary.json").read_text())
    assert summary["result"] == "NO-TESTS"
    assert summary["detail"] == detail
    assert summary["entry"] == 'imx95-frdm+"ara240"'
    assert summary["runner"] == "lab\\imx95\tfrdm"
    assert summary["bundle"] == MANIFEST
