"""A3 wheel layout: self-contained, typed, namespace intact."""

import pathlib
import subprocess
import sys

import pytest

WHEEL_DIR = pathlib.Path("target/wheels")


@pytest.mark.skipif(
    not list(WHEEL_DIR.glob("*.whl")), reason="no wheels built; run `make wheel`"
)
def test_layout_checker_passes():
    r = subprocess.run(
        [sys.executable, "scripts/check_wheel_layout.py", str(WHEEL_DIR)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr
