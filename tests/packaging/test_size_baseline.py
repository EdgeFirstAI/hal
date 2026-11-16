"""Tests of the size baseline script and the locally recorded baseline."""

import json
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("rustc") is None,
    reason="size_baseline.sh records git and rustc; hardware runners have neither",
)
@pytest.mark.skipif(
    sys.platform == "win32",
    reason="size_baseline.sh measures the Unix artifact layout (lib*.so / .dylib, "
    "du, stat); Windows has no size baseline yet",
)
def test_baseline_script_emits_required_keys(tmp_path):
    out = tmp_path / "b.json"
    subprocess.run(["bash", "scripts/size_baseline.sh", str(out)], check=True)
    data = json.loads(out.read_text())
    assert "artifacts" in data
    assert "wheels" in data
    assert data["commit"]
    assert data["rustc"]
