"""Baseline guards for the packaging split.

Under A3 each wheel is self-contained, so the meaningful unit is the shipped
extension module. Per-crate rlib sizes do not predict wheel size, because a
statically linked consumer keeps only what it calls.
"""

import json
import pathlib
import shutil
import subprocess

import pytest

BASELINE = pathlib.Path("docs/baselines/2026-08-15-phase1-baseline.json")


@pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("rustc") is None,
    reason="size_baseline.sh records git and rustc; hardware runners have neither",
)
def test_baseline_script_emits_required_keys(tmp_path):
    out = tmp_path / "b.json"
    subprocess.run(["bash", "scripts/size_baseline.sh", str(out)], check=True)
    data = json.loads(out.read_text())
    assert "artifacts" in data and "wheels" in data
    assert data["commit"] and data["rustc"]


@pytest.mark.skipif(
    not BASELINE.exists(),
    reason="baselines are local measurement artifacts; this repo keeps "
    "benchmark output untracked (benchmarks/results/ is .gitkeep only). "
    "Run scripts/size_baseline.sh to generate.",
)
def test_local_baseline_records_the_monolith():
    """The number Phase 1 is measured against, when a baseline has been taken."""
    data = json.loads(BASELINE.read_text())
    wheels = data["wheels"]
    assert wheels, "baseline must record at least the pre-split wheel"
    mono = next(iter(wheels.values()))
    assert mono["on_disk"] > 4_000_000, "pre-split wheel was 4.32 MB on disk"
    assert mono["uncompressed_so"] > 17_000_000
