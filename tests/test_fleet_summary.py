# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Tests for the hardware-test / hardware-bench fleet roll-up.

The roll-up turns one artifact per board into the run's summary and decides
whether the run is red. These tests pin what the workflows depend on: every
requested board gets a row naming the runner that ran it, a board that failed
or uploaded nothing fails the run, a board reported unavailable does not, and
a run that scheduled nothing fails rather than passing having tested nothing.
"""

import csv
import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github" / "scripts" / "fleet_summary.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("fleet_summary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fleet_summary = _load()


def _entry(runner):
    return {"runner": runner, "labels": ["self-hosted", *runner.split("+")]}


def _test_result(root, entry, result="PASS", runner="imx8mpevk-04", passed=2007):
    d = root / f"hardware-test-{entry}"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(
        json.dumps(
            {
                "result": result,
                "detail": "50 binaries, 55 tests skipped",
                "runner": runner,
                "host": "board",
                "tests": {
                    "passed": passed,
                    "failed": 0 if result == "PASS" else 3,
                    "ignored": 20,
                    "skipped": 55,
                },
                "bundle": {
                    "commit": "60208bd0",
                    "glibc": "2.35",
                    "features": "edgefirst-image/dma_test_formats",
                },
            }
        )
    )


def _run(argv, capsys):
    status = fleet_summary.main(argv)
    return status, capsys.readouterr().out


def test_tests_all_pass_names_each_runner(tmp_path, capsys):
    _test_result(tmp_path, "imx8mp-evk", runner="imx8mpevk-06")
    _test_result(tmp_path, "iq9075-evk", runner="iq9075-evk")
    status, out = _run(
        [
            "tests",
            "--results",
            str(tmp_path),
            "--requested",
            json.dumps([_entry("imx8mp-evk"), _entry("iq9075-evk")]),
        ],
        capsys,
    )
    assert status == 0
    assert "`imx8mpevk-06`" in out
    assert "`iq9075-evk`" in out
    assert "| 2007 |" in out
    assert "features `edgefirst-image/dma_test_formats`" in out


def test_tests_failed_board_fails_the_run(tmp_path, capsys):
    _test_result(tmp_path, "rpi5", result="FAIL")
    status, out = _run(
        [
            "tests",
            "--results",
            str(tmp_path),
            "--requested",
            json.dumps([_entry("rpi5")]),
        ],
        capsys,
    )
    assert status == 1
    assert "FAIL" in out


def test_tests_board_without_artifact_fails_the_run(tmp_path, capsys):
    _test_result(tmp_path, "rpi5")
    status, out = _run(
        [
            "tests",
            "--results",
            str(tmp_path),
            "--requested",
            json.dumps([_entry("rpi5"), _entry("orin-nano")]),
        ],
        capsys,
    )
    assert status == 1
    assert "`orin-nano` | — | NO RESULT" in out


def test_unavailable_board_is_reported_but_not_a_failure(tmp_path, capsys):
    _test_result(tmp_path, "rpi5")
    unavailable = [
        {"runner": "imx95-phytec", "reason": "no online runner carries imx95-phytec"}
    ]
    status, out = _run(
        [
            "tests",
            "--results",
            str(tmp_path),
            "--requested",
            json.dumps([_entry("rpi5")]),
            "--unavailable",
            json.dumps(unavailable),
        ],
        capsys,
    )
    assert status == 0
    assert "UNAVAILABLE" in out
    assert "no online runner carries imx95-phytec" in out


def test_nothing_scheduled_fails(tmp_path, capsys):
    unavailable = [{"runner": "rpi5", "reason": "no online runner carries rpi5"}]
    status, out = _run(
        [
            "tests",
            "--results",
            str(tmp_path),
            "--requested",
            "[]",
            "--unavailable",
            json.dumps(unavailable),
        ],
        capsys,
    )
    assert status == 1
    assert "nothing was tested" in out


def test_label_set_entry_escapes_nothing_it_should_not(tmp_path, capsys):
    _test_result(tmp_path, "imx95-frdm+ara240", runner="imx95-frdm")
    status, out = _run(
        [
            "tests",
            "--results",
            str(tmp_path),
            "--requested",
            json.dumps([_entry("imx95-frdm+ara240")]),
        ],
        capsys,
    )
    assert status == 0
    assert "`imx95-frdm+ara240`" in out


def _bench_result(root, entry, runner, cases, medians, failed_case=None):
    d = root / f"hardware-bench-{entry}"
    d.mkdir(parents=True)
    (d / "summary.json").write_text(
        json.dumps(
            {
                "runner": runner,
                "cases": [
                    {
                        "case": c,
                        "status": "failed" if c == failed_case else "ok",
                        "exit": 1 if c == failed_case else 0,
                        "seconds": 30,
                    }
                    for c in cases
                ],
            }
        )
    )
    (d / "system.txt").write_text(
        "host=board\ncpu=Cortex-A78C\ncpus=8\ngpu=Adreno663v1\n"
    )
    for case in cases:
        # The harness's --json shape, including a name the harness repeats.
        benches = [
            {
                "name": n,
                "median_us": m,
                "p95_us": m * 2,
                "min_us": m,
                "max_us": m * 3,
                "mean_us": m,
                "p99_us": m * 3,
                "iterations": 100,
            }
            for n, m in medians.get(case, [])
        ]
        (d / f"{case}.json").write_text(json.dumps({"benchmarks": benches}))


def test_bench_headline_and_all_cases_across_boards(tmp_path, capsys):
    gl = [
        ("letterbox/1920x1080/YUYV->640x640/RGBA", 3029),
        ("letterbox/1920x1080/YUYV->640x640/RGB", 3029),
        ("letterbox/1920x1080/YUYV->640x640/RGB", 3171),
    ]
    _bench_result(
        tmp_path,
        "iq9075-evk",
        "iq9075-evk",
        ["pipeline-opengl"],
        {"pipeline-opengl": gl},
    )
    _bench_result(
        tmp_path,
        "imx95-evk",
        "imx95-evk",
        ["pipeline-opengl"],
        {"pipeline-opengl": [("letterbox/1920x1080/YUYV->640x640/RGBA", 1200)]},
    )
    out_csv = tmp_path / "benchmarks.csv"
    status, out = _run(
        [
            "bench",
            "--results",
            str(tmp_path),
            "--requested",
            json.dumps([_entry("iq9075-evk"), _entry("imx95-evk")]),
            "--csv",
            str(out_csv),
        ],
        capsys,
    )
    assert status == 0
    assert "| GL letterbox 1080p YUYV→RGBA | 3.03 ms | 1.20 ms |" in out
    # A repeated harness name stays two rows rather than one overwriting the other.
    assert "`letterbox/1920x1080/YUYV->640x640/RGB #2`" in out
    assert "Cortex-A78C (8)" in out
    assert "Adreno663v1" in out
    rows = list(csv.DictReader(out_csv.open()))
    assert len(rows) == 4
    assert {r["runner"] for r in rows} == {"iq9075-evk", "imx95-evk"}


def test_bench_failed_case_fails_the_run(tmp_path, capsys):
    _bench_result(
        tmp_path,
        "rpi5",
        "rpi5-hailo8l",
        ["tensor", "pipeline-opengl"],
        {"tensor": [("alloc/dma/u8/1080p", 854)]},
        failed_case="pipeline-opengl",
    )
    status, out = _run(
        [
            "bench",
            "--results",
            str(tmp_path),
            "--requested",
            json.dumps([_entry("rpi5")]),
        ],
        capsys,
    )
    assert status == 1
    assert "failed: pipeline-opengl" in out


def test_bench_board_without_artifact_fails_the_run(tmp_path, capsys):
    status, out = _run(
        [
            "bench",
            "--results",
            str(tmp_path),
            "--requested",
            json.dumps([_entry("orin-nano")]),
        ],
        capsys,
    )
    assert status == 1
    assert "NO RESULT" in out


def test_fmt_us():
    assert fleet_summary.fmt_us(None) == "—"
    assert fleet_summary.fmt_us(854) == "854 µs"
    assert fleet_summary.fmt_us(3029) == "3.03 ms"


def _flatten(root, artifact):
    """What download-artifact does with a pattern's only match."""
    src = root / artifact
    for f in src.iterdir():
        f.rename(root / f.name)
    src.rmdir()


def test_single_artifact_extracted_flat_is_found_by_entry(tmp_path, capsys):
    _test_result(tmp_path, "iq9075-evk", runner="iq9075-evk")
    summary = tmp_path / "hardware-test-iq9075-evk" / "summary.json"
    data = json.loads(summary.read_text())
    data["entry"] = "iq9075-evk"
    summary.write_text(json.dumps(data))
    _flatten(tmp_path, "hardware-test-iq9075-evk")
    unavailable = [
        {"runner": "no-such-board", "reason": "no online runner carries no-such-board"}
    ]
    status, out = _run(
        [
            "tests",
            "--results",
            str(tmp_path),
            "--requested",
            json.dumps([_entry("iq9075-evk")]),
            "--unavailable",
            json.dumps(unavailable),
        ],
        capsys,
    )
    assert status == 0
    assert "✅ PASS" in out
    assert "UNAVAILABLE" in out


def test_flat_result_does_not_answer_for_another_board(tmp_path, capsys):
    _test_result(tmp_path, "rpi5", runner="rpi5-hailo8l")
    summary = tmp_path / "hardware-test-rpi5" / "summary.json"
    data = json.loads(summary.read_text())
    data["entry"] = "rpi5"
    summary.write_text(json.dumps(data))
    _flatten(tmp_path, "hardware-test-rpi5")
    status, out = _run(
        [
            "tests",
            "--results",
            str(tmp_path),
            "--requested",
            json.dumps([_entry("rpi5"), _entry("orin-nano")]),
        ],
        capsys,
    )
    assert status == 1
    assert "`orin-nano` | — | NO RESULT" in out


def test_bench_single_artifact_extracted_flat(tmp_path, capsys):
    _bench_result(
        tmp_path,
        "iq9075-evk",
        "iq9075-evk",
        ["tensor"],
        {"tensor": [("alloc/dma/u8/1080p", 862)]},
    )
    summary = tmp_path / "hardware-bench-iq9075-evk" / "summary.json"
    data = json.loads(summary.read_text())
    data["entry"] = "iq9075-evk"
    summary.write_text(json.dumps(data))
    _flatten(tmp_path, "hardware-bench-iq9075-evk")
    status, out = _run(
        [
            "bench",
            "--results",
            str(tmp_path),
            "--requested",
            json.dumps([_entry("iq9075-evk")]),
        ],
        capsys,
    )
    assert status == 0
    assert "| DMA alloc 1080p | 862 µs |" in out
