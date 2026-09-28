#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Au-Zone Technologies. All Rights Reserved.
"""
Roll the per-board results of hardware-test.yml or hardware-bench.yml into one
markdown summary, and gate on it.

Each board job uploads one artifact, downloaded here as one directory per
board: `hardware-test-<entry>/` holding scripts/on-target-run.sh's
`summary.json`, or `hardware-bench-<entry>/` holding scripts/on-target-bench.sh's
`summary.json`, `system.txt` and one `<case>.json` per case. `<entry>` is the
board entry as requested (`imx8mp-evk`, `imx95-frdm+ara240`); the summary
records which runner actually ran it.

Every requested entry gets a row. An entry that produced no artifact (the job
failed before uploading, was cancelled, or timed out) is a failure. An entry
the plan found no online runner for is listed as unavailable and does not fail
the run: it was reported rather than queued, which is the point. A run that
scheduled no board at all tested nothing, and fails.

Usage:
  fleet_summary.py tests --results DIR --requested JSON --unavailable JSON
  fleet_summary.py bench --results DIR --requested JSON --unavailable JSON [--csv PATH]

`--requested` and `--unavailable` are the plan job's `board_matrix` and
`unavailable` outputs. Markdown goes to stdout; the exit status is 1 when any
scheduled board failed or produced nothing.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

# Rows worth reading across boards at a glance: (case, benchmark name, label).
# Everything else is in the per-case sections.
HEADLINE = [
    (
        "pipeline-opengl",
        "letterbox/1920x1080/YUYV->640x640/RGBA",
        "GL letterbox 1080p YUYV→RGBA",
    ),
    (
        "pipeline-opengl",
        "letterbox/3840x2160/NV12->640x640/RGBA",
        "GL letterbox 4K NV12→RGBA",
    ),
    (
        "pipeline-cpu",
        "letterbox/1920x1080/YUYV->640x640/RGBA",
        "CPU letterbox 1080p YUYV→RGBA",
    ),
    (
        "pipeline-cpu",
        "letterbox/3840x2160/NV12->640x640/RGBA",
        "CPU letterbox 4K NV12→RGBA",
    ),
    ("tensor", "alloc/dma/u8/1080p", "DMA alloc 1080p"),
    ("codec", "codec/jpeg/nv12/zidane_720p", "JPEG decode 720p → NV12"),
    ("decoder", "decoder/yolo/quant", "YOLO decode (quantized)"),
    ("mask-opengl", "materialize_masks/scaled_640x640/i8", "Masks scaled 640 i8 (GL)"),
    ("parallel_processors", "parallel/gpu_bound/n1", "GL convert, 1 processor"),
    ("parallel_processors", "parallel/gpu_bound/n2", "GL convert, 2 processors"),
]


def fmt_us(value):
    if value is None:
        return "—"
    if value >= 1000:
        return f"{value / 1000:.2f} ms"
    return f"{value:.0f} µs"


def artifact_dir(results, prefix, entry):
    return Path(results) / f"{prefix}-{entry}"


def load_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def md_escape(text):
    return str(text).replace("|", "\\|")


def summarize_tests(results, requested, unavailable):
    lines = ["## On-hardware tests", ""]
    lines += [
        "| Board | Runner | Result | Passed | Failed | Ignored | Skipped | Detail |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    failed = False
    bundle = None
    for entry in (e["runner"] for e in requested):
        summary = load_json(
            artifact_dir(results, "hardware-test", entry) / "summary.json"
        )
        if summary is None:
            failed = True
            lines.append(
                f"| `{md_escape(entry)}` | — | NO RESULT | | | | | "
                "the board job produced no summary; see its log |"
            )
            continue
        bundle = bundle or summary.get("bundle")
        tests = summary.get("tests", {})
        result = summary.get("result", "?")
        if result != "PASS":
            failed = True
        lines.append(
            f"| `{md_escape(entry)}` | `{md_escape(summary.get('runner') or summary.get('host', '?'))}` "
            f"| {'✅' if result == 'PASS' else '❌'} {result} "
            f"| {tests.get('passed', 0)} | {tests.get('failed', 0)} | {tests.get('ignored', 0)} "
            f"| {tests.get('skipped', 0)} | {md_escape(summary.get('detail', ''))} |"
        )
    for u in unavailable:
        lines.append(
            f"| `{md_escape(u['runner'])}` | — | ⚠️ UNAVAILABLE | | | | | "
            f"{md_escape(u['reason'])} |"
        )
    lines.append("")
    if bundle:
        lines.append(
            f"Bundle: commit `{bundle.get('commit', '?')}`, glibc {bundle.get('glibc', '?')}, "
            f"features `{bundle.get('features') or 'default'}`."
        )
        lines.append("")
    lines.append(
        "A skipped test is not a passed test: each board's artifact has one log per "
        "binary and `capabilities.txt` to attribute skips."
    )
    return "\n".join(lines) + "\n", failed


def read_system(path):
    system = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, _, value = line.partition("=")
            system.setdefault(key, value)
    except OSError:
        pass
    return system


def case_rows(case_json):
    """(name, median_us, p95_us) per benchmark, repeated names disambiguated."""
    data = load_json(case_json) or {}
    seen = {}
    rows = []
    for bench in data.get("benchmarks", []):
        name = bench.get("name", "?")
        seen[name] = seen.get(name, 0) + 1
        if seen[name] > 1:
            name = f"{name} #{seen[name]}"
        rows.append((name, bench.get("median_us"), bench.get("p95_us")))
    return rows


def summarize_bench(results, requested, unavailable, csv_path=None):
    entries = [e["runner"] for e in requested]
    boards = {}
    failed = False
    for entry in entries:
        base = artifact_dir(results, "hardware-bench", entry)
        summary = load_json(base / "summary.json")
        if summary is None:
            failed = True
            boards[entry] = None
            continue
        cases = {c["case"]: c for c in summary.get("cases", [])}
        if any(c["status"] != "ok" for c in cases.values()):
            failed = True
        medians = {}
        for case in cases:
            for name, median, p95 in case_rows(base / f"{case}.json"):
                medians[(case, name)] = (median, p95)
        boards[entry] = {
            "runner": summary.get("runner", ""),
            "system": read_system(base / "system.txt"),
            "cases": cases,
            "medians": medians,
        }

    lines = ["## On-hardware benchmarks", ""]
    lines += [
        "| Board | Runner | CPU | GPU | Cases | Wall time |",
        "|---|---|---|---|---|---:|",
    ]
    for entry in entries:
        board = boards[entry]
        if board is None:
            lines.append(f"| `{md_escape(entry)}` | — | | | ❌ NO RESULT | |")
            continue
        system = board["system"]
        gpu = system.get("gpu") or next(
            (
                v
                for k, v in system.items()
                if k.startswith("devfreq.") and "gpu" in k.lower()
            ),
            "",
        )
        ok = sum(1 for c in board["cases"].values() if c["status"] == "ok")
        bad = [c["case"] for c in board["cases"].values() if c["status"] != "ok"]
        secs = sum(c.get("seconds", 0) for c in board["cases"].values())
        status = f"✅ {ok}" if not bad else f"❌ {ok} ok, failed: {', '.join(bad)}"
        lines.append(
            f"| `{md_escape(entry)}` | `{md_escape(board['runner'] or system.get('host', '?'))}` "
            f"| {md_escape(system.get('cpu', ''))} ({system.get('cpus', '?')}) "
            f"| {md_escape(gpu)} | {status} | {secs // 60}m{secs % 60:02d}s |"
        )
    for u in unavailable:
        lines.append(
            f"| `{md_escape(u['runner'])}` | — | | | ⚠️ UNAVAILABLE: {md_escape(u['reason'])} | |"
        )
    lines.append("")

    measured = [e for e in entries if boards[e]]
    header = (
        "| Benchmark (median) | "
        + " | ".join(f"`{md_escape(e)}`" for e in measured)
        + " |"
    )
    rule = "|---|" + "---:|" * len(measured)
    if measured:
        lines += ["### Headline", "", header, rule]
        for case, name, label in HEADLINE:
            cells = [
                fmt_us((boards[e]["medians"].get((case, name)) or (None,))[0])
                for e in measured
            ]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

        all_cases = []
        for e in measured:
            for case in boards[e]["cases"]:
                if case not in all_cases:
                    all_cases.append(case)
        lines += ["### All cases", ""]
        for case in all_cases:
            names = []
            for e in measured:
                for c, name in boards[e]["medians"]:
                    if c == case and name not in names:
                        names.append(name)
            lines += [
                f"<details><summary><code>{case}</code> ({len(names)} benchmarks)</summary>",
                "",
                header,
                rule,
            ]
            for name in names:
                cells = [
                    fmt_us((boards[e]["medians"].get((case, name)) or (None,))[0])
                    for e in measured
                ]
                lines.append(f"| `{md_escape(name)}` | " + " | ".join(cells) + " |")
            lines += ["", "</details>", ""]

    if csv_path:
        with open(csv_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["board", "runner", "case", "benchmark", "median_us", "p95_us"]
            )
            for e in measured:
                for (case, name), (median, p95) in boards[e]["medians"].items():
                    writer.writerow([e, boards[e]["runner"], case, name, median, p95])

    lines.append(
        "Per-board JSON, logs and `system.txt` (governors, clocks, temperatures) "
        "are in each board's artifact; the `benchmarks.csv` artifact holds every median."
    )
    return "\n".join(lines) + "\n", failed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("mode", choices=["tests", "bench"])
    parser.add_argument("--results", required=True)
    parser.add_argument("--requested", required=True, help="plan board_matrix JSON")
    parser.add_argument("--unavailable", default="[]", help="plan unavailable JSON")
    parser.add_argument("--csv", help="bench: write every median to this CSV")
    args = parser.parse_args(argv)

    requested = json.loads(args.requested or "[]")
    unavailable = json.loads(args.unavailable or "[]")
    if args.mode == "tests":
        text, failed = summarize_tests(args.results, requested, unavailable)
    else:
        text, failed = summarize_bench(args.results, requested, unavailable, args.csv)
    if not requested:
        text += "\n**No board was scheduled, so nothing was tested.**\n"
        failed = True
    sys.stdout.write(text)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
