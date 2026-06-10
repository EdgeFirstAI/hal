#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
"""Compare edgefirst-bench --json outputs for refactor perf gates.

Each side is one or more JSON files produced by a benchmark binary's
``--json`` flag (repetitions of the same benchmark). Per benchmark case the
median-of-medians (and p95) across repetitions is compared and the ratio
reported. Exit code is non-zero when any case regresses past the gate
thresholds, so this can run directly in CI or a pre-merge checklist.

Gates (from the GL convergence refactor plan):
  * median: >3% regression fails
  * p95:    >5% regression fails

Usage:
  bench_compare.py --before matrix.r1.json matrix.r2.json matrix.r3.json \
                   --after  new.r1.json new.r2.json new.r3.json \
                   [--median-gate 3.0] [--p95-gate 5.0]

Cases present on only one side are listed as added/removed but never fail
the gate (cells legitimately appear when a refactor adds coverage, e.g.
macOS letterbox cells after the platform convergence).
"""

import argparse
import json
import statistics
import sys


def load_side(paths):
    """name -> {"median_us": [...], "p95_us": [...]} across repetitions."""
    cases = {}
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        for bench in data.get("benchmarks", []):
            entry = cases.setdefault(bench["name"], {"median_us": [], "p95_us": []})
            entry["median_us"].append(bench["median_us"])
            entry["p95_us"].append(bench["p95_us"])
    return cases


def rep_median(values):
    return statistics.median(values) if values else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", nargs="+", required=True, help="baseline JSON file(s)")
    ap.add_argument("--after", nargs="+", required=True, help="candidate JSON file(s)")
    ap.add_argument("--median-gate", type=float, default=3.0,
                    help="max allowed median regression in percent (default 3)")
    ap.add_argument("--p95-gate", type=float, default=5.0,
                    help="max allowed p95 regression in percent (default 5)")
    args = ap.parse_args()

    before = load_side(args.before)
    after = load_side(args.after)

    common = sorted(set(before) & set(after))
    added = sorted(set(after) - set(before))
    removed = sorted(set(before) - set(after))

    failures = []
    improvements = []
    width = max((len(n) for n in common), default=20)

    print(f"{'case':<{width}}  {'before med':>10}  {'after med':>10}  "
          f"{'Δmed':>7}  {'Δp95':>7}")
    for name in common:
        b_med = rep_median(before[name]["median_us"])
        a_med = rep_median(after[name]["median_us"])
        b_p95 = rep_median(before[name]["p95_us"])
        a_p95 = rep_median(after[name]["p95_us"])
        if not b_med or not b_p95:
            continue
        d_med = (a_med - b_med) / b_med * 100.0
        d_p95 = (a_p95 - b_p95) / b_p95 * 100.0
        flag = ""
        if d_med > args.median_gate or d_p95 > args.p95_gate:
            flag = "  << REGRESSION"
            failures.append((name, d_med, d_p95))
        elif d_med < -args.median_gate:
            flag = "  (improved)"
            improvements.append((name, d_med))
        print(f"{name:<{width}}  {b_med:>9}µ  {a_med:>9}µ  "
              f"{d_med:>+6.1f}%  {d_p95:>+6.1f}%{flag}")

    for name in added:
        print(f"{name:<{width}}  [new case — no baseline]")
    for name in removed:
        print(f"{name:<{width}}  [removed — present only in baseline]")

    print(f"\n{len(common)} compared, {len(improvements)} improved, "
          f"{len(failures)} regressed, {len(added)} added, {len(removed)} removed")
    if failures:
        print(f"\nGATE FAILED (median >{args.median_gate}% or p95 >{args.p95_gate}%):")
        for name, d_med, d_p95 in failures:
            print(f"  {name}: median {d_med:+.1f}%, p95 {d_p95:+.1f}%")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
