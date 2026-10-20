#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Au-Zone Technologies. All Rights Reserved.
"""
Collect the `WARNING:` lines tests print, so skips and other warnings are
audited rather than lost in passing-test output.

Tests report a run that could not exercise its subject with
`WARNING: SKIPPED: <reason>` written straight to stderr
(crates/tensor/tests/support/outcome.rs). nextest hides a passing test's
output, so those lines only survive in JUnit `system-out`/`system-err` and in
raw logs (scripts/on-target-run.sh keeps one per test binary). This script
reads both, groups identical warnings, and prints a markdown table. Under
GitHub Actions it also appends the table to the step summary and raises one
annotation per distinct warning, up to a cap.

Usage:
  test_warnings.py PATH [PATH ...]   # files or directories; *.xml is JUnit, anything else is a log

The exit status is always 0: a warning is for auditing, not a failure.
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

WARNING = re.compile(r"WARNING: (.+)")
# GitHub shows at most 10 warning annotations per step; the rest are summarized.
ANNOTATION_CAP = 10


def warnings_in_text(text):
    return [m.group(1).strip() for m in WARNING.finditer(text)]


def warnings_in_junit(path):
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return []
    found = []
    for node in root.iter():
        if node.tag in ("system-out", "system-err") and node.text:
            found.extend(warnings_in_text(node.text))
    return found


def collect(paths):
    counts = Counter()
    for arg in paths:
        base = Path(arg)
        files = sorted(p for p in base.rglob("*") if p.is_file()) if base.is_dir() else [base]
        for f in files:
            if not f.exists():
                continue
            if f.suffix == ".xml":
                counts.update(warnings_in_junit(f))
            else:
                counts.update(warnings_in_text(f.read_text(errors="replace")))
    return counts


def markdown(counts):
    total = sum(counts.values())
    if not total:
        return "### Test warnings\n\nNone.\n"
    lines = [
        "### Test warnings",
        "",
        f"{total} warning(s), {len(counts)} distinct.",
        "",
        "| Count | Warning |",
        "|---:|---|",
    ]
    for message, n in counts.most_common():
        escaped = message.replace("|", "\\|")
        lines.append(f"| {n} | {escaped} |")
    return "\n".join(lines) + "\n"


def annotations(counts):
    out = []
    for message, n in counts.most_common(ANNOTATION_CAP):
        suffix = f" (x{n})" if n > 1 else ""
        out.append(f"::warning title=Test warning::{message}{suffix}")
    rest = len(counts) - ANNOTATION_CAP
    if rest > 0:
        out.append(f"::warning title=Test warnings::{rest} more distinct warning(s) in the step summary")
    return out


def main(argv):
    if not argv:
        print(__doc__.strip(), file=sys.stderr)
        return 0
    counts = collect(argv)
    report = markdown(counts)
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(report)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        for line in annotations(counts):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
