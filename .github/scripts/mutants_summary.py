#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Au-Zone Technologies. All Rights Reserved.
"""
Roll tonight's cargo-mutants shards up into one markdown report.

Each shard uploads a `mutants.out/` directory. The authoritative source is
`outcomes.json`, which carries both the totals and a record per mutant: the
file, the function, the mutation applied, and the path to the diff that was
compiled and tested. The per-outcome `.txt` files carry the same mutants as
bare names and are used only when a shard failed to write its JSON.

A surviving mutant is one the suite passed with. That is usually a line
nothing asserts on, but not always -- a mutation to code the target does not
compile survives for want of ever being built, which is why the sweep runs on
both architectures and the outcomes are merged. Survivors are reported with
the diff nothing noticed, grouped under file and function, so the gap is
readable without downloading an artifact.

Usage:
    python .github/scripts/mutants_summary.py --shards shards/
    python .github/scripts/mutants_summary.py --shards shards/ --github-actions

Exit codes:
    0  every tested mutant was caught
    1  at least one mutant survived
    2  no shard produced a readable report, or no mutant was tested
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# GitHub truncates $GITHUB_STEP_SUMMARY above 1 MiB, and truncation lands
# mid-markdown: the last table or <details> renders as raw text. Staying under
# the cap deliberately keeps the report well formed.
STEP_SUMMARY_LIMIT = 1024 * 1024

# cargo-mutants diffs span the whole mutated function, which for a long
# function buries the one changed line under a screen of context.
DIFF_CONTEXT = 3

SUMMARY_TO_KEY = {
    "CaughtMutant": "caught",
    "MissedMutant": "missed",
    "Timeout": "timeout",
    "Unviable": "unviable",
}

OUTCOME_KEYS = ("caught", "missed", "timeout", "unviable")

# `tested` deliberately excludes unviable: a mutant that did not compile was
# never put to the suite, so counting it would flatter the caught percentage.
TESTED_KEYS = ("caught", "missed", "timeout")

MEANINGS = {
    "caught": "a test failed once the code was broken, so the behaviour is asserted",
    "missed": "the mutation compiled and the whole suite still passed",
    "timeout": "the mutant ran long enough to look like a hang rather than a failure",
    "unviable": "the mutated code did not compile, so it says nothing about the tests",
}

LABELS = {
    "caught": "Caught",
    "missed": "Survived",
    "timeout": "Timed out",
    "unviable": "Unviable",
}

# `crates/decoder/src/lib.rs:595:5: replace arg_max_i8 -> (i8, usize) with (0, 0)`
MUTANT_NAME = re.compile(r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+): (?P<what>.*)$")

TRUNCATION_NOTE = (
    "> Report truncated to fit GitHub's step-summary limit{extra}. "
    "Every survivor and its full diff is in the `mutants-shard-*` artifacts."
)


class Mutant:
    """One mutant's outcome, flattened out of whichever source supplied it."""

    def __init__(self, key, file, function, line, what, diff=None, arch="", col=None):
        self.key = key
        self.file = file
        self.function = function
        self.line = line
        self.col = col
        self.what = what
        self.diff = diff
        # Every architecture that ran this mutant, filled in when the sweeps
        # are merged.
        self.arches = set()
        # Which sweep produced it. The same source line is a different mutant
        # per architecture: a NEON body is compiled on arm64 and absent on
        # x86_64, so a survivor means nothing until you know which ran it.
        self.arch = arch


def parse_name(name):
    """Split a cargo-mutants mutant name into (file, line, description)."""
    match = MUTANT_NAME.match(name)
    if not match:
        return "", None, None, name
    return (
        match.group("file"),
        int(match.group("line")),
        int(match.group("col")),
        match.group("what"),
    )


def arch_of(shard_dir):
    """Name the architecture a shard artifact came from.

    The artifact is named after the runner, so `mutants-shard-3-ubuntu-24.04-arm`
    is the arm64 sweep and anything else is the x86_64 one.
    """
    return "arm64" if shard_dir.name.endswith("-arm") else "x86_64"


def read_outcomes_json(out_dir, arch=""):
    """Read one shard's outcomes.json, or None when it is absent or unusable."""
    path = out_dir / "outcomes.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as err:
        print(f"warning: {path} is unreadable ({err})", file=sys.stderr)
        return None

    mutants = []
    for outcome in payload.get("outcomes", []):
        scenario = outcome.get("scenario")
        # The baseline run is the string "Baseline"; mutants are {"Mutant": {...}}.
        if not isinstance(scenario, dict) or "Mutant" not in scenario:
            continue
        key = SUMMARY_TO_KEY.get(outcome.get("summary"))
        if key is None:
            continue
        mutant = scenario["Mutant"]
        file, line, col, what = parse_name(mutant.get("name", ""))
        span = mutant.get("span") or {}
        start = span.get("start") or {}
        line = start.get("line", line)
        col = start.get("column", col)
        function = (mutant.get("function") or {}).get("function_name", "")
        # Only survivors get their diff read; the caught ones are the bulk of
        # the corpus and nobody needs to see a mutation a test already failed.
        diff = read_diff(out_dir, outcome.get("diff_path")) if key == "missed" else None
        mutants.append(
            Mutant(
                key=key,
                file=mutant.get("file") or file,
                function=function,
                line=line,
                what=what,
                diff=diff,
                arch=arch,
                col=col,
            )
        )
    return mutants


def read_text_files(out_dir, arch=""):
    """Rebuild a shard's outcomes from the per-outcome .txt files."""
    mutants = []
    for key in OUTCOME_KEYS:
        path = out_dir / f"{key}.txt"
        if not path.is_file():
            continue
        for name in path.read_text(encoding="utf-8").splitlines():
            name = name.strip()
            if not name:
                continue
            file, line, col, what = parse_name(name)
            mutants.append(
                Mutant(
                    key=key,
                    file=file,
                    function="",
                    line=line,
                    what=what,
                    arch=arch,
                    col=col,
                )
            )
    return mutants


def read_diff(out_dir, diff_path):
    """Load a mutant's diff, trimmed to the changed lines and their context."""
    if not diff_path:
        return None
    path = out_dir / diff_path
    if not path.is_file():
        return None
    try:
        return focus_diff(path.read_text(encoding="utf-8", errors="replace"))
    except OSError:
        return None


def focus_diff(text):
    """Drop the file header and elide context far from any changed line.

    cargo-mutants emits the whole mutated function, so a one-character change
    inside a long function arrives under dozens of unchanged lines. Keeping
    DIFF_CONTEXT lines either side of each change makes the mutation the thing
    you see first.
    """
    lines = text.splitlines()
    while lines and lines[0].startswith(("--- ", "+++ ")):
        lines.pop(0)

    changed = [
        i
        for i, line in enumerate(lines)
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    if not changed:
        return "\n".join(lines).strip("\n")

    keep = set()
    for i in changed:
        for j in range(max(0, i - DIFF_CONTEXT), min(len(lines), i + DIFF_CONTEXT + 1)):
            keep.add(j)

    out = []
    previous = None
    for i in sorted(keep):
        if previous is not None and i > previous + 1:
            out.append("@@ ...")
        out.append(lines[i])
        previous = i
    return "\n".join(out).strip("\n")


def collect(shards_dir):
    """Read every shard under shards_dir. Returns (mutants, shard count)."""
    mutants = []
    shards = 0
    for out_dir in sorted(Path(shards_dir).glob("mutants-shard-*/mutants.out")):
        if not out_dir.is_dir():
            continue
        shards += 1
        arch = arch_of(out_dir.parent)
        found = read_outcomes_json(out_dir, arch)
        if found is None:
            found = read_text_files(out_dir, arch)
        mutants.extend(found)
    return merge_architectures(mutants), shards


# Best outcome first: one architecture catching a mutant settles it.
OUTCOME_RANK = {"caught": 0, "timeout": 1, "missed": 2, "unviable": 3}


def merge_architectures(mutants):
    """Fold the same mutant seen on several architectures into one.

    Both sweeps mutate the same source, but `#[cfg(target_arch = ...)]` code is
    compiled into only one of the builds, so the other's tests cannot reach it
    and it survives there no matter how good they are. A mutant is therefore
    caught when any architecture's tests caught it, and only counts as having
    survived when every architecture that tested it let it through. Survivors
    carry the set of architectures that ran them, which is how a mutant only
    one build could test is told apart from a plain test gap.
    """
    merged = {}
    order = []
    # A line holding two of the same operator produces two mutants that
    # cargo-mutants names identically, so the name alone cannot pair them up
    # across sweeps. Both sweeps enumerate the same slice in the same order,
    # which makes "the nth mutant with this name" a stable identity.
    seen = {}
    for mutant in mutants:
        base = (mutant.file, mutant.line, mutant.col, mutant.what)
        nth = seen.get((mutant.arch, base), 0)
        seen[(mutant.arch, base)] = nth + 1
        key = base + (nth,)
        if key not in merged:
            merged[key] = mutant
            mutant.arches = set()
            order.append(key)
        best = merged[key]
        if OUTCOME_RANK[mutant.key] < OUTCOME_RANK[best.key]:
            # Keep the decisive outcome, and the diff that came with it.
            mutant.arches = best.arches
            merged[key] = mutant
            best = mutant
        if mutant.arch:
            best.arches.add(mutant.arch)
        if best.diff is None and mutant.diff is not None:
            best.diff = mutant.diff
    return [merged[key] for key in order]


def tally(mutants):
    counts = {key: 0 for key in OUTCOME_KEYS}
    for mutant in mutants:
        counts[mutant.key] += 1
    return counts


def pct(part, whole):
    return "0.0%" if not whole else f"{100.0 * part / whole:.1f}%"


def plural(n, word):
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def group_survivors(mutants):
    """Group survivors by file, then by function, each in first-seen order."""
    files = {}
    file_order = []
    for mutant in mutants:
        if mutant.key != "missed":
            continue
        if mutant.file not in files:
            files[mutant.file] = ({}, [])
            file_order.append(mutant.file)
        functions, function_order = files[mutant.file]
        name = mutant.function or "(unknown function)"
        if name not in functions:
            functions[name] = []
            function_order.append(name)
        functions[name].append(mutant)
    return [
        (file, [(name, files[file][0][name]) for name in files[file][1]])
        for file in file_order
    ]


def render_header(counts, shards, total, start, arches=()):
    tested = sum(counts[key] for key in TESTED_KEYS)
    scope = plural(shards, "shard")
    if total and start is not None:
        scope += f" from index {start} of {total}"
    # Naming the architectures only when both ran keeps the common case quiet
    # and makes the split obvious on the nights it matters.
    if len(arches) > 1:
        scope += f", on {' and '.join(arches)}"

    lines = [f"### Mutation testing — {scope}", ""]
    lines.append(
        f"**{plural(tested, 'mutant')} tested · "
        f"{counts['caught']} caught ({pct(counts['caught'], tested)}) · "
        f"{counts['missed']} survived ({pct(counts['missed'], tested)})**"
    )
    lines.append("")

    slices = [
        f'    "{LABELS[key]}" : {counts[key]}' for key in OUTCOME_KEYS if counts[key]
    ]
    if slices:
        lines.extend(["```mermaid", "pie showData", "    title Mutation outcomes"])
        lines.extend(slices)
        lines.extend(["```", ""])

    lines.append("| outcome | count | share of tested | what it means |")
    lines.append("| --- | ---: | ---: | --- |")
    for key in OUTCOME_KEYS:
        # Unviable mutants are not part of `tested`, so a share of it would be
        # arithmetic on two different denominators.
        share = "—" if key == "unviable" else pct(counts[key], tested)
        lines.append(f"| {key} | {counts[key]} | {share} | {MEANINGS[key]} |")
    lines.append("")
    return lines


def render_gaps(groups, mutants):
    """Rank files by how many mutants survived in them."""
    tested_per_file = {}
    for mutant in mutants:
        if mutant.key in TESTED_KEYS:
            tested_per_file[mutant.file] = tested_per_file.get(mutant.file, 0) + 1

    rows = sorted(
        ((file, sum(len(entries) for _, entries in funcs)) for file, funcs in groups),
        key=lambda row: (-row[1], row[0]),
    )
    lines = ["#### Where the gaps are", ""]
    lines.append("| file | survivors | tested | survival rate |")
    lines.append("| --- | ---: | ---: | ---: |")
    for file, survivors in rows:
        tested = tested_per_file.get(file, survivors)
        lines.append(
            f"| `{file}` | {survivors} | {tested} | {pct(survivors, tested)} |"
        )
    lines.append("")
    return lines


def render_survivors(groups, with_diffs, limit=None, name_arch=False):
    """Render the grouped survivor list. Returns (lines, survivors omitted)."""
    lines = []
    shown = 0
    omitted = 0
    for file, functions in groups:
        pending_file = [f"#### `{file}`", ""]
        for function, entries in functions:
            heading = f"**`{function}`** — {plural(len(entries), 'survivor')}"
            pending_function = [heading, ""]
            for mutant in entries:
                if limit is not None and shown >= limit:
                    omitted += 1
                    continue
                where = f"L{mutant.line} · " if mutant.line else ""
                if name_arch and mutant.arches:
                    where = f"[{', '.join(sorted(mutant.arches))}] {where}"
                if with_diffs and mutant.diff:
                    pending_function.extend(
                        [
                            f"<details><summary>{where}{mutant.what}</summary>",
                            "",
                            "```diff",
                            mutant.diff,
                            "```",
                            "",
                            "</details>",
                            "",
                        ]
                    )
                else:
                    pending_function.append(f"- {where}{mutant.what}")
                shown += 1
            # Two lines means the heading and its blank line, with every mutant
            # under it dropped by the limit; emitting it would be a lie.
            if len(pending_function) > 2:
                if not with_diffs:
                    pending_function.append("")
                pending_file.extend(pending_function)
        if len(pending_file) > 2:
            lines.extend(pending_file)
    return lines, omitted


def render(mutants, shards, total, start, budget):
    counts = tally(mutants)
    groups = group_survivors(mutants)
    arches = sorted({a for m in mutants for a in m.arches})
    name_arch = len(arches) > 1
    header = render_header(counts, shards, total, start, arches)

    if not counts["missed"]:
        clean = "No mutant survived: every tested mutant failed a test."
        return "\n".join(header + [clean]) + "\n"

    header.extend(render_gaps(groups, mutants))
    header.append(
        f"### {plural(counts['missed'], 'mutant')} survived — no test noticed the change"
    )
    header.append("")

    def fits(body, tail):
        text = "\n".join(header + body + tail) + "\n"
        return text if len(text.encode("utf-8")) <= budget else None

    # Prefer the full report, then the same list without diffs, then a list cut
    # to whatever fits. Diffs are the first thing to go because the grouped
    # names still say which functions to look at.
    body, _ = render_survivors(groups, with_diffs=True, name_arch=name_arch)
    text = fits(body, [])
    if text:
        return text

    body, _ = render_survivors(groups, with_diffs=False, name_arch=name_arch)
    text = fits(body, ["", TRUNCATION_NOTE.format(extra="")])
    if text:
        return text

    limit = counts["missed"]
    while limit > 0:
        limit = limit // 2
        body, omitted = render_survivors(
            groups, with_diffs=False, limit=limit, name_arch=name_arch
        )
        extra = f", {plural(omitted, 'survivor')} not listed"
        text = fits(body, ["", TRUNCATION_NOTE.format(extra=extra)])
        if text:
            return text

    # Nothing but the header fits; hand back a clean prefix rather than a
    # half-written table.
    whole = "\n".join(header + [TRUNCATION_NOTE.format(extra="")])
    return whole.encode("utf-8")[:budget].decode("utf-8", "ignore")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards", default="shards", help="directory holding mutants-shard-*/"
    )
    parser.add_argument(
        "--total", default=os.environ.get("TOTAL"), help="shards in the corpus"
    )
    parser.add_argument(
        "--start", default=os.environ.get("START"), help="first shard index"
    )
    parser.add_argument(
        "--output", "-o", help="write the report here instead of stdout"
    )
    parser.add_argument(
        "--github-actions",
        "-g",
        action="store_true",
        help="append to $GITHUB_STEP_SUMMARY",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=STEP_SUMMARY_LIMIT,
        help="maximum report size in bytes",
    )
    args = parser.parse_args(argv)

    mutants, shards = collect(args.shards)
    counts = tally(mutants)
    tested = sum(counts[key] for key in TESTED_KEYS)

    if not shards:
        report = "### Mutation testing\n\nNo shard uploaded a report.\n"
    elif not tested:
        report = "\n".join(render_header(counts, shards, args.total, args.start)) + "\n"
    else:
        report = render(mutants, shards, args.total, args.start, args.budget)

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    elif args.github_actions and os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(report)
    else:
        sys.stdout.write(report)

    # A run that enumerates mutants but tests none is a no-op wearing a green
    # tick, which is how this lane's first execution presented.
    if not shards:
        print("::error::no shard uploaded a mutation report", file=sys.stderr)
        return 2
    if not tested:
        print("::error::no mutants were tested; the report is empty", file=sys.stderr)
        return 2

    print(f"tested={tested} caught={counts['caught']} missed={counts['missed']}")
    if counts["missed"]:
        print(
            f"::error::{plural(counts['missed'], 'mutant')} survived: "
            "the suite passed with the mutation applied",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
