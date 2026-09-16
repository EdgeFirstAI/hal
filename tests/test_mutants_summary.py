# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Tests for the mutation-testing roll-up report.

The roll-up reads every shard's ``mutants.out/outcomes.json`` and renders one
markdown summary for the GitHub Actions step summary. These tests pin the two
things the nightly lane depends on: the counts are the sum of the shards, and a
surviving mutant is reported with the diff that no test noticed.
"""

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1] / ".github" / "scripts" / "mutants_summary.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("mutants_summary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mutants_summary = _load()


SUMMARY_FOR = {
    "caught": "CaughtMutant",
    "missed": "MissedMutant",
    "timeout": "Timeout",
    "unviable": "Unviable",
}


def mutant(outcome, file, function, line, name, diff=None):
    """Build one outcome record in cargo-mutants' outcomes.json shape."""
    slug = file.replace("/", "__") + f"_line_{line}"
    return {
        "scenario": {
            "Mutant": {
                "package": "edgefirst-decoder",
                "file": file,
                "function": {"function_name": function, "return_type": ""},
                "genre": "BinaryOperator",
                "name": name,
                "span": {"start": {"line": line, "column": 5}},
            }
        },
        "summary": SUMMARY_FOR[outcome],
        "diff_path": f"diff/{slug}.diff" if diff is not None else None,
        "_diff_body": diff,
    }


def write_shard(root, index, outcomes, write_json=True):
    """Materialise one shard artifact directory and return its mutants.out."""
    out = root / f"mutants-shard-{index}" / "mutants.out"
    (out / "diff").mkdir(parents=True, exist_ok=True)

    tallies = {key: [] for key in SUMMARY_FOR}
    records = []
    for record in outcomes:
        record = dict(record)
        body = record.pop("_diff_body", None)
        if body is not None:
            (out / record["diff_path"]).write_text(body, encoding="utf-8")
        mutant_name = record["scenario"]["Mutant"]["name"]
        for key, summary in SUMMARY_FOR.items():
            if record["summary"] == summary:
                tallies[key].append(mutant_name)
        records.append(record)

    for key, names in tallies.items():
        text = "".join(line + "\n" for line in names)
        (out / f"{key}.txt").write_text(text, encoding="utf-8")

    if write_json:
        payload = {key: len(names) for key, names in tallies.items()}
        payload["total_mutants"] = len(records)
        payload["outcomes"] = [
            {"scenario": "Baseline", "summary": "Success", "diff_path": None}
        ] + records
        (out / "outcomes.json").write_text(json.dumps(payload), encoding="utf-8")

    return out


def render(tmp_path, *args):
    """Run the report and return (exit code, markdown)."""
    report = tmp_path / "report.md"
    code = mutants_summary.main(
        ["--shards", str(tmp_path / "shards"), "--output", str(report)] + list(args)
    )
    return code, report.read_text(encoding="utf-8") if report.exists() else ""


@pytest.fixture
def shards(tmp_path):
    root = tmp_path / "shards"
    root.mkdir()
    return root


def test_totals_are_the_sum_across_shards(tmp_path, shards):
    write_shard(
        shards,
        0,
        [
            mutant("caught", "crates/decoder/src/lib.rs", "arg_max", 580, "m0"),
            mutant("caught", "crates/decoder/src/lib.rs", "arg_max", 581, "m1"),
            mutant("unviable", "crates/decoder/src/lib.rs", "arg_max", 582, "m2"),
        ],
    )
    write_shard(
        shards,
        1,
        [
            mutant("caught", "crates/tracker/src/lib.rs", "step", 10, "m3"),
            mutant("timeout", "crates/tracker/src/lib.rs", "step", 11, "m4"),
        ],
    )

    _, md = render(tmp_path)

    assert "4 mutants tested" in md
    assert "3 caught" in md
    assert "| unviable | 1 |" in md
    assert "| timeout | 1 |" in md
    assert "2 shards" in md


def test_pie_chart_carries_the_outcome_counts(tmp_path, shards):
    write_shard(
        shards,
        0,
        [
            mutant("caught", "crates/decoder/src/lib.rs", "arg_max", 580, "m0"),
            mutant("caught", "crates/decoder/src/lib.rs", "arg_max", 581, "m1"),
            mutant("missed", "crates/decoder/src/lib.rs", "arg_max", 582, "m2"),
        ],
    )

    _, md = render(tmp_path)

    assert "```mermaid" in md
    pie = md.split("```mermaid", 1)[1].split("```", 1)[0]
    assert "pie" in pie
    assert '"Caught" : 2' in pie
    assert '"Survived" : 1' in pie


def test_surviving_mutant_reports_the_diff_no_test_noticed(tmp_path, shards):
    diff = (
        "--- crates/decoder/src/lib.rs\n"
        "+++ replace arg_max_i8 -> (i8, usize) with (0, 0)\n"
        "@@ -595,6 +595,2 @@\n"
        "-    let mut best = i8::MIN;\n"
        "+    (0, 0) /* ~ changed by cargo-mutants ~ */\n"
    )
    write_shard(
        shards,
        0,
        [
            mutant(
                "missed",
                "crates/decoder/src/lib.rs",
                "arg_max_i8",
                595,
                "crates/decoder/src/lib.rs:595:5: replace arg_max_i8 -> (i8, usize) with (0, 0)",
                diff=diff,
            )
        ],
    )

    _, md = render(tmp_path)

    assert "replace arg_max_i8 -> (i8, usize) with (0, 0)" in md
    assert "<details>" in md
    assert "+    (0, 0) /* ~ changed by cargo-mutants ~ */" in md
    assert "-    let mut best = i8::MIN;" in md


def test_caught_mutant_diff_is_not_dumped_into_the_summary(tmp_path, shards):
    write_shard(
        shards,
        0,
        [
            mutant(
                "caught",
                "crates/decoder/src/lib.rs",
                "arg_max",
                580,
                "caught mutant",
                diff="--- a\n+++ b\n-    unique_caught_marker\n",
            )
        ],
    )

    _, md = render(tmp_path)

    assert "unique_caught_marker" not in md


def test_survivors_are_grouped_by_file_and_function(tmp_path, shards):
    write_shard(
        shards,
        0,
        [
            mutant(
                "missed", "crates/decoder/src/lib.rs", "arg_max", 580, "a", diff="-x\n"
            ),
            mutant(
                "missed", "crates/decoder/src/lib.rs", "arg_max", 581, "b", diff="-y\n"
            ),
            mutant(
                "missed", "crates/decoder/src/byte.rs", "decode", 10, "c", diff="-z\n"
            ),
        ],
    )

    _, md = render(tmp_path)

    assert md.count("crates/decoder/src/lib.rs") >= 1
    assert "`arg_max`" in md
    assert "`decode`" in md
    # arg_max holds two survivors and must say so rather than repeat the heading.
    assert "2 survivors" in md
    assert md.count("**`arg_max`**") == 1


def test_gap_ranking_puts_the_worst_file_first(tmp_path, shards):
    write_shard(
        shards,
        0,
        [
            mutant(
                "missed",
                "crates/decoder/src/byte.rs",
                "decode",
                i,
                f"b{i}",
                diff="-x\n",
            )
            for i in range(5)
        ]
        + [
            mutant(
                "missed", "crates/decoder/src/lib.rs", "arg_max", 1, "l1", diff="-x\n"
            )
        ],
    )

    _, md = render(tmp_path)

    ranking = md.split("Where the gaps are", 1)[1]
    assert ranking.index("byte.rs") < ranking.index("lib.rs")


def test_exit_code_is_one_when_mutants_survive(tmp_path, shards):
    write_shard(
        shards,
        0,
        [
            mutant("caught", "crates/decoder/src/lib.rs", "arg_max", 580, "a"),
            mutant(
                "missed", "crates/decoder/src/lib.rs", "arg_max", 581, "b", diff="-x\n"
            ),
        ],
    )

    code, _ = render(tmp_path)

    assert code == 1


def test_exit_code_is_zero_when_nothing_survives(tmp_path, shards):
    write_shard(
        shards,
        0,
        [
            mutant("caught", "crates/decoder/src/lib.rs", "arg_max", 580, "a"),
            mutant("unviable", "crates/decoder/src/lib.rs", "arg_max", 581, "b"),
        ],
    )

    code, md = render(tmp_path)

    assert code == 0
    assert "survived" in md


def test_exit_code_is_two_when_no_shard_produced_a_report(tmp_path, shards):
    code, _ = render(tmp_path)

    assert code == 2


def test_exit_code_is_two_when_every_shard_tested_nothing(tmp_path, shards):
    write_shard(
        shards, 0, [mutant("unviable", "crates/decoder/src/lib.rs", "f", 1, "a")]
    )

    code, _ = render(tmp_path)

    assert code == 2


def test_counts_fall_back_to_text_files_when_outcomes_json_is_missing(tmp_path, shards):
    write_shard(
        shards,
        0,
        [
            mutant("caught", "crates/decoder/src/lib.rs", "arg_max", 580, "a"),
            mutant("missed", "crates/decoder/src/lib.rs", "arg_max", 581, "b"),
        ],
        write_json=False,
    )

    code, md = render(tmp_path)

    assert code == 1
    assert "2 mutants tested" in md
    assert "1 caught" in md


def test_oversized_survivor_set_is_truncated_within_budget(tmp_path, shards):
    big = "".join(f"-    line {i} of a very long diff hunk\n" for i in range(200))
    write_shard(
        shards,
        0,
        [
            mutant(
                "missed",
                "crates/decoder/src/lib.rs",
                f"f{i}",
                i,
                f"mutant {i}",
                diff=big,
            )
            for i in range(60)
        ],
    )

    code, md = render(tmp_path, "--budget", "20000")

    assert code == 1
    assert len(md.encode("utf-8")) <= 20000
    assert "truncated" in md.lower()
    assert "mutants-shard-" in md


def test_survivor_is_labelled_with_the_architecture_it_survived_on(tmp_path, shards):
    write_shard(
        shards,
        "0-ubuntu-24.04",
        [
            mutant(
                "missed",
                "crates/decoder/src/lib.rs",
                "arg_max_i8",
                595,
                "a",
                diff="-x\n",
            )
        ],
    )
    write_shard(
        shards,
        "0-ubuntu-24.04-arm",
        [
            mutant(
                "missed",
                "crates/decoder/src/lib.rs",
                "arg_max_i8",
                595,
                "b",
                diff="-y\n",
            )
        ],
    )

    _, md = render(tmp_path)

    assert "arm64" in md
    assert "x86_64" in md
    assert "2 shards" in md


def test_architecture_is_not_named_when_only_one_was_swept(tmp_path, shards):
    write_shard(
        shards,
        "0-ubuntu-24.04",
        [
            mutant(
                "missed", "crates/decoder/src/lib.rs", "arg_max", 580, "a", diff="-x\n"
            )
        ],
    )

    _, md = render(tmp_path)

    assert "arm64" not in md
    assert "x86_64" not in md


def test_github_actions_mode_appends_to_the_step_summary(tmp_path, shards, monkeypatch):
    write_shard(
        shards,
        0,
        [
            mutant(
                "missed", "crates/decoder/src/lib.rs", "arg_max", 580, "a", diff="-x\n"
            )
        ],
    )
    summary = tmp_path / "step-summary.md"
    summary.write_text("existing content\n", encoding="utf-8")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    code = mutants_summary.main(["--shards", str(shards), "--github-actions"])

    assert code == 1
    written = summary.read_text(encoding="utf-8")
    assert written.startswith("existing content\n")
    assert "Mutation testing" in written
