# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0

"""Tests for the mutation-testing roll-up report.

The roll-up reads every shard's ``mutants.out/outcomes.json`` and renders one
markdown summary for the GitHub Actions step summary. These tests pin what the
nightly lane depends on: the counts are the sum of the shards, a surviving
mutant is reported with the diff that no test noticed, and a mutant in code a
leg does not compile (cfg'd out, or left Fresh by the build) is classified as
unbuilt rather than counted as a survivor -- while a leg that fails to build a
file its target compiles fails the roll-up.
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


def mutant(
    outcome,
    file,
    function,
    line,
    name,
    diff=None,
    package="edgefirst-decoder",
    col=5,
    log=None,
):
    """Build one outcome record in cargo-mutants' outcomes.json shape.

    `log` is the body of the mutant's log file, written by write_shard.
    """
    slug = file.replace("/", "__") + f"_line_{line}_col_{col}"
    return {
        "scenario": {
            "Mutant": {
                "package": package,
                "file": file,
                "function": {"function_name": function, "return_type": ""},
                "genre": "BinaryOperator",
                "name": name,
                "span": {"start": {"line": line, "column": col}},
            }
        },
        "summary": SUMMARY_FOR[outcome],
        "diff_path": f"diff/{slug}.diff" if diff is not None else None,
        "log_path": f"log/{slug}.log" if log is not None else None,
        "_diff_body": diff,
        "_log_body": log,
    }


def write_shard(root, index, outcomes, write_json=True, baseline=None, marker=None):
    """Materialise one shard artifact directory and return its mutants.out.

    `baseline` is the body of log/baseline.log; `marker` is written as the
    workflow's leg.json beside mutants.out.
    """
    out = root / f"mutants-shard-{index}" / "mutants.out"
    (out / "diff").mkdir(parents=True, exist_ok=True)
    (out / "log").mkdir(parents=True, exist_ok=True)
    if baseline is not None:
        (out / "log" / "baseline.log").write_text(baseline, encoding="utf-8")
    if marker is not None:
        (out.parent / "leg.json").write_text(json.dumps(marker), encoding="utf-8")

    tallies = {key: [] for key in SUMMARY_FOR}
    records = []
    for record in outcomes:
        record = dict(record)
        body = record.pop("_diff_body", None)
        if body is not None:
            (out / record["diff_path"]).write_text(body, encoding="utf-8")
        log = record.pop("_log_body", None)
        if log is not None:
            (out / record["log_path"]).write_text(log, encoding="utf-8")
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
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    code = mutants_summary.main(
        [
            "--shards",
            str(tmp_path / "shards"),
            "--repo",
            str(repo),
            "--output",
            str(report),
        ]
        + list(args)
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

    assert "[linux-arm64] L595" in md
    assert "[linux-x86_64] L595" in md
    assert "2 shards" in md


def test_two_mutants_sharing_a_line_and_description_stay_distinct(tmp_path, shards):
    """cargo-mutants names a mutant `file:line:col: what`, and a line holding
    two of the same operator yields two mutants with the same name. Merging
    them would quietly drop half the corpus."""
    name = "crates/decoder/src/modelpack.rs:409:30: replace * with + in split_float"
    write_shard(
        shards,
        0,
        [
            mutant(
                "missed",
                "crates/decoder/src/modelpack.rs",
                "split_float",
                409,
                name,
                diff="-a\n",
            ),
            mutant(
                "missed",
                "crates/decoder/src/modelpack.rs",
                "split_float",
                409,
                name,
                diff="-b\n",
            ),
        ],
    )

    _, md = render(tmp_path)

    assert "2 mutants tested" in md, md
    assert md.count("```diff") == 2


def test_mutant_caught_on_one_architecture_is_not_a_survivor(tmp_path, shards):
    """The same source line is swept on both architectures. `#[cfg]`-gated code
    is absent from one of the builds, so it cannot be caught there; catching it
    anywhere means the tests do assert on it."""
    name = "crates/decoder/src/lib.rs:595:5: replace arg_max_i8 with (0, 0)"
    write_shard(
        shards,
        "0-ubuntu-24.04",
        [
            mutant(
                "missed",
                "crates/decoder/src/lib.rs",
                "arg_max_i8",
                595,
                name,
                diff="-x\n",
            )
        ],
    )
    write_shard(
        shards,
        "0-ubuntu-24.04-arm",
        [mutant("caught", "crates/decoder/src/lib.rs", "arg_max_i8", 595, name)],
    )

    code, md = render(tmp_path)

    assert code == 0
    assert "No mutant survived" in md
    assert "1 mutant tested" in md, md


def test_mutant_missed_everywhere_is_reported_once_for_all_architectures(
    tmp_path, shards
):
    name = "crates/decoder/src/lib.rs:580:20: replace > with >= in arg_max"
    for suffix in ("0-ubuntu-24.04", "0-ubuntu-24.04-arm"):
        write_shard(
            shards,
            suffix,
            [
                mutant(
                    "missed",
                    "crates/decoder/src/lib.rs",
                    "arg_max",
                    580,
                    name,
                    diff="-x\n",
                )
            ],
        )

    code, md = render(tmp_path)

    assert code == 1
    assert "1 mutant tested" in md, md
    assert md.count("```diff") == 1
    assert "[linux-arm64, linux-x86_64]" in md


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

    assert "[linux-x86_64]" not in md


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

    code = mutants_summary.main(
        ["--shards", str(shards), "--repo", str(tmp_path), "--github-actions"]
    )

    assert code == 1
    written = summary.read_text(encoding="utf-8")
    assert written.startswith("existing content\n")
    assert "Mutation testing" in written


def test_architecture_label_names_only_the_ones_that_ran_it(tmp_path, shards):
    """`unviable` means the mutant never compiled, so that architecture did not
    test it and must not appear in the survivor's label."""
    only_x86 = "crates/decoder/src/lib.rs:100:5: replace + with - in f"
    both = "crates/decoder/src/lib.rs:200:5: replace * with / in g"
    write_shard(
        shards,
        "0-ubuntu-24.04",
        [
            mutant(
                "missed", "crates/decoder/src/lib.rs", "f", 100, only_x86, diff="-x\n"
            ),
            mutant("missed", "crates/decoder/src/lib.rs", "g", 200, both, diff="-y\n"),
        ],
    )
    write_shard(
        shards,
        "0-ubuntu-24.04-arm",
        [
            mutant("unviable", "crates/decoder/src/lib.rs", "f", 100, only_x86),
            mutant("missed", "crates/decoder/src/lib.rs", "g", 200, both, diff="-y\n"),
        ],
    )

    _, md = render(tmp_path)

    assert "[linux-x86_64] L100" in md, md
    assert "[linux-arm64, linux-x86_64] L200" in md, md


def test_gap_ranking_is_by_survival_rate_not_raw_count(tmp_path, shards):
    """A file where everything tested survived is a worse gap than one with more
    survivors but far more coverage."""
    wide = [
        mutant("missed", "crates/decoder/src/byte.rs", "b", i, f"b{i}", diff="-x\n")
        for i in range(5)
    ] + [
        mutant("caught", "crates/decoder/src/byte.rs", "b", 100 + i, f"bc{i}")
        for i in range(95)
    ]
    narrow = [
        mutant("missed", "crates/decoder/src/lib.rs", "l", i, f"l{i}", diff="-x\n")
        for i in range(4)
    ]
    write_shard(shards, 0, wide + narrow)

    _, md = render(tmp_path)

    ranking = md.split("Where the gaps are", 1)[1].split("###", 1)[0]
    assert ranking.index("lib.rs") < ranking.index("byte.rs"), ranking


def test_timeouts_are_not_reported_as_every_mutant_caught(tmp_path, shards):
    write_shard(
        shards,
        0,
        [
            mutant("caught", "crates/decoder/src/lib.rs", "f", 1, "a"),
            mutant("timeout", "crates/decoder/src/lib.rs", "f", 2, "b"),
        ],
    )

    code, md = render(tmp_path)

    assert code == 0
    assert "every tested mutant failed a test" not in md, md
    assert "timed out" in md.lower(), md


# ---------------------------------------------------------------------------
# cfg-aware classification
# ---------------------------------------------------------------------------

LINUX_X86 = mutants_summary.target_for_runner("ubuntu-24.04")
LINUX_ARM = mutants_summary.target_for_runner("ubuntu-24.04-arm")
MACOS = mutants_summary.target_for_runner("macos-latest")
WINDOWS = mutants_summary.target_for_runner("windows-latest")

TENSOR_TOML = """[package]
name = "edgefirst-tensor"

[features]
default = ["ndarray", "static"]
static = []
tracing = []
"""


def write_repo(tmp_path, files):
    """Write source files into the report's --repo checkout."""
    repo = tmp_path / "repo"
    for rel, text in files.items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return repo


def build_log(name, package, state, changed=None):
    """A mutant log in cargo-mutants' layout, trimmed from a real shard.

    `state` is what cargo printed for the mutated package in the build phase:
    "fresh" (nothing rebuilt) or "dirty" (rebuilt because `changed` changed).
    """
    version = f"{package} v0.32.1 (/home/runner/work/hal/hal/crates/tensor)"
    if state == "fresh":
        build = [f"       Fresh {version}"]
    else:
        build = [
            (
                f"       Dirty {version}: the file `{changed}` has changed "
                "(1789880212.226619186s, 6s after last build at 1789880206.573984547s)"
            ),
            f"   Compiling {version}",
            (
                "     Running `rustc --crate-name edgefirst_tensor --edition=2021 "
                "crates/tensor/src/lib.rs`"
            ),
        ]
    lines = [
        f"*** {name}",
        "",
        "*** mutation diff:",
        "--- crates/tensor/src/lib.rs",
        "+++ replace",
        "",
        (
            "*** /home/runner/.rustup/toolchains/1.94.0-x86_64-unknown-linux-gnu/bin/cargo "
            f"test --no-run --verbose --package={package}@0.32.1"
        ),
        "       Fresh cfg-if v1.0.4",
        *build,
        "    Finished `test` profile [unoptimized + debuginfo] target(s) in 0.06s",
        "",
        "*** result: Success",
        "",
        (
            "*** /home/runner/.rustup/toolchains/1.94.0-x86_64-unknown-linux-gnu/bin/cargo "
            f"test --verbose --package={package}@0.32.1 -- --test-threads=1"
        ),
        f"       Fresh {version}",
        "test tests::a ... ok",
        "*** result: Success",
    ]
    return "\n".join(lines) + "\n"


def tensor_mutant(outcome, file, line, what, runner_log=None, col=5, diff="-x\n"):
    name = f"{file}:{line}:{col}: {what}"
    return mutant(
        outcome,
        file,
        "f",
        line,
        name,
        diff=diff if outcome == "missed" else None,
        package="edgefirst-tensor",
        col=col,
        log=runner_log,
    )


def evaluate(expr, target=LINUX_X86, features=("ndarray", "static")):
    return mutants_summary.cfg_eval(expr, target, set(features))


@pytest.mark.parametrize(
    "expr, target, expected",
    [
        ('target_os = "linux"', LINUX_X86, True),
        ('target_os = "linux"', WINDOWS, False),
        ('target_arch = "aarch64"', LINUX_ARM, True),
        ('target_arch = "aarch64"', LINUX_X86, False),
        ('target_arch = "aarch64"', MACOS, True),
        ('target_family = "windows"', WINDOWS, True),
        ('target_env = "msvc"', WINDOWS, True),
        ('target_pointer_width = "64"', LINUX_X86, True),
        ("unix", MACOS, True),
        ("unix", WINDOWS, False),
        ("windows", WINDOWS, True),
        ("test", LINUX_X86, True),
        ("debug_assertions", LINUX_X86, True),
        ('all(unix, not(target_os = "macos"))', LINUX_X86, True),
        ('all(unix, not(target_os = "macos"))', MACOS, False),
        ('any(target_os = "macos", target_os = "ios")', MACOS, True),
        ('any(target_os = "macos", target_os = "ios")', LINUX_X86, False),
        ('all(target_os = "linux", feature = "static")', LINUX_X86, True),
        ('all(target_os = "android", feature = "static")', LINUX_X86, False),
    ],
)
def test_cfg_evaluator_decides_target_predicates(expr, target, expected):
    assert evaluate(expr, target) is expected


def test_cfg_evaluator_reads_features_from_the_leg():
    assert evaluate('feature = "static"') is True
    assert evaluate('feature = "tracing"') is False
    assert evaluate('feature = "tracing"', features=("tracing",)) is True


@pytest.mark.parametrize("flag", ["coverage", "coverage_nightly", "miri", "docsrs"])
def test_cfg_evaluator_treats_tooling_flags_as_off(flag):
    assert evaluate(flag) is False
    assert evaluate(f"not({flag})") is True


def test_cfg_evaluator_leaves_unknown_predicates_undecided():
    """Undecided is treated as built, so an unknown predicate can only keep a
    survivor counted, never hide one."""
    assert evaluate('target_feature = "neon"') is None
    assert evaluate('all(unix, target_feature = "neon")') is None
    assert evaluate('all(windows, target_feature = "neon")') is False
    assert evaluate('any(unix, target_feature = "neon")') is True


def rust(*lines):
    """Join source lines, one argument per line so each can carry its number."""
    return "\n".join(lines)


def regions_false(text, target=LINUX_X86, features=("ndarray", "static")):
    """Lines the scanner considers cfg'd out for a target."""
    source = mutants_summary.SourceFile(text)
    out = []
    for number, line in enumerate(text.split("\n"), start=1):
        if not line.strip():
            continue
        offset = source.offset(number, len(line) - len(line.lstrip()) + 1)
        verdict, _ = source.status_at(offset, target, set(features))
        if verdict is False:
            out.append(number)
    return out


def test_scanner_covers_a_cfg_gated_fn_and_nothing_after_it():
    text = rust(
        '#[cfg(target_os = "windows")]',  # 1
        "fn only_windows() -> u32 {",  # 2
        "    1 + 2",  # 3
        "}",  # 4
        "fn everywhere() -> u32 {",  # 5
        "    3",  # 6
        "}",  # 7
    )
    assert regions_false(text) == [1, 2, 3, 4]
    assert regions_false(text, WINDOWS) == []


def test_scanner_honours_statement_level_cfg():
    text = rust(
        "fn f(x: u32) -> u32 {",  # 1
        '    #[cfg(target_os = "macos")]',  # 2
        "    if x > 1 {",  # 3
        "        return x * 2;",  # 4
        "    } else {",  # 5
        "        return x;",  # 6
        "    }",  # 7
        "    x + 1",  # 8
        "}",  # 9
    )
    assert regions_false(text) == [2, 3, 4, 5, 6, 7]
    assert regions_false(text, MACOS) == []


def test_scanner_ends_a_field_or_arm_at_its_comma():
    text = rust(
        "struct S {",  # 1
        '    #[cfg(target_os = "windows")]',  # 2
        "    handle: HashMap<u32, u32>,",  # 3
        "    size: usize,",  # 4
        "}",  # 5
        "fn g(k: Kind) -> u32 {",  # 6
        "    match k {",  # 7
        '        #[cfg(target_os = "android")]',  # 8
        "        Kind::Ahb => {",  # 9
        "            1",  # 10
        "        }",  # 11
        "        Kind::Mem => 2,",  # 12
        "    }",  # 13
        "}",  # 14
    )
    assert regions_false(text) == [2, 3, 8, 9, 10, 11]


def test_scanner_ignores_cfg_inside_comments_and_strings():
    text = rust(
        '/// Gated like `#[cfg(target_os = "windows")]` elsewhere.',  # 1
        "fn f() -> &'static str {",  # 2
        '    "#[cfg(target_os = \\"windows\\")] {"',  # 3
        "}",  # 4
        "fn g() -> char { '{' }",  # 5
        "fn h() -> u32 {",  # 6
        "    7",  # 7
        "}",  # 8
    )
    assert regions_false(text) == []


def test_scanner_applies_an_inner_cfg_to_the_whole_file():
    text = rust(
        "// SPDX header",  # 1
        '#![cfg(target_arch = "aarch64")]',  # 2
        "fn neon() -> u32 {",  # 3
        "    4",  # 4
        "}",  # 5
    )
    assert regions_false(text, LINUX_X86) == [1, 2, 3, 4, 5]
    assert regions_false(text, LINUX_ARM) == []


def test_module_chain_decides_whether_a_file_is_compiled(tmp_path):
    repo = write_repo(
        tmp_path,
        {
            "crates/tensor/Cargo.toml": TENSOR_TOML,
            "crates/tensor/src/lib.rs": (
                '#[cfg(target_os = "windows")]\npub mod d3d11;\nmod mem;\n'
            ),
            "crates/tensor/src/d3d11/mod.rs": "pub mod adapter;\n",
            "crates/tensor/src/d3d11/adapter.rs": "fn pick() -> u32 { 0 }\n",
            "crates/tensor/src/mem.rs": "fn alloc() -> u32 { 0 }\n",
            "crates/tensor/src/orphan.rs": "fn lost() -> u32 { 0 }\n",
        },
    )
    index = mutants_summary.SourceIndex(repo)
    feats = {"static"}
    adapter = "crates/tensor/src/d3d11/adapter.rs"
    assert index.reachable(adapter, LINUX_X86, feats)[0] is False
    assert index.reachable(adapter, WINDOWS, feats)[0] is True
    assert index.reachable("crates/tensor/src/mem.rs", LINUX_X86, feats)[0] is True
    # A file whose declaration cannot be found is undecided, never cfg'd out.
    assert index.reachable("crates/tensor/src/orphan.rs", LINUX_X86, feats)[0] is None


def cfg_repo(tmp_path):
    return write_repo(
        tmp_path,
        {
            "crates/tensor/Cargo.toml": TENSOR_TOML,
            "crates/tensor/src/lib.rs": rust(
                '#[cfg(target_os = "windows")]',  # 1
                "pub mod d3d11;",  # 2
                "pub fn shared(x: u32) -> u32 {",  # 3
                "    x + 1",  # 4
                "}",  # 5
                '#[cfg(target_os = "macos")]',  # 6
                "fn mac_only(x: u32) -> u32 {",  # 7
                "    x * 2",  # 8
                "}",  # 9
                '#[cfg(feature = "tracing")]',  # 10
                "fn traced(x: u32) -> u32 {",  # 11
                "    x - 1",  # 12
                "}",  # 13
            )
            + "\n",
            "crates/tensor/src/d3d11/mod.rs": "fn pick(x: u32) -> u32 {\n    x\n}\n",
        },
    )


LIB = "crates/tensor/src/lib.rs"
D3D = "crates/tensor/src/d3d11/mod.rs"


def test_mutant_in_code_no_leg_builds_is_unbuilt_not_a_survivor(tmp_path, shards):
    cfg_repo(tmp_path)
    write_shard(
        shards,
        "0-ubuntu-24.04",
        [
            tensor_mutant("missed", LIB, 8, "replace * with + in mac_only"),
            tensor_mutant("caught", LIB, 4, "replace + with - in shared"),
        ],
    )

    code, md = render(tmp_path)

    assert code == 0, md
    assert "| unbuilt | 1 |" in md
    assert "not built on any leg" in md
    assert 'cfg(target_os = "macos")' in md
    assert "survived — no test noticed" not in md


def test_mutant_missed_where_built_and_unbuilt_elsewhere_is_a_survivor(
    tmp_path, shards
):
    cfg_repo(tmp_path)
    what = "replace + with - in shared"
    mac = "replace * with + in mac_only"
    write_shard(
        shards,
        "0-ubuntu-24.04",
        [tensor_mutant("missed", LIB, 4, what), tensor_mutant("missed", LIB, 8, mac)],
    )
    write_shard(
        shards,
        "0-macos-latest",
        [tensor_mutant("missed", LIB, 4, what), tensor_mutant("missed", LIB, 8, mac)],
    )

    code, md = render(tmp_path)

    assert code == 1, md
    assert "2 mutants tested" in md, md
    # The macOS-only mutant is labelled only with the leg that built it.
    assert "[macos-arm64] L8" in md, md
    assert "[linux-x86_64, macos-arm64] L4" in md, md


def test_mutant_caught_on_the_only_leg_that_builds_it_is_caught(tmp_path, shards):
    cfg_repo(tmp_path)
    what = "replace * with + in mac_only"
    write_shard(shards, "0-ubuntu-24.04", [tensor_mutant("missed", LIB, 8, what)])
    write_shard(shards, "0-macos-latest", [tensor_mutant("caught", LIB, 8, what)])

    code, md = render(tmp_path)

    assert code == 0, md
    assert "1 mutant tested" in md
    assert "| unbuilt | 0 |" in md


def test_feature_gated_region_uses_the_features_the_leg_built_with(tmp_path, shards):
    cfg_repo(tmp_path)
    what = "replace - with + in traced"
    write_shard(shards, "0-ubuntu-24.04", [tensor_mutant("missed", LIB, 12, what)])

    code, md = render(tmp_path)
    assert code == 2, md
    assert "| unbuilt | 1 |" in md

    baseline = (
        "     Running `rustc --crate-name edgefirst_tensor --edition=2021 "
        "crates/tensor/src/lib.rs --cfg 'feature=\"default\"' "
        "--cfg 'feature=\"static\"' --cfg 'feature=\"tracing\"'`\n"
    )
    write_shard(
        shards,
        "0-ubuntu-24.04",
        [tensor_mutant("missed", LIB, 12, what)],
        baseline=baseline,
    )

    code, md = render(tmp_path)
    assert code == 1, md
    assert "| unbuilt | 0 |" in md


def test_a_fresh_build_marks_the_mutant_unbuilt(tmp_path, shards):
    """cargo leaving the crate Fresh means the mutated file is not in its
    dependency graph, whatever the source scan could or could not decide."""
    write_repo(tmp_path, {"crates/tensor/Cargo.toml": TENSOR_TOML})
    what = "replace f -> u32 with 0"
    file = "crates/tensor/src/unscanned.rs"
    log = build_log(f"{file}:3:5: {what}", "edgefirst-tensor", "fresh")
    write_shard(
        shards,
        "0-ubuntu-24.04",
        [
            tensor_mutant("missed", file, 3, what, runner_log=log),
            tensor_mutant("caught", file, 9, "replace + with -"),
        ],
    )

    code, md = render(tmp_path)

    assert code == 0, md
    assert "| unbuilt | 1 |" in md
    assert "build left the crate fresh" in md


def test_a_rebuild_caused_by_another_file_does_not_prove_the_mutant_built(
    tmp_path, shards
):
    """The first mutant in a new file rebuilds because the previous file was
    just restored; the module scan still decides that the file is cfg'd out."""
    cfg_repo(tmp_path)
    what = "replace pick -> u32 with 0"
    log = build_log(f"{D3D}:2:5: {what}", "edgefirst-tensor", "dirty", changed=LIB)
    write_shard(
        shards,
        "0-ubuntu-24.04",
        [
            tensor_mutant("missed", D3D, 2, what, runner_log=log),
            tensor_mutant("caught", LIB, 4, "replace + with - in shared"),
        ],
    )

    code, md = render(tmp_path)

    assert code == 0, md
    assert "| unbuilt | 1 |" in md
    assert "`mod d3d11` in crates/tensor/src/lib.rs" in md


def test_a_leg_that_leaves_an_expected_file_unbuilt_fails_the_roll_up(tmp_path, shards):
    """A Windows leg that never compiles d3d11 is misconfigured: its results
    there mean nothing, so the roll-up must not pass."""
    cfg_repo(tmp_path)
    what = "replace pick -> u32 with 0"
    log = build_log(f"{D3D}:2:5: {what}", "edgefirst-tensor", "fresh")
    write_shard(
        shards,
        "0-windows-latest",
        [
            tensor_mutant("missed", D3D, 2, what, runner_log=log),
            tensor_mutant("caught", LIB, 4, "replace + with - in shared"),
        ],
    )

    code, md = render(tmp_path)

    assert code == 1, md
    assert "Broken legs" in md
    assert f"| windows-x86_64 | `{D3D}` | 1 |" in md
    # The mutant itself never counts as a survivor.
    assert "| missed | 0 |" in md


def test_a_fresh_build_where_the_target_does_not_compile_the_file_is_fine(
    tmp_path, shards
):
    cfg_repo(tmp_path)
    what = "replace pick -> u32 with 0"
    log = build_log(f"{D3D}:2:5: {what}", "edgefirst-tensor", "fresh")
    write_shard(
        shards,
        "0-ubuntu-24.04",
        [
            tensor_mutant("missed", D3D, 2, what, runner_log=log),
            tensor_mutant("caught", LIB, 4, "replace + with - in shared"),
        ],
    )

    code, md = render(tmp_path)

    assert code == 0, md
    assert "Broken legs" not in md


@pytest.mark.parametrize(
    "changed",
    [
        D3D,
        D3D.replace("/", "\\"),
        D3D.split("crates/tensor/", 1)[1],
        "/home/runner/work/hal/hal/" + D3D,
    ],
    ids=["workspace-relative", "windows", "crate-relative", "absolute"],
)
def test_a_rebuild_of_the_mutated_file_overrules_the_scan(tmp_path, shards, changed):
    """If cargo rebuilt because this very file changed, the file is compiled
    and the scan was wrong: count the survivor and say so, whichever form
    cargo printed the path in."""
    cfg_repo(tmp_path)
    what = "replace pick -> u32 with 0"
    log = build_log(f"{D3D}:2:5: {what}", "edgefirst-tensor", "dirty", changed=changed)
    write_shard(
        shards,
        "0-ubuntu-24.04",
        [tensor_mutant("missed", D3D, 2, what, runner_log=log)],
    )

    code, md = render(tmp_path)

    assert code == 1, md
    assert "cfg scan called these files cfg'd out" in md


def test_a_rebuild_of_another_file_does_not_overrule_the_scan(tmp_path, shards):
    """cargo naming the previously restored file says nothing about this one."""
    cfg_repo(tmp_path)
    what = "replace pick -> u32 with 0"
    log = build_log(f"{D3D}:2:5: {what}", "edgefirst-tensor", "dirty", changed=LIB)
    write_shard(
        shards,
        "0-ubuntu-24.04",
        [tensor_mutant("missed", D3D, 2, what, runner_log=log)],
    )

    _, md = render(tmp_path)

    assert "| unbuilt | 1 |" in md, md
    assert "cfg scan called these files cfg'd out" not in md


def test_names_same_file_matches_whole_path_components():
    f = "crates/tensor/src/d3d11/mod.rs"
    assert mutants_summary.names_same_file(f, f)
    assert mutants_summary.names_same_file("src/d3d11/mod.rs", f)
    assert mutants_summary.names_same_file("./src/d3d11/mod.rs", f)
    assert mutants_summary.names_same_file("/work/hal/" + f, f)
    assert mutants_summary.names_same_file("crates\\tensor\\src\\d3d11\\mod.rs", f)
    assert not mutants_summary.names_same_file("d3d11/xmod.rs", f)
    assert not mutants_summary.names_same_file("mod.rs.orig", f)
    assert not mutants_summary.names_same_file("", f)
    assert not mutants_summary.names_same_file(None, f)


def _plan(*shards):
    return json.dumps(
        {
            "include": [
                {"corpus": "tensor", "shard": n, "total": 16, "runner": runner}
                for n, runner in shards
            ]
        }
    )


def test_a_scheduled_shard_with_no_report_fails_the_roll_up(tmp_path, shards):
    cfg_repo(tmp_path)
    write_shard(
        shards,
        "tensor-0-ubuntu-24.04",
        [tensor_mutant("caught", LIB, 4, "replace + with - in shared")],
    )
    plan = _plan((0, "ubuntu-24.04"), (0, "windows-latest"))

    code, md = render(tmp_path, "--expected-matrix", plan)

    assert code == 1, md
    assert "### Missing shards" in md
    assert "`mutants-shard-tensor-0-windows-latest`" in md
    assert (
        "tensor-0-ubuntu-24.04`"
        not in md.split("### Missing shards", 1)[1].split("###")[0]
    )


def test_every_scheduled_shard_reporting_leaves_no_missing_section(tmp_path, shards):
    cfg_repo(tmp_path)
    write_shard(
        shards,
        "tensor-0-ubuntu-24.04",
        [tensor_mutant("caught", LIB, 4, "replace + with - in shared")],
    )

    code, md = render(tmp_path, "--expected-matrix", _plan((0, "ubuntu-24.04")))

    assert code == 0, md
    assert "Missing shards" not in md


def test_legs_table_reports_each_legs_dma_state(tmp_path, shards):
    cfg_repo(tmp_path)
    for runner, dma in (
        ("ubuntu-24.04", "required"),
        ("ubuntu-24.04-arm", "unavailable"),
        ("windows-latest", "n/a"),
    ):
        write_shard(
            shards,
            f"0-{runner}",
            [tensor_mutant("caught", LIB, 4, "replace + with - in shared")],
            marker={"runner": runner, "dma": dma},
        )

    code, md = render(tmp_path)

    assert code == 0, md
    legs = md.split("#### Legs", 1)[1]
    assert "| linux-x86_64 | `ubuntu-24.04` | 1 |" in legs
    row = {
        line.split("|")[1].strip(): line for line in legs.splitlines() if "`" in line
    }
    assert row["linux-x86_64"].endswith("| required |")
    assert row["linux-arm64"].endswith("| unavailable |")
    assert row["windows-x86_64"].endswith("| n/a |")


def test_corpus_prefixed_artifact_names_resolve_their_runner():
    match = mutants_summary.SHARD_DIR.match("mutants-shard-tensor-3-windows-latest")
    assert match.group("corpus") == "tensor"
    assert match.group("index") == "3"
    assert match.group("runner") == "windows-latest"
    match = mutants_summary.SHARD_DIR.match("mutants-shard-12-ubuntu-24.04-arm")
    assert match.group("corpus") is None
    assert match.group("runner") == "ubuntu-24.04-arm"


@pytest.mark.parametrize(
    "runner, label",
    [
        ("ubuntu-24.04", "linux-x86_64"),
        ("ubuntu-24.04-arm", "linux-arm64"),
        ("macos-latest", "macos-arm64"),
        ("windows-latest", "windows-x86_64"),
    ],
)
def test_matrix_runners_map_to_their_targets(runner, label):
    assert mutants_summary.target_for_runner(runner).label == label


def test_unknown_runner_is_never_discounted(tmp_path, shards):
    cfg_repo(tmp_path)
    what = "replace * with + in mac_only"
    write_shard(shards, "0-self-hosted-board", [tensor_mutant("missed", LIB, 8, what)])

    code, md = render(tmp_path)

    assert code == 1, md
    assert "| unbuilt | 0 |" in md


def test_windows_backslash_log_paths_are_read(tmp_path, shards):
    """The Windows leg records log paths with backslashes; the roll-up runs on
    Linux and must still read the build phase from them."""
    cfg_repo(tmp_path)
    what = "replace pick -> u32 with 0"
    log = build_log(f"{D3D}:2:5: {what}", "edgefirst-tensor", "fresh")
    record = tensor_mutant("missed", D3D, 2, what, runner_log=log)
    caught = tensor_mutant("caught", LIB, 4, "replace + with - in shared")
    write_shard(shards, "0-windows-latest", [record, caught])
    out = shards / "mutants-shard-0-windows-latest" / "mutants.out"
    outcomes = json.loads((out / "outcomes.json").read_text(encoding="utf-8"))
    for outcome in outcomes["outcomes"][1:]:
        if outcome["log_path"]:
            outcome["log_path"] = outcome["log_path"].replace("/", "\\")
    (out / "outcomes.json").write_text(json.dumps(outcomes), encoding="utf-8")

    code, md = render(tmp_path)

    assert code == 1, md
    assert "Broken legs" in md
