#!/usr/bin/env python3
"""Generate JUnit XML from Rust test output files.

Parses Rust test binary output and generates JUnit XML compatible with
GitHub Actions test result publishers.

Usage:
    python scripts/generate_junit_xml.py build/rust-test-results build/rust_hardware_results.xml
"""

import argparse
import glob
import os
import re
import sys
from datetime import datetime

# libtest's per-binary summary, one per suite run in the file:
#   "test result: ok. 29 passed; 2 failed; 3 ignored; 0 measured; 515 filtered out"
SUMMARY_RE = re.compile(
    r'test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored'
)

# The start of a single test's line: "test some::name ... ".
# `test result: ok.` cannot match: after the name it needs a literal " ... ".
TEST_START_RE = re.compile(r'^test ([\w:]+) \.\.\. ?', re.M)

# A verdict, either immediately after the "... " or at the start of a later
# line. With --test-threads=1 libtest prints "test NAME ... ", then the test
# runs and its own log output lands on the same stream, then the verdict is
# printed. So the verdict can be many lines below the name it belongs to.
VERDICT_RE = re.compile(r'(?:\A|^)(ok|FAILED|ignored)\b', re.M)

# Boundaries that a verdict search must never run past.
STOP_RE = re.compile(r'^(?:test result:|failures:|running \d+ tests?)', re.M)


def _parse_tests(content: str, test_binary: str) -> list:
    """Extract one entry per `test NAME ... VERDICT`, tolerating log noise.

    The naive single-line pattern `test ([\\w:]+) \\.\\.\\. (ok|FAILED|ignored)`
    silently drops every test that logs between its name and its verdict,
    which on the hardware lane is most of the GL suite. Here the name and the
    verdict are matched separately and paired by position, with the search
    for a verdict bounded by the next test's name or the suite summary so a
    test that produces no verdict at all (a process abort mid-run) cannot
    steal the next test's.
    """
    starts = list(TEST_START_RE.finditer(content))
    tests = []
    for i, start in enumerate(starts):
        window_end = starts[i + 1].start() if i + 1 < len(starts) else len(content)
        window = content[start.end():window_end]
        stop = STOP_RE.search(window)
        if stop:
            window = window[:stop.start()]
        verdict = VERDICT_RE.search(window)
        if verdict is None:
            # No verdict before the next test or the summary: the test did
            # not report (e.g. the process died inside it). Counted by the
            # summary line if libtest got far enough to print one; not
            # inventable here, so it is left out of the testcase list rather
            # than guessed at.
            continue
        tests.append({
            'name': start.group(1),
            'classname': test_binary,
            'status': verdict.group(1),
            'time': '0.001',
        })
    return tests


def generate_junit_xml(results_dir: str, output_file: str, min_tests: int = 0) -> int:
    """Parse Rust test output files and generate JUnit XML.

    Args:
        results_dir: Directory containing test output .txt files
        output_file: Path to write the JUnit XML output
        min_tests: Minimum number of executed tests (passed + failed, summed
            from the per-binary `test result:` summary lines) the run must
            report. When the executed count is below this floor, an
            `::error::` is printed and a non-zero status is returned, after
            the XML has still been written.

    Returns:
        0 on success, 1 if the executed count is below `min_tests`.
    """
    tests = []
    total_passed = 0
    total_failed = 0
    total_ignored = 0

    if not os.path.exists(results_dir):
        print(f"No results directory: {results_dir}")
        # Create empty JUnit XML
        with open(output_file, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<testsuites name="rust-hardware" tests="0" failures="0" errors="0"></testsuites>\n')
        if min_tests > 0:
            print(f"::error::junit reports 0 executed tests, floor is {min_tests}")
            return 1
        return 0

    for txt_file in glob.glob(f"{results_dir}/*.txt"):
        test_binary = os.path.basename(txt_file).replace('.txt', '')
        with open(txt_file, 'r') as f:
            content = f.read()

        # Every summary line in the file, not just the first: one output file
        # can hold more than one suite.
        summaries = list(SUMMARY_RE.finditer(content))
        if not summaries:
            continue
        for summary in summaries:
            total_passed += int(summary.group(1))
            total_failed += int(summary.group(2))
            total_ignored += int(summary.group(3))

        tests.extend(_parse_tests(content, test_binary))

    # The authoritative count of tests that actually ran. It comes from
    # libtest's own summary lines, so unlike a per-test scrape it cannot move
    # when a test becomes more or less chatty.
    executed = total_passed + total_failed

    # Generate JUnit XML
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S')
    with open(output_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(f'<testsuites name="rust-hardware" tests="{len(tests)}" '
                f'failures="{total_failed}" errors="0" timestamp="{timestamp}">\n')
        f.write(f'  <testsuite name="rust-hardware" tests="{len(tests)}" '
                f'failures="{total_failed}" errors="0">\n')
        for test in tests:
            f.write(f'    <testcase name="{test["name"]}" '
                    f'classname="{test["classname"]}" time="{test["time"]}"')
            if test['status'] == 'FAILED':
                f.write('>\n      <failure message="Test failed"/>\n    </testcase>\n')
            elif test['status'] == 'ignored':
                f.write('>\n      <skipped/>\n    </testcase>\n')
            else:
                f.write('/>\n')
        f.write('  </testsuite>\n')
        f.write('</testsuites>\n')

    print(f"Generated {output_file} with {executed} executed tests "
          f"({total_passed} passed, {total_failed} failed, "
          f"{total_ignored} ignored), {len(tests)} named test cases")

    # The testcase list is a scrape and the summaries are authoritative, so
    # say so when they disagree instead of letting the smaller number pass
    # for the truth.
    named_expected = executed + total_ignored
    if len(tests) != named_expected:
        print(f"::warning::named {len(tests)} test cases but the summary lines "
              f"report {named_expected} (passed + failed + ignored); the "
              f"executed-count floor uses the summary lines")

    if executed < min_tests:
        print(f"::error::junit reports {executed} executed tests, floor is {min_tests}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate JUnit XML from Rust test output files."
    )
    parser.add_argument("results_dir", help="Directory containing test output .txt files")
    parser.add_argument("output_file", help="Path to write the JUnit XML output")
    parser.add_argument(
        "--min-tests",
        type=int,
        default=0,
        help="Minimum number of executed tests (passed + failed, summed from "
             "the per-binary summary lines) the run must report "
             "(default: 0, i.e. no floor)",
    )
    args = parser.parse_args()

    return generate_junit_xml(args.results_dir, args.output_file, args.min_tests)


if __name__ == "__main__":
    sys.exit(main())
