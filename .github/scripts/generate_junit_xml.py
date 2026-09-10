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


def generate_junit_xml(results_dir: str, output_file: str, min_tests: int = 0) -> int:
    """Parse Rust test output files and generate JUnit XML.

    Args:
        results_dir: Directory containing test output .txt files
        output_file: Path to write the JUnit XML output
        min_tests: Minimum number of tests the generated XML must contain.
            When the parsed test count is below this floor, an
            `::error::` is printed and a non-zero status is returned,
            after the XML has still been written.

    Returns:
        0 on success, 1 if the test count is below `min_tests`.
    """
    tests = []
    total_passed = 0
    total_failed = 0

    if not os.path.exists(results_dir):
        print(f"No results directory: {results_dir}")
        # Create empty JUnit XML
        with open(output_file, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write('<testsuites name="rust-hardware" tests="0" failures="0" errors="0"></testsuites>\n')
        if 0 < min_tests:
            print(f"::error::junit has 0 tests, floor is {min_tests}")
            return 1
        return 0

    for txt_file in glob.glob(f"{results_dir}/*.txt"):
        test_binary = os.path.basename(txt_file).replace('.txt', '')
        with open(txt_file, 'r') as f:
            content = f.read()

        # Parse test result line: "test result: ok. X passed; Y failed; Z ignored"
        match = re.search(
            r'test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored',
            content
        )
        if match:
            passed = int(match.group(2))
            failed = int(match.group(3))
            total_passed += passed
            total_failed += failed

            # Parse individual test lines: "test name ... ok/FAILED"
            for test_match in re.finditer(
                r'test ([\w:]+) \.\.\. (ok|FAILED|ignored)',
                content
            ):
                test_name = test_match.group(1)
                test_status = test_match.group(2)
                tests.append({
                    'name': test_name,
                    'classname': test_binary,
                    'status': test_status,
                    'time': '0.001'
                })

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

    print(f"Generated {output_file} with {len(tests)} tests "
          f"({total_passed} passed, {total_failed} failed)")

    if len(tests) < min_tests:
        print(f"::error::junit has {len(tests)} tests, floor is {min_tests}")
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
        help="Minimum number of tests the generated XML must contain "
             "(default: 0, i.e. no floor)",
    )
    args = parser.parse_args()

    return generate_junit_xml(args.results_dir, args.output_file, args.min_tests)


if __name__ == "__main__":
    sys.exit(main())
