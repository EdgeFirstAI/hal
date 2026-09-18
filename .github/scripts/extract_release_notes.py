#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025 Au-Zone Technologies. All Rights Reserved.
#
# Print the CHANGELOG section for a version, or exit 1 if it has none.
#
# Replaces an awk range that silently produced nothing. `/## \[VERSION\]/,/^## \[/`
# ends on the first line matching the second pattern -- and the heading the
# range starts on matches it too, so the range was always a single record that
# the body then skipped with `next`. Every release since the script was written
# shipped empty notes: v0.32.0's GitHub Release body is zero characters.
#
# Usage:
#   python3 .github/scripts/extract_release_notes.py CHANGELOG.md 0.32.1

from __future__ import annotations

import re
import sys
from pathlib import Path


def extract(text: str, version: str) -> str | None:
    """Return the body of the `## [version]` section, or None if absent."""
    # A candidate has no section of its own; it releases the base version's.
    base = version.split("-rc")[0].split("rc")[0]
    esc = re.escape(base)
    # The version must be a complete token: a prefix match would take the
    # `## [1.2.30]` section for release 1.2.3 and ship another release's notes.
    pattern = rf"^## (?:\[{esc}\]|{esc}\b)[^\n]*\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, text, re.M | re.S)
    return match.group(1).strip() if match else None


def main(changelog: str, version: str) -> int:
    path = Path(changelog)
    if not path.exists():
        print(f"::error::{changelog} not found", file=sys.stderr)
        return 1
    body = extract(path.read_text(encoding="utf-8"), version)
    if not body:
        print(
            f"::error file={changelog}::no '## [{version}]' section, "
            f"or it is empty — the release would ship no notes",
            file=sys.stderr,
        )
        return 1
    print(body)
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: extract_release_notes.py CHANGELOG.md VERSION", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
