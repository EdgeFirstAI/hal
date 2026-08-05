#!/usr/bin/env bash
# audit-injection.sh — find GitHub Actions script-injection sites.
#
# Flags `${{ ... }}` expressions that appear INSIDE a `run:` block, which is
# the only place they become shell code. The same expression in `if:`,
# `with:`, or `env:` is evaluated by the Actions expression engine and passed
# as a value — those are safe and are deliberately not reported.
#
# Usage:  audit-injection.sh [repo-root ...]      (default: current dir)
# Exit:   1 if any finding, else 0
set -uo pipefail

python3 - "$@" <<'PY'
import re, sys, os, glob

# Contexts an attacker can set. github.event.* is broadly attacker-influenced;
# these are the ones that carry free-form text.
RISKY = re.compile(r'''github\.(
      head_ref
    | event\.(
          pull_request\.(title|body|head\.(ref|label|repo\.[a-z_.]+))
        | issue\.(title|body)
        | (comment|review|review_comment)\.body
        | commits\.|head_commit\.
        | pages\.
        | client_payload\.
        | discussion\.|workflow_run\.(head_branch|head_commit)
      )
  )''', re.X)

findings = 0
roots = sys.argv[1:] or ["."]
for root in roots:
    for path in sorted(glob.glob(os.path.join(root, ".github/workflows/*.y*ml"))):
        lines = open(path, encoding="utf-8").read().split("\n")
        in_run, run_indent = False, 0
        for n, line in enumerate(lines, 1):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            if in_run and stripped and indent <= run_indent:
                in_run = False
            if re.match(r'-?\s*run:\s*[|>]?', stripped) and stripped.startswith(("run:", "- run:")):
                in_run, run_indent = True, indent
                rest = stripped.split("run:", 1)[1]
                if "${{" in rest and RISKY.search(rest):
                    print(f"{path}:{n}: INJECTION  {stripped[:100]}")
                    findings += 1
                continue
            if in_run and "${{" in line and RISKY.search(line):
                print(f"{path}:{n}: INJECTION  {stripped[:100]}")
                findings += 1

print(f"\n{findings} injection site(s)." if findings else "\nNo injection sites found.")
sys.exit(1 if findings else 0)
PY
