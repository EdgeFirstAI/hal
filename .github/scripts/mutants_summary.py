#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright © 2026 Au-Zone Technologies. All Rights Reserved.
"""
Roll tonight's cargo-mutants shards up into one markdown report and gate on it.

Each shard uploads a `mutants.out/` directory, plus a `leg.json` marker the
workflow writes beside it. The authoritative source is `outcomes.json`, which
carries a record per mutant: the file, the function, the mutation applied, the
diff, and the log of the build and test phases. The per-outcome `.txt` files
carry the same mutants as bare names and are used only when a shard failed to
write its JSON.

cargo-mutants mutates source text and knows nothing about `#[cfg]`. A mutant
in code the leg's target does not compile passes the suite by construction, so
a raw "missed" says nothing about the tests until it is known that the leg
built the mutated code. Every mutant is therefore classified per leg:

* **unbuilt** when the build phase left the mutated crate `Fresh` (the file is
  not in the crate's dependency graph on this target), when the file's module
  declaration chain is cfg'd out for the leg, or when the mutated line lies in
  a `#[cfg(...)]` item, statement or block (or under a file-level
  `#![cfg(...)]`) that evaluates false for the leg's target and features;
* otherwise the outcome cargo-mutants reported.

Legs are then merged by mutant name: caught on any leg wins; built on no leg
is unbuilt (listed per file, never counted); otherwise the mutant is missed
when every leg that built it missed it.

Usage:
    python .github/scripts/mutants_summary.py --shards shards/
    python .github/scripts/mutants_summary.py --shards shards/ --github-actions

Exit codes:
    0  every mutant built on some leg was caught (timeouts are reported, not gated)
    1  at least one legitimate survivor, or a leg left unbuilt a file its
       target is expected to compile (the leg is broken, not the tests weak)
    2  no shard produced a readable report, or no mutant was tested
"""

import argparse
import bisect
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

OUTCOME_KEYS = ("caught", "missed", "timeout", "unviable", "unbuilt")

# `tested` excludes unviable and unbuilt: neither was put to the suite, so
# counting them would flatter the caught percentage.
TESTED_KEYS = ("caught", "missed", "timeout")

MEANINGS = {
    "caught": "a test failed once the code was broken, so the behaviour is asserted",
    "missed": "the mutation compiled and the whole suite still passed",
    "timeout": "the mutant ran long enough to look like a hang rather than a failure",
    "unviable": "the mutated code did not compile, so it says nothing about the tests",
    "unbuilt": "no leg that ran it compiles the mutated code (cfg'd out), so it says nothing about the tests",
}

LABELS = {
    "caught": "Caught",
    "missed": "Survived",
    "timeout": "Timed out",
    "unviable": "Unviable",
    "unbuilt": "Not built",
}

# `crates/decoder/src/lib.rs:595:5: replace arg_max_i8 -> (i8, usize) with (0, 0)`
MUTANT_NAME = re.compile(r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+): (?P<what>.*)$")

# `mutants-shard-7-ubuntu-24.04-arm`, `mutants-shard-tensor-3-windows-latest`.
# The corpus prefix is optional so older artifact names still parse.
SHARD_DIR = re.compile(
    r"^mutants-shard-(?:(?P<corpus>[a-z][a-z0-9_]*)-)?(?P<index>\d+)(?:-(?P<runner>.+))?$"
)

TRUNCATION_NOTE = (
    "> Report truncated to fit GitHub's step-summary limit{extra}. "
    "Every survivor and its full diff is in the `mutants-shard-*` artifacts."
)


# --------------------------------------------------------------------------
# Leg targets
# --------------------------------------------------------------------------


class Target:
    """The cfg environment a leg compiles for."""

    def __init__(self, label, os_, arch, family, env):
        self.label = label
        self.os = os_
        self.arch = arch
        self.family = family
        self.env = env
        self.pointer_width = "64"


def target_for_runner(runner):
    """Map a runner label to its compile target, or None when it is unknown.

    An unknown runner still has its outcomes merged; only the cfg evaluation
    is skipped for it, so its survivors are never discounted by guesswork.
    """
    if not runner:
        return None
    r = runner.lower()
    arm = r.endswith("arm") or "-arm64" in r
    if r.startswith(("ubuntu", "linux")):
        arch = "aarch64" if arm else "x86_64"
        label = "linux-arm64" if arm else "linux-x86_64"
        return Target(label, "linux", arch, "unix", "gnu")
    if r.startswith("macos"):
        # macos-13 and the -intel images are the only x86_64 macOS runners.
        intel = r.startswith("macos-13") or "intel" in r
        arch = "x86_64" if intel else "aarch64"
        label = "macos-x86_64" if intel else "macos-arm64"
        return Target(label, "macos", arch, "unix", "")
    if r.startswith("windows"):
        arch = "aarch64" if arm else "x86_64"
        label = "windows-arm64" if arm else "windows-x86_64"
        return Target(label, "windows", arch, "windows", "msvc")
    return None


class Leg:
    """One runner's sweep: every shard artifact that came from it."""

    def __init__(self, runner):
        self.runner = runner or ""
        self.target = target_for_runner(runner)
        if self.target is not None:
            self.label = self.target.label
        else:
            self.label = runner or ""
        self.shards = 0
        self.dma = {}
        # package -> set of features seen on rustc command lines in baseline.log
        self.features = {}
        self.counts = {key: 0 for key in OUTCOME_KEYS}
        # file -> number of mutants the static scan expected this leg to build
        # but whose build phase left the crate Fresh.
        self.broken = {}
        # file -> number of mutants the static scan called cfg'd out but whose
        # build phase rebuilt the crate for that very file.
        self.scan_disagreements = {}


# --------------------------------------------------------------------------
# cfg evaluation
# --------------------------------------------------------------------------

# Predicates that are false in every build this lane makes.
FALSE_FLAGS = {"coverage", "coverage_nightly", "miri", "docsrs", "doc", "doctest"}
TRUE_FLAGS = {"test", "debug_assertions"}

CFG_TOKEN = re.compile(r'[A-Za-z_][A-Za-z0-9_]*|"(?:[^"\\]|\\.)*"|[(),=]')


def cfg_eval(expr, target, features):
    """Evaluate a cfg predicate: True, False, or None when it cannot be decided.

    None is treated as "built" by every caller, so a predicate this evaluator
    does not understand can only ever keep a survivor counted, never hide one.
    """
    tokens = CFG_TOKEN.findall(expr)
    pos = [0]

    def peek():
        return tokens[pos[0]] if pos[0] < len(tokens) else None

    def take():
        if pos[0] >= len(tokens):
            raise ValueError("unexpected end of cfg")
        tok = tokens[pos[0]]
        pos[0] += 1
        return tok

    def predicate(key, value):
        if key == "feature":
            return value in features
        table = {
            "target_os": target.os,
            "target_arch": target.arch,
            "target_family": target.family,
            "target_env": target.env,
            "target_pointer_width": target.pointer_width,
        }
        if key in table:
            return table[key] == value
        return None

    def flag(name):
        if name == "unix":
            return target.family == "unix"
        if name == "windows":
            return target.family == "windows"
        if name in TRUE_FLAGS:
            return True
        if name in FALSE_FLAGS:
            return False
        return None

    def parse():
        tok = take()
        if tok in ("all", "any", "not") and peek() == "(":
            take()
            args = []
            while peek() != ")":
                args.append(parse())
                if peek() == ",":
                    take()
            take()
            if tok == "not":
                return None if not args or args[0] is None else not args[0]
            if tok == "all":
                if False in args:
                    return False
                return None if None in args else True
            if True in args:
                return True
            return None if None in args else False
        if peek() == "=":
            take()
            value = take().strip('"')
            return predicate(tok, value)
        return flag(tok)

    try:
        return parse()
    except (ValueError, IndexError):
        return None


# --------------------------------------------------------------------------
# Rust source scanning
# --------------------------------------------------------------------------


def mask_source(src):
    """Blank comments and the contents of string/char literals, keeping offsets.

    The result has the same length as `src` and the same newlines, so offsets
    and line numbers carry over. Quote characters are kept, contents are
    replaced with spaces, which keeps `#[cfg(` inside a doc comment or a
    string from being read as an attribute and keeps a `{` inside a string
    from unbalancing the brace matching.
    """
    out = list(src)
    n = len(src)
    i = 0

    def blank(a, b):
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = src[i]
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            j = src.find("\n", i)
            j = n if j == -1 else j
            blank(i, j)
            i = j
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            depth = 1
            j = i + 2
            while j < n and depth:
                if src.startswith("/*", j):
                    depth += 1
                    j += 2
                elif src.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    j += 1
            blank(i, j)
            i = j
            continue
        raw = re.match(r'b?r(#*)"', src[i : i + 260]) if c in "rb" else None
        if raw and (i == 0 or not (src[i - 1].isalnum() or src[i - 1] == "_")):
            hashes = raw.group(1)
            start = i + raw.end()
            end = src.find('"' + hashes, start)
            end = n if end == -1 else end
            blank(start, end)
            i = end + 1 + len(hashes)
            continue
        if c == '"':
            j = i + 1
            while j < n and src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            blank(i + 1, min(j, n))
            i = j + 1
            continue
        if c == "'":
            # A char literal ('x', '\n', '\u{1F600}') or a lifetime ('a).
            if i + 1 < n and src[i + 1] == "\\":
                # Search from past the escaped character so '\'' ends at
                # its last quote.
                j = src.find("'", i + 3)
                if j != -1 and j - i < 14:
                    blank(i + 1, j)
                    i = j + 1
                    continue
            elif i + 2 < n and src[i + 2] == "'":
                blank(i + 1, i + 2)
                i += 3
                continue
            i += 1
            continue
        i += 1
    return "".join(out)


OPENERS = {"(": ")", "[": "]", "{": "}"}
CLOSERS = {")", "]", "}"}

# Items whose body ends the item; commas at depth 0 inside them (generic
# parameter lists) never end it.
BLOCK_ITEMS = {"fn", "impl", "mod", "trait", "struct", "enum", "union", "macro_rules"}
# Items that end at their `;`.
SEMI_ITEMS = {"let", "const", "static", "type", "use", "extern"}
ITEM_PREFIXES = {"pub", "crate", "unsafe", "async", "extern", "const"}

WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\S")


def match_close(masked, start):
    """Offset of the bracket closing the one at `start`, or len(masked)."""
    stack = []
    for j in range(start, len(masked)):
        c = masked[j]
        if c in OPENERS:
            stack.append(OPENERS[c])
        elif c in CLOSERS:
            if not stack:
                return j
            stack.pop()
            if not stack:
                return j
    return len(masked)


def item_kind(masked, start):
    """Classify what an attribute at `start` is attached to.

    Returns "block" for fn/impl/mod/struct-like items (ends at the body),
    "semi" for let/const/use-like items (ends at `;`), and "expr" for
    anything else: a field, variant, match arm, expression or statement.
    """
    j = start
    n = len(masked)
    while True:
        while j < n and masked[j].isspace():
            j += 1
        if masked.startswith("#", j):
            # Another attribute (#[...] or #![...]): skip it.
            k = masked.find("[", j)
            if k == -1:
                return "expr"
            j = match_close(masked, k) + 1
            continue
        m = WORD.match(masked, j)
        if not m:
            return "expr"
        word = m.group(0)
        if word == "pub":
            j = m.end()
            while j < n and masked[j].isspace():
                j += 1
            if masked.startswith("(", j):
                j = match_close(masked, j) + 1
            continue
        if word in BLOCK_ITEMS:
            return "block"
        if word in SEMI_ITEMS and word not in ITEM_PREFIXES:
            return "semi"
        if word in ITEM_PREFIXES:
            # `const fn`, `unsafe fn`, `extern "C" fn` versus `const X: T = ..;`
            # and `extern crate x;`: decide on the next word.
            j = m.end()
            while j < n and masked[j].isspace():
                j += 1
            if masked.startswith('"', j):
                j = masked.find('"', j + 1) + 1
            nxt = WORD.match(masked, j)
            nxt = nxt.group(0) if nxt else ""
            if nxt in BLOCK_ITEMS or nxt in ITEM_PREFIXES:
                continue
            if nxt == "{":
                # `extern "C" { .. }` is an item; `unsafe { .. }`,
                # `async { .. }` and `const { .. }` are block expressions.
                return "block" if word == "extern" else "expr"
            if nxt == "move":
                return "expr"
            return "semi"
        return "expr"


def item_end(masked, start):
    """Offset where the item or expression an attribute at `start` ends."""
    kind = item_kind(masked, start)
    n = len(masked)
    depth = 0
    j = start
    while j < n:
        c = masked[j]
        if c in OPENERS:
            if kind == "block" and c == "{" and depth == 0:
                return match_close(masked, j)
            depth += 1
        elif c in CLOSERS:
            if depth == 0:
                # The enclosing block closed: the attributed element was the
                # last one in it and had no terminator.
                return j - 1
            depth -= 1
            if depth == 0 and c == "}" and kind == "expr":
                # A block-bodied arm, statement or expression. It continues
                # only into `else` or a method call on the block's value.
                k = j + 1
                while k < n and masked[k].isspace():
                    k += 1
                if masked.startswith("else", k) or masked.startswith(".", k):
                    j = k
                    continue
                if masked.startswith(",", k) or masked.startswith(";", k):
                    return k
                return j
        elif depth == 0 and (c == ";" or (c == "," and kind == "expr")):
            return j
        j += 1
    return n


CFG_OUTER = re.compile(r"#\s*\[\s*cfg\s*\(")
CFG_INNER = re.compile(r"#\s*!\s*\[\s*cfg\s*\(")
MOD_DECL = re.compile(r"\bmod\s+(?:r#)?([A-Za-z_][A-Za-z0-9_]*)\s*;")


class SourceFile:
    """A Rust file's cfg regions, with offsets into the original text."""

    def __init__(self, text):
        self.text = text
        self.masked = mask_source(text)
        self.line_starts = [0]
        for m in re.finditer("\n", text):
            self.line_starts.append(m.end())
        self.regions = []
        self._scan()

    def _expr(self, open_paren):
        close = match_close(self.masked, open_paren)
        return self.text[open_paren + 1 : close], close

    def _scan(self):
        masked = self.masked
        # Brace pairs, to find the block an inner attribute applies to.
        pairs = []
        stack = []
        for j, c in enumerate(masked):
            if c == "{":
                stack.append(j)
            elif c == "}" and stack:
                pairs.append((stack.pop(), j))
        for m in CFG_INNER.finditer(masked):
            expr, _ = self._expr(m.end() - 1)
            enclosing = [p for p in pairs if p[0] < m.start() < p[1]]
            if enclosing:
                a, b = max(enclosing, key=lambda p: p[0])
                self.regions.append((a, b, expr, True))
            else:
                self.regions.append((0, len(masked), expr, True))
        for m in CFG_OUTER.finditer(masked):
            expr, close = self._expr(m.end() - 1)
            attr_end = masked.find("]", close) + 1
            if attr_end <= 0:
                continue
            end = item_end(masked, attr_end)
            self.regions.append((m.start(), end, expr, False))

    def offset(self, line, col):
        """Offset of a 1-based line and column (columns counted in chars)."""
        if not line or line < 1 or line > len(self.line_starts):
            return None
        base = self.line_starts[line - 1]
        return base + max((col or 1) - 1, 0)

    def line_of(self, offset):
        return bisect.bisect_right(self.line_starts, offset)

    def status_at(self, offset, target, features):
        """(True|False|None, the deciding cfg expression) for an offset."""
        verdict = True
        unknown = []
        for a, b, expr, _inner in self.regions:
            if a <= offset <= b:
                value = cfg_eval(expr, target, features)
                if value is False:
                    return False, " ".join(expr.split())
                if value is None:
                    verdict = None
                    unknown.append(expr)
        return verdict, "; ".join(" ".join(e.split()) for e in unknown)

    def whole_file_status(self, target, features):
        """The file-level `#![cfg]` verdict, ignoring nested inner attributes."""
        for a, b, expr, inner in self.regions:
            if inner and a == 0 and b == len(self.masked):
                value = cfg_eval(expr, target, features)
                if value is not True:
                    return value, " ".join(expr.split())
        return True, ""

    def mod_decl(self, name):
        for m in MOD_DECL.finditer(self.masked):
            if m.group(1) == name:
                return m.start()
        return None


class SourceIndex:
    """Reads repository sources on demand and answers cfg questions about them."""

    def __init__(self, repo):
        self.repo = Path(repo) if repo else None
        self.files = {}
        self.manifests = {}

    def source(self, rel):
        if rel in self.files:
            return self.files[rel]
        found = None
        if self.repo is not None:
            path = self.repo / rel
            try:
                found = SourceFile(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                found = None
        self.files[rel] = found
        return found

    def crate_dir(self, rel):
        """The nearest ancestor holding a Cargo.toml, relative to the repo."""
        if self.repo is None:
            return None
        parts = Path(rel).parts
        for k in range(len(parts) - 1, 0, -1):
            candidate = Path(*parts[:k])
            if (self.repo / candidate / "Cargo.toml").is_file():
                return candidate
        return None

    def manifest(self, crate):
        """(package name, default features) from a crate's Cargo.toml."""
        if crate in self.manifests:
            return self.manifests[crate]
        name, defaults = None, set()
        try:
            text = (self.repo / crate / "Cargo.toml").read_text(encoding="utf-8")
        except OSError:
            text = ""
        section = None
        for raw in text.splitlines():
            line = raw.split("#", 1)[0].strip()
            header = re.match(r"^\[([^\]]+)\]$", line)
            if header:
                section = header.group(1).strip()
                continue
            if section == "package" and name is None:
                m = re.match(r'^name\s*=\s*"([^"]+)"', line)
                if m:
                    name = m.group(1)
            if section == "features":
                m = re.match(r"^default\s*=\s*\[(.*)\]", line)
                if m:
                    defaults = set(re.findall(r'"([^"/]+)"', m.group(1)))
        self.manifests[crate] = (name, defaults)
        return self.manifests[crate]

    def is_root(self, rel, crate):
        inside = Path(rel).relative_to(crate).parts
        if len(inside) == 1:
            return inside[0] == "build.rs"
        if inside[0] == "src" and len(inside) == 2:
            return inside[1] in ("lib.rs", "main.rs")
        if inside[0] == "src" and len(inside) == 3 and inside[1] == "bin":
            return True
        return inside[0] in ("tests", "benches", "examples") and len(inside) == 2

    def reachable(self, rel, target, features, depth=0):
        """Whether the module chain from the crate root to `rel` is compiled.

        Returns (True|False|None, reason). None means the scan could not tell
        (the file is missing, or its `mod` declaration was not found where
        rustc would look), which callers treat as compiled.
        """
        crate = self.crate_dir(rel)
        if crate is None or self.source(rel) is None or depth > 32:
            return None, ""
        path = Path(rel)
        if self.is_root(rel, crate):
            return True, ""
        if path.name == "mod.rs":
            name, directory = path.parent.name, path.parent.parent
        else:
            name, directory = path.stem, path.parent
        if directory == crate / "src":
            parents = [directory / "lib.rs", directory / "main.rs"]
        else:
            parents = [
                directory.parent / (directory.name + ".rs"),
                directory / "mod.rs",
            ]
        for parent in parents:
            parent_rel = parent.as_posix()
            source = self.source(parent_rel)
            if source is None:
                continue
            at = source.mod_decl(name)
            if at is None:
                continue
            verdict, why = source.status_at(at, target, features)
            if verdict is False:
                return False, f"`mod {name}` in {parent_rel}: cfg({why})"
            whole, whole_why = source.whole_file_status(target, features)
            if whole is False:
                return False, f"{parent_rel}: #![cfg({whole_why})]"
            up, up_why = self.reachable(parent_rel, target, features, depth + 1)
            if up is False:
                return False, up_why
            if verdict is None or up is None or whole is None:
                return None, ""
            return True, ""
        return None, ""


# --------------------------------------------------------------------------
# cargo-mutants logs
# --------------------------------------------------------------------------

LOG_PHASE = re.compile(r"^\*\*\* (?P<cmd>.*)$")
FRESH = re.compile(r"^\s*Fresh (?P<pkg>\S+) v")
DIRTY = re.compile(
    r"^\s*Dirty (?P<pkg>\S+) v[^:]*: the file `(?P<file>[^`]+)` has changed"
)
COMPILING = re.compile(r"^\s*Compiling (?P<pkg>\S+) v")
RUSTC_FEATURES = re.compile(r"--crate-name (?P<crate>\S+) .*")
FEATURE_CFG = re.compile(r"""feature=\\?["'](?P<f>[^"'\\]+)""")


def build_signal(out_dir, log_path, package):
    """What the build phase did to the mutated crate.

    Returns ("fresh", None) when cargo found nothing to rebuild -- the mutated
    file is not in the crate's dependency graph on this target -- or
    ("dirty", file) naming the file cargo said changed, ("compiled", None)
    when it rebuilt without saying why, and (None, None) when the log is
    missing or has no build phase. The dirty file is not always the mutated
    one: when a shard moves on to a new file, cargo-mutants has just restored
    the previous file, and cargo may name that instead.
    """
    if not log_path or not package:
        return None, None
    # Paths recorded on the Windows leg use backslashes; the roll-up runs on
    # Linux.
    path = out_dir / log_path.replace("\\", "/")
    try:
        handle = path.open(encoding="utf-8", errors="replace")
    except OSError:
        return None, None
    in_build = False
    compiled = False
    with handle:
        for line in handle:
            phase = LOG_PHASE.match(line)
            if phase:
                cmd = phase.group("cmd")
                if in_build and cmd.startswith("result"):
                    break
                if "cargo" in cmd and ("--no-run" in cmd or " build" in cmd):
                    in_build = True
                continue
            if not in_build:
                continue
            m = FRESH.match(line)
            if m and m.group("pkg") == package:
                return "fresh", None
            m = DIRTY.match(line)
            if m and m.group("pkg") == package:
                return "dirty", m.group("file").replace("\\", "/")
            m = COMPILING.match(line)
            if m and m.group("pkg") == package:
                compiled = True
    return ("compiled", None) if compiled else (None, None)


def baseline_features(out_dir):
    """package-crate-name -> features rustc was given in the baseline build."""
    found = {}
    path = out_dir / "log" / "baseline.log"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return found
    for line in text.splitlines():
        if "--crate-name" not in line or "Running" not in line:
            continue
        m = RUSTC_FEATURES.search(line)
        if not m:
            continue
        crate = m.group("crate")
        feats = set(FEATURE_CFG.findall(line))
        found.setdefault(crate, set()).update(feats)
    return found


# --------------------------------------------------------------------------
# Reading shards
# --------------------------------------------------------------------------


class Mutant:
    """One mutant's outcome, merged across every leg that ran it."""

    def __init__(
        self, key, file, function, line, what, diff=None, col=None, package=""
    ):
        self.key = key
        self.file = file
        self.function = function
        self.line = line
        self.col = col
        self.what = what
        self.diff = diff
        self.package = package
        # Legs that built and tested it, filled in by the merge.
        self.arches = set()
        # leg label -> why that leg did not build it.
        self.unbuilt_on = {}
        # The leg this record was read from, and where its log lives.
        self.leg = ""
        self.log = (None, None)


def parse_name(name):
    """Split a cargo-mutants mutant name into (file, line, col, description)."""
    match = MUTANT_NAME.match(name)
    if not match:
        return "", None, None, name
    return (
        match.group("file"),
        int(match.group("line")),
        int(match.group("col")),
        match.group("what"),
    )


def read_outcomes_json(out_dir, leg_label=""):
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
        record = scenario["Mutant"]
        file, line, col, what = parse_name(record.get("name", ""))
        span = record.get("span") or {}
        start = span.get("start") or {}
        line = start.get("line", line)
        col = start.get("column", col)
        function = (record.get("function") or {}).get("function_name", "")
        # Only survivors get their diff read; the caught ones are the bulk of
        # the corpus and nobody needs to see a mutation a test already failed.
        diff = read_diff(out_dir, outcome.get("diff_path")) if key == "missed" else None
        mutant = Mutant(
            key=key,
            file=(record.get("file") or file).replace("\\", "/"),
            function=function,
            line=line,
            what=what,
            diff=diff,
            col=col,
            package=record.get("package", ""),
        )
        mutant.leg = leg_label
        mutant.log = (out_dir, outcome.get("log_path"))
        mutants.append(mutant)
    return mutants


def read_text_files(out_dir, leg_label=""):
    """Rebuild a shard's outcomes from the per-outcome .txt files."""
    mutants = []
    for key in ("caught", "missed", "timeout", "unviable"):
        path = out_dir / f"{key}.txt"
        if not path.is_file():
            continue
        for name in path.read_text(encoding="utf-8").splitlines():
            name = name.strip()
            if not name:
                continue
            file, line, col, what = parse_name(name)
            mutant = Mutant(
                key=key,
                file=file.replace("\\", "/"),
                function="",
                line=line,
                what=what,
                col=col,
            )
            mutant.leg = leg_label
            mutants.append(mutant)
    return mutants


def read_diff(out_dir, diff_path):
    """Load a mutant's diff, trimmed to the changed lines and their context."""
    if not diff_path:
        return None
    path = out_dir / diff_path.replace("\\", "/")
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


def read_leg_marker(shard_dir):
    """The workflow's leg.json beside mutants.out, or {}."""
    path = shard_dir / "leg.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def names_same_file(dirty, workspace_path):
    """Whether cargo's `Dirty ...: the file X` names `workspace_path`.

    Cargo prints the path relative to the workspace root on every leg seen so
    far, but the form is cargo's choice: accept it crate-relative
    (`src/lib.rs`) or absolute as well, by matching on whole path components.
    """
    if not dirty:
        return False
    dirty = dirty.replace("\\", "/")
    if dirty.startswith("./"):
        dirty = dirty[2:]
    return (
        dirty == workspace_path
        or dirty.endswith("/" + workspace_path)
        or (not dirty.startswith("/") and workspace_path.endswith("/" + dirty))
    )


def classify(mutant, leg, index):
    """Decide whether `leg` built `mutant`. Returns a reason string when not.

    Also records on the leg any file whose build contradicts the static scan.
    """
    if mutant.key == "unviable":
        # It failed to compile, so the compiler read it.
        return None
    out_dir, log_path = mutant.log
    signal, dirty_file = (None, None)
    if out_dir is not None:
        signal, dirty_file = build_signal(out_dir, log_path, mutant.package)

    target = leg.target
    features = None
    if target is not None and index is not None:
        crate = index.crate_dir(mutant.file)
        crate_name, defaults = (
            index.manifest(crate) if crate is not None else (None, set())
        )
        package = mutant.package or crate_name or ""
        seen = leg.features.get(package.replace("-", "_"))
        features = seen if seen else defaults

    file_ok, file_why = (None, "")
    whole_ok, whole_why = (None, "")
    source = index.source(mutant.file) if index is not None else None
    if target is not None and source is not None:
        file_ok, file_why = index.reachable(mutant.file, target, features)
        whole_ok, whole_why = source.whole_file_status(target, features)

    if signal == "fresh":
        if file_ok is True and whole_ok is True:
            leg.broken[mutant.file] = leg.broken.get(mutant.file, 0) + 1
        return "not compiled on this target (build left the crate fresh)"

    if file_ok is False:
        if signal == "dirty" and names_same_file(dirty_file, mutant.file):
            # cargo rebuilt because this very file changed, so it is in the
            # dependency graph and the scan was wrong. Trust the build.
            leg.scan_disagreements[mutant.file] = (
                leg.scan_disagreements.get(mutant.file, 0) + 1
            )
        else:
            return f"module not compiled: {file_why}"

    if source is not None and target is not None:
        offset = source.offset(mutant.line, mutant.col)
        if offset is not None:
            verdict, why = source.status_at(offset, target, features)
            if verdict is False:
                if whole_ok is False:
                    return f"file cfg'd out: #![cfg({whole_why})]"
                return f"cfg'd out: cfg({why})"
    return None


def collect(shards_dir, repo=None):
    """Read every shard under shards_dir. Returns (mutants, legs, shard count)."""
    index = SourceIndex(repo) if repo is not None else None
    legs = {}
    mutants = []
    shards = 0
    for out_dir in sorted(Path(shards_dir).glob("mutants-shard-*/mutants.out")):
        if not out_dir.is_dir():
            continue
        shards += 1
        shard_dir = out_dir.parent
        match = SHARD_DIR.match(shard_dir.name)
        marker = read_leg_marker(shard_dir)
        runner = marker.get("runner") or (match.group("runner") if match else None)
        leg = legs.get(runner or "")
        if leg is None:
            leg = Leg(runner)
            legs[runner or ""] = leg
        leg.shards += 1
        dma = marker.get("dma")
        if dma:
            leg.dma[dma] = leg.dma.get(dma, 0) + 1
        for crate, feats in baseline_features(out_dir).items():
            leg.features.setdefault(crate, set()).update(feats)

        found = read_outcomes_json(out_dir, leg.label)
        if found is None:
            found = read_text_files(out_dir, leg.label)
        for mutant in found:
            reason = classify(mutant, leg, index)
            if reason is not None:
                mutant.unbuilt_on[leg.label] = reason
                mutant.key = "unbuilt"
            leg.counts[mutant.key] += 1
            mutants.append(mutant)
    return merge_legs(mutants), list(legs.values()), shards


# Best outcome first: one leg catching a mutant settles it. Unbuilt ranks last
# because a leg that never compiled the mutant has no say in its outcome.
OUTCOME_RANK = {"caught": 0, "timeout": 1, "missed": 2, "unviable": 3, "unbuilt": 4}


def merge_legs(mutants):
    """Fold the same mutant seen on several legs into one.

    Every leg mutates the same source, but cfg-gated code is compiled into only
    some of the builds. A mutant is therefore caught when any leg caught it,
    unbuilt when no leg built it, and only counts as having survived when every
    leg that built it let it through. Survivors carry the legs that ran them.
    """
    merged = {}
    order = []
    # A line holding two of the same operator produces two mutants that
    # cargo-mutants names identically, so the name alone cannot pair them up
    # across legs. Every leg enumerates the same corpus in the same order,
    # which makes "the nth mutant with this name" a stable identity.
    seen = {}
    for mutant in mutants:
        base = (mutant.file, mutant.line, mutant.col, mutant.what)
        nth = seen.get((mutant.leg, base), 0)
        seen[(mutant.leg, base)] = nth + 1
        key = base + (nth,)
        if key not in merged:
            merged[key] = mutant
            order.append(key)
            best = mutant
        else:
            best = merged[key]
            if OUTCOME_RANK[mutant.key] < OUTCOME_RANK[best.key]:
                # Keep the decisive outcome, and the diff that came with it.
                mutant.arches |= best.arches
                mutant.unbuilt_on.update(best.unbuilt_on)
                if mutant.diff is None:
                    mutant.diff = best.diff
                merged[key] = mutant
                best = mutant
            else:
                best.unbuilt_on.update(mutant.unbuilt_on)
                if best.diff is None and mutant.diff is not None:
                    best.diff = mutant.diff
        # Only legs that actually ran it. An unviable mutant never compiled
        # and an unbuilt one was never compiled, so naming either leg would
        # claim a build tested a mutation it never produced.
        if mutant.leg and mutant.key in TESTED_KEYS:
            best.arches.add(mutant.leg)
    return [merged[key] for key in order]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


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


def render_header(counts, shards, total, start, legs=()):
    tested = sum(counts[key] for key in TESTED_KEYS)
    scope = plural(shards, "shard")
    if total and start is not None:
        scope += f" from index {start} of {total}"
    labels = [leg.label for leg in legs if leg.label]
    # Naming the legs only when more than one ran keeps a single-leg dispatch
    # quiet and makes the split obvious on the nights it matters.
    if len(labels) > 1:
        scope += f", on {', '.join(sorted(labels))}"

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
        # Unviable and unbuilt mutants are not part of `tested`, so a share of
        # it would be arithmetic on two different denominators.
        share = "—" if key in ("unviable", "unbuilt") else pct(counts[key], tested)
        lines.append(f"| {key} | {counts[key]} | {share} | {MEANINGS[key]} |")
    lines.append("")
    return lines


def render_legs(legs):
    """One row per leg: its target, its raw outcomes, and its DMA state."""
    if not legs or (len(legs) == 1 and not legs[0].runner):
        return []
    lines = ["#### Legs", ""]
    lines.append(
        "| leg | runner | shards | caught | missed | timeout | unviable | unbuilt | DMA |"
    )
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for leg in sorted(legs, key=lambda l: l.label):
        if leg.dma:
            dma = ", ".join(
                f"{state}" if n == leg.shards else f"{state} ({n})"
                for state, n in sorted(leg.dma.items())
            )
        else:
            dma = "—"
        c = leg.counts
        lines.append(
            f"| {leg.label or '—'} | `{leg.runner or '—'}` | {leg.shards} | "
            f"{c['caught']} | {c['missed']} | {c['timeout']} | {c['unviable']} | "
            f"{c['unbuilt']} | {dma} |"
        )
    lines.append("")
    lines.append(
        "Per-leg counts are before merging; a mutant another leg caught still "
        "shows as missed on a leg that could not catch it."
    )
    lines.append("")
    return lines


def missing_shards(shards_dir, expected_matrix):
    """Shards the plan scheduled that left no report.

    `expected_matrix` is the plan job's `{"include": [...]}` matrix. A shard
    that timed out or was cancelled uploads nothing, so without this check
    the night's totals would silently cover fewer mutants than scheduled.
    """
    if not expected_matrix:
        return []
    entries = json.loads(expected_matrix).get("include", [])
    found = {
        path.parent.name
        for path in Path(shards_dir).glob("mutants-shard-*/mutants.out")
        if path.is_dir()
    }
    expected = [
        f"mutants-shard-{e['corpus']}-{e['shard']}-{e['runner']}" for e in entries
    ]
    return sorted(name for name in expected if name not in found)


def render_missing(missing):
    if not missing:
        return []
    lines = ["### Missing shards — scheduled but no report", ""]
    lines.append(
        "These shards were planned tonight but uploaded no `mutants.out` "
        "(timed out, cancelled, or failed before cargo-mutants wrote it). "
        "Their mutants are absent from every total below."
    )
    lines.append("")
    lines += [f"- `{name}`" for name in missing]
    lines.append("")
    return lines


def render_broken(legs):
    """Files a leg should have compiled but left untouched."""
    rows = [
        (leg.label, file, n) for leg in legs for file, n in sorted(leg.broken.items())
    ]
    if not rows:
        return []
    lines = ["### Broken legs — expected files were not built", ""]
    lines.append(
        "These files are compiled for the leg's target by the module tree, but "
        "the build phase left the crate untouched, so every mutant in them "
        "passed without being compiled. The leg is misconfigured; its results "
        "for these files mean nothing."
    )
    lines.append("")
    lines.append("| leg | file | mutants |")
    lines.append("| --- | --- | ---: |")
    for label, file, n in rows:
        lines.append(f"| {label} | `{file}` | {n} |")
    lines.append("")
    return lines


def render_disagreements(legs):
    rows = [
        (leg.label, file, n)
        for leg in legs
        for file, n in sorted(leg.scan_disagreements.items())
    ]
    if not rows:
        return []
    lines = [
        "> The cfg scan called these files cfg'd out, but the build rebuilt the "
        "crate for them, so they were counted as built: "
        + ", ".join(f"`{file}` on {label} ({n})" for label, file, n in rows),
        "",
    ]
    return lines


def render_gaps(groups, mutants):
    """Rank files by how many mutants survived in them."""
    tested_per_file = {}
    for mutant in mutants:
        if mutant.key in TESTED_KEYS:
            tested_per_file[mutant.file] = tested_per_file.get(mutant.file, 0) + 1

    # By rate first: four survivors out of four tested is a worse gap than
    # five out of a hundred, and the count alone puts them the wrong way round.
    # Count breaks ties so the bigger job of the two comes first.
    rows = sorted(
        (
            (file, survivors, tested_per_file.get(file, survivors))
            for file, survivors in (
                (file, sum(len(entries) for _, entries in funcs))
                for file, funcs in groups
            )
        ),
        key=lambda row: (-(row[1] / row[2] if row[2] else 1.0), -row[1], row[0]),
    )
    lines = ["#### Where the gaps are", ""]
    lines.append("| file | survivors | tested | survival rate |")
    lines.append("| --- | ---: | ---: | ---: |")
    for file, survivors, tested in rows:
        lines.append(
            f"| `{file}` | {survivors} | {tested} | {pct(survivors, tested)} |"
        )
    lines.append("")
    return lines


def render_files(mutants):
    """Every file's merged outcomes, collapsed: the full picture per file."""
    per_file = {}
    for mutant in mutants:
        row = per_file.setdefault(mutant.file, {key: 0 for key in OUTCOME_KEYS})
        row[mutant.key] += 1
    if not per_file:
        return []
    lines = ["<details><summary>Outcomes per file</summary>", ""]
    lines.append("| file | caught | missed | timeout | unviable | unbuilt |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for file in sorted(per_file):
        c = per_file[file]
        lines.append(
            f"| `{file}` | {c['caught']} | {c['missed']} | {c['timeout']} | "
            f"{c['unviable']} | {c['unbuilt']} |"
        )
    lines.extend(["", "</details>", ""])
    return lines


def render_unbuilt(mutants):
    """Files holding mutants no leg built, with why each leg skipped them."""
    per_file = {}
    for mutant in mutants:
        if mutant.key != "unbuilt":
            continue
        entry = per_file.setdefault(mutant.file, [0, {}])
        entry[0] += 1
        for leg, reason in mutant.unbuilt_on.items():
            entry[1].setdefault(leg, {})
            entry[1][leg][reason] = entry[1][leg].get(reason, 0) + 1
    if not per_file:
        return []
    total = sum(entry[0] for entry in per_file.values())
    lines = [
        f"#### {plural(total, 'mutant')} not built on any leg that ran them",
        "",
        (
            "Not counted as survivors: the mutated code is cfg'd out for every "
            "target that ran it tonight."
        ),
        "",
        "| file | unbuilt | why, per leg |",
        "| --- | ---: | --- |",
    ]
    for file in sorted(per_file, key=lambda f: (-per_file[f][0], f)):
        count, reasons = per_file[file]
        why = "<br>".join(
            f"{leg}: "
            + "; ".join(
                f"{reason.replace('|', '&#124;')}" + (f" ×{n}" if n > 1 else "")
                for reason, n in sorted(per_leg.items(), key=lambda kv: -kv[1])[:3]
            )
            for leg, per_leg in sorted(reasons.items())
        )
        lines.append(f"| `{file}` | {count} | {why} |")
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


def render(mutants, legs, shards, total, start, budget):
    counts = tally(mutants)
    groups = group_survivors(mutants)
    labelled = sorted({a for m in mutants for a in m.arches})
    name_arch = len(labelled) > 1
    header = render_header(counts, shards, total, start, legs)
    header = render_broken(legs) + header
    header.extend(render_legs(legs))
    header.extend(render_disagreements(legs))
    header.extend(render_unbuilt(mutants))
    header.extend(render_files(mutants))

    if not counts["missed"]:
        clean = "No mutant survived: every tested mutant failed a test."
        if counts["timeout"]:
            # A timeout is not a pass. The mutant ran long enough to look like
            # a hang, so whether a test would have caught it stays unresolved,
            # and calling that a clean sweep overstates the run.
            clean = (
                f"No mutant survived, but {plural(counts['timeout'], 'mutant')} "
                "timed out rather than failing a test, so this is not a clean "
                "sweep."
            )
        return fit_whole(header + [clean], budget)

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

    return fit_whole(header + [TRUNCATION_NOTE.format(extra="")], budget)


def fit_whole(lines, budget):
    """Join lines, cutting to a clean prefix when the whole does not fit."""
    whole = "\n".join(lines) + "\n"
    if len(whole.encode("utf-8")) <= budget:
        return whole
    return whole.encode("utf-8")[:budget].decode("utf-8", "ignore")


def default_repo():
    return Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shards", default="shards", help="directory holding mutants-shard-*/"
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="checkout the mutants were generated from (default: this script's repo)",
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
        "--expected-matrix",
        default=os.environ.get("EXPECTED_MATRIX"),
        help='the plan job\'s {"include": [...]} matrix; a scheduled shard '
        "with no report fails the roll-up",
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

    repo = Path(args.repo) if args.repo else default_repo()
    missing = missing_shards(args.shards, args.expected_matrix)
    mutants, legs, shards = collect(args.shards, repo)
    counts = tally(mutants)
    tested = sum(counts[key] for key in TESTED_KEYS)
    broken = sum(n for leg in legs for n in leg.broken.values())

    if not shards:
        report = "### Mutation testing\n\nNo shard uploaded a report.\n"
    elif not tested:
        lines = render_broken(legs) + render_header(
            counts, shards, args.total, args.start, legs
        )
        lines += render_legs(legs) + render_unbuilt(mutants)
        report = fit_whole(lines, args.budget)
    else:
        report = render(mutants, legs, shards, args.total, args.start, args.budget)
    if missing:
        report = "\n".join(render_missing(missing)) + "\n" + report

    if args.output:
        Path(args.output).write_text(report, encoding="utf-8")
    elif args.github_actions and os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as handle:
            handle.write(report)
    else:
        sys.stdout.write(report)

    # A run that enumerates mutants but tests none is a no-op wearing a green
    # tick, which is how this lane's first execution presented.
    for name in missing:
        print(f"::error::{name} was scheduled but uploaded no report", file=sys.stderr)
    if not shards:
        print("::error::no shard uploaded a mutation report", file=sys.stderr)
        return 2
    if not tested:
        print("::error::no mutants were tested; the report is empty", file=sys.stderr)
        return 2

    print(
        f"tested={tested} caught={counts['caught']} missed={counts['missed']} "
        f"timeout={counts['timeout']} unviable={counts['unviable']} "
        f"unbuilt={counts['unbuilt']}"
    )
    code = 0
    for leg in legs:
        for file, n in sorted(leg.broken.items()):
            print(
                f"::error::{leg.label} did not build {file} ({plural(n, 'mutant')}) "
                "although its target compiles it; the leg is misconfigured",
                file=sys.stderr,
            )
    if broken or missing:
        code = 1
    if counts["missed"]:
        print(
            f"::error::{plural(counts['missed'], 'mutant')} survived: "
            "the suite passed with the mutation applied",
            file=sys.stderr,
        )
        code = 1
    return code


if __name__ == "__main__":
    sys.exit(main())
