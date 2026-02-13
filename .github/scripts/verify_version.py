#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright © 2025 Au-Zone Technologies. All Rights Reserved.
#
# Verify version consistency across all workspace files before release.
#
# Checks:
#   1. Cargo.toml workspace version
#   2. Cargo.toml internal dependency versions
#   3. Cargo.lock versions for all workspace crates
#   4. pyproject.toml Python package version
#   5. CHANGELOG.md has an entry for the version
#   6. NOTICE has correct versions for internal crates (no stale entries)
#
# Usage:
#   python3 .github/scripts/verify_version.py           # reads version from Cargo.toml
#   python3 .github/scripts/verify_version.py 0.5.0     # verify specific version

import re
import sys
from pathlib import Path

# Workspace crates that share the workspace version
WORKSPACE_CRATES = [
    "edgefirst-hal",
    "edgefirst-decoder",
    "edgefirst-image",
    "edgefirst-tensor",
    "edgefirst-tracker",
]

# Workspace leaf crates (not referenced as dependencies by other crates)
WORKSPACE_LEAF_CRATES = [
    "edgefirst-hal-capi",
]

# Python package name (uses underscore)
PYTHON_PACKAGE = "edgefirst_hal"

# Crates with independent versions (not checked against workspace version)
INDEPENDENT_CRATES = {}


def read_file(path: str) -> str:
    return Path(path).read_text()


def extract_workspace_version(cargo_toml: str) -> str | None:
    """Extract version from [workspace.package] section."""
    in_workspace_package = False
    for line in cargo_toml.splitlines():
        if line.strip() == "[workspace.package]":
            in_workspace_package = True
            continue
        if in_workspace_package and line.startswith("["):
            break
        if in_workspace_package:
            m = re.match(r'^version\s*=\s*"([^"]+)"', line)
            if m:
                return m.group(1)
    return None


def check_cargo_toml(version: str) -> list[str]:
    """Check root Cargo.toml for version consistency."""
    errors = []
    content = read_file("Cargo.toml")

    # Check workspace version
    ws_version = extract_workspace_version(content)
    if ws_version != version:
        errors.append(
            f"Cargo.toml: workspace version is '{ws_version}', expected '{version}'"
        )

    # Check internal dependency versions
    for crate in WORKSPACE_CRATES:
        pattern = rf'{crate}\s*=\s*\{{[^}}]*version\s*=\s*"([^"]+)"'
        m = re.search(pattern, content)
        if m:
            dep_version = m.group(1)
            if dep_version != version:
                errors.append(
                    f"Cargo.toml: {crate} dependency version is '{dep_version}', "
                    f"expected '{version}'"
                )
        else:
            errors.append(
                f"Cargo.toml: {crate} dependency not found in workspace dependencies"
            )

    return errors


def check_cargo_lock(version: str) -> list[str]:
    """Check Cargo.lock has correct versions for workspace crates."""
    errors = []
    content = read_file("Cargo.lock")

    all_crates = WORKSPACE_CRATES + WORKSPACE_LEAF_CRATES + [PYTHON_PACKAGE]
    for crate in all_crates:
        # Match [[package]] entries: name = "crate-name" followed by version = "X.Y.Z"
        pattern = rf'\[\[package\]\]\s*\nname = "{re.escape(crate)}"\nversion = "([^"]+)"'
        matches = re.findall(pattern, content)
        if not matches:
            errors.append(f"Cargo.lock: {crate} not found")
        else:
            for found_version in matches:
                if found_version != version:
                    errors.append(
                        f"Cargo.lock: {crate} version is '{found_version}', "
                        f"expected '{version}'"
                    )

    return errors


def check_pyproject_toml(version: str) -> list[str]:
    """Check Python package version in pyproject.toml."""
    errors = []
    pyproject = Path("crates/python/pyproject.toml")

    if not pyproject.exists():
        errors.append("crates/python/pyproject.toml: file not found")
        return errors

    content = pyproject.read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if m:
        py_version = m.group(1)
        if py_version != version:
            errors.append(
                f"crates/python/pyproject.toml: version is '{py_version}', "
                f"expected '{version}'"
            )
    else:
        errors.append("crates/python/pyproject.toml: version field not found")

    return errors


def check_changelog(version: str) -> list[str]:
    """Check CHANGELOG.md has an entry for this version."""
    errors = []
    changelog = Path("CHANGELOG.md")

    if not changelog.exists():
        errors.append("CHANGELOG.md: file not found")
        return errors

    content = changelog.read_text()

    # Look for ## [X.Y.Z] heading
    pattern = rf"## \[{re.escape(version)}\]"
    if not re.search(pattern, content):
        errors.append(
            f"CHANGELOG.md: no entry found for version {version} "
            f"(expected '## [{version}]' heading)"
        )

    # Check it's not only in [Unreleased]
    unreleased_pattern = rf"## \[Unreleased\].*## \[{re.escape(version)}\]"
    if not re.search(unreleased_pattern, content, re.DOTALL):
        errors.append(
            f"CHANGELOG.md: version {version} entry should appear after [Unreleased]"
        )

    return errors


def check_notice(version: str) -> list[str]:
    """Check NOTICE file has correct versions for internal crates."""
    errors = []
    notice = Path("NOTICE")

    if not notice.exists():
        errors.append("NOTICE: file not found")
        return errors

    content = notice.read_text()
    lines = content.splitlines()

    # Check internal crate versions
    for crate in WORKSPACE_CRATES + WORKSPACE_LEAF_CRATES:
        # Look for entries like "  * edgefirst-decoder 0.5.0 (Apache-2.0)"
        crate_entries = [l for l in lines if f"* {crate} " in l]
        if not crate_entries:
            # Not an error if the crate isn't a dependency of anything
            # (it may not appear in NOTICE)
            continue

        for entry in crate_entries:
            m = re.search(rf"\* {re.escape(crate)} (\S+)", entry)
            if m:
                found_version = m.group(1)
                if found_version != version:
                    errors.append(
                        f"NOTICE: {crate} has version '{found_version}', "
                        f"expected '{version}'"
                    )

    # Check Python crate
    py_entries = [l for l in lines if f"* {PYTHON_PACKAGE} " in l]
    for entry in py_entries:
        m = re.search(rf"\* {re.escape(PYTHON_PACKAGE)} (\S+)", entry)
        if m:
            found_version = m.group(1)
            if found_version != version:
                errors.append(
                    f"NOTICE: {PYTHON_PACKAGE} has version '{found_version}', "
                    f"expected '{version}'"
                )

    # Check for stale "unknown" version entries for internal crates
    all_internal = WORKSPACE_CRATES + WORKSPACE_LEAF_CRATES + [PYTHON_PACKAGE]
    for crate in all_internal:
        unknown_entries = [l for l in lines if f"* {crate} unknown" in l]
        if unknown_entries:
            errors.append(
                f"NOTICE: {crate} has stale 'unknown' version entry"
            )

    # Check for old crate names that no longer exist
    old_entries = [l for l in lines if "* edgefirst " in l]
    if old_entries:
        errors.append(
            "NOTICE: found entries for 'edgefirst' (old crate name, "
            "renamed to 'edgefirst-hal')"
        )

    # Check for non-edgefirst gbm entries (should be edgefirst-gbm)
    gbm_entries = [l for l in lines if re.search(r"\* gbm[ -]", l) and "edgefirst" not in l]
    if gbm_entries:
        errors.append(
            "NOTICE: found entries for 'gbm'/'gbm-sys' "
            "(should be 'edgefirst-gbm'/'edgefirst-gbm-sys')"
        )

    return errors


def check_crate_cargo_tomls(version: str) -> list[str]:
    """Check individual crate Cargo.toml files for version inheritance."""
    errors = []

    for crate_dir in Path("crates").iterdir():
        if not crate_dir.is_dir():
            continue

        cargo_toml = crate_dir / "Cargo.toml"
        if not cargo_toml.exists():
            continue

        content = cargo_toml.read_text()
        crate_name = None
        for line in content.splitlines():
            m = re.match(r'^name\s*=\s*"([^"]+)"', line)
            if m:
                crate_name = m.group(1)
                break

        if not crate_name:
            continue

        # Skip independent crates
        if crate_name in INDEPENDENT_CRATES:
            continue

        # Check for workspace version inheritance
        if "version.workspace = true" not in content:
            # Check for explicit version
            m = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
            if m:
                found_version = m.group(1)
                if found_version != version:
                    errors.append(
                        f"crates/{crate_dir.name}/Cargo.toml: {crate_name} has "
                        f"explicit version '{found_version}', expected '{version}' "
                        f"or 'version.workspace = true'"
                    )

    return errors


def main():
    # Determine version to check
    if len(sys.argv) > 1:
        version = sys.argv[1]
    else:
        # Read from workspace Cargo.toml
        content = read_file("Cargo.toml")
        version = extract_workspace_version(content)
        if not version:
            print("Error: could not extract workspace version from Cargo.toml",
                  file=sys.stderr)
            sys.exit(1)

    print(f"Verifying version consistency for v{version}")
    print("=" * 60)

    all_errors = []

    checks = [
        ("Cargo.toml (workspace)", check_cargo_toml),
        ("Cargo.lock", check_cargo_lock),
        ("Crate Cargo.toml files", check_crate_cargo_tomls),
        ("pyproject.toml", check_pyproject_toml),
        ("CHANGELOG.md", check_changelog),
        ("NOTICE", check_notice),
    ]

    for name, check_fn in checks:
        errors = check_fn(version)
        if errors:
            print(f"\n  {name}")
            for error in errors:
                print(f"    {error}")
            all_errors.extend(errors)
        else:
            print(f"  {name}")

    print()
    if all_errors:
        print(f"{len(all_errors)} error(s) found")
        sys.exit(1)
    else:
        print(f"All version checks passed for v{version}")
        sys.exit(0)


if __name__ == "__main__":
    main()
