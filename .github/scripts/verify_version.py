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
# The edgefirst-hal meta-crate was deleted in 0.29: it was re-exports plus
# trace.rs, and depending on it made every consumer link the whole tree.
WORKSPACE_CRATES = [
    "edgefirst-codec",
    "edgefirst-decoder",
    "edgefirst-decoder-abi",
    "edgefirst-image",
    "edgefirst-tensor",
    "edgefirst-tensor-abi",
    "edgefirst-tensor-ffi",
    "edgefirst-tracker",
]

# Workspace leaf crates (not referenced as dependencies by other crates).
# The C monolith `edgefirst-hal-capi` is gone; modular C libraries live in
# STANDALONE_LEAF_CRATES (workspace-excluded).
WORKSPACE_LEAF_CRATES = []

# Python extension crates. Four self-contained modules since 0.29; their
# distributions take the version dynamically from Cargo.toml.
PYTHON_CRATES = [
    "edgefirst-python-common",
    "edgefirst-python-tensor",
    "edgefirst-python-codec",
    "edgefirst-python-image",
    "edgefirst-python-decoder",
    "edgefirst-python-tracker",
]

# Crates with independent versions (not checked against workspace version)
INDEPENDENT_CRATES = {}

# The five modular C-API leaves (Plan R2): standalone packages excluded from
# the root workspace, so their versions never show up in the root Cargo.lock.
# Each carries its own Cargo.lock next to its manifest.
STANDALONE_LEAF_CRATES = [
    "edgefirst-tensor-capi",
    "edgefirst-image-capi",
    "edgefirst-codec-capi",
    "edgefirst-decoder-capi",
    "edgefirst-tracker-capi",
]


def read_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


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

    all_crates = WORKSPACE_CRATES + WORKSPACE_LEAF_CRATES + PYTHON_CRATES
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

    # The standalone leaves are excluded from the root workspace, so check
    # each one's own Cargo.lock for its own package instead.
    for crate in STANDALONE_LEAF_CRATES:
        crate_dir = crate.removeprefix("edgefirst-")
        lock_path = f"crates/{crate_dir}/Cargo.lock"
        content = read_file(lock_path)
        pattern = rf'\[\[package\]\]\s*\nname = "{re.escape(crate)}"\nversion = "([^"]+)"'
        matches = re.findall(pattern, content)
        if not matches:
            errors.append(f"{lock_path}: {crate} not found")
        else:
            for found_version in matches:
                if found_version != version:
                    errors.append(
                        f"{lock_path}: {crate} version is '{found_version}', "
                        f"expected '{version}'"
                    )

    return errors


def check_pyproject_toml(version: str) -> list[str]:
    """Check the five Python distributions.

    Since 0.29 they take their version dynamically from Cargo.toml, so there is
    no literal to drift. What CAN drift is each dependent's `~=` pin on
    edgefirst-tensor -- and a stale pin means publishing something nobody can
    install, which PyPI will not let you correct in place.
    """
    errors = []
    packages = ["tensor", "codec", "image", "decoder", "tracker"]

    for pkg in packages:
        pyproject = Path(f"crates/python-{pkg}/pyproject.toml")
        if not pyproject.exists():
            errors.append(f"{pyproject}: file not found")
            continue

        content = pyproject.read_text(encoding="utf-8")

        if 'dynamic = ["version"]' not in content:
            errors.append(
                f"{pyproject}: expected `dynamic = [\"version\"]` so the version "
                f"comes from Cargo.toml rather than a literal that can drift"
            )

        pin = re.search(r'edgefirst-tensor\s*~=\s*([0-9][0-9.]*)', content)
        if pin:
            pinned = pin.group(1)
            # Exact match, not just major.minor: a PR review caught a stale
            # `~= 0.29.0` pin surviving two patch releases (0.29.1, 0.29.2)
            # because this used to only check the major.minor prefix, which
            # a `~=` pin always satisfies for any patch in the same series --
            # that's the whole point of the operator, so it can never catch
            # the release procedure's real requirement (the exact version).
            if pinned != version:
                errors.append(
                    f"{pyproject}: pins edgefirst-tensor ~= {pinned}, "
                    f"expected ~= {version}"
                )
        elif pkg not in ("tensor", "tracker"):
            errors.append(f"{pyproject}: no edgefirst-tensor pin found")

    return errors

def check_changelog(version: str) -> list[str]:
    """Check CHANGELOG.md has an entry for this version."""
    errors = []
    changelog = Path("CHANGELOG.md")

    if not changelog.exists():
        errors.append("CHANGELOG.md: file not found")
        return errors

    content = changelog.read_text(encoding="utf-8")

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

    content = notice.read_text(encoding="utf-8")
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

    # Check the Python extension crates
    for pkg in PYTHON_CRATES:
        for entry in [l for l in lines if f"* {pkg} " in l]:
            m = re.search(rf"\* {re.escape(pkg)} (\S+)", entry)
            if m and m.group(1) != version:
                errors.append(
                    f"NOTICE: {pkg} has version '{m.group(1)}', "
                    f"expected '{version}'"
                )

    # Check for stale "unknown" version entries for internal crates
    all_internal = WORKSPACE_CRATES + WORKSPACE_LEAF_CRATES + PYTHON_CRATES
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

        content = cargo_toml.read_text(encoding="utf-8")
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
