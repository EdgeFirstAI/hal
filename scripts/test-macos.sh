#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Build, codesign, and run the EdgeFirst HAL test suite on macOS.
#
# Usage:
#   scripts/test-macos.sh                                # build + sign + run full suite
#   scripts/test-macos.sh -E 'package(edgefirst-image)'  # extra args forwarded to nextest run
#   COVERAGE=1 scripts/test-macos.sh                     # the same run instrumented, then LCOV
#
# The test binaries are built once and every one is signed with
# entitlements.plist. nextest then runs them from the recorded build
# (--binaries-metadata) without invoking cargo. The forwarded arguments reach
# `cargo nextest run` only: select tests with filters (-E, name patterns),
# not cargo build options.
#
# Prerequisites:
#   brew install startergo/angle/angle
#   codesign --force --sign - /opt/homebrew/opt/angle/lib/libEGL.dylib
#   codesign --force --sign - /opt/homebrew/opt/angle/lib/libGLESv2.dylib
# (see TESTING.md § macOS Setup)

set -euo pipefail

# Locate the workspace root: the parent of this script's directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(dirname "$SCRIPT_DIR")"
ENTITLEMENTS="$WORKSPACE_ROOT/entitlements.plist"

if [[ ! -f "$ENTITLEMENTS" ]]; then
    echo "error: $ENTITLEMENTS not found" >&2
    exit 1
fi

cd "$WORKSPACE_ROOT"

PROFILE="${PROFILE:-debug}"
PROFILE_FLAG=()
if [[ "$PROFILE" == "release" ]]; then
    PROFILE_FLAG=(--release)
fi

COVERAGE="${COVERAGE:-0}"
COV_LCOV="${COV_LCOV:-target/coverage_rust.lcov}"
COV_IGNORE='(\.cargo|/rustc/|/target/)'

# The workspace without the probe, bench and Python crates.
WS_EXCLUDES=(
    --workspace
    --exclude gpu-probe
    --exclude edgefirst-bench
    --exclude edgefirst-python-tensor
    --exclude edgefirst-python-codec
    --exclude edgefirst-python-image
    --exclude edgefirst-python-decoder
    --exclude edgefirst-python-tracker --exclude edgefirst-python-kernels
    --exclude edgefirst-python-common
)

# Features for the build.
FEATURES=(--features edgefirst-image/opengl)

# Builds the test binaries, signs each one nextest lists, and runs nextest
# over that build with the forwarded arguments.
build_sign_run() {
    local meta_dir="${CARGO_TARGET_DIR:-target}/nextest-reuse"
    local binaries="$meta_dir/binaries.json"
    local metadata="$meta_dir/cargo-metadata.json"
    mkdir -p "$meta_dir"

    echo "→ Building workspace tests ($PROFILE)…"
    cargo nextest list --list-type binaries-only --message-format json \
        ${PROFILE_FLAG[@]+"${PROFILE_FLAG[@]}"} "${WS_EXCLUDES[@]}" "${FEATURES[@]}" \
        > "$binaries" || return
    cargo metadata --format-version 1 > "$metadata" || return

    echo "→ Codesigning every test binary with library-validation disabled…"
    local count=0 bin
    while IFS= read -r bin; do
        codesign --force --sign - --entitlements "$ENTITLEMENTS" "$bin" > /dev/null 2>&1 || return
        count=$((count + 1))
    done < <(grep -o '"binary-path":"[^"]*"' "$binaries" | sed 's/^"binary-path":"//; s/"$//')
    if [[ "$count" -eq 0 ]]; then
        echo "error: $binaries lists no test binaries" >&2
        return 1
    fi
    echo "  signed $count test binaries"

    echo "→ Running nextest…"
    HAL_TEST_ALLOW_DLOPEN_ANGLE=1 cargo nextest run \
        --binaries-metadata "$binaries" --cargo-metadata "$metadata" "$@"
}

if [[ "$COVERAGE" != "1" ]]; then
    build_sign_run "$@"
    exit 0
fi

echo "→ Coverage: cleaning previous profraw…"
cargo llvm-cov clean --workspace

# The instrumented build, its profraw and the report share cargo-llvm-cov's
# target/llvm-cov-target directory. The report is written whether or not the
# tests pass, and the script exits with the test run's status.
rc=0
(
    env_sh="$(cargo llvm-cov show-env --export-prefix 2> /dev/null)" || exit
    eval "$env_sh"
    export CARGO_TARGET_DIR="$CARGO_LLVM_COV_TARGET_DIR/llvm-cov-target"
    export LLVM_PROFILE_FILE="$CARGO_TARGET_DIR/hal-%p-%12m.profraw"
    build_sign_run "$@"
) || rc=$?

echo "→ Generating LCOV report → ${COV_LCOV}"
cargo llvm-cov report ${PROFILE_FLAG[@]+"${PROFILE_FLAG[@]}"} --lcov \
    --output-path "${COV_LCOV}" --ignore-filename-regex "$COV_IGNORE"
echo "→ Coverage report written to ${COV_LCOV}"

echo "→ Overall coverage summary (per-file detail is in ${COV_LCOV})"
cargo llvm-cov report ${PROFILE_FLAG[@]+"${PROFILE_FLAG[@]}"} --summary-only \
    --ignore-filename-regex "$COV_IGNORE"
exit "$rc"
