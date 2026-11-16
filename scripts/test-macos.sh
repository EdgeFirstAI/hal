#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Build, codesign, and run the EdgeFirst HAL test suite on macOS.
#
# Usage:
#   scripts/test-macos.sh                  # build + sign + run full suite
#   scripts/test-macos.sh -p edgefirst-image  # extra args forwarded to nextest
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

# Features for the build and every nextest invocation.
FEATURES=(--features edgefirst-image/opengl)

sign_test_binaries() {
    # Sign every test binary in the given deps dir with the library-
    # validation-disabled entitlement. Skips .d manifests and dSYM bundles.
    local deps_dir="$1"
    find "$deps_dir" -maxdepth 1 -type f -perm -u+x \
        \( -name "edgefirst*-*" -o -name "hal*-*" \) \
        ! -name "*.d" ! -name "*.dSYM" \
        -exec codesign --force --sign - --entitlements "$ENTITLEMENTS" {} \; \
        > /dev/null 2>&1
}

if [[ "$COVERAGE" != "1" ]]; then
    # ----- Normal path: build → sign → run -----
    DEPS_DIR="target/$PROFILE/deps"
    echo "→ Building workspace tests ($PROFILE)…"
    cargo build --tests ${PROFILE_FLAG[@]+"${PROFILE_FLAG[@]}"} "${WS_EXCLUDES[@]}" "${FEATURES[@]}"
    echo "→ Codesigning test binaries with library-validation disabled…"
    sign_test_binaries "$DEPS_DIR"
    echo "→ Running nextest…"
    export HAL_TEST_ALLOW_DLOPEN_ANGLE=1
    exec cargo nextest run ${PROFILE_FLAG[@]+"${PROFILE_FLAG[@]}"} "${WS_EXCLUDES[@]}" "${FEATURES[@]}" "$@"
fi

# ----- Coverage path: two passes with the sign step between them -----
#
#   Pass 1  build (instrumented) + run; the GL/ANGLE-dlopen tests self-skip
#           while HAL_TEST_ALLOW_DLOPEN_ANGLE is unset.
#   sign    the instrumented binaries cargo-llvm-cov built.
#   Pass 2  re-run with HAL_TEST_ALLOW_DLOPEN_ANGLE=1.
#   report  merge all accumulated profraw into LCOV.
echo "→ Coverage: cleaning previous profraw…"
cargo llvm-cov clean --workspace

# cargo-llvm-cov's target dir, holding the instrumented test binaries to sign.
COV_DEPS_DIR="target/llvm-cov-target/$PROFILE/deps"

echo "→ Coverage pass 1/2: build + run (GL dlopen tests self-skip)…"
cargo llvm-cov nextest --no-report ${PROFILE_FLAG[@]+"${PROFILE_FLAG[@]}"} \
    "${WS_EXCLUDES[@]}" "${FEATURES[@]}" "$@"

echo "→ Codesigning instrumented test binaries…"
sign_test_binaries "$COV_DEPS_DIR"

echo "→ Coverage pass 2/2: re-run with ANGLE dlopen enabled…"
HAL_TEST_ALLOW_DLOPEN_ANGLE=1 cargo llvm-cov nextest --no-report \
    ${PROFILE_FLAG[@]+"${PROFILE_FLAG[@]}"} "${WS_EXCLUDES[@]}" "${FEATURES[@]}" "$@"

echo "-> Generating LCOV report -> ${COV_LCOV}"
cargo llvm-cov report --lcov --output-path "${COV_LCOV}" \
    --ignore-filename-regex '(\.cargo|/rustc/|/target/)'
echo "-> Coverage report written to ${COV_LCOV}"

echo "-> Overall coverage summary (per-file detail is in ${COV_LCOV})"
cargo llvm-cov report --summary-only \
    --ignore-filename-regex '(\.cargo|/rustc/|/target/)'
