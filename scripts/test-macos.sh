#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Build, codesign, and run the EdgeFirst HAL test suite on macOS.
#
# Why this script exists: macOS 26 (Tahoe) enforces hardened-runtime
# library validation. A binary that `dlopen`s ANGLE (libEGL.dylib /
# libGLESv2.dylib from Homebrew, whose signatures are broken by
# `install_name_tool` at brew install time) needs the
# `disable-library-validation` entitlement. Without it the process
# SIGKILLs at dylib load time before reaching `main` and the test
# silently disappears from the nextest output.
#
# Usage:
#   scripts/test-macos.sh                  # build + sign + run full suite
#   scripts/test-macos.sh -p edgefirst-image  # extra args forwarded to nextest
#
# Prerequisites:
#   brew install startergo/angle/angle
#   codesign --force --sign - /opt/homebrew/opt/angle/lib/libEGL.dylib
#   codesign --force --sign - /opt/homebrew/opt/angle/lib/libGLESv2.dylib
# (the re-sign step is documented in TESTING.md § macOS Setup; see also
# Homebrew/brew#19144 for why install_name_tool breaks signatures.)

set -euo pipefail

# Locate the workspace root: scripts/ lives at the workspace root, so
# the parent of this script's directory is what we want.
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

WS_EXCLUDES=(--workspace --exclude gpu-probe --exclude edgefirst-bench)

# dma_test_formats un-gates the gl/tests.rs zero-copy tier on macOS
# (IOSurface-backed @Dma fixtures). MUST be identical across the build
# and every nextest invocation below — a feature mismatch between the
# coverage passes triggers a rebuild between the codesign step and the
# run, leaving pass 2 with unsigned binaries (GL tests SIGKILL/skip).
FEATURES=(--features edgefirst-image/dma_test_formats)

sign_test_binaries() {
    # Sign every test binary in the given deps dir with the library-
    # validation-disabled entitlement so dlopen() of ad-hoc-signed ANGLE
    # works on Tahoe. Skips .d manifests and dSYM bundles.
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

# ----- Coverage path: two-pass so the codesign seam survives -----
#
# `cargo llvm-cov nextest` builds instrumented binaries and runs them in one
# shot, with no seam to re-sign between. ANGLE dlopen needs the binaries
# signed with the library-validation entitlement, and that sign step MUST sit
# between the build and the run. So:
#
#   Pass 1  build (instrumented) + run — the GL/ANGLE-dlopen tests gate on
#           HAL_TEST_ALLOW_DLOPEN_ANGLE and self-skip here (unset), so they do
#           not SIGKILL on the unsigned binaries; every non-GL test runs and
#           emits profraw.
#   sign    the instrumented binaries cargo-llvm-cov just built.
#   Pass 2  re-run with HAL_TEST_ALLOW_DLOPEN_ANGLE=1; the binaries are signed
#           and up-to-date (no rebuild), so the IOSurface F16 GL tests now
#           execute under instrumentation and emit their profraw too.
#   report  merge all accumulated profraw into LCOV.
#
# This is the only way the macOS lane covers the ANGLE/IOSurface F16 render
# path — no other coverage lane has a GPU that runs it.
echo "→ Coverage: cleaning previous profraw…"
cargo llvm-cov clean --workspace

# cargo-llvm-cov builds into its own target dir; this is where the instrumented
# test binaries to sign live.
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
