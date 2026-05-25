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
DEPS_DIR="target/$PROFILE/deps"

echo "→ Building workspace tests ($PROFILE)…"
cargo build --tests ${PROFILE_FLAG[@]+"${PROFILE_FLAG[@]}"} \
    --workspace --exclude gpu-probe --exclude edgefirst-bench

echo "→ Codesigning test binaries with library-validation disabled…"
# Sign every executable file in target/<profile>/deps that looks like a
# test binary. Skip .d files (cargo dep manifests) and anything that's
# already a script.
find "$DEPS_DIR" -maxdepth 1 -type f -perm -u+x \
    \( -name "edgefirst*-*" -o -name "hal*-*" \) \
    ! -name "*.d" ! -name "*.dSYM" \
    -exec codesign --force --sign - --entitlements "$ENTITLEMENTS" {} \; \
    > /dev/null 2>&1

echo "→ Running nextest…"
# We've just signed every test binary with the library-validation
# entitlement, so dlopen()-based tests are safe to run here. Unit tests
# that probe ANGLE via dlopen gate themselves on this env var so that a
# plain `cargo test` (no codesign) does not SIGKILL on Tahoe.
export HAL_TEST_ALLOW_DLOPEN_ANGLE=1
exec cargo nextest run ${PROFILE_FLAG[@]+"${PROFILE_FLAG[@]}"} \
    --workspace --exclude gpu-probe --exclude edgefirst-bench \
    "$@"
