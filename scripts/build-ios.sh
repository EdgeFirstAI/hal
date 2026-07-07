#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Build the EdgeFirst HAL for iOS (device + simulator) and validate the link
# closure against the ANGLE xcframeworks.
#
# This is the one-command entry point for iOS: it ensures the Rust iOS
# targets are installed, builds the HAL library closure for both the device
# and simulator triples (default features, incl. `opengl`), then runs
# `scripts/validate-ios-link.sh` for each to prove the native symbol closure
# resolves against the ANGLE + Apple system frameworks.
#
# Runtime EGL initialization on a device/simulator is out of scope — that
# needs the app shell (a future Swift-bindings effort). See README.md § iOS.
#
# Usage:
#   scripts/build-ios.sh                # build + validate both device & sim
#   scripts/build-ios.sh device         # device only
#   scripts/build-ios.sh sim            # simulator only
#   scripts/build-ios.sh --no-validate  # build only, skip link validation
#   scripts/build-ios.sh --release      # force release (default; debug with --debug)
#
# Prerequisites:
#   - Xcode + the iOS SDKs (xcode-select)
#   - ../angle-package/dist/{EGL,GLESv2}.xcframework (for validation only)
#
# Exit codes: 0 success; 1 build/validation failure; 2 prerequisites missing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${HAL_ROOT}"

VARIANTS=("device" "sim")
VALIDATE=1
PROFILE="release"
EXTRA_CARGO_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        device|sim) VARIANTS=("$1") ;;
        --no-validate) VALIDATE=0 ;;
        --release) PROFILE="release" ;;
        --debug) PROFILE="dev" ;;
        --) shift; EXTRA_CARGO_ARGS+=("$@"); break ;;
        *) EXTRA_CARGO_ARGS+=("$1") ;;
    esac
    shift
done

# --- Rust target check -------------------------------------------------------

NEEDED_TARGETS=()
for v in "${VARIANTS[@]}"; do
    case "${v}" in
        device) NEEDED_TARGETS+=("aarch64-apple-ios") ;;
        sim) NEEDED_TARGETS+=("aarch64-apple-ios-sim") ;;
    esac
done

for t in "${NEEDED_TARGETS[@]}"; do
    if ! rustup target list --installed 2>/dev/null | grep -q "^${t}$"; then
        echo "==> installing Rust target ${t}"
        rustup target add "${t}"
    fi
done

# --- Build + validate --------------------------------------------------------

CARGO_PROFILE_ARGS=()
if [[ "${PROFILE}" == "release" ]]; then
    CARGO_PROFILE_ARGS+=(--release)
fi

for v in "${VARIANTS[@]}"; do
    case "${v}" in
        device) TRIPLE="aarch64-apple-ios" ;;
        sim) TRIPLE="aarch64-apple-ios-sim" ;;
    esac

    echo ""
    echo "==> building edgefirst-hal for ${v} (${TRIPLE}, ${PROFILE})"
    # shellcheck disable=SC2086
    cargo build --target "${TRIPLE}" ${CARGO_PROFILE_ARGS[@]+"${CARGO_PROFILE_ARGS[@]}"} \
        -p edgefirst-hal ${EXTRA_CARGO_ARGS[@]+"${EXTRA_CARGO_ARGS[@]}"}

    if [[ ${VALIDATE} -eq 1 ]]; then
        echo ""
        echo "==> validating ${v} link closure"
        bash "${SCRIPT_DIR}/validate-ios-link.sh" "${v}"
    fi
done

echo ""
echo "==> DONE: iOS build${VALIDATE:+ + validation} complete for: ${VARIANTS[*]}"
