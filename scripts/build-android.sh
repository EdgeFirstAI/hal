#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# One-command Android build of the EdgeFirst HAL.
#
# Builds the HAL and the C API (the future JNI `.so`) for the requested
# Android ABI(s) at the HAL's minimum API level (26), via cargo-ndk —
# which supplies the NDK's API-26 clang linker and sysroot (see
# .cargo/config.toml § Android targets).
#
# Usage:
#   scripts/build-android.sh                # arm64 + x86_64, release
#   scripts/build-android.sh arm64          # single ABI
#   scripts/build-android.sh x86_64
#
# Prerequisites:
#   - rustup target add aarch64-linux-android x86_64-linux-android
#   - cargo-ndk (cargo install cargo-ndk)
#   - Android NDK r26+ (ANDROID_NDK_HOME or auto-detected by cargo-ndk)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

API_LEVEL=26

declare -a ABIS
case "${1:-all}" in
    arm64)  ABIS=(arm64-v8a) ;;
    x86_64) ABIS=(x86_64) ;;
    all)    ABIS=(arm64-v8a x86_64) ;;
    *)
        echo "error: unknown ABI '${1}' (expected 'arm64', 'x86_64', or no argument)" >&2
        exit 2
        ;;
esac

triple_for() {
    case "$1" in
        arm64-v8a) echo aarch64-linux-android ;;
        x86_64)    echo x86_64-linux-android ;;
    esac
}

for abi in "${ABIS[@]}"; do
    triple="$(triple_for "${abi}")"
    if ! rustup target list --installed 2>/dev/null | grep -q "^${triple}$"; then
        echo "error: Rust target '${triple}' not installed." >&2
        echo "       Run: rustup target add ${triple}" >&2
        exit 2
    fi
done

if ! command -v cargo-ndk >/dev/null 2>&1; then
    echo "error: cargo-ndk not installed. Run: cargo install cargo-ndk" >&2
    exit 2
fi

cd "${HAL_ROOT}"

for abi in "${ABIS[@]}"; do
    triple="$(triple_for "${abi}")"
    echo "==> cargo ndk -t ${abi} -P ${API_LEVEL} build --release (edgefirst-hal + capi)"
    cargo ndk -t "${abi}" -P "${API_LEVEL}" build --release \
        -p edgefirst-hal -p edgefirst-hal-capi

    SO="${HAL_ROOT}/target/${triple}/release/libedgefirst_hal.so"
    if [[ -f "${SO}" ]]; then
        echo "    JNI library: $(du -h "${SO}" | cut -f1) ${SO}"
    fi
done

echo "==> SUCCESS: Android build complete (API ${API_LEVEL}: ${ABIS[*]})"
