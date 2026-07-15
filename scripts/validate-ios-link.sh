#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Validate the iOS link closure of the EdgeFirst HAL.
#
# Builds the PRODUCTION C API staticlib (`edgefirst-hal-capi` →
# `libedgefirst_hal.a`; a Rust staticlib archives the full dependency
# closure into one `.a`), then links a trivial `main.c` that references
# real C API entry points against it + the ANGLE xcframeworks (downloaded
# via scripts/fetch-angle.sh from the EdgeFirstAI/angle-package release)
# + the Apple system frameworks the HAL references via
# `#[link(kind = "framework")]` (IOSurface, CoreFoundation, Metal). A
# clean link proves the native symbol closure of the shipped artifact is
# complete.
#
# Because EGL/GLES symbols are resolved at RUNTIME via `libloading`
# (`Library::this()` on iOS), the Rust staticlib has NO link-time references
# to `eglInitialize` etc. — so this script additionally runs `nm` on the
# ANGLE framework binaries to confirm the EGL entry-point names the runtime
# loader will look up are actually exported. That is the real "ANGLE symbol
# closure resolves" check; the link step itself only exercises the Rust +
# Apple-system-framework closure.
#
# Runtime EGL initialization on a device/simulator is out of scope (it
# needs the app shell — a future Swift-bindings effort).
#
# Usage:
#   scripts/validate-ios-link.sh device   # aarch64-apple-ios (arm64 device)
#   scripts/validate-ios-link.sh sim      # aarch64-apple-ios-sim (arm64 simulator)
#   scripts/validate-ios-link.sh          # defaults to "device"
#
# Prerequisites:
#   - rustup target add aarch64-apple-ios aarch64-apple-ios-sim
#   - ANGLE xcframeworks: fetched automatically by scripts/fetch-angle.sh
#     from the public EdgeFirstAI/angle-package release (no credentials
#     needed — a GH_TOKEN/GITHUB_TOKEN is honored if set but not required).
#
# Exit codes:
#   0  link + nm symbol check both succeeded
#   1  build, link, or symbol check failed
#   2  prerequisites missing (target not installed, dist not found)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

VARIANT="${1:-device}"

case "${VARIANT}" in
    device)
        TRIPLE="aarch64-apple-ios"
        SDK="iphoneos"
        SLICE="ios-arm64"
        CLANG_TARGET="arm64-apple-ios16.0"
        ;;
    sim)
        TRIPLE="aarch64-apple-ios-sim"
        SDK="iphonesimulator"
        SLICE="ios-arm64-simulator"
        CLANG_TARGET="arm64-apple-ios16.0-simulator"
        ;;
    *)
        echo "error: unknown variant '${VARIANT}' (expected 'device' or 'sim')" >&2
        exit 2
        ;;
esac

echo "==> Validating iOS link closure (${VARIANT}: ${TRIPLE})"

# --- Prerequisite checks -----------------------------------------------------

if ! rustup target list --installed 2>/dev/null | grep -q "^${TRIPLE}$"; then
    echo "error: Rust target '${TRIPLE}' not installed." >&2
    echo "       Run: rustup target add ${TRIPLE}" >&2
    exit 2
fi

# --- Acquire ANGLE xcframeworks ---------------------------------------------
#
# Download + verify + extract the pre-built signed xcframeworks from the
# EdgeFirstAI/angle-package release (shared with the macOS builder). The
# extracted layout is ${ANGLE_DIST}/{EGL,GLESv2}.xcframework/<slice>/...
ANGLE_DIST="${HAL_ROOT}/target/angle"
echo "==> fetching ANGLE xcframeworks (release v2.1.28252)"
bash "${SCRIPT_DIR}/fetch-angle.sh" "${ANGLE_DIST}" v2.1.28252

EGL_FW="${ANGLE_DIST}/EGL.xcframework/${SLICE}/libEGL.framework/libEGL"
GLESV2_FW="${ANGLE_DIST}/GLESv2.xcframework/${SLICE}/libGLESv2.framework/libGLESv2"
EGL_XCFW="${ANGLE_DIST}/EGL.xcframework"
GLESV2_XCFW="${ANGLE_DIST}/GLESv2.xcframework"

if [[ ! -f "${EGL_FW}" || ! -f "${GLESV2_FW}" ]]; then
    echo "error: ANGLE frameworks not found at ${ANGLE_DIST} after fetch" >&2
    echo "       Expected: ${EGL_FW}" >&2
    echo "                ${GLESV2_FW}" >&2
    exit 2
fi

# --- Build the C API staticlib ------------------------------------------------

echo "==> cargo build --target ${TRIPLE} --release -p edgefirst-hal-capi"
cd "${HAL_ROOT}"
cargo build --target "${TRIPLE}" --release -p edgefirst-hal-capi

STATIC_LIB="${HAL_ROOT}/target/${TRIPLE}/release/libedgefirst_hal.a"
if [[ ! -f "${STATIC_LIB}" ]]; then
    echo "error: staticlib not produced: ${STATIC_LIB}" >&2
    exit 1
fi
echo "    staticlib: $(du -h "${STATIC_LIB}" | cut -f1) ${STATIC_LIB}"

# --- nm symbol check: the archive references the IOSurface entry points ------
#
# A Rust staticlib archives every dependency object, so the tensor crate's
# IOSurface FFI must show up as undefined references. If these vanish, the
# Apple-gated module fell out of the build and the link step below would
# pass vacuously. (Mach-O prepends an underscore to C symbols.)
echo "==> nm: staticlib references the IOSurface entry points"
UNDEF_SYMS="$(nm -u "${STATIC_LIB}" 2>/dev/null || true)"
for sym in _IOSurfaceCreate _IOSurfaceLock _IOSurfaceUnlock _IOSurfaceGetID; do
    if ! grep -q "^${sym}$" <<< "${UNDEF_SYMS}"; then
        echo "    MISSING: staticlib carries no undefined ref to ${sym}" >&2
        exit 1
    fi
done
echo "    all IOSurface references present in the archive"

# --- nm symbol check: confirm ANGLE exports the EGL entry points -------------
#
# The runtime `Dynamic` EGL loader (edgefirst-egl) resolves these by name via
# dlsym. If any are missing from the framework binary, EGL bring-up would
# fail at runtime — catch it here, statically.
echo "==> nm: verifying ANGLE framework exports EGL entry points"

REQUIRED_EGL_SYMBOLS=(
    eglGetProcAddress
    eglGetPlatformDisplayEXT
    eglInitialize
    eglTerminate
    eglBindAPI
    eglChooseConfig
    eglCreateContext
    eglCreatePbufferSurface
    eglMakeCurrent
    eglDestroyContext
    eglDestroySurface
    eglCreatePbufferFromClientBuffer
    eglBindTexImage
    eglReleaseTexImage
)

missing=0
for sym in "${REQUIRED_EGL_SYMBOLS[@]}"; do
    # Mach-O prepends a single underscore to C symbols, so the exported name
    # is `_eglInitialize` etc. Match " T _<sym>" (defined external text symbol).
    if ! nm -gU "${EGL_FW}" 2>/dev/null | grep -q " T _${sym}$"; then
        echo "    MISSING: ${sym} not exported by libEGL (${SLICE})" >&2
        missing=$((missing + 1))
    fi
done
if [[ ${missing} -gt 0 ]]; then
    echo "error: ${missing} EGL symbol(s) missing from ${EGL_FW}" >&2
    exit 1
fi
echo "    all ${#REQUIRED_EGL_SYMBOLS[@]} EGL entry points present in libEGL"

# --- Link step ---------------------------------------------------------------
#
# Link a main.c that REFERENCES real C API entry points against: the Rust
# staticlib, the ANGLE xcframeworks (via -F + -framework), and the Apple
# system frameworks the HAL references through `#[link(kind =
# "framework")]`. The references are what make this a real test: a
# static-archive member is linked only to resolve an undefined symbol, so
# without them ld drops the entire archive and the link succeeds no
# matter how broken the closure is. The chosen entry points pull the deep
# closure in: `hal_tensor_new_image` reaches the IOSurface allocation
# path and `hal_tensor_iosurface_id` the handle surface (making
# `-framework IOSurface` / CoreFoundation load-bearing), and
# `hal_image_processor_new` pulls the whole GL engine — so `-dead_strip`
# cannot drop them and removing any required `-framework` fails the link
# with undefined symbols.
#
# The `.a` is passed by explicit path: `-ledgefirst_hal` would prefer the
# cdylib `libedgefirst_hal.dylib` sitting in the same target directory,
# and a shared-library link defers symbol resolution to load time —
# silently gutting the check.
echo "==> link: ${CLANG_TARGET} main.c + libedgefirst_hal.a + frameworks"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT
MAIN_C="${WORKDIR}/main.c"
cat > "${MAIN_C}" <<'EOF'
// Declared here (no header, signatures irrelevant) — C linkage resolves
// by name only, and the addresses are never called. Taking the addresses
// forces undefined references so ld pulls the archive members in.
extern void hal_image_processor_new(void);
extern void hal_tensor_new_image(void);
extern void hal_tensor_iosurface_id(void);
int main(void) {
    void (*volatile roots[])(void) = {
        hal_image_processor_new,
        hal_tensor_new_image,
        hal_tensor_iosurface_id,
    };
    return roots[0] == roots[1];
}
EOF

OUTPUT="${WORKDIR}/ios-link-check-${VARIANT}"

# shellcheck disable=SC2086
xcrun -sdk "${SDK}" clang \
    -target "${CLANG_TARGET}" \
    -dead_strip \
    "${MAIN_C}" \
    "${STATIC_LIB}" \
    -F "${EGL_XCFW}/${SLICE}" -framework libEGL \
    -F "${GLESV2_XCFW}/${SLICE}" -framework libGLESv2 \
    -framework IOSurface \
    -framework CoreFoundation \
    -framework Metal \
    -o "${OUTPUT}"

echo "    linked OK: ${OUTPUT} ($(file "${OUTPUT}" | cut -d: -f2-))"

echo "==> SUCCESS: iOS (${VARIANT}) link closure is complete"
