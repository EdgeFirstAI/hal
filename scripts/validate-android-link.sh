#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Validate the Android link closure of the EdgeFirst HAL.
#
# Builds the PRODUCTION C API staticlib (`edgefirst-hal-capi` →
# `libedgefirst_hal.a`; a Rust staticlib archives the full dependency
# closure into one `.a`), then links a trivial `main.c` that references
# real C API entry points against it + the NDK system-library stubs
# (`libnativewindow`, `libandroid`, `liblog`) that the HAL references via
# `#[link(name = "nativewindow")]`. A clean link proves the native symbol
# closure of the shipped artifact is complete.
#
# Because EGL/GLES symbols are resolved at RUNTIME (`libloading` dlopen of
# libEGL.so + `eglGetProcAddress`), the Rust staticlib has NO link-time
# references to `eglInitialize` etc. — so this script additionally runs
# `llvm-nm` on the NDK's API-26 libEGL/libGLESv2 stubs to confirm the entry
# points the runtime loader will look up are actually exported at the HAL's
# minimum API level. (The NDK stubs mirror the platform's exported ABI per
# API level, so this catches "symbol needs a newer API" statically.)
#
# Runtime GL correctness and performance run on-device via the internal
# hal-mobile Device Farm harness (its `edgefirst-android-validation`
# crate drives the real ImageProcessor through JNI) — out of scope for
# this script.
#
# Usage:
#   scripts/validate-android-link.sh arm64    # aarch64-linux-android (device)
#   scripts/validate-android-link.sh x86_64   # x86_64-linux-android (emulator)
#   scripts/validate-android-link.sh          # defaults to "arm64"
#
# Prerequisites:
#   - rustup target add aarch64-linux-android x86_64-linux-android
#   - cargo-ndk (cargo install cargo-ndk)
#   - Android NDK r26+ (ANDROID_NDK_HOME, ANDROID_NDK_ROOT, or auto-detected
#     under $ANDROID_HOME/ndk or ~/Library/Android/sdk/ndk)
#
# Exit codes:
#   0  link + nm symbol checks succeeded
#   1  build, link, or symbol check failed
#   2  prerequisites missing (target/NDK/cargo-ndk not found)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

API_LEVEL=26
ABI="${1:-arm64}"

case "${ABI}" in
    arm64)
        TRIPLE="aarch64-linux-android"
        NDK_ABI="arm64-v8a"
        ;;
    x86_64)
        TRIPLE="x86_64-linux-android"
        NDK_ABI="x86_64"
        ;;
    *)
        echo "error: unknown ABI '${ABI}' (expected 'arm64' or 'x86_64')" >&2
        exit 2
        ;;
esac

echo "==> Validating Android link closure (${ABI}: ${TRIPLE}, API ${API_LEVEL})"

# --- Prerequisite checks -----------------------------------------------------

if ! rustup target list --installed 2>/dev/null | grep -q "^${TRIPLE}$"; then
    echo "error: Rust target '${TRIPLE}' not installed." >&2
    echo "       Run: rustup target add ${TRIPLE}" >&2
    exit 2
fi

if ! command -v cargo-ndk >/dev/null 2>&1; then
    echo "error: cargo-ndk not installed. Run: cargo install cargo-ndk" >&2
    exit 2
fi

# Locate the NDK: honor ANDROID_NDK_HOME / ANDROID_NDK_ROOT, else pick the
# newest NDK under the SDK locations cargo-ndk also searches.
find_ndk() {
    for var in "${ANDROID_NDK_HOME:-}" "${ANDROID_NDK_ROOT:-}"; do
        if [[ -n "${var}" && -d "${var}" ]]; then
            echo "${var}"
            return 0
        fi
    done
    for base in "${ANDROID_HOME:-}/ndk" "${HOME}/Library/Android/sdk/ndk" \
                "${HOME}/Android/Sdk/ndk" "/usr/local/lib/android/sdk/ndk"; do
        if [[ -d "${base}" ]]; then
            local newest
            newest="$(ls -1 "${base}" | sort -V | tail -1)"
            if [[ -n "${newest}" ]]; then
                echo "${base}/${newest}"
                return 0
            fi
        fi
    done
    return 1
}

NDK="$(find_ndk)" || {
    echo "error: Android NDK not found (set ANDROID_NDK_HOME)" >&2
    exit 2
}
echo "    NDK: ${NDK}"

# Host tag for the prebuilt toolchain (the darwin prebuilt is x86_64-named
# even on Apple Silicon; Rosetta/native shims handle it).
HOST_TAG="$(ls "${NDK}/toolchains/llvm/prebuilt" | head -1)"
TOOLCHAIN="${NDK}/toolchains/llvm/prebuilt/${HOST_TAG}"
CLANG="${TOOLCHAIN}/bin/${TRIPLE}${API_LEVEL}-clang"
LLVM_NM="${TOOLCHAIN}/bin/llvm-nm"
SYSROOT_LIBS="${TOOLCHAIN}/sysroot/usr/lib/${TRIPLE}/${API_LEVEL}"

for f in "${CLANG}" "${LLVM_NM}" "${SYSROOT_LIBS}/libEGL.so" \
         "${SYSROOT_LIBS}/libGLESv2.so" "${SYSROOT_LIBS}/libnativewindow.so"; do
    if [[ ! -e "${f}" ]]; then
        echo "error: NDK component not found: ${f}" >&2
        exit 2
    fi
done

# --- Build the C API staticlib ------------------------------------------------

echo "==> cargo ndk -t ${NDK_ABI} -P ${API_LEVEL} build --release -p edgefirst-hal-capi"
cd "${HAL_ROOT}"
ANDROID_NDK_HOME="${NDK}" cargo ndk -t "${NDK_ABI}" -P "${API_LEVEL}" \
    build --release -p edgefirst-hal-capi

STATIC_LIB="${HAL_ROOT}/target/${TRIPLE}/release/libedgefirst_hal.a"
if [[ ! -f "${STATIC_LIB}" ]]; then
    echo "error: staticlib not produced: ${STATIC_LIB}" >&2
    exit 1
fi
echo "    staticlib: $(du -h "${STATIC_LIB}" | cut -f1) ${STATIC_LIB}"

# --- nm symbol checks ---------------------------------------------------------
#
# (1) The staticlib must carry UNDEFINED references to the AHardwareBuffer
#     entry points — a Rust staticlib archives every dependency object, so
#     the tensor crate's Android FFI must show up here. If these refs
#     vanish, the android module fell out of the build and the link step
#     below would pass vacuously.
echo "==> llvm-nm: staticlib references the AHardwareBuffer entry points"
# Capture once: piping llvm-nm straight into `grep -q` under pipefail turns
# grep's early-exit SIGPIPE into a spurious pipeline failure.
UNDEF_SYMS="$("${LLVM_NM}" -u "${STATIC_LIB}" 2>/dev/null || true)"
for sym in AHardwareBuffer_allocate AHardwareBuffer_lock AHardwareBuffer_unlock \
           AHardwareBuffer_release AHardwareBuffer_acquire AHardwareBuffer_describe; do
    if ! grep -q " U ${sym}$" <<< "${UNDEF_SYMS}"; then
        echo "    MISSING: staticlib carries no undefined ref to ${sym}" >&2
        echo "             (force_closure no longer reaches the FFI — false-positive risk)" >&2
        exit 1
    fi
done
echo "    all AHardwareBuffer references present in the archive"

# (2) The NDK API-${API_LEVEL} stubs must export the symbols the runtime
#     dlopen/eglGetProcAddress path resolves. Exported names carry ELF
#     version suffixes (e.g. `AHardwareBuffer_allocate@@LIBNATIVEWINDOW`),
#     so match on the bare name prefix.
echo "==> llvm-nm: NDK API-${API_LEVEL} stubs export the runtime-resolved entry points"
# Same capture-once shape as above (pipefail vs `grep -q` SIGPIPE).
EGL_EXPORTS="$("${LLVM_NM}" -D --defined-only "${SYSROOT_LIBS}/libEGL.so" 2>/dev/null || true)"
GLES_EXPORTS="$("${LLVM_NM}" -D --defined-only "${SYSROOT_LIBS}/libGLESv2.so" 2>/dev/null || true)"
missing=0
for sym in eglGetDisplay eglInitialize eglBindAPI eglChooseConfig \
           eglCreateContext eglCreatePbufferSurface eglMakeCurrent \
           eglDestroyContext eglDestroySurface eglGetProcAddress \
           eglGetNativeClientBufferANDROID eglCreateImageKHR eglDestroyImageKHR; do
    if ! grep -qE " T ${sym}(@|$)" <<< "${EGL_EXPORTS}"; then
        echo "    MISSING: ${sym} not exported by libEGL.so (API ${API_LEVEL})" >&2
        missing=$((missing + 1))
    fi
done
for sym in glEGLImageTargetTexture2DOES glEGLImageTargetRenderbufferStorageOES; do
    if ! grep -qE " T ${sym}(@|$)" <<< "${GLES_EXPORTS}"; then
        echo "    MISSING: ${sym} not exported by libGLESv2.so (API ${API_LEVEL})" >&2
        missing=$((missing + 1))
    fi
done
if [[ ${missing} -gt 0 ]]; then
    echo "error: ${missing} runtime symbol(s) missing from the API-${API_LEVEL} stubs" >&2
    exit 1
fi
echo "    all runtime entry points exported at API ${API_LEVEL}"

# --- Link step ---------------------------------------------------------------
#
# Link a main.c that REFERENCES real C API entry points against the Rust
# staticlib + the NDK system libraries. The references are what make this
# a real test: a static-archive member is linked only to resolve an
# undefined symbol, so without them the linker drops the entire archive
# and the check is a false positive. The chosen entry points pull the deep
# closure in: `hal_tensor_new_image`/`hal_tensor_from_hardware_buffer`
# reach the AHardwareBuffer allocation/import FFI (making
# `-lnativewindow` load-bearing) and `hal_image_processor_new` pulls the
# whole GL engine.
#
# The `.a` is passed by explicit path: `-ledgefirst_hal` would prefer the
# cdylib `libedgefirst_hal.so` sitting in the same target directory, and a
# shared-library link defers symbol resolution to load time — silently
# gutting the check.
echo "==> link: ${TRIPLE}${API_LEVEL}-clang main.c + libedgefirst_hal.a"

WORKDIR="$(mktemp -d)"
trap 'rm -rf "${WORKDIR}"' EXIT
MAIN_C="${WORKDIR}/main.c"
cat > "${MAIN_C}" <<'EOF'
// Declared here (no header, signatures irrelevant) — C linkage resolves
// by name only, and the addresses are never called. Taking the addresses
// forces undefined references so the linker pulls the archive members in.
extern void hal_image_processor_new(void);
extern void hal_tensor_new_image(void);
extern void hal_tensor_from_hardware_buffer(void);
int main(void) {
    void (*volatile roots[])(void) = {
        hal_image_processor_new,
        hal_tensor_new_image,
        hal_tensor_from_hardware_buffer,
    };
    return roots[0] == roots[1];
}
EOF

OUTPUT="${WORKDIR}/android-link-check-${ABI}"

"${CLANG}" \
    "${MAIN_C}" \
    "${STATIC_LIB}" \
    -lnativewindow \
    -landroid \
    -llog \
    -lm \
    -ldl \
    -o "${OUTPUT}"

echo "    linked OK: ${OUTPUT} ($(file "${OUTPUT}" | cut -d: -f2-))"

echo "==> SUCCESS: Android (${ABI}) link closure is complete"
