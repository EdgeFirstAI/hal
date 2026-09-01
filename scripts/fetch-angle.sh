#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Download and verify a pre-built ANGLE package from the
# EdgeFirstAI/angle-package GitHub release, then extract it to a local
# directory. Used by the macOS, iOS and Windows builders so they share one
# signed ANGLE artifact instead of each rebuilding from source (or, on
# macOS, relying on the Homebrew tap + re-sign dance).
#
# The release ships, per tag (both built from the same pinned ANGLE commit):
#   angle-xcframeworks-<tag>.zip          — EGL.xcframework + GLESv2.xcframework (macOS/iOS)
#   angle-xcframeworks-<tag>.zip.sha256   — checksum
#   angle-windows-x64-<tag>.zip           — bin/libEGL.dll + libGLESv2.dll, lib/*.lib,
#                                           include/, LICENSE, BUILD_INFO.txt (Windows)
#   angle-windows-x64-<tag>.zip.sha256    — checksum
#
# Output layout (extracted):
#   macOS/iOS:
#     <dest>/EGL.xcframework/{ios-arm64,ios-arm64-simulator,macos-arm64}/...
#     <dest>/GLESv2.xcframework/{ios-arm64,ios-arm64-simulator,macos-arm64}/...
#     <dest>/macos-flat-lib/{libEGL,libGLESv2}.dylib   (EDGEFIRST_ANGLE_PATH for macOS)
#     <dest>/BUILD_INFO.txt
#   Windows:
#     <dest>/windows-x64/bin/{libEGL,libGLESv2}.dll     (EDGEFIRST_ANGLE_PATH for Windows)
#     <dest>/windows-x64/{lib,include}/, LICENSE, BUILD_INFO.txt
#
# Usage:
#   scripts/fetch-angle.sh [--windows|--macos] [dest-dir] [tag]
#
#   --windows / --macos  which package to fetch. Default: from the host
#                        (`uname -s` MINGW*/MSYS*/CYGWIN* → windows, else macos);
#                        FETCH_ANGLE_PLATFORM=windows|macos overrides.
#   dest-dir  defaults to target/angle  (a cached download location)
#   tag       defaults to v2.1.28252  (matches the ANGLE GL_VERSION string)
#
# Environment:
#   GH_TOKEN / GITHUB_TOKEN  optional. EdgeFirstAI/angle-package is public, so no
#                            auth is required to download the release. A token (or
#                            a local `gh auth login`) is still honored if present
#                            and raises GitHub's API rate limit, but is not needed.
#
# Exit codes: 0 success (or already cached + verified); 1 download/verify/extract failure.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HAL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PLATFORM="${FETCH_ANGLE_PLATFORM:-}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --windows) PLATFORM=windows; shift ;;
        --macos)   PLATFORM=macos; shift ;;
        -h|--help) sed -n '5,44p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) break ;;
    esac
done
if [[ -z "${PLATFORM}" ]]; then
    case "$(uname -s 2>/dev/null || echo unknown)" in
        MINGW*|MSYS*|CYGWIN*|Windows_NT) PLATFORM=windows ;;
        *) PLATFORM=macos ;;
    esac
fi
case "${PLATFORM}" in windows|macos) ;; *) echo "fetch-angle: unknown platform '${PLATFORM}' (windows|macos)" >&2; exit 1 ;; esac

DEST="${1:-${HAL_ROOT}/target/angle}"
TAG="${2:-v2.1.28252}"
REPO="EdgeFirstAI/angle-package"
if [[ "${PLATFORM}" == windows ]]; then
    ZIP_NAME="angle-windows-x64-${TAG}.zip"
else
    ZIP_NAME="angle-xcframeworks-${TAG}.zip"
fi
SHA_NAME="${ZIP_NAME}.sha256"
DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${TAG}/${ZIP_NAME}"
SHA_URL="https://github.com/${REPO}/releases/download/${TAG}/${SHA_NAME}"

CACHE_DIR="${HAL_ROOT}/target/angle-cache"
mkdir -p "${CACHE_DIR}"
ZIP_PATH="${CACHE_DIR}/${ZIP_NAME}"
SHA_PATH="${CACHE_DIR}/${SHA_NAME}"

# --- Download (if not cached) ------------------------------------------------
#
# The EdgeFirstAI/angle-package repo is public, so no authentication is required.
# Prefer `gh release download` (handles redirects; uses a token if one happens to
# be set); fall back to plain curl, which downloads the public asset with no token.

download_asset() {
    local url="$1" out="$2" name="$3"
    if command -v gh >/dev/null 2>&1; then
        # `gh release download` handles redirects (and auth if a token is set).
        # --clobber overwrites a stale cached copy; --output names the file.
        if gh release download "${TAG}" --repo "${REPO}" --pattern "${name}" \
            --dir "${CACHE_DIR}" --clobber 2>/dev/null; then
            return 0
        fi
        echo "fetch-angle: gh download failed for ${name}, trying curl…" >&2
    fi
    # curl fallback: the release is public, so no token is needed (one is used if set).
    local auth=()
    if [[ -n "${GH_TOKEN:-${GITHUB_TOKEN:-}}" ]]; then
        auth=(-H "Authorization: Bearer ${GH_TOKEN:-${GITHUB_TOKEN}}")
    fi
    # shellcheck disable=SC2086
    curl -fsSL ${auth[@]+"${auth[@]}"} -o "${out}" "${url}"
}

if [[ ! -f "${ZIP_PATH}" ]]; then
    echo "fetch-angle: downloading ${ZIP_NAME} from ${REPO} ${TAG}"
    download_asset "${DOWNLOAD_URL}" "${ZIP_PATH}" "${ZIP_NAME}"
else
    echo "fetch-angle: using cached ${ZIP_PATH}"
fi

# Always (re)fetch the checksum — it's tiny and guards against a stale
# checksum if the release was re-published. Capture failure rather than
# swallowing it: this artifact is linked straight into the app, so verifying
# it is a supply-chain requirement, not a nicety.
sha_download_ok=1
download_asset "${SHA_URL}" "${SHA_PATH}" "${SHA_NAME}" 2>/dev/null || sha_download_ok=0

# --- Verify the checksum -----------------------------------------------------
#
# The .sha256 file may be "<hash>  <filename>" (shasum format) or bare "<hash>".
# We take the first whitespace-delimited token as the hash, which handles both.
#
# Fail CLOSED: if the checksum cannot be fetched/read we refuse to use the
# unverified zip. Set FETCH_ANGLE_ALLOW_UNVERIFIED=1 to override (e.g. an
# offline dev box working from a trusted cached zip).

if [[ ${sha_download_ok} -eq 0 || ! -s "${SHA_PATH}" ]]; then
    if [[ "${FETCH_ANGLE_ALLOW_UNVERIFIED:-0}" == "1" ]]; then
        echo "fetch-angle: WARNING — no sha256 available; proceeding UNVERIFIED (FETCH_ANGLE_ALLOW_UNVERIFIED=1)" >&2
    else
        echo "fetch-angle: ERROR — could not fetch/read ${SHA_NAME}; refusing to use an unverified artifact." >&2
        echo "  This artifact is linked into the app; a missing checksum is treated as a hard failure." >&2
        echo "  Set FETCH_ANGLE_ALLOW_UNVERIFIED=1 to override (not recommended)." >&2
        exit 1
    fi
else
    # First whitespace-delimited token = the hash (bare or "hash  filename").
    EXPECTED="$(awk '{print $1; exit}' "${SHA_PATH}")"
    # macOS ships shasum, Git Bash ships sha256sum, and plain Windows has certutil.
    if command -v sha256sum >/dev/null 2>&1; then
        ACTUAL="$(sha256sum "${ZIP_PATH}" | awk '{print $1}')"
    elif command -v shasum >/dev/null 2>&1; then
        ACTUAL="$(shasum -a 256 "${ZIP_PATH}" | awk '{print $1}')"
    elif command -v certutil >/dev/null 2>&1; then
        ACTUAL="$(certutil -hashfile "$(cygpath -w "${ZIP_PATH}" 2>/dev/null || echo "${ZIP_PATH}")" SHA256 | sed -n '2p' | tr -d ' \r')"
    else
        echo "fetch-angle: ERROR — no sha256 tool found (need sha256sum, shasum, or certutil)" >&2
        exit 1
    fi
    ACTUAL="$(echo "${ACTUAL}" | tr 'A-F' 'a-f')"
    EXPECTED="$(echo "${EXPECTED}" | tr 'A-F' 'a-f')"
    if [[ "${EXPECTED}" != "${ACTUAL}" ]]; then
        echo "fetch-angle: CHECKSUM MISMATCH" >&2
        echo "  expected: ${EXPECTED}" >&2
        echo "  actual:   ${ACTUAL}" >&2
        echo "  (delete ${ZIP_PATH} to force a re-download)" >&2
        exit 1
    fi
    echo "fetch-angle: sha256 verified (${ACTUAL:0:12}…)"
fi

# --- Extract -----------------------------------------------------------------

# Portable unzip: Git Bash has unzip; Windows 10+ tar reads zips; PowerShell
# Expand-Archive is the last resort.
extract_zip() {
    local zip="$1" dest="$2"
    mkdir -p "${dest}"
    if command -v unzip >/dev/null 2>&1; then
        # -x "*/._*" skips macOS AppleDouble metadata files (harmless but noisy).
        unzip -q -o "${zip}" -d "${dest}" -x "*/._*" "*/.DS_Store"
    elif command -v tar >/dev/null 2>&1 && tar -tf "${zip}" >/dev/null 2>&1; then
        tar -xf "${zip}" -C "${dest}"
    elif command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -Command \
            "Expand-Archive -LiteralPath '$(cygpath -w "${zip}")' -DestinationPath '$(cygpath -w "${dest}")' -Force"
    else
        echo "fetch-angle: ERROR — no unzip/tar/Expand-Archive available to extract ${zip}" >&2
        exit 1
    fi
}

if [[ "${PLATFORM}" == windows ]]; then
    WIN_DIR="${DEST}/windows-x64"
    if [[ -f "${WIN_DIR}/bin/libEGL.dll" && -f "${WIN_DIR}/bin/libGLESv2.dll" ]]; then
        echo "fetch-angle: already extracted at ${WIN_DIR}"
    else
        echo "fetch-angle: extracting to ${WIN_DIR}"
        rm -rf "${WIN_DIR}"
        extract_zip "${ZIP_PATH}" "${WIN_DIR}"
        # The Windows zip is flat (bin/ lib/ include/ at the root); tolerate a
        # top-level dir defensively, as the macOS branch does for dist/.
        if [[ ! -f "${WIN_DIR}/bin/libEGL.dll" ]]; then
            for sub in "${WIN_DIR}"/*/; do
                if [[ -f "${sub}bin/libEGL.dll" ]]; then
                    mv "${sub}"* "${WIN_DIR}/" 2>/dev/null || true
                    rmdir "${sub}" 2>/dev/null || true
                    break
                fi
            done
        fi
        [[ -f "${WIN_DIR}/bin/libEGL.dll" && -f "${WIN_DIR}/bin/libGLESv2.dll" ]] || {
            echo "fetch-angle: ERROR — ${ZIP_NAME} did not contain bin/libEGL.dll + bin/libGLESv2.dll" >&2
            exit 1
        }
    fi
    echo "fetch-angle: ready at ${WIN_DIR}"
    echo "  DLLs:    ${WIN_DIR}/bin  (set EDGEFIRST_ANGLE_PATH here; libEGL.dll and libGLESv2.dll must stay siblings)"
    echo "  headers: ${WIN_DIR}/include   import libs: ${WIN_DIR}/lib"
    if command -v cygpath >/dev/null 2>&1; then
        echo "  PowerShell: \$env:EDGEFIRST_ANGLE_PATH = '$(cygpath -w "${WIN_DIR}/bin")'"
    fi
    exit 0
fi

# Re-extract only if the destination is missing the EGL framework marker.
if [[ -f "${DEST}/EGL.xcframework/macos-arm64/libEGL.framework/libEGL" ]]; then
    echo "fetch-angle: already extracted at ${DEST}"
else
    echo "fetch-angle: extracting to ${DEST}"
    extract_zip "${ZIP_PATH}" "${DEST}"
    # The zip's top level is "dist/"; move its contents up so ${DEST} IS the
    # dist dir (consumers expect ${DEST}/EGL.xcframework directly).
    if [[ -d "${DEST}/dist" ]]; then
        # shellcheck disable=SC2211
        mv "${DEST}"/dist/* "${DEST}"/ 2>/dev/null || true
        rmdir "${DEST}/dist" 2>/dev/null || true
    fi
fi

# --- Stage the macOS flat-lib layout ----------------------------------------
#
# The macOS runtime path `dlopen`s libEGL, and ANGLE's libEGL in turn
# `dlopen`s `libGLESv2.dylib` from its OWN directory (located via dladdr).
# The framework bundle layout (`libEGL.framework/libEGL`) does NOT satisfy
# this — ANGLE would look for `libEGL.framework/libGLESv2.dylib`, which
# doesn't exist. So for the macOS host-test path we stage a flat directory of
# `libEGL.dylib` + `libGLESv2.dylib` siblings copied from the framework
# binaries. Point `EDGEFIRST_ANGLE_PATH` at this dir.
#
# (The iOS path is unaffected — it resolves via Library::this() from the
# embedded frameworks and never goes through this dlopen flow.)
FLAT_DIR="${DEST}/macos-flat-lib"
if [[ -f "${FLAT_DIR}/libEGL.dylib" && -f "${FLAT_DIR}/libGLESv2.dylib" ]]; then
    echo "fetch-angle: flat-lib already staged at ${FLAT_DIR}"
else
    echo "fetch-angle: staging macOS flat-lib at ${FLAT_DIR}"
    mkdir -p "${FLAT_DIR}"
    cp "${DEST}/EGL.xcframework/macos-arm64/libEGL.framework/libEGL" \
       "${FLAT_DIR}/libEGL.dylib"
    cp "${DEST}/GLESv2.xcframework/macos-arm64/libGLESv2.framework/libGLESv2" \
       "${FLAT_DIR}/libGLESv2.dylib"
fi

# Ad-hoc re-sign the flattened dylibs. The release framework binaries are
# Developer-ID-signed + notarized, but that signature is scoped to the
# framework BUNDLE (it seals the framework's Info.plist). Pulling the binary
# out to a flat `libEGL.dylib` invalidates it: `codesign --verify` reports
# "invalid Info.plist (plist or signature have been modified)" and `dlopen`
# fails with "code signature invalid" (an invalid signature is rejected more
# firmly than an unsigned binary). A fresh ad-hoc signature is self-contained
# (no bundle dependency) and loads cleanly; the macOS test binaries carry the
# `disable-library-validation` entitlement, so they accept ad-hoc-signed
# dylibs. This mirrors the Homebrew re-sign step, and only the flat-lib needs
# it — the xcframework/iOS-embed path keeps the original signed frameworks
# intact. Runs every time (idempotent) so it also repairs a stale cached copy.
if command -v codesign >/dev/null 2>&1; then
    echo "fetch-angle: ad-hoc re-signing flat-lib dylibs (bundle-scoped signature is invalid once flattened)"
    codesign --force --sign - "${FLAT_DIR}/libGLESv2.dylib"
    codesign --force --sign - "${FLAT_DIR}/libEGL.dylib"
fi

echo "fetch-angle: ready at ${DEST}"
echo "  EGL (xcframework):    ${DEST}/EGL.xcframework   (iOS link / app embed)"
echo "  GLESv2 (xcframework): ${DEST}/GLESv2.xcframework (iOS link / app embed)"
echo "  macOS flat-lib:       ${FLAT_DIR}  (set EDGEFIRST_ANGLE_PATH here for macOS tests)"
