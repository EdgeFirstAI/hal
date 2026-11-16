#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Download and verify a pre-built ANGLE package from the
# EdgeFirstAI/angle-package GitHub release, then extract it to a local
# directory.
#
# The release ships, per tag:
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
#                        (`uname -s` MINGW*/MSYS*/CYGWIN* → windows, else macos).
#   dest-dir    defaults to target/angle
#   tag         defaults to v2.1.28252
#   -h, --help  print this header
#
# Environment:
#   FETCH_ANGLE_PLATFORM           windows|macos; the package to fetch when no
#                                  option names one.
#   FETCH_ANGLE_ALLOW_UNVERIFIED   1 accepts the zip when its checksum cannot be
#                                  fetched or read.
#   GH_TOKEN / GITHUB_TOKEN        optional; sent with the download when set.
#
# Exit codes: 0 success (or already cached + verified); 1 download, verify or
# extract failure.

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
# Uses `gh release download` when gh is present, and curl otherwise.

download_asset() {
    local url="$1" out="$2" name="$3"
    if command -v gh >/dev/null 2>&1; then
        if gh release download "${TAG}" --repo "${REPO}" --pattern "${name}" \
            --dir "${CACHE_DIR}" --clobber 2>/dev/null; then
            return 0
        fi
        echo "fetch-angle: gh download failed for ${name}, trying curl…" >&2
    fi
    # curl fallback; a token is sent if one is set.
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

# Always (re)fetch the checksum.
sha_download_ok=1
download_asset "${SHA_URL}" "${SHA_PATH}" "${SHA_NAME}" 2>/dev/null || sha_download_ok=0

# --- Verify the checksum -----------------------------------------------------
#
# The .sha256 file may be "<hash>  <filename>" (shasum format) or bare "<hash>";
# the first whitespace-delimited token is the hash.
#
# When the checksum cannot be fetched or read the zip is refused, unless
# FETCH_ANGLE_ALLOW_UNVERIFIED=1 is set.

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
    EXPECTED="$(awk '{print $1; exit}' "${SHA_PATH}")"
    # Uses sha256sum, shasum or certutil, whichever is present.
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

# Extracts with unzip, tar or PowerShell Expand-Archive, whichever is present.
extract_zip() {
    local zip="$1" dest="$2"
    mkdir -p "${dest}"
    if command -v unzip >/dev/null 2>&1; then
        # -x "*/._*" skips macOS AppleDouble metadata files.
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
        # The Windows zip is flat (bin/ lib/ include/ at the root); a top-level
        # directory is also accepted.
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

EGL_BIN="${DEST}/EGL.xcframework/macos-arm64/libEGL.framework/libEGL"
GLES_BIN="${DEST}/GLESv2.xcframework/macos-arm64/libGLESv2.framework/libGLESv2"

# Re-extract unless both framework binaries are present.
if [[ -f "${EGL_BIN}" && -f "${GLES_BIN}" ]]; then
    echo "fetch-angle: already extracted at ${DEST}"
else
    echo "fetch-angle: extracting to ${DEST}"
    # Extract to a staging dir, then move the payload in, replacing whatever is
    # already there.
    STAGING="${DEST}/.extract"
    rm -rf "${STAGING}"
    extract_zip "${ZIP_PATH}" "${STAGING}"

    # The zip's top level is "dist/"; the payload moves to ${DEST} directly.
    # The prefix may be absent.
    SRC="${STAGING}/dist"
    [[ -d "${SRC}" ]] || SRC="${STAGING}"

    for item in EGL.xcframework GLESv2.xcframework BUILD_INFO.txt LICENSE; do
        [[ -e "${SRC}/${item}" ]] || continue
        rm -rf "${DEST:?}/${item}"
        mv "${SRC}/${item}" "${DEST}/"
    done
    rm -rf "${STAGING}"

    # Remove a flat-lib left by an earlier extract.
    rm -rf "${DEST}/macos-flat-lib"
fi

# Check that both framework binaries exist.
for required in "${EGL_BIN}" "${GLES_BIN}"; do
    [[ -f "${required}" ]] || {
        echo "fetch-angle: ERROR - ${required} missing after extracting ${ZIP_NAME}" >&2
        echo "fetch-angle: ${DEST} holds: $(ls -A "${DEST}" 2>/dev/null | tr '\n' ' ')" >&2
        exit 1
    }
done

# --- Stage the macOS flat-lib layout ----------------------------------------
#
# A flat directory of `libEGL.dylib` + `libGLESv2.dylib` siblings copied from
# the framework binaries. `EDGEFIRST_ANGLE_PATH` points at this directory.
FLAT_DIR="${DEST}/macos-flat-lib"
if [[ -f "${FLAT_DIR}/libEGL.dylib" && -f "${FLAT_DIR}/libGLESv2.dylib" ]]; then
    echo "fetch-angle: flat-lib already staged at ${FLAT_DIR}"
else
    echo "fetch-angle: staging macOS flat-lib at ${FLAT_DIR}"
    mkdir -p "${FLAT_DIR}"
    cp "${EGL_BIN}" "${FLAT_DIR}/libEGL.dylib"
    cp "${GLES_BIN}" "${FLAT_DIR}/libGLESv2.dylib"
fi

# Ad-hoc re-sign the flattened dylibs. Runs every time.
if command -v codesign >/dev/null 2>&1; then
    echo "fetch-angle: ad-hoc re-signing flat-lib dylibs (bundle-scoped signature is invalid once flattened)"
    codesign --force --sign - "${FLAT_DIR}/libGLESv2.dylib"
    codesign --force --sign - "${FLAT_DIR}/libEGL.dylib"
fi

echo "fetch-angle: ready at ${DEST}"
echo "  EGL (xcframework):    ${DEST}/EGL.xcframework   (iOS link / app embed)"
echo "  GLESv2 (xcframework): ${DEST}/GLESv2.xcframework (iOS link / app embed)"
echo "  macOS flat-lib:       ${FLAT_DIR}  (set EDGEFIRST_ANGLE_PATH here for macOS tests)"
