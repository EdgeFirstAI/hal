#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Package the five modular C libraries into one relocatable archive:
#
#   edgefirst-hal-<version>-<target>.tar.gz   (Linux)
#   edgefirst-hal-<version>-<target>.zip      (Windows, macOS)
#
# Usage:
#   scripts/package-capi.sh [--version V] [--target LABEL] [--libdir DIR] [--outdir DIR]
#
# TARGET is a host label used in the archive name (x86_64-linux, aarch64-macos,
# x86_64-windows), not necessarily a rustc triple. LIBDIR is where cargo wrote
# the built libraries (target/release, or target/<triple>/release).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

VERSION=""
TARGET=""
LIBDIR=""
OUTDIR="dist"
LEAVES="tensor image codec decoder tracker"

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \?//'
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="${2:?}"; shift 2 ;;
    --target) TARGET="${2:?}"; shift 2 ;;
    --libdir) LIBDIR="${2:?}"; shift 2 ;;
    --outdir) OUTDIR="${2:?}"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "unknown argument: $1" >&2; usage ;;
  esac
done

# `python3` first (Linux/macOS, GitHub runners); Git Bash on a Windows dev box
# usually has only `python`, and `python3` may resolve to the Microsoft Store
# stub, which exits 49 with "Python was not found" — so probe by running it.
PY=""
for cand in python3 python; do
  if command -v "${cand}" >/dev/null 2>&1 && "${cand}" -c 'pass' >/dev/null 2>&1; then
    PY="${cand}"
    break
  fi
done
if [[ -z "${PY}" ]]; then
  echo "FAIL: no working python3/python on PATH" >&2
  exit 1
fi

if [[ -z "${VERSION}" ]]; then
  VERSION="$("${PY}" -c '
import re, pathlib
text = pathlib.Path("Cargo.toml").read_text()
in_ws = False
for line in text.splitlines():
    if line.strip() == "[workspace.package]":
        in_ws = True
        continue
    if in_ws and line.startswith("["):
        break
    if in_ws:
        m = re.match(r"^version\s*=\s*\"([^\"]+)\"", line)
        if m:
            print(m.group(1)); break
')"
fi
if [[ -z "${VERSION}" ]]; then
  echo "FAIL: could not read workspace version" >&2
  exit 1
fi

if [[ -z "${TARGET}" ]]; then
  host="$(rustc -vV | awk '/^host:/{print $2}')"
  case "${host}" in
    x86_64-unknown-linux-*) TARGET=x86_64-linux ;;
    aarch64-unknown-linux-*) TARGET=aarch64-linux ;;
    aarch64-apple-darwin) TARGET=aarch64-macos ;;
    x86_64-apple-darwin) TARGET=x86_64-macos ;;
    x86_64-pc-windows-*|x86_64-*-windows-*) TARGET=x86_64-windows ;;
    aarch64-pc-windows-*) TARGET=aarch64-windows ;;
    *) TARGET="${host}" ;;
  esac
fi

if [[ -z "${LIBDIR}" ]]; then
  LIBDIR="target/release"
fi

MAJOR="${VERSION%%.*}"
rest="${VERSION#*.}"
MINOR="${rest%%.*}"
PATCH="${rest#*.}"
PATCH="${PATCH%%[!0-9]*}"  # drop +metadata / -rcN from the archive filename piece

PKG_NAME="edgefirst-hal-${VERSION}-${TARGET}"
STAGE="${OUTDIR}/${PKG_NAME}"
rm -rf "${STAGE}"
mkdir -p "${STAGE}/include/edgefirst" "${STAGE}/lib/pkgconfig"
# Windows: DLLs in bin/, import libraries in lib/. Unix: shared objects in lib/.
# The archive never ships a Rust staticlib (libedgefirst_*.a / edgefirst_*.lib).

cp LICENSE NOTICE packaging/c/INSTALL.txt packaging/c/README.md "${STAGE}/"

for leaf in ${LEAVES}; do
  cp "crates/${leaf}-capi/include/edgefirst/${leaf}.h" "${STAGE}/include/edgefirst/"
done
cp crates/decoder-abi/include/edgefirst/detect.h "${STAGE}/include/edgefirst/"

copy_or_die() {
  local src="$1" dest="$2"
  if [[ ! -s "${src}" ]]; then
    echo "FAIL: missing or empty ${src}" >&2
    exit 1
  fi
  cp "${src}" "${dest}"
}

for leaf in ${LEAVES}; do
  if [[ "${TARGET}" == *-linux ]]; then
    REAL="libedgefirst_${leaf}.so.${MAJOR}.${MINOR}.${PATCH}"
    copy_or_die "${LIBDIR}/libedgefirst_${leaf}.so" "${STAGE}/lib/${REAL}"
    ln -s "${REAL}" "${STAGE}/lib/libedgefirst_${leaf}.so.${MAJOR}.${MINOR}"
    ln -s "libedgefirst_${leaf}.so.${MAJOR}.${MINOR}" "${STAGE}/lib/libedgefirst_${leaf}.so.${MAJOR}"
    ln -s "libedgefirst_${leaf}.so.${MAJOR}" "${STAGE}/lib/libedgefirst_${leaf}.so"
  elif [[ "${TARGET}" == *-macos ]]; then
    REAL="libedgefirst_${leaf}.${MAJOR}.dylib"
    copy_or_die "${LIBDIR}/libedgefirst_${leaf}.dylib" "${STAGE}/lib/${REAL}"
    ln -s "${REAL}" "${STAGE}/lib/libedgefirst_${leaf}.dylib"
  elif [[ "${TARGET}" == *-windows ]]; then
    mkdir -p "${STAGE}/bin"
    # Cargo writes edgefirst_X.dll + edgefirst_X.dll.lib (import) +
    # edgefirst_X.lib (Rust staticlib) into the same directory. Ship the
    # DLL and the import library under the conventional names
    # bin/edgefirst_X.dll and lib/edgefirst_X.lib. Never copy the staticlib:
    # linking a C consumer against it embeds a second rust std.
    if [[ -f "${LIBDIR}/edgefirst_${leaf}.dll" ]]; then
      copy_or_die "${LIBDIR}/edgefirst_${leaf}.dll" "${STAGE}/bin/edgefirst_${leaf}.dll"
      implib="${LIBDIR}/edgefirst_${leaf}.dll.lib"
    elif [[ -f "${LIBDIR}/libedgefirst_${leaf}.dll" ]]; then
      copy_or_die "${LIBDIR}/libedgefirst_${leaf}.dll" "${STAGE}/bin/edgefirst_${leaf}.dll"
      implib="${LIBDIR}/libedgefirst_${leaf}.dll.a"
    else
      echo "FAIL: no DLL for ${leaf} in ${LIBDIR}" >&2
      exit 1
    fi
    if [[ ! -s "${implib}" ]]; then
      echo "FAIL: missing DLL import library ${implib} (refusing to ship the Rust staticlib)" >&2
      exit 1
    fi
    copy_or_die "${implib}" "${STAGE}/lib/edgefirst_${leaf}.lib"
    # Import libraries are a few hundred KB. The Rust staticlib is tens of MB;
    # if we ever copy the wrong file, fail here instead of publishing it.
    implib_bytes="$(wc -c < "${STAGE}/lib/edgefirst_${leaf}.lib")"
    if [[ "${implib_bytes}" -gt 5000000 ]]; then
      echo "FAIL: ${STAGE}/lib/edgefirst_${leaf}.lib is ${implib_bytes} B — that is the Rust staticlib, not the DLL import library" >&2
      exit 1
    fi
  else
    echo "FAIL: unknown target family in '${TARGET}' (want *-linux, *-macos, *-windows)" >&2
    exit 1
  fi
done

# Windows: bundle ANGLE (GLES over Direct3D 11) next to edgefirst_image.dll
# when EDGEFIRST_ANGLE_PATH points at the fetched DLLs (scripts/fetch-angle.sh),
# so a consumer gets the GPU backend with zero configuration — the runtime
# loader looks next to the loading module before anything else. ANGLE is
# BSD-3-Clause; its licence ships alongside. Without EDGEFIRST_ANGLE_PATH the
# archive is CPU-only, exactly as before.
if [[ "${TARGET}" == *-windows ]]; then
  if [[ -n "${EDGEFIRST_ANGLE_PATH:-}" && -f "${EDGEFIRST_ANGLE_PATH}/libEGL.dll" && -f "${EDGEFIRST_ANGLE_PATH}/libGLESv2.dll" ]]; then
    cp "${EDGEFIRST_ANGLE_PATH}/libEGL.dll" "${EDGEFIRST_ANGLE_PATH}/libGLESv2.dll" "${STAGE}/bin/"
    mkdir -p "${STAGE}/share/licenses/angle"
    angle_root="$(cd "${EDGEFIRST_ANGLE_PATH}/.." && pwd)"
    if [[ -f "${angle_root}/LICENSE" ]]; then
      cp "${angle_root}/LICENSE" "${STAGE}/share/licenses/angle/LICENSE"
    else
      echo "WARN: ${angle_root}/LICENSE not found; shipping ANGLE DLLs without their licence file" >&2
    fi
    [[ -f "${angle_root}/BUILD_INFO.txt" ]] && cp "${angle_root}/BUILD_INFO.txt" "${STAGE}/share/licenses/angle/ANGLE_BUILD_INFO.txt"
    echo "package-capi: bundled ANGLE (libEGL.dll, libGLESv2.dll) from ${EDGEFIRST_ANGLE_PATH}" >&2
  else
    echo "package-capi: EDGEFIRST_ANGLE_PATH unset or incomplete — Windows archive ships without ANGLE (CPU-only GPU fallback)" >&2
  fi
fi

for leaf in ${LEAVES}; do
  sed "s/@VERSION@/${VERSION}/" \
    "crates/${leaf}-capi/edgefirst-${leaf}.pc.in" \
    > "${STAGE}/lib/pkgconfig/edgefirst-${leaf}.pc"
done
sed "s/@VERSION@/${VERSION}/" \
  crates/decoder-abi/edgefirst-decoder-abi.pc.in \
  > "${STAGE}/lib/pkgconfig/edgefirst-decoder-abi.pc"

# Libs.private is Unix-only. macOS has no libdl (dlopen lives in libSystem);
# Windows does not consume these .pc files for linking in the shipped layout.
rewrite_libs_private() {
  local pc="$1" replacement="$2"
  local tmp="${pc}.new"
  sed "s/^Libs.private:.*/Libs.private: ${replacement}/" "${pc}" > "${tmp}"
  mv "${tmp}" "${pc}"
}
if [[ "${TARGET}" == *-macos ]]; then
  for pc in "${STAGE}/lib/pkgconfig/"*.pc; do
    rewrite_libs_private "${pc}" "-lm -lpthread"
  done
elif [[ "${TARGET}" == *-windows ]]; then
  for pc in "${STAGE}/lib/pkgconfig/"*.pc; do
    rewrite_libs_private "${pc}" ""
  done
fi

mkdir -p "${OUTDIR}"
# Archive root is PKG_NAME. Linux: tar.gz (keeps the SONAME symlink chain).
# Windows and macOS: zip. zipfile.write follows dylib symlinks, so the macOS
# zip contains both the versioned and unversioned names as regular files —
# extractable with Explorer / Archive Utility / python -m zipfile.
if [[ "${TARGET}" == *-linux ]]; then
  ARCHIVE="${OUTDIR}/${PKG_NAME}.tar.gz"
  tar -C "${OUTDIR}" -czf "${ARCHIVE}" "${PKG_NAME}"
else
  ARCHIVE="${OUTDIR}/${PKG_NAME}.zip"
  "${PY}" - "${OUTDIR}" "${PKG_NAME}" <<'PY'
import os, sys, zipfile

outdir, pkg_name = sys.argv[1], sys.argv[2]
src = os.path.join(outdir, pkg_name)
zip_path = os.path.join(outdir, pkg_name + ".zip")
parent = os.path.abspath(outdir)
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for dirpath, _, filenames in os.walk(src):
        for name in filenames:
            path = os.path.join(dirpath, name)
            arcname = os.path.relpath(path, parent).replace("\\", "/")
            zf.write(path, arcname)
PY
fi
if [[ ! -s "${ARCHIVE}" ]]; then
  echo "FAIL: did not write ${ARCHIVE}" >&2
  exit 1
fi
rm -rf "${STAGE}"
echo "${ARCHIVE}"
