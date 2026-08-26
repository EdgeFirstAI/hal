#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Package the five modular C libraries into one relocatable tarball:
#
#   edgefirst-hal-<version>-<target>.tar.gz
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

if [[ -z "${VERSION}" ]]; then
  VERSION="$(python3 -c '
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
PATCH="${PATCH%%[!0-9]*}"  # drop +metadata / -rcN from the tarball filename piece

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
# GNU tar follows the directory argument; produce a tarball whose root is PKG_NAME.
tar -C "${OUTDIR}" -czf "${OUTDIR}/${PKG_NAME}.tar.gz" "${PKG_NAME}"
rm -rf "${STAGE}"
echo "${OUTDIR}/${PKG_NAME}.tar.gz"
