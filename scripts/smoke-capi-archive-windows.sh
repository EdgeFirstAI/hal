#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Windows counterpart to smoke-capi-archive.sh. pkg-config is not part of
# the Windows archive contract; this compiles one `#include` consumer per
# library against the packaged headers and import libs.
#
# Usage: scripts/smoke-capi-archive-windows.sh <archive.tar.gz>
#
# Requires a C compiler on PATH: cl (after vcvars), clang-cl, or clang.
# The GitHub windows-latest job sources MSVC via ilammy/msvc-dev-cmd.
set -euo pipefail
ARCHIVE="${1:?usage: $0 <archive.tar.gz>}"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "FAIL: ${ARCHIVE} not found" >&2
  exit 1
fi

WORKDIR="${TMPDIR:-/tmp}/smoke-capi-win.$$"
mkdir -p "${WORKDIR}"
trap 'rm -rf "${WORKDIR}"' EXIT
tar xzf "${ARCHIVE}" -C "${WORKDIR}"
PREFIX="$(find "${WORKDIR}" -mindepth 1 -maxdepth 1 -type d | head -1)"
if [[ -z "${PREFIX}" ]] || [[ ! -d "${PREFIX}/include/edgefirst" ]]; then
  echo "FAIL: archive has no include/edgefirst — cannot measure" >&2
  exit 1
fi

INC="${PREFIX}/include"
LIB="${PREFIX}/lib"
BIN="${PREFIX}/bin"

if [[ ! -d "${BIN}" ]]; then
  echo "FAIL: archive has no bin/ — Windows DLLs belong in bin/, import libs in lib/" >&2
  exit 1
fi
# Published names are lib/edgefirst_X.lib (import) and bin/edgefirst_X.dll.
# Cargo's edgefirst_X.dll.lib and the Rust staticlib must not appear.
if find "${PREFIX}" \( -name '*.dll.lib' -o -name '*.a' \) | grep -q .; then
  echo "FAIL: archive ships cargo's .dll.lib or a static library; C packages are shared-only" >&2
  find "${PREFIX}" \( -name '*.dll.lib' -o -name '*.a' \) >&2
  exit 1
fi

# Prefer cl (MSVC), then clang-cl, then clang.
CC=""
for cand in cl clang-cl clang; do
  if command -v "${cand}" >/dev/null 2>&1; then
    CC="${cand}"
    break
  fi
done
if [[ -z "${CC}" ]]; then
  echo "FAIL: no C compiler (cl, clang-cl, or clang) on PATH" >&2
  exit 1
fi
echo "using compiler: ${CC}"

find_implib() {
  local leaf="$1"
  local f="${LIB}/edgefirst_${leaf}.lib"
  if [[ -f "${f}" ]]; then
    printf '%s' "${f}"
    return 0
  fi
  return 1
}

ran=0
for leaf in tensor image codec decoder tracker; do
  src="${WORKDIR}/${leaf}.c"
  exe="${WORKDIR}/${leaf}.exe"
  printf '#include <edgefirst/%s.h>\nint main(void){return 0;}\n' "${leaf}" > "${src}"
  implib="$(find_implib "${leaf}")" || {
    echo "FAIL: no import library for ${leaf} in ${LIB}" >&2
    ls -la "${LIB}" >&2 || true
    exit 1
  }
  case "${CC}" in
    cl)
      # cl writes the exe next to the source unless /Fe is given.
      if ! cl /nologo /W3 /WX /TC "${src}" /I "${INC}" /Fe"${exe}" \
          /link /LIBPATH:"${LIB}" "$(cygpath -w "${implib}" 2>/dev/null || echo "${implib}")"; then
        echo "FAIL: compile ${leaf} (cl)" >&2
        exit 1
      fi
      ;;
    clang-cl)
      if ! clang-cl /nologo /W3 /WX /TC "${src}" /I "${INC}" /Fe"${exe}" \
          /link /LIBPATH:"${LIB}" "${implib}"; then
        echo "FAIL: compile ${leaf} (clang-cl)" >&2
        exit 1
      fi
      ;;
    clang)
      if ! clang -std=c11 -Wall -Wextra -Werror -o "${exe}" "${src}" \
          -I "${INC}" "${implib}"; then
        echo "FAIL: compile ${leaf} (clang)" >&2
        exit 1
      fi
      ;;
  esac
  # Running requires the DLL next to the exe (no SONAME / rpath on Windows).
  dll="${BIN}/edgefirst_${leaf}.dll"
  if [[ ! -f "${dll}" ]]; then
    echo "FAIL: no DLL for ${leaf} in ${BIN}" >&2
    exit 1
  fi
  cp "${dll}" "$(dirname "${exe}")/"
  # Siblings load edgefirst_tensor.dll at process start; copy it too.
  if [[ -f "${BIN}/edgefirst_tensor.dll" ]]; then
    cp "${BIN}/edgefirst_tensor.dll" "$(dirname "${exe}")/"
  fi
  if ! "${exe}"; then
    echo "FAIL: run ${leaf} (compile succeeded, load/execute failed)" >&2
    exit 1
  fi
  ran=$((ran + 1))
  echo "ok ${leaf}"
done

if [[ "${ran}" -ne 5 ]]; then
  echo "FAIL: only ${ran}/5 libraries smoked" >&2
  exit 1
fi
echo "ALL PACKAGED WINDOWS HEADERS COMPILE AND ALL LIBRARIES RUN"
