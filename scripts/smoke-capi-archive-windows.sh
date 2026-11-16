#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Windows counterpart to smoke-capi-archive.sh: compiles one `#include`
# consumer per library against the packaged headers and import libs.
#
# Usage: scripts/smoke-capi-archive-windows.sh <archive.zip>
#
# Requires a C compiler on PATH: cl (after vcvars), clang-cl, or clang.
set -euo pipefail
ARCHIVE="${1:?usage: $0 <archive.zip>}"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "FAIL: ${ARCHIVE} not found" >&2
  exit 1
fi

WORKDIR="${TMPDIR:-/tmp}/smoke-capi-win.$$"
mkdir -p "${WORKDIR}"
trap 'rm -rf "${WORKDIR}"' EXIT
# Picks the first of python3 and python that runs.
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
case "${ARCHIVE}" in
  *.zip) "${PY}" -m zipfile -e "${ARCHIVE}" "${WORKDIR}" ;;
  *) echo "FAIL: unknown archive type ${ARCHIVE} (want .zip)" >&2; exit 1 ;;
esac
PREFIX="$(find "${WORKDIR}" -mindepth 1 -maxdepth 1 -type d | head -1)"
if [[ -z "${PREFIX}" ]] || [[ ! -d "${PREFIX}/include/edgefirst" ]]; then
  echo "FAIL: archive has no include/edgefirst — cannot measure" >&2
  exit 1
fi

INC="${PREFIX}/include"
LIBDIR="${PREFIX}/lib"
BIN="${PREFIX}/bin"

if [[ ! -d "${BIN}" ]]; then
  echo "FAIL: archive has no bin/ — Windows DLLs belong in bin/, import libs in lib/" >&2
  exit 1
fi
# Published names are lib/edgefirst_X.lib (import) and bin/edgefirst_X.dll.
# Fails when the archive holds a .dll.lib or a static library.
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

winpath() {
  local path="$1"
  cygpath -w "${path}" 2>/dev/null || printf '%s' "${path}"
}

find_implib() {
  local leaf="$1"
  local f="${LIBDIR}/edgefirst_${leaf}.lib"
  if [[ -f "${f}" ]]; then
    printf '%s' "${f}"
    return 0
  fi
  return 1
}

ran=0
for leaf in tensor image codec decoder tracker kernels; do
  src="${WORKDIR}/${leaf}.c"
  exe="${WORKDIR}/${leaf}.exe"
  printf '#include <edgefirst/%s.h>\nint main(void){return 0;}\n' "${leaf}" > "${src}"
  implib="$(find_implib "${leaf}")" || {
    echo "FAIL: no import library for ${leaf} in ${LIBDIR}" >&2
    ls -la "${LIBDIR}" >&2 || true
    exit 1
  }
  case "${CC}" in
    cl)
      # MSYS_NO_PATHCONV keeps MSVC's slash flags intact; file arguments are
      # Windows paths. /Fo keeps the .obj in the work dir.
      if ! MSYS_NO_PATHCONV=1 cl /nologo /W3 /WX /TC "$(winpath "${src}")" \
          /I "$(winpath "${INC}")" /Fe"$(winpath "${exe}")" /Fo"$(winpath "${WORKDIR}")\\" \
          /link /LIBPATH:"$(winpath "${LIBDIR}")" "$(winpath "${implib}")"; then
        echo "FAIL: compile ${leaf} (cl)" >&2
        exit 1
      fi
      ;;
    clang-cl)
      if ! MSYS_NO_PATHCONV=1 clang-cl /nologo /W3 /WX /TC "$(winpath "${src}")" \
          /I "$(winpath "${INC}")" /Fe"$(winpath "${exe}")" /Fo"$(winpath "${WORKDIR}")\\" \
          /link /LIBPATH:"$(winpath "${LIBDIR}")" "$(winpath "${implib}")"; then
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
    *) echo "FAIL: unsupported compiler ${CC}" >&2; exit 1 ;;
  esac
  # Copies the DLL next to the exe.
  dll="${BIN}/edgefirst_${leaf}.dll"
  if [[ ! -f "${dll}" ]]; then
    echo "FAIL: no DLL for ${leaf} in ${BIN}" >&2
    exit 1
  fi
  cp "${dll}" "$(dirname "${exe}")/"
  # Copies edgefirst_tensor.dll, which the siblings load at process start.
  if [[ -f "${BIN}/edgefirst_tensor.dll" ]]; then
    cp "${BIN}/edgefirst_tensor.dll" "$(dirname "${exe}")/"
  fi
  # Copies the bundled ANGLE DLLs, when present, next to the exe.
  for angle_dll in libEGL.dll libGLESv2.dll; do
    if [[ -f "${BIN}/${angle_dll}" ]]; then
      cp "${BIN}/${angle_dll}" "$(dirname "${exe}")/"
    fi
  done
  if ! "${exe}"; then
    echo "FAIL: run ${leaf} (compile succeeded, load/execute failed)" >&2
    exit 1
  fi
  ran=$((ran + 1))
  echo "ok ${leaf}"
done

if [[ "${ran}" -ne 6 ]]; then
  echo "FAIL: only ${ran}/6 libraries smoked" >&2
  exit 1
fi
if [[ -f "${BIN}/libEGL.dll" && -f "${BIN}/libGLESv2.dll" ]]; then
  echo "ANGLE bundled: bin/libEGL.dll + bin/libGLESv2.dll (Direct3D 11 GPU backend available to consumers)"
else
  echo "ANGLE not bundled: archive is CPU-only unless EDGEFIRST_ANGLE_PATH is set at run time"
fi
echo "ALL PACKAGED WINDOWS HEADERS COMPILE AND ALL LIBRARIES RUN"
