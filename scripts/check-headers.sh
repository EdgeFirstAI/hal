#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Compile every shipped header as C11 and C++17 under -Werror, then link and
# run one consumer per library. A missing compiler, a missing library, or a
# skipped link or run step fails the script.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
LIBDIR="target/${PROFILE:-debug}"; fails=0
CC="${CC:-cc}"; CXX="${CXX:-c++}"
WORKDIR="${TMPDIR:-/tmp}/check-headers.$$"
mkdir -p "${WORKDIR}" || { echo "FAIL: cannot create scratch dir ${WORKDIR}"; exit 1; }
trap 'rm -rf "${WORKDIR}"' EXIT
INCS="-Icrates/decoder-abi/include"
for l in tensor image codec decoder tracker kernels; do INCS="${INCS} -Icrates/${l}-capi/include"; done

# Fails when a copy of detect.h exists under a *-capi/include directory.
for dup in crates/image-capi/include/edgefirst/detect.h \
           crates/decoder-capi/include/edgefirst/detect.h \
           crates/tracker-capi/include/edgefirst/detect.h; do
  if [[ -e "${dup}" ]]; then
    echo "FAIL ${dup} duplicates crates/decoder-abi/include/edgefirst/detect.h"
    fails=$((fails+1))
  fi
done
if [[ ! -s crates/decoder-abi/include/edgefirst/detect.h ]]; then
  echo "FAIL crates/decoder-abi/include/edgefirst/detect.h is missing or empty -- cannot measure"
  fails=$((fails+1))
fi

# Fails when a compiler cannot be invoked.
if ! command -v "${CC}" >/dev/null 2>&1; then
  echo "FAIL: C compiler '${CC}' not found -- cannot measure"; exit 1
fi
if ! command -v "${CXX}" >/dev/null 2>&1; then
  echo "FAIL: C++ compiler '${CXX}' not found -- cannot measure"; exit 1
fi

ran_link=0
for l in tensor image codec decoder tracker kernels; do
  h="crates/${l}-capi/include/edgefirst/${l}.h"
  if [[ ! -s "${h}" ]] || [[ ! -r "${h}" ]]; then
    echo "FAIL ${h} is missing, empty, or unreadable -- cannot measure"
    fails=$((fails+1))
    continue
  fi
  printf '#include <edgefirst/%s.h>\nint main(void){return 0;}\n' "$l" > "${WORKDIR}/h_$l.c"
  cp "${WORKDIR}/h_$l.c" "${WORKDIR}/h_$l.cpp"
  "${CC}"  -std=c11   -Wall -Wextra -Wpedantic -Werror ${INCS} -fsyntax-only "${WORKDIR}/h_$l.c" \
    || { echo "FAIL ${h} as C11"; fails=$((fails+1)); }
  "${CXX}" -std=c++17 -Wall -Wextra -Wpedantic -Werror ${INCS} -fsyntax-only -x c++ "${WORKDIR}/h_$l.cpp" \
    || { echo "FAIL ${h} as C++17"; fails=$((fails+1)); }

  src="crates/${l}-capi/tests/c/test_link_and_run.c"
  if [[ ! -f "${src}" ]]; then
    echo "FAIL ${l}: no ${src} -- link-and-run cannot be measured for this library"
    fails=$((fails+1))
    continue
  fi
  case "$(uname -s)" in
    Darwin) so="${LIBDIR}/libedgefirst_${l}.dylib" ;;
    *) so="${LIBDIR}/libedgefirst_${l}.so" ;;
  esac
  if [[ ! -s "${so}" ]] || [[ ! -r "${so}" ]]; then
    echo "FAIL ${l}: ${so} is missing, empty, or unreadable -- cannot link"
    fails=$((fails+1))
    continue
  fi
  bin="${WORKDIR}/link_${l}"
  if ! "${CC}" -std=c11 -Wall -Wextra -Werror -o "${bin}" "${src}" ${INCS} \
      -L"${LIBDIR}" -ledgefirst_${l} -ledgefirst_tensor -Wl,-rpath,"${PWD}/${LIBDIR}"; then
    echo "FAIL ${l}: link"
    fails=$((fails+1))
    continue
  fi
  if ! "${bin}"; then
    echo "FAIL ${l}: run (link succeeded, execution failed)"
    fails=$((fails+1))
    continue
  fi
  ran_link=$((ran_link+1))
done

# `ran_link` counts the libraries linked and run; it must be 6.
if [[ "${ran_link}" -ne 6 ]]; then
  echo "FAIL: only ${ran_link}/6 libraries were actually linked and run -- cannot measure the rest"
  fails=$((fails+1))
fi

[[ "${fails}" -eq 0 ]] && { echo "ALL HEADERS COMPILE AND ALL LIBRARIES RUN"; exit 0; }
echo "${fails} FAILURE(S)"; exit 1
