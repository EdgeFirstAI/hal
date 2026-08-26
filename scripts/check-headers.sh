#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Compile every shipped header as C11 AND C++17 under -Werror, then LINK AND
# RUN one consumer per library. `-fsyntax-only` proves a header parses; only
# an executed binary proves the library loads and its symbols resolve.
#
# INABILITY TO MEASURE IS NOT A PASS. A missing compiler, a missing library,
# or a link/run step that gets silently skipped must all count as a failure
# -- never as "0 checked, want 0, PASS". See scripts/check-single-home.sh's
# cannot_measure() for the same rule applied to the sibling gate script.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "${ROOT}"
LIBDIR="target/${PROFILE:-debug}"; fails=0
CC="${CC:-cc}"; CXX="${CXX:-c++}"
WORKDIR="${TMPDIR:-/tmp}/check-headers.$$"
mkdir -p "${WORKDIR}" || { echo "FAIL: cannot create scratch dir ${WORKDIR}"; exit 1; }
trap 'rm -rf "${WORKDIR}"' EXIT
INCS="-Icrates/decoder-abi/include"
for l in tensor image codec decoder tracker; do INCS="${INCS} -Icrates/${l}-capi/include"; done

# G4: detect.h has one home. Copies under *-capi/include are the duplication
# this split exists to remove, and they are not packaged.
for dup in crates/image-capi/include/edgefirst/detect.h \
           crates/decoder-capi/include/edgefirst/detect.h \
           crates/tracker-capi/include/edgefirst/detect.h; do
  if [ -e "${dup}" ]; then
    echo "FAIL ${dup} duplicates crates/decoder-abi/include/edgefirst/detect.h"
    fails=$((fails+1))
  fi
done
if [ ! -s crates/decoder-abi/include/edgefirst/detect.h ]; then
  echo "FAIL crates/decoder-abi/include/edgefirst/detect.h is missing or empty -- cannot measure"
  fails=$((fails+1))
fi

# A compiler that cannot be invoked at all must fail loudly rather than let
# every `-fsyntax-only`/link step below silently no-op through a missing
# binary -- `command -v` catches that before the loop ever starts trusting
# CC/CXX's exit code as a real verdict.
if ! command -v "${CC}" >/dev/null 2>&1; then
  echo "FAIL: C compiler '${CC}' not found -- cannot measure"; exit 1
fi
if ! command -v "${CXX}" >/dev/null 2>&1; then
  echo "FAIL: C++ compiler '${CXX}' not found -- cannot measure"; exit 1
fi

ran_link=0
for l in tensor image codec decoder tracker; do
  h="crates/${l}-capi/include/edgefirst/${l}.h"
  if [ ! -s "${h}" ] || [ ! -r "${h}" ]; then
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
  if [ ! -f "${src}" ]; then
    echo "FAIL ${l}: no ${src} -- link-and-run cannot be measured for this library"
    fails=$((fails+1))
    continue
  fi
  so="${LIBDIR}/libedgefirst_${l}.so"
  if [ ! -s "${so}" ] || [ ! -r "${so}" ]; then
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

# The gated count (`fails`) is legitimately 0 in the passing state, so it
# cannot double as proof the link-and-run step ran at all -- a loop body that
# silently skipped every library (e.g. every `[ -f "${src}" ]` check failing
# open) would also leave `fails` at 0. `ran_link` is independent of pass/fail
# outcome and must be exactly 5 (one per library) for this run to count.
if [ "${ran_link}" -ne 5 ]; then
  echo "FAIL: only ${ran_link}/5 libraries were actually linked and run -- cannot measure the rest"
  fails=$((fails+1))
fi

[ "${fails}" -eq 0 ] && { echo "ALL HEADERS COMPILE AND ALL LIBRARIES RUN"; exit 0; }
echo "${fails} FAILURE(S)"; exit 1
