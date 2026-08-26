#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Extract a packaged edgefirst-hal tarball and compile+run one consumer per
# library using ONLY pkg-config (no source-tree -I). This is the test that
# catches a missing detect.h, a broken .pc Requires line, or a SONAME that
# does not resolve at runtime.
#
# Also:
#   - compiles the C example extracted from the packaged INSTALL.txt so the
#     two cannot drift
#   - links each library with -Wl,-Bstatic around the edgefirst -l flags
#     (Linux) or the explicit .a files (macOS), proving Libs.private
#
# Usage: scripts/smoke-capi-archive.sh <archive.tar.gz>
set -euo pipefail
ARCHIVE="${1:?usage: $0 <archive.tar.gz>}"
CC="${CC:-cc}"

if [[ ! -f "${ARCHIVE}" ]]; then
  echo "FAIL: ${ARCHIVE} not found" >&2
  exit 1
fi
if ! command -v "${CC}" >/dev/null 2>&1; then
  echo "FAIL: C compiler '${CC}' not found" >&2
  exit 1
fi
if ! command -v pkg-config >/dev/null 2>&1; then
  echo "FAIL: pkg-config not found -- cannot measure the packaged layout" >&2
  exit 1
fi

WORKDIR="${TMPDIR:-/tmp}/smoke-capi.$$"
mkdir -p "${WORKDIR}"
trap 'rm -rf "${WORKDIR}"' EXIT
tar xzf "${ARCHIVE}" -C "${WORKDIR}"
# The archive has exactly one top-level directory.
PREFIX="$(find "${WORKDIR}" -mindepth 1 -maxdepth 1 -type d | head -1)"
if [[ -z "${PREFIX}" ]] || [[ ! -d "${PREFIX}/lib/pkgconfig" ]]; then
  echo "FAIL: archive has no lib/pkgconfig — cannot measure" >&2
  exit 1
fi

export PKG_CONFIG_PATH="${PREFIX}/lib/pkgconfig"
export LD_LIBRARY_PATH="${PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export DYLD_LIBRARY_PATH="${PREFIX}/lib${DYLD_LIBRARY_PATH:+:${DYLD_LIBRARY_PATH}}"

uname_s="$(uname -s)"

ran=0
for leaf in tensor image codec decoder tracker; do
  src="${WORKDIR}/${leaf}.c"
  bin="${WORKDIR}/${leaf}.bin"
  printf '#include <edgefirst/%s.h>\nint main(void){return 0;}\n' "${leaf}" > "${src}"
  # shellcheck disable=SC2046
  if ! "${CC}" -std=c11 -Wall -Wextra -Werror -o "${bin}" "${src}" \
      $(pkg-config --cflags --libs "edgefirst-${leaf}"); then
    echo "FAIL: compile ${leaf} via pkg-config" >&2
    exit 1
  fi
  if ! "${bin}"; then
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
echo "ALL PACKAGED HEADERS COMPILE AND ALL LIBRARIES RUN"

# ---------------------------------------------------------------------------
# INSTALL.txt example — extract the C block and compile it.
# ---------------------------------------------------------------------------
install_txt="${PREFIX}/INSTALL.txt"
if [[ ! -f "${install_txt}" ]]; then
  echo "FAIL: packaged INSTALL.txt missing" >&2
  exit 1
fi
example_c="${WORKDIR}/install_example.c"
awk '
  /^  #include <edgefirst\/codec.h>/ { p=1 }
  p {
    line = $0
    sub(/^  /, "", line)
    print line
  }
  p && /^  }$/ { exit }
' "${install_txt}" > "${example_c}"
if ! grep -q 'ef_tensor_image_alloc' "${example_c}"; then
  echo "FAIL: could not extract the INSTALL.txt C example" >&2
  cat "${example_c}" >&2
  exit 1
fi
example_bin="${WORKDIR}/install_example.bin"
# shellcheck disable=SC2046
if ! "${CC}" -std=c11 -Wall -Wextra -Werror -o "${example_bin}" "${example_c}" \
    $(pkg-config --cflags --libs edgefirst-codec); then
  echo "FAIL: INSTALL.txt example does not compile" >&2
  exit 1
fi
echo "INSTALL example compiles"

# ---------------------------------------------------------------------------
# Static link: force the edgefirst archives, leave system libs dynamic.
# ---------------------------------------------------------------------------
split_pkg_libs() {
  # Sets ef_libs ( -ledgefirst_* ) and sys_libs (the rest of --libs-only-l).
  local leaf="$1"
  ef_libs=""
  sys_libs=""
  local tok
  # shellcheck disable=SC2046
  for tok in $(pkg-config --static --libs-only-l "edgefirst-${leaf}"); do
    case "${tok}" in
      -ledgefirst_*) ef_libs="${ef_libs} ${tok}" ;;
      *) sys_libs="${sys_libs} ${tok}" ;;
    esac
  done
}

static_ran=0
for leaf in tensor image codec decoder tracker; do
  src="${WORKDIR}/${leaf}.c"
  bin="${WORKDIR}/${leaf}.static.bin"
  # shellcheck disable=SC2046
  cflags="$(pkg-config --static --cflags "edgefirst-${leaf}")"
  libdirs="$(pkg-config --static --libs-only-L "edgefirst-${leaf}")"
  split_pkg_libs "${leaf}"
  if [[ "${uname_s}" == Darwin ]]; then
    a_files=""
    for tok in ${ef_libs}; do
      name="${tok#-l}"
      a="${PREFIX}/lib/lib${name}.a"
      if [[ ! -f "${a}" ]]; then
        echo "FAIL: missing static archive ${a}" >&2
        exit 1
      fi
      a_files="${a_files} ${a}"
    done
    # shellcheck disable=SC2086
    if ! "${CC}" -std=c11 -Wall -Wextra -Werror -o "${bin}" "${src}" \
        ${cflags} ${a_files} ${sys_libs}; then
      echo "FAIL: static compile ${leaf}" >&2
      exit 1
    fi
    if command -v otool >/dev/null 2>&1; then
      if otool -L "${bin}" | grep -q 'libedgefirst_'; then
        echo "FAIL: ${leaf} still dynamically linked to an edgefirst lib" >&2
        otool -L "${bin}" >&2
        exit 1
      fi
    fi
  else
    # shellcheck disable=SC2086
    if ! "${CC}" -std=c11 -Wall -Wextra -Werror -o "${bin}" "${src}" \
        ${cflags} ${libdirs} -Wl,-Bstatic ${ef_libs} -Wl,-Bdynamic ${sys_libs}; then
      echo "FAIL: static compile ${leaf}" >&2
      exit 1
    fi
    if command -v ldd >/dev/null 2>&1; then
      if ldd "${bin}" | grep -q edgefirst; then
        echo "FAIL: ${leaf} still dynamically linked to an edgefirst lib" >&2
        ldd "${bin}" >&2
        exit 1
      fi
    fi
  fi
  if ! "${bin}"; then
    echo "FAIL: run static ${leaf}" >&2
    exit 1
  fi
  static_ran=$((static_ran + 1))
  echo "ok static ${leaf}"
done

if [[ "${static_ran}" -ne 5 ]]; then
  echo "FAIL: only ${static_ran}/5 libraries static-smoked" >&2
  exit 1
fi
echo "ALL PACKAGED LIBRARIES LINK TRULY STATIC"
