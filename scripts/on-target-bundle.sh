#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Cross-builds a self-contained on-target bundle: everything a board needs to
# run HAL's test suite or its benchmarks. scripts/on-target-test.sh deploys
# one over ssh; the hardware-test and hardware-bench workflows ship one to
# each board as an artifact.
#
# Usage:
#   ./scripts/on-target-bundle.sh tests <arch> <out-dir>
#   ./scripts/on-target-bundle.sh bench <arch> <out-dir>
#
# <arch> is aarch64 or x86_64. <out-dir> is replaced.
#
# tests bundle:
#   bin/                  the Rust test binaries of CRATES
#   target/debug/         the C-API libraries, their .so.0 names,
#                         test_two_library_user and test_codec_api
#   crates/*-capi/        src/ and include/, which check-single-home.sh reads
#   python/, tests/codec  with PYTHON_WHEELS: the unpacked wheels and the
#                         Python codec tests
#   python-cpXY/          with PYTHON_WHEELS: the unpacked wheels built for
#                         one CPython version only, such as numpy
#   scripts/              on-target-run.sh, check-single-home.sh,
#                         dma-heap-setup.sh
#   testdata/             testdata/ merged with every crates/*/testdata
#   MANIFEST              commit, arch, glibc floor, features
# bench bundle:
#   bin/                  the benchmark binaries
#   scripts/              on-target-bench.sh
#   testdata/, MANIFEST   as above
#
# Env:
#   CRATES        tests: cargo packages to test (default: every crate with tests)
#   CAPI_CRATES   tests: the -capi leaves to build (default: all of them)
#   FEATURES      tests: cargo features for the test build, space- or comma-
#                 separated; `<pkg>/<feature>` when more than one package is
#                 selected (default: none)
#   GLIBC         glibc floor the binaries link against (default: 2.35, the
#                 project floor; see README "Toolchain and Platform Floors")
#   TESTDATA      1 to include testdata/ (default: 1). scripts/on-target-test.sh
#                 passes its SYNC_TESTDATA here.
#   PYTHON_WHEELS tests: a directory of aarch64 wheels (edgefirst-tensor,
#                 edgefirst-codec, pytest and its dependencies, and numpy for
#                 each CPython version the boards run) to bundle with the
#                 Python codec tests (default: none, and the board skips them)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

MODE="${1:-}"; ARCH="${2:-}"; OUT="${3:-}"
if [[ -z "${MODE}" || -z "${ARCH}" || -z "${OUT}" ]]; then
  echo "usage: $0 <tests|bench> <aarch64|x86_64> <out-dir>" >&2
  exit 2
fi
case "${MODE}" in tests|bench) ;; *) echo "unknown mode '${MODE}'" >&2; exit 2 ;; esac
case "${ARCH}" in aarch64|x86_64) ;; *) echo "unsupported arch '${ARCH}'" >&2; exit 2 ;; esac

CRATES="${CRATES:-edgefirst-tensor edgefirst-codec edgefirst-image edgefirst-decoder edgefirst-tracker edgefirst-kernels}"
# Each -capi leaf is its own workspace, built by `--manifest-path`.
CAPI_CRATES="${CAPI_CRATES:-tensor-capi image-capi codec-capi decoder-capi tracker-capi kernels-capi}"
GLIBC="${GLIBC:-2.35}"
FEATURES="${FEATURES:-}"
TESTDATA="${TESTDATA:-1}"
PYTHON_WHEELS="${PYTHON_WHEELS:-}"
TRIPLE="${ARCH}-unknown-linux-gnu.${GLIBC}"
LOGS="${ROOT}/target/on-target-bundle-logs"
mkdir -p "${LOGS}"

# Prints the executables of one artifact kind from cargo's JSON output.
executables() {
  local json="$1" kind="$2"
  python3 - "${json}" "${kind}" <<'PY'
import json, sys
src, kind = sys.argv[1], sys.argv[2]
out = set()
for line in open(src):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        m = json.loads(line)
    except json.JSONDecodeError:
        continue
    if m.get("reason") != "compiler-artifact" or not m.get("executable"):
        continue
    # A test binary has profile.test set; doctest and build-script artifacts
    # have no executable. A bench harness is a target of kind "bench".
    if kind == "test" and m.get("profile", {}).get("test") \
       and "bench" not in m.get("target", {}).get("kind", []):
        out.add(m["executable"])
    elif kind == "bench" and "bench" in m.get("target", {}).get("kind", []):
        out.add(m["executable"])
print("\n".join(sorted(out)))
PY
  return 0
}

build_tests() {
  local pkgs=() c
  for c in ${CRATES}; do pkgs+=(-p "$c"); done
  # Passes --features only when FEATURES is set.
  local feats=()
  if [[ -n "${FEATURES}" ]]; then
    feats=(--features "${FEATURES}")
    echo "==> test build features: ${FEATURES}"
  fi
  echo "==> building tests for ${TRIPLE}"
  local json="${LOGS}/tests-${ARCH}.json"
  if ! cargo-zigbuild test --no-run --release --target "${TRIPLE}" \
        "${pkgs[@]}" ${feats[@]+"${feats[@]}"} --message-format=json > "${json}" 2>"${json}.err"; then
    echo "BUILD FAILED for ${TRIPLE}; last lines of stderr:" >&2
    tail -30 "${json}.err" >&2
    return 1
  fi
  executables "${json}" test > "${LOGS}/tests-${ARCH}.txt"
  echo "    $(grep -c . "${LOGS}/tests-${ARCH}.txt") test binaries"
  return 0
}

build_benches() {
  echo "==> building benchmarks for ${TRIPLE}"
  local json="${LOGS}/bench-${ARCH}.json"
  # Builds with the Makefile's `bench` feature set, without the python crates.
  if ! cargo-zigbuild zigbuild --release --benches --target "${TRIPLE}" \
        --features opengl,ndarray --workspace \
        --exclude edgefirst-python-common --exclude edgefirst-python-tensor \
        --exclude edgefirst-python-codec --exclude edgefirst-python-image \
        --exclude edgefirst-python-decoder --exclude edgefirst-python-tracker --exclude edgefirst-python-kernels \
        --message-format=json > "${json}" 2>"${json}.err"; then
    echo "BUILD FAILED for ${TRIPLE}; last lines of stderr:" >&2
    tail -30 "${json}.err" >&2
    return 1
  fi
  executables "${json}" bench > "${LOGS}/bench-${ARCH}.txt"
  echo "    $(grep -c . "${LOGS}/bench-${ARCH}.txt") benchmark binaries"
  return 0
}

# Builds the C-API libraries (debug), test_two_library_user and
# test_codec_api.
build_libs() {
  echo "==> building the C-API libraries and test_two_library_user for ${TRIPLE}"
  # Builds into the repository's `target` directory, where each leaf's
  # build.rs looks for libedgefirst_tensor.so.
  local c
  for c in ${CAPI_CRATES}; do
    local blog="${LOGS}/libbuild-${c}-${ARCH}.log"
    local cfg=()
    case "${c}" in image-capi|codec-capi|decoder-capi) cfg=(--config "${ROOT}/.cargo/tensor-dynamic.toml") ;; esac
    if ! cargo-zigbuild build --target "${TRIPLE}" ${cfg[@]+"${cfg[@]}"} \
          --manifest-path "crates/${c}/Cargo.toml" --target-dir target \
          > "${blog}" 2>&1; then
      echo "BUILD FAILED for ${c} (${TRIPLE}); last lines of ${blog}:" >&2
      tail -30 "${blog}" >&2
      return 1
    fi
  done

  # Creates the `.so.0` names the libraries' sonames refer to, in the bare
  # rustc triple's output directory.
  local outdir="target/${ARCH}-unknown-linux-gnu/debug"
  local l
  for l in tensor image codec decoder tracker kernels; do
    ln -sf "libedgefirst_${l}.so" "${outdir}/libedgefirst_${l}.so.0"
  done

  # Cross-compiles test_two_library_user with zig cc.
  local two_lib_log="${LOGS}/libbuild-two-library-user-${ARCH}.log"
  if ! zig cc -std=c11 -Wall -Wextra -target "${ARCH}-linux-gnu.${GLIBC}" \
        -o "${outdir}/test_two_library_user" \
        crates/codec-capi/tests/c/test_two_library_user.c \
        -Icrates/codec-capi/include -Icrates/tensor-capi/include \
        -L"${outdir}" -ledgefirst_codec -ledgefirst_tensor \
        -Wl,-rpath,'$ORIGIN' \
        > "${two_lib_log}" 2>&1; then
    echo "BUILD FAILED for test_two_library_user (${TRIPLE}); last lines of ${two_lib_log}:" >&2
    tail -30 "${two_lib_log}" >&2
    return 1
  fi

  # Cross-compiles test_codec_api, the codec C API program, with zig cc.
  local codeclog="${LOGS}/libbuild-codec-api-${ARCH}.log"
  if ! zig cc -std=c11 -Wall -Wextra -Werror -target "${ARCH}-linux-gnu.${GLIBC}" \
        -o "${outdir}/test_codec_api" \
        crates/codec-capi/tests/c/test_codec_api.c \
        -Icrates/codec-capi/include -Icrates/tensor-capi/include \
        -L"${outdir}" -ledgefirst_codec -ledgefirst_tensor \
        -Wl,-rpath,'$ORIGIN' \
        > "${codeclog}" 2>&1; then
    echo "BUILD FAILED for test_codec_api (${TRIPLE}); last lines of ${codeclog}:" >&2
    tail -30 "${codeclog}" >&2
    return 1
  fi
  return 0
}

# Unpacks the wheels into python/, and the wheels tagged for one CPython
# version (cp311-cp311) into python-cp311/, and copies the Python codec tests
# into tests/.
bundle_python() {
  local wheels="$1" w dest tag
  echo "==> unpacking the Python wheels from ${wheels}"
  mkdir -p "${OUT}/python" "${OUT}/tests"
  for w in "${wheels}"/*.whl; do
    [[ -e "${w}" ]] || { echo "no wheels in ${wheels}" >&2; return 1; }
    dest="${OUT}/python"
    if [[ "$(basename "${w}")" =~ -(cp3[0-9]+)-(cp3[0-9]+)-[^-]+\.whl$ ]] \
        && [[ "${BASH_REMATCH[1]}" == "${BASH_REMATCH[2]}" ]]; then
      tag="${BASH_REMATCH[1]}"
      dest="${OUT}/python-${tag}"
    fi
    mkdir -p "${dest}"
    python3 -m zipfile -e "${w}" "${dest}"
    echo "    $(basename "${w}")"
  done
  cp tests/__init__.py tests/conftest.py "${OUT}/tests/"
  cp -a tests/codec "${OUT}/tests/"
  return 0
}

# Merges testdata/ and every crates/*/testdata into one tree. A crate file
# that would overwrite another is an error.
merge_testdata() {
  local dst="$1"
  mkdir -p "${dst}"
  [[ -d testdata ]] && cp -a testdata/. "${dst}/"
  local dir f rel
  for dir in crates/*/testdata; do
    [[ -d "${dir}" ]] || continue
    while IFS= read -r -d '' f; do
      rel="${f#"${dir}"/}"
      if [[ -e "${dst}/${rel}" ]]; then
        echo "error: ${dir}/${rel} collides with testdata/${rel}" >&2
        return 1
      fi
    done < <(find "${dir}" -type f -print0)
    cp -a "${dir}/." "${dst}/"
  done
  return 0
}

rm -rf "${OUT}"
mkdir -p "${OUT}/bin" "${OUT}/scripts"

if [[ "${MODE}" == tests ]]; then
  build_tests
  build_libs
  while IFS= read -r bin; do
    [[ -n "${bin}" ]] && cp "${bin}" "${OUT}/bin/"
  done < "${LOGS}/tests-${ARCH}.txt"

  libdir="target/${ARCH}-unknown-linux-gnu/debug"
  mkdir -p "${OUT}/target/debug"
  cp -a "${libdir}"/libedgefirst_*.so "${libdir}"/libedgefirst_*.so.0 \
        "${libdir}/test_two_library_user" "${libdir}/test_codec_api" "${OUT}/target/debug/"
  for c in ${CAPI_CRATES}; do
    mkdir -p "${OUT}/crates/${c}"
    cp -a "crates/${c}/src" "crates/${c}/include" "${OUT}/crates/${c}/"
  done
  cp scripts/on-target-run.sh scripts/check-single-home.sh \
     .github/scripts/dma-heap-setup.sh "${OUT}/scripts/"
  if [[ -n "${PYTHON_WHEELS}" ]]; then
    bundle_python "${PYTHON_WHEELS}"
  fi
else
  build_benches
  while IFS= read -r bin; do
    [[ -n "${bin}" ]] && cp "${bin}" "${OUT}/bin/"
  done < "${LOGS}/bench-${ARCH}.txt"
  cp scripts/on-target-bench.sh "${OUT}/scripts/"
fi

if [[ "${TESTDATA}" == "1" ]]; then
  echo "==> merging testdata"
  merge_testdata "${OUT}/testdata"
fi

{
  echo "commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "dirty=$([[ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ]] && echo yes || echo no)"
  echo "mode=${MODE}"
  echo "arch=${ARCH}"
  echo "glibc=${GLIBC}"
  echo "features=${FEATURES}"
  echo "built=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${OUT}/MANIFEST"

echo "==> bundle ready: ${OUT} ($(du -sh --apparent-size "${OUT}" | cut -f1))"
