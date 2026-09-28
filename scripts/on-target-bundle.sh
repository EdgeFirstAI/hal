#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Cross-build a self-contained on-target bundle: everything a board needs to
# run HAL's full test suite or its benchmarks, and nothing it has to build.
# scripts/on-target-test.sh deploys one over ssh; the hardware-test and
# hardware-bench workflows ship one to each board as an artifact. Both then run
# the same board-side script from inside it, so what runs on a board does not
# depend on how it got there.
#
# Usage:
#   ./scripts/on-target-bundle.sh tests <arch> <out-dir>
#   ./scripts/on-target-bundle.sh bench <arch> <out-dir>
#
# <arch> is aarch64 or x86_64. <out-dir> is replaced.
#
# tests bundle:
#   bin/                  the Rust test binaries of CRATES
#   target/debug/         the five C-API libraries, their .so.0 names, and
#                         the G3 two-library-user binary
#   crates/*-capi/        src/ and include/, which check-single-home.sh reads
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
#   CRATES        tests: cargo packages to test (default: the five with tests)
#   CAPI_CRATES   tests: the -capi leaves to build (default: all five)
#   FEATURES      tests: cargo features for the test build, space- or comma-
#                 separated; `<pkg>/<feature>` when more than one package is
#                 selected (default: none)
#   GLIBC         glibc floor the binaries link against (default: 2.35, the
#                 project floor; see README "Toolchain and Platform Floors")
#   TESTDATA      1 to include testdata/ (default: 1). scripts/on-target-test.sh
#                 passes its SYNC_TESTDATA here.

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

CRATES="${CRATES:-edgefirst-tensor edgefirst-codec edgefirst-image edgefirst-decoder edgefirst-tracker}"
# Each -capi leaf is its own standalone `[workspace]` (see each Cargo.toml's
# own comment), so these are built by `--manifest-path`, not `-p`.
CAPI_CRATES="${CAPI_CRATES:-tensor-capi image-capi codec-capi decoder-capi tracker-capi}"
GLIBC="${GLIBC:-2.35}"
FEATURES="${FEATURES:-}"
TESTDATA="${TESTDATA:-1}"
TRIPLE="${ARCH}-unknown-linux-gnu.${GLIBC}"
LOGS="${ROOT}/target/on-target-bundle-logs"
mkdir -p "${LOGS}"

# Executables of one artifact kind from cargo's JSON output. Located this way
# rather than by globbing for a `<hash>` suffix: globbing picks up stale
# binaries from previous builds, which is how you end up confidently testing
# code you did not just change.
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
  # Empty FEATURES must not become a bare `--features ''`, which cargo reads
  # as a request for a feature named "" and rejects.
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
  # The Makefile's `bench` feature set. The python crates are excluded because
  # they default to edgefirst-tensor's `dynamic` backend while everything else
  # defaults to `static`, and one invocation selecting both cannot compile.
  if ! cargo-zigbuild zigbuild --release --benches --target "${TRIPLE}" \
        --features opengl,ndarray --workspace \
        --exclude edgefirst-python-common --exclude edgefirst-python-tensor \
        --exclude edgefirst-python-codec --exclude edgefirst-python-image \
        --exclude edgefirst-python-decoder --exclude edgefirst-python-tracker \
        --message-format=json > "${json}" 2>"${json}.err"; then
    echo "BUILD FAILED for ${TRIPLE}; last lines of stderr:" >&2
    tail -30 "${json}.err" >&2
    return 1
  fi
  executables "${json}" bench > "${LOGS}/bench-${ARCH}.txt"
  echo "    $(grep -c . "${LOGS}/bench-${ARCH}.txt") benchmark binaries"
  return 0
}

# The five C-API libraries + the G3 two-library-user binary. G1/G2
# (embedded-symbol/dynamic-link inspection, scripts/check-single-home.sh) and
# G3 (the two-library JPEG user) are the ones worth re-verifying per board,
# under that board's own toolchain -- a compiler's dead-code elimination,
# symbol visibility, and linker defaults can all differ by target in ways a
# single x86_64 CI runner cannot surface.
#
# DEBUG, not release: `check-single-home.sh`'s own G1 needs an unstripped
# `.so` to see `static_backend` symbols at all -- release now ships with
# `strip = true` (see that script's own `BASELINE_BYTES` comment). No
# `target/release` is bundled at all, so G5 (footprint, a release-only,
# host-specific metric that is not meaningful cross-architecture) correctly
# reads `cannot_measure` on every board -- a named, attributable gap, not a
# regression. Same reasoning for G7 (Miri): `scripts/miri.sh` is not
# bundled, since Miri only ever runs on the build host.
build_libs() {
  echo "==> building the five C-API libraries + G3 for ${TRIPLE}"
  # `--target-dir` MUST be the repo's own plain `target` dir, not a scratch
  # directory of this script's own choosing -- every sibling leaf's
  # build.rs computes its `-L` search path for libedgefirst_tensor.so as a
  # FIXED `<crate_dir>/../../target[/<TARGET-triple>]/<profile>`, derived
  # from CARGO_MANIFEST_DIR, not from whatever `--target-dir` the build was
  # invoked with. A custom `--target-dir` here would build real, correct
  # libraries that the very next sibling's build.rs then fails to find (or
  # worse, silently finds a stale HOST-architecture one already sitting in
  # target/debug from an unrelated local build). Cross vs. native output does
  # not collide: cargo nests cross-compiled artifacts under
  # target/<TARGET-triple>/<profile>/, never bare target/<profile>/.
  local c
  for c in ${CAPI_CRATES}; do
    local blog="${LOGS}/libbuild-${c}-${ARCH}.log"
    if ! cargo-zigbuild build --target "${TRIPLE}" \
          --manifest-path "crates/${c}/Cargo.toml" --target-dir target \
          > "${blog}" 2>&1; then
      echo "BUILD FAILED for ${c} (${TRIPLE}); last lines of ${blog}:" >&2
      tail -30 "${blog}" >&2
      return 1
    fi
  done

  # build.rs sets `-soname libedgefirst_X.so.0`, but cargo only ever writes
  # the unversioned name -- the same gap `make capi-symlinks` closes locally,
  # and without it no C binary can resolve the library via its DT_NEEDED
  # entry.
  #
  # The bare rustc triple, not ${TRIPLE}: cargo-zigbuild's glibc-version
  # suffix (the `.2.35`) selects which zig sysroot to link against, but is
  # never part of the on-disk directory cargo creates.
  local outdir="target/${ARCH}-unknown-linux-gnu/debug"
  local l
  for l in tensor image codec decoder tracker; do
    ln -sf "libedgefirst_${l}.so" "${outdir}/libedgefirst_${l}.so.0"
  done

  # G3: cross-compile the two-library-user C test the same way `make
  # test-two-library-user` builds it locally, via zig's own `cc` -- zig is
  # already the linker cargo-zigbuild uses for the Rust side, so it needs no
  # extra toolchain.
  local g3log="${LOGS}/libbuild-g3-${ARCH}.log"
  if ! zig cc -std=c11 -Wall -Wextra -target "${ARCH}-linux-gnu.${GLIBC}" \
        -o "${outdir}/test_two_library_user" \
        crates/codec-capi/tests/c/test_two_library_user.c \
        -Icrates/codec-capi/include -Icrates/tensor-capi/include \
        -L"${outdir}" -ledgefirst_codec -ledgefirst_tensor \
        -Wl,-rpath,'$ORIGIN' \
        > "${g3log}" 2>&1; then
    echo "BUILD FAILED for the G3 binary (${TRIPLE}); last lines of ${g3log}:" >&2
    tail -30 "${g3log}" >&2
    return 1
  fi
  return 0
}

# testdata/ plus every crates/*/testdata merged into one tree: every
# EDGEFIRST_TESTDATA_DIR-aware fixture loader resolves relative to one root.
# A crate file that would overwrite another is an error, as in ci-setup.sh.
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
        "${libdir}/test_two_library_user" "${OUT}/target/debug/"
  for c in ${CAPI_CRATES}; do
    mkdir -p "${OUT}/crates/${c}"
    cp -a "crates/${c}/src" "crates/${c}/include" "${OUT}/crates/${c}/"
  done
  cp scripts/on-target-run.sh scripts/check-single-home.sh \
     .github/scripts/dma-heap-setup.sh "${OUT}/scripts/"
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
