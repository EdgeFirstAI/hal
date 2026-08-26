#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Cross-build the Rust test binaries AND the five modular C-API libraries
# (+ the G3 two-library-user C binary), deploy both to SSH hosts, and run
# the suite -- plus scripts/check-single-home.sh's G1/G2/G4/G9/G11 -- on
# real hardware.
#
# CI already runs the hardware suite on its own runners. This exists for the
# boards CI does not have: run it against whatever hardware you have on hand,
# which is how you catch behaviour that differs by SoC (older cores taking
# lower-precision SIMD fallbacks, kernels built without DMA-BUF heaps, vendor
# GPU quirks) rather than only on the one runner CI owns.
#
# Hosts are yours to supply — there are no defaults. Anything reachable by
# `ssh <host>` works; put the connection details in ~/.ssh/config.
#
# Usage:
#   ./scripts/on-target-test.sh <ssh-host> [ssh-host...]
#   EDGEFIRST_TARGETS="hostA hostB" ./scripts/on-target-test.sh
#
# Env:
#   EDGEFIRST_TARGETS  space-separated SSH hosts (alternative to arguments)
#   CRATES             cargo packages to test  (default: the five with tests)
#   CAPI_CRATES        the five -capi leaves to build+deploy for G1/G2/G3/G4/
#                      G9/G11 (default: all five; each is its own standalone
#                      workspace, see each crate's own Cargo.toml comment)
#   GLIBC              glibc floor for the build (default: the project floor,
#                      see README "Toolchain and Platform Floors")
#   FILTER             test-name filter passed to each binary
#   REMOTE_DIR         remote scratch dir      (default: /tmp/hal-ontarget)
#   SYNC_TESTDATA      1 to rsync testdata/    (default: 1)
#
# Exit status is non-zero if any host reported a test failure. A host that is
# unreachable, or that lacks the hardware a test needs, is reported separately
# and does NOT mask a real failure elsewhere. Same rule for the deployed
# check-single-home.sh: G5 (footprint) and G7 (Miri) only ever make sense on
# the build host and are never deployed, so they correctly read
# cannot_measure on every board -- a named, attributable gap for those two
# gates specifically, not counted as a failure here. G1/G1b/G2/G4/G9/G11 have
# everything they need on a board and a failure there IS counted.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CRATES="${CRATES:-edgefirst-tensor edgefirst-codec edgefirst-image edgefirst-decoder edgefirst-tracker}"
# The five modular C-API leaves (single-tensor-home, task 12 / G8): each is
# its own standalone `[workspace]` (see each Cargo.toml's own comment), so
# these are built by `--manifest-path`, not `-p`, unlike CRATES above.
CAPI_CRATES="${CAPI_CRATES:-tensor-capi image-capi codec-capi decoder-capi tracker-capi}"
# The project's chosen glibc floor, not a probed value: binaries are built
# against it so one set runs on every supported target. Declared alongside the
# MSRV in the README -- change it there and here together.
GLIBC="${GLIBC:-2.35}"
FILTER="${FILTER:-}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/hal-ontarget}"
SYNC_TESTDATA="${SYNC_TESTDATA:-1}"
RESULTS="${ROOT}/target/on-target-results"

if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
elif [[ -n "${EDGEFIRST_TARGETS:-}" ]]; then
  read -r -a TARGETS <<< "${EDGEFIRST_TARGETS}"
else
  cat >&2 <<'USAGE'
error: no SSH hosts given.

  ./scripts/on-target-test.sh <ssh-host> [ssh-host...]
  EDGEFIRST_TARGETS="hostA hostB" ./scripts/on-target-test.sh

Any host reachable by `ssh <host>` works; put connection details in
~/.ssh/config. Hosts are deliberately not hard-coded — this script has no
knowledge of any particular board.
USAGE
  exit 2
fi

ssh_q() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$@"; }

mkdir -p "${RESULTS}"

# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
# Which architectures to build, and what hardware each host has, are discovered
# from the hosts themselves, so adding a board is purely a matter of naming it
# on the command line. The glibc floor is NOT discovered -- it is a declared
# project floor (see above).
declare -a OK_HOSTS=() OK_ARCH=() OK_CAPS=()
declare -a SUMMARY=()

echo "==> probing ${#TARGETS[@]} host(s)"
for target in "${TARGETS[@]}"; do
  if ! ssh_q "${target}" true 2>/dev/null; then
    printf '    %-20s unreachable\n' "${target}"
    SUMMARY+=("${target}|UNREACHABLE|-|-")
    continue
  fi
  info="$(ssh_q "${target}" '
    echo "arch=$(uname -m)"
    h=$(ls /dev/dma_heap 2>/dev/null | tr "\n" "," )
    echo "caps=dma_heap=${h:-none} render=$(ls /dev/dri 2>/dev/null | grep -c render) galcore=$([ -e /dev/galcore ] && echo yes || echo no) neutron=$([ -e /dev/neutron0 ] && echo yes || echo no)"
  ')"
  arch="$(sed -n 's/^arch=//p' <<< "${info}")"
  caps="$(sed -n 's/^caps=//p' <<< "${info}")"
  case "${arch}" in
    aarch64|x86_64) ;;
    *) printf '    %-20s unsupported arch %s\n' "${target}" "${arch}"
       SUMMARY+=("${target}|BADARCH|${arch}|-"); continue ;;
  esac
  printf '    %-20s %-8s %s\n' "${target}" "${arch}" "${caps}"
  OK_HOSTS+=("${target}"); OK_ARCH+=("${arch}"); OK_CAPS+=("${caps}")
done

if [[ ${#OK_HOSTS[@]} -eq 0 ]]; then
  echo "no usable hosts" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
# Binaries are located from cargo's JSON output rather than by globbing for a
# `<hash>` suffix. Globbing picks up stale binaries from previous builds, which
# is how you end up confidently testing code you did not just change.
build_for() {
  local triple="$1" listfile="$2"
  local pkgs=() c
  for c in ${CRATES}; do pkgs+=(-p "$c"); done

  echo "==> building tests for ${triple}"
  local json="${RESULTS}/build-${triple//[.\/]/_}.json"
  if ! cargo-zigbuild test --no-run --release --target "${triple}" \
        "${pkgs[@]}" --message-format=json > "${json}" 2>"${json}.err"; then
    echo "BUILD FAILED for ${triple}; last lines of stderr:" >&2
    tail -30 "${json}.err" >&2
    return 1
  fi

  python3 - "${json}" "${listfile}" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
out = []
for line in open(src):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        m = json.loads(line)
    except json.JSONDecodeError:
        continue
    # A test binary is a compiler-artifact with profile.test set and a
    # non-null executable. Doctest and build-script artifacts have neither.
    if m.get("reason") == "compiler-artifact" and m.get("executable") \
       and m.get("profile", {}).get("test"):
        out.append(m["executable"])
open(dst, "w").write("\n".join(sorted(set(out))) + "\n")
print(f"    {len(set(out))} test binaries")
PY
}

list_for_arch() { echo "${RESULTS}/bins-$1.txt"; }

# The five C-API libraries + the G3 two-library-user binary (single-tensor-
# home, task 12 / G8): `on-target-test.sh` deployed Rust test binaries only
# until now. G1/G2 (embedded-symbol/dynamic-link inspection,
# scripts/check-single-home.sh) and G3 (the two-library JPEG user) are the
# ones worth re-verifying per board, under that board's own toolchain --
# a compiler's dead-code elimination, symbol visibility, and linker defaults
# can all differ by target in ways a single x86_64 CI runner cannot surface.
#
# DEBUG, not release: `check-single-home.sh`'s own G1 needs an unstripped
# `.so` to see `static_backend` symbols at all -- release now ships with
# `strip = true` (see that script's own `BASELINE_BYTES` comment). No
# `target/release` is deployed at all, so G5 (footprint, a release-only,
# host-specific metric that is not meaningful cross-architecture) correctly
# reads `cannot_measure` on every board -- a named, attributable gap, not a
# regression. Same reasoning for G7 (Miri): `scripts/miri.sh` is not
# deployed, since Miri only ever runs on the build host.
build_libs_for() {
  local triple="$1" libdir="$2" arch="$3"
  echo "==> building the five C-API libraries + G3 for ${triple}"
  # `libdir` MUST be the repo's own plain `target` dir, not a scratch
  # directory of this script's own choosing -- every sibling leaf's
  # build.rs computes its `-L` search path for libedgefirst_tensor.so as a
  # FIXED `<crate_dir>/../../target[/<TARGET-triple>]/<profile>`, derived
  # from CARGO_MANIFEST_DIR, not from whatever `--target-dir` the build was
  # invoked with. A custom `--target-dir` here would build real, correct
  # libraries that the very next sibling's build.rs then fails to find (or
  # worse, silently finds a stale HOST-architecture one already sitting in
  # target/debug from an unrelated local build -- see that comment in each
  # build.rs for the "incompatible with aarch64linux" failure this caused
  # the first time). Cross vs. native output does not collide: cargo nests
  # cross-compiled artifacts under target/<TARGET-triple>/<profile>/, never
  # bare target/<profile>/, so this build never touches a local x86_64
  # target/debug or target/release.
  local c
  for c in ${CAPI_CRATES}; do
    local blog="${RESULTS}/libbuild-${c}-${triple//[.\/]/_}.log"
    if ! cargo-zigbuild build --target "${triple}" \
          --manifest-path "crates/${c}/Cargo.toml" --target-dir "${libdir}" \
          > "${blog}" 2>&1; then
      echo "BUILD FAILED for ${c} (${triple}); last lines of ${blog}:" >&2
      tail -30 "${blog}" >&2
      return 1
    fi
  done

  # build.rs sets `-soname libedgefirst_X.so.0`, but cargo only ever writes
  # the unversioned name -- the same gap `make capi-symlinks` closes locally,
  # and without it no C binary can resolve the library via its DT_NEEDED
  # entry (see that Makefile target's own comment for why this was never
  # noticed before this branch added the first C linker of these libraries).
  #
  # `${triple%%.*}`, not `${triple}`: cargo-zigbuild's glibc-version suffix
  # (the `.2.35` in `aarch64-unknown-linux-gnu.2.35`) selects which zig
  # sysroot to link against, but is never part of the on-disk directory
  # cargo actually creates -- that stays the bare rustc target triple,
  # `target/<arch>-unknown-linux-gnu/debug`. Found the hard way: the first
  # run of this function looked in `target/<triple-with-suffix>/debug`,
  # found nothing (`ln: No such file or directory`), and the G3 build
  # failed with a `strategy 'paths_first': searched paths: none` -- the
  # five libraries had built correctly all along, one directory over.
  local outdir="${libdir}/${triple%%.*}/debug"
  local l
  for l in tensor image codec decoder tracker; do
    ln -sf "libedgefirst_${l}.so" "${outdir}/libedgefirst_${l}.so.0"
  done

  # G3: cross-compile the two-library-user C test the same way `make
  # test-two-library-user` builds it locally (see that Makefile target),
  # via zig's own `cc` -- there is no system cross-gcc installed for every
  # board's architecture, but zig is already the linker cargo-zigbuild uses
  # for the Rust side, so it needs no extra toolchain.
  local g3log="${RESULTS}/libbuild-g3-${triple//[.\/]/_}.log"
  if ! zig cc -std=c11 -Wall -Wextra -target "${arch}-linux-gnu.${GLIBC}" \
        -o "${outdir}/test_two_library_user" \
        crates/codec-capi/tests/c/test_two_library_user.c \
        -Icrates/codec-capi/include -Icrates/tensor-capi/include \
        -L"${outdir}" -ledgefirst_codec -ledgefirst_tensor \
        -Wl,-rpath,'$ORIGIN' \
        > "${g3log}" 2>&1; then
    echo "BUILD FAILED for the G3 binary (${triple}); last lines of ${g3log}:" >&2
    tail -30 "${g3log}" >&2
    return 1
  fi
}

# Always the plain repo `target` dir (see build_libs_for's own comment for
# why) -- kept as a function, matching list_for_arch's shape, rather than a
# bare constant, so both build steps read symmetrically at their call sites.
list_libdir_for_arch() { echo "target"; }

for arch in aarch64 x86_64; do
  needed=0
  for a in "${OK_ARCH[@]}"; do [[ "$a" == "${arch}" ]] && needed=1; done
  [[ ${needed} -eq 1 ]] || continue
  build_for "${arch}-unknown-linux-gnu.${GLIBC}" "$(list_for_arch "${arch}")" || exit 1
  build_libs_for "${arch}-unknown-linux-gnu.${GLIBC}" "$(list_libdir_for_arch "${arch}")" "${arch}" || exit 1
done

# ---------------------------------------------------------------------------
# Deploy + run
# ---------------------------------------------------------------------------
overall=0

for i in "${!OK_HOSTS[@]}"; do
  target="${OK_HOSTS[$i]}"; arch="${OK_ARCH[$i]}"; caps="${OK_CAPS[$i]}"
  echo
  echo "============================================================"
  echo "==> ${target}   (${arch})"
  echo "    ${caps}"
  echo "============================================================"

  listfile="$(list_for_arch "${arch}")"

  # Clear previous logs: binaries carry a content hash, so a rebuilt binary
  # writes a NEW log name and the old one lingers. Tallying the directory
  # afterwards would then count a stale run as if it were this one.
  out_dir="${RESULTS}/${target//[^A-Za-z0-9._-]/_}"
  rm -rf "${out_dir}"; mkdir -p "${out_dir}"
  echo "${caps}" > "${out_dir}/capabilities.txt"

  # Same hazard remotely: drop stale binaries so a renamed or deleted test
  # cannot keep running from a previous deploy.
  ssh_q "${target}" "rm -rf '${REMOTE_DIR}/bin' && mkdir -p '${REMOTE_DIR}/bin'"

  # Read with a loop rather than `mapfile`: macOS ships bash 3.2, and the
  # build host is usually a Mac.
  bins=()
  while IFS= read -r line; do
    [[ -n "${line}" ]] && bins+=("${line}")
  done < "${listfile}"

  # rsync, not scp: testdata/ is large and mostly unchanged run to run.
  rsync -az --info=none "${bins[@]}" "${target}:${REMOTE_DIR}/bin/" || {
    echo "SKIP: binary sync failed"; SUMMARY+=("${target}|SYNCFAIL|${arch}|-"); continue; }

  if [[ "${SYNC_TESTDATA}" == "1" && -d "${ROOT}/testdata" ]]; then
    echo "    syncing testdata"
    rsync -az --delete --info=none "${ROOT}/testdata/" "${target}:${REMOTE_DIR}/testdata/" || {
      echo "SKIP: testdata sync failed"; SUMMARY+=("${target}|SYNCFAIL|${arch}|-"); continue; }
  fi

  # Deploy the five C-API libraries + their `.so.0` symlinks + the G3
  # binary, `scripts/check-single-home.sh` itself, and the minimal source
  # subset that script's own gates read directly: `crates/*-capi/src` (G4's
  # grep for `EfTensorVtable`/`is_own_mint`) and `crates/*-capi/include`
  # (G9/G11's headers). Laid out under `${REMOTE_DIR}` to match the same
  # `target/debug/...` + `crates/...` + `scripts/...` shape
  # `check-single-home.sh` expects locally, so the script runs unmodified.
  # No glibc-version suffix here either -- same reason as build_libs_for's
  # own `outdir` (see its comment): that suffix never reaches the on-disk
  # directory name.
  libdir="$(list_libdir_for_arch "${arch}")/${arch}-unknown-linux-gnu/debug"
  if [[ ! -d "${libdir}" ]]; then
    echo "SKIP: no library build for ${arch}"
    SUMMARY+=("${target}|SYNCFAIL|${arch}|no ${arch} library build"); continue
  fi
  ssh_q "${target}" "mkdir -p '${REMOTE_DIR}/target/debug' '${REMOTE_DIR}/scripts'"
  rsync -az --info=none "${libdir}"/libedgefirst_*.so "${libdir}"/libedgefirst_*.so.0 \
        "${libdir}/test_two_library_user" "${target}:${REMOTE_DIR}/target/debug/" || {
    echo "SKIP: library sync failed"; SUMMARY+=("${target}|SYNCFAIL|${arch}|-"); continue; }
  rsync -az --info=none "${ROOT}/scripts/check-single-home.sh" "${target}:${REMOTE_DIR}/scripts/" || {
    echo "SKIP: check-single-home.sh sync failed"; SUMMARY+=("${target}|SYNCFAIL|${arch}|-"); continue; }
  # `--relative` + a `./` marker keeps `crates/<leaf>/{src,include}` as the
  # path stored remotely, from a single rsync invocation, rather than five
  # separate ones. The marker must sit BEFORE `crates`, not between
  # `crates/<leaf>` and `{src,include}`: rsync's `--relative` strips
  # everything up to and including the `/./` and keeps only what follows it,
  # so `crates/tensor-capi/./include` would deploy to `$REMOTE_DIR/include/`
  # -- dropping the `crates/tensor-capi/` prefix `check-single-home.sh`
  # expects -- rather than `$REMOTE_DIR/crates/tensor-capi/include/`.
  # shellcheck disable=SC2086
  rsync -az --info=none --relative \
        ./crates/tensor-capi/src ./crates/tensor-capi/include \
        ./crates/image-capi/src ./crates/image-capi/include \
        ./crates/codec-capi/src ./crates/codec-capi/include \
        ./crates/decoder-capi/src ./crates/decoder-capi/include \
        ./crates/tracker-capi/src ./crates/tracker-capi/include \
        "${target}:${REMOTE_DIR}/" || {
    echo "SKIP: source sync failed"; SUMMARY+=("${target}|SYNCFAIL|${arch}|-"); continue; }

  # Per-host environment is derived from what the probe FOUND, not from the
  # host's name — so it applies to any board with the same hardware. Vivante
  # GPUs have an intermittent driver double-free that otherwise masquerades as
  # a regression in whatever you just changed.
  extra_env=""
  [[ "${caps}" == *"galcore=yes"* ]] && extra_env="EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS=1"

  pass=0; fail=0; failed_bins=()
  for bin in "${bins[@]}"; do
    name="$(basename "${bin}")"
    echo "  -- ${name}"
    log="${out_dir}/${name}.log"
    # --test-threads=1 is a hard invariant on target: GL driver concurrency
    # bugs, per-process G2D state, and CMA pool exhaustion each require it.
    if ssh_q "${target}" \
        "cd '${REMOTE_DIR}' && EDGEFIRST_TESTDATA_DIR='${REMOTE_DIR}/testdata' \
         ${extra_env} ./bin/${name} --test-threads=1 ${FILTER}" \
         >"${log}" 2>&1; then
      pass=$((pass + 1))
      # Surface skips: a hardware-gated test that returns early still reports
      # "ok", so the count alone cannot tell you whether the path ran.
      if grep -q "SKIPPED" "${log}"; then
        echo "     ok, but $(grep -c "SKIPPED" "${log}") skipped"
      fi
    else
      fail=$((fail + 1)); failed_bins+=("${name}")
      echo "     FAILED (see ${log})"
      grep -E "^(test .* FAILED|failures:|error)" "${log}" | head -8 | sed 's/^/     /'
    fi
  done

  # G1/G2/G4/G9/G11 via scripts/check-single-home.sh, deployed above.
  #
  # G5 (footprint) and G7 (Miri) correctly read `cannot_measure` here --
  # no target/release build and no scripts/miri.sh were shipped to this
  # board, on purpose (see the deploy comment above). That is this run's
  # own named, attributable gap for those two gates specifically, not a
  # regression, so a `cannot_measure` on ONLY G5/G7 does not count as a
  # failure below; a FAIL (real or cannot_measure) on any of G1/G1b/G2/G4/
  # G9/G11 does -- those gates have everything they need on this board and
  # a failure there is exactly what this run exists to catch.
  echo "  -- check-single-home.sh (G1/G2/G4/G9/G11)"
  ghlog="${out_dir}/check-single-home.log"
  ssh_q "${target}" "cd '${REMOTE_DIR}' && ./scripts/check-single-home.sh" >"${ghlog}" 2>&1
  gh_real_fail="$(grep -cE '^\s*(G1|G1b|G2|G4|G9|G11)\s+FAIL' "${ghlog}")"
  gh_expected_gap="$(grep -cE '^\s*(G5|G7)\s+FAIL' "${ghlog}")"
  if [[ "${gh_real_fail}" -eq 0 ]]; then
    pass=$((pass + 1))
    echo "     ok -- G1/G1b/G2/G4/G9/G11 clean; ${gh_expected_gap} expected gap(s) (G5/G7 need the build host, see ${ghlog})"
  else
    fail=$((fail + 1)); failed_bins+=("check-single-home.sh")
    echo "     FAILED: ${gh_real_fail} real gate failure(s) (see ${ghlog})"
    grep -E 'FAIL' "${ghlog}" | sed 's/^/     /'
  fi

  # G3: the minimal two-library user, same binary `make test-two-library-
  # user` runs locally, cross-built above and run from testdata's own
  # directory the same way the Rust suites are.
  echo "  -- test_two_library_user (G3)"
  g3log="${out_dir}/test_two_library_user.log"
  if ssh_q "${target}" \
      "cd '${REMOTE_DIR}' && ${extra_env} ./target/debug/test_two_library_user" \
      >"${g3log}" 2>&1; then
    pass=$((pass + 1))
  else
    fail=$((fail + 1)); failed_bins+=("test_two_library_user")
    echo "     FAILED (see ${g3log})"
    cat "${g3log}" | sed 's/^/     /'
  fi

  skipped="$(cat "${out_dir}"/*.log 2>/dev/null | grep -c "SKIPPED")"
  if [[ ${fail} -gt 0 ]]; then
    SUMMARY+=("${target}|FAIL|${arch}|${pass} ok, ${fail} failed: ${failed_bins[*]}")
    overall=1
  else
    SUMMARY+=("${target}|PASS|${arch}|${pass} binaries, ${skipped} tests skipped, G1/G2/G3/G4/G9/G11 clean")
  fi
done

# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------
echo
echo "============================================================"
printf "%-20s %-12s %-8s %s\n" "HOST" "RESULT" "ARCH" "DETAIL"
echo "------------------------------------------------------------"
for row in "${SUMMARY[@]}"; do
  IFS='|' read -r b r a d <<< "${row}"
  printf "%-20s %-12s %-8s %s\n" "$b" "$r" "$a" "$d"
done
echo "============================================================"
echo "A skipped test is not a passed test. Check each host's"
echo "capabilities.txt to attribute skips to a missing device node."
echo "logs: ${RESULTS}"
exit ${overall}
