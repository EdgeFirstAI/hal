#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Cross-build a tests bundle (scripts/on-target-bundle.sh: the Rust test
# binaries, the five modular C-API libraries and the G3 two-library-user C
# binary), deploy it to SSH hosts, and run the suite -- plus
# scripts/check-single-home.sh's G1/G2/G4/G9/G11 -- on real hardware through
# the bundle's own scripts/on-target-run.sh.
#
# The hardware-test workflow runs the same bundle and the same board-side
# script on the self-hosted fleet. This is the ssh path to the same run, for
# any board you can reach, which is how you catch behaviour that differs by
# SoC (older cores taking lower-precision SIMD fallbacks, kernels built
# without DMA-BUF heaps, vendor GPU quirks) on hardware the fleet lacks, and
# how a new board is validated before it is given a fleet label.
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
#   FEATURES           cargo features for the TEST build, space- or
#                      comma-separated (default: none, so the crates build
#                      with their own defaults). Use it to reach tests that
#                      are behind a feature and would otherwise never run on
#                      a board -- `FEATURES=dma_test_formats` is the DMA-BUF
#                      import suite, which CI's hardware lane builds and this
#                      script did not. With more than one package selected,
#                      cargo wants a feature that is not shared spelled
#                      `<pkg>/<feature>`. Applied to the test build only, not
#                      to the C-API leaves.
#   FILTER             test-name filter passed to each binary. Matches the test
#                      FUNCTION name, not the file or binary name; one naming a
#                      source file matches nothing and every binary runs zero
#                      tests, reported as NO-TESTS.
#   REMOTE_DIR         remote bundle dir       (default: /tmp/hal-ontarget)
#   SYNC_TESTDATA      1 to rsync testdata/    (default: 1); 0 keeps the
#                      testdata already on the host
#   REQUIRE_GPU        1 to fail a host without a DRM render node as NO-GPU
#                      instead of letting every GPU test skip (default: 0)
#
# Exit status is non-zero if any host reported a test failure, and also if a
# host ran no tests at all, reported as NO-TESTS and distinct from FAIL: a run
# that exercised nothing cannot exit zero. A host that is unreachable, or that
# lacks the hardware a test needs, is reported separately and does NOT mask a
# real failure elsewhere. So is a host another run is using, as BUSY: two runs
# against one host share REMOTE_DIR on it and the host's results directory
# here, and would read back each other's results. Same rule for the deployed
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
FEATURES="${FEATURES:-}"
FILTER="${FILTER:-}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/hal-ontarget}"
SYNC_TESTDATA="${SYNC_TESTDATA:-1}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"
RESULTS="${ROOT}/target/on-target-results"
BUNDLES="${ROOT}/target/on-target-bundle"
readonly RULE='============================================================'

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

ssh_q() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$@"; return "$?"; }

mkdir -p "${RESULTS}"

# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------
# Which architectures to build, and what hardware each host has, are discovered
# from the hosts themselves, so adding a board is purely a matter of naming it
# on the command line. The glibc floor is NOT discovered -- it is a declared
# project floor (see above).
declare -a OK_HOSTS=() OK_ARCH=()
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
  OK_HOSTS+=("${target}"); OK_ARCH+=("${arch}")
done

if [[ ${#OK_HOSTS[@]} -eq 0 ]]; then
  echo "no usable hosts" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
for arch in aarch64 x86_64; do
  needed=0
  for a in "${OK_ARCH[@]}"; do [[ "$a" == "${arch}" ]] && needed=1; done
  [[ ${needed} -eq 1 ]] || continue
  CRATES="${CRATES}" CAPI_CRATES="${CAPI_CRATES}" GLIBC="${GLIBC}" \
    FEATURES="${FEATURES}" TESTDATA="${SYNC_TESTDATA}" \
    "${ROOT}/scripts/on-target-bundle.sh" tests "${arch}" "${BUNDLES}/tests-${arch}" || exit 1
done

# ---------------------------------------------------------------------------
# Deploy + run
# ---------------------------------------------------------------------------
overall=0

# One run per host at a time, held for the whole deploy-run-fetch of that
# host. mkdir is the lock because it is atomic in every shell, BusyBox's
# included. The owner line (machine, pid, start time) is the lock's token: a
# run releases the lock only while the owner file still holds its own token,
# so it never removes a lock another run took, including one taken after the
# stale-lock command below cleared this run's. A lock left by a killed run is
# cleared by hand (the BUSY message says how) or by the board's next reboot,
# since REMOTE_DIR is in /tmp.
LOCK="${REMOTE_DIR}.lock"
CURRENT_LOCK=""
LOCK_OWNER=""

# Take the lock on $1. Returns 0 when taken, 3 when another run holds it,
# 4 when the owner file could not be written (the directory is removed
# again), and ssh's 255 when the host could not be reached.
acquire_lock() {
  local host="$1" owner
  owner="$(hostname) pid $$ since $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  ssh_q "${host}" "mkdir '${LOCK}' 2>/dev/null || exit 3; echo '${owner}' > '${LOCK}/owner' || { rm -rf '${LOCK}'; exit 4; }"
  local rc=$?
  if [[ ${rc} -eq 0 ]]; then
    CURRENT_LOCK="${host}"; LOCK_OWNER="${owner}"
  fi
  return ${rc}
}

release_lock() {
  if [[ -n "${CURRENT_LOCK}" ]]; then
    ssh_q "${CURRENT_LOCK}" "[ \"\$(cat '${LOCK}/owner' 2>/dev/null)\" = '${LOCK_OWNER}' ] && rm -rf '${LOCK}'" > /dev/null 2>&1
    CURRENT_LOCK=""; LOCK_OWNER=""
  fi
  return 0
}
trap release_lock EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for i in "${!OK_HOSTS[@]}"; do
  release_lock
  target="${OK_HOSTS[$i]}"; arch="${OK_ARCH[$i]}"
  echo
  echo "${RULE}"
  echo "==> ${target}   (${arch})"
  echo "${RULE}"

  acquire_lock "${target}"
  case $? in
    0) ;;
    3)
      held="$(ssh_q "${target}" "cat '${LOCK}/owner' 2>/dev/null")"
      echo "SKIP: another on-target run is using ${target}${held:+ (${held})}"
      echo "      if that run is gone: ssh ${target} rm -rf '${LOCK}'"
      SUMMARY+=("${target}|BUSY|${arch}|another run holds ${LOCK}${held:+: ${held}}")
      continue ;;
    255)
      echo "SKIP: ${target} did not answer while taking the lock"
      SUMMARY+=("${target}|UNREACHABLE|${arch}|lost while taking ${LOCK}")
      continue ;;
    *)
      echo "SKIP: could not write ${LOCK}/owner on ${target}"
      SUMMARY+=("${target}|SYNCFAIL|${arch}|could not write ${LOCK}/owner")
      continue ;;
  esac

  out_dir="${RESULTS}/${target//[^A-Za-z0-9._-]/_}"
  rm -rf "${out_dir}"; mkdir -p "${out_dir}"

  # --delete: a renamed or deleted test binary must not keep running from a
  # previous deploy. rsync, not scp: testdata/ is large and mostly unchanged
  # run to run. With SYNC_TESTDATA=0 the bundle has no testdata/, and the
  # exclude keeps --delete from removing the host's copy. A minimal BSP image
  # may have no rsync; tar over ssh replaces the whole bundle instead, keeping
  # testdata/ the same way.
  ssh_q "${target}" "mkdir -p '${REMOTE_DIR}'"
  if ssh_q "${target}" "command -v rsync" > /dev/null 2>&1; then
    has_rsync=1
    excludes=(--exclude /results/)
    [[ "${SYNC_TESTDATA}" == "1" ]] || excludes+=(--exclude /testdata/)
    rsync -az --delete --info=none "${excludes[@]}" \
      "${BUNDLES}/tests-${arch}/" "${target}:${REMOTE_DIR}/"
  else
    has_rsync=0
    echo "    no rsync on ${target}; copying the bundle with tar"
    keep=""
    [[ "${SYNC_TESTDATA}" == "1" ]] || keep="! -name testdata"
    tar -C "${BUNDLES}/tests-${arch}" -cf - . | ssh_q "${target}" \
      "cd '${REMOTE_DIR}' && find . -mindepth 1 -maxdepth 1 ${keep} -exec rm -rf {} + && tar -xf -"
  fi
  if [[ $? -ne 0 ]]; then
    echo "SKIP: bundle sync failed"; SUMMARY+=("${target}|SYNCFAIL|${arch}|-"); continue
  fi

  run_env="FILTER=$(printf '%q' "${FILTER}") HAL_ONTARGET_REQUIRE_GPU=$(printf '%q' "${REQUIRE_GPU}")"
  ssh_q "${target}" "${run_env} '${REMOTE_DIR}/scripts/on-target-run.sh' '${REMOTE_DIR}/results'"
  rc=$?

  if [[ ${has_rsync} -eq 1 ]]; then
    rsync -az --info=none "${target}:${REMOTE_DIR}/results/" "${out_dir}/"
  else
    ssh_q "${target}" "tar -C '${REMOTE_DIR}/results' -cf - ." | tar -C "${out_dir}" -xf -
  fi
  if [[ $? -ne 0 ]]; then
    echo "SKIP: results sync failed"; SUMMARY+=("${target}|SYNCFAIL|${arch}|results not retrieved"); overall=1; continue
  fi
  if [[ ! -s "${out_dir}/summary.txt" ]]; then
    SUMMARY+=("${target}|FAIL|${arch}|on-target-run.sh exited ${rc} without a summary")
    overall=1; continue
  fi
  IFS='|' read -r result detail < "${out_dir}/summary.txt"
  totals="$(python3 -c 'import json,sys; print("%(passed)d passed, %(failed)d failed, %(ignored)d ignored" % json.load(open(sys.argv[1]))["tests"])' "${out_dir}/summary.json")"
  SUMMARY+=("${target}|${result}|${arch}|${totals:+${totals}; }${detail}")
  [[ "${result}" == PASS && ${rc} -eq 0 ]] || overall=1
done

release_lock

# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------
echo
echo "${RULE}"
printf "%-20s %-12s %-8s %s\n" "HOST" "RESULT" "ARCH" "DETAIL"
echo "------------------------------------------------------------"
for row in "${SUMMARY[@]}"; do
  IFS='|' read -r b r a d <<< "${row}"
  printf "%-20s %-12s %-8s %s\n" "$b" "$r" "$a" "$d"
done
echo "${RULE}"
echo "A skipped test is not a passed test. Check each host's"
echo "capabilities.txt to attribute skips to a missing device node."
echo "logs: ${RESULTS}"
exit ${overall}
