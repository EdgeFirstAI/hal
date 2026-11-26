#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Cross-builds a tests bundle with scripts/on-target-bundle.sh, deploys it to
# SSH hosts and runs the bundle's scripts/on-target-run.sh on each. The
# hardware-test workflow runs the same bundle and board-side script on the
# self-hosted fleet.
#
# Usage:
#   ./scripts/on-target-test.sh <ssh-host> [ssh-host...]
#   EDGEFIRST_TARGETS="hostA hostB" ./scripts/on-target-test.sh
#
# Env:
#   EDGEFIRST_TARGETS  space-separated SSH hosts (alternative to arguments)
#   CRATES             cargo packages to test  (default: every crate with tests)
#   CAPI_CRATES        the -capi leaves to build and deploy (default: all of
#                      them)
#   GLIBC              glibc floor for the build (default: the project floor,
#                      see README "Toolchain and Platform Floors")
#   FEATURES           cargo features for the test build, space- or
#                      comma-separated (default: none); `<pkg>/<feature>` when
#                      more than one package is selected. Not applied to the
#                      C-API leaves.
#   FILTER             test-name filter passed to each binary. It matches the
#                      test function name; a filter that matches nothing makes
#                      every binary run zero tests, reported as NO-TESTS.
#   REMOTE_DIR         remote bundle dir       (default: /tmp/hal-ontarget)
#   SYNC_TESTDATA      1 to rsync testdata/    (default: 1); 0 keeps the
#                      testdata already on the host
#   REQUIRE_GPU        1 to fail a host without a DRM render node as NO-GPU
#                      (default: 0)
#   REQUIRE_DMA        1 to fail a host whose DMA heap probe cannot allocate as
#                      NO-DMA, and to fail every DMA test that would skip
#                      (default: 0)
#   GDB                1 to run each test binary under gdb on the host, which
#                      logs every thread's backtrace if it crashes or hangs
#                      (default: 0)
#   GDB_TIMEOUT        seconds a binary may run under gdb before it counts as
#                      hung (default: 900)
#
# Exits non-zero when any host reports a test failure or ran no tests
# (NO-TESTS). An unreachable host and a host another run is using (BUSY) are
# reported separately.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CRATES="${CRATES:-edgefirst-tensor edgefirst-codec edgefirst-image edgefirst-decoder edgefirst-tracker edgefirst-kernels}"
# Each -capi leaf is its own workspace, built by `--manifest-path`.
CAPI_CRATES="${CAPI_CRATES:-tensor-capi image-capi codec-capi decoder-capi tracker-capi kernels-capi}"
# The glibc floor the binaries are built against.
GLIBC="${GLIBC:-2.35}"
FEATURES="${FEATURES:-}"
FILTER="${FILTER:-}"
REMOTE_DIR="${REMOTE_DIR:-/tmp/hal-ontarget}"
SYNC_TESTDATA="${SYNC_TESTDATA:-1}"
REQUIRE_GPU="${REQUIRE_GPU:-0}"
REQUIRE_DMA="${REQUIRE_DMA:-0}"
GDB="${GDB:-0}"
GDB_TIMEOUT="${GDB_TIMEOUT:-900}"
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
# Reads each host's architecture and hardware from the host.
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

# Takes one lock per host for its deploy, run and fetch: a directory made
# with mkdir, holding an owner file with this run's token. A run releases a
# lock only while the owner file holds its own token.
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

  # Syncs the bundle with rsync --delete, or with tar over ssh on a host
  # without rsync. With SYNC_TESTDATA=0 the host's testdata/ is kept.
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

  run_env="FILTER=$(printf '%q' "${FILTER}") HAL_ONTARGET_REQUIRE_GPU=$(printf '%q' "${REQUIRE_GPU}") HAL_ONTARGET_REQUIRE_DMA=$(printf '%q' "${REQUIRE_DMA}") HAL_ONTARGET_GDB=$(printf '%q' "${GDB}") HAL_ONTARGET_GDB_TIMEOUT=$(printf '%q' "${GDB_TIMEOUT}")"
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
