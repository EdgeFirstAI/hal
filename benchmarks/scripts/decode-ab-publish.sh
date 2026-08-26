#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Published decoder A/B (BENCHMARKS.md § JPEG Decode).
#
# Release profile, no perf/trace instrumentation. Arms alternate inside one
# session (HAL / turbo islow / turbo ifast × YUV / RGB), pinned to one core.
# The claim number is the MEDIAN p50 across rounds, reported with every
# round's p50 and the min–max spread (best-of-N favours the low tail).
# Build/run provenance for each host is captured to provenance.txt.
#
# Usage:
#   EDGEFIRST_BENCH_ORIN_FALLBACK=<ssh-host> \
#     ./benchmarks/scripts/decode-ab-publish.sh imx8mp-frdm imx95-pro rpi5-hailo orin-nano
#
# Env:
#   LIMIT / WARMUP / ROUNDS / PIN / CARGO_PROFILE
#     boards default 200 / 20 / 3 / 0 / release

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_WS="${ROOT}/benchmarks"
RESULTS="${BENCH_WS}/results"
TARGET_TRIPLE="aarch64-unknown-linux-gnu"
PROFILE="${CARGO_PROFILE:-release}"
LIMIT="${LIMIT:-200}"
WARMUP="${WARMUP:-20}"
ROUNDS="${ROUNDS:-3}"
PIN="${PIN:-0}"
REMOTE_COCO="${EDGEFIRST_BENCH_COCO_REMOTE:-/data/coco/val2017}"
REMOTE_BIN_DIR="/tmp/edgefirst-bench-decode-ab"

if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=(imx8mp-frdm imx95-pro rpi5-hailo orin-nano)
fi

resolve_target() {
  local t="$1"
  if ssh -o BatchMode=yes -o ConnectTimeout=5 "$t" 'true' 2>/dev/null; then
    echo "$t"
    return 0
  fi
  local fallback="${EDGEFIRST_BENCH_ORIN_FALLBACK:-}"
  if [[ "$t" == "orin-nano" && -n "${fallback}" ]] \
      && ssh -o BatchMode=yes -o ConnectTimeout=5 "${fallback}" 'true' 2>/dev/null; then
    echo "${fallback}"
    return 0
  fi
  return 1
}

remote_coco_path() {
  local host="$1"
  ssh "${host}" "bash -s" <<EOS
set -euo pipefail
for d in '${REMOTE_COCO}' "\$HOME/coco/val2017" /tmp/coco/val2017 /root/coco/val2017; do
  if [[ -d "\$d" ]] && compgen -G "\$d/*.[Jj][Pp][Gg]" >/dev/null; then
    echo "\$d"
    exit 0
  fi
done
exit 1
EOS
}

parse_p50() {
  # First "p50=N.NNN ms" in a log (HAL prints decode_p50 afterwards).
  grep -oE 'p50=[0-9.]+ ms' "$1" | head -1 | sed -E 's/p50=([0-9.]+) ms/\1/'
}

# "median min max" of the newline-separated numbers on stdin (even count
# averages the two middle values).
median_spread() {
  sort -g | awk '
    { a[NR] = $1 }
    END {
      if (NR == 0) exit 1
      m = (NR % 2) ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2
      printf "%.3f %.3f %.3f\n", m, a[1], a[NR]
    }'
}

# Toolchain on the build side; kernel / CPU / governor / clocks / libturbojpeg
# resolution on the run side. What a reader needs to reproduce the numbers.
capture_provenance() { # host out_dir
  local host="$1" out="$2"
  {
    echo "captured: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "--- local build ---"
    echo "rustc: $(rustc -V 2>/dev/null || echo unknown)"
    echo "cargo-zigbuild: $(cargo zigbuild --version 2>/dev/null || echo unknown)"
    echo "profile: ${PROFILE}  n=${LIMIT}  warmup=${WARMUP}  rounds=${ROUNDS}  pin=${PIN}"
    echo "turbojpeg cross cc: zig $(zig version 2>/dev/null || echo unknown)"
    echo "--- ${host} ---"
    ssh "${host}" "bash -s" -- "${PIN}" <<'EOS'
PIN="${1:-0}"
uname -a
grep -m1 -E 'model name|^Model' /proc/cpuinfo 2>/dev/null || true
for f in scaling_governor scaling_cur_freq cpuinfo_max_freq; do
  p="/sys/devices/system/cpu/cpu${PIN}/cpufreq/${f}"
  [ -r "$p" ] && echo "cpu${PIN} ${f}: $(cat "$p")"
done
lib="$(ldconfig -p 2>/dev/null | awk '/libturbojpeg\.so/ { print $NF; exit }')"
echo "libturbojpeg: ${lib:-not-in-ldconfig}"
if [ -n "${lib:-}" ]; then
  real="$(readlink -f "${lib}")"
  echo "libturbojpeg resolved: ${real}"
  command -v dpkg >/dev/null 2>&1 && dpkg -S "${real}" 2>/dev/null | head -1
fi
true
EOS
  } >"${out}/provenance.txt" 2>&1 || echo "  WARN: provenance capture incomplete (see ${out}/provenance.txt)"
}

echo "==> Building hal_cpu for ${TARGET_TRIPLE} (profile=${PROFILE})"
(
  cd "${BENCH_WS}"
  cargo zigbuild --profile "${PROFILE}" --target "${TARGET_TRIPLE}" -p hal_cpu
)

BIN="${BENCH_WS}/target/${TARGET_TRIPLE}/${PROFILE}/hal_cpu"
TJ_BIN="${BENCH_WS}/modules/turbojpeg/build/turbojpeg_bench.aarch64"
make -C "${BENCH_WS}/modules/turbojpeg" aarch64

for target in "${TARGETS[@]}"; do
  echo "============================================"
  echo "==> publish A/B ${target}  n=${LIMIT} rounds=${ROUNDS} pin=${PIN}"
  echo "============================================"
  if ! host="$(resolve_target "${target}")"; then
    echo "SKIP: ${target} unreachable"
    continue
  fi
  board_label="${target}"
  out_dir="${RESULTS}/${board_label}/decode-ab-publish"
  mkdir -p "${out_dir}"
  rm -f "${out_dir}"/*.log "${out_dir}"/summary.txt

  if ! coco="$(remote_coco_path "${host}")"; then
    echo "WARN: no COCO on ${host}; run sync-coco.sh first. Skipping."
    continue
  fi
  echo "  host=${host}  COCO=${coco}"

  ssh "${host}" "mkdir -p '${REMOTE_BIN_DIR}'"
  scp -q "${BIN}" "${host}:${REMOTE_BIN_DIR}/hal_cpu"
  scp -q "${TJ_BIN}" "${host}:${REMOTE_BIN_DIR}/turbojpeg_bench"
  ssh "${host}" "chmod +x '${REMOTE_BIN_DIR}/hal_cpu' '${REMOTE_BIN_DIR}/turbojpeg_bench'"

  pin_cmd="taskset -c ${PIN}"
  if ! ssh "${host}" "command -v taskset >/dev/null"; then
    echo "  note: no taskset on ${host}; running unpinned"
    pin_cmd=""
  fi

  for round in $(seq 1 "${ROUNDS}"); do
    for fmt in yuv rgb; do
      if [[ "${fmt}" == yuv ]]; then
        decode_fmt=native
      else
        decode_fmt=rgb
      fi

      echo "  -- r${round} HAL ${fmt}"
      # shellcheck disable=SC2029
      ssh "${host}" \
        "cd '${REMOTE_BIN_DIR}' && unset EDGEFIRST_TRACE && \
         EDGEFIRST_BENCH_COCO='${coco}' ${pin_cmd} \
         ./hal_cpu --limit ${LIMIT} --warmup ${WARMUP} --board '${board_label}' \
         --tensor-mem mem --decode-only --decode-fmt ${decode_fmt}" \
        >"${out_dir}/r${round}_hal_${fmt}.log" 2>&1 || echo "  FAIL HAL ${fmt} r${round}"

      echo "  -- r${round} turbo islow ${fmt}"
      # shellcheck disable=SC2029
      ssh "${host}" \
        "cd '${REMOTE_BIN_DIR}' && EDGEFIRST_BENCH_COCO='${coco}' ${pin_cmd} \
         ./turbojpeg_bench --limit ${LIMIT} --warmup ${WARMUP} \
         --board '${board_label}' --decode-only --format ${fmt} --dct accurate" \
        >"${out_dir}/r${round}_tj_islow_${fmt}.log" 2>&1 || echo "  FAIL TJ islow ${fmt} r${round}"

      echo "  -- r${round} turbo ifast ${fmt}"
      # shellcheck disable=SC2029
      ssh "${host}" \
        "cd '${REMOTE_BIN_DIR}' && EDGEFIRST_BENCH_COCO='${coco}' ${pin_cmd} \
         ./turbojpeg_bench --limit ${LIMIT} --warmup ${WARMUP} \
         --board '${board_label}' --decode-only --format ${fmt} --dct fast" \
        >"${out_dir}/r${round}_tj_ifast_${fmt}.log" 2>&1 || echo "  FAIL TJ ifast ${fmt} r${round}"
    done
  done

  capture_provenance "${host}" "${out_dir}"

  {
    echo "==== ${board_label} median-of-${ROUNDS} (n=${LIMIT}, pin=${PIN}, profile=${PROFILE})"
    for arm in hal tj_islow tj_ifast; do
      for fmt in yuv rgb; do
        vals=""
        rounds=""
        for round in $(seq 1 "${ROUNDS}"); do
          f="${out_dir}/r${round}_${arm}_${fmt}.log"
          [[ -f "${f}" ]] || continue
          p="$(parse_p50 "${f}" || true)"
          [[ -z "${p}" ]] && continue
          rounds="${rounds}${rounds:+, }r${round}=${p}"
          vals="${vals}${vals:+ }${p}"
        done
        if [[ -n "${vals}" ]]; then
          read -r med lo hi < <(printf '%s\n' ${vals} | median_spread)
          n_rounds="$(printf '%s\n' ${vals} | wc -l | tr -d ' ')"
          short=""
          [[ "${n_rounds}" != "${ROUNDS}" ]] && short=" [only ${n_rounds}/${ROUNDS} rounds]"
          printf "  %-10s %-4s  median=%s ms  spread=[%s,%s]  (%s)%s\n" \
            "${arm}" "${fmt}" "${med}" "${lo}" "${hi}" "${rounds}" "${short}"
        else
          printf "  %-10s %-4s  median=?  (no p50)\n" "${arm}" "${fmt}"
        fi
      done
    done
  } | tee "${out_dir}/summary.txt"

  echo "OK: ${out_dir}"
done
