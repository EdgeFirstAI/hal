#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Published decoder A/B (BENCHMARKS.md § JPEG Decode).
#
# Release profile, no perf/trace instrumentation. Arms alternate inside one
# session (HAL / turbo islow / turbo ifast × YUV / RGB), pinned to one core.
# Best-of-N p50 is the claim number.
#
# Usage:
#   EDGEFIRST_BENCH_ORIN_FALLBACK=adis-uav1 \
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

  {
    echo "==== ${board_label} best-of-${ROUNDS} (n=${LIMIT}, pin=${PIN}, profile=${PROFILE})"
    for arm in hal tj_islow tj_ifast; do
      for fmt in yuv rgb; do
        best=""
        rounds=""
        for round in $(seq 1 "${ROUNDS}"); do
          f="${out_dir}/r${round}_${arm}_${fmt}.log"
          p="$(parse_p50 "${f}" || true)"
          [[ -z "${p}" ]] && continue
          rounds="${rounds}${rounds:+, }r${round}=${p}"
          if [[ -z "${best}" ]] || awk "BEGIN{exit !(${p} < ${best})}"; then
            best="${p}"
          fi
        done
        printf "  %-10s %-4s  best=%s ms  (%s)\n" "${arm}" "${fmt}" "${best:-?}" "${rounds:-no p50}"
      done
    done
  } | tee "${out_dir}/summary.txt"

  echo "OK: ${out_dir}"
done
