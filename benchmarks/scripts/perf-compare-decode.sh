#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Profile HAL vs TurboJPEG decode hotspots on a Linux target with perf.
#
# Usage:
#   ./benchmarks/scripts/perf-compare-decode.sh imx95-pro
#
# Runs COCO (primary Nv24 workload) and a zidane NV12 fixture set.
# Builds with Cargo profile `profiling` (optimized + symbols).
#
# Produces under benchmarks/results/<host>/perf/:
#   coco_hal_cpu_mem.stat.txt / .report.txt
#   zidane_hal_cpu_mem.*
#   coco_turbojpeg, …

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_WS="${ROOT}/benchmarks"
HOST="${1:-imx95-pro}"
LIMIT="${LIMIT:-80}"
PROFILE="${CARGO_PROFILE:-profiling}"
REMOTE_BIN="/tmp/edgefirst-bench"
OUT_LOCAL="${BENCH_WS}/results/${HOST}/perf"
TARGET_TRIPLE="aarch64-unknown-linux-gnu"

mkdir -p "${OUT_LOCAL}"

echo "==> Building aarch64 HAL modules (profile=${PROFILE})"
(
  cd "${BENCH_WS}"
  cargo zigbuild --profile "${PROFILE}" --target "${TARGET_TRIPLE}" \
    -p hal_cpu -p hal_gl -p hal_v4l2_gl
)

BIN_DIR="${BENCH_WS}/target/${TARGET_TRIPLE}/${PROFILE}"
ssh "${HOST}" "mkdir -p '${REMOTE_BIN}/perf' '${REMOTE_BIN}/sets/zidane'"
scp -q "${BIN_DIR}/hal_cpu" "${BIN_DIR}/hal_gl" "${BIN_DIR}/hal_v4l2_gl" \
  "${HOST}:${REMOTE_BIN}/"
# Reference decoder arm: native C, so a profile of it is a profile of
# libturbojpeg rather than of an interpreter calling it.
make -C "${BENCH_WS}/modules/turbojpeg" aarch64
scp -q "${BENCH_WS}/modules/turbojpeg/build/turbojpeg_bench.aarch64" \
  "${HOST}:${REMOTE_BIN}/turbojpeg_bench"
ssh "${HOST}" "chmod +x '${REMOTE_BIN}/turbojpeg_bench'"
scp -q "${ROOT}/testdata/zidane.jpg" "${HOST}:${REMOTE_BIN}/fixtures_zidane.jpg"
# shellcheck disable=SC2029
ssh "${HOST}" "d='${REMOTE_BIN}/sets/zidane'; rm -rf \"\$d\"; mkdir -p \"\$d\"; \
  for i in \$(seq -w 1 ${LIMIT}); do cp '${REMOTE_BIN}/fixtures_zidane.jpg' \"\$d/\$i.jpg\"; done"

COCO="$(ssh "${HOST}" 'for d in /data/coco/val2017 $HOME/coco/val2017 /root/coco/val2017; do
  [[ -d "$d" ]] && echo "$d" && exit 0; done; exit 1')"
ZIDANE="${REMOTE_BIN}/sets/zidane"
echo "  COCO=${COCO}"
echo "  ZIDANE=${ZIDANE}"

# Lower paranoid if we can (root boards often already allow).
ssh "${HOST}" 'if [[ $(id -u) -eq 0 ]]; then echo -1 > /proc/sys/kernel/perf_event_paranoid 2>/dev/null || true; fi
  cat /proc/sys/kernel/perf_event_paranoid'

run_stat() {
  local label="$1"
  local dataset="$2"
  shift 2
  echo "==> perf stat: ${label} (dataset=${dataset})"
  # shellcheck disable=SC2029
  ssh "${HOST}" "cd '${REMOTE_BIN}' && \
    EDGEFIRST_BENCH_COCO='${dataset}' perf stat -e cycles,instructions,cache-misses,cache-references,task-clock,context-switches,page-faults \
      -o '${REMOTE_BIN}/perf/${label}.stat.txt' -- $* --limit ${LIMIT} --board '${HOST}'" \
    2>&1 | tee "${OUT_LOCAL}/${label}.run.log" || true
  scp -q "${HOST}:${REMOTE_BIN}/perf/${label}.stat.txt" "${OUT_LOCAL}/" 2>/dev/null || true
}

run_record() {
  local label="$1"
  local dataset="$2"
  shift 2
  echo "==> perf record: ${label} (dataset=${dataset})"
  # shellcheck disable=SC2029
  ssh "${HOST}" "cd '${REMOTE_BIN}' && \
    EDGEFIRST_BENCH_COCO='${dataset}' perf record -F 99 -g --call-graph dwarf \
      -o '${REMOTE_BIN}/perf/${label}.data' -- $* --limit ${LIMIT} --board '${HOST}'" \
    2>&1 | tee -a "${OUT_LOCAL}/${label}.run.log" || true
  ssh "${HOST}" "perf report -i '${REMOTE_BIN}/perf/${label}.data' --stdio --no-children \
      --percent-limit 1 2>/dev/null | head -80" \
    | tee "${OUT_LOCAL}/${label}.report.txt" || true
}

# --- COCO primary (Nv24-heavy) ---
run_stat coco_hal_cpu_mem "${COCO}" ./hal_cpu --tensor-mem mem
run_stat coco_hal_cpu_dma "${COCO}" ./hal_cpu --tensor-mem dma
run_stat coco_turbojpeg "${COCO}" ./turbojpeg_bench --decode-only --format yuv --dct accurate
run_stat coco_hal_v4l2_gl "${COCO}" ./hal_v4l2_gl
run_stat coco_hal_gl "${COCO}" ./hal_gl

# --- zidane diagnostic (NV12 720p) ---
run_stat zidane_hal_cpu_mem "${ZIDANE}" ./hal_cpu --tensor-mem mem
run_stat zidane_turbojpeg "${ZIDANE}" ./turbojpeg_bench --decode-only --format yuv --dct accurate
run_stat zidane_hal_v4l2_gl "${ZIDANE}" ./hal_v4l2_gl

run_record coco_hal_cpu_mem "${COCO}" ./hal_cpu --tensor-mem mem
run_record zidane_hal_cpu_mem "${ZIDANE}" ./hal_cpu --tensor-mem mem
run_record coco_turbojpeg "${COCO}" ./turbojpeg_bench --decode-only --format yuv --dct accurate

echo "OK: ${OUT_LOCAL}"
ls -la "${OUT_LOCAL}"
