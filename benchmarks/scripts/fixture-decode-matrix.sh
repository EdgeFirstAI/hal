#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Diagnostic decode matrix on fixed fixtures (zidane NV12 + optional 444).
# COCO remains the primary Article-1 workload (deploy-and-run.sh).
#
# Usage:
#   ./benchmarks/scripts/fixture-decode-matrix.sh [host...]
#
# Env:
#   LIMIT   copies of each fixture for stable p50 (default 40)
#   FIXTURES  space-separated names under testdata/ (default: zidane zidane_444 giraffe)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_WS="${ROOT}/benchmarks"
RESULTS="${BENCH_WS}/results"
TARGET_TRIPLE="aarch64-unknown-linux-gnu"
PROFILE="${CARGO_PROFILE:-profiling}"
LIMIT="${LIMIT:-40}"
REMOTE_BIN="/tmp/edgefirst-bench"
FIXTURES="${FIXTURES:-zidane zidane_444 giraffe}"

if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=(imx8mp-frdm imx95-pro imx95-frdm rpi5-hailo)
fi

resolve_target() {
  local t="$1"
  if ssh -o BatchMode=yes -o ConnectTimeout=5 "$t" 'true' 2>/dev/null; then
    echo "$t"
    return 0
  fi
  return 1
}

echo "==> Building HAL modules (${PROFILE}, ${TARGET_TRIPLE})"
(
  cd "${BENCH_WS}"
  cargo zigbuild --profile "${PROFILE}" --target "${TARGET_TRIPLE}" \
    -p hal_cpu -p hal_gl -p hal_v4l2_gl
)

BIN_DIR="${BENCH_WS}/target/${TARGET_TRIPLE}/${PROFILE}"
if [[ ! -x "${BIN_DIR}/hal_cpu" ]]; then
  # zigbuild may place profiling under target/<triple>/<profile>
  BIN_DIR="${BENCH_WS}/target/${TARGET_TRIPLE}/${PROFILE}"
fi

# TurboJPEG reference arm: native C, dlopens libturbojpeg on the board.
TJ_BIN="${BENCH_WS}/modules/turbojpeg/build/turbojpeg_bench.aarch64"
make -C "${BENCH_WS}/modules/turbojpeg" aarch64

for target in "${TARGETS[@]}"; do
  echo "============================================"
  echo "==> fixture matrix ${target}"
  echo "============================================"
  if ! host="$(resolve_target "${target}")"; then
    echo "SKIP: ${target} unreachable"
    continue
  fi
  out_dir="${RESULTS}/${host}/fixtures"
  mkdir -p "${out_dir}"

  ssh "${host}" "mkdir -p '${REMOTE_BIN}/fixtures' '${REMOTE_BIN}/sets'"
  scp -q "${BIN_DIR}/hal_cpu" "${BIN_DIR}/hal_gl" "${BIN_DIR}/hal_v4l2_gl" \
    "${host}:${REMOTE_BIN}/"
  scp -q "${TJ_BIN}" "${host}:${REMOTE_BIN}/turbojpeg_bench"
  ssh "${host}" "chmod +x '${REMOTE_BIN}/turbojpeg_bench'"

  for name in ${FIXTURES}; do
    src="${ROOT}/testdata/${name}.jpg"
    if [[ ! -f "${src}" ]]; then
      echo "  SKIP missing ${src}"
      continue
    fi
    scp -q "${src}" "${host}:${REMOTE_BIN}/fixtures/${name}.jpg"
    # shellcheck disable=SC2029
    ssh "${host}" "d='${REMOTE_BIN}/sets/${name}'; rm -rf \"\$d\"; mkdir -p \"\$d\"; \
      for i in \$(seq -w 1 ${LIMIT}); do cp '${REMOTE_BIN}/fixtures/${name}.jpg' \"\$d/\$i.jpg\"; done"

    echo "  -- HAL SW ${name}"
    # shellcheck disable=SC2029
    ssh "${host}" "cd '${REMOTE_BIN}' && EDGEFIRST_BENCH_COCO='${REMOTE_BIN}/sets/${name}' \
      ./hal_cpu --tensor-mem mem --limit ${LIMIT} --warmup 5 --board '${host}' \
      --csv '${REMOTE_BIN}/fixtures/hal_cpu_${name}.csv'" \
      2>&1 | tee "${out_dir}/hal_cpu_${name}.log" || true
    scp -q "${host}:${REMOTE_BIN}/fixtures/hal_cpu_${name}.csv" "${out_dir}/" 2>/dev/null || true

    echo "  -- HAL V4L2+GL ${name}"
    # shellcheck disable=SC2029
    ssh "${host}" "cd '${REMOTE_BIN}' && EDGEFIRST_BENCH_COCO='${REMOTE_BIN}/sets/${name}' \
      ./hal_v4l2_gl --limit ${LIMIT} --warmup 5 --board '${host}' \
      --csv '${REMOTE_BIN}/fixtures/hal_v4l2_gl_${name}.csv'" \
      2>&1 | tee "${out_dir}/hal_v4l2_gl_${name}.log" || true
    scp -q "${host}:${REMOTE_BIN}/fixtures/hal_v4l2_gl_${name}.csv" "${out_dir}/" 2>/dev/null || true

    echo "  -- TurboJPEG ${name}"
    # shellcheck disable=SC2029
    ssh "${host}" "cd '${REMOTE_BIN}' && EDGEFIRST_BENCH_COCO='${REMOTE_BIN}/sets/${name}' \
      ./turbojpeg_bench --limit ${LIMIT} --warmup 5 --board '${host}' \
      --decode-only --format yuv --dct accurate \
      --csv '${REMOTE_BIN}/fixtures/turbojpeg_${name}.csv'" \
      2>&1 | tee "${out_dir}/turbojpeg_${name}.log" || true
    scp -q "${host}:${REMOTE_BIN}/fixtures/turbojpeg_${name}.csv" "${out_dir}/" 2>/dev/null || true
  done

  echo "OK: ${out_dir}"
done
