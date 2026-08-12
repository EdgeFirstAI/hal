#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Cross-build Rust HAL modules, deploy, and smoke-run on SSH targets.
#
# Usage:
#   ./benchmarks/scripts/deploy-and-run.sh [host...]
#
# Env:
#   LIMIT                 smoke image count (default 50; 0 = full set)
#   EDGEFIRST_BENCH_COCO_REMOTE  remote JPEG dir (default /data/coco/val2017)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_WS="${ROOT}/benchmarks"
RESULTS="${BENCH_WS}/results"
TARGET_TRIPLE="aarch64-unknown-linux-gnu"
# profiling = release opts + debug symbols (perf / chrome traces).
PROFILE="${CARGO_PROFILE:-profiling}"
LIMIT="${LIMIT:-50}"
REMOTE_COCO="${EDGEFIRST_BENCH_COCO_REMOTE:-/data/coco/val2017}"
REMOTE_BIN_DIR="/tmp/edgefirst-bench"

MODULES=(hal_cpu hal_gl hal_g2d hal_v4l2_gl)

if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=(imx8mp-frdm imx95-frdm imx95-pro rpi5-hailo orin-nano)
fi

resolve_target() {
  local t="$1"
  if ssh -o BatchMode=yes -o ConnectTimeout=5 "$t" 'true' 2>/dev/null; then
    echo "$t"
    return 0
  fi
  # Optional alternate SSH alias for the same Orin Nano hardware
  # (EDGEFIRST_BENCH_ORIN_FALLBACK). Public board labels stay "orin-nano".
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
for d in '${REMOTE_COCO}' "\$HOME/coco/val2017" /tmp/coco/val2017; do
  if [[ -d "\$d" ]] && compgen -G "\$d/*.[Jj][Pp][Gg]" >/dev/null; then
    echo "\$d"
    exit 0
  fi
done
exit 1
EOS
}

echo "==> Building Rust modules for ${TARGET_TRIPLE} (profile=${PROFILE})"
(
  cd "${BENCH_WS}"
  cargo zigbuild --profile "${PROFILE}" --target "${TARGET_TRIPLE}" \
    -p hal_cpu -p hal_gl -p hal_g2d -p hal_v4l2_gl
)

BIN_DIR="${BENCH_WS}/target/${TARGET_TRIPLE}/${PROFILE}"

for target in "${TARGETS[@]}"; do
  echo "============================================"
  echo "==> deploy+run ${target}"
  echo "============================================"
  if ! host="$(resolve_target "${target}")"; then
    echo "SKIP: ${target} unreachable"
    continue
  fi
  # Public board label is always the requested target (e.g. orin-nano), even
  # when SSH resolved to an internal alias for the same hardware.
  board_label="${target}"
  out_dir="${RESULTS}/${board_label}"
  mkdir -p "${out_dir}"

  # `set -e` would take the whole run down on the first board without a
  # dataset, so take the exit status rather than letting it escape.
  coco="$(remote_coco_path "${host}")" || coco=""
  if [[ -z "${coco}" ]]; then
    echo "WARN: no COCO on ${host}; run sync-coco.sh first. Skipping runs."
    continue
  fi
  echo "  COCO: ${coco}"

  ssh "${host}" "mkdir -p '${REMOTE_BIN_DIR}'"
  for mod in "${MODULES[@]}"; do
    scp -q "${BIN_DIR}/${mod}" "${host}:${REMOTE_BIN_DIR}/${mod}"
  done

  run_module() {
    local mod="$1"
    local extra_env="${2:-}"
    echo "  -- ${mod}"
    # shellcheck disable=SC2029
    if ssh "${host}" \
      "cd '${REMOTE_BIN_DIR}' && ${extra_env} EDGEFIRST_BENCH_COCO='${coco}' \
       ./${mod} --limit ${LIMIT} --board '${board_label}' \
       --csv '${REMOTE_BIN_DIR}/${mod}.csv'" 2>&1 | tee "${out_dir}/${mod}.log"; then
      scp -q "${host}:${REMOTE_BIN_DIR}/${mod}.csv" "${out_dir}/${mod}.csv" || true
    else
      echo "  FAIL/SKIP ${mod} (see ${out_dir}/${mod}.log)"
    fi
  }

  for mod in "${MODULES[@]}"; do
    run_module "${mod}"
  done

  echo "OK: results in ${out_dir}"
done
