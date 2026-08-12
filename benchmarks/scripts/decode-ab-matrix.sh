#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Decode-only HAL vs TurboJPEG A/B on SSH aarch64 targets (YUV + RGB).
#
# Times JPEG → memory only (no ImageProcessor::convert / letterbox):
#   HAL  --decode-only --decode-fmt native|rgb
#   TurboJPEG --decode-only --format yuv|rgb
#
# Usage:
#   ./benchmarks/scripts/decode-ab-matrix.sh [host...]
#
# Env:
#   LIMIT                      smoke image count (default 50; 0 = full set)
#   WARMUP                     warmup iterations (default 10)
#   EDGEFIRST_BENCH_COCO_REMOTE  remote JPEG dir (default /data/coco/val2017)
#   CARGO_PROFILE              cargo profile (default profiling)
#   FORMATS                    yuv,rgb (default both)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_WS="${ROOT}/benchmarks"
RESULTS="${BENCH_WS}/results"
TARGET_TRIPLE="aarch64-unknown-linux-gnu"
PROFILE="${CARGO_PROFILE:-profiling}"
LIMIT="${LIMIT:-50}"
WARMUP="${WARMUP:-10}"
REMOTE_COCO="${EDGEFIRST_BENCH_COCO_REMOTE:-/data/coco/val2017}"
REMOTE_BIN_DIR="/tmp/edgefirst-bench-decode-ab"
FORMATS="${FORMATS:-yuv,rgb}"
# TurboJPEG IDCT accuracy class. `accurate` (islow) matches HAL's kernel and is
# libjpeg-turbo's own decompression default; `fast` selects ifast.
DCT="${DCT:-accurate}"

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

hal_decode_fmt() {
  case "$1" in
    yuv) echo native ;;
    rgb) echo rgb ;;
    *)
      echo "error: unknown FORMAT '$1'" >&2
      exit 1
      ;;
  esac
}

echo "==> Building hal_cpu for ${TARGET_TRIPLE} (profile=${PROFILE})"
(
  cd "${BENCH_WS}"
  cargo zigbuild --profile "${PROFILE}" --target "${TARGET_TRIPLE}" -p hal_cpu
)

BIN="${BENCH_WS}/target/${TARGET_TRIPLE}/${PROFILE}/hal_cpu"

# The TurboJPEG arm is native C, like the HAL arm is native Rust; it dlopens
# libturbojpeg on the board, so cross-building it needs no sysroot.
TJ_BIN="${BENCH_WS}/modules/turbojpeg/build/turbojpeg_bench.aarch64"
make -C "${BENCH_WS}/modules/turbojpeg" aarch64

for target in "${TARGETS[@]}"; do
  echo "============================================"
  echo "==> decode A/B ${target}"
  echo "============================================"
  if ! host="$(resolve_target "${target}")"; then
    echo "SKIP: ${target} unreachable"
    continue
  fi
  board_label="${target}"
  out_dir="${RESULTS}/${board_label}/decode-ab"
  mkdir -p "${out_dir}"

  if ! coco="$(remote_coco_path "${host}")"; then
    echo "WARN: no COCO on ${host}; run sync-coco.sh first. Skipping."
    continue
  fi
  echo "  COCO: ${coco}"

  ssh "${host}" "mkdir -p '${REMOTE_BIN_DIR}'"
  scp -q "${BIN}" "${host}:${REMOTE_BIN_DIR}/hal_cpu"
  scp -q "${TJ_BIN}" "${host}:${REMOTE_BIN_DIR}/turbojpeg_bench"
  ssh "${host}" "chmod +x '${REMOTE_BIN_DIR}/turbojpeg_bench'"

  IFS=',' read -r -a fmts <<< "${FORMATS}"
  for fmt in "${fmts[@]}"; do
    fmt="$(echo "${fmt}" | tr -d '[:space:]')"
    [[ -z "${fmt}" ]] && continue
    decode_fmt="$(hal_decode_fmt "${fmt}")"

    echo "  -- hal_cpu decode-only ${fmt}"
    # shellcheck disable=SC2029
    if ssh "${host}" \
      "cd '${REMOTE_BIN_DIR}' && EDGEFIRST_BENCH_COCO='${coco}' \
       ./hal_cpu --limit ${LIMIT} --warmup ${WARMUP} --board '${board_label}' \
       --tensor-mem mem --decode-only --decode-fmt ${decode_fmt} \
       --csv '${REMOTE_BIN_DIR}/hal_cpu_${fmt}.csv'" \
      2>&1 | tee "${out_dir}/hal_cpu_${fmt}.log"; then
      scp -q "${host}:${REMOTE_BIN_DIR}/hal_cpu_${fmt}.csv" \
        "${out_dir}/hal_cpu_${fmt}.csv" || true
    else
      echo "  FAIL hal_cpu ${fmt}"
    fi

    echo "  -- turbojpeg decode-only ${fmt}"
    # shellcheck disable=SC2029
    if ssh "${host}" \
      "cd '${REMOTE_BIN_DIR}' && EDGEFIRST_BENCH_COCO='${coco}' \
       ./turbojpeg_bench --limit ${LIMIT} --warmup ${WARMUP} \
       --board '${board_label}' --decode-only --format ${fmt} --dct ${DCT} \
       --csv '${REMOTE_BIN_DIR}/turbojpeg_${fmt}.csv'" \
      2>&1 | tee "${out_dir}/turbojpeg_${fmt}.log"; then
      scp -q "${host}:${REMOTE_BIN_DIR}/turbojpeg_${fmt}.csv" \
        "${out_dir}/turbojpeg_${fmt}.csv" || true
    else
      echo "  FAIL turbojpeg ${fmt}"
    fi
  done

  echo "OK: results in ${out_dir}"
done
