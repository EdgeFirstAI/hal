#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Container entrypoint for per-processor JPEG **decode-only** A/B.
#
# Matrix (default): for each of yuv + rgb, run HAL and TurboJPEG into memory.
#   HAL  yuv → --decode-only --decode-fmt native   (NV12/16/24)
#   HAL  rgb → --decode-only --decode-fmt rgb      (fused RGB, 4:4:4)
#   Turbo yuv → --decode-only --format yuv         (tjDecompressToYUV2)
#   Turbo rgb → --decode-only --format rgb         (tjDecompress2 TJPF_RGB)
#
# ImageProcessor::convert / letterbox is intentionally not timed.
#
# Env contract (documented in benchmarks/README.md):
#   EDGEFIRST_BENCH_COCO   JPEG directory (falls back to /opt/coco-smoke then
#                          /opt/testdata when empty)
#   EDGEFIRST_CODEC_FORCE_INTEL / EDGEFIRST_CODEC_FORCE_NEON  optional tier A/B
#   BOARD                  board / CPU label written into the CSV
#   RESULTS_DIR            output directory (default /results)
#   LIMIT / WARMUP         smoke knobs (default 50 / 10; LIMIT=0 = full set)
#   TENSOR_MEM             mem|dma|auto (default mem)
#   MODULES                comma list: hal_cpu,turbojpeg (default both)
#   FORMATS                comma list: yuv,rgb (default both)
#   DCT                    turbojpeg IDCT: accurate|fast (default accurate,
#                          matching HAL's islow kernel and libjpeg-turbo's own
#                          decompression default)
#   EXTRA_ARGS             extra argv appended to hal_cpu
set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-/results}"
COCO="${EDGEFIRST_BENCH_COCO:-/data/coco}"
BOARD="${BOARD:-unknown}"
LIMIT="${LIMIT:-50}"
WARMUP="${WARMUP:-10}"
TENSOR_MEM="${TENSOR_MEM:-mem}"
MODULES="${MODULES:-hal_cpu,turbojpeg}"
FORMATS="${FORMATS:-yuv,rgb}"
DCT="${DCT:-accurate}"
mkdir -p "${RESULTS_DIR}"

jpeg_count() {
  local d="$1"
  find "$d" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) 2>/dev/null | wc -l
}

if [[ ! -d "${COCO}" ]] || [[ "$(jpeg_count "${COCO}")" -eq 0 ]]; then
  if [[ -d /opt/coco-smoke ]] && [[ "$(jpeg_count /opt/coco-smoke)" -gt 0 ]]; then
    echo "warn: ${COCO} empty; using /opt/coco-smoke ($(jpeg_count /opt/coco-smoke) JPEGs)" >&2
    COCO=/opt/coco-smoke
  elif [[ -d /opt/testdata ]]; then
    echo "warn: ${COCO} empty; using /opt/testdata fixtures" >&2
    COCO=/opt/testdata
  else
    echo "error: COCO directory missing: ${COCO}" >&2
    exit 1
  fi
fi

export EDGEFIRST_BENCH_COCO="${COCO}"

tier_tag="auto"
if [[ -n "${EDGEFIRST_CODEC_FORCE_INTEL:-}" ]]; then
  tier_tag="intel-${EDGEFIRST_CODEC_FORCE_INTEL}"
elif [[ -n "${EDGEFIRST_CODEC_FORCE_NEON:-}" ]]; then
  tier_tag="neon-${EDGEFIRST_CODEC_FORCE_NEON}"
fi

limit_args=()
if [[ "${LIMIT}" != "0" ]]; then
  limit_args=(--limit "${LIMIT}")
fi

hal_decode_fmt() {
  case "$1" in
    yuv) echo native ;;
    rgb) echo rgb ;;
    *)
      echo "error: unknown FORMAT '$1' (expected yuv|rgb)" >&2
      exit 1
      ;;
  esac
}

IFS=',' read -r -a mods <<< "${MODULES}"
IFS=',' read -r -a fmts <<< "${FORMATS}"

for fmt in "${fmts[@]}"; do
  fmt="$(echo "${fmt}" | tr -d '[:space:]')"
  [[ -z "${fmt}" ]] && continue
  decode_fmt="$(hal_decode_fmt "${fmt}")"

  for mod in "${mods[@]}"; do
    mod="$(echo "${mod}" | tr -d '[:space:]')"
    case "${mod}" in
      hal_cpu|hal)
        csv="${RESULTS_DIR}/${BOARD}_hal_cpu_${fmt}_${tier_tag}.csv"
        echo "===== MODULE hal_cpu decode-only format=${fmt} (${BOARD}) =====" >&2
        # shellcheck disable=SC2086
        /usr/local/bin/hal_cpu \
          --board "${BOARD}" \
          --tensor-mem "${TENSOR_MEM}" \
          --warmup "${WARMUP}" \
          --decode-only \
          --decode-fmt "${decode_fmt}" \
          --csv "${csv}" \
          "${limit_args[@]}" \
          ${EXTRA_ARGS:-}
        echo "===== CSV ${csv} ====="
        cat "${csv}"
        ;;
      turbojpeg|turbo)
        csv="${RESULTS_DIR}/${BOARD}_turbojpeg_${fmt}.csv"
        echo "===== MODULE turbojpeg decode-only format=${fmt} (${BOARD}) =====" >&2
        /usr/local/bin/turbojpeg_bench \
          --coco "${COCO}" \
          --board "${BOARD}" \
          --warmup "${WARMUP}" \
          --decode-only \
          --format "${fmt}" \
          --dct "${DCT}" \
          --csv "${csv}" \
          "${limit_args[@]}"
        echo "===== CSV ${csv} ====="
        cat "${csv}"
        ;;
      "")
        ;;
      *)
        echo "error: unknown MODULE '${mod}' (expected hal_cpu|turbojpeg)" >&2
        exit 1
        ;;
    esac
  done
done
