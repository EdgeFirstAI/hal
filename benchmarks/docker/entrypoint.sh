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
#   DATASET_S3             optional corpus source fetched into the JPEG dir
#                          before the run, using the task's IAM role (AWS
#                          Batch): s3://bucket/key.tar.gz|.tgz|.zip extracts
#                          an archive; s3://bucket/prefix/ syncs a directory
#   DATASET_URL            same, over plain https (e.g. a presigned URL)
#   EDGEFIRST_CODEC_FORCE_INTEL / EDGEFIRST_CODEC_FORCE_NEON  optional tier A/B
#   BOARD                  board / CPU label written into the CSV
#   RESULTS_DIR            output directory (default /results)
#   LIMIT / WARMUP         smoke knobs (default 50 / 10; LIMIT=0 = full set)
#   ROUNDS                 interleaved rounds (default 1). With ROUNDS>1 the
#                          whole module matrix repeats per round in one
#                          session (the board-sweep protocol: report the
#                          median across rounds); CSVs gain a _rN suffix
#   PIN                    optional core to pin every arm to via taskset
#   TENSOR_MEM             mem|dma|auto (default mem)
#   MODULES                comma list: hal_cpu,turbojpeg,zune,image,stb,wuffs
#                          (default all six; zune covers yuv+rgb, image/stb/
#                          wuffs are rgb-only and skip the yuv pass)
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
MODULES="${MODULES:-hal_cpu,turbojpeg,zune,image,stb,wuffs}"
FORMATS="${FORMATS:-yuv,rgb}"
DCT="${DCT:-accurate}"
ROUNDS="${ROUNDS:-1}"
mkdir -p "${RESULTS_DIR}"

# Optional single-core pinning (matches the board sweeps' taskset -c).
runner=()
if [[ -n "${PIN:-}" ]]; then
  if command -v taskset >/dev/null 2>&1; then
    runner=(taskset -c "${PIN}")
  else
    echo "warn: PIN=${PIN} requested but taskset is unavailable; running unpinned" >&2
  fi
fi
run_pinned() { "${runner[@]}" "$@"; }

jpeg_count() {
  local d="$1"
  find "$d" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' \) 2>/dev/null | wc -l
}

extract_archive() { # file dest
  local f="$1" dest="$2"
  case "$f" in
    *.tar.gz | *.tgz) tar -xzf "$f" -C "$dest" ;;
    *.tar) tar -xf "$f" -C "$dest" ;;
    *.zip) unzip -qo "$f" -d "$dest" ;;
    *)
      echo "error: unsupported archive type: $f" >&2
      exit 1
      ;;
  esac
}

# Optional corpus fetch (AWS Batch: the job role authenticates DATASET_S3, no
# secrets in the job definition). Skipped when the directory already has
# JPEGs, so a mounted volume wins over a re-download.
if [[ -n "${DATASET_S3:-}${DATASET_URL:-}" ]] && [[ "$(jpeg_count "${COCO}")" -eq 0 ]]; then
  mkdir -p "${COCO}"
  if [[ -n "${DATASET_S3:-}" ]]; then
    case "${DATASET_S3}" in
      *.tar.gz | *.tgz | *.tar | *.zip)
        echo "==> fetching corpus archive ${DATASET_S3}" >&2
        fname="/tmp/$(basename "${DATASET_S3}")"
        aws s3 cp --no-progress "${DATASET_S3}" "${fname}"
        extract_archive "${fname}" "${COCO}"
        rm -f "${fname}"
        ;;
      *)
        echo "==> syncing corpus prefix ${DATASET_S3}" >&2
        aws s3 sync --no-progress "${DATASET_S3}" "${COCO}"
        ;;
    esac
  else
    echo "==> fetching corpus archive from URL" >&2
    fname="/tmp/corpus-archive.$(basename "${DATASET_URL%%\?*}")"
    curl -fSL -o "${fname}" "${DATASET_URL}"
    extract_archive "${fname}" "${COCO}"
    rm -f "${fname}"
  fi
  # macOS BSD tar can ship AppleDouble metadata (._*) unless the archive was
  # created with COPYFILE_DISABLE=1; they match *.jpg globs and are not JPEGs.
  find "${COCO}" -name '._*' -delete 2>/dev/null || true
  # Archives may nest a single top-level directory; flatten one level.
  if [[ "$(jpeg_count "${COCO}")" -eq 0 ]]; then
    inner="$(find "${COCO}" -mindepth 1 -maxdepth 1 -type d | head -1)"
    if [[ -n "${inner}" ]] && [[ "$(jpeg_count "${inner}")" -gt 0 ]]; then
      COCO="${inner}"
    fi
  fi
  echo "==> corpus ready: ${COCO} ($(jpeg_count "${COCO}") JPEGs)" >&2
fi

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

# Always pass --limit explicitly: every harness honours `--limit 0` as "full
# set", whereas omitting the flag would fall back to per-arm defaults that
# DISAGREE (50 for hal_cpu/turbojpeg/stb/wuffs, 200 for rust_jpeg).
limit_args=(--limit "${LIMIT}")

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

for round in $(seq 1 "${ROUNDS}"); do
rtag=""
if [[ "${ROUNDS}" != "1" ]]; then
  rtag="_r${round}"
  echo "######## ROUND ${round}/${ROUNDS} ########" >&2
fi
for fmt in "${fmts[@]}"; do
  fmt="$(echo "${fmt}" | tr -d '[:space:]')"
  [[ -z "${fmt}" ]] && continue
  decode_fmt="$(hal_decode_fmt "${fmt}")"

  for mod in "${mods[@]}"; do
    mod="$(echo "${mod}" | tr -d '[:space:]')"
    case "${mod}" in
      hal_cpu|hal)
        csv="${RESULTS_DIR}/${BOARD}_hal_cpu_${fmt}_${tier_tag}${rtag:-}.csv"
        echo "===== MODULE hal_cpu decode-only format=${fmt} (${BOARD}) =====" >&2
        # shellcheck disable=SC2086
        run_pinned /usr/local/bin/hal_cpu \
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
        csv="${RESULTS_DIR}/${BOARD}_turbojpeg_${fmt}${rtag:-}.csv"
        echo "===== MODULE turbojpeg decode-only format=${fmt} (${BOARD}) =====" >&2
        run_pinned /usr/local/bin/turbojpeg_bench \
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
      zune)
        csv="${RESULTS_DIR}/${BOARD}_zune_${fmt}${rtag:-}.csv"
        echo "===== MODULE zune decode-only format=${fmt} (${BOARD}) =====" >&2
        run_pinned /usr/local/bin/rust_jpeg \
          --coco "${COCO}" \
          --board "${BOARD}" \
          --warmup "${WARMUP}" \
          --engine zune \
          --format "${fmt}" \
          --csv "${csv}" \
          "${limit_args[@]}"
        echo "===== CSV ${csv} ====="
        cat "${csv}"
        ;;
      image)
        if [[ "${fmt}" != rgb ]]; then
          echo "(skip image: rgb-only arm, format=${fmt})" >&2
          continue
        fi
        csv="${RESULTS_DIR}/${BOARD}_image_rgb${rtag:-}.csv"
        echo "===== MODULE image decode-only format=rgb (${BOARD}) =====" >&2
        run_pinned /usr/local/bin/rust_jpeg \
          --coco "${COCO}" \
          --board "${BOARD}" \
          --warmup "${WARMUP}" \
          --engine image \
          --format rgb \
          --csv "${csv}" \
          "${limit_args[@]}"
        echo "===== CSV ${csv} ====="
        cat "${csv}"
        ;;
      stb|wuffs)
        if [[ "${fmt}" != rgb ]]; then
          echo "(skip ${mod}: rgb-only arm, format=${fmt})" >&2
          continue
        fi
        csv="${RESULTS_DIR}/${BOARD}_${mod}_rgb${rtag:-}.csv"
        echo "===== MODULE ${mod} decode-only format=rgb (${BOARD}) =====" >&2
        run_pinned "/usr/local/bin/${mod}_bench" \
          --coco "${COCO}" \
          --board "${BOARD}" \
          --warmup "${WARMUP}" \
          --decode-only \
          --format rgb \
          --csv "${csv}" \
          "${limit_args[@]}"
        echo "===== CSV ${csv} ====="
        cat "${csv}"
        ;;
      "")
        ;;
      *)
        echo "error: unknown MODULE '${mod}' (expected hal_cpu|turbojpeg|zune|image|stb|wuffs)" >&2
        exit 1
        ;;
    esac
  done
done
done

csv_count="$(find "${RESULTS_DIR}" -maxdepth 1 -name '*.csv' 2>/dev/null | wc -l | tr -d ' ')"
if [[ "${csv_count}" -eq 0 ]]; then
  echo "error: matrix produced no CSVs (MODULES=${MODULES}, FORMATS=${FORMATS} — rgb-only modules skip the yuv pass)" >&2
  exit 1
fi
