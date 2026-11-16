#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# 4:2:0 control corpus: transcode COCO val2017 (4:4:4 as distributed) to
# 4:2:0 with libjpeg-turbo cjpeg.
#
# The transcode re-encodes at a fixed --quality (default 90). Pass
# --with-444-twin to also emit a 4:4:4 re-encode at the same quality.
#
# Greyscale sources pass through djpeg|cjpeg as single-component JPEGs.
#
# COCO images carry individual Flickr CC licenses (some NC): do not publish
# or redistribute the transcoded JPEGs.
#
# Usage:
#   ./benchmarks/scripts/make-coco-yuv420.sh [--src DIR] [--quality N]
#       [--jobs N] [--with-444-twin]
#
# Defaults: --src ~/Dataset/COCO/val2017, output alongside it as
# val2017-yuv420 (and val2017-yuv444 for the twin), --quality 90,
# --jobs = CPU count. Requires cjpeg/djpeg (libjpeg-turbo) on PATH.

set -euo pipefail

SRC="${HOME}/Dataset/COCO/val2017"
QUALITY=90
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
WITH_TWIN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="$2"; shift 2 ;;
    --quality) QUALITY="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --with-444-twin) WITH_TWIN=1; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v cjpeg >/dev/null || { echo "cjpeg not found (install libjpeg-turbo)" >&2; exit 1; }
command -v djpeg >/dev/null || { echo "djpeg not found (install libjpeg-turbo)" >&2; exit 1; }
[[ -d "${SRC}" ]] || { echo "source not found: ${SRC}" >&2; exit 1; }

transcode_dir() { # sample_arg out_dir
  local sample="$1" out="$2"
  mkdir -p "${out}"
  echo "==> ${SRC} → ${out}  (cjpeg -quality ${QUALITY} -sample ${sample}, ${JOBS} jobs)"
  # djpeg → PPM/PGM → cjpeg. -sample applies to colour images; greyscale
  # sources come out of djpeg as PGM and re-encode as single-component JPEG.
  find "${SRC}" -maxdepth 1 \( -iname '*.jpg' -o -iname '*.jpeg' \) -print0 |
    xargs -0 -P "${JOBS}" -I{} sh -c '
      src="$1"; out_dir="$2"; quality="$3"; sample="$4"
      name="$(basename "${src}")"
      djpeg "${src}" | cjpeg -quality "${quality}" -sample "${sample}" \
        -outfile "${out_dir}/${name}" || echo "FAIL ${name}" >&2
    ' _ {} "${out}" "${QUALITY}" "${sample}"

  local n_src n_out
  n_src="$(find "${SRC}" -maxdepth 1 \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l | tr -d ' ')"
  n_out="$(find "${out}" -maxdepth 1 \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l | tr -d ' ')"
  echo "    ${n_out}/${n_src} files"
  [[ "${n_out}" == "${n_src}" ]] || { echo "ERROR: incomplete transcode in ${out}" >&2; exit 1; }

  # Record the recipe next to the corpus.
  {
    echo "source: ${SRC}"
    echo "recipe: djpeg | cjpeg -quality ${QUALITY} -sample ${sample}"
    echo "cjpeg: $(command -v cjpeg)"
    cjpeg -version 2>&1 | head -1
    echo "generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "note: COCO-derived; individual Flickr CC licenses — do not redistribute"
  } > "${out}/MANIFEST.txt"
}

transcode_dir 2x2 "$(dirname "${SRC}")/$(basename "${SRC}")-yuv420"
if [[ "${WITH_TWIN}" == 1 ]]; then
  transcode_dir 1x1 "$(dirname "${SRC}")/$(basename "${SRC}")-yuv444"
fi

echo "OK. Verify with: python3 benchmarks/scripts/corpus_stats.py <out_dir>"
