#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# CLIC 2025 large-image corpus: fetch the challenge's image validation + test
# sets (62 lossless PNGs, ~2048 px long edge) and encode the JPEG benchmark
# corpora from them.
#
# Why CLIC 2025: the recognized, license-clean large-image set. The CLIC
# tasks page states "The images are released using the Unsplash license"
# (free including commercial use, no attribution required), and the download
# URLs are live — unlike CLIC 2020, whose archive is gone. COCO val2017
# stays the primary corpus (~0.27 MP); this one covers the ≥2 MP class.
#
# Every standard corpus ships lossless sources, so a JPEG decode benchmark on
# it is always "sources + an encode recipe". The recipe here is pinned and
# recorded in each output MANIFEST.txt: libjpeg-turbo cjpeg, fixed quality,
# one directory per subsampling (4:2:0 and 4:4:4 from identical pixels, so
# the pair isolates subsampling exactly). For the restart-marker control run
# make-dri-corpus.sh on an output directory afterwards (lossless jpegtran).
#
# Usage:
#   ./benchmarks/scripts/fetch-clic2025.sh [--dest DIR] [--quality N] [--jobs N]
#
# Defaults: --dest ~/Dataset/CLIC2025, --quality 90, --jobs = CPU count.
# Requires: curl, unzip, cjpeg (libjpeg-turbo), and one PNG reader out of
# ImageMagick `magick`, python3+Pillow, or macOS `sips`.

set -euo pipefail

DEST="${HOME}/Dataset/CLIC2025"
QUALITY=90
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="$2"; shift 2 ;;
    --quality) QUALITY="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Pinned downloads (checked 2026-08-13, from https://clic2025.compression.cc/tasks/).
VAL_URL="https://d152a8jkvz9wzs.cloudfront.net/data/clic2025_image_validation.zip"
VAL_SHA256="735f8da90c860e872db6b9dc6fac88d42d0ba021a7a5898b89c062f0634b3149"
TEST_URL="https://storage.googleapis.com/clic_datasets/clic2025_image_test.zip"
TEST_SHA256="c479a92b7579f716826822fe892abbe583cfa7cd7f07730a14227a98c7870e92"

command -v cjpeg >/dev/null || { echo "cjpeg not found (install libjpeg-turbo)" >&2; exit 1; }
command -v unzip >/dev/null || { echo "unzip not found" >&2; exit 1; }
if command -v sha256sum >/dev/null 2>&1; then SHA256() { sha256sum "$1" | cut -d' ' -f1; }
else SHA256() { shasum -a 256 "$1" | cut -d' ' -f1; }; fi

# PNG → PPM converter: ImageMagick, then python3+Pillow, then macOS sips.
if command -v magick >/dev/null 2>&1; then
  PNG2PPM=magick
elif python3 -c "import PIL" >/dev/null 2>&1; then
  PNG2PPM=pillow
elif command -v sips >/dev/null 2>&1; then
  PNG2PPM=sips
else
  echo "no PNG reader found (need ImageMagick, python3+Pillow, or sips)" >&2
  exit 1
fi

fetch() { # url sha out
  local url="$1" sha="$2" out="$3"
  if [[ -f "${out}" && "$(SHA256 "${out}")" == "${sha}" ]]; then
    echo "  cached: ${out}"
    return
  fi
  echo "  fetching $(basename "${out}")"
  curl -fSL -o "${out}.tmp" "${url}"
  [[ "$(SHA256 "${out}.tmp")" == "${sha}" ]] || { echo "sha256 mismatch for ${url}" >&2; exit 1; }
  mv "${out}.tmp" "${out}"
}

mkdir -p "${DEST}/zips" "${DEST}/png"
echo "==> Fetching CLIC 2025 image sets to ${DEST}"
fetch "${VAL_URL}" "${VAL_SHA256}" "${DEST}/zips/clic2025_image_validation.zip"
fetch "${TEST_URL}" "${TEST_SHA256}" "${DEST}/zips/clic2025_image_test.zip"
unzip -qo "${DEST}/zips/clic2025_image_validation.zip" -d "${DEST}/png" -x "__MACOSX/*"
unzip -qo "${DEST}/zips/clic2025_image_test.zip" -d "${DEST}/png" -x "__MACOSX/*"
n_png="$(find "${DEST}/png" -maxdepth 1 -iname '*.png' | wc -l | tr -d ' ')"
echo "  ${n_png} PNGs (expect 62: 32 validation + 30 test)"

encode_dir() { # sample_arg out_dir
  local sample="$1" out="$2"
  mkdir -p "${out}"
  echo "==> Encoding ${out} (cjpeg -quality ${QUALITY} -sample ${sample}, png2ppm=${PNG2PPM}, ${JOBS} jobs)"
  find "${DEST}/png" -maxdepth 1 -iname '*.png' -print0 |
    PNG2PPM="${PNG2PPM}" xargs -0 -P "${JOBS}" -I{} sh -c '
      src="$1"; out_dir="$2"; quality="$3"; sample="$4"
      name="$(basename "${src}" .png)"
      ppm="${out_dir}/${name}.ppm.tmp"
      case "${PNG2PPM}" in
        magick) magick "${src}" "ppm:${ppm}" ;;
        pillow) python3 -c "import sys; from PIL import Image; \
Image.open(sys.argv[1]).convert(\"RGB\").save(sys.argv[2], format=\"PPM\")" "${src}" "${ppm}" ;;
        sips)   sips -s format bmp "${src}" --out "${ppm}" >/dev/null ;;
      esac
      cjpeg -quality "${quality}" -sample "${sample}" -outfile "${out_dir}/${name}.jpg" "${ppm}" \
        || echo "FAIL ${name}" >&2
      rm -f "${ppm}"
    ' _ {} "${out}" "${QUALITY}" "${sample}"

  local n_out
  n_out="$(find "${out}" -maxdepth 1 -iname '*.jpg' | wc -l | tr -d ' ')"
  echo "    ${n_out}/${n_png} files"
  [[ "${n_out}" == "${n_png}" ]] || { echo "ERROR: incomplete encode in ${out}" >&2; exit 1; }

  {
    echo "source: CLIC 2025 image validation + test sets (Unsplash license)"
    echo "  ${VAL_URL} sha256=${VAL_SHA256}"
    echo "  ${TEST_URL} sha256=${TEST_SHA256}"
    echo "recipe: png → ppm (${PNG2PPM}) → cjpeg -quality ${QUALITY} -sample ${sample}"
    echo "cjpeg: $(command -v cjpeg)"
    cjpeg -version 2>&1 | head -1
    echo "generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  } > "${out}/MANIFEST.txt"
}

encode_dir 2x2 "${DEST}/jpeg-yuv420"
encode_dir 1x1 "${DEST}/jpeg-yuv444"

echo "OK. Next steps:"
echo "  python3 benchmarks/scripts/corpus_stats.py ${DEST}/jpeg-yuv420 ${DEST}/jpeg-yuv444"
echo "  ./benchmarks/scripts/make-dri-corpus.sh --src ${DEST}/jpeg-yuv420   # restart-marker control"
