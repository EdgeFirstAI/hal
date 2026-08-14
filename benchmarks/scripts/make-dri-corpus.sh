#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Restart-interval (DRI) control corpus: losslessly add restart markers to
# every JPEG in a directory with `jpegtran -restart`.
#
# jpegtran re-serialises the SAME DCT coefficients — pixels and quality are
# bit-identical to the source; the ONLY change is the restart-marker
# structure. That makes this the exact isolate for the claim that
# libjpeg-turbo's fast Huffman decoder is disabled on DRI streams (its README
# documents up to ~20% slower): rerun the decoder A/B on the -dri directory
# and any timing change is attributable to restart handling alone.
#
# `-restart 1` = one restart marker per MCU row, the camera-stream-like
# density (hardware encoders emit restarts for error resilience).
#
# Usage:
#   ./benchmarks/scripts/make-dri-corpus.sh --src DIR [--restart N] [--jobs N]
#
# Output: sibling directory <src>-dri. Requires jpegtran (libjpeg-turbo).

set -euo pipefail

SRC=""
RESTART=1
JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="$2"; shift 2 ;;
    --restart) RESTART="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v jpegtran >/dev/null || { echo "jpegtran not found (install libjpeg-turbo)" >&2; exit 1; }
[[ -n "${SRC}" && -d "${SRC}" ]] || { echo "pass --src DIR (got: '${SRC}')" >&2; exit 1; }

OUT="${SRC%/}-dri"
mkdir -p "${OUT}"
echo "==> ${SRC} → ${OUT}  (jpegtran -restart ${RESTART} -copy all, lossless, ${JOBS} jobs)"

find "${SRC}" -maxdepth 1 \( -iname '*.jpg' -o -iname '*.jpeg' \) -print0 |
  xargs -0 -P "${JOBS}" -I{} sh -c '
    src="$1"; out_dir="$2"; restart="$3"
    name="$(basename "${src}")"
    jpegtran -restart "${restart}" -copy all -outfile "${out_dir}/${name}" "${src}" \
      || echo "FAIL ${name}" >&2
  ' _ {} "${OUT}" "${RESTART}"

n_src="$(find "${SRC}" -maxdepth 1 \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l | tr -d ' ')"
n_out="$(find "${OUT}" -maxdepth 1 \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l | tr -d ' ')"
echo "    ${n_out}/${n_src} files"
[[ "${n_out}" == "${n_src}" ]] || { echo "ERROR: incomplete DRI transcode" >&2; exit 1; }

{
  echo "source: ${SRC}"
  echo "recipe: jpegtran -restart ${RESTART} -copy all (lossless; coefficients unchanged)"
  echo "jpegtran: $(command -v jpegtran)"
  jpegtran -version 2>&1 | head -1
  echo "generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "${OUT}/MANIFEST.txt"

echo "OK. Verify with: python3 benchmarks/scripts/corpus_stats.py ${OUT}"
