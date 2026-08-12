#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Capability probe for Article-1 boards. Writes results/<board>/probe.txt
#
# Usage: ./benchmarks/probe/probe.sh [host...]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS="${ROOT}/benchmarks/results"

if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=(imx8mp-frdm imx95-frdm imx95-pro rpi5-hailo orin-nano)
fi

REMOTE_SCRIPT=$(cat <<'EOS'
set -euo pipefail
echo "=== host: $(hostname) ==="
echo "=== uname ==="
uname -a
echo
echo "=== dma_heap ==="
ls -la /dev/dma_heap/ 2>/dev/null || echo "(none)"
echo
echo "=== galcore / G2D ==="
ls -la /dev/galcore 2>/dev/null || echo "(no /dev/galcore)"
echo
echo "=== dri ==="
ls -la /dev/dri/ 2>/dev/null || echo "(no /dev/dri)"
echo
echo "=== V4L2 JPEG M2M candidates ==="
if command -v v4l2-ctl >/dev/null 2>&1; then
  for n in /dev/video*; do
    [[ -e "$n" ]] || continue
    echo "--- $n ---"
    v4l2-ctl -d "$n" --all 2>/dev/null | grep -E 'Driver name|Card type|Video Capture|Video Output|jpeg|JPEG|m2m|M2M' || true
    v4l2-ctl -d "$n" --list-formats-ext 2>/dev/null | head -40 || true
  done
else
  echo "v4l2-ctl not installed"
  ls -la /dev/video* 2>/dev/null || echo "(no /dev/video*)"
fi
echo
echo "=== gst jpeg / v4l2 plugins (informational) ==="
if command -v gst-inspect-1.0 >/dev/null 2>&1; then
  gst-inspect-1.0 2>/dev/null | grep -Ei 'jpeg|v4l2' || true
else
  echo "gst-inspect-1.0 not installed"
fi
echo
echo "=== eglinfo dma_buf (if present) ==="
if command -v eglinfo >/dev/null 2>&1; then
  eglinfo 2>/dev/null | grep -i dma_buf || echo "(no dma_buf lines)"
else
  echo "eglinfo not installed"
fi
echo
echo "=== nvjpeg / CUDA hints ==="
ls /usr/lib*/libnvjpeg.so* 2>/dev/null || echo "(no libnvjpeg)"
command -v nvidia-smi >/dev/null && nvidia-smi -L || echo "(no nvidia-smi)"
EOS
)

for target in "${TARGETS[@]}"; do
  echo "============================================"
  echo "==> probe ${target}"
  echo "============================================"
  out_dir="${RESULTS}/${target}"
  mkdir -p "${out_dir}"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${target}" "bash -s" <<<"${REMOTE_SCRIPT}" \
      >"${out_dir}/probe.txt" 2>&1; then
    echo "FAIL: ${target} unreachable or probe failed (see ${out_dir}/probe.txt)" >&2
    # Optional alternate SSH alias for the same Orin Nano hardware.
    fallback="${EDGEFIRST_BENCH_ORIN_FALLBACK:-}"
    if [[ "${target}" == "orin-nano" && -n "${fallback}" ]]; then
      echo "==> trying ${fallback} as Orin Nano fallback"
      if ssh -o BatchMode=yes -o ConnectTimeout=5 "${fallback}" "bash -s" <<<"${REMOTE_SCRIPT}" \
          >"${out_dir}/probe.txt" 2>&1; then
        echo "OK: orin-nano probe (via fallback) → ${out_dir}/probe.txt"
        continue
      fi
    fi
    continue
  fi
  echo "OK: ${out_dir}/probe.txt"
done
