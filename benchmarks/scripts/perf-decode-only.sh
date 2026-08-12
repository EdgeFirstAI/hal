#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Profile the software JPEG **decode-only** path (no letterbox/convert) and
# bucket samples into decode stages: entropy, IDCT, colour, plane write.
#
# Usage:
#   ./benchmarks/scripts/perf-decode-only.sh            # local host
#   ./benchmarks/scripts/perf-decode-only.sh imx95-pro  # SSH target (aarch64)
#
# Env:
#   LIMIT / WARMUP   images / warmup (default 1000 / 20; 300 on remote)
#   FORMATS          yuv,rgb (default both)
#   FREQ             perf sample frequency (default 3999 local, 999 remote:
#                    the small cores spend long enough in the handler at 3999
#                    to skew what is being measured). Set it to pin both.
#   BOARD            label for outputs (default host name)
#
# Outputs under benchmarks/results/<board>/perf-decode/:
#   hal_<fmt>.data / .report.txt / .srcline.txt / .stages.txt

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_WS="${ROOT}/benchmarks"
HOST="${1:-}"
FORMATS="${FORMATS:-yuv,rgb}"
FREQ="${FREQ:-}"
PROFILE="${CARGO_PROFILE:-profiling}"
TARGET_TRIPLE="aarch64-unknown-linux-gnu"
REMOTE_DIR="/tmp/edgefirst-bench-decode-ab"

hal_decode_fmt() {
  case "$1" in
    yuv) echo native ;;
    rgb) echo rgb ;;
    *) echo "error: unknown FORMAT '$1' (expected yuv|rgb)" >&2; exit 1 ;;
  esac
}

# Bucket a `perf report -s sym,srcline` dump into decode stages.
bucket_stages() {
  python3 - "$1" <<'PY'
import re, sys
buckets = {}
for line in open(sys.argv[1]):
    m = re.match(r"\s+([0-9.]+)%\s+\[[.k]\]\s+(\S+)\s+(\S+)\s*$", line)
    if not m:
        continue
    pct, sym, src = float(m.group(1)), m.group(2), m.group(3)
    if "huffman.rs" in src or "bitstream.rs" in src:
        b = "entropy (huffman+bitstream)"
    elif "idct" in sym:
        b = "IDCT (incl. dequant)"
    elif "ycbcr" in sym or "color.rs" in src:
        b = "colour YCbCr->RGB"
    elif "interleave_uv" in sym or "write_" in sym:
        b = "UV interleave / plane write"
    elif sym.startswith("edgefirst_codec") or "mcu.rs" in src:
        b = "mcu loop / other codec"
    else:
        b = "kernel / other"
    buckets[b] = buckets.get(b, 0) + pct
for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
    print(f"  {v:6.2f}%  {k}")
PY
}

if [[ -z "${HOST}" ]]; then
  BOARD="${BOARD:-$(hostname)}"
  LIMIT="${LIMIT:-1000}"
  WARMUP="${WARMUP:-20}"
  OUT="${BENCH_WS}/results/${BOARD}/perf-decode"
  mkdir -p "${OUT}"
  ( cd "${BENCH_WS}" && cargo build --profile "${PROFILE}" -p hal_cpu )
  HAL="${BENCH_WS}/target/${PROFILE}/hal_cpu"

  if [[ "$(cat /proc/sys/kernel/perf_event_paranoid)" -gt 1 ]]; then
    echo "note: perf_event_paranoid > 1; run" \
         "'sudo sh -c \"echo -1 > /proc/sys/kernel/perf_event_paranoid\"'" >&2
  fi

  IFS=',' read -r -a fmts <<< "${FORMATS}"
  for fmt in "${fmts[@]}"; do
    dfmt="$(hal_decode_fmt "${fmt}")"
    echo "==> perf record hal_cpu decode-only ${fmt} (${BOARD})"
    perf record -F "${FREQ:-3999}" -o "${OUT}/hal_${fmt}.data" -- \
      "${HAL}" --limit "${LIMIT}" --warmup "${WARMUP}" --board "${BOARD}" \
      --tensor-mem mem --decode-only --decode-fmt "${dfmt}" \
      >/dev/null 2>"${OUT}/hal_${fmt}.run.log"
    perf report -i "${OUT}/hal_${fmt}.data" --stdio --no-children -s sym \
      --percent-limit 1 2>/dev/null > "${OUT}/hal_${fmt}.report.txt" || true
    perf report -i "${OUT}/hal_${fmt}.data" --stdio --no-children -s srcline \
      --percent-limit 0.5 2>/dev/null > "${OUT}/hal_${fmt}.srcline.txt" || true
    perf report -i "${OUT}/hal_${fmt}.data" --stdio --no-children -s sym,srcline \
      --percent-limit 0 2>/dev/null > "${OUT}/hal_${fmt}.raw.txt" || true
    { echo "==== ${BOARD} hal_${fmt} decode-only stage breakdown"
      bucket_stages "${OUT}/hal_${fmt}.raw.txt"
      grep -E 'p50=' "${OUT}/hal_${fmt}.run.log" || true
    } | tee "${OUT}/hal_${fmt}.stages.txt"
  done
  echo "OK: ${OUT}"
  exit 0
fi

# --- remote (aarch64 SSH target) ---
BOARD="${BOARD:-${HOST}}"
LIMIT="${LIMIT:-300}"
WARMUP="${WARMUP:-10}"
OUT="${BENCH_WS}/results/${BOARD}/perf-decode"
mkdir -p "${OUT}"

echo "==> Building hal_cpu for ${TARGET_TRIPLE} (profile=${PROFILE})"
( cd "${BENCH_WS}" && cargo zigbuild --profile "${PROFILE}" --target "${TARGET_TRIPLE}" -p hal_cpu )

ssh "${HOST}" "mkdir -p '${REMOTE_DIR}'"
scp -q "${BENCH_WS}/target/${TARGET_TRIPLE}/${PROFILE}/hal_cpu" "${HOST}:${REMOTE_DIR}/hal_cpu"

COCO="$(ssh "${HOST}" 'for d in /data/coco/val2017 $HOME/coco/val2017 /root/coco/val2017; do
  [[ -d "$d" ]] && echo "$d" && exit 0; done; exit 1')"
echo "  COCO=${COCO}"

IFS=',' read -r -a fmts <<< "${FORMATS}"
for fmt in "${fmts[@]}"; do
  dfmt="$(hal_decode_fmt "${fmt}")"
  echo "==> perf record hal_cpu decode-only ${fmt} (${BOARD})"
  # shellcheck disable=SC2029
  ssh "${HOST}" "cd '${REMOTE_DIR}' && EDGEFIRST_BENCH_COCO='${COCO}' \
    perf record -F ${FREQ:-999} -o perf_${fmt}.data -- \
    ./hal_cpu --limit ${LIMIT} --warmup ${WARMUP} --board '${BOARD}' \
    --tensor-mem mem --decode-only --decode-fmt ${dfmt}" \
    >/dev/null 2>"${OUT}/hal_${fmt}.run.log" || true
  # perf report on small cores is slow; keep the sample set modest.
  ssh "${HOST}" "cd '${REMOTE_DIR}' && perf report -i perf_${fmt}.data --stdio \
    --no-children -s sym --percent-limit 1 2>/dev/null" \
    > "${OUT}/hal_${fmt}.report.txt" || true
  ssh "${HOST}" "cd '${REMOTE_DIR}' && perf report -i perf_${fmt}.data --stdio \
    --no-children -s sym,srcline --percent-limit 0 2>/dev/null" \
    > "${OUT}/hal_${fmt}.raw.txt" || true
  { echo "==== ${BOARD} hal_${fmt} decode-only stage breakdown"
    bucket_stages "${OUT}/hal_${fmt}.raw.txt"
  } | tee "${OUT}/hal_${fmt}.stages.txt"
done

echo "OK: ${OUT}"
