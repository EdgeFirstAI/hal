#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Local (no-SSH) variant of decode-ab-sweep.sh for mbp-m2-max and any other
# macOS host — the script runs directly on the machine being measured
# instead of over SSH, since macOS boards in this fleet don't have a
# separate reachable SSH target the way the Linux boards do. Same
# methodology: interleaved rounds, n=LIMIT, warmup=WARMUP, median of
# rounds' p50 with spread, same eight arms, same output format
# (results/mbp-m2-max/decode-ab-sweep/<corpus>/summary.txt) as the real
# sweep — see BENCHMARKS.md § JPEG Decode for how these numbers are used.
#
# macOS has no taskset-equivalent, so this runs unpinned; see BENCHMARKS.md's
# discussion of the resulting spread on this board specifically.
#
# Usage:
#   ./benchmarks/scripts/decode-ab-sweep-macos-local.sh /path/to/coco/val2017
#   ./benchmarks/scripts/decode-ab-sweep-macos-local.sh /path/to/corpora/val2017-yuv420
#
# Env:
#   LIMIT / WARMUP / ROUNDS   as decode-ab-sweep.sh (default 200 / 20 / 3)
#   SKIP_BUILD=1              skip the cargo/make build step (binaries already built)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_WS="${ROOT}/benchmarks"
RESULTS="${BENCH_WS}/results"
LIMIT="${LIMIT:-200}"
WARMUP="${WARMUP:-20}"
ROUNDS="${ROUNDS:-3}"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <coco-dir>" >&2
  exit 2
fi
COCO="$1"
CORPUS_NAME="$(basename "${COCO}")"
BIN="${BENCH_WS}/target/release"
CBIN="${BENCH_WS}/modules"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "error: this script is the macOS-local variant; use decode-ab-sweep.sh for Linux SSH boards" >&2
  exit 1
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  echo "==> Building hal_cpu + rust_jpeg (release)"
  (cd "${BENCH_WS}" && cargo build --release -p hal_cpu -p rust_jpeg)
  echo "==> Building turbojpeg_bench + stb_bench + wuffs_bench"
  make -C "${BENCH_WS}/modules/turbojpeg" >/dev/null
  make -C "${BENCH_WS}/modules/stb" >/dev/null
  make -C "${BENCH_WS}/modules/wuffs" >/dev/null
fi

out_dir="${RESULTS}/mbp-m2-max/decode-ab-sweep/${CORPUS_NAME}"
mkdir -p "${out_dir}"

parse_p50() {
  grep -oE 'p50=[0-9.]+ ms' "$1" | head -1 | sed -E 's/p50=([0-9.]+) ms/\1/'
}
median_spread() {
  sort -g | awk '
    { a[NR] = $1 }
    END {
      if (NR == 0) exit 1
      m = (NR % 2) ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2
      printf "%.3f %.3f %.3f\n", m, a[1], a[NR]
    }'
}

run_arm() { # name round fmt command...
  local name="$1" round="$2" fmt="$3"
  shift 3
  echo "  -- r${round} ${name} ${fmt}"
  ( cd "${BENCH_WS}" && unset EDGEFIRST_TRACE && EDGEFIRST_BENCH_COCO="${COCO}" "$@" ) \
    >"${out_dir}/r${round}_${name}_${fmt}.log" 2>&1 \
    || echo "  FAIL ${name} ${fmt} r${round}"
}

for round in $(seq 1 "${ROUNDS}"); do
  for fmt in yuv rgb; do
    decode_fmt=$([[ "${fmt}" == yuv ]] && echo native || echo rgb)
    run_arm hal "${round}" "${fmt}" \
      "${BIN}/hal_cpu" --limit "${LIMIT}" --warmup "${WARMUP}" --board mbp-m2-max \
      --tensor-mem mem --decode-only --decode-fmt "${decode_fmt}"
    EDGEFIRST_CODEC_DCT=fast run_arm hal_fast "${round}" "${fmt}" \
      "${BIN}/hal_cpu" --limit "${LIMIT}" --warmup "${WARMUP}" --board mbp-m2-max \
      --tensor-mem mem --decode-only --decode-fmt "${decode_fmt}"
    run_arm tj_islow "${round}" "${fmt}" \
      "${CBIN}/turbojpeg/build/turbojpeg_bench" --limit "${LIMIT}" --warmup "${WARMUP}" \
      --board mbp-m2-max --decode-only --format "${fmt}" --dct accurate
    run_arm tj_ifast "${round}" "${fmt}" \
      "${CBIN}/turbojpeg/build/turbojpeg_bench" --limit "${LIMIT}" --warmup "${WARMUP}" \
      --board mbp-m2-max --decode-only --format "${fmt}" --dct fast
    run_arm zune "${round}" "${fmt}" \
      "${BIN}/rust_jpeg" --limit "${LIMIT}" --warmup "${WARMUP}" --engine zune --format "${fmt}"
    if [[ "${fmt}" == rgb ]]; then
      run_arm image "${round}" "${fmt}" \
        "${BIN}/rust_jpeg" --limit "${LIMIT}" --warmup "${WARMUP}" --engine image --format rgb
      run_arm stb "${round}" "${fmt}" \
        "${CBIN}/stb/build/stb_bench" --limit "${LIMIT}" --warmup "${WARMUP}" --board mbp-m2-max \
        --decode-only --format rgb
      run_arm wuffs "${round}" "${fmt}" \
        "${CBIN}/wuffs/build/wuffs_bench" --limit "${LIMIT}" --warmup "${WARMUP}" --board mbp-m2-max \
        --decode-only --format rgb
      run_arm tj_fastupsample "${round}" "${fmt}" \
        "${CBIN}/turbojpeg/build/turbojpeg_bench" --limit "${LIMIT}" --warmup "${WARMUP}" \
        --board mbp-m2-max --decode-only --format rgb --dct accurate --upsample fast
    fi
  done
done

{
  echo "==== mbp-m2-max ${CORPUS_NAME} median-of-${ROUNDS} (n=${LIMIT}, pin=none-macos, profile=release)"
  for arm in hal hal_fast tj_islow tj_ifast tj_fastupsample zune image stb wuffs; do
    for fmt in yuv rgb; do
      case "${arm}" in
        tj_fastupsample | image | stb | wuffs) [[ "${fmt}" == yuv ]] && continue ;;
      esac
      vals=""
      rounds=""
      for round in $(seq 1 "${ROUNDS}"); do
        f="${out_dir}/r${round}_${arm}_${fmt}.log"
        [[ -f "${f}" ]] || continue
        p="$(parse_p50 "${f}" || true)"
        [[ -z "${p}" ]] && continue
        vals="${vals}${p}"$'\n'
        rounds="${rounds}r${round}=${p}, "
      done
      [[ -z "${vals}" ]] && continue
      read -r med lo hi < <(printf '%s' "${vals}" | median_spread)
      printf "  %-10s %-3s median=%s ms  spread=[%s,%s]  (%s)\n" \
        "${arm}" "${fmt}" "${med}" "${lo}" "${hi}" "${rounds%, }"
    done
  done
} | tee "${out_dir}/summary.txt"

echo "OK: ${out_dir}"
