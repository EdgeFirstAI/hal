#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Full decoder A/B sweep (BENCHMARKS.md § JPEG Decode): six arms per host.
#
#   hal        EdgeFirst accurate (islow-class, the default)
#   hal_fast   EdgeFirst DctMethod::Fast (AAN, opt-in; EDGEFIRST_CODEC_DCT=fast)
#   tj_islow   libjpeg-turbo accurate IDCT (its default)
#   tj_ifast   libjpeg-turbo fast IDCT
#   zune       zune-jpeg (Rust; YCbCr / RGB output)
#   image      image crate (Rust; RGB only — its API has no raw-YUV output)
#
# Release profile, no perf/trace. Arms alternate inside one session per round,
# pinned to one core. Best-of-N p50 is the claim number. Supports aarch64
# boards and x86_64 hosts (the turbojpeg C bench is compiled on the remote
# host with its native gcc; Rust arms are cross-built with zigbuild).
#
# Usage:
#   ./benchmarks/scripts/decode-ab-sweep.sh imx8mp-frdm imx95-pro rpi5-hailo sebstation
#
# Env:
#   LIMIT / WARMUP / ROUNDS / PIN / CARGO_PROFILE   as decode-ab-publish.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_WS="${ROOT}/benchmarks"
RESULTS="${BENCH_WS}/results"
PROFILE="${CARGO_PROFILE:-release}"
LIMIT="${LIMIT:-200}"
WARMUP="${WARMUP:-20}"
ROUNDS="${ROUNDS:-3}"
PIN="${PIN:-0}"
REMOTE_COCO="${EDGEFIRST_BENCH_COCO_REMOTE:-/data/coco/val2017}"
# Home-relative: /tmp is quota-limited tmpfs on some hosts (sebstation).
REMOTE_BIN_DIR="edgefirst-bench-decode-sweep"

if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=(imx8mp-frdm imx95-pro rpi5-hailo sebstation)
fi

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

parse_p50() {
  grep -oE 'p50=[0-9.]+ ms' "$1" | head -1 | sed -E 's/p50=([0-9.]+) ms/\1/'
}

build_for() {
  local triple="$1"
  (
    cd "${BENCH_WS}"
    cargo zigbuild --profile "${PROFILE}" --target "${triple}" -p hal_cpu -p rust_jpeg
  )
}

for target in "${TARGETS[@]}"; do
  echo "============================================"
  echo "==> sweep ${target}  n=${LIMIT} rounds=${ROUNDS} pin=${PIN}"
  echo "============================================"
  if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "${target}" 'true' 2>/dev/null; then
    echo "SKIP: ${target} unreachable"
    continue
  fi
  arch="$(ssh "${target}" 'uname -m')"
  case "${arch}" in
    aarch64) triple="aarch64-unknown-linux-gnu" ;;
    x86_64) triple="x86_64-unknown-linux-gnu" ;;
    *) echo "SKIP: ${target} unsupported arch ${arch}"; continue ;;
  esac

  out_dir="${RESULTS}/${target}/decode-ab-sweep"
  mkdir -p "${out_dir}"

  if ! coco="$(remote_coco_path "${target}")"; then
    echo "WARN: no COCO on ${target}; run sync-coco.sh first. Skipping."
    continue
  fi
  echo "  arch=${arch}  COCO=${coco}"

  echo "==> Building hal_cpu + rust_jpeg for ${triple}"
  build_for "${triple}"

  ssh "${target}" "mkdir -p '${REMOTE_BIN_DIR}'"
  scp -q "${BENCH_WS}/target/${triple}/${PROFILE}/hal_cpu" "${target}:${REMOTE_BIN_DIR}/hal_cpu"
  scp -q "${BENCH_WS}/target/${triple}/${PROFILE}/rust_jpeg" "${target}:${REMOTE_BIN_DIR}/rust_jpeg"

  # turbojpeg bench: prebuilt aarch64 cross-binary, or native gcc on x86.
  if [[ "${arch}" == aarch64 ]]; then
    make -C "${BENCH_WS}/modules/turbojpeg" aarch64
    scp -q "${BENCH_WS}/modules/turbojpeg/build/turbojpeg_bench.aarch64" \
      "${target}:${REMOTE_BIN_DIR}/turbojpeg_bench"
  else
    scp -q "${BENCH_WS}/modules/turbojpeg/bench.c" "${target}:${REMOTE_BIN_DIR}/bench.c"
    ssh "${target}" "gcc -O2 -o '${REMOTE_BIN_DIR}/turbojpeg_bench' '${REMOTE_BIN_DIR}/bench.c' -ldl"
  fi
  ssh "${target}" "chmod +x ${REMOTE_BIN_DIR}/hal_cpu ${REMOTE_BIN_DIR}/rust_jpeg ${REMOTE_BIN_DIR}/turbojpeg_bench"

  pin_cmd="taskset -c ${PIN}"
  if ! ssh "${target}" "command -v taskset >/dev/null"; then
    echo "  note: no taskset on ${target}; running unpinned"
    pin_cmd=""
  fi

  run_arm() { # name round fmt command...
    local name="$1" round="$2" fmt="$3"
    shift 3
    echo "  -- r${round} ${name} ${fmt}"
    # shellcheck disable=SC2029
    ssh "${target}" \
      "cd '${REMOTE_BIN_DIR}' && unset EDGEFIRST_TRACE && \
       EDGEFIRST_BENCH_COCO='${coco}' ${pin_cmd} $*" \
      >"${out_dir}/r${round}_${name}_${fmt}.log" 2>&1 \
      || echo "  FAIL ${name} ${fmt} r${round}"
  }

  for round in $(seq 1 "${ROUNDS}"); do
    for fmt in yuv rgb; do
      decode_fmt=$([[ "${fmt}" == yuv ]] && echo native || echo rgb)
      run_arm hal "${round}" "${fmt}" \
        "./hal_cpu --limit ${LIMIT} --warmup ${WARMUP} --board '${target}' \
         --tensor-mem mem --decode-only --decode-fmt ${decode_fmt}"
      run_arm hal_fast "${round}" "${fmt}" \
        "env EDGEFIRST_CODEC_DCT=fast ./hal_cpu --limit ${LIMIT} --warmup ${WARMUP} \
         --board '${target}' --tensor-mem mem --decode-only --decode-fmt ${decode_fmt}"
      run_arm tj_islow "${round}" "${fmt}" \
        "./turbojpeg_bench --limit ${LIMIT} --warmup ${WARMUP} --board '${target}' \
         --decode-only --format ${fmt} --dct accurate"
      run_arm tj_ifast "${round}" "${fmt}" \
        "./turbojpeg_bench --limit ${LIMIT} --warmup ${WARMUP} --board '${target}' \
         --decode-only --format ${fmt} --dct fast"
      run_arm zune "${round}" "${fmt}" \
        "./rust_jpeg --limit ${LIMIT} --warmup ${WARMUP} --engine zune --format ${fmt}"
      if [[ "${fmt}" == rgb ]]; then
        run_arm image "${round}" "${fmt}" \
          "./rust_jpeg --limit ${LIMIT} --warmup ${WARMUP} --engine image --format rgb"
      fi
    done
  done

  {
    echo "==== ${target} best-of-${ROUNDS} (n=${LIMIT}, pin=${PIN}, profile=${PROFILE})"
    for arm in hal hal_fast tj_islow tj_ifast zune image; do
      for fmt in yuv rgb; do
        [[ "${arm}" == image && "${fmt}" == yuv ]] && continue
        best=""
        rounds=""
        for round in $(seq 1 "${ROUNDS}"); do
          f="${out_dir}/r${round}_${arm}_${fmt}.log"
          [[ -f "${f}" ]] || continue
          p="$(parse_p50 "${f}" || true)"
          [[ -z "${p}" ]] && continue
          rounds="${rounds}${rounds:+, }r${round}=${p}"
          if [[ -z "${best}" ]] || awk "BEGIN{exit !(${p} < ${best})}"; then
            best="${p}"
          fi
        done
        printf "  %-10s %-4s  best=%s ms  (%s)\n" "${arm}" "${fmt}" "${best:-?}" "${rounds:-no p50}"
      done
    done
  } | tee "${out_dir}/summary.txt"

  echo "OK: ${out_dir}"
done
