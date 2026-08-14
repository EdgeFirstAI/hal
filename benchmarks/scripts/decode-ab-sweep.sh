#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Full decoder A/B sweep (BENCHMARKS.md § JPEG Decode): eight arms per host.
#
#   hal        EdgeFirst accurate (islow-class, the default)
#   hal_fast   EdgeFirst DctMethod::Fast (AAN, opt-in; EDGEFIRST_CODEC_DCT=fast)
#   tj_islow   libjpeg-turbo accurate IDCT (its default)
#   tj_ifast   libjpeg-turbo fast IDCT
#   zune       zune-jpeg (Rust; YCbCr / RGB output)
#   image      image crate (Rust; RGB only — its API has no raw-YUV output)
#   stb        stb_image (C single-header; RGB only, allocates per call)
#   wuffs      Wuffs v0.4 (Google, memory-safe C; RGB only)
#
# Release profile, no perf/trace. Arms alternate inside one session per round,
# pinned to one core. The claim number is the MEDIAN p50 across rounds, with
# every round's p50 and the min–max spread reported next to it (best-of-N
# systematically favours the low tail; the median of interleaved rounds does
# not). Build/run provenance for each host is captured to provenance.txt.
# Supports aarch64 boards and x86_64 hosts (the turbojpeg C bench is compiled
# on the remote host with its native gcc; Rust arms are cross-built with
# zigbuild).
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
# Corpus identity (val2017, val2017-yuv420, jpeg-yuv420, …): names the
# results subdirectory so control-corpus runs never overwrite the primary,
# and extends the remote fallback search to ~/corpora/<name> (hosts where
# /data is not writable, e.g. adis-uav1).
CORPUS_NAME="$(basename "${REMOTE_COCO}")"
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
for d in '${REMOTE_COCO}' "\$HOME/corpora/${CORPUS_NAME}" "\$HOME/coco/val2017" /tmp/coco/val2017 /root/coco/val2017; do
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

# "median min max" of the newline-separated numbers on stdin (even count
# averages the two middle values).
median_spread() {
  sort -g | awk '
    { a[NR] = $1 }
    END {
      if (NR == 0) exit 1
      m = (NR % 2) ? a[(NR + 1) / 2] : (a[NR / 2] + a[NR / 2 + 1]) / 2
      printf "%.3f %.3f %.3f\n", m, a[1], a[NR]
    }'
}

# Record everything a reader needs to reproduce this host's numbers:
# toolchain + comparator versions on the build side, kernel / CPU / governor /
# clocks / libturbojpeg resolution on the run side.
capture_provenance() { # host out_dir
  local host="$1" out="$2"
  {
    echo "captured: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "--- local build ---"
    echo "rustc: $(rustc -V 2>/dev/null || echo unknown)"
    echo "cargo-zigbuild: $(cargo zigbuild --version 2>/dev/null || echo unknown)"
    echo "profile: ${PROFILE}  n=${LIMIT}  warmup=${WARMUP}  rounds=${ROUNDS}  pin=${PIN}"
    for pkg in zune-jpeg image; do
      v="$(awk -v p="$pkg" '$0 == "name = \"" p "\"" { getline; gsub(/version = |"/, ""); print; exit }' \
        "${BENCH_WS}/Cargo.lock")"
      echo "${pkg}: ${v:-unknown}"
    done
    echo "zune-jpeg features: x86,neon,std (explicit in modules/rust_jpeg/Cargo.toml)"
    echo "stb/wuffs cross cc: zig $(zig version 2>/dev/null || echo unknown) (aarch64/x86_64 targets)"
    echo "--- ${host} ---"
    ssh "${host}" "bash -s" -- "${PIN}" <<'EOS'
PIN="${1:-0}"
uname -a
grep -m1 -E 'model name|^Model' /proc/cpuinfo 2>/dev/null || true
for f in scaling_governor scaling_cur_freq cpuinfo_max_freq; do
  p="/sys/devices/system/cpu/cpu${PIN}/cpufreq/${f}"
  [ -r "$p" ] && echo "cpu${PIN} ${f}: $(cat "$p")"
done
lib="$(ldconfig -p 2>/dev/null | awk '/libturbojpeg\.so/ { print $NF; exit }')"
echo "libturbojpeg: ${lib:-not-in-ldconfig}"
if [ -n "${lib:-}" ]; then
  real="$(readlink -f "${lib}")"
  echo "libturbojpeg resolved: ${real}"
  command -v dpkg >/dev/null 2>&1 && dpkg -S "${real}" 2>/dev/null | head -1
fi
command -v gcc >/dev/null 2>&1 && gcc --version | head -1
true
EOS
  } >"${out}/provenance.txt" 2>&1 || echo "  WARN: provenance capture incomplete (see ${out}/provenance.txt)"
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
  echo "==> sweep ${target}  corpus=${CORPUS_NAME}  n=${LIMIT} rounds=${ROUNDS} pin=${PIN}"
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

  out_dir="${RESULTS}/${target}/decode-ab-sweep/${CORPUS_NAME}"
  mkdir -p "${out_dir}"

  if ! coco="$(remote_coco_path "${target}")"; then
    echo "WARN: no COCO on ${target}; run sync-coco.sh first. Skipping."
    continue
  fi
  if [[ "$(basename "${coco}")" != "${CORPUS_NAME}" ]]; then
    echo "SKIP: ${target} resolved '${coco}' but this sweep is labelled corpus '${CORPUS_NAME}'" \
      "— refusing to publish one corpus's numbers under another's name (sync it first)"
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
    ssh "${target}" "gcc -O2 -o '${REMOTE_BIN_DIR}/turbojpeg_bench' '${REMOTE_BIN_DIR}/bench.c' -ldl -lm"
  fi
  # stb + wuffs benches: statically compiled cross-binaries for either arch
  # (zig cc; their pinned single-file sources are fetched by `make deps`).
  # Non-fatal: a missing zig / offline fetch loses the stb+wuffs arms for
  # this target (their runs FAIL individually) without killing the sweep.
  if make -C "${BENCH_WS}/modules/stb" "${arch}" \
    && make -C "${BENCH_WS}/modules/wuffs" "${arch}"; then
    scp -q "${BENCH_WS}/modules/stb/build/stb_bench.${arch}" \
      "${target}:${REMOTE_BIN_DIR}/stb_bench" || true
    scp -q "${BENCH_WS}/modules/wuffs/build/wuffs_bench.${arch}" \
      "${target}:${REMOTE_BIN_DIR}/wuffs_bench" || true
  else
    echo "  WARN: stb/wuffs cross-build failed; their arms will FAIL on ${target}"
  fi
  ssh "${target}" "chmod +x ${REMOTE_BIN_DIR}/hal_cpu ${REMOTE_BIN_DIR}/rust_jpeg ${REMOTE_BIN_DIR}/turbojpeg_bench ${REMOTE_BIN_DIR}/stb_bench ${REMOTE_BIN_DIR}/wuffs_bench"

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
        run_arm stb "${round}" "${fmt}" \
          "./stb_bench --limit ${LIMIT} --warmup ${WARMUP} --board '${target}' \
           --decode-only --format rgb"
        run_arm wuffs "${round}" "${fmt}" \
          "./wuffs_bench --limit ${LIMIT} --warmup ${WARMUP} --board '${target}' \
           --decode-only --format rgb"
      fi
    done
  done

  capture_provenance "${target}" "${out_dir}"

  {
    echo "==== ${target} ${CORPUS_NAME} median-of-${ROUNDS} (n=${LIMIT}, pin=${PIN}, profile=${PROFILE})"
    for arm in hal hal_fast tj_islow tj_ifast zune image stb wuffs; do
      for fmt in yuv rgb; do
        case "${arm}" in
          image | stb | wuffs) [[ "${fmt}" == yuv ]] && continue ;;
        esac
        vals=""
        rounds=""
        for round in $(seq 1 "${ROUNDS}"); do
          f="${out_dir}/r${round}_${arm}_${fmt}.log"
          [[ -f "${f}" ]] || continue
          p="$(parse_p50 "${f}" || true)"
          [[ -z "${p}" ]] && continue
          rounds="${rounds}${rounds:+, }r${round}=${p}"
          vals="${vals}${vals:+ }${p}"
        done
        if [[ -n "${vals}" ]]; then
          read -r med lo hi < <(printf '%s\n' ${vals} | median_spread)
          n_rounds="$(printf '%s\n' ${vals} | wc -l | tr -d ' ')"
          short=""
          [[ "${n_rounds}" != "${ROUNDS}" ]] && short=" [only ${n_rounds}/${ROUNDS} rounds]"
          printf "  %-10s %-4s  median=%s ms  spread=[%s,%s]  (%s)%s\n" \
            "${arm}" "${fmt}" "${med}" "${lo}" "${hi}" "${rounds}" "${short}"
        else
          printf "  %-10s %-4s  median=?  (no p50)\n" "${arm}" "${fmt}"
        fi
      done
    done
  } | tee "${out_dir}/summary.txt"

  echo "OK: ${out_dir}"
done
