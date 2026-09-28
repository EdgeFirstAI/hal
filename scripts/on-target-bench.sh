#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Run HAL's benchmarks on the board this runs on, from inside a bench bundle
# built by scripts/on-target-bundle.sh. The hardware-bench workflow runs it on
# a self-hosted runner; by hand:
#
#   ./scripts/on-target-bundle.sh bench aarch64 target/on-target-bundle/bench-aarch64
#   rsync -a --delete target/on-target-bundle/bench-aarch64/ <host>:/tmp/hal-bench/
#   ssh <host> /tmp/hal-bench/scripts/on-target-bench.sh /tmp/hal-bench-results
#
# Usage: <bundle>/scripts/on-target-bench.sh <results-dir>
#
# Env:
#   CASES            comma-separated case names to run (default: all); see
#                    CASES below
#   HAL_BOARD_ENTRY  the board entry this run is for, recorded in
#                    summary.json for the fleet roll-up
#
# Each case writes <case>.json (the harness's --json output) and <case>.txt.
# system.txt records what the numbers depend on: CPU, governors, GPU clocks
# and temperatures. Governors are read, never changed: a benchmark measures
# the board as it ships. summary.json lists every case's exit status and wall
# time. Exits non-zero if any case failed or its binary is missing.

set -uo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="${1:?usage: $0 <results-dir>}"
SELECT="${CASES:-}"

rm -rf "${RESULTS}"
mkdir -p "${RESULTS}"
RESULTS="$(cd "${RESULTS}" && pwd)"
export EDGEFIRST_TESTDATA_DIR="${BUNDLE}/testdata"

# name|environment|binary. The -opengl and -cpu pairs force one backend each,
# so a GL case that silently fell back to the CPU fails instead of reporting a
# CPU number as a GPU one. The names match the JSON files BENCHMARKS.md's
# tables are generated from (benchmarks/<platform>/<name>.json).
CASES=(
  "tensor||tensor_benchmark"
  "codec||codec_benchmark"
  "decoder||decoder_benchmark"
  "tracker||tracker_benchmark"
  "pipeline-opengl|EDGEFIRST_FORCE_BACKEND=opengl|pipeline_benchmark"
  "pipeline-cpu|EDGEFIRST_FORCE_BACKEND=cpu|pipeline_benchmark"
  "image-opengl|EDGEFIRST_FORCE_BACKEND=opengl|image_benchmark"
  "image-cpu|EDGEFIRST_FORCE_BACKEND=cpu|image_benchmark"
  "mask-opengl|EDGEFIRST_FORCE_BACKEND=opengl|mask_benchmark"
  "mask-cpu|EDGEFIRST_FORCE_BACKEND=cpu|mask_benchmark"
  "mask_decode||mask_decode_benchmark"
  "decode_pipeline||decode_pipeline_benchmark"
  "convert_matrix||convert_matrix_benchmark"
  "batch_convert||batch_convert_benchmark"
  "tiled_convert||tiled_convert_benchmark"
  "cpu_preprocess||cpu_preprocess_benchmark"
  "parallel_processors||parallel_processors_benchmark"
  "nv_path-sampler|EDGEFIRST_NV_CONVERT_PATH=sampler|nv_path_benchmark"
  "nv_path-shader|EDGEFIRST_NV_CONVERT_PATH=shader|nv_path_benchmark"
  "nvjpeg|EDGEFIRST_ENABLE_NVJPEG=1|nvjpeg_benchmark"
)

# nvJPEG is opt-in (EDGEFIRST_ENABLE_NVJPEG, set on its case) and engages only
# where CUDA and libnvjpeg load, so the case measures on a Jetson and skips
# cleanly elsewhere. libnvjpeg is not on a Jetson's default loader path.
for cuda in /usr/local/cuda/lib64 /usr/local/cuda/targets/aarch64-linux/lib; do
  [[ -d "${cuda}" ]] && export LD_LIBRARY_PATH="${cuda}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
done

{
  echo "date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "runner=${RUNNER_NAME:-}"
  echo "kernel=$(uname -srm)"
  [[ -f /etc/os-release ]] && sed -n 's/^PRETTY_NAME=/os=/p' /etc/os-release | tr -d '"'
  if command -v lscpu > /dev/null 2>&1; then
    lscpu | sed -n 's/^Model name: *\(.*\)/cpu=\1/p' | sort -u
  else
    sed -n 's/^\(model name\|CPU part\)[[:space:]]*: */cpu=/p' /proc/cpuinfo | sort -u
  fi
  echo "cpus=$(grep -c ^processor /proc/cpuinfo)"
  for p in /sys/devices/system/cpu/cpufreq/policy*; do
    [[ -d "${p}" ]] || continue
    echo "cpufreq.$(basename "${p}")=$(cat "${p}/related_cpus") $(cat "${p}/scaling_governor") $(cat "${p}/scaling_min_freq")-$(cat "${p}/scaling_max_freq")"
  done
  for d in /sys/class/devfreq/*; do
    [[ -e "${d}/governor" ]] || continue
    echo "devfreq.$(basename "${d}")=$(cat "${d}/governor") $(cat "${d}/min_freq" 2>/dev/null)-$(cat "${d}/max_freq" 2>/dev/null) cur=$(cat "${d}/cur_freq" 2>/dev/null)"
  done
  [[ -r /sys/class/kgsl/kgsl-3d0/gpu_model ]] && echo "gpu=$(cat /sys/class/kgsl/kgsl-3d0/gpu_model)"
  for z in /sys/class/thermal/thermal_zone*; do
    [[ -r "${z}/temp" ]] && echo "thermal.$(cat "${z}/type" 2>/dev/null || basename "${z}")=$(cat "${z}/temp")"
  done
  echo "mem=$(sed -n 's/^MemTotal: *//p' /proc/meminfo)"
  [[ -f "${BUNDLE}/MANIFEST" ]] && sed 's/^/bundle./' "${BUNDLE}/MANIFEST"
} > "${RESULTS}/system.txt" 2>/dev/null

status=0
records=()
for c in "${CASES[@]}"; do
  IFS='|' read -r name envs prefix <<< "${c}"
  if [[ -n "${SELECT}" && ",${SELECT// /}," != *",${name},"* ]]; then
    continue
  fi
  exe="$(ls "${BUNDLE}/bin/${prefix}"-* 2>/dev/null | head -n 1)"
  if [[ -z "${exe}" ]]; then
    echo "${name}: MISSING (no ${prefix} in the bundle)"
    records+=("${name}|missing|0")
    status=1
    continue
  fi
  echo "=== ${name}${envs:+ (${envs})}"
  t0=$(date +%s)
  # shellcheck disable=SC2086  # envs is empty or one KEY=VALUE word
  env ${envs} "${exe}" --bench --json "${RESULTS}/${name}.json" > "${RESULTS}/${name}.txt" 2>&1
  rc=$?
  secs=$(( $(date +%s) - t0 ))
  echo "    rc=${rc} ${secs}s"
  records+=("${name}|${rc}|${secs}")
  [[ ${rc} -eq 0 ]] || status=1
  # Let the GPU and CPU clocks settle between cases.
  sleep 5
done

if [[ ${#records[@]} -eq 0 ]]; then
  echo "error: CASES='${SELECT}' matched no case" >&2
  exit 1
fi

printf '%s\n' "${records[@]}" | python3 -c '
import json, os, sys
cases = []
for line in sys.stdin:
    name, rc, secs = line.rstrip("\n").split("|")
    cases.append({"case": name,
                  "status": "missing" if rc == "missing" else ("ok" if rc == "0" else "failed"),
                  "exit": None if rc == "missing" else int(rc),
                  "seconds": int(secs)})
json.dump({"entry": os.environ.get("HAL_BOARD_ENTRY", ""),
           "runner": os.environ.get("RUNNER_NAME", ""), "cases": cases},
          open(sys.argv[1], "w"), indent=2)
' "${RESULTS}/summary.json"

exit ${status}
