#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Rsync COCO val2017 JPEGs to boards for on-target latency sweeps.
#
# Usage:
#   ./benchmarks/scripts/sync-coco.sh [host...]
#
# Env:
#   EDGEFIRST_BENCH_COCO          local source (default: ~/Datasets/COCO/val2017)
#   EDGEFIRST_BENCH_COCO_REMOTE   preferred remote dest (default: /data/coco/val2017)

set -euo pipefail

SRC="${EDGEFIRST_BENCH_COCO:-${HOME}/Datasets/COCO/val2017}"
REMOTE_PREFERRED="${EDGEFIRST_BENCH_COCO_REMOTE:-/data/coco/val2017}"

if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
else
  TARGETS=(imx8mp-frdm imx95-frdm imx95-pro rpi5-hailo orin-nano)
fi

if [[ ! -d "${SRC}" ]]; then
  echo "ERROR: local COCO dir not found: ${SRC}" >&2
  exit 1
fi

resolve_target() {
  local t="$1"
  if ssh -o BatchMode=yes -o ConnectTimeout=5 "$t" 'true' 2>/dev/null; then
    echo "$t"
    return 0
  fi
  local fallback="${EDGEFIRST_BENCH_ORIN_FALLBACK:-}"
  if [[ "$t" == "orin-nano" && -n "${fallback}" ]] \
      && ssh -o BatchMode=yes -o ConnectTimeout=5 "${fallback}" 'true' 2>/dev/null; then
    echo "${fallback}"
    return 0
  fi
  return 1
}

for target in "${TARGETS[@]}"; do
  echo "============================================"
  echo "==> sync COCO → ${target}"
  echo "============================================"
  if ! host="$(resolve_target "${target}")"; then
    echo "SKIP: ${target} unreachable"
    continue
  fi
  if [[ "${host}" != "${target}" ]]; then
    echo "(using fallback host ${host})"
  fi

  # Prefer /data/coco/val2017; fall back to ~/coco/val2017.
  # Require the rsync user can create a probe file (mkdir -p alone is insufficient
  # when a root-owned /data/coco exists).
  remote_dir="$(ssh "${host}" "bash -s" <<EOS
set -euo pipefail
pref='${REMOTE_PREFERRED}'
try_dir() {
  local d="\$1"
  mkdir -p "\$d" 2>/dev/null || sudo mkdir -p "\$d" 2>/dev/null || return 1
  sudo chown "\$(id -un):\$(id -gn)" "\$d" 2>/dev/null || true
  local probe="\$d/.edgefirst_bench_write_test"
  if touch "\$probe" 2>/dev/null; then
    rm -f "\$probe"
    echo "\$d"
    return 0
  fi
  return 1
}
if try_dir "\$pref"; then
  exit 0
fi
mkdir -p "\$HOME/coco/val2017"
echo "\$HOME/coco/val2017"
EOS
)"
  echo "  remote: ${host}:${remote_dir}"
  rsync -a --info=progress2 "${SRC}/" "${host}:${remote_dir}/"
  count="$(ssh "${host}" "find '${remote_dir}' -maxdepth 1 \\( -iname '*.jpg' -o -iname '*.jpeg' \\) | wc -l")"
  echo "OK: ${host} has ${count} JPEGs at ${remote_dir}"
done
