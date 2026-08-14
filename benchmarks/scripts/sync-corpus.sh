#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
# SPDX-License-Identifier: Apache-2.0
#
# Rsync any benchmark corpus directory to boards, mirroring sync-coco.sh's
# host handling. Made for the control corpora (val2017-yuv420, val2017-dri,
# CLIC2025 jpeg-yuv420/yuv444/-dri): each lands at /data/corpora/<basename>
# (fallback ~/corpora/<basename>), and a sweep is pointed at it with
#   EDGEFIRST_BENCH_COCO_REMOTE=/data/corpora/<basename> ./decode-ab-sweep.sh …
#
# Usage:
#   ./benchmarks/scripts/sync-corpus.sh --src DIR [host...]
#
# Env:
#   EDGEFIRST_BENCH_ORIN_FALLBACK   fallback host for orin-nano (as sync-coco.sh)

set -euo pipefail

SRC=""
ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --src) SRC="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
[[ -n "${SRC}" && -d "${SRC}" ]] || { echo "pass --src DIR (got: '${SRC}')" >&2; exit 1; }
NAME="$(basename "${SRC%/}")"

if [[ ${#ARGS[@]} -gt 0 ]]; then
  TARGETS=("${ARGS[@]}")
else
  TARGETS=(imx8mp-frdm imx95-frdm imx95-pro rpi5-hailo orin-nano)
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
  echo "==> sync ${NAME} → ${target}"
  echo "============================================"
  if ! host="$(resolve_target "${target}")"; then
    echo "SKIP: ${target} unreachable"
    continue
  fi
  [[ "${host}" != "${target}" ]] && echo "(using fallback host ${host})"

  remote_dir="$(ssh "${host}" "bash -s" <<EOS
set -euo pipefail
pref='/data/corpora/${NAME}'
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
mkdir -p "\$HOME/corpora/${NAME}"
echo "\$HOME/corpora/${NAME}"
EOS
)"
  echo "  remote: ${host}:${remote_dir}"
  # Minimal Yocto images ship without rsync; stream a tar instead (corpora are
  # immutable, so losing incremental transfer costs nothing on a re-sync of an
  # already-complete directory — the JPEG count check below short-circuits it).
  src_count="$(find "${SRC}" -maxdepth 1 \( -iname '*.jpg' -o -iname '*.jpeg' \) | wc -l | tr -d ' ')"
  remote_count() {
    ssh "${host}" "find '${remote_dir}' -maxdepth 1 \\( -iname '*.jpg' -o -iname '*.jpeg' \\) | wc -l" | tr -d ' '
  }
  if [[ "$(remote_count)" == "${src_count}" ]]; then
    echo "OK: ${host} already has ${src_count} JPEGs at ${remote_dir}"
    continue
  fi
  if ssh "${host}" 'command -v rsync >/dev/null'; then
    rsync -a --info=progress2 "${SRC}/" "${host}:${remote_dir}/"
  else
    echo "  (no rsync on ${host}; streaming tar)"
    tar -C "${SRC}" -cf - . | ssh "${host}" "tar -C '${remote_dir}' -xf -"
  fi
  count="$(remote_count)"
  echo "OK: ${host} has ${count} JPEGs at ${remote_dir}"
done
