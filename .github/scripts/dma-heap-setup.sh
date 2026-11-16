#!/usr/bin/env bash
# Open the Linux DMA system heap to the job user, check that it allocates, and
# arm the HAL_TEST_REQUIRE_DMA gate only when it does.
#
# Makes the node world read-write when it can (root, or passwordless sudo) on
# a GitHub-hosted runner, then allocates and maps one page through the heap
# ioctl. A successful allocation sets HAL_TEST_REQUIRE_DMA=1, which makes a
# DMA test that would skip fail (see crates/tensor/TESTING.md).
#
# Never fails the job: a missing heap, sudo or python3 leaves the gate unset
# and prints the cause.
#
# Exports, via $GITHUB_ENV when it is set and to this shell otherwise:
#   HAL_TEST_REQUIRE_DMA=1          only when the probe allocated
#   HAL_CI_DMA_STATE=required|unavailable|n/a
# Appends one line to $GITHUB_STEP_SUMMARY when it is set.
#
#   DMA_HEAP_NODE   heap to open and probe (default /dev/dma_heap/system)
set -uo pipefail

node="${DMA_HEAP_NODE:-/dev/dma_heap/system}"

persist() {
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        echo "$1=$2" >> "${GITHUB_ENV}"
    fi
    export "$1=$2"
}

report() {
    local state="$1" line="$2"
    persist HAL_CI_DMA_STATE "${state}"
    echo "dma-heap-setup: ${line}"
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        echo "${line}" >> "${GITHUB_STEP_SUMMARY}"
    fi
}

if [[ "$(uname -s)" != Linux* ]]; then
    report n/a "DMA: n/a (no DMA heap on $(uname -s))"
    exit 0
fi

if [[ ! -e "${node}" ]]; then
    report unavailable "DMA: unavailable — DMA tests skipped (${node} does not exist)"
    exit 0
fi

# Permissions are changed on a GitHub-hosted runner only; elsewhere the
# script only probes.
if [[ ! -r "${node}" || ! -w "${node}" ]] && [[ "${RUNNER_ENVIRONMENT:-}" != "github-hosted" ]]; then
    echo "dma-heap-setup: ${node} is not read-write for $(id -un); not changing permissions outside a GitHub-hosted runner"
elif [[ ! -r "${node}" || ! -w "${node}" ]]; then
    if [[ "$(id -u)" == 0 ]]; then
        chmod a+rw "${node}" || true
    elif command -v sudo > /dev/null 2>&1 && sudo -n true > /dev/null 2>&1; then
        sudo -n chmod a+rw "${node}" || true
    else
        echo "dma-heap-setup: ${node} is not read-write for $(id -un) and there is no passwordless sudo"
    fi
fi
ls -l "$(dirname "${node}")" || true

if ! command -v python3 > /dev/null 2>&1; then
    report unavailable "DMA: unavailable — DMA tests skipped (no python3 to probe ${node})"
    exit 0
fi

# DMA_HEAP_IOCTL_ALLOC is _IOWR('H', 0, struct dma_heap_allocation_data), and
# that struct is { u64 len; u32 fd; u32 fd_flags; u64 heap_flags; }.
probe_output="$(python3 - "${node}" 2>&1 <<'PY'
import fcntl
import mmap
import os
import struct
import sys

DMA_HEAP_IOCTL_ALLOC = 0xC0184800
LAYOUT = "=QIIQ"
SIZE = 4096

heap = os.open(sys.argv[1], os.O_RDWR | os.O_CLOEXEC)
try:
    request = bytearray(struct.pack(LAYOUT, SIZE, 0, os.O_RDWR | os.O_CLOEXEC, 0))
    fcntl.ioctl(heap, DMA_HEAP_IOCTL_ALLOC, request, True)
    buf = struct.unpack(LAYOUT, bytes(request))[1]
    try:
        view = mmap.mmap(buf, SIZE)
        pattern = bytes(range(256)) * (SIZE // 256)
        view[:] = pattern
        if view[:] != pattern:
            sys.exit("read-back mismatch")
        view.close()
    finally:
        os.close(buf)
finally:
    os.close(heap)
print(f"allocated and mapped {SIZE} bytes")
PY
)"
probe_status=$?

if [[ ${probe_status} -eq 0 ]]; then
    persist HAL_TEST_REQUIRE_DMA 1
    if [[ "${RUNNER_ENVIRONMENT:-}" == "github-hosted" ]]; then
        report required "DMA: required (system heap, host-coherent)"
    else
        report required "DMA: required (${node})"
    fi
else
    reason="$(printf '%s' "${probe_output}" | tail -n 1)"
    report unavailable "DMA: unavailable — DMA tests skipped (${node}: ${reason})"
fi
exit 0
