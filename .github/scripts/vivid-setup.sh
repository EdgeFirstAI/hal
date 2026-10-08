#!/usr/bin/env bash
# Load the kernel's virtual V4L2 capture driver, vivid, on a GitHub-hosted
# Ubuntu runner and arm the HAL_TEST_REQUIRE_VIVID gate only when its nodes
# appear. crates/tensor/tests/vivid_capture_import.rs imports real padded
# V4L2 capture buffers from it.
#
# vivid ships in linux-modules-extra for the runner's kernel, which is not
# preinstalled. When the archive has no such package (it has not caught up
# with the runner image) the tests skip, and this says why.
#
# Only ephemeral GitHub-hosted VMs are touched: loading a kernel module and
# opening its nodes to every user would outlive the job on a persistent
# machine. Never fails the job.
#
# Exports, via $GITHUB_ENV when it is set and to this shell otherwise:
#   HAL_TEST_REQUIRE_VIVID=1          only when the vivid nodes appeared
#   HAL_CI_VIVID_STATE=required|unavailable|n/a
# Appends one line to $GITHUB_STEP_SUMMARY when it is set.
set -uo pipefail

persist() {
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        echo "$1=$2" >> "${GITHUB_ENV}"
    fi
    export "$1=$2"
}

report() {
    local state="$1" line="$2"
    persist HAL_CI_VIVID_STATE "${state}"
    echo "vivid-setup: ${line}"
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        echo "${line}" >> "${GITHUB_STEP_SUMMARY}"
    fi
}

if [[ "$(uname -s)" != Linux* || "${RUNNER_ENVIRONMENT:-}" != "github-hosted" ]]; then
    report n/a "vivid: n/a (not a GitHub-hosted Linux runner)"
    exit 0
fi

pkg="linux-modules-extra-$(uname -r)"
if ! apt-cache show "${pkg}" > /dev/null 2>&1; then
    report unavailable "vivid: unavailable — vivid tests skipped (${pkg} is not in the archive)"
    exit 0
fi
if ! sudo apt-get install -y -q "${pkg}" \
    || ! sudo modprobe vivid n_devs=2 node_types=0x1,0x1 multiplanar=1,2; then
    report unavailable "vivid: unavailable — vivid tests skipped (installing or loading vivid failed)"
    exit 0
fi
# udev creates and permissions the nodes after modprobe returns; wait for it,
# or it resets the modes changed below.
sudo udevadm settle
nodes=()
for d in /sys/class/video4linux/video*; do
    [[ "$(cat "${d}/name" 2>/dev/null)" == vivid-* ]] && nodes+=("/dev/$(basename "${d}")")
done
if [[ ${#nodes[@]} -eq 0 ]]; then
    report unavailable "vivid: unavailable — vivid tests skipped (no vivid nodes after modprobe)"
    exit 0
fi
sudo chmod a+rw "${nodes[@]}"
persist HAL_TEST_REQUIRE_VIVID 1
report required "vivid: required (${nodes[*]})"
