#!/usr/bin/env bash
# Caller pre-command for shared rust-quick / rust-full jobs.
# Installs host deps, merges crate LFS testdata, fetches ANGLE on Apple/Windows,
# and on Linux arms the DMA require gate when the heap allocates
# (dma-heap-setup.sh).
#
# Environment the shared workflows guarantee to a pre-command: the caller
# checkout as working directory, GH_TOKEN / GITHUB_TOKEN, and GITHUB_ENV /
# GITHUB_PATH for exporting state to later steps.
#
#   SKIP_PACKAGES=1   skip apt (self-hosted boards already have the toolchain)
#   SKIP_TESTDATA=1   skip the crate testdata merge, but still export the path.
#                     Board jobs take the merged tree from the ci-testdata
#                     artifact instead; their checkout has LFS pointers, not
#                     content, so merging there would stage stub files that the
#                     artifact then has to overwrite.
set -euo pipefail

persist() {
    local key="$1" value="$2"
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        echo "${key}=${value}" >> "${GITHUB_ENV}"
    fi
    export "${key}=${value}"
}

merge_testdata() {
    mkdir -p testdata
    if [[ "${SKIP_TESTDATA:-0}" == "1" ]]; then
        echo "ci-setup: skipping testdata merge (SKIP_TESTDATA)"
    else
        for dir in crates/*/testdata; do
            [[ -d "${dir}" ]] || continue
            while IFS= read -r -d '' f; do
                rel="${f#"${dir}"/}"
                if [[ -e "testdata/${rel}" ]]; then
                    echo "::error::${dir}/${rel} collides with an existing testdata/${rel}"
                    exit 1
                fi
            done < <(find "${dir}" -type f -print0)
            cp -a "${dir}/." testdata/
        done
    fi
    persist EDGEFIRST_TESTDATA_DIR "${PWD}/testdata"
}

os="$(uname -s)"
case "${os}" in
    Linux*)
        if [[ "${SKIP_PACKAGES:-0}" == "1" || "${RUNNER_ENVIRONMENT:-}" == "self-hosted" ]]; then
            echo "ci-setup: skipping apt (SKIP_PACKAGES or self-hosted board)"
        else
            sudo apt-get update
            sudo apt-get install -y clang libclang-dev libopencv-dev pkg-config nasm
        fi
        merge_testdata
        # Opens the DMA heap to the job user and, only when a probe allocation
        # succeeds, exports HAL_TEST_REQUIRE_DMA=1 through $GITHUB_ENV so the
        # test step fails instead of skipping DMA tests. A pre-command's own
        # exports do not reach later steps. Never fails this script.
        bash "$(dirname "${BASH_SOURCE[0]}")/dma-heap-setup.sh" \
            || echo "::warning::dma-heap-setup.sh failed; DMA tests may skip"
        ;;
    Darwin*)
        merge_testdata
        bash scripts/fetch-angle.sh
        persist EDGEFIRST_ANGLE_PATH "${PWD}/target/angle/macos-flat-lib"
        ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        merge_testdata
        bash scripts/fetch-angle.sh --windows
        persist EDGEFIRST_ANGLE_PATH "${PWD}/target/angle/windows-x64/bin"
        ;;
    *)
        echo "::warning::ci-setup.sh: unknown OS '${os}'; testdata merge only"
        merge_testdata
        ;;
esac
