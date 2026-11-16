#!/usr/bin/env bash
# Caller pre-command for shared rust-quick / rust-full jobs.
# Installs host deps, merges crate LFS testdata, fetches ANGLE on Apple/Windows,
# and on Linux arms the DMA require gate when the heap allocates
# (dma-heap-setup.sh) and the vivid gate when vivid loads (vivid-setup.sh).
#
# Environment the shared workflows guarantee to a pre-command: the caller
# checkout as working directory, GH_TOKEN / GITHUB_TOKEN, and GITHUB_ENV /
# GITHUB_PATH for exporting state to later steps.
#
#   SKIP_PACKAGES=1   skip apt
#   SKIP_TESTDATA=1   skip the crate testdata merge, but still export the path
#   BOARD=1           arm the on-target gates from what the board has:
#                     HAL_TEST_REQUIRE_GL=1 with a DRM render node,
#                     HAL_TEST_REQUIRE_G2D=1 with libg2d.so.2 installed
#                     (HAL_TEST_SKIP_G2D=1 instead on i.MX 95),
#                     EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS=1 with /dev/galcore.
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
            sudo apt-get install -y clang libclang-dev pkg-config nasm
        fi
        # Removes unused SDKs from a hosted image.
        if [[ "${RUNNER_ENVIRONMENT:-}" == "github-hosted" ]]; then
            sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache/CodeQL
            df -h / | tail -1
        fi
        merge_testdata
        # Opens the DMA heap to the job user and, when a probe allocation
        # succeeds, exports HAL_TEST_REQUIRE_DMA=1 through $GITHUB_ENV.
        bash "$(dirname "${BASH_SOURCE[0]}")/dma-heap-setup.sh" \
            || echo "::warning::dma-heap-setup.sh failed; DMA tests may skip"
        # Loads vivid on hosted runners and, when its nodes appear, exports
        # HAL_TEST_REQUIRE_VIVID=1.
        bash "$(dirname "${BASH_SOURCE[0]}")/vivid-setup.sh" \
            || echo "::warning::vivid-setup.sh failed; vivid tests may skip"
        if [[ "${BOARD:-0}" == "1" ]]; then
            # Requires GL on a board with a render node, and skips the known
            # Vivante driver bugs on a board with /dev/galcore.
            if compgen -G "/dev/dri/renderD*" >/dev/null; then
                persist HAL_TEST_REQUIRE_GL 1
            fi
            if [[ -e /dev/galcore ]]; then
                persist EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS 1
            fi
            # Requires G2D on a board with libg2d.so.2; on an i.MX 95 the G2D
            # tests skip.
            if [[ "$(cat /sys/devices/soc0/soc_id 2>/dev/null)" == "i.MX95" ]]; then
                persist HAL_TEST_SKIP_G2D 1
            elif compgen -G "/usr/lib/libg2d.so.2*" >/dev/null || compgen -G "/usr/lib/*/libg2d.so.2*" >/dev/null; then
                persist HAL_TEST_REQUIRE_G2D 1
            fi
            echo "ci-setup: board gates HAL_TEST_REQUIRE_GL=${HAL_TEST_REQUIRE_GL:-unset}" \
                "HAL_TEST_REQUIRE_G2D=${HAL_TEST_REQUIRE_G2D:-unset}" \
                "HAL_TEST_SKIP_G2D=${HAL_TEST_SKIP_G2D:-unset}" \
                "EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS=${EDGEFIRST_SKIP_VIVANTE_KNOWN_BUGS:-unset}"
        fi
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
