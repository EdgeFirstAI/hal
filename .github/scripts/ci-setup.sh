#!/usr/bin/env bash
# Caller pre-command for shared rust-quick / rust-full jobs.
# Installs host deps, merges crate LFS testdata, fetches ANGLE on Apple/Windows.
# Env exports persist via GITHUB_ENV when running in Actions.
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
        ;;
    Darwin*)
        merge_testdata
        bash scripts/fetch-angle.sh
        persist EDGEFIRST_ANGLE_PATH "${PWD}/target/angle/macos-flat-lib"
        persist GH_TOKEN "${GH_TOKEN:-${GITHUB_TOKEN:-}}"
        ;;
    MINGW*|MSYS*|CYGWIN*|Windows_NT)
        merge_testdata
        bash scripts/fetch-angle.sh --windows
        persist EDGEFIRST_ANGLE_PATH "${PWD}/target/angle/windows-x64/bin"
        persist GH_TOKEN "${GH_TOKEN:-${GITHUB_TOKEN:-}}"
        ;;
    *)
        echo "::warning::ci-setup.sh: unknown OS '${os}'; testdata merge only"
        merge_testdata
        ;;
esac
