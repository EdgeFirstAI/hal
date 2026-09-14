#!/usr/bin/env bash
# Fetch the org license-policy scripts so local `make sbom` enforces exactly the
# policy CI enforces.
#
# hal used to keep its own copies of generate_sbom.sh and check_license_policy.py.
# They drifted from the org copies, so the Quick tier and `make sbom` disagreed
# about the same dependency tree. The policy has one home now: EdgeFirstAI/.github.
#
# The commit is read from the uses: pin in ci.yml, so there is no second SHA to
# maintain. Prints the directory holding the scripts.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dest="${root}/target/ci-scripts"

sha="$(grep -oE 'EdgeFirstAI/\.github/\.github/workflows/[^[:space:]]+@[0-9a-f]{40}' \
    "${root}/.github/workflows/ci.yml" | head -1 | sed 's/.*@//')"
if [[ -z "${sha}" ]]; then
    echo "error: no EdgeFirstAI/.github pin found in .github/workflows/ci.yml" >&2
    exit 1
fi

stamp="${dest}/.sha"
if [[ -f "${stamp}" && "$(cat "${stamp}")" == "${sha}" ]]; then
    echo "${dest}"
    exit 0
fi

mkdir -p "${dest}"
base="https://raw.githubusercontent.com/EdgeFirstAI/.github/${sha}/.github/scripts"
for script in generate_sbom.sh check_license_policy.py validate_notice.py \
              scancode_packages_to_cyclonedx.py; do
    if ! curl -fsSL -o "${dest}/${script}" "${base}/${script}"; then
        echo "error: could not fetch ${script} from EdgeFirstAI/.github@${sha}" >&2
        echo "       (network required the first time; cached in ${dest} after)" >&2
        exit 1
    fi
done
chmod +x "${dest}/generate_sbom.sh"
echo "${sha}" > "${stamp}"
echo "${dest}"
