#!/usr/bin/env bash
# Fail when a caller pins EdgeFirstAI/.github at one SHA in `uses:` and a
# different SHA in `shared-sha`. Dependabot cannot bump the input.
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mismatch=0

while IFS= read -r -d '' file; do
    mapfile -t uses < <(grep -oE 'EdgeFirstAI/\.github/\.github/workflows/[^[:space:]]+@[0-9a-f]{40}' "${file}" \
        | sed 's/.*@//' | sort -u)
    mapfile -t shas < <(grep -E 'shared-sha:' "${file}" | grep -oE '[0-9a-f]{40}' | sort -u)
    if [[ ${#uses[@]} -eq 0 ]]; then
        continue
    fi
    if [[ ${#shas[@]} -eq 0 ]]; then
        # Callers such as tag-release.yml pin uses: only (callee has no shared-sha).
        continue
    fi
    if [[ ${#uses[@]} -ne 1 || ${#shas[@]} -ne 1 || "${uses[0]}" != "${shas[0]}" ]]; then
        echo "::error file=${file}::uses pin (${uses[*]-none}) != shared-sha (${shas[*]-none})"
        mismatch=1
    fi
done < <(find "${root}/.github/workflows" -name '*.yml' -print0)

if [[ "${mismatch}" -ne 0 ]]; then
    echo "Pin uses: and shared-sha to the same EdgeFirstAI/.github commit."
    exit 1
fi
echo "shared-sha matches uses: pin"
