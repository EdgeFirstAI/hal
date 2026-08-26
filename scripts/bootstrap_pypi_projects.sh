#!/usr/bin/env bash
# One-time bootstrap for the five PyPI distributions introduced in 0.29.
#
# PyPI Trusted Publishing can only be configured on a project that already
# exists, so each new name needs one manual upload with a local API token.
# After this runs, configure the Trusted Publisher for each project and the
# release workflow takes over — no token is needed again.
#
#   Owner: EdgeFirstAI   Repository: hal
#   Workflow: release.yml   Environment: pypi
#
# Usage:
#   export TWINE_USERNAME=__token__
#   export TWINE_PASSWORD=pypi-...        # account-scoped token; project-scoped
#                                         # tokens cannot create a new project
#   scripts/bootstrap_pypi_projects.sh [--dry-run]
set -euo pipefail

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

if ! command -v twine >/dev/null && ! python3 -c "import twine" 2>/dev/null; then
    echo "ERROR: pip install twine"; [ "$DRY" -eq 0 ] && exit 1
fi
if [ "$DRY" -eq 0 ] && [ -z "${TWINE_PASSWORD:-}" ]; then
    echo "ERROR: TWINE_PASSWORD is unset. Use an ACCOUNT-scoped token —"
    echo "       a project-scoped one cannot create a project that does not exist."
    exit 1
fi

[ -d target/wheels ] || { echo "ERROR: no target/wheels; run 'make wheel' first"; exit 1; }

# Dependency order, same as the release workflow: the core first, since every
# sibling declares a ~= dependency on it.
for pkg in tensor codec image decoder tracker; do
    files=(target/wheels/edgefirst_${pkg}-*)
    if [ ! -e "${files[0]}" ]; then
        echo "ERROR: no artifacts for edgefirst-${pkg}"; exit 1
    fi
    echo "==> edgefirst-${pkg}"
    printf '    %s\n' "${files[@]##*/}"
    if [ "$DRY" -eq 1 ]; then
        echo "    (dry run — not uploading)"
    else
        twine upload --non-interactive --skip-existing "${files[@]}"
    fi
done

cat <<'NEXT'

Next, for each of edgefirst-tensor / -codec / -image / -decoder / -tracker:
  https://pypi.org/manage/project/<name>/settings/publishing/
  Owner: EdgeFirstAI   Repository: hal
  Workflow: release.yml   Environment: pypi

Once all five are configured, revoke the account-scoped token — the release
workflow authenticates through Trusted Publishing and needs no secret.
NEXT
