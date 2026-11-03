#!/usr/bin/env bash
# Optional upload of each new PyPI distribution with a local API token.
#
# A new distribution normally needs no upload: add a pending publisher at
# https://pypi.org/manage/account/publishing/ and the first tagged release
# creates the project. A pending publisher does not reserve the name, so run
# this only to claim a name before that release, then configure the project's
# Trusted Publisher; the release workflow needs no token after that.
#
#   Owner: EdgeFirstAI   Repository: hal
#   Workflow: publish.yml   Environment: pypi
#
# Usage:
#   export TWINE_USERNAME=__token__
#   export TWINE_PASSWORD=pypi-...        # account-scoped token; project-scoped
#                                         # tokens cannot create a new project
#   scripts/bootstrap_pypi_projects.sh [--dry-run]
#   PACKAGES=<pkg> scripts/bootstrap_pypi_projects.sh   # one distribution, e.g. kernels
set -euo pipefail

DRY=0
[[ "${1:-}" = "--dry-run" ]] && DRY=1

if ! command -v twine >/dev/null && ! python3 -c "import twine" 2>/dev/null; then
    echo "ERROR: pip install twine" >&2; [[ "$DRY" -eq 0 ]] && exit 1
fi
if [[ "$DRY" -eq 0 ]] && [[ -z "${TWINE_PASSWORD:-}" ]]; then
    echo "ERROR: TWINE_PASSWORD is unset. Use an ACCOUNT-scoped token —" >&2
    echo "       a project-scoped one cannot create a project that does not exist." >&2
    exit 1
fi

[[ -d target/wheels ]] || { echo "ERROR: no target/wheels; run 'make wheel' first" >&2; exit 1; }

# Dependency order, same as the release workflow: the core first, since every
# sibling declares a ~= dependency on it.
for pkg in ${PACKAGES:-tensor codec image decoder tracker kernels}; do
    files=(target/wheels/edgefirst_${pkg}-*)
    if [[ ! -e "${files[0]}" ]]; then
        echo "ERROR: no artifacts for edgefirst-${pkg}" >&2; exit 1
    fi
    echo "==> edgefirst-${pkg}"
    printf '    %s\n' "${files[@]##*/}"
    if [[ "$DRY" -eq 1 ]]; then
        echo "    (dry run — not uploading)"
    else
        twine upload --non-interactive --skip-existing "${files[@]}"
    fi
done

cat <<'NEXT'

Next, for each project uploaded above:
  https://pypi.org/manage/project/<name>/settings/publishing/
  Owner: EdgeFirstAI   Repository: hal
  Workflow: publish.yml   Environment: pypi

Once every new project is configured, revoke the account-scoped token — the release
workflow authenticates through Trusted Publishing and needs no secret.
NEXT
