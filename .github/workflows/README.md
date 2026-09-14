# GitHub Actions Workflows

HAL uses the three-tier CI model from
[CICD Pipelines](https://au-zone.atlassian.net/wiki/spaces/EAM/pages/2750906369/CICD+Pipelines).
Reusable jobs live in [EdgeFirstAI/.github](https://github.com/EdgeFirstAI/.github)
and are pinned by commit SHA (`uses:` and `shared-sha` must match).

## Tiers

| Tier | When | What |
|------|------|------|
| **Quick** | every non-draft PR push, and push to `main` | fmt (hard gate), clippy, nextest `-j 1`, ruff check, dependency license — one Linux job, ~20 min budget |
| **Full** | reviewer adds `ci:full` (sticky on later pushes), `workflow_dispatch`, or merge queue | host matrix, HAL extras (iOS/Android/software-GL/C-API/Python), coverage, scancode, Sonar |
| **Hardware** | `ci:hardware` (or `ci:full`) on a same-repo PR | on-target i.MX 8M Plus via the shared three-phase pattern |
| **Nightly** | 03:17 UTC if `main` moved | Full + cargo hack + audit + G13 differential |
| **Release** | `vX.Y.Z` tag from `tag-release.yml` | wheels, C-API, PyPI, crates.io, GitHub Release |

Draft PRs run nothing. Open as draft, mark ready when you want Quick.

Add **`ci:full`** before approving when the PR touches the build system, `unsafe`, FFI/C-API, GL/G2D/DMA, platform `cfg`, dependencies, or public API. Skip it for docs and comment-only changes. Release PRs (`release/X.Y.Z` → `main`) must carry `ci:full`.

`ci-gate` is the only required status check (repo `protect-main` ruleset). Docs-only PRs skip Quick/Full; the gate still passes. Org `protect-release-tags` already applies; keep the repo tag ruleset until the org ruleset is confirmed as the single source.

## Workflows

| File | Role |
|------|------|
| `ci.yml` | Gate + Quick + Full callers |
| `hal-full.yml` | HAL-only Full lanes (`workflow_call`) |
| `nightly.yml` | Change-gated nightly |
| `differential.yml` | G13; nightly and manual only |
| `tag-release.yml` | Shared caller: merged `release/X.Y.Z` → annotated `vX.Y.Z` |
| `release.yml` | HAL publish path (wheels / C-API / PyPI / crates). Filename and `pypi` environment stay here — PyPI Trusted Publishing cannot use a reusable workflow in another repo. |
| `sbom.yml` | Generate SBOM (`workflow_call` / `workflow_dispatch`). `release.yml` attaches it. |
| `benchmark.yml` | `workflow_dispatch` only |

Never tag by hand. Do not introduce `-xlarge` / `-8core` labels unless a Full lane measured over 20 minutes on a standard runner; record that exception.

## Shared pin

Callers currently pin `dc93642641b94fa8971aa6275408f7ecca2b5a06` (pre-command hook, [EdgeFirstAI/.github#22](https://github.com/EdgeFirstAI/.github/pull/22)). Dependabot bumps action SHAs but not `shared-sha:`; keep them identical or `check-shared-sha.sh` fails.

## Runners

Quick and Full host lanes use free standard hosted runners (`ubuntu-24.04`, `ubuntu-24.04-arm`, `macos-latest`, `windows-latest`). The board is `nxp-imx8mp-latest`. Larger runners are a recorded exception only.
