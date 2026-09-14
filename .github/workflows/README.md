# GitHub Actions Workflows

HAL uses the three-tier CI model from
[CICD Pipelines](https://au-zone.atlassian.net/wiki/spaces/EAM/pages/2750906369/CICD+Pipelines).
Reusable jobs live in [EdgeFirstAI/.github](https://github.com/EdgeFirstAI/.github)
and are pinned by commit SHA in `uses:`.

## Tiers

| Tier | When | What |
|------|------|------|
| **Quick** | every non-draft PR push, and push to `main` | fmt (hard gate), clippy, nextest `-j 1`, ruff, dependency license — one Linux job, ~20 min budget |
| **Full** | reviewer adds `ci:full` (sticky on later pushes), `workflow_dispatch`, or merge queue | host matrix, HAL extras (iOS/Android/software-GL/C-API/Python), coverage, scancode, Sonar |
| **Hardware** | `ci:hardware` (or `ci:full`) on a same-repo PR | on-target i.MX 8M Plus via the shared three-phase pattern |
| **Nightly** | 03:17 UTC if `main` moved | Full + cargo hack + audit + G13 differential |
| **Release** | `vX.Y.Z` tag from `tag-release.yml` | wheels, C-API, PyPI, crates.io, GitHub Release |

Draft PRs run nothing. Open as draft, mark ready when you want Quick.

Add **`ci:full`** before approving when the PR touches the build system, `unsafe`, FFI/C-API, GL/G2D/DMA, platform `cfg`, dependencies, or public API. Skip it for docs and comment-only changes. Release PRs (`release/X.Y.Z` → `main`) must carry `ci:full`.

`ci-gate` is the only required status check (org `protect-main-ci` ruleset). Docs-only PRs skip Quick/Full; the gate still passes. hal is in the org `protect-main` / `protect-main-ci` / `protect-release-tags` rulesets; the hand-made repo-level duplicates are retired.

## Workflows

| File | Role |
|------|------|
| `ci.yml` | Gate + Quick + Full callers |
| `hal-full.yml` | HAL-only Full lanes (`workflow_call`) |
| `nightly.yml` | Change-gated nightly |
| `differential.yml` | G13; nightly and manual only |
| `tag-release.yml` | Shared caller: merged `release/X.Y.Z` → annotated `vX.Y.Z` |
| `release.yml` | HAL publish path (wheels / C-API / PyPI / crates). Filename and `pypi` environment stay here — PyPI Trusted Publishing cannot use a reusable workflow in another repo. |
| `sbom.yml` | Caller of the shared full scancode SBOM. `release.yml` attaches it. |
| `sonar.yml` | Coverage upload, called by both `ci.yml` (PR decoration) and `nightly.yml` (`main` baseline) |
| `benchmark.yml` | `workflow_dispatch` only |

Never tag by hand. Do not introduce `-xlarge` / `-8core` labels unless a Full lane measured over 20 minutes on a standard runner; record that exception.

## Shared pin

Callers pin `cf505b2dbc5fd784a7acbf4ec8ca1e821c02dda3`. The SHA appears only in `uses:`, and Dependabot bumps it. The shared workflows find their own composite actions through `job.workflow_repository` / `job.workflow_sha`, so there is no second value to keep in sync.

The license policy lives in `EdgeFirstAI/.github` and is the only copy. `make sbom` fetches it at the pinned commit (`.github/scripts/fetch-ci-scripts.sh`), so local runs and CI enforce the same policy.

## Runners

Quick and Full host lanes use free standard hosted runners (`ubuntu-24.04`, `ubuntu-24.04-arm`, `macos-latest`, `windows-latest`). The board is `nxp-imx8mp-latest`. Larger runners are a recorded exception only.
