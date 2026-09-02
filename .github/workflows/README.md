# GitHub Actions Workflows

This directory contains the CI/CD workflows for EdgeFirst HAL.

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GitHub Events                                  │
└─────────────────────────────────────────────────────────────────────────────┘
    │                │                  │                │                │
    │ Push/PR        │ Tag (vX.Y.Z)     │ Push/PR        │ PR merged      │ Manual
    │                │                  │ + Release      │ to main        │
    ▼                ▼                  ▼                ▼                ▼
┌──────────┐   ┌──────────────┐   ┌────────────┐   ┌────────────────┐  ┌──────────────┐
│ test.yml │   │ release.yml  │   │  sbom.yml  │   │ tag-release.yml│  │ benchmark.yml│
│ (CI)     │   │ (Publishing) │   │(Compliance)│   │  (Tagging)     │  │ (On-demand)  │
└──────────┘   └──────────────┘   └────────────┘   └────────────────┘  └──────────────┘
```

`tag-release.yml` closes the loop: merging a `release/**` PR into `main` pushes the
`vX.Y.Z` tag, which is what triggers `release.yml`.

## Workflows

### test.yml - Continuous Integration

**Triggers:** Push/PR to `main`, `develop`, or `release/**`

| Job | Runner | Purpose |
|-----|--------|---------|
| `checkout-lfs` | ubuntu-22.04 | Fetch Git LFS testdata once, share as an artifact |
| `doc-tests` | ubuntu-22.04-xlarge | Rust documentation tests |
| `build-and-test-x86` | ubuntu-22.04-xlarge | x86_64 build, test, Rust + Python + C API coverage |
| `build-and-test-macos` | macos-latest | ANGLE/IOSurface GL path, C API, coverage |
| `build-ios` | macos-latest | Clippy + build the native Rust API (device and simulator) |
| `build-android` | ubuntu-22.04 | Clippy + build the native Rust API (arm64 and x86_64) |
| `build-and-test-windows` | windows-latest | ANGLE/Direct3D 11 GL path (WARP in CI), C API, wheels, coverage |
| `software-gl-coverage` | ubuntu-22.04-xlarge | GL tests under Mesa llvmpipe, for coverage of the GL paths no hardware runner reaches |
| `build-arm` | ubuntu-22.04-arm-xlarge | Cross-build aarch64 test binaries (also feeds the hardware runner) |
| `test-arm` | ubuntu-22.04-arm | Run the aarch64 binaries, collect coverage |
| `hardware-test` | nxp-imx8mp-latest | On-target testing (G2D, DMA-heap, Vivante GL) |
| `process-hardware-coverage` | ubuntu-22.04-arm | Convert the hardware runner's profraw to LCOV |
| `sonarcloud` | ubuntu-22.04 | Aggregate all coverage, static analysis |

Three-phase on-target testing keeps the hardware runner doing only what it must:
`build-arm` cross-compiles, `hardware-test` runs the binaries on the board, and
`process-hardware-coverage` converts the raw profiling data back on a host with the
matching toolchain.

The iOS and Android lanes build the native Rust API only (`edgefirst-tensor`,
`-image`, `-codec`, `-decoder`, `-tracker`) — this repo's mobile responsibility ends
at "the Rust API compiles and lints clean" on those targets. Bindings, packaging,
and any C artifact for mobile belong to `mobile-sdk`, which binds to these crates
via boltffi. GitHub has no runner that can execute either GPU stack, so on-device
correctness is gated separately (Android via the internal hal-mobile Device Farm
harness — see [TESTING.md](../../TESTING.md#android-on-device-validation-device-farm)).

The macOS test lane fetches the signed ANGLE xcframeworks from the **public**
`EdgeFirstAI/angle-package` release via `scripts/fetch-angle.sh`. No credentials are
involved, so the full ANGLE-backed validation runs on pushes, same-repo PRs, and fork
PRs alike. (Fork PRs still need a maintainer to approve the run, per GitHub's default
first-time-contributor policy.)

### release.yml - Publishing

**Triggers:** Tags matching `vX.Y.Z` or `vX.Y.ZrcN`; also `workflow_dispatch` for a
dry run that builds artifacts without publishing.

| Job | Purpose |
|-----|---------|
| `build-wheels` | Build Python wheels (Linux, Windows, macOS) |
| `build-capi` | Build the C API shared library per target |
| `publish-pypi` | Publish wheels to PyPI |
| `publish-crates` | Publish the Rust crates to crates.io |
| `create-release` | Create the GitHub Release with changelog and artifacts |

The three publish/release jobs are gated on `github.event_name == 'push'` plus a
`refs/tags/` ref, so a manual dispatch builds but never publishes.

### sbom.yml - License Compliance

**Triggers:** Push/PR to `main`, `develop`, or `release/**`; releases

| Job | Purpose |
|-----|---------|
| `sbom-compliance` | Generate SBOM, validate licenses |
| `release-sbom` | Attach SBOM to GitHub releases |

### tag-release.yml - Release Tagging

**Triggers:** PR closed against `main`

Pushes the `vX.Y.Z` tag when a `release/X.Y.Z` branch merges, which is what starts
`release.yml`. Never tag by hand; the tag follows the merge, not the other way round.
The job derives the version from the branch name and rejects anything that is not
`release/` plus three dot-separated numbers.

### benchmark.yml - On-Demand Benchmarks

**Triggers:** `workflow_dispatch` only

| Job | Runner | Purpose |
|-----|--------|---------|
| `build-benchmarks` | ubuntu-22.04-arm | Cross-build the aarch64 benchmark binaries |
| `run-rust-benchmarks` | nxp-imx8mp-latest | Rust benchmarks on i.MX 8M Plus |
| `run-python-benchmarks` | nxp-imx8mp-latest | Python benchmarks on i.MX 8M Plus |
| `process-results` | ubuntu-22.04-arm | Generate result tables and charts |

Benchmarks are not part of CI. They tie up the hardware runner for a long time and the
numbers are only meaningful when collected deliberately — see
[BENCHMARKS.md](../../BENCHMARKS.md).

## Action Versions

All workflows use hash-pinned actions, per Au-Zone SPS. Read the pins from the workflow
files themselves — a copy here would go stale the first time Dependabot bumps one.

```bash
grep -rhoE 'uses: [^ ]+@[a-f0-9]{40}.*' .github/workflows/*.yml | sort -u
```

Each pin carries a trailing `# vX.Y.Z` comment naming the tag it corresponds to.

## Runners

| Runner | Architecture | Notes |
|--------|--------------|-------|
| `ubuntu-22.04` | x86_64 | Standard hosted runner |
| `ubuntu-22.04-xlarge` | x86_64 | 16 vCPU; used for the build-heavy lanes |
| `ubuntu-22.04-arm` | aarch64 | Test and post-processing |
| `ubuntu-22.04-arm-xlarge` | aarch64 | Cross-compilation |
| `macos-latest` | arm64 | Apple Silicon; ANGLE → Metal |
| `windows-latest` | x86_64 | ANGLE → Direct3D 11 (WARP software adapter; no GPU) |
| `nxp-imx8mp-latest` | aarch64 | Self-hosted board: G2D, DMA-heap, Vivante GL |

## Coverage Strategy

Six jobs upload coverage artifacts, and `sonarcloud` merges them:

| Artifact | Source | Covers |
|----------|--------|--------|
| `coverage-x86_64` | `build-and-test-x86` | Rust + Python + C API on x86_64 |
| `coverage-macos` | `build-and-test-macos` | ANGLE/IOSurface GL paths |
| `coverage-windows` | `build-and-test-windows` | ANGLE/Direct3D 11 leaf and PBO paths under WARP |
| `coverage-software-gl` | `software-gl-coverage` | GL paths under Mesa llvmpipe |
| `coverage-aarch64` | `test-arm` | Rust + Python on aarch64 |
| `coverage-imx8mp-processed` | `process-hardware-coverage` | DMA-heap, G2D, Vivante GL |

The imx8mp path needs the extra `process-hardware-coverage` step because the board
writes raw `.profraw` files; converting them to LCOV requires the same toolchain that
built the instrumented binaries, which the board does not have.

## Scripts

Supporting scripts in `.github/scripts/`:

| Script | Purpose |
|--------|---------|
| `generate_sbom.sh` | Generate CycloneDX SBOM |
| `check_license_policy.py` | Validate dependency licenses |
| `generate_notice.py` | Generate the NOTICE file |
| `verify_version.py` | Check version consistency across the workspace |
| `setup-coverage-env.sh` | Export the coverage instrumentation environment |
| `fix_coverage_paths.py` | Rewrite cross-compiled source paths in LCOV output |
| `coverage_summary.py` | Print a per-crate coverage summary |
| `generate_junit_xml.py` | Convert nextest output to JUnit XML for the PR check |
| `extract_benchmark_results.py` | Pull benchmark JSON out of the runner artifacts |
| `generate_benchmark_tables.py` | Render the result tables in BENCHMARKS.md |
| `benchmark_common.py` | Shared platform configuration for the benchmark scripts |
| `build-opencv-benchmark.sh` | Build the OpenCV baseline benchmark |
| `deploy-bench-masks.sh` | Deploy mask benchmark fixtures to the board |
| `audit-injection.sh` | Scan workflows for script-injection patterns |

## Local Development

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for local testing instructions and
[TESTING.md](../../TESTING.md) for the workspace testing rules.
