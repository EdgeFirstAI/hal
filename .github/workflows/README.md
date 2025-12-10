# GitHub Actions Workflows

This directory contains the CI/CD workflows for EdgeFirst HAL.

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Events                            │
└─────────────────────────────────────────────────────────────────┘
         │                    │                    │
         │ Push/PR            │ Tag (X.Y.Z)        │ Release
         ▼                    ▼                    ▼
┌────────────────┐   ┌────────────────┐   ┌────────────────┐
│   test.yml     │   │  release.yml   │   │   sbom.yml     │
│  (CI/Testing)  │   │  (Publishing)  │   │  (Compliance)  │
└────────────────┘   └────────────────┘   └────────────────┘
```

## Workflows

### test.yml - Continuous Integration

**Triggers:** Push/PR to `main` or `develop`

Multi-platform testing with coverage collection:

| Job | Runner | Purpose |
|-----|--------|---------|
| `checkout-lfs` | ubuntu-22.04 | Fetch Git LFS testdata as artifact |
| `doc-tests` | ubuntu-22.04 | Rust documentation tests |
| `build-and-test-x86` | ubuntu-22.04 | x86_64 build, test, coverage |
| `build-and-test-arm` | ubuntu-22.04-arm-private | aarch64 build, test, coverage |
| `hardware-test` | nxp-imx8mp-latest | On-target testing (G2D, DMA) |
| `process-hardware-coverage` | ubuntu-22.04-arm-private | Convert profraw to LCOV |
| `sonarcloud` | ubuntu-22.04 | Aggregate coverage, static analysis |

**Key Features:**
- Three-phase on-target testing (build → test → process)
- Coverage instrumentation via cargo-llvm-cov
- Python coverage via slipcover
- Hardware benchmarks on main branch

### release.yml - Publishing

**Triggers:** Tags matching `X.Y.Z` or `X.Y.ZrcN`

| Job | Purpose |
|-----|---------|
| `build-wheels` | Build Python wheels (Linux, Windows, macOS) |
| `publish-pypi` | Publish to PyPI (stable releases only) |
| `create-release` | Create GitHub Release with changelog |

### sbom.yml - License Compliance

**Triggers:** Push/PR to `main` or `develop`, releases

| Job | Purpose |
|-----|---------|
| `sbom-compliance` | Generate SBOM, validate licenses |
| `release-sbom` | Attach SBOM to GitHub releases |

## Action Versions

All workflows use hash-pinned actions for security (per Au-Zone SPS v2.1):

```yaml
actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5         # v4
actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065     # v5
actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02  # v4
actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093 # v4
dtolnay/rust-toolchain@6d9817901c499d6b02debbb57edb38d33daa680b   # stable
Swatinem/rust-cache@779680da715d629ac1d338a641029a2f4372abb5     # v2.8.2
taiki-e/install-action@493d7f216ecab2af0602481ce809ab2c72836fa1  # v2.62.62
softprops/action-gh-release@5be0e66d93ac7ed76da52eca8bb058f665c3a5fe # v2.4.2
```

## Runners

| Runner | Architecture | Capabilities |
|--------|--------------|--------------|
| `ubuntu-22.04` | x86_64 | Full toolchain, Docker |
| `ubuntu-22.04-arm-private` | aarch64 | Full toolchain, private |
| `nxp-imx8mp-latest` | aarch64 | Hardware (G2D, DMA), test-only |

## Coverage Strategy

Coverage is collected from three platforms and aggregated:

1. **x86_64**: Full Rust + Python coverage
2. **aarch64**: Full Rust + Python coverage  
3. **imx8mp**: Hardware-specific paths (DMA, G2D acceleration)

The `process-hardware-coverage` job converts raw profiling data from hardware tests
to LCOV format using the same toolchain that built the instrumented binaries.

## Scripts

Supporting scripts in `.github/scripts/`:

| Script | Purpose |
|--------|---------|
| `generate_sbom.sh` | Generate CycloneDX SBOM |
| `check_license_policy.py` | Validate dependency licenses |
| `generate_notice.py` | Generate NOTICE file |

## Local Development

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for local testing instructions.

