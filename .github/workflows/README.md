# GitHub Actions Workflow Architecture

This document provides a comprehensive overview of the GitHub Actions workflows for the EdgeFirst HAL project.

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        GitHub Events                            │
└─────────────────────────────────────────────────────────────────┘
         │                                  │
         │ Push/PR                          │ Tag (X.Y.Z)
         ▼                                  ▼
┌──────────────────────────────┐  ┌──────────────────────────┐
│      Test Workflow           │  │   Release Workflow       │
│  (test.yml with coverage)    │  │     (release.yml)        │
│                              │  │                          │
└──────────────────────────────┘  └──────────────────────────┘
```

## Workflow Files

### 1. Test Workflow (`.github/workflows/test.yml`)

**Purpose:** Code quality checks, testing, and coverage for both Rust and Python

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Manual workflow dispatch

**Jobs:**

```
test.yml
├── test (primary - Python 3.11)
│   ├── Checkout with Git LFS (for testdata)
│   ├── Check formatting (cargo fmt)
│   ├── Run clippy linter
│   ├── Install cargo-llvm-cov
│   ├── Build Rust with coverage instrumentation
│   ├── Build Python bindings (inherits coverage)
│   ├── Run Python tests with slipcover coverage
│   ├── Generate coverage reports (lcov.info, coverage.xml)
│   ├── Display coverage summary
│   └── Upload coverage artifacts
└── test-python-versions (matrix: Python 3.8, 3.12)
    ├── Checkout with Git LFS (for testdata)
    ├── Build Python wheel (maturin)
    ├── Install wheel
    └── Run Python tests (unittest)
```

**Key Features:**
- **Unified test and coverage**: All tests run with coverage enabled (no separate workflow)
- **Git LFS integration**: Automatically fetches binary testdata assets
- **Clippy linting**: Code quality checks with clippy
- **Multi-version Python testing**: Primary tests on 3.11 with coverage, compatibility tests on 3.8 and 3.12
- **Locked dependencies**: Ensures reproducible builds with `--locked` flag
- **Runtime library detection**: Tests gracefully skip when G2D/DMA unavailable
- **Swatinem/rust-cache**: Intelligent caching with incremental compilation state preservation

**Coverage Tools:**
- **Rust:** cargo-llvm-cov (LLVM-based coverage)
- **Python:** slipcover (zero-overhead coverage)

**Test Behavior:**
Tests that require hardware-specific libraries (G2D) or kernel features (DMA-BUF) will automatically skip with informative messages when these dependencies are unavailable:
- `SKIPPED: test_g2d_* - G2D library (libg2d.so.2) not available`
- `SKIPPED: test_*_dma - DMA memory allocation not available (permission denied or no DMA-BUF support)`

This allows tests to pass in CI/CD environments without the specific hardware while still running successfully on target platforms.

**Artifacts Generated:**
- `coverage-reports/lcov.info` - Rust coverage in LCOV format
- `coverage-reports/coverage.xml` - Python coverage in Cobertura XML format

**Note:** This project does not currently integrate with Codecov or SonarCloud per project requirements.

### 2. Release Workflow (`.github/workflows/release.yml`)

**Purpose:** Complete release automation with PyPI publishing

**Triggers:**
- Tags matching semantic versioning:
  - Stable releases: `[0-9]+.[0-9]+.[0-9]+`
    - Examples: `0.1.0`, `1.0.0`, `2.1.3`
  - Release candidates: `[0-9]+.[0-9]+.[0-9]+rc[0-9]+`
    - Examples: `0.1.0rc1`, `1.0.0rc2`

**Jobs:**

```
release.yml
├── create-release
│   ├── Extract version from tag
│   ├── Extract CHANGELOG notes
│   └── Create GitHub release (prerelease if rc)
└── build-and-publish (matrix: Linux, Windows, macOS)
    ├── Build Python wheels (maturin)
    ├── Upload wheels to GitHub release
    └── Publish to PyPI (stable releases only)
```

**Matrix Strategy:**

| Runner | Target | Wheel Artifact |
|--------|--------|---------------|
| ubuntu-latest | x86_64-unknown-linux-gnu | wheels-linux-x86_64 |
| windows-latest | x86_64-pc-windows-msvc | wheels-windows-x86_64 |
| macos-latest | x86_64-apple-darwin | wheels-macos-x86_64 |

**Key Features:**
- Multi-platform wheel building (Linux, Windows, macOS)
- PyPI Trusted Publisher authentication (OIDC, no API token required)
- Automatic CHANGELOG extraction for release notes
- Pre-release detection (rc versions)
- Conditional PyPI publishing (only for stable releases)

**Important:** The version in `Cargo.toml` must match the git tag. There is no automated verification in this workflow, so ensure versions are synchronized before tagging.

**Note:** This project publishes Python wheels only (no CLI binaries, no crates.io publication).

## Caching Strategy

All workflows use Swatinem/rust-cache for intelligent Rust build caching:

```
cache:
├── cargo registry   (~/.cargo/registry)
├── cargo index      (~/.cargo/git)
├── build artifacts  (target/)
└── incremental compilation state
```

**Cache Key Strategy:**
- Test workflow: `test` (Swatinem/rust-cache handles automatic keying)
- Release workflow: `release-{target}` (per-target architecture)

**Benefits:**
- 10x faster builds on cache hit
- Reduced CI costs
- Better reliability with fewer network dependencies

## Version Format

The project uses version formats compatible with both Python (PEP 440) and Rust (Cargo/SemVer):

| Type | Format | Examples | Rust | Python |
|------|--------|----------|------|--------|
| Stable | X.Y.Z | 0.1.0, 1.0.0 | ✅ | ✅ |
| Release Candidate | X.Y.ZrcN | 0.1.0rc1, 1.0.0rc2 | ✅ | ✅ |

**Important:** Do NOT use separators (dots or hyphens) in pre-release versions. Use `1.0.0rc1`, not `1.0.0-rc.1` or `1.0.0.rc.1`.

## Testing Strategy

### Continuous Testing (CI)

```
Every Push/PR
     │
     ├──► cargo fmt check
     ├──► cargo clippy
     ├──► cargo llvm-cov (build with coverage)
     ├──► maturin develop (Python with coverage)
     ├──► Python unittest with slipcover
     ├──► Generate coverage reports
     │       ├──► lcov.info (Rust)
     │       └──► coverage.xml (Python)
     └──► Python 3.8/3.12 compatibility tests
```

### Coverage Reporting

Coverage is integrated into every test run (not a separate workflow):

```
Test Workflow
     │
     ├──► Rust Coverage (cargo llvm-cov)
     │       └──► lcov.info
     │
     └──► Python Coverage (slipcover)
             └──► coverage.xml
                  │
                  └──► Upload as artifacts
```

### Pre-Release Testing

Before creating a release tag, manually test:
1. Trigger test workflow (tests run with coverage automatically)
2. Download and verify coverage artifacts
3. Build wheels locally to test

## Release Process

### For Maintainers

1. **Update version in Cargo.toml**

   ```toml
   [workspace.package]
   version = "0.1.0"  # Update this
   ```

2. **Update CHANGELOG.md**

   ```markdown
   ## [0.1.0] - 2025-11-10
   
   ### Added
   - Feature description
   
   ### Fixed
   - Bug fix description
   ```

3. **Commit changes**

   ```bash
   git add Cargo.toml CHANGELOG.md
   git commit -m "Release 0.1.0 preparations"
   git push
   ```

4. **Create and push tag**

   ```bash
   git tag 0.1.0
   git push origin 0.1.0
   ```

5. **Monitor workflow**
   - Go to Actions tab
   - Watch "Release" workflow
   - Verify all jobs complete successfully

6. **Verify release**
   - Check GitHub release page
   - Verify PyPI publication
   - Test `pip install edgefirst-hal`

### For Release Candidates

```bash
# Tag with rc suffix
git tag 0.1.0rc1
git push origin 0.1.0rc1
```

Release candidates will:
- Create a GitHub pre-release
- Upload wheels to GitHub release
- **NOT** publish to PyPI (manual only if needed)

## Environment Variables

### Common Environment Variables

```yaml
CARGO_TERM_COLOR: always    # Colored cargo output
RUST_BACKTRACE: 1           # Full backtraces on error
```

**Note:** Coverage environment variables (RUSTFLAGS, LLVM_PROFILE_FILE) are set automatically by cargo-llvm-cov.

## Secrets Management

### Required Secrets

| Secret | Purpose | Used In |
|--------|---------|---------|
| None | PyPI uses Trusted Publisher (OIDC) | Release workflow |

### PyPI Trusted Publisher Setup

This project uses PyPI's Trusted Publisher feature (OIDC) which does not require API tokens:

1. Go to PyPI project settings
2. Add GitHub Actions as a Trusted Publisher
3. Configure:
   - Owner: `EdgeFirstAI`
   - Repository: `hal`
   - Workflow: `release.yml`
   - Environment: `pypi`

The workflow uses the `pypi` environment with `id-token: write` permission for secure, token-less authentication.

## Monitoring and Debugging

### Viewing Workflow Runs

1. Go to Actions tab in repository
2. Select workflow from left sidebar
3. Click on specific run to see details

### Debugging Failed Jobs

1. Click on failed job
2. Expand failed step
3. Review logs
4. Check for:
   - Compilation errors
   - Test failures
   - Missing testdata (Git LFS)
   - Network issues

### Re-running Workflows

- Click "Re-run jobs" button
- Select "Re-run failed jobs" or "Re-run all jobs"

## Best Practices

### For Contributors

1. **Always run locally first:**
   ```bash
   cargo fmt --all
   cargo clippy --all-targets --all-features
   cargo test --all-features
   ```

2. **Test Python bindings:**
   ```bash
   pip install maturin
   maturin develop -m crates/python/Cargo.toml
   python -m unittest discover -s tests -p "test*.py"
   ```

3. **Keep PRs focused:**
   - One feature/fix per PR
   - Include tests
   - Update documentation

### For Maintainers

1. **Review workflow runs:**
   - Check all jobs pass
   - Review coverage reports
   - Verify testdata fetched correctly

2. **Test before releasing:**
   - Manual workflow dispatch
   - Verify artifacts
   - Test installations

3. **Monitor releases:**
   - Watch workflow completion
   - Verify PyPI publication
   - Test downloads

## Local Testing

Run tests locally before pushing:

```bash
# Install git-lfs
git lfs install
git lfs pull

# Test Rust
cargo fmt --check
cargo clippy --all-targets --all-features
cargo test --all-features

# Test Python
pip install maturin
maturin develop -m crates/python/Cargo.toml
python -m unittest discover -s tests -p "test*.py"

# Coverage (optional)
cargo install cargo-llvm-cov
pip install slipcover unittest-xml-reporting

# Rust coverage
cargo llvm-cov --all-features --lcov --output-path lcov.info
cargo llvm-cov report

# Python coverage
source <(cargo llvm-cov show-env --export-prefix)
cargo build --all-features
maturin develop -m crates/python/Cargo.toml
python -m slipcover --xml --out coverage.xml -m xmlrunner discover -s tests -p "test*.py"
```

## Skipped Integrations

Per project requirements, the following are NOT implemented:
- Codecov integration
- SonarCloud quality analysis
- CLI binary builds/packaging
- crates.io publication
- Separate coverage workflow (coverage integrated into test workflow)

## Support

For questions about workflows:
- Review this document
- Check workflow YAML files for implementation details
- Open an issue if problems persist

