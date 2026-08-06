# Contributing to EdgeFirst HAL

Thanks for your interest in contributing. The EdgeFirst Hardware Abstraction Layer (HAL) is part of the EdgeFirst Perception stack. It gives edge AI and computer vision pipelines a hardware-accelerated image, tensor, and post-processing layer that runs the same way on embedded Linux, macOS, Android, and iOS.

## Code of Conduct

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

## Ways to Contribute

- **Code**: Features, bug fixes, performance improvements
- **Documentation**: Improvements, examples, tutorials
- **Testing**: Bug reports, test coverage, hardware platform validation
- **Community**: Answer questions, write blog posts, speak at meetups

## Before You Start

1. Check existing [issues](https://github.com/EdgeFirstAI/hal/issues) and [pull requests](https://github.com/EdgeFirstAI/hal/pulls)
2. For significant changes, open an issue for discussion first
3. Review our [roadmap](https://github.com/EdgeFirstAI/hal/issues) to understand project direction

## Development Setup

### Prerequisites

**System Requirements:**
- Rust stable. The workspace declares no MSRV; CI pins `1.94.0` (see `RUST_STABLE_VERSION` in `.github/workflows/test.yml`), so build against that or newer.
- Python 3.8 or later (for Python bindings)
- Linux, macOS, Android, or iOS for the full feature set. Windows is compile-check only — CI runs `cargo check` there and nothing more.
- Optional: NXP i.MX platform for G2D hardware acceleration testing

**Development Tools:**
- `cargo` - Rust package manager
- `rustfmt` - Code formatter (installed with Rust)
- `clippy` - Linting tool (installed with Rust)
- `cargo-nextest` and `cargo-llvm-cov` - Test runner and coverage; the `make test-*` targets require both
- `maturin` - For building Python bindings
- `pytest` - For Python tests

#### Platform-specific notes (macOS / iOS / ANGLE)

The HAL's CPU path builds and tests on Linux, macOS, and Windows with no
extra setup. The **OpenGL (GPU) backend on Apple platforms needs ANGLE**
(Google's GLES→Metal translator), which is *not* part of macOS/iOS.

- ANGLE is an open-source Google project, and the EdgeFirst **pre-built,
  signed + notarized xcframeworks are published from the public
  repository** ([`EdgeFirstAI/angle-package`](https://github.com/EdgeFirstAI/angle-package)).
  Anyone can fetch them with `scripts/fetch-angle.sh` — no credentials or
  organization membership required. See
  [README.md § macOS GPU Acceleration](README.md#macos-gpu-acceleration).
- Prefer a package manager on **macOS**? Install ANGLE via the public
  Homebrew tap instead (`brew install startergo/angle/angle`, then re-sign
  the dylibs — see README). On **iOS** there is no Homebrew equivalent, so
  use `scripts/fetch-angle.sh` (or build the CPU-only path:
  `cargo build --target aarch64-apple-ios --no-default-features --features ndarray,tracing`).
- Either way, you can contribute to the Linux/CPU paths without any ANGLE
  at all — just disable the `opengl` feature
  (`--no-default-features --features ndarray,tracing`).

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/EdgeFirstAI/hal.git
cd hal

# Build all Rust crates
cargo build --workspace

# Run tests — single-threaded, always (see below)
cargo test --workspace -- --test-threads=1

# Build Python bindings (optional)
pip install maturin
maturin develop -m crates/python/Cargo.toml

# Run Python tests (requires Python bindings)
python -m pytest tests/
```

**Tests must run single-threaded.** GPU driver concurrency bugs, G2D per-process
state, and CMA pool exhaustion each independently require it, so pass
`--test-threads=1` to `cargo test` and `-j 1` to `cargo nextest`. The Makefile
targets do this for you. [TESTING.md § Single-Threaded
Execution](TESTING.md#single-threaded-execution) has the full reasoning.

### Hardware Platform Testing

For testing on NXP i.MX platforms with G2D acceleration:
- Install the vendor G2D libraries. G2D is not a build feature — the HAL `dlopen`s
  `libg2d.so.2` at runtime and skips the backend when it is absent, so the same binary
  works on boards with and without it.
- Test both the accelerated and the fallback CPU paths. `EDGEFIRST_DISABLE_G2D=1`
  forces the fallback; `EDGEFIRST_FORCE_BACKEND=g2d` pins the accelerated path so a
  silent fallback fails the test instead of passing.

## Contribution Process

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/hal.git
cd hal
git remote add upstream https://github.com/EdgeFirstAI/hal.git
```

### 2. Create Feature Branch

Prefix the branch with its kind, then a short slug. Ticketed work carries the JIRA
key. Prefixes in active use:

- `feat/` or `feature/` — `feature/EDGEAI-1018-sahi-tiling`, `feat/crop-contract-gl-planar`
- `bugfix/` — `bugfix/EDGEAI-1353-egl-loader-ub-ios-resolution`
- `ci/`, `bench/` — infrastructure and measurement work
- `release/X.Y.Z` — release branches; merging one to `main` is what pushes the version tag

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Follow the code style guidelines (see below)
- Add tests for new functionality
- Update documentation in README.md and inline docs
- Ensure all tests pass locally

### 4. Test Your Changes

The Makefile is the local gate and matches what CI runs:

```bash
make format lint check   # required before every commit
make test                # Rust + Python + C API, with coverage
```

Or run the pieces directly:

```bash
# Rust tests — single-threaded
cargo test --workspace -- --test-threads=1

# Linting, exactly as CI runs it (plain `cargo clippy` will let CI failures through)
cargo clippy --workspace --all-targets --features default,opencv -- -D warnings

# Format code
cargo fmt --all

# Build and test Python bindings
maturin develop -m crates/python/Cargo.toml
python -m pytest tests/
```

### 5. Commit and Push

```bash
git add .
git commit -s -m "EDGEAI-123: Add tensor operation for matrix multiplication"
git push origin feature/your-feature-name
```

All commits must be DCO-signed (`git commit -s`).

**Commit Message Convention:**

For feature/bug work with a JIRA ticket, lead with the key:

```
EDGEAI-123: Add tensor operation for matrix multiplication
```

Otherwise use Conventional Commits, scoped to the crate where that helps:

```
feat(image): route all planar heap destinations through the two-pass GL plan
fix(ci): stop interpolating the PR branch name into tag-release.yml
test(image): per-architecture golden fixtures for cropped-convert
docs: reconcile the three batch memory representations
```

Release commits are their own shape: `Release v0.28.0`.

### 6. Submit Pull Request

1. Go to the [hal repository](https://github.com/EdgeFirstAI/hal)
2. Click "New Pull Request"
3. Select your fork and branch
4. In the description, cover:
   - What changed and why
   - Related issue or JIRA ticket numbers
   - Testing performed, including which hardware you ran on
   - A CHANGELOG.md entry under `[Unreleased]`
5. Wait for CI checks to pass
6. Address review feedback

## Code Style

### Rust Code

We follow standard Rust conventions:

- Use `rustfmt` for formatting (configuration in `rustfmt.toml`)
- Use `clippy` for linting
- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Document all public APIs with doc comments (`///`)
- Use descriptive variable and function names
- Prefer composition over inheritance
- Use `Result<T, E>` for error handling

**Before committing:**
```bash
cargo fmt --all
cargo clippy --workspace
```

### Python Code

For Python bindings:

- Follow PEP 8 style guide
- Use type hints where possible
- Document functions with docstrings
- Match Python naming conventions (snake_case)
- Use PyO3 best practices

### Documentation

- All public Rust APIs must have doc comments
- Include usage examples in doc comments
- Update README.md for user-facing changes
- Add inline comments for complex logic
- Keep documentation up-to-date with code changes

## Testing Requirements

### Test Coverage

- **Minimum coverage**: 70% for new code
- Unit tests for all new functions
- Integration tests for cross-crate functionality
- Python binding tests for exposed APIs

### Writing Tests

**Rust Tests:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() {
        // Arrange
        let input = setup_test_data();
        
        // Act
        let result = function_under_test(input);
        
        // Assert
        assert_eq!(result, expected_value);
    }
}
```

**Python Tests:** the suite in `tests/` is pytest, not `unittest`:

```python
import pytest
from edgefirst_hal import Tensor, TensorMemory, PixelFormat


@pytest.mark.parametrize("mem", [TensorMemory.MEM, TensorMemory.SHM])
def test_image_tensor_honours_requested_memory(mem):
    tensor = Tensor.image(640, 480, PixelFormat.Rgb, mem, access="readwrite")
    assert tensor.memory == mem
```

### Running Tests

```bash
# All Rust tests
cargo test --workspace -- --test-threads=1

# Specific crate
cargo test -p edgefirst-image -- --test-threads=1

# Specific test
cargo test test_name -- --test-threads=1

# Python tests (requires maturin develop first)
python -m pytest tests/ -v

# With coverage — what `make test-rust` runs
cargo install cargo-llvm-cov --locked
cargo llvm-cov nextest --workspace --exclude edgefirst_hal \
  --lcov --output-path target/rust-coverage.lcov -j 1

# With coverage (Python, after building with instrumentation)
pip install slipcover
python -m slipcover -m pytest tests/
```

The Python crate is excluded from workspace Rust runs because it needs a live
Python interpreter. [TESTING.md](TESTING.md) covers cross-compilation, on-target
runs, and the hardware gating rules.

## CI/CD Workflows

The project uses GitHub Actions for continuous integration. Workflows are in `.github/workflows/`.

### Test Workflow (`test.yml`)

Runs on every push and PR to `main`, `develop`, or `release/**`:

- **Formatting check**: `cargo fmt --all -- --check` (advisory)
- **Linting**: `cargo clippy --workspace --all-targets ... -- -D warnings`
- **Multi-platform testing**: x86_64, aarch64, NXP i.MX8M Plus hardware
- **Software GL**: the image crate's GL tests under Mesa llvmpipe, covering GL paths
  no hardware runner reaches
- **macOS**: Rust tests (incl. the ANGLE/IOSurface GL render path) +
  C API. Fetches the signed ANGLE xcframeworks from the public
  `angle-package` release via `scripts/fetch-angle.sh`.
- **iOS**: build + link validation for `aarch64-apple-ios` (device) and
  `aarch64-apple-ios-sim` (no runtime tests yet). Also fetches ANGLE from
  the public release.
- **Android**: clippy + build + link validation for `aarch64-linux-android`
  (device) and `x86_64-linux-android` (emulator) via `cargo-ndk` with a
  pinned NDK (r27c) at the API-26 floor — no runtime tests in CI (GitHub
  runners have no Android GPU); on-device correctness/performance is gated
  separately by the internal hal-mobile Device Farm harness.
  Local prerequisites:
  `rustup target add aarch64-linux-android x86_64-linux-android`,
  `cargo install cargo-ndk`, and an NDK (r26+) via `ANDROID_NDK_HOME`.
- **Windows**: compile check (`cargo check`).
- **Coverage collection**: Rust (cargo-llvm-cov) + Python (slipcover)
- **SonarCloud analysis**: Static analysis and coverage aggregation

> The macOS and iOS lanes fetch the ANGLE xcframeworks from the **public**
> `angle-package` release (`scripts/fetch-angle.sh`) — no credentials
> needed, so the download works for pushes, same-repo PRs, and fork PRs
> alike. The full ANGLE-backed validation (the macOS ANGLE/IOSurface GL
> render tests and the iOS link closure) therefore runs on every event,
> forks included — no special-casing. (Fork PRs still require a maintainer
> to approve the workflow run, per GitHub's default first-time-contributor
> policy.)

### Release Workflow (`release.yml`)

Triggered by version tags (`vX.Y.Z` or `vX.Y.ZrcN`):

- Builds Python wheels for Linux, Windows, and macOS
- Builds the C API shared library per target
- Publishes to PyPI and crates.io
- Creates GitHub Release with changelog

### Tag Release Workflow (`tag-release.yml`)

Runs when a `release/X.Y.Z` PR merges into `main`, and pushes the `vX.Y.Z` tag that
starts `release.yml`. Never tag by hand — the tag follows the merge.

### SBOM Workflow (`sbom.yml`)

Runs on push/PR to `main`, `develop`, or `release/**`, and on releases:

- Generates Software Bill of Materials (CycloneDX format)
- Validates license compliance
- Attaches SBOM to releases

### Benchmark Workflow (`benchmark.yml`)

Manual dispatch only. Builds the aarch64 benchmark binaries, runs the Rust and Python
suites on the i.MX 8M Plus runner, and regenerates the result tables. Benchmarks are
never part of CI — they hold the hardware runner for a long time.

## Benchmarking

For performance-critical changes:

```bash
# All workspace benchmarks
make bench

# One crate
cargo bench -p edgefirst-image
cargo bench -p edgefirst-decoder
cargo bench -p edgefirst-tensor

# One binary
cargo bench -p edgefirst-image --bench pipeline_benchmark
```

These use the workspace's own `edgefirst-bench` harness (`harness = false`), not
Criterion. [BENCHMARKS.md](BENCHMARKS.md) lists every binary and the per-platform
baselines to compare against.

## Documentation Guidelines

### Inline Documentation

Doc examples are compiled and run by the `doc-tests` CI job, so they have to work.
Use `# ` to hide setup lines and `no_run` when the example needs a file or a GPU that
CI does not have. This one is lifted from `ImageProcessor::new` in
`crates/image/src/lib.rs`:

````rust
/// Creates a new `ImageProcessor` instance, initializing available
/// hardware converters based on the system capabilities and environment
/// variables.
///
/// # Examples
/// ```rust,no_run
/// # use edgefirst_image::{ImageProcessor, Rotation, Flip, Crop, ImageProcessorTrait};
/// # use edgefirst_codec::{peek_info, ImageDecoder, ImageLoad};
/// # use edgefirst_tensor::{CpuAccess, PixelFormat, DType, Tensor, TensorMemory};
/// # fn main() -> Result<(), edgefirst_image::Error> {
/// let image = std::fs::read("zidane.jpg")?;
/// // The codec emits the source's native format (a colour JPEG decodes to
/// // NV12) and configures the destination tensor during the decode.
/// let info = peek_info(&image).expect("peek");
/// let mut src = Tensor::<u8>::image(info.width, info.height, info.format,
///                                    Some(TensorMemory::Mem), CpuAccess::ReadWrite)?;
/// let mut decoder = ImageDecoder::new();
/// src.load_image(&mut decoder, &image).expect("decode");
/// let mut converter = ImageProcessor::new()?;
/// let mut dst =
///     converter.create_image(640, 480, PixelFormat::Rgb, DType::U8, None, CpuAccess::ReadWrite)?;
/// converter.convert(&src.into(), &mut dst, Rotation::None, Flip::None, Crop::default())?;
/// # Ok(())
/// # }
/// ```
pub fn new() -> Result<Self> {
````

For functions whose failure modes matter to the caller, add `# Arguments` and
`# Errors` sections naming the concrete error variants.

### README Updates

When adding new features:
- Update the Features section
- Add usage examples
- Update architecture diagrams if needed
- Add to the appropriate crate's documentation

## Hardware Platform Considerations

When contributing hardware-specific code:

- Test on actual hardware when possible
- Provide fallback implementations for non-accelerated platforms
- Document hardware requirements clearly
- Use feature flags for platform-specific code
- Consider power efficiency and performance trade-offs

## Pull Request Review Process

### What to Expect

1. **Automated Checks**: CI will run tests, linting, and formatting checks
2. **Initial Review**: A maintainer will review within 5 business days
3. **Feedback**: Address comments and suggestions
4. **Approval**: Once approved, a maintainer will merge

### Review Criteria

- Code quality and style compliance
- Test coverage (minimum 70%)
- Documentation completeness
- No breaking changes (unless discussed)
- Performance considerations
- Security implications

## Getting Help

- **Questions**: Use [GitHub Discussions](https://github.com/EdgeFirstAI/hal/discussions)
- **Bug Reports**: Open an [issue](https://github.com/EdgeFirstAI/hal/issues)
- **Security**: Follow [SECURITY.md](SECURITY.md) — do not open a public issue

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0. No additional contributor agreement is required.

All contributed code must be:
- Your original work or properly attributed
- Compatible with Apache-2.0 license
- Free of proprietary dependencies (unless optional)

## Recognition

Contributors are recognized in the release notes, the CHANGELOG, and the GitHub
contributor graph.

Thanks for contributing to EdgeFirst HAL.
