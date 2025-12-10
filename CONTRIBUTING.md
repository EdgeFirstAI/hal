# Contributing to EdgeFirst HAL

Thank you for your interest in contributing! The EdgeFirst Hardware Abstraction Layer (HAL) is part of the EdgeFirst Perception stack, advancing edge AI and computer vision capabilities with hardware-accelerated abstractions for embedded Linux platforms.

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
- Rust 1.70 or later
- Python 3.8 or later (for Python bindings)
- Linux (recommended for hardware acceleration), macOS, or Windows
- Optional: NXP i.MX platform for G2D hardware acceleration testing

**Development Tools:**
- `cargo` - Rust package manager
- `rustfmt` - Code formatter (installed with Rust)
- `clippy` - Linting tool (installed with Rust)
- `maturin` - For building Python bindings
- `pytest` - For Python tests

### Clone and Build

```bash
# Clone the repository
git clone https://github.com/EdgeFirstAI/hal.git
cd hal

# Build all Rust crates
cargo build --workspace

# Run tests
cargo test --workspace

# Build Python bindings (optional)
pip install maturin
maturin develop -m crates/python/Cargo.toml

# Run Python tests (requires Python bindings)
python -m pytest tests/
```

### Hardware Platform Testing

For testing on NXP i.MX platforms with G2D acceleration:
- Ensure you have G2D libraries installed
- The build will automatically enable G2D support if detected
- Test both accelerated and fallback CPU paths

## Contribution Process

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/hal.git
cd hal
git remote add upstream https://github.com/EdgeFirstAI/hal.git
```

### 2. Create Feature Branch

Use descriptive naming conventions:
- `feature/add-tensor-operation`
- `bugfix/issue-123-memory-leak`
- `docs/improve-decoder-examples`

```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

- Follow the code style guidelines (see below)
- Add tests for new functionality
- Update documentation in README.md and inline docs
- Ensure all tests pass locally

### 4. Test Your Changes

```bash
# Run Rust tests
cargo test --workspace

# Run Rust linting
cargo clippy --workspace

# Format code (requires Rust nightly)
cargo +nightly fmt --all

# Build and test Python bindings
maturin develop -m crates/python/Cargo.toml
python -m pytest tests/
```

### 5. Commit and Push

```bash
git add .
git commit -m "feat: add new tensor operation for matrix multiplication"
git push origin feature/your-feature-name
```

**Commit Message Convention:**
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `test:` - Adding or updating tests
- `refactor:` - Code refactoring
- `perf:` - Performance improvements
- `chore:` - Maintenance tasks

### 6. Submit Pull Request

1. Go to the [hal repository](https://github.com/EdgeFirstAI/hal)
2. Click "New Pull Request"
3. Select your fork and branch
4. Fill out the PR template with:
   - Description of changes
   - Related issue numbers (if any)
   - Testing performed
   - Screenshots (if UI changes)
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
cargo +nightly fmt --all
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

**Python Tests:**
```python
import unittest
import edgefirst_hal as ef

class TestFeature(unittest.TestCase):
    def test_basic_operation(self):
        # Test implementation
        result = ef.some_function()
        self.assertEqual(result, expected_value)
```

### Running Tests

```bash
# All Rust tests
cargo test --workspace

# Specific crate
cargo test -p edgefirst_image

# Specific test
cargo test test_name

# Python tests (requires maturin develop first)
python -m pytest tests/ -v

# With coverage (Rust)
cargo install cargo-llvm-cov
cargo llvm-cov --workspace --lcov --output-path lcov.info

# With coverage (Python, after building with instrumentation)
pip install slipcover
python -m slipcover -m pytest tests/
```

## CI/CD Workflows

The project uses GitHub Actions for continuous integration. Workflows are in `.github/workflows/`.

### Test Workflow (`test.yml`)

Runs on every push and PR to `main` or `develop`:

- **Formatting check**: `cargo +nightly fmt --all -- --check`
- **Linting**: `cargo clippy --workspace`
- **Multi-platform testing**: x86_64, aarch64, NXP i.MX8M Plus hardware
- **Coverage collection**: Rust (cargo-llvm-cov) + Python (slipcover)
- **SonarCloud analysis**: Static analysis and coverage aggregation

### Release Workflow (`release.yml`)

Triggered by version tags (`X.Y.Z` or `X.Y.ZrcN`):

- Builds Python wheels for Linux, Windows, and macOS
- Publishes to PyPI (stable releases only)
- Creates GitHub Release with changelog

### SBOM Workflow (`sbom.yml`)

Runs on push/PR and releases:

- Generates Software Bill of Materials (CycloneDX format)
- Validates license compliance
- Attaches SBOM to releases

## Benchmarking

For performance-critical changes:

```bash
# Run benchmarks
cargo bench -p edgefirst_image
cargo bench -p edgefirst_decoder
cargo bench -p edgefirst_tensor

# Compare before/after performance
```

## Documentation Guidelines

### Inline Documentation

```rust
/// Converts an image from one format to another.
///
/// # Arguments
///
/// * `src` - Source tensor image
/// * `dst` - Destination tensor image (must be pre-allocated)
/// * `options` - Conversion options (resize, rotate, etc.)
///
/// # Examples
///
/// ```rust
/// use edgefirst::image::{TensorImage, ImageProcessor};
///
/// let src = TensorImage::load("input.jpg", None, None)?;
/// let mut dst = TensorImage::new(640, 640, RGB, None)?;
/// let converter = ImageProcessor::new()?;
/// converter.convert(&src, &mut dst, Default::default())?;
/// ```
///
/// # Errors
///
/// Returns `ImageError` if conversion fails or formats are incompatible.
pub fn convert(&self, src: &TensorImage, dst: &mut TensorImage, options: ConvertOptions) -> Result<()> {
    // Implementation
}
```

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
- **Real-time Chat**: Join our community channel (link in README)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0. No additional contributor agreement is required.

All contributed code must be:
- Your original work or properly attributed
- Compatible with Apache-2.0 license
- Free of proprietary dependencies (unless optional)

## Recognition

Contributors are recognized in:
- Release notes
- CONTRIBUTORS.md file
- GitHub contributor graph
- Annual project acknowledgments

Thank you for contributing to EdgeFirst HAL! 🚀
