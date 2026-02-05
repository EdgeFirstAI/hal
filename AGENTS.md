# AGENTS.md - AI Assistant Development Guidelines

**Purpose:** Instructions for AI coding assistants (GitHub Copilot, Cursor, Claude Code, etc.) working on this project.

**Version:** 1.0
**Last Updated:** November 2025

---

## Git Workflow

### Branch Naming

**Required Format:** `<type>/<PROJECTKEY-###>[-optional-description]`

```bash
feature/EDGEAI-123-add-authentication
bugfix/STUDIO-456-fix-memory-leak
hotfix/MAIVIN-789-security-patch
```

- JIRA key is **REQUIRED** (format: `PROJECTKEY-###`)
- Description is optional but recommended
- Use kebab-case for descriptions

### Commit Messages

**Required Format:** `PROJECTKEY-###: Brief description of what was done`

```bash
EDGEAI-123: Add JWT authentication to user API
STUDIO-456: Fix memory leak in CUDA kernel allocation
```

- Subject line: 50-72 characters
- Focus on WHAT changed, not HOW
- No type prefixes (`feat:`, `fix:`)

### Pull Requests

- **2 approvals** required for `main`
- **1 approval** required for `develop`
- All CI/CD checks must pass
- PR title: `PROJECTKEY-### Brief description`
- Link to JIRA ticket in description

---

## Code Quality

### General Principles

- **Consistency:** Follow existing codebase patterns
- **Readability:** Code is read more than written
- **Simplicity:** Prefer straightforward solutions
- **Error Handling:** Validate inputs, provide clear error messages
- **Performance:** Consider time/space complexity for edge deployment

### Language Standards

- **Rust:** `cargo fmt` and `cargo clippy`
- **Python:** PEP 8, type hints preferred
- **C/C++:** Follow project's `.clang-format`
- **Go:** `go fmt`, Effective Go guidelines
- **JavaScript/TypeScript:** ESLint, Prettier, prefer TypeScript

### SonarQube Integration

If project has `sonar-project.properties`:
- Address critical and high-severity issues before PR
- Maintain or improve quality gate scores
- Use SonarLint VSCode plugin for real-time feedback

---

## Testing Requirements

### Coverage Standards

- **Minimum:** 70% (check project-specific thresholds)
- **Critical paths:** 90%+ coverage
- **Edge cases:** Explicit tests for boundary conditions
- **Error paths:** Validate error handling

### Test Organization

Follow project conventions (see `CONTRIBUTING.md`). Common patterns:

**Rust:**
```rust
// Unit tests at end of implementation file
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_data_valid_input() {
        // test implementation
    }
}
```

**Python:**
```
tests/
├── unit/test_module.py
└── integration/test_api_workflow.py
```

### Running Tests

```bash
make test          # Run all tests
make coverage      # Check coverage
cargo test         # Rust
pytest tests/      # Python
go test ./...      # Go
```

---

## License Policy

**CRITICAL:** Strict license policy for all dependencies.

### ✅ Allowed Licenses

- MIT, Apache-2.0, BSD-2/3-Clause, ISC, 0BSD, Unlicense

### ⚠️ Review Required

- MPL-2.0 (file-level copyleft - dependencies only, NOT embedded source)
- LGPL-2.1/3.0 (if dynamically linked)

### ❌ Disallowed Licenses

- GPL (any version)
- AGPL (any version)
- Creative Commons with NC/ND restrictions
- SSPL, BSL, OSL-3.0

**CI/CD automatically:**
- Generates SBOM using scancode-toolkit
- Validates license compliance
- **Blocks PR merges on violations**

**Before adding dependencies:**
1. Check license compatibility
2. Verify no GPL/AGPL in dependency tree
3. Escalate to technical leadership if needed

---

## Security Practices

### Vulnerability Reporting

- Email: `support@au-zone.com` (subject: "Security Vulnerability")
- Expected acknowledgment: 48 hours
- See project's `SECURITY.md` for full process

### Secure Coding

**Input Validation:**
- Validate all external inputs
- Use allowlists over blocklists
- Enforce size/length limits

**Authentication & Authorization:**
- Never hardcode credentials or API keys
- Use environment variables or secure vaults
- Follow principle of least privilege

**Common Vulnerabilities to Avoid:**
- SQL Injection → Use parameterized queries
- XSS → Escape output, use CSP headers
- CSRF → Use tokens
- Path Traversal → Validate file paths
- Command Injection → Avoid shell execution
- Buffer Overflows → Bounds checking, safe functions

---

## Documentation

### When to Document

- Public APIs, functions, classes (ALWAYS)
- Complex algorithms or non-obvious logic
- Performance considerations
- Thread safety and concurrency
- Hardware-specific code

### Documentation Style

```python
def preprocess_image(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """
    Resize and normalize image for model inference.

    Args:
        image: Input image as HWC numpy array (uint8)
        target_size: Target dimensions as (width, height)

    Returns:
        Preprocessed image as CHW float32 array normalized to [0, 1]

    Raises:
        ValueError: If image dimensions are invalid

    Performance:
        Uses bilinear interpolation. For better quality with 2x cost,
        use bicubic via config.interpolation = 'bicubic'
    """
```

### Project Documentation Updates

When modifying code, update:
- README if user-facing behavior changes
- API docs if function signatures change
- CHANGELOG for all user-visible changes
- Configuration guides if new options added

---

## Project-Specific Guidelines

### EdgeFirst HAL

This document provides guidance for AI coding assistants (GitHub Copilot, Cursor, etc.) working with the EdgeFirst Hardware Abstraction Layer codebase.

**Project Overview:**

EdgeFirst HAL is a Rust-based hardware abstraction layer with Python bindings, providing zero-copy memory management, hardware-accelerated image processing, and ML model post-processing for embedded Linux platforms.

### Technology Stack

- **Language:** Rust 1.70+ with Python 3.8+ bindings
- **Build system:** Cargo workspace, maturin for Python
- **Key dependencies:** ndarray, rayon, pyo3, enum_dispatch
- **Target platforms:** Linux x86_64, ARM64 (NXP i.MX, generic embedded)

### Architecture

- **Pattern:** Layered with fallback chains (hardware → software)
- **Data flow:** Zero-copy tensor abstractions with DMA/SHM/heap fallback
- **Error handling:** Result types with specific error enums

### Build and Deployment

```bash
# Build
cargo build --workspace

# Test
cargo test --workspace
pytest tests/

# Documentation
cargo doc --workspace --no-deps
```

### Performance Targets

- **Inference latency:** Minimize overhead in tensor operations
- **Memory footprint:** Zero-copy where possible
- **Startup time:** Fast initialization with lazy hardware detection

### Hardware Specifics

- **NPU support:** Generic tensor abstraction (backend-agnostic)
- **GPU acceleration:** NXP G2D, OpenGL (platform-dependent)
- **Platform quirks:** DMA-heap availability varies; always provide CPU fallback

## Key Architecture Patterns

### 1. Workspace Structure

This is a Cargo workspace with multiple crates:
- `crates/tensor/` - Zero-copy tensor abstractions
- `crates/image/` - Hardware-accelerated image processing
- `crates/decoder/` - YOLO and model output decoding
- `crates/tracker/` - Object tracking algorithms
- `crates/python/` - PyO3 Python bindings
- `crates/g2d-sys/` - FFI bindings for NXP G2D
- `crates/edgefirst/` - Top-level re-export crate

### 2. Naming Conventions

**Rust Types:**
- Core types: `Tensor<T>`, `TensorImage`, `ImageProcessor`, `Decoder`
- Trait names: `TensorTrait<T>`, `ImageProcessorTrait`
- Enum variants: PascalCase (e.g., `DmaTensor`, `ShmTensor`, `MemTensor`)

**Python Wrapper Types:**
- Use `Py` prefix: `PyTensor`, `PyTensorImage`, `PyImageProcessor`
- This distinguishes Python-facing types from internal Rust types
- Located in `crates/python/src/`

**File Organization:**
- One module per major type (e.g., `tensor.rs`, `image.rs`, `decoder.rs`)
- Enums use `enum_dispatch` for zero-cost polymorphism

### 3. Memory Management Pattern

**Critical Convention:** The HAL uses a fallback chain for memory allocation:

```rust
// Automatic fallback: DMA → Shared Memory → Heap
let tensor = Tensor::<u8>::new(&[height, width, channels], None)?;

// Explicit type selection
let tensor = Tensor::<u8>::new(&[height, width, channels], Some(TensorType::Dma))?;
```

**Implementation details:**
- `DmaTensor<T>`: Linux DMA-heap for zero-copy hardware access
- `ShmTensor<T>`: POSIX shared memory for IPC
- `MemTensor<T>`: Standard heap allocation as fallback
- Environment variable `EDGEFIRST_TENSOR_FORCE_MEM=1` forces heap allocation

### 4. Error Handling

All public APIs return `Result<T, E>` with specific error types:
- `TensorError` for tensor operations
- `ImageError` for image processing
- `DecoderError` for model decoding

**Pattern:**
```rust
use crate::error::{ImageError, ImageResult};

pub fn convert(&self, src: &TensorImage, dst: &mut TensorImage) -> ImageResult<()> {
    // Implementation
}
```

### 5. Hardware Acceleration Pattern

Image processing uses a fallback chain:

```rust
// Preference: G2D → OpenGL → CPU
impl ImageProcessorTrait for ImageProcessor {
    fn convert(&mut self, src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        if let Some(g2d) = &mut self.g2d {
            // Try G2D hardware acceleration
            if g2d.can_convert(src.format, dst.format) {
                return g2d.convert(src, dst, options);
            }
        }
        // Fallback to CPU
        self.cpu.convert(src, dst, options)
    }
}
```

### 6. Python Bindings Pattern

**PyO3 Integration:**
```rust
#[pyclass(name = "TensorImage")]
pub struct PyTensorImage {
    inner: TensorImage,
}

#[pymethods]
impl PyTensorImage {
    #[new]
    pub fn new(width: usize, height: usize, format: PyFourCC) -> PyResult<Self> {
        let inner = TensorImage::new(width, height, format.into(), None)
            .map_err(|e| PyErr::new::<PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }
}
```

**NumPy Integration:**
- Use `PyReadonlyArrayDyn` for input arrays
- Use `PyArrayDyn` for output arrays  
- Always validate array dtype matches Rust type `T`

### 7. Testing Patterns

**Test Data Location:**
- Test data files in `testdata/` at workspace root
- Accessible via relative paths from crate tests

**Python Test Pattern:**
```python
# tests/image/test_feature.py
import unittest
import edgefirst_hal as ef

class TestFeature(unittest.TestCase):
    def test_operation(self):
        # Arrange
        input_data = ...
        
        # Act
        result = ef.operation(input_data)
        
        # Assert
        self.assertEqual(result, expected)
```

**Rust Test Pattern:**
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature() -> Result<()> {
        let input = setup();
        let result = function(input)?;
        assert_eq!(result, expected);
        Ok(())
    }
}
```

## Common Tasks

### Adding a New Tensor Operation

1. Add method to `TensorTrait<T>` in `crates/tensor/src/lib.rs`
2. Implement for `DmaTensor`, `ShmTensor`, `MemTensor`
3. Add Python binding in `crates/python/src/tensor.rs`
4. Add test in `tests/test_tensor.py`
5. Update type stubs in `crates/python/edgefirst_hal.pyi`

### Adding a New Image Format

1. Add FourCC variant in dependency `four-char-code`
2. Update format conversion logic in `crates/image/src/`
3. Add G2D support (if hardware supports it) in `crates/image/src/g2d.rs`
4. Add CPU fallback in `crates/image/src/cpu.rs`
5. Add Python tests in `tests/image/`

### Adding a New Decoder Type

1. Create decoder struct in `crates/decoder/src/`
2. Implement decoding logic following YOLO pattern
3. Add builder support in `DecoderBuilder`
4. Add Python binding in `crates/python/src/decoder.rs`
5. Add test with model outputs in `tests/decoder/`

## Build and Test Commands

```bash
# Build all crates
cargo build --workspace

# Test all Rust code
cargo test --workspace

# Build Python bindings
maturin develop -m crates/python/Cargo.toml

# Test Python bindings
python -m pytest tests/

# Run specific crate tests
cargo test -p edgefirst_tensor
cargo test -p edgefirst_image
cargo test -p edgefirst_decoder

# Format code (requires Rust nightly)
cargo +nightly fmt --all

# Lint
cargo clippy --workspace

# Run benchmarks
cargo bench -p edgefirst_image
```

## Code Style Preferences

### Rust
- Use `rustfmt` configuration in `rustfmt.toml`
- Prefer `?` over `unwrap()` in production code
- Use `Result<T, E>` for all fallible operations
- Document public APIs with `///` doc comments
- Include usage examples in doc comments

### Python
- Follow PEP 8 style guide
- Use type hints in `.pyi` stub files
- Match Python naming (snake_case functions, PascalCase classes)
- Provide good error messages from Rust errors

## Dependencies to Know

**Core Rust:**
- `ndarray` - N-dimensional array operations
- `rayon` - Data parallelism
- `enum_dispatch` - Zero-cost enum polymorphism

**Image Processing:**
- `fast_image_resize` - CPU-based resizing
- `zune-jpeg`, `zune-png` - Image decoding

**Python:**
- `pyo3` - Rust-Python bindings
- `numpy` - Array integration

**Platform-Specific:**
- `dma-heap` - Linux DMA allocation
- `nix` - Unix system calls

## Platform Considerations

- **Linux (NXP i.MX)**: Full hardware acceleration via G2D
- **Linux (Generic)**: DMA-heap and OpenGL available
- **macOS/Windows**: CPU-only fallback paths

When adding features:
- Always provide CPU fallback
- Use feature flags for optional hardware support
- Test on multiple platforms if possible

## Performance Tips

- Prefer zero-copy operations via tensor views
- Use memory-mapped file descriptors for large data
- Leverage hardware accelerators when available
- Use Rayon for CPU parallelization
- Minimize allocations in hot paths

## Documentation Standards

All public APIs must include:
```rust
/// Brief one-line description.
///
/// More detailed explanation if needed.
///
/// # Arguments
///
/// * `arg1` - Description
/// * `arg2` - Description
///
/// # Returns
///
/// Description of return value
///
/// # Errors
///
/// When this function returns an error and why
///
/// # Examples
///
/// ```rust
/// use edgefirst_hal::tensor::Tensor;
///
/// let tensor = Tensor::<u8>::new(&[640, 480, 3], None)?;
/// ```
pub fn function_name(arg1: Type1, arg2: Type2) -> Result<ReturnType> {
    // Implementation
}
```

## Common Pitfalls to Avoid

1. **Don't mix tensor types unnecessarily** - Use `None` for automatic selection
2. **Don't forget CPU fallbacks** - Hardware acceleration may not be available
3. **Don't unwrap in library code** - Always return `Result` for errors
4. **Don't forget Python type stubs** - Update `.pyi` when adding Python APIs
5. **Don't skip tests** - Add both Rust and Python tests for new features
6. **Don't forget documentation** - Public APIs need doc comments

## File Descriptor Management

When working with tensors that expose file descriptors:
```rust
// Clone FD for sharing with other process
let fd = tensor.clone_fd()?;

// FD is automatically closed when tensor is dropped
// Don't manually close FDs obtained from tensors
```

## Contact

For questions about architecture or conventions:
- See [CONTRIBUTING.md](CONTRIBUTING.md)
- Ask in GitHub Discussions
- Email: support@au-zone.com

---

## Working with AI Assistants

### For GitHub Copilot / Cursor

- Verify suggestions match project conventions
- Run linters after accepting suggestions
- Ensure meaningful test assertions
- Follow security best practices

### For Claude Code / Chat-Based

1. **Provide context:** Share relevant files and requirements
2. **Verify outputs:** Review critically before committing
3. **Iterate:** Refine through follow-up questions
4. **Document decisions:** Capture architectural choices
5. **Test thoroughly:** AI-generated code needs verification

### Common Pitfalls

- **Hallucinated APIs:** Verify library functions exist
- **Outdated patterns:** Check current best practices
- **Over-engineering:** Prefer simple solutions
- **Missing edge cases:** Explicitly test boundaries
- **License violations:** AI may suggest incompatible code

---

## Getting Help

**Development questions:**
- Check `CONTRIBUTING.md` for setup instructions
- Review existing code for patterns
- Search GitHub Issues
- Ask in GitHub Discussions (public repos)

**Security concerns:**
- Email: `support@au-zone.com` (subject: "Security Vulnerability")
- Do not disclose publicly

**License questions:**
- Review license policy above
- Check project's `LICENSE` file
- Contact technical leadership if unclear

---

## Process Documentation Reference

**For Au-Zone Internal Developers:**

This project follows Au-Zone Technologies Software Process standards. Complete process documentation is available in the internal Software Process repository.

### Key Process Areas

**Git Workflow:**
- Branch naming: `<type>/PROJECTKEY-###[-description]`
- Commit format: `PROJECTKEY-###: Brief description`
- Multi-repository coordination: Use same JIRA key across related repos
- Release management: Semantic versioning with comprehensive pre-release checklist

**JIRA Integration:**
- Automatic ticket transitions on branch/PR events
- Time tracking with Tempo
- Blocker management and dependency tracking

**SBOM and License Compliance:**
- Automated via `.github/scripts/generate_sbom.sh`
- Scancode-toolkit for unified scanning (all languages)
- License policy enforced in CI/CD via `.github/scripts/check_license_policy.py`
- MPL-2.0 allowed as dependency only, not in source code
- Multi-language project support with parent directory workaround

**Testing Standards:**
- 70% minimum coverage per component/language
- Testing pyramid: Unit (70%+), Integration (20-25%), E2E (5-10%)
- Performance benchmarks for critical paths
- Multi-language testing strategies

**Release Process:**
- Semantic versioning (MAJOR.MINOR.PATCH)
- Pre-release checklist (code quality, docs, SBOM, security)
- CHANGELOG.md following Keep a Changelog format
- Automated publishing to package registries

**Documentation Requirements:**
- 5 mandatory files: README, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, LICENSE
- CHANGELOG.md for all versioned releases
- Mermaid diagrams required (no ASCII art)
- ARCHITECTURE.md for complex projects

**For External Contributors:**

Open source contributors should follow the simplified guidelines in CONTRIBUTING.md. Internal process complexity is handled by automation and Au-Zone team members.

---

## Quick Reference

**Branch:** `feature/JIRA-123-description`
**Commit:** `JIRA-123: Brief description`
**PR:** 2 approvals for main, 1 for develop
**Licenses:** ✅ MIT/Apache-2.0/BSD | ❌ GPL/AGPL
**Tests:** 70% minimum coverage
**Security:** Email `support@au-zone.com`

---

*This document helps AI assistants contribute effectively while maintaining quality, security, and consistency.*
