# AI Assistant Instructions for EdgeFirst HAL

This document provides guidance for AI coding assistants (GitHub Copilot, Cursor, etc.) working with the EdgeFirst Hardware Abstraction Layer codebase.

## Project Overview

EdgeFirst HAL is a Rust-based hardware abstraction layer with Python bindings, providing zero-copy memory management, hardware-accelerated image processing, and ML model post-processing for embedded Linux platforms.

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
- Core types: `Tensor<T>`, `TensorImage`, `ImageConverter`, `Decoder`
- Trait names: `TensorTrait<T>`, `ImageConverterTrait`
- Enum variants: PascalCase (e.g., `DmaTensor`, `ShmTensor`, `MemTensor`)

**Python Wrapper Types:**
- Use `Py` prefix: `PyTensor`, `PyTensorImage`, `PyImageConverter`
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
impl ImageConverterTrait for ImageConverter {
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

# Format code
cargo fmt --all

# Lint
cargo clippy --workspace -- -D warnings

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
/// use edgefirst::tensor::Tensor;
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
