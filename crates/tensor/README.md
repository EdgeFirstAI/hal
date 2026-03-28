# edgefirst-tensor

[![Crates.io](https://img.shields.io/crates/v/edgefirst-tensor.svg)](https://crates.io/crates/edgefirst-tensor)
[![Documentation](https://docs.rs/edgefirst-tensor/badge.svg)](https://docs.rs/edgefirst-tensor)
[![License](https://img.shields.io/crates/l/edgefirst-tensor.svg)](LICENSE)

**Zero-copy tensor memory management for edge AI applications.**

This crate provides a unified interface for managing multi-dimensional arrays (tensors) with support for different memory backends optimized for ML inference pipelines.

## Memory Types

| Type | Description | Use Case |
|------|-------------|----------|
| **DMA** | Linux DMA-BUF allocation | Hardware accelerators (GPU, NPU, video codecs) |
| **SHM** | POSIX shared memory | Inter-process communication, zero-copy IPC |
| **Mem** | Standard heap allocation | General purpose, maximum compatibility |
| **PBO** | OpenGL Pixel Buffer Object | GPU-accelerated image processing (created by ImageProcessor) |

## Features

- **Automatic memory selection** - Tries DMA → SHM → Mem based on availability
- **Zero-copy sharing** - Share tensors between processes via file descriptors
- **Memory mapping** - Efficient CPU access to tensor data
- **ndarray integration** - Optional conversion to/from `ndarray::Array` (feature: `ndarray`)

## Quick Start

```rust
use edgefirst_tensor::{Tensor, TensorMemory, TensorTrait, TensorMapTrait};

// Create a tensor with automatic memory selection
let tensor = Tensor::<f32>::new(&[1, 3, 224, 224], None, None)?;
println!("Memory type: {:?}", tensor.memory());

// Create with explicit memory type
let dma_tensor = Tensor::<u8>::new(&[1920, 1080, 4], Some(TensorMemory::Dma), None)?;

// Map tensor for CPU access
let mut map = tensor.map()?;
map.as_mut_slice().fill(0.0);

// Share via file descriptor (Unix only)
#[cfg(unix)]
let fd = tensor.clone_fd()?;
```

## Platform Support

| Platform | DMA | SHM | Mem | PBO |
|----------|-----|-----|-----|-----|
| Linux | Yes | Yes | Yes | Yes (with OpenGL) |
| macOS | No | Yes | Yes | No |
| Other Unix | No | Yes | Yes | No |
| Windows | No | No | Yes | No |

## Feature Flags

- `ndarray` (default) - Enable `ndarray` integration for array conversions

## Environment Variables

- `EDGEFIRST_TENSOR_FORCE_MEM` - Set to `1` or `true` to force heap allocation

## PlaneDescriptor

`PlaneDescriptor` wraps a duplicated file descriptor for use with
`ImageProcessor::import_image()`. It captures optional stride and offset
metadata alongside the fd so that the importer gets a complete picture of the
plane layout without additional out-of-band parameters.

```rust,no_run
use edgefirst_tensor::PlaneDescriptor;
use std::os::fd::BorrowedFd;

// SAFETY: replace 42 with a real, valid fd from a DMA-BUF allocation.
let pd = unsafe { PlaneDescriptor::new(BorrowedFd::borrow_raw(42)) }
    .expect("failed to duplicate fd — check that the fd is valid")
    .with_stride(2048)  // optional: row stride in bytes
    .with_offset(0);    // optional: plane offset in bytes
```

The fd is duplicated eagerly in `new()` — a bad fd fails immediately rather
than inside `import_image`. The caller retains ownership of the original fd.

## DMA-BUF fd Accessors

`TensorDyn` exposes two fd accessors for DMA-backed tensors (Linux only):

- `dmabuf(&self) -> Result<BorrowedFd<'_>>` — Borrow the DMA-BUF fd tied to the tensor's lifetime.
- `dmabuf_clone(&self) -> Result<OwnedFd>` — Duplicate the DMA-BUF fd. Fails with `Error::NotImplemented` if the tensor is not DMA-backed.

```rust,ignore
// Share the buffer with an external consumer (e.g. NPU delegate)
let fd = tensor.dmabuf_clone()?;
delegate.register_buffer(fd)?;
```

## Pixel Format Metadata

Attach a `PixelFormat` to any tensor for image processing:

- `set_format(format: PixelFormat) -> Result<()>` — Validates shape compatibility and stores the format.
- `with_format(format: PixelFormat) -> Result<Self>` — Builder-style consuming variant.

```rust,ignore
let mut t = TensorDyn::new(&[480, 640, 3], DType::U8, None, None)?;
t.set_format(PixelFormat::Rgb)?;
```

## Row Stride

For externally allocated buffers with row padding (e.g. V4L2 camera frames):

- `row_stride(&self) -> Option<usize>` — Stored stride, `None` if tightly packed.
- `effective_row_stride(&self) -> Option<usize>` — Stored stride, or computed from format and width if not set.
- `set_row_stride(stride: usize) -> Result<()>` — Set stride in bytes. Format must be set first.
- `with_row_stride(stride: usize) -> Result<Self>` — Builder-style consuming variant.

## Plane Offset

For buffers where image data does not start at byte 0 of the fd:

- `plane_offset(&self) -> Option<usize>` — Offset in bytes, `None` if zero.
- `set_plane_offset(offset: usize)` — Set byte offset.
- `with_plane_offset(offset: usize) -> Self` — Builder-style consuming variant.

## BufferIdentity

`BufferIdentity` provides a stable cache key for a tensor's underlying buffer.
It is created fresh on every allocation or import and carries:

- `id() -> u64` — Monotonically increasing integer. Changes whenever the buffer changes. Suitable as a HashMap key or EGL image cache key.
- `weak() -> Weak<()>` — Goes dead when the owning tensor (and all clones) are dropped, allowing caches to detect stale entries without holding a strong reference.

`buffer_identity()` is accessible on typed tensors via `TensorTrait`:

```rust,ignore
use edgefirst_tensor::{Tensor, TensorTrait};

let t = Tensor::<u8>::new(&[1920, 1080, 3], None, None)?;
let key = t.buffer_identity().id();
let guard = t.buffer_identity().weak();
// Later: guard.upgrade().is_none() means the tensor was dropped.
```

`BufferIdentity` is used internally by the image processing backends as an EGL
image cache key to avoid redundant GPU texture imports across frames.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE) for details.
