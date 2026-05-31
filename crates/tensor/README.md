# edgefirst-tensor

[![Crates.io](https://img.shields.io/crates/v/edgefirst-tensor.svg)](https://crates.io/crates/edgefirst-tensor)
[![Documentation](https://docs.rs/edgefirst-tensor/badge.svg)](https://docs.rs/edgefirst-tensor)
[![License](https://img.shields.io/crates/l/edgefirst-tensor.svg)](LICENSE)

**Zero-copy tensor memory management for edge AI applications.**

This crate provides a unified interface for managing multi-dimensional arrays (tensors) with support for different memory backends optimized for ML inference pipelines.

## Role in edgefirst-hal

`edgefirst-tensor` is the foundation of the data-plane crates in the
EdgeFirst HAL workspace. The image, decoder, capi, and gpu-probe crates
all depend on it; the tracker and bench crates are independent and
operate on their own types.

- [`edgefirst-image`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/) consumes `Tensor<u8>` / `TensorDyn` for image processor input/output buffers and provides the `PboOps` trait impl that backs `PboTensor`.
- [`edgefirst-decoder`](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/) reads model output via `Tensor<T>` and `TensorMap`.
- [`edgefirst-hal`](https://github.com/EdgeFirstAI/hal/blob/main/crates/hal/) re-exports this crate as `edgefirst_hal::tensor`.
- [`edgefirst-hal-capi`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/) crosses the FFI boundary using `from_fd`, `clone_fd`, and `from_planes`.
- [`gpu-probe`](https://github.com/EdgeFirstAI/hal/blob/main/crates/gpu-probe/) uses it to allocate the DMA-BUF round-trip buffer the probe verifies.
- [`edgefirst-tracker`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/) and [`edgefirst-bench`](https://github.com/EdgeFirstAI/hal/blob/main/crates/bench/) do **not** depend on this crate — tracker works against `DetectionBox` and `nalgebra`, bench wraps `serde_json` for benchmark IO.

This crate has **no internal `edgefirst-*` dependencies**.

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

## CUDA Tensor Mapping

On CUDA-capable devices (e.g. Jetson Orin-series) the float PBO produced
by `ImageProcessor::convert()` can be mapped directly to a CUDA device
pointer. No link-time dependency on `libcudart` — the symbols are resolved
at runtime via `dlopen`.

### Availability probe

```rust
use edgefirst_tensor::is_cuda_available;

if is_cuda_available() {
    println!("CUDA runtime present; zero-copy path available");
}
```

### Usage — try CUDA map, fall back to host

```rust
use edgefirst_tensor::{TensorTrait, TensorMapTrait};

// Per-frame: prefer zero-copy CUDA, fall back to host map
if let Some(cuda) = dst.cuda_map() {
    // cuda.device_ptr() — raw CUDA device pointer, valid until `cuda` drops.
    // cuda.len()        — byte length of the mapped region.
    trt_enqueue(cuda.device_ptr(), cuda.len());
    // Drop `cuda` here — releases the PBO before the next convert().
} else {
    let host = dst.map()?;
    trt_enqueue_host(host.as_slice());
}
```

`cuda_map()` returns `None` when:
- `libcudart` is not present at runtime.
- The tensor is not PBO- or DMA-BUF-backed.
- CUDA registration of the backing buffer failed (logged at `warn`).

The `CudaMap` guard must be dropped before the next `ImageProcessor::convert()`
call that writes into the same tensor — the GL pipeline must not touch a
PBO while CUDA has it mapped. See
[ARCHITECTURE.md § Zero-copy CUDA tensor mapping](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/ARCHITECTURE.md#zero-copy-cuda-tensor-mapping)
for the full aliasing rules, DMA-BUF import path, and drop-order contract.

### C API

```c
if (hal_is_cuda_available()) {
    void *map = hal_tensor_cuda_map(tensor);
    if (map) {
        size_t size   = 0;
        void *dev_ptr = hal_tensor_cuda_device_ptr(map, &size);
        trt_enqueue(dev_ptr, size);
        hal_tensor_cuda_unmap(map);  // must call before next convert()
    }
}
```

### Python

```python
import edgefirst_hal as ef

if ef.is_cuda_available():
    cm = dst.cuda_map()          # returns CudaMap or None
    if cm is not None:
        with cm:                 # context manager — unmap on __exit__
            trt_context.execute(cm.device_ptr)
```

## Documentation

- Architecture overview: [ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/ARCHITECTURE.md)
- Testing guide: [TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/TESTING.md)
- Full API reference: [docs.rs/edgefirst-tensor](https://docs.rs/edgefirst-tensor)
- Project README: [../../README.md](https://github.com/EdgeFirstAI/hal/blob/main/README.md)

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE) for details.
