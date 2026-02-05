# macOS Compatibility

This document summarizes the work done to enable macOS support for the EdgeFirst HAL library and outlines remaining work for full feature parity.

## Current Status

The library now compiles and runs on macOS with CPU fallbacks and POSIX shared memory support.

| Feature | Linux | macOS | Notes |
|---------|-------|-------|-------|
| Tensor (Mem) | ✅ | ✅ | System memory allocation |
| Tensor (SHM) | ✅ | ✅ | POSIX shared memory via `shm_open`/`mmap` |
| Tensor (DMA) | ✅ | ❌ | Linux kernel feature (dmabuf) |
| Image CPU | ✅ | ✅ | Software processing with SIMD |
| Image G2D | ✅ | ❌ | NXP i.MX hardware acceleration |
| Image OpenGL | ✅ | ❌ | Requires platform-specific implementation |
| Python bindings | ✅ | ✅ | Full support |

## Changes Made

### Tensor Crate (`crates/tensor`)

**Cargo.toml**
- Moved `nix` dependency from Linux-only to all Unix platforms (`cfg(unix)`)
- Kept `dma-heap` as Linux-only (`cfg(target_os = "linux")`)
- Moved `procfs` dev-dependency to Linux-only

**src/lib.rs**
- Updated module exports with platform-specific guards
- Added three-tier auto-selection logic:
  - Linux: DMA → SHM → Mem (with fallback)
  - macOS/BSD: SHM → Mem (with fallback)
  - Windows: Mem only
- Added `cfg(all(unix, not(target_os = "linux")))` guards for macOS-specific code paths

**src/error.rs**
- Changed `NixError` to `cfg(unix)` (was Linux-only)
- Kept `UnknownDeviceType` as `cfg(target_os = "linux")`
- Added `Display` and `std::error::Error` trait implementations

**src/mem.rs**
- Changed `from_fd`/`clone_fd` stub methods from `cfg(target_os = "linux")` to `cfg(unix)`

### Image Crate (`crates/image`)

**src/lib.rs**
- Wrapped G2D-specific test code in `cfg(target_os = "linux")`

**src/cpu.rs**
- Fixed test import to use `crate::RGBA` instead of `g2d_sys::RGBA`

**src/error.rs**
- Added `Display` and `std::error::Error` trait implementations

### Python Bindings (`crates/python`)

**src/tensor.rs**
- Updated cfg attributes from `target_os = "linux"` to `unix` for SHM-related code
- Added proper enum variant handling for `TensorMemory::Shm` on all Unix platforms

## Remaining Work

### OpenGL Support on macOS

The current OpenGL implementation (`crates/image/src/opengl_headless.rs`) is Linux-specific because it relies on:

1. **GBM (Generic Buffer Management)** - Linux DRM/KMS subsystem for buffer allocation
2. **EGL with GBM platform** - For creating headless OpenGL contexts
3. **DMA-BUF** - For zero-copy texture import/export

#### Recommended Approach for macOS OpenGL

**Option 1: CGL + IOSurface (Recommended)**
- Use Apple's Core OpenGL (CGL) for context creation
- Use `IOSurface` for zero-copy buffer sharing (similar to DMA-BUF)
- Requires `core-graphics`, `io-surface` crates

```rust
#[cfg(target_os = "macos")]
mod opengl_macos {
    // CGL context creation
    // IOSurface for zero-copy
}
```

**Option 2: ANGLE**
- Use Google's ANGLE library for cross-platform EGL/OpenGL ES
- Translates OpenGL ES to Metal on macOS
- More portable but adds dependency

**Option 3: Metal (Future)**
- Native Apple GPU API
- Best performance on macOS
- Would require a separate `metal.rs` backend

#### Implementation Tasks

1. [ ] Create `opengl_macos.rs` with CGL context management
2. [ ] Implement IOSurface integration for zero-copy textures
3. [ ] Add feature flag `opengl-macos` in `Cargo.toml`
4. [ ] Update `ImageProcessor` to select appropriate backend
5. [ ] Add macOS-specific OpenGL tests

### Windows Support (Future)

For Windows compatibility, the following work is needed:

**Tensor Crate**
- [ ] Windows shared memory via `CreateFileMapping`/`MapViewOfFile` (if IPC needed)
- [ ] Basic `Mem` tensors already work

**Image Crate**
- [ ] CPU backend already works
- [ ] WGL or ANGLE for OpenGL support
- [ ] Consider DirectX compute for GPU acceleration

## Testing

Run the full test suite on macOS:

```bash
cargo test --workspace
```

Run benchmarks:

```bash
cargo bench -p edgefirst-tensor
cargo bench -p edgefirst-image
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `EDGEFIRST_TENSOR_FORCE_MEM` | Force Mem tensor allocation (skip SHM) |
| `EDGEFIRST_DISABLE_G2D` | Disable G2D hardware acceleration (Linux) |

## Platform Detection Pattern

The codebase uses these conditional compilation patterns:

```rust
// Linux-only (DMA, G2D, current OpenGL)
#[cfg(target_os = "linux")]

// All Unix (SHM, file descriptors)
#[cfg(unix)]

// macOS/BSD only (future macOS-specific code)
#[cfg(all(unix, not(target_os = "linux")))]

// Windows only (future)
#[cfg(windows)]

// Non-Unix (Windows, WASM, etc.)
#[cfg(not(unix))]
```
