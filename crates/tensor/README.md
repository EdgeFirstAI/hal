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

| Platform | DMA | SHM | Mem |
|----------|-----|-----|-----|
| Linux | Yes | Yes | Yes |
| macOS | No | Yes | Yes |
| Other Unix | No | Yes | Yes |
| Windows | No | No | Yes |

## Feature Flags

- `ndarray` (default) - Enable `ndarray` integration for array conversions

## Environment Variables

- `EDGEFIRST_TENSOR_FORCE_MEM` - Set to `1` or `true` to force heap allocation

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE) for details.
