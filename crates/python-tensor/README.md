# edgefirst-tensor

Zero-copy tensor memory for edge AI inference pipelines — DMA-BUF, IOSurface, AHardwareBuffer, OpenGL PBO, POSIX shared memory and heap behind one Python API.

[![PyPI](https://img.shields.io/pypi/v/edgefirst-tensor.svg)](https://pypi.org/project/edgefirst-tensor/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Part of the EdgeFirst HAL

`edgefirst-tensor` is one of five Python packages built from the [EdgeFirst Hardware Abstraction Layer](https://github.com/EdgeFirstAI/hal).

**The [`EdgeFirstAI/hal`](https://github.com/EdgeFirstAI/hal) repository is the home for all of them** — source, issue tracker, architecture documentation and release notes.

| Package | Provides |
|---|---|
| [`edgefirst-tensor`](https://pypi.org/project/edgefirst-tensor/) | Zero-copy tensor allocation and host/GPU/CUDA mapping (this package) |
| [`edgefirst-codec`](https://pypi.org/project/edgefirst-codec/) | JPEG and PNG decoding directly into pre-allocated tensors |
| [`edgefirst-image`](https://pypi.org/project/edgefirst-image/) | GPU-accelerated colour conversion, resize, letterbox, tiling and drawing |
| [`edgefirst-decoder`](https://pypi.org/project/edgefirst-decoder/) | YOLO and ModelPack output decoding |
| [`edgefirst-tracker`](https://pypi.org/project/edgefirst-tracker/) | ByteTrack multi-object tracking |

This package is the foundation the other four build on; installing any of them installs this one.

## Installation

```bash
pip install edgefirst-tensor
```

Requires Python 3.8 or newer and NumPy. Wheels are published for Linux (x86_64, aarch64), macOS (arm64), and Windows (x86_64).

The `_codec`, `_image`, `_decoder` and `_tracker` extensions locate `libedgefirst_tensor.so` via `DT_RUNPATH=$ORIGIN/../tensor`. That assumes every `edgefirst.*` package lands in the same `site-packages` tree, which `pip` normally guarantees. A split layout (`pip install --target`, some vendored trees) will fail at import with `libedgefirst_tensor.so.0: cannot open shared object file`. On Windows the library is `edgefirst_tensor.dll` in that same `edgefirst/tensor/` directory and there is no rpath: each sibling package's `__init__.py` registers the directory with `os.add_dll_directory()` before loading its extension, because Python 3.8+ does not consult `PATH` for extension-module DLLs.

Packages install under the [PEP 420](https://peps.python.org/pep-0420/) `edgefirst.*` namespace, so imports are `edgefirst.tensor`, `edgefirst.codec`, and so on. No package ships an `edgefirst/__init__.py` — a single one would shadow the namespace and hide its siblings.

## Quick start

Allocate an image tensor, fill it through a mapped host view, and read it back as NumPy:

```python
import numpy as np
from edgefirst.tensor import Tensor, PixelFormat

# Allocate once; a real pipeline reuses the tensor every frame.
# mem=None selects the best backend available on the platform.
tensor = Tensor.image(1920, 1080, PixelFormat.Rgb, None, "readwrite")

with tensor.map() as view:
    frame = np.asarray(view)  # shape, dtype and strides all carried
    frame[:] = 128

print(tensor.shape, tensor.format, tensor.dtype)
```

`map()` returns a `HostView` implementing the buffer protocol, so `np.asarray` wraps the tensor's memory without copying — and because the view publishes shape, dtype and the real row stride, no manual `reshape` is needed and a pitch-aligned DMA buffer is read correctly rather than sheared. The view is released when the `with` block exits.

The map also owns its cache-coherency bracket, and `access` chooses the direction. The default `"readwrite"` flushes the whole buffer on release; a reader does not need that, and on a non-coherent Arm DMA-BUF backing skipping it is a per-frame saving:

```python
with tensor.map("read") as view:
    frame = np.asarray(view)  # read-only view, not writable
```

`pin_host()` is the exception: it is deliberately decoupled from coherency so a pinned address can survive across `convert()` calls, which is why it pairs with an explicit `cpu_access()` bracket instead.

### Handing memory to an external runtime

`pin_host()` returns a stable host address that outlives any map guard and carries no borrow of the tensor, so a pinned buffer can be given to an inference runtime (TFLite custom allocations, ONNX Runtime external tensors) while your frame loop keeps writing to it:

```python
pin = tensor.pin_host("readwrite")
print(hex(pin.ptr), pin.len, pin.alignment)

# pin.ptr stays valid for the lifetime of `pin` — across re-maps and across
# ImageProcessor.convert() calls — so an external runtime can hold on to it.
pin.release()
```

## What this package provides

| API | Purpose |
|---|---|
| `Tensor` | Allocation, reshape, host mapping, NumPy interchange |
| `Tensor.image()` | Allocate with image dimensions and a pixel format |
| `Tensor.map()` / `HostView` | Buffer-protocol host access, released on scope exit |
| `Tensor.pin_host()` / `HostPin` | Stable host address for external runtimes |
| `Tensor.cuda_map()` / `CudaMap` | Zero-copy CUDA device pointer for TensorRT (Jetson) |
| `TensorMemory`, `PixelFormat`, `Region` | Backend selection, pixel layout, sub-regions |
| `Quantization`, `Colorimetry` | Quantization parameters and colour metadata |
| `is_dma_available()` and friends | Runtime capability probes |
| `Tracing`, `build_info()` | Diagnostics |

## Interoperability

Each `edgefirst-*` package is a separate PyO3 extension module, so `edgefirst.tensor.Tensor` and, say, `edgefirst.codec.Tensor` are different Python classes even though they wrap the same Rust type — a known PyO3 limitation ([issue #1444](https://github.com/PyO3/pyo3/issues/1444)). A tensor still crosses package boundaries safely, through the `__edgefirst_tensor__` capsule protocol every `Tensor` implements:

```python
# CORRECT — works regardless of which edgefirst.* package produced obj
if hasattr(obj, "__edgefirst_tensor__"):
    ...

# WRONG — always False for a tensor from a sibling package
if isinstance(obj, edgefirst.image.Tensor):
    ...
```

`edgefirst.tensor.EdgeFirstTensorExportable` is a `typing.Protocol` you can annotate a cross-package parameter with, so a type checker accepts a tensor from any `edgefirst.*` package. See [`crates/python-common/INTEROP.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-common/INTEROP.md) for the full protocol — capsule names, lifetime and ownership rules, and versioning.

## Versioning and changelog

All four `edgefirst-*` packages are versioned and released together with the HAL itself, so a given version number refers to the same source tree in every language. Because of that there is no per-package changelog: release notes for every version live in the single [**CHANGELOG.md**](https://github.com/EdgeFirstAI/hal/blob/main/CHANGELOG.md) in the `hal` repository, which follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Links

- [Changelog](https://github.com/EdgeFirstAI/hal/blob/main/CHANGELOG.md) — release notes for all packages
- [Source](https://github.com/EdgeFirstAI/hal) — the `hal` monorepo
- [Issue tracker](https://github.com/EdgeFirstAI/hal/issues)
- [Package documentation](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/README.md) — the underlying Rust crate, including the memory backend matrix and CUDA mapping
- [EdgeFirst](https://edgefirst.ai)

## License

Apache-2.0
