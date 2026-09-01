<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# EdgeFirst HAL Python Bindings — Architecture

## Overview

The Python bindings are **independent extension modules** rather than one
`edgefirst_hal` module. `python-common` is the rlib holding the shared PyO3
binding code; each `python-<pkg>` crate is a thin `cdylib` that selects the
parts it needs by cargo feature and registers them into one module.

```
crates/python-common/          rlib — all binding code, feature-gated
  src/tensor.rs                PyTensor, PyTensorMap, PyPixelFormat, PyRegion
  src/colorimetry.rs           PyColorimetry
  src/image.rs                 PyImageProcessor
  src/tiling.rs                SAHI tiling bindings
  src/decoder.rs               PyDecoder
  src/tracker.rs               PyTracker

crates/python-tensor/          cdylib -> edgefirst.tensor      (_tensor.so)
crates/python-codec/           cdylib -> edgefirst.codec       (_codec.so)
crates/python-image/           cdylib -> edgefirst.image       (_image.so)
crates/python-decoder/         cdylib -> edgefirst.decoder     (_decoder.so)
crates/python-tracker/         cdylib -> edgefirst.tracker     (_tracker.so)
```

Each sibling extension **links `libedgefirst_tensor.so`**. Tensor
implementation lives in that one shared library; the other wheels do not
embed a second copy. `edgefirst-tensor`'s wheel ships the `.so` (and the
`.so.0` soname symlink); codec/image/decoder set `DT_NEEDED` +
`RUNPATH=$ORIGIN/../tensor`. Tracker does not link tensor: it consumes
plain detection values. On Windows the same library is `edgefirst_tensor.dll`
(no `lib` prefix, no SONAME) and there is no rpath: codec/image/decoder's
`__init__.py` call `os.add_dll_directory()` on `edgefirst/tensor/` before
importing their `.pyd`.

A JPEG-only user installs `edgefirst-codec` + `edgefirst-tensor`. An
image-only user installs `edgefirst-image` + `edgefirst-tensor` and does
not pull the model decoder.

`maturin` is invoked with `--auditwheel skip` so auditwheel does not vendor
a second tensor copy into every wheel.

## Per-module type identity

Each extension module caches its own `#[pyclass]` type objects in its own
`static LazyTypeObject`. So:

```python
edgefirst.tensor.Tensor is edgefirst.image.Tensor   # False — by design
```

Cross-package handoff is **duck-typed** through the
`__edgefirst_tensor__` capsule protocol rather than an `isinstance` check.
A producer exposes `__edgefirst_tensor__()` returning a `PyCapsule` named
`edgefirst_tensor_v1` wrapping a `#[repr(C)] TensorDesc`; a consumer reads
the descriptor without ever naming the producer's type.

The capsule owns both the descriptor **and** the producer's `HostPin`, so
the address stays valid for the capsule's life.

## PEP 420 namespace

There is no `edgefirst/__init__.py` in any wheel. `edgefirst` is an implicit
namespace package, which is what lets independently-installed distributions
contribute submodules to it. A single regular `__init__.py` anywhere would
shadow the namespace and make the others unimportable —
`tests/packaging/test_namespace.py` gates exactly this.

## Stable ABI / abi3

All packages build against the limited API. CI uses `abi3-py311`; the
release pipeline additionally builds `abi3-py38`. They **must agree** — a
mixed set silently forks the supported interpreter range, so the
wheel-layout gate compares the full `(python, abi, platform)` tag triple.

The buffer protocol is py311-only, which is why `abi3-py38` builds omit it.

## NumPy → Tensor copy strategy

`python-common/src/tensor.rs` holds `copy_numpy_to_tensor_dyn`, which
inspects the source array's strides and dispatches three ways:

| Path | Source layout | Strategy | Cost |
|------|---------------|----------|------|
| 1 | Fully contiguous | `copy_from_slice` (memcpy), rayon-parallel ≥ 256 KiB | Lower bound |
| 2 | Strided with contiguous inner rows | Per-row memcpy over outer dims | ≈ 5 % of Path 1 |
| 3 | Fully strided (transposed, every-other) | `np.ascontiguousarray()` then Path 1 | ≈ 4× Path 1 |

`from_numpy` **copies**; it is not zero-copy. For a genuinely zero-copy
handoff use the capsule protocol or a pinned tensor.

A stride-padded **destination** is handled ahead of all three: a DMA-BUF or
PBO allocated with GPU pitch alignment exposes `stride × height` while the
logical element count is smaller, so the copy places `row_elems` per row and
steps the padding. `HostView.numpy()` must honour the same row stride as
`memoryview` — a tight `W*C` read of a padded DMA tensor shears every row
after the first.

## Mapping, pinning and sync

`Tensor.map()` returns a `TensorMap` context manager that unmaps on exit —
required because `Pbo` and `Dma` mappings hold driver state that must be
released deterministically.

The tensor core separates **address** from **coherency**: `pin_host()`
yields a `HostPin` that borrows nothing and outlives any guard, and
`sync_for_cpu` / `sync_for_device` bracket CPU access independently.

## Process-shutdown safety

The image crate's GL backend installs a defence-in-depth shutdown strategy
to survive Python's non-deterministic finalization order — see
[`crates/image/ARCHITECTURE.md#process-shutdown-resource-cleanup`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md#process-shutdown-resource-cleanup).
The bindings inherit it; they add no Python-specific finalizers and rely on
the Rust `Drop` chain.

## Build

```bash
maturin develop -m crates/python-tensor/Cargo.toml   # editable, one package
make build-python                                    # all packages
make wheels                                          # + layout gate
```

`module-name` in each `pyproject.toml` uses the dotted path
(`edgefirst.tensor._tensor`) with `python-source` pointing at the package's
`python/` tree.

## Cross-References

- Testing: [TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-common/TESTING.md)
- Tensor core: [../tensor/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/ARCHITECTURE.md)
- C ABI: [../tensor-capi/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor-capi/ARCHITECTURE.md)
