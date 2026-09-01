# edgefirst-image

GPU-accelerated image preprocessing for edge AI — colour conversion, resize, letterbox, rotation, tiled inference and annotation drawing, in one call per frame.

[![PyPI](https://img.shields.io/pypi/v/edgefirst-image.svg)](https://pypi.org/project/edgefirst-image/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Part of the EdgeFirst HAL

`edgefirst-image` is one of five Python packages built from the [EdgeFirst Hardware Abstraction Layer](https://github.com/EdgeFirstAI/hal).

**The [`EdgeFirstAI/hal`](https://github.com/EdgeFirstAI/hal) repository is the home for all of them** — source, issue tracker, architecture documentation and release notes.

| Package | Provides |
|---|---|
| [`edgefirst-tensor`](https://pypi.org/project/edgefirst-tensor/) | Zero-copy tensor allocation and host/GPU/CUDA mapping |
| [`edgefirst-codec`](https://pypi.org/project/edgefirst-codec/) | JPEG and PNG decoding directly into pre-allocated tensors |
| [`edgefirst-image`](https://pypi.org/project/edgefirst-image/) | GPU-accelerated colour conversion, resize, letterbox, tiling and drawing (this package) |
| [`edgefirst-decoder`](https://pypi.org/project/edgefirst-decoder/) | YOLO and ModelPack output decoding |
| [`edgefirst-tracker`](https://pypi.org/project/edgefirst-tracker/) | ByteTrack multi-object tracking |

## Installation

```bash
pip install edgefirst-image
```

Requires Python 3.8 or newer; `edgefirst-tensor` and NumPy are installed automatically. Wheels are published for Linux (x86_64, aarch64), macOS (arm64), and Windows (x86_64). This package does not depend on `edgefirst-decoder`.

Packages install under the [PEP 420](https://peps.python.org/pep-0420/) `edgefirst.*` namespace, so the import is `edgefirst.image`.

A single OpenGL ES engine backs this package on every platform — Linux via native EGL and DMA-BUF, macOS and iOS via ANGLE over Metal and IOSurface, Android via native EGL and AHardwareBuffer, Windows via ANGLE over Direct3D 11 with PBO transfers — alongside NXP G2D on i.MX and a portable CPU fallback everywhere else. Backend selection is automatic; no code changes are needed to move between them. The Windows wheel bundles ANGLE's `libEGL.dll` and `libGLESv2.dll` next to the extension module, so no setup is required there; `EDGEFIRST_ANGLE_ADAPTER` picks the GPU on multi-adapter machines.

## Quick start

Preprocess a frame into a model input tensor. `convert()` performs colour conversion, resize, letterboxing, rotation and cropping in a single GPU pass:

```python
import numpy as np
from edgefirst.image import ImageProcessor, PixelFormat

processor = ImageProcessor()

# Allocate once, outside the loop.
src = processor.create_image(1280, 720, PixelFormat.Rgb, "uint8", "readwrite")
model_input = processor.create_image(640, 640, PixelFormat.Rgb, "uint8", "readwrite")

# Fill `src` from your capture source.
with src.map() as view:
    np.frombuffer(view, dtype=np.uint8)[:] = 200

# Omit letterbox= to stretch instead of preserving aspect ratio.
processor.convert(src, model_input, letterbox=[114, 114, 114, 255])

with model_input.map() as view:
    frame = np.frombuffer(view, dtype=np.uint8).reshape(640, 640, 3)
```

Use `create_image()` rather than allocating tensors yourself: it supplies DMA-BUF or PBO backing for zero-copy GPU import and GPU pitch alignment, which the plain allocator cannot guarantee.

`PixelFormat` and the other value types (`TensorMemory`, `Region`, the colour axis enums) are accepted from any `edgefirst.*` package, not just this one — they compare and hash equal across packages by value, so `==`, dict keys and set membership all work regardless of which package's copy you pass. Tensors, `Decoder` and `ProtoData` cross packages too, through the capsule protocols. Importing `PixelFormat` from `edgefirst.image` when calling into this package is still good style for readability, not a requirement. See the Interoperability section below for the one thing that does not cross: `isinstance` against a concrete class.

## What this package provides

| API | Purpose |
|---|---|
| `ImageProcessor` | The engine; owns the GPU context and caches |
| `create_image()` | Allocate a GPU-backed, pitch-aligned image tensor |
| `convert()` | Colour conversion, resize, letterbox, rotation, crop in one pass |
| `Normalization`, `ColorMode` | Model input scaling and channel order |
| `Flip`, `Rotation`, `Fit` | Geometry controls |
| `TilingConfig`, `TileSpec`, `tile_grid()` | Tiled (SAHI) inference layout. Merge (`TiledFrameAccumulator`, `lift_tile_boxes`) lives on `edgefirst-decoder`. |
| `MaskResolution`, drawing APIs | Segmentation mask and bounding box annotation |
| `align_width_for_gpu_pitch()` and friends | Pitch alignment helpers |

### Tiled inference (SAHI)

Small objects vanish when a high-resolution frame is squeezed into a 640×640 model input. The tiling APIs render an overlapping tile grid in a single GPU pass and merge per-tile detections back to full-frame coordinates, so small-object recall improves without a second inference pipeline.

## Interoperability

`convert()`, `draw_decoded_masks()`, `draw_proto_masks()`, `materialize_masks()` and the other `ImageProcessor` entry points accept a tensor or `ProtoData` from *any* `edgefirst.*` package. Fused decode+draw lives on `Decoder.draw_onto`. Each extension module registers its own type objects ([PyO3 issue #1444](https://github.com/PyO3/pyo3/issues/1444) — `isinstance` across packages is always `False`), so acceptance goes through duck-typed capsule protocols instead:

```python
# CORRECT — works regardless of which edgefirst.* package produced obj
if hasattr(obj, "__edgefirst_tensor__"):
    ...

# WRONG — always False for a tensor from a sibling package
if isinstance(obj, edgefirst.codec.Tensor):
    ...
```

`edgefirst.image.EdgeFirstTensorExportable`, `EdgeFirstDecoderExportable` and `EdgeFirstProtoDataExportable` are `typing.Protocol` classes (re-exported here from `edgefirst.tensor` / `edgefirst.decoder`) you can annotate a cross-package parameter with. See [`crates/python-common/INTEROP.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-common/INTEROP.md) for the full protocol — capsule names, lifetime and ownership rules, and versioning.

## Versioning and changelog

All five `edgefirst-*` packages are versioned and released together with the HAL itself, so a given version number refers to the same source tree in every language. Because of that there is no per-package changelog: release notes for every version live in the single [**CHANGELOG.md**](https://github.com/EdgeFirstAI/hal/blob/main/CHANGELOG.md) in the `hal` repository, which follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Links

- [Changelog](https://github.com/EdgeFirstAI/hal/blob/main/CHANGELOG.md) — release notes for all packages
- [Source](https://github.com/EdgeFirstAI/hal) — the `hal` monorepo
- [Issue tracker](https://github.com/EdgeFirstAI/hal/issues)
- [Package documentation](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/README.md) — the underlying Rust crate
- [EdgeFirst](https://edgefirst.ai)

## License

Apache-2.0
