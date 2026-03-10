# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.9.1] - 2026-03-10

### Added

- **BGRA destination format**: `create_image()` and `convert()` now accept
  BGRA as a destination format for Cairo/Wayland compositing (ARGB32 on
  little-endian). Supported natively on OpenGL (via `GL_BGRA`) and G2D
  (`G2D_BGRA8888`); CPU backend uses R/B channel swizzle after RGBA
  conversion. Available in Rust (`BGRA` constant), Python (`FourCC.BGRA`),
  and C (fourcc `"BGRA"`). `draw_masks()` and `draw_masks_proto()` support
  BGRA destination images on the OpenGL backend; CPU mask rendering
  accepts only RGBA/RGB destinations.

- **`EDGEFIRST_FORCE_BACKEND` environment variable**: forces `ImageProcessor`
  to initialize only the specified backend (`cpu`, `g2d`, or `opengl`) with
  no fallback chain. Useful for benchmarking individual backends in isolation.

- **Hybrid CPU+GL mask rendering path**: `draw_masks_proto()` now materializes
  segmentations on CPU via `materialize_segmentations()` then composites via
  OpenGL, which is 2.5–27× faster than the previous full-GPU path across
  tested platforms (imx8mp, imx95, rpi5, x86).

- **`CPUProcessor::materialize_segmentations()`**: new public method that
  computes per-detection segmentation masks from raw prototype data without
  rendering them onto an image.

- **Platform benchmark infrastructure**: migrated all benchmarks from Divan to
  `edgefirst-bench` with JSON output support (`BenchSuite`), added dedicated
  benchmark binaries (`tensor_benchmark`, `pipeline_benchmark`,
  `mask_benchmark`, `decoder_benchmark`, `opencv_benchmark`), and added
  benchmark data collection scripts and `BENCHMARKS.md` results document.

- **Test coverage for new APIs**: 21 new tests covering NORM_LIMIT regression
  (protobox), `materialize_segmentations`, `EDGEFIRST_FORCE_BACKEND` env var,
  hybrid mask path error handling, `decode_*_proto` functions, and GL smoke
  tests.

### Fixed

- **YOLO NORM_LIMIT too restrictive**: relaxed `NORM_LIMIT` from 1.01 to 2.0
  in `protobox()` to allow YOLO models that legitimately predict bounding box
  coordinates slightly > 1.0 for objects near frame edges. Coordinates > 2.0
  still return `InvalidShape` to catch un-normalized pixel-space boxes.

### Changed

- ARCHITECTURE.md updated to v2.6: documents hybrid mask path, benchmark
  infrastructure, and `EDGEFIRST_FORCE_BACKEND` env var
- README.md updated with benchmarking section and quick-reference commands

## [0.9.0] - 2026-03-04

### Added

- **`hal_decoder_draw_masks()` C API function**: fused decode+render path
  that mirrors the Python `Decoder.draw_masks()` binding — takes decoder,
  processor, raw model outputs, and destination image; returns rendered overlay
  and detection boxes in a single call. Internally selects proto-based GPU
  matmul path for seg models, with automatic fallback to decoded-mask rendering
  for detection-only models.

- **`hal_decoder_decode_masks()` C API function**: decodes segmentation model
  outputs and returns per-detection pixel masks at the specified output
  resolution. Mirrors the Python `Decoder.decode_masks()` binding.

- **PBO (Pixel Buffer Object) tensor backend** (`edgefirst-tensor`):
  `PboTensor<T>` provides GPU-native buffer storage for platforms where DMA-buf
  is unavailable (e.g. NVIDIA desktop GPUs). PBO tensors are managed by the
  OpenGL thread via a `WeakSender` channel that allows clean shutdown without
  blocking on orphaned tensors.

- **`ImageProcessor::create_image()`** factory method that probes GPU
  capabilities at initialization and selects the optimal memory backend:
  DMA-buf > PBO > heap memory. This is now the preferred way to allocate
  images for use with `convert()`. Available in Rust, Python
  (`ImageProcessor.create_image()`), and C
  (`hal_image_processor_create_image()`).

- **PBO convert paths** for all source/destination combinations:
  `convert_pbo_to_pbo()` (both PBO), `convert_any_to_pbo()` (Mem/DMA source
  to PBO destination), and `convert_pbo_to_mem()` (PBO source to Mem
  destination). All paths use direct GL buffer bindings
  (`GL_PIXEL_UNPACK_BUFFER` / `GL_PIXEL_PACK_BUFFER`) to avoid the deadlock
  that would occur if the GL thread called `tensor.map()` on a PBO tensor.

- **`hal_image_processor_create_image()`** C API function with documentation,
  null-safety checks, and errno-based error reporting

- **Python `ImageProcessor.create_image()`** method with format parameter
  defaulting to RGBA

- Python tests: `test_create_image`, `test_create_image_formats`,
  `test_create_image_convert`, `test_create_image_roundtrip`

- C API tests: `test_image_processor_create_image`,
  `test_image_processor_create_image_null_params`,
  `test_image_processor_create_image_convert`

- Rust integration test: `test_convert_pbo_to_pbo` exercising PBO-to-PBO
  conversion with SSIM comparison against CPU reference

### Fixed

- **GL thread shutdown hang**: `GlPboOps` used a strong `Sender` clone that
  kept the GL thread's message channel alive after `GLProcessorThreaded` was
  dropped, causing `handle.join()` to block indefinitely. Changed to
  `WeakSender` so the channel closes when the last `ImageProcessor` reference
  is dropped.

- **PBO-to-PBO deadlock**: `convert_pbo_to_pbo()` called `draw_src_texture()`
  which invoked `tensor.map()` on the GL thread, sending a message back to
  itself. Added `draw_src_texture_from_pbo()` that binds the source PBO as
  `GL_PIXEL_UNPACK_BUFFER` with a NULL `glTexImage2D` pointer for zero-copy
  upload.

- **Mixed PBO/Mem deadlock**: `convert_dest_non_dma()` called
  `dst.tensor().map()` on the GL thread for PBO destinations. Added dedicated
  `convert_any_to_pbo()` and `convert_pbo_to_mem()` methods that use GL buffer
  bindings instead of tensor mapping.

### Changed

- **Mask API renamed for clarity** — API names now use consistent verbs:
  `decode` = get mask data back, `draw` = overlay onto image. The compound
  names that accumulated during iterative optimization have been replaced
  with shorter, intent-revealing names across all language bindings:

  **Renamed methods:**

  | Layer | Old Name | New Name |
  |-------|----------|----------|
  | Rust trait | `render_from_protos()` | `draw_masks_proto()` |
  | Rust trait | `render_to_image()` | `draw_masks()` |
  | Rust trait | `render_masks_from_protos()` | `decode_masks_atlas()` |
  | Rust enum | `ImageRenderProtos` | `DrawMasksProto` |
  | Rust enum | `ImageRender` | `DrawMasks` |
  | Rust enum | `RenderMasksFromProtos` | `DecodeMasksAtlas` |
  | Python `Decoder` | `decode_and_render()` | `draw_masks()` |
  | Python `ImageProcessor` | `render_to_image()` | `draw_masks()` |
  | C API | `hal_image_processor_render_to_image()` | `hal_image_processor_draw_masks()` |

  **New methods (no previous equivalent):**

  | Layer | New Name |
  |-------|----------|
  | Python `Decoder` | `decode_masks()` |
  | C API | `hal_decoder_draw_masks()` |
  | C API | `hal_decoder_decode_masks()` |

  **Migration guide:**
  - **Python users**: rename `decoder.decode_and_render(...)` →
    `decoder.draw_masks(...)` and `processor.render_to_image(...)` →
    `processor.draw_masks(...)`; `decoder.decode_masks(...)` is new
  - **C users**: rename `hal_image_processor_render_to_image()` →
    `hal_image_processor_draw_masks()`; `hal_decoder_draw_masks()` and
    `hal_decoder_decode_masks()` are new
  - **Rust users**: rename `render_from_protos()` → `draw_masks_proto()`,
    `render_to_image()` → `draw_masks()`, and
    `render_masks_from_protos()` → `decode_masks_atlas()` on `ImageProcessorTrait`
    implementors

- **`Decoder.decode_masks()` return type changed** — now returns individual
  per-detection masks as `List[ndarray]` (each shape `(H, W)`, uint8) instead
  of the raw atlas tuple. The atlas packing is now an internal optimization.

- ARCHITECTURE.md updated to v2.4: documents PBO tensor architecture,
  `create_image()` backend selection, PBO convert dispatch table, and
  `WeakSender` shutdown design

- README.md updated: `create_image()` presented as preferred image allocation
  method with usage guidance for Rust, Python, and C; DMA-buf permission
  requirements; platform GPU support table

## [0.8.0] - 2026-02-24

### Added

- GPU segmentation mask atlas rendering: `decode_masks()` Python binding and
  `render_mask_atlas()` on `ImageProcessorTrait` with GL, CPU, and G2D (stub)
  backends — renders all masks in a single GPU pass with one PBO readback,
  eliminating CPU mask computation and per-mask GL resize roundtrips
- EGL display probe and override API: `probe_egl_displays()`, `EglDisplayKind` enum,
  `EglDisplayInfo` struct, and `ImageProcessor::with_config()` constructor in Rust;
  `EglDisplayKind`, `EglDisplayInfo`, `probe_egl_displays()`, and
  `ImageProcessor(egl_display=...)` in Python — enables selecting or avoiding specific
  EGL display types on problematic hardware (e.g. Vivante GBM on i.MX8)
- Python API for programmatic decoder configuration (EDGEAI-774): `DecoderType`,
  `DecoderVersion`, `DimName` enums, `Output` class with factory methods
  (`detection`, `boxes`, `scores`, `protos`, `segmentation`, `mask_coefficients`,
  `mask`), and `Decoder.new_from_outputs()` constructor
- Decoder builder API with programmatic output configuration and C API refactoring
  to opaque `hal_decoder_params` pattern with setter functions
- `yolov8` serde alias for `DecoderType::Ultralytics` so old model metadata
  with `decoder: yolov8` deserializes without migration
- Dict-format `dshape` deserialization in decoder configs: accepts both
  array-of-single-key-dicts (`[{"batch": 1}, ...]`) and array-of-tuples formats
- Segmentation pipeline comparison script (`example_seg_pipeline.py`): end-to-end
  OpenCV vs HAL decode vs HAL fused benchmark for YOLOv8-seg INT8 TFLite

### Fixed

- NV16 handling in `TensorImage::new()`, `width()`, `height()`, `channels()`:
  constructor created wrong 3D shape instead of 2D `[H*2, W]`, and accessors
  only special-cased NV12, causing index-out-of-bounds panics
- NV12/NV16 in `TensorImageRef::from_borrowed_tensor`: expected all formats to
  have 3D tensor shapes, causing index-out-of-bounds panic for semi-planar
  formats via the borrowed reference path (`hal_image_processor_convert_ref`)

## [0.7.0] - 2026-02-17

### Added

- NV12/NV16 2D tensor support: `TensorImage::from_tensor()` now accepts 2D
  tensors with shapes [H*3/2, W] for NV12 and [H*2, W] for NV16, with
  shape validation and format-specific documentation
- C API: `hal_tensor_image_from_tensor()` with dtype/ndim validation,
  `hal_tensor_image_map_create()`, and `HalTensor::ndim()` helper
- C API: pkg-config (`edgefirst-hal.pc`) and SOVERSION support with standard
  GNU/Linux versioned shared library symlinks in release archives
- Shared DRM render node file descriptor (via `OnceLock`) across all DMA-buf
  PRIME imports, avoiding deadlocks on Vivante with concurrent V4L2/VPU usage
- EGL/GL shutdown cleanup architecture documentation in ARCHITECTURE.md with
  root cause analysis and industry cross-references

### Fixed

- EGL/GL cleanup crashes during process shutdown: defense-in-depth strategy
  using `Box::leak` (prevent `dlclose`), `ManuallyDrop` (skip
  `eglReleaseThread`), `catch_unwind` (catch panics from invalidated function
  pointers), and omitted `eglTerminate` (avoid Vivante double-free)
- OpenGL EGL initialization: reorder GBM/DRM probe before default display to
  avoid blocking when Wayland compositor is not running
- Test suite: DMA permission guards for rootless CI, Python test runner
  migration to pytest, pre-commit hook enforcement
- C API test failures after SOVERSION change
- Decoder: remove incorrect `#[serde(flatten)]` on `Protos.quantization`

## [0.6.2] - 2026-02-13

### Added

- `TensorImage.from_fd()` Python method to create tensor images from
  file descriptors (Linux only), enabling zero-copy DMA-buf sharing
  with external processes
- Linux aarch64 (ARM64) Python wheels built natively on `ubuntu-22.04-arm`
  with `manylinux2014` compatibility, published to PyPI and GitHub Releases

### Changed

- PyO3 stable ABI is now opt-in via `abi3-py311` and `abi3-py38` Cargo
  features instead of being the default; CI workflows pass
  `--features abi3-py311` to build portable wheels for Python 3.11+
  while local development builds target the exact installed Python version

## [0.6.1] - 2026-02-13

### Added

- Linux aarch64 (ARM64) Python wheels built natively on `ubuntu-22.04-arm`
  with `manylinux2014` compatibility, published to PyPI and GitHub Releases

### Fixed

- Enable PyO3 stable ABI (`abi3-py311`) so wheels are recognized as compatible
  binary distributions by pip on Python 3.11+ instead of falling back to
  source builds

## [0.6.0] - 2026-02-13

### Changed

- Migrate `g2d-sys` from local crate to published `g2d-sys 1.2.0` from crates.io
  (now maintained at [github.com/EdgeFirstAI/g2d-rs](https://github.com/EdgeFirstAI/g2d-rs))
- G2D `convert` now calls `g2d.finish()` once after queuing all operations
  (clear + blit), reducing GPU synchronization barriers from 2 to 1 per frame
- Letterbox clear is now controlled by the caller via `Crop::dst_color`:
  `Some(color)` clears the destination with the specified color before blit,
  `None` skips the clear (preserving whatever is in the destination buffer,
  enabling custom backgrounds or caller-managed clearing)
- Always use hardware `g2d_clear` for supported pixel formats (2-byte and
  4-byte formats); CPU fallback only for RGB888 which G2D does not support
- Move benchmarks from `test.yml` to dedicated `benchmark.yml` workflow
  triggered on-demand via `workflow_dispatch` with QuickChart result charts

### Fixed

- Implement DRM PRIME attachment for DMA-buf cache coherency — `DMA_BUF_IOCTL_SYNC`
  was previously a no-op on cached CMA heaps because no `dma_buf_attach` existed.
  Each `DmaTensor` now creates a persistent DRM attachment via
  `DRM_IOCTL_PRIME_FD_TO_HANDLE` to enable proper cache invalidation and flushing

### Added

- `DrmAttachment` struct in `edgefirst-tensor` for persistent DRM PRIME imports
- Realistic letterbox pipeline benchmarks (`bench_letterbox_pipeline`) comparing
  CPU vs G2D vs OpenGL for 1080p/4K YUYV/NV12 → 640×640 RGBA letterbox resize
- `CPUProcessor::fill_image_outside_crop` is now public

## [0.5.2] - 2026-02-12

### Changed

- Consolidate all scripts to `.github/scripts/` for CI/CD automation
- Migrate `AGENTS.md` to `.github/copilot-instructions.md` (GitHub Copilot standard location)
- Add detailed release process documentation with version verification steps
- Update commit message guidelines: JIRA keys not required for housekeeping tasks

## [0.5.1] - 2026-02-11

### Changed

- GitHub release workflow improvements for trusted publishing

## [0.5.0] - 2026-02-09

First published release of EdgeFirst HAL to [crates.io](https://crates.io) and [PyPI](https://pypi.org).

### Core Crates

- **`edgefirst-hal`** - Main HAL crate re-exporting tensor, image, and decoder functionality
- **`edgefirst-tensor`** - Zero-copy tensor memory management with DMA-heap, shared memory,
  and heap allocation backends
- **`edgefirst-image`** - High-performance image processing with hardware-accelerated conversion
  and resizing (G2D, OpenGL ES, CPU fallback)
- **`edgefirst-decoder`** - ML model output decoder for YOLO (v5/v8/v11/v26) and ModelPack
  object detection and instance segmentation with NMS
- **`edgefirst-tracker`** - ByteTrack multi-object tracking with Kalman filtering
- **`g2d-sys`** - Low-level FFI bindings for NXP i.MX G2D 2D graphics acceleration

### Added

- **Tensor Memory Management** (`edgefirst-tensor`):
  - Zero-copy memory buffers with DMA-heap, shared memory, and heap backends
  - `TensorImage` for loading JPEG/PNG images into tensor-backed memory
  - DMA buffer synchronization controls and file descriptor management
  - `is_dma_available()` public API for runtime capability detection
  - ndarray integration (optional, enabled by default)
  - Cross-platform support: Linux DMA/SHM, macOS/Windows heap fallback

- **Image Processing** (`edgefirst-image`):
  - Format conversion: YUYV, NV12, NV16, RGB, RGBA, GREY, Planar RGB/RGBA (8BPS)
  - Hardware-accelerated resize via NXP G2D and OpenGL ES 3.0
  - CPU fallback converter for all platforms
  - Source and destination crop with letterboxing and fill color
  - Rotation (0/90/180/270) and horizontal/vertical flip
  - EXIF-aware automatic image orientation
  - Normalization modes: signed (-1..1), unsigned (0..1), raw, with f16 support
  - Multi-threaded normalization with rayon
  - Automatic converter selection: tries all available backends (G2D, OpenGL, CPU)
  - OpenGL renders to texture with dedicated thread for stability
  - GBM dynamic loading via `edgefirst-gbm` for EGL surface management
  - Segmentation mask and bounding box overlay rendering (OpenGL, G2D, CPU)

- **Model Decoder** (`edgefirst-decoder`):
  - YOLO v5/v8/v11 object detection decoding with class-aware and class-agnostic NMS
  - YOLO v26 detection support with architecture version disambiguation
  - YOLO instance segmentation with quantized mask decoding (f32, i8, i32)
  - ModelPack detection and segmentation decoding
  - Split decoder for models with separate box and class outputs
  - End-to-end model support (models with built-in NMS)
  - Configuration via YAML/JSON files and EdgeFirst Model Metadata format
  - Configurable score threshold, IoU threshold, and NMS mode
  - Dequantization support for int8/uint8/int32 quantized model outputs
  - Generic decode API with automatic model type and architecture detection

- **Multi-Object Tracker** (`edgefirst-tracker`):
  - ByteTrack algorithm with Kalman filtering
  - Configurable track birth/death thresholds
  - IoU-based association with LAPJV assignment

- **Python Bindings** (`edgefirst-hal` on PyPI):
  - PyO3-based Python API with numpy and f16 integration
  - `TensorImage` for image loading, format query, and tensor memory control
  - `ImageProcessor` for format conversion, resize, crop, and normalization
  - `Decoder` for YOLO/ModelPack model output post-processing
  - Config creation from Python dictionaries via pythonize
  - `PyArrayLike` inputs (accepts both numpy arrays and Python lists)
  - File descriptor management (`from_fd`, `fd` property)
  - Destination crop and fill color controls
  - Python type stubs (`.pyi`) for IDE autocompletion
  - Support for Python 3.8+ (ABI3) and 3.11+ (ABI3-311)

- **NXP G2D Bindings** (`g2d-sys`):
  - FFI bindings for NXP i.MX G2D hardware 2D acceleration
  - Dynamic loading via `libloading` for runtime availability detection
  - Version-aware API support (legacy and modern G2D, imx95)

- **Platform Support**:
  - Linux x86_64 and aarch64 with full hardware acceleration
  - macOS (Apple Silicon and x86_64) with CPU-only processing
  - Windows with heap memory tensors and CPU processing
  - NXP i.MX8/i.MX9 with G2D and OpenGL ES hardware acceleration

- **CI/CD Infrastructure**:
  - GitHub Actions workflows for testing, coverage, and releases
  - Multi-platform CI: x86_64, aarch64, macOS ARM
  - On-target hardware testing with NXP i.MX boards
  - Multi-platform Python wheel builds (Linux manylinux2014, macOS, Windows)
  - PyPI trusted publishing with OIDC
  - SBOM generation and license compliance checking
  - Comprehensive Criterion and Divan benchmark suites
  - Shared benchmark module with common test utilities

- **Publishing**:
  - All workspace crates published to crates.io
  - `edgefirst-gbm` / `edgefirst-gbm-sys` published as standalone crates for GBM
    dynamic loading support (fork of Smithay/gbm.rs)
  - Comprehensive crate documentation with README, keywords, and categories
  - API documentation with doc-tests for all public interfaces

### Notes

- Python bindings are distributed via PyPI only (`publish = false` for the Python crate)
- `g2d-sys` maintains its own version (1.0.1) independent of the workspace
- `edgefirst-gbm` 0.18.1 and `edgefirst-gbm-sys` 0.4.1 are published separately
  from the [EdgeFirstAI/gbm.rs](https://github.com/EdgeFirstAI/gbm.rs) repository
- Apache-2.0 license with NOTICE file for third-party attributions
