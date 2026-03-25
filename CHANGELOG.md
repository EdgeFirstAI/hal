# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **YOLO-SEG 2-way split decoder** (`ModelType::YoloSegDet2Way`) — supports
  TFLite INT8 segmentation models with 3 outputs: a combined detection tensor
  `[1, nc+4, N]` (boxes + scores), a separate mask-coefficient tensor
  `[1, 32, N]`, and a prototype-mask tensor `[1, H/4, W/4, 32]`.  The
  builder auto-selects this variant when `Detection` + `MaskCoefficients` +
  `Protos` outputs are provided.  Supported in all decode paths (float,
  quantized, and proto variants).

- **`MaskOverlay` compositing options** for `draw_masks()` and
  `draw_masks_proto()` — new `overlay` parameter with two controls:
  - `background: Option<&TensorDyn>` — when set, copies the background
    image into `dst` before compositing masks, allowing masks to be
    rendered over a separate frame.
  - `opacity: f32` (default `1.0`) — scales the alpha of every rendered
    mask and bounding box color; `0.5` produces semi-transparent overlays.

  Available in Rust, Python (`background=None, opacity=1.0` keyword args
  on `Decoder.draw_masks()` and `ImageProcessor.draw_masks()`), and C
  (`hal_decoder_draw_masks()` gains `background` and `opacity` params).
  All 7 GL fragment shaders updated with `uniform float opacity`.

- **GLES 3.1 context upgrade** — EGL context creation now requests
  GLES 3.1 (compute shaders) with automatic fallback to 3.0.

### Changed

- **`ImageProcessorTrait::draw_masks()` and `draw_masks_proto()`** gain a
  new `overlay: MaskOverlay<'_>` parameter. Implementors must update their
  trait impls. Pass `MaskOverlay::default()` for backward-compatible
  behaviour.

- **C API `hal_decoder_draw_masks()`** gains `background` (`const
  hal_tensor*`, pass `NULL` for none) and `opacity` (`float`, pass `1.0`
  for none) parameters before `out_boxes`. Existing C callers must update.

### Performance

- **Fused dequant+matmul kernel** for `materialize_segmentations()` —
  computes `mask_coeff @ protos` directly from the i8 proto tensor without
  allocating a 3.1 MB f32 copy. Includes fast sigmoid approximation
  (~10× faster than libm `expf`) and 4-way loop unrolling.
  Hybrid path speedup: 1.25–1.34× across all targets.

- **Zero-copy i8 proto extraction** — `extract_proto_data_quant()` uses
  `TypeId` specialization to avoid per-element `as_()` conversion when
  the proto tensor is already `i8`; uses flat `to_owned()` memcpy instead.

- **`SendablePtr` for `ProtoData`** in the GL threaded path — eliminates
  an 819 KB–3.3 MB deep clone per frame.

- **`glTexSubImage3D` fast path** — proto texture dimensions are tracked;
  reuses the existing GL texture object when dimensions match (every frame
  after the first), avoiding driver-side reallocation.

- **Cached opacity uniform** — skips 28 redundant GL state changes per
  frame on the default `opacity=1.0` path.

- **GLES 3.1 compute shader for HWC→CHW proto repack** — opt-in via
  `EDGEFIRST_PROTO_COMPUTE=1`. Uploads HWC i8 data to an SSBO and
  transposes to `GL_TEXTURE_2D_ARRAY` via compute dispatch. 2.2–2.4×
  speedup on the fused GL proto path (imx8mp, imx95).

### Removed

- **`decode_masks` / `decode_masks_atlas` atlas-based mask readback path** —
  removed the entire atlas decode pipeline which was too slow on embedded
  targets (435 ms on imx8mp Vivante GC7000UL). This removes:
  - `hal_decoder_decode_masks()` C API function
  - `Decoder.decode_masks()` Python method and type stub
  - `ImageProcessorTrait::decode_masks_atlas()` Rust trait method
  - `MaskRegion` and `MaskResult` types
  - Three GL logit-threshold shaders (`proto_mask_logit_*`)
  - Mask FBO, PBO, and atlas rendering infrastructure in the GL backend
  - `bench_decode_masks_atlas` benchmark

  Use `draw_masks()` / `draw_masks_proto()` for GPU-accelerated mask
  overlay, or `materialize_segmentations()` + `draw_masks()` for the
  hybrid CPU decode path.

## [0.12.0] - 2026-03-24

### Added

- **Row stride and plane offset for DMA-BUF tensors** — tensors from
  `import_image` now carry per-plane stride and offset metadata, propagated
  end-to-end into EGL `DMA_BUF_PLANE*_PITCH` / `PLANE*_OFFSET` attributes.
  This enables correct GPU rendering of V4L2 buffers with row padding
  (e.g. `bytesperline=2048` on a 1920-wide frame). New APIs:

  | Layer | New APIs |
  |-------|----------|
  | Rust | `set_row_stride()`, `with_row_stride()`, `effective_row_stride()`, `set_plane_offset()`, `plane_offset()` |
  | C | `hal_plane_descriptor_set_stride()`, `hal_plane_descriptor_set_offset()`, `hal_tensor_row_stride()` |
  | Python | `import_image(..., stride=, offset=, chroma_stride=, chroma_offset=)` |

  Changing the pixel format to a different value via `set_format()` clears
  stored stride and offset; re-setting the same format preserves both.
  `Tensor::map()` rejects strided or offset tensors to prevent incorrect
  CPU access on padded buffers.

- **VYUY pixel format** (`PixelFormat::Vyuy`) — packed YUV 4:2:2 with VYUY
  byte order. Available in Rust, C (`HAL_PIXEL_FORMAT_VYUY`), and Python.
  Note: EGL DMA-BUF import produces incorrect output on Vivante (~0.28
  SSIM); the GL backend auto-falls back to the CPU/G2D path for this format.

- **Software OpenGL renderer rejection** — the GL backend detects
  llvmpipe/softpipe/swrast at init and returns `Error::NotSupported`,
  causing the auto-detection fallback chain to select CPU or G2D instead.

- **GL thread panic safety** — the OpenGL worker thread now wraps all
  message handlers in `catch_unwind`. A caught panic sets a `poisoned` flag;
  all subsequent GL calls return `Error::Internal` instead of hanging.

- **C API preprocessing benchmark** (`bench_preproc.c`) — reference for
  GStreamer/V4L2 integrators covering seven DMA-BUF import patterns, a
  format conversion matrix, and a two-stage chained pipeline.

### Removed

- **`hal_tensor_from_planes()`** — use `hal_import_image()` with separate
  `hal_plane_descriptor_new()` descriptors instead. The new API supports
  per-plane stride/offset and is processor-aware (EGL cache compatible).

### Changed

- **Contiguous NV12 UV offset is stride-aware** — the UV plane byte offset
  for single-fd NV12 is now `effective_row_stride() * height` instead of
  `width * height`, fixing corrupted chroma on padded buffers.

- **Multiplane `import_image` rejects unsupported dtypes** — only `U8` and
  `I8` are valid for multiplane NV12 imports; other dtypes return
  `Error::NotSupported` instead of silently producing a wrong-type tensor.

### Fixed

- **NV12 UV sampling incorrect for padded buffers** — V4L2 buffers with
  stride > width caused corrupted chroma because the UV plane offset passed
  to EGL was computed from width, not from the actual stride.

- **`hal_import_image` double-free** — passing the same pointer as both
  `image` and `chroma` caused UB. Now returns `EINVAL`.

## [0.11.0] - 2026-03-22

### Added

- **Zero-copy external buffer import** (`import_image`) across Rust, C,
  and Python APIs. Enables GPU rendering directly into a caller-owned
  DMA-BUF — eliminating the `memcpy` between HAL's output buffer and an
  NPU delegate's input buffer. Supports per-plane stride and offset for
  padded allocators (V4L2, GStreamer). The caller retains ownership of
  the fd; HAL borrows it via `dup(fd)`.

  **Rust:**
  ```rust,ignore
  use edgefirst_tensor::PlaneDescriptor;
  let pd = PlaneDescriptor::new(vx_fd.as_fd())?.with_stride(bytesperline);
  let mut dst = processor.import_image(pd, None, 640, 640, PixelFormat::Rgb, DType::U8)?;
  processor.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::letterbox())?;
  ```

  **Python:**
  ```python
  dst = processor.import_image(vx_fd, 640, 640, ef.PixelFormat.Rgb, stride=bytesperline)
  processor.convert(src, dst)
  ```

  **C:**
  ```c
  struct hal_plane_descriptor *pd = hal_plane_descriptor_new(vx_fd);
  hal_plane_descriptor_set_stride(pd, bytesperline);
  struct hal_tensor *dst = hal_import_image(proc, pd, NULL,
                                             640, 640,
                                             HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8);
  ```

  Returns `Error::NotSupported` if the fd is not DMA-backed (e.g. POSIX
  shm fd), ensuring the zero-copy contract is enforced at the API boundary.

- **Tensor pixel-format metadata** — `TensorDyn::set_format()` and
  `TensorDyn::with_format()` attach a `PixelFormat` to any tensor,
  enabling `convert()` to use raw tensors created via `from_fd()` or
  `Tensor::new()` as image destinations without going through
  `create_image()`. Shape validation ensures the format matches the
  tensor dimensions. Available in Python (`Tensor.set_format()`) and
  C (`hal_tensor_set_format()`).

- **DMA-BUF fd accessors** — `TensorDyn::dmabuf()` (borrows the fd) and
  `TensorDyn::dmabuf_clone()` (returns an owned duplicate) for exporting
  HAL-allocated DMA-BUF buffers to external consumers.
  `Tensor::<T>::dmabuf()` provides the same on typed tensors. Available
  in Python (`Tensor.dmabuf_clone()`) and C (`hal_tensor_dmabuf_clone()`).

  Use case: HAL allocates via `create_image()`, consumer imports the fd:
  ```rust,ignore
  let hal_dst = processor.create_image(640, 640, PixelFormat::Rgb, DType::U8, None)?;
  let fd = hal_dst.dmabuf_clone()?;  // Error if not DMA-backed
  delegate.register_buffer(fd)?;
  ```

- PyPI README (`crates/python/README.md`) with quick-start, zero-copy
  external buffer examples, and platform support matrix

### Removed

- **`hal_image_processor_convert_ref`** — removed from Rust, C header, and
  ARCHITECTURE.md. This function was redundant with the `hal_tensor_set_format()`
  + `hal_image_processor_convert()` two-step and had been a recurring source of
  safety bugs (use-after-free in v0.9, unsound `ptr::read`/`ManuallyDrop` in
  v0.10.0). Callers should use:
  ```c
  hal_tensor_set_format(dst, format);
  hal_image_processor_convert(proc, src, dst, rotation, flip, crop);
  ```

### Fixed

- Corrected errno mapping for `hal_import_image`: `InvalidShape` / `NotAnImage`
  and `Tensor(InvalidArgument/InvalidShape/ShapeMismatch)` map to `EINVAL`
  (not `EIO`); `UnsupportedFormat` / `NotSupported` map to `ENOTSUP`

## [0.10.0] - 2026-03-20

### Changed

- **BREAKING: Unified Tensor API** — `TensorImage` removed from all APIs.
  The `Tensor` type is now the single tensor type across Rust, C, and Python.
  - **Python**: `TensorImage(w, h, fmt)` → `Tensor.image(w, h, fmt)`;
    `TensorImage.load()` → `Tensor.load()`; `FourCC` → `PixelFormat`
  - **C**: `hal_tensor_image_*` functions replaced by `hal_tensor_*` equivalents
    (e.g., `hal_tensor_new_image()`, `hal_tensor_load_image()`,
    `hal_tensor_width()`, `hal_tensor_pixel_format()`);
    `HalTensorImage` removed — use `HalTensor` for all tensors
  - **Rust**: `TensorDyn` is the unified type-erased tensor;
    `PyTensor` and `HalTensor` now wrap `TensorDyn` directly

### Added

- `TensorDyn::new()`, `TensorDyn::from_fd()`, `TensorDyn::size()`,
  `TensorDyn::memory()`, `TensorDyn::reshape()`, `TensorDyn::clone_fd()`
  methods for complete type-erased tensor operations
- `HalDtype::F16` variant in C API for float16 tensor support
- NV12 even-height validation in `Tensor::image()`
- Semi-planar shape constraint validation in `Tensor::set_format()`
- **`dtype` parameter on `create_image()`** across Rust, C, and Python APIs.
  Supports `DType::I8` / `"int8"` for direct signed int8 tensor output;
  PBO-backed images support `u8` and `i8` byte-sized types, other dtypes
  prefer DMA-buf when available and fall back to system memory
- **Int8 GPU shaders** for all GL render paths (DMA, non-DMA, non-DMA
  planar, PBO-to-PBO). The fragment shader applies XOR 0x80 bias on the
  GPU, eliminating the CPU readback + XOR post-pass for ~2× faster int8
  letterbox pipeline on GPU-capable platforms
- Comprehensive `trace`-level logging in the convert dispatch path for
  profiling backend selection and per-stage timing
- **Dual Python wheel builds** (`abi3-py311` and `abi3-py38`). Python 3.11+
  users get zero-copy buffer protocol support; Python 3.8–3.10 users get a
  compatible fallback. Pip automatically selects the best wheel

### Fixed

- EGL image cache evict-before-create causing unnecessary cache churn
  and potential fd accumulation on GREY format fallback
- Use-after-free in `hal_image_processor_convert_ref` (replaced
  `Box::from_raw` with `ptr::read` + `ManuallyDrop`)
- Missing `return Ok(())` in stable `normalize_to_float_16` RGBA path
  that caused double-processing
- Clippy `iflet_redundant_pattern_matching` and `collapsible_match` lints
- Broken `FourCC` imports and variant casing in Python tests
- Stale `.display()` calls in G2D tests and duplicate `.as_u8()` in GL tests
- Int8 letterbox bias on all four GL render paths — `glClear` bypasses the
  fragment shader, so `crop.dst_color` must be XOR'd on the CPU side before
  calling `glClearColor`
- Alpha-channel preservation in CPU and G2D int8 XOR bias — now skips
  alpha bytes for Rgba/Bgra formats, matching GL shader behavior
- GL auto-backend priority: OpenGL is now preferred over G2D when both
  are available
- Pinned Rust toolchain versions in CI to prevent profraw format corruption
  across build/coverage-processing jobs; `--failure-mode=all` for
  `llvm-profdata merge` to skip corrupted profraw files from GPU driver
  crashes

## [0.9.1] - 2026-03-10

### Added

- **BGRA destination format**: `create_image()` and `convert()` now accept
  BGRA as a destination format for Cairo/Wayland compositing (ARGB32 on
  little-endian). Supported natively on OpenGL (via `GL_BGRA`) and G2D
  (`G2D_BGRA8888`); CPU backend uses R/B channel swizzle after RGBA
  conversion. Available in Rust (`PixelFormat::Bgra`), Python (`PixelFormat.Bgra`),
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
