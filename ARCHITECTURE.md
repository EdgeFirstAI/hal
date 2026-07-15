# EdgeFirst HAL — Architecture

This document is the **cross-crate** architecture story for the EdgeFirst
HAL workspace. It covers the design patterns shared across crates, the
performance-tracing infrastructure, the cross-cutting story behind
DMA-BUF identity and tensor caching, and the source-code organization.
Per-crate architecture detail (class diagrams, internal layouts,
backend-specific algorithms, lifecycle quirks) lives in each
sub-crate's `ARCHITECTURE.md`:

| Crate | Per-crate architecture |
|-------|------------------------|
| `tensor` | [crates/tensor/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/ARCHITECTURE.md) — backend dispatch (DMA-BUF/IOSurface/AHardwareBuffer/SHM/Mem/PBO), multi-plane DMA-BUF, BufferIdentity, CpuAccess + compression metadata |
| `codec` | [crates/codec/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec/ARCHITECTURE.md) — custom baseline JPEG decoder, SIMD dispatch (NEON/SSE4.1/SSSE3/SSE2), zero-allocation scratch model, strided/EXIF-rotated output |
| `image` | [crates/image/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md) — unified GL engine (one `GLProcessorST`; Linux DMA-BUF, macOS/iOS ANGLE + IOSurface, Android AHardwareBuffer), `GlPlatform` porting seam, EGL image cache, batch engine (`convert_deferred`/`flush`), G2D, CPU, Vivante workarounds, shutdown safety |
| `decoder` | [crates/decoder/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/ARCHITECTURE.md) — model-type selection, dshape contract, per-scale framework, fused proto path |
| `tracker` | [crates/tracker/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/ARCHITECTURE.md) — ByteTrack two-pass association, Kalman state |
| `hal` (umbrella) | [crates/hal/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/hal/ARCHITECTURE.md) — re-export layer + tracing subscriber |
| `capi` (C API) | [crates/capi/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/ARCHITECTURE.md) — opaque-handle ABI, performance recommendations, Delegate DMA-BUF framework, mobile zero-copy surface |
| `python` | [crates/python/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/python/ARCHITECTURE.md) — PyO3 bindings, numpy 3-path copy strategy, abi3 wheels |

The high-level system diagram lives at the top of
[README.md § System Architecture](https://github.com/EdgeFirstAI/hal/blob/main/README.md#system-architecture);
this document does not reproduce it.

---

## Per-Crate Summary

Each sub-crate has a single responsibility in the inference pipeline:

- [`edgefirst-tensor`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/) — the foundation. Provides `Tensor<T>` and `TensorDyn` with interchangeable backends — the `TensorMemory::Dma` zero-copy slot maps to DMA-BUF on Linux, IOSurface on macOS/iOS, and AHardwareBuffer on Android, alongside SHM / Mem / PBO — plus multi-plane composition for V4L2 NV12M, the `BufferIdentity` cache key (interned on `AHardwareBuffer_getId` on Android), the required `CpuAccess` declaration and tile-compression metadata on image tensors, and the `PboOps` trait that lets the GL backend manage PBO lifetimes through a `WeakSender` channel.
- [`edgefirst-codec`](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec/) — Image decoding (JPEG, PNG) into pre-allocated tensor buffers with support for u8, u16, i8, i16, and f32 pixel types. Supports strided output for GPU pitch-aligned DMA-BUF/PBO tensors. Designed for the allocate-once, decode-in-loop pattern.
- [`edgefirst-image`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/) — the GPU/G2D/CPU image processor. Owns the GL thread, EGL image caches, and shutdown defense layers. Provides format conversion, geometric transforms, and three mask-rendering pipelines (materialized, fused proto, tracked). The GL backend is a **single engine** (`GLProcessorST`) that runs on every supported OS: Linux uses native EGL + DMA-BUF import, macOS uses ANGLE + IOSurface, Android uses native EGL + AHardwareBuffer EGLImage import (iOS builds ride the ANGLE platform) — platform differences are confined to the `GlPlatform` compile-time porting contract (`gl/platform/`). Batch preprocessing is supported via `convert_deferred`/`flush`: sibling tiles share one EGLImage import (parent-keyed) and one GPU sync per batch.
- [`edgefirst-decoder`](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/) — model output post-processing. YOLOv5/v8/v11/v26 (incl. end-to-end) and ModelPack. NEON-optimized per-scale split-tensor framework. Validates `shape` / `dshape` declarations against the physical-memory-order contract at builder time.
- [`edgefirst-tracker`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/) — ByteTrack with Kalman-smoothed trajectories. Generic over the detection box type; the decoder's `DetectBox` plugs in via the `DetectionBox` trait.
- [`edgefirst-hal`](https://github.com/EdgeFirstAI/hal/blob/main/crates/hal/) — umbrella crate. Re-exports the five functional crates and owns the optional Chrome JSON tracing subscriber.
- [`edgefirst-hal-capi`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/) — C ABI layer with cbindgen-generated header. Defines the [Delegate DMA-BUF framework](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/ARCHITECTURE.md#delegate-dma-buf-framework) ABI used by NXP Neutron, VxDelegate, and other TFLite delegates.
- [`crates/python`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python/) — PyO3 bindings, published as `edgefirst-hal` on PyPI. Contains the three-path numpy copy dispatcher.

The internal dependency graph and external dependency list live in
[README.md § Dependencies](https://github.com/EdgeFirstAI/hal/blob/main/README.md#dependencies).

---

## Platform Support Matrix

The HAL spans embedded Linux, desktop Linux, macOS/iOS, and Android,
with different acceleration primitives per tier. The `TensorMemory`
enum is shared across all tiers (same discriminants over the C ABI);
the underlying storage and the GL transfer backend differ.

| Capability | Embedded Linux (i.MX, RPi5, Jetson) | Desktop Linux (x86_64) | macOS (Apple Silicon) | Android (API 26+) |
|------------|--------------------------------------|------------------------|------------------------|--------------------|
| `TensorMemory::Mem` | Heap | Heap | Heap | Heap |
| `TensorMemory::Shm` | `shm_open` | `shm_open` | `shm_open` | `shm_open` |
| `TensorMemory::Dma` | DMA-BUF heap (`/dev/dma_heap/*`) | DMA-BUF heap if mountable; PBO otherwise | IOSurface (CoreFoundation framework) | AHardwareBuffer (NDK, gralloc) |
| `TensorMemory::Pbo` | GLES PBO | GLES PBO | — (no PBO on the macOS backend) | — (AHB covers the zero-copy roles) |
| GL transfer backend | `TransferBackend::DmaBuf` (Vivante, Mali, V3D) | `DmaBuf` or `Pbo` (NVIDIA discrete uses `Pbo`) | `IOSurface` via ANGLE | AHardwareBuffer EGLImage (native EGL) |
| GL → backend translation | Native EGL → driver (vendor blob or Mesa) | Native EGL → driver | ANGLE EGL → Metal | Native EGL → driver (Adreno/Mali/PowerVR/Xclipse) |
| Hardware 2D blitter | G2D on NXP i.MX | — | — | — |
| Zero-copy import API | `EGL_EXT_image_dma_buf_import` | Same, when available | `EGL_ANGLE_iosurface_client_buffer` | `EGL_ANDROID_image_native_buffer` |
| Cross-process buffer handle | DMA-BUF fd (over `SCM_RIGHTS`) | Same | IOSurfaceID (`u32` via Mach port or XPC) | `AHardwareBuffer` (Binder / `sendHandleToUnixSocket`) |
| Probe function | `is_dma_available()` | Same | `is_iosurface_available()` | `is_ahardwarebuffer_available()` |
| Portable probe | `is_gpu_buffer_available()` — works on all four | | | |

The portable `is_gpu_buffer_available()` is the recommended cross-platform
gate when the question is "can I ask for `TensorMemory::Dma` and expect a
zero-copy GPU-importable buffer?" The platform-specific probes
(`is_dma_available`, `is_iosurface_available`) remain when callers need
to know *which* primitive is in use — e.g. to decide whether to call
`hal_tensor_clone_fd` (Linux) vs `hal_tensor_iosurface_id` (macOS).

**iOS (16+)** shares the macOS column's architecture — ANGLE (EGL→Metal)
via the prebuilt xcframeworks and IOSurface-backed `TensorMemory::Dma`
tensors. CI builds and link-validates `aarch64-apple-ios` and
`aarch64-apple-ios-sim` on every PR — `scripts/validate-ios-link.sh`
links the production C API staticlib against the ANGLE + Apple
frameworks; runtime execution on-device awaits the Swift app-shell
effort.

**Image allocation on every tier declares CPU access.** `Tensor::image`
/ `ImageProcessor::create_image` take a required `CpuAccess`
(`None`/`Read`/`Write`/`ReadWrite`): hardware access is implied, CPU
mapping is the opt-in, and `None` keeps Android allocations eligible for
vendor tile compression. The `ImageDesc` builder additionally requests
compression metadata (`Compression::Any` / a specific scheme). This is a
cross-crate contract (tensor → image → capi → python); the normative
description lives in
[crates/tensor/ARCHITECTURE.md § CPU access declaration](crates/tensor/ARCHITECTURE.md).

### Float preprocessing capability

`ImageProcessor::supported_render_dtypes()` returns a `RenderDtypeSupport
{ f16, f32 }` struct after probing the GPU's float color-buffer extensions
at construction time. Use it once at startup to decide which destination
dtype to request; `convert()` always succeeds (GPU or CPU fallback).

**Per-platform capability**

| Platform / GPU | F16 | F32 |
|----------------|-----|-----|
| V3D / Broadcom (RPi 5) | PBO readback + zero-copy DMA-BUF (`DRM_FORMAT_ABGR16161616F`) | PBO readback |
| Mali-G310 / Panfrost (i.MX 95) | PBO readback + zero-copy DMA-BUF (`DRM_FORMAT_ABGR16161616F`) | PBO readback |
| Vivante GC7000UL (i.MX 8M Plus) | **Disabled → CPU fallback** (float readback 170–320 ms) | **Disabled → CPU fallback** |
| Tegra Orin / NVIDIA (orin-nano) | PBO → host buffer; **PBO → CUDA device ptr (zero-copy, implemented)** | PBO → host buffer; **PBO → CUDA device ptr (zero-copy, implemented)** — `cuda_map()` registers the PBO with CUDA on the GL worker thread; the device pointer is usable from any thread via the per-device CUDA primary context |
| macOS ANGLE (RGBA16F IOSurface) | F16 `PlanarRgb` zero-copy IOSurface | Not supported (ANGLE rejects `(GL_FLOAT, *)`) |
| CPU fallback | Always present — never errors | Always present — never errors |

**Data layout produced by the GPU paths**

| DType / layout | GL render target | Tensor shape |
|----------------|-----------------|--------------|
| F16 NCHW `PlanarRgb` | RGBA16F-packed `(W/4, 3H)` — four contiguous f16 planar elements per RGBA16F pixel | `[3, H, W]` f16 |
| F32 NHWC `Rgb` | R32F-wide `(W×3, H)` — one f32 per R channel | `[H, W, 3]` f32 |

**Key constraints**

- Source must be `Rgba` for the GPU float path; other sources fall back to CPU.
- F32 DMA-BUF is impossible (no 32-bit-float DRM fourcc); `create_image(memory: Some(Dma), dtype: F32)` returns `NotSupported`.
- F16 packing requires `W % 4 == 0` (validated at allocation; non-multiples return `InvalidShape`).
- Rotation or flip with a float destination falls back to CPU.
- Normalization is `[0, 1]` only; per-channel mean/std is a future item.
- CPU fallback widens after a u8-precision resize.

**Consumer contract**

```rust,no_run
# use edgefirst_image::{ImageProcessor, ImageProcessorTrait, Rotation, Flip, Crop};
# use edgefirst_tensor::{PixelFormat, DType};
# fn main() -> Result<(), edgefirst_image::Error> {
let proc = ImageProcessor::new()?;
let support = proc.supported_render_dtypes();
// Pick the best float dtype; fall back to U8 if the GPU cannot render floats.
let dst_dtype = if support.f16 { DType::F16 } else if support.f32 { DType::F32 } else { DType::U8 };
// memory: None → auto-selects float PBO when supported, else heap.
// convert() always succeeds; GPU path used when available, CPU otherwise.
let mut dst = proc.create_image(640, 640, PixelFormat::PlanarRgb, dst_dtype, None)?;
# Ok(())
# }
```

---

## Zero-copy CUDA Tensor Mapping

This section describes the cross-crate mechanism that lets the float PBO
produced by `ImageProcessor::convert()` reach a CUDA/TensorRT consumer
with no host round-trip. The per-crate detail (type model, handle lifetimes,
drop order) lives in
[`crates/tensor/ARCHITECTURE.md § Zero-copy CUDA tensor mapping`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/ARCHITECTURE.md#zero-copy-cuda-tensor-mapping);
this section covers the cross-crate data flow and the platform constraints.

### Data flow: FBO → PBO → CUDA → TensorRT

```
ImageProcessor::convert()
│
│  GL worker thread
│  ┌──────────────────────────────────────────────────────────┐
│  │  FBO render (resize / letterbox / colorspace / dtype)    │
│  │       ↓ glReadPixels into GL_PIXEL_PACK_BUFFER          │
│  │  PBO (linear f16 NCHW or f32 NHWC in GPU memory)         │
│  │       ↓ cudaGraphicsGLRegisterBuffer (once at alloc)     │
│  │       ↓ cudaGraphicsMapResources (per cuda_map() call)   │
│  │  CUDA device pointer (primary context, thread-usable)    │
│  └──────────────────────────────────────────────────────────┘
│
│  Caller thread (any thread)
│  ┌────────────────────────────────────────────┐
│  │  CudaMap guard exposes device_ptr() / len() │
│  │  TensorRT enqueue_v3() reads device memory  │
│  │  Drop CudaMap → cudaGraphicsUnmapResources  │
│  │    (PBO released; next convert() can write) │
│  └────────────────────────────────────────────┘
```

`convert()` renders into an FBO and reads out via `glReadPixels` into a
`GL_PIXEL_PACK_BUFFER` (PBO). Because the PBO is registered with CUDA via
`cudaGraphicsGLRegisterBuffer`, mapping it with `cudaGraphicsMapResources`
yields a contiguous linear device pointer that TensorRT's
`IExecutionContext::enqueue_v3` (or equivalent) can consume directly.

### GL-thread constraint

`cudaGraphicsGLRegisterBuffer` and `cudaGraphicsMapResources` must be called
from the **same thread that owns the OpenGL context** — the GL worker thread
inside `GLProcessorThreaded`. The resulting device pointer is, however,
usable from any thread via the per-device CUDA primary context (CUDA's
cross-thread sharing model). The RAII `CudaMap` guard is `Send`, so the
inference thread can hold it while the GL thread proceeds with other work.

### Aliasing rule

GL must not write into a PBO while CUDA has it mapped. The aliasing rule is a
caller convention enforced by the scoped `CudaMap` guard lifetime: the caller
maps per inference and must drop the guard before the next `convert()` call
writes into the same PBO. `cuda_map()` fast-fails to `None` when CUDA is
unavailable for the tensor (no handle attached or `libcudart` absent); it does
not track currently-active maps. Violating the drop-before-convert ordering is
the standard undefined-behavior hazard in CUDA–GL interop.

### DMA-BUF import path

For tensors backed by a DMA-BUF fd (e.g. from a V4L2 capture buffer),
CUDA can import the buffer directly via `cudaImportExternalMemory` with
`cudaExternalMemoryHandleTypeOpaqueFd`. This path is independent of the
GL thread: the DMA-BUF fd is `dup`'d before being handed to CUDA (CUDA
takes ownership of the dup'd fd on success), and the resulting
`CudaExternalMemory` handle yields a persistent device pointer without
a per-map round-trip.

### Runtime loading (dlopen)

CUDA support is loaded at runtime via `dlopen("libcudart.so")` using a
per-process `OnceLock` symbol table. There is no link-time dependency on
`libcudart` and no compile-time feature gate — consistent with the HAL's
dlopen/ioctl approach for other optional platform capabilities. On a host
without `libcudart`, `is_cuda_available()` returns `false` and all
`cuda_map()` calls return `None` immediately.

### Drop order

Within a `PboTensor`'s lifetime, the CUDA handle is dropped before the
PBO storage: `cudaGraphicsUnregisterResource` fires in the handle's
`Drop` impl, and `glDeleteBuffers` fires in the PBO's `Drop` impl.
Reversing this order would dereference freed GL state from the CUDA
driver and is prevented by the ownership structure in
[`crates/tensor/ARCHITECTURE.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/ARCHITECTURE.md#zero-copy-cuda-tensor-mapping).

### API surfaces

| Language | Probe | Map | Handle |
|----------|-------|-----|--------|
| Rust | `is_cuda_available() -> bool` | `Tensor::cuda_map() -> Option<CudaMap>` | `CudaMap` — `device_ptr()`, `len()` |
| C | `hal_is_cuda_available()` | `hal_tensor_cuda_map()` → `hal_tensor_cuda_device_ptr()` → `hal_tensor_cuda_unmap()` | opaque handle |
| Python | `edgefirst_hal.is_cuda_available()` | `Tensor.cuda_map() -> CudaMap | None` | context manager — `.device_ptr`, `.size` |

See [`crates/tensor/README.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/README.md#cuda-tensor-mapping)
for usage snippets and
[`TESTING.md § CUDA tensor mapping`](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#cuda-tensor-mapping)
for the validation approach.

---

## Batched Preprocessing

Inference engines that support batching expect a single, fully-assembled batched
input tensor. The batch dimension `N` is always the **leading** dimension,
prepended to whatever layout the base tensor uses — packed `[N, H, W, C]` or
planar `[N, C, H, W]`. The HAL assembles that batch **forward** — calling
`convert()` once per source image into a distinct *tile* of one reused
destination tensor — rather than reconstructing it backward from per-element
sub-views. This is the primary motivation for the destination-region
(`view`/`batch`) API.

```
jpeg/png ─► source ─► convert ─► (batch tile n) ─► invoke ─► output ─► decode
            codec sets         glViewport into        full         batch-aware:
            shape/stride/      one reused dst         batched       whole-map +
            format             tensor                 tensor        ndarray index
```

1. **Decode → source.** The codec decodes an arbitrary-resolution image into a
   pre-allocated source tensor (buffer may be oversized) and sets its `shape`,
   `row_stride` (GPU-aligned — 64 B embedded, 256 B Nvidia), and `PixelFormat`
   to match the decoded content. The source EGLImage is keyed on these
   attributes, so it re-imports when they change — expected per distinct image.
2. **Convert → tile.** A batch is built by calling `convert()` once per source
   image into a destination sub-view: `convert(src, dst.batch(n), …)` or
   `convert(src, dst.view(region), …)`, or — to render the whole batch as **one
   import + one sync** — `convert_deferred(src, dst.batch(n), …)` in a loop
   followed by a single `flush()`. A `view`/`batch` sub-view resolves its
   **parent** (`view_origin`), so on Linux DMA-BUF the GL backend keys the
   EGLImage import on the *parent* identity+geometry — every sibling tile shares
   **one** import and is a `glViewport`/`glScissor` band into it (the offset is
   render state, never a cache key). `convert_deferred` skips the per-tile
   `glFinish`; `flush()` issues a single `finish_via_fence`. `batch(0)` on an
   N==1 tensor is byte- and identity-equivalent to the whole tensor.
3. **Invoke → decode.** The engine runs on the whole pre-assembled batched
   tensor and returns a batched output. The decoder is batch-aware: it `map()`s
   the whole output once and indexes each element with an ndarray slice — no
   tensor sub-view needed.

`convert()` always outputs an RGB-family color (`Grey`/`Rgb`/`Rgba`), packed
`HWC` or planar `CHW` — never YUV. Because `N` is the leading dimension, a tile
is element *n*, contiguous in memory whichever layout is used (a row-band in the
physical buffer). `convert_deferred` + `flush` render all N tiles into a single
parent import via `glViewport`, syncing once after the last tile (a plain
`convert` per tile still works and finishes eagerly). The first batch engine
covers the single-pass `Rgba`/`Bgra`/`Grey` u8/i8 DMA path; two-pass packed-RGB,
planar, and the macOS GL backend fall back to an eager per-tile convert (correct,
not yet one-import). The per-backend lowering (GL `glViewport`,
G2D destination crop, CPU offset+stride) and the cache-key invariant live in
[`crates/image/ARCHITECTURE.md § Batched preprocessing`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md#batched-preprocessing-building-a-batch-via-convert).
The `BufferIdentity`-sharing contract for regions lives in
[`crates/tensor/ARCHITECTURE.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/ARCHITECTURE.md#bufferidentity-and-egl-image-caching).

---

## Design Patterns

The workspace consistently applies a small set of Rust idioms across all
crates. Knowing which pattern is in play makes individual files much
easier to read.

### 1. Trait-based polymorphism

Common operations cross backend boundaries via traits:

- `TensorTrait<T>` — every tensor backend implements this; `shape`, `size`, `map`, `clone_fd`, `buffer_identity` are uniform across DMA / SHM / Mem / PBO.
- `ImageProcessorTrait` — `convert`, `draw_decoded_masks`, `draw_proto_masks`, `set_class_colors` work the same way against `ImageProcessor`, `G2DProcessor`, `GLProcessorThreaded`, `GLProcessorST`, `CPUProcessor`.
- `DetectionBox` — the decoder's `DetectBox` and any third-party detection type implement this so the tracker can read XYXY boxes, scores, and labels without copying.
- `PboOps` — the GL backend implements this trait (defined in `tensor`) so PBO tensors can route map/unmap/delete operations back to the GL thread without making `tensor` depend on `image`.

### 2. Enum dispatch

The hot `ImageProcessor` dispatch point uses the
[`enum_dispatch`](https://docs.rs/enum_dispatch) crate
(`#[enum_dispatch(ImageProcessor)]` in
[`crates/image/src/lib.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/src/lib.rs))
to avoid dynamic dispatch overhead on `convert()` and the `draw_*`
APIs — the match-style code reads like trait-object dispatch but compiles
to a direct call.

`TensorDyn` is a hand-written `match` over a `DType` discriminant rather
than an `enum_dispatch` macro, and the tracker is monomorphic over
`DetectionBox`, so neither uses the `enum_dispatch` crate itself; the
pattern (hot dispatch via an enum + compile-time fan-out) is shared, but
the mechanism differs.

### 3. Builder pattern

Complex multi-parameter constructors use a fluent builder:
[`DecoderBuilder`](https://docs.rs/edgefirst-decoder/latest/edgefirst_decoder/struct.DecoderBuilder.html),
[`ByteTrackBuilder`](https://docs.rs/edgefirst-tracker/latest/edgefirst_tracker/bytetrack/struct.ByteTrackBuilder.html),
and the in-progress `hal_decoder_params` C type. Builders enforce
invariants in `.build()` rather than scattering checks across setters.

### 4. Zero-copy operations

Used pervasively to avoid per-frame allocations:

- Memory-mapped hardware buffers (DMA-BUF, SHM, IOSurface, AHardwareBuffer)
- `&[T]` slice views into tensor maps
- ndarray `ArrayView` for math operations
- `WeakSender<T>` for cross-thread channels that should not extend lifetime

### 5. Hardware fallback chain

`ImageProcessor::new()` runs the GPU probe once (DMA-BUF round-trip,
GLES 3.1, PBO availability) and initializes every viable backend
(`gl`, `g2d`, `cpu`). The probe never re-runs after construction. Each
`convert()` / `draw_*()` call still walks the **OpenGL → G2D → CPU**
chain at dispatch time, falling through when a backend cannot service
the specific (src/dst format, memory type, operation) tuple — G2D
declines anything that requires GPU compute (e.g. mask compositing,
fused proto draws), and the CPU backend acts as the universal floor.
GL handles tricky platform cases via in-backend workarounds (for
example, NV12 → PlanarRgb on Vivante uses an automatic two-pass path
within the GL backend rather than declining) — only true capability
gaps cascade down the chain. Use `EDGEFIRST_FORCE_BACKEND=...` to pin
a single backend; this disables the fallback chain entirely — if the
forced backend cannot service the requested operation, the call fails
with `Error::ForcedBackendUnavailable` rather than dropping down to
the next backend. The `Tensor::new()` allocator chains DMA → SHM → Mem with
the same probe-once philosophy but always uses the first viable
backend per call. Both chains are defeatable via the
`EDGEFIRST_DISABLE_*` and `EDGEFIRST_FORCE_*` environment variables
for testing and benchmarking.

### 6. Type-safe foreign interfaces

Raw FFI bindings (`dma-heap`, `g2d-sys`, `khronos-egl`) are wrapped in
safe Rust types that enforce correct usage at compile time. The unsafe
boundary is concentrated in `crates/tensor/src/dma.rs`,
`crates/image/src/g2d.rs`, and `crates/image/src/gl/`; nothing
downstream sees `unsafe` blocks.

### 7. Python wrapper naming convention

PyO3 wrapper types use a `Py` prefix internally (e.g. `PyTensor`,
`PyPixelFormat`) to distinguish them from their Rust counterparts. The
Python-facing `Tensor` class wraps `TensorDyn` internally; users see
the unprefixed name. This convention makes it explicit which types are
Python-facing and which are internal Rust types — important when a
class needs both a `#[pyclass]` impl and an internal Rust impl.

### 8. Thread safety

The `Send + Sync` story across the workspace:

- `Tensor<T>` / `TensorDyn` — `Send + Sync`. Safe to share across threads.
- `Decoder` — `Send + Sync` for read operations (decoding). The builder consumes itself on `.build()`.
- `ImageProcessor` — `Send + Sync`. Whether concurrent GL work runs in parallel is a per-driver policy: on Vivante `galcore` (i.MX 8M Plus) and virtualized/paravirtual GPUs every command serializes on the global `GL_MUTEX`; on other drivers (Mali/Panfrost, V3D, Tegra, llvmpipe, real Apple GPU) instances execute GL concurrently (see [`crates/image/ARCHITECTURE.md § GL Concurrency Model`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md#gl-concurrency-model-serialization-policy)). Either way, one `ImageProcessor` per worker thread is the portable choice.
- `ByteTrack<T>` — `Send + Sync`. Mutable methods take `&mut self`, so concurrent updates require external synchronization.

### 9. Error handling

Each crate defines its own `Error` / `Result` pair (`DecoderError`,
`edgefirst_image::Error`, `edgefirst_tensor::Error`). Both
`edgefirst_image::Error` and `edgefirst_tensor::Error` implement
`From<std::io::Error>` so `?` propagates cleanly from file I/O and from
DMA-BUF / SHM syscalls. `DecoderError` does not, because the decoder
crate never opens files or fds — its inputs are already-loaded tensors
and JSON/YAML configuration strings.

The C API translates all errors into POSIX `errno` codes; see
[`crates/capi/ARCHITECTURE.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/ARCHITECTURE.md#error-convention).

---

## Performance Tracing Architecture

This section is the **architecture rationale** for the tracing
infrastructure. The user-facing how-to-use-it lives in
[README.md § Performance Tracing](https://github.com/EdgeFirstAI/hal/blob/main/README.md#performance-tracing).

### Design goals

1. **Near-zero cost when disabled** — no heap allocations, no
   formatting, no function calls on the hot path when no subscriber is
   active.
2. **Always compiled in** — span sites are present in all builds; only
   the capture infrastructure (subscriber + file writer) is
   feature-gated.
3. **Language-agnostic capture API** — Rust, Python, and C callers all
   use the same underlying mechanism.
4. **One process, one session** — simplifies the subscriber model and
   avoids runtime complexity from dynamic subscriber management.

### Zero-cost implementation

The [`tracing`](https://docs.rs/tracing) crate's `trace_span!` macro
compiles each span site to:

```text
static CALLSITE: DefaultCallsite = ...;       // registered once at first use
if INTEREST.load(Relaxed) != NEVER {           // single atomic load — the hot path
    // subscriber is interested → create span, record fields
} else {
    Span::none()                               // disabled — no work done
}
```

When no subscriber is installed (the default), the interest cache is
`NEVER` and the entire span creation is skipped. Properties:

- **No heap allocation** — field values use `tracing::field::debug(&val)`
  which stores a reference; actual `Debug` formatting is deferred to the
  subscriber's record method and only executes when actively tracing.
- **No string formatting** — the `?field` syntax wraps values lazily;
  the `Display` / `Debug` impl is never called when disabled.
- **No function calls** — the macro inlines to a single `Relaxed` atomic
  load followed by a branch-not-taken.
- **`Span::record()` guard** — for fields recorded after span creation
  (e.g. detection counts computed mid-function), `record()` checks
  `is_disabled()` and returns immediately when no subscriber cares.

### Span naming conventions

| Prefix | Meaning | Example |
|--------|---------|---------|
| (none) | Core algorithm phase | `decode`, `nms`, `score_filter` |
| `cpu_` | CPU backend operation | `cpu_format_convert`, `cpu_resize` |
| `gl_` | OpenGL backend operation | `gl_convert`, `gl_pass1_to_rgba` |
| `g2d_` | G2D hardware backend | `g2d_convert` |
| `py_` | Python binding entry point | `py_decode`, `py_convert` |
| `gl_pass1_` / `gl_pass2_` | Multi-pass GL sub-operation | `gl_pass1_to_rgba`, `gl_pass2_pack_rgb` |
| `image.gl_init`, `image.convert` (with `backend = "gl"`) | macOS GL processor entry points — same `<crate>.<function>` shape as the Linux GL spans, with a `platform` field tagging the OS and a `backend` field tagging the dispatch target. |

Field conventions:

- `n` or `n_*` — counts (detections, candidates, tracks)
- `mode` — algorithm variant (float / quant, proto / scaled)
- `*_fmt` — pixel format enum value
- `*_memory` — tensor memory backend (`Dma` / `Shm` / `Mem`)
- `layout` — data layout (`nhwc` / `nchw`)
- `pass` — multi-pass identifier (`pre_resize` / `post_resize` / `direct`)
- `platform` — `"linux"` or `"macos"` — emitted by spans that live in the GL platform layer
- `backend` — for `image.gl.platform_init`, the chosen transfer backend (`"dmabuf"` / `"iosurface"` / `"pbo"` / `"sync"`)

Each per-crate `ARCHITECTURE.md` documents the spans that crate emits.

### Crate layering

```text
┌─────────────────────────────────────────────────────────┐
│                    Application Code                      │
├─────────────────────────────────────────────────────────┤
│  edgefirst-hal (subscriber install, start/stop API)      │
│  ├─ tracing-chrome (Chrome JSON writer)                  │
│  └─ tracing-subscriber (subscriber registry)             │
├─────────────────────────────────────────────────────────┤
│  edgefirst-decoder │ edgefirst-image   │ edgefirst-      │
│  (decode spans)    │ (convert spans)   │ tracker         │
│                    │                   │ (update spans)  │
├────────────────────┴───────────────────┴────────────────┤
│  edgefirst-tensor  (alloc / map spans)                   │
├─────────────────────────────────────────────────────────┤
│  tracing crate (span macros, callsite interest cache)    │
└─────────────────────────────────────────────────────────┘
```

- **Inner crates** (`tensor`, `image`, `decoder`, `tracker`) depend on
  `tracing` as a **required** (non-optional) dependency. The span
  macros are always compiled. Cost when disabled: one `Relaxed` atomic
  load per span site.
- **Umbrella crate** (`edgefirst-hal`) gates `tracing-chrome` and
  `tracing-subscriber` behind the `tracing` feature (default on). These
  provide the capture infrastructure — the subscriber that actually
  writes the Chrome JSON file.
- **Binding crates** (Python, C API) forward the feature flag and
  provide language-appropriate start/stop APIs.

### Subscriber model

The HAL uses Rust's **global subscriber** model
(`set_global_default`):

- Only one subscriber per process lifetime (Rust's `tracing` design
  constraint).
- `start_tracing(path)` installs a Chrome JSON subscriber on first call.
- `stop_tracing()` drops the `FlushGuard`, flushing buffered spans to disk.
- After stop, the subscriber remains installed but the guard is gone — a
  second `start_tracing()` returns `TracingError::SessionExhausted`.
- If user code installs its own subscriber before calling
  `start_tracing()`, the HAL returns `TracingError::SubscriberInstallFailed`.

This single-session model is acceptable for profiling workflows where
one trace per process run is the norm. Applications needing multiple
trace files run separate processes.

### Error handling

The tracing API uses poison-resistant mutex access
(`unwrap_or_else(|e| e.into_inner())`) so a panic in one thread does not
permanently poison the tracing state and crash the process.

Error variants:

- `AlreadyActive` — a session is currently capturing
- `SessionExhausted` — a session was previously started and stopped
- `SubscriberInstallFailed` — another subscriber was already installed

### Multi-pass pipeline visibility

Image conversion operations that use multiple internal passes emit
per-pass spans to reveal the breakdown:

CPU 3-pass (format → resize → format):

```text
image_convert
└─ cpu_format_convert (pass="pre_resize", from=Nv12, to=Rgb)
└─ cpu_resize
└─ cpu_format_convert (pass="post_resize", from=Rgb, to=Rgba)
```

OpenGL 2-pass packed RGB:

```text
gl_convert
└─ gl_pass1_to_rgba (dst_w=640, dst_h=480)
└─ gl_pass2_pack_rgb (render_w=640, render_h=480)
```

OpenGL 2-pass Vivante NV12 → Planar workaround:

```text
gl_convert
└─ gl_pass1_to_rgba (dst_w=640, dst_h=480)
└─ gl_pass2_to_planar (dst_w=640, dst_h=480)
```

Spans within a multi-pass sequence are non-overlapping — the first
pass guard is explicitly dropped before the second pass span is
entered, producing clean sequential slices in the Perfetto timeline.

### Relationship to perf and benchmarks

| Tool | What it shows | When to use |
|------|---------------|-------------|
| HAL tracing | Span-level timing, pipeline structure, per-call metadata | Understanding pipeline structure, finding which stage is slow |
| `perf record` | Instruction-level CPU hotspots, cache misses, branch mispredictions | Optimizing within a single span |
| HAL benchmarks | Statistical timing (mean / p95 / p99) across many iterations | Measuring improvement from optimizations |

Recommended workflow:

1. Run with HAL tracing to identify the slow span(s).
2. Use `perf record` targeting the specific operation to find CPU hotspots.
3. Optimize the hotspot.
4. Re-run benchmarks to quantify the improvement.
5. Re-run tracing to confirm the span duration decreased.

See [BENCHMARKS.md](https://github.com/EdgeFirstAI/hal/blob/main/BENCHMARKS.md)
for benchmark infrastructure.

---

## Source Code Organization

```text
hal/
├── crates/
│   ├── tensor/             # edgefirst-tensor
│   ├── codec/              # edgefirst-codec (image decode into tensors)
│   ├── image/              # edgefirst-image
│   ├── decoder/            # edgefirst-decoder
│   ├── tracker/            # edgefirst-tracker
│   ├── hal/                # edgefirst-hal (umbrella)
│   ├── capi/               # edgefirst-hal-capi (C ABI)
│   ├── python/             # edgefirst_hal (PyO3 bindings)
│   ├── bench/              # edgefirst-bench (workspace dev-dep)
│   └── gpu-probe/          # internal CLI for GPU capability probing
├── tests/                  # Project-level Python tests (C integration tests live under crates/capi/tests/)
├── testdata/               # Git LFS-tracked fixtures (images, model outputs)
├── benchmarks/             # Per-platform benchmark JSON results
├── scripts/                # Build / audit / release tooling
├── .github/workflows/      # CI: test.yml, release.yml, benchmark.yml, sbom.yml
├── README.md               # Cross-cutting overview + Optimization Guide
├── ARCHITECTURE.md         # This file
├── TESTING.md              # Cross-cutting testing guide
├── BENCHMARKS.md           # Empirical performance reference
├── CHANGELOG.md            # Release history
└── Makefile                # Common workflow wrappers
```

Each `crates/<name>/` directory carries its own `README.md`,
`ARCHITECTURE.md`, and `TESTING.md` with the crate-specific story.

---

## Appendix C: DMA-BUF Identity and Tensor Caching

This is a cross-cutting story spanning the `tensor`, `image`, and `capi`
crates plus downstream integrators (V4L2 / GStreamer / libcamera). It
deserves a single canonical home, hence its place at the workspace root
rather than in any single per-crate doc.

### The problem: fd numbers are not stable buffer identifiers

A DMA-BUF is exported from the kernel as a file descriptor. Many callers
assume the same fd number means the same buffer and use fd as the cache
key for imported tensors (`hal_import_image`, EGL image creation, etc.).
**This assumption is wrong** and leads to cache misses or incorrect
hits.

The lifecycle of a DMA-BUF fd in a typical GStreamer pipeline:

1. A V4L2 decoder or libcamera source creates a buffer pool at startup,
   exporting each DMA-BUF once (`VIDIOC_EXPBUF`). The fd numbers are
   stable as long as the buffer pool exists.
2. A GStreamer `GstBuffer` wraps the DMA-BUF fd in a `GstMemory`
   object.
3. When the downstream element finishes with the buffer and unrefs it,
   the `GstMemory` refcount may drop to zero, **closing the fd**.
4. The upstream driver re-exports the buffer for the next frame,
   potentially receiving a **different fd number** even though the
   underlying physical buffer is the same.
5. Any cache keyed by fd number sees a miss even though the buffer
   content, EGL image, and GPU mapping are identical to a previous
   frame.

This fd recycling happens in practice with `v4l2h264dec`, `v4l2src`,
and `libcamerasrc`. Pool sizes are bounded (typically 4–16 buffers),
so fd numbers cycle through a small set, but there is no guarantee
that a particular fd number always refers to the same physical buffer.

### The solution: DMA-BUF inode as stable identity

The Linux kernel identifies each `dma_buf` object with a unique inode
in the anonymous inode filesystem. The inode is assigned when the
DMA-BUF is created and remains constant for its lifetime, regardless
of how many times it is exported or what fd numbers are assigned to
it.

```c
struct stat st;
fstat(fd, &st);
ino_t inode = st.st_ino;
```

`fstat` is a cheap syscall (microseconds), but it does run on **every
buffer handoff** because the inode is the lookup key — it must be
computed before the cache table is consulted. The cache lookup itself
is a hash-table probe; only the import path (`hal_import_image`) is
skipped on hits. If the per-frame `fstat` is undesirable on a
particular pipeline, layer an fd-to-inode memoization above the cache
(invalidated whenever an fd is closed). For a typical 4–16 buffer
pool, the steady-state cost is one `fstat` per frame and zero EGL
re-imports.

Cache key design for multi-plane buffers:

```c
typedef struct {
    ino_t inode;   // identifies the dma_buf kernel object
    gsize offset;  // byte offset within the DMA-BUF (NV12 planar)
} DmaBufCacheKey;
```

The `offset` is needed because a single DMA-BUF may contain multiple
planes at different byte offsets (NV12 luma at offset 0, chroma at
`stride * height`). The `(inode, offset)` pair uniquely identifies a
plane.

### Cache warm-up and steady state

Pool behaviour in practice:

| Stage | Frames | EGL import | Preprocessing time (i.MX 95) |
|-------|--------|------------|------------------------------|
| Warm-up | 1 – N | Yes | ~5–6 ms (import + GL) |
| Steady state | N+1 onwards | No | ~5–6 ms (GL only) |

Where N is the buffer pool depth (typically 9 for `v4l2h264dec` at
1080p with the NXP Amphion Wave5 VPU).

The preprocessing time in steady state is dominated by GL computation
(resize + letterbox + colorspace + quantization on Mali-G310: ~5–6 ms
at 1920×1080 → 640×640 INT8), not the EGL import. However, the EGL
import overhead does matter in low-latency or short-clip scenarios
where the pipeline never fully warms up.

### EGL image cache inside HAL

The image backend maintains an EGL image cache keyed by the
**tensor's** `BufferIdentity.id` — not by the DMA-BUF fd. Every call
to `hal_import_image()` / `hal_tensor_from_fd()` allocates a fresh
`BufferIdentity`, so calling those functions repeatedly with the same
fd produces a new key each time and misses the cache on every call.
The cache only hits when the **same tensor object** is reused across
frames.

This means the cache cannot rescue a pipeline that re-imports its
camera buffers every frame: each import is a brand-new identity.
HAL emits a "swept dead entry" log line when a tensor is dropped
while its EGL image is still in the cache; a steady stream of these
in a running pipeline is a sign that the calling code is churning
tensors.

### Android: AHardwareBuffer identity and getId interning

Android has the same shape of problem with a different system primitive.
CameraX/ImageReader pipelines recycle a small pool of AHardwareBuffers
but hand a fresh wrap across JNI every frame — the moral equivalent of
re-importing the same DMA-BUF. Unlike Linux, the HAL solves it
internally: `from_hardware_buffer` (and the allocation paths, so
export → re-import unifies) interns identities on
`AHardwareBuffer_getId`, the system's stable 64-bit allocation id
(API 31+, resolved via `dlsym` to keep the API-26 link floor). Every
re-wrap of the same physical buffer resolves to the same
`BufferIdentity`, and the EGLImage cache hits in steady state exactly as
the pooled-tensor contract above. The pointer is deliberately NOT used
as a key on older APIs (a released buffer's address can be reallocated —
the ABA aliasing class); API 26–30 keeps fresh-per-wrap identities,
correct but uncached, visible in the miss counters. See
[`crates/tensor/ARCHITECTURE.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/ARCHITECTURE.md#bufferidentity-and-egl-image-caching)
for the intern-table mechanics.

**The fix is at the calling layer** (the GStreamer / V4L2 / libcamera
adaptor that hands buffers to the HAL): maintain a cache of
`hal_tensor *` objects keyed by `(inode, offset)`, and never free them
between frames. Holding the tensor alive keeps its `BufferIdentity`
stable, which keeps the in-HAL EGL image cache hitting. This ensures
`hal_import_image` is called exactly once per unique DMA-BUF over the
lifetime of the pipeline.

For the per-tensor `BufferIdentity` mechanism that the EGL cache uses
internally, see
[`crates/tensor/ARCHITECTURE.md#bufferidentity-and-egl-image-caching`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/ARCHITECTURE.md#bufferidentity-and-egl-image-caching)
and
[`crates/image/ARCHITECTURE.md#egl-image-cache`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md#egl-image-cache).

### Reference implementation pattern (GStreamer adaptor)

A representative GStreamer source/transform element that hands camera
buffers to the HAL implements the inode-based cache as follows:

```c
typedef struct { ino_t inode; gsize offset; } InputCacheKey;

// On each input frame:
int fd = gst_dmabuf_memory_get_fd(mem);
gsize offset = 0;
gst_memory_get_sizes(mem, &offset, NULL);

struct stat st;
fstat(fd, &st);
InputCacheKey key = { .inode = st.st_ino, .offset = offset };

hal_tensor *tensor = g_hash_table_lookup(input_cache, &key);
if (!tensor) {
    // First time seeing this buffer — import and cache
    tensor = hal_import_image(processor, pd, chroma, /* ... */);
    g_hash_table_insert(input_cache,
                        g_memdup2(&key, sizeof key),
                        tensor);
}
// tensor is valid for the lifetime of the pipeline
```

The cache is invalidated on `set_caps` (resolution or format change)
and on `stop` (pipeline teardown). It is **never** invalidated
per-frame.

This pattern, applied above HAL, is what makes the steady-state
behaviour in the table above achievable. Without it, the warm-up row
applies on every frame.

---

## Contributing

See
[CONTRIBUTING.md](https://github.com/EdgeFirstAI/hal/blob/main/CONTRIBUTING.md)
for development environment setup, build instructions, testing
guidelines, code-style standards, and the pull-request process.

## Support

- Documentation: <https://doc.edgefirst.ai>
- GitHub Issues: <https://github.com/EdgeFirstAI/hal/issues>
- Email: <support@au-zone.com>
