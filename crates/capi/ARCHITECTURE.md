# edgefirst-hal-capi Architecture

## Overview

`edgefirst-hal-capi` is the C API layer over the EdgeFirst HAL Rust
workspace. It exposes tensor allocation, hardware-accelerated image
processing, ML decoder post-processing, ByteTrack object tracking, and the
delegate DMA-BUF framework as a stable C ABI suitable for consumption from
C, C++, GStreamer plugins, OpenCV pipelines, NPU delegates, and Python
extensions written outside PyO3. Headers are generated from Rust source
annotations via `cbindgen`; the implementation builds as both `staticlib`
and `cdylib`. On mobile the two artifacts have distinct roles: the
`cdylib` (`libedgefirst_hal.so`) is the Android JNI library, and the
`staticlib` (`libedgefirst_hal.a`) is the iOS static-embedding artifact
and the link-closure anchor for `scripts/validate-android-link.sh` in CI
(a Rust staticlib archives the full HAL dependency closure).

This crate is the highest-stakes ABI surface in the workspace. Performance
recommendations and lifecycle rules are not advisory — getting them wrong
loses the zero-copy property of the underlying Rust APIs and makes the
DMA-BUF / EGL / GL pipeline several times slower than its theoretical
ceiling. The largest section of this document is therefore the
[Performance recommendations](#performance-recommendations-dma-buf--egl-path)
that downstream integrators must follow.

## Module Map

| Module | Source | Responsibility |
|--------|--------|----------------|
| [`lib.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/src/lib.rs) | local | crate-wide setup, panic-safe FFI helpers, error reporting |
| [`tensor.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/src/tensor.rs) | local | ~1.2k lines — `hal_tensor_*` create/map/reshape/fd-share, `HalCpuAccess`/`HalCompression` enums, AHardwareBuffer wrap + handle surface, `hal_tensor_view`/`hal_tensor_batch` sub-regions, `hal_plane_descriptor_*` |
| [`image.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/src/image.rs) | local | ~2.6k lines — `hal_image_processor_*`, `hal_image_desc_*` builder, tensor image load/save, pre-allocated codec decode, draw masks (and tracked variants) |
| [`decoder.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/src/decoder.rs) | local | ~3.2k lines — `hal_decoder_*` create / decode detection / decode segmentation |
| [`tracker.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/src/tracker.rs) | local | ~300 lines — `hal_bytetrack_*` create / update / get_active_tracks |
| [`delegate.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/src/delegate.rs) | local | ~200 lines — Delegate DMA-BUF ABI types and camera adaptor format info |
| [`trace.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/src/trace.rs) | local | `hal_start_tracing` / `hal_stop_tracing` — surfaces the umbrella's tracing API |
| [`error.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/src/error.rs) | local | ~120 lines — error code conversion and `hal_error_message` helper |
| [`log.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/src/log.rs) | local | ~50 lines — `hal_log_init_file` / `hal_log_init_callback` |
| [`include/edgefirst/hal.h`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/include/edgefirst/hal.h) | generated | cbindgen-emitted header consumed by C/C++ users |

## Key Types

The ABI is built on **opaque handle pointers**: C code holds
`struct hal_tensor *`, `struct hal_image_processor *`, etc., and never
inspects the layout. This insulates downstream binaries from internal
struct churn — only the function signatures form the stable contract.

| C type | Wraps | Free with |
|--------|-------|-----------|
| `struct hal_tensor *` | `edgefirst_tensor::TensorDyn` | `hal_tensor_free` |
| `struct hal_tensor_map *` | `TensorMap<T>` (RAII guard) | `hal_tensor_map_unmap` |
| `struct hal_plane_descriptor *` | `edgefirst_tensor::PlaneDescriptor` | `hal_plane_descriptor_free` ONLY if the descriptor was never passed to `hal_import_image`. The import call consumes both descriptors unconditionally (success and failure paths alike — see `hal_import_image` docs), so calling `_free` after `_import_image` is a double-free. |
| `struct hal_image_processor *` | `edgefirst_image::ImageProcessor` | `hal_image_processor_free` |
| `struct HalImageDesc *` | `edgefirst_tensor::ImageDesc` (builder) | `hal_image_desc_free`. NOT consumed by `hal_tensor_new_image_desc` / `hal_image_processor_create_image_desc` (unlike plane descriptors) — a desc is reusable across create calls and independent of the tensors it created. |
| `struct hal_decoder *` | `edgefirst_decoder::Decoder` | `hal_decoder_free` |
| `struct hal_decoder_params *` | `edgefirst_decoder::DecoderBuilder` (in-progress) | `hal_decoder_params_free`. The handle is **not** consumed by `hal_decoder_new` — that call takes `const struct hal_decoder_params *` and clones the configuration into the new decoder, so the caller still owns the params and must free them after `hal_decoder_new` returns. |
| `struct hal_detect_box_list *` | `Vec<DetectBox>` | `hal_detect_box_list_free` |
| `struct hal_segmentation_list *` | `Vec<Segmentation>` | `hal_segmentation_list_free` |
| `struct hal_bytetrack *` | `edgefirst_tracker::ByteTrack<DetectBox>` | `hal_bytetrack_free` |
| `struct hal_track_info_list *` | `Vec<TrackInfo>` | `hal_track_info_list_free` |
| `hal_delegate_t` | opaque `void *` for delegate DMA-BUF queries | (caller-managed) |

All `hal_*_free` functions accept `NULL` safely (no-op).

### Tensor sub-regions (views / batch tiles)

`hal_tensor_view(t, hal_region)` and `hal_tensor_batch(t, n)` return a new
`struct hal_tensor *` that **shares the parent's `BufferIdentity`** (zero-copy,
no new GPU import) and is freed independently with `hal_tensor_free` — freeing a
view does not affect the parent. `hal_region` is a by-value struct
`{ size_t x, y, width, height; }` in pixels (no free). A view is
interchangeable with a tensor at `convert()`: pass `hal_tensor_batch(dst, n)` as
`dst` to render into batch element *n* — the `convert()` signature is unchanged.
An out-of-bounds region or `n ≥ N` returns `NULL` with `errno = EINVAL`. These
replace the former `hal_tensor_subview`; `plane_offset` is no longer a public
sub-region mechanism (it remains an internal import attribute for foreign /
multi-plane DMA-BUFs).

## CUDA Zero-Copy

`hal_is_cuda_available()` probes whether `libcudart` loaded and all CUDA
interop symbols resolved. The result is cached in a `OnceLock` after the
first probe — all subsequent calls are a single atomic load.

`hal_tensor_cuda_map()` returns an opaque `void *` that boxes a `CudaMap`
struct. The `Box` is converted to a raw pointer via `Box::into_raw` and its
lifetime is **transmuted to `'static`** at the ABI boundary — this is safe
only because the C caller contract guarantees `hal_tensor_cuda_unmap` is
called before the tensor is freed or mutated. Violating that ordering is
undefined behavior and will corrupt the deallocator.

The device pointer returned by `hal_tensor_cuda_device_ptr` is valid while
the map handle is live. It is usable across threads via the CUDA primary
context, which is shared per-device and does not require an explicit `cudaSetDevice`
on each thread. However, the CUDA runtime's internal locking is not the same
as OpenGL's `GL_MUTEX` — GL and CUDA operations on the same PBO buffer must
be sequenced: complete the GL render (including `glFinish`) before mapping
for CUDA, and unmap before submitting further GL commands that touch the buffer.

For the GL-registered-buffer path (`hal_tensor_cuda_map` on a PBO tensor),
the `cudaGraphicsMapResources` / `cudaGraphicsResourceGetMappedPointer` calls
are issued on the **GL worker thread** (inside the tokio `spawn_blocking` task
that holds `GL_MUTEX`). The returned device pointer is then handed off to the
caller; it is valid until `hal_tensor_cuda_unmap`, which issues
`cudaGraphicsUnmapResources` — also on the GL worker thread.

For the DMA-BUF path (`try_init_dma_cuda` on a DmaTensor), the import uses
`cudaImportExternalMemory` with `cudaExternalMemoryHandleTypeOpaqueFd`. The
fd is consumed by the import call; subsequent maps reuse the imported memory
object without further kernel involvement.

## Mobile Zero-Copy (IOSurface / AHardwareBuffer)

`TensorMemory::Dma` tensors are IOSurface-backed on macOS/iOS
(`hal_tensor_iosurface_id` exposes the cross-process handle) and
AHardwareBuffer-backed on Android. The C ABI carries the full mobile
allocation contract:

### CPU access declaration (breaking, all platforms)

`hal_tensor_new_image` and `hal_image_processor_create_image` take a
required `enum HalCpuAccess` (`HAL_CPU_ACCESS_NONE` / `_READ` / `_WRITE`
/ `_READ_WRITE`). Hardware (GPU/NPU) access is always implied; the enum
declares intended **CPU** mapping. `HAL_CPU_ACCESS_READ_WRITE`
reproduces the pre-declaration behavior byte-for-byte;
`HAL_CPU_ACCESS_NONE` keeps the allocation eligible for vendor tile
compression on Android (UBWC/AFBC/PVRIC/DCC) and costs nothing on
Linux/macOS. Precise declarations select cheaper mappings (read-only
IOSurface locks, dma-buf sync direction, write-combined maps). Mapping
beyond the declaration is best-effort — it may be refused or slow, warns
once per buffer, and increments `hal_unplanned_cpu_access_count()`.

### ImageDesc creation path and compression metadata

`hal_image_desc_new(width, height, format, dtype)` plus
`hal_image_desc_set_memory` / `_set_access` / `_set_compression` feed
`hal_tensor_new_image_desc` and
`hal_image_processor_create_image_desc` — the full-featured allocation
front door (opaque builder, so future fields are ABI-non-breaking).
`HAL_COMPRESSION_ANY` requests a vendor-compressed layout where the
device supports it: the recorded scheme is readable via
`hal_tensor_compression` (never `HAL_COMPRESSION_ANY`), linear fallbacks
are counted by `hal_compression_fallback_count()`, and
`hal_platform_compression_support(format, dtype)` is the conservative
capability probe. Requesting a specific scheme that the device cannot
provide is an error, and compression requires `HAL_CPU_ACCESS_NONE` —
a CPU-mapped buffer pins a linear layout by definition.

### NPU-direct AHardwareBuffer surface (Android)

For NPU runtimes that consume AHardwareBuffers natively (e.g. QNN):

- `hal_tensor_from_hardware_buffer` wraps an externally produced buffer
  zero-copy. Buffer identities are interned on `AHardwareBuffer_getId`
  (API 31+), so CameraX/ImageReader-style re-wraps of the same buffer
  share one `BufferIdentity` and hit the EGLImage import cache.
- `hal_tensor_hardware_buffer_ptr` /
  `hal_tensor_hardware_buffer_physical_dims` /
  `hal_tensor_recorded_row_stride` expose the raw handle and
  gralloc-padded geometry for the runtime's own import call.
- `hal_tensor_copy_to_flat` is the checked escape hatch when a consumer
  requires packed rows.
- `hal_image_processor_convert_fence` renders and returns an
  `EGL_ANDROID_native_fence_sync` fd, enabling GPU→NPU handoff without a
  CPU stall.

On-device behavior of this surface is validated by the internal
hal-mobile Device Farm harness (see the root
[TESTING.md § Android On-Device Validation](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)).

## Error Convention

| Return type | Success | Failure | Detail |
|-------------|---------|---------|--------|
| `int` | `0` | `-1` with `errno` set | `EINVAL`, `ENOMEM`, `EIO`, `ENOTSUP`, `ERANGE`, ... |
| pointer | non-NULL | `NULL` with `errno` set | Same set |
| `size_t` | actual size | `0` if handle is `NULL` | Treat `0` as "missing handle" |

`hal_error_message(errno_value)` returns a static string description for
logging.

### errno lifecycle

The HAL only sets `errno` on failure paths; success paths leave it
untouched. Some functions touch `errno` internally even when they
succeed (e.g. probing whether an optional kernel device exists), so the
value may be non-zero after a successful call. Integrators that surface
`errno` to a higher logging layer should snapshot it **immediately**
after the HAL call they care about, and clear it (`errno = 0;`) before
the next call whose errno they intend to inspect. Never rely on `errno`
to indicate success — always check the return value (`-1` / `NULL` /
`0`) first and only then inspect `errno`.

## cbindgen Pipeline

`crates/capi/build.rs` runs `cbindgen` against
[`crates/capi/cbindgen.toml`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/cbindgen.toml)
during a normal `cargo build`. The header is emitted to
[`crates/capi/include/edgefirst/hal.h`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/include/edgefirst/hal.h)
and is committed to the repository so downstream consumers can read the
ABI without compiling Rust. The committed header is regenerated whenever
the Rust source changes; contributors are expected to re-run
`cargo build -p edgefirst-hal-capi` and commit the updated header along
with their FFI change. CI does not currently run an explicit
`git diff --exit-code` parity check on the header, so review is the
gate against drift.

The header preamble defines `HAL_DTYPE_*`, `HAL_PIXEL_FORMAT_*`,
`HAL_TENSOR_MEMORY_*`, `HAL_ROTATION_*`, `HAL_FLIP_*`, and the
`HAL_LOG_LEVEL_*` enums in plain C — no `enum class` or other C++-specific
constructs — to keep the surface usable from straight C.

## Performance Recommendations (DMA-BUF / EGL Path)

> **Reference implementation:** the
> [`bench_preproc`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/tests/bench_preproc.c)
> C benchmark demonstrates every pattern below in a complete, runnable
> program. Build with `make bench` from
> [`crates/capi/tests/`](https://github.com/EdgeFirstAI/hal/tree/main/crates/capi/tests).

These patterns apply when using the DMA-BUF tensor APIs
(`hal_import_image()`, `hal_tensor_from_fd()`) together with
`hal_image_processor_convert()` from C or C++.

### Core principle: allocate once, reuse every frame, free on exit

Every call to the tensor-from-fd family of functions allocates a new
`BufferIdentity` with a globally unique ID. The OpenGL EGL image cache is
keyed by this ID (see
[`crates/image/ARCHITECTURE.md#egl-image-cache`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md#egl-image-cache)).
A new identity = cache miss = full `eglCreateImageKHR` import on the next
`convert()` call. On the i.MX 8M Plus this costs roughly 0.5–1.5 ms per
import — enough to drop below real-time at 30 fps when source and
destination are both re-imported every frame.

The correct lifecycle is three phases:

```text
INIT                           LOOP                          TEARDOWN
─────────────────────────      ──────────────────────────    ─────────────
Allocate processor             Reuse same tensors            Free tensors
Allocate src/dst tensors       Call convert()                Free processor
(from fd or create_image)      (EGL cache hits)
```

### Batched preprocessing (building an N-batch via convert)

To assemble an `[N, …]` model input, allocate the batched destination once via
`create_image`, then loop: for each source, call
`hal_image_processor_convert(p, src, hal_tensor_batch(dst, n), …)` to render
into element *n*. Every `hal_tensor_batch(dst, n)` shares the parent's
`BufferIdentity`, so on the OpenGL backend the destination EGLImage is imported
**once** (keyed on the parent identity+geometry) and each tile is placed into it
with `glViewport`/`glScissor` — the per-tile offset is render state, never a
separate import. For one import **and** one GPU sync, use
`hal_image_processor_convert_deferred(p, src, hal_tensor_batch(dst, n), …)` in
the loop and call `hal_image_processor_flush(p)` **once** at the end (it skips
the per-tile `glFinish` and issues a single fence; a deferred destination is not
safe to read until `flush` returns). Create the N `hal_tensor_batch` handles
once and reuse them across frames, like the tensors and the source pool.

**Constraint:** passing a `view`/`batch` of the *same* parent as both `src` and
`dst` of one `convert()` is undefined (the GL backend binds the whole EGLImage
as both texture and FBO), even if the rectangles are disjoint.

### Initialization

Allocate all tensors before entering the processing loop. When the source
dimensions are not known until the first frame arrives (e.g., V4L2
resolution negotiation), allocate on first use and keep the tensors for
all subsequent frames.

> **Important:** when using `hal_image_processor_convert()`, all
> internally-allocated tensors (intermediate buffers, output buffers)
> **must** be created via `hal_image_processor_create_image()`, **not**
> via `hal_tensor_new()` or `hal_tensor_from_fd()`. The processor's
> `create_image` method selects the optimal memory backend for the active
> GPU: DMA-BUF when the GPU uses EGLImage imports (Vivante, Mali), PBO
> when it uses pixel buffer transfers (NVIDIA desktop), heap as fallback.
> Using `hal_tensor_new()` with a hardcoded memory type bypasses this
> selection and can force a slow transfer path.
>
> **`hal_import_image` is NOT the same as `create_image`.** Despite living
> on the processor object, `hal_import_image()` does not use the
> processor's backend intelligence — it simply wraps an external DMA-BUF
> fd as a DMA tensor. It exists only for importing buffers that are
> **externally allocated** by V4L2, GStreamer, or codec output.

| Function | Use for | Memory selection |
|----------|---------|-----------------|
| `hal_image_processor_create_image()` | Intermediate and output buffers | **Auto** (DMA / PBO / Mem based on GPU) |
| `hal_import_image()` | External DMA-BUF import only | Always DMA (wraps caller's fd) |
| `hal_tensor_new()` / `hal_tensor_from_fd()` | Low-level tensor creation | Caller-specified (no GPU awareness) |

```c
// --- Initialization (once) -----------------------------------------------

struct hal_image_processor *proc = hal_image_processor_new();

// Source: external DMA-BUF from a V4L2 capture buffer.
// hal_plane_descriptor_new dups the fd — caller keeps its copy.
struct hal_plane_descriptor *pd = hal_plane_descriptor_new(v4l2_buf.m.fd);
struct hal_tensor *src = hal_import_image(
    proc, pd, NULL, width, height, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
// pd is consumed by hal_import_image — do NOT free

// Intermediate: processor-allocated RGBA for chained conversion.
// MUST use create_image (not hal_tensor_new) for optimal memory backend.
struct hal_tensor *mid = hal_image_processor_create_image(
    proc, model_w, model_h, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);

// Destination: PlanarRgb for model input, also processor-allocated.
struct hal_tensor *dst = hal_image_processor_create_image(
    proc, model_w, model_h, HAL_PIXEL_FORMAT_PLANAR_RGB, HAL_DTYPE_U8);
```

When the external buffer has row padding (`stride > width * bytes_per_pixel`),
set the stride on the plane descriptor:

```c
struct hal_plane_descriptor *pd = hal_plane_descriptor_new(v4l2_buf.m.fd);
hal_plane_descriptor_set_stride(pd, v4l2_fmt.fmt.pix.bytesperline);
struct hal_tensor *src = hal_import_image(
    proc, pd, NULL, width, height, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
```

### Main Processing Loop

Reuse the same tensor objects on every frame. The EGL image cache hits on
the second and all subsequent iterations because `BufferIdentity.id` has
not changed.

```c
while (running) {
    // Upstream writes new pixel data into the DMA-BUF (V4L2 DQBUF, decoder
    // output, etc.). The tensor and its cached EGLImage remain valid — no
    // need to recreate anything. EGLImage is a handle to live physical
    // memory, not a snapshot.

    // Two-pass conversion: NV12 → RGBA → PlanarRgb
    hal_image_processor_convert(proc, src, mid,
                                HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
    hal_image_processor_convert(proc, mid, dst,
                                HAL_ROTATION_NONE, HAL_FLIP_NONE, NULL);

    // Feed dst to the model ...
}
```

Both `mid` entries (destination in pass 1, source in pass 2) live in
independent EGL image caches (`dst_egl_cache` and `src_egl_cache`), so
both sides achieve cache hits after the first frame.

The `glFinish()` issued at the end of each `convert()` call guarantees
coherency, making chained calls safe without explicit synchronization.

### Teardown

Free tensors only when the pipeline is torn down — typically at program
exit or when a pipeline is reconfigured (e.g., resolution change):

```c
hal_tensor_free(dst);
hal_tensor_free(mid);
hal_tensor_free(src);
hal_image_processor_free(proc);
```

### Buffer pool integration (V4L2 / GStreamer)

When upstream provides DMA-BUF fds from a buffer pool (V4L2 MMAP,
GStreamer allocator, codec output ring), a small number of physical
buffers cycle through a queue. Map each pool slot to a HAL tensor that is
created once and reused whenever that slot is dequeued.

```c
#define N_BUFS 4
struct hal_tensor *pool_tensors[N_BUFS] = { NULL };

// At DQBUF time, lazily create or reuse the tensor for this buffer index.
int buf_index = v4l2_buf.index;  // 0..N_BUFS-1

if (pool_tensors[buf_index] == NULL) {
    // First time this pool slot is seen — create the HAL tensor.
    struct hal_plane_descriptor *pd = hal_plane_descriptor_new(v4l2_buf.m.fd);
    pool_tensors[buf_index] = hal_import_image(
        proc, pd, NULL, width, height,
        HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
    // pd is consumed — do NOT free
}

// Reuse the existing tensor — cache hit on every subsequent dequeue.
hal_image_processor_convert(proc, pool_tensors[buf_index], dst, ...);

// QBUF returns the buffer to the driver; the tensor stays alive.
ioctl(fd, VIDIOC_QBUF, &v4l2_buf);

// --- Teardown (when stopping capture) ---
for (int i = 0; i < N_BUFS; i++) {
    hal_tensor_free(pool_tensors[i]);  // NULL-safe
    pool_tensors[i] = NULL;
}
```

With `hal_tensor_from_fd()` the pattern is similar — it dups internally,
so the caller retains the original fd:

```c
if (pool_tensors[buf_index] == NULL) {
    size_t shape[] = { height, width, 1 };  // NV12 luma plane
    pool_tensors[buf_index] = hal_tensor_from_fd(
        HAL_DTYPE_U8, v4l2_buf.m.fd, shape, 3, "v4l2_src");
    hal_tensor_set_format(pool_tensors[buf_index], HAL_PIXEL_FORMAT_NV12);
}
```

### fd Ownership Summary

| Function | fd ownership | When to `dup()` |
|----------|--------------|-----------------|
| `hal_plane_descriptor_new()` | Dups eagerly — caller retains original | Never |
| `hal_import_image()` | Consumes both descriptors (success or fail) | Never (descriptors already duped the fd) |
| `hal_tensor_from_fd()` | Dups internally — caller retains original | Never |

### Tensor ownership: cached imports vs. fresh allocations

Once you adopt an inode-keyed cache, the `struct hal_tensor *` you hand
to `hal_image_processor_convert()` will be one of two kinds:

| Kind | Source | Caller must `hal_tensor_free`? | Notes |
|------|--------|--------------------------------|-------|
| **Cached** | returned from your `(inode, offset)` cache lookup; created **once** at the first cache miss via `hal_import_image()` / `hal_tensor_from_fd()` | **No** — the cache owns it for the lifetime of the pipeline | Freeing mid-pipeline drops the EGL image cache entry, churns `BufferIdentity`, and turns every subsequent frame into an import miss |
| **Fresh** | allocated for this call (system-memory fallback when the source has no DMA-BUF, intermediate `create_image()` for chained `convert()`) | **Yes** — exactly once, after the last `convert()` that uses it | Forgetting leaks the underlying memory backend (DMA-BUF fd, PBO, or heap) |

A robust pattern tracks the distinction with a small flag alongside the
tensor pointer, e.g. an output parameter from the helper that produces
the source tensor:

```c
struct hal_tensor *src = lookup_or_import(cache, fd, &src_owned);
hal_image_processor_convert(proc, src, dst, /* ... */);
if (src_owned) hal_tensor_free(src);   // only when not cached
```

The same distinction applies to destination tensors when chaining
multiple `convert()` calls: the intermediate `create_image()` allocation
is owned by the chain orchestrator and must be freed once at the end
(not after each `convert()`).

### Anti-patterns

The following patterns cause severe performance regressions. Each is
explained with the underlying cost so the reason is clear, not just the
rule.

#### 1. Creating a tensor from fd every frame

```c
// BAD: ~1-3 ms overhead per frame from EGLImage re-import
while (running) {
    struct hal_plane_descriptor *pd = hal_plane_descriptor_new(v4l2_buf.m.fd);
    struct hal_tensor *src = hal_import_image(
        proc, pd, NULL, width, height, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
    hal_image_processor_convert(proc, src, dst, ...);
    hal_tensor_free(src);
}
```

*Why it is slow:* every call allocates a new `BufferIdentity` with a fresh
ID. The EGL image cache keys on `(BufferIdentity.id, chroma_id)`, so the
cache never hits. Each `convert()` call must call `eglCreateImageKHR` to
re-import the DMA-BUF as a GL texture. On the Vivante GC7000UL this takes
0.5–1.5 ms. When both source and destination are re-imported, the cost
doubles.

#### 2. Freeing and reallocating tensors between frames

```c
// BAD: same cost as #1, plus heap allocation overhead
while (running) {
    struct hal_tensor *dst = hal_image_processor_create_image(
        proc, model_w, model_h, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8);
    hal_image_processor_convert(proc, src, dst, ...);
    // ... use dst ...
    hal_tensor_free(dst);
}
```

*Why it is slow:* in addition to EGL image cache misses, every iteration
allocates and frees a DMA-BUF (or PBO), which involves kernel calls
(`dma-heap ioctl` or `glBufferData`). On memory-constrained embedded
targets, CMA pool fragmentation can also cause allocation failures after
sustained operation.

#### 3. Using the same tensor as both src and dst

```c
// BAD: undefined behavior
hal_image_processor_convert(proc, tensor, tensor, ...);
```

*Why it fails:* the OpenGL backend binds `src` as a texture and `dst` as a
framebuffer attachment. Sampling from and rendering to the same image in a
single draw call is undefined behavior per the OpenGL ES specification.
Results range from correct output (by accident) to GPU hangs.

#### 4. Using `hal_tensor_new()` instead of `hal_image_processor_create_image()`

```c
// BAD: bypasses processor's memory backend selection
size_t shape[] = { 640, 640, 3 };
struct hal_tensor *dst = hal_tensor_new(HAL_DTYPE_U8, shape, 3,
                                         HAL_TENSOR_MEMORY_DMA, "output");
hal_image_processor_convert(proc, src, dst, ...);
```

*Why it is slow:* `hal_image_processor_create_image()` inspects the active
GPU backend and selects the optimal memory type — DMA-BUF for
EGLImage-capable GPUs (Vivante, Mali), PBO for desktop GPUs (NVIDIA), heap
for CPU-only. Calling `hal_tensor_new()` with a hardcoded memory type
skips this selection. On a PBO-preferred system, a DMA tensor forces a
slow `glTexSubImage2D` upload; on a DMA-preferred system, a heap tensor
forces a full CPU readback.

#### 5. Ignoring row stride for padded buffers

```c
// BAD: corrupted output (skewed image)
struct hal_plane_descriptor *pd = hal_plane_descriptor_new(padded_fd);
struct hal_tensor *src = hal_import_image(
    proc, pd, NULL, width, height, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
```

*Why it fails:* many V4L2 drivers and GStreamer allocators pad rows to
alignment boundaries (e.g., 128-byte or 256-byte alignment). If the
stride is not communicated to HAL, the GPU interprets rows at
`width * bpp` spacing while the actual data is at `stride` spacing,
producing a skewed or corrupted image. Set the stride on the plane
descriptor via `hal_plane_descriptor_set_stride()` when
`bytesperline > width * bytes_per_pixel`.

### Live-memory semantics

After a V4L2 decoder, camera ISP, or codec writes new pixel data into a
DMA-BUF, the existing tensor and its cached EGLImage remain valid. **Do
not** free and recreate the tensor. Simply call `convert()` again. The GPU
reads the updated content because EGLImage is a handle to physical memory,
not a snapshot of its contents at import time. This is the key property
that makes the allocate-once / reuse-every-frame pattern work.

## Delegate DMA-BUF Framework

The Delegate DMA-BUF Framework defines an ABI contract for querying
DMA-BUF tensor information from external TFLite delegates (NXP Neutron
NPU, VxDelegate, etc.). The HAL owns the type definitions; function
implementations live in delegate shared libraries.

The API is **query-only by design**: delegates own all DMA-BUF
allocations internally. Consumers query tensor metadata (fd, shape,
dtype) and synchronize caches — they never register, allocate, bind, or
release buffers through this API.

Each delegate ships a self-contained `hal_dmabuf.h` header with all type
definitions and function declarations. Consumers do **not** need the full
`edgefirst/hal.h` — they either include the delegate's header or load
symbols via `dlsym`.

### Zero-copy data flow

```text
Camera DMA-BUF
  → hal_plane_descriptor (wraps camera_fd, stride, offset)
  → hal_import_image(proc, camera_pd, NULL, w, h, format, dtype)
  → HAL Tensor with image attributes (camera source)

NPU input DMA-BUF (fd/shape from hal_dmabuf_get_tensor_info)
  → hal_plane_descriptor (wraps npu_input_fd, stride, offset)
  → hal_import_image(proc, npu_pd, NULL, model_w, model_h, format, dtype)
  → HAL Tensor with image attributes (NPU destination)

ImageProcessor::convert(camera_tensor, npu_tensor)
  → DMA→DMA convert (resize, colorspace, letterbox)

hal_dmabuf_sync_for_device(delegate, input_tensor_index)
  → flush CPU caches so NPU can read

NPU inference

hal_dmabuf_sync_for_cpu(delegate, output_tensor_index) [optional]
  → invalidate CPU caches so CPU sees NPU writes
```

Both camera and NPU input tensors are imported via `hal_import_image()`
so that `convert()` has the pixel format, dimensions, stride, and plane
metadata it requires. `hal_tensor_from_fd()` creates raw tensors without
image attributes; to use such tensors as a `convert()` source or
destination you must either import them as images (via
`hal_import_image()`) or attach the required image metadata (e.g. with
`hal_tensor_set_format()`).

Output DMA-BUF access via `hal_dmabuf_get_tensor_info()` is optional and
depends on the use case: useful when feeding outputs to GPU (e.g., mask
rendering via OpenGL), but for CPU-side post-processing (YOLO decode,
NMS) it is simpler to read tensor data directly since the decode will
memcpy anyway.

### Type definitions

**`hal_delegate_t`** — opaque handle for any delegate, defined as
`void *`. Implementations receive this as a parameter and cast it
internally to their concrete delegate type. The delegate lifetime is
managed by the caller; the HAL never creates or destroys delegates.

> **Important:** exported function signatures **must** use
> `hal_delegate_t` (`void *`), not `TfLiteDelegate *` or any other
> concrete type. This ensures ABI compatibility across different delegate
> implementations.

**`hal_dmabuf_tensor_info`** — describes a single delegate tensor's
DMA-BUF:

| Field | Type | Description |
|-------|------|-------------|
| `size` | `size_t` | Buffer size in bytes (may exceed logical tensor size for padded buffers) |
| `offset` | `size_t` | Byte offset within the DMA-BUF |
| `shape` | `size_t[HAL_DMABUF_MAX_NDIM]` | Tensor dimensions (max 8) |
| `ndim` | `size_t` | Number of valid entries in `shape` |
| `fd` | `int` | DMA-BUF file descriptor (**borrowed** — do not close) |
| `dtype` | `hal_dtype` | Element data type |

Fields are ordered to eliminate padding on LP64 (`size_t` first, then
smaller 4-byte fields). Total: 96 bytes on LP64.

All fields are **mandatory**. Implementations must populate `shape`,
`ndim`, and `dtype` in addition to `fd`, `offset`, and `size`. An
implementation that cannot determine the shape should set `ndim = 0`.

**`hal_camera_adaptor_format_info`** — describes a camera format
adaptor:

| Field | Type | Description |
|-------|------|-------------|
| `input_channels` | `int` | Number of input channels (e.g., 4 for RGBA) |
| `output_channels` | `int` | Number of output channels (e.g., 3 for RGB) |
| `fourcc` | `char[HAL_FOURCC_MAX_LEN]` | NUL-terminated V4L2 FourCC string (ASCII, ≤ 4 bytes + NUL) |

### DMA-BUF function ABI

These functions are **not implemented in the HAL**. They document the
exact ABI that delegate shared libraries must export and that consumers
probe via `dlsym`. All exported symbols must use
`__attribute__((visibility("default")))`.

```c
/* Get the delegate's internal handle.
 * When TFLite creates a delegate via TfLiteExternalDelegateCreate(), it
 * wraps the real delegate in an opaque adapter. This function returns
 * the inner delegate pointer that the hal_dmabuf_* functions expect.
 * Returns the inner delegate handle, or NULL if no delegate has been
 * created. */
hal_delegate_t hal_dmabuf_get_instance(void);

/* Returns 1 if DMA-BUF tensor access is supported, 0 otherwise.
 * Does not set errno. */
int hal_dmabuf_is_supported(hal_delegate_t delegate);

/* Get DMA-BUF tensor info for a given tensor index.
 * Returns 0 on success, -1 on error (sets errno).
 * info_size enables forward-compatible versioning: pass
 * sizeof(hal_dmabuf_tensor_info). Implementations must:
 *   1. memset(info, 0, info_size) before populating
 *   2. Only write fields whose offsetof + sizeof fits within info_size
 * Errno: EINVAL (NULL info, negative tensor_index, info_size too small),
 *        ENOTSUP (DMA-BUF not supported),
 *        ERANGE (tensor_index out of range),
 *        EIO (DMA-BUF ioctl or internal failure) */
int hal_dmabuf_get_tensor_info(hal_delegate_t delegate,
                               int tensor_index,
                               hal_dmabuf_tensor_info *info,
                               size_t info_size);

/* Flush CPU caches → device can read.
 * Call after writing to an input tensor (e.g. via convert()), before NPU. */
int hal_dmabuf_sync_for_device(hal_delegate_t delegate, int tensor_index);

/* Invalidate CPU caches → CPU sees device writes.
 * Call after NPU inference completes, before reading output tensor data. */
int hal_dmabuf_sync_for_cpu(hal_delegate_t delegate, int tensor_index);
```

`tensor_index` must be non-negative; negative values return `-1` with
`errno = EINVAL`.

### Camera adaptor functions

Some delegates support NPU-accelerated format conversion (e.g. RGBA →
RGB channel slicing, uint8 → int8 quantization) that runs as part of
the inference graph. These functions allow consumers to query format
support without vendor-specific symbols.

Format conversion is configured **before** graph compilation via
delegate options (e.g. `DelegateOptions::option("camera_adaptor",
"rgba")`), not through this query API.

```c
/* Returns 1 if the given format is supported, 0 otherwise.
 * Does not set errno. Delegates without camera adaptor support
 * always return 0. */
int hal_camera_adaptor_is_supported(hal_delegate_t delegate,
                                    const char *format);

/* Query camera adaptor format information.
 * Returns 0 on success, -1 on error (sets errno).
 * info_size enables forward-compatible versioning.
 * Errno: EINVAL (NULL format or info),
 *        ENOTSUP (format not supported by this delegate) */
int hal_camera_adaptor_get_format_info(hal_delegate_t delegate,
                                       const char *format,
                                       hal_camera_adaptor_format_info *info,
                                       size_t info_size);
```

### errno requirements

Implementations **must** set `errno` before returning `-1`:

| Function | EINVAL | ENOTSUP | ERANGE | EIO |
|----------|--------|---------|--------|-----|
| `hal_dmabuf_get_tensor_info` | NULL info, negative index, info_size too small | DMA-BUF not supported | tensor_index out of range | ioctl or internal failure |
| `hal_dmabuf_sync_for_device` | NULL delegate, negative index | — | tensor_index out of range | ioctl failure |
| `hal_dmabuf_sync_for_cpu` | NULL delegate, negative index | — | tensor_index out of range | ioctl failure |
| `hal_camera_adaptor_get_format_info` | NULL format or info | format not supported | — | — |

Functions that return 1/0 (`hal_dmabuf_is_supported`,
`hal_camera_adaptor_is_supported`) do **not** set errno.

### Integration pattern

1. The delegate shared library (e.g., `libvx_delegate.so`,
   `libneutron_delegate.so`) implements the functions above and exports
   them as public C symbols with default visibility.
2. Consumers probe for the symbols at runtime via `dlsym` on the
   delegate's shared library.
3. Consumers call `hal_dmabuf_get_instance()` to obtain the inner
   delegate handle (needed when the delegate was created via
   `TfLiteExternalDelegateCreate()`).
4. If `hal_dmabuf_is_supported()` returns 1, consumers call
   `hal_dmabuf_get_tensor_info()` to obtain the DMA-BUF fd and shape for
   each tensor of interest.
5. The fd and metadata are passed to `hal_import_image()` to create a
   HAL tensor with full image attributes for use with
   `hal_image_processor_convert()`.

### Lifecycle and ownership

- The **delegate** owns all DMA-BUF allocations. File descriptors
  returned by `hal_dmabuf_get_tensor_info()` are borrowed — callers must
  not close them.
- Sync functions (`sync_for_device`, `sync_for_cpu`) wrap
  `DMA_BUF_IOCTL_SYNC` and bracket hardware access. They must be called
  at the correct points in the pipeline (see data flow above).
- The delegate instance itself is created and destroyed by the caller
  (e.g., via the TFLite C API). The HAL never manages delegate
  lifetime.

### Symbol visibility

All exported `hal_dmabuf_*` and `hal_camera_adaptor_*` functions must be
annotated with `__attribute__((visibility("default")))` so they remain
visible even when the delegate is compiled with `-fvisibility=hidden`.

### Thread safety

Delegate functions are not required to be thread-safe for concurrent
calls on the same delegate instance. Callers must serialize access per
delegate. Different delegate instances may be used concurrently from
different threads.

## Logging API

HAL logging is off by default. Initialise it once per process before any
other HAL calls. Both APIs are race-safe: only the first successful call
takes effect; subsequent calls return `-1` with `errno = EALREADY`.

```c
#include <edgefirst/hal.h>

// Option 1: write [LEVEL] target: message lines to a FILE*
hal_log_init_file(stderr, HAL_LOG_LEVEL_DEBUG);

// Option 2: forward each record to a custom callback
void my_logger(hal_log_level level, const char *target,
               const char *message, void *userdata) {
    fprintf(stderr, "[%d] %s: %s\n", level, target, message);
}
hal_log_init_callback(my_logger, NULL, HAL_LOG_LEVEL_INFO);
```

Available log levels: `HAL_LOG_LEVEL_ERROR`, `HAL_LOG_LEVEL_WARN`,
`HAL_LOG_LEVEL_INFO`, `HAL_LOG_LEVEL_DEBUG`, `HAL_LOG_LEVEL_TRACE`.

## Inter-Crate Interfaces

| Direction | Crate | Interface |
|-----------|-------|-----------|
| Wraps | [`edgefirst-tensor`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/) | `Tensor`, `TensorDyn`, `PlaneDescriptor`, `from_planes` |
| Wraps | [`edgefirst-image`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/) | `ImageProcessor`, draw / convert APIs |
| Wraps | [`edgefirst-decoder`](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/) | `Decoder`, `DecoderBuilder`, `DetectBox`, `Segmentation` |
| Wraps | [`edgefirst-tracker`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/) | `ByteTrack`, `TrackInfo` |
| Wraps | [`edgefirst-hal`](https://github.com/EdgeFirstAI/hal/blob/main/crates/hal/) | `trace::start_tracing` / `stop_tracing` (feature `tracing`) |

## Cross-References

- Project architecture: [../../ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md)
- Image-side EGL cache and PBO dispatch: [../image/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md)
- Tensor architecture (BufferIdentity, multi-plane DMA-BUF): [../tensor/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/ARCHITECTURE.md)
- DMA-BUF identity story: [ARCHITECTURE.md#appendix-c-dma-buf-identity-and-tensor-caching](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md#appendix-c-dma-buf-identity-and-tensor-caching)
- Optimization guide: [README.md#optimization-guide](https://github.com/EdgeFirstAI/hal/blob/main/README.md#optimization-guide)
- Performance tracing usage: [README.md#performance-tracing](https://github.com/EdgeFirstAI/hal/blob/main/README.md#performance-tracing)
