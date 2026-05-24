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
| `tensor` | [crates/tensor/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/ARCHITECTURE.md) — backend dispatch, multi-plane DMA-BUF, BufferIdentity |
| `codec` | [crates/codec/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec/ARCHITECTURE.md) — custom baseline JPEG decoder, SIMD dispatch (NEON/SSE4.1/SSSE3/SSE2), zero-allocation scratch model, strided/EXIF-rotated output |
| `image` | [crates/image/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md) — GL/G2D/CPU, EGL image cache, GL_MUTEX, Vivante workaround, shutdown safety |
| `decoder` | [crates/decoder/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/ARCHITECTURE.md) — model-type selection, dshape contract, per-scale framework, fused proto path |
| `tracker` | [crates/tracker/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/ARCHITECTURE.md) — ByteTrack two-pass association, Kalman state |
| `hal` (umbrella) | [crates/hal/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/hal/ARCHITECTURE.md) — re-export layer + tracing subscriber |
| `capi` (C API) | [crates/capi/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/ARCHITECTURE.md) — opaque-handle ABI, performance recommendations, Delegate DMA-BUF framework |
| `python` | [crates/python/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/python/ARCHITECTURE.md) — PyO3 bindings, numpy 3-path copy strategy, abi3 wheels |

The high-level system diagram lives at the top of
[README.md § System Architecture](https://github.com/EdgeFirstAI/hal/blob/main/README.md#system-architecture);
this document does not reproduce it.

---

## Per-Crate Summary

Each sub-crate has a single responsibility in the inference pipeline:

- [`edgefirst-tensor`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/) — the foundation. Provides `Tensor<T>` and `TensorDyn` with four interchangeable backends (DMA / SHM / Mem / PBO), multi-plane composition for V4L2 NV12M, the `BufferIdentity` cache key, and the `PboOps` trait that lets the GL backend manage PBO lifetimes through a `WeakSender` channel.
- [`edgefirst-codec`](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec/) — Image decoding (JPEG, PNG) into pre-allocated tensor buffers with support for u8, u16, i8, i16, and f32 pixel types. Supports strided output for GPU pitch-aligned DMA-BUF/PBO tensors. Designed for the allocate-once, decode-in-loop pattern.
- [`edgefirst-image`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/) — the GPU/G2D/CPU image processor. Owns the GL thread, EGL image caches, and shutdown defense layers. Provides format conversion, geometric transforms, and three mask-rendering pipelines (materialized, fused proto, tracked).
- [`edgefirst-decoder`](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/) — model output post-processing. YOLOv5/v8/v11/v26 (incl. end-to-end) and ModelPack. NEON-optimized per-scale split-tensor framework. Validates `shape` / `dshape` declarations against the physical-memory-order contract at builder time.
- [`edgefirst-tracker`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/) — ByteTrack with Kalman-smoothed trajectories. Generic over the detection box type; the decoder's `DetectBox` plugs in via the `DetectionBox` trait.
- [`edgefirst-hal`](https://github.com/EdgeFirstAI/hal/blob/main/crates/hal/) — umbrella crate. Re-exports the five functional crates and owns the optional Chrome JSON tracing subscriber.
- [`edgefirst-hal-capi`](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/) — C ABI layer with cbindgen-generated header. Defines the [Delegate DMA-BUF framework](https://github.com/EdgeFirstAI/hal/blob/main/crates/capi/ARCHITECTURE.md#delegate-dma-buf-framework) ABI used by NXP Neutron, VxDelegate, and other TFLite delegates.
- [`crates/python`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python/) — PyO3 bindings, published as `edgefirst-hal` on PyPI. Contains the three-path numpy copy dispatcher.

The internal dependency graph and external dependency list live in
[README.md § Dependencies](https://github.com/EdgeFirstAI/hal/blob/main/README.md#dependencies).

---

## Platform Support Matrix

The HAL targets three tiers of platforms with different acceleration
primitives. The `TensorMemory` enum is shared across all tiers (same
discriminants over the C ABI); the underlying storage and the GL
transfer backend differ.

| Capability | Embedded Linux (i.MX, RPi5, Jetson) | Desktop Linux (x86_64) | macOS (Apple Silicon) |
|------------|--------------------------------------|------------------------|------------------------|
| `TensorMemory::Mem` | Heap | Heap | Heap |
| `TensorMemory::Shm` | `shm_open` | `shm_open` | `shm_open` |
| `TensorMemory::Dma` | DMA-BUF heap (`/dev/dma_heap/*`) | DMA-BUF heap if mountable; PBO otherwise | IOSurface (CoreFoundation framework) |
| `TensorMemory::Pbo` | GLES PBO | GLES PBO | — (no PBO on the macOS backend) |
| GL transfer backend | `TransferBackend::DmaBuf` (Vivante, Mali, V3D) | `DmaBuf` or `Pbo` (NVIDIA discrete uses `Pbo`) | `IOSurface` via ANGLE |
| GL → backend translation | Native EGL → driver (vendor blob or Mesa) | Native EGL → driver | ANGLE EGL → Metal |
| Hardware 2D blitter | G2D on NXP i.MX | — | — |
| Zero-copy import API | `EGL_EXT_image_dma_buf_import` | Same, when available | `EGL_ANGLE_iosurface_client_buffer` |
| Cross-process buffer handle | DMA-BUF fd (over `SCM_RIGHTS`) | Same | IOSurfaceID (`u32` via Mach port or XPC) |
| Probe function | `is_dma_available()` | Same | `is_iosurface_available()` |
| Portable probe | `is_gpu_buffer_available()` — works on all three |

The portable `is_gpu_buffer_available()` is the recommended cross-platform
gate when the question is "can I ask for `TensorMemory::Dma` and expect a
zero-copy GPU-importable buffer?" The platform-specific probes
(`is_dma_available`, `is_iosurface_available`) remain when callers need
to know *which* primitive is in use — e.g. to decide whether to call
`hal_tensor_clone_fd` (Linux) vs `hal_tensor_iosurface_id` (macOS).

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

- Memory-mapped file descriptors (DMA-BUF, SHM)
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
- `ImageProcessor` — `Send + Sync`, but concurrent use serializes on the global `GL_MUTEX` (see [`crates/image/ARCHITECTURE.md#gl-command-serialization-gl_mutex`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md#gl-command-serialization-gl_mutex)). For best performance: one `ImageProcessor` per worker thread.
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
| `image.gl.` | macOS GL platform layer (parallel processor) | `image.gl.platform_init`, `image.gl.import_buffer`, `image.gl.bind_texture`, `image.gl.convert` |
| `tensor.iosurface.` | macOS IOSurface tensor allocation | `tensor.iosurface.create` |

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
