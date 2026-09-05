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
| `codec` | [crates/codec/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec/ARCHITECTURE.md) — custom baseline JPEG decoder, SIMD dispatch (NEON/AVX2/SSE4.1/SSE2), opt-in fused RGB/NV12 decode output and `DctMethod` accuracy/speed selection, zero-allocation scratch model, strided output with EXIF orientation reported (not applied) |
| `image` | [crates/image/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/ARCHITECTURE.md) — unified GL engine (one `GLProcessorST`; Linux DMA-BUF, macOS/iOS ANGLE + IOSurface, Android AHardwareBuffer), `GlPlatform` porting seam, EGL image cache, batch engine (`convert_deferred`/`flush`), G2D, CPU, Vivante workarounds, shutdown safety |
| `decoder` | [crates/decoder/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/ARCHITECTURE.md) — model-type selection, dshape contract, per-scale framework, fused proto path |
| `tracker` | [crates/tracker/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/ARCHITECTURE.md) — ByteTrack two-pass association, Kalman state |
| `tensor-capi` / `codec-capi` / `image-capi` / `decoder-capi` / `tracker-capi` | Five modular C libraries (`libedgefirst_{tensor,codec,image,decoder,tracker}`). Detection layouts live in header-only `edgefirst/detect.h`. |
| `python-*` (five wheels) | [crates/python-common/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-common/ARCHITECTURE.md) — PyO3 bindings, five extension modules linking `libedgefirst_tensor.so` (tracker does not), PEP 420 namespace, capsule handoff |

The high-level system diagram lives at the top of
[README.md § System Architecture](https://github.com/EdgeFirstAI/hal/blob/main/README.md#system-architecture);
this document does not reproduce it.

---

## Per-Crate Summary

Each sub-crate has a single responsibility in the inference pipeline:

- [`edgefirst-tensor`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/) — the foundation. Provides `Tensor<T>` and `TensorDyn` with interchangeable backends — the `TensorMemory::DmaBuf` zero-copy slot maps to DMA-BUF on Linux, IOSurface on macOS/iOS, and AHardwareBuffer on Android, alongside SHM / Mem / PBO — plus multi-plane composition for V4L2 NV12M, the `BufferIdentity` cache key (interned on `AHardwareBuffer_getId` on Android), the required `CpuAccess` declaration and tile-compression metadata on image tensors, and the `PboOps` trait that lets the GL backend manage PBO lifetimes through a `WeakSender` channel.
- [`edgefirst-codec`](https://github.com/EdgeFirstAI/hal/blob/main/crates/codec/) — Image decoding (JPEG, PNG) into pre-allocated tensor buffers with support for u8, u16, i8, i16, and f32 pixel types. Supports strided output for GPU pitch-aligned DMA-BUF/PBO tensors. Designed for the allocate-once, decode-in-loop pattern.
- [`edgefirst-image`](https://github.com/EdgeFirstAI/hal/blob/main/crates/image/) — the GPU/G2D/CPU image processor. Owns the GL thread, EGL image caches, and shutdown defense layers. Provides format conversion, geometric transforms, and three mask-rendering pipelines (materialized, fused proto, tracked). The GL backend is a **single engine** (`GLProcessorST`) that runs on every supported OS: Linux uses native EGL + DMA-BUF import, macOS uses ANGLE + IOSurface, Android uses native EGL + AHardwareBuffer EGLImage import (iOS builds ride the ANGLE platform) — platform differences are confined to the `GlPlatform` compile-time porting contract (`gl/platform/`). Batch preprocessing is supported via `convert_deferred`/`flush`: sibling tiles share one EGLImage import (parent-keyed) and one GPU sync per batch. Also owns the input half of SAHI tiling (`tiling.rs`) — `TilingConfig`, `plan_tiles`, `alloc_tile_batch`, `tile_into`, `tile_one` — which rides that same batch engine to render an overlapping tile grid with one import and one flush.
- [`edgefirst-decoder`](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/) — model output post-processing. YOLOv5/v8/v11/v26 (incl. end-to-end) and ModelPack. NEON-optimized per-scale split-tensor framework. Validates `shape` / `dshape` declarations against the physical-memory-order contract at builder time. Owns the output half of SAHI tiling (`tiling` module) — `TilePlacement` (the record shared with `edgefirst-image`), `lift_tile_boxes`, greedy `merge_tiled_detections` (keep-best suppression by default, enclosing union opt-in via `MergeMode`), and the streaming `TiledFrameAccumulator`.
- [`edgefirst-tracker`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tracker/) — ByteTrack with Kalman-smoothed trajectories. Generic over the detection box type; the decoder's `DetectBox` plugs in via the `DetectionBox` trait.
- [`edgefirst-tensor-capi`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor-capi/) — C ABI for tensors (`libedgefirst_tensor`, `edgefirst/tensor.h`). Sibling leaves cover codec, image, decoder, and tracker. Detection layouts are header-only in `edgefirst/detect.h`.
- [`crates/python-common`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-common/) — the shared PyO3 binding rlib, plus four thin `cdylib` crates (`python-tensor`, `python-codec`, `python-image`, `python-decoder`) published as the independent `edgefirst-tensor` / `edgefirst-codec` / `edgefirst-image` / `edgefirst-decoder` wheels under the `edgefirst.` PEP 420 namespace. Contains the three-path numpy copy dispatcher.

There is no umbrella crate. `edgefirst-hal` was deleted in 0.29 — every
crate is a real, independently-consumable library, and the optional Chrome
JSON tracing subscriber moved into `edgefirst-tensor` behind its `tracing`
feature.

The internal dependency graph and external dependency list live in
[README.md § Dependencies](https://github.com/EdgeFirstAI/hal/blob/main/README.md#dependencies).

---

## Platform Support Matrix

The HAL spans embedded Linux, desktop Linux, macOS/iOS, and Android,
with different acceleration primitives per tier. The `TensorMemory`
enum is shared across all tiers (same discriminants over the C ABI);
the underlying storage and the GL transfer backend differ.

| Capability | Embedded Linux (i.MX, RPi5, Jetson) | Desktop Linux (x86_64) | macOS (Apple Silicon) | Android (API 26+) | Windows (x86_64) |
|------------|--------------------------------------|------------------------|------------------------|--------------------|-------------------|
| `TensorMemory::Mem` | Heap | Heap | Heap | Heap | Heap |
| `TensorMemory::Shm` | `shm_open` | `shm_open` | `shm_open` | Import-only — bionic has no `shm_open`, so allocation reports `NotImplemented`; `from_fd` works | — |
| `TensorMemory::DmaBuf` | DMA-BUF heap (`/dev/dma_heap/*`) | DMA-BUF heap if mountable; PBO otherwise | IOSurface (CoreFoundation framework) | AHardwareBuffer (NDK, gralloc) | — (D3D11 shared textures are a planned follow-on) |
| `TensorMemory::Pbo` | GLES PBO | GLES PBO | — (no PBO on the macOS backend) | — (AHB covers the zero-copy roles) | GLES PBO (the GPU destination kind) |
| GL transfer backend | `TransferBackend::DmaBuf` (Vivante, Mali, V3D) | `DmaBuf` or `Pbo` (NVIDIA discrete uses `Pbo`) | `IOSurface` via ANGLE | AHardwareBuffer EGLImage (native EGL) | `Pbo` via ANGLE |
| GL → backend translation | Native EGL → driver (vendor blob or Mesa) | Native EGL → driver | ANGLE EGL → Metal | Native EGL → driver (Adreno/Mali/PowerVR/Xclipse) | ANGLE EGL → Direct3D 11 |
| Hardware 2D blitter | G2D on NXP i.MX | — | — | — | — |
| Zero-copy import API | `EGL_EXT_image_dma_buf_import` | Same, when available | `EGL_ANGLE_iosurface_client_buffer` | `EGL_ANDROID_image_native_buffer` | — (`EGL_ANGLE_d3d_texture_client_buffer` reserved for the follow-on) |
| Cross-process buffer handle | DMA-BUF fd (over `SCM_RIGHTS`) | Same | IOSurfaceID (`u32` via Mach port or XPC) | `AHardwareBuffer` (Binder / `sendHandleToUnixSocket`) | — |
| Probe function | `is_dma_available()` | Same | `is_iosurface_available()` | `is_ahardwarebuffer_available()` | — (`false`; ask `ImageProcessor` whether GL is live) |
| Portable probe | `is_gpu_buffer_available()` — works on all four zero-copy tiers; `false` on Windows | | | | |

The portable `is_gpu_buffer_available()` is the recommended cross-platform
gate when the question is "can I ask for `TensorMemory::DmaBuf` and expect a
zero-copy GPU-importable buffer?" The platform-specific probes
(`is_dma_available`, `is_iosurface_available`) remain when callers need
to know *which* primitive is in use — e.g. to decide whether to call
`ef_tensor_clone_fd` (Linux) vs `ef_tensor_from_iosurface_id` / `ef_tensor_iosurface_ref` (macOS).

**Windows (x86_64)** runs the same GL engine on ANGLE's Direct3D 11 backend
(`crates/image/src/gl/platform/windows.rs`). No zero-copy buffer kind exists
there yet, so it behaves like desktop Linux on an NVIDIA discrete GPU: `Mem`
sources, `Pbo` destinations, `GL_PIXEL_PACK_BUFFER` readback. The adapter is
chosen with `EDGEFIRST_ANGLE_ADAPTER` (hardware / WARP / LUID / name match);
WARP is classified as a software renderer. D3D11 shared-texture tensors and
CUDA-via-D3D11 interop are a separate follow-on.

**iOS (16+)** shares the macOS column's architecture — ANGLE (EGL→Metal)
via the prebuilt xcframeworks and IOSurface-backed `TensorMemory::DmaBuf`
tensors. CI builds and lints the native Rust API for `aarch64-apple-ios`
and `aarch64-apple-ios-sim` on every PR; this repo's mobile
responsibility ends at that — link-closure validation against the ANGLE
+ Apple frameworks and runtime execution on-device belong to
`mobile-sdk`, which binds to these crates via boltffi.

**Image allocation on every tier declares CPU access.** `Tensor::image`
/ `ImageProcessor::create_image` take a required `CpuAccess`
(`None`/`Read`/`Write`/`ReadWrite`): hardware access is implied, CPU
mapping is the opt-in, and `None` keeps Android allocations eligible for
vendor tile compression. The `ImageDesc` builder additionally requests
compression metadata (`Compression::Any` / a specific scheme). This is a
cross-crate contract (tensor → image → capi → python); the normative
description lives in
[crates/tensor/ARCHITECTURE.md § CPU access declaration](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/ARCHITECTURE.md#cpu-access-declaration-cpuaccess).

### Float preprocessing capability

`ImageProcessor::supported_render_dtypes()` returns a `RenderDtypeSupport
{ f32, f16 }` struct after probing the GPU's float colour-buffer extensions
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
| Windows ANGLE / Direct3D 11 | PBO readback (gated on `GL_EXT_color_buffer_half_float`, probed at display init) | PBO readback (gated on `GL_EXT_color_buffer_float`) |
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
# use edgefirst_tensor::{PixelFormat, DType, CpuAccess};
# fn main() -> Result<(), edgefirst_image::Error> {
let proc = ImageProcessor::new()?;
let support = proc.supported_render_dtypes();
// Pick the best float dtype; fall back to U8 if the GPU cannot render floats.
let dst_dtype = if support.f16 { DType::F16 } else if support.f32 { DType::F32 } else { DType::U8 };
// memory: None → auto-selects float PBO when supported, else heap.
// access: None → hardware-only destination, the cheapest declaration.
// convert() always succeeds; GPU path used when available, CPU otherwise.
let mut dst = proc.create_image(
    640, 640, PixelFormat::PlanarRgb, dst_dtype, None, CpuAccess::None)?;
# Ok(())
# }
```

---

## C ABI Stability and Versioning

The five C libraries (`libedgefirst_{tensor,codec,image,decoder,tracker}`)
are independently versioned artifacts that a consumer links, ships and
upgrades separately. This section is the contract that governs them: what a
version number promises, what mixing versions is allowed to do, and which
mechanisms enforce it. It applies to every `-capi` crate; per-library detail
lives in each one's `ARCHITECTURE.md`.

### What a version number promises

| Change to a library's C surface | Pre-1.0 (`0.N.z`) | Post-1.0 (`X.Y.Z`) |
|---|---|---|
| **Breaking** — a struct layout moves, a symbol is removed or changes signature, or an existing call changes what it computes | minor `N` | major `X` |
| **Additive** — new symbols, or a struct extended safely (below) | minor `N` | minor `Y` |
| **No ABI impact** — bug fixes, performance work, internal refactors | patch `z` | patch `Z` |

**The patch guarantee holds in both eras.** ABI is stable across `z`: any
`0.29.z` is drop-in for any other `0.29.z`. A `0` major buys the right to
break across *minors*, not the right to break arbitrarily — so "pin the
version you built against", the advice in
[README.md](https://github.com/EdgeFirstAI/hal/blob/main/README.md) and
`packaging/c/README.md`, means pin the minor and take patches freely.

**Post-1.0 the majors do the work.** A break costs a major bump, so a
consumer that pins `1.x` is safe against every release that carries the `1`.
Extending a struct is explicitly a *minor* — it is not forbidden — provided
it is done safely in the sense defined below.

### Mixing versions across the five libraries

Because the libraries ship and upgrade independently, a deployment can end up
with a `libedgefirst_tensor.so` and a `libedgefirst_image.so` built from
different releases. Which pairs are compatible depends on the era, because
what a minor bump is permitted to do differs between them.

**Pre-1.0**, a minor may break ABI, so the incompatible boundary is the minor:

- *Same minor, any patch* — fully compatible. Must work with no degradation
  and no diagnostic.
- *Different minor* — not compatible, and must not be silently mis-executed.

**Post-1.0**, a minor is additive only, so the boundary moves to the major:

- *Same major, any minor or patch* — compatible. A 1.2 consumer against a 1.3
  library is fine by construction: everything 1.3 added is new surface the 1.2
  consumer never calls, and nothing it does call has moved.
- *Different major* — not compatible.

**What enforces it.** Post-1.0 the enforcement is structural rather than a
runtime check, and it lands exactly on the boundary the post-1.0 rule draws: a
breaking change costs a major bump, the SONAME carries the major, and the
dynamic loader refuses to bind a consumer built against
`libedgefirst_X.so.1` to `libedgefirst_X.so.2`. Nothing needs to be verified
at call time because the mismatched pair never links. Note that the SONAME
deliberately does *not* separate a 1.2 library from a 1.3 one — post-1.0 that
pair is compatible, so there is nothing to reject.

Pre-1.0 there is no such enforcement, and this document states that plainly
rather than implying a guarantee the code does not provide. Every library's
SONAME is `libedgefirst_X.so.0`, so the loader binds any 0.x to any other
0.x — including the 0.N/0.N+1 pair the pre-1.0 rule above calls incompatible
— and the `ef_*_abi_version()` probes below are consulted by nothing.
**Across a pre-1.0 minor the rule is therefore an obligation on whoever
deploys, not a runtime promise the libraries make**: ship and deploy the five
libraries as one set from one release, and pin the minor. That is what
"pin the archive version you built against" in `packaging/c/README.md` is
asking for, and it is the whole of the mitigation until the probes are wired.

**Why the SONAME carries the major only.** Each `build.rs` emits
`-Wl,-soname,libedgefirst_X.so.{major}`, matching the glibc/OpenSSL/zlib
convention: the SONAME is copied verbatim into every dependent's `DT_NEEDED`,
so embedding a minor or patch would force a downstream relink on every
release. That is why the loader can only ever police the major boundary —
which is the right boundary post-1.0 and the wrong one before it, as above.

**The `ef_*_abi_version()` probes are how that gap is meant to be closed.**
Every C library exports one (`ef_tensor_abi_version`, `ef_image_abi_version`,
`ef_codec_abi_version`, `ef_decoder_abi_version`, `ef_tracker_abi_version`).
It returns a monotonic ABI generation for that library, hand-maintained and
independent of the package version, and it **must be bumped whenever that
library's C surface changes in a way that is not backward compatible** — a
layout change, a removed or re-signatured symbol, or a change to *documented*
semantics that an existing caller would experience as a different contract.
The semantics-only case is the one most easily missed and the one that most
needs the signal: if a call keeps its name, its signature and its struct
layouts but computes something different, the consumer gets no link error, no
size mismatch and no loader diagnostic. The probe is the only thing left that
can tell them.

**A bug fix is not a probe bump**, even though a caller can observe it.
Bringing behaviour into line with the documented contract is what a patch
release is for, and it stays inside the patch row of the table above.
Advancing the generation for it would be actively harmful: a consumer doing
an exact-equality probe check would start rejecting patch releases it should
accept, which is the opposite of what the probe is for. The test is whether
the *documented* contract changed, not whether any observable byte did.

> **Not yet enforced.** Nothing in this tree compares a probe to anything.
> All five are defined, declared in their headers, and called only by the
> per-crate `test_double_include.c` link smoke tests. The "detect and report"
> half of the mixed-version rule above is therefore **unimplemented**: today
> a minor-skewed pair links cleanly and misbehaves silently. See the comment
> on `TensorDyn::compression` in
> [`crates/tensor/src/tensor_dyn/dynamic_backend.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/tensor/src/tensor_dyn/dynamic_backend.rs),
> which documents the same gap from the consumer side — an unrecognised wire
> value is logged rather than refused, because "there is no ABI-version
> negotiation to refuse it first". Closing this means each library checking
> its dependencies' probes once at initialization and failing loudly on a
> mismatch. It must be closed before 1.0, when the probes become the
> contract rather than a convention.

### Evolving a by-value struct

By-value structs (`ef_merge_config`, `ef_tensor_view`, `ef_tile_placement`,
`ef_detect_box`, …) are the hard case: unlike an opaque handle, their size
and field offsets are baked into every consumer's call site the moment it
compiles. Two independent properties decide whether a field can be added in
place.

**Size-safety.** A caller allocates the struct and the library reads it, so
the library must never read past what the caller allocated. Adding a field
that fits in *existing tail padding* keeps `sizeof` and every earlier offset
unchanged, so an old caller's allocation is still exactly the right size.
Adding a field that grows `sizeof` is not size-safe and cannot be done in
place.

**Value-safety.** Size-safety is not sufficient. C guarantees nothing about
the contents of padding bytes: a caller that filled the struct field by field
never initialized the tail pad, so a library that starts reading it reads
garbage. A field added into padding is therefore only safe when the consumer
is *known* to have been built against the newer definition. Post-1.0 a major
bump establishes that through the SONAME, and the loader enforces it. Pre-1.0
nothing establishes it, so the requirement is waived rather than met — see
the rule below.

This yields the rule:

- **Pre-1.0, in place is a minor bump.** A field may be added in place when
  it is size-safe. Value-safety is not achieved — an older caller's padding is
  still unwritten — it is *waived*, because a pre-1.0 minor is permitted to
  break and consumers are required to move as a set. Bump
  `ef_*_abi_version()` so a consumer that does check has a signal, and say in
  the CHANGELOG that the release is not drop-in, rather than letting the
  unchanged `sizeof` imply that it is.
- **Post-1.0, in place is a major bump.** Padding reuse cannot deliver
  value-safety, so the change is breaking, and breaking costs a major. That
  is not a hardship — it is the mechanism: the major moves the SONAME, the
  loader refuses every stale consumer, and the mismatch is caught before a
  single field is read.
- **Post-1.0, to extend without a major, add a suffixed successor type**
  (`ef_tensor_view2`) plus the entry points that take it, and leave the
  original untouched. Nothing existing moves, so every current consumer keeps
  working unmodified — which is precisely what makes the change additive and
  therefore a minor. The cost is real and permanent: each successor doubles
  the entry points that take that struct. Weigh that against a major bump
  rather than reaching for it reflexively.

**Why there is no `struct_size` handshake.** A first-member
`uint32_t struct_size` that the caller sets — Linux's `copy_struct_from_user`
convention (`clone_args`, `sched_attr`), Win32's `cbSize` — is the one
mechanism that would let a struct grow *in place* within a minor post-1.0.
It was considered for this codebase and **declined**, because it only works
as a convention and this vocabulary cannot adopt it as one. Of the fourteen
by-value structs across the five libraries, only three are caller-filled
configuration where the handshake means anything:

| Class | Structs | Verdict |
|---|---|---|
| Caller-filled config | `ef_merge_config`, `ef_tiling_config`, `ef_crop` | the handshake would work |
| Library-filled `*out` | `ef_tensor_view`, `ef_tensor_plane`, `ef_image_desc_view`, `EfViewOrigin`, `ef_quantization_info`, `ef_tile_spec`, `ef_decoder_track`, `ef_track_info` | inverts the contract — the caller would be declaring its buffer size, paid per call on hot accessors |
| Arrayed elements | `ef_detect_box`, `ef_segmentation`, `ef_tile_placement` | actively harmful |

The third class settles it. These travel as contiguous blocks —
`ef_detect_box_list_data()` hands back a packed array a consumer can `memcpy`
or wrap zero-copy — so a per-element size field would add four identical
redundant bytes per element (about 1.2 KB on a 300-box frame) and break that
contract outright. A convention that cannot cover the vocabulary is not a
convention, and applying it to `ef_merge_config` alone would leave
`ef_tiling_config` — same shape, same hazard, same library family — without
it. That asymmetry is the kind that produced a successor struct in the first
place.

`ef_crop` is the sharpest single argument: it documents a zeroed struct as
meaning "the whole source, same as passing `NULL`". A mandatory size field
would turn `{0}` from a useful default into an error.

So by-value structs here evolve by major bump or by successor type, and the
SONAME is the handshake. Revisit this only if the config structs ever
outnumber the data structs, and revisit it for all three at once.

**Layout goldens pin all of this — where they exist.** A
`tests/c/test_layout_goldens.c` is a set of `_Static_assert`s on `sizeof` and
`offsetof` for a library's by-value structs, compiled by a Rust test, backed
by matching `offset_of!` assertions on the Rust side in the `-abi` crate. A
failure means a layout moved; the fix is to decide whether that was intended
and, if so, to carry the version bumps this section requires. Coverage is
currently partial:

| Library | By-value structs in its header | Layout goldens |
|---|---|---|
| `tensor` | `ef_tensor_view`, `ef_tensor_plane`, `ef_image_desc_view`, `ef_quantization_info`, … | yes |
| `decoder` | `ef_detect_box`, `ef_segmentation`, `ef_merge_config`, `ef_tile_placement` (via `detect.h`) | yes |
| `image` | `ef_crop`, `ef_tiling_config`, `ef_tile_spec` | **none — gap** |
| `tracker` | `ef_track_info` | **none — gap** |
| `codec` | none (opaque handles only) | not needed |

The two gaps are the same drift class the goldens exist to catch, in headers
that have it: `image` and `tracker` should gain goldens before 1.0, when a
layout move stops being a permitted minor-bump event and becomes a major one.
`crates/tensor-capi/tests/check_abi.rs` covers the complementary drift class
— the exported `ef_*` symbol set, the `DECLARED` list in `tensor-ffi`, and
the header declarations must agree exactly.

**Worked example.** `ef_merge_config` shipped without a `mode` field in every
0.29 release. 0.29.0 through 0.29.4 each held its layout and its
`ef_decoder_abi_version` unchanged, exactly as the patch guarantee requires,
so the compatibility boundary is 0.29.4 → 0.30.0 and not any boundary inside
the 0.29 series. The field then went into the 4-byte tail pad the struct
already had: `sizeof` stayed 32 and no earlier offset moved, so the change is
size-safe — a caller built against any 0.29 header still allocates exactly the
right number of bytes. It is *not* value-safe: that caller never wrote the
pad, so a 0.30 library reading `mode` from it reads whatever was on the stack.

Both facts together are why the change takes a **minor** bump to 0.30.0. The
patch guarantee would have forbidden it outright, which is why none of 0.29.2,
0.29.3 or 0.29.4 could have carried it; the minor is what licenses the break
and what tells a consumer not to mix. `ef_decoder_abi_version` went to `2` as
well, even though no layout changed, because the default tiled merge changed
from the enclosing union to keep-best suppression — the semantics-only case
above, where the probe is the consumer's only signal. Note which side of the
bug-fix line that falls on: the union was the *documented* behaviour, so
replacing it changes the contract rather than correcting a deviation from it.
Had the union merely been a bug against a keep-best specification, the fix
would have been a patch with no probe bump.

Note what the mismatch does *not* do: `mode_from()` rejects any value that is
not `0` or `1`, so a garbage pad usually yields a refused call. "Usually" is
the problem — fresh stack and `calloc` memory reads as `0`, which is a
*valid* mode, so the same mismatch silently selects keep-best on one run and
refuses on another. That is the mixed-version gap above in miniature, and the
reason the probe check has to be real rather than emergent.

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
| C | `ef_is_cuda_available()` | `ef_tensor_cuda_map()` → `ef_tensor_cuda_device_ptr()` → `ef_tensor_cuda_unmap()` | opaque handle |
| Python | `edgefirst.tensor.is_cuda_available()` | `Tensor.cuda_map() -> CudaMap | None` | context manager — `.device_ptr`, `.size` |

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

### Three batch memory representations, reconciled

Three conventions for addressing a batch coexist across HAL and its consumers, and nothing previously stated how they relate.

1. **Engine batch inputs — `dst.batch(n)`.** The convention above: `N` the leading dimension over a packed `[N, H, W, C]` or planar `[N, C, H, W]` tensor, addressed by batch index. For a packed destination, `Tensor::batch(n)` (`crates/tensor/src/lib.rs`) treats the underlying buffer as a tall `(W, N·H)` parent and composes a `view_origin` so the GL backend imports it once and renders tile `n` as a `glViewport` row-band at `y = n·H` — the mechanism the "Convert → tile" step above relies on. Planar/semi-planar tensors take the plain per-element subview instead, with no composed `view_origin`; batching a non-packed layout stays on the per-slot path.
2. **Tile rendering — `dst.view(Region)`.** `ImageProcessor::tile_into`'s destination is expected to come from `alloc_tile_batch`, which builds the same tall `[tile_w, N·tile_h]` parent shape as (1) but through `create_image` — so DMA pitch alignment applies — rather than through a pre-shaped batched tensor. `render_tile` addresses tile `index` as a spatial region view, `dst.view(Region::new(0, index·tile_h, tile_w, tile_h))`, of that parent when the destination has room for the whole tile count; a single-slot `tile_one` destination without that capacity is converted into directly instead of viewed. So (1) and (2) share the same tall-packed-buffer physical layout — batch index and region view are two ways of reaching the same row-band — but (2) is reached through the general region-view API rather than `batch(n)`, and always goes through `create_image`'s pitch alignment.
3. **Profiler device-batch mode — `plane_offset = e × input_frame_size`.** An external convention used by the EdgeFirst Studio profiler's device batch invoke path, documented in the profiler repository's `SAHI.md` §D2: the accelerator addresses element `e` of a batch by a fixed per-element frame-size offset rather than through either of the above. It is not part of this repository's tensor or tiling API and is out of scope for HAL's memory-layout guarantees — a caller building a batched invoke against it must not assume `dst.batch(n)` or `dst.view(Region)` semantics apply.

Two caveats from `SAHI.md`'s audit of the tall batch apply to (1) and (2), since they share physical layout, and are carried over verbatim:

> If `row_stride > tile_w · channels`, the buffer is **not** a dense `[N, tile_h, tile_w, C]` tensor and no engine can consume it directly. (§D1)

> HAL's root `ARCHITECTURE.md` notes the one-import batch engine covers "the single-pass `Rgba`/`Bgra`/`Grey` u8/i8 DMA path", with two-pass packed-RGB, planar, and macOS falling back to eager per-tile converts. (§D5)

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
and the `ef_decoder_params` C struct. Builders enforce invariants in
`.build()` rather than scattering checks across setters.

### 4. Zero-copy operations

Used pervasively to avoid per-frame allocations:

- Memory-mapped hardware buffers (DMA-BUF, SHM, IOSurface, AHardwareBuffer)
- `&[T]` slice views into tensor maps
- ndarray `ArrayView` for math operations
- tokio's `WeakSender<T>` for cross-thread channels that should not extend lifetime

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

The C API translates all errors into POSIX `errno` codes; see each leaf's
header (`edgefirst/tensor.h`, `codec.h`, `image.h`, `decoder.h`, `tracker.h`).

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

Span names are dotted and hierarchical:
`<crate>.<operation>[.<backend>][.<sub-step>]`. The leading segment is the
emitting crate, so a trace can be filtered to one layer without knowing the
individual operation names. The same span name is used on every OS — the
platform and dispatch target are recorded as *fields*, not baked into the
name, so an `image.convert.gl` slice is directly comparable across Linux,
macOS, and Android.

| Segment | Meaning | Examples |
|---------|---------|----------|
| `codec.` | Image decode | `codec.decode_jpeg`, `codec.decode_jpeg.mcu_loop`, `codec.decode_png.zune_decode` |
| `tensor.` | Tensor lifecycle | `tensor.alloc`, `tensor.map` |
| `image.` | Image processing entry points | `image.convert`, `image.flush`, `image.gl_init`, `image.materialize_masks`, `image.draw_decoded_masks` |
| `image.convert.<backend>` | Backend that serviced the convert | `image.convert.cpu`, `image.convert.gl`, `image.convert.g2d` |
| `image.convert.<backend>.<step>` | Sub-step within a backend | `image.convert.cpu.format_convert`, `image.convert.gl.egl_import`, `image.convert.gl.pack_rgb.pass1_rgba`, `image.convert.gl.pack_rgb.pass2_pack` |
| `image.plan_tiles` / `image.tile_into` / `image.tile_one` | SAHI tiling input side | fields `tiles`, `overlap`, `index`, `count` |
| `decoder.` | Model output post-processing | `decoder.decode`, `decoder.nms_get_boxes`, `decoder.decode_proto`, `decoder.per_scale_run` |
| `decoder.<op>.<step>` | Sub-step within a decode phase | `decoder.nms_get_boxes.score_filter`, `decoder.nms_get_boxes.suppress`, `decoder.decode_proto.extract_proto_data` |
| `decoder.tiled.` | SAHI tiling output side | `decoder.tiled.lift`, `decoder.tiled.merge` |
| `tracker.` | ByteTrack association | `tracker.update`, `tracker.update.match_high_conf`, `tracker.update.predict` |
| `python.` | Python binding entry point | `python.convert`, `python.decode`, `python.tile_into`, `python.materialize_masks` |

Every span site uses `tracing::trace_span!`, except `image.gl_init`, which
uses `info_span!` so GL bring-up stays visible at a coarser filter level.

Field conventions:

- `n` or `n_*` — counts (detections, candidates, tracks)
- `mode` — algorithm variant (float / quant, proto / scaled)
- `*_fmt` — pixel format enum value
- `*_memory` — tensor memory backend (`Dma` / `Shm` / `Mem`)
- `layout` — data layout (`nhwc` / `nchw`)
- `pass` — multi-pass identifier (`pre_resize` / `post_resize` / `direct`)
- `platform` — `"linux"` / `"macos"` / `"ios"` / `"android"` — emitted by spans in the GL platform layer
- `backend` — on `image.gl_init`, the chosen transfer backend (`"dmabuf"` / `"iosurface"` / `"ahardwarebuffer"` / `"pbo"` / `"sync"`)
- `tiles`, `index`, `count`, `overlap` — SAHI tiling geometry on the `image.*_tile*` and `decoder.tiled.*` spans

Each per-crate `ARCHITECTURE.md` documents the spans that crate emits.

### Crate layering

```text
┌─────────────────────────────────────────────────────────┐
│                    Application Code                      │
├─────────────────────────────────────────────────────────┤
│  edgefirst-tensor::trace                                 │
│  (subscriber install, start/stop API)                    │
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
- **`edgefirst-tensor::trace`** gates `tracing-chrome` and
  `tracing-subscriber` behind the `tracing` feature (default on). These
  provide the capture infrastructure — the subscriber that actually
  writes the Chrome JSON file. It lived in the `edgefirst-hal` umbrella
  crate until 0.29; with that crate deleted, the subscriber moved down to
  the one crate every other crate already depends on.
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
image.convert
└─ image.convert.cpu.format_convert (pass="pre_resize", from=Nv12, to=Rgb)
└─ image.convert.cpu.resize_flip_rotate
└─ image.convert.cpu.format_convert (pass="post_resize", from=Rgb, to=Rgba)
```

OpenGL 2-pass packed RGB:

```text
image.convert
└─ image.convert.gl
   └─ image.convert.gl.pack_rgb.pass1_rgba (dst_w=640, dst_h=480)
   └─ image.convert.gl.pack_rgb.pass2_pack (render_w=640, render_h=480)
```

OpenGL 2-pass NV → Planar (the Vivante workaround, and the general
texture-lowered planar destination path):

```text
image.convert
└─ image.convert.gl
   └─ image.convert.gl.nv_to_planar.pass1_rgba (dst_w=640, dst_h=480)
   └─ image.convert.gl.nv_to_planar.pass2_deinterleave (dst_w=640, dst_h=480)
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
│   ├── tensor-capi/        # libedgefirst_tensor (C ABI; workspace-excluded)
│   ├── codec-capi/         # libedgefirst_codec
│   ├── image-capi/         # libedgefirst_image
│   ├── decoder-capi/       # libedgefirst_decoder
│   ├── tracker-capi/       # libedgefirst_tracker
│   ├── python-common/      # shared PyO3 binding code (rlib)
│   ├── python-tensor/      # -> edgefirst.tensor   wheel
│   ├── python-codec/       # -> edgefirst.codec    wheel
│   ├── python-image/       # -> edgefirst.image    wheel
│   ├── python-decoder/     # -> edgefirst.decoder  wheel
│   ├── egl/                # edgefirst-egl (trimmed khronos-egl fork, dynamic load only)
│   ├── gl/                 # edgefirst-gl (trimmed gls fork)
│   ├── bench/              # edgefirst-bench (workspace dev-dep)
│   └── gpu-probe/          # internal CLI for GPU capability probing
├── tests/                  # Project-level Python tests (C tests live under crates/*-capi/tests/)
├── testdata/               # Git LFS-tracked fixtures (images, model outputs)
├── benchmarks/             # Per-platform benchmark JSON results
├── scripts/                # Build / audit / release tooling
├── .github/workflows/      # CI: test.yml, release.yml, tag-release.yml, benchmark.yml, sbom.yml
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
key for imported tensors (`ef_tensor_builder_wrap`, EGL image creation, etc.).
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
is a hash-table probe; only the import path (`ef_tensor_builder_wrap`) is
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
**tensor's** `BufferIdentity.id`, which is derived from the buffer's
system key — a DMA-BUF's `(st_dev, st_ino)`, an `IOSurfaceID`. Two
tensors built independently over one buffer therefore derive the *same*
id, so a pipeline that re-imports its camera buffers every frame
(`ef_tensor_builder_wrap()`, or a `convert()` that
crosses a package boundary) keys onto the entry the previous frame
created instead of missing on every call.

Cache entries are deliberately **retained past the lifetime of the
tensor that produced them**, because a re-imported tensor is
constructed, used and dropped within the call: an entry tied to that
tensor would be destroyed before the next frame could hit it. Retention
is safe — and is what closes the key-reuse hazard — because the cached
entry owns the platform import object (an `EGLImage`, an ANGLE
IOSurface pbuffer), which holds its own driver-side reference to the
buffer. The buffer therefore cannot be freed, and its system key cannot
be recycled onto a different buffer, while an entry that names it
lives. Entries leave only by LRU eviction.

#### Sizing the cache

Capacity comes from `ImageProcessorConfig::egl_cache_capacity`, then
`EDGEFIRST_EGL_CACHE_CAPACITY`, then a default of 16 — config first so a
stray environment variable in a deployment cannot override an embedder
that measured its own pool.

The bound is **per `ImageProcessor`, times three caches** (source,
destination, and the NV R8 source cache) — not per process and not per
library. Two `ImageProcessor`s in one process have six independent
caches and six independent capacities.

It is a buffer **count**, not a size, and because a retained import pins
its buffer the memory it holds depends entirely on how big the caller's
buffers are:

```text
worst-case pinned bytes
    = capacity x 3 caches x frame size x live ImageProcessors

capacity 16, 640x640 RGB   (1.2 MB)  ->   59 MB per processor
capacity 16, 1080p NV12    (3.1 MB)  ->  149 MB per processor
capacity 16, 4K NV12      (12.4 MB)  ->  597 MB per processor
```

Those are worst cases, and whether one is approached is up to the
caller. A pooled producer — V4L2 capture, typically 4–8 buffers over one
or two sizes — settles at its pool size and never nears the bound; a
stream that allocates a fresh buffer per frame holds the last `capacity`
of them outright. That is the difference between "16 is free" and "16 is
597 MB". (The three-cache multiplier is an upper bound rather than an
exact total: the source and NV R8 caches often hold two imports of the
*same* buffer, and two entries naming one buffer pin it once.)

Size from measurement, not guesswork: `CacheStats::peak_entries` reports
the working set actually reached, and `CacheStats::evictions` says when
the bound is costing re-imports — it should be zero in a steady state.

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
`ef_tensor *` objects keyed by `(inode, offset)`, and never free them
between frames. Holding the tensor alive keeps its `BufferIdentity`
stable, which keeps the in-HAL EGL image cache hitting. This ensures
`ef_tensor_builder_wrap` is called exactly once per unique DMA-BUF over the
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

ef_tensor *tensor = g_hash_table_lookup(input_cache, &key);
if (!tensor) {
    // First time seeing this buffer — import and cache
    tensor = ef_tensor_builder_wrap(builder /* planes already set */);
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
