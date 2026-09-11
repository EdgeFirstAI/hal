// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Cross-platform seam for the OpenGL backend.
//!
//! Linux uses GBM + EGL + DMA-BUF; macOS uses ANGLE + IOSurface. The two
//! platforms have very different EGL bring-up and buffer import flows —
//! the macOS path goes through `eglGetPlatformDisplayEXT` (ANGLE Metal
//! display) and `eglCreatePbufferFromClientBuffer` (IOSurface import),
//! while Linux goes through the standard EGL display path and
//! `eglCreateImageKHR` (DMA-BUF import).
//!
//! [`GlPlatform`] is the compile-time porting contract: exactly one
//! implementation is selected per build via the [`Platform`] type alias
//! (static dispatch — no vtable on the per-frame path, no type-parameter
//! infection of the processor or dispatch wrapper). The portable engine
//! reaches platform buffers only through this trait; a new platform
//! implements the trait or does not compile — it cannot fork convert
//! logic. Windows (ANGLE over Direct3D 11, `windows.rs`) landed exactly
//! that way: a leaf with PBO transfers and no zero-copy import yet.
//!
//! The trait grows with the convergence steps: today it covers display
//! bring-up; the buffer-import methods land when the portable engine's
//! import path routes through it (PR-A step A3), and the macOS
//! implementation (`angle.rs`) lands at step A4. The Linux platform
//! helpers it delegates to live in [`super::context`] and
//! [`super::dma_import`].
//!
//! [`PlatformCaps`] is the capability surface the portable code keys
//! decisions on (serialization policy, transfer backend, float render
//! support) — platform differences surface as caps bits feeding pure
//! decision tables, never as new `cfg` branches in the engine. Caps are
//! captured ONCE per processor at worker startup, never per message.

#[cfg(target_os = "android")]
pub(super) mod android;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(super) mod angle;
#[cfg(target_os = "linux")]
pub(super) mod linux;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(super) mod macos;
#[cfg(target_os = "windows")]
pub(super) mod windows;

use super::EglDisplayKind;
use edgefirst_tensor::{PixelFormat, Tensor, TensorDyn};

/// Capability surface a platform reports for one initialized display +
/// context. Captured once at processor/worker construction (see
/// `threaded.rs` — the worker reads it before entering its message loop)
/// and treated as immutable for the processor's life.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlatformCaps {
    /// Active pixel-transfer method (DMA-BUF / IOSurface / PBO / Sync).
    pub(crate) transfer_backend: super::TransferBackend,
    /// Float render-target support (F32/F16 color attachments), already
    /// adjusted for driver quirks (e.g. Vivante's pathological float
    /// readback reports `false`).
    pub(crate) render_dtypes: crate::RenderDtypeSupport,
    /// Whether GL command submission must be serialized process-wide
    /// (one message at a time across ALL processors). `true` only for
    /// Vivante/galcore, which is not thread-safe for concurrent GL
    /// across contexts; everywhere else lifecycle-only locking applies
    /// and processors run in parallel. See the `GL_MUTEX` doc comment
    /// in `context.rs` for the full policy table.
    pub(crate) serialize_gl: bool,
    /// Whether `GL_TEXTURE_EXTERNAL_OES` sampling of multi-plane imports
    /// is available (the Linux NV "Path A"). ANGLE/Metal has no external
    /// sampler — NV sources there always take the single-plane R8
    /// shader path (`import_buffer_nv_r8`).
    /// Consumed when the engine's source-sampling selection runs on
    /// macOS (PR-A step A7); until then only constructed.
    #[allow(dead_code)]
    pub(crate) external_oes: bool,
    /// Whether `convert_with_fence` can export a real native fence fd
    /// (`EGL_ANDROID_native_fence_sync` on this display). When false the
    /// fenced entry points silently take the blocking path and return
    /// no fd.
    pub(crate) native_fence_sync: bool,
}

/// Platform-neutral identity of a "packed" render surface: the float
/// paths render planar/RGB byte streams through an RGBA-shaped surface
/// whose pixel count encodes the byte layout (the caller computes the
/// surface dims). Linux maps these to DRM fourccs, macOS to IOSurface
/// pixel layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PackedImportFormat {
    /// 4 bytes/pixel RGBA8 (Linux `DrmFourcc::Abgr8888`).
    Rgba8888,
    /// 8 bytes/pixel RGBA16F (Linux `DrmFourcc::Abgr16161616f`).
    Rgba16161616F,
    /// 16 bytes/pixel RGBA32F (Windows `R32G32B32A32_FLOAT`).
    Rgba32323232F,
}

impl PackedImportFormat {
    /// Bytes per packed surface pixel.
    pub(crate) fn bytes_per_pixel(self) -> usize {
        match self {
            PackedImportFormat::Rgba8888 => 4,
            PackedImportFormat::Rgba16161616F => 8,
            PackedImportFormat::Rgba32323232F => 16,
        }
    }
}

/// A destination whose bytes start at a plane offset the engine cannot lower
/// to a viewport: one rebuilt from a descriptor, which carries the offset but
/// not the `view_origin` a `view()` would have given it.
///
/// Shared by the platforms whose import binds a whole buffer from its origin.
/// ANGLE over a D3D11 texture and ANGLE over an IOSurface both do, and neither
/// extension has a byte-offset attribute, so each one's `dst_import_places`
/// asks the same question of the same two fields. Linux's DMA-BUF import can
/// express the offset and keeps the default.
///
/// Gated to its callers' platforms: `angle` and `windows` are the only
/// modules that use it, and both are `cfg`-gated, so on Linux and Android it
/// would be dead code and fail those lanes under `-D warnings`.
#[cfg(any(target_os = "windows", target_os = "macos", target_os = "ios"))]
pub(super) fn unplaced_destination<T>(img: &Tensor<T>) -> Option<usize>
where
    T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
{
    match (img.plane_offset(), img.view_origin()) {
        (Some(offset), None) if offset != 0 => Some(offset),
        _ => None,
    }
}

/// Refuses to import a source whose pixels do not start at the buffer's
/// origin.
///
/// Shared by the platforms whose import binds a whole buffer from its
/// origin: ANGLE over a D3D11 texture (`EGL_ANGLE_image_d3d11_texture`) and
/// ANGLE over an IOSurface (`EGL_ANGLE_iosurface_client_buffer`). Neither
/// extension has a byte-offset attribute, and the engine samples a source
/// from the origin of its import, so a `view()` -- or a whole tensor at a
/// foreign offset -- attached that way converts the parent's top-left tile
/// in place of the region it names, silently. Declining sends the source
/// through `map()`, which starts at the offset: the engine's texture upload
/// for a packed source into a packed destination, and the R8 upload of the
/// combined plane for a single-plane NV source, whose import failure arms took
/// over from `draw_src_texture` (which has no NV case) in #166. What is left
/// on `ImageProcessor::convert`'s CPU fallback is a planar destination's GL
/// lowering and a true-multiplane NV12, which cannot be uploaded as one R8
/// texture; a forced OpenGL backend returns the refusal to the caller.
/// Issue #161 at the convert level.
/// Linux's DMA-BUF import expresses the offset itself
/// (`EGL_DMA_BUF_PLANE0_OFFSET_EXT`) and does not need this.
///
/// `binding` names the extension in the message, so a log line says which
/// leaf declined.
#[cfg(any(target_os = "windows", target_os = "macos", target_os = "ios"))]
pub(super) fn refuse_offset_source<T>(
    img: &Tensor<T>,
    what: &str,
    binding: &str,
) -> crate::Result<()>
where
    T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
{
    match img.plane_offset() {
        None | Some(0) => Ok(()),
        Some(offset) => Err(crate::Error::NotSupported(format!(
            "GL convert: {what} starts {offset} bytes into its buffer, and {binding} \
             can only bind the whole buffer from its origin; uploading the window \
             instead"
        ))),
    }
}

/// The compile-time platform contract for the portable GL engine.
///
/// One implementation per OS, selected by the [`Platform`] alias. Methods
/// are associated functions (no `&self`) — the platform is stateless; all
/// state lives in the `Display` it creates.
pub(super) trait GlPlatform {
    /// Owning handle for the platform's GL/EGL bring-up state: display,
    /// context, capability probes. On Linux this is
    /// [`super::context::GlContext`]; on macOS (step A4) the per-processor
    /// ANGLE context over the shared Metal display.
    type Display;

    /// Owned zero-copy buffer import: an `EGLImage` over a DMA-BUF on
    /// Linux; an EGL pbuffer over an IOSurface on macOS (step A4). The
    /// import cache stores these; Drop releases the platform object.
    type Import;

    /// `Copy` handle to a cached import, safe to pass around while the
    /// cache owns the import object: `egl::Image` on Linux,
    /// `egl::Surface` (the pbuffer) on macOS.
    type ImportHandle: Copy;

    /// Whether [`Self::attach_tex_image_2d`] bindings persist on the GL
    /// texture object across GPU passes. Linux EGLImage targets persist
    /// (enabling the binding-skip cache keyed by `BufferImportKey`);
    /// macOS `eglBindTexImage` bindings are released at the end of each
    /// synced pass ([`Self::end_gpu_pass`]) per the EGL pbuffer contract,
    /// so the skip cache must stay cold there. Windows/ANGLE is a third
    /// `false` for a third reason: ANGLE does not observe writes made to
    /// the D3D11 texture outside GL, so a persisted binding would serve a
    /// stale frame.
    const PERSISTENT_TEX_BINDINGS: bool;

    /// Which zero-copy float render paths this platform is known to serve.
    ///
    /// The classifier ([`super::float_dispatch::classify_float_render`])
    /// stays pure and takes this as an input rather than reading a `cfg`.
    /// Only [`ZeroCopyFloatSet::PlanarF16`] has ever run on Linux, macOS and
    /// Android, so those leaves report it and nothing more; the Windows leaf
    /// reports [`ZeroCopyFloatSet::All`], which is the set this branch
    /// validated on an RTX 3070 and on WARP. Widening a leaf is a decision
    /// backed by an on-target run, which is why it is a value here and not a
    /// property of the classifier.
    ///
    /// [`ZeroCopyFloatSet::PlanarF16`]: super::float_dispatch::ZeroCopyFloatSet::PlanarF16
    /// [`ZeroCopyFloatSet::All`]: super::float_dispatch::ZeroCopyFloatSet::All
    const ZERO_COPY_FLOAT: super::float_dispatch::ZeroCopyFloatSet;

    /// Whether `GL_TEXTURE_EXTERNAL_OES` sampling of imports exists on
    /// this platform (Linux: yes — the NV "Path A" and the legacy packed
    /// DMA source path; ANGLE/Metal: no — every import binds as
    /// `TEXTURE_2D`). Compile-time so the unsupported branch is
    /// statically eliminated.
    const EXTERNAL_OES: bool;

    /// Load the process-global GL function-pointer table exactly once.
    /// `edgefirst_gl` bindings are gl_generator `static mut` tables — loading must
    /// happen once per process, never per processor. Linux resolves via
    /// this display's `eglGetProcAddress`; macOS already loaded at
    /// shared-ANGLE-display init, so this is a no-op there.
    fn load_gl_once(display: &Self::Display);

    /// Bring up the platform display + context for one processor.
    /// `kind` selects the EGL display flavour on Linux and is ignored
    /// (with a debug log) on macOS, where ANGLE is the only display.
    fn init_display(kind: Option<EglDisplayKind>) -> crate::Result<Self::Display>;

    /// Import a tensor's zero-copy buffer, typed at `fmt`, for sampling
    /// (`for_dst = false`) or rendering into (`for_dst = true`). The
    /// distinction matters for views: a destination view imports its
    /// PARENT buffer (the tile offset becomes viewport state), a source
    /// view imports its own region. On Linux this is an `EGLImage` over
    /// the tensor's DMA-BUF (multi-plane NV12 and the 64-byte stride
    /// alignment invariant live in `dma_import.rs`); on macOS an EGL
    /// pbuffer over the tensor's IOSurface.
    ///
    /// Callers cache the result in [`super::cache::ImportCache`] keyed by
    /// [`super::cache::BufferImportKey`] — this is the miss path only.
    fn import_buffer(
        display: &Self::Display,
        img: &Tensor<u8>,
        fmt: PixelFormat,
        for_dst: bool,
    ) -> crate::Result<Self::Import>;

    /// Confirm this tensor's [`BufferIdentity`] still names the object the
    /// platform would import, before the import cache is consulted.
    ///
    /// Called on **every** import, hits included, which is the whole point:
    /// [`import_buffer`](Self::import_buffer) and friends run only on a miss,
    /// so a check made there cannot see a cached entry being served for a
    /// tensor whose identity has stopped matching its buffer. The cache keys
    /// on the identity and outlives the tensor that produced it, so an
    /// identity that names the wrong object is served a stale import with no
    /// error and no crash — wrong pixels, which is exactly the failure this
    /// exists to turn into a refusal.
    ///
    /// Default `Ok(())`: a platform whose identity is an OS-level key the
    /// kernel cannot recycle while the import holds it (a dma-buf inode, an
    /// `IOSurfaceID`) has nothing to re-check. Windows overrides it, because
    /// its identity is a raw `ID3D11Texture2D` address that the tensor crate
    /// derives independently in each of its two backends.
    ///
    /// An error refuses the zero-copy path and falls the frame back to the
    /// CPU converter, so a regression costs throughput instead of output.
    fn validate_import_identity<T>(_img: &Tensor<T>, _what: &str) -> crate::Result<()>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        Ok(())
    }

    /// Whether a zero-copy import of `img` as a destination can place the
    /// render where the tensor's bytes start.
    ///
    /// A `view()` destination carries a `view_origin`, and the engine
    /// imports its parent and lowers the tile to a viewport. A destination
    /// reconstructed from a descriptor carries only a `plane_offset`: no
    /// `view_origin` is transported, so the viewport is the surface's origin
    /// and the import itself has to start at the offset. Linux's DMA-BUF
    /// import does. A platform whose import always binds the whole buffer
    /// from its origin returns `false` for such a tensor, and the engine
    /// lowers it to the mapped texture path, whose readback writes through
    /// `map()` at the offset.
    ///
    /// Default `true`: every import that can express an offset.
    fn dst_import_places<T>(_img: &Tensor<T>) -> bool
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        true
    }

    /// Import an NV12/NV16/NV24 tensor's combined semi-planar plane as ONE
    /// R8 buffer (luma + interleaved chroma addressed by the shader — the
    /// "Path B" NV sampling strategy). On Linux a single-plane R8 EGLImage
    /// at the buffer's physical pitch; on macOS the same shape as an R8
    /// (`L008`) IOSurface pbuffer binding.
    fn import_buffer_nv_r8(
        display: &Self::Display,
        img: &Tensor<u8>,
        fmt: PixelFormat,
    ) -> crate::Result<Self::Import>;

    /// Import a tensor's zero-copy buffer as a packed RGBA-shaped render
    /// surface of `width`×`height` pixels (see [`PackedImportFormat`] —
    /// the float paths' RGB/planar byte streams rendered through RGBA
    /// pixels). The caller computes the packed surface dims; the
    /// platform derives pitch/offset from the tensor.
    fn import_buffer_packed<T>(
        display: &Self::Display,
        img: &Tensor<T>,
        width: usize,
        height: usize,
        fmt: PackedImportFormat,
    ) -> crate::Result<Self::Import>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element;

    /// The `Copy` handle for a cached import.
    fn import_handle(import: &Self::Import) -> Self::ImportHandle;

    /// The texel size of the texture an import covers, when that can be
    /// larger than the tensor's logical image; `None` when the import is
    /// always exactly the logical image.
    ///
    /// Linux reports the logical size unless the source was rebased onto an
    /// aligned DMA-BUF offset, where the import is `x_shift_px` texels wider
    /// and [`Self::import_origin`] says where the image starts on it; macOS
    /// gives the pbuffer explicit dimensions, so there the import *is* the
    /// logical image.
    /// `EGL_ANGLE_image_d3d11_texture` has no sub-extent attribute: a pool
    /// buffer narrowed by `configure_image` keeps its texture and imports all
    /// of it, so Windows reports the texture's texel size and the engine
    /// scales the source rectangle by `logical / extent`
    /// ([`super::render::scale_roi_to_import`]) and clamps every sample to the
    /// logical image ([`super::render::sample_clamp_rect`]). A leaf that
    /// returns `None` reaches both as the identity.
    ///
    /// Every sampling site clamps, the two `GL_TEXTURE_EXTERNAL_OES` camera
    /// programs (`draw_camera_texture_to_rgb_planar`,
    /// `draw_camera_texture_eglimage`) included: they now carry the same
    /// `src_extent` uniform the `sampler2D` programs do, which is what lets
    /// the Linux leaf report a narrowed extent and a nonzero
    /// [`Self::import_origin`] and still use them.
    fn import_extent(import: &Self::Import) -> Option<(u32, u32)>;

    /// Texels from the imported texture's origin to the tensor's own first
    /// pixel, when a leaf can start the logical image partway into the
    /// import; `(0, 0)` — the default — when the logical image always begins
    /// at the texture's origin.
    ///
    /// The companion of [`Self::import_extent`]: the extent says how much
    /// texture the import covers, this says where in it the logical image
    /// starts, and the engine folds both through one
    /// [`super::render::ImportMap`] at every site that turns logical-image
    /// coordinates into texture coordinates. Linux returns a nonzero origin
    /// for a source rebased onto a 64-byte-aligned DMA-BUF base, which the
    /// import then widens by the same shift (issue #170); every other leaf
    /// keeps the default.
    fn import_origin(_import: &Self::Import) -> (u32, u32) {
        (0, 0)
    }

    /// Attach the import as the image of the CURRENTLY BOUND
    /// `GL_TEXTURE_2D` texture object. Linux:
    /// `glEGLImageTargetTexture2DOES` (persists — see
    /// [`Self::PERSISTENT_TEX_BINDINGS`]); macOS: `eglBindTexImage`
    /// (recorded on the display and released by [`Self::end_gpu_pass`]).
    ///
    /// # Safety
    /// The intended texture must be bound on the active texture unit and
    /// the handle's import must be alive (cache-owned).
    unsafe fn attach_tex_image_2d(
        display: &Self::Display,
        handle: Self::ImportHandle,
    ) -> crate::Result<()>;

    /// Attach the import to the CURRENTLY BOUND
    /// `GL_TEXTURE_EXTERNAL_OES` texture (the Linux NV multi-plane
    /// sampling path). Errors on platforms without the OES extension —
    /// unreachable in practice because path selection consults
    /// [`PlatformCaps::external_oes`] first.
    ///
    /// # Safety
    /// As [`Self::attach_tex_image_2d`].
    unsafe fn attach_tex_image_external(
        display: &Self::Display,
        handle: Self::ImportHandle,
    ) -> crate::Result<()>;

    /// Attach the import as the storage of the CURRENTLY BOUND GL
    /// renderbuffer (the Linux Mali direct-RGB destination path, enabled
    /// by `EDGEFIRST_OPENGL_RENDERSURFACE`). Errors where renderbuffer
    /// import targets do not exist (macOS — the env knob has no effect
    /// there beyond this error).
    ///
    /// # Safety
    /// The intended renderbuffer must be bound and the handle's import
    /// alive.
    unsafe fn attach_renderbuffer_storage(
        display: &Self::Display,
        handle: Self::ImportHandle,
    ) -> crate::Result<()>;

    /// Called by the dispatch wrapper on the worker thread before each
    /// message is handled, after the serialization lock is taken. No-op
    /// on Linux, ANGLE/Metal and Android. ANGLE/D3D11 keeps one state
    /// manager per display and re-syncs per-context state only on
    /// `eglMakeCurrent`, so its implementation re-makes this processor's
    /// context current when another processor's context issued the
    /// previous GL commands; otherwise contexts alternating between
    /// threads, even fully serialized, render with the previous context's
    /// state.
    fn begin_gpu_pass(display: &Self::Display);

    /// Release every texture attachment recorded since the last call.
    /// MUST be called only after the GPU work consuming those
    /// attachments has been synced (`glFinish`/fence) — the engine's
    /// sync funnel (eager convert boundary, batch flush) is the call
    /// site. No-op on Linux (bindings persist by design).
    fn end_gpu_pass(display: &Self::Display);

    /// Whether [`Self::export_completion_fence`] can return a real fence
    /// on this display (Android with `EGL_ANDROID_native_fence_sync`;
    /// false on Linux/ANGLE). Callers use this to short-circuit to the
    /// blocking convert without a special message round-trip.
    fn native_fence_sync(display: &Self::Display) -> bool;

    /// Export a kernel sync-fence fd guarding every GL command submitted
    /// so far on the CURRENT context (the GL→NPU handoff — the consumer
    /// waits on the fd instead of the CPU blocking in `glFinish`).
    /// `Ok(None)` where native fence sync does not exist; the caller then
    /// falls back to the blocking sync. Must run on the GL worker thread.
    /// The handle type is the platform-neutral [`super::CompletionFence`]
    /// (an fd on Unix, an NT handle on Windows).
    ///
    /// `recorded` is the value [`Self::record_completion`] just recorded on
    /// the destination for this same convert, when the caller has one. A leaf
    /// whose completion fence *is* the device fence turns that value into the
    /// event instead of signalling a second one, so the returned event and
    /// `Tensor::gpu_completion` name the same point. A leaf whose fence is a
    /// GL native fence ignores it.
    fn export_completion_fence(
        display: &Self::Display,
        recorded: Option<u64>,
    ) -> crate::Result<Option<super::CompletionFence>>;

    /// Called by the engine after a convert or a mask draw into `dst` has
    /// been issued and, for the blocking path, completed. Platforms with a
    /// device fence record the value covering that work on the destination so
    /// GPU consumers can wait on it (`Tensor::gpu_completion`), and return it.
    /// No-op elsewhere, which answers `None`. Called after every convert,
    /// including deferred (batched) ones, and after every successful draw — a
    /// leaf that needs the GL work submitted flushes GL first, whatever
    /// `submit` says.
    ///
    /// `submit` is `false` while a deferred batch is open. The value is
    /// still allocated and its signal queued in order, but the device's
    /// command buffer is submitted once from [`Self::submit_recorded`] at the
    /// batch's flush rather than once per tile — N tiles would otherwise cost
    /// N submissions, which is exactly what batching exists to avoid. Every
    /// writer of the engine's `defer_finish` flag sets it inside a convert,
    /// so a draw never runs under one and always passes `true`.
    fn record_completion(display: &Self::Display, dst: &mut TensorDyn, submit: bool)
        -> Option<u64>;

    /// Submit the signals [`Self::record_completion`] queued with
    /// `submit == false`. Called once when a deferred batch flushes. No-op on
    /// a leaf that submits as it goes.
    fn submit_recorded(display: &Self::Display);
}

/// The one platform implementation for this build.
#[cfg(target_os = "linux")]
pub(super) type Platform = linux::LinuxEgl;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(super) type Platform = angle::AngleClientBuffer;
#[cfg(target_os = "android")]
pub(super) type Platform = android::AndroidEgl;
#[cfg(target_os = "windows")]
pub(super) type Platform = windows::AngleD3d11;

// Compile-time check that the selected platform implements the contract —
// a partial port fails here, not at a call site deep in the engine.
const _: fn() = || {
    fn assert_platform<P: GlPlatform>() {}
    assert_platform::<Platform>();
};

#[cfg(test)]
mod tests {
    use super::PackedImportFormat;

    #[test]
    fn rgba32323232f_is_16_bytes_per_pixel() {
        assert_eq!(PackedImportFormat::Rgba32323232F.bytes_per_pixel(), 16);
    }

    /// The refusal is a pure function of `plane_offset`, so a host tensor
    /// exercises it without a GL context.
    #[cfg(any(target_os = "windows", target_os = "macos", target_os = "ios"))]
    #[test]
    fn an_offset_source_is_refused_and_an_origin_source_is_not() {
        use edgefirst_tensor::{Tensor, TensorMemory};

        let mut t = Tensor::<u8>::new(&[4, 4, 4], Some(TensorMemory::Mem), None).expect("alloc");
        assert!(
            super::refuse_offset_source(&t, "source", "ext").is_ok(),
            "no offset recorded: attachable"
        );
        t.set_plane_offset(0);
        assert!(
            super::refuse_offset_source(&t, "source", "ext").is_ok(),
            "a zero offset is the origin: attachable"
        );
        t.set_plane_offset(16);
        let err = super::refuse_offset_source(&t, "source", "ext")
            .expect_err("a source 16 bytes in cannot be bound from the origin");
        assert!(
            matches!(err, crate::Error::NotSupported(_)),
            "NotSupported is what sends the engine to the upload path; got {err}"
        );
    }
}
