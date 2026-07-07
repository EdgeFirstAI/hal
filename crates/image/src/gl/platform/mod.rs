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
//! (e.g. Windows/ANGLE-D3D11) implements the trait or does not compile —
//! it cannot fork convert logic.
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

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(super) mod angle;
#[cfg(target_os = "linux")]
pub(super) mod linux;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(super) mod macos;

use super::EglDisplayKind;
use edgefirst_tensor::{PixelFormat, Tensor};

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
}

impl PackedImportFormat {
    /// Bytes per packed surface pixel.
    pub(crate) fn bytes_per_pixel(self) -> usize {
        match self {
            PackedImportFormat::Rgba8888 => 4,
            PackedImportFormat::Rgba16161616F => 8,
        }
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
    /// so the skip cache must stay cold there.
    const PERSISTENT_TEX_BINDINGS: bool;

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
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync;

    /// The `Copy` handle for a cached import.
    fn import_handle(import: &Self::Import) -> Self::ImportHandle;

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

    /// Release every texture attachment recorded since the last call.
    /// MUST be called only after the GPU work consuming those
    /// attachments has been synced (`glFinish`/fence) — the engine's
    /// sync funnel (eager convert boundary, batch flush) is the call
    /// site. No-op on Linux (bindings persist by design).
    fn end_gpu_pass(display: &Self::Display);
}

/// The one platform implementation for this build.
#[cfg(target_os = "linux")]
pub(super) type Platform = linux::LinuxEgl;
#[cfg(any(target_os = "macos", target_os = "ios"))]
pub(super) type Platform = angle::AngleClientBuffer;

// Compile-time check that the selected platform implements the contract —
// a partial port fails here, not at a call site deep in the engine.
const _: fn() = || {
    fn assert_platform<P: GlPlatform>() {}
    assert_platform::<Platform>();
};
