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

#[cfg(target_os = "linux")]
pub(super) mod linux;
#[cfg(target_os = "macos")]
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
}

/// The one platform implementation for this build. macOS gains its alias
/// when `angle.rs` lands (PR-A step A4); until then the macOS backend is
/// `MacosGlProcessor`, which does not route through the trait.
#[cfg(target_os = "linux")]
pub(super) type Platform = linux::LinuxEgl;

// Compile-time check that the selected platform implements the contract —
// a partial port fails here, not at a call site deep in the engine.
#[cfg(target_os = "linux")]
const _: fn() = || {
    fn assert_platform<P: GlPlatform>() {}
    assert_platform::<Platform>();
};
