// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Cross-platform seam for the OpenGL backend.
//!
//! [`GlPlatform`] is the compile-time porting contract: one implementation
//! is selected per build through the [`Platform`] type alias. The portable
//! engine reaches platform buffers only through this trait.
//!
//! * Linux: the standard EGL display path, DMA-BUF import through
//!   `eglCreateImageKHR`. Its helpers live in [`super::context`] and
//!   [`super::dma_import`].
//! * macOS / iOS: ANGLE Metal display through `eglGetPlatformDisplayEXT`,
//!   IOSurface import through `eglCreatePbufferFromClientBuffer`.
//! * Android: native EGL, AHardwareBuffer import.
//! * Windows: ANGLE over Direct3D 11, D3D11 texture import.
//!
//! [`PlatformCaps`] is the capability surface the portable code keys
//! decisions on (serialization policy, transfer backend, float render
//! support). It is captured once per processor at worker startup.

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
/// context. Captured once at processor/worker construction and immutable for
/// the processor's life.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlatformCaps {
    /// Active pixel-transfer method (DMA-BUF / IOSurface / PBO / Sync).
    pub(crate) transfer_backend: super::TransferBackend,
    /// Float render-target support (F32/F16 color attachments), adjusted
    /// for driver quirks.
    pub(crate) render_dtypes: crate::RenderDtypeSupport,
    /// Whether GL command submission is serialized process-wide (one
    /// message at a time across all processors).
    pub(crate) serialize_gl: bool,
    /// Whether `GL_TEXTURE_EXTERNAL_OES` sampling of multi-plane imports
    /// is available. Constructed but not read.
    #[allow(dead_code)]
    pub(crate) external_oes: bool,
    /// Whether `convert_with_fence` can export a native fence fd
    /// (`EGL_ANDROID_native_fence_sync` on this display). When false the
    /// fenced entry points take the blocking path and return no fd.
    pub(crate) native_fence_sync: bool,
    /// Whether `GL_EXT_texture_format_BGRA8888` is available, which makes
    /// BGRA a renderable destination format.
    pub(crate) has_bgra: bool,
}

/// Platform-neutral identity of a packed render surface: the float paths
/// render planar/RGB byte streams through an RGBA-shaped surface whose pixel
/// count encodes the byte layout. Linux maps these to DRM fourccs, macOS to
/// IOSurface pixel layouts.
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

/// The plane offset of a destination that carries a nonzero `plane_offset`
/// and no `view_origin`; `None` otherwise.
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
/// origin: returns `NotSupported` when `img` carries a nonzero plane offset.
/// `what` and `binding` name the role and the extension in the message.
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
/// are associated functions; all state lives in the `Display` the platform
/// creates.
pub(super) trait GlPlatform {
    /// Owning handle for the platform's GL/EGL bring-up state: display,
    /// context, capability probes. On Linux this is
    /// [`super::context::GlContext`]; on macOS the per-processor ANGLE
    /// context over the shared Metal display.
    type Display;

    /// Owned zero-copy buffer import: an `EGLImage` over a DMA-BUF on
    /// Linux; an EGL pbuffer over an IOSurface on macOS. The import cache
    /// stores these; Drop releases the platform object.
    type Import;

    /// `Copy` handle to a cached import, safe to pass around while the
    /// cache owns the import object: `egl::Image` on Linux,
    /// `egl::Surface` (the pbuffer) on macOS.
    type ImportHandle: Copy;

    /// Whether [`Self::attach_tex_image_2d`] bindings persist on the GL
    /// texture object across GPU passes. `true` on Linux and Android, where
    /// EGLImage targets persist. `false` on macOS, where `eglBindTexImage`
    /// bindings are released at the end of each synced pass
    /// ([`Self::end_gpu_pass`]), and on Windows, where the engine re-attaches
    /// on every use.
    const PERSISTENT_TEX_BINDINGS: bool;

    /// Which zero-copy float render paths this platform serves, as read by
    /// [`super::float_dispatch::classify_float_render`]. The Linux, macOS
    /// and Android leaves report [`ZeroCopyFloatSet::PlanarF16`]; the Windows
    /// leaf reports [`ZeroCopyFloatSet::All`].
    ///
    /// [`ZeroCopyFloatSet::PlanarF16`]: super::float_dispatch::ZeroCopyFloatSet::PlanarF16
    /// [`ZeroCopyFloatSet::All`]: super::float_dispatch::ZeroCopyFloatSet::All
    const ZERO_COPY_FLOAT: super::float_dispatch::ZeroCopyFloatSet;

    /// Whether `GL_TEXTURE_EXTERNAL_OES` sampling of imports exists on
    /// this platform.
    const EXTERNAL_OES: bool;

    /// Load the process-global GL function-pointer table once per process.
    /// Linux resolves through this display's `eglGetProcAddress`; the ANGLE
    /// leaves load at shared-display init and do nothing here.
    fn load_gl_once(display: &Self::Display);

    /// Bring up the platform display + context for one processor.
    /// `kind` selects the EGL display flavour on Linux and is ignored
    /// (with a debug log) on the ANGLE leaves.
    fn init_display(kind: Option<EglDisplayKind>) -> crate::Result<Self::Display>;

    /// Import a tensor's zero-copy buffer, typed at `fmt`, for sampling
    /// (`for_dst = false`) or rendering into (`for_dst = true`). A
    /// destination view imports its parent buffer; a source view imports its
    /// own region. On Linux this is an `EGLImage` over the tensor's DMA-BUF;
    /// on macOS an EGL pbuffer over the tensor's IOSurface.
    ///
    /// Callers cache the result in [`super::cache::ImportCache`] keyed by
    /// [`super::cache::BufferImportKey`].
    fn import_buffer(
        display: &Self::Display,
        img: &Tensor<u8>,
        fmt: PixelFormat,
        for_dst: bool,
    ) -> crate::Result<Self::Import>;

    /// Confirm this tensor's [`BufferIdentity`] names the object the
    /// platform would import. Called on every import, before the import
    /// cache is consulted.
    ///
    /// Defaults to `Ok(())`. An error refuses the zero-copy path.
    fn validate_import_identity<T>(_img: &Tensor<T>, _what: &str) -> crate::Result<()>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        Ok(())
    }

    /// Whether a zero-copy import of `img` as a destination can place the
    /// render where the tensor's bytes start. A platform whose import binds
    /// the whole buffer from its origin returns `false` for a destination
    /// that carries a `plane_offset` and no `view_origin`.
    ///
    /// Defaults to `true`.
    fn dst_import_places<T>(_img: &Tensor<T>) -> bool
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        true
    }

    /// Import an NV12/NV16/NV24 tensor's combined semi-planar plane as one
    /// R8 buffer (luma + interleaved chroma addressed by the shader). Where
    /// [`Self::PACKED_R8_IMPORT`] is set, also a packed `Rgb` or `Yuyv`
    /// tensor. On Linux a single-plane R8 EGLImage at the buffer's physical
    /// pitch; on macOS an R8 (`L008`) IOSurface pbuffer binding.
    fn import_buffer_nv_r8(
        display: &Self::Display,
        img: &Tensor<u8>,
        fmt: PixelFormat,
    ) -> crate::Result<Self::Import>;

    /// Whether [`Self::import_buffer_nv_r8`] also imports a packed `Rgb` or
    /// `Yuyv` buffer (as `height` rows of `row_stride` bytes).
    const PACKED_R8_IMPORT: bool = false;

    /// Whether the display lists the 24-bit `DRM_FORMAT_BGR888` (`BG24`,
    /// memory order R, G, B: the HAL's packed `Rgb`) among its importable
    /// buffer formats.
    fn lists_packed_rgb_import(_display: &Self::Display) -> bool {
        false
    }

    /// Whether [`Self::import_buffer_nv_chroma`] is implemented.
    const NV_CHROMA_IMPORT: bool = false;

    /// The luma plane's texel extent when `img` is a two-plane NV12 buffer
    /// whose imports are its planes: [`Self::import_buffer_nv_r8`] then holds
    /// the luma plane alone and [`Self::import_buffer_nv_chroma`] the chroma
    /// plane (half the extent). `None` for a combined-plane buffer.
    fn nv_native_planes(_img: &Tensor<u8>) -> Option<(usize, usize)> {
        None
    }

    /// Import the chroma plane of a single-plane NV12/NV16/NV24 tensor as a
    /// two-channel (U, V) texture of one texel per chroma sample. Platforms
    /// without it return `NotSupported`.
    fn import_buffer_nv_chroma(
        _display: &Self::Display,
        _img: &Tensor<u8>,
        fmt: PixelFormat,
    ) -> crate::Result<Self::Import> {
        Err(crate::Error::NotSupported(format!(
            "no {fmt:?} chroma-plane import on this platform"
        )))
    }

    /// Import a tensor's zero-copy buffer as a packed RGBA-shaped render
    /// surface of `width`×`height` pixels (see [`PackedImportFormat`]). The
    /// caller computes the packed surface dims; the platform derives
    /// pitch/offset from the tensor.
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
    /// The engine scales the source rectangle by `logical / extent`
    /// ([`super::render::scale_roi_to_import`]) and clamps every sample to the
    /// logical image ([`super::render::sample_clamp_rect`]).
    fn import_extent(import: &Self::Import) -> Option<(u32, u32)>;

    /// Texels from the imported texture's origin to the tensor's own first
    /// pixel. Defaults to `(0, 0)`. Linux returns a nonzero origin for a
    /// source rebased onto a 64-byte-aligned DMA-BUF base.
    fn import_origin(_import: &Self::Import) -> (u32, u32) {
        (0, 0)
    }

    /// Attach the import as the image of the currently bound
    /// `GL_TEXTURE_2D` texture object. Linux:
    /// `glEGLImageTargetTexture2DOES`; macOS: `eglBindTexImage` (recorded on
    /// the display and released by [`Self::end_gpu_pass`]).
    ///
    /// # Safety
    /// The intended texture must be bound on the active texture unit and
    /// the handle's import must be alive.
    unsafe fn attach_tex_image_2d(
        display: &Self::Display,
        handle: Self::ImportHandle,
    ) -> crate::Result<()>;

    /// Attach the import to the currently bound `GL_TEXTURE_EXTERNAL_OES`
    /// texture. Errors on platforms without the OES extension.
    ///
    /// # Safety
    /// As [`Self::attach_tex_image_2d`].
    unsafe fn attach_tex_image_external(
        display: &Self::Display,
        handle: Self::ImportHandle,
    ) -> crate::Result<()>;

    /// Attach the import as the storage of the currently bound GL
    /// renderbuffer (the destination path enabled by
    /// `EDGEFIRST_OPENGL_RENDERSURFACE`). Errors where renderbuffer import
    /// targets do not exist.
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
    /// on Linux, ANGLE/Metal and Android. ANGLE/D3D11 re-makes this
    /// processor's context current when another processor's context issued
    /// the previous GL commands.
    fn begin_gpu_pass(display: &Self::Display);

    /// Release every texture attachment recorded since the last call.
    /// Must be called only after the GPU work consuming those attachments
    /// has been synced (`glFinish`/fence). No-op on Linux.
    fn end_gpu_pass(display: &Self::Display);

    /// Whether [`Self::export_completion_fence`] can return a fence on this
    /// display.
    fn native_fence_sync(display: &Self::Display) -> bool;

    /// Export a fence guarding every GL command submitted so far on the
    /// current context. `Ok(None)` where native fence sync does not exist.
    /// Must run on the GL worker thread. The handle type is the
    /// platform-neutral [`super::CompletionFence`] (an fd on Unix, an NT
    /// handle on Windows).
    ///
    /// `recorded` is the value [`Self::record_completion`] recorded on the
    /// destination for this same convert, when the caller has one. A leaf
    /// whose completion fence is the device fence returns the event for that
    /// value; a leaf whose fence is a GL native fence ignores it.
    fn export_completion_fence(
        display: &Self::Display,
        recorded: Option<u64>,
    ) -> crate::Result<Option<super::CompletionFence>>;

    /// Called by the engine after a convert or a mask draw into `dst` has
    /// been issued and, for the blocking path, completed. Platforms with a
    /// device fence record the value covering that work on the destination
    /// (`Tensor::gpu_completion`) and return it; other platforms return
    /// `None`.
    ///
    /// `submit` is `false` while a deferred batch is open: the value is
    /// allocated and its signal queued, and the device's command buffer is
    /// submitted from [`Self::submit_recorded`] at the batch's flush.
    fn record_completion(display: &Self::Display, dst: &mut TensorDyn, submit: bool)
        -> Option<u64>;

    /// Submit the signals [`Self::record_completion`] queued with
    /// `submit == false`. Called once when a deferred batch flushes.
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

// Compile-time check that the selected platform implements the contract.
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

    /// A source with a nonzero plane offset is refused; one at the origin is not.
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
