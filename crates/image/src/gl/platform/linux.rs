// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Linux implementation of [`GlPlatform`]: EGL display bring-up via
//! GBM/PlatformDevice/Default and DMA-BUF buffer import.
//!
//! Delegates to [`super::super::context`] (display/context bring-up,
//! transfer-backend probe) and [`super::super::dma_import`] (DMA-BUF
//! attribute assembly).

use super::super::context::{egl_ext, Egl, GlContext};
use super::super::dma_import::DmaImportAttrs;
use super::super::EglDisplayKind;
use super::{GlPlatform, NvChromaFormat, NvDstPlanes};
use edgefirst_egl as egl;
use edgefirst_tensor::{PixelFormat, Tensor, TensorDyn};
use std::rc::Rc;

/// An owned EGLImage import over a DMA-BUF. Dropping destroys the
/// EGLImage under the display-level lifecycle lock. This is the Linux
/// [`GlPlatform::Import`].
pub(in crate::opengl_headless) struct EglImage {
    pub(in crate::opengl_headless) egl_image: egl::Image,
    pub(in crate::opengl_headless) egl: Rc<Egl>,
    pub(in crate::opengl_headless) display: egl::Display,
    /// Texel the logical image starts at inside this import, and the
    /// import's own texel size: `(0, 0)` / `None` unless a source was
    /// rebased onto an aligned DMA-BUF offset.
    pub(in crate::opengl_headless) sample_origin_px: (u32, u32),
    pub(in crate::opengl_headless) sample_extent_px: Option<(u32, u32)>,
}

impl Drop for EglImage {
    fn drop(&mut self) {
        if self.egl_image.as_ptr() == egl::NO_IMAGE {
            return;
        }

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Display-level EGL op, under the image lifecycle lock.
            let _image_guard = super::super::context::image_lifecycle_guard();
            // Destroy the image with tensor CPU mappings excluded.
            let e = edgefirst_tensor::pin::with_cpu_mappings_excluded(|| {
                GlContext::egl_destroy_image_with_fallback(&self.egl, self.display, self.egl_image)
            });
            if let Err(e) = e {
                log::error!("Could not destroy EGL image: {e:?}");
            }
        }));
    }
}

/// Marker type: Linux EGL + DMA-BUF platform. All state lives in the
/// [`GlContext`] created by [`GlPlatform::init_display`].
pub(crate) struct LinuxEgl;

impl GlPlatform for LinuxEgl {
    type Display = GlContext;
    type Import = EglImage;
    type ImportHandle = egl::Image;

    // EGLImage targets persist on the texture object.
    const PERSISTENT_TEX_BINDINGS: bool = true;

    /// The planar F16 zero-copy render only.
    const ZERO_COPY_FLOAT: super::super::float_dispatch::ZeroCopyFloatSet =
        super::super::float_dispatch::ZeroCopyFloatSet::PlanarF16;
    const EXTERNAL_OES: bool = true;
    const NV_CHROMA_IMPORT: bool = true;
    const PACKED_R8_IMPORT: bool = true;

    fn init_display(kind: Option<EglDisplayKind>) -> crate::Result<GlContext> {
        GlContext::new(kind)
    }

    fn load_gl_once(display: &GlContext) {
        static GL_LOADED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        GL_LOADED.get_or_init(|| {
            edgefirst_gl::load_with(|s| {
                display
                    .egl
                    .get_proc_address(s)
                    .map_or(std::ptr::null(), |p| p as *const _)
            });
        });
    }

    fn import_handle(import: &EglImage) -> egl::Image {
        import.egl_image
    }

    fn import_extent(import: &EglImage) -> Option<(u32, u32)> {
        // `None` except for a source rebased onto an aligned offset, which
        // imports `x_shift_px` texels wider.
        import.sample_extent_px
    }

    fn import_origin(import: &EglImage) -> (u32, u32) {
        import.sample_origin_px
    }

    unsafe fn attach_tex_image_2d(_display: &GlContext, handle: egl::Image) -> crate::Result<()> {
        unsafe {
            edgefirst_gl::gl::EGLImageTargetTexture2DOES(
                edgefirst_gl::gl::TEXTURE_2D,
                handle.as_ptr(),
            );
        }
        Ok(())
    }

    unsafe fn attach_tex_image_external(
        _display: &GlContext,
        handle: egl::Image,
    ) -> crate::Result<()> {
        edgefirst_gl::egl_image_target_texture_2d_oes(
            edgefirst_gl::gl::TEXTURE_EXTERNAL_OES,
            handle.as_ptr(),
        );
        Ok(())
    }

    unsafe fn attach_renderbuffer_storage(
        _display: &GlContext,
        handle: egl::Image,
    ) -> crate::Result<()> {
        unsafe {
            edgefirst_gl::gl::EGLImageTargetRenderbufferStorageOES(
                edgefirst_gl::gl::RENDERBUFFER,
                handle.as_ptr(),
            );
        }
        Ok(())
    }

    fn begin_gpu_pass(_display: &GlContext) {
        // No-op on native EGL.
    }

    fn end_gpu_pass(_display: &GlContext) {
        // Nothing to release.
    }

    fn native_fence_sync(_display: &GlContext) -> bool {
        false
    }

    fn export_completion_fence(
        _display: &GlContext,
        _recorded: Option<u64>,
    ) -> crate::Result<Option<super::super::CompletionFence>> {
        Ok(None)
    }

    fn record_completion(_display: &GlContext, _dst: &mut TensorDyn, _submit: bool) -> Option<u64> {
        None
    }

    fn submit_recorded(_display: &GlContext) {
        // No device fence: nothing to submit.
    }

    fn import_buffer_packed<T>(
        display: &GlContext,
        img: &Tensor<T>,
        width: usize,
        height: usize,
        fmt: super::PackedImportFormat,
    ) -> crate::Result<EglImage>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        use std::os::fd::AsRawFd;
        let drm_format = match fmt {
            super::PackedImportFormat::Rgba8888 => drm_fourcc::DrmFourcc::Abgr8888,
            super::PackedImportFormat::Rgba16161616F => drm_fourcc::DrmFourcc::Abgr16161616f,
            super::PackedImportFormat::Rgba32323232F => {
                return Err(crate::Error::NotSupported(
                    "RGBA32F zero-copy import has no buffer format on this platform".into(),
                ))
            }
        };
        let bpp = fmt.bytes_per_pixel();
        let fd = img
            .dmabuf()
            .map_err(|e| {
                crate::Error::NotImplemented(format!(
                    "import_buffer_packed requires DMA tensor ({e})"
                ))
            })?
            .as_raw_fd();

        // The tensor's stored stride when available, otherwise the
        // tightly-packed pitch.
        let pitch = img.effective_row_stride().unwrap_or(width * bpp);
        let offset = img.plane_offset().unwrap_or(0);
        let egl_img_attr = [
            egl_ext::LINUX_DRM_FOURCC as egl::Attrib,
            drm_format as u32 as egl::Attrib,
            edgefirst_egl::WIDTH as egl::Attrib,
            width as egl::Attrib,
            edgefirst_egl::HEIGHT as egl::Attrib,
            height as egl::Attrib,
            egl_ext::DMA_BUF_PLANE0_PITCH as egl::Attrib,
            pitch as egl::Attrib,
            egl_ext::DMA_BUF_PLANE0_OFFSET as egl::Attrib,
            offset as egl::Attrib,
            egl_ext::DMA_BUF_PLANE0_FD as egl::Attrib,
            fd as egl::Attrib,
            egl::IMAGE_PRESERVED as egl::Attrib,
            egl::TRUE as egl::Attrib,
            edgefirst_egl::NONE as egl::Attrib,
        ];
        new_egl_image_owned(display, egl_ext::LINUX_DMA_BUF, &egl_img_attr)
    }

    fn import_buffer(
        display: &GlContext,
        img: &Tensor<u8>,
        fmt: PixelFormat,
        for_dst: bool,
    ) -> crate::Result<EglImage> {
        let attrs = DmaImportAttrs::from_tensor(img, fmt, for_dst)?;
        let mut image =
            new_egl_image_owned(display, egl_ext::LINUX_DMA_BUF, &attrs.to_egl_attribs())?;
        // A source rebased onto an aligned base records its origin and its
        // widened extent.
        if attrs.x_shift_px != 0 {
            image.sample_origin_px = (attrs.x_shift_px, 0);
            image.sample_extent_px = Some((attrs.width as u32, attrs.height as u32));
        }
        Ok(image)
    }

    fn lists_packed_rgb_import(display: &GlContext) -> bool {
        dma_buf_import_formats(display).contains(&(drm_fourcc::DrmFourcc::Bgr888 as u32))
    }

    /// Imports the luma plane alone of a source [`Self::nv_native_planes`]
    /// accepts, and the combined plane of any other.
    fn import_buffer_nv_r8(
        display: &GlContext,
        img: &Tensor<u8>,
        fmt: PixelFormat,
    ) -> crate::Result<EglImage> {
        let attrs = if Self::nv_native_planes(img).is_some() {
            DmaImportAttrs::from_tensor_nv12_planes(img)?.0
        } else {
            DmaImportAttrs::from_tensor_nv_r8(img, fmt)?
        };
        new_egl_image_owned(display, egl_ext::LINUX_DMA_BUF, &attrs.to_egl_attribs())
    }

    /// Imports the chroma plane of a source [`Self::nv_native_planes`]
    /// accepts from its own buffer, and of any other from the combined plane.
    fn import_buffer_nv_chroma(
        display: &GlContext,
        img: &Tensor<u8>,
        fmt: PixelFormat,
    ) -> crate::Result<EglImage> {
        let attrs = if Self::nv_native_planes(img).is_some() {
            DmaImportAttrs::from_tensor_nv12_planes(img)?.1
        } else {
            DmaImportAttrs::from_tensor_nv_chroma_gr88(img, fmt)?
        };
        new_egl_image_owned(display, egl_ext::LINUX_DMA_BUF, &attrs.to_egl_attribs())
    }

    /// The luma extent of a multiplane NV12 source whose luma and chroma
    /// planes are DMA-BUFs.
    fn nv_native_planes(img: &Tensor<u8>) -> Option<(usize, usize)> {
        use edgefirst_tensor::{TensorMemory, TensorTrait};
        let chroma = img.chroma()?;
        (img.format() == Some(PixelFormat::Nv12)
            && img.memory() == TensorMemory::DmaBuf
            && chroma.memory() == TensorMemory::DmaBuf)
            .then_some((img.width()?, img.height()?))
    }

    const NV12_DST: bool = true;

    /// A single-plane destination imports `Combined`, a multiplane one
    /// `Split`.
    fn import_nv12_dst(
        display: &GlContext,
        img: &Tensor<u8>,
    ) -> crate::Result<NvDstPlanes<EglImage>> {
        let import = |attrs: &DmaImportAttrs| {
            new_egl_image_owned(display, egl_ext::LINUX_DMA_BUF, &attrs.to_egl_attribs())
        };
        if img.is_multiplane() {
            let (luma, chroma) = DmaImportAttrs::from_tensor_nv12_dst_split(img)?;
            Ok(NvDstPlanes::Split {
                luma: import(&luma)?,
                chroma: import(&chroma)?,
                chroma_rows: chroma.height as u32,
                chroma_format: NvChromaFormat::R8Interleaved,
            })
        } else {
            let attrs = DmaImportAttrs::from_tensor_nv12_dst_r8(img)?;
            Ok(NvDstPlanes::Combined {
                plane: import(&attrs)?,
                width: attrs.width as u32,
                height: attrs.height as u32,
            })
        }
    }
}

/// The DRM fourccs `eglQueryDmaBufFormatsEXT` lists for this display; empty
/// where the entry point (`EGL_EXT_image_dma_buf_import_modifiers`) is absent
/// or the query fails.
fn dma_buf_import_formats(display: &GlContext) -> Vec<u32> {
    let Some(f) = display.egl.get_proc_address("eglQueryDmaBufFormatsEXT") else {
        return Vec::new();
    };
    type QueryFormats = unsafe extern "system" fn(
        dpy: *mut std::ffi::c_void,
        max_formats: egl::Int,
        formats: *mut egl::Int,
        num_formats: *mut egl::Int,
    ) -> egl::Boolean;
    // SAFETY: `eglQueryDmaBufFormatsEXT` has exactly this signature
    // (EGL_EXT_image_dma_buf_import_modifiers), and the pointer came from
    // this display's `eglGetProcAddress`.
    let query: QueryFormats = unsafe { std::mem::transmute(f) };
    let dpy = display.display.as_ptr();
    let mut n: egl::Int = 0;
    // SAFETY: a size query (max 0, null list) on a live display.
    if unsafe { query(dpy, 0, std::ptr::null_mut(), &mut n) } == 0 || n <= 0 {
        return Vec::new();
    }
    let mut formats = vec![0 as egl::Int; n as usize];
    // SAFETY: `formats` holds `n` elements, the most the call writes.
    if unsafe { query(dpy, n, formats.as_mut_ptr(), &mut n) } == 0 {
        return Vec::new();
    }
    formats.truncate(n.max(0) as usize);
    formats.into_iter().map(|f| f as u32).collect()
}

/// Create an owned EGLImage from raw EGL import attributes. Every EGLImage
/// creation goes through here (DMA-BUF, NV R8, and the float path's RGB
/// renderbuffer import).
pub(in crate::opengl_headless) fn new_egl_image_owned(
    display: &GlContext,
    target: egl::Enum,
    attrib_list: &[egl::Attrib],
) -> crate::Result<EglImage> {
    let _span = tracing::trace_span!("image.convert.gl.egl_import", target).entered();
    // EGLImage creation runs under the image lifecycle lock.
    let _image_guard = super::super::context::image_lifecycle_guard();
    let image = GlContext::egl_create_image_with_fallback(
        &display.egl,
        display.display,
        unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) },
        target,
        unsafe { egl::ClientBuffer::from_ptr(std::ptr::null_mut()) },
        attrib_list,
    )?;
    Ok(EglImage {
        egl_image: image,
        display: display.display,
        egl: Rc::clone(&display.egl),
        sample_origin_px: (0, 0),
        sample_extent_px: None,
    })
}
