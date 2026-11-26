// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! macOS and iOS implementation of [`GlPlatform`]: ANGLE (GLES→Metal)
//! display bring-up and IOSurface buffer import.
//!
//! Owns the process-global shared ANGLE display ([`SharedAngleDisplay`] /
//! [`shared_display`]).
//!
//! [`AngleDisplay`] is the per-processor display: each processor's
//! dedicated worker thread creates its own EGL context (plus a small dummy
//! pbuffer), makes it current once, and holds it for the thread's life.
//!
//! Buffer import wraps the tensor's IOSurface as an EGL pbuffer via
//! `EGL_ANGLE_iosurface_client_buffer` (see `iosurface_import.rs` for the
//! attribute assembly). The owned [`IoSurfacePbuffer`] destroys the
//! pbuffer on drop; the [`super::super::cache::ImportCache`] holds these.

use super::super::iosurface_import;
use super::super::Egl;
use super::macos::ApplePlatform;
use super::{GlPlatform, NvChromaFormat, NvDstPlanes};
use crate::{Error, Result};
use edgefirst_egl as egl;
use edgefirst_tensor::{DType, PixelFormat, Tensor, TensorDyn};
use log::debug;
use std::ffi::c_void;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// EGL constants shared by the display bring-up and per-processor context
// creation (from EGL/eglext headers).
// ---------------------------------------------------------------------------

const EGL_OPENGL_ES3_BIT: i32 = 0x0040;
const EGL_PBUFFER_BIT: i32 = 0x0001;
const EGL_RENDERABLE_TYPE: i32 = 0x3040;
const EGL_SURFACE_TYPE: i32 = 0x3033;
const EGL_RED_SIZE: i32 = 0x3024;
const EGL_GREEN_SIZE: i32 = 0x3023;
const EGL_BLUE_SIZE: i32 = 0x3022;
const EGL_ALPHA_SIZE: i32 = 0x3021;
pub(in crate::opengl_headless) const EGL_CONTEXT_CLIENT_VERSION: i32 = 0x3098;
const EGL_BACK_BUFFER: i32 = 0x3084;

// ---------------------------------------------------------------------------
// One-shot GL function-pointer table, loaded through EGL's
// `eglGetProcAddress` once per process.
// ---------------------------------------------------------------------------

static GL_LOADED: OnceLock<()> = OnceLock::new();

fn load_gl_once(egl: &Egl) {
    GL_LOADED.get_or_init(|| {
        edgefirst_gl::load_with(|name| match egl.get_proc_address(name) {
            Some(ptr) => ptr as *const c_void,
            None => std::ptr::null(),
        });
    });
}

// ---------------------------------------------------------------------------
// Process-global ANGLE EGL display.
// ---------------------------------------------------------------------------

/// All process-global EGL state. Use [`shared_display`] to access.
pub(in crate::opengl_headless) struct SharedAngleDisplay {
    /// Static-lifetime EGL handle. The ANGLE libEGL.dylib is leaked at
    /// first dlopen and never closed.
    pub(in crate::opengl_headless) egl: Egl,
    pub(in crate::opengl_headless) display: egl::Display,
    pub(in crate::opengl_headless) config: egl::Config,
    /// The shared display's own context, made current at init to load GL
    /// and probe the extensions.
    pub(in crate::opengl_headless) context: egl::Context,
    /// Tiny scratch surface the shared context is made current on.
    pub(in crate::opengl_headless) dummy_pbuffer: egl::Surface,
    /// `GL_EXT_color_buffer_float` is exposed by this ANGLE/Metal
    /// configuration. Gates F32 destination tensors on the IOSurface
    /// render path.
    pub(in crate::opengl_headless) supports_f32_color: bool,
    /// `GL_EXT_color_buffer_half_float` is exposed by this
    /// ANGLE/Metal configuration. Gates F16 destination tensors on
    /// the IOSurface render path.
    pub(in crate::opengl_headless) supports_f16_color: bool,
}

// SAFETY: every member is a leaked static, an EGL handle, or a pointer to
// driver-owned state. ANGLE synchronizes display-level EGL entry points
// internally.
unsafe impl Send for SharedAngleDisplay {}
unsafe impl Sync for SharedAngleDisplay {}

static SHARED_DISPLAY: OnceLock<std::result::Result<SharedAngleDisplay, String>> = OnceLock::new();

/// Acquire a reference to the process-global ANGLE display, initialising
/// it on first call. Subsequent calls return the same handle or the same
/// cached error.
pub(in crate::opengl_headless) fn shared_display() -> Result<&'static SharedAngleDisplay> {
    SHARED_DISPLAY
        .get_or_init(|| init_shared_display().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|s| Error::Io(std::io::Error::other(s.clone())))
}

fn init_shared_display() -> Result<SharedAngleDisplay> {
    let platform = if cfg!(target_os = "ios") {
        "ios"
    } else {
        "macos"
    };
    let _span = tracing::info_span!("image.gl_init", platform, backend = "iosurface",).entered();

    // 1. Load ANGLE libEGL and bring up an EGL instance.
    let egl_lib = ApplePlatform::load_egl_lib()
        .map_err(|e| Error::Io(std::io::Error::other(format!("ANGLE libEGL: {e}"))))?;
    let egl: Egl = unsafe {
        edgefirst_egl::Instance::<
            edgefirst_egl::Dynamic<&'static libloading::Library, edgefirst_egl::EGL1_4>,
        >::load_required_from(egl_lib)
    }
    .map_err(|e| Error::Io(std::io::Error::other(format!("EGL load: {e:?}"))))?;
    debug!("EGL dynamic instance loaded, version = {:?}", egl.version());

    // 2. Metal-backed display from ApplePlatform.
    let display = ApplePlatform::create_display(&egl)?;
    let (maj, min) = egl
        .initialize(display)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglInitialize: {e:?}"))))?;
    debug!("ANGLE EGL {maj}.{min} initialised (process-global shared display)");

    egl.bind_api(egl::OPENGL_ES_API)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglBindAPI: {e:?}"))))?;

    // 3. Choose an EGL config that supports GLES 3 + PBUFFER +
    //    EGL_BIND_TO_TEXTURE_TARGET_ANGLE = EGL_TEXTURE_2D.
    let cfg_attribs = [
        EGL_RENDERABLE_TYPE,
        EGL_OPENGL_ES3_BIT,
        EGL_SURFACE_TYPE,
        EGL_PBUFFER_BIT,
        EGL_RED_SIZE,
        8,
        EGL_GREEN_SIZE,
        8,
        EGL_BLUE_SIZE,
        8,
        EGL_ALPHA_SIZE,
        8,
        iosurface_import::EGL_BIND_TO_TEXTURE_TARGET_ANGLE,
        0x305F, // EGL_TEXTURE_2D
        egl::NONE,
    ];
    let config = egl
        .choose_first_config(display, &cfg_attribs)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglChooseConfig: {e:?}"))))?
        .ok_or_else(|| {
            Error::NotSupported("no EGL config with GLES3+PBUFFER+TEXTURE_2D bind".into())
        })?;

    // 4. GLES3 context.
    let ctx_attribs = [EGL_CONTEXT_CLIENT_VERSION, 3, egl::NONE];
    let context = egl
        .create_context(display, config, None, &ctx_attribs)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglCreateContext: {e:?}"))))?;

    // 5. Dummy pbuffer for context-current bring-up.
    let dummy_attribs = [egl::WIDTH, 16, egl::HEIGHT, 16, egl::NONE];
    let dummy_pbuffer = egl
        .create_pbuffer_surface(display, config, &dummy_attribs)
        .map_err(|e| {
            // Destroy the context created above.
            let _ = egl.destroy_context(display, context);
            Error::Io(std::io::Error::other(format!(
                "eglCreatePbufferSurface(dummy): {e:?}"
            )))
        })?;

    // 6. Make the context current and load the GL function pointers.
    if let Err(e) = egl.make_current(
        display,
        Some(dummy_pbuffer),
        Some(dummy_pbuffer),
        Some(context),
    ) {
        let _ = egl.destroy_surface(display, dummy_pbuffer);
        let _ = egl.destroy_context(display, context);
        return Err(Error::Io(std::io::Error::other(format!(
            "eglMakeCurrent(dummy): {e:?}"
        ))));
    }
    load_gl_once(&egl);

    // Probe the float-color-buffer extensions while the context is
    // current.
    let extensions = unsafe {
        let ptr = edgefirst_gl::gl::GetString(edgefirst_gl::gl::EXTENSIONS);
        if ptr.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(ptr as *const std::os::raw::c_char)
                .to_string_lossy()
                .into_owned()
        }
    };
    let supports_f32_color = extensions
        .split_ascii_whitespace()
        .any(|e| e == "GL_EXT_color_buffer_float");
    let supports_f16_color = extensions
        .split_ascii_whitespace()
        .any(|e| e == "GL_EXT_color_buffer_half_float");
    debug!(
        "ANGLE: GL_EXT_color_buffer_float={supports_f32_color}, \
         GL_EXT_color_buffer_half_float={supports_f16_color}"
    );

    let _ = egl.make_current(display, None, None, None);

    Ok(SharedAngleDisplay {
        egl,
        display,
        config,
        context,
        dummy_pbuffer,
        supports_f32_color,
        supports_f16_color,
    })
}

// ---------------------------------------------------------------------------
// Per-processor display: a private context on the shared ANGLE display.
// ---------------------------------------------------------------------------

/// One processor's GL bring-up state: a private EGL context (plus dummy
/// pbuffer) on the process-global shared ANGLE display.
///
/// Created on the processor's dedicated worker thread, made current there
/// once, and held current for the thread's life. Not `Send`: the context is
/// current on its creating thread and must be dropped there.
pub(in crate::opengl_headless) struct AngleDisplay {
    pub(in crate::opengl_headless) shared: &'static SharedAngleDisplay,
    context: egl::Context,
    dummy_pbuffer: egl::Surface,
    /// Counterparts of the two `GlContext` members the portable engine
    /// reads: the IOSurface transfer backend, and no GLES 3.1 compute.
    pub(in crate::opengl_headless) transfer_backend: super::super::TransferBackend,
    pub(in crate::opengl_headless) has_compute: bool,
    /// Texture attachments made via `eglBindTexImage` since the last
    /// `GlPlatform::end_gpu_pass`, which releases them.
    active_binds: std::cell::RefCell<Vec<egl::Surface>>,
    /// The number of `GlPlatform::end_gpu_pass` calls so far.
    bind_epoch: std::cell::Cell<u64>,
}

impl AngleDisplay {
    /// Assemble the capability surface for this display (see
    /// `PlatformCaps`): IOSurface transfer and float render support from the
    /// shared display's extension probes.
    #[allow(dead_code)]
    pub(in crate::opengl_headless) fn platform_caps(&self) -> super::PlatformCaps {
        super::PlatformCaps {
            transfer_backend: super::super::TransferBackend::IOSurface,
            render_dtypes: crate::RenderDtypeSupport {
                f32: self.shared.supports_f32_color,
                f16: self.shared.supports_f16_color,
            },
            serialize_gl: true,
            external_oes: false,
            native_fence_sync: false,
            has_bgra: false,
            cpu_latency_qos: false,
        }
    }
}

impl Drop for AngleDisplay {
    fn drop(&mut self) {
        // Release the context from this thread, then destroy it.
        let d = self.shared;
        let _ = d.egl.make_current(d.display, None, None, None);
        let _ = d.egl.destroy_surface(d.display, self.dummy_pbuffer);
        let _ = d.egl.destroy_context(d.display, self.context);
    }
}

/// An owned IOSurface→EGL-pbuffer import. Dropping destroys the pbuffer.
/// Cached in `ImportCache<IoSurfacePbuffer>` keyed by `BufferImportKey`.
pub(in crate::opengl_headless) struct IoSurfacePbuffer {
    shared: &'static SharedAngleDisplay,
    pub(in crate::opengl_headless) surface: egl::Surface,
}

impl Drop for IoSurfacePbuffer {
    fn drop(&mut self) {
        let _ = self
            .shared
            .egl
            .destroy_surface(self.shared.display, self.surface);
    }
}

/// Marker type: macOS ANGLE + IOSurface platform.
pub(crate) struct AngleClientBuffer;

impl GlPlatform for AngleClientBuffer {
    type Display = AngleDisplay;
    type Import = IoSurfacePbuffer;
    type ImportHandle = egl::Surface;

    /// A destination that carries a plane offset and no `view_origin` is
    /// not placed by a zero-copy import, unless its planes are a planar
    /// surface's own (see [`Self::nv_native_planes`]).
    fn dst_import_places<T>(img: &Tensor<T>) -> bool
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        super::unplaced_destination(img).is_none() || planar_nv12_extent(img).is_some()
    }

    fn tex_binding_epoch(display: &AngleDisplay) -> Option<u64> {
        // eglBindTexImage attachments last until end_gpu_pass releases them.
        Some(display.bind_epoch.get())
    }

    /// The planar F16 zero-copy render only.
    const ZERO_COPY_FLOAT: super::super::float_dispatch::ZeroCopyFloatSet =
        super::super::float_dispatch::ZeroCopyFloatSet::PlanarF16;
    const EXTERNAL_OES: bool = false;
    const NV12_DST: bool = true;
    const NV12_SPLIT_CHROMA: NvChromaFormat = NvChromaFormat::Rg8;

    fn load_gl_once(_display: &AngleDisplay) {
        // Loaded once at shared-display init.
    }

    fn import_handle(import: &IoSurfacePbuffer) -> egl::Surface {
        import.surface
    }

    fn import_extent(_import: &IoSurfacePbuffer) -> Option<(u32, u32)> {
        // The pbuffer is created with explicit EGL_WIDTH/EGL_HEIGHT: the
        // import is the tensor's logical image.
        None
    }

    unsafe fn attach_tex_image_2d(display: &AngleDisplay, handle: egl::Surface) -> Result<()> {
        display
            .shared
            .egl
            .bind_tex_image(display.shared.display, handle, EGL_BACK_BUFFER)
            .map_err(|e| Error::Io(std::io::Error::other(format!("eglBindTexImage: {e:?}"))))?;
        display.active_binds.borrow_mut().push(handle);
        Ok(())
    }

    unsafe fn attach_tex_image_external(
        _display: &AngleDisplay,
        _handle: egl::Surface,
    ) -> Result<()> {
        Err(Error::NotSupported(
            "GL_TEXTURE_EXTERNAL_OES is not available on ANGLE/Metal".into(),
        ))
    }

    unsafe fn attach_renderbuffer_storage(
        _display: &AngleDisplay,
        _handle: egl::Surface,
    ) -> Result<()> {
        Err(Error::NotSupported(
            "renderbuffer import targets are not available on ANGLE/Metal \
             (EDGEFIRST_OPENGL_RENDERSURFACE has no effect on macOS)"
                .into(),
        ))
    }

    fn native_fence_sync(_display: &AngleDisplay) -> bool {
        false
    }

    fn export_completion_fence(
        _display: &AngleDisplay,
        _recorded: Option<u64>,
    ) -> crate::Result<Option<super::super::CompletionFence>> {
        Ok(None)
    }

    fn record_completion(
        _display: &AngleDisplay,
        _dst: &mut TensorDyn,
        _submit: bool,
    ) -> Option<u64> {
        None
    }

    fn submit_recorded(_display: &AngleDisplay) {
        // No device fence: nothing to submit.
    }

    fn begin_gpu_pass(_display: &AngleDisplay) {
        // No-op on ANGLE/Metal.
    }

    fn end_gpu_pass(display: &AngleDisplay) {
        // Release every pbuffer attachment recorded during the pass.
        for surface in display.active_binds.borrow_mut().drain(..) {
            let _ = display.shared.egl.release_tex_image(
                display.shared.display,
                surface,
                EGL_BACK_BUFFER,
            );
        }
        display.bind_epoch.set(display.bind_epoch.get() + 1);
    }

    fn import_buffer_packed<T>(
        display: &AngleDisplay,
        img: &Tensor<T>,
        width: usize,
        height: usize,
        fmt: super::PackedImportFormat,
    ) -> Result<IoSurfacePbuffer>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        let surface_ref = img.iosurface_ref().ok_or_else(|| {
            Error::NotSupported("packed import: tensor is not IOSurface-backed".into())
        })?;
        // Map the packed surface dims back onto the logical image layout:
        // RGBA8 maps 1:1; the RGBA16F planar packing (W/4, 3·H) maps to the
        // logical (W, H) of a (PlanarRgb, F16) image.
        let (pf, dt, w, h) = match fmt {
            super::PackedImportFormat::Rgba8888 => (PixelFormat::Rgba, DType::U8, width, height),
            super::PackedImportFormat::Rgba16161616F => {
                if !height.is_multiple_of(3) {
                    return Err(Error::NotSupported(format!(
                        "packed RGBA16F surface height {height} is not a 3-plane stack"
                    )));
                }
                (PixelFormat::PlanarRgb, DType::F16, width * 4, height / 3)
            }
            super::PackedImportFormat::Rgba32323232F => {
                return Err(Error::NotSupported(
                    "RGBA32F zero-copy import has no buffer format on this platform".into(),
                ))
            }
        };
        let shared = display.shared;
        // SAFETY: as in `import_buffer`.
        let surface = unsafe {
            iosurface_import::create_iosurface_pbuffer(
                &shared.egl,
                shared.display,
                shared.config,
                surface_ref,
                pf,
                dt,
                w,
                h,
            )?
        };
        Ok(IoSurfacePbuffer { shared, surface })
    }

    fn init_display(kind: Option<super::super::EglDisplayKind>) -> Result<AngleDisplay> {
        if let Some(kind) = kind {
            debug!("EglDisplayKind::{kind} ignored on macOS — ANGLE/Metal is the only display");
        }
        let shared = shared_display()?;
        let ctx_attribs = [EGL_CONTEXT_CLIENT_VERSION, 3, egl::NONE];
        // No share-list: each processor owns its programs, VAOs and FBOs.
        let context = shared
            .egl
            .create_context(shared.display, shared.config, None, &ctx_attribs)
            .map_err(|e| {
                Error::Io(std::io::Error::other(format!(
                    "eglCreateContext (per-processor): {e:?}"
                )))
            })?;
        let dummy_attribs = [egl::WIDTH, 16, egl::HEIGHT, 16, egl::NONE];
        let dummy_pbuffer = shared
            .egl
            .create_pbuffer_surface(shared.display, shared.config, &dummy_attribs)
            .map_err(|e| {
                let _ = shared.egl.destroy_context(shared.display, context);
                Error::Io(std::io::Error::other(format!(
                    "eglCreatePbufferSurface (per-processor dummy): {e:?}"
                )))
            })?;
        // Made current once on the calling thread and held for the thread's
        // life.
        if let Err(e) = shared.egl.make_current(
            shared.display,
            Some(dummy_pbuffer),
            Some(dummy_pbuffer),
            Some(context),
        ) {
            let _ = shared.egl.destroy_surface(shared.display, dummy_pbuffer);
            let _ = shared.egl.destroy_context(shared.display, context);
            return Err(Error::Io(std::io::Error::other(format!(
                "eglMakeCurrent (per-processor): {e:?}"
            ))));
        }
        Ok(AngleDisplay {
            shared,
            context,
            dummy_pbuffer,
            transfer_backend: super::super::TransferBackend::IOSurface,
            has_compute: false,
            active_binds: std::cell::RefCell::new(Vec::new()),
            bind_epoch: std::cell::Cell::new(0),
        })
    }

    fn import_buffer(
        display: &AngleDisplay,
        img: &Tensor<u8>,
        fmt: PixelFormat,
        for_dst: bool,
    ) -> Result<IoSurfacePbuffer> {
        let surface_ref = img.iosurface_ref().ok_or_else(|| {
            Error::NotSupported("GL convert: tensor is not IOSurface-backed".into())
        })?;
        if matches!(
            fmt,
            PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
        ) {
            return Err(Error::NotSupported(
                "ANGLE IOSurface import has no multi-plane NV binding — use the R8 path".into(),
            ));
        }
        // A source whose pixels start at a plane offset is refused.
        if !for_dst {
            super::refuse_offset_source(img, "source", "EGL_ANGLE_iosurface_client_buffer")?;
        }
        // A destination view()/batch() tile imports its parent surface.
        let (w, h) = match (for_dst, img.view_origin()) {
            (true, Some(vo)) => (vo.parent_width, vo.parent_height),
            _ => (
                img.width()
                    .ok_or_else(|| Error::InvalidShape("import width".into()))?,
                img.height()
                    .ok_or_else(|| Error::InvalidShape("import height".into()))?,
            ),
        };
        let shared = display.shared;
        // SAFETY: `surface_ref` is borrowed from a live tensor.
        let surface = unsafe {
            iosurface_import::create_iosurface_pbuffer(
                &shared.egl,
                shared.display,
                shared.config,
                surface_ref,
                fmt,
                DType::U8,
                w,
                h,
            )?
        };
        Ok(IoSurfacePbuffer { shared, surface })
    }

    /// Imports plane 0 of a planar NV12 surface (see [`Self::nv_native_planes`])
    /// as `GL_RED`; any other NV source as its whole surface in one R8 binding.
    fn import_buffer_nv_r8(
        display: &AngleDisplay,
        img: &Tensor<u8>,
        _fmt: PixelFormat,
    ) -> Result<IoSurfacePbuffer> {
        match Self::nv_native_planes(img) {
            Some((w, h)) => import_plane(display, img, 0, iosurface_import::GL_RED, w, h),
            None => import_nv_combined(display, img),
        }
    }

    /// Imports plane 1 of a planar NV12 surface, from the chroma tensor's
    /// surface, as `GL_RG` at half the luma extent.
    fn import_buffer_nv_chroma(
        display: &AngleDisplay,
        img: &Tensor<u8>,
        fmt: PixelFormat,
    ) -> Result<IoSurfacePbuffer> {
        let ((w, h), chroma) = Self::nv_native_planes(img)
            .zip(img.chroma())
            .ok_or_else(|| {
                Error::NotSupported(format!(
                    "{fmt:?} source is not a planar IOSurface and has no chroma plane of its own"
                ))
            })?;
        import_plane(display, chroma, 1, iosurface_import::GL_RG, w / 2, h / 2)
    }

    /// The luma extent of a multiplane NV12 tensor whose luma and chroma wrap
    /// one planar IOSurface at its plane 0 and plane 1.
    fn nv_native_planes(img: &Tensor<u8>) -> Option<(usize, usize)> {
        planar_nv12_extent(img)
    }

    /// A destination [`Self::nv_native_planes`] accepts imports `Split`:
    /// plane 0 as `GL_RED` and plane 1 as `GL_RG` at half the luma extent.
    /// Any other imports its whole surface as `Combined`.
    fn import_nv12_dst(
        display: &AngleDisplay,
        img: &Tensor<u8>,
    ) -> Result<NvDstPlanes<IoSurfacePbuffer>> {
        if let (Some((w, h)), Some(chroma)) = (Self::nv_native_planes(img), img.chroma()) {
            return Ok(NvDstPlanes::Split {
                luma: import_plane(display, img, 0, iosurface_import::GL_RED, w, h)?,
                chroma: import_plane(display, chroma, 1, iosurface_import::GL_RG, w / 2, h / 2)?,
                chroma_rows: (h / 2) as u32,
                chroma_format: Self::NV12_SPLIT_CHROMA,
            });
        }
        let (width, height) = img.iosurface_physical_dims().ok_or_else(|| {
            Error::NotSupported(
                "GL convert: NV12 destination has no IOSurface physical dims".into(),
            )
        })?;
        Ok(NvDstPlanes::Combined {
            plane: import_nv_combined(display, img)?,
            width: width as u32,
            height: height as u32,
        })
    }
}

/// The luma extent of a multiplane NV12 tensor whose luma and chroma wrap
/// one planar IOSurface at its plane 0 and plane 1.
fn planar_nv12_extent<T>(img: &Tensor<T>) -> Option<(usize, usize)>
where
    T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
{
    if img.format() != Some(PixelFormat::Nv12) {
        return None;
    }
    let chroma = img.chroma()?;
    let id = iosurface_import::surface_id(img)?;
    if iosurface_import::surface_id(chroma) != Some(id) {
        return None;
    }
    let at_plane = |t: &Tensor<T>, plane: i32| {
        let offset = t.plane_offset().unwrap_or(0);
        (iosurface_import::plane_index(t, offset) == Some(plane)).then_some(())
    };
    at_plane(img, 0)?;
    at_plane(chroma, 1)?;
    Some((img.width()?, img.height()?))
}

/// Binds a tensor's whole IOSurface as one R8 (`L008`) texture at the
/// surface's physical extent, for a shader that addresses Y and UV texels
/// within it.
fn import_nv_combined(display: &AngleDisplay, img: &Tensor<u8>) -> Result<IoSurfacePbuffer> {
    super::refuse_offset_source(img, "NV source", "EGL_ANGLE_iosurface_client_buffer")?;
    let surface_ref = img.iosurface_ref().ok_or_else(|| {
        Error::NotSupported("GL convert: NV source is not IOSurface-backed".into())
    })?;
    let (pw, ph) = img.iosurface_physical_dims().ok_or_else(|| {
        Error::NotSupported("GL convert: NV source has no IOSurface physical dims".into())
    })?;
    let shared = display.shared;
    // SAFETY: `surface_ref` is borrowed from a live tensor.
    let surface = unsafe {
        iosurface_import::create_iosurface_pbuffer(
            &shared.egl,
            shared.display,
            shared.config,
            surface_ref,
            PixelFormat::Grey,
            DType::U8,
            pw,
            ph,
        )?
    };
    Ok(IoSurfacePbuffer { shared, surface })
}

/// Binds plane `plane` of `img`'s IOSurface as a `width`x`height` texture of
/// `internal_format`.
fn import_plane(
    display: &AngleDisplay,
    img: &Tensor<u8>,
    plane: i32,
    internal_format: i32,
    width: usize,
    height: usize,
) -> Result<IoSurfacePbuffer> {
    let surface_ref = img.iosurface_ref().ok_or_else(|| {
        Error::NotSupported("GL convert: NV plane is not IOSurface-backed".into())
    })?;
    let shared = display.shared;
    // SAFETY: `surface_ref` is borrowed from a live tensor whose surface has
    // plane `plane` of at least `width`x`height`.
    let surface = unsafe {
        iosurface_import::create_iosurface_plane_pbuffer(
            &shared.egl,
            shared.display,
            shared.config,
            surface_ref,
            plane,
            internal_format,
            width,
            height,
        )?
    };
    Ok(IoSurfacePbuffer { shared, surface })
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use std::time::Instant;

    /// Per-processor context bring-up takes under 50 ms each.
    #[test]
    #[ignore = "needs ANGLE + Apple GPU; run on demand"]
    fn per_processor_context_bring_up_latency() {
        // The first call also pays shared-display init; it is timed separately.
        let t0 = Instant::now();
        let first = AngleClientBuffer::init_display(None).expect("first display");
        let first_ms = t0.elapsed().as_secs_f64() * 1e3;
        drop(first);

        let handles: Vec<_> = (0..4)
            .map(|i| {
                std::thread::spawn(move || {
                    let t0 = Instant::now();
                    let d = AngleClientBuffer::init_display(None).expect("display");
                    let ms = t0.elapsed().as_secs_f64() * 1e3;
                    drop(d);
                    (i, ms)
                })
            })
            .collect();
        for h in handles {
            let (i, ms) = h.join().expect("thread");
            println!("per-processor context {i}: {ms:.1} ms");
            assert!(
                ms < 50.0,
                "context bring-up {ms:.1} ms exceeds 50 ms budget"
            );
        }
        println!("first (incl. shared display init): {first_ms:.1} ms");
    }
}
