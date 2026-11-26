// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Android implementation of [`GlPlatform`]: native EGL display bring-up
//! and AHardwareBuffer buffer import.
//!
//! * Import: the AHardwareBuffer becomes an `EGLImage`
//!   (`ahardwarebuffer_import.rs`) bound to a `GL_TEXTURE_2D` via
//!   `glEGLImageTargetTexture2DOES`. The binding persists on the texture
//!   object across GPU passes (a constant `tex_binding_epoch`), and
//!   `end_gpu_pass` is a no-op.
//! * Dispatch: imports bind as `TEXTURE_2D`; `EXTERNAL_OES = false`.
//! * Display bring-up: `eglGetDisplay(EGL_DEFAULT_DISPLAY)` on the native
//!   GPU driver. `EglDisplayKind` is ignored.
//!
//! The bring-up flow is a shared display plus a per-processor private
//! context and dummy pbuffer, made current once and held for the worker
//! thread's life.
//!
//! A GPU reset / `EGL_CONTEXT_LOST` is not detected: the shared display,
//! every per-processor context and all cached EGLImages have no rebuild
//! path.

use super::super::{ahardwarebuffer_import, native_fence, Egl};
use super::GlPlatform;
use crate::{Error, Result};
use edgefirst_egl as egl;
use edgefirst_tensor::{PixelFormat, Tensor, TensorDyn};
use log::{debug, warn};
use std::ffi::c_void;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// EGL constants shared by display bring-up and per-processor context
// creation (from EGL headers).
// ---------------------------------------------------------------------------

const EGL_OPENGL_ES3_BIT: i32 = 0x0040;
const EGL_PBUFFER_BIT: i32 = 0x0001;
const EGL_RENDERABLE_TYPE: i32 = 0x3040;
const EGL_SURFACE_TYPE: i32 = 0x3033;
const EGL_RED_SIZE: i32 = 0x3024;
const EGL_GREEN_SIZE: i32 = 0x3023;
const EGL_BLUE_SIZE: i32 = 0x3022;
const EGL_ALPHA_SIZE: i32 = 0x3021;
const EGL_CONTEXT_CLIENT_VERSION: i32 = 0x3098;

/// Cached libEGL handle for Android's system `libEGL.so`. Leaked at first
/// load.
static EGL_LIB: OnceLock<&libloading::Library> = OnceLock::new();

fn load_egl_lib() -> Result<&'static libloading::Library> {
    if let Some(lib) = EGL_LIB.get() {
        return Ok(lib);
    }
    // SAFETY: the Android system libEGL's initializers are sound to run.
    let lib = unsafe { libloading::Library::new("libEGL.so") }.map_err(|e| {
        Error::Io(std::io::Error::other(format!(
            "failed to load libEGL.so: {e}"
        )))
    })?;
    let leaked: &'static libloading::Library = Box::leak(Box::new(lib));
    Ok(EGL_LIB.get_or_init(|| leaked))
}

// ---------------------------------------------------------------------------
// Process-global native EGL display.
// ---------------------------------------------------------------------------

/// All process-global EGL state. Use [`shared_display`] to access.
pub(in crate::opengl_headless) struct SharedAndroidDisplay {
    pub(in crate::opengl_headless) egl: Egl,
    pub(in crate::opengl_headless) display: egl::Display,
    pub(in crate::opengl_headless) config: egl::Config,
    /// `GL_EXT_color_buffer_float` is exposed by this driver. Gates F32
    /// destination tensors on the AHardwareBuffer render path.
    pub(in crate::opengl_headless) supports_f32_color: bool,
    /// `GL_EXT_color_buffer_half_float` is exposed by this driver. Gates
    /// F16 destination tensors.
    pub(in crate::opengl_headless) supports_f16_color: bool,
    /// `EGL_ANDROID_native_fence_sync` (+ `EGL_KHR_fence_sync`) exposed
    /// by this display, as probed from the EGL extension string. Gates
    /// `export_completion_fence`.
    pub(in crate::opengl_headless) supports_native_fence: bool,
}

// SAFETY: every member is a leaked static or an EGL handle, and the system
// EGL synchronizes display-level entry points internally.
unsafe impl Send for SharedAndroidDisplay {}
unsafe impl Sync for SharedAndroidDisplay {}

static SHARED_DISPLAY: OnceLock<std::result::Result<SharedAndroidDisplay, String>> =
    OnceLock::new();

/// Acquire a reference to the process-global native EGL display,
/// initialising it on first call. A failed initialisation is cached.
pub(in crate::opengl_headless) fn shared_display() -> Result<&'static SharedAndroidDisplay> {
    SHARED_DISPLAY
        .get_or_init(|| init_shared_display().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|s| Error::Io(std::io::Error::other(s.clone())))
}

fn init_shared_display() -> Result<SharedAndroidDisplay> {
    let _span = tracing::info_span!(
        "image.gl_init",
        platform = "android",
        backend = "ahardwarebuffer",
    )
    .entered();

    // 1. Load the system libEGL and bring up an EGL instance.
    let egl_lib = load_egl_lib()?;
    let egl: Egl = unsafe {
        edgefirst_egl::Instance::<
            edgefirst_egl::Dynamic<&'static libloading::Library, edgefirst_egl::EGL1_4>,
        >::load_required_from(egl_lib)
    }
    .map_err(|e| Error::Io(std::io::Error::other(format!("EGL load: {e:?}"))))?;

    // 2. Native default display.
    let display = unsafe { egl.get_display(egl::DEFAULT_DISPLAY) }.ok_or_else(|| {
        Error::Io(std::io::Error::other(
            "eglGetDisplay(EGL_DEFAULT_DISPLAY) returned NO_DISPLAY",
        ))
    })?;
    let (maj, min) = egl
        .initialize(display)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglInitialize: {e:?}"))))?;
    debug!("Native Android EGL {maj}.{min} initialised (process-global shared display)");

    // EGL (display-level) extension probe for the native fence.
    let egl_extensions = egl
        .query_string(Some(display), egl::EXTENSIONS)
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let supports_native_fence = native_fence::display_supports_native_fence(&egl_extensions);
    debug!("Android EGL: EGL_ANDROID_native_fence_sync={supports_native_fence}");

    egl.bind_api(egl::OPENGL_ES_API)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglBindAPI: {e:?}"))))?;

    // 3. GLES3 + pbuffer config.
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
        egl::NONE,
    ];
    let config = egl
        .choose_first_config(display, &cfg_attribs)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglChooseConfig: {e:?}"))))?
        .ok_or_else(|| Error::NotSupported("no EGL config with GLES3+PBUFFER+RGBA8".into()))?;

    // 4. Throwaway context for probing GL symbols and extensions.
    let ctx_attribs = [EGL_CONTEXT_CLIENT_VERSION, 3, egl::NONE];
    let context = egl
        .create_context(display, config, None, &ctx_attribs)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglCreateContext: {e:?}"))))?;
    let dummy_attribs = [egl::WIDTH, 16, egl::HEIGHT, 16, egl::NONE];
    let dummy = egl
        .create_pbuffer_surface(display, config, &dummy_attribs)
        .map_err(|e| {
            let _ = egl.destroy_context(display, context);
            Error::Io(std::io::Error::other(format!(
                "eglCreatePbufferSurface(dummy): {e:?}"
            )))
        })?;
    if let Err(e) = egl.make_current(display, Some(dummy), Some(dummy), Some(context)) {
        let _ = egl.destroy_surface(display, dummy);
        let _ = egl.destroy_context(display, context);
        return Err(Error::Io(std::io::Error::other(format!(
            "eglMakeCurrent(dummy): {e:?}"
        ))));
    }

    // 5. Load the GL function-pointer table once per process.
    load_gl_once_inner(&egl);

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
    // Fail the display bring-up when GL_OES_EGL_image, the extension behind
    // glEGLImageTargetTexture2DOES, is missing.
    if !extensions
        .split_ascii_whitespace()
        .any(|e| e == "GL_OES_EGL_image")
    {
        let _ = egl.make_current(display, None, None, None);
        let _ = egl.destroy_surface(display, dummy);
        let _ = egl.destroy_context(display, context);
        return Err(Error::NotSupported(
            "GL_OES_EGL_image not exposed by this driver — the AHardwareBuffer \
             import path cannot work"
                .into(),
        ));
    }
    let supports_f32_color = extensions
        .split_ascii_whitespace()
        .any(|e| e == "GL_EXT_color_buffer_float");
    let supports_f16_color = extensions
        .split_ascii_whitespace()
        .any(|e| e == "GL_EXT_color_buffer_half_float");
    debug!(
        "Android GLES: GL_EXT_color_buffer_float={supports_f32_color}, \
         GL_EXT_color_buffer_half_float={supports_f16_color}"
    );
    if !supports_f16_color {
        warn!(
            "GL_EXT_color_buffer_half_float not available — F16 render \
             destinations will fall back per RenderDtypeSupport"
        );
    }

    // Tear down the throwaway probe context.
    let _ = egl.make_current(display, None, None, None);
    let _ = egl.destroy_surface(display, dummy);
    let _ = egl.destroy_context(display, context);

    Ok(SharedAndroidDisplay {
        egl,
        display,
        config,
        supports_f32_color,
        supports_f16_color,
        supports_native_fence,
    })
}

// ---------------------------------------------------------------------------
// One-shot GL function-pointer table.
// ---------------------------------------------------------------------------

static GL_LOADED: OnceLock<()> = OnceLock::new();

fn load_gl_once_inner(egl: &Egl) {
    GL_LOADED.get_or_init(|| {
        edgefirst_gl::load_with(|name| match egl.get_proc_address(name) {
            Some(ptr) => ptr as *const c_void,
            None => std::ptr::null(),
        });
    });
}

// ---------------------------------------------------------------------------
// Per-processor display: a private context on the shared native display.
// ---------------------------------------------------------------------------

/// One processor's GL bring-up state: a private EGL context (plus dummy
/// pbuffer) on the process-global shared native display.
///
/// Created on the processor's dedicated worker thread, made current there
/// once, and held current for the thread's life. Not `Send`: the context is
/// current on its creating thread and must be dropped there.
pub(in crate::opengl_headless) struct AndroidGlContext {
    pub(in crate::opengl_headless) shared: &'static SharedAndroidDisplay,
    context: egl::Context,
    dummy_pbuffer: egl::Surface,
    /// Counterparts of the two `GlContext` members the portable engine
    /// reads: the AHardwareBuffer transfer backend, and `has_compute` off.
    pub(in crate::opengl_headless) transfer_backend: super::super::TransferBackend,
    pub(in crate::opengl_headless) has_compute: bool,
}

impl Drop for AndroidGlContext {
    fn drop(&mut self) {
        // Release the context from this thread, then destroy it.
        let d = self.shared;
        let _ = d.egl.make_current(d.display, None, None, None);
        let _ = d.egl.destroy_surface(d.display, self.dummy_pbuffer);
        let _ = d.egl.destroy_context(d.display, self.context);
    }
}

/// An owned AHardwareBuffer→EGLImage import. Dropping destroys the
/// EGLImage; the AHardwareBuffer stays owned by the tensor. Cached in
/// `ImportCache<AndroidEglImage>` keyed by `BufferImportKey`.
pub(in crate::opengl_headless) struct AndroidEglImage {
    shared: &'static SharedAndroidDisplay,
    pub(in crate::opengl_headless) image: egl::Image,
}

impl Drop for AndroidEglImage {
    fn drop(&mut self) {
        ahardwarebuffer_import::destroy_ahardwarebuffer_eglimage(
            &self.shared.egl,
            self.shared.display,
            self.image,
        );
    }
}

/// Extract the tensor's AHardwareBuffer pointer, or return an error when
/// the tensor has none.
fn tensor_ahb_ptr<T>(img: &Tensor<T>) -> Result<*mut c_void>
where
    T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync,
{
    img.hardware_buffer_ptr().ok_or_else(|| {
        Error::NotSupported("GL convert: tensor is not AHardwareBuffer-backed".into())
    })
}

/// Marker type: Android native EGL + AHardwareBuffer platform.
pub(crate) struct AndroidEgl;

impl GlPlatform for AndroidEgl {
    type Display = AndroidGlContext;
    type Import = AndroidEglImage;
    type ImportHandle = egl::Image;

    fn tex_binding_epoch(_display: &AndroidGlContext) -> Option<u64> {
        // EGLImage targets persist on the texture object.
        Some(0)
    }

    /// The planar F16 zero-copy render only. Every other float tuple falls
    /// through to the CPU converter.
    const ZERO_COPY_FLOAT: super::super::float_dispatch::ZeroCopyFloatSet =
        super::super::float_dispatch::ZeroCopyFloatSet::PlanarF16;
    const EXTERNAL_OES: bool = false;

    fn load_gl_once(display: &AndroidGlContext) {
        // Loaded at shared-display init; re-entry is a no-op.
        load_gl_once_inner(&display.shared.egl);
    }

    fn init_display(kind: Option<super::super::EglDisplayKind>) -> Result<AndroidGlContext> {
        if let Some(kind) = kind {
            debug!("EglDisplayKind::{kind} ignored on Android — the native default display is the only one");
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
        Ok(AndroidGlContext {
            shared,
            context,
            dummy_pbuffer,
            transfer_backend: super::super::TransferBackend::AHardwareBuffer,
            has_compute: false,
        })
    }

    fn import_handle(import: &AndroidEglImage) -> egl::Image {
        import.image
    }

    fn import_extent(_import: &AndroidEglImage) -> Option<(u32, u32)> {
        // The AHardwareBuffer is the tensor's logical image.
        None
    }

    unsafe fn attach_tex_image_2d(_display: &AndroidGlContext, handle: egl::Image) -> Result<()> {
        unsafe {
            edgefirst_gl::gl::EGLImageTargetTexture2DOES(
                edgefirst_gl::gl::TEXTURE_2D,
                handle.as_ptr(),
            );
        }
        Ok(())
    }

    unsafe fn attach_tex_image_external(
        _display: &AndroidGlContext,
        _handle: egl::Image,
    ) -> Result<()> {
        Err(Error::NotSupported(
            "GL_TEXTURE_EXTERNAL_OES sampling is not enabled on Android yet".into(),
        ))
    }

    unsafe fn attach_renderbuffer_storage(
        _display: &AndroidGlContext,
        handle: egl::Image,
    ) -> Result<()> {
        unsafe {
            edgefirst_gl::gl::EGLImageTargetRenderbufferStorageOES(
                edgefirst_gl::gl::RENDERBUFFER,
                handle.as_ptr(),
            );
        }
        Ok(())
    }

    fn begin_gpu_pass(_display: &AndroidGlContext) {
        // No-op on native EGL.
    }

    fn end_gpu_pass(_display: &AndroidGlContext) {
        // Nothing to release.
    }

    fn native_fence_sync(display: &AndroidGlContext) -> bool {
        display.shared.supports_native_fence
    }

    fn export_completion_fence(
        display: &AndroidGlContext,
        _recorded: Option<u64>,
    ) -> crate::Result<Option<super::super::CompletionFence>> {
        if !display.shared.supports_native_fence {
            return Ok(None);
        }
        native_fence::export_native_fence_fd(&display.shared.egl, display.shared.display).map(Some)
    }

    fn record_completion(
        _display: &AndroidGlContext,
        _dst: &mut TensorDyn,
        _submit: bool,
    ) -> Option<u64> {
        None
    }

    fn submit_recorded(_display: &AndroidGlContext) {
        // No device fence: nothing to submit.
    }

    fn import_buffer(
        display: &AndroidGlContext,
        img: &Tensor<u8>,
        fmt: PixelFormat,
        _for_dst: bool,
    ) -> Result<AndroidEglImage> {
        if matches!(
            fmt,
            PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
        ) {
            return Err(Error::NotSupported(
                "AHardwareBuffer import has no NV binding yet (external-OES flip pending)".into(),
            ));
        }
        // A destination view()/batch() tile shares its parent's buffer
        // pointer, which is what gets imported.
        let ptr = tensor_ahb_ptr(img)?;
        let shared = display.shared;
        // SAFETY: `ptr` is borrowed from a live tensor.
        let image = unsafe {
            ahardwarebuffer_import::create_ahardwarebuffer_eglimage(
                &shared.egl,
                shared.display,
                ptr,
            )?
        };
        Ok(AndroidEglImage { shared, image })
    }

    fn import_buffer_nv_r8(
        _display: &AndroidGlContext,
        _img: &Tensor<u8>,
        _fmt: PixelFormat,
    ) -> Result<AndroidEglImage> {
        Err(Error::NotSupported(
            "AHardwareBuffer R8 NV binding requires API 29 (HAL floor is 26)".into(),
        ))
    }

    fn import_buffer_packed<T>(
        display: &AndroidGlContext,
        img: &Tensor<T>,
        width: usize,
        height: usize,
        fmt: super::PackedImportFormat,
    ) -> Result<AndroidEglImage>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        if fmt == super::PackedImportFormat::Rgba32323232F {
            return Err(Error::NotSupported(
                "RGBA32F zero-copy import has no buffer format on this platform".into(),
            ));
        }
        let ptr = tensor_ahb_ptr(img)?;
        // The caller's packed dims must match the buffer's own physical
        // descriptor.
        if let Some((pw, ph)) = img.hardware_buffer_physical_dims() {
            if (pw, ph) != (width, height) {
                return Err(Error::NotSupported(format!(
                    "packed import: AHardwareBuffer physical dims {pw}x{ph} do not match the \
                     requested packed surface {width}x{height}"
                )));
            }
        }
        let shared = display.shared;
        // SAFETY: as in `import_buffer`.
        let image = unsafe {
            ahardwarebuffer_import::create_ahardwarebuffer_eglimage(
                &shared.egl,
                shared.display,
                ptr,
            )?
        };
        Ok(AndroidEglImage { shared, image })
    }
}
