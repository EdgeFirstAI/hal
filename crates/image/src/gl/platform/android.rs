// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Android implementation of [`GlPlatform`]: native EGL display bring-up
//! and AHardwareBuffer buffer import.
//!
//! Android is a hybrid of the two existing backends:
//!
//! * **Import mechanism = Linux.** The AHardwareBuffer becomes an
//!   `EGLImage` (`ahardwarebuffer_import.rs`) bound to a `GL_TEXTURE_2D`
//!   via `glEGLImageTargetTexture2DOES`, and the binding **persists** on
//!   the texture object across GPU passes — `PERSISTENT_TEX_BINDINGS =
//!   true`, the engine's binding-skip cache applies, `end_gpu_pass` is a
//!   no-op. (iOS/ANGLE releases its pbuffer binds per pass.)
//! * **Dispatch semantics = macOS (for now).** Imports bind as
//!   `TEXTURE_2D`; `EXTERNAL_OES = false`. Android's driver does expose
//!   `GL_OES_EGL_image_external`, but only the RGBA8→TEXTURE_2D import
//!   was validated by the Phase-1 probe — flipping to the external-OES
//!   YUV sampler for camera NV12 buffers is a planned follow-up gated on
//!   on-device testing.
//! * **Display bring-up = its own.** `eglGetDisplay(EGL_DEFAULT_DISPLAY)`
//!   on the real GPU driver (Adreno/Mali/…): no ANGLE translation layer,
//!   no GBM/PlatformDevice probing. Android ships a first-class GLES
//!   implementation, so `EglDisplayKind` is ignored here like on macOS.
//!
//! The bring-up flow (shared display + per-processor private context +
//! dummy pbuffer, made current once and held for the worker thread's
//! life) is the [`super::angle`] shape, productized from the validated
//! Phase-1 probe (`hal-mobile/crates/edgefirst-android-probe`).
//!
//! ## Lifecycle caveat: GPU reset / context loss
//!
//! Because the HAL renders offscreen (no window surface), ordinary app
//! backgrounding does NOT invalidate anything here. A GPU reset /
//! `EGL_CONTEXT_LOST`, however, would poison the leaked shared display,
//! every per-processor context, and all cached EGLImages with no rebuild
//! path — the same (rare, catastrophic) stance as the Linux and macOS
//! backends, but statistically more likely on mobile. Detection and
//! rebuild are a known follow-up shared across platforms; embedding apps
//! that must survive GPU resets should recreate their `ImageProcessor`
//! in a fresh process.

use super::super::{ahardwarebuffer_import, Egl};
use super::GlPlatform;
use crate::{Error, Result};
use edgefirst_egl as egl;
use edgefirst_tensor::{PixelFormat, Tensor};
use log::{debug, warn};
use std::ffi::c_void;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// EGL constants shared by display bring-up and per-processor context
// creation (from EGL headers; edgefirst_egl does not export these).
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

/// Cached libEGL handle. Android's system `libEGL.so` lives on the
/// default linker search path. Leaked at first load to avoid
/// dlclose-during-shutdown crashes (same pattern as Linux's `EGL_LIB` in
/// `context.rs` and the macOS ANGLE loader).
static EGL_LIB: OnceLock<&'static libloading::Library> = OnceLock::new();

fn load_egl_lib() -> Result<&'static libloading::Library> {
    if let Some(lib) = EGL_LIB.get() {
        return Ok(lib);
    }
    // SAFETY: dlopen is unsafe because the loaded library can run
    // initializers. The Android system libEGL is well-behaved.
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
    /// F16 destination tensors (the primary NPU-input path) — confirmed
    /// on Adreno 840 and the API-36 emulator by the Phase-1 probe.
    pub(in crate::opengl_headless) supports_f16_color: bool,
}

// SAFETY: every member is either a leaked static or an EGL handle;
// Android's system EGL synchronizes display-level entry points
// internally (specification-conformant), same contract as ANGLE.
unsafe impl Send for SharedAndroidDisplay {}
unsafe impl Sync for SharedAndroidDisplay {}

static SHARED_DISPLAY: OnceLock<std::result::Result<SharedAndroidDisplay, String>> =
    OnceLock::new();

/// Acquire a reference to the process-global native EGL display,
/// initialising it on first call. The error case is cached too — once EGL
/// fails to load we don't keep retrying.
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

    // 2. Native default display — the real GPU driver, no ANGLE.
    let display = unsafe { egl.get_display(egl::DEFAULT_DISPLAY) }.ok_or_else(|| {
        Error::Io(std::io::Error::other(
            "eglGetDisplay(EGL_DEFAULT_DISPLAY) returned NO_DISPLAY",
        ))
    })?;
    let (maj, min) = egl
        .initialize(display)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglInitialize: {e:?}"))))?;
    debug!("Native Android EGL {maj}.{min} initialised (process-global shared display)");

    egl.bind_api(egl::OPENGL_ES_API)
        .map_err(|e| Error::Io(std::io::Error::other(format!("eglBindAPI: {e:?}"))))?;

    // 3. GLES3 + pbuffer config (offscreen only — the HAL renders to FBOs
    //    backed by AHardwareBuffer EGLImages; no window surface exists).
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

    // 4. Throwaway context so GL symbols and extensions can be probed.
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

    // 5. Load the GL function-pointer table once per process, via the
    //    now-initialised display (drivers may need a current context to
    //    expose GLES symbols through eglGetProcAddress).
    load_gl_once_inner(&egl);

    // Probe the float-color-buffer extensions while the context is still
    // current. Adreno/Mali expose both on every device class the HAL
    // targets; consumers fall back per dtype where missing.
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
    // GL_OES_EGL_image is the extension behind glEGLImageTargetTexture2DOES
    // — the ONLY way any AHardwareBuffer reaches the GPU here. Universal on
    // Android GLES drivers, but if it were ever absent the attach call
    // would be a null function pointer (segfault); fail the display
    // bring-up cleanly instead so the engine falls back to CPU.
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

    // The throwaway probe context is torn down; each processor creates
    // its own private context in `init_display`.
    let _ = egl.make_current(display, None, None, None);
    let _ = egl.destroy_surface(display, dummy);
    let _ = egl.destroy_context(display, context);

    Ok(SharedAndroidDisplay {
        egl,
        display,
        config,
        supports_f32_color,
        supports_f16_color,
    })
}

// ---------------------------------------------------------------------------
// One-shot GL function-pointer table (same rationale as angle.rs).
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

/// One processor's GL bring-up state: a PRIVATE EGL context (plus dummy
/// pbuffer) on the process-global shared native display.
///
/// Created on the processor's dedicated worker thread, made current there
/// ONCE, and held current for the thread's life — the Linux `GlContext` /
/// macOS `AngleDisplay` model. NOT `Send`: the context is current on its
/// creating thread and must be dropped there too (the dispatch wrapper
/// guarantees both).
pub(in crate::opengl_headless) struct AndroidGlContext {
    pub(in crate::opengl_headless) shared: &'static SharedAndroidDisplay,
    context: egl::Context,
    dummy_pbuffer: egl::Surface,
    /// Duck-typed counterparts of the two `GlContext` members the
    /// portable engine reads. AHardwareBuffer is the only zero-copy
    /// transfer backend on Android. GLES 3.1+ compute exists on the
    /// native drivers but is untested by the Phase-1 probe, so it stays
    /// off until validated on-device (the engine's CPU fallbacks apply).
    pub(in crate::opengl_headless) transfer_backend: super::super::TransferBackend,
    pub(in crate::opengl_headless) has_compute: bool,
}

impl Drop for AndroidGlContext {
    fn drop(&mut self) {
        // Runs on the owning worker thread (the only place the wrapper
        // drops it): release the context from this thread, then destroy.
        let d = self.shared;
        let _ = d.egl.make_current(d.display, None, None, None);
        let _ = d.egl.destroy_surface(d.display, self.dummy_pbuffer);
        let _ = d.egl.destroy_context(d.display, self.context);
    }
}

/// An owned AHardwareBuffer→EGLImage import. Dropping destroys the
/// EGLImage (the AHardwareBuffer itself stays owned by the tensor; the
/// import cache's `Weak` guard ties the two lifetimes together). Cached
/// in `ImportCache<AndroidEglImage>` keyed by `BufferImportKey`, exactly
/// like Linux `EglImage`s.
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

/// Extract the tensor's AHardwareBuffer pointer, or explain why the
/// zero-copy import cannot proceed.
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

    // EGLImage targets persist on the texture object — the engine's
    // binding-skip cache (BufferImportKey on Texture) applies, exactly
    // like Linux DMA-BUF.
    const PERSISTENT_TEX_BINDINGS: bool = true;
    // Only the RGBA8/RGBA16F → TEXTURE_2D import was probe-validated;
    // the external-OES YUV sampler flip is a planned on-device follow-up.
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
        // No share-list: each processor owns its programs/VAOs/FBOs, the
        // same isolation Linux and macOS contexts have.
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
        // Made current ONCE on the calling (worker) thread and held for
        // the thread's life.
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

    unsafe fn attach_tex_image_2d(_display: &AndroidGlContext, handle: egl::Image) -> Result<()> {
        edgefirst_gl::gl::EGLImageTargetTexture2DOES(edgefirst_gl::gl::TEXTURE_2D, handle.as_ptr());
        Ok(())
    }

    unsafe fn attach_tex_image_external(
        _display: &AndroidGlContext,
        _handle: egl::Image,
    ) -> Result<()> {
        // Unreachable in practice: PlatformCaps::external_oes mirrors
        // EXTERNAL_OES (false), so path selection never picks the
        // external-sampler path. Becomes real when the YUV camera flip
        // lands.
        Err(Error::NotSupported(
            "GL_TEXTURE_EXTERNAL_OES sampling is not enabled on Android yet".into(),
        ))
    }

    unsafe fn attach_renderbuffer_storage(
        _display: &AndroidGlContext,
        handle: egl::Image,
    ) -> Result<()> {
        edgefirst_gl::gl::EGLImageTargetRenderbufferStorageOES(
            edgefirst_gl::gl::RENDERBUFFER,
            handle.as_ptr(),
        );
        Ok(())
    }

    fn end_gpu_pass(_display: &AndroidGlContext) {
        // EGLImage texture targets persist by design; nothing to release.
    }

    fn import_buffer(
        display: &AndroidGlContext,
        img: &Tensor<u8>,
        fmt: PixelFormat,
        _for_dst: bool,
    ) -> Result<AndroidEglImage> {
        // Semi-planar YUV import needs the external-OES sampler (or the
        // API-29 R8 plane binding) — both deferred. NV tensors cannot be
        // AHardwareBuffer-backed today anyway (no format mapping at
        // allocation), so this arm is defensive.
        if matches!(
            fmt,
            PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24
        ) {
            return Err(Error::NotSupported(
                "AHardwareBuffer import has no NV binding yet (external-OES flip pending)".into(),
            ));
        }
        // The AHardwareBuffer is self-describing (format/dims/stride fixed
        // at allocation), so no geometry or view handling is needed here:
        // a dst view()/batch() tile shares its parent's buffer pointer,
        // which is exactly what gets imported — the per-tile offset is
        // viewport state, and `BufferImportKey` already collapses sibling
        // views onto the parent import.
        let ptr = tensor_ahb_ptr(img)?;
        let shared = display.shared;
        // SAFETY: ptr borrowed from a live tensor; the import cache
        // entry's `guard` ties the EGLImage's lifetime to it.
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
        // The single-plane R8 AHardwareBuffer format needs API 29; the HAL
        // floor is 26, and NV tensors are not AHardwareBuffer-backed today
        // (no allocation mapping), so the R8 "Path B" has no zero-copy
        // source on Android yet.
        Err(Error::NotSupported(
            "AHardwareBuffer R8 NV binding requires API 29 (HAL floor is 26)".into(),
        ))
    }

    fn import_buffer_packed<T>(
        display: &AndroidGlContext,
        img: &Tensor<T>,
        width: usize,
        height: usize,
        _fmt: super::PackedImportFormat,
    ) -> Result<AndroidEglImage>
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync,
    {
        let ptr = tensor_ahb_ptr(img)?;
        // The AHardwareBuffer is self-describing — the import cannot
        // reinterpret it at different dimensions the way the Linux DMA-BUF
        // attribute list can. The caller's packed dims must therefore
        // match the buffer's own physical descriptor (they do for every
        // tensor allocated by `Tensor::image` — `new_image` sizes the
        // buffer with the same `packed_rgba16f_layout` the caller used);
        // a mismatch means the buffer was allocated with different
        // geometry and sampling it would read the wrong pitch.
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
