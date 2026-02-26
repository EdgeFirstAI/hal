// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
//! EGL context bootstrap for headless GPU probing.
//!
//! Opens a DRM render node, creates a GBM device, obtains an EGL platform
//! display, and initialises a GLES 3.0 PBuffer context. This is a standalone
//! version of the HAL's `opengl_headless::GlContext` stripped down for
//! diagnostic / benchmarking use only.

use gbm::{
    drm::{control::Device as DrmControlDevice, Device as DrmDevice},
    AsRaw, Device,
};
use khronos_egl::{self as egl, Attrib, Display, Dynamic, Instance, EGL1_4};
use log::{debug, info};
use std::{
    collections::BTreeSet,
    ffi::{c_char, c_void, CStr},
    os::fd::AsFd,
    ptr::null_mut,
    sync::OnceLock,
};

// ---------------------------------------------------------------------------
// EGL / DMA-buf extension constants
// ---------------------------------------------------------------------------

const PLATFORM_GBM_KHR: u32 = 0x31D7;

#[allow(dead_code)]
const LINUX_DMA_BUF: u32 = 0x3270;
#[allow(dead_code)]
const LINUX_DRM_FOURCC: u32 = 0x3271;
#[allow(dead_code)]
const DMA_BUF_PLANE0_FD: u32 = 0x3272;
#[allow(dead_code)]
const DMA_BUF_PLANE0_OFFSET: u32 = 0x3273;
#[allow(dead_code)]
const DMA_BUF_PLANE0_PITCH: u32 = 0x3274;

// ---------------------------------------------------------------------------
// EGL library loader (leaked to avoid dlclose crashes)
// ---------------------------------------------------------------------------

type Egl = Instance<Dynamic<&'static libloading::Library, EGL1_4>>;

static EGL_LIB: OnceLock<&'static libloading::Library> = OnceLock::new();

fn get_egl_lib() -> Result<&'static libloading::Library, String> {
    if let Some(egl) = EGL_LIB.get() {
        Ok(egl)
    } else {
        let egl = unsafe { libloading::Library::new("libEGL.so.1") }
            .map_err(|e| format!("failed to load libEGL.so.1: {e}"))?;
        let egl: &'static libloading::Library = Box::leak(Box::new(egl));
        Ok(EGL_LIB.get_or_init(|| egl))
    }
}

// ---------------------------------------------------------------------------
// DRM card wrapper
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Card(std::fs::File);

impl AsFd for Card {
    fn as_fd(&self) -> std::os::unix::io::BorrowedFd<'_> {
        self.0.as_fd()
    }
}

impl DrmDevice for Card {}
impl DrmControlDevice for Card {}

impl Card {
    fn open(path: &str) -> Result<Self, String> {
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        options.write(true);
        options
            .open(path)
            .map(Card)
            .map_err(|e| format!("{path}: {e}"))
    }

    fn open_any() -> Result<Self, String> {
        let targets = ["/dev/dri/renderD128", "/dev/dri/card0", "/dev/dri/card1"];
        let mut last_err = String::new();
        for path in &targets {
            match Self::open(path) {
                Ok(card) => {
                    info!("Opened DRM node: {path}");
                    return Ok(card);
                }
                Err(e) => {
                    debug!("Skipping {path}: {e}");
                    last_err = e;
                }
            }
        }
        Err(format!(
            "no usable DRM render node found (last: {last_err})"
        ))
    }
}

// ---------------------------------------------------------------------------
// GpuContext
// ---------------------------------------------------------------------------

/// Holds an initialised EGL display, GLES 3.0 context, and the backing GBM
/// device. Provides helper methods for querying GL/EGL strings and creating
/// EGLImages from DMA-buf file descriptors.
pub struct GpuContext {
    egl: Egl,
    display: Display,
    surface: egl::Surface,
    ctx: egl::Context,
    // The GBM device must outlive the EGL display.
    _gbm: Device<Card>,
}

impl GpuContext {
    /// Bootstrap a headless GLES 3.0 context via GBM + EGL.
    pub fn new() -> Result<Self, String> {
        // 1. Open DRM render node and create GBM device
        let card = Card::open_any()?;
        let gbm = Device::new(card).map_err(|e| format!("GBM device creation failed: {e}"))?;
        debug!("GBM device: {gbm:?}");

        // 2. Load EGL
        let egl: Egl = unsafe {
            Instance::<Dynamic<_, EGL1_4>>::load_required_from(get_egl_lib()?)
                .map_err(|e| format!("EGL instance load failed: {e}"))?
        };

        // 3. Get platform display (EGL 1.5 -> EXT fallback)
        let display = egl_get_platform_display_with_fallback(
            &egl,
            PLATFORM_GBM_KHR,
            gbm.as_raw() as *mut c_void,
            &[egl::ATTRIB_NONE],
        )?;
        debug!("EGL display: {display:?}");

        // 4. Initialise EGL
        egl.initialize(display)
            .map_err(|e| format!("eglInitialize failed: {e}"))?;

        // 5. Choose config: GLES 3.0, RGBA8, PBUFFER
        let attributes = [
            egl::SURFACE_TYPE,
            egl::PBUFFER_BIT,
            egl::RENDERABLE_TYPE,
            egl::OPENGL_ES3_BIT,
            egl::RED_SIZE,
            8,
            egl::GREEN_SIZE,
            8,
            egl::BLUE_SIZE,
            8,
            egl::ALPHA_SIZE,
            8,
            egl::NONE,
        ];
        let config = egl
            .choose_first_config(display, &attributes)
            .map_err(|e| format!("eglChooseConfig failed: {e}"))?
            .ok_or("no suitable EGL config found (GLES3 RGBA8 PBuffer)")?;
        debug!("EGL config: {config:?}");

        // 6. Create 64x64 PBuffer surface
        let surface = egl
            .create_pbuffer_surface(
                display,
                config,
                &[egl::WIDTH, 64, egl::HEIGHT, 64, egl::NONE],
            )
            .map_err(|e| format!("eglCreatePbufferSurface failed: {e}"))?;

        // 7. Bind OPENGL_ES_API, create GLES 3.0 context, make current
        egl.bind_api(egl::OPENGL_ES_API)
            .map_err(|e| format!("eglBindAPI failed: {e}"))?;

        let context_attributes = [egl::CONTEXT_MAJOR_VERSION, 3, egl::NONE];
        let ctx = egl
            .create_context(display, config, None, &context_attributes)
            .map_err(|e| format!("eglCreateContext failed: {e}"))?;

        egl.make_current(display, Some(surface), Some(surface), Some(ctx))
            .map_err(|e| format!("eglMakeCurrent failed: {e}"))?;

        // 8. Load GL function pointers
        gls::load_with(|s| {
            egl.get_proc_address(s)
                .map_or(std::ptr::null(), |p| p as *const _)
        });

        Ok(GpuContext {
            egl,
            display,
            surface,
            ctx,
            _gbm: gbm,
        })
    }

    // -------------------------------------------------------------------
    // EGL string queries
    // -------------------------------------------------------------------

    /// Query an EGL string (e.g. `egl::VENDOR`, `egl::VERSION`).
    pub fn egl_query(&self, name: egl::Int) -> Option<String> {
        self.egl
            .query_string(Some(self.display), name)
            .ok()
            .map(|s| s.to_string_lossy().into_owned())
    }

    /// Sorted list of EGL extensions advertised by the display.
    pub fn egl_extensions(&self) -> Vec<String> {
        self.egl_query(egl::EXTENSIONS)
            .map(|s| {
                let mut exts: Vec<String> = s.split_ascii_whitespace().map(String::from).collect();
                exts.sort();
                exts
            })
            .unwrap_or_default()
    }

    // -------------------------------------------------------------------
    // GL string queries
    // -------------------------------------------------------------------

    /// GL_RENDERER string.
    pub fn gl_renderer(&self) -> String {
        gls::get_string(gls::gl::RENDERER).unwrap_or_else(|_| "<unknown>".into())
    }

    /// GL_VERSION string.
    pub fn gl_version(&self) -> String {
        gls::get_string(gls::gl::VERSION).unwrap_or_else(|_| "<unknown>".into())
    }

    /// GL_VENDOR string.
    pub fn gl_vendor(&self) -> String {
        gls::get_string(gls::gl::VENDOR).unwrap_or_else(|_| "<unknown>".into())
    }

    /// Sorted list of GL extensions.
    pub fn gl_extensions(&self) -> Vec<String> {
        let ext_str = unsafe {
            let ptr = gls::gl::GetString(gls::gl::EXTENSIONS);
            if ptr.is_null() {
                return Vec::new();
            }
            CStr::from_ptr(ptr as *const c_char)
                .to_string_lossy()
                .into_owned()
        };
        let mut exts: Vec<String> = ext_str
            .split_ascii_whitespace()
            .map(String::from)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        exts.sort();
        exts
    }

    /// Query a single GL integer parameter (e.g. `gls::gl::MAX_TEXTURE_SIZE`).
    pub fn gl_get_integer(&self, param: gls::GLenum) -> i32 {
        let mut value: i32 = 0;
        unsafe {
            gls::gl::GetIntegerv(param, &mut value);
        }
        value
    }

    /// Check whether the GPU supports RGB8 renderbuffers and the
    /// `glEGLImageTargetRenderbufferStorageOES` entry point needed to back
    /// them with an EGLImage.
    ///
    /// Returns `true` when **both** conditions are met:
    /// 1. RGB8 renderbuffer format is available — either via OpenGL ES 3.0+
    ///    (which mandates `GL_RGB8`) or the `GL_OES_rgb8_rgba8` extension.
    /// 2. The `glEGLImageTargetRenderbufferStorageOES` function pointer is
    ///    resolvable through `eglGetProcAddress`.
    pub fn has_rgb8_renderbuffer(&self) -> bool {
        let version = self.gl_version();
        let has_rgb8_format = if version.contains("OpenGL ES 3.") {
            true
        } else {
            let ext_str = unsafe {
                let ptr = gls::gl::GetString(gls::gl::EXTENSIONS);
                if ptr.is_null() {
                    return false;
                }
                CStr::from_ptr(ptr as *const c_char)
                    .to_string_lossy()
                    .into_owned()
            };
            ext_str
                .split_ascii_whitespace()
                .any(|e| e == "GL_OES_rgb8_rgba8")
        };

        if !has_rgb8_format {
            return false;
        }

        self.egl
            .get_proc_address("glEGLImageTargetRenderbufferStorageOES")
            .is_some()
    }


    // -------------------------------------------------------------------
    // EGLImage / DMA-buf helpers
    // -------------------------------------------------------------------

    /// Returns `true` if the EGL stack supports `eglCreateImage(KHR)` with
    /// `EGL_LINUX_DMA_BUF_EXT` target — required for zero-copy DMA-buf
    /// import.
    pub fn has_egl_create_image_khr(&self) -> bool {
        // EGL 1.5 has native create_image
        if self.egl.upcast::<egl::EGL1_5>().is_some() {
            return true;
        }
        // Fallback: check extension + entry point
        let exts = self
            .egl
            .query_string(Some(self.display), egl::EXTENSIONS)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let has_ext = exts
            .split_ascii_whitespace()
            .any(|e| e == "EGL_EXT_image_dma_buf_import");
        if !has_ext {
            return false;
        }
        self.egl.get_proc_address("eglCreateImageKHR").is_some()
            && self.egl.get_proc_address("eglDestroyImageKHR").is_some()
    }

    /// Create an EGLImage backed by a DMA-buf file descriptor.
    ///
    /// Uses EGL 1.5 `eglCreateImage` when available, otherwise falls back
    /// to `eglCreateImageKHR`.
    pub fn create_egl_image_dma(
        &self,
        fd: i32,
        width: i32,
        height: i32,
        drm_fourcc: u32,
        pitch: i32,
    ) -> Result<egl::Image, String> {
        let attribs: Vec<Attrib> = vec![
            egl::WIDTH as Attrib,
            width as Attrib,
            egl::HEIGHT as Attrib,
            height as Attrib,
            LINUX_DRM_FOURCC as Attrib,
            drm_fourcc as Attrib,
            DMA_BUF_PLANE0_FD as Attrib,
            fd as Attrib,
            DMA_BUF_PLANE0_OFFSET as Attrib,
            0,
            DMA_BUF_PLANE0_PITCH as Attrib,
            pitch as Attrib,
            egl::ATTRIB_NONE,
        ];

        egl_create_image_with_fallback(
            &self.egl,
            self.display,
            unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) },
            LINUX_DMA_BUF,
            unsafe { egl::ClientBuffer::from_ptr(null_mut()) },
            &attribs,
        )
    }

    /// Destroy a previously-created EGLImage.
    pub fn destroy_egl_image(&self, image: egl::Image) -> Result<(), String> {
        egl_destroy_image_with_fallback(&self.egl, self.display, image)
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        // Defence-in-depth cleanup: destroy context, surface, and terminate
        // the EGL display inside catch_unwind to absorb driver panics.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = self.egl.make_current(self.display, None, None, None);
            let _ = self.egl.destroy_context(self.display, self.ctx);
            let _ = self.egl.destroy_surface(self.display, self.surface);
            let _ = self.egl.terminate(self.display);
        }));
        std::panic::set_hook(prev_hook);
    }
}

// ---------------------------------------------------------------------------
// EGL platform display with fallback (EGL 1.5 -> EXT)
// ---------------------------------------------------------------------------

fn egl_get_platform_display_with_fallback(
    egl: &Egl,
    platform: egl::Enum,
    native_display: *mut c_void,
    attrib_list: &[Attrib],
) -> Result<Display, String> {
    // Try EGL 1.5 native path first
    if let Some(egl15) = egl.upcast::<egl::EGL1_5>() {
        return unsafe { egl15.get_platform_display(platform, native_display, attrib_list) }
            .map_err(|e| format!("eglGetPlatformDisplay (1.5) failed: {e}"));
    }

    // Fallback: eglGetPlatformDisplayEXT
    if let Some(ext) = egl.get_proc_address("eglGetPlatformDisplayEXT") {
        let func: unsafe extern "system" fn(
            platform: egl::Enum,
            native_display: *mut c_void,
            attrib_list: *const Attrib,
        ) -> egl::EGLDisplay = unsafe { std::mem::transmute(ext) };

        let disp = unsafe { func(platform, native_display, attrib_list.as_ptr()) };
        if disp != egl::NO_DISPLAY {
            return Ok(unsafe { Display::from_ptr(disp) });
        }
        return Err("eglGetPlatformDisplayEXT returned EGL_NO_DISPLAY".into());
    }

    Err("neither EGL 1.5 nor eglGetPlatformDisplayEXT available".into())
}

// ---------------------------------------------------------------------------
// EGLImage create / destroy with fallback
// ---------------------------------------------------------------------------

fn egl_create_image_with_fallback(
    egl: &Egl,
    display: Display,
    ctx: egl::Context,
    target: egl::Enum,
    buffer: egl::ClientBuffer,
    attrib_list: &[Attrib],
) -> Result<egl::Image, String> {
    // EGL 1.5 native path
    if let Some(egl15) = egl.upcast::<egl::EGL1_5>() {
        return egl15
            .create_image(display, ctx, target, buffer, attrib_list)
            .map_err(|e| format!("eglCreateImage (1.5) failed: {e}"));
    }

    // Fallback: eglCreateImageKHR (uses EGLint attribs, not EGLAttrib)
    if let Some(ext) = egl.get_proc_address("eglCreateImageKHR") {
        let func: unsafe extern "system" fn(
            display: egl::EGLDisplay,
            ctx: egl::EGLContext,
            target: egl::Enum,
            buffer: egl::EGLClientBuffer,
            attrib_list: *const egl::Int,
        ) -> egl::EGLImage = unsafe { std::mem::transmute(ext) };

        let int_attribs: Vec<egl::Int> = attrib_list.iter().map(|x| *x as egl::Int).collect();

        let image = unsafe {
            func(
                display.as_ptr(),
                ctx.as_ptr(),
                target,
                buffer.as_ptr(),
                int_attribs.as_ptr(),
            )
        };
        if image != egl::NO_IMAGE {
            return Ok(unsafe { egl::Image::from_ptr(image) });
        }
        return Err("eglCreateImageKHR returned EGL_NO_IMAGE".into());
    }

    Err("neither EGL 1.5 create_image nor eglCreateImageKHR available".into())
}

fn egl_destroy_image_with_fallback(
    egl: &Egl,
    display: Display,
    image: egl::Image,
) -> Result<(), String> {
    // EGL 1.5 native path
    if let Some(egl15) = egl.upcast::<egl::EGL1_5>() {
        return egl15
            .destroy_image(display, image)
            .map_err(|e| format!("eglDestroyImage (1.5) failed: {e}"));
    }

    // Fallback: eglDestroyImageKHR
    if let Some(ext) = egl.get_proc_address("eglDestroyImageKHR") {
        let func: unsafe extern "system" fn(
            display: egl::EGLDisplay,
            image: egl::EGLImage,
        ) -> egl::Boolean = unsafe { std::mem::transmute(ext) };

        let res = unsafe { func(display.as_ptr(), image.as_ptr()) };
        if res == egl::TRUE {
            return Ok(());
        }
        return Err("eglDestroyImageKHR failed".into());
    }

    Err("neither EGL 1.5 destroy_image nor eglDestroyImageKHR available".into())
}
