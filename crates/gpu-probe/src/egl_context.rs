// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![allow(dead_code)]
//! EGL context bootstrap for headless GPU probing.
//!
//! Obtains an EGL display (via PlatformDevice or GBM fallback), creates a
//! surfaceless GLES 3.0 context using `EGL_KHR_no_config_context`, and makes
//! it current with `EGL_NO_SURFACE`. This is a standalone version of the
//! HAL's `opengl_headless::GlContext` stripped down for diagnostic /
//! benchmarking use only.

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
const PLATFORM_DEVICE_EXT: u32 = 0x313F;

/// EGL_KHR_no_config_context: null config for eglCreateContext.
const NO_CONFIG_KHR: egl::Config =
    unsafe { std::mem::transmute(std::ptr::null_mut::<std::ffi::c_void>()) };

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
// Second plane (interleaved CbCr for NV12/NV16/NV24 native YUV import).
const DMA_BUF_PLANE1_FD: u32 = 0x3275;
const DMA_BUF_PLANE1_OFFSET: u32 = 0x3276;
const DMA_BUF_PLANE1_PITCH: u32 = 0x3277;
// YUV import hints (EGL_EXT_image_dma_buf_import) — mirror the HAL's
// BT.601 full-range (JFIF) stop-gap so the probe imports natively-sampled
// YUV the same way `DmaImportAttrs::to_egl_attribs` does.
const YUV_COLOR_SPACE_HINT: u32 = 0x327B;
const SAMPLE_RANGE_HINT: u32 = 0x327C;
const ITU_REC601: u32 = 0x327F;
const YUV_FULL_RANGE: u32 = 0x3282;
// EGL_IMAGE_PRESERVED_KHR — preserve pixel data across the import.
const IMAGE_PRESERVED: u32 = 0x30D2;

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

/// Backing storage for the EGL display — keeps the underlying resource alive.
enum DisplayBacking {
    /// PlatformDevice display — no external resources needed.
    Device,
    /// GBM display — the GBM device must outlive the EGL display.
    Gbm(Device<Card>),
}

/// Holds an initialised EGL display, surfaceless GLES 3.0 context, and the
/// backing display resource. Provides helper methods for querying GL/EGL
/// strings and creating EGLImages from DMA-buf file descriptors.
pub struct GpuContext {
    egl: Egl,
    display: Display,
    ctx: egl::Context,
    _backing: DisplayBacking,
}

impl GpuContext {
    /// Bootstrap a headless GLES 3.0 surfaceless context.
    ///
    /// Tries PlatformDevice first (zero external deps), falls back to GBM.
    pub fn new() -> Result<Self, String> {
        let egl: Egl = unsafe {
            Instance::<Dynamic<_, EGL1_4>>::load_required_from(get_egl_lib()?)
                .map_err(|e| format!("EGL instance load failed: {e}"))?
        };

        // Try PlatformDevice first
        match Self::get_platform_device_display(&egl) {
            Ok((display, backing)) => {
                return Self::init_surfaceless(egl, display, backing);
            }
            Err(e) => debug!("PlatformDevice not available: {e}"),
        }

        // Fall back to GBM
        let (display, backing) = Self::get_gbm_display(&egl)?;
        Self::init_surfaceless(egl, display, backing)
    }

    fn get_platform_device_display(egl: &Egl) -> Result<(Display, DisplayBacking), String> {
        let client_exts = egl
            .query_string(None, egl::EXTENSIONS)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();

        if !client_exts.contains("EGL_EXT_device_enumeration") {
            return Err("EGL_EXT_device_enumeration not available".into());
        }
        if !client_exts.contains("EGL_EXT_platform_device") {
            return Err("EGL_EXT_platform_device not available".into());
        }

        let query_devices_fn = egl
            .get_proc_address("eglQueryDevicesEXT")
            .ok_or("eglQueryDevicesEXT not available")?;
        let query_devices: unsafe extern "system" fn(
            max_devices: egl::Int,
            devices: *mut *mut c_void,
            num_devices: *mut egl::Int,
        ) -> egl::Boolean = unsafe { std::mem::transmute(query_devices_fn) };

        let mut devices = [null_mut(); 10];
        let mut num_devices: egl::Int = 0;
        unsafe { query_devices(10, devices.as_mut_ptr(), &mut num_devices) };
        if num_devices == 0 {
            return Err("eglQueryDevicesEXT returned 0 devices".into());
        }
        debug!("EGL devices found: {num_devices}");

        let display = egl_get_platform_display_with_fallback(
            egl,
            PLATFORM_DEVICE_EXT,
            devices[0],
            &[egl::ATTRIB_NONE],
        )?;

        Ok((display, DisplayBacking::Device))
    }

    fn get_gbm_display(egl: &Egl) -> Result<(Display, DisplayBacking), String> {
        let card = Card::open_any()?;
        let gbm = Device::new(card).map_err(|e| format!("GBM device creation failed: {e}"))?;
        debug!("GBM device: {gbm:?}");

        let display = egl_get_platform_display_with_fallback(
            egl,
            PLATFORM_GBM_KHR,
            gbm.as_raw() as *mut c_void,
            &[egl::ATTRIB_NONE],
        )?;

        Ok((display, DisplayBacking::Gbm(gbm)))
    }

    fn init_surfaceless(
        egl: Egl,
        display: Display,
        backing: DisplayBacking,
    ) -> Result<Self, String> {
        debug!("EGL display: {display:?}");

        egl.initialize(display)
            .map_err(|e| format!("eglInitialize failed: {e}"))?;

        // Verify required extensions
        let ext_str = egl
            .query_string(Some(display), egl::EXTENSIONS)
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();

        if !ext_str.contains("EGL_KHR_surfaceless_context") {
            return Err("EGL_KHR_surfaceless_context not supported".into());
        }
        if !ext_str.contains("EGL_KHR_no_config_context") {
            return Err("EGL_KHR_no_config_context not supported".into());
        }

        egl.bind_api(egl::OPENGL_ES_API)
            .map_err(|e| format!("eglBindAPI failed: {e}"))?;

        let context_attributes = [egl::CONTEXT_MAJOR_VERSION, 3, egl::NONE];
        let ctx = egl
            .create_context(display, NO_CONFIG_KHR, None, &context_attributes)
            .map_err(|e| format!("eglCreateContext failed: {e}"))?;

        egl.make_current(display, None, None, Some(ctx))
            .map_err(|e| format!("eglMakeCurrent (surfaceless) failed: {e}"))?;

        // Load GL function pointers
        gls::load_with(|s| {
            egl.get_proc_address(s)
                .map_or(std::ptr::null(), |p| p as *const _)
        });

        Ok(GpuContext {
            egl,
            display,
            ctx,
            _backing: backing,
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

    /// Create an EGLImage from a (possibly multi-plane) DMA-buf.
    ///
    /// This is the richer counterpart to [`create_egl_image_dma`]: it carries a
    /// per-plane `offset`, an optional second plane (interleaved CbCr), and the
    /// BT.601 full-range YUV import hints. It mirrors exactly what the HAL's
    /// `DmaImportAttrs::to_egl_attribs` emits, so a successful import here proves
    /// the real HAL allocation is GPU-importable.
    ///
    /// * `plane1` — second plane for NV12/NV16/NV24 native semi-planar import.
    /// * `yuv_hints` — add BT.601 full-range colour-space hints (required for
    ///   native YUV fourccs sampled via `samplerExternalOES`).
    pub fn create_egl_image_planar(
        &self,
        width: i32,
        height: i32,
        drm_fourcc: u32,
        plane0: EglPlane,
        plane1: Option<EglPlane>,
        yuv_hints: bool,
    ) -> Result<egl::Image, String> {
        let mut attribs: Vec<Attrib> = vec![
            egl::WIDTH as Attrib,
            width as Attrib,
            egl::HEIGHT as Attrib,
            height as Attrib,
            LINUX_DRM_FOURCC as Attrib,
            drm_fourcc as Attrib,
            DMA_BUF_PLANE0_FD as Attrib,
            plane0.fd as Attrib,
            DMA_BUF_PLANE0_OFFSET as Attrib,
            plane0.offset as Attrib,
            DMA_BUF_PLANE0_PITCH as Attrib,
            plane0.pitch as Attrib,
            IMAGE_PRESERVED as Attrib,
            egl::TRUE as Attrib,
        ];

        if let Some(p1) = plane1 {
            attribs.extend_from_slice(&[
                DMA_BUF_PLANE1_FD as Attrib,
                p1.fd as Attrib,
                DMA_BUF_PLANE1_OFFSET as Attrib,
                p1.offset as Attrib,
                DMA_BUF_PLANE1_PITCH as Attrib,
                p1.pitch as Attrib,
            ]);
        }

        if yuv_hints {
            attribs.extend_from_slice(&[
                YUV_COLOR_SPACE_HINT as Attrib,
                ITU_REC601 as Attrib,
                SAMPLE_RANGE_HINT as Attrib,
                YUV_FULL_RANGE as Attrib,
            ]);
        }

        attribs.push(egl::ATTRIB_NONE);

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

/// One DMA-buf plane's import parameters (fd / byte offset / row pitch).
#[derive(Clone, Copy, Debug)]
pub struct EglPlane {
    pub fd: i32,
    pub offset: i32,
    pub pitch: i32,
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        // Defence-in-depth cleanup: destroy context and terminate the EGL
        // display inside catch_unwind to absorb driver panics.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = self.egl.make_current(self.display, None, None, None);
            let _ = self.egl.destroy_context(self.display, self.ctx);
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
