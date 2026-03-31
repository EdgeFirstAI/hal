// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::{EglDisplayInfo, EglDisplayKind, TransferBackend};
use crate::Error;
use gbm::{
    drm::{control::Device as DrmControlDevice, Device as DrmDevice},
    AsRaw, Device,
};
use khronos_egl::{self as egl, Attrib, Display, Dynamic, Instance, EGL1_4};
use log::debug;
use std::{
    ffi::c_void,
    mem::ManuallyDrop,
    rc::Rc,
    sync::{Mutex, OnceLock},
};

/// EGL library handle. Intentionally leaked (never dlclose'd) to avoid SIGBUS
/// on process exit: GPU drivers may keep internal state that outlives explicit
/// EGL cleanup, and dlclose can unmap memory still referenced by the driver.
static EGL_LIB: OnceLock<&'static libloading::Library> = OnceLock::new();

/// Global mutex that serializes **all** OpenGL and EGL operations across
/// every `GLProcessorST` instance in the process.
///
/// Some GPU drivers (notably Vivante `galcore` on i.MX8M Plus) are not
/// thread-safe for concurrent EGL/GL calls, even when each thread targets
/// an independent EGL display and context. Concurrent access can corrupt
/// driver-internal state, leading to deadlocks or SIGSEGV inside kernel
/// ioctls. Broadcom V3D exhibits a milder variant where concurrent
/// `eglTerminate` breaks display ref-counting.
///
/// This mutex is acquired by `GLProcessorThreaded` for **every** operation
/// on its dedicated GL thread:
/// - Initialization (`GLProcessorST::new` — EGL init, shader compilation,
///   DMA-BUF verification)
/// - Every message dispatch (convert, draw, PBO create/download, etc.)
/// - Teardown (`GLProcessorST::drop` → `GlContext::drop`)
///
/// This ensures that only one GL thread interacts with the GPU driver at
/// any given time. Multiple `ImageProcessor` instances remain usable from
/// different application threads — their GL commands are simply serialized
/// through this mutex rather than executed in parallel.
pub(super) static GL_MUTEX: Mutex<()> = Mutex::new(());

pub(super) fn get_egl_lib() -> Result<&'static libloading::Library, crate::Error> {
    if let Some(egl) = EGL_LIB.get() {
        Ok(egl)
    } else {
        let egl = unsafe { libloading::Library::new("libEGL.so.1")? };
        // Leak the library to prevent dlclose on process exit
        let egl: &'static libloading::Library = Box::leak(Box::new(egl));
        Ok(EGL_LIB.get_or_init(|| egl))
    }
}

pub(super) type Egl = Instance<Dynamic<&'static libloading::Library, EGL1_4>>;

/// Check whether an EGL display supports the surfaceless + no-config context
/// extensions required by the HAL's FBO-based rendering pipeline.
///
/// Queries `eglQueryString(display, EGL_EXTENSIONS)` and checks for
/// `EGL_KHR_surfaceless_context` and `EGL_KHR_no_config_context`.
fn probe_display_extensions(egl: &Egl, display: egl::Display) -> bool {
    let Ok(ext_str) = egl.query_string(Some(display), egl::EXTENSIONS) else {
        return false;
    };
    let exts = ext_str.to_string_lossy();

    let required = ["EGL_KHR_surfaceless_context", "EGL_KHR_no_config_context"];

    for r in &required {
        if !exts.contains(r) {
            log::debug!("Display missing required extension: {r}");
            return false;
        }
    }

    egl.bind_api(egl::OPENGL_ES_API).is_ok()
}

/// Probe for available EGL displays supporting headless OpenGL ES 3.0.
///
/// Returns validated displays in priority order (PlatformDevice, GBM,
/// Default). Each display is validated with `eglInitialize` + extension
/// checks for `EGL_KHR_surfaceless_context` and `EGL_KHR_no_config_context`.
/// Probed state is cleaned up with `eglTerminate` — no EGL resources are
/// left alive.
///
/// An empty list means OpenGL is not available on this system.
///
/// # Errors
///
/// Returns an error only if `libEGL.so.1` cannot be loaded. Individual
/// display probe failures are silently skipped.
pub fn probe_egl_displays() -> Result<Vec<EglDisplayInfo>, Error> {
    let egl: Egl = unsafe { Instance::<Dynamic<_, EGL1_4>>::load_required_from(get_egl_lib()?)? };

    let mut results = Vec::new();

    // PlatformDevice first (zero external deps, works on NVIDIA + newer Vivante)
    if let Ok(display_type) = GlContext::egl_get_platform_display_from_device(&egl) {
        let display = display_type.as_display();
        if egl.initialize(display).is_ok() {
            if probe_display_extensions(&egl, display) {
                results.push(EglDisplayInfo {
                    kind: EglDisplayKind::PlatformDevice,
                    description: "EGL platform device via EGL_EXT_device_enumeration".to_string(),
                });
            }
            let _ = egl.terminate(display);
        }
    }

    // GBM second (needed for Mali + old Vivante)
    if let Ok(display_type) = GlContext::egl_get_gbm_display(&egl) {
        let display = display_type.as_display();
        if egl.initialize(display).is_ok() {
            if probe_display_extensions(&egl, display) {
                results.push(EglDisplayInfo {
                    kind: EglDisplayKind::Gbm,
                    description: "GBM via /dev/dri/renderD128".to_string(),
                });
            }
            let _ = egl.terminate(display);
        }
    }

    // Default last (needs compositor)
    if let Ok(display_type) = GlContext::egl_get_default_display(&egl) {
        let display = display_type.as_display();
        if egl.initialize(display).is_ok() {
            if probe_display_extensions(&egl, display) {
                results.push(EglDisplayInfo {
                    kind: EglDisplayKind::Default,
                    description: "EGL default display".to_string(),
                });
            }
            let _ = egl.terminate(display);
        }
    }

    Ok(results)
}

pub(super) struct GlContext {
    pub(super) transfer_backend: TransferBackend,
    pub(super) display: EglDisplayType,
    pub(super) ctx: egl::Context,
    /// Whether the context is GLES 3.1+ (compute shaders available).
    pub(super) has_compute: bool,
    /// Wrapped in ManuallyDrop because the khronos-egl Dynamic instance's
    /// Drop calls eglReleaseThread() which can panic during process shutdown
    /// if the EGL library has been partially unloaded. We drop it explicitly
    /// inside catch_unwind in GlContext::drop.
    pub(super) egl: ManuallyDrop<Rc<Egl>>,
}

pub(super) enum EglDisplayType {
    Default(egl::Display),
    Gbm(egl::Display, #[allow(dead_code)] Device<Card>),
    PlatformDisplay(egl::Display),
}

impl EglDisplayType {
    pub(super) fn as_display(&self) -> egl::Display {
        match self {
            EglDisplayType::Default(disp) => *disp,
            EglDisplayType::Gbm(disp, _) => *disp,
            EglDisplayType::PlatformDisplay(disp) => *disp,
        }
    }
}

impl GlContext {
    pub(super) fn new(kind: Option<EglDisplayKind>) -> Result<GlContext, crate::Error> {
        // Create an EGL API instance.
        let egl: Rc<Egl> =
            Rc::new(unsafe { Instance::<Dynamic<_, EGL1_4>>::load_required_from(get_egl_lib()?)? });

        if let Some(kind) = kind {
            // Specific display type requested — try only that one.
            let display_fn = match kind {
                EglDisplayKind::Gbm => Self::egl_get_gbm_display as fn(&Egl) -> _,
                EglDisplayKind::PlatformDevice => Self::egl_get_platform_display_from_device,
                EglDisplayKind::Default => Self::egl_get_default_display,
            };
            return Self::try_initialize_egl(egl, display_fn).map_err(|e| {
                log::debug!("Failed to initialize EGL with {kind} display: {e:?}");
                e
            });
        }

        // Try PlatformDevice first (zero external deps, works on NVIDIA + newer Vivante)
        if let Ok(headless) =
            Self::try_initialize_egl(egl.clone(), Self::egl_get_platform_display_from_device)
        {
            return Ok(headless);
        } else {
            log::debug!("Didn't initialize EGL with platform display from device enumeration");
        }

        // GBM second (needed for Mali + old Vivante that lack EGL_EXT_platform_device)
        if let Ok(headless) = Self::try_initialize_egl(egl.clone(), Self::egl_get_gbm_display) {
            return Ok(headless);
        } else {
            log::debug!("Didn't initialize EGL with GBM Display");
        }

        // Default display last (needs compositor)
        if let Ok(headless) = Self::try_initialize_egl(egl.clone(), Self::egl_get_default_display) {
            return Ok(headless);
        } else {
            log::debug!("Didn't initialize EGL with Default Display");
        }

        Err(Error::OpenGl(
            "Could not initialize EGL with any known method".to_string(),
        ))
    }

    fn try_initialize_egl(
        egl: Rc<Egl>,
        display_fn: impl Fn(&Egl) -> Result<EglDisplayType, crate::Error>,
    ) -> Result<GlContext, crate::Error> {
        let display = display_fn(&egl)?;
        log::debug!("egl initialize with display: {:x?}", display.as_display());
        egl.initialize(display.as_display())?;

        // Verify required extensions for surfaceless + no-config context
        let ext_str = egl.query_string(Some(display.as_display()), egl::EXTENSIONS)?;
        let exts = ext_str.to_string_lossy();

        if !exts.contains("EGL_KHR_surfaceless_context") {
            return Err(crate::Error::GLVersion(
                "EGL display does not support EGL_KHR_surfaceless_context".to_string(),
            ));
        }

        if !exts.contains("EGL_KHR_no_config_context") {
            return Err(crate::Error::GLVersion(
                "EGL display does not support EGL_KHR_no_config_context".to_string(),
            ));
        }

        egl.bind_api(egl::OPENGL_ES_API)?;

        // No-config context: pass EGL_NO_CONFIG_KHR (null) instead of a
        // real config. The context is not bound to any specific framebuffer
        // format — it works with any FBO attachment format.
        // Try GLES 3.1 first (compute shaders), fall back to 3.0.
        let ctx_31 = [
            egl::CONTEXT_MAJOR_VERSION,
            3,
            egl::CONTEXT_MINOR_VERSION,
            1,
            egl::NONE,
        ];
        let (ctx, has_compute) =
            match egl.create_context(display.as_display(), egl_ext::NO_CONFIG_KHR, None, &ctx_31) {
                Ok(ctx) => {
                    debug!("Created GLES 3.1 context (compute shaders available)");
                    (ctx, true)
                }
                Err(_) => {
                    let ctx_30 = [egl::CONTEXT_MAJOR_VERSION, 3, egl::NONE];
                    let ctx = egl.create_context(
                        display.as_display(),
                        egl_ext::NO_CONFIG_KHR,
                        None,
                        &ctx_30,
                    )?;
                    debug!("Created GLES 3.0 context (no compute shaders)");
                    (ctx, false)
                }
            };
        debug!("ctx: {ctx:?}");

        // Surfaceless context: no PBuffer surface needed. All rendering
        // goes through FBOs backed by EGLImages.
        egl.make_current(display.as_display(), None, None, Some(ctx))?;

        let has_dma_extensions = Self::egl_check_support_dma(&egl, display.as_display()).is_ok();
        let transfer_backend = if has_dma_extensions {
            TransferBackend::DmaBuf
        } else {
            TransferBackend::Sync
        };
        Ok(GlContext {
            display,
            ctx,
            has_compute,
            egl: ManuallyDrop::new(egl),
            transfer_backend,
        })
    }

    fn egl_get_default_display(egl: &Egl) -> Result<EglDisplayType, crate::Error> {
        // get the default display
        if let Some(display) = unsafe { egl.get_display(egl::DEFAULT_DISPLAY) } {
            debug!("default display: {display:?}");
            return Ok(EglDisplayType::Default(display));
        }

        Err(Error::OpenGl(
            "Could not obtain EGL Default Display".to_string(),
        ))
    }

    fn egl_get_gbm_display(egl: &Egl) -> Result<EglDisplayType, crate::Error> {
        // init a GBM device
        let gbm = Device::new(Card::open_global()?)?;

        debug!("gbm: {gbm:?}");
        let display = Self::egl_get_platform_display_with_fallback(
            egl,
            egl_ext::PLATFORM_GBM_KHR,
            gbm.as_raw() as *mut c_void,
            &[egl::ATTRIB_NONE],
        )?;

        Ok(EglDisplayType::Gbm(display, gbm))
    }

    fn egl_get_platform_display_from_device(egl: &Egl) -> Result<EglDisplayType, crate::Error> {
        let extensions = egl.query_string(None, egl::EXTENSIONS)?;
        let extensions = extensions.to_string_lossy();
        log::debug!("EGL Extensions: {}", extensions);

        if !extensions.contains("EGL_EXT_device_enumeration") {
            return Err(Error::GLVersion(
                "EGL doesn't supported EGL_EXT_device_enumeration extension".to_string(),
            ));
        }

        type EGLDeviceEXT = *mut c_void;
        let devices = if let Some(ext) = egl.get_proc_address("eglQueryDevicesEXT") {
            let func: unsafe extern "system" fn(
                max_devices: egl::Int,
                devices: *mut EGLDeviceEXT,
                num_devices: *mut egl::Int,
            ) -> egl::Boolean = unsafe { std::mem::transmute(ext) };
            let mut devices = [std::ptr::null_mut(); 10];
            let mut num_devices = 0;
            let ok = unsafe { func(devices.len() as i32, devices.as_mut_ptr(), &mut num_devices) };
            if ok == egl::FALSE {
                return Err(Error::GLVersion("eglQueryDevicesEXT failed".to_string()));
            }
            for i in 0..num_devices {
                log::debug!("EGL device: {:?}", devices[i as usize]);
            }
            devices[0..num_devices as usize].to_vec()
        } else {
            return Err(Error::GLVersion(
                "EGL doesn't supported eglQueryDevicesEXT function".to_string(),
            ));
        };

        if !extensions.contains("EGL_EXT_platform_device") {
            return Err(Error::GLVersion(
                "EGL doesn't supported EGL_EXT_platform_device extension".to_string(),
            ));
        }

        if devices.is_empty() {
            return Err(Error::GLVersion(
                "EGL_EXT_device_enumeration returned 0 devices".to_string(),
            ));
        }
        let disp = Self::egl_get_platform_display_with_fallback(
            egl,
            egl_ext::PLATFORM_DEVICE_EXT,
            devices[0],
            &[egl::ATTRIB_NONE],
        )?;
        Ok(EglDisplayType::PlatformDisplay(disp))
    }

    fn egl_check_support_dma(egl: &Egl, display: egl::Display) -> Result<(), crate::Error> {
        // Query **display** extensions (not client extensions) because
        // EGL_EXT_image_dma_buf_import is a display extension on Mali and
        // other GPUs — it does not appear in the client extension string.
        let extensions = egl.query_string(Some(display), egl::EXTENSIONS)?;
        let extensions = extensions.to_string_lossy();
        log::debug!("EGL Display Extensions (DMA check): {}", extensions);

        if !extensions.contains("EGL_EXT_image_dma_buf_import") {
            return Err(crate::Error::GLVersion(
                "EGL does not support EGL_EXT_image_dma_buf_import extension".to_string(),
            ));
        }

        if egl.get_proc_address("eglCreateImageKHR").is_none() {
            return Err(crate::Error::GLVersion(
                "EGL does not support eglCreateImageKHR function".to_string(),
            ));
        }

        if egl.get_proc_address("eglDestroyImageKHR").is_none() {
            return Err(crate::Error::GLVersion(
                "EGL does not support eglDestroyImageKHR function".to_string(),
            ));
        }
        Ok(())
    }

    fn egl_get_platform_display_with_fallback(
        egl: &Egl,
        platform: egl::Enum,
        native_display: *mut c_void,
        attrib_list: &[Attrib],
    ) -> Result<Display, Error> {
        if let Some(egl) = egl.upcast::<egl::EGL1_5>() {
            unsafe { egl.get_platform_display(platform, native_display, attrib_list) }
                .map_err(|e| e.into())
        } else if let Some(ext) = egl.get_proc_address("eglGetPlatformDisplayEXT") {
            let func: unsafe extern "system" fn(
                platform: egl::Enum,
                native_display: *mut c_void,
                attrib_list: *const Attrib,
            ) -> egl::EGLDisplay = unsafe { std::mem::transmute(ext) };
            let disp = unsafe { func(platform, native_display, attrib_list.as_ptr()) };
            if disp != egl::NO_DISPLAY {
                Ok(unsafe { Display::from_ptr(disp) })
            } else {
                Err(egl.get_error().map(|e| e.into()).unwrap_or(Error::Internal(
                    "EGL failed but no error was reported".to_owned(),
                )))
            }
        } else {
            Err(Error::EGLLoad(egl::LoadError::InvalidVersion {
                provided: egl.version(),
                required: khronos_egl::Version::EGL1_5,
            }))
        }
    }

    pub(super) fn egl_create_image_with_fallback(
        egl: &Egl,
        display: Display,
        ctx: egl::Context,
        target: egl::Enum,
        buffer: egl::ClientBuffer,
        attrib_list: &[Attrib],
    ) -> Result<egl::Image, Error> {
        if let Some(egl) = egl.upcast::<egl::EGL1_5>() {
            egl.create_image(display, ctx, target, buffer, attrib_list)
                .map_err(|e| e.into())
        } else if let Some(ext) = egl.get_proc_address("eglCreateImageKHR") {
            log::trace!("eglCreateImageKHR addr: {:?}", ext);
            let func: unsafe extern "system" fn(
                display: egl::EGLDisplay,
                ctx: egl::EGLContext,
                target: egl::Enum,
                buffer: egl::EGLClientBuffer,
                attrib_list: *const egl::Int,
            ) -> egl::EGLImage = unsafe { std::mem::transmute(ext) };
            let new_attrib_list = attrib_list
                .iter()
                .map(|x| *x as egl::Int)
                .collect::<Vec<_>>();

            let image = unsafe {
                func(
                    display.as_ptr(),
                    ctx.as_ptr(),
                    target,
                    buffer.as_ptr(),
                    new_attrib_list.as_ptr(),
                )
            };
            if image != egl::NO_IMAGE {
                Ok(unsafe { egl::Image::from_ptr(image) })
            } else {
                Err(egl.get_error().map(|e| e.into()).unwrap_or(Error::Internal(
                    "EGL failed but no error was reported".to_owned(),
                )))
            }
        } else {
            Err(Error::EGLLoad(egl::LoadError::InvalidVersion {
                provided: egl.version(),
                required: khronos_egl::Version::EGL1_5,
            }))
        }
    }

    pub(super) fn egl_destroy_image_with_fallback(
        egl: &Egl,
        display: Display,
        image: egl::Image,
    ) -> Result<(), Error> {
        if let Some(egl) = egl.upcast::<egl::EGL1_5>() {
            egl.destroy_image(display, image).map_err(|e| e.into())
        } else if let Some(ext) = egl.get_proc_address("eglDestroyImageKHR") {
            let func: unsafe extern "system" fn(
                display: egl::EGLDisplay,
                image: egl::EGLImage,
            ) -> egl::Boolean = unsafe { std::mem::transmute(ext) };
            let res = unsafe { func(display.as_ptr(), image.as_ptr()) };
            if res == egl::TRUE {
                Ok(())
            } else {
                Err(egl.get_error().map(|e| e.into()).unwrap_or(Error::Internal(
                    "EGL failed but no error was reported".to_owned(),
                )))
            }
        } else {
            Err(Error::EGLLoad(egl::LoadError::InvalidVersion {
                provided: egl.version(),
                required: khronos_egl::Version::EGL1_5,
            }))
        }
    }
}

impl Drop for GlContext {
    fn drop(&mut self) {
        // During process shutdown (e.g. Python interpreter exit), the EGL/GL
        // shared libraries may already be partially unloaded, causing panics
        // or heap corruption when calling cleanup functions. We suppress
        // panic output and catch panics to prevent propagation.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = self
                .egl
                .make_current(self.display.as_display(), None, None, None);

            let _ = self
                .egl
                .destroy_context(self.display.as_display(), self.ctx);

            // eglTerminate is ref-counted per the EGL spec: each eglInitialize
            // increments a counter and each eglTerminate decrements it. The
            // display is only truly torn down when the last reference is
            // released. catch_unwind absorbs any driver-side misbehaviour.
            let _ = self.egl.terminate(self.display.as_display());
        }));
        std::panic::set_hook(prev_hook);

        // The Rc<Egl> (ManuallyDrop) is intentionally NOT dropped. The
        // khronos-egl Dynamic instance's Drop calls eglReleaseThread() which
        // panics if the EGL library has been unloaded (local/x86_64) or
        // causes heap corruption by calling into invalid memory (ARM).
    }
}

#[derive(Debug)]
/// A simple wrapper for a device node.
pub(super) struct Card(std::fs::File);

/// Implementing `AsFd` is a prerequisite to implementing the traits found
/// in this crate. Here, we are just calling `as_fd()` on the inner File.
impl std::os::unix::io::AsFd for Card {
    fn as_fd(&self) -> std::os::unix::io::BorrowedFd<'_> {
        self.0.as_fd()
    }
}

/// With `AsFd` implemented, we can now implement `drm::Device`.
impl DrmDevice for Card {}
impl DrmControlDevice for Card {}

/// Simple helper methods for opening a `Card`.
impl Card {
    pub fn open(path: &str) -> Result<Self, crate::Error> {
        let mut options = std::fs::OpenOptions::new();
        options.read(true);
        options.write(true);
        let c = options.open(path);
        match c {
            Ok(c) => Ok(Card(c)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(Error::NotFound(format!("File not found: {path}")))
            }
            Err(e) => Err(e.into()),
        }
    }

    pub fn open_global() -> Result<Self, crate::Error> {
        let targets = ["/dev/dri/renderD128", "/dev/dri/card0", "/dev/dri/card1"];
        let e = Self::open(targets[0]);
        if let Ok(t) = e {
            return Ok(t);
        }
        for t in &targets[1..] {
            if let Ok(t) = Self::open(t) {
                return Ok(t);
            }
        }
        e
    }
}

pub(super) mod egl_ext {
    #![allow(dead_code)]
    pub(crate) const LINUX_DMA_BUF: u32 = 0x3270;
    pub(crate) const LINUX_DRM_FOURCC: u32 = 0x3271;
    pub(crate) const DMA_BUF_PLANE0_FD: u32 = 0x3272;
    pub(crate) const DMA_BUF_PLANE0_OFFSET: u32 = 0x3273;
    pub(crate) const DMA_BUF_PLANE0_PITCH: u32 = 0x3274;
    pub(crate) const DMA_BUF_PLANE1_FD: u32 = 0x3275;
    pub(crate) const DMA_BUF_PLANE1_OFFSET: u32 = 0x3276;
    pub(crate) const DMA_BUF_PLANE1_PITCH: u32 = 0x3277;
    pub(crate) const DMA_BUF_PLANE2_FD: u32 = 0x3278;
    pub(crate) const DMA_BUF_PLANE2_OFFSET: u32 = 0x3279;
    pub(crate) const DMA_BUF_PLANE2_PITCH: u32 = 0x327A;
    pub(crate) const YUV_COLOR_SPACE_HINT: u32 = 0x327B;
    pub(crate) const SAMPLE_RANGE_HINT: u32 = 0x327C;
    pub(crate) const YUV_CHROMA_HORIZONTAL_SITING_HINT: u32 = 0x327D;
    pub(crate) const YUV_CHROMA_VERTICAL_SITING_HINT: u32 = 0x327E;

    pub(crate) const ITU_REC601: u32 = 0x327F;
    pub(crate) const ITU_REC709: u32 = 0x3280;
    pub(crate) const ITU_REC2020: u32 = 0x3281;

    pub(crate) const YUV_FULL_RANGE: u32 = 0x3282;
    pub(crate) const YUV_NARROW_RANGE: u32 = 0x3283;

    pub(crate) const YUV_CHROMA_SITING_0: u32 = 0x3284;
    pub(crate) const YUV_CHROMA_SITING_0_5: u32 = 0x3285;

    pub(crate) const PLATFORM_GBM_KHR: u32 = 0x31D7;

    pub(crate) const PLATFORM_DEVICE_EXT: u32 = 0x313F;

    /// EGL_KHR_no_config_context: null config for eglCreateContext.
    /// Defined as ((EGLConfig)0) in the EGL spec.
    ///
    /// # Safety
    /// The EGL spec defines EGL_NO_CONFIG_KHR as a null pointer. This is
    /// a safe transmute since `Config` is a newtype wrapper around `*mut c_void`.
    pub(crate) const NO_CONFIG_KHR: khronos_egl::Config =
        unsafe { std::mem::transmute(std::ptr::null_mut::<std::ffi::c_void>()) };
}
