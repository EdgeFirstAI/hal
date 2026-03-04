// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "linux")]
#![cfg(feature = "opengl")]

use edgefirst_decoder::{DetectBox, ProtoData, ProtoTensor, Segmentation};
use edgefirst_tensor::{TensorMemory, TensorTrait};
use four_char_code::FourCharCode;
use gbm::{
    drm::{buffer::DrmFourcc, control::Device as DrmControlDevice, Device as DrmDevice},
    AsRaw, Device,
};
use khronos_egl::{self as egl, Attrib, Display, Dynamic, Instance, EGL1_4};
use log::{debug, error};
use std::{
    collections::BTreeSet,
    ffi::{c_char, c_void, CStr, CString},
    mem::ManuallyDrop,
    os::fd::AsRawFd,
    ptr::{null, null_mut, NonNull},
    rc::Rc,
    str::FromStr,
    sync::OnceLock,
    thread::JoinHandle,
    time::Instant,
};
use tokio::sync::mpsc::{Sender, WeakSender};

macro_rules! function {
    () => {{
        fn f() {}
        fn type_name_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        let name = type_name_of(f);

        // Find and cut the rest of the path
        match &name[..name.len() - 3].rfind(':') {
            Some(pos) => &name[pos + 1..name.len() - 3],
            None => &name[..name.len() - 3],
        }
    }};
}

use crate::{
    fourcc_is_int8, fourcc_is_packed_rgb, CPUProcessor, Crop, Error, Flip, ImageProcessorTrait,
    MaskRegion, Rect, Rotation, TensorImage, TensorImageRef, BGRA, DEFAULT_COLORS, GREY, NV12,
    PLANAR_RGB, PLANAR_RGBA, PLANAR_RGB_INT8, RGB, RGBA, RGB_INT8, VYUY, YUYV,
};

/// Identifies the type of EGL display used for headless OpenGL ES rendering.
///
/// The HAL creates a surfaceless GLES 3.0 context
/// (`EGL_KHR_surfaceless_context` + `EGL_KHR_no_config_context`) and
/// renders exclusively through FBOs backed by EGLImages imported from
/// DMA-buf file descriptors. No window or PBuffer surface is created.
///
/// Displays are probed in priority order: PlatformDevice first (zero
/// external dependencies), then GBM, then Default. Use
/// [`probe_egl_displays`] to discover which are available and
/// [`ImageProcessorConfig::egl_display`](crate::ImageProcessorConfig::egl_display)
/// to override the auto-detection.
///
/// # Display Types
///
/// - **`PlatformDevice`** — Uses `EGL_EXT_device_enumeration` to query
///   available EGL devices via `eglQueryDevicesEXT`, then selects the first
///   device with `eglGetPlatformDisplay(EGL_EXT_platform_device, ...)`.
///   Headless and compositor-free with zero external library dependencies.
///   Works on NVIDIA GPUs and newer Vivante drivers.
///
/// - **`Gbm`** — Opens a DRM render node (e.g. `/dev/dri/renderD128`) and
///   creates a GBM (Generic Buffer Manager) device, then calls
///   `eglGetPlatformDisplay(EGL_PLATFORM_GBM_KHR, gbm_device)`. Requires
///   `libgbm` and a DRM render node. Needed on ARM Mali (i.MX95) and older
///   Vivante drivers that do not expose `EGL_EXT_platform_device`.
///
/// - **`Default`** — Calls `eglGetDisplay(EGL_DEFAULT_DISPLAY)`, letting the
///   EGL implementation choose the display. On Wayland systems this connects
///   to the compositor; on X11 it connects to the X server. May block on
///   headless systems where a compositor is expected but not running.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EglDisplayKind {
    Gbm,
    PlatformDevice,
    Default,
}

impl std::fmt::Display for EglDisplayKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EglDisplayKind::Gbm => write!(f, "GBM"),
            EglDisplayKind::PlatformDevice => write!(f, "PlatformDevice"),
            EglDisplayKind::Default => write!(f, "Default"),
        }
    }
}

/// A validated, available EGL display discovered by [`probe_egl_displays`].
#[derive(Debug, Clone)]
pub struct EglDisplayInfo {
    /// The type of EGL display.
    pub kind: EglDisplayKind,
    /// Human-readable description for logging/diagnostics
    /// (e.g. "GBM via /dev/dri/renderD128").
    pub description: String,
}

/// EGL library handle. Intentionally leaked (never dlclose'd) to avoid SIGBUS
/// on process exit: GPU drivers may keep internal state that outlives explicit
/// EGL cleanup, and dlclose can unmap memory still referenced by the driver.
static EGL_LIB: OnceLock<&'static libloading::Library> = OnceLock::new();

fn get_egl_lib() -> Result<&'static libloading::Library, crate::Error> {
    if let Some(egl) = EGL_LIB.get() {
        Ok(egl)
    } else {
        let egl = unsafe { libloading::Library::new("libEGL.so.1")? };
        // Leak the library to prevent dlclose on process exit
        let egl: &'static libloading::Library = Box::leak(Box::new(egl));
        Ok(EGL_LIB.get_or_init(|| egl))
    }
}

type Egl = Instance<Dynamic<&'static libloading::Library, EGL1_4>>;

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

/// Tracks which data-transfer method is active for moving pixels
/// between CPU memory and GPU textures/framebuffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransferBackend {
    /// Zero-copy via EGLImage imported from DMA-buf file descriptors.
    /// Available on i.MX8 (Vivante), i.MX95 (Mali), Jetson, and any
    /// platform where `EGL_EXT_image_dma_buf_import` is present AND
    /// the GPU can actually render through DMA-buf-backed textures.
    DmaBuf,

    /// GPU buffer via Pixel Buffer Object. Used when DMA-buf is unavailable
    /// but OpenGL is present. Data stays in GPU-accessible memory.
    Pbo,

    /// Synchronous `glTexSubImage2D` upload + `glReadnPixels` readback.
    /// Used when DMA-buf is unavailable or when the DMA-buf verification
    /// probe fails (e.g. NVIDIA discrete GPUs where EGLImage creation
    /// succeeds but rendered data is all zeros).
    Sync,
}

impl TransferBackend {
    /// Returns `true` if DMA-buf zero-copy is available.
    pub(crate) fn is_dma(self) -> bool {
        self == TransferBackend::DmaBuf
    }

    /// Returns `true` if PBO transfer is active.
    #[allow(dead_code)]
    pub(crate) fn is_pbo(self) -> bool {
        self == TransferBackend::Pbo
    }
}

pub(crate) struct GlContext {
    pub(crate) transfer_backend: TransferBackend,
    pub(crate) display: EglDisplayType,
    pub(crate) ctx: egl::Context,
    /// Wrapped in ManuallyDrop because the khronos-egl Dynamic instance's
    /// Drop calls eglReleaseThread() which can panic during process shutdown
    /// if the EGL library has been partially unloaded. We drop it explicitly
    /// inside catch_unwind in GlContext::drop.
    pub(crate) egl: ManuallyDrop<Rc<Egl>>,
}

pub(crate) enum EglDisplayType {
    Default(egl::Display),
    Gbm(egl::Display, #[allow(dead_code)] Device<Card>),
    PlatformDisplay(egl::Display),
}

impl EglDisplayType {
    fn as_display(&self) -> egl::Display {
        match self {
            EglDisplayType::Default(disp) => *disp,
            EglDisplayType::Gbm(disp, _) => *disp,
            EglDisplayType::PlatformDisplay(disp) => *disp,
        }
    }
}

impl GlContext {
    pub(crate) fn new(kind: Option<EglDisplayKind>) -> Result<GlContext, crate::Error> {
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
        let context_attributes = [egl::CONTEXT_MAJOR_VERSION, 3, egl::NONE, egl::NONE];
        let ctx = egl.create_context(
            display.as_display(),
            egl_ext::NO_CONFIG_KHR,
            None,
            &context_attributes,
        )?;
        debug!("ctx: {ctx:?}");

        // Surfaceless context: no PBuffer surface needed. All rendering
        // goes through FBOs backed by EGLImages.
        egl.make_current(display.as_display(), None, None, Some(ctx))?;

        let has_dma_extensions = Self::egl_check_support_dma(&egl).is_ok();
        let transfer_backend = if has_dma_extensions {
            TransferBackend::DmaBuf
        } else {
            TransferBackend::Sync
        };
        Ok(GlContext {
            display,
            ctx,
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
            ) -> *const c_char = unsafe { std::mem::transmute(ext) };
            let mut devices = [std::ptr::null_mut(); 10];
            let mut num_devices = 0;
            unsafe { func(devices.len() as i32, devices.as_mut_ptr(), &mut num_devices) };
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

    fn egl_check_support_dma(egl: &Egl) -> Result<(), crate::Error> {
        let extensions = egl.query_string(None, egl::EXTENSIONS)?;
        let extensions = extensions.to_string_lossy();
        log::debug!("EGL Extensions: {}", extensions);

        if egl.upcast::<egl::EGL1_5>().is_some() {
            return Ok(());
        }

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

    fn egl_create_image_with_fallback(
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

    fn egl_destroy_image_with_fallback(
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
pub(crate) struct Card(std::fs::File);

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

#[derive(Debug, Clone, Copy)]
struct RegionOfInterest {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

#[allow(clippy::type_complexity)]
enum GLProcessorMessage {
    ImageConvert(
        SendablePtr<TensorImage>,
        SendablePtr<TensorImage>,
        Rotation,
        Flip,
        Crop,
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    SetColors(
        Vec<[u8; 4]>,
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    DrawMasks(
        SendablePtr<TensorImage>,
        SendablePtr<DetectBox>,
        SendablePtr<Segmentation>,
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    DrawMasksProto(
        SendablePtr<TensorImage>,
        SendablePtr<DetectBox>,
        Box<ProtoData>,
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    SetInt8Interpolation(
        Int8InterpolationMode,
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    DecodeMasksAtlas(
        SendablePtr<DetectBox>,
        Box<ProtoData>,
        usize, // output_width
        usize, // output_height
        tokio::sync::oneshot::Sender<Result<(Vec<u8>, Vec<MaskRegion>), Error>>,
    ),
    PboCreate(
        usize, // buffer size in bytes
        tokio::sync::oneshot::Sender<Result<u32, Error>>,
    ),
    PboMap(
        u32,   // buffer_id
        usize, // size
        tokio::sync::oneshot::Sender<Result<edgefirst_tensor::PboMapping, Error>>,
    ),
    PboUnmap(
        u32, // buffer_id
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
    PboDelete(u32), // fire-and-forget, no reply
}

/// Implements PboOps by sending commands to the GL thread.
///
/// Uses a `WeakSender` so that PBO images don't keep the GL thread's channel
/// alive. When the `GLProcessorThreaded` is dropped, its `Sender` is the last
/// strong reference — dropping it closes the channel and lets the GL thread
/// exit. PBO operations after that return `PboDisconnected`.
struct GlPboOps {
    sender: WeakSender<GLProcessorMessage>,
}

// SAFETY: GlPboOps sends all GL operations to the dedicated GL thread via a
// channel. `map_buffer` returns a CPU-visible pointer from `glMapBufferRange`
// that remains valid until `unmap_buffer` calls `glUnmapBuffer` on the GL thread.
// `delete_buffer` sends a fire-and-forget deletion command to the GL thread.
unsafe impl edgefirst_tensor::PboOps for GlPboOps {
    fn map_buffer(
        &self,
        buffer_id: u32,
        size: usize,
    ) -> edgefirst_tensor::Result<edgefirst_tensor::PboMapping> {
        let sender = self
            .sender
            .upgrade()
            .ok_or(edgefirst_tensor::Error::PboDisconnected)?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        sender
            .blocking_send(GLProcessorMessage::PboMap(buffer_id, size, tx))
            .map_err(|_| edgefirst_tensor::Error::PboDisconnected)?;
        rx.blocking_recv()
            .map_err(|_| edgefirst_tensor::Error::PboDisconnected)?
            .map_err(|e| {
                edgefirst_tensor::Error::NotImplemented(format!("GL PBO map failed: {e:?}"))
            })
    }

    fn unmap_buffer(&self, buffer_id: u32) -> edgefirst_tensor::Result<()> {
        let sender = self
            .sender
            .upgrade()
            .ok_or(edgefirst_tensor::Error::PboDisconnected)?;
        let (tx, rx) = tokio::sync::oneshot::channel();
        sender
            .blocking_send(GLProcessorMessage::PboUnmap(buffer_id, tx))
            .map_err(|_| edgefirst_tensor::Error::PboDisconnected)?;
        rx.blocking_recv()
            .map_err(|_| edgefirst_tensor::Error::PboDisconnected)?
            .map_err(|e| {
                edgefirst_tensor::Error::NotImplemented(format!("GL PBO unmap failed: {e:?}"))
            })
    }

    fn delete_buffer(&self, buffer_id: u32) {
        if let Some(sender) = self.sender.upgrade() {
            let _ = sender.blocking_send(GLProcessorMessage::PboDelete(buffer_id));
        }
    }
}

/// OpenGL multi-threaded image converter. The actual conversion is done in a
/// separate rendering thread, as OpenGL contexts are not thread-safe. This can
/// be safely sent between threads. The `convert()` call sends the conversion
/// request to the rendering thread and waits for the result.
#[derive(Debug)]
pub struct GLProcessorThreaded {
    // This is only None when the converter is being dropped.
    handle: Option<JoinHandle<()>>,

    // This is only None when the converter is being dropped.
    sender: Option<Sender<GLProcessorMessage>>,
    transfer_backend: TransferBackend,
}

unsafe impl Send for GLProcessorThreaded {}
unsafe impl Sync for GLProcessorThreaded {}

struct SendablePtr<T: Send> {
    ptr: NonNull<T>,
    len: usize,
}

unsafe impl<T> Send for SendablePtr<T> where T: Send {}

impl GLProcessorThreaded {
    /// Creates a new OpenGL multi-threaded image converter.
    pub fn new(kind: Option<EglDisplayKind>) -> Result<Self, Error> {
        let (send, mut recv) = tokio::sync::mpsc::channel::<GLProcessorMessage>(1);

        let (create_ctx_send, create_ctx_recv) = tokio::sync::oneshot::channel();

        let func = move || {
            let mut gl_converter = match GLProcessorST::new(kind) {
                Ok(gl) => gl,
                Err(e) => {
                    let _ = create_ctx_send.send(Err(e));
                    return;
                }
            };
            let _ = create_ctx_send.send(Ok(gl_converter.gl_context.transfer_backend));
            while let Some(msg) = recv.blocking_recv() {
                match msg {
                    GLProcessorMessage::ImageConvert(src, mut dst, rotation, flip, crop, resp) => {
                        // SAFETY: This is safe because the convert() function waits for the resp to
                        // be sent before dropping the borrow for src and dst
                        let src = unsafe { src.ptr.as_ref() };
                        let dst = unsafe { dst.ptr.as_mut() };
                        let res = gl_converter.convert(src, dst, rotation, flip, crop);
                        let _ = resp.send(res);
                    }
                    GLProcessorMessage::DrawMasks(mut dst, det, seg, resp) => {
                        // SAFETY: This is safe because the draw_masks() function waits for the
                        // resp to be sent before dropping the borrow for dst, detect, and
                        // segmentation
                        let dst = unsafe { dst.ptr.as_mut() };
                        let det = unsafe { std::slice::from_raw_parts(det.ptr.as_ptr(), det.len) };
                        let seg = unsafe { std::slice::from_raw_parts(seg.ptr.as_ptr(), seg.len) };
                        let res = gl_converter.draw_masks(dst, det, seg);
                        let _ = resp.send(res);
                    }
                    GLProcessorMessage::DrawMasksProto(mut dst, det, proto_data, resp) => {
                        // SAFETY: Same safety invariant as DrawMasks — caller
                        // blocks on resp before dropping borrows.
                        let dst = unsafe { dst.ptr.as_mut() };
                        let det = unsafe { std::slice::from_raw_parts(det.ptr.as_ptr(), det.len) };
                        let res = gl_converter.draw_masks_proto(dst, det, &proto_data);
                        let _ = resp.send(res);
                    }
                    GLProcessorMessage::SetColors(colors, resp) => {
                        let res = gl_converter.set_class_colors(&colors);
                        let _ = resp.send(res);
                    }
                    GLProcessorMessage::SetInt8Interpolation(mode, resp) => {
                        gl_converter.set_int8_interpolation_mode(mode);
                        let _ = resp.send(Ok(()));
                    }
                    GLProcessorMessage::DecodeMasksAtlas(
                        det,
                        proto_data,
                        output_width,
                        output_height,
                        resp,
                    ) => {
                        let det = unsafe { std::slice::from_raw_parts(det.ptr.as_ptr(), det.len) };
                        let res = gl_converter.decode_masks_atlas(
                            det,
                            &proto_data,
                            output_width,
                            output_height,
                        );
                        let _ = resp.send(res);
                    }
                    GLProcessorMessage::PboCreate(size, resp) => {
                        let result = unsafe {
                            let mut id: u32 = 0;
                            gls::gl::GenBuffers(1, &mut id);
                            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, id);
                            gls::gl::BufferData(
                                gls::gl::PIXEL_PACK_BUFFER,
                                size as isize,
                                std::ptr::null(),
                                gls::gl::STREAM_COPY,
                            );
                            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                            match check_gl_error("PboCreate", 0) {
                                Ok(()) => Ok(id),
                                Err(e) => {
                                    gls::gl::DeleteBuffers(1, &id);
                                    Err(e)
                                }
                            }
                        };
                        let _ = resp.send(result);
                    }
                    GLProcessorMessage::PboMap(buffer_id, size, resp) => {
                        let result = unsafe {
                            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, buffer_id);
                            let ptr = gls::gl::MapBufferRange(
                                gls::gl::PIXEL_PACK_BUFFER,
                                0,
                                size as isize,
                                gls::gl::MAP_READ_BIT | gls::gl::MAP_WRITE_BIT,
                            );
                            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                            if ptr.is_null() {
                                Err(crate::Error::OpenGl(
                                    "glMapBufferRange returned null".to_string(),
                                ))
                            } else {
                                Ok(edgefirst_tensor::PboMapping {
                                    ptr: ptr as *mut u8,
                                    size,
                                })
                            }
                        };
                        let _ = resp.send(result);
                    }
                    GLProcessorMessage::PboUnmap(buffer_id, resp) => {
                        let result = unsafe {
                            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, buffer_id);
                            let ok = gls::gl::UnmapBuffer(gls::gl::PIXEL_PACK_BUFFER);
                            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                            if ok == gls::gl::FALSE {
                                Err(Error::OpenGl(
                                    "PBO data was corrupted during mapping".into(),
                                ))
                            } else {
                                check_gl_error("PboUnmap", 0)
                            }
                        };
                        let _ = resp.send(result);
                    }
                    GLProcessorMessage::PboDelete(buffer_id) => unsafe {
                        gls::gl::DeleteBuffers(1, &buffer_id);
                    },
                }
            }
        };

        // let handle = tokio::task::spawn(func());
        let handle = std::thread::spawn(func);

        let transfer_backend = match create_ctx_recv.blocking_recv() {
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(Error::Internal(
                    "GL converter error messaging closed without update".to_string(),
                ));
            }
            Ok(Ok(tb)) => tb,
        };

        Ok(Self {
            handle: Some(handle),
            sender: Some(send),
            transfer_backend,
        })
    }
}

impl ImageProcessorTrait for GLProcessorThreaded {
    fn convert(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        crop.check_crop(src, dst)?;
        if !GLProcessorST::check_src_format_supported(self.transfer_backend, src) {
            return Err(crate::Error::NotSupported(format!(
                "Opengl doesn't support {} source texture",
                src.fourcc().display()
            )));
        }

        if !GLProcessorST::check_dst_format_supported(self.transfer_backend, dst) {
            return Err(crate::Error::NotSupported(format!(
                "Opengl doesn't support {} destination texture",
                dst.fourcc().display()
            )));
        }

        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .unwrap()
            .blocking_send(GLProcessorMessage::ImageConvert(
                SendablePtr {
                    ptr: src.into(),
                    len: 1,
                },
                SendablePtr {
                    ptr: dst.into(),
                    len: 1,
                },
                rotation,
                flip,
                crop,
                err_send,
            ))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    fn convert_ref(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImageRef<'_>,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        // OpenGL doesn't support PLANAR_RGB output, delegate to CPU
        let mut cpu = CPUProcessor::new();
        cpu.convert_ref(src, dst, rotation, flip, crop)
    }

    fn draw_masks(
        &mut self,
        dst: &mut TensorImage,
        detect: &[crate::DetectBox],
        segmentation: &[crate::Segmentation],
    ) -> crate::Result<()> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .unwrap()
            .blocking_send(GLProcessorMessage::DrawMasks(
                SendablePtr {
                    ptr: dst.into(),
                    len: 1,
                },
                SendablePtr {
                    ptr: NonNull::new(detect.as_ptr() as *mut DetectBox).unwrap(),
                    len: detect.len(),
                },
                SendablePtr {
                    ptr: NonNull::new(segmentation.as_ptr() as *mut Segmentation).unwrap(),
                    len: segmentation.len(),
                },
                err_send,
            ))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    fn draw_masks_proto(
        &mut self,
        dst: &mut TensorImage,
        detect: &[DetectBox],
        proto_data: &ProtoData,
    ) -> crate::Result<()> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .unwrap()
            .blocking_send(GLProcessorMessage::DrawMasksProto(
                SendablePtr {
                    ptr: NonNull::new(dst as *mut TensorImage).unwrap(),
                    len: 1,
                },
                SendablePtr {
                    ptr: NonNull::new(detect.as_ptr() as *mut DetectBox).unwrap(),
                    len: detect.len(),
                },
                Box::new(proto_data.clone()),
                err_send,
            ))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    fn decode_masks_atlas(
        &mut self,
        detect: &[DetectBox],
        proto_data: ProtoData,
        output_width: usize,
        output_height: usize,
    ) -> crate::Result<(Vec<u8>, Vec<MaskRegion>)> {
        GLProcessorThreaded::decode_masks_atlas(
            self,
            detect,
            proto_data,
            output_width,
            output_height,
        )
    }

    fn set_class_colors(&mut self, colors: &[[u8; 4]]) -> Result<(), crate::Error> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .unwrap()
            .blocking_send(GLProcessorMessage::SetColors(colors.to_vec(), err_send))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }
}

impl GLProcessorThreaded {
    /// Sets the interpolation mode for int8 proto textures.
    pub fn set_int8_interpolation_mode(
        &mut self,
        mode: Int8InterpolationMode,
    ) -> Result<(), crate::Error> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .unwrap()
            .blocking_send(GLProcessorMessage::SetInt8Interpolation(mode, err_send))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        err_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    /// Decode all detection masks into a compact atlas via the GL thread.
    ///
    /// Returns `(atlas_pixels, regions)` where `atlas_pixels` is a contiguous
    /// `Vec<u8>` of shape `[atlas_h, output_width]` (compact, bbox-sized strips)
    /// and `regions` describes each detection's location within the atlas.
    pub fn decode_masks_atlas(
        &mut self,
        detect: &[DetectBox],
        proto_data: ProtoData,
        output_width: usize,
        output_height: usize,
    ) -> Result<(Vec<u8>, Vec<MaskRegion>), crate::Error> {
        let (resp_send, resp_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .unwrap()
            .blocking_send(GLProcessorMessage::DecodeMasksAtlas(
                SendablePtr {
                    ptr: NonNull::new(detect.as_ptr() as *mut DetectBox).unwrap(),
                    len: detect.len(),
                },
                Box::new(proto_data),
                output_width,
                output_height,
                resp_send,
            ))
            .map_err(|_| Error::Internal("GL converter thread exited".to_string()))?;
        resp_recv.blocking_recv().map_err(|_| {
            Error::Internal("GL converter error messaging closed without update".to_string())
        })?
    }

    /// Create a PBO-backed TensorImage on the GL thread.
    pub fn create_pbo_image(
        &self,
        width: usize,
        height: usize,
        fourcc: four_char_code::FourCharCode,
    ) -> Result<crate::TensorImage, Error> {
        let sender = self
            .sender
            .as_ref()
            .ok_or(Error::OpenGl("GL processor is shutting down".to_string()))?;

        let channels = crate::fourcc_channels(fourcc)?;
        let size = width * height * channels;
        if size == 0 {
            return Err(Error::OpenGl("Invalid image dimensions".to_string()));
        }

        // Allocate PBO on the GL thread
        let (tx, rx) = tokio::sync::oneshot::channel();
        sender
            .blocking_send(GLProcessorMessage::PboCreate(size, tx))
            .map_err(|_| Error::OpenGl("GL thread channel closed".to_string()))?;
        let buffer_id = rx
            .blocking_recv()
            .map_err(|_| Error::OpenGl("GL thread did not respond".to_string()))??;

        let ops: std::sync::Arc<dyn edgefirst_tensor::PboOps> = std::sync::Arc::new(GlPboOps {
            sender: sender.downgrade(),
        });

        let shape = if crate::fourcc_planar(fourcc)? {
            vec![channels, height, width]
        } else {
            vec![height, width, channels]
        };

        let pbo_tensor =
            edgefirst_tensor::PboTensor::<u8>::from_pbo(buffer_id, size, &shape, None, ops)
                .map_err(|e| Error::OpenGl(format!("PBO tensor creation failed: {e:?}")))?;
        let tensor = edgefirst_tensor::Tensor::Pbo(pbo_tensor);
        crate::TensorImage::from_tensor(tensor, fourcc)
            .map_err(|e| Error::OpenGl(format!("Failed to wrap PBO tensor as image: {e:?}")))
    }

    /// Returns the active transfer backend.
    #[allow(dead_code)]
    pub(crate) fn transfer_backend(&self) -> TransferBackend {
        self.transfer_backend
    }
}

impl Drop for GLProcessorThreaded {
    fn drop(&mut self) {
        drop(self.sender.take());
        let _ = self.handle.take().and_then(|h| h.join().ok());
    }
}

/// Interpolation mode for int8 proto textures (GL_R8I cannot use GL_LINEAR).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Int8InterpolationMode {
    /// texelFetch at nearest texel — simplest, fastest GPU execution.
    Nearest,
    /// texelFetch × 4 neighbors with shader-computed bilinear weights (default).
    Bilinear,
    /// Two-pass: dequant int8→f16 FBO, then existing f16 shader with GL_LINEAR.
    TwoPass,
}

/// Selects which EGLImage cache to use.
#[derive(Debug)]
enum CacheKind {
    Src,
    Dst,
}

/// A cached EGLImage with a weak reference to the source tensor's guard.
struct CachedEglImage {
    egl_image: EglImage,
    /// Weak reference to the source Tensor's BufferIdentity guard.
    guard: std::sync::Weak<()>,
    /// Optional GL renderbuffer backed by this EGLImage (used by direct RGB path).
    renderbuffer: Option<u32>,
    /// Monotonic access counter for LRU eviction.
    last_used: u64,
}

/// EGLImage cache owned by GLProcessorST.
///
/// Uses a HashMap with a monotonic counter for LRU eviction: each access
/// updates the entry's `last_used` timestamp, and eviction removes the entry
/// with the smallest `last_used` value.
struct EglImageCache {
    entries: std::collections::HashMap<u64, CachedEglImage>,
    capacity: usize,
    hits: u64,
    misses: u64,
    /// Monotonic counter incremented on each access for LRU tracking.
    access_counter: u64,
}

impl EglImageCache {
    fn new(capacity: usize) -> Self {
        Self {
            entries: std::collections::HashMap::with_capacity(capacity),
            capacity,
            hits: 0,
            misses: 0,
            access_counter: 0,
        }
    }

    /// Allocate a new LRU timestamp.
    fn next_timestamp(&mut self) -> u64 {
        self.access_counter += 1;
        self.access_counter
    }

    /// Evict the least recently used entry.
    fn evict_lru(&mut self) {
        if let Some((&evict_id, _)) = self.entries.iter().min_by_key(|(_, entry)| entry.last_used) {
            if let Some(evicted) = self.entries.remove(&evict_id) {
                if let Some(rbo) = evicted.renderbuffer {
                    unsafe { gls::gl::DeleteRenderbuffers(1, &rbo) };
                }
            }
        }
    }

    /// Sweep dead entries (tensor dropped, Weak is dead).
    fn sweep(&mut self) {
        let before = self.entries.len();
        self.entries.retain(|_id, entry| {
            let alive = entry.guard.upgrade().is_some();
            if !alive {
                if let Some(rbo) = entry.renderbuffer {
                    unsafe { gls::gl::DeleteRenderbuffers(1, &rbo) };
                }
            }
            alive
        });
        let swept = before - self.entries.len();
        if swept > 0 {
            log::debug!("EglImageCache: swept {swept} dead entries");
        }
    }
}

impl Drop for EglImageCache {
    fn drop(&mut self) {
        for entry in self.entries.values() {
            if let Some(rbo) = entry.renderbuffer {
                unsafe { gls::gl::DeleteRenderbuffers(1, &rbo) };
            }
        }
        log::debug!(
            "EglImageCache stats: {} hits, {} misses, {} entries remaining",
            self.hits,
            self.misses,
            self.entries.len()
        );
    }
}

/// OpenGL single-threaded image converter.
pub struct GLProcessorST {
    camera_eglimage_texture: Texture,
    camera_normal_texture: Texture,
    render_texture: Texture,
    segmentation_texture: Texture,
    segmentation_program: GlProgram,
    instanced_segmentation_program: GlProgram,
    proto_texture: Texture,
    proto_segmentation_program: GlProgram,
    proto_segmentation_int8_nearest_program: GlProgram,
    proto_segmentation_int8_bilinear_program: GlProgram,
    proto_dequant_int8_program: GlProgram,
    proto_segmentation_f32_program: GlProgram,
    color_program: GlProgram,
    /// Whether GL_OES_texture_float_linear is available (allows GL_LINEAR on R32F textures).
    has_float_linear: bool,
    /// Interpolation mode for int8 proto textures.
    int8_interpolation_mode: Int8InterpolationMode,
    /// Intermediate FBO texture for two-pass int8 dequant path.
    proto_dequant_texture: Texture,
    proto_mask_logit_int8_bilinear_program: GlProgram,
    proto_mask_logit_int8_nearest_program: GlProgram,
    proto_mask_logit_f32_program: GlProgram,
    /// Dedicated FBO for mask rendering.
    mask_fbo: u32,
    /// R8 texture attached to mask_fbo.
    mask_fbo_texture: u32,
    /// Current allocated width of mask FBO texture.
    mask_fbo_width: usize,
    /// Current allocated height of mask FBO texture.
    mask_fbo_height: usize,
    /// PBO buffer ID for atlas readback (0 = not allocated).
    mask_atlas_pbo: u32,
    vertex_buffer: Buffer,
    texture_buffer: Buffer,
    /// Persistent FBO for the convert() render path.
    /// Created once, reused by re-attaching textures each frame.
    convert_fbo: FrameBuffer,
    /// EGLImage cache for source DMA buffers.
    src_egl_cache: EglImageCache,
    /// EGLImage cache for destination DMA buffers.
    dst_egl_cache: EglImageCache,
    /// Intermediate RGBA texture for two-pass packed RGB conversion.
    /// Pass 1 renders YUYV/NV12→RGBA here; Pass 2 packs RGBA→RGB to DMA dest.
    packed_rgb_intermediate_tex: Texture,
    /// FBO for pass 1 of packed RGB conversion (renders to intermediate texture).
    packed_rgb_fbo: FrameBuffer,
    /// Current allocated size of the intermediate texture (0,0 = unallocated).
    packed_rgb_intermediate_size: (usize, usize),
    texture_program: GlProgram,
    texture_program_yuv: GlProgram,
    texture_program_planar: GlProgram,
    /// Shader: existing planar RGB with int8 bias (XOR 0x80) applied to output.
    texture_program_planar_int8: GlProgram,
    /// Shader: packed RGB -> RGBA8 packing (2D texture source, pass 2).
    packed_rgba8_program_2d: GlProgram,
    /// Shader: packed RGB int8 -> RGBA8 packing with XOR 0x80 (2D texture source, pass 2).
    packed_rgba8_int8_program_2d: GlProgram,
    /// Shader: direct RGB render with int8 XOR 0x80 bias (2D texture source).
    texture_int8_program: GlProgram,
    /// Shader: direct RGB render with int8 XOR 0x80 bias (external OES source).
    texture_int8_program_yuv: GlProgram,
    /// Whether the GPU supports direct RGB rendering via BGR888 renderbuffer.
    support_rgb_direct: bool,
    gl_context: GlContext,
}

impl Drop for GLProcessorST {
    fn drop(&mut self) {
        unsafe {
            {
                if self.mask_fbo != 0 {
                    gls::gl::DeleteFramebuffers(1, &self.mask_fbo);
                }
                if self.mask_fbo_texture != 0 {
                    gls::gl::DeleteTextures(1, &self.mask_fbo_texture);
                }
                if self.mask_atlas_pbo != 0 {
                    gls::gl::DeleteBuffers(1, &self.mask_atlas_pbo);
                }
            }
        }
    }
}

impl ImageProcessorTrait for GLProcessorST {
    fn convert(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        crop.check_crop(src, dst)?;
        if !Self::check_src_format_supported(self.gl_context.transfer_backend, src) {
            return Err(crate::Error::NotSupported(format!(
                "Opengl doesn't support {} source texture",
                src.fourcc().display()
            )));
        }

        if !Self::check_dst_format_supported(self.gl_context.transfer_backend, dst) {
            return Err(crate::Error::NotSupported(format!(
                "Opengl doesn't support {} destination texture",
                dst.fourcc().display()
            )));
        }
        log::debug!(
            "dst tensor: {:?} src tensor :{:?}",
            dst.tensor().memory(),
            src.tensor().memory()
        );
        check_gl_error(function!(), line!())?;
        if self.gl_context.transfer_backend.is_dma() && dst.tensor().memory() == TensorMemory::Dma {
            // Packed RGB is now supported via DMA with buffer reinterpretation
            let res = self.convert_dest_dma(dst, src, rotation, flip, crop);
            return res;
        }
        // PBO-to-PBO: both tensors are PBO-backed, use GL buffer bindings for
        // both upload and readback (zero CPU copy for both directions)
        if src.tensor().memory() == TensorMemory::Pbo && dst.tensor().memory() == TensorMemory::Pbo
        {
            return self.convert_pbo_to_pbo(dst, src, rotation, flip, crop);
        }
        // PBO dst with non-PBO src: use normal texture upload for src (which
        // maps the Mem/DMA tensor), but PBO PACK readback for dst.
        // This avoids the deadlock that would occur if convert_dest_non_dma
        // tried to map() the PBO dst on the GL thread.
        if dst.tensor().memory() == TensorMemory::Pbo {
            return self.convert_any_to_pbo(dst, src, rotation, flip, crop);
        }
        // PBO src with non-PBO dst: the src tensor's map() would deadlock on
        // the GL thread, so use PBO UNPACK upload. Readback goes to Mem dst
        // via normal ReadnPixels into mapped memory.
        if src.tensor().memory() == TensorMemory::Pbo {
            return self.convert_pbo_to_mem(dst, src, rotation, flip, crop);
        }
        let start = Instant::now();
        let res = self.convert_dest_non_dma(dst, src, rotation, flip, crop);
        log::debug!("convert_dest_non_dma takes {:?}", start.elapsed());
        res
    }

    fn convert_ref(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImageRef<'_>,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        // OpenGL doesn't support PLANAR_RGB output, delegate to CPU
        let mut cpu = CPUProcessor::new();
        cpu.convert_ref(src, dst, rotation, flip, crop)
    }

    fn draw_masks(
        &mut self,
        dst: &mut TensorImage,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
    ) -> Result<(), crate::Error> {
        use crate::FunctionTimer;

        let _timer = FunctionTimer::new("GLProcessorST::draw_masks");
        if !matches!(dst.fourcc(), RGBA | BGRA | RGB) {
            return Err(crate::Error::NotSupported(
                "Opengl image rendering only supports RGBA, BGRA, or RGB images".to_string(),
            ));
        }

        let is_dma = match dst.tensor.memory() {
            edgefirst_tensor::TensorMemory::Dma if self.setup_renderbuffer_dma(dst).is_ok() => true,
            _ => {
                // Add dest rect to make sure dst is rendered fully
                self.setup_renderbuffer_non_dma(
                    dst,
                    Crop::new().with_dst_rect(Some(Rect::new(0, 0, 0, 0))),
                )?;
                false
            }
        };

        gls::enable(gls::gl::BLEND);
        gls::blend_func_separate(
            gls::gl::SRC_ALPHA,
            gls::gl::ONE_MINUS_SRC_ALPHA,
            gls::gl::ZERO,
            gls::gl::ONE,
        );

        self.render_box(dst, detect)?;
        self.render_segmentation(detect, segmentation)?;

        gls::finish();
        if !is_dma {
            let mut dst_map = dst.tensor().map()?;
            let format = match dst.fourcc() {
                RGB => gls::gl::RGB,
                RGBA => gls::gl::RGBA,
                BGRA => 0x80E1, // GL_BGRA (GL_EXT_texture_format_BGRA8888)
                _ => unreachable!(),
            };
            unsafe {
                gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
                gls::gl::ReadnPixels(
                    0,
                    0,
                    dst.width() as i32,
                    dst.height() as i32,
                    format,
                    gls::gl::UNSIGNED_BYTE,
                    dst.tensor.len() as i32,
                    dst_map.as_mut_ptr() as *mut c_void,
                );
            }
        }

        Ok(())
    }

    fn draw_masks_proto(
        &mut self,
        dst: &mut TensorImage,
        detect: &[DetectBox],
        proto_data: &ProtoData,
    ) -> crate::Result<()> {
        use crate::FunctionTimer;

        let _timer = FunctionTimer::new("GLProcessorST::draw_masks_proto");
        if !matches!(dst.fourcc(), RGBA | BGRA | RGB) {
            return Err(crate::Error::NotSupported(
                "Opengl image rendering only supports RGBA, BGRA, or RGB images".to_string(),
            ));
        }

        let is_dma = match dst.tensor.memory() {
            edgefirst_tensor::TensorMemory::Dma if self.setup_renderbuffer_dma(dst).is_ok() => true,
            _ => {
                self.setup_renderbuffer_non_dma(
                    dst,
                    Crop::new().with_dst_rect(Some(Rect::new(0, 0, 0, 0))),
                )?;
                false
            }
        };

        gls::enable(gls::gl::BLEND);
        gls::blend_func_separate(
            gls::gl::SRC_ALPHA,
            gls::gl::ONE_MINUS_SRC_ALPHA,
            gls::gl::ZERO,
            gls::gl::ONE,
        );

        self.render_box(dst, detect)?;
        self.render_proto_segmentation(detect, proto_data)?;

        gls::finish();
        if !is_dma {
            let mut dst_map = dst.tensor().map()?;
            let format = match dst.fourcc() {
                RGB => gls::gl::RGB,
                RGBA => gls::gl::RGBA,
                BGRA => 0x80E1, // GL_BGRA (GL_EXT_texture_format_BGRA8888)
                _ => unreachable!(),
            };
            unsafe {
                gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
                gls::gl::ReadnPixels(
                    0,
                    0,
                    dst.width() as i32,
                    dst.height() as i32,
                    format,
                    gls::gl::UNSIGNED_BYTE,
                    dst.tensor.len() as i32,
                    dst_map.as_mut_ptr() as *mut c_void,
                );
            }
        }

        Ok(())
    }

    fn decode_masks_atlas(
        &mut self,
        detect: &[DetectBox],
        proto_data: ProtoData,
        output_width: usize,
        output_height: usize,
    ) -> crate::Result<(Vec<u8>, Vec<MaskRegion>)> {
        GLProcessorST::decode_masks_atlas(self, detect, &proto_data, output_width, output_height)
    }

    fn set_class_colors(&mut self, colors: &[[u8; 4]]) -> crate::Result<()> {
        if colors.is_empty() {
            return Ok(());
        }
        let mut colors_f32 = colors
            .iter()
            .map(|c| {
                [
                    c[0] as f32 / 255.0,
                    c[1] as f32 / 255.0,
                    c[2] as f32 / 255.0,
                    c[3] as f32 / 255.0,
                ]
            })
            .take(20)
            .collect::<Vec<[f32; 4]>>();

        self.segmentation_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.instanced_segmentation_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.proto_segmentation_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.proto_segmentation_int8_nearest_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.proto_segmentation_int8_bilinear_program
            .load_uniform_4fv(c"colors", &colors_f32)?;
        self.proto_segmentation_f32_program
            .load_uniform_4fv(c"colors", &colors_f32)?;

        colors_f32.iter_mut().for_each(|c| {
            c[3] = 1.0; // set alpha to 1.0 for color rendering
        });
        self.color_program
            .load_uniform_4fv(c"colors", &colors_f32)?;

        Ok(())
    }
}

impl GLProcessorST {
    pub fn new(kind: Option<EglDisplayKind>) -> Result<GLProcessorST, crate::Error> {
        let gl_context = GlContext::new(kind)?;
        gls::load_with(|s| {
            gl_context
                .egl
                .get_proc_address(s)
                .map_or(std::ptr::null(), |p| p as *const _)
        });

        let has_float_linear = Self::gl_check_support()?;

        // Uploads and downloads are all packed with no alignment requirements
        unsafe {
            gls::gl::PixelStorei(gls::gl::PACK_ALIGNMENT, 1);
            gls::gl::PixelStorei(gls::gl::UNPACK_ALIGNMENT, 1);
        }

        let texture_program_planar =
            GlProgram::new(generate_vertex_shader(), generate_planar_rgb_shader())?;

        let texture_program =
            GlProgram::new(generate_vertex_shader(), generate_texture_fragment_shader())?;

        let texture_program_yuv = GlProgram::new(
            generate_vertex_shader(),
            generate_texture_fragment_shader_yuv(),
        )?;

        let segmentation_program =
            GlProgram::new(generate_vertex_shader(), generate_segmentation_shader())?;
        segmentation_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;
        let instanced_segmentation_program = GlProgram::new(
            generate_vertex_shader(),
            generate_instanced_segmentation_shader(),
        )?;
        instanced_segmentation_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        // Existing f16 proto shader (RGBA16F, 4 protos per layer)
        let proto_segmentation_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_segmentation_shader(),
        )?;
        proto_segmentation_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        // Int8 proto shaders (R8I, 1 proto per layer, 32 layers)
        let proto_segmentation_int8_nearest_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_segmentation_shader_int8_nearest(),
        )?;
        proto_segmentation_int8_nearest_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        let proto_segmentation_int8_bilinear_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_segmentation_shader_int8_bilinear(),
        )?;
        proto_segmentation_int8_bilinear_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        let proto_dequant_int8_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_dequant_shader_int8(),
        )?;

        // F32 proto shader (R32F, 1 proto per layer, 32 layers)
        let proto_segmentation_f32_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_segmentation_shader_f32(),
        )?;
        proto_segmentation_f32_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        let color_program = GlProgram::new(generate_vertex_shader(), generate_color_shader())?;
        color_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        // Binary logit-threshold mask shaders (atlas path — skip sigmoid)
        let proto_mask_logit_int8_nearest_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_mask_logit_shader_int8_nearest(),
        )?;
        let proto_mask_logit_int8_bilinear_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_mask_logit_shader_int8_bilinear(),
        )?;
        let proto_mask_logit_f32_program = GlProgram::new(
            generate_vertex_shader(),
            generate_proto_mask_logit_shader_f32(),
        )?;

        // Int8 variant of the existing planar RGB shader (for PLANAR_RGB_INT8 destinations).
        let texture_program_planar_int8 =
            GlProgram::new(generate_vertex_shader(), generate_planar_rgb_int8_shader())?;

        // RGB packing shaders (2D only — used in pass 2 of two-pass pipeline)
        let packed_rgba8_program_2d =
            GlProgram::new(generate_vertex_shader(), generate_packed_rgba8_shader_2d())?;
        let packed_rgba8_int8_program_2d = GlProgram::new(
            generate_vertex_shader(),
            generate_packed_rgba8_int8_shader_2d(),
        )?;

        // Int8 direct-render shaders (for RGB_INT8 destinations via direct path)
        let texture_int8_program =
            GlProgram::new(generate_vertex_shader(), generate_texture_int8_shader())?;
        let texture_int8_program_yuv =
            GlProgram::new(generate_vertex_shader(), generate_texture_int8_shader_yuv())?;

        let camera_eglimage_texture = Texture::new();
        let camera_normal_texture = Texture::new();
        let render_texture = Texture::new();
        let segmentation_texture = Texture::new();
        let proto_texture = Texture::new();
        let proto_dequant_texture = Texture::new();
        let vertex_buffer = Buffer::new(0, 3, 100);
        let texture_buffer = Buffer::new(1, 2, 100);

        let mut converter = GLProcessorST {
            gl_context,
            texture_program,
            texture_program_yuv,
            texture_program_planar,
            texture_program_planar_int8,
            packed_rgba8_program_2d,
            packed_rgba8_int8_program_2d,
            texture_int8_program,
            texture_int8_program_yuv,
            support_rgb_direct: false, // will be probed in Task 3
            camera_eglimage_texture,
            camera_normal_texture,
            segmentation_texture,
            proto_texture,
            proto_segmentation_int8_nearest_program,
            proto_segmentation_int8_bilinear_program,
            proto_dequant_int8_program,
            proto_segmentation_f32_program,
            has_float_linear,
            int8_interpolation_mode: Int8InterpolationMode::Bilinear,
            proto_dequant_texture,
            proto_mask_logit_int8_bilinear_program,
            proto_mask_logit_int8_nearest_program,
            proto_mask_logit_f32_program,
            mask_fbo: 0,
            mask_fbo_texture: 0,
            mask_fbo_width: 0,
            mask_fbo_height: 0,
            mask_atlas_pbo: 0,
            vertex_buffer,
            texture_buffer,
            convert_fbo: FrameBuffer::new(),
            src_egl_cache: EglImageCache::new(8),
            dst_egl_cache: EglImageCache::new(8),
            packed_rgb_intermediate_tex: Texture::new(),
            packed_rgb_fbo: FrameBuffer::new(),
            packed_rgb_intermediate_size: (0, 0),
            render_texture,
            segmentation_program,
            instanced_segmentation_program,
            proto_segmentation_program,
            color_program,
        };
        check_gl_error(function!(), line!())?;

        // Probe GPU capability for direct RGB rendering
        converter.support_rgb_direct = converter.probe_rgb_direct_support();

        // Verify DMA-buf actually works (catches NVIDIA discrete GPUs where
        // EGLImage creation succeeds but rendered data is all zeros)
        if converter.gl_context.transfer_backend.is_dma() && !converter.verify_dma_buf_roundtrip() {
            log::info!("DMA-buf verification failed — falling back to PBO transfers");
            converter.gl_context.transfer_backend = TransferBackend::Pbo;
            // RGB direct rendering also requires DMA, so disable it
            converter.support_rgb_direct = false;
        }

        // If DMA-buf failed/unavailable but GL is alive, use PBO transfers
        if converter.gl_context.transfer_backend == TransferBackend::Sync {
            log::info!("Upgrading transfer backend from Sync to Pbo (GL context available)");
            converter.gl_context.transfer_backend = TransferBackend::Pbo;
        }

        log::debug!(
            "GLConverter created (transfer={:?}, rgb_direct={})",
            converter.gl_context.transfer_backend,
            converter.support_rgb_direct
        );
        Ok(converter)
    }

    /// Probe whether the GPU supports direct RGB rendering via BGR888 DMA-buf
    /// backed renderbuffer. Creates a small test FBO and checks completeness.
    /// Returns `false` on any failure (DMA unavailable, EGLImage rejected, FBO incomplete).
    fn probe_rgb_direct_support(&self) -> bool {
        if !self.gl_context.transfer_backend.is_dma() {
            log::debug!("probe_rgb_direct: no DMA support");
            return false;
        }

        // Check glEGLImageTargetRenderbufferStorageOES is available
        if self
            .gl_context
            .egl
            .get_proc_address("glEGLImageTargetRenderbufferStorageOES")
            .is_none()
        {
            log::debug!("probe_rgb_direct: glEGLImageTargetRenderbufferStorageOES not available");
            return false;
        }

        // Allocate a small test DMA buffer (64x64 RGB = 12288 bytes)
        let test_img = match TensorImage::new(64, 64, RGB, Some(TensorMemory::Dma)) {
            Ok(img) => img,
            Err(e) => {
                log::debug!("probe_rgb_direct: failed to allocate test DMA buffer: {e}");
                return false;
            }
        };

        // Create EGLImage from the test DMA buffer
        let egl_image =
            match self.create_egl_image_with_dims(&test_img, 64, 64, DrmFourcc::Bgr888, 3) {
                Ok(img) => img,
                Err(e) => {
                    log::debug!("probe_rgb_direct: EGLImage creation failed: {e}");
                    return false;
                }
            };

        // Create renderbuffer, bind EGLImage, create FBO, check completeness
        let result = unsafe {
            let mut rbo = 0u32;
            gls::gl::GenRenderbuffers(1, &mut rbo);
            gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, rbo);
            gls::gl::EGLImageTargetRenderbufferStorageOES(
                gls::gl::RENDERBUFFER,
                egl_image.egl_image.as_ptr(),
            );

            let gl_err = gls::gl::GetError();
            if gl_err != gls::gl::NO_ERROR {
                log::debug!(
                    "probe_rgb_direct: EGLImageTargetRenderbufferStorageOES failed: {gl_err:#X}"
                );
                gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, 0);
                gls::gl::DeleteRenderbuffers(1, &rbo);
                return false;
            }

            let mut fbo = 0u32;
            gls::gl::GenFramebuffers(1, &mut fbo);
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
            gls::gl::FramebufferRenderbuffer(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::RENDERBUFFER,
                rbo,
            );

            let status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
            let complete = status == gls::gl::FRAMEBUFFER_COMPLETE;

            // Cleanup
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
            gls::gl::DeleteFramebuffers(1, &fbo);
            gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, 0);
            gls::gl::DeleteRenderbuffers(1, &rbo);

            complete
        };
        // egl_image and test_img drop automatically here

        log::info!("probe_rgb_direct: BGR888 renderbuffer FBO support = {result}");
        result
    }

    /// Verify that DMA-buf EGLImage round-trip actually works on this GPU.
    ///
    /// Renders a solid red quad to a 64x64 DMA-buf-backed RGBA texture via
    /// EGLImage, then reads it back and checks that the center pixel is red.
    /// Returns `true` if the data round-trips correctly.
    ///
    /// This catches GPUs like NVIDIA discrete where `eglCreateImage` from
    /// `dma_heap` fds succeeds but the rendered data is all zeros.
    fn verify_dma_buf_roundtrip(&mut self) -> bool {
        // Allocate a 64x64 RGBA DMA source tensor and fill it with solid red
        let src = match TensorImage::new(64, 64, RGBA, Some(TensorMemory::Dma)) {
            Ok(img) => img,
            Err(e) => {
                log::info!("verify_dma_buf_roundtrip: failed to allocate DMA source: {e}");
                return false;
            }
        };

        {
            let mut map = match src.tensor().map() {
                Ok(m) => m,
                Err(e) => {
                    log::info!("verify_dma_buf_roundtrip: failed to map DMA source: {e}");
                    return false;
                }
            };
            for pixel in map.chunks_exact_mut(4) {
                pixel[0] = 255; // R
                pixel[1] = 0; // G
                pixel[2] = 0; // B
                pixel[3] = 255; // A
            }
        }

        // Allocate a 64x64 RGBA DMA destination tensor
        let mut dst = match TensorImage::new(64, 64, RGBA, Some(TensorMemory::Dma)) {
            Ok(img) => img,
            Err(e) => {
                log::info!("verify_dma_buf_roundtrip: failed to allocate DMA destination: {e}");
                return false;
            }
        };

        // Run the full DMA-buf EGLImage render pipeline
        if let Err(e) =
            self.convert_dest_dma(&mut dst, &src, Rotation::None, Flip::None, Crop::no_crop())
        {
            log::info!("verify_dma_buf_roundtrip: convert_dest_dma failed: {e}");
            return false;
        }

        // Read back the center pixel at (32, 32) from the destination
        let map = match dst.tensor().map() {
            Ok(m) => m,
            Err(e) => {
                log::info!("verify_dma_buf_roundtrip: failed to map DMA destination: {e}");
                return false;
            }
        };

        let offset = (32 * 64 + 32) * 4;
        if map.len() < offset + 4 {
            log::info!("verify_dma_buf_roundtrip: destination buffer too small");
            return false;
        }

        let r = map[offset];
        let g = map[offset + 1];
        let b = map[offset + 2];
        let a = map[offset + 3];

        let pass = r > 250 && g < 5 && b < 5 && a > 250;

        if pass {
            log::info!("verify_dma_buf_roundtrip: PASSED (center pixel RGBA={r},{g},{b},{a})");
        } else {
            log::info!(
                "verify_dma_buf_roundtrip: FAILED (center pixel RGBA={r},{g},{b},{a}, \
                 expected ~255,0,0,255)"
            );
        }

        pass
    }

    /// Compute padded bbox regions and atlas offsets for a set of detections.
    ///
    /// Returns the vector of `MaskRegion` with stacked atlas_y_offset values
    /// and the total compact atlas height.
    fn compute_atlas_regions(
        detect: &[DetectBox],
        output_width: usize,
        output_height: usize,
        padding: usize,
    ) -> (Vec<MaskRegion>, usize) {
        let ow = output_width as i32;
        let oh = output_height as i32;
        let owf = output_width as f32;
        let ohf = output_height as f32;
        let pad = padding as i32;

        let mut regions = Vec::with_capacity(detect.len());
        let mut atlas_y = 0usize;
        for det in detect.iter() {
            let bbox_x = (det.bbox.xmin * owf).round() as i32;
            let bbox_y = (det.bbox.ymin * ohf).round() as i32;
            let bbox_w = ((det.bbox.xmax - det.bbox.xmin) * owf).round() as i32;
            let bbox_h = ((det.bbox.ymax - det.bbox.ymin) * ohf).round() as i32;
            let bbox_x = bbox_x.max(0).min(ow);
            let bbox_y = bbox_y.max(0).min(oh);
            let bbox_w = bbox_w.max(1).min(ow - bbox_x);
            let bbox_h = bbox_h.max(1).min(oh - bbox_y);

            let padded_x = (bbox_x - pad).max(0);
            let padded_y = (bbox_y - pad).max(0);
            let padded_w = ((bbox_x + bbox_w + pad).min(ow) - padded_x).max(1);
            let padded_h = ((bbox_y + bbox_h + pad).min(oh) - padded_y).max(1);

            regions.push(MaskRegion {
                atlas_y_offset: atlas_y,
                padded_x: padded_x as usize,
                padded_y: padded_y as usize,
                padded_w: padded_w as usize,
                padded_h: padded_h as usize,
                bbox_x: bbox_x as usize,
                bbox_y: bbox_y as usize,
                bbox_w: bbox_w as usize,
                bbox_h: bbox_h as usize,
            });
            atlas_y += padded_h as usize;
        }
        (regions, atlas_y)
    }

    /// Sets the interpolation mode for int8 proto textures.
    pub fn set_int8_interpolation_mode(&mut self, mode: Int8InterpolationMode) {
        self.int8_interpolation_mode = mode;
        log::debug!("Int8 interpolation mode set to {:?}", mode);
    }

    /// Ensures the mask FBO + R8 texture are allocated at the given dimensions.
    /// Creates or resizes the FBO and texture as needed.
    fn ensure_mask_fbo(&mut self, width: usize, height: usize) -> crate::Result<()> {
        if self.mask_fbo_width == width && self.mask_fbo_height == height && self.mask_fbo != 0 {
            return Ok(());
        }

        // Create FBO if needed
        if self.mask_fbo == 0 {
            unsafe {
                gls::gl::GenFramebuffers(1, &mut self.mask_fbo);
            }
        }
        // Create texture if needed
        if self.mask_fbo_texture == 0 {
            unsafe {
                gls::gl::GenTextures(1, &mut self.mask_fbo_texture);
            }
        }

        // Allocate R8 texture
        unsafe {
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.mask_fbo_texture);
            gls::gl::TexImage2D(
                gls::gl::TEXTURE_2D,
                0,
                gls::gl::R8 as i32,
                width as i32,
                height as i32,
                0,
                gls::gl::RED,
                gls::gl::UNSIGNED_BYTE,
                std::ptr::null(),
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::NEAREST as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::NEAREST as i32,
            );
        }

        // Attach to FBO
        unsafe {
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, self.mask_fbo);
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                self.mask_fbo_texture,
                0,
            );
            let status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
            if status != gls::gl::FRAMEBUFFER_COMPLETE {
                return Err(crate::Error::OpenGl(format!(
                    "Mask FBO incomplete: status=0x{status:X}"
                )));
            }
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
        }

        self.mask_fbo_width = width;
        self.mask_fbo_height = height;
        log::debug!("Mask FBO allocated at {width}x{height}");
        Ok(())
    }

    /// Ensures the mask atlas FBO and PBO are allocated for the given total
    /// atlas dimensions.  Unlike `ensure_mask_atlas`, the caller provides
    /// the exact atlas height (e.g. sum of padded bbox heights).
    fn ensure_mask_atlas_size(&mut self, width: usize, atlas_height: usize) -> crate::Result<()> {
        if self.mask_fbo_width == width
            && self.mask_fbo_height >= atlas_height
            && self.mask_fbo != 0
            && self.mask_atlas_pbo != 0
        {
            return Ok(());
        }
        self.ensure_mask_fbo(width, atlas_height)?;
        let pbo_size = width * atlas_height;
        unsafe {
            if self.mask_atlas_pbo == 0 {
                gls::gl::GenBuffers(1, &mut self.mask_atlas_pbo);
            }
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, self.mask_atlas_pbo);
            gls::gl::BufferData(
                gls::gl::PIXEL_PACK_BUFFER,
                pbo_size as isize,
                std::ptr::null(),
                gls::gl::DYNAMIC_READ,
            );
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
        }
        Ok(())
    }

    /// Decode all detection masks into a single atlas texture and read back
    /// as a contiguous buffer, with one PBO readback for all masks.
    ///
    /// Returns `(atlas_pixels, metadata)` where `atlas_pixels` is a contiguous
    /// `Vec<u8>` of size `output_width * compact_atlas_height` (where
    /// `compact_atlas_height` is the sum of padded bbox heights) and `metadata`
    /// contains per-detection bbox info (with empty pixel vecs).
    pub fn decode_masks_atlas(
        &mut self,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        output_width: usize,
        output_height: usize,
    ) -> crate::Result<(Vec<u8>, Vec<MaskRegion>)> {
        use crate::FunctionTimer;

        let _timer = FunctionTimer::new("GLProcessorST::decode_masks_atlas");

        if detect.is_empty() || proto_data.mask_coefficients.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let padding = 4usize;

        let (height, width, num_protos) = proto_data.protos.dim();
        let texture_target = gls::gl::TEXTURE_2D_ARRAY;

        // Pre-compute atlas regions and total height to size the FBO/PBO
        let (regions, compact_atlas_height) =
            Self::compute_atlas_regions(detect, output_width, output_height, padding);

        // Save current FBO and viewport
        let (saved_fbo, saved_viewport) = unsafe {
            let mut fbo: i32 = 0;
            gls::gl::GetIntegerv(gls::gl::FRAMEBUFFER_BINDING, &mut fbo);
            let mut vp = [0i32; 4];
            gls::gl::GetIntegerv(gls::gl::VIEWPORT, vp.as_mut_ptr());
            (fbo as u32, vp)
        };

        // Ensure atlas FBO and PBO are allocated for the compact size
        self.ensure_mask_atlas_size(output_width, compact_atlas_height)?;

        // Upload proto texture array and select the logit-threshold shader
        gls::active_texture(gls::gl::TEXTURE0);
        gls::bind_texture(texture_target, self.proto_texture.id);
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MIN_FILTER,
            gls::gl::NEAREST as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MAG_FILTER,
            gls::gl::NEAREST as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_S,
            gls::gl::CLAMP_TO_EDGE as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_T,
            gls::gl::CLAMP_TO_EDGE as i32,
        );

        let atlas_result = match &proto_data.protos {
            ProtoTensor::Quantized {
                protos,
                quantization,
            } => {
                let mut tex_data = vec![0i8; height * width * num_protos];
                for k in 0..num_protos {
                    for y in 0..height {
                        for x in 0..width {
                            tex_data[k * height * width + y * width + x] = protos[[y, x, k]];
                        }
                    }
                }
                gls::tex_image3d(
                    texture_target,
                    0,
                    gls::gl::R8I as i32,
                    width as i32,
                    height as i32,
                    num_protos as i32,
                    0,
                    gls::gl::RED_INTEGER,
                    gls::gl::BYTE,
                    Some(&tex_data),
                );

                let proto_scale = quantization.scale;
                let proto_scaled_zp = -(quantization.zero_point as f32) * quantization.scale;

                let program = match self.int8_interpolation_mode {
                    Int8InterpolationMode::Nearest => &self.proto_mask_logit_int8_nearest_program,
                    _ => &self.proto_mask_logit_int8_bilinear_program,
                };
                gls::use_program(program.id);
                program.load_uniform_1i(c"num_protos", num_protos as i32)?;
                program.load_uniform_1f(c"proto_scale", proto_scale)?;

                self.render_mask_atlas_compact(
                    program,
                    regions,
                    &proto_data.mask_coefficients,
                    output_width,
                    output_height,
                    Some(proto_scaled_zp),
                )
            }
            ProtoTensor::Float(protos_f32) => {
                let mut tex_data = vec![0.0f32; height * width * num_protos];
                for k in 0..num_protos {
                    for y in 0..height {
                        for x in 0..width {
                            tex_data[k * height * width + y * width + x] = protos_f32[[y, x, k]];
                        }
                    }
                }
                gls::tex_image3d(
                    texture_target,
                    0,
                    gls::gl::R32F as i32,
                    width as i32,
                    height as i32,
                    num_protos as i32,
                    0,
                    gls::gl::RED,
                    gls::gl::FLOAT,
                    Some(&tex_data),
                );
                if self.has_float_linear {
                    gls::tex_parameteri(
                        texture_target,
                        gls::gl::TEXTURE_MIN_FILTER,
                        gls::gl::LINEAR as i32,
                    );
                    gls::tex_parameteri(
                        texture_target,
                        gls::gl::TEXTURE_MAG_FILTER,
                        gls::gl::LINEAR as i32,
                    );
                }

                let program = &self.proto_mask_logit_f32_program;
                gls::use_program(program.id);
                program.load_uniform_1i(c"num_protos", num_protos as i32)?;

                self.render_mask_atlas_compact(
                    program,
                    regions,
                    &proto_data.mask_coefficients,
                    output_width,
                    output_height,
                    None,
                )
            }
        };

        // Restore previous FBO + viewport
        unsafe {
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, saved_fbo);
            gls::gl::Viewport(
                saved_viewport[0],
                saved_viewport[1],
                saved_viewport[2],
                saved_viewport[3],
            );
        }

        let (atlas_pixels, regions) = atlas_result?;
        Ok((atlas_pixels, regions))
    }

    /// Render all detection masks into a compact atlas where each strip is
    /// sized to the padded bounding box, not the full output resolution.
    ///
    /// The atlas width equals `output_width`; each detection occupies a
    /// horizontal strip whose height is the padded bbox height.  Strips are
    /// stacked vertically.  A single PBO readback retrieves the entire atlas.
    ///
    /// Returns `(atlas_pixels, regions)` where `regions` describes each
    /// detection's location within the atlas.
    #[allow(clippy::too_many_arguments)]
    fn render_mask_atlas_compact(
        &self,
        program: &GlProgram,
        regions: Vec<MaskRegion>,
        mask_coefficients: &[Vec<f32>],
        output_width: usize,
        output_height: usize,
        proto_scaled_zp: Option<f32>,
    ) -> crate::Result<(Vec<u8>, Vec<MaskRegion>)> {
        if regions.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }

        let owf = output_width as f32;
        let ohf = output_height as f32;

        let atlas_height = regions.last().map_or(0, |r| r.atlas_y_offset + r.padded_h);
        let ahf = atlas_height as f32;

        unsafe {
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, self.mask_fbo);
            gls::gl::Viewport(0, 0, output_width as i32, atlas_height as i32);
            gls::gl::Disable(gls::gl::BLEND);
            gls::gl::ClearColor(0.0, 0.0, 0.0, 0.0);
            gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
        }

        if let Some(first_coeff) = mask_coefficients.first() {
            if first_coeff.len() > 32 {
                log::warn!(
                    "render_mask_atlas_compact: {} mask coefficients exceeds shader \
                     limit of 32 — coefficients will be truncated",
                    first_coeff.len()
                );
            }
        }

        for (region, coeff) in regions.iter().zip(mask_coefficients.iter()) {
            let mut packed_coeff = [[0.0f32; 4]; 8];
            for (j, val) in coeff.iter().enumerate().take(32) {
                packed_coeff[j / 4][j % 4] = *val;
            }
            program.load_uniform_4fv(c"mask_coeff", &packed_coeff)?;

            // For int8 paths: upload precomputed coeff_sum * scaled_zp
            if let Some(szp) = proto_scaled_zp {
                let coeff_sum: f32 = coeff.iter().take(32).sum();
                program.load_uniform_1f(c"coeff_sum_x_szp", coeff_sum * szp)?;
            }

            // The bbox quad position in the atlas:
            // - X: the padded bbox horizontal position (same as in output coords)
            // - Y: the strip's vertical offset in the atlas
            let dst_left = region.padded_x as f32 / owf * 2.0 - 1.0;
            let dst_right = (region.padded_x + region.padded_w) as f32 / owf * 2.0 - 1.0;
            let dst_bottom = region.atlas_y_offset as f32 / ahf * 2.0 - 1.0;
            let dst_top = (region.atlas_y_offset + region.padded_h) as f32 / ahf * 2.0 - 1.0;

            // Proto texture coords map the padded bbox to proto space
            let src_left = region.padded_x as f32 / owf;
            let src_right = (region.padded_x + region.padded_w) as f32 / owf;
            let src_bottom = region.padded_y as f32 / ohf;
            let src_top = (region.padded_y + region.padded_h) as f32 / ohf;

            unsafe {
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
                let verts: [f32; 12] = [
                    dst_left, dst_top, 0.0, dst_right, dst_top, 0.0, dst_right, dst_bottom, 0.0,
                    dst_left, dst_bottom, 0.0,
                ];
                gls::gl::BufferSubData(
                    gls::gl::ARRAY_BUFFER,
                    0,
                    (size_of::<f32>() * 12) as isize,
                    verts.as_ptr() as *const c_void,
                );

                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
                gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
                let tc: [f32; 8] = [
                    src_left, src_top, src_right, src_top, src_right, src_bottom, src_left,
                    src_bottom,
                ];
                gls::gl::BufferSubData(
                    gls::gl::ARRAY_BUFFER,
                    0,
                    (size_of::<f32>() * 8) as isize,
                    tc.as_ptr() as *const c_void,
                );

                let idx: [u32; 4] = [0, 1, 2, 3];
                gls::gl::DrawElements(
                    gls::gl::TRIANGLE_FAN,
                    4,
                    gls::gl::UNSIGNED_INT,
                    idx.as_ptr() as *const c_void,
                );
            }
        }

        // Single readback for the compact atlas
        let atlas_bytes = output_width * atlas_height;
        let mut pixels = vec![0u8; atlas_bytes];

        unsafe {
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, self.mask_atlas_pbo);
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                output_width as i32,
                atlas_height as i32,
                gls::gl::RED,
                gls::gl::UNSIGNED_BYTE,
                atlas_bytes as i32,
                std::ptr::null_mut(),
            );
            gls::gl::Finish();

            let ptr = gls::gl::MapBufferRange(
                gls::gl::PIXEL_PACK_BUFFER,
                0,
                atlas_bytes as isize,
                gls::gl::MAP_READ_BIT,
            );
            if ptr.is_null() {
                gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
                return Err(crate::Error::OpenGl(
                    "Failed to map compact atlas PBO for readback".to_string(),
                ));
            }
            std::ptr::copy_nonoverlapping(ptr as *const u8, pixels.as_mut_ptr(), atlas_bytes);
            gls::gl::UnmapBuffer(gls::gl::PIXEL_PACK_BUFFER);
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
        }

        Ok((pixels, regions))
    }

    fn check_src_format_supported(backend: TransferBackend, img: &TensorImage) -> bool {
        if backend.is_dma() && img.tensor().memory() == TensorMemory::Dma {
            // EGLImage supports RGBA, GREY, YUYV, and NV12 for DMA buffers.
            // VYUY excluded: Vivante GPU accepts the DRM fourcc but produces
            // incorrect output (similarity ~0.28 vs reference).
            matches!(img.fourcc(), RGBA | GREY | YUYV | NV12)
        } else {
            matches!(img.fourcc(), RGB | RGBA | GREY)
        }
    }

    fn check_dst_format_supported(backend: TransferBackend, img: &TensorImage) -> bool {
        if backend.is_dma() && img.tensor().memory() == TensorMemory::Dma {
            matches!(
                img.fourcc(),
                RGBA | BGRA | GREY | PLANAR_RGB | RGB | RGB_INT8 | PLANAR_RGB_INT8
            )
        } else {
            matches!(img.fourcc(), RGB | RGBA | BGRA | GREY | RGB_INT8)
        }
    }

    /// Checks required GL extensions and returns whether optional capabilities
    /// are available. Returns `has_float_linear` (GL_OES_texture_float_linear).
    fn gl_check_support() -> Result<bool, crate::Error> {
        if let Ok(version) = gls::get_string(gls::gl::SHADING_LANGUAGE_VERSION) {
            log::debug!("GL Shading Language Version: {version:?}");
        } else {
            log::warn!("Could not get GL Shading Language Version");
        }

        let extensions = unsafe {
            let str = gls::gl::GetString(gls::gl::EXTENSIONS);
            if str.is_null() {
                return Err(crate::Error::GLVersion(
                    "GL returned no supported extensions".to_string(),
                ));
            }
            CStr::from_ptr(str as *const c_char)
                .to_string_lossy()
                .to_string()
        };
        log::debug!("GL Extensions: {extensions}");
        let required_ext = ["GL_OES_EGL_image_external_essl3"];
        let extensions = extensions.split_ascii_whitespace().collect::<BTreeSet<_>>();
        for required in required_ext {
            if !extensions.contains(required) {
                return Err(crate::Error::GLVersion(format!(
                    "GL does not support {required} extension",
                )));
            }
        }

        let has_float_linear = extensions.contains("GL_OES_texture_float_linear");
        log::debug!("GL_OES_texture_float_linear: {has_float_linear}");

        Ok(has_float_linear)
    }

    fn setup_renderbuffer_dma(&mut self, dst: &TensorImage) -> crate::Result<()> {
        self.convert_fbo.bind();

        let (width, height) = if matches!(dst.fourcc(), PLANAR_RGB | PLANAR_RGB_INT8) {
            let width = dst.width();
            let height = dst.height() * 3;
            (width as i32, height as i32)
        } else {
            (dst.width() as i32, dst.height() as i32)
        };
        let dest_egl = self.get_or_create_egl_image(CacheKind::Dst, dst)?;
        unsafe {
            gls::gl::UseProgram(self.texture_program_yuv.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.render_texture.id);
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, dest_egl.as_ptr());
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                self.render_texture.id,
                0,
            );
            check_gl_error(function!(), line!())?;
            gls::gl::Viewport(0, 0, width, height);
        }
        Ok(())
    }

    fn convert_dest_dma(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        assert!(self.gl_context.transfer_backend.is_dma());
        if fourcc_is_packed_rgb(dst.fourcc()) {
            if self.support_rgb_direct {
                self.convert_to_rgb_direct(src, dst, rotation, flip, crop)
            } else {
                // Two-pass packed RGB is slower than G2D/CPU; decline so
                // ImageProcessor falls through to a faster backend.
                Err(crate::Error::NotSupported(
                    "OpenGL two-pass packed RGB disabled (no direct RGB support)".into(),
                ))
            }
        } else if dst.is_planar() {
            self.setup_renderbuffer_dma(dst)?;
            self.convert_to_planar(src, dst, rotation, flip, crop)
        } else {
            self.setup_renderbuffer_dma(dst)?;
            self.convert_to(src, dst, rotation, flip, crop)
        }
    }

    fn setup_renderbuffer_non_dma(&mut self, dst: &TensorImage, crop: Crop) -> crate::Result<()> {
        debug_assert!(matches!(
            dst.fourcc(),
            RGB | RGBA | BGRA | GREY | PLANAR_RGB | RGB_INT8
        ));
        let (width, height) = if dst.is_planar() {
            let width = dst.width() / 4;
            let height = match dst.fourcc() {
                RGBA => dst.height() * 4,
                RGB => dst.height() * 3,
                GREY => dst.height(),
                _ => unreachable!(),
            };
            (width as i32, height as i32)
        } else {
            (dst.width() as i32, dst.height() as i32)
        };

        let format = if dst.is_planar() {
            gls::gl::RED
        } else {
            match dst.fourcc() {
                RGB | RGB_INT8 => gls::gl::RGB,
                RGBA => gls::gl::RGBA,
                BGRA => 0x80E1, // GL_BGRA (GL_EXT_texture_format_BGRA8888)
                GREY => gls::gl::RED,
                _ => unreachable!(),
            }
        };

        let start = Instant::now();
        self.convert_fbo.bind();

        let map;

        let pixels = if crop.dst_rect.is_none_or(|crop| {
            crop.top == 0
                && crop.left == 0
                && crop.height == dst.height()
                && crop.width == dst.width()
        }) {
            std::ptr::null()
        } else {
            map = dst.tensor().map()?;
            map.as_ptr() as *const c_void
        };
        unsafe {
            gls::gl::UseProgram(self.texture_program.id);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.render_texture.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );

            gls::gl::TexImage2D(
                gls::gl::TEXTURE_2D,
                0,
                format as i32,
                width,
                height,
                0,
                format,
                gls::gl::UNSIGNED_BYTE,
                pixels,
            );
            check_gl_error(function!(), line!())?;
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                self.render_texture.id,
                0,
            );
            check_gl_error(function!(), line!())?;
            gls::gl::Viewport(0, 0, width, height);
        }
        log::debug!("Set up framebuffer takes {:?}", start.elapsed());
        Ok(())
    }

    fn convert_dest_non_dma(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        self.setup_renderbuffer_non_dma(dst, crop)?;
        let start = Instant::now();
        if dst.is_planar() {
            self.convert_to_planar(src, dst, rotation, flip, crop)?;
        } else {
            self.convert_to(src, dst, rotation, flip, crop)?;
        }
        log::debug!("Draw to framebuffer takes {:?}", start.elapsed());
        let start = Instant::now();
        let dest_format = match dst.fourcc() {
            RGB | RGB_INT8 => gls::gl::RGB,
            RGBA => gls::gl::RGBA,
            BGRA => 0x80E1, // GL_BGRA (GL_EXT_texture_format_BGRA8888)
            GREY => gls::gl::RED,
            _ => unreachable!(),
        };

        unsafe {
            let mut dst_map = dst.tensor().map()?;
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                dst.width() as i32,
                dst.height() as i32,
                dest_format,
                gls::gl::UNSIGNED_BYTE,
                dst.tensor.len() as i32,
                dst_map.as_mut_ptr() as *mut c_void,
            );
            // Apply XOR 0x80 for int8 formats (convert uint8 → int8 representation)
            if fourcc_is_int8(dst.fourcc()) {
                for byte in dst_map.iter_mut() {
                    *byte ^= 0x80;
                }
            }
        }
        log::debug!("Read from framebuffer takes {:?}", start.elapsed());
        Ok(())
    }

    /// Convert between two PBO-backed images.
    ///
    /// Source PBO is bound as `GL_PIXEL_UNPACK_BUFFER` for zero-copy texture upload
    /// (avoids `tensor.map()` to prevent GL-thread deadlocks). Destination uses
    /// `GL_PIXEL_PACK_BUFFER` for zero-copy readback into the PBO.
    fn convert_pbo_to_pbo(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        // Safety check: neither PBO must be mapped; extract buffer IDs before releasing borrows
        let (src_buffer_id, dst_buffer_id) = {
            let src_pbo = match &src.tensor {
                edgefirst_tensor::Tensor::Pbo(p) => p,
                _ => {
                    return Err(crate::Error::OpenGl(
                        "convert_pbo_to_pbo: src is not a PBO tensor".to_string(),
                    ))
                }
            };
            let dst_pbo = match &dst.tensor {
                edgefirst_tensor::Tensor::Pbo(p) => p,
                _ => {
                    return Err(crate::Error::OpenGl(
                        "convert_pbo_to_pbo: dst is not a PBO tensor".to_string(),
                    ))
                }
            };

            if src_pbo.is_mapped() || dst_pbo.is_mapped() {
                return Err(crate::Error::OpenGl(
                    "Cannot convert PBO tensors while they are mapped".to_string(),
                ));
            }

            (src_pbo.buffer_id(), dst_pbo.buffer_id())
        };

        // Setup renderbuffer (same as non-DMA path)
        self.setup_renderbuffer_non_dma(dst, crop)?;

        // Upload source from PBO and render.
        // We cannot call convert_to/draw_src_texture directly because they
        // call src.tensor().map() which sends a message back to THIS thread,
        // causing a deadlock. Instead, bind the source PBO as UNPACK buffer
        // and upload to the texture with a NULL pointer — GL reads directly
        // from the PBO, zero CPU copy.
        let start = Instant::now();
        self.draw_src_texture_from_pbo(src, src_buffer_id, dst, rotation, flip, crop)?;
        log::debug!("PBO render takes {:?}", start.elapsed());

        // Readback into destination PBO instead of CPU memory
        let start_read = Instant::now();
        let dest_format = match dst.fourcc() {
            crate::RGB | crate::RGB_INT8 => gls::gl::RGB,
            crate::RGBA => gls::gl::RGBA,
            crate::GREY => gls::gl::RED,
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "PBO readback not supported for {}",
                    dst.fourcc().display()
                )))
            }
        };

        unsafe {
            // Bind destination PBO as PACK buffer — glReadnPixels will write into it
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, dst_buffer_id);
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                dst.width() as i32,
                dst.height() as i32,
                dest_format,
                gls::gl::UNSIGNED_BYTE,
                dst.tensor.len() as i32,
                std::ptr::null_mut(), // NULL pointer = write to bound PACK buffer
            );
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
            gls::gl::Finish();
        }

        check_gl_error(function!(), line!())?;

        // Handle int8 XOR if needed (must map PBO to do this on the GL thread
        // directly, since we're already on the GL thread)
        if fourcc_is_int8(dst.fourcc()) {
            unsafe {
                gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, dst_buffer_id);
                let ptr = gls::gl::MapBufferRange(
                    gls::gl::PIXEL_PACK_BUFFER,
                    0,
                    dst.tensor.len() as isize,
                    gls::gl::MAP_READ_BIT | gls::gl::MAP_WRITE_BIT,
                );
                if !ptr.is_null() {
                    let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, dst.tensor.len());
                    for byte in slice.iter_mut() {
                        *byte ^= 0x80;
                    }
                    gls::gl::UnmapBuffer(gls::gl::PIXEL_PACK_BUFFER);
                }
                gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
            }
            check_gl_error(function!(), line!())?;
        }

        log::debug!("PBO readback takes {:?}", start_read.elapsed());
        Ok(())
    }

    /// Upload source image from a PBO and render to the current framebuffer.
    /// This is the PBO equivalent of draw_src_texture — instead of mapping
    /// the tensor to CPU and calling glTexImage2D with a data pointer, we
    /// bind the source PBO as GL_PIXEL_UNPACK_BUFFER and pass NULL, causing
    /// GL to read directly from the PBO (zero CPU copy).
    fn draw_src_texture_from_pbo(
        &mut self,
        src: &TensorImage,
        src_buffer_id: u32,
        dst: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<(), Error> {
        let texture_target = gls::gl::TEXTURE_2D;
        let texture_format = match src.fourcc() {
            crate::RGB | crate::RGB_INT8 => gls::gl::RGB,
            crate::RGBA => gls::gl::RGBA,
            crate::GREY => gls::gl::RED,
            _ => {
                return Err(Error::NotSupported(format!(
                    "PBO upload not supported for {:?}",
                    src.fourcc()
                )));
            }
        };

        let has_crop = crop.dst_rect.is_some_and(|x| {
            x.left != 0 || x.top != 0 || x.width != dst.width() || x.height != dst.height()
        });

        // top and bottom are flipped because OpenGL uses 0,0 as bottom left
        let src_roi = if let Some(crop) = crop.src_rect {
            RegionOfInterest {
                left: crop.left as f32 / src.width() as f32,
                top: (crop.top + crop.height) as f32 / src.height() as f32,
                right: (crop.left + crop.width) as f32 / src.width() as f32,
                bottom: crop.top as f32 / src.height() as f32,
            }
        } else {
            RegionOfInterest {
                left: 0.,
                top: 1.,
                right: 1.,
                bottom: 0.,
            }
        };

        let cvt_screen_coord = |normalized| normalized * 2.0 - 1.0;
        let mut dst_roi = if let Some(crop) = crop.dst_rect {
            RegionOfInterest {
                left: cvt_screen_coord(crop.left as f32 / dst.width() as f32),
                top: cvt_screen_coord((crop.top + crop.height) as f32 / dst.height() as f32),
                right: cvt_screen_coord((crop.left + crop.width) as f32 / dst.width() as f32),
                bottom: cvt_screen_coord(crop.top as f32 / dst.height() as f32),
            }
        } else {
            RegionOfInterest {
                left: -1.,
                top: 1.,
                right: 1.,
                bottom: -1.,
            }
        };

        let rotation_offset = match rotation {
            crate::Rotation::None => 0,
            crate::Rotation::Clockwise90 => 1,
            crate::Rotation::Rotate180 => 2,
            crate::Rotation::CounterClockwise90 => 3,
        };

        unsafe {
            if has_crop {
                if let Some(dst_color) = crop.dst_color {
                    gls::gl::ClearColor(
                        dst_color[0] as f32 / 255.0,
                        dst_color[1] as f32 / 255.0,
                        dst_color[2] as f32 / 255.0,
                        dst_color[3] as f32 / 255.0,
                    );
                    gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
                }
            }

            gls::gl::UseProgram(self.texture_program.id);
            gls::gl::BindTexture(texture_target, self.camera_normal_texture.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );
            if src.fourcc() == crate::GREY {
                for swizzle in [
                    gls::gl::TEXTURE_SWIZZLE_R,
                    gls::gl::TEXTURE_SWIZZLE_G,
                    gls::gl::TEXTURE_SWIZZLE_B,
                ] {
                    gls::gl::TexParameteri(gls::gl::TEXTURE_2D, swizzle, gls::gl::RED as i32);
                }
            } else {
                for (swizzle, src_component) in [
                    (gls::gl::TEXTURE_SWIZZLE_R, gls::gl::RED),
                    (gls::gl::TEXTURE_SWIZZLE_G, gls::gl::GREEN),
                    (gls::gl::TEXTURE_SWIZZLE_B, gls::gl::BLUE),
                ] {
                    gls::gl::TexParameteri(gls::gl::TEXTURE_2D, swizzle, src_component as i32);
                }
            }

            // Bind source PBO as UNPACK buffer — glTexImage2D reads from it
            gls::gl::BindBuffer(gls::gl::PIXEL_UNPACK_BUFFER, src_buffer_id);
            gls::gl::TexImage2D(
                texture_target,
                0,
                texture_format as i32,
                src.width() as i32,
                src.height() as i32,
                0,
                texture_format,
                gls::gl::UNSIGNED_BYTE,
                std::ptr::null(), // NULL = read from bound UNPACK buffer
            );
            gls::gl::BindBuffer(gls::gl::PIXEL_UNPACK_BUFFER, 0);

            // Force texture cache state to be rebuilt next call
            self.camera_normal_texture.width = 0;

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            match flip {
                crate::Flip::None => {}
                crate::Flip::Vertical => {
                    std::mem::swap(&mut dst_roi.top, &mut dst_roi.bottom);
                }
                crate::Flip::Horizontal => {
                    std::mem::swap(&mut dst_roi.left, &mut dst_roi.right);
                }
            }

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0., // left top
                dst_roi.right,
                dst_roi.top,
                0., // right top
                dst_roi.right,
                dst_roi.bottom,
                0., // right bottom
                dst_roi.left,
                dst_roi.bottom,
                0., // left bottom
            ];
            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                (camera_vertices.len() * std::mem::size_of::<f32>()) as isize,
                camera_vertices.as_ptr() as *const c_void,
                gls::gl::STATIC_DRAW,
            );
            gls::gl::VertexAttribPointer(
                self.vertex_buffer.buffer_index,
                3,
                gls::gl::FLOAT,
                gls::gl::FALSE,
                0,
                std::ptr::null(),
            );

            let texture_coords: [[f32; 8]; 4] = [
                [
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                ],
                [
                    src_roi.left,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                ],
                [
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                ],
                [
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.top,
                ],
            ];
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
            gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                (texture_coords[0].len() * std::mem::size_of::<f32>()) as isize,
                texture_coords[rotation_offset].as_ptr() as *const c_void,
                gls::gl::STATIC_DRAW,
            );
            gls::gl::VertexAttribPointer(
                self.texture_buffer.buffer_index,
                2,
                gls::gl::FLOAT,
                gls::gl::FALSE,
                0,
                std::ptr::null(),
            );
            gls::gl::DrawArrays(gls::gl::TRIANGLE_FAN, 0, 4);
            gls::gl::DisableVertexAttribArray(self.vertex_buffer.buffer_index);
            gls::gl::DisableVertexAttribArray(self.texture_buffer.buffer_index);

            gls::gl::Finish();
        }

        check_gl_error(function!(), line!())?;
        Ok(())
    }

    /// Convert any source (Mem/DMA) to a PBO destination.
    /// Source is uploaded via normal texture path (maps tensor for CPU upload).
    /// Destination readback uses PBO PACK binding (no map on GL thread).
    fn convert_any_to_pbo(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let dst_buffer_id = match &dst.tensor {
            edgefirst_tensor::Tensor::Pbo(p) => {
                if p.is_mapped() {
                    return Err(crate::Error::OpenGl(
                        "Cannot convert to a mapped PBO tensor".to_string(),
                    ));
                }
                p.buffer_id()
            }
            _ => {
                return Err(crate::Error::OpenGl(
                    "convert_any_to_pbo: dst is not a PBO tensor".to_string(),
                ))
            }
        };

        self.setup_renderbuffer_non_dma(dst, crop)?;
        let start = Instant::now();
        if dst.is_planar() {
            self.convert_to_planar(src, dst, rotation, flip, crop)?;
        } else {
            self.convert_to(src, dst, rotation, flip, crop)?;
        }
        log::debug!("any-to-PBO render takes {:?}", start.elapsed());

        // PBO readback
        let start_read = Instant::now();
        let dest_format = match dst.fourcc() {
            crate::RGB | crate::RGB_INT8 => gls::gl::RGB,
            crate::RGBA => gls::gl::RGBA,
            crate::GREY => gls::gl::RED,
            _ => {
                return Err(crate::Error::NotSupported(format!(
                    "PBO readback not supported for {}",
                    dst.fourcc().display()
                )))
            }
        };
        unsafe {
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, dst_buffer_id);
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                dst.width() as i32,
                dst.height() as i32,
                dest_format,
                gls::gl::UNSIGNED_BYTE,
                dst.tensor.len() as i32,
                std::ptr::null_mut(),
            );
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
            gls::gl::Finish();
        }
        check_gl_error(function!(), line!())?;

        if fourcc_is_int8(dst.fourcc()) {
            unsafe {
                gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, dst_buffer_id);
                let ptr = gls::gl::MapBufferRange(
                    gls::gl::PIXEL_PACK_BUFFER,
                    0,
                    dst.tensor.len() as isize,
                    gls::gl::MAP_READ_BIT | gls::gl::MAP_WRITE_BIT,
                );
                if !ptr.is_null() {
                    let slice = std::slice::from_raw_parts_mut(ptr as *mut u8, dst.tensor.len());
                    for byte in slice.iter_mut() {
                        *byte ^= 0x80;
                    }
                    gls::gl::UnmapBuffer(gls::gl::PIXEL_PACK_BUFFER);
                }
                gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
            }
            check_gl_error(function!(), line!())?;
        }

        log::debug!("any-to-PBO readback takes {:?}", start_read.elapsed());
        Ok(())
    }

    /// Convert a PBO source to a non-PBO (Mem) destination.
    /// Source is uploaded via PBO UNPACK binding (no map on GL thread).
    /// Destination readback uses normal ReadnPixels into mapped Mem tensor.
    fn convert_pbo_to_mem(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let src_buffer_id = match &src.tensor {
            edgefirst_tensor::Tensor::Pbo(p) => {
                if p.is_mapped() {
                    return Err(crate::Error::OpenGl(
                        "Cannot convert from a mapped PBO tensor".to_string(),
                    ));
                }
                p.buffer_id()
            }
            _ => {
                return Err(crate::Error::OpenGl(
                    "convert_pbo_to_mem: src is not a PBO tensor".to_string(),
                ))
            }
        };

        self.setup_renderbuffer_non_dma(dst, crop)?;
        let start = Instant::now();
        self.draw_src_texture_from_pbo(src, src_buffer_id, dst, rotation, flip, crop)?;
        log::debug!("PBO-to-mem render takes {:?}", start.elapsed());

        // Normal readback into Mem dst
        let start = Instant::now();
        let dest_format = match dst.fourcc() {
            crate::RGB | crate::RGB_INT8 => gls::gl::RGB,
            crate::RGBA => gls::gl::RGBA,
            crate::GREY => gls::gl::RED,
            _ => unreachable!(),
        };
        unsafe {
            let mut dst_map = dst.tensor().map()?;
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                dst.width() as i32,
                dst.height() as i32,
                dest_format,
                gls::gl::UNSIGNED_BYTE,
                dst.tensor.len() as i32,
                dst_map.as_mut_ptr() as *mut c_void,
            );
            if fourcc_is_int8(dst.fourcc()) {
                for byte in dst_map.iter_mut() {
                    *byte ^= 0x80;
                }
            }
        }
        log::debug!("PBO-to-mem readback takes {:?}", start.elapsed());
        Ok(())
    }

    fn convert_to(
        &mut self,
        src: &TensorImage,
        dst: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<(), crate::Error> {
        check_gl_error(function!(), line!())?;

        let has_crop = crop.dst_rect.is_some_and(|x| {
            x.left != 0 || x.top != 0 || x.width != dst.width() || x.height != dst.height()
        });
        if has_crop {
            if let Some(dst_color) = crop.dst_color {
                unsafe {
                    gls::gl::ClearColor(
                        dst_color[0] as f32 / 255.0,
                        dst_color[1] as f32 / 255.0,
                        dst_color[2] as f32 / 255.0,
                        dst_color[3] as f32 / 255.0,
                    );
                    gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
                };
            }
        }

        // top and bottom are flipped because OpenGL uses 0,0 as bottom left
        let src_roi = if let Some(crop) = crop.src_rect {
            RegionOfInterest {
                left: crop.left as f32 / src.width() as f32,
                top: (crop.top + crop.height) as f32 / src.height() as f32,
                right: (crop.left + crop.width) as f32 / src.width() as f32,
                bottom: crop.top as f32 / src.height() as f32,
            }
        } else {
            RegionOfInterest {
                left: 0.,
                top: 1.,
                right: 1.,
                bottom: 0.,
            }
        };

        // top and bottom are flipped because OpenGL uses 0,0 as bottom left
        let cvt_screen_coord = |normalized| normalized * 2.0 - 1.0;
        let dst_roi = if let Some(crop) = crop.dst_rect {
            RegionOfInterest {
                left: cvt_screen_coord(crop.left as f32 / dst.width() as f32),
                top: cvt_screen_coord((crop.top + crop.height) as f32 / dst.height() as f32),
                right: cvt_screen_coord((crop.left + crop.width) as f32 / dst.width() as f32),
                bottom: cvt_screen_coord(crop.top as f32 / dst.height() as f32),
            }
        } else {
            RegionOfInterest {
                left: -1.,
                top: 1.,
                right: 1.,
                bottom: -1.,
            }
        };
        let rotation_offset = match rotation {
            crate::Rotation::None => 0,
            crate::Rotation::Clockwise90 => 1,
            crate::Rotation::Rotate180 => 2,
            crate::Rotation::CounterClockwise90 => 3,
        };
        if self.gl_context.transfer_backend.is_dma() && src.tensor().memory() == TensorMemory::Dma {
            match self.get_or_create_egl_image(CacheKind::Src, src) {
                Ok(src_egl) => self.draw_camera_texture_eglimage(
                    src,
                    src_egl,
                    src_roi,
                    dst_roi,
                    rotation_offset,
                    flip,
                )?,
                Err(e) => {
                    log::warn!("EGL image creation failed for {:?}: {:?}", src.fourcc(), e);
                    let start = Instant::now();
                    self.draw_src_texture(src, src_roi, dst_roi, rotation_offset, flip)?;
                    log::debug!("draw_src_texture takes {:?}", start.elapsed());
                }
            }
        } else {
            let start = Instant::now();
            self.draw_src_texture(src, src_roi, dst_roi, rotation_offset, flip)?;
            log::debug!("draw_src_texture takes {:?}", start.elapsed());
        }

        let start = Instant::now();
        unsafe { gls::gl::Finish() };
        log::debug!("gl_Finish takes {:?}", start.elapsed());
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    fn convert_to_planar(
        &mut self,
        src: &TensorImage,
        dst: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<(), crate::Error> {
        // if let Some(crop) = crop.src_rect
        //     && (crop.left > 0
        //         || crop.top > 0
        //         || crop.height < src.height()
        //         || crop.width < src.width())
        // {
        //     return Err(crate::Error::NotSupported(
        //         "Cropping in planar RGB mode is not supported".to_string(),
        //     ));
        // }

        // if let Some(crop) = crop.dst_rect
        //     && (crop.left > 0
        //         || crop.top > 0
        //         || crop.height < src.height()
        //         || crop.width < src.width())
        // {
        //     return Err(crate::Error::NotSupported(
        //         "Cropping in planar RGB mode is not supported".to_string(),
        //     ));
        // }

        let alpha = match dst.fourcc() {
            PLANAR_RGB | PLANAR_RGB_INT8 => false,
            PLANAR_RGBA => true,
            _ => {
                return Err(crate::Error::NotSupported(
                    "Destination format must be PLANAR_RGB, PLANAR_RGB_INT8, or PLANAR_RGBA"
                        .to_string(),
                ));
            }
        };
        let is_int8 = fourcc_is_int8(dst.fourcc());

        // top and bottom are flipped because OpenGL uses 0,0 as bottom left
        let src_roi = if let Some(crop) = crop.src_rect {
            RegionOfInterest {
                left: crop.left as f32 / src.width() as f32,
                top: (crop.top + crop.height) as f32 / src.height() as f32,
                right: (crop.left + crop.width) as f32 / src.width() as f32,
                bottom: crop.top as f32 / src.height() as f32,
            }
        } else {
            RegionOfInterest {
                left: 0.,
                top: 1.,
                right: 1.,
                bottom: 0.,
            }
        };

        // top and bottom are flipped because OpenGL uses 0,0 as bottom left
        let cvt_screen_coord = |normalized| normalized * 2.0 - 1.0;
        let dst_roi = if let Some(crop) = crop.dst_rect {
            RegionOfInterest {
                left: cvt_screen_coord(crop.left as f32 / dst.width() as f32),
                top: cvt_screen_coord((crop.top + crop.height) as f32 / dst.height() as f32),
                right: cvt_screen_coord((crop.left + crop.width) as f32 / dst.width() as f32),
                bottom: cvt_screen_coord(crop.top as f32 / dst.height() as f32),
            }
        } else {
            RegionOfInterest {
                left: -1.,
                top: 1.,
                right: 1.,
                bottom: -1.,
            }
        };
        let rotation_offset = match rotation {
            crate::Rotation::None => 0,
            crate::Rotation::Clockwise90 => 1,
            crate::Rotation::Rotate180 => 2,
            crate::Rotation::CounterClockwise90 => 3,
        };

        let has_crop = crop.dst_rect.is_some_and(|x| {
            x.left != 0 || x.top != 0 || x.width != dst.width() || x.height != dst.height()
        });
        if has_crop {
            if let Some(dst_color) = crop.dst_color {
                self.clear_rect_planar(
                    dst.width(),
                    dst.height(),
                    dst_roi,
                    [
                        dst_color[0] as f32 / 255.0,
                        dst_color[1] as f32 / 255.0,
                        dst_color[2] as f32 / 255.0,
                        dst_color[3] as f32 / 255.0,
                    ],
                    alpha,
                )?;
            }
        }

        let src_egl = self.get_or_create_egl_image(CacheKind::Src, src)?;

        self.draw_camera_texture_to_rgb_planar(
            src_egl,
            src_roi,
            dst_roi,
            rotation_offset,
            flip,
            alpha,
            is_int8,
        )?;
        unsafe { gls::gl::Finish() };

        Ok(())
    }

    /// Render packed RGB (or RGB_INT8) to a DMA destination buffer using a
    /// two-pass architecture:
    ///
    /// **Pass 1:** Render source → intermediate RGBA texture via `convert_to()`
    /// (reuses the battle-tested RGBA path with full crop/letterbox/rotation/flip).
    ///
    /// **Pass 2:** Pack intermediate RGBA → RGB DMA destination using a simple
    /// packing shader with 2D sampler. The destination DMA buffer is reinterpreted
    /// as RGBA8 at (W*3/4) x H dimensions.
    fn convert_to_packed_rgb(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let dst_w = dst.width();
        let dst_h = dst.height();
        let is_int8 = fourcc_is_int8(dst.fourcc());

        // Width must satisfy PackedRgba8 constraint: W*3 divisible by 4
        if !(dst_w * 3).is_multiple_of(4) {
            return Err(crate::Error::NotSupported(format!(
                "Packed RGB requires width*3 divisible by 4, got width={dst_w}"
            )));
        }

        let render_w = dst_w * 3 / 4;
        let render_h = dst_h;

        log::debug!(
            "convert_to_packed_rgb: {dst_w}x{dst_h} -> {render_w}x{render_h} two-pass int8={is_int8}",
        );

        // --- Pass 1: Render source → intermediate RGBA texture ---
        self.ensure_packed_rgb_intermediate(dst_w, dst_h)?;
        self.packed_rgb_fbo.bind();
        unsafe {
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                self.packed_rgb_intermediate_tex.id,
                0,
            );
            check_gl_error(function!(), line!())?;
            gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
        }
        // convert_to() renders to the currently-bound FBO (packed_rgb_fbo → intermediate).
        // It uses dst only for width/height in ROI coordinate math.
        // Handles: source binding (DMA EGLImage or upload), crop, letterbox, rotation, flip.
        self.convert_to(src, dst, rotation, flip, crop)?;

        // --- Pass 2: Pack intermediate RGBA → RGB DMA destination ---
        self.convert_fbo.bind();
        let dest_egl =
            self.get_or_create_egl_image_rgb(dst, render_w, render_h, DrmFourcc::Abgr8888, 4)?;
        unsafe {
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.render_texture.id);
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::NEAREST as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::NEAREST as i32,
            );
            gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, dest_egl.as_ptr());
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                self.render_texture.id,
                0,
            );
            check_gl_error(function!(), line!())?;
            gls::gl::Viewport(0, 0, render_w as i32, render_h as i32);
        }

        // Bind intermediate RGBA texture as source for the packing shader
        let program = if is_int8 {
            &self.packed_rgba8_int8_program_2d
        } else {
            &self.packed_rgba8_program_2d
        };
        unsafe {
            gls::gl::UseProgram(program.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE1);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.packed_rgb_intermediate_tex.id);
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::NEAREST as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::NEAREST as i32,
            );
        }

        // Set uniform: tex = TEXTURE1 (intermediate RGBA texture)
        unsafe {
            let loc_tex = gls::gl::GetUniformLocation(program.id, c"tex".as_ptr());
            gls::gl::Uniform1i(loc_tex, 1);
        }

        // Draw full-viewport quad to pack RGBA→RGB
        self.draw_fullscreen_quad()?;

        unsafe { gls::gl::Finish() };
        Ok(())
    }

    /// Render directly to an RGB8 renderbuffer backed by BGR888 DMA-buf.
    /// Single-pass: no intermediate texture, no packing shader.
    fn convert_to_rgb_direct(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let is_int8 = fourcc_is_int8(dst.fourcc());

        log::debug!(
            "convert_to_rgb_direct: {}x{} single-pass int8={is_int8}",
            dst.width(),
            dst.height(),
        );

        // Get or create cached renderbuffer
        let (rbo, width, height) = self.get_or_create_rgb_direct_rbo(dst)?;

        // Bind FBO with renderbuffer attachment
        self.convert_fbo.bind();
        unsafe {
            gls::gl::FramebufferRenderbuffer(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::RENDERBUFFER,
                rbo,
            );
            check_gl_error(function!(), line!())?;

            let status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
            if status != gls::gl::FRAMEBUFFER_COMPLETE {
                log::warn!("convert_to_rgb_direct: FBO incomplete (0x{status:x}), falling back");
                return self.convert_to_packed_rgb(src, dst, rotation, flip, crop);
            }

            gls::gl::Viewport(0, 0, width, height);
        }

        // For int8, temporarily swap to int8 shader programs and bias the clear color
        let crop = if is_int8 {
            std::mem::swap(&mut self.texture_program, &mut self.texture_int8_program);
            std::mem::swap(
                &mut self.texture_program_yuv,
                &mut self.texture_int8_program_yuv,
            );
            // Bias the letterbox clear color with XOR 0x80 since glClear bypasses
            // the fragment shader — the int8 bias must be applied to the color directly.
            let mut crop = crop;
            if let Some(ref mut color) = crop.dst_color {
                color[0] ^= 0x80;
                color[1] ^= 0x80;
                color[2] ^= 0x80;
            }
            crop
        } else {
            crop
        };

        let result = self.convert_to(src, dst, rotation, flip, crop);

        // Swap back
        if is_int8 {
            std::mem::swap(&mut self.texture_program, &mut self.texture_int8_program);
            std::mem::swap(
                &mut self.texture_program_yuv,
                &mut self.texture_int8_program_yuv,
            );
        }

        result
    }

    /// Allocates or resizes the intermediate RGBA texture for two-pass packed RGB.
    fn ensure_packed_rgb_intermediate(&mut self, width: usize, height: usize) -> crate::Result<()> {
        if self.packed_rgb_intermediate_size == (width, height) {
            return Ok(());
        }
        unsafe {
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.packed_rgb_intermediate_tex.id);
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::NEAREST as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::NEAREST as i32,
            );
            gls::gl::TexImage2D(
                gls::gl::TEXTURE_2D,
                0,
                gls::gl::RGBA as i32,
                width as i32,
                height as i32,
                0,
                gls::gl::RGBA,
                gls::gl::UNSIGNED_BYTE,
                std::ptr::null(),
            );
            check_gl_error(function!(), line!())?;
        }
        self.packed_rgb_intermediate_size = (width, height);
        Ok(())
    }

    /// Draw a fullscreen quad for the currently-bound shader program.
    /// Used by the pass-2 packing shader in the two-pass packed RGB pipeline.
    fn draw_fullscreen_quad(&self) -> Result<(), crate::Error> {
        unsafe {
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            let vertices: [f32; 12] = [
                -1.0, 1.0, 0.0, // top-left
                1.0, 1.0, 0.0, // top-right
                1.0, -1.0, 0.0, // bottom-right
                -1.0, -1.0, 0.0, // bottom-left
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * vertices.len()) as isize,
                vertices.as_ptr() as *const c_void,
            );

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
            gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

            // Texture coordinates (the packed shader uses gl_FragCoord, not tc,
            // but we still need valid buffers for the vertex attribute layout)
            let tex_coords: [f32; 8] = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * tex_coords.len()) as isize,
                tex_coords.as_ptr() as *const c_void,
            );

            let indices: [u32; 4] = [0, 1, 2, 3];
            gls::gl::DrawElements(
                gls::gl::TRIANGLE_FAN,
                indices.len() as i32,
                gls::gl::UNSIGNED_INT,
                indices.as_ptr() as *const c_void,
            );
        }
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    fn clear_rect_planar(
        &self,
        width: usize,
        height: usize,
        dst_roi: RegionOfInterest,
        color: [f32; 4],
        alpha: bool,
    ) -> Result<(), Error> {
        if !alpha && color[0] == color[1] && color[1] == color[2] {
            unsafe {
                gls::gl::ClearColor(color[0], color[0], color[0], 1.0);
                gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
            };
        }

        let split = if alpha { 4 } else { 3 };

        unsafe {
            gls::gl::Enable(gls::gl::SCISSOR_TEST);
            let x = (((dst_roi.left + 1.0) / 2.0) * width as f32).round() as i32;
            let y = (((dst_roi.bottom + 1.0) / 2.0) * height as f32).round() as i32;
            let width = (((dst_roi.right - dst_roi.left) / 2.0) * width as f32).round() as i32;
            let height = (((dst_roi.top - dst_roi.bottom) / 2.0) * height as f32 / split as f32)
                .round() as i32;
            for (i, c) in color.iter().enumerate().take(split) {
                gls::gl::Scissor(x, y + i as i32 * height, width, height);
                gls::gl::ClearColor(*c, *c, *c, 1.0);
                gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
            }
            gls::gl::Disable(gls::gl::SCISSOR_TEST);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_camera_texture_to_rgb_planar(
        &self,
        egl_img: egl::Image,
        src_roi: RegionOfInterest,
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
        alpha: bool,
        int8: bool,
    ) -> Result<(), Error> {
        let texture_target = gls::gl::TEXTURE_EXTERNAL_OES;
        match flip {
            Flip::None => {}
            Flip::Vertical => {
                std::mem::swap(&mut dst_roi.top, &mut dst_roi.bottom);
            }
            Flip::Horizontal => {
                std::mem::swap(&mut dst_roi.left, &mut dst_roi.right);
            }
        }
        unsafe {
            let program = if int8 {
                &self.texture_program_planar_int8
            } else {
                &self.texture_program_planar
            };
            gls::gl::UseProgram(program.id);
            gls::gl::BindTexture(texture_target, self.camera_eglimage_texture.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_WRAP_S,
                gls::gl::CLAMP_TO_EDGE as i32,
            );

            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_WRAP_T,
                gls::gl::CLAMP_TO_EDGE as i32,
            );

            gls::egl_image_target_texture_2d_oes(texture_target, egl_img.as_ptr());
            check_gl_error(function!(), line!())?;
            let y_centers = if alpha {
                vec![-3.0 / 4.0, -1.0 / 4.0, 1.0 / 4.0, 3.0 / 4.0]
            } else {
                vec![-2.0 / 3.0, 0.0, 2.0 / 3.0]
            };
            let swizzles = [gls::gl::RED, gls::gl::GREEN, gls::gl::BLUE, gls::gl::ALPHA];
            // starts from bottom
            for (i, y_center) in y_centers.iter().enumerate() {
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
                let camera_vertices: [f32; 12] = [
                    dst_roi.left,
                    dst_roi.top / 3.0 + y_center,
                    0., // left top
                    dst_roi.right,
                    dst_roi.top / 3.0 + y_center,
                    0., // right top
                    dst_roi.right,
                    dst_roi.bottom / 3.0 + y_center,
                    0., // right bottom
                    dst_roi.left,
                    dst_roi.bottom / 3.0 + y_center,
                    0., // left bottom
                ];
                gls::gl::BufferData(
                    gls::gl::ARRAY_BUFFER,
                    (size_of::<f32>() * camera_vertices.len()) as isize,
                    camera_vertices.as_ptr() as *const c_void,
                    gls::gl::DYNAMIC_DRAW,
                );

                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
                gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
                let texture_vertices: [f32; 16] = [
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                ];

                gls::gl::BufferData(
                    gls::gl::ARRAY_BUFFER,
                    (size_of::<f32>() * 8) as isize,
                    (texture_vertices[(rotation_offset * 2)..]).as_ptr() as *const c_void,
                    gls::gl::DYNAMIC_DRAW,
                );
                let vertices_index: [u32; 4] = [0, 1, 2, 3];
                // self.texture_program_planar
                //     .load_uniform_1i(c"color_index", 2 - i as i32);

                gls::gl::TexParameteri(
                    texture_target,
                    gls::gl::TEXTURE_SWIZZLE_R,
                    swizzles[i] as i32,
                );

                gls::gl::DrawElements(
                    gls::gl::TRIANGLE_FAN,
                    vertices_index.len() as i32,
                    gls::gl::UNSIGNED_INT,
                    vertices_index.as_ptr() as *const c_void,
                );
            }
            check_gl_error(function!(), line!())?;
        }
        Ok(())
    }

    fn draw_src_texture(
        &mut self,
        src: &TensorImage,
        src_roi: RegionOfInterest,
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
    ) -> Result<(), Error> {
        let texture_target = gls::gl::TEXTURE_2D;
        let texture_format = match src.fourcc() {
            RGB => gls::gl::RGB,
            RGBA => gls::gl::RGBA,
            GREY => gls::gl::RED,
            _ => {
                return Err(Error::NotSupported(format!(
                    "draw_src_texture does not support {:?} (use DMA-BUF path for YUV)",
                    src.fourcc()
                )));
            }
        };
        unsafe {
            gls::gl::UseProgram(self.texture_program.id);
            gls::gl::BindTexture(texture_target, self.camera_normal_texture.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );
            if src.fourcc() == GREY {
                for swizzle in [
                    gls::gl::TEXTURE_SWIZZLE_R,
                    gls::gl::TEXTURE_SWIZZLE_G,
                    gls::gl::TEXTURE_SWIZZLE_B,
                ] {
                    gls::gl::TexParameteri(gls::gl::TEXTURE_2D, swizzle, gls::gl::RED as i32);
                }
            } else {
                for (swizzle, src) in [
                    (gls::gl::TEXTURE_SWIZZLE_R, gls::gl::RED),
                    (gls::gl::TEXTURE_SWIZZLE_G, gls::gl::GREEN),
                    (gls::gl::TEXTURE_SWIZZLE_B, gls::gl::BLUE),
                ] {
                    gls::gl::TexParameteri(gls::gl::TEXTURE_2D, swizzle, src as i32);
                }
            }
            self.camera_normal_texture.update_texture(
                texture_target,
                src.width(),
                src.height(),
                texture_format,
                &src.tensor().map()?,
            );

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            match flip {
                Flip::None => {}
                Flip::Vertical => {
                    std::mem::swap(&mut dst_roi.top, &mut dst_roi.bottom);
                }
                Flip::Horizontal => {
                    std::mem::swap(&mut dst_roi.left, &mut dst_roi.right);
                }
            }

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0., // left top
                dst_roi.right,
                dst_roi.top,
                0., // right top
                dst_roi.right,
                dst_roi.bottom,
                0., // right bottom
                dst_roi.left,
                dst_roi.bottom,
                0., // left bottom
            ];
            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                (size_of::<f32>() * camera_vertices.len()) as isize,
                camera_vertices.as_ptr() as *const c_void,
                gls::gl::DYNAMIC_DRAW,
            );
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
            gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
            let texture_vertices: [f32; 16] = [
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
            ];

            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                (size_of::<f32>() * 8) as isize,
                (texture_vertices[(rotation_offset * 2)..]).as_ptr() as *const c_void,
                gls::gl::DYNAMIC_DRAW,
            );
            let vertices_index: [u32; 4] = [0, 1, 2, 3];
            gls::gl::DrawElements(
                gls::gl::TRIANGLE_FAN,
                vertices_index.len() as i32,
                gls::gl::UNSIGNED_INT,
                vertices_index.as_ptr() as *const c_void,
            );
            check_gl_error(function!(), line!())?;

            Ok(())
        }
    }

    fn draw_camera_texture_eglimage(
        &self,
        src: &TensorImage,
        egl_img: egl::Image,
        src_roi: RegionOfInterest,
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
    ) -> Result<(), Error> {
        // let texture_target = gls::gl::TEXTURE_2D;
        let texture_target = gls::gl::TEXTURE_EXTERNAL_OES;
        unsafe {
            gls::gl::UseProgram(self.texture_program_yuv.id);
            gls::gl::BindTexture(texture_target, self.camera_eglimage_texture.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );

            if src.fourcc() == GREY {
                for swizzle in [
                    gls::gl::TEXTURE_SWIZZLE_R,
                    gls::gl::TEXTURE_SWIZZLE_G,
                    gls::gl::TEXTURE_SWIZZLE_B,
                ] {
                    gls::gl::TexParameteri(gls::gl::TEXTURE_2D, swizzle, gls::gl::RED as i32);
                }
            } else {
                for (swizzle, src) in [
                    (gls::gl::TEXTURE_SWIZZLE_R, gls::gl::RED),
                    (gls::gl::TEXTURE_SWIZZLE_G, gls::gl::GREEN),
                    (gls::gl::TEXTURE_SWIZZLE_B, gls::gl::BLUE),
                ] {
                    gls::gl::TexParameteri(gls::gl::TEXTURE_2D, swizzle, src as i32);
                }
            }

            gls::egl_image_target_texture_2d_oes(texture_target, egl_img.as_ptr());
            check_gl_error(function!(), line!())?;
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            match flip {
                Flip::None => {}
                Flip::Vertical => {
                    std::mem::swap(&mut dst_roi.top, &mut dst_roi.bottom);
                }
                Flip::Horizontal => {
                    std::mem::swap(&mut dst_roi.left, &mut dst_roi.right);
                }
            }

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0., // left top
                dst_roi.right,
                dst_roi.top,
                0., // right top
                dst_roi.right,
                dst_roi.bottom,
                0., // right bottom
                dst_roi.left,
                dst_roi.bottom,
                0., // left bottom
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * camera_vertices.len()) as isize,
                camera_vertices.as_ptr() as *const c_void,
            );

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
            gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

            let texture_vertices: [f32; 16] = [
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * 8) as isize,
                (texture_vertices[(rotation_offset * 2)..]).as_ptr() as *const c_void,
            );

            let vertices_index: [u32; 4] = [0, 1, 2, 3];
            gls::gl::DrawElements(
                gls::gl::TRIANGLE_FAN,
                vertices_index.len() as i32,
                gls::gl::UNSIGNED_INT,
                vertices_index.as_ptr() as *const c_void,
            );
        }
        check_gl_error(function!(), line!())?;
        Ok(())
    }

    fn create_image_from_dma2(&self, src: &TensorImage) -> Result<EglImage, crate::Error> {
        let width;
        let height;
        let format;
        let channels;

        // NV12 is semi-planar but handled specially via EGL multi-plane import
        if src.fourcc() == NV12 {
            if !src.width().is_multiple_of(4) {
                return Err(Error::NotSupported(
                    "OpenGL EGLImage doesn't support image widths which are not multiples of 4"
                        .to_string(),
                ));
            }
            width = src.width();
            height = src.height();
            format = fourcc_to_drm(NV12)?;
            channels = 1; // Y plane pitch is 1 byte per pixel
        } else if src.is_planar() {
            if !src.width().is_multiple_of(16) {
                return Err(Error::NotSupported(
                    "OpenGL Planar RGB EGLImage doesn't support image widths which are not multiples of 16"
                        .to_string(),
                ));
            }
            match src.fourcc() {
                PLANAR_RGB | PLANAR_RGB_INT8 => {
                    format = DrmFourcc::R8;
                    width = src.width();
                    height = src.height() * 3;
                    channels = 1;
                }
                fourcc => {
                    return Err(crate::Error::NotSupported(format!(
                        "Unsupported Planar FourCC {fourcc:?}"
                    )));
                }
            };
        } else {
            if !src.width().is_multiple_of(4) {
                return Err(Error::NotSupported(
                    "OpenGL EGLImage doesn't support image widths which are not multiples of 4"
                        .to_string(),
                ));
            }
            width = src.width();
            height = src.height();
            format = fourcc_to_drm(src.fourcc())?;
            channels = src.channels();
        }

        let fd = match &src.tensor {
            edgefirst_tensor::Tensor::Dma(dma_tensor) => dma_tensor.fd.as_raw_fd(),
            edgefirst_tensor::Tensor::Shm(_) => {
                return Err(Error::NotImplemented(
                    "OpenGL EGLImage doesn't support SHM".to_string(),
                ));
            }
            edgefirst_tensor::Tensor::Mem(_) => {
                return Err(Error::NotImplemented(
                    "OpenGL EGLImage doesn't support MEM".to_string(),
                ));
            }
            edgefirst_tensor::Tensor::Pbo(_) => {
                return Err(Error::NotImplemented(
                    "OpenGL EGLImage doesn't support PBO".to_string(),
                ));
            }
        };

        // For NV12, plane0 pitch is width (Y is 1 byte/pixel)
        // For other formats, pitch is width * channels
        let plane0_pitch = if src.fourcc() == NV12 {
            width
        } else {
            width * channels
        };

        let mut egl_img_attr = vec![
            egl_ext::LINUX_DRM_FOURCC as Attrib,
            format as Attrib,
            khronos_egl::WIDTH as Attrib,
            width as Attrib,
            khronos_egl::HEIGHT as Attrib,
            height as Attrib,
            egl_ext::DMA_BUF_PLANE0_PITCH as Attrib,
            plane0_pitch as Attrib,
            egl_ext::DMA_BUF_PLANE0_OFFSET as Attrib,
            0 as Attrib,
            egl_ext::DMA_BUF_PLANE0_FD as Attrib,
            fd as Attrib,
            egl::IMAGE_PRESERVED as Attrib,
            egl::TRUE as Attrib,
        ];

        // NV12 requires a second plane for UV data
        if src.fourcc() == NV12 {
            let uv_offset = width * height; // Y plane size
            egl_img_attr.append(&mut vec![
                egl_ext::DMA_BUF_PLANE1_FD as Attrib,
                fd as Attrib,
                egl_ext::DMA_BUF_PLANE1_OFFSET as Attrib,
                uv_offset as Attrib,
                egl_ext::DMA_BUF_PLANE1_PITCH as Attrib,
                width as Attrib, // UV plane has same width as Y plane
            ]);
        }

        if matches!(src.fourcc(), YUYV | VYUY | NV12) {
            egl_img_attr.append(&mut vec![
                egl_ext::YUV_COLOR_SPACE_HINT as Attrib,
                egl_ext::ITU_REC709 as Attrib,
                egl_ext::SAMPLE_RANGE_HINT as Attrib,
                egl_ext::YUV_NARROW_RANGE as Attrib,
            ]);
        }

        egl_img_attr.push(khronos_egl::NONE as Attrib);

        match self.new_egl_image_owned(egl_ext::LINUX_DMA_BUF, &egl_img_attr) {
            Ok(v) => Ok(v),
            Err(e) => Err(e),
        }
    }

    fn new_egl_image_owned(
        &'_ self,
        target: egl::Enum,
        attrib_list: &[Attrib],
    ) -> Result<EglImage, Error> {
        let image = GlContext::egl_create_image_with_fallback(
            &self.gl_context.egl,
            self.gl_context.display.as_display(),
            unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) },
            target,
            unsafe { egl::ClientBuffer::from_ptr(null_mut()) },
            attrib_list,
        )?;
        Ok(EglImage {
            egl_image: image,
            display: self.gl_context.display.as_display(),
            egl: Rc::clone(&self.gl_context.egl),
        })
    }

    /// Look up or create an EGLImage for a DMA tensor, returning the EGL image handle.
    ///
    /// Returns `egl::Image` (a `Copy` type wrapping `*const c_void`) to avoid borrow
    /// conflicts with the caller. The cache retains ownership of the `EglImage` value;
    /// the handle remains valid as long as the entry lives in the cache.
    fn get_or_create_egl_image(
        &mut self,
        cache: CacheKind,
        img: &TensorImage,
    ) -> Result<egl::Image, crate::Error> {
        let id = img.buffer_identity().id();

        // Sweep dead entries opportunistically before looking up.
        match cache {
            CacheKind::Src => self.src_egl_cache.sweep(),
            CacheKind::Dst => self.dst_egl_cache.sweep(),
        }

        {
            let egl_cache = match cache {
                CacheKind::Src => &mut self.src_egl_cache,
                CacheKind::Dst => &mut self.dst_egl_cache,
            };
            let ts = egl_cache.next_timestamp();
            if let Some(cached) = egl_cache.entries.get_mut(&id) {
                egl_cache.hits += 1;
                cached.last_used = ts;
                log::trace!("EglImageCache {:?} hit: id={id:#x}", cache);
                return Ok(cached.egl_image.egl_image);
            }
            egl_cache.misses += 1;
            log::trace!("EglImageCache {:?} miss: id={id:#x}", cache);
            // Evict least-recently-used entry if at capacity.
            if egl_cache.entries.len() >= egl_cache.capacity {
                egl_cache.evict_lru();
            }
        }

        let egl_image_obj = self.create_image_from_dma2(img)?;
        let handle = egl_image_obj.egl_image;
        let guard = img.buffer_identity().weak();
        let egl_cache = match cache {
            CacheKind::Src => &mut self.src_egl_cache,
            CacheKind::Dst => &mut self.dst_egl_cache,
        };
        let ts = egl_cache.next_timestamp();
        egl_cache.entries.insert(
            id,
            CachedEglImage {
                egl_image: egl_image_obj,
                guard,
                renderbuffer: None,
                last_used: ts,
            },
        );
        Ok(handle)
    }

    /// Create an EGLImage from a DMA buffer with explicitly specified internal
    /// dimensions and format. Used when the GL render surface differs from the
    /// logical TensorImage dimensions (e.g., packed RGB reinterpretation).
    fn create_egl_image_with_dims(
        &self,
        img: &TensorImage,
        width: usize,
        height: usize,
        drm_format: DrmFourcc,
        bpp: usize,
    ) -> Result<EglImage, crate::Error> {
        let fd = match &img.tensor {
            edgefirst_tensor::Tensor::Dma(dma_tensor) => dma_tensor.fd.as_raw_fd(),
            _ => {
                return Err(Error::NotImplemented(
                    "create_egl_image_with_dims requires DMA tensor".to_string(),
                ));
            }
        };

        let pitch = width * bpp;
        let egl_img_attr = vec![
            egl_ext::LINUX_DRM_FOURCC as Attrib,
            drm_format as u32 as Attrib,
            khronos_egl::WIDTH as Attrib,
            width as Attrib,
            khronos_egl::HEIGHT as Attrib,
            height as Attrib,
            egl_ext::DMA_BUF_PLANE0_PITCH as Attrib,
            pitch as Attrib,
            egl_ext::DMA_BUF_PLANE0_OFFSET as Attrib,
            0 as Attrib,
            egl_ext::DMA_BUF_PLANE0_FD as Attrib,
            fd as Attrib,
            egl::IMAGE_PRESERVED as Attrib,
            egl::TRUE as Attrib,
            khronos_egl::NONE as Attrib,
        ];

        self.new_egl_image_owned(egl_ext::LINUX_DMA_BUF, &egl_img_attr)
    }

    /// Get or create an EGLImage for a packed RGB DMA destination with
    /// reinterpreted dimensions. Uses the dst cache keyed by buffer identity.
    fn get_or_create_egl_image_rgb(
        &mut self,
        img: &TensorImage,
        width: usize,
        height: usize,
        drm_format: DrmFourcc,
        bpp: usize,
    ) -> Result<egl::Image, crate::Error> {
        let id = img.buffer_identity().id();
        self.dst_egl_cache.sweep();

        let ts = self.dst_egl_cache.next_timestamp();
        if let Some(cached) = self.dst_egl_cache.entries.get_mut(&id) {
            self.dst_egl_cache.hits += 1;
            cached.last_used = ts;
            log::trace!("EglImageCache dst (RGB) hit: id={id:#x}");
            return Ok(cached.egl_image.egl_image);
        }
        self.dst_egl_cache.misses += 1;
        log::trace!("EglImageCache dst (RGB) miss: id={id:#x}");

        if self.dst_egl_cache.entries.len() >= self.dst_egl_cache.capacity {
            self.dst_egl_cache.evict_lru();
        }

        let egl_image_obj = self.create_egl_image_with_dims(img, width, height, drm_format, bpp)?;
        let handle = egl_image_obj.egl_image;
        let guard = img.buffer_identity().weak();
        let ts = self.dst_egl_cache.next_timestamp();
        self.dst_egl_cache.entries.insert(
            id,
            CachedEglImage {
                egl_image: egl_image_obj,
                guard,
                renderbuffer: None,
                last_used: ts,
            },
        );
        Ok(handle)
    }

    /// Get or create an EGLImage + renderbuffer for direct RGB rendering.
    /// Both are cached in dst_egl_cache keyed by buffer identity.
    /// Returns (renderbuffer_id, width, height).
    fn get_or_create_rgb_direct_rbo(
        &mut self,
        dst: &TensorImage,
    ) -> crate::Result<(u32, i32, i32)> {
        let id = dst.buffer_identity().id();
        let width = dst.width() as i32;
        let height = dst.height() as i32;

        self.dst_egl_cache.sweep();

        // Check cache for existing entry with renderbuffer
        let ts = self.dst_egl_cache.next_timestamp();
        if let Some(cached) = self.dst_egl_cache.entries.get_mut(&id) {
            if let Some(rbo) = cached.renderbuffer {
                self.dst_egl_cache.hits += 1;
                cached.last_used = ts;
                log::trace!("EglImageCache dst (rgb_direct) hit: id={id:#x}");
                return Ok((rbo, width, height));
            }
        }
        self.dst_egl_cache.misses += 1;
        log::trace!("EglImageCache dst (rgb_direct) miss: id={id:#x}");

        // Evict least-recently-used entry if at capacity
        if self.dst_egl_cache.entries.len() >= self.dst_egl_cache.capacity {
            self.dst_egl_cache.evict_lru();
        }

        // Create EGLImage from BGR888 DMA-buf
        let egl_image_obj =
            self.create_egl_image_with_dims(dst, dst.width(), dst.height(), DrmFourcc::Bgr888, 3)?;

        // Create renderbuffer and bind EGLImage to it
        let rbo = unsafe {
            let mut rbo = 0u32;
            gls::gl::GenRenderbuffers(1, &mut rbo);
            gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, rbo);
            gls::gl::EGLImageTargetRenderbufferStorageOES(
                gls::gl::RENDERBUFFER,
                egl_image_obj.egl_image.as_ptr(),
            );
            if let Err(e) = check_gl_error(function!(), line!()) {
                gls::gl::DeleteRenderbuffers(1, &rbo);
                return Err(e);
            }
            rbo
        };

        // Cache both
        let guard = dst.buffer_identity().weak();
        let ts = self.dst_egl_cache.next_timestamp();
        self.dst_egl_cache.entries.insert(
            id,
            CachedEglImage {
                egl_image: egl_image_obj,
                guard,
                renderbuffer: Some(rbo),
                last_used: ts,
            },
        );

        Ok((rbo, width, height))
    }

    // Reshapes the segmentation to be compatible with RGBA texture array rendering.
    fn reshape_segmentation_to_rgba(&self, segmentation: &[u8], shape: [usize; 3]) -> Vec<u8> {
        let [height, width, classes] = shape;

        let n_layer_stride = height * width * 4;
        let n_row_stride = width * 4;
        let n_col_stride = 4;
        let row_stride = width * classes;
        let col_stride = classes;

        let mut new_segmentation = vec![0u8; n_layer_stride * classes.div_ceil(4)];

        for i in 0..height {
            for j in 0..width {
                for k in 0..classes.div_ceil(4) * 4 {
                    if k >= classes {
                        new_segmentation[n_layer_stride * (k / 4)
                            + i * n_row_stride
                            + j * n_col_stride
                            + k % 4] = 0;
                    } else {
                        new_segmentation[n_layer_stride * (k / 4)
                            + i * n_row_stride
                            + j * n_col_stride
                            + k % 4] = segmentation[i * row_stride + j * col_stride + k];
                    }
                }
            }
        }

        new_segmentation
    }

    fn render_modelpack_segmentation(
        &mut self,
        dst_roi: RegionOfInterest,
        segmentation: &[u8],
        shape: [usize; 3],
    ) -> Result<(), crate::Error> {
        log::debug!("start render_segmentation_to_image");

        // TODO: Implement specialization for 2 classes and 4 classes which shouldn't
        // need rearranging the data
        let new_segmentation = self.reshape_segmentation_to_rgba(segmentation, shape);

        let [height, width, classes] = shape;

        let format = gls::gl::RGBA;
        let texture_target = gls::gl::TEXTURE_2D_ARRAY;
        self.segmentation_program
            .load_uniform_1i(c"background_index", shape[2] as i32 - 1)?;

        gls::use_program(self.segmentation_program.id);

        gls::bind_texture(texture_target, self.segmentation_texture.id);
        gls::active_texture(gls::gl::TEXTURE0);
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MIN_FILTER,
            gls::gl::LINEAR as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MAG_FILTER,
            gls::gl::LINEAR as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_S,
            gls::gl::CLAMP_TO_EDGE as i32,
        );

        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_T,
            gls::gl::CLAMP_TO_EDGE as i32,
        );

        gls::tex_image3d(
            texture_target,
            0,
            format as i32,
            width as i32,
            height as i32,
            classes.div_ceil(4) as i32,
            0,
            format,
            gls::gl::UNSIGNED_BYTE,
            Some(&new_segmentation),
        );

        let src_roi = RegionOfInterest {
            left: 0.,
            top: 1.,
            right: 1.,
            bottom: 0.,
        };

        unsafe {
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0., // left top
                dst_roi.right,
                dst_roi.top,
                0., // right top
                dst_roi.right,
                dst_roi.bottom,
                0., // right bottom
                dst_roi.left,
                dst_roi.bottom,
                0., // left bottom
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * camera_vertices.len()) as isize,
                camera_vertices.as_ptr() as *const c_void,
            );

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
            gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

            let texture_vertices: [f32; 8] = [
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * 8) as isize,
                (texture_vertices[0..]).as_ptr() as *const c_void,
            );

            let vertices_index: [u32; 4] = [0, 1, 2, 3];
            gls::gl::DrawElements(
                gls::gl::TRIANGLE_FAN,
                vertices_index.len() as i32,
                gls::gl::UNSIGNED_INT,
                vertices_index.as_ptr() as *const c_void,
            );
        }

        Ok(())
    }

    fn render_yolo_segmentation(
        &mut self,
        dst_roi: RegionOfInterest,
        segmentation: &[u8],
        shape: [usize; 2],
        class: usize,
    ) -> Result<(), crate::Error> {
        log::debug!("start render_yolo_segmentation");

        let [height, width] = shape;

        let format = gls::gl::RED;
        let texture_target = gls::gl::TEXTURE_2D;
        gls::use_program(self.instanced_segmentation_program.id);
        self.instanced_segmentation_program
            .load_uniform_1i(c"class_index", class as i32)?;
        gls::bind_texture(texture_target, self.segmentation_texture.id);
        gls::active_texture(gls::gl::TEXTURE0);
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MIN_FILTER,
            gls::gl::LINEAR as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MAG_FILTER,
            gls::gl::LINEAR as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_S,
            gls::gl::CLAMP_TO_EDGE as i32,
        );

        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_T,
            gls::gl::CLAMP_TO_EDGE as i32,
        );

        gls::tex_image2d(
            texture_target,
            0,
            format as i32,
            width as i32,
            height as i32,
            0,
            format,
            gls::gl::UNSIGNED_BYTE,
            Some(segmentation),
        );

        let src_roi = RegionOfInterest {
            left: 0.,
            top: 1.,
            right: 1.,
            bottom: 0.,
        };

        unsafe {
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

            let camera_vertices: [f32; 12] = [
                dst_roi.left,
                dst_roi.top,
                0., // left top
                dst_roi.right,
                dst_roi.top,
                0., // right top
                dst_roi.right,
                dst_roi.bottom,
                0., // right bottom
                dst_roi.left,
                dst_roi.bottom,
                0., // left bottom
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * camera_vertices.len()) as isize,
                camera_vertices.as_ptr() as *const c_void,
            );

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
            gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

            let texture_vertices: [f32; 8] = [
                src_roi.left,
                src_roi.top,
                src_roi.right,
                src_roi.top,
                src_roi.right,
                src_roi.bottom,
                src_roi.left,
                src_roi.bottom,
            ];
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (size_of::<f32>() * 8) as isize,
                (texture_vertices).as_ptr() as *const c_void,
            );

            let vertices_index: [u32; 4] = [0, 1, 2, 3];
            gls::gl::DrawElements(
                gls::gl::TRIANGLE_FAN,
                vertices_index.len() as i32,
                gls::gl::UNSIGNED_INT,
                vertices_index.as_ptr() as *const c_void,
            );
            gls::gl::Finish();
        }

        Ok(())
    }

    /// Repack proto tensor `(H, W, num_protos)` as f32 into RGBA f16 layers
    /// suitable for upload to a GL_TEXTURE_2D_ARRAY with GL_RGBA16F.
    ///
    /// Returns `(repacked_bytes, num_layers)` where each layer is H*W*4 half-floats.
    fn repack_protos_to_rgba_f16(protos: &ndarray::Array3<f32>) -> (Vec<u8>, usize) {
        let (height, width, num_protos) = protos.dim();
        let num_layers = num_protos.div_ceil(4);
        // Each layer is H*W*4 half-floats, each half-float is 2 bytes
        let layer_stride = height * width * 4;
        let mut buf = vec![0u16; layer_stride * num_layers];

        for y in 0..height {
            for x in 0..width {
                for k in 0..num_layers * 4 {
                    let val = if k < num_protos {
                        half::f16::from_f32(protos[[y, x, k]])
                    } else {
                        half::f16::ZERO
                    };
                    let layer = k / 4;
                    let channel = k % 4;
                    buf[layer * layer_stride + y * width * 4 + x * 4 + channel] = val.to_bits();
                }
            }
        }

        // Reinterpret u16 buffer as bytes
        let byte_buf = unsafe {
            std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * 2).to_vec()
        };
        (byte_buf, num_layers)
    }

    /// Render YOLO proto segmentation masks using the fused GPU pipeline.
    ///
    /// Dispatches to the appropriate shader based on `ProtoTensor` variant:
    /// - `Quantized`: uploads raw int8 as `GL_R8I`, dequantizes in shader
    /// - `Float`: uploads as `GL_R32F` with hardware bilinear (if available),
    ///   or falls back to f16 repack path
    fn render_proto_segmentation(
        &mut self,
        detect: &[DetectBox],
        proto_data: &ProtoData,
    ) -> crate::Result<()> {
        if detect.is_empty() || proto_data.mask_coefficients.is_empty() {
            return Ok(());
        }

        let (height, width, num_protos) = proto_data.protos.dim();
        let texture_target = gls::gl::TEXTURE_2D_ARRAY;

        match &proto_data.protos {
            ProtoTensor::Quantized {
                protos,
                quantization,
            } => {
                self.render_proto_segmentation_int8(
                    detect,
                    &proto_data.mask_coefficients,
                    protos,
                    quantization,
                    height,
                    width,
                    num_protos,
                    texture_target,
                )?;
            }
            ProtoTensor::Float(protos_f32) => {
                if self.has_float_linear {
                    self.render_proto_segmentation_f32(
                        detect,
                        &proto_data.mask_coefficients,
                        protos_f32,
                        height,
                        width,
                        num_protos,
                        texture_target,
                    )?;
                } else {
                    // Fallback: repack to RGBA16F and use existing f16 shader
                    self.render_proto_segmentation_f16(
                        detect,
                        &proto_data.mask_coefficients,
                        protos_f32,
                        height,
                        width,
                        num_protos,
                        texture_target,
                    )?;
                }
            }
        }

        unsafe { gls::gl::Finish() };
        Ok(())
    }

    /// Render detection quads using the active program. Shared by all proto
    /// shader paths.
    fn render_proto_detection_quads(
        &self,
        program: &GlProgram,
        detect: &[DetectBox],
        mask_coefficients: &[Vec<f32>],
    ) -> crate::Result<()> {
        let cvt_screen_coord = |normalized: f32| normalized * 2.0 - 1.0;

        for (det, coeff) in detect.iter().zip(mask_coefficients.iter()) {
            let mut packed_coeff = [[0.0f32; 4]; 8];
            for (i, val) in coeff.iter().enumerate().take(32) {
                packed_coeff[i / 4][i % 4] = *val;
            }

            program.load_uniform_4fv(c"mask_coeff", &packed_coeff)?;
            program.load_uniform_1i(c"class_index", det.label as i32)?;

            let dst_roi = RegionOfInterest {
                left: cvt_screen_coord(det.bbox.xmin),
                top: cvt_screen_coord(det.bbox.ymax),
                right: cvt_screen_coord(det.bbox.xmax),
                bottom: cvt_screen_coord(det.bbox.ymin),
            };

            // Proto texture coords: tex row 0 = image top (data uploaded in
            // row-major order where y=0 is top of image, and GL treats the
            // first row of pixel data as the bottom of the texture — but
            // texelFetch(y=0) returns that bottom row, which is our image top).
            // So tc.y=0 → image top, tc.y=1 → image bottom.
            // At NDC top (higher Y = image bottom = ymax), we want tc.y = ymax.
            // At NDC bottom (lower Y = image top = ymin), we want tc.y = ymin.
            let src_roi = RegionOfInterest {
                left: det.bbox.xmin,
                top: det.bbox.ymax,
                right: det.bbox.xmax,
                bottom: det.bbox.ymin,
            };

            unsafe {
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);

                let camera_vertices: [f32; 12] = [
                    dst_roi.left,
                    dst_roi.top,
                    0.,
                    dst_roi.right,
                    dst_roi.top,
                    0.,
                    dst_roi.right,
                    dst_roi.bottom,
                    0.,
                    dst_roi.left,
                    dst_roi.bottom,
                    0.,
                ];
                gls::gl::BufferSubData(
                    gls::gl::ARRAY_BUFFER,
                    0,
                    (size_of::<f32>() * camera_vertices.len()) as isize,
                    camera_vertices.as_ptr() as *const c_void,
                );

                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
                gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);

                let texture_vertices: [f32; 8] = [
                    src_roi.left,
                    src_roi.top,
                    src_roi.right,
                    src_roi.top,
                    src_roi.right,
                    src_roi.bottom,
                    src_roi.left,
                    src_roi.bottom,
                ];
                gls::gl::BufferSubData(
                    gls::gl::ARRAY_BUFFER,
                    0,
                    (size_of::<f32>() * 8) as isize,
                    texture_vertices.as_ptr() as *const c_void,
                );

                let vertices_index: [u32; 4] = [0, 1, 2, 3];
                gls::gl::DrawElements(
                    gls::gl::TRIANGLE_FAN,
                    vertices_index.len() as i32,
                    gls::gl::UNSIGNED_INT,
                    vertices_index.as_ptr() as *const c_void,
                );
            }
        }
        Ok(())
    }

    /// Int8 proto path: upload raw i8 protos as `GL_R8I`, dispatch by
    /// interpolation mode.
    #[allow(clippy::too_many_arguments)]
    fn render_proto_segmentation_int8(
        &mut self,
        detect: &[DetectBox],
        mask_coefficients: &[Vec<f32>],
        protos: &ndarray::Array3<i8>,
        quantization: &edgefirst_decoder::Quantization,
        height: usize,
        width: usize,
        num_protos: usize,
        texture_target: u32,
    ) -> crate::Result<()> {
        // Upload raw int8 protos as R8I texture array (1 proto per layer)
        gls::bind_texture(texture_target, self.proto_texture.id);
        gls::active_texture(gls::gl::TEXTURE0);
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MIN_FILTER,
            gls::gl::NEAREST as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MAG_FILTER,
            gls::gl::NEAREST as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_S,
            gls::gl::CLAMP_TO_EDGE as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_T,
            gls::gl::CLAMP_TO_EDGE as i32,
        );

        // Protos are (H, W, num_protos) in row-major. We need to repack to
        // layer-first layout: layer k = all (H, W) texels for proto k.
        let mut tex_data = vec![0i8; height * width * num_protos];
        for k in 0..num_protos {
            for y in 0..height {
                for x in 0..width {
                    tex_data[k * height * width + y * width + x] = protos[[y, x, k]];
                }
            }
        }

        gls::tex_image3d(
            texture_target,
            0,
            gls::gl::R8I as i32,
            width as i32,
            height as i32,
            num_protos as i32,
            0,
            gls::gl::RED_INTEGER,
            gls::gl::BYTE,
            Some(&tex_data),
        );

        let proto_scale = quantization.scale;
        let proto_scaled_zp = -(quantization.zero_point as f32) * quantization.scale;

        match self.int8_interpolation_mode {
            Int8InterpolationMode::Nearest => {
                let program = &self.proto_segmentation_int8_nearest_program;
                gls::use_program(program.id);
                program.load_uniform_1i(c"num_protos", num_protos as i32)?;
                program.load_uniform_1f(c"proto_scale", proto_scale)?;
                program.load_uniform_1f(c"proto_scaled_zp", proto_scaled_zp)?;
                self.render_proto_detection_quads(program, detect, mask_coefficients)?;
            }
            Int8InterpolationMode::Bilinear => {
                let program = &self.proto_segmentation_int8_bilinear_program;
                gls::use_program(program.id);
                program.load_uniform_1i(c"num_protos", num_protos as i32)?;
                program.load_uniform_1f(c"proto_scale", proto_scale)?;
                program.load_uniform_1f(c"proto_scaled_zp", proto_scaled_zp)?;
                self.render_proto_detection_quads(program, detect, mask_coefficients)?;
            }
            Int8InterpolationMode::TwoPass => {
                self.render_proto_int8_two_pass(
                    detect,
                    mask_coefficients,
                    quantization,
                    height,
                    width,
                    num_protos,
                    texture_target,
                )?;
            }
        }

        Ok(())
    }

    /// Two-pass int8 path: dequant int8→RGBA16F FBO, then render with
    /// existing f16 shader using GL_LINEAR.
    #[allow(clippy::too_many_arguments)]
    fn render_proto_int8_two_pass(
        &self,
        detect: &[DetectBox],
        mask_coefficients: &[Vec<f32>],
        quantization: &edgefirst_decoder::Quantization,
        height: usize,
        width: usize,
        num_protos: usize,
        texture_target: u32,
    ) -> crate::Result<()> {
        let num_layers = num_protos.div_ceil(4);

        // Save the caller's FBO and viewport so we can restore after dequant.
        let (saved_fbo, saved_viewport) = unsafe {
            let mut fbo: i32 = 0;
            gls::gl::GetIntegerv(gls::gl::FRAMEBUFFER_BINDING, &mut fbo);
            let mut vp = [0i32; 4];
            gls::gl::GetIntegerv(gls::gl::VIEWPORT, vp.as_mut_ptr());
            (fbo as u32, vp)
        };

        // Pass 1: Dequantize int8 → RGBA16F texture via framebuffer
        let dequant_fbo = FrameBuffer::new();
        gls::bind_texture(texture_target, self.proto_dequant_texture.id);
        gls::tex_image3d::<u8>(
            texture_target,
            0,
            gls::gl::RGBA16F as i32,
            width as i32,
            height as i32,
            num_layers as i32,
            0,
            gls::gl::RGBA,
            gls::gl::HALF_FLOAT,
            None,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MIN_FILTER,
            gls::gl::LINEAR as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MAG_FILTER,
            gls::gl::LINEAR as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_S,
            gls::gl::CLAMP_TO_EDGE as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_T,
            gls::gl::CLAMP_TO_EDGE as i32,
        );

        let proto_scale = quantization.scale;
        let proto_scaled_zp = -(quantization.zero_point as f32) * quantization.scale;

        let dequant_program = &self.proto_dequant_int8_program;
        gls::use_program(dequant_program.id);
        dequant_program.load_uniform_1f(c"proto_scale", proto_scale)?;
        dequant_program.load_uniform_1f(c"proto_scaled_zp", proto_scaled_zp)?;

        // Bind the int8 proto texture to TEXTURE0 for the dequant shader
        gls::active_texture(gls::gl::TEXTURE0);
        gls::bind_texture(texture_target, self.proto_texture.id);

        // Render each RGBA16F layer (4 protos per layer)
        for layer in 0..num_layers {
            dequant_fbo.bind();
            unsafe {
                gls::gl::FramebufferTextureLayer(
                    gls::gl::FRAMEBUFFER,
                    gls::gl::COLOR_ATTACHMENT0,
                    self.proto_dequant_texture.id,
                    0,
                    layer as i32,
                );
                gls::gl::Viewport(0, 0, width as i32, height as i32);
            }
            dequant_program.load_uniform_1i(c"base_layer", (layer * 4) as i32)?;

            // Full-screen quad
            unsafe {
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
                let verts: [f32; 12] = [
                    -1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, 1.0, 0.0, -1.0, 1.0, 0.0,
                ];
                gls::gl::BufferSubData(
                    gls::gl::ARRAY_BUFFER,
                    0,
                    (size_of::<f32>() * 12) as isize,
                    verts.as_ptr() as *const c_void,
                );
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
                gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
                let tc: [f32; 8] = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
                gls::gl::BufferSubData(
                    gls::gl::ARRAY_BUFFER,
                    0,
                    (size_of::<f32>() * 8) as isize,
                    tc.as_ptr() as *const c_void,
                );
                let idx: [u32; 4] = [0, 1, 2, 3];
                gls::gl::DrawElements(
                    gls::gl::TRIANGLE_FAN,
                    4,
                    gls::gl::UNSIGNED_INT,
                    idx.as_ptr() as *const c_void,
                );
            }
        }

        // Drop the dequant FBO (its Drop unbinds to 0) and restore the caller's.
        drop(dequant_fbo);
        unsafe {
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, saved_fbo);
            gls::gl::Viewport(
                saved_viewport[0],
                saved_viewport[1],
                saved_viewport[2],
                saved_viewport[3],
            );
        }

        // Pass 2: render with existing f16 shader reading from dequant texture
        let program = &self.proto_segmentation_program;
        gls::use_program(program.id);
        gls::active_texture(gls::gl::TEXTURE0);
        gls::bind_texture(texture_target, self.proto_dequant_texture.id);
        program.load_uniform_1i(c"num_layers", num_layers as i32)?;
        self.render_proto_detection_quads(program, detect, mask_coefficients)?;

        Ok(())
    }

    /// F32 proto path: upload as `GL_R32F` with `GL_LINEAR` filtering.
    #[allow(clippy::too_many_arguments)]
    fn render_proto_segmentation_f32(
        &self,
        detect: &[DetectBox],
        mask_coefficients: &[Vec<f32>],
        protos_f32: &ndarray::Array3<f32>,
        height: usize,
        width: usize,
        num_protos: usize,
        texture_target: u32,
    ) -> crate::Result<()> {
        let program = &self.proto_segmentation_f32_program;
        gls::use_program(program.id);
        gls::bind_texture(texture_target, self.proto_texture.id);
        gls::active_texture(gls::gl::TEXTURE0);
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MIN_FILTER,
            gls::gl::LINEAR as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MAG_FILTER,
            gls::gl::LINEAR as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_S,
            gls::gl::CLAMP_TO_EDGE as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_T,
            gls::gl::CLAMP_TO_EDGE as i32,
        );

        // Repack protos to layer-first layout: (num_protos, H, W)
        let mut tex_data = vec![0.0f32; height * width * num_protos];
        for k in 0..num_protos {
            for y in 0..height {
                for x in 0..width {
                    tex_data[k * height * width + y * width + x] = protos_f32[[y, x, k]];
                }
            }
        }

        gls::tex_image3d(
            texture_target,
            0,
            gls::gl::R32F as i32,
            width as i32,
            height as i32,
            num_protos as i32,
            0,
            gls::gl::RED,
            gls::gl::FLOAT,
            Some(&tex_data),
        );

        program.load_uniform_1i(c"num_protos", num_protos as i32)?;
        self.render_proto_detection_quads(program, detect, mask_coefficients)?;

        Ok(())
    }

    /// F16 fallback path: repack f32 protos to RGBA16F and use existing
    /// f16 shader with GL_LINEAR. Used when GL_OES_texture_float_linear
    /// is not available.
    #[allow(clippy::too_many_arguments)]
    fn render_proto_segmentation_f16(
        &self,
        detect: &[DetectBox],
        mask_coefficients: &[Vec<f32>],
        protos_f32: &ndarray::Array3<f32>,
        height: usize,
        width: usize,
        num_protos: usize,
        texture_target: u32,
    ) -> crate::Result<()> {
        let num_layers = num_protos.div_ceil(4);
        let (tex_data, _) = Self::repack_protos_to_rgba_f16(protos_f32);

        let program = &self.proto_segmentation_program;
        gls::use_program(program.id);
        gls::bind_texture(texture_target, self.proto_texture.id);
        gls::active_texture(gls::gl::TEXTURE0);
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MIN_FILTER,
            gls::gl::LINEAR as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_MAG_FILTER,
            gls::gl::LINEAR as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_S,
            gls::gl::CLAMP_TO_EDGE as i32,
        );
        gls::tex_parameteri(
            texture_target,
            gls::gl::TEXTURE_WRAP_T,
            gls::gl::CLAMP_TO_EDGE as i32,
        );

        gls::tex_image3d(
            texture_target,
            0,
            gls::gl::RGBA16F as i32,
            width as i32,
            height as i32,
            num_layers as i32,
            0,
            gls::gl::RGBA,
            gls::gl::HALF_FLOAT,
            Some(&tex_data),
        );

        program.load_uniform_1i(c"num_layers", num_layers as i32)?;
        self.render_proto_detection_quads(program, detect, mask_coefficients)?;

        Ok(())
    }

    fn render_segmentation(
        &mut self,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
    ) -> crate::Result<()> {
        if segmentation.is_empty() {
            return Ok(());
        }

        let is_modelpack = segmentation[0].segmentation.shape()[2] > 1;
        // top and bottom are flipped because OpenGL uses 0,0 as bottom left
        let cvt_screen_coord = |normalized| normalized * 2.0 - 1.0;
        if is_modelpack {
            let seg = &segmentation[0];
            let dst_roi = RegionOfInterest {
                left: cvt_screen_coord(seg.xmin),
                top: cvt_screen_coord(seg.ymax),
                right: cvt_screen_coord(seg.xmax),
                bottom: cvt_screen_coord(seg.ymin),
            };
            let segment = seg.segmentation.as_standard_layout();
            let slice = segment.as_slice().ok_or(Error::Internal(
                "Cannot get slice of segmentation".to_owned(),
            ))?;

            self.render_modelpack_segmentation(
                dst_roi,
                slice,
                [
                    seg.segmentation.shape()[0],
                    seg.segmentation.shape()[1],
                    seg.segmentation.shape()[2],
                ],
            )?;
        } else {
            for (seg, det) in segmentation.iter().zip(detect) {
                let dst_roi = RegionOfInterest {
                    left: cvt_screen_coord(seg.xmin),
                    top: cvt_screen_coord(seg.ymax),
                    right: cvt_screen_coord(seg.xmax),
                    bottom: cvt_screen_coord(seg.ymin),
                };

                let segment = seg.segmentation.as_standard_layout();
                let slice = segment.as_slice().ok_or(Error::Internal(
                    "Cannot get slice of segmentation".to_owned(),
                ))?;

                self.render_yolo_segmentation(
                    dst_roi,
                    slice,
                    [seg.segmentation.shape()[0], seg.segmentation.shape()[1]],
                    det.label,
                )?;
            }
        }

        gls::disable(gls::gl::BLEND);
        Ok(())
    }

    fn render_box(&mut self, dst: &TensorImage, detect: &[DetectBox]) -> Result<(), Error> {
        unsafe {
            gls::gl::UseProgram(self.color_program.id);
            let rescale = |x: f32| x * 2.0 - 1.0;
            let thickness = 3.0;
            for d in detect {
                self.color_program
                    .load_uniform_1i(c"class_index", d.label as i32)?;
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
                let bbox: [f32; 4] = d.bbox.into();
                let outer_box = [
                    bbox[0] - thickness / dst.width() as f32,
                    bbox[1] - thickness / dst.height() as f32,
                    bbox[2] + thickness / dst.width() as f32,
                    bbox[3] + thickness / dst.height() as f32,
                ];
                let camera_vertices: [f32; 24] = [
                    rescale(bbox[0]),
                    rescale(bbox[3]),
                    0., // bottom left
                    rescale(bbox[2]),
                    rescale(bbox[3]),
                    0., // bottom right
                    rescale(bbox[2]),
                    rescale(bbox[1]),
                    0., // top right
                    rescale(bbox[0]),
                    rescale(bbox[1]),
                    0., // top left
                    rescale(outer_box[0]),
                    rescale(outer_box[3]),
                    0., // bottom left
                    rescale(outer_box[2]),
                    rescale(outer_box[3]),
                    0., // bottom right
                    rescale(outer_box[2]),
                    rescale(outer_box[1]),
                    0., // top right
                    rescale(outer_box[0]),
                    rescale(outer_box[1]),
                    0., // top left
                ];
                gls::gl::BufferData(
                    gls::gl::ARRAY_BUFFER,
                    (size_of::<f32>() * camera_vertices.len()) as isize,
                    camera_vertices.as_ptr() as *const c_void,
                    gls::gl::DYNAMIC_DRAW,
                );

                let vertices_index: [u32; 10] = [0, 1, 5, 2, 6, 3, 7, 0, 4, 5];
                gls::gl::DrawElements(
                    gls::gl::TRIANGLE_STRIP,
                    vertices_index.len() as i32,
                    gls::gl::UNSIGNED_INT,
                    vertices_index.as_ptr() as *const c_void,
                );
            }
        }
        check_gl_error(function!(), line!())?;
        Ok(())
    }
}
struct EglImage {
    egl_image: egl::Image,
    egl: Rc<Egl>,
    display: egl::Display,
}

impl Drop for EglImage {
    fn drop(&mut self) {
        if self.egl_image.as_ptr() == egl::NO_IMAGE {
            return;
        }

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let e =
                GlContext::egl_destroy_image_with_fallback(&self.egl, self.display, self.egl_image);
            if let Err(e) = e {
                error!("Could not destroy EGL image: {e:?}");
            }
        }));
    }
}

struct Texture {
    id: u32,
    target: gls::gl::types::GLenum,
    width: usize,
    height: usize,
    format: gls::gl::types::GLenum,
}

impl Default for Texture {
    fn default() -> Self {
        Self::new()
    }
}

impl Texture {
    fn new() -> Self {
        let mut id = 0;
        unsafe { gls::gl::GenTextures(1, &raw mut id) };
        Self {
            id,
            target: 0,
            width: 0,
            height: 0,
            format: 0,
        }
    }

    fn update_texture(
        &mut self,
        target: gls::gl::types::GLenum,
        width: usize,
        height: usize,
        format: gls::gl::types::GLenum,
        data: &[u8],
    ) {
        if target != self.target
            || width != self.width
            || height != self.height
            || format != self.format
        {
            unsafe {
                gls::gl::TexImage2D(
                    target,
                    0,
                    format as i32,
                    width as i32,
                    height as i32,
                    0,
                    format,
                    gls::gl::UNSIGNED_BYTE,
                    data.as_ptr() as *const c_void,
                );
            }
            self.target = target;
            self.format = format;
            self.width = width;
            self.height = height;
        } else {
            unsafe {
                gls::gl::TexSubImage2D(
                    target,
                    0,
                    0,
                    0,
                    width as i32,
                    height as i32,
                    format,
                    gls::gl::UNSIGNED_BYTE,
                    data.as_ptr() as *const c_void,
                );
            }
        }
    }
}

impl Drop for Texture {
    fn drop(&mut self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            gls::gl::DeleteTextures(1, &raw mut self.id)
        }));
    }
}

struct Buffer {
    id: u32,
    buffer_index: u32,
}

impl Buffer {
    fn new(buffer_index: u32, size_per_point: usize, max_points: usize) -> Buffer {
        let mut id = 0;
        unsafe {
            gls::gl::EnableVertexAttribArray(buffer_index);
            gls::gl::GenBuffers(1, &raw mut id);
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, id);
            gls::gl::VertexAttribPointer(
                buffer_index,
                size_per_point as i32,
                gls::gl::FLOAT,
                gls::gl::FALSE,
                0,
                null(),
            );
            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                (size_of::<f32>() * size_per_point * max_points) as isize,
                null(),
                gls::gl::DYNAMIC_DRAW,
            );
        }

        Buffer { id, buffer_index }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            gls::gl::DeleteBuffers(1, &raw mut self.id)
        }));
    }
}

struct FrameBuffer {
    id: u32,
}

impl FrameBuffer {
    fn new() -> FrameBuffer {
        let mut id = 0;
        unsafe {
            gls::gl::GenFramebuffers(1, &raw mut id);
        }

        FrameBuffer { id }
    }

    fn bind(&self) {
        unsafe { gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, self.id) };
    }

    fn unbind(&self) {
        unsafe { gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0) };
    }
}

impl Drop for FrameBuffer {
    fn drop(&mut self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.unbind();
            unsafe {
                gls::gl::DeleteFramebuffers(1, &raw mut self.id);
            }
        }));
    }
}

pub struct GlProgram {
    id: u32,
    vertex_id: u32,
    fragment_id: u32,
}

impl GlProgram {
    fn new(vertex_shader: &str, fragment_shader: &str) -> Result<Self, crate::Error> {
        let id = unsafe { gls::gl::CreateProgram() };
        let vertex_id = unsafe { gls::gl::CreateShader(gls::gl::VERTEX_SHADER) };
        if compile_shader_from_str(vertex_id, vertex_shader, "shader_vert").is_err() {
            log::debug!("Vertex shader source:\n{}", vertex_shader);
            return Err(crate::Error::OpenGl(format!(
                "Shader compile error: {vertex_shader}"
            )));
        }
        unsafe {
            gls::gl::AttachShader(id, vertex_id);
        }

        let fragment_id = unsafe { gls::gl::CreateShader(gls::gl::FRAGMENT_SHADER) };
        if compile_shader_from_str(fragment_id, fragment_shader, "shader_frag").is_err() {
            log::debug!("Fragment shader source:\n{}", fragment_shader);
            return Err(crate::Error::OpenGl(format!(
                "Shader compile error: {fragment_shader}"
            )));
        }

        unsafe {
            gls::gl::AttachShader(id, fragment_id);
            gls::gl::LinkProgram(id);
            gls::gl::UseProgram(id);
        }

        Ok(Self {
            id,
            vertex_id,
            fragment_id,
        })
    }

    #[allow(dead_code)]
    fn load_uniform_1f(&self, name: &CStr, value: f32) -> Result<(), crate::Error> {
        unsafe {
            gls::gl::UseProgram(self.id);
            let location = gls::gl::GetUniformLocation(self.id, name.as_ptr());
            gls::gl::Uniform1f(location, value);
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn load_uniform_1i(&self, name: &CStr, value: i32) -> Result<(), crate::Error> {
        unsafe {
            gls::gl::UseProgram(self.id);
            let location = gls::gl::GetUniformLocation(self.id, name.as_ptr());
            gls::gl::Uniform1i(location, value);
        }
        Ok(())
    }

    fn load_uniform_4fv(&self, name: &CStr, value: &[[f32; 4]]) -> Result<(), crate::Error> {
        unsafe {
            gls::gl::UseProgram(self.id);
            let location = gls::gl::GetUniformLocation(self.id, name.as_ptr());
            if location == -1 {
                return Err(crate::Error::OpenGl(format!(
                    "Could not find uniform location for '{}'",
                    name.to_string_lossy().into_owned()
                )));
            }
            gls::gl::Uniform4fv(location, value.len() as i32, value.as_flattened().as_ptr());
        }
        check_gl_error(function!(), line!())?;
        Ok(())
    }
}

impl Drop for GlProgram {
    fn drop(&mut self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            gls::gl::DeleteProgram(self.id);
            gls::gl::DeleteShader(self.fragment_id);
            gls::gl::DeleteShader(self.vertex_id);
        }));
    }
}

fn compile_shader_from_str(shader: u32, shader_source: &str, shader_name: &str) -> Result<(), ()> {
    let src = match CString::from_str(shader_source) {
        Ok(v) => v,
        Err(_) => return Err(()),
    };
    let src_ptr = src.as_ptr();
    unsafe {
        gls::gl::ShaderSource(shader, 1, &raw const src_ptr, null());
        gls::gl::CompileShader(shader);
        let mut is_compiled = 0;
        gls::gl::GetShaderiv(shader, gls::gl::COMPILE_STATUS, &raw mut is_compiled);
        if is_compiled == 0 {
            let mut max_length = 0;
            gls::gl::GetShaderiv(shader, gls::gl::INFO_LOG_LENGTH, &raw mut max_length);
            let mut error_log: Vec<u8> = vec![0; max_length as usize];
            gls::gl::GetShaderInfoLog(
                shader,
                max_length,
                &raw mut max_length,
                error_log.as_mut_ptr() as *mut c_char,
            );
            error!(
                "Shader '{}' failed: {:?}\n",
                shader_name,
                CString::from_vec_with_nul(error_log)
                    .unwrap()
                    .into_string()
                    .unwrap()
            );
            gls::gl::DeleteShader(shader);
            return Err(());
        }
        Ok(())
    }
}

fn check_gl_error(name: &str, line: u32) -> Result<(), Error> {
    unsafe {
        let err = gls::gl::GetError();
        if err != gls::gl::NO_ERROR {
            error!("GL Error: {name}:{line}: {err:#X}");
            // panic!("GL Error: {err}");
            return Err(Error::OpenGl(format!("{err:#X}")));
        }
    }
    Ok(())
}

fn fourcc_to_drm(fourcc: FourCharCode) -> Result<DrmFourcc, Error> {
    match fourcc {
        RGBA => Ok(DrmFourcc::Abgr8888),
        BGRA => Ok(DrmFourcc::Argb8888),
        YUYV => Ok(DrmFourcc::Yuyv),
        VYUY => Ok(DrmFourcc::Vyuy),
        RGB | RGB_INT8 => Ok(DrmFourcc::Bgr888),
        GREY => Ok(DrmFourcc::R8),
        NV12 => Ok(DrmFourcc::Nv12),
        PLANAR_RGB | PLANAR_RGB_INT8 => Ok(DrmFourcc::R8),
        _ => Err(Error::NotSupported(format!(
            "FourCC {fourcc:?} has no DRM format mapping"
        ))),
    }
}

mod egl_ext {
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

fn generate_vertex_shader() -> &'static str {
    "\
#version 300 es
precision mediump float;
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 texCoord;

out vec3 fragPos;
out vec2 tc;

void main() {
    fragPos = pos;
    tc = texCoord;

    gl_Position = vec4(pos, 1.0);
}
"
}

fn generate_texture_fragment_shader() -> &'static str {
    "\
#version 300 es

precision mediump float;
uniform sampler2D tex;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

void main(){
    color = texture(tex, tc);
}
"
}

fn generate_texture_fragment_shader_yuv() -> &'static str {
    "\
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;
uniform samplerExternalOES tex;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

void main(){
    color = texture(tex, tc);
}
"
}

fn generate_planar_rgb_shader() -> &'static str {
    "\
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;
uniform samplerExternalOES tex;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

void main(){
    color = texture(tex, tc);
}
"
}

/// Int8 variant of [`generate_planar_rgb_shader`]. Applies XOR 0x80 bias
/// to each RGB channel (uint8 → int8 conversion) using the bit-exact
/// quantize+mod approach: `floor(v * 255 + 0.5) + 128 mod 256 / 255`.
fn generate_planar_rgb_int8_shader() -> &'static str {
    "\
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
uniform samplerExternalOES tex;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

vec3 int8_bias(vec3 v) {
    vec3 q = floor(v * 255.0 + 0.5);
    return mod(q + 128.0, 256.0) / 255.0;
}

void main(){
    vec4 c = texture(tex, tc);
    color = vec4(int8_bias(c.rgb), c.a);
}
"
}

/// Int8 variant of [`generate_texture_fragment_shader`]. Applies `fract(v + 0.5)`
/// to each RGB channel for XOR 0x80 bias (uint8 → int8 conversion).
/// Used by the direct RGB render path for RGB_INT8 output.
fn generate_texture_int8_shader() -> &'static str {
    "\
#version 300 es
precision highp float;
uniform sampler2D tex;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

// XOR 0x80 bias: quantize to uint8, add 128 mod 256, normalize back.
// This matches the CPU `byte ^ 0x80` operation exactly.
vec3 int8_bias(vec3 v) {
    vec3 q = floor(v * 255.0 + 0.5);
    return mod(q + 128.0, 256.0) / 255.0;
}

void main(){
    vec4 c = texture(tex, tc);
    color = vec4(int8_bias(c.rgb), c.a);
}
"
}

/// Int8 variant of [`generate_texture_fragment_shader_yuv`]. Applies XOR 0x80 bias
/// to each RGB channel (uint8 → int8 conversion).
/// Used by the direct RGB render path for RGB_INT8 output with external OES sources.
fn generate_texture_int8_shader_yuv() -> &'static str {
    "\
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
uniform samplerExternalOES tex;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

vec3 int8_bias(vec3 v) {
    vec3 q = floor(v * 255.0 + 0.5);
    return mod(q + 128.0, 256.0) / 255.0;
}

void main(){
    vec4 c = texture(tex, tc);
    color = vec4(int8_bias(c.rgb), c.a);
}
"
}

/// this shader requires a reshape of the segmentation output tensor to (H, W,
/// C/4, 4)
fn generate_segmentation_shader() -> &'static str {
    "\
#version 300 es
precision mediump float;
precision mediump sampler2DArray;

uniform sampler2DArray tex;
uniform vec4 colors[20];
uniform int background_index;

in vec3 fragPos;
in vec2 tc;
in vec4 fragColor;

out vec4 color;

float max_arg(const in vec4 args, out int argmax) {
    if (args[0] >= args[1] && args[0] >= args[2] && args[0] >= args[3]) {
        argmax = 0;
        return args[0];
    }
    if (args[1] >= args[0] && args[1] >= args[2] && args[1] >= args[3]) {
        argmax = 1;
        return args[1];
    }
    if (args[2] >= args[0] && args[2] >= args[1] && args[2] >= args[3]) {
        argmax = 2;
        return args[2];
    }
    argmax = 3;
    return args[3];
}

void main() {
    mediump int layers = textureSize(tex, 0).z;
    float max_all = -4.0;
    int max_ind = 0;
    for (int i = 0; i < layers; i++) {
        vec4 d = texture(tex, vec3(tc, i));
        int max_ind_ = 0;
        float max_ = max_arg(d, max_ind_);
        if (max_ <= max_all) { continue; }
        max_all = max_;
        max_ind = i*4 + max_ind_;
    }
    if (max_ind == background_index) {
        discard;
    }
    max_ind = max_ind % 20;
    color = colors[max_ind];
}
"
}

fn generate_instanced_segmentation_shader() -> &'static str {
    "\
#version 300 es
precision mediump float;
uniform sampler2D mask0;
uniform vec4 colors[20];
uniform int class_index;
in vec3 fragPos;
in vec2 tc;
in vec4 fragColor;

out vec4 color;
void main() {
    float r0 = texture(mask0, tc).r;
    int arg = int(r0>=0.5);
    if (arg == 0) {
        discard;
    }
    color = colors[class_index % 20];
}
"
}

fn generate_proto_segmentation_shader() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp sampler2DArray;

uniform sampler2DArray proto_tex;  // ceil(num_protos/4) layers, RGBA = 4 channels per layer
uniform vec4 mask_coeff[8];        // 32 coefficients packed as 8 vec4s
uniform vec4 colors[20];
uniform int class_index;
uniform int num_layers;

in vec2 tc;
out vec4 color;

void main() {
    float acc = 0.0;
    for (int i = 0; i < num_layers; i++) {
        // texture() returns bilinearly interpolated proto values (GL_LINEAR)
        acc += dot(mask_coeff[i], texture(proto_tex, vec3(tc, float(i))));
    }
    float mask = 1.0 / (1.0 + exp(-acc));  // sigmoid
    if (mask < 0.5) discard;
    color = colors[class_index % 20];
}
"
}

/// Int8 proto shader — nearest-neighbor only.
///
/// Uses `texelFetch()` at the nearest texel. No interpolation. Simplest and
/// fastest GPU execution but may show staircase artifacts at mask edges.
///
/// Layout: `GL_R8I` texture with 1 proto per layer (32 layers).
/// Mask coefficients packed as `vec4[8]`, indexed `mask_coeff[k/4][k%4]`.
fn generate_proto_segmentation_shader_int8_nearest() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
precision highp isampler2DArray;

uniform isampler2DArray proto_tex;  // 32 layers, R channel = 1 proto per layer
uniform vec4 mask_coeff[8];         // 32 coefficients packed as 8 vec4s
uniform vec4 colors[20];
uniform int class_index;
uniform int num_protos;
uniform float proto_scale;
uniform float proto_scaled_zp;      // -zero_point * scale

in vec2 tc;
out vec4 color;

void main() {
    ivec3 tex_size = textureSize(proto_tex, 0);
    int ix = clamp(int(tc.x * float(tex_size.x)), 0, tex_size.x - 1);
    int iy = clamp(int(tc.y * float(tex_size.y)), 0, tex_size.y - 1);

    float acc = 0.0;
    for (int k = 0; k < num_protos; k++) {
        float raw = float(texelFetch(proto_tex, ivec3(ix, iy, k), 0).r);
        float val = raw * proto_scale + proto_scaled_zp;
        acc += mask_coeff[k / 4][k % 4] * val;
    }
    float mask = 1.0 / (1.0 + exp(-acc));
    if (mask < 0.5) discard;
    color = colors[class_index % 20];
}
"
}

/// Int8 proto shader — shader-based bilinear interpolation (recommended).
///
/// Uses `texelFetch()` to fetch 4 neighboring texels per fragment, dequantizes
/// each, and computes bilinear weights from `fract(tc * textureSize)`.
///
/// Layout: `GL_R8I` texture with 1 proto per layer (32 layers).
fn generate_proto_segmentation_shader_int8_bilinear() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
precision highp isampler2DArray;

uniform isampler2DArray proto_tex;  // 32 layers, R channel = 1 proto per layer
uniform vec4 mask_coeff[8];         // 32 coefficients packed as 8 vec4s
uniform vec4 colors[20];
uniform int class_index;
uniform int num_protos;
uniform float proto_scale;
uniform float proto_scaled_zp;      // -zero_point * scale

in vec2 tc;
out vec4 color;

void main() {
    ivec3 tex_size = textureSize(proto_tex, 0);
    // Compute continuous position (matching GL_LINEAR convention: center at +0.5)
    vec2 pos = tc * vec2(tex_size.xy) - 0.5;
    vec2 f = fract(pos);
    ivec2 p0 = ivec2(floor(pos));
    ivec2 p1 = p0 + 1;
    // Clamp to texture bounds
    p0 = clamp(p0, ivec2(0), tex_size.xy - 1);
    p1 = clamp(p1, ivec2(0), tex_size.xy - 1);

    float w00 = (1.0 - f.x) * (1.0 - f.y);
    float w10 = f.x * (1.0 - f.y);
    float w01 = (1.0 - f.x) * f.y;
    float w11 = f.x * f.y;

    float acc = 0.0;
    for (int k = 0; k < num_protos; k++) {
        float r00 = float(texelFetch(proto_tex, ivec3(p0.x, p0.y, k), 0).r);
        float r10 = float(texelFetch(proto_tex, ivec3(p1.x, p0.y, k), 0).r);
        float r01 = float(texelFetch(proto_tex, ivec3(p0.x, p1.y, k), 0).r);
        float r11 = float(texelFetch(proto_tex, ivec3(p1.x, p1.y, k), 0).r);
        float interp = r00 * w00 + r10 * w10 + r01 * w01 + r11 * w11;
        float val = interp * proto_scale + proto_scaled_zp;
        acc += mask_coeff[k / 4][k % 4] * val;
    }
    float mask = 1.0 / (1.0 + exp(-acc));
    if (mask < 0.5) discard;
    color = colors[class_index % 20];
}
"
}

/// Int8 dequantization pass shader (two-pass Option C, pass 1).
///
/// Reads `GL_R8I` texel, dequantizes, and writes float to `GL_RGBA16F` render
/// target. This shader processes 4 protos at a time (packing into RGBA).
/// After this pass, the existing f16 shader reads the dequantized texture with
/// `GL_LINEAR`.
fn generate_proto_dequant_shader_int8() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
precision highp isampler2DArray;

uniform isampler2DArray proto_tex;  // 32 layers of R8I (1 proto per layer)
uniform float proto_scale;
uniform float proto_scaled_zp;      // -zero_point * scale
uniform int base_layer;             // first proto index for this output layer (0, 4, 8, ...)

in vec2 tc;
out vec4 color;

void main() {
    ivec3 tex_size = textureSize(proto_tex, 0);
    int ix = clamp(int(tc.x * float(tex_size.x)), 0, tex_size.x - 1);
    int iy = clamp(int(tc.y * float(tex_size.y)), 0, tex_size.y - 1);

    vec4 result;
    for (int c = 0; c < 4; c++) {
        int layer = base_layer + c;
        float raw = float(texelFetch(proto_tex, ivec3(ix, iy, layer), 0).r);
        result[c] = raw * proto_scale + proto_scaled_zp;
    }
    color = result;
}
"
}

/// F32 proto shader — direct R32F texture with hardware bilinear filtering.
///
/// Same structure as int8 bilinear shader but uses `texture()` for hardware
/// interpolation (requires `GL_OES_texture_float_linear`). No dequantization.
///
/// Layout: `GL_R32F` texture with 1 proto per layer (32 layers).
fn generate_proto_segmentation_shader_f32() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp sampler2DArray;

uniform sampler2DArray proto_tex;  // 32 layers, R channel = 1 proto per layer
uniform vec4 mask_coeff[8];        // 32 coefficients packed as 8 vec4s
uniform vec4 colors[20];
uniform int class_index;
uniform int num_protos;

in vec2 tc;
out vec4 color;

void main() {
    float acc = 0.0;
    for (int k = 0; k < num_protos; k++) {
        // texture() returns bilinearly interpolated proto value (GL_LINEAR on R32F)
        float val = texture(proto_tex, vec3(tc, float(k))).r;
        acc += mask_coeff[k / 4][k % 4] * val;
    }
    float mask = 1.0 / (1.0 + exp(-acc));
    if (mask < 0.5) discard;
    color = colors[class_index % 20];
}
"
}

/// Binary mask shader — int8, nearest-neighbor, logit threshold.
///
/// Outputs binary `acc > 0 ? 1.0 : 0.0` instead of `sigmoid(acc)`.  Avoids
/// the `exp()` per fragment; used by `decode_masks_atlas` where only mask
/// presence matters.
fn generate_proto_mask_logit_shader_int8_nearest() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
precision highp isampler2DArray;

uniform isampler2DArray proto_tex;
uniform vec4 mask_coeff[8];
uniform int num_protos;
uniform float proto_scale;
uniform float coeff_sum_x_szp;

in vec2 tc;
out vec4 color;

void main() {
    ivec3 tex_size = textureSize(proto_tex, 0);
    int ix = clamp(int(tc.x * float(tex_size.x)), 0, tex_size.x - 1);
    int iy = clamp(int(tc.y * float(tex_size.y)), 0, tex_size.y - 1);

    int groups = (num_protos + 3) / 4;
    float acc = 0.0;
    for (int i = 0; i < groups; i++) {
        int base = i * 4;
        vec4 raw = vec4(
            float(texelFetch(proto_tex, ivec3(ix, iy, min(base, num_protos - 1)), 0).r),
            float(texelFetch(proto_tex, ivec3(ix, iy, min(base + 1, num_protos - 1)), 0).r),
            float(texelFetch(proto_tex, ivec3(ix, iy, min(base + 2, num_protos - 1)), 0).r),
            float(texelFetch(proto_tex, ivec3(ix, iy, min(base + 3, num_protos - 1)), 0).r)
        );
        acc += dot(mask_coeff[i], raw);
    }
    float logit = acc * proto_scale + coeff_sum_x_szp;
    float mask = logit > 0.0 ? 1.0 : 0.0;
    color = vec4(mask, 0.0, 0.0, 1.0);
}
"
}

/// Binary mask shader — int8, shader-based bilinear interpolation, logit threshold.
///
/// Outputs binary `acc > 0 ? 1.0 : 0.0` instead of `sigmoid(acc)`.  Used by
/// `decode_masks_atlas` for int8 models with bilinear interpolation.
fn generate_proto_mask_logit_shader_int8_bilinear() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
precision highp isampler2DArray;

uniform isampler2DArray proto_tex;
uniform vec4 mask_coeff[8];
uniform int num_protos;
uniform float proto_scale;
uniform float coeff_sum_x_szp;

in vec2 tc;
out vec4 color;

void main() {
    ivec3 tex_size = textureSize(proto_tex, 0);
    vec2 pos = tc * vec2(tex_size.xy) - 0.5;
    vec2 f = fract(pos);
    ivec2 p0 = ivec2(floor(pos));
    ivec2 p1 = p0 + 1;
    p0 = clamp(p0, ivec2(0), tex_size.xy - 1);
    p1 = clamp(p1, ivec2(0), tex_size.xy - 1);

    float w00 = (1.0 - f.x) * (1.0 - f.y);
    float w10 = f.x * (1.0 - f.y);
    float w01 = (1.0 - f.x) * f.y;
    float w11 = f.x * f.y;

    int groups = (num_protos + 3) / 4;
    float acc = 0.0;
    for (int i = 0; i < groups; i++) {
        int base = i * 4;
        int l0 = min(base, num_protos - 1);
        int l1 = min(base + 1, num_protos - 1);
        int l2 = min(base + 2, num_protos - 1);
        int l3 = min(base + 3, num_protos - 1);
        vec4 r00 = vec4(
            float(texelFetch(proto_tex, ivec3(p0.x, p0.y, l0), 0).r),
            float(texelFetch(proto_tex, ivec3(p0.x, p0.y, l1), 0).r),
            float(texelFetch(proto_tex, ivec3(p0.x, p0.y, l2), 0).r),
            float(texelFetch(proto_tex, ivec3(p0.x, p0.y, l3), 0).r)
        );
        vec4 r10 = vec4(
            float(texelFetch(proto_tex, ivec3(p1.x, p0.y, l0), 0).r),
            float(texelFetch(proto_tex, ivec3(p1.x, p0.y, l1), 0).r),
            float(texelFetch(proto_tex, ivec3(p1.x, p0.y, l2), 0).r),
            float(texelFetch(proto_tex, ivec3(p1.x, p0.y, l3), 0).r)
        );
        vec4 r01 = vec4(
            float(texelFetch(proto_tex, ivec3(p0.x, p1.y, l0), 0).r),
            float(texelFetch(proto_tex, ivec3(p0.x, p1.y, l1), 0).r),
            float(texelFetch(proto_tex, ivec3(p0.x, p1.y, l2), 0).r),
            float(texelFetch(proto_tex, ivec3(p0.x, p1.y, l3), 0).r)
        );
        vec4 r11 = vec4(
            float(texelFetch(proto_tex, ivec3(p1.x, p1.y, l0), 0).r),
            float(texelFetch(proto_tex, ivec3(p1.x, p1.y, l1), 0).r),
            float(texelFetch(proto_tex, ivec3(p1.x, p1.y, l2), 0).r),
            float(texelFetch(proto_tex, ivec3(p1.x, p1.y, l3), 0).r)
        );
        vec4 interp = r00 * w00 + r10 * w10 + r01 * w01 + r11 * w11;
        acc += dot(mask_coeff[i], interp);
    }
    float logit = acc * proto_scale + coeff_sum_x_szp;
    float mask = logit > 0.0 ? 1.0 : 0.0;
    color = vec4(mask, 0.0, 0.0, 1.0);
}
"
}

/// Binary mask shader — f32 protos with hardware bilinear filtering, logit threshold.
///
/// Outputs binary `acc > 0 ? 1.0 : 0.0` instead of `sigmoid(acc)`.  Used by
/// `decode_masks_atlas` for f32 models.
fn generate_proto_mask_logit_shader_f32() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp sampler2DArray;

uniform sampler2DArray proto_tex;
uniform vec4 mask_coeff[8];
uniform int num_protos;

in vec2 tc;
out vec4 color;

void main() {
    int groups = (num_protos + 3) / 4;
    float acc = 0.0;
    for (int i = 0; i < groups; i++) {
        int base = i * 4;
        vec4 val = vec4(
            texture(proto_tex, vec3(tc, float(min(base, num_protos - 1)))).r,
            texture(proto_tex, vec3(tc, float(min(base + 1, num_protos - 1)))).r,
            texture(proto_tex, vec3(tc, float(min(base + 2, num_protos - 1)))).r,
            texture(proto_tex, vec3(tc, float(min(base + 3, num_protos - 1)))).r
        );
        acc += dot(mask_coeff[i], val);
    }
    float mask = acc > 0.0 ? 1.0 : 0.0;
    color = vec4(mask, 0.0, 0.0, 1.0);
}
"
}

fn generate_color_shader() -> &'static str {
    "\
#version 300 es
precision mediump float;
uniform vec4 colors[20];
uniform int class_index;

out vec4 color;
void main() {
    int index = class_index % 20;
    color = colors[index];
}
"
}

/// Packed RGB -> RGBA8 packing shader (2D texture source, pass 2).
///
/// Reads from an intermediate RGBA texture and packs 3 RGB channels into
/// RGBA8 output pixels. Each output pixel stores 4 consecutive bytes of the
/// destination RGB buffer. Uses only 2 texture fetches per fragment (down
/// from 4) by exploiting the fact that 4 consecutive bytes span at most 2
/// source pixels.
fn generate_packed_rgba8_shader_2d() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
uniform sampler2D tex;
out vec4 color;
void main() {
    // gl_FragCoord is at pixel center (n+0.5). Use floor() for robust
    // integer pixel index on all GPUs (Vivante, Mali, Adreno).
    int out_x = int(floor(gl_FragCoord.x));
    int out_y = int(floor(gl_FragCoord.y));
    int base = out_x * 4;
    // 4 consecutive byte indices map to at most 2 source pixels
    int px0 = base / 3;
    int px1 = (base + 3) / 3;
    vec4 s0 = texelFetch(tex, ivec2(px0, out_y), 0);
    vec4 s1 = (px1 != px0) ? texelFetch(tex, ivec2(px1, out_y), 0) : s0;
    // Extract channels based on phase (base % 3)
    int phase = base - px0 * 3;
    if (phase == 0) {
        color = vec4(s0.r, s0.g, s0.b, s1.r);
    } else if (phase == 1) {
        color = vec4(s0.g, s0.b, s1.r, s1.g);
    } else {
        color = vec4(s0.b, s1.r, s1.g, s1.b);
    }
}
"
}

/// Packed RGB -> RGBA8 packing shader with int8 XOR 0x80 bias (2D source, pass 2).
///
/// Same packing logic as [`generate_packed_rgba8_shader_2d`] but applies
/// bit-exact XOR 0x80 bias via quantize+mod: `floor(v * 255 + 0.5) + 128
/// mod 256 / 255`. This matches the CPU `byte ^ 0x80` operation exactly.
fn generate_packed_rgba8_int8_shader_2d() -> &'static str {
    "\
#version 300 es
precision highp float;
precision highp int;
uniform sampler2D tex;
out vec4 color;

vec4 int8_bias(vec4 v) {
    vec4 q = floor(v * 255.0 + 0.5);
    return mod(q + 128.0, 256.0) / 255.0;
}

void main() {
    // gl_FragCoord is at pixel center (n+0.5). Use floor() for robust
    // integer pixel index on all GPUs (Vivante, Mali, Adreno).
    int out_x = int(floor(gl_FragCoord.x));
    int out_y = int(floor(gl_FragCoord.y));
    int base = out_x * 4;
    // 4 consecutive byte indices map to at most 2 source pixels
    int px0 = base / 3;
    int px1 = (base + 3) / 3;
    vec4 s0 = texelFetch(tex, ivec2(px0, out_y), 0);
    vec4 s1 = (px1 != px0) ? texelFetch(tex, ivec2(px1, out_y), 0) : s0;
    // Extract channels based on phase (base % 3), then apply int8 bias
    int phase = base - px0 * 3;
    if (phase == 0) {
        color = int8_bias(vec4(s0.r, s0.g, s0.b, s1.r));
    } else if (phase == 1) {
        color = int8_bias(vec4(s0.g, s0.b, s1.r, s1.g));
    } else {
        color = int8_bias(vec4(s0.b, s1.r, s1.g, s1.b));
    }
}
"
}

#[cfg(test)]
#[cfg(feature = "opengl")]
mod gl_tests {
    use super::*;
    use crate::{TensorImage, BGRA, RGBA};
    #[cfg(feature = "dma_test_formats")]
    use crate::{NV12, YUYV};
    use edgefirst_tensor::{TensorMapTrait, TensorTrait};
    #[cfg(feature = "dma_test_formats")]
    use edgefirst_tensor::{is_dma_available, TensorMemory};
    use image::buffer::ConvertBuffer;
    use ndarray::Array3;

    #[test]
    fn test_segmentation() {
        use edgefirst_decoder::Segmentation;

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let mut image = TensorImage::load(
            include_bytes!("../../../testdata/giraffe.jpg"),
            Some(RGBA),
            None,
        )
        .unwrap();

        let mut segmentation = Array3::from_shape_vec(
            (2, 160, 160),
            include_bytes!("../../../testdata/modelpack_seg_2x160x160.bin").to_vec(),
        )
        .unwrap();
        segmentation.swap_axes(0, 1);
        segmentation.swap_axes(1, 2);
        let segmentation = segmentation.as_standard_layout().to_owned();

        let seg = Segmentation {
            segmentation,
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
        };

        let mut renderer = GLProcessorThreaded::new(None).unwrap();
        renderer.draw_masks(&mut image, &[], &[seg]).unwrap();
    }

    #[test]
    fn test_segmentation_mem() {
        use edgefirst_decoder::Segmentation;

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let mut image = TensorImage::load(
            include_bytes!("../../../testdata/giraffe.jpg"),
            Some(RGBA),
            Some(edgefirst_tensor::TensorMemory::Mem),
        )
        .unwrap();

        let mut segmentation = Array3::from_shape_vec(
            (2, 160, 160),
            include_bytes!("../../../testdata/modelpack_seg_2x160x160.bin").to_vec(),
        )
        .unwrap();
        segmentation.swap_axes(0, 1);
        segmentation.swap_axes(1, 2);
        let segmentation = segmentation.as_standard_layout().to_owned();

        let seg = Segmentation {
            segmentation,
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
        };

        let mut renderer = GLProcessorThreaded::new(None).unwrap();
        renderer.draw_masks(&mut image, &[], &[seg]).unwrap();
    }

    #[test]
    fn test_segmentation_yolo() {
        use edgefirst_decoder::Segmentation;
        use ndarray::Array3;

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let mut image = TensorImage::load(
            include_bytes!("../../../testdata/giraffe.jpg"),
            Some(RGBA),
            None,
        )
        .unwrap();

        let segmentation = Array3::from_shape_vec(
            (76, 55, 1),
            include_bytes!("../../../testdata/yolov8_seg_crop_76x55.bin").to_vec(),
        )
        .unwrap();

        let detect = DetectBox {
            bbox: [0.59375, 0.25, 0.9375, 0.725].into(),
            score: 0.99,
            label: 1,
        };

        let seg = Segmentation {
            segmentation,
            xmin: 0.59375,
            ymin: 0.25,
            xmax: 0.9375,
            ymax: 0.725,
        };

        let mut renderer = GLProcessorThreaded::new(None).unwrap();
        renderer
            .set_class_colors(&[[255, 255, 0, 233], [128, 128, 255, 100]])
            .unwrap();
        renderer.draw_masks(&mut image, &[detect], &[seg]).unwrap();

        let expected = TensorImage::load(
            include_bytes!("../../../testdata/output_render_gl.jpg"),
            Some(RGBA),
            None,
        )
        .unwrap();

        compare_images(&image, &expected, 0.99, function!());
    }

    #[test]
    fn test_boxes() {
        use edgefirst_decoder::DetectBox;

        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let mut image = TensorImage::load(
            include_bytes!("../../../testdata/giraffe.jpg"),
            Some(RGBA),
            None,
        )
        .unwrap();

        let detect = DetectBox {
            bbox: [0.59375, 0.25, 0.9375, 0.725].into(),
            score: 0.99,
            label: 0,
        };
        let mut renderer = GLProcessorThreaded::new(None).unwrap();
        renderer
            .set_class_colors(&[[255, 255, 0, 233], [128, 128, 255, 100]])
            .unwrap();
        renderer.draw_masks(&mut image, &[detect], &[]).unwrap();
    }

    static GL_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Helper function to check if OpenGL is available
    fn is_opengl_available() -> bool {
        #[cfg(all(target_os = "linux", feature = "opengl"))]
        {
            *GL_AVAILABLE.get_or_init(|| GLProcessorThreaded::new(None).is_ok())
        }

        #[cfg(not(all(target_os = "linux", feature = "opengl")))]
        {
            false
        }
    }

    fn compare_images(img1: &TensorImage, img2: &TensorImage, threshold: f64, name: &str) {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");
        assert_eq!(img1.fourcc(), img2.fourcc(), "FourCC differ");
        assert!(
            matches!(img1.fourcc(), RGB | RGBA | GREY | PLANAR_RGB),
            "FourCC must be RGB or RGBA for comparison"
        );

        let image1 = match img1.fourcc() {
            RGB => image::RgbImage::from_vec(
                img1.width() as u32,
                img1.height() as u32,
                img1.tensor().map().unwrap().to_vec(),
            )
            .unwrap(),
            RGBA => image::RgbaImage::from_vec(
                img1.width() as u32,
                img1.height() as u32,
                img1.tensor().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            GREY => image::GrayImage::from_vec(
                img1.width() as u32,
                img1.height() as u32,
                img1.tensor().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            PLANAR_RGB => image::GrayImage::from_vec(
                img1.width() as u32,
                (img1.height() * 3) as u32,
                img1.tensor().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            _ => return,
        };

        let image2 = match img2.fourcc() {
            RGB => image::RgbImage::from_vec(
                img2.width() as u32,
                img2.height() as u32,
                img2.tensor().map().unwrap().to_vec(),
            )
            .unwrap(),
            RGBA => image::RgbaImage::from_vec(
                img2.width() as u32,
                img2.height() as u32,
                img2.tensor().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            GREY => image::GrayImage::from_vec(
                img2.width() as u32,
                img2.height() as u32,
                img2.tensor().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            PLANAR_RGB => image::GrayImage::from_vec(
                img2.width() as u32,
                (img2.height() * 3) as u32,
                img2.tensor().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),
            _ => return,
        };

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &image1,
            &image2,
        )
        .expect("Image Comparison failed");
        if similarity.score < threshold {
            // image1.save(format!("{name}_1.png"));
            // image2.save(format!("{name}_2.png"));
            similarity
                .image
                .to_color_map()
                .save(format!("{name}.png"))
                .unwrap();
            panic!(
                "{name}: converted image and target image have similarity score too low: {} < {}",
                similarity.score, threshold
            )
        }
    }

    // =========================================================================
    // NV12 Reference Validation Tests
    // These tests compare OpenGL NV12 conversions against ffmpeg-generated
    // references
    // =========================================================================

    #[cfg(feature = "dma_test_formats")]
    fn load_raw_image(
        width: usize,
        height: usize,
        fourcc: FourCharCode,
        memory: Option<TensorMemory>,
        bytes: &[u8],
    ) -> Result<TensorImage, crate::Error> {
        let img = TensorImage::new(width, height, fourcc, memory)?;
        let mut map = img.tensor().map()?;
        map.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        Ok(img)
    }

    /// Test OpenGL NV12→RGBA conversion against ffmpeg reference
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_nv12_to_rgba_reference() {
        if !is_dma_available() {
            return;
        }
        // Load NV12 source with DMA
        let src = load_raw_image(
            1280,
            720,
            NV12,
            Some(TensorMemory::Dma),
            include_bytes!("../../../testdata/camera720p.nv12"),
        )
        .unwrap();

        // Load RGBA reference (ffmpeg-generated)
        let reference = load_raw_image(
            1280,
            720,
            RGBA,
            None,
            include_bytes!("../../../testdata/camera720p.rgba"),
        )
        .unwrap();

        // Convert using OpenGL
        let mut dst = TensorImage::new(1280, 720, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        gl.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        // Copy to CPU for comparison
        let cpu_dst = TensorImage::new(1280, 720, RGBA, None).unwrap();
        cpu_dst
            .tensor()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(dst.tensor().map().unwrap().as_slice());

        compare_images(&reference, &cpu_dst, 0.98, "opengl_nv12_to_rgba_reference");
    }

    /// Test OpenGL YUYV→RGBA conversion against ffmpeg reference
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_yuyv_to_rgba_reference() {
        if !is_dma_available() {
            return;
        }
        // Load YUYV source with DMA
        let src = load_raw_image(
            1280,
            720,
            YUYV,
            Some(TensorMemory::Dma),
            include_bytes!("../../../testdata/camera720p.yuyv"),
        )
        .unwrap();

        // Load RGBA reference (ffmpeg-generated)
        let reference = load_raw_image(
            1280,
            720,
            RGBA,
            None,
            include_bytes!("../../../testdata/camera720p.rgba"),
        )
        .unwrap();

        // Convert using OpenGL
        let mut dst = TensorImage::new(1280, 720, RGBA, Some(TensorMemory::Dma)).unwrap();
        let mut gl = GLProcessorThreaded::new(None).unwrap();
        gl.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .unwrap();

        // Copy to CPU for comparison
        let cpu_dst = TensorImage::new(1280, 720, RGBA, None).unwrap();
        cpu_dst
            .tensor()
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(dst.tensor().map().unwrap().as_slice());

        compare_images(&reference, &cpu_dst, 0.98, "opengl_yuyv_to_rgba_reference");
    }

    // =========================================================================
    // EGL Display Probe & Override Tests
    // =========================================================================

    /// Validate that probe_egl_displays() discovers available display types
    /// and returns them in priority order (GBM first).
    ///
    /// On headless i.MX hardware, GBM and PlatformDevice are typically
    /// available. Default requires a running compositor (Wayland/X11) and
    /// may not be present on headless targets.
    #[test]
    fn test_probe_egl_displays() {
        let displays = match probe_egl_displays() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED: {} - EGL not available: {e:?}", function!());
                return;
            }
        };

        if displays.is_empty() {
            eprintln!("SKIPPED: {} - No EGL displays available", function!());
            return;
        }

        let kinds: Vec<_> = displays.iter().map(|d| d.kind).collect();
        eprintln!("Probed EGL displays: {kinds:?}");
        for d in &displays {
            eprintln!("  {:?}: {}", d.kind, d.description);
        }

        // Verify priority ordering: PlatformDevice > GBM > Default.
        // Not all display types are available on every system, but the
        // ones that are present must appear in this order.
        let priority = |k: &EglDisplayKind| match k {
            EglDisplayKind::PlatformDevice => 0,
            EglDisplayKind::Gbm => 1,
            EglDisplayKind::Default => 2,
        };
        for w in kinds.windows(2) {
            assert!(
                priority(&w[0]) < priority(&w[1]),
                "Display ordering violated: {:?} should come after {:?}",
                w[1],
                w[0],
            );
        }
    }

    /// Validate that explicitly selecting each available display kind via
    /// GLProcessorThreaded::new(Some(kind)) succeeds and produces a working
    /// converter.
    #[test]
    fn test_override_each_display_kind() {
        let displays = match probe_egl_displays() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED: {} - EGL not available: {e:?}", function!());
                return;
            }
        };

        if displays.is_empty() {
            eprintln!("SKIPPED: {} - No EGL displays available", function!());
            return;
        }

        for display in &displays {
            eprintln!(
                "Testing override: {:?} ({})",
                display.kind, display.description
            );
            let mut gl = GLProcessorThreaded::new(Some(display.kind)).unwrap_or_else(|e| {
                panic!(
                    "GLProcessorThreaded::new(Some({:?})) failed: {e:?}",
                    display.kind
                )
            });

            // Smoke test: do a simple RGBA → RGBA conversion to verify the
            // GL context is fully functional.
            let src = TensorImage::load(
                include_bytes!("../../../testdata/zidane.jpg"),
                Some(RGBA),
                None,
            )
            .unwrap();
            let mut dst = TensorImage::new(320, 240, RGBA, None).unwrap();
            gl.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
                .unwrap_or_else(|e| {
                    panic!("convert() with {:?} display failed: {e:?}", display.kind)
                });
            eprintln!("  {:?} display: convert OK", display.kind);
        }
    }

    /// Validate that requesting a display kind that doesn't exist on the
    /// system returns an error rather than falling back silently.
    #[test]
    fn test_override_unavailable_display_errors() {
        let displays = match probe_egl_displays() {
            Ok(d) => d,
            Err(e) => {
                eprintln!("SKIPPED: {} - EGL not available: {e:?}", function!());
                return;
            }
        };
        let available_kinds: Vec<_> = displays.iter().map(|d| d.kind).collect();

        // Find a kind that is NOT available; if all three are available,
        // this test has nothing to verify — skip it.
        let unavailable = [
            EglDisplayKind::PlatformDevice,
            EglDisplayKind::Gbm,
            EglDisplayKind::Default,
        ]
        .into_iter()
        .find(|k| !available_kinds.contains(k));

        if let Some(kind) = unavailable {
            eprintln!("Testing override with unavailable kind: {kind:?}");
            let result = GLProcessorThreaded::new(Some(kind));
            assert!(
                result.is_err(),
                "Expected error for unavailable display kind {kind:?}, got Ok"
            );
            eprintln!("  Correctly returned error: {:?}", result.unwrap_err());
        } else {
            eprintln!(
                "SKIPPED: {} - All three display kinds are available",
                function!()
            );
        }
    }

    /// Validate that auto-detection (None) still works — this is the existing
    /// default behaviour and must not regress.
    #[test]
    fn test_auto_detect_display() {
        if !is_opengl_available() {
            eprintln!("SKIPPED: {} - OpenGL not available", function!());
            return;
        }

        let mut gl = GLProcessorThreaded::new(None).expect("auto-detect should succeed");
        let src = TensorImage::load(
            include_bytes!("../../../testdata/zidane.jpg"),
            Some(RGBA),
            None,
        )
        .unwrap();
        let mut dst = TensorImage::new(320, 240, RGBA, None).unwrap();
        gl.convert(&src, &mut dst, Rotation::None, Flip::None, Crop::no_crop())
            .expect("auto-detect convert should succeed");
    }

    #[test]
    fn test_packed_rgb_width_constraint() {
        // Standard ML model input widths — all satisfy W*3 % 4 == 0
        assert_eq!((640usize * 3) % 4, 0);
        assert_eq!((320usize * 3) % 4, 0);
        assert_eq!((1280usize * 3) % 4, 0);

        // Non-divisible widths should be rejected
        assert_ne!((322usize * 3) % 4, 0);
        assert_ne!((333usize * 3) % 4, 0);
    }

    // =========================================================================
    // Packed RGB Correctness Tests (two-pass pipeline)
    // These tests compare GL RGBA output (alpha stripped) against GL packed
    // RGB output. Both use the same GPU color conversion, so differences
    // isolate packing shader bugs rather than CPU-vs-GPU YUV conversion.
    // They require DMA + OpenGL hardware (on-target only).
    // =========================================================================

    /// Compare two byte slices pixel-by-pixel with tolerance.
    /// Panics with details if any byte differs by more than `tolerance`.
    #[cfg(feature = "dma_test_formats")]
    fn assert_pixels_match(expected: &[u8], actual: &[u8], tolerance: u8) {
        assert_eq!(expected.len(), actual.len(), "Buffer size mismatch");
        let mut max_diff: u8 = 0;
        let mut diff_count: usize = 0;
        let mut first_diff_idx = None;
        for (i, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
            let diff = (e as i16 - a as i16).unsigned_abs() as u8;
            if diff > tolerance {
                diff_count += 1;
                if first_diff_idx.is_none() {
                    first_diff_idx = Some(i);
                }
            }
            max_diff = max_diff.max(diff);
        }
        assert!(
            diff_count == 0,
            "Pixel mismatch: {diff_count} bytes differ (max_diff={max_diff}, first at index {})",
            first_diff_idx.unwrap_or(0)
        );
    }

    /// Build a letterbox crop that fits src into dst_w x dst_h, preserving aspect ratio.
    #[cfg(feature = "dma_test_formats")]
    fn letterbox_crop(src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Crop {
        let src_aspect = src_w as f64 / src_h as f64;
        let dst_aspect = dst_w as f64 / dst_h as f64;
        let (new_w, new_h) = if src_aspect > dst_aspect {
            let new_h = (dst_w as f64 / src_aspect).round() as usize;
            (dst_w, new_h)
        } else {
            let new_w = (dst_h as f64 * src_aspect).round() as usize;
            (new_w, dst_h)
        };
        let left = (dst_w - new_w) / 2;
        let top = (dst_h - new_h) / 2;
        Crop::new()
            .with_dst_rect(Some(crate::Rect::new(left, top, new_w, new_h)))
            .with_dst_color(Some([114, 114, 114, 255]))
    }

    /// Strip alpha from RGBA bytes → packed RGB bytes.
    #[cfg(feature = "dma_test_formats")]
    fn rgba_to_rgb(rgba: &[u8]) -> Vec<u8> {
        assert_eq!(
            rgba.len() % 4,
            0,
            "RGBA buffer length must be divisible by 4"
        );
        let mut rgb = Vec::with_capacity(rgba.len() / 4 * 3);
        for pixel in rgba.chunks_exact(4) {
            rgb.push(pixel[0]);
            rgb.push(pixel[1]);
            rgb.push(pixel[2]);
        }
        rgb
    }

    /// Convert uint8 RGB bytes to int8 (XOR 0x80 each byte).
    #[cfg(feature = "dma_test_formats")]
    fn uint8_to_int8(data: &[u8]) -> Vec<u8> {
        data.iter().map(|&b| b ^ 0x80).collect()
    }

    /// YUYV 1080p → RGB 640x640 with letterbox (two-pass packed RGB pipeline).
    /// Compares GL RGBA (alpha-stripped) against GL packed RGB to validate packing.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_rgb_correctness() {
        if !is_dma_available() {
            return;
        }
        let src_dma = load_raw_image(
            1920,
            1080,
            YUYV,
            Some(TensorMemory::Dma),
            include_bytes!("../../../testdata/camera1080p.yuyv"),
        )
        .unwrap();

        let crop = letterbox_crop(1920, 1080, 640, 640);
        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // GL RGBA reference
        let mut dst_rgba = TensorImage::new(640, 640, RGBA, Some(TensorMemory::Dma)).unwrap();
        gl.convert(&src_dma, &mut dst_rgba, Rotation::None, Flip::None, crop)
            .unwrap();

        // GL packed RGB output
        let mut dst_rgb = TensorImage::new(640, 640, RGB, Some(TensorMemory::Dma)).unwrap();
        gl.convert(&src_dma, &mut dst_rgb, Rotation::None, Flip::None, crop)
            .unwrap();

        let rgba_data = dst_rgba.tensor().map().unwrap();
        let expected_rgb = rgba_to_rgb(rgba_data.as_slice());
        let gl_data = dst_rgb.tensor().map().unwrap();
        assert_pixels_match(&expected_rgb, gl_data.as_slice(), 1);
    }

    /// YUYV 1080p → RGB_INT8 640x640 with letterbox.
    /// Compares GL RGBA (alpha-stripped, XOR 0x80) against GL packed RGB_INT8.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_rgb_int8_correctness() {
        if !is_dma_available() {
            return;
        }
        let src_dma = load_raw_image(
            1920,
            1080,
            YUYV,
            Some(TensorMemory::Dma),
            include_bytes!("../../../testdata/camera1080p.yuyv"),
        )
        .unwrap();

        let crop = letterbox_crop(1920, 1080, 640, 640);
        // Use GLProcessorST with direct RGB disabled to validate two-pass int8
        // pipeline against RGBA reference. The direct path renders to a different
        // framebuffer format (RGB8 renderbuffer vs RGBA8 texture) which produces
        // different YUV interpolation results; it is validated separately by
        // test_opengl_rgb_direct_matches_two_pass.
        let mut gl = match GLProcessorST::new(None) {
            Ok(gl) => gl,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        gl.support_rgb_direct = false;

        // GL RGBA reference
        let mut dst_rgba = TensorImage::new(640, 640, RGBA, Some(TensorMemory::Dma)).unwrap();
        gl.convert(&src_dma, &mut dst_rgba, Rotation::None, Flip::None, crop)
            .unwrap();

        // GL packed RGB_INT8 output (two-pass path)
        let mut dst_rgb = TensorImage::new(640, 640, RGB_INT8, Some(TensorMemory::Dma)).unwrap();
        gl.convert(&src_dma, &mut dst_rgb, Rotation::None, Flip::None, crop)
            .unwrap();

        let rgba_data = dst_rgba.tensor().map().unwrap();
        let expected_rgb = uint8_to_int8(&rgba_to_rgb(rgba_data.as_slice()));
        let gl_data = dst_rgb.tensor().map().unwrap();
        assert_pixels_match(&expected_rgb, gl_data.as_slice(), 1);
    }

    /// YUYV 1080p → RGB 1920x1080 (no letterbox, same size).
    /// Compares GL RGBA (alpha-stripped) against GL packed RGB without scaling.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_rgb_no_letterbox_correctness() {
        if !is_dma_available() {
            return;
        }
        let src_dma = load_raw_image(
            1920,
            1080,
            YUYV,
            Some(TensorMemory::Dma),
            include_bytes!("../../../testdata/camera1080p.yuyv"),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // GL RGBA reference (no letterbox — 1920 satisfies W*3 % 4 == 0)
        let mut dst_rgba = TensorImage::new(1920, 1080, RGBA, Some(TensorMemory::Dma)).unwrap();
        gl.convert(
            &src_dma,
            &mut dst_rgba,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // GL packed RGB output
        let mut dst_rgb = TensorImage::new(1920, 1080, RGB, Some(TensorMemory::Dma)).unwrap();
        gl.convert(
            &src_dma,
            &mut dst_rgb,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        let rgba_data = dst_rgba.tensor().map().unwrap();
        let expected_rgb = rgba_to_rgb(rgba_data.as_slice());
        let gl_data = dst_rgb.tensor().map().unwrap();
        assert_pixels_match(&expected_rgb, gl_data.as_slice(), 1);
    }

    // =========================================================================
    // Direct RGB Render Path Tests
    // These tests exercise the single-pass BGR888 renderbuffer path added by
    // the GL cache work (EDGEAI-776). They require DMA + OpenGL hardware.
    // =========================================================================

    /// Verify that the direct RGB probe runs without crashing.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_probe_rgb_direct_support() {
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let gl = match GLProcessorST::new(None) {
            Ok(gl) => gl,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        // The probe runs during new(). Just check the field is set.
        eprintln!(
            "support_rgb_direct = {} (probe completed without crash)",
            gl.support_rgb_direct
        );
    }

    /// Compare direct RGB path against two-pass path pixel-for-pixel.
    /// If GPU doesn't support direct RGB, this test is a no-op.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_rgb_direct_matches_two_pass() {
        if !is_dma_available() {
            eprintln!("SKIPPED: {} - DMA not available", function!());
            return;
        }
        let mut gl = match GLProcessorST::new(None) {
            Ok(gl) => gl,
            Err(e) => {
                eprintln!("SKIPPED: {} - GL not available: {e}", function!());
                return;
            }
        };
        if !gl.support_rgb_direct {
            eprintln!("SKIPPED: {} - GPU does not support direct RGB", function!());
            return;
        }

        // Create RGBA source with deterministic pattern
        // Use 640x480 source → 320x320 output so pitch (320*3=960) is 64-byte aligned
        // for Mali GPU DMA-buf import requirements.
        let src = TensorImage::new(640, 480, RGBA, Some(TensorMemory::Dma)).unwrap();
        {
            let mut map = src.tensor().map().unwrap();
            for (i, byte) in map.as_mut_slice().iter_mut().enumerate() {
                *byte = (i % 251) as u8; // deterministic pattern
            }
        }

        let crop = crate::Crop {
            src_rect: None,
            dst_rect: None,
            dst_color: None,
        };

        // Direct path (support_rgb_direct = true)
        let mut dst_direct = TensorImage::new(320, 320, RGB, Some(TensorMemory::Dma)).unwrap();
        gl.convert(&src, &mut dst_direct, Rotation::None, Flip::None, crop)
            .unwrap();

        // Force two-pass path
        gl.support_rgb_direct = false;
        let mut dst_twop = TensorImage::new(320, 320, RGB, Some(TensorMemory::Dma)).unwrap();
        gl.convert(&src, &mut dst_twop, Rotation::None, Flip::None, crop)
            .unwrap();
        gl.support_rgb_direct = true;

        // Compare
        let map_direct = dst_direct.tensor().map().unwrap();
        let map_twop = dst_twop.tensor().map().unwrap();
        // Allow ±1 tolerance for potential rounding differences
        let mut max_diff = 0i32;
        for (a, b) in map_direct.as_slice().iter().zip(map_twop.as_slice().iter()) {
            let diff = (*a as i32 - *b as i32).abs();
            max_diff = max_diff.max(diff);
        }
        eprintln!("RGB direct vs two-pass max pixel diff: {max_diff}");
        assert!(max_diff <= 1, "Pixel mismatch > 1: max_diff={max_diff}");
    }

    // ---- BGRA destination tests ----

    /// Test OpenGL NV12→BGRA conversion with DMA buffers.
    /// Compares against NV12→RGBA by verifying R↔B swap.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_nv12_to_bgra() {
        if !is_dma_available() {
            eprintln!("SKIPPED: test_opengl_nv12_to_bgra - DMA not available");
            return;
        }

        let src = load_raw_image(
            1280,
            720,
            NV12,
            Some(TensorMemory::Dma),
            include_bytes!("../../../testdata/camera720p.nv12"),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Convert to RGBA as reference
        let mut rgba_dst =
            TensorImage::new(1280, 720, RGBA, Some(TensorMemory::Dma)).unwrap();
        gl.convert(
            &src,
            &mut rgba_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Convert to BGRA
        let mut bgra_dst =
            TensorImage::new(1280, 720, BGRA, Some(TensorMemory::Dma)).unwrap();
        gl.convert(
            &src,
            &mut bgra_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        // Compare: BGRA[B,G,R,A] should match RGBA[R,G,B,A] with R↔B swapped
        let bgra_map = bgra_dst.tensor().map().unwrap();
        let rgba_map = rgba_dst.tensor().map().unwrap();
        let bgra_buf = bgra_map.as_slice();
        let rgba_buf = rgba_map.as_slice();

        assert_eq!(bgra_buf.len(), rgba_buf.len());
        let mut max_diff = 0i32;
        for (bc, rc) in bgra_buf.chunks_exact(4).zip(rgba_buf.chunks_exact(4)) {
            max_diff = max_diff.max((bc[0] as i32 - rc[2] as i32).abs()); // B
            max_diff = max_diff.max((bc[1] as i32 - rc[1] as i32).abs()); // G
            max_diff = max_diff.max((bc[2] as i32 - rc[0] as i32).abs()); // R
            max_diff = max_diff.max((bc[3] as i32 - rc[3] as i32).abs()); // A
        }
        eprintln!("NV12→BGRA vs NV12→RGBA max channel diff: {max_diff}");
        assert!(
            max_diff <= 1,
            "BGRA/RGBA channel mismatch > 1: max_diff={max_diff}"
        );
    }

    /// Test OpenGL YUYV→BGRA conversion with DMA buffers.
    #[test]
    #[cfg(all(target_os = "linux", feature = "dma_test_formats"))]
    fn test_opengl_yuyv_to_bgra() {
        if !is_dma_available() {
            eprintln!("SKIPPED: test_opengl_yuyv_to_bgra - DMA not available");
            return;
        }

        let src = load_raw_image(
            1280,
            720,
            YUYV,
            Some(TensorMemory::Dma),
            include_bytes!("../../../testdata/camera720p.yuyv"),
        )
        .unwrap();

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        let mut rgba_dst =
            TensorImage::new(1280, 720, RGBA, Some(TensorMemory::Dma)).unwrap();
        gl.convert(
            &src,
            &mut rgba_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        let mut bgra_dst =
            TensorImage::new(1280, 720, BGRA, Some(TensorMemory::Dma)).unwrap();
        gl.convert(
            &src,
            &mut bgra_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();

        let bgra_map = bgra_dst.tensor().map().unwrap();
        let rgba_map = rgba_dst.tensor().map().unwrap();
        let bgra_buf = bgra_map.as_slice();
        let rgba_buf = rgba_map.as_slice();

        let mut max_diff = 0i32;
        for (bc, rc) in bgra_buf.chunks_exact(4).zip(rgba_buf.chunks_exact(4)) {
            max_diff = max_diff.max((bc[0] as i32 - rc[2] as i32).abs());
            max_diff = max_diff.max((bc[1] as i32 - rc[1] as i32).abs());
            max_diff = max_diff.max((bc[2] as i32 - rc[0] as i32).abs());
            max_diff = max_diff.max((bc[3] as i32 - rc[3] as i32).abs());
        }
        eprintln!("YUYV→BGRA vs YUYV→RGBA max channel diff: {max_diff}");
        assert!(
            max_diff <= 1,
            "BGRA/RGBA channel mismatch > 1: max_diff={max_diff}"
        );
    }

    /// Test draw_masks() with BGRA destination (segmentation).
    /// Draws the same masks to both RGBA and BGRA, then verifies R↔B swap.
    #[test]
    fn test_draw_masks_bgra() {
        use edgefirst_decoder::Segmentation;

        if !is_opengl_available() {
            eprintln!("SKIPPED: test_draw_masks_bgra - OpenGL not available");
            return;
        }

        let seg_bytes =
            include_bytes!("../../../testdata/modelpack_seg_2x160x160.bin").to_vec();

        // Build segmentation data (shared between both renders)
        let make_seg = || {
            let mut s = Array3::from_shape_vec((2, 160, 160), seg_bytes.clone()).unwrap();
            s.swap_axes(0, 1);
            s.swap_axes(1, 2);
            let s = s.as_standard_layout().to_owned();
            Segmentation {
                segmentation: s,
                xmin: 0.0,
                ymin: 0.0,
                xmax: 1.0,
                ymax: 1.0,
            }
        };

        let mut gl = GLProcessorThreaded::new(None).unwrap();

        // Render to RGBA
        let mut rgba_img = TensorImage::load(
            include_bytes!("../../../testdata/giraffe.jpg"),
            Some(RGBA),
            None,
        )
        .unwrap();
        gl.draw_masks(&mut rgba_img, &[], &[make_seg()]).unwrap();

        // Render to BGRA (convert source to BGRA first)
        let rgba_src = TensorImage::load(
            include_bytes!("../../../testdata/giraffe.jpg"),
            Some(RGBA),
            None,
        )
        .unwrap();
        let mut bgra_img =
            TensorImage::new(rgba_src.width(), rgba_src.height(), BGRA, None).unwrap();
        gl.convert(
            &rgba_src,
            &mut bgra_img,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();
        gl.draw_masks(&mut bgra_img, &[], &[make_seg()]).unwrap();

        // Verify BGRA output matches RGBA output with R↔B swapped
        let rgba_map = rgba_img.tensor().map().unwrap();
        let bgra_map = bgra_img.tensor().map().unwrap();
        let rgba_buf = rgba_map.as_slice();
        let bgra_buf = bgra_map.as_slice();
        assert_eq!(rgba_buf.len(), bgra_buf.len());

        let mut max_diff = 0i32;
        for (rc, bc) in rgba_buf.chunks_exact(4).zip(bgra_buf.chunks_exact(4)) {
            max_diff = max_diff.max((rc[0] as i32 - bc[2] as i32).abs()); // R
            max_diff = max_diff.max((rc[1] as i32 - bc[1] as i32).abs()); // G
            max_diff = max_diff.max((rc[2] as i32 - bc[0] as i32).abs()); // B
            max_diff = max_diff.max((rc[3] as i32 - bc[3] as i32).abs()); // A
        }
        eprintln!("draw_masks BGRA vs RGBA max channel diff: {max_diff}");
        assert!(
            max_diff <= 1,
            "draw_masks BGRA/RGBA channel mismatch > 1: max_diff={max_diff}"
        );
    }

    /// Test draw_masks() with BGRA destination using Mem memory (boxes).
    /// Draws same boxes to RGBA and BGRA, then verifies R↔B swap.
    #[test]
    fn test_draw_masks_bgra_mem() {
        use edgefirst_decoder::DetectBox;

        if !is_opengl_available() {
            eprintln!("SKIPPED: test_draw_masks_bgra_mem - OpenGL not available");
            return;
        }

        let detect = DetectBox {
            bbox: [0.59375, 0.25, 0.9375, 0.725].into(),
            score: 0.99,
            label: 0,
        };
        let colors = [[255, 255, 0, 233], [128, 128, 255, 100]];

        let mut gl = GLProcessorThreaded::new(None).unwrap();
        gl.set_class_colors(&colors).unwrap();

        // Render boxes to RGBA
        let mut rgba_img = TensorImage::load(
            include_bytes!("../../../testdata/giraffe.jpg"),
            Some(RGBA),
            Some(edgefirst_tensor::TensorMemory::Mem),
        )
        .unwrap();
        gl.draw_masks(&mut rgba_img, &[detect], &[]).unwrap();

        // Render boxes to BGRA
        let rgba_src = TensorImage::load(
            include_bytes!("../../../testdata/giraffe.jpg"),
            Some(RGBA),
            Some(edgefirst_tensor::TensorMemory::Mem),
        )
        .unwrap();
        let mut bgra_img = TensorImage::new(
            rgba_src.width(),
            rgba_src.height(),
            BGRA,
            Some(edgefirst_tensor::TensorMemory::Mem),
        )
        .unwrap();
        gl.convert(
            &rgba_src,
            &mut bgra_img,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )
        .unwrap();
        gl.draw_masks(&mut bgra_img, &[detect], &[]).unwrap();

        // Verify BGRA output matches RGBA output with R↔B swapped
        let rgba_map = rgba_img.tensor().map().unwrap();
        let bgra_map = bgra_img.tensor().map().unwrap();
        let rgba_buf = rgba_map.as_slice();
        let bgra_buf = bgra_map.as_slice();

        let mut max_diff = 0i32;
        for (rc, bc) in rgba_buf.chunks_exact(4).zip(bgra_buf.chunks_exact(4)) {
            max_diff = max_diff.max((rc[0] as i32 - bc[2] as i32).abs());
            max_diff = max_diff.max((rc[1] as i32 - bc[1] as i32).abs());
            max_diff = max_diff.max((rc[2] as i32 - bc[0] as i32).abs());
            max_diff = max_diff.max((rc[3] as i32 - bc[3] as i32).abs());
        }
        eprintln!("draw_masks_mem BGRA vs RGBA max channel diff: {max_diff}");
        assert!(
            max_diff <= 1,
            "draw_masks_mem BGRA/RGBA channel mismatch > 1: max_diff={max_diff}"
        );
    }
}
