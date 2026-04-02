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

/// Shared EGL display — created once, reused by all `GlContext` instances.
///
/// On Vivante galcore (i.MX8M Plus), opening multiple DRM fds causes
/// kernel-level deadlocks between the galcore and DRM ioctl paths. Sharing
/// one display (and its backing GBM device / DRM fd) avoids the conflict.
///
/// The display is never terminated (intentional leak). Same pattern as
/// `EGL_LIB` above and `SHARED_DRM_FD` in `crates/tensor/src/dmabuf.rs`.
struct SharedEglDisplay {
    display: egl::Display,
    kind: EglDisplayKind,
    transfer_backend: TransferBackend,
    /// Kept alive — dropping the GBM device closes the DRM fd and
    /// invalidates the EGL display.
    _resources: SharedDisplayResources,
}

/// Backing resources that must outlive the EGL display.
enum SharedDisplayResources {
    Gbm(#[allow(dead_code)] Device<Card>),
    PlatformDevice,
    Default,
}

// SAFETY: SharedEglDisplay is only accessed under GL_MUTEX or OnceLock
// initialization (which is itself synchronized). The egl::Display handle
// is a raw pointer, but all EGL calls using it are serialized by GL_MUTEX.
unsafe impl Send for SharedEglDisplay {}
unsafe impl Sync for SharedEglDisplay {}

static SHARED_DISPLAY: OnceLock<Result<SharedEglDisplay, String>> = OnceLock::new();

/// Initialize the shared EGL display. Called once per process via `OnceLock`.
///
/// Probes display types in priority order (PlatformDevice → GBM → Default),
/// initializes the first that works, and detects DMA-BUF support via
/// display extension queries. No temporary GL context is created —
/// Vivante galcore corrupts state on context create/destroy cycles.
type DisplayFn = fn(&Egl) -> Result<EglDisplayType, crate::Error>;

fn init_shared_display(
    egl: &Rc<Egl>,
    kind: Option<EglDisplayKind>,
) -> Result<SharedEglDisplay, String> {
    let display_fns: Vec<(EglDisplayKind, DisplayFn)> = if let Some(kind) = kind {
        let f = match kind {
            EglDisplayKind::Gbm => GlContext::egl_get_gbm_display as DisplayFn,
            EglDisplayKind::PlatformDevice => GlContext::egl_get_platform_display_from_device,
            EglDisplayKind::Default => GlContext::egl_get_default_display,
        };
        vec![(kind, f)]
    } else {
        vec![
            (
                EglDisplayKind::PlatformDevice,
                GlContext::egl_get_platform_display_from_device as DisplayFn,
            ),
            (EglDisplayKind::Gbm, GlContext::egl_get_gbm_display),
            (EglDisplayKind::Default, GlContext::egl_get_default_display),
        ]
    };

    for (kind, display_fn) in display_fns {
        let display_type = match display_fn(egl) {
            Ok(d) => d,
            Err(e) => {
                log::debug!("Shared display: {kind} display creation failed: {e:?}");
                continue;
            }
        };
        let display = display_type.as_display();

        if let Err(e) = egl.initialize(display) {
            log::debug!("Shared display: eglInitialize failed for {kind}: {e:?}");
            continue;
        }

        // Verify required extensions
        let ext_str = match egl.query_string(Some(display), egl::EXTENSIONS) {
            Ok(s) => s,
            Err(e) => {
                log::debug!("Shared display: eglQueryString failed for {kind}: {e:?}");
                let _ = egl.terminate(display);
                continue;
            }
        };
        let exts = ext_str.to_string_lossy();
        if !exts.contains("EGL_KHR_surfaceless_context")
            || !exts.contains("EGL_KHR_no_config_context")
        {
            log::debug!("Shared display: {kind} missing required extensions");
            let _ = egl.terminate(display);
            continue;
        }

        if egl.bind_api(egl::OPENGL_ES_API).is_err() {
            log::debug!("Shared display: eglBindApi failed for {kind}");
            let _ = egl.terminate(display);
            continue;
        }

        // Check DMA-BUF support (display extension query — no context needed)
        let transfer_backend = if GlContext::egl_check_support_dma(egl, display).is_ok() {
            TransferBackend::DmaBuf
        } else {
            TransferBackend::Sync
        };

        // Do NOT create a temporary context here to probe has_compute.
        // On Vivante galcore, creating and destroying a context before
        // the real one corrupts driver state and causes deadlocks.
        // Each GlContext::new() will try GLES 3.1 → 3.0 itself.

        let resources = match display_type {
            EglDisplayType::Gbm(_, gbm) => SharedDisplayResources::Gbm(gbm),
            EglDisplayType::PlatformDisplay(_) => SharedDisplayResources::PlatformDevice,
            EglDisplayType::Default(_) => SharedDisplayResources::Default,
        };

        log::info!("Shared EGL display initialized: kind={kind}, transfer={transfer_backend:?}");

        return Ok(SharedEglDisplay {
            display,
            kind,
            transfer_backend,
            _resources: resources,
        });
    }

    Err("Could not initialize EGL with any known method".to_string())
}

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

/// Probe for available EGL displays supporting headless OpenGL ES 3.0.
///
/// When a shared display already exists (from a prior `GlContext::new()`
/// call or earlier probe), returns only that display's kind. Opening
/// additional DRM fds would risk galcore deadlocks on Vivante.
///
/// When no shared display exists, probes in priority order
/// (PlatformDevice, GBM, Default) and caches the first successful
/// display in `SHARED_DISPLAY` for future use. The display is
/// intentionally never terminated (same leak pattern as `EGL_LIB`).
///
/// An empty list means OpenGL is not available on this system.
///
/// # Errors
///
/// Returns an error only if `libEGL.so.1` cannot be loaded. Individual
/// display probe failures are silently skipped.
pub fn probe_egl_displays() -> Result<Vec<EglDisplayInfo>, Error> {
    // Serialize with all other EGL/GL operations — see `GL_MUTEX` doc comment.
    let _guard = GL_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

    let egl: Egl = unsafe { Instance::<Dynamic<_, EGL1_4>>::load_required_from(get_egl_lib()?)? };

    // If a shared display already exists, report only its kind.
    // Opening additional DRM fds to probe other display types would
    // deadlock galcore on Vivante when the shared display is alive.
    if let Some(Ok(shared)) = SHARED_DISPLAY.get() {
        let description = match shared.kind {
            EglDisplayKind::PlatformDevice => {
                "EGL platform device via EGL_EXT_device_enumeration".to_string()
            }
            EglDisplayKind::Gbm => "GBM via /dev/dri/renderD128".to_string(),
            EglDisplayKind::Default => "EGL default display".to_string(),
        };
        return Ok(vec![EglDisplayInfo {
            kind: shared.kind,
            description,
        }]);
    }

    // No shared display yet — safe to probe all types. The first
    // successful probe populates SHARED_DISPLAY for future use.
    let egl_rc: Rc<Egl> = Rc::new(egl);

    // Try to initialize the shared display (probes in priority order).
    // This populates SHARED_DISPLAY as a side effect.
    let shared_result = SHARED_DISPLAY.get_or_init(|| init_shared_display(&egl_rc, None));

    let mut results = Vec::new();

    if let Ok(shared) = shared_result {
        let description = match shared.kind {
            EglDisplayKind::PlatformDevice => {
                "EGL platform device via EGL_EXT_device_enumeration".to_string()
            }
            EglDisplayKind::Gbm => "GBM via /dev/dri/renderD128".to_string(),
            EglDisplayKind::Default => "EGL default display".to_string(),
        };
        results.push(EglDisplayInfo {
            kind: shared.kind,
            description,
        });
    }

    // Note: we intentionally do NOT probe additional display types here.
    // The shared display is the only one that will be used, and probing
    // others would open DRM fds that risk galcore deadlocks if a GL
    // context is later created on the shared display.

    Ok(results)
}

pub(super) struct GlContext {
    pub(super) transfer_backend: TransferBackend,
    pub(super) display: egl::Display,
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
        let egl: Rc<Egl> =
            Rc::new(unsafe { Instance::<Dynamic<_, EGL1_4>>::load_required_from(get_egl_lib()?)? });

        // Get or create the shared display
        let shared = SHARED_DISPLAY
            .get_or_init(|| init_shared_display(&egl, kind))
            .as_ref()
            .map_err(|e| Error::OpenGl(e.clone()))?;

        // If caller requested a specific kind, verify it matches
        if let Some(requested) = kind {
            if requested != shared.kind {
                return Err(Error::OpenGl(format!(
                    "Requested EGL display kind {requested} but shared display is {} \
                     (only one display type can be active per process to avoid \
                     Vivante galcore deadlocks)",
                    shared.kind,
                )));
            }
        }

        let display = shared.display;
        let transfer_backend = shared.transfer_backend;

        // Bind API for this thread (required per EGL spec — thread-local state)
        egl.bind_api(egl::OPENGL_ES_API)?;

        // Create a new GL context on the shared display.
        // Try GLES 3.1 first (compute shaders), fall back to 3.0.
        let ctx_31 = [
            egl::CONTEXT_MAJOR_VERSION,
            3,
            egl::CONTEXT_MINOR_VERSION,
            1,
            egl::NONE,
        ];
        let (ctx, has_compute) =
            match egl.create_context(display, egl_ext::NO_CONFIG_KHR, None, &ctx_31) {
                Ok(ctx) => {
                    debug!("Created GLES 3.1 context (compute shaders available)");
                    (ctx, true)
                }
                Err(_) => {
                    let ctx_30 = [egl::CONTEXT_MAJOR_VERSION, 3, egl::NONE];
                    let ctx = egl.create_context(display, egl_ext::NO_CONFIG_KHR, None, &ctx_30)?;
                    debug!("Created GLES 3.0 context (no compute shaders)");
                    (ctx, false)
                }
            };
        debug!("ctx: {ctx:?}");

        egl.make_current(display, None, None, Some(ctx))?;

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
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = self.egl.make_current(self.display, None, None, None);

            let _ = self.egl.destroy_context(self.display, self.ctx);

            // The EGL display is process-global (SharedEglDisplay in
            // SHARED_DISPLAY) and intentionally never terminated. Calling
            // eglTerminate here would tear down the display for all active
            // GlContext instances. The display is leaked on process exit,
            // same as EGL_LIB.
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
