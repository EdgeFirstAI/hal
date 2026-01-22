// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "linux")]
#![cfg(feature = "opengl")]

use edgefirst_decoder::DetectBox;
#[cfg(feature = "decoder")]
use edgefirst_decoder::Segmentation;
use edgefirst_tensor::{TensorMemory, TensorTrait};
use four_char_code::FourCharCode;
use gbm::{
    AsRaw, Device,
    drm::{Device as DrmDevice, buffer::DrmFourcc, control::Device as DrmControlDevice},
};
use khronos_egl::{self as egl, Attrib, Display, Dynamic, EGL1_4, Instance};
use log::{debug, error};
use std::{
    collections::BTreeSet,
    ffi::{CStr, CString, c_char, c_void},
    os::fd::AsRawFd,
    ptr::{NonNull, null, null_mut},
    rc::Rc,
    str::FromStr,
    sync::OnceLock,
    thread::JoinHandle,
    time::Instant,
};
use tokio::sync::mpsc::Sender;

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

#[cfg(feature = "decoder")]
use crate::DEFAULT_COLORS;
use crate::{
    Crop, Error, Flip, GREY, ImageProcessorTrait, NV12, PLANAR_RGB, PLANAR_RGBA, RGB, RGBA, Rect,
    Rotation, TensorImage, YUYV,
};

static EGL_LIB: OnceLock<libloading::Library> = OnceLock::new();

fn get_egl_lib() -> Result<&'static libloading::Library, crate::Error> {
    if let Some(egl) = EGL_LIB.get() {
        Ok(egl)
    } else {
        let egl = unsafe { libloading::Library::new("libEGL.so.1")? };
        Ok(EGL_LIB.get_or_init(|| egl))
    }
}

type Egl = Instance<Dynamic<&'static libloading::Library, EGL1_4>>;
pub(crate) struct GlContext {
    pub(crate) support_dma: bool,
    pub(crate) surface: Option<egl::Surface>,
    pub(crate) display: EglDisplayType,
    pub(crate) ctx: egl::Context,
    pub(crate) egl: Rc<Egl>,
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
    pub(crate) fn new() -> Result<GlContext, crate::Error> {
        // Create an EGL API instance.
        let egl: Rc<Egl> =
            Rc::new(unsafe { Instance::<Dynamic<_, EGL1_4>>::load_required_from(get_egl_lib()?)? });

        if let Ok(headless) = Self::try_initialize_egl(egl.clone(), Self::egl_get_default_display) {
            return Ok(headless);
        } else {
            log::debug!("Didn't initialize EGL with Default Display");
        }

        if let Ok(headless) = Self::try_initialize_egl(egl.clone(), Self::egl_get_gbm_display) {
            return Ok(headless);
        } else {
            log::debug!("Didn't initialize EGL with GBM Display");
        }

        if let Ok(headless) =
            Self::try_initialize_egl(egl.clone(), Self::egl_get_platform_display_from_device)
        {
            return Ok(headless);
        } else {
            log::debug!("Didn't initialize EGL with platform display from device enumeration");
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

        let config =
            if let Some(config) = egl.choose_first_config(display.as_display(), &attributes)? {
                config
            } else {
                return Err(crate::Error::NotImplemented(
                    "Did not find valid OpenGL ES config".to_string(),
                ));
            };

        debug!("config: {config:?}");

        let surface = Some(egl.create_pbuffer_surface(
            display.as_display(),
            config,
            &[egl::WIDTH, 64, egl::HEIGHT, 64, egl::NONE],
        )?);

        egl.bind_api(egl::OPENGL_ES_API)?;
        let context_attributes = [egl::CONTEXT_MAJOR_VERSION, 3, egl::NONE, egl::NONE];

        let ctx = egl.create_context(display.as_display(), config, None, &context_attributes)?;
        debug!("ctx: {ctx:?}");

        egl.make_current(display.as_display(), surface, surface, Some(ctx))?;

        let support_dma = Self::egl_check_support_dma(&egl).is_ok();
        let headless = GlContext {
            display,
            ctx,
            egl,
            surface,
            support_dma,
        };
        Ok(headless)
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

        // just use the first device?
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
        // Err(crate::Error::GLVersion("EGL Version too low".to_string()))
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

    fn egl_destory_image_with_fallback(
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
        let _ = self
            .egl
            .make_current(self.display.as_display(), None, None, None);

        let _ = self
            .egl
            .destroy_context(self.display.as_display(), self.ctx);

        if let Some(surface) = self.surface.take() {
            let _ = self.egl.destroy_surface(self.display.as_display(), surface);
        }

        let _ = self.egl.terminate(self.display.as_display());
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
        let targets = ["/dev/dri/render128", "/dev/dri/card0", "/dev/dri/card1"];
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
    ImageRender(
        SendablePtr<TensorImage>,
        SendablePtr<DetectBox>,
        SendablePtr<Segmentation>,
        tokio::sync::oneshot::Sender<Result<(), Error>>,
    ),
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
    support_dma: bool,
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
    pub fn new() -> Result<Self, Error> {
        let (send, mut recv) = tokio::sync::mpsc::channel::<GLProcessorMessage>(1);

        let (create_ctx_send, create_ctx_recv) = tokio::sync::oneshot::channel();

        let func = move || {
            let mut gl_converter = match GLProcessorST::new() {
                Ok(gl) => gl,
                Err(e) => {
                    let _ = create_ctx_send.send(Err(e));
                    return;
                }
            };
            let _ = create_ctx_send.send(Ok(gl_converter.gl_context.support_dma));
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
                    GLProcessorMessage::ImageRender(mut dst, det, seg, resp) => {
                        // SAFETY: This is safe because the render_to_image() function waits for the
                        // resp to be sent before dropping the borrow for dst, detect, and
                        // segmentation
                        let dst = unsafe { dst.ptr.as_mut() };
                        let det = unsafe { std::slice::from_raw_parts(det.ptr.as_ptr(), det.len) };
                        let seg = unsafe { std::slice::from_raw_parts(seg.ptr.as_ptr(), seg.len) };
                        let res = gl_converter.render_to_image(dst, det, seg);
                        let _ = resp.send(res);
                    }
                    GLProcessorMessage::SetColors(colors, resp) => {
                        let res = gl_converter.set_class_colors(&colors);
                        let _ = resp.send(res);
                    }
                }
            }
        };

        // let handle = tokio::task::spawn(func());
        let handle = std::thread::spawn(func);

        let support_dma = match create_ctx_recv.blocking_recv() {
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(Error::Internal(
                    "GL converter error messaging closed without update".to_string(),
                ));
            }
            Ok(Ok(supports_dma)) => supports_dma,
        };

        Ok(Self {
            handle: Some(handle),
            sender: Some(send),
            support_dma,
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
        if !GLProcessorST::check_src_format_supported(self.support_dma, src) {
            return Err(crate::Error::NotSupported(format!(
                "Opengl doesn't support {} source texture",
                src.fourcc().display()
            )));
        }

        if !GLProcessorST::check_dst_format_supported(self.support_dma, dst) {
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

    #[cfg(feature = "decoder")]
    fn render_to_image(
        &mut self,
        dst: &mut TensorImage,
        detect: &[crate::DetectBox],
        segmentation: &[crate::Segmentation],
    ) -> crate::Result<()> {
        let (err_send, err_recv) = tokio::sync::oneshot::channel();
        self.sender
            .as_ref()
            .unwrap()
            .blocking_send(GLProcessorMessage::ImageRender(
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

    #[cfg(feature = "decoder")]
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

impl Drop for GLProcessorThreaded {
    fn drop(&mut self) {
        drop(self.sender.take());
        let _ = self.handle.take().and_then(|h| h.join().ok());
    }
}

/// OpenGL single-threaded image converter.
pub struct GLProcessorST {
    camera_eglimage_texture: Texture,
    camera_normal_texture: Texture,
    render_texture: Texture,
    #[cfg(feature = "decoder")]
    segmentation_texture: Texture,
    #[cfg(feature = "decoder")]
    segmentation_program: GlProgram,
    #[cfg(feature = "decoder")]
    instanced_segmentation_program: GlProgram,
    #[cfg(feature = "decoder")]
    color_program: GlProgram,
    vertex_buffer: Buffer,
    texture_buffer: Buffer,
    texture_program: GlProgram,
    texture_program_yuv: GlProgram,
    texture_program_planar: GlProgram,
    gl_context: GlContext,
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
        if !Self::check_src_format_supported(self.gl_context.support_dma, src) {
            return Err(crate::Error::NotSupported(format!(
                "Opengl doesn't support {} source texture",
                src.fourcc().display()
            )));
        }

        if !Self::check_dst_format_supported(self.gl_context.support_dma, dst) {
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
        if self.gl_context.support_dma
            && dst.tensor().memory() == TensorMemory::Dma
            && dst.fourcc() != RGB
        // DMA generally doesn't support RGB
        {
            let res = self.convert_dest_dma(dst, src, rotation, flip, crop);
            return res;
        }
        let start = Instant::now();
        let res = self.convert_dest_non_dma(dst, src, rotation, flip, crop);
        log::debug!("convert_dest_non_dma takes {:?}", start.elapsed());
        res
    }

    #[cfg(feature = "decoder")]
    fn render_to_image(
        &mut self,
        dst: &mut TensorImage,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
    ) -> Result<(), crate::Error> {
        use crate::FunctionTimer;

        let _timer = FunctionTimer::new("GLProcessorST::render_to_image");
        if !matches!(dst.fourcc(), RGBA | RGB) {
            return Err(crate::Error::NotSupported(
                "Opengl image rendering only supports RGBA or RGB images".to_string(),
            ));
        }

        let (_render_buffer, is_dma) = match dst.tensor.memory() {
            edgefirst_tensor::TensorMemory::Dma => {
                if let Ok(render_buffer) = self.setup_renderbuffer_dma(dst) {
                    (render_buffer, true)
                } else {
                    (
                        self.setup_renderbuffer_non_dma(
                            dst,
                            Crop::new().with_dst_rect(Some(Rect::new(0, 0, 0, 0))),
                        )?,
                        false,
                    )
                }
            }
            _ => (
                self.setup_renderbuffer_non_dma(
                    dst,
                    Crop::new().with_dst_rect(Some(Rect::new(0, 0, 0, 0))),
                )?,
                false,
            ), // Add dest rect to make sure dst is rendered fully
        };

        self.render_box(dst, detect)?;
        self.render_segmentation(detect, segmentation)?;

        gls::finish();
        if !is_dma {
            let mut dst_map = dst.tensor().map()?;
            let format = match dst.fourcc() {
                RGB => gls::gl::RGB,
                RGBA => gls::gl::RGBA,
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

    #[cfg(feature = "decoder")]
    fn set_class_colors(&mut self, colors: &[[u8; 4]]) -> crate::Result<()> {
        if colors.is_empty() {
            return Ok(());
        }
        let colors_f32 = colors
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
        self.color_program
            .load_uniform_4fv(c"colors", &colors_f32)?;

        Ok(())
    }
}

impl GLProcessorST {
    pub fn new() -> Result<GLProcessorST, crate::Error> {
        let gl_context = GlContext::new()?;
        gls::load_with(|s| {
            gl_context
                .egl
                .get_proc_address(s)
                .map_or(std::ptr::null(), |p| p as *const _)
        });

        Self::gl_check_support()?;

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

        #[cfg(feature = "decoder")]
        let segmentation_program =
            GlProgram::new(generate_vertex_shader(), generate_segmentation_shader())?;
        #[cfg(feature = "decoder")]
        segmentation_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;
        #[cfg(feature = "decoder")]
        let instanced_segmentation_program = GlProgram::new(
            generate_vertex_shader(),
            generate_instanced_segmentation_shader(),
        )?;
        #[cfg(feature = "decoder")]
        instanced_segmentation_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        #[cfg(feature = "decoder")]
        let color_program = GlProgram::new(generate_vertex_shader(), generate_color_shader())?;
        #[cfg(feature = "decoder")]
        color_program.load_uniform_4fv(c"colors", &DEFAULT_COLORS)?;

        let camera_eglimage_texture = Texture::new();
        let camera_normal_texture = Texture::new();
        let render_texture = Texture::new();
        let segmentation_texture = Texture::new();
        let vertex_buffer = Buffer::new(0, 3, 100);
        let texture_buffer = Buffer::new(1, 2, 100);

        let converter = GLProcessorST {
            gl_context,
            texture_program,
            texture_program_yuv,
            texture_program_planar,
            camera_eglimage_texture,
            camera_normal_texture,
            #[cfg(feature = "decoder")]
            segmentation_texture,
            vertex_buffer,
            texture_buffer,
            render_texture,
            #[cfg(feature = "decoder")]
            segmentation_program,
            #[cfg(feature = "decoder")]
            instanced_segmentation_program,
            #[cfg(feature = "decoder")]
            color_program,
        };
        check_gl_error(function!(), line!())?;

        log::debug!("GLConverter created");
        Ok(converter)
    }

    fn check_src_format_supported(support_dma: bool, img: &TensorImage) -> bool {
        if support_dma && img.tensor().memory() == TensorMemory::Dma {
            // generally EGLImage doesn't support RGB
            matches!(img.fourcc(), RGBA | GREY | YUYV)
        } else {
            matches!(img.fourcc(), RGB | RGBA | GREY)
        }
    }

    fn check_dst_format_supported(support_dma: bool, img: &TensorImage) -> bool {
        if support_dma && img.tensor().memory() == TensorMemory::Dma {
            // generally EGLImage doesn't support RGB
            matches!(img.fourcc(), RGBA | GREY | PLANAR_RGB)
        } else {
            matches!(img.fourcc(), RGB | RGBA | GREY)
        }
    }

    fn gl_check_support() -> Result<(), crate::Error> {
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
        let required_ext = [
            "GL_OES_EGL_image_external_essl3",
            "GL_OES_surfaceless_context",
        ];
        let extensions = extensions.split_ascii_whitespace().collect::<BTreeSet<_>>();
        for required in required_ext {
            if !extensions.contains(required) {
                return Err(crate::Error::GLVersion(format!(
                    "GL does not support {required} extension",
                )));
            }
        }

        Ok(())
    }

    fn setup_renderbuffer_dma(&mut self, dst: &TensorImage) -> crate::Result<FrameBuffer> {
        let frame_buffer = FrameBuffer::new();
        frame_buffer.bind();

        let (width, height) = if matches!(dst.fourcc(), PLANAR_RGB) {
            let width = dst.width();
            let height = dst.height() * 3;
            (width as i32, height as i32)
        } else {
            (dst.width() as i32, dst.height() as i32)
        };
        let dest_img = self.create_image_from_dma2(dst)?;
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
            gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, dest_img.egl_image.as_ptr());
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
        Ok(frame_buffer)
    }

    fn convert_dest_dma(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        assert!(self.gl_context.support_dma);
        let _framebuffer = self.setup_renderbuffer_dma(dst)?;
        if dst.is_planar() {
            self.convert_to_planar(src, dst, rotation, flip, crop)
        } else {
            self.convert_to(src, dst, rotation, flip, crop)
        }
    }

    fn setup_renderbuffer_non_dma(
        &mut self,
        dst: &TensorImage,
        crop: Crop,
    ) -> crate::Result<FrameBuffer> {
        debug_assert!(matches!(dst.fourcc(), RGB | RGBA | GREY | PLANAR_RGB));
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
                RGB => gls::gl::RGB,
                RGBA => gls::gl::RGBA,
                GREY => gls::gl::RED,
                _ => unreachable!(),
            }
        };

        let start = Instant::now();
        let frame_buffer = FrameBuffer::new();
        frame_buffer.bind();

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
        Ok(frame_buffer)
    }

    fn convert_dest_non_dma(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let _framebuffer = self.setup_renderbuffer_non_dma(dst, crop)?;
        let start = Instant::now();
        if dst.is_planar() {
            self.convert_to_planar(src, dst, rotation, flip, crop)?;
        } else {
            self.convert_to(src, dst, rotation, flip, crop)?;
        }
        log::debug!("Draw to framebuffer takes {:?}", start.elapsed());
        let start = Instant::now();
        let dest_format = match dst.fourcc() {
            RGB => gls::gl::RGB,
            RGBA => gls::gl::RGBA,
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
        }
        log::debug!("Read from framebuffer takes {:?}", start.elapsed());
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
        if has_crop && let Some(dst_color) = crop.dst_color {
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
        if self.gl_context.support_dma
            && let Ok(new_egl_image) = self.create_image_from_dma2(src)
        {
            self.draw_camera_texture_eglimage(
                src,
                &new_egl_image,
                src_roi,
                dst_roi,
                rotation_offset,
                flip,
            )?
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
        &self,
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
            PLANAR_RGB => false,
            PLANAR_RGBA => true,
            _ => {
                return Err(crate::Error::NotSupported(
                    "Destination format must be PLANAR_RGB or PLANAR_RGBA".to_string(),
                ));
            }
        };

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
        if has_crop && let Some(dst_color) = crop.dst_color {
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

        let new_egl_image = self.create_image_from_dma2(src)?;

        self.draw_camera_texture_to_rgb_planar(
            &new_egl_image,
            src_roi,
            dst_roi,
            rotation_offset,
            flip,
            alpha,
        )?;
        unsafe { gls::gl::Finish() };

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
        egl_img: &EglImage,
        src_roi: RegionOfInterest,
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
        alpha: bool,
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
            // self.texture_program.load_uniform_1f(c"width", width as f32);
            gls::gl::UseProgram(self.texture_program_planar.id);
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

            gls::egl_image_target_texture_2d_oes(texture_target, egl_img.egl_image.as_ptr());
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
            _ => unreachable!(),
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
        egl_img: &EglImage,
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

            gls::egl_image_target_texture_2d_oes(texture_target, egl_img.egl_image.as_ptr());
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
        if src.is_planar() {
            if !src.width().is_multiple_of(16) {
                return Err(Error::NotSupported(
                    "OpenGL Planar RGB EGLImage doesn't support image widths which are not multiples of 16"
                        .to_string(),
                ));
            }
            match src.fourcc() {
                PLANAR_RGB => {
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
            format = fourcc_to_drm(src.fourcc());
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
        };

        let mut egl_img_attr = vec![
            egl_ext::LINUX_DRM_FOURCC as Attrib,
            format as Attrib,
            khronos_egl::WIDTH as Attrib,
            width as Attrib,
            khronos_egl::HEIGHT as Attrib,
            height as Attrib,
            egl_ext::DMA_BUF_PLANE0_PITCH as Attrib,
            (width * channels) as Attrib,
            egl_ext::DMA_BUF_PLANE0_OFFSET as Attrib,
            0 as Attrib,
            egl_ext::DMA_BUF_PLANE0_FD as Attrib,
            fd as Attrib,
            egl::IMAGE_PRESERVED as Attrib,
            egl::TRUE as Attrib,
        ];
        if matches!(src.fourcc(), YUYV | NV12) {
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
            egl: self.gl_context.egl.clone(),
        })
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

    #[cfg(feature = "decoder")]
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

    #[cfg(feature = "decoder")]
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

    fn render_segmentation(
        &mut self,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
    ) -> crate::Result<()> {
        if segmentation.is_empty() {
            return Ok(());
        }
        gls::enable(gls::gl::BLEND);
        gls::blend_func(gls::gl::SRC_ALPHA, gls::gl::ONE_MINUS_SRC_ALPHA);

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

        let e = GlContext::egl_destory_image_with_fallback(&self.egl, self.display, self.egl_image);
        if let Err(e) = e {
            error!("Could not destroy EGL image: {e:?}");
        }
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
        unsafe { gls::gl::DeleteTextures(1, &raw mut self.id) };
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
        unsafe { gls::gl::DeleteBuffers(1, &raw mut self.id) };
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
        self.unbind();
        unsafe {
            gls::gl::DeleteFramebuffers(1, &raw mut self.id);
        }
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
        unsafe {
            gls::gl::DeleteProgram(self.id);
            gls::gl::DeleteShader(self.fragment_id);
            gls::gl::DeleteShader(self.vertex_id);
        }
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

fn fourcc_to_drm(fourcc: FourCharCode) -> DrmFourcc {
    match fourcc {
        RGBA => DrmFourcc::Abgr8888,
        YUYV => DrmFourcc::Yuyv,
        RGB => DrmFourcc::Bgr888,
        GREY => DrmFourcc::R8,
        _ => todo!(),
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

#[cfg(test)]
#[cfg(feature = "opengl")]
mod gl_tests {
    use super::*;
    use crate::{RGBA, TensorImage};
    use ndarray::Array3;

    #[test]
    #[cfg(feature = "decoder")]
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

        let mut renderer = GLProcessorThreaded::new().unwrap();
        renderer.render_to_image(&mut image, &[], &[seg]).unwrap();

        image.save_jpeg("test_segmentation.jpg", 80).unwrap();
    }

    #[test]
    #[cfg(feature = "decoder")]
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

        let mut renderer = GLProcessorThreaded::new().unwrap();
        renderer.render_to_image(&mut image, &[], &[seg]).unwrap();

        image.save_jpeg("test_segmentation_mem.jpg", 80).unwrap();
    }

    #[test]
    #[cfg(feature = "decoder")]
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
            label: 0,
        };

        let seg = Segmentation {
            segmentation,
            xmin: 0.59375,
            ymin: 0.25,
            xmax: 0.9375,
            ymax: 0.725,
        };

        let mut renderer = GLProcessorThreaded::new().unwrap();
        renderer
            .set_class_colors(&[[255, 255, 0, 233], [128, 128, 0, 20]])
            .unwrap();
        renderer
            .render_to_image(&mut image, &[detect], &[seg])
            .unwrap();

        image.save_jpeg("test_segmentation_yolo.jpg", 80).unwrap();
    }

    #[test]
    #[cfg(feature = "decoder")]
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
        let mut renderer = GLProcessorThreaded::new().unwrap();
        renderer
            .set_class_colors(&[[255, 255, 0, 233], [128, 128, 0, 20]])
            .unwrap();
        renderer
            .render_to_image(&mut image, &[detect], &[])
            .unwrap();

        image.save_jpeg("test_boxes.jpg", 80).unwrap();
    }

    static GL_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // Helper function to check if OpenGL is available
    fn is_opengl_available() -> bool {
        #[cfg(all(target_os = "linux", feature = "opengl"))]
        {
            *GL_AVAILABLE.get_or_init(|| GLProcessorThreaded::new().is_ok())
        }

        #[cfg(not(all(target_os = "linux", feature = "opengl")))]
        {
            false
        }
    }
}
