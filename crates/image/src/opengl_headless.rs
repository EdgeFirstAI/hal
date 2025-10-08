#![cfg(target_os = "linux")]
#![cfg(feature = "opengl")]
use edgefirst_tensor::TensorTrait;
use four_char_code::FourCharCode;
use gbm::{
    AsRaw, Device,
    drm::{Device as DrmDevice, buffer::DrmFourcc, control::Device as DrmControlDevice},
};
use khronos_egl::{self as egl, Attrib, Display, EGL1_4};
use log::{debug, error};
use std::{
    collections::BTreeSet,
    ffi::{CStr, CString, c_char, c_void},
    os::fd::AsRawFd,
    ptr::{null, null_mut},
    rc::Rc,
    str::FromStr,
};

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
    Crop, Error, Flip, GREY, ImageConverterTrait, NV12, RGB, RGBA, Rotation, TensorImage, YUYV,
};

pub(crate) struct GlContext {
    pub(crate) support_dma: bool,
    pub(crate) surface: egl::Surface,
    pub(crate) display: egl::Display,
    pub(crate) ctx: egl::Context,
    pub(crate) egl: Rc<egl::Instance<egl::Dynamic<libloading::Library, egl::EGL1_4>>>,
}

impl GlContext {
    pub(crate) fn new() -> Result<GlContext, crate::Error> {
        // Create an EGL API instance.
        let lib = unsafe { libloading::Library::new("libEGL.so.1") }?;
        let egl = unsafe { egl::DynamicInstance::<egl::EGL1_4>::load_required_from(lib)? };
        let support_dma = Self::egl_check_support_dma(&egl).is_ok();
        let display = Self::egl_get_display(&egl)?;

        egl.initialize(display)?;
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
            egl::NONE,
        ];

        let config = if let Some(config) = egl.choose_first_config(display, &attributes)? {
            config
        } else {
            return Err(crate::Error::NotImplemented(
                "Did not find valid OpenGL ES config".to_string(),
            ));
        };

        debug!("config: {config:?}");

        let context_attributes = [egl::CONTEXT_CLIENT_VERSION, 2, egl::NONE, egl::NONE];

        let ctx = egl
            .create_context(display, config, None, &context_attributes)
            .unwrap();
        debug!("ctx: {ctx:?}");

        let surface = egl.create_pbuffer_surface(
            display,
            config,
            &[egl::WIDTH, 1, egl::HEIGHT, 1, egl::NONE],
        )?;
        egl.make_current(display, Some(surface), Some(surface), Some(ctx))?;
        let _ = egl.swap_interval(display, 0);

        let headless = GlContext {
            display,
            ctx,
            egl: Rc::new(egl),
            surface,
            support_dma,
        };
        Ok(headless)
    }

    fn make_current(&self) -> Result<(), crate::Error> {
        self.egl.make_current(
            self.display,
            Some(self.surface),
            Some(self.surface),
            Some(self.ctx),
        )?;
        Ok(())
    }

    fn egl_get_display(
        egl: &egl::Instance<egl::Dynamic<libloading::Library, EGL1_4>>,
    ) -> Result<Display, crate::Error> {
        if let Ok(display) = Self::egl_get_gbm_display(egl) {
            debug!("gbm display: {display:?}");
            return Ok(display);
        }

        // get the default display
        if let Some(display) = unsafe { egl.get_display(egl::DEFAULT_DISPLAY) } {
            debug!("default display: {display:?}");
            return Ok(display);
        }

        if let Ok(display) = Self::egl_get_platform_display_from_device(egl) {
            debug!("platform display from device: {display:?}");
            return Ok(display);
        }

        Err(Error::OpenGl("Could not obtain EGL Display".to_string()))
    }

    fn egl_get_gbm_display(
        egl: &egl::Instance<egl::Dynamic<libloading::Library, EGL1_4>>,
    ) -> Result<Display, crate::Error> {
        // init a GBM device
        let gbm = Device::new(Card::open_global()?)?;

        debug!("gbm: {gbm:?}");
        let display = Self::egl_get_platform_display_with_fallback(
            egl,
            egl_ext::PLATFORM_GBM_KHR,
            gbm.as_raw() as *mut c_void,
            &[egl::ATTRIB_NONE],
        )?;

        Ok(display)
    }

    fn egl_check_support_dma(
        egl: &egl::Instance<egl::Dynamic<libloading::Library, EGL1_4>>,
    ) -> Result<(), crate::Error> {
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
        egl: &egl::Instance<egl::Dynamic<libloading::Library, EGL1_4>>,
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
                Err(egl.get_error().unwrap().into())
            }
        } else {
            Err(Error::EGLLoad(egl::LoadError::InvalidVersion {
                provided: egl.version(),
                required: khronos_egl::Version::EGL1_5,
            }))
        }
    }

    fn egl_create_image_with_fallback(
        egl: &egl::Instance<egl::Dynamic<libloading::Library, EGL1_4>>,
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
                Err(egl.get_error().unwrap().into())
            }
        } else {
            Err(Error::EGLLoad(egl::LoadError::InvalidVersion {
                provided: egl.version(),
                required: khronos_egl::Version::EGL1_5,
            }))
        }
    }

    fn egl_destory_image_with_fallback(
        egl: &egl::Instance<egl::Dynamic<libloading::Library, EGL1_4>>,
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
                Err(egl.get_error().unwrap().into())
            }
        } else {
            Err(Error::EGLLoad(egl::LoadError::InvalidVersion {
                provided: egl.version(),
                required: khronos_egl::Version::EGL1_5,
            }))
        }
    }

    fn egl_get_platform_display_from_device(
        egl: &egl::Instance<egl::Dynamic<libloading::Library, EGL1_4>>,
    ) -> Result<Display, Error> {
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
        Self::egl_get_platform_display_with_fallback(
            egl,
            egl_ext::PLATFORM_DEVICE_EXT,
            devices[0],
            &[egl::ATTRIB_NONE],
        )
    }
}

impl Drop for GlContext {
    fn drop(&mut self) {
        let _ = self.egl.destroy_surface(self.display, self.surface);
        let _ = self.egl.destroy_context(self.display, self.ctx);
        let _ = self.egl.terminate(self.display);
    }
}

#[derive(Debug)]
/// A simple wrapper for a device node.
pub struct Card(std::fs::File);

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
        let targets = ["/dev/dri/card0", "/dev/dri/card1", "/dev/dri/render128"];
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

#[derive(Debug)]
struct RegionOfInterest {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

pub struct GLConverter {
    camera_texture: Texture,
    render_texture: Texture,
    vertex_buffer: Buffer,
    texture_buffer: Buffer,
    texture_program: GlProgram,
    texture_program_planar: GlProgram,
    gl_context: GlContext,
}

impl ImageConverterTrait for GLConverter {
    fn convert(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        crop.check_crop(src, dst)?;
        check_gl_error(function!(), line!())?;
        self.gl_context.make_current()?;
        if self.gl_context.support_dma
            && let edgefirst_tensor::Tensor::Dma(_) = dst.tensor()
        {
            return self.convert_dest_dma(dst, src, rotation, flip, crop);
        }
        if dst.is_planar() && matches!(dst.fourcc(), RGB | RGBA) {
            if rotation != Rotation::None || flip != Flip::None {
                return Err(Error::NotSupported(
                    "Rotation or Flip not supported for planar RGB".to_string(),
                ));
            }
            self.convert_dest_non_dma(dst, src, rotation, flip, crop)?;
        } else {
            self.convert_dest_non_dma(dst, src, rotation, flip, crop)?;
        }
        Ok(())
    }
}

impl GLConverter {
    pub fn new() -> Result<GLConverter, crate::Error> {
        let gl_context = GlContext::new()?;

        gls::load_with(|s| {
            gl_context
                .egl
                .get_proc_address(s)
                .map_or(std::ptr::null(), |p| p as *const _)
        });
        Self::gl_check_support()?;

        let texture_program_planar =
            GlProgram::new(generate_vertex_shader(), generate_planar_rgb_shader())?;

        let texture_program =
            GlProgram::new(generate_vertex_shader(), generate_texture_fragment_shader())?;
        let camera_texture = Texture::new();
        let render_texture = Texture::new();
        let vertex_buffer = Buffer::new(0, 3, 100);
        let texture_buffer = Buffer::new(1, 2, 100);
        let converter = GLConverter {
            gl_context,
            texture_program,
            texture_program_planar,
            camera_texture,
            vertex_buffer,
            texture_buffer,
            render_texture,
        };
        check_gl_error(function!(), line!())?;
        log::debug!("GLConverter created");
        Ok(converter)
    }

    fn gl_check_support() -> Result<(), crate::Error> {
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
        let required_ext = ["GL_OES_EGL_image_external", "GL_OES_surfaceless_context"];
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

    fn convert_dest_dma(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        assert!(self.gl_context.support_dma);
        let frame_buffer = FrameBuffer::new();
        frame_buffer.bind();

        let (width, height) = if dst.is_planar() && matches!(dst.fourcc(), RGB | RGBA) {
            let width = src.width() / 4;
            let height = match src.fourcc() {
                RGBA => src.height() * 4,
                RGB => src.height() * 3,
                fourcc => {
                    return Err(crate::Error::NotSupported(format!(
                        "Unsupported Planar FourCC {fourcc:?}"
                    )));
                }
            };
            (width as i32, height as i32)
        } else {
            (dst.width() as i32, dst.height() as i32)
        };
        let dest_img = self.create_image_from_dma2(dst)?;
        unsafe {
            gls::gl::UseProgram(self.texture_program.id);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.render_texture.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::NEAREST as i32,
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

        if dst.is_planar() {
            self.convert_to_planar(src, width as usize, crop)
        } else {
            self.convert_to(src, dst, rotation, flip, crop)
        }
    }

    fn convert_dest_non_dma(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: crate::Rotation,
        flip: Flip,
        crop: Crop,
    ) -> crate::Result<()> {
        let frame_buffer = FrameBuffer::new();
        frame_buffer.bind();

        let (width, height) = if dst.is_planar() && matches!(dst.fourcc(), RGB | RGBA) {
            let width = src.width() / 4;
            let height = match src.fourcc() {
                RGBA => src.height() * 4,
                RGB => src.height() * 3,
                fourcc => {
                    return Err(crate::Error::NotSupported(format!(
                        "Unsupported Planar FourCC {}",
                        fourcc.display()
                    )));
                }
            };
            (width as i32, height as i32)
        } else {
            (dst.width() as i32, dst.height() as i32)
        };

        let format = if dst.is_planar() {
            gls::gl::RGBA
        } else {
            match dst.fourcc() {
                RGB => gls::gl::RGB,
                RGBA => gls::gl::RGBA,
                // GREY => gls::gl::RGB,
                GREY => gls::gl::R8,
                f => {
                    return Err(crate::Error::NotSupported(format!(
                        "Opengl doesn't support {} destination texture",
                        f.display()
                    )));
                }
            }
        };

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
                gls::gl::NEAREST as i32,
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

        if dst.is_planar() {
            self.convert_to_planar(src, width as usize, crop)?;
        } else {
            self.convert_to(src, dst, rotation, flip, crop)?;
        }

        let dest_format = match dst.fourcc() {
            RGB => gls::gl::RGB,
            RGBA => gls::gl::RGBA,
            GREY => gls::gl::RED,
            f => {
                return Err(Error::NotSupported(format!(
                    "{} textures aren't supported by OpenGL",
                    f.display()
                )));
            }
        };
        unsafe {
            gls::gl::PixelStorei(gls::gl::PACK_ALIGNMENT, 1);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.render_texture.id);

            // TODO: the gls library doesn't have bindings for glGetTexImage?
            let func = self
                .gl_context
                .egl
                .get_proc_address("glGetTexImage")
                .ok_or_else(|| Error::GLVersion("Missing glGetTexImage function".to_string()))?;

            let gl_get_tex_image: extern "system" fn(
                target: gls::gl::GLenum,
                level: gls::gl::GLint,
                format: gls::gl::GLenum,
                type_: gls::gl::GLenum,
                pixels: *mut c_void,
            ) = std::mem::transmute(func);

            gl_get_tex_image(
                gls::gl::TEXTURE_2D,
                0,
                dest_format,
                gls::gl::UNSIGNED_BYTE,
                dst.tensor().map()?.as_mut_ptr() as *mut c_void,
            );
        }
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
        // unsafe {
        //     gls::gl::ClearColor(1.0, 1.0, 1.0, 1.0);
        //     gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
        // };

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

        let result = if let Ok(new_egl_image) = self.create_image_from_dma2(src) {
            self.draw_camera_texture_eglimage(
                src,
                &new_egl_image,
                src_roi,
                dst_roi,
                rotation_offset,
                flip,
            )
        } else {
            self.draw_camera_texture(src, src_roi, dst_roi, rotation_offset, flip)
        };
        unsafe { gls::gl::Finish() };
        check_gl_error(function!(), line!())?;
        result
    }

    fn convert_to_planar(
        &self,
        src: &TensorImage,
        width: usize,
        crop: Crop,
    ) -> Result<(), crate::Error> {
        if let Some(crop) = crop.src_rect
            && (crop.left > 0
                || crop.top > 0
                || crop.height < src.height()
                || crop.width < src.width())
        {
            return Err(crate::Error::NotSupported(
                "Cropping in planar RGB mode is not supported".to_string(),
            ));
        }

        if let Some(crop) = crop.dst_rect
            && (crop.left > 0
                || crop.top > 0
                || crop.height < src.height()
                || crop.width < src.width())
        {
            return Err(crate::Error::NotSupported(
                "Cropping in planar RGB mode is not supported".to_string(),
            ));
        }

        let new_egl_image = self.create_image_from_dma2(src)?;
        unsafe {
            gls::gl::ClearColor(0.0, 0.0, 0.0, 0.0);
            gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
        };
        self.draw_camera_texture_to_rgb_planar(&self.camera_texture, &new_egl_image, width)?;
        unsafe { gls::gl::Finish() };

        Ok(())
    }

    fn draw_camera_texture_to_rgb_planar(
        &self,
        texture: &Texture,
        egl_img: &EglImage,
        width: usize,
    ) -> Result<(), Error> {
        let texture_target = gls::gl::TEXTURE_2D;
        unsafe {
            self.texture_program.load_uniform_1f(c"width", width as f32);
            gls::gl::UseProgram(self.texture_program_planar.id);
            gls::gl::BindTexture(texture_target, texture.id);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::NEAREST as i32,
            );
            gls::gl::TexParameteri(
                texture_target,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::egl_image_target_texture_2d_oes(texture_target, egl_img.egl_image.as_ptr());
            check_gl_error(function!(), line!())?;

            // starts from bottom
            for i in 0..3 {
                self.texture_program.load_uniform_1i(c"color_index", 2 - i);
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
                gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
                let camera_vertices: [f32; 12] = [
                    -1.,
                    -1. + i as f32 * 2. / 3.,
                    0.,
                    1.,
                    -1. + i as f32 * 2. / 3.,
                    0.,
                    1.,
                    -1. / 3. + i as f32 * 2. / 3.,
                    0.,
                    -1.,
                    -1. / 3. + i as f32 * 2. / 3.,
                    0.,
                ];
                gls::gl::BufferData(
                    gls::gl::ARRAY_BUFFER,
                    (size_of::<f32>() * camera_vertices.len()) as isize,
                    camera_vertices.as_ptr() as *const c_void,
                    gls::gl::DYNAMIC_DRAW,
                );

                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texture_buffer.id);
                gls::gl::EnableVertexAttribArray(self.texture_buffer.buffer_index);
                let texture_vertices: [f32; 8] = [0., 1., 1., 1., 1., 0., 0., 0.];
                gls::gl::BufferData(
                    gls::gl::ARRAY_BUFFER,
                    (size_of::<f32>() * texture_vertices.len()) as isize,
                    texture_vertices.as_ptr() as *const c_void,
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
            }
        }
        Ok(())
    }

    fn draw_camera_texture(
        &mut self,
        img: &TensorImage,
        src_roi: RegionOfInterest,
        mut dst_roi: RegionOfInterest,
        rotation_offset: usize,
        flip: Flip,
    ) -> Result<(), Error> {
        let texture_target = gls::gl::TEXTURE_2D;
        let texture_format = match img.fourcc() {
            RGB => gls::gl::RGB,
            RGBA => gls::gl::RGBA,
            GREY => gls::gl::RED,
            f => {
                return Err(Error::NotSupported(format!(
                    "{} textures aren't supported by OpenGL",
                    f.display()
                )));
            }
        };
        unsafe {
            gls::gl::UseProgram(self.texture_program.id);
            gls::gl::BindTexture(texture_target, self.camera_texture.id);
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
            if img.fourcc() == GREY {
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

            self.camera_texture.update_texture(
                texture_target,
                img.width(),
                img.height(),
                texture_format,
                &img.tensor().map()?,
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
        let texture_target = gls::gl::TEXTURE_2D;
        unsafe {
            gls::gl::UseProgram(self.texture_program.id);
            gls::gl::BindTexture(texture_target, self.camera_texture.id);
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
        if src.is_planar {
            if !src.width().is_multiple_of(16) {
                return Err(Error::NotSupported(
                    "OpenGL Planar RGB EGLImage doesn't support image widths which are not multiples of 16"
                        .to_string(),
                ));
            }
            width = src.width() / 4;
            match src.fourcc() {
                RGBA => {
                    format = DrmFourcc::Abgr8888;
                    height = src.height() * 4;
                }
                RGB => {
                    format = DrmFourcc::Abgr8888;
                    height = src.height() * 3;
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
            src.row_stride() as Attrib,
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
            self.gl_context.display,
            unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) },
            target,
            unsafe { egl::ClientBuffer::from_ptr(null_mut()) },
            attrib_list,
        )?;
        Ok(EglImage {
            egl_image: image,
            display: self.gl_context.display,
            egl: self.gl_context.egl.clone(),
        })
    }
}
struct EglImage {
    egl_image: egl::Image,
    egl: Rc<egl::Instance<egl::Dynamic<libloading::Library, egl::EGL1_4>>>,
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

pub struct Texture {
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
    pub fn new() -> Self {
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

    pub fn update_texture(
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

pub struct Buffer {
    id: u32,
    buffer_index: u32,
}

impl Buffer {
    pub fn new(buffer_index: u32, size_per_point: usize, max_points: usize) -> Buffer {
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

pub struct FrameBuffer {
    id: u32,
}

impl FrameBuffer {
    pub fn new() -> FrameBuffer {
        let mut id = 0;
        unsafe {
            gls::gl::GenFramebuffers(1, &raw mut id);
        }

        FrameBuffer { id }
    }

    pub fn bind(&self) {
        unsafe { gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, self.id) };
    }

    pub fn unbind(&self) {
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
}

impl GlProgram {
    pub fn new(vertex_shader: &str, fragment_shader: &str) -> Result<Self, crate::Error> {
        let id = unsafe { gls::gl::CreateProgram() };

        unsafe {
            let vertex_id = gls::gl::CreateShader(gls::gl::VERTEX_SHADER);
            if compile_shader_from_str(vertex_id, vertex_shader, "shader_vert").is_err() {
                return Err(crate::Error::OpenGl(format!(
                    "Shader compile error: {vertex_shader}"
                )));
            }
            gls::gl::AttachShader(id, vertex_id);

            let fragment_id = gls::gl::CreateShader(gls::gl::FRAGMENT_SHADER);
            if compile_shader_from_str(fragment_id, fragment_shader, "shader_frag").is_err() {
                return Err(crate::Error::OpenGl(format!(
                    "Shader compile error: {fragment_shader}"
                )));
            }

            gls::gl::AttachShader(id, fragment_id);
            gls::gl::LinkProgram(id);
            gls::gl::UseProgram(id);
        }

        let pv = [
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ];
        let m = [
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ];

        unsafe {
            let pv_location = gls::gl::GetUniformLocation(id, c"PV".as_ptr());
            gls::gl::UniformMatrix4fv(pv_location, 1, 0, pv.as_ptr());

            let m_location = gls::gl::GetUniformLocation(id, c"M".as_ptr());
            gls::gl::UniformMatrix4fv(m_location, 1, 0, m.as_ptr());
        }
        Ok(Self { id })
    }

    fn load_uniform_1f(&self, name: &CStr, value: f32) {
        unsafe {
            gls::gl::UseProgram(self.id);
            let location = gls::gl::GetUniformLocation(self.id, name.as_ptr());
            gls::gl::Uniform1f(location, value);
        }
    }

    fn load_uniform_1i(&self, name: &CStr, value: i32) {
        unsafe {
            gls::gl::UseProgram(self.id);
            let location = gls::gl::GetUniformLocation(self.id, name.as_ptr());
            gls::gl::Uniform1i(location, value);
        }
    }
}

impl Drop for GlProgram {
    fn drop(&mut self) {
        unsafe {
            gls::gl::DeleteProgram(self.id);
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
            let mut error_log: Vec<u8> = vec![0; max_length as usize + 1];
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
    pub const LINUX_DMA_BUF: u32 = 0x3270;
    pub const LINUX_DRM_FOURCC: u32 = 0x3271;
    pub const DMA_BUF_PLANE0_FD: u32 = 0x3272;
    pub const DMA_BUF_PLANE0_OFFSET: u32 = 0x3273;
    pub const DMA_BUF_PLANE0_PITCH: u32 = 0x3274;
    pub const DMA_BUF_PLANE1_FD: u32 = 0x3275;
    pub const DMA_BUF_PLANE1_OFFSET: u32 = 0x3276;
    pub const DMA_BUF_PLANE1_PITCH: u32 = 0x3277;
    pub const DMA_BUF_PLANE2_FD: u32 = 0x3278;
    pub const DMA_BUF_PLANE2_OFFSET: u32 = 0x3279;
    pub const DMA_BUF_PLANE2_PITCH: u32 = 0x327A;
    pub const YUV_COLOR_SPACE_HINT: u32 = 0x327B;
    pub const SAMPLE_RANGE_HINT: u32 = 0x327C;
    pub const YUV_CHROMA_HORIZONTAL_SITING_HINT: u32 = 0x327D;
    pub const YUV_CHROMA_VERTICAL_SITING_HINT: u32 = 0x327E;

    pub const ITU_REC601: u32 = 0x327F;
    pub const ITU_REC709: u32 = 0x3280;
    pub const ITU_REC2020: u32 = 0x3281;

    pub const YUV_FULL_RANGE: u32 = 0x3282;
    pub const YUV_NARROW_RANGE: u32 = 0x3283;

    pub const YUV_CHROMA_SITING_0: u32 = 0x3284;
    pub const YUV_CHROMA_SITING_0_5: u32 = 0x3285;

    pub const PLATFORM_GBM_KHR: u32 = 0x31D7;

    pub const PLATFORM_DEVICE_EXT: u32 = 0x313F;
}

pub fn generate_vertex_shader() -> &'static str {
    "
#version 300 es
precision mediump float;
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 texCoord;

uniform mat4 M;
uniform mat4 PV;

out vec3 fragPos;
out vec2 tc;

void main() {
    fragPos = pos;
    tc = texCoord;

    gl_Position = vec4(pos, 1.0);
}
"
}

pub fn generate_texture_fragment_shader() -> &'static str {
    "
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

pub fn generate_planar_rgb_shader() -> &'static str {
    "
#version 300 es
// #extension GL_OES_EGL_image_external : require

precision mediump float;
uniform sampler2D tex;
uniform float width;
uniform int color_index;
in vec3 fragPos;
in vec2 tc;

out vec4 color;

void main(){
    float r = texture(tex, vec2(tc[0] - 1.0/width, tc[1]))[color_index];
    float g = texture(tex, vec2(tc[0] + 0.0/width, tc[1]))[color_index];
    float b = texture(tex, vec2(tc[0] + 1.0/width, tc[1]))[color_index];
    float a = texture(tex, vec2(tc[0] + 2.0/width, tc[1]))[color_index];
 
    color = vec4(r, g, b, a);
}
"
}
