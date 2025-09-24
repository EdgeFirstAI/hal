#![cfg(target_os = "linux")]
use drm::{Device as DrmDevice, buffer::DrmFourcc, control::Device as DrmControlDevice};
use edgefirst_tensor::TensorTrait;
use four_char_code::FourCharCode;
use gbm::{AsRaw, Device};
use khronos_egl::{self as egl, Attrib, Config};
use log::{debug, error};
use std::{
    ffi::{CStr, CString, c_char, c_void},
    os::fd::AsRawFd,
    ptr::{null, null_mut},
    rc::Rc,
    str::FromStr,
};

use crate::{Error, GREY, ImageConverterTrait, RGB, RGBA, TensorImage, YUYV};

pub struct Headless {
    pub surface: egl::Surface,
    pub gbm_surface: gbm::Surface<()>,
    pub ctx: egl::Context,
    pub display: egl::Display,
    pub egl: Rc<egl::Instance<egl::Dynamic<libloading::Library, egl::EGL1_5>>>,
    pub _gbm: Device<Card>,
    pub size: (usize, usize),
    pub format: FourCharCode,
    pub config: Config,
}

impl Headless {
    pub fn new_with_dest(
        mut width: usize,
        mut height: usize,
        is_planar: bool,
    ) -> Result<Headless, crate::Error> {
        if is_planar {
            if !width.is_multiple_of(4) {
                return Err(crate::Error::InvalidShape(
                    "OpenGL converter requires planar RGB width to be a multiple of 4".to_string(),
                ));
            }
            width /= 4;
            height *= 3;
        }

        // Create an EGL API instance.
        let lib = unsafe { libloading::Library::new("libEGL.so.1") }?;
        let egl = unsafe { egl::DynamicInstance::<egl::EGL1_5>::load_required_from(lib)? };

        // init a GBM device
        let gbm = Device::new(Card::open_global()?)?;

        debug!("gbm: {gbm:?}");
        let display = unsafe {
            egl.get_platform_display(
                egl_ext::PLATFORM_GBM_KHR,
                gbm.as_raw() as *mut c_void,
                &[egl::ATTRIB_NONE],
            )?
        };

        debug!("display: {display:?}");

        egl.initialize(display)?;
        let attributes = [
            egl::SURFACE_TYPE,
            egl::WINDOW_BIT,
            egl::RENDERABLE_TYPE,
            egl::OPENGL_ES2_BIT,
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

        let ctx = egl.create_context(display, config, None, &context_attributes)?;
        debug!("ctx: {ctx:?}");
        let gbm_surface: gbm::Surface<()> = gbm.create_surface(
            width as u32,
            height as u32,
            DrmFourcc::Abgr8888,
            gbm::BufferObjectFlags::RENDERING | gbm::BufferObjectFlags::SCANOUT,
        )?;
        debug!("gbm_surface: {gbm_surface:?}");
        let surface = match unsafe {
            egl.create_platform_window_surface(
                display,
                config,
                gbm_surface.as_raw() as *mut c_void,
                &[egl::ATTRIB_NONE],
            )
        } {
            Ok(v) => v,
            Err(e) => return Err(crate::Error::EGL(e)),
        };
        debug!("surface: {surface:?}");
        egl.make_current(display, Some(surface), Some(surface), Some(ctx))?;
        let _ = egl.swap_interval(display, 0);

        let headless = Headless {
            surface,
            display,
            ctx,
            gbm_surface,
            _gbm: gbm,
            egl: Rc::new(egl),
            config,
            size: (width, height),
            format: RGBA,
        };
        Ok(headless)
    }

    fn new_surface(
        &mut self,
        width: usize,
        height: usize,
        fourcc: FourCharCode,
    ) -> Result<(), crate::Error> {
        let _ = self.egl.destroy_surface(self.display, self.surface);

        let format = match fourcc {
            RGBA => DrmFourcc::Abgr8888,
            RGB => DrmFourcc::Bgr888,
            YUYV => DrmFourcc::Yuyv,
            _ => {
                return Err(Error::NotSupported(
                    "Destination format not supported".to_string(),
                ));
            }
        };

        let gbm_surface: gbm::Surface<()> = self._gbm.create_surface(
            width as u32,
            height as u32,
            format,
            gbm::BufferObjectFlags::RENDERING,
        )?;
        debug!("gbm_surface: {gbm_surface:?}");
        let surface = unsafe {
            self.egl.create_platform_window_surface(
                self.display,
                self.config,
                gbm_surface.as_raw() as *mut c_void,
                &[egl::ATTRIB_NONE],
            )?
        };
        debug!("surface: {surface:?}");

        self.gbm_surface = gbm_surface;
        self.surface = surface;
        self.size = (width, height);
        self.format = fourcc;
        self.make_current()?;
        Ok(())
    }

    fn make_current(&self) -> Result<(), crate::Error> {
        Ok(self.egl.make_current(
            self.display,
            Some(self.surface),
            Some(self.surface),
            Some(self.ctx),
        )?)
    }
}

impl Drop for Headless {
    fn drop(&mut self) {
        let _ = self.egl.destroy_surface(self.display, self.surface);
        let _ = self.egl.destroy_context(self.display, self.ctx);
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
        Ok(Card(options.open(path)?))
    }

    pub fn open_global() -> Result<Self, crate::Error> {
        Self::open("/dev/dri/card0")
    }
}

struct RegionOfInterest {
    left: f32,
    top: f32,
    right: f32,
    bottom: f32,
}

pub struct GLConverter {
    gbm_rendering: Headless,
    texture_program: GlProgram,
    texture_program_planar: GlProgram,
    camera_texture: Texture,
    vertex_buffer: Buffer,
    texture_buffer: Buffer,
}

impl ImageConverterTrait for GLConverter {
    fn convert(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: crate::Rotation,
        crop: Option<crate::Rect>,
    ) -> crate::Result<()> {
        check_gl_error().unwrap();
        if matches!(dst.tensor(), edgefirst_tensor::Tensor::Dma(_)) {
            return self.convert_dest_dma(dst, src, rotation, crop);
        }

        if dst.is_planar() {
            if self.gbm_rendering.size != (dst.width() / 4, dst.height() * 3)
                && self
                    .resize(dst.width() / 4, dst.height() * 3, RGBA)
                    .is_err()
            {
                return Err(Error::NotSupported(format!(
                    "Could not resize OpenGL context to {}x{} with format {}",
                    dst.width() / 4,
                    dst.height() * 3,
                    RGBA.display(),
                )));
            }
            self.convert_to_planar(src, crop)?;
        } else {
            if (self.gbm_rendering.size != (dst.width(), dst.height())
                || self.gbm_rendering.format != dst.fourcc())
                && self
                    .resize(dst.width(), dst.height(), dst.fourcc())
                    .is_err()
            {
                return Err(Error::NotSupported(format!(
                    "Could not resize OpenGL context to {}x{} with format {}",
                    dst.width(),
                    dst.height(),
                    dst.fourcc().display(),
                )));
            }
            self.convert_to(src, rotation, crop, true)?;
        }

        self.gbm_rendering
            .egl
            .swap_buffers(self.gbm_rendering.display, self.gbm_rendering.surface)?;

        let bo = unsafe { self.gbm_rendering.gbm_surface.lock_front_buffer()? };

        match &mut dst.tensor {
            edgefirst_tensor::Tensor::Shm(_shm_tensor) => {
                let mut mmap = _shm_tensor.map()?;
                bo.map(0, 0, bo.width(), bo.height(), |x| {
                    mmap.copy_from_slice(x.buffer())
                })?;
            }
            edgefirst_tensor::Tensor::Mem(_mem_tensor) => {
                let mut mmap = _mem_tensor.map()?;
                bo.map(0, 0, bo.width(), bo.height(), |x| {
                    mmap.copy_from_slice(x.buffer())
                })?;
            }
            edgefirst_tensor::Tensor::Dma(_) => unreachable!(),
        }

        Ok(())
    }
}

impl GLConverter {
    pub fn new() -> Result<GLConverter, crate::Error> {
        Self::new_with_size(640, 640, false)
    }

    pub fn new_with_size(
        width: usize,
        height: usize,
        is_planar: bool,
    ) -> Result<GLConverter, crate::Error> {
        let gbm_rendering = Headless::new_with_dest(width, height, is_planar)?;
        gls::load_with(|s| {
            gbm_rendering
                .egl
                .get_proc_address(s)
                .map_or(std::ptr::null(), |p| p as *const _)
        });
        let texture_program_planar =
            GlProgram::new(generate_vertex_shader(), generate_planar_rgb_shader())?;

        let texture_program =
            GlProgram::new(generate_vertex_shader(), generate_texture_fragment_shader())?;
        let camera_texture = Texture::new();

        let vertex_buffer = Buffer::new(0, 3, 100);
        let texture_buffer = Buffer::new(1, 2, 100);
        check_gl_error().unwrap();
        let converter = GLConverter {
            gbm_rendering,
            texture_program,
            texture_program_planar,
            camera_texture,
            vertex_buffer,
            texture_buffer,
        };
        converter.warmup(3)?;
        check_gl_error().unwrap();
        Ok(converter)
    }

    fn resize(&mut self, width: usize, height: usize, fourcc: FourCharCode) -> Result<(), Error> {
        self.gbm_rendering.new_surface(width, height, fourcc)?;
        unsafe {
            gls::gl::Viewport(0, 0, width as i32, height as i32);
        }
        self.warmup(1)
    }

    fn convert_dest_dma(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: crate::Rotation,
        crop: Option<crate::Rect>,
    ) -> crate::Result<()> {
        let frame_buffer = FrameBuffer::new();
        let render_buffer = RenderBuffer::new();

        render_buffer.bind();
        frame_buffer.bind();

        if dst.is_planar() {
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

            let dest_img = self.create_image_from_dma2(dst).unwrap();

            unsafe {
                gls::gl::EGLImageTargetRenderbufferStorageOES(
                    gls::gl::RENDERBUFFER,
                    dest_img.egl_image.as_ptr(),
                );
                gls::gl::FramebufferRenderbuffer(
                    gls::gl::FRAMEBUFFER,
                    gls::gl::COLOR_ATTACHMENT0,
                    gls::gl::RENDERBUFFER,
                    render_buffer.id,
                );
                gls::gl::Viewport(0, 0, width as i32, height as i32);
            }
            return self.convert_to_planar(src, crop);
        }

        let dest_img = self.create_image_from_dma2(dst).unwrap();
        unsafe {
            gls::gl::EGLImageTargetRenderbufferStorageOES(
                gls::gl::RENDERBUFFER,
                dest_img.egl_image.as_ptr(),
            );
            gls::gl::FramebufferRenderbuffer(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::RENDERBUFFER,
                render_buffer.id,
            );
            gls::gl::Viewport(0, 0, dst.width() as i32, dst.height() as i32);
        }
        self.convert_to(src, rotation, crop, false)
    }

    fn convert_to(
        &mut self,
        src: &TensorImage,
        rotation: crate::Rotation,
        crop: Option<crate::Rect>,
        flip: bool,
    ) -> Result<(), crate::Error> {
        check_gl_error().unwrap();
        // self.gbm_rendering.make_current()?;
        unsafe {
            gls::gl::ClearColor(1.0, 1.0, 1.0, 1.0);
            gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
        };

        let roi = if let Some(crop) = crop {
            // top and bottom are flipped because OpenGL uses 0,0 as bottom left
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

        let rotation_offset = match rotation {
            crate::Rotation::None => 0,
            crate::Rotation::Clockwise90 => 1,
            crate::Rotation::Rotate180 => 2,
            crate::Rotation::CounterClockwise90 => 3,
        };
        check_gl_error().unwrap();
        let result = if let Ok(new_egl_image) = self.create_image_from_dma2(src) {
            self.draw_camera_texture_eglimage(src, &new_egl_image, roi, rotation_offset, flip)
        } else {
            self.draw_camera_texture(src, roi, rotation_offset, flip)
        };
        unsafe { gls::gl::Finish() };
        result
    }

    fn warmup(&self, n: usize) -> Result<(), crate::Error> {
        let mut _bo;
        for _ in 0..n {
            unsafe {
                gls::gl::ClearColor(0.0, 0.0, 0.0, 0.0);
                gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
            };
            self.gbm_rendering
                .egl
                .swap_buffers(self.gbm_rendering.display, self.gbm_rendering.surface)?;
            _bo = unsafe { self.gbm_rendering.gbm_surface.lock_front_buffer()? };
        }
        Ok(())
    }

    fn convert_to_planar(
        &self,
        src: &TensorImage,
        crop: Option<crate::Rect>,
    ) -> Result<(), crate::Error> {
        if let Some(crop) = crop
            && (crop.left > 0
                || crop.top > 0
                || crop.height < src.height()
                || crop.width < src.width())
        {
            return Err(crate::Error::NotSupported(
                "Cropping in planar RGB mode is not supported".to_string(),
            ));
        }

        self.gbm_rendering.make_current()?;
        let new_egl_image = self.create_image_from_dma2(src)?;
        unsafe {
            gls::gl::ClearColor(0.0, 0.0, 0.0, 0.0);
            gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
        };
        self.draw_camera_texture_to_rgb_planar(
            &self.camera_texture,
            &new_egl_image,
            self.gbm_rendering.size.0,
        )?;
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
            check_gl_error()?;

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
                check_gl_error()?;
            }
        }
        Ok(())
    }

    fn draw_camera_texture(
        &mut self,
        img: &TensorImage,
        roi: RegionOfInterest,
        rotation_offset: usize,
        flip: bool,
    ) -> Result<(), Error> {
        let texture_target = gls::gl::TEXTURE_2D;
        let texture_format = match img.fourcc() {
            RGB => gls::gl::RGB,
            RGBA => gls::gl::RGBA,
            GREY => gls::gl::RED,
            _ => {
                return Err(Error::NotSupported(
                    "YUYV textures aren't supposed by OpenGL".to_string(),
                ));
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

            check_gl_error().unwrap();
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
            let mut cam = RegionOfInterest {
                left: -1.,
                bottom: -1.,
                right: 1.,
                top: 1.,
            };
            if flip {
                std::mem::swap(&mut cam.top, &mut cam.bottom);
            }
            let camera_vertices: [f32; 12] = [
                cam.left, cam.top, 0., // left top
                cam.right, cam.top, 0., // right top
                cam.right, cam.bottom, 0., // right bottom
                cam.left, cam.bottom, 0., // left bottom
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
                roi.left, roi.top, roi.right, roi.top, roi.right, roi.bottom, roi.left, roi.bottom,
                roi.left, roi.top, roi.right, roi.top, roi.right, roi.bottom, roi.left, roi.bottom,
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
            check_gl_error().unwrap();

            Ok(())
        }
    }

    fn draw_camera_texture_eglimage(
        &self,
        src: &TensorImage,
        egl_img: &EglImage,
        roi: RegionOfInterest,
        rotation_offset: usize,
        flip: bool,
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
            check_gl_error().unwrap();
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer.id);
            gls::gl::EnableVertexAttribArray(self.vertex_buffer.buffer_index);
            let mut cam = RegionOfInterest {
                left: -1.,
                bottom: -1.,
                right: 1.,
                top: 1.,
            };
            if flip {
                std::mem::swap(&mut cam.top, &mut cam.bottom);
            }
            let camera_vertices: [f32; 12] = [
                cam.left, cam.top, 0., // left top
                cam.right, cam.top, 0., // right top
                cam.right, cam.bottom, 0., // right bottom
                cam.left, cam.bottom, 0., // left bottom
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
                roi.left, roi.top, roi.right, roi.top, roi.right, roi.bottom, roi.left, roi.bottom,
                roi.left, roi.top, roi.right, roi.top, roi.right, roi.bottom, roi.left, roi.bottom,
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
        check_gl_error().unwrap();
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

        let egl_img_attr = [
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
            khronos_egl::NONE as Attrib,
        ];

        match self.new_egl_image_owned(
            egl_ext::LINUX_DMA_BUF,
            unsafe { egl::ClientBuffer::from_ptr(null_mut()) },
            &egl_img_attr,
        ) {
            Ok(v) => Ok(v),
            Err(e) => Err(crate::Error::EGL(e)),
        }
    }

    fn new_egl_image_owned(
        &'_ self,
        target: egl::Enum,
        buffer: egl::ClientBuffer,
        attrib_list: &[Attrib],
    ) -> Result<EglImage, egl::Error> {
        let image = self.gbm_rendering.egl.create_image(
            self.gbm_rendering.display,
            unsafe { egl::Context::from_ptr(egl::NO_CONTEXT) },
            target,
            buffer,
            attrib_list,
        )?;
        Ok(EglImage {
            egl_image: image,
            display: self.gbm_rendering.display,
            egl: self.gbm_rendering.egl.clone(),
        })
    }
}
struct EglImage {
    egl_image: egl::Image,
    egl: Rc<egl::Instance<egl::Dynamic<libloading::Library, egl::EGL1_5>>>,
    display: egl::Display,
}

impl Drop for EglImage {
    fn drop(&mut self) {
        let e = self.egl.destroy_image(self.display, self.egl_image);
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

pub struct RenderBuffer {
    id: u32,
}

impl RenderBuffer {
    pub fn new() -> RenderBuffer {
        let mut id = 0;
        unsafe {
            gls::gl::GenRenderbuffers(1, &raw mut id);
        }

        RenderBuffer { id }
    }

    pub fn bind(&self) {
        unsafe { gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, self.id) };
    }

    pub fn unbind(&self) {
        unsafe { gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, 0) };
    }
}

impl Drop for RenderBuffer {
    fn drop(&mut self) {
        self.unbind();
        unsafe { gls::gl::DeleteRenderbuffers(1, &raw mut self.id) };
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

fn check_gl_error() -> Result<(), Error> {
    unsafe {
        let err = gls::gl::GetError();
        if err != gls::gl::NO_ERROR {
            error!("GL Error: {err}");
            return Err(Error::OpenGl(format!("{err}")));
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
