// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::context::{Egl, GlContext};
use super::shaders::{check_gl_error, compile_shader_from_str};
use khronos_egl as egl;
use log::error;
use std::ffi::{c_void, CStr};
use std::ptr::null;
use std::rc::Rc;

pub(super) struct EglImage {
    pub(super) egl_image: egl::Image,
    pub(super) egl: Rc<Egl>,
    pub(super) display: egl::Display,
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

pub(super) struct Texture {
    pub(super) id: u32,
    pub(super) target: gls::gl::types::GLenum,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) format: gls::gl::types::GLenum,
}

impl Default for Texture {
    fn default() -> Self {
        Self::new()
    }
}

impl Texture {
    pub(super) fn new() -> Self {
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

    pub(super) fn update_texture(
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

pub(super) struct Buffer {
    pub(super) id: u32,
    pub(super) buffer_index: u32,
}

impl Buffer {
    pub(super) fn new(buffer_index: u32, size_per_point: usize, max_points: usize) -> Buffer {
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

pub(super) struct FrameBuffer {
    pub(super) id: u32,
}

impl FrameBuffer {
    pub(super) fn new() -> FrameBuffer {
        let mut id = 0;
        unsafe {
            gls::gl::GenFramebuffers(1, &raw mut id);
        }

        FrameBuffer { id }
    }

    pub(super) fn bind(&self) {
        unsafe { gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, self.id) };
    }

    pub(super) fn unbind(&self) {
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

pub(super) struct GlProgram {
    pub(super) id: u32,
    vertex_id: u32,
    fragment_id: u32,
}

impl GlProgram {
    pub(super) fn new(vertex_shader: &str, fragment_shader: &str) -> Result<Self, crate::Error> {
        let id = unsafe { gls::gl::CreateProgram() };
        let vertex_id = unsafe { gls::gl::CreateShader(gls::gl::VERTEX_SHADER) };
        if compile_shader_from_str(vertex_id, vertex_shader, "shader_vert").is_err() {
            log::debug!("Vertex shader source:\n{}", vertex_shader);
            unsafe {
                gls::gl::DeleteShader(vertex_id);
                gls::gl::DeleteProgram(id);
            }
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
            unsafe {
                gls::gl::DeleteShader(vertex_id);
                gls::gl::DeleteShader(fragment_id);
                gls::gl::DeleteProgram(id);
            }
            return Err(crate::Error::OpenGl(format!(
                "Shader compile error: {fragment_shader}"
            )));
        }

        unsafe {
            gls::gl::AttachShader(id, fragment_id);
            gls::gl::LinkProgram(id);

            let mut link_status = 0;
            gls::gl::GetProgramiv(id, gls::gl::LINK_STATUS, &raw mut link_status);
            if link_status == 0 {
                let mut log_len = 0;
                gls::gl::GetProgramiv(id, gls::gl::INFO_LOG_LENGTH, &raw mut log_len);
                let mut log_buf: Vec<u8> = vec![0; log_len as usize];
                gls::gl::GetProgramInfoLog(
                    id,
                    log_len,
                    std::ptr::null_mut(),
                    log_buf.as_mut_ptr() as *mut std::ffi::c_char,
                );
                let msg = String::from_utf8_lossy(&log_buf);
                log::error!("Program link failed: {msg}");
                gls::gl::DeleteShader(vertex_id);
                gls::gl::DeleteShader(fragment_id);
                gls::gl::DeleteProgram(id);
                return Err(crate::Error::OpenGl(format!("Program link error: {msg}")));
            }

            gls::gl::UseProgram(id);
        }

        Ok(Self {
            id,
            vertex_id,
            fragment_id,
        })
    }

    #[allow(dead_code)]
    pub(super) fn load_uniform_1f(&self, name: &CStr, value: f32) -> Result<(), crate::Error> {
        unsafe {
            gls::gl::UseProgram(self.id);
            let location = gls::gl::GetUniformLocation(self.id, name.as_ptr());
            gls::gl::Uniform1f(location, value);
        }
        Ok(())
    }

    #[allow(dead_code)]
    pub(super) fn load_uniform_1i(&self, name: &CStr, value: i32) -> Result<(), crate::Error> {
        unsafe {
            gls::gl::UseProgram(self.id);
            let location = gls::gl::GetUniformLocation(self.id, name.as_ptr());
            gls::gl::Uniform1i(location, value);
        }
        Ok(())
    }

    pub(super) fn load_uniform_4fv(
        &self,
        name: &CStr,
        value: &[[f32; 4]],
    ) -> Result<(), crate::Error> {
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
