// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::cache::BufferImportKey;
use super::platform::{GlPlatform, Platform};
use super::shaders::check_gl_error;
use std::ffi::{c_void, CStr};
use std::ptr::null;

pub(super) struct Texture {
    pub(super) id: u32,
    pub(super) target: gls::gl::types::GLenum,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) format: gls::gl::types::GLenum,
    /// Which EGLImage (identified by buffer identity key) is currently bound
    /// to this texture via `glEGLImageTargetTexture2DOES`. `None` means no
    /// EGLImage is bound (or the binding has been invalidated).
    bound_egl_key: Option<BufferImportKey>,
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
            bound_egl_key: None,
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
            // TexImage2D reallocates the texture, invalidating any EGLImage binding.
            self.bound_egl_key = None;
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

    /// Attach a platform import to this GL_TEXTURE_2D texture if the key
    /// differs from what's already bound. Returns `true` if the attach was
    /// performed, `false` if skipped (already bound). The binding-skip
    /// cache applies only where attachments persist
    /// (`GlPlatform::PERSISTENT_TEX_BINDINGS`) — on macOS every call
    /// attaches and the platform releases at its sync point.
    ///
    /// # Safety
    /// Caller must ensure the texture is bound to the active texture unit
    /// and `handle`'s import is alive (cache-owned).
    pub(super) unsafe fn bind_egl_image(
        &mut self,
        display: &<Platform as GlPlatform>::Display,
        key: BufferImportKey,
        handle: <Platform as GlPlatform>::ImportHandle,
    ) -> crate::Result<bool> {
        if Platform::PERSISTENT_TEX_BINDINGS && self.bound_egl_key == Some(key) {
            return Ok(false);
        }
        Platform::attach_tex_image_2d(display, handle)?;
        if Platform::PERSISTENT_TEX_BINDINGS {
            self.bound_egl_key = Some(key);
        }
        Ok(true)
    }

    /// Attach a platform import to this GL_TEXTURE_EXTERNAL_OES texture if
    /// the key differs from what's already bound. Returns `true` if the
    /// attach was performed, `false` if skipped. Errors on platforms
    /// without the OES extension (`PlatformCaps::external_oes` gates the
    /// path before it gets here).
    ///
    /// # Safety
    /// As [`Self::bind_egl_image`].
    pub(super) unsafe fn bind_egl_image_external(
        &mut self,
        display: &<Platform as GlPlatform>::Display,
        key: BufferImportKey,
        handle: <Platform as GlPlatform>::ImportHandle,
    ) -> crate::Result<bool> {
        if Platform::PERSISTENT_TEX_BINDINGS && self.bound_egl_key == Some(key) {
            return Ok(false);
        }
        Platform::attach_tex_image_external(display, handle)?;
        if Platform::PERSISTENT_TEX_BINDINGS {
            self.bound_egl_key = Some(key);
        }
        Ok(true)
    }

    /// Invalidate the cached EGL binding key. Must be called when the
    /// EGLImage cache evicts the entry this texture was using, or when
    /// `TexImage2D` overwrites the texture storage.
    pub(super) fn invalidate_egl_binding(&mut self) {
        self.bound_egl_key = None;
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
}

impl GlProgram {
    pub(super) fn new(vertex_shader: &str, fragment_shader: &str) -> Result<Self, crate::Error> {
        // Shared compile+link (deletes the shaders after a successful link) —
        // see `gl::core::compile_program`.
        let id = unsafe { super::core::compile_program(vertex_shader, fragment_shader)? };
        unsafe { gls::gl::UseProgram(id) };
        Ok(Self { id })
    }

    pub(super) fn load_uniform_1f(&self, name: &CStr, value: f32) -> Result<(), crate::Error> {
        unsafe {
            gls::gl::UseProgram(self.id);
            let location = gls::gl::GetUniformLocation(self.id, name.as_ptr());
            gls::gl::Uniform1f(location, value);
        }
        Ok(())
    }

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
        }));
    }
}
