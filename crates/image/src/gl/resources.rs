// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::cache::BufferImportKey;
use super::platform::{GlPlatform, Platform};
use super::shaders::check_gl_error;
use std::ffi::{c_void, CStr};
use std::ptr::null;

/// Where [`Texture::upload`] reads its pixels from.
#[derive(Clone, Copy)]
pub(super) enum UploadSource<'a> {
    /// CPU-visible bytes, starting at the image's first pixel.
    Bytes(&'a [u8]),
    /// A GL buffer object, read through `PIXEL_UNPACK_BUFFER` from `offset`
    /// bytes in: a copy the GPU makes, with no CPU mapping.
    Pbo { buffer_id: u32, offset: usize },
}

impl<'a> UploadSource<'a> {
    /// The same source, starting `bytes` further in. Bytes past the end
    /// leave an empty window, which [`Texture::upload`] refuses.
    pub(super) fn offset_by(self, bytes: usize) -> Self {
        match self {
            Self::Bytes(data) => Self::Bytes(data.get(bytes..).unwrap_or(&[])),
            Self::Pbo { buffer_id, offset } => Self::Pbo {
                buffer_id,
                offset: offset + bytes,
            },
        }
    }
}

pub(super) struct Texture {
    pub(super) id: u32,
    pub(super) target: edgefirst_gl::gl::types::GLenum,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) format: edgefirst_gl::gl::types::GLenum,
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
        unsafe { edgefirst_gl::gl::GenTextures(1, &raw mut id) };
        Self {
            id,
            target: 0,
            width: 0,
            height: 0,
            format: 0,
            bound_egl_key: None,
        }
    }

    /// Upload `src` as this texture's contents, reallocating storage only
    /// when the target, size or format changed since the last upload.
    ///
    /// `required` is how many bytes GL will read: `(height - 1)` row strides
    /// plus the last row's own pixels, where the stride is whatever
    /// `UNPACK_ROW_LENGTH` the caller has set (`UNPACK_ALIGNMENT` is 1 from
    /// `new`, so rows carry no extra padding). GL reads through a raw
    /// pointer and cannot see a slice's length, so [`UploadSource::Bytes`]
    /// shorter than that would have the driver read past the mapping --
    /// which is not a hypothetical: a tensor's map is clamped to what its
    /// window actually covers (`HostView::len_elems`), so a plane offset
    /// restored from an untrusted descriptor shortens the map while width
    /// and height stay whole. Refusing here turns that into a fallback to
    /// the CPU converter instead of an out-of-bounds read. See
    /// `crates/tensor/src/d3d11/texture.rs::check_view_offset` and
    /// `IoSurfaceTensor::check_view_offset`, which refuse the same window at
    /// the source; this is the check that does not depend on which backing
    /// produced the map. An [`UploadSource::Pbo`] read past its buffer is
    /// refused by GL itself (`GL_INVALID_OPERATION`), which the caller's
    /// `check_gl_error` reports.
    ///
    /// `format` is also the internal format, except that `RED` and `RG` are
    /// allocated as the sized `R8` and `RG8`: OpenGL ES 3.0 has no unsized
    /// internal format for either.
    pub(super) fn upload(
        &mut self,
        target: edgefirst_gl::gl::types::GLenum,
        width: usize,
        height: usize,
        format: edgefirst_gl::gl::types::GLenum,
        required: usize,
        src: UploadSource<'_>,
    ) -> crate::Result<()> {
        let (pixels, pbo) = match src {
            UploadSource::Bytes(data) => {
                if data.len() < required {
                    return Err(crate::Error::NotSupported(format!(
                        "GL upload: {width}x{height} texture needs {required} B but the \
                         source maps only {} B -- a window that does not cover its own \
                         image (a restored plane offset past the buffer?); converting \
                         on the CPU instead",
                        data.len()
                    )));
                }
                (data.as_ptr() as *const c_void, None)
            }
            // With a buffer bound to `PIXEL_UNPACK_BUFFER`, the `pixels`
            // argument is a byte offset into it rather than an address.
            UploadSource::Pbo { buffer_id, offset } => (offset as *const c_void, Some(buffer_id)),
        };
        let internal_format = match format {
            edgefirst_gl::gl::RED => edgefirst_gl::gl::R8,
            edgefirst_gl::gl::RG => edgefirst_gl::gl::RG8,
            other => other,
        };
        unsafe {
            if let Some(buffer_id) = pbo {
                edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_UNPACK_BUFFER, buffer_id);
            }
            if target != self.target
                || width != self.width
                || height != self.height
                || format != self.format
            {
                edgefirst_gl::gl::TexImage2D(
                    target,
                    0,
                    internal_format as i32,
                    width as i32,
                    height as i32,
                    0,
                    format,
                    edgefirst_gl::gl::UNSIGNED_BYTE,
                    pixels,
                );
                self.target = target;
                self.format = format;
                self.width = width;
                self.height = height;
                // TexImage2D reallocates the texture, invalidating any EGLImage binding.
                self.bound_egl_key = None;
            } else {
                edgefirst_gl::gl::TexSubImage2D(
                    target,
                    0,
                    0,
                    0,
                    width as i32,
                    height as i32,
                    format,
                    edgefirst_gl::gl::UNSIGNED_BYTE,
                    pixels,
                );
            }
            if pbo.is_some() {
                edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::PIXEL_UNPACK_BUFFER, 0);
            }
        }
        Ok(())
    }

    /// Forget the recorded storage so the next [`Self::upload`] allocates
    /// afresh with `TexImage2D` instead of writing into whatever the texture
    /// holds now -- which, after an attach, is a client buffer.
    pub(super) fn forget_storage(&mut self) {
        self.target = 0;
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
        unsafe { Platform::attach_tex_image_2d(display, handle)? };
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
        unsafe { Platform::attach_tex_image_external(display, handle)? };
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
            edgefirst_gl::gl::DeleteTextures(1, &raw mut self.id)
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
            edgefirst_gl::gl::EnableVertexAttribArray(buffer_index);
            edgefirst_gl::gl::GenBuffers(1, &raw mut id);
            edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, id);
            edgefirst_gl::gl::VertexAttribPointer(
                buffer_index,
                size_per_point as i32,
                edgefirst_gl::gl::FLOAT,
                edgefirst_gl::gl::FALSE,
                0,
                null(),
            );
            edgefirst_gl::gl::BufferData(
                edgefirst_gl::gl::ARRAY_BUFFER,
                (size_of::<f32>() * size_per_point * max_points) as isize,
                null(),
                edgefirst_gl::gl::DYNAMIC_DRAW,
            );
        }

        Buffer { id, buffer_index }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            edgefirst_gl::gl::DeleteBuffers(1, &raw mut self.id)
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
            edgefirst_gl::gl::GenFramebuffers(1, &raw mut id);
        }

        FrameBuffer { id }
    }

    pub(super) fn bind(&self) {
        unsafe { edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, self.id) };
    }

    pub(super) fn unbind(&self) {
        unsafe { edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, 0) };
    }
}

impl Drop for FrameBuffer {
    fn drop(&mut self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.unbind();
            unsafe {
                edgefirst_gl::gl::DeleteFramebuffers(1, &raw mut self.id);
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
        unsafe { edgefirst_gl::gl::UseProgram(id) };
        Ok(Self { id })
    }

    pub(super) fn load_uniform_1f(&self, name: &CStr, value: f32) -> Result<(), crate::Error> {
        unsafe {
            edgefirst_gl::gl::UseProgram(self.id);
            let location = edgefirst_gl::gl::GetUniformLocation(self.id, name.as_ptr());
            edgefirst_gl::gl::Uniform1f(location, value);
        }
        Ok(())
    }

    pub(super) fn load_uniform_1i(&self, name: &CStr, value: i32) -> Result<(), crate::Error> {
        unsafe {
            edgefirst_gl::gl::UseProgram(self.id);
            let location = edgefirst_gl::gl::GetUniformLocation(self.id, name.as_ptr());
            edgefirst_gl::gl::Uniform1i(location, value);
        }
        Ok(())
    }

    pub(super) fn load_uniform_4fv(
        &self,
        name: &CStr,
        value: &[[f32; 4]],
    ) -> Result<(), crate::Error> {
        unsafe {
            edgefirst_gl::gl::UseProgram(self.id);
            let location = edgefirst_gl::gl::GetUniformLocation(self.id, name.as_ptr());
            if location == -1 {
                return Err(crate::Error::OpenGl(format!(
                    "Could not find uniform location for '{}'",
                    name.to_string_lossy().into_owned()
                )));
            }
            edgefirst_gl::gl::Uniform4fv(
                location,
                value.len() as i32,
                value.as_flattened().as_ptr(),
            );
        }
        check_gl_error(function!(), line!())?;
        Ok(())
    }
}

impl Drop for GlProgram {
    fn drop(&mut self) {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            edgefirst_gl::gl::DeleteProgram(self.id);
        }));
    }
}
