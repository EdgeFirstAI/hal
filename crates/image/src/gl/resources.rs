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
    /// Which import (identified by buffer identity key) is bound to this
    /// texture, and the platform's binding epoch when it was bound
    /// (`GlPlatform::tex_binding_epoch`). `None` means no import is bound,
    /// the binding has been invalidated, or the platform re-attaches on
    /// every use.
    bound_egl: Option<(BufferImportKey, u64)>,
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
            bound_egl: None,
        }
    }

    /// Upload `src` as this texture's contents, reallocating storage only
    /// when the target, size or format changed since the last upload.
    ///
    /// `required` is how many bytes GL will read: `(height - 1)` row strides
    /// plus the last row's own pixels, where the stride is whatever
    /// `UNPACK_ROW_LENGTH` the caller has set. An [`UploadSource::Bytes`]
    /// shorter than that is refused with `NotSupported`.
    ///
    /// `format` is also the internal format, except that `RED` and `RG` are
    /// allocated as the sized `R8` and `RG8`.
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
            // argument is a byte offset into it.
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
                self.bound_egl = None;
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

    /// Forget the recorded storage: the next [`Self::upload`] allocates
    /// afresh with `TexImage2D`.
    pub(super) fn forget_storage(&mut self) {
        self.target = 0;
    }

    /// Attach a platform import to this GL_TEXTURE_2D texture unless the
    /// same key is still bound in the platform's current binding epoch
    /// (`GlPlatform::tex_binding_epoch`). Returns `true` if the attach was
    /// performed, `false` if skipped (already bound). Where the platform has
    /// no epoch, every call attaches.
    ///
    /// # Safety
    /// Caller must ensure the texture is bound to the active texture unit
    /// and `handle`'s import is alive.
    pub(super) unsafe fn bind_egl_image(
        &mut self,
        display: &<Platform as GlPlatform>::Display,
        key: BufferImportKey,
        handle: <Platform as GlPlatform>::ImportHandle,
    ) -> crate::Result<bool> {
        let bound = Platform::tex_binding_epoch(display).map(|epoch| (key, epoch));
        if bound.is_some() && self.bound_egl == bound {
            return Ok(false);
        }
        unsafe { Platform::attach_tex_image_2d(display, handle)? };
        self.bound_egl = bound;
        Ok(true)
    }

    /// Attach a platform import to this GL_TEXTURE_EXTERNAL_OES texture if
    /// the key differs from what's already bound. Returns `true` if the
    /// attach was performed, `false` if skipped. Errors on platforms
    /// without the OES extension.
    ///
    /// # Safety
    /// As [`Self::bind_egl_image`].
    pub(super) unsafe fn bind_egl_image_external(
        &mut self,
        display: &<Platform as GlPlatform>::Display,
        key: BufferImportKey,
        handle: <Platform as GlPlatform>::ImportHandle,
    ) -> crate::Result<bool> {
        let bound = Platform::tex_binding_epoch(display).map(|epoch| (key, epoch));
        if bound.is_some() && self.bound_egl == bound {
            return Ok(false);
        }
        unsafe { Platform::attach_tex_image_external(display, handle)? };
        self.bound_egl = bound;
        Ok(true)
    }

    /// Invalidate the cached EGL binding key. Must be called when the
    /// EGLImage cache evicts the entry this texture was using, or when
    /// `TexImage2D` overwrites the texture storage.
    pub(super) fn invalidate_egl_binding(&mut self) {
        self.bound_egl = None;
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

/// The setup a [`LazyProgram`] runs once after its link, with the program
/// bound: it loads constant uniforms and returns the link-time state `L`.
type ProgramSetup<L> = Box<dyn Fn(&GlProgram) -> Result<L, crate::Error>>;

/// A GL program over the shared vertex shader and `fragment`, compiled and
/// linked the first time [`get`](Self::get) is called on the GL thread.
///
/// A failed build is logged once and kept: every later `get` returns
/// [`crate::Error::NotSupported`] without building again.
pub(super) struct LazyProgram<L = ()> {
    fragment: std::borrow::Cow<'static, str>,
    setup: ProgramSetup<L>,
    linked: std::cell::OnceCell<Result<(GlProgram, L), String>>,
}

impl<L> LazyProgram<L> {
    /// A program over `fragment` whose `setup` runs after the link.
    pub(super) fn new(
        fragment: impl Into<std::borrow::Cow<'static, str>>,
        setup: impl Fn(&GlProgram) -> Result<L, crate::Error> + 'static,
    ) -> Self {
        Self {
            fragment: fragment.into(),
            setup: Box::new(setup),
            linked: std::cell::OnceCell::new(),
        }
    }

    /// The program, linked and set up now.
    ///
    /// # Errors
    ///
    /// The compile, link or setup error. Nothing is logged or kept.
    pub(super) fn link_now(self) -> Result<Self, crate::Error> {
        let built = self.build()?;
        let _ = self.linked.set(Ok(built));
        Ok(self)
    }

    /// Compiles, links and sets up the program, leaving the program bound
    /// before the call bound.
    fn build(&self) -> Result<(GlProgram, L), crate::Error> {
        let mut bound = 0i32;
        // SAFETY: on the GL thread with the processor's context current.
        unsafe { edgefirst_gl::gl::GetIntegerv(edgefirst_gl::gl::CURRENT_PROGRAM, &mut bound) };
        let built = GlProgram::new(super::shaders::generate_vertex_shader(), &self.fragment)
            .and_then(|program| (self.setup)(&program).map(|state| (program, state)));
        // SAFETY: as above; `bound` was the current program or 0.
        unsafe { edgefirst_gl::gl::UseProgram(bound as u32) };
        built
    }

    /// The program and its link-time state, linking it on the first call.
    /// The program bound before the call stays bound.
    ///
    /// # Errors
    ///
    /// [`crate::Error::NotSupported`] when the program fails to compile, link
    /// or set up, on this call and every later one.
    pub(super) fn get(&self) -> Result<(&GlProgram, &L), crate::Error> {
        let linked = self.linked.get_or_init(|| {
            self.build().map_err(|e| {
                log::warn!(
                    "a GL program failed to build; conversions that need it decline on GL: {e}"
                );
                e.to_string()
            })
        });
        match linked {
            Ok((program, state)) => Ok((program, state)),
            Err(e) => Err(crate::Error::NotSupported(format!(
                "a GL program failed to build: {e}"
            ))),
        }
    }

    /// The program, linking it on the first call; see [`get`](Self::get).
    pub(super) fn program(&self) -> Result<&GlProgram, crate::Error> {
        self.get().map(|(program, _)| program)
    }

    /// The program's GL name, linking it on the first call; see
    /// [`get`](Self::get).
    pub(super) fn id(&self) -> Result<u32, crate::Error> {
        self.get().map(|(program, _)| program.id)
    }

    /// The program's link-time state, linking it on the first call; see
    /// [`get`](Self::get).
    pub(super) fn state(&self) -> Result<&L, crate::Error> {
        self.get().map(|(_, state)| state)
    }

    /// The program if it is linked, without building it.
    pub(super) fn linked(&self) -> Option<&GlProgram> {
        match self.linked.get() {
            Some(Ok((program, _))) => Some(program),
            _ => None,
        }
    }

    /// Whether a build was attempted, successful or not.
    #[cfg(test)]
    pub(super) fn is_built(&self) -> bool {
        self.linked.get().is_some()
    }

    /// Records a failed build with `message` in an unbuilt program.
    #[cfg(test)]
    pub(super) fn fail_build(&self, message: &str) {
        let _ = self.linked.set(Err(message.to_owned()));
    }
}

impl LazyProgram<()> {
    /// A program over `fragment` with no setup.
    pub(super) fn plain(fragment: impl Into<std::borrow::Cow<'static, str>>) -> Self {
        Self::new(fragment, |_| Ok(()))
    }

    /// A program over `fragment` whose sampler `name` reads texture unit
    /// `unit`.
    pub(super) fn sampling(
        fragment: impl Into<std::borrow::Cow<'static, str>>,
        name: &'static CStr,
        unit: i32,
    ) -> Self {
        Self::new(fragment, move |program| program.load_uniform_1i(name, unit))
    }
}
