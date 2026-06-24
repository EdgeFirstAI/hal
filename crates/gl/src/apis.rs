//! Safe-ish wrappers over the raw `gl` bindings.
//!
//! Trimmed from upstream `gls` to only the functions the EdgeFirst HAL uses.

// The texture-image wrappers mirror the GL entry points' arity by design.
#![allow(clippy::too_many_arguments)]

use crate::{gl, Error, GLchar, GLeglImageOES, GLenum, GLint, GLsizei, GLuint};
use std::ffi::CStr;

/// Select active texture unit.
pub fn active_texture(texture: GLenum) {
    unsafe { gl::ActiveTexture(texture) }
}

/// Bind a named texture to a texturing target.
pub fn bind_texture(target: GLenum, texture: GLuint) {
    unsafe { gl::BindTexture(target, texture) }
}

/// Specify pixel arithmetic for RGB and alpha components separately.
pub fn blend_func_separate(
    sfactor_rgb: GLenum,
    dfactor_rgb: GLenum,
    sfactor_alpha: GLenum,
    dfactor_alpha: GLenum,
) {
    unsafe { gl::BlendFuncSeparate(sfactor_rgb, dfactor_rgb, sfactor_alpha, dfactor_alpha) }
}

/// Disable a server-side GL capability.
pub fn disable(cap: GLenum) {
    unsafe { gl::Disable(cap) }
}

/// Define a 2D texture image from an `EGLImage` (zero-copy import).
// `image` is an opaque EGL handle passed straight to the driver; keeping this a
// safe wrapper matches upstream `gls` and the HAL's call sites.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub fn egl_image_target_texture_2d_oes(target: GLenum, image: GLeglImageOES) {
    unsafe { gl::EGLImageTargetTexture2DOES(target, image) }
}

/// Enable a server-side GL capability.
pub fn enable(cap: GLenum) {
    unsafe { gl::Enable(cap) }
}

/// Block until all GL execution is complete.
pub fn finish() {
    unsafe { gl::Finish() }
}

/// Return a string describing the current GL connection.
pub fn get_string(name: GLenum) -> Result<String, Error> {
    unsafe {
        let name: *const GLchar = gl::GetString(name) as *const GLchar;
        if name.is_null() {
            Err(Error::new())
        } else {
            Ok(CStr::from_ptr(name).to_string_lossy().into_owned())
        }
    }
}

/// Specify a two-dimensional texture image.
pub fn tex_image2d<T>(
    target: GLenum,
    level: GLint,
    internalformat: GLint,
    width: GLsizei,
    height: GLsizei,
    border: GLint,
    format: GLenum,
    type_: GLenum,
    pixels: Option<&[T]>,
) where
    T: Sized,
{
    let ptr = match pixels {
        Some(v) => v.as_ptr() as *const core::ffi::c_void,
        None => std::ptr::null(),
    };
    unsafe {
        gl::TexImage2D(
            target,
            level,
            internalformat,
            width,
            height,
            border,
            format,
            type_,
            ptr,
        );
    }
}

/// Specify a three-dimensional texture image.
pub fn tex_image3d<T>(
    target: GLenum,
    level: GLint,
    internalformat: GLint,
    width: GLsizei,
    height: GLsizei,
    depth: GLsizei,
    border: GLint,
    format: GLenum,
    type_: GLenum,
    pixels: Option<&[T]>,
) where
    T: Sized,
{
    let ptr = match pixels {
        Some(v) => v.as_ptr() as *const core::ffi::c_void,
        None => std::ptr::null(),
    };
    unsafe {
        gl::TexImage3D(
            target,
            level,
            internalformat,
            width,
            height,
            depth,
            border,
            format,
            type_,
            ptr,
        );
    }
}

/// Set an integer texture parameter.
pub fn tex_parameteri(target: GLenum, pname: GLenum, param: GLint) {
    unsafe {
        gl::TexParameteri(target, pname, param);
    }
}

/// Install a program object as part of current rendering state.
pub fn use_program(program: GLuint) {
    unsafe {
        gl::UseProgram(program);
    }
}
