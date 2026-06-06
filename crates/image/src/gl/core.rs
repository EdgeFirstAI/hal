// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Portable renderer helpers shared by both platform backends.
//!
//! Pure, platform-neutral logic (no `gbm`/DMA-BUF or IOSurface types) that both
//! the Linux (`gl::processor`) and macOS (`gl::macos_processor`) backends call,
//! so the shared GL render logic has one definition. Compiled on every platform
//! (the `gl` module is feature-gated on `opengl`, not on the OS).

use crate::{Crop, Error};
use std::ffi::CString;

/// Compile a GLSL shader of `kind` from source; returns the shader id.
///
/// On failure the shader is deleted and an [`Error::OpenGl`] carrying the
/// driver info-log is returned. The caller must have a current GL context.
///
/// # Safety
/// Requires a current GL context.
unsafe fn compile_shader(kind: u32, src: &str) -> crate::Result<u32> {
    let shader = gls::gl::CreateShader(kind);
    let c = CString::new(src).map_err(|e| Error::OpenGl(format!("shader CString: {e}")))?;
    let ptr = c.as_ptr();
    let len = src.len() as i32;
    gls::gl::ShaderSource(shader, 1, &ptr, &len);
    gls::gl::CompileShader(shader);
    let mut ok = 0i32;
    gls::gl::GetShaderiv(shader, gls::gl::COMPILE_STATUS, &mut ok);
    if ok == 0 {
        let mut log = [0u8; 4096];
        let mut log_len = 0i32;
        gls::gl::GetShaderInfoLog(shader, log.len() as i32, &mut log_len, log.as_mut_ptr() as *mut _);
        let msg = String::from_utf8_lossy(&log[..log_len.max(0) as usize]).into_owned();
        gls::gl::DeleteShader(shader);
        return Err(Error::OpenGl(format!(
            "shader compile failed (kind=0x{kind:x}): {msg}"
        )));
    }
    Ok(shader)
}

/// Compile + link a vertex/fragment program from source; returns the program id.
///
/// Shared by both backends. The two shaders are detached and deleted after a
/// successful link (GL frees them when the program is deleted), so the returned
/// program id is the only resource the caller must track. On any error the
/// partially-built shaders/program are cleaned up.
///
/// # Safety
/// Requires a current GL context.
pub(super) unsafe fn compile_program(vertex_src: &str, fragment_src: &str) -> crate::Result<u32> {
    let vs = compile_shader(gls::gl::VERTEX_SHADER, vertex_src)?;
    // Own `vs`/`fs`/`program` so any early return cleans them up.
    struct ProgramBuild {
        vs: Option<u32>,
        fs: Option<u32>,
        program: Option<u32>,
    }
    impl Drop for ProgramBuild {
        fn drop(&mut self) {
            unsafe {
                if let Some(p) = self.program {
                    gls::gl::DeleteProgram(p);
                }
                if let Some(s) = self.fs {
                    gls::gl::DeleteShader(s);
                }
                if let Some(s) = self.vs {
                    gls::gl::DeleteShader(s);
                }
            }
        }
    }
    let mut state = ProgramBuild {
        vs: Some(vs),
        fs: None,
        program: None,
    };

    let fs = compile_shader(gls::gl::FRAGMENT_SHADER, fragment_src)?;
    state.fs = Some(fs);

    let program = gls::gl::CreateProgram();
    state.program = Some(program);
    gls::gl::AttachShader(program, vs);
    gls::gl::AttachShader(program, fs);
    gls::gl::LinkProgram(program);
    let mut ok = 0i32;
    gls::gl::GetProgramiv(program, gls::gl::LINK_STATUS, &mut ok);
    if ok == 0 {
        let mut log = [0u8; 4096];
        let mut log_len = 0i32;
        gls::gl::GetProgramInfoLog(program, log.len() as i32, &mut log_len, log.as_mut_ptr() as *mut _);
        let msg = String::from_utf8_lossy(&log[..log_len.max(0) as usize]).into_owned();
        return Err(Error::OpenGl(format!("program link failed: {msg}")));
    }

    // Success: detach + delete shaders, disarm cleanup, return the program.
    gls::gl::DeleteShader(state.vs.take().unwrap());
    gls::gl::DeleteShader(state.fs.take().unwrap());
    let program = state.program.take().unwrap();
    std::mem::forget(state);
    Ok(program)
}

/// Set `MIN`/`MAG` filtering on the currently-bound texture of `target`.
///
/// Leaves `WRAP_S`/`WRAP_T` at the GL default (`REPEAT`) — correct for textures
/// sampled strictly inside `[0,1]` (the render-quad sources here). Use
/// [`set_tex_filter_clamp`] only where the original code also set the wrap mode.
///
/// # Safety
/// Requires a current GL context and a bound texture on `target`.
pub(super) unsafe fn set_tex_filter(target: u32, filter: u32) {
    gls::gl::TexParameteri(target, gls::gl::TEXTURE_MIN_FILTER, filter as i32);
    gls::gl::TexParameteri(target, gls::gl::TEXTURE_MAG_FILTER, filter as i32);
}

/// Set `MIN`/`MAG` filtering and `CLAMP_TO_EDGE` `WRAP_S`/`WRAP_T` on the
/// currently-bound texture of `target` (the four-call cluster).
///
/// # Safety
/// Requires a current GL context and a bound texture on `target`.
pub(super) unsafe fn set_tex_filter_clamp(target: u32, filter: u32) {
    set_tex_filter(target, filter);
    gls::gl::TexParameteri(target, gls::gl::TEXTURE_WRAP_S, gls::gl::CLAMP_TO_EDGE as i32);
    gls::gl::TexParameteri(target, gls::gl::TEXTURE_WRAP_T, gls::gl::CLAMP_TO_EDGE as i32);
}

/// Check that the currently-bound draw framebuffer is complete.
///
/// Returns `Err(status)` with the raw `GL_FRAMEBUFFER_*` status code when the
/// framebuffer is not `FRAMEBUFFER_COMPLETE`, so each backend can format its own
/// context-specific error (and, on Linux, unbind before falling back to CPU).
///
/// # Safety
/// Requires a current GL context with a framebuffer bound to `FRAMEBUFFER`.
pub(super) unsafe fn check_framebuffer_complete() -> Result<(), u32> {
    let status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
    if status == gls::gl::FRAMEBUFFER_COMPLETE {
        Ok(())
    } else {
        Err(status)
    }
}

/// Crop-derived float render uniforms shared by both float render paths.
///
/// Computes `(src_rect_uv, dst_rect_px, pad_color)` from `crop` after running
/// [`Crop::check_crop_dims`] (the dimension validation must happen first, as it
/// can error). `src_rect_uv` is normalized to source dims; `dst_rect_px` is in
/// single-plane pixel coords; `pad_color` is normalized `[0,1]`. When a rect is
/// `None` the whole image is sampled/painted (identity transform).
pub(super) fn float_crop_uniforms(
    crop: &Crop,
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> crate::Result<([f32; 4], [f32; 4], [f32; 3])> {
    crop.check_crop_dims(src_w, src_h, dst_w, dst_h)?;
    let src_rect_uv = match crop.src_rect {
        Some(r) => [
            r.left as f32 / src_w as f32,
            r.top as f32 / src_h as f32,
            r.width as f32 / src_w as f32,
            r.height as f32 / src_h as f32,
        ],
        None => [0.0, 0.0, 1.0, 1.0],
    };
    let dst_rect_px = match crop.dst_rect {
        Some(r) => [r.left as f32, r.top as f32, r.width as f32, r.height as f32],
        None => [0.0, 0.0, dst_w as f32, dst_h as f32],
    };
    let pad_color = match crop.dst_color {
        Some([r, g, b, _]) => [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0],
        None => [0.0, 0.0, 0.0],
    };
    Ok((src_rect_uv, dst_rect_px, pad_color))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::float_crop_uniforms;
    use crate::{Crop, Rect};

    #[test]
    fn identity_crop_is_full_image() {
        let (src_uv, dst_px, pad) =
            float_crop_uniforms(&Crop::no_crop(), 640, 480, 320, 240).unwrap();
        assert_eq!(src_uv, [0.0, 0.0, 1.0, 1.0]);
        assert_eq!(dst_px, [0.0, 0.0, 320.0, 240.0]);
        assert_eq!(pad, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn src_rect_normalizes_to_source_dims() {
        let crop = Crop {
            src_rect: Some(Rect::new(160, 120, 320, 240)),
            ..Crop::no_crop()
        };
        let (src_uv, _, _) = float_crop_uniforms(&crop, 640, 480, 320, 240).unwrap();
        assert_eq!(src_uv, [0.25, 0.25, 0.5, 0.5]);
    }

    #[test]
    fn dst_rect_is_pixel_coords_and_pad_color_normalizes() {
        let crop = Crop {
            dst_rect: Some(Rect::new(10, 20, 100, 50)),
            dst_color: Some([255, 128, 0, 255]),
            ..Crop::no_crop()
        };
        let (_, dst_px, pad) = float_crop_uniforms(&crop, 640, 480, 320, 240).unwrap();
        assert_eq!(dst_px, [10.0, 20.0, 100.0, 50.0]);
        assert_eq!(pad, [1.0, 128.0 / 255.0, 0.0]);
    }

    #[test]
    fn out_of_bounds_src_rect_errors() {
        // src_rect exceeds the source extent → check_crop_dims rejects it first.
        let crop = Crop {
            src_rect: Some(Rect::new(6, 0, 4, 4)), // 6 + 4 = 10 > src_w = 8
            ..Crop::no_crop()
        };
        let err = float_crop_uniforms(&crop, 8, 8, 8, 8).unwrap_err();
        assert!(
            matches!(err, crate::Error::CropInvalid(_)),
            "expected CropInvalid, got {err:?}"
        );
    }
}
