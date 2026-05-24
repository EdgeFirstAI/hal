// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! macOS GL image processor backed by ANGLE + IOSurface.
//!
//! Mirrors the role of `GLProcessorThreaded` on Linux — the parallel
//! Linux implementation lives in `gl/threaded.rs` and is structurally
//! more elaborate because of vendor-driver thread-safety constraints
//! (Vivante galcore in particular). ANGLE's Metal backend is
//! thread-safe enough that we run GL inline under a mutex instead of
//! through a dedicated thread + command channel.
//!
//! Format coverage in this initial implementation:
//!   * YUYV → RGBA / BGRA — full shader-based BT.709 limited-range conversion
//!
//! Other format pairs and the mask-rendering / decoder paths return
//! `NotImplemented` and fall back to the CPU backend, matching the
//! contract the Linux backend uses for unsupported combinations on a
//! given GPU driver.
//!
//! See `crates/image/src/gl/platform/macos.rs` for the platform layer
//! this processor builds on, and `crates/image/src/gl/iosurface_import.rs`
//! for the IOSurface allocation + EGL pbuffer attribute setup.

#![cfg(target_os = "macos")]

use super::iosurface_import;
use super::platform::macos::MacosPlatform;
use super::platform::GlPlatform;
use super::Egl;
use crate::{Crop, Error, Flip, ImageProcessorTrait, MaskOverlay, Result, Rotation};
use edgefirst_decoder::{DetectBox, ProtoData, Segmentation};
use edgefirst_tensor::{PixelFormat, TensorDyn};
use khronos_egl as egl;
use log::debug;
use std::ffi::{c_void, CString};
use std::sync::{Mutex, OnceLock};

// ---------------------------------------------------------------------------
// EGL constants reused across the macOS path. The "production" constants in
// `super::iosurface_import` cover the IOSurface-pbuffer attribute set; these
// are the additional constants needed at MacosGlProcessor::new time.
// ---------------------------------------------------------------------------

const EGL_OPENGL_ES3_BIT: i32 = 0x0040;
const EGL_PBUFFER_BIT: i32 = 0x0001;
const EGL_RENDERABLE_TYPE: i32 = 0x3040;
const EGL_SURFACE_TYPE: i32 = 0x3033;
const EGL_RED_SIZE: i32 = 0x3024;
const EGL_GREEN_SIZE: i32 = 0x3023;
const EGL_BLUE_SIZE: i32 = 0x3022;
const EGL_ALPHA_SIZE: i32 = 0x3021;
const EGL_CONTEXT_CLIENT_VERSION: i32 = 0x3098;
const EGL_BACK_BUFFER: i32 = 0x3084;

// ---------------------------------------------------------------------------
// Shaders. YUYV-as-GL_RG sampling: each source texel is (Y, C) where C
// alternates U/V every other column. We sample the current and partner
// texel to recover both chroma values for each output pixel, then apply
// the BT.709 limited-range matrix.
//
// The shader matches the spike at `spikes/angle_iosurface/`. Bit-near-
// exact (≤1 LSB) match to the CPU scalar reference was validated there.
// ---------------------------------------------------------------------------

const VERTEX_SHADER: &str = r#"#version 300 es
precision mediump float;
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv_in;
out vec2 v_uv;
void main() {
    v_uv = uv_in;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"#;

const YUYV_TO_RGBA_FRAGMENT: &str = r#"#version 300 es
precision mediump float;
uniform sampler2D src;
uniform vec2 src_size;
in vec2 v_uv;
out vec4 frag;

void main() {
    vec2 texel = vec2(1.0) / src_size;
    vec2 col = floor(v_uv * src_size);
    bool even = mod(col.x, 2.0) < 0.5;
    vec2 self_uv = (col + vec2(0.5)) * texel;
    vec2 pair_uv = (col + vec2(even ? 1.5 : -0.5, 0.5)) * texel;

    vec4 self_rg = texture(src, self_uv);
    vec4 pair_rg = texture(src, pair_uv);
    float y = self_rg.r;
    float u, v;
    if (even) { u = self_rg.g; v = pair_rg.g; }
    else      { v = self_rg.g; u = pair_rg.g; }

    float yp = (y * 255.0 - 16.0) * (1.164 / 255.0);
    float up = u - 128.0/255.0;
    float vp = v - 128.0/255.0;
    float r = clamp(yp + 1.793 * vp, 0.0, 1.0);
    float g = clamp(yp - 0.213 * up - 0.533 * vp, 0.0, 1.0);
    float b = clamp(yp + 2.112 * up, 0.0, 1.0);
    frag = vec4(r, g, b, 1.0);
}
"#;

// ---------------------------------------------------------------------------
// One-shot GL function-pointer table.
//
// `gls::load_with` populates global function pointers — exists once per
// process. We load via EGL's `eglGetProcAddress` so the symbols come
// from ANGLE's libGLESv2.dylib.
// ---------------------------------------------------------------------------

static GL_LOADED: OnceLock<()> = OnceLock::new();

fn load_gl_once(egl: &Egl) {
    GL_LOADED.get_or_init(|| {
        gls::load_with(|name| match egl.get_proc_address(name) {
            Some(ptr) => ptr as *const c_void,
            None => std::ptr::null(),
        });
    });
}

// ---------------------------------------------------------------------------
// The processor itself.
//
// Holds: EGL display + config + context, the compiled YUYV→RGBA program,
// a fullscreen-quad VAO/VBO, an FBO for off-screen rendering, and a
// pair of GL textures used for transient binding of the source/dest
// IOSurface pbuffers.
//
// GL state is shared across calls to amortize shader compilation and
// VAO/FBO setup. A mutex serializes calls to `convert` so EGL state
// changes (eglMakeCurrent, eglBindTexImage) are not racing.
// ---------------------------------------------------------------------------

pub struct MacosGlProcessor {
    egl: Egl,
    display: egl::Display,
    config: egl::Config,
    context: egl::Context,
    /// Tiny scratch surface kept alive so the context can be made
    /// current outside of a `convert` call (e.g. for shader recompile).
    dummy_pbuffer: egl::Surface,

    program_yuyv_to_rgba: u32,
    uniform_src: i32,
    uniform_src_size: i32,
    vao: u32,
    vbo: u32,
    fbo: u32,
    src_tex: u32,
    dst_tex: u32,

    lock: Mutex<()>,
}

// SAFETY: MacosGlProcessor's only non-`Sync` state is the EGL context,
// which we serialize behind `lock`. Calls to GL/EGL go through the
// mutex via `with_current_context`.
unsafe impl Send for MacosGlProcessor {}
unsafe impl Sync for MacosGlProcessor {}

impl std::fmt::Debug for MacosGlProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MacosGlProcessor")
            .field("backend", &"ANGLE+IOSurface")
            .finish()
    }
}

impl MacosGlProcessor {
    pub fn new() -> Result<Self> {
        let _span = tracing::info_span!(
            "image.gl.platform_init",
            platform = "macos",
            backend = "iosurface",
        )
        .entered();

        // 1. Load ANGLE libEGL and bring up an EGL instance.
        let egl_lib = MacosPlatform::load_egl_lib()
            .map_err(|e| Error::Io(std::io::Error::other(format!("ANGLE libEGL: {e}"))))?;
        let egl: Egl = unsafe {
            khronos_egl::Instance::<
                khronos_egl::Dynamic<&'static libloading::Library, khronos_egl::EGL1_4>,
            >::load_required_from(egl_lib)
        }
        .map_err(|e| Error::Io(std::io::Error::other(format!("EGL load: {e:?}"))))?;

        // 2. Get the Metal-backed display from MacosPlatform and initialize.
        let (display, _platform_display) = MacosPlatform::create_display(&egl)?;
        let (maj, min) = egl
            .initialize(display)
            .map_err(|e| Error::Io(std::io::Error::other(format!("eglInitialize: {e:?}"))))?;
        debug!("MacosGlProcessor: EGL {maj}.{min} initialised via ANGLE");

        egl.bind_api(egl::OPENGL_ES_API)
            .map_err(|e| Error::Io(std::io::Error::other(format!("eglBindAPI: {e:?}"))))?;

        // 3. Choose an EGL config that supports GLES 3 + PBUFFER +
        //    EGL_BIND_TO_TEXTURE_TARGET_ANGLE = EGL_TEXTURE_2D.
        let cfg_attribs = [
            EGL_RENDERABLE_TYPE,
            EGL_OPENGL_ES3_BIT,
            EGL_SURFACE_TYPE,
            EGL_PBUFFER_BIT,
            EGL_RED_SIZE,
            8,
            EGL_GREEN_SIZE,
            8,
            EGL_BLUE_SIZE,
            8,
            EGL_ALPHA_SIZE,
            8,
            iosurface_import::EGL_BIND_TO_TEXTURE_TARGET_ANGLE,
            0x305F, // EGL_TEXTURE_2D
            egl::NONE,
        ];
        let config = egl
            .choose_first_config(display, &cfg_attribs)
            .map_err(|e| Error::Io(std::io::Error::other(format!("eglChooseConfig: {e:?}"))))?
            .ok_or_else(|| {
                Error::NotSupported("no EGL config with GLES3+PBUFFER+TEXTURE_2D bind".into())
            })?;

        // 4. Create the GLES3 context.
        let ctx_attribs = [EGL_CONTEXT_CLIENT_VERSION, 3, egl::NONE];
        let context = egl
            .create_context(display, config, None, &ctx_attribs)
            .map_err(|e| Error::Io(std::io::Error::other(format!("eglCreateContext: {e:?}"))))?;

        // 5. 16×16 dummy pbuffer to make the context current for shader
        //    compilation and resource creation.
        let dummy_attribs = [egl::WIDTH, 16, egl::HEIGHT, 16, egl::NONE];
        let dummy_pbuffer = egl
            .create_pbuffer_surface(display, config, &dummy_attribs)
            .map_err(|e| {
                Error::Io(std::io::Error::other(format!(
                    "eglCreatePbufferSurface(dummy): {e:?}"
                )))
            })?;
        egl.make_current(display, Some(dummy_pbuffer), Some(dummy_pbuffer), Some(context))
            .map_err(|e| Error::Io(std::io::Error::other(format!("eglMakeCurrent: {e:?}"))))?;

        // 6. Load GL function pointers from ANGLE's libGLESv2 via EGL.
        load_gl_once(&egl);

        // 7. Compile the YUYV → RGBA program.
        let program_yuyv_to_rgba = unsafe { compile_program(VERTEX_SHADER, YUYV_TO_RGBA_FRAGMENT)? };
        let (uniform_src, uniform_src_size) = unsafe {
            let loc_src = gls::gl::GetUniformLocation(
                program_yuyv_to_rgba,
                c"src".as_ptr() as *const _,
            );
            let loc_size = gls::gl::GetUniformLocation(
                program_yuyv_to_rgba,
                c"src_size".as_ptr() as *const _,
            );
            (loc_src, loc_size)
        };

        // 8. Fullscreen-quad VBO + VAO.
        #[rustfmt::skip]
        let quad: [f32; 16] = [
            -1.0,-1.0,  0.0, 0.0,
             1.0,-1.0,  1.0, 0.0,
            -1.0, 1.0,  0.0, 1.0,
             1.0, 1.0,  1.0, 1.0,
        ];
        let (vao, vbo) = unsafe {
            let mut vbo = 0u32;
            let mut vao = 0u32;
            gls::gl::GenBuffers(1, &mut vbo);
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, vbo);
            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                (quad.len() * std::mem::size_of::<f32>()) as isize,
                quad.as_ptr() as *const _,
                gls::gl::STATIC_DRAW,
            );
            gls::gl::GenVertexArrays(1, &mut vao);
            gls::gl::BindVertexArray(vao);
            gls::gl::VertexAttribPointer(0, 2, gls::gl::FLOAT, 0, 16, std::ptr::null());
            gls::gl::EnableVertexAttribArray(0);
            gls::gl::VertexAttribPointer(1, 2, gls::gl::FLOAT, 0, 16, 8 as *const _);
            gls::gl::EnableVertexAttribArray(1);
            (vao, vbo)
        };

        // 9. FBO + two transient texture handles.
        let (fbo, src_tex, dst_tex) = unsafe {
            let mut fbo = 0u32;
            let mut src_tex = 0u32;
            let mut dst_tex = 0u32;
            gls::gl::GenFramebuffers(1, &mut fbo);
            gls::gl::GenTextures(1, &mut src_tex);
            gls::gl::GenTextures(1, &mut dst_tex);
            for tex in [src_tex, dst_tex] {
                gls::gl::BindTexture(gls::gl::TEXTURE_2D, tex);
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_MIN_FILTER,
                    gls::gl::NEAREST as i32,
                );
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_MAG_FILTER,
                    gls::gl::NEAREST as i32,
                );
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_WRAP_S,
                    gls::gl::CLAMP_TO_EDGE as i32,
                );
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_WRAP_T,
                    gls::gl::CLAMP_TO_EDGE as i32,
                );
            }
            (fbo, src_tex, dst_tex)
        };

        // 10. Release the context so subsequent convert() calls can
        //     reclaim it under the mutex.
        egl.make_current(display, None, None, None).ok();

        Ok(Self {
            egl,
            display,
            config,
            context,
            dummy_pbuffer,
            program_yuyv_to_rgba,
            uniform_src,
            uniform_src_size,
            vao,
            vbo,
            fbo,
            src_tex,
            dst_tex,
            lock: Mutex::new(()),
        })
    }

    /// Whether the requested conversion is supported by the GL backend.
    /// Used by `ImageProcessor::convert` to decide whether to dispatch
    /// here or fall back to CPU.
    pub fn supports(src_fmt: PixelFormat, dst_fmt: PixelFormat) -> bool {
        matches!(
            (src_fmt, dst_fmt),
            (PixelFormat::Yuyv, PixelFormat::Rgba | PixelFormat::Bgra)
        )
    }

    /// The actual conversion path. Caller guarantees `supports(src_fmt, dst_fmt)`.
    fn convert_yuyv_to_rgba(&self, src: &TensorDyn, dst: &mut TensorDyn) -> Result<()> {
        let _span = tracing::trace_span!(
            "image.gl.convert",
            backend = "iosurface",
            src_fmt = ?src.format(),
            dst_fmt = ?dst.format(),
        )
        .entered();

        let src_w = src.width().ok_or_else(|| Error::InvalidShape("src width".into()))?;
        let src_h = src
            .height()
            .ok_or_else(|| Error::InvalidShape("src height".into()))?;
        let dst_w = dst.width().ok_or_else(|| Error::InvalidShape("dst width".into()))?;
        let dst_h = dst
            .height()
            .ok_or_else(|| Error::InvalidShape("dst height".into()))?;

        // Validation: same-size only in this first cut. Resize support
        // is straightforward (just change the viewport + texture sample
        // ratio) but not in scope for the initial integration.
        if src_w != dst_w || src_h != dst_h {
            return Err(Error::NotImplemented(format!(
                "MacosGlProcessor: resize not yet supported (src {src_w}×{src_h} → dst {dst_w}×{dst_h}); CPU fallback handles this"
            )));
        }

        let src_u8 = src.as_u8().ok_or_else(|| {
            Error::NotSupported("GL backend requires u8 source tensor".into())
        })?;
        let dst_u8 = dst.as_u8_mut().ok_or_else(|| {
            Error::NotSupported("GL backend requires u8 destination tensor".into())
        })?;

        // Both tensors MUST be IOSurface-backed for the zero-copy path.
        // If not, fall back to CPU (the caller's dispatch chain handles
        // this — we just return NotSupported here).
        let src_iosurface = src_u8.iosurface_ref().ok_or_else(|| {
            Error::NotSupported(
                "GL convert: source tensor is not IOSurface-backed".into(),
            )
        })?;
        let dst_iosurface = dst_u8.iosurface_ref().ok_or_else(|| {
            Error::NotSupported(
                "GL convert: destination tensor is not IOSurface-backed".into(),
            )
        })?;

        let _guard = self.lock.lock().unwrap();

        // SAFETY: serialized by self.lock; EGL/GL calls require a
        // current context; tensor pointers are alive for the call's
        // duration.
        unsafe {
            self.egl
                .make_current(
                    self.display,
                    Some(self.dummy_pbuffer),
                    Some(self.dummy_pbuffer),
                    Some(self.context),
                )
                .map_err(|e| {
                    Error::Io(std::io::Error::other(format!("eglMakeCurrent: {e:?}")))
                })?;

            // Bind source IOSurface to a pbuffer + glBindTexImage.
            let src_pbuf = iosurface_import::create_iosurface_pbuffer(
                &self.egl,
                self.display,
                self.config,
                src_iosurface,
                src.format().unwrap_or(PixelFormat::Rgba),
                src_w,
                src_h,
            )?;
            // Bind destination IOSurface similarly.
            let dst_pbuf = iosurface_import::create_iosurface_pbuffer(
                &self.egl,
                self.display,
                self.config,
                dst_iosurface,
                dst.format().unwrap_or(PixelFormat::Rgba),
                dst_w,
                dst_h,
            )?;

            // Source texture binding.
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.src_tex);
            self.egl
                .bind_tex_image(self.display, src_pbuf, EGL_BACK_BUFFER)
                .map_err(|e| {
                    Error::Io(std::io::Error::other(format!(
                        "eglBindTexImage(src): {e:?}"
                    )))
                })?;

            // Destination texture binding + attach to FBO.
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.dst_tex);
            self.egl
                .bind_tex_image(self.display, dst_pbuf, EGL_BACK_BUFFER)
                .map_err(|e| {
                    Error::Io(std::io::Error::other(format!(
                        "eglBindTexImage(dst): {e:?}"
                    )))
                })?;
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, self.fbo);
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                self.dst_tex,
                0,
            );
            let fbo_status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
            if fbo_status != gls::gl::FRAMEBUFFER_COMPLETE {
                return Err(Error::Io(std::io::Error::other(format!(
                    "FBO incomplete: 0x{fbo_status:x}"
                ))));
            }

            // Render.
            gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
            gls::gl::UseProgram(self.program_yuyv_to_rgba);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.src_tex);
            gls::gl::Uniform1i(self.uniform_src, 0);
            gls::gl::Uniform2f(self.uniform_src_size, src_w as f32, src_h as f32);
            gls::gl::BindVertexArray(self.vao);
            gls::gl::DrawArrays(gls::gl::TRIANGLE_STRIP, 0, 4);
            gls::gl::Finish();

            // Cleanup transient pbuffers.
            let _ = self.egl.destroy_surface(self.display, src_pbuf);
            let _ = self.egl.destroy_surface(self.display, dst_pbuf);

            self.egl.make_current(self.display, None, None, None).ok();
        }
        Ok(())
    }
}

impl Drop for MacosGlProcessor {
    fn drop(&mut self) {
        let _guard = self.lock.lock().ok();
        unsafe {
            // Best-effort cleanup; failures here are logged but not
            // propagated since Drop must not panic.
            let _ = self.egl.make_current(
                self.display,
                Some(self.dummy_pbuffer),
                Some(self.dummy_pbuffer),
                Some(self.context),
            );
            gls::gl::DeleteFramebuffers(1, &self.fbo);
            gls::gl::DeleteTextures(1, &self.src_tex);
            gls::gl::DeleteTextures(1, &self.dst_tex);
            gls::gl::DeleteBuffers(1, &self.vbo);
            gls::gl::DeleteVertexArrays(1, &self.vao);
            gls::gl::DeleteProgram(self.program_yuyv_to_rgba);
            let _ = self.egl.make_current(self.display, None, None, None);
            let _ = self.egl.destroy_surface(self.display, self.dummy_pbuffer);
            let _ = self.egl.destroy_context(self.display, self.context);
            // Display is process-wide-shared by ANGLE convention; not
            // terminated here to avoid disturbing other consumers.
        }
    }
}

impl ImageProcessorTrait for MacosGlProcessor {
    fn convert(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        if !matches!(rotation, Rotation::None) || !matches!(flip, Flip::None) {
            return Err(Error::NotImplemented(
                "MacosGlProcessor: rotation/flip not yet supported; CPU fallback handles this"
                    .into(),
            ));
        }
        if crop.src_rect.is_some() || crop.dst_rect.is_some() {
            return Err(Error::NotImplemented(
                "MacosGlProcessor: crop not yet supported; CPU fallback handles this".into(),
            ));
        }
        let (src_fmt, dst_fmt) = match (src.format(), dst.format()) {
            (Some(s), Some(d)) => (s, d),
            _ => {
                return Err(Error::NotSupported(
                    "MacosGlProcessor: untyped tensors (None format) not supported".into(),
                ));
            }
        };
        if !Self::supports(src_fmt, dst_fmt) {
            return Err(Error::NotSupported(format!(
                "MacosGlProcessor: {src_fmt:?} → {dst_fmt:?} not in the initial GL coverage set"
            )));
        }
        self.convert_yuyv_to_rgba(src, dst)
    }

    fn draw_decoded_masks(
        &mut self,
        _dst: &mut TensorDyn,
        _detect: &[DetectBox],
        _segmentation: &[Segmentation],
        _overlay: MaskOverlay<'_>,
    ) -> Result<()> {
        Err(Error::NotImplemented(
            "MacosGlProcessor: draw_decoded_masks not yet ported (use CPU backend)".into(),
        ))
    }

    fn draw_proto_masks(
        &mut self,
        _dst: &mut TensorDyn,
        _detect: &[DetectBox],
        _proto_data: &ProtoData,
        _overlay: MaskOverlay<'_>,
    ) -> Result<()> {
        Err(Error::NotImplemented(
            "MacosGlProcessor: draw_proto_masks not yet ported (use CPU backend)".into(),
        ))
    }

    fn set_class_colors(&mut self, _colors: &[[u8; 4]]) -> Result<()> {
        // Class-color lookup table is only used by mask rendering, which
        // currently falls back to CPU on macOS. Accepting the call as a
        // no-op keeps the API surface symmetric with Linux.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shader helpers
// ---------------------------------------------------------------------------

unsafe fn compile_program(vertex_src: &str, fragment_src: &str) -> Result<u32> {
    let vs = compile_shader(gls::gl::VERTEX_SHADER, vertex_src)?;
    let fs = compile_shader(gls::gl::FRAGMENT_SHADER, fragment_src)?;
    let program = gls::gl::CreateProgram();
    gls::gl::AttachShader(program, vs);
    gls::gl::AttachShader(program, fs);
    gls::gl::LinkProgram(program);
    let mut ok = 0i32;
    gls::gl::GetProgramiv(program, gls::gl::LINK_STATUS, &mut ok);
    if ok == 0 {
        let mut log = [0u8; 4096];
        let mut len = 0i32;
        gls::gl::GetProgramInfoLog(program, log.len() as i32, &mut len, log.as_mut_ptr() as *mut _);
        return Err(Error::Internal(format!(
            "program link failed: {}",
            String::from_utf8_lossy(&log[..len.max(0) as usize])
        )));
    }
    gls::gl::DeleteShader(vs);
    gls::gl::DeleteShader(fs);
    Ok(program)
}

unsafe fn compile_shader(kind: u32, src: &str) -> Result<u32> {
    let shader = gls::gl::CreateShader(kind);
    let c = CString::new(src).map_err(|e| Error::Internal(format!("shader CString: {e}")))?;
    let ptr = c.as_ptr();
    let len = src.len() as i32;
    gls::gl::ShaderSource(shader, 1, &ptr, &len);
    gls::gl::CompileShader(shader);
    let mut ok = 0i32;
    gls::gl::GetShaderiv(shader, gls::gl::COMPILE_STATUS, &mut ok);
    if ok == 0 {
        let mut log = [0u8; 4096];
        let mut len = 0i32;
        gls::gl::GetShaderInfoLog(shader, log.len() as i32, &mut len, log.as_mut_ptr() as *mut _);
        return Err(Error::Internal(format!(
            "shader compile failed (kind=0x{kind:x}): {}",
            String::from_utf8_lossy(&log[..len.max(0) as usize])
        )));
    }
    Ok(shader)
}

