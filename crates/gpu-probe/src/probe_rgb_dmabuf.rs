// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Packed RGB DMA-buf import probe.
//!
//! A packed 24-bit `Rgb` source has no native GPU texel format, so the
//! zero-copy GL path refuses it and the convert runs on the CPU. This probe
//! measures, per GPU, every way the HAL could put such a buffer in front of a
//! shader, against the exact allocation [`Tensor::image`] produces:
//!
//! * **native-2d / native-ext** — import as `DRM_FORMAT_BGR888` (`BG24`,
//!   memory order R,G,B) and sample through `sampler2D` or
//!   `samplerExternalOES`. Filtered by the texture unit, one fetch per pixel,
//!   if the driver accepts the fourcc at all.
//! * **r8** — import the whole buffer as one R8 texture `stride` texels wide,
//!   as Path B imports NV16/NV24, and assemble each pixel from three
//!   `texelFetch`es (twelve for a bilinear sample).
//! * **rgba8** — import as `ABGR8888` (`AB24`) at `stride / 4` texels and
//!   pick each pixel's three bytes out of one or two RGBA texels (eight
//!   fetches for a bilinear sample).
//! * **upload** — not zero-copy: `glTexSubImage2D` of `GL_RGB` from the mapped
//!   buffer, the route a non-DMA `Rgb` source already takes.
//!
//! Each path is first checked for exactness with a 1:1 draw whose every byte
//! is compared against the source, at even and odd logical sizes, then timed
//! on warm bilinear resizes. An RGBA source through the existing external
//! import is timed alongside as the reference cost of a zero-copy source.

use crate::bench_render::{compile_program, create_quad_vao, VERTEX_SRC};
use crate::egl_context::{EglPlane, GpuContext};
use crate::probe_nv_dmabuf::{gl_error, gl_has_extension};
use edgefirst_gl::gl;
use edgefirst_tensor::{PixelFormat, Tensor, TensorMapTrait, TensorMemory, TensorTrait};
use gbm::drm::buffer::DrmFourcc;
use std::ffi::CString;
use std::time::Instant;

const WARMUP: usize = 5;
const ITERS: usize = 30;

/// Byte written into the row padding, so a path that reads past the logical
/// row shows up as a mismatch instead of a plausible pixel.
const PAD_BYTE: u8 = 0xEE;

/// Bilinear (or, at scale 1, exact) sample from an R8 import `stride` texels
/// wide: pixel `x` is bytes `3x .. 3x+2` of its row.
const R8_FRAGMENT_SRC: &str = "\
#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D tex;
uniform ivec2 src_size;
uniform vec2 scale;
out vec4 color;
vec3 px(ivec2 p) {
    int b = p.x * 3;
    return vec3(texelFetch(tex, ivec2(b, p.y), 0).r,
                texelFetch(tex, ivec2(b + 1, p.y), 0).r,
                texelFetch(tex, ivec2(b + 2, p.y), 0).r);
}
void main() {
    vec2 s = clamp(gl_FragCoord.xy * scale - 0.5, vec2(0.0), vec2(src_size - 1));
    ivec2 i0 = ivec2(floor(s));
    ivec2 i1 = min(i0 + 1, src_size - 1);
    vec2 f = s - vec2(i0);
    vec3 top = mix(px(i0), px(ivec2(i1.x, i0.y)), f.x);
    vec3 bot = mix(px(ivec2(i0.x, i1.y)), px(i1), f.x);
    color = vec4(mix(top, bot, f.y), 1.0);
}
";

/// [`R8_FRAGMENT_SRC`] with each `texelFetch` replaced by a NEAREST
/// `texture()` at the texel centre, to tell a slow `texelFetch` apart from a
/// slow fetch count.
const R8_TEX_FRAGMENT_SRC: &str = "\
#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D tex;
uniform ivec2 src_size;
uniform vec2 scale;
out vec4 color;
vec3 px(ivec2 p) {
    vec2 inv = 1.0 / vec2(textureSize(tex, 0));
    float b = float(p.x * 3) + 0.5;
    float y = (float(p.y) + 0.5) * inv.y;
    return vec3(texture(tex, vec2(b * inv.x, y)).r,
                texture(tex, vec2((b + 1.0) * inv.x, y)).r,
                texture(tex, vec2((b + 2.0) * inv.x, y)).r);
}
void main() {
    vec2 s = clamp(gl_FragCoord.xy * scale - 0.5, vec2(0.0), vec2(src_size - 1));
    ivec2 i0 = ivec2(floor(s));
    ivec2 i1 = min(i0 + 1, src_size - 1);
    vec2 f = s - vec2(i0);
    vec3 top = mix(px(i0), px(ivec2(i1.x, i0.y)), f.x);
    vec3 bot = mix(px(ivec2(i0.x, i1.y)), px(i1), f.x);
    color = vec4(mix(top, bot, f.y), 1.0);
}
";

/// Bilinear from the R8 import with the vertical half done by the texture
/// unit: LINEAR filtering, sampled at each byte column's exact centre so only
/// rows blend, leaving six fetches and one horizontal `mix` in the shader.
const R8_VLERP_FRAGMENT_SRC: &str = "\
#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D tex;
uniform ivec2 src_size;
uniform vec2 scale;
uniform float nudge;
out vec4 color;
vec3 col(int x, float v, float inv_w) {
    float b = float(x * 3) + 0.5 + nudge;
    return vec3(texture(tex, vec2(b * inv_w, v)).r,
                texture(tex, vec2((b + 1.0) * inv_w, v)).r,
                texture(tex, vec2((b + 2.0) * inv_w, v)).r);
}
void main() {
    vec2 inv = 1.0 / vec2(textureSize(tex, 0));
    vec2 s = clamp(gl_FragCoord.xy * scale - 0.5, vec2(0.0), vec2(src_size - 1));
    int x0 = int(floor(s.x));
    int x1 = min(x0 + 1, src_size.x - 1);
    float fx = s.x - float(x0);
    float v = (s.y + 0.5) * inv.y;
    color = vec4(mix(col(x0, v, inv.x), col(x1, v, inv.x), fx), 1.0);
}
";

/// Nearest sample from the R8 import: three `texelFetch`es per pixel, the
/// floor of what any R8 assembly costs.
const R8_NEAREST_FRAGMENT_SRC: &str = "\
#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D tex;
uniform ivec2 src_size;
uniform vec2 scale;
out vec4 color;
void main() {
    ivec2 p = min(ivec2(gl_FragCoord.xy * scale), src_size - 1);
    int b = p.x * 3;
    color = vec4(texelFetch(tex, ivec2(b, p.y), 0).r,
                 texelFetch(tex, ivec2(b + 1, p.y), 0).r,
                 texelFetch(tex, ivec2(b + 2, p.y), 0).r, 1.0);
}
";

/// The same sample from an RGBA8 import `stride / 4` texels wide: pixel `x`
/// starts at byte `3x`, i.e. component `3x & 3` of texel `3x >> 2`, and
/// spills into the next texel when that component is 2 or 3.
const RGBA8_FRAGMENT_SRC: &str = "\
#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D tex;
uniform ivec2 src_size;
uniform vec2 scale;
out vec4 color;
vec3 px(ivec2 p) {
    int b = p.x * 3;
    int t = b >> 2;
    int o = b & 3;
    int last = textureSize(tex, 0).x - 1;
    vec4 a = texelFetch(tex, ivec2(t, p.y), 0);
    vec4 c = texelFetch(tex, ivec2(min(t + 1, last), p.y), 0);
    if (o == 0) return a.rgb;
    if (o == 1) return a.gba;
    if (o == 2) return vec3(a.ba, c.r);
    return vec3(a.a, c.rg);
}
void main() {
    vec2 s = clamp(gl_FragCoord.xy * scale - 0.5, vec2(0.0), vec2(src_size - 1));
    ivec2 i0 = ivec2(floor(s));
    ivec2 i1 = min(i0 + 1, src_size - 1);
    vec2 f = s - vec2(i0);
    vec3 top = mix(px(i0), px(ivec2(i1.x, i0.y)), f.x);
    vec3 bot = mix(px(ivec2(i0.x, i1.y)), px(i1), f.x);
    color = vec4(mix(top, bot, f.y), 1.0);
}
";

/// Filtered sample through the texture unit, for every path whose texture
/// holds one RGB(A) texel per pixel.
const FILTERED_2D_FRAGMENT_SRC: &str = "\
#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D tex;
uniform ivec2 src_size;
uniform vec2 scale;
out vec4 color;
void main() {
    vec2 uv = gl_FragCoord.xy * scale / vec2(src_size);
    color = vec4(texture(tex, uv).rgb, 1.0);
}
";

const FILTERED_EXT_FRAGMENT_SRC: &str = "\
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
precision highp int;
uniform highp samplerExternalOES tex;
uniform ivec2 src_size;
uniform vec2 scale;
out vec4 color;
void main() {
    vec2 uv = gl_FragCoord.xy * scale / vec2(src_size);
    color = vec4(texture(tex, uv).rgb, 1.0);
}
";

#[derive(Clone, Copy, PartialEq, Debug)]
enum Path {
    Native2d,
    NativeExt,
    R8,
    R8Tex,
    R8VLerp,
    R8VLerpNudge,
    R8Nearest,
    Rgba8,
    Upload,
    /// An `Rgba` source through the external import: the reference cost.
    RgbaRefExt,
}

impl Path {
    fn name(self) -> &'static str {
        match self {
            Path::Native2d => "native-2d",
            Path::NativeExt => "native-ext",
            Path::R8 => "r8",
            Path::R8Tex => "r8-texture",
            Path::R8VLerp => "r8-vlerp",
            Path::R8VLerpNudge => "r8-vlerp-nudge",
            Path::R8Nearest => "r8-nearest",
            Path::Rgba8 => "rgba8",
            Path::Upload => "upload",
            Path::RgbaRefExt => "rgba-src-ext (ref)",
        }
    }
}

struct Program {
    id: u32,
    tex: i32,
    src_size: i32,
    scale: i32,
    nudge: i32,
}

struct Programs {
    r8: Program,
    r8_tex: Program,
    r8_vlerp: Program,
    r8_nearest: Program,
    rgba8: Program,
    filtered_2d: Program,
    filtered_ext: Option<Program>,
    vao: u32,
}

fn program(frag: &str) -> Result<Program, String> {
    let vs = CString::new(VERTEX_SRC).unwrap();
    let fs = CString::new(frag).unwrap();
    let id = compile_program(&vs, &fs)?;
    let loc = |n: &str| unsafe {
        let c = CString::new(n).unwrap();
        gl::GetUniformLocation(id, c.as_ptr())
    };
    Ok(Program {
        id,
        tex: loc("tex"),
        src_size: loc("src_size"),
        scale: loc("scale"),
        nudge: loc("nudge"),
    })
}

impl Programs {
    fn new() -> Result<Self, String> {
        let filtered_ext = if gl_has_extension("GL_OES_EGL_image_external_essl3") {
            program(FILTERED_EXT_FRAGMENT_SRC).ok()
        } else {
            None
        };
        let (vao, _, _) = create_quad_vao();
        Ok(Programs {
            r8: program(R8_FRAGMENT_SRC).map_err(|e| format!("r8 program: {e}"))?,
            r8_tex: program(R8_TEX_FRAGMENT_SRC).map_err(|e| format!("r8 texture program: {e}"))?,
            r8_vlerp: program(R8_VLERP_FRAGMENT_SRC)
                .map_err(|e| format!("r8 vlerp program: {e}"))?,
            r8_nearest: program(R8_NEAREST_FRAGMENT_SRC)
                .map_err(|e| format!("r8 nearest program: {e}"))?,
            rgba8: program(RGBA8_FRAGMENT_SRC).map_err(|e| format!("rgba8 program: {e}"))?,
            filtered_2d: program(FILTERED_2D_FRAGMENT_SRC)
                .map_err(|e| format!("filtered 2d program: {e}"))?,
            filtered_ext,
            vao,
        })
    }

    fn for_path(&self, p: Path) -> Option<&Program> {
        match p {
            Path::R8 => Some(&self.r8),
            Path::R8Tex => Some(&self.r8_tex),
            Path::R8VLerp | Path::R8VLerpNudge => Some(&self.r8_vlerp),
            Path::R8Nearest => Some(&self.r8_nearest),
            Path::Rgba8 => Some(&self.rgba8),
            Path::Native2d | Path::Upload => Some(&self.filtered_2d),
            Path::NativeExt | Path::RgbaRefExt => self.filtered_ext.as_ref(),
        }
    }
}

fn fourcc_str(f: u32) -> String {
    f.to_le_bytes().iter().map(|&b| b as char).collect()
}

fn pattern(x: usize, y: usize, c: usize) -> u8 {
    ((x * 7 + y * 13 + c * 101) ^ (x >> 3) ^ (y << 2)) as u8
}

/// A source allocated exactly as the HAL allocates it, filled with
/// [`pattern`] and [`PAD_BYTE`] in the row padding.
fn alloc_src(fmt: PixelFormat, w: usize, h: usize) -> Result<(Tensor<u8>, usize), String> {
    let src = Tensor::<u8>::image(
        w,
        h,
        fmt,
        Some(TensorMemory::DmaBuf),
        edgefirst_tensor::CpuAccess::ReadWrite,
    )
    .map_err(|e| format!("alloc {fmt:?} {w}x{h}: {e}"))?;
    let stride = src
        .effective_row_stride()
        .ok_or_else(|| "no row stride".to_string())?;
    let ch = fmt.channels();
    {
        let mut map = src.map().map_err(|e| format!("map: {e}"))?;
        let data = map.as_mut_slice();
        for y in 0..h {
            let row = &mut data[y * stride..(y + 1) * stride];
            row.fill(PAD_BYTE);
            for x in 0..w {
                for c in 0..ch {
                    row[x * ch + c] = if c == 3 { 255 } else { pattern(x, y, c) };
                }
            }
        }
    }
    Ok((src, stride))
}

/// A source texture bound and ready to sample, plus what to release.
struct SrcTexture {
    tex: u32,
    target: u32,
    image: Option<edgefirst_egl::Image>,
}

impl SrcTexture {
    fn release(self, ctx: &GpuContext) {
        unsafe { gl::DeleteTextures(1, &self.tex) };
        if let Some(img) = self.image {
            let _ = ctx.destroy_egl_image(img);
        }
    }
}

fn import(
    ctx: &GpuContext,
    path: Path,
    src: &Tensor<u8>,
    w: usize,
    h: usize,
    stride: usize,
) -> Result<SrcTexture, String> {
    while unsafe { gl::GetError() } != gl::NO_ERROR {}
    let fd = src.clone_fd().map_err(|e| format!("clone_fd: {e}"))?;
    let fd = std::os::fd::AsRawFd::as_raw_fd(&fd);
    let offset = src.plane_offset().unwrap_or(0) as i32;
    let plane = EglPlane {
        fd,
        offset,
        pitch: stride as i32,
    };
    let (tw, fourcc, target, filter) = match path {
        Path::Native2d => (w, DrmFourcc::Bgr888, gl::TEXTURE_2D, gl::LINEAR),
        Path::NativeExt => (w, DrmFourcc::Bgr888, gl::TEXTURE_EXTERNAL_OES, gl::LINEAR),
        Path::R8 | Path::R8Tex | Path::R8Nearest => {
            (stride, DrmFourcc::R8, gl::TEXTURE_2D, gl::NEAREST)
        }
        Path::R8VLerp | Path::R8VLerpNudge => (stride, DrmFourcc::R8, gl::TEXTURE_2D, gl::LINEAR),
        Path::Rgba8 => {
            if !stride.is_multiple_of(4) {
                return Err(format!("stride {stride} not a multiple of 4"));
            }
            (stride / 4, DrmFourcc::Abgr8888, gl::TEXTURE_2D, gl::NEAREST)
        }
        Path::RgbaRefExt => (w, DrmFourcc::Abgr8888, gl::TEXTURE_EXTERNAL_OES, gl::LINEAR),
        Path::Upload => return upload(src, w, h, stride),
    };
    let image = ctx
        .create_egl_image_planar(tw as i32, h as i32, fourcc as u32, plane, None, false)
        .map_err(|e| {
            format!(
                "eglCreateImage({} {tw}x{h}@{stride}): {e}",
                fourcc_str(fourcc as u32)
            )
        })?;
    let mut tex = 0u32;
    unsafe {
        gl::GenTextures(1, &mut tex);
        gl::BindTexture(target, tex);
        gl::EGLImageTargetTexture2DOES(target, image.as_ptr());
        set_sampling(target, filter);
    }
    if let Err(e) = gl_error("EGLImageTargetTexture2DOES") {
        unsafe { gl::DeleteTextures(1, &tex) };
        let _ = ctx.destroy_egl_image(image);
        return Err(e);
    }
    Ok(SrcTexture {
        tex,
        target,
        image: Some(image),
    })
}

unsafe fn set_sampling(target: u32, filter: u32) {
    gl::TexParameteri(target, gl::TEXTURE_MIN_FILTER, filter as i32);
    gl::TexParameteri(target, gl::TEXTURE_MAG_FILTER, filter as i32);
    gl::TexParameteri(target, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
    gl::TexParameteri(target, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
}

/// Allocate an `RGB8` texture and fill it from the mapped buffer.
fn upload(src: &Tensor<u8>, w: usize, h: usize, stride: usize) -> Result<SrcTexture, String> {
    let mut tex = 0u32;
    unsafe {
        gl::GenTextures(1, &mut tex);
        gl::BindTexture(gl::TEXTURE_2D, tex);
        gl::TexStorage2D(gl::TEXTURE_2D, 1, gl::RGB8, w as i32, h as i32);
        set_sampling(gl::TEXTURE_2D, gl::LINEAR);
    }
    let st = SrcTexture {
        tex,
        target: gl::TEXTURE_2D,
        image: None,
    };
    if let Err(e) = upload_into(src, w, h, stride) {
        unsafe { gl::DeleteTextures(1, &tex) };
        return Err(e);
    }
    Ok(st)
}

/// Re-upload the source into the currently bound `TEXTURE_2D`; one call per
/// frame on the upload path.
fn upload_into(src: &Tensor<u8>, w: usize, h: usize, stride: usize) -> Result<(), String> {
    let map = src.map().map_err(|e| format!("map: {e}"))?;
    let data = map.as_slice();
    unsafe {
        gl::PixelStorei(gl::UNPACK_ALIGNMENT, 1);
        if stride.is_multiple_of(3) {
            gl::PixelStorei(gl::UNPACK_ROW_LENGTH, (stride / 3) as i32);
            gl::TexSubImage2D(
                gl::TEXTURE_2D,
                0,
                0,
                0,
                w as i32,
                h as i32,
                gl::RGB,
                gl::UNSIGNED_BYTE,
                data.as_ptr().cast(),
            );
        } else {
            // A row length is counted in pixels, so a stride that is not a
            // whole number of RGB pixels has to go up a row at a time.
            gl::PixelStorei(gl::UNPACK_ROW_LENGTH, 0);
            for y in 0..h {
                gl::TexSubImage2D(
                    gl::TEXTURE_2D,
                    0,
                    0,
                    y as i32,
                    w as i32,
                    1,
                    gl::RGB,
                    gl::UNSIGNED_BYTE,
                    data[y * stride..].as_ptr().cast(),
                );
            }
        }
        gl::PixelStorei(gl::UNPACK_ROW_LENGTH, 0);
    }
    gl_error("glTexSubImage2D(GL_RGB)")
}

/// An `RGBA8` render target of the given size, bound as the draw framebuffer.
struct Target {
    fbo: u32,
    tex: u32,
    w: usize,
    h: usize,
}

impl Target {
    fn new(w: usize, h: usize) -> Result<Self, String> {
        let (mut fbo, mut tex) = (0u32, 0u32);
        unsafe {
            gl::GenTextures(1, &mut tex);
            gl::BindTexture(gl::TEXTURE_2D, tex);
            gl::TexStorage2D(gl::TEXTURE_2D, 1, gl::RGBA8, w as i32, h as i32);
            gl::GenFramebuffers(1, &mut fbo);
            gl::BindFramebuffer(gl::FRAMEBUFFER, fbo);
            gl::FramebufferTexture2D(
                gl::FRAMEBUFFER,
                gl::COLOR_ATTACHMENT0,
                gl::TEXTURE_2D,
                tex,
                0,
            );
            let status = gl::CheckFramebufferStatus(gl::FRAMEBUFFER);
            if status != gl::FRAMEBUFFER_COMPLETE {
                gl::DeleteFramebuffers(1, &fbo);
                gl::DeleteTextures(1, &tex);
                return Err(format!("FBO incomplete: 0x{status:04X}"));
            }
        }
        Ok(Target { fbo, tex, w, h })
    }

    fn read(&self) -> Vec<u8> {
        let mut out = vec![0u8; self.w * self.h * 4];
        unsafe {
            gl::BindFramebuffer(gl::FRAMEBUFFER, self.fbo);
            gl::PixelStorei(gl::PACK_ALIGNMENT, 1);
            gl::ReadPixels(
                0,
                0,
                self.w as i32,
                self.h as i32,
                gl::RGBA,
                gl::UNSIGNED_BYTE,
                out.as_mut_ptr().cast(),
            );
        }
        out
    }
}

impl Drop for Target {
    fn drop(&mut self) {
        unsafe {
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
            gl::DeleteFramebuffers(1, &self.fbo);
            gl::DeleteTextures(1, &self.tex);
        }
    }
}

fn draw(
    progs: &Programs,
    prog: &Program,
    path: Path,
    src: &SrcTexture,
    tgt: &Target,
    sw: usize,
    sh: usize,
) {
    unsafe {
        gl::BindFramebuffer(gl::FRAMEBUFFER, tgt.fbo);
        gl::Viewport(0, 0, tgt.w as i32, tgt.h as i32);
        gl::UseProgram(prog.id);
        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(src.target, src.tex);
        gl::Uniform1i(prog.tex, 0);
        gl::Uniform2i(prog.src_size, sw as i32, sh as i32);
        if prog.nudge >= 0 {
            let n = std::env::var("RGB_PROBE_NUDGE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1.0 / 1024.0);
            gl::Uniform1f(prog.nudge, if path == Path::R8VLerpNudge { n } else { 0.0 });
        }
        gl::Uniform2f(
            prog.scale,
            sw as f32 / tgt.w as f32,
            sh as f32 / tgt.h as f32,
        );
        gl::BindVertexArray(progs.vao);
        gl::DrawElements(gl::TRIANGLES, 6, gl::UNSIGNED_INT, std::ptr::null());
    }
}

/// 1:1 draw, every byte compared. Returns the mismatch summary on failure.
fn check_exact(
    ctx: &GpuContext,
    progs: &Programs,
    path: Path,
    w: usize,
    h: usize,
) -> Result<(), String> {
    let prog = progs
        .for_path(path)
        .ok_or("GL_OES_EGL_image_external_essl3 absent")?;
    let (src, stride) = alloc_src(PixelFormat::Rgb, w, h)?;
    let tex = import(ctx, path, &src, w, h, stride)?;
    // NEAREST: a LINEAR sample at a computed texel centre may pick up a
    // sliver of its neighbour on GPUs with coarse sub-texel precision, which
    // says nothing about whether the import's layout is right. The vlerp
    // path keeps LINEAR: its horizontal taps rely on exactly that precision.
    if !matches!(path, Path::R8VLerp | Path::R8VLerpNudge) {
        unsafe {
            gl::BindTexture(tex.target, tex.tex);
            set_sampling(tex.target, gl::NEAREST);
        }
    }
    let tgt = match Target::new(w, h) {
        Ok(t) => t,
        Err(e) => {
            tex.release(ctx);
            return Err(e);
        }
    };
    draw(progs, prog, path, &tex, &tgt, w, h);
    unsafe { gl::Finish() };
    let drawn = gl_error("draw");
    let out = tgt.read();
    tex.release(ctx);
    drawn?;
    let (mut bad, mut max_diff, mut first) = (0usize, 0u8, None);
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let got = out[(y * w + x) * 4 + c];
                let want = pattern(x, y, c);
                let d = got.abs_diff(want);
                if d > 0 {
                    bad += 1;
                    max_diff = max_diff.max(d);
                    first.get_or_insert((x, y, c, got, want));
                }
            }
        }
    }
    match first {
        None => Ok(()),
        Some((x, y, c, got, want)) => Err(format!(
            "{bad} bytes differ (max {max_diff}); first at ({x},{y}) ch{c}: got {got} want {want}"
        )),
    }
}

/// Render `sw x sh -> dw x dh` through `path` and return the RGBA readback.
fn render_scaled(
    ctx: &GpuContext,
    progs: &Programs,
    path: Path,
    (sw, sh): (usize, usize),
    (dw, dh): (usize, usize),
) -> Result<Vec<u8>, String> {
    let prog = progs
        .for_path(path)
        .ok_or("GL_OES_EGL_image_external_essl3 absent")?;
    let (src, stride) = alloc_src(PixelFormat::Rgb, sw, sh)?;
    let tex = import(ctx, path, &src, sw, sh, stride)?;
    let tgt = match Target::new(dw, dh) {
        Ok(t) => t,
        Err(e) => {
            tex.release(ctx);
            return Err(e);
        }
    };
    draw(progs, prog, path, &tex, &tgt, sw, sh);
    unsafe { gl::Finish() };
    let drawn = gl_error("scaled draw");
    let out = tgt.read();
    tex.release(ctx);
    drawn?;
    Ok(out)
}

/// Max byte difference, and how many bytes differ by more than 1, between
/// `path` and the shader-filtered `r8-texture` reference on a rescale.
fn compare_scaled(
    ctx: &GpuContext,
    progs: &Programs,
    path: Path,
    src: (usize, usize),
    dst: (usize, usize),
) -> Result<(u8, usize), String> {
    let want = render_scaled(ctx, progs, Path::R8Tex, src, dst)?;
    let got = render_scaled(ctx, progs, path, src, dst)?;
    let (mut max, mut over1) = (0u8, 0usize);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        if i % 4 == 3 {
            continue;
        }
        let d = g.abs_diff(*w);
        max = max.max(d);
        over1 += usize::from(d > 1);
    }
    Ok((max, over1))
}

/// Median milliseconds per frame for a warm `sw x sh -> dw x dh` draw.
fn time_path(
    ctx: &GpuContext,
    progs: &Programs,
    path: Path,
    (sw, sh): (usize, usize),
    (dw, dh): (usize, usize),
) -> Result<f64, String> {
    let prog = progs
        .for_path(path)
        .ok_or("GL_OES_EGL_image_external_essl3 absent")?;
    let fmt = if path == Path::RgbaRefExt {
        PixelFormat::Rgba
    } else {
        PixelFormat::Rgb
    };
    let (src, stride) = alloc_src(fmt, sw, sh)?;
    let tex = import(ctx, path, &src, sw, sh, stride)?;
    let tgt = match Target::new(dw, dh) {
        Ok(t) => t,
        Err(e) => {
            tex.release(ctx);
            return Err(e);
        }
    };
    let mut samples = Vec::with_capacity(ITERS);
    let mut result = Ok(());
    for i in 0..WARMUP + ITERS {
        let t0 = Instant::now();
        if path == Path::Upload {
            unsafe { gl::BindTexture(gl::TEXTURE_2D, tex.tex) };
            if let Err(e) = upload_into(&src, sw, sh, stride) {
                result = Err(e);
                break;
            }
        }
        draw(progs, prog, path, &tex, &tgt, sw, sh);
        unsafe { gl::Finish() };
        if i >= WARMUP {
            samples.push(t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    let drawn = gl_error("timed draw");
    tex.release(ctx);
    result?;
    drawn?;
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(samples[samples.len() / 2])
}

fn report_formats(ctx: &GpuContext) {
    let Some(formats) = ctx.query_dmabuf_formats() else {
        println!("  eglQueryDmaBufFormatsEXT: unavailable");
        return;
    };
    let mut names: Vec<String> = formats.iter().map(|&f| fourcc_str(f)).collect();
    names.sort();
    println!(
        "  importable fourccs ({}): {}",
        names.len(),
        names.join(" ")
    );
    for (label, f) in [
        ("BG24 (Rgb)", DrmFourcc::Bgr888),
        ("RG24", DrmFourcc::Rgb888),
        ("AB24 (Rgba)", DrmFourcc::Abgr8888),
        ("R8", DrmFourcc::R8),
        ("GR88", DrmFourcc::Gr88),
    ] {
        let listed = formats.contains(&(f as u32));
        let mods = ctx
            .query_dmabuf_modifiers(f as u32)
            .map(|m| {
                if m.is_empty() {
                    "no explicit modifiers".to_string()
                } else {
                    m.iter()
                        .map(|(m, ext)| format!("{m:#x}{}", if *ext { " ext-only" } else { "" }))
                        .collect::<Vec<_>>()
                        .join(", ")
                }
            })
            .unwrap_or_else(|| "modifier query failed".into());
        println!(
            "  {label:<12} {:<10} {mods}",
            if listed { "listed" } else { "NOT listed" }
        );
    }
}

/// `RGB_PROBE_PATHS` (comma-separated path names) and `RGB_PROBE_SIZES`
/// (comma-separated `WxH`) narrow a run to single cases, so a kernel GPU
/// fault can be pinned on the case that raised it.
fn selected<T>(items: Vec<T>, var: &str, key: impl Fn(&T) -> String) -> Vec<T> {
    match std::env::var(var) {
        Ok(v) => {
            let keep: Vec<&str> = v.split(',').collect();
            items
                .into_iter()
                .filter(|i| keep.contains(&key(i).as_str()))
                .collect()
        }
        Err(_) => items,
    }
}

pub fn run(ctx: &GpuContext) {
    println!("== Probe: packed RGB DMA-buf import ==");
    println!("  renderer: {}", ctx.gl_renderer());
    report_formats(ctx);

    let progs = match Programs::new() {
        Ok(p) => p,
        Err(e) => {
            println!("  FAIL: {e}");
            println!();
            return;
        }
    };

    let paths = vec![
        Path::Native2d,
        Path::NativeExt,
        Path::R8,
        Path::R8Tex,
        Path::R8VLerp,
        Path::R8VLerpNudge,
        Path::R8Nearest,
        Path::Rgba8,
        Path::Upload,
    ];
    let paths = selected(paths, "RGB_PROBE_PATHS", |p| p.name().to_string());
    let size_key = |&(w, h): &(usize, usize)| format!("{w}x{h}");
    println!("  -- exactness, 1:1 draw, every byte --");
    let sizes = vec![
        (320, 240),
        (321, 241),
        (640, 480),
        (789, 384),
        (1280, 720),
        (1920, 1080),
    ];
    for (w, h) in selected(sizes, "RGB_PROBE_SIZES", size_key) {
        for &p in &paths {
            match check_exact(ctx, &progs, p, w, h) {
                Ok(()) => println!("  PASS {w:>4}x{h:<4} {:<10}", p.name()),
                Err(e) => println!("  FAIL {w:>4}x{h:<4} {:<10} {e}", p.name()),
            }
        }
    }

    println!("  -- rescale vs r8-texture: max byte diff, bytes off by >1 --");
    for (src, dst) in [
        ((1920, 1080), (640, 640)),
        ((321, 241), (640, 640)),
        ((789, 384), (640, 640)),
    ] {
        for &p in paths
            .iter()
            .filter(|&&p| p != Path::R8Tex && p != Path::R8Nearest)
        {
            let label = format!("{}x{} -> {}x{}", src.0, src.1, dst.0, dst.1);
            match compare_scaled(ctx, &progs, p, src, dst) {
                Ok((max, over1)) => {
                    println!("  {label:<22} {:<12} max {max:3}  >1: {over1}", p.name())
                }
                Err(e) => println!("  {label:<22} {:<12} n/a ({e})", p.name()),
            }
        }
    }

    println!("  -- warm draw, median of {ITERS} (ms) --");
    let mut timed = paths.clone();
    if std::env::var_os("RGB_PROBE_PATHS").is_none() {
        timed.push(Path::RgbaRefExt);
    }
    let pairs = vec![
        ((1920, 1080), (640, 640)),
        ((1280, 720), (640, 640)),
        ((640, 480), (640, 640)),
        ((1920, 1080), (1920, 1080)),
    ];
    for (src, dst) in selected(pairs, "RGB_PROBE_SIZES", |(s, _)| size_key(s)) {
        for &p in &timed {
            let label = format!("{}x{} -> {}x{}", src.0, src.1, dst.0, dst.1);
            match time_path(ctx, &progs, p, src, dst) {
                Ok(ms) => println!("  {label:<22} {:<20} {ms:8.3}", p.name()),
                Err(e) => println!("  {label:<22} {:<20} n/a ({e})", p.name()),
            }
        }
    }
    println!();
}
