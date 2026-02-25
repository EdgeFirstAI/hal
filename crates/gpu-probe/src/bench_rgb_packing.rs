// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! RGB packing benchmarks — measures the cost of different strategies for
//! producing packed RGB (`RGBRGB...`) and planar RGB (`RRR...GGG...BBB...`)
//! output from an RGBA source via GL rendering.
//!
//! OpenGL ES doesn't allow rendering to RGB-only surfaces (only RGBA), so the
//! HAL needs efficient paths to produce packed/planar RGB for model input
//! tensors. This module benchmarks five strategies:
//!
//! 1. **packed_r8** — packed RGB into R8 texture, W*3 x H
//! 2. **packed_rgba8** — packed RGB into RGBA8 texture, W*3/4 x H
//! 3. **planar_r8_1pass** — planar RGB into R8 texture, W x H*3, single pass
//! 4. **planar_r8_3pass** — planar RGB into R8 texture, W x H*3, 3 draw calls
//! 5. **planar_rgba8_1pass** — planar RGB into RGBA8 texture, W/4 x H*3, single pass
//! 6. **planar_rgba8_3pass** — planar RGB into RGBA8 texture, W/4 x H*3, 3 draw calls

use crate::bench::{run_bench, BenchResult};
use crate::bench_render;
use crate::egl_context::GpuContext;
use edgefirst_tensor::{Tensor, TensorMapTrait, TensorMemory, TensorTrait};
use std::ffi::CString;
use std::os::unix::io::AsRawFd;
use std::ptr::null;

// ---------------------------------------------------------------------------
// Fragment shaders
// ---------------------------------------------------------------------------

/// Strategy 1: Packed RGB → R8 output.
/// Output is (W*3) x H, GL_R8. Each output pixel maps to one channel of
/// the source RGBA texture.
const FRAG_PACKED_R8: &str = "\
#version 300 es
precision highp float;
uniform sampler2D tex;
uniform float src_w;
uniform float src_h;
out vec4 color;
void main() {
    float ox = gl_FragCoord.x - 0.5;
    float oy = gl_FragCoord.y - 0.5;
    float src_x = floor(ox / 3.0);
    int ch = int(mod(ox, 3.0));
    vec2 uv = vec2((src_x + 0.5) / src_w, (oy + 0.5) / src_h);
    vec4 s = texture(tex, uv);
    float v = (ch == 0) ? s.r : ((ch == 1) ? s.g : s.b);
    color = vec4(v, 0.0, 0.0, 1.0);
}
";

/// Strategy 2: Packed RGB → RGBA8 output.
/// Output is (W*3/4) x H, GL_RGBA8. Each RGBA output pixel packs data from
/// the interleaved pattern [R₀G₀B₀R₁] [G₁B₁R₂G₂] [B₂R₃G₃B₃].
const FRAG_PACKED_RGBA8: &str = "\
#version 300 es
precision highp float;
uniform sampler2D tex;
uniform float src_w;
uniform float src_h;
out vec4 color;
void main() {
    float ox = gl_FragCoord.x - 0.5;
    float oy = gl_FragCoord.y - 0.5;
    int out_x = int(ox);
    int base = out_x * 4;
    vec4 result;
    for (int i = 0; i < 4; i++) {
        int idx = base + i;
        int src_x = idx / 3;
        int ch = idx - src_x * 3;
        vec2 uv = vec2((float(src_x) + 0.5) / src_w, (oy + 0.5) / src_h);
        vec4 s = texture(tex, uv);
        float v = (ch == 0) ? s.r : ((ch == 1) ? s.g : s.b);
        result[i] = v;
    }
    color = result;
}
";

/// Strategy 3a: Planar RGB → R8 single-pass.
/// Output is W x (H*3), GL_R8. Top third = R, middle = G, bottom = B.
const FRAG_PLANAR_R8_1PASS: &str = "\
#version 300 es
precision highp float;
uniform sampler2D tex;
uniform float src_w;
uniform float src_h;
out vec4 color;
void main() {
    float ox = gl_FragCoord.x - 0.5;
    float oy = gl_FragCoord.y - 0.5;
    int plane = int(floor(oy / src_h));
    float src_y = oy - float(plane) * src_h;
    vec2 uv = vec2((ox + 0.5) / src_w, (src_y + 0.5) / src_h);
    vec4 s = texture(tex, uv);
    float v = (plane == 0) ? s.r : ((plane == 1) ? s.g : s.b);
    color = vec4(v, 0.0, 0.0, 1.0);
}
";

/// Strategy 3b: Planar RGB → R8 multi-pass uses the simple passthrough
/// shader with TEXTURE_SWIZZLE_R per channel (3 draw calls).
const FRAG_PLANAR_R8_3PASS: &str = "\
#version 300 es
precision highp float;
uniform sampler2D tex;
in vec2 tc;
out vec4 color;
void main() { color = vec4(texture(tex, tc).r, 0.0, 0.0, 1.0); }
";

/// Strategy 5: Planar RGB → RGBA8 single-pass output.
/// Output is (W/4) x (H*3), GL_RGBA8. Each output pixel packs 4 adjacent
/// same-channel values.
const FRAG_PLANAR_RGBA8_1PASS: &str = "\
#version 300 es
precision highp float;
uniform sampler2D tex;
uniform float src_w;
uniform float src_h;
out vec4 color;
void main() {
    float ox = gl_FragCoord.x - 0.5;
    float oy = gl_FragCoord.y - 0.5;
    int plane = int(floor(oy / src_h));
    float src_y = oy - float(plane) * src_h;
    int out_x = int(ox);
    vec4 result;
    for (int i = 0; i < 4; i++) {
        int src_x = out_x * 4 + i;
        vec2 uv = vec2((float(src_x) + 0.5) / src_w, (src_y + 0.5) / src_h);
        vec4 s = texture(tex, uv);
        float v = (plane == 0) ? s.r : ((plane == 1) ? s.g : s.b);
        result[i] = v;
    }
    color = result;
}
";

/// Strategy 6: Planar RGB → RGBA8 multi-pass output.
/// Output is (W/4) x (H*3), GL_RGBA8. Uses 3 draw calls with viewport
/// offsets. Each pass packs 4 adjacent pixels of one channel into RGBA.
/// The channel is selected via `src_w` uniform (reused as output-width hint).
const FRAG_PLANAR_RGBA8_3PASS: &str = "\
#version 300 es
precision highp float;
uniform sampler2D tex;
uniform float src_w;
uniform float src_h;
in vec2 tc;
out vec4 color;
void main() {
    float ox = gl_FragCoord.x - 0.5;
    int out_x = int(ox);
    vec4 result;
    for (int i = 0; i < 4; i++) {
        int src_x = out_x * 4 + i;
        vec2 uv = vec2((float(src_x) + 0.5) / src_w, tc.y);
        result[i] = texture(tex, uv).r;
    }
    color = result;
}
";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// DRM FourCC for R8 (single-channel 8-bit).
fn r8_fourcc() -> u32 {
    gbm::drm::buffer::DrmFourcc::R8 as u32
}

/// Compile a shader program and retrieve uniform locations.
/// Returns `(program, tex_loc, src_w_loc, src_h_loc)` or `None` on failure.
fn compile_packing_program(frag_src: &str) -> Option<(u32, i32, i32, i32)> {
    let vert = CString::new(bench_render::VERTEX_SRC)
        .expect("vertex shader source contains interior null byte");
    let frag = CString::new(frag_src).expect("fragment shader source contains interior null byte");
    let program = match bench_render::compile_program(&vert, &frag) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("  shader compile failed: {e}");
            return None;
        }
    };
    unsafe {
        let tex_loc = gls::gl::GetUniformLocation(program, CString::new("tex").unwrap().as_ptr());
        let sw_loc = gls::gl::GetUniformLocation(program, CString::new("src_w").unwrap().as_ptr());
        let sh_loc = gls::gl::GetUniformLocation(program, CString::new("src_h").unwrap().as_ptr());
        Some((program, tex_loc, sw_loc, sh_loc))
    }
}

/// Fill a DMA tensor with a deterministic RGBA gradient for verification:
/// R = x%256, G = y%256, B = (x+y)%256, A = 255.
fn fill_verify_pattern(tensor: &Tensor<u8>, width: u32, height: u32) {
    let mut map = tensor
        .map()
        .expect("failed to map tensor for fill_verify_pattern");
    let slice = map.as_mut_slice();
    for y in 0..height {
        for x in 0..width {
            let offset = ((y * width + x) * 4) as usize;
            slice[offset] = (x % 256) as u8;
            slice[offset + 1] = (y % 256) as u8;
            slice[offset + 2] = ((x + y) % 256) as u8;
            slice[offset + 3] = 255;
        }
    }
}

/// Try to create an R8 EGLImage. Returns `None` if the driver doesn't support it.
fn try_create_r8_egl_image(
    ctx: &GpuContext,
    fd: i32,
    width: u32,
    height: u32,
) -> Option<khronos_egl::Image> {
    let pitch = width as i32; // 1 byte per pixel
    ctx.create_egl_image_dma(fd, width as i32, height as i32, r8_fourcc(), pitch)
        .ok()
}

/// Try to create an RGBA EGLImage. Returns `None` if the driver rejects it.
fn try_create_rgba_egl_image(
    ctx: &GpuContext,
    fd: i32,
    width: u32,
    height: u32,
) -> Option<khronos_egl::Image> {
    let pitch = (width * 4) as i32;
    ctx.create_egl_image_dma(
        fd,
        width as i32,
        height as i32,
        bench_render::rgba_fourcc(),
        pitch,
    )
    .ok()
}

/// Create a GL texture backed by an EGLImage, with NEAREST filtering.
fn create_texture_from_image(image: &khronos_egl::Image) -> u32 {
    unsafe {
        let mut tex = 0u32;
        gls::gl::GenTextures(1, &mut tex);
        gls::gl::BindTexture(gls::gl::TEXTURE_2D, tex);
        gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, image.as_ptr());
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
        tex
    }
}

/// Create an FBO and attach a texture. Returns `(fbo, ok)`.
fn create_fbo_with_texture(tex: u32) -> (u32, bool) {
    unsafe {
        let mut fbo = 0u32;
        gls::gl::GenFramebuffers(1, &mut fbo);
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
        gls::gl::FramebufferTexture2D(
            gls::gl::FRAMEBUFFER,
            gls::gl::COLOR_ATTACHMENT0,
            gls::gl::TEXTURE_2D,
            tex,
            0,
        );
        let status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
        (fbo, status == gls::gl::FRAMEBUFFER_COMPLETE)
    }
}

/// Bind program, set source texture on unit 0, set src_w/src_h uniforms.
unsafe fn bind_program_and_source(
    program: u32,
    tex_loc: i32,
    sw_loc: i32,
    sh_loc: i32,
    src_tex: u32,
    src_w: f32,
    src_h: f32,
) {
    gls::gl::UseProgram(program);
    gls::gl::ActiveTexture(gls::gl::TEXTURE0);
    gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
    gls::gl::Uniform1i(tex_loc, 0);
    if sw_loc >= 0 {
        gls::gl::Uniform1f(sw_loc, src_w);
    }
    if sh_loc >= 0 {
        gls::gl::Uniform1f(sh_loc, src_h);
    }
}

/// Draw the fullscreen quad and finish.
unsafe fn draw_and_finish(vao: u32) {
    gls::gl::BindVertexArray(vao);
    gls::gl::DrawElements(gls::gl::TRIANGLES, 6, gls::gl::UNSIGNED_INT, null());
    gls::gl::Finish();
}

/// Cleanup: delete FBO, textures, destroy EGLImages.
unsafe fn cleanup_gl(fbo: u32, textures: &[u32]) {
    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
    gls::gl::BindVertexArray(0);
    if fbo != 0 {
        gls::gl::DeleteFramebuffers(1, &fbo);
    }
    for &t in textures {
        if t != 0 {
            gls::gl::DeleteTextures(1, &t);
        }
    }
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// Run verification tests at the same `(src, dst)` configs used by the
/// benchmarks. This ensures we exercise exactly the resolutions and DMA
/// buffer sizes the benchmarks rely on and avoids false failures from
/// non-standard intermediate dimensions.
pub fn run_verify(ctx: &GpuContext) {
    println!("== Verification: RGB Packing ==");

    let (vao, vbo, ebo) = bench_render::create_quad_vao();

    let configs: &[(u32, u32, &str)] = &[
        (640, 640, "1080p_to_640"),
        (320, 320, "1080p_to_320"),
    ];

    for &(dst_w, dst_h, label) in configs {
        // Source: dst_w x dst_h RGBA — the packing shaders operate on the
        // resized frame, so the source for verification IS the dst dimensions.
        let src_w = dst_w;
        let src_h = dst_h;
        let src_bytes = (src_w * src_h * 4) as usize;
        let src_tensor = match Tensor::<u8>::new(&[src_bytes], Some(TensorMemory::Dma), None) {
            Ok(t) => t,
            Err(e) => {
                println!("  SKIP {label}: src DMA allocation failed: {e}");
                continue;
            }
        };
        fill_verify_pattern(&src_tensor, src_w, src_h);

        let src_fd_owned = src_tensor.clone_fd().unwrap();
        let src_img = match try_create_rgba_egl_image(ctx, src_fd_owned.as_raw_fd(), src_w, src_h) {
            Some(img) => img,
            None => {
                println!("  SKIP {label}: src RGBA EGLImage creation failed");
                continue;
            }
        };
        let src_tex = create_texture_from_image(&src_img);

        // --- Strategy 1: packed_r8 ---
        verify_packed_r8(ctx, src_tex, src_w, src_h, vao, label);

        // --- Strategy 2: packed_rgba8 ---
        verify_packed_rgba8(ctx, src_tex, src_w, src_h, vao, label);

        // --- Strategy 3a: planar_r8_1pass ---
        verify_planar_r8_1pass(ctx, src_tex, src_w, src_h, vao, label);

        // --- Strategy 3b: planar_r8_3pass ---
        verify_planar_r8_3pass(ctx, src_tex, src_w, src_h, vao, label);

        // --- Strategy 5: planar_rgba8_1pass ---
        verify_planar_rgba8_1pass(ctx, src_tex, src_w, src_h, vao, label);

        // --- Strategy 6: planar_rgba8_3pass ---
        verify_planar_rgba8_3pass(ctx, src_tex, src_w, src_h, vao, label);

        // Cleanup this config's source
        unsafe {
            gls::gl::DeleteTextures(1, &src_tex);
        }
        ctx.destroy_egl_image(src_img)
            .expect("destroy src EGLImage failed");
    }

    unsafe {
        gls::gl::DeleteVertexArrays(1, &vao);
        gls::gl::DeleteBuffers(1, &vbo);
        gls::gl::DeleteBuffers(1, &ebo);
    }

    println!();
}

fn verify_packed_r8(
    ctx: &GpuContext,
    src_tex: u32,
    src_w: u32,
    src_h: u32,
    vao: u32,
    label: &str,
) {
    let name = format!("rgb_verify/packed_r8/{label}");

    let (program, tex_loc, sw_loc, sh_loc) = match compile_packing_program(FRAG_PACKED_R8) {
        Some(p) => p,
        None => {
            println!("  {name:40} SKIP (shader compile failed)");
            return;
        }
    };

    let dst_w = src_w * 3;
    let dst_h = src_h;
    let dst_bytes = (dst_w * dst_h) as usize; // R8 = 1 byte/pixel
    let dst_tensor = match Tensor::<u8>::new(&[dst_bytes], Some(TensorMemory::Dma), None) {
        Ok(t) => t,
        Err(e) => {
            println!("  {name:40} SKIP (dst DMA alloc failed: {e})");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_fd_owned = dst_tensor.clone_fd().unwrap();
    let dst_img = match try_create_r8_egl_image(ctx, dst_fd_owned.as_raw_fd(), dst_w, dst_h) {
        Some(img) => img,
        None => {
            println!("  {name:40} SKIP (R8 EGLImage {dst_w}x{dst_h} failed)");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_tex = create_texture_from_image(&dst_img);
    let (fbo, fbo_ok) = create_fbo_with_texture(dst_tex);
    if !fbo_ok {
        println!("  {name:40} SKIP (FBO incomplete)");
        unsafe {
            cleanup_gl(fbo, &[dst_tex]);
            gls::gl::DeleteProgram(program);
        }
        let _ = ctx.destroy_egl_image(dst_img);
        return;
    }

    unsafe {
        gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
        bind_program_and_source(
            program,
            tex_loc,
            sw_loc,
            sh_loc,
            src_tex,
            src_w as f32,
            src_h as f32,
        );
        draw_and_finish(vao);
        cleanup_gl(fbo, &[dst_tex]);
        gls::gl::DeleteProgram(program);
    }
    let _ = ctx.destroy_egl_image(dst_img);

    // Verify
    let map = dst_tensor
        .map()
        .expect("failed to map dst tensor for verify");
    let slice = map.as_slice();
    let mut pass = true;
    for y in 0..src_h {
        for x in 0..src_w {
            let r_expected = (x % 256) as u8;
            let g_expected = (y % 256) as u8;
            let b_expected = ((x + y) % 256) as u8;
            let base = (y * dst_w + x * 3) as usize;
            let vals = [slice[base], slice[base + 1], slice[base + 2]];
            let expected = [r_expected, g_expected, b_expected];
            for c in 0..3 {
                let diff = (vals[c] as i16 - expected[c] as i16).unsigned_abs() as u8;
                if diff > 1 {
                    if pass {
                        println!(
                            "    MISMATCH at ({x},{y}) ch={c}: got {} expected {} (diff {diff})",
                            vals[c], expected[c]
                        );
                    }
                    pass = false;
                }
            }
        }
    }
    println!("  {name:40} {}", if pass { "PASS" } else { "FAIL" });
}

fn verify_packed_rgba8(
    ctx: &GpuContext,
    src_tex: u32,
    src_w: u32,
    src_h: u32,
    vao: u32,
    label: &str,
) {
    let name = format!("rgb_verify/packed_rgba8/{label}");
    let (program, tex_loc, sw_loc, sh_loc) = match compile_packing_program(FRAG_PACKED_RGBA8) {
        Some(p) => p,
        None => {
            println!("  {name:40} SKIP (shader compile failed)");
            return;
        }
    };

    // dst is (W*3/4) x H, RGBA8 = 4 bytes/pixel
    let dst_w = src_w * 3 / 4;
    let dst_h = src_h;
    let dst_bytes = (dst_w * dst_h * 4) as usize;
    let dst_tensor = match Tensor::<u8>::new(&[dst_bytes], Some(TensorMemory::Dma), None) {
        Ok(t) => t,
        Err(e) => {
            println!("  {name:40} SKIP (dst DMA alloc failed: {e})");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_fd_owned = dst_tensor.clone_fd().unwrap();
    let dst_img = match try_create_rgba_egl_image(ctx, dst_fd_owned.as_raw_fd(), dst_w, dst_h) {
        Some(img) => img,
        None => {
            println!("  {name:40} SKIP (RGBA EGLImage {dst_w}x{dst_h} failed)");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };
    let dst_tex = create_texture_from_image(&dst_img);
    let (fbo, fbo_ok) = create_fbo_with_texture(dst_tex);
    if !fbo_ok {
        println!("  {name:40} SKIP (FBO incomplete)");
        unsafe {
            cleanup_gl(fbo, &[dst_tex]);
            gls::gl::DeleteProgram(program);
        }
        let _ = ctx.destroy_egl_image(dst_img);
        return;
    }

    unsafe {
        gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
        bind_program_and_source(
            program,
            tex_loc,
            sw_loc,
            sh_loc,
            src_tex,
            src_w as f32,
            src_h as f32,
        );
        draw_and_finish(vao);
        cleanup_gl(fbo, &[dst_tex]);
        gls::gl::DeleteProgram(program);
    }
    let _ = ctx.destroy_egl_image(dst_img);

    // Verify: the RGBA output contains packed RGB bytes
    let map = dst_tensor
        .map()
        .expect("failed to map dst tensor for verify");
    let slice = map.as_slice();
    let mut pass = true;
    // Total RGB bytes = src_w * 3 per row
    for y in 0..src_h {
        for x in 0..src_w {
            let r_expected = (x % 256) as u8;
            let g_expected = (y % 256) as u8;
            let b_expected = ((x + y) % 256) as u8;
            let expected = [r_expected, g_expected, b_expected];
            for c in 0..3u32 {
                let byte_idx = (x * 3 + c) as usize;
                let rgba_pixel = byte_idx / 4;
                let rgba_ch = byte_idx % 4;
                let actual = slice[(y as usize * dst_w as usize + rgba_pixel) * 4 + rgba_ch];
                let diff = (actual as i16 - expected[c as usize] as i16).unsigned_abs() as u8;
                if diff > 1 {
                    if pass {
                        println!(
                            "    MISMATCH at ({x},{y}) ch={c}: got {actual} expected {} (diff {diff})",
                            expected[c as usize]
                        );
                    }
                    pass = false;
                }
            }
        }
    }
    println!("  {name:40} {}", if pass { "PASS" } else { "FAIL" });
}

fn verify_planar_r8_1pass(
    ctx: &GpuContext,
    src_tex: u32,
    src_w: u32,
    src_h: u32,
    vao: u32,
    label: &str,
) {
    let name = format!("rgb_verify/planar_r8_1pass/{label}");

    let (program, tex_loc, sw_loc, sh_loc) = match compile_packing_program(FRAG_PLANAR_R8_1PASS) {
        Some(p) => p,
        None => {
            println!("  {name:40} SKIP (shader compile failed)");
            return;
        }
    };

    let dst_w = src_w;
    let dst_h = src_h * 3;
    let dst_bytes = (dst_w * dst_h) as usize;
    let dst_tensor = match Tensor::<u8>::new(&[dst_bytes], Some(TensorMemory::Dma), None) {
        Ok(t) => t,
        Err(e) => {
            println!("  {name:40} SKIP (dst DMA alloc failed: {e})");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_fd_owned = dst_tensor.clone_fd().unwrap();
    let dst_img = match try_create_r8_egl_image(ctx, dst_fd_owned.as_raw_fd(), dst_w, dst_h) {
        Some(img) => img,
        None => {
            println!("  {name:40} SKIP (R8 EGLImage {dst_w}x{dst_h} failed)");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_tex = create_texture_from_image(&dst_img);
    let (fbo, fbo_ok) = create_fbo_with_texture(dst_tex);
    if !fbo_ok {
        println!("  {name:40} SKIP (FBO incomplete)");
        unsafe {
            cleanup_gl(fbo, &[dst_tex]);
            gls::gl::DeleteProgram(program);
        }
        let _ = ctx.destroy_egl_image(dst_img);
        return;
    }

    unsafe {
        gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
        bind_program_and_source(
            program,
            tex_loc,
            sw_loc,
            sh_loc,
            src_tex,
            src_w as f32,
            src_h as f32,
        );
        draw_and_finish(vao);
        cleanup_gl(fbo, &[dst_tex]);
        gls::gl::DeleteProgram(program);
    }
    let _ = ctx.destroy_egl_image(dst_img);

    // Verify: plane 0 = R, plane 1 = G, plane 2 = B
    let map = dst_tensor
        .map()
        .expect("failed to map dst tensor for verify");
    let slice = map.as_slice();
    let mut pass = true;
    let plane_size = (src_w * src_h) as usize;
    for y in 0..src_h {
        for x in 0..src_w {
            let expected = [(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8];
            for (plane, &exp) in expected.iter().enumerate() {
                let idx = plane * plane_size + (y * src_w + x) as usize;
                let diff = (slice[idx] as i16 - exp as i16).unsigned_abs() as u8;
                if diff > 1 {
                    if pass {
                        println!(
                            "    MISMATCH at ({x},{y}) plane={plane}: got {} expected {exp} (diff {diff})",
                            slice[idx]
                        );
                    }
                    pass = false;
                }
            }
        }
    }
    println!("  {name:40} {}", if pass { "PASS" } else { "FAIL" });
}

fn verify_planar_r8_3pass(
    ctx: &GpuContext,
    src_tex: u32,
    src_w: u32,
    src_h: u32,
    vao: u32,
    label: &str,
) {
    let name = format!("rgb_verify/planar_r8_3pass/{label}");

    let (program, tex_loc, _sw_loc, _sh_loc) = match compile_packing_program(FRAG_PLANAR_R8_3PASS) {
        Some(p) => p,
        None => {
            println!("  {name:40} SKIP (shader compile failed)");
            return;
        }
    };

    let dst_w = src_w;
    let dst_h = src_h * 3;
    let dst_bytes = (dst_w * dst_h) as usize;
    let dst_tensor = match Tensor::<u8>::new(&[dst_bytes], Some(TensorMemory::Dma), None) {
        Ok(t) => t,
        Err(e) => {
            println!("  {name:40} SKIP (dst DMA alloc failed: {e})");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_fd_owned = dst_tensor.clone_fd().unwrap();
    let dst_img = match try_create_r8_egl_image(ctx, dst_fd_owned.as_raw_fd(), dst_w, dst_h) {
        Some(img) => img,
        None => {
            println!("  {name:40} SKIP (R8 EGLImage {dst_w}x{dst_h} failed)");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_tex = create_texture_from_image(&dst_img);
    let (fbo, fbo_ok) = create_fbo_with_texture(dst_tex);
    if !fbo_ok {
        println!("  {name:40} SKIP (FBO incomplete)");
        unsafe {
            cleanup_gl(fbo, &[dst_tex]);
            gls::gl::DeleteProgram(program);
        }
        let _ = ctx.destroy_egl_image(dst_img);
        return;
    }

    // 3-pass rendering: one draw call per channel with TEXTURE_SWIZZLE_R
    let swizzles = [gls::gl::RED, gls::gl::GREEN, gls::gl::BLUE];
    unsafe {
        gls::gl::UseProgram(program);
        gls::gl::ActiveTexture(gls::gl::TEXTURE0);
        gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
        gls::gl::Uniform1i(tex_loc, 0);
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);

        for (i, &swizzle) in swizzles.iter().enumerate() {
            gls::gl::Viewport(0, (i as i32) * (src_h as i32), src_w as i32, src_h as i32);
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_SWIZZLE_R,
                swizzle as i32,
            );
            gls::gl::BindVertexArray(vao);
            gls::gl::DrawElements(gls::gl::TRIANGLES, 6, gls::gl::UNSIGNED_INT, null());
        }
        gls::gl::Finish();

        // Reset swizzle
        gls::gl::TexParameteri(
            gls::gl::TEXTURE_2D,
            gls::gl::TEXTURE_SWIZZLE_R,
            gls::gl::RED as i32,
        );
        cleanup_gl(fbo, &[dst_tex]);
        gls::gl::DeleteProgram(program);
    }
    let _ = ctx.destroy_egl_image(dst_img);

    // Verify same as planar_r8_1pass
    let map = dst_tensor
        .map()
        .expect("failed to map dst tensor for verify");
    let slice = map.as_slice();
    let mut pass = true;
    let plane_size = (src_w * src_h) as usize;
    for y in 0..src_h {
        for x in 0..src_w {
            let expected = [(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8];
            for (plane, &exp) in expected.iter().enumerate() {
                let idx = plane * plane_size + (y * src_w + x) as usize;
                let diff = (slice[idx] as i16 - exp as i16).unsigned_abs() as u8;
                if diff > 1 {
                    if pass {
                        println!(
                            "    MISMATCH at ({x},{y}) plane={plane}: got {} expected {exp} (diff {diff})",
                            slice[idx]
                        );
                    }
                    pass = false;
                }
            }
        }
    }
    println!("  {name:40} {}", if pass { "PASS" } else { "FAIL" });
}

fn verify_planar_rgba8_1pass(
    ctx: &GpuContext,
    src_tex: u32,
    src_w: u32,
    src_h: u32,
    vao: u32,
    label: &str,
) {
    let name = format!("rgb_verify/planar_rgba8_1pass/{label}");
    let (program, tex_loc, sw_loc, sh_loc) = match compile_packing_program(FRAG_PLANAR_RGBA8_1PASS)
    {
        Some(p) => p,
        None => {
            println!("  {name:40} SKIP (shader compile failed)");
            return;
        }
    };

    // dst is (W/4) x (H*3), RGBA8
    let dst_w = src_w / 4;
    let dst_h = src_h * 3;
    let dst_bytes = (dst_w * dst_h * 4) as usize;
    let dst_tensor = match Tensor::<u8>::new(&[dst_bytes], Some(TensorMemory::Dma), None) {
        Ok(t) => t,
        Err(e) => {
            println!("  {name:40} SKIP (dst DMA alloc failed: {e})");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_fd_owned = dst_tensor.clone_fd().unwrap();
    let dst_img = match try_create_rgba_egl_image(ctx, dst_fd_owned.as_raw_fd(), dst_w, dst_h) {
        Some(img) => img,
        None => {
            println!("  {name:40} SKIP (RGBA EGLImage {dst_w}x{dst_h} failed)");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };
    let dst_tex = create_texture_from_image(&dst_img);
    let (fbo, fbo_ok) = create_fbo_with_texture(dst_tex);
    if !fbo_ok {
        println!("  {name:40} SKIP (FBO incomplete)");
        unsafe {
            cleanup_gl(fbo, &[dst_tex]);
            gls::gl::DeleteProgram(program);
        }
        let _ = ctx.destroy_egl_image(dst_img);
        return;
    }

    unsafe {
        gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
        bind_program_and_source(
            program,
            tex_loc,
            sw_loc,
            sh_loc,
            src_tex,
            src_w as f32,
            src_h as f32,
        );
        draw_and_finish(vao);
        cleanup_gl(fbo, &[dst_tex]);
        gls::gl::DeleteProgram(program);
    }
    let _ = ctx.destroy_egl_image(dst_img);

    // Verify: 3 planes in the RGBA buffer, each plane has W/4 RGBA pixels
    // meaning 4 channel values per pixel = 4 source pixels packed
    let map = dst_tensor
        .map()
        .expect("failed to map dst tensor for verify");
    let slice = map.as_slice();
    let mut pass = true;
    for y in 0..src_h {
        for x in 0..src_w {
            let expected = [(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8];
            for (plane, &exp) in expected.iter().enumerate() {
                // In the output, plane data is at row (plane * src_h + y)
                // Each RGBA pixel holds 4 adjacent source pixels
                let rgba_pixel = x / 4;
                let rgba_ch = (x % 4) as usize;
                let row = plane as u32 * src_h + y;
                let idx = (row * dst_w + rgba_pixel) as usize * 4 + rgba_ch;
                let diff = (slice[idx] as i16 - exp as i16).unsigned_abs() as u8;
                if diff > 1 {
                    if pass {
                        println!(
                            "    MISMATCH at ({x},{y}) plane={plane}: got {} expected {exp} (diff {diff})",
                            slice[idx]
                        );
                    }
                    pass = false;
                }
            }
        }
    }
    println!("  {name:40} {}", if pass { "PASS" } else { "FAIL" });
}

fn verify_planar_rgba8_3pass(
    ctx: &GpuContext,
    src_tex: u32,
    src_w: u32,
    src_h: u32,
    vao: u32,
    label: &str,
) {
    let name = format!("rgb_verify/planar_rgba8_3pass/{label}");
    let (program, tex_loc, sw_loc, sh_loc) = match compile_packing_program(FRAG_PLANAR_RGBA8_3PASS)
    {
        Some(p) => p,
        None => {
            println!("  {name:40} SKIP (shader compile failed)");
            return;
        }
    };

    // dst is (W/4) x (H*3), RGBA8
    let dst_w = src_w / 4;
    let dst_h = src_h * 3;
    let dst_bytes = (dst_w * dst_h * 4) as usize;
    let dst_tensor = match Tensor::<u8>::new(&[dst_bytes], Some(TensorMemory::Dma), None) {
        Ok(t) => t,
        Err(e) => {
            println!("  {name:40} SKIP (dst DMA alloc failed: {e})");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_fd_owned = dst_tensor.clone_fd().unwrap();
    let dst_img = match try_create_rgba_egl_image(ctx, dst_fd_owned.as_raw_fd(), dst_w, dst_h) {
        Some(img) => img,
        None => {
            println!("  {name:40} SKIP (RGBA EGLImage {dst_w}x{dst_h} failed)");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };
    let dst_tex = create_texture_from_image(&dst_img);
    let (fbo, fbo_ok) = create_fbo_with_texture(dst_tex);
    if !fbo_ok {
        println!("  {name:40} SKIP (FBO incomplete)");
        unsafe {
            cleanup_gl(fbo, &[dst_tex]);
            gls::gl::DeleteProgram(program);
        }
        let _ = ctx.destroy_egl_image(dst_img);
        return;
    }

    // 3-pass: one draw call per channel with TEXTURE_SWIZZLE_R
    let swizzles = [gls::gl::RED, gls::gl::GREEN, gls::gl::BLUE];
    unsafe {
        gls::gl::UseProgram(program);
        gls::gl::ActiveTexture(gls::gl::TEXTURE0);
        gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
        gls::gl::Uniform1i(tex_loc, 0);
        if sw_loc >= 0 {
            gls::gl::Uniform1f(sw_loc, src_w as f32);
        }
        if sh_loc >= 0 {
            gls::gl::Uniform1f(sh_loc, src_h as f32);
        }
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);

        for (i, &swizzle) in swizzles.iter().enumerate() {
            gls::gl::Viewport(0, (i as i32) * (src_h as i32), dst_w as i32, src_h as i32);
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_SWIZZLE_R,
                swizzle as i32,
            );
            gls::gl::BindVertexArray(vao);
            gls::gl::DrawElements(gls::gl::TRIANGLES, 6, gls::gl::UNSIGNED_INT, null());
        }
        gls::gl::Finish();

        // Reset swizzle
        gls::gl::TexParameteri(
            gls::gl::TEXTURE_2D,
            gls::gl::TEXTURE_SWIZZLE_R,
            gls::gl::RED as i32,
        );
        cleanup_gl(fbo, &[dst_tex]);
        gls::gl::DeleteProgram(program);
    }
    let _ = ctx.destroy_egl_image(dst_img);

    // Verify: same layout as planar_rgba8_1pass
    let map = dst_tensor
        .map()
        .expect("failed to map dst tensor for verify");
    let slice = map.as_slice();
    let mut pass = true;
    for y in 0..src_h {
        for x in 0..src_w {
            let expected = [(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8];
            for (plane, &exp) in expected.iter().enumerate() {
                let rgba_pixel = x / 4;
                let rgba_ch = (x % 4) as usize;
                let row = plane as u32 * src_h + y;
                let idx = (row * dst_w + rgba_pixel) as usize * 4 + rgba_ch;
                let diff = (slice[idx] as i16 - exp as i16).unsigned_abs() as u8;
                if diff > 1 {
                    if pass {
                        println!(
                            "    MISMATCH at ({x},{y}) plane={plane}: got {} expected {exp} (diff {diff})",
                            slice[idx]
                        );
                    }
                    pass = false;
                }
            }
        }
    }
    println!("  {name:40} {}", if pass { "PASS" } else { "FAIL" });
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

/// Run RGB packing benchmarks and return collected results.
pub fn run(ctx: &GpuContext) -> Vec<BenchResult> {
    println!("== Benchmark: RGB Packing ==");

    let mut results = Vec::new();
    let (vao, vbo, ebo) = bench_render::create_quad_vao();

    let configs: &[(u32, u32, u32, u32, &str)] = &[
        (1920, 1080, 640, 640, "1080p_to_640"),
        (1920, 1080, 320, 320, "1080p_to_320"),
    ];

    for &(src_w, src_h, dst_w, dst_h, label) in configs {
        // Allocate source RGBA tensor
        let src_bytes = (src_w * src_h * 4) as usize;
        let src_tensor = match Tensor::<u8>::new(&[src_bytes], Some(TensorMemory::Dma), None) {
            Ok(t) => t,
            Err(e) => {
                println!("  SKIP {label}: src DMA allocation failed: {e}");
                continue;
            }
        };
        let src_fd_owned = src_tensor.clone_fd().unwrap();
        let src_fd = src_fd_owned.as_raw_fd();
        let src_img = match try_create_rgba_egl_image(ctx, src_fd, src_w, src_h) {
            Some(img) => img,
            None => {
                println!("  SKIP {label}: src RGBA EGLImage creation failed");
                continue;
            }
        };
        let src_tex = create_texture_from_image(&src_img);

        // Probe R8 support once per config
        let r8_supported = {
            let test_size = (dst_w * dst_h) as usize;
            match Tensor::<u8>::new(&[test_size], Some(TensorMemory::Dma), None) {
                Ok(t) => {
                    let fd = t.clone_fd().unwrap();
                    let ok = try_create_r8_egl_image(ctx, fd.as_raw_fd(), dst_w, dst_h).is_some();
                    if let Some(img) = try_create_r8_egl_image(ctx, fd.as_raw_fd(), dst_w, dst_h) {
                        let _ = ctx.destroy_egl_image(img);
                    }
                    ok
                }
                Err(_) => false,
            }
        };

        // --- Strategy 1: packed_r8 ---
        bench_packed_r8(
            ctx,
            src_tex,
            src_w,
            src_h,
            dst_w,
            dst_h,
            label,
            vao,
            r8_supported,
            &mut results,
        );

        // --- Strategy 2: packed_rgba8 ---
        bench_packed_rgba8(
            ctx,
            src_tex,
            src_w,
            src_h,
            dst_w,
            dst_h,
            label,
            vao,
            &mut results,
        );

        // --- Strategy 3a: planar_r8_1pass ---
        bench_planar_r8_1pass(
            ctx,
            src_tex,
            src_w,
            src_h,
            dst_w,
            dst_h,
            label,
            vao,
            r8_supported,
            &mut results,
        );

        // --- Strategy 3b: planar_r8_3pass ---
        bench_planar_r8_3pass(
            ctx,
            src_tex,
            src_w,
            src_h,
            dst_w,
            dst_h,
            label,
            vao,
            r8_supported,
            &mut results,
        );

        // --- Strategy 5: planar_rgba8_1pass ---
        bench_planar_rgba8_1pass(
            ctx,
            src_tex,
            src_w,
            src_h,
            dst_w,
            dst_h,
            label,
            vao,
            &mut results,
        );

        // --- Strategy 6: planar_rgba8_3pass ---
        bench_planar_rgba8_3pass(
            ctx,
            src_tex,
            src_w,
            src_h,
            dst_w,
            dst_h,
            label,
            vao,
            &mut results,
        );

        // Cleanup source resources
        unsafe {
            gls::gl::DeleteTextures(1, &src_tex);
        }
        ctx.destroy_egl_image(src_img)
            .expect("destroy src EGLImage failed");
    }

    unsafe {
        gls::gl::DeleteVertexArrays(1, &vao);
        gls::gl::DeleteBuffers(1, &vbo);
        gls::gl::DeleteBuffers(1, &ebo);
    }

    println!();
    results
}

#[allow(clippy::too_many_arguments)]
fn bench_packed_r8(
    ctx: &GpuContext,
    src_tex: u32,
    _src_w: u32,
    _src_h: u32,
    dst_w: u32,
    dst_h: u32,
    label: &str,
    vao: u32,
    r8_supported: bool,
    results: &mut Vec<BenchResult>,
) {
    let bench_name = format!("packed_r8/{label}");
    if !r8_supported {
        println!("  {bench_name:40} SKIP (R8 not supported)");
        return;
    }

    let (program, tex_loc, sw_loc, sh_loc) = match compile_packing_program(FRAG_PACKED_R8) {
        Some(p) => p,
        None => {
            println!("  {bench_name:40} SKIP (shader compile failed)");
            return;
        }
    };

    let out_w = dst_w * 3;
    let out_h = dst_h;
    let dst_bytes = (out_w * out_h) as usize;
    let dst_tensor = match Tensor::<u8>::new(&[dst_bytes], Some(TensorMemory::Dma), None) {
        Ok(t) => t,
        Err(e) => {
            println!("  {bench_name:40} SKIP (dst DMA alloc failed: {e})");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_fd_owned = dst_tensor.clone_fd().unwrap();
    let dst_img = match try_create_r8_egl_image(ctx, dst_fd_owned.as_raw_fd(), out_w, out_h) {
        Some(img) => img,
        None => {
            println!("  {bench_name:40} SKIP (R8 EGLImage failed)");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_tex = create_texture_from_image(&dst_img);
    let (fbo, fbo_ok) = create_fbo_with_texture(dst_tex);
    if !fbo_ok {
        println!("  {bench_name:40} SKIP (FBO incomplete)");
        unsafe {
            cleanup_gl(fbo, &[dst_tex]);
            gls::gl::DeleteProgram(program);
        }
        let _ = ctx.destroy_egl_image(dst_img);
        return;
    }

    let r = run_bench(&bench_name, 10, 200, || unsafe {
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
        gls::gl::Viewport(0, 0, out_w as i32, out_h as i32);
        bind_program_and_source(
            program,
            tex_loc,
            sw_loc,
            sh_loc,
            src_tex,
            dst_w as f32,
            dst_h as f32,
        );
        draw_and_finish(vao);
    });
    r.print_summary();
    results.push(r);

    unsafe {
        cleanup_gl(fbo, &[dst_tex]);
        gls::gl::DeleteProgram(program);
    }
    let _ = ctx.destroy_egl_image(dst_img);
}

#[allow(clippy::too_many_arguments)]
fn bench_packed_rgba8(
    ctx: &GpuContext,
    src_tex: u32,
    _src_w: u32,
    _src_h: u32,
    dst_w: u32,
    dst_h: u32,
    label: &str,
    vao: u32,
    results: &mut Vec<BenchResult>,
) {
    let bench_name = format!("packed_rgba8/{label}");
    let (program, tex_loc, sw_loc, sh_loc) = match compile_packing_program(FRAG_PACKED_RGBA8) {
        Some(p) => p,
        None => {
            println!("  {bench_name:40} SKIP (shader compile failed)");
            return;
        }
    };

    let out_w = dst_w * 3 / 4;
    let out_h = dst_h;
    let dst_bytes = (out_w * out_h * 4) as usize;
    let dst_tensor = match Tensor::<u8>::new(&[dst_bytes], Some(TensorMemory::Dma), None) {
        Ok(t) => t,
        Err(e) => {
            println!("  {bench_name:40} SKIP (dst DMA alloc failed: {e})");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_fd_owned = dst_tensor.clone_fd().unwrap();
    let dst_img = match try_create_rgba_egl_image(ctx, dst_fd_owned.as_raw_fd(), out_w, out_h) {
        Some(img) => img,
        None => {
            println!("  {bench_name:40} SKIP (RGBA EGLImage failed)");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };
    let dst_tex = create_texture_from_image(&dst_img);
    let (fbo, fbo_ok) = create_fbo_with_texture(dst_tex);
    if !fbo_ok {
        println!("  {bench_name:40} SKIP (FBO incomplete)");
        unsafe {
            cleanup_gl(fbo, &[dst_tex]);
            gls::gl::DeleteProgram(program);
        }
        let _ = ctx.destroy_egl_image(dst_img);
        return;
    }

    let r = run_bench(&bench_name, 10, 200, || unsafe {
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
        gls::gl::Viewport(0, 0, out_w as i32, out_h as i32);
        bind_program_and_source(
            program,
            tex_loc,
            sw_loc,
            sh_loc,
            src_tex,
            dst_w as f32,
            dst_h as f32,
        );
        draw_and_finish(vao);
    });
    r.print_summary();
    results.push(r);

    unsafe {
        cleanup_gl(fbo, &[dst_tex]);
        gls::gl::DeleteProgram(program);
    }
    let _ = ctx.destroy_egl_image(dst_img);
}

#[allow(clippy::too_many_arguments)]
fn bench_planar_r8_1pass(
    ctx: &GpuContext,
    src_tex: u32,
    _src_w: u32,
    _src_h: u32,
    dst_w: u32,
    dst_h: u32,
    label: &str,
    vao: u32,
    r8_supported: bool,
    results: &mut Vec<BenchResult>,
) {
    let bench_name = format!("planar_r8_1pass/{label}");
    if !r8_supported {
        println!("  {bench_name:40} SKIP (R8 not supported)");
        return;
    }

    let (program, tex_loc, sw_loc, sh_loc) = match compile_packing_program(FRAG_PLANAR_R8_1PASS) {
        Some(p) => p,
        None => {
            println!("  {bench_name:40} SKIP (shader compile failed)");
            return;
        }
    };

    let out_w = dst_w;
    let out_h = dst_h * 3;
    let dst_bytes = (out_w * out_h) as usize;
    let dst_tensor = match Tensor::<u8>::new(&[dst_bytes], Some(TensorMemory::Dma), None) {
        Ok(t) => t,
        Err(e) => {
            println!("  {bench_name:40} SKIP (dst DMA alloc failed: {e})");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_fd_owned = dst_tensor.clone_fd().unwrap();
    let dst_img = match try_create_r8_egl_image(ctx, dst_fd_owned.as_raw_fd(), out_w, out_h) {
        Some(img) => img,
        None => {
            println!("  {bench_name:40} SKIP (R8 EGLImage failed)");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_tex = create_texture_from_image(&dst_img);
    let (fbo, fbo_ok) = create_fbo_with_texture(dst_tex);
    if !fbo_ok {
        println!("  {bench_name:40} SKIP (FBO incomplete)");
        unsafe {
            cleanup_gl(fbo, &[dst_tex]);
            gls::gl::DeleteProgram(program);
        }
        let _ = ctx.destroy_egl_image(dst_img);
        return;
    }

    let r = run_bench(&bench_name, 10, 200, || unsafe {
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
        gls::gl::Viewport(0, 0, out_w as i32, out_h as i32);
        bind_program_and_source(
            program,
            tex_loc,
            sw_loc,
            sh_loc,
            src_tex,
            dst_w as f32,
            dst_h as f32,
        );
        draw_and_finish(vao);
    });
    r.print_summary();
    results.push(r);

    unsafe {
        cleanup_gl(fbo, &[dst_tex]);
        gls::gl::DeleteProgram(program);
    }
    let _ = ctx.destroy_egl_image(dst_img);
}

#[allow(clippy::too_many_arguments)]
fn bench_planar_r8_3pass(
    ctx: &GpuContext,
    src_tex: u32,
    _src_w: u32,
    _src_h: u32,
    dst_w: u32,
    dst_h: u32,
    label: &str,
    vao: u32,
    r8_supported: bool,
    results: &mut Vec<BenchResult>,
) {
    let bench_name = format!("planar_r8_3pass/{label}");
    if !r8_supported {
        println!("  {bench_name:40} SKIP (R8 not supported)");
        return;
    }

    let (program, tex_loc, _sw_loc, _sh_loc) = match compile_packing_program(FRAG_PLANAR_R8_3PASS) {
        Some(p) => p,
        None => {
            println!("  {bench_name:40} SKIP (shader compile failed)");
            return;
        }
    };

    let out_w = dst_w;
    let out_h = dst_h * 3;
    let dst_bytes = (out_w * out_h) as usize;
    let dst_tensor = match Tensor::<u8>::new(&[dst_bytes], Some(TensorMemory::Dma), None) {
        Ok(t) => t,
        Err(e) => {
            println!("  {bench_name:40} SKIP (dst DMA alloc failed: {e})");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_fd_owned = dst_tensor.clone_fd().unwrap();
    let dst_img = match try_create_r8_egl_image(ctx, dst_fd_owned.as_raw_fd(), out_w, out_h) {
        Some(img) => img,
        None => {
            println!("  {bench_name:40} SKIP (R8 EGLImage failed)");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_tex = create_texture_from_image(&dst_img);
    let (fbo, fbo_ok) = create_fbo_with_texture(dst_tex);
    if !fbo_ok {
        println!("  {bench_name:40} SKIP (FBO incomplete)");
        unsafe {
            cleanup_gl(fbo, &[dst_tex]);
            gls::gl::DeleteProgram(program);
        }
        let _ = ctx.destroy_egl_image(dst_img);
        return;
    }

    let swizzles = [gls::gl::RED, gls::gl::GREEN, gls::gl::BLUE];

    let r = run_bench(&bench_name, 10, 200, || unsafe {
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
        gls::gl::UseProgram(program);
        gls::gl::ActiveTexture(gls::gl::TEXTURE0);
        gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
        gls::gl::Uniform1i(tex_loc, 0);

        for (i, &swizzle) in swizzles.iter().enumerate() {
            gls::gl::Viewport(0, (i as i32) * (dst_h as i32), dst_w as i32, dst_h as i32);
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_SWIZZLE_R,
                swizzle as i32,
            );
            gls::gl::BindVertexArray(vao);
            gls::gl::DrawElements(gls::gl::TRIANGLES, 6, gls::gl::UNSIGNED_INT, null());
        }
        gls::gl::Finish();
    });
    r.print_summary();
    results.push(r);

    // Reset swizzle and cleanup
    unsafe {
        gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
        gls::gl::TexParameteri(
            gls::gl::TEXTURE_2D,
            gls::gl::TEXTURE_SWIZZLE_R,
            gls::gl::RED as i32,
        );
        cleanup_gl(fbo, &[dst_tex]);
        gls::gl::DeleteProgram(program);
    }
    let _ = ctx.destroy_egl_image(dst_img);
}

#[allow(clippy::too_many_arguments)]
fn bench_planar_rgba8_1pass(
    ctx: &GpuContext,
    src_tex: u32,
    _src_w: u32,
    _src_h: u32,
    dst_w: u32,
    dst_h: u32,
    label: &str,
    vao: u32,
    results: &mut Vec<BenchResult>,
) {
    let bench_name = format!("planar_rgba8_1pass/{label}");
    let (program, tex_loc, sw_loc, sh_loc) = match compile_packing_program(FRAG_PLANAR_RGBA8_1PASS)
    {
        Some(p) => p,
        None => {
            println!("  {bench_name:40} SKIP (shader compile failed)");
            return;
        }
    };

    let out_w = dst_w / 4;
    let out_h = dst_h * 3;
    let dst_bytes = (out_w * out_h * 4) as usize;
    let dst_tensor = match Tensor::<u8>::new(&[dst_bytes], Some(TensorMemory::Dma), None) {
        Ok(t) => t,
        Err(e) => {
            println!("  {bench_name:40} SKIP (dst DMA alloc failed: {e})");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_fd_owned = dst_tensor.clone_fd().unwrap();
    let dst_img = match try_create_rgba_egl_image(ctx, dst_fd_owned.as_raw_fd(), out_w, out_h) {
        Some(img) => img,
        None => {
            println!("  {bench_name:40} SKIP (RGBA EGLImage failed)");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };
    let dst_tex = create_texture_from_image(&dst_img);
    let (fbo, fbo_ok) = create_fbo_with_texture(dst_tex);
    if !fbo_ok {
        println!("  {bench_name:40} SKIP (FBO incomplete)");
        unsafe {
            cleanup_gl(fbo, &[dst_tex]);
            gls::gl::DeleteProgram(program);
        }
        let _ = ctx.destroy_egl_image(dst_img);
        return;
    }

    let r = run_bench(&bench_name, 10, 200, || unsafe {
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
        gls::gl::Viewport(0, 0, out_w as i32, out_h as i32);
        bind_program_and_source(
            program,
            tex_loc,
            sw_loc,
            sh_loc,
            src_tex,
            dst_w as f32,
            dst_h as f32,
        );
        draw_and_finish(vao);
    });
    r.print_summary();
    results.push(r);

    unsafe {
        cleanup_gl(fbo, &[dst_tex]);
        gls::gl::DeleteProgram(program);
    }
    let _ = ctx.destroy_egl_image(dst_img);
}

#[allow(clippy::too_many_arguments)]
fn bench_planar_rgba8_3pass(
    ctx: &GpuContext,
    src_tex: u32,
    _src_w: u32,
    _src_h: u32,
    dst_w: u32,
    dst_h: u32,
    label: &str,
    vao: u32,
    results: &mut Vec<BenchResult>,
) {
    let bench_name = format!("planar_rgba8_3pass/{label}");
    let (program, tex_loc, sw_loc, sh_loc) = match compile_packing_program(FRAG_PLANAR_RGBA8_3PASS)
    {
        Some(p) => p,
        None => {
            println!("  {bench_name:40} SKIP (shader compile failed)");
            return;
        }
    };

    let out_w = dst_w / 4;
    let out_h = dst_h * 3;
    let dst_bytes = (out_w * out_h * 4) as usize;
    let dst_tensor = match Tensor::<u8>::new(&[dst_bytes], Some(TensorMemory::Dma), None) {
        Ok(t) => t,
        Err(e) => {
            println!("  {bench_name:40} SKIP (dst DMA alloc failed: {e})");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };

    let dst_fd_owned = dst_tensor.clone_fd().unwrap();
    let dst_img = match try_create_rgba_egl_image(ctx, dst_fd_owned.as_raw_fd(), out_w, out_h) {
        Some(img) => img,
        None => {
            println!("  {bench_name:40} SKIP (RGBA EGLImage failed)");
            unsafe {
                gls::gl::DeleteProgram(program);
            }
            return;
        }
    };
    let dst_tex = create_texture_from_image(&dst_img);
    let (fbo, fbo_ok) = create_fbo_with_texture(dst_tex);
    if !fbo_ok {
        println!("  {bench_name:40} SKIP (FBO incomplete)");
        unsafe {
            cleanup_gl(fbo, &[dst_tex]);
            gls::gl::DeleteProgram(program);
        }
        let _ = ctx.destroy_egl_image(dst_img);
        return;
    }

    let swizzles = [gls::gl::RED, gls::gl::GREEN, gls::gl::BLUE];

    let r = run_bench(&bench_name, 10, 200, || unsafe {
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
        gls::gl::UseProgram(program);
        gls::gl::ActiveTexture(gls::gl::TEXTURE0);
        gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
        gls::gl::Uniform1i(tex_loc, 0);
        if sw_loc >= 0 {
            gls::gl::Uniform1f(sw_loc, dst_w as f32);
        }
        if sh_loc >= 0 {
            gls::gl::Uniform1f(sh_loc, dst_h as f32);
        }

        for (i, &swizzle) in swizzles.iter().enumerate() {
            gls::gl::Viewport(0, (i as i32) * (dst_h as i32), out_w as i32, dst_h as i32);
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_SWIZZLE_R,
                swizzle as i32,
            );
            gls::gl::BindVertexArray(vao);
            gls::gl::DrawElements(gls::gl::TRIANGLES, 6, gls::gl::UNSIGNED_INT, null());
        }
        gls::gl::Finish();
    });
    r.print_summary();
    results.push(r);

    // Reset swizzle and cleanup
    unsafe {
        gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
        gls::gl::TexParameteri(
            gls::gl::TEXTURE_2D,
            gls::gl::TEXTURE_SWIZZLE_R,
            gls::gl::RED as i32,
        );
        cleanup_gl(fbo, &[dst_tex]);
        gls::gl::DeleteProgram(program);
    }
    let _ = ctx.destroy_egl_image(dst_img);
}
