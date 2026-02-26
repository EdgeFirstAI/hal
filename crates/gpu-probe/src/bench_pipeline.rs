// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! DMA-buf pipeline verification and rebinding benchmarks.
//!
//! Isolates EGLImage rebinding cost — the operation the HAL's `convert()` path
//! performs every frame when a new DMA-buf arrives. Verifies pixel correctness
//! of the full dmabuf→GL→dmabuf pipeline and compares solid-fill versus
//! synthetic-pattern texture sources.

use crate::bench::{run_bench, BenchResult};
use crate::bench_render;
use crate::egl_context::GpuContext;
use edgefirst_tensor::{Tensor, TensorMapTrait, TensorMemory, TensorTrait};
use std::ffi::CString;
use std::os::unix::io::AsRawFd;
use std::ptr::null;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Fill every pixel in a DMA tensor with a solid RGBA value.
fn fill_solid(tensor: &Tensor<u8>, pixel: [u8; 4]) {
    let mut map = tensor.map().expect("failed to map tensor for fill_solid");
    let slice = map.as_mut_slice();
    for chunk in slice.chunks_exact_mut(4) {
        chunk.copy_from_slice(&pixel);
    }
}

/// Fill a DMA tensor with a synthetic RGBA gradient pattern.
///
/// R increases left→right, G increases top→bottom, B=128 constant, A=255.
/// Produces non-uniform pixel data that exercises the bilinear filter during
/// resize.
fn fill_gradient(tensor: &Tensor<u8>, width: u32, height: u32) {
    let mut map = tensor
        .map()
        .expect("failed to map tensor for fill_gradient");
    let slice = map.as_mut_slice();
    for y in 0..height {
        for x in 0..width {
            let offset = ((y * width + x) * 4) as usize;
            slice[offset] = (x * 255 / width.max(1)) as u8; // R
            slice[offset + 1] = (y * 255 / height.max(1)) as u8; // G
            slice[offset + 2] = 128; // B
            slice[offset + 3] = 255; // A
        }
    }
}

/// Verify every pixel in a DMA tensor matches `expected` within per-channel
/// `tolerance`. Prints the first mismatch on failure and returns false.
fn verify_pixels(tensor: &Tensor<u8>, expected: [u8; 4], tolerance: u8) -> bool {
    let map = tensor
        .map()
        .expect("failed to map tensor for verify_pixels");
    let slice = map.as_slice();
    for (i, chunk) in slice.chunks_exact(4).enumerate() {
        for c in 0..4 {
            let diff = (chunk[c] as i16 - expected[c] as i16).unsigned_abs() as u8;
            if diff > tolerance {
                println!(
                    "    MISMATCH at pixel {i}: got [{}, {}, {}, {}] expected [{}, {}, {}, {}] (tolerance {tolerance})",
                    chunk[0], chunk[1], chunk[2], chunk[3],
                    expected[0], expected[1], expected[2], expected[3],
                );
                return false;
            }
        }
    }
    true
}

/// Verify that the destination tensor has non-zero, varied data.
///
/// For the gradient resize test where exact values depend on resize filtering,
/// we check that the center pixel is non-zero and that at least two different
/// pixel values exist.
fn verify_not_zero(tensor: &Tensor<u8>) -> bool {
    let map = tensor
        .map()
        .expect("failed to map tensor for verify_not_zero");
    let slice = map.as_slice();

    if slice.len() < 4 {
        println!("    FAIL: tensor too small");
        return false;
    }

    // Check center pixel is non-zero
    let center = slice.len() / 2;
    let center_start = center - (center % 4);
    let center_pixel = &slice[center_start..center_start + 4];
    if center_pixel == [0, 0, 0, 0] {
        println!("    FAIL: center pixel is all zeros");
        return false;
    }

    // Check that at least two different pixel values exist
    let first = &slice[0..4];
    let has_variation = slice.chunks_exact(4).any(|px| px != first);
    if !has_variation {
        println!(
            "    FAIL: all pixels identical [{}, {}, {}, {}]",
            first[0], first[1], first[2], first[3],
        );
        return false;
    }

    true
}

// ---------------------------------------------------------------------------
// Full pipeline helper: create EGLImages → bind textures → FBO → render →
// glFinish → cleanup
// ---------------------------------------------------------------------------

/// Execute the full DMA-buf pipeline once (create → render → destroy).
#[allow(clippy::too_many_arguments)]
fn run_full_pipeline(
    ctx: &GpuContext,
    src_fd: i32,
    src_w: u32,
    src_h: u32,
    dst_fd: i32,
    dst_w: u32,
    dst_h: u32,
    fourcc: u32,
    program: u32,
    tex_uniform: i32,
    vao: u32,
) {
    let src_pitch = (src_w * 4) as i32;
    let dst_pitch = (dst_w * 4) as i32;

    let src_img = ctx
        .create_egl_image_dma(src_fd, src_w as i32, src_h as i32, fourcc, src_pitch)
        .expect("src create_egl_image_dma failed");
    let dst_img = ctx
        .create_egl_image_dma(dst_fd, dst_w as i32, dst_h as i32, fourcc, dst_pitch)
        .expect("dst create_egl_image_dma failed");

    unsafe {
        let mut src_tex = 0u32;
        gls::gl::GenTextures(1, &mut src_tex);
        gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
        gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, src_img.as_ptr());
        gls::gl::TexParameteri(
            gls::gl::TEXTURE_2D,
            gls::gl::TEXTURE_MIN_FILTER,
            gls::gl::LINEAR as i32,
        );
        gls::gl::TexParameteri(
            gls::gl::TEXTURE_2D,
            gls::gl::TEXTURE_MAG_FILTER,
            gls::gl::LINEAR as i32,
        );

        let mut dst_tex = 0u32;
        gls::gl::GenTextures(1, &mut dst_tex);
        gls::gl::BindTexture(gls::gl::TEXTURE_2D, dst_tex);
        gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, dst_img.as_ptr());

        let mut fbo = 0u32;
        gls::gl::GenFramebuffers(1, &mut fbo);
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
        gls::gl::FramebufferTexture2D(
            gls::gl::FRAMEBUFFER,
            gls::gl::COLOR_ATTACHMENT0,
            gls::gl::TEXTURE_2D,
            dst_tex,
            0,
        );

        gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
        gls::gl::UseProgram(program);
        gls::gl::ActiveTexture(gls::gl::TEXTURE0);
        gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
        gls::gl::Uniform1i(tex_uniform, 0);

        gls::gl::BindVertexArray(vao);
        gls::gl::DrawElements(gls::gl::TRIANGLES, 6, gls::gl::UNSIGNED_INT, null());
        gls::gl::Finish();

        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
        gls::gl::BindVertexArray(0);
        gls::gl::DeleteFramebuffers(1, &fbo);
        gls::gl::DeleteTextures(1, &dst_tex);
        gls::gl::DeleteTextures(1, &src_tex);
    }

    ctx.destroy_egl_image(src_img)
        .expect("destroy src EGLImage failed");
    ctx.destroy_egl_image(dst_img)
        .expect("destroy dst EGLImage failed");
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// Run pipeline verification tests. Prints PASS/FAIL for each test.
pub fn run_verify(ctx: &GpuContext) {
    println!("== Verification: DMA-buf Pipeline ==");

    let vert_cstr = CString::new(bench_render::VERTEX_SRC)
        .expect("vertex shader source contains interior null byte");
    let frag_cstr = CString::new(bench_render::FRAGMENT_SRC)
        .expect("fragment shader source contains interior null byte");
    let program = match bench_render::compile_program(&vert_cstr, &frag_cstr) {
        Ok(p) => p,
        Err(e) => {
            println!("  SKIP: shader program compilation failed: {e}");
            println!();
            return;
        }
    };

    let tex_uniform = unsafe {
        let name = CString::new("tex").unwrap();
        gls::gl::GetUniformLocation(program, name.as_ptr())
    };
    let (vao, vbo, ebo) = bench_render::create_quad_vao();
    let fourcc = bench_render::rgba_fourcc();

    // --- Test 1: Solid fill (640x640 → 640x640, same size) ---
    {
        let (w, h) = (640u32, 640u32);
        let bytes = (w * h * 4) as usize;

        let src = Tensor::<u8>::new(&[bytes], Some(TensorMemory::Dma), None);
        let dst = Tensor::<u8>::new(&[bytes], Some(TensorMemory::Dma), None);

        match (src, dst) {
            (Ok(src), Ok(dst)) => {
                fill_solid(&src, [0xFF, 0x00, 0x00, 0xFF]);

                let src_fd = src.clone_fd().unwrap();
                let dst_fd = dst.clone_fd().unwrap();

                run_full_pipeline(
                    ctx,
                    src_fd.as_raw_fd(),
                    w,
                    h,
                    dst_fd.as_raw_fd(),
                    w,
                    h,
                    fourcc,
                    program,
                    tex_uniform,
                    vao,
                );

                let pass = verify_pixels(&dst, [0xFF, 0x00, 0x00, 0xFF], 2);
                println!(
                    "  {:40} {}",
                    "pipeline_verify/solid_640x640",
                    if pass { "PASS" } else { "FAIL" }
                );
            }
            _ => {
                println!(
                    "  {:40} SKIP (DMA allocation failed)",
                    "pipeline_verify/solid_640x640"
                );
            }
        }
    }

    // --- Test 2: Gradient resize (1920x1080 → 640x640) ---
    {
        let (src_w, src_h) = (1920u32, 1080u32);
        let (dst_w, dst_h) = (640u32, 640u32);
        let src_bytes = (src_w * src_h * 4) as usize;
        let dst_bytes = (dst_w * dst_h * 4) as usize;

        let src = Tensor::<u8>::new(&[src_bytes], Some(TensorMemory::Dma), None);
        let dst = Tensor::<u8>::new(&[dst_bytes], Some(TensorMemory::Dma), None);

        match (src, dst) {
            (Ok(src), Ok(dst)) => {
                fill_gradient(&src, src_w, src_h);

                let src_fd = src.clone_fd().unwrap();
                let dst_fd = dst.clone_fd().unwrap();

                run_full_pipeline(
                    ctx,
                    src_fd.as_raw_fd(),
                    src_w,
                    src_h,
                    dst_fd.as_raw_fd(),
                    dst_w,
                    dst_h,
                    fourcc,
                    program,
                    tex_uniform,
                    vao,
                );

                let pass = verify_not_zero(&dst);
                println!(
                    "  {:40} {}",
                    "pipeline_verify/gradient_1080p_to_640",
                    if pass { "PASS" } else { "FAIL" }
                );
            }
            _ => {
                println!(
                    "  {:40} SKIP (DMA allocation failed)",
                    "pipeline_verify/gradient_1080p_to_640"
                );
            }
        }
    }

    // Cleanup shared resources
    unsafe {
        gls::gl::DeleteVertexArrays(1, &vao);
        gls::gl::DeleteBuffers(1, &vbo);
        gls::gl::DeleteBuffers(1, &ebo);
        gls::gl::DeleteProgram(program);
    }

    println!();
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

/// Run DMA-buf pipeline benchmarks and return collected results.
pub fn run(ctx: &GpuContext) -> Vec<BenchResult> {
    println!("== Benchmark: DMA-buf Pipeline ==");

    let mut results = Vec::new();

    let vert_cstr = CString::new(bench_render::VERTEX_SRC)
        .expect("vertex shader source contains interior null byte");
    let frag_cstr = CString::new(bench_render::FRAGMENT_SRC)
        .expect("fragment shader source contains interior null byte");
    let program = match bench_render::compile_program(&vert_cstr, &frag_cstr) {
        Ok(p) => p,
        Err(e) => {
            println!("  SKIP: shader program compilation failed: {e}");
            println!();
            return results;
        }
    };

    let tex_uniform = unsafe {
        let name = CString::new("tex").unwrap();
        gls::gl::GetUniformLocation(program, name.as_ptr())
    };
    let (vao, vbo, ebo) = bench_render::create_quad_vao();
    let fourcc = bench_render::rgba_fourcc();

    let configs: &[(u32, u32, u32, u32, &str)] = &[
        (1920, 1080, 640, 640, "1080p_to_640"),
        (1920, 1080, 320, 320, "1080p_to_320"),
        (3840, 2160, 640, 640, "4k_to_640"),
    ];

    for &(src_w, src_h, dst_w, dst_h, label) in configs {
        let src_pitch = (src_w * 4) as i32;
        let dst_pitch = (dst_w * 4) as i32;
        let src_bytes = (src_w * src_h * 4) as usize;
        let dst_bytes = (dst_w * dst_h * 4) as usize;

        let src_tensor = match Tensor::<u8>::new(&[src_bytes], Some(TensorMemory::Dma), None) {
            Ok(t) => t,
            Err(e) => {
                println!("  SKIP {label}: src DMA allocation failed: {e}");
                continue;
            }
        };
        let dst_tensor = match Tensor::<u8>::new(&[dst_bytes], Some(TensorMemory::Dma), None) {
            Ok(t) => t,
            Err(e) => {
                println!("  SKIP {label}: dst DMA allocation failed: {e}");
                continue;
            }
        };

        let src_fd_owned = src_tensor.clone_fd().unwrap();
        let dst_fd_owned = dst_tensor.clone_fd().unwrap();
        let src_fd = src_fd_owned.as_raw_fd();
        let dst_fd = dst_fd_owned.as_raw_fd();

        // -----------------------------------------------------------
        // Variant 1: pipeline_per_frame — full resource create/destroy
        // -----------------------------------------------------------
        {
            let name = format!("pipeline_per_frame/{label}");
            let r = run_bench(&name, 10, 200, || {
                let src_img = ctx
                    .create_egl_image_dma(src_fd, src_w as i32, src_h as i32, fourcc, src_pitch)
                    .expect("src create_egl_image_dma failed");
                let dst_img = ctx
                    .create_egl_image_dma(dst_fd, dst_w as i32, dst_h as i32, fourcc, dst_pitch)
                    .expect("dst create_egl_image_dma failed");

                unsafe {
                    let mut src_tex = 0u32;
                    gls::gl::GenTextures(1, &mut src_tex);
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
                    gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, src_img.as_ptr());
                    gls::gl::TexParameteri(
                        gls::gl::TEXTURE_2D,
                        gls::gl::TEXTURE_MIN_FILTER,
                        gls::gl::LINEAR as i32,
                    );
                    gls::gl::TexParameteri(
                        gls::gl::TEXTURE_2D,
                        gls::gl::TEXTURE_MAG_FILTER,
                        gls::gl::LINEAR as i32,
                    );

                    let mut dst_tex = 0u32;
                    gls::gl::GenTextures(1, &mut dst_tex);
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, dst_tex);
                    gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, dst_img.as_ptr());

                    let mut fbo = 0u32;
                    gls::gl::GenFramebuffers(1, &mut fbo);
                    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
                    gls::gl::FramebufferTexture2D(
                        gls::gl::FRAMEBUFFER,
                        gls::gl::COLOR_ATTACHMENT0,
                        gls::gl::TEXTURE_2D,
                        dst_tex,
                        0,
                    );

                    gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
                    gls::gl::UseProgram(program);
                    gls::gl::ActiveTexture(gls::gl::TEXTURE0);
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
                    gls::gl::Uniform1i(tex_uniform, 0);

                    gls::gl::BindVertexArray(vao);
                    gls::gl::DrawElements(gls::gl::TRIANGLES, 6, gls::gl::UNSIGNED_INT, null());
                    gls::gl::Finish();

                    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
                    gls::gl::BindVertexArray(0);
                    gls::gl::DeleteFramebuffers(1, &fbo);
                    gls::gl::DeleteTextures(1, &dst_tex);
                    gls::gl::DeleteTextures(1, &src_tex);
                }

                ctx.destroy_egl_image(src_img)
                    .expect("destroy src EGLImage failed");
                ctx.destroy_egl_image(dst_img)
                    .expect("destroy dst EGLImage failed");
            });
            r.print_summary();
            results.push(r);
        }

        // -----------------------------------------------------------
        // Variant 2: pipeline_cached — only bind + render + glFinish
        // -----------------------------------------------------------
        {
            let src_img = ctx
                .create_egl_image_dma(src_fd, src_w as i32, src_h as i32, fourcc, src_pitch)
                .expect("src create_egl_image_dma failed");
            let dst_img = ctx
                .create_egl_image_dma(dst_fd, dst_w as i32, dst_h as i32, fourcc, dst_pitch)
                .expect("dst create_egl_image_dma failed");

            let (src_tex, dst_tex, fbo) = unsafe {
                let mut src_tex = 0u32;
                gls::gl::GenTextures(1, &mut src_tex);
                gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
                gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, src_img.as_ptr());
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_MIN_FILTER,
                    gls::gl::LINEAR as i32,
                );
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_MAG_FILTER,
                    gls::gl::LINEAR as i32,
                );

                let mut dst_tex = 0u32;
                gls::gl::GenTextures(1, &mut dst_tex);
                gls::gl::BindTexture(gls::gl::TEXTURE_2D, dst_tex);
                gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, dst_img.as_ptr());

                let mut fbo = 0u32;
                gls::gl::GenFramebuffers(1, &mut fbo);
                gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
                gls::gl::FramebufferTexture2D(
                    gls::gl::FRAMEBUFFER,
                    gls::gl::COLOR_ATTACHMENT0,
                    gls::gl::TEXTURE_2D,
                    dst_tex,
                    0,
                );

                (src_tex, dst_tex, fbo)
            };

            let name = format!("pipeline_cached/{label}");
            let r = run_bench(&name, 10, 200, || unsafe {
                gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
                gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);

                gls::gl::UseProgram(program);
                gls::gl::ActiveTexture(gls::gl::TEXTURE0);
                gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
                gls::gl::Uniform1i(tex_uniform, 0);

                gls::gl::BindVertexArray(vao);
                gls::gl::DrawElements(gls::gl::TRIANGLES, 6, gls::gl::UNSIGNED_INT, null());
                gls::gl::Finish();
            });
            r.print_summary();
            results.push(r);

            unsafe {
                gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
                gls::gl::BindVertexArray(0);
                gls::gl::DeleteFramebuffers(1, &fbo);
                gls::gl::DeleteTextures(1, &dst_tex);
                gls::gl::DeleteTextures(1, &src_tex);
            }
            ctx.destroy_egl_image(dst_img)
                .expect("destroy dst EGLImage failed");
            ctx.destroy_egl_image(src_img)
                .expect("destroy src EGLImage failed");
        }

        // -----------------------------------------------------------
        // Variant 3: pipeline_rebind_src — only src EGLImage recreated
        // -----------------------------------------------------------
        {
            // Pre-create dst side (stays cached)
            let dst_img = ctx
                .create_egl_image_dma(dst_fd, dst_w as i32, dst_h as i32, fourcc, dst_pitch)
                .expect("dst create_egl_image_dma failed");

            let (src_tex, dst_tex, fbo) = unsafe {
                let mut src_tex = 0u32;
                gls::gl::GenTextures(1, &mut src_tex);

                let mut dst_tex = 0u32;
                gls::gl::GenTextures(1, &mut dst_tex);
                gls::gl::BindTexture(gls::gl::TEXTURE_2D, dst_tex);
                gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, dst_img.as_ptr());

                let mut fbo = 0u32;
                gls::gl::GenFramebuffers(1, &mut fbo);
                gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
                gls::gl::FramebufferTexture2D(
                    gls::gl::FRAMEBUFFER,
                    gls::gl::COLOR_ATTACHMENT0,
                    gls::gl::TEXTURE_2D,
                    dst_tex,
                    0,
                );

                (src_tex, dst_tex, fbo)
            };

            let name = format!("pipeline_rebind_src/{label}");
            let r = run_bench(&name, 10, 200, || {
                // Create fresh src EGLImage and rebind on existing texture
                let src_img = ctx
                    .create_egl_image_dma(src_fd, src_w as i32, src_h as i32, fourcc, src_pitch)
                    .expect("src create_egl_image_dma failed");

                unsafe {
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
                    gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, src_img.as_ptr());
                    gls::gl::TexParameteri(
                        gls::gl::TEXTURE_2D,
                        gls::gl::TEXTURE_MIN_FILTER,
                        gls::gl::LINEAR as i32,
                    );
                    gls::gl::TexParameteri(
                        gls::gl::TEXTURE_2D,
                        gls::gl::TEXTURE_MAG_FILTER,
                        gls::gl::LINEAR as i32,
                    );

                    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
                    gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
                    gls::gl::UseProgram(program);
                    gls::gl::ActiveTexture(gls::gl::TEXTURE0);
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
                    gls::gl::Uniform1i(tex_uniform, 0);

                    gls::gl::BindVertexArray(vao);
                    gls::gl::DrawElements(gls::gl::TRIANGLES, 6, gls::gl::UNSIGNED_INT, null());
                    gls::gl::Finish();
                }

                ctx.destroy_egl_image(src_img)
                    .expect("destroy src EGLImage failed");
            });
            r.print_summary();
            results.push(r);

            unsafe {
                gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
                gls::gl::BindVertexArray(0);
                gls::gl::DeleteFramebuffers(1, &fbo);
                gls::gl::DeleteTextures(1, &dst_tex);
                gls::gl::DeleteTextures(1, &src_tex);
            }
            ctx.destroy_egl_image(dst_img)
                .expect("destroy dst EGLImage failed");
        }

        // -----------------------------------------------------------
        // Variant 4: pipeline_rebind_dst — only dst EGLImage recreated
        // -----------------------------------------------------------
        {
            // Pre-create src side (stays cached)
            let src_img = ctx
                .create_egl_image_dma(src_fd, src_w as i32, src_h as i32, fourcc, src_pitch)
                .expect("src create_egl_image_dma failed");

            let (src_tex, dst_tex, fbo) = unsafe {
                let mut src_tex = 0u32;
                gls::gl::GenTextures(1, &mut src_tex);
                gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
                gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, src_img.as_ptr());
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_MIN_FILTER,
                    gls::gl::LINEAR as i32,
                );
                gls::gl::TexParameteri(
                    gls::gl::TEXTURE_2D,
                    gls::gl::TEXTURE_MAG_FILTER,
                    gls::gl::LINEAR as i32,
                );

                let mut dst_tex = 0u32;
                gls::gl::GenTextures(1, &mut dst_tex);

                let mut fbo = 0u32;
                gls::gl::GenFramebuffers(1, &mut fbo);

                (src_tex, dst_tex, fbo)
            };

            let name = format!("pipeline_rebind_dst/{label}");
            let r = run_bench(&name, 10, 200, || {
                // Create fresh dst EGLImage and rebind + re-attach FBO
                let dst_img = ctx
                    .create_egl_image_dma(dst_fd, dst_w as i32, dst_h as i32, fourcc, dst_pitch)
                    .expect("dst create_egl_image_dma failed");

                unsafe {
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, dst_tex);
                    gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, dst_img.as_ptr());

                    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
                    gls::gl::FramebufferTexture2D(
                        gls::gl::FRAMEBUFFER,
                        gls::gl::COLOR_ATTACHMENT0,
                        gls::gl::TEXTURE_2D,
                        dst_tex,
                        0,
                    );
                    gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);

                    gls::gl::UseProgram(program);
                    gls::gl::ActiveTexture(gls::gl::TEXTURE0);
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
                    gls::gl::Uniform1i(tex_uniform, 0);

                    gls::gl::BindVertexArray(vao);
                    gls::gl::DrawElements(gls::gl::TRIANGLES, 6, gls::gl::UNSIGNED_INT, null());
                    gls::gl::Finish();
                }

                ctx.destroy_egl_image(dst_img)
                    .expect("destroy dst EGLImage failed");
            });
            r.print_summary();
            results.push(r);

            unsafe {
                gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
                gls::gl::BindVertexArray(0);
                gls::gl::DeleteFramebuffers(1, &fbo);
                gls::gl::DeleteTextures(1, &dst_tex);
                gls::gl::DeleteTextures(1, &src_tex);
            }
            ctx.destroy_egl_image(src_img)
                .expect("destroy src EGLImage failed");
        }

        // -----------------------------------------------------------
        // Variant 5: pipeline_rebind_both — both EGLImages recreated
        // -----------------------------------------------------------
        {
            let (src_tex, dst_tex, fbo) = unsafe {
                let mut src_tex = 0u32;
                gls::gl::GenTextures(1, &mut src_tex);

                let mut dst_tex = 0u32;
                gls::gl::GenTextures(1, &mut dst_tex);

                let mut fbo = 0u32;
                gls::gl::GenFramebuffers(1, &mut fbo);

                (src_tex, dst_tex, fbo)
            };

            let name = format!("pipeline_rebind_both/{label}");
            let r = run_bench(&name, 10, 200, || {
                let src_img = ctx
                    .create_egl_image_dma(src_fd, src_w as i32, src_h as i32, fourcc, src_pitch)
                    .expect("src create_egl_image_dma failed");
                let dst_img = ctx
                    .create_egl_image_dma(dst_fd, dst_w as i32, dst_h as i32, fourcc, dst_pitch)
                    .expect("dst create_egl_image_dma failed");

                unsafe {
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
                    gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, src_img.as_ptr());
                    gls::gl::TexParameteri(
                        gls::gl::TEXTURE_2D,
                        gls::gl::TEXTURE_MIN_FILTER,
                        gls::gl::LINEAR as i32,
                    );
                    gls::gl::TexParameteri(
                        gls::gl::TEXTURE_2D,
                        gls::gl::TEXTURE_MAG_FILTER,
                        gls::gl::LINEAR as i32,
                    );

                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, dst_tex);
                    gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, dst_img.as_ptr());

                    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
                    gls::gl::FramebufferTexture2D(
                        gls::gl::FRAMEBUFFER,
                        gls::gl::COLOR_ATTACHMENT0,
                        gls::gl::TEXTURE_2D,
                        dst_tex,
                        0,
                    );
                    gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);

                    gls::gl::UseProgram(program);
                    gls::gl::ActiveTexture(gls::gl::TEXTURE0);
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
                    gls::gl::Uniform1i(tex_uniform, 0);

                    gls::gl::BindVertexArray(vao);
                    gls::gl::DrawElements(gls::gl::TRIANGLES, 6, gls::gl::UNSIGNED_INT, null());
                    gls::gl::Finish();
                }

                ctx.destroy_egl_image(src_img)
                    .expect("destroy src EGLImage failed");
                ctx.destroy_egl_image(dst_img)
                    .expect("destroy dst EGLImage failed");
            });
            r.print_summary();
            results.push(r);

            unsafe {
                gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
                gls::gl::BindVertexArray(0);
                gls::gl::DeleteFramebuffers(1, &fbo);
                gls::gl::DeleteTextures(1, &dst_tex);
                gls::gl::DeleteTextures(1, &src_tex);
            }
        }

        // -----------------------------------------------------------
        // Variant 6: pipeline_gradient — per-frame with gradient src
        // -----------------------------------------------------------
        {
            // Fill source with gradient pattern once before benchmarking
            fill_gradient(&src_tensor, src_w, src_h);

            let name = format!("pipeline_gradient/{label}");
            let r = run_bench(&name, 10, 200, || {
                let src_img = ctx
                    .create_egl_image_dma(src_fd, src_w as i32, src_h as i32, fourcc, src_pitch)
                    .expect("src create_egl_image_dma failed");
                let dst_img = ctx
                    .create_egl_image_dma(dst_fd, dst_w as i32, dst_h as i32, fourcc, dst_pitch)
                    .expect("dst create_egl_image_dma failed");

                unsafe {
                    let mut src_tex = 0u32;
                    gls::gl::GenTextures(1, &mut src_tex);
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
                    gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, src_img.as_ptr());
                    gls::gl::TexParameteri(
                        gls::gl::TEXTURE_2D,
                        gls::gl::TEXTURE_MIN_FILTER,
                        gls::gl::LINEAR as i32,
                    );
                    gls::gl::TexParameteri(
                        gls::gl::TEXTURE_2D,
                        gls::gl::TEXTURE_MAG_FILTER,
                        gls::gl::LINEAR as i32,
                    );

                    let mut dst_tex = 0u32;
                    gls::gl::GenTextures(1, &mut dst_tex);
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, dst_tex);
                    gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, dst_img.as_ptr());

                    let mut fbo = 0u32;
                    gls::gl::GenFramebuffers(1, &mut fbo);
                    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
                    gls::gl::FramebufferTexture2D(
                        gls::gl::FRAMEBUFFER,
                        gls::gl::COLOR_ATTACHMENT0,
                        gls::gl::TEXTURE_2D,
                        dst_tex,
                        0,
                    );

                    gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
                    gls::gl::UseProgram(program);
                    gls::gl::ActiveTexture(gls::gl::TEXTURE0);
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
                    gls::gl::Uniform1i(tex_uniform, 0);

                    gls::gl::BindVertexArray(vao);
                    gls::gl::DrawElements(gls::gl::TRIANGLES, 6, gls::gl::UNSIGNED_INT, null());
                    gls::gl::Finish();

                    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
                    gls::gl::BindVertexArray(0);
                    gls::gl::DeleteFramebuffers(1, &fbo);
                    gls::gl::DeleteTextures(1, &dst_tex);
                    gls::gl::DeleteTextures(1, &src_tex);
                }

                ctx.destroy_egl_image(src_img)
                    .expect("destroy src EGLImage failed");
                ctx.destroy_egl_image(dst_img)
                    .expect("destroy dst EGLImage failed");
            });
            r.print_summary();
            results.push(r);
        }
    }

    // Cleanup shared resources
    unsafe {
        gls::gl::DeleteVertexArrays(1, &vao);
        gls::gl::DeleteBuffers(1, &vbo);
        gls::gl::DeleteBuffers(1, &ebo);
        gls::gl::DeleteProgram(program);
    }

    println!();
    results
}
