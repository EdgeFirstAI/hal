// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Direct RGB render verification and benchmarks.
//!
//! Renders an RGBA DMA-buf source through a fullscreen quad into an RGB888
//! (BGR888 fourcc) DMA-buf renderbuffer, bypassing the texture-based FBO
//! attachment used elsewhere. This exercises the `GL_OES_EGL_image` →
//! renderbuffer path and measures EGLImage + RBO + FBO creation cost for the
//! direct RGB output pipeline.

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
fn fill_solid_rgba(tensor: &Tensor<u8>, pixel: [u8; 4]) {
    let mut map = tensor
        .map()
        .expect("failed to map tensor for fill_solid_rgba");
    let slice = map.as_mut_slice();
    for chunk in slice.chunks_exact_mut(4) {
        chunk.copy_from_slice(&pixel);
    }
}

/// Verify every 3-byte RGB pixel in a DMA tensor matches `expected` within
/// per-channel `tolerance`. Prints the first mismatch on failure.
fn verify_rgb_pixels(tensor: &Tensor<u8>, expected: [u8; 3], tolerance: u8) -> bool {
    let map = tensor
        .map()
        .expect("failed to map tensor for verify_rgb_pixels");
    let slice = map.as_slice();
    for (i, chunk) in slice.chunks_exact(3).enumerate() {
        for c in 0..3 {
            let diff = (chunk[c] as i16 - expected[c] as i16).unsigned_abs() as u8;
            if diff > tolerance {
                println!(
                    "    MISMATCH at pixel {i}: got [{}, {}, {}] expected [{}, {}, {}] (tolerance {tolerance})",
                    chunk[0], chunk[1], chunk[2],
                    expected[0], expected[1], expected[2],
                );
                return false;
            }
        }
    }
    true
}

/// Create a renderbuffer + FBO backed by a DMA-buf EGLImage.
///
/// Returns `(rbo, fbo, egl_image)` on success. The FBO is left unbound.
///
/// # Safety
///
/// Caller must ensure a current GL context and valid DMA fd/dimensions.
unsafe fn create_rgb_rbo_fbo(
    ctx: &GpuContext,
    fd: i32,
    w: u32,
    h: u32,
    fourcc: u32,
    pitch: i32,
) -> Result<(u32, u32, khronos_egl::Image), String> {
    let egl_image = ctx.create_egl_image_dma(fd, w as i32, h as i32, fourcc, pitch)?;

    let mut rbo = 0u32;
    gls::gl::GenRenderbuffers(1, &mut rbo);
    gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, rbo);
    gls::gl::EGLImageTargetRenderbufferStorageOES(gls::gl::RENDERBUFFER, egl_image.as_ptr());

    let mut fbo = 0u32;
    gls::gl::GenFramebuffers(1, &mut fbo);
    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
    gls::gl::FramebufferRenderbuffer(
        gls::gl::FRAMEBUFFER,
        gls::gl::COLOR_ATTACHMENT0,
        gls::gl::RENDERBUFFER,
        rbo,
    );

    let status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
    if status != gls::gl::FRAMEBUFFER_COMPLETE {
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
        gls::gl::DeleteFramebuffers(1, &fbo);
        gls::gl::DeleteRenderbuffers(1, &rbo);
        ctx.destroy_egl_image(egl_image)
            .expect("destroy EGLImage failed during cleanup");
        return Err(format!("FBO incomplete: status 0x{status:X}"));
    }

    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);

    Ok((rbo, fbo, egl_image))
}

/// Destroy a renderbuffer + FBO + EGLImage created by [`create_rgb_rbo_fbo`].
///
/// # Safety
///
/// Caller must ensure the resources are valid and a current GL context.
unsafe fn destroy_rgb_rbo_fbo(ctx: &GpuContext, rbo: u32, fbo: u32, egl_image: khronos_egl::Image) {
    gls::gl::DeleteFramebuffers(1, &fbo);
    gls::gl::DeleteRenderbuffers(1, &rbo);
    ctx.destroy_egl_image(egl_image)
        .expect("destroy EGLImage failed");
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------

/// Run direct-RGB-render verification. Prints PASS/FAIL and returns success.
pub fn run_verify(ctx: &GpuContext) -> bool {
    println!("== Verification: Direct RGB Render ==");

    let vert_cstr = CString::new(bench_render::VERTEX_SRC)
        .expect("vertex shader source contains interior null byte");
    let frag_cstr = CString::new(bench_render::FRAGMENT_SRC)
        .expect("fragment shader source contains interior null byte");
    let program = match bench_render::compile_program(&vert_cstr, &frag_cstr) {
        Ok(p) => p,
        Err(e) => {
            println!("  SKIP: shader program compilation failed: {e}");
            println!();
            return false;
        }
    };

    let tex_uniform = unsafe {
        let name = CString::new("tex").unwrap();
        gls::gl::GetUniformLocation(program, name.as_ptr())
    };
    let (vao, vbo, ebo) = bench_render::create_quad_vao();
    let rgba_fourcc = bench_render::rgba_fourcc();
    let rgb_fourcc = bench_render::rgb888_fourcc();

    let mut pass = false;

    // --- Solid red RGBA source → RGB888 destination (640x640) ---
    {
        let (w, h) = (640u32, 640u32);
        let src_bytes = (w * h * 4) as usize;
        let dst_bytes = (w * h * 3) as usize;

        let src = Tensor::<u8>::new(&[src_bytes], Some(TensorMemory::Dma), None);
        let dst = Tensor::<u8>::new(&[dst_bytes], Some(TensorMemory::Dma), None);

        match (src, dst) {
            (Ok(src), Ok(dst)) => {
                fill_solid_rgba(&src, [0xFF, 0x00, 0x00, 0xFF]);

                let src_fd_owned = src.clone_fd().unwrap();
                let dst_fd_owned = dst.clone_fd().unwrap();
                let src_fd = src_fd_owned.as_raw_fd();
                let dst_fd = dst_fd_owned.as_raw_fd();
                let src_pitch = (w * 4) as i32;
                let dst_pitch = (w * 3) as i32;

                // Import source as RGBA EGLImage → texture
                let src_img = match ctx.create_egl_image_dma(
                    src_fd,
                    w as i32,
                    h as i32,
                    rgba_fourcc,
                    src_pitch,
                ) {
                    Ok(img) => img,
                    Err(e) => {
                        println!(
                            "  {:40} SKIP (src EGLImage failed: {e})",
                            "rgb_direct_verify/solid_640x640"
                        );
                        unsafe {
                            gls::gl::DeleteVertexArrays(1, &vao);
                            gls::gl::DeleteBuffers(1, &vbo);
                            gls::gl::DeleteBuffers(1, &ebo);
                            gls::gl::DeleteProgram(program);
                        }
                        println!();
                        return false;
                    }
                };

                let src_tex = unsafe {
                    let mut tex = 0u32;
                    gls::gl::GenTextures(1, &mut tex);
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, tex);
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
                    tex
                };

                // Import destination as BGR888 EGLImage → renderbuffer → FBO
                match unsafe { create_rgb_rbo_fbo(ctx, dst_fd, w, h, rgb_fourcc, dst_pitch) } {
                    Ok((rbo, fbo, dst_img)) => {
                        // Render fullscreen quad
                        unsafe {
                            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
                            gls::gl::Viewport(0, 0, w as i32, h as i32);

                            gls::gl::UseProgram(program);
                            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
                            gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
                            gls::gl::Uniform1i(tex_uniform, 0);

                            gls::gl::BindVertexArray(vao);
                            gls::gl::DrawElements(
                                gls::gl::TRIANGLES,
                                6,
                                gls::gl::UNSIGNED_INT,
                                null(),
                            );
                            gls::gl::Finish();

                            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
                            gls::gl::BindVertexArray(0);
                        }

                        // Verify destination pixels
                        pass = verify_rgb_pixels(&dst, [0xFF, 0x00, 0x00], 2);

                        // Cleanup destination resources
                        unsafe {
                            destroy_rgb_rbo_fbo(ctx, rbo, fbo, dst_img);
                        }
                    }
                    Err(e) => {
                        println!(
                            "  {:40} SKIP (dst RBO/FBO failed: {e})",
                            "rgb_direct_verify/solid_640x640"
                        );
                    }
                }

                // Cleanup source resources
                unsafe {
                    gls::gl::DeleteTextures(1, &src_tex);
                }
                ctx.destroy_egl_image(src_img)
                    .expect("destroy src EGLImage failed");

                println!(
                    "  {:40} {}",
                    "rgb_direct_verify/solid_640x640",
                    if pass { "PASS" } else { "FAIL" }
                );
            }
            _ => {
                println!(
                    "  {:40} SKIP (DMA allocation failed)",
                    "rgb_direct_verify/solid_640x640"
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
    pass
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

/// Run direct-RGB-render benchmarks and return collected results.
pub fn run(ctx: &GpuContext) -> Vec<BenchResult> {
    println!("== Benchmark: Direct RGB Render ==");

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
    let rgba_fourcc = bench_render::rgba_fourcc();
    let rgb_fourcc = bench_render::rgb888_fourcc();

    let configs: &[(u32, u32, u32, u32, &str)] = &[
        (1920, 1080, 640, 640, "1080p_to_640"),
        (1920, 1080, 320, 320, "1080p_to_320"),
        (3840, 2160, 640, 640, "4k_to_640"),
    ];

    for &(src_w, src_h, dst_w, dst_h, label) in configs {
        let src_pitch = (src_w * 4) as i32;
        let dst_pitch = (dst_w * 3) as i32;
        let src_bytes = (src_w * src_h * 4) as usize;
        let dst_bytes = (dst_w * dst_h * 3) as usize;

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
        // Variant 1: rgb_direct_per_frame — full resource create/destroy
        // -----------------------------------------------------------
        {
            let name = format!("rgb_direct_per_frame/{label}");
            let r = run_bench(&name, 10, 200, || {
                // Create source EGLImage + texture
                let src_img = ctx
                    .create_egl_image_dma(
                        src_fd,
                        src_w as i32,
                        src_h as i32,
                        rgba_fourcc,
                        src_pitch,
                    )
                    .expect("src create_egl_image_dma failed");

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

                    // Create destination RBO + FBO from RGB EGLImage
                    let (rbo, fbo, dst_img) =
                        create_rgb_rbo_fbo(ctx, dst_fd, dst_w, dst_h, rgb_fourcc, dst_pitch)
                            .expect("dst create_rgb_rbo_fbo failed");

                    // Render
                    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
                    gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
                    gls::gl::UseProgram(program);
                    gls::gl::ActiveTexture(gls::gl::TEXTURE0);
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
                    gls::gl::Uniform1i(tex_uniform, 0);

                    gls::gl::BindVertexArray(vao);
                    gls::gl::DrawElements(gls::gl::TRIANGLES, 6, gls::gl::UNSIGNED_INT, null());
                    gls::gl::Finish();

                    // Cleanup
                    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
                    gls::gl::BindVertexArray(0);
                    gls::gl::DeleteTextures(1, &src_tex);
                    destroy_rgb_rbo_fbo(ctx, rbo, fbo, dst_img);
                }

                ctx.destroy_egl_image(src_img)
                    .expect("destroy src EGLImage failed");
            });
            r.print_summary();
            results.push(r);
        }

        // -----------------------------------------------------------
        // Variant 2: rgb_direct_cached — pre-created resources, render only
        // -----------------------------------------------------------
        {
            let src_img = ctx
                .create_egl_image_dma(src_fd, src_w as i32, src_h as i32, rgba_fourcc, src_pitch)
                .expect("src create_egl_image_dma failed");

            let src_tex = unsafe {
                let mut tex = 0u32;
                gls::gl::GenTextures(1, &mut tex);
                gls::gl::BindTexture(gls::gl::TEXTURE_2D, tex);
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
                tex
            };

            let (rbo, fbo, dst_img) =
                unsafe { create_rgb_rbo_fbo(ctx, dst_fd, dst_w, dst_h, rgb_fourcc, dst_pitch) }
                    .expect("dst create_rgb_rbo_fbo failed");

            let name = format!("rgb_direct_cached/{label}");
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

            // Cleanup cached resources
            unsafe {
                gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
                gls::gl::BindVertexArray(0);
                gls::gl::DeleteTextures(1, &src_tex);
                destroy_rgb_rbo_fbo(ctx, rbo, fbo, dst_img);
            }
            ctx.destroy_egl_image(src_img)
                .expect("destroy src EGLImage failed");
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
