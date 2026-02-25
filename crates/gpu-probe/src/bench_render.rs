// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Full render pipeline benchmarks — measures the complete GL render path
//! including EGLImage creation, texture binding, FBO setup, and draw calls.
//!
//! This is the headline benchmark: it replicates what the HAL's `convert()`
//! does at the raw GL level, comparing per-call resource creation versus
//! cached resources. The delta between the two IS the EGLImage + FBO creation
//! overhead we want to measure.

use crate::bench::{run_bench, BenchResult};
use crate::egl_context::GpuContext;
use edgefirst_tensor::{Tensor, TensorMemory, TensorTrait};
use std::ffi::CString;
use std::os::unix::io::AsRawFd;
use std::ptr::null;

/// Vertex shader source (simple passthrough with texture coordinates).
pub(crate) const VERTEX_SRC: &str = "\
#version 300 es
precision mediump float;
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 texCoord;
out vec2 tc;
void main() { tc = texCoord; gl_Position = vec4(pos, 1.0); }
";

/// Fragment shader source (simple texture sampling).
pub(crate) const FRAGMENT_SRC: &str = "\
#version 300 es
precision mediump float;
uniform sampler2D tex;
in vec2 tc;
out vec4 color;
void main() { color = texture(tex, tc); }
";

/// Interleaved vertex data for a fullscreen quad: [x, y, u, v] per vertex.
/// Positions: (-1,-1), (1,-1), (1,1), (-1,1)
/// UVs:       (0,0),   (1,0),  (1,1), (0,1)
#[rustfmt::skip]
const QUAD_VERTICES: [f32; 16] = [
    -1.0, -1.0,  0.0, 0.0,
     1.0, -1.0,  1.0, 0.0,
     1.0,  1.0,  1.0, 1.0,
    -1.0,  1.0,  0.0, 1.0,
];

/// Triangle indices for two triangles forming the quad.
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

/// DRM FourCC for RGBA (AB24 / ABGR8888).
pub(crate) fn rgba_fourcc() -> u32 {
    gbm::drm::buffer::DrmFourcc::Abgr8888 as u32
}

/// Compile a shader program from vertex + fragment source strings.
/// Returns the program ID.
pub(crate) fn compile_program(vert: &CString, frag: &CString) -> u32 {
    unsafe {
        let program = gls::gl::CreateProgram();

        let vs = gls::gl::CreateShader(gls::gl::VERTEX_SHADER);
        let vs_ptr = vert.as_ptr();
        gls::gl::ShaderSource(vs, 1, &raw const vs_ptr, null());
        gls::gl::CompileShader(vs);
        gls::gl::AttachShader(program, vs);

        let fs = gls::gl::CreateShader(gls::gl::FRAGMENT_SHADER);
        let fs_ptr = frag.as_ptr();
        gls::gl::ShaderSource(fs, 1, &raw const fs_ptr, null());
        gls::gl::CompileShader(fs);
        gls::gl::AttachShader(program, fs);

        gls::gl::LinkProgram(program);

        // Detach shaders so they can be deleted while program remains
        gls::gl::DetachShader(program, vs);
        gls::gl::DetachShader(program, fs);
        gls::gl::DeleteShader(vs);
        gls::gl::DeleteShader(fs);

        program
    }
}

/// Create a VAO with interleaved vertex + UV data and an element buffer.
/// Returns (vao, vbo, ebo).
pub(crate) fn create_quad_vao() -> (u32, u32, u32) {
    unsafe {
        let mut vao = 0u32;
        gls::gl::GenVertexArrays(1, &mut vao);
        gls::gl::BindVertexArray(vao);

        // VBO: interleaved [x, y, u, v]
        let mut vbo = 0u32;
        gls::gl::GenBuffers(1, &mut vbo);
        gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, vbo);
        gls::gl::BufferData(
            gls::gl::ARRAY_BUFFER,
            std::mem::size_of_val(&QUAD_VERTICES) as isize,
            QUAD_VERTICES.as_ptr() as *const std::ffi::c_void,
            gls::gl::STATIC_DRAW,
        );

        // Attribute 0: position (vec2, declared as vec3 in shader — z defaults to 0)
        let stride = (4 * std::mem::size_of::<f32>()) as i32;
        gls::gl::VertexAttribPointer(0, 2, gls::gl::FLOAT, gls::gl::FALSE, stride, null());
        gls::gl::EnableVertexAttribArray(0);

        // Attribute 1: texCoord (vec2)
        gls::gl::VertexAttribPointer(
            1,
            2,
            gls::gl::FLOAT,
            gls::gl::FALSE,
            stride,
            (2 * std::mem::size_of::<f32>()) as *const std::ffi::c_void,
        );
        gls::gl::EnableVertexAttribArray(1);

        // EBO: indices
        let mut ebo = 0u32;
        gls::gl::GenBuffers(1, &mut ebo);
        gls::gl::BindBuffer(gls::gl::ELEMENT_ARRAY_BUFFER, ebo);
        gls::gl::BufferData(
            gls::gl::ELEMENT_ARRAY_BUFFER,
            std::mem::size_of_val(&QUAD_INDICES) as isize,
            QUAD_INDICES.as_ptr() as *const std::ffi::c_void,
            gls::gl::STATIC_DRAW,
        );

        gls::gl::BindVertexArray(0);

        (vao, vbo, ebo)
    }
}

/// Run full render pipeline benchmarks and return collected results.
pub fn run(ctx: &GpuContext) -> Vec<BenchResult> {
    println!("== Benchmark: Full Render Pipeline ==");

    let mut results = Vec::new();

    // ---------------------------------------------------------------
    // Setup: compile shader program and create quad VAO
    // ---------------------------------------------------------------
    let vert_cstr =
        CString::new(VERTEX_SRC).expect("vertex shader source contains interior null byte");
    let frag_cstr =
        CString::new(FRAGMENT_SRC).expect("fragment shader source contains interior null byte");

    let program = compile_program(&vert_cstr, &frag_cstr);
    if program == 0 {
        println!("  SKIP: shader program compilation failed");
        println!();
        return results;
    }

    // Get uniform location for the texture sampler
    let tex_uniform = unsafe {
        let name = CString::new("tex").unwrap();
        gls::gl::GetUniformLocation(program, name.as_ptr())
    };

    let (vao, vbo, ebo) = create_quad_vao();

    let fourcc = rgba_fourcc();

    // ---------------------------------------------------------------
    // Test configurations: (src_w, src_h, dst_w, dst_h, label)
    // ---------------------------------------------------------------
    let configs: &[(u32, u32, u32, u32, &str)] = &[
        (1920, 1080, 640, 640, "1080p_to_640"),
        (1920, 1080, 320, 320, "1080p_to_320"),
        (3840, 2160, 640, 640, "4k_to_640"),
    ];

    for &(src_w, src_h, dst_w, dst_h, label) in configs {
        let src_pitch = src_w * 4;
        let dst_pitch = dst_w * 4;
        let src_bytes = (src_pitch * src_h) as usize;
        let dst_bytes = (dst_pitch * dst_h) as usize;

        // Allocate DMA tensors for src and dst
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
        // Test A: render_per_call — full resource create/destroy per iteration
        // -----------------------------------------------------------
        {
            let name = format!("render_per_call/{label}");
            let r = run_bench(&name, 10, 200, || {
                // 1-2. Create EGLImages from DMA fds
                let src_img = ctx
                    .create_egl_image_dma(
                        src_fd,
                        src_w as i32,
                        src_h as i32,
                        fourcc,
                        src_pitch as i32,
                    )
                    .expect("src create_egl_image_dma failed");
                let dst_img = ctx
                    .create_egl_image_dma(
                        dst_fd,
                        dst_w as i32,
                        dst_h as i32,
                        fourcc,
                        dst_pitch as i32,
                    )
                    .expect("dst create_egl_image_dma failed");

                unsafe {
                    // 3. Create src texture, bind EGLImage, set LINEAR filtering
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

                    // 4. Create dst texture, bind EGLImage
                    let mut dst_tex = 0u32;
                    gls::gl::GenTextures(1, &mut dst_tex);
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, dst_tex);
                    gls::gl::EGLImageTargetTexture2DOES(gls::gl::TEXTURE_2D, dst_img.as_ptr());

                    // 5. Create FBO, attach dst texture
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

                    // 6. Viewport
                    gls::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);

                    // 7. UseProgram, bind src texture to TEXTURE0
                    gls::gl::UseProgram(program);
                    gls::gl::ActiveTexture(gls::gl::TEXTURE0);
                    gls::gl::BindTexture(gls::gl::TEXTURE_2D, src_tex);
                    gls::gl::Uniform1i(tex_uniform, 0);

                    // 8. Draw
                    gls::gl::BindVertexArray(vao);
                    gls::gl::DrawElements(gls::gl::TRIANGLES, 6, gls::gl::UNSIGNED_INT, null());

                    // 9. Finish
                    gls::gl::Finish();

                    // 10. Cleanup GL resources
                    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
                    gls::gl::BindVertexArray(0);
                    gls::gl::DeleteFramebuffers(1, &fbo);
                    gls::gl::DeleteTextures(1, &dst_tex);
                    gls::gl::DeleteTextures(1, &src_tex);
                }

                // Destroy EGLImages
                ctx.destroy_egl_image(src_img)
                    .expect("destroy src EGLImage failed");
                ctx.destroy_egl_image(dst_img)
                    .expect("destroy dst EGLImage failed");
            });
            r.print_summary();
            results.push(r);
        }

        // -----------------------------------------------------------
        // Test B: render_cached — pre-create resources, measure render only
        // -----------------------------------------------------------
        {
            // Pre-create resources (done once)
            let src_img = ctx
                .create_egl_image_dma(src_fd, src_w as i32, src_h as i32, fourcc, src_pitch as i32)
                .expect("src create_egl_image_dma failed");
            let dst_img = ctx
                .create_egl_image_dma(dst_fd, dst_w as i32, dst_h as i32, fourcc, dst_pitch as i32)
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

            let name = format!("render_cached/{label}");
            let r = run_bench(&name, 10, 200, || unsafe {
                // Only the render path — resources are already set up
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
                gls::gl::DeleteFramebuffers(1, &fbo);
                gls::gl::DeleteTextures(1, &dst_tex);
                gls::gl::DeleteTextures(1, &src_tex);
            }
            ctx.destroy_egl_image(dst_img)
                .expect("destroy dst EGLImage failed");
            ctx.destroy_egl_image(src_img)
                .expect("destroy src EGLImage failed");
        }
    }

    // ---------------------------------------------------------------
    // Cleanup shared resources
    // ---------------------------------------------------------------
    unsafe {
        gls::gl::DeleteVertexArrays(1, &vao);
        gls::gl::DeleteBuffers(1, &vbo);
        gls::gl::DeleteBuffers(1, &ebo);
        gls::gl::DeleteProgram(program);
    }

    println!();
    results
}
