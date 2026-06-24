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

/// DRM FourCC for packed BGR888 (RGB888 in DRM naming).
pub(crate) fn rgb888_fourcc() -> u32 {
    gbm::drm::buffer::DrmFourcc::Bgr888 as u32
}

/// Compile a shader program from vertex + fragment source strings.
///
/// Returns `Ok(program_id)` on success. On failure, deletes all intermediate
/// GL objects and returns `Err` with the relevant info log.
pub(crate) fn compile_program(vert: &CString, frag: &CString) -> Result<u32, String> {
    unsafe {
        let program = edgefirst_gl::gl::CreateProgram();

        // --- Vertex shader ---
        let vs = edgefirst_gl::gl::CreateShader(edgefirst_gl::gl::VERTEX_SHADER);
        let vs_ptr = vert.as_ptr();
        edgefirst_gl::gl::ShaderSource(vs, 1, &raw const vs_ptr, null());
        edgefirst_gl::gl::CompileShader(vs);
        let mut vs_ok: i32 = 0;
        edgefirst_gl::gl::GetShaderiv(vs, edgefirst_gl::gl::COMPILE_STATUS, &mut vs_ok);
        if vs_ok == 0 {
            let mut len: i32 = 0;
            edgefirst_gl::gl::GetShaderiv(vs, edgefirst_gl::gl::INFO_LOG_LENGTH, &mut len);
            let mut log = vec![0u8; len.max(1) as usize];
            edgefirst_gl::gl::GetShaderInfoLog(
                vs,
                len,
                std::ptr::null_mut(),
                log.as_mut_ptr().cast(),
            );
            edgefirst_gl::gl::DeleteShader(vs);
            edgefirst_gl::gl::DeleteProgram(program);
            let msg = String::from_utf8_lossy(&log).into_owned();
            return Err(format!("vertex shader compile failed: {msg}"));
        }
        edgefirst_gl::gl::AttachShader(program, vs);

        // --- Fragment shader ---
        let fs = edgefirst_gl::gl::CreateShader(edgefirst_gl::gl::FRAGMENT_SHADER);
        let fs_ptr = frag.as_ptr();
        edgefirst_gl::gl::ShaderSource(fs, 1, &raw const fs_ptr, null());
        edgefirst_gl::gl::CompileShader(fs);
        let mut fs_ok: i32 = 0;
        edgefirst_gl::gl::GetShaderiv(fs, edgefirst_gl::gl::COMPILE_STATUS, &mut fs_ok);
        if fs_ok == 0 {
            let mut len: i32 = 0;
            edgefirst_gl::gl::GetShaderiv(fs, edgefirst_gl::gl::INFO_LOG_LENGTH, &mut len);
            let mut log = vec![0u8; len.max(1) as usize];
            edgefirst_gl::gl::GetShaderInfoLog(
                fs,
                len,
                std::ptr::null_mut(),
                log.as_mut_ptr().cast(),
            );
            edgefirst_gl::gl::DetachShader(program, vs);
            edgefirst_gl::gl::DeleteShader(vs);
            edgefirst_gl::gl::DeleteShader(fs);
            edgefirst_gl::gl::DeleteProgram(program);
            let msg = String::from_utf8_lossy(&log).into_owned();
            return Err(format!("fragment shader compile failed: {msg}"));
        }
        edgefirst_gl::gl::AttachShader(program, fs);

        // --- Link ---
        edgefirst_gl::gl::LinkProgram(program);
        let mut link_ok: i32 = 0;
        edgefirst_gl::gl::GetProgramiv(program, edgefirst_gl::gl::LINK_STATUS, &mut link_ok);
        if link_ok == 0 {
            let mut len: i32 = 0;
            edgefirst_gl::gl::GetProgramiv(program, edgefirst_gl::gl::INFO_LOG_LENGTH, &mut len);
            let mut log = vec![0u8; len.max(1) as usize];
            edgefirst_gl::gl::GetProgramInfoLog(
                program,
                len,
                std::ptr::null_mut(),
                log.as_mut_ptr().cast(),
            );
            edgefirst_gl::gl::DetachShader(program, vs);
            edgefirst_gl::gl::DetachShader(program, fs);
            edgefirst_gl::gl::DeleteShader(vs);
            edgefirst_gl::gl::DeleteShader(fs);
            edgefirst_gl::gl::DeleteProgram(program);
            let msg = String::from_utf8_lossy(&log).into_owned();
            return Err(format!("program link failed: {msg}"));
        }

        // Detach shaders so they can be deleted while program remains
        edgefirst_gl::gl::DetachShader(program, vs);
        edgefirst_gl::gl::DetachShader(program, fs);
        edgefirst_gl::gl::DeleteShader(vs);
        edgefirst_gl::gl::DeleteShader(fs);

        Ok(program)
    }
}

/// Create a VAO with interleaved vertex + UV data and an element buffer.
/// Returns (vao, vbo, ebo).
pub(crate) fn create_quad_vao() -> (u32, u32, u32) {
    unsafe {
        let mut vao = 0u32;
        edgefirst_gl::gl::GenVertexArrays(1, &mut vao);
        edgefirst_gl::gl::BindVertexArray(vao);

        // VBO: interleaved [x, y, u, v]
        let mut vbo = 0u32;
        edgefirst_gl::gl::GenBuffers(1, &mut vbo);
        edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ARRAY_BUFFER, vbo);
        edgefirst_gl::gl::BufferData(
            edgefirst_gl::gl::ARRAY_BUFFER,
            std::mem::size_of_val(&QUAD_VERTICES) as isize,
            QUAD_VERTICES.as_ptr() as *const std::ffi::c_void,
            edgefirst_gl::gl::STATIC_DRAW,
        );

        // Attribute 0: position (vec2, declared as vec3 in shader — z defaults to 0)
        let stride = (4 * std::mem::size_of::<f32>()) as i32;
        edgefirst_gl::gl::VertexAttribPointer(
            0,
            2,
            edgefirst_gl::gl::FLOAT,
            edgefirst_gl::gl::FALSE,
            stride,
            null(),
        );
        edgefirst_gl::gl::EnableVertexAttribArray(0);

        // Attribute 1: texCoord (vec2)
        edgefirst_gl::gl::VertexAttribPointer(
            1,
            2,
            edgefirst_gl::gl::FLOAT,
            edgefirst_gl::gl::FALSE,
            stride,
            (2 * std::mem::size_of::<f32>()) as *const std::ffi::c_void,
        );
        edgefirst_gl::gl::EnableVertexAttribArray(1);

        // EBO: indices
        let mut ebo = 0u32;
        edgefirst_gl::gl::GenBuffers(1, &mut ebo);
        edgefirst_gl::gl::BindBuffer(edgefirst_gl::gl::ELEMENT_ARRAY_BUFFER, ebo);
        edgefirst_gl::gl::BufferData(
            edgefirst_gl::gl::ELEMENT_ARRAY_BUFFER,
            std::mem::size_of_val(&QUAD_INDICES) as isize,
            QUAD_INDICES.as_ptr() as *const std::ffi::c_void,
            edgefirst_gl::gl::STATIC_DRAW,
        );

        edgefirst_gl::gl::BindVertexArray(0);

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

    let program = match compile_program(&vert_cstr, &frag_cstr) {
        Ok(p) => p,
        Err(e) => {
            println!("  SKIP: shader program compilation failed: {e}");
            println!();
            return results;
        }
    };

    // Get uniform location for the texture sampler
    let tex_uniform = unsafe {
        let name = CString::new("tex").unwrap();
        edgefirst_gl::gl::GetUniformLocation(program, name.as_ptr())
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
                    edgefirst_gl::gl::GenTextures(1, &mut src_tex);
                    edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, src_tex);
                    edgefirst_gl::gl::EGLImageTargetTexture2DOES(
                        edgefirst_gl::gl::TEXTURE_2D,
                        src_img.as_ptr(),
                    );
                    edgefirst_gl::gl::TexParameteri(
                        edgefirst_gl::gl::TEXTURE_2D,
                        edgefirst_gl::gl::TEXTURE_MIN_FILTER,
                        edgefirst_gl::gl::LINEAR as i32,
                    );
                    edgefirst_gl::gl::TexParameteri(
                        edgefirst_gl::gl::TEXTURE_2D,
                        edgefirst_gl::gl::TEXTURE_MAG_FILTER,
                        edgefirst_gl::gl::LINEAR as i32,
                    );

                    // 4. Create dst texture, bind EGLImage
                    let mut dst_tex = 0u32;
                    edgefirst_gl::gl::GenTextures(1, &mut dst_tex);
                    edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, dst_tex);
                    edgefirst_gl::gl::EGLImageTargetTexture2DOES(
                        edgefirst_gl::gl::TEXTURE_2D,
                        dst_img.as_ptr(),
                    );

                    // 5. Create FBO, attach dst texture
                    let mut fbo = 0u32;
                    edgefirst_gl::gl::GenFramebuffers(1, &mut fbo);
                    edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, fbo);
                    edgefirst_gl::gl::FramebufferTexture2D(
                        edgefirst_gl::gl::FRAMEBUFFER,
                        edgefirst_gl::gl::COLOR_ATTACHMENT0,
                        edgefirst_gl::gl::TEXTURE_2D,
                        dst_tex,
                        0,
                    );

                    // 6. Viewport
                    edgefirst_gl::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);

                    // 7. UseProgram, bind src texture to TEXTURE0
                    edgefirst_gl::gl::UseProgram(program);
                    edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
                    edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, src_tex);
                    edgefirst_gl::gl::Uniform1i(tex_uniform, 0);

                    // 8. Draw
                    edgefirst_gl::gl::BindVertexArray(vao);
                    edgefirst_gl::gl::DrawElements(
                        edgefirst_gl::gl::TRIANGLES,
                        6,
                        edgefirst_gl::gl::UNSIGNED_INT,
                        null(),
                    );

                    // 9. Finish
                    edgefirst_gl::gl::Finish();

                    // 10. Cleanup GL resources
                    edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, 0);
                    edgefirst_gl::gl::BindVertexArray(0);
                    edgefirst_gl::gl::DeleteFramebuffers(1, &fbo);
                    edgefirst_gl::gl::DeleteTextures(1, &dst_tex);
                    edgefirst_gl::gl::DeleteTextures(1, &src_tex);
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
                edgefirst_gl::gl::GenTextures(1, &mut src_tex);
                edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, src_tex);
                edgefirst_gl::gl::EGLImageTargetTexture2DOES(
                    edgefirst_gl::gl::TEXTURE_2D,
                    src_img.as_ptr(),
                );
                edgefirst_gl::gl::TexParameteri(
                    edgefirst_gl::gl::TEXTURE_2D,
                    edgefirst_gl::gl::TEXTURE_MIN_FILTER,
                    edgefirst_gl::gl::LINEAR as i32,
                );
                edgefirst_gl::gl::TexParameteri(
                    edgefirst_gl::gl::TEXTURE_2D,
                    edgefirst_gl::gl::TEXTURE_MAG_FILTER,
                    edgefirst_gl::gl::LINEAR as i32,
                );

                let mut dst_tex = 0u32;
                edgefirst_gl::gl::GenTextures(1, &mut dst_tex);
                edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, dst_tex);
                edgefirst_gl::gl::EGLImageTargetTexture2DOES(
                    edgefirst_gl::gl::TEXTURE_2D,
                    dst_img.as_ptr(),
                );

                let mut fbo = 0u32;
                edgefirst_gl::gl::GenFramebuffers(1, &mut fbo);
                edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, fbo);
                edgefirst_gl::gl::FramebufferTexture2D(
                    edgefirst_gl::gl::FRAMEBUFFER,
                    edgefirst_gl::gl::COLOR_ATTACHMENT0,
                    edgefirst_gl::gl::TEXTURE_2D,
                    dst_tex,
                    0,
                );

                (src_tex, dst_tex, fbo)
            };

            let name = format!("render_cached/{label}");
            let r = run_bench(&name, 10, 200, || unsafe {
                // Only the render path — resources are already set up
                edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, fbo);
                edgefirst_gl::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);

                edgefirst_gl::gl::UseProgram(program);
                edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
                edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, src_tex);
                edgefirst_gl::gl::Uniform1i(tex_uniform, 0);

                edgefirst_gl::gl::BindVertexArray(vao);
                edgefirst_gl::gl::DrawElements(
                    edgefirst_gl::gl::TRIANGLES,
                    6,
                    edgefirst_gl::gl::UNSIGNED_INT,
                    null(),
                );

                edgefirst_gl::gl::Finish();
            });
            r.print_summary();
            results.push(r);

            // Cleanup cached resources
            unsafe {
                edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, 0);
                edgefirst_gl::gl::BindVertexArray(0);
                edgefirst_gl::gl::DeleteFramebuffers(1, &fbo);
                edgefirst_gl::gl::DeleteTextures(1, &dst_tex);
                edgefirst_gl::gl::DeleteTextures(1, &src_tex);
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
        edgefirst_gl::gl::DeleteVertexArrays(1, &vao);
        edgefirst_gl::gl::DeleteBuffers(1, &vbo);
        edgefirst_gl::gl::DeleteBuffers(1, &ebo);
        edgefirst_gl::gl::DeleteProgram(program);
    }

    println!();
    results
}
