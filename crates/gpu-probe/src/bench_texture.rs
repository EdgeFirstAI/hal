// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Texture upload/download benchmarks — measures TexSubImage2D upload and
//! ReadPixels download latency at various resolutions.

use crate::bench::{run_bench, BenchResult};
use crate::egl_context::GpuContext;

/// Run texture upload and download benchmarks and return collected results.
pub fn run(_ctx: &GpuContext) -> Vec<BenchResult> {
    println!("== Benchmark: Texture Upload/Download ==");

    let configs: &[(i32, i32, &str)] = &[
        (640, 640, "640x640"),
        (1920, 1080, "1080p"),
        (3840, 2160, "4K"),
    ];

    let mut results = Vec::new();

    for &(w, h, label) in configs {
        let pixel_count = (w * h * 4) as usize;

        // Create a texture with TexImage2D (RGBA, null data).
        let mut tex: u32 = 0;
        unsafe {
            edgefirst_gl::gl::GenTextures(1, &mut tex);
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, tex);
            edgefirst_gl::gl::TexImage2D(
                edgefirst_gl::gl::TEXTURE_2D,
                0,
                edgefirst_gl::gl::RGBA as i32,
                w,
                h,
                0,
                edgefirst_gl::gl::RGBA,
                edgefirst_gl::gl::UNSIGNED_BYTE,
                std::ptr::null(),
            );
        }

        // Upload data buffer.
        let data = vec![128u8; pixel_count];

        // Benchmark: TexSubImage2D upload + Finish
        {
            let name = format!("tex_upload_sub/{label}");
            let r = run_bench(&name, 10, 200, || unsafe {
                edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, tex);
                edgefirst_gl::gl::TexSubImage2D(
                    edgefirst_gl::gl::TEXTURE_2D,
                    0,
                    0,
                    0,
                    w,
                    h,
                    edgefirst_gl::gl::RGBA,
                    edgefirst_gl::gl::UNSIGNED_BYTE,
                    data.as_ptr() as *const std::ffi::c_void,
                );
                edgefirst_gl::gl::Finish();
            });
            r.print_summary();
            results.push(r);
        }

        // Create an FBO with that texture attached for readback.
        let mut fbo: u32 = 0;
        unsafe {
            edgefirst_gl::gl::GenFramebuffers(1, &mut fbo);
            edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, fbo);
            edgefirst_gl::gl::FramebufferTexture2D(
                edgefirst_gl::gl::FRAMEBUFFER,
                edgefirst_gl::gl::COLOR_ATTACHMENT0,
                edgefirst_gl::gl::TEXTURE_2D,
                tex,
                0,
            );
        }

        // Readback buffer.
        let mut readback = vec![0u8; pixel_count];

        // Benchmark: ReadPixels download + Finish
        {
            let name = format!("tex_readback_read_pixels/{label}");
            let r = run_bench(&name, 10, 200, || unsafe {
                edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::READ_FRAMEBUFFER, fbo);
                edgefirst_gl::gl::ReadPixels(
                    0,
                    0,
                    w,
                    h,
                    edgefirst_gl::gl::RGBA,
                    edgefirst_gl::gl::UNSIGNED_BYTE,
                    readback.as_mut_ptr() as *mut std::ffi::c_void,
                );
                edgefirst_gl::gl::Finish();
            });
            r.print_summary();
            results.push(r);
        }

        // Cleanup
        unsafe {
            edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, 0);
            edgefirst_gl::gl::DeleteFramebuffers(1, &fbo);
            edgefirst_gl::gl::DeleteTextures(1, &tex);
        }
    }

    println!();
    results
}
