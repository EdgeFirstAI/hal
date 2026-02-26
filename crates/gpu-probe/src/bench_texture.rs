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
            gls::gl::GenTextures(1, &mut tex);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, tex);
            gls::gl::TexImage2D(
                gls::gl::TEXTURE_2D,
                0,
                gls::gl::RGBA as i32,
                w,
                h,
                0,
                gls::gl::RGBA,
                gls::gl::UNSIGNED_BYTE,
                std::ptr::null(),
            );
        }

        // Upload data buffer.
        let data = vec![128u8; pixel_count];

        // Benchmark: TexSubImage2D upload + Finish
        {
            let name = format!("tex_upload_sub/{label}");
            let r = run_bench(&name, 10, 200, || unsafe {
                gls::gl::BindTexture(gls::gl::TEXTURE_2D, tex);
                gls::gl::TexSubImage2D(
                    gls::gl::TEXTURE_2D,
                    0,
                    0,
                    0,
                    w,
                    h,
                    gls::gl::RGBA,
                    gls::gl::UNSIGNED_BYTE,
                    data.as_ptr() as *const std::ffi::c_void,
                );
                gls::gl::Finish();
            });
            r.print_summary();
            results.push(r);
        }

        // Create an FBO with that texture attached for readback.
        let mut fbo: u32 = 0;
        unsafe {
            gls::gl::GenFramebuffers(1, &mut fbo);
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
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
                gls::gl::BindFramebuffer(gls::gl::READ_FRAMEBUFFER, fbo);
                gls::gl::ReadPixels(
                    0,
                    0,
                    w,
                    h,
                    gls::gl::RGBA,
                    gls::gl::UNSIGNED_BYTE,
                    readback.as_mut_ptr() as *mut std::ffi::c_void,
                );
                gls::gl::Finish();
            });
            r.print_summary();
            results.push(r);
        }

        // Cleanup
        unsafe {
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
            gls::gl::DeleteFramebuffers(1, &fbo);
            gls::gl::DeleteTextures(1, &tex);
        }
    }

    println!();
    results
}
