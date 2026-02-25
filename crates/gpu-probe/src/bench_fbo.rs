// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! FBO lifecycle benchmarks — measures create/bind/delete, bind/unbind reuse,
//! and full-cycle render with an attached colour texture.

use crate::bench::{run_bench, BenchResult};
use crate::egl_context::GpuContext;

/// Run FBO lifecycle benchmarks and return collected results.
pub fn run(_ctx: &GpuContext) -> Vec<BenchResult> {
    println!("== Benchmark: FBO Lifecycle ==");

    let mut results = Vec::new();

    // 1. fbo_create_bind_delete — full lifecycle each iteration
    {
        let r = run_bench("fbo_create_bind_delete", 10, 1000, || unsafe {
            let mut fbo: u32 = 0;
            gls::gl::GenFramebuffers(1, &mut fbo);
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
            gls::gl::DeleteFramebuffers(1, &fbo);
        });
        r.print_summary();
        results.push(r);
    }

    // 2. fbo_bind_unbind_reuse — create once, measure bind/unbind only
    {
        let mut fbo: u32 = 0;
        unsafe {
            gls::gl::GenFramebuffers(1, &mut fbo);
        }

        let r = run_bench("fbo_bind_unbind_reuse", 10, 1000, || unsafe {
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
        });
        r.print_summary();
        results.push(r);

        unsafe {
            gls::gl::DeleteFramebuffers(1, &fbo);
        }
    }

    // 3. fbo_full_cycle_render — texture + FBO + viewport + clear + finish
    {
        let r = run_bench("fbo_full_cycle_render", 10, 500, || unsafe {
            let mut tex: u32 = 0;
            gls::gl::GenTextures(1, &mut tex);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, tex);
            gls::gl::TexImage2D(
                gls::gl::TEXTURE_2D,
                0,
                gls::gl::RGBA as i32,
                640,
                640,
                0,
                gls::gl::RGBA,
                gls::gl::UNSIGNED_BYTE,
                std::ptr::null(),
            );

            let mut fbo: u32 = 0;
            gls::gl::GenFramebuffers(1, &mut fbo);
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
            gls::gl::FramebufferTexture2D(
                gls::gl::FRAMEBUFFER,
                gls::gl::COLOR_ATTACHMENT0,
                gls::gl::TEXTURE_2D,
                tex,
                0,
            );

            gls::gl::Viewport(0, 0, 640, 640);
            gls::gl::ClearColor(0.0, 0.0, 0.0, 1.0);
            gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
            gls::gl::Finish();

            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
            gls::gl::DeleteFramebuffers(1, &fbo);
            gls::gl::DeleteTextures(1, &tex);
        });
        r.print_summary();
        results.push(r);
    }

    println!();
    results
}
