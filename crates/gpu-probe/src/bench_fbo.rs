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
            edgefirst_gl::gl::GenFramebuffers(1, &mut fbo);
            edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, fbo);
            edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, 0);
            edgefirst_gl::gl::DeleteFramebuffers(1, &fbo);
        });
        r.print_summary();
        results.push(r);
    }

    // 2. fbo_bind_unbind_reuse — create once, measure bind/unbind only
    {
        let mut fbo: u32 = 0;
        unsafe {
            edgefirst_gl::gl::GenFramebuffers(1, &mut fbo);
        }

        let r = run_bench("fbo_bind_unbind_reuse", 10, 1000, || unsafe {
            edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, fbo);
            edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, 0);
        });
        r.print_summary();
        results.push(r);

        unsafe {
            edgefirst_gl::gl::DeleteFramebuffers(1, &fbo);
        }
    }

    // 3. fbo_full_cycle_render — texture + FBO + viewport + clear + finish
    {
        let r = run_bench("fbo_full_cycle_render", 10, 500, || unsafe {
            let mut tex: u32 = 0;
            edgefirst_gl::gl::GenTextures(1, &mut tex);
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, tex);
            edgefirst_gl::gl::TexImage2D(
                edgefirst_gl::gl::TEXTURE_2D,
                0,
                edgefirst_gl::gl::RGBA as i32,
                640,
                640,
                0,
                edgefirst_gl::gl::RGBA,
                edgefirst_gl::gl::UNSIGNED_BYTE,
                std::ptr::null(),
            );

            let mut fbo: u32 = 0;
            edgefirst_gl::gl::GenFramebuffers(1, &mut fbo);
            edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, fbo);
            edgefirst_gl::gl::FramebufferTexture2D(
                edgefirst_gl::gl::FRAMEBUFFER,
                edgefirst_gl::gl::COLOR_ATTACHMENT0,
                edgefirst_gl::gl::TEXTURE_2D,
                tex,
                0,
            );

            edgefirst_gl::gl::Viewport(0, 0, 640, 640);
            edgefirst_gl::gl::ClearColor(0.0, 0.0, 0.0, 1.0);
            edgefirst_gl::gl::Clear(edgefirst_gl::gl::COLOR_BUFFER_BIT);
            edgefirst_gl::gl::Finish();

            edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, 0);
            edgefirst_gl::gl::DeleteFramebuffers(1, &fbo);
            edgefirst_gl::gl::DeleteTextures(1, &tex);
        });
        r.print_summary();
        results.push(r);
    }

    println!();
    results
}
