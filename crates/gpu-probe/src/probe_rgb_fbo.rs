// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! RGB8 renderbuffer FBO probe — tests whether the GPU supports rendering
//! into a packed RGB888 DMA-buf via an EGLImage-backed renderbuffer.

use crate::bench_render;
use crate::egl_context::GpuContext;
use edgefirst_tensor::{Tensor, TensorMemory, TensorTrait};
use std::os::unix::io::AsRawFd;

/// OpenGL ES `GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS` (0x8CD9).
const FRAMEBUFFER_INCOMPLETE_DIMENSIONS: u32 = 0x8CD9;

/// OpenGL ES `GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE` (0x8D56).
const FRAMEBUFFER_INCOMPLETE_MULTISAMPLE: u32 = 0x8D56;

/// Returns `"YES"` when `b` is true, `"no"` otherwise.
fn yn(b: bool) -> &'static str {
    if b {
        "YES"
    } else {
        "no"
    }
}

/// Map a `glCheckFramebufferStatus` return value to a human-readable name.
fn fbo_status_name(status: u32) -> &'static str {
    match status {
        gls::gl::FRAMEBUFFER_COMPLETE => "FRAMEBUFFER_COMPLETE",
        gls::gl::FRAMEBUFFER_INCOMPLETE_ATTACHMENT => "FRAMEBUFFER_INCOMPLETE_ATTACHMENT",
        gls::gl::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT => {
            "FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT"
        }
        FRAMEBUFFER_INCOMPLETE_DIMENSIONS => "FRAMEBUFFER_INCOMPLETE_DIMENSIONS",
        gls::gl::FRAMEBUFFER_UNSUPPORTED => "FRAMEBUFFER_UNSUPPORTED",
        FRAMEBUFFER_INCOMPLETE_MULTISAMPLE => "FRAMEBUFFER_INCOMPLETE_MULTISAMPLE",
        _ => "UNKNOWN",
    }
}

/// Probe whether the GPU can render into an RGB888 DMA-buf via an
/// EGLImage-backed renderbuffer.
///
/// Prints a structured report and returns `true` if the FBO was complete.
pub fn run(ctx: &GpuContext) -> bool {
    println!("=== RGB8 Renderbuffer FBO Probe ===");

    // --- Check 1: GL_OES_rgb8_rgba8 extension ---
    let gl_exts = ctx.gl_extensions();
    let has_ext = gl_exts.iter().any(|e| e == "GL_OES_rgb8_rgba8");
    println!("  {:42} {}", "GL_OES_rgb8_rgba8", yn(has_ext));

    // --- Check 2: glEGLImageTargetRenderbufferStorageOES entry point ---
    let has_rbo = ctx.has_rgb8_renderbuffer();
    println!(
        "  {:42} {}",
        "glEGLImageTargetRenderbufferStorageOES",
        yn(has_rbo)
    );

    if !has_rbo {
        println!(
            "  {:42} SKIP (entry point not available)",
            "FBO completeness"
        );
        println!();
        return false;
    }

    // --- Check 3: FBO completeness with RGB888 EGLImage renderbuffer ---
    let width: u32 = 640;
    let height: u32 = 640;
    let bpp: u32 = 3;
    let pitch = width * bpp;
    let byte_count = (pitch * height) as usize;

    // Allocate DMA tensor
    let tensor = match Tensor::<u8>::new(&[byte_count], Some(TensorMemory::Dma), None) {
        Ok(t) => t,
        Err(e) => {
            println!("  {:42} SKIP (DMA alloc failed: {e})", "FBO completeness");
            println!();
            return false;
        }
    };

    let fd_owned = match tensor.clone_fd() {
        Ok(fd) => fd,
        Err(e) => {
            println!("  {:42} SKIP (clone_fd failed: {e})", "FBO completeness");
            println!();
            return false;
        }
    };

    // Create EGLImage from DMA-buf with DRM_FORMAT_BGR888
    let fourcc = bench_render::rgb888_fourcc();
    let egl_image = match ctx.create_egl_image_dma(
        fd_owned.as_raw_fd(),
        width as i32,
        height as i32,
        fourcc,
        pitch as i32,
    ) {
        Ok(img) => img,
        Err(e) => {
            println!(
                "  {:42} SKIP (EGLImage creation failed: {e})",
                "FBO completeness"
            );
            println!();
            return false;
        }
    };

    let complete = unsafe {
        // Create renderbuffer backed by the EGLImage
        let mut rbo: u32 = 0;
        gls::gl::GenRenderbuffers(1, &mut rbo);
        gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, rbo);
        gls::gl::EGLImageTargetRenderbufferStorageOES(gls::gl::RENDERBUFFER, egl_image.as_ptr());

        // Create FBO and attach the renderbuffer
        let mut fbo: u32 = 0;
        gls::gl::GenFramebuffers(1, &mut fbo);
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
        gls::gl::FramebufferRenderbuffer(
            gls::gl::FRAMEBUFFER,
            gls::gl::COLOR_ATTACHMENT0,
            gls::gl::RENDERBUFFER,
            rbo,
        );

        // Check completeness
        let status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
        let is_complete = status == gls::gl::FRAMEBUFFER_COMPLETE;

        if is_complete {
            println!("  {:42} PASS", "FBO completeness");
        } else {
            println!(
                "  {:42} FAIL ({})",
                "FBO completeness",
                fbo_status_name(status)
            );
        }

        // Cleanup GL resources
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
        gls::gl::DeleteFramebuffers(1, &fbo);
        gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, 0);
        gls::gl::DeleteRenderbuffers(1, &rbo);

        is_complete
    };

    // Cleanup EGL resource
    let _ = ctx.destroy_egl_image(egl_image);

    println!();
    complete
}
