// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! GPU capability probe — prints a structured report of EGL/GL capabilities,
//! limits, extension support, and device-node availability.

use crate::egl_context::GpuContext;
use khronos_egl as egl;
use std::path::Path;

/// Returns `"YES"` when `b` is true, `"no"` otherwise.
fn yn(b: bool) -> &'static str {
    if b {
        "YES"
    } else {
        "no"
    }
}

/// Print a full structured capability report for the given GPU context.
pub fn run_probes(ctx: &GpuContext) {
    // -----------------------------------------------------------------
    // Identity
    // -----------------------------------------------------------------
    println!("=== Identity ===");
    println!(
        "  {:42} {}",
        "EGL Vendor",
        ctx.egl_query(egl::VENDOR).unwrap_or_else(|| "n/a".into())
    );
    println!(
        "  {:42} {}",
        "EGL Version",
        ctx.egl_query(egl::VERSION).unwrap_or_else(|| "n/a".into())
    );
    println!("  {:42} {}", "GL Renderer", ctx.gl_renderer());
    println!("  {:42} {}", "GL Version", ctx.gl_version());
    println!("  {:42} {}", "GL Vendor", ctx.gl_vendor());
    println!();

    // -----------------------------------------------------------------
    // GL Limits
    // -----------------------------------------------------------------
    println!("=== GL Limits ===");

    let limits: &[(&str, gls::GLenum)] = &[
        ("MAX_TEXTURE_SIZE", gls::gl::MAX_TEXTURE_SIZE),
        ("MAX_RENDERBUFFER_SIZE", gls::gl::MAX_RENDERBUFFER_SIZE),
        ("MAX_VIEWPORT_DIMS", gls::gl::MAX_VIEWPORT_DIMS),
        ("MAX_TEXTURE_IMAGE_UNITS", gls::gl::MAX_TEXTURE_IMAGE_UNITS),
        (
            "MAX_COMBINED_TEXTURE_IMAGE_UNITS",
            gls::gl::MAX_COMBINED_TEXTURE_IMAGE_UNITS,
        ),
        ("MAX_VERTEX_ATTRIBS", gls::gl::MAX_VERTEX_ATTRIBS),
        ("MAX_VARYING_VECTORS", gls::gl::MAX_VARYING_VECTORS),
        (
            "MAX_FRAGMENT_UNIFORM_VECTORS",
            gls::gl::MAX_FRAGMENT_UNIFORM_VECTORS,
        ),
        ("MAX_COLOR_ATTACHMENTS", gls::gl::MAX_COLOR_ATTACHMENTS),
    ];

    for (name, param) in limits {
        let value = ctx.gl_get_integer(*param);
        println!("  {:42} {}", name, value);
    }
    println!();

    // -----------------------------------------------------------------
    // DMA-BUF Support
    // -----------------------------------------------------------------
    println!("=== DMA-BUF Support ===");

    let egl_exts = ctx.egl_extensions();
    let has_dmabuf_import = egl_exts.iter().any(|e| e == "EGL_EXT_image_dma_buf_import");
    let has_dmabuf_modifiers = egl_exts
        .iter()
        .any(|e| e == "EGL_EXT_image_dma_buf_import_modifiers");

    println!(
        "  {:42} {}",
        "EGL_EXT_image_dma_buf_import",
        yn(has_dmabuf_import)
    );
    println!(
        "  {:42} {}",
        "EGL_EXT_image_dma_buf_import_modifiers",
        yn(has_dmabuf_modifiers)
    );
    println!(
        "  {:42} {}",
        "eglCreateImageKHR available",
        yn(ctx.has_egl_create_image_khr())
    );
    println!();

    // -----------------------------------------------------------------
    // Key GL Extensions
    // -----------------------------------------------------------------
    println!("=== Key GL Extensions ===");

    let gl_exts = ctx.gl_extensions();
    let key_extensions = [
        "GL_OES_EGL_image_external",
        "GL_OES_EGL_image_external_essl3",
        "GL_OES_surfaceless_context",
        "GL_OES_texture_float_linear",
        "GL_OES_texture_half_float",
        "GL_OES_texture_half_float_linear",
        "GL_EXT_texture_format_BGRA8888",
        "GL_EXT_color_buffer_float",
        "GL_EXT_color_buffer_half_float",
        "GL_NV_pixel_buffer_object",
        "GL_EXT_map_buffer_range",
        "GL_OES_mapbuffer",
    ];

    for ext in &key_extensions {
        let present = gl_exts.iter().any(|e| e == ext);
        println!("  {:42} {}", ext, yn(present));
    }
    println!();

    // -----------------------------------------------------------------
    // DMA Heap
    // -----------------------------------------------------------------
    println!("=== DMA Heap ===");

    let dma_heap_paths = ["/dev/dma_heap/linux,cma", "/dev/dma_heap/system"];
    for path in &dma_heap_paths {
        println!("  {:42} {}", path, yn(Path::new(path).exists()));
    }
    println!();

    // -----------------------------------------------------------------
    // DRM
    // -----------------------------------------------------------------
    println!("=== DRM ===");

    let drm_paths = ["/dev/dri/renderD128", "/dev/dri/card0", "/dev/dri/card1"];
    for path in &drm_paths {
        println!("  {:42} {}", path, yn(Path::new(path).exists()));
    }
    println!();

    // -----------------------------------------------------------------
    // Full extension lists
    // -----------------------------------------------------------------
    println!("=== EGL Extensions ({}) ===", egl_exts.len());
    for ext in &egl_exts {
        println!("  {ext}");
    }
    println!();

    println!("=== GL Extensions ({}) ===", gl_exts.len());
    for ext in &gl_exts {
        println!("  {ext}");
    }
}
