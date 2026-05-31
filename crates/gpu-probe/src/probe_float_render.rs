// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Float render-target capability probe — F16 / F32.
//!
//! Answers, per target, what the GPU supports for **rendering** half-float
//! (F16) and float (F32) color attachments and how the result can be moved
//! to an inference runtime with the fastest available API:
//!
//! * **Extensions** — `GL_EXT_color_buffer_float` / `_half_float`.
//! * **Internal-format FBO renderability** — `RGBA16F`, `R16F`, `RG16F`
//!   (F16) and `R32F`, `RG32F`, `RGBA32F` (F32) as texture-backed color
//!   attachments, validated with `glCheckFramebufferStatus`.
//! * **DMA-buf zero-copy float render** — import a float DMA-buf as an
//!   EGLImage-backed renderbuffer, render a known pattern, read it back and
//!   verify. Only F16 has a portable DRM fourcc (`DRM_FORMAT_ABGR16161616F`);
//!   there is **no** 32-bit-float DRM fourcc, so F32 dma-buf render is N/A.
//! * **PBO readback** — render to an internal float FBO, `glReadPixels` into
//!   a `GL_PIXEL_PACK_BUFFER`, map and verify; timed at representative packed
//!   dimensions so the two transfer mechanisms can be compared directly.
//!
//! The probe never panics: every GL/EGL step is guarded and reported as
//! PASS / FAIL / SKIP / N/A so a single driver quirk cannot abort the run.
//!
//! ## Layout context (why these dimensions)
//!
//! The HAL float-preprocessing target layouts are:
//! * **orin-nano (TensorRT, F16):** NCHW planar `[3,H,W]` packed 4 contiguous
//!   `f16` per RGBA16F texel → surface `(W/4, 3·H)`. For 640×640 that is
//!   `(160, 1920)` RGBA16F = 2,457,600 bytes.
//! * **rpi5-hailo (HailoRT, F32):** NHWC tightly-packed `[H,W,3]`. `RGB32F`
//!   is not color-renderable, so the candidate is a single-channel `R32F`
//!   surface `(W·3, H)` where the shader emits one channel per fragment. For
//!   640×640×3 that is `(1920, 640)` R32F = 4,915,200 bytes.

use crate::egl_context::GpuContext;
use edgefirst_tensor::{Tensor, TensorMemory, TensorTrait};
use std::os::unix::io::AsRawFd;
use std::ptr::null;
use std::time::Instant;

/// `DRM_FORMAT_ABGR16161616F` — the only float DRM fourcc usable as a
/// renderable RGBA16F dma-buf. Packed little-endian R,G,B,A half-floats.
const DRM_FORMAT_ABGR16161616F: u32 = 1_211_384_385;

/// `GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS` (0x8CD9).
const FRAMEBUFFER_INCOMPLETE_DIMENSIONS: u32 = 0x8CD9;

/// Iterations for the transfer-timing loops. Kept small so the probe stays
/// quick on slow embedded GPUs while still averaging out scheduler jitter.
const TIMING_ITERS: u32 = 50;

fn yn(b: bool) -> &'static str {
    if b {
        "YES"
    } else {
        "no"
    }
}

/// Drain and return the current GL error (0 == `GL_NO_ERROR`).
fn gl_err() -> u32 {
    unsafe { gls::gl::GetError() }
}

fn gl_err_name(e: u32) -> &'static str {
    match e {
        0 => "GL_NO_ERROR",
        0x0500 => "GL_INVALID_ENUM",
        0x0501 => "GL_INVALID_VALUE",
        0x0502 => "GL_INVALID_OPERATION",
        0x0505 => "GL_OUT_OF_MEMORY",
        0x0506 => "GL_INVALID_FRAMEBUFFER_OPERATION",
        _ => "GL_UNKNOWN",
    }
}

fn fbo_status_name(status: u32) -> &'static str {
    match status {
        gls::gl::FRAMEBUFFER_COMPLETE => "COMPLETE",
        gls::gl::FRAMEBUFFER_INCOMPLETE_ATTACHMENT => "INCOMPLETE_ATTACHMENT",
        gls::gl::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT => "INCOMPLETE_MISSING_ATTACHMENT",
        FRAMEBUFFER_INCOMPLETE_DIMENSIONS => "INCOMPLETE_DIMENSIONS",
        gls::gl::FRAMEBUFFER_UNSUPPORTED => "UNSUPPORTED",
        _ => "UNKNOWN",
    }
}

/// A renderable internal-format candidate.
struct FormatCase {
    /// Human label, e.g. `"RGBA16F"`.
    name: &'static str,
    /// GL internal format (e.g. `gls::gl::RGBA16F`).
    internal: u32,
    /// Client format for `TexImage2D` / `ReadPixels` (e.g. `gls::gl::RGBA`).
    format: u32,
    /// Client type (`HALF_FLOAT` or `FLOAT`).
    ty: u32,
    /// Bytes per texel for the readback buffer sizing.
    bpp: usize,
}

/// Create a texture-backed FBO with `case`'s internal format and report
/// whether the framebuffer is complete. Returns `(complete, gl_error)`.
///
/// # Safety
/// Requires a current GL context. Generates and deletes its own GL objects.
unsafe fn texture_fbo_complete(case: &FormatCase, w: i32, h: i32) -> (bool, u32) {
    let mut tex: u32 = 0;
    gls::gl::GenTextures(1, &mut tex);
    gls::gl::BindTexture(gls::gl::TEXTURE_2D, tex);
    gls::gl::TexParameteri(
        gls::gl::TEXTURE_2D,
        gls::gl::TEXTURE_MIN_FILTER,
        gls::gl::NEAREST as i32,
    );
    gls::gl::TexParameteri(
        gls::gl::TEXTURE_2D,
        gls::gl::TEXTURE_MAG_FILTER,
        gls::gl::NEAREST as i32,
    );
    gls::gl::TexImage2D(
        gls::gl::TEXTURE_2D,
        0,
        case.internal as i32,
        w,
        h,
        0,
        case.format,
        case.ty,
        null(),
    );
    let alloc_err = gl_err();

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
    let status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
    let complete = status == gls::gl::FRAMEBUFFER_COMPLETE;
    if !complete && alloc_err == 0 {
        log::debug!(
            "{}: FBO status {} ({:#x})",
            case.name,
            fbo_status_name(status),
            status
        );
    }

    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
    gls::gl::DeleteFramebuffers(1, &fbo);
    gls::gl::BindTexture(gls::gl::TEXTURE_2D, 0);
    gls::gl::DeleteTextures(1, &tex);

    (complete, alloc_err)
}

/// Render (clear) a known color to a texture-backed float FBO, read it back
/// into a PBO, map it and verify the first texel. Returns the per-iteration
/// transfer time (render + ReadPixels + map) averaged over `TIMING_ITERS`,
/// or `None` if any step fails.
///
/// # Safety
/// Requires a current GL context.
unsafe fn pbo_readback_timed(case: &FormatCase, w: i32, h: i32) -> Option<f64> {
    // Destination float texture + FBO.
    let mut tex: u32 = 0;
    gls::gl::GenTextures(1, &mut tex);
    gls::gl::BindTexture(gls::gl::TEXTURE_2D, tex);
    gls::gl::TexParameteri(
        gls::gl::TEXTURE_2D,
        gls::gl::TEXTURE_MIN_FILTER,
        gls::gl::NEAREST as i32,
    );
    gls::gl::TexParameteri(
        gls::gl::TEXTURE_2D,
        gls::gl::TEXTURE_MAG_FILTER,
        gls::gl::NEAREST as i32,
    );
    gls::gl::TexImage2D(
        gls::gl::TEXTURE_2D,
        0,
        case.internal as i32,
        w,
        h,
        0,
        case.format,
        case.ty,
        null(),
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
    if gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER) != gls::gl::FRAMEBUFFER_COMPLETE {
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
        gls::gl::DeleteFramebuffers(1, &fbo);
        gls::gl::DeleteTextures(1, &tex);
        return None;
    }

    // GLES 3.0 §4.3.2 only guarantees the implementation's preferred read pair.
    // Query it and prefer it over the nominal (format, type) when they differ so
    // strict drivers don't return GL_INVALID_OPERATION from glReadPixels.
    let mut impl_fmt: i32 = 0;
    let mut impl_ty: i32 = 0;
    gls::gl::GetIntegerv(gls::gl::IMPLEMENTATION_COLOR_READ_FORMAT, &mut impl_fmt);
    gls::gl::GetIntegerv(gls::gl::IMPLEMENTATION_COLOR_READ_TYPE, &mut impl_ty);
    let use_impl = impl_fmt != 0
        && impl_ty != 0
        && (impl_fmt as u32 != case.format || impl_ty as u32 != case.ty);
    let (rd_fmt, rd_ty, rd_bpp) = if use_impl {
        // Promote to the driver-preferred pair. Compute bytes-per-pixel for the
        // promoted (format, type) so the PBO is sized correctly.
        let comps = match impl_fmt as u32 {
            gls::gl::RED => 1,
            gls::gl::RG => 2,
            gls::gl::RGB => 3,
            _ => 4, // RGBA and anything else
        };
        let tsize = match impl_ty as u32 {
            gls::gl::FLOAT => 4,
            gls::gl::HALF_FLOAT => 2,
            gls::gl::UNSIGNED_BYTE => 1,
            _ => 4,
        };
        log::debug!(
            "{}: readback format promoted from ({:#x},{:#x}) to ({:#x},{:#x})",
            case.name,
            case.format,
            case.ty,
            impl_fmt as u32,
            impl_ty as u32,
        );
        (impl_fmt as u32, impl_ty as u32, comps * tsize)
    } else {
        (case.format, case.ty, case.bpp)
    };

    let byte_size = (w as usize) * (h as usize) * rd_bpp;

    // Pixel-pack buffer for the readback.
    let mut pbo: u32 = 0;
    gls::gl::GenBuffers(1, &mut pbo);
    gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, pbo);
    gls::gl::BufferData(
        gls::gl::PIXEL_PACK_BUFFER,
        byte_size as isize,
        null(),
        gls::gl::STREAM_READ,
    );
    gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);

    let mut verified = false;
    let mut total = 0.0_f64;
    let mut ok = true;

    for i in 0..TIMING_ITERS {
        let start = Instant::now();

        gls::gl::Viewport(0, 0, w, h);
        // Distinct per-iteration clear color so a stale-buffer read cannot
        // masquerade as a successful transfer.
        let c = 0.25 + (i % 4) as f32 * 0.1;
        gls::gl::ClearColor(c, 0.5, 0.75, 1.0);
        gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);

        gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, pbo);
        gls::gl::ReadPixels(0, 0, w, h, rd_fmt, rd_ty, null::<std::ffi::c_void>() as _);

        let map = gls::gl::MapBufferRange(
            gls::gl::PIXEL_PACK_BUFFER,
            0,
            byte_size as isize,
            gls::gl::MAP_READ_BIT,
        );
        if map.is_null() {
            ok = false;
            gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);
            break;
        }
        // Sanity-check the channel actually transferred. The clear color
        // encodes to multi-byte half/float words whose *low* byte can be
        // zero (e.g. half 0.25 == 0x3400 → bytes 00,34), so scan a span of
        // the first few texels for any non-zero byte rather than testing a
        // single byte.
        if i == 0 {
            let scan = byte_size.min(64);
            let bytes = std::slice::from_raw_parts(map as *const u8, scan);
            verified = bytes.iter().any(|&b| b != 0);
        }
        gls::gl::UnmapBuffer(gls::gl::PIXEL_PACK_BUFFER);
        gls::gl::BindBuffer(gls::gl::PIXEL_PACK_BUFFER, 0);

        total += start.elapsed().as_secs_f64() * 1000.0;
    }

    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
    gls::gl::DeleteFramebuffers(1, &fbo);
    gls::gl::DeleteTextures(1, &tex);
    gls::gl::DeleteBuffers(1, &pbo);

    if ok && verified {
        Some(total / TIMING_ITERS as f64)
    } else {
        None
    }
}

/// Outcome of the F16 dma-buf render attempt, disambiguating the stage that
/// failed so a missing dma-heap (allocation) is not confused with a driver
/// that cannot import/render an RGBA16F dma-buf.
enum DmaF16Status {
    /// No `glEGLImageTargetRenderbufferStorageOES` entry point.
    NoRenderbufferExt,
    /// DMA-buf allocation failed (e.g. no `/dev/dma_heap`).
    AllocFailed(String),
    /// `eglCreateImage` rejected the RGBA16F dma-buf import.
    EglImportFailed(String),
    /// EGLImage imported but the FBO was incomplete.
    FboIncomplete(String),
    /// Rendered, but the readback came back all-zero (silent driver trap).
    ReadbackEmpty,
    /// Full render + readback round-trip succeeded.
    Ok,
}

impl DmaF16Status {
    fn verified(&self) -> bool {
        matches!(self, DmaF16Status::Ok)
    }
    fn report(&self) -> String {
        match self {
            DmaF16Status::Ok => "PASS (rendered + read back)".into(),
            DmaF16Status::ReadbackEmpty => "PARTIAL (FBO complete, readback empty)".into(),
            DmaF16Status::FboIncomplete(s) => format!("FAIL (FBO {s})"),
            DmaF16Status::EglImportFailed(s) => format!("FAIL (EGLImage import rejected: {s})"),
            DmaF16Status::AllocFailed(s) => format!("N/A (DMA alloc failed: {s})"),
            DmaF16Status::NoRenderbufferExt => {
                "N/A (no glEGLImageTargetRenderbufferStorageOES)".into()
            }
        }
    }
}

/// Render (clear) into an F16 dma-buf via an EGLImage-backed renderbuffer and
/// read it back to confirm the float dma-buf render path works end to end.
///
/// # Safety
/// Requires a current GL context.
unsafe fn dmabuf_f16_render(ctx: &GpuContext, w: u32, h: u32) -> DmaF16Status {
    if !ctx.has_rgb8_renderbuffer() {
        // Same entry point (glEGLImageTargetRenderbufferStorageOES) gates this.
        return DmaF16Status::NoRenderbufferExt;
    }
    let bpp: u32 = 8; // RGBA16F
    let pitch = w * bpp;
    let byte_count = (pitch * h) as usize;

    let tensor = match Tensor::<u8>::new(&[byte_count], Some(TensorMemory::Dma), None) {
        Ok(t) => t,
        Err(e) => return DmaF16Status::AllocFailed(e.to_string()),
    };
    let fd = match tensor.clone_fd() {
        Ok(fd) => fd,
        Err(e) => return DmaF16Status::AllocFailed(format!("clone_fd: {e}")),
    };

    let egl_image = match ctx.create_egl_image_dma(
        fd.as_raw_fd(),
        w as i32,
        h as i32,
        DRM_FORMAT_ABGR16161616F,
        pitch as i32,
    ) {
        Ok(img) => img,
        Err(e) => return DmaF16Status::EglImportFailed(e),
    };

    let mut rbo: u32 = 0;
    gls::gl::GenRenderbuffers(1, &mut rbo);
    gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, rbo);
    gls::gl::EGLImageTargetRenderbufferStorageOES(gls::gl::RENDERBUFFER, egl_image.as_ptr());

    let mut fbo: u32 = 0;
    gls::gl::GenFramebuffers(1, &mut fbo);
    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
    gls::gl::FramebufferRenderbuffer(
        gls::gl::FRAMEBUFFER,
        gls::gl::COLOR_ATTACHMENT0,
        gls::gl::RENDERBUFFER,
        rbo,
    );

    let status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
    let complete = status == gls::gl::FRAMEBUFFER_COMPLETE;
    let mut verified = false;

    if complete {
        gls::gl::Viewport(0, 0, w as i32, h as i32);
        gls::gl::ClearColor(0.5, 0.25, 0.75, 1.0);
        gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
        gls::gl::Finish();

        // Read back through the FBO as RGBA16F to confirm the render landed
        // (guards against the silent all-zeros EGLImage trap some drivers hit).
        let mut buf = vec![0u8; (w * h * bpp) as usize];
        gls::gl::ReadPixels(
            0,
            0,
            w as i32,
            h as i32,
            gls::gl::RGBA,
            gls::gl::HALF_FLOAT,
            buf.as_mut_ptr() as *mut std::ffi::c_void,
        );
        verified = buf.iter().any(|&b| b != 0);
    }

    gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
    gls::gl::DeleteFramebuffers(1, &fbo);
    gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, 0);
    gls::gl::DeleteRenderbuffers(1, &rbo);
    let _ = ctx.destroy_egl_image(egl_image);

    if !complete {
        DmaF16Status::FboIncomplete(format!("{} {:#x}", fbo_status_name(status), status))
    } else if verified {
        DmaF16Status::Ok
    } else {
        DmaF16Status::ReadbackEmpty
    }
}

/// Run the float render-target capability probe and print a structured report
/// plus a single machine-greppable `VERDICT` line.
pub fn run(ctx: &GpuContext) {
    println!("=== Float Render-Target Probe (F16 / F32) ===");
    println!("  renderer: {}", ctx.gl_renderer());
    println!("  version : {}", ctx.gl_version());

    // --- Extensions ---
    let gl_exts = ctx.gl_extensions();
    let has_f32_ext = gl_exts.iter().any(|e| e == "GL_EXT_color_buffer_float");
    let has_f16_ext = gl_exts
        .iter()
        .any(|e| e == "GL_EXT_color_buffer_half_float");
    println!("  {:38} {}", "GL_EXT_color_buffer_float", yn(has_f32_ext));
    println!(
        "  {:38} {}",
        "GL_EXT_color_buffer_half_float",
        yn(has_f16_ext)
    );
    println!();

    // --- Internal-format FBO renderability (texture-backed) ---
    let cases = [
        FormatCase {
            name: "RGBA16F",
            internal: gls::gl::RGBA16F,
            format: gls::gl::RGBA,
            ty: gls::gl::HALF_FLOAT,
            bpp: 8,
        },
        FormatCase {
            name: "R16F",
            internal: gls::gl::R16F,
            format: gls::gl::RED,
            ty: gls::gl::HALF_FLOAT,
            bpp: 2,
        },
        FormatCase {
            name: "RG16F",
            internal: gls::gl::RG16F,
            format: gls::gl::RG,
            ty: gls::gl::HALF_FLOAT,
            bpp: 4,
        },
        FormatCase {
            name: "R32F",
            internal: gls::gl::R32F,
            format: gls::gl::RED,
            ty: gls::gl::FLOAT,
            bpp: 4,
        },
        FormatCase {
            name: "RG32F",
            internal: gls::gl::RG32F,
            format: gls::gl::RG,
            ty: gls::gl::FLOAT,
            bpp: 8,
        },
        FormatCase {
            name: "RGBA32F",
            internal: gls::gl::RGBA32F,
            format: gls::gl::RGBA,
            ty: gls::gl::FLOAT,
            bpp: 16,
        },
    ];

    println!("  -- Texture-backed FBO renderability (256x256) --");
    let mut renderable: std::collections::BTreeMap<&str, bool> = Default::default();
    for case in &cases {
        let _ = gl_err(); // clear stale error
        let (complete, err) = unsafe { texture_fbo_complete(case, 256, 256) };
        renderable.insert(case.name, complete);
        let note = if err != 0 {
            format!("FAIL ({})", gl_err_name(err))
        } else if complete {
            "PASS".to_string()
        } else {
            "FAIL (FBO incomplete)".to_string()
        };
        println!("  {:38} {}", case.name, note);
    }
    println!();

    // --- DMA-buf float render ---
    println!("  -- DMA-buf zero-copy float render --");
    let _ = gl_err();
    let f16_dma = unsafe { dmabuf_f16_render(ctx, 256, 256) };
    let f16_dma_verified = f16_dma.verified();
    println!(
        "  {:38} {}",
        "F16 RGBA16F (DRM_FORMAT_ABGR16161616F)",
        f16_dma.report()
    );
    println!(
        "  {:38} N/A (no 32-bit-float DRM format exists)",
        "F32 (any DRM fourcc)"
    );
    println!();

    // --- PBO readback timing at representative packed dims ---
    println!("  -- PBO readback (render -> ReadPixels -> map), timed --");
    // F16 NCHW-packed for 640x640x3: (W/4, 3H) RGBA16F = (160, 1920).
    let f16_case = &cases[0]; // RGBA16F
    let f16_pbo = if *renderable.get("RGBA16F").unwrap_or(&false) {
        unsafe { pbo_readback_timed(f16_case, 160, 1920) }
    } else {
        None
    };
    // F32 NHWC-tight for 640x640x3: (W*3, H) R32F = (1920, 640).
    let f32_case = &cases[3]; // R32F
    let f32_pbo = if *renderable.get("R32F").unwrap_or(&false) {
        unsafe { pbo_readback_timed(f32_case, 1920, 640) }
    } else {
        None
    };
    match f16_pbo {
        Some(ms) => println!(
            "  {:38} {:.3} ms/frame (640x640x3 f16 NCHW)",
            "F16 RGBA16F PBO", ms
        ),
        None => println!("  {:38} unsupported", "F16 RGBA16F PBO"),
    }
    match f32_pbo {
        Some(ms) => println!(
            "  {:38} {:.3} ms/frame (640x640x3 f32 NHWC)",
            "F32 R32F PBO", ms
        ),
        None => println!("  {:38} unsupported", "F32 R32F PBO"),
    }
    println!();

    // --- Verdict (machine-greppable) ---
    let f16_fbo = *renderable.get("RGBA16F").unwrap_or(&false);
    let f32_fbo = *renderable.get("R32F").unwrap_or(&false);
    let f16_fastest = if f16_dma_verified {
        "dmabuf".to_string()
    } else if let Some(ms) = f16_pbo {
        format!("pbo({ms:.2}ms)")
    } else {
        "none".to_string()
    };
    let f32_fastest = match f32_pbo {
        // F32 dma-buf is impossible (no fourcc), so PBO is the ceiling.
        Some(ms) => format!("pbo({ms:.2}ms)"),
        None => "none".to_string(),
    };
    println!(
        "VERDICT renderer=\"{}\" f16[ext={} fbo={} dmabuf={} pbo={}] f32[ext={} fbo={} dmabuf=NA pbo={}] fastest_f16={} fastest_f32={}",
        ctx.gl_renderer(),
        yn(has_f16_ext),
        yn(f16_fbo),
        yn(f16_dma_verified),
        f16_pbo.map(|m| format!("{m:.2}ms")).unwrap_or_else(|| "no".into()),
        yn(has_f32_ext),
        yn(f32_fbo),
        f32_pbo.map(|m| format!("{m:.2}ms")).unwrap_or_else(|| "no".into()),
        f16_fastest,
        f32_fastest,
    );
    println!();
}
