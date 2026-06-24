// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NV12/NV16/NV24 DMA-buf import + render-surface probe.
//!
//! This probe answers the question rule #4 poses: *can we create the proper
//! textures and render surfaces with DMA-buf backing, without errors, for the
//! exact allocations the HAL produces?* It is the ground-truth isolation
//! harness that decouples "is the HAL allocation GPU-importable" from any
//! higher-level `convert()` behaviour.
//!
//! For every `fmt ∈ {NV12, NV16, NV24}` × a dimension matrix of realistic
//! QVGA-and-up sizes that includes odd *logical* sizes (321×240, 321×241,
//! 789×384 — sub-QVGA sizes are avoided because Mali rejects below-minimum
//! textures, which would confound "odd" with "too small") it:
//!
//!   1. Allocates the source via the **real** allocator
//!      ([`Tensor::image`]) so the internal row stride is the production
//!      even/64-aligned value (rule #1) — odd dims live only in the logical
//!      width, never in the buffer stride.
//!   2. Rebuilds the EGL import attributes with the **same formulas** as the
//!      HAL's `DmaImportAttrs`:
//!        * **Path B** — single-plane R8 of `(stride, combined_h)`, sampled
//!          with `texelFetch` (mirrors `from_tensor_nv_r8`).
//!        * **Path A** — native 2-plane YUV (`DrmFourcc::Nv12/16/24`) sampled
//!          via `samplerExternalOES` (mirrors `from_tensor`). Native NV16/NV24
//!          is exploratory — the HAL is R8-only for those today.
//!   3. Imports the source, binds it as a texture, creates a destination RGBA
//!      render surface, runs a real draw, and reads back a pixel — proving the
//!      whole texture→FBO→render→readback path, not just EGLImage creation.
//!   4. Repeats each case with a **64-byte-aligned** destination stride and a
//!      **tightly-packed** destination stride, so per-GPU destination alignment
//!      limitations surface as their own column.
//!
//! The output is a per-platform PASS/FAIL matrix that can be diffed against
//! `GPU_R8_ODD.md` across imx8mp-frdm / imx95-frdm / rpi5-hailo / orin-nano.

use crate::bench_render::{compile_program, create_quad_vao, VERTEX_SRC};
use crate::egl_context::{EglPlane, GpuContext};
use edgefirst_tensor::{PixelFormat, Tensor, TensorMemory, TensorTrait};
use std::ffi::CString;
use std::os::fd::{AsRawFd, OwnedFd};

/// Path B fragment shader: sample the combined-plane R8 import with
/// `texelFetch` — the exact operation strict tiled GPUs (Mali, V3D) enforce
/// bounds on, and what the real `generate_nv_to_rgba_shader_2d` uses.
const R8_FRAGMENT_SRC: &str = "\
#version 300 es
precision highp float;
precision highp int;
uniform sampler2D tex;
in vec2 tc;
out vec4 color;
void main() {
    ivec2 sz = textureSize(tex, 0);
    ivec2 p = clamp(ivec2(gl_FragCoord.xy), ivec2(0), sz - 1);
    float v = float(texelFetch(tex, p, 0).r);
    color = vec4(v, v, v, 1.0);
}
";

/// Path A fragment shader: sample the native YUV import via
/// `samplerExternalOES` (driver does the YUV→RGB). Mirrors the HAL Path A
/// shaders which sample with `texture()` (texelFetch is illegal on external).
const EXTERNAL_FRAGMENT_SRC: &str = "\
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision highp float;
uniform samplerExternalOES tex;
in vec2 tc;
out vec4 color;
void main() { color = texture(tex, tc); }
";

/// Which source import path a case exercises.
#[derive(Clone, Copy, PartialEq)]
enum SrcPath {
    /// Single-plane R8 combined import (`texelFetch`, `TEXTURE_2D`).
    R8,
    /// Native 2-plane YUV import (`samplerExternalOES`, `TEXTURE_EXTERNAL_OES`).
    NativeYuv,
}

/// A pixel format under test plus its DRM fourcc for native import.
struct NvFormat {
    name: &'static str,
    fmt: PixelFormat,
    fourcc: u32,
}

/// Compiled programs + shared VAO reused across all cases.
struct Programs {
    r8: u32,
    r8_tex_loc: i32,
    external: Option<u32>,
    external_tex_loc: i32,
    vao: u32,
}

/// Combined (luma + chroma) buffer height in stride-wide rows — delegates to
/// the canonical [`PixelFormat::combined_plane_height`] (falls back to `h` for
/// non-semi-planar formats, which this probe never feeds it).
fn combined_height(fmt: PixelFormat, h: usize) -> usize {
    fmt.combined_plane_height(h).unwrap_or(h)
}

/// Native 2-plane chroma pitch (bytes/row) for the given luma stride —
/// matches the DRM semi-planar layout the driver expects. Derives from the
/// canonical [`PixelFormat::chroma_layout`] (`uv_rows_per_luma`): NV24's
/// full-resolution interleaved CbCr is `2*luma_stride`, NV12/NV16 is one
/// `luma_stride` row.
fn native_chroma_pitch(fmt: PixelFormat, luma_stride: usize) -> usize {
    fmt.chroma_layout()
        .map(|c| luma_stride * c.uv_rows_per_luma)
        .unwrap_or(luma_stride)
}

/// Query whether the current GL context advertises an extension.
fn gl_has_extension(name: &str) -> bool {
    unsafe {
        let mut count: i32 = 0;
        edgefirst_gl::gl::GetIntegerv(edgefirst_gl::gl::NUM_EXTENSIONS, &mut count);
        for i in 0..count {
            let ptr = edgefirst_gl::gl::GetStringi(edgefirst_gl::gl::EXTENSIONS, i as u32);
            if ptr.is_null() {
                continue;
            }
            let ext = std::ffi::CStr::from_ptr(ptr.cast());
            if ext.to_bytes() == name.as_bytes() {
                return true;
            }
        }
        false
    }
}

/// Drain and return the first pending GL error as a readable string, if any.
fn gl_error(stage: &str) -> Result<(), String> {
    let err = unsafe { edgefirst_gl::gl::GetError() };
    if err == edgefirst_gl::gl::NO_ERROR {
        return Ok(());
    }
    let name = match err {
        edgefirst_gl::gl::INVALID_ENUM => "GL_INVALID_ENUM",
        edgefirst_gl::gl::INVALID_VALUE => "GL_INVALID_VALUE",
        edgefirst_gl::gl::INVALID_OPERATION => "GL_INVALID_OPERATION",
        edgefirst_gl::gl::INVALID_FRAMEBUFFER_OPERATION => "GL_INVALID_FRAMEBUFFER_OPERATION",
        edgefirst_gl::gl::OUT_OF_MEMORY => "GL_OUT_OF_MEMORY",
        other => return Err(format!("{stage}: GL error 0x{other:04X}")),
    };
    // Clear any further queued errors so they don't leak into the next case.
    while unsafe { edgefirst_gl::gl::GetError() } != edgefirst_gl::gl::NO_ERROR {}
    Err(format!("{stage}: {name}"))
}

/// Run a single import→texture→FBO→draw→readback case.
///
/// Returns `Ok(())` if every stage succeeds with no EGL/GL error, otherwise
/// `Err("<stage>: <detail>")`.
#[allow(clippy::too_many_arguments)]
fn run_case(
    ctx: &GpuContext,
    progs: &Programs,
    path: SrcPath,
    f: &NvFormat,
    w: usize,
    h: usize,
    src_fd: i32,
    stride: usize,
    plane_offset: usize,
    dst_fd: i32,
    dst_w: usize,
    dst_h: usize,
    dst_stride: usize,
) -> Result<(), String> {
    // Drain any stale error from a previous case.
    while unsafe { edgefirst_gl::gl::GetError() } != edgefirst_gl::gl::NO_ERROR {}

    // --- 1. Source EGLImage (the allocation under test) -------------------
    let (src_img, tex_target, program, tex_loc) = match path {
        SrcPath::R8 => {
            let combined_h = combined_height(f.fmt, h);
            let img = ctx
                .create_egl_image_planar(
                    stride as i32,
                    combined_h as i32,
                    gbm::drm::buffer::DrmFourcc::R8 as u32,
                    EglPlane {
                        fd: src_fd,
                        offset: plane_offset as i32,
                        pitch: stride as i32,
                    },
                    None,
                    false,
                )
                .map_err(|e| format!("eglCreateImage(src R8 {stride}x{combined_h}): {e}"))?;
            (
                img,
                edgefirst_gl::gl::TEXTURE_2D,
                progs.r8,
                progs.r8_tex_loc,
            )
        }
        SrcPath::NativeYuv => {
            let program = progs
                .external
                .ok_or_else(|| "GL_OES_EGL_image_external_essl3 absent".to_string())?;
            let p1 = EglPlane {
                fd: src_fd,
                offset: (plane_offset + stride * h) as i32,
                pitch: native_chroma_pitch(f.fmt, stride) as i32,
            };
            let img = ctx
                .create_egl_image_planar(
                    w as i32,
                    h as i32,
                    f.fourcc,
                    EglPlane {
                        fd: src_fd,
                        offset: plane_offset as i32,
                        pitch: stride as i32,
                    },
                    Some(p1),
                    true,
                )
                .map_err(|e| format!("eglCreateImage(src {} {w}x{h}): {e}", f.name))?;
            (
                img,
                edgefirst_gl::gl::TEXTURE_EXTERNAL_OES,
                program,
                progs.external_tex_loc,
            )
        }
    };

    // --- 2. Destination RGBA EGLImage (the render surface under test) -----
    let dst_img = match ctx.create_egl_image_dma(
        dst_fd,
        dst_w as i32,
        dst_h as i32,
        gbm::drm::buffer::DrmFourcc::Abgr8888 as u32,
        dst_stride as i32,
    ) {
        Ok(img) => img,
        Err(e) => {
            let _ = ctx.destroy_egl_image(src_img);
            return Err(format!(
                "eglCreateImage(dst RGBA {dst_w}x{dst_h}@{dst_stride}): {e}"
            ));
        }
    };

    // --- 3-7. Bind textures, build FBO, draw, read back -------------------
    let result = (|| -> Result<(), String> {
        unsafe {
            // Source texture.
            let mut src_tex = 0u32;
            edgefirst_gl::gl::GenTextures(1, &mut src_tex);
            edgefirst_gl::gl::BindTexture(tex_target, src_tex);
            edgefirst_gl::gl::EGLImageTargetTexture2DOES(tex_target, src_img.as_ptr());
            // R8 texelFetch wants NEAREST + no mips; external wants LINEAR.
            let filter = if path == SrcPath::R8 {
                edgefirst_gl::gl::NEAREST
            } else {
                edgefirst_gl::gl::LINEAR
            } as i32;
            edgefirst_gl::gl::TexParameteri(
                tex_target,
                edgefirst_gl::gl::TEXTURE_MIN_FILTER,
                filter,
            );
            edgefirst_gl::gl::TexParameteri(
                tex_target,
                edgefirst_gl::gl::TEXTURE_MAG_FILTER,
                filter,
            );
            edgefirst_gl::gl::TexParameteri(
                tex_target,
                edgefirst_gl::gl::TEXTURE_WRAP_S,
                edgefirst_gl::gl::CLAMP_TO_EDGE as i32,
            );
            edgefirst_gl::gl::TexParameteri(
                tex_target,
                edgefirst_gl::gl::TEXTURE_WRAP_T,
                edgefirst_gl::gl::CLAMP_TO_EDGE as i32,
            );
            gl_error("bind src texture")?;

            // Destination texture + FBO.
            let mut dst_tex = 0u32;
            edgefirst_gl::gl::GenTextures(1, &mut dst_tex);
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, dst_tex);
            edgefirst_gl::gl::EGLImageTargetTexture2DOES(
                edgefirst_gl::gl::TEXTURE_2D,
                dst_img.as_ptr(),
            );
            gl_error("bind dst texture")?;

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
            let status = edgefirst_gl::gl::CheckFramebufferStatus(edgefirst_gl::gl::FRAMEBUFFER);
            if status != edgefirst_gl::gl::FRAMEBUFFER_COMPLETE {
                edgefirst_gl::gl::DeleteFramebuffers(1, &fbo);
                edgefirst_gl::gl::DeleteTextures(1, &dst_tex);
                edgefirst_gl::gl::DeleteTextures(1, &src_tex);
                return Err(format!("FBO incomplete: status 0x{status:04X}"));
            }

            // Draw.
            edgefirst_gl::gl::Viewport(0, 0, dst_w as i32, dst_h as i32);
            edgefirst_gl::gl::UseProgram(program);
            edgefirst_gl::gl::ActiveTexture(edgefirst_gl::gl::TEXTURE0);
            edgefirst_gl::gl::BindTexture(tex_target, src_tex);
            edgefirst_gl::gl::Uniform1i(tex_loc, 0);
            edgefirst_gl::gl::BindVertexArray(progs.vao);
            edgefirst_gl::gl::DrawElements(
                edgefirst_gl::gl::TRIANGLES,
                6,
                edgefirst_gl::gl::UNSIGNED_INT,
                std::ptr::null(),
            );
            edgefirst_gl::gl::Finish();
            gl_error("draw")?;

            // Read back one pixel — proves the render surface is actually
            // readable, catching write/layout failures EGLImage creation alone
            // would not surface.
            let mut px = [0u8; 4];
            edgefirst_gl::gl::ReadPixels(
                0,
                0,
                1,
                1,
                edgefirst_gl::gl::RGBA,
                edgefirst_gl::gl::UNSIGNED_BYTE,
                px.as_mut_ptr().cast(),
            );
            let read_err = gl_error("readback");

            edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, 0);
            edgefirst_gl::gl::BindVertexArray(0);
            edgefirst_gl::gl::DeleteFramebuffers(1, &fbo);
            edgefirst_gl::gl::DeleteTextures(1, &dst_tex);
            edgefirst_gl::gl::DeleteTextures(1, &src_tex);
            read_err
        }
    })();

    let _ = ctx.destroy_egl_image(dst_img);
    let _ = ctx.destroy_egl_image(src_img);
    result
}

/// Allocate an NV source and an RGBA destination, returning the fds + geometry.
/// Keeps the `OwnedFd`s alive for the caller's case duration.
struct Buffers {
    _src: Tensor<u8>,
    _dst: Tensor<u8>,
    src_fd: OwnedFd,
    dst_fd: OwnedFd,
    stride: usize,
    plane_offset: usize,
    dst_w: usize,
    dst_h: usize,
    dst_stride: usize,
}

/// Allocate the source NV buffer (real padded stride) and a destination RGBA
/// buffer with either a 64-aligned or a tightly-packed row stride.
fn alloc_buffers(f: &NvFormat, w: usize, h: usize, dst_aligned: bool) -> Result<Buffers, String> {
    let src = Tensor::<u8>::image(w, h, f.fmt, Some(TensorMemory::Dma))
        .map_err(|e| format!("src alloc: {e}"))?;
    let stride = src
        .effective_row_stride()
        .ok_or_else(|| "src has no row stride".to_string())?;
    let plane_offset = src.plane_offset().unwrap_or(0);

    // Destination is the render target — even dims only (rule #2). The point of
    // interest is the *stride*: 64-aligned vs the minimum packed width*4.
    let dst_w = w.next_multiple_of(2);
    let dst_h = h.next_multiple_of(2);
    let packed = dst_w * 4;
    let dst_stride = if dst_aligned {
        packed.next_multiple_of(64)
    } else {
        packed
    };
    let dst = Tensor::<u8>::image_with_stride(
        dst_w,
        dst_h,
        PixelFormat::Rgba,
        dst_stride,
        Some(TensorMemory::Dma),
    )
    .map_err(|e| format!("dst alloc (stride {dst_stride}): {e}"))?;

    let src_fd = src.clone_fd().map_err(|e| format!("src clone_fd: {e}"))?;
    let dst_fd = dst.clone_fd().map_err(|e| format!("dst clone_fd: {e}"))?;

    Ok(Buffers {
        _src: src,
        _dst: dst,
        src_fd,
        dst_fd,
        stride,
        plane_offset,
        dst_w,
        dst_h,
        dst_stride,
    })
}

/// Render a PASS/FAIL cell, collecting the detail for failures.
fn cell(ok: &Result<(), String>) -> &'static str {
    match ok {
        Ok(()) => "ok",
        Err(_) => "FAIL",
    }
}

/// Entry point: run the full NV DMA-buf import + render probe.
pub fn run(ctx: &GpuContext) {
    println!("=== NV DMA-buf Import + Render Probe (NV12/NV16/NV24) ===");

    if !ctx.has_egl_create_image_khr() {
        println!("  SKIP: EGL_EXT_image_dma_buf_import not available");
        println!();
        return;
    }

    // Compile the two source-sampling programs once.
    let vert = match CString::new(VERTEX_SRC) {
        Ok(c) => c,
        Err(_) => {
            println!("  SKIP: vertex shader has interior NUL");
            println!();
            return;
        }
    };
    let r8_frag = CString::new(R8_FRAGMENT_SRC).unwrap();
    let r8 = match compile_program(&vert, &r8_frag) {
        Ok(p) => p,
        Err(e) => {
            println!("  SKIP: R8 program compile failed: {e}");
            println!();
            return;
        }
    };
    let r8_tex_loc =
        unsafe { edgefirst_gl::gl::GetUniformLocation(r8, CString::new("tex").unwrap().as_ptr()) };

    let has_external = gl_has_extension("GL_OES_EGL_image_external_essl3");
    let (external, external_tex_loc) = if has_external {
        let frag = CString::new(EXTERNAL_FRAGMENT_SRC).unwrap();
        match compile_program(&vert, &frag) {
            Ok(p) => {
                let loc = unsafe {
                    edgefirst_gl::gl::GetUniformLocation(p, CString::new("tex").unwrap().as_ptr())
                };
                (Some(p), loc)
            }
            Err(e) => {
                println!("  note: external program compile failed ({e}); Path A skipped");
                (None, -1)
            }
        }
    } else {
        (None, -1)
    };

    let progs = Programs {
        r8,
        r8_tex_loc,
        external,
        external_tex_loc,
        vao: create_quad_vao().0,
    };

    println!(
        "  GL_OES_EGL_image_external_essl3: {}",
        if has_external {
            "present"
        } else {
            "ABSENT (Path A native-YUV skipped)"
        }
    );
    println!("  Path B = R8 texelFetch (TEXTURE_2D); Path A = native YUV (samplerExternalOES)");
    println!("  dst64 = 64-aligned dst stride; dstPk = packed (width*4) dst stride");
    println!();

    let formats = [
        NvFormat {
            name: "NV12",
            fmt: PixelFormat::Nv12,
            fourcc: gbm::drm::buffer::DrmFourcc::Nv12 as u32,
        },
        NvFormat {
            name: "NV16",
            fmt: PixelFormat::Nv16,
            fourcc: gbm::drm::buffer::DrmFourcc::Nv16 as u32,
        },
        NvFormat {
            name: "NV24",
            fmt: PixelFormat::Nv24,
            fourcc: gbm::drm::buffer::DrmFourcc::Nv24 as u32,
        },
    ];

    // (w, h, label). Realistic QVGA-and-up sizes only: Mali rejects EGLImages
    // below a minimum size (see probe_min_sizes), so sub-QVGA tests would
    // confound "odd-dimension" failures with "too-small" failures. Odd variants
    // are QVGA ±1 so the odd axis is isolated at a realistic scale.
    let dims: &[(usize, usize, &str)] = &[
        (320, 240, "320x240 QVGA even"),
        (321, 240, "321x240 odd-W"),
        (321, 241, "321x241 odd-both"),
        (789, 384, "789x384 real odd"),
        (1920, 1080, "1920x1080"),
    ];

    for f in &formats {
        println!("  {}:", f.name);
        println!(
            "    {:<16} {:>6} | {:<8} {:<8} {:<8} {:<8}",
            "dim", "stride", "B/dst64", "B/dstPk", "A/dst64", "A/dstPk"
        );
        let mut failures: Vec<(String, String)> = Vec::new();

        for &(w, h, label) in dims {
            // Four cells: {Path B, Path A} × {dst 64-aligned, dst packed}.
            let mut cells = ["----"; 4];
            let mut stride_str = String::from("?");

            for (col, (path, aligned)) in [
                (SrcPath::R8, true),
                (SrcPath::R8, false),
                (SrcPath::NativeYuv, true),
                (SrcPath::NativeYuv, false),
            ]
            .into_iter()
            .enumerate()
            {
                if path == SrcPath::NativeYuv && progs.external.is_none() {
                    cells[col] = "skip";
                    continue;
                }
                let bufs = match alloc_buffers(f, w, h, aligned) {
                    Ok(b) => b,
                    Err(e) => {
                        cells[col] = "ALLOC";
                        failures.push((format!("{label} col{col}"), e));
                        continue;
                    }
                };
                stride_str = bufs.stride.to_string();
                let r = run_case(
                    ctx,
                    &progs,
                    path,
                    f,
                    w,
                    h,
                    bufs.src_fd.as_raw_fd(),
                    bufs.stride,
                    bufs.plane_offset,
                    bufs.dst_fd.as_raw_fd(),
                    bufs.dst_w,
                    bufs.dst_h,
                    bufs.dst_stride,
                );
                cells[col] = cell(&r);
                if let Err(detail) = r {
                    let pathn = if path == SrcPath::R8 { "B" } else { "A" };
                    let dstn = if aligned { "dst64" } else { "dstPk" };
                    failures.push((format!("{label} {pathn}/{dstn}"), detail));
                }
            }

            println!(
                "    {:<16} {:>6} | {:<8} {:<8} {:<8} {:<8}",
                label, stride_str, cells[0], cells[1], cells[2], cells[3]
            );
        }

        if !failures.is_empty() {
            // Collapse identical details (e.g. a single permission/alloc error
            // repeated across all cells) to one line with a representative
            // location and a count, so genuine per-cell rejections stand out.
            println!("    failures:");
            let mut seen: Vec<(String, String, usize)> = Vec::new();
            for (where_, detail) in &failures {
                if let Some(e) = seen.iter_mut().find(|(_, d, _)| d == detail) {
                    e.2 += 1;
                } else {
                    seen.push((where_.clone(), detail.clone(), 1));
                }
            }
            for (where_, detail, count) in &seen {
                if *count > 1 {
                    println!("      [{where_} ×{count}] {detail}");
                } else {
                    println!("      [{where_}] {detail}");
                }
            }
        }
        println!();
    }

    // Rule #2 ground truth: which platforms accept an ODD-dimension
    // destination render surface (independent of the NV source path).
    probe_odd_destination(ctx);

    // Cleanup shared programs / VAO.
    unsafe {
        edgefirst_gl::gl::DeleteProgram(progs.r8);
        if let Some(p) = progs.external {
            edgefirst_gl::gl::DeleteProgram(p);
        }
        edgefirst_gl::gl::DeleteVertexArrays(1, &progs.vao);
    }
}

/// Probe whether an **odd-dimension destination** render surface can be created
/// and rendered to. This is the ground truth for rule #2's
/// known-unsupported-platforms list: a `convert()` destination is RGBA with a
/// 64-aligned stride, so we allocate exactly that at odd logical width/height,
/// build an EGLImage + FBO, clear it, and read back a pixel. Independent of the
/// NV source path — an odd dst either works on a GPU or it doesn't.
fn probe_odd_destination(ctx: &GpuContext) {
    println!("  Odd-destination render surfaces (RGBA, 64-aligned stride):");
    // Realistic QVGA-and-up sizes (Mali rejects sub-minimum EGLImages).
    let cases: &[(usize, usize, &str)] = &[
        (320, 240, "320x240 QVGA even"),
        (321, 240, "321x240 odd-W"),
        (320, 241, "320x241 odd-H"),
        (321, 241, "321x241 odd-both"),
        (789, 384, "789x384 odd-W real"),
    ];
    println!(
        "    {:<22} {:<10} {:<10}",
        "case", "preserved", "no-preserve"
    );
    for &(w, h, label) in cases {
        // `preserved` mirrors the HAL convert dst (to_egl_attribs sets
        // EGL_IMAGE_PRESERVED=TRUE); `no-preserve` mirrors a plain render target.
        let with = run_odd_dst_case(ctx, w, h, true);
        let without = run_odd_dst_case(ctx, w, h, false);
        let fmt = |r: &Result<(), String>| match r {
            Ok(()) => "ok".to_string(),
            Err(_) => "FAIL".to_string(),
        };
        println!("    {label:<22} {:<10} {:<10}", fmt(&with), fmt(&without));
        for (tag, r) in [("preserved", &with), ("no-preserve", &without)] {
            if let Err(e) = r {
                println!("      [{label} {tag}] {e}");
            }
        }
    }
    println!();
}

/// Create an odd-dimension RGBA dst, attach to an FBO, clear, and read back —
/// proving the render surface is usable. Returns `Err("<stage>: <detail>")`.
///
/// `preserved` toggles `EGL_IMAGE_PRESERVED=TRUE`: the HAL convert dst path
/// (`to_egl_attribs`) sets it; a plain render target does not. This isolates
/// whether IMAGE_PRESERVED is what makes Mali reject odd-dimension dst imports.
fn run_odd_dst_case(ctx: &GpuContext, w: usize, h: usize, preserved: bool) -> Result<(), String> {
    while unsafe { edgefirst_gl::gl::GetError() } != edgefirst_gl::gl::NO_ERROR {}

    let stride = (w * 4).next_multiple_of(64);
    let dst =
        Tensor::<u8>::image_with_stride(w, h, PixelFormat::Rgba, stride, Some(TensorMemory::Dma))
            .map_err(|e| format!("dst alloc: {e}"))?;
    let dst_fd = dst.clone_fd().map_err(|e| format!("clone_fd: {e}"))?;
    let fourcc = gbm::drm::buffer::DrmFourcc::Abgr8888 as u32;
    let plane0 = EglPlane {
        fd: dst_fd.as_raw_fd(),
        offset: 0,
        pitch: stride as i32,
    };

    // `create_egl_image_planar` sets IMAGE_PRESERVED (like the HAL convert dst);
    // `create_egl_image_dma` does not.
    let img = if preserved {
        ctx.create_egl_image_planar(w as i32, h as i32, fourcc, plane0, None, false)
    } else {
        ctx.create_egl_image_dma(
            dst_fd.as_raw_fd(),
            w as i32,
            h as i32,
            fourcc,
            stride as i32,
        )
    }
    .map_err(|e| format!("eglCreateImage(dst {w}x{h}@{stride}): {e}"))?;

    let result = (|| -> Result<(), String> {
        unsafe {
            let mut tex = 0u32;
            edgefirst_gl::gl::GenTextures(1, &mut tex);
            edgefirst_gl::gl::BindTexture(edgefirst_gl::gl::TEXTURE_2D, tex);
            edgefirst_gl::gl::EGLImageTargetTexture2DOES(
                edgefirst_gl::gl::TEXTURE_2D,
                img.as_ptr(),
            );
            gl_error("bind dst texture")?;

            let mut fbo = 0u32;
            edgefirst_gl::gl::GenFramebuffers(1, &mut fbo);
            edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, fbo);
            edgefirst_gl::gl::FramebufferTexture2D(
                edgefirst_gl::gl::FRAMEBUFFER,
                edgefirst_gl::gl::COLOR_ATTACHMENT0,
                edgefirst_gl::gl::TEXTURE_2D,
                tex,
                0,
            );
            let status = edgefirst_gl::gl::CheckFramebufferStatus(edgefirst_gl::gl::FRAMEBUFFER);
            if status != edgefirst_gl::gl::FRAMEBUFFER_COMPLETE {
                edgefirst_gl::gl::DeleteFramebuffers(1, &fbo);
                edgefirst_gl::gl::DeleteTextures(1, &tex);
                return Err(format!("FBO incomplete: status 0x{status:04X}"));
            }

            edgefirst_gl::gl::Viewport(0, 0, w as i32, h as i32);
            edgefirst_gl::gl::ClearColor(0.2, 0.4, 0.6, 1.0);
            edgefirst_gl::gl::Clear(edgefirst_gl::gl::COLOR_BUFFER_BIT);
            edgefirst_gl::gl::Finish();
            gl_error("clear")?;

            let mut px = [0u8; 4];
            edgefirst_gl::gl::ReadPixels(
                0,
                0,
                1,
                1,
                edgefirst_gl::gl::RGBA,
                edgefirst_gl::gl::UNSIGNED_BYTE,
                px.as_mut_ptr().cast(),
            );
            let read_err = gl_error("readback");

            edgefirst_gl::gl::BindFramebuffer(edgefirst_gl::gl::FRAMEBUFFER, 0);
            edgefirst_gl::gl::DeleteFramebuffers(1, &fbo);
            edgefirst_gl::gl::DeleteTextures(1, &tex);
            read_err
        }
    })();

    let _ = ctx.destroy_egl_image(img);
    result
}
