// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Integer texture format probe — verifies GL_R8UI, GL_RGBA8UI, GL_R8I,
//! GL_RGBA8I support for texture creation, FBO color-renderability, and
//! the XOR 0x80 uint8→int8 reinterpretation trick used to feed quantized
//! model inputs through a uint8 render surface.

use crate::egl_context::GpuContext;

fn yn(b: bool) -> &'static str {
    if b {
        "YES"
    } else {
        "no"
    }
}

fn gl_err_str(e: u32) -> &'static str {
    match e {
        gls::gl::NO_ERROR => "OK",
        gls::gl::INVALID_ENUM => "INVALID_ENUM",
        gls::gl::INVALID_VALUE => "INVALID_VALUE",
        gls::gl::INVALID_OPERATION => "INVALID_OP",
        gls::gl::INVALID_FRAMEBUFFER_OPERATION => "INVALID_FBO_OP",
        gls::gl::OUT_OF_MEMORY => "OOM",
        _ => "UNKNOWN",
    }
}

fn fbo_str(s: u32) -> &'static str {
    match s {
        gls::gl::FRAMEBUFFER_COMPLETE => "COMPLETE",
        gls::gl::FRAMEBUFFER_INCOMPLETE_ATTACHMENT => "INCOMPLETE_ATTACH",
        gls::gl::FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT => "MISSING_ATTACH",
        gls::gl::FRAMEBUFFER_UNSUPPORTED => "UNSUPPORTED",
        _ => "UNKNOWN",
    }
}

/// Drain any pending GL errors.
fn drain_errors() {
    unsafe { while gls::gl::GetError() != gls::gl::NO_ERROR {} }
}

struct FormatTest {
    name: &'static str,
    internal: i32,
    format: u32,
    pixel_type: u32,
}

const INT_FORMATS: &[FormatTest] = &[
    FormatTest {
        name: "GL_R8UI",
        internal: gls::gl::R8UI as i32,
        format: gls::gl::RED_INTEGER,
        pixel_type: gls::gl::UNSIGNED_BYTE,
    },
    FormatTest {
        name: "GL_RG8UI",
        internal: gls::gl::RG8UI as i32,
        format: gls::gl::RG_INTEGER,
        pixel_type: gls::gl::UNSIGNED_BYTE,
    },
    FormatTest {
        name: "GL_RGBA8UI",
        internal: gls::gl::RGBA8UI as i32,
        format: gls::gl::RGBA_INTEGER,
        pixel_type: gls::gl::UNSIGNED_BYTE,
    },
    FormatTest {
        name: "GL_R8I",
        internal: gls::gl::R8I as i32,
        format: gls::gl::RED_INTEGER,
        pixel_type: gls::gl::BYTE,
    },
    FormatTest {
        name: "GL_RG8I",
        internal: gls::gl::RG8I as i32,
        format: gls::gl::RG_INTEGER,
        pixel_type: gls::gl::BYTE,
    },
    FormatTest {
        name: "GL_RGBA8I",
        internal: gls::gl::RGBA8I as i32,
        format: gls::gl::RGBA_INTEGER,
        pixel_type: gls::gl::BYTE,
    },
];

const REF_FORMATS: &[FormatTest] = &[
    FormatTest {
        name: "GL_R8",
        internal: gls::gl::R8 as i32,
        format: gls::gl::RED,
        pixel_type: gls::gl::UNSIGNED_BYTE,
    },
    FormatTest {
        name: "GL_RG8",
        internal: gls::gl::RG8 as i32,
        format: gls::gl::RG,
        pixel_type: gls::gl::UNSIGNED_BYTE,
    },
    FormatTest {
        name: "GL_RGBA8",
        internal: gls::gl::RGBA8 as i32,
        format: gls::gl::RGBA,
        pixel_type: gls::gl::UNSIGNED_BYTE,
    },
    FormatTest {
        name: "GL_R8_SNORM",
        internal: gls::gl::R8_SNORM as i32,
        format: gls::gl::RED,
        pixel_type: gls::gl::BYTE,
    },
];

/// Test whether a format can be allocated as a texture and attached as an
/// FBO color target. Returns `(tex_ok, fbo_complete)`.
fn test_format(ft: &FormatTest, w: i32, h: i32) -> (bool, bool) {
    drain_errors();

    let mut tex: u32 = 0;
    let mut fbo: u32 = 0;
    let tex_ok;
    let mut fbo_complete = false;

    unsafe {
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
            ft.internal,
            w,
            h,
            0,
            ft.format,
            ft.pixel_type,
            std::ptr::null(),
        );
        let tex_err = gls::gl::GetError();
        tex_ok = tex_err == gls::gl::NO_ERROR;

        print!("  {:<20}  TexImage2D: {:<12}", ft.name, gl_err_str(tex_err));

        if tex_ok {
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
            fbo_complete = status == gls::gl::FRAMEBUFFER_COMPLETE;
            print!("  FBO: {:<18}", fbo_str(status));
            if fbo_complete {
                print!("  RENDERABLE");
            }
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
            gls::gl::DeleteFramebuffers(1, &fbo);
        } else {
            print!("  FBO: (skipped)");
        }

        gls::gl::DeleteTextures(1, &tex);
    }
    println!();

    (tex_ok, fbo_complete)
}

/// Test the XOR 0x80 render path: clear a RGBA8UI FBO with XOR'd values,
/// read back, and verify byte-level correctness.
fn test_xor_render(w: i32, h: i32) {
    println!("\n=== XOR 0x80 Render Pipeline Test ({w}x{h}) ===");

    // --- Test 1: RGBA8UI clear + integer readback ---
    println!("\n  [1] RGBA8UI clear + integer readback:");
    drain_errors();

    let mut tex: u32 = 0;
    let mut fbo: u32 = 0;

    unsafe {
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
            gls::gl::RGBA8UI as i32,
            w,
            h,
            0,
            gls::gl::RGBA_INTEGER,
            gls::gl::UNSIGNED_BYTE,
            std::ptr::null(),
        );
        let err = gls::gl::GetError();
        if err != gls::gl::NO_ERROR {
            println!("      RGBA8UI TexImage2D failed: {}", gl_err_str(err));
            gls::gl::DeleteTextures(1, &tex);
            return;
        }

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
        if status != gls::gl::FRAMEBUFFER_COMPLETE {
            println!("      FBO not complete: {}", fbo_str(status));
            gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
            gls::gl::DeleteFramebuffers(1, &fbo);
            gls::gl::DeleteTextures(1, &tex);
            return;
        }

        // XOR test values: uint8 {0, 64, 128, 255} ^ 0x80
        //   0 ^ 0x80 = 0x80, 64 ^ 0x80 = 0xC0, 128 ^ 0x80 = 0, 255 ^ 0x80 = 0x7F
        let clear_val: [u32; 4] = [0x80, 0xC0, 0x00, 0x7F];
        gls::gl::ClearBufferuiv(gls::gl::COLOR, 0, clear_val.as_ptr());
        gls::gl::Finish();

        let mut px: [u32; 4] = [0; 4];
        gls::gl::ReadPixels(
            0,
            0,
            1,
            1,
            gls::gl::RGBA_INTEGER,
            gls::gl::UNSIGNED_INT,
            px.as_mut_ptr() as *mut std::ffi::c_void,
        );
        let rp_err = gls::gl::GetError();
        if rp_err != gls::gl::NO_ERROR {
            println!("      ReadPixels failed: {}", gl_err_str(rp_err));
        } else {
            println!("      Input uint8:   0,   64,  128,  255");
            println!(
                "      XOR 0x80:      {},  {},  {},  {}",
                0x80u32, 0xC0u32, 0x00u32, 0x7Fu32,
            );
            println!(
                "      Readback:      {},  {},  {},  {}",
                px[0], px[1], px[2], px[3]
            );

            let ok = px[0] == 0x80
                && px[1] == 0xC0
                && px[2] == 0x00
                && px[3] == 0x7F;
            println!("      Verify:        {}", if ok { "PASS" } else { "FAIL" });

            if ok {
                println!(
                    "      As int8:       {}, {}, {}, {}",
                    (px[0] & 0xFF) as u8 as i8,
                    (px[1] & 0xFF) as u8 as i8,
                    (px[2] & 0xFF) as u8 as i8,
                    (px[3] & 0xFF) as u8 as i8
                );
                println!("      (uint8 0->int8 -128, 128->int8 0: correct zero-shift)");
            }
        }

        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
        gls::gl::DeleteFramebuffers(1, &fbo);
        gls::gl::DeleteTextures(1, &tex);
    }

    // --- Test 2: R8UI single-channel (model input scenario) ---
    println!("\n  [2] R8UI single-channel (model input path):");
    drain_errors();

    unsafe {
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
            gls::gl::R8UI as i32,
            w,
            h,
            0,
            gls::gl::RED_INTEGER,
            gls::gl::UNSIGNED_BYTE,
            std::ptr::null(),
        );
        let err = gls::gl::GetError();
        if err != gls::gl::NO_ERROR {
            println!("      R8UI TexImage2D failed: {}", gl_err_str(err));
            gls::gl::DeleteTextures(1, &tex);
            return;
        }

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
        if status != gls::gl::FRAMEBUFFER_COMPLETE {
            println!(
                "      FBO not complete: {} (not color-renderable)",
                fbo_str(status)
            );
            println!("      NOTE: R8UI may not be renderable — use RGBA8UI or compute shader");
        } else {
            let clear_val: [u32; 4] = [42 ^ 0x80, 0, 0, 0];
            gls::gl::ClearBufferuiv(gls::gl::COLOR, 0, clear_val.as_ptr());
            gls::gl::Finish();

            let mut px: [u32; 4] = [0; 4];
            gls::gl::ReadPixels(
                0,
                0,
                1,
                1,
                gls::gl::RED_INTEGER,
                gls::gl::UNSIGNED_INT,
                px.as_mut_ptr() as *mut std::ffi::c_void,
            );
            let err = gls::gl::GetError();
            if err == gls::gl::NO_ERROR && px[0] == (42 ^ 0x80) {
                println!(
                    "      Clear+ReadPixels: PASS (value={}, as int8={})",
                    px[0],
                    (px[0] & 0xFF) as u8 as i8
                );
            } else if err == gls::gl::NO_ERROR {
                println!(
                    "      Clear+ReadPixels: MISMATCH (got {}, expected {})",
                    px[0],
                    42u32 ^ 0x80
                );
            } else {
                println!("      ReadPixels failed: {}", gl_err_str(err));
            }
        }

        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
        gls::gl::DeleteFramebuffers(1, &fbo);
        gls::gl::DeleteTextures(1, &tex);
    }
}

/// Test compute shader compilation for the XOR 0x80 kernel.
fn test_compute_shaders(_ctx: &GpuContext) {
    println!("\n  [3] Compute shader availability:");

    let mut gl_major: i32 = 0;
    let mut gl_minor: i32 = 0;
    unsafe {
        gls::gl::GetIntegerv(gls::gl::MAJOR_VERSION, &mut gl_major);
        gls::gl::GetIntegerv(gls::gl::MINOR_VERSION, &mut gl_minor);
    }

    if gl_major < 3 || (gl_major == 3 && gl_minor < 1) {
        println!("      GLES {gl_major}.{gl_minor}: compute shaders NOT available (need 3.1+)");
        println!("      Fallback: use fragment shader with integer FBO target");
        return;
    }

    println!("      GLES {gl_major}.{gl_minor}: compute shaders AVAILABLE");

    unsafe {
        let mut wg = [0i32; 3];
        gls::gl::GetIntegeri_v(gls::gl::MAX_COMPUTE_WORK_GROUP_SIZE, 0, &mut wg[0]);
        gls::gl::GetIntegeri_v(gls::gl::MAX_COMPUTE_WORK_GROUP_SIZE, 1, &mut wg[1]);
        gls::gl::GetIntegeri_v(gls::gl::MAX_COMPUTE_WORK_GROUP_SIZE, 2, &mut wg[2]);
        println!(
            "      Max workgroup size: {} x {} x {}",
            wg[0], wg[1], wg[2]
        );

        let mut max_img: i32 = 0;
        gls::gl::GetIntegerv(gls::gl::MAX_COMPUTE_IMAGE_UNIFORMS, &mut max_img);
        println!("      Max compute image uniforms: {max_img}");
    }

    // r8ui XOR compute shader
    let cs_r8ui = c"\
#version 310 es\n\
layout(local_size_x = 16, local_size_y = 16) in;\n\
layout(r8ui, binding = 0) readonly uniform highp uimage2D srcImg;\n\
layout(r8ui, binding = 1) writeonly uniform highp uimage2D dstImg;\n\
void main() {\n\
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n\
    uint v = imageLoad(srcImg, pos).r;\n\
    imageStore(dstImg, pos, uvec4(v ^ 0x80u, 0u, 0u, 0u));\n\
}\n";

    let (compiled, linked) = try_compile_compute(cs_r8ui);
    println!(
        "      XOR compute shader (r8ui):   compile={}  link={}",
        yn(compiled),
        yn(linked)
    );

    // rgba8ui XOR compute shader
    let cs_rgba8ui = c"\
#version 310 es\n\
layout(local_size_x = 16, local_size_y = 16) in;\n\
layout(rgba8ui, binding = 0) readonly uniform highp uimage2D srcImg;\n\
layout(rgba8ui, binding = 1) writeonly uniform highp uimage2D dstImg;\n\
void main() {\n\
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);\n\
    uvec4 v = imageLoad(srcImg, pos);\n\
    imageStore(dstImg, pos, v ^ uvec4(0x80u));\n\
}\n";

    let (compiled, linked) = try_compile_compute(cs_rgba8ui);
    println!(
        "      XOR compute shader (rgba8ui): compile={}  link={}",
        yn(compiled),
        yn(linked)
    );
}

/// Try to compile and link a compute shader. Returns `(compiled, linked)`.
fn try_compile_compute(source: &std::ffi::CStr) -> (bool, bool) {
    unsafe {
        drain_errors();

        let cs = gls::gl::CreateShader(gls::gl::COMPUTE_SHADER);
        let src_ptr = source.as_ptr();
        gls::gl::ShaderSource(cs, 1, &raw const src_ptr, std::ptr::null());
        gls::gl::CompileShader(cs);

        let mut compiled: i32 = 0;
        gls::gl::GetShaderiv(cs, gls::gl::COMPILE_STATUS, &mut compiled);

        if compiled == 0 {
            let mut log_len: i32 = 0;
            gls::gl::GetShaderiv(cs, gls::gl::INFO_LOG_LENGTH, &mut log_len);
            if log_len > 0 {
                let mut log = vec![0u8; log_len as usize];
                gls::gl::GetShaderInfoLog(
                    cs,
                    log_len,
                    std::ptr::null_mut(),
                    log.as_mut_ptr() as *mut _,
                );
                let msg = String::from_utf8_lossy(&log);
                println!("      Compile error: {msg}");
            }
            gls::gl::DeleteShader(cs);
            return (false, false);
        }

        let prog = gls::gl::CreateProgram();
        gls::gl::AttachShader(prog, cs);
        gls::gl::LinkProgram(prog);

        let mut linked: i32 = 0;
        gls::gl::GetProgramiv(prog, gls::gl::LINK_STATUS, &mut linked);

        if linked == 0 {
            let mut log_len: i32 = 0;
            gls::gl::GetProgramiv(prog, gls::gl::INFO_LOG_LENGTH, &mut log_len);
            if log_len > 0 {
                let mut log = vec![0u8; log_len as usize];
                gls::gl::GetProgramInfoLog(
                    prog,
                    log_len,
                    std::ptr::null_mut(),
                    log.as_mut_ptr() as *mut _,
                );
                let msg = String::from_utf8_lossy(&log);
                println!("      Link error: {msg}");
            }
        }

        gls::gl::DetachShader(prog, cs);
        gls::gl::DeleteShader(cs);
        gls::gl::DeleteProgram(prog);

        (true, linked != 0)
    }
}

/// Run the full integer texture probe suite.
pub fn run(ctx: &GpuContext) {
    println!("=== Integer Texture Format Support ===");
    for ft in INT_FORMATS {
        test_format(ft, 256, 256);
    }

    println!("\n=== Normalized Formats (reference) ===");
    for ft in REF_FORMATS {
        test_format(ft, 256, 256);
    }

    test_xor_render(256, 256);
    test_compute_shaders(ctx);

    println!("\n=== Summary ===");
    println!("  For the uint8->int8 XOR 0x80 optimization:");
    println!("  - Write XOR'd uint8 values to GL_R8UI or GL_RGBA8UI surface");
    println!("  - Model runtime reads raw bytes as int8 (zero-copy reinterpret)");
    println!("  - No DRM fourcc for signed int8 — 'cheat' via uint8 is the way");
    println!();
}
