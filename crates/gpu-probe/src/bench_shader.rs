// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Shader compilation benchmarks — measures compile + link latency for vertex
//! and fragment shader programs of varying complexity.

use crate::bench::{run_bench, BenchResult};
use crate::egl_context::GpuContext;
use std::ffi::CString;
use std::ptr::null;

/// Shared vertex shader source used by all shader pairs.
const VERTEX_SRC: &str = "\
#version 300 es
precision mediump float;
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 texCoord;
out vec2 tc;
void main() { tc = texCoord; gl_Position = vec4(pos, 1.0); }
";

/// Simple texture sampling fragment shader.
const FRAG_SIMPLE: &str = "\
#version 300 es
precision mediump float;
uniform sampler2D tex;
in vec2 tc;
out vec4 color;
void main() { color = texture(tex, tc); }
";

/// YUV external texture fragment shader (requires OES_EGL_image_external_essl3).
const FRAG_YUV_EXTERNAL: &str = "\
#version 300 es
#extension GL_OES_EGL_image_external_essl3 : require
precision mediump float;
uniform samplerExternalOES tex;
in vec2 tc;
out vec4 color;
void main() { color = texture(tex, tc); }
";

/// Complex LUT-based color mapping fragment shader.
const FRAG_COMPLEX: &str = "\
#version 300 es
precision mediump float;
uniform sampler2D tex;
uniform float threshold;
uniform vec4 color_lut[16];
in vec2 tc;
out vec4 color;
void main() {
    vec4 texel = texture(tex, tc);
    float luma = dot(texel.rgb, vec3(0.299, 0.587, 0.114));
    int idx = clamp(int(luma * 15.0), 0, 15);
    color = mix(texel, color_lut[idx], step(threshold, luma));
}
";

/// Compile and link a shader program, then delete all resources.
///
/// This performs the full lifecycle: create program, create + compile + attach
/// vertex shader, create + compile + attach fragment shader, link, then delete.
fn compile_link_delete(vert_src: &CString, frag_src: &CString) {
    unsafe {
        let program = gls::gl::CreateProgram();

        // Vertex shader
        let vs = gls::gl::CreateShader(gls::gl::VERTEX_SHADER);
        let vs_ptr = vert_src.as_ptr();
        gls::gl::ShaderSource(vs, 1, &raw const vs_ptr, null());
        gls::gl::CompileShader(vs);
        gls::gl::AttachShader(program, vs);

        // Fragment shader
        let fs = gls::gl::CreateShader(gls::gl::FRAGMENT_SHADER);
        let fs_ptr = frag_src.as_ptr();
        gls::gl::ShaderSource(fs, 1, &raw const fs_ptr, null());
        gls::gl::CompileShader(fs);
        gls::gl::AttachShader(program, fs);

        // Link
        gls::gl::LinkProgram(program);

        // Cleanup
        gls::gl::DeleteProgram(program);
        gls::gl::DeleteShader(vs);
        gls::gl::DeleteShader(fs);
    }
}

/// Run shader compilation benchmarks and return collected results.
pub fn run(_ctx: &GpuContext) -> Vec<BenchResult> {
    println!("== Benchmark: Shader Compilation ==");

    let vert_cstr =
        CString::new(VERTEX_SRC).expect("vertex shader source contains interior null byte");

    let shaders: &[(&str, &str)] = &[
        ("simple", FRAG_SIMPLE),
        ("yuv_external", FRAG_YUV_EXTERNAL),
        ("complex", FRAG_COMPLEX),
    ];

    let mut results = Vec::new();

    for &(label, frag_src) in shaders {
        let frag_cstr =
            CString::new(frag_src).expect("fragment shader source contains interior null byte");

        // Verify the shader pair compiles at least once before benchmarking.
        // The YUV external shader requires GL_OES_EGL_image_external_essl3
        // which may not be available on all drivers.
        let compiles = unsafe {
            let vs = gls::gl::CreateShader(gls::gl::VERTEX_SHADER);
            let vs_ptr = vert_cstr.as_ptr();
            gls::gl::ShaderSource(vs, 1, &raw const vs_ptr, null());
            gls::gl::CompileShader(vs);
            let mut vs_ok: i32 = 0;
            gls::gl::GetShaderiv(vs, gls::gl::COMPILE_STATUS, &mut vs_ok);

            let fs = gls::gl::CreateShader(gls::gl::FRAGMENT_SHADER);
            let fs_ptr = frag_cstr.as_ptr();
            gls::gl::ShaderSource(fs, 1, &raw const fs_ptr, null());
            gls::gl::CompileShader(fs);
            let mut fs_ok: i32 = 0;
            gls::gl::GetShaderiv(fs, gls::gl::COMPILE_STATUS, &mut fs_ok);

            gls::gl::DeleteShader(vs);
            gls::gl::DeleteShader(fs);

            vs_ok != 0 && fs_ok != 0
        };

        if !compiles {
            println!(
                "  SKIP shader_compile_link/{label}: shader compilation failed on this driver"
            );
            continue;
        }

        let name = format!("shader_compile_link/{label}");
        let vert = vert_cstr.clone();
        let frag = frag_cstr;

        let r = run_bench(&name, 5, 100, || {
            compile_link_delete(&vert, &frag);
        });
        r.print_summary();
        results.push(r);
    }

    println!();
    results
}
