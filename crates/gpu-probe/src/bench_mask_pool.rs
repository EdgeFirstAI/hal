// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Tier 2-a mask pool proof-of-concept benchmark.
//!
//! Compares two paths for rendering N segmentation masks into an RGBA8 FBO:
//!
//! 1. **`baseline_per_instance`** — one `glTexImage2D` per detection, one
//!    `glDrawElements` per detection, optional `glFinish` per detection
//!    (matches the Tier 1 implementation in
//!    `crates/image/src/gl/processor.rs:render_yolo_segmentation`).
//! 2. **`instanced_array`** — one `glTexImage3D` upload of all N masks into
//!    a `GL_TEXTURE_2D_ARRAY`, followed by a single `glDrawElementsInstanced`
//!    that expands a unit quad to N detections via per-instance vertex
//!    attributes (`bbox_ndc`, `mask_extent`, `layer`, `class_index`).
//!
//! Both paths render into the same RGBA8 FBO and pixel-validate against
//! each other on the first frame, then sweep `N ∈ {2, 5, 10, 20, 40, 80}`.
//!
//! The point of the bench is to answer:
//!
//! - Does instancing actually win on Vivante GC7000UL and Mali-G310?
//! - How does the gap scale with N?
//! - What is the practical floor for the per-frame draw cost?
//!
//! This isolates the GL behaviour from the rest of the HAL — no
//! `Tensor` / `Decoder` / `materialize_masks` involvement. If instancing
//! wins here, it will win in the HAL too.

use crate::bench::{run_bench, BenchResult};
use crate::bench_render::compile_program;
use crate::egl_context::GpuContext;
use std::ffi::{c_void, CString};
use std::ptr::null;

// ---------------------------------------------------------------------------
// Test parameters
// ---------------------------------------------------------------------------

/// Output framebuffer dimensions — matches HAL's typical letterbox output.
const FBO_W: i32 = 640;
const FBO_H: i32 = 640;

/// Mask cell dimensions in texels. Matches YOLOv8-seg proto resolution
/// upper bound; using a fixed cell size for the bench means both paths
/// upload the same byte volume.
const CELL_W: i32 = 40;
const CELL_H: i32 = 40;

/// Detection counts to sweep. Picked to bracket realistic scenes:
/// 2 ≈ zidane, 40 ≈ crowd, 80 = stress.
const N_DETECT_SWEEP: &[u32] = &[2, 5, 10, 20, 40, 80];

const WARMUP: usize = 10;
const ITERATIONS: usize = 100;

// ---------------------------------------------------------------------------
// Shaders — match the HAL's instanced_segmentation_shader behaviour
// ---------------------------------------------------------------------------

/// Baseline vertex shader: trivial passthrough with attribute-supplied
/// position + texcoord (matches HAL's `generate_vertex_shader`).
const BASELINE_VS: &str = "\
#version 300 es
precision mediump float;
layout(location = 0) in vec3 pos;
layout(location = 1) in vec2 texCoord;
out vec2 tc;
void main() { tc = texCoord; gl_Position = vec4(pos, 1.0); }
";

/// Baseline fragment shader: 1 texel fetch + smoothstep + discard.
/// Identical to HAL's `generate_instanced_segmentation_shader`.
const BASELINE_FS: &str = "\
#version 300 es
precision mediump float;
uniform sampler2D mask0;
uniform vec4 colors[20];
uniform int class_index;
uniform float opacity;
in vec2 tc;
out vec4 color;
void main() {
    float r0 = texture(mask0, tc).r;
    float edge = smoothstep(0.5, 0.65, r0);
    if (edge <= 0.0) discard;
    vec4 c = colors[class_index % 20];
    color = vec4(c.rgb, c.a * edge * opacity);
}
";

/// Instanced vertex shader: per-instance attributes drive the quad
/// position, texcoord scaling, layer index, and class index.
const INSTANCED_VS: &str = "\
#version 300 es
precision mediump float;
layout(location = 0) in vec2 unit_pos;        // shared unit quad
layout(location = 1) in vec4 bbox_ndc;        // per-instance: xmin ymin xmax ymax
layout(location = 2) in vec2 mask_extent_in;  // per-instance: w/cell_w, h/cell_h
layout(location = 3) in float layer_in;       // per-instance: array layer (float for portability)
layout(location = 4) in float class_in;       // per-instance: colour palette index
flat out int v_layer;
flat out int v_class;
flat out vec2 v_mask_extent;
out vec2 v_tc;
void main() {
    vec2 uv = unit_pos * 0.5 + 0.5;          // 0..1 across the bbox
    vec2 ndc = mix(bbox_ndc.xy, bbox_ndc.zw, uv);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_tc = uv;
    v_layer = int(layer_in + 0.5);
    v_class = int(class_in + 0.5);
    v_mask_extent = mask_extent_in;
}
";

/// Instanced fragment shader: same as baseline but samples a 2D array
/// texture using the per-instance layer index.
const INSTANCED_FS: &str = "\
#version 300 es
precision mediump float;
uniform mediump sampler2DArray mask_array;
uniform vec4 colors[20];
uniform float opacity;
flat in int v_layer;
flat in int v_class;
flat in vec2 v_mask_extent;
in vec2 v_tc;
out vec4 color;
void main() {
    vec2 layer_tc = v_tc * v_mask_extent;
    float r0 = texture(mask_array, vec3(layer_tc, float(v_layer))).r;
    float edge = smoothstep(0.5, 0.65, r0);
    if (edge <= 0.0) discard;
    vec4 c = colors[v_class % 20];
    color = vec4(c.rgb, c.a * edge * opacity);
}
";

// ---------------------------------------------------------------------------
// Default 20-colour palette (matches HAL's DEFAULT_COLORS)
// ---------------------------------------------------------------------------

const DEFAULT_COLORS: [[f32; 4]; 20] = [
    [0.0, 0.5, 0.0, 0.5],
    [1.0, 0.4, 0.0, 0.5],
    [0.5, 0.0, 0.5, 0.5],
    [0.0, 0.5, 0.5, 0.5],
    [0.5, 0.5, 0.0, 0.5],
    [0.8, 0.0, 0.2, 0.5],
    [0.2, 0.4, 0.8, 0.5],
    [0.4, 0.8, 0.2, 0.5],
    [0.8, 0.2, 0.6, 0.5],
    [0.2, 0.8, 0.6, 0.5],
    [0.6, 0.2, 0.4, 0.5],
    [0.4, 0.2, 0.8, 0.5],
    [0.6, 0.6, 0.2, 0.5],
    [0.2, 0.6, 0.6, 0.5],
    [0.8, 0.6, 0.2, 0.5],
    [0.2, 0.4, 0.4, 0.5],
    [0.6, 0.4, 0.8, 0.5],
    [0.4, 0.6, 0.4, 0.5],
    [0.8, 0.4, 0.4, 0.5],
    [0.4, 0.4, 0.6, 0.5],
];

// ---------------------------------------------------------------------------
// Synthetic mask data — sigmoid-shaped blob with noise, deterministic
// ---------------------------------------------------------------------------

/// Build a `cell_w × cell_h` R8 mask with a soft circular blob roughly
/// mimicking a YOLOv8-seg sigmoid output: high in the centre, falling off
/// to edges. Per-mask seed varies the centre slightly so consecutive
/// uploads differ.
fn synthesize_mask(seed: u32) -> Vec<u8> {
    let mut data = vec![0u8; (CELL_W * CELL_H) as usize];
    let cx = (CELL_W as f32) * (0.4 + 0.2 * ((seed % 7) as f32 / 7.0));
    let cy = (CELL_H as f32) * (0.4 + 0.2 * ((seed % 5) as f32 / 5.0));
    let r = (CELL_W as f32) * 0.35;
    for y in 0..CELL_H {
        for x in 0..CELL_W {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let d = (dx * dx + dy * dy).sqrt();
            // Sigmoid-shaped falloff, scaled to [0, 255].
            let t = (r - d) * 0.5;
            let s = 1.0 / (1.0 + (-t).exp());
            data[(y * CELL_W + x) as usize] = (s * 255.0) as u8;
        }
    }
    data
}

/// Build a synthetic detection layout: N quads scattered across the FBO,
/// each ~80×80 pixels, deterministic positions.
struct Detection {
    bbox_ndc: [f32; 4], // xmin ymin xmax ymax in NDC
    class_index: u32,
}

fn synthesize_detections(n: u32) -> Vec<Detection> {
    let mut out = Vec::with_capacity(n as usize);
    // Lay out N quads on a √N × √N grid covering the FBO with margin.
    let cols = (n as f32).sqrt().ceil() as u32;
    let rows = n.div_ceil(cols);
    let cell_w_ndc = 1.8 / cols as f32; // total width 1.8 (NDC -0.9..0.9)
    let cell_h_ndc = 1.8 / rows as f32;
    let quad_w = cell_w_ndc * 0.8;
    let quad_h = cell_h_ndc * 0.8;
    for i in 0..n {
        let col = i % cols;
        let row = i / cols;
        let cx = -0.9 + (col as f32 + 0.5) * cell_w_ndc;
        let cy = -0.9 + (row as f32 + 0.5) * cell_h_ndc;
        out.push(Detection {
            bbox_ndc: [
                cx - quad_w * 0.5,
                cy - quad_h * 0.5,
                cx + quad_w * 0.5,
                cy + quad_h * 0.5,
            ],
            class_index: i % 20,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// FBO setup — RGBA8 destination renderbuffer, 640×640
// ---------------------------------------------------------------------------

fn create_dst_fbo() -> (u32, u32) {
    unsafe {
        let mut fbo = 0u32;
        let mut rbo = 0u32;
        gls::gl::GenFramebuffers(1, &mut fbo);
        gls::gl::GenRenderbuffers(1, &mut rbo);
        gls::gl::BindRenderbuffer(gls::gl::RENDERBUFFER, rbo);
        gls::gl::RenderbufferStorage(gls::gl::RENDERBUFFER, gls::gl::RGBA8, FBO_W, FBO_H);
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, fbo);
        gls::gl::FramebufferRenderbuffer(
            gls::gl::FRAMEBUFFER,
            gls::gl::COLOR_ATTACHMENT0,
            gls::gl::RENDERBUFFER,
            rbo,
        );
        let status = gls::gl::CheckFramebufferStatus(gls::gl::FRAMEBUFFER);
        assert_eq!(
            status,
            gls::gl::FRAMEBUFFER_COMPLETE,
            "FBO incomplete: 0x{:x}",
            status
        );
        gls::gl::Viewport(0, 0, FBO_W, FBO_H);
        (fbo, rbo)
    }
}

fn destroy_dst_fbo(fbo: u32, rbo: u32) {
    unsafe {
        gls::gl::BindFramebuffer(gls::gl::FRAMEBUFFER, 0);
        gls::gl::DeleteFramebuffers(1, &fbo);
        gls::gl::DeleteRenderbuffers(1, &rbo);
    }
}

// ---------------------------------------------------------------------------
// Vivante detection — match HAL behaviour: per-instance glFinish only on Vivante
// ---------------------------------------------------------------------------

fn detect_vivante() -> bool {
    unsafe {
        let r = gls::gl::GetString(gls::gl::RENDERER);
        if r.is_null() {
            return false;
        }
        let s = std::ffi::CStr::from_ptr(r as *const _)
            .to_string_lossy()
            .to_lowercase();
        s.contains("vivante") || s.contains("gc7000") || s.contains("galcore")
    }
}

// ---------------------------------------------------------------------------
// Path A: baseline per-instance (matches HAL Tier 1)
// ---------------------------------------------------------------------------

struct BaselinePath {
    program: u32,
    tex: u32,
    vao: u32,
    vertex_buffer: u32,
    texcoord_buffer: u32,
    ebo: u32,
    is_vivante: bool,
    class_index_loc: i32,
}

impl BaselinePath {
    fn new(is_vivante: bool) -> Result<Self, String> {
        let vs = CString::new(BASELINE_VS).unwrap();
        let fs = CString::new(BASELINE_FS).unwrap();
        let program = compile_program(&vs, &fs)?;

        unsafe {
            // Set static uniforms (colors, opacity, mask0 sampler unit)
            gls::gl::UseProgram(program);
            let colors_loc = gls::gl::GetUniformLocation(program, c"colors".as_ptr());
            gls::gl::Uniform4fv(colors_loc, 20, DEFAULT_COLORS.as_flattened().as_ptr());
            let opacity_loc = gls::gl::GetUniformLocation(program, c"opacity".as_ptr());
            gls::gl::Uniform1f(opacity_loc, 1.0);
            let mask0_loc = gls::gl::GetUniformLocation(program, c"mask0".as_ptr());
            gls::gl::Uniform1i(mask0_loc, 0);

            let class_index_loc = gls::gl::GetUniformLocation(program, c"class_index".as_ptr());

            let mut tex = 0u32;
            gls::gl::GenTextures(1, &mut tex);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, tex);
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_WRAP_S,
                gls::gl::CLAMP_TO_EDGE as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D,
                gls::gl::TEXTURE_WRAP_T,
                gls::gl::CLAMP_TO_EDGE as i32,
            );

            // VAO captures the attribute-pointer state so we only pay the
            // setup cost once. Per-call we just `BindVertexArray` and
            // `BufferSubData` the new vertex / texcoord values.
            let mut vao = 0u32;
            gls::gl::GenVertexArrays(1, &mut vao);
            gls::gl::BindVertexArray(vao);

            let mut bufs = [0u32; 3];
            gls::gl::GenBuffers(3, bufs.as_mut_ptr());
            let vertex_buffer = bufs[0];
            let texcoord_buffer = bufs[1];
            let ebo = bufs[2];

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, vertex_buffer);
            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                (12 * std::mem::size_of::<f32>()) as isize,
                std::ptr::null(),
                gls::gl::DYNAMIC_DRAW,
            );
            gls::gl::EnableVertexAttribArray(0);
            gls::gl::VertexAttribPointer(0, 3, gls::gl::FLOAT, gls::gl::FALSE, 0, null());

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, texcoord_buffer);
            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                (8 * std::mem::size_of::<f32>()) as isize,
                std::ptr::null(),
                gls::gl::DYNAMIC_DRAW,
            );
            gls::gl::EnableVertexAttribArray(1);
            gls::gl::VertexAttribPointer(1, 2, gls::gl::FLOAT, gls::gl::FALSE, 0, null());

            let indices: [u32; 4] = [0, 1, 2, 3];
            gls::gl::BindBuffer(gls::gl::ELEMENT_ARRAY_BUFFER, ebo);
            gls::gl::BufferData(
                gls::gl::ELEMENT_ARRAY_BUFFER,
                std::mem::size_of_val(&indices) as isize,
                indices.as_ptr() as *const c_void,
                gls::gl::STATIC_DRAW,
            );

            gls::gl::BindVertexArray(0);

            Ok(BaselinePath {
                program,
                tex,
                vao,
                vertex_buffer,
                texcoord_buffer,
                ebo,
                is_vivante,
                class_index_loc,
            })
        }
    }

    /// Render `N` masks the way HAL's Tier 1 `render_yolo_segmentation` does.
    fn render(&self, detections: &[Detection], masks: &[Vec<u8>]) {
        unsafe {
            // Hoisted-out-of-loop state (matches Tier 1).
            gls::gl::UseProgram(self.program);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D, self.tex);
            gls::gl::BindVertexArray(self.vao);

            // Texel-centre UVs (matches Tier 1) — constant across the
            // batch since cell size is fixed.
            let u_lo = 0.5 / CELL_W as f32;
            let v_lo = 0.5 / CELL_H as f32;
            let u_hi = (CELL_W as f32 - 0.5) / CELL_W as f32;
            let v_hi = (CELL_H as f32 - 0.5) / CELL_H as f32;
            let tcs: [f32; 8] = [u_lo, v_hi, u_hi, v_hi, u_hi, v_lo, u_lo, v_lo];
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.texcoord_buffer);
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (8 * std::mem::size_of::<f32>()) as isize,
                tcs.as_ptr() as *const c_void,
            );

            for (det, mask) in detections.iter().zip(masks.iter()) {
                // Per-instance allocating upload (implicit orphan).
                gls::gl::TexImage2D(
                    gls::gl::TEXTURE_2D,
                    0,
                    gls::gl::R8 as i32,
                    CELL_W,
                    CELL_H,
                    0,
                    gls::gl::RED,
                    gls::gl::UNSIGNED_BYTE,
                    mask.as_ptr() as *const c_void,
                );

                gls::gl::Uniform1i(self.class_index_loc, det.class_index as i32);

                let verts: [f32; 12] = [
                    det.bbox_ndc[0],
                    det.bbox_ndc[3],
                    0.0, // top-left  (NDC top = ymax)
                    det.bbox_ndc[2],
                    det.bbox_ndc[3],
                    0.0, // top-right
                    det.bbox_ndc[2],
                    det.bbox_ndc[1],
                    0.0, // bottom-right
                    det.bbox_ndc[0],
                    det.bbox_ndc[1],
                    0.0, // bottom-left
                ];
                gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.vertex_buffer);
                gls::gl::BufferSubData(
                    gls::gl::ARRAY_BUFFER,
                    0,
                    (12 * std::mem::size_of::<f32>()) as isize,
                    verts.as_ptr() as *const c_void,
                );

                gls::gl::DrawElements(gls::gl::TRIANGLE_FAN, 4, gls::gl::UNSIGNED_INT, null());

                if self.is_vivante {
                    gls::gl::Finish();
                }
            }
            gls::gl::BindVertexArray(0);
        }
    }
}

impl Drop for BaselinePath {
    fn drop(&mut self) {
        unsafe {
            gls::gl::DeleteTextures(1, &self.tex);
            let bufs = [self.vertex_buffer, self.texcoord_buffer, self.ebo];
            gls::gl::DeleteBuffers(3, bufs.as_ptr());
            gls::gl::DeleteVertexArrays(1, &self.vao);
            gls::gl::DeleteProgram(self.program);
        }
    }
}

// ---------------------------------------------------------------------------
// Path B: instanced texture array (POC for Tier 2-a)
// ---------------------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct MaskInstanceAttr {
    bbox_ndc: [f32; 4],
    mask_extent: [f32; 2],
    layer: f32,
    class_index: f32,
}

struct InstancedPath {
    program: u32,
    tex_array: u32,
    capacity: u32,
    vao: u32,
    unit_quad_vbo: u32,
    instance_vbo: u32,
    ebo: u32,
    /// Pre-allocated host-side buffer for the per-instance attribute table.
    /// Reused across `render()` calls so the timing loop doesn't pay for a
    /// `Vec` allocation on every iteration. Capacity is fixed at construction
    /// to `InstancedPath::capacity`.
    instance_attrs: Vec<MaskInstanceAttr>,
}

impl InstancedPath {
    fn new(capacity: u32) -> Result<Self, String> {
        let vs = CString::new(INSTANCED_VS).unwrap();
        let fs = CString::new(INSTANCED_FS).unwrap();
        let program = compile_program(&vs, &fs)?;

        unsafe {
            gls::gl::UseProgram(program);
            let colors_loc = gls::gl::GetUniformLocation(program, c"colors".as_ptr());
            gls::gl::Uniform4fv(colors_loc, 20, DEFAULT_COLORS.as_flattened().as_ptr());
            let opacity_loc = gls::gl::GetUniformLocation(program, c"opacity".as_ptr());
            gls::gl::Uniform1f(opacity_loc, 1.0);
            let sampler_loc = gls::gl::GetUniformLocation(program, c"mask_array".as_ptr());
            gls::gl::Uniform1i(sampler_loc, 0);

            // Allocate the texture array at full capacity.
            let mut tex_array = 0u32;
            gls::gl::GenTextures(1, &mut tex_array);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D_ARRAY, tex_array);
            gls::gl::TexImage3D(
                gls::gl::TEXTURE_2D_ARRAY,
                0,
                gls::gl::R8 as i32,
                CELL_W,
                CELL_H,
                capacity as i32,
                0,
                gls::gl::RED,
                gls::gl::UNSIGNED_BYTE,
                std::ptr::null(),
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D_ARRAY,
                gls::gl::TEXTURE_MIN_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D_ARRAY,
                gls::gl::TEXTURE_MAG_FILTER,
                gls::gl::LINEAR as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D_ARRAY,
                gls::gl::TEXTURE_WRAP_S,
                gls::gl::CLAMP_TO_EDGE as i32,
            );
            gls::gl::TexParameteri(
                gls::gl::TEXTURE_2D_ARRAY,
                gls::gl::TEXTURE_WRAP_T,
                gls::gl::CLAMP_TO_EDGE as i32,
            );

            // VAO captures all 5 vertex attribute pointer bindings + EBO
            // binding. Per render() call we just BindVertexArray + 1
            // BufferSubData for the instance attrs.
            let mut vao = 0u32;
            gls::gl::GenVertexArrays(1, &mut vao);
            gls::gl::BindVertexArray(vao);

            // Shared unit quad: 4 vertices, attribute location 0.
            let unit_quad: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0];
            let mut bufs = [0u32; 3];
            gls::gl::GenBuffers(3, bufs.as_mut_ptr());
            let unit_quad_vbo = bufs[0];
            let instance_vbo = bufs[1];
            let ebo = bufs[2];

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, unit_quad_vbo);
            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                std::mem::size_of_val(&unit_quad) as isize,
                unit_quad.as_ptr() as *const c_void,
                gls::gl::STATIC_DRAW,
            );
            gls::gl::EnableVertexAttribArray(0);
            gls::gl::VertexAttribPointer(0, 2, gls::gl::FLOAT, gls::gl::FALSE, 0, null());
            gls::gl::VertexAttribDivisor(0, 0);

            // Per-instance attribute buffer: pre-size to capacity.
            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, instance_vbo);
            gls::gl::BufferData(
                gls::gl::ARRAY_BUFFER,
                (std::mem::size_of::<MaskInstanceAttr>() * capacity as usize) as isize,
                std::ptr::null(),
                gls::gl::DYNAMIC_DRAW,
            );
            let stride = std::mem::size_of::<MaskInstanceAttr>() as i32;
            // bbox_ndc (vec4) at offset 0
            gls::gl::EnableVertexAttribArray(1);
            gls::gl::VertexAttribPointer(1, 4, gls::gl::FLOAT, gls::gl::FALSE, stride, null());
            gls::gl::VertexAttribDivisor(1, 1);
            // mask_extent (vec2) at offset 16
            gls::gl::EnableVertexAttribArray(2);
            gls::gl::VertexAttribPointer(
                2,
                2,
                gls::gl::FLOAT,
                gls::gl::FALSE,
                stride,
                16 as *const c_void,
            );
            gls::gl::VertexAttribDivisor(2, 1);
            // layer (float) at offset 24
            gls::gl::EnableVertexAttribArray(3);
            gls::gl::VertexAttribPointer(
                3,
                1,
                gls::gl::FLOAT,
                gls::gl::FALSE,
                stride,
                24 as *const c_void,
            );
            gls::gl::VertexAttribDivisor(3, 1);
            // class_index (float) at offset 28
            gls::gl::EnableVertexAttribArray(4);
            gls::gl::VertexAttribPointer(
                4,
                1,
                gls::gl::FLOAT,
                gls::gl::FALSE,
                stride,
                28 as *const c_void,
            );
            gls::gl::VertexAttribDivisor(4, 1);

            let indices: [u32; 4] = [0, 1, 2, 3];
            gls::gl::BindBuffer(gls::gl::ELEMENT_ARRAY_BUFFER, ebo);
            gls::gl::BufferData(
                gls::gl::ELEMENT_ARRAY_BUFFER,
                std::mem::size_of_val(&indices) as isize,
                indices.as_ptr() as *const c_void,
                gls::gl::STATIC_DRAW,
            );

            gls::gl::BindVertexArray(0);

            Ok(InstancedPath {
                program,
                tex_array,
                capacity,
                vao,
                unit_quad_vbo,
                instance_vbo,
                ebo,
                instance_attrs: Vec::with_capacity(capacity as usize),
            })
        }
    }

    /// Render `N` masks via a single instanced draw.
    ///
    /// `flat_masks` must be a contiguous `(N, CELL_H, CELL_W)` buffer —
    /// the same layout the real `MaskPool` will use, so this matches the
    /// production cost.
    ///
    /// The per-instance attribute table is reused from `self.instance_attrs`
    /// (pre-allocated to `capacity`) so the timing loop never pays for a
    /// `Vec` allocation. `&mut self` is required for the in-place rebuild;
    /// the GL draw is otherwise unchanged.
    fn render(&mut self, detections: &[Detection], flat_masks: &[u8]) {
        assert!(detections.len() <= self.capacity as usize);
        let n = detections.len() as i32;
        let expected_bytes = (CELL_W * CELL_H * n) as usize;
        assert_eq!(flat_masks.len(), expected_bytes);

        // Rebuild the instance attribute table in place. The pre-allocated
        // capacity guarantees `extend` here never reallocates.
        self.instance_attrs.clear();
        self.instance_attrs
            .extend(detections.iter().enumerate().map(|(i, d)| {
                MaskInstanceAttr {
                    bbox_ndc: d.bbox_ndc,
                    mask_extent: [1.0, 1.0], // POC: every mask fills its cell
                    layer: i as f32,
                    class_index: d.class_index as f32,
                }
            }));
        debug_assert!(
            self.instance_attrs.len() <= self.instance_attrs.capacity(),
            "instance_attrs reallocated mid-frame — capacity sizing is wrong"
        );

        unsafe {
            gls::gl::UseProgram(self.program);
            gls::gl::ActiveTexture(gls::gl::TEXTURE0);
            gls::gl::BindTexture(gls::gl::TEXTURE_2D_ARRAY, self.tex_array);

            // Single contiguous upload for all N layers — matches the
            // production code path where the mask pool is a flat
            // DMA-BUF or PBO and every detection writes directly into it.
            gls::gl::TexSubImage3D(
                gls::gl::TEXTURE_2D_ARRAY,
                0,
                0,
                0,
                0,
                CELL_W,
                CELL_H,
                n,
                gls::gl::RED,
                gls::gl::UNSIGNED_BYTE,
                flat_masks.as_ptr() as *const c_void,
            );

            gls::gl::BindBuffer(gls::gl::ARRAY_BUFFER, self.instance_vbo);
            gls::gl::BufferSubData(
                gls::gl::ARRAY_BUFFER,
                0,
                (std::mem::size_of::<MaskInstanceAttr>() * self.instance_attrs.len()) as isize,
                self.instance_attrs.as_ptr() as *const c_void,
            );

            // VAO holds the attribute pointer state (set once at init).
            gls::gl::BindVertexArray(self.vao);
            gls::gl::DrawElementsInstanced(
                gls::gl::TRIANGLE_FAN,
                4,
                gls::gl::UNSIGNED_INT,
                null(),
                detections.len() as i32,
            );
            gls::gl::BindVertexArray(0);
        }
    }
}

impl Drop for InstancedPath {
    fn drop(&mut self) {
        unsafe {
            gls::gl::DeleteTextures(1, &self.tex_array);
            let bufs = [self.unit_quad_vbo, self.instance_vbo, self.ebo];
            gls::gl::DeleteBuffers(3, bufs.as_ptr());
            gls::gl::DeleteVertexArrays(1, &self.vao);
            gls::gl::DeleteProgram(self.program);
        }
    }
}

// ---------------------------------------------------------------------------
// Pixel-level cross-validation: do both paths produce the same image?
// ---------------------------------------------------------------------------

fn read_back_fbo() -> Vec<u8> {
    let mut buf = vec![0u8; (FBO_W * FBO_H * 4) as usize];
    unsafe {
        gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
        gls::gl::ReadPixels(
            0,
            0,
            FBO_W,
            FBO_H,
            gls::gl::RGBA,
            gls::gl::UNSIGNED_BYTE,
            buf.as_mut_ptr() as *mut c_void,
        );
    }
    buf
}

fn compare_paths(
    detections: &[Detection],
    masks: &[Vec<u8>],
    flat_masks: &[u8],
    baseline: &BaselinePath,
    instanced: &mut InstancedPath,
) -> (u32, u32) {
    unsafe {
        // Render baseline
        gls::gl::ClearColor(0.0, 0.0, 0.0, 0.0);
        gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
        gls::gl::Enable(gls::gl::BLEND);
        gls::gl::BlendFuncSeparate(
            gls::gl::SRC_ALPHA,
            gls::gl::ONE_MINUS_SRC_ALPHA,
            gls::gl::ZERO,
            gls::gl::ONE,
        );
        baseline.render(detections, masks);
        gls::gl::Finish();
        let img_a = read_back_fbo();

        gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
        instanced.render(detections, flat_masks);
        gls::gl::Finish();
        let img_b = read_back_fbo();

        // Optional dump for diagnosis: set MASK_POOL_DUMP=/tmp/mp to write
        // both images as raw RGBA8 the bench harness can post-mortem.
        if let Ok(prefix) = std::env::var("MASK_POOL_DUMP") {
            let _ = std::fs::write(format!("{prefix}_baseline.rgba"), &img_a);
            let _ = std::fs::write(format!("{prefix}_instanced.rgba"), &img_b);
            println!(
                "  Dumped: {prefix}_baseline.rgba and {prefix}_instanced.rgba ({}x{})",
                FBO_W, FBO_H
            );
        }

        // Count differing pixels and max channel-sum diff across ALL
        // four channels — mask shaders encode coverage primarily in
        // alpha, so an RGB-only diff would miss a real divergence in
        // transparency. Max theoretical per-pixel sum is 4 × 255 = 1020.
        let mut diff_pixels = 0u32;
        let mut max_diff = 0u32;
        for (a, b) in img_a.chunks_exact(4).zip(img_b.chunks_exact(4)) {
            let d = ((a[0] as i32 - b[0] as i32).abs()
                + (a[1] as i32 - b[1] as i32).abs()
                + (a[2] as i32 - b[2] as i32).abs()
                + (a[3] as i32 - b[3] as i32).abs()) as u32;
            if d > 0 {
                diff_pixels += 1;
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        (diff_pixels, max_diff)
    }
}

// ---------------------------------------------------------------------------
// Bench entry point
// ---------------------------------------------------------------------------

pub fn run(_ctx: &GpuContext) -> Vec<BenchResult> {
    println!("== Benchmark: Mask Pool POC (Tier 2-a) ==");
    let is_vivante = detect_vivante();
    println!(
        "  GPU: {}  cell={}x{}  fbo={}x{}",
        if is_vivante {
            "Vivante (per-instance glFinish ON)"
        } else {
            "non-Vivante"
        },
        CELL_W,
        CELL_H,
        FBO_W,
        FBO_H
    );
    println!();

    let mut results = Vec::new();
    let (fbo, rbo) = create_dst_fbo();

    let baseline = match BaselinePath::new(is_vivante) {
        Ok(b) => b,
        Err(e) => {
            println!("  SKIP: baseline path init failed: {e}");
            destroy_dst_fbo(fbo, rbo);
            return results;
        }
    };
    let max_capacity = *N_DETECT_SWEEP.iter().max().unwrap_or(&80);
    let mut instanced = match InstancedPath::new(max_capacity) {
        Ok(i) => i,
        Err(e) => {
            println!("  SKIP: instanced path init failed: {e}");
            destroy_dst_fbo(fbo, rbo);
            return results;
        }
    };

    unsafe {
        gls::gl::Enable(gls::gl::BLEND);
        gls::gl::BlendFuncSeparate(
            gls::gl::SRC_ALPHA,
            gls::gl::ONE_MINUS_SRC_ALPHA,
            gls::gl::ZERO,
            gls::gl::ONE,
        );
    }

    // Cross-validate at N=10 first so any divergence is caught before
    // we trust the timing numbers.
    {
        let dets = synthesize_detections(10);
        let masks: Vec<Vec<u8>> = (0..10).map(synthesize_mask).collect();
        let flat_masks: Vec<u8> = masks.iter().flat_map(|m| m.iter().copied()).collect();
        let (diff_px, max_diff) =
            compare_paths(&dets, &masks, &flat_masks, &baseline, &mut instanced);
        let total_px = (FBO_W * FBO_H) as u32;
        println!(
            "  Cross-validation at N=10: differing pixels = {} / {} ({:.1}%), max channel sum diff = {}",
            diff_px,
            total_px,
            100.0 * diff_px as f32 / total_px as f32,
            max_diff
        );
        // 5% threshold accommodates sub-texel filtering precision differences
        // between `sampler2D` and `sampler2DArray` on some drivers (NVIDIA).
        // A real correctness failure would diverge across the whole quad.
        if diff_px * 20 > total_px {
            println!("  WARNING: >5% pixel divergence — bench results may not be apples-to-apples");
        }
        println!();
    }

    for &n in N_DETECT_SWEEP {
        let dets = synthesize_detections(n);
        let masks: Vec<Vec<u8>> = (0..n).map(synthesize_mask).collect();
        let flat_masks: Vec<u8> = masks.iter().flat_map(|m| m.iter().copied()).collect();

        // -------- Path A: baseline_per_instance --------
        {
            let name = format!("baseline_per_instance/N={n}");
            let r = run_bench(&name, WARMUP, ITERATIONS, || unsafe {
                gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
                baseline.render(&dets, &masks);
                gls::gl::Finish();
            });
            r.print_summary();
            results.push(r);
        }

        // -------- Path B: instanced_array --------
        {
            let name = format!("instanced_array/N={n}");
            let r = run_bench(&name, WARMUP, ITERATIONS, || unsafe {
                gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
                instanced.render(&dets, &flat_masks);
                gls::gl::Finish();
            });
            r.print_summary();
            results.push(r);
        }
        println!();
    }

    // -----------------------------------------------------------------
    // Diagnostic: how much does framebuffer readback cost on this GPU?
    //
    // Hypothesis: a large fraction of HAL `draw_decoded_masks` time on
    // Mali is the `glReadnPixels` of the 640×640 RGBA8 framebuffer at
    // the end of the call, not the per-mask draw work. If this bench
    // shows readback ≈ HAL Draw - mask_render_cost, then Tier 2-a's
    // win is in the mask draw cost is small and the bigger win is to
    // make the destination DMA-BUF.
    {
        println!("== Diagnostic: framebuffer readback cost ==");
        let mut buf = vec![0u8; (FBO_W * FBO_H * 4) as usize];
        // Render something so the framebuffer has content to read back.
        unsafe {
            gls::gl::Clear(gls::gl::COLOR_BUFFER_BIT);
            let dets = synthesize_detections(20);
            let masks: Vec<Vec<u8>> = (0..20).map(synthesize_mask).collect();
            baseline.render(&dets, &masks);
            gls::gl::Finish();
        }
        let r = run_bench("readback_640x640_rgba8", WARMUP, ITERATIONS, || unsafe {
            gls::gl::ReadBuffer(gls::gl::COLOR_ATTACHMENT0);
            gls::gl::ReadnPixels(
                0,
                0,
                FBO_W,
                FBO_H,
                gls::gl::RGBA,
                gls::gl::UNSIGNED_BYTE,
                buf.len() as i32,
                buf.as_mut_ptr() as *mut c_void,
            );
        });
        r.print_summary();
        results.push(r);
        println!();
    }

    drop(baseline);
    drop(instanced);
    destroy_dst_fbo(fbo, rbo);

    results
}
