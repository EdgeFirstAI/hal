// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! GL shader sources shared across platform backends (macOS ANGLE + Linux).

/// RGBA8 -> packed RGBA16F PlanarRgb F16 fragment shader.
///
/// The render target is a single `RGBA16F` texture sized `(W/4, 3*H)`.
/// Each output texel packs 4 half-float channel samples into its four
/// components, so a linear readout of the rendered texture produces a
/// tightly-packed `[3, H, W]` F16 buffer (CHW / PlanarRgb order, one plane
/// per row-band).  Width `W` must be a multiple of 4.
///
/// This is the single source of truth consumed by both the Linux (PBO/DMA)
/// and macOS (IOSurface/ANGLE) render paths. The GLSL bytes are preserved
/// exactly as validated by the on-target Linux F16 round-trip tests and the
/// macOS IOSurface integration tests.
pub(crate) const PLANAR_RGB_F16_PACKED_FRAGMENT: &str = r#"#version 300 es
precision highp float;
precision highp int;
uniform sampler2D src;
uniform vec2 dst_image_size;  // (W, H) — destination plane size
uniform vec4 src_rect_uv;     // (origin_u, origin_v, size_u, size_v)
uniform vec4 dst_rect_px;     // (origin_x, origin_y, w, h) within one plane
uniform vec3 pad_color;       // per-channel normalized pad value
out vec4 frag;

// Sample one planar element. Returns the per-channel value at
// (in_plane_x, in_plane_y) for the given plane (0=R, 1=G, 2=B).
// Pad value is returned when (in_plane_x, in_plane_y) is outside
// `dst_rect_px`.
float sample_planar_element(float in_plane_x, float in_plane_y, float plane) {
    bool inside_dst = (in_plane_x >= dst_rect_px.x) &&
                      (in_plane_x <  dst_rect_px.x + dst_rect_px.z) &&
                      (in_plane_y >= dst_rect_px.y) &&
                      (in_plane_y <  dst_rect_px.y + dst_rect_px.w);
    if (inside_dst) {
        float u = (in_plane_x - dst_rect_px.x) / dst_rect_px.z;
        float v = (in_plane_y - dst_rect_px.y) / dst_rect_px.w;
        vec2 src_uv = src_rect_uv.xy + vec2(u, v) * src_rect_uv.zw;
        vec4 rgba = texture(src, src_uv);
        if (plane < 0.5) {
            return rgba.r;
        } else if (plane < 1.5) {
            return rgba.g;
        } else {
            return rgba.b;
        }
    } else {
        if (plane < 0.5) {
            return pad_color.r;
        } else if (plane < 1.5) {
            return pad_color.g;
        } else {
            return pad_color.b;
        }
    }
}

void main() {
    // gl_FragCoord is at pixel center (n+0.5). Floor for the integer
    // index of the output pixel on the (W/4 × 3H) surface.
    int ox = int(floor(gl_FragCoord.x));
    int oy = int(floor(gl_FragCoord.y));

    float plane = floor(float(oy) / dst_image_size.y);
    float in_plane_y = float(oy) - plane * dst_image_size.y + 0.5;

    // Sample 4 consecutive in-plane x positions. Pixel center is
    // (x+0.5) so the first element of pixel `ox` starts at logical
    // tensor column `ox*4` — add 0.5 for the texel center sampled
    // by the bilinear filter.
    float base_x = float(ox * 4) + 0.5;
    float e0 = sample_planar_element(base_x + 0.0, in_plane_y, plane);
    float e1 = sample_planar_element(base_x + 1.0, in_plane_y, plane);
    float e2 = sample_planar_element(base_x + 2.0, in_plane_y, plane);
    float e3 = sample_planar_element(base_x + 3.0, in_plane_y, plane);

    // RGBA16F framebuffer narrows each f32 to f16 on write. The
    // resulting byte layout in the locked IOSurface is exactly the
    // [3, H, W] f16 planar order the consumer expects.
    frag = vec4(e0, e1, e2, e3);
}
"#;
