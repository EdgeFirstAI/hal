#version 300 es
#extension GL_EXT_YUV_target : require
precision highp float;
uniform __samplerExternal2DY2YEXT src;
uniform highp vec4 src_extent;
// Per-tensor colorimetry (YUV→RGB matrix + range), set by draw_nv_texture_2d
// from the source tensor's resolved colorimetry. Path B applies the matrix in
// the shader, so it is correct regardless of driver EGL color-hint support.
uniform float y_offset;
uniform float y_scale;
uniform float c_vr;
uniform float c_ug;
uniform float c_vg;
uniform float c_ub;
in vec3 fragPos;
in highp vec2 tc;
out vec4 color;

// Floor expanded luma at 0 to match the CPU `yuv` crate's saturating (Y-16)
// term (limited footroom Y<16 → 0). The top is left uncapped — the crate lets
// headroom exceed 1.0 and relies on the final RGB clamp, so the GL path must
// too. No-op for full range (y_offset=0, y_scale=1).
vec3 nv_yuv_to_rgb(float yv, float u, float v) {
    float yp = max((yv - y_offset) * y_scale, 0.0);
    float up = u - 128.0 / 255.0;
    float vp = v - 128.0 / 255.0;
    return clamp(vec3(yp + c_vr * vp, yp - c_ug * up - c_vg * vp, yp + c_ub * up), 0.0, 1.0);
}

void main() {
    vec3 yuv = texture(src, clamp(tc, src_extent.xy, src_extent.zw)).rgb;
    vec3 rgb = nv_yuv_to_rgb(yuv.r, yuv.g, yuv.b);
    color = vec4(rgb, 1.0);
}
