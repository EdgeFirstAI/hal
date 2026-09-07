// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// S2: YUV coverage. Native DXGI_FORMAT_NV12 through the plane import, the
// HAL's combined-plane R8 layout (NV12 / NV16 / NV24 as one R8 texture,
// decoded by the HAL's Path B shader verbatim), packed YUYV / VYUY as an
// RG8 texture, odd geometry, and the EGLStream route Chromium uses.
#include "probe.h"

#include <algorithm>
#include <cstring>

// BT.601 limited range, in the shader's normalized form.
static const float kYOffset = 16.0f / 255.0f;
static const float kYScale = 255.0f / 219.0f;
static const float kCvr = 1.596f, kCug = 0.391f, kCvg = 0.813f, kCub = 2.018f;

static void set_colorimetry(GLuint prog) {
  glUniform1f(glGetUniformLocation(prog, "y_offset"), kYOffset);
  glUniform1f(glGetUniformLocation(prog, "y_scale"), kYScale);
  glUniform1f(glGetUniformLocation(prog, "c_vr"), kCvr);
  glUniform1f(glGetUniformLocation(prog, "c_ug"), kCug);
  glUniform1f(glGetUniformLocation(prog, "c_vg"), kCvg);
  glUniform1f(glGetUniformLocation(prog, "c_ub"), kCub);
}

static void cpu_yuv_to_rgb(uint8_t Y, uint8_t U, uint8_t V, uint8_t* out) {
  float yp = std::max((Y / 255.0f - kYOffset) * kYScale, 0.0f);
  float up = U / 255.0f - 128.0f / 255.0f;
  float vp = V / 255.0f - 128.0f / 255.0f;
  float r = std::clamp(yp + kCvr * vp, 0.0f, 1.0f);
  float g = std::clamp(yp - kCug * up - kCvg * vp, 0.0f, 1.0f);
  float b = std::clamp(yp + kCub * up, 0.0f, 1.0f);
  out[0] = (uint8_t)(r * 255.0f + 0.5f);
  out[1] = (uint8_t)(g * 255.0f + 0.5f);
  out[2] = (uint8_t)(b * 255.0f + 0.5f);
  out[3] = 255;
}

// Count pixels whose RGB differs from the reference by more than `tol`.
static size_t rgb_mismatch(const std::vector<uint8_t>& got, const std::vector<uint8_t>& want, int tol,
                           size_t* first) {
  size_t bad = 0;
  if (first) *first = SIZE_MAX;
  for (size_t p = 0; p + 3 < got.size(); p += 4) {
    int d = std::max({abs((int)got[p] - want[p]), abs((int)got[p + 1] - want[p + 1]),
                      abs((int)got[p + 2] - want[p + 2])});
    if (d > tol) {
      if (first && *first == SIZE_MAX) *first = p / 4;
      bad++;
    }
  }
  return bad;
}

// Semi-planar test image: Y = pattern channel 0, U = channel 1, V = channel 2
// sampled at the chroma grid for the given subsampling shifts.
struct YuvImage {
  uint32_t w, h, sx, sy;
  std::vector<uint8_t> y, u, v;  // full-res Y; chroma at (w >> sx) x (h >> sy)
  void init(uint32_t W, uint32_t H, uint32_t shift_x, uint32_t shift_y) {
    w = W;
    h = H;
    sx = shift_x;
    sy = shift_y;
    y.resize((size_t)w * h);
    for (uint32_t yy = 0; yy < h; yy++)
      for (uint32_t x = 0; x < w; x++) y[(size_t)yy * w + x] = (uint8_t)(16 + (pattern_k(x, yy, 0, 1) * 219) / 255);
    uint32_t cw = w >> sx, ch = h >> sy;
    u.resize((size_t)cw * ch);
    v.resize((size_t)cw * ch);
    for (uint32_t cy = 0; cy < ch; cy++)
      for (uint32_t cx = 0; cx < cw; cx++) {
        u[(size_t)cy * cw + cx] = (uint8_t)(16 + (pattern_k(cx, cy, 1, 1) * 224) / 255);
        v[(size_t)cy * cw + cx] = (uint8_t)(16 + (pattern_k(cx, cy, 2, 1) * 224) / 255);
      }
  }
  std::vector<uint8_t> reference_rgba() const {
    std::vector<uint8_t> out((size_t)w * h * 4);
    uint32_t cw = w >> sx;
    for (uint32_t yy = 0; yy < h; yy++)
      for (uint32_t x = 0; x < w; x++) {
        size_t ci = (size_t)(yy >> sy) * cw + (x >> sx);
        cpu_yuv_to_rgb(y[(size_t)yy * w + x], u[ci], v[ci], &out[((size_t)yy * w + x) * 4]);
      }
    return out;
  }
  // Tight NV12-style buffer: Y rows then interleaved UV rows (row = w bytes
  // for 4:2:0 / 4:2:2, 2w bytes for 4:4:4).
  std::vector<uint8_t> semi_planar() const {
    uint32_t cw = w >> sx, ch = h >> sy;
    std::vector<uint8_t> out(y);
    out.resize((size_t)w * h + (size_t)cw * ch * 2);
    uint8_t* uv = out.data() + (size_t)w * h;
    for (size_t i = 0; i < (size_t)cw * ch; i++) {
      uv[i * 2] = u[i];
      uv[i * 2 + 1] = v[i];
    }
    return out;
  }
};

// Sample one imported plane (R8 or RG8) into an RGBA8 FBO; returns bytes.
static bool sample_plane(GLuint prog, Quad& quad, GlSession& s, Import& im, uint32_t w, uint32_t h,
                         std::vector<uint8_t>& out) {
  PlainFbo fbo;
  if (!fbo.init(w, h, false)) return false;
  glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo);
  glViewport(0, 0, (GLsizei)w, (GLsizei)h);
  glUseProgram(prog);
  glActiveTexture(GL_TEXTURE0);
  bool ok = import_bind(s, im);
  glUniform1i(glGetUniformLocation(prog, "tex"), 0);
  quad.draw();
  import_release(s, im);
  read_fbo_rgba8(w, h, out);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  fbo.destroy();
  return ok;
}

static const char* kVyuyRgbaFragment =
    "#version 300 es\n"
    "precision highp float;\n"
    "uniform highp sampler2D tex;\n"
    "uniform vec2 src_size;\n"
    "uniform float y_offset;\n"
    "uniform float y_scale;\n"
    "uniform float c_vr;\n"
    "uniform float c_ug;\n"
    "uniform float c_vg;\n"
    "uniform float c_ub;\n"
    "in vec3 fragPos;\n"
    "in vec2 tc;\n"
    "out vec4 color;\n"
    "void main() {\n"
    "    vec2 texel = vec2(1.0) / src_size;\n"
    "    vec2 col = floor(tc * src_size);\n"
    "    bool even = mod(col.x, 2.0) < 0.5;\n"
    "    vec2 self_uv = (col + vec2(0.5)) * texel;\n"
    "    vec2 pair_uv = (col + vec2(even ? 1.5 : -0.5, 0.5)) * texel;\n"
    "    vec4 self_rg = texture(tex, self_uv);\n"
    "    vec4 pair_rg = texture(tex, pair_uv);\n"
    // VYUY: byte order V Y0 U Y1, so luma is the G channel of every RG texel
    // and chroma the R channel, V on even texels and U on odd ones.
    "    float y = self_rg.g;\n"
    "    float u, v;\n"
    "    if (even) { v = self_rg.r; u = pair_rg.r; }\n"
    "    else      { u = self_rg.r; v = pair_rg.r; }\n"
    "    float yp = max((y - y_offset) * y_scale, 0.0);\n"
    "    float up = u - 128.0 / 255.0;\n"
    "    float vp = v - 128.0 / 255.0;\n"
    "    float r = clamp(yp + c_vr * vp, 0.0, 1.0);\n"
    "    float g = clamp(yp - c_ug * up - c_vg * vp, 0.0, 1.0);\n"
    "    float b = clamp(yp + c_ub * up, 0.0, 1.0);\n"
    "    color = vec4(r, g, b, 1.0);\n"
    "}\n";

static const char* kNv12ExternalFragment =
    "#version 300 es\n"
    "#extension GL_OES_EGL_image_external_essl3 : require\n"
    "precision highp float;\n"
    "uniform samplerExternalOES y_tex;\n"
    "uniform samplerExternalOES uv_tex;\n"
    "uniform float y_offset;\n"
    "uniform float y_scale;\n"
    "uniform float c_vr;\n"
    "uniform float c_ug;\n"
    "uniform float c_vg;\n"
    "uniform float c_ub;\n"
    "in vec3 fragPos;\n"
    "in vec2 tc;\n"
    "out vec4 color;\n"
    "void main() {\n"
    "    float yv = texture(y_tex, tc).r;\n"
    "    vec2 uv = texture(uv_tex, tc).rg;\n"
    "    float yp = max((yv - y_offset) * y_scale, 0.0);\n"
    "    float up = uv.x - 128.0 / 255.0;\n"
    "    float vp = uv.y - 128.0 / 255.0;\n"
    "    float r = clamp(yp + c_vr * vp, 0.0, 1.0);\n"
    "    float g = clamp(yp - c_ug * up - c_vg * vp, 0.0, 1.0);\n"
    "    float b = clamp(yp + c_ub * up, 0.0, 1.0);\n"
    "    color = vec4(r, g, b, 1.0);\n"
    "}\n";

void run_s2(GlSession& s) {
  Quad quad;
  quad.init();
  std::string log;
  GLuint p_sample = compile_program(kVertexShader, kSampleFragment, &log);
  GLuint p_two = compile_program(kVertexShader, kNv12TwoPlaneFragment, &log);
  GLuint p_nv = compile_program(kVertexShader, kNvRgbaFragment, &log);
  if (!p_nv) report("S2.0", Verdict::Fail, "HAL NV_RGBA_FRAGMENT did not compile: %s", log.c_str());
  GLuint p_yuyv = compile_program(kVertexShader, kYuyvRgbaFragment, &log);
  if (!p_yuyv) report("S2.0", Verdict::Fail, "HAL YUYV_RGBA_2D_FRAGMENT did not compile: %s", log.c_str());
  GLuint p_vyuy = compile_program(kVertexShader, kVyuyRgbaFragment, &log);
  GLuint p_ext = compile_program(kVertexShader, kNv12ExternalFragment, &log);
  report("S2.0", p_ext ? Verdict::Pass : Verdict::Info, "samplerExternalOES (essl3) program: %s",
         p_ext ? "compiles" : log.c_str());
  const int kTol = 2;

  // ---- Native NV12 through the plane import -----------------------------
  const uint32_t W = 256, H = 128;
  YuvImage img;
  img.init(W, H, 1, 1);
  HRESULT hr;
  ComPtr<ID3D11Texture2D> nv12 = create_tex(s.d3d, W, H, DXGI_FORMAT_NV12,
                                            D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0, &hr);
  if (!nv12) {
    report("S2.1", Verdict::Fail, "CreateTexture2D NV12 %ux%u RT|SR: %s", W, H, hr_str(hr));
  } else {
    if (!upload_tex(s.d3d, nv12.Get(), img.semi_planar()))
      report("S2.1", Verdict::Fail, "NV12 upload through staging failed");
    std::vector<uint8_t> back;
    UINT pitch = 0;
    readback_tex(s.d3d, nv12.Get(), back, &pitch);
    auto sp = img.semi_planar();
    size_t bad = count_mismatch(back.data(), sp.data(), sp.size());
    report("S2.1", bad == 0 ? Verdict::Pass : Verdict::Fail,
           "NV12 %ux%u staging round trip (UV plane at RowPitch*Height): %zu mismatches, pitch %u", W, H, bad,
           pitch);
    for (Route route : {Route::Pbuffer, Route::EglImage}) {
      Import y = import_texture(s, route, nv12.Get(), 0, 0);
      Import uv = import_texture(s, route, nv12.Get(), 0, 1);
      if (!y.ok || !uv.ok) {
        report("S2.2", Verdict::Fail, "%-8s NV12 plane import: Y %s/0x%04x UV %s/0x%04x", route_name(route),
               egl_err_str(y.egl_error), y.gl_error, egl_err_str(uv.egl_error), uv.gl_error);
        import_destroy(s, y);
        import_destroy(s, uv);
        continue;
      }
      // Plane 0 sampled as R8.
      std::vector<uint8_t> got;
      sample_plane(p_sample, quad, s, y, W, H, got);
      size_t ybad = 0;
      for (size_t i = 0; i < (size_t)W * H; i++) ybad += got[i * 4] != img.y[i];
      report("S2.2", ybad == 0 ? Verdict::Pass : Verdict::Fail, "%-8s NV12 plane 0 sampled as R8: %zu mismatches",
             route_name(route), ybad);
      // Plane 1 sampled as RG8 at half size.
      sample_plane(p_sample, quad, s, uv, W / 2, H / 2, got);
      size_t uvbad = 0;
      for (size_t i = 0; i < (size_t)(W / 2) * (H / 2); i++)
        uvbad += (got[i * 4] != img.u[i]) + (got[i * 4 + 1] != img.v[i]);
      report("S2.2", uvbad == 0 ? Verdict::Pass : Verdict::Fail, "%-8s NV12 plane 1 sampled as RG8: %zu mismatches",
             route_name(route), uvbad);
      // Full decode with the two-sampler shader.
      PlainFbo fbo;
      fbo.init(W, H, false);
      glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo);
      glViewport(0, 0, W, H);
      glUseProgram(p_two);
      set_colorimetry(p_two);
      glActiveTexture(GL_TEXTURE0);
      import_bind(s, y);
      glActiveTexture(GL_TEXTURE1);
      import_bind(s, uv);
      glUniform1i(glGetUniformLocation(p_two, "y_tex"), 0);
      glUniform1i(glGetUniformLocation(p_two, "uv_tex"), 1);
      quad.draw();
      import_release(s, uv);
      glActiveTexture(GL_TEXTURE0);
      import_release(s, y);
      read_fbo_rgba8(W, H, got);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      fbo.destroy();
      size_t first;
      size_t dbad = rgb_mismatch(got, img.reference_rgba(), kTol, &first);
      report("S2.3", dbad == 0 ? Verdict::Pass : Verdict::Fail,
             "%-8s NV12 two-plane decode vs CPU BT.601 (tol %d): %zu pixels off", route_name(route), kTol, dbad);
      // Planes as render targets (GL writing NV12).
      if (route == Route::EglImage) {
        for (int plane = 0; plane < 2; plane++) {
          Import& im = plane == 0 ? y : uv;
          GLuint fb;
          glGenFramebuffers(1, &fb);
          glBindFramebuffer(GL_FRAMEBUFFER, fb);
          glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, im.tex, 0);
          GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
          report("S2.4", Verdict::Info, "%-8s NV12 plane %d as FBO colour attachment: %s", route_name(route),
                 plane, gl_fbo_status_str(st));
          glBindFramebuffer(GL_FRAMEBUFFER, 0);
          glDeleteFramebuffers(1, &fb);
        }
      }
      import_destroy(s, y);
      import_destroy(s, uv);
      gl_clear_errors("S2 nv12");
    }
  }
  // Odd geometry for native NV12.
  {
    ComPtr<ID3D11Texture2D> odd = create_tex(s.d3d, 641, 481, DXGI_FORMAT_NV12, D3D11_BIND_SHADER_RESOURCE, 0, &hr);
    report("S2.5", Verdict::Info, "CreateTexture2D NV12 641x481: %s", odd ? "accepted" : hr_str(hr));
    ComPtr<ID3D11Texture2D> odd2 = create_tex(s.d3d, 642, 481, DXGI_FORMAT_NV12, D3D11_BIND_SHADER_RESOURCE, 0, &hr);
    report("S2.5", Verdict::Info, "CreateTexture2D NV12 642x481: %s", odd2 ? "accepted" : hr_str(hr));
  }

  // ---- HAL combined-plane R8 layout: NV12 / NV16 / NV24 -------------------
  struct NvCase {
    const char* name;
    uint32_t sx, sy;
    int chroma_lines;
  };
  static const NvCase nv_cases[] = {{"NV12", 1, 1, 1}, {"NV16", 1, 0, 1}, {"NV24", 0, 0, 2}};
  if (p_nv) {
    for (const NvCase& c : nv_cases) {
      for (uint32_t w : {256u, 254u}) {
        // Odd logical widths need the HAL's stride rules; the probe keeps
        // even widths and varies the height instead.
        for (uint32_t h : {128u, 126u}) {
          YuvImage im;
          im.init(w, h, c.sx, c.sy);
          uint32_t cw = w >> c.sx, ch = h >> c.sy;
          uint32_t chroma_rows = (cw * 2 * ch) / w;  // chroma bytes / tex_width
          uint32_t rows = h + chroma_rows;
          std::vector<uint8_t> combined = im.semi_planar();
          if (combined.size() != (size_t)w * rows) {
            report("S2.6", Verdict::Fail, "%s combined plane %ux%u: size %zu != %u rows", c.name, w, h,
                   combined.size(), rows);
            continue;
          }
          ComPtr<ID3D11Texture2D> tex = create_tex(s.d3d, w, rows, DXGI_FORMAT_R8_UNORM,
                                                   D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, 0, &hr);
          if (!tex) {
            report("S2.6", Verdict::Fail, "%s R8 %ux%u: %s", c.name, w, rows, hr_str(hr));
            continue;
          }
          upload_tex(s.d3d, tex.Get(), combined);
          for (Route route : {Route::Pbuffer, Route::EglImage}) {
            Import ri = import_texture(s, route, tex.Get(), GL_RED_EXT);
            if (!ri.ok) {
              report("S2.6", Verdict::Fail, "%s %s R8 import: %s", c.name, route_name(route), egl_err_str(ri.egl_error));
              import_destroy(s, ri);
              continue;
            }
            PlainFbo fbo;
            fbo.init(w, h, false);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo);
            glViewport(0, 0, (GLsizei)w, (GLsizei)h);
            glUseProgram(p_nv);
            set_colorimetry(p_nv);
            glUniform2i(glGetUniformLocation(p_nv, "img_size"), (GLint)w, (GLint)h);
            glUniform1i(glGetUniformLocation(p_nv, "tex_width"), (GLint)w);
            glUniform2i(glGetUniformLocation(p_nv, "chroma_shift"), (GLint)c.sx, (GLint)c.sy);
            glUniform1i(glGetUniformLocation(p_nv, "chroma_lines"), c.chroma_lines);
            glActiveTexture(GL_TEXTURE0);
            import_bind(s, ri);
            glUniform1i(glGetUniformLocation(p_nv, "src"), 0);
            quad.draw();
            import_release(s, ri);
            std::vector<uint8_t> got;
            read_fbo_rgba8(w, h, got);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            fbo.destroy();
            size_t first;
            size_t bad = rgb_mismatch(got, im.reference_rgba(), kTol, &first);
            report("S2.6", bad == 0 ? Verdict::Pass : Verdict::Fail,
                   "%s combined-plane R8 %ux%u via %-8s + HAL Path B shader: %zu pixels off", c.name, w, h,
                   route_name(route), bad);
            import_destroy(s, ri);
          }
        }
      }
    }
  }

  // ---- Packed YUYV / VYUY as RG8 -------------------------------------------
  if (p_yuyv && p_vyuy) {
    YuvImage im;
    im.init(W, H, 1, 0);  // 4:2:2
    for (int vyuy = 0; vyuy < 2; vyuy++) {
      std::vector<uint8_t> packed((size_t)W * H * 2);
      for (uint32_t y = 0; y < H; y++)
        for (uint32_t x = 0; x < W; x += 2) {
          size_t o = ((size_t)y * W + x) * 2;
          uint8_t Y0 = im.y[(size_t)y * W + x], Y1 = im.y[(size_t)y * W + x + 1];
          uint8_t U = im.u[(size_t)y * (W / 2) + x / 2], V = im.v[(size_t)y * (W / 2) + x / 2];
          if (!vyuy) {
            packed[o] = Y0; packed[o + 1] = U; packed[o + 2] = Y1; packed[o + 3] = V;
          } else {
            packed[o] = V; packed[o + 1] = Y0; packed[o + 2] = U; packed[o + 3] = Y1;
          }
        }
      // RENDER_TARGET is included because the pbuffer route refuses
      // textures without it (S1.7), even when only sampled.
      ComPtr<ID3D11Texture2D> tex = create_tex(s.d3d, W, H, DXGI_FORMAT_R8G8_UNORM,
                                               D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET, 0, &hr);
      upload_tex(s.d3d, tex.Get(), packed);
      for (Route route : {Route::Pbuffer, Route::EglImage}) {
        Import ri = import_texture(s, route, tex.Get(), GL_RG_EXT);
        if (!ri.ok) {
          report("S2.7", Verdict::Fail, "%s %s RG8 import: %s", vyuy ? "VYUY" : "YUYV", route_name(route),
                 egl_err_str(ri.egl_error));
          import_destroy(s, ri);
          continue;
        }
        GLuint prog = vyuy ? p_vyuy : p_yuyv;
        PlainFbo fbo;
        fbo.init(W, H, false);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo);
        glViewport(0, 0, W, H);
        glUseProgram(prog);
        set_colorimetry(prog);
        glUniform2f(glGetUniformLocation(prog, "src_size"), (float)W, (float)H);
        // The HAL sets a sample clamp alongside src_size (render::sample_clamp_rect).
        // This texture is exactly the logical image, so the clamp is the whole
        // texture inset half a texel. The swizzled VYUY variant has no such
        // uniform; its location is -1 and the call is a no-op.
        glUniform4f(glGetUniformLocation(prog, "src_extent"), 0.5f / W, 0.5f / H,
                    (W - 0.5f) / W, (H - 0.5f) / H);
        glActiveTexture(GL_TEXTURE0);
        import_bind(s, ri);
        glUniform1i(glGetUniformLocation(prog, "tex"), 0);
        quad.draw();
        import_release(s, ri);
        std::vector<uint8_t> got;
        read_fbo_rgba8(W, H, got);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        fbo.destroy();
        size_t first;
        size_t bad = rgb_mismatch(got, im.reference_rgba(), kTol, &first);
        report("S2.7", bad == 0 ? Verdict::Pass : Verdict::Fail, "%s as RG8 via %-8s + %s shader: %zu pixels off",
               vyuy ? "VYUY" : "YUYV", route_name(route), vyuy ? "swizzled" : "HAL YUYV", bad);
        import_destroy(s, ri);
      }
    }
  }

  // ---- EGLStream NV12 (producer D3D texture, consumer external textures) --
  if (nv12 && p_ext && eglCreateStreamKHR && eglStreamConsumerGLTextureExternalAttribsNV &&
      eglCreateStreamProducerD3DTextureANGLE && eglStreamPostD3DTextureANGLE && eglStreamConsumerAcquireKHR) {
    upload_tex(s.d3d, nv12.Get(), img.semi_planar());
    const EGLint stream_attribs[] = {EGL_NONE};
    EGLStreamKHR stream = eglCreateStreamKHR(s.dpy, stream_attribs);
    std::string step;
    if (stream == EGL_NO_STREAM_KHR) step = std::string("eglCreateStreamKHR ") + egl_err_str(eglGetError());
    GLuint ext_tex[2] = {0, 0};
    if (step.empty()) {
      glGenTextures(2, ext_tex);
      for (int i = 0; i < 2; i++) {
        glActiveTexture(GL_TEXTURE0 + i);
        glBindTexture(GL_TEXTURE_EXTERNAL_OES, ext_tex[i]);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      }
      const EGLAttrib consumer[] = {EGL_COLOR_BUFFER_TYPE, EGL_YUV_BUFFER_EXT, EGL_YUV_NUMBER_OF_PLANES_EXT, 2,
                                    EGL_YUV_PLANE0_TEXTURE_UNIT_NV, 0, EGL_YUV_PLANE1_TEXTURE_UNIT_NV, 1, EGL_NONE};
      if (!eglStreamConsumerGLTextureExternalAttribsNV(s.dpy, stream, consumer))
        step = std::string("eglStreamConsumerGLTextureExternalAttribsNV ") + egl_err_str(eglGetError());
    }
    if (step.empty()) {
      const EGLAttrib producer[] = {EGL_NONE};
      if (!eglCreateStreamProducerD3DTextureANGLE(s.dpy, stream, producer))
        step = std::string("eglCreateStreamProducerD3DTextureANGLE ") + egl_err_str(eglGetError());
    }
    if (step.empty()) {
      const EGLAttrib post[] = {EGL_D3D_TEXTURE_SUBRESOURCE_ID_ANGLE, 0, EGL_NONE};
      if (!eglStreamPostD3DTextureANGLE(s.dpy, stream, nv12.Get(), post))
        step = std::string("eglStreamPostD3DTextureANGLE ") + egl_err_str(eglGetError());
    }
    if (step.empty() && !eglStreamConsumerAcquireKHR(s.dpy, stream))
      step = std::string("eglStreamConsumerAcquireKHR ") + egl_err_str(eglGetError());
    if (step.empty()) {
      PlainFbo fbo;
      fbo.init(W, H, false);
      glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo);
      glViewport(0, 0, W, H);
      glUseProgram(p_ext);
      set_colorimetry(p_ext);
      glUniform1i(glGetUniformLocation(p_ext, "y_tex"), 0);
      glUniform1i(glGetUniformLocation(p_ext, "uv_tex"), 1);
      quad.draw();
      std::vector<uint8_t> got;
      read_fbo_rgba8(W, H, got);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      fbo.destroy();
      size_t first;
      size_t bad = rgb_mismatch(got, img.reference_rgba(), kTol, &first);
      report("S2.8", bad == 0 ? Verdict::Pass : Verdict::Fail,
             "EGLStream NV12 (D3D producer -> two samplerExternalOES) decode: %zu pixels off", bad);
      if (eglStreamConsumerReleaseKHR) eglStreamConsumerReleaseKHR(s.dpy, stream);
    } else {
      report("S2.8", Verdict::Info, "EGLStream NV12 route stopped at %s", step.c_str());
    }
    if (ext_tex[0]) glDeleteTextures(2, ext_tex);
    if (stream != EGL_NO_STREAM_KHR && eglDestroyStreamKHR) eglDestroyStreamKHR(s.dpy, stream);
    glActiveTexture(GL_TEXTURE0);
  } else {
    report("S2.8", Verdict::Skip, "EGLStream NV12 route: entry points or shader unavailable");
  }

  if (p_sample) glDeleteProgram(p_sample);
  if (p_two) glDeleteProgram(p_two);
  if (p_nv) glDeleteProgram(p_nv);
  if (p_yuyv) glDeleteProgram(p_yuyv);
  if (p_vyuy) glDeleteProgram(p_vyuy);
  if (p_ext) glDeleteProgram(p_ext);
  quad.destroy();
  gl_clear_errors("S2 end");
}
