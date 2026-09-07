// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// S0: harness sanity. S1: D3D11 texture import through both EGL routes for
// every format ANGLE accepts, as sampled source and as render target, with
// creation, bind and convert timings and the bind/misc flag matrix.
#include "probe.h"

#include <cstring>

static bool has_ext(const std::string& s, const char* tok) {
  std::string padded = " " + s + " ";
  std::string needle = std::string(" ") + tok + " ";
  return padded.find(needle) != std::string::npos;
}

// ---------------------------------------------------------------------------
// S0
// ---------------------------------------------------------------------------

void run_s0(GlSession& s) {
  report("S0.1", Verdict::Pass, "display + ES 3.%d context up on %s", s.es_minor, mode_name(s.mode));
  report("S0.2", s.d3d.dev ? Verdict::Pass : Verdict::Fail,
         "ANGLE's ID3D11Device queried through EGL_ANGLE_device_d3d (FL 0x%x, flags 0x%x)",
         s.d3d.feature_level, s.d3d.creation_flags);
  ComPtr<ID3D11Multithread> mt;
  HRESULT hr = s.d3d.dev->QueryInterface(IID_PPV_ARGS(&mt));
  report("S0.3", Verdict::Info, "ID3D11Multithread: %s, protected=%d", hr_str(hr),
         mt ? (int)mt->GetMultithreadProtected() : -1);

  static const char* egl_required[] = {"EGL_ANGLE_d3d_texture_client_buffer",
                                       "EGL_ANGLE_image_d3d11_texture", "EGL_KHR_image_base",
                                       "EGL_ANGLE_keyed_mutex", "EGL_ANGLE_stream_producer_d3d_texture"};
  for (const char* e : egl_required)
    report("S0.4", has_ext(s.display_exts, e) ? Verdict::Pass : Verdict::Fail, "display advertises %s", e);
  static const char* egl_absent[] = {"EGL_KHR_fence_sync", "EGL_ANDROID_native_fence_sync",
                                     "EGL_ANGLE_global_fence_sync"};
  for (const char* e : egl_absent)
    report("S0.5", Verdict::Info, "%s: %s", e, has_ext(s.display_exts, e) ? "present" : "absent");
  static const char* gl_required[] = {"GL_OES_EGL_image", "GL_EXT_texture_format_BGRA8888",
                                      "GL_EXT_texture_rg", "GL_EXT_color_buffer_float",
                                      "GL_EXT_color_buffer_half_float", "GL_EXT_texture_norm16"};
  for (const char* e : gl_required)
    report("S0.6", has_ext(s.gl_exts, e) ? Verdict::Pass : Verdict::Fail, "GL advertises %s", e);
  report("S0.7", Verdict::Info, "device extensions: %s", s.device_exts.c_str());

  // Plain FBO clear + readback.
  Quad quad;
  quad.init();
  PlainFbo fbo;
  if (!fbo.init(64, 32, false)) {
    report("S0.8", Verdict::Fail, "plain RGBA8 FBO incomplete");
  } else {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo);
    glViewport(0, 0, 64, 32);
    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    std::vector<uint8_t> px;
    read_fbo_rgba8(64, 32, px);
    // 0.5 * 255 = 127.5 lands on either side depending on the driver's
    // UNORM rounding; both are correct.
    bool ok = px.size() == 64 * 32 * 4 && px[0] == 64 && (px[1] == 127 || px[1] == 128) &&
              px[2] == 191 && px[3] == 255;
    report("S0.8", ok ? Verdict::Pass : Verdict::Fail, "plain FBO clear + glReadPixels (%d,%d,%d,%d)",
           px[0], px[1], px[2], px[3]);
  }
  fbo.destroy();

  // D3D11 texture round trip through staging and through UpdateSubresource.
  for (int via_update = 0; via_update < 2; via_update++) {
    ComPtr<ID3D11Texture2D> tex = create_tex(s.d3d, 256, 128, DXGI_FORMAT_R8G8B8A8_UNORM,
                                             D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0, &hr);
    if (!tex) {
      report("S0.9", Verdict::Fail, "CreateTexture2D RGBA8 256x128: %s", hr_str(hr));
      continue;
    }
    auto pat = make_pattern(DXGI_FORMAT_R8G8B8A8_UNORM, 256, 128);
    upload_tex(s.d3d, tex.Get(), pat, via_update != 0);
    std::vector<uint8_t> back;
    UINT pitch = 0;
    readback_tex(s.d3d, tex.Get(), back, &pitch);
    size_t bad = count_mismatch(pat.data(), back.data(), pat.size());
    report("S0.9", bad == 0 ? Verdict::Pass : Verdict::Fail,
           "D3D11 RGBA8 256x128 upload via %s + staging readback: %zu mismatches (staging pitch %u)",
           via_update ? "UpdateSubresource" : "staging Map/CopyResource", bad, pitch);
  }
  quad.destroy();
  gl_clear_errors("S0");
}

// ---------------------------------------------------------------------------
// S1 helpers
// ---------------------------------------------------------------------------

static bool is_float_fmt(DXGI_FORMAT f) {
  return f == DXGI_FORMAT_R16G16B16A16_FLOAT || f == DXGI_FORMAT_R32G32B32A32_FLOAT;
}

struct Programs {
  GLuint sample = 0, gradient = 0;
  GLint sample_tex = -1, gradient_denom = -1;
  bool init() {
    std::string log;
    sample = compile_program(kVertexShader, kSampleFragment, &log);
    if (!sample) {
      report("S1.0", Verdict::Fail, "sample program: %s", log.c_str());
      return false;
    }
    gradient = compile_program(kVertexShader, kGradientFragment, &log);
    if (!gradient) {
      report("S1.0", Verdict::Fail, "gradient program: %s", log.c_str());
      return false;
    }
    sample_tex = glGetUniformLocation(sample, "tex");
    gradient_denom = glGetUniformLocation(gradient, "denom");
    return true;
  }
  void destroy() {
    if (sample) glDeleteProgram(sample);
    if (gradient) glDeleteProgram(gradient);
  }
};

// Sample the imported texture into a plain FBO and compare with the pattern.
// Returns "ok", "ok (flipped)" or a mismatch description.
static std::string check_source(GlSession& s, Programs& pr, Quad& quad, Import& im, DXGI_FORMAT fmt,
                                uint32_t w, uint32_t h, uint32_t seed) {
  bool flt = is_float_fmt(fmt);
  PlainFbo fbo;
  if (!fbo.init(w, h, flt)) return "plain FBO incomplete";
  glBindFramebuffer(GL_FRAMEBUFFER, fbo.fbo);
  glViewport(0, 0, (GLsizei)w, (GLsizei)h);
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT);
  glUseProgram(pr.sample);
  glActiveTexture(GL_TEXTURE0);
  if (!import_bind(s, im)) {
    fbo.destroy();
    return std::string("bind failed: ") + egl_err_str(im.egl_error);
  }
  glUniform1i(pr.sample_tex, 0);
  quad.draw();
  import_release(s, im);
  glBindTexture(GL_TEXTURE_2D, 0);
  std::string result;
  auto expect = pattern_as_rgba_f32(fmt, w, h, seed);
  std::vector<float> expect_flip((size_t)w * h * 4);
  for (uint32_t y = 0; y < h; y++)
    memcpy(&expect_flip[(size_t)y * w * 4], &expect[(size_t)(h - 1 - y) * w * 4], w * 16);
  if (flt) {
    std::vector<float> got;
    read_fbo_rgba_f32(w, h, got);
    size_t first;
    size_t bad = count_mismatch_f32(got.data(), expect.data(), got.size(), 1e-6f, &first);
    if (bad == 0) {
      result = "ok";
    } else {
      size_t bad2 = count_mismatch_f32(got.data(), expect_flip.data(), got.size(), 1e-6f, nullptr);
      if (bad2 == 0)
        result = "ok (rows flipped)";
      else {
        char b[160];
        snprintf(b, sizeof b, "%zu/%zu floats differ, first at %zu: got %g want %g", bad, got.size(),
                 first, got[first], expect[first]);
        result = b;
      }
    }
  } else {
    std::vector<uint8_t> got;
    read_fbo_rgba8(w, h, got);
    std::vector<uint8_t> e8(got.size()), e8f(got.size());
    for (size_t i = 0; i < got.size(); i++) {
      e8[i] = (uint8_t)(expect[i] * 255.0f + 0.5f);
      e8f[i] = (uint8_t)(expect_flip[i] * 255.0f + 0.5f);
    }
    size_t first;
    size_t bad = count_mismatch(got.data(), e8.data(), got.size(), &first);
    if (bad == 0) {
      result = "ok";
    } else if (count_mismatch(got.data(), e8f.data(), got.size()) == 0) {
      result = "ok (rows flipped)";
    } else {
      char b[160];
      snprintf(b, sizeof b, "%zu/%zu bytes differ, first at %zu: got %u want %u", bad, got.size(), first,
               got[first], e8[first]);
      result = b;
    }
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  fbo.destroy();
  return result;
}

// Render the gradient into the import (pbuffer: as the current draw surface;
// EGLImage: as an FBO attachment) and compare the D3D11 texture bytes.
static std::string check_dest(GlSession& s, Programs& pr, Quad& quad, Import& im, ID3D11Texture2D* tex,
                              DXGI_FORMAT fmt, uint32_t w, uint32_t h, bool use_renderbuffer,
                              std::string* orientation) {
  bool flt = is_float_fmt(fmt);
  GLuint fbo = 0, rbo = 0;
  if (im.route == Route::Pbuffer) {
    if (!s.make_current(im.pbuffer, s.ctx)) {
      s.restore_current();
      return std::string("eglMakeCurrent(pbuffer) failed");
    }
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
  } else {
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    if (use_renderbuffer) {
      if (!glEGLImageTargetRenderbufferStorageOES) return "glEGLImageTargetRenderbufferStorageOES missing";
      glGenRenderbuffers(1, &rbo);
      glBindRenderbuffer(GL_RENDERBUFFER, rbo);
      glEGLImageTargetRenderbufferStorageOES(GL_RENDERBUFFER, (GLeglImageOES)im.image);
      GLenum e = glGetError();
      if (e != GL_NO_ERROR) {
        glDeleteRenderbuffers(1, &rbo);
        glDeleteFramebuffers(1, &fbo);
        char b[96];
        snprintf(b, sizeof b, "renderbuffer storage: gl error 0x%04x", e);
        return b;
      }
      glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, rbo);
    } else {
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, im.tex, 0);
    }
    GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    if (st != GL_FRAMEBUFFER_COMPLETE) {
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      if (rbo) glDeleteRenderbuffers(1, &rbo);
      glDeleteFramebuffers(1, &fbo);
      return std::string("FBO ") + gl_fbo_status_str(st);
    }
  }
  glViewport(0, 0, (GLsizei)w, (GLsizei)h);
  glUseProgram(pr.gradient);
  glUniform1f(pr.gradient_denom, flt ? 256.0f : 255.0f);
  quad.draw();
  glFlush();
  GLenum ge = glGetError();
  if (im.route == Route::Pbuffer) {
    // The texture contents are defined only after the pbuffer stops being
    // the current draw surface.
    s.restore_current();
  } else {
    glFinish();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    if (rbo) glDeleteRenderbuffers(1, &rbo);
    glDeleteFramebuffers(1, &fbo);
  }
  if (ge != GL_NO_ERROR) {
    char b[64];
    snprintf(b, sizeof b, "gl error 0x%04x after draw", ge);
    return b;
  }
  std::vector<uint8_t> back;
  if (!readback_tex(s.d3d, tex, back)) return "D3D11 readback failed";
  auto want = make_gradient(fmt, w, h, false);
  auto want_flip = make_gradient(fmt, w, h, true);
  size_t first;
  size_t bad = count_mismatch(back.data(), want.data(), want.size(), &first);
  if (bad == 0) {
    if (orientation) *orientation = "row 0 = gl y 0";
    return "ok";
  }
  if (count_mismatch(back.data(), want_flip.data(), want.size()) == 0) {
    if (orientation) *orientation = "row 0 = gl y max (flipped)";
    return "ok (rows flipped)";
  }
  char b[160];
  snprintf(b, sizeof b, "%zu/%zu bytes differ, first at %zu: got %u want %u", bad, want.size(), first,
           back[first], want[first]);
  return b;
}

static bool starts_ok(const std::string& r) { return r.rfind("ok", 0) == 0; }

// ---------------------------------------------------------------------------
// S1
// ---------------------------------------------------------------------------

void run_s1(GlSession& s, const char* prefix) {
  char id[16];
  auto ID = [&](const char* sub) {
    snprintf(id, sizeof id, "%s.%s", prefix, sub);
    return id;
  };
  Programs pr;
  if (!pr.init()) return;
  Quad quad;
  quad.init();
  const uint32_t W = 256, H = 128;
  static const DXGI_FORMAT formats[] = {
      DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM,     DXGI_FORMAT_R8_UNORM,
      DXGI_FORMAT_R8G8_UNORM,     DXGI_FORMAT_R16_UNORM,          DXGI_FORMAT_R16G16_UNORM,
      DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R32G32B32A32_FLOAT};

  for (Route route : {Route::Pbuffer, Route::EglImage}) {
    for (DXGI_FORMAT fmt : formats) {
      HRESULT hr;
      ComPtr<ID3D11Texture2D> tex =
          create_tex(s.d3d, W, H, fmt, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0, &hr);
      if (!tex) {
        report(ID("1"), Verdict::Fail, "%-8s %-20s CreateTexture2D: %s", route_name(route), fmt_name(fmt),
               hr_str(hr));
        continue;
      }
      auto pat = make_pattern(fmt, W, H, 7);
      upload_tex(s.d3d, tex.Get(), pat);
      Import im = import_texture(s, route, tex.Get(), gl_internal_format_for(fmt));
      if (!im.ok) {
        report(ID("1"), Verdict::Fail, "%-8s %-20s import: egl %s gl 0x%04x", route_name(route),
               fmt_name(fmt), egl_err_str(im.egl_error), im.gl_error);
        import_destroy(s, im);
        continue;
      }
      std::string src = check_source(s, pr, quad, im, fmt, W, H, 7);
      report(ID("1"), starts_ok(src) ? Verdict::Pass : Verdict::Fail, "%-8s %-20s as sampled source: %s",
             route_name(route), fmt_name(fmt), src.c_str());
      std::string orient;
      std::string dst = check_dest(s, pr, quad, im, tex.Get(), fmt, W, H, false, &orient);
      report(ID("2"), starts_ok(dst) ? Verdict::Pass : Verdict::Fail, "%-8s %-20s as render target: %s%s%s",
             route_name(route), fmt_name(fmt), dst.c_str(), orient.empty() ? "" : "; ", orient.c_str());
      if (route == Route::EglImage) {
        std::string rb = check_dest(s, pr, quad, im, tex.Get(), fmt, W, H, true, nullptr);
        report(ID("3"), starts_ok(rb) ? Verdict::Pass : Verdict::Fail,
               "%-8s %-20s as renderbuffer target: %s", route_name(route), fmt_name(fmt), rb.c_str());
      }
      // After the destination test the texture holds the gradient; sample it
      // once more to confirm the same import object sees the new contents.
      if (route == Route::EglImage && fmt == DXGI_FORMAT_R8G8B8A8_UNORM) {
        upload_tex(s.d3d, tex.Get(), pat);
        std::string again = check_source(s, pr, quad, im, fmt, W, H, 7);
        report(ID("4"), starts_ok(again) ? Verdict::Pass : Verdict::Fail,
               "%-8s %-20s re-upload after render, sampled again: %s", route_name(route), fmt_name(fmt),
               again.c_str());
      }
      import_destroy(s, im);
      gl_clear_errors("S1 format loop");
    }
  }

  // Import without EGL_TEXTURE_INTERNAL_FORMAT_ANGLE.
  for (Route route : {Route::Pbuffer, Route::EglImage}) {
    ComPtr<ID3D11Texture2D> tex = create_tex(s.d3d, W, H, DXGI_FORMAT_R8G8B8A8_UNORM,
                                             D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0);
    auto pat = make_pattern(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 3);
    upload_tex(s.d3d, tex.Get(), pat);
    Import im = import_texture(s, route, tex.Get(), 0);
    std::string r = im.ok ? check_source(s, pr, quad, im, DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 3)
                          : std::string("import: ") + egl_err_str(im.egl_error);
    report(ID("5"), starts_ok(r) ? Verdict::Pass : Verdict::Fail,
           "%-8s RGBA8 import without EGL_TEXTURE_INTERNAL_FORMAT_ANGLE: %s", route_name(route), r.c_str());
    import_destroy(s, im);
  }

  // Pbuffer bound through a config without EGL_BIND_TO_TEXTURE_RGBA.
  if (s.cfg_nobind && s.cfg_nobind != s.cfg) {
    ComPtr<ID3D11Texture2D> tex = create_tex(s.d3d, W, H, DXGI_FORMAT_R8G8B8A8_UNORM,
                                             D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0);
    const EGLint attribs[] = {EGL_TEXTURE_FORMAT, EGL_TEXTURE_RGBA, EGL_TEXTURE_TARGET, EGL_TEXTURE_2D,
                              EGL_NONE};
    EGLSurface pb = eglCreatePbufferFromClientBuffer(s.dpy, EGL_D3D_TEXTURE_ANGLE, (EGLClientBuffer)tex.Get(),
                                                     s.cfg_nobind, attribs);
    if (pb == EGL_NO_SURFACE) {
      report(ID("6"), Verdict::Info, "pbuffer on a config without BIND_TO_TEXTURE_RGBA: create fails %s",
             egl_err_str(eglGetError()));
    } else {
      GLuint t;
      glGenTextures(1, &t);
      glBindTexture(GL_TEXTURE_2D, t);
      EGLBoolean ok = eglBindTexImage(s.dpy, pb, EGL_BACK_BUFFER);
      report(ID("6"), Verdict::Info, "pbuffer on a config without BIND_TO_TEXTURE_RGBA: create ok, bind %s",
             ok ? "ok" : egl_err_str(eglGetError()));
      if (ok) eglReleaseTexImage(s.dpy, pb, EGL_BACK_BUFFER);
      glDeleteTextures(1, &t);
      eglDestroySurface(s.dpy, pb);
    }
  } else {
    report(ID("6"), Verdict::Info, "eglChooseConfig returned the same config with and without BIND_TO_TEXTURE_RGBA");
  }

  // Bind / misc flag matrix (RGBA8).
  struct FlagCase {
    UINT bind;
    UINT misc;
    const char* name;
  };
  static const FlagCase cases[] = {
      {D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0, "RT|SR misc=0"},
      {D3D11_BIND_SHADER_RESOURCE, 0, "SR    misc=0"},
      {D3D11_BIND_RENDER_TARGET, 0, "RT    misc=0"},
      {D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, D3D11_RESOURCE_MISC_SHARED, "RT|SR SHARED"},
      {D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
       D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE, "RT|SR SHARED|NTHANDLE"},
      {D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
       D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX | D3D11_RESOURCE_MISC_SHARED_NTHANDLE,
       "RT|SR KEYEDMUTEX|NTHANDLE"},
      {D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX,
       "RT|SR KEYEDMUTEX"},
  };
  for (const FlagCase& fc : cases) {
    HRESULT hr;
    ComPtr<ID3D11Texture2D> tex = create_tex(s.d3d, W, H, DXGI_FORMAT_R8G8B8A8_UNORM, fc.bind, fc.misc, &hr);
    if (!tex) {
      report(ID("7"), Verdict::Info, "%-26s CreateTexture2D: %s", fc.name, hr_str(hr));
      continue;
    }
    auto pat = make_pattern(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 11);
    std::string line;
    for (Route route : {Route::Pbuffer, Route::EglImage}) {
      // Keyed-mutex textures: the owner holds key 0 across the GL work too.
      KeyedLock lock(tex.Get());
      // The previous route's destination test overwrote the texture.
      upload_tex(s.d3d, tex.Get(), pat);
      Import im = import_texture(s, route, tex.Get(), GL_RGBA);
      line += route_name(route);
      line += "=";
      if (!im.ok) {
        line += std::string("create:") + egl_err_str(im.egl_error);
        if (im.gl_error) {
          char b[24];
          snprintf(b, sizeof b, "/gl0x%04x", im.gl_error);
          line += b;
        }
        // An EGLImage over a texture without SHADER_RESOURCE cannot back a
        // GL texture but may still back a renderbuffer.
        if (route == Route::EglImage && im.image != EGL_NO_IMAGE_KHR && (fc.bind & D3D11_BIND_RENDER_TARGET)) {
          std::string rb = check_dest(s, pr, quad, im, tex.Get(), DXGI_FORMAT_R8G8B8A8_UNORM, W, H, true, nullptr);
          line += " rb-dst[" + rb + "]";
        }
      } else {
        std::string src = (fc.bind & D3D11_BIND_SHADER_RESOURCE)
                              ? check_source(s, pr, quad, im, DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 11)
                              : "n/a";
        std::string dst = (fc.bind & D3D11_BIND_RENDER_TARGET)
                              ? check_dest(s, pr, quad, im, tex.Get(), DXGI_FORMAT_R8G8B8A8_UNORM, W, H,
                                           false, nullptr)
                              : "n/a";
        line += "src[" + src + "] dst[" + dst + "]";
      }
      line += "  ";
      import_destroy(s, im);
      gl_clear_errors("S1 flag matrix");
    }
    report(ID("7"), Verdict::Info, "%-26s %s", fc.name, line.c_str());
  }

  // Timings: import create/destroy, per-pass bind/release, 1080p convert.
  for (Route route : {Route::Pbuffer, Route::EglImage}) {
    ComPtr<ID3D11Texture2D> tex = create_tex(s.d3d, 1920, 1080, DXGI_FORMAT_R8G8B8A8_UNORM,
                                             D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0);
    char what[128];
    Stats st = time_loop(g_opt.iters, [&] {
      Import im = import_texture(s, route, tex.Get(), GL_RGBA);
      import_destroy(s, im);
    });
    snprintf(what, sizeof what, "%s import create+destroy, 1080p RGBA8", route_name(route));
    report_stats(ID("T1"), what, st);

    Import im = import_texture(s, route, tex.Get(), GL_RGBA);
    st = time_loop(g_opt.iters, [&] {
      import_bind(s, im);
      import_release(s, im);
    });
    snprintf(what, sizeof what, "%s per-pass bind+release (source)", route_name(route));
    report_stats(ID("T2"), what, st);
    import_destroy(s, im);
  }
  for (Route route : {Route::Pbuffer, Route::EglImage}) {
    const uint32_t CW = 1920, CH = 1080;
    ComPtr<ID3D11Texture2D> src = create_tex(s.d3d, CW, CH, DXGI_FORMAT_R8G8B8A8_UNORM,
                                             D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0);
    ComPtr<ID3D11Texture2D> dst = create_tex(s.d3d, CW, CH, DXGI_FORMAT_R8G8B8A8_UNORM,
                                             D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0);
    auto pat = make_pattern(DXGI_FORMAT_R8G8B8A8_UNORM, CW, CH, 5);
    upload_tex(s.d3d, src.Get(), pat);
    Import si = import_texture(s, route, src.Get(), GL_RGBA);
    Import di = import_texture(s, route, dst.Get(), GL_RGBA);
    if (!si.ok || !di.ok) {
      report(ID("T3"), Verdict::Skip, "%s 1080p convert: import failed", route_name(route));
      import_destroy(s, si);
      import_destroy(s, di);
      continue;
    }
    GLuint fbo = 0;
    if (route == Route::EglImage) {
      glGenFramebuffers(1, &fbo);
      glBindFramebuffer(GL_FRAMEBUFFER, fbo);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, di.tex, 0);
    }
    auto convert = [&] {
      if (route == Route::Pbuffer)
        s.make_current(di.pbuffer, s.ctx);
      else
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
      glViewport(0, 0, CW, CH);
      glUseProgram(pr.sample);
      glActiveTexture(GL_TEXTURE0);
      import_bind(s, si);
      glUniform1i(pr.sample_tex, 0);
      quad.draw();
      import_release(s, si);
      if (route == Route::Pbuffer) s.restore_current();
      glFinish();
    };
    convert();
    std::vector<uint8_t> back;
    readback_tex(s.d3d, dst.Get(), back);
    size_t bad = count_mismatch(back.data(), pat.data(), pat.size());
    std::vector<uint8_t> pat_flip((size_t)CW * CH * 4);
    for (uint32_t y = 0; y < CH; y++) memcpy(&pat_flip[(size_t)y * CW * 4], &pat[(size_t)(CH - 1 - y) * CW * 4], CW * 4);
    size_t bad_flip = count_mismatch(back.data(), pat_flip.data(), pat.size());
    report(ID("T3"), (bad == 0 || bad_flip == 0) ? Verdict::Pass : Verdict::Fail,
           "%s 1080p RGBA8 texture->texture convert bytes: %zu mismatches%s", route_name(route),
           bad == 0 ? bad : bad_flip, bad != 0 && bad_flip == 0 ? " (rows flipped)" : "");
    Stats st = time_loop(g_opt.iters, convert);
    char what[128];
    snprintf(what, sizeof what, "%s 1080p RGBA8 texture->texture convert + glFinish", route_name(route));
    report_stats(ID("T3"), what, st);
    if (fbo) {
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glDeleteFramebuffers(1, &fbo);
    }
    import_destroy(s, si);
    import_destroy(s, di);
  }
  quad.destroy();
  pr.destroy();
  gl_clear_errors("S1 end");
}
