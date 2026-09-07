// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// S6: device ownership. ANGLE on a device the probe created (hardware and
// WARP), behaviour diff against the ANGLE-owned display, device lifetime
// across display teardown, and the third option: a separate allocation
// device whose textures reach ANGLE's device through shared handles.
#include "probe.h"

#include <cstring>
#include <set>
#include <sstream>

static std::set<std::string> tokens(const std::string& s) {
  std::set<std::string> out;
  std::istringstream is(s);
  std::string t;
  while (is >> t) out.insert(t);
  return out;
}

static void diff_exts(const char* id, const char* what, const std::string& a, const std::string& b) {
  auto A = tokens(a), B = tokens(b);
  std::string only_a, only_b;
  for (const auto& t : A)
    if (!B.count(t)) only_a += t + " ";
  for (const auto& t : B)
    if (!A.count(t)) only_b += t + " ";
  if (only_a.empty() && only_b.empty())
    report(id, Verdict::Pass, "%s: identical", what);
  else
    report(id, Verdict::Info, "%s: only on ANGLE-owned [%s] only on injected [%s]", what, only_a.c_str(), only_b.c_str());
}

bool quick_import_check(GlSession& s, const char* id, const char* label) {
  bool all = true;
  const uint32_t W = 256, H = 128;
  std::string log;
  GLuint p_grad = compile_program(kVertexShader, kGradientFragment, &log);
  if (!p_grad) {
    report(id, Verdict::Fail, "%s: gradient program: %s", label, log.c_str());
    return false;
  }
  Quad quad;
  quad.init();
  auto want = make_gradient(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, false);
  for (Route route : {Route::Pbuffer, Route::EglImage}) {
    ComPtr<ID3D11Texture2D> tex = create_tex(s.d3d, W, H, DXGI_FORMAT_R8G8B8A8_UNORM,
                                             D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0);
    Import im = import_texture(s, route, tex.Get(), GL_RGBA);
    if (!im.ok) {
      report(id, Verdict::Fail, "%s: %s import: %s / gl 0x%04x", label, route_name(route), egl_err_str(im.egl_error), im.gl_error);
      all = false;
      import_destroy(s, im);
      continue;
    }
    GLuint fbo = 0;
    if (route == Route::Pbuffer) {
      s.make_current(im.pbuffer, s.ctx);
    } else {
      glGenFramebuffers(1, &fbo);
      glBindFramebuffer(GL_FRAMEBUFFER, fbo);
      glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, im.tex, 0);
    }
    glViewport(0, 0, W, H);
    glUseProgram(p_grad);
    glUniform1f(glGetUniformLocation(p_grad, "denom"), 255.0f);
    quad.draw();
    glFlush();
    if (route == Route::Pbuffer) {
      s.restore_current();
    } else {
      glFinish();
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      glDeleteFramebuffers(1, &fbo);
    }
    std::vector<uint8_t> back;
    readback_tex(s.d3d, tex.Get(), back);
    size_t bad = count_mismatch(back.data(), want.data(), want.size());
    report(id, bad == 0 ? Verdict::Pass : Verdict::Fail, "%s: %s gradient render into RGBA8 texture: %zu mismatches", label,
           route_name(route), bad);
    all &= bad == 0;
    import_destroy(s, im);
  }
  quad.destroy();
  glDeleteProgram(p_grad);
  gl_clear_errors("quick_import_check");
  return all;
}

static bool same_object(IUnknown* a, IUnknown* b) {
  ComPtr<IUnknown> ua, ub;
  if (!a || !b) return false;
  a->QueryInterface(IID_PPV_ARGS(&ua));
  b->QueryInterface(IID_PPV_ARGS(&ub));
  return ua.Get() == ub.Get();
}

void run_s6(GlSession& s) {
  bool warp = s.mode == DisplayMode::AngleWarp;
  // ---- S6.1 injected device, same adapter kind as the main session --------
  GlSession inj;
  if (!inj.bring_up(warp ? DisplayMode::InjectWarp : DisplayMode::InjectHardware)) {
    report("S6.1", Verdict::Fail, "bring-up on an injected device failed");
    s.restore_current();
    return;
  }
  print_session(inj);
  report("S6.1", Verdict::Pass, "ANGLE display on an injected device (%s)", mode_name(inj.mode));
  report("S6.1", same_object(inj.d3d.dev.Get(), inj.injected.dev.Get()) ? Verdict::Pass : Verdict::Fail,
         "EGL_D3D11_DEVICE_ANGLE returns the injected device object: %s",
         same_object(inj.d3d.dev.Get(), inj.injected.dev.Get()) ? "yes" : "no");
  report("S6.1", (inj.d3d.creation_flags & D3D11_CREATE_DEVICE_VIDEO_SUPPORT) ? Verdict::Pass : Verdict::Info,
         "injected device creation flags 0x%x%s%s", inj.d3d.creation_flags,
         inj.last_error.empty() ? "" : "; earlier attempts: ", inj.last_error.c_str());
  report("S6.1", inj.es_minor == s.es_minor ? Verdict::Pass : Verdict::Info, "ES version: ANGLE-owned 3.%d, injected 3.%d",
         s.es_minor, inj.es_minor);
  diff_exts("S6.1", "EGL display extensions", s.display_exts, inj.display_exts);
  diff_exts("S6.1", "GL extensions", s.gl_exts, inj.gl_exts);
  report("S6.1", inj.gl_renderer == s.gl_renderer ? Verdict::Pass : Verdict::Info, "GL_RENDERER injected: %s", inj.gl_renderer.c_str());

  // ---- S6.2 the import and sync suites on the injected display ----------
  quick_import_check(inj, "S6.2", "injected");
  run_s1(inj, "S6.S1");
  run_s4(inj, "S6.S4");

  // ---- S6.3 display lifetime vs device lifetime ---------------------------
  {
    EGLDisplay again = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, inj.egl_device, nullptr);
    report("S6.3", Verdict::Info, "second eglGetPlatformDisplayEXT on the same EGLDevice: %s",
           again == inj.dpy ? "same EGLDisplay" : (again == EGL_NO_DISPLAY ? "EGL_NO_DISPLAY" : "different EGLDisplay"));
  }
  D3D keep = inj.injected;
  inj.shutdown();
  {
    ComPtr<ID3D11Texture2D> tex = create_tex(keep, 64, 64, DXGI_FORMAT_R8G8B8A8_UNORM, D3D11_BIND_SHADER_RESOURCE, 0);
    auto pat = make_pattern(DXGI_FORMAT_R8G8B8A8_UNORM, 64, 64, 4);
    std::vector<uint8_t> back;
    bool ok = tex && upload_tex(keep, tex.Get(), pat) && readback_tex(keep, tex.Get(), back) &&
              count_mismatch(back.data(), pat.data(), pat.size()) == 0;
    HRESULT removed = keep.dev->GetDeviceRemovedReason();
    report("S6.3", ok && removed == S_OK ? Verdict::Pass : Verdict::Fail,
           "device usable after eglTerminate + eglReleaseDeviceANGLE: %s (removed reason %s)", ok ? "yes" : "no", hr_str(removed));
  }
  {
    GlSession inj2;
    bool ok = inj2.bring_up(warp ? DisplayMode::InjectWarp : DisplayMode::InjectHardware, keep.dev.Get());
    report("S6.3", ok ? Verdict::Pass : Verdict::Fail, "second display lifetime on the same device: %s", ok ? "up" : "failed");
    if (ok) quick_import_check(inj2, "S6.3", "second lifetime");
    inj2.shutdown();
  }

  // ---- S6.4 injected WARP (the CI shape) ----------------------------------
  if (!warp) {
    GlSession w;
    if (w.bring_up(DisplayMode::InjectWarp)) {
      report("S6.4", Verdict::Pass, "injected WARP device: %s, FL 0x%x, flags 0x%x%s%s", w.gl_renderer.c_str(), w.d3d.feature_level,
             w.d3d.creation_flags, w.last_error.empty() ? "" : "; earlier attempts: ", w.last_error.c_str());
      quick_import_check(w, "S6.4", "injected WARP");
      w.shutdown();
    } else {
      report("S6.4", Verdict::Fail, "injected WARP bring-up failed: %s", w.last_error.c_str());
    }
  }
  s.restore_current();

  // ---- S6.5 third option: separate allocation device + shared handles -----
  {
    AdapterInfo ad;
    D3D alloc;
    HRESULT hr;
    bool ok = (warp || pick_adapter(g_opt.adapter, false, &ad)) && create_device(warp ? nullptr : &ad, warp, 0, &alloc, &hr);
    if (!ok) {
      report("S6.5", Verdict::Fail, "allocation device: %s", hr_str(hr));
    } else {
      const uint32_t W = 1920, H = 1080;
      ComPtr<ID3D11Texture2D> src = create_tex(alloc, W, H, DXGI_FORMAT_R8G8B8A8_UNORM,
                                               D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
                                               D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE);
      auto want = make_gradient(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, false);
      HANDLE h = create_shared_handle(src.Get(), nullptr, &hr);
      ComPtr<ID3D11Texture2D> opened;
      HRESULT hopen = h ? s.d3d.dev1->OpenSharedResource1(h, IID_PPV_ARGS(&opened)) : E_FAIL;
      report("S6.5", SUCCEEDED(hopen) ? Verdict::Pass : Verdict::Fail,
             "texture allocated on a separate device opened on ANGLE's device: %s", hr_str(hopen));
      if (SUCCEEDED(hopen)) {
        Stats st = time_loop(g_opt.iters, [&] {
          ComPtr<ID3D11Texture2D> t = create_tex(alloc, 256, 128, DXGI_FORMAT_R8G8B8A8_UNORM,
                                                 D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
                                                 D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE);
          HRESULT hh;
          HANDLE hd = create_shared_handle(t.Get(), nullptr, &hh);
          ComPtr<ID3D11Texture2D> o;
          s.d3d.dev1->OpenSharedResource1(hd, IID_PPV_ARGS(&o));
          CloseHandle(hd);
        });
        report_stats("S6.T", "create shared 256x128 + CreateSharedHandle + OpenSharedResource1 + release", st);
        // Render into the opened texture from ANGLE, fence, read it back on
        // the allocation device.
        std::string log;
        GLuint p_grad = compile_program(kVertexShader, kGradientFragment, &log);
        Quad quad;
        quad.init();
        for (Route route : {Route::Pbuffer, Route::EglImage}) {
          Import im = import_texture(s, route, opened.Get(), GL_RGBA);
          if (!im.ok) {
            report("S6.5", Verdict::Fail, "%s import of the opened texture: %s", route_name(route), egl_err_str(im.egl_error));
            import_destroy(s, im);
            continue;
          }
          ComPtr<ID3D11Fence> fence;
          s.d3d.dev5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence));
          HANDLE fh = nullptr;
          fence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &fh);
          ComPtr<ID3D11Fence> fence2;
          alloc.dev5->OpenSharedFence(fh, IID_PPV_ARGS(&fence2));
          CloseHandle(fh);
          GLuint fbo = 0;
          if (route == Route::EglImage) {
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, im.tex, 0);
          }
          UINT64 v = 0;
          auto convert = [&] {
            if (route == Route::Pbuffer) s.make_current(im.pbuffer, s.ctx);
            else glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glViewport(0, 0, W, H);
            glUseProgram(p_grad);
            glUniform1f(glGetUniformLocation(p_grad, "denom"), 255.0f);
            quad.draw();
            glFlush();
            if (route == Route::Pbuffer) s.restore_current();
            s.d3d.ctx4->Signal(fence.Get(), ++v);
            alloc.ctx4->Wait(fence2.Get(), v);
          };
          convert();
          std::vector<uint8_t> back;
          readback_tex(alloc, src.Get(), back);
          size_t bad = count_mismatch(back.data(), want.data(), want.size());
          report("S6.5", bad == 0 ? Verdict::Pass : Verdict::Fail,
                 "%s render on ANGLE's device, fence, readback on the allocation device: %zu mismatches", route_name(route), bad);
          Stats st = time_loop(g_opt.iters, [&] {
            convert();
            alloc.ctx->Flush();
            ComPtr<ID3D11Query> q;
            D3D11_QUERY_DESC qd{D3D11_QUERY_EVENT, 0};
            alloc.dev->CreateQuery(&qd, &q);
            alloc.ctx->End(q.Get());
            while (alloc.ctx->GetData(q.Get(), nullptr, 0, 0) == S_FALSE) YieldProcessor();
          });
          char what[128];
          snprintf(what, sizeof what, "%s 1080p gradient into opened-shared texture + cross-device fence wait", route_name(route));
          report_stats("S6.T", what, st);
          if (fbo) {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glDeleteFramebuffers(1, &fbo);
          }
          import_destroy(s, im);
        }
        // Reference: same render into a texture native to ANGLE's device.
        {
          ComPtr<ID3D11Texture2D> native = create_tex(s.d3d, W, H, DXGI_FORMAT_R8G8B8A8_UNORM,
                                                      D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0);
          Import im = import_texture(s, Route::EglImage, native.Get(), GL_RGBA);
          GLuint fbo;
          glGenFramebuffers(1, &fbo);
          glBindFramebuffer(GL_FRAMEBUFFER, fbo);
          glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, im.tex, 0);
          Stats st = time_loop(g_opt.iters, [&] {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
            glViewport(0, 0, W, H);
            glUseProgram(p_grad);
            glUniform1f(glGetUniformLocation(p_grad, "denom"), 255.0f);
            quad.draw();
            glFinish();
          });
          report_stats("S6.T", "eglimage 1080p gradient into a native texture + glFinish (reference)", st);
          glBindFramebuffer(GL_FRAMEBUFFER, 0);
          glDeleteFramebuffers(1, &fbo);
          import_destroy(s, im);
        }
        quad.destroy();
        glDeleteProgram(p_grad);
      }
      if (h) CloseHandle(h);
    }
  }
  s.restore_current();
  gl_clear_errors("S6 end");
}
