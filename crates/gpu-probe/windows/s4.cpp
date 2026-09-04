// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// S4: synchronization. What a same-device consumer needs after GL work,
// the D3D11 fence as the completion primitive (event wait, shared handle,
// cross-device wait), and the immediate context driven from a non-GL
// thread while two GL contexts alternate the way HAL processors do.
#include "probe.h"

#include <atomic>
#include <cstring>
#include <mutex>
#include <thread>

static const uint32_t W = 512, H = 512;

// Draw either the gradient or a solid clear so consecutive frames differ.
struct Frame {
  GLuint fbo = 0;
  GLuint p_grad = 0;
  GLint denom = -1;
  Quad quad;
  uint32_t w = W, h = H;
  bool init(GLuint tex, uint32_t width = W, uint32_t height = H) {
    w = width;
    h = height;
    std::string log;
    p_grad = compile_program(kVertexShader, kGradientFragment, &log);
    if (!p_grad) return false;
    denom = glGetUniformLocation(p_grad, "denom");
    quad.init();
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
    return glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
  }
  void draw(bool gradient) {
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glViewport(0, 0, (GLsizei)w, (GLsizei)h);
    if (gradient) {
      glUseProgram(p_grad);
      glUniform1f(denom, 255.0f);
      quad.draw();
    } else {
      glClearColor(0.2f, 0.4f, 0.6f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);
    }
  }
  void destroy() {
    if (fbo) glDeleteFramebuffers(1, &fbo);
    if (p_grad) glDeleteProgram(p_grad);
    quad.destroy();
  }
};

static std::vector<uint8_t> solid_reference() {
  std::vector<uint8_t> v((size_t)W * H * 4);
  for (size_t i = 0; i < (size_t)W * H; i++) {
    v[i * 4] = 51;
    v[i * 4 + 1] = 102;
    v[i * 4 + 2] = 153;
    v[i * 4 + 3] = 255;
  }
  return v;
}

// 0 = matches gradient, 1 = matches solid, -1 = neither.
static int classify(const std::vector<uint8_t>& px, const std::vector<uint8_t>& grad,
                    const std::vector<uint8_t>& solid) {
  if (count_mismatch(px.data(), grad.data(), grad.size()) == 0) return 0;
  if (count_mismatch(px.data(), solid.data(), solid.size()) == 0) return 1;
  return -1;
}

void s4_variant(GlSession& s, const char* id, int variant);

void run_s4(GlSession& s, const char* prefix) {
  char id[16];
  auto ID = [&](const char* sub) {
    snprintf(id, sizeof id, "%s.%s", prefix, sub);
    return id;
  };
  auto grad = make_gradient(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, false);
  auto solid = solid_reference();

  ComPtr<ID3D11Texture2D> tex = create_tex(s.d3d, W, H, DXGI_FORMAT_R8G8B8A8_UNORM,
                                           D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
                                           D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE);
  Import im = import_texture(s, Route::EglImage, tex.Get(), GL_RGBA);
  Frame frame;
  if (!im.ok || !frame.init(im.tex)) {
    report(ID("0"), Verdict::Fail, "setup failed");
    return;
  }
  ComPtr<ID3D11Texture2D> staging = create_staging_like(s.d3d, tex.Get(), D3D11_CPU_ACCESS_READ);

  // ---- S4.1 same-device consumer after GL work, with and without flush ----
  for (int mode = 0; mode < 3; mode++) {
    int stale = 0, garbage = 0;
    const int N = 60;
    for (int i = 0; i < N; i++) {
      bool gradient = (i & 1) == 0;
      frame.draw(gradient);
      if (mode == 1) glFlush();
      if (mode == 2) glFinish();
      std::vector<uint8_t> px;
      readback_tex(s.d3d, tex.Get(), px, nullptr, staging.Get());
      int c = classify(px, grad, solid);
      if (c < 0) garbage++;
      else if (c != (gradient ? 0 : 1)) stale++;
    }
    report(ID("1"), (stale == 0 && garbage == 0) ? Verdict::Pass : Verdict::Fail,
           "same immediate context: draw + %-8s then CopyResource/Map: %d stale, %d garbage of %d",
           mode == 0 ? "nothing" : (mode == 1 ? "glFlush" : "glFinish"), stale, garbage, N);
  }

  // ---- S4.2 D3D11 fence on ANGLE's device ---------------------------------
  ComPtr<ID3D11Fence> fence;
  HRESULT hr = s.d3d.dev5 ? s.d3d.dev5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence)) : E_NOINTERFACE;
  UINT64 fence_value = 0;
  HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  if (!fence) {
    report(ID("2"), Verdict::Fail, "ID3D11Device5::CreateFence(SHARED): %s", hr_str(hr));
  } else {
    report(ID("2"), Verdict::Pass, "ID3D11Device5::CreateFence(SHARED) on ANGLE's device");
    int stale = 0;
    const int N = 60;
    for (int i = 0; i < N; i++) {
      bool gradient = (i & 1) == 0;
      frame.draw(gradient);
      glFlush();
      s.d3d.ctx4->Signal(fence.Get(), ++fence_value);
      fence->SetEventOnCompletion(fence_value, event);
      DWORD w = WaitForSingleObject(event, 5000);
      if (w != WAIT_OBJECT_0) {
        stale += 1000;
        break;
      }
      std::vector<uint8_t> px;
      readback_tex(s.d3d, tex.Get(), px, nullptr, staging.Get());
      if (classify(px, grad, solid) != (gradient ? 0 : 1)) stale++;
    }
    report(ID("2"), stale == 0 ? Verdict::Pass : Verdict::Fail,
           "draw + glFlush + Signal(fence) + SetEventOnCompletion wait: %d stale of %d", stale, N);
    Stats st = time_loop(g_opt.iters, [&] {
      s.d3d.ctx4->Signal(fence.Get(), ++fence_value);
      fence->SetEventOnCompletion(fence_value, event);
      WaitForSingleObject(event, 5000);
    });
    report_stats(ID("T1"), "fence Signal + event wait, GPU idle", st);
    st = time_loop(g_opt.iters, [&] {
      frame.draw(true);
      glFlush();
      s.d3d.ctx4->Signal(fence.Get(), ++fence_value);
      fence->SetEventOnCompletion(fence_value, event);
      WaitForSingleObject(event, 5000);
    });
    report_stats(ID("T1"), "512x512 draw + glFlush + fence Signal + event wait", st);
    st = time_loop(g_opt.iters, [&] {
      frame.draw(true);
      glFinish();
    });
    report_stats(ID("T1"), "512x512 draw + glFinish (for comparison)", st);
    st = time_loop(g_opt.iters, [&] {
      frame.draw(true);
      glFlush();
      s.d3d.ctx4->Signal(fence.Get(), ++fence_value);
      while (fence->GetCompletedValue() < fence_value) YieldProcessor();
    });
    report_stats(ID("T1"), "512x512 draw + glFlush + fence Signal + GetCompletedValue spin", st);
  }

  // ---- S4.3 shared fence + shared texture on a second D3D11 device --------
  if (fence) {
    HANDLE fence_h = nullptr;
    hr = fence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &fence_h);
    HRESULT hr_tex;
    HANDLE tex_h = create_shared_handle(tex.Get(), nullptr, &hr_tex);
    D3D other;
    HRESULT hr_dev;
    AdapterInfo ad;
    bool warp = s.mode == DisplayMode::AngleWarp || s.mode == DisplayMode::InjectWarp;
    bool have_adapter = warp || pick_adapter(g_opt.adapter, false, &ad);
    bool ok_dev = have_adapter && create_device(warp ? nullptr : &ad, warp, 0, &other, &hr_dev);
    if (FAILED(hr) || !tex_h || !ok_dev) {
      report(ID("3"), Verdict::Fail, "shared handles: fence %s, texture %s, second device %s", hr_str(hr),
             hr_str(hr_tex), ok_dev ? "ok" : hr_str(hr_dev));
    } else {
      ComPtr<ID3D11Fence> fence2;
      ComPtr<ID3D11Texture2D> tex2;
      HRESULT h1 = other.dev5->OpenSharedFence(fence_h, IID_PPV_ARGS(&fence2));
      HRESULT h2 = other.dev1->OpenSharedResource1(tex_h, IID_PPV_ARGS(&tex2));
      if (FAILED(h1) || FAILED(h2)) {
        report(ID("3"), Verdict::Fail, "second device: OpenSharedFence %s, OpenSharedResource1 %s", hr_str(h1), hr_str(h2));
      } else {
        report(ID("3"), Verdict::Pass, "second D3D11 device opened the fence and the texture through NT handles");
        ComPtr<ID3D11Texture2D> staging2 = create_staging_like(other, tex2.Get(), D3D11_CPU_ACCESS_READ);
        for (int use_wait = 1; use_wait >= 0; use_wait--) {
          int stale = 0, garbage = 0;
          const int N = 200;
          for (int i = 0; i < N; i++) {
            bool gradient = (i & 1) == 0;
            frame.draw(gradient);
            glFlush();
            s.d3d.ctx4->Signal(fence.Get(), ++fence_value);
            if (use_wait) other.ctx4->Wait(fence2.Get(), fence_value);
            std::vector<uint8_t> px;
            readback_tex(other, tex2.Get(), px, nullptr, staging2.Get());
            int c = classify(px, grad, solid);
            if (c < 0) garbage++;
            else if (c != (gradient ? 0 : 1)) stale++;
          }
          report(ID("3"), Verdict::Info,
                 "second device reads the shared texture %s GPU Wait(fence): %d stale, %d garbage of %d",
                 use_wait ? "after" : "WITHOUT", stale, garbage, N);
        }
      }
    }
    if (fence_h) CloseHandle(fence_h);
    if (tex_h) CloseHandle(tex_h);
  }

  // ---- S4.4 two GL contexts on two threads + a non-GL consumer thread -----
  // Worker A renders a 512x512 gradient, worker B a 256x256 gradient, each
  // with its own context permanently current, serialized by one mutex the
  // way HAL processors are. The consumer copies both textures to staging
  // and checks them from a third thread.
  // Variant 0: HAL model (per-message re-sync, consumer under the mutex).
  // Variant 1: consumer without the mutex, ID3D11Multithread protection on.
  // Variant 2: no re-sync (control; the ANGLE state quirk the HAL hit).
  for (int variant = 0; variant < 2; variant++) s4_variant(s, ID("4"), variant);
  // The control crashed the process on the RTX (ANGLE state manager with
  // no re-sync), so it runs in a child and only its exit code is recorded.
  {
    DWORD code = spawn_self(L"--s4-control", 60000);
    report(ID("4"), Verdict::Info, "control (no per-message re-sync) in a child process: exit code 0x%08lx%s",
           (unsigned long)code, code == 0 ? "" : code == 0xC0000005 ? " (ACCESS_VIOLATION)" : code == 0xFFFFFFFF ? " (timeout)" : "");
  }
  ComPtr<ID3D11Multithread> mt;
  s.d3d.dev->QueryInterface(IID_PPV_ARGS(&mt));
  if (mt) {
    for (int prot = 0; prot < 2; prot++) {
      mt->SetMultithreadProtected(prot ? TRUE : FALSE);
      Stats st = time_loop(g_opt.iters, [&] {
        frame.draw(true);
        glFinish();
      });
      report_stats(ID("T2"), prot ? "512x512 draw + glFinish, multithread protection ON" : "512x512 draw + glFinish, protection OFF", st);
    }
    mt->SetMultithreadProtected(FALSE);
  }

  if (event) CloseHandle(event);
  frame.destroy();
  import_destroy(s, im);
  gl_clear_errors("S4 end");
}

void run_s4_control(GlSession& s) { s4_variant(s, "S4.4", 2); }

void s4_variant(GlSession& s, const char* id, int variant) {
  auto grad = make_gradient(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, false);
  ComPtr<ID3D11Multithread> mt;
  s.d3d.dev->QueryInterface(IID_PPV_ARGS(&mt));
  const uint32_t WB = 256, HB = 256;
  auto gradB = make_gradient(DXGI_FORMAT_R8G8B8A8_UNORM, WB, HB, false);
  {
    if (variant == 1 && !mt) {
      report(id, Verdict::Skip, "ID3D11Multithread unavailable");
      return;
    }
    if (mt) mt->SetMultithreadProtected(variant == 1 ? TRUE : FALSE);
    ComPtr<ID3D11Texture2D> texA = create_tex(s.d3d, W, H, DXGI_FORMAT_R8G8B8A8_UNORM,
                                              D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0);
    ComPtr<ID3D11Texture2D> texB = create_tex(s.d3d, WB, HB, DXGI_FORMAT_R8G8B8A8_UNORM,
                                              D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0);
    ComPtr<ID3D11Texture2D> stA = create_staging_like(s.d3d, texA.Get(), D3D11_CPU_ACCESS_READ);
    ComPtr<ID3D11Texture2D> stB = create_staging_like(s.d3d, texB.Get(), D3D11_CPU_ACCESS_READ);
    std::mutex gl_mu;
    std::atomic<void*> last_ctx{nullptr};
    std::atomic<int> framesA{0}, framesB{0};
    std::atomic<bool> stop{false};
    std::atomic<int> setup_failed{0};
    const int kFramesPerThread = 1500;
    bool resync = variant != 2;
    eglMakeCurrent(s.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    // Context creation and teardown on the ANGLE/D3D11 display are not
    // safe concurrently (the HAL holds its lifecycle lock around processor
    // bring-up for that reason), so both happen under the mutex.
    auto worker = [&](ID3D11Texture2D* t, uint32_t tw, uint32_t th, std::atomic<int>& frames) {
      EGLContext c;
      EGLSurface d;
      Import wim;
      Frame f;
      {
        std::lock_guard<std::mutex> lk(gl_mu);
        c = s.create_context();
        d = s.create_dummy();
        if (c == EGL_NO_CONTEXT || d == EGL_NO_SURFACE || !eglMakeCurrent(s.dpy, d, d, c)) {
          setup_failed++;
          return;
        }
        wim = import_texture(s, Route::EglImage, t, GL_RGBA);
        if (!wim.ok || !f.init(wim.tex, tw, th)) {
          setup_failed++;
          return;
        }
        last_ctx = (void*)c;
      }
      for (int i = 0; i < kFramesPerThread; i++) {
        std::lock_guard<std::mutex> lk(gl_mu);
        if (resync && last_ctx.exchange((void*)c) != (void*)c) {
          eglMakeCurrent(s.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
          eglMakeCurrent(s.dpy, d, d, c);
        }
        f.draw(true);
        glFlush();
        frames++;
      }
      std::lock_guard<std::mutex> lk(gl_mu);
      if (resync && last_ctx.exchange((void*)c) != (void*)c) {
        eglMakeCurrent(s.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        eglMakeCurrent(s.dpy, d, d, c);
      }
      f.destroy();
      import_destroy(s, wim);
      eglMakeCurrent(s.dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
      eglDestroySurface(s.dpy, d);
      eglDestroyContext(s.dpy, c);
    };
    std::atomic<int> badA{0}, badB{0}, reads{0};
    auto consumer = [&] {
      while (!stop) {
        std::vector<uint8_t> px;
        if (framesA > 0) {
          if (variant == 1) {
            readback_tex(s.d3d, texA.Get(), px, nullptr, stA.Get());
          } else {
            std::lock_guard<std::mutex> lk(gl_mu);
            readback_tex(s.d3d, texA.Get(), px, nullptr, stA.Get());
          }
          if (count_mismatch(px.data(), grad.data(), grad.size()) != 0) badA++;
          reads++;
        }
        if (framesB > 0) {
          if (variant == 1) {
            readback_tex(s.d3d, texB.Get(), px, nullptr, stB.Get());
          } else {
            std::lock_guard<std::mutex> lk(gl_mu);
            readback_tex(s.d3d, texB.Get(), px, nullptr, stB.Get());
          }
          if (count_mismatch(px.data(), gradB.data(), gradB.size()) != 0) badB++;
          reads++;
        }
      }
    };
    double t0 = now_us();
    std::thread ta(worker, texA.Get(), W, H, std::ref(framesA));
    std::thread tb(worker, texB.Get(), WB, HB, std::ref(framesB));
    std::thread tc(consumer);
    ta.join();
    tb.join();
    stop = true;
    tc.join();
    double dt = now_us() - t0;
    s.restore_current();
    if (mt) mt->SetMultithreadProtected(FALSE);
    if (setup_failed) {
      report(id, Verdict::Fail, "variant %d: worker setup failed", variant);
      return;
    }
    bool ok = badA == 0 && badB == 0;
    report(id, variant == 2 ? Verdict::Info : (ok ? Verdict::Pass : Verdict::Fail),
           "%s: %d frames, %d consumer reads, wrong A=%d B=%d (%.1f ms, %.1f us/frame)",
           variant == 0 ? "re-sync + consumer under mutex"
           : variant == 1 ? "re-sync + consumer lock-free with ID3D11Multithread"
                          : "control: no per-message re-sync",
           framesA.load() + framesB.load(), reads.load(), badA.load(), badB.load(), dt / 1000.0,
           dt / (framesA.load() + framesB.load()));
  }
}
