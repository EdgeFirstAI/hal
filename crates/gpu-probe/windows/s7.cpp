// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// S7: cross-process. The parent shares a texture and a fence by name, a
// child process (this executable with --child) opens both on its own
// ANGLE display, reads the parent's bytes, renders into the texture, reads
// it through CUDA, and signals the fence; the parent waits and verifies.
#include "probe.h"

#include <d3d12.h>

#include <cstring>

static const uint32_t W = 512, H = 512;

void run_s7(GlSession& s) {
  DWORD pid = GetCurrentProcessId();
  wchar_t texname[96], fencename[96];
  swprintf(texname, 96, L"Local\\edgefirst-d3d11-probe-tex-%lu", pid);
  swprintf(fencename, 96, L"Local\\edgefirst-d3d11-probe-fence-%lu", pid);

  ComPtr<ID3D11Texture2D> tex = create_tex(s.d3d, W, H, DXGI_FORMAT_R8G8B8A8_UNORM,
                                           D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE,
                                           D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE);
  auto pat = make_pattern(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 21);
  upload_tex(s.d3d, tex.Get(), pat);
  s.d3d.ctx->Flush();
  HRESULT hr;
  HANDLE h = create_shared_handle(tex.Get(), texname, &hr);
  report("S7.1", h ? Verdict::Pass : Verdict::Fail, "named NT handle for the texture: %s (handle %p)", hr_str(hr), h);
  ComPtr<ID3D11Fence> fence;
  HANDLE fh = nullptr;
  hr = s.d3d.dev5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence));
  if (fence) hr = fence->CreateSharedHandle(nullptr, GENERIC_ALL, fencename, &fh);
  report("S7.1", fh ? Verdict::Pass : Verdict::Fail, "named NT handle for the fence: %s", hr_str(hr));
  // Legacy KMT handle on a MISC_SHARED texture: a global value.
  ComPtr<ID3D11Texture2D> kmt_tex = create_tex(s.d3d, W, H, DXGI_FORMAT_R8G8B8A8_UNORM,
                                               D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, D3D11_RESOURCE_MISC_SHARED);
  auto pat2 = make_pattern(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 22);
  upload_tex(s.d3d, kmt_tex.Get(), pat2);
  s.d3d.ctx->Flush();
  HANDLE kmt = nullptr;
  {
    ComPtr<IDXGIResource> r;
    kmt_tex.As(&r);
    if (r) r->GetSharedHandle(&kmt);
  }
  report("S7.1", kmt ? Verdict::Pass : Verdict::Info, "legacy IDXGIResource::GetSharedHandle (KMT) value: %p", kmt);
  if (!h || !fh) return;

  // Spawn the child.
  wchar_t exe[MAX_PATH];
  GetModuleFileNameW(nullptr, exe, MAX_PATH);
  std::wstring cmd = L"\"" + std::wstring(exe) + L"\" --child " + texname + L" --child-fence " + fencename;
  if (kmt) {
    wchar_t k[32];
    swprintf(k, 32, L" --child-kmt %p", kmt);
    cmd += k;
  }
  if (g_opt.warp) cmd += L" --warp";
  if (!g_opt.angle_dir.empty()) {
    int n = MultiByteToWideChar(CP_UTF8, 0, g_opt.angle_dir.c_str(), -1, nullptr, 0);
    std::wstring w(n > 0 ? n - 1 : 0, L'\0');
    if (n > 1) MultiByteToWideChar(CP_UTF8, 0, g_opt.angle_dir.c_str(), -1, w.data(), n);
    cmd += L" --angle \"" + w + L"\"";
  }
  STARTUPINFOW si{};
  si.cb = sizeof si;
  PROCESS_INFORMATION pi{};
  fflush(stdout);
  std::vector<wchar_t> cmdbuf(cmd.begin(), cmd.end());
  cmdbuf.push_back(0);
  if (!CreateProcessW(nullptr, cmdbuf.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) {
    report("S7.2", Verdict::Fail, "CreateProcess: error %lu", GetLastError());
    return;
  }
  HANDLE event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
  fence->SetEventOnCompletion(1, event);
  double t0 = now_us();
  DWORD w = WaitForSingleObject(event, 30000);
  double dt = now_us() - t0;
  report("S7.2", w == WAIT_OBJECT_0 ? Verdict::Pass : Verdict::Fail, "parent waited for the child's fence signal: %s (%.1f ms)",
         w == WAIT_OBJECT_0 ? "signalled" : "timeout", dt / 1000.0);
  if (w == WAIT_OBJECT_0) {
    std::vector<uint8_t> back;
    readback_tex(s.d3d, tex.Get(), back);
    auto want = make_gradient(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, false);
    size_t bad = count_mismatch(back.data(), want.data(), want.size());
    report("S7.2", bad == 0 ? Verdict::Pass : Verdict::Fail, "parent sees the child's GL render in the shared texture: %zu mismatches", bad);
  }
  WaitForSingleObject(pi.hProcess, 30000);
  DWORD code = 0;
  GetExitCodeProcess(pi.hProcess, &code);
  report("S7.2", code == 0 ? Verdict::Pass : Verdict::Fail, "child exit code %lu", code);
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  CloseHandle(event);
  CloseHandle(fh);
  CloseHandle(h);
}

int run_s7_child() {
  int failures = 0;
  GlSession s;
  if (!s.bring_up(g_opt.warp ? DisplayMode::AngleWarp : DisplayMode::AngleHardware)) {
    report("S7.C", Verdict::Fail, "child: ANGLE bring-up failed");
    return 1;
  }
  int n = MultiByteToWideChar(CP_UTF8, 0, g_opt.child_name.c_str(), -1, nullptr, 0);
  std::wstring texname(n > 0 ? n - 1 : 0, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, g_opt.child_name.c_str(), -1, texname.data(), n);
  n = MultiByteToWideChar(CP_UTF8, 0, g_opt.child_fence.c_str(), -1, nullptr, 0);
  std::wstring fencename(n > 0 ? n - 1 : 0, L'\0');
  MultiByteToWideChar(CP_UTF8, 0, g_opt.child_fence.c_str(), -1, fencename.data(), n);

  ComPtr<ID3D11Texture2D> tex;
  HRESULT hr = s.d3d.dev1->OpenSharedResourceByName(texname.c_str(), DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                                                    IID_PPV_ARGS(&tex));
  report("S7.C", SUCCEEDED(hr) ? Verdict::Pass : Verdict::Fail, "child: OpenSharedResourceByName on its own ANGLE device: %s", hr_str(hr));
  if (FAILED(hr)) return 1;
  D3D11_TEXTURE2D_DESC desc;
  tex->GetDesc(&desc);
  report("S7.C", Verdict::Info, "child: opened texture %ux%u fmt %d misc 0x%x", desc.Width, desc.Height, desc.Format, desc.MiscFlags);
  {
    std::vector<uint8_t> back;
    readback_tex(s.d3d, tex.Get(), back);
    auto pat = make_pattern(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 21);
    size_t bad = count_mismatch(back.data(), pat.data(), pat.size());
    report("S7.C", bad == 0 ? Verdict::Pass : Verdict::Fail, "child: reads the parent's upload: %zu mismatches", bad);
    failures += bad != 0;
  }
  // D3D11 opens fences only by handle; the name lookup lives on the D3D12
  // device, which resolves the same kernel object.
  ComPtr<ID3D11Fence> fence;
  {
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<IDXGIDevice> dxgi;
    s.d3d.dev.As(&dxgi);
    if (dxgi) dxgi->GetAdapter(&adapter);
    ComPtr<ID3D12Device> d12;
    hr = D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d12));
    HANDLE fh = nullptr;
    if (SUCCEEDED(hr)) hr = d12->OpenSharedHandleByName(fencename.c_str(), GENERIC_ALL, &fh);
    if (SUCCEEDED(hr)) hr = s.d3d.dev5->OpenSharedFence(fh, IID_PPV_ARGS(&fence));
    if (fh) CloseHandle(fh);
  }
  report("S7.C", SUCCEEDED(hr) ? Verdict::Pass : Verdict::Fail,
         "child: fence by name (ID3D12Device::OpenSharedHandleByName -> ID3D11Device5::OpenSharedFence): %s", hr_str(hr));
  if (FAILED(hr)) return 1;

  if (!g_opt.child_kmt.empty()) {
    unsigned long long v = 0;
    sscanf_s(g_opt.child_kmt.c_str(), "%llx", &v);
    ComPtr<ID3D11Texture2D> kt;
    hr = s.d3d.dev->OpenSharedResource((HANDLE)(uintptr_t)v, IID_PPV_ARGS(&kt));
    std::vector<uint8_t> back;
    size_t bad = SIZE_MAX;
    if (kt) {
      readback_tex(s.d3d, kt.Get(), back);
      auto pat2 = make_pattern(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 22);
      bad = count_mismatch(back.data(), pat2.data(), pat2.size());
    }
    report("S7.C", SUCCEEDED(hr) && bad == 0 ? Verdict::Pass : Verdict::Info,
           "child: legacy KMT handle %s opened with OpenSharedResource: %s, %zu mismatches", g_opt.child_kmt.c_str(), hr_str(hr), bad);
  }

  // Render into the parent's texture, verify through CUDA, signal.
  Import im = import_texture(s, Route::EglImage, tex.Get(), GL_RGBA);
  if (!im.ok) {
    report("S7.C", Verdict::Fail, "child: EGLImage import: %s / gl 0x%04x", egl_err_str(im.egl_error), im.gl_error);
    return 1;
  }
  std::string log;
  GLuint p_grad = compile_program(kVertexShader, kGradientFragment, &log);
  Quad quad;
  quad.init();
  GLuint fbo;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, im.tex, 0);
  glViewport(0, 0, W, H);
  glUseProgram(p_grad);
  glUniform1f(glGetUniformLocation(p_grad, "denom"), 255.0f);
  quad.draw();
  glFinish();
  auto want = make_gradient(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, false);
  if (!g_opt.warp) {
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<IDXGIDevice> dxgi;
    s.d3d.dev.As(&dxgi);
    if (dxgi) dxgi->GetAdapter(&adapter);
    std::vector<uint8_t> out;
    std::string detail;
    bool ok = cuda_external_read_rgba8(tex.Get(), adapter.Get(), W, H, out, &detail);
    size_t bad = ok ? count_mismatch(out.data(), want.data(), want.size()) : SIZE_MAX;
    report("S7.C", ok && bad == 0 ? Verdict::Pass : (ok ? Verdict::Fail : Verdict::Skip),
           "child: CUDA external-memory read of the opened texture: %s%s", ok ? "" : detail.c_str(),
           ok ? (bad ? "bad bytes" : "0 mismatches") : "");
    failures += ok && bad != 0;
  }
  s.d3d.ctx4->Signal(fence.Get(), 1);
  s.d3d.ctx->Flush();
  report("S7.C", Verdict::Pass, "child: rendered the gradient and signalled fence value 1");
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);
  glDeleteProgram(p_grad);
  quad.destroy();
  import_destroy(s, im);
  s.shutdown();
  fflush(stdout);
  return failures ? 1 : 0;
}
