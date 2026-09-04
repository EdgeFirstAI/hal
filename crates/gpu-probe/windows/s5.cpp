// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// S5: CUDA on the same D3D11 texture. Runtime API through whichever
// cudart64_*.dll is present and driver API through nvcuda.dll, both loaded
// at run time. Register / map / linearize with verification, external
// memory through the NT handle, ordering against GL with and without the
// D3D11 fence as an external semaphore, and timing against the PBO path.
#include "probe.h"

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cudaD3D11.h>
#include <cudaD3D11Typedefs.h>
#include <cuda_runtime_api.h>
#include <cuda_d3d11_interop.h>

#include <cstring>

// ---------------------------------------------------------------------------
// Runtime API table
// ---------------------------------------------------------------------------

struct CudaRt {
  HMODULE lib = nullptr;
  std::string dll;
  cudaError_t (*RuntimeGetVersion)(int*) = nullptr;
  cudaError_t (*DriverGetVersion)(int*) = nullptr;
  cudaError_t (*GetDeviceCount)(int*) = nullptr;
  cudaError_t (*SetDevice)(int) = nullptr;
  cudaError_t (*D3D11GetDevice)(int*, IDXGIAdapter*) = nullptr;
  cudaError_t (*GraphicsD3D11RegisterResource)(cudaGraphicsResource_t*, ID3D11Resource*, unsigned) = nullptr;
  cudaError_t (*GraphicsMapResources)(int, cudaGraphicsResource_t*, cudaStream_t) = nullptr;
  cudaError_t (*GraphicsSubResourceGetMappedArray)(cudaArray_t*, cudaGraphicsResource_t, unsigned, unsigned) = nullptr;
  cudaError_t (*GraphicsUnmapResources)(int, cudaGraphicsResource_t*, cudaStream_t) = nullptr;
  cudaError_t (*GraphicsUnregisterResource)(cudaGraphicsResource_t) = nullptr;
  cudaError_t (*ArrayGetInfo)(cudaChannelFormatDesc*, cudaExtent*, unsigned*, cudaArray_t) = nullptr;
  cudaError_t (*Malloc)(void**, size_t) = nullptr;
  cudaError_t (*Free)(void*) = nullptr;
  cudaError_t (*Memcpy2DFromArray)(void*, size_t, cudaArray_const_t, size_t, size_t, size_t, size_t, cudaMemcpyKind) = nullptr;
  cudaError_t (*Memcpy)(void*, const void*, size_t, cudaMemcpyKind) = nullptr;
  cudaError_t (*MemcpyAsync)(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) = nullptr;
  cudaError_t (*DeviceSynchronize)() = nullptr;
  cudaError_t (*GetLastError)() = nullptr;
  const char* (*GetErrorString)(cudaError_t) = nullptr;
  cudaError_t (*ImportExternalMemory)(cudaExternalMemory_t*, const cudaExternalMemoryHandleDesc*) = nullptr;
  cudaError_t (*ExternalMemoryGetMappedMipmappedArray)(cudaMipmappedArray_t*, cudaExternalMemory_t, const cudaExternalMemoryMipmappedArrayDesc*) = nullptr;
  cudaError_t (*GetMipmappedArrayLevel)(cudaArray_t*, cudaMipmappedArray_const_t, unsigned) = nullptr;
  cudaError_t (*FreeMipmappedArray)(cudaMipmappedArray_t) = nullptr;
  cudaError_t (*DestroyExternalMemory)(cudaExternalMemory_t) = nullptr;
  cudaError_t (*ImportExternalSemaphore)(cudaExternalSemaphore_t*, const cudaExternalSemaphoreHandleDesc*) = nullptr;
  cudaError_t (*WaitExternalSemaphoresAsync)(const cudaExternalSemaphore_t*, const cudaExternalSemaphoreWaitParams*, unsigned, cudaStream_t) = nullptr;
  cudaError_t (*DestroyExternalSemaphore)(cudaExternalSemaphore_t) = nullptr;
  cudaError_t (*StreamCreate)(cudaStream_t*) = nullptr;
  cudaError_t (*StreamSynchronize)(cudaStream_t) = nullptr;
  cudaError_t (*StreamDestroy)(cudaStream_t) = nullptr;

  template <typename F>
  bool sym(F& f, const char* name, const char* alt = nullptr) {
    f = (F)GetProcAddress(lib, name);
    if (!f && alt) f = (F)GetProcAddress(lib, alt);
    if (!f) printf("         cudart lacks %s\n", name);
    return f != nullptr;
  }

  bool load() {
    if (lib) return true;
    static const char* names[] = {"cudart64_13.dll", "cudart64_12.dll", "cudart64_110.dll", "cudart64_101.dll"};
    std::vector<std::string> dirs = {""};
    char env[MAX_PATH];
    if (GetEnvironmentVariableA("CUDA_PATH", env, sizeof env)) dirs.push_back(std::string(env) + "\\bin\\");
    for (const std::string& d : dirs) {
      for (const char* n : names) {
        std::string p = d + n;
        lib = d.empty() ? LoadLibraryA(n) : LoadLibraryExA(p.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
        if (lib) {
          char full[MAX_PATH];
          GetModuleFileNameA(lib, full, sizeof full);
          dll = full;
          break;
        }
      }
      if (lib) break;
    }
    if (!lib) return false;
    bool ok = true;
    ok &= sym(RuntimeGetVersion, "cudaRuntimeGetVersion");
    ok &= sym(DriverGetVersion, "cudaDriverGetVersion");
    ok &= sym(GetDeviceCount, "cudaGetDeviceCount");
    ok &= sym(SetDevice, "cudaSetDevice");
    ok &= sym(D3D11GetDevice, "cudaD3D11GetDevice");
    ok &= sym(GraphicsD3D11RegisterResource, "cudaGraphicsD3D11RegisterResource");
    ok &= sym(GraphicsMapResources, "cudaGraphicsMapResources");
    ok &= sym(GraphicsSubResourceGetMappedArray, "cudaGraphicsSubResourceGetMappedArray");
    ok &= sym(GraphicsUnmapResources, "cudaGraphicsUnmapResources");
    ok &= sym(GraphicsUnregisterResource, "cudaGraphicsUnregisterResource");
    ok &= sym(ArrayGetInfo, "cudaArrayGetInfo");
    ok &= sym(Malloc, "cudaMalloc");
    ok &= sym(Free, "cudaFree");
    ok &= sym(Memcpy2DFromArray, "cudaMemcpy2DFromArray");
    ok &= sym(Memcpy, "cudaMemcpy");
    ok &= sym(MemcpyAsync, "cudaMemcpyAsync");
    ok &= sym(DeviceSynchronize, "cudaDeviceSynchronize");
    ok &= sym(GetLastError, "cudaGetLastError");
    ok &= sym(GetErrorString, "cudaGetErrorString");
    ok &= sym(ImportExternalMemory, "cudaImportExternalMemory");
    ok &= sym(ExternalMemoryGetMappedMipmappedArray, "cudaExternalMemoryGetMappedMipmappedArray");
    ok &= sym(GetMipmappedArrayLevel, "cudaGetMipmappedArrayLevel");
    ok &= sym(FreeMipmappedArray, "cudaFreeMipmappedArray");
    ok &= sym(DestroyExternalMemory, "cudaDestroyExternalMemory");
    ok &= sym(ImportExternalSemaphore, "cudaImportExternalSemaphore");
    ok &= sym(WaitExternalSemaphoresAsync, "cudaWaitExternalSemaphoresAsync_v2", "cudaWaitExternalSemaphoresAsync");
    ok &= sym(DestroyExternalSemaphore, "cudaDestroyExternalSemaphore");
    ok &= sym(StreamCreate, "cudaStreamCreate");
    ok &= sym(StreamSynchronize, "cudaStreamSynchronize");
    ok &= sym(StreamDestroy, "cudaStreamDestroy");
    return ok;
  }
  const char* err(cudaError_t e) { return GetErrorString ? GetErrorString(e) : "?"; }
};

static CudaRt g_rt;

// ---------------------------------------------------------------------------
// Driver API table (nvcuda.dll, versioned entry points)
// ---------------------------------------------------------------------------

struct CudaDrv {
  HMODULE lib = nullptr;
  PFN_cuInit Init = nullptr;
  PFN_cuDriverGetVersion DriverGetVersion = nullptr;
  PFN_cuDeviceGetLuid DeviceGetLuid = nullptr;
  PFN_cuDeviceGetName DeviceGetName = nullptr;
  PFN_cuD3D11GetDevice D3D11GetDevice = nullptr;
  PFN_cuDevicePrimaryCtxRetain PrimaryCtxRetain = nullptr;
  PFN_cuDevicePrimaryCtxRelease PrimaryCtxRelease = nullptr;
  PFN_cuCtxSetCurrent CtxSetCurrent = nullptr;
  PFN_cuGraphicsD3D11RegisterResource GraphicsD3D11RegisterResource = nullptr;
  PFN_cuGraphicsMapResources GraphicsMapResources = nullptr;
  PFN_cuGraphicsSubResourceGetMappedArray GraphicsSubResourceGetMappedArray = nullptr;
  PFN_cuGraphicsUnmapResources GraphicsUnmapResources = nullptr;
  PFN_cuGraphicsUnregisterResource GraphicsUnregisterResource = nullptr;
  PFN_cuArrayGetDescriptor ArrayGetDescriptor = nullptr;
  PFN_cuMemAlloc MemAlloc = nullptr;
  PFN_cuMemFree MemFree = nullptr;
  PFN_cuMemcpy2D Memcpy2D = nullptr;
  PFN_cuMemcpyDtoH MemcpyDtoH = nullptr;
  PFN_cuCtxSynchronize CtxSynchronize = nullptr;
  PFN_cuGetErrorString GetErrorString = nullptr;

  template <typename F>
  bool sym(F& f, const char* name) {
    f = (F)GetProcAddress(lib, name);
    if (!f) printf("         nvcuda lacks %s\n", name);
    return f != nullptr;
  }
  bool load() {
    lib = LoadLibraryA("nvcuda.dll");
    if (!lib) return false;
    bool ok = true;
    ok &= sym(Init, "cuInit");
    ok &= sym(DriverGetVersion, "cuDriverGetVersion");
    ok &= sym(DeviceGetLuid, "cuDeviceGetLuid");
    ok &= sym(DeviceGetName, "cuDeviceGetName");
    ok &= sym(D3D11GetDevice, "cuD3D11GetDevice");
    ok &= sym(PrimaryCtxRetain, "cuDevicePrimaryCtxRetain");
    ok &= sym(PrimaryCtxRelease, "cuDevicePrimaryCtxRelease_v2");
    ok &= sym(CtxSetCurrent, "cuCtxSetCurrent");
    ok &= sym(GraphicsD3D11RegisterResource, "cuGraphicsD3D11RegisterResource");
    ok &= sym(GraphicsMapResources, "cuGraphicsMapResources");
    ok &= sym(GraphicsSubResourceGetMappedArray, "cuGraphicsSubResourceGetMappedArray");
    ok &= sym(GraphicsUnmapResources, "cuGraphicsUnmapResources");
    ok &= sym(GraphicsUnregisterResource, "cuGraphicsUnregisterResource");
    ok &= sym(ArrayGetDescriptor, "cuArrayGetDescriptor_v2");
    ok &= sym(MemAlloc, "cuMemAlloc_v2");
    ok &= sym(MemFree, "cuMemFree_v2");
    ok &= sym(Memcpy2D, "cuMemcpy2D_v2");
    ok &= sym(MemcpyDtoH, "cuMemcpyDtoH_v2");
    ok &= sym(CtxSynchronize, "cuCtxSynchronize");
    ok &= sym(GetErrorString, "cuGetErrorString");
    return ok;
  }
  const char* err(CUresult r) {
    const char* s = "?";
    if (GetErrorString) GetErrorString(r, &s);
    return s;
  }
};

static const char* channel_desc_str(const cudaChannelFormatDesc& d) {
  thread_local char buf[64];
  const char* kind = d.f == cudaChannelFormatKindUnsigned ? "u" : d.f == cudaChannelFormatKindSigned ? "s" : d.f == cudaChannelFormatKindFloat ? "f" : "?";
  snprintf(buf, sizeof buf, "%s%d,%d,%d,%d", kind, d.x, d.y, d.z, d.w);
  return buf;
}

// Map the registered resource, copy sub-resource 0 into linear device
// memory and back to the host; returns the mismatch count or SIZE_MAX.
static size_t rt_linearize_and_verify(CudaRt& rt, cudaGraphicsResource_t res, uint32_t w, uint32_t h, uint32_t bpp,
                                      const std::vector<uint8_t>& expect, std::string* detail) {
  cudaError_t e = rt.GraphicsMapResources(1, &res, nullptr);
  if (e) {
    *detail = std::string("map: ") + rt.err(e);
    return SIZE_MAX;
  }
  cudaArray_t arr = nullptr;
  e = rt.GraphicsSubResourceGetMappedArray(&arr, res, 0, 0);
  if (e) {
    *detail = std::string("array: ") + rt.err(e);
    rt.GraphicsUnmapResources(1, &res, nullptr);
    return SIZE_MAX;
  }
  cudaChannelFormatDesc cd{};
  cudaExtent ext{};
  unsigned flags = 0;
  rt.ArrayGetInfo(&cd, &ext, &flags, arr);
  void* dptr = nullptr;
  size_t bytes = (size_t)w * h * bpp;
  rt.Malloc(&dptr, bytes);
  e = rt.Memcpy2DFromArray(dptr, (size_t)w * bpp, arr, 0, 0, (size_t)w * bpp, h, cudaMemcpyDeviceToDevice);
  std::vector<uint8_t> host(bytes);
  if (!e) e = rt.Memcpy(host.data(), dptr, bytes, cudaMemcpyDeviceToHost);
  rt.Free(dptr);
  rt.GraphicsUnmapResources(1, &res, nullptr);
  char b[128];
  snprintf(b, sizeof b, "array %s %zux%zu", channel_desc_str(cd), ext.width, ext.height);
  *detail = b;
  if (e) {
    *detail += std::string(" copy: ") + rt.err(e);
    return SIZE_MAX;
  }
  return count_mismatch(host.data(), expect.data(), std::min(bytes, expect.size()));
}

bool cuda_external_read_rgba8(ID3D11Texture2D* tex, IDXGIAdapter* adapter, uint32_t w, uint32_t h,
                              std::vector<uint8_t>& out, std::string* detail) {
  CudaRt& rt = g_rt;
  if (!rt.load()) {
    *detail = "no cudart";
    return false;
  }
  int dev = -1;
  cudaError_t e = rt.D3D11GetDevice(&dev, adapter);
  if (e) {
    *detail = std::string("cudaD3D11GetDevice: ") + rt.err(e);
    return false;
  }
  rt.SetDevice(dev);
  HRESULT hr;
  HANDLE hnd = create_shared_handle(tex, nullptr, &hr);
  if (!hnd) {
    *detail = std::string("CreateSharedHandle: ") + hr_str(hr);
    return false;
  }
  cudaExternalMemoryHandleDesc hd{};
  hd.type = cudaExternalMemoryHandleTypeD3D11Resource;
  hd.handle.win32.handle = hnd;
  hd.size = (size_t)w * h * 4;
  hd.flags = cudaExternalMemoryDedicated;
  cudaExternalMemory_t ext = nullptr;
  e = rt.ImportExternalMemory(&ext, &hd);
  CloseHandle(hnd);
  if (e) {
    *detail = std::string("cudaImportExternalMemory: ") + rt.err(e);
    return false;
  }
  cudaExternalMemoryMipmappedArrayDesc md{};
  md.formatDesc = cudaChannelFormatDesc{8, 8, 8, 8, cudaChannelFormatKindUnsigned};
  md.extent = cudaExtent{w, h, 0};
  md.numLevels = 1;
  cudaMipmappedArray_t mm = nullptr;
  e = rt.ExternalMemoryGetMappedMipmappedArray(&mm, ext, &md);
  bool ok = false;
  if (!e) {
    cudaArray_t level0 = nullptr;
    rt.GetMipmappedArrayLevel(&level0, mm, 0);
    void* dptr = nullptr;
    rt.Malloc(&dptr, (size_t)w * h * 4);
    out.resize((size_t)w * h * 4);
    e = rt.Memcpy2DFromArray(dptr, w * 4, level0, 0, 0, w * 4, h, cudaMemcpyDeviceToDevice);
    if (!e) e = rt.Memcpy(out.data(), dptr, out.size(), cudaMemcpyDeviceToHost);
    rt.Free(dptr);
    rt.FreeMipmappedArray(mm);
    ok = e == cudaSuccess;
    if (!ok) *detail = std::string("copy: ") + rt.err(e);
  } else {
    *detail = std::string("mipmapped array: ") + rt.err(e);
  }
  rt.DestroyExternalMemory(ext);
  return ok;
}

// A D3D11 texture with an EGLImage import and an FBO, for the timing loops.
struct GlTarget {
  ComPtr<ID3D11Texture2D> tex;
  Import im;
  GLuint fbo = 0;
  bool init(GlSession& s, uint32_t w, uint32_t h, DXGI_FORMAT fmt, UINT misc) {
    tex = create_tex(s.d3d, w, h, fmt, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, misc);
    if (!tex) return false;
    im = import_texture(s, Route::EglImage, tex.Get(), gl_internal_format_for(fmt));
    if (!im.ok) return false;
    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, im.tex, 0);
    return glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE;
  }
  void destroy(GlSession& s) {
    if (fbo) glDeleteFramebuffers(1, &fbo);
    import_destroy(s, im);
    fbo = 0;
  }
};

void run_s5(GlSession& s) {
  CudaRt& rt = g_rt;
  CudaDrv drv;
  bool have_rt = rt.load();
  bool have_drv = drv.load();
  if (!have_rt && !have_drv) {
    report("S5.1", Verdict::Skip, "no cudart64_*.dll and no nvcuda.dll found");
    return;
  }
  if (have_rt) {
    int rv = 0, dv = 0;
    rt.RuntimeGetVersion(&rv);
    rt.DriverGetVersion(&dv);
    int n = 0;
    cudaError_t e = rt.GetDeviceCount(&n);
    report("S5.1", e == cudaSuccess && n > 0 ? Verdict::Pass : Verdict::Skip, "runtime %s: version %d, driver %d, %d device(s) (%s)",
           rt.dll.c_str(), rv, dv, n, rt.err(e));
    if (e || n == 0) have_rt = false;
  } else {
    report("S5.1", Verdict::Skip, "no cudart64_*.dll found; runtime API sub-tests skipped");
  }
  if (have_drv) {
    CUresult r = drv.Init(0);
    int dv = 0;
    drv.DriverGetVersion(&dv);
    report("S5.1", r == CUDA_SUCCESS ? Verdict::Pass : Verdict::Skip, "driver API nvcuda.dll: cuInit %s, version %d", drv.err(r), dv);
    if (r) have_drv = false;
  }
  if (s.mode == DisplayMode::AngleWarp || s.mode == DisplayMode::InjectWarp) {
    report("S5.2", Verdict::Skip, "WARP adapter has no CUDA device");
    return;
  }

  // ---- S5.2 device match ------------------------------------------------
  ComPtr<IDXGIAdapter> adapter;
  {
    ComPtr<IDXGIDevice> dxgi;
    s.d3d.dev.As(&dxgi);
    if (dxgi) dxgi->GetAdapter(&adapter);
  }
  int rt_dev = -1;
  if (have_rt) {
    cudaError_t e = rt.D3D11GetDevice(&rt_dev, adapter.Get());
    if (!e) rt.SetDevice(rt_dev);
    report("S5.2", e == cudaSuccess ? Verdict::Pass : Verdict::Fail, "cudaD3D11GetDevice(ANGLE's adapter) -> device %d (%s)", rt_dev, rt.err(e));
    if (e) have_rt = false;
  }
  CUdevice cu_dev = -1;
  CUcontext cu_ctx = nullptr;
  if (have_drv) {
    CUresult r = drv.D3D11GetDevice(&cu_dev, adapter.Get());
    char luid[8] = {};
    unsigned mask = 0;
    char name[128] = {};
    if (!r) {
      drv.DeviceGetLuid(luid, &mask, cu_dev);
      drv.DeviceGetName(name, sizeof name, cu_dev);
    }
    LUID l{};
    memcpy(&l, luid, 8);
    bool same = l.LowPart == s.d3d.luid.LowPart && l.HighPart == s.d3d.luid.HighPart;
    report("S5.2", r == CUDA_SUCCESS && same ? Verdict::Pass : Verdict::Fail,
           "cuD3D11GetDevice -> %d (%s), cuDeviceGetLuid %s vs D3D11 adapter %s: %s", cu_dev, name, luid_str(l).c_str(),
           luid_str(s.d3d.luid).c_str(), same ? "match" : "DIFFERENT");
    if (!r) {
      r = drv.PrimaryCtxRetain(&cu_ctx, cu_dev);
      if (!r) r = drv.CtxSetCurrent(cu_ctx);
      if (r) {
        report("S5.2", Verdict::Fail, "primary context: %s", drv.err(r));
        have_drv = false;
      }
    } else {
      have_drv = false;
    }
  }

  // ---- S5.3 register matrix (runtime API) ---------------------------------
  struct FmtCase {
    DXGI_FORMAT fmt;
    GLenum gl;
  };
  static const FmtCase fmts[] = {{DXGI_FORMAT_R8G8B8A8_UNORM, GL_RGBA}, {DXGI_FORMAT_B8G8R8A8_UNORM, GL_BGRA_EXT},
                                 {DXGI_FORMAT_R8_UNORM, GL_RED_EXT},     {DXGI_FORMAT_R8G8_UNORM, GL_RG_EXT},
                                 {DXGI_FORMAT_R16G16B16A16_FLOAT, GL_RGBA}, {DXGI_FORMAT_R32G32B32A32_FLOAT, GL_RGBA},
                                 {DXGI_FORMAT_NV12, 0}};
  struct MiscCase {
    UINT misc;
    const char* name;
  };
  static const MiscCase miscs[] = {{0, "misc=0"},
                                   {D3D11_RESOURCE_MISC_SHARED, "SHARED"},
                                   {D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE, "SHARED|NTHANDLE"},
                                   {D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX | D3D11_RESOURCE_MISC_SHARED_NTHANDLE, "KEYEDMUTEX|NTHANDLE"},
                                   {D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX, "KEYEDMUTEX"}};
  const uint32_t W = 256, H = 128;
  if (have_rt) {
    for (const FmtCase& fc : fmts) {
      std::string line;
      for (const MiscCase& mc : miscs) {
        HRESULT hr;
        ComPtr<ID3D11Texture2D> tex = create_tex(s.d3d, W, H, fc.fmt, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, mc.misc, &hr);
        line += std::string(mc.name) + "=";
        if (!tex) {
          line += std::string("create:") + hr_str(hr) + "  ";
          continue;
        }
        auto pat = make_pattern(fc.fmt, W, H, 9);
        {
          KeyedLock lock(tex.Get());
          upload_tex(s.d3d, tex.Get(), pat);
          s.d3d.ctx->Flush();
        }
        cudaGraphicsResource_t res = nullptr;
        cudaError_t e = rt.GraphicsD3D11RegisterResource(&res, tex.Get(), cudaGraphicsRegisterFlagsNone);
        if (e) {
          line += std::string("register:") + rt.err(e) + "  ";
          rt.GetLastError();
          continue;
        }
        std::string detail;
        uint32_t bpp = fc.fmt == DXGI_FORMAT_NV12 ? 1 : bytes_per_pixel(fc.fmt);
        size_t bad;
        {
          KeyedLock lock(tex.Get());
          bad = rt_linearize_and_verify(rt, res, W, H, bpp, pat, &detail);
        }
        rt.GraphicsUnregisterResource(res);
        if (bad == SIZE_MAX) line += "[" + detail + "]  ";
        else {
          char b[32];
          snprintf(b, sizeof b, " %zu bad]  ", bad);
          line += "[" + detail + b;
        }
        if (mc.misc & D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX) {
          e = rt.GraphicsD3D11RegisterResource(&res, tex.Get(), cudaGraphicsRegisterFlagsNone);
          if (!e) {
            bad = rt_linearize_and_verify(rt, res, W, H, bpp, pat, &detail);
            rt.GraphicsUnregisterResource(res);
            char b[96];
            snprintf(b, sizeof b, "(mutex not held: %s)  ", bad == SIZE_MAX ? detail.c_str() : (bad ? "bad bytes" : "ok"));
            line += b;
          }
        }
      }
      report("S5.3", Verdict::Info, "%-20s %s", fmt_name(fc.fmt), line.c_str());
    }
  }

  // ---- S5.4 external memory through the NT handle -------------------------
  if (have_rt) {
    for (int keyed = 0; keyed < 2; keyed++) {
      UINT misc = keyed ? (D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX | D3D11_RESOURCE_MISC_SHARED_NTHANDLE)
                        : (D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE);
      ComPtr<ID3D11Texture2D> tex = create_tex(s.d3d, W, H, DXGI_FORMAT_R8G8B8A8_UNORM,
                                               D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, misc);
      auto pat = make_pattern(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 13);
      {
        KeyedLock lock(tex.Get());
        upload_tex(s.d3d, tex.Get(), pat);
        s.d3d.ctx->Flush();
      }
      HRESULT hr;
      HANDLE h = create_shared_handle(tex.Get(), nullptr, &hr);
      if (!h) {
        report("S5.4", Verdict::Fail, "CreateSharedHandle: %s", hr_str(hr));
        continue;
      }
      // D3D11 has no allocation-size query; try the tight size, the
      // 128-byte-pitch size, a generous overestimate, and zero.
      size_t sizes[] = {(size_t)W * H * 4, (size_t)((W * 4 + 127) & ~127u) * H, (size_t)W * H * 4 * 2, 0};
      const char* size_names[] = {"tight", "pitch128", "2x", "zero"};
      for (int si = 0; si < 4; si++) {
        cudaExternalMemoryHandleDesc hd{};
        hd.type = cudaExternalMemoryHandleTypeD3D11Resource;
        hd.handle.win32.handle = h;
        hd.size = sizes[si];
        hd.flags = cudaExternalMemoryDedicated;
        cudaExternalMemory_t ext = nullptr;
        cudaError_t e = rt.ImportExternalMemory(&ext, &hd);
        if (e) {
          report("S5.4", Verdict::Info, "%s cudaImportExternalMemory(D3D11Resource, size %s=%zu): %s",
                 keyed ? "keyed-mutex" : "shared", size_names[si], sizes[si], rt.err(e));
          rt.GetLastError();
          continue;
        }
        cudaExternalMemoryMipmappedArrayDesc md{};
        md.offset = 0;
        md.formatDesc = cudaChannelFormatDesc{8, 8, 8, 8, cudaChannelFormatKindUnsigned};
        md.extent = cudaExtent{W, H, 0};
        md.flags = 0;
        md.numLevels = 1;
        cudaMipmappedArray_t mm = nullptr;
        e = rt.ExternalMemoryGetMappedMipmappedArray(&mm, ext, &md);
        std::string outcome;
        if (e) {
          outcome = std::string("mipmapped array: ") + rt.err(e);
        } else {
          cudaArray_t level0 = nullptr;
          rt.GetMipmappedArrayLevel(&level0, mm, 0);
          void* dptr = nullptr;
          rt.Malloc(&dptr, (size_t)W * H * 4);
          std::vector<uint8_t> host((size_t)W * H * 4);
          KeyedLock lock(tex.Get());
          e = rt.Memcpy2DFromArray(dptr, W * 4, level0, 0, 0, W * 4, H, cudaMemcpyDeviceToDevice);
          if (!e) e = rt.Memcpy(host.data(), dptr, host.size(), cudaMemcpyDeviceToHost);
          rt.Free(dptr);
          if (e) outcome = std::string("copy: ") + rt.err(e);
          else {
            size_t bad = count_mismatch(host.data(), pat.data(), pat.size());
            char b[64];
            snprintf(b, sizeof b, "%zu bad bytes", bad);
            outcome = b;
          }
          rt.FreeMipmappedArray(mm);
        }
        report("S5.4", Verdict::Info, "%s cudaImportExternalMemory(D3D11Resource, size %s=%zu): ok; %s",
               keyed ? "keyed-mutex" : "shared", size_names[si], sizes[si], outcome.c_str());
        rt.DestroyExternalMemory(ext);
      }
      CloseHandle(h);
    }
  }

  // ---- S5.5 timing: register path vs PBO baseline vs external memory ------
  if (have_rt) {
    Quad quad;
    quad.init();
    std::string log;
    GLuint p_grad = compile_program(kVertexShader, kGradientFragment, &log);
    GLint denom = glGetUniformLocation(p_grad, "denom");
    for (DXGI_FORMAT fmt : {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT}) {
      const uint32_t CW = 1920, CH = 1080;
      uint32_t bpp = bytes_per_pixel(fmt);
      float dn = fmt == DXGI_FORMAT_R8G8B8A8_UNORM ? 255.0f : 256.0f;
      auto want = make_gradient(fmt, CW, CH, false);
      char what[160];
      void* dlinear = nullptr;
      rt.Malloc(&dlinear, (size_t)CW * CH * bpp);
      std::vector<uint8_t> host((size_t)CW * CH * bpp);

      // Register path needs a texture without NT-handle sharing (S5.3).
      GlTarget reg;
      if (!reg.init(s, CW, CH, fmt, 0)) {
        report("S5.5", Verdict::Fail, "%s register-path target setup failed", fmt_name(fmt));
        continue;
      }
      auto draw_into = [&](GlTarget& t) {
        glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
        glViewport(0, 0, CW, CH);
        glUseProgram(p_grad);
        glUniform1f(denom, dn);
        quad.draw();
        glFlush();
      };
      cudaGraphicsResource_t res = nullptr;
      cudaError_t e = rt.GraphicsD3D11RegisterResource(&res, reg.tex.Get(), cudaGraphicsRegisterFlagsNone);
      if (e) {
        report("S5.5", Verdict::Fail, "%s register: %s", fmt_name(fmt), rt.err(e));
      } else {
        auto reg_copy = [&] {
          rt.GraphicsMapResources(1, &res, nullptr);
          cudaArray_t arr;
          rt.GraphicsSubResourceGetMappedArray(&arr, res, 0, 0);
          rt.Memcpy2DFromArray(dlinear, (size_t)CW * bpp, arr, 0, 0, (size_t)CW * bpp, CH, cudaMemcpyDeviceToDevice);
          rt.GraphicsUnmapResources(1, &res, nullptr);
          rt.DeviceSynchronize();
        };
        Stats st = time_loop(g_opt.iters, [&] {
          draw_into(reg);
          reg_copy();
        });
        snprintf(what, sizeof what, "%s 1080p: GL draw + CUDA map + D2D linearize + unmap + sync", fmt_name(fmt));
        report_stats("S5.T", what, st);
        st = time_loop(g_opt.iters, reg_copy);
        snprintf(what, sizeof what, "%s 1080p: CUDA map + D2D linearize + unmap + sync (no draw)", fmt_name(fmt));
        report_stats("S5.T", what, st);
        draw_into(reg);
        reg_copy();
        rt.Memcpy(host.data(), dlinear, host.size(), cudaMemcpyDeviceToHost);
        size_t bad = count_mismatch(host.data(), want.data(), want.size());
        report("S5.5", bad == 0 ? Verdict::Pass : Verdict::Fail, "%s 1080p register-path linearized copy matches the GL render: %zu mismatches",
               fmt_name(fmt), bad);
      }

      // Baseline: PBO readback + cudaMemcpy H2D from the mapped pointer.
      {
        GLint read_type = 0;
        glBindFramebuffer(GL_FRAMEBUFFER, reg.fbo);
        glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &read_type);
        GLenum type = fmt == DXGI_FORMAT_R8G8B8A8_UNORM ? GL_UNSIGNED_BYTE : (read_type == GL_HALF_FLOAT ? GL_HALF_FLOAT : GL_FLOAT);
        size_t px_bytes = type == GL_UNSIGNED_BYTE ? 4 : (type == GL_HALF_FLOAT ? 8 : 16);
        size_t pbo_size = (size_t)CW * CH * px_bytes;
        GLuint pbo;
        glGenBuffers(1, &pbo);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
        glBufferData(GL_PIXEL_PACK_BUFFER, (GLsizeiptr)pbo_size, nullptr, GL_STREAM_READ);
        void* dbase = nullptr;
        rt.Malloc(&dbase, pbo_size);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        Stats st = time_loop(g_opt.iters, [&] {
          glBindFramebuffer(GL_FRAMEBUFFER, reg.fbo);
          glViewport(0, 0, CW, CH);
          glUseProgram(p_grad);
          glUniform1f(denom, dn);
          quad.draw();
          glReadPixels(0, 0, CW, CH, GL_RGBA, type, nullptr);
          void* p = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, (GLsizeiptr)pbo_size, GL_MAP_READ_BIT);
          if (p) {
            rt.Memcpy(dbase, p, pbo_size, cudaMemcpyHostToDevice);
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
          }
          rt.DeviceSynchronize();
        });
        snprintf(what, sizeof what, "%s 1080p baseline: GL draw + glReadPixels(PBO %s) + map + cudaMemcpy H2D + sync", fmt_name(fmt),
                 type == GL_UNSIGNED_BYTE ? "u8" : (type == GL_HALF_FLOAT ? "f16" : "f32"));
        report_stats("S5.T", what, st);
        rt.Free(dbase);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        glDeleteBuffers(1, &pbo);
      }

      // External memory: NT-handle texture, no per-frame map/unmap.
      GlTarget ext_t;
      if (ext_t.init(s, CW, CH, fmt, D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE)) {
        HRESULT hr;
        HANDLE h = create_shared_handle(ext_t.tex.Get(), nullptr, &hr);
        cudaExternalMemoryHandleDesc hd{};
        hd.type = cudaExternalMemoryHandleTypeD3D11Resource;
        hd.handle.win32.handle = h;
        hd.size = (size_t)CW * CH * bpp;
        hd.flags = cudaExternalMemoryDedicated;
        cudaExternalMemory_t ext = nullptr;
        cudaError_t e2 = h ? rt.ImportExternalMemory(&ext, &hd) : cudaErrorInvalidValue;
        if (!e2) {
          cudaExternalMemoryMipmappedArrayDesc md{};
          md.formatDesc = fmt == DXGI_FORMAT_R8G8B8A8_UNORM ? cudaChannelFormatDesc{8, 8, 8, 8, cudaChannelFormatKindUnsigned}
                                                            : cudaChannelFormatDesc{16, 16, 16, 16, cudaChannelFormatKindFloat};
          md.extent = cudaExtent{CW, CH, 0};
          md.numLevels = 1;
          cudaMipmappedArray_t mm = nullptr;
          e2 = rt.ExternalMemoryGetMappedMipmappedArray(&mm, ext, &md);
          if (!e2) {
            cudaArray_t level0 = nullptr;
            rt.GetMipmappedArrayLevel(&level0, mm, 0);
            Stats st = time_loop(g_opt.iters, [&] {
              draw_into(ext_t);
              rt.Memcpy2DFromArray(dlinear, (size_t)CW * bpp, level0, 0, 0, (size_t)CW * bpp, CH, cudaMemcpyDeviceToDevice);
              rt.DeviceSynchronize();
            });
            snprintf(what, sizeof what, "%s 1080p: GL draw + D2D linearize from external-memory array + sync (no map)", fmt_name(fmt));
            report_stats("S5.T", what, st);
            st = time_loop(g_opt.iters, [&] {
              rt.Memcpy2DFromArray(dlinear, (size_t)CW * bpp, level0, 0, 0, (size_t)CW * bpp, CH, cudaMemcpyDeviceToDevice);
              rt.DeviceSynchronize();
            });
            snprintf(what, sizeof what, "%s 1080p: D2D linearize from external-memory array + sync (no draw)", fmt_name(fmt));
            report_stats("S5.T", what, st);
            rt.Memcpy(host.data(), dlinear, host.size(), cudaMemcpyDeviceToHost);
            size_t bad = count_mismatch(host.data(), want.data(), want.size());
            report("S5.5", bad == 0 ? Verdict::Pass : Verdict::Fail,
                   "%s 1080p external-memory array copy matches the GL render: %zu mismatches", fmt_name(fmt), bad);
            rt.FreeMipmappedArray(mm);
          } else {
            report("S5.5", Verdict::Info, "%s external memory mipmapped array: %s", fmt_name(fmt), rt.err(e2));
          }
          rt.DestroyExternalMemory(ext);
        } else {
          report("S5.5", Verdict::Info, "%s cudaImportExternalMemory for timing: %s", fmt_name(fmt), h ? rt.err(e2) : hr_str(hr));
        }
        if (h) CloseHandle(h);
      }

      // ---- S5.6 ordering: GL render then CUDA read, with / without fence --
      if (res && fmt == DXGI_FORMAT_R8G8B8A8_UNORM) {
        std::vector<uint8_t> solid((size_t)CW * CH * 4);
        for (size_t i = 0; i < (size_t)CW * CH; i++) {
          solid[i * 4] = 51;
          solid[i * 4 + 1] = 102;
          solid[i * 4 + 2] = 153;
          solid[i * 4 + 3] = 255;
        }
        cudaStream_t stream = nullptr;
        rt.StreamCreate(&stream);
        ComPtr<ID3D11Fence> fence;
        cudaExternalSemaphore_t sem = nullptr;
        UINT64 fence_value = 0;
        if (s.d3d.dev5 && SUCCEEDED(s.d3d.dev5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence)))) {
          HANDLE fh = nullptr;
          if (SUCCEEDED(fence->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &fh))) {
            cudaExternalSemaphoreHandleDesc sd{};
            sd.type = cudaExternalSemaphoreHandleTypeD3D11Fence;
            sd.handle.win32.handle = fh;
            cudaError_t es = rt.ImportExternalSemaphore(&sem, &sd);
            report("S5.6", es == cudaSuccess ? Verdict::Pass : Verdict::Fail, "cudaImportExternalSemaphore(D3D11Fence): %s", rt.err(es));
            if (es) sem = nullptr;
            CloseHandle(fh);
          }
        }
        for (int mode = 0; mode < 4; mode++) {
          if (mode == 3 && !sem) break;
          int stale = 0, garbage = 0;
          const int N = 200;
          for (int i = 0; i < N; i++) {
            bool gradient = (i & 1) == 0;
            glBindFramebuffer(GL_FRAMEBUFFER, reg.fbo);
            glViewport(0, 0, CW, CH);
            if (gradient) {
              glUseProgram(p_grad);
              glUniform1f(denom, dn);
              quad.draw();
            } else {
              glClearColor(0.2f, 0.4f, 0.6f, 1.0f);
              glClear(GL_COLOR_BUFFER_BIT);
            }
            if (mode == 1 || mode == 3) glFlush();
            if (mode == 2) glFinish();
            if (mode == 3) {
              s.d3d.ctx4->Signal(fence.Get(), ++fence_value);
              cudaExternalSemaphoreWaitParams wp{};
              wp.params.fence.value = fence_value;
              rt.WaitExternalSemaphoresAsync(&sem, &wp, 1, stream);
            }
            rt.GraphicsMapResources(1, &res, stream);
            cudaArray_t arr;
            rt.GraphicsSubResourceGetMappedArray(&arr, res, 0, 0);
            rt.Memcpy2DFromArray(dlinear, (size_t)CW * 4, arr, 0, 0, (size_t)CW * 4, CH, cudaMemcpyDeviceToDevice);
            rt.GraphicsUnmapResources(1, &res, stream);
            rt.MemcpyAsync(host.data(), dlinear, host.size(), cudaMemcpyDeviceToHost, stream);
            rt.StreamSynchronize(stream);
            bool is_grad = count_mismatch(host.data(), want.data(), want.size()) == 0;
            bool is_solid = !is_grad && count_mismatch(host.data(), solid.data(), solid.size()) == 0;
            if (!is_grad && !is_solid) garbage++;
            else if (is_grad != gradient) stale++;
          }
          report("S5.6", (stale == 0 && garbage == 0) ? Verdict::Pass : Verdict::Fail,
                 "GL render + %-28s then CUDA map/copy: %d stale, %d garbage of %d",
                 mode == 0 ? "nothing" : mode == 1 ? "glFlush" : mode == 2 ? "glFinish" : "glFlush + fence -> CUDA wait", stale, garbage, N);
        }
        // Same, through the external-memory array (no map call to order things).
        if (ext_t.fbo) {
          HRESULT hr;
          HANDLE h = create_shared_handle(ext_t.tex.Get(), nullptr, &hr);
          cudaExternalMemoryHandleDesc hd{};
          hd.type = cudaExternalMemoryHandleTypeD3D11Resource;
          hd.handle.win32.handle = h;
          hd.size = (size_t)CW * CH * 4;
          hd.flags = cudaExternalMemoryDedicated;
          cudaExternalMemory_t ext = nullptr;
          cudaMipmappedArray_t mm = nullptr;
          cudaArray_t level0 = nullptr;
          if (h && !rt.ImportExternalMemory(&ext, &hd)) {
            cudaExternalMemoryMipmappedArrayDesc md{};
            md.formatDesc = cudaChannelFormatDesc{8, 8, 8, 8, cudaChannelFormatKindUnsigned};
            md.extent = cudaExtent{CW, CH, 0};
            md.numLevels = 1;
            if (!rt.ExternalMemoryGetMappedMipmappedArray(&mm, ext, &md)) rt.GetMipmappedArrayLevel(&level0, mm, 0);
          }
          for (int mode = 0; level0 && mode < 4; mode++) {
            if (mode == 3 && !sem) break;
            int stale = 0, garbage = 0;
            const int N = 200;
            for (int i = 0; i < N; i++) {
              bool gradient = (i & 1) == 0;
              glBindFramebuffer(GL_FRAMEBUFFER, ext_t.fbo);
              glViewport(0, 0, CW, CH);
              if (gradient) {
                glUseProgram(p_grad);
                glUniform1f(denom, dn);
                quad.draw();
              } else {
                glClearColor(0.2f, 0.4f, 0.6f, 1.0f);
                glClear(GL_COLOR_BUFFER_BIT);
              }
              if (mode == 1 || mode == 3) glFlush();
              if (mode == 2) glFinish();
              if (mode == 3) {
                s.d3d.ctx4->Signal(fence.Get(), ++fence_value);
                cudaExternalSemaphoreWaitParams wp{};
                wp.params.fence.value = fence_value;
                rt.WaitExternalSemaphoresAsync(&sem, &wp, 1, stream);
              }
              rt.MemcpyAsync(dlinear, dlinear, 0, cudaMemcpyDeviceToDevice, stream);  // keep stream ordering explicit
              rt.Memcpy2DFromArray(dlinear, (size_t)CW * 4, level0, 0, 0, (size_t)CW * 4, CH, cudaMemcpyDeviceToDevice);
              rt.MemcpyAsync(host.data(), dlinear, host.size(), cudaMemcpyDeviceToHost, stream);
              rt.StreamSynchronize(stream);
              rt.DeviceSynchronize();
              bool is_grad = count_mismatch(host.data(), want.data(), want.size()) == 0;
              bool is_solid = !is_grad && count_mismatch(host.data(), solid.data(), solid.size()) == 0;
              if (!is_grad && !is_solid) garbage++;
              else if (is_grad != gradient) stale++;
            }
            report("S5.6", Verdict::Info,
                   "external-memory array: GL render + %-28s then CUDA copy: %d stale, %d garbage of %d",
                   mode == 0 ? "nothing" : mode == 1 ? "glFlush" : mode == 2 ? "glFinish" : "glFlush + fence -> CUDA wait", stale, garbage, N);
          }
          if (mm) rt.FreeMipmappedArray(mm);
          if (ext) rt.DestroyExternalMemory(ext);
          if (h) CloseHandle(h);
        }
        if (sem) rt.DestroyExternalSemaphore(sem);
        rt.StreamDestroy(stream);
      }
      if (res) rt.GraphicsUnregisterResource(res);
      rt.Free(dlinear);
      glBindFramebuffer(GL_FRAMEBUFFER, 0);
      ext_t.destroy(s);
      reg.destroy(s);
    }
    if (p_grad) glDeleteProgram(p_grad);
    quad.destroy();
  }

  // ---- S5.7 driver API register + map + linearize -------------------------
  if (have_drv) {
    ComPtr<ID3D11Texture2D> tex = create_tex(s.d3d, W, H, DXGI_FORMAT_R8G8B8A8_UNORM, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0);
    auto pat = make_pattern(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, 17);
    upload_tex(s.d3d, tex.Get(), pat);
    s.d3d.ctx->Flush();
    CUgraphicsResource res = nullptr;
    CUresult r = drv.GraphicsD3D11RegisterResource(&res, tex.Get(), 0);
    if (r) {
      report("S5.7", Verdict::Fail, "cuGraphicsD3D11RegisterResource: %s", drv.err(r));
    } else {
      r = drv.GraphicsMapResources(1, &res, nullptr);
      CUarray arr = nullptr;
      if (!r) r = drv.GraphicsSubResourceGetMappedArray(&arr, res, 0, 0);
      CUDA_ARRAY_DESCRIPTOR ad{};
      if (!r) drv.ArrayGetDescriptor(&ad, arr);
      CUdeviceptr dptr = 0;
      if (!r) r = drv.MemAlloc(&dptr, (size_t)W * H * 4);
      if (!r) {
        CUDA_MEMCPY2D cp{};
        cp.srcMemoryType = CU_MEMORYTYPE_ARRAY;
        cp.srcArray = arr;
        cp.dstMemoryType = CU_MEMORYTYPE_DEVICE;
        cp.dstDevice = dptr;
        cp.dstPitch = W * 4;
        cp.WidthInBytes = W * 4;
        cp.Height = H;
        r = drv.Memcpy2D(&cp);
      }
      std::vector<uint8_t> host((size_t)W * H * 4);
      if (!r) r = drv.MemcpyDtoH(host.data(), dptr, host.size());
      if (dptr) drv.MemFree(dptr);
      drv.GraphicsUnmapResources(1, &res, nullptr);
      drv.GraphicsUnregisterResource(res);
      size_t bad = r ? SIZE_MAX : count_mismatch(host.data(), pat.data(), pat.size());
      report("S5.7", bad == 0 ? Verdict::Pass : Verdict::Fail,
             "driver API: register + map + cuMemcpy2D(array->linear) + DtoH: %s (array fmt %d ch %u %zux%zu)",
             r ? drv.err(r) : (bad ? "bad bytes" : "0 mismatches"), (int)ad.Format, ad.NumChannels, ad.Width, ad.Height);
    }
    if (cu_ctx) drv.PrimaryCtxRelease(cu_dev);
  }
  gl_clear_errors("S5 end");
}
