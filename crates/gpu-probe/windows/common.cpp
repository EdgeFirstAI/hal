// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Reporting, timing, pixel patterns, DXGI adapter selection and D3D11
// texture helpers shared by every probe section.
#include "probe.h"

#include <algorithm>
#include <cstdarg>
#include <cstring>
#include <mutex>

Options g_opt;

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

static int g_pass = 0, g_fail = 0, g_skip = 0;
static std::mutex g_report_mu;

void report(const char* id, Verdict v, const char* fmt, ...) {
  char buf[4096];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof buf, fmt, ap);
  va_end(ap);
  const char* tag = "INFO";
  std::lock_guard<std::mutex> lk(g_report_mu);
  switch (v) {
    case Verdict::Pass: tag = "PASS"; g_pass++; break;
    case Verdict::Fail: tag = "FAIL"; g_fail++; break;
    case Verdict::Skip: tag = "SKIP"; g_skip++; break;
    case Verdict::Info: break;
  }
  printf("[%-6s] %s  %s\n", id, tag, buf);
  fflush(stdout);
}

void timing(const char* id, const char* what, int n, double median_us, double mean_us,
            double min_us) {
  std::lock_guard<std::mutex> lk(g_report_mu);
  printf("[%-6s] TIME  %-58s n=%-4d median=%9.1f us  mean=%9.1f us  min=%9.1f us\n", id, what,
         n, median_us, mean_us, min_us);
  fflush(stdout);
}

int summary() {
  printf("\n== summary: %d PASS, %d FAIL, %d SKIP ==\n", g_pass, g_fail, g_skip);
  return g_fail ? 1 : 0;
}

bool section_enabled(const char* id) {
  if (g_opt.only.empty()) return true;
  for (const auto& o : g_opt.only) {
    if (_stricmp(o.c_str(), id) == 0) return true;
  }
  return false;
}

const char* hr_str(HRESULT hr) {
  thread_local char buf[96];
  const char* name = nullptr;
  switch ((uint32_t)hr) {
    case 0x00000000: name = "S_OK"; break;
    case 0x80070057: name = "E_INVALIDARG"; break;
    case 0x80004002: name = "E_NOINTERFACE"; break;
    case 0x80004005: name = "E_FAIL"; break;
    case 0x8007000E: name = "E_OUTOFMEMORY"; break;
    case 0x80070005: name = "E_ACCESSDENIED"; break;
    case 0x80004001: name = "E_NOTIMPL"; break;
    case 0x887A0001: name = "DXGI_ERROR_INVALID_CALL"; break;
    case 0x887A0002: name = "DXGI_ERROR_NOT_FOUND"; break;
    case 0x887A0004: name = "DXGI_ERROR_UNSUPPORTED"; break;
    case 0x887A0005: name = "DXGI_ERROR_DEVICE_REMOVED"; break;
    case 0x887A0006: name = "DXGI_ERROR_DEVICE_HUNG"; break;
    case 0x887A0007: name = "DXGI_ERROR_DEVICE_RESET"; break;
    case 0x887A000A: name = "DXGI_ERROR_WAS_STILL_DRAWING"; break;
    case 0x887A0020: name = "DXGI_ERROR_DRIVER_INTERNAL_ERROR"; break;
    case 0x887A0022: name = "DXGI_ERROR_NAME_ALREADY_EXISTS"; break;
    default: break;
  }
  if (name)
    snprintf(buf, sizeof buf, "%s (0x%08lx)", name, (unsigned long)hr);
  else
    snprintf(buf, sizeof buf, "0x%08lx", (unsigned long)hr);
  return buf;
}

const char* egl_err_str(EGLint e) {
  switch (e) {
    case EGL_SUCCESS: return "EGL_SUCCESS";
    case EGL_NOT_INITIALIZED: return "EGL_NOT_INITIALIZED";
    case EGL_BAD_ACCESS: return "EGL_BAD_ACCESS";
    case EGL_BAD_ALLOC: return "EGL_BAD_ALLOC";
    case EGL_BAD_ATTRIBUTE: return "EGL_BAD_ATTRIBUTE";
    case EGL_BAD_CONFIG: return "EGL_BAD_CONFIG";
    case EGL_BAD_CONTEXT: return "EGL_BAD_CONTEXT";
    case EGL_BAD_CURRENT_SURFACE: return "EGL_BAD_CURRENT_SURFACE";
    case EGL_BAD_DISPLAY: return "EGL_BAD_DISPLAY";
    case EGL_BAD_MATCH: return "EGL_BAD_MATCH";
    case EGL_BAD_NATIVE_PIXMAP: return "EGL_BAD_NATIVE_PIXMAP";
    case EGL_BAD_NATIVE_WINDOW: return "EGL_BAD_NATIVE_WINDOW";
    case EGL_BAD_PARAMETER: return "EGL_BAD_PARAMETER";
    case EGL_BAD_SURFACE: return "EGL_BAD_SURFACE";
    case EGL_CONTEXT_LOST: return "EGL_CONTEXT_LOST";
    default: {
      thread_local char buf[32];
      snprintf(buf, sizeof buf, "0x%04x", e);
      return buf;
    }
  }
}

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

double now_us() {
  static LARGE_INTEGER freq = [] {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    return f;
  }();
  LARGE_INTEGER c;
  QueryPerformanceCounter(&c);
  return (double)c.QuadPart * 1e6 / (double)freq.QuadPart;
}

Stats time_loop(int iters, const std::function<void()>& body) {
  for (int i = 0; i < 3; i++) body();
  std::vector<double> samples;
  samples.reserve(iters);
  for (int i = 0; i < iters; i++) {
    double t0 = now_us();
    body();
    samples.push_back(now_us() - t0);
  }
  std::sort(samples.begin(), samples.end());
  Stats s;
  s.n = iters;
  s.min_us = samples.front();
  s.max_us = samples.back();
  s.median_us = samples[samples.size() / 2];
  double sum = 0;
  for (double v : samples) sum += v;
  s.mean_us = sum / samples.size();
  return s;
}

void report_stats(const char* id, const char* what, const Stats& s) {
  timing(id, what, s.n, s.median_us, s.mean_us, s.min_us);
}

// ---------------------------------------------------------------------------
// Pixel patterns
// ---------------------------------------------------------------------------

uint32_t bytes_per_pixel(DXGI_FORMAT fmt) {
  switch (fmt) {
    case DXGI_FORMAT_R8_UNORM: return 1;
    case DXGI_FORMAT_R8G8_UNORM: return 2;
    case DXGI_FORMAT_R16_UNORM: return 2;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_B8G8R8A8_UNORM:
    case DXGI_FORMAT_R10G10B10A2_UNORM:
    case DXGI_FORMAT_R16G16_UNORM: return 4;
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return 8;
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return 16;
    case DXGI_FORMAT_NV12: return 1;  // per Y byte; callers use h * 3 / 2 rows
    default: return 0;
  }
}

const char* fmt_name(DXGI_FORMAT fmt) {
  switch (fmt) {
    case DXGI_FORMAT_R8_UNORM: return "R8_UNORM";
    case DXGI_FORMAT_R8G8_UNORM: return "R8G8_UNORM";
    case DXGI_FORMAT_R16_UNORM: return "R16_UNORM";
    case DXGI_FORMAT_R16G16_UNORM: return "R16G16_UNORM";
    case DXGI_FORMAT_R8G8B8A8_UNORM: return "R8G8B8A8_UNORM";
    case DXGI_FORMAT_B8G8R8A8_UNORM: return "B8G8R8A8_UNORM";
    case DXGI_FORMAT_R10G10B10A2_UNORM: return "R10G10B10A2_UNORM";
    case DXGI_FORMAT_R16G16B16A16_FLOAT: return "R16G16B16A16_FLOAT";
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return "R32G32B32A32_FLOAT";
    case DXGI_FORMAT_NV12: return "NV12";
    default: return "?";
  }
}

uint16_t f32_to_f16(float f) {
  uint32_t x;
  memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffffu;
  if ((x & 0x7fffffffu) == 0) return (uint16_t)sign;
  if (exp <= 0) {
    // Subnormal half: shift the mantissa (with the implicit 1) right.
    if (exp < -10) return (uint16_t)sign;
    mant |= 0x800000u;
    uint32_t shift = (uint32_t)(14 - exp);
    uint32_t hm = mant >> shift;
    if ((mant >> (shift - 1)) & 1u) hm++;
    return (uint16_t)(sign | hm);
  }
  if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
  uint32_t hm = mant >> 13;
  if (mant & 0x1000u) {
    hm++;
    if (hm == 0x400u) {
      hm = 0;
      exp++;
      if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
    }
  }
  return (uint16_t)(sign | ((uint32_t)exp << 10) | hm);
}

float f16_to_f32(uint16_t h) {
  uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1f;
  uint32_t mant = h & 0x3ffu;
  uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign;
    } else {
      int e = -1;
      do {
        e++;
        mant <<= 1;
      } while ((mant & 0x400u) == 0);
      mant &= 0x3ffu;
      out = sign | ((uint32_t)(127 - 15 - e) << 23) | (mant << 13);
    }
  } else if (exp == 0x1f) {
    out = sign | 0x7f800000u | (mant << 13);
  } else {
    out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  memcpy(&f, &out, 4);
  return f;
}

// Logical channel count and storage order for the UNORM8 / 16-bit formats.
static int channel_count(DXGI_FORMAT fmt) {
  switch (fmt) {
    case DXGI_FORMAT_R8_UNORM:
    case DXGI_FORMAT_R16_UNORM: return 1;
    case DXGI_FORMAT_R8G8_UNORM:
    case DXGI_FORMAT_R16G16_UNORM: return 2;
    default: return 4;
  }
}

static void store_pixel(DXGI_FORMAT fmt, uint8_t* p, const uint32_t k[4]) {
  switch (fmt) {
    case DXGI_FORMAT_R8_UNORM: p[0] = (uint8_t)k[0]; break;
    case DXGI_FORMAT_R8G8_UNORM:
      p[0] = (uint8_t)k[0];
      p[1] = (uint8_t)k[1];
      break;
    case DXGI_FORMAT_R8G8B8A8_UNORM:
      for (int c = 0; c < 4; c++) p[c] = (uint8_t)k[c];
      break;
    case DXGI_FORMAT_B8G8R8A8_UNORM:
      p[0] = (uint8_t)k[2];
      p[1] = (uint8_t)k[1];
      p[2] = (uint8_t)k[0];
      p[3] = (uint8_t)k[3];
      break;
    case DXGI_FORMAT_R16_UNORM: {
      uint16_t v = (uint16_t)(k[0] * 257);
      memcpy(p, &v, 2);
      break;
    }
    case DXGI_FORMAT_R16G16_UNORM: {
      uint16_t v[2] = {(uint16_t)(k[0] * 257), (uint16_t)(k[1] * 257)};
      memcpy(p, v, 4);
      break;
    }
    case DXGI_FORMAT_R16G16B16A16_FLOAT: {
      uint16_t v[4];
      for (int c = 0; c < 4; c++) v[c] = f32_to_f16((float)k[c] / 256.0f);
      memcpy(p, v, 8);
      break;
    }
    case DXGI_FORMAT_R32G32B32A32_FLOAT: {
      float v[4];
      for (int c = 0; c < 4; c++) v[c] = (float)k[c] / 256.0f;
      memcpy(p, v, 16);
      break;
    }
    default: break;
  }
}

std::vector<uint8_t> make_pattern(DXGI_FORMAT fmt, uint32_t w, uint32_t h, uint32_t seed) {
  std::vector<uint8_t> out;
  if (fmt == DXGI_FORMAT_NV12) {
    out.resize((size_t)w * h * 3 / 2);
    for (uint32_t y = 0; y < h; y++)
      for (uint32_t x = 0; x < w; x++) out[(size_t)y * w + x] = (uint8_t)pattern_k(x, y, 0, seed);
    uint8_t* uv = out.data() + (size_t)w * h;
    for (uint32_t cy = 0; cy < h / 2; cy++)
      for (uint32_t cx = 0; cx < w / 2; cx++) {
        uv[(size_t)cy * w + cx * 2] = (uint8_t)pattern_k(cx, cy, 1, seed);
        uv[(size_t)cy * w + cx * 2 + 1] = (uint8_t)pattern_k(cx, cy, 2, seed);
      }
    return out;
  }
  uint32_t bpp = bytes_per_pixel(fmt);
  out.resize((size_t)w * h * bpp);
  for (uint32_t y = 0; y < h; y++)
    for (uint32_t x = 0; x < w; x++) {
      uint32_t k[4];
      for (uint32_t c = 0; c < 4; c++) k[c] = pattern_k(x, y, c, seed);
      store_pixel(fmt, out.data() + ((size_t)y * w + x) * bpp, k);
    }
  return out;
}

std::vector<float> pattern_as_rgba_f32(DXGI_FORMAT fmt, uint32_t w, uint32_t h, uint32_t seed) {
  std::vector<float> out((size_t)w * h * 4);
  int nc = channel_count(fmt);
  bool is_float =
      fmt == DXGI_FORMAT_R16G16B16A16_FLOAT || fmt == DXGI_FORMAT_R32G32B32A32_FLOAT;
  for (uint32_t y = 0; y < h; y++)
    for (uint32_t x = 0; x < w; x++) {
      float* p = out.data() + ((size_t)y * w + x) * 4;
      for (int c = 0; c < 4; c++) {
        if (c < nc) {
          uint32_t k = pattern_k(x, y, (uint32_t)c, seed);
          p[c] = is_float ? (float)k / 256.0f : (float)k / 255.0f;
        } else {
          p[c] = (c == 3) ? 1.0f : 0.0f;
        }
      }
    }
  return out;
}

std::vector<uint8_t> make_gradient(DXGI_FORMAT fmt, uint32_t w, uint32_t h, bool flip_y) {
  uint32_t bpp = bytes_per_pixel(fmt);
  std::vector<uint8_t> out((size_t)w * h * bpp);
  for (uint32_t y = 0; y < h; y++) {
    uint32_t yy = flip_y ? (h - 1 - y) : y;
    for (uint32_t x = 0; x < w; x++) {
      uint32_t k[4] = {x & 255, yy & 255, (x + yy) & 255, 255};
      store_pixel(fmt, out.data() + ((size_t)y * w + x) * bpp, k);
    }
  }
  return out;
}

size_t count_mismatch(const uint8_t* a, const uint8_t* b, size_t n, size_t* first) {
  size_t bad = 0;
  if (first) *first = SIZE_MAX;
  for (size_t i = 0; i < n; i++) {
    if (a[i] != b[i]) {
      if (first && *first == SIZE_MAX) *first = i;
      bad++;
    }
  }
  return bad;
}

size_t count_mismatch_f32(const float* a, const float* b, size_t n, float tol, size_t* first) {
  size_t bad = 0;
  if (first) *first = SIZE_MAX;
  for (size_t i = 0; i < n; i++) {
    float d = a[i] - b[i];
    if (d < 0) d = -d;
    if (!(d <= tol)) {
      if (first && *first == SIZE_MAX) *first = i;
      bad++;
    }
  }
  return bad;
}

// ---------------------------------------------------------------------------
// DXGI / D3D11
// ---------------------------------------------------------------------------

std::string wide_to_utf8(const wchar_t* w) {
  int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
  std::string s(n > 0 ? n - 1 : 0, '\0');
  if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
  return s;
}

std::string luid_str(LUID l) {
  char buf[32];
  snprintf(buf, sizeof buf, "%#lx:%#lx", (long)l.HighPart, (unsigned long)l.LowPart);
  return buf;
}

std::vector<AdapterInfo> enumerate_adapters() {
  std::vector<AdapterInfo> out;
  ComPtr<IDXGIFactory1> factory;
  if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return out;
  for (UINT i = 0;; i++) {
    AdapterInfo a;
    if (factory->EnumAdapters1(i, &a.adapter) == DXGI_ERROR_NOT_FOUND) break;
    a.adapter->GetDesc1(&a.desc);
    out.push_back(a);
  }
  return out;
}

bool pick_adapter(const std::string& sel, bool warp, AdapterInfo* out) {
  if (warp) {
    ComPtr<IDXGIFactory4> f4;
    if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&f4)))) return false;
    if (FAILED(f4->EnumWarpAdapter(IID_PPV_ARGS(&out->adapter)))) return false;
    ComPtr<IDXGIAdapter1> a1;
    out->adapter.As(&a1);
    if (a1) a1->GetDesc1(&out->desc);
    return true;
  }
  auto all = enumerate_adapters();
  if (!sel.empty()) {
    long high = 0;
    unsigned long low = 0;
    if (sscanf_s(sel.c_str(), "%li:%lu", &high, &low) != 2 &&
        sscanf_s(sel.c_str(), "%lx:%lx", &high, &low) != 2)
      return false;
    for (auto& a : all)
      if (a.desc.AdapterLuid.HighPart == high && a.desc.AdapterLuid.LowPart == low) {
        *out = a;
        return true;
      }
    return false;
  }
  for (auto& a : all)
    if (!(a.desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
      *out = a;
      return true;
    }
  return false;
}

bool D3D::wrap(ID3D11Device* device) {
  dev = device;
  dev.As(&dev1);
  dev.As(&dev5);
  dev->GetImmediateContext(&ctx);
  ctx.As(&ctx4);
  feature_level = dev->GetFeatureLevel();
  creation_flags = dev->GetCreationFlags();
  ComPtr<IDXGIDevice> dxgi;
  if (SUCCEEDED(dev.As(&dxgi))) {
    ComPtr<IDXGIAdapter> ad;
    if (SUCCEEDED(dxgi->GetAdapter(&ad))) {
      DXGI_ADAPTER_DESC d;
      ad->GetDesc(&d);
      luid = d.AdapterLuid;
    }
  }
  return dev && ctx;
}

bool create_device(const AdapterInfo* adapter, bool warp, UINT flags, D3D* out, HRESULT* hr) {
  static const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_12_1, D3D_FEATURE_LEVEL_12_0,
                                             D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
  ComPtr<ID3D11Device> dev;
  ComPtr<ID3D11DeviceContext> ctx;
  D3D_FEATURE_LEVEL fl;
  IDXGIAdapter* a = (adapter && !warp) ? adapter->adapter.Get() : nullptr;
  D3D_DRIVER_TYPE type =
      warp ? D3D_DRIVER_TYPE_WARP : (a ? D3D_DRIVER_TYPE_UNKNOWN : D3D_DRIVER_TYPE_HARDWARE);
  HRESULT r = D3D11CreateDevice(a, type, nullptr, flags, levels, ARRAYSIZE(levels),
                                D3D11_SDK_VERSION, &dev, &fl, &ctx);
  if (hr) *hr = r;
  if (FAILED(r)) return false;
  return out->wrap(dev.Get());
}

ComPtr<ID3D11Texture2D> create_tex(D3D& d, uint32_t w, uint32_t h, DXGI_FORMAT fmt, UINT bind,
                                   UINT misc, HRESULT* hr_out, D3D11_USAGE usage, UINT cpu) {
  D3D11_TEXTURE2D_DESC desc{};
  desc.Width = w;
  desc.Height = h;
  desc.MipLevels = 1;
  desc.ArraySize = 1;
  desc.Format = fmt;
  desc.SampleDesc.Count = 1;
  desc.Usage = usage;
  desc.BindFlags = bind;
  desc.CPUAccessFlags = cpu;
  desc.MiscFlags = misc;
  ComPtr<ID3D11Texture2D> tex;
  HRESULT hr = d.dev->CreateTexture2D(&desc, nullptr, &tex);
  if (hr_out) *hr_out = hr;
  return tex;
}

ComPtr<ID3D11Texture2D> create_staging_like(D3D& d, ID3D11Texture2D* tex, UINT cpu_access) {
  D3D11_TEXTURE2D_DESC desc;
  tex->GetDesc(&desc);
  desc.Usage = D3D11_USAGE_STAGING;
  desc.BindFlags = 0;
  desc.MiscFlags = 0;
  desc.CPUAccessFlags = cpu_access;
  ComPtr<ID3D11Texture2D> st;
  d.dev->CreateTexture2D(&desc, nullptr, &st);
  return st;
}

static void tight_geometry(const D3D11_TEXTURE2D_DESC& desc, uint32_t* rows, uint32_t* row_bytes) {
  if (desc.Format == DXGI_FORMAT_NV12) {
    *rows = desc.Height + desc.Height / 2;
    *row_bytes = desc.Width;
  } else {
    *rows = desc.Height;
    *row_bytes = desc.Width * bytes_per_pixel(desc.Format);
  }
}

static thread_local std::vector<ID3D11Resource*> g_keyed_held;

KeyedLock::KeyedLock(ID3D11Resource* r) : res(r) {
  for (ID3D11Resource* h : g_keyed_held)
    if (h == r) return;  // already held by an enclosing scope
  if (SUCCEEDED(r->QueryInterface(IID_PPV_ARGS(&km)))) {
    held = SUCCEEDED(km->AcquireSync(0, 2000));
    if (held) g_keyed_held.push_back(r);
  }
}

KeyedLock::~KeyedLock() {
  if (!held) return;
  km->ReleaseSync(0);
  for (size_t i = g_keyed_held.size(); i-- > 0;)
    if (g_keyed_held[i] == res) {
      g_keyed_held.erase(g_keyed_held.begin() + (ptrdiff_t)i);
      break;
    }
}

bool upload_tex(D3D& d, ID3D11Texture2D* tex, const std::vector<uint8_t>& tight,
                bool via_update_subresource) {
  D3D11_TEXTURE2D_DESC desc;
  tex->GetDesc(&desc);
  uint32_t rows, row_bytes;
  tight_geometry(desc, &rows, &row_bytes);
  if (tight.size() < (size_t)rows * row_bytes) return false;
  KeyedLock lock(tex);
  if (via_update_subresource) {
    d.ctx->UpdateSubresource(tex, 0, nullptr, tight.data(), row_bytes, 0);
    return true;
  }
  ComPtr<ID3D11Texture2D> st = create_staging_like(d, tex, D3D11_CPU_ACCESS_WRITE);
  if (!st) return false;
  D3D11_MAPPED_SUBRESOURCE m;
  if (FAILED(d.ctx->Map(st.Get(), 0, D3D11_MAP_WRITE, 0, &m))) return false;
  // For NV12 the UV plane follows the Y plane at RowPitch * Height.
  for (uint32_t r = 0; r < rows; r++)
    memcpy((uint8_t*)m.pData + (size_t)r * m.RowPitch, tight.data() + (size_t)r * row_bytes,
           row_bytes);
  d.ctx->Unmap(st.Get(), 0);
  d.ctx->CopyResource(tex, st.Get());
  return true;
}

bool readback_tex(D3D& d, ID3D11Texture2D* tex, std::vector<uint8_t>& tight, UINT* row_pitch,
                  ID3D11Texture2D* staging) {
  D3D11_TEXTURE2D_DESC desc;
  tex->GetDesc(&desc);
  uint32_t rows, row_bytes;
  tight_geometry(desc, &rows, &row_bytes);
  ComPtr<ID3D11Texture2D> owned;
  if (!staging) {
    owned = create_staging_like(d, tex, D3D11_CPU_ACCESS_READ);
    staging = owned.Get();
    if (!staging) return false;
  }
  KeyedLock lock(tex);
  d.ctx->CopyResource(staging, tex);
  D3D11_MAPPED_SUBRESOURCE m;
  HRESULT hr = d.ctx->Map(staging, 0, D3D11_MAP_READ, 0, &m);
  if (FAILED(hr)) return false;
  if (row_pitch) *row_pitch = m.RowPitch;
  tight.resize((size_t)rows * row_bytes);
  for (uint32_t r = 0; r < rows; r++)
    memcpy(tight.data() + (size_t)r * row_bytes, (const uint8_t*)m.pData + (size_t)r * m.RowPitch,
           row_bytes);
  d.ctx->Unmap(staging, 0);
  return true;
}

DWORD spawn_self(const std::wstring& extra_args, DWORD timeout_ms) {
  wchar_t exe[MAX_PATH];
  GetModuleFileNameW(nullptr, exe, MAX_PATH);
  std::wstring cmd = L"\"" + std::wstring(exe) + L"\" " + extra_args;
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
  std::vector<wchar_t> buf(cmd.begin(), cmd.end());
  buf.push_back(0);
  if (!CreateProcessW(nullptr, buf.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi)) return 0xFFFFFFFF;
  DWORD w = WaitForSingleObject(pi.hProcess, timeout_ms);
  DWORD code = 0xFFFFFFFF;
  if (w == WAIT_OBJECT_0) GetExitCodeProcess(pi.hProcess, &code);
  else TerminateProcess(pi.hProcess, 0xFFFFFFFE);
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  return code;
}

HANDLE create_shared_handle(ID3D11Texture2D* tex, const wchar_t* name, HRESULT* hr) {
  ComPtr<IDXGIResource1> res;
  HRESULT r = tex->QueryInterface(IID_PPV_ARGS(&res));
  if (FAILED(r)) {
    if (hr) *hr = r;
    return nullptr;
  }
  HANDLE h = nullptr;
  r = res->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                              name, &h);
  if (hr) *hr = r;
  return SUCCEEDED(r) ? h : nullptr;
}
