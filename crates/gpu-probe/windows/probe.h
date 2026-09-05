// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Windows D3D11 texture / ANGLE / CUDA / D3D12 interop probe: shared
// declarations. The probe answers the spike questions in
// docs/superpowers/specs/2026-09-02-windows-d3d11-exploration-plan.md
// (S0-S9) with PASS / FAIL / SKIP lines and timing tables. It links only
// the inbox D3D libraries; ANGLE, CUDA and DirectML are loaded at runtime
// so a box without them reports SKIP instead of failing to start.
#pragma once

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11_4.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#define EGL_EGL_PROTOTYPES 0
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <EGL/eglext_angle.h>
#define GL_GLES_PROTOTYPES 0
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

// ---------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------

struct Options {
  bool warp = false;          // ANGLE on the D3D11 WARP adapter
  bool debug_layer = false;   // D3D11 debug layer on injected devices
  std::string angle_dir;      // directory holding libEGL.dll / libGLESv2.dll
  std::string adapter;        // "" (default hardware) or "<high>:<low>" LUID
  std::vector<std::string> only;  // section ids to run ("s1", "s5"); empty = all
  int iters = 100;            // timing iterations
  std::string child_name;     // S7: run as the child that opens a named handle
  std::string child_fence;    // S7: name of the shared fence
  std::string child_kmt;      // S7: legacy (KMT) shared handle value, hex
  bool s4_control = false;    // S4: run only the no-re-sync control (child process)
};
extern Options g_opt;

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

enum class Verdict { Pass, Fail, Skip, Info };
void report(const char* id, Verdict v, const char* fmt, ...);
void timing(const char* id, const char* what, int n, double median_us, double mean_us,
            double min_us);
int summary();  // prints totals; nonzero when any FAIL was reported
bool section_enabled(const char* id);
const char* hr_str(HRESULT hr);
const char* egl_err_str(EGLint e);

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

double now_us();
struct Stats {
  double median_us = 0, mean_us = 0, min_us = 0, max_us = 0;
  int n = 0;
};
Stats time_loop(int iters, const std::function<void()>& body);
void report_stats(const char* id, const char* what, const Stats& s);

// ---------------------------------------------------------------------------
// Pixel patterns
// ---------------------------------------------------------------------------

uint32_t bytes_per_pixel(DXGI_FORMAT fmt);
const char* fmt_name(DXGI_FORMAT fmt);
uint16_t f32_to_f16(float f);
float f16_to_f32(uint16_t h);
// Deterministic per-channel value for pixel (x, y), channel c: 0..255.
inline uint32_t pattern_k(uint32_t x, uint32_t y, uint32_t c, uint32_t seed) {
  return (x * 3 + y * 5 + c * 17 + seed) & 0xFF;
}
// Tight buffer for `fmt` at (w, h) holding pattern_k values (UNORM8 as
// bytes, 16F/32F as k/256).
std::vector<uint8_t> make_pattern(DXGI_FORMAT fmt, uint32_t w, uint32_t h, uint32_t seed = 0);
// The pattern as GL would sample it into an RGBA float framebuffer.
std::vector<float> pattern_as_rgba_f32(DXGI_FORMAT fmt, uint32_t w, uint32_t h, uint32_t seed = 0);
// Gradient written by the probe's fragment shader: R = x & 255, G = y & 255,
// B = (x + y) & 255, A = 255 (UNORM8 as bytes, floats as k/256).
std::vector<uint8_t> make_gradient(DXGI_FORMAT fmt, uint32_t w, uint32_t h, bool flip_y);
// Number of differing bytes; `first` receives the first differing offset.
size_t count_mismatch(const uint8_t* a, const uint8_t* b, size_t n, size_t* first = nullptr);
// Tolerant float compare (for 16F round trips); returns count over `tol`.
size_t count_mismatch_f32(const float* a, const float* b, size_t n, float tol,
                          size_t* first = nullptr);

// ---------------------------------------------------------------------------
// DXGI / D3D11
// ---------------------------------------------------------------------------

struct AdapterInfo {
  ComPtr<IDXGIAdapter1> adapter;
  DXGI_ADAPTER_DESC1 desc{};
};
std::vector<AdapterInfo> enumerate_adapters();
// Resolve the option string ("" = first non-software adapter, "<high>:<low>")
// or WARP.
bool pick_adapter(const std::string& sel, bool warp, AdapterInfo* out);
std::string luid_str(LUID l);
std::string wide_to_utf8(const wchar_t* w);

struct D3D {
  ComPtr<ID3D11Device> dev;
  ComPtr<ID3D11Device1> dev1;
  ComPtr<ID3D11Device5> dev5;
  ComPtr<ID3D11DeviceContext> ctx;
  ComPtr<ID3D11DeviceContext4> ctx4;
  LUID luid{};
  D3D_FEATURE_LEVEL feature_level{};
  UINT creation_flags = 0;
  // Wrap an existing device (ANGLE's, or one created by create_device).
  bool wrap(ID3D11Device* device);
};
// HAL-style device creation for injection and for the second-device tests.
bool create_device(const AdapterInfo* adapter, bool warp, UINT flags, D3D* out, HRESULT* hr);

ComPtr<ID3D11Texture2D> create_tex(D3D& d, uint32_t w, uint32_t h, DXGI_FORMAT fmt, UINT bind,
                                   UINT misc, HRESULT* hr_out = nullptr,
                                   D3D11_USAGE usage = D3D11_USAGE_DEFAULT, UINT cpu = 0);
ComPtr<ID3D11Texture2D> create_staging_like(D3D& d, ID3D11Texture2D* tex, UINT cpu_access);
// Tight <-> texture through a staging copy on the immediate context. For
// NV12 the tight layout is Y plane then interleaved UV plane.
bool upload_tex(D3D& d, ID3D11Texture2D* tex, const std::vector<uint8_t>& tight,
                bool via_update_subresource = false);
bool readback_tex(D3D& d, ID3D11Texture2D* tex, std::vector<uint8_t>& tight,
                  UINT* row_pitch = nullptr, ID3D11Texture2D* staging = nullptr);
HANDLE create_shared_handle(ID3D11Texture2D* tex, const wchar_t* name, HRESULT* hr);

// Holds key 0 of a D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX texture for the
// scope. Commands issued while nobody holds the mutex are dropped by the
// runtime, and ANGLE does not acquire it on the caller's behalf, so the
// owner must hold it across upload, GL work and readback alike. Nested
// locks on the same resource from one thread are no-ops.
struct KeyedLock {
  ComPtr<IDXGIKeyedMutex> km;
  ID3D11Resource* res = nullptr;
  bool held = false;
  explicit KeyedLock(ID3D11Resource* r);
  ~KeyedLock();
};

// ---------------------------------------------------------------------------
// EGL / GL entry points (loaded from ANGLE's DLLs at runtime)
// ---------------------------------------------------------------------------

#define EGL_CORE_FUNCS(X)                                        \
  X(PFNEGLGETPROCADDRESSPROC, eglGetProcAddress)                 \
  X(PFNEGLGETERRORPROC, eglGetError)                             \
  X(PFNEGLINITIALIZEPROC, eglInitialize)                         \
  X(PFNEGLTERMINATEPROC, eglTerminate)                           \
  X(PFNEGLQUERYSTRINGPROC, eglQueryString)                       \
  X(PFNEGLCHOOSECONFIGPROC, eglChooseConfig)                     \
  X(PFNEGLGETCONFIGATTRIBPROC, eglGetConfigAttrib)               \
  X(PFNEGLCREATECONTEXTPROC, eglCreateContext)                   \
  X(PFNEGLDESTROYCONTEXTPROC, eglDestroyContext)                 \
  X(PFNEGLCREATEPBUFFERSURFACEPROC, eglCreatePbufferSurface)     \
  X(PFNEGLDESTROYSURFACEPROC, eglDestroySurface)                 \
  X(PFNEGLMAKECURRENTPROC, eglMakeCurrent)                       \
  X(PFNEGLBINDAPIPROC, eglBindAPI)                               \
  X(PFNEGLBINDTEXIMAGEPROC, eglBindTexImage)                     \
  X(PFNEGLRELEASETEXIMAGEPROC, eglReleaseTexImage)               \
  X(PFNEGLQUERYSURFACEPROC, eglQuerySurface)                     \
  X(PFNEGLWAITCLIENTPROC, eglWaitClient)                         \
  X(PFNEGLCREATEPBUFFERFROMCLIENTBUFFERPROC, eglCreatePbufferFromClientBuffer)

#define EGL_EXT_FUNCS(X)                                                        \
  X(PFNEGLGETPLATFORMDISPLAYEXTPROC, eglGetPlatformDisplayEXT)                  \
  X(PFNEGLQUERYDISPLAYATTRIBEXTPROC, eglQueryDisplayAttribEXT)                  \
  X(PFNEGLQUERYDEVICEATTRIBEXTPROC, eglQueryDeviceAttribEXT)                    \
  X(PFNEGLQUERYDEVICESTRINGEXTPROC, eglQueryDeviceStringEXT)                    \
  X(PFNEGLCREATEDEVICEANGLEPROC, eglCreateDeviceANGLE)                          \
  X(PFNEGLRELEASEDEVICEANGLEPROC, eglReleaseDeviceANGLE)                        \
  X(PFNEGLCREATEIMAGEKHRPROC, eglCreateImageKHR)                                \
  X(PFNEGLDESTROYIMAGEKHRPROC, eglDestroyImageKHR)                              \
  X(PFNEGLQUERYSURFACEPOINTERANGLEPROC, eglQuerySurfacePointerANGLE)            \
  X(PFNEGLCREATESTREAMKHRPROC, eglCreateStreamKHR)                              \
  X(PFNEGLDESTROYSTREAMKHRPROC, eglDestroyStreamKHR)                            \
  X(PFNEGLSTREAMCONSUMERGLTEXTUREEXTERNALATTRIBSNVPROC,                         \
    eglStreamConsumerGLTextureExternalAttribsNV)                                \
  X(PFNEGLCREATESTREAMPRODUCERD3DTEXTUREANGLEPROC, eglCreateStreamProducerD3DTextureANGLE) \
  X(PFNEGLSTREAMPOSTD3DTEXTUREANGLEPROC, eglStreamPostD3DTextureANGLE)          \
  X(PFNEGLSTREAMCONSUMERACQUIREKHRPROC, eglStreamConsumerAcquireKHR)            \
  X(PFNEGLSTREAMCONSUMERRELEASEKHRPROC, eglStreamConsumerReleaseKHR)

#define GL_FUNCS(X)                                                    \
  X(PFNGLGETSTRINGPROC, glGetString)                                   \
  X(PFNGLGETERRORPROC, glGetError)                                     \
  X(PFNGLGETINTEGERVPROC, glGetIntegerv)                               \
  X(PFNGLGENTEXTURESPROC, glGenTextures)                               \
  X(PFNGLDELETETEXTURESPROC, glDeleteTextures)                         \
  X(PFNGLBINDTEXTUREPROC, glBindTexture)                               \
  X(PFNGLACTIVETEXTUREPROC, glActiveTexture)                           \
  X(PFNGLTEXIMAGE2DPROC, glTexImage2D)                                 \
  X(PFNGLTEXSTORAGE2DPROC, glTexStorage2D)                             \
  X(PFNGLTEXPARAMETERIPROC, glTexParameteri)                           \
  X(PFNGLGENFRAMEBUFFERSPROC, glGenFramebuffers)                       \
  X(PFNGLDELETEFRAMEBUFFERSPROC, glDeleteFramebuffers)                 \
  X(PFNGLBINDFRAMEBUFFERPROC, glBindFramebuffer)                       \
  X(PFNGLFRAMEBUFFERTEXTURE2DPROC, glFramebufferTexture2D)             \
  X(PFNGLFRAMEBUFFERRENDERBUFFERPROC, glFramebufferRenderbuffer)       \
  X(PFNGLCHECKFRAMEBUFFERSTATUSPROC, glCheckFramebufferStatus)         \
  X(PFNGLGENRENDERBUFFERSPROC, glGenRenderbuffers)                     \
  X(PFNGLDELETERENDERBUFFERSPROC, glDeleteRenderbuffers)               \
  X(PFNGLBINDRENDERBUFFERPROC, glBindRenderbuffer)                     \
  X(PFNGLVIEWPORTPROC, glViewport)                                     \
  X(PFNGLCLEARCOLORPROC, glClearColor)                                 \
  X(PFNGLCLEARPROC, glClear)                                           \
  X(PFNGLREADPIXELSPROC, glReadPixels)                                 \
  X(PFNGLREADBUFFERPROC, glReadBuffer)                                 \
  X(PFNGLPIXELSTOREIPROC, glPixelStorei)                               \
  X(PFNGLCREATESHADERPROC, glCreateShader)                             \
  X(PFNGLDELETESHADERPROC, glDeleteShader)                             \
  X(PFNGLSHADERSOURCEPROC, glShaderSource)                             \
  X(PFNGLCOMPILESHADERPROC, glCompileShader)                           \
  X(PFNGLGETSHADERIVPROC, glGetShaderiv)                               \
  X(PFNGLGETSHADERINFOLOGPROC, glGetShaderInfoLog)                     \
  X(PFNGLCREATEPROGRAMPROC, glCreateProgram)                           \
  X(PFNGLDELETEPROGRAMPROC, glDeleteProgram)                           \
  X(PFNGLATTACHSHADERPROC, glAttachShader)                             \
  X(PFNGLLINKPROGRAMPROC, glLinkProgram)                               \
  X(PFNGLGETPROGRAMIVPROC, glGetProgramiv)                             \
  X(PFNGLGETPROGRAMINFOLOGPROC, glGetProgramInfoLog)                   \
  X(PFNGLUSEPROGRAMPROC, glUseProgram)                                 \
  X(PFNGLGETUNIFORMLOCATIONPROC, glGetUniformLocation)                 \
  X(PFNGLUNIFORM1IPROC, glUniform1i)                                   \
  X(PFNGLUNIFORM1FPROC, glUniform1f)                                   \
  X(PFNGLUNIFORM2FPROC, glUniform2f)                                   \
  X(PFNGLUNIFORM2IPROC, glUniform2i)                                   \
  X(PFNGLUNIFORM4FPROC, glUniform4f)                                   \
  X(PFNGLGENBUFFERSPROC, glGenBuffers)                                 \
  X(PFNGLDELETEBUFFERSPROC, glDeleteBuffers)                           \
  X(PFNGLBINDBUFFERPROC, glBindBuffer)                                 \
  X(PFNGLBUFFERDATAPROC, glBufferData)                                 \
  X(PFNGLMAPBUFFERRANGEPROC, glMapBufferRange)                         \
  X(PFNGLUNMAPBUFFERPROC, glUnmapBuffer)                               \
  X(PFNGLGENVERTEXARRAYSPROC, glGenVertexArrays)                       \
  X(PFNGLDELETEVERTEXARRAYSPROC, glDeleteVertexArrays)                 \
  X(PFNGLBINDVERTEXARRAYPROC, glBindVertexArray)                       \
  X(PFNGLVERTEXATTRIBPOINTERPROC, glVertexAttribPointer)               \
  X(PFNGLENABLEVERTEXATTRIBARRAYPROC, glEnableVertexAttribArray)       \
  X(PFNGLDRAWARRAYSPROC, glDrawArrays)                                 \
  X(PFNGLFLUSHPROC, glFlush)                                           \
  X(PFNGLFINISHPROC, glFinish)                                         \
  X(PFNGLFENCESYNCPROC, glFenceSync)                                   \
  X(PFNGLCLIENTWAITSYNCPROC, glClientWaitSync)                         \
  X(PFNGLDELETESYNCPROC, glDeleteSync)

#define GL_EXT_FUNCS(X)                                                              \
  X(PFNGLEGLIMAGETARGETTEXTURE2DOESPROC, glEGLImageTargetTexture2DOES)               \
  X(PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC, glEGLImageTargetRenderbufferStorageOES)

#define PROBE_DECLARE_FN(T, n) extern T n;
EGL_CORE_FUNCS(PROBE_DECLARE_FN)
EGL_EXT_FUNCS(PROBE_DECLARE_FN)
GL_FUNCS(PROBE_DECLARE_FN)
GL_EXT_FUNCS(PROBE_DECLARE_FN)
#undef PROBE_DECLARE_FN

// Loads libGLESv2.dll and libEGL.dll from `dir` and resolves every entry
// point above. Extension entry points that are missing stay null.
bool load_angle(const std::string& dir);
std::string default_angle_dir();

// ---------------------------------------------------------------------------
// GL session: one ANGLE display, one context current on the calling thread
// ---------------------------------------------------------------------------

enum class DisplayMode { AngleHardware, AngleWarp, InjectHardware, InjectWarp, AngleD3D11On12 };
const char* mode_name(DisplayMode m);

struct GlSession {
  DisplayMode mode = DisplayMode::AngleHardware;
  EGLDisplay dpy = EGL_NO_DISPLAY;
  EGLConfig cfg = nullptr;         // ES3 pbuffer config with EGL_BIND_TO_TEXTURE_RGBA
  EGLConfig cfg_nobind = nullptr;  // same without the bind attribute
  EGLContext ctx = EGL_NO_CONTEXT;
  EGLSurface dummy = EGL_NO_SURFACE;
  EGLDeviceEXT egl_device = nullptr;
  D3D d3d;                         // the device behind the display
  D3D injected;                    // owner of the device when mode is Inject*
  std::string display_exts, device_exts, gl_exts, gl_renderer, gl_version;
  std::string last_error;  // why bring_up failed, for the report line
  int es_minor = -1;

  // Creates the display for `mode`, an ES 3.x context and a 16x16 dummy
  // pbuffer, and makes them current on the calling thread. For the Inject
  // modes `existing` hands ANGLE an already created device instead of a
  // new one.
  bool bring_up(DisplayMode mode, ID3D11Device* existing = nullptr);
  EGLContext create_context();
  EGLSurface create_dummy();
  bool make_current(EGLSurface s, EGLContext c);
  bool restore_current();  // dummy + ctx
  void shutdown();
};

// ---------------------------------------------------------------------------
// GL helpers
// ---------------------------------------------------------------------------

extern const char* kVertexShader;        // the HAL's VERTEX_SHADER, verbatim
extern const char* kSampleFragment;      // color = texture(tex, tc)
extern const char* kGradientFragment;    // the gradient in make_gradient()
extern const char* kNvRgbaFragment;      // the HAL's NV_RGBA_FRAGMENT, verbatim
extern const char* kYuyvRgbaFragment;    // the HAL's YUYV_RGBA_2D_FRAGMENT, verbatim
extern const char* kNv12TwoPlaneFragment;  // Y (R8) + UV (RG8) two-sampler decode

GLuint compile_program(const char* vs, const char* fs, std::string* log = nullptr);
struct Quad {
  GLuint vao = 0, vbo = 0;
  void init();
  void draw() const;
  void destroy();
};
// Plain FBO with a fresh texture as colour attachment.
struct PlainFbo {
  GLuint fbo = 0, tex = 0;
  uint32_t w = 0, h = 0;
  bool is_float = false;
  bool init(uint32_t w, uint32_t h, bool is_float);
  void destroy();
};
// Reads the current READ_FRAMEBUFFER as RGBA8 or RGBA32F (tight).
bool read_fbo_rgba8(uint32_t w, uint32_t h, std::vector<uint8_t>& out);
bool read_fbo_rgba_f32(uint32_t w, uint32_t h, std::vector<float>& out);
void set_nearest(GLuint tex);
const char* gl_fbo_status_str(GLenum s);
bool gl_clear_errors(const char* where);  // returns true when no error pending

// ---------------------------------------------------------------------------
// Import routes for a D3D11 texture
// ---------------------------------------------------------------------------

enum class Route { Pbuffer, EglImage };
const char* route_name(Route r);
GLenum gl_internal_format_for(DXGI_FORMAT fmt);  // EGL_TEXTURE_INTERNAL_FORMAT_ANGLE value

struct Import {
  Route route = Route::Pbuffer;
  EGLSurface pbuffer = EGL_NO_SURFACE;
  EGLImageKHR image = EGL_NO_IMAGE_KHR;
  GLuint tex = 0;   // GL texture object the import is attached to
  EGLint egl_error = EGL_SUCCESS;
  GLenum gl_error = GL_NO_ERROR;
  bool ok = false;
};
// plane < 0 for non-YUV textures. `internal_format` 0 = omit the attribute.
Import import_texture(GlSession& s, Route route, ID3D11Texture2D* tex, GLenum internal_format,
                      int plane = -1);
// Pbuffer: eglBindTexImage; EGLImage: no-op (binding is persistent).
bool import_bind(GlSession& s, Import& im);
bool import_release(GlSession& s, Import& im);
void import_destroy(GlSession& s, Import& im);

// ---------------------------------------------------------------------------
// Sections
// ---------------------------------------------------------------------------

void print_session(const GlSession& s);  // main.cpp
// Gradient render into an RGBA8 texture through both routes with readback;
// the smallest end-to-end check of a display (s6, s8).
bool quick_import_check(GlSession& s, const char* id, const char* label);
// Reads an RGBA8 texture through CUDA external memory (a fresh NT handle,
// cudaImportExternalMemory, level 0 copy). For the S7 child.
bool cuda_external_read_rgba8(ID3D11Texture2D* tex, IDXGIAdapter* adapter, uint32_t w, uint32_t h,
                              std::vector<uint8_t>& out, std::string* detail);

void run_s0(GlSession& s);
void run_s1(GlSession& s, const char* prefix = "S1");
void run_s2(GlSession& s);
void run_s3(GlSession& s);
void run_s4(GlSession& s, const char* prefix = "S4");
void run_s4_control(GlSession& s);
// Runs this executable again with `extra_args` (plus --warp / --angle as
// given to this process), waits up to `timeout_ms`, returns the exit code
// or 0xFFFFFFFF on timeout / launch failure.
DWORD spawn_self(const std::wstring& extra_args, DWORD timeout_ms);
void run_s5(GlSession& s);
void run_s6(GlSession& s);
void run_s7(GlSession& s);
int run_s7_child();
void run_s8(GlSession& s);
