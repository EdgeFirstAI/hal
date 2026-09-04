// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// ANGLE loading, display bring-up in every ownership mode, GL helpers and
// the two D3D11 texture import routes.
#include "probe.h"

#include <cstring>

#define PROBE_DEFINE_FN(T, n) T n = nullptr;
EGL_CORE_FUNCS(PROBE_DEFINE_FN)
EGL_EXT_FUNCS(PROBE_DEFINE_FN)
GL_FUNCS(PROBE_DEFINE_FN)
GL_EXT_FUNCS(PROBE_DEFINE_FN)
#undef PROBE_DEFINE_FN

static HMODULE g_libegl = nullptr;
static HMODULE g_libgles = nullptr;

static bool file_exists(const std::string& p) {
  DWORD a = GetFileAttributesA(p.c_str());
  return a != INVALID_FILE_ATTRIBUTES && !(a & FILE_ATTRIBUTE_DIRECTORY);
}

std::string default_angle_dir() {
  char env[MAX_PATH];
  DWORD n = GetEnvironmentVariableA("EDGEFIRST_ANGLE_PATH", env, sizeof env);
  if (n > 0 && n < sizeof env && file_exists(std::string(env) + "\\libEGL.dll")) return env;
  char exe[MAX_PATH];
  GetModuleFileNameA(nullptr, exe, sizeof exe);
  std::string dir = exe;
  size_t slash = dir.find_last_of("\\/");
  dir = slash == std::string::npos ? "." : dir.substr(0, slash);
  // The build script places the exe in target/d3d11-probe/, next to
  // target/angle/windows-x64/bin.
  std::string candidate = dir + "\\..\\angle\\windows-x64\\bin";
  if (file_exists(candidate + "\\libEGL.dll")) return candidate;
  candidate = "target\\angle\\windows-x64\\bin";
  if (file_exists(candidate + "\\libEGL.dll")) return candidate;
  return "";
}

bool load_angle(const std::string& dir) {
  std::string gles = dir + "\\libGLESv2.dll";
  std::string egl = dir + "\\libEGL.dll";
  g_libgles = LoadLibraryExA(gles.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  g_libegl = LoadLibraryExA(egl.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  if (!g_libgles || !g_libegl) {
    fprintf(stderr, "cannot load ANGLE from %s (error %lu)\n", dir.c_str(), GetLastError());
    return false;
  }
  bool ok = true;
#define LOAD_EGL(T, n)                                            \
  n = (T)GetProcAddress(g_libegl, #n);                            \
  if (!n) {                                                       \
    fprintf(stderr, "libEGL.dll lacks %s\n", #n);                 \
    ok = false;                                                   \
  }
  EGL_CORE_FUNCS(LOAD_EGL)
#undef LOAD_EGL
  if (!ok) return false;
#define LOAD_GL(T, n)                                                     \
  n = (T)GetProcAddress(g_libgles, #n);                                   \
  if (!n) n = (T)eglGetProcAddress(#n);                                   \
  if (!n) {                                                               \
    fprintf(stderr, "libGLESv2.dll lacks %s\n", #n);                      \
    ok = false;                                                           \
  }
  GL_FUNCS(LOAD_GL)
#undef LOAD_GL
#define LOAD_EXT(T, n) n = (T)eglGetProcAddress(#n);
  EGL_EXT_FUNCS(LOAD_EXT)
  GL_EXT_FUNCS(LOAD_EXT)
#undef LOAD_EXT
  return ok;
}

const char* mode_name(DisplayMode m) {
  switch (m) {
    case DisplayMode::AngleHardware: return "angle-owned device, hardware adapter";
    case DisplayMode::AngleWarp: return "angle-owned device, WARP";
    case DisplayMode::InjectHardware: return "injected device, hardware adapter";
    case DisplayMode::InjectWarp: return "injected device, WARP";
    case DisplayMode::AngleD3D11On12: return "angle-owned device, D3D11on12";
  }
  return "?";
}

// ---------------------------------------------------------------------------
// GlSession
// ---------------------------------------------------------------------------

static bool has_token(const std::string& s, const char* tok) {
  size_t n = strlen(tok);
  size_t p = 0;
  while ((p = s.find(tok, p)) != std::string::npos) {
    bool left = p == 0 || s[p - 1] == ' ';
    bool right = p + n == s.size() || s[p + n] == ' ';
    if (left && right) return true;
    p += n;
  }
  return false;
}

bool GlSession::bring_up(DisplayMode m, ID3D11Device* existing) {
  mode = m;
  bool warp = (m == DisplayMode::AngleWarp || m == DisplayMode::InjectWarp);
  bool inject = (m == DisplayMode::InjectHardware || m == DisplayMode::InjectWarp);
  if (inject) {
    if (existing) {
      injected.wrap(existing);
    } else {
      AdapterInfo ad;
      if (!warp && !pick_adapter(g_opt.adapter, false, &ad)) {
        fprintf(stderr, "no adapter matches '%s'\n", g_opt.adapter.c_str());
        return false;
      }
      // Video support is what Media Foundation needs; when an adapter
      // refuses it (WARP does), fall back so the display still comes up
      // and the report shows which flags took.
      UINT debug = g_opt.debug_layer ? D3D11_CREATE_DEVICE_DEBUG : 0;
      UINT attempts[] = {D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT | debug,
                         D3D11_CREATE_DEVICE_BGRA_SUPPORT | debug, debug};
      HRESULT hr = E_FAIL;
      bool created = false;
      for (UINT flags : attempts) {
        if (create_device(warp ? nullptr : &ad, warp, flags, &injected, &hr)) {
          created = true;
          break;
        }
        char b[96];
        snprintf(b, sizeof b, "D3D11CreateDevice(flags 0x%x): %s; ", flags, hr_str(hr));
        last_error += b;
      }
      if (!created) return false;
    }
    if (!eglCreateDeviceANGLE) {
      fprintf(stderr, "eglCreateDeviceANGLE unavailable\n");
      return false;
    }
    egl_device = eglCreateDeviceANGLE(EGL_D3D11_DEVICE_ANGLE, injected.dev.Get(), nullptr);
    if (!egl_device) {
      fprintf(stderr, "eglCreateDeviceANGLE failed: %s\n", egl_err_str(eglGetError()));
      return false;
    }
    dpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_DEVICE_EXT, egl_device, nullptr);
  } else {
    std::vector<EGLint> attribs = {EGL_PLATFORM_ANGLE_TYPE_ANGLE,
                                   EGL_PLATFORM_ANGLE_TYPE_D3D11_ANGLE};
    attribs.push_back(EGL_PLATFORM_ANGLE_DEVICE_TYPE_ANGLE);
    attribs.push_back(warp ? EGL_PLATFORM_ANGLE_DEVICE_TYPE_D3D_WARP_ANGLE
                           : EGL_PLATFORM_ANGLE_DEVICE_TYPE_HARDWARE_ANGLE);
    if (!warp && !g_opt.adapter.empty()) {
      AdapterInfo ad;
      if (!pick_adapter(g_opt.adapter, false, &ad)) {
        fprintf(stderr, "no adapter matches '%s'\n", g_opt.adapter.c_str());
        return false;
      }
      attribs.push_back(EGL_PLATFORM_ANGLE_D3D_LUID_HIGH_ANGLE);
      attribs.push_back(ad.desc.AdapterLuid.HighPart);
      attribs.push_back(EGL_PLATFORM_ANGLE_D3D_LUID_LOW_ANGLE);
      attribs.push_back((EGLint)ad.desc.AdapterLuid.LowPart);
    }
    if (m == DisplayMode::AngleD3D11On12) {
      attribs.push_back(EGL_PLATFORM_ANGLE_D3D11ON12_ANGLE);
      attribs.push_back(EGL_TRUE);
    }
    if (g_opt.debug_layer) {
      attribs.push_back(EGL_PLATFORM_ANGLE_DEBUG_LAYERS_ENABLED_ANGLE);
      attribs.push_back(EGL_TRUE);
    }
    attribs.push_back(EGL_NONE);
    dpy = eglGetPlatformDisplayEXT(EGL_PLATFORM_ANGLE_ANGLE, EGL_DEFAULT_DISPLAY, attribs.data());
  }
  if (dpy == EGL_NO_DISPLAY) {
    last_error += std::string("eglGetPlatformDisplayEXT: ") + egl_err_str(eglGetError());
    fprintf(stderr, "%s\n", last_error.c_str());
    return false;
  }
  EGLint major, minor;
  if (!eglInitialize(dpy, &major, &minor)) {
    last_error += std::string("eglInitialize: ") + egl_err_str(eglGetError());
    fprintf(stderr, "%s\n", last_error.c_str());
    dpy = EGL_NO_DISPLAY;
    return false;
  }
  display_exts = eglQueryString(dpy, EGL_EXTENSIONS);
  eglBindAPI(EGL_OPENGL_ES_API);

  const EGLint cfg_attribs[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE,
                                EGL_PBUFFER_BIT,     EGL_RED_SIZE,       8,
                                EGL_GREEN_SIZE,      8,                  EGL_BLUE_SIZE,
                                8,                   EGL_ALPHA_SIZE,     8,
                                EGL_BIND_TO_TEXTURE_RGBA, EGL_TRUE,      EGL_NONE};
  EGLint n = 0;
  if (!eglChooseConfig(dpy, cfg_attribs, &cfg, 1, &n) || n == 0) {
    fprintf(stderr, "eglChooseConfig (bind-to-texture): %s\n", egl_err_str(eglGetError()));
    return false;
  }
  const EGLint cfg_attribs2[] = {EGL_RENDERABLE_TYPE, EGL_OPENGL_ES3_BIT, EGL_SURFACE_TYPE,
                                 EGL_PBUFFER_BIT,     EGL_RED_SIZE,       8,
                                 EGL_GREEN_SIZE,      8,                  EGL_BLUE_SIZE,
                                 8,                   EGL_ALPHA_SIZE,     8,
                                 EGL_BIND_TO_TEXTURE_RGBA, EGL_FALSE,     EGL_NONE};
  if (!eglChooseConfig(dpy, cfg_attribs2, &cfg_nobind, 1, &n) || n == 0) cfg_nobind = nullptr;

  ctx = create_context();
  if (ctx == EGL_NO_CONTEXT) return false;
  dummy = create_dummy();
  if (dummy == EGL_NO_SURFACE) return false;
  if (!make_current(dummy, ctx)) return false;

  gl_renderer = (const char*)glGetString(GL_RENDERER);
  gl_version = (const char*)glGetString(GL_VERSION);
  gl_exts = (const char*)glGetString(GL_EXTENSIONS);

  EGLAttrib dev_attr = 0;
  if (eglQueryDisplayAttribEXT && eglQueryDisplayAttribEXT(dpy, EGL_DEVICE_EXT, &dev_attr)) {
    EGLDeviceEXT dev = (EGLDeviceEXT)dev_attr;
    if (eglQueryDeviceStringEXT) {
      const char* s = eglQueryDeviceStringEXT(dev, EGL_EXTENSIONS);
      device_exts = s ? s : "";
    }
    EGLAttrib d3d_attr = 0;
    if (eglQueryDeviceAttribEXT && eglQueryDeviceAttribEXT(dev, EGL_D3D11_DEVICE_ANGLE, &d3d_attr) &&
        d3d_attr) {
      d3d.wrap((ID3D11Device*)d3d_attr);
    }
  }
  if (!d3d.dev) {
    fprintf(stderr, "could not query ANGLE's ID3D11Device\n");
    return false;
  }
  return true;
}

EGLContext GlSession::create_context() {
  for (int minor : {1, 0}) {
    const EGLint attribs[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_CONTEXT_MINOR_VERSION, minor,
                              EGL_NONE};
    EGLContext c = eglCreateContext(dpy, cfg, EGL_NO_CONTEXT, attribs);
    if (c != EGL_NO_CONTEXT) {
      if (es_minor < 0) es_minor = minor;
      return c;
    }
  }
  fprintf(stderr, "eglCreateContext: %s\n", egl_err_str(eglGetError()));
  return EGL_NO_CONTEXT;
}

EGLSurface GlSession::create_dummy() {
  const EGLint attribs[] = {EGL_WIDTH, 16, EGL_HEIGHT, 16, EGL_NONE};
  EGLSurface s = eglCreatePbufferSurface(dpy, cfg, attribs);
  if (s == EGL_NO_SURFACE) fprintf(stderr, "eglCreatePbufferSurface: %s\n", egl_err_str(eglGetError()));
  return s;
}

bool GlSession::make_current(EGLSurface s, EGLContext c) {
  if (!eglMakeCurrent(dpy, s, s, c)) {
    fprintf(stderr, "eglMakeCurrent: %s\n", egl_err_str(eglGetError()));
    return false;
  }
  return true;
}

bool GlSession::restore_current() { return make_current(dummy, ctx); }

void GlSession::shutdown() {
  if (dpy == EGL_NO_DISPLAY) return;
  eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
  if (dummy != EGL_NO_SURFACE) eglDestroySurface(dpy, dummy);
  if (ctx != EGL_NO_CONTEXT) eglDestroyContext(dpy, ctx);
  eglTerminate(dpy);
  if (egl_device && eglReleaseDeviceANGLE) eglReleaseDeviceANGLE(egl_device);
  dpy = EGL_NO_DISPLAY;
  dummy = EGL_NO_SURFACE;
  ctx = EGL_NO_CONTEXT;
  egl_device = nullptr;
}

// ---------------------------------------------------------------------------
// Shaders. The vertex, NV and YUYV sources are the HAL's
// (crates/image/src/gl/shaders_common.rs) copied verbatim so the probe
// exercises the same GLSL the engine would compile.
// ---------------------------------------------------------------------------

const char* kVertexShader =
    "#version 300 es\n"
    "precision mediump float;\n"
    "layout(location = 0) in vec3 pos;\n"
    "layout(location = 1) in vec2 texCoord;\n"
    "\n"
    "out vec3 fragPos;\n"
    "out vec2 tc;\n"
    "\n"
    "void main() {\n"
    "    fragPos = pos;\n"
    "    tc = texCoord;\n"
    "\n"
    "    gl_Position = vec4(pos, 1.0);\n"
    "}\n";

const char* kSampleFragment =
    "#version 300 es\n"
    "precision highp float;\n"
    "uniform highp sampler2D tex;\n"
    "in vec3 fragPos;\n"
    "in vec2 tc;\n"
    "out vec4 color;\n"
    "void main() { color = texture(tex, tc); }\n";

const char* kGradientFragment =
    "#version 300 es\n"
    "precision highp float;\n"
    "uniform float denom;\n"
    "in vec3 fragPos;\n"
    "in vec2 tc;\n"
    "out vec4 color;\n"
    "void main() {\n"
    "    float x = floor(gl_FragCoord.x);\n"
    "    float y = floor(gl_FragCoord.y);\n"
    "    color = vec4(mod(x, 256.0) / denom, mod(y, 256.0) / denom,\n"
    "                 mod(x + y, 256.0) / denom, 255.0 / denom);\n"
    "}\n";

const char* kNvRgbaFragment =
    "#version 300 es\n"
    "precision highp float;\n"
    "precision highp int;\n"
    "uniform highp sampler2D src;\n"
    "uniform ivec2 img_size;\n"
    "uniform int tex_width;\n"
    "uniform ivec2 chroma_shift;\n"
    "uniform int chroma_lines;\n"
    "uniform float y_offset;\n"
    "uniform float y_scale;\n"
    "uniform float c_vr;\n"
    "uniform float c_ug;\n"
    "uniform float c_vg;\n"
    "uniform float c_ub;\n"
    "in vec3 fragPos;\n"
    "in vec2 tc;\n"
    "out vec4 color;\n"
    "\n"
    "void main() {\n"
    "    int w = img_size.x;\n"
    "    int h = img_size.y;\n"
    "    int x = clamp(int(tc.x * float(w)), 0, w - 1);\n"
    "    int y = clamp(int(tc.y * float(h)), 0, h - 1);\n"
    "    float yv = texelFetch(src, ivec2(x, y), 0).r;\n"
    "    int ccol = x >> chroma_shift.x;\n"
    "    int crow = y >> chroma_shift.y;\n"
    "    int ccol2 = ccol * 2;\n"
    "    int carry = ccol2 >= tex_width ? 1 : 0;\n"
    "    int cy = h + crow * chroma_lines + carry;\n"
    "    int cx = ccol2 - carry * tex_width;\n"
    "    float u = texelFetch(src, ivec2(cx, cy), 0).r;\n"
    "    float v = texelFetch(src, ivec2(cx + 1, cy), 0).r;\n"
    "    float yp = max((yv - y_offset) * y_scale, 0.0);\n"
    "    float up = u - 128.0 / 255.0;\n"
    "    float vp = v - 128.0 / 255.0;\n"
    "    float r = clamp(yp + c_vr * vp, 0.0, 1.0);\n"
    "    float g = clamp(yp - c_ug * up - c_vg * vp, 0.0, 1.0);\n"
    "    float b = clamp(yp + c_ub * up, 0.0, 1.0);\n"
    "    color = vec4(r, g, b, 1.0);\n"
    "}\n";

const char* kYuyvRgbaFragment =
    "#version 300 es\n"
    "precision highp float;\n"
    "uniform highp sampler2D tex;\n"
    "uniform vec2 src_size;\n"
    "uniform vec4 src_extent;\n"
    "uniform float y_offset;\n"
    "uniform float y_scale;\n"
    "uniform float c_vr;\n"
    "uniform float c_ug;\n"
    "uniform float c_vg;\n"
    "uniform float c_ub;\n"
    "in vec3 fragPos;\n"
    "in vec2 tc;\n"
    "out vec4 color;\n"
    "\n"
    "void main() {\n"
    "    vec2 texel = vec2(1.0) / src_size;\n"
    "    vec2 col = floor(clamp(tc, src_extent.xy, src_extent.zw) * src_size);\n"
    "    bool even = mod(col.x, 2.0) < 0.5;\n"
    "    vec2 self_uv = (col + vec2(0.5)) * texel;\n"
    "    vec2 pair_uv = (col + vec2(even ? 1.5 : -0.5, 0.5)) * texel;\n"
    "\n"
    "    vec4 self_rg = texture(tex, self_uv);\n"
    "    vec4 pair_rg = texture(tex, pair_uv);\n"
    "    float y = self_rg.r;\n"
    "    float u, v;\n"
    "    if (even) { u = self_rg.g; v = pair_rg.g; }\n"
    "    else      { v = self_rg.g; u = pair_rg.g; }\n"
    "\n"
    "    float yp = max((y - y_offset) * y_scale, 0.0);\n"
    "    float up = u - 128.0 / 255.0;\n"
    "    float vp = v - 128.0 / 255.0;\n"
    "    float r = clamp(yp + c_vr * vp, 0.0, 1.0);\n"
    "    float g = clamp(yp - c_ug * up - c_vg * vp, 0.0, 1.0);\n"
    "    float b = clamp(yp + c_ub * up, 0.0, 1.0);\n"
    "    color = vec4(r, g, b, 1.0);\n"
    "}\n";

// Two-sampler NV12 decode for the native DXGI_FORMAT_NV12 plane import,
// where Y arrives as an R8 texture and UV as an RG8 texture at half size.
// The colour math is the HAL's, so the CPU reference in s2 applies.
const char* kNv12TwoPlaneFragment =
    "#version 300 es\n"
    "precision highp float;\n"
    "uniform highp sampler2D y_tex;\n"
    "uniform highp sampler2D uv_tex;\n"
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

static GLuint compile_shader(GLenum type, const char* src, std::string* log) {
  GLuint sh = glCreateShader(type);
  glShaderSource(sh, 1, &src, nullptr);
  glCompileShader(sh);
  GLint ok = 0;
  glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char buf[2048];
    GLsizei n = 0;
    glGetShaderInfoLog(sh, sizeof buf, &n, buf);
    if (log) *log = std::string(buf, n);
    glDeleteShader(sh);
    return 0;
  }
  return sh;
}

GLuint compile_program(const char* vs, const char* fs, std::string* log) {
  GLuint v = compile_shader(GL_VERTEX_SHADER, vs, log);
  if (!v) return 0;
  GLuint f = compile_shader(GL_FRAGMENT_SHADER, fs, log);
  if (!f) {
    glDeleteShader(v);
    return 0;
  }
  GLuint p = glCreateProgram();
  glAttachShader(p, v);
  glAttachShader(p, f);
  glLinkProgram(p);
  glDeleteShader(v);
  glDeleteShader(f);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char buf[2048];
    GLsizei n = 0;
    glGetProgramInfoLog(p, sizeof buf, &n, buf);
    if (log) *log = std::string(buf, n);
    glDeleteProgram(p);
    return 0;
  }
  return p;
}

void Quad::init() {
  // Two triangles; tc.y = 0 at the bottom edge so texel row r of the
  // source lands on fragment row r of the target (no flip in GL terms).
  static const float verts[] = {
      -1, -1, 0, 0, 0,  1, -1, 0, 1, 0,  -1, 1, 0, 0, 1,
      -1, 1,  0, 0, 1,  1, -1, 0, 1, 0,  1,  1, 0, 1, 1,
  };
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glGenBuffers(1, &vbo);
  glBindBuffer(GL_ARRAY_BUFFER, vbo);
  glBufferData(GL_ARRAY_BUFFER, sizeof verts, verts, GL_STATIC_DRAW);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
  glEnableVertexAttribArray(1);
  glBindVertexArray(0);
}

void Quad::draw() const {
  glBindVertexArray(vao);
  glDrawArrays(GL_TRIANGLES, 0, 6);
  glBindVertexArray(0);
}

void Quad::destroy() {
  if (vbo) glDeleteBuffers(1, &vbo);
  if (vao) glDeleteVertexArrays(1, &vao);
  vbo = vao = 0;
}

bool PlainFbo::init(uint32_t width, uint32_t height, bool flt) {
  w = width;
  h = height;
  is_float = flt;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexStorage2D(GL_TEXTURE_2D, 1, flt ? GL_RGBA32F : GL_RGBA8, (GLsizei)w, (GLsizei)h);
  set_nearest(tex);
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);
  GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  return st == GL_FRAMEBUFFER_COMPLETE;
}

void PlainFbo::destroy() {
  if (fbo) glDeleteFramebuffers(1, &fbo);
  if (tex) glDeleteTextures(1, &tex);
  fbo = tex = 0;
}

bool read_fbo_rgba8(uint32_t w, uint32_t h, std::vector<uint8_t>& out) {
  out.resize((size_t)w * h * 4);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadPixels(0, 0, (GLsizei)w, (GLsizei)h, GL_RGBA, GL_UNSIGNED_BYTE, out.data());
  return glGetError() == GL_NO_ERROR;
}

bool read_fbo_rgba_f32(uint32_t w, uint32_t h, std::vector<float>& out) {
  out.resize((size_t)w * h * 4);
  glPixelStorei(GL_PACK_ALIGNMENT, 4);
  glReadPixels(0, 0, (GLsizei)w, (GLsizei)h, GL_RGBA, GL_FLOAT, out.data());
  return glGetError() == GL_NO_ERROR;
}

void set_nearest(GLuint tex) {
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}

const char* gl_fbo_status_str(GLenum s) {
  switch (s) {
    case GL_FRAMEBUFFER_COMPLETE: return "COMPLETE";
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT: return "INCOMPLETE_ATTACHMENT";
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT: return "MISSING_ATTACHMENT";
    case GL_FRAMEBUFFER_INCOMPLETE_DIMENSIONS: return "INCOMPLETE_DIMENSIONS";
    case GL_FRAMEBUFFER_UNSUPPORTED: return "UNSUPPORTED";
    case GL_FRAMEBUFFER_INCOMPLETE_MULTISAMPLE: return "INCOMPLETE_MULTISAMPLE";
    default: return "?";
  }
}

bool gl_clear_errors(const char* where) {
  bool clean = true;
  for (GLenum e; (e = glGetError()) != GL_NO_ERROR;) {
    printf("         gl error 0x%04x at %s\n", e, where);
    clean = false;
  }
  return clean;
}

// ---------------------------------------------------------------------------
// Import routes
// ---------------------------------------------------------------------------

const char* route_name(Route r) { return r == Route::Pbuffer ? "pbuffer" : "eglimage"; }

GLenum gl_internal_format_for(DXGI_FORMAT fmt) {
  switch (fmt) {
    case DXGI_FORMAT_R8G8B8A8_UNORM:
    case DXGI_FORMAT_R16G16B16A16_FLOAT:
    case DXGI_FORMAT_R32G32B32A32_FLOAT: return GL_RGBA;
    case DXGI_FORMAT_B8G8R8A8_UNORM: return GL_BGRA_EXT;
    case DXGI_FORMAT_R8_UNORM: return GL_RED_EXT;
    case DXGI_FORMAT_R8G8_UNORM: return GL_RG_EXT;
    case DXGI_FORMAT_R16_UNORM: return GL_R16_EXT;
    case DXGI_FORMAT_R16G16_UNORM: return GL_RG16_EXT;
    case DXGI_FORMAT_R10G10B10A2_UNORM: return GL_RGB10_A2_EXT;
    default: return 0;
  }
}

Import import_texture(GlSession& s, Route route, ID3D11Texture2D* tex, GLenum internal_format,
                      int plane) {
  Import im;
  im.route = route;
  std::vector<EGLint> attribs;
  if (route == Route::Pbuffer) {
    attribs = {EGL_TEXTURE_FORMAT, EGL_TEXTURE_RGBA, EGL_TEXTURE_TARGET, EGL_TEXTURE_2D};
  }
  if (internal_format) {
    attribs.push_back(EGL_TEXTURE_INTERNAL_FORMAT_ANGLE);
    attribs.push_back((EGLint)internal_format);
  }
  if (plane >= 0) {
    attribs.push_back(EGL_D3D11_TEXTURE_PLANE_ANGLE);
    attribs.push_back(plane);
  }
  attribs.push_back(EGL_NONE);
  glGenTextures(1, &im.tex);
  set_nearest(im.tex);
  if (route == Route::Pbuffer) {
    im.pbuffer = eglCreatePbufferFromClientBuffer(s.dpy, EGL_D3D_TEXTURE_ANGLE,
                                                  (EGLClientBuffer)tex, s.cfg, attribs.data());
    if (im.pbuffer == EGL_NO_SURFACE) {
      im.egl_error = eglGetError();
      glDeleteTextures(1, &im.tex);
      im.tex = 0;
      return im;
    }
    im.ok = true;
    return im;
  }
  if (!eglCreateImageKHR || !glEGLImageTargetTexture2DOES) {
    im.egl_error = EGL_BAD_ACCESS;
    return im;
  }
  im.image = eglCreateImageKHR(s.dpy, EGL_NO_CONTEXT, EGL_D3D11_TEXTURE_ANGLE,
                               (EGLClientBuffer)tex, attribs.data());
  if (im.image == EGL_NO_IMAGE_KHR) {
    im.egl_error = eglGetError();
    glDeleteTextures(1, &im.tex);
    im.tex = 0;
    return im;
  }
  glBindTexture(GL_TEXTURE_2D, im.tex);
  glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, (GLeglImageOES)im.image);
  im.gl_error = glGetError();
  im.ok = im.gl_error == GL_NO_ERROR;
  return im;
}

bool import_bind(GlSession& s, Import& im) {
  glBindTexture(GL_TEXTURE_2D, im.tex);
  if (im.route == Route::Pbuffer) {
    if (!eglBindTexImage(s.dpy, im.pbuffer, EGL_BACK_BUFFER)) {
      im.egl_error = eglGetError();
      return false;
    }
  }
  return true;
}

bool import_release(GlSession& s, Import& im) {
  if (im.route == Route::Pbuffer) {
    if (!eglReleaseTexImage(s.dpy, im.pbuffer, EGL_BACK_BUFFER)) {
      im.egl_error = eglGetError();
      return false;
    }
  }
  return true;
}

void import_destroy(GlSession& s, Import& im) {
  if (im.tex) glDeleteTextures(1, &im.tex);
  if (im.pbuffer != EGL_NO_SURFACE) eglDestroySurface(s.dpy, im.pbuffer);
  if (im.image != EGL_NO_IMAGE_KHR && eglDestroyImageKHR) eglDestroyImageKHR(s.dpy, im.image);
  im.tex = 0;
  im.pbuffer = EGL_NO_SURFACE;
  im.image = EGL_NO_IMAGE_KHR;
  im.ok = false;
}
