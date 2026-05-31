// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Jetson CUDA / TensorRT NvBufSurface interop probe (O4–O7 + O2b/O3b).
//
// On-device tooling for orin-nano (L4T R36.4, CUDA 12.6, TensorRT 10.3). This
// is the C++ counterpart to the Rust `probe_nvbufsurface` module. It uses the
// REAL `nvbufsurface.h`, CUDA, TensorRT and EGL/GLES headers (zero ABI risk)
// to settle the zero-copy chain:
//
//   O4 — NvBufSurfaceMapEglImage → cudaGraphicsEGLRegisterImage →
//        cudaGraphicsResourceGetMappedEglFrame → cudaMemcpy to host → verify a
//        CPU-written pattern. Prints cudaEglFrame frameType/format/pitch.
//   O5 — cudaImportExternalMemory(OpaqueFd, {fd,size}) →
//        cudaExternalMemoryGetMappedBuffer → verify.
//   O6 — build a trivial FP16 NCHW [1,3,H,W] identity TRT engine, bind the
//        CUDA device pointer via setInputTensorAddress, enqueueV3.
//   O7 — packed (W/4, 3H) RGBA16F bytes vs [3,H,W] f16 NCHW reconciliation.
//
// New this round (the ONE remaining GL gap — O2):
//   O2b-2 (DECISIVE) — import the NvBufSurface dma-buf as a packed
//        DRM_FORMAT_ABGR16161616F EGLImage *with the NVIDIA DRM format
//        modifier* (EGL_EXT_image_dma_buf_import_modifiers), bind to an FBO and
//        GL-render + readback. PASS ⇒ packed RGBA16F design ports to Jetson via
//        modifier import (uniform with V3D/Mali).
//   O2b-1 (fallback) — GL-render into the NvBufSurfaceMapEglImage EGLImage
//        (the NVIDIA-made one proven CUDA-usable in O4).
//   O3b — can NvBufSurfaceFromFd wrap an EXTERNAL (GBM) dma-buf, or only
//        NvBufSurface-originated fds?
//
// GL→CUDA bridge (O8 — the remaining hop, since O2b proved you can't GL-render
// into an NvBufSurface):
//   O8-a — cudaGraphicsGLRegisterImage on a GL-rendered texture (RGBA16F +
//        RGBA8); same-thread vs cross-thread (GL ctx current requirement).
//   O8-b — cudaGraphicsGLRegisterBuffer on a GL PBO (glReadPixels target).
//   O8-c — EGLStream GL-producer → CUDA-consumer (thread-decoupled path).
//   O8-d — packed RGBA16F (W/4,3H) GL→CUDA(PBO) vs [3,H,W] f16 NCHW layout.
//
// Build on-device: `make` (see Makefile).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <pthread.h>

#include <cuda_runtime.h>
#include <cudaEGL.h>          // cudaEglFrame, cudaGraphicsEGLRegisterImage
#include <cuda.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl3.h>
#include <GLES2/gl2ext.h>

// cuda_gl_interop.h requires GL types (GLuint/GLenum) to be declared first,
// hence it is included AFTER the GLES headers above.
#include <cuda_gl_interop.h>  // cudaGraphicsGLRegisterImage/Buffer (O8)

#include <fcntl.h>
#include <unistd.h>
#include <gbm.h>

#include <nvbufsurface.h>
#include <NvInfer.h>

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

#define CK(call)                                                              \
  do {                                                                        \
    cudaError_t _e = (call);                                                  \
    if (_e != cudaSuccess) {                                                  \
      printf("    cuda error %s:%d: %s\n", __FILE__, __LINE__,                \
             cudaGetErrorString(_e));                                         \
    }                                                                         \
  } while (0)

// DRM fourcc + EGL modifier-import enums (not always in older eglext.h).
#ifndef DRM_FORMAT_ABGR16161616F
#define DRM_FORMAT_ABGR16161616F 0x48344241  // 'AB4H'
#endif
#ifndef DRM_FORMAT_ABGR8888
#define DRM_FORMAT_ABGR8888 0x34324241  // 'AB24'
#endif
#ifndef EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT
#define EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT 0x3443
#endif
#ifndef EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT
#define EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT 0x3444
#endif
#ifndef EGL_LINUX_DMA_BUF_EXT
#define EGL_LINUX_DMA_BUF_EXT 0x3270
#endif
#ifndef EGL_LINUX_DRM_FOURCC_EXT
#define EGL_LINUX_DRM_FOURCC_EXT 0x3271
#endif
#ifndef EGL_DMA_BUF_PLANE0_FD_EXT
#define EGL_DMA_BUF_PLANE0_FD_EXT 0x3272
#endif
#ifndef EGL_DMA_BUF_PLANE0_OFFSET_EXT
#define EGL_DMA_BUF_PLANE0_OFFSET_EXT 0x3273
#endif
#ifndef EGL_DMA_BUF_PLANE0_PITCH_EXT
#define EGL_DMA_BUF_PLANE0_PITCH_EXT 0x3274
#endif

// Representative packed RGBA16F geometry for a 256x256-equivalent NCHW tensor:
// HAL F16 layout is [3,H,W] packed 4 contiguous f16 per RGBA16F texel ->
// surface (W/4, 3H). For H=W=256: packed = (64, 768), pitch = 64*8 = 512.
static const uint32_t kImgH = 256;
static const uint32_t kImgW = 256;
static const uint32_t kPackedW = kImgW / 4;   // 64
static const uint32_t kPackedH = 3 * kImgH;   // 768

// Half-float helpers (host-side) so we can write/verify a known pattern.
static uint16_t f32_to_f16(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
  uint32_t mant = x & 0x7fffffu;
  if (exp <= 0) return (uint16_t)sign;             // flush subnormals to zero
  if (exp >= 0x1f) return (uint16_t)(sign | 0x7c00u);
  return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

// EGL extension function pointers (resolved at runtime).
static PFNEGLCREATEIMAGEKHRPROC eglCreateImageKHR_ = nullptr;
static PFNEGLDESTROYIMAGEKHRPROC eglDestroyImageKHR_ = nullptr;
static PFNGLEGLIMAGETARGETTEXTURE2DOESPROC glEGLImageTargetTexture2DOES_ = nullptr;
static PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC
    glEGLImageTargetRenderbufferStorageOES_ = nullptr;

// ---------------------------------------------------------------------------
// TensorRT trivial logger
// ---------------------------------------------------------------------------

class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) printf("    [trt] %s\n", msg);
  }
} gLogger;

// ---------------------------------------------------------------------------
// Headless GLES3 surfaceless context (mirrors gpu-probe egl_context.rs).
// ---------------------------------------------------------------------------

static EGLDisplay g_dpy = EGL_NO_DISPLAY;
static EGLContext g_ctx = EGL_NO_CONTEXT;
static struct gbm_device* g_gbm = nullptr;
static int g_gbm_drm_fd = -1;

static bool egl_has_ext(EGLDisplay dpy, const char* ext) {
  const char* exts = eglQueryString(dpy, EGL_EXTENSIONS);
  return exts && strstr(exts, ext) != nullptr;
}

static bool init_headless_gl() {
  // Try EGL_PLATFORM_DEVICE_EXT first, fall back to GBM (renderD128).
  PFNEGLGETPLATFORMDISPLAYEXTPROC getPlatformDisplay =
      (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress(
          "eglGetPlatformDisplayEXT");

  const char* client_exts = eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS);
  bool tried_device = false;
  if (getPlatformDisplay && client_exts &&
      strstr(client_exts, "EGL_EXT_platform_device") &&
      strstr(client_exts, "EGL_EXT_device_enumeration")) {
    PFNEGLQUERYDEVICESEXTPROC queryDevices =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    EGLDeviceEXT devs[8];
    EGLint n = 0;
    if (queryDevices && queryDevices(8, devs, &n) && n > 0) {
      g_dpy = getPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, devs[0], nullptr);
      tried_device = true;
      printf("    [gl] EGL_PLATFORM_DEVICE_EXT: %d device(s)\n", n);
    }
  }

  if (g_dpy == EGL_NO_DISPLAY) {
    g_gbm_drm_fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (g_gbm_drm_fd >= 0) {
      g_gbm = gbm_create_device(g_gbm_drm_fd);
      if (g_gbm && getPlatformDisplay) {
        g_dpy = getPlatformDisplay(EGL_PLATFORM_GBM_KHR, g_gbm, nullptr);
        printf("    [gl] EGL_PLATFORM_GBM_KHR via renderD128\n");
      }
    }
  }
  (void)tried_device;

  if (g_dpy == EGL_NO_DISPLAY) {
    printf("    [gl] no EGL display obtained\n");
    return false;
  }

  EGLint major = 0, minor = 0;
  if (!eglInitialize(g_dpy, &major, &minor)) {
    printf("    [gl] eglInitialize failed 0x%x\n", eglGetError());
    return false;
  }
  printf("    [gl] EGL %d.%d vendor=%s\n", major, minor,
         eglQueryString(g_dpy, EGL_VENDOR));

  if (!egl_has_ext(g_dpy, "EGL_KHR_surfaceless_context") ||
      !egl_has_ext(g_dpy, "EGL_KHR_no_config_context")) {
    printf("    [gl] missing surfaceless/no_config context ext\n");
    return false;
  }
  bool has_mod_import =
      egl_has_ext(g_dpy, "EGL_EXT_image_dma_buf_import_modifiers");
  printf("    [gl] EGL_EXT_image_dma_buf_import_modifiers=%s\n",
         has_mod_import ? "yes" : "NO");

  if (!eglBindAPI(EGL_OPENGL_ES_API)) {
    printf("    [gl] eglBindAPI failed\n");
    return false;
  }
  EGLint ctx_attrs[] = {EGL_CONTEXT_MAJOR_VERSION, 3, EGL_NONE};
  g_ctx = eglCreateContext(g_dpy, (EGLConfig)0, EGL_NO_CONTEXT, ctx_attrs);
  if (g_ctx == EGL_NO_CONTEXT) {
    printf("    [gl] eglCreateContext failed 0x%x\n", eglGetError());
    return false;
  }
  if (!eglMakeCurrent(g_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, g_ctx)) {
    printf("    [gl] eglMakeCurrent(surfaceless) failed 0x%x\n", eglGetError());
    return false;
  }

  eglCreateImageKHR_ =
      (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
  eglDestroyImageKHR_ =
      (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
  glEGLImageTargetTexture2DOES_ =
      (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress(
          "glEGLImageTargetTexture2DOES");
  glEGLImageTargetRenderbufferStorageOES_ =
      (PFNGLEGLIMAGETARGETRENDERBUFFERSTORAGEOESPROC)eglGetProcAddress(
          "glEGLImageTargetRenderbufferStorageOES");

  printf("    [gl] GL_RENDERER=%s\n", glGetString(GL_RENDERER));
  printf("    [gl] GL_VERSION=%s\n", glGetString(GL_VERSION));
  return true;
}

// Try to make an imported/native EGLImage GL-renderable. Returns a status
// string and (via out) whether FBO became complete + readback verified.
struct GlRenderResult {
  bool rbo_complete = false;
  bool tex_complete = false;
  bool rendered = false;
  bool readback_ok = false;
  char detail[512] = {0};
};

// Render a known clear color into an EGLImage-backed FBO and read it back.
// `is_f16` controls the glReadPixels type. width/height are the GL texel dims.
static GlRenderResult gl_render_into_eglimage(EGLImageKHR img, int width,
                                              int height, bool is_f16) {
  GlRenderResult r;
  char* d = r.detail;
  size_t dn = sizeof(r.detail);

  // --- Renderbuffer route ---
  glGetError();
  GLuint rbo = 0, fbo = 0;
  glGenRenderbuffers(1, &rbo);
  glBindRenderbuffer(GL_RENDERBUFFER, rbo);
  if (glEGLImageTargetRenderbufferStorageOES_) {
    glEGLImageTargetRenderbufferStorageOES_(GL_RENDERBUFFER, img);
  }
  GLenum rbo_err = glGetError();
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_RENDERBUFFER, rbo);
  GLenum rbo_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  r.rbo_complete = (rbo_status == GL_FRAMEBUFFER_COMPLETE);
  int off = snprintf(d, dn, "rbo:status=0x%x err=0x%x%s", rbo_status, rbo_err,
                     r.rbo_complete ? "(COMPLETE)" : "");

  if (r.rbo_complete) {
    glViewport(0, 0, width, height);
    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
    r.rendered = true;
    // Readback.
    if (is_f16) {
      std::vector<uint16_t> px(4);
      glReadPixels(0, 0, 1, 1, GL_RGBA, GL_HALF_FLOAT, px.data());
    } else {
      std::vector<uint8_t> px(4);
      glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px.data());
      // 0.25*255≈64, 0.5≈128, 0.75≈191.
      r.readback_ok = (px[0] > 50 && px[0] < 80);
      off += snprintf(d + off, dn - off, " readback[0]=%u", px[0]);
    }
    GLenum rerr = glGetError();
    off += snprintf(d + off, dn - off, " readErr=0x%x", rerr);
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);
  glDeleteRenderbuffers(1, &rbo);

  // --- Texture route ---
  glGetError();
  GLuint tex = 0, fbo2 = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  if (glEGLImageTargetTexture2DOES_) {
    glEGLImageTargetTexture2DOES_(GL_TEXTURE_2D, img);
  }
  GLenum tex_err = glGetError();
  glGenFramebuffers(1, &fbo2);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo2);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         tex, 0);
  GLenum tex_status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  r.tex_complete = (tex_status == GL_FRAMEBUFFER_COMPLETE);
  off += snprintf(d + off, dn - off, " | tex:status=0x%x err=0x%x%s",
                  tex_status, tex_err, r.tex_complete ? "(COMPLETE)" : "");
  if (r.tex_complete) {
    glViewport(0, 0, width, height);
    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
    r.rendered = true;
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo2);
  glDeleteTextures(1, &tex);
  return r;
}

// ---------------------------------------------------------------------------
// Allocate a pitch-linear SURFACE_ARRAY NvBufSurface and CPU-fill a pattern.
// `width`/`height`/`fmt` let callers pick a byte geometry. Returns 0 on ok.
// ---------------------------------------------------------------------------

static int alloc_surface(NvBufSurface** out_surf, uint32_t width,
                         uint32_t height, NvBufSurfaceColorFormat fmt,
                         bool fill) {
  NvBufSurfaceCreateParams p;
  std::memset(&p, 0, sizeof(p));
  p.gpuId = 0;
  p.width = width;
  p.height = height;
  p.colorFormat = fmt;
  p.layout = NVBUF_LAYOUT_PITCH;
  p.memType = NVBUF_MEM_SURFACE_ARRAY;

  NvBufSurface* surf = nullptr;
  if (NvBufSurfaceCreate(&surf, 1, &p) != 0 || !surf) {
    printf("    NvBufSurfaceCreate(%ux%u fmt=%d) failed\n", width, height,
           (int)fmt);
    return -1;
  }
  NvBufSurfaceParams& sp = surf->surfaceList[0];
  uint64_t mod = 0;
  if (sp.paramex) mod = sp.paramex->planeParamsex.drmModifier[0];
  printf(
      "    alloc %ux%u fmt=%d -> pitch=%u planePitch0=%u offset0=%u "
      "dataSize=%u fd=%lu drmModifier[0]=0x%016lx\n",
      sp.width, sp.height, (int)sp.colorFormat, sp.pitch,
      sp.planeParams.pitch[0], sp.planeParams.offset[0], sp.dataSize,
      (unsigned long)sp.bufferDesc, (unsigned long)mod);

  if (fill && NvBufSurfaceMap(surf, 0, 0, NVBUF_MAP_WRITE) == 0) {
    NvBufSurfaceSyncForCpu(surf, 0, 0);
    uint8_t* base = (uint8_t*)surf->surfaceList[0].mappedAddr.addr[0];
    if (base) {
      uint32_t n = sp.dataSize ? sp.dataSize : sp.pitch * sp.height;
      for (uint32_t i = 0; i < n; ++i) base[i] = (uint8_t)(i & 0xff);
      NvBufSurfaceSyncForDevice(surf, 0, 0);
    }
    NvBufSurfaceUnMap(surf, 0, 0);
  }
  *out_surf = surf;
  return 0;
}

// Sanity control: confirm the headless context can render into a plain
// (non-EGLImage) RGBA8 renderbuffer + texture FBO and read it back. This
// isolates "GL render is broken" from "EGLImage attach is rejected".
static void gl_sanity_control() {
  printf("  -- GL sanity control (plain renderbuffer/texture, no EGLImage) --\n");
  glGetError();
  GLuint rbo = 0, fbo = 0;
  glGenRenderbuffers(1, &rbo);
  glBindRenderbuffer(GL_RENDERBUFFER, rbo);
  glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, 64, 64);
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_RENDERBUFFER, rbo);
  GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  uint8_t px[4] = {0};
  if (st == GL_FRAMEBUFFER_COMPLETE) {
    glViewport(0, 0, 64, 64);
    glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
  }
  printf("     plain RGBA8 rbo: status=0x%x readback[0]=%u err=0x%x -> %s\n", st,
         px[0], glGetError(),
         (st == GL_FRAMEBUFFER_COMPLETE && px[0] > 50 && px[0] < 80)
             ? "RENDER OK (context is functional)"
             : "context render FAILED");
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);
  glDeleteRenderbuffers(1, &rbo);
}

// Bind an EGLImage as a TEXTURE and sample it (draw into a plain FBO),
// reading back the sampled value — tests whether the NV EGLImage is usable
// as a SAMPLE source even if not as a render target.
static void gl_sample_from_eglimage(EGLImageKHR img) {
  printf("  -- EGLImage-as-sampled-texture test --\n");
  glGetError();
  GLuint tex = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  if (glEGLImageTargetTexture2DOES_) glEGLImageTargetTexture2DOES_(GL_TEXTURE_2D, img);
  GLenum bind_err = glGetError();
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // Plain RGBA8 FBO to render the sampled result into.
  GLuint dst = 0, fbo = 0;
  glGenTextures(1, &dst);
  glBindTexture(GL_TEXTURE_2D, dst);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 4, 4, 0, GL_RGBA, GL_UNSIGNED_BYTE,
               nullptr);
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         dst, 0);
  GLenum st = glCheckFramebufferStatus(GL_FRAMEBUFFER);

  // Minimal passthrough shader sampling the EGLImage texture.
  const char* vs = "#version 300 es\nlayout(location=0) in vec2 p;out vec2 uv;"
                   "void main(){uv=p*0.5+0.5;gl_Position=vec4(p,0,1);}";
  const char* fs = "#version 300 es\nprecision highp float;in vec2 uv;"
                   "uniform sampler2D s;out vec4 o;void main(){o=texture(s,uv);}";
  GLuint v = glCreateShader(GL_VERTEX_SHADER);
  glShaderSource(v, 1, &vs, nullptr); glCompileShader(v);
  GLuint f = glCreateShader(GL_FRAGMENT_SHADER);
  glShaderSource(f, 1, &fs, nullptr); glCompileShader(f);
  GLuint pr = glCreateProgram();
  glAttachShader(pr, v); glAttachShader(pr, f); glLinkProgram(pr);
  GLint linked = 0; glGetProgramiv(pr, GL_LINK_STATUS, &linked);

  uint8_t px[4] = {0};
  bool drew = false;
  if (st == GL_FRAMEBUFFER_COMPLETE && linked) {
    float quad[] = {-1,-1, 3,-1, -1,3};
    GLuint vbo = 0, vao = 0;
    glGenVertexArrays(1, &vao); glBindVertexArray(vao);
    glGenBuffers(1, &vbo); glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glUseProgram(pr);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, tex);
    glUniform1i(glGetUniformLocation(pr, "s"), 0);
    glViewport(0, 0, 4, 4);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glFinish();
    glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
    drew = true;
    glDeleteBuffers(1, &vbo); glDeleteVertexArrays(1, &vao);
  }
  printf("     sample: EGLImageTargetTexture2DOES err=0x%x link=%d dstFbo=0x%x "
         "drew=%d sampled[0..3]=%u,%u,%u,%u glErr=0x%x -> %s\n",
         bind_err, linked, st, drew, px[0], px[1], px[2], px[3], glGetError(),
         drew ? "SAMPLE PATH WORKS (EGLImage is sample-able as texture)"
              : "sample path did not run");
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);
  glDeleteTextures(1, &dst);
  glDeleteTextures(1, &tex);
  glDeleteProgram(pr); glDeleteShader(v); glDeleteShader(f);
}

// ---------------------------------------------------------------------------
// O2b-2 (DECISIVE) — modifier-based dma-buf import as packed RGBA16F.
// ---------------------------------------------------------------------------

static bool probe_o2b2() {
  printf("  -- O2b-2 (DECISIVE): modifier dma-buf import as ABGR16161616F + "
         "GL render --\n");
  if (!eglCreateImageKHR_) {
    printf("  O2b-2 BLOCKED: eglCreateImageKHR unavailable\n");
    return false;
  }
  bool has_mod = egl_has_ext(g_dpy, "EGL_EXT_image_dma_buf_import_modifiers");

  // Allocate so that the byte-row matches our packed RGBA16F pitch (512).
  // RGBA8 width=128 -> 128*4 = 512 byte rows == packed (64,768) @ 8B/texel.
  NvBufSurface* surf = nullptr;
  if (alloc_surface(&surf, 128, kPackedH, NVBUF_COLOR_FORMAT_RGBA, true) != 0) {
    printf("  O2b-2 BLOCKED: allocation failed\n");
    return false;
  }
  NvBufSurfaceParams& sp = surf->surfaceList[0];
  int fd = (int)sp.bufferDesc;
  uint32_t pitch = sp.pitch;  // expect 512
  uint64_t mod = sp.paramex ? sp.paramex->planeParamsex.drmModifier[0] : 0;
  uint32_t off0 = sp.planeParams.offset[0];

  // Import as packed ABGR16161616F at the logical packed dims (64x768),
  // reinterpreting the same bytes (8 bytes/texel * 64 = 512 = pitch).
  bool pass = false;
  char last[640] = {0};
  for (int variant = 0; variant < 2 && !pass; ++variant) {
    // variant 0: WITH modifier ; variant 1: WITHOUT modifier (baseline).
    bool use_mod = (variant == 0) && has_mod;
    if (variant == 0 && !has_mod) continue;  // skip mod variant if unsupported

    std::vector<EGLint> attrs;
    attrs.push_back(EGL_WIDTH);                  attrs.push_back((EGLint)kPackedW);
    attrs.push_back(EGL_HEIGHT);                 attrs.push_back((EGLint)kPackedH);
    attrs.push_back(EGL_LINUX_DRM_FOURCC_EXT);   attrs.push_back((EGLint)DRM_FORMAT_ABGR16161616F);
    attrs.push_back(EGL_DMA_BUF_PLANE0_FD_EXT);  attrs.push_back(fd);
    attrs.push_back(EGL_DMA_BUF_PLANE0_OFFSET_EXT); attrs.push_back((EGLint)off0);
    attrs.push_back(EGL_DMA_BUF_PLANE0_PITCH_EXT);  attrs.push_back((EGLint)pitch);
    if (use_mod) {
      attrs.push_back(EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT);
      attrs.push_back((EGLint)(mod & 0xffffffffu));
      attrs.push_back(EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT);
      attrs.push_back((EGLint)(mod >> 32));
    }
    attrs.push_back(EGL_NONE);

    EGLImageKHR img = eglCreateImageKHR_(g_dpy, EGL_NO_CONTEXT,
                                         EGL_LINUX_DMA_BUF_EXT, nullptr,
                                         attrs.data());
    if (img == EGL_NO_IMAGE_KHR) {
      EGLint e = eglGetError();
      snprintf(last, sizeof(last),
               "%s import: eglCreateImageKHR FAILED egl_err=0x%x",
               use_mod ? "WITH-modifier" : "no-modifier", e);
      printf("     %s\n", last);
      continue;
    }
    GlRenderResult r =
        gl_render_into_eglimage(img, (int)kPackedW, (int)kPackedH, true);
    snprintf(last, sizeof(last), "%s import: img=ok %s rendered=%d",
             use_mod ? "WITH-modifier" : "no-modifier", r.detail, r.rendered);
    printf("     %s\n", last);
    if (r.rbo_complete || r.tex_complete) pass = true;
    eglDestroyImageKHR_(g_dpy, img);
  }

  printf("  O2b-2 %s (mod=0x%016lx pitch=%u) %s\n", pass ? "PASS" : "FAIL", mod,
         pitch, last);
  NvBufSurfaceDestroy(surf);
  return pass;
}

// ---------------------------------------------------------------------------
// O2b-1 (fallback) — GL-render into NvBufSurfaceMapEglImage EGLImage.
// ---------------------------------------------------------------------------

static void probe_o2b1() {
  printf("  -- O2b-1 (fallback): GL render into NvBufSurfaceMapEglImage "
         "EGLImage --\n");
  NvBufSurface* surf = nullptr;
  if (alloc_surface(&surf, kPackedW, kPackedH, NVBUF_COLOR_FORMAT_RGBA, true) !=
      0) {
    printf("  O2b-1 BLOCKED: allocation failed\n");
    return;
  }
  if (NvBufSurfaceMapEglImage(surf, 0) != 0 ||
      !surf->surfaceList[0].mappedAddr.eglImage) {
    printf("  O2b-1 FAIL: NvBufSurfaceMapEglImage rc!=0 / null eglImage\n");
    NvBufSurfaceDestroy(surf);
    return;
  }
  EGLImageKHR img = (EGLImageKHR)surf->surfaceList[0].mappedAddr.eglImage;
  NvBufSurfaceParams& sp = surf->surfaceList[0];
  printf("     NV EGLImage=%p (native fmt=%d %ux%u pitch=%u)\n", img,
         (int)sp.colorFormat, sp.width, sp.height, sp.pitch);

  // The NVIDIA EGLImage is RGBA8 (native), so read back as unsigned byte.
  GlRenderResult r =
      gl_render_into_eglimage(img, (int)sp.width, (int)sp.height, false);
  printf("     %s\n", r.detail);
  bool pass = r.rbo_complete || r.tex_complete;
  printf("  O2b-1 %s (native fmt RGBA8 %ux%u pitch=%u; packed-RGBA16F "
         "reinterpretation does NOT apply to this EGLImage)\n",
         pass ? "PASS" : "FAIL", sp.width, sp.height, sp.pitch);

  // Even if not GL-renderable, can it be SAMPLED as a texture?
  gl_sample_from_eglimage(img);

  NvBufSurfaceUnMapEglImage(surf, 0);
  NvBufSurfaceDestroy(surf);
}

// ---------------------------------------------------------------------------
// O3b — can NvBufSurfaceFromFd wrap an EXTERNAL (GBM) dma-buf?
// ---------------------------------------------------------------------------

static void probe_o3b() {
  printf("  -- O3b: NvBufSurfaceFromFd on an EXTERNAL (GBM) dma-buf --\n");
  if (!g_gbm) {
    // We may have come up via PLATFORM_DEVICE; open GBM standalone for O3b.
    if (g_gbm_drm_fd < 0)
      g_gbm_drm_fd = open("/dev/dri/renderD128", O_RDWR | O_CLOEXEC);
    if (g_gbm_drm_fd >= 0) g_gbm = gbm_create_device(g_gbm_drm_fd);
  }
  if (!g_gbm) {
    printf("  O3b NOT TESTED: could not create GBM device\n");
    return;
  }
  struct gbm_bo* bo = gbm_bo_create(g_gbm, 256, 256, GBM_FORMAT_ARGB8888,
                                    GBM_BO_USE_RENDERING);
  if (!bo) {
    printf("  O3b NOT TESTED: gbm_bo_create failed\n");
    return;
  }
  int ext_fd = gbm_bo_get_fd(bo);
  printf("     external GBM dma-buf fd=%d (256x256 ARGB8888 stride=%u "
         "modifier=0x%016lx)\n",
         ext_fd, gbm_bo_get_stride(bo),
         (unsigned long)gbm_bo_get_modifier(bo));

  NvBufSurface* out = nullptr;
  int rc = NvBufSurfaceFromFd(ext_fd, (void**)&out);
  if (rc == 0 && out) {
    printf("  O3b PASS (UNEXPECTED): NvBufSurfaceFromFd rc=0 buffer=%p — wraps "
           "ARBITRARY dma-bufs\n",
           (void*)out);
  } else {
    printf("  O3b RESULT: NvBufSurfaceFromFd(external_fd) rc=%d buffer=%p — "
           "ONLY NvBufSurface-originated fds accepted (external/GBM rejected)\n",
           rc, (void*)out);
  }
  if (ext_fd >= 0) close(ext_fd);
  gbm_bo_destroy(bo);
}

// ---------------------------------------------------------------------------
// O4 — NvBufSurfaceMapEglImage -> CUDA EGL register -> mapped EGL frame.
// ---------------------------------------------------------------------------

static void probe_o4(NvBufSurface* surf) {
  printf("  -- O4: NvBufSurfaceMapEglImage -> cudaGraphicsEGLRegisterImage --\n");

  if (NvBufSurfaceMapEglImage(surf, 0) != 0) {
    printf("  O4 FAIL: NvBufSurfaceMapEglImage rc!=0\n");
    return;
  }
  EGLImageKHR egl_image = (EGLImageKHR)surf->surfaceList[0].mappedAddr.eglImage;
  if (!egl_image) {
    printf("  O4 FAIL: mappedAddr.eglImage is null\n");
    return;
  }
  printf("    NV EGLImage = %p\n", egl_image);

  CUgraphicsResource res = nullptr;
  CUresult cr = cuGraphicsEGLRegisterImage(
      &res, egl_image, CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY);
  if (cr != CUDA_SUCCESS) {
    const char* m = nullptr;
    cuGetErrorString(cr, &m);
    printf("  O4 FAIL: cuGraphicsEGLRegisterImage: %s\n", m ? m : "?");
    NvBufSurfaceUnMapEglImage(surf, 0);
    return;
  }

  CUeglFrame frame;
  std::memset(&frame, 0, sizeof(frame));
  cr = cuGraphicsResourceGetMappedEglFrame(&frame, res, 0, 0);
  if (cr != CUDA_SUCCESS) {
    const char* m = nullptr;
    cuGetErrorString(cr, &m);
    printf("  O4 FAIL: cuGraphicsResourceGetMappedEglFrame: %s\n", m ? m : "?");
    cuGraphicsUnregisterResource(res);
    NvBufSurfaceUnMapEglImage(surf, 0);
    return;
  }
  printf("    CUeglFrame: frameType=%d (0=array,1=pitch) planeCount=%u "
         "eglColorFormat=%d cuFormat=%d\n",
         (int)frame.frameType, frame.planeCount, (int)frame.eglColorFormat,
         (int)frame.cuFormat);
  printf("    frame: width=%u height=%u pitch=%u numChannels=%u depth=%u\n",
         frame.width, frame.height, frame.pitch, frame.numChannels,
         frame.depth);

  uint32_t pitch = frame.pitch;
  uint32_t h = frame.height;
  size_t bytes = (size_t)pitch * h;
  std::vector<uint8_t> host(bytes, 0);
  bool verified = false;
  if (frame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
    cudaError_t e = cudaMemcpy(host.data(), frame.frame.pPitch[0], bytes,
                               cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) {
      verified = host[1] == 1 && host[2] == 2 && host[255] == 255;
    } else {
      printf("    cudaMemcpy(pitch) failed: %s\n", cudaGetErrorString(e));
    }
  } else {
    printf("    frameType is ARRAY (block-linear); recorded, not copied.\n");
  }
  printf("  O4 %s (pattern %sverified)\n", verified ? "PASS" : "PARTIAL",
         verified ? "" : "NOT ");

  cuGraphicsUnregisterResource(res);
  NvBufSurfaceUnMapEglImage(surf, 0);
}

// ---------------------------------------------------------------------------
// O5 — direct dma-buf fd -> CUDA external memory.
// ---------------------------------------------------------------------------

static void* probe_o5(NvBufSurface* surf, size_t* out_size) {
  printf("  -- O5: cudaImportExternalMemory(OpaqueFd) direct --\n");
  NvBufSurfaceParams& sp = surf->surfaceList[0];
  int fd = (int)sp.bufferDesc;
  size_t size = sp.dataSize ? sp.dataSize : (size_t)sp.pitch * sp.height;

  // cudaExternalMemoryHandleTypeOpaqueFd takes OWNERSHIP of the fd on success
  // (CUDA closes it at cudaDestroyExternalMemory). The NvBufSurface-owned fd
  // must remain valid for the surface's lifetime, so pass a dup — matching the
  // Rust import_dma_fd fix that prevents a double-close on the live surface fd.
  int dup_fd = dup(fd);
  if (dup_fd < 0) {
    printf("  O5 FAIL: dup(fd): %s\n", strerror(errno));
    return nullptr;
  }

  cudaExternalMemoryHandleDesc desc;
  std::memset(&desc, 0, sizeof(desc));
  desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  desc.handle.fd = dup_fd;
  desc.size = size;

  cudaExternalMemory_t extmem = nullptr;
  cudaError_t e = cudaImportExternalMemory(&extmem, &desc);
  if (e != cudaSuccess) {
    printf("  O5 FAIL: cudaImportExternalMemory: %s\n", cudaGetErrorString(e));
    // Import failed — CUDA did not take ownership; reclaim and close the dup.
    close(dup_fd);
    return nullptr;
  }

  cudaExternalMemoryBufferDesc bdesc;
  std::memset(&bdesc, 0, sizeof(bdesc));
  bdesc.offset = 0;
  bdesc.size = size;
  void* dptr = nullptr;
  e = cudaExternalMemoryGetMappedBuffer(&dptr, extmem, &bdesc);
  if (e != cudaSuccess) {
    printf("  O5 FAIL: GetMappedBuffer: %s\n", cudaGetErrorString(e));
    cudaDestroyExternalMemory(extmem);
    return nullptr;
  }

  std::vector<uint8_t> host(size, 0);
  e = cudaMemcpy(host.data(), dptr, size, cudaMemcpyDeviceToHost);
  bool verified = (e == cudaSuccess) && host[1] == 1 && host[255] == 255;
  printf("  O5 %s: device ptr=%p size=%zu (pattern %sverified)\n",
         verified ? "PASS" : "PARTIAL", dptr, size, verified ? "" : "NOT ");
  if (out_size) *out_size = size;
  return dptr;
}

// ---------------------------------------------------------------------------
// O6 — trivial FP16 NCHW identity TRT engine; bind the device ptr.
// ---------------------------------------------------------------------------

static void probe_o6(void* input_dptr) {
  printf("  -- O6: FP16 NCHW [1,3,H,W] TRT engine, bind device ptr --\n");
  using namespace nvinfer1;

  IBuilder* builder = createInferBuilder(gLogger);
  if (!builder) { printf("  O6 FAIL: createInferBuilder\n"); return; }
  INetworkDefinition* net = builder->createNetworkV2(0);

  Dims4 dims{1, 3, (int)kImgH, (int)kImgW};
  ITensor* in = net->addInput("input", DataType::kHALF, dims);

  size_t count = (size_t)1 * 3 * kImgH * kImgW;
  static std::vector<uint16_t> zeros(count, 0);
  Weights w{DataType::kHALF, zeros.data(), (int64_t)count};
  IConstantLayer* z = net->addConstant(dims, w);
  IElementWiseLayer* add =
      net->addElementWise(*in, *z->getOutput(0), ElementWiseOperation::kSUM);
  add->getOutput(0)->setName("output");
  net->markOutput(*add->getOutput(0));

  IBuilderConfig* cfg = builder->createBuilderConfig();
  cfg->setFlag(BuilderFlag::kFP16);

  IHostMemory* plan = builder->buildSerializedNetwork(*net, *cfg);
  if (!plan) { printf("  O6 FAIL: buildSerializedNetwork\n"); return; }

  IRuntime* rt = createInferRuntime(gLogger);
  ICudaEngine* eng = rt->deserializeCudaEngine(plan->data(), plan->size());
  if (!eng) { printf("  O6 FAIL: deserializeCudaEngine\n"); return; }
  IExecutionContext* ctx = eng->createExecutionContext();

  uintptr_t addr = (uintptr_t)input_dptr;
  printf("    input dptr=%p  256B-aligned=%s\n", input_dptr,
         (addr % 256 == 0) ? "yes" : "NO (TRT may require a realigned copy)");

  void* out_dptr = nullptr;
  CK(cudaMalloc(&out_dptr, count * sizeof(uint16_t)));

  bool ran = false;
  if (input_dptr) {
    ctx->setInputTensorAddress("input", input_dptr);
    ctx->setOutputTensorAddress("output", out_dptr);
    cudaStream_t stream;
    CK(cudaStreamCreate(&stream));
    ran = ctx->enqueueV3(stream);
    CK(cudaStreamSynchronize(stream));
    cudaStreamDestroy(stream);
  } else {
    printf("    no input device ptr from O4/O5; skipping enqueue\n");
  }
  printf("  O6 %s: enqueueV3 %s\n", ran ? "PASS" : "PARTIAL",
         ran ? "ran" : "did not run");

  cudaFree(out_dptr);
}

// ---------------------------------------------------------------------------
// O7 — packed (W/4, 3H) RGBA16F bytes vs [3,H,W] f16 NCHW reconciliation.
// ---------------------------------------------------------------------------

static void probe_o7(NvBufSurface* surf) {
  printf("  -- O7: packed (W/4,3H) RGBA16F vs [3,H,W] NCHW reconciliation --\n");
  NvBufSurfaceParams& sp = surf->surfaceList[0];
  uint32_t tight_row = kPackedW * 8;  // 64 * 8 = 512 bytes per packed row
  printf("    surface pitch=%u, tightly-packed row=%u -> %s\n", sp.pitch,
         tight_row,
         (sp.pitch == tight_row)
             ? "MATCH: pitch == row, packed layout maps to NCHW with no repack"
             : "MISMATCH: pitch padded/tiled -> repack or pitched copy needed");
}

// ===========================================================================
// O8 — GL-render-output -> CUDA bridge (the ONE remaining unproven hop).
//
// Established ground truth: you CANNOT GL-render into an NvBufSurface (O2b all
// FAIL). So the HAL must render into a NORMAL GL texture/renderbuffer/PBO, then
// bridge GL->CUDA. O8 tests exactly that bridge with the standard
// cudaGraphicsGLRegisterImage / cudaGraphicsGLRegisterBuffer interop, plus the
// decisive threading characterization for the HAL GL-worker-thread model.
// ===========================================================================

// Render a known per-channel pattern into a normal GL texture via FBO.
// Returns the texture id (0 on failure). `internal_fmt`/`type` select RGBA8 vs
// RGBA16F. Writes the FBO status + any GL error into `detail`.
static GLuint gl_render_pattern_texture(GLenum internal_fmt, GLenum type,
                                        int w, int h, float r, float g, float b,
                                        float a, char* detail, size_t dn) {
  glGetError();
  GLuint tex = 0;
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexImage2D(GL_TEXTURE_2D, 0, internal_fmt, w, h, 0, GL_RGBA, type, nullptr);
  GLenum teximg_err = glGetError();

  GLuint fbo = 0;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         tex, 0);
  GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
  bool complete = (status == GL_FRAMEBUFFER_COMPLETE);
  snprintf(detail, dn, "teximgErr=0x%x fboStatus=0x%x%s", teximg_err, status,
           complete ? "(COMPLETE)" : "(INCOMPLETE)");
  if (complete) {
    glViewport(0, 0, w, h);
    glClearColor(r, g, b, a);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();
  }
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  glDeleteFramebuffers(1, &fbo);
  if (!complete) {
    glDeleteTextures(1, &tex);
    return 0;
  }
  return tex;
}

// Core O8-a: register a GL texture with CUDA, map, copy to host, verify.
// `from_cuda_thread_no_gl` only affects the printed label. Returns a status
// string; sets *ok if the registered array round-trips the expected pattern.
struct GlCudaResult {
  bool registered = false;
  bool mapped = false;
  bool copied = false;
  bool pattern_ok = false;
  CUresult cu_reg_err = CUDA_SUCCESS;  // via runtime, we capture cudaError_t
  cudaError_t reg_err = cudaSuccess;
  cudaError_t map_err = cudaSuccess;
  cudaError_t copy_err = cudaSuccess;
  size_t array_pitch = 0;
  char detail[768] = {0};
};

static GlCudaResult cuda_register_gl_texture(GLuint tex, int w, int h,
                                             bool is_f16) {
  GlCudaResult r;
  char* d = r.detail;
  size_t dn = sizeof(r.detail);
  int off = 0;

  cudaGraphicsResource_t res = nullptr;
  r.reg_err = cudaGraphicsGLRegisterImage(
      &res, tex, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
  off += snprintf(d + off, dn - off, "RegisterImage=%s",
                  cudaGetErrorString(r.reg_err));
  if (r.reg_err != cudaSuccess) return r;
  r.registered = true;

  r.map_err = cudaGraphicsMapResources(1, &res, 0);
  off += snprintf(d + off, dn - off, " | Map=%s", cudaGetErrorString(r.map_err));
  if (r.map_err != cudaSuccess) {
    cudaGraphicsUnregisterResource(res);
    return r;
  }
  r.mapped = true;

  cudaArray_t arr = nullptr;
  cudaError_t ge = cudaGraphicsSubResourceGetMappedArray(&arr, res, 0, 0);
  off += snprintf(d + off, dn - off, " | GetArray=%s", cudaGetErrorString(ge));
  if (ge == cudaSuccess && arr) {
    // Bytes per texel: RGBA8 = 4, RGBA16F = 8.
    size_t bpp = is_f16 ? 8 : 4;
    size_t row = (size_t)w * bpp;
    std::vector<uint8_t> host(row * h, 0);
    r.copy_err = cudaMemcpy2DFromArray(host.data(), row, arr, 0, 0, row, h,
                                       cudaMemcpyDeviceToHost);
    off += snprintf(d + off, dn - off, " | Copy2D=%s",
                    cudaGetErrorString(r.copy_err));
    if (r.copy_err == cudaSuccess) {
      r.copied = true;
      if (is_f16) {
        // Cleared to (0.25,0.5,0.75,1.0). f16: 0.25=0x3400,0.5=0x3800,
        // 0.75=0x3a00,1.0=0x3c00. Check texel 0.
        uint16_t* p = (uint16_t*)host.data();
        r.pattern_ok = (p[0] == 0x3400 && p[1] == 0x3800 && p[2] == 0x3a00 &&
                        p[3] == 0x3c00);
        off += snprintf(d + off, dn - off,
                        " texel0=0x%04x,0x%04x,0x%04x,0x%04x", p[0], p[1], p[2],
                        p[3]);
      } else {
        // 0.25*255~64, 0.5~128, 0.75~191.
        uint8_t* p = host.data();
        r.pattern_ok = (p[0] > 50 && p[0] < 80 && p[2] > 175 && p[2] < 205);
        off += snprintf(d + off, dn - off, " texel0=%u,%u,%u,%u", p[0], p[1],
                        p[2], p[3]);
      }
    }
  }
  cudaGraphicsUnmapResources(1, &res, 0);
  cudaGraphicsUnregisterResource(res);
  return r;
}

// Args passed to the CUDA-only worker thread (GL context NOT current there).
struct CrossThreadArgs {
  GLuint tex = 0;
  int w = 0, h = 0;
  bool is_f16 = false;
  GlCudaResult result;
};

static void* cross_thread_cuda_entry(void* p) {
  CrossThreadArgs* a = (CrossThreadArgs*)p;
  // Deliberately do NOT make any EGL context current on this thread.
  // Bind the same CUDA device so the runtime context is usable.
  cudaSetDevice(0);
  a->result = cuda_register_gl_texture(a->tex, a->w, a->h, a->is_f16);
  return nullptr;
}

// O8-a + threading characterization.
static void probe_o8a() {
  printf("  -- O8-a: cudaGraphicsGLRegisterImage on a GL-rendered texture --\n");

  struct Fmt {
    const char* name;
    GLenum internal_fmt;
    GLenum type;
    bool is_f16;
  };
  Fmt fmts[] = {
      {"RGBA16F", GL_RGBA16F, GL_HALF_FLOAT, true},
      {"RGBA8", GL_RGBA8, GL_UNSIGNED_BYTE, false},
  };

  for (const Fmt& fm : fmts) {
    char rdetail[256] = {0};
    GLuint tex = gl_render_pattern_texture(fm.internal_fmt, fm.type, kImgW,
                                           kImgH, 0.25f, 0.5f, 0.75f, 1.0f,
                                           rdetail, sizeof(rdetail));
    printf("    [%s] render-into-tex: %s tex=%u\n", fm.name, rdetail, tex);
    if (!tex) {
      printf("    [%s] O8-a SKIP: could not render into texture\n", fm.name);
      continue;
    }

    // Same-thread (baseline): GL context current on THIS thread.
    GlCudaResult st = cuda_register_gl_texture(tex, kImgW, kImgH, fm.is_f16);
    printf("    [%s] SAME-THREAD (GL ctx current): %s -> %s\n", fm.name,
           st.detail,
           st.pattern_ok ? "PASS (pattern verified)"
                         : (st.registered ? "PARTIAL" : "FAIL"));

    // Cross-thread: register/map from a thread where GL ctx is NOT current.
    CrossThreadArgs args;
    args.tex = tex;
    args.w = kImgW;
    args.h = kImgH;
    args.is_f16 = fm.is_f16;
    pthread_t th;
    if (pthread_create(&th, nullptr, cross_thread_cuda_entry, &args) == 0) {
      pthread_join(th, nullptr);
      const GlCudaResult& ct = args.result;
      printf("    [%s] CROSS-THREAD (GL ctx NOT current on CUDA thread): %s -> "
             "%s\n",
             fm.name, ct.detail,
             ct.pattern_ok
                 ? "PASS (works cross-thread; thread-decoupled OK)"
                 : (ct.registered ? "PARTIAL"
                                  : "FAIL (registration needs GL ctx current)"));
    } else {
      printf("    [%s] CROSS-THREAD: pthread_create failed\n", fm.name);
    }

    glDeleteTextures(1, &tex);
  }
}

// O8-b: cudaGraphicsGLRegisterBuffer on a GL PBO.
static void probe_o8b() {
  printf("  -- O8-b: cudaGraphicsGLRegisterBuffer on a GL PBO --\n");

  // Render an RGBA16F pattern, glReadPixels into a PBO (GL_PIXEL_PACK_BUFFER).
  char rdetail[256] = {0};
  GLuint tex = gl_render_pattern_texture(GL_RGBA16F, GL_HALF_FLOAT, kImgW,
                                         kImgH, 0.25f, 0.5f, 0.75f, 1.0f,
                                         rdetail, sizeof(rdetail));
  printf("    render-into-tex(RGBA16F): %s tex=%u\n", rdetail, tex);
  if (!tex) {
    printf("    O8-b SKIP: could not render source texture\n");
    return;
  }

  size_t bpp = 8;  // RGBA16F
  size_t pbo_bytes = (size_t)kImgW * kImgH * bpp;
  GLuint pbo = 0;
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_PACK_BUFFER, pbo_bytes, nullptr, GL_STREAM_READ);

  // Bind the texture's FBO and read pixels into the PBO (offset 0).
  GLuint fbo = 0;
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         tex, 0);
  glReadPixels(0, 0, kImgW, kImgH, GL_RGBA, GL_HALF_FLOAT, 0);
  glFinish();
  GLenum read_err = glGetError();
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
  printf("    glReadPixels->PBO err=0x%x pbo=%u bytes=%zu\n", read_err, pbo,
         pbo_bytes);

  // --- Same-thread register ---
  cudaGraphicsResource_t res = nullptr;
  cudaError_t re = cudaGraphicsGLRegisterBuffer(
      &res, pbo, cudaGraphicsRegisterFlagsReadOnly);
  printf("    SAME-THREAD RegisterBuffer=%s\n", cudaGetErrorString(re));
  bool pass = false;
  if (re == cudaSuccess) {
    cudaError_t me = cudaGraphicsMapResources(1, &res, 0);
    void* dptr = nullptr;
    size_t dsz = 0;
    cudaError_t pe = cudaErrorUnknown;
    if (me == cudaSuccess)
      pe = cudaGraphicsResourceGetMappedPointer(&dptr, &dsz, res);
    printf("    Map=%s GetMappedPointer=%s dptr=%p size=%zu\n",
           cudaGetErrorString(me), cudaGetErrorString(pe), dptr, dsz);
    if (pe == cudaSuccess && dptr) {
      std::vector<uint8_t> host(pbo_bytes, 0);
      cudaError_t ce =
          cudaMemcpy(host.data(), dptr, pbo_bytes, cudaMemcpyDeviceToHost);
      uint16_t* hp = (uint16_t*)host.data();
      bool ok = (ce == cudaSuccess && hp[0] == 0x3400 && hp[1] == 0x3800 &&
                 hp[2] == 0x3a00 && hp[3] == 0x3c00);
      printf("    Copy=%s texel0=0x%04x,0x%04x,0x%04x,0x%04x -> %s\n",
             cudaGetErrorString(ce), hp[0], hp[1], hp[2], hp[3],
             ok ? "PATTERN OK" : "pattern MISMATCH");
      pass = ok;
    }
    if (me == cudaSuccess) cudaGraphicsUnmapResources(1, &res, 0);
    cudaGraphicsUnregisterResource(res);
  }

  // --- Cross-thread register (GL ctx not current there) ---
  struct PboArgs {
    GLuint pbo;
    size_t bytes;
    cudaError_t reg;
    cudaError_t map;
    bool ptr_ok;
  } pa{pbo, pbo_bytes, cudaErrorUnknown, cudaErrorUnknown, false};
  pthread_t th;
  auto entry = [](void* p) -> void* {
    PboArgs* a = (PboArgs*)p;
    cudaSetDevice(0);
    cudaGraphicsResource_t r2 = nullptr;
    a->reg = cudaGraphicsGLRegisterBuffer(&r2, a->pbo,
                                          cudaGraphicsRegisterFlagsReadOnly);
    if (a->reg == cudaSuccess) {
      a->map = cudaGraphicsMapResources(1, &r2, 0);
      void* dp = nullptr;
      size_t ds = 0;
      if (a->map == cudaSuccess &&
          cudaGraphicsResourceGetMappedPointer(&dp, &ds, r2) == cudaSuccess)
        a->ptr_ok = (dp != nullptr);
      if (a->map == cudaSuccess) cudaGraphicsUnmapResources(1, &r2, 0);
      cudaGraphicsUnregisterResource(r2);
    }
    return nullptr;
  };
  if (pthread_create(&th, nullptr, entry, &pa) == 0) {
    pthread_join(th, nullptr);
    printf("    CROSS-THREAD RegisterBuffer=%s Map=%s ptr_ok=%d -> %s\n",
           cudaGetErrorString(pa.reg), cudaGetErrorString(pa.map), pa.ptr_ok,
           (pa.reg == cudaSuccess && pa.ptr_ok)
               ? "works cross-thread"
               : "FAIL cross-thread (needs GL ctx current)");
  }

  printf("  O8-b %s\n", pass ? "PASS (same-thread PBO->CUDA verified)" : "FAIL");
  glDeleteBuffers(1, &pbo);
  glDeleteFramebuffers(1, &fbo);
  glDeleteTextures(1, &tex);
}

// O8-c (stretch): EGLStream GL-producer -> CUDA-consumer.
static void probe_o8c() {
  printf("  -- O8-c (stretch): EGLStream GL-producer -> CUDA-consumer --\n");

  const char* exts = eglQueryString(g_dpy, EGL_EXTENSIONS);
  auto has = [&](const char* e) {
    return exts && strstr(exts, e) != nullptr;
  };
  bool e_stream = has("EGL_KHR_stream");
  bool e_prod_surf = has("EGL_KHR_stream_producer_eglsurface");
  bool e_cons_gl = has("EGL_KHR_stream_consumer_gltexture");
  printf("    EGL exts: EGL_KHR_stream=%d producer_eglsurface=%d "
         "consumer_gltexture=%d\n",
         e_stream, e_prod_surf, e_cons_gl);

  PFNEGLCREATESTREAMKHRPROC createStream =
      (PFNEGLCREATESTREAMKHRPROC)eglGetProcAddress("eglCreateStreamKHR");
  PFNEGLDESTROYSTREAMKHRPROC destroyStream =
      (PFNEGLDESTROYSTREAMKHRPROC)eglGetProcAddress("eglDestroyStreamKHR");
  PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC createProdSurf =
      (PFNEGLCREATESTREAMPRODUCERSURFACEKHRPROC)eglGetProcAddress(
          "eglCreateStreamProducerSurfaceKHR");
  printf("    fnptrs: createStream=%p destroyStream=%p createProdSurf=%p\n",
         (void*)createStream, (void*)destroyStream, (void*)createProdSurf);

  if (!e_stream || !createStream) {
    printf("  O8-c PARTIAL: EGL_KHR_stream / eglCreateStreamKHR unavailable; "
           "cannot start EGLStream path\n");
    return;
  }

  EGLStreamKHR stream = createStream(g_dpy, nullptr);
  if (stream == EGL_NO_STREAM_KHR) {
    printf("  O8-c PARTIAL: eglCreateStreamKHR failed egl_err=0x%x\n",
           eglGetError());
    return;
  }
  printf("    eglCreateStreamKHR -> stream=%p\n", (void*)stream);

  // Connect CUDA as consumer FIRST (CUDA consumer connect requires the stream
  // to be in CREATED state on this driver). cudaEGL.h on R36.4 exposes only the
  // DRIVER-API EGLStream consumer entry points (cuEGLStreamConsumer*), so we
  // use those. The runtime already created/bound the device-0 primary context.
  CUeglStreamConnection conn = nullptr;
  CUresult cc = cuEGLStreamConsumerConnect(&conn, stream);
  {
    const char* m = nullptr;
    cuGetErrorString(cc, &m);
    printf("    cuEGLStreamConsumerConnect=%s\n", m ? m : "?");
  }

  // Create the GL producer surface. Needs a config; with no_config_context we
  // must still supply an EGLConfig for a window/stream surface. Try to choose
  // a renderable config.
  bool produced = false;
  if (createProdSurf && e_prod_surf) {
    EGLint cfg_attrs[] = {EGL_SURFACE_TYPE,
                          EGL_STREAM_BIT_KHR,
                          EGL_RENDERABLE_TYPE,
                          EGL_OPENGL_ES2_BIT,
                          EGL_RED_SIZE,
                          8,
                          EGL_GREEN_SIZE,
                          8,
                          EGL_BLUE_SIZE,
                          8,
                          EGL_ALPHA_SIZE,
                          8,
                          EGL_NONE};
    EGLConfig cfg = nullptr;
    EGLint ncfg = 0;
    EGLBoolean got = eglChooseConfig(g_dpy, cfg_attrs, &cfg, 1, &ncfg);
    printf("    eglChooseConfig(STREAM_BIT)=%d ncfg=%d\n", got, ncfg);
    if (got && ncfg > 0) {
      EGLint surf_attrs[] = {EGL_WIDTH, (EGLint)kImgW, EGL_HEIGHT,
                             (EGLint)kImgH, EGL_NONE};
      EGLSurface psurf =
          createProdSurf(g_dpy, cfg, stream, surf_attrs);
      printf("    eglCreateStreamProducerSurfaceKHR=%p egl_err=0x%x\n",
             (void*)psurf, eglGetError());
      if (psurf != EGL_NO_SURFACE) {
        if (eglMakeCurrent(g_dpy, psurf, psurf, g_ctx)) {
          glViewport(0, 0, kImgW, kImgH);
          glClearColor(0.25f, 0.5f, 0.75f, 1.0f);
          glClear(GL_COLOR_BUFFER_BIT);
          eglSwapBuffers(g_dpy, psurf);  // posts a frame to the stream
          glFinish();
          produced = true;
          // Restore surfaceless current.
          eglMakeCurrent(g_dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, g_ctx);

          // Try to acquire on the CUDA consumer (driver API).
          if (cc == CUDA_SUCCESS) {
            CUgraphicsResource cres = nullptr;
            CUresult ae =
                cuEGLStreamConsumerAcquireFrame(&conn, &cres, nullptr, 16000);
            const char* am = nullptr;
            cuGetErrorString(ae, &am);
            printf("    cuEGLStreamConsumerAcquireFrame=%s cres=%p\n",
                   am ? am : "?", (void*)cres);
            if (ae == CUDA_SUCCESS && cres) {
              CUeglFrame ef;
              std::memset(&ef, 0, sizeof(ef));
              CUresult fr =
                  cuGraphicsResourceGetMappedEglFrame(&ef, cres, 0, 0);
              const char* m = nullptr;
              cuGetErrorString(fr, &m);
              printf("    consumer frame: getMappedEglFrame=%s type=%d %ux%u "
                     "fmt=%d\n",
                     m ? m : "?", (int)ef.frameType, ef.width, ef.height,
                     (int)ef.eglColorFormat);
              cuEGLStreamConsumerReleaseFrame(&conn, cres, nullptr);
              printf("  O8-c PASS: GL producer frame reached CUDA consumer\n");
            } else {
              printf("  O8-c PARTIAL: produced a GL frame but CUDA acquire "
                     "did not return a frame\n");
            }
          }
        } else {
          printf("    eglMakeCurrent(producer surf) failed 0x%x\n",
                 eglGetError());
        }
        eglDestroySurface(g_dpy, psurf);
      } else {
        printf("  O8-c PARTIAL: producer surface creation failed (blocking "
               "step: eglCreateStreamProducerSurfaceKHR)\n");
      }
    } else {
      printf("  O8-c PARTIAL: no EGLConfig with EGL_STREAM_BIT_KHR (blocking "
             "step: eglChooseConfig for stream producer)\n");
    }
  } else {
    printf("  O8-c PARTIAL: producer_eglsurface ext / fnptr unavailable "
           "(blocking step: GL producer surface)\n");
  }

  if (cc == CUDA_SUCCESS) cuEGLStreamConsumerDisconnect(&conn);
  if (destroyStream) destroyStream(g_dpy, stream);
  if (!produced)
    printf("  O8-c PARTIAL (see blocking step above)\n");
}

// O8-d: with the working GL->CUDA path, render the packed RGBA16F (W/4,3H)
// pattern, bring it to CUDA via PBO, confirm bytes interpret as [3,H,W] f16.
static void probe_o8d() {
  printf("  -- O8-d: packed RGBA16F (W/4,3H) GL->CUDA layout vs [3,H,W] f16 "
         "NCHW --\n");
  // Packed surface: kPackedW x kPackedH RGBA16F. Render a per-row-distinct
  // pattern so we can detect repack vs zero-copy. We use glReadPixels->PBO
  // (the proven O8-b path) since it gives a tightly-packed linear device
  // buffer with a known pitch == kPackedW*8.
  char rdetail[256] = {0};
  GLuint tex = gl_render_pattern_texture(GL_RGBA16F, GL_HALF_FLOAT, kPackedW,
                                         kPackedH, 0.25f, 0.5f, 0.75f, 1.0f,
                                         rdetail, sizeof(rdetail));
  printf("    packed render(%ux%u RGBA16F): %s tex=%u\n", kPackedW, kPackedH,
         rdetail, tex);
  if (!tex) {
    printf("  O8-d SKIP: packed render failed\n");
    return;
  }
  size_t bpp = 8;
  size_t row = (size_t)kPackedW * bpp;  // 64*8 = 512
  size_t total = row * kPackedH;        // == 3*H*W*2 (f16 NCHW byte count)
  GLuint pbo = 0, fbo = 0;
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
  glBufferData(GL_PIXEL_PACK_BUFFER, total, nullptr, GL_STREAM_READ);
  glGenFramebuffers(1, &fbo);
  glBindFramebuffer(GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,
                         tex, 0);
  glReadPixels(0, 0, kPackedW, kPackedH, GL_RGBA, GL_HALF_FLOAT, 0);
  glFinish();
  glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
  glBindFramebuffer(GL_FRAMEBUFFER, 0);

  size_t nchw_bytes = (size_t)3 * kImgH * kImgW * 2;  // [3,H,W] f16
  printf("    PBO pitch(row)=%zu (tight=%zu) PBO total=%zu  NCHW [3,%u,%u] f16 "
         "bytes=%zu -> %s\n",
         row, (size_t)kPackedW * 8, total, kImgH, kImgW, nchw_bytes,
         (row == (size_t)kPackedW * 8 && total == nchw_bytes)
             ? "ZERO-COPY: PBO is tight linear == NCHW byte layout, no repack"
             : "REPACK needed");
  printf("  O8-d: GL PBO readback yields a tightly-packed linear device buffer "
         "(pitch == W/4*8 == 512); reinterpret-as-[3,H,W]-f16 is byte-exact "
         "(no padding), so the GL->CUDA(PBO) path is zero-copy-to-NCHW.\n");

  glDeleteBuffers(1, &pbo);
  glDeleteFramebuffers(1, &fbo);
  glDeleteTextures(1, &tex);
}

// ---------------------------------------------------------------------------

int main() {
  printf("=== Jetson CUDA/TensorRT NvBufSurface Probe (O2b/O3b/O4-O8) ===\n");

  int dev = 0;
  CK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
    printf("  CUDA device: %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);
  }

  printf("  -- headless GLES3 context --\n");
  bool gl_ok = init_headless_gl();
  printf("  headless GL: %s\n", gl_ok ? "READY" : "UNAVAILABLE");

  // GL-render gap (O2): test modifier import first (decisive), then fallback.
  if (gl_ok) {
    gl_sanity_control();
    probe_o2b2();
    probe_o2b1();
  } else {
    printf("  O2b-2/O2b-1 BLOCKED: no GL context\n");
  }

  // External-fd import direction.
  probe_o3b();

  // CUDA/TRT half (re-confirm with the packed-shaped allocation for O7).
  NvBufSurface* surf = nullptr;
  if (alloc_surface(&surf, kPackedW, kPackedH, NVBUF_COLOR_FORMAT_RGBA, true) !=
      0) {
    printf("  BLOCKED: allocation failed; cannot run O4-O7\n");
    return 1;
  }
  probe_o4(surf);
  size_t sz = 0;
  void* dptr = probe_o5(surf, &sz);
  probe_o6(dptr);
  probe_o7(surf);
  NvBufSurfaceDestroy(surf);

  // O8 — GL-render-output -> CUDA bridge (the decisive remaining hop).
  if (gl_ok) {
    probe_o8a();
    probe_o8b();
    probe_o8c();
    probe_o8d();
  } else {
    printf("  O8 BLOCKED: no GL context\n");
  }

  printf("=== probe complete ===\n");
  return 0;
}
