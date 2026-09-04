// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Entry point: option parsing, ANGLE bring-up, section dispatch.
#include "probe.h"

#include <cstring>

static void usage() {
  printf(
      "d3d11_probe [options]\n"
      "  --warp             run ANGLE on the D3D11 WARP software adapter (S9)\n"
      "  --adapter H:L      pick the hardware adapter by LUID (decimal or 0x hex)\n"
      "  --angle DIR        directory holding libEGL.dll and libGLESv2.dll\n"
      "                     (default: EDGEFIRST_ANGLE_PATH, then target/angle/windows-x64/bin)\n"
      "  --only s1,s5       run only these sections (s0..s8)\n"
      "  --iters N          timing iterations (default 100)\n"
      "  --debug            enable the D3D11 debug layer / ANGLE debug layers\n"
      "  --child NAME       internal: S7 child process opening shared resource NAME\n"
      "  --child-fence NAME internal: S7 child process fence name\n");
}

static bool parse_args(int argc, char** argv) {
  for (int i = 1; i < argc; i++) {
    const char* a = argv[i];
    auto next = [&](std::string* out) {
      if (i + 1 >= argc) return false;
      *out = argv[++i];
      return true;
    };
    std::string v;
    if (!strcmp(a, "--warp")) {
      g_opt.warp = true;
    } else if (!strcmp(a, "--debug")) {
      g_opt.debug_layer = true;
    } else if (!strcmp(a, "--adapter")) {
      if (!next(&g_opt.adapter)) return false;
    } else if (!strcmp(a, "--angle")) {
      if (!next(&g_opt.angle_dir)) return false;
    } else if (!strcmp(a, "--iters")) {
      if (!next(&v)) return false;
      g_opt.iters = atoi(v.c_str());
      if (g_opt.iters < 1) g_opt.iters = 1;
    } else if (!strcmp(a, "--only")) {
      if (!next(&v)) return false;
      size_t p = 0;
      while (p <= v.size()) {
        size_t q = v.find(',', p);
        if (q == std::string::npos) q = v.size();
        if (q > p) g_opt.only.push_back(v.substr(p, q - p));
        p = q + 1;
      }
    } else if (!strcmp(a, "--child")) {
      if (!next(&g_opt.child_name)) return false;
    } else if (!strcmp(a, "--child-fence")) {
      if (!next(&g_opt.child_fence)) return false;
    } else if (!strcmp(a, "--child-kmt")) {
      if (!next(&g_opt.child_kmt)) return false;
    } else if (!strcmp(a, "--s4-control")) {
      g_opt.s4_control = true;
    } else if (!strcmp(a, "--help") || !strcmp(a, "-h")) {
      usage();
      exit(0);
    } else {
      fprintf(stderr, "unknown option %s\n", a);
      return false;
    }
  }
  return true;
}

static void print_flags(UINT f) {
  printf("         creation flags 0x%x:%s%s%s%s%s\n", f,
         (f & D3D11_CREATE_DEVICE_SINGLETHREADED) ? " SINGLETHREADED" : "",
         (f & D3D11_CREATE_DEVICE_DEBUG) ? " DEBUG" : "",
         (f & D3D11_CREATE_DEVICE_BGRA_SUPPORT) ? " BGRA_SUPPORT" : "",
         (f & D3D11_CREATE_DEVICE_VIDEO_SUPPORT) ? " VIDEO_SUPPORT" : "",
         f == 0 ? " (none)" : "");
}

void print_session(const GlSession& s) {
  printf("         mode: %s\n", mode_name(s.mode));
  printf("         GL_RENDERER: %s\n", s.gl_renderer.c_str());
  printf("         GL_VERSION: %s (ES 3.%d context)\n", s.gl_version.c_str(), s.es_minor);
  printf("         D3D11 device: feature level 0x%x, adapter LUID %s\n", s.d3d.feature_level,
         luid_str(s.d3d.luid).c_str());
  print_flags(s.d3d.creation_flags);
  printf("         QueryInterface: Device1=%s Device5=%s Context4=%s\n", s.d3d.dev1 ? "ok" : "no",
         s.d3d.dev5 ? "ok" : "no", s.d3d.ctx4 ? "ok" : "no");
}

int main(int argc, char** argv) {
  if (!parse_args(argc, argv)) {
    usage();
    return 2;
  }
  std::string dir = g_opt.angle_dir.empty() ? default_angle_dir() : g_opt.angle_dir;
  if (dir.empty()) {
    fprintf(stderr, "ANGLE DLLs not found; pass --angle DIR or set EDGEFIRST_ANGLE_PATH\n");
    return 2;
  }
  if (!load_angle(dir)) return 2;

  if (!g_opt.child_name.empty()) return run_s7_child();
  if (g_opt.s4_control) printf("(S4 control child process)\n");

  printf("d3d11_probe: Windows D3D11 texture / ANGLE / CUDA / D3D12 interop probe\n");
  printf("  ANGLE: %s\n", dir.c_str());
  printf("  client EGL extensions: %s\n\n", eglQueryString(EGL_NO_DISPLAY, EGL_EXTENSIONS));

  GlSession s;
  if (!s.bring_up(g_opt.warp ? DisplayMode::AngleWarp : DisplayMode::AngleHardware)) {
    fprintf(stderr, "ANGLE bring-up failed\n");
    return 1;
  }
  print_session(s);
  printf("\n");

  if (g_opt.s4_control) {
    run_s4_control(s);
    s.restore_current();
    s.shutdown();
    return 0;  // the parent reports the exit code; no summary line of its own
  }

  if (section_enabled("s0")) run_s0(s);
  if (section_enabled("s1")) run_s1(s);
  if (section_enabled("s2")) run_s2(s);
  if (section_enabled("s3")) run_s3(s);
  if (section_enabled("s4")) run_s4(s);
  if (section_enabled("s5")) run_s5(s);
  if (section_enabled("s6")) run_s6(s);
  if (section_enabled("s7")) run_s7(s);
  if (section_enabled("s8")) run_s8(s);

  s.restore_current();
  s.shutdown();
  return summary();
}
