// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// S3: CPU access through a staging texture. Read and write latency by
// size and format, DO_NOT_WAIT behaviour, and the staging row pitch table
// that decides whether a stride can be a constant or must be queried.
#include "probe.h"

#include <cstring>

struct SizeCase {
  uint32_t w, h;
  const char* name;
};
static const SizeCase kSizes[] = {{640, 480, "640x480"}, {1280, 720, "720p"}, {1920, 1080, "1080p"}, {3840, 2160, "4K"}};

void run_s3(GlSession& s) {
  Quad quad;
  quad.init();
  std::string log;
  GLuint p_grad = compile_program(kVertexShader, kGradientFragment, &log);

  for (DXGI_FORMAT fmt : {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT}) {
    for (const SizeCase& sz : kSizes) {
      HRESULT hr;
      ComPtr<ID3D11Texture2D> tex = create_tex(s.d3d, sz.w, sz.h, fmt,
                                               D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE, 0, &hr);
      if (!tex) {
        report("S3.1", Verdict::Fail, "%s %s create: %s", fmt_name(fmt), sz.name, hr_str(hr));
        continue;
      }
      ComPtr<ID3D11Texture2D> st_read = create_staging_like(s.d3d, tex.Get(), D3D11_CPU_ACCESS_READ);
      ComPtr<ID3D11Texture2D> st_write = create_staging_like(s.d3d, tex.Get(), D3D11_CPU_ACCESS_WRITE);
      uint32_t row_bytes = sz.w * bytes_per_pixel(fmt);
      std::vector<uint8_t> host((size_t)row_bytes * sz.h, 0x5A);
      char what[128];

      // Read: CopyResource + Map(READ) + row copy + Unmap, GPU idle.
      Stats st = time_loop(g_opt.iters / 2 + 1, [&] {
        s.d3d.ctx->CopyResource(st_read.Get(), tex.Get());
        D3D11_MAPPED_SUBRESOURCE m;
        if (SUCCEEDED(s.d3d.ctx->Map(st_read.Get(), 0, D3D11_MAP_READ, 0, &m))) {
          for (uint32_t r = 0; r < sz.h; r++)
            memcpy(host.data() + (size_t)r * row_bytes, (uint8_t*)m.pData + (size_t)r * m.RowPitch, row_bytes);
          s.d3d.ctx->Unmap(st_read.Get(), 0);
        }
      });
      snprintf(what, sizeof what, "%s %-8s read: CopyResource+Map+memcpy (GPU idle)", fmt_name(fmt), sz.name);
      report_stats("S3.1", what, st);

      // Read right after a GL render into the texture (EGLImage route), so
      // the Map stalls on the GPU work the way a map() after convert() would.
      Import im = import_texture(s, Route::EglImage, tex.Get(), gl_internal_format_for(fmt));
      GLuint fbo = 0;
      if (im.ok && p_grad) {
        glGenFramebuffers(1, &fbo);
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, im.tex, 0);
        glViewport(0, 0, (GLsizei)sz.w, (GLsizei)sz.h);
        glUseProgram(p_grad);
        glUniform1f(glGetUniformLocation(p_grad, "denom"), fmt == DXGI_FORMAT_R8G8B8A8_UNORM ? 255.0f : 256.0f);
        Stats st2 = time_loop(g_opt.iters / 2 + 1, [&] {
          quad.draw();
          glFlush();
          s.d3d.ctx->CopyResource(st_read.Get(), tex.Get());
          D3D11_MAPPED_SUBRESOURCE m;
          if (SUCCEEDED(s.d3d.ctx->Map(st_read.Get(), 0, D3D11_MAP_READ, 0, &m))) {
            for (uint32_t r = 0; r < sz.h; r++)
              memcpy(host.data() + (size_t)r * row_bytes, (uint8_t*)m.pData + (size_t)r * m.RowPitch, row_bytes);
            s.d3d.ctx->Unmap(st_read.Get(), 0);
          }
        });
        snprintf(what, sizeof what, "%s %-8s GL draw + glFlush + read (stall included)", fmt_name(fmt), sz.name);
        report_stats("S3.2", what, st2);
        // The same render alone, for subtraction.
        Stats st3 = time_loop(g_opt.iters / 2 + 1, [&] {
          quad.draw();
          glFinish();
        });
        snprintf(what, sizeof what, "%s %-8s GL draw + glFinish alone", fmt_name(fmt), sz.name);
        report_stats("S3.2", what, st3);
        // Baseline: what the HAL does today on Windows, glReadPixels into a
        // PBO and a CPU map of that PBO.
        {
          GLint read_type = 0;
          glGetIntegerv(GL_IMPLEMENTATION_COLOR_READ_TYPE, &read_type);
          GLenum type = fmt == DXGI_FORMAT_R8G8B8A8_UNORM ? GL_UNSIGNED_BYTE
                        : (read_type == GL_HALF_FLOAT ? GL_HALF_FLOAT : GL_FLOAT);
          size_t px_bytes = type == GL_UNSIGNED_BYTE ? 4 : (type == GL_HALF_FLOAT ? 8 : 16);
          size_t pbo_size = (size_t)sz.w * sz.h * px_bytes;
          GLuint pbo;
          glGenBuffers(1, &pbo);
          glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo);
          glBufferData(GL_PIXEL_PACK_BUFFER, (GLsizeiptr)pbo_size, nullptr, GL_STREAM_READ);
          std::vector<uint8_t> host2(pbo_size);
          glPixelStorei(GL_PACK_ALIGNMENT, 1);
          Stats st4 = time_loop(g_opt.iters / 2 + 1, [&] {
            quad.draw();
            glReadPixels(0, 0, (GLsizei)sz.w, (GLsizei)sz.h, GL_RGBA, type, nullptr);
            void* p = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, (GLsizeiptr)pbo_size, GL_MAP_READ_BIT);
            if (p) {
              memcpy(host2.data(), p, pbo_size);
              glUnmapBuffer(GL_PIXEL_PACK_BUFFER);
            }
          });
          GLenum ge = glGetError();
          snprintf(what, sizeof what, "%s %-8s baseline: GL draw + glReadPixels(PBO, %s) + map + memcpy%s",
                   fmt_name(fmt), sz.name,
                   type == GL_UNSIGNED_BYTE ? "u8" : (type == GL_HALF_FLOAT ? "f16" : "f32"),
                   ge ? " [GL ERROR]" : "");
          report_stats("S3.2", what, st4);
          glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
          glDeleteBuffers(1, &pbo);
        }
        // Correctness of the post-render read once.
        std::vector<uint8_t> back;
        readback_tex(s.d3d, tex.Get(), back, nullptr, st_read.Get());
        auto want = make_gradient(fmt, sz.w, sz.h, false);
        size_t bad = count_mismatch(back.data(), want.data(), want.size());
        report("S3.2", bad == 0 ? Verdict::Pass : Verdict::Fail, "%s %-8s bytes after GL draw + glFlush + staging read: %zu mismatches",
               fmt_name(fmt), sz.name, bad);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glDeleteFramebuffers(1, &fbo);
      }
      import_destroy(s, im);

      // Write: Map(WRITE) staging + memcpy + Unmap + CopyResource to default.
      st = time_loop(g_opt.iters / 2 + 1, [&] {
        D3D11_MAPPED_SUBRESOURCE m;
        if (SUCCEEDED(s.d3d.ctx->Map(st_write.Get(), 0, D3D11_MAP_WRITE, 0, &m))) {
          for (uint32_t r = 0; r < sz.h; r++)
            memcpy((uint8_t*)m.pData + (size_t)r * m.RowPitch, host.data() + (size_t)r * row_bytes, row_bytes);
          s.d3d.ctx->Unmap(st_write.Get(), 0);
        }
        s.d3d.ctx->CopyResource(tex.Get(), st_write.Get());
      });
      snprintf(what, sizeof what, "%s %-8s write: Map(WRITE)+memcpy+CopyResource (CPU side)", fmt_name(fmt), sz.name);
      report_stats("S3.3", what, st);
      st = time_loop(g_opt.iters / 2 + 1, [&] {
        s.d3d.ctx->UpdateSubresource(tex.Get(), 0, nullptr, host.data(), row_bytes, 0);
      });
      snprintf(what, sizeof what, "%s %-8s write: UpdateSubresource (CPU side)", fmt_name(fmt), sz.name);
      report_stats("S3.3", what, st);
      // Write followed by a GPU consumer flush, to include the actual upload.
      st = time_loop(g_opt.iters / 2 + 1, [&] {
        D3D11_MAPPED_SUBRESOURCE m;
        if (SUCCEEDED(s.d3d.ctx->Map(st_write.Get(), 0, D3D11_MAP_WRITE, 0, &m))) {
          for (uint32_t r = 0; r < sz.h; r++)
            memcpy((uint8_t*)m.pData + (size_t)r * m.RowPitch, host.data() + (size_t)r * row_bytes, row_bytes);
          s.d3d.ctx->Unmap(st_write.Get(), 0);
        }
        s.d3d.ctx->CopyResource(tex.Get(), st_write.Get());
        s.d3d.ctx->CopyResource(st_read.Get(), tex.Get());
        D3D11_MAPPED_SUBRESOURCE m2;
        if (SUCCEEDED(s.d3d.ctx->Map(st_read.Get(), 0, D3D11_MAP_READ, 0, &m2))) s.d3d.ctx->Unmap(st_read.Get(), 0);
      });
      snprintf(what, sizeof what, "%s %-8s write + read back (full round trip)", fmt_name(fmt), sz.name);
      report_stats("S3.3", what, st);

      // DO_NOT_WAIT: how often is the copy still in flight right after issue?
      if (fmt == DXGI_FORMAT_R8G8B8A8_UNORM) {
        int still_drawing = 0, polls_total = 0;
        const int N = 50;
        for (int i = 0; i < N; i++) {
          s.d3d.ctx->CopyResource(st_read.Get(), tex.Get());
          D3D11_MAPPED_SUBRESOURCE m;
          HRESULT r = s.d3d.ctx->Map(st_read.Get(), 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &m);
          int polls = 0;
          while (r == DXGI_ERROR_WAS_STILL_DRAWING) {
            polls++;
            r = s.d3d.ctx->Map(st_read.Get(), 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT, &m);
          }
          if (polls) still_drawing++;
          polls_total += polls;
          if (SUCCEEDED(r)) s.d3d.ctx->Unmap(st_read.Get(), 0);
        }
        report("S3.4", Verdict::Info, "%s %-8s Map(DO_NOT_WAIT) right after CopyResource: WAS_STILL_DRAWING on %d/%d, %d polls total",
               fmt_name(fmt), sz.name, still_drawing, N, polls_total);
      }
    }
  }

  // Row pitch table.
  static const uint32_t widths[] = {1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 320, 640, 641, 1280, 1281, 1920, 1921, 3840};
  for (DXGI_FORMAT fmt : {DXGI_FORMAT_R8_UNORM, DXGI_FORMAT_R8G8_UNORM, DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_NV12}) {
    std::string line;
    bool stable = true;
    uint32_t common_align = 0;
    for (uint32_t w : widths) {
      if (fmt == DXGI_FORMAT_NV12 && (w & 1)) continue;
      UINT pitch = 0;
      bool same = true;
      for (int rep = 0; rep < 3; rep++) {
        HRESULT hr;
        ComPtr<ID3D11Texture2D> st = create_tex(s.d3d, w, 8, fmt, 0, 0, &hr, D3D11_USAGE_STAGING, D3D11_CPU_ACCESS_READ);
        if (!st) {
          pitch = 0;
          break;
        }
        D3D11_MAPPED_SUBRESOURCE m;
        if (FAILED(s.d3d.ctx->Map(st.Get(), 0, D3D11_MAP_READ, 0, &m))) break;
        if (rep == 0) pitch = m.RowPitch;
        else if (m.RowPitch != pitch) same = false;
        s.d3d.ctx->Unmap(st.Get(), 0);
      }
      if (!same) stable = false;
      char b[32];
      snprintf(b, sizeof b, " %u:%u", w, pitch);
      line += b;
      uint32_t tight = w * bytes_per_pixel(fmt);
      if (pitch > 0) {
        // Smallest power of two the pitch is a multiple of, across widths.
        uint32_t a = pitch & (~pitch + 1);
        common_align = common_align ? std::min(common_align, a) : a;
        (void)tight;
      }
    }
    report("S3.5", Verdict::Info, "staging RowPitch %s (width:pitch)%s; pitch alignment >= %u bytes%s", fmt_name(fmt), line.c_str(),
           common_align, stable ? "" : "; UNSTABLE across allocations");
  }

  if (p_grad) glDeleteProgram(p_grad);
  quad.destroy();
  gl_clear_errors("S3 end");
}
