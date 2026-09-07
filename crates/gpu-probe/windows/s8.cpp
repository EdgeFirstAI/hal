// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// S8: the D3D12 side. A D3D11 texture and fence opened on a D3D12 device
// through NT handles, the linear texture-to-buffer copy D3D11 lacks, a
// DirectML operator over that buffer, ANGLE's own D3D11on12 platform, and
// a probe-owned D3D12 device wrapped by D3D11On12 and injected into ANGLE
// so textures unwrap to ID3D12Resource without any handle.
#include "probe.h"

#include <d3d12.h>
#include <d3d11on12.h>
#include <DirectML.h>

#include <algorithm>
#include <cstring>

struct D12 {
  ComPtr<ID3D12Device> dev;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12CommandAllocator> alloc;
  ComPtr<ID3D12GraphicsCommandList> list;
  ComPtr<ID3D12Fence> fence;
  UINT64 value = 0;
  HANDLE event = nullptr;
  HRESULT init(IDXGIAdapter* adapter) {
    HRESULT hr = D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&dev));
    if (FAILED(hr)) return hr;
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = dev->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    if (FAILED(hr)) return hr;
    hr = dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    if (FAILED(hr)) return hr;
    hr = dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list));
    if (FAILED(hr)) return hr;
    list->Close();
    hr = dev->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    return hr;
  }
  void begin() {
    alloc->Reset();
    list->Reset(alloc.Get(), nullptr);
  }
  bool submit_and_wait() {
    list->Close();
    ID3D12CommandList* l = list.Get();
    queue->ExecuteCommandLists(1, &l);
    queue->Signal(fence.Get(), ++value);
    fence->SetEventOnCompletion(value, event);
    return WaitForSingleObject(event, 10000) == WAIT_OBJECT_0;
  }
  ComPtr<ID3D12Resource> buffer(UINT64 size, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state) {
    D3D12_HEAP_PROPERTIES hp{};
    hp.Type = heap;
    D3D12_RESOURCE_DESC rd{};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.Format = DXGI_FORMAT_UNKNOWN;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;
    ComPtr<ID3D12Resource> r;
    dev->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr, IID_PPV_ARGS(&r));
    return r;
  }
  ~D12() {
    if (event) CloseHandle(event);
  }
};

static void barrier(ID3D12GraphicsCommandList* l, ID3D12Resource* r, D3D12_RESOURCE_STATES from, D3D12_RESOURCE_STATES to) {
  D3D12_RESOURCE_BARRIER b{};
  b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  b.Transition.pResource = r;
  b.Transition.StateBefore = from;
  b.Transition.StateAfter = to;
  b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  l->ResourceBarrier(1, &b);
}

// Records a texture -> buffer copy at offset 0 with the driver's footprint.
static void record_tex_to_buffer(D12& d, ID3D12Resource* tex, ID3D12Resource* buf, D3D12_PLACED_SUBRESOURCE_FOOTPRINT* fp_out,
                                 UINT64* total_out) {
  D3D12_RESOURCE_DESC td = tex->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
  UINT rows = 0;
  UINT64 rowsize = 0, total = 0;
  d.dev->GetCopyableFootprints(&td, 0, 1, 0, &fp, &rows, &rowsize, &total);
  D3D12_TEXTURE_COPY_LOCATION dst{};
  dst.pResource = buf;
  dst.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  dst.PlacedFootprint = fp;
  D3D12_TEXTURE_COPY_LOCATION src{};
  src.pResource = tex;
  src.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  src.SubresourceIndex = 0;
  d.list->CopyTextureRegion(&dst, 0, 0, 0, &src, nullptr);
  if (fp_out) *fp_out = fp;
  if (total_out) *total_out = total;
}

static UINT64 footprint_total(D12& d, ID3D12Resource* tex, UINT* pitch) {
  D3D12_RESOURCE_DESC td = tex->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT fp{};
  UINT rows = 0;
  UINT64 rowsize = 0, total = 0;
  d.dev->GetCopyableFootprints(&td, 0, 1, 0, &fp, &rows, &rowsize, &total);
  if (pitch) *pitch = fp.Footprint.RowPitch;
  return total;
}

// Compare a pitched readback with a tight reference.
static size_t compare_pitched(const uint8_t* got, UINT pitch, const std::vector<uint8_t>& want, uint32_t row_bytes, uint32_t rows) {
  size_t bad = 0;
  for (uint32_t r = 0; r < rows; r++) bad += count_mismatch(got + (size_t)r * pitch, want.data() + (size_t)r * row_bytes, row_bytes);
  return bad;
}

struct GlTarget8 {
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

typedef HRESULT(WINAPI* PFN_DMLCreateDevice)(ID3D12Device*, DML_CREATE_DEVICE_FLAGS, REFIID, void**);

// Runs DML_OPERATOR_ELEMENT_WISE_IDENTITY over `in` (FLOAT16 [1,H,W,4] with
// a row pitch) into a tight output and compares with `want`.
static void dml_identity_check(D12& d, ID3D12Resource* in, UINT in_pitch, UINT64 in_size, uint32_t W, uint32_t H,
                               const std::vector<uint8_t>& want) {
  HMODULE dml = LoadLibraryA("DirectML.dll");
  PFN_DMLCreateDevice create = dml ? (PFN_DMLCreateDevice)GetProcAddress(dml, "DMLCreateDevice") : nullptr;
  if (!create) {
    report("S8.3", Verdict::Skip, "DirectML.dll not available");
    return;
  }
  ComPtr<IDMLDevice> dmldev;
  HRESULT hr = create(d.dev.Get(), DML_CREATE_DEVICE_FLAG_NONE, IID_PPV_ARGS(&dmldev));
  if (FAILED(hr)) {
    report("S8.3", Verdict::Fail, "DMLCreateDevice: %s", hr_str(hr));
    return;
  }
  {
    DML_FEATURE_LEVEL levels[] = {DML_FEATURE_LEVEL_1_0, DML_FEATURE_LEVEL_2_0, DML_FEATURE_LEVEL_3_0,
#ifdef DML_FEATURE_LEVEL_4_0
                                  DML_FEATURE_LEVEL_4_0,
#endif
#ifdef DML_FEATURE_LEVEL_5_0
                                  DML_FEATURE_LEVEL_5_0,
#endif
    };
    DML_FEATURE_QUERY_FEATURE_LEVELS q{};
    q.RequestedFeatureLevelCount = ARRAYSIZE(levels);
    q.RequestedFeatureLevels = levels;
    DML_FEATURE_DATA_FEATURE_LEVELS fd{};
    dmldev->CheckFeatureSupport(DML_FEATURE_FEATURE_LEVELS, sizeof q, &q, sizeof fd, &fd);
    char path[MAX_PATH];
    GetModuleFileNameA(dml, path, sizeof path);
    report("S8.3", Verdict::Info, "DirectML %s, max feature level 0x%x", path, (unsigned)fd.MaxSupportedFeatureLevel);
  }
  UINT elems_per_row = in_pitch / 2;
  UINT sizes[4] = {1, H, W, 4};
  UINT strides[4] = {H * elems_per_row, elems_per_row, 4, 1};
  UINT64 in_total = ((UINT64)(H - 1) * elems_per_row + (UINT64)(W - 1) * 4 + 4) * 2;
  in_total = (in_total + 3) & ~3ull;
  UINT64 out_total = (UINT64)H * W * 4 * 2;
  DML_BUFFER_TENSOR_DESC in_bt{DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, sizes, strides, in_total, 0};
  DML_TENSOR_DESC in_td{DML_TENSOR_TYPE_BUFFER, &in_bt};
  DML_BUFFER_TENSOR_DESC out_bt{DML_TENSOR_DATA_TYPE_FLOAT16, DML_TENSOR_FLAG_NONE, 4, sizes, nullptr, out_total, 0};
  DML_TENSOR_DESC out_td{DML_TENSOR_TYPE_BUFFER, &out_bt};
  DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC idd{&in_td, &out_td, nullptr};
  DML_OPERATOR_DESC od{DML_OPERATOR_ELEMENT_WISE_IDENTITY, &idd};
  ComPtr<IDMLOperator> op;
  hr = dmldev->CreateOperator(&od, IID_PPV_ARGS(&op));
  ComPtr<IDMLCompiledOperator> cop;
  if (SUCCEEDED(hr)) hr = dmldev->CompileOperator(op.Get(), DML_EXECUTION_FLAG_NONE, IID_PPV_ARGS(&cop));
  ComPtr<IDMLOperatorInitializer> init;
  IDMLCompiledOperator* ops[] = {cop.Get()};
  if (SUCCEEDED(hr)) hr = dmldev->CreateOperatorInitializer(1, ops, IID_PPV_ARGS(&init));
  if (FAILED(hr)) {
    report("S8.3", Verdict::Fail, "DML operator setup: %s", hr_str(hr));
    return;
  }
  DML_BINDING_PROPERTIES ip = init->GetBindingProperties(), ep = cop->GetBindingProperties();
  UINT ndesc = std::max(ip.RequiredDescriptorCount, ep.RequiredDescriptorCount);
  if (ndesc == 0) ndesc = 1;
  D3D12_DESCRIPTOR_HEAP_DESC hd{D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, ndesc, D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE, 0};
  ComPtr<ID3D12DescriptorHeap> heap;
  d.dev->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&heap));
  DML_BINDING_TABLE_DESC btd{init.Get(), heap->GetCPUDescriptorHandleForHeapStart(), heap->GetGPUDescriptorHandleForHeapStart(), ndesc};
  ComPtr<IDMLBindingTable> table;
  hr = dmldev->CreateBindingTable(&btd, IID_PPV_ARGS(&table));
  UINT64 temp_size = std::max(ip.TemporaryResourceSize, ep.TemporaryResourceSize);
  ComPtr<ID3D12Resource> temp, persistent;
  if (temp_size) temp = d.buffer(temp_size, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  if (ep.PersistentResourceSize)
    persistent = d.buffer(ep.PersistentResourceSize, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  ComPtr<IDMLCommandRecorder> rec;
  dmldev->CreateCommandRecorder(IID_PPV_ARGS(&rec));
  ID3D12DescriptorHeap* heaps[] = {heap.Get()};
  // Initialize.
  if (temp && ip.TemporaryResourceSize) {
    DML_BUFFER_BINDING tb{temp.Get(), 0, ip.TemporaryResourceSize};
    DML_BINDING_DESC tbd{DML_BINDING_TYPE_BUFFER, &tb};
    table->BindTemporaryResource(&tbd);
  }
  if (persistent) {
    DML_BUFFER_BINDING pb{persistent.Get(), 0, ep.PersistentResourceSize};
    DML_BINDING_DESC pbd{DML_BINDING_TYPE_BUFFER, &pb};
    table->BindOutputs(1, &pbd);
  }
  d.begin();
  d.list->SetDescriptorHeaps(1, heaps);
  rec->RecordDispatch(d.list.Get(), init.Get(), table.Get());
  d.submit_and_wait();
  // Execute.
  btd.Dispatchable = cop.Get();
  table->Reset(&btd);
  ComPtr<ID3D12Resource> out = d.buffer(out_total, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  DML_BUFFER_BINDING inb{in, 0, in_size};
  DML_BINDING_DESC ind{DML_BINDING_TYPE_BUFFER, &inb};
  table->BindInputs(1, &ind);
  DML_BUFFER_BINDING outb{out.Get(), 0, out_total};
  DML_BINDING_DESC outd{DML_BINDING_TYPE_BUFFER, &outb};
  table->BindOutputs(1, &outd);
  if (temp && ep.TemporaryResourceSize) {
    DML_BUFFER_BINDING tb{temp.Get(), 0, ep.TemporaryResourceSize};
    DML_BINDING_DESC tbd{DML_BINDING_TYPE_BUFFER, &tb};
    table->BindTemporaryResource(&tbd);
  }
  if (persistent) {
    DML_BUFFER_BINDING pb{persistent.Get(), 0, ep.PersistentResourceSize};
    DML_BINDING_DESC pbd{DML_BINDING_TYPE_BUFFER, &pb};
    table->BindPersistentResource(&pbd);
  }
  ComPtr<ID3D12Resource> readback = d.buffer(out_total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  d.begin();
  d.list->SetDescriptorHeaps(1, heaps);
  barrier(d.list.Get(), in, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  rec->RecordDispatch(d.list.Get(), cop.Get(), table.Get());
  barrier(d.list.Get(), out.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
  d.list->CopyResource(readback.Get(), out.Get());
  barrier(d.list.Get(), in, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
  bool done = d.submit_and_wait();
  void* p = nullptr;
  D3D12_RANGE rr{0, (SIZE_T)out_total};
  hr = readback->Map(0, &rr, &p);
  size_t bad = SIZE_MAX;
  if (SUCCEEDED(hr) && p) {
    bad = count_mismatch((const uint8_t*)p, want.data(), std::min((size_t)out_total, want.size()));
    D3D12_RANGE none{0, 0};
    readback->Unmap(0, &none);
  }
  report("S8.3", done && bad == 0 ? Verdict::Pass : Verdict::Fail,
         "DirectML identity over the linearized RGBA16F frame (strided [1,%u,%u,4], pitch %u): %s", H, W, in_pitch,
         bad == SIZE_MAX ? "map failed" : (bad ? "bad bytes" : "0 mismatches"));
}

void run_s8(GlSession& s) {
  bool warp = s.mode == DisplayMode::AngleWarp || s.mode == DisplayMode::InjectWarp;
  AdapterInfo ad;
  if (!pick_adapter(g_opt.adapter, warp, &ad)) {
    report("S8.1", Verdict::Fail, "no adapter for D3D12");
    return;
  }
  D12 d;
  HRESULT hr = d.init(ad.adapter.Get());
  if (FAILED(hr)) {
    report("S8.1", Verdict::Fail, "D3D12CreateDevice on %s: %s", wide_to_utf8(ad.desc.Description).c_str(), hr_str(hr));
    return;
  }
  report("S8.1", Verdict::Pass, "D3D12 device on %s (LUID %s)", wide_to_utf8(ad.desc.Description).c_str(), luid_str(ad.desc.AdapterLuid).c_str());

  std::string log;
  GLuint p_grad = compile_program(kVertexShader, kGradientFragment, &log);
  GLint denom = glGetUniformLocation(p_grad, "denom");
  Quad quad;
  quad.init();
  const uint32_t W = 1920, H = 1080;

  // ---- S8.2 open the D3D11 texture and fence in D3D12, copy linear --------
  ComPtr<ID3D11Fence> fence11;
  ComPtr<ID3D12Fence> fence12;
  UINT64 fv = 0;
  s.d3d.dev5->CreateFence(0, D3D11_FENCE_FLAG_SHARED, IID_PPV_ARGS(&fence11));
  {
    HANDLE fh = nullptr;
    fence11->CreateSharedHandle(nullptr, GENERIC_ALL, nullptr, &fh);
    hr = d.dev->OpenSharedHandle(fh, IID_PPV_ARGS(&fence12));
    CloseHandle(fh);
    report("S8.2", SUCCEEDED(hr) ? Verdict::Pass : Verdict::Fail, "D3D11 fence opened in D3D12: %s", hr_str(hr));
  }
  for (DXGI_FORMAT fmt : {DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_FORMAT_R16G16B16A16_FLOAT}) {
    GlTarget8 t;
    if (!t.init(s, W, H, fmt, D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE)) {
      report("S8.2", Verdict::Fail, "%s target setup failed", fmt_name(fmt));
      continue;
    }
    float dn = fmt == DXGI_FORMAT_R8G8B8A8_UNORM ? 255.0f : 256.0f;
    uint32_t bpp = bytes_per_pixel(fmt);
    auto want = make_gradient(fmt, W, H, false);
    auto draw = [&] {
      glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
      glViewport(0, 0, W, H);
      glUseProgram(p_grad);
      glUniform1f(denom, dn);
      quad.draw();
      glFlush();
      s.d3d.ctx4->Signal(fence11.Get(), ++fv);
    };
    HANDLE th = create_shared_handle(t.tex.Get(), nullptr, &hr);
    ComPtr<ID3D12Resource> res12;
    hr = th ? d.dev->OpenSharedHandle(th, IID_PPV_ARGS(&res12)) : E_FAIL;
    if (th) CloseHandle(th);
    if (FAILED(hr)) {
      report("S8.2", Verdict::Fail, "%s: ID3D12Device::OpenSharedHandle on the texture: %s", fmt_name(fmt), hr_str(hr));
      t.destroy(s);
      continue;
    }
    D3D12_RESOURCE_DESC rd = res12->GetDesc();
    UINT pitch = 0;
    UINT64 total = footprint_total(d, res12.Get(), &pitch);
    report("S8.2", Verdict::Pass, "%s: texture opened in D3D12 (flags 0x%x, layout %d, footprint pitch %u, total %llu)", fmt_name(fmt),
           (unsigned)rd.Flags, (int)rd.Layout, pitch, (unsigned long long)total);
    ComPtr<ID3D12Resource> readback = d.buffer(total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
    ComPtr<ID3D12Resource> linear = d.buffer(total, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
    // Correctness: GL draw, fence, D3D12 waits, copies to readback.
    draw();
    d.queue->Wait(fence12.Get(), fv);
    d.begin();
    record_tex_to_buffer(d, res12.Get(), readback.Get(), nullptr, nullptr);
    bool done = d.submit_and_wait();
    void* p = nullptr;
    D3D12_RANGE rr{0, (SIZE_T)total};
    size_t bad = SIZE_MAX;
    if (done && SUCCEEDED(readback->Map(0, &rr, &p))) {
      bad = compare_pitched((const uint8_t*)p, pitch, want, W * bpp, H);
      D3D12_RANGE none{0, 0};
      readback->Unmap(0, &none);
    }
    report("S8.2", bad == 0 ? Verdict::Pass : Verdict::Fail, "%s: GL render -> fence -> D3D12 CopyTextureRegion to buffer: %s", fmt_name(fmt),
           bad == SIZE_MAX ? "copy did not complete" : (bad ? "bad bytes" : "0 mismatches"));
    // Without the queue wait, to show whether the fence is needed.
    {
      int stale = 0;
      const int N = 100;
      for (int i = 0; i < N; i++) {
        bool grad = (i & 1) == 0;
        glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
        glViewport(0, 0, W, H);
        if (grad) {
          glUseProgram(p_grad);
          glUniform1f(denom, dn);
          quad.draw();
        } else {
          glClearColor(0, 0, 0, 0);
          glClear(GL_COLOR_BUFFER_BIT);
        }
        glFlush();
        d.begin();
        record_tex_to_buffer(d, res12.Get(), readback.Get(), nullptr, nullptr);
        d.submit_and_wait();
        if (SUCCEEDED(readback->Map(0, &rr, &p))) {
          size_t b = compare_pitched((const uint8_t*)p, pitch, want, W * bpp, H);
          bool is_grad = b == 0;
          if (is_grad != grad) stale++;
          D3D12_RANGE none{0, 0};
          readback->Unmap(0, &none);
        }
      }
      report("S8.2", Verdict::Info, "%s: D3D12 copy WITHOUT waiting on the fence: %d stale of %d", fmt_name(fmt), stale, N);
    }
    // Timing: draw + fence + wait + linearize into a DEFAULT buffer.
    Stats st = time_loop(g_opt.iters, [&] {
      draw();
      d.queue->Wait(fence12.Get(), fv);
      d.begin();
      record_tex_to_buffer(d, res12.Get(), linear.Get(), nullptr, nullptr);
      d.submit_and_wait();
    });
    char what[160];
    snprintf(what, sizeof what, "%s 1080p: GL draw + fence + D3D12 queue Wait + CopyTextureRegion to linear buffer + completion", fmt_name(fmt));
    report_stats("S8.T", what, st);
    st = time_loop(g_opt.iters, [&] {
      d.begin();
      record_tex_to_buffer(d, res12.Get(), linear.Get(), nullptr, nullptr);
      d.submit_and_wait();
    });
    snprintf(what, sizeof what, "%s 1080p: D3D12 CopyTextureRegion to linear buffer + completion (no draw)", fmt_name(fmt));
    report_stats("S8.T", what, st);
    // ---- S8.3 DirectML over the linear buffer (RGBA16F) -------------------
    if (fmt == DXGI_FORMAT_R16G16B16A16_FLOAT) {
      draw();
      d.queue->Wait(fence12.Get(), fv);
      d.begin();
      record_tex_to_buffer(d, res12.Get(), linear.Get(), nullptr, nullptr);
      d.submit_and_wait();
      dml_identity_check(d, linear.Get(), pitch, total, W, H, want);
    }
    t.destroy(s);
  }

  // ---- S8.4 ANGLE's own D3D11on12 platform --------------------------------
  {
    GlSession on12;
    if (on12.bring_up(DisplayMode::AngleD3D11On12)) {
      ComPtr<ID3D11On12Device> o12;
      HRESULT h1 = on12.d3d.dev->QueryInterface(IID_PPV_ARGS(&o12));
      ComPtr<ID3D12Device> d12q;
      HRESULT h2 = on12.d3d.dev->QueryInterface(IID_PPV_ARGS(&d12q));
      report("S8.4", Verdict::Pass, "ANGLE D3D11on12 display: %s, FL 0x%x; QI ID3D11On12Device %s, QI ID3D12Device %s", on12.gl_renderer.c_str(),
             on12.d3d.feature_level, hr_str(h1), hr_str(h2));
      quick_import_check(on12, "S8.4", "ANGLE D3D11on12");
      on12.shutdown();
    } else {
      report("S8.4", Verdict::Info, "ANGLE D3D11on12 platform did not come up: %s", on12.last_error.c_str());
    }
    s.restore_current();
  }

  // ---- S8.5 probe-owned D3D12 device, D3D11On12 layer, injected into ANGLE
  {
    ComPtr<ID3D11Device> dev11;
    ComPtr<ID3D11DeviceContext> ctx11;
    IUnknown* queues[] = {d.queue.Get()};
    D3D_FEATURE_LEVEL fl{};
    hr = D3D11On12CreateDevice(d.dev.Get(), D3D11_CREATE_DEVICE_BGRA_SUPPORT, nullptr, 0, queues, 1, 0, &dev11, &ctx11, &fl);
    if (FAILED(hr)) {
      report("S8.5", Verdict::Fail, "D3D11On12CreateDevice: %s", hr_str(hr));
    } else {
      GlSession inj;
      if (!inj.bring_up(warp ? DisplayMode::InjectWarp : DisplayMode::InjectHardware, dev11.Get())) {
        report("S8.5", Verdict::Fail, "ANGLE bring-up on the injected D3D11On12 device failed");
      } else {
        report("S8.5", Verdict::Pass, "ANGLE on an injected D3D11On12 device: %s, FL 0x%x, ES 3.%d", inj.gl_renderer.c_str(),
               inj.d3d.feature_level, inj.es_minor);
        quick_import_check(inj, "S8.5", "injected D3D11On12");
        ComPtr<ID3D11On12Device2> o2;
        HRESULT h2 = dev11->QueryInterface(IID_PPV_ARGS(&o2));
        report("S8.5", SUCCEEDED(h2) ? Verdict::Pass : Verdict::Fail, "ID3D11On12Device2 (UnwrapUnderlyingResource): %s", hr_str(h2));
        if (o2) {
          GlTarget8 t;
          if (t.init(inj, W, H, DXGI_FORMAT_R8G8B8A8_UNORM, 0)) {
            auto want = make_gradient(DXGI_FORMAT_R8G8B8A8_UNORM, W, H, false);
            GLuint pg = compile_program(kVertexShader, kGradientFragment, &log);
            Quad q2;
            q2.init();
            auto draw = [&] {
              glBindFramebuffer(GL_FRAMEBUFFER, t.fbo);
              glViewport(0, 0, W, H);
              glUseProgram(pg);
              glUniform1f(glGetUniformLocation(pg, "denom"), 255.0f);
              q2.draw();
              glFlush();
              inj.d3d.ctx->Flush();
            };
            draw();
            ComPtr<ID3D12Resource> unwrapped;
            HRESULT hu = o2->UnwrapUnderlyingResource(t.tex.Get(), d.queue.Get(), IID_PPV_ARGS(&unwrapped));
            report("S8.5", SUCCEEDED(hu) ? Verdict::Pass : Verdict::Fail, "UnwrapUnderlyingResource on the GL-rendered texture: %s", hr_str(hu));
            if (unwrapped) {
              UINT pitch = 0;
              UINT64 total = footprint_total(d, unwrapped.Get(), &pitch);
              ComPtr<ID3D12Resource> readback = d.buffer(total, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
              d.begin();
              record_tex_to_buffer(d, unwrapped.Get(), readback.Get(), nullptr, nullptr);
              bool done = d.submit_and_wait();
              o2->ReturnUnderlyingResource(t.tex.Get(), 0, nullptr, nullptr);
              void* p = nullptr;
              D3D12_RANGE rr{0, (SIZE_T)total};
              size_t bad = SIZE_MAX;
              if (done && SUCCEEDED(readback->Map(0, &rr, &p))) {
                bad = compare_pitched((const uint8_t*)p, pitch, want, W * 4, H);
                D3D12_RANGE none{0, 0};
                readback->Unmap(0, &none);
              }
              report("S8.5", bad == 0 ? Verdict::Pass : Verdict::Fail,
                     "GL render on 11on12 -> unwrap -> D3D12 copy to buffer (no handle, no fence): %s",
                     bad == SIZE_MAX ? "copy did not complete" : (bad ? "bad bytes" : "0 mismatches"));
              ComPtr<ID3D12Resource> linear = d.buffer(total, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_DEST);
              Stats st = time_loop(g_opt.iters, [&] {
                draw();
                ComPtr<ID3D12Resource> u;
                o2->UnwrapUnderlyingResource(t.tex.Get(), d.queue.Get(), IID_PPV_ARGS(&u));
                d.begin();
                record_tex_to_buffer(d, u.Get(), linear.Get(), nullptr, nullptr);
                d.submit_and_wait();
                o2->ReturnUnderlyingResource(t.tex.Get(), 0, nullptr, nullptr);
              });
              report_stats("S8.T", "11on12: GL draw + Flush + unwrap + D3D12 copy to linear buffer + completion + return", st);
            }
            glDeleteProgram(pg);
            q2.destroy();
          } else {
            report("S8.5", Verdict::Fail, "target on the 11on12 device failed");
          }
          t.destroy(inj);
        }
      }
      inj.shutdown();
      s.restore_current();
    }
  }
  quad.destroy();
  glDeleteProgram(p_grad);
  gl_clear_errors("S8 end");
}
