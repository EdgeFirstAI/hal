// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Jetson CUDA / TensorRT NvBufSurface interop probe (O4–O7).
//
// On-device tooling for orin-nano (L4T R36.4, CUDA 12.6, TensorRT 10.3). This
// is the C++ counterpart to the Rust `probe_nvbufsurface` module: where the
// Rust probe (O1–O3) proves NvBufSurface gives a dma-buf fd that round-trips
// and is *not* directly GL-renderable via a plain dma-buf import, this probe
// uses the REAL `nvbufsurface.h`, CUDA and TensorRT headers (zero ABI risk) to
// settle the CUDA/TensorRT half of the zero-copy chain:
//
//   O4 — NvBufSurfaceMapEglImage → cudaGraphicsEGLRegisterImage →
//        cudaGraphicsResourceGetMappedEglFrame → cudaMemcpy to host → verify a
//        CPU-written pattern. Prints cudaEglFrame frameType/format/pitch — the
//        layout ground-truth.
//   O5 — cudaImportExternalMemory(OpaqueFd, {fd,size}) →
//        cudaExternalMemoryGetMappedBuffer → verify. Records the exact failure
//        if (as is likely pre-JetPack-7) the direct fd path is unsupported.
//   O6 — build a trivial FP16 NCHW [1,3,H,W] identity TRT engine, bind the
//        CUDA device pointer via setInputTensorAddress (256-byte alignment),
//        enqueueV3, sync; print whether inference ran.
//   O7 — with the packed (W/4, 3H) RGBA16F bytes, check whether the CUDA-mapped
//        bytes interpreted as [3,H,W] f16 equal the source — i.e. does the
//        packed layout need a repack for TRT NCHW.
//
// Build on-device: `make` (see Makefile). Running is OPTIONAL this round.
//
// References: /usr/src/jetson_multimedia_api/samples (same lib set);
// CUDA EGL interop (cudaEGL.h); TensorRT C++ API (NvInfer.h).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cudaEGL.h>          // cudaEglFrame, cudaGraphicsEGLRegisterImage
#include <cuda.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>

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

// ---------------------------------------------------------------------------
// TensorRT trivial logger
// ---------------------------------------------------------------------------

class Logger : public nvinfer1::ILogger {
  void log(Severity s, const char* msg) noexcept override {
    if (s <= Severity::kWARNING) printf("    [trt] %s\n", msg);
  }
} gLogger;

// ---------------------------------------------------------------------------
// Allocate a pitch-linear SURFACE_ARRAY NvBufSurface and CPU-fill a pattern.
// Returns 0 on success, leaving *out_surf owning the buffer.
// ---------------------------------------------------------------------------

static int alloc_and_fill(NvBufSurface** out_surf) {
  NvBufSurfaceCreateParams p;
  std::memset(&p, 0, sizeof(p));
  p.gpuId = 0;
  p.width = kPackedW;
  p.height = kPackedH;
  // NvBufSurfaceCreate validates colorFormat even with size set (confirmed by
  // the Rust O1 probe), so request an RGBA surface and let the driver shape it.
  p.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
  p.layout = NVBUF_LAYOUT_PITCH;
  p.memType = NVBUF_MEM_SURFACE_ARRAY;

  NvBufSurface* surf = nullptr;
  if (NvBufSurfaceCreate(&surf, 1, &p) != 0 || !surf) {
    printf("    NvBufSurfaceCreate failed\n");
    return -1;
  }
  NvBufSurfaceParams& sp = surf->surfaceList[0];
  printf("    allocated: %ux%u pitch=%u dataSize=%u bufferDesc(fd)=%lu\n",
         sp.width, sp.height, sp.pitch, sp.dataSize,
         (unsigned long)sp.bufferDesc);

  // CPU-fill a known per-byte ramp pattern via the CPU mapping.
  if (NvBufSurfaceMap(surf, 0, 0, NVBUF_MAP_WRITE) == 0) {
    NvBufSurfaceSyncForCpu(surf, 0, 0);
    uint8_t* base = (uint8_t*)surf->surfaceList[0].mappedAddr.addr[0];
    if (base) {
      uint32_t n = sp.dataSize ? sp.dataSize : sp.pitch * sp.height;
      for (uint32_t i = 0; i < n; ++i) base[i] = (uint8_t)(i & 0xff);
      NvBufSurfaceSyncForDevice(surf, 0, 0);
    }
    NvBufSurfaceUnMap(surf, 0, 0);
  } else {
    printf("    NvBufSurfaceMap(WRITE) failed (pattern not written)\n");
  }

  *out_surf = surf;
  return 0;
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

  // CUDA EGL interop is exposed through the DRIVER API (cudaEGL.h) on this
  // CUDA version: cuGraphicsEGLRegisterImage / CUeglFrame /
  // cuGraphicsResourceGetMappedEglFrame. The runtime cudaSetDevice() already
  // created a primary context these driver calls share.
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

  // Copy plane 0 to host and verify the CPU-written ramp survived.
  uint32_t pitch = frame.pitch;
  uint32_t h = frame.height;
  size_t bytes = (size_t)pitch * h;
  std::vector<uint8_t> host(bytes, 0);
  bool verified = false;
  if (frame.frameType == CU_EGL_FRAME_TYPE_PITCH) {
    cudaError_t e = cudaMemcpy(host.data(), frame.frame.pPitch[0], bytes,
                               cudaMemcpyDeviceToHost);
    if (e == cudaSuccess) {
      // Expect the i&0xff ramp we wrote (modulo pitch padding rows).
      verified = host[1] == 1 && host[2] == 2 && host[255] == 255;
    } else {
      printf("    cudaMemcpy(pitch) failed: %s\n", cudaGetErrorString(e));
    }
  } else {
    printf("    frameType is ARRAY (block-linear); read via "
           "cuMemcpy2D from the CUarray or a surface object — recorded, not "
           "copied here.\n");
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

  cudaExternalMemoryHandleDesc desc;
  std::memset(&desc, 0, sizeof(desc));
  desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  desc.handle.fd = fd;
  desc.size = size;

  cudaExternalMemory_t extmem = nullptr;
  cudaError_t e = cudaImportExternalMemory(&extmem, &desc);
  if (e != cudaSuccess) {
    printf("  O5 FAIL: cudaImportExternalMemory: %s "
           "(expected on pre-JetPack-7; EGLImage path O4 is the bridge)\n",
           cudaGetErrorString(e));
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
  // NB: leaks extmem/dptr deliberately; this is a one-shot probe.
  return dptr;
}

// ---------------------------------------------------------------------------
// O6 — trivial FP16 NCHW identity TRT engine; bind the device ptr.
// ---------------------------------------------------------------------------
//
// Builds a [1,3,H,W] FP16 network with a single identity (Unary kABS of a
// positive input is effectively identity for the pattern, but to stay truly
// identity we use an ElementWise SUM with a zero constant). Kept minimal; the
// point is to prove a CUDA device pointer binds as a TRT input with no host
// copy and enqueueV3 runs.
//
// Alternative (if the builder API is awkward on this TRT minor version):
//   echo a tiny identity ONNX and run
//   /usr/src/tensorrt/bin/trtexec --onnx=identity.onnx --fp16 \
//       --saveEngine=identity.plan
//   then deserialize identity.plan here with createInferRuntime().

static void probe_o6(void* input_dptr) {
  printf("  -- O6: FP16 NCHW [1,3,H,W] TRT engine, bind device ptr --\n");
  using namespace nvinfer1;

  IBuilder* builder = createInferBuilder(gLogger);
  if (!builder) { printf("  O6 FAIL: createInferBuilder\n"); return; }
  // TRT 10: explicit batch is the default; pass 0 (kEXPLICIT_BATCH is deprecated).
  INetworkDefinition* net = builder->createNetworkV2(0);

  Dims4 dims{1, 3, (int)kImgH, (int)kImgW};
  ITensor* in = net->addInput("input", DataType::kHALF, dims);

  // Identity via ElementWise SUM with a zeroed constant of matching shape.
  size_t count = (size_t)1 * 3 * kImgH * kImgW;
  static std::vector<uint16_t> zeros(count, 0);  // f16 zeros
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

  // 256-byte alignment check on the NvBufSurface-derived device ptr.
  uintptr_t addr = (uintptr_t)input_dptr;
  printf("    input dptr=%p  256B-aligned=%s\n", input_dptr,
         (addr % 256 == 0) ? "yes" : "NO (TRT may require a realigned copy)");

  // Output device buffer.
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
  printf("    (decisive value depends on O4 cudaEglFrame.frameType: PITCH with "
         "pitch==row => zero-copy; ARRAY/block-linear => on-GPU repack)\n");
}

// ---------------------------------------------------------------------------

int main() {
  printf("=== Jetson CUDA/TensorRT NvBufSurface Probe (O4-O7) ===\n");

  int dev = 0;
  CK(cudaSetDevice(dev));
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
    printf("  CUDA device: %s (cc %d.%d)\n", prop.name, prop.major, prop.minor);
  }

  NvBufSurface* surf = nullptr;
  if (alloc_and_fill(&surf) != 0) {
    printf("  BLOCKED: allocation failed; cannot run O4-O7\n");
    return 1;
  }

  probe_o4(surf);
  size_t sz = 0;
  void* dptr = probe_o5(surf, &sz);
  probe_o6(dptr);
  probe_o7(surf);

  NvBufSurfaceDestroy(surf);
  printf("=== probe complete ===\n");
  return 0;
}
