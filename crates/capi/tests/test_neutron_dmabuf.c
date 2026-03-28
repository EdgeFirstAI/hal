// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Standalone test for GPU rendering into Neutron NPU DMA-BUF tensor.
//
// Compile on target:
//   gcc -std=c11 -Wall -Wextra -Wno-comment -o test_neutron_dmabuf \
//       test_neutron_dmabuf.c -I../include -ledgefirst_hal -ldl -lm
//
// Run:
//   NEUTRON_ENABLE_ZERO_COPY=1 ./test_neutron_dmabuf /path/to/model.imx95.tflite

#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <edgefirst/hal.h>

// ── TFLite C API (minimal dlsym-loaded subset) ─────────────────────────
typedef struct TfLiteModel TfLiteModel;
typedef struct TfLiteInterpreterOptions TfLiteInterpreterOptions;
typedef struct TfLiteInterpreter TfLiteInterpreter;
typedef struct TfLiteDelegate TfLiteDelegate;
typedef struct TfLiteTensor TfLiteTensor;
typedef enum { kTfLiteOk = 0, kTfLiteError = 1 } TfLiteStatus;

typedef TfLiteModel *(*ModelCreate_fn)(const char *);
typedef void (*ModelDelete_fn)(TfLiteModel *);
typedef TfLiteInterpreterOptions *(*OptionsCreate_fn)(void);
typedef void (*OptionsDelete_fn)(TfLiteInterpreterOptions *);
typedef void (*OptionsAddDelegate_fn)(TfLiteInterpreterOptions *, TfLiteDelegate *);
typedef TfLiteInterpreter *(*InterpreterCreate_fn)(const TfLiteModel *, const TfLiteInterpreterOptions *);
typedef void (*InterpreterDelete_fn)(TfLiteInterpreter *);
typedef TfLiteStatus (*AllocateTensors_fn)(TfLiteInterpreter *);
typedef int (*GetInputCount_fn)(const TfLiteInterpreter *);
typedef const TfLiteTensor *(*GetInputTensor_fn)(const TfLiteInterpreter *, int);
typedef int (*TensorNumDims_fn)(const TfLiteTensor *);
typedef int (*TensorDim_fn)(const TfLiteTensor *, int);
typedef size_t (*TensorByteSize_fn)(const TfLiteTensor *);
typedef void *(*ExtDelegateCreate_fn)(const void *);
typedef void (*ExtDelegateDelete_fn)(void *);

// HAL delegate symbols
typedef int (*hal_is_supported_fn)(void *);
typedef void *(*hal_get_instance_fn)(void);
typedef int (*hal_get_info_fn)(void *, int, hal_dmabuf_tensor_info *, size_t);

// ── Helpers ─────────────────────────────────────────────────────────────

#define FAIL(fmt, ...) do { fprintf(stderr, "FAIL: " fmt "\n", ##__VA_ARGS__); exit(1); } while (0)
#define INFO(fmt, ...) fprintf(stderr, "INFO: " fmt "\n", ##__VA_ARGS__)

static void *must_dlsym(void *h, const char *name) {
    void *s = dlsym(h, name);
    if (!s) FAIL("dlsym(%s): %s", name, dlerror());
    return s;
}

static const char *fmt_name(enum hal_pixel_format f) {
    switch (f) {
        case HAL_PIXEL_FORMAT_NV12: return "NV12";
        case HAL_PIXEL_FORMAT_RGBA: return "RGBA";
        case HAL_PIXEL_FORMAT_BGRA: return "BGRA";
        case HAL_PIXEL_FORMAT_RGB: return "RGB";
        case HAL_PIXEL_FORMAT_GREY: return "GREY";
        case HAL_PIXEL_FORMAT_PLANAR_RGB: return "PlanarRGB";
        default: return "?";
    }
}

static const char *dtype_name(enum hal_dtype d) {
    switch (d) {
        case HAL_DTYPE_U8: return "U8";
        case HAL_DTYPE_I8: return "I8";
        default: return "?";
    }
}

static const char *backend_name(enum hal_compute_backend b) {
    switch (b) {
        case HAL_COMPUTE_BACKEND_AUTO: return "AUTO";
        case HAL_COMPUTE_BACKEND_OPENGL: return "GL";
        case HAL_COMPUTE_BACKEND_G2D: return "G2D";
        case HAL_COMPUTE_BACKEND_CPU: return "CPU";
        default: return "?";
    }
}

// ── Single test: import fd at offset, convert with specified backend ────

static int test_convert(int neutron_fd, size_t offset,
                        int dst_w, int dst_h,
                        enum hal_pixel_format src_fmt, int src_w, int src_h,
                        enum hal_pixel_format dst_fmt, enum hal_dtype dst_dtype,
                        enum hal_compute_backend backend,
                        const char *label)
{
    fprintf(stderr, "\n  %-50s  ", label);

    struct hal_image_processor *proc = hal_image_processor_new_with_backend(backend);
    if (!proc) { fprintf(stderr, "SKIP (no %s backend)\n", backend_name(backend)); return 0; }

    // Source: HAL-allocated
    struct hal_tensor *src = hal_image_processor_create_image(proc,
        src_w, src_h, src_fmt, HAL_DTYPE_U8);
    if (!src) {
        fprintf(stderr, "SKIP (src alloc failed)\n");
        hal_image_processor_free(proc);
        return 0;
    }

    // Destination: Neutron fd
    struct hal_plane_descriptor *pd = hal_plane_descriptor_new(neutron_fd);
    if (!pd) { fprintf(stderr, "FAIL (pd_new)\n"); hal_tensor_free(src); hal_image_processor_free(proc); return -1; }
    if (offset > 0)
        hal_plane_descriptor_set_offset(pd, offset);

    struct hal_tensor *dst = hal_import_image(proc, pd, NULL,
        dst_w, dst_h, dst_fmt, dst_dtype);
    if (!dst) {
        fprintf(stderr, "FAIL (import errno=%d: %s)\n", errno, strerror(errno));
        hal_tensor_free(src);
        hal_image_processor_free(proc);
        return -1;
    }

    struct hal_crop crop = hal_crop_new();
    int rc = hal_image_processor_convert(proc, src, dst,
        HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);

    if (rc == 0)
        fprintf(stderr, "PASS (%s)\n", backend_name(backend));
    else
        fprintf(stderr, "FAIL (convert rc=%d)\n", rc);

    hal_tensor_free(dst);
    hal_tensor_free(src);
    hal_image_processor_free(proc);
    return rc;
}

// ── Main ────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: NEUTRON_ENABLE_ZERO_COPY=1 %s <model.tflite> [delegate.so]\n", argv[0]);
        return 1;
    }
    const char *model_path = argv[1];
    const char *delegate_path = argc > 2 ? argv[2] : "libneutron_delegate.so";

    hal_log_init_file(stderr, HAL_LOG_LEVEL_WARN);

    // ── Load TFLite + delegate + model ──────────────────────────────────
    const char *libs[] = { "libtensorflow-lite.so", "libtensorflow-lite.so.2.19.0", NULL };
    void *tfl = NULL;
    for (int i = 0; libs[i]; i++) { tfl = dlopen(libs[i], RTLD_LAZY); if (tfl) break; }
    if (!tfl) FAIL("Cannot load TFLite: %s", dlerror());

    void *dlg_h = dlopen(delegate_path, RTLD_LAZY);
    if (!dlg_h) FAIL("Cannot load delegate: %s", dlerror());

    ModelCreate_fn ModelCreate = must_dlsym(tfl, "TfLiteModelCreateFromFile");
    ModelDelete_fn ModelDelete = must_dlsym(tfl, "TfLiteModelDelete");
    OptionsCreate_fn OptionsCreate = must_dlsym(tfl, "TfLiteInterpreterOptionsCreate");
    OptionsDelete_fn OptionsDelete = must_dlsym(tfl, "TfLiteInterpreterOptionsDelete");
    OptionsAddDelegate_fn AddDelegate = must_dlsym(tfl, "TfLiteInterpreterOptionsAddDelegate");
    InterpreterCreate_fn InterpCreate = must_dlsym(tfl, "TfLiteInterpreterCreate");
    InterpreterDelete_fn InterpDelete = must_dlsym(tfl, "TfLiteInterpreterDelete");
    AllocateTensors_fn Alloc = must_dlsym(tfl, "TfLiteInterpreterAllocateTensors");
    GetInputCount_fn InCount = must_dlsym(tfl, "TfLiteInterpreterGetInputTensorCount");
    GetInputTensor_fn InTensor = must_dlsym(tfl, "TfLiteInterpreterGetInputTensor");
    TensorNumDims_fn NDims = must_dlsym(tfl, "TfLiteTensorNumDims");
    TensorDim_fn Dim = must_dlsym(tfl, "TfLiteTensorDim");
    TensorByteSize_fn ByteSize = must_dlsym(tfl, "TfLiteTensorByteSize");
    ExtDelegateCreate_fn ExtCreate = must_dlsym(tfl, "TfLiteExternalDelegateCreate");
    ExtDelegateDelete_fn ExtDelete = dlsym(tfl, "TfLiteExternalDelegateDelete");

    struct { const char *path; int count; const char **keys; const char **values; } opts =
        { delegate_path, 0, NULL, NULL };
    TfLiteDelegate *delegate = (TfLiteDelegate *)ExtCreate(&opts);
    if (!delegate) FAIL("ExtDelegateCreate failed");

    TfLiteModel *model = ModelCreate(model_path);
    if (!model) FAIL("ModelCreate(%s) failed", model_path);
    TfLiteInterpreterOptions *options = OptionsCreate();
    AddDelegate(options, delegate);
    TfLiteInterpreter *interp = InterpCreate(model, options);
    if (!interp) FAIL("InterpreterCreate failed");
    if (Alloc(interp) != kTfLiteOk) FAIL("AllocateTensors failed");

    int n = InCount(interp);
    INFO("Model: %s (%d inputs)", model_path, n);
    for (int i = 0; i < n; i++) {
        const TfLiteTensor *t = InTensor(interp, i);
        int nd = NDims(t);
        fprintf(stderr, "  input[%d]: %zu bytes [", i, ByteSize(t));
        for (int d = 0; d < nd; d++) fprintf(stderr, "%s%d", d?",":"", Dim(t, d));
        fprintf(stderr, "]\n");
    }

    // ── Probe HAL delegate ──────────────────────────────────────────────
    hal_get_instance_fn get_inst = dlsym(dlg_h, "hal_dmabuf_get_instance");
    hal_is_supported_fn is_sup = dlsym(dlg_h, "hal_dmabuf_is_supported");
    hal_get_info_fn get_info = dlsym(dlg_h, "hal_dmabuf_get_tensor_info");
    if (!get_inst || !is_sup || !get_info) FAIL("HAL delegate symbols not found");

    void *hal_dlg = get_inst();
    if (!hal_dlg) FAIL("get_instance() returned NULL");
    if (!is_sup(hal_dlg)) FAIL("is_supported()=0 — set NEUTRON_ENABLE_ZERO_COPY=1");

    hal_dmabuf_tensor_info info = {};
    if (get_info(hal_dlg, 0, &info, sizeof(info)) != 0)
        FAIL("get_tensor_info(0) failed");
    INFO("Neutron input: fd=%d offset=%zu size=%zu\n", info.fd, info.offset, info.size);

    int pass = 0, fail = 0, skip = 0;

    // ══════════════════════════════════════════════════════════════════
    // SECTION 1: Neutron fd WITH offset — U8 formats only (no I8 post-proc)
    // ══════════════════════════════════════════════════════════════════
    fprintf(stderr, "═══ SECTION 1: Neutron fd=%d offset=%zu — U8 formats ═══\n", info.fd, info.offset);

    struct { enum hal_pixel_format sfmt; int sw; int sh;
             enum hal_pixel_format dfmt; enum hal_dtype ddt; int dw; int dh;
             enum hal_compute_backend be; const char *label; } sec1[] = {
        // AUTO backend (fallback chain)
        { HAL_PIXEL_FORMAT_NV12, 1920, 1080, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, 640, 640, HAL_COMPUTE_BACKEND_AUTO, "NV12→RGBA U8 AUTO" },
        { HAL_PIXEL_FORMAT_NV12, 1920, 1080, HAL_PIXEL_FORMAT_RGB,  HAL_DTYPE_U8, 640, 640, HAL_COMPUTE_BACKEND_AUTO, "NV12→RGB U8 AUTO" },
        { HAL_PIXEL_FORMAT_NV12, 1920, 1080, HAL_PIXEL_FORMAT_BGRA, HAL_DTYPE_U8, 640, 640, HAL_COMPUTE_BACKEND_AUTO, "NV12→BGRA U8 AUTO" },
        { HAL_PIXEL_FORMAT_NV12, 1920, 1080, HAL_PIXEL_FORMAT_GREY, HAL_DTYPE_U8, 640, 640, HAL_COMPUTE_BACKEND_AUTO, "NV12→GREY U8 AUTO" },
        // Force GL
        { HAL_PIXEL_FORMAT_NV12, 1920, 1080, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, 640, 640, HAL_COMPUTE_BACKEND_OPENGL, "NV12→RGBA U8 GL-only" },
        { HAL_PIXEL_FORMAT_NV12, 1920, 1080, HAL_PIXEL_FORMAT_RGB,  HAL_DTYPE_U8, 640, 640, HAL_COMPUTE_BACKEND_OPENGL, "NV12→RGB U8 GL-only" },
        // Force G2D
        { HAL_PIXEL_FORMAT_NV12, 1920, 1080, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, 640, 640, HAL_COMPUTE_BACKEND_G2D, "NV12→RGBA U8 G2D-only" },
        { HAL_PIXEL_FORMAT_NV12, 1920, 1080, HAL_PIXEL_FORMAT_RGB,  HAL_DTYPE_U8, 640, 640, HAL_COMPUTE_BACKEND_G2D, "NV12→RGB U8 G2D-only" },
        // Different src formats
        { HAL_PIXEL_FORMAT_RGBA, 640, 640,   HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, 640, 640, HAL_COMPUTE_BACKEND_AUTO, "RGBA→RGBA U8 (passthrough)" },
        { HAL_PIXEL_FORMAT_RGBA, 640, 640,   HAL_PIXEL_FORMAT_RGB,  HAL_DTYPE_U8, 640, 640, HAL_COMPUTE_BACKEND_AUTO, "RGBA→RGB U8" },
        // Smaller destination
        { HAL_PIXEL_FORMAT_NV12, 1920, 1080, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, 320, 320, HAL_COMPUTE_BACKEND_AUTO, "NV12→RGBA U8 320x320" },
    };
    for (int i = 0; i < (int)(sizeof(sec1)/sizeof(sec1[0])); i++) {
        int rc = test_convert(info.fd, info.offset,
            sec1[i].dw, sec1[i].dh, sec1[i].sfmt, sec1[i].sw, sec1[i].sh,
            sec1[i].dfmt, sec1[i].ddt, sec1[i].be, sec1[i].label);
        if (rc == 0) pass++; else if (rc < 0) fail++; else skip++;
    }

    // ══════════════════════════════════════════════════════════════════
    // SECTION 2: Neutron fd WITH offset — I8 formats (tests int8 path)
    // ══════════════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ SECTION 2: Neutron fd=%d offset=%zu — I8 formats ═══\n", info.fd, info.offset);

    struct { enum hal_pixel_format dfmt; enum hal_compute_backend be; const char *label; } sec2[] = {
        { HAL_PIXEL_FORMAT_RGB,  HAL_COMPUTE_BACKEND_AUTO,   "NV12→RGB I8 AUTO" },
        { HAL_PIXEL_FORMAT_RGBA, HAL_COMPUTE_BACKEND_AUTO,   "NV12→RGBA I8 AUTO" },
        { HAL_PIXEL_FORMAT_RGB,  HAL_COMPUTE_BACKEND_OPENGL, "NV12→RGB I8 GL-only" },
        { HAL_PIXEL_FORMAT_RGB,  HAL_COMPUTE_BACKEND_G2D,    "NV12→RGB I8 G2D-only" },
        { HAL_PIXEL_FORMAT_RGBA, HAL_COMPUTE_BACKEND_G2D,    "NV12→RGBA I8 G2D-only" },
    };
    for (int i = 0; i < (int)(sizeof(sec2)/sizeof(sec2[0])); i++) {
        int rc = test_convert(info.fd, info.offset,
            640, 640, HAL_PIXEL_FORMAT_NV12, 1920, 1080,
            sec2[i].dfmt, HAL_DTYPE_I8, sec2[i].be, sec2[i].label);
        if (rc == 0) pass++; else if (rc < 0) fail++; else skip++;
    }

    // ══════════════════════════════════════════════════════════════════
    // SECTION 3: Neutron fd WITHOUT offset (offset=0)
    // ══════════════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ SECTION 3: Neutron fd=%d offset=0 ═══\n", info.fd);

    struct { enum hal_pixel_format dfmt; enum hal_dtype ddt;
             enum hal_compute_backend be; const char *label; } sec3[] = {
        { HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, HAL_COMPUTE_BACKEND_AUTO,   "NV12→RGBA U8 offset=0 AUTO" },
        { HAL_PIXEL_FORMAT_RGB,  HAL_DTYPE_U8, HAL_COMPUTE_BACKEND_AUTO,   "NV12→RGB U8 offset=0 AUTO" },
        { HAL_PIXEL_FORMAT_RGB,  HAL_DTYPE_I8, HAL_COMPUTE_BACKEND_AUTO,   "NV12→RGB I8 offset=0 AUTO" },
        { HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, HAL_COMPUTE_BACKEND_OPENGL, "NV12→RGBA U8 offset=0 GL-only" },
        { HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, HAL_COMPUTE_BACKEND_G2D,    "NV12→RGBA U8 offset=0 G2D-only" },
    };
    for (int i = 0; i < (int)(sizeof(sec3)/sizeof(sec3[0])); i++) {
        int rc = test_convert(info.fd, 0,
            640, 640, HAL_PIXEL_FORMAT_NV12, 1920, 1080,
            sec3[i].dfmt, sec3[i].ddt, sec3[i].be, sec3[i].label);
        if (rc == 0) pass++; else if (rc < 0) fail++; else skip++;
    }

    // ══════════════════════════════════════════════════════════════════
    // SECTION 4: HAL-allocated reference (dma_heap, no Neutron fd)
    // ══════════════════════════════════════════════════════════════════
    fprintf(stderr, "\n═══ SECTION 4: HAL-allocated dma_heap reference ═══\n");

    struct { enum hal_pixel_format dfmt; enum hal_dtype ddt;
             enum hal_compute_backend be; const char *label; } sec4[] = {
        { HAL_PIXEL_FORMAT_RGB,  HAL_DTYPE_I8, HAL_COMPUTE_BACKEND_AUTO,   "NV12→RGB I8 (dma_heap) AUTO" },
        { HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, HAL_COMPUTE_BACKEND_AUTO,   "NV12→RGBA U8 (dma_heap) AUTO" },
        { HAL_PIXEL_FORMAT_RGB,  HAL_DTYPE_U8, HAL_COMPUTE_BACKEND_OPENGL, "NV12→RGB U8 (dma_heap) GL-only" },
        { HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, HAL_COMPUTE_BACKEND_G2D,    "NV12→RGBA U8 (dma_heap) G2D-only" },
    };
    for (int i = 0; i < (int)(sizeof(sec4)/sizeof(sec4[0])); i++) {
        fprintf(stderr, "\n  %-50s  ", sec4[i].label);
        struct hal_image_processor *p = hal_image_processor_new_with_backend(sec4[i].be);
        if (!p) { fprintf(stderr, "SKIP (no %s)\n", backend_name(sec4[i].be)); skip++; continue; }
        struct hal_tensor *s = hal_image_processor_create_image(p, 1920, 1080, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
        struct hal_tensor *d = hal_image_processor_create_image(p, 640, 640, sec4[i].dfmt, sec4[i].ddt);
        if (!s || !d) { fprintf(stderr, "SKIP (alloc)\n"); skip++; }
        else {
            struct hal_crop c = hal_crop_new();
            int rc = hal_image_processor_convert(p, s, d, HAL_ROTATION_NONE, HAL_FLIP_NONE, &c);
            if (rc == 0) { fprintf(stderr, "PASS (%s)\n", backend_name(sec4[i].be)); pass++; }
            else { fprintf(stderr, "FAIL (rc=%d)\n", rc); fail++; }
        }
        if (d) hal_tensor_free(d);
        if (s) hal_tensor_free(s);
        hal_image_processor_free(p);
    }

    // ── Summary ─────────────────────────────────────────────────────────
    fprintf(stderr, "\n══════════════════════════════════════════════════\n");
    fprintf(stderr, "SUMMARY: %d passed, %d failed, %d skipped\n", pass, fail, skip);
    fprintf(stderr, "══════════════════════════════════════════════════\n");

    InterpDelete(interp); OptionsDelete(options); ModelDelete(model);
    if (ExtDelete) ExtDelete(delegate);
    return fail > 0 ? 1 : 0;
}
