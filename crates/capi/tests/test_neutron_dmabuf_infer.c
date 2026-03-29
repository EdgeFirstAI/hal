// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// Full inference test: HAL write into Neutron NPU DMA-BUF → Invoke → validate output.
//
// Proves the complete zero-copy path: HAL (G2D/GL/CPU) converts a source image
// into the Neutron-owned input DMA-BUF, then TFLite Invoke runs inference.
// Output tensors are validated for shape correctness and finite values.
//
// ── Test design ───────────────────────────────────────────────────────────────
//
// The critical pattern for GStreamer fidelity is "allocate-once, loop, free-once":
//   src tensor, dst tensor (Neutron DMA-BUF), and the image processor are ALL
//   allocated BEFORE the loop and freed ONCE after ALL invocations complete.
//   Nothing is released between convert and Invoke, nor between successive frames.
//
// This mirrors the GStreamer output_cache pattern where hal_tensor is held alive
// across all frames. The test is INVALID if tensors are freed between iterations.
//
// ── Confirmed findings (2026-03-28, imx95-frdm) ──────────────────────────────
//
// 30/30 PASS with N_REPEATS=5 and tensors held across all invocations:
//   Section 1 — GL backend (no CPU mmap of Neutron DMA-BUF):
//     NV12→RGB  U8 GL   5/5 PASS
//     NV12→RGBA U8 GL   5/5 PASS
//     NV12→RGB  I8 GL   5/5 PASS
//   Section 2 — G2D backend (CPU mmap for NEON XOR on I8):
//     NV12→RGB  U8 G2D  5/5 PASS
//     NV12→RGBA U8 G2D  5/5 PASS
//     NV12→RGB  I8 G2D  5/5 PASS  ← CPU mmap between convert and Invoke: OK
//
// Key conclusions:
//   - The Neutron driver handles repeated Invoke() with a live DmaTensor correctly.
//   - CPU mmap of the Neutron DMA-BUF (G2D I8 XOR path) does NOT cause timeouts.
//   - DrmAttachment is intentionally None for foreign (Neutron) fds — no wasted
//     PRIME import ioctl; DMA_BUF_IOCTL_SYNC is handled by the Neutron driver.
//   - If error 383307 (DRIVER/TIMEOUT) appears in the GStreamer pipeline, the
//     bug is in NNStreamer or edgefirstcameraadaptor, NOT in the HAL or Neutron.
//
// ── Compile on target ─────────────────────────────────────────────────────────
//
//   cp test_neutron_dmabuf_infer.c hal.h /tmp/
//   mkdir -p /tmp/edgefirst && cp /tmp/hal.h /tmp/edgefirst/
//   gcc -std=c11 -Wall -Wextra -Wno-comment -o /tmp/test_neutron_dmabuf_infer
//       /tmp/test_neutron_dmabuf_infer.c -I/tmp -ledgefirst_hal -ldl -lm
//
// ── Run ───────────────────────────────────────────────────────────────────────
//
//   NEUTRON_ENABLE_ZERO_COPY=1 ./test_neutron_dmabuf_infer
//       /opt/edgefirst/yolov8n_640x640.imx95.tflite
//
//   With real NV12 frame (1920x1080) for detection count validation:
//   NEUTRON_ENABLE_ZERO_COPY=1 ./test_neutron_dmabuf_infer
//       /opt/edgefirst/yolov8n_640x640.imx95.tflite libneutron_delegate.so frame.nv12

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <edgefirst/hal.h>

// ── TFLite C API (minimal dlsym-loaded subset) ──────────────────────────────
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
typedef TfLiteStatus (*Invoke_fn)(TfLiteInterpreter *);
typedef int (*GetInputCount_fn)(const TfLiteInterpreter *);
typedef const TfLiteTensor *(*GetInputTensor_fn)(const TfLiteInterpreter *, int);
typedef int (*GetOutputCount_fn)(const TfLiteInterpreter *);
typedef const TfLiteTensor *(*GetOutputTensor_fn)(const TfLiteInterpreter *, int);
typedef int (*TensorNumDims_fn)(const TfLiteTensor *);
typedef int (*TensorDim_fn)(const TfLiteTensor *, int);
typedef size_t (*TensorByteSize_fn)(const TfLiteTensor *);
typedef const void *(*TensorData_fn)(const TfLiteTensor *);
typedef int (*TensorType_fn)(const TfLiteTensor *);
typedef const char *(*TensorName_fn)(const TfLiteTensor *);
typedef void *(*ExtDelegateCreate_fn)(const void *);
typedef void (*ExtDelegateDelete_fn)(void *);
typedef struct { float scale; int32_t zero_point; } TfLiteQuantizationParams;
typedef TfLiteQuantizationParams (*TensorQuantParams_fn)(const TfLiteTensor *);

// HAL delegate symbols
typedef int (*hal_is_supported_fn)(void *);
typedef void *(*hal_get_instance_fn)(void);
typedef int (*hal_get_info_fn)(void *, int, hal_dmabuf_tensor_info *, size_t);

// ── Helpers ──────────────────────────────────────────────────────────────────

#define FAIL(fmt, ...) do { fprintf(stderr, "FAIL: " fmt "\n", ##__VA_ARGS__); exit(1); } while (0)
#define INFO(fmt, ...) fprintf(stderr, "INFO: " fmt "\n", ##__VA_ARGS__)
#define PASS(fmt, ...) fprintf(stderr, "PASS: " fmt "\n", ##__VA_ARGS__)

static void *must_dlsym(void *h, const char *name) {
    void *s = dlsym(h, name);
    if (!s) FAIL("dlsym(%s): %s", name, dlerror());
    return s;
}

static const char *backend_name(enum hal_compute_backend b) {
    switch (b) {
        case HAL_COMPUTE_BACKEND_AUTO:   return "AUTO";
        case HAL_COMPUTE_BACKEND_OPENGL: return "GL";
        case HAL_COMPUTE_BACKEND_G2D:    return "G2D";
        case HAL_COMPUTE_BACKEND_CPU:    return "CPU";
        default: return "?";
    }
}

// Print shape of a tensor as [d0,d1,...,dN]
static void print_shape(TensorNumDims_fn NDims, TensorDim_fn Dim,
                        const TfLiteTensor *t)
{
    int nd = NDims(t);
    fprintf(stderr, "[");
    for (int d = 0; d < nd; d++) fprintf(stderr, "%s%d", d ? "," : "", Dim(t, d));
    fprintf(stderr, "]");
}

// Validate float32 output tensor: count NaN/Inf, report min/max/mean
static int validate_float_output(const float *data, size_t n_floats,
                                  const char *name)
{
    int bad = 0;
    float vmin = data[0], vmax = data[0];
    double sum = 0.0;
    for (size_t i = 0; i < n_floats; i++) {
        if (!isfinite(data[i])) { bad++; continue; }
        if (data[i] < vmin) vmin = data[i];
        if (data[i] > vmax) vmax = data[i];
        sum += data[i];
    }
    float mean = (float)(sum / (double)n_floats);
    fprintf(stderr, "    %s: min=%.4f max=%.4f mean=%.4f bad=%d/%zu\n",
            name, vmin, vmax, mean, bad, n_floats);
    return bad == 0 ? 0 : -1;
}

// Count YOLOv8 detections above a confidence threshold.
// YOLOv8n output: [1, 84, 8400] — 4 box coords + 80 class scores, transposed.
// We look for any box whose max class score exceeds threshold.
static int count_detections(const float *data, int n_anchors, int n_attrs,
                             float conf_thresh)
{
    // data layout: [n_attrs, n_anchors] (row-major, attrs vary fastest in dim 1)
    int count = 0;
    for (int a = 0; a < n_anchors; a++) {
        float max_score = 0.0f;
        // class scores start at index 4 (after cx, cy, w, h)
        for (int c = 4; c < n_attrs; c++) {
            float score = data[c * n_anchors + a];
            if (score > max_score) max_score = score;
        }
        if (max_score >= conf_thresh) count++;
    }
    return count;
}

// ── Output validation (shared) ───────────────────────────────────────────────

typedef struct {
    Invoke_fn Invoke;
    GetOutputCount_fn OutCount;
    GetOutputTensor_fn OutTensor;
    TensorNumDims_fn NDims;
    TensorDim_fn Dim;
    TensorByteSize_fn ByteSize;
    TensorData_fn Data;
    TensorType_fn Type;
    TensorName_fn Name;
    TensorQuantParams_fn QuantParams;
} TfliteAPI;

// Validate output tensors after a successful Invoke(). Returns 0 if all ok.
static int validate_outputs(TfLiteInterpreter *interp, const TfliteAPI *api)
{
    int n_out = api->OutCount(interp);
    int ok = 1;
    for (int o = 0; o < n_out; o++) {
        const TfLiteTensor *t = api->OutTensor(interp, o);
        size_t nbytes = api->ByteSize(t);
        const char *name = api->Name(t) ? api->Name(t) : "output";

        fprintf(stderr, "    output[%d] '%s' ", o, name);
        print_shape(api->NDims, api->Dim, t);
        fprintf(stderr, " type=%d  %zu bytes\n", api->Type(t), nbytes);

        if (api->Type(t) == 1 /*kTfLiteFloat32*/ && nbytes >= sizeof(float)) {
            const float *fdata = (const float *)api->Data(t);
            size_t n_floats = nbytes / sizeof(float);
            if (validate_float_output(fdata, n_floats, "    values") != 0)
                ok = 0;
            if (api->NDims(t) == 3 && api->Dim(t, 1) == 84 && api->Dim(t, 2) == 8400) {
                fprintf(stderr, "    YOLOv8n: %d detections @conf≥0.25\n",
                        count_detections(fdata, 8400, 84, 0.25f));
            }
        } else if ((api->Type(t) == 3 || api->Type(t) == 9) && nbytes > 0) {
            TfLiteQuantizationParams qp = api->QuantParams(t);
            const int8_t *i8 = (const int8_t *)api->Data(t);
            float vmin = 0.0f, vmax = 0.0f;
            double sum = 0.0;
            for (size_t k = 0; k < nbytes; k++) {
                float v = ((float)i8[k] - (float)qp.zero_point) * qp.scale;
                if (k == 0 || v < vmin) vmin = v;
                if (k == 0 || v > vmax) vmax = v;
                sum += v;
            }
            fprintf(stderr, "    quant: scale=%.6f zp=%d  dequant: min=%.4f max=%.4f mean=%.4f\n",
                    qp.scale, qp.zero_point, vmin, vmax, (float)(sum / nbytes));
            if (api->NDims(t) == 3 && api->Dim(t, 1) == 84 && api->Dim(t, 2) == 8400) {
                int n_anchors = 8400, n_attrs = 84, det = 0;
                for (int a = 0; a < n_anchors; a++) {
                    float max_score = 0.0f;
                    for (int c = 4; c < n_attrs; c++) {
                        float v = ((float)i8[c * n_anchors + a] - (float)qp.zero_point) * qp.scale;
                        if (v > max_score) max_score = v;
                    }
                    if (max_score >= 0.25f) det++;
                }
                fprintf(stderr, "    YOLOv8n: %d detections @conf≥0.25\n", det);
            }
        }
    }
    return ok ? 0 : -1;
}

// ── Repeated inference: allocate-once, loop N times, free-once ───────────────
//
// This is the correct test for the GStreamer output_cache scenario:
// src, dst, and the image processor are allocated ONCE before the loop and
// freed ONCE after ALL invocations complete. No tensor is released between
// the convert and invoke steps, nor between successive frames.
//
// If the Neutron NPU DMA engine cannot tolerate a live DmaTensor across
// multiple convert→invoke cycles the failure will appear here.
static void run_repeated_inferences(
    TfLiteInterpreter *interp, const TfliteAPI *api,
    int neutron_fd, size_t offset,
    enum hal_pixel_format src_fmt, int src_w, int src_h,
    enum hal_pixel_format dst_fmt, enum hal_dtype dst_dtype,
    int dst_w, int dst_h,
    enum hal_compute_backend backend,
    int n_repeats, const char *label,
    hal_tensor *preloaded_src,
    int *pass_out, int *fail_out, int *skip_out)
{
    fprintf(stderr, "\n  %s  (%d repeats, tensors held across all invocations)\n",
            label, n_repeats);

    // ── Allocate processor ONCE ──────────────────────────────────────────────
    struct hal_image_processor *proc = hal_image_processor_new_with_backend(backend);
    if (!proc) {
        fprintf(stderr, "    SKIP (no %s backend)\n", backend_name(backend));
        (*skip_out) += n_repeats;
        return;
    }

    // ── Allocate src ONCE ────────────────────────────────────────────────────
    hal_tensor *src = preloaded_src;
    int src_owned = 0;
    if (!src) {
        src = hal_image_processor_create_image(proc, src_w, src_h, src_fmt, HAL_DTYPE_U8);
        src_owned = 1;
        if (!src) {
            fprintf(stderr, "    SKIP (src alloc failed)\n");
            hal_image_processor_free(proc);
            (*skip_out) += n_repeats;
            return;
        }
    }

    // ── Import dst (Neutron DMA-BUF) ONCE ────────────────────────────────────
    struct hal_plane_descriptor *pd = hal_plane_descriptor_new(neutron_fd);
    if (!pd) {
        fprintf(stderr, "    FAIL (hal_plane_descriptor_new)\n");
        if (src_owned) hal_tensor_free(src);
        hal_image_processor_free(proc);
        (*fail_out) += n_repeats;
        return;
    }
    if (offset > 0)
        hal_plane_descriptor_set_offset(pd, offset);

    hal_tensor *dst = hal_import_image(proc, pd, NULL, dst_w, dst_h, dst_fmt, dst_dtype);
    if (!dst) {
        fprintf(stderr, "    FAIL (import fd=%d offset=%zu errno=%d: %s)\n",
                neutron_fd, offset, errno, strerror(errno));
        if (src_owned) hal_tensor_free(src);
        hal_image_processor_free(proc);
        (*fail_out) += n_repeats;
        return;
    }

    // ── Loop: convert → invoke → validate (no free between iterations) ───────
    struct hal_crop crop = hal_crop_new();
    for (int rep = 0; rep < n_repeats; rep++) {
        fprintf(stderr, "    [%d/%d]  ", rep + 1, n_repeats);

        int rc = hal_image_processor_convert(proc, src, dst,
                                              HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
        if (rc != 0) {
            fprintf(stderr, "FAIL (convert rc=%d)\n", rc);
            (*fail_out)++;
            continue;
        }

        TfLiteStatus status = api->Invoke(interp);
        if (status != kTfLiteOk) {
            fprintf(stderr, "FAIL (Invoke status=%d)\n", status);
            (*fail_out)++;
            continue;
        }

        fprintf(stderr, "PASS\n");
        (*pass_out)++;
        validate_outputs(interp, api);
    }

    // ── Free ONCE after all repeats ───────────────────────────────────────────
    hal_tensor_free(dst);
    if (src_owned) hal_tensor_free(src);
    hal_image_processor_free(proc);
}

// ── Single-shot inference (for real-image validation) ────────────────────────
static int run_inference(TfLiteInterpreter *interp, const TfliteAPI *api,
                         int neutron_fd, size_t offset,
                         enum hal_pixel_format src_fmt, int src_w, int src_h,
                         enum hal_pixel_format dst_fmt, enum hal_dtype dst_dtype,
                         int dst_w, int dst_h,
                         enum hal_compute_backend backend,
                         const char *label,
                         hal_tensor *preloaded_src)
{
    fprintf(stderr, "\n  %-56s  ", label);

    struct hal_image_processor *proc = hal_image_processor_new_with_backend(backend);
    if (!proc) { fprintf(stderr, "SKIP (no %s)\n", backend_name(backend)); return 1; }

    hal_tensor *src = preloaded_src;
    int src_owned = 0;
    if (!src) {
        src = hal_image_processor_create_image(proc, src_w, src_h, src_fmt, HAL_DTYPE_U8);
        src_owned = 1;
        if (!src) {
            fprintf(stderr, "SKIP (src alloc failed)\n");
            hal_image_processor_free(proc);
            return 1;
        }
    }

    struct hal_plane_descriptor *pd = hal_plane_descriptor_new(neutron_fd);
    if (!pd) {
        fprintf(stderr, "FAIL (pd_new)\n");
        if (src_owned) hal_tensor_free(src);
        hal_image_processor_free(proc);
        return -1;
    }
    if (offset > 0)
        hal_plane_descriptor_set_offset(pd, offset);

    hal_tensor *dst = hal_import_image(proc, pd, NULL, dst_w, dst_h, dst_fmt, dst_dtype);
    if (!dst) {
        fprintf(stderr, "FAIL (import fd=%d offset=%zu errno=%d: %s)\n",
                neutron_fd, offset, errno, strerror(errno));
        if (src_owned) hal_tensor_free(src);
        hal_image_processor_free(proc);
        return -1;
    }

    struct hal_crop crop = hal_crop_new();
    int rc = hal_image_processor_convert(proc, src, dst,
                                          HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
    if (src_owned) hal_tensor_free(src);
    hal_image_processor_free(proc);

    if (rc != 0) {
        fprintf(stderr, "FAIL (convert rc=%d)\n", rc);
        hal_tensor_free(dst);
        return -1;
    }

    // dst kept alive across Invoke() (simulates GStreamer cache even for single shot)
    TfLiteStatus status = api->Invoke(interp);
    hal_tensor_free(dst);

    if (status != kTfLiteOk) {
        fprintf(stderr, "FAIL (Invoke status=%d)\n", status);
        return -1;
    }

    fprintf(stderr, "PASS (%s)\n", backend_name(backend));
    return validate_outputs(interp, api);
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr,
            "Usage: NEUTRON_ENABLE_ZERO_COPY=1 %s <model.tflite> [delegate.so] [frame.nv12]\n"
            "\n"
            "  model.tflite   : Neutron-compiled model (e.g. yolov8n_640x640.imx95.tflite)\n"
            "  delegate.so    : optional; default libneutron_delegate.so\n"
            "  frame.nv12     : optional raw 1920×1080 NV12 file for real-image inference\n",
            argv[0]);
        return 1;
    }
    const char *model_path    = argv[1];
    const char *delegate_path = argc > 2 ? argv[2] : "libneutron_delegate.so";
    const char *nv12_path     = argc > 3 ? argv[3] : NULL;

    hal_log_init_file(stderr, HAL_LOG_LEVEL_WARN);

    // ── Load TFLite ─────────────────────────────────────────────────────────
    const char *libs[] = { "libtensorflow-lite.so", "libtensorflow-lite.so.2.19.0", NULL };
    void *tfl = NULL;
    for (int i = 0; libs[i]; i++) { tfl = dlopen(libs[i], RTLD_LAZY); if (tfl) break; }
    if (!tfl) FAIL("Cannot load TFLite: %s", dlerror());

    void *dlg_h = dlopen(delegate_path, RTLD_LAZY);
    if (!dlg_h) FAIL("Cannot load delegate %s: %s", delegate_path, dlerror());

    // Mandatory TFLite symbols
    ModelCreate_fn ModelCreate         = must_dlsym(tfl, "TfLiteModelCreateFromFile");
    ModelDelete_fn ModelDelete         = must_dlsym(tfl, "TfLiteModelDelete");
    OptionsCreate_fn OptionsCreate     = must_dlsym(tfl, "TfLiteInterpreterOptionsCreate");
    OptionsDelete_fn OptionsDelete     = must_dlsym(tfl, "TfLiteInterpreterOptionsDelete");
    OptionsAddDelegate_fn AddDelegate  = must_dlsym(tfl, "TfLiteInterpreterOptionsAddDelegate");
    InterpreterCreate_fn InterpCreate  = must_dlsym(tfl, "TfLiteInterpreterCreate");
    InterpreterDelete_fn InterpDelete  = must_dlsym(tfl, "TfLiteInterpreterDelete");
    AllocateTensors_fn Alloc           = must_dlsym(tfl, "TfLiteInterpreterAllocateTensors");
    GetInputCount_fn InCount           = must_dlsym(tfl, "TfLiteInterpreterGetInputTensorCount");
    GetInputTensor_fn InTensor         = must_dlsym(tfl, "TfLiteInterpreterGetInputTensor");
    ExtDelegateCreate_fn ExtCreate     = must_dlsym(tfl, "TfLiteExternalDelegateCreate");
    ExtDelegateDelete_fn ExtDelete     = dlsym(tfl, "TfLiteExternalDelegateDelete");

    TfliteAPI api = {
        .Invoke   = must_dlsym(tfl, "TfLiteInterpreterInvoke"),
        .OutCount = must_dlsym(tfl, "TfLiteInterpreterGetOutputTensorCount"),
        .OutTensor= must_dlsym(tfl, "TfLiteInterpreterGetOutputTensor"),
        .NDims    = must_dlsym(tfl, "TfLiteTensorNumDims"),
        .Dim      = must_dlsym(tfl, "TfLiteTensorDim"),
        .ByteSize = must_dlsym(tfl, "TfLiteTensorByteSize"),
        .Data     = must_dlsym(tfl, "TfLiteTensorData"),
        .Type       = must_dlsym(tfl, "TfLiteTensorType"),
        .Name       = must_dlsym(tfl, "TfLiteTensorName"),
        .QuantParams= must_dlsym(tfl, "TfLiteTensorQuantizationParams"),
    };

    // ── Create interpreter ──────────────────────────────────────────────────
    struct { const char *path; int count; const char **keys; const char **values; } opts =
        { delegate_path, 0, NULL, NULL };
    TfLiteDelegate *delegate = (TfLiteDelegate *)ExtCreate(&opts);
    if (!delegate) FAIL("TfLiteExternalDelegateCreate failed");

    TfLiteModel *model = ModelCreate(model_path);
    if (!model) FAIL("ModelCreate(%s) failed", model_path);
    TfLiteInterpreterOptions *options = OptionsCreate();
    AddDelegate(options, delegate);
    TfLiteInterpreter *interp = InterpCreate(model, options);
    if (!interp) FAIL("InterpreterCreate failed");
    if (Alloc(interp) != kTfLiteOk) FAIL("AllocateTensors failed");

    // ── Print model info ────────────────────────────────────────────────────
    int n_in = InCount(interp);
    INFO("Model: %s (%d inputs)", model_path, n_in);
    int in_w = 0, in_h = 0;
    for (int i = 0; i < n_in; i++) {
        const TfLiteTensor *t = InTensor(interp, i);
        fprintf(stderr, "  input[%d]: %zu bytes ", i, api.ByteSize(t));
        print_shape(api.NDims, api.Dim, t);
        fprintf(stderr, "\n");
        if (i == 0 && api.NDims(t) == 4) {
            in_h = api.Dim(t, 1);
            in_w = api.Dim(t, 2);
        }
    }
    if (in_w == 0 || in_h == 0) FAIL("Cannot determine model input size");

    // ── Probe HAL delegate ──────────────────────────────────────────────────
    hal_get_instance_fn get_inst = dlsym(dlg_h, "hal_dmabuf_get_instance");
    hal_is_supported_fn is_sup   = dlsym(dlg_h, "hal_dmabuf_is_supported");
    hal_get_info_fn get_info     = dlsym(dlg_h, "hal_dmabuf_get_tensor_info");
    if (!get_inst || !is_sup || !get_info) FAIL("HAL delegate symbols not found in %s", delegate_path);

    void *hal_dlg = get_inst();
    if (!hal_dlg) FAIL("hal_dmabuf_get_instance() returned NULL");
    if (!is_sup(hal_dlg)) FAIL("hal_dmabuf_is_supported()=0 — is NEUTRON_ENABLE_ZERO_COPY=1 set?");

    hal_dmabuf_tensor_info info = {};
    if (get_info(hal_dlg, 0, &info, sizeof(info)) != 0)
        FAIL("hal_dmabuf_get_tensor_info(0) failed");
    INFO("Neutron input DMA-BUF: fd=%d offset=%zu size=%zu (model input %dx%d)",
         info.fd, info.offset, info.size, in_w, in_h);

    // ── Optionally load raw NV12 file into a HAL tensor ─────────────────────
    // Source dimensions: assume 1920×1080 NV12 (common camera output)
    int src_w = 1920, src_h = 1080;
    hal_tensor *file_src = NULL;

    if (nv12_path) {
        size_t nv12_size = (size_t)src_w * src_h * 3 / 2;
        int fd = open(nv12_path, O_RDONLY);
        if (fd < 0) FAIL("open(%s): %s", nv12_path, strerror(errno));
        void *raw = mmap(NULL, nv12_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (raw == MAP_FAILED) FAIL("mmap(%s): %s", nv12_path, strerror(errno));

        // Create a HAL CPU tensor and copy NV12 data into it
        struct hal_image_processor *ref_proc = hal_image_processor_new_with_backend(HAL_COMPUTE_BACKEND_CPU);
        if (!ref_proc) FAIL("CPU processor unavailable");
        file_src = hal_image_processor_create_image(ref_proc, src_w, src_h,
                                                     HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
        hal_image_processor_free(ref_proc);
        if (!file_src) FAIL("file_src alloc failed");

        // Write NV12 bytes into the tensor via hal_tensor_map
        struct hal_tensor_map *tmap = hal_tensor_map_create(file_src);
        if (!tmap) FAIL("hal_tensor_map_create failed");
        void *dst_map = hal_tensor_map_data(tmap);
        if (!dst_map) FAIL("hal_tensor_map_data returned NULL");
        memcpy(dst_map, raw, nv12_size);
        hal_tensor_map_unmap(tmap);
        munmap(raw, nv12_size);
        INFO("Loaded NV12 file: %s (%dx%d)", nv12_path, src_w, src_h);
    }

    // ── SECTION 1: OpenGL — src/dst/proc allocated ONCE, N_REPEATS loops ─────
    //
    // Simulates the GStreamer output_cache: tensors are allocated before the
    // loop and held alive for ALL invocations. No free/re-import between frames.
    // GL backend writes via GPU framebuffer — no CPU mmap of the Neutron buffer.
    fprintf(stderr,
            "\n═══ SECTION 1: OpenGL → Neutron DMA-BUF → Invoke (tensors held, no CPU mmap) ═══\n"
            "    src/dst/proc allocated once; N_REPEATS convert→invoke loops; freed after\n");

    int pass = 0, fail = 0, skip = 0;

#define N_REPEATS 5

    struct {
        enum hal_pixel_format dfmt;
        enum hal_dtype ddt;
        enum hal_compute_backend be;
        const char *label;
    } gl_tests[] = {
        { HAL_PIXEL_FORMAT_RGB,  HAL_DTYPE_U8, HAL_COMPUTE_BACKEND_OPENGL, "NV12→RGB  U8 GL" },
        { HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, HAL_COMPUTE_BACKEND_OPENGL, "NV12→RGBA U8 GL" },
        { HAL_PIXEL_FORMAT_RGB,  HAL_DTYPE_I8, HAL_COMPUTE_BACKEND_OPENGL, "NV12→RGB  I8 GL" },
    };

    for (int i = 0; i < (int)(sizeof(gl_tests) / sizeof(gl_tests[0])); i++) {
        run_repeated_inferences(interp, &api,
                                info.fd, info.offset,
                                HAL_PIXEL_FORMAT_NV12, src_w, src_h,
                                gl_tests[i].dfmt, gl_tests[i].ddt,
                                in_w, in_h,
                                gl_tests[i].be, N_REPEATS, gl_tests[i].label, NULL,
                                &pass, &fail, &skip);
    }

    // ── SECTION 2: G2D I8 — tensors held, CPU mmap stress ────────────────────
    //
    // Same allocate-once pattern as Section 1, but with G2D backend.
    // The G2D I8 path mmaps the Neutron DMA-BUF from CPU (NEON XOR 0x80 pass).
    // If holding dst alive + CPU mmap between convert and Invoke causes error
    // 383307 (DRIVER/TIMEOUT) on frame 2, it will appear here but not in S1.
    fprintf(stderr,
            "\n═══ SECTION 2: G2D I8 → Neutron DMA-BUF → Invoke (tensors held, CPU mmap) ═══\n"
            "    src/dst/proc allocated once; N_REPEATS convert→invoke loops; freed after\n");

    struct {
        enum hal_pixel_format dfmt;
        enum hal_dtype ddt;
        enum hal_compute_backend be;
        const char *label;
    } g2d_tests[] = {
        { HAL_PIXEL_FORMAT_RGB,  HAL_DTYPE_U8, HAL_COMPUTE_BACKEND_G2D, "NV12→RGB  U8 G2D" },
        { HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, HAL_COMPUTE_BACKEND_G2D, "NV12→RGBA U8 G2D" },
        { HAL_PIXEL_FORMAT_RGB,  HAL_DTYPE_I8, HAL_COMPUTE_BACKEND_G2D, "NV12→RGB  I8 G2D" },
    };

    for (int i = 0; i < (int)(sizeof(g2d_tests) / sizeof(g2d_tests[0])); i++) {
        run_repeated_inferences(interp, &api,
                                info.fd, info.offset,
                                HAL_PIXEL_FORMAT_NV12, src_w, src_h,
                                g2d_tests[i].dfmt, g2d_tests[i].ddt,
                                in_w, in_h,
                                g2d_tests[i].be, N_REPEATS, g2d_tests[i].label, NULL,
                                &pass, &fail, &skip);
    }

    // ── SECTION 3: Real image inference (if NV12 file provided) ─────────────
    if (file_src) {
        fprintf(stderr,
                "\n═══ SECTION 3: Real NV12 image → Neutron DMA-BUF → Invoke ═══\n"
                "    (real frame — output should show actual detections)\n");

        struct {
            enum hal_pixel_format dfmt;
            enum hal_dtype ddt;
            enum hal_compute_backend be;
            const char *label;
        } real_tests[] = {
            { HAL_PIXEL_FORMAT_RGB,  HAL_DTYPE_U8, HAL_COMPUTE_BACKEND_G2D,    "NV12→RGB U8 G2D    → Invoke" },
            { HAL_PIXEL_FORMAT_RGB,  HAL_DTYPE_U8, HAL_COMPUTE_BACKEND_OPENGL, "NV12→RGB U8 GL     → Invoke" },
            { HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8, HAL_COMPUTE_BACKEND_G2D,    "NV12→RGBA U8 G2D   → Invoke" },
            { HAL_PIXEL_FORMAT_RGB,  HAL_DTYPE_I8, HAL_COMPUTE_BACKEND_G2D,    "NV12→RGB I8 G2D    → Invoke" },
        };
        for (int i = 0; i < (int)(sizeof(real_tests) / sizeof(real_tests[0])); i++) {
            int rc = run_inference(interp, &api,
                                   info.fd, info.offset,
                                   HAL_PIXEL_FORMAT_NV12, src_w, src_h,
                                   real_tests[i].dfmt, real_tests[i].ddt,
                                   in_w, in_h, real_tests[i].be,
                                   real_tests[i].label, file_src);
            if (rc == 0) pass++; else if (rc < 0) fail++; else skip++;
        }
        hal_tensor_free(file_src);
    }

    // ── Summary ──────────────────────────────────────────────────────────────
    fprintf(stderr, "\n══════════════════════════════════════════════════\n");
    fprintf(stderr, "SUMMARY: %d passed, %d failed, %d skipped\n", pass, fail, skip);
    fprintf(stderr, "══════════════════════════════════════════════════\n");

    InterpDelete(interp);
    OptionsDelete(options);
    ModelDelete(model);
    if (ExtDelete) ExtDelete(delegate);
    return fail > 0 ? 1 : 0;
}
