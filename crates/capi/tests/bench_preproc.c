// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// C API performance benchmark for image preprocessing pipeline.
//
// Gold-standard reference for C API integrators (GStreamer, V4L2, etc.).
// Demonstrates correct DMA-BUF import patterns and measures their cost.
//
// Benchmark sections:
//   1. DMA-BUF import patterns (hal_import_image + hal_plane_descriptor):
//      a. Import + reuse: import once, convert N times (optimal)
//      b. Import pool: V4L2-style buffer pool rotation with imported fds
//      c. Import with stride: padded buffers (V4L2 bytesperline)
//      d. Import multiplane NV12: separate Y and UV plane fds
//      e. Import per frame: re-import every iteration (anti-pattern)
//   2. Internal allocation patterns (hal_image_processor_create_image):
//      a. Reuse: allocate once, convert N times (format matrix)
//      b. Chained pipeline: NV12 -> RGBA -> PlanarRgb I8
//      c. Recreate per frame: allocate + free each iteration (anti-pattern)
//
// Build:   make bench
// Run:     ./build/bench_preproc
// Env:     BENCH_ITERATIONS=200         Override iteration count (default 100)
//          EDGEFIRST_FORCE_BACKEND=cpu   Pin a specific compute backend
//          BENCH_LOG=1                   Enable HAL debug logging

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <float.h>

#include "../include/edgefirst/hal.h"

// ============================================================================
// Configuration
// ============================================================================

#define DEFAULT_ITERATIONS 100
#define WARMUP_ITERATIONS  5

// Source resolution (typical 1080p camera)
#define SRC_W 1920
#define SRC_H 1080

// Destination resolution (standard YOLO model input)
#define DST_W 640
#define DST_H 640

// Number of buffers in the pool simulation
#define POOL_SIZE 4

// ============================================================================
// Timing helpers
// ============================================================================

// Returns elapsed time in milliseconds between two timespec values.
static double elapsed_ms(const struct timespec *start, const struct timespec *end) {
    double s  = (double)(end->tv_sec  - start->tv_sec);
    double ns = (double)(end->tv_nsec - start->tv_nsec);
    return s * 1000.0 + ns / 1.0e6;
}

// ============================================================================
// Result collection
// ============================================================================

typedef struct {
    const char *pattern;
    const char *format;
    double avg_ms;
    double min_ms;
    double max_ms;
    int    iterations;
    int    ok;  // 1 = ran successfully, 0 = skipped
    const char *skip_reason;
} bench_result;

static void result_print_header(void) {
    printf("%-40s %-28s %9s %9s %9s  (n)\n",
           "Pattern", "Format", "Avg (ms)", "Min (ms)", "Max (ms)");
    printf("-------------------------------------"
           "-------------------------------------"
           "------------------------\n");
}

static void result_print(const bench_result *r) {
    if (!r->ok) {
        printf("%-40s %-28s [skipped: %s]\n", r->pattern, r->format, r->skip_reason);
        return;
    }
    printf("%-40s %-28s %9.2f %9.2f %9.2f  (%d)\n",
           r->pattern, r->format, r->avg_ms, r->min_ms, r->max_ms, r->iterations);
}

// ============================================================================
// Letterbox crop helpers
// ============================================================================

// Compute the letterbox destination rectangle that fits src into dst while
// preserving aspect ratio. Matches the Rust calculate_letterbox() logic.
static void calculate_letterbox(size_t src_w, size_t src_h,
                                size_t dst_w, size_t dst_h,
                                size_t *out_left, size_t *out_top,
                                size_t *out_w, size_t *out_h) {
    double scale_w = (double)dst_w / (double)src_w;
    double scale_h = (double)dst_h / (double)src_h;
    double scale   = (scale_w < scale_h) ? scale_w : scale_h;

    *out_w = (size_t)(src_w * scale);
    *out_h = (size_t)(src_h * scale);
    *out_left = (dst_w - *out_w) / 2;
    *out_top  = (dst_h - *out_h) / 2;
}

// Build a letterbox crop for the standard SRC -> DST sizes.
static struct hal_crop make_letterbox_crop(void) {
    size_t left, top, new_w, new_h;
    calculate_letterbox(SRC_W, SRC_H, DST_W, DST_H, &left, &top, &new_w, &new_h);

    struct hal_crop crop = hal_crop_new();
    struct hal_rect rect = hal_rect_new(left, top, new_w, new_h);
    hal_crop_set_dst_rect(&crop, &rect);
    hal_crop_set_dst_color(&crop, 114, 114, 114, 255);
    return crop;
}

// ============================================================================
// DMA-BUF fd helper
//
// Allocates a DMA tensor via HAL and clones its fd.  This simulates
// receiving a DMA-BUF fd from an external source (V4L2, GStreamer, etc.).
// Returns the fd (caller must close) or -1 if DMA is unavailable.
// ============================================================================

static int allocate_dma_fd(size_t w, size_t h, enum hal_pixel_format fmt) {
    size_t shape[3] = {h, w, 0};
    size_t ndim;

    // Compute shape for the requested format
    switch (fmt) {
    case HAL_PIXEL_FORMAT_RGBA:
    case HAL_PIXEL_FORMAT_BGRA:
        shape[2] = 4; ndim = 3; break;
    case HAL_PIXEL_FORMAT_RGB:
        shape[2] = 3; ndim = 3; break;
    case HAL_PIXEL_FORMAT_GREY:
        shape[2] = 1; ndim = 3; break;
    case HAL_PIXEL_FORMAT_NV12:
        shape[0] = h * 3 / 2; shape[1] = w; ndim = 2; break;
    case HAL_PIXEL_FORMAT_NV16:
        shape[0] = h * 2; shape[1] = w; ndim = 2; break;
    case HAL_PIXEL_FORMAT_YUYV:
    case HAL_PIXEL_FORMAT_VYUY:
        shape[2] = 2; ndim = 3; break;
    case HAL_PIXEL_FORMAT_PLANAR_RGB:
        shape[0] = 3; shape[1] = h; shape[2] = w; ndim = 3; break;
    case HAL_PIXEL_FORMAT_PLANAR_RGBA:
        shape[0] = 4; shape[1] = h; shape[2] = w; ndim = 3; break;
    default:
        return -1;
    }

    struct hal_tensor *tmp = hal_tensor_new(
        HAL_DTYPE_U8, shape, ndim, HAL_TENSOR_MEMORY_DMA, NULL);
    if (!tmp) return -1;

    int fd = hal_tensor_dmabuf_clone(tmp);
    hal_tensor_free(tmp);
    return fd;
}

// Allocate a DMA fd for a standalone Y or UV plane (2D: [height, width]).
static int allocate_plane_fd(size_t h, size_t w) {
    size_t shape[2] = {h, w};
    struct hal_tensor *tmp = hal_tensor_new(
        HAL_DTYPE_U8, shape, 2, HAL_TENSOR_MEMORY_DMA, NULL);
    if (!tmp) return -1;

    int fd = hal_tensor_dmabuf_clone(tmp);
    hal_tensor_free(tmp);
    return fd;
}

// ============================================================================
// Section 1a: DMA-BUF import + reuse (optimal pattern)
//
// This is the recommended GStreamer / V4L2 integration pattern:
//   1. Receive a DMA-BUF fd from the camera/decoder
//   2. Build a hal_plane_descriptor (dups the fd)
//   3. Call hal_import_image() once (consumes the descriptor)
//   4. Reuse the returned tensor across all frames from the same buffer
//   5. Only re-import when the fd changes (new buffer from pool)
//
// After warmup, every convert() is an EGL image cache hit.
// ============================================================================

static bench_result bench_import_reuse(struct hal_image_processor *proc,
                                       int iterations) {
    bench_result res = {
        .pattern = "Import + reuse (recommended)",
        .format  = "NV12->RGBA 1080p",
        .ok      = 0,
    };

    // Simulate receiving a DMA-BUF fd from an external source
    int ext_fd = allocate_dma_fd(SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12);
    if (ext_fd < 0) {
        res.skip_reason = "DMA allocation unavailable";
        return res;
    }

    // Build a plane descriptor — dups the fd immediately.
    // The caller retains ownership of ext_fd.
    struct hal_plane_descriptor *pd = hal_plane_descriptor_new(ext_fd);
    close(ext_fd);  // We can close our fd — the descriptor holds its own dup
    if (!pd) {
        res.skip_reason = "plane descriptor creation failed";
        return res;
    }

    // Import the image — consumes pd (do NOT free pd after this)
    struct hal_tensor *src = hal_import_image(
        proc, pd, NULL, SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
    // pd is consumed — do NOT call hal_plane_descriptor_free(pd)
    if (!src) {
        res.skip_reason = "hal_import_image failed";
        return res;
    }

    // Allocate destination (reused across all frames)
    struct hal_tensor *dst = hal_image_processor_create_image(
        proc, DST_W, DST_H, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);
    if (!dst) {
        hal_tensor_free(src);
        res.skip_reason = "dst allocation failed";
        return res;
    }

    struct hal_crop crop = make_letterbox_crop();

    // Pre-flight
    if (hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop) != 0) {
        hal_tensor_free(src);
        hal_tensor_free(dst);
        res.skip_reason = "unsupported format combo";
        return res;
    }

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
    }

    // Measured iterations — same tensor handle reused every frame.
    // The EGL image is cached after warmup, so this measures pure GPU cost.
    double total = 0.0, min_t = DBL_MAX, max_t = 0.0;
    for (int i = 0; i < iterations; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double ms = elapsed_ms(&t0, &t1);
        total += ms;
        if (ms < min_t) min_t = ms;
        if (ms > max_t) max_t = ms;
    }

    hal_tensor_free(src);
    hal_tensor_free(dst);

    res.ok = 1;  res.iterations = iterations;
    res.avg_ms = total / iterations;
    res.min_ms = min_t;  res.max_ms = max_t;
    return res;
}

// ============================================================================
// Section 1b: DMA-BUF import pool (V4L2 MPLANE buffer rotation)
//
// Simulates a V4L2 buffer pool of POOL_SIZE DMA-BUFs.  Each buffer is
// imported once via hal_import_image() and reused across frames.  The pool
// rotates round-robin.  After one full rotation in warmup, every entry has
// a cached EGLImage and the per-frame cost should match single-buffer reuse.
// ============================================================================

static bench_result bench_import_pool(struct hal_image_processor *proc,
                                      int iterations) {
    bench_result res = {
        .pattern = "Import pool (4 bufs rotating)",
        .format  = "NV12->RGBA 1080p",
        .ok      = 0,
    };

    struct hal_tensor *pool[POOL_SIZE] = {NULL};

    // Simulate a V4L2 buffer pool: allocate POOL_SIZE DMA-BUFs and import each
    for (int i = 0; i < POOL_SIZE; i++) {
        int ext_fd = allocate_dma_fd(SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12);
        if (ext_fd < 0) {
            for (int j = 0; j < i; j++) hal_tensor_free(pool[j]);
            res.skip_reason = "DMA allocation unavailable";
            return res;
        }

        struct hal_plane_descriptor *pd = hal_plane_descriptor_new(ext_fd);
        close(ext_fd);
        if (!pd) {
            for (int j = 0; j < i; j++) hal_tensor_free(pool[j]);
            res.skip_reason = "plane descriptor creation failed";
            return res;
        }

        pool[i] = hal_import_image(
            proc, pd, NULL, SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
        if (!pool[i]) {
            for (int j = 0; j < i; j++) hal_tensor_free(pool[j]);
            res.skip_reason = "hal_import_image failed";
            return res;
        }
    }

    struct hal_tensor *dst = hal_image_processor_create_image(
        proc, DST_W, DST_H, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);
    if (!dst) {
        for (int i = 0; i < POOL_SIZE; i++) hal_tensor_free(pool[i]);
        res.skip_reason = "dst allocation failed";
        return res;
    }

    struct hal_crop crop = make_letterbox_crop();

    // Pre-flight
    if (hal_image_processor_convert(proc, pool[0], dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop) != 0) {
        for (int i = 0; i < POOL_SIZE; i++) hal_tensor_free(pool[i]);
        hal_tensor_free(dst);
        res.skip_reason = "unsupported format combo";
        return res;
    }

    // Warmup: cycle all pool entries so each gets a cached EGLImage
    for (int i = 0; i < WARMUP_ITERATIONS * POOL_SIZE; i++) {
        hal_image_processor_convert(proc, pool[i % POOL_SIZE], dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
    }

    // Measured iterations — rotating through the pool
    double total = 0.0, min_t = DBL_MAX, max_t = 0.0;
    for (int i = 0; i < iterations; i++) {
        struct hal_tensor *src = pool[i % POOL_SIZE];

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double ms = elapsed_ms(&t0, &t1);
        total += ms;
        if (ms < min_t) min_t = ms;
        if (ms > max_t) max_t = ms;
    }

    for (int i = 0; i < POOL_SIZE; i++) hal_tensor_free(pool[i]);
    hal_tensor_free(dst);

    res.ok = 1;  res.iterations = iterations;
    res.avg_ms = total / iterations;
    res.min_ms = min_t;  res.max_ms = max_t;
    return res;
}

// ============================================================================
// Section 1c: DMA-BUF import with stride (V4L2 bytesperline)
//
// Demonstrates hal_plane_descriptor_set_stride() for buffers with row
// padding (e.g., V4L2 drivers that align rows to cache-line boundaries).
// A stride of 2048 for 1920px NV12 simulates 128-byte row alignment.
// ============================================================================

static bench_result bench_import_stride(struct hal_image_processor *proc,
                                        int iterations) {
    bench_result res = {
        .pattern = "Import + stride (padded rows)",
        .format  = "NV12->RGBA 1080p stride=2048",
        .ok      = 0,
    };

    // Allocate a buffer large enough for the padded stride.
    // We use a contiguous NV12 buffer (stride * H * 3/2 bytes).
    int ext_fd = allocate_dma_fd(SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12);
    if (ext_fd < 0) {
        res.skip_reason = "DMA allocation unavailable";
        return res;
    }

    struct hal_plane_descriptor *pd = hal_plane_descriptor_new(ext_fd);
    close(ext_fd);
    if (!pd) {
        res.skip_reason = "plane descriptor creation failed";
        return res;
    }

    // Set stride: 2048 bytes per row (1920px + 128 bytes padding)
    hal_plane_descriptor_set_stride(pd, 2048);

    struct hal_tensor *src = hal_import_image(
        proc, pd, NULL, SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
    if (!src) {
        res.skip_reason = "hal_import_image with stride failed";
        return res;
    }

    struct hal_tensor *dst = hal_image_processor_create_image(
        proc, DST_W, DST_H, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);
    if (!dst) {
        hal_tensor_free(src);
        res.skip_reason = "dst allocation failed";
        return res;
    }

    struct hal_crop crop = make_letterbox_crop();

    if (hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop) != 0) {
        hal_tensor_free(src);
        hal_tensor_free(dst);
        res.skip_reason = "unsupported format combo (stride)";
        return res;
    }

    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
    }

    double total = 0.0, min_t = DBL_MAX, max_t = 0.0;
    for (int i = 0; i < iterations; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double ms = elapsed_ms(&t0, &t1);
        total += ms;
        if (ms < min_t) min_t = ms;
        if (ms > max_t) max_t = ms;
    }

    hal_tensor_free(src);
    hal_tensor_free(dst);

    res.ok = 1;  res.iterations = iterations;
    res.avg_ms = total / iterations;
    res.min_ms = min_t;  res.max_ms = max_t;
    return res;
}

// ============================================================================
// Section 1d: DMA-BUF multiplane NV12 import (V4L2 MPLANE)
//
// Demonstrates separate Y and UV plane fds with per-plane stride.
// This is the pattern for V4L2_PIX_FMT_NV12M where the kernel provides
// independent DMA-BUF fds for luma and chroma.
// ============================================================================

static bench_result bench_import_multiplane(struct hal_image_processor *proc,
                                            int iterations) {
    bench_result res = {
        .pattern = "Import multiplane NV12 (Y+UV)",
        .format  = "NV12->RGBA 1080p",
        .ok      = 0,
    };

    // Allocate separate Y and UV plane DMA-BUFs
    int y_fd = allocate_plane_fd(SRC_H, SRC_W);             // Y: 1080 x 1920
    int uv_fd = allocate_plane_fd(SRC_H / 2, SRC_W);        // UV: 540 x 1920
    if (y_fd < 0 || uv_fd < 0) {
        if (y_fd >= 0) close(y_fd);
        if (uv_fd >= 0) close(uv_fd);
        res.skip_reason = "DMA plane allocation unavailable";
        return res;
    }

    // Build plane descriptors for Y and UV
    struct hal_plane_descriptor *y_pd = hal_plane_descriptor_new(y_fd);
    struct hal_plane_descriptor *uv_pd = hal_plane_descriptor_new(uv_fd);
    close(y_fd);
    close(uv_fd);
    if (!y_pd || !uv_pd) {
        if (y_pd) hal_plane_descriptor_free(y_pd);
        if (uv_pd) hal_plane_descriptor_free(uv_pd);
        res.skip_reason = "plane descriptor creation failed";
        return res;
    }

    // Import multiplane NV12: y_pd for luma, uv_pd for chroma
    // Both descriptors are consumed by hal_import_image()
    struct hal_tensor *src = hal_import_image(
        proc, y_pd, uv_pd, SRC_W, SRC_H,
        HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
    // y_pd and uv_pd are consumed — do NOT free them
    if (!src) {
        res.skip_reason = "multiplane hal_import_image failed";
        return res;
    }

    struct hal_tensor *dst = hal_image_processor_create_image(
        proc, DST_W, DST_H, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);
    if (!dst) {
        hal_tensor_free(src);
        res.skip_reason = "dst allocation failed";
        return res;
    }

    struct hal_crop crop = make_letterbox_crop();

    if (hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop) != 0) {
        hal_tensor_free(src);
        hal_tensor_free(dst);
        res.skip_reason = "unsupported format combo (multiplane)";
        return res;
    }

    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
    }

    double total = 0.0, min_t = DBL_MAX, max_t = 0.0;
    for (int i = 0; i < iterations; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double ms = elapsed_ms(&t0, &t1);
        total += ms;
        if (ms < min_t) min_t = ms;
        if (ms > max_t) max_t = ms;
    }

    hal_tensor_free(src);
    hal_tensor_free(dst);

    res.ok = 1;  res.iterations = iterations;
    res.avg_ms = total / iterations;
    res.min_ms = min_t;  res.max_ms = max_t;
    return res;
}

// ============================================================================
// Section 1e: DMA-BUF import per frame (anti-pattern)
//
// Re-imports a new DMA-BUF every iteration, forcing an EGL image cache miss
// each time.  This quantifies the overhead of not reusing tensor handles.
// The correct pattern is to import once and reuse (Section 1a).
// ============================================================================

static bench_result bench_import_recreate(struct hal_image_processor *proc,
                                          int iterations) {
    bench_result res = {
        .pattern = "Import per frame (ANTI-PATTERN)",
        .format  = "NV12->RGBA 1080p",
        .ok      = 0,
    };

    struct hal_tensor *dst = hal_image_processor_create_image(
        proc, DST_W, DST_H, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);
    if (!dst) {
        res.skip_reason = "dst allocation failed";
        return res;
    }

    struct hal_crop crop = make_letterbox_crop();

    // Pre-flight
    {
        int ext_fd = allocate_dma_fd(SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12);
        if (ext_fd < 0) {
            hal_tensor_free(dst);
            res.skip_reason = "DMA allocation unavailable";
            return res;
        }
        struct hal_plane_descriptor *pd = hal_plane_descriptor_new(ext_fd);
        close(ext_fd);
        if (!pd) {
            hal_tensor_free(dst);
            res.skip_reason = "plane descriptor creation failed";
            return res;
        }
        struct hal_tensor *test = hal_import_image(
            proc, pd, NULL, SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
        if (!test) {
            hal_tensor_free(dst);
            res.skip_reason = "hal_import_image failed";
            return res;
        }
        if (hal_image_processor_convert(proc, test, dst,
                                         HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop) != 0) {
            hal_tensor_free(test);
            hal_tensor_free(dst);
            res.skip_reason = "unsupported format combo";
            return res;
        }
        hal_tensor_free(test);
    }

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        int ext_fd = allocate_dma_fd(SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12);
        if (ext_fd < 0) break;
        struct hal_plane_descriptor *pd = hal_plane_descriptor_new(ext_fd);
        close(ext_fd);
        if (!pd) break;
        struct hal_tensor *src = hal_import_image(
            proc, pd, NULL, SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
        if (src) {
            hal_image_processor_convert(proc, src, dst,
                                         HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
            hal_tensor_free(src);
        }
    }

    // Measured iterations — re-import every frame (cache miss each time)
    double total = 0.0, min_t = DBL_MAX, max_t = 0.0;
    int measured = 0;
    for (int i = 0; i < iterations; i++) {
        int ext_fd = allocate_dma_fd(SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12);
        if (ext_fd < 0) break;
        struct hal_plane_descriptor *pd = hal_plane_descriptor_new(ext_fd);
        close(ext_fd);
        if (!pd) break;

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        struct hal_tensor *src = hal_import_image(
            proc, pd, NULL, SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
        if (src) {
            hal_image_processor_convert(proc, src, dst,
                                         HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
            hal_tensor_free(src);
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double ms = elapsed_ms(&t0, &t1);
        total += ms;
        if (ms < min_t) min_t = ms;
        if (ms > max_t) max_t = ms;
        measured++;
    }

    hal_tensor_free(dst);

    if (measured == 0) {
        res.skip_reason = "all allocations failed";
        return res;
    }
    res.ok = 1;  res.iterations = measured;
    res.avg_ms = total / measured;
    res.min_ms = min_t;  res.max_ms = max_t;
    return res;
}

// ============================================================================
// Section 2a: Internal allocation + reuse (format matrix)
//
// Uses hal_image_processor_create_image() for both source and destination.
// Allocates once, converts N times.  This is the fastest pattern for
// internally-allocated buffers.
// ============================================================================

static bench_result bench_reuse(struct hal_image_processor *proc,
                                enum hal_pixel_format src_fmt,
                                enum hal_pixel_format dst_fmt,
                                enum hal_dtype dst_dtype,
                                const char *format_label,
                                int iterations) {
    bench_result res = {
        .pattern = "Reuse tensors (internal alloc)",
        .format  = format_label,
        .ok      = 0,
    };

    struct hal_tensor *src = hal_image_processor_create_image(
        proc, SRC_W, SRC_H, src_fmt, HAL_DTYPE_U8);
    if (!src) { res.skip_reason = "src allocation failed"; return res; }

    struct hal_tensor *dst = hal_image_processor_create_image(
        proc, DST_W, DST_H, dst_fmt, dst_dtype);
    if (!dst) {
        hal_tensor_free(src);
        res.skip_reason = "dst allocation failed";
        return res;
    }

    struct hal_crop crop = make_letterbox_crop();

    if (hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop) != 0) {
        hal_tensor_free(src);
        hal_tensor_free(dst);
        res.skip_reason = "unsupported format combo";
        return res;
    }

    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
    }

    double total = 0.0, min_t = DBL_MAX, max_t = 0.0;
    for (int i = 0; i < iterations; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double ms = elapsed_ms(&t0, &t1);
        total += ms;
        if (ms < min_t) min_t = ms;
        if (ms > max_t) max_t = ms;
    }

    hal_tensor_free(src);
    hal_tensor_free(dst);

    res.ok = 1;  res.iterations = iterations;
    res.avg_ms = total / iterations;
    res.min_ms = min_t;  res.max_ms = max_t;
    return res;
}

// ============================================================================
// Section 2b: Chained two-stage pipeline
//
// NV12 (1080p) -> RGBA (640x640) -> PlanarRgb I8 (640x640)
// All three tensors allocated once and reused.
// ============================================================================

static bench_result bench_chained(struct hal_image_processor *proc,
                                  int iterations) {
    bench_result res = {
        .pattern = "Chained (NV12->RGBA->PlanarRgb)",
        .format  = "1080p->640x640",
        .ok      = 0,
    };

    struct hal_tensor *src = hal_image_processor_create_image(
        proc, SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
    struct hal_tensor *mid = hal_image_processor_create_image(
        proc, DST_W, DST_H, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);
    struct hal_tensor *dst = hal_image_processor_create_image(
        proc, DST_W, DST_H, HAL_PIXEL_FORMAT_PLANAR_RGB, HAL_DTYPE_I8);

    if (!src || !mid || !dst) {
        if (src) hal_tensor_free(src);
        if (mid) hal_tensor_free(mid);
        if (dst) hal_tensor_free(dst);
        res.skip_reason = "allocation failed";
        return res;
    }

    struct hal_crop crop = make_letterbox_crop();

    if (hal_image_processor_convert(proc, src, mid,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop) != 0 ||
        hal_image_processor_convert(proc, mid, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, NULL) != 0) {
        hal_tensor_free(src);
        hal_tensor_free(mid);
        hal_tensor_free(dst);
        res.skip_reason = "pipeline stage unsupported";
        return res;
    }

    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        hal_image_processor_convert(proc, src, mid,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
        hal_image_processor_convert(proc, mid, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, NULL);
    }

    double total = 0.0, min_t = DBL_MAX, max_t = 0.0;
    for (int i = 0; i < iterations; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        hal_image_processor_convert(proc, src, mid,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
        hal_image_processor_convert(proc, mid, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, NULL);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double ms = elapsed_ms(&t0, &t1);
        total += ms;
        if (ms < min_t) min_t = ms;
        if (ms > max_t) max_t = ms;
    }

    hal_tensor_free(src);
    hal_tensor_free(mid);
    hal_tensor_free(dst);

    res.ok = 1;  res.iterations = iterations;
    res.avg_ms = total / iterations;
    res.min_ms = min_t;  res.max_ms = max_t;
    return res;
}

// ============================================================================
// Section 2c: Recreate per frame (anti-pattern, internal allocation)
// ============================================================================

static bench_result bench_recreate(struct hal_image_processor *proc,
                                   int iterations) {
    bench_result res = {
        .pattern = "Recreate per frame (ANTI-PATTERN)",
        .format  = "NV12->RGBA 1080p",
        .ok      = 0,
    };

    struct hal_tensor *dst = hal_image_processor_create_image(
        proc, DST_W, DST_H, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);
    if (!dst) { res.skip_reason = "dst allocation failed"; return res; }

    struct hal_crop crop = make_letterbox_crop();

    // Pre-flight
    {
        struct hal_tensor *test = hal_image_processor_create_image(
            proc, SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
        if (!test) {
            hal_tensor_free(dst);
            res.skip_reason = "src allocation failed";
            return res;
        }
        if (hal_image_processor_convert(proc, test, dst,
                                         HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop) != 0) {
            hal_tensor_free(test);
            hal_tensor_free(dst);
            res.skip_reason = "unsupported format combo";
            return res;
        }
        hal_tensor_free(test);
    }

    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        struct hal_tensor *src = hal_image_processor_create_image(
            proc, SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
        if (src) {
            hal_image_processor_convert(proc, src, dst,
                                         HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
            hal_tensor_free(src);
        }
    }

    double total = 0.0, min_t = DBL_MAX, max_t = 0.0;
    int measured = 0;
    for (int i = 0; i < iterations; i++) {
        struct hal_tensor *src = hal_image_processor_create_image(
            proc, SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
        if (!src) break;

        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
        hal_tensor_free(src);
        clock_gettime(CLOCK_MONOTONIC, &t1);

        double ms = elapsed_ms(&t0, &t1);
        total += ms;
        if (ms < min_t) min_t = ms;
        if (ms > max_t) max_t = ms;
        measured++;
    }

    hal_tensor_free(dst);

    if (measured == 0) { res.skip_reason = "all allocations failed"; return res; }
    res.ok = 1;  res.iterations = measured;
    res.avg_ms = total / measured;
    res.min_ms = min_t;  res.max_ms = max_t;
    return res;
}

// ============================================================================
// Main
// ============================================================================

int main(void) {
    int iterations = DEFAULT_ITERATIONS;
    const char *env_iter = getenv("BENCH_ITERATIONS");
    if (env_iter) {
        int n = atoi(env_iter);
        if (n > 0) iterations = n;
    }

    const char *env_log = getenv("BENCH_LOG");
    if (env_log) {
        hal_log_init_file(stderr, HAL_LOG_LEVEL_DEBUG);
    }

    printf("\nEdgeFirst HAL C API Benchmark -- Preprocessing Pipeline\n");
    printf("=======================================================\n");

    struct hal_image_processor *proc = hal_image_processor_new();
    if (!proc) {
        fprintf(stderr, "Error: failed to create image processor "
                        "(no backend available)\n");
        return 1;
    }

    printf("Iterations: %d (after %d warmup)\n", iterations, WARMUP_ITERATIONS);
    printf("Source: %dx%d  Destination: %dx%d\n\n", SRC_W, SRC_H, DST_W, DST_H);

    enum { MAX_RESULTS = 32 };
    bench_result results[MAX_RESULTS];
    int n_results = 0;

    // ── Section 1: DMA-BUF import patterns (hal_import_image) ────────────
    //
    // These benchmarks demonstrate the recommended integration pattern for
    // GStreamer, V4L2, and other frameworks that provide DMA-BUF fds.

    printf("--- DMA-BUF Import Patterns (hal_import_image) ---\n\n");

    results[n_results++] = bench_import_reuse(proc, iterations);
    results[n_results++] = bench_import_pool(proc, iterations);
    results[n_results++] = bench_import_stride(proc, iterations);
    results[n_results++] = bench_import_multiplane(proc, iterations);
    results[n_results++] = bench_import_recreate(proc, iterations);

    // ── Section 2: Internal allocation format matrix ─────────────────────
    //
    // Uses hal_image_processor_create_image() for when HAL owns the buffers.
    // Measures pure conversion cost across format combinations.

    // NV12 source
    results[n_results++] = bench_reuse(
        proc, HAL_PIXEL_FORMAT_NV12, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8,
        "NV12->RGBA", iterations);
    results[n_results++] = bench_reuse(
        proc, HAL_PIXEL_FORMAT_NV12, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_I8,
        "NV12->RGBA I8", iterations);
    results[n_results++] = bench_reuse(
        proc, HAL_PIXEL_FORMAT_NV12, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8,
        "NV12->RGB", iterations);
    results[n_results++] = bench_reuse(
        proc, HAL_PIXEL_FORMAT_NV12, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_I8,
        "NV12->RGB I8", iterations);
    results[n_results++] = bench_reuse(
        proc, HAL_PIXEL_FORMAT_NV12, HAL_PIXEL_FORMAT_PLANAR_RGB, HAL_DTYPE_U8,
        "NV12->PlanarRgb", iterations);
    results[n_results++] = bench_reuse(
        proc, HAL_PIXEL_FORMAT_NV12, HAL_PIXEL_FORMAT_PLANAR_RGB, HAL_DTYPE_I8,
        "NV12->PlanarRgb I8", iterations);

    // YUYV source
    results[n_results++] = bench_reuse(
        proc, HAL_PIXEL_FORMAT_YUYV, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8,
        "YUYV->RGBA", iterations);
    results[n_results++] = bench_reuse(
        proc, HAL_PIXEL_FORMAT_YUYV, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_I8,
        "YUYV->RGBA I8", iterations);
    results[n_results++] = bench_reuse(
        proc, HAL_PIXEL_FORMAT_YUYV, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_U8,
        "YUYV->RGB", iterations);
    results[n_results++] = bench_reuse(
        proc, HAL_PIXEL_FORMAT_YUYV, HAL_PIXEL_FORMAT_RGB, HAL_DTYPE_I8,
        "YUYV->RGB I8", iterations);
    results[n_results++] = bench_reuse(
        proc, HAL_PIXEL_FORMAT_YUYV, HAL_PIXEL_FORMAT_PLANAR_RGB, HAL_DTYPE_U8,
        "YUYV->PlanarRgb", iterations);
    results[n_results++] = bench_reuse(
        proc, HAL_PIXEL_FORMAT_YUYV, HAL_PIXEL_FORMAT_PLANAR_RGB, HAL_DTYPE_I8,
        "YUYV->PlanarRgb I8", iterations);

    // ── Section 3: Pipeline patterns ─────────────────────────────────────

    results[n_results++] = bench_chained(proc, iterations);

    // ── Section 4: Anti-patterns ─────────────────────────────────────────

    results[n_results++] = bench_recreate(proc, iterations);

    // --- Print results table ---

    printf("\n");
    result_print_header();

    for (int i = 0; i < n_results; i++) {
        result_print(&results[i]);
    }

    // --- Comparison notes ---

    double import_reuse_avg = 0.0, import_recreate_avg = 0.0;
    double import_pool_avg = 0.0, alloc_recreate_avg = 0.0;

    for (int i = 0; i < n_results; i++) {
        if (!results[i].ok) continue;
        if (strcmp(results[i].pattern, "Import + reuse (recommended)") == 0)
            import_reuse_avg = results[i].avg_ms;
        if (strcmp(results[i].pattern, "Import per frame (ANTI-PATTERN)") == 0)
            import_recreate_avg = results[i].avg_ms;
        if (strcmp(results[i].pattern, "Import pool (4 bufs rotating)") == 0)
            import_pool_avg = results[i].avg_ms;
        if (strcmp(results[i].pattern, "Recreate per frame (ANTI-PATTERN)") == 0)
            alloc_recreate_avg = results[i].avg_ms;
    }

    printf("\nNotes:\n");
    if (import_reuse_avg > 0.0 && import_recreate_avg > 0.0) {
        printf("  - 'Import per frame' is ~%.1fx slower than 'Import + reuse' "
               "due to EGL image cache misses\n",
               import_recreate_avg / import_reuse_avg);
    }
    if (import_reuse_avg > 0.0 && import_pool_avg > 0.0) {
        printf("  - 'Import pool' matches 'Import + reuse' after warmup "
               "(all %d entries cached, ratio: %.2fx)\n",
               POOL_SIZE, import_pool_avg / import_reuse_avg);
    }
    if (import_reuse_avg > 0.0 && alloc_recreate_avg > 0.0) {
        printf("  - 'Recreate per frame' (internal alloc) is ~%.1fx slower "
               "than import + reuse\n",
               alloc_recreate_avg / import_reuse_avg);
    }

    printf("\n");

    hal_image_processor_free(proc);
    return 0;
}
