// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//
// C API performance benchmark for image preprocessing pipeline.
//
// Measures the cost of different tensor usage patterns through the C API:
//   1. Tensor reuse (optimal): allocate once, convert N times
//   2. Tensor recreation (anti-pattern): create_image + free each iteration
//   3. Chained conversion: two-stage pipeline with tensor reuse
//   4. Buffer pool rotation: multiple source buffers, round-robin
//
// Build:   make bench
// Run:     ./build/bench_preproc
// Env:     BENCH_ITERATIONS=200         Override iteration count (default 100)
//          EDGEFIRST_FORCE_BACKEND=cpu   Pin a specific compute backend

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
    printf("%-36s %-28s %9s %9s %9s  (n)\n",
           "Pattern", "Format", "Avg (ms)", "Min (ms)", "Max (ms)");
    printf("-------------------------------------"
           "-------------------------------------"
           "--------------------\n");
}

static void result_print(const bench_result *r) {
    if (!r->ok) {
        printf("%-36s %-28s [skipped: %s]\n", r->pattern, r->format, r->skip_reason);
        return;
    }
    printf("%-36s %-28s %9.2f %9.2f %9.2f  (%d)\n",
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
// Benchmark 1: Tensor reuse (optimal pattern)
//
// Allocates source and destination tensors once, then calls convert() N times
// with the same handles. This is the fastest pattern because the EGL image
// cache hits on every iteration after warmup.
// ============================================================================

static bench_result bench_reuse(struct hal_image_processor *proc,
                                enum hal_pixel_format src_fmt,
                                enum hal_pixel_format dst_fmt,
                                enum hal_dtype dst_dtype,
                                const char *format_label,
                                int iterations) {
    bench_result res = {
        .pattern = "Reuse tensors",
        .format  = format_label,
        .ok      = 0,
    };

    struct hal_tensor *src = hal_image_processor_create_image(
        proc, SRC_W, SRC_H, src_fmt, HAL_DTYPE_U8);
    if (!src) {
        res.skip_reason = "src allocation failed";
        return res;
    }

    struct hal_tensor *dst = hal_image_processor_create_image(
        proc, DST_W, DST_H, dst_fmt, dst_dtype);
    if (!dst) {
        hal_tensor_free(src);
        res.skip_reason = "dst allocation failed";
        return res;
    }

    struct hal_crop crop = make_letterbox_crop();

    // Pre-flight: check that this format combination is supported.
    if (hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop) != 0) {
        hal_tensor_free(src);
        hal_tensor_free(dst);
        res.skip_reason = "unsupported format combo";
        return res;
    }

    // Warmup (unmeasured)
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        hal_image_processor_convert(proc, src, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
    }

    // Measured iterations
    double total = 0.0;
    double min_t = DBL_MAX;
    double max_t = 0.0;

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

    res.ok         = 1;
    res.iterations = iterations;
    res.avg_ms     = total / iterations;
    res.min_ms     = min_t;
    res.max_ms     = max_t;
    return res;
}

// ============================================================================
// Benchmark 2: Recreate tensor per frame (anti-pattern)
//
// Each iteration: allocate a brand new source tensor via
// hal_image_processor_create_image(), convert, then free.  Every new
// allocation gets a fresh BufferIdentity (new DMA-BUF fd or PBO), forcing
// an EGL image cache miss on every iteration.  This quantifies the overhead
// of not reusing tensors across frames.
// ============================================================================

static bench_result bench_recreate(struct hal_image_processor *proc,
                                   int iterations) {
    bench_result res = {
        .pattern = "Recreate tensor per frame",
        .format  = "NV12->RGBA 1080p",
        .ok      = 0,
    };

    // Destination tensor is reused (we are only measuring source-side churn).
    struct hal_tensor *dst = hal_image_processor_create_image(
        proc, DST_W, DST_H, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);
    if (!dst) {
        res.skip_reason = "dst allocation failed";
        return res;
    }

    struct hal_crop crop = make_letterbox_crop();

    // Pre-flight: verify the format combination works.
    {
        struct hal_tensor *test_src = hal_image_processor_create_image(
            proc, SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
        if (!test_src) {
            hal_tensor_free(dst);
            res.skip_reason = "src allocation failed";
            return res;
        }
        if (hal_image_processor_convert(proc, test_src, dst,
                                         HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop) != 0) {
            hal_tensor_free(test_src);
            hal_tensor_free(dst);
            res.skip_reason = "unsupported format combo";
            return res;
        }
        hal_tensor_free(test_src);
    }

    // Warmup (unmeasured)
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        struct hal_tensor *src = hal_image_processor_create_image(
            proc, SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
        if (src) {
            hal_image_processor_convert(proc, src, dst,
                                         HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
            hal_tensor_free(src);
        }
    }

    // Measured iterations
    double total = 0.0;
    double min_t = DBL_MAX;
    double max_t = 0.0;

    int measured = 0;
    for (int i = 0; i < iterations; i++) {
        struct hal_tensor *src = hal_image_processor_create_image(
            proc, SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
        if (!src) break;  // allocation failure — stop measuring

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

    if (measured == 0) {
        res.skip_reason = "all allocations failed";
        return res;
    }
    res.ok         = 1;
    res.iterations = measured;
    res.avg_ms     = total / measured;
    res.min_ms     = min_t;
    res.max_ms     = max_t;
    return res;
}

// ============================================================================
// Benchmark 3: Chained two-stage pipeline
//
// NV12 (1080p) -> RGBA (640x640) -> PlanarRgb I8 (640x640)
// All three tensors allocated once and reused.  Demonstrates that chaining
// convert() calls is safe (glFinish ensures coherency between stages) and
// measures the combined cost of a realistic two-pass preprocessing pipeline.
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
    if (!src) {
        res.skip_reason = "src allocation failed";
        return res;
    }

    struct hal_tensor *mid = hal_image_processor_create_image(
        proc, DST_W, DST_H, HAL_PIXEL_FORMAT_RGBA, HAL_DTYPE_U8);
    if (!mid) {
        hal_tensor_free(src);
        res.skip_reason = "mid allocation failed";
        return res;
    }

    struct hal_tensor *dst = hal_image_processor_create_image(
        proc, DST_W, DST_H, HAL_PIXEL_FORMAT_PLANAR_RGB, HAL_DTYPE_I8);
    if (!dst) {
        hal_tensor_free(src);
        hal_tensor_free(mid);
        res.skip_reason = "dst allocation failed";
        return res;
    }

    struct hal_crop crop = make_letterbox_crop();

    // Pre-flight: stage 1 (NV12 -> RGBA with letterbox)
    if (hal_image_processor_convert(proc, src, mid,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop) != 0) {
        hal_tensor_free(src);
        hal_tensor_free(mid);
        hal_tensor_free(dst);
        res.skip_reason = "stage 1 unsupported";
        return res;
    }

    // Pre-flight: stage 2 (RGBA -> PlanarRgb I8, same size, no crop)
    if (hal_image_processor_convert(proc, mid, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, NULL) != 0) {
        hal_tensor_free(src);
        hal_tensor_free(mid);
        hal_tensor_free(dst);
        res.skip_reason = "stage 2 unsupported";
        return res;
    }

    // Warmup
    for (int i = 0; i < WARMUP_ITERATIONS; i++) {
        hal_image_processor_convert(proc, src, mid,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
        hal_image_processor_convert(proc, mid, dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, NULL);
    }

    // Measured iterations
    double total = 0.0;
    double min_t = DBL_MAX;
    double max_t = 0.0;

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

    res.ok         = 1;
    res.iterations = iterations;
    res.avg_ms     = total / iterations;
    res.min_ms     = min_t;
    res.max_ms     = max_t;
    return res;
}

// ============================================================================
// Benchmark 4: Buffer pool simulation
//
// Allocates POOL_SIZE source tensors up front (simulating a V4L2 buffer pool)
// and rotates through them round-robin each iteration.  After warmup, every
// tensor in the pool has a cached EGLImage, so the per-iteration cost should
// match the single-tensor reuse benchmark.  This validates that the EGL cache
// handles a small working set correctly.
// ============================================================================

static bench_result bench_pool(struct hal_image_processor *proc,
                               int iterations) {
    bench_result res = {
        .pattern = "Buffer pool (4 bufs rotating)",
        .format  = "NV12->RGBA 1080p",
        .ok      = 0,
    };

    struct hal_tensor *pool[POOL_SIZE] = {NULL};

    for (int i = 0; i < POOL_SIZE; i++) {
        pool[i] = hal_image_processor_create_image(
            proc, SRC_W, SRC_H, HAL_PIXEL_FORMAT_NV12, HAL_DTYPE_U8);
        if (!pool[i]) {
            for (int j = 0; j < i; j++) hal_tensor_free(pool[j]);
            res.skip_reason = "pool src allocation failed";
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

    // Warmup: cycle through all pool entries so each gets a cached EGLImage.
    for (int i = 0; i < WARMUP_ITERATIONS * POOL_SIZE; i++) {
        hal_image_processor_convert(proc, pool[i % POOL_SIZE], dst,
                                     HAL_ROTATION_NONE, HAL_FLIP_NONE, &crop);
    }

    // Measured iterations
    double total = 0.0;
    double min_t = DBL_MAX;
    double max_t = 0.0;

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

    res.ok         = 1;
    res.iterations = iterations;
    res.avg_ms     = total / iterations;
    res.min_ms     = min_t;
    res.max_ms     = max_t;
    return res;
}

// ============================================================================
// Main
// ============================================================================

int main(void) {
    // Read iteration count from environment (default: 100).
    int iterations = DEFAULT_ITERATIONS;
    const char *env_iter = getenv("BENCH_ITERATIONS");
    if (env_iter) {
        int n = atoi(env_iter);
        if (n > 0) iterations = n;
    }

    // Optionally enable HAL debug logging via RUST_LOG / stderr.
    const char *env_log = getenv("BENCH_LOG");
    if (env_log) {
        hal_log_init_file(stderr, HAL_LOG_LEVEL_DEBUG);
    }

    printf("\nEdgeFirst HAL C API Benchmark -- Preprocessing Pipeline\n");
    printf("=======================================================\n");

    // Create the image processor (auto-selects best backend).
    struct hal_image_processor *proc = hal_image_processor_new();
    if (!proc) {
        fprintf(stderr, "Error: failed to create image processor "
                        "(no backend available)\n");
        return 1;
    }

    printf("Iterations: %d (after %d warmup)\n", iterations, WARMUP_ITERATIONS);
    printf("Source: %dx%d  Destination: %dx%d\n\n", SRC_W, SRC_H, DST_W, DST_H);

    // Collect all results for summary table.
    enum { MAX_RESULTS = 32 };
    bench_result results[MAX_RESULTS];
    int n_results = 0;

    // ── Section 1: Letterbox resize format matrix (tensor reuse) ─────────
    //
    // All combinations of {NV12, YUYV} × {RGBA, RGB, PlanarRgb} × {U8, I8}
    // at 1080p → 640x640 with letterbox crop. This is the primary benchmark
    // and should match the Rust pipeline_benchmark results.

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

    // ── Section 2: Anti-pattern comparison ───────────────────────────────

    results[n_results++] = bench_recreate(proc, iterations);

    // ── Section 3: Chained two-stage pipeline ────────────────────────────

    results[n_results++] = bench_chained(proc, iterations);

    // ── Section 4: Buffer pool simulation ────────────────────────────────

    results[n_results++] = bench_pool(proc, iterations);

    // --- Print results table ---

    printf("\n");
    result_print_header();

    for (int i = 0; i < n_results; i++) {
        result_print(&results[i]);
    }

    // --- Notes ---

    // Find reuse NV12->RGBA and recreate results for comparison.
    double reuse_avg   = 0.0;
    double recreate_avg = 0.0;
    double pool_avg    = 0.0;
    int have_comparison = 0;

    for (int i = 0; i < n_results; i++) {
        if (results[i].ok && strcmp(results[i].pattern, "Reuse tensors") == 0
            && strcmp(results[i].format, "NV12->RGBA") == 0) {
            reuse_avg = results[i].avg_ms;
        }
        if (results[i].ok && strcmp(results[i].pattern, "Recreate tensor per frame") == 0) {
            recreate_avg = results[i].avg_ms;
        }
        if (results[i].ok && strcmp(results[i].pattern, "Buffer pool (4 bufs rotating)") == 0) {
            pool_avg = results[i].avg_ms;
        }
    }

    printf("\nNotes:\n");
    if (reuse_avg > 0.0 && recreate_avg > 0.0) {
        have_comparison = 1;
        printf("  - 'Recreate tensor per frame' is ~%.1fx slower than 'Reuse tensors' "
               "due to EGL image cache misses\n", recreate_avg / reuse_avg);
    }
    if (reuse_avg > 0.0 && pool_avg > 0.0) {
        printf("  - 'Buffer pool' matches 'Reuse tensors' after warmup "
               "(all %d entries cached, ratio: %.2fx)\n",
               POOL_SIZE, pool_avg / reuse_avg);
    }
    if (!have_comparison) {
        printf("  - Comparison between reuse and recreate patterns unavailable "
               "(one or both were skipped)\n");
    }

    printf("\n");

    hal_image_processor_free(proc);
    return 0;
}
