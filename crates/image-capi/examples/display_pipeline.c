/*
 * SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * Display-pipeline building blocks from C, checked on the device.
 *
 * The pattern: the GPU renders the camera frame and the detection boxes into
 * a hardware-only (EF_CPU_ACCESS_NONE) DMA-BUF canvas whose fd goes to the
 * compositor, a CPU-written overlay lives in an EF_CPU_ACCESS_WRITE image, and
 * the decoder turns model outputs into boxes. On Linux both images come from
 * the uncached CMA heap where available, so the compositor's per-frame mapping
 * costs no CPU cache maintenance; the library refuses any CPU read of them.
 *
 * Converts and draws use separate processors: on the i.MX 95's Mali GPU a
 * draw leaves state behind that corrupts the next convert on the same
 * processor.
 *
 * Usage: display_pipeline <capture_dir> [iters]
 *
 * capture_dir holds the model's edgefirst.json, its raw outputs out<i>.bin,
 * and manifest.txt with one line per output:
 *     <index> <ndim> <dim>... <scale> <zero_point>
 * (int8 outputs). Each step prints PASS or FAIL; the decode and draw steps
 * also print timings.
 */

#define _GNU_SOURCE
#include <edgefirst/decoder.h>
#include <edgefirst/image.h>
#include <edgefirst/tensor.h>

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define W 640
#define H 480
#define MAX_OUTPUTS 16
#define MAX_RANK 8

static int failures;

static void check(const char *name, int ok, const char *detail)
{
    printf("%s  %s%s%s\n", ok ? "PASS" : "FAIL", name, detail ? ": " : "", detail ? detail : "");
    if (!ok)
        failures++;
}

static double now_us(clockid_t clock)
{
    struct timespec ts;
    clock_gettime(clock, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
}

static int cmp_double(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* The heap a DMA-BUF fd came from, as /proc/self/fdinfo names it. */
static void exporter(int fd, char *out, size_t len)
{
    char path[64], line[256];
    snprintf(path, sizeof path, "/proc/self/fdinfo/%d", fd);
    snprintf(out, len, "unknown");
    FILE *f = fopen(path, "r");
    if (!f)
        return;
    while (fgets(line, sizeof line, f)) {
        if (strncmp(line, "exp_name:", 9) == 0) {
            char *name = line + 9;
            name += strspn(name, " \t");
            name[strcspn(name, "\n")] = '\0';
            snprintf(out, len, "%s", name);
        }
    }
    fclose(f);
}

static ef_tensor *image(ef_image_processor *p, uint32_t access)
{
    return ef_image_processor_create_image(p, W, H, "rgba8", EF_DTYPE_U8, EF_STORAGE_KIND_DMA_BUF,
                                           access);
}

/* Fill every pixel with one colour through a write-only map. */
static int fill(ef_tensor *t, const uint8_t rgba[4])
{
    struct ef_tensor_view v;
    if (ef_tensor_map(t, EF_CPU_ACCESS_WRITE, &v) != 0)
        return -1;
    for (size_t i = 0; i + 4 <= v.len; i += 4)
        memcpy(v.ptr + i, rgba, 4);
    return ef_tensor_unmap(t);
}

/* The RGBA pixel at (x, y) of a CPU-readable image. */
static void pixel(ef_tensor *t, int x, int y, uint8_t out[4])
{
    struct ef_tensor_view v;
    int64_t stride = ef_tensor_row_stride(t);
    if (stride <= 0)
        stride = W * 4;
    memset(out, 0, 4);
    if (ef_tensor_map(t, EF_CPU_ACCESS_READ, &v) != 0)
        return;
    memcpy(out, v.ptr + y * stride + x * 4, 4);
    ef_tensor_unmap(t);
}

/* Load the captured int8 outputs listed in manifest.txt. */
static size_t load_outputs(const char *dir, ef_tensor **outs)
{
    char path[512];
    snprintf(path, sizeof path, "%s/manifest.txt", dir);
    FILE *m = fopen(path, "r");
    if (!m)
        return 0;
    size_t n = 0;
    unsigned index, ndim;
    while (n < MAX_OUTPUTS && fscanf(m, "%u %u", &index, &ndim) == 2) {
        uint64_t dims[MAX_RANK];
        if (ndim == 0 || ndim > MAX_RANK) {
            fprintf(stderr, "manifest.txt: output %u has rank %u, outside 1..%d\n", index, ndim,
                    MAX_RANK);
            goto out;
        }
        size_t len = 1;
        for (unsigned i = 0; i < ndim; i++) {
            unsigned long long d;
            if (fscanf(m, "%llu", &d) != 1)
                goto out;
            dims[i] = d;
            len *= d;
        }
        float scale;
        int32_t zp;
        if (fscanf(m, "%f %d", &scale, &zp) != 2)
            goto out;

        ef_tensor_builder *b = ef_tensor_builder_new();
        ef_tensor_builder_dtype(b, EF_DTYPE_I8);
        ef_tensor_builder_shape(b, dims, ndim);
        ef_tensor_builder_storage(b, EF_STORAGE_KIND_MEM);
        ef_tensor *t = ef_tensor_builder_alloc(b);
        ef_tensor_builder_free(b);
        if (!t)
            goto out;
        ef_tensor_quantization_set(t, -1, &scale, &zp, 1);

        snprintf(path, sizeof path, "%s/out%u.bin", dir, index);
        FILE *f = fopen(path, "rb");
        struct ef_tensor_view v;
        if (!f || ef_tensor_map(t, EF_CPU_ACCESS_WRITE, &v) != 0 || v.len < len ||
            fread(v.ptr, 1, len, f) != len) {
            if (f)
                fclose(f);
            ef_tensor_free(t);
            goto out;
        }
        ef_tensor_unmap(t);
        fclose(f);
        outs[n++] = t;
    }
out:
    fclose(m);
    return n;
}

static void report(const char *what, double *times, int iters, double cpu_us)
{
    double sum = 0;
    for (int i = 0; i < iters; i++)
        sum += times[i];
    qsort(times, iters, sizeof *times, cmp_double);
    printf("TIME  %s: median %.1f us, mean %.1f us, cpu %.1f us/iter (n=%d)\n", what,
           times[iters / 2], sum / iters, cpu_us, iters);
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <capture_dir> [iters]\n", argv[0]);
        return 2;
    }
    const char *dir = argv[1];
    int iters = argc > 2 ? atoi(argv[2]) : 1000;
    char detail[256];
    uint8_t px[4];

    ef_image_processor *conv = ef_image_processor_new();
    ef_image_processor *draw = ef_image_processor_new();
    ef_image_processor *rb = ef_image_processor_new();
    if (!conv || !draw || !rb) {
        fprintf(stderr, "no image processor\n");
        return 1;
    }

    /* Box style: 4 px wide, half-transparent red. */
    const uint8_t colors[1][4] = {{255, 0, 0, 128}};
    check("set_class_colors", ef_image_processor_set_class_colors(draw, colors, 1) == 0, NULL);
    check("set_box_line_width 4", ef_image_processor_set_box_line_width(draw, 4) == 0, NULL);
    check("set_box_line_width 0 is EINVAL",
          ef_image_processor_set_box_line_width(draw, 0) == EINVAL, NULL);

    /* The display images. */
    ef_tensor *camera = image(conv, EF_CPU_ACCESS_READ_WRITE);
    ef_tensor *frame = image(conv, EF_CPU_ACCESS_NONE);
    ef_tensor *canvas = image(draw, EF_CPU_ACCESS_NONE);
    ef_tensor *overlay = image(draw, EF_CPU_ACCESS_WRITE);
    ef_tensor *readback = image(rb, EF_CPU_ACCESS_READ_WRITE);
    if (!camera || !frame || !canvas || !overlay || !readback) {
        fprintf(stderr, "image allocation failed\n");
        return 1;
    }
    check("canvas declared NONE", ef_tensor_cpu_access(canvas) == EF_CPU_ACCESS_NONE, NULL);
    check("overlay declared WRITE", ef_tensor_cpu_access(overlay) == EF_CPU_ACCESS_WRITE, NULL);

    int fd = ef_tensor_clone_fd(canvas);
    exporter(fd, detail, sizeof detail);
    /* A negative row stride means tightly packed rows. */
    int64_t stride = ef_tensor_row_stride(canvas);
    snprintf(detail + strlen(detail), sizeof detail - strlen(detail), ", fd %d, stride %lld", fd,
             (long long)(stride > 0 ? stride : W * 4));
    check("canvas fd exported", fd >= 0, detail);
    if (access("/dev/dma_heap/linux,cma-uncached", F_OK) == 0)
        check("canvas from uncached CMA", strstr(detail, "cma-uncached") != NULL, NULL);
    if (fd >= 0)
        close(fd);

    /* Enforcement: no CPU read of either image, no CPU write of the canvas. */
    struct ef_tensor_view v;
    check("read map of the NONE canvas is EACCES",
          ef_tensor_map(canvas, EF_CPU_ACCESS_READ, &v) == EACCES, NULL);
    check("write map of the NONE canvas is EACCES",
          ef_tensor_map(canvas, EF_CPU_ACCESS_WRITE, &v) == EACCES, NULL);
    check("read map of the WRITE overlay is EACCES",
          ef_tensor_map(overlay, EF_CPU_ACCESS_READ, &v) == EACCES, NULL);
    const uint8_t blue[4] = {10, 20, 200, 255};
    check("write map of the WRITE overlay", fill(overlay, blue) == 0, NULL);

    ef_image_processor *cpu = ef_image_processor_new_with_backend(EF_COMPUTE_BACKEND_CPU);
    const ef_detect_box one = {0.25f, 0.25f, 0.75f, 0.75f, 0.9f, 0};
    check("CPU draw onto the NONE canvas is EACCES",
          ef_image_processor_draw_decoded_masks(cpu, canvas, &one, 1, NULL, 0, NULL, 1.0f, NULL,
                                                EF_COLOR_MODE_CLASS) == EACCES,
          NULL);
    check("CPU convert from the WRITE overlay is EACCES",
          ef_image_processor_convert(cpu, overlay, readback, NULL) == EACCES, NULL);
    ef_image_processor_free(cpu);

    /* GL: camera -> NONE frame, boxes over it into the NONE canvas, read back. */
    const uint8_t grey[4] = {100, 100, 100, 255};
    fill(camera, grey);
    check("GL convert into the NONE frame", ef_image_processor_convert(conv, camera, frame, NULL) == 0,
          NULL);
    check("GL draw into the NONE canvas",
          ef_image_processor_draw_decoded_masks(draw, canvas, &one, 1, NULL, 0, frame, 1.0f, NULL,
                                                EF_COLOR_MODE_CLASS) == 0,
          NULL);
    check("GL reads the canvas back", ef_image_processor_convert(rb, canvas, readback, NULL) == 0,
          NULL);
    int lx = (int)(0.25f * W) - 2; /* inside the 4 px outline, left of the box */
    pixel(readback, lx, H / 2, px);
    snprintf(detail, sizeof detail, "pixel %u,%u,%u, want ~178,50,50", px[0], px[1], px[2]);
    check("box outline blends at alpha 128", abs(px[0] - 178) <= 3 && abs(px[1] - 50) <= 3, detail);
    pixel(readback, W / 2, H / 2, px);
    snprintf(detail, sizeof detail, "pixel %u,%u,%u", px[0], px[1], px[2]);
    check("box interior shows the frame", px[0] == 100 && px[1] == 100 && px[2] == 100, detail);
    check("GL samples the WRITE overlay", ef_image_processor_convert(rb, overlay, readback, NULL) == 0,
          NULL);
    pixel(readback, 0, 0, px);
    check("overlay pixels arrive", px[0] == 10 && px[1] == 20 && px[2] == 200, NULL);

    /* Decoder: the captured outputs, timed, then drawn, timed. */
    char cfg[512];
    snprintf(cfg, sizeof cfg, "%s/edgefirst.json", dir);
    ef_decoder_params *params = ef_decoder_params_new();
    ef_decoder_params_set_config_file(params, cfg);
    ef_decoder_params_set_score_threshold(params, 0.25f);
    ef_decoder_params_set_iou_threshold(params, 0.7f);
    ef_decoder *dec = ef_decoder_new(params);
    ef_decoder_params_free(params);
    ef_tensor *outs[MAX_OUTPUTS];
    size_t n_outs = load_outputs(dir, outs);
    check("decoder and outputs", dec != NULL && n_outs > 0, NULL);
    if (dec && n_outs > 0) {
        struct ef_detect_box_list *boxes = NULL;
        int rc = ef_decoder_decode(dec, (const ef_tensor *const *)outs, n_outs, &boxes, NULL);
        snprintf(detail, sizeof detail, "%zu boxes", rc == 0 ? (size_t)ef_detect_box_list_len(boxes) : 0);
        check("decode", rc == 0 && ef_detect_box_list_len(boxes) > 0, detail);

        double *times = calloc(iters, sizeof *times);
        double cpu0 = now_us(CLOCK_PROCESS_CPUTIME_ID);
        for (int i = 0; i < iters; i++) {
            struct ef_detect_box_list *l = NULL;
            double t0 = now_us(CLOCK_MONOTONIC);
            ef_decoder_decode(dec, (const ef_tensor *const *)outs, n_outs, &l, NULL);
            times[i] = now_us(CLOCK_MONOTONIC) - t0;
            ef_detect_box_list_free(l);
        }
        report("c decode", times, iters, (now_us(CLOCK_PROCESS_CPUTIME_ID) - cpu0) / iters);

        int draws = iters < 200 ? iters : 200;
        cpu0 = now_us(CLOCK_PROCESS_CPUTIME_ID);
        for (int i = 0; i < draws; i++) {
            double t0 = now_us(CLOCK_MONOTONIC);
            ef_image_processor_draw_decoded_masks(draw, canvas, ef_detect_box_list_data(boxes),
                                                  ef_detect_box_list_len(boxes), NULL, 0, frame,
                                                  1.0f, NULL, EF_COLOR_MODE_CLASS);
            times[i] = now_us(CLOCK_MONOTONIC) - t0;
        }
        report("c GL draw into the NONE canvas", times, draws,
               (now_us(CLOCK_PROCESS_CPUTIME_ID) - cpu0) / draws);
        free(times);
        ef_detect_box_list_free(boxes);
    }
    for (size_t i = 0; i < n_outs; i++)
        ef_tensor_free(outs[i]);
    ef_decoder_free(dec);

    ef_tensor_free(camera);
    ef_tensor_free(frame);
    ef_tensor_free(canvas);
    ef_tensor_free(overlay);
    ef_tensor_free(readback);
    ef_image_processor_free(conv);
    ef_image_processor_free(draw);
    ef_image_processor_free(rb);
    printf("%s\n", failures ? "FAILED" : "all checks passed");
    return failures ? 1 : 0;
}
