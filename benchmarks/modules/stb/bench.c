/* SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * stb_image decode-only benchmark — the single-header baseline arm of the
 * EdgeFirst decoder A/B (vs hal_cpu --decode-fmt rgb and the other RGB arms).
 *
 * stb_image is the recognizable floor: a single-file public-domain/MIT
 * decoder with a fixed accurate-class integer IDCT (derived from jidctint)
 * and SIMD IDCT/YCbCr paths (SSE2 auto-enabled on x86-64, NEON enabled here
 * for aarch64 via STBI_NEON). Accuracy class measured against djpeg
 * (libjpeg-turbo, accurate IDCT) over 4:4:4/4:2:0/DRI/greyscale/large
 * samples: max abs diff 3, mean <=0.07 — accurate-class rounding, not a
 * different accuracy trade (2026-08-13 parity check). RGB output only,
 * forced to 3 channels. Its API
 * has no decode-into-preallocated-buffer call, so this arm allocates the
 * output per frame by design — that per-call allocation is part of what any
 * stb_image user pays and is recorded in the CSV notes rather than hidden.
 *
 * Harness choices that move the number live in ../cbench/cbench.h and are
 * identical across the C arms; see the three rules in benchmarks/README.md.
 *
 * stb_image.h is fetched (pinned commit + sha256) by `make deps`; see the
 * Makefile. Build: make -C benchmarks/modules/stb
 */

#include "cbench.h"

#if defined(__aarch64__) || defined(_M_ARM64)
#define STBI_NEON
#endif
#define STBI_ONLY_JPEG
#define STBI_NO_STDIO
#define STB_IMAGE_IMPLEMENTATION
/* Third-party header; its own warnings are not ours to fix. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include "stb_image.h"
#pragma GCC diagnostic pop

int main(int argc, char **argv) {
    CbenchArgs args = cbench_parse_args(argc, argv);
    if (strcmp(args.format, "rgb") != 0)
        cbench_die("--format must be rgb (stb_image exposes no raw-YUV output)");

    size_t total = 0, n = 0;
    char **paths = cbench_list_jpegs(args.coco, &total);
    CbenchImage *images = cbench_preload(paths, total, args.limit, &n);

#if defined(STBI_NEON)
    const char *simd = "neon";
#elif defined(__x86_64__) || defined(_M_X64)
    const char *simd = "sse2";
#else
    const char *simd = "scalar";
#endif
    fprintf(stderr, "=== stb_image (decode-only, format=rgb, simd=%s) — %zu images ===\n", simd,
            n);

    /* Warm up on the first image, as every other arm does. */
    for (size_t i = 0; i < args.warmup; i++) {
        int w, h, comp;
        unsigned char *px = stbi_load_from_memory(images[0].bytes, (int)images[0].len, &w, &h,
                                                  &comp, 3);
        if (!px) cbench_die("warmup decode failed: %s", stbi_failure_reason());
        stbi_image_free(px);
    }

    double *samples = (double *)malloc(n * sizeof(*samples));
    double total_mpix = 0.0;
    double cpu0 = cbench_process_cpu_seconds();
    double wall0 = cbench_now_ms();

    for (size_t i = 0; i < n; i++) {
        int w, h, comp;
        double t0 = cbench_now_ms();
        unsigned char *px = stbi_load_from_memory(images[i].bytes, (int)images[i].len, &w, &h,
                                                  &comp, 3);
        if (!px)
            cbench_die("decode failed on %s: %s", images[i].name, stbi_failure_reason());
        /* The free stays inside the timed region: allocate-decode-free is the
         * complete per-frame cost of stb_image's allocating API in a hot loop. */
        stbi_image_free(px);
        samples[i] = cbench_now_ms() - t0;
        total_mpix += (double)w * (double)h / 1e6;
        if (args.verbose)
            fprintf(stderr, "  [%4zu/%zu] %.3f ms  %d×%d  %s\n", i + 1, n, samples[i], w, h,
                    images[i].name);
    }

    double wall_s = (cbench_now_ms() - wall0) / 1000.0;
    double cpu_s = cbench_process_cpu_seconds() - cpu0;
    double cpu_pct = wall_s > 0.0 ? 100.0 * cpu_s / wall_s : 0.0;

    char notes[160];
    snprintf(notes, sizeof(notes),
             "backend=stb_image-rgb;scope=decode-only;harness=c;simd=%s;alloc=per-call", simd);
    cbench_report(&args, "stb", notes, samples, n, total_mpix, cpu_pct);

    free(samples);
    for (size_t i = 0; i < n; i++) {
        free(images[i].bytes);
        free(images[i].name);
    }
    free(images);
    for (size_t i = 0; i < total; i++) free(paths[i]);
    free(paths);
    return 0;
}
