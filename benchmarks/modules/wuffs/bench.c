/* SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * Wuffs (Google) JPEG decode-only benchmark — the memory-safe C arm of the
 * EdgeFirst decoder A/B (vs hal_cpu --decode-fmt rgb and the other RGB arms).
 *
 * Wuffs transpiles to a single C file (Apache-2.0), fetched at a pinned
 * release tag by `make deps` (see the Makefile). Its JPEG decoder has a
 * single fixed IDCT with no accurate/fast knob — measured **bit-exact**
 * against djpeg (libjpeg-turbo, accurate IDCT) over 4:4:4/4:2:0/DRI/
 * greyscale/large samples (max abs diff 0; 2026-08-13 parity check), so its
 * row compares within the accurate class with zero accuracy caveat. Output goes through Wuffs'
 * swizzler; this arm asks for 3-byte RGB and falls back through BGR and the
 * 4-byte formats if the swizzler refuses, recording the format actually used
 * in the CSV notes (a 4-byte destination writes 33% more bytes than the
 * other RGB arms and must be read accordingly).
 *
 * Steady-state allocation: the destination and work buffers grow to their
 * high-water mark and are reused, like the HAL and turbo arms. Wuffs decoder
 * objects are single-use per image, so the per-image reset
 * (wuffs_jpeg__decoder__initialize, which zeroes the decoder struct) is
 * inside the timed region — it is a cost every Wuffs user pays per frame.
 *
 * Harness choices that move the number live in ../cbench/cbench.h and are
 * identical across the C arms; see the three rules in benchmarks/README.md.
 *
 * Build: make -C benchmarks/modules/wuffs
 */

#include "cbench.h"

#define WUFFS_IMPLEMENTATION
#define WUFFS_CONFIG__MODULES
#define WUFFS_CONFIG__MODULE__BASE
#define WUFFS_CONFIG__MODULE__JPEG
#include "wuffs-v0.4.c"

typedef struct {
    wuffs_jpeg__decoder *dec;
    uint32_t pixfmt;
    size_t bytes_per_pixel;
    uint8_t *dst;
    size_t dst_cap;
    uint8_t *workbuf;
    size_t workbuf_cap;
    uint32_t width, height;
} WuffsBench;

typedef struct {
    uint32_t repr;
    size_t bpp;
    const char *name;
} WuffsCandidate;

#define WUFFS_CANDIDATE_COUNT 4

/* Default probe order: 3-byte formats first (matches the other RGB arms). */
static const WuffsCandidate wuffs_candidates_3bpp_first[WUFFS_CANDIDATE_COUNT] = {
    {WUFFS_BASE__PIXEL_FORMAT__RGB, 3, "RGB"},
    {WUFFS_BASE__PIXEL_FORMAT__BGR, 3, "BGR"},
    {WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL, 4, "RGBA_NONPREMUL"},
    {WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL, 4, "BGRA_NONPREMUL"},
};

/* EDGEFIRST_WUFFS_FORCE_4BPP=1 order: 4-byte formats first, to measure
 * Wuffs' native (non-swizzled) output path directly. */
static const WuffsCandidate wuffs_candidates_4bpp_first[WUFFS_CANDIDATE_COUNT] = {
    {WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL, 4, "RGBA_NONPREMUL"},
    {WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL, 4, "BGRA_NONPREMUL"},
    {WUFFS_BASE__PIXEL_FORMAT__RGB, 3, "RGB"},
    {WUFFS_BASE__PIXEL_FORMAT__BGR, 3, "BGR"},
};

/* Full per-image decode: reset, parse config, swizzle-decode the frame.
 * Returns NULL on success or a status message on failure. */
static const char *wuffs_decode_one(WuffsBench *b, const CbenchImage *img) {
    wuffs_base__status st = wuffs_jpeg__decoder__initialize(
        b->dec, sizeof__wuffs_jpeg__decoder(), WUFFS_VERSION, 0);
    if (!wuffs_base__status__is_ok(&st)) return wuffs_base__status__message(&st);

    wuffs_base__io_buffer src = wuffs_base__ptr_u8__reader(img->bytes, img->len, true);
    wuffs_base__image_config ic = {0};
    st = wuffs_jpeg__decoder__decode_image_config(b->dec, &ic, &src);
    if (!wuffs_base__status__is_ok(&st)) return wuffs_base__status__message(&st);

    uint32_t w = wuffs_base__pixel_config__width(&ic.pixcfg);
    uint32_t h = wuffs_base__pixel_config__height(&ic.pixcfg);
    wuffs_base__pixel_config__set(&ic.pixcfg, b->pixfmt, WUFFS_BASE__PIXEL_SUBSAMPLING__NONE, w,
                                  h);

    size_t need = (size_t)w * (size_t)h * b->bytes_per_pixel;
    if (need > b->dst_cap) {
        b->dst = (uint8_t *)realloc(b->dst, need);
        if (!b->dst) cbench_die("out of memory for %zu byte output", need);
        b->dst_cap = need;
    }
    wuffs_base__pixel_buffer pb = {0};
    st = wuffs_base__pixel_buffer__set_from_slice(&pb, &ic.pixcfg,
                                                  wuffs_base__make_slice_u8(b->dst, need));
    if (!wuffs_base__status__is_ok(&st)) return wuffs_base__status__message(&st);

    st = wuffs_jpeg__decoder__decode_frame_config(b->dec, NULL, &src);
    if (!wuffs_base__status__is_ok(&st)) return wuffs_base__status__message(&st);

    wuffs_base__range_ii_u64 wr = wuffs_jpeg__decoder__workbuf_len(b->dec);
    if (wr.max_incl > b->workbuf_cap) {
        b->workbuf = (uint8_t *)realloc(b->workbuf, wr.max_incl);
        if (!b->workbuf) cbench_die("out of memory for %llu byte workbuf",
                                    (unsigned long long)wr.max_incl);
        b->workbuf_cap = wr.max_incl;
    }
    st = wuffs_jpeg__decoder__decode_frame(
        b->dec, &pb, &src, WUFFS_BASE__PIXEL_BLEND__SRC,
        wuffs_base__make_slice_u8(b->workbuf, b->workbuf_cap), NULL);
    if (!wuffs_base__status__is_ok(&st)) return wuffs_base__status__message(&st);

    b->width = w;
    b->height = h;
    return NULL;
}

int main(int argc, char **argv) {
    CbenchArgs args = cbench_parse_args(argc, argv);
    if (strcmp(args.format, "rgb") != 0)
        cbench_die("--format must be rgb (this arm decodes via Wuffs' RGB-family swizzles)");

    size_t total = 0, n = 0;
    char **paths = cbench_list_jpegs(args.coco, &total);
    CbenchImage *images = cbench_preload(paths, total, args.limit, &n);

    WuffsBench b = {0};
    b.dec = wuffs_jpeg__decoder__alloc();
    if (!b.dec) cbench_die("wuffs_jpeg__decoder__alloc failed");

    /* Pick the destination format once, before anything is timed: prefer the
     * 3-byte formats that match the other RGB arms, fall back to 4-byte.
     * EDGEFIRST_WUFFS_FORCE_4BPP=1 skips straight to the 4-byte candidates,
     * for an isolated 3-byte-swizzle-vs-4-byte-native A/B on Wuffs' own
     * decoder (not a comparison against another arm) — see BENCHMARKS.md
     * § JPEG Decode's Wuffs accuracy/performance note. */
    int force_4bpp = getenv("EDGEFIRST_WUFFS_FORCE_4BPP") != NULL;
    const WuffsCandidate *candidates = force_4bpp ? wuffs_candidates_4bpp_first
                                                   : wuffs_candidates_3bpp_first;
    const char *fmt_name = NULL;
    const char *probe_err = "no candidate tried";
    for (size_t i = 0; i < WUFFS_CANDIDATE_COUNT; i++) {
        b.pixfmt = candidates[i].repr;
        b.bytes_per_pixel = candidates[i].bpp;
        probe_err = wuffs_decode_one(&b, &images[0]);
        if (!probe_err) {
            fmt_name = candidates[i].name;
            break;
        }
    }
    if (!fmt_name) cbench_die("no supported destination format (last error: %s)", probe_err);

    fprintf(stderr,
            "=== wuffs (decode-only, format=rgb, dst=%s, decoder=%zu B/frame reset) — %zu images "
            "===\n",
            fmt_name, sizeof__wuffs_jpeg__decoder(), n);

    /* Warm up on the first image, as every other arm does. */
    for (size_t i = 0; i < args.warmup; i++) {
        const char *err = wuffs_decode_one(&b, &images[0]);
        if (err) cbench_die("warmup decode failed: %s", err);
    }

    double *samples = (double *)malloc(n * sizeof(*samples));
    double total_mpix = 0.0;
    double cpu0 = cbench_process_cpu_seconds();
    double wall0 = cbench_now_ms();

    for (size_t i = 0; i < n; i++) {
        double t0 = cbench_now_ms();
        const char *err = wuffs_decode_one(&b, &images[i]);
        if (err) cbench_die("decode failed on %s: %s", images[i].name, err);
        samples[i] = cbench_now_ms() - t0;
        total_mpix += (double)b.width * (double)b.height / 1e6;
        if (args.verbose)
            fprintf(stderr, "  [%4zu/%zu] %.3f ms  %u×%u  %s\n", i + 1, n, samples[i], b.width,
                    b.height, images[i].name);
    }

    double wall_s = (cbench_now_ms() - wall0) / 1000.0;
    double cpu_s = cbench_process_cpu_seconds() - cpu0;
    double cpu_pct = wall_s > 0.0 ? 100.0 * cpu_s / wall_s : 0.0;

    char notes[192];
    snprintf(notes, sizeof(notes),
             "backend=wuffs-jpeg;scope=decode-only;harness=c;dst=%s;alloc=high-water-reuse",
             fmt_name);
    cbench_report(&args, "wuffs", notes, samples, n, total_mpix, cpu_pct);

    free(b.dec);
    free(b.dst);
    free(b.workbuf);
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
