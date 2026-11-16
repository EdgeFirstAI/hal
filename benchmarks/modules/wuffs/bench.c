/* SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * Wuffs (Google) JPEG decode-only benchmark: the memory-safe C arm of the
 * EdgeFirst decoder A/B (vs hal_cpu --decode-fmt rgb and the other RGB arms).
 *
 * Wuffs transpiles to a single C file (Apache-2.0), fetched at a pinned
 * release tag by `make deps` (see the Makefile). Output goes through Wuffs'
 * swizzler; this arm asks for 3-byte RGB and falls back through BGR and the
 * 4-byte formats if the swizzler refuses, recording the format used in the
 * CSV notes.
 *
 * The destination and work buffers grow to their high-water mark and are
 * reused. The per-image decoder reset (wuffs_jpeg__decoder__initialize) is
 * inside the timed region.
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

/* Default probe order: 3-byte formats first. */
static const WuffsCandidate wuffs_candidates_3bpp_first[WUFFS_CANDIDATE_COUNT] = {
    {WUFFS_BASE__PIXEL_FORMAT__RGB, 3, "RGB"},
    {WUFFS_BASE__PIXEL_FORMAT__BGR, 3, "BGR"},
    {WUFFS_BASE__PIXEL_FORMAT__RGBA_NONPREMUL, 4, "RGBA_NONPREMUL"},
    {WUFFS_BASE__PIXEL_FORMAT__BGRA_NONPREMUL, 4, "BGRA_NONPREMUL"},
};

/* EDGEFIRST_WUFFS_FORCE_4BPP=1 order: 4-byte formats first. */
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

/* --parity: matches CbenchParityDecodeFn. Always decodes to 3-byte RGB.
 * Lazily allocates its own WuffsBench decoder/scratch on first call and
 * reuses it across the corpus. */
static int wuffs_parity_decode(const CbenchImage *img, unsigned char **out, size_t *out_cap,
                               int *w, int *h) {
    static WuffsBench b = {0};
    if (!b.dec) {
        b.dec = wuffs_jpeg__decoder__alloc();
        if (!b.dec) cbench_die("wuffs_jpeg__decoder__alloc failed");
        b.pixfmt = WUFFS_BASE__PIXEL_FORMAT__RGB;
        b.bytes_per_pixel = 3;
    }
    const char *err = wuffs_decode_one(&b, img);
    if (err) return 1;
    size_t need = (size_t)b.width * (size_t)b.height * 3;
    if (need > *out_cap) {
        *out = (unsigned char *)realloc(*out, need);
        if (!*out) cbench_die("out of memory for %zu byte output", need);
        *out_cap = need;
    }
    memcpy(*out, b.dst, need);
    *w = (int)b.width;
    *h = (int)b.height;
    return 0;
}

int main(int argc, char **argv) {
    cbench_pin_qos();
    CbenchArgs args = cbench_parse_args(argc, argv);
    if (strcmp(args.format, "rgb") != 0)
        cbench_die("--format must be rgb (this arm decodes via Wuffs' RGB-family swizzles)");
    if (args.parity) {
        cbench_run_parity(&args, "wuffs", wuffs_parity_decode);
        return 0;
    }

    size_t total = 0, n = 0;
    char **paths = cbench_list_jpegs(args.coco, &total);
    CbenchImage *images = cbench_preload(paths, total, args.limit, &n);

    WuffsBench b = {0};
    b.dec = wuffs_jpeg__decoder__alloc();
    if (!b.dec) cbench_die("wuffs_jpeg__decoder__alloc failed");

    /* Pick the destination format once, before anything is timed: prefer the
     * 3-byte formats, fall back to 4-byte. EDGEFIRST_WUFFS_FORCE_4BPP=1 skips
     * straight to the 4-byte candidates. */
    const char *force_4bpp_env = getenv("EDGEFIRST_WUFFS_FORCE_4BPP");
    int force_4bpp = force_4bpp_env != NULL && strcmp(force_4bpp_env, "1") == 0;
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

    /* Warm up on the first image. */
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
