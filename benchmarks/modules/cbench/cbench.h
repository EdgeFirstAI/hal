/* SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * Shared C harness for single-library decode arms (stb_image, Wuffs).
 *
 * Every definition that moves a number is copied verbatim from
 * modules/turbojpeg/bench.c, which in turn mirrors benchmarks/common: the
 * evenly-spaced image selection, preload-before-timing, CLOCK_MONOTONIC
 * around decode alone, percentile index, median-CI ranks, MP/s definition,
 * peak-RSS source, and CSV schema. A new arm built on this header therefore
 * differs from the Rust and TurboJPEG arms only in the decoder it calls.
 * If a definition changes here, change bench.c and benchmarks/common to
 * match (and vice versa).
 */

#ifndef EDGEFIRST_CBENCH_H
#define EDGEFIRST_CBENCH_H

#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

#ifdef __APPLE__
#include <pthread/qos.h>
#endif

/* macOS has no taskset-equivalent core pinning, and unpinned P-core/E-core
 * migration between rounds is a real, measured source of extra spread on
 * mbp-m2-max (see BENCHMARKS.md's discussion of this board's Max-spread
 * column). QOS_CLASS_USER_INTERACTIVE is an *advisory* hint, not a hard
 * affinity pin — the scheduler is still free to migrate the thread — but it
 * biases toward the performance cores and away from being pre-empted by
 * lower-QoS background work, which is the best available substitute for
 * taskset on this platform. Call once, before the timed loop. No-op
 * (including on non-Apple platforms, where taskset does the real job).
 */
static void cbench_pin_qos(void) {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noreturn, format(printf, 1, 2)))
#endif
static void
cbench_die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

/* --- timing / stats: identical definitions to benchmarks/common ---------- */

static double cbench_now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int cbench_cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* `idx = round(p * (n - 1))`, as TimingStats::from_samples does. */
static double cbench_percentile(const double *sorted, size_t n, double p) {
    if (n == 0) return 0.0;
    double pos = p * ((double)n - 1.0);
    size_t idx = (size_t)(pos + 0.5);
    if (idx >= n) idx = n - 1;
    return sorted[idx];
}

/* 0-based sorted-sample indices bounding the ~95% CI for the median, identical
 * to benchmarks/common median_ci_indices(): 1-based ranks
 * floor(n/2 - 1.96*sqrt(n)/2) and ceil(n/2 + 1 + 1.96*sqrt(n)/2), clamped. */
static void cbench_median_ci_indices(size_t n, size_t *lo, size_t *hi) {
    if (n == 0) {
        *lo = *hi = 0;
        return;
    }
    double nf = (double)n;
    double half_width = 0.98 * sqrt(nf); /* 1.96*sqrt(n)/2 */
    double lo_rank = floor(nf / 2.0 - half_width);
    double hi_rank = ceil(nf / 2.0 + 1.0 + half_width);
    if (lo_rank < 1.0) lo_rank = 1.0;
    if (hi_rank > nf) hi_rank = nf;
    *lo = (size_t)lo_rank - 1;
    *hi = (size_t)hi_rank - 1;
}

/* Peak RSS, matching peak_rss_mb(). Linux VmHWM is kB; Darwin ru_maxrss is bytes. */
static double cbench_peak_rss_mb(void) {
#ifdef __APPLE__
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) != 0) return 0.0;
    return (double)ru.ru_maxrss / (1024.0 * 1024.0);
#else
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return 0.0;
    char line[256];
    double mb = 0.0;
    while (fgets(line, sizeof(line), f)) {
        long kb;
        if (sscanf(line, "VmHWM: %ld kB", &kb) == 1) {
            mb = (double)kb / 1024.0;
            break;
        }
    }
    fclose(f);
    return mb;
#endif
}

/* Process CPU time; the decode loop is single-threaded, so this doubles as the
 * busiest-core figure. From the CPU clock rather than /proc/self/stat (jiffy
 * quantisation misreports short runs). */
static double cbench_process_cpu_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* --- image list: sorted, then evenly spaced, as list_jpegs() does -------- */

typedef struct {
    char *name;
    unsigned char *bytes;
    size_t len;
} CbenchImage;

static int cbench_cmp_str(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static int cbench_is_jpeg(const char *name) {
    const char *dot = strrchr(name, '.');
    return dot && (strcasecmp(dot, ".jpg") == 0 || strcasecmp(dot, ".jpeg") == 0);
}

static char **cbench_list_jpegs(const char *dir, size_t *out_n) {
    DIR *d = opendir(dir);
    if (!d) cbench_die("COCO directory not found: %s", dir);
    size_t cap = 1024, n = 0;
    char **paths = (char **)malloc(cap * sizeof(*paths));
    struct dirent *e;
    while ((e = readdir(d))) {
        if (!cbench_is_jpeg(e->d_name)) continue;
        if (n == cap) {
            cap *= 2;
            paths = (char **)realloc(paths, cap * sizeof(*paths));
        }
        size_t len = strlen(dir) + strlen(e->d_name) + 2;
        paths[n] = (char *)malloc(len);
        snprintf(paths[n], len, "%s/%s", dir, e->d_name);
        n++;
    }
    closedir(d);
    if (n == 0) cbench_die("no JPEG files in %s", dir);
    qsort(paths, n, sizeof(*paths), cbench_cmp_str);
    *out_n = n;
    return paths;
}

static CbenchImage *cbench_preload(char **paths, size_t total, size_t limit, size_t *out_n) {
    size_t n = (limit > 0 && limit < total) ? limit : total;
    CbenchImage *images = (CbenchImage *)calloc(n, sizeof(*images));
    for (size_t i = 0; i < n; i++) {
        /* paths[i * total / n] — the Rust harness's spacing, so every arm sees
         * the same subset of val2017 for a given --limit. */
        const char *path = (limit > 0 && limit < total) ? paths[i * total / n] : paths[i];
        FILE *f = fopen(path, "rb");
        if (!f) cbench_die("open %s", path);
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (len <= 0) cbench_die("empty %s", path);
        images[i].bytes = (unsigned char *)malloc((size_t)len);
        if (fread(images[i].bytes, 1, (size_t)len, f) != (size_t)len) cbench_die("read %s", path);
        fclose(f);
        images[i].len = (size_t)len;
        const char *slash = strrchr(path, '/');
        images[i].name = strdup(slash ? slash + 1 : path);
    }
    *out_n = n;
    return images;
}

/* --- CLI + report -------------------------------------------------------- */

typedef struct {
    const char *coco;
    const char *board;
    const char *format; /* modules validate; stb/wuffs arms are rgb-only */
    const char *csv_path;
    size_t limit, warmup;
    int verbose;
    /* --parity: skip the timed loop entirely and instead report pixel
     * parity (cosine/mean|d|/max|d|/PSNR) against dlopen'd libturbojpeg
     * islow — see cbench_run_parity(). `parity_lib` overrides the search,
     * same env var the Rust parity/turbojpeg arms use. */
    int parity;
    const char *parity_lib;
} CbenchArgs;

static void cbench_usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--coco DIR] [--limit N] [--warmup N] [--board LABEL]\n"
            "          [--format rgb] [--decode-only] [--csv PATH] [--verbose]\n"
            "          [--parity]\n",
            argv0);
    exit(2);
}

static CbenchArgs cbench_parse_args(int argc, char **argv) {
    CbenchArgs a = {
        .coco = getenv("EDGEFIRST_BENCH_COCO"),
        .board = "unknown",
        .format = "rgb",
        .csv_path = NULL,
        .limit = 50,
        .warmup = 10,
        .verbose = 0,
        .parity = 0,
        .parity_lib = getenv("EDGEFIRST_TURBOJPEG_LIB"),
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
#define CBENCH_NEXT(opt) \
    (i + 1 < argc ? argv[++i] : (cbench_usage(argv[0]), (char *)NULL))
        if (!strcmp(arg, "--coco")) a.coco = CBENCH_NEXT("--coco");
        else if (!strcmp(arg, "--limit")) a.limit = strtoul(CBENCH_NEXT("--limit"), NULL, 10);
        else if (!strcmp(arg, "--warmup")) a.warmup = strtoul(CBENCH_NEXT("--warmup"), NULL, 10);
        else if (!strcmp(arg, "--board")) a.board = CBENCH_NEXT("--board");
        else if (!strcmp(arg, "--format")) a.format = CBENCH_NEXT("--format");
        else if (!strcmp(arg, "--csv")) a.csv_path = CBENCH_NEXT("--csv");
        else if (!strcmp(arg, "--verbose")) a.verbose = 1;
        else if (!strcmp(arg, "--parity")) a.parity = 1;
        /* Accepted and ignored: these arms only do decode-only, which is what
         * the flag selects on the HAL side. */
        else if (!strcmp(arg, "--decode-only")) continue;
        else cbench_usage(argv[0]);
#undef CBENCH_NEXT
    }
    if (!a.coco) cbench_die("set EDGEFIRST_BENCH_COCO or pass --coco");
    return a;
}

/* Standard console line + CSV row (schema identical to benchmarks/common and
 * bench.c). `samples` is sorted in place. `notes` must not contain commas. */
static void cbench_report(const CbenchArgs *a, const char *module, const char *notes,
                          double *samples, size_t n, double total_mpix, double cpu_pct) {
    qsort(samples, n, sizeof(*samples), cbench_cmp_double);
    double sum_ms = 0.0;
    for (size_t i = 0; i < n; i++) sum_ms += samples[i];
    double p50 = cbench_percentile(samples, n, 0.50);
    double p95 = cbench_percentile(samples, n, 0.95);
    double p99 = cbench_percentile(samples, n, 0.99);
    double mean = n > 0 ? sum_ms / (double)n : 0.0;
    size_t ci_lo_idx, ci_hi_idx;
    cbench_median_ci_indices(n, &ci_lo_idx, &ci_hi_idx);
    double ci_lo = n > 0 ? samples[ci_lo_idx] : 0.0;
    double ci_hi = n > 0 ? samples[ci_hi_idx] : 0.0;
    double mpix_per_s = sum_ms > 0.0 ? total_mpix / (sum_ms / 1000.0) : 0.0;
    double rss = cbench_peak_rss_mb();

    fprintf(stderr,
            "  p50=%.3f ms  ci95=[%.3f,%.3f]  mean=%.3f  p95=%.3f ms  p99=%.3f ms  %.1f MP/s  "
            "peak RSS=%.1f MB  n=%zu\n",
            p50, ci_lo, ci_hi, mean, p95, p99, mpix_per_s, rss, n);
    fprintf(stderr, "  cpu: process=%.1f%%\n", cpu_pct);

    if (a->csv_path) {
        FILE *f = fopen(a->csv_path, "w");
        if (!f) cbench_die("create %s", a->csv_path);
        fprintf(f,
                "board,class,module,ms_p50,ms_p95,ms_p99,ms_mean,ms_p50_ci_lo,ms_p50_ci_hi,"
                "mpix_per_s,peak_rss_mb,"
                "cpu_pct_process,cpu_pct_system,cpu_pct_peak_core,n_images,notes\n");
        fprintf(f,
                "%s,decode,%s,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f,%.1f,%zu,%s\n",
                a->board, module, p50, p95, p99, mean, ci_lo, ci_hi, mpix_per_s, rss, cpu_pct,
                0.0, cpu_pct, n, notes);
        fclose(f);
    }
}

/* --- accuracy parity vs TurboJPEG islow (--parity) ----------------------- */
/*
 * Neither stb_image nor Wuffs does box-upsample chroma resampling of its
 * own, so both are always graded against plain accurate `islow` — never
 * `TJFLAG_FASTUPSAMPLE` (that flag is EdgeFirst's/turbo's matched-config
 * comparison, not applicable here). This retires the doc-comment-only
 * "measured against djpeg" claims each arm's header used to carry with a
 * real, reproducible, in-process number tied to a `--parity` invocation
 * anyone can re-run — see BENCHMARKS.md § JPEG Decode.
 */

#define CBENCH_TJPF_RGB 0
#define CBENCH_TJFLAG_ACCURATEDCT 4096

typedef void *(*cbench_tj_init_fn)(void);
typedef int (*cbench_tj_destroy_fn)(void *);
typedef int (*cbench_tj_header_fn)(void *, const unsigned char *, unsigned long, int *, int *,
                                   int *, int *);
typedef int (*cbench_tj_decomp_fn)(void *, const unsigned char *, unsigned long, unsigned char *,
                                   int, int, int, int, int);
typedef char *(*cbench_tj_error_fn)(void);

typedef struct {
    void *handle;
    cbench_tj_init_fn init;
    cbench_tj_destroy_fn destroy;
    cbench_tj_header_fn header;
    cbench_tj_decomp_fn decompress;
    cbench_tj_error_fn error;
    const char *path;
} CbenchTurboJpeg;

/* Same candidate list as rgb_parity / turbojpeg/bench.c's tj_load. */
static void cbench_tj_load(CbenchTurboJpeg *tj, const char *override) {
    if (override && *override) {
        tj->handle = dlopen(override, RTLD_NOW);
        if (!tj->handle) cbench_die("EDGEFIRST_TURBOJPEG_LIB=%s: dlopen failed: %s", override,
                                    dlerror());
        tj->path = override;
    }
    static const char *candidates[] = {
#ifdef __APPLE__
        "/opt/homebrew/opt/jpeg-turbo/lib/libturbojpeg.dylib",
        "/opt/homebrew/lib/libturbojpeg.dylib",
        "/usr/local/opt/jpeg-turbo/lib/libturbojpeg.dylib",
        "libturbojpeg.dylib",
        "libturbojpeg.0.dylib",
#endif
        "libturbojpeg.so.0",
        "libturbojpeg.so",
        "libturbojpeg.so.0.2.0",
        "/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0",
        "/usr/lib/aarch64-linux-gnu/libturbojpeg.so",
        "/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0",
        "/usr/lib/x86_64-linux-gnu/libturbojpeg.so",
        "/opt/libjpeg-turbo/lib64/libturbojpeg.so",
    };
    if (!tj->handle) {
        for (size_t i = 0; i < sizeof(candidates) / sizeof(*candidates); i++) {
            tj->handle = dlopen(candidates[i], RTLD_NOW);
            if (tj->handle) {
                tj->path = candidates[i];
                break;
            }
        }
    }
    if (!tj->handle) cbench_die("libturbojpeg not found (%s)", dlerror());

    tj->init = (cbench_tj_init_fn)dlsym(tj->handle, "tjInitDecompress");
    tj->destroy = (cbench_tj_destroy_fn)dlsym(tj->handle, "tjDestroy");
    tj->header = (cbench_tj_header_fn)dlsym(tj->handle, "tjDecompressHeader3");
    tj->decompress = (cbench_tj_decomp_fn)dlsym(tj->handle, "tjDecompress2");
    tj->error = (cbench_tj_error_fn)dlsym(tj->handle, "tjGetErrorStr");
    if (!tj->init || !tj->destroy || !tj->header || !tj->decompress)
        cbench_die("libturbojpeg at %s is missing required symbols", tj->path);

    Dl_info info;
    if (dladdr((void *)tj->init, &info) && info.dli_fname) tj->path = info.dli_fname;
}

static const char *cbench_tj_err(const CbenchTurboJpeg *tj) {
    return tj->error ? tj->error() : "(no tjGetErrorStr)";
}

/* Decode `img` to tight interleaved RGB into `*out` (realloc'd as needed,
 * `*out_cap` tracks the allocation so repeat calls reuse it — same
 * high-water-mark discipline the timed loop uses). Returns 0 on success,
 * nonzero to skip this image (recorded, not fatal — a handful of
 * arm-specific decode rejections, e.g. a shape one decoder declines, should
 * not abort an otherwise-representative parity run). */
typedef int (*CbenchParityDecodeFn)(const CbenchImage *img, unsigned char **out, size_t *out_cap,
                                    int *w, int *h);

static double cbench_parity_psnr(double mse) {
    return mse == 0.0 ? INFINITY : 10.0 * log10(255.0 * 255.0 / mse);
}

/* Runs the full corpus through `decode_fn` and through dlopen'd turbo
 * islow, reports aggregate cosine/mean|d|/max|d|/PSNR — the Table
 * A conformance numbers for an arm with no chroma-upsample filter choice
 * of its own (see the file-level comment above). */
static void cbench_run_parity(const CbenchArgs *a, const char *module,
                              CbenchParityDecodeFn decode_fn) {
    CbenchTurboJpeg tj = {0};
    cbench_tj_load(&tj, a->parity_lib);
    fprintf(stderr, "libturbojpeg: %s\n", tj.path);
    void *handle = tj.init();
    if (!handle) cbench_die("tjInitDecompress failed: %s", cbench_tj_err(&tj));

    size_t total = 0, n = 0;
    char **paths = cbench_list_jpegs(a->coco, &total);
    CbenchImage *images = cbench_preload(paths, total, a->limit, &n);

    unsigned char *own_buf = NULL;
    size_t own_cap = 0;
    unsigned char *tj_buf = NULL;
    size_t tj_cap = 0;

    size_t n_images = 0, n_skipped = 0, n_dim_mismatch = 0;
    double worst_cosine = 1.0, worst_psnr = INFINITY;
    int global_max_diff = 0;
    double sum_cosine = 0.0, sum_mean_abs = 0.0, sum_psnr = 0.0;

    for (size_t i = 0; i < n; i++) {
        int ow, oh;
        if (decode_fn(&images[i], &own_buf, &own_cap, &ow, &oh) != 0) {
            n_skipped++;
            continue;
        }

        int tw, th, subsamp, colorspace;
        if (tj.header(handle, images[i].bytes, (unsigned long)images[i].len, &tw, &th, &subsamp,
                      &colorspace))
            cbench_die("tjDecompressHeader3 failed on %s: %s", images[i].name, cbench_tj_err(&tj));
        size_t need = (size_t)tw * (size_t)th * 3;
        if (need > tj_cap) {
            tj_buf = (unsigned char *)realloc(tj_buf, need);
            if (!tj_buf) cbench_die("out of memory for %zu byte output", need);
            tj_cap = need;
        }
        if (tj.decompress(handle, images[i].bytes, (unsigned long)images[i].len, tj_buf, tw, 0, th,
                          CBENCH_TJPF_RGB, CBENCH_TJFLAG_ACCURATEDCT))
            cbench_die("tjDecompress2 failed on %s: %s", images[i].name, cbench_tj_err(&tj));

        if (tw != ow || th != oh) {
            n_dim_mismatch++;
            continue;
        }

        size_t count = (size_t)ow * (size_t)oh * 3;
        double dot = 0.0, na = 0.0, nf = 0.0, se = 0.0, sum_abs = 0.0;
        int max_diff = 0;
        for (size_t p = 0; p < count; p++) {
            double x = (double)own_buf[p], y = (double)tj_buf[p];
            dot += x * y;
            na += x * x;
            nf += y * y;
            int d = (int)own_buf[p] - (int)tj_buf[p];
            if (d < 0) d = -d;
            se += (double)(d * d);
            sum_abs += (double)d;
            if (d > max_diff) max_diff = d;
        }
        double cosine = dot / (sqrt(na) * sqrt(nf) > 1e-12 ? sqrt(na) * sqrt(nf) : 1e-12);
        double psnr = cbench_parity_psnr(se / (double)count);

        n_images++;
        sum_cosine += cosine;
        sum_mean_abs += sum_abs / (double)count;
        sum_psnr += (psnr > 99.0 ? 99.0 : psnr);
        if (cosine < worst_cosine) worst_cosine = cosine;
        if (psnr < worst_psnr) worst_psnr = psnr;
        if (max_diff > global_max_diff) global_max_diff = max_diff;
        if (a->verbose)
            fprintf(stderr, "%s: cosine=%.7f max|d|=%d psnr=%.2f\n", images[i].name, cosine,
                    max_diff, psnr);
    }

    if (n_images == 0) cbench_die("no images decoded on both arms");

    printf("== %s RGB vs turbo islow accurate over %zu images\n", module, n_images);
    printf("   (skipped: %zu unsupported, %zu dim mismatch)\n", n_skipped, n_dim_mismatch);
    printf("  cosine:   mean=%.7f  worst=%.7f\n", sum_cosine / (double)n_images, worst_cosine);
    printf("  mean|d|:  %.4f\n", sum_mean_abs / (double)n_images);
    printf("  max|d|:   %d\n", global_max_diff);
    printf("  psnr:     mean=%.2f dB  worst=%.2f dB\n", sum_psnr / (double)n_images, worst_psnr);

    free(own_buf);
    free(tj_buf);
    tj.destroy(handle);
    dlclose(tj.handle);
    for (size_t i = 0; i < n; i++) {
        free(images[i].bytes);
        free(images[i].name);
    }
    free(images);
    for (size_t i = 0; i < total; i++) free(paths[i]);
    free(paths);
}

#endif /* EDGEFIRST_CBENCH_H */
