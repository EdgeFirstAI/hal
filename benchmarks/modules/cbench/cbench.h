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
} CbenchArgs;

static void cbench_usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--coco DIR] [--limit N] [--warmup N] [--board LABEL]\n"
            "          [--format rgb] [--decode-only] [--csv PATH] [--verbose]\n",
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

#endif /* EDGEFIRST_CBENCH_H */
