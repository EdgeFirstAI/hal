/* SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
 * SPDX-License-Identifier: Apache-2.0
 *
 * libjpeg-turbo decode-only benchmark — the TurboJPEG arm of the EdgeFirst
 * decoder A/B.
 *
 *   --format yuv   tjDecompressToYUV2   (vs hal_cpu --decode-fmt native)
 *   --format rgb   tjDecompress2 TJPF_RGB (vs hal_cpu --decode-fmt rgb)
 *
 * Native C on purpose. The comparison is between two native decoding
 * libraries, so the harness around libturbojpeg has to be as thin as the Rust
 * one around EdgeFirst: a Python driver puts interpreter dispatch and an FFI
 * marshalling layer inside the timed region on one side of the A/B only.
 *
 * Mirrors `benchmarks/common` (the hal_cpu harness) exactly where it affects
 * the numbers: same evenly-spaced image selection, same preload-before-timing,
 * same CLOCK_MONOTONIC around decode alone, same percentile index, same MP/s
 * definition, same peak-RSS source, same CSV schema.
 *
 * libturbojpeg is resolved with dlopen rather than linked, so this
 * cross-compiles for the boards without needing their headers or libraries.
 *
 * Build: make -C benchmarks/modules/turbojpeg
 */

#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>

/* --- TurboJPEG ABI (subset). Values from turbojpeg.h. ------------------- */
#define TJPF_RGB 0
#define TJPF_BGR 1
#define TJFLAG_FASTDCT 2048
#define TJFLAG_ACCURATEDCT 4096

typedef void *(*tj_init_fn)(void);
typedef int (*tj_destroy_fn)(void *);
typedef int (*tj_header_fn)(void *, const unsigned char *, unsigned long, int *, int *, int *,
                            int *);
typedef int (*tj_to_yuv_fn)(void *, const unsigned char *, unsigned long, unsigned char *, int, int,
                            int, int);
typedef int (*tj_decomp_fn)(void *, const unsigned char *, unsigned long, unsigned char *, int, int,
                            int, int, int);
typedef unsigned long (*tj_bufsize_yuv_fn)(int, int, int, int);
typedef char *(*tj_error_fn)(void);

static struct {
    void *handle;
    tj_init_fn init;
    tj_destroy_fn destroy;
    tj_header_fn header;
    tj_to_yuv_fn to_yuv;
    tj_decomp_fn decompress;
    tj_bufsize_yuv_fn bufsize_yuv;
    tj_error_fn error;
    const char *path;
} tj;

static void die(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(1);
}

static const char *tj_err(void) { return tj.error ? tj.error() : "(no tjGetErrorStr)"; }

/* Same candidate list as the retired Python driver: distributions disagree on
 * whether libturbojpeg ships a bare .so, and Jetson/Ubuntu images often have
 * only the versioned one. */
static void tj_load(void) {
    static const char *candidates[] = {
        "libturbojpeg.so.0",
        "libturbojpeg.so",
        "libturbojpeg.so.0.2.0",
        "/usr/lib/aarch64-linux-gnu/libturbojpeg.so.0",
        "/usr/lib/aarch64-linux-gnu/libturbojpeg.so",
        "/usr/lib/x86_64-linux-gnu/libturbojpeg.so.0",
        "/usr/lib/x86_64-linux-gnu/libturbojpeg.so",
        "/opt/libjpeg-turbo/lib64/libturbojpeg.so",
    };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(*candidates); i++) {
        tj.handle = dlopen(candidates[i], RTLD_NOW);
        if (tj.handle) {
            tj.path = candidates[i];
            break;
        }
    }
    if (!tj.handle) die("libturbojpeg not found (%s)", dlerror());

    tj.init = (tj_init_fn)dlsym(tj.handle, "tjInitDecompress");
    tj.destroy = (tj_destroy_fn)dlsym(tj.handle, "tjDestroy");
    tj.header = (tj_header_fn)dlsym(tj.handle, "tjDecompressHeader3");
    tj.to_yuv = (tj_to_yuv_fn)dlsym(tj.handle, "tjDecompressToYUV2");
    tj.decompress = (tj_decomp_fn)dlsym(tj.handle, "tjDecompress2");
    tj.bufsize_yuv = (tj_bufsize_yuv_fn)dlsym(tj.handle, "tjBufSizeYUV2");
    tj.error = (tj_error_fn)dlsym(tj.handle, "tjGetErrorStr");
    if (!tj.init || !tj.destroy || !tj.header || !tj.to_yuv || !tj.decompress || !tj.bufsize_yuv)
        die("libturbojpeg at %s is missing required symbols", tj.path);
}

/* --- timing / stats: identical definitions to benchmarks/common ---------- */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* `idx = round(p * (n - 1))`, as TimingStats::from_samples does. */
static double percentile(const double *sorted, size_t n, double p) {
    if (n == 0) return 0.0;
    double pos = p * ((double)n - 1.0);
    size_t idx = (size_t)(pos + 0.5);
    if (idx >= n) idx = n - 1;
    return sorted[idx];
}

/* VmHWM, matching peak_rss_mb(). */
static double peak_rss_mb(void) {
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
}

/* Process CPU time over the run; the decode loop is single-threaded, so this
 * doubles as the busiest-core figure the Rust harness samples per-core.
 *
 * From the CPU clock rather than /proc/self/stat: the latter is quantised to
 * clock ticks, which at a small --limit is a large fraction of the measured
 * window and reports impossible percentages. */
static double process_cpu_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts) != 0) return 0.0;
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* --- image list: sorted, then evenly spaced, as list_jpegs() does -------- */

typedef struct {
    char *name;
    unsigned char *bytes;
    size_t len;
} Image;

static int cmp_str(const void *a, const void *b) {
    return strcmp(*(const char *const *)a, *(const char *const *)b);
}

static int is_jpeg(const char *name) {
    const char *dot = strrchr(name, '.');
    return dot && (strcasecmp(dot, ".jpg") == 0 || strcasecmp(dot, ".jpeg") == 0);
}

static char **list_jpegs(const char *dir, size_t *out_n) {
    DIR *d = opendir(dir);
    if (!d) die("COCO directory not found: %s", dir);
    size_t cap = 1024, n = 0;
    char **paths = malloc(cap * sizeof(*paths));
    struct dirent *e;
    while ((e = readdir(d))) {
        if (!is_jpeg(e->d_name)) continue;
        if (n == cap) {
            cap *= 2;
            paths = realloc(paths, cap * sizeof(*paths));
        }
        size_t len = strlen(dir) + strlen(e->d_name) + 2;
        paths[n] = malloc(len);
        snprintf(paths[n], len, "%s/%s", dir, e->d_name);
        n++;
    }
    closedir(d);
    if (n == 0) die("no JPEG files in %s", dir);
    qsort(paths, n, sizeof(*paths), cmp_str);
    *out_n = n;
    return paths;
}

static Image *preload(char **paths, size_t total, size_t limit, size_t *out_n) {
    size_t n = (limit > 0 && limit < total) ? limit : total;
    Image *images = calloc(n, sizeof(*images));
    for (size_t i = 0; i < n; i++) {
        /* paths[i * total / n] — the Rust harness's spacing, so both arms see
         * the same subset of val2017 for a given --limit. */
        const char *path = (limit > 0 && limit < total) ? paths[i * total / n] : paths[i];
        FILE *f = fopen(path, "rb");
        if (!f) die("open %s", path);
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        fseek(f, 0, SEEK_SET);
        if (len <= 0) die("empty %s", path);
        images[i].bytes = malloc((size_t)len);
        if (fread(images[i].bytes, 1, (size_t)len, f) != (size_t)len) die("read %s", path);
        fclose(f);
        images[i].len = (size_t)len;
        const char *slash = strrchr(path, '/');
        images[i].name = strdup(slash ? slash + 1 : path);
    }
    *out_n = n;
    return images;
}

/* --- decode ------------------------------------------------------------- */

typedef struct {
    void *handle;
    int pixel_format; /* -1 = YUV, else TJPF_* */
    int dct_flag;     /* TJFLAG_ACCURATEDCT or TJFLAG_FASTDCT */
    unsigned char *buf;
    size_t buf_cap;
    int width, height;
} Decoder;

static void decode_one(Decoder *d, const Image *img) {
    int w, h, subsamp, colorspace;
    if (tj.header(d->handle, img->bytes, (unsigned long)img->len, &w, &h, &subsamp, &colorspace))
        die("tjDecompressHeader3 failed on %s: %s", img->name, tj_err());

    size_t need;
    if (d->pixel_format < 0) {
        need = (size_t)tj.bufsize_yuv(w, 1, h, subsamp);
    } else {
        need = (size_t)w * (size_t)h * 3;
    }
    if (need > d->buf_cap) {
        /* Grows to the high-water mark and is reused, as the HAL arm's tensor
         * is: neither side pays an allocation per frame in steady state. */
        d->buf = realloc(d->buf, need);
        if (!d->buf) die("out of memory for %zu byte output", need);
        d->buf_cap = need;
    }

    if (d->pixel_format < 0) {
        if (tj.to_yuv(d->handle, img->bytes, (unsigned long)img->len, d->buf, w, 1, h,
                      d->dct_flag))
            die("tjDecompressToYUV2 failed on %s: %s", img->name, tj_err());
    } else {
        if (tj.decompress(d->handle, img->bytes, (unsigned long)img->len, d->buf, w, 0, h,
                          d->pixel_format, d->dct_flag))
            die("tjDecompress2 failed on %s: %s", img->name, tj_err());
    }
    d->width = w;
    d->height = h;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s [--coco DIR] [--limit N] [--warmup N] [--board LABEL]\n"
            "          [--format yuv|rgb|bgr] [--dct accurate|fast] [--decode-only]\n"
            "          [--csv PATH] [--verbose]\n",
            argv0);
    exit(2);
}

int main(int argc, char **argv) {
    const char *coco = getenv("EDGEFIRST_BENCH_COCO");
    const char *board = "unknown";
    const char *format = "yuv";
    const char *dct = "accurate";
    const char *csv_path = NULL;
    size_t limit = 50, warmup = 10;
    int verbose = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
#define NEXT(opt)                                                                                  \
    (i + 1 < argc ? argv[++i] : (usage(argv[0]), (char *)NULL)) /* opt requires a value */
        if (!strcmp(a, "--coco")) coco = NEXT("--coco");
        else if (!strcmp(a, "--limit")) limit = strtoul(NEXT("--limit"), NULL, 10);
        else if (!strcmp(a, "--warmup")) warmup = strtoul(NEXT("--warmup"), NULL, 10);
        else if (!strcmp(a, "--board")) board = NEXT("--board");
        else if (!strcmp(a, "--format")) format = NEXT("--format");
        else if (!strcmp(a, "--dct")) dct = NEXT("--dct");
        else if (!strcmp(a, "--csv")) csv_path = NEXT("--csv");
        else if (!strcmp(a, "--verbose")) verbose = 1;
        /* Accepted and ignored: this binary only does decode-only, which is
         * what the flag selects on the HAL side. */
        else if (!strcmp(a, "--decode-only")) continue;
        else usage(argv[0]);
#undef NEXT
    }
    if (!coco) die("set EDGEFIRST_BENCH_COCO or pass --coco");

    int pixel_format;
    if (!strcmp(format, "yuv")) pixel_format = -1;
    else if (!strcmp(format, "rgb")) pixel_format = TJPF_RGB;
    else if (!strcmp(format, "bgr")) pixel_format = TJPF_BGR;
    else die("--format must be yuv, rgb or bgr (got %s)", format);

    /* HAL's IDCT is the accurate one (`islow`: CONST_BITS 13, PASS1_BITS 2), so
     * that is the default here. TJFLAG_FASTDCT selects libjpeg-turbo's `ifast`
     * kernel, which is a different accuracy class — faster, and not what HAL is
     * doing. Kept selectable so the gap between the two can be reported. */
    int dct_flag;
    if (!strcmp(dct, "accurate")) dct_flag = TJFLAG_ACCURATEDCT;
    else if (!strcmp(dct, "fast")) dct_flag = TJFLAG_FASTDCT;
    else die("--dct must be accurate or fast (got %s)", dct);

    tj_load();

    size_t total = 0, n = 0;
    char **paths = list_jpegs(coco, &total);
    Image *images = preload(paths, total, limit, &n);

    const char *backend = pixel_format < 0   ? "libturbojpeg-yuv"
                          : pixel_format == TJPF_RGB ? "libturbojpeg-rgb"
                                                     : "libturbojpeg-bgr";
    fprintf(stderr, "=== turbojpeg (decode-only, format=%s, dct=%s) — %zu images, decoder=%s ===\n",
            format, dct, n, backend);
    fprintf(stderr, "  lib: %s\n", tj.path);

    Decoder d = {0};
    d.handle = tj.init();
    if (!d.handle) die("tjInitDecompress failed: %s", tj_err());
    d.pixel_format = pixel_format;
    d.dct_flag = dct_flag;

    /* Warm up over the whole measured pipeline so decoder setup and first-touch
     * page faults land here rather than in the samples. */
    for (size_t i = 0; i < warmup; i++) decode_one(&d, &images[0]);

    double *samples = malloc(n * sizeof(*samples));
    double total_mpix = 0.0;
    double cpu0 = process_cpu_seconds();
    double wall0 = now_ms();

    for (size_t i = 0; i < n; i++) {
        double t0 = now_ms();
        decode_one(&d, &images[i]);
        samples[i] = now_ms() - t0;
        total_mpix += (double)d.width * (double)d.height / 1e6;
        if (verbose)
            fprintf(stderr, "  [%4zu/%zu] %.3f ms  %d×%d  %s\n", i + 1, n, samples[i], d.width,
                    d.height, images[i].name);
    }

    double wall_s = (now_ms() - wall0) / 1000.0;
    double cpu_s = process_cpu_seconds() - cpu0;
    double cpu_pct = wall_s > 0.0 ? 100.0 * cpu_s / wall_s : 0.0;

    qsort(samples, n, sizeof(*samples), cmp_double);
    double sum_ms = 0.0;
    for (size_t i = 0; i < n; i++) sum_ms += samples[i];
    double p50 = percentile(samples, n, 0.50);
    double p95 = percentile(samples, n, 0.95);
    double p99 = percentile(samples, n, 0.99);
    double mpix_per_s = sum_ms > 0.0 ? total_mpix / (sum_ms / 1000.0) : 0.0;
    double rss = peak_rss_mb();

    fprintf(stderr, "  p50=%.3f ms  p95=%.3f ms  p99=%.3f ms  %.1f MP/s  peak RSS=%.1f MB  n=%zu\n",
            p50, p95, p99, mpix_per_s, rss, n);
    fprintf(stderr, "  cpu: process=%.1f%%\n", cpu_pct);

    if (csv_path) {
        char notes[160];
        snprintf(notes, sizeof(notes), "backend=%s;scope=decode-only;harness=c;dct=%s", backend,
                 dct);
        FILE *f = fopen(csv_path, "w");
        if (!f) die("create %s", csv_path);
        fprintf(f,
                "board,class,module,ms_p50,ms_p95,ms_p99,mpix_per_s,peak_rss_mb,"
                "cpu_pct_process,cpu_pct_system,cpu_pct_peak_core,n_images,notes\n");
        fprintf(f, "%s,decode,turbojpeg,%.3f,%.3f,%.3f,%.3f,%.1f,%.1f,%.1f,%.1f,%zu,%s\n",
                board, p50, p95, p99, mpix_per_s, rss, cpu_pct, 0.0, cpu_pct, n,
                notes);
        fclose(f);
    }

    tj.destroy(d.handle);
    free(d.buf);
    free(samples);
    for (size_t i = 0; i < n; i++) {
        free(images[i].bytes);
        free(images[i].name);
    }
    free(images);
    for (size_t i = 0; i < total; i++) free(paths[i]);
    free(paths);
    dlclose(tj.handle);
    return 0;
}
