# turbojpeg

libjpeg-turbo, the reference arm of the EdgeFirst decoder A/B.

Native C, because the comparison is between two native libraries — EdgeFirst
(Rust) and libjpeg-turbo (C) — and the harness around each has to be equally
thin. `bench.c` mirrors `benchmarks/common` where it matters: the same
evenly-spaced image selection, the same preload before timing,
`CLOCK_MONOTONIC` around decode alone, the same percentile index, the same MP/s
definition, and the same CSV schema.

```bash
make                    # -> build/turbojpeg_bench
make aarch64            # -> build/turbojpeg_bench.aarch64 (boards; no sysroot needed)

# Decoder A/B vs HAL --decode-only (JPEG → memory; no letterbox)
./build/turbojpeg_bench --decode-only --format yuv --limit 50 --board x86-desktop \
  --csv ../../results/x86-desktop/turbojpeg_yuv.csv
./build/turbojpeg_bench --decode-only --format rgb --limit 50 --board x86-desktop \
  --csv ../../results/x86-desktop/turbojpeg_rgb.csv
```

| `--format` | API | Fair HAL compare |
|------------|-----|------------------|
| `yuv` | `tjDecompressToYUV2` (planar YUV) | `--decode-only --decode-fmt native` |
| `rgb` | `tjDecompress2` `TJPF_RGB` | `--decode-only --decode-fmt rgb` |

Layouts on the YUV arm differ (HAL NV* vs TurboJPEG planar); both are
decode-to-YUV without an RGB colour step. Letterbox/convert is preprocessing
beyond this module's scope.

`--dct` selects the IDCT accuracy class. It defaults to `accurate` (`islow`),
which is both what HAL implements and libjpeg-turbo's own decompression default;
`fast` selects `ifast`, a different accuracy class that runs 5–8% quicker.

`libturbojpeg` is resolved at run time by `dlopen`, so the aarch64 cross build
needs no board headers or libraries — only `aarch64-linux-gnu-gcc`.
