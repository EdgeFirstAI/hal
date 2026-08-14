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

`--upsample` selects the chroma-upsampling accuracy class on the `--format
rgb` path (no-op on `yuv`, which never upsamples). Defaults to `accurate`
(libjpeg's fancy/triangle filter, `do_fancy_upsampling=TRUE`, also turbo's
own default); `fast` sets `TJFLAG_FASTUPSAMPLE` (box/nearest-neighbour) — the
accuracy class EdgeFirst's fused native-4:2:0→RGB write uses, so this is the
matched-accuracy-class comparator for that arm, same discipline as `--dct`.

`libturbojpeg` is resolved at run time by `dlopen`, so the aarch64 cross build
needs no board headers or libraries — only `aarch64-linux-gnu-gcc`. Set
`EDGEFIRST_TURBOJPEG_LIB=/path/to/libturbojpeg.so` to dlopen an exact path
instead of the built-in candidate search — for A/B'ing a source-built
libjpeg-turbo against the distro-packaged one on the same host.
