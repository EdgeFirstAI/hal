# hal_cpu

HAL software JPEG decode. Default path also runs HAL CPU letterbox convert
(`EDGEFIRST_FORCE_BACKEND=cpu`). For decoder A/B vs TurboJPEG, use
`--decode-only` (no `ImageProcessor::convert`).

```bash
# Decode-only YUV (native NV12/16/24) vs TurboJPEG --format yuv
cargo run --profile profiling -p hal_cpu -- --decode-only --decode-fmt native \
  --limit 50 --board x86-desktop --tensor-mem mem \
  --csv ../../results/x86-desktop/hal_cpu_yuv.csv

# Decode-only fused RGB (4:4:4) vs TurboJPEG --format rgb
cargo run --profile profiling -p hal_cpu -- --decode-only --decode-fmt rgb \
  --limit 50 --board x86-desktop --tensor-mem mem \
  --csv ../../results/x86-desktop/hal_cpu_rgb.csv

# Article-1 e2e (decode + letterbox) — not part of the TurboJPEG decoder table
cargo run --profile profiling -p hal_cpu -- --limit 50 --board x86-desktop \
  --csv ../../results/x86-desktop/hal_cpu_e2e.csv
```

Build from the `benchmarks/` workspace root.
