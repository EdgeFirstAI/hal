# EdgeFirst HAL — JPEG → Model-Input Benchmarks

Comparison modules for JPEG decode and the decode → letterboxed 640×640 RGB
pipeline. This tree is **excluded from the Cargo workspace** and from CI path
triggers so TurboJPEG never enters the HAL dependency graph.

Every arm is a native binary. The decoder A/B compares two native libraries, so
a harness that put an interpreter inside the timed region on one side would be
measuring the interpreter.

See [`BENCHMARKS_PLAN.md`](../BENCHMARKS_PLAN.md) for the full design.

## Layout

```text
benchmarks/
├── common/           # Shared Rust helpers (COCO walk, timing, CSV)
├── docker/           # Multi-processor container image (Dockerfile + entrypoint)
├── probe/            # Capability probe script
├── scripts/          # sync-coco, deploy-and-run, decode-ab-matrix,
│                     # perf-compare-decode, fixture-decode-matrix
├── modules/
│   ├── hal_cpu/      # HAL decode + HAL CPU convert          (Rust)
│   ├── turbojpeg/    # libjpeg-turbo decode, reference arm    (C)
│   ├── hal_gl/       # HAL decode → NV12 + HAL GL convert     (Rust)
│   ├── hal_g2d/      # HAL decode → NV12 + HAL G2D (i.MX)     (Rust)
│   └── hal_v4l2_gl/  # HAL V4L2 JPEG + HAL GL (i.MX 95)       (Rust)
└── results/<board>/  # CSV outputs (gitignored)
```

## Constants

| Item | Value |
|------|--------|
| Dataset | `$EDGEFIRST_BENCH_COCO` or `~/Datasets/COCO/val2017` |
| Output | 640×640 centered letterbox, pad **114**, RGB |
| Smoke | `--limit 50` (default), 10 warmup |
| Full | omit `--limit` (all 5000 JPEGs); latency only, no mAP |

## Build profile

Investigation / on-target runs use Cargo **`profiling`** (release opts +
debug symbols, no strip) so `perf` and chrome traces resolve symbols:

```bash
cd benchmarks
cargo zigbuild --profile profiling --target aarch64-unknown-linux-gnu -p hal_cpu
# deploy-and-run.sh / perf-compare-decode.sh / fixture-decode-matrix.sh
# default to CARGO_PROFILE=profiling
```

Optional chrome/Perfetto capture (HAL `tracing` spans):

```bash
EDGEFIRST_TRACE=/tmp/hal-trace.json ./target/profiling/hal_cpu --limit 20 ...
# open /tmp/hal-trace.json in https://ui.perfetto.dev/
```

## Host smoke

```bash
# HAL (profiling profile preferred)
cargo run --profile profiling --manifest-path benchmarks/modules/hal_cpu/Cargo.toml -- \
  --limit 50 --board x86-desktop --csv benchmarks/results/x86-desktop/hal_cpu.csv

# TurboJPEG reference arm
make -C benchmarks/modules/turbojpeg
./benchmarks/modules/turbojpeg/build/turbojpeg_bench --limit 50 --board x86-desktop \
  --decode-only --format yuv --csv benchmarks/results/x86-desktop/turbojpeg.csv
```

## Docker / multi-processor

The `benchmarks/docker/` image packages `hal_cpu` so the same binary can be run
on varied Intel (and other) CPUs to collect per-processor numbers and guide
ISA-tier opts (`IntelTier` / `NeonTier`).

Default container work is a **decode-only** YUV + RGB matrix (no letterbox /
`ImageProcessor::convert`):

| Arm | HAL | TurboJPEG |
|-----|-----|-----------|
| YUV | `--decode-only --decode-fmt native` (NV12/16/24) | `--decode-only --format yuv` (`tjDecompressToYUV2`) |
| RGB | `--decode-only --decode-fmt rgb` (fused, 4:4:4) | `--decode-only --format rgb` (`TJPF_RGB`) |

Layouts on the YUV arm differ (HAL semi-planar NV* vs TurboJPEG planar YUV);
both stop after decode into a YUV buffer without an RGB colour step.

This is a comparison of two native decoding libraries, so **both arms are native
binaries**: `hal_cpu` (Rust) and `modules/turbojpeg/bench.c` (C, `dlopen`ing
`libturbojpeg`). `bench.c` mirrors `benchmarks/common` on every choice that moves
the number — image selection, preload before timing, `CLOCK_MONOTONIC` around
decode alone, percentile index, MP/s, CSV schema — so the two harnesses differ
only in the library they call.

Three rules for any new decode arm, each of which has silently biased a result
here before:

- **No interpreter in the timed region.** It lands on one side of the A/B only,
  and costs 1.5% of decode time on an out-of-order core rising to 5.5% on an
  in-order one.
- **Adapt the input before the clock starts.** HAL takes a preloaded `&[u8]`
  directly; anything needing a different buffer type must build it up front.
  Timing that conversion charges one arm an allocation and a ~160 KB copy per
  sample, worth 3–5% on in-order cores.
- **Match the IDCT accuracy class.** HAL implements the accurate `islow` IDCT,
  which is also libjpeg-turbo's default, so `bench.c` defaults to
  `--dct accurate` (`TJFLAG_ACCURATEDCT`). `TJFLAG_FASTDCT` selects its
  lower-accuracy `ifast` kernel and is worth 5–8% of its time — a real
  difference, but between two different accuracy classes. `--dct fast` reports
  it explicitly.

Build from the **repository root**:

```bash
docker build -f benchmarks/docker/Dockerfile -t edgefirst-hal-jpeg-bench .
```

Run with a mounted JPEG tree and results directory:

```bash
docker run --rm \
  -e EDGEFIRST_BENCH_COCO=/data/coco \
  -e BOARD=my-cpu-label \
  -e LIMIT=50 \
  -e EDGEFIRST_CODEC_FORCE_INTEL=avx2 \
  -v /path/to/coco/val2017:/data/coco:ro \
  -v "$(pwd)/benchmarks/results:/results" \
  edgefirst-hal-jpeg-bench
```

| Variable | Effect |
|----------|--------|
| `EDGEFIRST_BENCH_COCO` | JPEG directory inside the container (default `/data/coco`) |
| `BOARD` | Label written into the CSV filename / rows |
| `RESULTS_DIR` | Output directory (default `/results`) |
| `LIMIT` / `WARMUP` | Smoke knobs (`LIMIT=0` = full set) |
| `EDGEFIRST_CODEC_FORCE_INTEL` | `scalar\|sse2\|sse41\|avx2` A/B |
| `EDGEFIRST_CODEC_FORCE_NEON` | `scalar\|baseline\|plus\|high` A/B (aarch64 images) |
| `MODULES` | Comma list: `hal_cpu`, `turbojpeg` (default both) |
| `FORMATS` | Comma list: `yuv`, `rgb` (default both) |
| `TENSOR_MEM` | `mem\|dma\|auto` (default `mem`) |
| `EXTRA_ARGS` | Extra argv appended to `hal_cpu` |

Prefers a mounted `EDGEFIRST_BENCH_COCO` tree; otherwise uses an optional
build-staged `/opt/coco-smoke` (50 COCO images) then `/opt/testdata` fixtures.

The image is the portable OSS contract. How you schedule it on cloud CPUs
(container hosts, batch systems, etc.) is outside this repository.

## On-target

```bash
# One-time dataset sync (~820 MB)
./benchmarks/scripts/sync-coco.sh imx8mp-frdm rpi5-hailo orin-nano

# Decode-only HAL vs TurboJPEG (YUV + RGB) — smoke / investigation
./benchmarks/scripts/decode-ab-matrix.sh imx8mp-frdm imx95-pro rpi5-hailo orin-nano

# Published decoder A/B (release, interleaved best-of-3, n=200)
CARGO_PROFILE=release EDGEFIRST_BENCH_ORIN_FALLBACK=adis-uav1 \
  ./benchmarks/scripts/decode-ab-publish.sh imx8mp-frdm imx95-pro rpi5-hailo orin-nano

# Full HAL COCO matrix (decode + letterbox convert, all HAL backends)
./benchmarks/scripts/deploy-and-run.sh imx8mp-frdm imx95-pro rpi5-hailo

# zidane / giraffe diagnostic (NV12 + 444 fixtures)
./benchmarks/scripts/fixture-decode-matrix.sh imx95-pro

# perf + COCO + zidane
./benchmarks/scripts/perf-compare-decode.sh imx95-pro
```

SSH hostnames (keys preconfigured): `imx8mp-frdm`, `imx95-frdm`,
`rpi5-hailo`, `orin-nano`. Optional: set `EDGEFIRST_BENCH_ORIN_FALLBACK` to an
alternate SSH alias for the same Orin Nano hardware when `orin-nano` is down.

## Environment

| Variable | Effect |
|----------|--------|
| `EDGEFIRST_BENCH_COCO` | Path to JPEG directory |
| `EDGEFIRST_CODEC_FORCE_INTEL` | Force x86 JPEG ISA tier (`scalar\|sse2\|sse41\|avx2`) |
| `EDGEFIRST_CODEC_FORCE_NEON` | Force aarch64 NEON tier (`scalar\|baseline\|plus\|high`) |
| `EDGEFIRST_FORCE_BACKEND` | `cpu` / `opengl` / `g2d` (HAL convert) |
| `EDGEFIRST_DISABLE_V4L2` | `1` forces software JPEG decode |
| `EDGEFIRST_ENABLE_NVJPEG` | `1` opts into CUDA nvJPEG in the codec |
| `EDGEFIRST_TRACE` | Chrome/Perfetto JSON path (HAL bench binaries) |
| `CARGO_PROFILE` | Override deploy build profile (default `profiling`) |
| `EDGEFIRST_BENCH_ORIN_FALLBACK` | Optional SSH alias used when `orin-nano` is unreachable |

## Buffer reuse (critical)

HAL modules follow the Studio profiler decode-pool pattern
(`~/Software/Studio/profiler/.../buffers.rs`):

1. **One** Grey/R8 source tensor sized `width×(3·max_h)` (NV24 capacity).
2. **One** 640×640 RGB destination tensor.
3. Hot loop: `load_image` + `convert` only — no `create_image` per frame.
4. JPEG bytes are **preloaded** so disk I/O is outside the timed section.
5. `crop.source` is set when the decoded image is smaller than the pool.
6. Runs report `identity_churn` — must stay **0** (BufferIdentity stable).

Do **not** allocate as `PixelFormat::Nv24`: that FourCC breaks the GL `GL_RED`
zero-copy bind path. Grey capacity + native `configure_image` is required.

## Metrics (CSV)

Each run writes latency percentiles plus CPU load sampled over the timed loop
(Linux `/proc`):

| Column | Meaning |
|--------|---------|
| `cpu_pct_process` | Process CPU% (can exceed 100% when multi-threaded) |
| `cpu_pct_system` | Whole-machine busy% during the run |
| `cpu_pct_peak_core` | Busiest single core% |

HAL modules also report `decode_p50` / `convert_p50` in `notes`, plus
`identity_churn` and `cpu_fallback_frames`. Use `--tensor-mem mem|dma|auto`
to force Mem vs DMA for decode/convert A/B (`hal_cpu` defaults to Mem).

For the **decoder A/B claim table** (HAL vs TurboJPEG, YUV + RGB, no convert),
see [BENCHMARKS.md § JPEG Decode](../BENCHMARKS.md#jpeg-decode-edgefirst-vs-libjpeg-turbo).
Raw CSVs live under `results/<board>/decode-ab/` (gitignored).

## Notes from smoke

- **COCO is the primary Article-1 workload.** Full `val2017` is **99.8% 4:4:4 →
  Nv24** — the custom decoder’s Nv24 path is a first-class optimization target.
- **`zidane.jpg` (1280×720, 4:2:0 → Nv12)** is a diagnostic / high-res fixture
  (continuity with May `BENCHMARKS.md`), not a replacement for COCO.
  Results under `results/<board>/fixtures/`.
- Host PBO GL still CPU-falls-back on Nv24; DMA-BUF boards run GL with
  `cpu_fallback_frames=0` after the Grey pool fix.
- `hal_g2d` needs `/dev/galcore` (i.MX 8MP). i.MX 95 has no galcore → CPU fallback.
- SSH aliases: `imx8mp-frdm`, `imx95-frdm`, `imx95-pro`, `rpi5-hailo`,
  `orin-nano` (optional `EDGEFIRST_BENCH_ORIN_FALLBACK` for the same HW).

### Threading

Both decode arms are **single-threaded**: HAL is explicitly so per worker, so
concurrent pipelines do not oversubscribe, and `turbojpeg_bench` decodes on the
calling thread. The comparison is therefore per-core throughput, not wall-clock
with an unbounded thread pool.

### Where the decode time goes

Stage buckets are self-time from `perf report -s sym,srcline` over COCO
val2017, decode-only (`benchmarks/scripts/perf-decode-only.sh`).

| Stage | x86 YUV | x86 RGB | A55 YUV | A55 RGB |
|-------|---------|---------|---------|---------|
| Entropy (Huffman + bitstream) | **57.3%** | ~52% | **~58%** | **~57%** |
| IDCT (incl. dequant) | 22.3% | ~20% | 30.0% | 27.5% |
| Colour YCbCr→RGB | — | 5.0% | — | 5.8% |
| UV interleave / plane write | 0.9% | — | 1.6% | — |
| mcu loop / kernel / other | ~19% | ~23% | ~10% | ~10% |

Entropy dominates on every target. `perf stat -M TopdownL1` on Rocket Lake:
**66% bad speculation**, and the mispredicts are inherent to JPEG rather than
structural — against libjpeg-turbo on the same workload we take 63.7M branch
misses to its 61.0M while retiring 18% *fewer* instructions.

**The remaining gap on in-order cores is the IDCT and the MCU write.** Profiling
libjpeg-turbo itself and normalising both sides within codec time:

| Board | Stage | TurboJPEG | HAL | |
|-------|-------|-----------|-----|-|
| imx95-pro (A55) | entropy | 4.66 ms | **3.62 ms** | HAL ahead 1.04 ms |
| | IDCT + dequant | **~2.26 ms** | 2.45 ms | HAL behind ~0.19 ms |
| | mcu loop / write | **0.70 ms** | 1.17 ms | HAL behind 0.47 ms |
| imx8mp-frdm (A53) | entropy | 5.07 ms | **4.31 ms** | HAL ahead 0.76 ms |
| | IDCT + dequant | **~2.42 ms** | 2.75 ms | HAL behind ~0.33 ms |
| | mcu loop / write | **0.72 ms** | 0.93 ms | HAL behind 0.22 ms |

Turbo's IDCT rows are adjusted: those profiles were taken with `TJFLAG_FASTDCT`,
so they measured its `ifast` kernel, and the accurate kernel we compare against
costs it a further 0.489 ms (A55) / 0.614 ms (A53) end to end. Since the flag
selects nothing but the IDCT, that premium is added to the IDCT row. Re-profile
with `--dct accurate` to measure it directly rather than by difference.

**Still open:** a residual 1.0% A53 YUV gap vs turbo `islow` (published
interleaved best-of-3, n=200, release profile). A two-block IDCT is deferred
(A53 spill risk). A55 is ahead of `islow`. Two-block AVX2 IDCT on x86 is still
untried. The JPEG Decode A/B table in `BENCHMARKS.md` is hand-maintained from
`decode-ab-publish.sh`; the generated tables further down that file must not be
typed in by hand.

### Settled — do not re-test

Each of these was measured and rejected; the reasons generalise.

- **Folding AC EOB/ZRL into the combined `fast_ac` table**: −6%. Terminating the
  block from inside the hit path costs a test on every coefficient to save one
  miss per block.
- **Passing the bit buffer as a by-value `BitState` struct**: +17% instructions.
  LLVM would not scalarise the aggregate and copied 24 bytes between stack slots
  in the loop header. The fix that did work was inlining the *cold* refill and
  slow-symbol paths, so the buffer stays in registers instead of a stack slot.
- **Raising the refill threshold**: refill already early-outs on `avail >= 32`
  and a bulk refill leaves ≥57 bits, so it touches memory about every fourth
  coefficient. A higher threshold makes refills more frequent, not fewer.
- **A 256-bit single-block AVX2 IDCT**: at `i16` a 128-bit register already
  holds a row of eight, so width only buys lane-crossing shuffles — 21.8 ns/block
  against SSE2's 21.2. Real width needs *two blocks* side by side, which is a
  two-block entry point in the MCU loop, not a wider kernel.
- **Porting NEON's sparse-rows shortcut to x86**: −1.1% YUV / +0.5% RGB, i.e.
  nothing. Blocks sparse enough to trigger it are mostly already taken by the
  whole-block DC-only path.

### Methodology cautions

- **Interleave the arms.** This host is shared with a `powersave` governor and
  turbo enabled; single runs of the same binary drift by up to 40%, and one
  briefly showed a faster kernel as a regression. Every published number is
  best-of-N with the arms alternating inside one loop.
- **`libturbojpeg` ships stripped.** Group its samples by mapping each IP to the
  nearest preceding `endbr64`/call target on x86, or every `bl` target in the
  disassembly on aarch64. The hot functions then identify as `decode_mcu` and
  `jsimd_idct_islow_*`, and the resulting buckets agree across unrelated board
  images.
- **Normalise both sides the same way.** HAL's stage buckets include kernel and
  I/O time and Turbo's do not, so comparing raw shares overstates our
  "everything else" by roughly 2×.
- **Identical codegen is a result, not a null result.** Switching the NEON
  butterfly to `_laneq_` intrinsics changed nothing on its own: with the
  constant array visible, LLVM folded each lane back to its literal and re-emitted
  the same `dup`s, byte for byte. `black_box` on the constant pointers forces one
  real load (600 → 497 instructions, 64 `dup`s → 1) and is worth 5% of the kernel
  on in-order cores. If a change produces identical disassembly, check the
  disassembly before concluding the idea does not pay.
- **Expect instruction-count wins to be core-dependent.** The same constant
  hoisting was worth −5.5% / −3.9% on the A55 / A53 kernel and ±0% on the A76:
  in-order cores pay for every instruction, out-of-order ones hide them.

Reproduce:

```bash
./benchmarks/scripts/perf-decode-only.sh              # local host
./benchmarks/scripts/perf-decode-only.sh imx95-pro    # SSH target
```

### Instrumentation

MCU-row spans: `codec.decode_jpeg.mcu_row.huffman_idct`,
`write_nv12` / `write_nv16` / `write_nv24` / `write_grey` / `write_rgb` (see
`crates/codec/ARCHITECTURE.md`).
