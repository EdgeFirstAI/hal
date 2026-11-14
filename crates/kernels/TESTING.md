<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-kernels Testing

## Test Layout

```
crates/kernels/
├── src/
│   ├── cpu.rs, pool.rs, view.rs, quant.rs, tflite.rs   # Unit tests next to the code: tier rules,
│   │                                                   #   pools that repeat/skip/invent indices,
│   │                                                   #   view bounds, quantize, TFLite helpers
│   ├── ops/*_neon.rs                                   # NEON vs portable, bit for bit
│   ├── ops/levels.rs, ops/rows.rs                      # Block sizing, row walking, overlap check
│   └── prim/                                           # Primitives moved from the decoder, with
│                                                       #   their scalar vs NEON / FP16 parity tests
├── tests/
│   ├── common/mod.rs        # Fixture loaders, tier matrix, misbehaving pools, pool/concurrency check
│   ├── softmax.rs           # softmax_i8 / softmax_i16 against the TFLite fixtures
│   ├── sigmoid_lut.rs       # LOGISTIC fixtures, SigmoidPlan paths, LutI8Plan
│   ├── decode_ops.rs        # DFL / dist2bbox / DFL box against a float64 reference and the decoder
│   ├── binary.rs            # int8 ADD / SUB / MUL fixtures, NEON-block vs scalar discrimination
│   ├── matmul.rs            # BATCH_MATMUL and 1×1 CONV_2D fixtures, ruy vs reference rounding
│   ├── softmax_dot.rs       # the DFL head fixture against its float definition and TFLite
│   ├── attention.rs         # the attention chain fixtures, limits, scratch, the crate's own chain
│   └── neon_coverage.rs     # every op path's tier on aarch64
└── benches/
    └── kernels_benchmark.rs # Per-tier softmax, sigmoid, DFL, dist2bbox and DFL box timings
```

Outside the crate: `crates/kernels-capi/src/tests.rs` and `crates/kernels-capi/tests/c/` (the C boundary), `tests/kernels/test_kernels.py` (the Python bindings), `tests/packaging/` (the wheel's stubs, namespace and `DT_NEEDED` isolation).

## Running Tests

All `cargo test` invocations pass `-- --test-threads=1` per the workspace single-threaded rule ([root TESTING.md § Single-threaded execution](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#single-threaded-execution)). The kernels are CPU-only, but the rule applies workspace-wide.

```bash
# All kernels tests (unit, integration, doc)
cargo test -p edgefirst-kernels -- --test-threads=1

# One integration suite
cargo test -p edgefirst-kernels --test softmax -- --test-threads=1

# A forced tier for every plan built with ::new (the tests also build each tier explicitly)
EDGEFIRST_KERNELS_FORCE_TIER=scalar cargo test -p edgefirst-kernels -- --test-threads=1

# The C boundary (workspace-excluded leaf)
(cd crates/kernels-capi && EF_REQUIRE_FRESH_ARTIFACTS=1 cargo test --target-dir ../../target -- --test-threads=1)

# The Python bindings, from the repo venv
make install-python PYTHON_PACKAGES=kernels
venv/bin/python -m pytest tests/kernels tests/packaging -q
```

The NEON paths only run on aarch64. Cross-compile and run them on a board with `scripts/on-target-test.sh` (`edgefirst-kernels` and `kernels-capi` are in its default `CRATES` / `CAPI_CRATES`); see [root TESTING.md § On-target tests](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md#on-target-tests). The i.MX 95 (Cortex-A55) runs every tier; the i.MX 8M Plus (Cortex-A53) runs `scalar` and `neon`.

## Fixtures

`testdata/kernels/*.safetensors` (Git LFS) are TFLite reference cases: seeded inputs and constant operands, the LiteRT interpreter's output, and the op, quantization and resolver in the file's metadata. The single-op `SOFTMAX` and `LOGISTIC` fixtures hold `input` and `output` and load through `common::load`; the int8 arithmetic, matrix and chain fixtures hold named tensors (`lhs`/`rhs`, `input`/`filter`/`bias`, `q`/`k`/`v`, then `output`) with `<role>_scale` / `<role>_zero_point` metadata and load through `common::load_graph`. Both read through `edgefirst_bench::testdata`, so `EDGEFIRST_TESTDATA_DIR` relocates them; the suite never needs TFLite at test time.

The int8 arithmetic and matrix fixtures run TFLite's own builtin kernels: the optimized resolver without the default delegates (the default resolver hands int8 `MUL`, `BATCH_MATMUL` and `CONV_2D` to XNNPACK, whose bytes differ), and the reference resolver for the `_ref` fixtures (metadata `kernels`: `optimized` or `reference`). They come from LiteRT's x86_64 wheel, which defines `USE_NEON` (SSE4.1 through `NEON_2_SSE`): its optimized `ADD` and `MUL` run the NEON-block arithmetic of an aarch64 build, so they equal `TfliteKernels::NeonRuy` and the `_ref` fixtures equal `TfliteKernels::Reference`. Its optimized `BATCH_MATMUL` and `CONV_2D`, though, requantize like the reference kernels (two roundings), not with ruy's single rounding, so every matrix fixture equals `TfliteKernels::Reference`. The `add_i8_pairs`, `mul_i8_odd_wide`, `batch_matmul_*` and `conv1x1_*` fixtures each separate the two variants, which the tests assert. The same models run through the TFLite 2.19.0 of the i.MX 8M Plus and i.MX 95 BSPs give `TfliteKernels::NeonRuy` bit for bit for every op, ruy's single rounding included, and the fused attention gives that TFLite's bytes on every tier; those outputs are not fixtures.

| Fixture | Op | Shape | Covers |
|---------|----|-------|--------|
| `softmax_i8_yolo11_attn`, `softmax_i8_yolo26_attn` | int8 `SOFTMAX` | `[1,2,400,400]`, `[1,2,1008,1008]` | the YOLO11 / YOLO26 attention maps |
| `softmax_i8_dfl` | int8 `SOFTMAX` | `[1,4,8400,16]` | DFL bins |
| `softmax_i8_odd_rows_{1,17,1009}` | int8 `SOFTMAX` | rows of 1, 17, 1009 | rows that are not a multiple of the vector width |
| `softmax_i16_attn`, `softmax_i16_odd_rows_{1,1009}` | int16 `SOFTMAX` | `[1,2,400,400]`, odd rows | the all-integer kernel |
| `softmax_i16_scale_small`, `softmax_i16_scale_large` | int16 `SOFTMAX` | `[6,257]` | input scales 1e-5 (rescale shift ≤ 0) and 4.5 (the largest accepted left shift) |
| `logistic_i8_scores`, `logistic_u8_scores` | int8 / uint8 `LOGISTIC` | `[1,80,8400]` | the score-head table |
| `logistic_i16_pot`, `logistic_i16_nonpot` | int16 `LOGISTIC` | `[1,80,8400]` | gemmlowp (scale 2⁻¹²) and reference-table paths |
| `add_i8`, `add_i8_odd` | int8 `ADD` | `[4,1024]`, `[3,1001]` | whole 16-element blocks; a tail of 11 and `RELU` |
| `add_i8_pairs`, `add_i8_pairs_ref` | int8 `ADD` | `[256,256]` | every input pair, at a quantization where the NEON-block and scalar `ADD` differ on one |
| `sub_i8_odd` | int8 `SUB` | `[3,1001]` | the scalar-only kernel |
| `mul_i8_odd`, `mul_i8_odd_relu6` | int8 `MUL` | `[3,1001]` | a small multiplier, `RELU6` |
| `mul_i8_odd_wide`, `mul_i8_odd_wide_ref` | int8 `MUL` | `[3,1001]` | a multiplier above one: the NEON blocks' int16 saturation and wrapping zero point |
| `mul_i8_scalar`, `mul_i8_scalar_odd` | int8 `MUL` | `[1,2,400,32]`, `[1,3,1001]` by `[1]` | a broadcast scalar (the attention Q scale), and its odd-length tail |
| `batch_matmul_i8`, `batch_matmul_i8_ref` | int8 `BATCH_MATMUL` | `[2,64,48]·[2,48,37]` | both zero points non-zero, `N` not a multiple of 4 |
| `batch_matmul_i8_adj` | int8 `BATCH_MATMUL` | `adj_x`, `adj_y` | transposed operands as strided views |
| `conv1x1_i8`, `conv1x1_i8_ref` | int8 `CONV_2D` 1×1 | `[1,10,13,24]` to 40 channels | per-channel filter scales, int32 bias, `RELU6` |
| `attention_i8_yolo26`, `attention_i8_odd` | `MUL` → `BATCH_MATMUL` (`adj_y`) → `SOFTMAX` → `BATCH_MATMUL` | h=2 n=400 dk=32 dv=64; h=3 n=37 dk=12 dv=9 | the YOLO26 attention block and a shape with no whole tiles |
| `dfl_head_i8` | `SOFTMAX` → 1×1 `CONV_2D` | `[1,4,2100,16]` to one channel | the DFL head `SoftmaxDotI8Plan` fuses: weights `round(i·127/15)` |

Regenerate them with the pinned LiteRT from the repo venv:

```bash
. venv/bin/activate
pip install -r scripts/requirements-fixtures.txt
python -I scripts/gen_kernel_fixtures.py testdata/kernels
```

The inputs and outputs are seeded and reproduce exactly, but the safetensors header orders its metadata differently on each run, so commit only the files whose content changed.

## Acceptance bars

| Op | Bar |
|----|-----|
| `softmax_i8` | scalar tier bit for bit against the fixture (glibc `expf`); NEON tiers within ±1 LSB |
| `softmax_i16`, int16 `LOGISTIC` | bit for bit on every tier |
| `binary_i8` (`ADD`, `SUB`, `MUL`) | bit for bit on every tier against the fixture's kernels (`NeonGemmlowp` and `NeonRuy` for the optimized resolver of the x86 wheel, `Reference` and `Ruy` for the reference one), contiguous and with the scalar broadcast as zero strides; the broadcast `MUL` through `table_for_rhs` and `LutI8Plan` too |
| `matmul_i8` (`BATCH_MATMUL`, 1×1 `CONV_2D`) | bit for bit on every tier with `TfliteKernels::Reference`; every tier and layout equal (exact accumulation) and within one LSB of the f64 requantization |
| `softmax_dot_i8` | every tier within one LSB of the float definition (exact exponentials, the expectation requantized in f64) on every row; within `⌈|TFLite − float|⌉ + 1` of TFLite; a smaller total error than TFLite's `SOFTMAX` → `CONV_2D` (measured: mean 0.25 LSB against 0.55); the crate's `SoftmaxI8Plan` and `MatMulI8Plan` reproduce the fixture bit for bit; NEON within ±1 LSB of the scalar tier at the edge of its error bound (weights −128 and 127), and a plan past the bound runs the scalar tier |
| `attention_i8` | every tier equal to the crate's own chain (`table` → `MatMulI8Plan` → `SoftmaxI8Plan` → `MatMulI8Plan`) on that tier, bit for bit, on the fixtures and at the limits (2048 tokens with `dk = dv = 256`, 2047 with 255, `dk = dv = 1`, one token, `dk = 0`) and with 16 tokens, whose softmax rows the chain runs in groups of sixteen; the scalar tier bit for bit against TFLite; NEON tiers within one LSB of TFLite (the NEON softmax's ±1 LSB; none differ on the Cortex-A53 and Cortex-A55 boards) |
| int8 / uint8 `LOGISTIC` (`LutI8Plan::sigmoid`, `SigmoidPlan`) | bit for bit (glibc `expf`) |
| DFL / dist2bbox / DFL box, integer outputs | within ±1 LSB of a float64 reference, for each integer input type, both input layouts and the real 3-level anchor lists |
| DFL / dist2bbox / DFL box, `f32` outputs | integer inputs: the decoder's level kernels bit for bit on a one-level call; float inputs on NEON within the f32 bar of the scalar tier; the 8-bit table kernels bit for bit with the portable kernel for every `reg_max` |
| DFL / dist2bbox / DFL box, `f16` outputs | scalar and `neon` tiers, and on `neon_fp16` `DflPlan`'s 8-bit inputs and every `Dist2BboxPlan`: the tier's `f32` output rounded once, bit for bit; other `neon_fp16` cases: within 1 f16 ulp of the float64 result (2 for distances and 3 for boxes at `reg_max` 64) at fixed and seeded random quantizations, in ulps of the value or of the full scale (640 px, `reg_max − 1` grid units) |
| Sigmoid to `f32` | integer inputs: the decoder's dequantize and sigmoid kernels bit for bit; every input: the f32 exp on every tier, `neon_fp16` equal to `neon` bit for bit |
| Sigmoid to `f16` | scalar and `neon` tiers: the tier's `f32` output rounded to f16, bit for bit; `neon_fp16`: within 2 f16 ulps of the float64 sigmoid, every input type at fixed quantizations, and the four integer types over a seeded sweep of 200 to 400 random quantizations each |
| int16 `LOGISTIC` NEON kernels | NEON equals scalar on all 65 536 inputs (POT, and the reference kernel at eight input scales up to the largest accepted) |
| `prim::neon_mask` integer dot products | the exact `Σ (m − zm)(p − zp)` (i32 for 8-bit pairs, the i64 value rounded to f32 once otherwise) for every pair of `i8`/`u8`/`i16`/`u16`, `neon` and `dotprod` kernels, NHWC and NCHW rows (including rows ending at the buffer end), 18 channel counts up to 257, the type's extreme zero points, and the largest magnitudes at `MAX_K` |
| `prim::neon_mask` float dot products | within `(K/4 + 5) · 2⁻²⁴ · Σ |m·p|` (NHWC) or `K · 2⁻²⁴ · Σ |m·p|` (NCHW) of the float64 sum, `f32`/`f16`/`i8`/`u8` protos |
| `prim::neon_mask` sigmoid to `u8`, threshold, NCHW dequantize | sigmoid within 1 of the scalar `round(255 / (1 + exp(−x)))` over 2.8 million integer and float logits (NaN, ±inf exact); threshold and dequantize bit for bit |
| `prim::neon_candidates` column argmax, row argmax, threshold positions | the scalar fold bit for bit (last class or position wins a tie, NaN takes over, a NaN best is kept) for `i8`/`u8`/`i16`/`u16`/`f32`/`f16` and the native FP16 `f16` kernels, 18 lengths up to 300 and 6 class counts up to 91, few-valued and full-range scores with NaN, signed zeros and infinities; every f16 compare against `half` over a 1 100-value sweep |
| `prim::neon_convert` / `neon_fp16` `scale_bias` | `x · scale + bias` bit for bit with the decoder's generic dequantize, every `i8`/`u8`/`i16`/`u16` value, into `f32` and (emulated and native FP16) `f16` |
| `prim::neon_nms` greedy NMS, `jaccard4_neon` | the decoder's scalar loop flag for flag over 23 box counts up to 333 (every tail of four), clustered, identical, zero-area, inverted and NaN/infinite boxes, class-agnostic and class-aware, eight thresholds (NaN and negative included), four caps and pre-flagged boxes; `jaccard4_neon` equals `jaccard` both ways round on 20,000 random and special box sets; the FP16 kernel (`nms_greedy_neon_fp16`) equals a scalar loop over the same f16-rounded operations flag for flag over 20 box counts up to 333 (every tail of eight), three coordinate scales, infinite and quiet-NaN boxes, six thresholds, four caps and pre-flagged boxes, and equals the f32 kernel on boxes whose overlaps are far from the threshold |
| `prim::sigmoid::fast_sigmoid`, `prim::neon_modelpack` | `fast_sigmoid` equals the decoder's original code on every 61st f32; `fast_sigmoid4` equals `fast_sigmoid` on every 127th f32, every value within 2¹⁶ ulps of ±80, ±88 and ±0, and the specials (an `#[ignore]` test, run by hand on the board, covers all 2³² inputs; CI runs the sampled tests); the split rows equal the scalar decode bit for bit for every input type, four dequantizations, eight grids up to 40×40×3 anchors with 1–91 classes and row ranges |
| `prim::neon_seg` | top byte bit for bit over every 8- and 16-bit value and random 32-bit values; f32 and f16 (emulated and native FP16) normalization equal the scalar code (`half` arithmetic for f16) on random, NaN, infinite, signed-zero and constant maps and maps of every finite f16; the per-pixel argmax equals the first maximum for 1–20, 31–33, 64, 255–257 and 300 channels |
| `prim::neon_semantic`, `prim::softmax::exp_nonpositive` | argmax (planar and interleaved, `f16` widened and native FP16) the first maximum for 12 class counts from 1 to 300 and 65536, 13 pixel counts up to 529 (strip and block tails), tie-heavy and NaN/infinite maps; the softmax confidence equals the scalar sum bit for bit for every score type and both layouts, and `exp_nonpositive4` equals `exp_nonpositive` on a million f32 patterns, NaN and infinities; `exp_nonpositive` equals the decoder's original code; threshold, binary probability confidence, `u8` confidence, row blends, `TBL` gathers (up- and downsampling, irregular positions, rows shorter than a window), column gathers, cell candidates (NaN and signalling NaN corners) and argmax (NaN corners), and class indices equal their scalar forms bit for bit; the sigmoid is within 2⁻²¹ of `1 / (1 + exp(−x))` relative to it over four million f32 patterns (worst measured 2.4·10⁻⁷) |

## Patterns every plan is tested for

- **Tier matrix.** `common::tiers()` is `scalar` plus every NEON tier the CPU has; a parity test runs each plan built `with_features` for each one and checks `plan.tier()`.
- **Bad parameters** (`new_rejects_bad_params`): zero, negative, NaN and infinite scales, `reg_max == 0`, bad levels; `InvalidArgument` or `Unsupported`, never a panic.
- **Mismatched run arguments** (`run_rejects_mismatch_without_writing`): wrong dtype, shape mismatch, an output view too small; `InvalidArgument` and the output left untouched.
- **Views** (`strided_views_match_contiguous`, `zero_sized_and_rank_zero_views`, `zero_anchor_views_are_a_no_op`): a transposed last axis, output strides different from the input's, zero-sized dimensions, overlapping output rows (`run_rejects_aliasing_output_rows`).
- **Pools and concurrency** (`pool_and_concurrent_runs_match_single_thread`, through `common::assert_pool_and_concurrency_match`): no pool, `ScopedThreads` with more threads than rows, a sequential pool, pools that repeat, skip, invent or never run indices, and one plan run from two threads at once; the output must be byte-identical to the no-pool run.
- **Send + Sync** (`plans_are_send_and_sync`).
- **Several inputs.** `binary_i8`, `matmul_i8` and `attention_i8` take two or three inputs: broadcast operands as zero strides, transposed operands (`adj_x`, `adj_y`, transposed K) as swapped strides, and Q, K and V as head slices of one token-major tensor.
- **Attention scratch** (`scratch_is_reused_and_mismatches_are_rejected`): one scratch serves many calls; a scratch built for other heads, queries, tokens or dims, or by a plan whose kernel packs another layout (a `neon` plan on a dotprod CPU), is `InvalidArgument` with the output untouched.

Skips follow the repo policy: a test that cannot run (an FP16 test on a CPU without FP16, a tier the host lacks in the benchmark) prints `WARNING: SKIPPED: <reason>` to stderr through `test_support::report_skip`, never silently.

## Benchmarks

`kernels_benchmark` (custom `harness = false`, on [`crates/bench`](https://github.com/EdgeFirstAI/hal/tree/main/crates/bench)) times `softmax_i8` and `softmax_i16` on `[1,2,400,400]` and `[1,2,1008,1008]` per tier against a stock-speed per-element reference, the sigmoid paths (int8 table, int16 POT and reference, int8 to `f32`, `f16` and `u8`), `DflBoxPlan` from int8 logits to `f32`, `f16` and `i8` boxes on the yolov8n (8400 anchors) and 1344×768 (21 168 anchors) tails, in both input layouts, from int16 and f32 logits to `f32` and `f16` boxes, `DflPlan` at `reg_max` 8, and `Dist2BboxPlan` from int8 and f32 distances to `f32`, `f16` and `i8` boxes. `--section <name>` runs one group: `softmax`, `sigmoid`, `transpose`, `dfl` (DFL and DFL box) or `dist2bbox`.

```bash
cargo bench -p edgefirst-kernels --bench kernels_benchmark -- --bench
```

On a board, `scripts/on-target-bench.sh` carries it as the `kernels` case. Run benchmarks sequentially, one board at a time, and keep results under `benchmarks/<platform>/`. These numbers are reported, not gated; the merge gate for kernel changes is `decoder_benchmark` under the `release` profile.

## Coverage Notes

- The crate is a workspace member, so `make test-rust` (`cargo llvm-cov nextest --workspace`) covers it; the C leaf's lines are outside that report (see the Makefile's `test-capi-modular` note), and the Python bindings' lines are measured through `python-common`.
- `edgefirst-kernels` is in the nightly mutation corpus with the decoder and tracker, on the Linux legs ([root TESTING.md § Mutation testing](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)).

## Cross-References

- Architecture and numerics: [ARCHITECTURE.md](ARCHITECTURE.md)
- Project testing patterns: [../../TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/TESTING.md)
- Decoder tests that exercise the shared primitives: [../decoder/TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/TESTING.md)
