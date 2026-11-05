<!--
SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
SPDX-License-Identifier: Apache-2.0
-->

# edgefirst-kernels Architecture

## Overview

`edgefirst-kernels` is the HAL's framework-neutral CPU kernel library. It takes plain slices plus a shape, element strides and per-tensor quantization, and runs NEON-first kernels on them through a plan/run API. It knows nothing about models, schemas or any framework's graph; where a plan reproduces an inference runtime's integer arithmetic (TFLite's int8 softmax, logistic, elementwise, matrix and fused attention kernels), that arithmetic is ported into `tflite.rs` and the plan names the TFLite kernels it matches (`TfliteKernels`). It depends only on `half` and `tracing`, never on `edgefirst-tensor`, GL/EGL or PyO3. The same kernels serve the decoder, an inference-framework delegate, and anyone calling the Rust, C (`libedgefirst_kernels`) or Python (`edgefirst.kernels`) API directly.

## Layering

```text
edgefirst-kernels          (deps: half, tracing)
  ├── edgefirst-decoder    (calls the prim:: primitives; keeps model and schema logic)
  ├── kernels-capi         (cdylib libedgefirst_kernels + include/edgefirst/kernels.h)
  └── python-common        (feature `kernels`; built into the edgefirst.kernels wheel)
```

The crate sits at the bottom of the graph. Because it never touches `edgefirst-tensor`, it is outside the tensor-backend `cfg` machinery, and neither `libedgefirst_kernels` nor `_kernels` has a `DT_NEEDED` on `libedgefirst_tensor` (`scripts/check-single-home.sh` G2t and `tests/packaging/test_dependency_isolation.py` assert it).

## Module Map

| Module | Responsibility |
|--------|----------------|
| [`lib.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels/src/lib.rs) | Re-exports: plans, params, views, `ByteSpan`, `Quant`, `DType`, `Activation`, `TfliteKernels`, `AttentionScratch`, `Pool`, `CpuFeatures`, `KernelError` |
| [`error.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels/src/error.rs) | `KernelError` (`Unsupported`, `InvalidArgument`, `ForcedTierUnavailable`) |
| [`quant.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels/src/quant.rs) | `Quant`, `DType` (C ABI discriminants), `Activation` (a fused TFLite activation), `TfliteKernels` (which TFLite kernels a plan reproduces: optimized, with or without the NEON blocks of int8 `ADD`/`MUL`, or reference), sealed `QuantInt`, `quantize` (round half away from zero, clamp), and the crate-internal params checks that make a float tensor's `Quant` the identity |
| [`view.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels/src/view.rs) | `View` / `ViewMut` strided views, `TensorRef` / `TensorMut` type-erased tensors, `ByteSpan` (a raw view's byte range, for the bindings' overlap check), `MAX_RANK` |
| [`pool.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels/src/pool.rs) | `Pool` trait, `ScopedThreads`, crate-internal `for_each_block` |
| [`cpu.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels/src/cpu.rs) | `CpuFeatures` probe, `Tier`, `EDGEFIRST_KERNELS_FORCE_TIER` |
| [`tflite.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels/src/tflite.rs) | TFLite v2.19.0 integer helpers ported bit for bit: `quantize_multiplier`, `multiply_by_quantized_multiplier`, `lut_populate_i16`, `lut_lookup_i16`, the gemmlowp pieces int16 logistic uses |
| [`ops/`](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels/src/ops/) | The plans: `softmax_i8` (+ `_neon`), `softmax_i16` (+ `_neon`), `lut_i8` (+ `_neon`), `sigmoid` (+ `sigmoid_i16_neon`), `dfl`, `dist2bbox`, `dfl_box`, `binary_i8` (+ `_neon`), `matmul_i8` (+ `_neon`), `softmax_dot_i8` (+ `_neon`), `attention_i8` (+ `_neon` for dotprod CPUs, `_neon_base` for the others); `rows.rs` (row walking, staging, overlap check) and `levels.rs` (`Level`, `BoxFormat`, anchor grids, the decode block loop) |
| [`convert.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels/src/convert.rs) | The public slice conversions: `dequant_*_to_f32` / `_to_f16` (the `neon_convert` kernels on aarch64, the `prim::dequant` loops elsewhere, the same bits on every input) and `nchw_to_nhwc` over the sealed `TransposeElement` (u8, i8, u16, i16, f16, u32, i32, f32), re-exported from `prim` |
| [`prim/`](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels/src/prim/) | `#[doc(hidden)]`, not covered by semver. Slice primitives shared with the decoder: dequant, softmax, sigmoid, box primitives (weighted sum, dist2bbox), anchor grids, the DFL exp table (`dfl_lut.rs`), NCHW→NHWC transpose (NEON interleave and register tiles for 1-, 2- and 4-byte elements), `neon_baseline.rs` (NEON and inline-asm FP16 kernels), `neon_candidates.rs` (detection candidate selection: the argmax over class columns in register tiles or over a contiguous row, ties to the last class, with the score threshold, and the positions at or above a threshold, bit-identical to the scalar fold; `f16` compared widened or with FP16 instructions), `neon_convert.rs` (NEON quantize, f16 narrow/widen and dequantize, and the decoder's generic `x · scale + bias` dequantize, bit-identical to the scalar conversions), `neon_sigmoid.rs` (one-pass NEON dequantize+sigmoid into f32 or f16), `neon_ltrb.rs` (LTRB dequantize fused with dist2bbox, sixteen anchors per step, into f32 or f16), `neon_mask.rs` (instance-mask dot products of a detection's coefficients with a crop row of NHWC or NCHW protos: exact integer logits, `sdot` on dotprod CPUs, and f32 FMA logits; the sigmoid to `u8`, the mask threshold and NCHW→NHWC proto dequantize; used by the decoder and `edgefirst-image`), `neon_nms.rs` (greedy NMS over padded coordinate arrays, four later boxes per vector against each kept box, with the decoder's `jaccard` decisions bit for bit), `neon_modelpack.rs` (ModelPack split rows: dequantize, the decoder's fast sigmoid vectorized bit for bit, box transform and objectness-weighted scores), `neon_seg.rs` (ModelPack segmentation maps: integer top byte to `u8`, float min/max normalization to `u8` in f32 or f16 arithmetic, the per-pixel first-max argmax), `neon_semantic.rs` (the semantic decoder's per-pixel class argmax over planar or interleaved scores, softmax and binary confidence, threshold, `TBL` row gathers, bilinear row blends, boundary-cell candidates and argmax, and class-index narrowing, bit-identical to the decoder's scalar code except the sigmoid) and `neon_fp16.rs` (FP16-arithmetic dequantize+sigmoid, DFL softmax, dist2bbox and `x · scale + bias` into f16). `sigmoid.rs` also holds the decoder's ModelPack fast sigmoid (`fast_sigmoid`, the Schraudolph exponent), and `softmax.rs` the semantic decoder's branch-free `exp_nonpositive` |
| [`test_support.rs`](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels/src/test_support.rs) | `#[cfg(test)]` `report_skip` (`WARNING: SKIPPED:` on stderr) |

## Plan/run contract

Every operator is a plan type with a params struct: `SoftmaxI8Plan::new(SoftmaxI8Params { .. })`, then `plan.run(input, output, pool)`.

- **`new` validates everything it can.** Bad numbers (a scale that is zero, negative, NaN or infinite; `reg_max == 0`; a level with `grid_w == 0` or `anchors % grid_w != 0`) are `InvalidArgument`. A valid configuration the plan does not handle (the wrong output quantization for a TFLite-exact softmax, a rank above `MAX_RANK`) is `Unsupported`. A delegate calls `new` during node selection and claims the node only if it succeeds. Tables (the softmax exp table, the LUT, the int16 tables, the anchor grid) are built here, once.
- **`new` selects the tier.** `new` reads `CpuFeatures::from_env_or_probe()`; `with_features(params, features)` takes them explicitly (tests, benchmarks) and reads no environment. `LutI8Plan` has two constructors, so it has two pairs: `from_table` / `from_table_with_features` and `sigmoid` / `sigmoid_with_features`. `plan.tier()` reports the kernel the plan will actually run.
- **Float tensors ignore their `Quant`.** In every op, a `Quant` applies only to an integer input or output; an `f16` or `f32` tensor's is ignored (the identity), so a delegate can pass `{0, 0}`.
- **`run` validates before it writes.** Wrong dtype, a shape mismatch, an output view too small for its shape, overlapping output strides, or (for the decode ops) level anchors that do not add up to the anchor axis return `InvalidArgument` with the output untouched.
- **`run` never allocates, takes no lock and never touches the global rayon pool.** Scratch is fixed-size stack staging (`ops::rows::STAGE` elements per row, `ops::levels::BLOCK` anchors per decode block) or lives in the plan, read-only after `new`. The one exception is attention's packed operands, which the caller owns: `AttentionI8Plan::scratch` builds an `AttentionScratch` once for a shape, and `run` takes it `&mut`, so each concurrent `run` needs its own. A scratch serves any plan whose kernel shares its layout and is rejected by the others.
- **Shape limits are checked, not exported.** `MatMulI8Plan::check_depth` and `AttentionI8Plan::check_shape` say whether the kernels' fixed stack buffers take a shape, so a delegate can decide before it claims a node.
- **Plans are `Send + Sync`** and `run(&self)` may be called on one plan from several threads at once.
- **Library code never panics.** Bad inputs are `KernelError`s; assertions moved from the decoder are unreachable from the plan API because the plans validate first.
- **One tracing span per `run`**, `kernels.<op>` with field `tier`, never per row or block (see [Tracing Spans](#tracing-spans)).

## Views, strides and overlap

`View<'a, T>` and `ViewMut<'a, T>` wrap a borrowed slice with a shape of rank ≤ `MAX_RANK` (6) and element strides, all non-negative. `contiguous` derives row-major strides; `strided` takes them explicitly, so RESHAPE (`reshape`, contiguous views only), SLICE and TRANSPOSE are expressed without copying. Construction checks that every addressed element lies inside the slice. The raw-pointer constructors (`from_raw_parts`) are `unsafe`, with the same contract stated as a safety requirement; the C and Python layers use them. They reject a NULL or misaligned pointer and an extent above `isize::MAX` bytes, which `slice::from_raw_parts` forbids, with `InvalidArgument`.

- A dimension of 0 makes `run` a no-op that returns `Ok`.
- Softmax reduces over the last axis; every other axis indexes independent rows. A dense input and output take the fast path (whole rows as slices); a last-axis stride other than 1, or output strides that differ from the input's, take the strided path, which stages rows of up to `ops::rows::STAGE` elements through the stack and walks longer rows element by element.
- **Overlap rule:** the output axes, sorted by stride, must each step past the extent of the smaller ones (`ops::rows::check_no_overlap`). This guarantees no output element is written twice, which is what makes concurrent row blocks race-free; it also rejects some interleaved layouts that do not actually alias. Input and output must not alias each other: safe Rust cannot express it. A view's slice covers its whole span, from its first element to one past its last, so the C and Python layers compare the two views' `ByteSpan`s before building either view and reject overlapping spans, interleaved views of one buffer included (`EINVAL` / `ValueError`).

## Pool contract

`Pool` is a safe trait the caller implements: `fn run(&self, n_tasks: usize, task: &(dyn Fn(usize) + Sync))`, "run these independent tasks and return when they are done". `None` runs on the calling thread; `ScopedThreads` is a ready-made pool that spawns scoped threads per call. The kernels never touch a global pool.

`for_each_block` splits the work into at most 64 disjoint blocks and gives each one a claim flag, so a block runs exactly once whatever the pool does: a pool that repeats an index, calls indices out of range, skips some or runs nothing still yields the same output, because unclaimed blocks run on the calling thread after `Pool::run` returns. Because `Pool` is a safe trait taking a borrowed task, a sound implementation cannot keep the task past `run`, so every block it ran has finished and its writes are visible when `run` returns. Results never depend on the pool or the number of threads.

Tasks keep staging buffers of up to about 48 KiB on the stack (attention on CPUs without FEAT_DotProd; `ops::rows` stages an input and an output row of up to 4096 elements and the matmul panels four rows of up to 4096, 32 KiB each), so pool worker threads need about 96 KiB of stack headroom beyond the pool's own use.

## CPU tiers

| Tier | Requires | Name for `EDGEFIRST_KERNELS_FORCE_TIER` |
|------|----------|-----------------------------------------|
| `Scalar` | nothing; portable code written to auto-vectorize | `scalar` |
| `Neon` | aarch64 NEON (every aarch64 CPU) | `neon` (alias `neon_baseline`) |
| `NeonFp16` | ARMv8.2 FP16 arithmetic (Cortex-A55, i.MX 95) | `neon_fp16` |
| `NeonDotprod` | `sdot`/`udot` (Cortex-A55) | `neon_dotprod` |

`CpuFeatures::probe()` uses runtime detection. Stable Rust has no FP16 or `sdot` intrinsics, so those kernels use inline asm with `.arch_extension fp16` / `.arch_extension dotprod`, reached only when the probe saw the feature. The single FP16 instructions live in `prim::fp16_asm` as `unsafe fn`s; plans and the decoder gate them with `CpuFeatures::fp16_usable()` (and `sdot` with `dotprod_usable()`), which require the feature flag and the CPU's own runtime support, so hand-built features cannot reach them on a CPU without the feature. FP16 `2^k` exponent injection is valid only for k ∈ [−14, 15]. x86_64 gets the portable scalar kernels.

`EDGEFIRST_KERNELS_FORCE_TIER` (case-insensitive) forces the tier every plan built with `new` targets; a tier the CPU lacks, or an unknown name, makes `new` fail with `ForcedTierUnavailable` rather than fall back. The decoder does not read it: its plans honour `EDGEFIRST_DECODER_FORCE_KERNEL`. On aarch64 every op path, for every input and output dtype pair and quantization variant, runs a NEON kernel: `Tier::Scalar` is the fallback for CPUs without NEON (and for a forced `scalar` tier) and the reference the NEON kernels are tested against, never a path NEON hardware takes. `tests/neon_coverage.rs` enforces this, path by path against the tier column below. `tier()` reports the kernel that runs, per path:

| Op | Path | Kernel | Tier on NEON hardware |
|----|------|--------|-----------------------|
| `softmax_i8` | int8 rows | `ops/softmax_i8_neon.rs` | `neon` |
| `softmax_i16` | int16 rows | `ops/softmax_i16_neon.rs` | `neon` |
| `lut_i8` | `i8`/`u8` table | `ops/lut_i8_neon.rs` (`tbl`) | `neon` |
| `sigmoid` | `i8 → i8`, `u8 → u8` | `LutI8Plan::sigmoid` | `neon` |
| `sigmoid` | TFLite `i16 → i16` (gemmlowp and reference) | `ops/sigmoid_i16_neon.rs` | `neon` |
| `sigmoid` | any input → `f32` | fused NEON dequantize+sigmoid (`prim::neon_baseline` for integer inputs, `prim::neon_sigmoid` for `f16`/`f32` inputs) | `neon` |
| `sigmoid` | any input → `f16` | `prim::neon_fp16` sigmoid on FP16 CPUs, else `prim::neon_sigmoid`: the f32 path rounded once, in the same pass | `neon_fp16`, else `neon` |
| `sigmoid` | any other pair (requantize) | NEON dequantize, f32 sigmoid, NEON quantize | `neon` |
| `dfl` | 8-bit inputs, any `reg_max` | `prim::dfl_lut` NEON table kernels | `neon` (also for `f16` outputs) |
| `dfl` | other inputs | NEON dequantize, fused NEON softmax (fp16 exp for 16-bit inputs to `f32` on FP16 CPUs), NEON quantize | `neon`, `neon_fp16` for the fp16 exp |
| `dfl` | 16-bit and float inputs → `f16` on FP16 CPUs | `prim::neon_fp16` DFL softmax | `neon_fp16` |
| `dist2bbox` | every output (`f16` the f32 boxes rounded once, which beats native FP16 boxes on Cortex-A55) | `prim::neon_ltrb`: the decoder's NEON dequantize and four-anchor dist2bbox fused, sixteen anchors per step; NEON quantize | `neon` |
| `dfl_box` | as `dfl` for the distances, as `dist2bbox` for the boxes; `f16` outputs of 8-bit inputs on FP16 CPUs run the table distances and the f16 box math | as above | `neon`, `neon_fp16` for `f16` outputs on FP16 CPUs |
| `binary_i8` | `ADD`, `SUB`, `MUL`, every `TfliteKernels` and activation | `ops/binary_i8_neon.rs` | `neon` |
| `matmul_i8` | `BATCH_MATMUL`, 1×1 `CONV_2D`, per-tensor or per-column requantization, bias | `ops/matmul_i8_neon.rs` (`sdot` on dotprod CPUs) | `neon_dotprod` on dotprod CPUs, else `neon` |
| `softmax_dot_i8` | 16-bin rows in groups of 16 (YOLO DFL heads); other bin counts and the last rows short of a group run the portable row | `ops/softmax_dot_i8_neon.rs` | `neon` |
| `attention_i8` | with or without the Q `MUL`, every `TfliteKernels` | `ops/attention_i8_neon.rs` on dotprod CPUs, `ops/attention_i8_neon_base.rs` otherwise | `neon_dotprod` on dotprod CPUs, else `neon` |

`convert::dequant_*` and `nchw_to_nhwc` are NEON for every element size and have no plan or tier; `prim::transpose::nchw_to_nhwc_scalar` is the element loop, for the decoder's scalar tier. `matmul_i8` and `attention_i8` report `neon_dotprod` on CPUs with `sdot`.

## Numerics

### `softmax_i8` (TFLite int8 `SOFTMAX`)

Reproduces TFLite's default int8 kernel, `optimized_ops::Softmax<int8_t, int8_t>` with the float table (v2.19.0, `PopulateSoftmaxLookupTable`), not `kReference`:

- Table: `table[255 − v] = expf(−input_scale · beta · v)` for `v` in 0..=255, built in `new`.
- Per row: `max = max(x)`; `sum = Σ table[255 − max + x]` in f32; `inv = 1 / (sum · output_scale)`; `q = round(table[255 − max + x] · inv) + output_zp` (half away from zero), clamped to [−128, 127]. The index spans exactly 0..=255.
- The output quantization must be scale 1/256 (±0.1%) and zero point −128, as TFLite's Prepare enforces; anything else is `Unsupported`. The input zero point is unused (softmax is shift-invariant).
- The scalar tier sums in element order, as TFLite does, and is bit-exact where `f32::exp` is glibc's `expf` (the Linux targets the fixtures come from; another libm can move a table entry by an ulp). The NEON kernel sums in four interleaved lanes and builds a per-row 256-entry output table mapped with `tbl`/`tbx`; reordering the sum can move a value on a rounding boundary, so NEON is within ±1 LSB. Rows of exactly 16 elements take their exponentials as exact products of two 16-entry 16-bit fixed-point tables and sum them in integers, sixteen rows at a time where they come in groups and one at a time otherwise, with the same arithmetic, so the op and the fused attention agree on them; they are within ±1 LSB of the scalar tier too, without bias. Strided rows longer than `ops::rows::STAGE` take the portable kernel.

### `softmax_i16` (TFLite int16 `SOFTMAX`)

Reproduces `reference_ops::SoftmaxInt16` bit for bit on every tier (TFLite has no optimized int16 softmax):

- Input zero point 0, output zero point 0 and scale 1/32768 (±0.1%); anything else is `Unsupported`.
- Two 513-entry int16 tables built in f64 as `LUTPopulate<int16_t>` does: `exp` over [−10, 0] and `1/(1+x)` over [0, 1], with `LUTLookup`'s linear interpolation.
- `x − max` is rescaled with `QuantizeMultiplier(input_scale · beta / (10/65535))` and `MultiplyByQuantizedMultiplier` (the default, non-`TFLITE_SINGLE_ROUNDING` variant), exps summed in an `i32`, the reciprocal taken through the `1/(1+x)` table, and each output is `(exp · recip + round) >> (31 − headroom)`, clamped to [0, 32767].
- All integer, so the sum order does not matter and the NEON kernel is bit-identical to the portable one. `run` returns `Unsupported` for rows longer than `i32::MAX / 32767` (65 538) elements, whose exp sum could overflow.

### `sigmoid` and `lut_i8`

`SigmoidPlan` picks its path at creation:

1. `i8 → i8` and `u8 → u8`: `LutI8Plan::sigmoid`, the 256-entry table TFLite's `LOGISTIC` builds (`LUTPopulateInt8`): `clamp(round(sigmoid(in_scale · (v − in_zp)) · (1/out_scale)) + out_zp)` in f32 with `expf`, half away from zero. Bit-identical to TFLite where `f32::exp` is glibc's `expf`. Unlike TFLite, any output quantization is accepted.
2. `i16 → i16` with input zero point 0 and output (1/32768, 0): TFLite's int16 `LOGISTIC` bit for bit on every tier, gemmlowp's fixed-point logistic for an input scale of 2⁻¹², otherwise the reference table kernel with TFLite's prepared multiplier. Input scales above about 5.3, where that kernel's `i32` product overflows, take path 5. The NEON kernels (`ops/sigmoid_i16_neon.rs`) are gemmlowp's `FixedPoint<int16x8_t>` ops and the reference arithmetic in `int32x4_t` lanes, reading the 256-entry table with one 32-bit load per lane from a table of adjacent entry pairs.
3. Any input to `f32`: dequantize and sigmoid in f32, with the f32 exp on every tier. Integer inputs run the decoder's fused NEON kernel on every NEON tier, with the decoder's results; f16 and f32 inputs run the one-pass `prim::neon_sigmoid` kernel, also the decoder's.
4. Any input to `f16`: on the FP16 tier, `prim::neon_fp16`: `exp(−|x|)` as `2^k · p(r)` with a degree-4 polynomial, `s = e / (1 + e)` in f16 and `1 − s` for `x ≥ 0` (so every f16 just below 1 is reachable), tested within 2 f16 ulps of the float64 sigmoid rounded to f16 (every 8- and 16-bit value and every f16 at fixed quantizations, and the integer inputs over a seeded sweep of random quantizations). Integer and f32 inputs are dequantized and range-reduced in f32 first (rounding `x` to f16 would cost up to `|x|` ulps in the negative tail); f16 inputs are reduced in f16. On the other tiers, that tier's path-3 result rounded to f16 once (nearest even), in one pass on NEON (`prim::neon_sigmoid`).
5. Anything else: dequantize to f32, path 3's f32 sigmoid, quantize once (the NEON quantize on NEON tiers); within ±1 LSB of a float64 sigmoid.

`LutI8Plan` applies any 256-entry table elementwise (`from_table`), NEON `tbl` over four 16-byte table registers; there is no 16-bit counterpart.

A float input's `Quant` is ignored on every path, so paths 3 and 4 convert it as is.

### `dfl`, `dist2bbox` and `dfl_box`

The decode ops cover a whole concatenated multi-scale tensor in one call. Their `levels: Vec<Level { anchors, grid_w, stride }>` describe the anchor axis in order and must add up to it; anchor `i` of a level sits at grid cell `(i % grid_w + 0.5, i / grid_w + 0.5)`, built in `new`. The input layout comes from the view's strides, so `[anchors, 4 · reg_max]` and `[4 · reg_max, anchors]` are both accepted without a copy; contiguous anchor-major is the fast path.

- `DflPlan`: softmax over each side's `reg_max` bins, then the bin-weighted sum, giving `[l, t, r, b]` in grid units. 8-bit inputs use the exp-table kernel `prim::dfl_lut`, NEON on NEON tiers for every `reg_max` (the decoder's 16-bin kernel for 16, `decode_dfl_lut_generic_neon` / `dfl_lut_ltrb_generic_neon` with the same bits otherwise); other inputs are dequantized (NEON on NEON tiers) and use the decoder's fused softmax, NEON on NEON tiers, with the fp16 exp for `i16` and `u16` inputs to `f32` outputs on the FP16 tier, as the decoder does.
- `Dist2BboxPlan`: `xc = (gx + (r − l)/2)·s`, `yc = (gy + (b − t)/2)·s`, `w = (l + r)·s`, `h = (t + b)·s`, the Ultralytics order; `BoxFormat::Xywh` or `Xyxy` (applied to the f32 `[xc, yc, w, h]`). Every input runs `prim::neon_ltrb` on NEON tiers, the decoder's NEON dequantize and four-anchor arithmetic in one pass, which the decoder's LTRB cells call too.
- `DflBoxPlan`: the two fused, without the intermediate tensor.

Inputs are `i8`, `u8`, `i16`, `u16`, `f16` or `f32`; outputs `f32`, `f16` or any of the four integer types with an output `Quant`. **`f32` outputs of integer inputs equal the decoder's**: the kernels are the decoder's own, and a one-level call gives the decoder's level result bit for bit. **Integer outputs round once**: the input is dequantized, computed in f32, and quantized once at the end (half away from zero, clamp; the NEON quantize on NEON tiers). That is within ±1 LSB of a float64 reference, for 8- and 16-bit outputs alike, and more accurate than TFLite's int8 chain (SOFTMAX → CONV_2D → MUL → SUB/ADD → MUL, requantizing at every step), which it does not reproduce bit for bit.

**`f16` outputs** are the tier's `f32` result rounded once (nearest even) on the scalar and `neon` tiers. On the FP16 tier every path but `dfl` from 8-bit inputs and `dist2bbox` runs `prim::neon_fp16`; for those two the f32 path rounded once measured faster on a Cortex-A55:

- DFL from 16-bit and float inputs: the NEON dequantize to f32, `x − max` in f32 rounded to f16, `exp` as `2^k · p(r)` in f16 (the sigmoid kernels' polynomial), eight bins per vector summed with their bin-weighted sum in two alternating f16 sets, then each side's lanes reduced and divided in f32 and rounded to f16.
- DFL from 8-bit inputs: `DflBoxPlan` runs the exp-table kernels' f32 distances and the f16 box math below (`tier()` is `NeonFp16`); `DflPlan` has no box math, so it rounds the f32 distances once and reports `Neon`. f16 sums of an f16 exp table were measured slower than the f32 table kernels on a Cortex-A55, whose cost is the table loads.
- dist2bbox fused in `dfl_box`: eight anchors per vector in f16, on the grid centres and stride rounded to f16, in the Ultralytics order for `Xywh`; `Xyxy` is computed directly as `[(gx − l)·s, (gy − t)·s, (gx + r)·s, (gy + b)·s]`.

They are tested against the float64 result rounded to f16, in f16 ulps of the value or of the output's full scale (640 px for boxes, `reg_max − 1` grid units for distances), whichever is larger: within 1 ulp up to `reg_max` 32, and at `reg_max` 64 within 2 ulps for distances and 3 for boxes, at fixed and seeded random quantizations.

### `binary_i8` (TFLite int8 `ADD`, `SUB`, `MUL`)

Bit for bit with the TFLite kernels the plan's `TfliteKernels` names, on every tier:

- `MUL`: `clamp(zo + MultiplyByQuantizedMultiplier((x − zx)·(y − zy), M, s))`, the multiplier from `sx·sy/so` in f32.
- `ADD` and `SUB`: both inputs shifted left by 20 and rescaled by `s/(2·max(sx, sy))`, then the sum or difference rescaled by `2·max(sx, sy)/(2^20·so)`.
- Under `TfliteKernels::NeonRuy` (every aarch64 build, and x86 builds with SSE4.1 through `NEON_2_SSE`), whole 16-element blocks take TFLite's NEON arithmetic (`MUL` saturates the rescaled product to int16 before adding `zo`; `ADD` applies the shifts as one exact shift before a single `vqrdmulh` and keeps the low 16 bits) and the last `n % 16` elements its scalar code. `SUB` always runs the scalar code, as do the reference kernels.
- `table_for_rhs(rhs, len)` is the 256-entry table of `x op rhs` for a call of `len` elements, or `None` when that call's NEON blocks and scalar tail would disagree for this `rhs`.
- Output multipliers that need a left shift beyond 30 (`sx·sy/so` of 2³⁰ or more) are refused (`Unsupported`): TFLite's shift overflows there and its NEON and scalar code disagree.

### `matmul_i8` (TFLite int8 `BATCH_MATMUL` and 1×1 `CONV_2D`)

- Accumulation is exact in i32: `Σ (l − zl)(r − zr) = Σ l·r − zr·Σ l − zl·Σ r + K·zl·zr`, so every tier and pool gives the same bytes.
- Requantization as TFLite: ruy's single rounding under `NeonRuy` and `Ruy` (multipliers below 0.5 only; ruy's NEON kernels truncate above), `MultiplyByQuantizedMultiplier` under `NeonGemmlowp` (gemmlowp rounds as the reference kernels) and `Reference`; per-column multipliers and an optional bias for `CONV_2D`; the fused activation clamp.
- The reduction axis is at most 4096 (`check_depth`). An empty reduction (`K = 0`) writes each column's requantized bias on every tier. The NEON kernel computes 4×4 tiles with `sdot` on dotprod CPUs and `smull`/`sadalp` otherwise; operands not contiguous along the reduction axis are packed four rows at a time.

### `attention_i8` (fused `[MUL] → BATCH_MATMUL → SOFTMAX → BATCH_MATMUL`)

Reproduces the four TFLite int8 ops in one kernel, bit for bit except that the int8 softmax rows on NEON are within ±1 LSB, as `softmax_i8`'s:

- The optional Q `MUL` through its 256-entry table (`BinaryI8Plan::table_for_rhs`), the scores requantized as the first `BATCH_MATMUL`, the int8 softmax row by row, the outputs requantized as the second.
- Q, K and V are packed once per call into the caller's `AttentionScratch`, in tasks on the pool, with the sums that let the zero points start the accumulators.
- Dotprod CPUs run 4×4 `sdot` tiles over 4-byte-interleaved blocks, the step loop scheduled for in-order cores such as the Cortex-A55. Other NEON CPUs run a widening kernel: the scores as `smlal` by element over 8-token key groups, the outputs as u8×u8 `umlal` into 16-bit accumulators on `p + 128` and `v + 128`. That is exact because `Σ P·V ≤ 255·Σ P` and tokens run in chunks where every row's `Σ P ≤ 257`, after tokens whose probability is zero in all four rows are dropped. Both kernels end at the same integers.
- At most 2048 tokens and head dims of 256 (`check_shape`). Each task keeps about 25 KB of stack, 22 KB more on CPUs without FEAT_DotProd.

### `softmax_dot_i8` (YOLO DFL softmax and projection)

The softmax over each side's bins and the 1×1 projection onto the bin indices, without TFLite's 8-bit probabilities between them: the expectation comes from the exponentials and is requantized once, so it sits closer to the float model than TFLite's two-op chain, within a few LSB of it. The scalar tier takes the exponentials from the int8 softmax's f32 table and sums in f32; the NEON kernel (rows of 16) takes them as 12-bit fixed-point products of two 16-entry tables and sums exactly in integers, within ±1 LSB of the scalar tier. A plan whose weight spread and requantization could carry the 12-bit rounding past one output step (a bound computed when the plan is built) runs the scalar tier instead; YOLO DFL heads are well inside it.

## C and Python surfaces

Every plan has the same shape in all three languages, with typed per-operator symbols and no string-keyed registry.

- **C** ([`../kernels-capi/ARCHITECTURE.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels-capi/ARCHITECTURE.md)): `ef_<op>_params`, `ef_<op>_plan_new(params, &plan)`, `ef_<op>_plan_run(plan, &in, &out, pool)`, `ef_<op>_plan_free(plan)`. Tensors are `ef_kernel_view { data, dtype, ndim, shape, strides }` with element strides (`NULL` is contiguous), levels a pointer plus a count, pools `ef_kernel_pool { ctx, run }` or `NULL`. Calls return 0, or −1 with `errno` `EINVAL` / `ENOTSUP` and a thread-local `ef_kernels_last_error()` message; every entry point catches panics.
- **Python** ([`../python-kernels/README.md`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-kernels/README.md)): one class per plan (`SoftmaxI8Plan(...)`), `run(input, output, *, threads=None)` over NumPy arrays of any non-negative strides, writing into the caller's array with the GIL released; `ValueError` for bad arguments, `NotImplementedError` for `Unsupported`.

Operators with two inputs take both views; the output may not overlap either, the inputs may overlap each other. Attention's scratch is an `ef_attention_scratch` handle built from a plan in C, and cached per shape inside the Python class.

## Adding an operator

1. **Portable kernel.** Write the reference kernel and its semantics doc comment (formula, rounding, accepted quantization) in `crates/kernels/src/ops/<op>.rs`, as `softmax_row_i8_portable` in `ops/softmax_i8.rs`. Port TFLite integer arithmetic into `src/tflite.rs` from the v2.19.0 sources, not from memory; put slice primitives the decoder could share in `src/prim/`.
2. **NEON kernels.** Add `src/ops/<op>_neon.rs` (`#[cfg(target_arch = "aarch64")]`, `#[target_feature(enable = "neon")]`; FP16 and dotprod through `.arch_extension` inline asm behind `CpuFeatures`), with a bit-for-bit or ±1 LSB test against the portable kernel, as in `ops/softmax_i8_neon.rs` and `ops/lut_i8_neon.rs`.
3. **Plan type.** `<Op>Params`, `<Op>Plan::new` / `with_features` (validation, tables, kernel selection), `tier()` and `run` (the `kernels.<op>` span, validation before any write, then `ops::rows::for_each_row` or the `ops::levels` block loop). Declare the module in `src/ops/mod.rs` and re-export the types from `src/lib.rs`. Cover `new_rejects_bad_params`, `run_rejects_mismatch_without_writing`, strided and zero-sized views, and `common::assert_pool_and_concurrency_match` in `crates/kernels/tests/<family>.rs`, across `common::tiers()`.
4. **TFLite fixture.** When the op has a TFLite counterpart, add a `Spec` to `scripts/gen_kernel_fixtures.py`, regenerate into `testdata/kernels/` (Git LFS) from the repo venv with `scripts/requirements-fixtures.txt`, and add a parity test that loads it through `common::load`.
5. **C API.** Add the `#[repr(C)]` params struct to `crates/kernels-capi/src/types.rs` and `ef_<op>_plan_new` / `_run` / `_free` to `src/plans.rs` (bodies through `error::ffi`, `run` through `run_plan`, which checks alignment and overlap); the build regenerates `include/edgefirst/kernels.h`. In `crates/kernels-capi/cbindgen.toml`, add the plan's opaque `typedef struct ef_<op>_plan ef_<op>_plan;` to `header`, the Rust plan type to `[export] exclude`, and `[export.rename]` entries for the plan and the params struct (`no_rust_type_name_leaks_into_the_c_api` fails otherwise). A new params struct is new ABI surface; adding a field to an existing one bumps `ef_kernels_abi_version`. Test it in `crates/kernels-capi/src/tests.rs`, and in `tests/c/test_link_and_run.c` when the op needs a C consumer.
6. **Python.** Add the class to `crates/python-common/src/kernels.rs`, register it in `crates/python-kernels/src/lib.rs`, stub it in `crates/python-kernels/python/edgefirst/kernels/__init__.pyi` (`tests/packaging/test_stub_parity.py` checks the stub), and test it in `tests/kernels/test_kernels.py`.
7. **Coverage gate.** Add the plan, every (input, output) pair it accepts and every quant variant to `crates/kernels/tests/neon_coverage.rs`, which requires the tier the table above names under `CpuFeatures::probe()` and `Tier::Scalar` under `CpuFeatures::default()`.
8. **Benchmark and docs.** Add a case to `crates/kernels/benches/kernels_benchmark.rs`; update this file (module map, numerics, [Tracing Spans](#tracing-spans)), the root `ARCHITECTURE.md` span catalog, [`TESTING.md`](TESTING.md) and `CHANGELOG.md`.

## Tracing Spans

Every plan's `run` opens one `tracing::trace_span!` named `kernels.<op>`, with field `tier` (`scalar` / `neon` / `neon_fp16` / `neon_dotprod`, as `Tier::as_str`), the same macro and level as the decoder's `decoder.per_scale_run`. Spans are per call, never per row or block, so they cost one relaxed atomic load when no subscriber is active. Linked into a delegate, they nest under the framework's own delegate slice in a trace. The decoder calls the `prim` functions directly, so a decoder trace contains none of them.

| Span | Plan | What it measures |
|------|------|------------------|
| `kernels.softmax_i8` | `SoftmaxI8Plan` | int8 softmax over the last axis, all rows |
| `kernels.softmax_i16` | `SoftmaxI16Plan` | int16 softmax over the last axis, all rows |
| `kernels.lut_i8` | `LutI8Plan` | 256-entry table lookup over the tensor |
| `kernels.sigmoid` | `SigmoidPlan` | elementwise sigmoid over the tensor, whichever path it takes |
| `kernels.dfl` | `DflPlan` | DFL over the whole anchor axis |
| `kernels.dist2bbox` | `Dist2BboxPlan` | dist2bbox over the whole anchor axis |
| `kernels.dfl_box` | `DflBoxPlan` | fused DFL and dist2bbox over the whole anchor axis |
| `kernels.binary_i8` | `BinaryI8Plan` | int8 `ADD`, `SUB` or `MUL` over the tensor (field `op`) |
| `kernels.matmul_i8` | `MatMulI8Plan` | int8 matrix product over every batch |
| `kernels.softmax_dot_i8` | `SoftmaxDotI8Plan` | DFL softmax and projection over every row |
| `kernels.attention_i8` | `AttentionI8Plan` | fused attention over every head, packing included |

## Inter-Crate Interfaces

| Direction | Crate | Interface |
|-----------|-------|-----------|
| Consumed by | [`edgefirst-decoder`](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/) | `prim::*` primitives and NEON tiers, `CpuFeatures`, `Quant` (from the decoder's `Quantization`) |
| Consumed by | [`edgefirst-kernels-capi`](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels-capi/) | Every plan, behind `kernels.h` |
| Consumed by | [`edgefirst-python-common`](https://github.com/EdgeFirstAI/hal/blob/main/crates/python-common/) (feature `kernels`) | Every plan, as the `edgefirst.kernels` classes |

## Cross-References

- Testing: [TESTING.md](TESTING.md)
- Decoder per-scale framework: [../decoder/ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/decoder/ARCHITECTURE.md)
- Project architecture and span catalog: [../../ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/ARCHITECTURE.md)
