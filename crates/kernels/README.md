# edgefirst-kernels

[![Crates.io](https://img.shields.io/crates/v/edgefirst-kernels.svg)](https://crates.io/crates/edgefirst-kernels)
[![Documentation](https://docs.rs/edgefirst-kernels/badge.svg)](https://docs.rs/edgefirst-kernels)
[![License](https://img.shields.io/crates/l/edgefirst-kernels.svg)](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE)

**Quantized and floating-point CPU kernels for edge AI inference.**

Softmax, sigmoid, 8-bit lookup tables and YOLO box decode (DFL, dist2bbox) over caller-owned slices, with NEON kernels selected at runtime on aarch64 and portable scalar kernels everywhere else. The int8 and int16 softmax and logistic reproduce TFLite's kernels, so a delegate can run them in place of the stock CPU ops.

## Role in edgefirst-hal

`edgefirst-kernels` sits at the bottom of the HAL's dependency graph: it depends only on `half` and `tracing`, never on `edgefirst-tensor`, GL or a framework. `edgefirst-decoder` builds its YOLO post-processing on these kernels; the C library `libedgefirst_kernels` ([`kernels-capi`](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels-capi/)) and the Python package `edgefirst-kernels` (`edgefirst.kernels`) expose the same plans, and neither links any other HAL library.

## Plans

| Plan | Operation | Numerics |
|------|-----------|----------|
| `SoftmaxI8Plan` | int8 softmax over the last axis | TFLite int8 `SOFTMAX`: scalar bit for bit, NEON within ±1 LSB |
| `SoftmaxI16Plan` | int16 softmax over the last axis | TFLite int16 `SOFTMAX`, bit for bit on every tier |
| `SigmoidPlan` | elementwise sigmoid, any of `i8`/`u8`/`i16`/`u16`/`f16`/`f32` in and out | int8/uint8 and int16 TFLite `LOGISTIC` bit for bit; integer inputs to `f32` as the decoder; `f16` outputs within 2 f16 ulps on FP16 CPUs; other combinations one rounding |
| `LutI8Plan` | 256-entry table over `i8`/`u8` (`from_table`, `sigmoid`) | exact |
| `DflPlan` | `[anchors, 4·reg_max]` logits to `[anchors, 4]` distances | `f32` outputs of integer inputs as the decoder; `f16` outputs within 1 f16 ulp on FP16 CPUs (2 at `reg_max` 64; ulps of max(value, `reg_max` − 1)), the `f32` result rounded once elsewhere and for 8-bit inputs; integer outputs one rounding, ±1 LSB |
| `Dist2BboxPlan` | distances to boxes over a multi-level anchor axis | `f32` outputs of integer inputs as the decoder; `f16` outputs the `f32` result rounded once on every tier; integer outputs one rounding, ±1 LSB |
| `DflBoxPlan` | fused DFL and dist2bbox | `f32` outputs of integer inputs as the decoder; `f16` boxes native FP16 on FP16 CPUs for every input, within 1 f16 ulp (3 at `reg_max` 64), the `f32` boxes rounded once elsewhere; integer outputs one rounding, ±1 LSB |

The `convert` module adds the per-tensor dequantization (`dequant_*_to_f32` / `_to_f16`) and NCHW→NHWC transpose slice kernels the decoder uses.

Every plan validates its parameters in `new` (returning `KernelError::Unsupported` for a valid configuration it does not handle, so `new` doubles as a capability check), then `run`s any number of times, concurrently, without allocating. Views take any non-negative element strides, so transposed and sliced tensors need no copy. Work can be split across a caller-supplied `Pool`; by default it runs on the calling thread.

## Quick Start

```rust
use edgefirst_kernels::{
    KernelError, Quant, SoftmaxI8Params, SoftmaxI8Plan, TensorMut, TensorRef, View, ViewMut,
};

fn main() -> Result<(), KernelError> {
    let plan = SoftmaxI8Plan::new(SoftmaxI8Params {
        input: Quant::new(0.1, 0),
        output: Quant::new(1.0 / 256.0, -128),
        beta: 1.0,
    })?;
    let input = [0i8, 0, 0, 0];
    let mut output = [0i8; 4];
    plan.run(
        TensorRef::I8(View::contiguous(&input, &[1, 4])?),
        TensorMut::I8(ViewMut::contiguous(&mut output, &[1, 4])?),
        None,
    )?;
    assert_eq!(output, [-64; 4]);
    Ok(())
}
```

YOLOv8 boxes from int8 DFL logits, one call for every level of the concatenated anchor axis:

```rust,ignore
use edgefirst_kernels::{BoxFormat, DType, DflBoxParams, DflBoxPlan, Level, Quant};

let plan = DflBoxPlan::new(DflBoxParams {
    input_dtype: DType::I8,
    input: Quant::new(0.08, -10),
    output_dtype: DType::F32,
    output: Quant::IDENTITY,
    reg_max: 16,
    levels: vec![
        Level { anchors: 6400, grid_w: 80, stride: 8.0 },
        Level { anchors: 1600, grid_w: 40, stride: 16.0 },
        Level { anchors: 400, grid_w: 20, stride: 32.0 },
    ],
    format: BoxFormat::Xywh,
})?;
```

## CPU tiers

`scalar`, `neon` (alias `neon_baseline`), `neon_fp16` and `neon_dotprod`, detected at runtime. `EDGEFIRST_KERNELS_FORCE_TIER=<tier>` forces the tier plans are built for; a tier the CPU lacks makes plan creation fail instead of falling back. On aarch64 every op path, for every dtype pair, runs a NEON kernel (native FP16 kernels for `f16` outputs on FP16 CPUs); `scalar` is the fallback for CPUs without NEON and the bit-level reference. `plan.tier()` reports the kernel that runs. Each `run` opens one `kernels.<op>` tracing span with a `tier` field.

## Documentation

- [ARCHITECTURE.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels/ARCHITECTURE.md) — plan/run contract, views and pools, numerics per op, and how to add an operator
- [TESTING.md](https://github.com/EdgeFirstAI/hal/blob/main/crates/kernels/TESTING.md) — TFLite fixtures, tier matrix, benchmarks

## License

Apache-2.0. See [LICENSE](https://github.com/EdgeFirstAI/hal/blob/main/LICENSE).
