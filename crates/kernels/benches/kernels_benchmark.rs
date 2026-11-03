// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::hint::black_box;
use std::num::NonZeroUsize;

use edgefirst_bench::{run_bench, BenchSuite};
use edgefirst_kernels::convert::nchw_to_nhwc;
use edgefirst_kernels::{
    BoxFormat, CpuFeatures, DType, DflBoxParams, DflBoxPlan, DflParams, DflPlan, Dist2BboxParams,
    Dist2BboxPlan, Level, Quant, ScopedThreads, SigmoidParams, SigmoidPlan, SoftmaxI16Params,
    SoftmaxI16Plan, SoftmaxI8Params, SoftmaxI8Plan, TensorMut, TensorRef, View, ViewMut,
};
use half::f16;
use rand::{rngs::StdRng, RngExt, SeedableRng};

const WARMUP: usize = 5;
const ITERATIONS: usize = 50;
const ITERATIONS_LARGE: usize = 20;

const TIERS: [&str; 4] = ["scalar", "neon", "neon_fp16", "neon_dotprod"];

/// Shapes `[1, 2, n, n]` softmaxed over the last axis.
const SOFTMAX_SIZES: [usize; 2] = [400, 1008];

fn rng() -> StdRng {
    StdRng::seed_from_u64(0x4B45_524E)
}

fn iters_for(elements: usize) -> usize {
    if elements > 1_000_000 {
        ITERATIONS_LARGE
    } else {
        ITERATIONS
    }
}

/// Features of every tier this CPU supports; the others print a skip warning per case.
fn tier_features(case: &str) -> Vec<(&'static str, CpuFeatures)> {
    let mut out = Vec::new();
    for tier in TIERS {
        match CpuFeatures::from_forced_tier(Some(tier)) {
            Ok(f) => out.push((tier, f)),
            Err(e) => println!("WARNING: SKIPPED: {case}/{tier}: {e}"),
        }
    }
    out
}

fn report(suite: &mut BenchSuite, result: &edgefirst_bench::BenchResult) {
    result.print_summary();
    suite.record(result);
}

/// A straightforward per-element float-table int8 softmax, the stock-speed reference.
fn scalar_tflite_loop(
    input: &[i8],
    output: &mut [i8],
    row_len: usize,
    table: &[f32; 256],
    out_zp: i32,
) {
    for (src, dst) in input
        .chunks_exact(row_len)
        .zip(output.chunks_exact_mut(row_len))
    {
        let mut max = i8::MIN;
        for &x in src {
            max = max.max(x);
        }
        let mut sum = 0.0f32;
        for &x in src {
            sum += table[(255 - i32::from(max) + i32::from(x)) as usize];
        }
        let inv = 1.0 / sum;
        for (&x, o) in src.iter().zip(dst.iter_mut()) {
            let p = table[(255 - i32::from(max) + i32::from(x)) as usize] * inv;
            let q = (p * 256.0).round() as i32 + out_zp;
            *o = q.clamp(-128, 127) as i8;
        }
    }
}

fn softmax_i8_input(len: usize, row_len: usize) -> Vec<i8> {
    let mut r = rng();
    let mut v = vec![0i8; len];
    // A per-row offset makes the row maxima vary.
    for row in v.chunks_mut(row_len) {
        let base: i32 = r.random_range(-100..60);
        for x in row {
            *x = (base + r.random_range(-28..68)).clamp(-128, 127) as i8;
        }
    }
    v
}

fn bench_softmax_i8(suite: &mut BenchSuite) {
    let input_q = Quant::new(0.128, -15);
    let output_q = Quant::new(1.0 / 256.0, -128);
    for n in SOFTMAX_SIZES {
        let shape = [1usize, 2, n, n];
        let len = 2 * n * n;
        let iters = iters_for(len);
        let shape_name = format!("1x2x{n}x{n}");
        let input = softmax_i8_input(len, n);
        let mut output = vec![0i8; len];

        let mut table = [0f32; 256];
        for (i, t) in table.iter_mut().enumerate() {
            *t = (input_q.scale * (i as f32 - 255.0)).exp();
        }
        let name = format!("kernels/softmax_i8/{shape_name}/contiguous/scalar_tflite_loop");
        let r = run_bench(&name, WARMUP, iters, || {
            scalar_tflite_loop(
                black_box(&input),
                &mut output,
                n,
                &table,
                output_q.zero_point,
            );
            black_box(&output);
        });
        report(suite, &r);

        let case = format!("kernels/softmax_i8/{shape_name}/contiguous");
        for (tier, features) in tier_features(&case) {
            let plan = SoftmaxI8Plan::with_features(
                SoftmaxI8Params {
                    input: input_q,
                    output: output_q,
                    beta: 1.0,
                },
                features,
            )
            .expect("softmax_i8 plan");
            let name = format!("{case}/{tier}");
            let src = TensorRef::I8(View::contiguous(&input, &shape).expect("input view"));
            let mut dst =
                TensorMut::I8(ViewMut::contiguous(&mut output, &shape).expect("output view"));
            let r = run_bench(&name, WARMUP, iters, || {
                plan.run(src, dst.reborrow(), None).expect("softmax_i8 run");
                black_box(&dst);
            });
            report(suite, &r);
        }
    }

    // The 1008x1008 case again with two worker threads.
    let n = 1008;
    let shape = [1usize, 2, n, n];
    let input = softmax_i8_input(2 * n * n, n);
    let mut output = vec![0i8; 2 * n * n];
    let pool = ScopedThreads::new(NonZeroUsize::new(2).expect("two threads"));
    let case = format!("kernels/softmax_i8/1x2x{n}x{n}/threads2");
    for (tier, features) in tier_features(&case) {
        let plan = SoftmaxI8Plan::with_features(
            SoftmaxI8Params {
                input: input_q,
                output: output_q,
                beta: 1.0,
            },
            features,
        )
        .expect("softmax_i8 plan");
        let name = format!("{case}/{tier}");
        let src = TensorRef::I8(View::contiguous(&input, &shape).expect("input view"));
        let mut dst = TensorMut::I8(ViewMut::contiguous(&mut output, &shape).expect("output view"));
        let r = run_bench(&name, WARMUP, ITERATIONS_LARGE, || {
            plan.run(src, dst.reborrow(), Some(&pool))
                .expect("softmax_i8 run");
            black_box(&dst);
        });
        report(suite, &r);
    }
}

fn bench_softmax_i16(suite: &mut BenchSuite) {
    let input_q = Quant::new(1.0 / 2048.0, 0);
    let output_q = Quant::new(1.0 / 32768.0, 0);
    for n in SOFTMAX_SIZES {
        let shape = [1usize, 2, n, n];
        let len = 2 * n * n;
        let mut r = rng();
        let input: Vec<i16> = (0..len).map(|_| r.random_range(-16000..16000)).collect();
        let mut output = vec![0i16; len];
        let case = format!("kernels/softmax_i16/1x2x{n}x{n}/contiguous");
        for (tier, features) in tier_features(&case) {
            let plan = SoftmaxI16Plan::with_features(
                SoftmaxI16Params {
                    input: input_q,
                    output: output_q,
                    beta: 1.0,
                },
                features,
            )
            .expect("softmax_i16 plan");
            let name = format!("{case}/{tier}");
            let src = TensorRef::I16(View::contiguous(&input, &shape).expect("input view"));
            let mut dst =
                TensorMut::I16(ViewMut::contiguous(&mut output, &shape).expect("output view"));
            let res = run_bench(&name, WARMUP, iters_for(len), || {
                plan.run(src, dst.reborrow(), None)
                    .expect("softmax_i16 run");
                black_box(&dst);
            });
            report(suite, &res);
        }
    }
}

fn bench_sigmoid(suite: &mut BenchSuite) {
    let shape = [1usize, 80, 8400];
    let len = 80 * 8400;
    let mut r = rng();
    let input: Vec<i8> = (0..len).map(|_| r.random::<i8>()).collect();
    let mut output = vec![0i8; len];
    let case = "kernels/sigmoid/1x80x8400/i8_i8";
    for (tier, features) in tier_features(case) {
        let plan = SigmoidPlan::with_features(
            SigmoidParams {
                input_dtype: DType::I8,
                input: Quant::new(0.1, -30),
                output_dtype: DType::I8,
                output: Quant::new(1.0 / 256.0, -128),
            },
            features,
        )
        .expect("sigmoid plan");
        let name = format!("{case}/{tier}");
        let src = TensorRef::I8(View::contiguous(&input, &shape).expect("input view"));
        let mut dst = TensorMut::I8(ViewMut::contiguous(&mut output, &shape).expect("output view"));
        let res = run_bench(&name, WARMUP, ITERATIONS, || {
            plan.run(src, dst.reborrow(), None).expect("sigmoid run");
            black_box(&dst);
        });
        report(suite, &res);
    }

    // The YOLO26 score tail: ten classes over the 1344x768 anchor axis, int8 to f32.
    let shape = [1usize, 10, 21168];
    let len = 10 * 21168;
    let input: Vec<i8> = (0..len).map(|_| r.random::<i8>()).collect();
    let mut output = vec![0f32; len];
    let case = "kernels/sigmoid/1x10x21168/i8_f32";
    for (tier, features) in tier_features(case) {
        let plan = SigmoidPlan::with_features(
            SigmoidParams {
                input_dtype: DType::I8,
                input: Quant::new(0.1, -30),
                output_dtype: DType::F32,
                output: Quant::IDENTITY,
            },
            features,
        )
        .expect("sigmoid plan");
        let name = format!("{case}/{tier}");
        let src = TensorRef::I8(View::contiguous(&input, &shape).expect("input view"));
        let mut dst =
            TensorMut::F32(ViewMut::contiguous(&mut output, &shape).expect("output view"));
        let res = run_bench(&name, WARMUP, ITERATIONS, || {
            plan.run(src, dst.reborrow(), None).expect("sigmoid run");
            black_box(&dst);
        });
        report(suite, &res);
    }
}

/// Times one sigmoid case on every tier.
fn bench_sigmoid_case(
    suite: &mut BenchSuite,
    case: &str,
    params: SigmoidParams,
    src: TensorRef<'_>,
    mut dst: TensorMut<'_>,
) {
    for (tier, features) in tier_features(case) {
        let plan = SigmoidPlan::with_features(params, features).expect("sigmoid plan");
        let name = format!("{case}/{tier}");
        let res = run_bench(&name, WARMUP, ITERATIONS, || {
            plan.run(src, dst.reborrow(), None).expect("sigmoid run");
            black_box(&dst);
        });
        report(suite, &res);
    }
}

/// The int16, f16-output and requantizing sigmoid paths on the `[1, 80, 8400]` score shape.
fn bench_sigmoid_paths(suite: &mut BenchSuite) {
    let shape = [1usize, 80, 8400];
    let len = 80 * 8400;
    let mut r = rng();
    let in_i16: Vec<i16> = (0..len).map(|_| r.random::<i16>()).collect();
    let in_i8: Vec<i8> = (0..len).map(|_| r.random::<i8>()).collect();
    let mut out_i16 = vec![0i16; len];
    let mut out_f16 = vec![f16::ZERO; len];
    let mut out_u8 = vec![0u8; len];
    let q16 = Quant::new(1.0 / 32768.0, 0);
    for (case, scale) in [
        ("kernels/sigmoid/1x80x8400/i16_i16_pot", 2f32.powi(-12)),
        ("kernels/sigmoid/1x80x8400/i16_i16_reference", 0.0005),
    ] {
        let params = SigmoidParams {
            input_dtype: DType::I16,
            input: Quant::new(scale, 0),
            output_dtype: DType::I16,
            output: q16,
        };
        let src = TensorRef::I16(View::contiguous(&in_i16, &shape).expect("input view"));
        let dst = TensorMut::I16(ViewMut::contiguous(&mut out_i16, &shape).expect("output view"));
        bench_sigmoid_case(suite, case, params, src, dst);
    }
    let src16 = TensorRef::I16(View::contiguous(&in_i16, &shape).expect("input view"));
    let params = SigmoidParams {
        input_dtype: DType::I16,
        input: Quant::new(0.0005, 0),
        output_dtype: DType::F16,
        output: Quant::IDENTITY,
    };
    let dst = TensorMut::F16(ViewMut::contiguous(&mut out_f16, &shape).expect("output view"));
    bench_sigmoid_case(
        suite,
        "kernels/sigmoid/1x80x8400/i16_f16",
        params,
        src16,
        dst,
    );
    let src = TensorRef::I8(View::contiguous(&in_i8, &shape).expect("input view"));
    let params = SigmoidParams {
        input_dtype: DType::I8,
        input: Quant::new(0.1, -30),
        output_dtype: DType::F16,
        output: Quant::IDENTITY,
    };
    let dst = TensorMut::F16(ViewMut::contiguous(&mut out_f16, &shape).expect("output view"));
    bench_sigmoid_case(suite, "kernels/sigmoid/1x80x8400/i8_f16", params, src, dst);
    let params = SigmoidParams {
        output_dtype: DType::U8,
        output: Quant::new(1.0 / 256.0, 0),
        ..params
    };
    let dst = TensorMut::U8(ViewMut::contiguous(&mut out_u8, &shape).expect("output view"));
    bench_sigmoid_case(suite, "kernels/sigmoid/1x80x8400/i8_u8", params, src, dst);
    let in_f32: Vec<f32> = (0..len).map(|_| r.random::<f32>() * 16.0 - 8.0).collect();
    let mut out_f32 = vec![0f32; len];
    let params = SigmoidParams {
        input_dtype: DType::F32,
        input: Quant::IDENTITY,
        output_dtype: DType::F32,
        output: Quant::IDENTITY,
    };
    let src = TensorRef::F32(View::contiguous(&in_f32, &shape).expect("input view"));
    let dst = TensorMut::F32(ViewMut::contiguous(&mut out_f32, &shape).expect("output view"));
    bench_sigmoid_case(suite, "kernels/sigmoid/1x80x8400/f32_f32", params, src, dst);
    let params = SigmoidParams {
        output_dtype: DType::F16,
        ..params
    };
    let dst = TensorMut::F16(ViewMut::contiguous(&mut out_f16, &shape).expect("output view"));
    bench_sigmoid_case(suite, "kernels/sigmoid/1x80x8400/f32_f16", params, src, dst);
    let in_f16: Vec<f16> = in_f32.iter().map(|&v| f16::from_f32(v)).collect();
    let params = SigmoidParams {
        input_dtype: DType::F16,
        ..params
    };
    let src = TensorRef::F16(View::contiguous(&in_f16, &shape).expect("input view"));
    let dst = TensorMut::F16(ViewMut::contiguous(&mut out_f16, &shape).expect("output view"));
    bench_sigmoid_case(suite, "kernels/sigmoid/1x80x8400/f16_f16", params, src, dst);
    let params = SigmoidParams {
        output_dtype: DType::F32,
        ..params
    };
    let dst = TensorMut::F32(ViewMut::contiguous(&mut out_f32, &shape).expect("output view"));
    bench_sigmoid_case(suite, "kernels/sigmoid/1x80x8400/f16_f32", params, src, dst);
}

/// `nchw_to_nhwc` on the 16-bit score layout and a 3-channel 8-bit image.
fn bench_transpose(suite: &mut BenchSuite) {
    let mut r = rng();
    let scores: Vec<i16> = (0..80 * 8400).map(|_| r.random::<i16>()).collect();
    let mut scores_out = vec![0i16; scores.len()];
    let res = run_bench(
        "kernels/nchw_to_nhwc/i16_80x8400",
        WARMUP,
        ITERATIONS,
        || {
            nchw_to_nhwc(&scores, 1, 8400, 80, &mut scores_out);
            black_box(&scores_out);
        },
    );
    report(suite, &res);
    let image: Vec<u8> = (0..3 * 640 * 640).map(|_| r.random::<u8>()).collect();
    let mut image_out = vec![0u8; image.len()];
    let res = run_bench(
        "kernels/nchw_to_nhwc/u8_c3_640x640",
        WARMUP,
        ITERATIONS,
        || {
            nchw_to_nhwc(&image, 640, 640, 3, &mut image_out);
            black_box(&image_out);
        },
    );
    report(suite, &res);
}

/// The three FPN levels of a 640x640 (8400 anchors) or 1344x768 (21168 anchors) input.
fn yolo_levels(anchors: usize) -> [Level; 3] {
    let (w, h) = if anchors == 8400 { (80, 80) } else { (168, 96) };
    [8usize, 16, 32].map(|s| Level {
        anchors: (w * 8 / s) * (h * 8 / s),
        grid_w: w * 8 / s,
        stride: s as f32,
    })
}

/// An `[anchors, channels]` view, anchor-major (contiguous) or channel-major (transposed).
fn layout_view<'a>(data: &'a [i8], anchors: usize, channels: usize, layout: &str) -> View<'a, i8> {
    if layout == "anchor_major" {
        View::contiguous(data, &[anchors, channels])
    } else {
        View::strided(data, &[anchors, channels], &[1, anchors as isize])
    }
    .expect("input view")
}

/// A contiguous `[anchors, 4]` output of `dtype` over the matching buffer.
fn box_output<'a>(
    dtype: DType,
    anchors: usize,
    f32s: &'a mut [f32],
    f16s: &'a mut [f16],
    i8s: &'a mut [i8],
) -> TensorMut<'a> {
    let shape = [anchors, 4];
    match dtype {
        DType::F32 => TensorMut::F32(ViewMut::contiguous(f32s, &shape).expect("output view")),
        DType::F16 => TensorMut::F16(ViewMut::contiguous(f16s, &shape).expect("output view")),
        _ => TensorMut::I8(ViewMut::contiguous(i8s, &shape).expect("output view")),
    }
}

fn bench_dfl_box(suite: &mut BenchSuite) {
    const REG_MAX: usize = 16;
    const CH: usize = 4 * REG_MAX;
    for anchors in [8400, 21168] {
        let levels = yolo_levels(anchors);
        let mut r = rng();
        let input: Vec<i8> = (0..anchors * CH).map(|_| r.random::<i8>()).collect();
        let mut out_f32 = vec![0f32; anchors * 4];
        let mut out_f16 = vec![f16::ZERO; anchors * 4];
        let mut out_i8 = vec![0i8; anchors * 4];
        let outputs: &[(&str, DType)] = if anchors == 8400 {
            &[
                ("i8_f32", DType::F32),
                ("i8_f16", DType::F16),
                ("i8_i8", DType::I8),
            ]
        } else {
            &[("i8_f32", DType::F32), ("i8_i8", DType::I8)]
        };
        for &(out_name, out_dtype) in outputs {
            for layout in ["anchor_major", "channel_major"] {
                let case = format!("kernels/dfl_box/{anchors}/{out_name}_{layout}");
                for (tier, features) in tier_features(&case) {
                    let plan = DflBoxPlan::with_features(
                        DflBoxParams {
                            input_dtype: DType::I8,
                            input: Quant::new(0.11, -20),
                            output_dtype: out_dtype,
                            output: Quant::new(0.25, -128),
                            reg_max: REG_MAX,
                            levels: levels.to_vec(),
                            format: BoxFormat::Xywh,
                        },
                        features,
                    )
                    .expect("dfl_box plan");
                    let name = format!("{case}/{tier}");
                    let src = TensorRef::I8(layout_view(&input, anchors, CH, layout));
                    let mut dst =
                        box_output(out_dtype, anchors, &mut out_f32, &mut out_f16, &mut out_i8);
                    let res = run_bench(&name, WARMUP, ITERATIONS, || {
                        plan.run(src, dst.reborrow(), None).expect("dfl_box run");
                        black_box(&dst);
                    });
                    report(suite, &res);
                }
            }
        }
    }
}

/// Times `plan.run` on every tier under `case`; `plan` builds the plan for a tier's features.
fn bench_decode_case<P>(
    suite: &mut BenchSuite,
    case: &str,
    plan: impl Fn(CpuFeatures) -> P,
    run: impl Fn(&P, TensorRef<'_>, TensorMut<'_>),
    src: TensorRef<'_>,
    mut dst: TensorMut<'_>,
) {
    for (tier, features) in tier_features(case) {
        let p = plan(features);
        let name = format!("{case}/{tier}");
        let res = run_bench(&name, WARMUP, ITERATIONS, || {
            run(&p, src, dst.reborrow());
            black_box(&dst);
        });
        report(suite, &res);
    }
}

/// 16-bit and float logits into the fused DFL box decode, and `DflPlan` at `reg_max` 8.
fn bench_decode_inputs(suite: &mut BenchSuite) {
    const ANCHORS: usize = 8400;
    let levels = yolo_levels(ANCHORS);
    let mut r = rng();
    let in_i16: Vec<i16> = (0..ANCHORS * 64).map(|_| r.random::<i16>()).collect();
    let in_f32: Vec<f32> = (0..ANCHORS * 64)
        .map(|_| r.random::<f32>() * 16.0 - 8.0)
        .collect();
    let in_i8: Vec<i8> = (0..ANCHORS * 32).map(|_| r.random::<i8>()).collect();
    let mut out_f32 = vec![0f32; ANCHORS * 4];
    let mut out_f16 = vec![f16::ZERO; ANCHORS * 4];
    let mut out_i8 = vec![0i8; ANCHORS * 4];
    let run_box = |p: &DflBoxPlan, s: TensorRef<'_>, d: TensorMut<'_>| {
        p.run(s, d, None).expect("dfl_box run");
    };
    for (name, output_dtype) in [("f32", DType::F32), ("f16", DType::F16)] {
        for input_dtype in [DType::I16, DType::F32] {
            let src = match input_dtype {
                DType::I16 => {
                    TensorRef::I16(View::contiguous(&in_i16, &[ANCHORS, 64]).expect("view"))
                }
                _ => TensorRef::F32(View::contiguous(&in_f32, &[ANCHORS, 64]).expect("view")),
            };
            let in_name = if input_dtype == DType::I16 {
                "i16"
            } else {
                "f32"
            };
            let case = format!("kernels/dfl_box/{ANCHORS}/{in_name}_{name}_anchor_major");
            let plan = |features| {
                DflBoxPlan::with_features(
                    DflBoxParams {
                        input_dtype,
                        input: Quant::new(0.0007, -100),
                        output_dtype,
                        output: Quant::IDENTITY,
                        reg_max: 16,
                        levels: levels.to_vec(),
                        format: BoxFormat::Xywh,
                    },
                    features,
                )
                .expect("dfl_box plan")
            };
            let dst = box_output(
                output_dtype,
                ANCHORS,
                &mut out_f32,
                &mut out_f16,
                &mut out_i8,
            );
            bench_decode_case(suite, &case, plan, run_box, src, dst);
        }
    }
    let src = TensorRef::I8(View::contiguous(&in_i8, &[ANCHORS, 32]).expect("view"));
    for (name, output_dtype) in [("i8_f32", DType::F32), ("i8_f16", DType::F16)] {
        let case = format!("kernels/dfl/{ANCHORS}/reg_max_8/{name}");
        let plan = |features| {
            DflPlan::with_features(
                DflParams {
                    input_dtype: DType::I8,
                    input: Quant::new(0.11, -20),
                    output_dtype,
                    output: Quant::IDENTITY,
                    reg_max: 8,
                },
                features,
            )
            .expect("dfl plan")
        };
        let run = |p: &DflPlan, s: TensorRef<'_>, d: TensorMut<'_>| {
            p.run(s, d, None).expect("dfl run");
        };
        let dst = box_output(
            output_dtype,
            ANCHORS,
            &mut out_f32,
            &mut out_f16,
            &mut out_i8,
        );
        bench_decode_case(suite, &case, plan, run, src, dst);
    }
}

/// The YOLO26 box tail: int8 LTRB distances (`reg_max` 1) over the 1344x768 anchor axis.
fn bench_dist2bbox(suite: &mut BenchSuite) {
    let anchors = 21168;
    let levels = yolo_levels(anchors);
    let mut r = rng();
    let input: Vec<i8> = (0..anchors * 4).map(|_| r.random::<i8>()).collect();
    let mut out_f32 = vec![0f32; anchors * 4];
    let mut out_f16 = vec![f16::ZERO; anchors * 4];
    let mut out_i8 = vec![0i8; anchors * 4];
    for (out_name, out_dtype) in [("i8_f32", DType::F32), ("i8_i8", DType::I8)] {
        for layout in ["anchor_major", "channel_major"] {
            let case = format!("kernels/dist2bbox/{anchors}/{out_name}_{layout}");
            for (tier, features) in tier_features(&case) {
                let plan = Dist2BboxPlan::with_features(
                    Dist2BboxParams {
                        input_dtype: DType::I8,
                        input: Quant::new(0.05, -128),
                        output_dtype: out_dtype,
                        output: Quant::new(4.0, -128),
                        levels: levels.to_vec(),
                        format: BoxFormat::Xywh,
                    },
                    features,
                )
                .expect("dist2bbox plan");
                let name = format!("{case}/{tier}");
                let src = TensorRef::I8(layout_view(&input, anchors, 4, layout));
                let mut dst =
                    box_output(out_dtype, anchors, &mut out_f32, &mut out_f16, &mut out_i8);
                let res = run_bench(&name, WARMUP, ITERATIONS, || {
                    plan.run(src, dst.reborrow(), None).expect("dist2bbox run");
                    black_box(&dst);
                });
                report(suite, &res);
            }
        }
    }
}

/// Float LTRB distances and `f16` outputs of the dist2bbox tail over the 1344x768 anchor axis.
fn bench_dist2bbox_paths(suite: &mut BenchSuite) {
    let anchors = 21168;
    let levels = yolo_levels(anchors);
    let mut r = rng();
    let in_i8: Vec<i8> = (0..anchors * 4).map(|_| r.random::<i8>()).collect();
    let in_f32: Vec<f32> = (0..anchors * 4).map(|_| r.random::<f32>() * 16.0).collect();
    let mut out_f32 = vec![0f32; anchors * 4];
    let mut out_f16 = vec![f16::ZERO; anchors * 4];
    let mut out_i8 = vec![0i8; anchors * 4];
    let run = |p: &Dist2BboxPlan, s: TensorRef<'_>, d: TensorMut<'_>| {
        p.run(s, d, None).expect("dist2bbox run");
    };
    for (name, input_dtype, output_dtype) in [
        ("f32_f32", DType::F32, DType::F32),
        ("f32_f16", DType::F32, DType::F16),
        ("i8_f16", DType::I8, DType::F16),
    ] {
        let src = match input_dtype {
            DType::I8 => TensorRef::I8(View::contiguous(&in_i8, &[anchors, 4]).expect("view")),
            _ => TensorRef::F32(View::contiguous(&in_f32, &[anchors, 4]).expect("view")),
        };
        let case = format!("kernels/dist2bbox/{anchors}/{name}_anchor_major");
        let plan = |features| {
            Dist2BboxPlan::with_features(
                Dist2BboxParams {
                    input_dtype,
                    input: Quant::new(0.05, -128),
                    output_dtype,
                    output: Quant::IDENTITY,
                    levels: levels.to_vec(),
                    format: BoxFormat::Xywh,
                },
                features,
            )
            .expect("dist2bbox plan")
        };
        let dst = box_output(
            output_dtype,
            anchors,
            &mut out_f32,
            &mut out_f16,
            &mut out_i8,
        );
        bench_decode_case(suite, &case, plan, run, src, dst);
    }
}

/// The instance-mask kernels over one 160x160x32 proto frame (25 600 pixels, K = 32), a row at a
/// time as the decoder calls them: the 8-bit dot product with and without the in-kernel `Σ p`
/// (coefficient zero point 20 and 0), the f32 dot product of f32 and f16 protos, and the sigmoid.
#[cfg(target_arch = "aarch64")]
fn bench_masks(suite: &mut BenchSuite) {
    use edgefirst_kernels::prim::neon_mask as nm;
    let (h, w, k) = (160usize, 160usize, 32usize);
    let mut r = rng();
    let protos: Vec<i8> = (0..h * w * k).map(|_| r.random_range(-128..=127)).collect();
    let coeff: Vec<i8> = (0..k).map(|_| r.random_range(-60..=60)).collect();
    let pf: Vec<f32> = protos.iter().map(|&v| f32::from(v) * 0.02).collect();
    let ph: Vec<f16> = pf.iter().map(|&v| f16::from_f32(v)).collect();
    let cf: Vec<f32> = coeff.iter().map(|&v| f32::from(v) * 0.02).collect();
    let mut logits = vec![0i32; w];
    let mut lf = vec![0f32; w];
    let mut out = vec![0u8; w];
    let dotprod = std::arch::is_aarch64_feature_detected!("dotprod");
    type Dot8 = unsafe fn(&[i8], i32, &[i8], i32, &mut [i32]) -> bool;
    let mut dots: Vec<(&str, Dot8)> = vec![("neon", nm::mask_dot8_nhwc_neon::<i8, i8>)];
    if dotprod {
        dots.push(("dotprod", nm::mask_dot8_nhwc_dotprod::<i8, i8>));
    }
    for (tier, f) in dots {
        for zm in [0, 20] {
            let name = format!("kernels/mask/dot8_i8_i8/160x160x32/zm{zm}/{tier}");
            let result = run_bench(&name, WARMUP, ITERATIONS, || {
                for row in protos.chunks_exact(w * k) {
                    // SAFETY: NEON is baseline; the dotprod kernel is listed only when detected.
                    black_box(unsafe { f(&coeff, zm, row, -115, &mut logits) });
                }
            });
            result.print_summary();
            suite.record(&result);
        }
    }
    let name = "kernels/mask/dot_wide_i16_i8/160x160x32/neon";
    let c16: Vec<i16> = coeff.iter().map(|&v| i16::from(v) * 100).collect();
    let result = run_bench(name, WARMUP, ITERATIONS, || {
        for row in protos.chunks_exact(w * k) {
            // SAFETY: NEON is baseline.
            black_box(unsafe { nm::mask_dot_wide_nhwc_neon(&c16, 20, row, -115, &mut lf) });
        }
    });
    result.print_summary();
    suite.record(&result);
    let name = "kernels/mask/dot_f32_f32/160x160x32/neon";
    let result = run_bench(name, WARMUP, ITERATIONS, || {
        for row in pf.chunks_exact(w * k) {
            // SAFETY: NEON is baseline.
            black_box(unsafe { nm::mask_dot_f32_nhwc_neon(&cf, row, &mut lf) });
        }
    });
    result.print_summary();
    suite.record(&result);
    let name = "kernels/mask/dot_f32_f16/160x160x32/neon";
    let result = run_bench(name, WARMUP, ITERATIONS, || {
        for row in ph.chunks_exact(w * k) {
            // SAFETY: NEON is baseline.
            black_box(unsafe { nm::mask_dot_f32_nhwc_neon(&cf, row, &mut lf) });
        }
    });
    result.print_summary();
    suite.record(&result);
    // The same work on four cache-resident rows reused 40 times.
    let hot = |name: &str, suite: &mut BenchSuite, f: &mut dyn FnMut(usize)| {
        let result = run_bench(name, WARMUP, ITERATIONS, || {
            for i in 0..h {
                f(i % 4);
            }
        });
        result.print_summary();
        suite.record(&result);
    };
    if dotprod {
        hot(
            "kernels/mask/dot8_i8_i8/hot4x160x32/zm20/dotprod",
            suite,
            &mut |r| {
                let row = &protos[r * w * k..(r + 1) * w * k];
                // SAFETY: FEAT_DotProd was detected.
                black_box(unsafe {
                    nm::mask_dot8_nhwc_dotprod(&coeff, 20, row, -115, &mut logits)
                });
            },
        );
    }
    hot(
        "kernels/mask/dot_f32_f16/hot4x160x32/neon",
        suite,
        &mut |r| {
            let row = &ph[r * w * k..(r + 1) * w * k];
            // SAFETY: NEON is baseline.
            black_box(unsafe { nm::mask_dot_f32_nhwc_neon(&cf, row, &mut lf) });
        },
    );
    hot(
        "kernels/mask/dot_f32_f32/hot4x160x32/neon",
        suite,
        &mut |r| {
            let row = &pf[r * w * k..(r + 1) * w * k];
            // SAFETY: NEON is baseline.
            black_box(unsafe { nm::mask_dot_f32_nhwc_neon(&cf, row, &mut lf) });
        },
    );
    for (i, v) in logits.iter_mut().enumerate() {
        *v = (i as i32 - 80) * 4000;
    }
    let name = "kernels/mask/sigmoid_u8/160x160/neon";
    let result = run_bench(name, WARMUP, ITERATIONS, || {
        for _ in 0..h {
            // SAFETY: NEON is baseline.
            black_box(unsafe { nm::mask_sigmoid_u8_i32_neon(&logits, 4.35e-4, &mut out) });
        }
    });
    result.print_summary();
    suite.record(&result);
    let name = "kernels/mask/sigmoid_u8/160x160/scalar";
    let result = run_bench(name, WARMUP, ITERATIONS, || {
        for _ in 0..h {
            for (o, &l) in out.iter_mut().zip(&logits) {
                let x = l as f32 * 4.35e-4;
                *o = ((1.0 / (1.0 + (-x).exp())) * 255.0).round() as u8;
            }
            black_box(&out);
        }
    });
    result.print_summary();
    suite.record(&result);
}

#[cfg(not(target_arch = "aarch64"))]
fn bench_masks(_: &mut BenchSuite) {}

fn main() {
    let mut suite = BenchSuite::from_args();
    // `--section <name>` runs one group: softmax, sigmoid, transpose, dfl, dist2bbox or mask.
    let only = std::env::args()
        .collect::<Vec<_>>()
        .windows(2)
        .find(|w| w[0] == "--section")
        .map(|w| w[1].to_ascii_lowercase());
    type Group<'a> = (&'a str, &'a [fn(&mut BenchSuite)]);
    let groups: [Group; 6] = [
        ("softmax", &[bench_softmax_i8, bench_softmax_i16]),
        ("sigmoid", &[bench_sigmoid, bench_sigmoid_paths]),
        ("transpose", &[bench_transpose]),
        ("dfl", &[bench_dfl_box, bench_decode_inputs]),
        ("dist2bbox", &[bench_dist2bbox, bench_dist2bbox_paths]),
        ("mask", &[bench_masks]),
    ];
    for (name, benches) in groups {
        if only.as_deref().is_some_and(|o| o != name) {
            continue;
        }
        for bench in benches {
            bench(&mut suite);
        }
    }
    suite.finish();
}
