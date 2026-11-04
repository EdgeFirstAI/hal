// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Every op path runs the NEON kernel `crates/kernels/ARCHITECTURE.md`'s tier table names on
//! aarch64; only a CPU without NEON runs the scalar tier.
//!
//! A new operator adds its plan, every (input, output) dtype pair it accepts, every quant
//! variant and its expected tier to `neon_coverage_of_every_op_path`. The int8-only ops
//! (`binary_i8`, `matmul_i8`, `softmax_dot_i8`, `attention_i8`) enumerate their rounding
//! (`TfliteKernels`), fused activation and requantization variants instead.

#![cfg(target_arch = "aarch64")]

use edgefirst_kernels::{
    Activation, AttentionI8Params, AttentionI8Plan, AttentionQMul, BinaryI8Params, BinaryI8Plan,
    BinaryOp, BoxFormat, CpuFeatures, DType, DflBoxParams, DflBoxPlan, DflParams, DflPlan,
    Dist2BboxParams, Dist2BboxPlan, KernelError, Level, LutI8Plan, MatMulI8Params, MatMulI8Plan,
    Quant, Requant, SigmoidParams, SigmoidPlan, SoftmaxDotI8Params, SoftmaxDotI8Plan,
    SoftmaxI16Params, SoftmaxI16Plan, SoftmaxI8Params, SoftmaxI8Plan, TfliteKernels, Tier,
};

const DTYPES: [DType; 6] = [
    DType::I8,
    DType::U8,
    DType::I16,
    DType::U16,
    DType::F16,
    DType::F32,
];

const REG_MAX: [usize; 5] = [1, 8, 16, 17, 64];

const TFLITE_KERNELS: [TfliteKernels; 3] = [
    TfliteKernels::NeonRuy,
    TfliteKernels::Ruy,
    TfliteKernels::Reference,
];

const ACTIVATIONS: [Activation; 4] = [
    Activation::None,
    Activation::Relu,
    Activation::ReluN1To1,
    Activation::Relu6,
];

const PROBS: Quant = Quant {
    scale: 1.0 / 256.0,
    zero_point: -128,
};

/// Plans checked so far, to catch an enumeration that silently shrinks.
#[derive(Default)]
struct Gate {
    cases: usize,
}

impl Gate {
    fn check(&mut self, what: &str, neon: Tier, want: Tier, scalar: Tier) {
        assert_eq!(neon, want, "{what}: tier under probe()");
        assert_eq!(scalar, Tier::Scalar, "{what}: not scalar without features");
        self.cases += 1;
    }
}

/// The tier the ARCHITECTURE table names for a path, `fp16` when the CPU has FP16 arithmetic.
fn fp16_or_neon(fp16: bool) -> Tier {
    if fp16 {
        Tier::NeonFp16
    } else {
        Tier::Neon
    }
}

/// `matmul_i8` and `attention_i8`: the `sdot` kernels on dotprod CPUs.
fn dotprod_or_neon(dotprod: bool) -> Tier {
    if dotprod {
        Tier::NeonDotprod
    } else {
        Tier::Neon
    }
}

/// `matmul_i8` variants: per-tensor and per-column requantization, with and without a bias,
/// under every fused activation and rounding.
fn matmul_variants() -> Vec<MatMulI8Params> {
    let mut out = Vec::new();
    for kernels in TFLITE_KERNELS {
        for activation in ACTIVATIONS {
            out.push(MatMulI8Params {
                activation,
                ..MatMulI8Params::tflite_batch_matmul(
                    Quant::new(0.05, -3),
                    Quant::new(0.04, 7),
                    Quant::new(0.5, 1),
                    kernels,
                )
                .expect("batch matmul params")
            });
            for bias in [Vec::new(), vec![5, -7, 9]] {
                out.push(
                    MatMulI8Params::tflite_conv(
                        Quant::new(0.05, -3),
                        &[0.01, 0.02, 0.03],
                        Quant::new(0.5, 1),
                        bias,
                        activation,
                        kernels,
                    )
                    .expect("conv params"),
                );
            }
        }
    }
    out
}

fn attention_variants() -> Vec<AttentionI8Params> {
    let base = AttentionI8Params {
        q: Quant::new(0.1251, -15),
        q_mul: None,
        k: Quant::new(0.1251, -15),
        v: Quant::new(0.1251, -15),
        scores: Quant::new(0.1279, -9),
        beta: 1.0,
        probs: PROBS,
        output: Quant::new(0.063_33, 2),
        kernels: TfliteKernels::NeonRuy,
    };
    let q_mul = AttentionQMul {
        table: core::array::from_fn(|i| i as u8),
        output: Quant::new(0.010_18, 6),
    };
    let mut out = Vec::new();
    for kernels in TFLITE_KERNELS {
        for q_mul in [None, Some(q_mul)] {
            out.push(AttentionI8Params {
                q_mul,
                kernels,
                ..base
            });
        }
    }
    out
}

/// `sigmoid`: `f16` outputs take the FP16 kernel on FP16 CPUs; every other pair is `neon`.
fn sigmoid_tier(output: DType, fp16: bool) -> Tier {
    fp16_or_neon(fp16 && output == DType::F16)
}

/// `dfl` (`boxes` false) and `dfl_box` (`boxes` true).
fn dfl_tier(input: DType, output: DType, boxes: bool, fp16: bool) -> Tier {
    let eight_bit = matches!(input, DType::I8 | DType::U8);
    let sixteen_bit = matches!(input, DType::I16 | DType::U16);
    fp16_or_neon(
        fp16 && match output {
            DType::F16 => boxes || !eight_bit,
            DType::F32 => sixteen_bit,
            _ => false,
        },
    )
}

fn input_quant(dtype: DType) -> Quant {
    match dtype {
        DType::I8 | DType::I16 => Quant::new(0.05, -3),
        DType::U8 | DType::U16 => Quant::new(0.05, 7),
        _ => Quant::IDENTITY,
    }
}

fn output_quant(dtype: DType) -> Quant {
    match dtype {
        DType::I8 => Quant::new(1.0 / 256.0, -128),
        DType::U8 => Quant::new(1.0 / 256.0, 0),
        DType::I16 => Quant::new(1.0 / 1024.0, 5),
        DType::U16 => Quant::new(1.0 / 1024.0, 9),
        _ => Quant::IDENTITY,
    }
}

fn levels() -> Vec<Level> {
    vec![
        Level {
            anchors: 16,
            grid_w: 4,
            stride: 8.0,
        },
        Level {
            anchors: 4,
            grid_w: 2,
            stride: 16.0,
        },
    ]
}

fn sigmoid_variants() -> Vec<SigmoidParams> {
    let mut out = Vec::new();
    for input_dtype in DTYPES {
        for output_dtype in DTYPES {
            out.push(SigmoidParams {
                input_dtype,
                input: input_quant(input_dtype),
                output_dtype,
                output: output_quant(output_dtype),
            });
        }
    }
    let i16_i16 = |scale: f32| SigmoidParams {
        input_dtype: DType::I16,
        input: Quant::new(scale, 0),
        output_dtype: DType::I16,
        output: Quant::new(1.0 / 32768.0, 0),
    };
    // TFLite int16 with gemmlowp (2^-12), the prepared reference multiplier, and a scale
    // above the overflow bound that takes the requantizing path.
    out.push(i16_i16(2f32.powi(-12)));
    out.push(i16_i16(0.0005));
    out.push(i16_i16(8.0));
    out
}

#[test]
fn neon_coverage_of_every_op_path() {
    let neon = CpuFeatures::probe();
    let none = CpuFeatures::default();
    assert!(neon.neon_baseline, "aarch64 probe lacks NEON");
    let fp16 = neon.fp16_usable();
    let mut g = Gate::default();

    for p in sigmoid_variants() {
        let what = format!("sigmoid {p:?}");
        let (a, b) = (
            SigmoidPlan::with_features(p, neon).expect("sigmoid probe"),
            SigmoidPlan::with_features(p, none).expect("sigmoid scalar"),
        );
        let want = sigmoid_tier(p.output_dtype, fp16);
        g.check(&what, a.tier(), want, b.tier());
    }

    for input_dtype in DTYPES {
        for output_dtype in DTYPES {
            for reg_max in REG_MAX {
                let p = DflParams {
                    input_dtype,
                    input: input_quant(input_dtype),
                    output_dtype,
                    output: output_quant(output_dtype),
                    reg_max,
                };
                let what = format!("dfl {p:?}");
                let (a, b) = (
                    DflPlan::with_features(p, neon).expect("dfl probe"),
                    DflPlan::with_features(p, none).expect("dfl scalar"),
                );
                let want = dfl_tier(input_dtype, output_dtype, false, fp16);
                g.check(&what, a.tier(), want, b.tier());

                let p = DflBoxParams {
                    input_dtype,
                    input: input_quant(input_dtype),
                    output_dtype,
                    output: output_quant(output_dtype),
                    reg_max,
                    levels: levels(),
                    format: BoxFormat::Xywh,
                };
                let what = format!("dfl_box {p:?}");
                let (a, b) = (
                    DflBoxPlan::with_features(p.clone(), neon).expect("dfl_box probe"),
                    DflBoxPlan::with_features(p, none).expect("dfl_box scalar"),
                );
                let want = dfl_tier(input_dtype, output_dtype, true, fp16);
                g.check(&what, a.tier(), want, b.tier());
            }

            for format in [BoxFormat::Xywh, BoxFormat::Xyxy] {
                let p = Dist2BboxParams {
                    input_dtype,
                    input: input_quant(input_dtype),
                    output_dtype,
                    output: output_quant(output_dtype),
                    levels: levels(),
                    format,
                };
                let what = format!("dist2bbox {p:?}");
                let (a, b) = (
                    Dist2BboxPlan::with_features(p.clone(), neon).expect("dist2bbox probe"),
                    Dist2BboxPlan::with_features(p, none).expect("dist2bbox scalar"),
                );
                g.check(&what, a.tier(), Tier::Neon, b.tier());
            }
        }
    }

    let p = SoftmaxI8Params {
        input: Quant::new(0.1, 0),
        output: Quant::new(1.0 / 256.0, -128),
        beta: 1.0,
    };
    g.check(
        "softmax_i8",
        SoftmaxI8Plan::with_features(p, neon)
            .expect("softmax_i8 probe")
            .tier(),
        Tier::Neon,
        SoftmaxI8Plan::with_features(p, none)
            .expect("softmax_i8 scalar")
            .tier(),
    );
    let p = SoftmaxI16Params {
        input: Quant::new(1.0 / 2048.0, 0),
        output: Quant::new(1.0 / 32768.0, 0),
        beta: 1.0,
    };
    g.check(
        "softmax_i16",
        SoftmaxI16Plan::with_features(p, neon)
            .expect("softmax_i16 probe")
            .tier(),
        Tier::Neon,
        SoftmaxI16Plan::with_features(p, none)
            .expect("softmax_i16 scalar")
            .tier(),
    );

    for dtype in [DType::I8, DType::U8] {
        let table = [0u8; 256];
        g.check(
            &format!("lut_i8 table {dtype:?}"),
            LutI8Plan::from_table_with_features(dtype, table, neon)
                .expect("lut probe")
                .tier(),
            Tier::Neon,
            LutI8Plan::from_table_with_features(dtype, table, none)
                .expect("lut scalar")
                .tier(),
        );
        let (input, output) = (input_quant(dtype), output_quant(dtype));
        g.check(
            &format!("lut_i8 sigmoid {dtype:?}"),
            LutI8Plan::sigmoid_with_features(dtype, input, output, neon)
                .expect("lut sigmoid probe")
                .tier(),
            Tier::Neon,
            LutI8Plan::sigmoid_with_features(dtype, input, output, none)
                .expect("lut sigmoid scalar")
                .tier(),
        );
    }

    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
        for kernels in TFLITE_KERNELS {
            for activation in ACTIVATIONS {
                let p = BinaryI8Params {
                    op,
                    lhs: Quant::new(0.05, -3),
                    rhs: Quant::new(0.04, 7),
                    output: Quant::new(0.08, 1),
                    activation,
                    kernels,
                };
                g.check(
                    &format!("binary_i8 {p:?}"),
                    BinaryI8Plan::with_features(p, neon)
                        .expect("binary probe")
                        .tier(),
                    Tier::Neon,
                    BinaryI8Plan::with_features(p, none)
                        .expect("binary scalar")
                        .tier(),
                );
            }
        }
    }

    let dotprod = neon.dotprod_usable();
    for p in matmul_variants() {
        g.check(
            &format!("matmul_i8 {p:?}"),
            MatMulI8Plan::with_features(p.clone(), neon)
                .expect("matmul probe")
                .tier(),
            dotprod_or_neon(dotprod),
            MatMulI8Plan::with_features(p, none)
                .expect("matmul scalar")
                .tier(),
        );
    }

    for kernels in TFLITE_KERNELS {
        for bias in [Vec::new(), vec![3]] {
            let p = SoftmaxDotI8Params {
                softmax: SoftmaxI8Params {
                    input: Quant::new(0.0947, 3),
                    output: PROBS,
                    beta: 1.0,
                },
                weights: (0..16).map(|i| i * 8).collect(),
                projection: MatMulI8Params::tflite_conv(
                    PROBS,
                    &[15.0 / 127.0],
                    Quant::new(0.0579, -128),
                    bias,
                    Activation::None,
                    kernels,
                )
                .expect("softmax_dot projection"),
            };
            g.check(
                &format!("softmax_dot_i8 {p:?}"),
                SoftmaxDotI8Plan::with_features(p.clone(), neon)
                    .expect("softmax_dot probe")
                    .tier(),
                Tier::Neon,
                SoftmaxDotI8Plan::with_features(p, none)
                    .expect("softmax_dot scalar")
                    .tier(),
            );
        }
    }

    for p in attention_variants() {
        g.check(
            &format!("attention_i8 {p:?}"),
            AttentionI8Plan::with_features(p, neon)
                .expect("attention probe")
                .tier(),
            dotprod_or_neon(dotprod),
            AttentionI8Plan::with_features(p, none)
                .expect("attention scalar")
                .tier(),
        );
    }

    // sigmoid: 36 pairs + 3 int16 variants; dfl and dfl_box: 36 pairs x 5 reg_max;
    // dist2bbox: 36 pairs x 2 formats; softmax 2; lut 4; binary_i8: 3 ops x 3 kernels x 4
    // activations; matmul_i8: 3 kernels x 4 activations x 3 (per-tensor, per-column with and
    // without bias); softmax_dot_i8: 3 kernels x 2 biases; attention_i8: 3 kernels x 2.
    assert_eq!(g.cases, 39 + 180 + 180 + 72 + 2 + 4 + 36 + 36 + 6 + 6);
}

#[test]
fn neon_coverage_rejections_are_errors() {
    let neon = CpuFeatures::probe();
    for dtype in [DType::I16, DType::U16, DType::F16, DType::F32] {
        assert!(matches!(
            LutI8Plan::from_table_with_features(dtype, [0; 256], neon),
            Err(KernelError::InvalidArgument(_))
        ));
        assert!(matches!(
            LutI8Plan::sigmoid_with_features(dtype, Quant::IDENTITY, Quant::IDENTITY, neon),
            Err(KernelError::InvalidArgument(_))
        ));
    }
    let dfl = |reg_max| DflParams {
        input_dtype: DType::I8,
        input: input_quant(DType::I8),
        output_dtype: DType::F32,
        output: Quant::IDENTITY,
        reg_max,
    };
    assert!(matches!(
        DflPlan::with_features(dfl(0), neon),
        Err(KernelError::InvalidArgument(_))
    ));
    assert!(matches!(
        DflPlan::with_features(dfl(65), neon),
        Err(KernelError::Unsupported(_))
    ));
    // Ruy rounding takes multipliers below 0.5 only.
    let wide = MatMulI8Params {
        requant: vec![Requant::from_real(0.75).expect("requant")],
        ..matmul_variants().remove(0)
    };
    assert!(matches!(
        MatMulI8Plan::with_features(wide, neon),
        Err(KernelError::Unsupported(_))
    ));
    assert!(matches!(
        AttentionI8Plan::check_shape(2049, 64, 64),
        Err(KernelError::Unsupported(_))
    ));
}
