// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! int8 `ADD`, `SUB` and `MUL` against the TFLite fixtures, and the plan contract.

mod common;

use common::Graph;
use edgefirst_kernels::{
    Activation, BinaryI8Params, BinaryI8Plan, BinaryOp, CpuFeatures, DType, KernelError, LutI8Plan,
    Pool, Quant, TensorMut, TensorRef, TfliteKernels, Tier, View, ViewMut,
};

/// Fixtures whose `rhs` has the `lhs` shape.
const ELEMENTWISE: &[&str] = &[
    "add_i8",
    "add_i8_odd",
    "add_i8_pairs",
    "add_i8_pairs_ref",
    "sub_i8_odd",
    "mul_i8_odd",
    "mul_i8_odd_relu6",
    "mul_i8_odd_wide",
    "mul_i8_odd_wide_ref",
];

/// Fixtures whose `rhs` is one element, broadcast over `lhs`.
const SCALAR: &[&str] = &["mul_i8_scalar", "mul_i8_scalar_odd"];

/// The kernels a fixture's resolver ran. The fixtures come from LiteRT's x86_64 wheel, which
/// defines `USE_NEON` (SSE4.1 through `NEON_2_SSE`), so its optimized `ADD` and `MUL` are the
/// NEON-block arithmetic of an aarch64 build (`NeonGemmlowp`, which `NeonRuy` equals for these
/// ops); the reference resolver runs the scalar code.
fn kernels(g: &Graph) -> TfliteKernels {
    match g.meta("kernels") {
        "optimized" => TfliteKernels::NeonGemmlowp,
        "reference" => TfliteKernels::Reference,
        other => panic!("{}: unexpected kernels {other}", g.name),
    }
}

/// The other variant, which the discriminating fixtures must reject.
fn other(k: TfliteKernels) -> TfliteKernels {
    if k == TfliteKernels::Reference {
        TfliteKernels::NeonGemmlowp
    } else {
        TfliteKernels::Reference
    }
}

fn op(g: &Graph) -> BinaryOp {
    match g.meta("op") {
        "ADD" => BinaryOp::Add,
        "SUB" => BinaryOp::Sub,
        "MUL" => BinaryOp::Mul,
        other => panic!("{}: unexpected op {other}", g.name),
    }
}

fn activation(g: &Graph) -> Activation {
    match g.meta("activation") {
        "none" => Activation::None,
        "relu" => Activation::Relu,
        "relu_n1_to_1" => Activation::ReluN1To1,
        "relu6" => Activation::Relu6,
        other => panic!("{}: unexpected activation {other}", g.name),
    }
}

fn params(g: &Graph, kernels: TfliteKernels) -> BinaryI8Params {
    BinaryI8Params {
        op: op(g),
        lhs: g.quant("lhs"),
        rhs: g.quant("rhs"),
        output: g.quant("output"),
        activation: activation(g),
        kernels,
    }
}

/// Runs `plan` on the fixture's operands, `rhs` broadcast through zero strides when it is one
/// element.
fn run_fixture(plan: &BinaryI8Plan, g: &Graph, pool: Option<&dyn Pool>) -> Vec<i8> {
    let (lhs, rhs) = (g.i8("lhs"), g.i8("rhs"));
    let shape = g.shape("lhs").to_vec();
    let rhs_view = if rhs.len() == 1 {
        View::strided(&rhs, &shape, &vec![0; shape.len()]).unwrap()
    } else {
        View::contiguous(&rhs, &shape).unwrap()
    };
    let mut out = vec![0x55u8 as i8; lhs.len()];
    plan.run(
        TensorRef::I8(View::contiguous(&lhs, &shape).unwrap()),
        TensorRef::I8(rhs_view),
        TensorMut::I8(ViewMut::contiguous(&mut out, &shape).unwrap()),
        pool,
    )
    .unwrap();
    out
}

#[test]
fn binary_plans_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<BinaryI8Plan>();
    assert_send_sync::<BinaryI8Params>();
}

#[test]
fn fixtures_load() {
    for name in ELEMENTWISE.iter().chain(SCALAR) {
        let g = common::load_graph(name);
        let n: usize = g.shape("lhs").iter().product();
        assert_eq!(g.i8("output").len(), n, "{name}");
        let rhs = g.i8("rhs").len();
        assert!(rhs == n || rhs == 1, "{name}");
        assert!(matches!(g.meta("kernels"), "optimized" | "reference"));
    }
}

/// Every tier gives the fixture's bytes with the fixture's kernels; without `USE_NEON` the
/// optimized resolver would equal the reference one.
#[test]
fn every_tier_matches_tflite_bit_exact() {
    for name in ELEMENTWISE.iter().chain(SCALAR) {
        let g = common::load_graph(name);
        let want = g.i8("output");
        let k = kernels(&g);
        let variants: &[TfliteKernels] = if k == TfliteKernels::Reference {
            &[TfliteKernels::Reference, TfliteKernels::Ruy]
        } else {
            &[TfliteKernels::NeonGemmlowp, TfliteKernels::NeonRuy]
        };
        for &kernels in variants {
            for f in common::tiers() {
                let plan = BinaryI8Plan::with_features(params(&g, kernels), f).unwrap();
                let got = run_fixture(&plan, &g, None);
                let (bad, worst) = common::diff_i8(&got, &want);
                assert_eq!(
                    bad,
                    0,
                    "{name} {kernels:?} tier {:?}: {bad} of {} differ, worst {worst}",
                    plan.tier(),
                    want.len()
                );
            }
        }
    }
}

/// These fixtures tell TFLite's NEON-block arithmetic from its scalar code, so the variant
/// that matches them is a finding, not a coincidence: `ADD` over every input pair (the two
/// kernels differ on one) and `MUL` with a multiplier above one (on about 4% of elements).
#[test]
fn fixtures_distinguish_neon_blocks_from_the_scalar_code() {
    for name in [
        "add_i8_pairs",
        "add_i8_pairs_ref",
        "mul_i8_odd_wide",
        "mul_i8_odd_wide_ref",
    ] {
        let g = common::load_graph(name);
        let k = other(kernels(&g));
        let plan = BinaryI8Plan::with_features(params(&g, k), CpuFeatures::default()).unwrap();
        let (bad, worst) = common::diff_i8(&run_fixture(&plan, &g, None), &g.i8("output"));
        eprintln!("{name}: {k:?} differs in {bad} elements (worst {worst})");
        assert!(bad > 0, "{name} does not discriminate the NEON blocks");
    }
}

/// A broadcast scalar `MUL` is a table lookup; the table applies to the whole call.
#[test]
fn scalar_mul_through_a_table_matches_tflite() {
    for name in SCALAR {
        let g = common::load_graph(name);
        let (lhs, rhs) = (g.i8("lhs"), g.i8("rhs")[0]);
        for kernels in [kernels(&g), TfliteKernels::Reference] {
            let plan =
                BinaryI8Plan::with_features(params(&g, kernels), CpuFeatures::default()).unwrap();
            let table = plan.table_for_rhs(rhs, lhs.len()).expect("scalar table");
            for f in common::tiers() {
                let lut = LutI8Plan::from_table_with_features(DType::I8, table, f).unwrap();
                let mut out = vec![0i8; lhs.len()];
                lut.run(
                    TensorRef::I8(View::contiguous(&lhs, &[lhs.len()]).unwrap()),
                    TensorMut::I8(ViewMut::contiguous(&mut out, &[lhs.len()]).unwrap()),
                    None,
                )
                .unwrap();
                assert_eq!(out, g.i8("output"), "{name} {kernels:?} {:?}", lut.tier());
            }
        }
    }
}

fn good() -> BinaryI8Params {
    BinaryI8Params {
        op: BinaryOp::Add,
        lhs: Quant::new(0.05, -3),
        rhs: Quant::new(0.04, 7),
        output: Quant::new(0.08, 1),
        activation: Activation::None,
        kernels: TfliteKernels::NeonRuy,
    }
}

#[test]
fn new_rejects_bad_params() {
    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
        let p = BinaryI8Params { op, ..good() };
        assert!(BinaryI8Plan::new(p).is_ok(), "{op:?}");
        let invalid = |q: BinaryI8Params| {
            matches!(BinaryI8Plan::new(q), Err(KernelError::InvalidArgument(_)))
        };
        for s in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert!(
                invalid(BinaryI8Params {
                    lhs: Quant::new(s, 0),
                    ..p
                }),
                "{op:?} lhs {s}"
            );
            assert!(
                invalid(BinaryI8Params {
                    rhs: Quant::new(s, 0),
                    ..p
                }),
                "{op:?} rhs {s}"
            );
            assert!(
                invalid(BinaryI8Params {
                    output: Quant::new(s, 0),
                    ..p
                }),
                "{op:?} output {s}"
            );
        }
        for zp in [-129, 128] {
            assert!(invalid(BinaryI8Params {
                lhs: Quant::new(0.1, zp),
                ..p
            }));
            assert!(invalid(BinaryI8Params {
                rhs: Quant::new(0.1, zp),
                ..p
            }));
            assert!(invalid(BinaryI8Params {
                output: Quant::new(0.1, zp),
                ..p
            }));
        }
        // Every fused activation leaves a non-empty clamp for any valid output quantization.
        for activation in [Activation::Relu, Activation::ReluN1To1, Activation::Relu6] {
            for output in [Quant::new(1.0e-6, 127), Quant::new(1.0e6, -128)] {
                let q = BinaryI8Params {
                    output,
                    activation,
                    ..p
                };
                assert!(
                    BinaryI8Plan::new(q).is_ok(),
                    "{op:?} {activation:?} {output:?}"
                );
            }
        }
    }
    // ADD's output rescale must stay below one; SUB has TFLite's greater-than-one path.
    let tiny_out = |op| BinaryI8Params {
        op,
        lhs: Quant::new(1.0, 0),
        rhs: Quant::new(1.0, 0),
        output: Quant::new(1.0e-9, 0),
        ..good()
    };
    assert!(matches!(
        BinaryI8Plan::new(tiny_out(BinaryOp::Add)),
        Err(KernelError::Unsupported(_))
    ));
    assert!(BinaryI8Plan::new(tiny_out(BinaryOp::Sub)).is_ok());
    // A MUL multiplier that overflows f32.
    assert!(matches!(
        BinaryI8Plan::new(BinaryI8Params {
            op: BinaryOp::Mul,
            lhs: Quant::new(f32::MAX, 0),
            rhs: Quant::new(f32::MAX, 0),
            ..good()
        }),
        Err(KernelError::InvalidArgument(_))
    ));
}

fn i8v<'a>(d: &'a [i8], s: &[usize]) -> TensorRef<'a> {
    TensorRef::I8(View::contiguous(&d[..s.iter().product()], s).unwrap())
}

#[test]
fn run_rejects_mismatch_without_writing() {
    let plan = BinaryI8Plan::new(good()).unwrap();
    let a = vec![1i8; 64];
    let b = vec![2i8; 64];
    let wide = vec![1i16; 64];
    let mut out = vec![0x55i8; 64];
    let untouched = |o: &[i8]| o.iter().all(|&v| v == 0x55);

    let r = plan.run(
        TensorRef::I16(View::contiguous(&wide, &[4, 16]).unwrap()),
        i8v(&b, &[4, 16]),
        TensorMut::I8(ViewMut::contiguous(&mut out, &[4, 16]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(untouched(&out));

    let r = plan.run(
        i8v(&a, &[4, 16]),
        TensorRef::U8(View::contiguous(&[0u8; 64], &[4, 16]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut out, &[4, 16]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(untouched(&out));

    let mut wide_out = vec![0x55i16; 64];
    let r = plan.run(
        i8v(&a, &[4, 16]),
        i8v(&b, &[4, 16]),
        TensorMut::I16(ViewMut::contiguous(&mut wide_out, &[4, 16]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(wide_out.iter().all(|&v| v == 0x55));

    for (ls, rs, os) in [
        (&[4usize, 16][..], &[4usize, 16][..], &[4usize, 15][..]),
        (&[4, 16], &[4, 15], &[4, 16]),
        (&[4, 15], &[4, 16], &[4, 16]),
        (&[64], &[4, 16], &[4, 16]),
        (&[4, 16], &[4, 16], &[2, 32]),
    ] {
        let n: usize = os.iter().product();
        let r = plan.run(
            i8v(&a, ls),
            i8v(&b, rs),
            TensorMut::I8(ViewMut::contiguous(&mut out[..n], os).unwrap()),
            None,
        );
        assert!(
            matches!(r, Err(KernelError::InvalidArgument(_))),
            "{ls:?} {rs:?} {os:?}: {r:?}"
        );
        assert!(untouched(&out));
    }
}

#[test]
fn run_rejects_aliasing_output_rows() {
    let plan = BinaryI8Plan::new(good()).unwrap();
    let a = vec![1i8; 32];
    let mut out = vec![0x55i8; 32];
    for strides in [[0isize, 1], [4, 1], [8, 0]] {
        let r = plan.run(
            TensorRef::I8(View::contiguous(&a, &[4, 8]).unwrap()),
            TensorRef::I8(View::contiguous(&a, &[4, 8]).unwrap()),
            TensorMut::I8(ViewMut::strided(&mut out, &[4, 8], &strides).unwrap()),
            None,
        );
        assert!(
            matches!(r, Err(KernelError::InvalidArgument(_))),
            "{strides:?}: {r:?}"
        );
        assert!(out.iter().all(|&v| v == 0x55));
    }
}

/// Transposed inputs, a padded output, a broadcast row and a broadcast column give the dense
/// result on every tier, and the padding is never written.
#[test]
fn strided_views_match_contiguous() {
    let (rows, cols) = (7usize, 37usize);
    let x: Vec<i8> = (0..rows * cols)
        .map(|i| (i * 31 % 256) as u8 as i8)
        .collect();
    let y: Vec<i8> = (0..rows * cols)
        .map(|i| (i * 17 % 251) as u8 as i8)
        .collect();
    // Column-major copies of x and y.
    let t = |v: &[i8]| -> Vec<i8> {
        let mut o = vec![0i8; v.len()];
        for r in 0..rows {
            for c in 0..cols {
                o[c * rows + r] = v[r * cols + c];
            }
        }
        o
    };
    let (xt, yt) = (t(&x), t(&y));
    let row: Vec<i8> = y[..cols].to_vec();
    let col: Vec<i8> = (0..rows).map(|r| y[r * cols]).collect();
    let shape = [rows, cols];
    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
        for kernels in [TfliteKernels::NeonRuy, TfliteKernels::Reference] {
            for f in common::tiers() {
                let plan = BinaryI8Plan::with_features(
                    BinaryI8Params {
                        op,
                        kernels,
                        ..good()
                    },
                    f,
                )
                .unwrap();
                let dense = |rhs: &[i8]| -> Vec<i8> {
                    let mut o = vec![0i8; rows * cols];
                    plan.run(
                        TensorRef::I8(View::contiguous(&x, &shape).unwrap()),
                        TensorRef::I8(View::contiguous(rhs, &shape).unwrap()),
                        TensorMut::I8(ViewMut::contiguous(&mut o, &shape).unwrap()),
                        None,
                    )
                    .unwrap();
                    o
                };
                let want = dense(&y);
                let pitch = cols + 5;
                let mut padded = vec![0x55i8; rows * pitch];
                plan.run(
                    TensorRef::I8(View::strided(&xt, &shape, &[1, rows as isize]).unwrap()),
                    TensorRef::I8(View::strided(&yt, &shape, &[1, rows as isize]).unwrap()),
                    TensorMut::I8(
                        ViewMut::strided(&mut padded, &shape, &[pitch as isize, 1]).unwrap(),
                    ),
                    None,
                )
                .unwrap();
                for r in 0..rows {
                    assert_eq!(
                        &padded[r * pitch..r * pitch + cols],
                        &want[r * cols..(r + 1) * cols],
                        "{op:?} {kernels:?} {:?} row {r}",
                        plan.tier()
                    );
                    assert!(padded[r * pitch + cols..(r + 1) * pitch]
                        .iter()
                        .all(|&v| v == 0x55));
                }
                // A transposed output.
                let mut out_t = vec![0i8; rows * cols];
                plan.run(
                    TensorRef::I8(View::contiguous(&x, &shape).unwrap()),
                    TensorRef::I8(View::contiguous(&y, &shape).unwrap()),
                    TensorMut::I8(
                        ViewMut::strided(&mut out_t, &shape, &[1, rows as isize]).unwrap(),
                    ),
                    None,
                )
                .unwrap();
                assert_eq!(t(&want), out_t, "{op:?} transposed output");
                // Broadcast rows and columns.
                let rows_b: Vec<i8> = (0..rows).flat_map(|_| row.iter().copied()).collect();
                let cols_b: Vec<i8> = (0..rows)
                    .flat_map(|r| std::iter::repeat_n(col[r], cols))
                    .collect();
                for (view, expanded) in [
                    (View::strided(&row, &shape, &[0, 1]).unwrap(), rows_b),
                    (View::strided(&col, &shape, &[1, 0]).unwrap(), cols_b),
                ] {
                    let mut o = vec![0i8; rows * cols];
                    plan.run(
                        TensorRef::I8(View::contiguous(&x, &shape).unwrap()),
                        TensorRef::I8(view),
                        TensorMut::I8(ViewMut::contiguous(&mut o, &shape).unwrap()),
                        None,
                    )
                    .unwrap();
                    // Elementwise blocks follow the flat output index, so a broadcast operand
                    // equals its dense expansion.
                    assert_eq!(o, dense(&expanded), "{op:?} broadcast");
                }
            }
        }
    }
}

#[test]
fn zero_sized_and_rank_zero_views() {
    for f in common::tiers() {
        let plan = BinaryI8Plan::with_features(good(), f).unwrap();
        let empty: [i8; 0] = [];
        let mut guard = [0x55i8; 4];
        for shape in [&[0usize, 16][..], &[4, 0], &[0]] {
            plan.run(
                TensorRef::I8(View::contiguous(&empty, shape).unwrap()),
                TensorRef::I8(View::contiguous(&empty, shape).unwrap()),
                TensorMut::I8(ViewMut::contiguous(&mut guard[..0], shape).unwrap()),
                None,
            )
            .unwrap();
        }
        assert_eq!(guard, [0x55; 4]);
        let (a, b) = ([20i8], [-10i8]);
        let mut o = [0x55i8];
        plan.run(
            TensorRef::I8(View::contiguous(&a, &[]).unwrap()),
            TensorRef::I8(View::contiguous(&b, &[]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut o, &[]).unwrap()),
            None,
        )
        .unwrap();
        let mut dense = [0i8];
        plan.run(
            TensorRef::I8(View::contiguous(&a, &[1]).unwrap()),
            TensorRef::I8(View::contiguous(&b, &[1]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut dense, &[1]).unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(o, dense);
    }
}

#[test]
fn pool_and_concurrent_runs_match_single_thread() {
    let (rows, cols) = (64usize, 1009usize);
    let x: Vec<i8> = (0..rows * cols)
        .map(|i| (i * 7919 % 256) as u8 as i8)
        .collect();
    let y: Vec<i8> = (0..rows * cols)
        .map(|i| (i * 104729 % 253) as u8 as i8)
        .collect();
    for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
        for f in common::tiers() {
            let plan = BinaryI8Plan::with_features(BinaryI8Params { op, ..good() }, f).unwrap();
            assert_eq!(plan.tier() == Tier::Scalar, f.tier() == Tier::Scalar);
            common::assert_pool_and_concurrency_match(|pool| {
                let mut out = vec![0i8; rows * cols];
                plan.run(
                    TensorRef::I8(View::contiguous(&x, &[rows, cols]).unwrap()),
                    TensorRef::I8(View::contiguous(&y, &[rows, cols]).unwrap()),
                    TensorMut::I8(ViewMut::contiguous(&mut out, &[rows, cols]).unwrap()),
                    pool,
                )
                .unwrap();
                out.iter().map(|&v| v as u8).collect()
            });
        }
    }
}
