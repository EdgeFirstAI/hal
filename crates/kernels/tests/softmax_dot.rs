// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The fused DFL softmax-then-dot against the TFLite `SOFTMAX` → `CONV_2D` fixture and its
//! float definition, and the plan contract.

mod common;

use common::Graph;
use edgefirst_kernels::{
    Activation, CpuFeatures, KernelError, MatMulI8Params, MatMulI8Plan, Quant, SoftmaxDotI8Params,
    SoftmaxDotI8Plan, SoftmaxI8Params, SoftmaxI8Plan, TensorMut, TensorRef, TfliteKernels, Tier,
    View, ViewMut,
};

const PROBS: Quant = Quant {
    scale: 1.0 / 256.0,
    zero_point: -128,
};

/// The fixture's chain as plan parameters; the host LiteRT's `CONV_2D` rounds like the
/// reference kernels (see `tests/matmul.rs`).
fn params(g: &Graph) -> SoftmaxDotI8Params {
    SoftmaxDotI8Params {
        softmax: SoftmaxI8Params {
            input: g.quant("input"),
            output: g.quant("probs"),
            beta: g.meta("beta").parse().unwrap(),
        },
        weights: g.i8("filter"),
        projection: MatMulI8Params::tflite_conv(
            g.quant("probs"),
            &g.f32_list("filter_scales"),
            g.quant("output"),
            vec![0],
            Activation::None,
            TfliteKernels::Reference,
        )
        .unwrap(),
    }
}

/// The output before rounding: `zo + E[w − zr] / probs_scale · real + bias`, clamped, with the
/// exact exponentials.
fn definition(p: &SoftmaxDotI8Params, row: &[i8]) -> f64 {
    let s = f64::from(p.softmax.input.scale * p.softmax.beta);
    let max = row.iter().copied().max().unwrap_or(0);
    let e: Vec<f64> = row
        .iter()
        .map(|&x| (s * f64::from(i32::from(x) - i32::from(max))).exp())
        .collect();
    let sum: f64 = e.iter().sum();
    let pr = &p.projection;
    let ws: f64 = e
        .iter()
        .zip(&p.weights)
        .map(|(e, &w)| e * f64::from(i32::from(w) - pr.rhs_zero_point))
        .sum();
    let r = pr.requant[0];
    let real = f64::from(r.multiplier) * 2f64.powi(r.shift - 31);
    let acc = ws / sum / f64::from(p.softmax.output.scale)
        + f64::from(pr.bias.first().copied().unwrap_or(0));
    let (lo, hi) = match pr.activation {
        Activation::None => (-128.0, 127.0),
        _ => (f64::from(pr.output.zero_point).max(-128.0), 127.0),
    };
    (f64::from(pr.output.zero_point) + acc * real).clamp(lo, hi)
}

fn run(plan: &SoftmaxDotI8Plan, x: &[i8], rows: usize, n: usize) -> Vec<i8> {
    let mut out = vec![0x55u8 as i8; rows];
    plan.run(
        TensorRef::I8(View::contiguous(x, &[rows, n]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut out, &[rows, 1]).unwrap()),
        None,
    )
    .unwrap();
    out
}

#[test]
fn softmax_dot_plans_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SoftmaxDotI8Plan>();
    assert_send_sync::<SoftmaxDotI8Params>();
}

/// The fixture is TFLite's `SOFTMAX` then `CONV_2D`: the crate's two single-op plans give its
/// bytes on the scalar tier.
#[test]
fn the_pair_of_plans_reproduces_the_fixture() {
    let g = common::load_graph("dfl_head_i8");
    let p = params(&g);
    let x = g.i8("input");
    let n = p.weights.len();
    let rows = x.len() / n;
    let f = CpuFeatures::default();
    let sm = SoftmaxI8Plan::with_features(p.softmax, f).unwrap();
    let mut probs = vec![0i8; x.len()];
    sm.run(
        TensorRef::I8(View::contiguous(&x, &[rows, n]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut probs, &[rows, n]).unwrap()),
        None,
    )
    .unwrap();
    let mm = MatMulI8Plan::with_features(p.projection.clone(), f).unwrap();
    let mut out = vec![0i8; rows];
    mm.run(
        TensorRef::I8(View::contiguous(&probs, &[rows, n]).unwrap()),
        TensorRef::I8(View::contiguous(&p.weights, &[n, 1]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut out, &[rows, 1]).unwrap()),
        None,
    )
    .unwrap();
    assert_eq!(common::diff_i8(&out, &g.i8("output")), (0, 0));
}

/// The fused plan does not round the probabilities to bytes, so it is not TFLite-exact: every
/// tier is within one LSB of the float definition, within one LSB of TFLite plus TFLite's own
/// error, and closer to the float result than TFLite in total.
#[test]
fn every_tier_is_within_its_bar_of_float_and_tflite() {
    let g = common::load_graph("dfl_head_i8");
    let p = params(&g);
    let x = g.i8("input");
    let tflite = g.i8("output");
    let n = p.weights.len();
    let rows = x.len() / n;
    assert_eq!(rows, tflite.len());
    let want: Vec<f64> = x.chunks(n).map(|r| definition(&p, r)).collect();
    let err_tflite: f64 = tflite
        .iter()
        .zip(&want)
        .map(|(&t, &w)| (f64::from(t) - w).abs())
        .sum();
    for f in common::tiers() {
        let plan = SoftmaxDotI8Plan::with_features(p.clone(), f).unwrap();
        let got = run(&plan, &x, rows, n);
        let mut err = 0.0;
        for (r, ((&o, &t), &w)) in got.iter().zip(&tflite).zip(&want).enumerate() {
            let (eo, et) = ((f64::from(o) - w).abs(), (f64::from(t) - w).abs());
            assert!(
                eo <= 1.0,
                "tier {:?} row {r}: {o} vs float {w:.3}",
                plan.tier()
            );
            let d = (i32::from(o) - i32::from(t)).abs();
            assert!(
                f64::from(d) <= et.ceil() + 1.0,
                "tier {:?} row {r}: {o} vs TFLite {t} (float {w:.3})",
                plan.tier()
            );
            err += eo;
        }
        let (bad, worst) = common::diff_i8(&got, &tflite);
        eprintln!(
            "softmax_dot tier={} vs TFLite: {bad} of {rows} differ (worst {worst}); mean |error| from float: fused {:.4}, TFLite {:.4}",
            plan.tier().as_str(),
            err / rows as f64,
            err_tflite / rows as f64
        );
        assert!(err < err_tflite, "fused {err:.1} vs TFLite {err_tflite:.1}");
    }
}

fn good() -> SoftmaxDotI8Params {
    SoftmaxDotI8Params {
        softmax: SoftmaxI8Params {
            input: Quant::new(0.0947, 3),
            output: PROBS,
            beta: 1.0,
        },
        weights: (0..16i32).map(|i| ((i * 127 + 7) / 15) as i8).collect(),
        projection: MatMulI8Params::tflite_conv(
            PROBS,
            &[15.0 / 127.0],
            Quant::new(0.0579, -128),
            vec![0],
            Activation::None,
            TfliteKernels::NeonRuy,
        )
        .unwrap(),
    }
}

#[test]
fn new_rejects_bad_params() {
    assert!(SoftmaxDotI8Plan::new(good()).is_ok());
    let invalid = |p: SoftmaxDotI8Params| {
        matches!(
            SoftmaxDotI8Plan::new(p),
            Err(KernelError::InvalidArgument(_))
        )
    };
    let unsupported = |p: SoftmaxDotI8Params| {
        matches!(SoftmaxDotI8Plan::new(p), Err(KernelError::Unsupported(_)))
    };
    let mut p = good();
    p.weights.clear();
    assert!(invalid(p));
    let mut p = good();
    p.projection.lhs_zero_point = 0;
    assert!(invalid(p));
    let mut p = good();
    p.projection.requant.push(p.projection.requant[0]);
    assert!(invalid(p));
    let mut p = good();
    p.projection.bias = vec![1, 2];
    assert!(invalid(p));
    for s in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        let mut p = good();
        p.softmax.input.scale = s;
        assert!(invalid(p), "input scale {s}");
        let mut p = good();
        p.softmax.beta = s;
        assert!(invalid(p), "beta {s}");
        let mut p = good();
        p.projection.output.scale = s;
        assert!(invalid(p), "output scale {s}");
    }
    // The softmax output must be TFLite's (1/256, −128), as for SoftmaxI8Plan.
    let mut p = good();
    p.softmax.output = Quant::new(1.0 / 128.0, -128);
    p.projection.lhs_zero_point = -128;
    assert!(unsupported(p));
    // Ruy rounding needs a multiplier below 0.5.
    let mut p = good();
    p.projection.requant[0] = edgefirst_kernels::Requant::from_real(0.75).unwrap();
    assert!(unsupported(p));
}

#[test]
fn run_rejects_mismatch_without_writing() {
    let plan = SoftmaxDotI8Plan::new(good()).unwrap();
    let x = vec![3i8; 4 * 32];
    let mut out = vec![0x55i8; 16];
    let untouched = |o: &[i8]| o.iter().all(|&v| v == 0x55);
    for (is, os) in [
        (&[4usize, 15][..], &[4usize, 1][..]),
        (&[4, 16], &[4, 2]),
        (&[4, 16], &[3, 1]),
        (&[4, 16], &[4]),
        (&[2, 2, 16], &[4, 1]),
        (&[], &[]),
    ] {
        let n: usize = is.iter().product();
        let m: usize = os.iter().product();
        let r = plan.run(
            TensorRef::I8(View::contiguous(&x[..n], is).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut out[..m], os).unwrap()),
            None,
        );
        assert!(
            matches!(r, Err(KernelError::InvalidArgument(_))),
            "{is:?} {os:?}: {r:?}"
        );
        assert!(untouched(&out));
    }
    let r = plan.run(
        TensorRef::U8(View::contiguous(&[0u8; 64], &[4, 16]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut out[..4], &[4, 1]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    // A row axis that is not unit-stride is unsupported.
    let r = plan.run(
        TensorRef::I8(View::strided(&x, &[4, 16], &[32, 2]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut out[..4], &[4, 1]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::Unsupported(_))), "{r:?}");
    assert!(untouched(&out));
}

#[test]
fn run_rejects_aliasing_output_rows() {
    let plan = SoftmaxDotI8Plan::new(good()).unwrap();
    let x = vec![3i8; 4 * 16];
    let mut out = vec![0x55i8; 8];
    for strides in [[0isize, 1], [1, 0]] {
        let r = plan.run(
            TensorRef::I8(View::contiguous(&x, &[4, 16]).unwrap()),
            TensorMut::I8(ViewMut::strided(&mut out, &[4, 2], &strides).unwrap()),
            None,
        );
        assert!(r.is_err(), "{strides:?}: {r:?}");
    }
    let r = plan.run(
        TensorRef::I8(View::contiguous(&x, &[4, 16]).unwrap()),
        TensorMut::I8(ViewMut::strided(&mut out, &[4, 1], &[0, 1]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(out.iter().all(|&v| v == 0x55));
}

/// Rows with a gap after each (NHWC-like), a padded and a transposed output, and row counts
/// that do not fill the NEON kernel's groups of 16: every tier equals its dense result.
#[test]
fn strided_views_match_contiguous() {
    let g = common::load_graph("dfl_head_i8");
    let p = params(&g);
    let x = g.i8("input");
    for rows in [1usize, 15, 16, 17, 37, 300] {
        let src = &x[..rows * 16];
        let mut gapped = vec![0x11i8; rows * 24];
        for r in 0..rows {
            gapped[r * 24..r * 24 + 16].copy_from_slice(&src[r * 16..(r + 1) * 16]);
        }
        for f in common::tiers() {
            let plan = SoftmaxDotI8Plan::with_features(p.clone(), f).unwrap();
            let dense = run(&plan, src, rows, 16);
            let mut padded = vec![0x55i8; rows * 3];
            plan.run(
                TensorRef::I8(View::strided(&gapped, &[rows, 16], &[24, 1]).unwrap()),
                TensorMut::I8(ViewMut::strided(&mut padded, &[rows, 1], &[3, 1]).unwrap()),
                None,
            )
            .unwrap();
            for r in 0..rows {
                assert_eq!(
                    padded[r * 3],
                    dense[r],
                    "{rows} rows, tier {:?}",
                    plan.tier()
                );
                assert_eq!(&padded[r * 3 + 1..r * 3 + 3], &[0x55, 0x55]);
            }
            // [rows/…, 1] as a rank-3 view whose two outer axes run in the other order.
            if rows % 4 == 0 {
                let (a, b) = (rows / 4, 4);
                let mut out_t = vec![0i8; rows];
                plan.run(
                    TensorRef::I8(View::contiguous(src, &[a, b, 16]).unwrap()),
                    TensorMut::I8(
                        ViewMut::strided(&mut out_t, &[a, b, 1], &[1, a as isize, 1]).unwrap(),
                    ),
                    None,
                )
                .unwrap();
                for i in 0..a {
                    for j in 0..b {
                        assert_eq!(out_t[j * a + i], dense[i * b + j]);
                    }
                }
            }
        }
    }
}

#[test]
fn zero_sized_views() {
    for f in common::tiers() {
        let plan = SoftmaxDotI8Plan::with_features(good(), f).unwrap();
        let empty: [i8; 0] = [];
        let mut guard = [0x55i8; 2];
        plan.run(
            TensorRef::I8(View::contiguous(&empty, &[0, 16]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut guard[..0], &[0, 1]).unwrap()),
            None,
        )
        .unwrap();
        plan.run(
            TensorRef::I8(View::contiguous(&empty, &[3, 0, 16]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut guard[..0], &[3, 0, 1]).unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(guard, [0x55; 2]);
    }
}

#[test]
fn pool_and_concurrent_runs_match_single_thread() {
    let g = common::load_graph("dfl_head_i8");
    let x = g.i8("input");
    let rows = x.len() / 16;
    for f in common::tiers() {
        let plan = SoftmaxDotI8Plan::with_features(params(&g), f).unwrap();
        assert_eq!(plan.tier() == Tier::Scalar, f.tier() == Tier::Scalar);
        common::assert_pool_and_concurrency_match(|pool| {
            let mut out = vec![0i8; rows];
            plan.run(
                TensorRef::I8(View::contiguous(&x, &[4, rows / 4, 16]).unwrap()),
                TensorMut::I8(ViewMut::contiguous(&mut out, &[4, rows / 4, 1]).unwrap()),
                pool,
            )
            .unwrap();
            out.iter().map(|&v| v as u8).collect()
        });
    }
}
