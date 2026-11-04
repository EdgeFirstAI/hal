// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! int8 `BATCH_MATMUL` and 1×1 `CONV_2D` against the TFLite fixtures, and the plan contract.

mod common;

use common::Graph;
use edgefirst_kernels::{
    Activation, CpuFeatures, KernelError, MatMulI8Params, MatMulI8Plan, Pool, Quant, Requant,
    TensorMut, TensorRef, TfliteKernels, Tier, View, ViewMut,
};

const BATCH_MATMUL: &[&str] = &[
    "batch_matmul_i8",
    "batch_matmul_i8_adj",
    "batch_matmul_i8_ref",
];
const CONV: &[&str] = &["conv1x1_i8", "conv1x1_i8_ref"];

/// The fixtures come from LiteRT's x86_64 wheel, whose optimized int8 matrix products run
/// through gemmlowp and requantize like its reference kernels (`MultiplyByQuantizedMultiplier`,
/// two roundings), not with the single rounding of ruy's aarch64 kernels.
const HOST: TfliteKernels = TfliteKernels::NeonGemmlowp;

fn activation(g: &Graph) -> Activation {
    match g.meta("activation") {
        "none" => Activation::None,
        "relu" => Activation::Relu,
        "relu_n1_to_1" => Activation::ReluN1To1,
        "relu6" => Activation::Relu6,
        other => panic!("{}: unexpected activation {other}", g.name),
    }
}

/// `[.., rows, cols]` row-major data as a view of `[.., cols, rows]` (its transpose).
fn transposed<'a>(data: &'a [i8], stored: &[usize]) -> View<'a, i8> {
    let r = stored.len();
    let (rows, cols) = (stored[r - 2], stored[r - 1]);
    let mut shape = stored.to_vec();
    shape.swap(r - 2, r - 1);
    let mut strides: Vec<isize> = vec![0; r];
    let mut step = (rows * cols) as isize;
    for i in (0..r - 2).rev() {
        strides[i] = step;
        step *= stored[i] as isize;
    }
    strides[r - 2] = 1;
    strides[r - 1] = cols as isize;
    View::strided(data, &shape, &strides).unwrap()
}

fn bmm_plan(g: &Graph, kernels: TfliteKernels, f: CpuFeatures) -> MatMulI8Plan {
    let p = MatMulI8Params::tflite_batch_matmul(
        g.quant("lhs"),
        g.quant("rhs"),
        g.quant("output"),
        kernels,
    )
    .unwrap();
    MatMulI8Plan::with_features(p, f).unwrap()
}

fn run_bmm(plan: &MatMulI8Plan, g: &Graph, pool: Option<&dyn Pool>) -> Vec<i8> {
    let (lhs, rhs) = (g.i8("lhs"), g.i8("rhs"));
    let (ls, rs) = (g.shape("lhs"), g.shape("rhs"));
    let l = if g.meta("adj_x") == "true" {
        transposed(&lhs, ls)
    } else {
        View::contiguous(&lhs, ls).unwrap()
    };
    let r = if g.meta("adj_y") == "true" {
        transposed(&rhs, rs)
    } else {
        View::contiguous(&rhs, rs).unwrap()
    };
    let mut shape = l.shape().to_vec();
    *shape.last_mut().unwrap() = *r.shape().last().unwrap();
    let mut out = vec![0x55u8 as i8; shape.iter().product()];
    plan.run(
        TensorRef::I8(l),
        TensorRef::I8(r),
        TensorMut::I8(ViewMut::contiguous(&mut out, &shape).unwrap()),
        pool,
    )
    .unwrap();
    out
}

fn conv_plan(g: &Graph, kernels: TfliteKernels, f: CpuFeatures) -> MatMulI8Plan {
    let p = MatMulI8Params::tflite_conv(
        g.quant("input"),
        &g.f32_list("filter_scales"),
        g.quant("output"),
        g.i32("bias"),
        activation(g),
        kernels,
    )
    .unwrap();
    MatMulI8Plan::with_features(p, f).unwrap()
}

/// The NHWC input as `[pixels, in]` times the `[out, in]` filter read as `[in, out]`.
fn run_conv(plan: &MatMulI8Plan, g: &Graph, pool: Option<&dyn Pool>) -> Vec<i8> {
    let (x, w) = (g.i8("input"), g.i8("filter"));
    let cin = *g.shape("input").last().unwrap();
    let cout = g.shape("filter")[0];
    let pixels = x.len() / cin;
    let mut out = vec![0x55u8 as i8; pixels * cout];
    plan.run(
        TensorRef::I8(View::contiguous(&x, &[pixels, cin]).unwrap()),
        TensorRef::I8(transposed(&w, &[cout, cin])),
        TensorMut::I8(ViewMut::contiguous(&mut out, &[pixels, cout]).unwrap()),
        pool,
    )
    .unwrap();
    out
}

#[test]
fn matmul_plans_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<MatMulI8Plan>();
    assert_send_sync::<MatMulI8Params>();
}

#[test]
fn batch_matmul_every_tier_matches_tflite_bit_exact() {
    for name in BATCH_MATMUL {
        let g = common::load_graph(name);
        let want = g.i8("output");
        for f in common::tiers() {
            let plan = bmm_plan(&g, HOST, f);
            let (bad, worst) = common::diff_i8(&run_bmm(&plan, &g, None), &want);
            assert_eq!(
                bad,
                0,
                "{name} tier {:?}: {bad} differ, worst {worst}",
                plan.tier()
            );
        }
    }
}

#[test]
fn conv_1x1_every_tier_matches_tflite_bit_exact() {
    for name in CONV {
        let g = common::load_graph(name);
        let want = g.i8("output");
        assert_eq!(activation(&g), Activation::Relu6);
        assert!(g.i32("bias").iter().any(|&b| b != 0));
        for f in common::tiers() {
            let plan = conv_plan(&g, HOST, f);
            let (bad, worst) = common::diff_i8(&run_conv(&plan, &g, None), &want);
            assert_eq!(
                bad,
                0,
                "{name} tier {:?}: {bad} differ, worst {worst}",
                plan.tier()
            );
        }
    }
}

/// The fixtures separate the two roundings: ruy's single rounding (`TfliteKernels::NeonRuy`)
/// gives other bytes, so the host's optimized products round like the reference kernels.
#[test]
fn fixtures_distinguish_ruy_rounding_from_the_reference() {
    for name in BATCH_MATMUL.iter().chain(CONV) {
        let g = common::load_graph(name);
        let f = CpuFeatures::default();
        let got = if name.starts_with("conv") {
            run_conv(&conv_plan(&g, TfliteKernels::NeonRuy, f), &g, None)
        } else {
            run_bmm(&bmm_plan(&g, TfliteKernels::NeonRuy, f), &g, None)
        };
        let (bad, worst) = common::diff_i8(&got, &g.i8("output"));
        eprintln!("{name}: ruy rounding differs in {bad} elements (worst {worst})");
        assert!(bad > 0 && worst == 1, "{name}: {bad} differ, worst {worst}");
    }
}

fn good() -> MatMulI8Params {
    MatMulI8Params::tflite_batch_matmul(
        Quant::new(0.05, -3),
        Quant::new(0.04, 7),
        Quant::new(0.5, 1),
        TfliteKernels::NeonRuy,
    )
    .unwrap()
}

#[test]
fn new_rejects_bad_params() {
    assert!(MatMulI8Plan::new(good()).is_ok());
    let invalid =
        |p: MatMulI8Params| matches!(MatMulI8Plan::new(p), Err(KernelError::InvalidArgument(_)));
    let unsupported =
        |p: MatMulI8Params| matches!(MatMulI8Plan::new(p), Err(KernelError::Unsupported(_)));
    for zp in [-129, 128] {
        assert!(invalid(MatMulI8Params {
            lhs_zero_point: zp,
            ..good()
        }));
        assert!(invalid(MatMulI8Params {
            rhs_zero_point: zp,
            ..good()
        }));
        assert!(invalid(MatMulI8Params {
            output: Quant::new(0.5, zp),
            ..good()
        }));
    }
    for s in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(invalid(MatMulI8Params {
            output: Quant::new(s, 0),
            ..good()
        }));
        assert!(Requant::from_real(f64::from(s)).is_err(), "{s}");
        assert!(Requant::tflite_batch_matmul(s, 0.1, 0.1).is_err(), "{s}");
        assert!(Requant::tflite_conv(0.1, 0.1, s).is_err(), "{s}");
    }
    assert!(invalid(MatMulI8Params {
        requant: vec![],
        ..good()
    }));
    for r in [
        Requant {
            multiplier: -1,
            shift: -3,
        },
        Requant {
            multiplier: 1 << 30,
            shift: 32,
        },
        Requant {
            multiplier: 1 << 30,
            shift: -32,
        },
    ] {
        assert!(
            invalid(MatMulI8Params {
                requant: vec![r],
                ..good()
            }),
            "{r:?}"
        );
    }
    // A multiplier of 0.5 or more (shift ≥ 0) is valid TFLite arithmetic, but ruy's NEON
    // kernels truncate there, so only the reference rounding takes it.
    let half = Requant::from_real(0.75).unwrap();
    assert_eq!(half.shift, 0);
    assert!(unsupported(MatMulI8Params {
        requant: vec![half],
        ..good()
    }));
    assert!(MatMulI8Plan::new(MatMulI8Params {
        requant: vec![half],
        kernels: TfliteKernels::Reference,
        ..good()
    })
    .is_ok());
    // Every fused activation leaves a non-empty clamp for any valid output quantization.
    for activation in [Activation::Relu, Activation::ReluN1To1, Activation::Relu6] {
        for output in [Quant::new(1.0e-6, 127), Quant::new(1.0e6, -128)] {
            assert!(MatMulI8Plan::new(MatMulI8Params {
                activation,
                output,
                ..good()
            })
            .is_ok());
        }
    }
    assert!(MatMulI8Plan::check_depth(4096).is_ok());
    assert!(matches!(
        MatMulI8Plan::check_depth(4097),
        Err(KernelError::Unsupported(_))
    ));
}

fn v<'a>(d: &'a [i8], s: &[usize]) -> TensorRef<'a> {
    TensorRef::I8(View::contiguous(&d[..s.iter().product()], s).unwrap())
}

#[test]
fn run_rejects_mismatch_without_writing() {
    let plan = MatMulI8Plan::new(good()).unwrap();
    let a = vec![1i8; 4 * 5 * 6];
    let b = vec![2i8; 4 * 6 * 7];
    let mut out = vec![0x55i8; 4 * 5 * 7 + 64];
    let untouched = |o: &[i8]| o.iter().all(|&x| x == 0x55);
    for (ls, rs, os) in [
        (&[5usize][..], &[5usize][..], &[5usize][..]),
        (&[5, 6], &[4, 6, 7], &[5, 7]),
        (&[4, 5, 6], &[4, 6, 7], &[5, 7]),
        (&[4, 5, 6], &[2, 6, 7], &[4, 5, 7]),
        (&[4, 5, 6], &[4, 6, 7], &[2, 5, 7]),
        (&[4, 5, 6], &[4, 5, 7], &[4, 5, 7]),
        (&[4, 5, 6], &[4, 6, 7], &[4, 6, 7]),
        (&[4, 5, 6], &[4, 6, 7], &[4, 5, 6]),
        (&[], &[], &[]),
    ] {
        let n: usize = os.iter().product();
        let r = plan.run(
            v(&a, ls),
            v(&b, rs),
            TensorMut::I8(ViewMut::contiguous(&mut out[..n], os).unwrap()),
            None,
        );
        assert!(
            matches!(r, Err(KernelError::InvalidArgument(_))),
            "{ls:?} {rs:?} {os:?}: {r:?}"
        );
        assert!(untouched(&out));
    }
    let r = plan.run(
        TensorRef::U8(View::contiguous(&[0u8; 30], &[5, 6]).unwrap()),
        v(&b, &[6, 7]),
        TensorMut::I8(ViewMut::contiguous(&mut out[..35], &[5, 7]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    let mut wide = vec![0x55i16; 35];
    let r = plan.run(
        v(&a, &[5, 6]),
        v(&b, &[6, 7]),
        TensorMut::I16(ViewMut::contiguous(&mut wide, &[5, 7]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(wide.iter().all(|&x| x == 0x55));
    // Per-column requantization and bias must have one entry per output column.
    for p in [
        MatMulI8Params {
            requant: vec![Requant::from_real(0.01).unwrap(); 3],
            ..good()
        },
        MatMulI8Params {
            bias: vec![1; 3],
            ..good()
        },
    ] {
        let plan = MatMulI8Plan::new(p).unwrap();
        let r = plan.run(
            v(&a, &[5, 6]),
            v(&b, &[6, 7]),
            TensorMut::I8(ViewMut::contiguous(&mut out[..35], &[5, 7]).unwrap()),
            None,
        );
        assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
        assert!(untouched(&out));
    }
    // A reduction axis past 4096 is unsupported, before anything is written.
    let k = 4097;
    let (long_a, long_b) = (vec![1i8; 2 * k], vec![1i8; k * 2]);
    let r = plan.run(
        v(&long_a, &[2, k]),
        v(&long_b, &[k, 2]),
        TensorMut::I8(ViewMut::contiguous(&mut out[..4], &[2, 2]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::Unsupported(_))), "{r:?}");
    assert!(untouched(&out));
}

#[test]
fn run_rejects_aliasing_output_rows() {
    let plan = MatMulI8Plan::new(good()).unwrap();
    let (a, b) = (vec![1i8; 4 * 3], vec![1i8; 3 * 8]);
    let mut out = vec![0x55i8; 32];
    for strides in [[0isize, 1], [4, 1], [8, 0], [1, 1]] {
        let r = plan.run(
            v(&a, &[4, 3]),
            v(&b, &[3, 8]),
            TensorMut::I8(ViewMut::strided(&mut out, &[4, 8], &strides).unwrap()),
            None,
        );
        assert!(
            matches!(r, Err(KernelError::InvalidArgument(_))),
            "{strides:?}: {r:?}"
        );
        assert!(out.iter().all(|&x| x == 0x55));
    }
}

/// `Σ_k (l − zl)(r − zr) + bias`, requantized in f64 and rounded half up: within one of the
/// plan, which must equal itself on every tier and layout.
fn definition(p: &MatMulI8Params, l: &[i8], r: &[i8], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut o = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i64;
            for d in 0..k {
                acc += (i64::from(l[i * k + d]) - i64::from(p.lhs_zero_point))
                    * (i64::from(r[d * n + j]) - i64::from(p.rhs_zero_point));
            }
            acc += i64::from(p.bias.get(j).copied().unwrap_or(0));
            let rq = if p.requant.len() == 1 {
                p.requant[0]
            } else {
                p.requant[j]
            };
            let real = f64::from(rq.multiplier) * 2f64.powi(rq.shift - 31);
            o[i * n + j] =
                (f64::from(p.output.zero_point) + acc as f64 * real).clamp(-128.0, 127.0);
        }
    }
    o
}

/// Transposed operands, a padded output, a broadcast batch, per-column requantization with a
/// bias, `n` below the tile width and `k` past the packing width give the dense result on
/// every tier, and output padding is never written.
#[test]
fn strided_views_match_contiguous() {
    let rng = |seed: u32, n: usize| -> Vec<i8> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                (s >> 8) as u8 as i8
            })
            .collect()
    };
    for (m, k, n) in [
        (13usize, 37usize, 3usize),
        (9, 300, 21),
        (4, 4097 - 1, 5),
        (1, 1, 1),
    ] {
        let (l, r) = (rng(3, m * k), rng(9, k * n));
        let per_col = MatMulI8Params {
            requant: (0..n)
                .map(|j| Requant::from_real(0.002 + 0.0003 * j as f64).unwrap())
                .collect(),
            bias: (0..n).map(|j| j as i32 * 37 - 200).collect(),
            activation: Activation::Relu,
            output: Quant::new(0.05, -20),
            ..good()
        };
        for p in [good(), per_col] {
            let want_f = definition(&p, &l, &r, m, k, n);
            let (lt, rt) = {
                let mut lt = vec![0i8; m * k];
                let mut rt = vec![0i8; k * n];
                for i in 0..m {
                    for d in 0..k {
                        lt[d * m + i] = l[i * k + d];
                    }
                }
                for d in 0..k {
                    for j in 0..n {
                        rt[j * k + d] = r[d * n + j];
                    }
                }
                (lt, rt)
            };
            let mut reference: Option<Vec<i8>> = None;
            for f in common::tiers() {
                let plan = MatMulI8Plan::with_features(p.clone(), f).unwrap();
                let mut dense = vec![0i8; m * n];
                plan.run(
                    v(&l, &[m, k]),
                    v(&r, &[k, n]),
                    TensorMut::I8(ViewMut::contiguous(&mut dense, &[m, n]).unwrap()),
                    None,
                )
                .unwrap();
                for (j, (&o, &w)) in dense.iter().zip(&want_f).enumerate() {
                    let lo = if p.activation == Activation::Relu {
                        -20.0
                    } else {
                        -128.0
                    };
                    assert!(
                        (f64::from(o) - w.max(lo)).abs() <= 1.0,
                        "({m},{k},{n}) {j}: {o} vs {w}"
                    );
                }
                match &reference {
                    Some(r) => assert_eq!(&dense, r, "tier {:?}", plan.tier()),
                    None => reference = Some(dense.clone()),
                }
                let pitch = n + 3;
                let mut padded = vec![0x55i8; m * pitch];
                plan.run(
                    TensorRef::I8(View::strided(&lt, &[m, k], &[1, m as isize]).unwrap()),
                    TensorRef::I8(View::strided(&rt, &[k, n], &[1, k as isize]).unwrap()),
                    TensorMut::I8(
                        ViewMut::strided(&mut padded, &[m, n], &[pitch as isize, 1]).unwrap(),
                    ),
                    None,
                )
                .unwrap();
                for i in 0..m {
                    assert_eq!(
                        &padded[i * pitch..i * pitch + n],
                        &dense[i * n..(i + 1) * n],
                        "({m},{k},{n}) tier {:?} row {i}",
                        plan.tier()
                    );
                    assert!(padded[i * pitch + n..(i + 1) * pitch]
                        .iter()
                        .all(|&x| x == 0x55));
                }
                // Three batches sharing one right-hand side through a zero batch stride, the
                // output transposed.
                let l3: Vec<i8> = l.iter().chain(&l).chain(&l).copied().collect();
                let mut out3 = vec![0i8; 3 * m * n];
                plan.run(
                    v(&l3, &[3, m, k]),
                    TensorRef::I8(View::strided(&r, &[3, k, n], &[0, n as isize, 1]).unwrap()),
                    TensorMut::I8(
                        ViewMut::strided(&mut out3, &[3, m, n], &[(m * n) as isize, 1, m as isize])
                            .unwrap(),
                    ),
                    None,
                )
                .unwrap();
                for b in 0..3 {
                    for i in 0..m {
                        for j in 0..n {
                            assert_eq!(out3[b * m * n + j * m + i], dense[i * n + j]);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn zero_sized_views() {
    for f in common::tiers() {
        let plan = MatMulI8Plan::with_features(good(), f).unwrap();
        let empty: [i8; 0] = [];
        let ones = [1i8; 128];
        let mut guard = [0x55i8; 8];
        for (ls, rs, os) in [
            ([0usize, 5, 6], [0usize, 6, 7], [0usize, 5, 7]),
            ([2, 0, 6], [2, 6, 7], [2, 0, 7]),
            ([2, 5, 6], [2, 6, 0], [2, 5, 0]),
        ] {
            let lhs: &[i8] = if ls.contains(&0) { &empty } else { &ones[..60] };
            let rhs: &[i8] = if rs.contains(&0) {
                &empty
            } else {
                &ones[..2 * 6 * 7]
            };
            let r = plan.run(
                TensorRef::I8(View::contiguous(lhs, &ls).unwrap()),
                TensorRef::I8(View::contiguous(rhs, &rs).unwrap()),
                TensorMut::I8(ViewMut::contiguous(&mut guard[..0], &os).unwrap()),
                None,
            );
            assert!(r.is_ok(), "{ls:?} {rs:?}: {r:?}");
        }
        assert_eq!(guard, [0x55; 8]);
        // An empty reduction axis: every output is the requantized bias.
        let p = MatMulI8Params {
            bias: vec![1000, -1000, 0],
            ..good()
        };
        let plan = MatMulI8Plan::with_features(p.clone(), f).unwrap();
        let mut out = [0x55i8; 6];
        plan.run(
            TensorRef::I8(View::contiguous(&empty, &[2, 0]).unwrap()),
            TensorRef::I8(View::contiguous(&empty, &[0, 3]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut out, &[2, 3]).unwrap()),
            None,
        )
        .unwrap();
        let want = definition(&p, &[], &[], 2, 0, 3);
        for (&o, &w) in out.iter().zip(&want) {
            assert!((f64::from(o) - w).abs() <= 0.5 + 1e-9, "{o} vs {w}");
        }
    }
}

#[test]
fn pool_and_concurrent_runs_match_single_thread() {
    let (b, m, k, n) = (3usize, 37usize, 300usize, 29usize);
    let l: Vec<i8> = (0..b * m * k)
        .map(|i| (i * 7919 % 256) as u8 as i8)
        .collect();
    let r: Vec<i8> = (0..b * k * n)
        .map(|i| (i * 104_729 % 253) as u8 as i8)
        .collect();
    for kernels in [TfliteKernels::NeonRuy, TfliteKernels::Reference] {
        for f in common::tiers() {
            let plan =
                MatMulI8Plan::with_features(MatMulI8Params { kernels, ..good() }, f).unwrap();
            assert_eq!(plan.tier() == Tier::Scalar, f.tier() == Tier::Scalar);
            for nn in [n, 2] {
                common::assert_pool_and_concurrency_match(|pool| {
                    let mut out = vec![0i8; b * m * nn];
                    plan.run(
                        TensorRef::I8(View::contiguous(&l, &[b, m, k]).unwrap()),
                        TensorRef::I8(
                            View::strided(&r, &[b, k, nn], &[(k * n) as isize, n as isize, 1])
                                .unwrap(),
                        ),
                        TensorMut::I8(ViewMut::contiguous(&mut out, &[b, m, nn]).unwrap()),
                        pool,
                    )
                    .unwrap();
                    out.iter().map(|&x| x as u8).collect()
                });
            }
        }
    }
}
