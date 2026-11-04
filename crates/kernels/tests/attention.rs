// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Fused int8 attention against the TFLite `MUL` → `BATCH_MATMUL` → `SOFTMAX` →
//! `BATCH_MATMUL` fixtures and the crate's own chain of single-op plans, its limits, scratch
//! and views, and the plan contract.

mod common;

use common::Graph;
use edgefirst_kernels::{
    Activation, AttentionI8Params, AttentionI8Plan, AttentionQMul, AttentionScratch,
    BinaryI8Params, BinaryI8Plan, BinaryOp, CpuFeatures, KernelError, MatMulI8Params, MatMulI8Plan,
    Pool, Quant, SoftmaxI8Params, SoftmaxI8Plan, TensorMut, TensorRef, TfliteKernels, Tier, View,
    ViewMut,
};

const FIXTURES: &[&str] = &["attention_i8_yolo26", "attention_i8_odd"];

const PROBS: Quant = Quant {
    scale: 1.0 / 256.0,
    zero_point: -128,
};

/// Heads, queries, key/value tokens, and the query/key and value dims.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Shape {
    h: usize,
    n: usize,
    m: usize,
    dk: usize,
    dv: usize,
}

/// The fixture's chain as plan parameters, for the host LiteRT: `TfliteKernels::NeonGemmlowp`
/// (see `tests/binary.rs` and `tests/matmul.rs`).
fn params(g: &Graph) -> AttentionI8Params {
    let q = g.quant("q");
    let mul = BinaryI8Plan::new(BinaryI8Params {
        op: BinaryOp::Mul,
        lhs: q,
        rhs: g.quant("mul_rhs"),
        output: g.quant("q_scaled"),
        activation: Activation::None,
        kernels: TfliteKernels::NeonGemmlowp,
    })
    .unwrap();
    let rhs: i8 = g.meta("mul_rhs_value").parse().unwrap();
    let table = mul
        .table_for_rhs(rhs, g.i8("q").len())
        .expect("one table for the whole MUL");
    AttentionI8Params {
        q,
        q_mul: Some(AttentionQMul {
            table,
            output: g.quant("q_scaled"),
        }),
        k: g.quant("k"),
        v: g.quant("v"),
        scores: g.quant("scores"),
        beta: g.meta("beta").parse().unwrap(),
        probs: g.quant("probs"),
        output: g.quant("output"),
        kernels: TfliteKernels::NeonGemmlowp,
    }
}

fn shape_of(g: &Graph) -> Shape {
    let (q, v) = (g.shape("q"), g.shape("v"));
    Shape {
        h: q[1],
        n: q[2],
        m: g.shape("k")[2],
        dk: q[3],
        dv: v[3],
    }
}

/// The crate's single-op plans one after the other, on the tier `f`: the table, `q'·kᵀ`, the
/// softmax and `p·v`, over contiguous `[h, ·, ·]` operands.
fn chain(p: &AttentionI8Params, f: CpuFeatures, s: Shape, q: &[i8], k: &[i8], v: &[i8]) -> Vec<i8> {
    let Shape { h, n, m, dk, dv } = s;
    let q2: Vec<i8> = match p.q_mul {
        Some(t) => q
            .iter()
            .map(|&x| t.table[usize::from(x as u8)] as i8)
            .collect(),
        None => q.to_vec(),
    };
    let qs = p.q_mul.map_or(p.q, |t| t.output);
    let bmm1 = MatMulI8Plan::with_features(
        MatMulI8Params::tflite_batch_matmul(qs, p.k, p.scores, p.kernels).unwrap(),
        f,
    )
    .unwrap();
    let mut scores = vec![0i8; h * n * m];
    bmm1.run(
        TensorRef::I8(View::contiguous(&q2, &[h, n, dk]).unwrap()),
        TensorRef::I8(View::strided(k, &[h, dk, m], &[(m * dk) as isize, 1, dk as isize]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut scores, &[h, n, m]).unwrap()),
        None,
    )
    .unwrap();
    let sm = SoftmaxI8Plan::with_features(
        SoftmaxI8Params {
            input: p.scores,
            output: p.probs,
            beta: p.beta,
        },
        f,
    )
    .unwrap();
    let mut probs = vec![0i8; h * n * m];
    sm.run(
        TensorRef::I8(View::contiguous(&scores, &[h, n, m]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut probs, &[h, n, m]).unwrap()),
        None,
    )
    .unwrap();
    let bmm2 = MatMulI8Plan::with_features(
        MatMulI8Params::tflite_batch_matmul(p.probs, p.v, p.output, p.kernels).unwrap(),
        f,
    )
    .unwrap();
    let mut out = vec![0i8; h * n * dv];
    bmm2.run(
        TensorRef::I8(View::contiguous(&probs, &[h, n, m]).unwrap()),
        TensorRef::I8(View::contiguous(v, &[h, m, dv]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut out, &[h, n, dv]).unwrap()),
        None,
    )
    .unwrap();
    out
}

/// The fused plan over contiguous `[h, ·, ·]` operands, with a fresh scratch.
fn fused(
    plan: &AttentionI8Plan,
    s: Shape,
    q: &[i8],
    k: &[i8],
    v: &[i8],
    pool: Option<&dyn Pool>,
) -> Vec<i8> {
    let mut scratch = plan.scratch(s.h, s.n, s.m, s.dk, s.dv).unwrap();
    let mut out = vec![0x55u8 as i8; s.h * s.n * s.dv];
    plan.run(
        TensorRef::I8(View::contiguous(q, &[s.h, s.n, s.dk]).unwrap()),
        TensorRef::I8(View::contiguous(k, &[s.h, s.m, s.dk]).unwrap()),
        TensorRef::I8(View::contiguous(v, &[s.h, s.m, s.dv]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut out, &[s.h, s.n, s.dv]).unwrap()),
        &mut scratch,
        pool,
    )
    .unwrap();
    out
}

/// Seeded int8 values spread like a layer output: mostly small, some large.
fn layer(seed: u32, len: usize, spread: u32) -> Vec<i8> {
    let mut s = seed.max(1);
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let v = (s % (2 * spread + 1)) as i32 - spread as i32
                + ((s >> 8) % 3) as i32 * ((s >> 12) % 40) as i32
                - 20;
            v.clamp(-128, 127) as i8
        })
        .collect()
}

#[test]
fn attention_plans_and_scratch_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AttentionI8Plan>();
    assert_send_sync::<AttentionI8Params>();
    assert_send_sync::<AttentionScratch>();
}

/// The scalar tier gives TFLite's bytes; the NEON tiers give the crate's chain on that tier
/// bit for bit, whose only departure from TFLite is the NEON softmax's ±1 LSB, and they stay
/// within one LSB of TFLite on these fixtures.
#[test]
fn fixtures_match_tflite_and_the_chain() {
    for name in FIXTURES {
        let g = common::load_graph(name);
        let p = params(&g);
        let s = shape_of(&g);
        let (q, k, v) = (g.i8("q"), g.i8("k"), g.i8("v"));
        let want = g.i8("output");
        for f in common::tiers() {
            let plan = AttentionI8Plan::with_features(p, f).unwrap();
            let got = fused(&plan, s, &q, &k, &v, None);
            assert_eq!(got, chain(&p, f, s, &q, &k, &v), "{name} {:?}", plan.tier());
            let (bad, worst) = common::diff_i8(&got, &want);
            eprintln!(
                "attention tier={} fixture={name}: {bad} of {} differ from TFLite (worst {worst})",
                plan.tier().as_str(),
                want.len()
            );
            if plan.tier() == Tier::Scalar {
                assert_eq!(bad, 0, "{name}: the scalar tier is TFLite-exact");
            } else {
                assert!(worst <= 1, "{name} {:?}: worst {worst}", plan.tier());
            }
        }
    }
}

/// The largest shapes the kernels take, and one less, on every tier: the fused plan equals the
/// chain, so the stack buffers and scratch hold every row and vector.
#[test]
fn limits_equal_the_chain() {
    let p = AttentionI8Params {
        q: Quant::new(0.05, 3),
        q_mul: None,
        k: Quant::new(0.05, -2),
        v: Quant::new(0.08, 1),
        scores: Quant::new(0.4, -5),
        beta: 1.0,
        probs: PROBS,
        output: Quant::new(0.06, 0),
        kernels: TfliteKernels::NeonRuy,
    };
    for s in [
        Shape {
            h: 2,
            n: 5,
            m: 2048,
            dk: 256,
            dv: 256,
        },
        Shape {
            h: 1,
            n: 6,
            m: 2047,
            dk: 255,
            dv: 255,
        },
        Shape {
            h: 2,
            n: 9,
            m: 2048,
            dk: 1,
            dv: 1,
        },
        Shape {
            h: 1,
            n: 3,
            m: 1,
            dk: 256,
            dv: 255,
        },
        Shape {
            h: 1,
            n: 4,
            m: 33,
            dk: 0,
            dv: 7,
        },
    ] {
        AttentionI8Plan::check_shape(s.m, s.dk, s.dv).unwrap();
        let q = layer(1, s.h * s.n * s.dk, 6);
        let k = layer(2, s.h * s.m * s.dk, 6);
        let v = layer(3, s.h * s.m * s.dv, 40);
        for f in common::tiers() {
            let plan = AttentionI8Plan::with_features(p, f).unwrap();
            assert_eq!(
                fused(&plan, s, &q, &k, &v, None),
                chain(&p, f, s, &q, &k, &v),
                "{s:?} {:?}",
                plan.tier()
            );
        }
    }
    let plan = AttentionI8Plan::new(p).unwrap();
    for (m, dk, dv) in [(2049, 8, 8), (8, 257, 8), (8, 8, 257), (4096, 512, 512)] {
        assert!(matches!(
            AttentionI8Plan::check_shape(m, dk, dv),
            Err(KernelError::Unsupported(_))
        ));
        assert!(matches!(
            plan.scratch(1, 4, m, dk, dv),
            Err(KernelError::Unsupported(_))
        ));
    }
    assert!(plan.scratch(usize::MAX, usize::MAX, 8, 8, 8).is_err());
}

/// Head slices of one token-major `[tokens, heads · (2·dk + dv)]` tensor, as a TFLite graph
/// holds Q, K and V, into a token-major output with a gap after each token: every tier
/// equals its run on contiguous copies.
#[test]
fn strided_views_match_contiguous() {
    let g = common::load_graph("attention_i8_odd");
    let p = params(&g);
    for s in [
        Shape {
            h: 3,
            n: 37,
            m: 37,
            dk: 12,
            dv: 9,
        },
        Shape {
            h: 2,
            n: 41,
            m: 41,
            dk: 32,
            dv: 64,
        },
    ] {
        let c = 2 * s.dk + s.dv;
        let x = layer(9, s.n * s.h * c, 30);
        let gather = |off: usize, d: usize| -> Vec<i8> {
            let mut o = Vec::with_capacity(s.h * s.n * d);
            for hh in 0..s.h {
                for i in 0..s.n {
                    let at = i * s.h * c + hh * c + off;
                    o.extend_from_slice(&x[at..at + d]);
                }
            }
            o
        };
        let (q, k, v) = (gather(0, s.dk), gather(s.dk, s.dk), gather(2 * s.dk, s.dv));
        let st = [c as isize, (s.h * c) as isize, 1];
        let pitch = s.h * s.dv + 3;
        for f in common::tiers() {
            let plan = AttentionI8Plan::with_features(p, f).unwrap();
            let dense = fused(&plan, s, &q, &k, &v, None);
            let mut out = vec![0x55i8; s.n * pitch];
            let mut scratch = plan.scratch(s.h, s.n, s.m, s.dk, s.dv).unwrap();
            plan.run(
                TensorRef::I8(View::strided(&x, &[s.h, s.n, s.dk], &st).unwrap()),
                TensorRef::I8(View::strided(&x[s.dk..], &[s.h, s.m, s.dk], &st).unwrap()),
                TensorRef::I8(View::strided(&x[2 * s.dk..], &[s.h, s.m, s.dv], &st).unwrap()),
                TensorMut::I8(
                    ViewMut::strided(
                        &mut out,
                        &[s.h, s.n, s.dv],
                        &[s.dv as isize, pitch as isize, 1],
                    )
                    .unwrap(),
                ),
                &mut scratch,
                None,
            )
            .unwrap();
            for hh in 0..s.h {
                for i in 0..s.n {
                    let at = i * pitch + hh * s.dv;
                    let d = (hh * s.n + i) * s.dv;
                    assert_eq!(
                        &out[at..at + s.dv],
                        &dense[d..d + s.dv],
                        "{s:?} {:?} head {hh} token {i}",
                        plan.tier()
                    );
                }
            }
            for i in 0..s.n {
                assert!(out[i * pitch + s.h * s.dv..(i + 1) * pitch]
                    .iter()
                    .all(|&b| b == 0x55));
            }
            // Transposed K and V (dim-major): the same result again.
            let kt: Vec<i8> = (0..s.h)
                .flat_map(|hh| {
                    let k = &k;
                    (0..s.dk).flat_map(move |d| (0..s.m).map(move |j| k[(hh * s.m + j) * s.dk + d]))
                })
                .collect();
            let mut out2 = vec![0i8; s.h * s.n * s.dv];
            plan.run(
                TensorRef::I8(View::contiguous(&q, &[s.h, s.n, s.dk]).unwrap()),
                TensorRef::I8(
                    View::strided(
                        &kt,
                        &[s.h, s.m, s.dk],
                        &[(s.m * s.dk) as isize, 1, s.m as isize],
                    )
                    .unwrap(),
                ),
                TensorRef::I8(View::contiguous(&v, &[s.h, s.m, s.dv]).unwrap()),
                TensorMut::I8(ViewMut::contiguous(&mut out2, &[s.h, s.n, s.dv]).unwrap()),
                &mut scratch,
                None,
            )
            .unwrap();
            assert_eq!(out2, dense, "{s:?} {:?} transposed keys", plan.tier());
        }
    }
}

/// One scratch serves many calls; a scratch built for other shapes, or by a plan whose kernel
/// packs another layout, is rejected before anything is written.
#[test]
fn scratch_is_reused_and_mismatches_are_rejected() {
    let g = common::load_graph("attention_i8_odd");
    let p = params(&g);
    let s = shape_of(&g);
    let (q, k, v) = (g.i8("q"), g.i8("k"), g.i8("v"));
    let (q2, k2, v2) = (
        layer(4, q.len(), 30),
        layer(5, k.len(), 30),
        layer(6, v.len(), 30),
    );
    let tiers = common::tiers();
    for &f in &tiers {
        let plan = AttentionI8Plan::with_features(p, f).unwrap();
        let mut scratch = plan.scratch(s.h, s.n, s.m, s.dk, s.dv).unwrap();
        for (q, k, v) in [(&q, &k, &v), (&q2, &k2, &v2), (&q, &k, &v)] {
            let mut out = vec![0i8; s.h * s.n * s.dv];
            plan.run(
                TensorRef::I8(View::contiguous(q, &[s.h, s.n, s.dk]).unwrap()),
                TensorRef::I8(View::contiguous(k, &[s.h, s.m, s.dk]).unwrap()),
                TensorRef::I8(View::contiguous(v, &[s.h, s.m, s.dv]).unwrap()),
                TensorMut::I8(ViewMut::contiguous(&mut out, &[s.h, s.n, s.dv]).unwrap()),
                &mut scratch,
                None,
            )
            .unwrap();
            assert_eq!(
                out,
                fused(&plan, s, q, k, v, None),
                "{:?} reused",
                plan.tier()
            );
        }
        let mut out = vec![0x55i8; s.h * s.n * s.dv];
        for (h, n, m, dk, dv) in [
            (s.h + 1, s.n, s.m, s.dk, s.dv),
            (s.h, s.n - 1, s.m, s.dk, s.dv),
            (s.h, s.n, s.m - 1, s.dk, s.dv),
            (s.h, s.n, s.m, s.dk + 1, s.dv),
            (s.h, s.n, s.m, s.dk, s.dv + 1),
        ] {
            let mut wrong = plan.scratch(h, n, m, dk, dv).unwrap();
            let r = plan.run(
                TensorRef::I8(View::contiguous(&q, &[s.h, s.n, s.dk]).unwrap()),
                TensorRef::I8(View::contiguous(&k, &[s.h, s.m, s.dk]).unwrap()),
                TensorRef::I8(View::contiguous(&v, &[s.h, s.m, s.dv]).unwrap()),
                TensorMut::I8(ViewMut::contiguous(&mut out, &[s.h, s.n, s.dv]).unwrap()),
                &mut wrong,
                None,
            );
            assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
            assert!(out.iter().all(|&b| b == 0x55));
        }
        // Another tier's scratch: accepted when its kernel packs the same layout.
        for &other in &tiers {
            let owner = AttentionI8Plan::with_features(p, other).unwrap();
            let mut foreign = owner.scratch(s.h, s.n, s.m, s.dk, s.dv).unwrap();
            let mut out = vec![0x55i8; s.h * s.n * s.dv];
            let r = plan.run(
                TensorRef::I8(View::contiguous(&q, &[s.h, s.n, s.dk]).unwrap()),
                TensorRef::I8(View::contiguous(&k, &[s.h, s.m, s.dk]).unwrap()),
                TensorRef::I8(View::contiguous(&v, &[s.h, s.m, s.dv]).unwrap()),
                TensorMut::I8(ViewMut::contiguous(&mut out, &[s.h, s.n, s.dv]).unwrap()),
                &mut foreign,
                None,
            );
            match r {
                Ok(()) => assert_eq!(out, fused(&plan, s, &q, &k, &v, None)),
                Err(e) => {
                    assert!(matches!(e, KernelError::InvalidArgument(_)), "{e:?}");
                    assert!(out.iter().all(|&b| b == 0x55));
                    // Only a NEON tier on a dotprod CPU packs a layout of its own.
                    assert!(
                        [plan.tier(), owner.tier()].contains(&Tier::Neon),
                        "{:?} on {:?}",
                        plan.tier(),
                        owner.tier()
                    );
                }
            }
        }
    }
}

fn good() -> AttentionI8Params {
    let g = common::load_graph("attention_i8_odd");
    params(&g)
}

#[test]
fn new_rejects_bad_params() {
    let p = good();
    assert!(AttentionI8Plan::new(p).is_ok());
    let invalid = |q: AttentionI8Params| {
        matches!(
            AttentionI8Plan::new(q),
            Err(KernelError::InvalidArgument(_))
        )
    };
    let unsupported =
        |q: AttentionI8Params| matches!(AttentionI8Plan::new(q), Err(KernelError::Unsupported(_)));
    let qm = p.q_mul.unwrap();
    for bad in [
        Quant::new(0.0, 0),
        Quant::new(-1.0, 0),
        Quant::new(f32::NAN, 0),
        Quant::new(f32::INFINITY, 0),
        Quant::new(0.1, 128),
        Quant::new(0.1, -129),
    ] {
        for q in [
            AttentionI8Params { q: bad, ..p },
            AttentionI8Params { k: bad, ..p },
            AttentionI8Params { v: bad, ..p },
            AttentionI8Params { scores: bad, ..p },
            AttentionI8Params { output: bad, ..p },
            AttentionI8Params {
                q_mul: Some(AttentionQMul { output: bad, ..qm }),
                ..p
            },
        ] {
            assert!(invalid(q), "{bad:?}");
        }
    }
    for beta in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(invalid(AttentionI8Params { beta, ..p }), "beta {beta}");
    }
    // The probabilities must be TFLite's int8 softmax output, (1/256, −128).
    assert!(unsupported(AttentionI8Params {
        probs: Quant::new(1.0 / 256.0, 0),
        ..p
    }));
    // A scores multiplier of 0.5 or more takes only the reference rounding.
    let wide = AttentionI8Params {
        scores: Quant::new(1.0e-5, 0),
        ..p
    };
    assert!(unsupported(AttentionI8Params {
        kernels: TfliteKernels::NeonRuy,
        ..wide
    }));
    assert!(AttentionI8Plan::new(AttentionI8Params {
        kernels: TfliteKernels::Reference,
        ..wide
    })
    .is_ok());
}

#[test]
fn run_rejects_mismatch_without_writing() {
    let plan = AttentionI8Plan::new(good()).unwrap();
    let s = Shape {
        h: 2,
        n: 5,
        m: 7,
        dk: 8,
        dv: 4,
    };
    let x = vec![1i8; 1024];
    let mut out = vec![0x55i8; 256];
    let view =
        |sh: &[usize]| TensorRef::I8(View::contiguous(&x[..sh.iter().product()], sh).unwrap());
    let good_q = [s.h, s.n, s.dk];
    let good_k = [s.h, s.m, s.dk];
    let good_v = [s.h, s.m, s.dv];
    let good_o = [s.h, s.n, s.dv];
    let mut scratch = plan.scratch(s.h, s.n, s.m, s.dk, s.dv).unwrap();
    let cases: [[&[usize]; 4]; 8] = [
        [&[s.n, s.dk], &good_k, &good_v, &good_o],
        [&good_q, &[1, s.h, s.m, s.dk], &good_v, &good_o],
        [&good_q, &[s.h + 1, s.m, s.dk], &good_v, &good_o],
        [&good_q, &[s.h, s.m, s.dk + 1], &good_v, &good_o],
        [&good_q, &good_k, &[s.h, s.m + 1, s.dv], &good_o],
        [&good_q, &good_k, &good_v, &[s.h, s.n + 1, s.dv]],
        [&good_q, &good_k, &good_v, &[s.h, s.n, s.dv + 1]],
        [&good_q, &good_k, &[s.h, s.m, s.dv + 1], &good_o],
    ];
    for [qs, ks, vs, os] in cases {
        let n: usize = os.iter().product();
        let r = plan.run(
            view(qs),
            view(ks),
            view(vs),
            TensorMut::I8(ViewMut::contiguous(&mut out[..n], os).unwrap()),
            &mut scratch,
            None,
        );
        assert!(
            matches!(r, Err(KernelError::InvalidArgument(_))),
            "{qs:?} {ks:?} {vs:?} {os:?}: {r:?}"
        );
        assert!(out.iter().all(|&b| b == 0x55));
    }
    let wide = vec![0i16; 80];
    let r = plan.run(
        TensorRef::I16(View::contiguous(&wide, &good_q).unwrap()),
        view(&good_k),
        view(&good_v),
        TensorMut::I8(ViewMut::contiguous(&mut out[..40], &good_o).unwrap()),
        &mut scratch,
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    let mut wide_out = vec![0x55i16; 40];
    let r = plan.run(
        view(&good_q),
        view(&good_k),
        view(&good_v),
        TensorMut::I16(ViewMut::contiguous(&mut wide_out, &good_o).unwrap()),
        &mut scratch,
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(wide_out.iter().all(|&b| b == 0x55));
    // An output whose heads land on the same bytes.
    let r = plan.run(
        view(&good_q),
        view(&good_k),
        view(&good_v),
        TensorMut::I8(ViewMut::strided(&mut out[..20], &good_o, &[0, s.dv as isize, 1]).unwrap()),
        &mut scratch,
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(out.iter().all(|&b| b == 0x55));
}

#[test]
fn zero_sized_views_are_a_no_op() {
    let p = good();
    for f in common::tiers() {
        let plan = AttentionI8Plan::with_features(p, f).unwrap();
        let x = [1i8; 64];
        let mut guard = [0x55i8; 16];
        for s in [
            Shape {
                h: 0,
                n: 4,
                m: 4,
                dk: 4,
                dv: 4,
            },
            Shape {
                h: 2,
                n: 0,
                m: 4,
                dk: 4,
                dv: 4,
            },
            Shape {
                h: 2,
                n: 2,
                m: 0,
                dk: 4,
                dv: 4,
            },
            Shape {
                h: 2,
                n: 2,
                m: 4,
                dk: 4,
                dv: 0,
            },
        ] {
            let mut scratch = plan.scratch(s.h, s.n, s.m, s.dk, s.dv).unwrap();
            let len = |a: usize, b: usize, c: usize| a * b * c;
            let olen = len(s.h, s.n, s.dv);
            let r = plan.run(
                TensorRef::I8(
                    View::contiguous(&x[..len(s.h, s.n, s.dk)], &[s.h, s.n, s.dk]).unwrap(),
                ),
                TensorRef::I8(
                    View::contiguous(&x[..len(s.h, s.m, s.dk)], &[s.h, s.m, s.dk]).unwrap(),
                ),
                TensorRef::I8(
                    View::contiguous(&x[..len(s.h, s.m, s.dv)], &[s.h, s.m, s.dv]).unwrap(),
                ),
                TensorMut::I8(ViewMut::contiguous(&mut guard[..olen], &[s.h, s.n, s.dv]).unwrap()),
                &mut scratch,
                None,
            );
            assert!(r.is_ok(), "{s:?}: {r:?}");
            assert!(guard.iter().all(|&b| b == 0x55), "{s:?} wrote");
        }
    }
}

#[test]
fn pool_and_concurrent_runs_match_single_thread() {
    let g = common::load_graph("attention_i8_yolo26");
    let p = params(&g);
    let s = shape_of(&g);
    let (q, k, v) = (g.i8("q"), g.i8("k"), g.i8("v"));
    for f in common::tiers() {
        let plan = AttentionI8Plan::with_features(p, f).unwrap();
        assert_eq!(plan.tier() == Tier::Scalar, f.tier() == Tier::Scalar);
        common::assert_pool_and_concurrency_match(|pool| {
            fused(&plan, s, &q, &k, &v, pool)
                .iter()
                .map(|&b| b as u8)
                .collect()
        });
    }
}
