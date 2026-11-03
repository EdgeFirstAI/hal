// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

mod common;

const FIXTURES: &[(&str, usize)] = &[
    ("softmax_i8_yolo11_attn", 1),
    ("softmax_i8_yolo26_attn", 1),
    ("softmax_i8_odd_rows_1009", 1),
    ("softmax_i8_odd_rows_17", 1),
    ("softmax_i8_odd_rows_1", 1),
    ("softmax_i8_dfl", 1),
    ("softmax_i16_attn", 2),
    ("softmax_i16_odd_rows_1009", 2),
    ("softmax_i16_odd_rows_1", 2),
    ("softmax_i16_scale_small", 2),
    ("softmax_i16_scale_large", 2),
    ("logistic_i8_scores", 1),
    ("logistic_u8_scores", 1),
    ("logistic_i16_nonpot", 2),
    ("logistic_i16_pot", 2),
];

#[test]
fn fixtures_load() {
    for &(name, elem) in FIXTURES {
        let f = common::load(name);
        let n: usize = f.shape.iter().product();
        assert_eq!(f.input.len(), n * elem, "{name} input");
        assert_eq!(f.output.len(), n * elem, "{name} output");
        assert!(
            f.input_quant.scale > 0.0 && f.output_quant.scale > 0.0,
            "{name}"
        );
        assert!(matches!(f.op.as_str(), "SOFTMAX" | "LOGISTIC"), "{name}");
    }
    assert!(!common::tiers().is_empty());
}

use common::Fixture;
use edgefirst_kernels::{
    CpuFeatures, KernelError, Pool, Quant, SoftmaxI8Params, SoftmaxI8Plan, TensorMut, TensorRef,
    View, ViewMut,
};

const SOFTMAX_I8: &[&str] = &[
    "softmax_i8_yolo11_attn",
    "softmax_i8_yolo26_attn",
    "softmax_i8_odd_rows_1009",
    "softmax_i8_odd_rows_17",
    "softmax_i8_odd_rows_1",
    "softmax_i8_dfl",
];

const OUT_Q: Quant = Quant {
    scale: 1.0 / 256.0,
    zero_point: -128,
};

fn params(f: &Fixture) -> SoftmaxI8Params {
    SoftmaxI8Params {
        input: f.input_quant,
        output: f.output_quant,
        beta: f.beta,
    }
}

fn as_i8(bytes: &[u8]) -> Vec<i8> {
    bytes.iter().map(|&b| b as i8).collect()
}

fn run_i8(plan: &SoftmaxI8Plan, input: &[i8], shape: &[usize], pool: Option<&dyn Pool>) -> Vec<i8> {
    let mut out = vec![0x55u8 as i8; input.len()];
    plan.run(
        TensorRef::I8(View::contiguous(input, shape).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut out, shape).unwrap()),
        pool,
    )
    .unwrap();
    out
}

#[test]
fn softmax_i8_plans_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SoftmaxI8Plan>();
    assert_send_sync::<SoftmaxI8Params>();
}

#[test]
fn softmax_i8_portable_matches_tflite_bit_exact() {
    for name in SOFTMAX_I8 {
        let f = common::load(name);
        let plan = SoftmaxI8Plan::with_features(params(&f), CpuFeatures::default()).unwrap();
        assert_eq!(plan.tier(), edgefirst_kernels::Tier::Scalar);
        let got = run_i8(&plan, &as_i8(&f.input), &f.shape, None);
        let want = as_i8(&f.output);
        let bad = got.iter().zip(&want).filter(|(g, w)| g != w).count();
        assert_eq!(bad, 0, "{name}: {bad} of {} elements differ", want.len());
    }
}

#[test]
fn softmax_i8_every_tier_within_one_lsb() {
    for feats in common::tiers() {
        for name in SOFTMAX_I8 {
            let f = common::load(name);
            let plan = SoftmaxI8Plan::with_features(params(&f), feats).unwrap();
            let got = run_i8(&plan, &as_i8(&f.input), &f.shape, None);
            let want = as_i8(&f.output);
            let mut off_by_one = 0usize;
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                let d = (i32::from(g) - i32::from(w)).abs();
                assert!(
                    d <= 1,
                    "{name} tier {:?}: element {i}: got {g} want {w}",
                    plan.tier()
                );
                off_by_one += usize::from(d == 1);
            }
            let pct = 100.0 * off_by_one as f64 / want.len().max(1) as f64;
            eprintln!(
                "softmax_i8 tier={} fixture={name}: {off_by_one} of {} off by one ({pct:.4}%){}",
                plan.tier().as_str(),
                want.len(),
                if pct > 0.1 { " ABOVE 0.1%" } else { "" }
            );
        }
    }
}

#[test]
fn new_rejects_bad_params() {
    let good = SoftmaxI8Params {
        input: Quant::new(0.1, 0),
        output: OUT_Q,
        beta: 1.0,
    };
    assert!(SoftmaxI8Plan::new(good).is_ok());
    let unsupported =
        |p: SoftmaxI8Params| matches!(SoftmaxI8Plan::new(p), Err(KernelError::Unsupported(_)));
    let invalid =
        |p: SoftmaxI8Params| matches!(SoftmaxI8Plan::new(p), Err(KernelError::InvalidArgument(_)));
    assert!(unsupported(SoftmaxI8Params {
        output: Quant::new(OUT_Q.scale, -127),
        ..good
    }));
    assert!(unsupported(SoftmaxI8Params {
        output: Quant::new(1.0 / 128.0, -128),
        ..good
    }));
    // Within TFLite's 0.1% tolerance on the output scale.
    assert!(SoftmaxI8Plan::new(SoftmaxI8Params {
        output: Quant::new(OUT_Q.scale * 1.0005, -128),
        ..good
    })
    .is_ok());
    for s in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            invalid(SoftmaxI8Params {
                input: Quant::new(s, 0),
                ..good
            }),
            "input scale {s}"
        );
        assert!(
            invalid(SoftmaxI8Params {
                output: Quant::new(s, -128),
                ..good
            }),
            "output scale {s}"
        );
        assert!(invalid(SoftmaxI8Params { beta: s, ..good }), "beta {s}");
    }
    // The product input_scale * beta must stay finite.
    assert!(invalid(SoftmaxI8Params {
        input: Quant::new(f32::MAX, 0),
        beta: 4.0,
        ..good
    }));
}

#[test]
fn run_rejects_mismatch_without_writing() {
    let plan = SoftmaxI8Plan::new(SoftmaxI8Params {
        input: Quant::new(0.1, 0),
        output: OUT_Q,
        beta: 1.0,
    })
    .unwrap();
    let input = vec![1i8; 800];
    let wide = vec![1i16; 800];
    let mut out = vec![0x55i8; 800];

    let r = plan.run(
        TensorRef::I16(View::contiguous(&wide, &[2, 400]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut out, &[2, 400]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(out.iter().all(|&b| b == 0x55));

    let r = plan.run(
        TensorRef::I8(View::contiguous(&input, &[2, 400]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut out[..798], &[2, 399]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(out.iter().all(|&b| b == 0x55));

    let mut wide_out = vec![0x55i16; 800];
    let r = plan.run(
        TensorRef::I8(View::contiguous(&input, &[2, 400]).unwrap()),
        TensorMut::I16(ViewMut::contiguous(&mut wide_out, &[2, 400]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(wide_out.iter().all(|&b| b == 0x55));

    let scalar = [3i8];
    let mut scalar_out = [0x55i8];
    let r = plan.run(
        TensorRef::I8(View::contiguous(&scalar, &[]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut scalar_out, &[]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert_eq!(scalar_out, [0x55]);
}

#[test]
fn run_rejects_aliasing_output_rows() {
    let plan = SoftmaxI8Plan::new(SoftmaxI8Params {
        input: Quant::new(0.1, 0),
        output: OUT_Q,
        beta: 1.0,
    })
    .unwrap();
    let input: Vec<i8> = (0..32).map(|v| v as i8).collect();
    // Zero row stride: all four rows land on the same 8 bytes.
    let mut out = vec![0x55i8; 32];
    let r = plan.run(
        TensorRef::I8(View::contiguous(&input, &[4, 8]).unwrap()),
        TensorMut::I8(ViewMut::strided(&mut out, &[4, 8], &[0, 1]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    // Row stride 4 < row length 8: consecutive rows overlap.
    let r = plan.run(
        TensorRef::I8(View::contiguous(&input, &[4, 8]).unwrap()),
        TensorMut::I8(ViewMut::strided(&mut out, &[4, 8], &[4, 1]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    // Zero stride inside a row.
    let r = plan.run(
        TensorRef::I8(View::contiguous(&input, &[4, 8]).unwrap()),
        TensorMut::I8(ViewMut::strided(&mut out, &[4, 8], &[8, 0]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(out.iter().all(|&b| b == 0x55));
    // A zero stride on a singleton axis cannot alias and is accepted.
    let r = plan.run(
        TensorRef::I8(View::contiguous(&input, &[1, 4, 8]).unwrap()),
        TensorMut::I8(ViewMut::strided(&mut out, &[1, 4, 8], &[0, 8, 1]).unwrap()),
        None,
    );
    assert!(r.is_ok(), "{r:?}");
}

#[test]
fn strided_input_matches_contiguous() {
    let f = common::load("softmax_i8_odd_rows_17");
    assert_eq!(f.shape, [5, 17]);
    let input = as_i8(&f.input);
    let mut transposed = vec![0i8; input.len()];
    for i in 0..5 {
        for j in 0..17 {
            transposed[j * 5 + i] = input[i * 17 + j];
        }
    }
    for feats in common::tiers() {
        let plan = SoftmaxI8Plan::with_features(params(&f), feats).unwrap();
        let want = run_i8(&plan, &input, &[5, 17], None);
        let tflite = as_i8(&f.output);
        let worst = want
            .iter()
            .zip(&tflite)
            .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
            .max();
        assert!(worst <= Some(1), "contiguous vs TFLite");
        let mut got = vec![0i8; input.len()];
        plan.run(
            TensorRef::I8(View::strided(&transposed, &[5, 17], &[1, 5]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut got, &[5, 17]).unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(got, want, "strided input, tier {:?}", plan.tier());

        // Transposed output, and a padded row-major output, both read back through the same strides.
        let mut got_t = vec![0x55i8; input.len()];
        plan.run(
            TensorRef::I8(View::strided(&transposed, &[5, 17], &[1, 5]).unwrap()),
            TensorMut::I8(ViewMut::strided(&mut got_t, &[5, 17], &[1, 5]).unwrap()),
            None,
        )
        .unwrap();
        for i in 0..5 {
            for j in 0..17 {
                assert_eq!(
                    got_t[j * 5 + i],
                    want[i * 17 + j],
                    "transposed output ({i},{j})"
                );
            }
        }
        let mut padded = vec![0x55i8; 5 * 20];
        plan.run(
            TensorRef::I8(View::contiguous(&input, &[5, 17]).unwrap()),
            TensorMut::I8(ViewMut::strided(&mut padded, &[5, 17], &[20, 1]).unwrap()),
            None,
        )
        .unwrap();
        for i in 0..5 {
            assert_eq!(&padded[i * 20..i * 20 + 17], &want[i * 17..i * 17 + 17]);
            assert!(padded[i * 20 + 17..i * 20 + 20].iter().all(|&b| b == 0x55));
        }
    }
}

#[test]
fn long_strided_rows_match_contiguous() {
    // Rows longer than the stack staging buffer take the element-wise path.
    let rows = 3;
    let n = 5000;
    let input: Vec<i8> = (0..rows * n)
        .map(|i| ((i * 7919 + i / 13) % 256) as u8 as i8)
        .collect();
    let mut transposed = vec![0i8; input.len()];
    for i in 0..rows {
        for j in 0..n {
            transposed[j * rows + i] = input[i * n + j];
        }
    }
    for feats in common::tiers() {
        let plan = SoftmaxI8Plan::with_features(
            SoftmaxI8Params {
                input: Quant::new(0.05, 0),
                output: OUT_Q,
                beta: 1.0,
            },
            feats,
        )
        .unwrap();
        let want = run_i8(&plan, &input, &[rows, n], None);
        let mut got = vec![0i8; input.len()];
        plan.run(
            TensorRef::I8(View::strided(&transposed, &[rows, n], &[1, rows as isize]).unwrap()),
            TensorMut::I8(ViewMut::strided(&mut got, &[rows, n], &[1, rows as isize]).unwrap()),
            None,
        )
        .unwrap();
        for i in 0..rows {
            for j in 0..n {
                let d = (i32::from(got[j * rows + i]) - i32::from(want[i * n + j])).abs();
                assert!(d <= 1, "({i},{j}) tier {:?}", plan.tier());
            }
        }
        // Long padded rows keep a unit last-axis stride and run the row kernel in place.
        let pitch = n + 7;
        let mut padded = vec![0x55i8; rows * pitch];
        plan.run(
            TensorRef::I8(View::contiguous(&input, &[rows, n]).unwrap()),
            TensorMut::I8(ViewMut::strided(&mut padded, &[rows, n], &[pitch as isize, 1]).unwrap()),
            None,
        )
        .unwrap();
        for i in 0..rows {
            assert_eq!(&padded[i * pitch..i * pitch + n], &want[i * n..(i + 1) * n]);
            assert!(padded[i * pitch + n..(i + 1) * pitch]
                .iter()
                .all(|&b| b == 0x55));
        }
    }
}

#[test]
fn empty_and_single_element_rows() {
    let plan = SoftmaxI8Plan::new(SoftmaxI8Params {
        input: Quant::new(0.1, 0),
        output: OUT_Q,
        beta: 1.0,
    })
    .unwrap();
    let empty: [i8; 0] = [];
    let mut empty_out: [i8; 0] = [];
    plan.run(
        TensorRef::I8(View::contiguous(&empty, &[0, 16]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut empty_out, &[0, 16]).unwrap()),
        None,
    )
    .unwrap();
    let mut guard = [0x55i8; 4];
    plan.run(
        TensorRef::I8(View::contiguous(&empty, &[4, 0]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut guard, &[4, 0]).unwrap()),
        None,
    )
    .unwrap();
    assert_eq!(guard, [0x55; 4]);

    let ones = [-128i8, 0, 5, 127];
    for feats in common::tiers() {
        let plan = SoftmaxI8Plan::with_features(
            SoftmaxI8Params {
                input: Quant::new(0.1, 0),
                output: OUT_Q,
                beta: 1.0,
            },
            feats,
        )
        .unwrap();
        assert_eq!(run_i8(&plan, &ones, &[4, 1], None), [127; 4]);
    }
}

#[test]
fn pool_and_concurrent_runs_match_single_thread() {
    let f = common::load("softmax_i8_yolo26_attn");
    let input = as_i8(&f.input);
    for feats in common::tiers() {
        let plan = SoftmaxI8Plan::with_features(params(&f), feats).unwrap();
        common::assert_pool_and_concurrency_match(|pool| {
            run_i8(&plan, &input, &f.shape, pool)
                .into_iter()
                .map(|v| v as u8)
                .collect()
        });
    }
}

#[test]
fn pool_runs_match_on_strided_views() {
    let f = common::load("softmax_i8_yolo11_attn");
    let input = as_i8(&f.input);
    let (rows, n) = (800usize, 400usize);
    for feats in common::tiers() {
        let plan = SoftmaxI8Plan::with_features(params(&f), feats).unwrap();
        let want = run_i8(&plan, &input, &[rows, n], None);
        // Padded output rows: staged in place.
        common::assert_pool_and_concurrency_match(|pool| {
            let mut out = vec![0x55i8; rows * (n + 3)];
            plan.run(
                TensorRef::I8(View::contiguous(&input, &[2, 400, n]).unwrap()),
                TensorMut::I8(
                    ViewMut::strided(
                        &mut out,
                        &[2, 400, n],
                        &[400 * (n as isize + 3), n as isize + 3, 1],
                    )
                    .unwrap(),
                ),
                pool,
            )
            .unwrap();
            let rows_out: Vec<u8> = out
                .chunks_exact(n + 3)
                .flat_map(|r| r[..n].iter().map(|&v| v as u8))
                .collect();
            assert_eq!(rows_out, want.iter().map(|&v| v as u8).collect::<Vec<_>>());
            rows_out
        });
        // Transposed output: scattered element by element.
        common::assert_pool_and_concurrency_match(|pool| {
            let mut out = vec![0x55i8; rows * n];
            plan.run(
                TensorRef::I8(View::contiguous(&input, &[rows, n]).unwrap()),
                TensorMut::I8(ViewMut::strided(&mut out, &[rows, n], &[1, rows as isize]).unwrap()),
                pool,
            )
            .unwrap();
            out.into_iter().map(|v| v as u8).collect()
        });
    }
}

/// Scalar model of the NEON row kernel: its two paths, summation order and `tbl`/`tbx` slicing.
fn neon_model_row(table: &[f32; 256], src: &[i8], dst: &mut [i8]) {
    let quantize = |p: f32| ((p.round() as i32) - 128).clamp(-128, 127) as i8;
    let (min, max) = (*src.iter().min().unwrap(), *src.iter().max().unwrap());
    let off = 255u8.wrapping_sub(max as u8);
    let index = |x: i8| (x as u8).wrapping_add(off);
    let range = usize::from(max.abs_diff(min));
    let entries = (range + 16) & !15;
    let vals: Vec<f32> = src.iter().map(|&x| table[usize::from(index(x))]).collect();
    let mut a = [0.0f32; 4];
    let quads = vals.len() / 4 * 4;
    for (j, &t) in vals[..quads].iter().enumerate() {
        a[j % 4] += t;
    }
    let mut sum = (a[0] + a[1]) + (a[2] + a[3]);
    for &t in &vals[quads..] {
        sum += t;
    }
    let inv = 1.0 / (sum * (1.0 / 256.0));
    if src.len() <= 64 && entries >= src.len() {
        for (d, &t) in dst.iter_mut().zip(&vals) {
            *d = quantize(t * inv);
        }
        return;
    }
    let start = 256 - entries;
    let mut otab = [0i8; 256];
    for u in start..256 {
        otab[u] = quantize(table[u] * inv);
    }
    let slices = 4 - (255 - range) / 64;
    let tbl = |slice: usize, i: u8| (i < 64).then(|| otab[slice * 64 + usize::from(i)]);
    for (d, &x) in dst.iter_mut().zip(src) {
        let idx = index(x);
        assert!(
            usize::from(idx) >= start,
            "index {idx} below the built table start {start}"
        );
        let mut r = tbl(3, idx.wrapping_sub(192)).unwrap_or(0);
        for (s, k) in [(2usize, 128u8), (1, 64), (0, 0)] {
            if slices > 3 - s {
                r = tbl(s, idx.wrapping_sub(k)).unwrap_or(r);
            }
        }
        *d = r;
    }
}

#[test]
fn neon_algorithm_model_within_one_lsb() {
    for name in SOFTMAX_I8 {
        let f = common::load(name);
        let mut table = [0.0f32; 256];
        for (v, t) in table.iter_mut().rev().enumerate() {
            *t = (-f.input_quant.scale * f.beta * v as f32).exp();
        }
        let n = *f.shape.last().unwrap();
        let input = as_i8(&f.input);
        let want = as_i8(&f.output);
        let mut got = vec![0i8; input.len()];
        for (s, d) in input.chunks_exact(n).zip(got.chunks_exact_mut(n)) {
            neon_model_row(&table, s, d);
        }
        let mut off_by_one = 0usize;
        for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
            let d = (i32::from(g) - i32::from(w)).abs();
            assert!(d <= 1, "{name}: element {i}: model {g} want {w}");
            off_by_one += usize::from(d == 1);
        }
        eprintln!(
            "neon model fixture={name}: {off_by_one} of {} off by one ({:.4}%)",
            want.len(),
            100.0 * off_by_one as f64 / want.len() as f64
        );
    }
}

#[test]
fn neon_algorithm_model_matches_portable_on_random_rows() {
    use rand::{RngExt, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x50F7_3A11);
    let input_scale = 0.05f32;
    let mut table = [0.0f32; 256];
    for (v, t) in table.iter_mut().rev().enumerate() {
        *t = (-input_scale * v as f32).exp();
    }
    let plan = SoftmaxI8Plan::with_features(
        SoftmaxI8Params {
            input: Quant::new(input_scale, 0),
            output: OUT_Q,
            beta: 1.0,
        },
        CpuFeatures::default(),
    )
    .unwrap();
    let mut lens: Vec<usize> = (1..=70).collect();
    lens.extend([255, 256, 257, 1009, 4097]);
    for &n in &lens {
        for &span in &[1i32, 16, 63, 64, 65, 128, 192, 256] {
            let base = rng.random_range(-128..=(128 - span));
            let mut row: Vec<i8> = (0..n)
                .map(|_| (base + rng.random_range(0..span)) as i8)
                .collect();
            if n >= 2 && span == 256 {
                (row[0], row[n - 1]) = (-128, 127);
            }
            let want = run_i8(&plan, &row, &[1, n], None);
            let mut got = vec![0i8; n];
            neon_model_row(&table, &row, &mut got);
            for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
                assert!(
                    (i32::from(g) - i32::from(w)).abs() <= 1,
                    "len {n} span {span} element {i}"
                );
            }
        }
    }
}

use edgefirst_kernels::{SoftmaxI16Params, SoftmaxI16Plan};

const SOFTMAX_I16: &[&str] = &[
    "softmax_i16_attn",
    "softmax_i16_odd_rows_1009",
    "softmax_i16_odd_rows_1",
    "softmax_i16_scale_small",
    "softmax_i16_scale_large",
];

const OUT_Q16: Quant = Quant {
    scale: 1.0 / 32768.0,
    zero_point: 0,
};

const GOOD_I16: SoftmaxI16Params = SoftmaxI16Params {
    input: Quant {
        scale: 1.0 / 2048.0,
        zero_point: 0,
    },
    output: OUT_Q16,
    beta: 1.0,
};

fn params_i16(f: &Fixture) -> SoftmaxI16Params {
    SoftmaxI16Params {
        input: f.input_quant,
        output: f.output_quant,
        beta: f.beta,
    }
}

fn as_i16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|b| i16::from_le_bytes([b[0], b[1]]))
        .collect()
}

fn run_i16(
    plan: &SoftmaxI16Plan,
    input: &[i16],
    shape: &[usize],
    pool: Option<&dyn Pool>,
) -> Vec<i16> {
    let mut out = vec![0x5555i16; input.len()];
    plan.run(
        TensorRef::I16(View::contiguous(input, shape).unwrap()),
        TensorMut::I16(ViewMut::contiguous(&mut out, shape).unwrap()),
        pool,
    )
    .unwrap();
    out
}

fn transpose<T: Copy + Default>(src: &[T], rows: usize, n: usize) -> Vec<T> {
    let mut t = vec![T::default(); src.len()];
    for i in 0..rows {
        for j in 0..n {
            t[j * rows + i] = src[i * n + j];
        }
    }
    t
}

#[test]
fn softmax_i16_plans_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SoftmaxI16Plan>();
    assert_send_sync::<SoftmaxI16Params>();
}

#[test]
fn softmax_i16_bit_exact_every_tier() {
    for feats in common::tiers() {
        for name in SOFTMAX_I16 {
            let f = common::load(name);
            let plan = SoftmaxI16Plan::with_features(params_i16(&f), feats).unwrap();
            let got = run_i16(&plan, &as_i16(&f.input), &f.shape, None);
            let want = as_i16(&f.output);
            let bad = got.iter().zip(&want).filter(|(g, w)| g != w).count();
            assert_eq!(
                bad,
                0,
                "{name} tier {}: {bad} of {} elements differ",
                plan.tier().as_str(),
                want.len()
            );
        }
    }
}

#[test]
fn softmax_i16_length_one_and_all_equal_rows() {
    let f = common::load("softmax_i16_odd_rows_1");
    assert_eq!(f.shape, [7, 1]);
    assert!(as_i16(&f.output).iter().all(|&v| v == 32767));
    for name in ["softmax_i16_odd_rows_1009", "softmax_i16_attn"] {
        let f = common::load(name);
        let n = *f.shape.last().unwrap();
        let input = as_i16(&f.input);
        assert!(
            input[..n].iter().all(|&v| v == input[0]),
            "{name}: row 0 is the all-equal row"
        );
        let want = as_i16(&f.output);
        for feats in common::tiers() {
            let plan = SoftmaxI16Plan::with_features(params_i16(&f), feats).unwrap();
            let got = run_i16(&plan, &input[..n], &[1, n], None);
            assert_eq!(got, want[..n], "{name} tier {}", plan.tier().as_str());
            assert!(got.iter().all(|&v| v == got[0]));
        }
    }
    for feats in common::tiers() {
        let plan = SoftmaxI16Plan::with_features(GOOD_I16, feats).unwrap();
        let ones = [i16::MIN, -1, 0, i16::MAX];
        assert_eq!(run_i16(&plan, &ones, &[4, 1], None), [32767; 4]);
    }
}

#[test]
fn new_rejects_bad_params_i16() {
    assert!(SoftmaxI16Plan::new(GOOD_I16).is_ok());
    let unsupported =
        |p: SoftmaxI16Params| matches!(SoftmaxI16Plan::new(p), Err(KernelError::Unsupported(_)));
    let invalid = |p: SoftmaxI16Params| {
        matches!(SoftmaxI16Plan::new(p), Err(KernelError::InvalidArgument(_)))
    };
    assert!(unsupported(SoftmaxI16Params {
        input: Quant::new(GOOD_I16.input.scale, 1),
        ..GOOD_I16
    }));
    assert!(unsupported(SoftmaxI16Params {
        output: Quant::new(OUT_Q16.scale, 1),
        ..GOOD_I16
    }));
    assert!(unsupported(SoftmaxI16Params {
        output: Quant::new(1.0 / 16384.0, 0),
        ..GOOD_I16
    }));
    // Within TFLite's 0.1% tolerance on the output scale.
    assert!(SoftmaxI16Plan::new(SoftmaxI16Params {
        output: Quant::new(OUT_Q16.scale * 1.0005, 0),
        ..GOOD_I16
    })
    .is_ok());
    // input_scale * beta >= 5 would overflow TFLite's int32 left shift of the input difference.
    assert!(unsupported(SoftmaxI16Params {
        input: Quant::new(5.5, 0),
        ..GOOD_I16
    }));
    assert!(SoftmaxI16Plan::new(SoftmaxI16Params {
        input: Quant::new(4.5, 0),
        ..GOOD_I16
    })
    .is_ok());
    for s in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(
            invalid(SoftmaxI16Params {
                input: Quant::new(s, 0),
                ..GOOD_I16
            }),
            "input scale {s}"
        );
        assert!(
            invalid(SoftmaxI16Params {
                output: Quant::new(s, 0),
                ..GOOD_I16
            }),
            "output scale {s}"
        );
        assert!(
            invalid(SoftmaxI16Params {
                beta: s,
                ..GOOD_I16
            }),
            "beta {s}"
        );
    }
    assert!(invalid(SoftmaxI16Params {
        input: Quant::new(f32::MAX, 0),
        beta: 4.0,
        ..GOOD_I16
    }));
}

#[test]
fn run_rejects_mismatch_without_writing_i16() {
    let plan = SoftmaxI16Plan::new(GOOD_I16).unwrap();
    let input = vec![1i16; 800];
    let narrow = vec![1i8; 800];
    let mut out = vec![0x5555i16; 800];

    let r = plan.run(
        TensorRef::I8(View::contiguous(&narrow, &[2, 400]).unwrap()),
        TensorMut::I16(ViewMut::contiguous(&mut out, &[2, 400]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(out.iter().all(|&b| b == 0x5555));

    let r = plan.run(
        TensorRef::I16(View::contiguous(&input, &[2, 400]).unwrap()),
        TensorMut::I16(ViewMut::contiguous(&mut out[..798], &[2, 399]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(out.iter().all(|&b| b == 0x5555));

    let mut narrow_out = vec![0x55i8; 800];
    let r = plan.run(
        TensorRef::I16(View::contiguous(&input, &[2, 400]).unwrap()),
        TensorMut::I8(ViewMut::contiguous(&mut narrow_out, &[2, 400]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(narrow_out.iter().all(|&b| b == 0x55));

    let mut u16_out = vec![0x5555u16; 800];
    let r = plan.run(
        TensorRef::I16(View::contiguous(&input, &[2, 400]).unwrap()),
        TensorMut::U16(ViewMut::contiguous(&mut u16_out, &[2, 400]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(u16_out.iter().all(|&b| b == 0x5555));

    // Aliasing output rows.
    let r = plan.run(
        TensorRef::I16(View::contiguous(&input, &[2, 400]).unwrap()),
        TensorMut::I16(ViewMut::strided(&mut out, &[2, 400], &[0, 1]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(out.iter().all(|&b| b == 0x5555));

    let scalar = [3i16];
    let mut scalar_out = [0x5555i16];
    let r = plan.run(
        TensorRef::I16(View::contiguous(&scalar, &[]).unwrap()),
        TensorMut::I16(ViewMut::contiguous(&mut scalar_out, &[]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert_eq!(scalar_out, [0x5555]);
}

#[test]
fn run_rejects_rows_whose_sum_could_overflow_i16() {
    let plan = SoftmaxI16Plan::new(GOOD_I16).unwrap();
    let n = 65_539;
    let input = vec![7i16; n];
    let mut out = vec![0x5555i16; n];
    let r = plan.run(
        TensorRef::I16(View::contiguous(&input, &[1, n]).unwrap()),
        TensorMut::I16(ViewMut::contiguous(&mut out, &[1, n]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::Unsupported(_))), "{r:?}");
    assert!(out.iter().all(|&b| b == 0x5555));
    // The longest row whose sum of exps fits in an i32.
    let got = run_i16(&plan, &input[..n - 1], &[1, n - 1], None);
    assert!(got.iter().all(|&v| v == got[0]));
}

#[test]
fn empty_rows_i16() {
    let plan = SoftmaxI16Plan::new(GOOD_I16).unwrap();
    let empty: [i16; 0] = [];
    let mut empty_out: [i16; 0] = [];
    plan.run(
        TensorRef::I16(View::contiguous(&empty, &[0, 16]).unwrap()),
        TensorMut::I16(ViewMut::contiguous(&mut empty_out, &[0, 16]).unwrap()),
        None,
    )
    .unwrap();
    let mut guard = [0x5555i16; 4];
    plan.run(
        TensorRef::I16(View::contiguous(&empty, &[4, 0]).unwrap()),
        TensorMut::I16(ViewMut::contiguous(&mut guard, &[4, 0]).unwrap()),
        None,
    )
    .unwrap();
    assert_eq!(guard, [0x5555; 4]);
}

#[test]
fn strided_views_match_contiguous_i16() {
    let f = common::load("softmax_i16_odd_rows_1009");
    let (rows, n) = (3usize, 1009usize);
    assert_eq!(f.shape, [rows, n]);
    let input = as_i16(&f.input);
    let want = as_i16(&f.output);
    let transposed = transpose(&input, rows, n);
    for feats in common::tiers() {
        let plan = SoftmaxI16Plan::with_features(params_i16(&f), feats).unwrap();
        let tier = plan.tier().as_str();
        // Transposed input, contiguous output: staged input row.
        let mut got = vec![0x5555i16; input.len()];
        plan.run(
            TensorRef::I16(View::strided(&transposed, &[rows, n], &[1, rows as isize]).unwrap()),
            TensorMut::I16(ViewMut::contiguous(&mut got, &[rows, n]).unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(got, want, "strided input, tier {tier}");
        // Transposed output: staged output row.
        let mut got_t = vec![0x5555i16; input.len()];
        plan.run(
            TensorRef::I16(View::contiguous(&input, &[rows, n]).unwrap()),
            TensorMut::I16(ViewMut::strided(&mut got_t, &[rows, n], &[1, rows as isize]).unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(
            got_t,
            transpose(&want, rows, n),
            "strided output, tier {tier}"
        );
        // Padded output rows.
        let pitch = n + 5;
        let mut padded = vec![0x5555i16; rows * pitch];
        plan.run(
            TensorRef::I16(View::contiguous(&input, &[rows, n]).unwrap()),
            TensorMut::I16(
                ViewMut::strided(&mut padded, &[rows, n], &[pitch as isize, 1]).unwrap(),
            ),
            None,
        )
        .unwrap();
        for i in 0..rows {
            assert_eq!(&padded[i * pitch..i * pitch + n], &want[i * n..(i + 1) * n]);
            assert!(padded[i * pitch + n..(i + 1) * pitch]
                .iter()
                .all(|&b| b == 0x5555));
        }
    }
}

#[test]
fn long_strided_rows_match_contiguous_i16() {
    // Rows longer than the stack staging buffer take the element-wise path.
    let (rows, n) = (3usize, 5000usize);
    let input: Vec<i16> = (0..rows * n)
        .map(|i| ((i * 7919 + i / 13) % 65536) as u16 as i16)
        .collect();
    let transposed = transpose(&input, rows, n);
    for feats in common::tiers() {
        let plan = SoftmaxI16Plan::with_features(GOOD_I16, feats).unwrap();
        let want = run_i16(&plan, &input, &[rows, n], None);
        let mut got = vec![0x5555i16; input.len()];
        plan.run(
            TensorRef::I16(View::strided(&transposed, &[rows, n], &[1, rows as isize]).unwrap()),
            TensorMut::I16(ViewMut::strided(&mut got, &[rows, n], &[1, rows as isize]).unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(
            got,
            transpose(&want, rows, n),
            "tier {}",
            plan.tier().as_str()
        );
    }
}

#[test]
fn pool_and_concurrent_runs_match_single_thread_i16() {
    let f = common::load("softmax_i16_attn");
    let input = as_i16(&f.input);
    let want = as_i16(&f.output);
    for feats in common::tiers() {
        let plan = SoftmaxI16Plan::with_features(params_i16(&f), feats).unwrap();
        common::assert_pool_and_concurrency_match(|pool| {
            let out = run_i16(&plan, &input, &f.shape, pool);
            assert_eq!(out, want);
            out.into_iter().flat_map(i16::to_le_bytes).collect()
        });
    }
}
