// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The semantic kernels against verbatim copies of the decoder's scalar code.

use super::*;
use crate::prim::softmax::exp_nonpositive;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// A score type the tests generate.
trait Gen: SemanticScore + PartialOrd + std::fmt::Debug {
    /// A score, from a few values when `ties` so classes often tie.
    fn gen(rng: &mut Rng, ties: bool) -> Self;
    /// Specials: NaN, infinities, signed zeros, extremes.
    fn specials() -> Vec<Self>;
    fn to_f32(self) -> f32;
}

macro_rules! int_gen {
    ($($t:ty),*) => {$(
        impl Gen for $t {
            fn gen(rng: &mut Rng, ties: bool) -> Self {
                if ties {
                    (rng.below(4) as $t).wrapping_add(<$t>::MAX - 3)
                } else {
                    rng.next() as $t
                }
            }
            fn specials() -> Vec<Self> {
                vec![<$t>::MIN, <$t>::MAX, 0, 1, <$t>::MIN + 1, <$t>::MAX - 1]
            }
            fn to_f32(self) -> f32 {
                self as f32
            }
        }
    )*};
}
int_gen!(u8, i8, u16, i16);

impl Gen for f32 {
    fn gen(rng: &mut Rng, ties: bool) -> Self {
        if ties {
            [0.5, -1.0, 2.0, 2.0][rng.below(4)]
        } else {
            rng.unit() * 40.0 - 30.0
        }
    }
    fn specials() -> Vec<Self> {
        vec![
            f32::NAN,
            -f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.0,
            -0.0,
            f32::MAX,
            f32::MIN,
            1e-45,
        ]
    }
    fn to_f32(self) -> f32 {
        self
    }
}

impl Gen for f16 {
    fn gen(rng: &mut Rng, ties: bool) -> Self {
        f16::from_f32(f32::gen(rng, ties))
    }
    fn specials() -> Vec<Self> {
        vec![
            f16::NAN,
            -f16::NAN,
            f16::INFINITY,
            f16::NEG_INFINITY,
            f16::ZERO,
            f16::NEG_ZERO,
            f16::MAX,
            f16::MIN,
            f16::from_bits(1),
        ]
    }
    fn to_f32(self) -> f32 {
        f16::to_f32(self)
    }
}

/// Planar scores: `classes` planes of `n` pixels, `cs >= n` apart; specials sprinkled in.
fn planes<T: Gen>(rng: &mut Rng, classes: usize, n: usize, cs: usize, ties: bool) -> Vec<T> {
    let specials = T::specials();
    let mut v: Vec<T> = (0..(classes - 1) * cs + n)
        .map(|_| T::gen(rng, ties))
        .collect();
    for _ in 0..v.len() / 13 {
        let i = rng.below(v.len());
        v[i] = specials[rng.below(specials.len())];
    }
    v
}

/// The decoder's `argmax_pixel`.
fn argmax_ref<T: Gen>(src: &[T], cs: usize, classes: usize, x: usize) -> u16 {
    let mut best = src[x];
    let mut idx = 0u16;
    for c in 1..classes {
        let v = src[x + c * cs];
        if v > best {
            best = v;
            idx = c as u16;
        }
    }
    idx
}

const CLASSES: [usize; 12] = [1, 2, 3, 4, 5, 16, 17, 21, 33, 150, 257, 300];
const PIXELS: [usize; 13] = [0, 1, 3, 7, 15, 16, 17, 31, 33, 64, 171, 256, 529];

fn check_argmax<T: Gen>(
    planar: impl Fn(&[T], usize, usize, &mut [u16]) -> bool,
    interleaved: impl Fn(&[T], usize, &mut [u16]) -> bool,
    what: &str,
) {
    let mut rng = Rng(0x5eed_2026);
    for classes in CLASSES {
        for n in PIXELS {
            for ties in [false, true] {
                let cs = n.max(1) + rng.below(3);
                let src = planes::<T>(&mut rng, classes, n, cs, ties);
                let mut got = vec![u16::MAX; n];
                assert!(planar(&src, cs, classes, &mut got), "{what}");
                for (x, &g) in got.iter().enumerate() {
                    let want = argmax_ref(&src, cs, classes, x);
                    assert_eq!(g, want, "{what} planar c={classes} n={n} cs={cs} x={x}");
                }
                let inter: Vec<T> = (0..n * classes)
                    .map(|i| src[(i % classes) * cs + i / classes])
                    .collect();
                let mut got = vec![u16::MAX; n];
                assert!(interleaved(&inter, classes, &mut got), "{what}");
                for (x, &g) in got.iter().enumerate() {
                    let want = argmax_ref(&src, cs, classes, x);
                    assert_eq!(g, want, "{what} interleaved c={classes} n={n} x={x}");
                }
            }
        }
    }
}

#[test]
fn argmax_matches_the_scalar_fold() {
    check_argmax::<u8>(u8::argmax_planar_neon, u8::argmax_interleaved_neon, "u8");
    check_argmax::<i8>(i8::argmax_planar_neon, i8::argmax_interleaved_neon, "i8");
    check_argmax::<u16>(u16::argmax_planar_neon, u16::argmax_interleaved_neon, "u16");
    check_argmax::<i16>(i16::argmax_planar_neon, i16::argmax_interleaved_neon, "i16");
    check_argmax::<f32>(f32::argmax_planar_neon, f32::argmax_interleaved_neon, "f32");
    check_argmax::<f16>(f16::argmax_planar_neon, f16::argmax_interleaved_neon, "f16");
    if std::arch::is_aarch64_feature_detected!("fp16") {
        // SAFETY: FEAT_FP16 was detected.
        check_argmax::<f16>(
            |s, cs, c, d| unsafe { argmax_planar_f16_neon_fp16(s, cs, c, d) },
            |s, c, d| unsafe { argmax_interleaved_f16_neon_fp16(s, c, d) },
            "f16 fp16",
        );
    } else {
        crate::test_support::report_skip("f16 FP16 argmax (no FEAT_FP16)");
    }
}

#[test]
fn argmax_handles_the_largest_class_count() {
    let classes = 1 << 16;
    for n in [1, 17] {
        let mut src = vec![0u8; classes * n];
        for x in 0..n {
            src[(classes - 1) * n + x] = 1;
        }
        let mut got = vec![0u16; n];
        assert!(u8::argmax_planar_neon(&src, n, classes, &mut got));
        assert!(got.iter().all(|&c| c == u16::MAX), "{got:?}");
    }
    let mut got = [0u16; 1];
    assert!(!u8::argmax_planar_neon(&[0u8; 2], 1, classes + 1, &mut got));
}

#[test]
fn argmax_rejects_invalid_arguments() {
    let src = [1u8; 40];
    let mut dst = [7u16; 17];
    assert!(
        !u8::argmax_planar_neon(&src, 20, 3, &mut dst),
        "planes past the end"
    );
    assert!(
        !u8::argmax_planar_neon(&src, 0, 2, &mut dst),
        "zero class stride"
    );
    assert!(!u8::argmax_planar_neon(&src, 17, 0, &mut dst), "no classes");
    assert!(
        !u8::argmax_interleaved_neon(&src, 3, &mut dst),
        "pixels past the end"
    );
    assert!(
        !u8::argmax_interleaved_neon(&src, 0, &mut dst),
        "no classes"
    );
    assert_eq!(dst, [7; 17]);
}

/// The decoder's softmax confidence of one pixel.
fn softmax_ref<T: Gen>(
    src: &[T],
    cs: usize,
    classes: usize,
    x: usize,
    top: f32,
    scale: f32,
) -> f32 {
    let mut s = 0f32;
    for c in 0..classes {
        s += exp_nonpositive(scale * (src[c * cs + x].to_f32() - top));
    }
    1.0 / s
}

fn check_softmax<T: Gen>(what: &str) {
    let mut rng = Rng(0x50f7_2026);
    for classes in CLASSES {
        for n in PIXELS {
            for scale in [1.0f32, 0.125, 3.5] {
                let cs = n.max(1) + rng.below(3);
                let src = planes::<T>(&mut rng, classes, n, cs, false);
                // The winner's score, as the decoder passes, or a top no score exceeds.
                let top: Vec<f32> = (0..n)
                    .map(|x| match rng.below(8) {
                        0 => f32::INFINITY,
                        1 => f32::NAN,
                        _ => src[argmax_ref(&src, cs, classes, x) as usize * cs + x].to_f32(),
                    })
                    .collect();
                let mut got = vec![f32::NAN; n];
                assert!(T::softmax_planar_neon(
                    &src, cs, classes, &top, scale, &mut got
                ));
                let inter: Vec<T> = (0..n * classes)
                    .map(|i| src[(i % classes) * cs + i / classes])
                    .collect();
                let mut got_i = vec![f32::NAN; n];
                assert!(T::softmax_interleaved_neon(
                    &inter, classes, &top, scale, &mut got_i
                ));
                for x in 0..n {
                    let want = softmax_ref(&src, cs, classes, x, top[x], scale);
                    let ctx = format!("{what} c={classes} n={n} x={x} scale={scale}");
                    assert_eq!(got[x].to_bits(), want.to_bits(), "planar {ctx}");
                    assert_eq!(got_i[x].to_bits(), want.to_bits(), "interleaved {ctx}");
                }
            }
        }
    }
}

#[test]
fn softmax_confidence_matches_the_scalar_sum_bit_for_bit() {
    check_softmax::<u8>("u8");
    check_softmax::<i8>("i8");
    check_softmax::<u16>("u16");
    check_softmax::<i16>("i16");
    check_softmax::<f32>("f32");
    check_softmax::<f16>("f16");
}

#[test]
fn exp_nonpositive4_matches_the_scalar_function() {
    let mut xs: Vec<f32> = (0..1u32 << 20)
        .map(|i| f32::from_bits(i.wrapping_mul(4099) | 0x8000_0000))
        .collect();
    xs.extend((0..4000).map(|i| -(i as f32) * 0.025));
    xs.extend([
        0.0,
        -0.0,
        f32::NAN,
        -f32::NAN,
        f32::NEG_INFINITY,
        -f32::MIN_POSITIVE,
    ]);
    for chunk in xs.chunks_exact(4) {
        // SAFETY: `chunk` holds 4 elements.
        let got = unsafe {
            let mut out = [0f32; 4];
            vst1q_f32(
                out.as_mut_ptr(),
                exp_nonpositive4(vld1q_f32(chunk.as_ptr())),
            );
            out
        };
        for (&x, g) in chunk.iter().zip(got) {
            assert_eq!(g.to_bits(), exp_nonpositive(x).to_bits(), "x={x:e}");
        }
    }
}

/// The decoder's sigmoid.
fn sigmoid_ref(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// Sigmoid tolerance: `2^-21` of the exact value, or `2^-120` where it underflows.
fn sigmoid_close(got: f32, want: f32) -> bool {
    (got.is_nan() && want.is_nan())
        || (got - want).abs() <= want.abs() * 2f32.powi(-21) + 2f32.powi(-120)
}

#[test]
fn sigmoid_is_within_its_bound() {
    let mut xs: Vec<f32> = (0..1u32 << 22)
        .map(|i| f32::from_bits(i.wrapping_mul(1021)))
        .collect();
    xs.extend((0..20_000).map(|i| (i as f32 - 10_000.0) * 0.01));
    xs.extend([
        0.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        88.0,
        -88.0,
        -104.0,
    ]);
    let mut got = vec![0f32; xs.len()];
    assert!(sigmoid_neon(&xs, &mut got));
    let mut worst = 0f32;
    for (&x, &g) in xs.iter().zip(&got) {
        let want = sigmoid_ref(x);
        assert!(sigmoid_close(g, want), "x={x:e} got {g:e} want {want:e}");
        if want.abs() > 1e-30 {
            worst = worst.max((g - want).abs() / want.abs());
        }
    }
    println!(
        "sigmoid worst relative error {worst:e} (bound {:e})",
        2f32.powi(-21)
    );
    assert!(!sigmoid_neon(&xs, &mut got[1..]));
}

fn check_binary<T: Gen>(what: &str) {
    let mut rng = Rng(0xb1_2026);
    for n in PIXELS.into_iter().chain([200]) {
        let src = planes::<T>(&mut rng, 1, n, n, false);
        for (zp, scale) in [(0.0f32, 1.0f32), (3.0, 0.125), (-17.0, 0.02)] {
            for logit in [false, true] {
                let mut got = vec![f32::NAN; n];
                assert!(T::binary_confidence_neon(&src, zp, scale, logit, &mut got));
                for (x, &g) in got.iter().enumerate() {
                    let p = (src[x].to_f32() - zp) * scale;
                    let ctx = format!("{what} n={n} x={x} logit={logit} p={p:e}");
                    if logit {
                        let s = sigmoid_ref(p);
                        let want = s.max(1.0 - s);
                        assert!(sigmoid_close(g, want), "{ctx}: {g:e} vs {want:e}");
                    } else {
                        assert_eq!(g.to_bits(), p.max(1.0 - p).to_bits(), "{ctx}");
                    }
                }
            }
        }
    }
}

#[test]
fn binary_confidence_matches_the_scalar_code() {
    check_binary::<u8>("u8");
    check_binary::<i8>("i8");
    check_binary::<u16>("u16");
    check_binary::<i16>("i16");
    check_binary::<f32>("f32");
    check_binary::<f16>("f16");
    let mut dst = [0f32; 3];
    assert!(!f32::binary_confidence_neon(
        &[0.0; 4], 0.0, 1.0, true, &mut dst
    ));
}

fn check_threshold<T: Gen>(what: &str) {
    let mut rng = Rng(0x7e_2026);
    let cuts = [
        0.0f32,
        -0.0,
        0.5,
        -0.5,
        1.0,
        127.0,
        126.5,
        -128.0,
        -128.5,
        255.0,
        254.5,
        32767.0,
        -32768.5,
        65535.0,
        65534.5,
        1e30,
        -1e30,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ];
    for n in PIXELS {
        let src = planes::<T>(&mut rng, 1, n, n, false);
        for cut in cuts {
            let mut got = vec![9u16; n];
            assert!(T::threshold_neon(&src, cut, &mut got));
            for (x, &g) in got.iter().enumerate() {
                let want = u16::from(src[x].to_f32() > cut);
                assert_eq!(g, want, "{what} n={n} x={x} v={:?} cut={cut}", src[x]);
            }
        }
    }
}

#[test]
fn threshold_matches_the_scalar_compare() {
    check_threshold::<u8>("u8");
    check_threshold::<i8>("i8");
    check_threshold::<u16>("u16");
    check_threshold::<i16>("i16");
    check_threshold::<f32>("f32");
    check_threshold::<f16>("f16");
    assert!(!u8::threshold_neon(&[0; 3], 0.0, &mut [0u16; 2]));
}

/// The decoder's `ConfOut for u8`.
fn conf_ref(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Floats around the confidence range and every rounding boundary of it.
fn conf_values() -> Vec<f32> {
    let mut v: Vec<f32> = (0..1u32 << 20)
        .map(|i| f32::from_bits(i.wrapping_mul(4093)))
        .collect();
    for k in 0..=256 {
        let b = (k as f32 - 0.5) / 255.0;
        for d in -3i32..=3 {
            v.push(f32::from_bits((b.to_bits() as i32 + d) as u32));
        }
    }
    v.extend([
        f32::NAN,
        -f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        1.0,
        1.5,
        -0.5,
    ]);
    v
}

#[test]
fn confidence_to_u8_matches_the_scalar_cast() {
    let src = conf_values();
    let mut got = vec![0u8; src.len()];
    assert!(confidence_to_u8_neon(&src, &mut got));
    for (&v, &g) in src.iter().zip(&got) {
        assert_eq!(g, conf_ref(v), "v={v:e}");
    }
    assert!(!confidence_to_u8_neon(&src, &mut got[1..]));
}

#[test]
fn blends_match_the_scalar_order() {
    let mut rng = Rng(0xb7e_2026);
    let specials = f32::specials();
    for n in PIXELS.into_iter().chain([683]) {
        let row = |rng: &mut Rng| -> Vec<f32> {
            (0..n)
                .map(|_| match rng.below(10) {
                    0 => specials[rng.below(specials.len())],
                    _ => rng.unit() * 1.2 - 0.1,
                })
                .collect()
        };
        let (a, b, wb) = (row(&mut rng), row(&mut rng), row(&mut rng));
        let wa: Vec<f32> = wb.iter().map(|w| 1.0 - w).collect();
        let mut got = vec![0f32; n];
        assert!(lerp_neon(&a, &b, &wa, &wb, &mut got));
        for i in 0..n {
            assert_eq!(
                got[i].to_bits(),
                (wa[i] * a[i] + wb[i] * b[i]).to_bits(),
                "lerp {i}"
            );
        }
        for fy in [0.0f32, 0.25, 0.7, 1.0] {
            let wy0 = 1.0 - fy;
            let want: Vec<f32> = (0..n).map(|i| wy0 * a[i] + fy * b[i]).collect();
            let mut got = vec![0f32; n];
            assert!(blend_neon(&a, &b, wy0, fy, &mut got));
            let mut conf = vec![0u8; n];
            assert!(blend_confidence_u8_neon(&a, &b, wy0, fy, &mut conf));
            let mut cls8 = vec![9u8; n];
            assert!(blend_threshold_neon(&a, &b, wy0, fy, 0.5, &mut cls8));
            let mut cls16 = vec![9u16; n];
            assert!(blend_threshold_neon(&a, &b, wy0, fy, 0.5, &mut cls16));
            for i in 0..n {
                assert_eq!(got[i].to_bits(), want[i].to_bits(), "blend {i}");
                assert_eq!(conf[i], conf_ref(want[i]), "conf {i}");
                assert_eq!(cls8[i], u8::from(want[i] > 0.5), "class u8 {i}");
                assert_eq!(cls16[i], u16::from(want[i] > 0.5), "class u16 {i}");
            }
        }
    }
    assert!(!blend_neon(&[0.0; 3], &[0.0; 2], 0.5, 0.5, &mut [0.0; 3]));
    assert!(!lerp_neon(
        &[0.0; 3],
        &[0.0; 3],
        &[0.0; 3],
        &[0.0; 2],
        &mut [0.0; 3]
    ));
}

/// The decoder's `Band::argmax`.
fn cell_ref(classes: &[u16], corners: &[[f32; 4]], fy: f32, fx: f32) -> u16 {
    let mut best = f32::NEG_INFINITY;
    let mut idx = 0;
    for (&class, v) in classes.iter().zip(corners) {
        let (wy0, wx0) = (1.0 - fy, 1.0 - fx);
        let s = wy0 * (wx0 * v[0] + fx * v[1]) + fy * (wx0 * v[2] + fx * v[3]);
        if s > best {
            best = s;
            idx = class;
        }
    }
    idx
}

#[test]
fn cell_argmax_matches_the_scalar_band() {
    let mut rng = Rng(0xce11_2026);
    let specials = f32::specials();
    for k in [0usize, 1, 2, 3, 5, 9, 40] {
        for run in [1usize, 3, 4, 5, 8, 9] {
            for _ in 0..30 {
                let classes: Vec<u16> = (0..k).map(|_| rng.below(65536) as u16).collect();
                let corners: Vec<[f32; 4]> = (0..k)
                    .map(|_| {
                        [0; 4].map(|_| match rng.below(12) {
                            0 => specials[rng.below(specials.len())],
                            1 => 2.0,
                            _ => rng.unit() * 4.0 - 2.0,
                        })
                    })
                    .collect();
                let fy = [0.0, 0.25, rng.unit(), 1.0][rng.below(4)];
                let fx: Vec<f32> = (0..run)
                    .map(|_| [0.0, 0.5, rng.unit()][rng.below(3)])
                    .collect();
                let mut got = vec![7u16; run];
                assert!(cell_argmax_neon(&classes, &corners, fy, &fx, &mut got));
                for (i, &g) in got.iter().enumerate() {
                    assert_eq!(
                        g,
                        cell_ref(&classes, &corners, fy, fx[i]),
                        "k={k} run={run} i={i}"
                    );
                }
            }
        }
    }
    assert!(!cell_argmax_neon(&[1], &[], 0.0, &[0.0], &mut [0]));
    assert!(!cell_argmax_neon(&[], &[], 0.0, &[0.0], &mut [0, 0]));
}

/// Nearest positions of `out` samples over `len` source elements, as the decoder computes them,
/// plus irregular ones.
fn positions(rng: &mut Rng, out: usize, len: usize) -> Vec<Vec<u32>> {
    let near = (0..out)
        .map(|d| ((len as f32 / out as f32) * (d as f32 + 0.5)).clamp(0.0, len as f32 - 1.0) as u32)
        .collect();
    let random = (0..out).map(|_| rng.below(len) as u32).collect();
    let clustered = (0..out)
        .map(|d| ((d / 16 * 7 + rng.below(9)) % len) as u32)
        .collect();
    vec![near, random, clustered]
}

#[test]
fn row_gathers_match_indexing() {
    let mut rng = Rng(0x6a7_2026);
    for len in [1usize, 2, 5, 15, 16, 31, 32, 33, 64, 168, 600] {
        for out in [0usize, 1, 7, 15, 16, 17, 40, 168, 683] {
            for index in positions(&mut rng, out, len) {
                let src16: Vec<u16> = (0..len).map(|_| rng.next() as u16).collect();
                let src32: Vec<f32> = (0..len)
                    .map(|_| f32::from_bits(rng.next() as u32))
                    .collect();
                let ctx = format!("len={len} out={out} index={index:?}");
                let plan = RowGather::u16(&index, len).unwrap();
                let mut got = vec![0u16; out];
                assert!(gather_u16_neon(&plan, &src16, &mut got), "{ctx}");
                let want: Vec<u16> = index.iter().map(|&i| src16[i as usize]).collect();
                assert_eq!(got, want, "u16 {ctx}");
                assert!(!gather_u16_neon(&plan, &src16[..len - 1], &mut got));
                let plan = RowGather::u16_low_byte(&index, len).unwrap();
                let mut got = vec![0u8; out];
                assert!(gather_u16_to_u8_neon(&plan, &src16, &mut got), "{ctx}");
                let want: Vec<u8> = index.iter().map(|&i| src16[i as usize] as u8).collect();
                assert_eq!(got, want, "u16 low byte {ctx}");
                let plan = RowGather::f32(&index, len).unwrap();
                let mut got = vec![0f32; out];
                assert!(gather_f32_neon(&plan, &src32, &mut got), "{ctx}");
                let want: Vec<u32> = index.iter().map(|&i| src32[i as usize].to_bits()).collect();
                let got: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                assert_eq!(got, want, "f32 {ctx}");
                assert!(
                    !gather_u16_neon(&plan, &src16, &mut vec![0u16; out]),
                    "wrong plan"
                );
            }
        }
    }
    assert!(RowGather::u16(&[0, 3], 3).is_none());
}

fn check_class_index<T: ClassIndex + Copy + std::fmt::Debug>(
    values: &[T],
    good: impl Fn(T, usize) -> Option<u16>,
    what: &str,
) {
    let mut rng = Rng(0xc1a5_2026);
    for limit in [1usize, 2, 21, 256, 65536] {
        for n in PIXELS {
            for bad_rate in [0usize, 1, 50] {
                let src: Vec<T> = (0..n)
                    .map(|_| {
                        let v = values[rng.below(values.len())];
                        if good(v, limit).is_some() || rng.below(100) < bad_rate {
                            v
                        } else {
                            values[0]
                        }
                    })
                    .collect();
                let want: Option<Vec<u16>> = src.iter().map(|&v| good(v, limit)).collect();
                let mut got = vec![0u16; n];
                let ok = T::class_index_neon(&src, limit, &mut got);
                assert_eq!(ok, want.is_some(), "{what} limit={limit} n={n} {src:?}");
                if let Some(want) = want {
                    assert_eq!(got, want, "{what} limit={limit} n={n}");
                }
            }
        }
    }
    assert!(!T::class_index_neon(&values[..1], 1 << 17, &mut [0]));
}

#[test]
fn class_index_matches_the_checked_conversion() {
    let fit = |v: i128, limit: usize| u16::try_from(v).ok().filter(|&c| (c as usize) < limit);
    let u8s: Vec<u8> = vec![0, 1, 2, 20, 21, 255];
    check_class_index(&u8s, |v, l| fit(v.into(), l), "u8");
    let u16s: Vec<u16> = vec![0, 1, 20, 255, 256, 65535];
    check_class_index(&u16s, |v, l| fit(v.into(), l), "u16");
    let u32s: Vec<u32> = vec![0, 1, 20, 255, 65535, 65536, u32::MAX, 1 << 31];
    check_class_index(&u32s, |v, l| fit(v.into(), l), "u32");
    let i32s: Vec<i32> = vec![0, 1, 20, 255, 65535, -1, i32::MIN, 65536, i32::MAX];
    check_class_index(&i32s, |v, l| fit(v.into(), l), "i32");
    let i64s: Vec<i64> = vec![0, 1, 20, 65535, -1, i64::MIN, 65536, 1 << 40, i64::MAX];
    check_class_index(&i64s, |v, l| fit(v.into(), l), "i64");
}

fn check_gather_columns<T: Gen>(what: &str) {
    let mut rng = Rng(0x6a7c_2026);
    for classes in [1usize, 2, 3, 17, 150] {
        for width in [1usize, 5, 16, 40] {
            for interleaved in [false, true] {
                let (cs, xs) = if interleaved {
                    (1, classes)
                } else {
                    (width + 3, 1)
                };
                let len = (classes - 1) * cs + (width - 1) * xs + 1;
                let src = planes::<T>(&mut rng, 1, len, len, false);
                let cols: Vec<u32> = (0..rng.below(width) + 1)
                    .map(|_| rng.below(width) as u32)
                    .collect();
                let mut got = vec![0f32; cols.len() * classes];
                assert!(
                    T::gather_columns_neon(&src, cs, xs, classes, &cols, &mut got),
                    "{what}"
                );
                for (j, &x) in cols.iter().enumerate() {
                    for c in 0..classes {
                        let want = src[c * cs + x as usize * xs].to_f32();
                        let g = got[j * classes + c];
                        let ctx =
                            format!("{what} c={classes} w={width} il={interleaved} j={j} c={c}");
                        assert_eq!(g.to_bits(), want.to_bits(), "{ctx}");
                    }
                }
                let far = [width as u32 * 9];
                assert!(!T::gather_columns_neon(
                    &src,
                    cs,
                    xs,
                    classes,
                    &far,
                    &mut got[..classes]
                ));
                assert!(!T::gather_columns_neon(
                    &src,
                    cs,
                    xs,
                    classes,
                    &cols,
                    &mut got[1..]
                ));
            }
        }
    }
}

#[test]
fn gather_columns_match_indexing() {
    check_gather_columns::<u8>("u8");
    check_gather_columns::<i8>("i8");
    check_gather_columns::<u16>("u16");
    check_gather_columns::<i16>("i16");
    check_gather_columns::<f32>("f32");
    check_gather_columns::<f16>("f16");
}

/// The decoder's `Band::build` over corner vectors.
fn candidates_ref(v: [&[f32]; 4]) -> (Vec<u16>, Vec<[f32; 4]>) {
    let mut floor = f32::NEG_INFINITY;
    for c in 0..v[0].len() {
        let k = v.map(|s| s[c]);
        floor = floor.max(k[0].min(k[1]).min(k[2]).min(k[3]));
    }
    let (mut classes, mut corners) = (Vec::new(), Vec::new());
    for c in 0..v[0].len() {
        let k = v.map(|s| s[c]);
        if k[0].max(k[1]).max(k[2]).max(k[3]) >= floor {
            classes.push(c as u16);
            corners.push(k);
        }
    }
    (classes, corners)
}

#[test]
fn cell_candidates_match_the_scalar_build() {
    let mut rng = Rng(0xca4d_2026);
    let specials = f32::specials();
    for n in [0usize, 1, 2, 3, 4, 5, 7, 8, 9, 21, 150, 301] {
        for _ in 0..40 {
            let gen = |rng: &mut Rng| -> Vec<f32> {
                (0..n)
                    .map(|_| match rng.below(16) {
                        0 => specials[rng.below(specials.len())],
                        1 => f32::from_bits(0x7f80_0001),
                        2..=4 => -2.0,
                        _ => rng.unit() * 4.0 - 30.0 + rng.below(3) as f32 * 10.0,
                    })
                    .collect()
            };
            let v = [gen(&mut rng), gen(&mut rng), gen(&mut rng), gen(&mut rng)];
            let refs = [&v[0][..], &v[1][..], &v[2][..], &v[3][..]];
            let (mut classes, mut corners) = (vec![9u16], vec![[1.0f32; 4]]);
            assert!(cell_candidates_neon(
                refs[0],
                refs[1],
                refs[2],
                refs[3],
                &mut classes,
                &mut corners
            ));
            let (want_c, want_v) = candidates_ref(refs);
            assert_eq!(classes[1..], want_c[..], "n={n}");
            let bits = |v: &[[f32; 4]]| v.iter().map(|c| c.map(f32::to_bits)).collect::<Vec<_>>();
            assert_eq!(bits(&corners[1..]), bits(&want_v), "n={n}");
        }
    }
    assert!(!cell_candidates_neon(
        &[0.0],
        &[],
        &[0.0],
        &[0.0],
        &mut Vec::new(),
        &mut Vec::new()
    ));
}
