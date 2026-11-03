// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::*;

/// Deterministic xorshift values.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// A value in `[lo, hi]`, the ends four times as likely as any other value.
    fn int(&mut self, lo: i32, hi: i32) -> i32 {
        match self.next() % 16 {
            0 => lo,
            1 => hi,
            _ => lo + (self.next() % (i64::from(hi) - i64::from(lo) + 1) as u64) as i32,
        }
    }

    fn f32(&mut self, mag: f32) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * mag
    }
}

/// Element construction for the generic tests.
trait FromI32: MaskInt + std::fmt::Debug {
    fn from_i32(v: i32) -> Self;
}

macro_rules! from_i32 {
    ($($t:ty),*) => {$(
        impl FromI32 for $t {
            fn from_i32(v: i32) -> Self {
                v as $t
            }
        }
    )*};
}
from_i32!(i8, u8, i16, u16);

fn ints<T: FromI32>(rng: &mut Rng, n: usize) -> Vec<T> {
    (0..n)
        .map(|_| T::from_i32(rng.int(T::MIN, T::MAX)))
        .collect()
}

fn zero_points<T: MaskInt>(rng: &mut Rng) -> Vec<i32> {
    vec![
        T::MIN,
        T::MAX,
        0,
        rng.int(T::MIN, T::MAX),
        rng.int(T::MIN, T::MAX),
    ]
}

/// The exact `Σ (m − zm)(p − zp)` of pixel `j`.
fn exact<M: MaskInt, P: MaskInt>(c: &[M], zm: i32, p: impl Fn(usize) -> P, zp: i32) -> i64 {
    c.iter()
        .enumerate()
        .map(|(k, &m)| i64::from(m.get() - zm) * i64::from(p(k).get() - zp))
        .sum()
}

const KS: [usize; 18] = [
    0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 24, 31, 32, 33, 48, 64, 100, 257,
];
const NS: [usize; 13] = [0, 1, 2, 3, 4, 5, 7, 8, 13, 15, 16, 17, 40];

fn dotprod() -> bool {
    std::arch::is_aarch64_feature_detected!("dotprod")
}

type Dot8<M, P> = unsafe fn(&[M], i32, &[P], i32, &mut [i32]) -> bool;

fn check_nhwc8<M: FromI32 + MaskInt8, P: FromI32 + MaskInt8>(seed: u64) {
    let mut kernels: Vec<(&str, Dot8<M, P>)> = vec![("neon", mask_dot8_nhwc_neon::<M, P>)];
    if dotprod() {
        kernels.push(("dotprod", mask_dot8_nhwc_dotprod::<M, P>));
    }
    let mut rng = Rng(seed);
    for &k in &KS {
        for &n in &NS {
            let c: Vec<M> = ints(&mut rng, k);
            let px: Vec<P> = ints(&mut rng, n * k);
            for zm in zero_points::<M>(&mut rng) {
                // Coefficients within 100 of `zm` take the folded path, the random ones mostly not.
                let near: Vec<M> = (0..k)
                    .map(|_| M::from_i32((zm + rng.int(-100, 100)).clamp(M::MIN, M::MAX)))
                    .collect();
                for zp in zero_points::<P>(&mut rng) {
                    for (set, c) in [("random", &c), ("near zm", &near)] {
                        let want: Vec<i32> = (0..n)
                            .map(|j| exact(c, zm, |q| px[j * k + q], zp) as i32)
                            .collect();
                        for (name, f) in &kernels {
                            let mut got = vec![i32::MIN; n];
                            // SAFETY: NEON is baseline; dotprod is listed only when detected.
                            assert!(unsafe { f(c, zm, &px, zp, &mut got) });
                            assert_eq!(got, want, "{name} {set} k {k} n {n} zm {zm} zp {zp}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn dot8_nhwc_is_exact() {
    check_nhwc8::<i8, i8>(1);
    check_nhwc8::<u8, u8>(2);
    check_nhwc8::<i8, u8>(3);
    check_nhwc8::<u8, i8>(4);
}

#[test]
fn dot8_at_max_k_fits_i32() {
    let k = MAX_K;
    let kernels: Vec<Dot8<u8, u8>> = if dotprod() {
        vec![mask_dot8_nhwc_neon, mask_dot8_nhwc_dotprod]
    } else {
        vec![mask_dot8_nhwc_neon]
    };
    // Largest magnitude: every (m − zm)(p − zp) at ±255².
    for (cv, zm, pv, zp) in [(255u8, 0, 255u8, 0), (0, 255, 255, 0), (0, 255, 0, 255)] {
        let c = vec![cv; k];
        let px = vec![pv; 5 * k];
        let want = (i64::from(cv) - i64::from(zm)) * (i64::from(pv) - i64::from(zp)) * k as i64;
        assert!(want.unsigned_abs() < 1 << 31);
        for f in &kernels {
            let mut got = [0i32; 5];
            // SAFETY: as in `check_nhwc8`.
            assert!(unsafe { f(&c, zm, &px, zp, &mut got) });
            assert_eq!(got, [want as i32; 5]);
        }
    }
    let c = vec![-128i8; k];
    let px = vec![-128i8; 4 * k];
    let mut got = [0i32; 4];
    // SAFETY: NEON is baseline.
    assert!(unsafe { mask_dot8_nhwc_neon(&c, 127, &px, 127, &mut got) });
    assert_eq!(got, [255 * 255 * k as i32; 4]);
}

#[test]
fn dot8_nchw_at_max_k_fits_i32() {
    let k = MAX_K;
    // 17 pixels: one 16-pixel block plus the overlapping last window.
    let n = 17;
    // Every (m − zm)·p or zp·(m − zm) term at ±255²: the result is ±2 130 674 175.
    for (cv, zm, pv, zp, want) in [
        (0u8, 255, 255u8, 0, -2_130_674_175i64),
        (0, 255, 0, 255, 2_130_674_175),
        (255, 0, 255, 0, 2_130_674_175),
    ] {
        assert_eq!(
            want,
            (i64::from(cv) - i64::from(zm)) * (i64::from(pv) - i64::from(zp)) * k as i64
        );
        let c = vec![cv; k];
        let planes = vec![pv; k * n];
        let mut got = vec![0i32; n];
        // SAFETY: NEON is baseline.
        assert!(unsafe { mask_dot8_nchw_neon(&c, zm, &planes, n, zp, &mut got) });
        assert_eq!(got, vec![want as i32; n]);
    }
    let c = vec![-128i8; k];
    let planes = vec![127i8; k * n];
    let mut got = vec![0i32; n];
    // SAFETY: NEON is baseline.
    assert!(unsafe { mask_dot8_nchw_neon(&c, 127, &planes, n, -128, &mut got) });
    assert_eq!(got, vec![-255 * 255 * k as i32; n]);
}

#[test]
fn invalid_arguments_write_nothing() {
    let c = [1i8; 4];
    let px = [2i8; 8];
    let mut d = [7i32; 2];
    let mut f = [7f32; 2];
    let mut u = [7u8; 2];
    // SAFETY: NEON is baseline.
    unsafe {
        assert!(!mask_dot8_nhwc_neon(&c, 128, &px, 0, &mut d));
        assert!(!mask_dot8_nhwc_neon(&c, 0, &px, -129, &mut d));
        assert!(!mask_dot8_nhwc_neon(&c, 0, &px[..7], 0, &mut d));
        assert!(!mask_dot8_nhwc_neon(&[1u8; 4], -1, &px, 0, &mut d));
        assert!(!mask_dot8_nhwc_neon::<i8, i8>(
            &vec![0i8; MAX_K + 1],
            0,
            &[],
            0,
            &mut []
        ));
        assert!(!mask_dot8_nchw_neon(&c, 0, &px[..5], 2, 0, &mut d));
        assert!(!mask_dot_wide_nhwc_neon(&c, 0, &px, 0, &mut f[..1]));
        assert!(!mask_dot_wide_nchw_neon(
            &[1u16; 4], 70000, &px, 2, 0, &mut f
        ));
        assert!(!mask_dot_f32_nhwc_neon(&[1f32; 4], &px[..6], &mut f));
        assert!(!mask_dot_f32_nchw_neon(&[1f32; 4], &px[..6], 2, &mut f));
        assert!(!mask_sigmoid_u8_i32_neon(&d[..1], 1.0, &mut u));
        assert!(!mask_sigmoid_u8_f32_neon(&f[..1], 1.0, &mut u));
        assert!(!mask_threshold_u8_neon(&[0u8; 3], 1, &mut u));
        assert!(!dequant_nchw_to_nhwc_u8_neon(
            &[0u8; 5],
            1,
            2,
            3,
            Quant::IDENTITY,
            &mut f
        ));
    }
    assert_eq!(d, [7; 2]);
    assert_eq!(f, [7.0; 2]);
    assert_eq!(u, [7; 2]);
}

fn check_wide_nhwc<M: FromI32, P: FromI32>(seed: u64) {
    let mut rng = Rng(seed);
    for &k in &KS {
        for &n in &NS {
            let c: Vec<M> = ints(&mut rng, k);
            let px: Vec<P> = ints(&mut rng, n * k);
            for zm in zero_points::<M>(&mut rng) {
                for zp in zero_points::<P>(&mut rng) {
                    let want: Vec<f32> = (0..n)
                        .map(|j| exact(&c, zm, |q| px[j * k + q], zp) as f32)
                        .collect();
                    let mut got = vec![f32::NAN; n];
                    // SAFETY: NEON is baseline.
                    assert!(unsafe { mask_dot_wide_nhwc_neon(&c, zm, &px, zp, &mut got) });
                    assert_eq!(got, want, "k {k} n {n} zm {zm} zp {zp}");
                }
            }
        }
    }
}

macro_rules! all_pairs {
    ($f:ident) => {
        $f::<i8, i8>(11);
        $f::<i8, u8>(12);
        $f::<i8, i16>(13);
        $f::<i8, u16>(14);
        $f::<u8, i8>(15);
        $f::<u8, u8>(16);
        $f::<u8, i16>(17);
        $f::<u8, u16>(18);
        $f::<i16, i8>(19);
        $f::<i16, u8>(20);
        $f::<i16, i16>(21);
        $f::<i16, u16>(22);
        $f::<u16, i8>(23);
        $f::<u16, u8>(24);
        $f::<u16, i16>(25);
        $f::<u16, u16>(26);
    };
}

#[test]
fn dot_wide_nhwc_is_the_exact_integer_rounded_once() {
    all_pairs!(check_wide_nhwc);
}

#[test]
fn dot_wide_at_max_k_is_exact() {
    let k = MAX_K;
    let c = vec![u16::MAX; k];
    let px = vec![0u16; 3 * k];
    let want = 65535f64 * -65535.0 * k as f64;
    let mut got = [0f32; 3];
    // SAFETY: NEON is baseline.
    assert!(unsafe { mask_dot_wide_nhwc_neon(&c, 0, &px, 65535, &mut got) });
    assert_eq!(got, [want as f32; 3]);
    let mut got = [0f32; 3];
    // SAFETY: NEON is baseline.
    assert!(unsafe { mask_dot_wide_nchw_neon(&c, 0, &px, 3, 65535, &mut got) });
    assert_eq!(got, [want as f32; 3]);
}

/// Crop rows of an NCHW frame: `(stride, offset, n)`, the last ones ending at the frame's end.
fn nchw_rows(w: usize, h: usize) -> Vec<(usize, usize, usize)> {
    let hw = w * h;
    let mut rows = Vec::new();
    for n in [1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 31, 33] {
        if n > w {
            continue;
        }
        rows.push((hw, 0, n));
        rows.push((hw, w - n, n));
        rows.push((hw, hw - n, n));
        rows.push((hw, (h / 2) * w + (w - n) / 2, n));
    }
    rows
}

type Nchw8<M, P> = unsafe fn(&[M], i32, &[P], usize, i32, &mut [i32]) -> bool;

fn check_nchw<M: FromI32, P: FromI32, D: PartialEq + std::fmt::Debug + Copy>(
    seed: u64,
    run: impl Fn(&[M], i32, &[P], usize, i32, &mut [D]) -> bool,
    cast: impl Fn(i64) -> D,
    init: D,
) {
    let mut rng = Rng(seed);
    let (w, h) = (37, 5);
    for k in [0usize, 1, 3, 8, 32, 33] {
        let frame: Vec<P> = ints(&mut rng, k * w * h);
        let c: Vec<M> = ints(&mut rng, k);
        for (stride, off, n) in nchw_rows(w, h) {
            for zm in zero_points::<M>(&mut rng) {
                for zp in zero_points::<P>(&mut rng) {
                    let planes = &frame[off.min(frame.len())..];
                    let want: Vec<D> = (0..n)
                        .map(|j| cast(exact(&c, zm, |q| planes[q * stride + j], zp)))
                        .collect();
                    let mut got = vec![init; n];
                    assert!(run(&c, zm, planes, stride, zp, &mut got));
                    assert_eq!(got, want, "k {k} off {off} n {n} zm {zm} zp {zp}");
                }
            }
        }
    }
}

fn check_nchw8<M: FromI32 + MaskInt8, P: FromI32 + MaskInt8>(seed: u64) {
    let f: Nchw8<M, P> = mask_dot8_nchw_neon::<M, P>;
    // SAFETY: NEON is baseline.
    check_nchw(
        seed,
        |c, zm, p, s, zp, d| unsafe { f(c, zm, p, s, zp, d) },
        |v| v as i32,
        i32::MIN,
    );
}

fn check_nchw_wide<M: FromI32, P: FromI32>(seed: u64) {
    check_nchw(
        seed,
        // SAFETY: NEON is baseline.
        |c, zm, p, s, zp, d| unsafe { mask_dot_wide_nchw_neon::<M, P>(c, zm, p, s, zp, d) },
        |v| v as f32,
        f32::NAN,
    );
}

#[test]
fn dot8_nchw_is_exact() {
    check_nchw8::<i8, i8>(31);
    check_nchw8::<u8, u8>(32);
    check_nchw8::<i8, u8>(33);
    check_nchw8::<u8, i8>(34);
}

#[test]
fn dot_wide_nchw_is_the_exact_integer_rounded_once() {
    all_pairs!(check_nchw_wide);
}

/// Float elements for the float-path tests.
trait FloatCase: MaskFloatIn + std::fmt::Debug {
    fn make(rng: &mut Rng) -> Self;
}

impl FloatCase for f32 {
    fn make(rng: &mut Rng) -> Self {
        rng.f32(4.0)
    }
}
impl FloatCase for f16 {
    fn make(rng: &mut Rng) -> Self {
        f16::from_f32(rng.f32(4.0))
    }
}
impl FloatCase for i8 {
    fn make(rng: &mut Rng) -> Self {
        rng.int(-128, 127) as i8
    }
}
impl FloatCase for u8 {
    fn make(rng: &mut Rng) -> Self {
        rng.int(0, 255) as u8
    }
}

/// `|got − Σ c·p| ≤ terms · 2⁻²⁴ · Σ |c·p|`, against the float64 sum; `Ok` holds the error as a
/// fraction of the bound.
fn within(got: f32, c: &[f32], p: impl Fn(usize) -> f64, terms: f64) -> Result<f64, String> {
    let (mut sum, mut mag) = (0f64, 0f64);
    for (k, &m) in c.iter().enumerate() {
        sum += f64::from(m) * p(k);
        mag += (f64::from(m) * p(k)).abs();
    }
    let bound = terms * f64::from(f32::EPSILON) / 2.0 * mag * 1.0001;
    let err = (f64::from(got) - sum).abs();
    if err <= bound {
        Ok(if bound > 0.0 { err / bound } else { 0.0 })
    } else {
        Err(format!("{got} vs {sum}: error {err} > bound {bound}"))
    }
}

fn check_float_nhwc<P: FloatCase>(seed: u64) {
    let mut worst = 0f64;
    let mut rng = Rng(seed);
    for &k in &KS {
        for &n in &NS {
            let c: Vec<f32> = (0..k).map(|_| rng.f32(2.0)).collect();
            let px: Vec<P> = (0..n * k).map(|_| P::make(&mut rng)).collect();
            let mut got = vec![f32::NAN; n];
            // SAFETY: NEON is baseline.
            assert!(unsafe { mask_dot_f32_nhwc_neon(&c, &px, &mut got) });
            for (j, &g) in got.iter().enumerate() {
                let terms = (k / 4 + 5) as f64;
                let r = within(g, &c, |q| f64::from(px[j * k + q].get()), terms)
                    .unwrap_or_else(|e| panic!("k {k} n {n} pixel {j}: {e}"));
                worst = worst.max(r);
            }
        }
    }
    println!(
        "f32 NHWC dot, {} protos: worst error {worst:.3} of the bound",
        std::any::type_name::<P>()
    );
}

fn check_float_nchw<P: FloatCase>(seed: u64) {
    let mut worst = 0f64;
    let mut rng = Rng(seed);
    let (w, h) = (37, 5);
    for k in [0usize, 1, 3, 8, 32, 33] {
        let frame: Vec<P> = (0..k * w * h).map(|_| P::make(&mut rng)).collect();
        let c: Vec<f32> = (0..k).map(|_| rng.f32(2.0)).collect();
        for (stride, off, n) in nchw_rows(w, h) {
            let planes = &frame[off.min(frame.len())..];
            let mut got = vec![f32::NAN; n];
            // SAFETY: NEON is baseline.
            assert!(unsafe { mask_dot_f32_nchw_neon(&c, planes, stride, &mut got) });
            for (j, &g) in got.iter().enumerate() {
                let p = |q: usize| f64::from(planes[q * stride + j].get());
                let r = within(g, &c, p, k.max(1) as f64)
                    .unwrap_or_else(|e| panic!("k {k} off {off} n {n} pixel {j}: {e}"));
                worst = worst.max(r);
            }
        }
    }
    println!(
        "f32 NCHW dot, {} protos: worst error {worst:.3} of the bound",
        std::any::type_name::<P>()
    );
}

#[test]
fn dot_f32_nhwc_within_bound() {
    check_float_nhwc::<f32>(41);
    check_float_nhwc::<f16>(42);
    check_float_nhwc::<i8>(43);
    check_float_nhwc::<u8>(44);
}

#[test]
fn dot_f32_nchw_within_bound() {
    check_float_nchw::<f32>(51);
    check_float_nchw::<f16>(52);
    check_float_nchw::<i8>(53);
    check_float_nchw::<u8>(54);
}

/// The decoder's scalar mask value.
fn scalar_u8(x: f32) -> u8 {
    ((1.0 / (1.0 + (-x).exp())) * 255.0).round() as u8
}

#[test]
fn sigmoid_u8_within_one_of_scalar() {
    let mut worst = 0i32;
    let mut differ = 0usize;
    let mut total = 0usize;
    for scale in [1.0f32, 0.02, 4.354e-4, 1e-6, 3.7] {
        let src: Vec<i32> = (-200_000..200_000)
            .chain([i32::MIN, i32::MAX, -1, 0, 1])
            .collect();
        let mut got = vec![0u8; src.len()];
        // SAFETY: NEON is baseline.
        assert!(unsafe { mask_sigmoid_u8_i32_neon(&src, scale, &mut got) });
        for (&x, &g) in src.iter().zip(&got) {
            let want = scalar_u8(x as f32 * scale);
            let d = (i32::from(g) - i32::from(want)).abs();
            worst = worst.max(d);
            differ += usize::from(d != 0);
            total += 1;
            assert!(d <= 1, "x {x} scale {scale}: {g} vs {want}");
        }
    }
    let mut f: Vec<f32> = (-400_000..400_000).map(|i| i as f32 * 5e-5).collect();
    f.extend([
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
        0.0,
        -0.0,
        1e-30,
    ]);
    let mut got = vec![0u8; f.len()];
    // SAFETY: NEON is baseline.
    assert!(unsafe { mask_sigmoid_u8_f32_neon(&f, 1.0, &mut got) });
    for (&x, &g) in f.iter().zip(&got) {
        let want = scalar_u8(x);
        let d = (i32::from(g) - i32::from(want)).abs();
        worst = worst.max(d);
        differ += usize::from(d != 0);
        total += 1;
        assert!(d <= 1, "x {x}: {g} vs {want}");
    }
    let mut nan = [7u8; 3];
    // SAFETY: NEON is baseline.
    assert!(unsafe { mask_sigmoid_u8_f32_neon(&[f32::NAN, -f32::NAN, 0.0], 1.0, &mut nan) });
    assert_eq!(nan, [scalar_u8(f32::NAN), scalar_u8(-f32::NAN), 128]);
    for x in [f32::INFINITY, f32::NEG_INFINITY] {
        let mut g = [0u8];
        // SAFETY: NEON is baseline.
        assert!(unsafe { mask_sigmoid_u8_f32_neon(&[x], 1.0, &mut g) });
        assert_eq!(g[0], scalar_u8(x));
    }
    println!("sigmoid u8: worst {worst}, {differ} of {total} differ from scalar");
}

#[test]
fn threshold_is_exact() {
    let src: Vec<u8> = (0..=255).chain(0..=40).collect();
    for t in [0u8, 1, 127, 128, 254, 255] {
        let mut got = vec![9u8; src.len()];
        // SAFETY: NEON is baseline.
        assert!(unsafe { mask_threshold_u8_neon(&src, t, &mut got) });
        let want: Vec<u8> = src.iter().map(|&x| u8::from(x >= t)).collect();
        assert_eq!(got, want);
    }
}

type DequantNchw<T> = unsafe fn(&[T], usize, usize, usize, Quant, &mut [f32]) -> bool;

fn check_dequant_nchw<T: FromI32 + LoadF32>(seed: u64, run: DequantNchw<T>) {
    let mut rng = Rng(seed);
    for (h, w, c) in [
        (1, 1, 1),
        (2, 16, 4),
        (3, 17, 5),
        (4, 33, 32),
        (5, 160, 32),
        (2, 15, 3),
    ] {
        let src: Vec<T> = ints(&mut rng, h * w * c);
        for q in [
            Quant::IDENTITY,
            Quant::new(0.0208, -115),
            Quant::new(3.1e-3, 40000),
        ] {
            let mut got = vec![f32::NAN; src.len()];
            // SAFETY: NEON is baseline.
            assert!(unsafe { run(&src, h, w, c, q, &mut got) });
            let want: Vec<u32> = (0..h * w * c)
                .map(|i| {
                    let (px, ch) = (i / c, i % c);
                    let v = src[ch * h * w + px].get() as f32;
                    ((v - q.zero_point as f32) * q.scale).to_bits()
                })
                .collect();
            let got: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want, "h {h} w {w} c {c} {q:?}");
        }
    }
}

#[test]
fn dequant_nchw_to_nhwc_is_bit_exact() {
    check_dequant_nchw::<i8>(61, dequant_nchw_to_nhwc_i8_neon);
    check_dequant_nchw::<u8>(62, dequant_nchw_to_nhwc_u8_neon);
    check_dequant_nchw::<i16>(63, dequant_nchw_to_nhwc_i16_neon);
    check_dequant_nchw::<u16>(64, dequant_nchw_to_nhwc_u16_neon);
}
