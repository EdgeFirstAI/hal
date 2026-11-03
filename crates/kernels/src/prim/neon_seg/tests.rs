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
}

/// Runs a `u8`-producing kernel into a 0xAA-filled output; `None` when it reports failure (and
/// then the output must be untouched).
fn run(n: usize, f: impl FnOnce(&mut [MaybeUninit<u8>]) -> bool) -> Option<Vec<u8>> {
    let mut out = vec![MaybeUninit::new(0xaa_u8); n];
    let ok = f(&mut out);
    // SAFETY: every element was initialised.
    let out: Vec<u8> = out
        .into_iter()
        .map(|x| unsafe { x.assume_init() })
        .collect();
    if !ok {
        assert!(out.iter().all(|&x| x == 0xaa));
        return None;
    }
    Some(out)
}

/// Lengths covering every tail of a 16-element step.
const LENGTHS: [usize; 12] = [0, 1, 2, 7, 15, 16, 17, 31, 32, 33, 100, 257];

/// Each integer type's top byte equals the decoder's formulas over every 8- and 16-bit value,
/// random and extreme 32-bit values, and every tail length; a length mismatch writes nothing.
#[test]
fn seg_to_u8_matches_the_decoder_formulas() {
    fn check<T: SegToU8 + std::fmt::Debug>(src: &[T], want: impl Fn(T) -> u8) {
        let got = run(src.len(), |d| T::seg_to_u8_neon(src, d)).unwrap();
        for (k, (&x, g)) in src.iter().zip(got).enumerate() {
            assert_eq!(g, want(x), "{x:?} at {k}");
        }
        for &n in LENGTHS.iter().filter(|&&n| n <= src.len()) {
            let got = run(n, |d| T::seg_to_u8_neon(&src[..n], d)).unwrap();
            assert!(got.iter().zip(&src[..n]).all(|(&g, &x)| g == want(x)));
        }
        if !src.is_empty() {
            assert!(run(src.len() - 1, |d| T::seg_to_u8_neon(src, d)).is_none());
        }
    }
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let i8s: Vec<i8> = (i8::MIN..=i8::MAX).collect();
    check(&i8s, |x| (x as i16 + 128) as u8);
    let u16s: Vec<u16> = (0..=u16::MAX).collect();
    check(&u16s, |x| (x >> 8) as u8);
    let i16s: Vec<i16> = (i16::MIN..=i16::MAX).collect();
    check(&i16s, |x| ((x as i32 + 32768) >> 8) as u8);
    let mut u32s: Vec<u32> = (0..70_000).map(|_| rng.next() as u32).collect();
    u32s.extend([0, 1, u32::MAX, 1 << 24, (1 << 24) - 1, 1 << 31]);
    check(&u32s, |x| (x >> 24) as u8);
    let mut i32s: Vec<i32> = (0..70_000).map(|_| rng.next() as i32).collect();
    i32s.extend([0, -1, i32::MIN, i32::MAX, 1 << 24, -(1 << 24)]);
    check(&i32s, |x| ((x as i64 + 2147483648) >> 24) as u8);
}

/// The decoder's scalar normalisation: `ndarray_stats` min/max (an error, so 0 and 255, on a
/// NaN or an empty map), then `((x − min) / (max − min) · 255) as u8` in `T` arithmetic.
fn normalize_scalar<T>(src: &[T], zero: T, full: T, to_u8: impl Fn(T) -> u8) -> Vec<u8>
where
    T: Copy + PartialOrd + std::ops::Sub<Output = T> + std::ops::Div<Output = T>,
    T: std::ops::Mul<Output = T>,
{
    let fold = |pick: std::cmp::Ordering| {
        let first = src.first()?;
        src.iter()
            .try_fold(*first, |acc, &x| match x.partial_cmp(&acc)? {
                o if o == pick => Some(x),
                _ => Some(acc),
            })
    };
    let max = fold(std::cmp::Ordering::Greater).unwrap_or(full);
    let min = fold(std::cmp::Ordering::Less).unwrap_or(zero);
    src.iter()
        .map(|&x| to_u8((x - min) / (max - min) * full))
        .collect()
}

/// Maps: random, with NaN, infinities, signed zeros, constant, one element, two values.
fn maps(rng: &mut Rng) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    for &n in &LENGTHS {
        let unit = |rng: &mut Rng| (rng.next() >> 40) as f32 / (1u64 << 24) as f32;
        out.push((0..n).map(|_| (unit(rng) - 0.3) * 20.0).collect::<Vec<_>>());
        out.push((0..n).map(|_| unit(rng) * 1e-3).collect());
        let mut special: Vec<f32> = (0..n).map(|_| unit(rng) * 300.0 - 10.0).collect();
        if n > 3 {
            special[n / 2] = f32::NAN;
        }
        out.push(special);
        let mut zeros: Vec<f32> = (0..n)
            .map(|k| if k % 2 == 0 { -0.0 } else { 0.0 })
            .collect();
        out.push(zeros.clone());
        if n > 2 {
            zeros[1] = 3.0;
        }
        out.push(zeros.clone());
        if n > 2 {
            zeros[1] = -3.0;
        }
        out.push(zeros);
        out.push(vec![1.5; n]);
        let mut inf: Vec<f32> = (0..n).map(|_| unit(rng)).collect();
        if n > 2 {
            inf[0] = f32::INFINITY;
            inf[n - 1] = f32::NEG_INFINITY;
        }
        out.push(inf);
    }
    out
}

/// f32 maps normalize to the scalar result bit for bit; a length mismatch writes nothing.
#[test]
fn seg_normalize_f32_matches_the_scalar_code() {
    let mut rng = Rng(0x0bad_cafe_dead_beef);
    for map in maps(&mut rng) {
        let want = normalize_scalar(&map, 0.0f32, 255.0, |x| x as u8);
        let got = run(map.len(), |d| f32::seg_normalize_neon(&map, d)).unwrap();
        assert_eq!(got, want, "{map:?}");
        if !map.is_empty() {
            assert!(run(map.len() + 1, |d| f32::seg_normalize_neon(&map, d)).is_none());
        }
    }
}

/// f16 maps normalize to `half` arithmetic bit for bit, emulated and (with FEAT_FP16) native,
/// over the random maps and maps holding every finite f16 of a range.
#[test]
fn seg_normalize_f16_matches_half_arithmetic() {
    let fp16 = std::arch::is_aarch64_feature_detected!("fp16");
    let mut rng = Rng(0x5eed_5eed_5eed_5eed);
    let mut all = maps(&mut rng)
        .into_iter()
        .map(|m| m.into_iter().map(f16::from_f32).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let finite: Vec<f16> = (0..=u16::MAX)
        .map(f16::from_bits)
        .filter(|x| x.is_finite())
        .collect();
    all.push(finite.clone());
    all.push(
        finite
            .iter()
            .copied()
            .filter(|x| x.to_f32().abs() < 1.0)
            .collect(),
    );
    all.push(
        finite
            .iter()
            .copied()
            .filter(|x| x.to_f32() >= -3.0 && x.to_f32() <= 1000.0)
            .collect(),
    );
    for map in all {
        let want = normalize_scalar(&map, f16::ZERO, f16::from_f32(255.0), |x| x.to_f32() as u8);
        let got = run(map.len(), |d| f16::seg_normalize_neon(&map, d)).unwrap();
        assert_eq!(got, want, "emulated, {} values", map.len());
        if fp16 {
            // SAFETY: FEAT_FP16 was detected.
            let got = run(map.len(), |d| unsafe {
                seg_normalize_f16_neon_fp16(&map, d)
            });
            assert_eq!(got.unwrap(), want, "native, {} values", map.len());
        }
    }
    if fp16 {
        let map = [f16::ONE; 3];
        // SAFETY: FEAT_FP16 was detected.
        assert!(run(2, |d| unsafe { seg_normalize_f16_neon_fp16(&map, d) }).is_none());
    } else {
        eprintln!("WARNING: SKIPPED: native FP16 normalize, no FEAT_FP16");
    }
}

/// The first channel holding the largest score, as `argminmax` and the decoder pick it.
fn argmax_scalar(src: &[u8], channels: usize) -> Vec<u8> {
    src.chunks_exact(channels)
        .map(|px| {
            let mut best = 0;
            for (c, &v) in px.iter().enumerate() {
                if v > px[best] {
                    best = c;
                }
            }
            best as u8
        })
        .collect()
}

/// The per-pixel argmax equals the scalar first-max for 1–20, 31–33, 64, 255–257 and 300
/// channels over every tail of 16 pixels, few-valued (ties everywhere) and full-range scores;
/// zero channels or a pixel count mismatch write nothing.
#[test]
fn channel_argmax_matches_the_scalar_first_max() {
    let mut rng = Rng(0xfeed_f00d_1234_5678);
    let channels: Vec<usize> = (1..=20)
        .chain([31, 32, 33, 64, 255, 256, 257, 300])
        .collect();
    for &c in &channels {
        for &pixels in &LENGTHS {
            for few in [true, false] {
                let src: Vec<u8> = (0..pixels * c)
                    .map(|_| {
                        if few {
                            [0u8, 7, 255][(rng.next() % 3) as usize]
                        } else {
                            rng.next() as u8
                        }
                    })
                    .collect();
                let want = argmax_scalar(&src, c);
                let got = run(pixels, |d| channel_argmax_u8_neon(&src, c, d)).unwrap();
                assert_eq!(got, want, "{c} channels, {pixels} pixels, few {few}");
            }
        }
    }
    let src = [3u8; 12];
    assert!(run(4, |d| channel_argmax_u8_neon(&src, 0, d)).is_none());
    assert!(run(5, |d| channel_argmax_u8_neon(&src, 3, d)).is_none());
    assert!(run(3, |d| channel_argmax_u8_neon(&src[..11], 4, d)).is_none());
}
