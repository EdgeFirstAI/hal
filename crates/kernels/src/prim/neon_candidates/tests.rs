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

/// Scores drawn either from a few values (ties everywhere) or from the whole type.
trait Score: Copy + PartialOrd + std::fmt::Debug {
    fn draw(rng: &mut Rng, narrow: bool) -> Self;
    fn bits(self) -> u64;
}

macro_rules! int_score {
    ($($t:ty),*) => {$(
        impl Score for $t {
            fn draw(rng: &mut Rng, narrow: bool) -> Self {
                if narrow {
                    (rng.next() % 5) as $t
                } else {
                    rng.next() as $t
                }
            }
            fn bits(self) -> u64 {
                self as u64
            }
        }
    )*};
}
int_score!(i8, u8, i16, u16);

/// Float specials: NaNs (quiet, signalling, negative), zeros and infinities.
const SPECIALS: [f32; 7] = [
    f32::NAN,
    -f32::NAN,
    0.0,
    -0.0,
    f32::INFINITY,
    f32::NEG_INFINITY,
    0.5,
];

impl Score for f32 {
    fn draw(rng: &mut Rng, narrow: bool) -> Self {
        match rng.next() % 64 {
            0 => SPECIALS[(rng.next() % 7) as usize],
            1 => f32::from_bits(0x7f80_0001),
            _ if narrow => [0.25f32, 0.5, -0.0, 0.0][(rng.next() % 4) as usize],
            _ => f32::from_bits(rng.next() as u32 & 0xbfff_ffff),
        }
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl Score for f16 {
    fn draw(rng: &mut Rng, narrow: bool) -> Self {
        match rng.next() % 64 {
            0 => f16::from_f32(SPECIALS[(rng.next() % 7) as usize]),
            1 => f16::from_bits(0x7c01),
            _ if narrow => f16::from_f32([0.25f32, 0.5, -0.0, 0.0][(rng.next() % 4) as usize]),
            _ => f16::from_bits(rng.next() as u16 & 0xbfff),
        }
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

/// The scalar fold the kernels reproduce.
fn fold<T: Score>(scores: impl Iterator<Item = T>) -> Option<(T, usize)> {
    let mut best: Option<(T, usize)> = None;
    for (c, s) in scores.enumerate() {
        best = match best {
            Some((b, k)) if b > s => Some((b, k)),
            _ => Some((s, c)),
        };
    }
    best
}

fn column_reference<T: Score>(cols: &[Vec<T>], threshold: T) -> Vec<(usize, usize)> {
    let n = cols.first().map_or(0, Vec::len);
    (0..n)
        .filter_map(|i| {
            let (s, c) = fold(cols.iter().map(|col| col[i]))?;
            (s.partial_cmp(&threshold) != Some(std::cmp::Ordering::Less)).then_some((i, c))
        })
        .collect()
}

const LENGTHS: [usize; 18] = [
    0, 1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 65, 100, 129, 300,
];
const CLASSES: [usize; 6] = [1, 2, 3, 17, 80, 91];

/// Every column, row and threshold-position case of one kernel set against the scalar fold.
macro_rules! check_kernels {
    ($t:ty, $lanes:ty, $column:path, $row:path, $ge:path) => {{
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut cases = 0usize;
        for narrow in [true, false] {
            for &n in &LENGTHS {
                for &nc in &CLASSES {
                    let cols: Vec<Vec<$t>> = (0..nc)
                        .map(|_| (0..n).map(|_| <$t>::draw(&mut rng, narrow)).collect())
                        .collect();
                    let threshold = <$t>::draw(&mut rng, narrow);
                    let views: Vec<&[$t]> = cols.iter().map(Vec::as_slice).collect();
                    let mut got = Vec::new();
                    // SAFETY: the caller of `check_kernels!` checked the kernel's CPU feature.
                    let ran = unsafe { $column(&views, threshold, |i, c| got.push((i, c))) };
                    assert!(ran, "column n={n} nc={nc}");
                    assert_eq!(
                        got,
                        column_reference(&cols, threshold),
                        "column n={n} nc={nc}"
                    );

                    let row: Vec<$t> = cols.iter().flatten().copied().collect();
                    // SAFETY: as above.
                    let r = unsafe { $row(&row) };
                    let expected = fold(row.iter().copied());
                    let has_nan = row.iter().any(|s| s.partial_cmp(s).is_none());
                    match r {
                        Some((s, i)) => {
                            let (es, ei) = expected.unwrap();
                            assert_eq!((s.bits(), i), (es.bits(), ei), "row len {}", row.len());
                        }
                        None => assert!(
                            row.len() < <$lanes as Lanes>::N || has_nan,
                            "row len {}",
                            row.len()
                        ),
                    }

                    let mut hits = Vec::new();
                    // SAFETY: as above.
                    unsafe { $ge(&row, threshold, |i| hits.push(i)) };
                    let expected: Vec<usize> =
                        (0..row.len()).filter(|&i| row[i] >= threshold).collect();
                    assert_eq!(hits, expected, "ge len {}", row.len());
                    cases += 1;
                }
            }
        }
        assert_eq!(cases, 2 * LENGTHS.len() * CLASSES.len());
    }};
}

#[test]
fn i8_kernels_match_the_scalar_fold() {
    check_kernels!(
        i8,
        I8,
        column_argmax_i8_neon,
        row_argmax_i8_neon,
        ge_indices_i8_neon
    );
}

#[test]
fn u8_kernels_match_the_scalar_fold() {
    check_kernels!(
        u8,
        U8,
        column_argmax_u8_neon,
        row_argmax_u8_neon,
        ge_indices_u8_neon
    );
}

#[test]
fn i16_kernels_match_the_scalar_fold() {
    check_kernels!(
        i16,
        I16,
        column_argmax_i16_neon,
        row_argmax_i16_neon,
        ge_indices_i16_neon
    );
}

#[test]
fn u16_kernels_match_the_scalar_fold() {
    check_kernels!(
        u16,
        U16,
        column_argmax_u16_neon,
        row_argmax_u16_neon,
        ge_indices_u16_neon
    );
}

#[test]
fn f32_kernels_match_the_scalar_fold() {
    check_kernels!(
        f32,
        F32,
        column_argmax_f32_neon,
        row_argmax_f32_neon,
        ge_indices_f32_neon
    );
}

#[test]
fn f16_kernels_match_the_scalar_fold() {
    check_kernels!(
        f16,
        F16Widened,
        column_argmax_f16_neon,
        row_argmax_f16_neon,
        ge_indices_f16_neon
    );
}

#[test]
fn f16_fp16_kernels_match_the_scalar_fold() {
    if !std::arch::is_aarch64_feature_detected!("fp16") {
        crate::test_support::report_skip("f16_fp16_kernels_match_the_scalar_fold needs FEAT_FP16");
        return;
    }
    check_kernels!(
        f16,
        F16Native,
        column_argmax_f16_neon_fp16,
        row_argmax_f16_neon_fp16,
        ge_indices_f16_neon_fp16
    );
}

/// Every f16 pair through the native and widened compares, against `half`'s ordering.
#[test]
fn f16_compares_match_half_on_every_pair_of_a_sweep() {
    let fp16 = std::arch::is_aarch64_feature_detected!("fp16");
    if !fp16 {
        crate::test_support::report_skip(
            "native half of f16_compares_match_half_on_every_pair_of_a_sweep needs FEAT_FP16",
        );
    }
    let values: Vec<u16> = (0..=u16::MAX)
        .step_by(61)
        .chain([
            0x0000, 0x8000, 0x7c00, 0xfc00, 0x7c01, 0x7e00, 0xfe00, 0x0001, 0x8001,
        ])
        .collect();
    for &a in &values {
        let av = f16::from_bits(a);
        for chunk in values.chunks(8) {
            let mut b = [0u16; 8];
            b[..chunk.len()].copy_from_slice(chunk);
            // SAFETY: `b` holds 8 values.
            let (va, vb) = unsafe { (F16Widened::splat(av), F16Widened::load(b.as_ptr().cast())) };
            let gt = F16Widened::bits(F16Widened::gt(va, vb));
            for (k, &bk) in chunk.iter().enumerate() {
                assert_eq!(
                    (gt >> (8 * k)) & 1 != 0,
                    av > f16::from_bits(bk),
                    "{a:#x} > {bk:#x}"
                );
            }
            if fp16 {
                // SAFETY: as above; FEAT_FP16 was detected.
                let (na, nb) =
                    unsafe { (F16Native::splat(av), F16Native::load(b.as_ptr().cast())) };
                let gt = F16Native::bits(F16Native::gt(na, nb));
                for (k, &bk) in chunk.iter().enumerate() {
                    assert_eq!(
                        (gt >> (8 * k)) & 1 != 0,
                        av > f16::from_bits(bk),
                        "native {a:#x} > {bk:#x}"
                    );
                }
            }
        }
    }
}

#[test]
fn ties_go_to_the_last_class_and_the_last_position() {
    let cols: Vec<Vec<i8>> = (0..5).map(|_| vec![7i8; 70]).collect();
    let views: Vec<&[i8]> = cols.iter().map(Vec::as_slice).collect();
    let mut got = Vec::new();
    // SAFETY: NEON is part of the aarch64 baseline.
    assert!(unsafe { column_argmax_i8_neon(&views, 7, |i, c| got.push((i, c))) });
    assert_eq!(got, (0..70).map(|i| (i, 4)).collect::<Vec<_>>());
    let row = [0.5f32, -0.0, 0.5, 0.0, 0.5, 0.25, -1.0, 0.5, 0.0, 0.1];
    // SAFETY: as above.
    assert_eq!(unsafe { row_argmax_f32_neon(&row) }, Some((0.5, 7)));
    let zeros = [0.0f32, -0.0, 0.0, -0.0, -0.0];
    // SAFETY: as above.
    let (s, i) = unsafe { row_argmax_f32_neon(&zeros) }.unwrap();
    assert_eq!((s.to_bits(), i), ((-0.0f32).to_bits(), 4));
}

#[test]
fn a_nan_best_is_kept_and_a_nan_row_is_left_to_the_caller() {
    let cols = [vec![0.1f32; 9], vec![f32::NAN; 9], vec![0.2f32; 9]];
    let views: Vec<&[f32]> = cols.iter().map(Vec::as_slice).collect();
    let mut got = Vec::new();
    // SAFETY: NEON is part of the aarch64 baseline.
    assert!(unsafe { column_argmax_f32_neon(&views, 0.15, |i, c| got.push((i, c))) });
    assert_eq!(got, (0..9).map(|i| (i, 2)).collect::<Vec<_>>());
    let cols = [vec![0.1f32; 9], vec![f32::NAN; 9]];
    let views: Vec<&[f32]> = cols.iter().map(Vec::as_slice).collect();
    got.clear();
    // SAFETY: as above.
    assert!(unsafe { column_argmax_f32_neon(&views, 0.9, |i, c| got.push((i, c))) });
    assert_eq!(got, (0..9).map(|i| (i, 1)).collect::<Vec<_>>());
    let mut row = [0.5f32; 20];
    row[3] = f32::NAN;
    // SAFETY: as above.
    assert_eq!(unsafe { row_argmax_f32_neon(&row) }, None);
}

#[test]
fn invalid_columns_are_rejected() {
    let a = [1u8; 16];
    let b = [1u8; 15];
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        assert!(!column_argmax_u8_neon(&[&a, &b], 0, |_, _| panic!(
            "reported"
        )));
        assert!(!column_argmax_u8_neon(&[], 0, |_, _| panic!("reported")));
        let many: Vec<&[u8]> = vec![&a; 257];
        assert!(!column_argmax_u8_neon(&many, 0, |_, _| panic!("reported")));
        let most: Vec<&[u8]> = vec![&a; 256];
        let mut got = Vec::new();
        assert!(column_argmax_u8_neon(&most, 0, |i, c| got.push((i, c))));
        assert_eq!(got, (0..16).map(|i| (i, 255)).collect::<Vec<_>>());
    }
}
