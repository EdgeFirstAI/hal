// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::prim::sigmoid::fast_sigmoid;

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

/// `fast_sigmoid4` of `xs` (a multiple of four long), lane by lane.
fn vector(xs: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; xs.len()];
    for (x, o) in xs.chunks_exact(4).zip(out.chunks_exact_mut(4)) {
        // SAFETY: four f32 each; NEON is part of the aarch64 baseline.
        unsafe { vst1q_f32(o.as_mut_ptr(), fast_sigmoid4(vld1q_f32(x.as_ptr()))) };
    }
    out
}

/// `fast_sigmoid4` against `fast_sigmoid` on every bit pattern of `patterns`.
fn sweep(patterns: impl Iterator<Item = u32>) {
    let mut xs = Vec::with_capacity(1 << 16);
    let check = |xs: &mut Vec<f32>| {
        xs.resize(xs.len().next_multiple_of(4), 0.0);
        for (x, y) in xs.iter().zip(vector(xs)) {
            assert_eq!(
                y.to_bits(),
                fast_sigmoid(*x).to_bits(),
                "{x:e} ({:#010x})",
                x.to_bits()
            );
        }
        xs.clear();
    };
    for bits in patterns {
        xs.push(f32::from_bits(bits));
        if xs.len() == 1 << 16 {
            check(&mut xs);
        }
    }
    check(&mut xs);
}

/// `fast_sigmoid4` equals `fast_sigmoid` on every 127th f32 bit pattern, every value within
/// 2^16 ulps of ±80, ±88 and ±0, and the specials.
#[test]
fn fast_sigmoid4_matches_the_scalar_function() {
    sweep((0..=u32::MAX).step_by(127));
    for edge in [80.0f32, -80.0, 88.0, -88.0, 0.0, -0.0] {
        let b = edge.to_bits();
        sweep(b.saturating_sub(1 << 16)..=b.saturating_add(1 << 16));
    }
    let specials = [
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7f80_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        f32::from_bits(1),
        f32::MAX,
        f32::MIN,
        80.0,
        -80.0,
    ];
    sweep(specials.iter().map(|x| x.to_bits()));
}

/// Every f32 bit pattern; slow, run by hand on the board.
#[test]
#[ignore]
fn fast_sigmoid4_matches_the_scalar_function_exhaustively() {
    sweep(0..=u32::MAX);
}

/// A split input with its scalar `as f32`.
trait Input: SplitIn + std::fmt::Debug {
    fn draw(rng: &mut Rng) -> Self;
    fn widen(self) -> f32;
}

macro_rules! int_input {
    ($($t:ty),*) => {$(
        impl Input for $t {
            fn draw(rng: &mut Rng) -> Self {
                rng.next() as $t
            }
            fn widen(self) -> f32 {
                self as f32
            }
        }
    )*};
}
int_input!(i8, u8, i16, u16);

impl Input for f32 {
    fn draw(rng: &mut Rng) -> Self {
        match rng.next() % 61 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => f32::NEG_INFINITY,
            3 => 85.0,
            4 => -0.0,
            _ => ((rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 24.0,
        }
    }
    fn widen(self) -> f32 {
        self
    }
}

impl Input for f16 {
    fn draw(rng: &mut Rng) -> Self {
        f16::from_f32(f32::draw(rng))
    }
    fn widen(self) -> f32 {
        self.to_f32()
    }
}

/// The decoder's scalar split decode of whole layer `src`, verbatim apart from the dequantize
/// closure.
fn scalar<S: Input>(
    src: &[S],
    dequant: Option<(f32, Option<f32>)>,
    height: usize,
    width: usize,
    anchors: &[[f32; 2]],
    nc: usize,
) -> (Vec<f32>, Vec<f32>) {
    let p: Vec<f32> = src
        .iter()
        .map(|&x| {
            let x = x.widen();
            fast_sigmoid(match dequant {
                None => x,
                Some((s, None)) => x * s,
                Some((s, Some(b))) => x * s + b,
            })
        })
        .collect();
    let na = anchors.len();
    let div_width = 1.0 / width as f32;
    let div_height = 1.0 / height as f32;
    let mut grid = Vec::new();
    for y in 0..height {
        for x in 0..width {
            for _ in 0..na {
                grid.push(x as f32 - 0.5);
                grid.push(y as f32 - 0.5);
            }
        }
    }
    let (mut bboxes, mut bscores) = (Vec::new(), Vec::new());
    for ((p, g), anchor) in p
        .chunks_exact(nc + 5)
        .zip(grid.as_chunks::<2>().0)
        .zip(anchors.iter().cycle())
    {
        let (x, y) = (p[0], p[1]);
        let x = (x * 2.0 + g[0]) * div_width;
        let y = (y * 2.0 + g[1]) * div_height;
        let (w, h) = (p[2], p[3]);
        let w = w * w * 4.0 * anchor[0];
        let h = h * h * 4.0 * anchor[1];
        bboxes.extend([x, y, w, h]);
        if nc == 1 {
            bscores.push(p[4]);
        } else {
            let obj = p[4];
            bscores.extend(p[5..].iter().map(|x| *x * obj));
        }
    }
    (bboxes, bscores)
}

/// The kernel over rows `first..first + rows` of `src`, into NaN-filled outputs; `None` when it
/// rejects them (and then the outputs must be untouched).
fn kernel<S: Input>(
    src: &[S],
    dequant: Option<(f32, Option<f32>)>,
    grid: SplitGrid<'_>,
    rows: usize,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let row = grid.classes + 5;
    let src = &src[grid.first_row * row..(grid.first_row + rows) * row];
    let mut boxes = vec![MaybeUninit::new(f32::NAN); rows * 4];
    let mut scores = vec![MaybeUninit::new(f32::NAN); rows * grid.classes];
    let ok = S::modelpack_split_rows_neon(src, dequant, &grid, &mut boxes, &mut scores);
    // SAFETY: every element was initialised with NaN.
    let read = |v: Vec<MaybeUninit<f32>>| v.into_iter().map(|x| unsafe { x.assume_init() });
    let (boxes, scores): (Vec<f32>, Vec<f32>) = (read(boxes).collect(), read(scores).collect());
    if !ok {
        assert!(boxes.iter().chain(&scores).all(|x| x.is_nan()));
        return None;
    }
    Some((boxes, scores))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Whole layers and row ranges of every input type equal the scalar decode bit for bit, for
/// grids up to 40 × 40 × 3 anchors, 1, 2, 3, 80 and 91 classes, with and without dequantize.
fn check_type<S: Input>(seed: u64) {
    let mut rng = Rng(seed);
    let anchors = [[0.13f32, 0.21], [0.25, 0.21], [0.23, 0.35]];
    let quants: [Option<(f32, Option<f32>)>; 4] = [
        None,
        Some((0.0855, None)),
        Some((0.0993, Some(-183.0 * 0.0993))),
        Some((0.002, Some(0.5))),
    ];
    for (h, w, na, nc) in [
        (1, 1, 1, 1),
        (1, 1, 1, 2),
        (3, 5, 3, 1),
        (9, 15, 3, 1),
        (4, 7, 2, 3),
        (17, 30, 3, 80),
        (40, 40, 3, 80),
        (5, 3, 3, 91),
    ] {
        let row = nc + 5;
        let total = h * w * na;
        let src: Vec<S> = (0..total * row).map(|_| S::draw(&mut rng)).collect();
        for dq in quants {
            let (want_b, want_s) = scalar(&src, dq, h, w, &anchors[..na], nc);
            let grid = |first_row| SplitGrid {
                width: w,
                height: h,
                anchors: &anchors[..na],
                classes: nc,
                first_row,
            };
            let (b, s) = kernel(&src, dq, grid(0), total).unwrap();
            let what = format!("{} {h}x{w}x{na}x{nc} {dq:?}", std::any::type_name::<S>());
            assert_eq!(bits(&b), bits(&want_b), "boxes {what}");
            assert_eq!(bits(&s), bits(&want_s), "scores {what}");
            for (first, rows) in [(1, total - 1), (total / 3, total / 2), (total - 1, 1)] {
                if rows == 0 {
                    continue;
                }
                let (b, s) = kernel(&src, dq, grid(first), rows).unwrap();
                assert_eq!(
                    bits(&b),
                    bits(&want_b[first * 4..(first + rows) * 4]),
                    "boxes {what} rows {first}+{rows}"
                );
                assert_eq!(
                    bits(&s),
                    bits(&want_s[first * nc..(first + rows) * nc]),
                    "scores {what} rows {first}+{rows}"
                );
            }
        }
    }
}

#[test]
fn split_rows_match_the_scalar_decode() {
    check_type::<i8>(1);
    check_type::<u8>(2);
    check_type::<i16>(3);
    check_type::<u16>(4);
    check_type::<f32>(5);
    check_type::<f16>(6);
}

/// Bad lengths, rows past the grid, no anchors, no classes, an empty grid or a row longer than
/// `MAX_ROW`: rejected with nothing written.
#[test]
fn invalid_arguments_write_nothing() {
    let anchors = [[1.0f32, 1.0]];
    let src = vec![0u8; 2 * 7 * 7];
    let base = SplitGrid {
        width: 7,
        height: 2,
        anchors: &anchors,
        classes: 2,
        first_row: 0,
    };
    assert!(kernel(&src, None, base, 14).is_some());
    let bad = [
        SplitGrid {
            first_row: 1,
            ..base
        },
        SplitGrid {
            anchors: &[],
            ..base
        },
        SplitGrid { width: 0, ..base },
        SplitGrid { height: 1, ..base },
    ];
    for grid in bad {
        let mut boxes = vec![MaybeUninit::new(f32::NAN); 14 * 4];
        let mut scores = vec![MaybeUninit::new(f32::NAN); 14 * 2];
        assert!(
            !u8::modelpack_split_rows_neon(&src, None, &grid, &mut boxes, &mut scores),
            "{grid:?}"
        );
    }
    let mut boxes = vec![MaybeUninit::new(f32::NAN); 14 * 4];
    let mut scores = [MaybeUninit::new(f32::NAN); 14 * 2 + 1];
    let rejected = [
        (&src[..src.len() - 1], 14 * 4, 14 * 2, 2),
        (&src[..], 14 * 4 - 1, 14 * 2, 2),
        (&src[..], 14 * 4, 14 * 2 + 1, 2),
        (&src[..14 * 5], 14 * 4, 0, 0),
    ];
    for (s, nb, ns, classes) in rejected {
        let grid = SplitGrid { classes, ..base };
        assert!(!u8::modelpack_split_rows_neon(
            s,
            None,
            &grid,
            &mut boxes[..nb],
            &mut scores[..ns]
        ));
    }
    let long = vec![0u8; MAX_ROW - 4];
    let grid = SplitGrid {
        width: 1,
        height: 1,
        classes: MAX_ROW - 4,
        ..base
    };
    let mut scores = vec![MaybeUninit::new(f32::NAN); MAX_ROW - 4];
    assert!(!u8::modelpack_split_rows_neon(
        &long,
        None,
        &grid,
        &mut boxes[..4],
        &mut scores
    ));
    let at_max = vec![0u8; MAX_ROW];
    let grid = SplitGrid {
        classes: MAX_ROW - 5,
        ..grid
    };
    assert!(u8::modelpack_split_rows_neon(
        &at_max,
        None,
        &grid,
        &mut boxes[..4],
        &mut scores[..MAX_ROW - 5]
    ));
    assert!(boxes
        .iter()
        .skip(4)
        .all(|x| unsafe { x.assume_init() }.is_nan()));
}
