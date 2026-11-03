// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON ModelPack split-detection decode: per anchor row of `5 + nc` values, the dequantize, the
//! Schraudolph sigmoid ([`super::sigmoid::fast_sigmoid`]), the box transform and the
//! objectness-weighted class scores.
//!
//! [`fast_sigmoid4`] is `fast_sigmoid` lane for lane: `FCVTZS`/`SCVTF` for its `as` casts, the
//! polynomial as separate products and sums (no FMA), an integer add into the bits and an IEEE
//! `FDIV`, so it equals the scalar function on every input, NaN and infinities included.
//! Dequantize is `x · scale + bias` (two roundings, as the decoder's generic dequantize) and the
//! box and score arithmetic is the decoder's scalar code, so every output equals the scalar
//! decode bit for bit. Rows are converted a stack block at a time and then decoded from it.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR
use std::mem::MaybeUninit;

use half::f16;

use super::neon_convert::{map_steps, F32x16, LoadF32, StoreFloat};
use super::sigmoid::{FAST_EXP_A, FAST_EXP_C0, FAST_EXP_C1, FAST_EXP_C2, FAST_EXP_MASK};

/// Sigmoid values per stack block; a row longer than this is not decoded.
pub const MAX_ROW: usize = 4096;

/// `fast_sigmoid` of four lanes, bit for bit.
#[inline(always)]
pub(crate) fn fast_sigmoid4(f: float32x4_t) -> float32x4_t {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let one = vdupq_n_f32(1.0);
        let mul = vcvtq_s32_f32(vmulq_f32(vdupq_n_f32(FAST_EXP_A), vnegq_f32(f)));
        let floor = vandq_s32(mul, vdupq_n_s32(FAST_EXP_MASK));
        let frac = vcvtq_f32_s32(vsubq_s32(mul, floor));
        let poly = vaddq_f32(
            vmulq_f32(
                vaddq_f32(
                    vmulq_f32(vdupq_n_f32(FAST_EXP_C0), frac),
                    vdupq_n_f32(FAST_EXP_C1),
                ),
                frac,
            ),
            vdupq_n_f32(FAST_EXP_C2),
        );
        let e = vreinterpretq_f32_u32(vaddq_u32(
            vreinterpretq_u32_f32(poly),
            vreinterpretq_u32_s32(floor),
        ));
        let smooth = vdivq_f32(one, vaddq_f32(one, e));
        let saturated = vbslq_f32(vcgtq_f32(f, vdupq_n_f32(0.0)), one, vdupq_n_f32(0.0));
        vbslq_f32(vcagtq_f32(f, vdupq_n_f32(80.0)), saturated, smooth)
    }
}

/// Where a run of anchor rows lies in one ModelPack split output `(height, width, na·(5 + nc))`.
#[derive(Debug, Clone, Copy)]
pub struct SplitGrid<'a> {
    /// Grid width.
    pub width: usize,
    /// Grid height.
    pub height: usize,
    /// The output's anchors `[w, h]`, cycling fastest after the grid cell.
    pub anchors: &'a [[f32; 2]],
    /// Classes `nc` per row.
    pub classes: usize,
    /// The output row of the first row passed.
    pub first_row: usize,
}

/// An input of the split decode.
pub trait SplitIn: Copy + Default {
    /// Decodes whole rows `src` (row `r` is output row `grid.first_row + r`): each value is
    /// dequantized (`x · scale + bias`, `x · scale` without a bias, unchanged without `dequant`)
    /// and passed through `fast_sigmoid`; then with grid cell `(x, y)` and anchor `[aw, ah]`,
    /// `boxes` gets `[(s0·2 + x − 0.5)/width, (s1·2 + y − 0.5)/height, s2·s2·4·aw, s3·s3·4·ah]`
    /// and `scores` the objectness `s4` (one class) or `s4` times each class value, every step
    /// rounded as the decoder's scalar code.
    ///
    /// Writes every element of `boxes` (4 per row) and `scores` (`nc` per row) and returns
    /// `true`, or returns `false` and writes nothing when the lengths, anchors or grid do not
    /// fit or a row is longer than [`MAX_ROW`].
    #[doc(hidden)]
    fn modelpack_split_rows_neon(
        src: &[Self],
        dequant: Option<(f32, Option<f32>)>,
        grid: &SplitGrid<'_>,
        boxes: &mut [MaybeUninit<f32>],
        scores: &mut [MaybeUninit<f32>],
    ) -> bool;
}

/// Fills `dst` with `fast_sigmoid(dq(x))` of `src`.
#[inline(always)]
fn sigmoid_block<S: LoadF32>(src: &[S], dst: &mut [f32], dq: impl Fn(F32x16) -> F32x16) {
    // SAFETY: `map_steps` hands out 16 valid elements at each pointer.
    map_steps(src, dst, |s, d| unsafe {
        f32::store16(d, dq(S::load16(s)).map(fast_sigmoid4));
    });
}

/// The row count of `src` when every length and the grid fit.
fn checked_rows<S>(src: &[S], grid: &SplitGrid<'_>, boxes: usize, scores: usize) -> Option<usize> {
    let nc = grid.classes;
    let row = nc.checked_add(5)?;
    if nc == 0 || row > MAX_ROW || grid.anchors.is_empty() || grid.width == 0 {
        return None;
    }
    let rows = src.len() / row;
    let all = grid
        .width
        .checked_mul(grid.height)?
        .checked_mul(grid.anchors.len())?;
    let fits = rows * row == src.len()
        && Some(boxes) == rows.checked_mul(4)
        && Some(scores) == rows.checked_mul(nc)
        && grid.first_row.checked_add(rows)? <= all;
    fits.then_some(rows)
}

#[inline(always)]
fn split_rows<S: LoadF32>(
    src: &[S],
    dequant: Option<(f32, Option<f32>)>,
    grid: &SplitGrid<'_>,
    boxes: &mut [MaybeUninit<f32>],
    scores: &mut [MaybeUninit<f32>],
) -> bool {
    if checked_rows(src, grid, boxes.len(), scores.len()).is_none() {
        return false;
    }
    let nc = grid.classes;
    let row = nc + 5;
    let na = grid.anchors.len();
    let div_width = 1.0 / grid.width as f32;
    let div_height = 1.0 / grid.height as f32;
    let (mut anchor, cell) = (grid.first_row % na, grid.first_row / na);
    let (mut x, mut y) = (cell % grid.width, cell / grid.width);
    let mut sig = [0f32; MAX_ROW];
    let per_block = MAX_ROW / row;
    let blocks = src
        .chunks(per_block * row)
        .zip(boxes.chunks_mut(per_block * 4))
        .zip(scores.chunks_mut(per_block * nc));
    for ((src, boxes), scores) in blocks {
        let Some(sig) = sig.get_mut(..src.len()) else {
            return false;
        };
        match dequant {
            None => sigmoid_block(src, sig, |v| v),
            Some((scale, None)) => {
                // SAFETY: NEON is part of the aarch64 baseline.
                let s = unsafe { vdupq_n_f32(scale) };
                sigmoid_block(src, sig, |v| v.map(|x| unsafe { vmulq_f32(x, s) }));
            }
            Some((scale, Some(bias))) => {
                // SAFETY: NEON is part of the aarch64 baseline.
                let (s, b) = unsafe { (vdupq_n_f32(scale), vdupq_n_f32(bias)) };
                sigmoid_block(src, sig, |v| {
                    v.map(|x| unsafe { vaddq_f32(vmulq_f32(x, s), b) })
                });
            }
        }
        let rows = sig
            .chunks_exact(row)
            .zip(boxes.chunks_exact_mut(4))
            .zip(scores.chunks_exact_mut(nc));
        for ((p, b), s) in rows {
            let (Some((&[px, py, pw, ph, obj], classes)), Some(&[aw, ah])) =
                (p.split_first_chunk::<5>(), grid.anchors.get(anchor))
            else {
                return false;
            };
            let gx = x as f32 - 0.5;
            let gy = y as f32 - 0.5;
            let decoded = [
                (px * 2.0 + gx) * div_width,
                (py * 2.0 + gy) * div_height,
                pw * pw * 4.0 * aw,
                ph * ph * 4.0 * ah,
            ];
            for (o, v) in b.iter_mut().zip(decoded) {
                o.write(v);
            }
            if nc == 1 {
                for o in s.iter_mut() {
                    o.write(obj);
                }
            } else {
                for (o, &c) in s.iter_mut().zip(classes) {
                    o.write(c * obj);
                }
            }
            anchor += 1;
            if anchor == na {
                anchor = 0;
                x += 1;
                if x == grid.width {
                    x = 0;
                    y += 1;
                }
            }
        }
    }
    true
}

macro_rules! split_in {
    ($($t:ty),*) => {$(
        impl SplitIn for $t {
            #[inline]
            fn modelpack_split_rows_neon(
                src: &[Self],
                dequant: Option<(f32, Option<f32>)>,
                grid: &SplitGrid<'_>,
                boxes: &mut [MaybeUninit<f32>],
                scores: &mut [MaybeUninit<f32>],
            ) -> bool {
                split_rows(src, dequant, grid, boxes, scores)
            }
        }
    )*};
}
split_in!(i8, u8, i16, u16, f32, f16);

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
