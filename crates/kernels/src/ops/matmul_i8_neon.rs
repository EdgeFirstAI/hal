// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON int8 matrix multiplication.
//!
//! Each task computes up to four output rows against every column in 4×4 tiles of dot
//! products along `K`: `sdot` on dotprod CPUs, `smull` + `sadalp` (two products summed in
//! i16 lanes would overflow at −128·−128) otherwise. Operands not contiguous along `K` are
//! packed into stack panels first. Zero points enter through row and column sums:
//! `Σ (l − zl)(r − zr) = Σ l·r − zr·Σ l − zl·Σ r + K·zl·zr`, all exact in i32.

#![cfg(target_arch = "aarch64")]

use core::mem::MaybeUninit;
use core::ops::Range;
use std::arch::aarch64::*; // NOSONAR

use super::binary_i8_neon::mbqm_exact;
use super::matmul_i8::{MatMulI8Plan, Requant, Shape, MATMUL_MAX_K, TILE};
use crate::quant::Rounding;

/// `sdot acc.4s, a.16b, b.16b`.
///
/// # Safety
/// The CPU must support FEAT_DotProd.
#[inline(always)]
pub(super) unsafe fn sdot(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    let r: int32x4_t;
    // SAFETY: register-only; the caller guarantees FEAT_DotProd.
    unsafe {
        core::arch::asm!(
            ".arch_extension dotprod",
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc => r,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack),
        );
    }
    r
}

/// `acc += Σ a·b` over 16 lanes into four i32 lanes, with `sdot` or `smull` + `sadalp`.
///
/// # Safety
/// The CPU must support FEAT_DotProd when `DOT`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn dot16<const DOT: bool>(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
    if DOT {
        // SAFETY: FEAT_DotProd (caller).
        unsafe { sdot(acc, a, b) }
    } else {
        let lo = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        let hi = vmull_high_s8(a, b);
        vpadalq_s16(vpadalq_s16(acc, lo), hi)
    }
}

/// `Σ x` of a contiguous run.
#[inline]
#[target_feature(enable = "neon")]
pub(super) fn sum_i8(x: &[i8]) -> i32 {
    let (chunks, tail) = x.as_chunks::<16>();
    let mut acc = vdupq_n_s32(0);
    for c in chunks {
        // SAFETY: `c` is 16 readable bytes.
        let v = unsafe { vld1q_s8(c.as_ptr()) };
        acc = vpadalq_s16(acc, vpaddlq_s8(v));
    }
    vaddvq_s32(acc) + tail.iter().map(|&v| i32::from(v)).sum::<i32>()
}

/// `TILE` runs of `k` elements, each contiguous: the operand's own memory when it is
/// contiguous along `K`, else a packed copy in `store`.
struct Panel<'a> {
    rows: [&'a [i8]; TILE],
    sums: [i32; TILE],
}

/// Rows `first..first + count` of a matrix whose element `(i, k)` sits at
/// `base + i·s_i + k·s_k` of `data`, as a panel; missing rows repeat the first one. `None` when
/// a row does not lie in `data`.
#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "neon")]
fn panel<'a>(
    data: &'a [i8],
    base: usize,
    (s_i, s_k): (usize, usize),
    first: usize,
    count: usize,
    k: usize,
    store: &'a mut [MaybeUninit<i8>; TILE * MATMUL_MAX_K],
    want_sums: bool,
) -> Option<Panel<'a>> {
    let mut rows: [&[i8]; TILE] = [&[]; TILE];
    if s_k == 1 || k == 1 {
        for (t, row) in rows.iter_mut().enumerate() {
            let at = base + (first + t.min(count - 1)) * s_i;
            *row = data.get(at..at + k).unwrap_or(&[]);
        }
    } else {
        let (chunks, _) = store.as_chunks_mut::<MATMUL_MAX_K>();
        for (t, dst) in chunks.iter_mut().enumerate().take(count) {
            let at = base + (first + t) * s_i;
            for (kk, d) in dst.iter_mut().take(k).enumerate() {
                d.write(data.get(at + kk * s_k).copied().unwrap_or(0));
            }
        }
        let filled: &'a [[MaybeUninit<i8>; MATMUL_MAX_K]] = chunks;
        for (t, row) in rows.iter_mut().enumerate() {
            *row = filled
                .get(t.min(count - 1))
                .and_then(|c| c.get(..k))
                // SAFETY: the first `k` (at most `MATMUL_MAX_K`) elements of rows `0..count` were
                // written above, and `MaybeUninit<i8>` has the layout of `i8`.
                .map(|c| unsafe { &*(core::ptr::from_ref(c) as *const [i8]) })
                .unwrap_or(&[]);
        }
    }
    if rows.iter().any(|r| r.len() < k) {
        return None;
    }
    let mut sums = [0i32; TILE];
    if want_sums {
        for (s, row) in sums.iter_mut().zip(rows) {
            *s = sum_i8(row);
        }
    }
    Some(Panel { rows, sums })
}

/// The 4×4 block of `Σ a·b` dot products of four rows of `a` and four of `b`, each at least
/// `k` long.
///
/// # Safety
/// Every row must hold at least `k` elements, and the CPU must support FEAT_DotProd when `DOT`.
#[inline]
#[target_feature(enable = "neon")]
pub(super) unsafe fn tile<const DOT: bool>(
    a: &[&[i8]; TILE],
    b: &[&[i8]; TILE],
    k: usize,
) -> [[i32; TILE]; TILE] {
    let mut acc = [[vdupq_n_s32(0); TILE]; TILE];
    let full = k / 16 * 16;
    let mut kk = 0;
    while kk < full {
        let av: [int8x16_t; TILE] = core::array::from_fn(|i| {
            // SAFETY: every row holds `k` elements and `kk + 16 <= k`.
            unsafe { vld1q_s8(a[i].as_ptr().add(kk)) }
        });
        let bv: [int8x16_t; TILE] = core::array::from_fn(|j| {
            // SAFETY: as above.
            unsafe { vld1q_s8(b[j].as_ptr().add(kk)) }
        });
        for i in 0..TILE {
            for j in 0..TILE {
                // SAFETY: FEAT_DotProd when `DOT` (caller).
                acc[i][j] = unsafe { dot16::<DOT>(acc[i][j], av[i], bv[j]) };
            }
        }
        kk += 16;
    }
    let mut out = [[0i32; TILE]; TILE];
    for i in 0..TILE {
        for j in 0..TILE {
            let tail: i32 = a[i]
                .get(full..k)
                .unwrap_or(&[])
                .iter()
                .zip(b[j].get(full..k).unwrap_or(&[]))
                .map(|(&x, &y)| i32::from(x) * i32::from(y))
                .sum();
            out[i][j] = vaddvq_s32(acc[i][j]) + tail;
        }
    }
    out
}

/// Packing buffers for one task, reused across its row blocks: a block only reads what it
/// packed, so they start uninitialized.
pub(super) struct Panels {
    a: [MaybeUninit<i8>; TILE * MATMUL_MAX_K],
    b: [MaybeUninit<i8>; TILE * MATMUL_MAX_K],
}

impl Panels {
    pub(super) fn new() -> Self {
        Self {
            a: [const { MaybeUninit::uninit() }; TILE * MATMUL_MAX_K],
            b: [const { MaybeUninit::uninit() }; TILE * MATMUL_MAX_K],
        }
    }
}

/// Rows `rows` (at most [`TILE`]) of batch `b`.
///
/// # Safety
/// The CPU must support NEON, and FEAT_DotProd when `DOT`. `dst` must be the output view's
/// base, the shape validated against all three views, and no other task may write these rows.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(super) unsafe fn rows<const DOT: bool>(
    p: &MatMulI8Plan,
    s: &Shape,
    l: &[i8],
    r: &[i8],
    dst: *mut i8,
    b: usize,
    rows: Range<usize>,
    panels: &mut Panels,
) {
    let (lb, rb, ob) = s.batch_offsets(b);
    let ((lm, lk), (rk, rn), (om, on)) = s.strides();
    let (k, count) = (s.k, rows.len());
    if count == 0 {
        return;
    }
    let Panels {
        a: a_store,
        b: b_store,
    } = panels;
    let Some(a) = panel(l, lb, (lm, lk), rows.start, count, k, a_store, p.zr != 0) else {
        return;
    };
    let base = (k as i32).wrapping_mul(p.zl).wrapping_mul(p.zr);
    let mut n0 = 0;
    while n0 < s.n {
        let cols = TILE.min(s.n - n0);
        // The rhs columns as rows: element (n, k) at `rb + n·rn + k·rk`.
        let Some(bp) = panel(r, rb, (rn, rk), n0, cols, k, b_store, p.zl != 0) else {
            return;
        };
        // SAFETY: panel rows hold `k` elements (`panel` checks); the caller guarantees the CPU
        // features.
        let dots = unsafe { tile::<DOT>(&a.rows, &bp.rows, k) };
        for (i, m) in rows.clone().enumerate() {
            for (j, (&dot, &col_sum)) in dots[i].iter().zip(&bp.sums).enumerate().take(cols) {
                let acc = dot - p.zr * a.sums[i] - p.zl * col_sum + base;
                // SAFETY: (m, n0 + j) lies in the validated output view and only this task
                // writes row `m`.
                unsafe {
                    dst.add(ob + m * om + (n0 + j) * on)
                        .write(p.finish(acc, n0 + j));
                }
            }
        }
        n0 += TILE;
    }
}

/// A requantization on four lanes, rounded as `rounding` says: `MultiplyByQuantizedMultiplier`
/// for [`Rounding::Reference`], ruy's `sqdmulh` and rounding shift for [`Rounding::Ruy`]
/// (whose shifts are negative).
#[inline]
#[target_feature(enable = "neon")]
pub(super) fn requant4(x: int32x4_t, r: Requant, rounding: Rounding) -> int32x4_t {
    match rounding {
        Rounding::Reference => mbqm_exact(x, r.multiplier, r.shift),
        Rounding::Ruy => vrshlq_s32(vqdmulhq_n_s32(x, r.multiplier), vdupq_n_s32(r.shift)),
    }
}

/// Four i32 vectors narrowed to 16 saturated bytes.
#[inline]
#[target_feature(enable = "neon")]
pub(super) fn narrow16(q: [int32x4_t; 4]) -> int8x16_t {
    let lo = vcombine_s16(vqmovn_s32(q[0]), vqmovn_s32(q[1]));
    let hi = vcombine_s16(vqmovn_s32(q[2]), vqmovn_s32(q[3]));
    vqmovn_high_s16(vqmovn_s16(lo), hi)
}

/// Rows per step of [`narrow`].
const NARROW_ROWS: usize = 16;

/// Rows `rows` of batch `b` against a narrow right-hand side (`n < TILE` columns), the left
/// operand contiguous along `K`: sixteen rows per step, each column's sixteen dot products
/// accumulated independently, reduced to four vectors, requantized together and stored as one
/// 16-byte run when output rows are adjacent. The right-hand columns are packed once per call.
///
/// # Safety
/// As [`rows`]; additionally `n < TILE` and the left operand's `K` stride is 1.
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(super) unsafe fn narrow<const DOT: bool>(
    p: &MatMulI8Plan,
    s: &Shape,
    l: &[i8],
    r: &[i8],
    dst: *mut i8,
    b: usize,
    rows: Range<usize>,
    panels: &mut Panels,
) {
    let (lb, rb, ob) = s.batch_offsets(b);
    let ((lm, _), (rk, rn), (om, on)) = s.strides();
    let (k, n) = (s.k, s.n);
    if rows.is_empty() || n == 0 || n >= TILE {
        return;
    }
    let Some(cols) = panel(r, rb, (rn, rk), 0, n, k, &mut panels.b, p.zl != 0) else {
        return;
    };
    let base = (k as i32).wrapping_mul(p.zl).wrapping_mul(p.zr);
    // Per column: the requantization, and everything but the products and row-sum term.
    let per_col: [(Requant, int32x4_t); TILE] = core::array::from_fn(|j| {
        let (req, bias) = p.column(j.min(n - 1));
        let fixed = base
            .wrapping_sub(p.zl.wrapping_mul(cols.sums[j]))
            .wrapping_add(bias);
        (req, vdupq_n_s32(fixed))
    });
    let (zr, zo) = (vdupq_n_s32(p.zr), vdupq_n_s32(p.zo));
    let (lo, hi) = (vdupq_n_s32(p.act_min), vdupq_n_s32(p.act_max));
    let mut m0 = rows.start;
    while m0 < rows.end {
        let count = (rows.end - m0).min(NARROW_ROWS);
        if lb + (m0 + count - 1) * lm + k > l.len() {
            return;
        }
        // Short blocks repeat their last row; only `count` rows are stored.
        // SAFETY: every such row lies in `l` (checked above).
        let rp: [*const i8; NARROW_ROWS] =
            core::array::from_fn(|i| unsafe { l.as_ptr().add(lb + (m0 + i.min(count - 1)) * lm) });
        let row_sums = if p.zr == 0 {
            [vdupq_n_s32(0); 4]
        } else {
            // SAFETY: every row holds `k` elements.
            unsafe { dots16::<DOT>(&rp, None, k) }
        };
        for (j, &(req, fixed)) in per_col.iter().enumerate().take(n) {
            // SAFETY: rows and the packed column hold `k` elements each.
            let dots = unsafe { dots16::<DOT>(&rp, Some(cols.rows[j]), k) };
            let q: [int32x4_t; 4] = core::array::from_fn(|g| {
                let x = vaddq_s32(vmlsq_s32(dots[g], zr, row_sums[g]), fixed);
                vminq_s32(
                    vmaxq_s32(vaddq_s32(requant4(x, req, p.rounding), zo), lo),
                    hi,
                )
            });
            let bytes = narrow16(q);
            // SAFETY: rows `m0..m0 + count` of column `j` lie in the validated output view and
            // only this task writes them.
            unsafe {
                let out = dst.add(ob + m0 * om + j * on);
                if om == 1 && count == NARROW_ROWS {
                    vst1q_s8(out, bytes);
                } else {
                    let mut v = [0i8; NARROW_ROWS];
                    vst1q_s8(v.as_mut_ptr(), bytes);
                    for (i, &x) in v.iter().enumerate().take(count) {
                        out.add(i * om).write(x);
                    }
                }
            }
        }
        m0 += count;
    }
}

/// `Σ row_i · col` for sixteen rows (`Σ row_i` without a column), as four vectors of four
/// row results in row order.
///
/// # Safety
/// Every row and the column must hold `k` elements; the CPU must support FEAT_DotProd when
/// `DOT`.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn dots16<const DOT: bool>(
    rows: &[*const i8; NARROW_ROWS],
    col: Option<&[i8]>,
    k: usize,
) -> [int32x4_t; 4] {
    let full = k / 16 * 16;
    let ones = vdupq_n_s8(1);
    let mut acc = [vdupq_n_s32(0); NARROW_ROWS];
    let mut kk = 0;
    while kk < full {
        // SAFETY: `kk + 16 <= k` elements of the column.
        let bv = col.map_or(ones, |c| unsafe { vld1q_s8(c.as_ptr().add(kk)) });
        for (a, &row) in acc.iter_mut().zip(rows) {
            // SAFETY: as above, of every row; FEAT_DotProd when `DOT` (caller).
            *a = unsafe { dot16::<DOT>(*a, vld1q_s8(row.add(kk)), bv) };
        }
        kk += 16;
    }
    let mut out: [int32x4_t; 4] = core::array::from_fn(|g| {
        reduce4([acc[4 * g], acc[4 * g + 1], acc[4 * g + 2], acc[4 * g + 3]])
    });
    if full < k {
        let tail: [i32; NARROW_ROWS] = core::array::from_fn(|i| {
            (full..k)
                .map(|t| {
                    // SAFETY: `t < k` elements of the row and the column.
                    let x = i32::from(unsafe { *rows[i].add(t) });
                    x * col.map_or(1, |c| i32::from(c.get(t).copied().unwrap_or(0)))
                })
                .sum()
        });
        for (g, o) in out.iter_mut().enumerate() {
            // SAFETY: four i32 of `tail`.
            *o = vaddq_s32(*o, unsafe { vld1q_s32(tail.as_ptr().add(4 * g)) });
        }
    }
    out
}

/// The lane sums of four vectors: `[Σ v0, Σ v1, Σ v2, Σ v3]`.
#[inline]
#[target_feature(enable = "neon")]
fn reduce4(v: [int32x4_t; TILE]) -> int32x4_t {
    vpaddq_s32(vpaddq_s32(v[0], v[1]), vpaddq_s32(v[2], v[3]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn requant4_matches_the_scalar_roundings() {
        let xs = [
            i32::MIN + 1,
            -1_000_001,
            -65_536,
            -96,
            -64,
            -63,
            -1,
            0,
            1,
            63,
            64,
            96,
            65_536,
            1_000_001,
            i32::MAX,
        ];
        for real in [0.000_37, 0.0123, 0.25, 0.499, 0.75, 3.0] {
            let r = Requant::from_real(real).unwrap();
            for rounding in [Rounding::Reference, Rounding::Ruy] {
                if !r.valid_for(rounding) {
                    continue;
                }
                for x in xs {
                    // SAFETY: NEON is part of the aarch64 baseline.
                    let v = unsafe { vgetq_lane_s32::<0>(requant4(vdupq_n_s32(x), r, rounding)) };
                    assert_eq!(v, r.apply(x, rounding), "{real} {rounding:?} {x}");
                }
            }
        }
    }
}
