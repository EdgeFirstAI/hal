// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON candidate selection for detection decoders: the best class of each anchor kept when it
//! clears a score threshold, from class-planar columns or from contiguous rows, and the positions
//! of a slice at or above a threshold.
//!
//! Every kernel reproduces the scalar fold `if !(best > s) { (best, class) = (s, c) }` from class
//! 0, so ties go to the last class and a NaN score takes over the running best, and keeps an
//! anchor unless its best score is below the threshold (a NaN best is kept). Results equal the
//! scalar fold's on every input, NaN and signed zeros included. Columns are processed in tiles of
//! one cache line per class whose running best and class stay in registers across every class.
//! `f16` scores compare as f32 after `FCVTL` (exact) on the baseline kernels and natively on the
//! `*_neon_fp16` ones, which need FEAT_FP16. A kernel returns `false` and reports nothing when its
//! arguments are invalid.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR
use std::cmp::Ordering;

use half::f16;

use super::fp16_asm::{fcmeq_f16x8, fcmge_f16x8, fcmgt_f16x8, fmax_f16x8, fmaxv_f16x8};
use super::neon_convert::widen8;

/// Trailing positions the row argmax checks in scalar before its vector scan; at most `N`.
const LAST_IN_SCALAR: usize = 4;

/// Bytes the column tiles prefetch ahead of the tile they read.
const PREFETCH_AHEAD: usize = 256;

/// Score lanes for the candidate kernels.
trait Lanes {
    /// Score type.
    type T: Copy + PartialOrd;
    /// `N` scores, as compared.
    type V: Copy;
    /// `N` class indices.
    type C: Copy;
    /// `N` lane flags.
    type M: Copy;
    /// Lanes per vector, at most 16.
    const N: usize;
    /// Classes a class vector can index.
    const MAX_CLASSES: usize;

    /// # Safety
    /// `p` must be valid for reading `N` scores.
    unsafe fn load(p: *const Self::T) -> Self::V;
    fn splat(t: Self::T) -> Self::V;
    fn class(c: usize) -> Self::C;
    /// Lanes where `a > b`.
    fn gt(a: Self::V, b: Self::V) -> Self::M;
    /// Lanes where `a >= b`.
    fn ge(a: Self::V, b: Self::V) -> Self::M;
    /// Lanes where `a == b`.
    fn eq(a: Self::V, b: Self::V) -> Self::M;
    /// `m ? a : b` per lane.
    fn select(m: Self::M, a: Self::V, b: Self::V) -> Self::V;
    /// `m ? a : b` per lane.
    fn select_class(m: Self::M, a: Self::C, b: Self::C) -> Self::C;
    /// Lane maximum; any NaN lane gives NaN.
    fn max(a: Self::V, b: Self::V) -> Self::V;
    /// The largest lane, splatted and as a score, or `None` when a lane is NaN.
    fn reduce_max(v: Self::V) -> Option<(Self::V, Self::T)>;
    /// Lane flags as `64 / N` bits per lane, lane 0 lowest.
    fn bits(m: Self::M) -> u64;
    /// Class indices in `out[..N]`.
    fn store_class(c: Self::C, out: &mut [u32; 16]);
}

/// 16 byte-lane flags as a nibble each (`SHRN`).
#[inline(always)]
fn bits8(m: uint8x16_t) -> u64 {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        vget_lane_u64::<0>(vreinterpret_u64_u8(vshrn_n_u16::<4>(vreinterpretq_u16_u8(
            m,
        ))))
    }
}

/// 8 halfword-lane flags as a byte each (`XTN`).
#[inline(always)]
fn bits16(m: uint16x8_t) -> u64 {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe { vget_lane_u64::<0>(vreinterpret_u64_u8(vmovn_u16(m))) }
}

/// 4 word-lane flags as a halfword each (`XTN`).
#[inline(always)]
fn bits32(m: uint32x4_t) -> u64 {
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe { vget_lane_u64::<0>(vreinterpret_u64_u16(vmovn_u32(m))) }
}

/// The set lanes of a [`Lanes::bits`] mask of `width` bits per lane, lowest first.
struct SetLanes {
    bits: u64,
    width: u32,
}

impl Iterator for SetLanes {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.bits == 0 {
            return None;
        }
        let lane = self.bits.trailing_zeros() / self.width;
        self.bits &= !((u64::MAX >> (64 - self.width)) << (lane * self.width));
        Some(lane as usize)
    }
}

macro_rules! int_lanes {
    ($name:ident, $t:ty, $v:ty, $c:ty, $n:expr, $max_classes:expr,
     $ld:ident, $dup:ident, $cdup:ident, $ct:ty, $gt:ident, $ge:ident, $eq:ident, $bsl:ident,
     $cbsl:ident, $vmax:ident, $vmaxv:ident, $bits:ident, $cst:ident) => {
        struct $name;

        impl Lanes for $name {
            type T = $t;
            type V = $v;
            type C = $c;
            type M = $c;
            const N: usize = $n;
            const MAX_CLASSES: usize = $max_classes;

            #[inline(always)]
            unsafe fn load(p: *const $t) -> $v {
                // SAFETY: the caller guarantees `N` readable scores.
                unsafe { $ld(p) }
            }
            #[inline(always)]
            fn splat(t: $t) -> $v {
                // SAFETY: NEON is part of the aarch64 baseline.
                unsafe { $dup(t) }
            }
            #[inline(always)]
            fn class(c: usize) -> $c {
                // SAFETY: as above; callers keep `c` below `MAX_CLASSES`.
                unsafe { $cdup(c as $ct) }
            }
            #[inline(always)]
            fn gt(a: $v, b: $v) -> $c {
                // SAFETY: as above.
                unsafe { $gt(a, b) }
            }
            #[inline(always)]
            fn ge(a: $v, b: $v) -> $c {
                // SAFETY: as above.
                unsafe { $ge(a, b) }
            }
            #[inline(always)]
            fn eq(a: $v, b: $v) -> $c {
                // SAFETY: as above.
                unsafe { $eq(a, b) }
            }
            #[inline(always)]
            fn select(m: $c, a: $v, b: $v) -> $v {
                // SAFETY: as above.
                unsafe { $bsl(m, a, b) }
            }
            #[inline(always)]
            fn select_class(m: $c, a: $c, b: $c) -> $c {
                // SAFETY: as above.
                unsafe { $cbsl(m, a, b) }
            }
            #[inline(always)]
            fn max(a: $v, b: $v) -> $v {
                // SAFETY: as above.
                unsafe { $vmax(a, b) }
            }
            #[inline(always)]
            fn reduce_max(v: $v) -> Option<($v, $t)> {
                // SAFETY: as above.
                let m = unsafe { $vmaxv(v) };
                Some((Self::splat(m), m))
            }
            #[inline(always)]
            fn bits(m: $c) -> u64 {
                $bits(m)
            }
            #[inline(always)]
            fn store_class(c: $c, out: &mut [u32; 16]) {
                let mut lanes = [0 as $ct; $n];
                // SAFETY: `lanes` holds `N` elements.
                unsafe { $cst(lanes.as_mut_ptr(), c) };
                for (o, l) in out.iter_mut().zip(lanes) {
                    *o = u32::from(l);
                }
            }
        }
    };
}

int_lanes!(
    I8, i8, int8x16_t, uint8x16_t, 16, 256, vld1q_s8, vdupq_n_s8, vdupq_n_u8, u8, vcgtq_s8,
    vcgeq_s8, vceqq_s8, vbslq_s8, vbslq_u8, vmaxq_s8, vmaxvq_s8, bits8, vst1q_u8
);
int_lanes!(
    U8, u8, uint8x16_t, uint8x16_t, 16, 256, vld1q_u8, vdupq_n_u8, vdupq_n_u8, u8, vcgtq_u8,
    vcgeq_u8, vceqq_u8, vbslq_u8, vbslq_u8, vmaxq_u8, vmaxvq_u8, bits8, vst1q_u8
);
int_lanes!(
    I16,
    i16,
    int16x8_t,
    uint16x8_t,
    8,
    65536,
    vld1q_s16,
    vdupq_n_s16,
    vdupq_n_u16,
    u16,
    vcgtq_s16,
    vcgeq_s16,
    vceqq_s16,
    vbslq_s16,
    vbslq_u16,
    vmaxq_s16,
    vmaxvq_s16,
    bits16,
    vst1q_u16
);
int_lanes!(
    U16,
    u16,
    uint16x8_t,
    uint16x8_t,
    8,
    65536,
    vld1q_u16,
    vdupq_n_u16,
    vdupq_n_u16,
    u16,
    vcgtq_u16,
    vcgeq_u16,
    vceqq_u16,
    vbslq_u16,
    vbslq_u16,
    vmaxq_u16,
    vmaxvq_u16,
    bits16,
    vst1q_u16
);

struct F32;

impl Lanes for F32 {
    type T = f32;
    type V = float32x4_t;
    type C = uint32x4_t;
    type M = uint32x4_t;
    const N: usize = 4;
    const MAX_CLASSES: usize = 1 << 32;

    #[inline(always)]
    unsafe fn load(p: *const f32) -> float32x4_t {
        // SAFETY: the caller guarantees `N` readable scores.
        unsafe { vld1q_f32(p) }
    }
    #[inline(always)]
    fn splat(t: f32) -> float32x4_t {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { vdupq_n_f32(t) }
    }
    #[inline(always)]
    fn class(c: usize) -> uint32x4_t {
        // SAFETY: as above; callers keep `c` below `MAX_CLASSES`.
        unsafe { vdupq_n_u32(c as u32) }
    }
    #[inline(always)]
    fn gt(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
        // SAFETY: as above.
        unsafe { vcgtq_f32(a, b) }
    }
    #[inline(always)]
    fn ge(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
        // SAFETY: as above.
        unsafe { vcgeq_f32(a, b) }
    }
    #[inline(always)]
    fn eq(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
        // SAFETY: as above.
        unsafe { vceqq_f32(a, b) }
    }
    #[inline(always)]
    fn select(m: uint32x4_t, a: float32x4_t, b: float32x4_t) -> float32x4_t {
        // SAFETY: as above.
        unsafe { vbslq_f32(m, a, b) }
    }
    #[inline(always)]
    fn select_class(m: uint32x4_t, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
        // SAFETY: as above.
        unsafe { vbslq_u32(m, a, b) }
    }
    #[inline(always)]
    fn max(a: float32x4_t, b: float32x4_t) -> float32x4_t {
        // SAFETY: as above.
        unsafe { vmaxq_f32(a, b) }
    }
    #[inline(always)]
    fn reduce_max(v: float32x4_t) -> Option<(float32x4_t, f32)> {
        // SAFETY: as above.
        let m = unsafe { vmaxvq_f32(v) };
        (!m.is_nan()).then(|| (Self::splat(m), m))
    }
    #[inline(always)]
    fn bits(m: uint32x4_t) -> u64 {
        bits32(m)
    }
    #[inline(always)]
    fn store_class(c: uint32x4_t, out: &mut [u32; 16]) {
        // SAFETY: `out` holds at least 4 elements.
        unsafe { vst1q_u32(out.as_mut_ptr(), c) };
    }
}

/// `f16` scores widened to f32 (`FCVTL`, exact) and compared there.
struct F16Widened;

impl Lanes for F16Widened {
    type T = f16;
    type V = [float32x4_t; 2];
    type C = [uint32x4_t; 2];
    type M = [uint32x4_t; 2];
    const N: usize = 8;
    const MAX_CLASSES: usize = 1 << 32;

    #[inline(always)]
    unsafe fn load(p: *const f16) -> Self::V {
        // SAFETY: the caller guarantees 8 readable scores; `f16` is a transparent `u16`.
        widen8(unsafe { vld1q_u16(p.cast()) })
    }
    #[inline(always)]
    fn splat(t: f16) -> Self::V {
        // SAFETY: NEON is part of the aarch64 baseline.
        let v = unsafe { vdupq_n_f32(t.to_f32()) };
        [v, v]
    }
    #[inline(always)]
    fn class(c: usize) -> Self::C {
        // SAFETY: as above; callers keep `c` below `MAX_CLASSES`.
        let v = unsafe { vdupq_n_u32(c as u32) };
        [v, v]
    }
    #[inline(always)]
    fn gt(a: Self::V, b: Self::V) -> Self::M {
        [F32::gt(a[0], b[0]), F32::gt(a[1], b[1])]
    }
    #[inline(always)]
    fn ge(a: Self::V, b: Self::V) -> Self::M {
        [F32::ge(a[0], b[0]), F32::ge(a[1], b[1])]
    }
    #[inline(always)]
    fn eq(a: Self::V, b: Self::V) -> Self::M {
        [F32::eq(a[0], b[0]), F32::eq(a[1], b[1])]
    }
    #[inline(always)]
    fn select(m: Self::M, a: Self::V, b: Self::V) -> Self::V {
        [F32::select(m[0], a[0], b[0]), F32::select(m[1], a[1], b[1])]
    }
    #[inline(always)]
    fn select_class(m: Self::M, a: Self::C, b: Self::C) -> Self::C {
        [
            F32::select_class(m[0], a[0], b[0]),
            F32::select_class(m[1], a[1], b[1]),
        ]
    }
    #[inline(always)]
    fn max(a: Self::V, b: Self::V) -> Self::V {
        [F32::max(a[0], b[0]), F32::max(a[1], b[1])]
    }
    #[inline(always)]
    fn reduce_max(v: Self::V) -> Option<(Self::V, f16)> {
        // The largest of widened f16 values narrows back exactly.
        F32::reduce_max(F32::max(v[0], v[1])).map(|(m, s)| ([m, m], f16::from_f32(s)))
    }
    #[inline(always)]
    fn bits(m: Self::M) -> u64 {
        // SAFETY: NEON is part of the aarch64 baseline.
        bits16(unsafe { vmovn_high_u32(vmovn_u32(m[0]), m[1]) })
    }
    #[inline(always)]
    fn store_class(c: Self::C, out: &mut [u32; 16]) {
        // SAFETY: `out` holds at least 8 elements.
        unsafe {
            vst1q_u32(out.as_mut_ptr(), c[0]);
            vst1q_u32(out.as_mut_ptr().add(4), c[1]);
        }
    }
}

/// `f16` scores compared with FP16 instructions.
struct F16Native;

impl Lanes for F16Native {
    type T = f16;
    type V = uint16x8_t;
    type C = uint16x8_t;
    type M = uint16x8_t;
    const N: usize = 8;
    const MAX_CLASSES: usize = 65536;

    #[inline(always)]
    unsafe fn load(p: *const f16) -> uint16x8_t {
        // SAFETY: the caller guarantees 8 readable scores; `f16` is a transparent `u16`.
        unsafe { vld1q_u16(p.cast()) }
    }
    #[inline(always)]
    fn splat(t: f16) -> uint16x8_t {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { vdupq_n_u16(t.to_bits()) }
    }
    #[inline(always)]
    fn class(c: usize) -> uint16x8_t {
        U16::class(c)
    }
    #[inline(always)]
    fn gt(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        // SAFETY: `F16Native` only runs inside the `*_neon_fp16` kernels, which need FEAT_FP16.
        unsafe { fcmgt_f16x8(a, b) }
    }
    #[inline(always)]
    fn ge(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        // SAFETY: as `gt`.
        unsafe { fcmge_f16x8(a, b) }
    }
    #[inline(always)]
    fn eq(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        // SAFETY: as `gt`.
        unsafe { fcmeq_f16x8(a, b) }
    }
    #[inline(always)]
    fn select(m: uint16x8_t, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        U16::select(m, a, b)
    }
    #[inline(always)]
    fn select_class(m: uint16x8_t, a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        U16::select_class(m, a, b)
    }
    #[inline(always)]
    fn max(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
        // SAFETY: as `gt`.
        unsafe { fmax_f16x8(a, b) }
    }
    #[inline(always)]
    fn reduce_max(v: uint16x8_t) -> Option<(uint16x8_t, f16)> {
        // SAFETY: as `gt`.
        let m = f16::from_bits(unsafe { fmaxv_f16x8(v) });
        (!m.is_nan()).then(|| (Self::splat(m), m))
    }
    #[inline(always)]
    fn bits(m: uint16x8_t) -> u64 {
        bits16(m)
    }
    #[inline(always)]
    fn store_class(c: uint16x8_t, out: &mut [u32; 16]) {
        U16::store_class(c, out);
    }
}

/// Hints the cache line at `p` into L1; `p` may lie past the data, where the hint is ignored.
#[inline(always)]
pub(crate) fn prefetch<T>(p: *const T) {
    // SAFETY: PRFM never faults and writes nothing.
    unsafe {
        core::arch::asm!(
            "prfm pldl1keep, [{p}]",
            p = in(reg) p,
            options(nostack, preserves_flags, readonly),
        );
    }
}

/// Anchors `base..base + R·N` of the column argmax, reporting kept `(anchor, class)` in order.
///
/// # Safety
/// Every column must hold at least `base + R·N` scores and `cols.len() <= L::MAX_CLASSES`.
#[inline(always)]
unsafe fn column_tile<L: Lanes, const R: usize>(
    cols: &[&[L::T]],
    base: usize,
    threshold: L::V,
    keep: &mut impl FnMut(usize, usize),
) {
    let Some((first, rest)) = cols.split_first() else {
        return;
    };
    // SAFETY: the caller guarantees `base + R·N` scores in every column.
    let mut best: [L::V; R] =
        std::array::from_fn(|r| unsafe { L::load(first.as_ptr().add(base + r * L::N)) });
    let mut class: [L::C; R] = [L::class(0); R];
    for (c, col) in rest.iter().enumerate() {
        let cv = L::class(c + 1);
        // SAFETY: as above.
        let p = unsafe { col.as_ptr().add(base) };
        prefetch(p.wrapping_byte_add(PREFETCH_AHEAD));
        for r in 0..R {
            // SAFETY: as above.
            let s = unsafe { L::load(p.add(r * L::N)) };
            let held = L::gt(best[r], s);
            best[r] = L::select(held, best[r], s);
            class[r] = L::select_class(held, class[r], cv);
        }
    }
    let width = (64 / L::N) as u32;
    for r in 0..R {
        let kept = !L::bits(L::gt(threshold, best[r]));
        if kept == 0 {
            continue;
        }
        let mut classes = [0u32; 16];
        L::store_class(class[r], &mut classes);
        for lane in (SetLanes { bits: kept, width }) {
            if let Some(&c) = classes.get(lane) {
                keep(base + r * L::N + lane, c as usize);
            }
        }
    }
}

/// The column argmax of [`column_argmax_i8_neon`] for any [`Lanes`].
#[inline(always)]
fn column_argmax<L: Lanes>(
    cols: &[&[L::T]],
    threshold: L::T,
    mut keep: impl FnMut(usize, usize),
) -> bool {
    let Some(n) = cols.first().map(|c| c.len()) else {
        return false;
    };
    if cols.len() > L::MAX_CLASSES || cols.iter().any(|c| c.len() != n) {
        return false;
    }
    const R: usize = 4;
    let thr = L::splat(threshold);
    let mut base = 0;
    while base + R * L::N <= n {
        // SAFETY: every column holds `n` scores and the class count was checked.
        unsafe { column_tile::<L, R>(cols, base, thr, &mut keep) };
        base += R * L::N;
    }
    while base + L::N <= n {
        // SAFETY: as above.
        unsafe { column_tile::<L, 1>(cols, base, thr, &mut keep) };
        base += L::N;
    }
    for i in base..n {
        let mut scores = cols.iter().filter_map(|col| col.get(i).copied());
        let Some(first) = scores.next() else {
            break;
        };
        let (mut best, mut class) = (first, 0);
        for (c, s) in scores.enumerate() {
            if best.partial_cmp(&s) != Some(Ordering::Greater) {
                (best, class) = (s, c + 1);
            }
        }
        if best.partial_cmp(&threshold) != Some(Ordering::Less) {
            keep(i, class);
        }
    }
    true
}

/// The row argmax of [`row_argmax_i8_neon`] for any [`Lanes`].
#[inline(always)]
fn row_argmax<L: Lanes>(row: &[L::T]) -> Option<(L::T, usize)> {
    let n = row.len();
    if n < L::N {
        return None;
    }
    let p = row.as_ptr();
    // SAFETY: `row` holds `n >= N` scores; every load below reads `N` of them.
    let best = unsafe {
        let mut m = L::load(p);
        let mut at = L::N;
        while at + L::N <= n {
            m = L::max(m, L::load(p.add(at)));
            at += L::N;
        }
        if at < n {
            m = L::max(m, L::load(p.add(n - L::N)));
        }
        m
    };
    let (best, best_score) = L::reduce_max(best)?;
    // Real rows mostly end in a tie at their maximum, which a scalar compare finds soonest.
    for i in (n - LAST_IN_SCALAR..n).rev() {
        if let Some(&s) = row.get(i).filter(|&&s| s == best_score) {
            return Some((s, i));
        }
    }
    let width = (64 / L::N) as u32;
    let mut at = n - L::N;
    loop {
        // SAFETY: `at + N <= n`.
        let hit = L::bits(L::eq(unsafe { L::load(p.add(at)) }, best));
        if hit != 0 {
            let i = at + ((63 - hit.leading_zeros()) / width) as usize;
            return row.get(i).map(|&s| (s, i));
        }
        if at == 0 {
            return None;
        }
        at = at.saturating_sub(L::N);
    }
}

/// The positions of [`ge_indices_i8_neon`] for any [`Lanes`].
#[inline(always)]
fn ge_indices<L: Lanes>(src: &[L::T], threshold: L::T, mut hit: impl FnMut(usize)) {
    let thr = L::splat(threshold);
    let width = (64 / L::N) as u32;
    let mut chunks = src.chunks_exact(L::N);
    for (k, chunk) in (&mut chunks).enumerate() {
        // SAFETY: the chunk holds `N` scores.
        let bits = L::bits(L::ge(unsafe { L::load(chunk.as_ptr()) }, thr));
        for lane in (SetLanes { bits, width }) {
            hit(k * L::N + lane);
        }
    }
    let done = src.len() - chunks.remainder().len();
    for (i, &s) in chunks.remainder().iter().enumerate() {
        if s >= threshold {
            hit(done + i);
        }
    }
}

macro_rules! candidate_kernels {
    ($t:ty, $lanes:ident, $column:ident, $row:ident, $ge:ident, $feature:literal) => {
        /// For each anchor `i` the class `c` with the largest `cols[c][i]` (ties to the last
        /// class, a NaN score taking over), reporting `keep(i, c)` in anchor order unless that
        /// score is below `threshold`; `false` when the columns differ in length, are empty or
        /// outnumber the class lanes.
        ///
        /// # Safety
        #[doc = $feature]
        #[inline]
        #[target_feature(enable = "neon")]
        pub unsafe fn $column(
            cols: &[&[$t]],
            threshold: $t,
            keep: impl FnMut(usize, usize),
        ) -> bool {
            column_argmax::<$lanes>(cols, threshold, keep)
        }

        /// The largest score of `row` and its last position, as the scalar fold finds them;
        /// `None` when `row` is shorter than a vector or holds a NaN, which the caller folds.
        ///
        /// # Safety
        #[doc = $feature]
        #[inline]
        #[target_feature(enable = "neon")]
        pub unsafe fn $row(row: &[$t]) -> Option<($t, usize)> {
            row_argmax::<$lanes>(row)
        }

        /// `hit(i)` for each position of `src` at or above `threshold`, in order.
        ///
        /// # Safety
        #[doc = $feature]
        #[inline]
        #[target_feature(enable = "neon")]
        pub unsafe fn $ge(src: &[$t], threshold: $t, hit: impl FnMut(usize)) {
            ge_indices::<$lanes>(src, threshold, hit);
        }
    };
}

candidate_kernels!(
    i8,
    I8,
    column_argmax_i8_neon,
    row_argmax_i8_neon,
    ge_indices_i8_neon,
    "The CPU must support NEON (every aarch64 CPU does)."
);
candidate_kernels!(
    u8,
    U8,
    column_argmax_u8_neon,
    row_argmax_u8_neon,
    ge_indices_u8_neon,
    "The CPU must support NEON (every aarch64 CPU does)."
);
candidate_kernels!(
    i16,
    I16,
    column_argmax_i16_neon,
    row_argmax_i16_neon,
    ge_indices_i16_neon,
    "The CPU must support NEON (every aarch64 CPU does)."
);
candidate_kernels!(
    u16,
    U16,
    column_argmax_u16_neon,
    row_argmax_u16_neon,
    ge_indices_u16_neon,
    "The CPU must support NEON (every aarch64 CPU does)."
);
candidate_kernels!(
    f32,
    F32,
    column_argmax_f32_neon,
    row_argmax_f32_neon,
    ge_indices_f32_neon,
    "The CPU must support NEON (every aarch64 CPU does)."
);
candidate_kernels!(
    f16,
    F16Widened,
    column_argmax_f16_neon,
    row_argmax_f16_neon,
    ge_indices_f16_neon,
    "The CPU must support NEON (every aarch64 CPU does)."
);
candidate_kernels!(
    f16,
    F16Native,
    column_argmax_f16_neon_fp16,
    row_argmax_f16_neon_fp16,
    ge_indices_f16_neon_fp16,
    "The CPU must support FEAT_FP16."
);

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
