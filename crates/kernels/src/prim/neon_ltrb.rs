// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON dist2bbox of LTRB distances fused with their dequantize, sixteen anchors per step.
//!
//! `vld4` loads deinterleave the four sides, so a step dequantizes and boxes sixteen anchors in
//! registers and stores them with `vst4`. Integer inputs dequantize as
//! `neon_baseline::dequant_*_to_f32_neon` does (`bias + x · scale`, fused), float inputs as
//! `prim::dequant` does, and each lane runs `neon_baseline::dist2bbox_4anchors_f32_neon`'s
//! arithmetic. A call's last `n % 4` anchors go through those two kernels' own tails, so the
//! f32 boxes equal the decoder's dequantize-then-dist2bbox composition bit for bit; the `f16`
//! kernel stores those f32 boxes rounded once. With `xyxy` both apply `[xc ∓ w/2, yc ∓ h/2]` to
//! the f32 boxes first, as the decode plans do. Baseline Advanced SIMD only.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR

use half::f16;

use super::box_primitives::dist2bbox_anchor_f32;
use super::neon_convert::{narrow8, widen8};
use crate::Quant;

/// Anchors per step.
const STEP: usize = 16;

/// Sixteen anchors' distances as f32 lanes, `[side][group]`: group `g` holds anchors `4g..4g + 4`.
pub type Sides = [[float32x4_t; 4]; 4];

/// An LTRB distance element the fused kernels read.
pub trait LtrbIn: Copy + Default {
    /// Integer inputs dequantize with the fused `bias + x · scale`.
    #[doc(hidden)]
    const INTEGER: bool;

    /// Sixteen anchors (64 elements) at `p` as exact f32 lanes, sides deinterleaved.
    ///
    /// # Safety
    /// `p` must be readable for 64 elements.
    #[doc(hidden)]
    unsafe fn load16(p: *const Self) -> Sides;

    /// The decoder's NEON dequantize of `src` into `dst` (same length): the fused FMA of
    /// `neon_baseline` for integers, `neon_convert`'s exact one for floats.
    #[doc(hidden)]
    fn dequant_neon(src: &[Self], q: Quant, dst: &mut [f32]);
}

/// The four sides of each 16-lane byte vector, widened to f32 groups.
macro_rules! bytes_to_sides {
    ($v:expr, $movl8:ident, $movl8h:ident, $movl16:ident, $movl16h:ident, $get:ident, $get16:ident, $cvt:ident) => {{
        let side = |v| {
            let (lo, hi) = ($movl8($get(v)), $movl8h(v));
            [
                $cvt($movl16($get16(lo))),
                $cvt($movl16h(lo)),
                $cvt($movl16($get16(hi))),
                $cvt($movl16h(hi)),
            ]
        };
        [side($v.0), side($v.1), side($v.2), side($v.3)]
    }};
}

macro_rules! halves_to_sides {
    ($a:expr, $b:expr, $movl:ident, $movlh:ident, $get:ident, $cvt:ident) => {{
        let side = |a, b| {
            [
                $cvt($movl($get(a))),
                $cvt($movlh(a)),
                $cvt($movl($get(b))),
                $cvt($movlh(b)),
            ]
        };
        [
            side($a.0, $b.0),
            side($a.1, $b.1),
            side($a.2, $b.2),
            side($a.3, $b.3),
        ]
    }};
}

impl LtrbIn for i8 {
    const INTEGER: bool = true;

    #[inline(always)]
    unsafe fn load16(p: *const Self) -> Sides {
        // SAFETY: the caller guarantees 64 readable elements; NEON is the aarch64 baseline.
        unsafe {
            let v = vld4q_s8(p);
            bytes_to_sides!(
                v,
                vmovl_s8,
                vmovl_high_s8,
                vmovl_s16,
                vmovl_high_s16,
                vget_low_s8,
                vget_low_s16,
                vcvtq_f32_s32
            )
        }
    }

    #[inline(always)]
    fn dequant_neon(src: &[Self], q: Quant, dst: &mut [f32]) {
        if src.len() == dst.len() {
            // SAFETY: NEON is part of the aarch64 baseline; the lengths are equal.
            unsafe { super::neon_baseline::dequant_i8_to_f32_neon(src, q, dst) };
        }
    }
}

impl LtrbIn for u8 {
    const INTEGER: bool = true;

    #[inline(always)]
    unsafe fn load16(p: *const Self) -> Sides {
        // SAFETY: the caller guarantees 64 readable elements; NEON is the aarch64 baseline.
        unsafe {
            let v = vld4q_u8(p);
            bytes_to_sides!(
                v,
                vmovl_u8,
                vmovl_high_u8,
                vmovl_u16,
                vmovl_high_u16,
                vget_low_u8,
                vget_low_u16,
                vcvtq_f32_u32
            )
        }
    }

    #[inline(always)]
    fn dequant_neon(src: &[Self], q: Quant, dst: &mut [f32]) {
        if src.len() == dst.len() {
            // SAFETY: NEON is part of the aarch64 baseline; the lengths are equal.
            unsafe { super::neon_baseline::dequant_u8_to_f32_neon(src, q, dst) };
        }
    }
}

impl LtrbIn for i16 {
    const INTEGER: bool = true;

    #[inline(always)]
    unsafe fn load16(p: *const Self) -> Sides {
        // SAFETY: the caller guarantees 64 readable elements; NEON is the aarch64 baseline.
        unsafe {
            let (a, b) = (vld4q_s16(p), vld4q_s16(p.add(32)));
            halves_to_sides!(a, b, vmovl_s16, vmovl_high_s16, vget_low_s16, vcvtq_f32_s32)
        }
    }

    #[inline(always)]
    fn dequant_neon(src: &[Self], q: Quant, dst: &mut [f32]) {
        if src.len() == dst.len() {
            // SAFETY: NEON is part of the aarch64 baseline; the lengths are equal.
            unsafe { super::neon_baseline::dequant_i16_to_f32_neon(src, q, dst) };
        }
    }
}

impl LtrbIn for u16 {
    const INTEGER: bool = true;

    #[inline(always)]
    unsafe fn load16(p: *const Self) -> Sides {
        // SAFETY: the caller guarantees 64 readable elements; NEON is the aarch64 baseline.
        unsafe {
            let (a, b) = (vld4q_u16(p), vld4q_u16(p.add(32)));
            halves_to_sides!(a, b, vmovl_u16, vmovl_high_u16, vget_low_u16, vcvtq_f32_u32)
        }
    }

    #[inline(always)]
    fn dequant_neon(src: &[Self], q: Quant, dst: &mut [f32]) {
        if src.len() == dst.len() {
            // SAFETY: NEON is part of the aarch64 baseline; the lengths are equal.
            unsafe { super::neon_baseline::dequant_u16_to_f32_neon(src, q, dst) };
        }
    }
}

impl LtrbIn for f16 {
    const INTEGER: bool = false;

    #[inline(always)]
    unsafe fn load16(p: *const Self) -> Sides {
        let p = p.cast::<u16>();
        // SAFETY: the caller guarantees 64 readable elements; `f16` is a transparent `u16`.
        let (a, b) = unsafe { (vld4q_u16(p), vld4q_u16(p.add(32))) };
        let side = |a, b| {
            let ([g0, g1], [g2, g3]) = (widen8(a), widen8(b));
            [g0, g1, g2, g3]
        };
        [
            side(a.0, b.0),
            side(a.1, b.1),
            side(a.2, b.2),
            side(a.3, b.3),
        ]
    }

    #[inline(always)]
    fn dequant_neon(src: &[Self], q: Quant, dst: &mut [f32]) {
        // SAFETY: NEON is part of the aarch64 baseline; it writes nothing on a length mismatch.
        unsafe { super::neon_convert::dequant_f16_to_f32_neon(src, q, dst) };
    }
}

impl LtrbIn for f32 {
    const INTEGER: bool = false;

    #[inline(always)]
    unsafe fn load16(p: *const Self) -> Sides {
        // SAFETY: the caller guarantees 64 readable elements.
        unsafe {
            let g = [
                vld4q_f32(p),
                vld4q_f32(p.add(16)),
                vld4q_f32(p.add(32)),
                vld4q_f32(p.add(48)),
            ];
            [
                g.map(|v| v.0),
                g.map(|v| v.1),
                g.map(|v| v.2),
                g.map(|v| v.3),
            ]
        }
    }

    #[inline(always)]
    fn dequant_neon(src: &[Self], q: Quant, dst: &mut [f32]) {
        // SAFETY: NEON is part of the aarch64 baseline; it writes nothing on a length mismatch.
        unsafe { super::neon_convert::dequant_f32_to_f32_neon(src, q, dst) };
    }
}

/// The per-lane dequantize of one input type and quantization.
pub(crate) trait Dequant: Copy {
    fn apply(self, x: float32x4_t) -> float32x4_t;

    /// Every lane of `s`.
    #[inline(always)]
    fn sides(self, s: Sides) -> Sides {
        s.map(|side| side.map(|x| self.apply(x)))
    }
}

/// Integer inputs: `bias + x · scale` fused, `bias = −zero_point · scale`.
#[derive(Clone, Copy)]
pub(crate) struct Fma {
    bias: float32x4_t,
    scale: f32,
}

impl Fma {
    #[inline(always)]
    pub(crate) fn new(q: Quant) -> Self {
        // SAFETY: NEON is part of the aarch64 baseline.
        let bias = unsafe { vdupq_n_f32(-(q.zero_point as f32) * q.scale) };
        Self {
            bias,
            scale: q.scale,
        }
    }
}

impl Dequant for Fma {
    #[inline(always)]
    fn apply(self, x: float32x4_t) -> float32x4_t {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { vfmaq_n_f32(self.bias, x, self.scale) }
    }
}

/// Float inputs under `Quant::IDENTITY`: the value itself.
#[derive(Clone, Copy)]
pub(crate) struct Identity;

impl Dequant for Identity {
    #[inline(always)]
    fn apply(self, x: float32x4_t) -> float32x4_t {
        x
    }
}

/// Float inputs otherwise: `(x − zero_point) · scale`.
#[derive(Clone, Copy)]
pub(crate) struct Affine {
    zp: float32x4_t,
    scale: float32x4_t,
}

impl Affine {
    #[inline(always)]
    pub(crate) fn new(q: Quant) -> Self {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            Self {
                zp: vdupq_n_f32(q.zero_point as f32),
                scale: vdupq_n_f32(q.scale),
            }
        }
    }
}

impl Dequant for Affine {
    #[inline(always)]
    fn apply(self, x: float32x4_t) -> float32x4_t {
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { vmulq_f32(vsubq_f32(x, self.zp), self.scale) }
    }
}

/// Evaluates `$body` with `$d` bound to the [`Dequant`] of input type `$t` under `$q`.
macro_rules! with_dequant {
    ($t:ty, $q:expr, |$d:ident| $body:expr) => {{
        use $crate::prim::neon_ltrb::{Affine, Fma, Identity, LtrbIn};
        let q = $q;
        if <$t as LtrbIn>::INTEGER {
            let $d = Fma::new(q);
            $body
        } else if q == $crate::Quant::IDENTITY {
            let $d = Identity;
            $body
        } else {
            let $d = Affine::new(q);
            $body
        }
    }};
}

/// `[xc, yc, w, h]` of each group of four anchors, as `dist2bbox_4anchors_f32_neon` computes.
///
/// # Safety
/// `gx` and `gy` must be readable for 16 floats.
#[inline(always)]
unsafe fn boxes16(s: Sides, gx: *const f32, gy: *const f32, stride: float32x4_t) -> Sides {
    let ([l0, l1, l2, l3], [t0, t1, t2, t3], [r0, r1, r2, r3], [b0, b1, b2, b3]) =
        (s[0], s[1], s[2], s[3]);
    // SAFETY: the caller guarantees 16 readable grid centres; NEON is the aarch64 baseline.
    unsafe {
        let half = vdupq_n_f32(0.5);
        let group = |g: usize, l, t, r, b| {
            let (gx, gy) = (vld1q_f32(gx.add(4 * g)), vld1q_f32(gy.add(4 * g)));
            [
                vmulq_f32(vfmaq_f32(gx, vsubq_f32(r, l), half), stride),
                vmulq_f32(vfmaq_f32(gy, vsubq_f32(b, t), half), stride),
                vmulq_f32(vaddq_f32(l, r), stride),
                vmulq_f32(vaddq_f32(t, b), stride),
            ]
        };
        [
            group(0, l0, t0, r0, b0),
            group(1, l1, t1, r1, b1),
            group(2, l2, t2, r2, b2),
            group(3, l3, t3, r3, b3),
        ]
    }
}

/// A box output element of the fused kernels.
pub trait LtrbOut: Copy + Default {
    /// Stores sixteen anchors' boxes, `[group][xc, yc, w, h]`, interleaved at `p`.
    ///
    /// # Safety
    /// `p` must be writable for 64 elements.
    #[doc(hidden)]
    unsafe fn store16(p: *mut Self, b: Sides);

    /// One anchor's f32 box.
    #[doc(hidden)]
    fn store4(out: &mut [Self], b: [f32; 4]);
}

impl LtrbOut for f32 {
    #[inline(always)]
    unsafe fn store16(p: *mut Self, b: Sides) {
        // SAFETY: the caller guarantees 64 writable floats.
        unsafe {
            for (g, [xc, yc, w, h]) in b.into_iter().enumerate() {
                vst4q_f32(p.add(16 * g), float32x4x4_t(xc, yc, w, h));
            }
        }
    }

    #[inline(always)]
    fn store4(out: &mut [Self], b: [f32; 4]) {
        for (o, v) in out.iter_mut().zip(b) {
            *o = v;
        }
    }
}

impl LtrbOut for f16 {
    #[inline(always)]
    unsafe fn store16(p: *mut Self, b: Sides) {
        let p = p.cast::<u16>();
        // SAFETY: the caller guarantees 64 writable f16; `f16` is a transparent `u16`.
        unsafe {
            let [g0, g1, g2, g3] = b;
            for (k, (a, c)) in [(g0, g1), (g2, g3)].into_iter().enumerate() {
                let v = uint16x8x4_t(
                    narrow8(a[0], c[0]),
                    narrow8(a[1], c[1]),
                    narrow8(a[2], c[2]),
                    narrow8(a[3], c[3]),
                );
                vst4q_u16(p.add(32 * k), v);
            }
        }
    }

    #[inline(always)]
    fn store4(out: &mut [Self], b: [f32; 4]) {
        for (o, v) in out.iter_mut().zip(b) {
            *o = f16::from_f32(v);
        }
    }
}

/// `[xc − w/2, yc − h/2, xc + w/2, yc + h/2]` of `[xc, yc, w, h]`, as the decode plans'
/// `BoxFormat::Xyxy` computes it from the f32 boxes.
#[inline(always)]
fn xyxy4(b: [float32x4_t; 4]) -> [float32x4_t; 4] {
    let [x, y, w, h] = b;
    // SAFETY: NEON is part of the aarch64 baseline.
    unsafe {
        let half = vdupq_n_f32(0.5);
        let (hw, hh) = (vmulq_f32(w, half), vmulq_f32(h, half));
        [
            vsubq_f32(x, hw),
            vsubq_f32(y, hh),
            vaddq_f32(x, hw),
            vaddq_f32(y, hh),
        ]
    }
}

/// [`xyxy4`] of one anchor.
#[inline(always)]
fn xyxy1([x, y, w, h]: [f32; 4]) -> [f32; 4] {
    let (hw, hh) = (w * 0.5, h * 0.5);
    [x - hw, y - hh, x + hw, y + hh]
}

/// The fused kernel for any output and format; writes nothing unless `dst` holds four values
/// per anchor of `src` and the grids cover every anchor.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn ltrb_boxes<T: LtrbIn, O: LtrbOut>(
    src: &[T],
    q: Quant,
    gx: &[f32],
    gy: &[f32],
    stride: f32,
    xyxy: bool,
    dst: &mut [O],
) {
    if xyxy {
        with_dequant!(T, q, |d| ltrb_boxes_with::<T, O, _, true>(
            src, q, d, gx, gy, stride, dst
        ));
    } else {
        with_dequant!(T, q, |d| ltrb_boxes_with::<T, O, _, false>(
            src, q, d, gx, gy, stride, dst
        ));
    }
}

#[inline(always)]
fn ltrb_boxes_with<T: LtrbIn, O: LtrbOut, D: Dequant, const XYXY: bool>(
    src: &[T],
    q: Quant,
    deq: D,
    gx: &[f32],
    gy: &[f32],
    stride: f32,
    dst: &mut [O],
) {
    let n = dst.len() / 4;
    if src.len() != dst.len() || !dst.len().is_multiple_of(4) || gx.len() < n || gy.len() < n {
        return;
    }
    // SAFETY: NEON is part of the aarch64 baseline.
    let s = unsafe { vdupq_n_f32(stride) };
    // SAFETY (both calls): `i` holds 64 inputs, `x` and `y` 16 centres each, `o` 64 outputs.
    let step = |i: *const T, x: *const f32, y: *const f32, o: *mut O| unsafe {
        let mut b = boxes16(deq.sides(T::load16(i)), x, y, s);
        if XYXY {
            b = b.map(xyxy4);
        }
        O::store16(o, b);
    };
    let mut src_steps = src.chunks_exact(4 * STEP);
    let mut out_steps = dst.chunks_exact_mut(4 * STEP);
    let grids = gx.chunks_exact(STEP).zip(gy.chunks_exact(STEP));
    for ((i, o), (x, y)) in (&mut src_steps).zip(&mut out_steps).zip(grids) {
        step(i.as_ptr(), x.as_ptr(), y.as_ptr(), o.as_mut_ptr());
    }
    let (rest, out) = (src_steps.remainder(), out_steps.into_remainder());
    if out.is_empty() {
        return;
    }
    let first = n - out.len() / 4;
    let quads = out.len() / 16;
    // Whole groups of four through the vector step on padded copies.
    if quads > 0 {
        let (Some(x), Some(y), Some(r)) = (
            gx.get(first..first + 4 * quads),
            gy.get(first..first + 4 * quads),
            rest.get(..16 * quads),
        ) else {
            return;
        };
        let mut i = [T::default(); 4 * STEP];
        let (mut xs, mut ys) = ([0.5f32; STEP], [0.5f32; STEP]);
        let mut o = [O::default(); 4 * STEP];
        i.iter_mut().zip(r).for_each(|(d, &v)| *d = v);
        xs.iter_mut().zip(x).for_each(|(d, &v)| *d = v);
        ys.iter_mut().zip(y).for_each(|(d, &v)| *d = v);
        step(i.as_ptr(), xs.as_ptr(), ys.as_ptr(), o.as_mut_ptr());
        out.iter_mut().zip(&o).for_each(|(d, &v)| *d = v);
    }
    // The last `n % 4` anchors as the decoder's NEON dequantize and scalar dist2bbox do.
    let k = out.len() / 4 - 4 * quads;
    let at = first + 4 * quads;
    let (Some(r), Some(o), Some(x), Some(y)) = (
        rest.get(16 * quads..),
        out.get_mut(16 * quads..),
        gx.get(at..at + k),
        gy.get(at..at + k),
    ) else {
        return;
    };
    let mut tail = [0.0f32; 12];
    let Some(tail) = tail.get_mut(..r.len()) else {
        return;
    };
    T::dequant_neon(r, q, tail);
    let anchors = tail.chunks_exact(4).zip(o.chunks_exact_mut(4));
    for ((d, o), (&x, &y)) in anchors.zip(x.iter().zip(y)) {
        if let &[l, t, r, b] = d {
            let b = dist2bbox_anchor_f32([l, t, r, b], x, y, stride);
            O::store4(o, if XYXY { xyxy1(b) } else { b });
        }
    }
}

/// `[xc, yc, w, h]` in f32 of each anchor's `[l, t, r, b]` in `src` (four per anchor), on grid
/// centres `gx`, `gy` and `stride`: bit for bit the decoder's NEON dequantize followed by
/// `dist2bbox_4anchors_f32_neon` and its scalar tail, then `[xc ∓ w/2, yc ∓ h/2]` when `xyxy`.
/// Writes nothing unless `dst` has the length of `src`, a multiple of 4, and the grids cover
/// every anchor.
///
/// # Safety
/// The CPU must support NEON (every aarch64 CPU does).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn ltrb_boxes_f32_neon<T: LtrbIn>(
    src: &[T],
    q: Quant,
    gx: &[f32],
    gy: &[f32],
    stride: f32,
    xyxy: bool,
    dst: &mut [f32],
) {
    ltrb_boxes(src, q, gx, gy, stride, xyxy, dst);
}

/// [`ltrb_boxes_f32_neon`]'s boxes rounded once to f16 (nearest even).
///
/// # Safety
/// The CPU must support NEON (every aarch64 CPU does).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn ltrb_boxes_f16_neon<T: LtrbIn>(
    src: &[T],
    q: Quant,
    gx: &[f32],
    gy: &[f32],
    stride: f32,
    xyxy: bool,
    dst: &mut [f16],
) {
    ltrb_boxes(src, q, gx, gy, stride, xyxy, dst);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prim::grids::make_anchor_grid;
    use crate::prim::neon_baseline::dist2bbox_4anchors_f32_neon;

    /// The decoder's composition: one dequantize of the call, four anchors per step, scalar tail.
    fn composition<T: LtrbIn>(src: &[T], q: Quant, gx: &[f32], gy: &[f32], s: f32) -> Vec<f32> {
        let n = src.len() / 4;
        let mut deq = vec![0f32; src.len()];
        T::dequant_neon(src, q, &mut deq);
        let mut out = vec![0f32; src.len()];
        for a in (0..n / 4 * 4).step_by(4) {
            // SAFETY: NEON is part of the aarch64 baseline; 16 values and 4 centres per step.
            unsafe {
                dist2bbox_4anchors_f32_neon(
                    &deq[a * 4..a * 4 + 16],
                    &gx[a..a + 4],
                    &gy[a..a + 4],
                    s,
                    &mut out[a * 4..a * 4 + 16],
                );
            }
        }
        for a in n / 4 * 4..n {
            let d = [deq[a * 4], deq[a * 4 + 1], deq[a * 4 + 2], deq[a * 4 + 3]];
            out[a * 4..a * 4 + 4].copy_from_slice(&dist2bbox_anchor_f32(d, gx[a], gy[a], s));
        }
        out
    }

    fn check<T: LtrbIn + std::fmt::Debug>(values: &[T], quants: &[Quant]) {
        for n in [
            0usize, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 19, 31, 32, 33, 47, 60, 61, 100,
        ] {
            let src: Vec<T> = values.iter().copied().cycle().take(n * 4).collect();
            let (gx, gy) = make_anchor_grid(1, n.max(1));
            for &q in quants {
                for xyxy in [false, true] {
                    let mut want = composition(&src, q, &gx, &gy, 16.0);
                    if xyxy {
                        for b in want.chunks_exact_mut(4) {
                            let v = xyxy1([b[0], b[1], b[2], b[3]]);
                            b.copy_from_slice(&v);
                        }
                    }
                    let mut got = vec![0f32; n * 4];
                    let mut got16 = vec![f16::ZERO; n * 4];
                    // SAFETY: NEON is part of the aarch64 baseline.
                    unsafe {
                        ltrb_boxes_f32_neon(&src, q, &gx, &gy, 16.0, xyxy, &mut got);
                        ltrb_boxes_f16_neon(&src, q, &gx, &gy, 16.0, xyxy, &mut got16);
                    }
                    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                        assert!(
                            g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                            "n {n} {q:?} value {i}: {g} vs {w}"
                        );
                    }
                    for (i, (g, w)) in got16.iter().zip(&want).enumerate() {
                        let w = f16::from_f32(*w);
                        assert!(
                            g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
                            "f16 n {n} {q:?} value {i}: {g} vs {w}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn fused_kernels_match_the_decoder_composition() {
        let ints: Vec<i32> = (0..997).map(|i| (i * 7919 + 13) % 65536).collect();
        let qs = [
            Quant::new(0.1, -128),
            Quant::new(0.037, 5),
            Quant::new(1e-3, 0),
        ];
        check(&ints.iter().map(|&v| v as i8).collect::<Vec<_>>(), &qs);
        check(&ints.iter().map(|&v| v as u8).collect::<Vec<_>>(), &qs);
        check(&ints.iter().map(|&v| v as i16).collect::<Vec<_>>(), &qs);
        check(&ints.iter().map(|&v| v as u16).collect::<Vec<_>>(), &qs);
        let mut floats: Vec<f32> = ints.iter().map(|&v| v as f32 * 3e-4 - 2.0).collect();
        floats.extend([
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            -0.0,
            f32::from_bits(1),
            1e30,
            -1e30,
        ]);
        let fq = [Quant::IDENTITY, Quant::new(0.5, 3)];
        check(&floats, &fq);
        check(
            &floats.iter().map(|&v| f16::from_f32(v)).collect::<Vec<_>>(),
            &fq,
        );
    }

    #[test]
    fn mismatched_lengths_write_nothing() {
        let (gx, gy) = make_anchor_grid(1, 4);
        let src = [1i8; 16];
        let mut dst = [7.0f32; 12];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe { ltrb_boxes_f32_neon(&src, Quant::new(0.1, 0), &gx, &gy, 8.0, false, &mut dst) };
        assert_eq!(dst, [7.0; 12]);
        let mut dst = [7.0f32; 20];
        // SAFETY: as above.
        unsafe {
            ltrb_boxes_f32_neon(
                &[1i8; 20],
                Quant::new(0.1, 0),
                &gx,
                &gy,
                8.0,
                false,
                &mut dst,
            )
        };
        assert_eq!(dst, [7.0; 20]);
    }
}
