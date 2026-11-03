// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! DFL box decode of 8-bit bins through an exp lookup table.
//!
//! With per-tensor quantization a bin's logit is `(q - zp) * scale`, so its
//! softmax weight relative to the largest bin `q_max` of its side is
//! `exp(-(q_max - q) * scale)`. For 8-bit inputs `q_max - q` is an integer in
//! `0..=255`: the weight is one lookup in a 256-entry table that depends
//! only on the scale, the zero point cancels, and the side's maximum is an
//! integer maximum over the raw bins. The largest bin weighs exactly 1, so a
//! side's weight sum is never below 1.
//!
//! Each side accumulates its weight sum and bin-weighted sum over even and
//! odd bins separately, then adds the two. The NEON kernel holds the four
//! sides in the four lanes of a vector, and the portable kernel performs the
//! same additions in the same order, so both produce identical boxes.

use super::box_primitives::dist2bbox_anchor_f32;
use half::f16;

/// `exp(-k * scale)` for `k` in `0..=255`, the softmax weight of a bin `k`
/// raw steps below its side's largest bin.
#[derive(Debug, Clone)]
pub struct DflExpTable {
    scale_bits: u32,
    exp: [f32; 256],
}

impl DflExpTable {
    #[inline]
    pub fn new(scale: f32) -> Self {
        let mut table = Self {
            scale_bits: scale.to_bits(),
            exp: [0.0; 256],
        };
        table.fill(scale);
        table
    }

    /// Rebuild the table if it was built for a different scale.
    #[inline]
    pub fn update(&mut self, scale: f32) {
        if !self.is_for(scale) {
            self.scale_bits = scale.to_bits();
            self.fill(scale);
        }
    }

    /// Whether a table can stand for `scale`: the weights `exp(-k * scale)`
    /// are at most 1 only for a finite, positive scale. Any other scale
    /// overflows the table, and the bins are decoded by the max-subtracted
    /// softmax instead.
    #[inline]
    pub fn supports(scale: f32) -> bool {
        scale.is_finite() && scale > 0.0
    }

    #[inline]
    pub fn is_for(&self, scale: f32) -> bool {
        self.scale_bits == scale.to_bits()
    }

    fn fill(&mut self, scale: f32) {
        let scale = f64::from(scale);
        for (k, e) in self.exp.iter_mut().enumerate() {
            *e = (-(k as f64) * scale).exp() as f32;
        }
    }
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for i8 {}
    impl Sealed for u8 {}
    impl Sealed for f32 {}
    impl Sealed for half::f16 {}
}

/// An 8-bit DFL bin type. Sealed: the NEON kernel reads the bins as bytes.
///
/// ```compile_fail
/// #[derive(Clone, Copy)]
/// struct Wide(u64);
/// impl edgefirst_kernels::prim::dfl_lut::DflBin for Wide {
///     fn raw(self) -> i32 {
///         0
///     }
/// }
/// ```
pub trait DflBin: Copy + sealed::Sealed {
    fn raw(self) -> i32;
}

impl DflBin for i8 {
    #[inline(always)]
    fn raw(self) -> i32 {
        i32::from(self)
    }
}

impl DflBin for u8 {
    #[inline(always)]
    fn raw(self) -> i32 {
        i32::from(self)
    }
}

/// A box output element type. Sealed.
pub trait BoxOut: Copy + sealed::Sealed {
    fn store(dst: &mut [Self], xywh: [f32; 4]);

    /// [`BoxOut::store`] of a vector's four lanes; `dst` must hold four elements.
    #[cfg(target_arch = "aarch64")]
    #[doc(hidden)]
    fn store_v(dst: &mut [Self], xywh: std::arch::aarch64::float32x4_t);
}

impl BoxOut for f32 {
    #[inline(always)]
    fn store(dst: &mut [Self], xywh: [f32; 4]) {
        dst[..4].copy_from_slice(&xywh);
    }

    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn store_v(dst: &mut [Self], xywh: std::arch::aarch64::float32x4_t) {
        if let Some(d) = dst.get_mut(..4) {
            // SAFETY: NEON is part of the aarch64 baseline; `d` holds four floats.
            unsafe { std::arch::aarch64::vst1q_f32(d.as_mut_ptr(), xywh) };
        }
    }
}

impl BoxOut for f16 {
    #[inline(always)]
    fn store(dst: &mut [Self], xywh: [f32; 4]) {
        for (d, v) in dst[..4].iter_mut().zip(xywh) {
            *d = f16::from_f32(v);
        }
    }

    /// `FCVTN` (ARMv8.0, round to nearest even), the bits of `f16::from_f32`.
    #[cfg(target_arch = "aarch64")]
    #[inline(always)]
    fn store_v(dst: &mut [Self], xywh: std::arch::aarch64::float32x4_t) {
        if let Some(d) = dst.get_mut(..4) {
            let h = super::neon_convert::narrow8(xywh, xywh);
            // SAFETY: NEON is part of the aarch64 baseline; `d` holds four f16, a `u16` each.
            unsafe {
                use std::arch::aarch64::{vget_low_u16, vst1_u16};
                vst1_u16(d.as_mut_ptr().cast::<u16>(), vget_low_u16(h));
            }
        }
    }
}

/// One level's anchor geometry for the DFL box decode.
#[derive(Debug, Clone, Copy)]
pub struct DflGeom<'a> {
    /// Bins per side; a level's anchor holds `4 * reg_max` bins.
    pub reg_max: usize,
    /// Anchors in the level (`h * w`).
    pub anchors: usize,
    /// Anchor centre x in grid units, one per anchor.
    pub grid_x: &'a [f32],
    /// Anchor centre y in grid units, one per anchor.
    pub grid_y: &'a [f32],
    /// FPN stride in pixels.
    pub stride: f32,
}

/// Portable decode of one level's NHWC `[h * w, 4 * reg_max]` 8-bit DFL bins
/// into `[xc, yc, w, h]` per anchor.
#[inline]
pub fn decode_dfl_lut<T: DflBin, O: BoxOut>(
    input: &[T],
    table: &DflExpTable,
    level: &DflGeom,
    dst: &mut [O],
) {
    let reg_max = level.reg_max;
    let n = level.anchors;
    debug_assert!(reg_max > 0);
    debug_assert_eq!(input.len(), n * 4 * reg_max);
    debug_assert_eq!(dst.len(), 4 * n);
    let exp = &table.exp;
    for (a, (bins, out)) in input
        .chunks_exact(4 * reg_max)
        .zip(dst.chunks_exact_mut(4))
        .enumerate()
    {
        let ltrb = anchor_ltrb(bins, reg_max, exp);
        let xywh = dist2bbox_anchor_f32(ltrb, level.grid_x[a], level.grid_y[a], level.stride);
        O::store(out, xywh);
    }
}

/// The `[l, t, r, b]` distances of one anchor's `4 * reg_max` bins.
#[inline(always)]
fn anchor_ltrb<T: DflBin>(bins: &[T], reg_max: usize, exp: &[f32; 256]) -> [f32; 4] {
    let mut ltrb = [0.0_f32; 4];
    for (d, side) in ltrb.iter_mut().zip(bins.chunks_exact(reg_max)) {
        let max = side.iter().map(|b| b.raw()).max().unwrap_or(0);
        let (mut sum_even, mut sum_odd) = (0.0_f32, 0.0_f32);
        let (mut ws_even, mut ws_odd) = (0.0_f32, 0.0_f32);
        for (i, b) in side.iter().enumerate() {
            let e = exp[(max - b.raw()) as usize];
            if i % 2 == 0 {
                sum_even += e;
                ws_even += e * i as f32;
            } else {
                sum_odd += e;
                ws_odd += e * i as f32;
            }
        }
        *d = (ws_even + ws_odd) / (sum_even + sum_odd);
    }
    ltrb
}

/// The `[l, t, r, b]` grid-unit distances of each anchor's `4 * reg_max` 8-bit bins, four per
/// anchor in `dst`: [`decode_dfl_lut`] before its dist2bbox.
#[inline]
pub fn dfl_lut_ltrb<T: DflBin>(input: &[T], table: &DflExpTable, reg_max: usize, dst: &mut [f32]) {
    if reg_max == 0 {
        return;
    }
    for (bins, out) in input.chunks_exact(4 * reg_max).zip(dst.chunks_exact_mut(4)) {
        for (o, d) in out.iter_mut().zip(anchor_ltrb(bins, reg_max, &table.exp)) {
            *o = d;
        }
    }
}

/// The four sides of one anchor's `4 * reg_max` bins.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) fn sides<T>(bins: &[T], reg_max: usize) -> Option<[&[T]; 4]> {
    let mut it = bins.chunks_exact(reg_max);
    Some([it.next()?, it.next()?, it.next()?, it.next()?])
}

#[cfg(target_arch = "aarch64")]
pub use neon::{
    decode_dfl_lut_generic_neon, decode_dfl_lut_neon, dfl_lut_ltrb_generic_neon, dfl_lut_ltrb_neon,
    NeonDflBin,
};

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{decode_dfl_lut, dfl_lut_ltrb, sides, BoxOut, DflBin, DflExpTable, DflGeom};
    use std::arch::aarch64::*; // NOSONAR

    /// An 8-bit DFL bin type the NEON kernel can take the side maxima of. Sealed through
    /// [`DflBin`].
    pub trait NeonDflBin: DflBin {
        /// Lane `s` of the result is the largest raw bin of side `s`, as the
        /// bit pattern of the input type.
        ///
        /// # Safety
        /// NEON must be available.
        unsafe fn side_max(s: [uint8x16_t; 4]) -> uint8x16_t;

        /// The largest raw bin of `side`, `i32::MIN` when it is empty.
        fn slice_max(side: &[Self]) -> i32;
    }

    /// The largest raw value of `side`: NEON lane maxima over 16- and 8-byte steps, then the
    /// rest.
    macro_rules! slice_max {
        ($side:ident, $vmax:ident, $vmaxv:ident, $load:ident, $vmax8:ident, $load8:ident) => {{
            let mut steps = $side.chunks_exact(16);
            let mut m = i32::MIN;
            if let Some(first) = steps.next() {
                // SAFETY: NEON is part of the aarch64 baseline; each step holds 16 bytes.
                unsafe {
                    let mut v = $load(first.as_ptr());
                    for s in &mut steps {
                        v = $vmax(v, $load(s.as_ptr()));
                    }
                    m = i32::from($vmaxv(v));
                }
            }
            let mut rest = steps.remainder().chunks_exact(8);
            if let Some(half) = rest.next() {
                // SAFETY: NEON is part of the aarch64 baseline; `half` holds 8 bytes.
                m = m.max(i32::from(unsafe { $vmax8($load8(half.as_ptr())) }));
            }
            rest.remainder().iter().fold(m, |m, &b| m.max(i32::from(b)))
        }};
    }

    impl NeonDflBin for i8 {
        #[inline(always)]
        unsafe fn side_max(s: [uint8x16_t; 4]) -> uint8x16_t {
            unsafe {
                let ab = vpmaxq_s8(vreinterpretq_s8_u8(s[0]), vreinterpretq_s8_u8(s[1]));
                let cd = vpmaxq_s8(vreinterpretq_s8_u8(s[2]), vreinterpretq_s8_u8(s[3]));
                let m = vpmaxq_s8(ab, cd);
                let m = vpmaxq_s8(m, m);
                vreinterpretq_u8_s8(vpmaxq_s8(m, m))
            }
        }

        #[inline(always)]
        fn slice_max(side: &[Self]) -> i32 {
            slice_max!(side, vmaxq_s8, vmaxvq_s8, vld1q_s8, vmaxv_s8, vld1_s8)
        }
    }

    impl NeonDflBin for u8 {
        #[inline(always)]
        unsafe fn side_max(s: [uint8x16_t; 4]) -> uint8x16_t {
            unsafe {
                let m = vpmaxq_u8(vpmaxq_u8(s[0], s[1]), vpmaxq_u8(s[2], s[3]));
                let m = vpmaxq_u8(m, m);
                vpmaxq_u8(m, m)
            }
        }

        #[inline(always)]
        fn slice_max(side: &[Self]) -> i32 {
            slice_max!(side, vmaxq_u8, vmaxvq_u8, vld1q_u8, vmaxv_u8, vld1_u8)
        }
    }

    /// Table indices of eight consecutive bins of a side whose largest raw bin is `max`: byte
    /// `j` is `max − bin j`, exact in a byte for either signedness. `bins` holds 8 bins.
    #[inline(always)]
    fn keys8<T: DflBin>(bins: &[T], max: i32) -> u64 {
        let Some(b) = bins.get(..8) else {
            return 0;
        };
        // SAFETY: NEON is part of the aarch64 baseline; `b` holds 8 one-byte bins.
        unsafe {
            let v = vld1_u8(b.as_ptr().cast::<u8>());
            vget_lane_u64::<0>(vreinterpret_u64_u8(vsub_u8(vdup_n_u8(max as u8), v)))
        }
    }

    /// Weighted and plain bin sums of one anchor, side `s` in lane `s`, accumulated over even
    /// and odd bins in bin order as `anchor_ltrb` does, so the results are its bits.
    #[inline(always)]
    fn anchor_sums_any<T: NeonDflBin>(
        sides: [&[T]; 4],
        exp: &[f32; 256],
    ) -> (float32x4_t, float32x4_t) {
        let [a, b, c, d] = sides;
        let m = [
            T::slice_max(a),
            T::slice_max(b),
            T::slice_max(c),
            T::slice_max(d),
        ];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            let zero = vdupq_n_f32(0.0);
            let (mut sum_even, mut sum_odd, mut ws_even, mut ws_odd) = (zero, zero, zero, zero);
            let weights = |e: [f32; 4]| vld1q_f32(e.as_ptr());
            let groups = a
                .chunks_exact(8)
                .zip(b.chunks_exact(8))
                .zip(c.chunks_exact(8).zip(d.chunks_exact(8)));
            let mut bin = 0.0f32;
            for ((a, b), (c, d)) in groups {
                let k = [
                    keys8(a, m[0]),
                    keys8(b, m[1]),
                    keys8(c, m[2]),
                    keys8(d, m[3]),
                ];
                let w = |k: u64, j: u32| exp[usize::from((k >> (8 * j)) as u8)];
                let e = |j: u32| weights([w(k[0], j), w(k[1], j), w(k[2], j), w(k[3], j)]);
                for j in [0, 2, 4, 6] {
                    let (even, odd) = (e(j), e(j + 1));
                    sum_even = vaddq_f32(sum_even, even);
                    ws_even = vaddq_f32(ws_even, vmulq_n_f32(even, bin));
                    sum_odd = vaddq_f32(sum_odd, odd);
                    ws_odd = vaddq_f32(ws_odd, vmulq_n_f32(odd, bin + 1.0));
                    bin += 2.0;
                }
            }
            // The true difference lies in 0..=255, so the u8 cast keeps it exactly.
            let w = |m: i32, x: T| exp[usize::from((m - x.raw()) as u8)];
            fn tail<T>(v: &[T]) -> &[T] {
                v.get(v.len() / 8 * 8..).unwrap_or(&[])
            }
            let rest = tail(a).iter().zip(tail(b)).zip(tail(c).iter().zip(tail(d)));
            for (i, ((&a, &b), (&c, &d))) in rest.enumerate() {
                let e = weights([w(m[0], a), w(m[1], b), w(m[2], c), w(m[3], d)]);
                if i % 2 == 0 {
                    sum_even = vaddq_f32(sum_even, e);
                    ws_even = vaddq_f32(ws_even, vmulq_n_f32(e, bin));
                } else {
                    sum_odd = vaddq_f32(sum_odd, e);
                    ws_odd = vaddq_f32(ws_odd, vmulq_n_f32(e, bin));
                }
                bin += 1.0;
            }
            (vaddq_f32(ws_even, ws_odd), vaddq_f32(sum_even, sum_odd))
        }
    }

    /// [`dfl_lut_ltrb`] for any `reg_max` with the four sides in the lanes of one vector,
    /// bit-identical to it; writes nothing unless `dst` holds four values per anchor of `input`.
    #[inline]
    pub fn dfl_lut_ltrb_generic_neon<T: NeonDflBin>(
        input: &[T],
        table: &DflExpTable,
        reg_max: usize,
        dst: &mut [f32],
    ) {
        if reg_max == 0
            || !input.len().is_multiple_of(4 * reg_max)
            || dst.len() != input.len() / reg_max
        {
            return;
        }
        for (bins, out) in input.chunks_exact(4 * reg_max).zip(dst.chunks_exact_mut(4)) {
            let Some(sides) = sides(bins, reg_max) else {
                return;
            };
            let (ws, sum) = anchor_sums_any(sides, &table.exp);
            // SAFETY: NEON is part of the aarch64 baseline; `out` holds 4 floats.
            unsafe { vst1q_f32(out.as_mut_ptr(), vdivq_f32(ws, sum)) };
        }
    }

    /// [`decode_dfl_lut`] for any `reg_max` with the four sides in the lanes of one vector,
    /// bit-identical to it; writes nothing unless the lengths match the level.
    #[inline]
    pub fn decode_dfl_lut_generic_neon<T: NeonDflBin, O: BoxOut>(
        input: &[T],
        table: &DflExpTable,
        level: &DflGeom,
        dst: &mut [O],
    ) {
        let (reg_max, n) = (level.reg_max, level.anchors);
        if reg_max == 0
            || Some(input.len()) != n.checked_mul(4 * reg_max)
            || Some(dst.len()) != n.checked_mul(4)
            || level.grid_x.len() < n
            || level.grid_y.len() < n
        {
            return;
        }
        // SAFETY: NEON is part of the aarch64 baseline.
        let stride = unsafe { vdup_n_f32(level.stride) };
        let anchors = input.chunks_exact(4 * reg_max).zip(dst.chunks_exact_mut(4));
        for (a, (bins, out)) in anchors.enumerate() {
            let Some(sides) = sides(bins, reg_max) else {
                return;
            };
            let (ws, sum) = anchor_sums_any(sides, &table.exp);
            // SAFETY: NEON is part of the aarch64 baseline; `a < n` and both grids hold `n`.
            unsafe { store_box(vdivq_f32(ws, sum), level, a, stride, out) };
        }
    }

    /// [`decode_dfl_lut`] for `reg_max == 16` with the four sides in the
    /// lanes of one vector; other `reg_max` values take the portable kernel.
    #[inline]
    pub fn decode_dfl_lut_neon<T: NeonDflBin, O: BoxOut>(
        input: &[T],
        table: &DflExpTable,
        level: &DflGeom,
        dst: &mut [O],
    ) {
        if level.reg_max != 16 {
            return decode_dfl_lut(input, table, level, dst);
        }
        let n = level.anchors;
        assert_eq!(input.len(), n * 64);
        assert_eq!(dst.len(), 4 * n);
        assert!(level.grid_x.len() >= n && level.grid_y.len() >= n);
        // SAFETY: NEON is mandatory on aarch64, and the lengths are checked
        // above.
        unsafe { level_16(input, table, level, dst) }
    }

    /// [`dfl_lut_ltrb`] for `reg_max == 16` with the four sides in the lanes of one vector;
    /// other `reg_max` values take the portable kernel. Bit-identical to it.
    #[inline]
    pub fn dfl_lut_ltrb_neon<T: NeonDflBin>(
        input: &[T],
        table: &DflExpTable,
        reg_max: usize,
        dst: &mut [f32],
    ) {
        if reg_max != 16 {
            return dfl_lut_ltrb(input, table, reg_max, dst);
        }
        // SAFETY: NEON is mandatory on aarch64.
        unsafe { ltrb_16(input, table, dst) }
    }

    #[target_feature(enable = "neon")]
    unsafe fn ltrb_16<T: NeonDflBin>(input: &[T], table: &DflExpTable, dst: &mut [f32]) {
        unsafe {
            let exp = table.exp.as_ptr();
            for (bins, out) in input.chunks_exact(64).zip(dst.chunks_exact_mut(4)) {
                // `bins` holds 64 bytes and `out` 4 floats.
                let (ws, sum) = anchor_sums::<T>(bins.as_ptr().cast::<u8>(), exp);
                vst1q_f32(out.as_mut_ptr(), vdivq_f32(ws, sum));
            }
        }
    }

    /// Weighted and plain bin sums of one anchor's 64 bins at `p`, side `s`
    /// in lane `s`, accumulated as `decode_dfl_lut` does.
    #[inline(always)]
    unsafe fn anchor_sums<T: NeonDflBin>(
        p: *const u8,
        exp: *const f32,
    ) -> (float32x4_t, float32x4_t) {
        unsafe {
            let sides = [
                vld1q_u8(p),
                vld1q_u8(p.add(16)),
                vld1q_u8(p.add(32)),
                vld1q_u8(p.add(48)),
            ];
            let m = T::side_max(sides);
            // The true difference lies in 0..=255 for either signedness, so
            // the wrapping u8 subtraction yields it exactly.
            let k = [
                vreinterpretq_u64_u8(vsubq_u8(vdupq_laneq_u8::<0>(m), sides[0])),
                vreinterpretq_u64_u8(vsubq_u8(vdupq_laneq_u8::<1>(m), sides[1])),
                vreinterpretq_u64_u8(vsubq_u8(vdupq_laneq_u8::<2>(m), sides[2])),
                vreinterpretq_u64_u8(vsubq_u8(vdupq_laneq_u8::<3>(m), sides[3])),
            ];
            let lo = [
                vgetq_lane_u64::<0>(k[0]),
                vgetq_lane_u64::<0>(k[1]),
                vgetq_lane_u64::<0>(k[2]),
                vgetq_lane_u64::<0>(k[3]),
            ];
            let hi = [
                vgetq_lane_u64::<1>(k[0]),
                vgetq_lane_u64::<1>(k[1]),
                vgetq_lane_u64::<1>(k[2]),
                vgetq_lane_u64::<1>(k[3]),
            ];
            // The weights of bin `8 * half + j` of the four sides.
            macro_rules! weights {
                ($w:ident, $j:literal) => {{
                    let v = [
                        *exp.add((($w[0] >> (8 * $j)) & 0xff) as usize),
                        *exp.add((($w[1] >> (8 * $j)) & 0xff) as usize),
                        *exp.add((($w[2] >> (8 * $j)) & 0xff) as usize),
                        *exp.add((($w[3] >> (8 * $j)) & 0xff) as usize),
                    ];
                    vld1q_f32(v.as_ptr())
                }};
            }
            // Bin 0 adds nothing to the weighted sum, and bin 1 adds its
            // weight unscaled; adding either to a zero sum is exact.
            let mut sum_even = weights!(lo, 0);
            let e = weights!(lo, 1);
            let mut sum_odd = e;
            let mut ws_odd = e;
            let e = weights!(lo, 2);
            sum_even = vaddq_f32(sum_even, e);
            let mut ws_even = vmulq_n_f32(e, 2.0);
            macro_rules! bin {
                ($w:ident, $j:literal, $bin:literal, $sum:ident, $ws:ident) => {{
                    let e = weights!($w, $j);
                    $sum = vaddq_f32($sum, e);
                    $ws = vaddq_f32($ws, vmulq_n_f32(e, $bin));
                }};
            }
            bin!(lo, 3, 3.0, sum_odd, ws_odd);
            bin!(lo, 4, 4.0, sum_even, ws_even);
            bin!(lo, 5, 5.0, sum_odd, ws_odd);
            bin!(lo, 6, 6.0, sum_even, ws_even);
            bin!(lo, 7, 7.0, sum_odd, ws_odd);
            bin!(hi, 0, 8.0, sum_even, ws_even);
            bin!(hi, 1, 9.0, sum_odd, ws_odd);
            bin!(hi, 2, 10.0, sum_even, ws_even);
            bin!(hi, 3, 11.0, sum_odd, ws_odd);
            bin!(hi, 4, 12.0, sum_even, ws_even);
            bin!(hi, 5, 13.0, sum_odd, ws_odd);
            bin!(hi, 6, 14.0, sum_even, ws_even);
            bin!(hi, 7, 15.0, sum_odd, ws_odd);
            (vaddq_f32(ws_even, ws_odd), vaddq_f32(sum_even, sum_odd))
        }
    }

    /// `[l, t, r, b]` distances of anchor `a` to `[xc, yc, w, h]` in `out`:
    /// `(g + (r - l, b - t) / 2) * stride` and `(l + r, t + b) * stride`, the
    /// operations of `dist2bbox_anchor_f32`.
    #[inline(always)]
    unsafe fn store_box<O: BoxOut>(
        d: float32x4_t,
        level: &DflGeom,
        a: usize,
        stride: float32x2_t,
        out: &mut [O],
    ) {
        unsafe {
            let rev = vextq_f32::<2>(d, d);
            let diff = vget_low_f32(vsubq_f32(rev, d));
            let size = vget_low_f32(vaddq_f32(d, rev));
            let g = vset_lane_f32::<1>(
                *level.grid_y.get_unchecked(a),
                vdup_n_f32(*level.grid_x.get_unchecked(a)),
            );
            let xy = vmul_f32(vadd_f32(g, vmul_n_f32(diff, 0.5)), stride);
            let wh = vmul_f32(size, stride);
            O::store_v(out, vcombine_f32(xy, wh));
        }
    }

    #[target_feature(enable = "neon")]
    unsafe fn level_16<T: NeonDflBin, O: BoxOut>(
        input: &[T],
        table: &DflExpTable,
        level: &DflGeom,
        dst: &mut [O],
    ) {
        unsafe {
            let exp = table.exp.as_ptr();
            let src = input.as_ptr().cast::<u8>();
            let stride = vdup_n_f32(level.stride);
            let last = (dst.len() / 4).saturating_sub(1);
            // Two anchors per step give the in-order Cortex-A55 independent
            // work to issue while the other anchor's loads and divide retire.
            let mut pairs = dst.chunks_exact_mut(8);
            for (i, out) in pairs.by_ref().enumerate() {
                let a = 2 * i;
                let p = src.add(a * 64);
                let (ws0, sum0) = anchor_sums::<T>(p, exp);
                let (ws1, sum1) = anchor_sums::<T>(p.add(64), exp);
                let d0 = vdivq_f32(ws0, sum0);
                let d1 = vdivq_f32(ws1, sum1);
                let (out0, out1) = out.split_at_mut(4);
                store_box(d0, level, a, stride, out0);
                store_box(d1, level, a + 1, stride, out1);
            }
            let rest = pairs.into_remainder();
            if !rest.is_empty() {
                let (ws, sum) = anchor_sums::<T>(src.add(last * 64), exp);
                store_box(vdivq_f32(ws, sum), level, last, stride, rest);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prim::grids::make_anchor_grid;

    /// One level's anchor grid, which [`Grid::geom`] borrows.
    struct Grid {
        w: usize,
        h: usize,
        reg_max: usize,
        grid_x: Box<[f32]>,
        grid_y: Box<[f32]>,
    }

    impl Grid {
        fn geom(&self) -> DflGeom<'_> {
            DflGeom {
                reg_max: self.reg_max,
                anchors: self.h * self.w,
                grid_x: &self.grid_x,
                grid_y: &self.grid_y,
                stride: 16.0,
            }
        }
    }

    fn level(w: usize, h: usize, reg_max: usize) -> Grid {
        let (grid_x, grid_y) = make_anchor_grid(h, w);
        Grid {
            w,
            h,
            reg_max,
            grid_x,
            grid_y,
        }
    }

    /// DFL logits shaped like a trained head: each side peaks at a bin and
    /// falls off, with noise, and saturated or flat sides mixed in.
    fn bins(n: usize, reg_max: usize, seed: u64) -> Vec<i8> {
        let mut x = seed | 1;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 32) as i32
        };
        let mut v = Vec::with_capacity(n * 4 * reg_max);
        for _ in 0..n * 4 {
            let kind = next().rem_euclid(16);
            let peak = next().rem_euclid(reg_max as i32);
            let top = next().rem_euclid(200) - 100;
            let fall = next().rem_euclid(12);
            for i in 0..reg_max as i32 {
                let b = match kind {
                    0 => 127,
                    1 => -128,
                    2 => {
                        if i == peak {
                            127
                        } else {
                            -128
                        }
                    }
                    _ => top - fall * (i - peak).abs() + next().rem_euclid(9) - 4,
                };
                v.push(b.clamp(-128, 127) as i8);
            }
        }
        v
    }

    #[test]
    fn table_holds_exp_of_negative_steps() {
        let t = DflExpTable::new(0.25);
        assert_eq!(t.exp[0], 1.0);
        assert!((f64::from(t.exp[4]) - (-1.0_f64).exp()).abs() < 1e-7);
        assert!(t.exp.windows(2).all(|w| w[1] <= w[0]));
        let mut u = t.clone();
        u.update(0.25);
        assert_eq!(u.exp, t.exp);
        u.update(0.5);
        assert!(u.is_for(0.5) && !u.is_for(0.25));
        assert!((f64::from(u.exp[2]) - (-1.0_f64).exp()).abs() < 1e-7);
    }

    #[test]
    fn f16_output_rounds_the_f32_boxes() {
        let grid = level(4, 4, 16);
        let lvl = grid.geom();
        let input = bins(16, 16, 3);
        let table = DflExpTable::new(0.147);
        let mut f = vec![0.0_f32; 64];
        let mut h = vec![f16::ZERO; 64];
        decode_dfl_lut(&input, &table, &lvl, &mut f);
        decode_dfl_lut(&input, &table, &lvl, &mut h);
        for (a, b) in f.iter().zip(&h) {
            assert_eq!(f16::from_f32(*a), *b);
        }
    }

    /// The distances are the ones the box decode feeds to dist2bbox.
    #[test]
    fn ltrb_is_the_box_decode_before_dist2bbox() {
        for reg_max in [16, 7] {
            let grid = level(9, 5, reg_max);
            let lvl = grid.geom();
            let input = bins(45, reg_max, 5);
            let table = DflExpTable::new(0.147);
            let mut boxes = vec![0.0_f32; 180];
            decode_dfl_lut(&input, &table, &lvl, &mut boxes);
            let mut ltrb = vec![0.0_f32; 180];
            dfl_lut_ltrb(&input, &table, reg_max, &mut ltrb);
            for (a, (d, b)) in ltrb.chunks_exact(4).zip(boxes.chunks_exact(4)).enumerate() {
                let want = dist2bbox_anchor_f32(
                    [d[0], d[1], d[2], d[3]],
                    lvl.grid_x[a],
                    lvl.grid_y[a],
                    lvl.stride,
                );
                assert_eq!(b, want, "rm={reg_max} anchor {a}");
            }
            #[cfg(target_arch = "aarch64")]
            {
                let mut neon = vec![0.0_f32; 180];
                dfl_lut_ltrb_neon(&input, &table, reg_max, &mut neon);
                assert_eq!(neon, ltrb, "neon rm={reg_max}");
                let unsigned: Vec<u8> = input.iter().map(|&b| b as u8).collect();
                dfl_lut_ltrb(&unsigned, &table, reg_max, &mut ltrb);
                dfl_lut_ltrb_neon(&unsigned, &table, reg_max, &mut neon);
                assert_eq!(neon, ltrb, "neon u8 rm={reg_max}");
            }
        }
    }

    /// The generic-`reg_max` NEON kernels give the portable kernels' results bit for bit, the
    /// bound `neon_matches_portable_bit_for_bit` holds the 16-bin kernel to.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn dfl_lut_generic_neon_matches_scalar() {
        for reg_max in [1, 2, 7, 8, 15, 16, 17, 32, 64] {
            for anchors in [1, 3, 4, 5, 33] {
                let grid = level(anchors, 1, reg_max);
                let lvl = grid.geom();
                let input = bins(anchors, reg_max, (reg_max * 64 + anchors) as u64);
                let unsigned: Vec<u8> = input.iter().map(|&b| b as u8).collect();
                for scale in [0.147, 0.02, 0.9] {
                    let table = DflExpTable::new(scale);
                    let what = format!("rm={reg_max} anchors={anchors} scale={scale}");
                    let n = 4 * anchors;
                    let (mut want, mut got) = (vec![0.0_f32; n], vec![f32::NAN; n]);
                    decode_dfl_lut(&input, &table, &lvl, &mut want);
                    decode_dfl_lut_generic_neon(&input, &table, &lvl, &mut got);
                    assert_eq!(bits(&got), bits(&want), "i8 boxes {what}");
                    decode_dfl_lut(&unsigned, &table, &lvl, &mut want);
                    decode_dfl_lut_generic_neon(&unsigned, &table, &lvl, &mut got);
                    assert_eq!(bits(&got), bits(&want), "u8 boxes {what}");
                    let (mut want16, mut got16) = (vec![f16::ZERO; n], vec![f16::NAN; n]);
                    decode_dfl_lut(&input, &table, &lvl, &mut want16);
                    decode_dfl_lut_generic_neon(&input, &table, &lvl, &mut got16);
                    assert_eq!(got16, want16, "i8 f16 boxes {what}");
                    dfl_lut_ltrb(&input, &table, reg_max, &mut want);
                    dfl_lut_ltrb_generic_neon(&input, &table, reg_max, &mut got);
                    assert_eq!(bits(&got), bits(&want), "i8 ltrb {what}");
                    dfl_lut_ltrb(&unsigned, &table, reg_max, &mut want);
                    dfl_lut_ltrb_generic_neon(&unsigned, &table, reg_max, &mut got);
                    assert_eq!(bits(&got), bits(&want), "u8 ltrb {what}");
                }
            }
        }
    }

    /// Mismatched lengths leave the generic NEON kernels' outputs untouched.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn dfl_lut_generic_neon_rejects_mismatched_lengths() {
        let grid = level(3, 1, 8);
        let lvl = grid.geom();
        let table = DflExpTable::new(0.1);
        let input = bins(3, 8, 1);
        let mut boxes = vec![f32::NAN; 11];
        decode_dfl_lut_generic_neon(&input, &table, &lvl, &mut boxes);
        assert!(boxes.iter().all(|v| v.is_nan()));
        let mut ltrb = vec![f32::NAN; 12];
        decode_dfl_lut_generic_neon(&input[..95], &table, &lvl, &mut ltrb);
        assert!(ltrb.iter().all(|v| v.is_nan()));
        dfl_lut_ltrb_generic_neon(&input[..95], &table, 8, &mut ltrb);
        assert!(ltrb.iter().all(|v| v.is_nan()));
        dfl_lut_ltrb_generic_neon(&input, &table, 0, &mut ltrb);
        assert!(ltrb.iter().all(|v| v.is_nan()));
    }

    #[cfg(target_arch = "aarch64")]
    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// The NEON kernel produces the portable kernel's boxes bit for bit.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_matches_portable_bit_for_bit() {
        for (scale, reg_max) in [(0.147, 16), (0.02, 16), (0.9, 16), (0.1, 8)] {
            let table = DflExpTable::new(scale);
            let grid = level(17, 3, reg_max);
            let lvl = grid.geom();
            let input = bins(51, reg_max, 11);
            let unsigned: Vec<u8> = input.iter().map(|&b| b as u8).collect();
            let mut want = vec![0.0_f32; 204];
            let mut got = vec![0.0_f32; 204];
            decode_dfl_lut(&input, &table, &lvl, &mut want);
            decode_dfl_lut_neon(&input, &table, &lvl, &mut got);
            assert_eq!(got, want, "i8 scale={scale} rm={reg_max}");
            decode_dfl_lut(&unsigned, &table, &lvl, &mut want);
            decode_dfl_lut_neon(&unsigned, &table, &lvl, &mut got);
            assert_eq!(got, want, "u8 scale={scale} rm={reg_max}");
            let mut want16 = vec![f16::ZERO; 204];
            let mut got16 = vec![f16::ZERO; 204];
            decode_dfl_lut(&input, &table, &lvl, &mut want16);
            decode_dfl_lut_neon(&input, &table, &lvl, &mut got16);
            assert_eq!(got16, want16, "f16 scale={scale} rm={reg_max}");
        }
    }
}
