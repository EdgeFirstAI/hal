// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON greedy non-maximum suppression over boxes sorted by descending score.
//!
//! Boxes are a structure of arrays (`xmin`, `ymin`, `xmax`, `ymax`, optional labels) padded to a
//! multiple of four, with a suppression flag per box. Each overlap test is the decoder's scalar
//! `float::jaccard(b_j, b_i, iou)` lane for lane: `FMAXNM`/`FMINNM` (what `f32::max`/`f32::min`
//! compile to), `inter = max(right − left, 0) · max(bottom − top, 0)`,
//! `union = (area_j + area_i) − inter`, overlap when `inter > iou · union`, every step rounded
//! and none fused, so each decision equals the scalar one on every input, NaN, infinite and
//! zero-area boxes included. [`nms_greedy_neon`] and [`jaccard4_neon`] use only ARMv8.0 Advanced
//! SIMD instructions; [`nms_greedy_neon_fp16`] runs the test in FP16 on centre and size, eight
//! boxes per vector.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR

use half::f16;

use super::fp16_asm::{
    fabd_f16x8, fadd_f16x8, fcmgt_f16x8, fmax_f16x8, fmin_f16x8, fmul_f16x8, fsub_f16x8,
};

/// Boxes per vector.
pub const LANES: usize = 4;

/// Boxes per vector of the FP16 kernel.
pub const LANES_FP16: usize = 8;

/// Lanes at or after `k` in a block, for the block holding the first box after the reference.
const FROM_LANE: [[u32; 4]; 4] = [
    [!0, !0, !0, !0],
    [0, !0, !0, !0],
    [0, 0, !0, !0],
    [0, 0, 0, !0],
];

/// The reference box `i`, broadcast.
struct Reference {
    xmin: float32x4_t,
    ymin: float32x4_t,
    xmax: float32x4_t,
    ymax: float32x4_t,
    area: float32x4_t,
    iou: float32x4_t,
}

impl Reference {
    #[inline(always)]
    fn new(b: [f32; 4], iou: f32) -> Self {
        let [xmin, ymin, xmax, ymax] = b;
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            Self {
                xmin: vdupq_n_f32(xmin),
                ymin: vdupq_n_f32(ymin),
                xmax: vdupq_n_f32(xmax),
                ymax: vdupq_n_f32(ymax),
                area: vdupq_n_f32((xmax - xmin) * (ymax - ymin)),
                iou: vdupq_n_f32(iou),
            }
        }
    }

    /// Lanes where `jaccard(b, self, iou)` holds for the four boxes `b`.
    #[inline(always)]
    fn overlaps(&self, b: [float32x4_t; 4]) -> uint32x4_t {
        let [xmin, ymin, xmax, ymax] = b;
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            let zero = vdupq_n_f32(0.0);
            let left = vmaxnmq_f32(xmin, self.xmin);
            let top = vmaxnmq_f32(ymin, self.ymin);
            let right = vminnmq_f32(xmax, self.xmax);
            let bottom = vminnmq_f32(ymax, self.ymax);
            let inter = vmulq_f32(
                vmaxnmq_f32(vsubq_f32(right, left), zero),
                vmaxnmq_f32(vsubq_f32(bottom, top), zero),
            );
            let area = vmulq_f32(vsubq_f32(xmax, xmin), vsubq_f32(ymax, ymin));
            let union = vsubq_f32(vaddq_f32(area, self.area), inter);
            vcgtq_f32(inter, vmulq_f32(self.iou, union))
        }
    }

    /// [`Self::overlaps`] for four vectors of boxes, one operation at a time across them: an
    /// in-order core then overlaps the four latency chains.
    #[inline(always)]
    fn overlaps4(&self, b: [[float32x4_t; 4]; 4]) -> [uint32x4_t; 4] {
        const K: [usize; 4] = [0, 1, 2, 3];
        // SAFETY: NEON is part of the aarch64 baseline.
        unsafe {
            let zero = vdupq_n_f32(0.0);
            let left = b.map(|[xmin, ..]| vmaxnmq_f32(xmin, self.xmin));
            let right = b.map(|[_, _, xmax, _]| vminnmq_f32(xmax, self.xmax));
            let top = b.map(|[_, ymin, ..]| vmaxnmq_f32(ymin, self.ymin));
            let bottom = b.map(|[.., ymax]| vminnmq_f32(ymax, self.ymax));
            let w = K.map(|k| vsubq_f32(right[k], left[k]));
            let h = K.map(|k| vsubq_f32(bottom[k], top[k]));
            let bw = b.map(|[xmin, _, xmax, _]| vsubq_f32(xmax, xmin));
            let bh = b.map(|[_, ymin, _, ymax]| vsubq_f32(ymax, ymin));
            let w = w.map(|v| vmaxnmq_f32(v, zero));
            let h = h.map(|v| vmaxnmq_f32(v, zero));
            let area = K.map(|k| vmulq_f32(bw[k], bh[k]));
            let inter = K.map(|k| vmulq_f32(w[k], h[k]));
            let sum = area.map(|a| vaddq_f32(a, self.area));
            let union = K.map(|k| vsubq_f32(sum[k], inter[k]));
            let bound = union.map(|u| vmulq_f32(self.iou, u));
            K.map(|k| vcgtq_f32(inter[k], bound[k]))
        }
    }
}

/// Greedy NMS: in index order, every box `i` not flagged in `dead` flags each later box `j` it
/// overlaps (`jaccard(b_j, b_i, iou)`, and only boxes with its label when `labels` is given),
/// stopping once `cap` boxes have survived. On return `dead[j] != 0` marks the boxes flagged
/// before the call and those suppressed.
///
/// The four coordinate slices, `labels` and `dead` must share one length, a multiple of
/// [`LANES`]; padding boxes must be flagged dead. Returns `false` and writes nothing otherwise.
///
/// # Safety
/// The CPU must support NEON (every aarch64 CPU does).
#[target_feature(enable = "neon")]
pub unsafe fn nms_greedy_neon(
    coords: [&[f32]; 4],
    labels: Option<&[u32]>,
    iou: f32,
    cap: usize,
    dead: &mut [u32],
) -> bool {
    let n = dead.len();
    if !n.is_multiple_of(LANES)
        || coords.iter().any(|c| c.len() != n)
        || labels.is_some_and(|l| l.len() != n)
    {
        return false;
    }
    let [xmin, ymin, xmax, ymax] = coords;
    let mut survivors = 0usize;
    for i in 0..n {
        // SAFETY: `i < n`, the length of every slice.
        let (is_dead, b, label) = unsafe {
            (
                *dead.get_unchecked(i) != 0,
                [
                    *xmin.get_unchecked(i),
                    *ymin.get_unchecked(i),
                    *xmax.get_unchecked(i),
                    *ymax.get_unchecked(i),
                ],
                labels.map(|l| *l.get_unchecked(i)),
            )
        };
        if is_dead {
            continue;
        }
        let r = Reference::new(b, iou);
        let first = (i + 1) & !(LANES - 1);
        // SAFETY: every block `j..j + 4` lies below `n`, a multiple of four.
        unsafe {
            let d = dead.as_mut_ptr();
            let load = |j: usize| {
                [
                    vld1q_f32(xmin.as_ptr().add(j)),
                    vld1q_f32(ymin.as_ptr().add(j)),
                    vld1q_f32(xmax.as_ptr().add(j)),
                    vld1q_f32(ymax.as_ptr().add(j)),
                ]
            };
            let same = label.map(|l| vdupq_n_u32(l));
            let hit = |j: usize| match (labels, same) {
                (Some(l), Some(same)) => vandq_u32(
                    r.overlaps(load(j)),
                    vceqq_u32(vld1q_u32(l.as_ptr().add(j)), same),
                ),
                _ => r.overlaps(load(j)),
            };
            let mut j = first;
            if j < n {
                let keep_before = vld1q_u32(FROM_LANE[(i + 1) % LANES].as_ptr());
                let m = vandq_u32(hit(j), keep_before);
                vst1q_u32(d.add(j), vorrq_u32(vld1q_u32(d.add(j)), m));
                j += LANES;
            }
            while j + 4 * LANES <= n {
                let o = [j, j + LANES, j + 2 * LANES, j + 3 * LANES];
                let mut m = r.overlaps4(o.map(load));
                if let (Some(l), Some(same)) = (labels, same) {
                    for (m, k) in m.iter_mut().zip(o) {
                        *m = vandq_u32(*m, vceqq_u32(vld1q_u32(l.as_ptr().add(k)), same));
                    }
                }
                for (m, k) in m.into_iter().zip(o) {
                    vst1q_u32(d.add(k), vorrq_u32(vld1q_u32(d.add(k)), m));
                }
                j += 4 * LANES;
            }
            while j < n {
                let m = hit(j);
                vst1q_u32(d.add(j), vorrq_u32(vld1q_u32(d.add(j)), m));
                j += LANES;
            }
        }
        survivors += 1;
        if survivors >= cap {
            break;
        }
    }
    true
}

/// Lanes at or after `k` in a block of eight.
const FROM_LANE_FP16: [[u16; 8]; 8] = {
    let mut t = [[0u16; 8]; 8];
    let mut k = 0;
    while k < 8 {
        let mut lane = k;
        while lane < 8 {
            t[k][lane] = !0;
            lane += 1;
        }
        k += 1;
    }
    t
};

/// `p` as eight f16 lanes.
///
/// # Safety
/// `p` must point at eight readable `f16`.
#[inline(always)]
unsafe fn load_f16x8(p: *const f16) -> uint16x8_t {
    // SAFETY: the caller guarantees eight readable values; `f16` is a transparent `u16`.
    unsafe { vld1q_u16(p.cast::<u16>()) }
}

/// Greedy NMS with every overlap test in FP16, eight boxes per vector, over boxes given as
/// `[cx, cy, w, h]` (centre and size): in index order, each box `i` not flagged in `dead` flags
/// every later box `j` with `inter > iou · (area_j + area_i − inter)`, stopping once `cap`
/// boxes have survived. On return `dead[j] != 0` marks the boxes flagged before the call and
/// those suppressed.
///
/// The overlap along an axis is `max(min(w_j/2 + w_i/2 − |c_j − c_i|, w_j, w_i), 0)`, which
/// keeps a box's extent at f16 relative precision wherever its centre lies. Each step is
/// rounded to f16, so a pair within about 2⁻⁹ of the threshold may be decided differently than
/// [`nms_greedy_neon`] decides it. Sizes must be scaled so that areas and their sums stay finite
/// and normal in f16; a box with a NaN field never suppresses and is never suppressed.
///
/// The box slices and `dead` must share one length `n`, a multiple of [`LANES_FP16`], and
/// `scratch` (overwritten) must hold `3 · n` values; padding boxes must be flagged dead.
/// Returns `false` and writes nothing otherwise.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[target_feature(enable = "neon")]
pub unsafe fn nms_greedy_neon_fp16(
    boxes: [&[f16]; 4],
    scratch: &mut [f16],
    iou: f16,
    cap: usize,
    dead: &mut [u16],
) -> bool {
    const L: usize = LANES_FP16;
    let n = dead.len();
    if !n.is_multiple_of(L) || boxes.iter().any(|c| c.len() != n) || scratch.len() != 3 * n {
        return false;
    }
    let [cx, cy, w, h] = boxes.map(<[f16]>::as_ptr);
    let (ex, rest) = scratch.split_at_mut(n);
    let (ey, area) = rest.split_at_mut(n);
    let (ex, ey, area) = (ex.as_mut_ptr(), ey.as_mut_ptr(), area.as_mut_ptr());
    let d = dead.as_mut_ptr();
    // SAFETY: every block `j..j + 8` lies below `n`, a multiple of eight and the length of
    // every array; the caller guarantees FEAT_FP16.
    unsafe {
        let half = vdupq_n_u16(f16::from_f32_const(0.5).to_bits());
        let mut j = 0;
        while j < n {
            let (bw, bh) = (load_f16x8(w.add(j)), load_f16x8(h.add(j)));
            vst1q_u16(ex.add(j).cast::<u16>(), fmul_f16x8(bw, half));
            vst1q_u16(ey.add(j).cast::<u16>(), fmul_f16x8(bh, half));
            vst1q_u16(area.add(j).cast::<u16>(), fmul_f16x8(bw, bh));
            j += L;
        }
        let (ex, ey, area) = (ex.cast_const(), ey.cast_const(), area.cast_const());
        let zero = vdupq_n_u16(0);
        let viou = vdupq_n_u16(iou.to_bits());
        let mut survivors = 0usize;
        for i in 0..n {
            if *d.add(i) != 0 {
                continue;
            }
            let splat = |p: *const f16| vdupq_n_u16((*p.add(i)).to_bits());
            let (rcx, rcy, rw, rh) = (splat(cx), splat(cy), splat(w), splat(h));
            let (rex, rey, rarea) = (splat(ex), splat(ey), splat(area));
            // Four blocks advance one operation at a time: an in-order core then overlaps
            // their latency chains instead of running one block's chain to its end.
            let hit4 = |j: usize| {
                let o = [j, j + L, j + 2 * L, j + 3 * L];
                let b = [0, 1, 2, 3];
                let dx = o.map(|k| fabd_f16x8(load_f16x8(cx.add(k)), rcx));
                let dy = o.map(|k| fabd_f16x8(load_f16x8(cy.add(k)), rcy));
                let sx = o.map(|k| fadd_f16x8(load_f16x8(ex.add(k)), rex));
                let sy = o.map(|k| fadd_f16x8(load_f16x8(ey.add(k)), rey));
                let px = b.map(|b| fsub_f16x8(sx[b], dx[b]));
                let py = b.map(|b| fsub_f16x8(sy[b], dy[b]));
                let px = b.map(|b| fmin_f16x8(px[b], load_f16x8(w.add(o[b]))));
                let py = b.map(|b| fmin_f16x8(py[b], load_f16x8(h.add(o[b]))));
                let px = px.map(|v| fmin_f16x8(v, rw));
                let py = py.map(|v| fmin_f16x8(v, rh));
                let px = px.map(|v| fmax_f16x8(v, zero));
                let py = py.map(|v| fmax_f16x8(v, zero));
                let sum = o.map(|k| fadd_f16x8(load_f16x8(area.add(k)), rarea));
                let inter = b.map(|b| fmul_f16x8(px[b], py[b]));
                let union = b.map(|b| fsub_f16x8(sum[b], inter[b]));
                let bound = union.map(|u| fmul_f16x8(viou, u));
                b.map(|b| fcmgt_f16x8(inter[b], bound[b]))
            };
            let hit = |j: usize| {
                let px = fsub_f16x8(
                    fadd_f16x8(load_f16x8(ex.add(j)), rex),
                    fabd_f16x8(load_f16x8(cx.add(j)), rcx),
                );
                let py = fsub_f16x8(
                    fadd_f16x8(load_f16x8(ey.add(j)), rey),
                    fabd_f16x8(load_f16x8(cy.add(j)), rcy),
                );
                let px = fmin_f16x8(fmin_f16x8(px, load_f16x8(w.add(j))), rw);
                let py = fmin_f16x8(fmin_f16x8(py, load_f16x8(h.add(j))), rh);
                let inter = fmul_f16x8(fmax_f16x8(px, zero), fmax_f16x8(py, zero));
                let union = fsub_f16x8(fadd_f16x8(load_f16x8(area.add(j)), rarea), inter);
                fcmgt_f16x8(inter, fmul_f16x8(viou, union))
            };
            let flag = |j: usize, m: uint16x8_t| {
                vst1q_u16(d.add(j), vorrq_u16(vld1q_u16(d.add(j)), m));
            };
            let mut j = (i + 1) & !(L - 1);
            if j < n {
                let after = vld1q_u16(FROM_LANE_FP16[(i + 1) % L].as_ptr());
                flag(j, vandq_u16(hit(j), after));
                j += L;
            }
            while j + 4 * L <= n {
                let m = hit4(j);
                for (b, m) in m.into_iter().enumerate() {
                    flag(j + b * L, m);
                }
                j += 4 * L;
            }
            while j < n {
                flag(j, hit(j));
                j += L;
            }
            survivors += 1;
            if survivors >= cap {
                break;
            }
        }
    }
    true
}

/// `[jaccard(a, b[k], iou); 4]` for four boxes `[xmin, ymin, xmax, ymax]`, equal to the scalar
/// test on every input (`jaccard` is symmetric in its boxes).
///
/// # Safety
/// The CPU must support NEON (every aarch64 CPU does).
#[inline]
#[target_feature(enable = "neon")]
pub unsafe fn jaccard4_neon(a: [f32; 4], b: &[[f32; 4]; 4], iou: f32) -> [bool; 4] {
    // `LD4` de-interleaves the four boxes into coordinate vectors.
    // SAFETY: `b` holds 16 contiguous f32.
    let v = unsafe { vld4q_f32(b.as_ptr().cast::<f32>()) };
    let m = Reference::new(a, iou).overlaps([v.0, v.1, v.2, v.3]);
    let mut out = [0u32; 4];
    // SAFETY: `out` holds four u32.
    unsafe { vst1q_u32(out.as_mut_ptr(), m) };
    out.map(|x| x != 0)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
