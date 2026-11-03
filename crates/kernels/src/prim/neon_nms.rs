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
//! zero-area boxes included. Only ARMv8.0 Advanced SIMD instructions are used.

#![cfg(target_arch = "aarch64")]

use std::arch::aarch64::*; // NOSONAR

/// Boxes per vector.
pub const LANES: usize = 4;

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
            while j + 2 * LANES <= n {
                let (m0, m1) = (hit(j), hit(j + LANES));
                vst1q_u32(d.add(j), vorrq_u32(vld1q_u32(d.add(j)), m0));
                vst1q_u32(d.add(j + LANES), vorrq_u32(vld1q_u32(d.add(j + LANES)), m1));
                j += 2 * LANES;
            }
            if j < n {
                let m = hit(j);
                vst1q_u32(d.add(j), vorrq_u32(vld1q_u32(d.add(j)), m));
            }
        }
        survivors += 1;
        if survivors >= cap {
            break;
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
