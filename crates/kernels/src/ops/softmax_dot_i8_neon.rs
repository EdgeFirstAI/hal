// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! NEON softmax-then-dot over 16 rows of 16 at once.
//!
//! The rows are transposed in registers so each lane holds one row and element `j` of all
//! sixteen rows sits in one vector. A row's exponentials relative to its maximum,
//! `exp(scale·d)` for `d = max − x`, come from two 16-entry tables as `2·A[d>>4]·B[d&15] / 2¹⁶`
//! (one `sqrdmulh`), at most 4095, so a row's sixteen terms sum exactly in a u16 lane and its
//! weighted sum exactly in an i32 lane. One division per row gives the expectation, which is
//! requantized as the matrix multiplication requantizes an accumulator.

use std::arch::aarch64::*; // NOSONAR

use super::matmul_i8_neon::{narrow16, requant4};
use super::rows::RowGroup;
use super::softmax_dot_i8::SoftmaxDotI8Plan;
use super::softmax_i8_neon::{load16, SplitExp};

/// The results of 16 rows of 16 logits each, written to the group's 16 one-element output
/// rows. Does nothing unless the group is 16 rows of 16 with 16 weights.
///
/// # Safety
/// The CPU must support NEON, which every aarch64 CPU does.
#[target_feature(enable = "neon")]
pub(super) unsafe fn rows16(p: &SoftmaxDotI8Plan, g: &RowGroup<'_, i8>) {
    let weights = p.weights();
    if g.count() != 16 || g.len() != 16 || g.dst_len() != 1 || weights.len() != 16 {
        return;
    }
    // SAFETY: the group is 16 rows of 16 (checked above).
    let (v, max) = unsafe { load16(g) };
    let tables = SplitExp::new(p.exp12());
    // Rows 0..8 and 8..16 of the sums; rows 4q..4q + 4 of the weighted sums.
    let mut sum = [vdupq_n_u16(0); 2];
    let mut ws = [vdupq_n_s32(0); 4];
    // Four elements per step, each stage across all four, so the in-order core always has
    // independent work.
    for (x, w) in v.as_chunks::<4>().0.iter().zip(weights.as_chunks::<4>().0) {
        let (a, b) = tables.entries(max, x);
        let s16 = vreinterpretq_s16_u16;
        for u in 0..4 {
            let e = [
                vqrdmulhq_s16(s16(a[u][0]), s16(b[u][0])),
                vqrdmulhq_s16(s16(a[u][1]), s16(b[u][1])),
            ];
            sum[0] = vaddq_u16(sum[0], vreinterpretq_u16_s16(e[0]));
            sum[1] = vaddq_u16(sum[1], vreinterpretq_u16_s16(e[1]));
            ws[0] = vmlal_n_s16(ws[0], vget_low_s16(e[0]), w[u]);
            ws[1] = vmlal_high_n_s16(ws[1], e[0], w[u]);
            ws[2] = vmlal_n_s16(ws[2], vget_low_s16(e[1]), w[u]);
            ws[3] = vmlal_high_n_s16(ws[3], e[1], w[u]);
        }
    }
    let sums: [float32x4_t; 4] = core::array::from_fn(|q| {
        vcvtq_f32_u32(if q % 2 == 0 {
            vmovl_u16(vget_low_u16(sum[q / 2]))
        } else {
            vmovl_high_u16(sum[q / 2])
        })
    });
    // The expectation in accumulator units, rounded once.
    let unit = p.unit();
    let acc: [int32x4_t; 4] = core::array::from_fn(|i| {
        vcvtaq_s32_f32(vmulq_n_f32(vdivq_f32(vcvtq_f32_s32(ws[i]), sums[i]), unit))
    });
    let (rq, bias, rounding, zo, lo, hi) = p.requant();
    let (bias, zo) = (vdupq_n_s32(bias), vdupq_n_s32(zo));
    let (lo, hi) = (vdupq_n_s32(lo), vdupq_n_s32(hi));
    let q: [int32x4_t; 4] = core::array::from_fn(|i| {
        let x = vaddq_s32(requant4(vaddq_s32(acc[i], bias), rq, rounding), zo);
        vminq_s32(vmaxq_s32(x, lo), hi)
    });
    let out = narrow16(q);
    if g.dst_contiguous() {
        // SAFETY: the 16 one-element output rows are consecutive and this group's alone.
        unsafe { vst1q_s8(g.dst_ptr(0), out) };
        return;
    }
    let mut lanes = [0i8; 16];
    // SAFETY: `lanes` holds 16 bytes.
    unsafe { vst1q_s8(lanes.as_mut_ptr(), out) };
    for (k, &b) in lanes.iter().enumerate() {
        // SAFETY: output row `k` is writable for one element and this group's alone.
        unsafe { g.dst_ptr(k).write(b) };
    }
}

#[cfg(test)]
mod tests {
    use super::super::softmax_dot_i8::tests::{dfl_weights, params};
    use super::*;
    use crate::cpu::CpuFeatures;
    use crate::view::{TensorMut, TensorRef, View, ViewMut};
    use rand::{RngExt, SeedableRng};

    #[test]
    fn neon_rows_match_the_scalar_tier_closely_and_without_bias() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5D07);
        let rows = 4096;
        let (mut off, mut net) = (0usize, 0i64);
        for &scale in &[0.03f32, 0.0947, 0.2, 0.45] {
            let p = params(scale, dfl_weights());
            let x: Vec<i8> = (0..rows * 16).map(|_| rng.random()).collect();
            let run = |f: CpuFeatures| {
                let plan = SoftmaxDotI8Plan::with_features(p.clone(), f).unwrap();
                let mut out = vec![0i8; rows];
                plan.run(
                    TensorRef::I8(View::contiguous(&x, &[rows, 16]).unwrap()),
                    TensorMut::I8(ViewMut::contiguous(&mut out, &[rows, 1]).unwrap()),
                    None,
                )
                .unwrap();
                out
            };
            let neon = run(CpuFeatures::from_forced_tier(Some("neon")).unwrap());
            let scalar = run(CpuFeatures::default());
            for (r, (&a, &b)) in neon.iter().zip(&scalar).enumerate() {
                let d = i64::from(a) - i64::from(b);
                assert!(d.abs() <= 1, "scale {scale} row {r}: neon {a} scalar {b}");
                off += usize::from(d != 0);
                net += d;
            }
        }
        let total = 4 * rows;
        assert!(off * 100 <= total, "{off} of {total} outputs off by one");
        assert!(
            net.unsigned_abs() as usize * 4 <= off.max(8),
            "bias {net} over {off}"
        );
    }
}
