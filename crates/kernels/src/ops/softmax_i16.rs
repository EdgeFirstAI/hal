// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! int16 `SOFTMAX` over the last axis, matching TFLite's all-integer `reference_ops::SoftmaxInt16`.

use std::borrow::Cow;

use super::rows::{for_each_row, StridedRow, StridedRowMut};
use crate::cpu::{CpuFeatures, Tier};
use crate::error::{KernelError, Result};
use crate::pool::Pool;
use crate::quant::Quant;
use crate::tflite::{
    lut_lookup_i16, lut_populate_i16, multiply_by_quantized_multiplier, quantize_multiplier,
};
use crate::view::{TensorMut, TensorRef};

/// Longest row whose sum of exps (each at most 32767) fits in an `i32`.
const MAX_ROW: usize = (i32::MAX / 32767) as usize;

/// Largest input left shift for which `(x - max) << shift` cannot overflow an `i32`.
const MAX_LEFT_SHIFT: i32 = 15;

/// Parameters of an int16 softmax.
///
/// TFLite requires input zero point 0, and output zero point 0 with scale 1/32768 (within 0.1%).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftmaxI16Params {
    pub input: Quant,
    pub output: Quant,
    pub beta: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    Portable,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// The per-plan constants a row kernel needs: the two tables and the input rescale.
#[derive(Clone, Debug)]
pub(super) struct RowConsts {
    pub(super) exp_lut: [i16; 513],
    pub(super) recip_lut: [i16; 513],
    pub(super) input_multiplier: i32,
    pub(super) input_left_shift: i32,
}

impl RowConsts {
    /// The tables TFLite's `SoftmaxPrepare` builds for int16, with the given input rescale.
    pub(super) fn new(input_multiplier: i32, input_left_shift: i32) -> Self {
        let range = f64::from(i32::from(i16::MAX) - i32::from(i16::MIN));
        Self {
            exp_lut: lut_populate_i16(10.0 / range, i32::from(i16::MAX), 2.0 / range, 0, f64::exp),
            recip_lut: lut_populate_i16(1.0 / range, i32::from(i16::MIN), 2.0 / range, 0, |v| {
                1.0 / (1.0 + v)
            }),
            input_multiplier,
            input_left_shift,
        }
    }
}

/// A validated int16 softmax over the last axis; every other axis indexes independent rows.
///
/// Reproduces TFLite's `reference_ops::SoftmaxInt16` bit for bit on every tier: it is all
/// integer, so the order of the row sum does not matter.
///
/// ```
/// use edgefirst_kernels::{Quant, SoftmaxI16Params, SoftmaxI16Plan, TensorMut, TensorRef, View, ViewMut};
///
/// let plan = SoftmaxI16Plan::new(SoftmaxI16Params {
///     input: Quant::new(1.0 / 2048.0, 0),
///     output: Quant::new(1.0 / 32768.0, 0),
///     beta: 1.0,
/// })?;
/// let input = [0i16; 4];
/// let mut output = [0i16; 4];
/// plan.run(
///     TensorRef::I16(View::contiguous(&input, &[1, 4])?),
///     TensorMut::I16(ViewMut::contiguous(&mut output, &[1, 4])?),
///     None,
/// )?;
/// assert_eq!(output, [8192; 4]);
/// # Ok::<(), edgefirst_kernels::KernelError>(())
/// ```
#[derive(Clone, Debug)]
pub struct SoftmaxI16Plan {
    consts: RowConsts,
    kernel: Kernel,
}

fn finite_positive(v: f32) -> bool {
    v.is_finite() && v > 0.0
}

impl SoftmaxI16Plan {
    /// Builds a plan for the CPU this process runs on, honouring `EDGEFIRST_KERNELS_FORCE_TIER`.
    pub fn new(p: SoftmaxI16Params) -> Result<Self> {
        Self::with_features(p, CpuFeatures::from_env_or_probe()?)
    }

    /// Builds a plan that runs the best kernel `f` allows.
    ///
    /// `input.scale * beta` must be below about 5 (`Unsupported` otherwise): beyond that TFLite's
    /// int32 rescale of the input difference overflows.
    pub fn with_features(p: SoftmaxI16Params, f: CpuFeatures) -> Result<Self> {
        if !finite_positive(p.input.scale) {
            return Err(KernelError::InvalidArgument(
                "softmax_i16 input scale must be finite and positive",
            ));
        }
        if !finite_positive(p.output.scale) {
            return Err(KernelError::InvalidArgument(
                "softmax_i16 output scale must be finite and positive",
            ));
        }
        if !finite_positive(p.beta) {
            return Err(KernelError::InvalidArgument(
                "softmax_i16 beta must be finite and positive",
            ));
        }
        // An f32 product, as TFLite computes it, before widening to f64.
        let scale_beta = p.input.scale * p.beta;
        if !scale_beta.is_finite() {
            return Err(KernelError::InvalidArgument(
                "softmax_i16 input scale times beta overflows",
            ));
        }
        if p.input.zero_point != 0 {
            return Err(KernelError::Unsupported(Cow::Borrowed(
                "softmax_i16 needs input zero point 0",
            )));
        }
        if p.output.zero_point != 0 {
            return Err(KernelError::Unsupported(Cow::Borrowed(
                "softmax_i16 needs output zero point 0",
            )));
        }
        let want = 1.0f32 / 32768.0;
        if (p.output.scale - want).abs() > 0.001f32 * 1.0 / 32768.0 {
            return Err(KernelError::Unsupported(Cow::Borrowed(
                "softmax_i16 needs output scale 1/32768",
            )));
        }
        let (multiplier, shift) = quantize_multiplier(f64::from(scale_beta) / (10.0 / 65535.0));
        if shift > MAX_LEFT_SHIFT {
            return Err(KernelError::Unsupported(Cow::Borrowed(
                "softmax_i16 needs input scale times beta below 5",
            )));
        }
        Ok(Self {
            consts: RowConsts::new(multiplier, shift),
            kernel: select_kernel(f),
        })
    }

    /// The tier of the kernel this plan runs; every NEON tier runs the baseline NEON kernel.
    pub fn tier(&self) -> Tier {
        match self.kernel {
            Kernel::Portable => Tier::Scalar,
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => Tier::Neon,
        }
    }

    /// Softmax of `input` into `output`, both i16 with the same shape of rank ≥ 1.
    ///
    /// Rows may hold at most 65538 elements (`Unsupported` otherwise), the most whose sum of
    /// exps fits TFLite's int32 accumulator. Rows are split across `pool` when one is given.
    /// Output axes sorted by stride must each step past the extent of the smaller ones.
    /// Arguments are validated before anything is written.
    pub fn run(
        &self,
        input: TensorRef<'_>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
    ) -> Result<()> {
        let _span =
            tracing::trace_span!("kernels.softmax_i16", tier = self.tier().as_str()).entered();
        let (TensorRef::I16(input), TensorMut::I16(mut output)) = (input, output) else {
            return Err(KernelError::InvalidArgument(
                "softmax_i16 needs i16 input and output",
            ));
        };
        if input.shape() != output.shape() {
            return Err(KernelError::InvalidArgument(
                "softmax_i16 input and output shapes differ",
            ));
        }
        if input.shape().last().is_some_and(|&n| n > MAX_ROW) {
            return Err(KernelError::Unsupported(Cow::Borrowed(
                "softmax_i16 rows are limited to 65538 elements",
            )));
        }
        for_each_row(
            &input,
            &mut output,
            pool,
            &|s, d| self.row(s, d),
            &|s, d| self.row_strided(s, d),
        )
    }

    #[inline]
    fn row(&self, src: &[i16], dst: &mut [i16]) {
        match self.kernel {
            Kernel::Portable => softmax_row_i16_portable(&self.consts, src, dst),
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is part of the aarch64 baseline.
            Kernel::Neon => unsafe {
                super::softmax_i16_neon::softmax_row_i16_neon(&self.consts, src, dst)
            },
        }
    }

    /// Long strided rows: the output cannot be read back, so the exps are computed twice.
    fn row_strided(&self, src: StridedRow<'_, i16>, dst: &mut StridedRowMut<'_, i16>) {
        if src.len() != dst.len() {
            return;
        }
        let c = &self.consts;
        let Some(max) = src.iter().max() else {
            return;
        };
        let sum = src
            .iter()
            .fold(0i32, |s, x| s.wrapping_add(i32::from(exp_q015(c, x, max))));
        let (recip, shift) = reciprocal(c, sum);
        for (k, x) in src.iter().enumerate() {
            dst.set(k, rescale(exp_q015(c, x, max), recip, shift));
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn select_kernel(f: CpuFeatures) -> Kernel {
    if f.neon_baseline {
        Kernel::Neon
    } else {
        Kernel::Portable
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn select_kernel(_: CpuFeatures) -> Kernel {
    Kernel::Portable
}

/// `SoftMaxCalculateExp`: `exp(x - max)` in Q0.15 through the exp table.
#[inline(always)]
pub(super) fn exp_q015(c: &RowConsts, x: i16, max: i16) -> i16 {
    let diff = i32::from(x) - i32::from(max);
    let scaled = multiply_by_quantized_multiplier(diff, c.input_multiplier, c.input_left_shift);
    let sym = scaled.wrapping_add(32767).clamp(-32768, 32767) as i16;
    lut_lookup_i16(sym, &c.exp_lut)
}

/// `1 / sum_of_exps` in Q0.15 through the `1/(1+x)` table, and the output right shift.
#[inline]
pub(super) fn reciprocal(c: &RowConsts, sum: i32) -> (i16, u32) {
    let headroom_plus_one = (sum as u32).leading_zeros();
    let shifted = ((i64::from(sum) << headroom_plus_one.saturating_sub(1)) + (1 << 13)) >> 14;
    let sym = (shifted as i32).wrapping_sub((1 << 15) + (1 << 16));
    let recip = lut_lookup_i16(sym.clamp(-32768, 32767) as i16, &c.recip_lut);
    // A non-empty row's sum is at least the max's exp, so the shift is in 14..=30.
    let right_shift = 31u32.saturating_sub(headroom_plus_one).max(1);
    (recip, right_shift)
}

/// `(exp * recip + round) >> right_shift`, clamped to `0..=32767`.
#[inline(always)]
pub(super) fn rescale(exp: i16, recip: i16, right_shift: u32) -> i16 {
    let shift = right_shift.min(31);
    let round = 1i64 << shift.saturating_sub(1);
    let r = ((i64::from(exp) * i64::from(recip) + round) >> shift) as i32;
    r.clamp(0, 32767) as i16
}

/// The reference row, caching the exps in `dst` between the passes as TFLite does.
///
/// Processes `min(src.len(), dst.len())` elements.
pub(super) fn softmax_row_i16_portable(c: &RowConsts, src: &[i16], dst: &mut [i16]) {
    let n = src.len().min(dst.len());
    let (Some(src), Some(dst)) = (src.get(..n), dst.get_mut(..n)) else {
        return;
    };
    let Some(&max) = src.iter().max() else {
        return;
    };
    let mut sum = 0i32;
    for (d, &x) in dst.iter_mut().zip(src) {
        *d = exp_q015(c, x, max);
        sum = sum.wrapping_add(i32::from(*d));
    }
    let (recip, shift) = reciprocal(c, sum);
    for d in dst.iter_mut() {
        *d = rescale(*d, recip, shift);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tflite::quantize_multiplier;
    use rand::{RngExt, SeedableRng};

    // Scalar models of the NEON instructions the row kernel uses, per the Arm pseudocode.
    fn vqrdmulh(a: i32, b: i32) -> i32 {
        let v = (2 * i128::from(a) * i128::from(b) + (1 << 31)) >> 32;
        v.clamp(i128::from(i32::MIN), i128::from(i32::MAX)) as i32
    }
    fn vrshl(x: i32, shift: i32) -> i32 {
        if shift >= 0 {
            x.wrapping_shl(shift as u32)
        } else {
            let r = (-shift) as u32;
            ((i64::from(x) + (1i64 << (r - 1))) >> r) as i32
        }
    }
    fn vqmovn(x: i32) -> i16 {
        x.clamp(-32768, 32767) as i16
    }
    fn vrshrn7(x: i32) -> i16 {
        ((i64::from(x) + 64) >> 7) as i16
    }

    /// The NEON kernel's `rescale_diff`, lane by lane.
    fn model_rescale_diff(d: i32, mult: i32, shift: i32) -> i32 {
        let (left, right) = (shift.clamp(0, 31), -shift.saturating_neg().clamp(0, 31));
        let p = vqrdmulh(d.wrapping_shl(left as u32), mult);
        let fixup = (p & right) >> 31;
        vrshl(p.saturating_add(fixup), right)
    }

    /// The NEON kernel's `lookup8`, lane by lane.
    fn model_lookup(v: i16, lut: &[i16; 513]) -> i16 {
        let i = usize::from((((v >> 7) + 256) as u16) & 0x1ff);
        let (base, next) = (lut[i], lut[i + 1]);
        let slope = next.wrapping_sub(base);
        base.wrapping_add(vrshrn7(i32::from(slope) * i32::from(v & 0x7f)))
    }

    /// The NEON row kernel's arithmetic, element by element.
    fn model_row(c: &RowConsts, src: &[i16], dst: &mut [i16]) {
        let max = *src.iter().max().unwrap();
        let mut sum = 0i32;
        for (d, &x) in dst.iter_mut().zip(src) {
            let s = model_rescale_diff(
                i32::from(x) - i32::from(max),
                c.input_multiplier,
                c.input_left_shift,
            );
            *d = model_lookup(vqmovn(s.wrapping_add(32767)), &c.exp_lut);
            sum = sum.wrapping_add(i32::from(*d));
        }
        let (recip, shift) = reciprocal(c, sum);
        for d in dst.iter_mut() {
            let p = i32::from(*d) * i32::from(recip);
            *d = vqmovn(vrshl(p, -(shift as i32))).max(0);
        }
    }

    fn rescales(rng: &mut impl rand::Rng) -> Vec<(i32, i32)> {
        let mut v: Vec<(i32, i32)> = [1e-9f64, 1e-6, 0.001, 0.05, 0.5, 1.0, 3.2, 100.0, 3e4]
            .iter()
            .map(|&r| quantize_multiplier(r))
            .collect();
        for shift in [-31, -30, -17, -2, -1, 0, 1, 15, 16, 24, 30, 31] {
            v.push((1 << 30, shift));
            v.push((i32::MAX, shift));
            v.push((rng.random_range(1 << 30..=i32::MAX), shift));
        }
        v.push((0, 0));
        v
    }

    #[test]
    fn neon_model_rescale_matches_tflite_exhaustively() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5016_0001);
        for (m, s) in rescales(&mut rng) {
            for d in -65535..=65535 {
                assert_eq!(
                    model_rescale_diff(d, m, s),
                    multiply_by_quantized_multiplier(d, m, s),
                    "diff {d} multiplier {m} shift {s}"
                );
            }
            for d in [i32::MIN, i32::MIN + 1, i32::MAX] {
                assert_eq!(
                    model_rescale_diff(d, m, s),
                    multiply_by_quantized_multiplier(d, m, s),
                    "diff {d} multiplier {m} shift {s}"
                );
            }
        }
        // INT32_MIN * INT32_MIN saturates in both.
        assert_eq!(model_rescale_diff(i32::MIN, i32::MIN, 0), i32::MAX);
    }

    #[test]
    fn neon_model_lookup_and_rescale_match_exhaustively() {
        let c = RowConsts::new(1 << 30, 0);
        let mut wild = [0i16; 513];
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5016_0002);
        for e in wild.iter_mut() {
            *e = rng.random();
        }
        for v in i16::MIN..=i16::MAX {
            for lut in [&c.exp_lut, &c.recip_lut, &wild] {
                assert_eq!(model_lookup(v, lut), lut_lookup_i16(v, lut), "value {v}");
            }
        }
        for shift in 1..=31u32 {
            for &recip in &[16384i16, 16387, 21845, 32767, -1, i16::MIN] {
                for e in (i16::MIN..=i16::MAX).step_by(7).chain([i16::MAX]) {
                    let p = i32::from(e) * i32::from(recip);
                    assert_eq!(
                        vqmovn(vrshl(p, -(shift as i32))).max(0),
                        rescale(e, recip, shift),
                        "exp {e} recip {recip} shift {shift}"
                    );
                }
            }
        }
    }

    #[test]
    fn neon_model_row_matches_portable() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x5016_0003);
        let mut lens: Vec<usize> = (1..=70).collect();
        lens.extend([1009, 4097]);
        for (m, s) in rescales(&mut rng) {
            let c = RowConsts::new(m, s);
            for &n in &lens {
                let mut rows: Vec<Vec<i16>> = [1i32, 64, 4096, 65536]
                    .iter()
                    .map(|&span| {
                        let lo = rng.random_range(-32768..=(32768 - span));
                        (0..n)
                            .map(|_| (lo + rng.random_range(0..span)) as i16)
                            .collect()
                    })
                    .collect();
                rows.extend([i16::MIN, 0, i16::MAX].map(|v| vec![v; n]));
                rows.push(
                    (0..n)
                        .map(|i| if i % 2 == 0 { i16::MIN } else { i16::MAX })
                        .collect(),
                );
                for row in rows {
                    let mut want = vec![0x5555i16; n];
                    let mut got = vec![0x5555i16; n];
                    softmax_row_i16_portable(&c, &row, &mut want);
                    model_row(&c, &row, &mut got);
                    assert_eq!(got, want, "len {n} multiplier {m} shift {s}");
                }
            }
        }
    }
}
