// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! int8 `ADD`, `SUB` and `MUL`, reproducing TFLite's int8 kernels bit for bit.

use std::borrow::Cow;

use super::rows::for_each_chunk2;
use crate::cpu::{CpuFeatures, Tier};
use crate::error::{KernelError, Result};
use crate::pool::Pool;
use crate::quant::{Activation, Quant, TfliteKernels};
use crate::tflite::{
    activation_range_i8, multiply_by_quantized_multiplier, quantize_multiplier,
    quantize_multiplier_smaller_than_one, rounding_divide_by_pot_neon,
    saturating_rounding_doubling_high_mul,
};
use crate::view::{TensorMut, TensorRef};

/// Width of the blocks TFLite's NEON `ADD` and `MUL` loops process.
pub(super) const BLOCK: usize = 16;

/// TFLite's int8 `ADD` / `SUB` input left shift.
const ADD_LEFT_SHIFT: i32 = 20;

/// The elementwise operation of a [`BinaryI8Plan`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum BinaryOp {
    Add = 0,
    Sub = 1,
    Mul = 2,
}

impl BinaryOp {
    fn name(self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
        }
    }
}

/// Parameters of an int8 `ADD`, `SUB` or `MUL`: `output = lhs op rhs`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BinaryI8Params {
    pub op: BinaryOp,
    pub lhs: Quant,
    pub rhs: Quant,
    pub output: Quant,
    pub activation: Activation,
    /// The TFLite kernels whose int8 arithmetic to reproduce.
    pub kernels: TfliteKernels,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    Portable,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// TFLite's `ArithmeticParams` for one int8 operator, with the per-element formulas.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Arith {
    pub(super) op: BinaryOp,
    pub(super) kernels: TfliteKernels,
    pub(super) lhs_offset: i32,
    pub(super) rhs_offset: i32,
    pub(super) out_offset: i32,
    /// `ADD` / `SUB`: the per-input multipliers, applied after the left shift.
    pub(super) lhs_mult: i32,
    pub(super) lhs_shift: i32,
    pub(super) rhs_mult: i32,
    pub(super) rhs_shift: i32,
    /// `MUL`: of the product; `ADD` / `SUB`: of the rescaled sum or difference.
    pub(super) out_mult: i32,
    pub(super) out_shift: i32,
    pub(super) act_min: i32,
    pub(super) act_max: i32,
}

/// `vshlq_s32` by a signed amount: left shift (wrapping) or truncating arithmetic right shift.
#[inline(always)]
pub(super) fn vshl(x: i32, shift: i32) -> i32 {
    if shift >= 0 {
        x.wrapping_shl(shift.min(31) as u32)
    } else {
        x >> (-shift).min(31)
    }
}

/// `vqmovn_s32`.
#[inline(always)]
fn sat_i16(x: i32) -> i16 {
    x.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

/// `vqmovn_s16`.
#[inline(always)]
fn sat_i8(x: i16) -> i8 {
    x.clamp(i16::from(i8::MIN), i16::from(i8::MAX)) as i8
}

impl Arith {
    pub(super) fn new(p: &BinaryI8Params) -> Result<Self> {
        for (q, what) in [(p.lhs, "lhs"), (p.rhs, "rhs"), (p.output, "output")] {
            if !(q.scale.is_finite() && q.scale > 0.0) {
                return Err(KernelError::InvalidArgument(match what {
                    "lhs" => "binary_i8 lhs scale must be finite and positive",
                    "rhs" => "binary_i8 rhs scale must be finite and positive",
                    _ => "binary_i8 output scale must be finite and positive",
                }));
            }
            if !(-128..=127).contains(&q.zero_point) {
                return Err(KernelError::InvalidArgument(
                    "binary_i8 zero points must lie in the int8 range",
                ));
            }
        }
        let (act_min, act_max) =
            activation_range_i8(p.activation, p.output.scale, p.output.zero_point);
        if act_min > act_max {
            return Err(KernelError::Unsupported(Cow::Borrowed(
                "binary_i8 activation range is empty for this output quantization",
            )));
        }
        let mut a = Arith {
            op: p.op,
            kernels: p.kernels,
            lhs_offset: -p.lhs.zero_point,
            rhs_offset: -p.rhs.zero_point,
            out_offset: p.output.zero_point,
            lhs_mult: 0,
            lhs_shift: 0,
            rhs_mult: 0,
            rhs_shift: 0,
            out_mult: 0,
            out_shift: 0,
            act_min,
            act_max,
        };
        match p.op {
            BinaryOp::Mul => {
                // mul.cc: `input1->params.scale * input2->params.scale / output->params.scale`,
                // all in f32, then widened.
                let real = f64::from(p.lhs.scale * p.rhs.scale / p.output.scale);
                if !(real.is_finite() && real > 0.0) {
                    return Err(KernelError::InvalidArgument(
                        "binary_i8 mul multiplier is not finite and positive",
                    ));
                }
                (a.out_mult, a.out_shift) = quantize_multiplier(real);
            }
            BinaryOp::Add | BinaryOp::Sub => {
                // add.cc / sub.cc: `2 * max(scale1, scale2)` in f32, the ratios in f64.
                let twice_max = f64::from(2.0 * p.lhs.scale.max(p.rhs.scale));
                let real_lhs = f64::from(p.lhs.scale) / twice_max;
                let real_rhs = f64::from(p.rhs.scale) / twice_max;
                let real_out =
                    twice_max / f64::from((1i32 << ADD_LEFT_SHIFT) as f32 * p.output.scale);
                let unsupported = || {
                    KernelError::Unsupported(Cow::Borrowed(
                        "binary_i8 scale ratios outside TFLite's int8 add/sub range",
                    ))
                };
                (a.lhs_mult, a.lhs_shift) =
                    quantize_multiplier_smaller_than_one(real_lhs).ok_or_else(unsupported)?;
                (a.rhs_mult, a.rhs_shift) =
                    quantize_multiplier_smaller_than_one(real_rhs).ok_or_else(unsupported)?;
                (a.out_mult, a.out_shift) = if real_out < 1.0 {
                    quantize_multiplier_smaller_than_one(real_out).ok_or_else(unsupported)?
                } else if p.op == BinaryOp::Sub && real_out.is_finite() {
                    // sub.cc's `QuantizeMultiplierGreaterThanOne`; add.cc has no such path.
                    quantize_multiplier(real_out)
                } else {
                    return Err(unsupported());
                };
            }
        }
        Ok(a)
    }

    #[inline(always)]
    fn clamp(&self, v: i32) -> i8 {
        v.clamp(self.act_min, self.act_max) as i8
    }

    /// TFLite's scalar code: `MulElementwise`, `AddFunc`, `SubElementwise`.
    #[inline(always)]
    pub(super) fn reference(&self, x: i8, y: i8) -> i8 {
        let xv = self.lhs_offset + i32::from(x);
        let yv = self.rhs_offset + i32::from(y);
        match self.op {
            BinaryOp::Mul => self.clamp(
                self.out_offset
                    + multiply_by_quantized_multiplier(xv * yv, self.out_mult, self.out_shift),
            ),
            BinaryOp::Add | BinaryOp::Sub => {
                let xs = multiply_by_quantized_multiplier(
                    xv * (1 << ADD_LEFT_SHIFT),
                    self.lhs_mult,
                    self.lhs_shift,
                );
                let ys = multiply_by_quantized_multiplier(
                    yv * (1 << ADD_LEFT_SHIFT),
                    self.rhs_mult,
                    self.rhs_shift,
                );
                let raw = if self.op == BinaryOp::Add {
                    xs + ys
                } else {
                    xs - ys
                };
                self.clamp(
                    multiply_by_quantized_multiplier(raw, self.out_mult, self.out_shift)
                        + self.out_offset,
                )
            }
        }
    }

    /// One lane of TFLite's NEON loop body (`MulElementwise` / `MulSimpleBroadcast`,
    /// `AddElementwiseInt8`). `SUB` has no NEON kernel and uses [`Arith::reference`].
    #[inline(always)]
    pub(super) fn neon_lane(&self, x: i8, y: i8) -> i8 {
        let xv = self.lhs_offset + i32::from(x);
        let yv = self.rhs_offset + i32::from(y);
        let q = match self.op {
            BinaryOp::Sub => return self.reference(x, y),
            BinaryOp::Mul => {
                let left = self.out_shift.max(0);
                let right = (-self.out_shift).max(0).unsigned_abs();
                let p = saturating_rounding_doubling_high_mul(vshl(xv * yv, left), self.out_mult);
                let p = sat_i16(rounding_divide_by_pot_neon(p, right));
                sat_i8(p.wrapping_add(self.out_offset as i16))
            }
            BinaryOp::Add => {
                let xs = saturating_rounding_doubling_high_mul(
                    vshl(xv, ADD_LEFT_SHIFT + self.lhs_shift),
                    self.lhs_mult,
                );
                let ys = saturating_rounding_doubling_high_mul(
                    vshl(yv, ADD_LEFT_SHIFT + self.rhs_shift),
                    self.rhs_mult,
                );
                let s = saturating_rounding_doubling_high_mul(xs.wrapping_add(ys), self.out_mult);
                let s = rounding_divide_by_pot_neon(s, (-self.out_shift).max(0).unsigned_abs());
                // `vmovn_s32` keeps the low 16 bits.
                sat_i8((s as i16).wrapping_add(self.out_offset as i16))
            }
        };
        q.clamp(self.act_min as i8, self.act_max as i8)
    }

    /// Flat indices from this one on take TFLite's scalar tail: every element unless the kernels
    /// run NEON blocks, then the last `n % 16`.
    pub(super) fn tail_start(&self, n: usize) -> usize {
        match self.op {
            BinaryOp::Add | BinaryOp::Mul if self.kernels.neon_blocks() => n - n % BLOCK,
            _ => 0,
        }
    }

    /// The portable kernel: elements `start..start + out.len()` of a tensor of `n`.
    pub(super) fn run_portable(
        &self,
        start: usize,
        tail: usize,
        x: &[i8],
        y: &[i8],
        out: &mut [i8],
    ) {
        for (k, ((o, &a), &b)) in out.iter_mut().zip(x).zip(y).enumerate() {
            *o = if start + k < tail {
                self.neon_lane(a, b)
            } else {
                self.reference(a, b)
            };
        }
    }
}

/// An int8 `ADD`, `SUB` or `MUL` of two same-shape tensors, bit-identical to TFLite's int8
/// kernels the plan names ([`TfliteKernels`]).
///
/// Formulas, from TFLite v2.19.0:
///
/// - `MUL`: `clamp(zo + MultiplyByQuantizedMultiplier((x − zx)·(y − zy), M, s))`, with the
///   multiplier from `sx·sy/so` computed in f32. TFLite's NEON blocks saturate the product's
///   rescale to int16 before adding `zo`, which only matters for multipliers near 0.5 or more.
/// - `ADD` and `SUB`: both inputs shifted left by 20 and rescaled by `s/(2·max(sx, sy))`, then
///   the sum or difference rescaled by `2·max(sx, sy)/(2^20·so)`. TFLite's NEON `ADD` blocks
///   apply the left shift and each input's right shift as one exact shift before a single
///   `vqrdmulh`, and keep the low 16 bits before adding `zo`; `SUB` always runs the scalar code.
/// - The fused activation clamps as `CalculateActivationRangeQuantized`.
///
/// Under `TfliteKernels::NeonRuy` and `NeonGemmlowp`, flat elements in whole 16-element blocks take TFLite's NEON
/// arithmetic and the last `n % 16` its scalar code, as TFLite's loops do for an elementwise
/// call (and for `MUL` by a broadcast scalar, the `n` being the other input's). TFLite's general
/// broadcast loops split the blocks differently, so pass broadcast inputs as zero-stride views
/// only when that matches the dispatch being reproduced.
///
/// ```
/// use edgefirst_kernels::{
///     Activation, BinaryI8Params, BinaryI8Plan, BinaryOp, Quant, TensorMut, TensorRef,
///     TfliteKernels, View, ViewMut,
/// };
///
/// let plan = BinaryI8Plan::new(BinaryI8Params {
///     op: BinaryOp::Add,
///     lhs: Quant::new(0.5, 0),
///     rhs: Quant::new(0.5, 0),
///     output: Quant::new(1.0, 0),
///     activation: Activation::None,
///     kernels: TfliteKernels::NeonRuy,
/// })?;
/// let (x, y) = ([2i8, 4, -6], [2i8, 2, 2]);
/// let mut out = [0i8; 3];
/// plan.run(
///     TensorRef::I8(View::contiguous(&x, &[3])?),
///     TensorRef::I8(View::contiguous(&y, &[3])?),
///     TensorMut::I8(ViewMut::contiguous(&mut out, &[3])?),
///     None,
/// )?;
/// assert_eq!(out, [2, 3, -2]);
/// # Ok::<(), edgefirst_kernels::KernelError>(())
/// ```
#[derive(Clone, Debug)]
pub struct BinaryI8Plan {
    arith: Arith,
    kernel: Kernel,
}

impl BinaryI8Plan {
    /// Builds a plan for the CPU this process runs on, honouring `EDGEFIRST_KERNELS_FORCE_TIER`.
    pub fn new(p: BinaryI8Params) -> Result<Self> {
        Self::with_features(p, CpuFeatures::from_env_or_probe()?)
    }

    /// Builds a plan that runs the best kernel `f` allows; reads no environment.
    pub fn with_features(p: BinaryI8Params, f: CpuFeatures) -> Result<Self> {
        Ok(Self {
            arith: Arith::new(&p)?,
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

    /// The output for every `lhs` value with `rhs` fixed at `rhs`, indexed by the raw byte as
    /// [`crate::LutI8Plan::from_table`] takes it, valid for every element of a call over `len`
    /// elements.
    ///
    /// `None` when that call would give some elements TFLite's NEON-block arithmetic and others
    /// its scalar code (an `ADD` or `MUL` with `len % 16 != 0` under
    /// `TfliteKernels::NeonRuy` or `NeonGemmlowp`) and the two disagree for this `rhs`.
    pub fn table_for_rhs(&self, rhs: i8, len: usize) -> Option<[u8; 256]> {
        let tail = self.arith.tail_start(len);
        let blocks: [u8; 256] =
            core::array::from_fn(|i| self.arith.neon_lane(i as u8 as i8, rhs) as u8);
        let scalar: [u8; 256] =
            core::array::from_fn(|i| self.arith.reference(i as u8 as i8, rhs) as u8);
        match (tail, len) {
            (0, _) => Some(scalar),
            (t, n) if t == n => Some(blocks),
            _ => (blocks == scalar).then_some(blocks),
        }
    }

    /// `output = lhs op rhs` elementwise; all three i8 with the same shape (any rank). A
    /// broadcast input is a view with zero strides.
    ///
    /// Work is split across `pool` when one is given. Output axes sorted by stride must each
    /// step past the extent of the smaller ones. Arguments are validated before anything is
    /// written.
    pub fn run(
        &self,
        lhs: TensorRef<'_>,
        rhs: TensorRef<'_>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
    ) -> Result<()> {
        let _span = tracing::trace_span!(
            "kernels.binary_i8",
            op = self.arith.op.name(),
            tier = self.tier().as_str()
        )
        .entered();
        let (TensorRef::I8(lhs), TensorRef::I8(rhs), TensorMut::I8(mut output)) =
            (lhs, rhs, output)
        else {
            return Err(KernelError::InvalidArgument(
                "binary_i8 needs i8 inputs and output",
            ));
        };
        let n = output.len();
        let tail = self.arith.tail_start(n);
        let arith = self.arith;
        match self.kernel {
            Kernel::Portable => for_each_chunk2(&lhs, &rhs, &mut output, pool, &|lo, x, y, o| {
                arith.run_portable(lo, tail, x, y, o);
            }),
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => for_each_chunk2(&lhs, &rhs, &mut output, pool, &|lo, x, y, o| {
                // SAFETY: NEON is part of the aarch64 baseline.
                unsafe { super::binary_i8_neon::run(&arith, lo, tail, x, y, o) }
            }),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::{View, ViewMut};

    fn params(op: BinaryOp, kernels: TfliteKernels) -> BinaryI8Params {
        BinaryI8Params {
            op,
            lhs: Quant::new(0.0579, -128),
            rhs: Quant::new(0.000_196, -128),
            output: Quant::new(0.002_894, -128),
            activation: Activation::None,
            kernels,
        }
    }

    /// A table, when there is one, gives what the plan gives for every element of a call of
    /// that length; `None` only when the NEON-block and scalar results disagree.
    #[test]
    fn rhs_tables_follow_the_call_length() {
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
            for kernels in [TfliteKernels::NeonRuy, TfliteKernels::Reference] {
                let plan = BinaryI8Plan::with_features(params(op, kernels), CpuFeatures::default())
                    .unwrap();
                for rhs in [-128i8, -3, 0, 77, 127] {
                    for len in [0usize, 1, 15, 16, 17, 32, 300] {
                        let lhs: Vec<i8> = (0..len).map(|i| (i * 37 % 256) as u8 as i8).collect();
                        let rv = vec![rhs; len];
                        let mut out = vec![0i8; len];
                        plan.run(
                            TensorRef::I8(View::contiguous(&lhs, &[len]).unwrap()),
                            TensorRef::I8(View::contiguous(&rv, &[len]).unwrap()),
                            TensorMut::I8(ViewMut::contiguous(&mut out, &[len]).unwrap()),
                            None,
                        )
                        .unwrap();
                        match plan.table_for_rhs(rhs, len) {
                            Some(t) => {
                                for (&x, &o) in lhs.iter().zip(&out) {
                                    assert_eq!(
                                        t[usize::from(x as u8)] as i8,
                                        o,
                                        "{op:?} {rhs} {len}"
                                    );
                                }
                            }
                            None => assert_ne!(
                                plan.table_for_rhs(rhs, 16),
                                plan.table_for_rhs(rhs, 0),
                                "{op:?} {rhs} {len}"
                            ),
                        }
                    }
                }
            }
        }
    }

    /// Row gaps, a broadcast axis, rows that do not divide the chunk size and a NEON-build
    /// tail: every tier gives what it gives on dense copies.
    #[test]
    fn strided_views_match_contiguous() {
        let shape = [4usize, 7, 41];
        let n: usize = shape.iter().product();
        let x: Vec<i8> = (0..4 * 7 * 48).map(|i| (i * 7 % 251) as i8).collect();
        let y: Vec<i8> = (0..4 * 41).map(|i| (i * 13 % 247) as i8).collect();
        let lhs = View::strided(&x, &shape, &[7 * 48, 48, 1]).unwrap();
        let rhs = View::strided(&y, &shape, &[41, 0, 1]).unwrap();
        let dense = |v: &View<'_, i8>| -> Vec<i8> {
            let mut d = Vec::with_capacity(n);
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    for k in 0..shape[2] {
                        let s = v.strides();
                        let o = i * s[0] as usize + j * s[1] as usize + k * s[2] as usize;
                        d.push(v.data()[o]);
                    }
                }
            }
            d
        };
        let (xd, yd) = (dense(&lhs), dense(&rhs));
        let tiers = [
            Some(CpuFeatures::default()),
            CpuFeatures::from_forced_tier(Some("neon")).ok(),
        ];
        for op in [BinaryOp::Add, BinaryOp::Sub, BinaryOp::Mul] {
            for f in tiers.iter().flatten() {
                let plan =
                    BinaryI8Plan::with_features(params(op, TfliteKernels::NeonRuy), *f).unwrap();
                let mut want = vec![0i8; n];
                plan.run(
                    TensorRef::I8(View::contiguous(&xd, &shape).unwrap()),
                    TensorRef::I8(View::contiguous(&yd, &shape).unwrap()),
                    TensorMut::I8(ViewMut::contiguous(&mut want, &shape).unwrap()),
                    None,
                )
                .unwrap();
                let mut out = vec![0x55i8; 4 * 7 * 50];
                plan.run(
                    TensorRef::I8(lhs),
                    TensorRef::I8(rhs),
                    TensorMut::I8(ViewMut::strided(&mut out, &shape, &[7 * 50, 50, 1]).unwrap()),
                    None,
                )
                .unwrap();
                let got: Vec<i8> = dense(&View::strided(&out, &shape, &[7 * 50, 50, 1]).unwrap());
                assert_eq!(got, want, "{op:?} {:?}", plan.tier());
                for (i, c) in out.chunks(50).enumerate() {
                    assert!(c[41..].iter().all(|&b| b == 0x55), "row {i} gap written");
                }
            }
        }
    }

    #[test]
    fn mul_neon_lane_matches_reference_for_small_multipliers() {
        let a = Arith::new(&params(BinaryOp::Mul, TfliteKernels::NeonRuy)).unwrap();
        for x in i8::MIN..=i8::MAX {
            for y in i8::MIN..=i8::MAX {
                assert_eq!(a.neon_lane(x, y), a.reference(x, y), "{x} {y}");
            }
        }
    }

    #[test]
    fn add_neon_lane_rounds_once() {
        let mut p = params(BinaryOp::Add, TfliteKernels::NeonRuy);
        p.lhs = Quant::new(0.002_894, -128);
        p.rhs = Quant::new(0.003_897, -128);
        p.output = Quant::new(0.004_491, -128);
        let a = Arith::new(&p).unwrap();
        let differ = (i8::MIN..=i8::MAX)
            .flat_map(|x| (i8::MIN..=i8::MAX).map(move |y| (x, y)))
            .filter(|&(x, y)| a.neon_lane(x, y) != a.reference(x, y))
            .count();
        // The two formulas agree almost everywhere, and differ by at most one where they do.
        assert!(differ < 65536 / 4, "{differ}");
        for x in i8::MIN..=i8::MAX {
            for y in i8::MIN..=i8::MAX {
                assert!((i32::from(a.neon_lane(x, y)) - i32::from(a.reference(x, y))).abs() <= 1);
            }
        }
    }

    #[test]
    fn tail_follows_build_and_op() {
        let neon = Arith::new(&params(BinaryOp::Add, TfliteKernels::NeonRuy)).unwrap();
        assert_eq!(neon.tail_start(37), 32);
        let sub = Arith::new(&params(BinaryOp::Sub, TfliteKernels::NeonRuy)).unwrap();
        assert_eq!(sub.tail_start(37), 0);
        let portable = Arith::new(&params(BinaryOp::Mul, TfliteKernels::Reference)).unwrap();
        assert_eq!(portable.tail_start(37), 0);
    }

    #[test]
    fn add_rejects_output_multiplier_above_one() {
        let mut p = params(BinaryOp::Add, TfliteKernels::NeonRuy);
        p.lhs = Quant::new(1.0, 0);
        p.rhs = Quant::new(1.0, 0);
        p.output = Quant::new(1.0e-9, 0);
        assert!(matches!(Arith::new(&p), Err(KernelError::Unsupported(_))));
        p.op = BinaryOp::Sub;
        assert!(Arith::new(&p).is_ok());
    }
}
