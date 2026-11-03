// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! DFL: softmax over each side's bins, then the bin-weighted sum, per anchor.

#[cfg(target_arch = "aarch64")]
use half::f16;

#[cfg(target_arch = "aarch64")]
use super::levels::run_blocks_f16;
use super::levels::{
    block_rows, check_reg_max, decode_shapes, input_quant, output_quant, run_blocks, DecodeInput,
    Isa, OutQuant, MAX_REG_MAX,
};
use crate::cpu::{CpuFeatures, Tier};
use crate::error::{KernelError, Result};
use crate::pool::Pool;
use crate::prim::box_primitives::weighted_sum_4sides_f32;
#[cfg(not(target_arch = "aarch64"))]
use crate::prim::dfl_lut::DflBin;
use crate::prim::dfl_lut::DflExpTable;
use crate::prim::softmax::softmax_inplace_f32;
use crate::quant::{DType, Quant};
use crate::view::{TensorMut, TensorRef, View};

/// Parameters of a DFL decode: `[anchors, 4 · reg_max]` logits to `[anchors, 4]` distances
/// `[l, t, r, b]` in grid units.
///
/// `input` applies to integer input types and is ignored for `f16` and `f32` inputs; `output`
/// applies to integer output types and is ignored for float outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DflParams {
    pub input_dtype: DType,
    pub input: Quant,
    pub output_dtype: DType,
    pub output: Quant,
    pub reg_max: usize,
}

/// The f32 softmax and weighted-sum kernel of the non-table paths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Softmax {
    /// `prim::softmax` per side, then `prim::box_primitives::weighted_sum_4sides_f32`.
    Scalar,
    /// `prim::neon_baseline::softmax_weighted_sum_4sides_f32_neon` after the NEON dequantize.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// The same with the fp16 exp, `softmax_weighted_sum_4sides_f32_neon_fp16`.
    #[cfg(target_arch = "aarch64")]
    NeonFp16,
}

impl Softmax {
    /// `[l, t, r, b]` of one anchor's dequantized logits; `deq` holds exactly `4 · reg_max`
    /// values and is clobbered.
    #[inline(always)]
    fn ltrb(self, deq: &mut [f32], reg_max: usize) -> [f32; 4] {
        match self {
            Self::Scalar => {
                for side in deq.chunks_exact_mut(reg_max) {
                    softmax_inplace_f32(side);
                }
                weighted_sum_4sides_f32(deq, reg_max)
            }
            // SAFETY: NEON is part of the aarch64 baseline and `deq` holds `4 · reg_max` values.
            #[cfg(target_arch = "aarch64")]
            Self::Neon => unsafe {
                crate::prim::neon_baseline::softmax_weighted_sum_4sides_f32_neon(deq, reg_max)
            },
            // SAFETY: selected only when the CPU has fp16; `deq` holds `4 · reg_max` values.
            #[cfg(target_arch = "aarch64")]
            Self::NeonFp16 => unsafe {
                crate::prim::neon_baseline::softmax_weighted_sum_4sides_f32_neon_fp16(deq, reg_max)
            },
        }
    }

    fn neon_dequant(self) -> bool {
        self != Self::Scalar
    }
}

/// How a DFL plan turns one anchor's bins into four distances.
#[derive(Clone, Debug)]
pub(crate) enum DflKernel {
    /// 8-bit bins through the exp table of `prim::dfl_lut`; `neon` runs its four-lane kernels.
    Lut { table: Box<DflExpTable>, neon: bool },
    /// Dequantize to f32, softmax and weighted sum.
    Softmax(Softmax),
    /// [`DflBoxPlan`](crate::DflBoxPlan)'s `f16` outputs of 8-bit bins on the FP16 tier: the
    /// NEON table kernels' f32 distances, then dist2bbox in f16.
    #[cfg(target_arch = "aarch64")]
    LutFp16 { table: Box<DflExpTable> },
    /// `f16` outputs of 16-bit and float inputs on the FP16 tier: the NEON dequantize to f32, the
    /// exp and eight-bin sums in f16, then the lane reduction and divide in f32
    /// (`prim::neon_fp16::dfl_softmax_ltrb_f16_neon_fp16`).
    #[cfg(target_arch = "aarch64")]
    SoftmaxFp16,
}

/// An 8-bit bin type of the table kernels.
#[cfg(target_arch = "aarch64")]
pub(crate) trait LutBin: crate::prim::dfl_lut::NeonDflBin + Default + Sync {}
/// An 8-bit bin type of the table kernels.
#[cfg(not(target_arch = "aarch64"))]
pub(crate) trait LutBin: DflBin + Default + Sync {}
impl LutBin for i8 {}
impl LutBin for u8 {}

impl DflKernel {
    /// The kernel for this input and output: the exp table for 8-bit inputs (NEON for every
    /// `reg_max`, with dist2bbox in f16 for `boxes` into `f16` on the FP16 tier); the FP16
    /// softmax for other inputs into `f16` on the FP16 tier; otherwise the f32 softmax, NEON on
    /// NEON tiers, with the fp16 exp for 16-bit inputs to `f32` on the FP16 tier, as the decoder
    /// does.
    pub(crate) fn select(
        input_dtype: DType,
        input: Quant,
        output_dtype: DType,
        boxes: bool,
        f: CpuFeatures,
    ) -> Self {
        // Only the FP16 tier's box kernels depend on it.
        #[cfg(not(target_arch = "aarch64"))]
        let _ = boxes;
        match (input_dtype, Isa::of(f), output_dtype) {
            #[cfg(target_arch = "aarch64")]
            (DType::I8 | DType::U8, Isa::NeonFp16, DType::F16) if boxes => Self::LutFp16 {
                table: Box::new(DflExpTable::new(input.scale)),
            },
            (DType::I8 | DType::U8, isa, _) => Self::Lut {
                table: Box::new(DflExpTable::new(input.scale)),
                neon: isa != Isa::Scalar,
            },
            #[cfg(target_arch = "aarch64")]
            (_, Isa::NeonFp16, DType::F16) => Self::SoftmaxFp16,
            #[cfg(target_arch = "aarch64")]
            (DType::I16 | DType::U16, Isa::NeonFp16, DType::F32) => {
                Self::Softmax(Softmax::NeonFp16)
            }
            #[cfg(target_arch = "aarch64")]
            (_, Isa::Neon | Isa::NeonFp16, _) => Self::Softmax(Softmax::Neon),
            (_, Isa::Scalar, _) => Self::Softmax(Softmax::Scalar),
        }
    }

    pub(crate) fn tier(&self) -> Tier {
        match self {
            Self::Lut { neon: true, .. } => Tier::Neon,
            Self::Lut { neon: false, .. } | Self::Softmax(Softmax::Scalar) => Tier::Scalar,
            #[cfg(target_arch = "aarch64")]
            Self::Softmax(Softmax::Neon) => Tier::Neon,
            #[cfg(target_arch = "aarch64")]
            Self::Softmax(Softmax::NeonFp16) | Self::LutFp16 { .. } | Self::SoftmaxFp16 => {
                Tier::NeonFp16
            }
        }
    }
}

/// Calls `f(k, [l, t, r, b])` for each anchor `k` of `src`, `4 · reg_max` bins each, with an
/// f32 softmax kernel; other kernels call nothing.
#[inline(always)]
pub(crate) fn for_each_ltrb<I: DecodeInput>(
    kernel: &DflKernel,
    q: Quant,
    reg_max: usize,
    src: &[I],
    mut f: impl FnMut(usize, [f32; 4]),
) {
    let DflKernel::Softmax(s) = kernel else {
        return;
    };
    let n = 4 * reg_max;
    let mut store = [0.0f32; 4 * MAX_REG_MAX];
    let Some(deq) = store.get_mut(..n) else {
        return;
    };
    for (k, bins) in src.chunks_exact(n).enumerate() {
        I::dequant_f32(bins, q, deq, s.neon_dequant());
        f(k, s.ltrb(deq, reg_max));
    }
}

/// `[l, t, r, b]` in f16 of each anchor of `src` into `dst` (four per anchor) with
/// [`DflKernel::SoftmaxFp16`]: the NEON dequantize of each anchor, then the FP16 kernel.
///
/// # Safety
/// The CPU must support FEAT_FP16.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn softmax_ltrb_f16<I: DecodeInput>(
    q: Quant,
    reg_max: usize,
    src: &[I],
    dst: &mut [f16],
) {
    let n = 4 * reg_max;
    let mut store = [0.0f32; 4 * MAX_REG_MAX];
    let Some(deq) = store.get_mut(..n) else {
        return;
    };
    for (bins, out) in src.chunks_exact(n).zip(dst.chunks_exact_mut(4)) {
        I::dequant_f32(bins, q, deq, true);
        // SAFETY: the caller guarantees FEAT_FP16.
        unsafe { crate::prim::neon_fp16::dfl_softmax_ltrb_f16_neon_fp16(deq, reg_max, out) };
    }
}

/// `[l, t, r, b]` in f32 of each anchor's 8-bit bins through the exp table: the 16-bin NEON
/// kernel, the any-`reg_max` NEON kernel, or the portable one, all with the same bits.
#[inline(always)]
pub(crate) fn lut_ltrb<T: LutBin>(
    src: &[T],
    table: &DflExpTable,
    reg_max: usize,
    neon: bool,
    dst: &mut [f32],
) {
    #[cfg(target_arch = "aarch64")]
    if neon {
        if reg_max == 16 {
            crate::prim::dfl_lut::dfl_lut_ltrb_neon(src, table, reg_max, dst);
        } else {
            crate::prim::dfl_lut::dfl_lut_ltrb_generic_neon(src, table, reg_max, dst);
        }
        return;
    }
    let _ = neon;
    crate::prim::dfl_lut::dfl_lut_ltrb(src, table, reg_max, dst);
}

/// A validated DFL decode over the anchors of a `[anchors, 4 · reg_max]` view.
///
/// Each side's `reg_max` bins go through a softmax, and the side's distance is the
/// bin-weighted sum `Σ p_i · i` in grid units, as the decoder computes it. The kernel is chosen
/// at creation:
///
/// 1. `f16` outputs of 16-bit and float inputs on the FP16 tier: `prim::neon_fp16`, the f16 exp
///    of the f32 logit differences and eight-bin sums in f16, then the lane reduction and divide
///    in f32, rounded to f16.
/// 2. `i8` and `u8` inputs, whatever the output: the decoder's exp-table kernel
///    (`prim::dfl_lut`), NEON on NEON tiers for every `reg_max` (the decoder's 16-bin kernel for
///    16, an any-`reg_max` kernel with the same bits otherwise). On the FP16 tier too, an `f16`
///    output is that f32 result rounded once, and the plan reports `Tier::Neon`.
/// 3. Other inputs on NEON: the NEON dequantize, then the decoder's fused NEON softmax and
///    weighted sum, with the fp16 exp for `i16` and `u16` inputs to `f32` outputs on the FP16
///    tier, as the decoder does.
/// 4. Otherwise the scalar f32 kernels.
///
/// `f32` outputs are stored directly. Except for path 1, `f16` outputs are the tier's `f32`
/// result rounded once (nearest even). Integer outputs are quantized once from the f32
/// result (round half away from zero, clamp). The input may be anchor-major or channel-major
/// (TFLite's `[4 · reg_max, anchors]` seen as `[anchors, 4 · reg_max]` with strides
/// `[1, anchors]`); both give the same bits.
///
/// ```
/// use edgefirst_kernels::{DType, DflParams, DflPlan, Quant, TensorMut, TensorRef, View, ViewMut};
///
/// let plan = DflPlan::new(DflParams {
///     input_dtype: DType::I8,
///     input: Quant::new(0.5, 0),
///     output_dtype: DType::F32,
///     output: Quant::IDENTITY,
///     reg_max: 4,
/// })?;
/// let mut bins = [-128i8; 16];
/// for (side, bin) in [0usize, 1, 2, 3].into_iter().enumerate() {
///     bins[side * 4 + bin] = 127;
/// }
/// let mut ltrb = [0.0f32; 4];
/// plan.run(
///     TensorRef::I8(View::contiguous(&bins, &[1, 16])?),
///     TensorMut::F32(ViewMut::contiguous(&mut ltrb, &[1, 4])?),
///     None,
/// )?;
/// assert_eq!(ltrb.map(f32::round), [0.0, 1.0, 2.0, 3.0]);
/// # Ok::<(), edgefirst_kernels::KernelError>(())
/// ```
#[derive(Clone, Debug)]
pub struct DflPlan {
    input_dtype: DType,
    output_dtype: DType,
    input: Quant,
    output: OutQuant,
    reg_max: usize,
    kernel: DflKernel,
}

impl DflPlan {
    /// Builds a plan for the CPU this process runs on, honouring `EDGEFIRST_KERNELS_FORCE_TIER`.
    pub fn new(p: DflParams) -> Result<Self> {
        Self::with_features(p, CpuFeatures::from_env_or_probe()?)
    }

    /// Builds a plan that runs the best kernel `f` allows.
    ///
    /// `reg_max` must be positive (`InvalidArgument`) and at most 64 (`Unsupported`). Integer
    /// input and output scales must be finite and positive, and their zero points inside their
    /// type's range (`InvalidArgument`). Every input and output type pair is supported.
    pub fn with_features(p: DflParams, f: CpuFeatures) -> Result<Self> {
        check_reg_max(p.reg_max)?;
        let input = input_quant(p.input_dtype, p.input)?;
        let output = output_quant(p.output_dtype, p.output)?;
        Ok(Self {
            input_dtype: p.input_dtype,
            output_dtype: p.output_dtype,
            input,
            output,
            reg_max: p.reg_max,
            kernel: DflKernel::select(p.input_dtype, input, p.output_dtype, false, f),
        })
    }

    /// The tier of the kernel this plan runs.
    pub fn tier(&self) -> Tier {
        self.kernel.tier()
    }

    /// Distances of every anchor of `input` (`[anchors, 4 · reg_max]`) into `output`
    /// (`[anchors, 4]`), with the plan's dtypes.
    ///
    /// Anchors are split across `pool` when one is given; results do not depend on the pool
    /// or the strides. Output axes sorted by stride must each step past the extent of the
    /// smaller ones. Arguments are validated before anything is written.
    pub fn run(
        &self,
        input: TensorRef<'_>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
    ) -> Result<()> {
        let _span = tracing::trace_span!("kernels.dfl", tier = self.tier().as_str()).entered();
        if input.dtype() != self.input_dtype || output.dtype() != self.output_dtype {
            return Err(KernelError::InvalidArgument(
                "dfl input and output dtypes differ from the plan's",
            ));
        }
        let rows = decode_shapes(input.shape(), output.shape(), 4 * self.reg_max)?;
        let segments = [rows];
        match input {
            TensorRef::I8(v) => self.run_lut(&v, output, pool, &segments),
            TensorRef::U8(v) => self.run_lut(&v, output, pool, &segments),
            TensorRef::I16(v) => self.run_dequant(&v, output, pool, &segments),
            TensorRef::U16(v) => self.run_dequant(&v, output, pool, &segments),
            TensorRef::F16(v) => self.run_dequant(&v, output, pool, &segments),
            TensorRef::F32(v) => self.run_dequant(&v, output, pool, &segments),
        }
    }

    fn run_lut<T: LutBin>(
        &self,
        input: &View<'_, T>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
        segments: &[usize],
    ) -> Result<()> {
        let rm = self.reg_max;
        match &self.kernel {
            DflKernel::Lut { table, neon } => run_blocks(
                input,
                output,
                pool,
                segments,
                block_rows(4 * rm),
                false,
                self.output,
                *neon,
                &|_, _, src, dst| lut_ltrb(src, table, rm, *neon, dst),
            ),
            _ => Err(KernelError::InvalidArgument(
                "dfl plan has no table for this input",
            )),
        }
    }

    fn run_dequant<I: DecodeInput>(
        &self,
        input: &View<'_, I>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
        segments: &[usize],
    ) -> Result<()> {
        let rm = self.reg_max;
        #[cfg(target_arch = "aarch64")]
        if matches!(self.kernel, DflKernel::SoftmaxFp16) {
            return run_blocks_f16(
                input,
                output,
                pool,
                segments,
                block_rows(4 * rm),
                // SAFETY: `SoftmaxFp16` is selected only when FEAT_FP16 was detected.
                &|_, _, src, dst| unsafe { softmax_ltrb_f16(self.input, rm, src, dst) },
            );
        }
        run_blocks(
            input,
            output,
            pool,
            segments,
            block_rows(4 * rm),
            false,
            self.output,
            self.tier() != Tier::Scalar,
            &|_, _, src, dst| {
                for_each_ltrb(&self.kernel, self.input, rm, src, |k, ltrb| {
                    if let Some(out) = dst.get_mut(4 * k..4 * k + 4) {
                        out.copy_from_slice(&ltrb);
                    }
                });
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [DType; 6] = [
        DType::I8,
        DType::U8,
        DType::I16,
        DType::U16,
        DType::F16,
        DType::F32,
    ];

    fn kernel(input_dtype: DType, output_dtype: DType, f: CpuFeatures) -> DflKernel {
        DflPlan::with_features(
            DflParams {
                input_dtype,
                input: Quant::new(0.1, 0),
                output_dtype,
                output: Quant::new(0.1, 0),
                reg_max: 16,
            },
            f,
        )
        .unwrap()
        .kernel
    }

    #[test]
    fn path_selection() {
        let s = CpuFeatures::default();
        for out in ALL {
            for d in [DType::I8, DType::U8] {
                assert!(matches!(
                    kernel(d, out, s),
                    DflKernel::Lut { neon: false, .. }
                ));
            }
            for d in [DType::I16, DType::U16, DType::F16, DType::F32] {
                assert!(matches!(
                    kernel(d, out, s),
                    DflKernel::Softmax(Softmax::Scalar)
                ));
                assert_eq!(kernel(d, out, s).tier(), Tier::Scalar);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_path_selection() {
        let n = CpuFeatures::from_forced_tier(Some("neon")).unwrap();
        for out in ALL {
            for d in [DType::I8, DType::U8] {
                assert!(matches!(
                    kernel(d, out, n),
                    DflKernel::Lut { neon: true, .. }
                ));
            }
            for d in [DType::I16, DType::U16, DType::F16, DType::F32] {
                assert!(matches!(
                    kernel(d, out, n),
                    DflKernel::Softmax(Softmax::Neon)
                ));
            }
        }
        for reg_max in [1, 7, 8, 17, 64] {
            let p = DflParams {
                input_dtype: DType::U8,
                input: Quant::new(0.1, 0),
                output_dtype: DType::F32,
                output: Quant::IDENTITY,
                reg_max,
            };
            assert_eq!(DflPlan::with_features(p, n).unwrap().tier(), Tier::Neon);
        }
        let Ok(h) = CpuFeatures::from_forced_tier(Some("neon_fp16")) else {
            crate::test_support::report_skip("no neon_fp16 tier on this CPU");
            return;
        };
        for d in [DType::I8, DType::U8] {
            // Without box math there is no f16 arithmetic: the f32 distances, rounded once.
            assert!(matches!(
                kernel(d, DType::F16, h),
                DflKernel::Lut { neon: true, .. }
            ));
            assert!(matches!(
                DflKernel::select(d, Quant::new(0.1, 0), DType::F16, true, h),
                DflKernel::LutFp16 { .. }
            ));
            assert!(matches!(
                kernel(d, DType::F32, h),
                DflKernel::Lut { neon: true, .. }
            ));
        }
        for d in [DType::I16, DType::U16, DType::F16, DType::F32] {
            assert!(matches!(kernel(d, DType::F16, h), DflKernel::SoftmaxFp16));
            assert!(matches!(
                kernel(d, DType::I16, h),
                DflKernel::Softmax(Softmax::Neon)
            ));
        }
        assert_eq!(kernel(DType::U16, DType::F32, h).tier(), Tier::NeonFp16);
        assert_eq!(kernel(DType::F16, DType::F32, h).tier(), Tier::Neon);
        assert_eq!(kernel(DType::F32, DType::F32, h).tier(), Tier::Neon);
        assert_eq!(kernel(DType::U16, DType::F16, h).tier(), Tier::NeonFp16);
    }
}
