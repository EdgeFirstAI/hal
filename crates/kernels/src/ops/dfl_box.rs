// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Fused DFL and dist2bbox: box logits to boxes in one pass, the YOLOv8/11 tail.

use super::dfl::{for_each_ltrb, DflKernel, LutBin};
use super::levels::{
    block_rows, check_reg_max, decode_shapes, input_quant, output_quant, run_blocks, AnchorGrid,
    BoxFormat, DecodeInput, Level, OutQuant,
};
#[cfg(target_arch = "aarch64")]
use super::levels::{run_blocks_f16, Span, BLOCK};
use crate::cpu::{CpuFeatures, Tier};
use crate::error::{KernelError, Result};
use crate::pool::Pool;
use crate::prim::box_primitives::dist2bbox_anchor_f32;
use crate::prim::dfl_lut::{decode_dfl_lut, DflGeom};
#[cfg(target_arch = "aarch64")]
use crate::prim::neon_fp16::{dist2bbox_f16, Ltrb8};
use crate::quant::{DType, Quant};
use crate::view::{TensorMut, TensorRef, View};

/// Parameters of a fused DFL box decode: `[anchors, 4 · reg_max]` logits to `[anchors, 4]`
/// boxes in pixels.
///
/// `levels` describe the anchor axis in order; their anchors must add up to it. `input`
/// applies to integer input types and is ignored for `f16` and `f32` inputs; `output` applies
/// to integer output types and is ignored for float outputs.
#[derive(Clone, Debug, PartialEq)]
pub struct DflBoxParams {
    pub input_dtype: DType,
    pub input: Quant,
    pub output_dtype: DType,
    pub output: Quant,
    pub reg_max: usize,
    pub levels: Vec<Level>,
    pub format: BoxFormat,
}

/// A validated DFL box decode over a concatenated multi-level anchor axis: [`DflPlan`]'s
/// distances, then [`Dist2BboxPlan`]'s boxes, without the intermediate tensor.
///
/// The kernels are chosen as [`DflPlan`] chooses them. On the FP16 tier, `f16` outputs run
/// dist2bbox in f16 on the f16-rounded grid and stride (`prim::neon_fp16`, which computes
/// [`BoxFormat::Xyxy`] directly as `(g ∓ d)·s`), on [`DflPlan`]'s FP16 distances for 16-bit
/// and float inputs and on the NEON table kernels' f32 distances for `i8` and `u8`. Otherwise
/// `i8` and `u8` inputs run the exp-table decode of `prim::dfl_lut` (NEON for every `reg_max`
/// on NEON tiers), other inputs the NEON dequantize and fused NEON
/// softmax on NEON (the fp16 exp for `i16` and `u16` inputs to `f32` outputs on the FP16 tier)
/// and the scalar f32 kernels without NEON, and dist2bbox runs in f32. For integer inputs a
/// one-level call gives the decoder's `f32` level result bit for bit. On every tier but FP16,
/// `f16` outputs are the tier's `f32` boxes rounded once; integer outputs are quantized once
/// from the f32 boxes, and [`BoxFormat::Xyxy`] is applied to the f32 `[xc, yc, w, h]`.
///
/// [`DflPlan`]: crate::DflPlan
/// [`Dist2BboxPlan`]: crate::Dist2BboxPlan
///
/// ```
/// use edgefirst_kernels::{
///     BoxFormat, DType, DflBoxParams, DflBoxPlan, Level, Quant, TensorMut, TensorRef, View,
///     ViewMut,
/// };
///
/// let plan = DflBoxPlan::new(DflBoxParams {
///     input_dtype: DType::I8,
///     input: Quant::new(0.5, 0),
///     output_dtype: DType::F32,
///     output: Quant::IDENTITY,
///     reg_max: 4,
///     levels: vec![Level { anchors: 1, grid_w: 1, stride: 8.0 }],
///     format: BoxFormat::Xywh,
/// })?;
/// // Distances [1, 1, 3, 3]: one-hot bins.
/// let mut bins = [-128i8; 16];
/// for (side, bin) in [1usize, 1, 3, 3].into_iter().enumerate() {
///     bins[side * 4 + bin] = 127;
/// }
/// let mut xywh = [0.0f32; 4];
/// plan.run(
///     TensorRef::I8(View::contiguous(&bins, &[1, 16])?),
///     TensorMut::F32(ViewMut::contiguous(&mut xywh, &[1, 4])?),
///     None,
/// )?;
/// assert_eq!(xywh, [12.0, 12.0, 32.0, 32.0]);
/// # Ok::<(), edgefirst_kernels::KernelError>(())
/// ```
#[derive(Clone, Debug)]
pub struct DflBoxPlan {
    input_dtype: DType,
    output_dtype: DType,
    input: Quant,
    output: OutQuant,
    reg_max: usize,
    format: BoxFormat,
    grid: AnchorGrid,
    kernel: DflKernel,
}

impl DflBoxPlan {
    /// Builds a plan for the CPU this process runs on, honouring `EDGEFIRST_KERNELS_FORCE_TIER`.
    pub fn new(p: DflBoxParams) -> Result<Self> {
        Self::with_features(p, CpuFeatures::from_env_or_probe()?)
    }

    /// Builds a plan that runs the best kernel `f` allows, and the anchor grid of `p.levels`.
    ///
    /// `reg_max` must be positive (`InvalidArgument`) and at most 64 (`Unsupported`). Levels
    /// need a positive `grid_w` that divides their anchors and a finite, positive stride; there
    /// must be at least one. Integer input and output scales must be finite and positive, and
    /// their zero points inside their type's range. Those violations are `InvalidArgument`.
    /// Every input and output type pair is supported.
    pub fn with_features(p: DflBoxParams, f: CpuFeatures) -> Result<Self> {
        check_reg_max(p.reg_max)?;
        let input = input_quant(p.input_dtype, p.input)?;
        let output = output_quant(p.output_dtype, p.output)?;
        let grid = AnchorGrid::new(&p.levels)?;
        Ok(Self {
            input_dtype: p.input_dtype,
            output_dtype: p.output_dtype,
            input,
            output,
            reg_max: p.reg_max,
            format: p.format,
            grid,
            kernel: DflKernel::select(p.input_dtype, input, p.output_dtype, true, f),
        })
    }

    /// The tier of the kernel this plan runs.
    pub fn tier(&self) -> Tier {
        self.kernel.tier()
    }

    /// Boxes of every anchor of `input` (`[anchors, 4 · reg_max]`) into `output`
    /// (`[anchors, 4]`), with the plan's dtypes; the anchor count must equal the levels' total.
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
        let _span = tracing::trace_span!("kernels.dfl_box", tier = self.tier().as_str()).entered();
        if input.dtype() != self.input_dtype || output.dtype() != self.output_dtype {
            return Err(KernelError::InvalidArgument(
                "dfl_box input and output dtypes differ from the plan's",
            ));
        }
        if decode_shapes(input.shape(), output.shape(), 4 * self.reg_max)? != self.grid.total() {
            return Err(KernelError::InvalidArgument(
                "dfl_box anchor count differs from the levels' total",
            ));
        }
        match input {
            TensorRef::I8(v) => self.run_lut(&v, output, pool),
            TensorRef::U8(v) => self.run_lut(&v, output, pool),
            TensorRef::I16(v) => self.run_dequant(&v, output, pool),
            TensorRef::U16(v) => self.run_dequant(&v, output, pool),
            TensorRef::F16(v) => self.run_dequant(&v, output, pool),
            TensorRef::F32(v) => self.run_dequant(&v, output, pool),
        }
    }

    fn run_lut<T: LutBin>(
        &self,
        input: &View<'_, T>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
    ) -> Result<()> {
        let rm = self.reg_max;
        let xyxy = self.format == BoxFormat::Xyxy;
        match &self.kernel {
            DflKernel::Lut { table, neon } => run_blocks(
                input,
                output,
                pool,
                self.grid.lens(),
                block_rows(4 * rm),
                xyxy,
                self.output,
                *neon,
                &|level, start, src, dst| {
                    let n = dst.len() / 4;
                    let Some(span) = self.grid.span(level, start, n) else {
                        return;
                    };
                    let geom = DflGeom {
                        reg_max: rm,
                        anchors: n,
                        grid_x: span.gx,
                        grid_y: span.gy,
                        stride: span.stride,
                    };
                    lut_boxes(src, table, &geom, *neon, dst);
                },
            ),
            #[cfg(target_arch = "aarch64")]
            DflKernel::LutFp16 { table } => {
                self.run_f16(input, output, pool, &|src, d: &mut [f32]| {
                    super::dfl::lut_ltrb(src, table, rm, true, d);
                })
            }
            _ => Err(KernelError::InvalidArgument(
                "dfl_box plan has no table for this input",
            )),
        }
    }

    fn run_dequant<I: DecodeInput>(
        &self,
        input: &View<'_, I>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
    ) -> Result<()> {
        let rm = self.reg_max;
        #[cfg(target_arch = "aarch64")]
        if matches!(self.kernel, DflKernel::SoftmaxFp16) {
            return self.run_f16(input, output, pool, &|src, dst| {
                // SAFETY: `SoftmaxFp16` is selected only when FEAT_FP16 was detected.
                unsafe { super::dfl::softmax_ltrb_f16(self.input, rm, src, dst) }
            });
        }
        run_blocks(
            input,
            output,
            pool,
            self.grid.lens(),
            block_rows(4 * rm),
            self.format == BoxFormat::Xyxy,
            self.output,
            self.tier() != Tier::Scalar,
            &|level, start, src, dst| {
                let Some(span) = self.grid.span(level, start, dst.len() / 4) else {
                    return;
                };
                for_each_ltrb(&self.kernel, self.input, rm, src, |k, ltrb| {
                    let (Some(&x), Some(&y), Some(out)) = (
                        span.gx.get(k),
                        span.gy.get(k),
                        dst.get_mut(4 * k..4 * k + 4),
                    ) else {
                        return;
                    };
                    out.copy_from_slice(&dist2bbox_anchor_f32(ltrb, x, y, span.stride));
                });
            },
        )
    }

    /// The FP16 path: `ltrb` writes a block's distances (f16, or f32 rounded to f16 by the box
    /// kernel), then dist2bbox runs in f16.
    #[cfg(target_arch = "aarch64")]
    fn run_f16<I: Copy + Default + Sync, L: Ltrb8>(
        &self,
        input: &View<'_, I>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
        ltrb: &(dyn Fn(&[I], &mut [L]) + Sync),
    ) -> Result<()> {
        let xyxy = self.format == BoxFormat::Xyxy;
        run_blocks_f16(
            input,
            output,
            pool,
            self.grid.lens(),
            block_rows(4 * self.reg_max),
            &|level, start, src, dst| {
                let Some(Span { gx, gy, stride }) = self.grid.span(level, start, dst.len() / 4)
                else {
                    return;
                };
                let mut store = [L::default(); 4 * BLOCK];
                let Some(d) = store.get_mut(..dst.len()) else {
                    return;
                };
                ltrb(src, d);
                // SAFETY: the FP16 kernels are selected only when FEAT_FP16 was detected.
                unsafe { dist2bbox_f16(d, gx, gy, stride, xyxy, dst) };
            },
        )
    }
}

/// Boxes of one block's 8-bit bins through the exp table: the decoder's 16-bin NEON kernel,
/// the any-`reg_max` NEON kernel, or the portable one, all with the same bits.
#[inline(always)]
fn lut_boxes<T: LutBin>(
    src: &[T],
    table: &crate::prim::dfl_lut::DflExpTable,
    geom: &DflGeom<'_>,
    neon: bool,
    dst: &mut [f32],
) {
    // The table kernels need exact lengths; the block guarantees them.
    if src.len() != geom.anchors * 4 * geom.reg_max || dst.len() != 4 * geom.anchors {
        return;
    }
    #[cfg(target_arch = "aarch64")]
    if neon {
        if geom.reg_max == 16 {
            crate::prim::dfl_lut::decode_dfl_lut_neon(src, table, geom, dst);
        } else {
            crate::prim::dfl_lut::decode_dfl_lut_generic_neon(src, table, geom, dst);
        }
        return;
    }
    let _ = neon;
    decode_dfl_lut(src, table, geom, dst);
}
