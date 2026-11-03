// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! dist2bbox: LTRB grid-unit distances to boxes in pixels over a multi-level anchor axis.

#[cfg(target_arch = "aarch64")]
use super::levels::run_blocks_f16;
use super::levels::{
    block_rows, decode_shapes, input_quant, output_quant, run_blocks, AnchorGrid, BoxFormat,
    DecodeInput, Isa, Level, OutQuant, Span,
};
use crate::cpu::{CpuFeatures, Tier};
use crate::error::{KernelError, Result};
use crate::pool::Pool;
use crate::prim::box_primitives::dist2bbox_anchor_f32;
use crate::quant::{DType, Quant};
use crate::view::{TensorMut, TensorRef, View};

/// Parameters of a dist2bbox decode: `[anchors, 4]` distances `[l, t, r, b]` in grid units to
/// `[anchors, 4]` boxes in pixels.
///
/// `levels` describe the anchor axis in order; their anchors must add up to it. `input`
/// applies to integer input types and is ignored for `f16` and `f32` inputs; `output` applies
/// to integer output types and is ignored for float outputs.
#[derive(Clone, Debug, PartialEq)]
pub struct Dist2BboxParams {
    pub input_dtype: DType,
    pub input: Quant,
    pub output_dtype: DType,
    pub output: Quant,
    pub levels: Vec<Level>,
    pub format: BoxFormat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    /// `prim::dequant` then `dist2bbox_anchor_f32` per anchor.
    Scalar,
    /// `prim::neon_ltrb`: the decoder's NEON dequantize and four-anchor dist2bbox, fused.
    #[cfg(target_arch = "aarch64")]
    Neon,
}

impl Kernel {
    fn select(f: CpuFeatures) -> Self {
        match Isa::of(f) {
            #[cfg(target_arch = "aarch64")]
            Isa::Neon | Isa::NeonFp16 => Self::Neon,
            Isa::Scalar => Self::Scalar,
        }
    }
}

/// A validated dist2bbox decode over a concatenated multi-level anchor axis.
///
/// Anchor `a` with distances `[l, t, r, b]`, cell centre `(gx, gy)` and level stride `s` gives
/// `xc = (gx + (r − l)/2)·s`, `yc = (gy + (b − t)/2)·s`, `w = (l + r)·s`, `h = (t + b)·s`, the
/// Ultralytics operation order, as `prim::box_primitives::dist2bbox_anchor_f32`. The kernel is
/// chosen at creation:
///
/// 1. On NEON (every NEON tier): the decoder's NEON dequantize and four-anchor dist2bbox, fused
///    into one pass of sixteen anchors per step (`prim::neon_ltrb`), bit for bit the decoder's
///    level result for a one-level call. On Cortex-A55 its f32 boxes rounded to f16 are faster
///    than native FP16 boxes, so the FP16 tier runs it too.
/// 2. Without NEON, the scalar f32 kernels.
///
/// `f32` outputs are stored directly. `f16` outputs are the tier's `f32` result rounded once
/// (nearest even); integer outputs are quantized once from the f32
/// result, and [`BoxFormat::Xyxy`] is applied to the f32 `[xc, yc, w, h]`. The input may be
/// anchor-major or channel-major (`[4, anchors]` seen as `[anchors, 4]` with strides
/// `[1, anchors]`); both give the same bits.
///
/// ```
/// use edgefirst_kernels::{
///     BoxFormat, DType, Dist2BboxParams, Dist2BboxPlan, Level, Quant, TensorMut, TensorRef,
///     View, ViewMut,
/// };
///
/// let plan = Dist2BboxPlan::new(Dist2BboxParams {
///     input_dtype: DType::F32,
///     input: Quant::IDENTITY,
///     output_dtype: DType::F32,
///     output: Quant::IDENTITY,
///     levels: vec![Level { anchors: 1, grid_w: 1, stride: 8.0 }],
///     format: BoxFormat::Xyxy,
/// })?;
/// let ltrb = [1.0f32, 1.0, 3.0, 3.0];
/// let mut xyxy = [0.0f32; 4];
/// plan.run(
///     TensorRef::F32(View::contiguous(&ltrb, &[1, 4])?),
///     TensorMut::F32(ViewMut::contiguous(&mut xyxy, &[1, 4])?),
///     None,
/// )?;
/// assert_eq!(xyxy, [-4.0, -4.0, 28.0, 28.0]);
/// # Ok::<(), edgefirst_kernels::KernelError>(())
/// ```
#[derive(Clone, Debug)]
pub struct Dist2BboxPlan {
    input_dtype: DType,
    output_dtype: DType,
    input: Quant,
    output: OutQuant,
    format: BoxFormat,
    grid: AnchorGrid,
    kernel: Kernel,
}

impl Dist2BboxPlan {
    /// Builds a plan for the CPU this process runs on, honouring `EDGEFIRST_KERNELS_FORCE_TIER`.
    pub fn new(p: Dist2BboxParams) -> Result<Self> {
        Self::with_features(p, CpuFeatures::from_env_or_probe()?)
    }

    /// Builds a plan that runs the best kernel `f` allows, and the anchor grid of `p.levels`.
    ///
    /// Levels need a positive `grid_w` that divides their anchors and a finite, positive
    /// stride; there must be at least one. Integer input and output scales must be finite and
    /// positive, and their zero points inside their type's range. Any violation is
    /// `InvalidArgument`. Every input and output type pair is supported.
    pub fn with_features(p: Dist2BboxParams, f: CpuFeatures) -> Result<Self> {
        let input = input_quant(p.input_dtype, p.input)?;
        let output = output_quant(p.output_dtype, p.output)?;
        let grid = AnchorGrid::new(&p.levels)?;
        let kernel = Kernel::select(f);
        Ok(Self {
            input_dtype: p.input_dtype,
            output_dtype: p.output_dtype,
            input,
            output,
            format: p.format,
            grid,
            kernel,
        })
    }

    /// The tier of the kernel this plan runs.
    pub fn tier(&self) -> Tier {
        match self.kernel {
            Kernel::Scalar => Tier::Scalar,
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => Tier::Neon,
        }
    }

    /// Boxes of every anchor of `input` (`[anchors, 4]`) into `output` (`[anchors, 4]`), with
    /// the plan's dtypes; the anchor count must equal the levels' total.
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
        let _span =
            tracing::trace_span!("kernels.dist2bbox", tier = self.tier().as_str()).entered();
        if input.dtype() != self.input_dtype || output.dtype() != self.output_dtype {
            return Err(KernelError::InvalidArgument(
                "dist2bbox input and output dtypes differ from the plan's",
            ));
        }
        if decode_shapes(input.shape(), output.shape(), 4)? != self.grid.total() {
            return Err(KernelError::InvalidArgument(
                "dist2bbox anchor count differs from the levels' total",
            ));
        }
        match input {
            TensorRef::I8(v) => self.run_typed(&v, output, pool),
            TensorRef::U8(v) => self.run_typed(&v, output, pool),
            TensorRef::I16(v) => self.run_typed(&v, output, pool),
            TensorRef::U16(v) => self.run_typed(&v, output, pool),
            TensorRef::F16(v) => self.run_typed(&v, output, pool),
            TensorRef::F32(v) => self.run_typed(&v, output, pool),
        }
    }

    fn run_typed<I: DecodeInput>(
        &self,
        input: &View<'_, I>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
    ) -> Result<()> {
        // `f16` boxes on NEON: the f32 boxes rounded in registers, any format.
        #[cfg(target_arch = "aarch64")]
        if self.kernel == Kernel::Neon && self.output_dtype == DType::F16 {
            let xyxy = self.format == BoxFormat::Xyxy;
            return run_blocks_f16(
                input,
                output,
                pool,
                self.grid.lens(),
                block_rows(4),
                &|level, start, src, dst| {
                    let Some(Span { gx, gy, stride }) = self.grid.span(level, start, dst.len() / 4)
                    else {
                        return;
                    };
                    let q = self.input;
                    // SAFETY: NEON is part of the aarch64 baseline.
                    unsafe {
                        crate::prim::neon_ltrb::ltrb_boxes_f16_neon(
                            src, q, gx, gy, stride, xyxy, dst,
                        );
                    }
                },
            );
        }
        run_blocks(
            input,
            output,
            pool,
            self.grid.lens(),
            block_rows(4),
            self.format == BoxFormat::Xyxy,
            self.output,
            self.tier() != Tier::Scalar,
            &|level, start, src, dst| {
                if let Some(span) = self.grid.span(level, start, dst.len() / 4) {
                    boxes_block(self.kernel, self.input, span, src, dst);
                }
            },
        )
    }
}

/// `[xc, yc, w, h]` of each anchor of `src` into `dst`, four values each.
#[inline(always)]
fn boxes_block<I: DecodeInput>(
    kernel: Kernel,
    q: Quant,
    span: Span<'_>,
    src: &[I],
    dst: &mut [f32],
) {
    let Span { gx, gy, stride } = span;
    let anchors = src
        .chunks_exact(4)
        .zip(dst.chunks_exact_mut(4))
        .zip(gx.iter().zip(gy));
    match kernel {
        Kernel::Scalar => {
            for ((ltrb, out), (&x, &y)) in anchors {
                let mut d = [0.0f32; 4];
                I::dequant_f32(ltrb, q, &mut d, false);
                out.copy_from_slice(&dist2bbox_anchor_f32(d, x, y, stride));
            }
        }
        // SAFETY: NEON is part of the aarch64 baseline.
        #[cfg(target_arch = "aarch64")]
        Kernel::Neon => unsafe {
            crate::prim::neon_ltrb::ltrb_boxes_f32_neon(src, q, gx, gy, stride, false, dst)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel(input_dtype: DType, output_dtype: DType, f: CpuFeatures) -> Kernel {
        Dist2BboxPlan::with_features(
            Dist2BboxParams {
                input_dtype,
                input: Quant::new(0.1, 0),
                output_dtype,
                output: Quant::new(0.1, 0),
                levels: vec![Level {
                    anchors: 4,
                    grid_w: 2,
                    stride: 8.0,
                }],
                format: BoxFormat::Xywh,
            },
            f,
        )
        .unwrap()
        .kernel
    }

    #[test]
    fn path_selection() {
        let s = CpuFeatures::default();
        for d in [DType::I8, DType::U16, DType::F16, DType::F32] {
            for o in [DType::F32, DType::F16, DType::I8] {
                assert_eq!(kernel(d, o, s), Kernel::Scalar);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            let n = CpuFeatures::from_forced_tier(Some("neon")).unwrap();
            let h = CpuFeatures::from_forced_tier(Some("neon_fp16")).ok();
            for d in [
                DType::I8,
                DType::U8,
                DType::I16,
                DType::U16,
                DType::F16,
                DType::F32,
            ] {
                for o in [DType::F32, DType::F16, DType::U16] {
                    assert_eq!(kernel(d, o, n), Kernel::Neon);
                    if let Some(h) = h {
                        assert_eq!(kernel(d, o, h), Kernel::Neon);
                    }
                }
            }
        }
    }
}
