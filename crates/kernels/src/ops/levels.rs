// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Anchor levels, box formats and the block loop shared by the decode plans.

use std::borrow::Cow;

use half::f16;

use super::rows::{for_each_row_block, RowBlockFn, STAGE};
use crate::error::{KernelError, Result};
use crate::pool::Pool;
use crate::prim::dequant as dq;
pub(crate) use crate::quant::{input_quant, output_quant, OutQuant};
use crate::quant::{quantize, Quant};
use crate::view::{TensorMut, View, ViewMut};

/// One FPN level of a concatenated anchor axis.
///
/// Anchor `i` of the level sits at grid cell `(i % grid_w + 0.5, i / grid_w + 0.5)`, and one
/// grid unit is `stride` pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Level {
    pub anchors: usize,
    pub grid_w: usize,
    pub stride: f32,
}

/// The four values of a decoded box; the discriminants are the C ABI values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum BoxFormat {
    /// `[xc, yc, w, h]`, what the YOLOv8/11 tail produces.
    Xywh = 0,
    /// `[xc − w/2, yc − h/2, xc + w/2, yc + h/2]`, from the f32 `[xc, yc, w, h]`.
    Xyxy = 1,
}

/// The instruction set a decode plan's kernels use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Isa {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// FP16 arithmetic: NEON plus FEAT_FP16, seen by the runtime check as well.
    #[cfg(target_arch = "aarch64")]
    NeonFp16,
}

impl Isa {
    /// The best instruction set `f` allows.
    #[cfg(target_arch = "aarch64")]
    pub(crate) fn of(f: crate::cpu::CpuFeatures) -> Self {
        if !f.neon_baseline {
            Self::Scalar
        } else if f.fp16_usable() {
            Self::NeonFp16
        } else {
            Self::Neon
        }
    }

    /// The best instruction set `f` allows.
    #[cfg(not(target_arch = "aarch64"))]
    pub(crate) fn of(_: crate::cpu::CpuFeatures) -> Self {
        Self::Scalar
    }
}

/// Most anchors in one block, the unit of staging and of parallel work.
// Keep a multiple of 4 so blocks group anchors as the decoder's 4-anchor NEON dist2bbox does.
pub(crate) const BLOCK: usize = 64;

/// Largest `reg_max` the DFL plans take.
pub(crate) const MAX_REG_MAX: usize = 64;

/// Anchors per block for input rows of `n_in` elements, so a staged block fits [`STAGE`].
pub(crate) fn block_rows(n_in: usize) -> usize {
    BLOCK.min(STAGE / n_in.max(1)).max(1)
}

/// Anchor centres of every level in grid units, built once when a plan is created.
#[derive(Clone, Debug)]
pub(crate) struct AnchorGrid {
    gx: Box<[f32]>,
    gy: Box<[f32]>,
    /// Anchors per level, in order.
    lens: Box<[usize]>,
    strides: Box<[f32]>,
    total: usize,
}

impl AnchorGrid {
    /// Validates `levels` and builds their centres, as `prim::grids::make_anchor_grid` does.
    pub(crate) fn new(levels: &[Level]) -> Result<Self> {
        if levels.is_empty() {
            return Err(KernelError::InvalidArgument(
                "box decode needs at least one level",
            ));
        }
        let mut total = 0usize;
        for l in levels {
            if l.grid_w == 0 {
                return Err(KernelError::InvalidArgument(
                    "level grid_w must be positive",
                ));
            }
            if l.anchors % l.grid_w != 0 {
                return Err(KernelError::InvalidArgument(
                    "level anchors must be a multiple of grid_w",
                ));
            }
            if !(l.stride.is_finite() && l.stride > 0.0) {
                return Err(KernelError::InvalidArgument(
                    "level stride must be finite and positive",
                ));
            }
            total = total
                .checked_add(l.anchors)
                .ok_or(KernelError::InvalidArgument("level anchors overflow"))?;
        }
        let too_large = |_| KernelError::InvalidArgument("anchor grid too large");
        let (mut gx, mut gy) = (Vec::new(), Vec::new());
        gx.try_reserve_exact(total).map_err(too_large)?;
        gy.try_reserve_exact(total).map_err(too_large)?;
        for l in levels {
            for i in 0..l.anchors {
                gx.push((i % l.grid_w) as f32 + 0.5);
                gy.push((i / l.grid_w) as f32 + 0.5);
            }
        }
        Ok(Self {
            gx: gx.into_boxed_slice(),
            gy: gy.into_boxed_slice(),
            lens: levels.iter().map(|l| l.anchors).collect(),
            strides: levels.iter().map(|l| l.stride).collect(),
            total,
        })
    }

    pub(crate) fn total(&self) -> usize {
        self.total
    }

    pub(crate) fn lens(&self) -> &[usize] {
        &self.lens
    }

    /// Centres of anchors `start..start + len`, and the stride of `level`, which holds them.
    #[inline]
    pub(crate) fn span(&self, level: usize, start: usize, len: usize) -> Option<Span<'_>> {
        let end = start.checked_add(len)?;
        Some(Span {
            gx: self.gx.get(start..end)?,
            gy: self.gy.get(start..end)?,
            stride: *self.strides.get(level)?,
        })
    }
}

/// The anchors of one block: centres in grid units and their level's stride.
#[derive(Clone, Copy)]
pub(crate) struct Span<'a> {
    pub(crate) gx: &'a [f32],
    pub(crate) gy: &'a [f32],
    pub(crate) stride: f32,
}

/// `reg_max` must be in `1..=MAX_REG_MAX`.
pub(crate) fn check_reg_max(reg_max: usize) -> Result<()> {
    if reg_max == 0 {
        return Err(KernelError::InvalidArgument("reg_max must be positive"));
    }
    if reg_max > MAX_REG_MAX {
        return Err(KernelError::Unsupported(Cow::Borrowed(
            "reg_max above 64 is not supported",
        )));
    }
    Ok(())
}

/// The anchor count of a `[anchors, n_in]` input and `[anchors, 4]` output.
pub(crate) fn decode_shapes(input: &[usize], output: &[usize], n_in: usize) -> Result<usize> {
    let (&[rows, cols], &[out_rows, 4]) = (input, output) else {
        return Err(KernelError::InvalidArgument(
            "decode views must be [anchors, channels] and [anchors, 4]",
        ));
    };
    if cols != n_in {
        return Err(KernelError::InvalidArgument(
            "decode input channels do not match the plan",
        ));
    }
    if rows != out_rows {
        return Err(KernelError::InvalidArgument(
            "decode input and output anchor counts differ",
        ));
    }
    Ok(rows)
}

/// The fused NEON LTRB kernels' input bound, empty without NEON.
#[cfg(target_arch = "aarch64")]
pub(crate) use crate::prim::neon_ltrb::LtrbIn as NeonLtrbIn;
/// The fused NEON LTRB kernels' input bound, empty without NEON.
#[cfg(not(target_arch = "aarch64"))]
pub(crate) trait NeonLtrbIn {}
#[cfg(not(target_arch = "aarch64"))]
impl<T> NeonLtrbIn for T {}

/// A decode input element: the `prim` dequantize kernels into f32.
pub(crate) trait DecodeInput: Copy + Default + Send + Sync + NeonLtrbIn {
    /// `prim::dequant`, or the NEON dequantize when `neon`; nothing is written unless the
    /// lengths match.
    fn dequant_f32(src: &[Self], q: Quant, dst: &mut [f32], neon: bool);
}

macro_rules! decode_input {
    ($t:ty, $to_f32:ident, $neon:path) => {
        impl DecodeInput for $t {
            #[inline(always)]
            fn dequant_f32(src: &[Self], q: Quant, dst: &mut [f32], neon: bool) {
                if src.len() != dst.len() {
                    return;
                }
                #[cfg(target_arch = "aarch64")]
                if neon {
                    // SAFETY: NEON is part of the aarch64 baseline; the lengths are equal.
                    unsafe { $neon(src, q, dst) };
                    return;
                }
                let _ = neon;
                dq::$to_f32(src, q, dst);
            }
        }
    };
}

#[cfg(target_arch = "aarch64")]
use crate::prim::{neon_baseline as nb, neon_convert as nc};

decode_input!(i8, dequant_i8_to_f32, nb::dequant_i8_to_f32_neon);
decode_input!(u8, dequant_u8_to_f32, nb::dequant_u8_to_f32_neon);
decode_input!(i16, dequant_i16_to_f32, nb::dequant_i16_to_f32_neon);
decode_input!(u16, dequant_u16_to_f32, nb::dequant_u16_to_f32_neon);
decode_input!(f16, dequant_f16_to_f32, nc::dequant_f16_to_f32_neon);
decode_input!(f32, dequant_f32_to_f32, nc::dequant_f32_to_f32_neon);

/// A decode output element, stored from the f32 result.
trait DecodeOutput: Copy + Default + Send {
    /// Stores `src` (as long as `dst`) with the scalar conversion, or its bit-identical NEON
    /// kernel when `neon`.
    fn store(dst: &mut [Self], src: &[f32], q: OutQuant, neon: bool);
    /// The slice itself when results can be written to it directly.
    fn as_f32(_dst: &mut [Self]) -> Option<&mut [f32]> {
        None
    }
}

impl DecodeOutput for f32 {
    #[inline(always)]
    fn store(dst: &mut [Self], src: &[f32], _: OutQuant, _: bool) {
        for (d, &v) in dst.iter_mut().zip(src) {
            *d = v;
        }
    }

    #[inline(always)]
    fn as_f32(dst: &mut [Self]) -> Option<&mut [f32]> {
        Some(dst)
    }
}

impl DecodeOutput for f16 {
    #[inline(always)]
    fn store(dst: &mut [Self], src: &[f32], _: OutQuant, neon: bool) {
        #[cfg(target_arch = "aarch64")]
        if neon && dst.len() == src.len() {
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe { nc::narrow_f32_to_f16_neon(src, dst) };
            return;
        }
        let _ = neon;
        for (d, &v) in dst.iter_mut().zip(src) {
            *d = f16::from_f32(v);
        }
    }
}

macro_rules! quantized_output {
    ($($t:ty, $neon:ident);*) => {$(
        impl DecodeOutput for $t {
            #[inline(always)]
            fn store(dst: &mut [Self], src: &[f32], q: OutQuant, neon: bool) {
                #[cfg(target_arch = "aarch64")]
                if neon && dst.len() == src.len() {
                    // SAFETY: NEON is part of the aarch64 baseline.
                    unsafe { nc::$neon(src, q.inv_scale, q.zero_point, dst) };
                    return;
                }
                let _ = neon;
                for (d, &v) in dst.iter_mut().zip(src) {
                    *d = quantize::<$t>(v, q.inv_scale, q.zero_point);
                }
            }
        }
    )*};
}
quantized_output!(
    i8, quantize_f32_to_i8_neon;
    u8, quantize_f32_to_u8_neon;
    i16, quantize_f32_to_i16_neon;
    u16, quantize_f32_to_u16_neon
);

#[inline(always)]
fn to_xyxy(values: &mut [f32]) {
    for b in values.chunks_exact_mut(4) {
        if let [xc, yc, w, h] = b {
            let (x, y, hw, hh) = (*xc, *yc, *w * 0.5, *h * 0.5);
            *xc = x - hw;
            *yc = y - hh;
            *w = x + hw;
            *h = y + hh;
        }
    }
}

/// Per-block computation: `(level, first anchor, input rows, 4 f32 values per anchor)`.
pub(crate) type BlockFn<'a, I> = RowBlockFn<'a, I, f32>;

/// Runs `compute` on blocks of at most `rows_per_block` anchors, each inside one segment of
/// `segments` (anchor counts, in order), and stores its four f32 values per anchor into
/// `output` in the output's type, through [`BoxFormat::Xyxy`] first when `xyxy`, with the NEON
/// store kernels when `neon`.
///
/// `compute` must write every value of its block. Blocks start at multiples of
/// `rows_per_block` from their segment's start, whatever the pool, so results depend only on
/// an anchor's index.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_blocks<I: Copy + Default + Sync>(
    input: &View<'_, I>,
    output: TensorMut<'_>,
    pool: Option<&dyn Pool>,
    segments: &[usize],
    rows_per_block: usize,
    xyxy: bool,
    q: OutQuant,
    neon: bool,
    compute: &BlockFn<'_, I>,
) -> Result<()> {
    let b = Blocks {
        segments,
        rows_per_block,
        xyxy,
        q,
        neon,
    };
    match output {
        TensorMut::I8(mut o) => b.run(input, &mut o, pool, compute),
        TensorMut::U8(mut o) => b.run(input, &mut o, pool, compute),
        TensorMut::I16(mut o) => b.run(input, &mut o, pool, compute),
        TensorMut::U16(mut o) => b.run(input, &mut o, pool, compute),
        TensorMut::F16(mut o) => b.run(input, &mut o, pool, compute),
        TensorMut::F32(mut o) => b.run(input, &mut o, pool, compute),
    }
}

/// Runs `compute` on blocks as [`run_blocks`] does, for kernels that write their `f16` values
/// straight into the block, box format included; `output` must be `f16`.
#[cfg(target_arch = "aarch64")]
pub(crate) fn run_blocks_f16<I: Copy + Default + Sync>(
    input: &View<'_, I>,
    output: TensorMut<'_>,
    pool: Option<&dyn Pool>,
    segments: &[usize],
    rows_per_block: usize,
    compute: &RowBlockFn<'_, I, f16>,
) -> Result<()> {
    let TensorMut::F16(mut output) = output else {
        return Err(KernelError::InvalidArgument(
            "decode plan expects an f16 output",
        ));
    };
    if rows_per_block > BLOCK {
        return Err(KernelError::InvalidArgument("decode block too large"));
    }
    for_each_row_block(input, &mut output, pool, segments, rows_per_block, compute)
}

struct Blocks<'a> {
    segments: &'a [usize],
    rows_per_block: usize,
    xyxy: bool,
    q: OutQuant,
    neon: bool,
}

impl Blocks<'_> {
    fn run<I: Copy + Default + Sync, O: DecodeOutput>(
        &self,
        input: &View<'_, I>,
        output: &mut ViewMut<'_, O>,
        pool: Option<&dyn Pool>,
        compute: &BlockFn<'_, I>,
    ) -> Result<()> {
        if self.rows_per_block > BLOCK {
            return Err(KernelError::InvalidArgument("decode block too large"));
        }
        for_each_row_block(
            input,
            output,
            pool,
            self.segments,
            self.rows_per_block,
            &|level, start, src, dst| {
                if !self.xyxy {
                    if let Some(d) = O::as_f32(dst) {
                        compute(level, start, src, d);
                        return;
                    }
                }
                let mut values = [0.0f32; 4 * BLOCK];
                let Some(values) = values.get_mut(..dst.len()) else {
                    return;
                };
                compute(level, start, src, values);
                if self.xyxy {
                    to_xyxy(values);
                }
                O::store(dst, values, self.q, self.neon);
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prim::grids::make_anchor_grid;
    use crate::quant::DType;

    #[test]
    fn grid_matches_make_anchor_grid_per_level() {
        let levels = [
            Level {
                anchors: 6,
                grid_w: 3,
                stride: 8.0,
            },
            Level {
                anchors: 0,
                grid_w: 5,
                stride: 4.0,
            },
            Level {
                anchors: 20,
                grid_w: 5,
                stride: 16.0,
            },
        ];
        let g = AnchorGrid::new(&levels).unwrap();
        assert_eq!(g.total(), 26);
        assert_eq!(g.lens(), &[6, 0, 20]);
        let (x0, y0) = make_anchor_grid(2, 3);
        let (x2, y2) = make_anchor_grid(4, 5);
        let s = g.span(0, 0, 6).unwrap();
        assert_eq!((s.gx, s.gy, s.stride), (&x0[..], &y0[..], 8.0));
        let s = g.span(2, 6, 20).unwrap();
        assert_eq!((s.gx, s.gy, s.stride), (&x2[..], &y2[..], 16.0));
        assert!(g.span(3, 0, 1).is_none());
        assert!(g.span(2, 20, 7).is_none());
        assert!(g.span(2, usize::MAX, 2).is_none());
    }

    #[test]
    fn xyxy_from_xywh() {
        let mut v = [10.0f32, 20.0, 4.0, 6.0, 0.0, 0.0, 1.0, 3.0];
        to_xyxy(&mut v);
        assert_eq!(v, [8.0, 17.0, 12.0, 23.0, -0.5, -1.5, 0.5, 1.5]);
    }

    /// The NEON stores and dequantizes give the scalar bits, at every block length.
    #[cfg(target_arch = "aarch64")]
    #[test]
    fn store_and_dequant_neon_match_scalar() {
        fn store<O: DecodeOutput, B: PartialEq + std::fmt::Debug>(
            src: &[f32],
            q: OutQuant,
            bits: impl Fn(&O) -> B,
        ) {
            let (mut a, mut b) = (vec![O::default(); src.len()], vec![O::default(); src.len()]);
            O::store(&mut a, src, q, true);
            O::store(&mut b, src, q, false);
            let (a, b): (Vec<B>, Vec<B>) =
                (a.iter().map(&bits).collect(), b.iter().map(&bits).collect());
            assert_eq!(a, b, "{q:?}");
        }
        let values: Vec<f32> = (0..4 * BLOCK + 3)
            .map(|i| (i as f32 - 130.0) * 0.731)
            .chain([f32::NAN, f32::INFINITY, -1e30, 0.5, -2.5])
            .collect();
        for n in [0, 1, 4, 15, 16, 17, values.len()] {
            let src = &values[..n];
            for (inv_scale, zero_point) in [(1.0, 0), (3.7, -128), (0.1, 255), (1e6, 65535)] {
                let q = OutQuant {
                    inv_scale,
                    zero_point,
                };
                store(src, q, |&v: &i8| v);
                store(src, q, |&v: &u8| v);
                store(src, q, |&v: &i16| v);
                store(src, q, |&v: &u16| v);
                store(src, q, |v: &f16| v.to_bits());
            }
            let halves: Vec<f16> = src.iter().map(|&v| f16::from_f32(v)).collect();
            for q in [Quant::IDENTITY, Quant::new(0.25, 3)] {
                let (mut a, mut b) = (vec![0.0f32; n], vec![0.0f32; n]);
                f32::dequant_f32(src, q, &mut a, true);
                f32::dequant_f32(src, q, &mut b, false);
                assert_eq!(
                    a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                f16::dequant_f32(&halves, q, &mut a, true);
                f16::dequant_f32(&halves, q, &mut b, false);
                assert_eq!(
                    a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn block_rows_fit_the_stage() {
        assert_eq!(block_rows(64), 64);
        assert_eq!(block_rows(4), 64);
        assert_eq!(block_rows(256), 16);
        assert_eq!(block_rows(0), 64);
    }

    #[test]
    fn quant_validation() {
        assert_eq!(
            input_quant(DType::F32, Quant::new(f32::NAN, 9)).unwrap(),
            Quant::IDENTITY
        );
        assert!(input_quant(DType::U8, Quant::new(0.1, 256)).is_err());
        assert!(input_quant(DType::I16, Quant::new(-0.1, 0)).is_err());
        assert!(output_quant(DType::F16, Quant::new(f32::NAN, 0)).is_ok());
        assert!(output_quant(DType::U16, Quant::new(1e-39, 0)).is_err());
        assert!(output_quant(DType::I8, Quant::new(0.1, -129)).is_err());
        assert!(matches!(
            check_reg_max(65),
            Err(KernelError::Unsupported(_))
        ));
        assert!(matches!(
            check_reg_max(0),
            Err(KernelError::InvalidArgument(_))
        ));
    }
}
