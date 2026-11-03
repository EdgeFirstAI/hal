// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Plan-time selected kernel dispatches.
//!
//! Scalar and NEON variants are dispatched at plan time. The `select()`
//! method picks the highest-tier kernel available for the current CPU.

use super::CpuFeatures;
use crate::per_scale::DecodeDtype;
use crate::schema::BoxEncoding;
use crate::{DecoderError, DecoderResult};
use edgefirst_tensor::DType;

/// Box-level dispatch tagged by `(encoding, input_dtype, output_dtype, tier)`.
///
/// `Dfl*` variants run dequant → softmax → weighted-sum → dist2bbox.
/// `Ltrb*` variants run dequant → dist2bbox.
///
/// 8-bit DFL inputs take the `*Lut` cells, which find each side's maximum over the raw bins and
/// weigh bins through an exp table (see [`edgefirst_kernels::prim::dfl_lut`]); `*LutNeon` is the
/// same arithmetic with the four sides in NEON lanes, bit for bit, for `f16` outputs too (the
/// f32 boxes rounded once). Other DFL inputs on NEON dequantize each anchor (the fused FMA for
/// 16-bit inputs, exact for floats), take the fused NEON softmax and weighted sum (the fp16 exp
/// for 16-bit inputs into f32 on FP16 tiers) and dist2bbox four anchors per step; their `f16`
/// outputs are the f32 boxes rounded once on `neon`, and native FP16 distances and boxes on
/// `neon_fp16`. LTRB on NEON is `prim::neon_ltrb`'s fused dequantize and dist2bbox on every tier;
/// its `f16` outputs are the f32 boxes rounded once, which beat native FP16 boxes on Cortex-A55.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::enum_variant_names)] // tier suffix is meaningful
pub(crate) enum BoxLevelDispatch {
    // ── DFL encoding (yolov8, yolo11) ────────────────────
    DflI8ToF32Lut,
    DflU8ToF32Lut,
    DflI16ToF32Scalar,
    DflU16ToF32Scalar,
    DflF16ToF32Scalar,
    DflF32ToF32Scalar,
    DflI8ToF16Lut,
    DflU8ToF16Lut,
    DflI16ToF16Scalar,
    DflU16ToF16Scalar,
    DflF16ToF16Scalar,
    DflF32ToF16Scalar,

    // ── LTRB encoding (yolo26) ───────────────────────────
    LtrbI8ToF32Scalar,
    LtrbU8ToF32Scalar,
    LtrbI16ToF32Scalar,
    LtrbU16ToF32Scalar,
    LtrbF16ToF32Scalar,
    LtrbF32ToF32Scalar,
    LtrbI8ToF16Scalar,
    LtrbU8ToF16Scalar,
    LtrbI16ToF16Scalar,
    LtrbU16ToF16Scalar,
    LtrbF16ToF16Scalar,
    LtrbF32ToF16Scalar,

    // NEON (aarch64, `neon_baseline`).
    #[cfg(target_arch = "aarch64")]
    DflI8ToF32LutNeon,
    #[cfg(target_arch = "aarch64")]
    DflU8ToF32LutNeon,
    #[cfg(target_arch = "aarch64")]
    DflI8ToF16LutNeon,
    #[cfg(target_arch = "aarch64")]
    DflU8ToF16LutNeon,
    #[cfg(target_arch = "aarch64")]
    DflI16ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    DflU16ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    DflF16ToF32Neon,
    #[cfg(target_arch = "aarch64")]
    DflF32ToF32Neon,
    #[cfg(target_arch = "aarch64")]
    DflI16ToF16Neon,
    #[cfg(target_arch = "aarch64")]
    DflU16ToF16Neon,
    #[cfg(target_arch = "aarch64")]
    DflF16ToF16Neon,
    #[cfg(target_arch = "aarch64")]
    DflF32ToF16Neon,
    #[cfg(target_arch = "aarch64")]
    LtrbI8ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    LtrbU8ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    LtrbI16ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    LtrbU16ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    LtrbF16ToF32Neon,
    #[cfg(target_arch = "aarch64")]
    LtrbF32ToF32Neon,
    #[cfg(target_arch = "aarch64")]
    LtrbI8ToF16Neon,
    #[cfg(target_arch = "aarch64")]
    LtrbU8ToF16Neon,
    #[cfg(target_arch = "aarch64")]
    LtrbI16ToF16Neon,
    #[cfg(target_arch = "aarch64")]
    LtrbU16ToF16Neon,
    #[cfg(target_arch = "aarch64")]
    LtrbF16ToF16Neon,
    #[cfg(target_arch = "aarch64")]
    LtrbF32ToF16Neon,

    // NEON FP16 (`neon_fp16`, Cortex-A55+); see the enum doc for which cells use it.
    #[cfg(target_arch = "aarch64")]
    DflI16ToF32NeonFp16,
    #[cfg(target_arch = "aarch64")]
    DflU16ToF32NeonFp16,
    #[cfg(target_arch = "aarch64")]
    DflI16ToF16NeonFp16,
    #[cfg(target_arch = "aarch64")]
    DflU16ToF16NeonFp16,
    #[cfg(target_arch = "aarch64")]
    DflF16ToF16NeonFp16,
    #[cfg(target_arch = "aarch64")]
    DflF32ToF16NeonFp16,
}

impl BoxLevelDispatch {
    /// Select the highest-tier kernel available for the given configuration.
    pub(crate) fn select(
        encoding: BoxEncoding,
        input: DType,
        output: DecodeDtype,
        features: &CpuFeatures,
    ) -> DecoderResult<Self> {
        #[cfg(target_arch = "aarch64")]
        if features.neon_baseline {
            let fp16 = if features.neon_fp16 {
                Self::select_neon_fp16(encoding, input, output)
            } else {
                None
            };
            if let Some(cell) = fp16.or_else(|| Self::select_neon(encoding, input, output)) {
                return Ok(cell);
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        let _ = features;
        Self::select_scalar(encoding, input, output)
    }

    /// The FP16-tier cell, where it differs from the `neon` one.
    #[cfg(target_arch = "aarch64")]
    fn select_neon_fp16(encoding: BoxEncoding, input: DType, output: DecodeDtype) -> Option<Self> {
        use BoxLevelDispatch::*;
        Some(match (encoding, input, output) {
            (BoxEncoding::Dfl, DType::I16, DecodeDtype::F32) => DflI16ToF32NeonFp16,
            (BoxEncoding::Dfl, DType::U16, DecodeDtype::F32) => DflU16ToF32NeonFp16,
            (BoxEncoding::Dfl, DType::I16, DecodeDtype::F16) => DflI16ToF16NeonFp16,
            (BoxEncoding::Dfl, DType::U16, DecodeDtype::F16) => DflU16ToF16NeonFp16,
            (BoxEncoding::Dfl, DType::F16, DecodeDtype::F16) => DflF16ToF16NeonFp16,
            (BoxEncoding::Dfl, DType::F32, DecodeDtype::F16) => DflF32ToF16NeonFp16,
            _ => return None,
        })
    }

    /// The baseline NEON cell.
    #[cfg(target_arch = "aarch64")]
    fn select_neon(encoding: BoxEncoding, input: DType, output: DecodeDtype) -> Option<Self> {
        use BoxLevelDispatch::*;
        Some(match (encoding, input, output) {
            (BoxEncoding::Dfl, DType::I8, DecodeDtype::F32) => DflI8ToF32LutNeon,
            (BoxEncoding::Dfl, DType::U8, DecodeDtype::F32) => DflU8ToF32LutNeon,
            (BoxEncoding::Dfl, DType::I8, DecodeDtype::F16) => DflI8ToF16LutNeon,
            (BoxEncoding::Dfl, DType::U8, DecodeDtype::F16) => DflU8ToF16LutNeon,
            (BoxEncoding::Dfl, DType::I16, DecodeDtype::F32) => DflI16ToF32NeonBase,
            (BoxEncoding::Dfl, DType::U16, DecodeDtype::F32) => DflU16ToF32NeonBase,
            (BoxEncoding::Dfl, DType::F16, DecodeDtype::F32) => DflF16ToF32Neon,
            (BoxEncoding::Dfl, DType::F32, DecodeDtype::F32) => DflF32ToF32Neon,
            (BoxEncoding::Dfl, DType::I16, DecodeDtype::F16) => DflI16ToF16Neon,
            (BoxEncoding::Dfl, DType::U16, DecodeDtype::F16) => DflU16ToF16Neon,
            (BoxEncoding::Dfl, DType::F16, DecodeDtype::F16) => DflF16ToF16Neon,
            (BoxEncoding::Dfl, DType::F32, DecodeDtype::F16) => DflF32ToF16Neon,
            (BoxEncoding::Direct, DType::I8, DecodeDtype::F32) => LtrbI8ToF32NeonBase,
            (BoxEncoding::Direct, DType::U8, DecodeDtype::F32) => LtrbU8ToF32NeonBase,
            (BoxEncoding::Direct, DType::I16, DecodeDtype::F32) => LtrbI16ToF32NeonBase,
            (BoxEncoding::Direct, DType::U16, DecodeDtype::F32) => LtrbU16ToF32NeonBase,
            (BoxEncoding::Direct, DType::F16, DecodeDtype::F32) => LtrbF16ToF32Neon,
            (BoxEncoding::Direct, DType::F32, DecodeDtype::F32) => LtrbF32ToF32Neon,
            (BoxEncoding::Direct, DType::I8, DecodeDtype::F16) => LtrbI8ToF16Neon,
            (BoxEncoding::Direct, DType::U8, DecodeDtype::F16) => LtrbU8ToF16Neon,
            (BoxEncoding::Direct, DType::I16, DecodeDtype::F16) => LtrbI16ToF16Neon,
            (BoxEncoding::Direct, DType::U16, DecodeDtype::F16) => LtrbU16ToF16Neon,
            (BoxEncoding::Direct, DType::F16, DecodeDtype::F16) => LtrbF16ToF16Neon,
            (BoxEncoding::Direct, DType::F32, DecodeDtype::F16) => LtrbF32ToF16Neon,
            _ => return None,
        })
    }

    fn select_scalar(
        encoding: BoxEncoding,
        input: DType,
        output: DecodeDtype,
    ) -> DecoderResult<Self> {
        use BoxLevelDispatch::*;
        let cell = match (encoding, input, output) {
            (BoxEncoding::Dfl, DType::I8, DecodeDtype::F32) => DflI8ToF32Lut,
            (BoxEncoding::Dfl, DType::U8, DecodeDtype::F32) => DflU8ToF32Lut,
            (BoxEncoding::Dfl, DType::I16, DecodeDtype::F32) => DflI16ToF32Scalar,
            (BoxEncoding::Dfl, DType::U16, DecodeDtype::F32) => DflU16ToF32Scalar,
            (BoxEncoding::Dfl, DType::F16, DecodeDtype::F32) => DflF16ToF32Scalar,
            (BoxEncoding::Dfl, DType::F32, DecodeDtype::F32) => DflF32ToF32Scalar,
            (BoxEncoding::Dfl, DType::I8, DecodeDtype::F16) => DflI8ToF16Lut,
            (BoxEncoding::Dfl, DType::U8, DecodeDtype::F16) => DflU8ToF16Lut,
            (BoxEncoding::Dfl, DType::I16, DecodeDtype::F16) => DflI16ToF16Scalar,
            (BoxEncoding::Dfl, DType::U16, DecodeDtype::F16) => DflU16ToF16Scalar,
            (BoxEncoding::Dfl, DType::F16, DecodeDtype::F16) => DflF16ToF16Scalar,
            (BoxEncoding::Dfl, DType::F32, DecodeDtype::F16) => DflF32ToF16Scalar,

            (BoxEncoding::Direct, DType::I8, DecodeDtype::F32) => LtrbI8ToF32Scalar,
            (BoxEncoding::Direct, DType::U8, DecodeDtype::F32) => LtrbU8ToF32Scalar,
            (BoxEncoding::Direct, DType::I16, DecodeDtype::F32) => LtrbI16ToF32Scalar,
            (BoxEncoding::Direct, DType::U16, DecodeDtype::F32) => LtrbU16ToF32Scalar,
            (BoxEncoding::Direct, DType::F16, DecodeDtype::F32) => LtrbF16ToF32Scalar,
            (BoxEncoding::Direct, DType::F32, DecodeDtype::F32) => LtrbF32ToF32Scalar,
            (BoxEncoding::Direct, DType::I8, DecodeDtype::F16) => LtrbI8ToF16Scalar,
            (BoxEncoding::Direct, DType::U8, DecodeDtype::F16) => LtrbU8ToF16Scalar,
            (BoxEncoding::Direct, DType::I16, DecodeDtype::F16) => LtrbI16ToF16Scalar,
            (BoxEncoding::Direct, DType::U16, DecodeDtype::F16) => LtrbU16ToF16Scalar,
            (BoxEncoding::Direct, DType::F16, DecodeDtype::F16) => LtrbF16ToF16Scalar,
            (BoxEncoding::Direct, DType::F32, DecodeDtype::F16) => LtrbF32ToF16Scalar,

            (enc, dt, _) => {
                return Err(DecoderError::NotSupported(format!(
                    "per-scale BoxLevelDispatch: encoding {enc:?} + input {dt:?} \
                     not in fast-path matrix"
                )));
            }
        };
        Ok(cell)
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // Wired by Task 15.
#[allow(clippy::enum_variant_names)] // tier suffix is meaningful
pub(crate) enum ScoreLevelDispatch {
    I8ToF32Scalar,
    U8ToF32Scalar,
    I16ToF32Scalar,
    U16ToF32Scalar,
    F16ToF32Scalar,
    F32ToF32Scalar,
    I8ToF16Scalar,
    U8ToF16Scalar,
    I16ToF16Scalar,
    U16ToF16Scalar,
    F16ToF16Scalar,
    F32ToF16Scalar,
    // NEON (aarch64, `neon_baseline`): integer inputs into f32.
    #[cfg(target_arch = "aarch64")]
    I8ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    U8ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    I16ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    U16ToF32NeonBase,

    // Tier 1-fused: NEON fused dequant+sigmoid in one pass (no intermediate
    // store/reload). Same CpuFeatures gate as NeonBase.
    #[cfg(target_arch = "aarch64")]
    I8ToF32NeonFused,
    #[cfg(target_arch = "aarch64")]
    U8ToF32NeonFused,
    #[cfg(target_arch = "aarch64")]
    I16ToF32NeonFused,
    #[cfg(target_arch = "aarch64")]
    U16ToF32NeonFused,

    // NEON: float inputs into f32, and every input into f16 with an f32 sigmoid rounded once.
    #[cfg(target_arch = "aarch64")]
    F16ToF32Neon,
    #[cfg(target_arch = "aarch64")]
    F32ToF32Neon,
    #[cfg(target_arch = "aarch64")]
    I8ToF16Neon,
    #[cfg(target_arch = "aarch64")]
    U8ToF16Neon,
    #[cfg(target_arch = "aarch64")]
    I16ToF16Neon,
    #[cfg(target_arch = "aarch64")]
    U16ToF16Neon,
    #[cfg(target_arch = "aarch64")]
    F16ToF16Neon,
    #[cfg(target_arch = "aarch64")]
    F32ToF16Neon,

    // Scores into f16 on `neon_fp16`: the native FP16 fused dequantize and sigmoid.
    #[cfg(target_arch = "aarch64")]
    I8ToF16NeonFp16,
    #[cfg(target_arch = "aarch64")]
    U8ToF16NeonFp16,
    #[cfg(target_arch = "aarch64")]
    I16ToF16NeonFp16,
    #[cfg(target_arch = "aarch64")]
    U16ToF16NeonFp16,
    #[cfg(target_arch = "aarch64")]
    F16ToF16NeonFp16,
    #[cfg(target_arch = "aarch64")]
    F32ToF16NeonFp16,
}

impl ScoreLevelDispatch {
    pub(crate) fn select(
        input: DType,
        output: DecodeDtype,
        features: &CpuFeatures,
    ) -> DecoderResult<Self> {
        #[cfg(target_arch = "aarch64")]
        {
            use ScoreLevelDispatch::*;
            // f32 outputs take the f32 NEON sigmoid on every tier; the FP16 exp is for f16 outputs.
            if features.neon_baseline {
                let cell = match (input, output) {
                    (DType::I8, DecodeDtype::F32) => Some(I8ToF32NeonFused),
                    (DType::U8, DecodeDtype::F32) => Some(U8ToF32NeonFused),
                    (DType::I16, DecodeDtype::F32) => Some(I16ToF32NeonFused),
                    (DType::U16, DecodeDtype::F32) => Some(U16ToF32NeonFused),
                    (DType::I8, DecodeDtype::F16) if features.neon_fp16 => Some(I8ToF16NeonFp16),
                    (DType::U8, DecodeDtype::F16) if features.neon_fp16 => Some(U8ToF16NeonFp16),
                    (DType::I16, DecodeDtype::F16) if features.neon_fp16 => Some(I16ToF16NeonFp16),
                    (DType::U16, DecodeDtype::F16) if features.neon_fp16 => Some(U16ToF16NeonFp16),
                    (DType::F16, DecodeDtype::F16) if features.neon_fp16 => Some(F16ToF16NeonFp16),
                    (DType::F32, DecodeDtype::F16) if features.neon_fp16 => Some(F32ToF16NeonFp16),
                    _ => Self::select_neon(input, output),
                };
                if let Some(cell) = cell {
                    return Ok(cell);
                }
            }
        }
        Self::select_scalar(input, output, features)
    }

    /// The NEON cell shared by every role: float inputs into f32, every
    /// input into f16.
    #[cfg(target_arch = "aarch64")]
    fn select_neon(input: DType, output: DecodeDtype) -> Option<Self> {
        use ScoreLevelDispatch::*;
        Some(match (input, output) {
            (DType::F16, DecodeDtype::F32) => F16ToF32Neon,
            (DType::F32, DecodeDtype::F32) => F32ToF32Neon,
            (DType::I8, DecodeDtype::F16) => I8ToF16Neon,
            (DType::U8, DecodeDtype::F16) => U8ToF16Neon,
            (DType::I16, DecodeDtype::F16) => I16ToF16Neon,
            (DType::U16, DecodeDtype::F16) => U16ToF16Neon,
            (DType::F16, DecodeDtype::F16) => F16ToF16Neon,
            (DType::F32, DecodeDtype::F16) => F32ToF16Neon,
            _ => return None,
        })
    }

    /// Dequant-only selector: picks NeonBase (no fused sigmoid) for
    /// mc/proto paths that have no activation step.
    pub(crate) fn select_dequant_only(
        input: DType,
        output: DecodeDtype,
        features: &CpuFeatures,
    ) -> DecoderResult<Self> {
        #[cfg(target_arch = "aarch64")]
        {
            use ScoreLevelDispatch::*;
            if features.neon_baseline {
                match (input, output) {
                    (DType::I8, DecodeDtype::F32) => return Ok(I8ToF32NeonBase),
                    (DType::U8, DecodeDtype::F32) => return Ok(U8ToF32NeonBase),
                    (DType::I16, DecodeDtype::F32) => return Ok(I16ToF32NeonBase),
                    (DType::U16, DecodeDtype::F32) => return Ok(U16ToF32NeonBase),
                    _ => {
                        if let Some(cell) = Self::select_neon(input, output) {
                            return Ok(cell);
                        }
                    }
                }
            }
        }
        Self::select_scalar(input, output, features)
    }

    fn select_scalar(
        input: DType,
        output: DecodeDtype,
        _features: &CpuFeatures,
    ) -> DecoderResult<Self> {
        use ScoreLevelDispatch::*;
        Ok(match (input, output) {
            (DType::I8, DecodeDtype::F32) => I8ToF32Scalar,
            (DType::U8, DecodeDtype::F32) => U8ToF32Scalar,
            (DType::I16, DecodeDtype::F32) => I16ToF32Scalar,
            (DType::U16, DecodeDtype::F32) => U16ToF32Scalar,
            (DType::F16, DecodeDtype::F32) => F16ToF32Scalar,
            (DType::F32, DecodeDtype::F32) => F32ToF32Scalar,
            (DType::I8, DecodeDtype::F16) => I8ToF16Scalar,
            (DType::U8, DecodeDtype::F16) => U8ToF16Scalar,
            (DType::I16, DecodeDtype::F16) => I16ToF16Scalar,
            (DType::U16, DecodeDtype::F16) => U16ToF16Scalar,
            (DType::F16, DecodeDtype::F16) => F16ToF16Scalar,
            (DType::F32, DecodeDtype::F16) => F32ToF16Scalar,
            (dt, _) => {
                return Err(DecoderError::NotSupported(format!(
                    "per-scale ScoreLevelDispatch: input {dt:?} not in current matrix"
                )));
            }
        })
    }
}

/// Mask-coefficient dispatch — same (I, F) matrix as scores but
/// `run()` takes no activation parameter. Newtype around
/// `ScoreLevelDispatch` to share the variant list.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // Wired by Task 15.
pub(crate) struct MaskCoefDispatch(pub(crate) ScoreLevelDispatch);

impl MaskCoefDispatch {
    pub(crate) fn select(
        input: DType,
        output: DecodeDtype,
        f: &CpuFeatures,
    ) -> DecoderResult<Self> {
        // Pure dequant (no activation) — skip FP16 sigmoid and fused
        // dequant+sigmoid tiers.
        ScoreLevelDispatch::select_dequant_only(input, output, f).map(Self)
    }
}

/// Proto dispatch — single-tensor dequant; same (I, F) matrix.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // Wired by Task 15.
pub(crate) struct ProtoDispatch(pub(crate) ScoreLevelDispatch);

impl ProtoDispatch {
    pub(crate) fn select(
        input: DType,
        output: DecodeDtype,
        f: &CpuFeatures,
    ) -> DecoderResult<Self> {
        // Pure dequant (no activation) — skip FP16 sigmoid and fused tiers.
        ScoreLevelDispatch::select_dequant_only(input, output, f).map(Self)
    }
}

/// Per-channel DFL box kernels: one scale and zero point per bin, which the per-tensor
/// [`BoxLevelDispatch`] cannot carry.
///
/// `Neon` dequantizes each bin exactly as the scalar cell does, then runs the f32 NEON softmax
/// and weighted sum and dist2bbox four anchors per step; its `f16` boxes are the f32 boxes
/// rounded once. `NeonFp16` (`f16` outputs on FP16 tiers) takes native FP16 distances and boxes
/// after the same dequantize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DflPerChannelDispatch {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "aarch64")]
    NeonFp16,
}

impl DflPerChannelDispatch {
    pub(crate) fn select(output: DecodeDtype, features: &CpuFeatures) -> Self {
        #[cfg(target_arch = "aarch64")]
        if features.neon_baseline {
            return if features.neon_fp16 && output == DecodeDtype::F16 {
                Self::NeonFp16
            } else {
                Self::Neon
            };
        }
        let _ = (output, features);
        Self::Scalar
    }

    /// Decodes one level; integer inputs only, as NPUs emit per-channel quantization on those.
    /// `scales` must hold one scale per bin and `zps` none (all zero) or one per bin
    /// (`InvalidConfig`), and the slices `level.h · level.w` anchors (`InvalidShape`), both
    /// checked before anything is written.
    pub(crate) fn run(
        &self,
        input: InputView<'_>,
        scales: &[f32],
        zps: &[i32],
        level: &LevelPlan,
        dst: DstSliceMut<'_>,
    ) -> DecoderResult<()> {
        use super::level_box::{
            decode_box_level_dfl_i16_to_f16_per_channel as i16_f16,
            decode_box_level_dfl_i16_to_f32_per_channel as i16_f32,
            decode_box_level_dfl_i8_to_f16_per_channel as i8_f16,
            decode_box_level_dfl_i8_to_f32_per_channel as i8_f32,
            decode_box_level_dfl_u16_to_f16_per_channel as u16_f16,
            decode_box_level_dfl_u16_to_f32_per_channel as u16_f32,
            decode_box_level_dfl_u8_to_f16_per_channel as u8_f16,
            decode_box_level_dfl_u8_to_f32_per_channel as u8_f32,
        };
        check_box_lengths(&input, &dst, false, level)?;
        let ch = 4 * level.reg_max;
        if scales.len() != ch || !(zps.is_empty() || zps.len() == ch) {
            return Err(DecoderError::InvalidConfig(format!(
                "per-channel DFL boxes need {ch} scales and no or {ch} zero points, got {} and {}",
                scales.len(),
                zps.len()
            )));
        }
        #[cfg(target_arch = "aarch64")]
        if *self != Self::Scalar {
            use super::level_box::dfl_per_channel_neon as pc;
            let fp16 = *self == Self::NeonFp16;
            // SAFETY: `NeonFp16` is selected only when FEAT_FP16 was probed.
            let ok = unsafe {
                match (input, dst) {
                    (InputView::I8(i), DstSliceMut::F32(d)) => pc(i, scales, zps, level, fp16, d),
                    (InputView::U8(i), DstSliceMut::F32(d)) => pc(i, scales, zps, level, fp16, d),
                    (InputView::I16(i), DstSliceMut::F32(d)) => pc(i, scales, zps, level, fp16, d),
                    (InputView::U16(i), DstSliceMut::F32(d)) => pc(i, scales, zps, level, fp16, d),
                    (InputView::I8(i), DstSliceMut::F16(d)) => pc(i, scales, zps, level, fp16, d),
                    (InputView::U8(i), DstSliceMut::F16(d)) => pc(i, scales, zps, level, fp16, d),
                    (InputView::I16(i), DstSliceMut::F16(d)) => pc(i, scales, zps, level, fp16, d),
                    (InputView::U16(i), DstSliceMut::F16(d)) => pc(i, scales, zps, level, fp16, d),
                    _ => false,
                }
            };
            return if ok {
                Ok(())
            } else {
                Err(DecoderError::KernelDispatchUnreachable(
                    "per-channel DFL: float inputs do not support per-channel quant".into(),
                ))
            };
        }
        match (input, dst) {
            (InputView::I8(i), DstSliceMut::F32(d)) => i8_f32(i, scales, zps, level, d),
            (InputView::U8(i), DstSliceMut::F32(d)) => u8_f32(i, scales, zps, level, d),
            (InputView::I16(i), DstSliceMut::F32(d)) => i16_f32(i, scales, zps, level, d),
            (InputView::U16(i), DstSliceMut::F32(d)) => u16_f32(i, scales, zps, level, d),
            (InputView::I8(i), DstSliceMut::F16(d)) => i8_f16(i, scales, zps, level, d),
            (InputView::U8(i), DstSliceMut::F16(d)) => u8_f16(i, scales, zps, level, d),
            (InputView::I16(i), DstSliceMut::F16(d)) => i16_f16(i, scales, zps, level, d),
            (InputView::U16(i), DstSliceMut::F16(d)) => u16_f16(i, scales, zps, level, d),
            _ => {
                return Err(DecoderError::KernelDispatchUnreachable(
                    "per-channel DFL: float inputs do not support per-channel quant".into(),
                ))
            }
        }
        Ok(())
    }
}

/// The NCHW → NHWC transpose of NCHW inputs: NEON on NEON tiers, the element loop otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransposeDispatch {
    Scalar,
    Neon,
}

impl TransposeDispatch {
    pub(crate) fn select(features: &CpuFeatures) -> Self {
        if features.neon_baseline {
            Self::Neon
        } else {
            Self::Scalar
        }
    }

    /// Transposes `src` (`[c, h, w]`) into `dst` (`[h, w, c]`); writes nothing unless both hold
    /// `h · w · c` elements.
    #[inline]
    pub(crate) fn run<T: edgefirst_kernels::prim::transpose::TransposeElement>(
        self,
        src: &[T],
        h: usize,
        w: usize,
        c: usize,
        dst: &mut [T],
    ) {
        use edgefirst_kernels::prim::transpose::{nchw_to_nhwc, nchw_to_nhwc_scalar};
        match self {
            Self::Neon => nchw_to_nhwc(src, h, w, c, dst),
            Self::Scalar => nchw_to_nhwc_scalar(src, h, w, c, dst),
        }
    }
}

// ── dtype-tagged input/output views ─────────────────────────────────────

use half::f16;

/// Input slice view, dtype-tagged. The dispatch arm matches on this
/// to pick the right concrete kernel function.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum InputView<'a> {
    I8(&'a [i8]),
    U8(&'a [u8]),
    I16(&'a [i16]),
    U16(&'a [u16]),
    F16(&'a [f16]),
    F32(&'a [f32]),
}

impl<'a> InputView<'a> {
    /// The number of elements.
    pub(crate) fn len(&self) -> usize {
        match self {
            InputView::I8(s) => s.len(),
            InputView::U8(s) => s.len(),
            InputView::I16(s) => s.len(),
            InputView::U16(s) => s.len(),
            InputView::F16(s) => s.len(),
            InputView::F32(s) => s.len(),
        }
    }

    /// The elements `range` of the view.
    pub(crate) fn slice(&self, range: std::ops::Range<usize>) -> InputView<'a> {
        match *self {
            InputView::I8(s) => InputView::I8(&s[range]),
            InputView::U8(s) => InputView::U8(&s[range]),
            InputView::I16(s) => InputView::I16(&s[range]),
            InputView::U16(s) => InputView::U16(&s[range]),
            InputView::F16(s) => InputView::F16(&s[range]),
            InputView::F32(s) => InputView::F32(&s[range]),
        }
    }
}

/// Mutable destination slice, dtype-tagged. The dispatch arm matches
/// on this to pick the f32 vs f16 output cell.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum DstSliceMut<'a> {
    F32(&'a mut [f32]),
    F16(&'a mut [f16]),
}

impl DstSliceMut<'_> {
    /// The number of elements.
    pub(crate) fn len(&self) -> usize {
        match self {
            DstSliceMut::F32(d) => d.len(),
            DstSliceMut::F16(d) => d.len(),
        }
    }
}

/// Checks one box level call: `level.h · level.w` anchors, each with `per` input values, four
/// outputs and a grid centre; a DFL `reg_max` must be in `1..=64`.
fn check_box_lengths(
    input: &InputView<'_>,
    dst: &DstSliceMut<'_>,
    ltrb: bool,
    level: &LevelPlan,
) -> DecoderResult<()> {
    let per = if ltrb {
        Some(4)
    } else if (1..=64).contains(&level.reg_max) {
        Some(4 * level.reg_max)
    } else {
        None
    };
    let n = level.h.checked_mul(level.w);
    let ok = n.zip(per).is_some_and(|(n, per)| {
        Some(input.len()) == n.checked_mul(per)
            && Some(dst.len()) == n.checked_mul(4)
            && level.grid_x.len() >= n
            && level.grid_y.len() >= n
    });
    if ok {
        return Ok(());
    }
    Err(DecoderError::InvalidShape(format!(
        "box level of {}x{} anchors (reg_max {}): {} input and {} output values do not match",
        level.h,
        level.w,
        level.reg_max,
        input.len(),
        dst.len()
    )))
}

// ── BoxLevelDispatch::run ───────────────────────────────────────────────

use super::level_box::{
    decode_box_level_dfl_f16_to_f16, decode_box_level_dfl_f16_to_f32,
    decode_box_level_dfl_f32_to_f16, decode_box_level_dfl_f32_to_f32,
    decode_box_level_dfl_i16_to_f16, decode_box_level_dfl_i16_to_f32,
    decode_box_level_dfl_i8_to_f16, decode_box_level_dfl_i8_to_f32,
    decode_box_level_dfl_u16_to_f16, decode_box_level_dfl_u16_to_f32,
    decode_box_level_dfl_u8_to_f16, decode_box_level_dfl_u8_to_f32,
    decode_box_level_ltrb_f16_to_f16, decode_box_level_ltrb_f16_to_f32,
    decode_box_level_ltrb_f32_to_f16, decode_box_level_ltrb_f32_to_f32,
    decode_box_level_ltrb_i16_to_f16, decode_box_level_ltrb_i16_to_f32,
    decode_box_level_ltrb_i8_to_f16, decode_box_level_ltrb_i8_to_f32,
    decode_box_level_ltrb_u16_to_f16, decode_box_level_ltrb_u16_to_f32,
    decode_box_level_ltrb_u8_to_f16, decode_box_level_ltrb_u8_to_f32,
};
#[cfg(target_arch = "aarch64")]
use super::level_box::{dfl_f16_neon_fp16, dfl_neon, ltrb_neon, Exp};
use crate::per_scale::plan::LevelPlan;
use crate::Quantization;
use edgefirst_kernels::prim::dfl_lut::{decode_dfl_lut, DflExpTable};

/// The exp-table decode of one level's 8-bit bins on NEON: the 16-bin kernel, or the
/// any-`reg_max` one, both bit-identical to the portable kernel.
#[cfg(target_arch = "aarch64")]
fn lut_neon<
    T: edgefirst_kernels::prim::dfl_lut::NeonDflBin,
    O: edgefirst_kernels::prim::dfl_lut::BoxOut,
>(
    input: &[T],
    table: &DflExpTable,
    level: &LevelPlan,
    dst: &mut [O],
) {
    use edgefirst_kernels::prim::dfl_lut::{decode_dfl_lut_generic_neon, decode_dfl_lut_neon};
    if level.reg_max == 16 {
        decode_dfl_lut_neon(input, table, &level.dfl_geom(), dst);
    } else {
        decode_dfl_lut_generic_neon(input, table, &level.dfl_geom(), dst);
    }
}

impl BoxLevelDispatch {
    /// The max-subtracted softmax decode of an exp-table cell's 8-bit bins,
    /// for a scale the table cannot stand for (see [`DflExpTable::supports`]).
    fn run_softmax(
        &self,
        input: InputView<'_>,
        q: Quantization,
        level: &LevelPlan,
        dst: DstSliceMut<'_>,
    ) -> DecoderResult<()> {
        match (input, dst) {
            (InputView::I8(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_i8_to_f32(i, q, level, d)
            }
            (InputView::U8(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_u8_to_f32(i, q, level, d)
            }
            (InputView::I8(i), DstSliceMut::F16(d)) => {
                decode_box_level_dfl_i8_to_f16(i, q, level, d)
            }
            (InputView::U8(i), DstSliceMut::F16(d)) => {
                decode_box_level_dfl_u8_to_f16(i, q, level, d)
            }
            _ => {
                return Err(DecoderError::KernelDispatchUnreachable(format!(
                    "BoxLevelDispatch::run_softmax mismatched arms for {self:?}"
                )))
            }
        }
        Ok(())
    }

    /// Whether this cell decodes LTRB distances (four values per anchor); exhaustive, so a new
    /// variant must state its encoding.
    pub(crate) fn is_ltrb(&self) -> bool {
        use BoxLevelDispatch::*;
        match self {
            DflI8ToF32Lut | DflU8ToF32Lut | DflI16ToF32Scalar | DflU16ToF32Scalar
            | DflF16ToF32Scalar | DflF32ToF32Scalar | DflI8ToF16Lut | DflU8ToF16Lut
            | DflI16ToF16Scalar | DflU16ToF16Scalar | DflF16ToF16Scalar | DflF32ToF16Scalar => {
                false
            }
            LtrbI8ToF32Scalar | LtrbU8ToF32Scalar | LtrbI16ToF32Scalar | LtrbU16ToF32Scalar
            | LtrbF16ToF32Scalar | LtrbF32ToF32Scalar | LtrbI8ToF16Scalar | LtrbU8ToF16Scalar
            | LtrbI16ToF16Scalar | LtrbU16ToF16Scalar | LtrbF16ToF16Scalar | LtrbF32ToF16Scalar => {
                true
            }
            #[cfg(target_arch = "aarch64")]
            DflI8ToF32LutNeon | DflU8ToF32LutNeon | DflI8ToF16LutNeon | DflU8ToF16LutNeon
            | DflI16ToF32NeonBase | DflU16ToF32NeonBase | DflF16ToF32Neon | DflF32ToF32Neon
            | DflI16ToF16Neon | DflU16ToF16Neon | DflF16ToF16Neon | DflF32ToF16Neon
            | DflI16ToF32NeonFp16 | DflU16ToF32NeonFp16 | DflI16ToF16NeonFp16
            | DflU16ToF16NeonFp16 | DflF16ToF16NeonFp16 | DflF32ToF16NeonFp16 => false,
            #[cfg(target_arch = "aarch64")]
            LtrbI8ToF32NeonBase | LtrbU8ToF32NeonBase | LtrbI16ToF32NeonBase
            | LtrbU16ToF32NeonBase | LtrbF16ToF32Neon | LtrbF32ToF32Neon | LtrbI8ToF16Neon
            | LtrbU8ToF16Neon | LtrbI16ToF16Neon | LtrbU16ToF16Neon | LtrbF16ToF16Neon
            | LtrbF32ToF16Neon => true,
        }
    }

    /// Whether this cell weighs DFL bins through a [`DflExpTable`].
    pub(crate) fn uses_exp_table(&self) -> bool {
        use BoxLevelDispatch::*;
        #[cfg(target_arch = "aarch64")]
        if matches!(
            self,
            DflI8ToF32LutNeon | DflU8ToF32LutNeon | DflI8ToF16LutNeon | DflU8ToF16LutNeon
        ) {
            return true;
        }
        matches!(
            self,
            DflI8ToF32Lut | DflU8ToF32Lut | DflI8ToF16Lut | DflU8ToF16Lut
        )
    }

    /// Run this dispatch's concrete kernel.
    ///
    /// Returns `InvalidShape`, before writing anything, unless `input`, `dst` and the grids hold
    /// the values of `level.h · level.w` anchors, and `KernelDispatchUnreachable` if the variant
    /// tag and (input, dst) dtypes disagree, an internal logic bug at construction time. The
    /// `*Lut` cells weigh bins through `table` when it was built for `q.scale`, and build their
    /// own otherwise.
    pub(crate) fn run(
        &self,
        input: InputView<'_>,
        q: Quantization,
        level: &LevelPlan,
        table: Option<&DflExpTable>,
        dst: DstSliceMut<'_>,
    ) -> DecoderResult<()> {
        use BoxLevelDispatch::*;
        check_box_lengths(&input, &dst, self.is_ltrb(), level)?;
        if self.uses_exp_table() && !DflExpTable::supports(q.scale) {
            return self.run_softmax(input, q, level, dst);
        }
        let built = match table {
            Some(t) if t.is_for(q.scale) => None,
            _ => self.uses_exp_table().then(|| DflExpTable::new(q.scale)),
        };
        let table = built.as_ref().or(table);
        let table = || table.ok_or_else(|| DecoderError::Internal("DFL exp table missing".into()));
        match (self, input, dst) {
            // DFL × f32 output
            (DflI8ToF32Lut, InputView::I8(i), DstSliceMut::F32(d)) => {
                decode_dfl_lut(i, table()?, &level.dfl_geom(), d)
            }
            (DflU8ToF32Lut, InputView::U8(i), DstSliceMut::F32(d)) => {
                decode_dfl_lut(i, table()?, &level.dfl_geom(), d)
            }
            (DflI16ToF32Scalar, InputView::I16(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_i16_to_f32(i, q, level, d)
            }
            (DflU16ToF32Scalar, InputView::U16(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_u16_to_f32(i, q, level, d)
            }
            (DflF16ToF32Scalar, InputView::F16(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_f16_to_f32(i, q, level, d)
            }
            (DflF32ToF32Scalar, InputView::F32(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_f32_to_f32(i, q, level, d)
            }
            // DFL × f16 output
            (DflI8ToF16Lut, InputView::I8(i), DstSliceMut::F16(d)) => {
                decode_dfl_lut(i, table()?, &level.dfl_geom(), d)
            }
            (DflU8ToF16Lut, InputView::U8(i), DstSliceMut::F16(d)) => {
                decode_dfl_lut(i, table()?, &level.dfl_geom(), d)
            }
            (DflI16ToF16Scalar, InputView::I16(i), DstSliceMut::F16(d)) => {
                decode_box_level_dfl_i16_to_f16(i, q, level, d)
            }
            (DflU16ToF16Scalar, InputView::U16(i), DstSliceMut::F16(d)) => {
                decode_box_level_dfl_u16_to_f16(i, q, level, d)
            }
            (DflF16ToF16Scalar, InputView::F16(i), DstSliceMut::F16(d)) => {
                decode_box_level_dfl_f16_to_f16(i, q, level, d)
            }
            (DflF32ToF16Scalar, InputView::F32(i), DstSliceMut::F16(d)) => {
                decode_box_level_dfl_f32_to_f16(i, q, level, d)
            }
            // LTRB × f32 output
            (LtrbI8ToF32Scalar, InputView::I8(i), DstSliceMut::F32(d)) => {
                decode_box_level_ltrb_i8_to_f32(i, q, level, d)
            }
            (LtrbU8ToF32Scalar, InputView::U8(i), DstSliceMut::F32(d)) => {
                decode_box_level_ltrb_u8_to_f32(i, q, level, d)
            }
            (LtrbI16ToF32Scalar, InputView::I16(i), DstSliceMut::F32(d)) => {
                decode_box_level_ltrb_i16_to_f32(i, q, level, d)
            }
            (LtrbU16ToF32Scalar, InputView::U16(i), DstSliceMut::F32(d)) => {
                decode_box_level_ltrb_u16_to_f32(i, q, level, d)
            }
            (LtrbF16ToF32Scalar, InputView::F16(i), DstSliceMut::F32(d)) => {
                decode_box_level_ltrb_f16_to_f32(i, q, level, d)
            }
            (LtrbF32ToF32Scalar, InputView::F32(i), DstSliceMut::F32(d)) => {
                decode_box_level_ltrb_f32_to_f32(i, q, level, d)
            }
            // LTRB × f16 output
            (LtrbI8ToF16Scalar, InputView::I8(i), DstSliceMut::F16(d)) => {
                decode_box_level_ltrb_i8_to_f16(i, q, level, d)
            }
            (LtrbU8ToF16Scalar, InputView::U8(i), DstSliceMut::F16(d)) => {
                decode_box_level_ltrb_u8_to_f16(i, q, level, d)
            }
            (LtrbI16ToF16Scalar, InputView::I16(i), DstSliceMut::F16(d)) => {
                decode_box_level_ltrb_i16_to_f16(i, q, level, d)
            }
            (LtrbU16ToF16Scalar, InputView::U16(i), DstSliceMut::F16(d)) => {
                decode_box_level_ltrb_u16_to_f16(i, q, level, d)
            }
            (LtrbF16ToF16Scalar, InputView::F16(i), DstSliceMut::F16(d)) => {
                decode_box_level_ltrb_f16_to_f16(i, q, level, d)
            }
            (LtrbF32ToF16Scalar, InputView::F32(i), DstSliceMut::F16(d)) => {
                decode_box_level_ltrb_f32_to_f16(i, q, level, d)
            }

            #[cfg(target_arch = "aarch64")]
            (DflI8ToF32LutNeon, InputView::I8(i), DstSliceMut::F32(d)) => {
                lut_neon(i, table()?, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (DflU8ToF32LutNeon, InputView::U8(i), DstSliceMut::F32(d)) => {
                lut_neon(i, table()?, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (DflI8ToF16LutNeon, InputView::I8(i), DstSliceMut::F16(d)) => {
                lut_neon(i, table()?, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (DflU8ToF16LutNeon, InputView::U8(i), DstSliceMut::F16(d)) => {
                lut_neon(i, table()?, level, d)
            }
            // SAFETY (every NEON arm): `select` picks FP16 cells only when FEAT_FP16 was probed.
            #[cfg(target_arch = "aarch64")]
            (DflI16ToF32NeonBase, InputView::I16(i), DstSliceMut::F32(d)) => unsafe {
                dfl_neon(i, q, level, Exp::F32, d)
            },
            #[cfg(target_arch = "aarch64")]
            (DflU16ToF32NeonBase, InputView::U16(i), DstSliceMut::F32(d)) => unsafe {
                dfl_neon(i, q, level, Exp::F32, d)
            },
            #[cfg(target_arch = "aarch64")]
            (DflF16ToF32Neon, InputView::F16(i), DstSliceMut::F32(d)) => unsafe {
                dfl_neon(i, q, level, Exp::F32, d)
            },
            #[cfg(target_arch = "aarch64")]
            (DflF32ToF32Neon, InputView::F32(i), DstSliceMut::F32(d)) => unsafe {
                dfl_neon(i, q, level, Exp::F32, d)
            },
            #[cfg(target_arch = "aarch64")]
            (DflI16ToF16Neon, InputView::I16(i), DstSliceMut::F16(d)) => unsafe {
                dfl_neon(i, q, level, Exp::F32, d)
            },
            #[cfg(target_arch = "aarch64")]
            (DflU16ToF16Neon, InputView::U16(i), DstSliceMut::F16(d)) => unsafe {
                dfl_neon(i, q, level, Exp::F32, d)
            },
            #[cfg(target_arch = "aarch64")]
            (DflF16ToF16Neon, InputView::F16(i), DstSliceMut::F16(d)) => unsafe {
                dfl_neon(i, q, level, Exp::F32, d)
            },
            #[cfg(target_arch = "aarch64")]
            (DflF32ToF16Neon, InputView::F32(i), DstSliceMut::F16(d)) => unsafe {
                dfl_neon(i, q, level, Exp::F32, d)
            },
            #[cfg(target_arch = "aarch64")]
            (DflI16ToF32NeonFp16, InputView::I16(i), DstSliceMut::F32(d)) => unsafe {
                dfl_neon(i, q, level, Exp::Fp16, d)
            },
            #[cfg(target_arch = "aarch64")]
            (DflU16ToF32NeonFp16, InputView::U16(i), DstSliceMut::F32(d)) => unsafe {
                dfl_neon(i, q, level, Exp::Fp16, d)
            },
            #[cfg(target_arch = "aarch64")]
            (DflI16ToF16NeonFp16, InputView::I16(i), DstSliceMut::F16(d)) => unsafe {
                dfl_f16_neon_fp16(i, q, level, d)
            },
            #[cfg(target_arch = "aarch64")]
            (DflU16ToF16NeonFp16, InputView::U16(i), DstSliceMut::F16(d)) => unsafe {
                dfl_f16_neon_fp16(i, q, level, d)
            },
            #[cfg(target_arch = "aarch64")]
            (DflF16ToF16NeonFp16, InputView::F16(i), DstSliceMut::F16(d)) => unsafe {
                dfl_f16_neon_fp16(i, q, level, d)
            },
            #[cfg(target_arch = "aarch64")]
            (DflF32ToF16NeonFp16, InputView::F32(i), DstSliceMut::F16(d)) => unsafe {
                dfl_f16_neon_fp16(i, q, level, d)
            },
            #[cfg(target_arch = "aarch64")]
            (LtrbI8ToF32NeonBase, InputView::I8(i), DstSliceMut::F32(d)) => {
                ltrb_neon(i, q, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (LtrbU8ToF32NeonBase, InputView::U8(i), DstSliceMut::F32(d)) => {
                ltrb_neon(i, q, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (LtrbI16ToF32NeonBase, InputView::I16(i), DstSliceMut::F32(d)) => {
                ltrb_neon(i, q, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (LtrbU16ToF32NeonBase, InputView::U16(i), DstSliceMut::F32(d)) => {
                ltrb_neon(i, q, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (LtrbF16ToF32Neon, InputView::F16(i), DstSliceMut::F32(d)) => ltrb_neon(i, q, level, d),
            #[cfg(target_arch = "aarch64")]
            (LtrbF32ToF32Neon, InputView::F32(i), DstSliceMut::F32(d)) => ltrb_neon(i, q, level, d),
            #[cfg(target_arch = "aarch64")]
            (LtrbI8ToF16Neon, InputView::I8(i), DstSliceMut::F16(d)) => ltrb_neon(i, q, level, d),
            #[cfg(target_arch = "aarch64")]
            (LtrbU8ToF16Neon, InputView::U8(i), DstSliceMut::F16(d)) => ltrb_neon(i, q, level, d),
            #[cfg(target_arch = "aarch64")]
            (LtrbI16ToF16Neon, InputView::I16(i), DstSliceMut::F16(d)) => ltrb_neon(i, q, level, d),
            #[cfg(target_arch = "aarch64")]
            (LtrbU16ToF16Neon, InputView::U16(i), DstSliceMut::F16(d)) => ltrb_neon(i, q, level, d),
            #[cfg(target_arch = "aarch64")]
            (LtrbF16ToF16Neon, InputView::F16(i), DstSliceMut::F16(d)) => ltrb_neon(i, q, level, d),
            #[cfg(target_arch = "aarch64")]
            (LtrbF32ToF16Neon, InputView::F32(i), DstSliceMut::F16(d)) => ltrb_neon(i, q, level, d),

            (variant, _, _) => {
                return Err(DecoderError::KernelDispatchUnreachable(format!(
                    "BoxLevelDispatch::run mismatched arms for {variant:?}"
                )))
            }
        }
        Ok(())
    }
}

// ── ScoreLevelDispatch::run ─────────────────────────────────────────────

use super::level_score::{
    decode_score_level_f16_to_f16, decode_score_level_f16_to_f32, decode_score_level_f32_to_f16,
    decode_score_level_f32_to_f32, decode_score_level_i16_to_f16, decode_score_level_i16_to_f32,
    decode_score_level_i8_to_f16, decode_score_level_i8_to_f32, decode_score_level_u16_to_f16,
    decode_score_level_u16_to_f32, decode_score_level_u8_to_f16, decode_score_level_u8_to_f32,
};
#[cfg(target_arch = "aarch64")]
use super::level_score::{
    decode_score_level_float_to_f32_neon, decode_score_level_to_f16_neon,
    decode_score_level_to_f16_neon_fp16,
};
#[cfg(target_arch = "aarch64")]
use super::level_score::{
    decode_score_level_i16_to_f32_neon, decode_score_level_i16_to_f32_neon_fused,
    decode_score_level_i8_to_f32_neon, decode_score_level_i8_to_f32_neon_fused,
    decode_score_level_u16_to_f32_neon, decode_score_level_u16_to_f32_neon_fused,
    decode_score_level_u8_to_f32_neon, decode_score_level_u8_to_f32_neon_fused,
};
use crate::per_scale::Activation;

impl ScoreLevelDispatch {
    #[allow(dead_code)] // Wired in Task 21.
    pub(crate) fn run(
        &self,
        input: InputView<'_>,
        q: Quantization,
        num_classes: usize,
        level: &LevelPlan,
        activation: Activation,
        dst: DstSliceMut<'_>,
    ) -> DecoderResult<()> {
        use ScoreLevelDispatch::*;
        match (self, input, dst) {
            (I8ToF32Scalar, InputView::I8(i), DstSliceMut::F32(d)) => {
                decode_score_level_i8_to_f32(i, q, num_classes, level, activation, d)
            }
            (U8ToF32Scalar, InputView::U8(i), DstSliceMut::F32(d)) => {
                decode_score_level_u8_to_f32(i, q, num_classes, level, activation, d)
            }
            (I16ToF32Scalar, InputView::I16(i), DstSliceMut::F32(d)) => {
                decode_score_level_i16_to_f32(i, q, num_classes, level, activation, d)
            }
            (U16ToF32Scalar, InputView::U16(i), DstSliceMut::F32(d)) => {
                decode_score_level_u16_to_f32(i, q, num_classes, level, activation, d)
            }
            (F16ToF32Scalar, InputView::F16(i), DstSliceMut::F32(d)) => {
                decode_score_level_f16_to_f32(i, q, num_classes, level, activation, d)
            }
            (F32ToF32Scalar, InputView::F32(i), DstSliceMut::F32(d)) => {
                decode_score_level_f32_to_f32(i, q, num_classes, level, activation, d)
            }
            (I8ToF16Scalar, InputView::I8(i), DstSliceMut::F16(d)) => {
                decode_score_level_i8_to_f16(i, q, num_classes, level, activation, d)
            }
            (U8ToF16Scalar, InputView::U8(i), DstSliceMut::F16(d)) => {
                decode_score_level_u8_to_f16(i, q, num_classes, level, activation, d)
            }
            (I16ToF16Scalar, InputView::I16(i), DstSliceMut::F16(d)) => {
                decode_score_level_i16_to_f16(i, q, num_classes, level, activation, d)
            }
            (U16ToF16Scalar, InputView::U16(i), DstSliceMut::F16(d)) => {
                decode_score_level_u16_to_f16(i, q, num_classes, level, activation, d)
            }
            (F16ToF16Scalar, InputView::F16(i), DstSliceMut::F16(d)) => {
                decode_score_level_f16_to_f16(i, q, num_classes, level, activation, d)
            }
            (F32ToF16Scalar, InputView::F32(i), DstSliceMut::F16(d)) => {
                decode_score_level_f32_to_f16(i, q, num_classes, level, activation, d)
            }
            // Tier-1 NEON-base arms. Reachable only when select() picked
            // the *NeonBase variant, which only happens with
            // CpuFeatures::neon_baseline true on aarch64.
            #[cfg(target_arch = "aarch64")]
            (I8ToF32NeonBase, InputView::I8(i), DstSliceMut::F32(d)) => {
                decode_score_level_i8_to_f32_neon(i, q, num_classes, level, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (U8ToF32NeonBase, InputView::U8(i), DstSliceMut::F32(d)) => {
                decode_score_level_u8_to_f32_neon(i, q, num_classes, level, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (I16ToF32NeonBase, InputView::I16(i), DstSliceMut::F32(d)) => {
                decode_score_level_i16_to_f32_neon(i, q, num_classes, level, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (U16ToF32NeonBase, InputView::U16(i), DstSliceMut::F32(d)) => {
                decode_score_level_u16_to_f32_neon(i, q, num_classes, level, activation, d)
            }

            // Tier-2 NEON FP16 score arms.

            // Tier 1-fused: fused dequant+sigmoid NEON arms.
            #[cfg(target_arch = "aarch64")]
            (I8ToF32NeonFused, InputView::I8(i), DstSliceMut::F32(d)) => {
                decode_score_level_i8_to_f32_neon_fused(i, q, num_classes, level, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (U8ToF32NeonFused, InputView::U8(i), DstSliceMut::F32(d)) => {
                decode_score_level_u8_to_f32_neon_fused(i, q, num_classes, level, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (I16ToF32NeonFused, InputView::I16(i), DstSliceMut::F32(d)) => {
                decode_score_level_i16_to_f32_neon_fused(i, q, num_classes, level, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (U16ToF32NeonFused, InputView::U16(i), DstSliceMut::F32(d)) => {
                decode_score_level_u16_to_f32_neon_fused(i, q, num_classes, level, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (F16ToF32Neon, InputView::F16(i), DstSliceMut::F32(d)) => {
                decode_score_level_float_to_f32_neon(i, q, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (F32ToF32Neon, InputView::F32(i), DstSliceMut::F32(d)) => {
                decode_score_level_float_to_f32_neon(i, q, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (I8ToF16Neon, InputView::I8(i), DstSliceMut::F16(d)) => {
                decode_score_level_to_f16_neon(i, q, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (U8ToF16Neon, InputView::U8(i), DstSliceMut::F16(d)) => {
                decode_score_level_to_f16_neon(i, q, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (I16ToF16Neon, InputView::I16(i), DstSliceMut::F16(d)) => {
                decode_score_level_to_f16_neon(i, q, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (U16ToF16Neon, InputView::U16(i), DstSliceMut::F16(d)) => {
                decode_score_level_to_f16_neon(i, q, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (F16ToF16Neon, InputView::F16(i), DstSliceMut::F16(d)) => {
                decode_score_level_to_f16_neon(i, q, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (F32ToF16Neon, InputView::F32(i), DstSliceMut::F16(d)) => {
                decode_score_level_to_f16_neon(i, q, activation, d)
            }
            // SAFETY (every NeonFp16 arm): `select` picks these only when FEAT_FP16 was probed.
            #[cfg(target_arch = "aarch64")]
            (I8ToF16NeonFp16, InputView::I8(i), DstSliceMut::F16(d)) => unsafe {
                decode_score_level_to_f16_neon_fp16(i, q, activation, d)
            },
            #[cfg(target_arch = "aarch64")]
            (U8ToF16NeonFp16, InputView::U8(i), DstSliceMut::F16(d)) => unsafe {
                decode_score_level_to_f16_neon_fp16(i, q, activation, d)
            },
            #[cfg(target_arch = "aarch64")]
            (I16ToF16NeonFp16, InputView::I16(i), DstSliceMut::F16(d)) => unsafe {
                decode_score_level_to_f16_neon_fp16(i, q, activation, d)
            },
            #[cfg(target_arch = "aarch64")]
            (U16ToF16NeonFp16, InputView::U16(i), DstSliceMut::F16(d)) => unsafe {
                decode_score_level_to_f16_neon_fp16(i, q, activation, d)
            },
            #[cfg(target_arch = "aarch64")]
            (F16ToF16NeonFp16, InputView::F16(i), DstSliceMut::F16(d)) => unsafe {
                decode_score_level_to_f16_neon_fp16(i, q, activation, d)
            },
            #[cfg(target_arch = "aarch64")]
            (F32ToF16NeonFp16, InputView::F32(i), DstSliceMut::F16(d)) => unsafe {
                decode_score_level_to_f16_neon_fp16(i, q, activation, d)
            },
            (variant, _, _) => {
                return Err(DecoderError::KernelDispatchUnreachable(format!(
                    "ScoreLevelDispatch::run mismatched arms for {variant:?}"
                )))
            }
        }
        Ok(())
    }
}

// ── MaskCoefDispatch::run / ProtoDispatch::run ──────────────────────────

#[cfg(target_arch = "aarch64")]
use super::level_mc::{decode_mc_float_to_f32_neon, decode_mc_to_f16_neon};
use super::level_mc::{
    decode_mc_level_f16_to_f16, decode_mc_level_f16_to_f32, decode_mc_level_f32_to_f16,
    decode_mc_level_f32_to_f32, decode_mc_level_i16_to_f16, decode_mc_level_i16_to_f32,
    decode_mc_level_i8_to_f16, decode_mc_level_i8_to_f32, decode_mc_level_u16_to_f16,
    decode_mc_level_u16_to_f32, decode_mc_level_u8_to_f16, decode_mc_level_u8_to_f32,
    decode_proto_f16_to_f16, decode_proto_f16_to_f32, decode_proto_f32_to_f16,
    decode_proto_f32_to_f32, decode_proto_i16_to_f16, decode_proto_i16_to_f32,
    decode_proto_i8_to_f16, decode_proto_i8_to_f32, decode_proto_u16_to_f16,
    decode_proto_u16_to_f32, decode_proto_u8_to_f16, decode_proto_u8_to_f32,
};
#[cfg(target_arch = "aarch64")]
use super::level_mc::{
    decode_mc_level_i16_to_f32_neon, decode_mc_level_i8_to_f32_neon,
    decode_mc_level_u16_to_f32_neon, decode_mc_level_u8_to_f32_neon, decode_proto_i16_to_f32_neon,
    decode_proto_i8_to_f32_neon, decode_proto_u16_to_f32_neon, decode_proto_u8_to_f32_neon,
};

impl MaskCoefDispatch {
    #[allow(dead_code)] // Wired in Task 21.
    pub(crate) fn run(
        &self,
        input: InputView<'_>,
        q: Quantization,
        num_mc: usize,
        level: &LevelPlan,
        dst: DstSliceMut<'_>,
    ) -> DecoderResult<()> {
        use ScoreLevelDispatch::*;
        match (self.0, input, dst) {
            (I8ToF32Scalar, InputView::I8(i), DstSliceMut::F32(d)) => {
                decode_mc_level_i8_to_f32(i, q, num_mc, level, d)
            }
            (U8ToF32Scalar, InputView::U8(i), DstSliceMut::F32(d)) => {
                decode_mc_level_u8_to_f32(i, q, num_mc, level, d)
            }
            (I16ToF32Scalar, InputView::I16(i), DstSliceMut::F32(d)) => {
                decode_mc_level_i16_to_f32(i, q, num_mc, level, d)
            }
            (U16ToF32Scalar, InputView::U16(i), DstSliceMut::F32(d)) => {
                decode_mc_level_u16_to_f32(i, q, num_mc, level, d)
            }
            (F16ToF32Scalar, InputView::F16(i), DstSliceMut::F32(d)) => {
                decode_mc_level_f16_to_f32(i, q, num_mc, level, d)
            }
            (F32ToF32Scalar, InputView::F32(i), DstSliceMut::F32(d)) => {
                decode_mc_level_f32_to_f32(i, q, num_mc, level, d)
            }
            (I8ToF16Scalar, InputView::I8(i), DstSliceMut::F16(d)) => {
                decode_mc_level_i8_to_f16(i, q, num_mc, level, d)
            }
            (U8ToF16Scalar, InputView::U8(i), DstSliceMut::F16(d)) => {
                decode_mc_level_u8_to_f16(i, q, num_mc, level, d)
            }
            (I16ToF16Scalar, InputView::I16(i), DstSliceMut::F16(d)) => {
                decode_mc_level_i16_to_f16(i, q, num_mc, level, d)
            }
            (U16ToF16Scalar, InputView::U16(i), DstSliceMut::F16(d)) => {
                decode_mc_level_u16_to_f16(i, q, num_mc, level, d)
            }
            (F16ToF16Scalar, InputView::F16(i), DstSliceMut::F16(d)) => {
                decode_mc_level_f16_to_f16(i, q, num_mc, level, d)
            }
            (F32ToF16Scalar, InputView::F32(i), DstSliceMut::F16(d)) => {
                decode_mc_level_f32_to_f16(i, q, num_mc, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (I8ToF32NeonBase, InputView::I8(i), DstSliceMut::F32(d)) => {
                decode_mc_level_i8_to_f32_neon(i, q, num_mc, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (U8ToF32NeonBase, InputView::U8(i), DstSliceMut::F32(d)) => {
                decode_mc_level_u8_to_f32_neon(i, q, num_mc, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (I16ToF32NeonBase, InputView::I16(i), DstSliceMut::F32(d)) => {
                decode_mc_level_i16_to_f32_neon(i, q, num_mc, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (U16ToF32NeonBase, InputView::U16(i), DstSliceMut::F32(d)) => {
                decode_mc_level_u16_to_f32_neon(i, q, num_mc, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (F16ToF32Neon, InputView::F16(i), DstSliceMut::F32(d)) => {
                decode_mc_float_to_f32_neon(i, q, d)
            }
            #[cfg(target_arch = "aarch64")]
            (F32ToF32Neon, InputView::F32(i), DstSliceMut::F32(d)) => {
                decode_mc_float_to_f32_neon(i, q, d)
            }
            #[cfg(target_arch = "aarch64")]
            (I8ToF16Neon, InputView::I8(i), DstSliceMut::F16(d)) => decode_mc_to_f16_neon(i, q, d),
            #[cfg(target_arch = "aarch64")]
            (U8ToF16Neon, InputView::U8(i), DstSliceMut::F16(d)) => decode_mc_to_f16_neon(i, q, d),
            #[cfg(target_arch = "aarch64")]
            (I16ToF16Neon, InputView::I16(i), DstSliceMut::F16(d)) => {
                decode_mc_to_f16_neon(i, q, d)
            }
            #[cfg(target_arch = "aarch64")]
            (U16ToF16Neon, InputView::U16(i), DstSliceMut::F16(d)) => {
                decode_mc_to_f16_neon(i, q, d)
            }
            #[cfg(target_arch = "aarch64")]
            (F16ToF16Neon, InputView::F16(i), DstSliceMut::F16(d)) => {
                decode_mc_to_f16_neon(i, q, d)
            }
            #[cfg(target_arch = "aarch64")]
            (F32ToF16Neon, InputView::F32(i), DstSliceMut::F16(d)) => {
                decode_mc_to_f16_neon(i, q, d)
            }
            (variant, _, _) => {
                return Err(DecoderError::KernelDispatchUnreachable(format!(
                    "MaskCoefDispatch::run mismatched arms for {variant:?}"
                )))
            }
        }
        Ok(())
    }
}

impl ProtoDispatch {
    /// Single-tensor dequant — no per-level structure. Takes flat input/output slices.
    #[allow(dead_code)] // Wired in Task 21.
    pub(crate) fn run(
        &self,
        input: InputView<'_>,
        q: Quantization,
        dst: DstSliceMut<'_>,
    ) -> DecoderResult<()> {
        use ScoreLevelDispatch::*;
        match (self.0, input, dst) {
            (I8ToF32Scalar, InputView::I8(i), DstSliceMut::F32(d)) => {
                decode_proto_i8_to_f32(i, q, d)
            }
            (U8ToF32Scalar, InputView::U8(i), DstSliceMut::F32(d)) => {
                decode_proto_u8_to_f32(i, q, d)
            }
            (I16ToF32Scalar, InputView::I16(i), DstSliceMut::F32(d)) => {
                decode_proto_i16_to_f32(i, q, d)
            }
            (U16ToF32Scalar, InputView::U16(i), DstSliceMut::F32(d)) => {
                decode_proto_u16_to_f32(i, q, d)
            }
            (F16ToF32Scalar, InputView::F16(i), DstSliceMut::F32(d)) => {
                decode_proto_f16_to_f32(i, q, d)
            }
            (F32ToF32Scalar, InputView::F32(i), DstSliceMut::F32(d)) => {
                decode_proto_f32_to_f32(i, q, d)
            }
            (I8ToF16Scalar, InputView::I8(i), DstSliceMut::F16(d)) => {
                decode_proto_i8_to_f16(i, q, d)
            }
            (U8ToF16Scalar, InputView::U8(i), DstSliceMut::F16(d)) => {
                decode_proto_u8_to_f16(i, q, d)
            }
            (I16ToF16Scalar, InputView::I16(i), DstSliceMut::F16(d)) => {
                decode_proto_i16_to_f16(i, q, d)
            }
            (U16ToF16Scalar, InputView::U16(i), DstSliceMut::F16(d)) => {
                decode_proto_u16_to_f16(i, q, d)
            }
            (F16ToF16Scalar, InputView::F16(i), DstSliceMut::F16(d)) => {
                decode_proto_f16_to_f16(i, q, d)
            }
            (F32ToF16Scalar, InputView::F32(i), DstSliceMut::F16(d)) => {
                decode_proto_f32_to_f16(i, q, d)
            }
            #[cfg(target_arch = "aarch64")]
            (I8ToF32NeonBase, InputView::I8(i), DstSliceMut::F32(d)) => {
                decode_proto_i8_to_f32_neon(i, q, d)
            }
            #[cfg(target_arch = "aarch64")]
            (U8ToF32NeonBase, InputView::U8(i), DstSliceMut::F32(d)) => {
                decode_proto_u8_to_f32_neon(i, q, d)
            }
            #[cfg(target_arch = "aarch64")]
            (I16ToF32NeonBase, InputView::I16(i), DstSliceMut::F32(d)) => {
                decode_proto_i16_to_f32_neon(i, q, d)
            }
            #[cfg(target_arch = "aarch64")]
            (U16ToF32NeonBase, InputView::U16(i), DstSliceMut::F32(d)) => {
                decode_proto_u16_to_f32_neon(i, q, d)
            }
            #[cfg(target_arch = "aarch64")]
            (F16ToF32Neon, InputView::F16(i), DstSliceMut::F32(d)) => {
                decode_mc_float_to_f32_neon(i, q, d)
            }
            #[cfg(target_arch = "aarch64")]
            (F32ToF32Neon, InputView::F32(i), DstSliceMut::F32(d)) => {
                decode_mc_float_to_f32_neon(i, q, d)
            }
            #[cfg(target_arch = "aarch64")]
            (I8ToF16Neon, InputView::I8(i), DstSliceMut::F16(d)) => decode_mc_to_f16_neon(i, q, d),
            #[cfg(target_arch = "aarch64")]
            (U8ToF16Neon, InputView::U8(i), DstSliceMut::F16(d)) => decode_mc_to_f16_neon(i, q, d),
            #[cfg(target_arch = "aarch64")]
            (I16ToF16Neon, InputView::I16(i), DstSliceMut::F16(d)) => {
                decode_mc_to_f16_neon(i, q, d)
            }
            #[cfg(target_arch = "aarch64")]
            (U16ToF16Neon, InputView::U16(i), DstSliceMut::F16(d)) => {
                decode_mc_to_f16_neon(i, q, d)
            }
            #[cfg(target_arch = "aarch64")]
            (F16ToF16Neon, InputView::F16(i), DstSliceMut::F16(d)) => {
                decode_mc_to_f16_neon(i, q, d)
            }
            #[cfg(target_arch = "aarch64")]
            (F32ToF16Neon, InputView::F32(i), DstSliceMut::F16(d)) => {
                decode_mc_to_f16_neon(i, q, d)
            }
            (variant, _, _) => {
                return Err(DecoderError::KernelDispatchUnreachable(format!(
                    "ProtoDispatch::run mismatched arms for {variant:?}"
                )))
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::CpuFeatures;
    use super::*;
    use crate::per_scale::DecodeDtype;
    use crate::schema::BoxEncoding;
    use edgefirst_tensor::DType;

    /// Every selectable box cell reports LTRB exactly for the direct encoding, on every tier.
    #[test]
    fn is_ltrb_follows_the_encoding_on_every_tier() {
        let inputs = [
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::F16,
            DType::F32,
        ];
        for tier in ["scalar", "neon", "neon_fp16", "neon_dotprod"] {
            let Ok(f) = super::super::decoder_cpu_features_from(Some(tier)) else {
                continue;
            };
            for input in inputs {
                for output in [DecodeDtype::F32, DecodeDtype::F16] {
                    for enc in [BoxEncoding::Dfl, BoxEncoding::Direct] {
                        let d = BoxLevelDispatch::select(enc, input, output, &f).unwrap();
                        let want = enc == BoxEncoding::Direct;
                        assert_eq!(d.is_ltrb(), want, "{tier} {enc:?} {input:?}: {d:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn box_dispatch_select_dfl_i8_to_f32_returns_scalar_when_no_simd() {
        let f = CpuFeatures::default();
        let d =
            BoxLevelDispatch::select(BoxEncoding::Dfl, DType::I8, DecodeDtype::F32, &f).unwrap();
        assert!(matches!(d, BoxLevelDispatch::DflI8ToF32Lut));
        assert!(d.uses_exp_table());
        let d =
            BoxLevelDispatch::select(BoxEncoding::Dfl, DType::I16, DecodeDtype::F32, &f).unwrap();
        assert!(!d.uses_exp_table());
    }

    /// A scale the exp table cannot represent (negative, zero, NaN) decodes
    /// through the max-subtracted softmax instead, finite and equal to it.
    #[test]
    fn box_dispatch_lut_falls_back_for_unusable_scales() {
        use crate::per_scale::kernels::level_box::{
            decode_box_level_dfl_i8_to_f16, decode_box_level_dfl_i8_to_f32,
        };
        let (gx, gy) = edgefirst_kernels::prim::grids::make_anchor_grid(1, 1);
        let level = LevelPlan {
            stride: 8.0,
            h: 1,
            w: 1,
            reg_max: 16,
            anchor_offset: 0,
            grid_x: gx,
            grid_y: gy,
            box_shape: vec![1, 1, 1, 64].into_boxed_slice(),
            score_shape: vec![1, 1, 1, 1].into_boxed_slice(),
            mc_shape: None,
            layout: crate::per_scale::plan::Layout::Nhwc,
        };
        // Per side: bin 0 at 127, then fifteen bins at -128.
        let bins: Vec<i8> = (0..64)
            .map(|i| if i % 16 == 0 { 127 } else { -128 })
            .collect();
        let mut tiers = vec![CpuFeatures::default()];
        if let Ok(f) = crate::per_scale::kernels::decoder_cpu_features() {
            tiers.push(f);
        }
        for scale in [-0.5_f32, 0.0, f32::NAN, f32::NEG_INFINITY] {
            let q = Quantization::new(scale, 0);
            let mut want = [0.0_f32; 4];
            decode_box_level_dfl_i8_to_f32(&bins, q, &level, &mut want);
            let mut want16 = [half::f16::ZERO; 4];
            decode_box_level_dfl_i8_to_f16(&bins, q, &level, &mut want16);
            for f in &tiers {
                let d = BoxLevelDispatch::select(BoxEncoding::Dfl, DType::I8, DecodeDtype::F32, f)
                    .unwrap();
                let mut got = [0.0_f32; 4];
                d.run(
                    InputView::I8(&bins),
                    q,
                    &level,
                    None,
                    DstSliceMut::F32(&mut got),
                )
                .unwrap();
                if scale.is_finite() {
                    assert!(
                        got.iter().all(|v| v.is_finite()),
                        "{d:?} scale {scale}: {got:?}"
                    );
                    for (g, w) in got.iter().zip(want) {
                        assert!(
                            (g - w).abs() < 1e-3,
                            "{d:?} scale {scale}: {got:?} vs {want:?}"
                        );
                    }
                } else {
                    assert_eq!(
                        got.map(f32::to_bits),
                        want.map(f32::to_bits),
                        "{d:?} scale {scale}"
                    );
                }
                let d = BoxLevelDispatch::select(BoxEncoding::Dfl, DType::I8, DecodeDtype::F16, f)
                    .unwrap();
                let mut got16 = [half::f16::ZERO; 4];
                d.run(
                    InputView::I8(&bins),
                    q,
                    &level,
                    None,
                    DstSliceMut::F16(&mut got16),
                )
                .unwrap();
                assert_eq!(
                    got16.map(half::f16::to_bits),
                    want16.map(half::f16::to_bits),
                    "{d:?} scale {scale}"
                );
            }
        }
        // The reviewer's case: the softmax puts every side at distance 8.
        let q = Quantization::new(-0.5, 0);
        let mut got = [0.0_f32; 4];
        BoxLevelDispatch::select(
            BoxEncoding::Dfl,
            DType::I8,
            DecodeDtype::F32,
            &CpuFeatures::default(),
        )
        .unwrap()
        .run(
            InputView::I8(&bins),
            q,
            &level,
            None,
            DstSliceMut::F32(&mut got),
        )
        .unwrap();
        assert!(got.iter().all(|v| v.is_finite()), "{got:?}");
    }

    /// A table built for another scale is not used: the kernel builds one
    /// for the scale it is given.
    #[test]
    fn box_dispatch_lut_ignores_a_stale_table() {
        let f = CpuFeatures::default();
        let d =
            BoxLevelDispatch::select(BoxEncoding::Dfl, DType::U8, DecodeDtype::F32, &f).unwrap();
        let (gx, gy) = edgefirst_kernels::prim::grids::make_anchor_grid(1, 1);
        let level = LevelPlan {
            stride: 8.0,
            h: 1,
            w: 1,
            reg_max: 16,
            anchor_offset: 0,
            grid_x: gx,
            grid_y: gy,
            box_shape: vec![1, 1, 1, 64].into_boxed_slice(),
            score_shape: vec![1, 1, 1, 1].into_boxed_slice(),
            mc_shape: None,
            layout: crate::per_scale::plan::Layout::Nhwc,
        };
        let bins: Vec<u8> = (0..64).map(|i| (i * 37 % 251) as u8).collect();
        let q = Quantization::new(0.2, 0);
        let run = |table: Option<&DflExpTable>| {
            let mut out = [0.0_f32; 4];
            d.run(
                InputView::U8(&bins),
                q,
                &level,
                table,
                DstSliceMut::F32(&mut out),
            )
            .unwrap();
            out
        };
        let fresh = run(None);
        assert_eq!(run(Some(&DflExpTable::new(0.2))), fresh);
        assert_eq!(run(Some(&DflExpTable::new(0.7))), fresh);
    }

    #[test]
    fn box_dispatch_select_ltrb_i8_to_f16_returns_scalar() {
        let f = CpuFeatures::default();
        let d =
            BoxLevelDispatch::select(BoxEncoding::Direct, DType::I8, DecodeDtype::F16, &f).unwrap();
        assert!(matches!(d, BoxLevelDispatch::LtrbI8ToF16Scalar));
    }

    #[test]
    fn box_dispatch_unsupported_dtype_errors() {
        let f = CpuFeatures::default();
        let r = BoxLevelDispatch::select(BoxEncoding::Dfl, DType::I32, DecodeDtype::F32, &f);
        assert!(r.is_err());
    }

    #[test]
    fn score_dispatch_supports_float_passthrough() {
        let f = CpuFeatures::default();
        let d = ScoreLevelDispatch::select(DType::F32, DecodeDtype::F32, &f).unwrap();
        assert!(matches!(d, ScoreLevelDispatch::F32ToF32Scalar));
    }

    #[test]
    fn score_dispatch_all_phase1_cells() {
        let f = CpuFeatures::default();
        let inputs = [
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::F16,
            DType::F32,
        ];
        for &i in &inputs {
            for &o in &[DecodeDtype::F32, DecodeDtype::F16] {
                ScoreLevelDispatch::select(i, o, &f)
                    .unwrap_or_else(|e| panic!("({i:?},{o:?}): {e:?}"));
            }
        }
    }

    #[test]
    fn mc_dispatch_select_works() {
        let f = CpuFeatures::default();
        let d = MaskCoefDispatch::select(DType::I8, DecodeDtype::F32, &f).unwrap();
        assert!(matches!(d.0, ScoreLevelDispatch::I8ToF32Scalar));
    }

    #[test]
    fn proto_dispatch_select_works() {
        let f = CpuFeatures::default();
        let d = ProtoDispatch::select(DType::I8, DecodeDtype::F16, &f).unwrap();
        assert!(matches!(d.0, ScoreLevelDispatch::I8ToF16Scalar));
    }

    #[test]
    fn box_dispatch_run_dfl_one_hot_via_dispatch_enum() {
        use crate::per_scale::plan::LevelPlan;
        use crate::Quantization;
        use edgefirst_kernels::prim::grids::make_anchor_grid;

        // Same fixture as the level_box test but routed through dispatch.
        let mut logits = [0.0_f32; 64];
        for side in 0..4 {
            logits[side * 16 + 5] = 100.0;
        }
        let (gx, gy) = make_anchor_grid(1, 1);
        let level = LevelPlan {
            stride: 8.0,
            h: 1,
            w: 1,
            reg_max: 16,
            anchor_offset: 0,
            grid_x: gx,
            grid_y: gy,
            box_shape: vec![1, 1, 1, 64].into_boxed_slice(),
            score_shape: vec![1, 1, 1, 80].into_boxed_slice(),
            mc_shape: None,
            layout: crate::per_scale::plan::Layout::Nhwc,
        };
        let mut out = [0.0_f32; 4];
        let dispatch = BoxLevelDispatch::DflF32ToF32Scalar;
        dispatch
            .run(
                InputView::F32(&logits),
                Quantization::identity(),
                &level,
                None,
                DstSliceMut::F32(&mut out),
            )
            .unwrap();
        assert!((out[0] - 4.0).abs() < 1e-3);
        assert!((out[2] - 80.0).abs() < 1e-3);
    }
}
