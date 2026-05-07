// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Plan-time selected kernel dispatches.
//!
//! Phase 1 only implements scalar variants. Phase 2 adds NEON tier
//! variants without changing this enum's shape — `select()` adds
//! higher-tier branches and falls through to scalar.

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
/// Phase 1 only emits `*Scalar` variants. Phase 2 adds NEON tiers.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // Wired by Task 15.
#[allow(clippy::enum_variant_names)] // tier suffix is meaningful (Phase 2 adds Neon variants)
pub(crate) enum BoxLevelDispatch {
    // ── DFL encoding (yolov8, yolo11) ────────────────────
    DflI8ToF32Scalar,
    DflU8ToF32Scalar,
    DflI16ToF32Scalar,
    DflU16ToF32Scalar,
    DflF16ToF32Scalar,
    DflF32ToF32Scalar,
    DflI8ToF16Scalar,
    DflU8ToF16Scalar,
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

    // Tier 1 NEON-baseline (aarch64 only). Selected when
    // CpuFeatures::neon_baseline is true. f16 outputs and float inputs
    // fall through to scalar in this milestone; Phase 2-B adds them.
    #[cfg(target_arch = "aarch64")]
    DflI8ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    DflU8ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    DflI16ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    DflU16ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    LtrbI8ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    LtrbU8ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    LtrbI16ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    LtrbU16ToF32NeonBase,

    // Tier 2 NEON FP16 (Cortex-A55+). Selected when CpuFeatures::neon_fp16
    // is true. Same dequant + dist2bbox path as NeonBase but DFL softmax
    // runs in 8-lane f16. LTRB has no softmax → no FP16 variant needed.
    #[cfg(target_arch = "aarch64")]
    DflI8ToF32NeonFp16,
    #[cfg(target_arch = "aarch64")]
    DflU8ToF32NeonFp16,
    #[cfg(target_arch = "aarch64")]
    DflI16ToF32NeonFp16,
    #[cfg(target_arch = "aarch64")]
    DflU16ToF32NeonFp16,
}

impl BoxLevelDispatch {
    /// Highest available tier wins. Phase 1: always scalar.
    pub(crate) fn select(
        encoding: BoxEncoding,
        input: DType,
        output: DecodeDtype,
        features: &CpuFeatures,
    ) -> DecoderResult<Self> {
        use BoxLevelDispatch::*;

        #[cfg(target_arch = "aarch64")]
        if features.neon_fp16 {
            // FP16 tier wins for DFL only — LTRB has no softmax.
            match (encoding, input, output) {
                (BoxEncoding::Dfl, DType::I8, DecodeDtype::F32) => return Ok(DflI8ToF32NeonFp16),
                (BoxEncoding::Dfl, DType::U8, DecodeDtype::F32) => return Ok(DflU8ToF32NeonFp16),
                (BoxEncoding::Dfl, DType::I16, DecodeDtype::F32) => return Ok(DflI16ToF32NeonFp16),
                (BoxEncoding::Dfl, DType::U16, DecodeDtype::F32) => return Ok(DflU16ToF32NeonFp16),
                _ => {} // fall through to NeonBase
            }
        }
        #[cfg(target_arch = "aarch64")]
        if features.neon_baseline {
            match (encoding, input, output) {
                (BoxEncoding::Dfl, DType::I8, DecodeDtype::F32) => return Ok(DflI8ToF32NeonBase),
                (BoxEncoding::Dfl, DType::U8, DecodeDtype::F32) => return Ok(DflU8ToF32NeonBase),
                (BoxEncoding::Dfl, DType::I16, DecodeDtype::F32) => return Ok(DflI16ToF32NeonBase),
                (BoxEncoding::Dfl, DType::U16, DecodeDtype::F32) => return Ok(DflU16ToF32NeonBase),
                (BoxEncoding::Direct, DType::I8, DecodeDtype::F32) => {
                    return Ok(LtrbI8ToF32NeonBase)
                }
                (BoxEncoding::Direct, DType::U8, DecodeDtype::F32) => {
                    return Ok(LtrbU8ToF32NeonBase)
                }
                (BoxEncoding::Direct, DType::I16, DecodeDtype::F32) => {
                    return Ok(LtrbI16ToF32NeonBase)
                }
                (BoxEncoding::Direct, DType::U16, DecodeDtype::F32) => {
                    return Ok(LtrbU16ToF32NeonBase)
                }
                _ => {} // fall through to scalar
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        let _ = features;

        let cell = match (encoding, input, output) {
            (BoxEncoding::Dfl, DType::I8, DecodeDtype::F32) => DflI8ToF32Scalar,
            (BoxEncoding::Dfl, DType::U8, DecodeDtype::F32) => DflU8ToF32Scalar,
            (BoxEncoding::Dfl, DType::I16, DecodeDtype::F32) => DflI16ToF32Scalar,
            (BoxEncoding::Dfl, DType::U16, DecodeDtype::F32) => DflU16ToF32Scalar,
            (BoxEncoding::Dfl, DType::F16, DecodeDtype::F32) => DflF16ToF32Scalar,
            (BoxEncoding::Dfl, DType::F32, DecodeDtype::F32) => DflF32ToF32Scalar,
            (BoxEncoding::Dfl, DType::I8, DecodeDtype::F16) => DflI8ToF16Scalar,
            (BoxEncoding::Dfl, DType::U8, DecodeDtype::F16) => DflU8ToF16Scalar,
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
                     not in Phase 1 fast-path matrix"
                )));
            }
        };
        Ok(cell)
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // Wired by Task 15.
#[allow(clippy::enum_variant_names)] // tier suffix is meaningful (Phase 2 adds Neon variants)
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
    // Tier 1 NEON-baseline (aarch64 only). Selected when
    // CpuFeatures::neon_baseline is true. Falls through to the scalar
    // variant for cells that don't have a NEON specialization yet
    // (currently the f16/f32 input cells; Phase 2-B adds them).
    #[cfg(target_arch = "aarch64")]
    I8ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    U8ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    I16ToF32NeonBase,
    #[cfg(target_arch = "aarch64")]
    U16ToF32NeonBase,

    // Tier 2 NEON FP16 (Cortex-A55+). Same dequant; sigmoid in 8-lane f16.
    #[cfg(target_arch = "aarch64")]
    I8ToF32NeonFp16,
    #[cfg(target_arch = "aarch64")]
    U8ToF32NeonFp16,
    #[cfg(target_arch = "aarch64")]
    I16ToF32NeonFp16,
    #[cfg(target_arch = "aarch64")]
    U16ToF32NeonFp16,
}

impl ScoreLevelDispatch {
    pub(crate) fn select(
        input: DType,
        output: DecodeDtype,
        features: &CpuFeatures,
    ) -> DecoderResult<Self> {
        use ScoreLevelDispatch::*;
        // Tier-1 NEON-base wins when the probe says it's available AND a
        // NEON variant exists for this cell. Otherwise the scalar arm
        // below is the fallback. This is the dispatch contract: a
        // *NeonBase variant existing in the returned value means the
        // CpuFeatures probe saw the corresponding hardware feature, so
        // calls into the unsafe `target_feature` kernel are sound.
        #[cfg(target_arch = "aarch64")]
        if features.neon_fp16 {
            match (input, output) {
                (DType::I8, DecodeDtype::F32) => return Ok(I8ToF32NeonFp16),
                (DType::U8, DecodeDtype::F32) => return Ok(U8ToF32NeonFp16),
                (DType::I16, DecodeDtype::F32) => return Ok(I16ToF32NeonFp16),
                (DType::U16, DecodeDtype::F32) => return Ok(U16ToF32NeonFp16),
                _ => {} // fall through to NeonBase
            }
        }
        #[cfg(target_arch = "aarch64")]
        if features.neon_baseline {
            match (input, output) {
                (DType::I8, DecodeDtype::F32) => return Ok(I8ToF32NeonBase),
                (DType::U8, DecodeDtype::F32) => return Ok(U8ToF32NeonBase),
                (DType::I16, DecodeDtype::F32) => return Ok(I16ToF32NeonBase),
                (DType::U16, DecodeDtype::F32) => return Ok(U16ToF32NeonBase),
                _ => {} // fall through to scalar
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        let _ = features;

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
                    "per-scale ScoreLevelDispatch: input {dt:?} not in Phase 1 matrix"
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
        // FP16 brings no advantage to mc/proto (they're pure dequant —
        // no exp/sigmoid/softmax) and the fp16 NEON variants don't have
        // mc/proto kernels wired. Force the score selector to skip the
        // fp16 tier for these consumers.
        let f_no_fp16 = CpuFeatures {
            neon_fp16: false,
            ..*f
        };
        ScoreLevelDispatch::select(input, output, &f_no_fp16).map(Self)
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
        let f_no_fp16 = CpuFeatures {
            neon_fp16: false,
            ..*f
        };
        ScoreLevelDispatch::select(input, output, &f_no_fp16).map(Self)
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

/// Mutable destination slice, dtype-tagged. The dispatch arm matches
/// on this to pick the f32 vs f16 output cell.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum DstSliceMut<'a> {
    F32(&'a mut [f32]),
    F16(&'a mut [f16]),
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
use super::level_box::{
    decode_box_level_dfl_i16_to_f32_neon, decode_box_level_dfl_i16_to_f32_neon_fp16,
    decode_box_level_dfl_i8_to_f32_neon, decode_box_level_dfl_i8_to_f32_neon_fp16,
    decode_box_level_dfl_u16_to_f32_neon, decode_box_level_dfl_u16_to_f32_neon_fp16,
    decode_box_level_dfl_u8_to_f32_neon, decode_box_level_dfl_u8_to_f32_neon_fp16,
    decode_box_level_ltrb_i16_to_f32_neon, decode_box_level_ltrb_i8_to_f32_neon,
    decode_box_level_ltrb_u16_to_f32_neon, decode_box_level_ltrb_u8_to_f32_neon,
};
use crate::per_scale::plan::LevelPlan;
use crate::Quantization;

impl BoxLevelDispatch {
    /// Run this dispatch's concrete kernel.
    /// Returns `KernelDispatchUnreachable` if the variant tag and
    /// (input, dst) dtypes disagree — this would indicate an internal
    /// logic bug at construction time.
    #[allow(dead_code)] // Wired in Task 21.
    pub(crate) fn run(
        &self,
        input: InputView<'_>,
        q: Quantization,
        level: &LevelPlan,
        dst: DstSliceMut<'_>,
    ) -> DecoderResult<()> {
        use BoxLevelDispatch::*;
        match (self, input, dst) {
            // DFL × f32 output
            (DflI8ToF32Scalar, InputView::I8(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_i8_to_f32(i, q, level, d)
            }
            (DflU8ToF32Scalar, InputView::U8(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_u8_to_f32(i, q, level, d)
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
            (DflI8ToF16Scalar, InputView::I8(i), DstSliceMut::F16(d)) => {
                decode_box_level_dfl_i8_to_f16(i, q, level, d)
            }
            (DflU8ToF16Scalar, InputView::U8(i), DstSliceMut::F16(d)) => {
                decode_box_level_dfl_u8_to_f16(i, q, level, d)
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

            // Tier-1 NEON-base arms.
            #[cfg(target_arch = "aarch64")]
            (DflI8ToF32NeonBase, InputView::I8(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_i8_to_f32_neon(i, q, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (DflU8ToF32NeonBase, InputView::U8(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_u8_to_f32_neon(i, q, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (DflI16ToF32NeonBase, InputView::I16(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_i16_to_f32_neon(i, q, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (DflU16ToF32NeonBase, InputView::U16(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_u16_to_f32_neon(i, q, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (LtrbI8ToF32NeonBase, InputView::I8(i), DstSliceMut::F32(d)) => {
                decode_box_level_ltrb_i8_to_f32_neon(i, q, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (LtrbU8ToF32NeonBase, InputView::U8(i), DstSliceMut::F32(d)) => {
                decode_box_level_ltrb_u8_to_f32_neon(i, q, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (LtrbI16ToF32NeonBase, InputView::I16(i), DstSliceMut::F32(d)) => {
                decode_box_level_ltrb_i16_to_f32_neon(i, q, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (LtrbU16ToF32NeonBase, InputView::U16(i), DstSliceMut::F32(d)) => {
                decode_box_level_ltrb_u16_to_f32_neon(i, q, level, d)
            }

            // Tier-2 NEON FP16 DFL arms.
            #[cfg(target_arch = "aarch64")]
            (DflI8ToF32NeonFp16, InputView::I8(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_i8_to_f32_neon_fp16(i, q, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (DflU8ToF32NeonFp16, InputView::U8(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_u8_to_f32_neon_fp16(i, q, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (DflI16ToF32NeonFp16, InputView::I16(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_i16_to_f32_neon_fp16(i, q, level, d)
            }
            #[cfg(target_arch = "aarch64")]
            (DflU16ToF32NeonFp16, InputView::U16(i), DstSliceMut::F32(d)) => {
                decode_box_level_dfl_u16_to_f32_neon_fp16(i, q, level, d)
            }

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
    decode_score_level_i16_to_f32_neon, decode_score_level_i16_to_f32_neon_fp16,
    decode_score_level_i8_to_f32_neon, decode_score_level_i8_to_f32_neon_fp16,
    decode_score_level_u16_to_f32_neon, decode_score_level_u16_to_f32_neon_fp16,
    decode_score_level_u8_to_f32_neon, decode_score_level_u8_to_f32_neon_fp16,
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
            #[cfg(target_arch = "aarch64")]
            (I8ToF32NeonFp16, InputView::I8(i), DstSliceMut::F32(d)) => {
                decode_score_level_i8_to_f32_neon_fp16(i, q, num_classes, level, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (U8ToF32NeonFp16, InputView::U8(i), DstSliceMut::F32(d)) => {
                decode_score_level_u8_to_f32_neon_fp16(i, q, num_classes, level, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (I16ToF32NeonFp16, InputView::I16(i), DstSliceMut::F32(d)) => {
                decode_score_level_i16_to_f32_neon_fp16(i, q, num_classes, level, activation, d)
            }
            #[cfg(target_arch = "aarch64")]
            (U16ToF32NeonFp16, InputView::U16(i), DstSliceMut::F32(d)) => {
                decode_score_level_u16_to_f32_neon_fp16(i, q, num_classes, level, activation, d)
            }
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

    #[test]
    fn box_dispatch_select_dfl_i8_to_f32_returns_scalar_when_no_simd() {
        let f = CpuFeatures::default();
        let d =
            BoxLevelDispatch::select(BoxEncoding::Dfl, DType::I8, DecodeDtype::F32, &f).unwrap();
        assert!(matches!(d, BoxLevelDispatch::DflI8ToF32Scalar));
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
        use crate::per_scale::kernels::grids::make_anchor_grid;
        use crate::per_scale::plan::LevelPlan;
        use crate::Quantization;

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
                DstSliceMut::F32(&mut out),
            )
            .unwrap();
        assert!((out[0] - 4.0).abs() < 1e-3);
        assert!((out[2] - 80.0).abs() < 1e-3);
    }
}
