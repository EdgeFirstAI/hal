// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Elementwise sigmoid for any input and output type, exact to TFLite `LOGISTIC` where it has one.

use half::f16;

use super::lut_i8::LutI8Plan;
use super::rows::{for_each_chunk, CHUNK};
use crate::cpu::{CpuFeatures, Tier};
use crate::error::{KernelError, Result};
use crate::pool::Pool;
use crate::prim::dequant as dq;
use crate::prim::sigmoid::sigmoid_slice_f32;
use crate::quant::{input_quant, int_range, output_quant, quantize, DType, Quant, QuantInt};
use crate::tflite::{logistic_i16_pot, logistic_i16_prepare, logistic_i16_reference};
use crate::view::{TensorMut, TensorRef, View, ViewMut};

/// Largest int16 reference multiplier for which `x · multiplier` fits an `i32` for every `x`.
pub(super) const MAX_I16_MULTIPLIER: i32 = 65535;

/// Parameters of a sigmoid.
///
/// `input` applies to integer input types and is ignored for `f16` and `f32` inputs; `output`
/// applies to integer output types and is ignored for float outputs, so a float tensor may
/// carry any `Quant`, `{0.0, 0}` included.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SigmoidParams {
    pub input_dtype: DType,
    pub input: Quant,
    pub output_dtype: DType,
    pub output: Quant,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum I16Kernel {
    /// `optimized_ops::Logistic` int16 (gemmlowp), for an input scale of 2⁻¹².
    Pot,
    /// `reference_integer_ops::Logistic` int16 with the prepared input rescale.
    Reference { multiplier: i32, left_shift: i32 },
}

/// The instruction set a path's kernels use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Isa {
    Scalar,
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// FP16 arithmetic, for f16 outputs only.
    #[cfg(target_arch = "aarch64")]
    NeonFp16,
}

impl Isa {
    fn tier(self) -> Tier {
        match self {
            Isa::Scalar => Tier::Scalar,
            #[cfg(target_arch = "aarch64")]
            Isa::Neon => Tier::Neon,
            #[cfg(target_arch = "aarch64")]
            Isa::NeonFp16 => Tier::NeonFp16,
        }
    }
}

#[derive(Clone, Debug)]
enum Path {
    Lut(Box<LutI8Plan>),
    I16(I16Kernel, Isa),
    Float(Isa),
    Requant {
        inv_scale: f32,
        zero_point: i32,
        isa: Isa,
    },
}

/// A validated elementwise sigmoid; shapes may have any rank.
///
/// The path is chosen at creation, in this order:
///
/// 1. `i8 → i8` and `u8 → u8`: a 256-entry table, [`LutI8Plan::sigmoid`]; bit-identical to
///    TFLite's int8/uint8 `LOGISTIC`.
/// 2. `i16 → i16` with input zero point 0, output zero point 0 and output scale exactly
///    1/32768: TFLite's int16 `LOGISTIC`, bit for bit on every tier. An input scale of 2⁻¹²
///    (by TFLite's `CheckedLog2`) runs gemmlowp's fixed-point logistic; any other scale runs
///    the reference table kernel with TFLite's prepared multiplier. Input scales above about
///    5.3, where that kernel's int32 product overflows, take path 5 instead.
/// 3. Any input to `f32`: dequantize and sigmoid in f32, with the f32 exp on every tier.
///    Integer inputs run the decoder's fused NEON kernel on every NEON tier; f16 and f32
///    inputs run the NEON f32 sigmoid.
/// 4. Any input to `f16`: on the FP16 tier, an FP16-arithmetic sigmoid, tested within 2 f16
///    ulps of the float64 sigmoid rounded to f16; on the other tiers, path 3's f32 result
///    rounded to f16 once (nearest even).
/// 5. Anything else: dequantize to f32 as `(x − zp) · scale`, the f32 sigmoid of path 3, then
///    [`quantize`] once (round half away from zero, clamp). Within one LSB of a float64
///    sigmoid.
///
/// ```
/// use edgefirst_kernels::{DType, Quant, SigmoidParams, SigmoidPlan, TensorMut, TensorRef, View, ViewMut};
///
/// let plan = SigmoidPlan::new(SigmoidParams {
///     input_dtype: DType::I8,
///     input: Quant::new(0.1, 0),
///     output_dtype: DType::I8,
///     output: Quant::new(1.0 / 256.0, -128),
/// })?;
/// let input = [0i8, 127, -128];
/// let mut output = [0i8; 3];
/// plan.run(
///     TensorRef::I8(View::contiguous(&input, &[3])?),
///     TensorMut::I8(ViewMut::contiguous(&mut output, &[3])?),
///     None,
/// )?;
/// assert_eq!(output, [0, 127, -128]);
/// # Ok::<(), edgefirst_kernels::KernelError>(())
/// ```
#[derive(Clone, Debug)]
pub struct SigmoidPlan {
    params: SigmoidParams,
    path: Path,
}

impl SigmoidPlan {
    /// Builds a plan for the CPU this process runs on, honouring `EDGEFIRST_KERNELS_FORCE_TIER`.
    pub fn new(p: SigmoidParams) -> Result<Self> {
        Self::with_features(p, CpuFeatures::from_env_or_probe()?)
    }

    /// Builds a plan that runs the best kernel `f` allows; reads no environment.
    ///
    /// Integer scales must be finite and positive, and integer zero points inside their type's
    /// range (`InvalidArgument` otherwise). Every input and output type pair is supported.
    pub fn with_features(p: SigmoidParams, f: CpuFeatures) -> Result<Self> {
        output_quant(p.output_dtype, p.output)?;
        let p = SigmoidParams {
            input: input_quant(p.input_dtype, p.input)?,
            output: if int_range(p.output_dtype).is_some() {
                p.output
            } else {
                Quant::IDENTITY
            },
            ..p
        };
        let (neon, fp16) = (neon_isa(f), fp16_isa(f));
        let path = match (p.input_dtype, p.output_dtype) {
            (DType::I8, DType::I8) | (DType::U8, DType::U8) => Path::Lut(Box::new(
                LutI8Plan::sigmoid_with_features(p.input_dtype, p.input, p.output, f)?,
            )),
            (_, DType::F16) => Path::Float(fp16),
            (_, DType::F32) => Path::Float(neon),
            (DType::I16, DType::I16) if tflite_i16_quant(p) => match i16_kernel(p.input.scale) {
                Some(k) => Path::I16(k, neon),
                None => requant(p.output, neon)?,
            },
            _ => requant(p.output, neon)?,
        };
        Ok(Self { params: p, path })
    }

    /// The tier of the kernel this plan runs.
    pub fn tier(&self) -> Tier {
        match &self.path {
            Path::Lut(lut) => lut.tier(),
            Path::I16(_, isa) | Path::Float(isa) | Path::Requant { isa, .. } => isa.tier(),
        }
    }

    /// Sigmoid of `input` into `output`, with the plan's dtypes and the same shape (any rank).
    ///
    /// Work is split across `pool` when one is given; results do not depend on the pool or the
    /// strides. Output axes sorted by stride must each step past the extent of the smaller
    /// ones. Arguments are validated before anything is written.
    pub fn run(
        &self,
        input: TensorRef<'_>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
    ) -> Result<()> {
        let _span = tracing::trace_span!("kernels.sigmoid", tier = self.tier().as_str()).entered();
        if input.dtype() != self.params.input_dtype || output.dtype() != self.params.output_dtype {
            return Err(KernelError::InvalidArgument(
                "sigmoid input and output dtypes differ from the plan's",
            ));
        }
        match &self.path {
            Path::Lut(lut) => lut.run_inner(input, output, pool),
            Path::I16(k, isa) => match (input, output) {
                (TensorRef::I16(i), TensorMut::I16(mut o)) => {
                    for_each_chunk(&i, &mut o, pool, &|s, d| logistic_i16(*k, *isa, s, d))
                }
                _ => Err(KernelError::InvalidArgument("sigmoid needs i16 views")),
            },
            Path::Float(isa) => {
                let q = self.params.input;
                match input {
                    TensorRef::I8(i) => run_float(*isa, q, &i, output, pool),
                    TensorRef::U8(i) => run_float(*isa, q, &i, output, pool),
                    TensorRef::I16(i) => run_float(*isa, q, &i, output, pool),
                    TensorRef::U16(i) => run_float(*isa, q, &i, output, pool),
                    TensorRef::F16(i) => run_float(*isa, q, &i, output, pool),
                    TensorRef::F32(i) => run_float(*isa, q, &i, output, pool),
                }
            }
            Path::Requant {
                inv_scale,
                zero_point,
                isa,
            } => {
                let r = Requant {
                    input: self.params.input,
                    inv_scale: *inv_scale,
                    zero_point: *zero_point,
                    isa: *isa,
                };
                match input {
                    TensorRef::I8(i) => r.run(&i, output, pool),
                    TensorRef::U8(i) => r.run(&i, output, pool),
                    TensorRef::I16(i) => r.run(&i, output, pool),
                    TensorRef::U16(i) => r.run(&i, output, pool),
                    TensorRef::F16(i) => r.run(&i, output, pool),
                    TensorRef::F32(i) => r.run(&i, output, pool),
                }
            }
        }
    }
}

/// TFLite's int16 `LOGISTIC` quantization: zero points 0 and output scale exactly 1/32768.
fn tflite_i16_quant(p: SigmoidParams) -> bool {
    p.input.zero_point == 0 && p.output.zero_point == 0 && p.output.scale == 1.0 / 32768.0
}

/// The kernel TFLite's `SigmoidEval` picks for int16, or `None` where its arithmetic overflows.
pub(super) fn i16_kernel(input_scale: f32) -> Option<I16Kernel> {
    if f64::from(input_scale) * 4096.0 * 3.0 >= f64::from(MAX_I16_MULTIPLIER + 1) {
        return None;
    }
    let (multiplier, left_shift) = logistic_i16_prepare(input_scale);
    Some(if multiplier > 0 {
        I16Kernel::Reference {
            multiplier,
            left_shift,
        }
    } else {
        I16Kernel::Pot
    })
}

fn requant(output: Quant, isa: Isa) -> Result<Path> {
    let inv_scale = 1.0f32 / output.scale;
    if !inv_scale.is_finite() {
        return Err(KernelError::InvalidArgument(
            "sigmoid output scale is too small",
        ));
    }
    Ok(Path::Requant {
        inv_scale,
        zero_point: output.zero_point,
        isa,
    })
}

#[cfg(target_arch = "aarch64")]
fn neon_isa(f: CpuFeatures) -> Isa {
    if f.neon_baseline {
        Isa::Neon
    } else {
        Isa::Scalar
    }
}

#[cfg(target_arch = "aarch64")]
fn fp16_isa(f: CpuFeatures) -> Isa {
    if f.fp16_usable() {
        Isa::NeonFp16
    } else {
        neon_isa(f)
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn neon_isa(_: CpuFeatures) -> Isa {
    Isa::Scalar
}

#[cfg(not(target_arch = "aarch64"))]
fn fp16_isa(_: CpuFeatures) -> Isa {
    Isa::Scalar
}

#[inline]
fn logistic_i16(k: I16Kernel, isa: Isa, src: &[i16], dst: &mut [i16]) {
    match (k, isa) {
        #[cfg(target_arch = "aarch64")]
        (I16Kernel::Pot, Isa::Neon | Isa::NeonFp16) => {
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe { super::sigmoid_i16_neon::logistic_i16_pot_neon(src, dst) }
        }
        #[cfg(target_arch = "aarch64")]
        (
            I16Kernel::Reference {
                multiplier,
                left_shift,
            },
            Isa::Neon | Isa::NeonFp16,
        ) => {
            // SAFETY: NEON is part of the aarch64 baseline.
            unsafe {
                super::sigmoid_i16_neon::logistic_i16_reference_neon(
                    multiplier, left_shift, src, dst,
                )
            }
        }
        (I16Kernel::Pot, Isa::Scalar) => {
            for (d, &x) in dst.iter_mut().zip(src) {
                *d = logistic_i16_pot(x);
            }
        }
        (
            I16Kernel::Reference {
                multiplier,
                left_shift,
            },
            Isa::Scalar,
        ) => {
            for (d, &x) in dst.iter_mut().zip(src) {
                *d = logistic_i16_reference(multiplier, left_shift, x);
            }
        }
    }
}

/// An input element type: its scalar dequantize, and its NEON f32 and FP16 sigmoid kernels.
trait SigmoidInput: Copy + Default + Sync {
    fn dequant_f32(src: &[Self], q: Quant, dst: &mut [f32]);
    /// NEON dequantize then sigmoid into f32; writes nothing when the lengths differ.
    ///
    /// # Safety
    /// The CPU must support NEON.
    #[cfg(target_arch = "aarch64")]
    unsafe fn sigmoid_f32_neon(src: &[Self], q: Quant, dst: &mut [f32]);
    /// The NEON f32 dequantize and sigmoid rounded once into f16, in one pass; equal to
    /// [`SigmoidInput::sigmoid_f32_neon`] rounded to f16. Writes nothing when the lengths differ.
    ///
    /// # Safety
    /// The CPU must support NEON.
    #[cfg(target_arch = "aarch64")]
    unsafe fn sigmoid_f16_neon(src: &[Self], q: Quant, dst: &mut [f16]);
    /// FP16-arithmetic dequantize and sigmoid into f16; writes nothing when the lengths differ.
    ///
    /// # Safety
    /// The CPU must support FEAT_FP16.
    #[cfg(target_arch = "aarch64")]
    unsafe fn sigmoid_f16_neon_fp16(src: &[Self], q: Quant, dst: &mut [f16]);
}

macro_rules! sigmoid_input {
    ($t:ty, $to_f32:ident, $f32:path, $f16:ident, $fp16:ident) => {
        impl SigmoidInput for $t {
            #[inline]
            fn dequant_f32(src: &[Self], q: Quant, dst: &mut [f32]) {
                dq::$to_f32(src, q, dst);
            }
            #[cfg(target_arch = "aarch64")]
            #[inline]
            unsafe fn sigmoid_f32_neon(src: &[Self], q: Quant, dst: &mut [f32]) {
                if src.len() == dst.len() {
                    // SAFETY: the caller guarantees NEON; the lengths are equal.
                    unsafe { $f32(src, q, dst) };
                }
            }
            #[cfg(target_arch = "aarch64")]
            #[inline]
            unsafe fn sigmoid_f16_neon(src: &[Self], q: Quant, dst: &mut [f16]) {
                // SAFETY: the caller guarantees NEON.
                unsafe { crate::prim::neon_sigmoid::$f16(src, q, dst) };
            }
            #[cfg(target_arch = "aarch64")]
            #[inline]
            unsafe fn sigmoid_f16_neon_fp16(src: &[Self], q: Quant, dst: &mut [f16]) {
                // SAFETY: the caller guarantees FEAT_FP16.
                unsafe { crate::prim::neon_fp16::$fp16(src, q, dst) };
            }
        }
    };
}

sigmoid_input!(
    i8,
    dequant_i8_to_f32,
    crate::prim::neon_baseline::dequant_sigmoid_i8_to_f32_neon,
    dequant_sigmoid_i8_to_f16_neon,
    dequant_sigmoid_i8_to_f16_neon_fp16
);
sigmoid_input!(
    u8,
    dequant_u8_to_f32,
    crate::prim::neon_baseline::dequant_sigmoid_u8_to_f32_neon,
    dequant_sigmoid_u8_to_f16_neon,
    dequant_sigmoid_u8_to_f16_neon_fp16
);
sigmoid_input!(
    i16,
    dequant_i16_to_f32,
    crate::prim::neon_baseline::dequant_sigmoid_i16_to_f32_neon,
    dequant_sigmoid_i16_to_f16_neon,
    dequant_sigmoid_i16_to_f16_neon_fp16
);
sigmoid_input!(
    u16,
    dequant_u16_to_f32,
    crate::prim::neon_baseline::dequant_sigmoid_u16_to_f32_neon,
    dequant_sigmoid_u16_to_f16_neon,
    dequant_sigmoid_u16_to_f16_neon_fp16
);
sigmoid_input!(
    f16,
    dequant_f16_to_f32,
    crate::prim::neon_sigmoid::dequant_sigmoid_f16_to_f32_neon,
    dequant_sigmoid_f16_to_f16_neon,
    dequant_sigmoid_f16_to_f16_neon_fp16
);
sigmoid_input!(
    f32,
    dequant_f32_to_f32,
    crate::prim::neon_sigmoid::dequant_sigmoid_f32_to_f32_neon,
    dequant_sigmoid_f32_to_f16_neon,
    dequant_sigmoid_f32_to_f16_neon_fp16
);

/// The f32 sigmoid of `src` on `isa`, into `dst` of the same length.
#[inline]
fn sigmoid_f32<I: SigmoidInput>(isa: Isa, src: &[I], q: Quant, dst: &mut [f32]) {
    match isa {
        Isa::Scalar => {
            I::dequant_f32(src, q, dst);
            sigmoid_slice_f32(dst);
        }
        // SAFETY: NEON is part of the aarch64 baseline.
        #[cfg(target_arch = "aarch64")]
        Isa::Neon | Isa::NeonFp16 => unsafe { I::sigmoid_f32_neon(src, q, dst) },
    }
}

/// The f16 sigmoid of `src` on `isa`: the FP16 kernel, or the f32 result rounded once.
#[inline]
fn sigmoid_f16<I: SigmoidInput>(isa: Isa, src: &[I], q: Quant, dst: &mut [f16]) {
    match isa {
        // SAFETY: `NeonFp16` is only selected when FEAT_FP16 was detected.
        #[cfg(target_arch = "aarch64")]
        Isa::NeonFp16 => unsafe { I::sigmoid_f16_neon_fp16(src, q, dst) },
        // SAFETY: NEON is part of the aarch64 baseline.
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => unsafe { I::sigmoid_f16_neon(src, q, dst) },
        Isa::Scalar => {
            let mut buf = [0.0f32; CHUNK];
            let (Some(buf), true) = (buf.get_mut(..src.len()), src.len() == dst.len()) else {
                return;
            };
            sigmoid_f32(isa, src, q, buf);
            for (o, &y) in dst.iter_mut().zip(buf.iter()) {
                *o = f16::from_f32(y);
            }
        }
    }
}

fn run_float<I: SigmoidInput>(
    isa: Isa,
    q: Quant,
    input: &View<'_, I>,
    output: TensorMut<'_>,
    pool: Option<&dyn Pool>,
) -> Result<()> {
    match output {
        TensorMut::F32(mut o) => {
            for_each_chunk(input, &mut o, pool, &|s, d| sigmoid_f32(isa, s, q, d))
        }
        TensorMut::F16(mut o) => {
            for_each_chunk(input, &mut o, pool, &|s, d| sigmoid_f16(isa, s, q, d))
        }
        _ => Err(KernelError::InvalidArgument("sigmoid needs a float output")),
    }
}

/// A quantized output type and its NEON quantize kernel.
trait SigmoidOutput: QuantInt + Default + Send {
    /// [`quantize`] of every element, bit for bit; writes nothing when the lengths differ.
    ///
    /// # Safety
    /// The CPU must support NEON.
    #[cfg(target_arch = "aarch64")]
    unsafe fn quantize_neon(src: &[f32], inv_scale: f32, zero_point: i32, dst: &mut [Self]);
}

macro_rules! sigmoid_output {
    ($t:ty, $neon:ident) => {
        impl SigmoidOutput for $t {
            #[cfg(target_arch = "aarch64")]
            #[inline]
            unsafe fn quantize_neon(
                src: &[f32],
                inv_scale: f32,
                zero_point: i32,
                dst: &mut [Self],
            ) {
                // SAFETY: the caller guarantees NEON.
                unsafe { crate::prim::neon_convert::$neon(src, inv_scale, zero_point, dst) };
            }
        }
    };
}

sigmoid_output!(i8, quantize_f32_to_i8_neon);
sigmoid_output!(u8, quantize_f32_to_u8_neon);
sigmoid_output!(i16, quantize_f32_to_i16_neon);
sigmoid_output!(u16, quantize_f32_to_u16_neon);

/// The one-rounding path: f32 dequantize, f32 sigmoid, one quantize.
struct Requant {
    input: Quant,
    inv_scale: f32,
    zero_point: i32,
    isa: Isa,
}

impl Requant {
    fn run<I: SigmoidInput>(
        &self,
        input: &View<'_, I>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
    ) -> Result<()> {
        match output {
            TensorMut::I8(mut o) => self.run_typed(input, &mut o, pool),
            TensorMut::U8(mut o) => self.run_typed(input, &mut o, pool),
            TensorMut::I16(mut o) => self.run_typed(input, &mut o, pool),
            TensorMut::U16(mut o) => self.run_typed(input, &mut o, pool),
            TensorMut::F16(_) | TensorMut::F32(_) => Err(KernelError::InvalidArgument(
                "sigmoid needs a quantized output",
            )),
        }
    }

    fn run_typed<I: SigmoidInput, O: SigmoidOutput>(
        &self,
        input: &View<'_, I>,
        output: &mut ViewMut<'_, O>,
        pool: Option<&dyn Pool>,
    ) -> Result<()> {
        for_each_chunk(input, output, pool, &|s, d| {
            let mut buf = [0.0f32; CHUNK];
            let (Some(buf), true) = (buf.get_mut(..s.len()), s.len() == d.len()) else {
                return;
            };
            sigmoid_f32(self.isa, s, self.input, buf);
            match self.isa {
                Isa::Scalar => {
                    for (o, &y) in d.iter_mut().zip(buf.iter()) {
                        *o = quantize::<O>(y, self.inv_scale, self.zero_point);
                    }
                }
                // SAFETY: NEON is part of the aarch64 baseline.
                #[cfg(target_arch = "aarch64")]
                Isa::Neon | Isa::NeonFp16 => unsafe {
                    O::quantize_neon(buf, self.inv_scale, self.zero_point, d)
                },
            }
        })
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

    fn plan(input_dtype: DType, input: Quant, output_dtype: DType, output: Quant) -> SigmoidPlan {
        plan_on(
            input_dtype,
            input,
            output_dtype,
            output,
            CpuFeatures::default(),
        )
    }

    fn plan_on(
        input_dtype: DType,
        input: Quant,
        output_dtype: DType,
        output: Quant,
        f: CpuFeatures,
    ) -> SigmoidPlan {
        let p = SigmoidParams {
            input_dtype,
            input,
            output_dtype,
            output,
        };
        SigmoidPlan::with_features(p, f).unwrap()
    }

    fn path(input_dtype: DType, input: Quant, output_dtype: DType, output: Quant) -> Path {
        plan(input_dtype, input, output_dtype, output).path
    }

    #[test]
    fn path_selection() {
        let q8 = Quant::new(0.1, 0);
        let q16 = Quant::new(1.0 / 32768.0, 0);
        assert!(matches!(path(DType::I8, q8, DType::I8, q8), Path::Lut(_)));
        assert!(matches!(path(DType::U8, q8, DType::U8, q8), Path::Lut(_)));
        assert!(matches!(
            path(DType::I8, q8, DType::U8, q8),
            Path::Requant { .. }
        ));
        assert!(matches!(
            path(DType::I16, Quant::new(2f32.powi(-12), 0), DType::I16, q16),
            Path::I16(I16Kernel::Pot, Isa::Scalar)
        ));
        assert!(matches!(
            path(DType::I16, Quant::new(0.0005, 0), DType::I16, q16),
            Path::I16(
                I16Kernel::Reference {
                    multiplier: 25165,
                    left_shift: 12
                },
                Isa::Scalar
            )
        ));
        // TFLite's quantization requirements, each broken in turn.
        for (i, o) in [
            (Quant::new(0.0005, 1), q16),
            (Quant::new(0.0005, 0), Quant::new(1.0 / 32768.0, 1)),
            (Quant::new(0.0005, 0), Quant::new(1.0 / 32000.0, 0)),
            (Quant::new(6.0, 0), q16),
        ] {
            assert!(
                matches!(path(DType::I16, i, DType::I16, o), Path::Requant { .. }),
                "{i:?} {o:?}"
            );
        }
        // Largest scale the int16 reference kernel takes without overflow.
        assert!(matches!(
            path(DType::I16, Quant::new(5.33, 0), DType::I16, q16),
            Path::I16(I16Kernel::Reference { .. }, _)
        ));
        for d in ALL {
            for o in [DType::F32, DType::F16] {
                assert!(
                    matches!(path(d, q8, o, q8), Path::Float(Isa::Scalar)),
                    "{d:?} -> {o:?}"
                );
            }
            for o in [DType::I8, DType::U8, DType::I16, DType::U16] {
                if (d, o) != (DType::I8, DType::I8) && (d, o) != (DType::U8, DType::U8) {
                    assert!(
                        matches!(path(d, q8, o, q8), Path::Requant { .. } | Path::I16(..)),
                        "{d:?} -> {o:?}"
                    );
                }
            }
        }
    }

    /// Every input/output pair with the given quantization, TFLite's int16 one included.
    fn every_pair() -> Vec<(DType, Quant, DType, Quant)> {
        let (q8, q16) = (Quant::new(0.1, 0), Quant::new(1.0 / 32768.0, 0));
        let mut pairs = vec![
            (DType::I16, Quant::new(2f32.powi(-12), 0), DType::I16, q16),
            (DType::I16, Quant::new(0.0005, 0), DType::I16, q16),
        ];
        for i in ALL {
            for o in ALL {
                pairs.push((i, q8, o, q8));
            }
        }
        pairs
    }

    #[test]
    fn scalar_features_run_scalar_everywhere() {
        for (i, qi, o, qo) in every_pair() {
            assert_eq!(plan(i, qi, o, qo).tier(), Tier::Scalar, "{i:?} -> {o:?}");
        }
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn every_path_is_neon_under_neon_features() {
        let neon = CpuFeatures::from_forced_tier(Some("neon")).unwrap();
        for (i, qi, o, qo) in every_pair() {
            let p = plan_on(i, qi, o, qo, neon);
            assert_eq!(p.tier(), Tier::Neon, "neon {i:?} -> {o:?}");
        }
        let Ok(fp16) = CpuFeatures::from_forced_tier(Some("neon_fp16")) else {
            crate::test_support::report_skip("no neon_fp16 tier on this CPU");
            return;
        };
        for (i, qi, o, qo) in every_pair() {
            let p = plan_on(i, qi, o, qo, fp16);
            // FP16 kernels run for f16 outputs only.
            let want = if o == DType::F16 {
                Tier::NeonFp16
            } else {
                Tier::Neon
            };
            assert_eq!(p.tier(), want, "neon_fp16 {i:?} -> {o:?}");
        }
        // The probed features of an FP16 CPU, dotprod included, still pick the FP16 kernels.
        let probed = CpuFeatures::probe();
        let q = Quant::new(0.1, 0);
        assert_eq!(
            plan_on(DType::I8, q, DType::F16, q, probed).tier(),
            Tier::NeonFp16
        );
    }
}
