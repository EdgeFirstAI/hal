// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Quantization parameters, element types and the quantize helper.

use crate::error::{KernelError, Result};

/// Per-tensor affine quantization: `real = (q - zero_point) * scale`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quant {
    pub scale: f32,
    pub zero_point: i32,
}

impl Quant {
    /// The identity mapping (scale 1, zero point 0).
    pub const IDENTITY: Quant = Quant {
        scale: 1.0,
        zero_point: 0,
    };

    pub fn new(scale: f32, zero_point: i32) -> Self {
        Self { scale, zero_point }
    }
}

/// Element type tag; the discriminants are the C ABI values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum DType {
    I8 = 0,
    U8 = 1,
    I16 = 2,
    U16 = 3,
    F16 = 4,
    F32 = 5,
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for i8 {}
    impl Sealed for u8 {}
    impl Sealed for i16 {}
    impl Sealed for u16 {}
}

/// Integer storage types for quantized tensors. Sealed.
pub trait QuantInt: Copy + Send + Sync + 'static + sealed::Sealed {
    const DTYPE: DType;
    const MIN: i32;
    const MAX: i32;
    fn to_i32(self) -> i32;
    /// Clamp `v` to `MIN..=MAX` and narrow.
    fn from_i32_clamped(v: i32) -> Self;
}

macro_rules! impl_quant_int {
    ($t:ty, $d:expr) => {
        impl QuantInt for $t {
            const DTYPE: DType = $d;
            const MIN: i32 = <$t>::MIN as i32;
            const MAX: i32 = <$t>::MAX as i32;
            #[inline(always)]
            fn to_i32(self) -> i32 {
                i32::from(self)
            }
            #[inline(always)]
            fn from_i32_clamped(v: i32) -> Self {
                v.clamp(<Self as QuantInt>::MIN, <Self as QuantInt>::MAX) as $t
            }
        }
    };
}
impl_quant_int!(i8, DType::I8);
impl_quant_int!(u8, DType::U8);
impl_quant_int!(i16, DType::I16);
impl_quant_int!(u16, DType::U16);

/// `clamp(round_half_away(x * inv_scale) + zero_point)`; NaN maps to the zero point.
#[inline(always)]
pub fn quantize<T: QuantInt>(x: f32, inv_scale: f32, zero_point: i32) -> T {
    let v = x * inv_scale;
    // `as i32` saturates; NaN is handled explicitly rather than relying on that cast.
    let q = if v.is_nan() { 0 } else { v.round() as i32 };
    T::from_i32_clamped(q.saturating_add(zero_point))
}

/// Range of an integer `d`, or `None` for a float type.
pub(crate) fn int_range(d: DType) -> Option<(i32, i32)> {
    match d {
        DType::I8 => Some((i8::MIN.into(), i8::MAX.into())),
        DType::U8 => Some((u8::MIN.into(), u8::MAX.into())),
        DType::I16 => Some((i16::MIN.into(), i16::MAX.into())),
        DType::U16 => Some((u16::MIN.into(), u16::MAX.into())),
        DType::F16 | DType::F32 => None,
    }
}

/// The quantization a plan applies to its input: validated for integer types, the identity for
/// float types (whose `Quant` is ignored).
pub(crate) fn input_quant(d: DType, q: Quant) -> Result<Quant> {
    let Some((lo, hi)) = int_range(d) else {
        return Ok(Quant::IDENTITY);
    };
    if !(q.scale.is_finite() && q.scale > 0.0) {
        return Err(KernelError::InvalidArgument(
            "input scale must be finite and positive",
        ));
    }
    if !(lo..=hi).contains(&q.zero_point) {
        return Err(KernelError::InvalidArgument(
            "input zero point must lie in the input type's range",
        ));
    }
    Ok(q)
}

/// How f32 results are stored: [`quantize`] for integer outputs, ignored for float outputs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct OutQuant {
    pub(crate) inv_scale: f32,
    pub(crate) zero_point: i32,
}

/// Validates the output quantization of an integer output type; float types ignore it.
pub(crate) fn output_quant(d: DType, q: Quant) -> Result<OutQuant> {
    let Some((lo, hi)) = int_range(d) else {
        return Ok(OutQuant {
            inv_scale: 1.0,
            zero_point: 0,
        });
    };
    if !(q.scale.is_finite() && q.scale > 0.0) {
        return Err(KernelError::InvalidArgument(
            "output scale must be finite and positive",
        ));
    }
    let inv_scale = 1.0f32 / q.scale;
    if !inv_scale.is_finite() {
        return Err(KernelError::InvalidArgument("output scale is too small"));
    }
    if !(lo..=hi).contains(&q.zero_point) {
        return Err(KernelError::InvalidArgument(
            "output zero point must lie in the output type's range",
        ));
    }
    Ok(OutQuant {
        inv_scale,
        zero_point: q.zero_point,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_rounds_half_away_and_clamps() {
        assert_eq!(quantize::<i8>(0.5, 1.0, 0), 1);
        assert_eq!(quantize::<i8>(-0.5, 1.0, 0), -1);
        assert_eq!(quantize::<i8>(1000.0, 1.0, 0), 127);
        assert_eq!(quantize::<i8>(-1000.0, 1.0, 0), -128);
        assert_eq!(quantize::<i8>(f32::NAN, 1.0, 5), 5);
        assert_eq!(quantize::<u16>(0.5, 1.0, 32768), 32769);
        assert_eq!(quantize::<u16>(-0.5, 1.0, 32768), 32767);
        assert_eq!(quantize::<u16>(1.0e9, 1.0, 32768), 65535);
        assert_eq!(quantize::<u16>(-1.0e9, 1.0, 32768), 0);
        assert_eq!(quantize::<u16>(f32::NAN, 1.0, 32768), 32768);
    }
}
