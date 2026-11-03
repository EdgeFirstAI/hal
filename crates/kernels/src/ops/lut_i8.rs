// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! A 256-entry byte table applied elementwise to int8 or uint8 tensors.

use super::rows::for_each_chunk;
use crate::cpu::{CpuFeatures, Tier};
use crate::error::{KernelError, Result};
use crate::pool::Pool;
use crate::quant::{DType, Quant};
use crate::view::{TensorMut, TensorRef};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    Portable,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// A byte type whose raw value indexes the table.
pub(super) trait Byte: Copy {
    fn to_byte(self) -> u8;
    fn from_byte(b: u8) -> Self;
}

impl Byte for u8 {
    #[inline(always)]
    fn to_byte(self) -> u8 {
        self
    }
    #[inline(always)]
    fn from_byte(b: u8) -> Self {
        b
    }
}

impl Byte for i8 {
    #[inline(always)]
    fn to_byte(self) -> u8 {
        self as u8
    }
    #[inline(always)]
    fn from_byte(b: u8) -> Self {
        b as i8
    }
}

/// An elementwise int8 or uint8 table lookup: `out = table[in as u8]`, the stored byte
/// reinterpreted as the tensor's type.
///
/// The table is indexed by the raw byte, as TFLite's `LookupTable` does, so an int8 table is
/// ordered `0, 1, …, 127, −128, …, −1`. Input and output share the plan's type.
///
/// ```
/// use edgefirst_kernels::{DType, LutI8Plan, TensorMut, TensorRef, View, ViewMut};
///
/// let table: [u8; 256] = core::array::from_fn(|i| (i as u8).wrapping_neg());
/// let plan = LutI8Plan::from_table(DType::I8, table)?;
/// let input = [1i8, -2, 0, 127];
/// let mut output = [0i8; 4];
/// plan.run(
///     TensorRef::I8(View::contiguous(&input, &[4])?),
///     TensorMut::I8(ViewMut::contiguous(&mut output, &[4])?),
///     None,
/// )?;
/// assert_eq!(output, [-1, 2, 0, -127]);
/// # Ok::<(), edgefirst_kernels::KernelError>(())
/// ```
#[derive(Clone, Debug)]
pub struct LutI8Plan {
    dtype: DType,
    table: [u8; 256],
    kernel: Kernel,
}

fn finite_positive(v: f32) -> bool {
    v.is_finite() && v > 0.0
}

/// `T`'s range for an int8 or uint8 `dtype`.
fn byte_range(dtype: DType) -> Option<(i32, i32)> {
    match dtype {
        DType::I8 => Some((i32::from(i8::MIN), i32::from(i8::MAX))),
        DType::U8 => Some((0, i32::from(u8::MAX))),
        _ => None,
    }
}

impl LutI8Plan {
    /// A plan applying `table`, for the CPU this process runs on (honouring
    /// `EDGEFIRST_KERNELS_FORCE_TIER`). `dtype` must be `I8` or `U8`.
    pub fn from_table(dtype: DType, table: [u8; 256]) -> Result<Self> {
        Self::from_table_with_features(dtype, table, CpuFeatures::from_env_or_probe()?)
    }

    /// [`LutI8Plan::from_table`] running the best kernel `f` allows; reads no environment.
    pub fn from_table_with_features(
        dtype: DType,
        table: [u8; 256],
        f: CpuFeatures,
    ) -> Result<Self> {
        if byte_range(dtype).is_none() {
            return Err(KernelError::InvalidArgument(
                "lut_i8 needs an i8 or u8 dtype",
            ));
        }
        Ok(Self {
            dtype,
            table,
            kernel: select_kernel(f),
        })
    }

    /// The sigmoid table TFLite's int8/uint8 `LOGISTIC` builds (`LUTPopulateInt8`), for the CPU
    /// this process runs on (honouring `EDGEFIRST_KERNELS_FORCE_TIER`).
    ///
    /// Each entry is `clamp(round(sigmoid(in_scale·(v − in_zp)) · (1/out_scale)) + out_zp)` in
    /// f32 with `expf`, rounding half away from zero, so the output is bit-identical to TFLite
    /// where `f32::exp` is glibc's `expf`, as on the Linux targets.
    /// Unlike TFLite, any output quantization is accepted, not only scale 1/256. Scales must be
    /// finite and positive, and zero points inside the type's range.
    pub fn sigmoid(dtype: DType, input: Quant, output: Quant) -> Result<Self> {
        Self::sigmoid_with_features(dtype, input, output, CpuFeatures::from_env_or_probe()?)
    }

    /// [`LutI8Plan::sigmoid`] running the best kernel `f` allows; reads no environment.
    pub fn sigmoid_with_features(
        dtype: DType,
        input: Quant,
        output: Quant,
        f: CpuFeatures,
    ) -> Result<Self> {
        let Some((min, max)) = byte_range(dtype) else {
            return Err(KernelError::InvalidArgument(
                "lut_i8 sigmoid needs an i8 or u8 dtype",
            ));
        };
        if !finite_positive(input.scale) || !finite_positive(output.scale) {
            return Err(KernelError::InvalidArgument(
                "lut_i8 sigmoid scales must be finite and positive",
            ));
        }
        if !(min..=max).contains(&input.zero_point) || !(min..=max).contains(&output.zero_point) {
            return Err(KernelError::InvalidArgument(
                "lut_i8 sigmoid zero points must lie in the type's range",
            ));
        }
        let inverse_scale = 1.0f32 / output.scale;
        if !inverse_scale.is_finite() {
            return Err(KernelError::InvalidArgument(
                "lut_i8 sigmoid output scale is too small",
            ));
        }
        let mut table = [0u8; 256];
        for val in min..=max {
            let dequantized = input.scale * (val - input.zero_point) as f32;
            let transformed = 1.0f32 / (1.0f32 + (-dequantized).exp());
            let rescaled = (transformed * inverse_scale).round();
            let quantized = (rescaled + output.zero_point as f32) as i32;
            let q = quantized.clamp(min, max);
            // `val as u8` and `q as u8` are the raw bytes of the int8 or uint8 values.
            if let Some(e) = table.get_mut(usize::from(val as u8)) {
                *e = q as u8;
            }
        }
        Self::from_table_with_features(dtype, table, f)
    }

    /// The tier of the kernel this plan runs; every NEON tier runs the baseline NEON kernel.
    pub fn tier(&self) -> Tier {
        match self.kernel {
            Kernel::Portable => Tier::Scalar,
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => Tier::Neon,
        }
    }

    /// Maps `input` through the table into `output`: same dtype as the plan, same shape (any
    /// rank). Work is split across `pool` when one is given. Output axes sorted by stride must
    /// each step past the extent of the smaller ones. Arguments are validated before anything
    /// is written.
    pub fn run(
        &self,
        input: TensorRef<'_>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
    ) -> Result<()> {
        let _span = tracing::trace_span!("kernels.lut_i8", tier = self.tier().as_str()).entered();
        self.run_inner(input, output, pool)
    }

    /// `run` without the span, for plans that embed this one.
    pub(super) fn run_inner(
        &self,
        input: TensorRef<'_>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
    ) -> Result<()> {
        match (self.dtype, input, output) {
            (DType::I8, TensorRef::I8(input), TensorMut::I8(mut output)) => {
                for_each_chunk(&input, &mut output, pool, &|s, d| self.apply(s, d))
            }
            (DType::U8, TensorRef::U8(input), TensorMut::U8(mut output)) => {
                for_each_chunk(&input, &mut output, pool, &|s, d| self.apply(s, d))
            }
            _ => Err(KernelError::InvalidArgument(
                "lut_i8 input and output must both have the plan's dtype",
            )),
        }
    }

    #[inline]
    fn apply<T: Byte>(&self, src: &[T], dst: &mut [T]) {
        match self.kernel {
            Kernel::Portable => lut_portable(&self.table, src, dst),
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is part of the aarch64 baseline.
            Kernel::Neon => unsafe { super::lut_i8_neon::lut_neon(&self.table, src, dst) },
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

/// The reference lookup over `min(src.len(), dst.len())` elements.
#[inline]
pub(super) fn lut_portable<T: Byte>(table: &[u8; 256], src: &[T], dst: &mut [T]) {
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = T::from_byte(table[usize::from(s.to_byte())]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{RngExt, SeedableRng};

    /// The NEON kernel's `tbl`/`tbx` sequence, lane by lane, per the Arm pseudocode.
    fn model_lut(table: &[u8; 256], x: u8) -> u8 {
        let tbl = |r: Option<u8>, block: usize, i: u8| -> Option<u8> {
            if usize::from(i) < 64 {
                Some(table[block * 64 + usize::from(i)])
            } else {
                r
            }
        };
        let r = tbl(Some(0), 0, x);
        let r = tbl(r, 1, x.wrapping_sub(64));
        let r = tbl(r, 2, x.wrapping_sub(128));
        tbl(r, 3, x.wrapping_sub(192)).unwrap()
    }

    #[test]
    fn neon_model_matches_portable_exhaustively() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0x0071_7b10);
        for _ in 0..16 {
            let table: [u8; 256] = core::array::from_fn(|_| rng.random());
            let all: Vec<u8> = (0..=255).collect();
            let mut want = vec![0u8; 256];
            lut_portable(&table, &all, &mut want);
            for (&x, &w) in all.iter().zip(&want) {
                assert_eq!(model_lut(&table, x), w, "byte {x}");
            }
        }
    }

    #[test]
    fn sigmoid_table_matches_tflite_formula_and_indexing() {
        let plan = LutI8Plan::sigmoid(
            DType::I8,
            Quant::new(0.1, -30),
            Quant::new(1.0 / 256.0, -128),
        )
        .unwrap();
        // Raw byte 0x80 is -128: sigmoid(0.1·(-98)) ≈ 5.5e-5 rounds to 0, then -128.
        assert_eq!(plan.table[0x80] as i8, -128);
        // v = -30 is the zero point: sigmoid(0) · 256 = 128, then 0.
        assert_eq!(plan.table[usize::from(-30i8 as u8)] as i8, 0);
        assert_eq!(plan.table[127] as i8, 127);
        let u = LutI8Plan::sigmoid(DType::U8, Quant::new(0.1, 100), Quant::new(1.0 / 256.0, 0))
            .unwrap();
        assert_eq!(u.table[100], 128);
        assert_eq!(u.table[255], 255);
    }
}
