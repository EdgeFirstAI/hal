// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! int8 `SOFTMAX` over the last axis, matching TFLite's float-table optimized kernel.

use std::borrow::Cow;

use super::rows::{for_each_row, StridedRow, StridedRowMut};
#[cfg(target_arch = "aarch64")]
use super::rows::{for_each_row_group, RowGroup, ROW_GROUP};
use crate::cpu::{CpuFeatures, Tier};
use crate::error::{KernelError, Result};
use crate::pool::Pool;
use crate::quant::Quant;
use crate::view::{TensorMut, TensorRef};

/// Parameters of an int8 softmax.
///
/// The output must be quantized with scale 1/256 (within 0.1%) and zero point −128, as TFLite
/// requires for int8 `SOFTMAX`. The input zero point is unused: softmax is shift-invariant.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftmaxI8Params {
    pub input: Quant,
    pub output: Quant,
    pub beta: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    Portable,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// A validated int8 softmax over the last axis; every other axis indexes independent rows.
///
/// Reproduces TFLite's `optimized_ops::Softmax<int8_t, int8_t>`: the scalar tier bit for bit
/// where `f32::exp` is glibc's `expf` (as on the Linux targets; another libm can move a table
/// entry by an ulp), the NEON tiers within ±1 LSB (they sum the row in a different order).
///
/// ```
/// use edgefirst_kernels::{Quant, SoftmaxI8Params, SoftmaxI8Plan, TensorMut, TensorRef, View, ViewMut};
///
/// let plan = SoftmaxI8Plan::new(SoftmaxI8Params {
///     input: Quant::new(0.1, 0),
///     output: Quant::new(1.0 / 256.0, -128),
///     beta: 1.0,
/// })?;
/// let input = [0i8, 0, 0, 0];
/// let mut output = [0i8; 4];
/// plan.run(
///     TensorRef::I8(View::contiguous(&input, &[1, 4])?),
///     TensorMut::I8(ViewMut::contiguous(&mut output, &[1, 4])?),
///     None,
/// )?;
/// assert_eq!(output, [-64; 4]);
/// # Ok::<(), edgefirst_kernels::KernelError>(())
/// ```
#[derive(Clone, Debug)]
pub struct SoftmaxI8Plan {
    /// `exp(scale·d)` at index `255 − d`, `scale` being `−input_scale·beta`.
    pub(super) table: [f32; 256],
    /// `exp(scale·d)` for `d = 16·hi + lo` as `A[hi]·B[lo]` in 16-bit fixed point: the byte
    /// planes `A` low, `A` high, `B` low, `B` high, for the 16-row NEON kernel.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    exp16: [[u8; 16]; 4],
    /// The largest entry of each 64-entry slice of `table`.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    slice_max: [f32; 4],
    output_scale: f32,
    output_zp: i32,
    kernel: Kernel,
}

fn finite_positive(v: f32) -> bool {
    v.is_finite() && v > 0.0
}

impl SoftmaxI8Plan {
    /// Builds a plan for the CPU this process runs on, honouring `EDGEFIRST_KERNELS_FORCE_TIER`.
    pub fn new(p: SoftmaxI8Params) -> Result<Self> {
        Self::with_features(p, CpuFeatures::from_env_or_probe()?)
    }

    /// Builds a plan that runs the best kernel `f` allows.
    pub fn with_features(p: SoftmaxI8Params, f: CpuFeatures) -> Result<Self> {
        if !finite_positive(p.input.scale) {
            return Err(KernelError::InvalidArgument(
                "softmax_i8 input scale must be finite and positive",
            ));
        }
        if !finite_positive(p.output.scale) {
            return Err(KernelError::InvalidArgument(
                "softmax_i8 output scale must be finite and positive",
            ));
        }
        if !finite_positive(p.beta) {
            return Err(KernelError::InvalidArgument(
                "softmax_i8 beta must be finite and positive",
            ));
        }
        let scale = -p.input.scale * p.beta;
        if !scale.is_finite() {
            return Err(KernelError::InvalidArgument(
                "softmax_i8 input scale times beta overflows",
            ));
        }
        if p.output.zero_point != -128 {
            return Err(KernelError::Unsupported(Cow::Borrowed(
                "softmax_i8 needs output zero point -128",
            )));
        }
        let want = 1.0f32 / 256.0;
        if (p.output.scale - want).abs() > 0.001f32 * 1.0 / 256.0 {
            return Err(KernelError::Unsupported(Cow::Borrowed(
                "softmax_i8 needs output scale 1/256",
            )));
        }
        let mut table = [0.0f32; 256];
        for (v, t) in table.iter_mut().rev().enumerate() {
            *t = (scale * v as f32).exp();
        }
        let exp16 = exp16_tables(scale);
        let slice_max = slice_max(&table);
        Ok(Self {
            table,
            exp16,
            slice_max,
            output_scale: p.output.scale,
            output_zp: p.output.zero_point,
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

    /// Softmax of `input` into `output`, both i8 with the same shape of rank ≥ 1.
    ///
    /// Rows are split across `pool` when one is given. Output axes sorted by stride must each
    /// step past the extent of the smaller ones (which rules out aliasing, and also some
    /// interleaved layouts). Arguments are validated before anything is written.
    /// Strided rows longer than the 4096-element staging buffer use the
    /// portable kernel, so on NEON tiers they may differ from a contiguous run by 1 LSB.
    pub fn run(
        &self,
        input: TensorRef<'_>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
    ) -> Result<()> {
        let _span =
            tracing::trace_span!("kernels.softmax_i8", tier = self.tier().as_str()).entered();
        let (TensorRef::I8(input), TensorMut::I8(mut output)) = (input, output) else {
            return Err(KernelError::InvalidArgument(
                "softmax_i8 needs i8 input and output",
            ));
        };
        if input.shape() != output.shape() {
            return Err(KernelError::InvalidArgument(
                "softmax_i8 input and output shapes differ",
            ));
        }
        #[cfg(target_arch = "aarch64")]
        if self.kernel == Kernel::Neon && input.shape().last() == Some(&ROW_GROUP) {
            let group = |g: &mut RowGroup<'_, i8>| {
                if g.count() == ROW_GROUP {
                    // SAFETY: NEON is part of the aarch64 baseline; rows are 16 long.
                    unsafe {
                        super::softmax_i8_neon::rows16(&self.exp16, self.output_scale, g);
                    }
                    return;
                }
                for k in 0..g.count() {
                    let s = g.input(k);
                    self.row(s, g.output(k));
                }
            };
            if let Some(r) = for_each_row_group(&input, &mut output, pool, &group) {
                return r;
            }
        }
        for_each_row(
            &input,
            &mut output,
            pool,
            &|s, d| self.row(s, d),
            &|s, d| self.row_strided(s, d),
        )
    }

    #[inline]
    pub(super) fn row(&self, src: &[i8], dst: &mut [i8]) {
        match self.kernel {
            Kernel::Portable => {
                softmax_row_i8_portable(&self.table, self.output_scale, self.output_zp, src, dst)
            }
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is part of the aarch64 baseline.
            Kernel::Neon => unsafe {
                super::softmax_i8_neon::softmax_row_i8_neon(
                    &self.table,
                    &self.slice_max,
                    self.output_scale,
                    self.output_zp,
                    src,
                    dst,
                )
            },
        }
    }

    fn row_strided(&self, src: StridedRow<'_, i8>, dst: &mut StridedRowMut<'_, i8>) {
        if src.len() != dst.len() {
            return;
        }
        let mut k = 0;
        softmax_row_generic(
            &self.table,
            self.output_scale,
            self.output_zp,
            src.iter(),
            |q| {
                dst.set(k, q);
                k += 1;
            },
        );
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

/// The largest entry of each 64-entry slice of `table`.
pub(super) fn slice_max(table: &[f32; 256]) -> [f32; 4] {
    let (slices, _) = table.as_chunks::<64>();
    core::array::from_fn(|k| {
        slices
            .get(k)
            .map_or(0.0, |s| s.iter().fold(0.0f32, |m, &t| m.max(t)))
    })
}

/// `exp(scale·d)` for `d = 16·hi + lo` as `A[hi]·B[lo]`, each rounded to 16-bit fixed point
/// (65535 is 1), split into byte planes: `A` low, `A` high, `B` low, `B` high. `scale` is
/// `−input_scale·beta`.
pub(super) fn exp16_tables(scale: f32) -> [[u8; 16]; 4] {
    split_exp_tables(scale, 65535.0)
}

/// `exp(scale·d)` for `d = 16·hi + lo` (`hi`, `lo` below 16) as `A[hi]` and `B[lo]`, each
/// `exp` times `one` rounded to a u16, split into byte planes: `A` low, `A` high, `B` low,
/// `B` high.
pub(super) fn split_exp_tables(scale: f32, one: f64) -> [[u8; 16]; 4] {
    let a: [u16; 16] =
        core::array::from_fn(|i| ((f64::from(scale) * 16.0 * i as f64).exp() * one).round() as u16);
    let b: [u16; 16] =
        core::array::from_fn(|i| ((f64::from(scale) * i as f64).exp() * one).round() as u16);
    [
        a.map(|v| v as u8),
        a.map(|v| (v >> 8) as u8),
        b.map(|v| v as u8),
        b.map(|v| (v >> 8) as u8),
    ]
}

/// The table offset `255 - max` as a byte, so `x as u8 + offset` (wrapping) is `255 - max + x`.
#[inline(always)]
pub(super) fn table_offset(max: i8) -> u8 {
    255u8.wrapping_sub(max as u8)
}

/// Table index of `x` in a row whose offset is `off`; always in `0..=255`.
#[inline(always)]
pub(super) fn table_index(x: i8, off: u8) -> usize {
    usize::from((x as u8).wrapping_add(off))
}

/// `clamp(round_half_away(p) + zp)` to i8, as TFLite's `QuantizeSoftmaxOutput` then clamp.
#[inline(always)]
pub(super) fn quantize_prob(p: f32, zp: i32) -> i8 {
    (p.round() as i32).saturating_add(zp).clamp(-128, 127) as i8
}

#[inline(always)]
fn softmax_row_generic<I: Iterator<Item = i8> + Clone>(
    table: &[f32; 256],
    out_scale: f32,
    zp: i32,
    src: I,
    mut emit: impl FnMut(i8),
) {
    let off = table_offset(src.clone().max().unwrap_or(i8::MIN));
    let mut sum = 0.0f32;
    for x in src.clone() {
        sum += table[table_index(x, off)];
    }
    let inv = 1.0 / (sum * out_scale);
    for x in src {
        emit(quantize_prob(table[table_index(x, off)] * inv, zp));
    }
}

/// The bit-exact reference row: sequential f32 sum in element order, as TFLite does.
///
/// Processes `min(src.len(), dst.len())` elements.
pub(crate) fn softmax_row_i8_portable(
    table: &[f32; 256],
    out_scale: f32,
    zp: i32,
    src: &[i8],
    dst: &mut [i8],
) {
    let src = src.get(..dst.len()).unwrap_or(src);
    let mut out = dst.iter_mut();
    softmax_row_generic(table, out_scale, zp, src.iter().copied(), |q| {
        if let Some(d) = out.next() {
            *d = q;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_index_spans_the_table() {
        assert_eq!(table_index(-128, table_offset(127)), 0);
        assert_eq!(table_index(127, table_offset(127)), 255);
        assert_eq!(table_index(-128, table_offset(-128)), 255);
        assert_eq!(table_index(0, table_offset(5)), 250);
    }

    #[test]
    fn quantize_prob_rounds_half_away_and_clamps() {
        assert_eq!(quantize_prob(0.5, -128), -127);
        assert_eq!(quantize_prob(0.49999997, -128), -128);
        assert_eq!(quantize_prob(256.0, -128), 127);
        assert_eq!(quantize_prob(127.5, -128), 0);
    }
}
