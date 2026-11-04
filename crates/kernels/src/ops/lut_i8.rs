// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! A 256-entry byte table applied elementwise to int8 or uint8 tensors.

use super::rows::{for_each_chunk, for_each_transposed_tile, for_each_unit_run};
use crate::cpu::{CpuFeatures, Tier};
use crate::error::{KernelError, Result};
use crate::pool::Pool;
use crate::quant::{DType, Quant};
use crate::view::{TensorMut, TensorRef, View, ViewMut};

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
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    window: Window,
    /// Every entry maps to itself: runs copy instead of looking up.
    identity: bool,
    kernel: Kernel,
}

/// The varying part of a byte table, in value order (`i8` values from −128, `u8` from 0):
/// entries are constant below `lo` and above `hi`, so a lookup may clamp into `lo..=hi` and
/// read `table[v − lo]`. Saturating activations (logistic, tanh) vary over a narrow window.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#[derive(Clone, Debug)]
pub(super) struct Window {
    /// `table[i]` is the entry of value-order position `lo + i`; past `hi − lo` it repeats
    /// the entry at `hi`.
    pub(super) table: [u8; 256],
    /// XORed into a byte to get its value-order position: `0x80` for `i8`, 0 for `u8`.
    pub(super) flip: u8,
    pub(super) lo: u8,
    pub(super) hi: u8,
}

#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
impl Window {
    pub(super) fn new(table: &[u8; 256], dtype: DType) -> Self {
        let flip = if dtype == DType::I8 { 0x80u8 } else { 0 };
        let at = |v: usize| table[(v as u8 ^ flip) as usize];
        let lo = (0..256).take_while(|&v| at(v) == at(0)).last().unwrap_or(0);
        let hi = (0..256)
            .rev()
            .take_while(|&v| at(v) == at(255))
            .last()
            .unwrap_or(255);
        let (lo, hi) = if lo > hi { (0, 0) } else { (lo, hi) };
        let table = core::array::from_fn(|i| at((lo + i).min(hi)));
        Self {
            table,
            flip,
            lo: lo as u8,
            hi: hi as u8,
        }
    }

    /// 64-byte blocks the window spans, 1 to 4.
    pub(super) fn blocks(&self) -> usize {
        (usize::from(self.hi - self.lo) + 1).div_ceil(64)
    }

    /// The reference lookup through the window over `min(src.len(), dst.len())` elements.
    pub(super) fn lookup_portable<T: Byte>(&self, src: &[T], dst: &mut [T]) {
        for (d, &s) in dst.iter_mut().zip(src) {
            let v = (s.to_byte() ^ self.flip).clamp(self.lo, self.hi);
            *d = T::from_byte(self.table[usize::from(v - self.lo)]);
        }
    }
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
            window: Window::new(&table, dtype),
            identity: table.iter().enumerate().all(|(i, &t)| usize::from(t) == i),
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
            // A table lookup does not depend on position, so transposed views can go by tiles.
            (DType::I8, TensorRef::I8(input), TensorMut::I8(mut output)) => {
                if let Some(r) = self.transposed(&input, &mut output, pool) {
                    return r;
                }
                let f = |s: &[i8], d: &mut [i8]| self.apply(s, d);
                lut_walk(&input, &mut output, pool, &f)
            }
            (DType::U8, TensorRef::U8(input), TensorMut::U8(mut output)) => {
                if let Some(r) = self.transposed(&input, &mut output, pool) {
                    return r;
                }
                let f = |s: &[u8], d: &mut [u8]| self.apply(s, d);
                lut_walk(&input, &mut output, pool, &f)
            }
            _ => Err(KernelError::InvalidArgument(
                "lut_i8 input and output must both have the plan's dtype",
            )),
        }
    }

    /// Transposed views on NEON: 16×16 tiles transposed and looked up in registers; `None`,
    /// touching nothing, when the views or the kernel do not suit it.
    #[cfg_attr(not(target_arch = "aarch64"), allow(unused_variables))]
    fn transposed<T: Byte + Send + Sync>(
        &self,
        input: &View<'_, T>,
        output: &mut ViewMut<'_, T>,
        pool: Option<&dyn Pool>,
    ) -> Option<Result<()>> {
        #[cfg(target_arch = "aarch64")]
        if self.kernel == Kernel::Neon
            && (input.as_contiguous().is_none() || output.as_contiguous_mut().is_none())
        {
            if self.identity {
                return super::rows::for_each_transposed_vector(input, output, pool, &|x| x);
            }
            use super::lut_i8_neon::lut_window_transposed;
            let w = &self.window;
            // SAFETY: NEON is part of the aarch64 baseline; `B` is the window's block count.
            return unsafe {
                match w.blocks() {
                    1 => lut_window_transposed::<1, T>(w, input, output, pool),
                    2 => lut_window_transposed::<2, T>(w, input, output, pool),
                    3 => lut_window_transposed::<3, T>(w, input, output, pool),
                    _ => lut_window_transposed::<4, T>(w, input, output, pool),
                }
            };
        }
        None
    }

    #[inline]
    fn apply<T: Byte>(&self, src: &[T], dst: &mut [T]) {
        if self.identity {
            let n = src.len().min(dst.len());
            if let (Some(d), Some(s)) = (dst.get_mut(..n), src.get(..n)) {
                d.copy_from_slice(s);
            }
            return;
        }
        match self.kernel {
            Kernel::Portable => lut_portable(&self.table, src, dst),
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is part of the aarch64 baseline.
            Kernel::Neon => unsafe {
                use super::lut_i8_neon::{lut_neon, lut_window};
                match self.window.blocks() {
                    1 => lut_window::<1, T>(&self.window, src, dst),
                    2 => lut_window::<2, T>(&self.window, src, dst),
                    3 => lut_window::<3, T>(&self.window, src, dst),
                    _ => lut_neon(&self.table, src, dst),
                }
            },
        }
    }
}

/// Walks a position-independent elementwise kernel: dense views in big chunks, views sharing a
/// contiguous axis by runs along it, transposed views by tiles, anything else element by
/// element.
fn lut_walk<T: Copy + Default + Send + Sync>(
    input: &View<'_, T>,
    output: &mut ViewMut<'_, T>,
    pool: Option<&dyn Pool>,
    f: &(dyn Fn(&[T], &mut [T]) + Sync),
) -> Result<()> {
    if input.as_contiguous().is_some() && output.as_contiguous_mut().is_some() {
        return for_each_chunk(input, output, pool, f);
    }
    for_each_unit_run(input, output, pool, f)
        .or_else(|| for_each_transposed_tile(input, output, pool, f))
        .unwrap_or_else(|| for_each_chunk(input, output, pool, f))
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

    /// Runs along an axis contiguous in both views, writing rows of a wider tensor.
    #[test]
    fn shared_unit_axis_runs_match() {
        let table: [u8; 256] = core::array::from_fn(|i| (i as u8) ^ 0x5a);
        let plan = LutI8Plan::from_table(DType::I8, table).unwrap();
        let (rows, len, wide, at) = (80usize, 1600usize, 8400usize, 6400usize);
        let src: Vec<i8> = (0..rows * len).map(|i| (i * 31 % 253) as i8).collect();
        let mut out = vec![7i8; rows * wide];
        let ov = ViewMut::strided(&mut out[at..], &[rows, len], &[wide as isize, 1]).unwrap();
        plan.run(
            TensorRef::I8(View::contiguous(&src, &[rows, len]).unwrap()),
            TensorMut::I8(ov),
            None,
        )
        .unwrap();
        for r in 0..rows {
            for c in 0..wide {
                let want = if (at..at + len).contains(&c) {
                    table[src[r * len + c - at] as u8 as usize] as i8
                } else {
                    7
                };
                assert_eq!(out[r * wide + c], want, "{r},{c}");
            }
        }
    }

    /// Four channels of a 16-channel `[80, 80, 16]` tensor written channel-major (a YOLO box
    /// head): the two spatial axes merge on both sides into one 6400-long transpose of 4-byte runs.
    #[test]
    fn channel_slices_transpose_by_runs() {
        let table: [u8; 256] =
            core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
        let identity: [u8; 256] = core::array::from_fn(|i| i as u8);
        let src: Vec<i8> = (0..80 * 80 * 16).map(|i| (i * 7919 % 251) as i8).collect();
        for (t, (w, ch)) in [table, identity]
            .into_iter()
            .flat_map(|t| [(t, (4usize, 3usize)), (t, (2, 9))])
        {
            let plan = LutI8Plan::from_table(DType::I8, t).unwrap();
            let iv = View::strided(&src[ch..], &[1, 80, 80, w], &[0, 1280, 16, 1]).unwrap();
            let mut got = vec![0x55i8; w * 6400];
            plan.run(
                TensorRef::I8(iv),
                TensorMut::I8(
                    ViewMut::strided(&mut got, &[1, 80, 80, w], &[0, 80, 1, 6400]).unwrap(),
                ),
                None,
            )
            .unwrap();
            for c in 0..w {
                for p in 0..6400 {
                    let v = src[ch + p * 16 + c];
                    assert_eq!(
                        got[c * 6400 + p],
                        t[v as u8 as usize] as i8,
                        "w {w} c {c} p {p}"
                    );
                }
            }
        }
    }

    /// Transposed views go by tiles; the result must equal the element-wise walk.
    #[test]
    fn transposed_views_match_contiguous() {
        let spread: [u8; 256] =
            core::array::from_fn(|i| (i as u8).wrapping_mul(37).wrapping_add(11));
        // Constant outside a 51-value window in value order: the one-block window path.
        let saturating: [u8; 256] =
            core::array::from_fn(|i| ((i ^ 0x80).saturating_sub(100).min(50) * 5) as u8);
        // The identity: the copy path.
        let identity: [u8; 256] = core::array::from_fn(|i| i as u8);
        for (table, &(a, b, c)) in [spread, saturating, identity].into_iter().flat_map(|t| {
            [
                (1usize, 37usize, 50usize),
                (3, 16, 16),
                (2, 80, 6400),
                (1, 17, 15),
                (1, 48, 130),
                // Input runs of four and two bytes, and one whose output axis is not 16-aligned.
                (1, 4, 6400),
                (3, 2, 48),
                (2, 4, 40),
            ]
            .iter()
            .map(move |s| (t, s))
            .collect::<Vec<_>>()
        }) {
            let plan = LutI8Plan::from_table(DType::I8, table).unwrap();
            let n = a * b * c;
            let src: Vec<i8> = (0..n).map(|i| (i * 7919 % 251) as i8).collect();
            // The input is [a, c, b] in memory, read as [a, b, c].
            let iv = View::strided(&src, &[a, b, c], &[(b * c) as isize, 1, b as isize]).unwrap();
            let mut got = vec![0i8; n];
            plan.run(
                TensorRef::I8(iv),
                TensorMut::I8(ViewMut::contiguous(&mut got, &[a, b, c]).unwrap()),
                Some(&crate::ScopedThreads::new(
                    core::num::NonZeroUsize::new(3).unwrap(),
                )),
            )
            .unwrap();
            for x in 0..a {
                for y in 0..b {
                    for z in 0..c {
                        let v = src[x * b * c + z * b + y];
                        assert_eq!(
                            got[(x * b + y) * c + z],
                            table[v as u8 as usize] as i8,
                            "{a}x{b}x{c} at {x},{y},{z}"
                        );
                    }
                }
            }
        }
    }

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
