// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! int8 matrix multiplication with TFLite's integer requantization: `BATCH_MATMUL`,
//! `FULLY_CONNECTED` and 1×1 `CONV_2D`.

use std::borrow::Cow;

use crate::cpu::{CpuFeatures, Tier};
use crate::error::{KernelError, Result};
use crate::pool::{for_each_block, Pool};
use crate::quant::{Activation, Quant, Rounding, TfliteKernels};
use crate::tflite::{activation_range_i8, multiply_by_quantized_multiplier, quantize_multiplier};
use crate::view::{TensorMut, TensorRef, View, ViewMut, MAX_RANK};

/// Longest reduction axis: operands not contiguous along it are packed into stack buffers of
/// this many elements per row, and `255 · 255 · MAX_K` stays inside an `i32`.
pub(crate) const MATMUL_MAX_K: usize = 4096;

/// Rows (and columns) of an output tile.
pub(super) const TILE: usize = 4;

/// A TFLite fixed-point requantization: `MultiplyByQuantizedMultiplier(acc, multiplier, shift)`,
/// that is `acc · multiplier · 2^(shift − 31)`, rounded as the plan's [`TfliteKernels`] do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(C)]
pub struct Requant {
    pub multiplier: i32,
    pub shift: i32,
}

impl Requant {
    /// `QuantizeMultiplier(real)`; `real` must be finite and positive.
    pub fn from_real(real: f64) -> Result<Self> {
        if !(real.is_finite() && real > 0.0) {
            return Err(KernelError::InvalidArgument(
                "requantization multiplier must be finite and positive",
            ));
        }
        let (multiplier, shift) = quantize_multiplier(real);
        Ok(Self { multiplier, shift })
    }

    /// The multiplier of TFLite's int8 `BATCH_MATMUL` (and per-tensor `FULLY_CONNECTED`):
    /// `GetQuantizedConvolutionMultipler`, `(lhs_scale · rhs_scale)` in f32, then divided by
    /// `output_scale` in f64.
    pub fn tflite_batch_matmul(lhs_scale: f32, rhs_scale: f32, output_scale: f32) -> Result<Self> {
        Self::from_real(f64::from(lhs_scale * rhs_scale) / f64::from(output_scale))
    }

    /// The multiplier of one output channel of TFLite's int8 `CONV_2D` and
    /// `DEPTHWISE_CONV_2D` (`PopulateConvolutionQuantizationParams`): the whole product in f64.
    pub fn tflite_conv(input_scale: f32, filter_scale: f32, output_scale: f32) -> Result<Self> {
        Self::from_real(f64::from(input_scale) * f64::from(filter_scale) / f64::from(output_scale))
    }

    /// Whether `rounding` can apply this requantization exactly.
    pub(super) fn valid_for(self, rounding: Rounding) -> bool {
        self.multiplier >= 0
            && match rounding {
                Rounding::Ruy => (-31..=-1).contains(&self.shift),
                Rounding::Reference => (-31..=31).contains(&self.shift),
            }
    }

    #[inline(always)]
    pub(super) fn apply(self, acc: i32, rounding: Rounding) -> i32 {
        match rounding {
            Rounding::Reference => {
                multiply_by_quantized_multiplier(acc, self.multiplier, self.shift)
            }
            Rounding::Ruy => {
                let total = (31 - self.shift).clamp(1, 62);
                ((i64::from(acc) * i64::from(self.multiplier) + (1i64 << (total - 1))) >> total)
                    as i32
            }
        }
    }
}

/// Parameters of an int8 matrix multiplication
/// `out[.., m, n] = clamp(zo + R[n](Σ_k (lhs[.., m, k] − zl)·(rhs[.., k, n] − zr) + bias[n]))`.
///
/// Build them with [`MatMulI8Params::tflite_batch_matmul`] or [`MatMulI8Params::tflite_conv`]
/// to reproduce those TFLite operators, or fill the fields directly.
#[derive(Clone, Debug, PartialEq)]
pub struct MatMulI8Params {
    pub lhs_zero_point: i32,
    pub rhs_zero_point: i32,
    /// The output zero point and scale; the scale only places the activation clamp.
    pub output: Quant,
    /// One requantization for every column, or one per output column.
    pub requant: Vec<Requant>,
    /// Added to each column's accumulator before requantization; empty for none.
    pub bias: Vec<i32>,
    pub activation: Activation,
    /// The TFLite kernels whose requantization rounding to reproduce.
    pub kernels: TfliteKernels,
}

impl MatMulI8Params {
    /// TFLite's int8 `BATCH_MATMUL`: no bias or activation. `adj_x` / `adj_y` are views with
    /// the last two axes' strides swapped.
    pub fn tflite_batch_matmul(
        lhs: Quant,
        rhs: Quant,
        output: Quant,
        kernels: TfliteKernels,
    ) -> Result<Self> {
        Ok(Self {
            lhs_zero_point: lhs.zero_point,
            rhs_zero_point: rhs.zero_point,
            output,
            requant: vec![Requant::tflite_batch_matmul(
                lhs.scale,
                rhs.scale,
                output.scale,
            )?],
            bias: Vec::new(),
            activation: Activation::None,
            kernels,
        })
    }

    /// A TFLite int8 `CONV_2D` that is a matrix multiplication (1×1 kernel, stride 1, no
    /// padding): `lhs` is the input `[pixels, in_channels]`, `rhs` the filter as
    /// `[in_channels, out_channels]` (zero point 0), one filter scale per output channel or one
    /// for all, and `bias` empty or one per output channel.
    pub fn tflite_conv(
        input: Quant,
        filter_scales: &[f32],
        output: Quant,
        bias: Vec<i32>,
        activation: Activation,
        kernels: TfliteKernels,
    ) -> Result<Self> {
        let requant = filter_scales
            .iter()
            .map(|&s| Requant::tflite_conv(input.scale, s, output.scale))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            lhs_zero_point: input.zero_point,
            rhs_zero_point: 0,
            output,
            requant,
            bias,
            activation,
            kernels,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    Portable,
    #[cfg(target_arch = "aarch64")]
    Neon,
    #[cfg(target_arch = "aarch64")]
    NeonDotprod,
}

/// A validated int8 matrix multiplication over the last two axes; the other axes are batch
/// axes and must match (a broadcast operand is a view with zero batch strides).
///
/// Accumulation is exact, so every tier and pool gives the same bytes. The requantization
/// rounds as TFLite's optimized int8 kernels (ruy) or its reference kernels do, per
/// [`MatMulI8Params::kernels`].
///
/// ```
/// use edgefirst_kernels::{
///     MatMulI8Params, MatMulI8Plan, Quant, TensorMut, TensorRef, TfliteKernels, View, ViewMut,
/// };
///
/// let plan = MatMulI8Plan::new(MatMulI8Params::tflite_batch_matmul(
///     Quant::new(0.5, 0),
///     Quant::new(0.5, 0),
///     Quant::new(1.0, 0),
///     TfliteKernels::NeonRuy,
/// )?)?;
/// let lhs = [1i8, 2, 3, 4]; // [2, 2]
/// let rhs = [4i8, 8, 12, 16]; // [2, 2]
/// let mut out = [0i8; 4];
/// plan.run(
///     TensorRef::I8(View::contiguous(&lhs, &[2, 2])?),
///     TensorRef::I8(View::contiguous(&rhs, &[2, 2])?),
///     TensorMut::I8(ViewMut::contiguous(&mut out, &[2, 2])?),
///     None,
/// )?;
/// assert_eq!(out, [7, 10, 15, 22]);
/// # Ok::<(), edgefirst_kernels::KernelError>(())
/// ```
#[derive(Clone, Debug)]
pub struct MatMulI8Plan {
    pub(super) zl: i32,
    pub(super) zr: i32,
    pub(super) zo: i32,
    requant: Vec<Requant>,
    bias: Vec<i32>,
    pub(super) rounding: Rounding,
    pub(super) act_min: i32,
    pub(super) act_max: i32,
    kernel: Kernel,
}

/// The geometry of one call, validated.
#[derive(Clone, Copy, Debug)]
pub(super) struct Shape {
    pub(super) batch: [usize; MAX_RANK],
    pub(super) batch_rank: usize,
    pub(super) m: usize,
    pub(super) k: usize,
    pub(super) n: usize,
    /// Element strides: batch axes, then the two matrix axes.
    pub(super) ls: [isize; MAX_RANK],
    pub(super) rs: [isize; MAX_RANK],
    pub(super) os: [isize; MAX_RANK],
}

impl Shape {
    pub(super) fn batches(&self) -> usize {
        self.batch[..self.batch_rank].iter().product()
    }

    /// Element offsets of batch `b` in the three operands.
    pub(super) fn batch_offsets(&self, mut b: usize) -> (usize, usize, usize) {
        let (mut l, mut r, mut o) = (0usize, 0usize, 0usize);
        for ax in (0..self.batch_rank).rev() {
            let d = self.batch[ax].max(1);
            let i = b % d;
            b /= d;
            l += i * self.ls[ax].unsigned_abs();
            r += i * self.rs[ax].unsigned_abs();
            o += i * self.os[ax].unsigned_abs();
        }
        (l, r, o)
    }

    /// `(row, depth)` strides of the lhs, `(depth, column)` of the rhs, `(row, column)` of the
    /// output.
    pub(super) fn strides(&self) -> ((usize, usize), (usize, usize), (usize, usize)) {
        let r = self.batch_rank;
        let u = |s: &[isize; MAX_RANK], i: usize| s[i].unsigned_abs();
        (
            (u(&self.ls, r), u(&self.ls, r + 1)),
            (u(&self.rs, r), u(&self.rs, r + 1)),
            (u(&self.os, r), u(&self.os, r + 1)),
        )
    }
}

impl MatMulI8Plan {
    /// Builds a plan for the CPU this process runs on, honouring `EDGEFIRST_KERNELS_FORCE_TIER`.
    pub fn new(p: MatMulI8Params) -> Result<Self> {
        Self::with_features(p, CpuFeatures::from_env_or_probe()?)
    }

    /// Builds a plan that runs the best kernel `f` allows; reads no environment.
    pub fn with_features(p: MatMulI8Params, f: CpuFeatures) -> Result<Self> {
        for zp in [p.lhs_zero_point, p.rhs_zero_point, p.output.zero_point] {
            if !(-128..=127).contains(&zp) {
                return Err(KernelError::InvalidArgument(
                    "matmul_i8 zero points must lie in the int8 range",
                ));
            }
        }
        if !(p.output.scale.is_finite() && p.output.scale > 0.0) {
            return Err(KernelError::InvalidArgument(
                "matmul_i8 output scale must be finite and positive",
            ));
        }
        if p.requant.is_empty() {
            return Err(KernelError::InvalidArgument(
                "matmul_i8 needs at least one requantization",
            ));
        }
        if p.requant.iter().any(|r| !r.valid_for(Rounding::Reference)) {
            return Err(KernelError::InvalidArgument(
                "matmul_i8 requantization out of range",
            ));
        }
        if p.requant.iter().any(|r| !r.valid_for(p.kernels.rounding())) {
            return Err(KernelError::Unsupported(Cow::Borrowed(
                "matmul_i8 ruy rounding needs multipliers below 0.5",
            )));
        }
        let (act_min, act_max) =
            activation_range_i8(p.activation, p.output.scale, p.output.zero_point);
        if act_min > act_max {
            return Err(KernelError::Unsupported(Cow::Borrowed(
                "matmul_i8 activation range is empty for this output quantization",
            )));
        }
        Ok(Self {
            zl: p.lhs_zero_point,
            zr: p.rhs_zero_point,
            zo: p.output.zero_point,
            requant: p.requant,
            bias: p.bias,
            rounding: p.kernels.rounding(),
            act_min,
            act_max,
            kernel: select_kernel(f),
        })
    }

    /// `Ok` when the kernels take a reduction axis of `k` elements; `Unsupported` past 4096
    /// (operands not contiguous along it are packed into fixed stack buffers).
    pub fn check_depth(k: usize) -> Result<()> {
        if k > MATMUL_MAX_K {
            return Err(KernelError::Unsupported(Cow::Borrowed(
                "matmul_i8 takes a reduction axis of at most 4096",
            )));
        }
        Ok(())
    }

    /// The tier of the kernel this plan runs.
    pub fn tier(&self) -> Tier {
        match self.kernel {
            Kernel::Portable => Tier::Scalar,
            #[cfg(target_arch = "aarch64")]
            Kernel::Neon => Tier::Neon,
            #[cfg(target_arch = "aarch64")]
            Kernel::NeonDotprod => Tier::NeonDotprod,
        }
    }

    /// The requantization and bias of column `n`.
    #[inline(always)]
    pub(super) fn column(&self, n: usize) -> (Requant, i32) {
        let r = if self.requant.len() == 1 {
            self.requant[0]
        } else {
            self.requant.get(n).copied().unwrap_or(self.requant[0])
        };
        (r, self.bias.get(n).copied().unwrap_or(0))
    }

    /// Requantizes an exact accumulator `Σ (l − zl)(r − zr)` of column `n`.
    #[inline(always)]
    pub(super) fn finish(&self, acc: i32, n: usize) -> i8 {
        let (r, bias) = self.column(n);
        (self.zo + r.apply(acc.wrapping_add(bias), self.rounding)).clamp(self.act_min, self.act_max)
            as i8
    }

    fn validate(
        &self,
        lhs: &View<'_, i8>,
        rhs: &View<'_, i8>,
        out: &ViewMut<'_, i8>,
    ) -> Result<Shape> {
        let rank = out.rank();
        if rank < 2 || lhs.rank() != rank || rhs.rank() != rank {
            return Err(KernelError::InvalidArgument(
                "matmul_i8 operands need the same rank, at least 2",
            ));
        }
        let (ls, rs, os) = (lhs.shape(), rhs.shape(), out.shape());
        let b = rank - 2;
        if ls[..b] != os[..b] || rs[..b] != os[..b] {
            return Err(KernelError::InvalidArgument("matmul_i8 batch axes differ"));
        }
        let (m, k, n) = (ls[b], ls[b + 1], rs[b + 1]);
        if rs[b] != k || os[b] != m || os[b + 1] != n {
            return Err(KernelError::InvalidArgument(
                "matmul_i8 shapes do not chain: [.., M, K] x [.., K, N] -> [.., M, N]",
            ));
        }
        Self::check_depth(k)?;
        if self.requant.len() != 1 && self.requant.len() != n {
            return Err(KernelError::InvalidArgument(
                "matmul_i8 needs one requantization or one per output column",
            ));
        }
        if !self.bias.is_empty() && self.bias.len() != n {
            return Err(KernelError::InvalidArgument(
                "matmul_i8 bias needs one entry per output column",
            ));
        }
        super::rows::check_no_overlap(os, out.strides())?;
        let mut shape = Shape {
            batch: [0; MAX_RANK],
            batch_rank: b,
            m,
            k,
            n,
            ls: [0; MAX_RANK],
            rs: [0; MAX_RANK],
            os: [0; MAX_RANK],
        };
        shape.batch[..b].copy_from_slice(&os[..b]);
        shape.ls[..rank].copy_from_slice(lhs.strides());
        shape.rs[..rank].copy_from_slice(rhs.strides());
        shape.os[..rank].copy_from_slice(out.strides());
        Ok(shape)
    }

    /// `out = lhs · rhs` over the last two axes: `[.., M, K] x [.., K, N] -> [.., M, N]`, all
    /// i8 and of the same rank (2 to [`MAX_RANK`]). Transposed operands are views with swapped
    /// strides; packing for the kernel happens on the stack.
    ///
    /// Work is split across `pool` when one is given. The output must not overlap itself.
    /// Arguments are validated before anything is written.
    pub fn run(
        &self,
        lhs: TensorRef<'_>,
        rhs: TensorRef<'_>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
    ) -> Result<()> {
        let _span =
            tracing::trace_span!("kernels.matmul_i8", tier = self.tier().as_str()).entered();
        let (TensorRef::I8(lhs), TensorRef::I8(rhs), TensorMut::I8(mut out)) = (lhs, rhs, output)
        else {
            return Err(KernelError::InvalidArgument(
                "matmul_i8 needs i8 operands and output",
            ));
        };
        let s = self.validate(&lhs, &rhs, &out)?;
        if s.m == 0 || s.n == 0 || s.batches() == 0 {
            return Ok(());
        }
        let (l, r) = (lhs.data(), rhs.data());
        let dst = OutPtr(out.data_mut().as_mut_ptr());
        let row_blocks = s.m.div_ceil(TILE);
        let tasks = s.batches() * row_blocks;
        // A narrow right-hand side over rows contiguous along K: one call per batch's run of
        // row blocks, which packs the columns once.
        let narrow = s.n < TILE && s.strides().0 .1 == 1 && self.kernel != Kernel::Portable;
        for_each_block(tasks, pool, &|range| {
            #[cfg(target_arch = "aarch64")]
            let mut panels = super::matmul_i8_neon::Panels::new();
            #[cfg(target_arch = "aarch64")]
            if narrow {
                let mut t = range.start;
                while t < range.end {
                    let b = t / row_blocks;
                    let end = ((b + 1) * row_blocks).min(range.end);
                    let rows = (t % row_blocks) * TILE..((end - b * row_blocks) * TILE).min(s.m);
                    // SAFETY: NEON is part of the aarch64 baseline and `NeonDotprod` is selected
                    // only when `sdot` is usable; `n < TILE`, the lhs is contiguous along K, and
                    // output rows of different tasks are disjoint.
                    unsafe {
                        if self.kernel == Kernel::NeonDotprod {
                            super::matmul_i8_neon::narrow::<true>(
                                self,
                                &s,
                                l,
                                r,
                                dst.0,
                                b,
                                rows,
                                &mut panels,
                            );
                        } else {
                            super::matmul_i8_neon::narrow::<false>(
                                self,
                                &s,
                                l,
                                r,
                                dst.0,
                                b,
                                rows,
                                &mut panels,
                            );
                        }
                    }
                    t = end;
                }
                return;
            }
            #[cfg(not(target_arch = "aarch64"))]
            let _ = narrow;
            for t in range {
                let (b, rb) = (t / row_blocks, t % row_blocks);
                let rows = rb * TILE..(rb * TILE + TILE).min(s.m);
                match self.kernel {
                    Kernel::Portable => portable_rows(self, &s, l, r, dst, b, rows),
                    #[cfg(target_arch = "aarch64")]
                    // SAFETY: NEON is part of the aarch64 baseline; output rows of different
                    // tasks are disjoint.
                    Kernel::Neon => unsafe {
                        super::matmul_i8_neon::rows::<false>(
                            self,
                            &s,
                            l,
                            r,
                            dst.0,
                            b,
                            rows,
                            &mut panels,
                        )
                    },
                    #[cfg(target_arch = "aarch64")]
                    // SAFETY: as above; `NeonDotprod` is selected only when `sdot` is usable.
                    Kernel::NeonDotprod => unsafe {
                        super::matmul_i8_neon::rows::<true>(
                            self,
                            &s,
                            l,
                            r,
                            dst.0,
                            b,
                            rows,
                            &mut panels,
                        )
                    },
                }
            }
        });
        Ok(())
    }
}

/// The output base pointer, shared by the tasks of one call.
#[derive(Clone, Copy)]
pub(super) struct OutPtr(pub(super) *mut i8);

// SAFETY: tasks write disjoint output rows (the output was checked not to overlap itself) and a
// safe `Pool` cannot keep a task past `run`.
unsafe impl Sync for OutPtr {}

/// Rows `rows` of batch `b`, element by element.
fn portable_rows(
    p: &MatMulI8Plan,
    s: &Shape,
    l: &[i8],
    r: &[i8],
    dst: OutPtr,
    b: usize,
    rows: core::ops::Range<usize>,
) {
    let (lb, rb, ob) = s.batch_offsets(b);
    let ((lm, lk), (rk, rn), (om, on)) = s.strides();
    let k = s.k;
    for m in rows {
        // Rows contiguous along K: slice dot products.
        let a_row = (lk == 1 || k == 1)
            .then(|| l.get(lb + m * lm..lb + m * lm + k))
            .flatten();
        for n in 0..s.n {
            let b_col = (rk == 1 || k == 1)
                .then(|| r.get(rb + n * rn..rb + n * rn + k))
                .flatten();
            if let (Some(a), Some(c)) = (a_row, b_col) {
                let acc: i32 = a
                    .iter()
                    .zip(c)
                    .map(|(&x, &y)| (i32::from(x) - p.zl) * (i32::from(y) - p.zr))
                    .sum();
                // SAFETY: the element lies in the output view (validated) and only this task
                // writes row `m`.
                unsafe { dst.0.add(ob + m * om + n * on).write(p.finish(acc, n)) };
                continue;
            }
            let mut acc = 0i32;
            for k in 0..s.k {
                let a = i32::from(l.get(lb + m * lm + k * lk).copied().unwrap_or(0));
                let c = i32::from(r.get(rb + k * rk + n * rn).copied().unwrap_or(0));
                acc += (a - p.zl) * (c - p.zr);
            }
            // SAFETY: the element lies in the output view (validated) and only this task
            // writes row `m`.
            unsafe { dst.0.add(ob + m * om + n * on).write(p.finish(acc, n)) };
        }
    }
}

#[cfg(target_arch = "aarch64")]
fn select_kernel(f: CpuFeatures) -> Kernel {
    if f.dotprod_usable() {
        Kernel::NeonDotprod
    } else if f.neon_baseline {
        Kernel::Neon
    } else {
        Kernel::Portable
    }
}

#[cfg(not(target_arch = "aarch64"))]
fn select_kernel(_: CpuFeatures) -> Kernel {
    Kernel::Portable
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u32);
    impl Rng {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 17;
            self.0 ^= self.0 << 5;
            self.0
        }
        fn i8s(&mut self, n: usize) -> Vec<i8> {
            (0..n).map(|_| self.next() as u8 as i8).collect()
        }
    }

    /// Every tier this CPU has, scalar first.
    fn tiers() -> Vec<CpuFeatures> {
        let mut v = vec![CpuFeatures::default()];
        for t in ["neon", "neon_dotprod"] {
            if let Ok(f) = CpuFeatures::from_forced_tier(Some(t)) {
                v.push(f);
            }
        }
        v
    }

    /// The definition, in i64.
    #[allow(clippy::too_many_arguments)]
    fn naive(
        p: &MatMulI8Params,
        lhs: &[i8],
        rhs: &[i8],
        m: usize,
        k: usize,
        n: usize,
        lhs_t: bool,
        rhs_t: bool,
    ) -> Vec<i8> {
        let plan = MatMulI8Plan::with_features(p.clone(), CpuFeatures::default()).unwrap();
        let mut out = vec![0i8; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0i64;
                for kk in 0..k {
                    let a = if lhs_t {
                        lhs[kk * m + i]
                    } else {
                        lhs[i * k + kk]
                    };
                    let b = if rhs_t {
                        rhs[j * k + kk]
                    } else {
                        rhs[kk * n + j]
                    };
                    acc += (i64::from(a) - i64::from(p.lhs_zero_point))
                        * (i64::from(b) - i64::from(p.rhs_zero_point));
                }
                out[i * n + j] = plan.finish(acc as i32, j);
            }
        }
        out
    }

    #[test]
    fn every_tier_and_layout_matches_the_definition() {
        let mut rng = Rng(0x1234_5678);
        let dims = [
            (1, 1, 1),
            (3, 5, 2),
            (4, 16, 4),
            (7, 33, 9),
            (13, 400, 6),
            (5, 64, 17),
            (37, 16, 1),
            (21, 48, 3),
            (9, 19, 2),
            (40, 32, 1),
            (33, 16, 2),
        ];
        for &(m, k, n) in &dims {
            for (zl, zr) in [(0, 0), (-15, 6), (-128, 127), (8, -128)] {
                for per_channel in [false, true] {
                    let mut p = MatMulI8Params::tflite_batch_matmul(
                        Quant::new(0.1251, zl),
                        Quant::new(0.0102, zr),
                        Quant::new(0.1279, -9),
                        TfliteKernels::NeonRuy,
                    )
                    .unwrap();
                    p.rhs_zero_point = zr;
                    if per_channel {
                        p.requant = (0..n)
                            .map(|j| Requant::from_real(0.001 + j as f64 * 0.0003).unwrap())
                            .collect();
                        p.bias = (0..n).map(|j| j as i32 * 1000 - 3000).collect();
                    }
                    let lhs = rng.i8s(m * k);
                    let rhs = rng.i8s(k * n);
                    for (lhs_t, rhs_t) in
                        [(false, false), (true, false), (false, true), (true, true)]
                    {
                        let want = naive(&p, &lhs, &rhs, m, k, n, lhs_t, rhs_t);
                        let lv = if lhs_t {
                            View::strided(&lhs, &[m, k], &[1, m as isize]).unwrap()
                        } else {
                            View::contiguous(&lhs, &[m, k]).unwrap()
                        };
                        let rv = if rhs_t {
                            View::strided(&rhs, &[k, n], &[1, k as isize]).unwrap()
                        } else {
                            View::contiguous(&rhs, &[k, n]).unwrap()
                        };
                        for f in tiers() {
                            let plan = MatMulI8Plan::with_features(p.clone(), f).unwrap();
                            let mut out = vec![0i8; m * n];
                            plan.run(
                                TensorRef::I8(lv),
                                TensorRef::I8(rv),
                                TensorMut::I8(ViewMut::contiguous(&mut out, &[m, n]).unwrap()),
                                None,
                            )
                            .unwrap();
                            assert_eq!(
                                out,
                                want,
                                "{m}x{k}x{n} zl={zl} zr={zr} pc={per_channel} t=({lhs_t},{rhs_t}) {:?}",
                                plan.tier()
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn batches_and_broadcast_and_strided_output() {
        let mut rng = Rng(99);
        let (b, m, k, n) = (3, 5, 20, 6);
        let p = MatMulI8Params::tflite_batch_matmul(
            Quant::new(0.05, 3),
            Quant::new(0.02, -4),
            Quant::new(0.07, 1),
            TfliteKernels::NeonRuy,
        )
        .unwrap();
        let lhs = rng.i8s(b * m * k);
        let rhs = rng.i8s(k * n);
        let mut want = vec![0i8; b * m * n];
        for bi in 0..b {
            let w = naive(
                &p,
                &lhs[bi * m * k..(bi + 1) * m * k],
                &rhs,
                m,
                k,
                n,
                false,
                false,
            );
            want[bi * m * n..(bi + 1) * m * n].copy_from_slice(&w);
        }
        for f in tiers() {
            let plan = MatMulI8Plan::with_features(p.clone(), f).unwrap();
            // Output stored transposed per batch: [b, m, n] with strides [m*n, 1, m].
            let mut out = vec![0i8; b * m * n];
            plan.run(
                TensorRef::I8(View::contiguous(&lhs, &[b, m, k]).unwrap()),
                TensorRef::I8(View::strided(&rhs, &[b, k, n], &[0, n as isize, 1]).unwrap()),
                TensorMut::I8(
                    ViewMut::strided(&mut out, &[b, m, n], &[(m * n) as isize, 1, m as isize])
                        .unwrap(),
                ),
                Some(&crate::ScopedThreads::new(
                    core::num::NonZeroUsize::new(3).unwrap(),
                )),
            )
            .unwrap();
            for bi in 0..b {
                for i in 0..m {
                    for j in 0..n {
                        assert_eq!(out[bi * m * n + j * m + i], want[bi * m * n + i * n + j]);
                    }
                }
            }
        }
    }

    #[test]
    fn rejects_bad_params_and_mismatches() {
        let p = MatMulI8Params::tflite_batch_matmul(
            Quant::new(0.05, 3),
            Quant::new(0.02, -4),
            Quant::new(0.07, 1),
            TfliteKernels::NeonRuy,
        )
        .unwrap();
        let mut bad = p.clone();
        bad.lhs_zero_point = 200;
        assert!(MatMulI8Plan::new(bad).is_err());
        let mut big = p.clone();
        big.requant = vec![Requant::from_real(0.75).unwrap()];
        assert!(matches!(
            MatMulI8Plan::new(big.clone()),
            Err(KernelError::Unsupported(_))
        ));
        big.kernels = TfliteKernels::Reference;
        assert!(MatMulI8Plan::new(big).is_ok());
        let mut bad = p.clone();
        bad.requant.clear();
        assert!(MatMulI8Plan::new(bad).is_err());
        assert!(Requant::from_real(f64::NAN).is_err());
        let plan = MatMulI8Plan::new(p).unwrap();
        let (a, c) = ([0i8; 6], [0i8; 6]);
        let mut out = [7i8; 4];
        let r = plan.run(
            TensorRef::I8(View::contiguous(&a, &[2, 3]).unwrap()),
            TensorRef::I8(View::contiguous(&c, &[2, 3]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut out, &[2, 2]).unwrap()),
            None,
        );
        assert!(r.is_err());
        assert_eq!(out, [7; 4]);
    }
}
