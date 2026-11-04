// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! int8 softmax over the last axis followed by a matrix multiplication with one weight column:
//! a YOLO DFL head (`SOFTMAX`, then a 1×1 `CONV_2D` with one output channel) in one pass. The
//! probabilities are never written, nor rounded to 8 bits: each row's weighted expectation is
//! requantized directly.

use std::borrow::Cow;

use super::matmul_i8::{MatMulI8Params, MatMulI8Plan, Requant};
use super::rows::for_each_row_group;
use super::rows::RowGroup;
#[cfg(target_arch = "aarch64")]
use super::rows::ROW_GROUP;
use super::softmax_i8::{
    split_exp_tables, table_index, table_offset, SoftmaxI8Params, SoftmaxI8Plan,
};
use crate::cpu::{CpuFeatures, Tier};
use crate::error::{KernelError, Result};
use crate::pool::Pool;
use crate::quant::Rounding;
use crate::view::{TensorMut, TensorRef};

/// Parameters of an int8 softmax followed by a product with one weight column.
///
/// The pair it stands for is [`SoftmaxI8Plan`] with `softmax`, then [`MatMulI8Plan`] with
/// `projection` and `weights` as its `[n, 1]` right-hand side. `projection` must have one
/// requantization, at most one bias, and the softmax output's zero point as its left zero
/// point.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftmaxDotI8Params {
    pub softmax: SoftmaxI8Params,
    /// One weight per element of a row.
    pub weights: Vec<i8>,
    pub projection: MatMulI8Params,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    Portable,
    #[cfg(target_arch = "aarch64")]
    Neon,
}

/// A validated int8 softmax-then-dot over the last axis: `[.., n]` logits to `[.., 1]`
/// outputs, every other axis indexing independent rows.
///
/// The pair rounds every probability to a byte before the product; this plan does not. Its
/// accumulator is the expectation `Σⱼ pⱼ·(wⱼ − zr)` in units of the probability scale, rounded
/// once, then biased, requantized and clamped as the projection does. Against the pair it is
/// closer to the float model: TFLite's rounding drops the mass of tiny probabilities and clamps
/// a near-certain one to 255/256, which pulls its result toward zero by up to about one LSB.
///
/// The scalar tier takes the exponentials from TFLite's f32 softmax table and sums in f32. The
/// NEON kernel (rows of 16) takes them as 12-bit fixed-point products of two 16-entry tables
/// and sums them exactly in integers; it agrees with the scalar tier within ±1 LSB.
///
/// ```
/// use edgefirst_kernels::{
///     MatMulI8Params, Quant, SoftmaxDotI8Params, SoftmaxDotI8Plan, SoftmaxI8Params, TensorMut,
///     TensorRef, TfliteKernels, View, ViewMut,
/// };
///
/// let probs = Quant::new(1.0 / 256.0, -128);
/// // Two equal logits over weights 0 and 2 (scale 1): the expectation is 1.
/// let plan = SoftmaxDotI8Plan::new(SoftmaxDotI8Params {
///     softmax: SoftmaxI8Params { input: Quant::new(0.1, 0), output: probs, beta: 1.0 },
///     weights: vec![0, 2],
///     projection: MatMulI8Params::tflite_batch_matmul(
///         probs,
///         Quant::new(1.0, 0),
///         Quant::new(0.25, 0),
///         TfliteKernels::NeonRuy,
///     )?,
/// })?;
/// let input = [7i8, 7];
/// let mut output = [0i8; 1];
/// plan.run(
///     TensorRef::I8(View::contiguous(&input, &[1, 2])?),
///     TensorMut::I8(ViewMut::contiguous(&mut output, &[1, 1])?),
///     None,
/// )?;
/// assert_eq!(output, [4]);
/// # Ok::<(), edgefirst_kernels::KernelError>(())
/// ```
#[derive(Clone, Debug)]
pub struct SoftmaxDotI8Plan {
    /// `exp(−input_scale·beta·d)` at index `255 − d`, as the int8 softmax's table.
    table: [f32; 256],
    /// Accumulator units per unit of probability: the inverse softmax output scale.
    unit: f32,
    projection: MatMulI8Plan,
    /// `exp(−input_scale·beta·d)` for `d = 16·hi + lo` as `2·A[hi]·B[lo] / 2¹⁶`, with
    /// `2·A[0]·B[0] / 2¹⁶` rounding to 4095: the byte planes `A` low, `A` high, `B` low, `B`
    /// high.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    exp12: [[u8; 16]; 4],
    /// The weights less the right zero point.
    weights: Vec<i16>,
    kernel: Kernel,
}

/// The fixed-point table scale: `2·K² / 2¹⁶` rounds to 4095, so sixteen terms sum in a u16.
const EXP12_ONE: f64 = 11584.0;

/// `exp(scale·d)` for `d = 16·hi + lo` as `2·A[hi]·B[lo] / 2¹⁶` (at most 4095), split into
/// byte planes: `A` low, `A` high, `B` low, `B` high. `scale` is `−input_scale·beta`.
fn exp12_tables(scale: f32) -> [[u8; 16]; 4] {
    split_exp_tables(scale, EXP12_ONE)
}

impl SoftmaxDotI8Plan {
    /// Builds a plan for the CPU this process runs on, honouring `EDGEFIRST_KERNELS_FORCE_TIER`.
    pub fn new(p: SoftmaxDotI8Params) -> Result<Self> {
        Self::with_features(p, CpuFeatures::from_env_or_probe()?)
    }

    /// Builds a plan that runs the best kernel `f` allows.
    ///
    /// `softmax` and `projection` are validated as their own plans validate them; `weights`
    /// must be non-empty, and `projection` must have one requantization, at most one bias and
    /// the softmax output zero point on its left (`InvalidArgument`).
    pub fn with_features(p: SoftmaxDotI8Params, f: CpuFeatures) -> Result<Self> {
        if p.weights.is_empty() {
            return Err(KernelError::InvalidArgument(
                "softmax_dot_i8 needs at least one weight",
            ));
        }
        if p.projection.requant.len() != 1 || p.projection.bias.len() > 1 {
            return Err(KernelError::InvalidArgument(
                "softmax_dot_i8 projects onto one column",
            ));
        }
        if p.projection.lhs_zero_point != p.softmax.output.zero_point {
            return Err(KernelError::InvalidArgument(
                "softmax_dot_i8 projection must read the softmax output's zero point",
            ));
        }
        let softmax = SoftmaxI8Plan::with_features(p.softmax, CpuFeatures::default())?;
        let zr = p.projection.rhs_zero_point;
        let projection = MatMulI8Plan::with_features(p.projection, CpuFeatures::default())?;
        let scale = -p.softmax.input.scale * p.softmax.beta;
        Ok(Self {
            table: softmax.table,
            unit: 1.0 / p.softmax.output.scale,
            projection,
            exp12: exp12_tables(scale),
            weights: p
                .weights
                .iter()
                .map(|&w| (i32::from(w) - zr) as i16)
                .collect(),
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

    /// Softmax-then-dot of `input` (`[.., n]`, `n` the number of weights) into `output`
    /// (`[.., 1]`), both i8.
    ///
    /// Rows are split across `pool` when one is given. The input's last axis must be
    /// unit-stride (`Unsupported` otherwise); output axes sorted by stride must each step past
    /// the extent of the smaller ones. Arguments are validated before anything is written.
    pub fn run(
        &self,
        input: TensorRef<'_>,
        output: TensorMut<'_>,
        pool: Option<&dyn Pool>,
    ) -> Result<()> {
        let _span =
            tracing::trace_span!("kernels.softmax_dot_i8", tier = self.tier().as_str()).entered();
        let (TensorRef::I8(input), TensorMut::I8(mut output)) = (input, output) else {
            return Err(KernelError::InvalidArgument(
                "softmax_dot_i8 needs i8 input and output",
            ));
        };
        let (Some((&n, outer)), Some((&m, out_outer))) =
            (input.shape().split_last(), output.shape().split_last())
        else {
            return Err(KernelError::InvalidArgument(
                "softmax_dot_i8 views must have rank ≥ 1",
            ));
        };
        if n != self.weights.len() || m != 1 || outer != out_outer {
            return Err(KernelError::InvalidArgument(
                "softmax_dot_i8 needs [.., weights] input and [.., 1] output",
            ));
        }
        let group = |g: &mut RowGroup<'_, i8>| {
            #[cfg(target_arch = "aarch64")]
            if self.kernel == Kernel::Neon && g.count() == ROW_GROUP && g.len() == ROW_GROUP {
                // SAFETY: NEON is part of the aarch64 baseline; the group is 16 rows of 16.
                unsafe { super::softmax_dot_i8_neon::rows16(self, g) };
                return;
            }
            for k in 0..g.count() {
                let v = self.row(g.input(k));
                if let Some(o) = g.output(k).first_mut() {
                    *o = v;
                }
            }
        };
        for_each_row_group(&input, &mut output, pool, &group).unwrap_or(Err(
            KernelError::Unsupported(Cow::Borrowed(
                "softmax_dot_i8 needs a unit-stride input row axis",
            )),
        ))
    }

    /// The scalar result of one row of `weights.len()` logits.
    fn row(&self, src: &[i8]) -> i8 {
        let off = table_offset(src.iter().copied().max().unwrap_or(i8::MIN));
        let (mut sum, mut ws) = (0.0f32, 0.0f32);
        for (&x, &w) in src.iter().zip(&self.weights) {
            let e = self.table[table_index(x, off)];
            sum += e;
            ws += e * f32::from(w);
        }
        self.projection
            .finish((ws / sum * self.unit).round() as i32, 0)
    }

    /// Accumulator units per unit of probability.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(super) fn unit(&self) -> f32 {
        self.unit
    }

    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(super) fn exp12(&self) -> &[[u8; 16]; 4] {
        &self.exp12
    }

    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(super) fn weights(&self) -> &[i16] {
        &self.weights
    }

    /// The requantization: multiplier, bias, rounding, output zero point and clamp.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(super) fn requant(&self) -> (Requant, i32, Rounding, i32, i32, i32) {
        let (r, bias) = self.projection.column(0);
        let p = &self.projection;
        (r, bias, p.rounding, p.zo, p.act_min, p.act_max)
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

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::quant::{Activation, Quant, TfliteKernels};
    use crate::view::{View, ViewMut};
    use rand::{RngExt, SeedableRng};

    /// The weights of a YOLO DFL head: `round(i·127/15)`.
    pub(in super::super) fn dfl_weights() -> Vec<i8> {
        (0..16i32).map(|i| ((i * 127 + 7) / 15) as i8).collect()
    }

    pub(in super::super) fn params(scale: f32, weights: Vec<i8>) -> SoftmaxDotI8Params {
        let probs = Quant::new(1.0 / 256.0, -128);
        SoftmaxDotI8Params {
            softmax: SoftmaxI8Params {
                input: Quant::new(scale, 3),
                output: probs,
                beta: 1.0,
            },
            weights,
            projection: MatMulI8Params::tflite_conv(
                probs,
                &[15.0 / 127.0],
                Quant::new(0.0579, -128),
                vec![0],
                Activation::None,
                TfliteKernels::NeonRuy,
            )
            .unwrap(),
        }
    }

    /// The two operators one after the other on the scalar tier.
    pub(in super::super) fn two_step(p: &SoftmaxDotI8Params, x: &[i8], rows: usize) -> Vec<i8> {
        let n = p.weights.len();
        let f = CpuFeatures::default();
        let sm = SoftmaxI8Plan::with_features(p.softmax, f).unwrap();
        let mut probs = vec![0i8; rows * n];
        sm.run(
            TensorRef::I8(View::contiguous(x, &[rows, n]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut probs, &[rows, n]).unwrap()),
            None,
        )
        .unwrap();
        let mm = MatMulI8Plan::with_features(p.projection.clone(), f).unwrap();
        let mut out = vec![0i8; rows];
        mm.run(
            TensorRef::I8(View::contiguous(&probs, &[rows, n]).unwrap()),
            TensorRef::I8(View::contiguous(&p.weights, &[n, 1]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut out, &[rows, 1]).unwrap()),
            None,
        )
        .unwrap();
        out
    }

    #[test]
    fn every_tier_follows_the_float_definition() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xDF1_D07);
        let mut tiers = vec![CpuFeatures::default()];
        if let Ok(f) = CpuFeatures::from_forced_tier(Some("neon")) {
            tiers.push(f);
        }
        for &scale in &[0.02f32, 0.0947, 0.25, 0.6] {
            for n in [16usize, 1, 5, 17, 64] {
                let weights: Vec<i8> = if n == 16 {
                    dfl_weights()
                } else {
                    (0..n).map(|_| rng.random_range(-128..=127)).collect()
                };
                let mut p = params(scale, weights);
                p.projection.rhs_zero_point = 2;
                let rows = 37;
                let x: Vec<i8> = (0..rows * n).map(|_| rng.random()).collect();
                for &f in &tiers {
                    let plan = SoftmaxDotI8Plan::with_features(p.clone(), f).unwrap();
                    let mut out = vec![0i8; rows];
                    plan.run(
                        TensorRef::I8(View::contiguous(&x, &[rows, n]).unwrap()),
                        TensorMut::I8(ViewMut::contiguous(&mut out, &[rows, 1]).unwrap()),
                        None,
                    )
                    .unwrap();
                    for (r, &o) in out.iter().enumerate() {
                        let want = definition(&p, &x[r * n..(r + 1) * n]);
                        assert!(
                            (f64::from(o) - want).abs() <= 1.0,
                            "scale {scale} n {n} row {r} {:?}: {o} vs {want:.3}",
                            plan.tier()
                        );
                    }
                }
            }
        }
    }

    /// The output in f64 before rounding: `zo + (E[w] + bias) / output_scale`, clamped.
    fn definition(p: &SoftmaxDotI8Params, row: &[i8]) -> f64 {
        let s = f64::from(p.softmax.input.scale * p.softmax.beta);
        let max = row.iter().copied().max().unwrap_or(0);
        let e: Vec<f64> = row
            .iter()
            .map(|&x| (s * f64::from(i32::from(x) - i32::from(max))).exp())
            .collect();
        let sum: f64 = e.iter().sum();
        let pr = &p.projection;
        let ws: f64 = e
            .iter()
            .zip(&p.weights)
            .map(|(e, &w)| e * f64::from(i32::from(w) - pr.rhs_zero_point))
            .sum();
        let r = pr.requant[0];
        let real = f64::from(r.multiplier) * 2f64.powi(r.shift - 31);
        let acc = ws / sum / f64::from(p.softmax.output.scale)
            + f64::from(pr.bias.first().copied().unwrap_or(0));
        (f64::from(pr.output.zero_point) + acc * real).clamp(-128.0, 127.0)
    }

    #[test]
    fn dfl_heads_differ_from_the_pair_only_by_its_rounding_and_sit_closer_to_float() {
        // The pair rounds every probability to a byte: that loses mass (with weights ≥ 0 it
        // reads low on balance) and adds noise either way, up to a few LSB on flat rows.
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xD0F1);
        let rows = 2048;
        let (mut off, mut net) = (0usize, 0i64);
        let (mut err_fused, mut err_pair) = (0.0f64, 0.0f64);
        for &scale in &[0.05f32, 0.0947, 0.2] {
            let p = params(scale, dfl_weights());
            let x: Vec<i8> = (0..rows * 16).map(|_| rng.random_range(-60..=60)).collect();
            let plan = SoftmaxDotI8Plan::with_features(p.clone(), CpuFeatures::default()).unwrap();
            let mut got = vec![0i8; rows];
            plan.run(
                TensorRef::I8(View::contiguous(&x, &[rows, 16]).unwrap()),
                TensorMut::I8(ViewMut::contiguous(&mut got, &[rows, 1]).unwrap()),
                None,
            )
            .unwrap();
            let pair = two_step(&p, &x, rows);
            for (r, (&a, &b)) in got.iter().zip(&pair).enumerate() {
                let want = definition(&p, &x[r * 16..(r + 1) * 16]);
                let (ef, ep) = ((f64::from(a) - want).abs(), (f64::from(b) - want).abs());
                let d = i64::from(a) - i64::from(b);
                assert!(
                    d.abs() as f64 <= ep.ceil() + 1.0,
                    "scale {scale} row {r}: fused {a} pair {b} float {want:.2}"
                );
                off += usize::from(d != 0);
                net += d;
                err_fused += ef;
                err_pair += ep;
            }
        }
        assert!(
            net >= 0,
            "fused reads below the pair on balance: {net} over {off}"
        );
        assert!(
            err_fused < err_pair,
            "fused {err_fused:.0} vs pair {err_pair:.0} from float"
        );
    }

    #[test]
    fn strided_views_and_activation() {
        // [2, 3, 16] logits read as rows from an NHWC-like layout, [2, 3, 1] output strided.
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let x: Vec<i8> = (0..2 * 3 * 64).map(|_| rng.random()).collect();
        let weights: Vec<i8> = (0..16).map(|i| i * 8 - 64).collect();
        let mut p = params(0.1, weights);
        p.projection.activation = Activation::Relu;
        p.projection.output = Quant::new(0.05, -20);
        let plan = SoftmaxDotI8Plan::new(p.clone()).unwrap();
        let input = View::strided(&x, &[2, 3, 16], &[192, 64, 1]).unwrap();
        let mut out = vec![99i8; 2 * 3 * 2];
        plan.run(
            TensorRef::I8(input),
            TensorMut::I8(ViewMut::strided(&mut out, &[2, 3, 1], &[6, 2, 1]).unwrap()),
            None,
        )
        .unwrap();
        for a in 0..2 {
            for b in 0..3 {
                let at = a * 192 + b * 64;
                let want = two_step(&p, &x[at..at + 16], 1)[0];
                let got = out[a * 6 + b * 2];
                assert!(
                    (i32::from(got) - i32::from(want)).abs() <= 1,
                    "{a},{b}: {got} vs {want}"
                );
                assert!(got >= -20, "relu clamp");
                assert_eq!(out[a * 6 + b * 2 + 1], 99, "gap written");
            }
        }
    }

    #[test]
    fn rejects_bad_params_and_shapes() {
        let p = params(0.1, vec![1; 16]);
        let mut q = p.clone();
        q.weights.clear();
        assert!(SoftmaxDotI8Plan::new(q).is_err());
        let mut q = p.clone();
        q.projection.lhs_zero_point = 0;
        assert!(SoftmaxDotI8Plan::new(q).is_err());
        let mut q = p.clone();
        q.projection.bias = vec![0, 0];
        assert!(SoftmaxDotI8Plan::new(q).is_err());
        let plan = SoftmaxDotI8Plan::new(p).unwrap();
        let x = [0i8; 32];
        let mut out = [0i8; 2];
        let bad_out = plan.run(
            TensorRef::I8(View::contiguous(&x, &[2, 16]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut out, &[1, 2]).unwrap()),
            None,
        );
        assert!(bad_out.is_err());
        let strided_row = plan.run(
            TensorRef::I8(View::strided(&x, &[1, 16], &[32, 2]).unwrap()),
            TensorMut::I8(ViewMut::contiguous(&mut out[..1], &[1, 1]).unwrap()),
            None,
        );
        assert!(strided_row.is_err());
    }
}
