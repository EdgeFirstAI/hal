// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Every per-scale box cell (DFL per-tensor and per-channel, LTRB) on every tier this CPU has.
//!
//! Cells whose arithmetic the NEON tiers keep are compared bit for bit: the 8-bit exp-table
//! cells with the scalar tier, integer LTRB and 16-bit DFL into f32 with the decoder's earlier
//! NEON composition, float LTRB into f32 with the scalar tier, and `f16` outputs of the f32 path
//! with that tier's f32 cell rounded once. The rest are held to the float64 reference: f32
//! outputs to [`f32_bar`], `f16` outputs to [`F16_ULPS`] (native FP16 cells
//! [`fp16_ulps`]) f16 ulps of the value or of 640 px.

use super::dispatch::{BoxLevelDispatch, DflPerChannelDispatch, DstSliceMut, InputView};
use super::{decoder_cpu_features_from, CpuFeatures};
use crate::per_scale::plan::{Layout, LevelPlan};
use crate::per_scale::DecodeDtype;
use crate::schema::BoxEncoding;
use crate::{DecoderError, Quantization};
use edgefirst_kernels::prim::grids::make_anchor_grid;
use edgefirst_tensor::DType;
use half::f16;

/// f16 outputs of the f32 path rounded once, in f16 ulps of `max(|value|, 640 px)`.
const F16_ULPS: f64 = 1.0;

/// f16 outputs of the native FP16 cells: the kernels crate's DFL box bound, 1 ulp up to `reg_max`
/// 32 and 3 at 64, where each lane sums four f16 steps and distances reach 63 grid units.
fn fp16_ulps(reg_max: usize) -> f64 {
    if reg_max > 32 {
        3.0
    } else {
        1.0
    }
}

/// Every tier this CPU can run, by its `EDGEFIRST_DECODER_FORCE_KERNEL` name.
fn tiers() -> Vec<(&'static str, CpuFeatures)> {
    ["scalar", "neon", "neon_fp16", "neon_dotprod"]
        .into_iter()
        .filter_map(|t| decoder_cpu_features_from(Some(t)).ok().map(|f| (t, f)))
        .collect()
}

const INPUTS: [DType; 6] = [
    DType::I8,
    DType::U8,
    DType::I16,
    DType::U16,
    DType::F16,
    DType::F32,
];
const INTEGERS: [DType; 4] = [DType::I8, DType::U8, DType::I16, DType::U16];
const OUTPUTS: [DecodeDtype; 2] = [DecodeDtype::F32, DecodeDtype::F16];

/// `(h, w, stride)`: anchor counts on and off every vector width, boxes out to about 640 px.
const LEVELS: [(usize, usize, f32); 7] = [
    (1, 1, 8.0),
    (1, 3, 8.0),
    (2, 7, 16.0),
    (1, 16, 32.0),
    (3, 21, 8.0),
    (1, 80, 8.0),
    (5, 17, 32.0),
];

fn level(h: usize, w: usize, stride: f32, reg_max: usize) -> LevelPlan {
    let (grid_x, grid_y) = make_anchor_grid(h, w);
    LevelPlan {
        stride,
        h,
        w,
        reg_max,
        anchor_offset: 0,
        grid_x,
        grid_y,
        box_shape: vec![1, h, w, 4 * reg_max].into_boxed_slice(),
        score_shape: vec![1, h, w, 1].into_boxed_slice(),
        mc_shape: None,
        layout: Layout::Nhwc,
    }
}

/// A box cell as the plan selects it: per-tensor or per-channel.
#[derive(Debug, Clone, Copy)]
enum Cell {
    Box(BoxLevelDispatch),
    PerChannel(DflPerChannelDispatch),
}

impl Cell {
    fn name(&self) -> String {
        match self {
            Cell::Box(d) => format!("{d:?}"),
            Cell::PerChannel(d) => format!("PerChannel{d:?}"),
        }
    }
}

/// The box encodings a cell can decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Dfl,
    DflPerChannel,
    Ltrb,
}

fn select(kind: Kind, input: DType, output: DecodeDtype, f: &CpuFeatures) -> Cell {
    match kind {
        Kind::Dfl => {
            Cell::Box(BoxLevelDispatch::select(BoxEncoding::Dfl, input, output, f).unwrap())
        }
        Kind::Ltrb => {
            Cell::Box(BoxLevelDispatch::select(BoxEncoding::Direct, input, output, f).unwrap())
        }
        Kind::DflPerChannel => Cell::PerChannel(DflPerChannelDispatch::select(output, f)),
    }
}

/// On NEON tiers no box cell is scalar. On FP16 tiers `f16` outputs take a native FP16 cell,
/// except the 8-bit exp-table cells and LTRB, whose f32 results are rounded once, and 16-bit
/// per-tensor DFL into f32 keeps its fp16 exp; no other cell is FP16.
#[test]
fn every_box_cell_runs_neon_on_neon_tiers() {
    for (tier, f) in tiers() {
        for kind in [Kind::Dfl, Kind::DflPerChannel, Kind::Ltrb] {
            let inputs: &[DType] = if kind == Kind::DflPerChannel {
                &INTEGERS
            } else {
                &INPUTS
            };
            for &input in inputs {
                for output in OUTPUTS {
                    let name = select(kind, input, output, &f).name();
                    let label = format!("{tier} {kind:?} {input:?}->{output:?}: {name}");
                    if !f.neon_baseline {
                        assert!(name.ends_with("Scalar") || name.ends_with("Lut"), "{label}");
                        continue;
                    }
                    assert!(name.contains("Neon"), "{label}");
                    let narrowed = kind == Kind::Dfl && matches!(input, DType::I8 | DType::U8)
                        || kind == Kind::Ltrb;
                    let fp16_exp = kind == Kind::Dfl
                        && matches!(input, DType::I16 | DType::U16)
                        && output == DecodeDtype::F32;
                    let fp16 = f.neon_fp16 && (output == DecodeDtype::F16 && !narrowed || fp16_exp);
                    assert_eq!(name.ends_with("NeonFp16"), fp16, "{label}");
                }
            }
        }
    }
}

/// The forced `scalar` tier selects scalar cells even where NEON exists.
#[test]
fn forced_scalar_tier_selects_scalar_box_cells() {
    let f = decoder_cpu_features_from(Some("scalar")).unwrap();
    for kind in [Kind::Dfl, Kind::DflPerChannel, Kind::Ltrb] {
        for output in OUTPUTS {
            let name = select(kind, DType::I16, output, &f).name();
            assert!(name.ends_with("Scalar"), "{kind:?} {output:?}: {name}");
        }
    }
}

/// Small deterministic generator.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// DFL logits of `anchors` anchors: per side a peak with a slope and noise, in about ±8.
fn dfl_logits(anchors: usize, reg_max: usize, seed: u64) -> Vec<f64> {
    let mut r = Lcg(seed);
    let mut v = Vec::with_capacity(anchors * 4 * reg_max);
    for _ in 0..anchors * 4 {
        let peak = r.next() * reg_max as f64;
        let fall = 0.5 + 2.0 * r.next();
        for i in 0..reg_max {
            v.push(8.0 - fall * (i as f64 - peak).abs() + r.next() - 0.5);
        }
    }
    v
}

/// LTRB distances of `anchors` anchors in grid units, 0 to 20.
fn ltrb_distances(anchors: usize, seed: u64) -> Vec<f64> {
    let mut r = Lcg(seed);
    (0..anchors * 4).map(|_| 20.0 * r.next()).collect()
}

/// Raw inputs of one dtype.
enum Raw {
    I8(Vec<i8>),
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
}

impl Raw {
    /// `x` quantized with `scale`/`zp` per element (`q(i)`), rounded and clamped to the type.
    fn quantize(x: &[f64], dtype: DType, q: impl Fn(usize) -> (f64, f64)) -> Self {
        let raw = |i: usize| {
            let (s, zp) = q(i);
            x[i] / s + zp
        };
        let n = x.len();
        let int = |i: usize, lo: f64, hi: f64| raw(i).round().clamp(lo, hi);
        match dtype {
            DType::I8 => Raw::I8((0..n).map(|i| int(i, -128.0, 127.0) as i8).collect()),
            DType::U8 => Raw::U8((0..n).map(|i| int(i, 0.0, 255.0) as u8).collect()),
            DType::I16 => Raw::I16((0..n).map(|i| int(i, -32768.0, 32767.0) as i16).collect()),
            DType::U16 => Raw::U16((0..n).map(|i| int(i, 0.0, 65535.0) as u16).collect()),
            DType::F16 => Raw::F16((0..n).map(|i| f16::from_f64(raw(i))).collect()),
            _ => Raw::F32((0..n).map(|i| raw(i) as f32).collect()),
        }
    }

    fn view(&self) -> InputView<'_> {
        match self {
            Raw::I8(v) => InputView::I8(v),
            Raw::U8(v) => InputView::U8(v),
            Raw::I16(v) => InputView::I16(v),
            Raw::U16(v) => InputView::U16(v),
            Raw::F16(v) => InputView::F16(v),
            Raw::F32(v) => InputView::F32(v),
        }
    }

    fn value(&self, i: usize) -> f64 {
        match self {
            Raw::I8(v) => f64::from(v[i]),
            Raw::U8(v) => f64::from(v[i]),
            Raw::I16(v) => f64::from(v[i]),
            Raw::U16(v) => f64::from(v[i]),
            Raw::F16(v) => v[i].to_f64(),
            Raw::F32(v) => f64::from(v[i]),
        }
    }

    fn is_float(&self) -> bool {
        matches!(self, Raw::F16(_) | Raw::F32(_))
    }

    /// The dequantized value of element `i`: exact for integers, and for floats the f32
    /// dequantize `prim::dequant` computes.
    fn x(&self, i: usize, q: Quantization) -> f64 {
        let v = self.value(i);
        if !self.is_float() {
            (v - f64::from(q.zero_point)) * f64::from(q.scale)
        } else if q == Quantization::identity() {
            v
        } else {
            f64::from((v as f32 - q.zero_point as f32) * q.scale)
        }
    }
}

fn quants(dtype: DType) -> Vec<Quantization> {
    let q = Quantization::new;
    match dtype {
        DType::I8 => vec![q(0.1, 0), q(0.17, -20)],
        DType::U8 => vec![q(0.1, 128), q(0.11, 100)],
        DType::I16 => vec![q(0.0005, 0), q(0.001, 100)],
        DType::U16 => vec![q(0.0005, 32768), q(0.001, 30000)],
        _ => vec![Quantization::identity(), q(0.5, 3)],
    }
}

/// LTRB quantizations: distances 0 to 20 grid units in range.
fn ltrb_quants(dtype: DType) -> Vec<Quantization> {
    let q = Quantization::new;
    match dtype {
        DType::I8 => vec![q(0.1, -128), q(0.09, -100)],
        DType::U8 => vec![q(0.1, 0), q(0.09, 20)],
        DType::I16 => vec![q(0.001, 0), q(0.0007, -12000)],
        DType::U16 => vec![q(0.001, 0), q(0.0004, 7)],
        _ => vec![Quantization::identity(), q(0.5, 3)],
    }
}

/// Float64 boxes of one anchor's distances.
fn boxes64(ltrb: [f64; 4], gx: f64, gy: f64, s: f64) -> [f64; 4] {
    let [l, t, r, b] = ltrb;
    [
        (gx + (r - l) / 2.0) * s,
        (gy + (b - t) / 2.0) * s,
        (l + r) * s,
        (t + b) * s,
    ]
}

/// Float64 softmax-weighted distance of one side.
fn side64(x: &[f64]) -> f64 {
    let m = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (mut sum, mut ws) = (0.0, 0.0);
    for (i, &v) in x.iter().enumerate() {
        let e = (v - m).exp();
        sum += e;
        ws += e * i as f64;
    }
    ws / sum
}

/// Float64 boxes of every anchor of `lvl` from dequantized inputs `x`.
fn reference(kind: Kind, lvl: &LevelPlan, x: &[f64]) -> Vec<f64> {
    let per = if kind == Kind::Ltrb {
        4
    } else {
        4 * lvl.reg_max
    };
    let mut out = Vec::with_capacity(lvl.h * lvl.w * 4);
    for (a, v) in x.chunks_exact(per).enumerate() {
        let ltrb = if kind == Kind::Ltrb {
            [v[0], v[1], v[2], v[3]]
        } else {
            let mut d = [0.0; 4];
            for (d, side) in d.iter_mut().zip(v.chunks_exact(lvl.reg_max)) {
                *d = side64(side);
            }
            d
        };
        let (gx, gy) = (f64::from(lvl.grid_x[a]), f64::from(lvl.grid_y[a]));
        out.extend(boxes64(ltrb, gx, gy, f64::from(lvl.stride)));
    }
    out
}

fn ulp_f32(v: f64) -> f64 {
    let v = (v.abs() as f32).max(f32::MIN_POSITIVE);
    f64::from(f32::from_bits(v.to_bits() + 1) - v)
}

fn ulp_f16(v: f64) -> f64 {
    let h = f16::from_f64(v.abs()).max(f16::MIN_POSITIVE);
    f64::from(f16::from_bits(h.to_bits() + 1).to_f32() - h.to_f32())
}

/// f32 outputs of tolerance cells: 2e-4 px at stride 16 and `reg_max` 16, plus 4 f32 ulps.
/// Above `reg_max` 16 the first term grows with the distance range and the square root of the
/// bins summed (rounding over `reg_max` terms): `(reg_max − 1) / 15 · √(reg_max / 16)`.
fn f32_bar(stride: f32, reg_max: usize, want: f64) -> f64 {
    let rm = reg_max as f64;
    let growth = ((rm - 1.0) / 15.0 * (rm / 16.0).sqrt()).max(1.0);
    2e-4 * f64::from(stride) / 16.0 * growth + 4.0 * ulp_f32(want)
}

/// The fp16-exp cells' f32 bar: 2% per exp weight moves a side by `0.02 · (reg_max − 1)` grid
/// units, twice for a box coordinate.
fn fp16_exp_bar(stride: f32, reg_max: usize) -> f64 {
    2.0 * 0.02 * (reg_max - 1) as f64 * f64::from(stride)
}

/// One cell's input set: raw values, how to run them, and their float64 boxes.
struct Case {
    lvl: LevelPlan,
    raw: Raw,
    q: Quantization,
    scales: Vec<f32>,
    zps: Vec<i32>,
    want: Vec<f64>,
}

impl Case {
    fn run(&self, cell: Cell, output: DecodeDtype) -> (Vec<f32>, Vec<f16>) {
        let n = self.lvl.h * self.lvl.w * 4;
        let (mut d32, mut d16) = (vec![0f32; n], vec![f16::ZERO; n]);
        let dst = match output {
            DecodeDtype::F32 => DstSliceMut::F32(&mut d32),
            DecodeDtype::F16 => DstSliceMut::F16(&mut d16),
        };
        match cell {
            Cell::Box(d) => d.run(self.raw.view(), self.q, &self.lvl, None, dst),
            Cell::PerChannel(d) => d.run(self.raw.view(), &self.scales, &self.zps, &self.lvl, dst),
        }
        .unwrap();
        (d32, d16)
    }
}

/// Every input set of `kind` and `input`: each level, each quantization, and for per-channel
/// DFL with and without zero points.
fn cases(kind: Kind, input: DType, reg_max: usize) -> Vec<Case> {
    let mut out = Vec::new();
    for (li, &(h, w, stride)) in LEVELS.iter().enumerate() {
        let rm = if kind == Kind::Ltrb { 1 } else { reg_max };
        let lvl = level(h, w, stride, rm);
        let anchors = h * w;
        let seed = 17 * li as u64 + reg_max as u64;
        let x = match kind {
            Kind::Ltrb => ltrb_distances(anchors, seed),
            _ => dfl_logits(anchors, rm, seed),
        };
        match kind {
            Kind::DflPerChannel => {
                let ch = 4 * rm;
                let base = quants(input)[0];
                let scales: Vec<f32> = (0..ch)
                    .map(|c| base.scale * (0.6 + 0.1 * (c % 9) as f32))
                    .collect();
                for with_zp in [true, false] {
                    let zps: Vec<i32> = if with_zp {
                        (0..ch)
                            .map(|c| base.zero_point + (c % 5) as i32 - 2)
                            .collect()
                    } else {
                        Vec::new()
                    };
                    let qc = |i: usize| {
                        let c = i % ch;
                        (
                            f64::from(scales[c]),
                            f64::from(zps.get(c).copied().unwrap_or(0)),
                        )
                    };
                    let raw = Raw::quantize(&x, input, qc);
                    let deq: Vec<f64> = (0..x.len())
                        .map(|i| {
                            let (s, zp) = qc(i);
                            (raw.value(i) - zp) * s
                        })
                        .collect();
                    let want = reference(kind, &lvl, &deq);
                    out.push(Case {
                        lvl: level(h, w, stride, rm),
                        raw,
                        q: base,
                        scales: scales.clone(),
                        zps,
                        want,
                    });
                }
            }
            _ => {
                let qs = if kind == Kind::Ltrb {
                    ltrb_quants(input)
                } else {
                    quants(input)
                };
                for q in qs {
                    let qf = |_| (f64::from(q.scale), f64::from(q.zero_point));
                    let raw = Raw::quantize(&x, input, qf);
                    let deq: Vec<f64> = (0..x.len()).map(|i| raw.x(i, q)).collect();
                    let want = reference(kind, &lvl, &deq);
                    out.push(Case {
                        lvl: level(h, w, stride, rm),
                        raw,
                        q,
                        scales: Vec::new(),
                        zps: Vec::new(),
                        want,
                    });
                }
            }
        }
    }
    out
}

/// `got` equals `want` bit for bit, NaN equal to NaN.
fn assert_same_f32(got: &[f32], want: &[f32], label: &str) {
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
            "{label} at {i}: {g} vs {w}"
        );
    }
}

fn assert_same_f16(got: &[f16], want: &[f16], label: &str) {
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()),
            "{label} at {i}: {g} vs {w}"
        );
    }
}

/// Worst error of `got` against the reference in bars ([`f32_bar`], or [`fp16_exp_bar`] for
/// `fp16_exp`); asserts each value within one.
fn check_f32(case: &Case, got: &[f32], fp16_exp: bool, label: &str) -> f64 {
    let mut worst = 0f64;
    for (i, (&g, &w)) in got.iter().zip(&case.want).enumerate() {
        let bar = if fp16_exp {
            fp16_exp_bar(case.lvl.stride, case.lvl.reg_max)
        } else {
            f32_bar(case.lvl.stride, case.lvl.reg_max, w)
        };
        let e = (f64::from(g) - w).abs() / bar;
        worst = worst.max(e);
        assert!(e <= 1.0, "{label} at {i}: got {g}, want {w} ({e:.2} bars)");
    }
    worst
}

/// Worst error of `got` against the reference rounded to f16, in f16 ulps of the value or of
/// 640 px; asserts each within `bound` when given.
fn check_f16(case: &Case, got: &[f16], bound: Option<f64>, label: &str) -> f64 {
    let floor = ulp_f16(640.0);
    let mut worst = 0f64;
    for (i, (&g, &w)) in got.iter().zip(&case.want).enumerate() {
        let want = f16::from_f64(w);
        let e = (g.to_f64() - want.to_f64()).abs() / ulp_f16(w).max(floor);
        worst = worst.max(e);
        if let Some(b) = bound {
            assert!(
                e <= b,
                "{label} at {i}: got {g}, want {want} ({w}): {e:.2} ulp"
            );
        }
    }
    worst
}

/// The decoder's earlier NEON compositions, rebuilt from the same `prim` calls.
#[cfg(target_arch = "aarch64")]
mod earlier {
    use super::*;
    use edgefirst_kernels::prim::box_primitives::dist2bbox_anchor_f32;
    use edgefirst_kernels::prim::neon_baseline as nb;
    use edgefirst_kernels::Quant;

    /// One NEON dequantize of the whole level, four anchors per `dist2bbox_4anchors_f32_neon`
    /// step and a scalar tail.
    pub(super) fn ltrb_f32(raw: &Raw, q: Quantization, lvl: &LevelPlan) -> Vec<f32> {
        let n = lvl.h * lvl.w;
        let q: Quant = q.into();
        let mut deq = vec![0f32; n * 4];
        // SAFETY: NEON is part of the aarch64 baseline; the lengths match.
        unsafe {
            match raw {
                Raw::I8(v) => nb::dequant_i8_to_f32_neon(v, q, &mut deq),
                Raw::U8(v) => nb::dequant_u8_to_f32_neon(v, q, &mut deq),
                Raw::I16(v) => nb::dequant_i16_to_f32_neon(v, q, &mut deq),
                Raw::U16(v) => nb::dequant_u16_to_f32_neon(v, q, &mut deq),
                _ => unreachable!("integer inputs only"),
            }
        }
        let mut out = vec![0f32; n * 4];
        for a in (0..n / 4 * 4).step_by(4) {
            // SAFETY: as above; 16 values and 4 grid centres per step.
            unsafe {
                nb::dist2bbox_4anchors_f32_neon(
                    &deq[a * 4..a * 4 + 16],
                    &lvl.grid_x[a..a + 4],
                    &lvl.grid_y[a..a + 4],
                    lvl.stride,
                    &mut out[a * 4..a * 4 + 16],
                );
            }
        }
        for a in n / 4 * 4..n {
            let d = [deq[a * 4], deq[a * 4 + 1], deq[a * 4 + 2], deq[a * 4 + 3]];
            let b = dist2bbox_anchor_f32(d, lvl.grid_x[a], lvl.grid_y[a], lvl.stride);
            out[a * 4..a * 4 + 4].copy_from_slice(&b);
        }
        out
    }

    /// Per anchor: the NEON dequantize, the fused softmax (fp16 exp when `fp16`), scalar
    /// dist2bbox.
    pub(super) fn dfl16_f32(raw: &Raw, q: Quantization, lvl: &LevelPlan, fp16: bool) -> Vec<f32> {
        let rm = lvl.reg_max;
        let n = lvl.h * lvl.w;
        let q: Quant = q.into();
        let mut out = vec![0f32; n * 4];
        let mut deq = vec![0f32; 4 * rm];
        for a in 0..n {
            let r = a * 4 * rm..(a + 1) * 4 * rm;
            // SAFETY: NEON is part of the aarch64 baseline; fp16 only on FP16 tiers.
            let ltrb = unsafe {
                match raw {
                    Raw::I16(v) => nb::dequant_i16_to_f32_neon(&v[r], q, &mut deq),
                    Raw::U16(v) => nb::dequant_u16_to_f32_neon(&v[r], q, &mut deq),
                    _ => unreachable!("16-bit inputs only"),
                }
                if fp16 {
                    nb::softmax_weighted_sum_4sides_f32_neon_fp16(&deq, rm)
                } else {
                    nb::softmax_weighted_sum_4sides_f32_neon(&deq, rm)
                }
            };
            let b = dist2bbox_anchor_f32(ltrb, lvl.grid_x[a], lvl.grid_y[a], lvl.stride);
            out[a * 4..a * 4 + 4].copy_from_slice(&b);
        }
        out
    }
}

/// How a NEON cell's f32 output is checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum F32Check {
    /// Bit for bit against the scalar tier.
    Scalar,
    /// Bit for bit against the decoder's earlier NEON composition.
    Earlier,
    /// Within [`f32_bar`] of the float64 reference.
    Bar,
}

fn f32_check(kind: Kind, input: DType) -> F32Check {
    let integer = INTEGERS.contains(&input);
    match kind {
        Kind::Dfl if matches!(input, DType::I8 | DType::U8) => F32Check::Scalar,
        Kind::Dfl if integer => F32Check::Earlier,
        Kind::Ltrb if integer => F32Check::Earlier,
        Kind::Ltrb => F32Check::Scalar,
        _ => F32Check::Bar,
    }
}

/// Every box cell on every tier, at `reg_max` 16, 8, 17, 32 and 64. Prints the worst error per
/// cell.
#[test]
fn every_box_cell_matches_its_reference_on_every_tier() {
    let scalar = CpuFeatures::default();
    let mut runs = Vec::new();
    for kind in [Kind::Dfl, Kind::DflPerChannel, Kind::Ltrb] {
        let inputs: &[DType] = if kind == Kind::DflPerChannel {
            &INTEGERS
        } else {
            &INPUTS
        };
        let reg_maxes: &[usize] = if kind == Kind::Ltrb {
            &[1]
        } else {
            &[16, 8, 17, 32, 64]
        };
        for &input in inputs {
            for &rm in reg_maxes {
                runs.push((kind, input, rm));
            }
        }
    }
    for (kind, input, rm) in runs {
        let cases = cases(kind, input, rm);
        for (tier, f) in tiers() {
            let c32 = select(kind, input, DecodeDtype::F32, &f);
            let c16 = select(kind, input, DecodeDtype::F16, &f);
            let label = format!("{tier} {kind:?} rm {rm} {input:?}");
            let (mut w32, mut w16) = (0f64, 0f64);
            for case in &cases {
                let l32 = format!("{label} {} {:?} {:?}", c32.name(), case.q, case.zps.len());
                let (g32, _) = case.run(c32, DecodeDtype::F32);
                let (_, g16) = case.run(c16, DecodeDtype::F16);
                let l16 = format!("{label} {} {:?} {:?}", c16.name(), case.q, case.zps.len());
                if !f.neon_baseline {
                    // The scalar tier is the reference; its bounds are reported only.
                    w32 = w32.max(check_f32(case, &g32, false, &l32));
                    w16 = w16.max(check_f16(case, &g16, None, &l16));
                    continue;
                }
                match f32_check(kind, input) {
                    F32Check::Scalar => {
                        let (want, _) = case.run(
                            select(kind, input, DecodeDtype::F32, &scalar),
                            DecodeDtype::F32,
                        );
                        assert_same_f32(&g32, &want, &l32);
                    }
                    #[cfg(target_arch = "aarch64")]
                    F32Check::Earlier => {
                        let want = if kind == Kind::Ltrb {
                            earlier::ltrb_f32(&case.raw, case.q, &case.lvl)
                        } else {
                            let fp16 = c32.name().ends_with("NeonFp16");
                            earlier::dfl16_f32(&case.raw, case.q, &case.lvl, fp16)
                        };
                        assert_same_f32(&g32, &want, &l32);
                    }
                    _ => {}
                }
                let fp16_exp = c32.name().ends_with("NeonFp16");
                w32 = w32.max(check_f32(case, &g32, fp16_exp, &l32));
                if c16.name().ends_with("NeonFp16") {
                    w16 = w16.max(check_f16(case, &g16, Some(fp16_ulps(rm)), &l16));
                } else {
                    // The f32 path rounded once: this tier's f32 cell, narrowed.
                    let narrowed: Vec<f16> = g32.iter().map(|&v| f16::from_f32(v)).collect();
                    assert_same_f16(&g16, &narrowed, &l16);
                    w16 = w16.max(check_f16(case, &g16, Some(F16_ULPS), &l16));
                }
            }
            eprintln!(
                "{label}: {} worst {w32:.3} bars; {} worst {w16:.2} f16 ulps",
                c32.name(),
                c16.name()
            );
        }
    }
}

/// Float LTRB specials (NaN, ±inf, ±0, subnormals, huge) decode on NEON as on the scalar tier.
#[test]
fn ltrb_float_specials_match_scalar() {
    let specials = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -0.0,
        0.0,
        f32::from_bits(1),
        f32::MIN_POSITIVE,
        1e30,
        -1e30,
        3.5,
        f32::MAX,
        -7.25,
    ];
    for &(h, w, stride) in &LEVELS {
        let lvl = level(h, w, stride, 1);
        let n = h * w * 4;
        let v32: Vec<f32> = (0..n)
            .map(|i| specials[(i * 7 + i / 5) % specials.len()])
            .collect();
        let v16: Vec<f16> = v32.iter().map(|&v| f16::from_f32(v)).collect();
        for (tier, f) in tiers() {
            for q in [Quantization::identity(), Quantization::new(0.5, 3)] {
                for output in OUTPUTS {
                    for input in [DType::F16, DType::F32] {
                        let run = |f: &CpuFeatures| {
                            let d = BoxLevelDispatch::select(BoxEncoding::Direct, input, output, f)
                                .unwrap();
                            let (mut d32, mut d16) = (vec![0f32; n], vec![f16::ZERO; n]);
                            let src = match input {
                                DType::F16 => InputView::F16(&v16),
                                _ => InputView::F32(&v32),
                            };
                            let dst = match output {
                                DecodeDtype::F32 => DstSliceMut::F32(&mut d32),
                                DecodeDtype::F16 => DstSliceMut::F16(&mut d16),
                            };
                            d.run(src, q, &lvl, None, dst).unwrap();
                            (format!("{d:?}"), d32, d16)
                        };
                        let (name, g32, g16) = run(&f);
                        if name.ends_with("NeonFp16") {
                            continue;
                        }
                        let label = format!("{tier} {name} {q:?} {h}x{w}");
                        if output == DecodeDtype::F32 {
                            let (_, w32, _) = run(&CpuFeatures::default());
                            assert_same_f32(&g32, &w32, &label);
                        } else if f.neon_baseline {
                            let d = BoxLevelDispatch::select(
                                BoxEncoding::Direct,
                                input,
                                DecodeDtype::F32,
                                &f,
                            )
                            .unwrap();
                            let mut d32 = vec![0f32; n];
                            let src = match input {
                                DType::F16 => InputView::F16(&v16),
                                _ => InputView::F32(&v32),
                            };
                            d.run(src, q, &lvl, None, DstSliceMut::F32(&mut d32))
                                .unwrap();
                            let narrowed: Vec<f16> =
                                d32.iter().map(|&v| f16::from_f32(v)).collect();
                            assert_same_f16(&g16, &narrowed, &label);
                        }
                    }
                }
            }
        }
    }
}

/// Mismatched per-channel scale or zero-point counts are `InvalidConfig`, before any write.
#[test]
fn per_channel_rejects_mismatched_quantization() {
    let lvl = level(1, 3, 8.0, 16);
    let raw = vec![0i8; 3 * 64];
    for (_, f) in tiers() {
        for output in OUTPUTS {
            let d = DflPerChannelDispatch::select(output, &f);
            for (scales, zps) in [
                (vec![0.1f32; 63], vec![]),
                (vec![0.1f32; 64], vec![0i32; 3]),
                (vec![0.1f32; 65], vec![]),
            ] {
                let (mut d32, mut d16) = (vec![SENTINEL; 12], vec![f16::from_f32(SENTINEL); 12]);
                let dst = match output {
                    DecodeDtype::F32 => DstSliceMut::F32(&mut d32),
                    DecodeDtype::F16 => DstSliceMut::F16(&mut d16),
                };
                let r = d.run(InputView::I8(&raw), &scales, &zps, &lvl, dst);
                let label = format!("{d:?} {} scales {} zps", scales.len(), zps.len());
                assert!(
                    matches!(r, Err(DecoderError::InvalidConfig(_))),
                    "{label}: {r:?}"
                );
                assert_untouched(&d32, &d16, &label);
            }
        }
    }
}

/// Value the length tests pre-fill outputs with.
const SENTINEL: f32 = -7.5;

fn assert_untouched(d32: &[f32], d16: &[f16], label: &str) {
    assert!(
        d32.iter().all(|&v| v == SENTINEL),
        "{label}: f32 output written"
    );
    let s16 = f16::from_f32(SENTINEL);
    assert!(d16.iter().all(|&v| v == s16), "{label}: f16 output written");
}

/// A box level call whose input, output or grids do not hold `h · w` anchors, or whose DFL
/// `reg_max` is out of range, is `InvalidShape` on every tier and cell, before any write.
#[test]
fn box_cells_reject_length_mismatches_before_writing() {
    let n = 21;
    // (case, reg_max of the plan, input values, output values, grid centres)
    let shapes = |per: usize, dfl: bool| {
        let mut v = vec![
            ("short input", 16, n * per - 1, n * 4, n),
            ("long input", 16, n * per + 4, n * 4, n),
            ("short output", 16, n * per, n * 4 - 4, n),
            ("long output", 16, n * per, n * 4 + 4, n),
            ("short grid", 16, n * per, n * 4, n - 1),
        ];
        if dfl {
            v.push(("reg_max 0", 0, 0, n * 4, n));
            v.push(("reg_max 65", 65, n * 4 * 65, n * 4, n));
        }
        v
    };
    for (tier, f) in tiers() {
        for kind in [Kind::Dfl, Kind::DflPerChannel, Kind::Ltrb] {
            let inputs: &[DType] = if kind == Kind::DflPerChannel {
                &INTEGERS
            } else {
                &INPUTS
            };
            let per = if kind == Kind::Ltrb { 4 } else { 64 };
            for &input in inputs {
                for output in OUTPUTS {
                    let cell = select(kind, input, output, &f);
                    for (what, rm, n_in, n_out, n_grid) in shapes(per, kind != Kind::Ltrb) {
                        let rm = if kind == Kind::Ltrb { 1 } else { rm };
                        let mut lvl = level(1, n, 8.0, rm);
                        lvl.grid_x = lvl.grid_x[..n_grid].into();
                        lvl.grid_y = lvl.grid_y[..n_grid].into();
                        let x = vec![0.5; n_in];
                        let raw = Raw::quantize(&x, input, |_| (0.1, 0.0));
                        let ch = 4 * rm;
                        let (scales, zps) = (vec![0.1f32; ch], vec![0i32; ch]);
                        let (mut d32, mut d16) =
                            (vec![SENTINEL; n_out], vec![f16::from_f32(SENTINEL); n_out]);
                        let dst = match output {
                            DecodeDtype::F32 => DstSliceMut::F32(&mut d32),
                            DecodeDtype::F16 => DstSliceMut::F16(&mut d16),
                        };
                        let q = Quantization::new(0.1, 0);
                        let r = match cell {
                            Cell::Box(d) => d.run(raw.view(), q, &lvl, None, dst),
                            Cell::PerChannel(d) => d.run(raw.view(), &scales, &zps, &lvl, dst),
                        };
                        let label = format!("{tier} {} {what}", cell.name());
                        assert!(
                            matches!(r, Err(DecoderError::InvalidShape(_))),
                            "{label}: {r:?}"
                        );
                        assert_untouched(&d32, &d16, &label);
                    }
                }
            }
        }
    }
}
