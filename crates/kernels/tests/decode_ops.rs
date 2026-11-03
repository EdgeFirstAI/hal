// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! DFL, dist2bbox and fused DFL-box decode plans against a float64 reference.

mod common;

use edgefirst_kernels::prim::dfl_lut::{decode_dfl_lut, DflExpTable, DflGeom};
use edgefirst_kernels::prim::grids::make_anchor_grid;
use edgefirst_kernels::{
    BoxFormat, CpuFeatures, DType, DflBoxParams, DflBoxPlan, DflParams, DflPlan, Dist2BboxParams,
    Dist2BboxPlan, KernelError, Level, Pool, Quant, TensorMut, TensorRef, Tier, View, ViewMut,
};
use half::f16;
use rand::{rngs::StdRng, RngExt, SeedableRng};

const IN_DTYPES: [DType; 6] = [
    DType::I8,
    DType::U8,
    DType::I16,
    DType::U16,
    DType::F16,
    DType::F32,
];
const OUT_DTYPES: [DType; 5] = [DType::F32, DType::F16, DType::I8, DType::I16, DType::U16];
const FORMATS: [BoxFormat; 2] = [BoxFormat::Xywh, BoxFormat::Xyxy];
const REG_MAX: usize = 16;

fn level(w: usize, h: usize, stride: f32) -> Level {
    Level {
        anchors: w * h,
        grid_w: w,
        stride,
    }
}

/// yolov8n at 640×640: 8400 anchors.
fn yolov8n_640() -> Vec<Level> {
    vec![level(80, 80, 8.0), level(40, 40, 16.0), level(20, 20, 32.0)]
}

/// t-1297 at 1344×768: 21168 anchors.
fn t1297_1344x768() -> Vec<Level> {
    vec![
        level(168, 96, 8.0),
        level(84, 48, 16.0),
        level(42, 24, 32.0),
    ]
}

/// The yolov8n strides on small odd grids, so blocks end mid-level and on 4-anchor tails.
fn yolov8n_small() -> Vec<Level> {
    vec![level(13, 13, 8.0), level(7, 7, 16.0), level(3, 3, 32.0)]
}

/// The t-1297 strides and aspect on small grids.
fn t1297_small() -> Vec<Level> {
    vec![level(28, 16, 8.0), level(14, 8, 16.0), level(7, 4, 32.0)]
}

fn total(levels: &[Level]) -> usize {
    levels.iter().map(|l| l.anchors).sum()
}

/// Per-anchor `(gx, gy, stride)`: cell centres at +0.5, row-major within each level.
fn anchor_geometry(levels: &[Level]) -> Vec<(f64, f64, f64)> {
    levels
        .iter()
        .flat_map(|l| {
            (0..l.anchors).map(move |i| {
                (
                    (i % l.grid_w) as f64 + 0.5,
                    (i / l.grid_w) as f64 + 0.5,
                    f64::from(l.stride),
                )
            })
        })
        .collect()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// DFL logits shaped like a trained head: each side peaks at a bin and falls off with noise;
/// flat, one-hot and saturated sides are mixed in.
fn dfl_logits(n: usize, reg_max: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed | 1);
    let mut v = Vec::with_capacity(n * 4 * reg_max);
    for _ in 0..n * 4 {
        let kind = rng.next() % 16;
        let peak = rng.unit() * (reg_max - 1) as f64;
        let top = rng.unit() * 16.0 - 8.0;
        let fall = rng.unit() * 3.0;
        for i in 0..reg_max {
            v.push(match kind {
                0 => 30.0,
                1 => -30.0,
                2 if i == peak.round() as usize => 20.0,
                2 => -20.0,
                3 => top,
                _ => top - fall * (i as f64 - peak).abs() + (rng.unit() - 0.5) * 0.6,
            });
        }
    }
    v
}

/// LTRB distances in grid units, zeros included.
fn ltrb_distances(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = Rng(seed | 1);
    (0..n * 4)
        .map(|_| {
            if rng.next().is_multiple_of(16) {
                0.0
            } else {
                rng.unit() * 20.0
            }
        })
        .collect()
}

/// Input quantization per dtype for DFL logits; floats carry the identity.
fn logit_quant(d: DType) -> Quant {
    match d {
        DType::I8 => Quant::new(0.147, 8),
        DType::U8 => Quant::new(0.147, 136),
        DType::I16 => Quant::new(0.0007, -100),
        DType::U16 => Quant::new(0.0004, 30000),
        DType::F16 | DType::F32 => Quant::IDENTITY,
    }
}

/// Input quantization per dtype for LTRB distances.
fn ltrb_quant(d: DType) -> Quant {
    match d {
        DType::I8 => Quant::new(0.1, -128),
        DType::U8 => Quant::new(0.1, 0),
        DType::I16 => Quant::new(0.001, -32768),
        DType::U16 => Quant::new(0.001, 0),
        DType::F16 | DType::F32 => Quant::IDENTITY,
    }
}

/// Output quantization per dtype for boxes in pixels (`f32`/`f16` ignore it).
fn box_quant(d: DType) -> Quant {
    match d {
        DType::I8 => Quant::new(8.0, -60),
        DType::I16 => Quant::new(0.0625, 0),
        DType::U16 => Quant::new(0.05, 16384),
        _ => Quant::IDENTITY,
    }
}

/// Output quantization per dtype for distances in grid units.
fn ltrb_out_quant(d: DType) -> Quant {
    match d {
        DType::I8 => Quant::new(1.0 / 16.0, -128),
        DType::I16 => Quant::new(1.0 / 2048.0, 0),
        DType::U16 => Quant::new(1.0 / 4096.0, 0),
        _ => Quant::IDENTITY,
    }
}

#[derive(Clone)]
enum Data {
    I8(Vec<i8>),
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
}

fn transpose<T: Copy + Default>(v: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut t = vec![T::default(); v.len()];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = v[r * cols + c];
        }
    }
    t
}

impl Data {
    fn quantize(x: &[f64], d: DType, q: Quant) -> Self {
        let qi = |v: f64, lo: f64, hi: f64| {
            ((v / f64::from(q.scale)).round() + f64::from(q.zero_point)).clamp(lo, hi)
        };
        match d {
            DType::I8 => Self::I8(x.iter().map(|&v| qi(v, -128.0, 127.0) as i8).collect()),
            DType::U8 => Self::U8(x.iter().map(|&v| qi(v, 0.0, 255.0) as u8).collect()),
            DType::I16 => Self::I16(x.iter().map(|&v| qi(v, -32768.0, 32767.0) as i16).collect()),
            DType::U16 => Self::U16(x.iter().map(|&v| qi(v, 0.0, 65535.0) as u16).collect()),
            DType::F16 => Self::F16(x.iter().map(|&v| f16::from_f64(v)).collect()),
            DType::F32 => Self::F32(x.iter().map(|&v| v as f32).collect()),
        }
    }

    /// The values the plan sees: dequantized integers, or the floats themselves.
    fn real(&self, q: Quant) -> Vec<f64> {
        let (s, z) = (f64::from(q.scale), f64::from(q.zero_point));
        match self {
            Self::I8(v) => v.iter().map(|&x| (f64::from(x) - z) * s).collect(),
            Self::U8(v) => v.iter().map(|&x| (f64::from(x) - z) * s).collect(),
            Self::I16(v) => v.iter().map(|&x| (f64::from(x) - z) * s).collect(),
            Self::U16(v) => v.iter().map(|&x| (f64::from(x) - z) * s).collect(),
            Self::F16(v) => v.iter().map(|x| f64::from(x.to_f32())).collect(),
            Self::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        }
    }

    fn transposed(&self, rows: usize, cols: usize) -> Self {
        match self {
            Self::I8(v) => Self::I8(transpose(v, rows, cols)),
            Self::U8(v) => Self::U8(transpose(v, rows, cols)),
            Self::I16(v) => Self::I16(transpose(v, rows, cols)),
            Self::U16(v) => Self::U16(transpose(v, rows, cols)),
            Self::F16(v) => Self::F16(transpose(v, rows, cols)),
            Self::F32(v) => Self::F32(transpose(v, rows, cols)),
        }
    }

    fn view(&self, shape: &[usize], strides: &[isize]) -> TensorRef<'_> {
        match self {
            Self::I8(v) => TensorRef::I8(View::strided(v, shape, strides).unwrap()),
            Self::U8(v) => TensorRef::U8(View::strided(v, shape, strides).unwrap()),
            Self::I16(v) => TensorRef::I16(View::strided(v, shape, strides).unwrap()),
            Self::U16(v) => TensorRef::U16(View::strided(v, shape, strides).unwrap()),
            Self::F16(v) => TensorRef::F16(View::strided(v, shape, strides).unwrap()),
            Self::F32(v) => TensorRef::F32(View::strided(v, shape, strides).unwrap()),
        }
    }
}

/// Anchor-major or channel-major storage of a `[rows, cols]` input.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Layout {
    AnchorMajor,
    ChannelMajor,
}

/// The input as the plan reads it: anchor-major, or TFLite's `[cols, rows]` seen through strides.
struct Input {
    data: Data,
    rows: usize,
    cols: usize,
    layout: Layout,
}

impl Input {
    fn new(anchor_major: &Data, rows: usize, cols: usize, layout: Layout) -> Self {
        let data = match layout {
            Layout::AnchorMajor => anchor_major.clone(),
            Layout::ChannelMajor => anchor_major.transposed(rows, cols),
        };
        Self {
            data,
            rows,
            cols,
            layout,
        }
    }

    fn view(&self) -> TensorRef<'_> {
        let strides = match self.layout {
            Layout::AnchorMajor => [self.cols as isize, 1],
            Layout::ChannelMajor => [1, self.rows as isize],
        };
        self.data.view(&[self.rows, self.cols], &strides)
    }
}

enum Out {
    I8(Vec<i8>),
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
}

impl Out {
    /// A buffer filled with a sentinel, so unwritten elements show up.
    fn new(d: DType, len: usize) -> Self {
        match d {
            DType::I8 => Self::I8(vec![0x55; len]),
            DType::U8 => Self::U8(vec![0x55; len]),
            DType::I16 => Self::I16(vec![0x5555; len]),
            DType::U16 => Self::U16(vec![0x5555; len]),
            DType::F16 => Self::F16(vec![f16::NAN; len]),
            DType::F32 => Self::F32(vec![f32::NAN; len]),
        }
    }

    fn view(&mut self, shape: &[usize], strides: &[isize]) -> TensorMut<'_> {
        match self {
            Self::I8(v) => TensorMut::I8(ViewMut::strided(v, shape, strides).unwrap()),
            Self::U8(v) => TensorMut::U8(ViewMut::strided(v, shape, strides).unwrap()),
            Self::I16(v) => TensorMut::I16(ViewMut::strided(v, shape, strides).unwrap()),
            Self::U16(v) => TensorMut::U16(ViewMut::strided(v, shape, strides).unwrap()),
            Self::F16(v) => TensorMut::F16(ViewMut::strided(v, shape, strides).unwrap()),
            Self::F32(v) => TensorMut::F32(ViewMut::strided(v, shape, strides).unwrap()),
        }
    }

    fn contiguous(&mut self, rows: usize) -> TensorMut<'_> {
        self.view(&[rows, 4], &[4, 1])
    }

    fn values(&self) -> Vec<f64> {
        match self {
            Self::I8(v) => v.iter().map(|&x| f64::from(x)).collect(),
            Self::U8(v) => v.iter().map(|&x| f64::from(x)).collect(),
            Self::I16(v) => v.iter().map(|&x| f64::from(x)).collect(),
            Self::U16(v) => v.iter().map(|&x| f64::from(x)).collect(),
            Self::F16(v) => v.iter().map(|x| f64::from(x.to_f32())).collect(),
            Self::F32(v) => v.iter().map(|&x| f64::from(x)).collect(),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            Self::I8(v) => v.iter().map(|&x| x as u8).collect(),
            Self::U8(v) => v.clone(),
            Self::I16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::U16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::F16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }

    fn is_float(&self) -> bool {
        matches!(self, Self::F16(_) | Self::F32(_))
    }
}

/// Softmax over each side's bins and the bin-weighted sum, in f64: `[l, t, r, b]` per anchor.
fn ref_ltrb(logits: &[f64], reg_max: usize) -> Vec<[f64; 4]> {
    logits
        .chunks_exact(4 * reg_max)
        .map(|anchor| {
            let mut d = [0.0f64; 4];
            for (slot, side) in d.iter_mut().zip(anchor.chunks_exact(reg_max)) {
                let m = side.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = side.iter().map(|v| (v - m).exp()).collect();
                let sum: f64 = e.iter().sum();
                *slot = e.iter().enumerate().map(|(i, e)| e * i as f64).sum::<f64>() / sum;
            }
            d
        })
        .collect()
}

/// dist2bbox in f64, Ultralytics order, then the requested format.
fn ref_boxes(ltrb: &[[f64; 4]], geom: &[(f64, f64, f64)], format: BoxFormat) -> Vec<f64> {
    ltrb.iter()
        .zip(geom)
        .flat_map(|(&[l, t, r, b], &(gx, gy, s))| {
            let (xc, yc) = ((gx + (r - l) * 0.5) * s, (gy + (b - t) * 0.5) * s);
            let (w, h) = ((l + r) * s, (t + b) * s);
            match format {
                BoxFormat::Xywh => [xc, yc, w, h],
                BoxFormat::Xyxy => [xc - w / 2.0, yc - h / 2.0, xc + w / 2.0, yc + h / 2.0],
            }
        })
        .collect()
}

/// `v` rounded to f16 once, nearest even.
fn narrow(v: &[f32]) -> Vec<f16> {
    v.iter().map(|&x| f16::from_f32(x)).collect()
}

fn ulp_f32(v: f64) -> f64 {
    let v = (v.abs() as f32).max(f32::MIN_POSITIVE);
    f64::from(f32::from_bits(v.to_bits() + 1) - v)
}

fn ulp_f16(v: f64) -> f64 {
    let h = f16::from_f64(v.abs()).max(f16::MIN_POSITIVE);
    f64::from(f16::from_bits(h.to_bits() + 1).to_f32() - h.to_f32())
}

/// How far a float result may sit from the f64 reference, per value.
#[derive(Clone, Copy, Debug)]
enum Bar {
    /// f32 kernels: the decoder's DFL bar of 2e-4 px at stride 16, scaled by stride, plus 4 f32
    /// ulps of the value (this test's coordinates reach 1800 px, the decoder's stay under 150).
    F32,
    /// The fp16-exp kernel: 2% relative error per exp weight (the primitive's test bar) moves
    /// a side by at most `0.02 · (reg_max − 1)` grid units.
    Fp16Exp { reg_max: usize },
    /// f16 outputs: one f16 ulp of the value, plus 3% of a grid unit for the FP16 kernels'
    /// roundings (the f16 exp weights and sums, each distance, and the box arithmetic).
    F16,
}

impl Bar {
    /// `scale` is pixels per grid unit (the stride, or 1 for distances); `w` the reference value.
    fn at(self, scale: f64, w: f64) -> f64 {
        match self {
            Self::F32 => 2e-4 * scale / 16.0 + 4.0 * ulp_f32(w),
            Self::Fp16Exp { reg_max } => 2.0 * 0.02 * (reg_max - 1) as f64 * scale,
            Self::F16 => ulp_f16(w) + 0.03 * scale,
        }
    }

    fn for_output(out: DType, tier: Tier, reg_max: usize) -> Self {
        match (out, tier) {
            (DType::F16, _) => Self::F16,
            (_, Tier::NeonFp16) => Self::Fp16Exp { reg_max },
            _ => Self::F32,
        }
    }
}

/// Largest errors seen per output dtype: float in absolute units, quantized in LSB.
#[derive(Default)]
struct Envelope {
    worst: Vec<(String, f64)>,
}

impl Envelope {
    fn note(&mut self, key: String, err: f64) {
        match self.worst.iter_mut().find(|(k, _)| *k == key) {
            Some((_, e)) => *e = e.max(err),
            None => self.worst.push((key, err)),
        }
    }

    fn report(&self, test: &str) {
        for (k, e) in &self.worst {
            eprintln!("{test}: {k}: max error {e:.3e}");
        }
    }
}

/// Checks `got` against the f64 reference and returns the largest error (LSB when quantized).
fn check(label: &str, got: &Out, want: &[f64], q: Quant, bar: Bar, scale: &[f64]) -> f64 {
    let g = got.values();
    assert_eq!(g.len(), want.len(), "{label}: length");
    let mut worst = 0.0f64;
    if got.is_float() {
        for (i, (&g, &w)) in g.iter().zip(want).enumerate() {
            let err = (g - w).abs();
            let limit = bar.at(scale[i / 4], w);
            assert!(
                err <= limit,
                "{label}: value {i}: got {g} want {w} err {err:.3e} > {limit:.3e}"
            );
            worst = worst.max(err);
        }
    } else {
        let (lo, hi) = match got {
            Out::I8(_) => (-128.0, 127.0),
            Out::U8(_) => (0.0, 255.0),
            Out::I16(_) => (-32768.0, 32767.0),
            _ => (0.0, 65535.0),
        };
        for (i, (&g, &w)) in g.iter().zip(want).enumerate() {
            let r = (w / f64::from(q.scale)).round() + f64::from(q.zero_point);
            let err = (g - r.clamp(lo, hi)).abs();
            assert!(
                err <= 1.0,
                "{label}: value {i}: got {g} want {r} (from {w})"
            );
            worst = worst.max(err);
        }
    }
    worst
}

fn dfl_box_params(
    input_dtype: DType,
    output_dtype: DType,
    levels: Vec<Level>,
    format: BoxFormat,
) -> DflBoxParams {
    DflBoxParams {
        input_dtype,
        input: logit_quant(input_dtype),
        output_dtype,
        output: box_quant(output_dtype),
        reg_max: REG_MAX,
        levels,
        format,
    }
}

#[test]
fn plans_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DflPlan>();
    assert_send_sync::<Dist2BboxPlan>();
    assert_send_sync::<DflBoxPlan>();
    assert_send_sync::<DflBoxParams>();
}

/// One input dtype on one level set: inputs in both layouts, and the f64 reference per format.
struct DflCase {
    rows: usize,
    inputs: [Input; 2],
    want: [Vec<f64>; 2],
    scale: Vec<f64>,
}

impl DflCase {
    fn new(d: DType, levels: &[Level], seed: u64) -> Self {
        let rows = total(levels);
        let cols = 4 * REG_MAX;
        let data = Data::quantize(&dfl_logits(rows, REG_MAX, seed), d, logit_quant(d));
        let ltrb = ref_ltrb(&data.real(logit_quant(d)), REG_MAX);
        let geom = anchor_geometry(levels);
        Self {
            rows,
            inputs: [
                Input::new(&data, rows, cols, Layout::AnchorMajor),
                Input::new(&data, rows, cols, Layout::ChannelMajor),
            ],
            want: FORMATS.map(|f| ref_boxes(&ltrb, &geom, f)),
            scale: geom.iter().map(|g| g.2).collect(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_dfl_box_case(
    feats: CpuFeatures,
    case: &DflCase,
    levels: &[Level],
    input_dtype: DType,
    output_dtype: DType,
    format_idx: usize,
    layout_idx: usize,
    env: &mut Envelope,
) {
    let format = FORMATS[format_idx];
    let params = dfl_box_params(input_dtype, output_dtype, levels.to_vec(), format);
    let plan = DflBoxPlan::with_features(params, feats).unwrap();
    let input = &case.inputs[layout_idx];
    let mut out = Out::new(output_dtype, case.rows * 4);
    plan.run(input.view(), out.contiguous(case.rows), None)
        .unwrap();
    let label = format!(
        "{input_dtype:?}->{output_dtype:?} {format:?} {:?} {} anchors tier {:?}",
        input.layout,
        case.rows,
        plan.tier()
    );
    let bar = Bar::for_output(output_dtype, plan.tier(), REG_MAX);
    let err = check(
        &label,
        &out,
        &case.want[format_idx],
        box_quant(output_dtype),
        bar,
        &case.scale,
    );
    env.note(format!("{output_dtype:?} out, {:?} tier", plan.tier()), err);
}

#[test]
fn dfl_box_matches_f64_reference() {
    // Full anchor counts for one combination per output dtype.
    let full = [
        (DType::F32, DType::I8, 0, 0, true),
        (DType::F16, DType::I16, 1, 1, false),
        (DType::I8, DType::U8, 1, 0, false),
        (DType::I16, DType::F32, 0, 1, true),
        (DType::U16, DType::F16, 1, 1, false),
    ];
    let mut env = Envelope::default();
    for feats in common::tiers() {
        for &(out, input, format, layout, v8) in &full {
            let levels = if v8 { yolov8n_640() } else { t1297_1344x768() };
            let case = DflCase::new(input, &levels, 41);
            run_dfl_box_case(feats, &case, &levels, input, out, format, layout, &mut env);
        }
        // Every combination on small grids with the same strides.
        for (li, levels) in [yolov8n_small(), t1297_small()].iter().enumerate() {
            for (di, &input) in IN_DTYPES.iter().enumerate() {
                let case = DflCase::new(input, levels, 7 + (li * 8 + di) as u64);
                for &out in &OUT_DTYPES {
                    for format in 0..2 {
                        for layout in 0..2 {
                            run_dfl_box_case(
                                feats, &case, levels, input, out, format, layout, &mut env,
                            );
                        }
                    }
                }
            }
        }
    }
    env.report("dfl_box");
}

#[test]
fn dfl_matches_f64_reference() {
    let mut env = Envelope::default();
    for feats in common::tiers() {
        for (levels, seed) in [(yolov8n_small(), 3u64), (yolov8n_640(), 5)] {
            let rows = total(&levels);
            let full = rows > 1000;
            for &input_dtype in &IN_DTYPES {
                if full && input_dtype != DType::I16 {
                    continue;
                }
                let q = logit_quant(input_dtype);
                let data = Data::quantize(&dfl_logits(rows, REG_MAX, seed), input_dtype, q);
                let want: Vec<f64> = ref_ltrb(&data.real(q), REG_MAX)
                    .into_iter()
                    .flatten()
                    .collect();
                let scale = vec![1.0; rows];
                for &output_dtype in &OUT_DTYPES {
                    let plan = DflPlan::with_features(
                        DflParams {
                            input_dtype,
                            input: q,
                            output_dtype,
                            output: ltrb_out_quant(output_dtype),
                            reg_max: REG_MAX,
                        },
                        feats,
                    )
                    .unwrap();
                    for layout in [Layout::AnchorMajor, Layout::ChannelMajor] {
                        let input = Input::new(&data, rows, 4 * REG_MAX, layout);
                        let mut out = Out::new(output_dtype, rows * 4);
                        plan.run(input.view(), out.contiguous(rows), None).unwrap();
                        let label = format!(
                            "dfl {input_dtype:?}->{output_dtype:?} {layout:?} tier {:?}",
                            plan.tier()
                        );
                        let bar = Bar::for_output(output_dtype, plan.tier(), REG_MAX);
                        let err = check(
                            &label,
                            &out,
                            &want,
                            ltrb_out_quant(output_dtype),
                            bar,
                            &scale,
                        );
                        env.note(format!("{output_dtype:?} out, {:?} tier", plan.tier()), err);
                    }
                }
            }
        }
    }
    env.report("dfl");
}

/// Runs a `Dist2BboxPlan` on LTRB `data` and checks it against the f64 reference.
#[allow(clippy::too_many_arguments)]
fn run_dist2bbox_case(
    feats: CpuFeatures,
    levels: &[Level],
    data: &Data,
    input_dtype: DType,
    input_q: Quant,
    output_dtype: DType,
    format: BoxFormat,
    layout: Layout,
    env: &mut Envelope,
) {
    let rows = total(levels);
    let geom = anchor_geometry(levels);
    let real = data.real(input_q);
    let ltrb: Vec<[f64; 4]> = real
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect();
    let want = ref_boxes(&ltrb, &geom, format);
    let scale: Vec<f64> = geom.iter().map(|g| g.2).collect();
    let plan = Dist2BboxPlan::with_features(
        Dist2BboxParams {
            input_dtype,
            input: input_q,
            output_dtype,
            output: box_quant(output_dtype),
            levels: levels.to_vec(),
            format,
        },
        feats,
    )
    .unwrap();
    let input = Input::new(data, rows, 4, layout);
    let mut out = Out::new(output_dtype, rows * 4);
    plan.run(input.view(), out.contiguous(rows), None).unwrap();
    let label = format!(
        "dist2bbox {input_dtype:?}->{output_dtype:?} {format:?} {layout:?} {rows} anchors tier {:?}",
        plan.tier()
    );
    let bar = match output_dtype {
        DType::F16 => Bar::F16,
        _ => Bar::F32,
    };
    let err = check(&label, &out, &want, box_quant(output_dtype), bar, &scale);
    env.note(format!("{output_dtype:?} out, {:?} tier", plan.tier()), err);
}

#[test]
fn dist2bbox_matches_f64_reference() {
    let mut env = Envelope::default();
    for feats in common::tiers() {
        for (li, levels) in [yolov8n_small(), t1297_small()].iter().enumerate() {
            for (di, &input_dtype) in IN_DTYPES.iter().enumerate() {
                let q = ltrb_quant(input_dtype);
                let data = Data::quantize(
                    &ltrb_distances(total(levels), (li * 8 + di) as u64 + 1),
                    input_dtype,
                    q,
                );
                for &output_dtype in &OUT_DTYPES {
                    for format in FORMATS {
                        for layout in [Layout::AnchorMajor, Layout::ChannelMajor] {
                            run_dist2bbox_case(
                                feats,
                                levels,
                                &data,
                                input_dtype,
                                q,
                                output_dtype,
                                format,
                                layout,
                                &mut env,
                            );
                        }
                    }
                }
            }
        }
    }
    env.report("dist2bbox");
}

#[test]
fn dist2bbox_yolo26_reg_max_1() {
    let levels = t1297_1344x768();
    assert_eq!(total(&levels), 21168);
    let q = Quant::new(0.08, -128);
    let data = Data::quantize(&ltrb_distances(21168, 26), DType::I8, q);
    let mut env = Envelope::default();
    for feats in common::tiers() {
        for format in FORMATS {
            for layout in [Layout::AnchorMajor, Layout::ChannelMajor] {
                for output_dtype in [DType::F32, DType::F16, DType::I16] {
                    run_dist2bbox_case(
                        feats,
                        &levels,
                        &data,
                        DType::I8,
                        q,
                        output_dtype,
                        format,
                        layout,
                        &mut env,
                    );
                }
            }
        }
    }
    env.report("dist2bbox_yolo26");
}

#[test]
fn dfl_and_dist2bbox_compose_to_dfl_box() {
    for feats in common::tiers() {
        for (input_dtype, levels) in [
            (DType::I8, yolov8n_640()),
            (DType::I16, yolov8n_small()),
            (DType::F32, t1297_small()),
        ] {
            let rows = total(&levels);
            let q = logit_quant(input_dtype);
            let data = Data::quantize(&dfl_logits(rows, REG_MAX, 9), input_dtype, q);
            let input = Input::new(&data, rows, 4 * REG_MAX, Layout::AnchorMajor);
            let dfl = DflPlan::with_features(
                DflParams {
                    input_dtype,
                    input: q,
                    output_dtype: DType::F32,
                    output: Quant::IDENTITY,
                    reg_max: REG_MAX,
                },
                feats,
            )
            .unwrap();
            let mut ltrb = vec![0.0f32; rows * 4];
            dfl.run(
                input.view(),
                TensorMut::F32(ViewMut::contiguous(&mut ltrb, &[rows, 4]).unwrap()),
                None,
            )
            .unwrap();
            for format in FORMATS {
                let d2b = Dist2BboxPlan::with_features(
                    Dist2BboxParams {
                        input_dtype: DType::F32,
                        input: Quant::IDENTITY,
                        output_dtype: DType::F32,
                        output: Quant::IDENTITY,
                        levels: levels.clone(),
                        format,
                    },
                    feats,
                )
                .unwrap();
                let mut composed = vec![0.0f32; rows * 4];
                d2b.run(
                    TensorRef::F32(View::contiguous(&ltrb, &[rows, 4]).unwrap()),
                    TensorMut::F32(ViewMut::contiguous(&mut composed, &[rows, 4]).unwrap()),
                    None,
                )
                .unwrap();
                let fused = DflBoxPlan::with_features(
                    DflBoxParams {
                        input_dtype,
                        input: q,
                        output_dtype: DType::F32,
                        output: Quant::IDENTITY,
                        reg_max: REG_MAX,
                        levels: levels.clone(),
                        format,
                    },
                    feats,
                )
                .unwrap();
                let mut boxes = vec![0.0f32; rows * 4];
                fused
                    .run(
                        input.view(),
                        TensorMut::F32(ViewMut::contiguous(&mut boxes, &[rows, 4]).unwrap()),
                        None,
                    )
                    .unwrap();
                for (i, (&c, &b)) in composed.iter().zip(&boxes).enumerate() {
                    assert!(
                        (c - b).abs() <= 1e-5 * b.abs().max(1.0),
                        "{input_dtype:?} {format:?} value {i}: composed {c} fused {b}"
                    );
                }
            }
        }
    }
}

#[test]
fn single_level_call_matches_decoder_level_path() {
    let (w, h, stride) = (80usize, 80usize, 8.0f32);
    let n = w * h;
    let q = Quant::new(0.147, 8);
    let input = match Data::quantize(&dfl_logits(n, REG_MAX, 77), DType::I8, q) {
        Data::I8(v) => v,
        _ => unreachable!(),
    };
    let unsigned: Vec<u8> = input.iter().map(|&b| (i16::from(b) + 128) as u8).collect();
    let (grid_x, grid_y) = make_anchor_grid(h, w);
    let geom = DflGeom {
        reg_max: REG_MAX,
        anchors: n,
        grid_x: &grid_x,
        grid_y: &grid_y,
        stride,
    };
    let table = DflExpTable::new(q.scale);
    let mut want = vec![0.0f32; n * 4];
    decode_dfl_lut(&input, &table, &geom, &mut want);
    let mut want16 = vec![f16::ZERO; n * 4];
    decode_dfl_lut(&input, &table, &geom, &mut want16);
    let transposed = transpose(&input, n, 4 * REG_MAX);
    for feats in common::tiers() {
        for (input_dtype, input_q) in [(DType::I8, q), (DType::U8, Quant::new(0.147, 136))] {
            for output_dtype in [DType::F32, DType::F16] {
                let plan = DflBoxPlan::with_features(
                    DflBoxParams {
                        input_dtype,
                        input: input_q,
                        output_dtype,
                        output: Quant::IDENTITY,
                        reg_max: REG_MAX,
                        levels: vec![level(w, h, stride)],
                        format: BoxFormat::Xywh,
                    },
                    feats,
                )
                .unwrap();
                let view = match input_dtype {
                    DType::I8 => TensorRef::I8(View::contiguous(&input, &[n, 64]).unwrap()),
                    _ => TensorRef::U8(View::contiguous(&unsigned, &[n, 64]).unwrap()),
                };
                let mut out = Out::new(output_dtype, n * 4);
                plan.run(view, out.contiguous(n), None).unwrap();
                match &out {
                    Out::F32(v) => assert_eq!(v, &want, "{input_dtype:?} tier {:?}", plan.tier()),
                    // The FP16 tier's f16 kernel is checked against its bound instead.
                    Out::F16(_) if plan.tier() == Tier::NeonFp16 => {}
                    Out::F16(v) => {
                        assert_eq!(v, &want16, "{input_dtype:?} f16 tier {:?}", plan.tier())
                    }
                    _ => unreachable!(),
                }
            }
        }
        // The channel-major layout decodes to the same bits.
        let plan = DflBoxPlan::with_features(
            DflBoxParams {
                input_dtype: DType::I8,
                input: q,
                output_dtype: DType::F32,
                output: Quant::IDENTITY,
                reg_max: REG_MAX,
                levels: vec![level(w, h, stride)],
                format: BoxFormat::Xywh,
            },
            feats,
        )
        .unwrap();
        let mut got = vec![0.0f32; n * 4];
        plan.run(
            TensorRef::I8(View::strided(&transposed, &[n, 64], &[1, n as isize]).unwrap()),
            TensorMut::F32(ViewMut::contiguous(&mut got, &[n, 4]).unwrap()),
            None,
        )
        .unwrap();
        assert_eq!(got, want, "channel-major, tier {:?}", plan.tier());
    }
}

/// The decoder's per-level box loops (`level_box.rs`), rebuilt from the same `prim` calls.
mod decoder_level {
    use super::*;
    use edgefirst_kernels::prim::box_primitives::{dist2bbox_anchor_f32, weighted_sum_4sides_f32};
    use edgefirst_kernels::prim::dequant::{dequant_i16_to_f32, dequant_i8_to_f32};
    use edgefirst_kernels::prim::softmax::softmax_inplace_f32;

    /// DFL i16 to f32: scalar, or the fused NEON kernel of `tier`.
    pub fn dfl_i16_f32(
        input: &[i16],
        q: Quant,
        gx: &[f32],
        gy: &[f32],
        s: f32,
        tier: Tier,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; gx.len() * 4];
        let mut deq = [0.0f32; 64];
        for (a, (bins, o)) in input
            .chunks_exact(64)
            .zip(out.chunks_exact_mut(4))
            .enumerate()
        {
            let ltrb = match tier {
                Tier::Scalar => {
                    dequant_i16_to_f32(bins, q, &mut deq);
                    for side in deq.chunks_exact_mut(16) {
                        softmax_inplace_f32(side);
                    }
                    weighted_sum_4sides_f32(&deq, 16)
                }
                #[cfg(target_arch = "aarch64")]
                Tier::Neon | Tier::NeonFp16 => unsafe {
                    use edgefirst_kernels::prim::neon_baseline as nb;
                    nb::dequant_i16_to_f32_neon(bins, q, &mut deq);
                    if tier == Tier::Neon {
                        nb::softmax_weighted_sum_4sides_f32_neon(&deq, 16)
                    } else {
                        nb::softmax_weighted_sum_4sides_f32_neon_fp16(&deq, 16)
                    }
                },
                _ => unreachable!("tier {tier:?}"),
            };
            o.copy_from_slice(&dist2bbox_anchor_f32(ltrb, gx[a], gy[a], s));
        }
        out
    }

    /// LTRB i8 to f32: per-anchor scalar, or one NEON dequantize and four anchors per step.
    pub fn ltrb_i8_f32(
        input: &[i8],
        q: Quant,
        gx: &[f32],
        gy: &[f32],
        s: f32,
        neon: bool,
    ) -> Vec<f32> {
        let n = gx.len();
        let mut out = vec![0.0f32; n * 4];
        if !neon {
            for (a, (l, o)) in input
                .chunks_exact(4)
                .zip(out.chunks_exact_mut(4))
                .enumerate()
            {
                let mut d = [0.0f32; 4];
                dequant_i8_to_f32(l, q, &mut d);
                o.copy_from_slice(&dist2bbox_anchor_f32(d, gx[a], gy[a], s));
            }
            return out;
        }
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use edgefirst_kernels::prim::neon_baseline as nb;
            let mut deq = vec![0.0f32; n * 4];
            nb::dequant_i8_to_f32_neon(input, q, &mut deq);
            for b in 0..n / 4 {
                let a = b * 4;
                nb::dist2bbox_4anchors_f32_neon(
                    &deq[a * 4..a * 4 + 16],
                    &gx[a..a + 4],
                    &gy[a..a + 4],
                    s,
                    &mut out[a * 4..a * 4 + 16],
                );
            }
            for a in n / 4 * 4..n {
                let d = [deq[a * 4], deq[a * 4 + 1], deq[a * 4 + 2], deq[a * 4 + 3]];
                out[a * 4..a * 4 + 4].copy_from_slice(&dist2bbox_anchor_f32(d, gx[a], gy[a], s));
            }
        }
        out
    }
}

#[test]
fn single_level_calls_match_decoder_kernels() {
    for (w, h, stride) in [(80usize, 80usize, 8.0f32), (13, 13, 32.0)] {
        let n = w * h;
        let (gx, gy) = make_anchor_grid(h, w);
        let lv = vec![level(w, h, stride)];
        let q16 = logit_quant(DType::I16);
        let Data::I16(bins) = Data::quantize(&dfl_logits(n, REG_MAX, 3), DType::I16, q16) else {
            unreachable!()
        };
        let q8 = ltrb_quant(DType::I8);
        let Data::I8(ltrb) = Data::quantize(&ltrb_distances(n, 4), DType::I8, q8) else {
            unreachable!()
        };
        for feats in common::tiers() {
            for output_dtype in [DType::F32, DType::F16] {
                let plan = DflBoxPlan::with_features(
                    DflBoxParams {
                        input_dtype: DType::I16,
                        input: q16,
                        output_dtype,
                        output: Quant::IDENTITY,
                        reg_max: REG_MAX,
                        levels: lv.clone(),
                        format: BoxFormat::Xywh,
                    },
                    feats,
                )
                .unwrap();
                let mut out = Out::new(output_dtype, n * 4);
                plan.run(
                    TensorRef::I16(View::contiguous(&bins, &[n, 64]).unwrap()),
                    out.contiguous(n),
                    None,
                )
                .unwrap();
                // f16 outputs are the tier's f32 result narrowed once, except on the FP16 tier.
                let want = decoder_level::dfl_i16_f32(&bins, q16, &gx, &gy, stride, plan.tier());
                match &out {
                    Out::F32(v) => assert_eq!(v, &want, "dfl i16 f32 tier {:?}", plan.tier()),
                    Out::F16(_) if plan.tier() == Tier::NeonFp16 => {}
                    Out::F16(v) => {
                        assert_eq!(v, &narrow(&want), "dfl i16 f16 tier {:?}", plan.tier())
                    }
                    _ => unreachable!(),
                }

                let plan = Dist2BboxPlan::with_features(
                    Dist2BboxParams {
                        input_dtype: DType::I8,
                        input: q8,
                        output_dtype,
                        output: Quant::IDENTITY,
                        levels: lv.clone(),
                        format: BoxFormat::Xywh,
                    },
                    feats,
                )
                .unwrap();
                let mut out = Out::new(output_dtype, n * 4);
                plan.run(
                    TensorRef::I8(View::contiguous(&ltrb, &[n, 4]).unwrap()),
                    out.contiguous(n),
                    None,
                )
                .unwrap();
                let neon = plan.tier() != Tier::Scalar;
                let want = decoder_level::ltrb_i8_f32(&ltrb, q8, &gx, &gy, stride, neon);
                match &out {
                    Out::F32(v) => assert_eq!(v, &want, "ltrb i8 f32 tier {:?}", plan.tier()),
                    Out::F16(_) if plan.tier() == Tier::NeonFp16 => {}
                    Out::F16(v) => {
                        assert_eq!(v, &narrow(&want), "ltrb i8 f16 tier {:?}", plan.tier())
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}

#[test]
fn run_rejects_anchor_total_mismatch() {
    let levels = yolov8n_640();
    let plan = DflBoxPlan::new(dfl_box_params(
        DType::I8,
        DType::F32,
        levels.clone(),
        BoxFormat::Xywh,
    ))
    .unwrap();
    for rows in [8399usize, 8401, 0] {
        let input = vec![0i8; rows * 64];
        let mut out = vec![7.0f32; rows * 4];
        let r = plan.run(
            TensorRef::I8(View::contiguous(&input, &[rows, 64]).unwrap()),
            TensorMut::F32(ViewMut::contiguous(&mut out, &[rows, 4]).unwrap()),
            None,
        );
        assert!(
            matches!(r, Err(KernelError::InvalidArgument(_))),
            "{rows}: {r:?}"
        );
        assert!(out.iter().all(|&v| v == 7.0));
    }
    let plan = Dist2BboxPlan::new(Dist2BboxParams {
        input_dtype: DType::F32,
        input: Quant::IDENTITY,
        output_dtype: DType::F32,
        output: Quant::IDENTITY,
        levels,
        format: BoxFormat::Xyxy,
    })
    .unwrap();
    let input = vec![0.0f32; 8401 * 4];
    let mut out = vec![7.0f32; 8401 * 4];
    let r = plan.run(
        TensorRef::F32(View::contiguous(&input, &[8401, 4]).unwrap()),
        TensorMut::F32(ViewMut::contiguous(&mut out, &[8401, 4]).unwrap()),
        None,
    );
    assert!(matches!(r, Err(KernelError::InvalidArgument(_))), "{r:?}");
    assert!(out.iter().all(|&v| v == 7.0));
}

#[test]
fn new_rejects_bad_params() {
    let invalid = |r: Result<(), KernelError>, what: &str| {
        assert!(
            matches!(r, Err(KernelError::InvalidArgument(_))),
            "{what}: {r:?}"
        );
    };
    let good = dfl_box_params(DType::I8, DType::I8, yolov8n_640(), BoxFormat::Xywh);
    assert!(DflBoxPlan::new(good.clone()).is_ok());
    let dfl_box = |f: &dyn Fn(&mut DflBoxParams)| {
        let mut p = good.clone();
        f(&mut p);
        DflBoxPlan::new(p).map(|_| ())
    };
    invalid(dfl_box(&|p| p.reg_max = 0), "reg_max 0");
    invalid(dfl_box(&|p| p.levels[1].grid_w = 0), "grid_w 0");
    invalid(dfl_box(&|p| p.levels[2].anchors = 401), "anchors % grid_w");
    invalid(dfl_box(&|p| p.levels.clear()), "no levels");
    invalid(dfl_box(&|p| p.levels[0].stride = f32::NAN), "stride NaN");
    invalid(dfl_box(&|p| p.levels[0].stride = 0.0), "stride 0");
    invalid(
        dfl_box(&|p| p.levels[0].anchors = usize::MAX - usize::MAX % 80),
        "anchor total overflows",
    );
    invalid(dfl_box(&|p| p.output.scale = f32::NAN), "out scale NaN");
    invalid(dfl_box(&|p| p.output.scale = 0.0), "out scale 0");
    invalid(dfl_box(&|p| p.output.scale = -1.0), "out scale negative");
    invalid(
        dfl_box(&|p| p.output.scale = f32::INFINITY),
        "out scale inf",
    );
    invalid(dfl_box(&|p| p.output.scale = 1e-40), "out scale too small");
    invalid(
        dfl_box(&|p| p.output.zero_point = 128),
        "out zero point range",
    );
    invalid(dfl_box(&|p| p.input.scale = f32::NAN), "in scale NaN");
    invalid(dfl_box(&|p| p.input.scale = 0.0), "in scale 0");
    invalid(
        dfl_box(&|p| p.input.zero_point = -129),
        "in zero point range",
    );
    assert!(matches!(
        dfl_box(&|p| p.reg_max = 65),
        Err(KernelError::Unsupported(_))
    ));
    assert!(dfl_box(&|p| p.reg_max = 64).is_ok());
    assert!(dfl_box(&|p| p.levels[0].anchors = 0).is_ok());
    assert!(dfl_box(&|p| p.levels.push(level(1, 0, 8.0))).is_ok());

    let dfl = |reg_max: usize, output: Quant| {
        DflPlan::new(DflParams {
            input_dtype: DType::U16,
            input: Quant::new(0.001, 0),
            output_dtype: DType::U8,
            output,
            reg_max,
        })
        .map(|_| ())
    };
    assert!(dfl(16, Quant::new(0.1, 0)).is_ok());
    invalid(dfl(0, Quant::new(0.1, 0)), "dfl reg_max 0");
    invalid(dfl(16, Quant::new(f32::NAN, 0)), "dfl out scale NaN");
    invalid(dfl(16, Quant::new(0.1, 256)), "dfl out zero point");

    let d2b = |levels: Vec<Level>, output: Quant| {
        Dist2BboxPlan::new(Dist2BboxParams {
            input_dtype: DType::I16,
            input: Quant::new(0.01, 0),
            output_dtype: DType::U16,
            output,
            levels,
            format: BoxFormat::Xyxy,
        })
        .map(|_| ())
    };
    assert!(d2b(t1297_1344x768(), Quant::new(0.1, 0)).is_ok());
    invalid(
        d2b(vec![level(0, 4, 8.0)], Quant::new(0.1, 0)),
        "d2b grid_w 0",
    );
    invalid(
        d2b(
            vec![Level {
                anchors: 10,
                grid_w: 4,
                stride: 8.0,
            }],
            Quant::new(0.1, 0),
        ),
        "d2b anchors % grid_w",
    );
    invalid(
        d2b(t1297_1344x768(), Quant::new(f32::NAN, 0)),
        "d2b out scale NaN",
    );
    invalid(
        d2b(t1297_1344x768(), Quant::new(0.1, -1)),
        "d2b out zero point",
    );
}

#[test]
fn float_input_ignores_quant() {
    let levels = yolov8n_small();
    let rows = total(&levels);
    let logits = dfl_logits(rows, REG_MAX, 13);
    let ltrb = ltrb_distances(rows, 14);
    let odd = [
        Quant::new(0.5, 3),
        Quant::new(f32::NAN, 7),
        Quant::new(0.0, 0),
    ];
    for feats in common::tiers() {
        for input_dtype in [DType::F16, DType::F32] {
            let data = Data::quantize(&logits, input_dtype, Quant::IDENTITY);
            let dist = Data::quantize(&ltrb, input_dtype, Quant::IDENTITY);
            let run = |input: Quant| -> [Vec<u8>; 3] {
                let mut p =
                    dfl_box_params(input_dtype, DType::F32, levels.clone(), BoxFormat::Xyxy);
                p.input = input;
                let plan = DflBoxPlan::with_features(p, feats).unwrap();
                let mut boxes = Out::new(DType::F32, rows * 4);
                plan.run(
                    data.view(&[rows, 64], &[64, 1]),
                    boxes.contiguous(rows),
                    None,
                )
                .unwrap();
                let plan = DflPlan::with_features(
                    DflParams {
                        input_dtype,
                        input,
                        output_dtype: DType::F16,
                        output: Quant::IDENTITY,
                        reg_max: REG_MAX,
                    },
                    feats,
                )
                .unwrap();
                let mut dists = Out::new(DType::F16, rows * 4);
                plan.run(
                    data.view(&[rows, 64], &[64, 1]),
                    dists.contiguous(rows),
                    None,
                )
                .unwrap();
                let plan = Dist2BboxPlan::with_features(
                    Dist2BboxParams {
                        input_dtype,
                        input,
                        output_dtype: DType::I16,
                        output: box_quant(DType::I16),
                        levels: levels.clone(),
                        format: BoxFormat::Xywh,
                    },
                    feats,
                )
                .unwrap();
                let mut d2b = Out::new(DType::I16, rows * 4);
                plan.run(dist.view(&[rows, 4], &[4, 1]), d2b.contiguous(rows), None)
                    .unwrap();
                [boxes.bytes(), dists.bytes(), d2b.bytes()]
            };
            let want = run(Quant::IDENTITY);
            for q in odd {
                assert_eq!(run(q), want, "{input_dtype:?} with {q:?}");
            }
        }
    }
}

#[test]
fn run_rejects_mismatch_without_writing() {
    let levels = vec![level(4, 4, 8.0), level(2, 2, 16.0)];
    let rows = 20;
    let dfl_box = DflBoxPlan::new(dfl_box_params(
        DType::I8,
        DType::F32,
        levels.clone(),
        BoxFormat::Xyxy,
    ))
    .unwrap();
    let dfl = DflPlan::new(DflParams {
        input_dtype: DType::I8,
        input: Quant::new(0.1, 0),
        output_dtype: DType::I16,
        output: Quant::new(0.01, 0),
        reg_max: REG_MAX,
    })
    .unwrap();
    let d2b = Dist2BboxPlan::new(Dist2BboxParams {
        input_dtype: DType::I8,
        input: Quant::new(0.1, 0),
        output_dtype: DType::F32,
        output: Quant::IDENTITY,
        levels,
        format: BoxFormat::Xywh,
    })
    .unwrap();
    let bins = vec![1i8; rows * 65];
    let bins_u8 = vec![1u8; rows * 64];
    let ltrb = vec![1i8; rows * 4];
    let mut f_out = vec![7.0f32; rows * 8];
    let mut i_out = vec![0x5555i16; rows * 8];

    type Run<'a> = &'a dyn Fn(TensorRef<'_>, TensorMut<'_>) -> Result<(), KernelError>;
    let box_run: Run<'_> = &|i, o| dfl_box.run(i, o, None);
    let dfl_run: Run<'_> = &|i, o| dfl.run(i, o, None);
    let d2b_run: Run<'_> = &|i, o| d2b.run(i, o, None);
    let bins_v = |shape: &[usize]| {
        TensorRef::I8(View::contiguous(&bins[..shape.iter().product()], shape).unwrap())
    };

    for (name, run, good_in, cols) in [
        ("dfl_box", box_run, bins_v(&[rows, 64]), 64usize),
        ("dfl", dfl_run, bins_v(&[rows, 64]), 64),
        (
            "dist2bbox",
            d2b_run,
            TensorRef::I8(View::contiguous(&ltrb, &[rows, 4]).unwrap()),
            4,
        ),
    ] {
        let want_float = name != "dfl";
        let expect_invalid = |r: Result<(), KernelError>, what: &str| {
            assert!(
                matches!(r, Err(KernelError::InvalidArgument(_))),
                "{name} {what}: {r:?}"
            );
        };
        // Wrong input dtype.
        expect_invalid(
            run(
                TensorRef::U8(View::contiguous(&bins_u8[..rows * cols], &[rows, cols]).unwrap()),
                out_view(want_float, &mut f_out, &mut i_out, &[rows, 4], &[4, 1]),
            ),
            "input dtype",
        );
        // Wrong output dtype.
        let wrong_out = if want_float {
            TensorMut::I16(ViewMut::contiguous(&mut i_out[..rows * 4], &[rows, 4]).unwrap())
        } else {
            TensorMut::F32(ViewMut::contiguous(&mut f_out[..rows * 4], &[rows, 4]).unwrap())
        };
        expect_invalid(run(good_in, wrong_out), "output dtype");
        // Rank 3, and bins per anchor that do not match the plan.
        expect_invalid(
            run(
                bins_v(&[1, rows, cols]),
                out_view(
                    want_float,
                    &mut f_out,
                    &mut i_out,
                    &[1, rows, 4],
                    &[rows as isize * 4, 4, 1],
                ),
            ),
            "rank 3",
        );
        expect_invalid(
            run(
                bins_v(&[rows, cols + 1]),
                out_view(want_float, &mut f_out, &mut i_out, &[rows, 4], &[4, 1]),
            ),
            "bins per anchor",
        );
        // Output with too few anchors, too few or too many values per anchor.
        for shape in [[rows - 1, 4], [rows, 3], [rows, 5]] {
            expect_invalid(
                run(
                    good_in,
                    out_view(
                        want_float,
                        &mut f_out,
                        &mut i_out,
                        &shape,
                        &[shape[1] as isize, 1],
                    ),
                ),
                "output shape",
            );
        }
        // Overlapping output.
        expect_invalid(
            run(
                good_in,
                out_view(want_float, &mut f_out, &mut i_out, &[rows, 4], &[1, 1]),
            ),
            "overlapping output",
        );
        assert!(
            f_out.iter().all(|&v| v == 7.0),
            "{name}: f32 output written"
        );
        assert!(
            i_out.iter().all(|&v| v == 0x5555),
            "{name}: i16 output written"
        );
    }
}

/// An f32 or i16 output view over one of two buffers.
fn out_view<'a>(
    float: bool,
    f: &'a mut [f32],
    i: &'a mut [i16],
    shape: &[usize],
    strides: &[isize],
) -> TensorMut<'a> {
    if float {
        TensorMut::F32(ViewMut::strided(f, shape, strides).unwrap())
    } else {
        TensorMut::I16(ViewMut::strided(i, shape, strides).unwrap())
    }
}

#[test]
fn zero_anchor_views_are_a_no_op() {
    let empty_i8: [i8; 0] = [];
    let mut empty_f32: [f32; 0] = [];
    let plan = DflBoxPlan::new(dfl_box_params(
        DType::I8,
        DType::F32,
        vec![level(5, 0, 8.0)],
        BoxFormat::Xywh,
    ))
    .unwrap();
    plan.run(
        TensorRef::I8(View::contiguous(&empty_i8, &[0, 64]).unwrap()),
        TensorMut::F32(ViewMut::contiguous(&mut empty_f32, &[0, 4]).unwrap()),
        None,
    )
    .unwrap();
    let plan = DflPlan::new(DflParams {
        input_dtype: DType::I8,
        input: Quant::new(0.1, 0),
        output_dtype: DType::F32,
        output: Quant::IDENTITY,
        reg_max: REG_MAX,
    })
    .unwrap();
    plan.run(
        TensorRef::I8(View::contiguous(&empty_i8, &[0, 64]).unwrap()),
        TensorMut::F32(ViewMut::contiguous(&mut empty_f32, &[0, 4]).unwrap()),
        None,
    )
    .unwrap();
    let plan = Dist2BboxPlan::new(Dist2BboxParams {
        input_dtype: DType::I8,
        input: Quant::new(0.1, 0),
        output_dtype: DType::F32,
        output: Quant::IDENTITY,
        levels: vec![level(5, 0, 8.0)],
        format: BoxFormat::Xyxy,
    })
    .unwrap();
    plan.run(
        TensorRef::I8(View::contiguous(&empty_i8, &[0, 4]).unwrap()),
        TensorMut::F32(ViewMut::contiguous(&mut empty_f32, &[0, 4]).unwrap()),
        None,
    )
    .unwrap();
}

#[test]
fn strided_views_match_contiguous() {
    let levels = yolov8n_small();
    let rows = total(&levels);
    for feats in common::tiers() {
        for (input_dtype, output_dtype) in [
            (DType::I8, DType::F32),
            (DType::I16, DType::U8),
            (DType::F16, DType::F16),
        ] {
            let q = logit_quant(input_dtype);
            let data = Data::quantize(&dfl_logits(rows, REG_MAX, 21), input_dtype, q);
            let mut p = dfl_box_params(input_dtype, output_dtype, levels.clone(), BoxFormat::Xyxy);
            p.output = match output_dtype {
                DType::U8 => Quant::new(8.0, 60),
                _ => p.output,
            };
            let plan = DflBoxPlan::with_features(p, feats).unwrap();
            let mut want = Out::new(output_dtype, rows * 4);
            plan.run(
                data.view(&[rows, 64], &[64, 1]),
                want.contiguous(rows),
                None,
            )
            .unwrap();
            let want = want.bytes();
            let elem = want.len() / (rows * 4);

            // Channel-major input, and padded anchor-major rows.
            let t = data.transposed(rows, 64);
            let padded = match &data {
                Data::I8(v) => Data::I8(pad_rows(v, rows, 64, 70)),
                Data::I16(v) => Data::I16(pad_rows(v, rows, 64, 70)),
                Data::F16(v) => Data::F16(pad_rows(v, rows, 64, 70)),
                _ => unreachable!(),
            };
            for (input, what) in [
                (t.view(&[rows, 64], &[1, rows as isize]), "channel-major"),
                (padded.view(&[rows, 64], &[70, 1]), "padded rows"),
            ] {
                let mut got = Out::new(output_dtype, rows * 4);
                plan.run(input, got.contiguous(rows), None).unwrap();
                assert_eq!(
                    got.bytes(),
                    want,
                    "{what} {input_dtype:?}, tier {:?}",
                    plan.tier()
                );
            }

            // Transposed output: `[4, rows]` in memory.
            let mut got = Out::new(output_dtype, rows * 4);
            plan.run(
                t.view(&[rows, 64], &[1, rows as isize]),
                got.view(&[rows, 4], &[1, rows as isize]),
                None,
            )
            .unwrap();
            let got = got.bytes();
            for a in 0..rows {
                for c in 0..4 {
                    assert_eq!(
                        &got[(c * rows + a) * elem..(c * rows + a + 1) * elem],
                        &want[(a * 4 + c) * elem..(a * 4 + c + 1) * elem],
                        "transposed output anchor {a} value {c}"
                    );
                }
            }

            // Padded output rows: the padding is left untouched.
            let mut got = Out::new(output_dtype, rows * 6);
            let sentinel = got.bytes();
            plan.run(
                data.view(&[rows, 64], &[64, 1]),
                got.view(&[rows, 4], &[6, 1]),
                None,
            )
            .unwrap();
            let got = got.bytes();
            for a in 0..rows {
                assert_eq!(
                    &got[a * 6 * elem..(a * 6 + 4) * elem],
                    &want[a * 4 * elem..(a + 1) * 4 * elem]
                );
                assert_eq!(
                    &got[(a * 6 + 4) * elem..(a + 1) * 6 * elem],
                    &sentinel[(a * 6 + 4) * elem..(a + 1) * 6 * elem]
                );
            }
        }
    }
}

fn pad_rows<T: Copy + Default>(v: &[T], rows: usize, cols: usize, pitch: usize) -> Vec<T> {
    let mut out = vec![T::default(); rows * pitch];
    for r in 0..rows {
        out[r * pitch..r * pitch + cols].copy_from_slice(&v[r * cols..(r + 1) * cols]);
    }
    out
}

#[test]
fn pool_and_concurrent_runs_match_single_thread() {
    let levels = yolov8n_640();
    let rows = total(&levels);
    for feats in common::tiers() {
        for (input_dtype, output_dtype, format) in [
            (DType::I8, DType::F32, BoxFormat::Xywh),
            (DType::U16, DType::I16, BoxFormat::Xyxy),
            (DType::F32, DType::F16, BoxFormat::Xyxy),
        ] {
            let q = logit_quant(input_dtype);
            let data = Data::quantize(&dfl_logits(rows, REG_MAX, 31), input_dtype, q);
            let t = data.transposed(rows, 64);
            let plan = DflBoxPlan::with_features(
                dfl_box_params(input_dtype, output_dtype, levels.clone(), format),
                feats,
            )
            .unwrap();
            common::assert_pool_and_concurrency_match(|pool: Option<&dyn Pool>| {
                let mut out = Out::new(output_dtype, rows * 4);
                plan.run(
                    t.view(&[rows, 64], &[1, rows as isize]),
                    out.contiguous(rows),
                    pool,
                )
                .unwrap();
                out.bytes()
            });
            let plan = DflPlan::with_features(
                DflParams {
                    input_dtype,
                    input: q,
                    output_dtype,
                    output: ltrb_out_quant(output_dtype),
                    reg_max: REG_MAX,
                },
                feats,
            )
            .unwrap();
            common::assert_pool_and_concurrency_match(|pool: Option<&dyn Pool>| {
                let mut out = Out::new(output_dtype, rows * 4);
                plan.run(data.view(&[rows, 64], &[64, 1]), out.contiguous(rows), pool)
                    .unwrap();
                out.bytes()
            });
            let dq = ltrb_quant(input_dtype);
            let dist = Data::quantize(&ltrb_distances(rows, 32), input_dtype, dq);
            let plan = Dist2BboxPlan::with_features(
                Dist2BboxParams {
                    input_dtype,
                    input: dq,
                    output_dtype,
                    output: box_quant(output_dtype),
                    levels: levels.clone(),
                    format,
                },
                feats,
            )
            .unwrap();
            common::assert_pool_and_concurrency_match(|pool: Option<&dyn Pool>| {
                let mut out = Out::new(output_dtype, rows * 4);
                plan.run(dist.view(&[rows, 4], &[4, 1]), out.contiguous(rows), pool)
                    .unwrap();
                out.bytes()
            });
        }
    }
}

/// One decode op with its inputs, run once per output dtype on one tier.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Dfl { reg_max: usize },
    Dist2Bbox,
    DflBox { reg_max: usize },
}

/// The inputs and f64 reference of one op, input dtype and format over the 640×640 levels.
struct OpCase {
    op: Op,
    input_dtype: DType,
    format: BoxFormat,
    levels: Vec<Level>,
    data: Data,
    q: Quant,
    /// The f64 result per output value.
    want: Vec<f64>,
}

impl OpCase {
    fn new(op: Op, input_dtype: DType, format: BoxFormat, seed: u64) -> Self {
        let q = match op {
            Op::Dist2Bbox => ltrb_quant(input_dtype),
            _ => logit_quant(input_dtype),
        };
        Self::with_quant(op, input_dtype, q, format, seed)
    }

    fn with_quant(op: Op, input_dtype: DType, q: Quant, format: BoxFormat, seed: u64) -> Self {
        let levels = yolov8n_640();
        let rows = total(&levels);
        let geom = anchor_geometry(&levels);
        let (data, want) = match op {
            Op::Dfl { reg_max } | Op::DflBox { reg_max } => {
                let data = Data::quantize(&dfl_logits(rows, reg_max, seed), input_dtype, q);
                let ltrb = ref_ltrb(&data.real(q), reg_max);
                let want = match op {
                    Op::Dfl { .. } => ltrb.into_iter().flatten().collect(),
                    _ => ref_boxes(&ltrb, &geom, format),
                };
                (data, want)
            }
            Op::Dist2Bbox => {
                let data = Data::quantize(&ltrb_distances(rows, seed), input_dtype, q);
                let ltrb: Vec<[f64; 4]> = data
                    .real(q)
                    .chunks_exact(4)
                    .map(|c| [c[0], c[1], c[2], c[3]])
                    .collect();
                (data, ref_boxes(&ltrb, &geom, format))
            }
        };
        Self {
            op,
            input_dtype,
            format,
            levels,
            data,
            q,
            want,
        }
    }

    fn cols(&self) -> usize {
        match self.op {
            Op::Dfl { reg_max } | Op::DflBox { reg_max } => 4 * reg_max,
            Op::Dist2Bbox => 4,
        }
    }

    /// The largest value of the output's natural scale: the 640 px image for boxes, the last
    /// bin for DFL distances.
    fn full_scale(&self) -> f64 {
        match self.op {
            Op::Dfl { reg_max } => (reg_max.max(2) - 1) as f64,
            _ => 640.0,
        }
    }

    /// Runs the op on `feats` into `output_dtype`, quantized with `output` when it is an integer.
    fn run(&self, feats: CpuFeatures, output_dtype: DType, output: Quant) -> (Out, Tier) {
        let rows = total(&self.levels);
        let input = Input::new(&self.data, rows, self.cols(), Layout::AnchorMajor);
        let mut out = Out::new(output_dtype, rows * 4);
        let (input_dtype, input_q) = (self.input_dtype, self.q);
        let tier = match self.op {
            Op::Dfl { reg_max } => {
                let plan = DflPlan::with_features(
                    DflParams {
                        input_dtype,
                        input: input_q,
                        output_dtype,
                        output,
                        reg_max,
                    },
                    feats,
                )
                .unwrap();
                plan.run(input.view(), out.contiguous(rows), None).unwrap();
                plan.tier()
            }
            Op::Dist2Bbox => {
                let plan = Dist2BboxPlan::with_features(
                    Dist2BboxParams {
                        input_dtype,
                        input: input_q,
                        output_dtype,
                        output,
                        levels: self.levels.clone(),
                        format: self.format,
                    },
                    feats,
                )
                .unwrap();
                plan.run(input.view(), out.contiguous(rows), None).unwrap();
                plan.tier()
            }
            Op::DflBox { reg_max } => {
                let plan = DflBoxPlan::with_features(
                    DflBoxParams {
                        input_dtype,
                        input: input_q,
                        output_dtype,
                        output,
                        reg_max,
                        levels: self.levels.clone(),
                        format: self.format,
                    },
                    feats,
                )
                .unwrap();
                plan.run(input.view(), out.contiguous(rows), None).unwrap();
                plan.tier()
            }
        };
        (out, tier)
    }
}

/// Every op, input dtype and format the f16-output tests cover.
fn f16_output_cases() -> Vec<OpCase> {
    let mut cases = Vec::new();
    for (i, &input_dtype) in IN_DTYPES.iter().enumerate() {
        let seed = 100 + i as u64;
        for reg_max in [16, 8, 7, 1, 17, 32, 64] {
            cases.push(OpCase::new(
                Op::Dfl { reg_max },
                input_dtype,
                BoxFormat::Xywh,
                seed,
            ));
            for format in FORMATS {
                cases.push(OpCase::new(
                    Op::DflBox { reg_max },
                    input_dtype,
                    format,
                    seed,
                ));
            }
        }
        for format in FORMATS {
            cases.push(OpCase::new(Op::Dist2Bbox, input_dtype, format, seed));
        }
    }
    cases
}

/// Integer inputs at seeded random quantizations: log-uniform scales over the useful range of
/// each type and zero points anywhere in it, for every op and format.
fn random_quant_cases() -> Vec<OpCase> {
    let mut rng = StdRng::seed_from_u64(0xdf1_f016);
    let mut cases = Vec::new();
    for input_dtype in [DType::I8, DType::U8, DType::I16, DType::U16] {
        let (bits, lo, hi) = match input_dtype {
            DType::I8 => ((-6.0, -1.0), -128, 127),
            DType::U8 => ((-6.0, -1.0), 0, 255),
            DType::I16 => ((-14.0, -6.0), -32768, 32767),
            _ => ((-14.0, -6.0), 0, 65535),
        };
        for op in [
            Op::Dfl { reg_max: 16 },
            Op::Dfl { reg_max: 8 },
            Op::DflBox { reg_max: 16 },
            Op::Dist2Bbox,
        ] {
            for k in 0..RANDOM_QUANTS {
                let scale = 2f32.powf(rng.random_range(bits.0..bits.1));
                let zero_point = rng.random_range(lo..=hi);
                let q = Quant::new(scale, zero_point);
                let format = FORMATS[k % 2];
                cases.push(OpCase::with_quant(op, input_dtype, q, format, rng.random()));
            }
        }
    }
    cases
}

/// Random quantizations per op and integer input type in the FP16 bound test.
const RANDOM_QUANTS: usize = 24;

/// An `f16` output that runs no FP16 kernel is the same tier's `f32` output rounded once,
/// nearest even, bit for bit: every case on the scalar tier and the NEON tiers without FP16
/// arithmetic, and `DflPlan`'s 8-bit table path on the FP16 tier.
#[test]
fn decode_f16_output_is_narrowed_f32_path() {
    let cases = f16_output_cases();
    for feats in common::tiers() {
        for case in &cases {
            let label = format!(
                "{:?} {:?} {:?} features {feats:?}",
                case.op, case.input_dtype, case.format
            );
            let (Out::F32(f32s), tier32) = case.run(feats, DType::F32, Quant::IDENTITY) else {
                unreachable!()
            };
            let (Out::F16(f16s), tier16) = case.run(feats, DType::F16, Quant::IDENTITY) else {
                unreachable!()
            };
            if tier16 == Tier::NeonFp16 {
                assert!(feats.neon_fp16, "{label}");
                continue;
            }
            assert_eq!(tier16, tier32, "{label}");
            assert_eq!(f16s, narrow(&f32s), "{label}");
        }
    }
}

/// The FP16 kernels' bound for `f16` outputs, in f16 ulps of the value or of the output's
/// full scale (640 px for boxes, `reg_max − 1` grid units for distances), whichever is larger:
/// 1 up to `reg_max` 32; at 64, where each lane sums four f16 steps, 2 for distances and 3 for
/// boxes, whose distances reach 63 grid units.
fn fp16_ulp_bound(op: Op) -> f64 {
    match op {
        Op::Dfl { reg_max } if reg_max > 32 => 2.0,
        Op::DflBox { reg_max } if reg_max > 32 => 3.0,
        _ => 1.0,
    }
}

/// On the FP16 tier, `f16` outputs run the FP16 kernels (`DflPlan`'s 8-bit table path and
/// `Dist2BboxPlan` excepted, whose f32 results are rounded once) and stay within
/// [`fp16_ulp_bound`] of the float64 result rounded to f16, at fixed and seeded random
/// quantizations.
#[test]
fn decode_f16_output_fp16_within_bound() {
    let Ok(feats) = CpuFeatures::from_forced_tier(Some("neon_fp16")) else {
        eprintln!("WARNING: SKIPPED: decode_f16_output_fp16_within_bound: no neon_fp16 tier");
        return;
    };
    for case in f16_output_cases().into_iter().chain(random_quant_cases()) {
        let label = format!(
            "{:?} {:?} {:?} {:?}",
            case.op, case.input_dtype, case.q, case.format
        );
        let (out, tier) = case.run(feats, DType::F16, Quant::IDENTITY);
        let fp16 = !matches!(
            (case.op, case.input_dtype),
            (Op::Dfl { .. }, DType::I8 | DType::U8) | (Op::Dist2Bbox, _)
        );
        let want_tier = if fp16 { Tier::NeonFp16 } else { Tier::Neon };
        assert_eq!(tier, want_tier, "{label}");
        let Out::F16(got) = out else { unreachable!() };
        let (floor, bound) = (ulp_f16(case.full_scale()), fp16_ulp_bound(case.op));
        for (i, (g, &w)) in got.iter().zip(&case.want).enumerate() {
            let want = f16::from_f64(w);
            let err = (g.to_f64() - want.to_f64()).abs() / ulp_f16(w).max(floor);
            assert!(
                err <= bound,
                "{label}: value {i}: got {g} want {want} ({w}): {err:.2} ulp > {bound}"
            );
        }
    }
}

/// Every NEON path against the scalar path on the same inputs.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;

    fn forced(name: &str) -> Option<CpuFeatures> {
        match CpuFeatures::from_forced_tier(Some(name)) {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!("WARNING: SKIPPED: tier {name}: {e}");
                None
            }
        }
    }

    fn run_box(feats: CpuFeatures, p: DflBoxParams, data: &Data, rows: usize) -> (Out, Tier) {
        let (output_dtype, cols) = (p.output_dtype, 4 * p.reg_max);
        let plan = DflBoxPlan::with_features(p, feats).unwrap();
        let mut out = Out::new(output_dtype, rows * 4);
        let strides = [cols as isize, 1];
        plan.run(
            data.view(&[rows, cols], &strides),
            out.contiguous(rows),
            None,
        )
        .unwrap();
        (out, plan.tier())
    }

    #[test]
    fn neon_lut_paths_bit_identical_to_scalar() {
        let scalar = CpuFeatures::from_forced_tier(Some("scalar")).unwrap();
        let levels = yolov8n_640();
        let rows = total(&levels);
        for name in ["neon", "neon_fp16", "neon_dotprod"] {
            let Some(feats) = forced(name) else { continue };
            for (input_dtype, reg_max) in [DType::I8, DType::U8]
                .into_iter()
                .flat_map(|d| [(d, REG_MAX), (d, 8), (d, 7)])
            {
                let q = logit_quant(input_dtype);
                let data = Data::quantize(&dfl_logits(rows, reg_max, 51), input_dtype, q);
                for output_dtype in [DType::F32, DType::F16, DType::I16] {
                    // The FP16 tier's f16 outputs run the FP16 kernels, checked against their bound.
                    if feats.neon_fp16 && output_dtype == DType::F16 {
                        continue;
                    }
                    for format in FORMATS {
                        let mut p =
                            dfl_box_params(input_dtype, output_dtype, levels.clone(), format);
                        p.reg_max = reg_max;
                        let (want, _) = run_box(scalar, p.clone(), &data, rows);
                        let (got, tier) = run_box(feats, p, &data, rows);
                        assert_eq!(tier, Tier::Neon);
                        assert_eq!(
                            got.bytes(),
                            want.bytes(),
                            "{name} {input_dtype:?} {output_dtype:?} reg_max {reg_max}"
                        );
                    }
                }
                let dfl = |f: CpuFeatures| {
                    let plan = DflPlan::with_features(
                        DflParams {
                            input_dtype,
                            input: q,
                            output_dtype: DType::F32,
                            output: Quant::IDENTITY,
                            reg_max,
                        },
                        f,
                    )
                    .unwrap();
                    let mut out = Out::new(DType::F32, rows * 4);
                    let cols = 4 * reg_max;
                    plan.run(
                        data.view(&[rows, cols], &[cols as isize, 1]),
                        out.contiguous(rows),
                        None,
                    )
                    .unwrap();
                    (out.bytes(), plan.tier())
                };
                let (want, _) = dfl(scalar);
                let (got, tier) = dfl(feats);
                assert_eq!(tier, Tier::Neon);
                assert_eq!(got, want, "dfl {name} {input_dtype:?} reg_max {reg_max}");
            }
        }
    }

    #[test]
    fn neon_softmax_paths_within_envelope_of_scalar() {
        let scalar = CpuFeatures::from_forced_tier(Some("scalar")).unwrap();
        let levels = yolov8n_640();
        let rows = total(&levels);
        let scale: Vec<f64> = anchor_geometry(&levels).iter().map(|g| g.2).collect();
        for (name, tier) in [("neon", Tier::Neon), ("neon_fp16", Tier::NeonFp16)] {
            let Some(feats) = forced(name) else { continue };
            for input_dtype in [DType::I16, DType::U16] {
                let q = logit_quant(input_dtype);
                let data = Data::quantize(&dfl_logits(rows, REG_MAX, 53), input_dtype, q);
                let p = dfl_box_params(input_dtype, DType::F32, levels.clone(), BoxFormat::Xywh);
                let (want, _) = run_box(scalar, p.clone(), &data, rows);
                let (got, got_tier) = run_box(feats, p, &data, rows);
                assert_eq!(got_tier, tier);
                let bar = Bar::for_output(DType::F32, tier, REG_MAX);
                check(
                    &format!("{name} {input_dtype:?}"),
                    &got,
                    &want.values(),
                    Quant::IDENTITY,
                    bar,
                    &scale,
                );
                // Quantized outputs run the f32 NEON kernel on every NEON tier.
                let p = dfl_box_params(input_dtype, DType::I16, levels.clone(), BoxFormat::Xyxy);
                let (_, q_tier) = run_box(feats, p, &data, rows);
                assert_eq!(q_tier, Tier::Neon);
            }
        }
    }

    #[test]
    fn neon_ltrb_path_within_envelope_of_scalar() {
        let scalar = CpuFeatures::from_forced_tier(Some("scalar")).unwrap();
        let Some(feats) = forced("neon") else { return };
        let levels = t1297_small();
        let rows = total(&levels);
        let scale: Vec<f64> = anchor_geometry(&levels).iter().map(|g| g.2).collect();
        for input_dtype in [DType::I8, DType::U8, DType::I16, DType::U16] {
            let q = ltrb_quant(input_dtype);
            let data = Data::quantize(&ltrb_distances(rows, 61), input_dtype, q);
            for format in FORMATS {
                let run = |f: CpuFeatures| {
                    let plan = Dist2BboxPlan::with_features(
                        Dist2BboxParams {
                            input_dtype,
                            input: q,
                            output_dtype: DType::F32,
                            output: Quant::IDENTITY,
                            levels: levels.clone(),
                            format,
                        },
                        f,
                    )
                    .unwrap();
                    let mut out = Out::new(DType::F32, rows * 4);
                    plan.run(data.view(&[rows, 4], &[4, 1]), out.contiguous(rows), None)
                        .unwrap();
                    (out, plan.tier())
                };
                let (want, _) = run(scalar);
                let (got, tier) = run(feats);
                assert_eq!(tier, Tier::Neon);
                check(
                    &format!("ltrb neon {input_dtype:?} {format:?}"),
                    &got,
                    &want.values(),
                    Quant::IDENTITY,
                    Bar::F32,
                    &scale,
                );
            }
        }
    }

    /// Float inputs to `f32` and `i8` run the f32 NEON kernels on every NEON tier, within the
    /// integer inputs' bound of the scalar tier: the f32 bar for `f32`, ±1 LSB for `i8`.
    #[test]
    fn neon_float_input_paths_within_envelope_of_scalar() {
        let scalar = CpuFeatures::from_forced_tier(Some("scalar")).unwrap();
        for name in ["neon", "neon_fp16", "neon_dotprod"] {
            let Some(feats) = forced(name) else { continue };
            for input_dtype in [DType::F16, DType::F32] {
                for op in [
                    Op::Dfl { reg_max: 16 },
                    Op::Dfl { reg_max: 7 },
                    Op::Dist2Bbox,
                ]
                .into_iter()
                .chain([16, 8].map(|reg_max| Op::DflBox { reg_max }))
                {
                    for format in FORMATS {
                        let case = OpCase::new(op, input_dtype, format, 71);
                        let label = format!("{name} {op:?} {input_dtype:?} {format:?}");
                        let (want, _) = case.run(scalar, DType::F32, Quant::IDENTITY);
                        let (got, tier) = case.run(feats, DType::F32, Quant::IDENTITY);
                        assert_eq!(tier, Tier::Neon, "{label}");
                        let scale: Vec<f64> = match op {
                            Op::Dfl { .. } => vec![1.0; case.want.len() / 4],
                            _ => anchor_geometry(&case.levels).iter().map(|g| g.2).collect(),
                        };
                        check(
                            &label,
                            &got,
                            &want.values(),
                            Quant::IDENTITY,
                            Bar::F32,
                            &scale,
                        );
                        let q = match op {
                            Op::Dfl { .. } => ltrb_out_quant(DType::I8),
                            _ => box_quant(DType::I8),
                        };
                        let (want, _) = case.run(scalar, DType::I8, q);
                        let (got, tier) = case.run(feats, DType::I8, q);
                        let (want, got) = (want.values(), got.values());
                        assert_eq!(tier, Tier::Neon, "{label} i8");
                        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                            assert!((g - w).abs() <= 1.0, "{label} i8 value {i}: {g} vs {w}");
                        }
                    }
                }
            }
        }
    }

    /// Every input/output pair runs a NEON kernel under NEON features: on the FP16 tier,
    /// `NeonFp16` for `f16` outputs (but `DflPlan`'s table path and `Dist2BboxPlan`) and the
    /// decoder's fp16-exp DFL for 16-bit inputs to `f32`; `Neon` otherwise.
    #[test]
    fn every_decode_path_is_neon_under_neon_features() {
        let all = [
            DType::I8,
            DType::U8,
            DType::I16,
            DType::U16,
            DType::F16,
            DType::F32,
        ];
        let probed = CpuFeatures::probe();
        let fp16 = CpuFeatures::from_forced_tier(Some("neon_fp16")).ok();
        let tiers = ["neon", "neon_fp16", "neon_dotprod"]
            .into_iter()
            .filter_map(|n| CpuFeatures::from_forced_tier(Some(n)).ok())
            .chain([probed]);
        for feats in tiers {
            let has_fp16 = feats.neon_fp16 && fp16.is_some();
            for i in all {
                for o in all {
                    let box_want = match (has_fp16, i, o) {
                        (true, _, DType::F16) => Tier::NeonFp16,
                        (true, DType::I16 | DType::U16, DType::F32) => Tier::NeonFp16,
                        _ => Tier::Neon,
                    };
                    // `DflPlan`'s table path has no box math to run in f16.
                    let dfl_want = match i {
                        DType::I8 | DType::U8 => Tier::Neon,
                        _ => box_want,
                    };
                    let d2b_want = Tier::Neon;
                    let q = |d: DType| match d {
                        DType::U8 => Quant::new(0.1, 3),
                        DType::U16 => Quant::new(0.1, 300),
                        _ => Quant::new(0.1, 0),
                    };
                    for reg_max in [1, 7, 8, 16, 17, 64] {
                        let p = DflParams {
                            input_dtype: i,
                            input: q(i),
                            output_dtype: o,
                            output: q(o),
                            reg_max,
                        };
                        let what = format!("{i:?}->{o:?} reg_max {reg_max} {feats:?}");
                        let dfl = DflPlan::with_features(p, feats).unwrap();
                        assert_eq!(dfl.tier(), dfl_want, "dfl {what}");
                        let b = DflBoxParams {
                            input_dtype: i,
                            input: q(i),
                            output_dtype: o,
                            output: q(o),
                            reg_max,
                            levels: yolov8n_small(),
                            format: BoxFormat::Xyxy,
                        };
                        let dfl_box = DflBoxPlan::with_features(b, feats).unwrap();
                        assert_eq!(dfl_box.tier(), box_want, "dfl_box {what}");
                    }
                    let p = Dist2BboxParams {
                        input_dtype: i,
                        input: q(i),
                        output_dtype: o,
                        output: q(o),
                        levels: yolov8n_small(),
                        format: BoxFormat::Xywh,
                    };
                    let d2b = Dist2BboxPlan::with_features(p, feats).unwrap();
                    assert_eq!(d2b.tier(), d2b_want, "dist2bbox {i:?}->{o:?} {feats:?}");
                }
            }
        }
    }
}
