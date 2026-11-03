// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Every score, mask-coefficient and proto cell on every tier this CPU has.
//!
//! Dequantize-only cells are bit-identical to the scalar tier. Sigmoid cells are held to a bound
//! against the float64 sigmoid of the exact dequantized value: f32 outputs
//! relative, f16 outputs in f16 ulps of that value rounded to f16.

use super::dispatch::{
    DstSliceMut, InputView, MaskCoefDispatch, ProtoDispatch, ScoreLevelDispatch,
};
use super::{decoder_cpu_features_from, CpuFeatures};
use crate::per_scale::plan::{Layout, LevelPlan};
use crate::per_scale::{Activation, DecodeDtype};
use crate::Quantization;
use edgefirst_kernels::prim::grids::make_anchor_grid;
use edgefirst_tensor::DType;
use half::f16;

/// f32 sigmoid outputs: error bound relative to `sigmoid(x) · (1 + |x|)` per tier, plus
/// `f32::MIN_POSITIVE` absolute.
const F32_REL_SCALAR: f64 = 4.0 * f32::EPSILON as f64;
const F32_REL_NEON: f64 = 4.0 * f32::EPSILON as f64;
/// f16 sigmoid outputs: ulps of the reference rounded to f16, per tier.
/// Scalar cells round `x` to f16 first, costing up to `|x| < 9.7` ulps, plus 1 for the roundings.
const F16_ULPS_SCALAR: i32 = 11;
const F16_ULPS_NEON: i32 = 1;
const F16_ULPS_NEON_FP16: i32 = 2;

/// Every tier this CPU can run, by its `EDGEFIRST_DECODER_FORCE_KERNEL` name.
fn tiers() -> Vec<(&'static str, CpuFeatures)> {
    ["scalar", "neon", "neon_fp16", "neon_dotprod"]
        .into_iter()
        .filter_map(|t| decoder_cpu_features_from(Some(t)).ok().map(|f| (t, f)))
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Score(Activation),
    MaskCoefs,
    Protos,
}

const KINDS: [Kind; 4] = [
    Kind::Score(Activation::Sigmoid),
    Kind::Score(Activation::None),
    Kind::MaskCoefs,
    Kind::Protos,
];
const INPUTS: [DType; 6] = [
    DType::I8,
    DType::U8,
    DType::I16,
    DType::U16,
    DType::F16,
    DType::F32,
];
const OUTPUTS: [DecodeDtype; 2] = [DecodeDtype::F32, DecodeDtype::F16];

#[derive(Debug, Clone, Copy)]
enum Cell {
    Score(ScoreLevelDispatch, Activation),
    MaskCoefs(MaskCoefDispatch),
    Protos(ProtoDispatch),
}

impl Cell {
    fn select(kind: Kind, input: DType, output: DecodeDtype, f: &CpuFeatures) -> Self {
        match kind {
            Kind::Score(a) => Cell::Score(ScoreLevelDispatch::select(input, output, f).unwrap(), a),
            Kind::MaskCoefs => Cell::MaskCoefs(MaskCoefDispatch::select(input, output, f).unwrap()),
            Kind::Protos => Cell::Protos(ProtoDispatch::select(input, output, f).unwrap()),
        }
    }

    fn name(&self) -> String {
        match self {
            Cell::Score(d, _) => format!("{d:?}"),
            Cell::MaskCoefs(d) => format!("{:?}", d.0),
            Cell::Protos(d) => format!("{:?}", d.0),
        }
    }

    fn run(&self, input: InputView<'_>, q: Quantization, dst: DstSliceMut<'_>) {
        let n = match &dst {
            DstSliceMut::F32(d) => d.len(),
            DstSliceMut::F16(d) => d.len(),
        };
        let level = level(n);
        match *self {
            Cell::Score(d, a) => d.run(input, q, 1, &level, a, dst),
            Cell::MaskCoefs(d) => d.run(input, q, 1, &level, dst),
            Cell::Protos(d) => d.run(input, q, dst),
        }
        .unwrap();
    }
}

fn level(n: usize) -> LevelPlan {
    let (grid_x, grid_y) = make_anchor_grid(1, n);
    LevelPlan {
        stride: 8.0,
        h: 1,
        w: n,
        reg_max: 16,
        anchor_offset: 0,
        grid_x,
        grid_y,
        box_shape: vec![1, 1, n, 64].into_boxed_slice(),
        score_shape: vec![1, 1, n, 1].into_boxed_slice(),
        mc_shape: Some(vec![1, 1, n, 1].into_boxed_slice()),
        layout: Layout::Nhwc,
    }
}

/// The scalar tier's variant names end in `Scalar`; NEON tiers select none of them, and
/// f16 scores take the native FP16 cell exactly on tiers with FEAT_FP16.
#[test]
fn every_cell_runs_neon_on_neon_tiers() {
    for (tier, f) in tiers() {
        for kind in KINDS {
            for input in INPUTS {
                for output in OUTPUTS {
                    let name = Cell::select(kind, input, output, &f).name();
                    let label = format!("{tier} {kind:?} {input:?}->{output:?}: {name}");
                    if !f.neon_baseline {
                        assert!(name.ends_with("Scalar"), "{label}");
                        continue;
                    }
                    assert!(!name.ends_with("Scalar"), "{label}");
                    let fp16_score =
                        f.neon_fp16 && output == DecodeDtype::F16 && matches!(kind, Kind::Score(_));
                    assert_eq!(name.ends_with("NeonFp16"), fp16_score, "{label}");
                }
            }
        }
    }
}

/// The forced tier is the one the cells run: `scalar` selects scalar cells even where NEON
/// exists, and `neon` never selects an FP16 cell.
#[test]
fn forced_tier_governs_cell_selection() {
    let scalar = decoder_cpu_features_from(Some("scalar")).unwrap();
    let cell = Cell::select(
        Kind::Score(Activation::Sigmoid),
        DType::I8,
        DecodeDtype::F16,
        &scalar,
    );
    assert!(cell.name().ends_with("Scalar"), "{}", cell.name());
    if let Ok(neon) = decoder_cpu_features_from(Some("neon")) {
        for input in INPUTS {
            let kind = Kind::Score(Activation::Sigmoid);
            let name = Cell::select(kind, input, DecodeDtype::F16, &neon).name();
            assert!(name.ends_with("Neon"), "{input:?}: {name}");
        }
    }
}

/// Raw inputs of one dtype: every value of the 8- and 16-bit types and of f16, a sweep of
/// f32 with its specials, each followed by seven more so the slice ends off a vector.
enum Raw {
    I8(Vec<i8>),
    U8(Vec<u8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
    F16(Vec<f16>),
    F32(Vec<f32>),
}

fn with_tail<T: Copy>(mut v: Vec<T>) -> Vec<T> {
    let tail: Vec<T> = v.iter().take(7).copied().collect();
    v.extend(tail);
    v
}

impl Raw {
    fn of(dtype: DType) -> Self {
        match dtype {
            DType::I8 => Raw::I8(with_tail((i8::MIN..=i8::MAX).collect())),
            DType::U8 => Raw::U8(with_tail((0..=u8::MAX).collect())),
            DType::I16 => Raw::I16(with_tail((i16::MIN..=i16::MAX).collect())),
            DType::U16 => Raw::U16(with_tail((0..=u16::MAX).collect())),
            DType::F16 => Raw::F16(with_tail((0..=u16::MAX).map(f16::from_bits).collect())),
            _ => {
                let mut v: Vec<f32> = (0..200_001)
                    .map(|i| (i as f32 - 100_000.0) * 5e-4)
                    .collect();
                v.extend([
                    f32::NAN,
                    -f32::NAN,
                    f32::INFINITY,
                    f32::NEG_INFINITY,
                    -0.0,
                    1e30,
                    -1e30,
                    -88.0,
                    -87.0,
                    f32::MIN_POSITIVE,
                    f32::from_bits(1),
                ]);
                Raw::F32(with_tail(v))
            }
        }
    }

    fn len(&self) -> usize {
        match self {
            Raw::I8(v) => v.len(),
            Raw::U8(v) => v.len(),
            Raw::I16(v) => v.len(),
            Raw::U16(v) => v.len(),
            Raw::F16(v) => v.len(),
            Raw::F32(v) => v.len(),
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

    fn is_float(&self) -> bool {
        matches!(self, Raw::F16(_) | Raw::F32(_))
    }

    /// Element `i` as f64.
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

    /// The value the sigmoid is taken of, in f64: exact for integers, and for floats the f32
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
        DType::I8 => vec![q(0.1, 0), q(0.2, -20), q(0.17, 108), q(0.07, -12)],
        DType::U8 => vec![q(0.1, 128), q(0.0039, 0), q(0.2, 110)],
        DType::I16 => vec![q(0.0005, 0), q(0.001, 100), q(1e-5, 0)],
        DType::U16 => vec![q(0.0005, 32768), q(1.0 / 65535.0, 0)],
        _ => vec![Quantization::identity(), q(0.5, 3)],
    }
}

/// `f16` distance in ulps between two non-negative results.
fn ulps(a: f16, b: f16) -> i32 {
    (i32::from(a.to_bits()) - i32::from(b.to_bits())).abs()
}

fn sigmoid64(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The bound a tier's sigmoid cell is held to.
fn f16_bound(f: &CpuFeatures) -> i32 {
    match (f.neon_baseline, f.neon_fp16) {
        (false, _) => F16_ULPS_SCALAR,
        (true, false) => F16_ULPS_NEON,
        (true, true) => F16_ULPS_NEON_FP16,
    }
}

/// Every cell on every tier against the scalar tier and the float64 reference. Prints the
/// worst error of each sigmoid cell.
#[test]
fn every_cell_matches_scalar_or_bound_on_every_tier() {
    let scalar = CpuFeatures::default();
    for input in INPUTS {
        let raw = Raw::of(input);
        let n = raw.len();
        for q in quants(input) {
            for kind in KINDS {
                for output in OUTPUTS {
                    let want = Cell::select(kind, input, output, &scalar);
                    let mut w32 = vec![0f32; n];
                    let mut w16 = vec![f16::ZERO; n];
                    match output {
                        DecodeDtype::F32 => want.run(raw.view(), q, DstSliceMut::F32(&mut w32)),
                        DecodeDtype::F16 => want.run(raw.view(), q, DstSliceMut::F16(&mut w16)),
                    }
                    for (tier, f) in tiers() {
                        let cell = Cell::select(kind, input, output, &f);
                        let label = format!("{tier} {} {q:?}", cell.name());
                        let mut g32 = vec![0f32; n];
                        let mut g16 = vec![f16::ZERO; n];
                        match output {
                            DecodeDtype::F32 => cell.run(raw.view(), q, DstSliceMut::F32(&mut g32)),
                            DecodeDtype::F16 => cell.run(raw.view(), q, DstSliceMut::F16(&mut g16)),
                        }
                        let sigmoid = kind == Kind::Score(Activation::Sigmoid);
                        match (sigmoid, output) {
                            (false, DecodeDtype::F32) => {
                                assert_bits_eq(&g32, &w32, f32::to_bits, &label);
                            }
                            (false, DecodeDtype::F16) => {
                                assert_bits_eq(&g16, &w16, f16::to_bits, &label);
                            }
                            (true, DecodeDtype::F32) => {
                                check_sigmoid_f32(&raw, q, &g32, &f, &label)
                            }
                            (true, DecodeDtype::F16) => {
                                check_sigmoid_f16(&raw, q, &g16, &f, &label)
                            }
                        }
                    }
                }
            }
        }
    }
}

fn assert_bits_eq<T: Copy + std::fmt::Debug, B: PartialEq + std::fmt::Debug>(
    got: &[T],
    want: &[T],
    bits: impl Fn(T) -> B,
    label: &str,
) {
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert_eq!(bits(g), bits(w), "{label} at {i}: {g:?} vs {w:?}");
    }
}

fn check_sigmoid_f32(raw: &Raw, q: Quantization, got: &[f32], f: &CpuFeatures, label: &str) {
    let bound = if f.neon_baseline {
        F32_REL_NEON
    } else {
        F32_REL_SCALAR
    };
    let tiny = f64::from(f32::MIN_POSITIVE);
    let mut worst = 0f64;
    for (i, &g) in got.iter().enumerate() {
        let x = raw.x(i, q);
        if x.is_nan() {
            assert!(g.is_nan(), "{label} at {i}: NaN gave {g}");
            continue;
        }
        let r = sigmoid64(x);
        if x.is_infinite() {
            assert_eq!(f64::from(g), r, "{label} at {i}: x {x}");
            continue;
        }
        let err = (f64::from(g) - r).abs();
        // The sigmoid's condition number is about |x|, so an f32-rounded x costs |x| eps.
        let scaled = r * (1.0 + x.abs());
        if r > tiny {
            worst = worst.max(err / scaled);
        }
        assert!(
            err <= bound * scaled + tiny,
            "{label} at {i}: x {x}: got {g}, want {r} (rel {:.3e})",
            err / r
        );
    }
    eprintln!(
        "{label}: worst error {:.2} f32 eps x (1 + |x|) relative",
        worst / f64::from(f32::EPSILON)
    );
}

fn check_sigmoid_f16(raw: &Raw, q: Quantization, got: &[f16], f: &CpuFeatures, label: &str) {
    let mut worst = 0;
    for (i, &g) in got.iter().enumerate() {
        let x = raw.x(i, q);
        if x.is_nan() {
            assert!(g.is_nan(), "{label} at {i}: NaN gave {g}");
            continue;
        }
        let r = f16::from_f64(sigmoid64(x));
        let e = ulps(g, r);
        worst = worst.max(e);
        let bound = f16_bound(f);
        assert!(
            e <= bound,
            "{label} at {i}: x {x}: got {g}, want {r} ({e} ulp)"
        );
    }
    eprintln!("{label}: worst {worst} f16 ulp");
}
