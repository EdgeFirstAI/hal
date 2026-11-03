// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Semantic decodes on every tier this CPU has against the scalar tier (bit for bit, the sigmoid
//! confidence within its bound), the kernel each tier runs, thread-count independence and the
//! forced-tier rule.

use edgefirst_tensor::{
    DType, Element, Quantization, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};
use half::f16;

use super::super::tiers;
use super::*;
use crate::per_scale::kernels::{decoder_cpu_features_from, forced_for_test};
use crate::schema::{Activation, DecodeMode, Upsample, UpsampleMethod};
use crate::semantic::{decode_semantic, SemanticConfig, SemanticOptions};
use crate::{DecoderError, DecoderResult};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn pick<T: Copy>(&mut self, from: &[T]) -> T {
        from[self.below(from.len())]
    }
}

/// The tier each kind of kernel runs: `Scalar` on the scalar tier, `NeonFp16` for FP16 kernels
/// on a tier with FEAT_FP16, `Neon` otherwise.
fn expected(isa: Isa, fp16: bool) -> Ran {
    if isa.is_scalar() {
        Ran::Scalar
    } else if fp16 && isa.fp16() {
        Ran::NeonFp16
    } else {
        Ran::Neon
    }
}

/// Which kernel each tier runs, per entry point and type; a NEON tier must not run scalar.
#[test]
fn every_neon_tier_runs_the_semantic_kernels() {
    fn scores<T: Copy + 'static>(isa: Isa, v: T, fp16: bool, what: &str) {
        let src = vec![v; 3 * 37];
        let mut classes = vec![0u16; 37];
        let mut conf = vec![0f32; 37];
        let top = vec![0f32; 37];
        let want = expected(isa, false);
        let planar = ScoreRow::Planar { cs: 37 };
        let ran = argmax_row(isa, &src, planar, 3, &mut classes);
        assert_eq!(ran, expected(isa, fp16), "argmax planar {what}");
        let ran = argmax_row(isa, &src, ScoreRow::Interleaved, 3, &mut classes);
        assert_eq!(ran, expected(isa, fp16), "argmax interleaved {what}");
        let ran = threshold_row(isa, &src[..37], 0.5, &mut classes);
        assert_eq!(ran, want, "threshold {what}");
        let ran = softmax_row(isa, &src, planar, 3, &top, 1.0, &mut conf);
        assert_eq!(ran, want, "softmax planar {what}");
        let ran = softmax_row(isa, &src, ScoreRow::Interleaved, 3, &top, 1.0, &mut conf);
        assert_eq!(ran, want, "softmax interleaved {what}");
        for logit in [false, true] {
            let ran = binary_row(isa, &src[..37], 0.0, 1.0, logit, &mut conf);
            assert_eq!(ran, want, "binary {logit} {what}");
        }
    }
    fn indices<T: Copy + 'static>(isa: Isa, v: T, what: &str) {
        let mut dst = [0u16; 19];
        let ran = class_index_row(isa, &[v; 19], 2, &mut dst);
        assert_eq!(ran, expected(isa, false), "class index {what}");
    }
    fn per_channel<T: Copy + Default + 'static>(isa: Isa, what: &str) {
        let data = vec![T::default(); 3 * 4 * 5];
        let mut out = vec![0f32; data.len()];
        let (s, z) = ([0.5f32; 3], [1i32; 3]);
        let planar = Planes {
            classes: 3,
            height: 4,
            width: 5,
            cs: 20,
            ys: 5,
            xs: 1,
        };
        let interleaved = Planes {
            cs: 1,
            ys: 15,
            xs: 3,
            ..planar
        };
        for grid in [planar, interleaved] {
            let ran = dequantize_per_channel(isa, &data, grid, &s, &z, &mut out);
            assert_eq!(ran, !isa.is_scalar(), "per-channel {what} {grid:?}");
        }
    }
    for (name, isa) in tiers() {
        let what = |t: &str| format!("{t} on {name}");
        scores(isa, 1u8, false, &what("u8"));
        scores(isa, 1i8, false, &what("i8"));
        scores(isa, 1u16, false, &what("u16"));
        scores(isa, 1i16, false, &what("i16"));
        scores(isa, 1.0f32, false, &what("f32"));
        scores(isa, f16::ONE, true, &what("f16"));
        let ran = argmax_row(isa, &[1.0f64; 6], ScoreRow::Interleaved, 3, &mut [0u16; 2]);
        assert_eq!(ran, Ran::Scalar, "f64 on {name}");
        indices(isa, 1u8, &what("u8"));
        indices(isa, 1u16, &what("u16"));
        indices(isa, 1i32, &what("i32"));
        indices(isa, 1u32, &what("u32"));
        indices(isa, 1i64, &what("i64"));
        per_channel::<i8>(isa, &what("i8"));
        per_channel::<u8>(isa, &what("u8"));
        per_channel::<i16>(isa, &what("i16"));
        per_channel::<u16>(isa, &what("u16"));
        let want = expected(isa, false);
        let mut out = [0f32; 21];
        assert_eq!(
            sigmoid_row(isa, &[0.5; 21], &mut out),
            want,
            "sigmoid on {name}"
        );
        let ran = cell_argmax(isa, &[1, 2], &[[0.0; 4]; 2], 0.5, &[0.5; 6], &mut [0; 6]);
        assert_eq!(ran, want, "cell on {name}");
        let corners: [&[f32]; 4] = [&[0.0; 5], &[1.0; 5], &[2.0; 5], &[3.0; 5]];
        let ran = cell_candidates(isa, corners, &mut Vec::new(), &mut Vec::new());
        assert_eq!(ran, want, "candidates on {name}");
        for (v, layout) in [(1i8, (12, 1, 3)), (1i8, (1, 3, 3))] {
            let mut out = [0f32; 6];
            let ran = gather_columns(isa, &[v; 36], layout, &[0, 3], &mut out);
            assert_eq!(ran, want, "gather {layout:?} on {name}");
        }
        let ran = gather_columns(isa, &[1.0f32; 36], (12, 1, 3), &[0, 3], &mut [0f32; 6]);
        assert_eq!(ran, want, "gather f32 on {name}");
        let neon = !isa.is_scalar();
        let (ys, xs) = ([0u32, 0, 1, 1], [0u32, 1, 2, 2, 3, 4, 4]);
        let map16 = [1u16; 10];
        let map32 = [0.5f32; 10];
        let ran = nearest_classes(isa, &map16, 5, &ys, &xs, &mut [0u8; 28]);
        assert_eq!(ran, neon, "nearest u8 on {name}");
        let ran = nearest_classes(isa, &map16, 5, &ys, &xs, &mut [0u16; 28]);
        assert_eq!(ran, neon, "nearest u16 on {name}");
        let ran = nearest_confidence(isa, &map32, 5, &ys, &xs, &mut [0f32; 28]);
        assert_eq!(ran, neon, "nearest conf f32 on {name}");
        let ran = nearest_confidence(isa, &map32, 5, &ys, &xs, &mut [0u8; 28]);
        assert_eq!(ran, neon, "nearest conf u8 on {name}");
        let frac = [0.25f32; 7];
        let taps = |p: &'static [u32], f| Taps {
            lo: p,
            hi: p,
            frac: f,
        };
        let ran = bilinear_confidence(
            isa,
            &map32,
            5,
            taps(&[0, 1], &frac[..2]),
            taps(&[0, 1, 2], &frac[..3]),
            &mut [0f32; 6],
        );
        assert_eq!(ran, neon, "bilinear conf f32 on {name}");
        let ran = bilinear_confidence(
            isa,
            &map32,
            5,
            taps(&[0, 1], &frac[..2]),
            taps(&[0, 1, 2], &frac[..3]),
            &mut [0u8; 6],
        );
        assert_eq!(ran, neon, "bilinear conf u8 on {name}");
        let fill = |_: usize, row: &mut [f32]| row.fill(1.0);
        let ran = bilinear_threshold(
            isa,
            5,
            fill,
            taps(&[0, 1], &frac[..2]),
            taps(&[0, 1, 2], &frac[..3]),
            0.5,
            &mut [0u8; 6],
        );
        assert_eq!(ran, neon, "bilinear threshold u8 on {name}");
        let ran = bilinear_threshold(
            isa,
            5,
            fill,
            taps(&[0, 1], &frac[..2]),
            taps(&[0, 1, 2], &frac[..3]),
            0.5,
            &mut [0u16; 6],
        );
        assert_eq!(ran, neon, "bilinear threshold u16 on {name}");
        assert_eq!(
            isa.is_scalar(),
            name == "scalar",
            "{name} resolved to {isa:?}"
        );
    }
}

fn tensor<T: Element + Copy>(shape: &[usize], values: &[T]) -> TensorDyn
where
    TensorDyn: From<Tensor<T>>,
{
    let t = Tensor::<T>::new(shape, Some(TensorMemory::Mem), None).unwrap();
    t.map().unwrap().as_mut_slice().copy_from_slice(values);
    t.into()
}

fn read<T: Element + Copy>(t: &TensorDyn) -> Vec<T> {
    t.as_typed::<T>()
        .unwrap()
        .map()
        .unwrap()
        .as_slice()
        .to_vec()
}

/// A decode's class map and, when requested, confidence (as f32).
type Decoded = (Vec<u16>, Option<Vec<f32>>);

fn decode(
    input: &TensorDyn,
    config: &SemanticConfig,
    (h, w, dst): (usize, usize, DType),
    conf: Option<DType>,
    options: &SemanticOptions,
) -> DecoderResult<Decoded> {
    let mut map = TensorDyn::new(&[h, w], dst, Some(TensorMemory::Mem), None).unwrap();
    let mut confidence =
        conf.map(|d| TensorDyn::new(&[h, w], d, Some(TensorMemory::Mem), None).unwrap());
    decode_semantic(input, config, &mut map, confidence.as_mut(), options)?;
    let classes = match dst {
        DType::U8 => read::<u8>(&map).into_iter().map(u16::from).collect(),
        _ => read::<u16>(&map),
    };
    let conf = confidence.map(|c| match c.dtype() {
        DType::U8 => read::<u8>(&c).into_iter().map(f32::from).collect(),
        _ => read::<f32>(&c),
    });
    Ok((classes, conf))
}

/// Axis orders of a `[1, ·, ·, ·]` score map: `(class, height, width)` axes.
const LAYOUTS: [(&str, (usize, usize, usize)); 4] = [
    ("nchw", (1, 2, 3)),
    ("nhwc", (3, 1, 2)),
    ("nhcw", (2, 1, 3)),
    ("nwhc", (3, 2, 1)),
];

/// Stored scores in `dtype` for a map of `n` values, with ties, specials and extremes.
fn scores(rng: &mut Rng, dtype: DType, n: usize, ties: bool) -> TensorDynBuilder {
    let unit: Vec<f32> = (0..n)
        .map(|_| {
            if ties {
                rng.below(3) as f32 / 2.0
            } else {
                rng.unit()
            }
        })
        .collect();
    let special = |rng: &mut Rng| rng.below(29) == 0;
    match dtype {
        DType::U8 => TensorDynBuilder::U8(unit.iter().map(|&u| (u * 255.0) as u8).collect()),
        DType::I8 => {
            TensorDynBuilder::I8(unit.iter().map(|&u| (u * 255.0 - 128.0) as i8).collect())
        }
        DType::U16 => TensorDynBuilder::U16(unit.iter().map(|&u| (u * 65535.0) as u16).collect()),
        DType::I16 => TensorDynBuilder::I16(
            unit.iter()
                .map(|&u| (u * 65535.0 - 32768.0) as i16)
                .collect(),
        ),
        DType::F32 | DType::F16 => {
            let v: Vec<f32> = unit
                .iter()
                .map(|&u| match special(rng) {
                    true => rng.pick(&[f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0]),
                    false => u * 16.0 - 8.0,
                })
                .collect();
            if dtype == DType::F32 {
                TensorDynBuilder::F32(v)
            } else {
                TensorDynBuilder::F16(v.into_iter().map(f16::from_f32).collect())
            }
        }
        other => panic!("{other:?}"),
    }
}

/// Typed score values awaiting a shape.
enum TensorDynBuilder {
    U8(Vec<u8>),
    I8(Vec<i8>),
    U16(Vec<u16>),
    I16(Vec<i16>),
    F32(Vec<f32>),
    F16(Vec<f16>),
}

impl TensorDynBuilder {
    fn build(&self, shape: &[usize]) -> TensorDyn {
        match self {
            Self::U8(v) => tensor(shape, v),
            Self::I8(v) => tensor(shape, v),
            Self::U16(v) => tensor(shape, v),
            Self::I16(v) => tensor(shape, v),
            Self::F32(v) => tensor(shape, v),
            Self::F16(v) => tensor(shape, v),
        }
    }
}

/// Whether the confidence of `config` comes from the NEON sigmoid, which may differ from the
/// scalar one within its bound.
fn sigmoid_confidence(classes: usize, activation: Option<Activation>, required: bool) -> bool {
    match (classes, activation) {
        (1, None) => true,
        (1, Some(_)) => false,
        (_, Some(_)) => false,
        (_, None) => required,
    }
}

/// `got` against the scalar tier's `want`: classes equal; confidence equal, or for a sigmoid
/// confidence within `2^-20` (f32) or 1 (u8) where the scalar f32 value `exact` rounds within
/// that bound of a u8 boundary.
fn check(got: &Decoded, want: &Decoded, sigmoid: bool, exact: Option<&[f32]>, ctx: &str) {
    assert_eq!(got.0, want.0, "classes {ctx}");
    let (Some(g), Some(w)) = (&got.1, &want.1) else {
        assert_eq!(got.1.is_some(), want.1.is_some(), "{ctx}");
        return;
    };
    for (i, (&a, &b)) in g.iter().zip(w).enumerate() {
        if a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) {
            continue;
        }
        assert!(sigmoid, "confidence {ctx} at {i}: {a} vs {b}");
        match exact {
            None => assert!(
                (a - b).abs() <= 2f32.powi(-20),
                "confidence {ctx} at {i}: {a} vs {b}"
            ),
            Some(exact) => {
                let t = exact[i].clamp(0.0, 1.0) * 255.0 + 0.5;
                let edge = (t - t.round()).abs() <= 255.0 * 2f32.powi(-20);
                assert!(
                    (a - b).abs() == 1.0 && edge,
                    "u8 confidence {ctx} at {i}: {a} vs {b}, scalar {t}"
                );
            }
        }
    }
}

/// One random decode setup.
struct Case {
    input: TensorDyn,
    config: SemanticConfig,
    out: (usize, usize, DType),
    conf: Option<DType>,
    options: SemanticOptions,
    sigmoid: bool,
    what: String,
}

fn case(rng: &mut Rng, dtype: DType, layout: usize, classes: usize) -> Case {
    let (lname, (ca, ha, wa)) = LAYOUTS[layout];
    let (h, w) = (
        rng.pick(&[1, 2, 3, 6, 9]),
        rng.pick(&[1, 3, 5, 15, 16, 17, 23, 40]),
    );
    let mut shape = [1usize; 4];
    shape[ca] = classes;
    shape[ha] = h;
    shape[wa] = w;
    let n = classes * h * w;
    let ties = rng.below(3) == 0;
    let mut input = scores(rng, dtype, n, ties).build(&shape);
    let mut config = SemanticConfig::scores(&shape, ca, ha, wa).unwrap();
    let integer = !matches!(dtype, DType::F32 | DType::F16);
    let mut quant = "none";
    if integer {
        if classes > 1 && rng.below(3) == 0 {
            let scales: Vec<f32> = (0..classes).map(|_| 0.01 + rng.unit() * 0.1).collect();
            let zps: Vec<i32> = (0..classes).map(|_| rng.below(200) as i32 - 100).collect();
            input
                .set_quantization(Quantization::per_channel(scales, zps, ca).unwrap())
                .unwrap();
            quant = "per-channel";
        } else {
            let zp = rng.below(200) as i32 - 100;
            input
                .set_quantization(Quantization::per_tensor(0.01 + rng.unit() * 0.1, zp))
                .unwrap();
            quant = "per-tensor";
        }
    }
    let method = rng.pick(&[UpsampleMethod::Bilinear, UpsampleMethod::Nearest]);
    let align_corners = rng.below(2) == 0;
    config = config
        .with_upsample(Upsample {
            method,
            align_corners,
        })
        .unwrap();
    let activation = if classes == 1 {
        rng.pick(&[None, Some(Activation::Sigmoid)])
    } else {
        rng.pick(&[None, Some(Activation::Softmax), Some(Activation::Sigmoid)])
    };
    let required = activation.is_none() && classes > 1 && rng.below(2) == 0;
    if let Some(a) = activation {
        config = config.with_activation(a).unwrap();
    } else if required {
        config = config
            .with_activation_required(Activation::Sigmoid)
            .unwrap();
    }
    if classes == 1 && rng.below(2) == 0 {
        config = config.with_threshold(rng.unit() - 0.25);
    }
    let mode = rng.pick(&[DecodeMode::Accurate, DecodeMode::Fast]);
    let mut options = SemanticOptions::new().with_mode(mode);
    if rng.below(2) == 0 {
        options = options.with_content_region([0.0, 0.125, 1.0, 0.875]);
    }
    let out_h = rng.pick(&[1, 4, 7, 19, 33]);
    let out_w = rng.pick(&[1, 5, 16, 23, 61, 150]);
    let dst = if classes > 256 || rng.below(4) == 0 {
        DType::U16
    } else {
        DType::U8
    };
    let conf = rng.pick(&[None, Some(DType::F32), Some(DType::U8)]);
    let what = format!(
        "{dtype:?} {lname} c={classes} {h}x{w}->{out_h}x{out_w} {quant} {method:?} ac={align_corners} \
         {activation:?} req={required} {mode:?} {:?} {dst:?} conf={conf:?}",
        options.content_region
    );
    Case {
        input,
        config,
        out: (out_h, out_w, dst),
        conf,
        options,
        sigmoid: sigmoid_confidence(classes, activation, required),
        what,
    }
}

/// Runs `c` on the scalar tier and every other tier and compares.
fn compare(c: &Case) {
    let run = |conf| decode(&c.input, &c.config, c.out, conf, &c.options);
    let want = forced_for_test::with("scalar", || run(c.conf)).unwrap();
    let exact = (c.sigmoid && c.conf == Some(DType::U8))
        .then(|| {
            forced_for_test::with("scalar", || run(Some(DType::F32)))
                .unwrap()
                .1
        })
        .flatten();
    for (name, _) in tiers() {
        let got = forced_for_test::with(name, || run(c.conf)).unwrap();
        check(
            &got,
            &want,
            c.sigmoid,
            exact.as_deref(),
            &format!("{} on {name}", c.what),
        );
    }
}

/// Every score type, layout and class count, random geometry, modes, quantization and
/// confidence, against the scalar tier.
#[test]
fn semantic_decodes_match_the_scalar_tier() {
    let mut rng = Rng(0x5e3a_2026);
    let dtypes = [
        DType::U8,
        DType::I8,
        DType::U16,
        DType::I16,
        DType::F32,
        DType::F16,
    ];
    for dtype in dtypes {
        for layout in 0..LAYOUTS.len() {
            for classes in [1, 2, 3, 5, 17, 21, 150, 300] {
                for _ in 0..3 {
                    compare(&case(&mut rng, dtype, layout, classes));
                }
            }
        }
    }
}

/// The decode at a realistic size, where rows split across rayon tasks, is the same on one
/// thread and on several.
#[test]
fn semantic_decodes_do_not_depend_on_the_thread_count() {
    let mut rng = Rng(0x7e4d_2026);
    let shapes = [((1, 21, 40, 60), (1, 2, 3)), ((1, 40, 60, 21), (3, 1, 2))];
    for (shape, (ca, ha, wa)) in shapes {
        let shape = [shape.0, shape.1, shape.2, shape.3];
        let n: usize = shape.iter().product();
        let input = scores(&mut rng, DType::F32, n, false).build(&shape);
        for (mode, required) in [(DecodeMode::Accurate, false), (DecodeMode::Fast, true)] {
            let mut config = SemanticConfig::scores(&shape, ca, ha, wa).unwrap();
            if required {
                config = config
                    .with_activation_required(Activation::Sigmoid)
                    .unwrap();
            }
            let options = SemanticOptions::new()
                .with_mode(mode)
                .with_content_region([0.0, 0.1, 1.0, 0.9]);
            for (name, _) in tiers() {
                let run = |threads: usize| {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    pool.install(|| {
                        forced_for_test::with(name, || {
                            decode(
                                &input,
                                &config,
                                (171, 245, DType::U8),
                                Some(DType::U8),
                                &options,
                            )
                        })
                    })
                    .unwrap()
                };
                let one = run(1);
                for threads in [2, 5] {
                    assert_eq!(
                        run(threads),
                        one,
                        "{mode:?} {shape:?} {threads} threads on {name}"
                    );
                }
            }
        }
    }
}

/// Class-map inputs of every type, in range and not, against the scalar tier: maps equal, and
/// the same error for a value the class map cannot hold.
#[test]
fn class_map_inputs_match_the_scalar_tier() {
    let mut rng = Rng(0xc1a5_2026);
    for (h, w) in [(1, 1), (3, 7), (9, 17), (5, 40)] {
        for dst in [DType::U8, DType::U16] {
            for bad in [None, Some(256i64), Some(-1), Some(70_000)] {
                let n = h * w;
                let mut v: Vec<i64> = (0..n).map(|_| rng.below(256) as i64).collect();
                if let Some(b) = bad {
                    v[rng.below(n)] = b;
                }
                let inputs: Vec<(&str, TensorDyn)> = vec![
                    (
                        "u8",
                        tensor(&[1, h, w], &v.iter().map(|&x| x as u8).collect::<Vec<_>>()),
                    ),
                    (
                        "u16",
                        tensor(&[1, h, w], &v.iter().map(|&x| x as u16).collect::<Vec<_>>()),
                    ),
                    (
                        "i32",
                        tensor(&[1, h, w], &v.iter().map(|&x| x as i32).collect::<Vec<_>>()),
                    ),
                    (
                        "u32",
                        tensor(&[1, h, w], &v.iter().map(|&x| x as u32).collect::<Vec<_>>()),
                    ),
                    ("i64", tensor(&[1, h, w], &v)),
                ];
                let config = SemanticConfig::class_map(&[1, h, w], 1, 2).unwrap();
                let options = SemanticOptions::new().with_content_region([0.0, 0.0, 1.0, 0.75]);
                for (what, input) in &inputs {
                    let run = || decode(input, &config, (2 * h + 1, 3 * w, dst), None, &options);
                    let want = forced_for_test::with("scalar", run).map_err(|e| format!("{e:?}"));
                    for (name, _) in tiers() {
                        let got = forced_for_test::with(name, run).map_err(|e| format!("{e:?}"));
                        assert_eq!(got, want, "{what} {h}x{w} {dst:?} bad={bad:?} on {name}");
                    }
                }
            }
        }
    }
}

/// Tier names this CPU cannot run, and one no CPU can.
fn unavailable_tiers() -> Vec<&'static str> {
    ["wibble", "neon", "neon_fp16", "neon_dotprod"]
        .into_iter()
        .filter(|t| decoder_cpu_features_from(Some(t)).is_err())
        .collect()
}

/// A semantic decode, of scores or of a class map, fails with `ForcedKernelUnavailable` under a
/// tier the CPU lacks or does not know.
#[test]
fn semantic_decode_follows_the_forced_tier_rule() {
    let scores = tensor(&[1, 3, 4, 5], &[0.5f32; 60]);
    let score_config = SemanticConfig::scores(&[1, 3, 4, 5], 1, 2, 3).unwrap();
    let map = tensor(&[1, 4, 5], &[1u8; 20]);
    let map_config = SemanticConfig::class_map(&[1, 4, 5], 1, 2).unwrap();
    let options = SemanticOptions::new();
    let runs = [(&scores, &score_config), (&map, &map_config)];
    for (input, config) in runs {
        let run = || decode(input, config, (8, 10, DType::U8), None, &options);
        run().unwrap();
        for tier in unavailable_tiers() {
            let r = forced_for_test::with(tier, run);
            assert!(
                matches!(r, Err(DecoderError::ForcedKernelUnavailable { .. })),
                "forced {tier}: {r:?}"
            );
        }
    }
}
