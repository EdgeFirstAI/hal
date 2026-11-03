// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Candidate filters and the generic dequantize on every tier this CPU has, against the scalar
//! tier bit for bit, and the kernel each tier runs.

use half::f16;
use ndarray::{Array2, ArrayView2, ShapeBuilder};
use num_traits::{AsPrimitive, Float, PrimInt};

use super::*;
use crate::byte::{argmax_candidates_quant, multilabel_candidates_quant};
use crate::float::{argmax_candidates, multilabel_candidates};
use crate::per_scale::kernels::{decoder_cpu_features_from, forced_for_test};
use crate::{dequantize_cpu_on, DetectBox, DetectBoxQuantized, Quantization, XYWH};

/// The tier a NEON-capable `isa` runs for a type with a baseline kernel and, when `fp16`, a
/// native FP16 one.
fn expected(isa: Isa, has_kernel: bool, fp16: bool) -> Ran {
    if isa.is_scalar() || !has_kernel {
        Ran::Scalar
    } else if isa.fp16() && fp16 {
        Ran::NeonFp16
    } else {
        Ran::Neon
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Which kernel each tier runs, per score type; a NEON tier must not run scalar for the six
/// types with kernels.
#[test]
fn every_neon_tier_runs_a_kernel_for_each_score_type() {
    fn check<T: Copy + 'static>(isa: Isa, v: T, has_kernel: bool, fp16: bool, what: &str) {
        let cols: Vec<Vec<T>> = vec![vec![v; 37]; 3];
        let views: Vec<&[T]> = cols.iter().map(Vec::as_slice).collect();
        let want = expected(isa, has_kernel, fp16);
        assert_eq!(
            column_argmax_kernel(isa, &views, v, |_, _| {}),
            want,
            "column {what}"
        );
        assert_eq!(
            ge_indices_kernel(isa, &cols[0], v, |_| {}),
            want,
            "ge {what}"
        );
        let row = row_argmax_kernel(isa, &cols[0]).map(|(i, t)| (t, i));
        let want_row = (want != Ran::Scalar).then_some((want, 36));
        assert_eq!(row, want_row, "row {what}");
    }
    for (name, isa) in tiers() {
        let what = |t: &str| format!("{t} on {name}");
        check(isa, 1i8, true, false, &what("i8"));
        check(isa, 1u8, true, false, &what("u8"));
        check(isa, 1i16, true, false, &what("i16"));
        check(isa, 1u16, true, false, &what("u16"));
        check(isa, 1.0f32, true, false, &what("f32"));
        check(isa, f16::ONE, true, true, &what("f16"));
        check(isa, 1i32, false, false, &what("i32"));
        check(isa, 1.0f64, false, false, &what("f64"));
        assert_eq!(
            isa.is_scalar(),
            name == "scalar",
            "{name} resolved to {isa:?}"
        );
    }
}

/// The dequantize kernel each tier runs, per type pair.
#[test]
fn every_neon_tier_runs_a_dequantize_kernel() {
    fn check<T: Copy + Default + 'static, F: Float + 'static>(isa: Isa, has: bool, fp16: bool) {
        let src = vec![T::default(); 33];
        let mut dst = vec![F::zero(); 33];
        // SAFETY: `scale_bias` writes only initialized values.
        let ran = scale_bias(isa, &src, F::one(), Some(F::one()), unsafe {
            as_uninit(&mut dst)
        });
        let name = std::any::type_name::<(T, F)>();
        assert_eq!(ran, expected(isa, has, fp16), "{name} on {isa:?}");
    }
    for (_, isa) in tiers() {
        check::<i8, f32>(isa, true, false);
        check::<u8, f32>(isa, true, false);
        check::<i16, f32>(isa, true, false);
        check::<u16, f32>(isa, true, false);
        check::<i8, f16>(isa, true, true);
        check::<u8, f16>(isa, true, true);
        check::<i16, f16>(isa, true, true);
        check::<u16, f16>(isa, true, true);
        check::<i32, f32>(isa, false, false);
        check::<i8, f64>(isa, false, false);
    }
}

/// A float score drawn so that ties, zeros of both signs, NaN and infinities all occur.
fn float_score(rng: &mut Rng) -> f32 {
    match rng.next() % 50 {
        0 => f32::NAN,
        1 => -0.0,
        2 => 0.0,
        3 => f32::INFINITY,
        4 => f32::NEG_INFINITY,
        5..=20 => [0.25f32, 0.5, 0.75][(rng.next() % 3) as usize],
        _ => rng.unit(),
    }
}

/// `[C + 4, N]` of `T` (boxes in the first 4 rows, then scores), as a decoder output.
fn output<T: Copy>(
    rng: &mut Rng,
    n: usize,
    nc: usize,
    score: impl Fn(&mut Rng) -> T,
    coord: impl Fn(f32) -> T,
) -> Array2<T> {
    Array2::from_shape_fn((4 + nc, n), |(r, _)| {
        if r < 4 {
            coord(rng.unit())
        } else {
            score(rng)
        }
    })
}

/// The three score layouts of `out` (`[C + 4, N]`) and `rows` (`[N, C + 4]`): the transposed
/// `[N, C]` view (contiguous columns), standard rows, and rows with a column step of 2.
type Views<'a, T> = Vec<(&'static str, ArrayView2<'a, T>, ArrayView2<'a, T>)>;

fn layouts<'a, T>(out: &'a Array2<T>, rows: &'a Array2<T>) -> Views<'a, T> {
    vec![
        (
            "columns",
            out.slice(ndarray::s![..4, ..]).reversed_axes(),
            out.slice(ndarray::s![4.., ..]).reversed_axes(),
        ),
        (
            "rows",
            rows.slice(ndarray::s![.., ..4]),
            rows.slice(ndarray::s![.., 4..]),
        ),
        (
            "strided",
            rows.slice(ndarray::s![.., ..4]),
            rows.slice(ndarray::s![.., 4..;2]),
        ),
    ]
}

/// Comparable bits of a float candidate.
fn float_key(b: &(DetectBox, usize)) -> (usize, u32, [u32; 4], usize) {
    let r = &b.0.bbox;
    (
        b.0.label,
        b.0.score.to_bits(),
        [r.xmin, r.ymin, r.xmax, r.ymax].map(f32::to_bits),
        b.1,
    )
}

fn quant_key<S: PrimInt + AsPrimitive<f32>>(
    b: &(DetectBoxQuantized<S>, usize),
) -> (usize, i64, [u32; 4], usize) {
    let r = &b.0.bbox;
    (
        b.0.label,
        b.0.score.to_i64().unwrap_or(i64::MIN),
        [r.xmin, r.ymin, r.xmax, r.ymax].map(f32::to_bits),
        b.1,
    )
}

const SHAPES: [(usize, usize); 9] = [
    (1, 1),
    (7, 3),
    (16, 2),
    (33, 80),
    (100, 1),
    (129, 17),
    (300, 91),
    (513, 80),
    (8400, 80),
];

fn check_float<T>(convert: impl Fn(f32) -> T + Copy, thresholds: &[f32])
where
    T: Float + AsPrimitive<f32> + Send + Sync + 'static,
{
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let mut checked = 0usize;
    for (n, nc) in SHAPES {
        let out = output(
            &mut rng,
            n,
            nc,
            |r| convert(float_score(r)),
            |v| convert(v * 640.0),
        );
        let rows = out.t().as_standard_layout().into_owned();
        for (layout, boxes, scores) in layouts(&out, &rows) {
            for &thr in &thresholds[..thresholds.len().min(if n * nc > 1 << 16 { 2 } else { 8 })] {
                let thr = convert(thr);
                let reference = argmax_candidates::<XYWH, _, _>(Isa::SCALAR, thr, boxes, scores);
                let reference_ml =
                    multilabel_candidates::<XYWH, _, _>(Isa::SCALAR, thr, boxes, scores);
                for (name, isa) in tiers() {
                    let got = argmax_candidates::<XYWH, _, _>(isa, thr, boxes, scores);
                    assert_eq!(
                        got.iter().map(float_key).collect::<Vec<_>>(),
                        reference.iter().map(float_key).collect::<Vec<_>>(),
                        "argmax {name} {layout} n={n} nc={nc}"
                    );
                    let got = multilabel_candidates::<XYWH, _, _>(isa, thr, boxes, scores);
                    assert_eq!(
                        got.iter().map(float_key).collect::<Vec<_>>(),
                        reference_ml.iter().map(float_key).collect::<Vec<_>>(),
                        "multilabel {name} {layout} n={n} nc={nc}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked >= SHAPES.len() * 3 * 2);
}

/// f32, f16 and f64 candidates (argmax and multi-label) equal the scalar tier on every tier,
/// with ties, NaN, signed zeros, infinities, tails, and thresholds nothing or everything passes.
#[test]
fn float_candidates_match_scalar_on_every_tier() {
    let thresholds = [0.5, 0.0, 2.0, f32::NEG_INFINITY, f32::NAN, -0.0];
    check_float(|v| v, &thresholds);
    check_float(f16::from_f32, &thresholds);
    check_float(f64::from, &thresholds);
}

fn check_quant<S>(thresholds: &[S])
where
    S: PrimInt + AsPrimitive<f32> + Send + Sync + 'static,
    u64: AsPrimitive<S>,
{
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let q = Quantization::new(0.05, 3);
    let mut checked = 0usize;
    for (n, nc) in SHAPES {
        let out = output(
            &mut rng,
            n,
            nc,
            |r| {
                if r.next() % 3 == 0 {
                    (r.next() % 4).as_()
                } else {
                    r.next().as_()
                }
            },
            |v| S::from(v * 100.0).unwrap_or_else(S::zero),
        );
        let rows = out.t().as_standard_layout().into_owned();
        for (layout, boxes, scores) in layouts(&out, &rows) {
            for &thr in &thresholds[..thresholds.len().min(if n * nc > 1 << 16 { 2 } else { 8 })] {
                let reference =
                    argmax_candidates_quant::<XYWH, _, _>(Isa::SCALAR, thr, boxes, scores, q);
                let reference_ml =
                    multilabel_candidates_quant::<XYWH, _, _>(Isa::SCALAR, thr, boxes, scores, q);
                for (name, isa) in tiers() {
                    let got = argmax_candidates_quant::<XYWH, _, _>(isa, thr, boxes, scores, q);
                    assert_eq!(
                        got.iter().map(quant_key).collect::<Vec<_>>(),
                        reference.iter().map(quant_key).collect::<Vec<_>>(),
                        "argmax {name} {layout} n={n} nc={nc}"
                    );
                    let got = multilabel_candidates_quant::<XYWH, _, _>(isa, thr, boxes, scores, q);
                    assert_eq!(
                        got.iter().map(quant_key).collect::<Vec<_>>(),
                        reference_ml.iter().map(quant_key).collect::<Vec<_>>(),
                        "multilabel {name} {layout} n={n} nc={nc}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked >= SHAPES.len() * 3 * 2);
}

/// i8, u8, i16, u16 and i32 candidates equal the scalar tier on every tier, including ties
/// (scores drawn from a few values a third of the time) and thresholds nothing or everything
/// passes.
#[test]
fn quant_candidates_match_scalar_on_every_tier() {
    check_quant::<i8>(&[0, 3, i8::MIN, i8::MAX]);
    check_quant::<u8>(&[0, 3, 200, u8::MAX]);
    check_quant::<i16>(&[0, 3, i16::MIN, i16::MAX]);
    check_quant::<u16>(&[0, 3, 40000, u16::MAX]);
    check_quant::<i32>(&[0, 3, i32::MIN]);
}

/// `dequantize_cpu`, `dequantize_cpu_chunked` and `dequantize_ndarray` equal the generic
/// formula on every tier, into f32 and f16, including `map`'s layout for transposed views.
#[test]
fn dequantize_matches_the_generic_formula_on_every_tier() {
    fn check<T, F>(values: &[T])
    where
        T: AsPrimitive<F> + std::fmt::Debug,
        F: Float + 'static + std::fmt::Debug,
        f32: AsPrimitive<F>,
        i32: AsPrimitive<F>,
    {
        for q in [
            Quantization::new(0.004_081_187_3, -123),
            Quantization::new(0.1, 0),
            Quantization::new(-0.5, 0),
            Quantization::new(3.5e-5, 31000),
        ] {
            let zp: F = q.zero_point.as_();
            let scale: F = q.scale.as_();
            let want: Vec<F> = values
                .iter()
                .map(|&d| {
                    if zp != F::zero() {
                        d.as_() * scale + -zp * scale
                    } else {
                        d.as_() * scale
                    }
                })
                .collect();
            for (name, isa) in tiers() {
                let mut got = vec![F::nan(); values.len()];
                dequantize_cpu_on(isa, values, q, &mut got);
                let same = got
                    .iter()
                    .zip(&want)
                    .all(|(a, b)| a == b || (a.is_nan() && b.is_nan()))
                    && got
                        .iter()
                        .zip(&want)
                        .all(|(a, b)| a.is_sign_negative() == b.is_sign_negative());
                assert!(same, "{} on {name}", std::any::type_name::<(T, F)>());
            }
        }
    }
    let i8s: Vec<i8> = (i8::MIN..=i8::MAX).collect();
    let u8s: Vec<u8> = (0..=u8::MAX).collect();
    let i16s: Vec<i16> = (i16::MIN..=i16::MAX).collect();
    let u16s: Vec<u16> = (0..=u16::MAX).collect();
    let i32s: Vec<i32> = (-70000..70000).step_by(7).collect();
    check::<_, f32>(&i8s);
    check::<_, f32>(&u8s);
    check::<_, f32>(&i16s);
    check::<_, f32>(&u16s);
    check::<_, f32>(&i32s);
    check::<_, f16>(&i8s);
    check::<_, f16>(&u8s);
    check::<_, f16>(&i16s);
    check::<_, f16>(&u16s);
    check::<_, f64>(&i16s);

    let q = Quantization::new(0.05, -7);
    let a = Array2::from_shape_vec(
        (37, 29).f(),
        (0..37 * 29).map(|v| (v % 251) as i16 - 120).collect(),
    )
    .unwrap();
    let (scale, bias) = (q.scale, Some(-(q.zero_point as f32) * q.scale));
    for (name, isa) in tiers() {
        for (layout, view) in [
            ("f", a.view()),
            ("t", a.t()),
            ("rev", a.slice(ndarray::s![..;-1, ..])),
        ] {
            let reference = view.mapv(|d| d as f32 * scale + -(q.zero_point as f32) * q.scale);
            match scale_bias_array(isa, &view, scale, bias) {
                Some(got) => {
                    assert_eq!(got, reference, "{layout} on {name}");
                    assert_eq!(got.strides(), reference.strides(), "{layout} on {name}");
                }
                None => assert!(isa.is_scalar() || layout == "rev", "{layout} on {name}"),
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

/// A zero tensor of `$t` with `$shape`.
macro_rules! zeros {
    ($t:ty, $shape:expr) => {{
        let shape: &[usize] = $shape;
        let n: usize = shape.iter().product();
        edgefirst_tensor::TensorDyn::from(
            edgefirst_tensor::Tensor::<$t>::from_slice(&vec![<$t>::default(); n], shape).unwrap(),
        )
    }};
}

fn assert_forced_failure(
    what: &str,
    decoder: &crate::Decoder,
    inputs: &[&edgefirst_tensor::TensorDyn],
) {
    let (mut boxes, mut masks) = (Vec::new(), Vec::new());
    decoder
        .decode(inputs, &mut boxes, &mut masks)
        .unwrap_or_else(|e| panic!("{what}: unforced decode failed: {e:?}"));
    for tier in unavailable_tiers() {
        let r = forced_for_test::with(tier, || decoder.decode(inputs, &mut boxes, &mut masks));
        assert!(
            matches!(r, Err(crate::DecoderError::ForcedKernelUnavailable { .. })),
            "{what} forced {tier}: {r:?}"
        );
    }
}

/// `decode` of a legacy YOLO detection model, quantized and float, fails with
/// `ForcedKernelUnavailable` under a tier the CPU lacks or does not know.
#[test]
fn yolo_decode_fails_on_an_unavailable_forced_tier() {
    use crate::{configs, ConfigOutput, DecoderBuilder};
    for (quantization, input) in [
        (
            Some(configs::QuantTuple(0.004, -123)),
            zeros!(i8, &[1, 84, 64]),
        ),
        (None, zeros!(f32, &[1, 84, 64])),
    ] {
        let decoder = DecoderBuilder::default()
            .add_output(ConfigOutput::Detection(configs::Detection {
                decoder: configs::DecoderType::Ultralytics,
                quantization,
                shape: vec![1, 84, 64],
                dshape: vec![],
                anchors: None,
                normalized: Some(true),
            }))
            .build()
            .unwrap();
        assert_forced_failure("yolo det", &decoder, &[&input]);
    }
}

/// `decode` of a quantized ModelPack detection model fails with `ForcedKernelUnavailable` under a
/// tier the CPU lacks or does not know.
#[test]
fn modelpack_decode_fails_on_an_unavailable_forced_tier() {
    use crate::{configs, DecoderBuilder};
    let decoder = DecoderBuilder::default()
        .with_config_modelpack_det(
            configs::Boxes {
                decoder: configs::DecoderType::ModelPack,
                quantization: Some((0.0047, 21).into()),
                shape: vec![1, 20, 1, 4],
                dshape: vec![
                    (configs::DimName::Batch, 1),
                    (configs::DimName::NumBoxes, 20),
                    (configs::DimName::Padding, 1),
                    (configs::DimName::BoxCoords, 4),
                ],
                normalized: Some(true),
            },
            configs::Scores {
                decoder: configs::DecoderType::ModelPack,
                quantization: Some((0.002, 0).into()),
                shape: vec![1, 20, 1],
                dshape: vec![
                    (configs::DimName::Batch, 1),
                    (configs::DimName::NumBoxes, 20),
                    (configs::DimName::NumClasses, 1),
                ],
            },
        )
        .build()
        .unwrap();
    let (b, s) = (zeros!(u8, &[1, 20, 1, 4]), zeros!(u8, &[1, 20, 1]));
    assert_forced_failure("modelpack det", &decoder, &[&b, &s]);
}

/// `decode` through a schema-v2 merge program (ARA-2 split boxes) fails with
/// `ForcedKernelUnavailable` under a tier the CPU lacks or does not know.
#[test]
fn merge_program_decode_fails_on_an_unavailable_forced_tier() {
    use crate::{schema::SchemaV2, DecoderBuilder};
    let json = edgefirst_bench::testdata::read_to_string("ara2_int8_edgefirst.json");
    let decoder = DecoderBuilder::new()
        .with_schema(SchemaV2::parse_json(&json).unwrap())
        .build()
        .unwrap();
    let inputs = [
        zeros!(i8, &[1, 2, 8400, 1]),
        zeros!(i8, &[1, 2, 8400, 1]),
        zeros!(u8, &[1, 80, 8400, 1]),
        zeros!(i8, &[1, 32, 8400, 1]),
        zeros!(i8, &[1, 32, 160, 160]),
    ];
    let refs: Vec<_> = inputs.iter().collect();
    assert_forced_failure("merge program", &decoder, &refs);
}

/// The public infallible helpers run the scalar code under such a tier.
#[test]
fn public_helpers_fall_back_to_scalar_on_an_unavailable_forced_tier() {
    let src: Vec<i8> = (-128..=127).collect();
    let q = Quantization::new(0.1, 3);
    let mut want = vec![0.0f32; src.len()];
    crate::dequantize_cpu(&src, q, &mut want);
    let scores = Array2::from_shape_fn((40, 3), |(i, c)| ((i * 7 + c) % 11) as f32 / 10.0);
    let boxes = Array2::from_elem((40, 4), 0.5f32);
    let kept =
        crate::float::postprocess_boxes_float::<XYWH, _, _>(0.5, boxes.view(), scores.view());
    for tier in unavailable_tiers() {
        forced_for_test::with(tier, || {
            let mut got = vec![0.0f32; src.len()];
            crate::dequantize_cpu(&src, q, &mut got);
            assert_eq!(got, want, "{tier}");
            let arr = ndarray::Array1::from(src.clone());
            let nd: ndarray::Array1<f32> = crate::dequantize_ndarray(arr.view(), q);
            assert_eq!(nd.to_vec(), want, "{tier}");
            let again = crate::float::postprocess_boxes_float::<XYWH, _, _>(
                0.5,
                boxes.view(),
                scores.view(),
            );
            assert_eq!(again, kept, "{tier}");
        });
    }
}

/// The parallel column argmax gives the same candidates on one thread and on four, and equals
/// the scalar tier, for a full 8400 × 80 score view.
#[test]
fn column_argmax_is_independent_of_the_thread_count() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let out = output(&mut rng, 8400, 80, float_score, |v| v * 640.0);
    let scores = out.slice(ndarray::s![4.., ..]).reversed_axes();
    let boxes = out.slice(ndarray::s![..4, ..]).reversed_axes();
    let cols = score_columns(&scores).unwrap();
    let reference = argmax_candidates::<XYWH, _, _>(Isa::SCALAR, 0.5, boxes, scores);
    let reference: Vec<(usize, usize)> = reference.iter().map(|(b, i)| (*i, b.label)).collect();
    for (name, isa) in tiers().into_iter().filter(|(_, isa)| !isa.is_scalar()) {
        let mut runs = Vec::new();
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            runs.push(pool.install(|| column_argmax(isa, &cols, 0.5f32)).unwrap());
        }
        assert_eq!(runs[0], runs[1], "{name}");
        assert_eq!(runs[0], reference, "{name}");
    }
}
