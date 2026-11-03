// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The ModelPack split decode, segmentation conversions and argmax on every tier this CPU has
//! against the scalar tier bit for bit, the kernel each tier runs, and the forced-tier rule.

use half::f16;
use ndarray::{s, Array3, Array4, ArrayView3, ArrayViewD, Axis, ShapeBuilder};

use super::super::tiers;
use super::*;
use crate::configs::{self, DecoderType, DimName};
use crate::decoder::ArrayViewDQuantized;
use crate::modelpack::{postprocess_modelpack_split_float, ModelPackDetectionConfig};
use crate::per_scale::kernels::{decoder_cpu_features_from, forced_for_test};
use crate::{dequantize_ndarray_on, DecoderBuilder, Quantization, Segmentation};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Raw logits around the sigmoid's useful range, with saturating and special values.
    fn logit(&mut self) -> f32 {
        match self.next() % 97 {
            0 => f32::NAN,
            1 => f32::INFINITY,
            2 => -90.0,
            3 => 80.0,
            4 => -0.0,
            _ => ((self.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 16.0,
        }
    }
}

const ANCHORS: [[f32; 2]; 3] = [[0.1375, 0.2074], [0.2542, 0.2148], [0.2313, 0.3519]];

fn bits(a: &ndarray::Array2<f32>) -> Vec<u32> {
    a.iter().map(|x| x.to_bits()).collect()
}

/// The tier `isa` resolves to for `split_rows_kernel` over one 1×1 row of `T`.
fn split_ran<T: Copy + Default + 'static>(isa: Isa) -> Ran {
    let src = vec![T::default(); 3 * 7];
    let rows = SplitRows {
        width: 1,
        height: 1,
        anchors: &ANCHORS,
        classes: 2,
        first_row: 0,
    };
    let mut b = vec![MaybeUninit::new(0f32); 12];
    let mut c = vec![MaybeUninit::new(0f32); 6];
    split_rows_kernel(isa, &src, None, &rows, &mut b, &mut c)
}

/// Which kernel each tier runs for the split rows, the segmentation conversions and the
/// argmax; a NEON tier must not run scalar where a kernel exists.
#[test]
fn every_neon_tier_runs_the_modelpack_kernels() {
    for (name, isa) in tiers() {
        let neon = if isa.is_scalar() {
            Ran::Scalar
        } else {
            Ran::Neon
        };
        assert_eq!(split_ran::<i8>(isa), neon, "split i8 on {name}");
        assert_eq!(split_ran::<u8>(isa), neon, "split u8 on {name}");
        assert_eq!(split_ran::<i16>(isa), neon, "split i16 on {name}");
        assert_eq!(split_ran::<u16>(isa), neon, "split u16 on {name}");
        assert_eq!(split_ran::<f32>(isa), neon, "split f32 on {name}");
        assert_eq!(split_ran::<f16>(isa), neon, "split f16 on {name}");
        assert_eq!(split_ran::<i32>(isa), Ran::Scalar, "split i32 on {name}");
        assert_eq!(split_ran::<f64>(isa), Ran::Scalar, "split f64 on {name}");

        let neon_some = !isa.is_scalar();
        macro_rules! to_u8 {
            ($t:ty, $kernel:expr) => {
                let a = Array3::<$t>::default((3, 5, 7));
                assert_eq!(
                    seg_to_u8_array(isa, &a.view()).is_some(),
                    neon_some && $kernel,
                    "seg {} on {name}",
                    stringify!($t)
                );
            };
        }
        to_u8!(i8, true);
        to_u8!(u16, true);
        to_u8!(i16, true);
        to_u8!(u32, true);
        to_u8!(i32, true);
        to_u8!(u8, false);
        let planar = Array3::<u8>::default((7, 3, 5)).permuted_axes([1, 2, 0]);
        assert_eq!(
            seg_to_u8_array(isa, &planar.view()).is_some(),
            neon_some,
            "seg planar u8 on {name}"
        );

        let f = Array3::<f32>::default((3, 5, 7));
        let ran = seg_normalize_array(isa, &f.view()).map(|(_, r)| r);
        assert_eq!(
            ran,
            neon_some.then_some(Ran::Neon),
            "normalize f32 on {name}"
        );
        let h = Array3::<f16>::default((3, 5, 7));
        let want = (!isa.is_scalar()).then_some(if isa.fp16() { Ran::NeonFp16 } else { Ran::Neon });
        let ran = seg_normalize_array(isa, &h.view()).map(|(_, r)| r);
        assert_eq!(ran, want, "normalize f16 on {name}");
        let d = Array3::<f64>::default((3, 5, 7));
        assert!(
            seg_normalize_array(isa, &d.view()).is_none(),
            "f64 on {name}"
        );

        let seg = vec![1u8; 5 * 9];
        assert_eq!(
            channel_argmax(isa, &seg, 5).is_some(),
            neon_some,
            "argmax on {name}"
        );
    }
}

/// Split outputs `(H, W, na·(5 + nc))` of random logits as `T`.
fn layers<T>(
    rng: &mut Rng,
    shapes: &[(usize, usize, usize)],
    to: impl Fn(f32) -> T,
) -> Vec<Array3<T>> {
    shapes
        .iter()
        .map(|&(h, w, nc)| Array3::from_shape_fn((h, w, 3 * (nc + 5)), |_| to(rng.logit())))
        .collect()
}

fn configs(n: usize) -> Vec<ModelPackDetectionConfig> {
    (0..n)
        .map(|_| ModelPackDetectionConfig {
            anchors: ANCHORS.to_vec(),
            quantization: None,
        })
        .collect()
}

/// The float split decode at realistic and odd shapes (40×40×3×85 with 20×20×3×85, one
/// class, odd grids) equals the scalar tier bit for bit on every tier, for `f32`, `f16` and
/// `f64` outputs, standard and transposed layouts.
#[test]
fn split_float_matches_the_scalar_tier() {
    let mut rng = Rng(0x5151_5151_5151_5151);
    let shape_sets: [&[(usize, usize, usize)]; 4] = [
        &[(40, 40, 80), (20, 20, 80)],
        &[(9, 15, 1), (17, 30, 1)],
        &[(3, 7, 2)],
        &[(1, 1, 3), (2, 1, 3)],
    ];
    for shapes in shape_sets {
        let f = layers(&mut rng, shapes, |x| x);
        let cfg = configs(f.len());
        let check =
            |what: &str, run: &dyn Fn(Isa) -> (ndarray::Array2<f32>, ndarray::Array2<f32>)| {
                let (wb, ws) = run(Isa::SCALAR);
                for (name, isa) in tiers() {
                    let (b, s) = run(isa);
                    assert_eq!(bits(&b), bits(&wb), "{what} boxes {shapes:?} on {name}");
                    assert_eq!(bits(&s), bits(&ws), "{what} scores {shapes:?} on {name}");
                }
            };
        let views: Vec<_> = f.iter().map(|a| a.view()).collect();
        check("f32", &|isa| {
            postprocess_modelpack_split_float(isa, &views, &cfg)
        });
        let h: Vec<Array3<f16>> = f.iter().map(|a| a.mapv(f16::from_f32)).collect();
        let views: Vec<_> = h.iter().map(|a| a.view()).collect();
        check("f16", &|isa| {
            postprocess_modelpack_split_float(isa, &views, &cfg)
        });
        let d: Vec<Array3<f64>> = f.iter().map(|a| a.mapv(f64::from)).collect();
        let views: Vec<_> = d.iter().map(|a| a.view()).collect();
        check("f64", &|isa| {
            postprocess_modelpack_split_float(isa, &views, &cfg)
        });
        let t: Vec<Array3<f32>> = f
            .iter()
            .map(|a| {
                let mut t = Array3::zeros(a.dim().f());
                t.assign(a);
                t
            })
            .collect();
        let views: Vec<_> = t.iter().map(|a| a.view()).collect();
        check("f32 fortran", &|isa| {
            postprocess_modelpack_split_float(isa, &views, &cfg)
        });
    }
}

/// The fused dequantize of `SplitDecode` equals `dequantize_ndarray` followed by the scalar float
/// decode, bit for bit, for every 8- and 16-bit type, zero and non-zero zero points and no
/// quantization.
#[test]
fn fused_split_dequantize_matches_dequantize_then_float() {
    fn check<T: Copy + Send + Sync + num_traits::AsPrimitive<f32> + 'static>(
        what: &str,
        layers: &[Array3<T>],
    ) {
        let cfg = configs(layers.len());
        for q in [
            Some(Quantization::new(0.0855, 174)),
            Some(Quantization::new(0.0993, -12)),
            Some(Quantization::new(0.02, 0)),
            None,
        ] {
            let dequantized: Vec<Array3<f32>> = layers
                .iter()
                .map(|a| match q {
                    Some(q) => dequantize_ndarray_on::<T, _, f32>(Isa::SCALAR, a.view(), q),
                    None => a.map(|&x| -> f32 { x.as_() }),
                })
                .collect();
            let views: Vec<_> = dequantized.iter().map(|a| a.view()).collect();
            let (wb, ws) = postprocess_modelpack_split_float(Isa::SCALAR, &views, &cfg);
            let dq = q.map(|q| {
                let (zp, scale) = (q.zero_point as f32, q.scale);
                (scale, (zp != 0.0).then(|| -zp * scale))
            });
            for (name, isa) in tiers().into_iter().filter(|(_, i)| !i.is_scalar()) {
                let sizes: Vec<_> = layers.iter().map(|a| (a.dim(), ANCHORS.len())).collect();
                let mut split = SplitDecode::new(isa, &sizes).unwrap();
                for a in layers {
                    assert!(split.push(a.view(), &ANCHORS, dq), "{what} on {name}");
                }
                let (b, s) = split.finish().unwrap();
                assert_eq!(bits(&b), bits(&wb), "{what} boxes {q:?} on {name}");
                assert_eq!(bits(&s), bits(&ws), "{what} scores {q:?} on {name}");
            }
        }
    }
    let mut rng = Rng(0x7777_1234_7777_1234);
    let shapes = [(40, 40, 80), (20, 20, 80), (3, 5, 80)];
    let raw: Vec<Array3<u64>> = shapes
        .iter()
        .map(|&(h, w, nc)| Array3::from_shape_fn((h, w, 3 * (nc + 5)), |_| rng.next()))
        .collect();
    let u8s: Vec<Array3<u8>> = raw.iter().map(|a| a.mapv(|x| x as u8)).collect();
    check("u8", &u8s);
    let i8s: Vec<Array3<i8>> = raw.iter().map(|a| a.mapv(|x| x as i8)).collect();
    check("i8", &i8s);
    let u16s: Vec<Array3<u16>> = raw.iter().map(|a| a.mapv(|x| x as u16)).collect();
    check("u16", &u16s);
    let i16s: Vec<Array3<i16>> = raw.iter().map(|a| a.mapv(|x| x as i16)).collect();
    check("i16", &i16s);
}

/// A quantized split model built for `layers` `(H, W, nc)`.
fn split_decoder(layers: &[(usize, usize, usize)], quant: Option<(f32, i32)>) -> crate::Decoder {
    let detections = layers
        .iter()
        .map(|&(h, w, nc)| configs::Detection {
            decoder: DecoderType::ModelPack,
            shape: vec![1, h, w, 3 * (nc + 5)],
            anchors: Some(ANCHORS.to_vec()),
            quantization: quant.map(Into::into),
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::Height, h),
                (DimName::Width, w),
                (DimName::NumAnchorsXFeatures, 3 * (nc + 5)),
            ],
            normalized: Some(true),
        })
        .collect();
    DecoderBuilder::default()
        .with_config_modelpack_det_split(detections)
        .with_score_threshold(0.25)
        .with_iou_threshold(0.45)
        .build()
        .unwrap()
}

/// Detection boxes as comparable bits.
fn box_bits(b: &[crate::DetectBox]) -> Vec<[u32; 6]> {
    b.iter()
        .map(|b| {
            [
                b.bbox.xmin.to_bits(),
                b.bbox.ymin.to_bits(),
                b.bbox.xmax.to_bits(),
                b.bbox.ymax.to_bits(),
                b.score.to_bits(),
                b.label as u32,
            ]
        })
        .collect()
}

/// A full quantized (`u8` with a zero point, `i8` without) and float (`f32`, `f16`) split decode at 40×40×3×85 plus
/// 20×20×3×85, NMS included, equals the scalar tier on every tier.
#[test]
fn split_decode_matches_the_scalar_tier() {
    let mut rng = Rng(0x0dd0_0dd0_0dd0_0dd0);
    let shapes = [(40, 40, 80), (20, 20, 80)];
    let raw: Vec<Array4<u64>> = shapes
        .iter()
        .map(|&(h, w, nc)| Array4::from_shape_fn((1, h, w, 3 * (nc + 5)), |_| rng.next()))
        .collect();
    let decoder_u8 = split_decoder(&shapes, Some((0.0855, 174)));
    let decoder_i8 = split_decoder(&shapes, Some((0.0855, 0)));
    let run = |decoder: &crate::Decoder, outputs: &[ArrayViewDQuantized]| {
        let (mut b, mut m) = (Vec::new(), Vec::new());
        decoder.decode_quantized(outputs, &mut b, &mut m).unwrap();
        box_bits(&b)
    };
    let u8s: Vec<Array4<u8>> = raw.iter().map(|a| a.mapv(|x| (x % 256) as u8)).collect();
    let i8s: Vec<Array4<i8>> = raw.iter().map(|a| a.mapv(|x| x as i8)).collect();
    let views_u8: Vec<ArrayViewDQuantized> = u8s.iter().map(|a| a.view().into()).collect();
    let views_i8: Vec<ArrayViewDQuantized> = i8s.iter().map(|a| a.view().into()).collect();
    for (what, d, views) in [
        ("u8", &decoder_u8, &views_u8),
        ("i8", &decoder_i8, &views_i8),
    ] {
        let want = forced_for_test::with("scalar", || run(d, views));
        assert!(!want.is_empty(), "{what}: the decode keeps boxes");
        for (name, _) in tiers() {
            let got = forced_for_test::with(name, || run(d, views));
            assert_eq!(got, want, "{what} on {name}");
        }
    }
    let float_decoder = split_decoder(&shapes, None);
    let f32s: Vec<Array4<f32>> = raw
        .iter()
        .map(|a| a.mapv(|x| ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 10.0))
        .collect();
    let f16s: Vec<Array4<f16>> = f32s.iter().map(|a| a.mapv(f16::from_f32)).collect();
    fn float_run<T>(d: &crate::Decoder, outputs: &[ArrayViewD<T>]) -> Vec<[u32; 6]>
    where
        T: num_traits::Float + num_traits::AsPrimitive<f32> + num_traits::AsPrimitive<u8>,
        T: Send + Sync + 'static,
        f32: num_traits::AsPrimitive<T>,
    {
        let (mut b, mut m) = (Vec::new(), Vec::new());
        d.decode_float(outputs, &mut b, &mut m).unwrap();
        box_bits(&b)
    }
    let v32: Vec<_> = f32s.iter().map(|a| a.view().into_dyn()).collect();
    let v16: Vec<_> = f16s.iter().map(|a| a.view().into_dyn()).collect();
    let want32 = forced_for_test::with("scalar", || float_run(&float_decoder, &v32));
    let want16 = forced_for_test::with("scalar", || float_run(&float_decoder, &v16));
    assert!(!want32.is_empty() && !want16.is_empty());
    for (name, _) in tiers() {
        let got = forced_for_test::with(name, || float_run(&float_decoder, &v32));
        assert_eq!(got, want32, "f32 on {name}");
        let got = forced_for_test::with(name, || float_run(&float_decoder, &v16));
        assert_eq!(got, want16, "f16 on {name}");
    }
}

/// Views of `a` `(H, W, C)`: standard, channel-planar memory seen through swapped axes (as a
/// ModelPack NCHW output arrives), Fortran order, and a negative stride.
fn seg_layouts<T: Copy + Default>(a: &Array3<T>) -> Vec<(&'static str, Array3<T>)> {
    let (h, w, c) = a.dim();
    let mut planar = Array3::<T>::default((c, h, w));
    planar.assign(&a.view().permuted_axes([2, 0, 1]));
    let planar = planar.permuted_axes([1, 2, 0]);
    let mut fortran = Array3::<T>::default((h, w, c).f());
    fortran.assign(a);
    let mut rev = a.slice(s![.., ..;-1, ..]).to_owned();
    rev.invert_axis(Axis(1));
    vec![
        ("standard", a.clone()),
        ("planar", planar),
        ("fortran", fortran),
        ("reversed", rev),
    ]
}

/// The quantized segmentation conversions equal the decoder's `mapv` formulas bit for bit, for
/// every type, odd shapes and every layout, in standard layout for a planar view and `map`'s
/// layout otherwise; negative strides, and `u8` unless planar, are left to the caller.
#[test]
fn seg_to_u8_matches_the_scalar_formulas() {
    let mut rng = Rng(0x4242_4242_4242_4242);
    for (h, w, c) in [(1, 1, 2), (7, 13, 3), (17, 31, 21), (2, 160, 2)] {
        let raw = Array3::from_shape_fn((h, w, c), |_| rng.next());
        macro_rules! check {
            ($t:ty, $f:expr) => {
                for (layout, a) in seg_layouts(&raw.mapv(|x| x as $t)) {
                    let want = a.mapv($f);
                    for (name, isa) in tiers() {
                        let what = format!("{} {layout} on {name}", stringify!($t));
                        let copy = stringify!($t) == "u8";
                        match seg_to_u8_array(isa, &a.view()) {
                            Some(got) => {
                                assert_eq!(got, want, "{what}");
                                assert_layout(&got, &want, layout, &what);
                            }
                            None => assert!(
                                isa.is_scalar()
                                    || layout == "reversed"
                                    || (copy && (layout != "planar" || a.is_standard_layout())),
                                "{what}"
                            ),
                        }
                    }
                }
            };
        }
        check!(i8, |x| (x as i16 + 128) as u8);
        check!(u16, |x| (x >> 8) as u8);
        check!(i16, |x| ((x as i32 + 32768) >> 8) as u8);
        check!(u32, |x| (x >> 24) as u8);
        check!(i32, |x| ((x as i64 + 2147483648) >> 24) as u8);
        check!(u8, |x| x);
    }
}

/// A NEON map of a planar view is standard layout; any other keeps `mapv`'s strides.
fn assert_layout(got: &Array3<u8>, want: &Array3<u8>, layout: &str, what: &str) {
    if layout == "planar" {
        assert!(got.is_standard_layout(), "{what}");
    } else {
        assert_eq!(got.strides(), want.strides(), "{what}");
    }
}

/// The decoder's scalar float normalisation of `seg`.
fn normalize_scalar<T>(seg: &ArrayView3<'_, T>) -> Array3<u8>
where
    T: num_traits::Float + num_traits::AsPrimitive<u8> + 'static,
    f32: num_traits::AsPrimitive<T>,
{
    use ndarray_stats::QuantileExt;
    use num_traits::AsPrimitive;
    let u8_max: T = 255.0_f32.as_();
    let max = *seg.max().unwrap_or(&u8_max);
    let min = *seg.min().unwrap_or(&0.0_f32.as_());
    seg.mapv(|x| ((x - min) / (max - min) * u8_max).as_())
}

/// The float segmentation normalisation equals the scalar code bit for bit on every tier, `f32`
/// and `f16`, for random maps, maps with NaN, infinities, signed zeros or one value, and every
/// layout.
#[test]
fn seg_normalize_matches_the_scalar_code() {
    let mut rng = Rng(0x9999_aaaa_bbbb_cccc);
    for (h, w, c) in [(1, 1, 2), (7, 13, 3), (17, 31, 21), (160, 160, 2)] {
        for kind in 0..5 {
            let a = Array3::from_shape_fn((h, w, c), |(y, x, k)| match kind {
                0 => rng.logit(),
                1 => ((rng.next() % 1000) as f32 - 300.0) * 0.01,
                2 if (y + x + k) % 2 == 0 => -0.0,
                2 => 0.0,
                3 => 4.5,
                _ if y == 0 && x == 0 && k == 1 => f32::INFINITY,
                _ => (rng.next() % 50) as f32,
            });
            for (layout, a) in seg_layouts(&a) {
                let want = normalize_scalar(&a.view());
                let h16 = a.mapv(f16::from_f32);
                let want16 = normalize_scalar(&h16.view());
                for (name, isa) in tiers() {
                    let what = format!("{h}x{w}x{c} kind {kind} {layout} on {name}");
                    match seg_normalize_array(isa, &a.view()) {
                        Some((got, _)) => {
                            assert_eq!(got, want, "f32 {what}");
                            assert_layout(&got, &want, layout, &what);
                        }
                        None => assert!(isa.is_scalar() || layout == "reversed", "f32 {what}"),
                    }
                    match seg_normalize_array(isa, &h16.view()) {
                        Some((got, _)) => {
                            assert_eq!(got, want16, "f16 {what}");
                            assert_layout(&got, &want16, layout, &what);
                        }
                        None => assert!(isa.is_scalar() || layout == "reversed", "f16 {what}"),
                    }
                }
            }
        }
    }
}

/// Mask bytes of a decoded segmentation.
fn mask_bytes(m: &[Segmentation]) -> Vec<Vec<u8>> {
    use edgefirst_tensor::{TensorMapTrait as _, TensorTrait as _};
    m.iter()
        .map(|s| {
            let t = s.segmentation.as_u8().expect("mask must be U8");
            let map = t.map_read().expect("map mask");
            map.as_slice().to_vec()
        })
        .collect()
}

fn seg_decoder(c: usize, h: usize, w: usize) -> crate::Decoder {
    DecoderBuilder::default()
        .with_config_modelpack_seg(configs::Segmentation {
            decoder: DecoderType::ModelPack,
            quantization: Some((1.0 / 255.0, 0).into()),
            shape: vec![1, c, h, w],
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::NumClasses, c),
                (DimName::Height, h),
                (DimName::Width, w),
            ],
        })
        .build()
        .unwrap()
}

/// A ModelPack segmentation decode (NCHW output) gives the scalar tier's mask on every tier,
/// for every quantized type and `f32`/`f16`/`f64`, and `segmentation_to_mask` of it agrees.
#[test]
fn seg_decode_matches_the_scalar_tier() {
    let mut rng = Rng(0x1357_9bdf_2468_ace0);
    let (c, h, w) = (5, 37, 61);
    let raw = Array4::from_shape_fn((1, c, h, w), |_| rng.next());
    let decoder = seg_decoder(c, h, w);
    let quant = |v: &ArrayViewDQuantized| {
        let (mut b, mut m) = (Vec::new(), Vec::new());
        decoder
            .decode_quantized(std::slice::from_ref(v), &mut b, &mut m)
            .unwrap();
        let mask = crate::segmentation_to_mask(seg3(&m[0]).view()).unwrap();
        (mask_bytes(&m), mask)
    };
    let a_u8 = raw.mapv(|x| x as u8);
    let a_i8 = raw.mapv(|x| x as i8);
    let a_u16 = raw.mapv(|x| x as u16);
    let a_i16 = raw.mapv(|x| x as i16);
    let a_u32 = raw.mapv(|x| x as u32);
    let a_i32 = raw.mapv(|x| x as i32);
    let all: [(&str, ArrayViewDQuantized); 6] = [
        ("u8", a_u8.view().into()),
        ("i8", a_i8.view().into()),
        ("u16", a_u16.view().into()),
        ("i16", a_i16.view().into()),
        ("u32", a_u32.view().into()),
        ("i32", a_i32.view().into()),
    ];
    for (what, v) in &all {
        let want = forced_for_test::with("scalar", || quant(v));
        for (name, _) in tiers() {
            assert_eq!(
                forced_for_test::with(name, || quant(v)),
                want,
                "{what} on {name}"
            );
        }
    }
    fn float<T>(d: &crate::Decoder, a: &Array4<T>) -> (Vec<Vec<u8>>, ndarray::Array2<u8>)
    where
        T: num_traits::Float + num_traits::AsPrimitive<f32> + num_traits::AsPrimitive<u8>,
        T: Send + Sync + 'static,
        f32: num_traits::AsPrimitive<T>,
    {
        let (mut b, mut m) = (Vec::new(), Vec::new());
        d.decode_float(&[a.view().into_dyn()], &mut b, &mut m)
            .unwrap();
        let mask = crate::segmentation_to_mask(seg3(&m[0]).view()).unwrap();
        (mask_bytes(&m), mask)
    }
    let f = raw.mapv(|x| ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 30.0);
    let f16s = f.mapv(f16::from_f32);
    let f64s = f.mapv(f64::from);
    let want = forced_for_test::with("scalar", || float(&decoder, &f));
    let want16 = forced_for_test::with("scalar", || float(&decoder, &f16s));
    let want64 = forced_for_test::with("scalar", || float(&decoder, &f64s));
    for (name, _) in tiers() {
        assert_eq!(
            forced_for_test::with(name, || float(&decoder, &f)),
            want,
            "f32 {name}"
        );
        let got = forced_for_test::with(name, || float(&decoder, &f16s));
        assert_eq!(got, want16, "f16 {name}");
        let got = forced_for_test::with(name, || float(&decoder, &f64s));
        assert_eq!(got, want64, "f64 {name}");
    }
}

/// An owned `(H, W, C)` copy of a decoded segmentation.
fn seg3(s: &Segmentation) -> Array3<u8> {
    use edgefirst_tensor::{TensorMapTrait as _, TensorTrait as _};
    let t = s.segmentation.as_u8().expect("mask must be U8");
    let m = t.map_read().expect("map mask");
    let sh = s.segmentation.shape();
    Array3::from_shape_vec((sh[0], sh[1], sh[2]), m.as_slice().to_vec()).expect("mask shape")
}

/// `segmentation_to_mask` (the per-pixel argmax) equals `argminmax`'s first maximum on every
/// tier for 2–21, 32, 33, 255 and 300 channels, odd widths, tie-heavy and full-range scores, and
/// non-standard layouts.
#[test]
fn segmentation_to_mask_matches_argminmax_on_every_tier() {
    use argminmax::ArgMinMax;
    let mut rng = Rng(0x2468_1357_2468_1357);
    let channels: Vec<usize> = (2..=21).chain([32, 33, 255, 300]).collect();
    for &c in &channels {
        for (h, w) in [(1, 1), (3, 17), (13, 31)] {
            for few in [true, false] {
                let a = Array3::from_shape_fn((h, w, c), |_| {
                    if few {
                        [0u8, 9, 255][(rng.next() % 3) as usize]
                    } else {
                        rng.next() as u8
                    }
                });
                let want = a
                    .as_standard_layout()
                    .as_slice()
                    .unwrap()
                    .chunks_exact(c)
                    .map(|px| px.argmax() as u8)
                    .collect::<Vec<_>>();
                for (layout, v) in seg_layouts(&a) {
                    for (name, _) in tiers() {
                        let got =
                            forced_for_test::with(name, || crate::segmentation_to_mask(v.view()));
                        let got = got.unwrap();
                        assert_eq!(
                            got.as_standard_layout().as_slice().unwrap(),
                            &want[..],
                            "{c} channels {h}x{w} few {few} {layout} on {name}"
                        );
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

/// ModelPack split (quantized and float) and segmentation (quantized and float) decodes fail
/// with `ForcedKernelUnavailable` under a tier the CPU lacks or does not know, and
/// `segmentation_to_mask` and the public NMS run the scalar code.
#[test]
fn modelpack_paths_follow_the_forced_tier_rule() {
    let unavailable = unavailable_tiers();
    let split = split_decoder(&[(3, 5, 2)], Some((0.1, 3)));
    let split_float = split_decoder(&[(3, 5, 2)], None);
    let seg = seg_decoder(3, 4, 6);
    let q = Array4::<u8>::from_elem((1, 3, 5, 21), 200);
    let f = Array4::<f32>::from_elem((1, 3, 5, 21), 2.0);
    let sq = Array4::<i8>::from_elem((1, 3, 4, 6), 1);
    let sf = Array4::<f32>::from_elem((1, 3, 4, 6), 1.0);
    type Run<'a> = Box<dyn Fn() -> crate::DecoderResult<()> + 'a>;
    let runs: [(&str, Run); 4] = [
        (
            "split quant",
            Box::new(|| {
                split.decode_quantized(&[q.view().into()], &mut Vec::new(), &mut Vec::new())
            }),
        ),
        (
            "split float",
            Box::new(|| {
                split_float.decode_float(&[f.view().into_dyn()], &mut Vec::new(), &mut Vec::new())
            }),
        ),
        (
            "seg quant",
            Box::new(|| {
                seg.decode_quantized(&[sq.view().into()], &mut Vec::new(), &mut Vec::new())
            }),
        ),
        (
            "seg float",
            Box::new(|| {
                seg.decode_float(&[sf.view().into_dyn()], &mut Vec::new(), &mut Vec::new())
            }),
        ),
    ];
    for (what, run) in &runs {
        run().unwrap_or_else(|e| panic!("{what}: unforced decode failed: {e:?}"));
        for tier in &unavailable {
            let r = forced_for_test::with(tier, run);
            assert!(
                matches!(r, Err(crate::DecoderError::ForcedKernelUnavailable { .. })),
                "{what} forced {tier}: {r:?}"
            );
        }
    }
    let mask = Array3::from_shape_fn((5, 7, 4), |(y, x, c)| ((y * 7 + x + c * 3) % 11) as u8);
    let want = crate::segmentation_to_mask(mask.view()).unwrap();
    for tier in &unavailable {
        let got = forced_for_test::with(tier, || crate::segmentation_to_mask(mask.view()));
        assert_eq!(got.unwrap(), want, "{tier}");
    }
}

/// One output's `((height, width, channels), anchors)`.
type Layer = ((usize, usize, usize), usize);

/// `SplitDecode::new` refuses, before reserving, outputs whose class counts `push` would refuse.
#[test]
fn split_decode_new_checks_class_counts_first() {
    for (name, isa) in tiers() {
        let ok = [((9, 15, 18), 3), ((17, 30, 18), 3)];
        assert_eq!(
            SplitDecode::new(isa, &ok).is_some(),
            !isa.is_scalar(),
            "{name}"
        );
        let bad: [&[Layer]; 5] = [
            &[((9, 15, 18), 3), ((17, 30, 21), 3)],
            &[((9, 15, 18), 3), ((1 << 20, 1 << 20, 3 << 40), 3)],
            &[((9, 15, 17), 3)],
            &[((9, 15, 15), 3)],
            &[((9, 15, 18), 0)],
        ];
        for outputs in bad {
            assert!(
                SplitDecode::new(isa, outputs).is_none(),
                "{name} {outputs:?}"
            );
        }
        let huge = [((usize::MAX, 2, 18), 3)];
        assert!(SplitDecode::new(isa, &huge).is_none(), "{name}");
    }
}
