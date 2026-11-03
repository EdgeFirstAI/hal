// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Instance masks, proto dequantize, conversions and threshold on every tier this CPU has.
//!
//! Masks on a NEON tier are within 1 of the scalar tier (the integer logits are exact, so only
//! the sigmoid differs; float logits also differ in summation order). The scalar tier's quantized
//! masks equal an ndarray `mapv` + `dot` reference bit for bit. Conversions and the threshold are
//! bit-identical on every tier.

use half::f16;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3, Axis};
use num_traits::{AsPrimitive, PrimInt};

use super::*;
use crate::yolo::{decode_segdet_f32_on, decode_segdet_quant_on};
use crate::{BoundingBox, DetectBox};

use crate::neon_dispatch::tiers;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn int(&mut self, lo: i64, hi: i64) -> i64 {
        lo + (self.next() % (hi - lo + 1) as u64) as i64
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

const H: usize = 23;
const W: usize = 37;

/// Boxes covering the frame, its edges, single cells, empty and inverted crops, and the
/// out-of-frame coordinates the decoder clamps.
fn boxes(rng: &mut Rng) -> Vec<DetectBox> {
    let mut b: Vec<[f32; 4]> = vec![
        [0.0, 0.0, 1.0, 1.0],
        [0.1, 0.2, 0.6, 0.9],
        [0.5, 0.5, 0.52, 0.53],
        [0.9, 0.0, 1.3, 0.4],
        [-0.2, 0.7, 0.3, 1.0],
        [0.4, 0.4, 0.4, 0.8],
        [0.7, 0.3, 0.2, 0.6],
        [0.0, 0.95, 0.05, 1.0],
    ];
    for _ in 0..12 {
        let (x, y) = (rng.unit(), rng.unit());
        let (w, h) = (rng.unit() * 0.6, rng.unit() * 0.6);
        b.push([x, y, (x + w).min(1.5), (y + h).min(1.5)]);
    }
    b.into_iter()
        .enumerate()
        .map(|(i, [xmin, ymin, xmax, ymax])| DetectBox {
            bbox: BoundingBox {
                xmin,
                ymin,
                xmax,
                ymax,
            },
            score: 0.5,
            label: i % 3,
        })
        .collect()
}

/// `[H, W, K]` protos as NHWC memory and as NCHW memory behind swapped strides.
fn proto_layouts<T: Copy>(v: &[T], k: usize) -> (Array3<T>, Array3<T>) {
    let nhwc = Array3::from_shape_vec((H, W, k), v.to_vec()).unwrap();
    let mut nchw = Array3::from_shape_vec((k, H, W), vec![v[0]; v.len()]).unwrap();
    for ((y, x, c), &e) in nhwc.indexed_iter() {
        nchw[[c, y, x]] = e;
    }
    (nhwc, nchw)
}

/// `[N, K]` coefficients behind transposed strides, as the decoder slices them from `[K, N]`.
fn coefficients<T: Copy>(v: &[T], n: usize, k: usize) -> Array2<T> {
    Array2::from_shape_vec((k, n), v.to_vec()).unwrap()
}

type Masks = Vec<(DetectBox, BoundingBox, Array3<u8>)>;

/// Masks within 1 of `scalar`, same boxes and crops; returns (worst, differing, total values).
fn compare(label: &str, got: &Masks, scalar: &Masks) -> (i32, usize, usize) {
    assert_eq!(got.len(), scalar.len(), "{label}");
    let (mut worst, mut differ, mut total) = (0, 0, 0);
    for (i, (g, s)) in got.iter().zip(scalar).enumerate() {
        assert_eq!(g.0, s.0, "{label} box {i}");
        assert_eq!(g.1, s.1, "{label} roi {i}");
        assert_eq!(g.2.dim(), s.2.dim(), "{label} mask shape {i}");
        for (&a, &b) in g.2.iter().zip(s.2.iter()) {
            let d = (i32::from(a) - i32::from(b)).abs();
            assert!(d <= 1, "{label} box {i}: {a} vs scalar {b}");
            worst = worst.max(d);
            differ += usize::from(d != 0);
            total += 1;
        }
    }
    (worst, differ, total)
}

/// Quantized masks through ndarray: `mapv` to i64, `dot`, then the scalar sigmoid.
fn ndarray_reference<M, P>(
    boxes: &[DetectBox],
    masks: ArrayView2<M>,
    protos: ArrayView3<P>,
    qm: Quantization,
    qp: Quantization,
) -> Vec<Array3<u8>>
where
    M: PrimInt + AsPrimitive<i64>,
    P: PrimInt + AsPrimitive<i64>,
{
    let (h, w, k) = protos.dim();
    boxes
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let (c, _) = crate::yolo::proto_crop(h, w, &b.bbox.to_canonical()).unwrap();
            let crop = protos.slice(ndarray::s![c.y0..c.y1, c.x0..c.x1, ..]);
            let (ch, cw) = (crop.dim().0, crop.dim().1);
            let mask = masks.row(i).mapv(|x| x.as_() - i64::from(qm.zero_point));
            let mask = mask.to_shape((1, k)).unwrap().to_owned();
            let p = crop.to_shape([ch * cw, k]).unwrap();
            let p = p
                .reversed_axes()
                .mapv(|x| x.as_() - i64::from(qp.zero_point));
            let seg = mask.dot(&p).into_shape_with_order((ch, cw, 1)).unwrap();
            let scale = qm.scale * qp.scale;
            seg.map(|x| {
                let val = *x as f32 * scale;
                ((1.0 / (1.0 + (-val).exp())) * 255.0).round() as u8
            })
        })
        .collect()
}

/// An integer type of the quantized mask tests.
trait IntCase: PrimInt + AsPrimitive<i64> + AsPrimitive<i128> + Send + Sync + 'static {
    const NAME: &'static str;
    fn make(rng: &mut Rng) -> Self;
    /// Zero points in range, plus one outside it when the type allows it.
    fn zero_points() -> (Vec<i32>, Option<i32>);
}

macro_rules! int_case {
    ($t:ty, $zps:expr, $bad:expr) => {
        impl IntCase for $t {
            const NAME: &'static str = stringify!($t);
            fn make(rng: &mut Rng) -> Self {
                rng.int(<$t>::MIN as i64, <$t>::MAX as i64) as $t
            }
            fn zero_points() -> (Vec<i32>, Option<i32>) {
                ($zps.to_vec(), $bad)
            }
        }
    };
}
int_case!(i8, [20, -128, 127], Some(200));
int_case!(u8, [128, 0, 255], Some(-1));
int_case!(i16, [-300, 0], Some(40000));
int_case!(u16, [40000, 7], Some(-5));
int_case!(i32, [3], None);

/// The kernel a NEON tier should pick for these types and layout.
fn expected(name: &str, eight: bool, nchw: bool, isa: Isa) -> QuantKernel {
    match (isa, eight, nchw) {
        _ if isa.is_scalar() => QuantKernel::Scalar,
        _ if name.contains("i32") => QuantKernel::Scalar,
        (_, true, true) => QuantKernel::Dot8Nchw,
        (_, true, false) if isa.dotprod() => QuantKernel::Dot8NhwcDotprod,
        (_, true, false) => QuantKernel::Dot8Nhwc,
        (_, false, true) => QuantKernel::WideNchw,
        (_, false, false) => QuantKernel::WideNhwc,
    }
}

fn check_quant<M: IntCase, P: IntCase>(seed: u64) {
    let mut stats = (0usize, 0usize);
    let mut rng = Rng(seed);
    let bx = boxes(&mut rng);
    let eight = std::mem::size_of::<M>() == 1 && std::mem::size_of::<P>() == 1;
    for k in [32usize, 7, 40] {
        let pv: Vec<P> = (0..H * W * k).map(|_| P::make(&mut rng)).collect();
        let (nhwc, nchw) = proto_layouts(&pv, k);
        let nchw_view = nchw.view().permuted_axes([1, 2, 0]);
        let cv: Vec<M> = (0..bx.len() * k).map(|_| M::make(&mut rng)).collect();
        let coeff = coefficients(&cv, bx.len(), k);
        let masks = coeff.t();
        let (zms, _) = M::zero_points();
        let (zps, _) = P::zero_points();
        for (zm, zp) in zms.iter().zip(zps.iter().cycle()) {
            let qm = Quantization::new(0.0187, *zm);
            let qp = Quantization::new(0.0213, *zp);
            for (layout, protos) in [("nhwc", nhwc.view()), ("nchw", nchw_view)] {
                let det: Vec<_> = bx
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(i, b)| (b, i))
                    .collect();
                let run = |isa| decode_segdet_quant_on(isa, det.clone(), masks, protos, qm, qp);
                let scalar = run(Isa::SCALAR).unwrap();
                if k == 32 {
                    let want = ndarray_reference(&bx, masks, protos, qm, qp);
                    for (s, w) in scalar.iter().zip(&want) {
                        assert_eq!(&s.2, w, "{}x{} scalar tier vs ndarray", M::NAME, P::NAME);
                    }
                }
                for (tier, isa) in tiers() {
                    let label = format!(
                        "{}x{} {layout} k {k} zm {zm} zp {zp} {tier}",
                        M::NAME,
                        P::NAME
                    );
                    let kernel = quant_kernel::<M, P>(isa, &protos, qm, qp);
                    assert_eq!(
                        kernel,
                        expected(&label, eight, layout == "nchw", isa),
                        "{label}"
                    );
                    let (_, differ, total) = compare(&label, &run(isa).unwrap(), &scalar);
                    if !isa.is_scalar() {
                        stats.0 += differ;
                        stats.1 += total;
                    }
                }
            }
        }
    }
    println!(
        "{}x{}: {} of {} NEON mask values differ from scalar",
        M::NAME,
        P::NAME,
        stats.0,
        stats.1
    );
}

#[test]
fn quant_masks_on_every_tier() {
    check_quant::<i8, i8>(1);
    check_quant::<u8, u8>(2);
    check_quant::<i8, u8>(3);
    check_quant::<u8, i8>(4);
    check_quant::<i16, i8>(5);
    check_quant::<i8, i16>(6);
    check_quant::<i16, i16>(7);
    check_quant::<u16, u8>(8);
    check_quant::<u8, u16>(9);
    check_quant::<u16, u16>(10);
    check_quant::<i32, i8>(11);
}

fn check_out_of_range<M: IntCase, P: IntCase>() {
    let protos = Array3::<P>::zeros((4, 4, 8));
    let ok = Quantization::new(0.1, 0);
    for (tier, isa) in tiers() {
        if let (_, Some(bad)) = M::zero_points() {
            let q = Quantization::new(0.1, bad);
            let k = quant_kernel::<M, P>(isa, &protos.view(), q, ok);
            assert_eq!(k, QuantKernel::Scalar, "{tier} {} zm {bad}", M::NAME);
        }
        if let (_, Some(bad)) = P::zero_points() {
            let q = Quantization::new(0.1, bad);
            let k = quant_kernel::<M, P>(isa, &protos.view(), ok, q);
            assert_eq!(k, QuantKernel::Scalar, "{tier} {} zp {bad}", P::NAME);
        }
        let strided = Array3::<P>::zeros((4, 8, 8));
        let strided = strided.slice(ndarray::s![.., ..;2, ..]);
        let k = quant_kernel::<M, P>(isa, &strided, ok, ok);
        assert_eq!(k, QuantKernel::Scalar, "{tier} strided protos");
    }
}

#[test]
fn quant_masks_outside_the_kernel_domain_run_scalar() {
    check_out_of_range::<i8, i8>();
    check_out_of_range::<u8, u16>();
    check_out_of_range::<i16, u8>();
}

/// A float type of the float mask tests.
trait FloatCase: num_traits::Float + AsPrimitive<f32> + Send + Sync + 'static {
    const NAME: &'static str;
    fn make(rng: &mut Rng, mag: f32) -> Self;
}

impl FloatCase for f32 {
    const NAME: &'static str = "f32";
    fn make(rng: &mut Rng, mag: f32) -> Self {
        (rng.unit() * 2.0 - 1.0) * mag
    }
}
impl FloatCase for f16 {
    const NAME: &'static str = "f16";
    fn make(rng: &mut Rng, mag: f32) -> Self {
        f16::from_f32((rng.unit() * 2.0 - 1.0) * mag)
    }
}
impl FloatCase for f64 {
    const NAME: &'static str = "f64";
    fn make(rng: &mut Rng, mag: f32) -> Self {
        f64::from((rng.unit() * 2.0 - 1.0) * mag)
    }
}

fn check_float<M: FloatCase, P: FloatCase>(seed: u64) {
    let mut stats = (0usize, 0usize);
    let mut rng = Rng(seed);
    let bx = boxes(&mut rng);
    for k in [32usize, 7, 40] {
        let pv: Vec<P> = (0..H * W * k).map(|_| P::make(&mut rng, 3.0)).collect();
        let (nhwc, nchw) = proto_layouts(&pv, k);
        let nchw_view = nchw.view().permuted_axes([1, 2, 0]);
        let cv: Vec<M> = (0..bx.len() * k).map(|_| M::make(&mut rng, 1.5)).collect();
        let coeff = coefficients(&cv, bx.len(), k);
        let masks = coeff.t();
        for (layout, protos) in [("nhwc", nhwc.view()), ("nchw", nchw_view)] {
            let det: Vec<_> = bx
                .iter()
                .copied()
                .enumerate()
                .map(|(i, b)| (b, i))
                .collect();
            let run = |isa| decode_segdet_f32_on(isa, det.clone(), masks, protos);
            let scalar = run(Isa::SCALAR).unwrap();
            for (tier, isa) in tiers() {
                let label = format!("{}x{} {layout} k {k} {tier}", M::NAME, P::NAME);
                let kernel = float_kernel::<P>(isa, &protos);
                let want = match (isa, P::NAME, layout) {
                    _ if isa.is_scalar() => FloatKernel::Scalar,
                    (_, "f64", _) => FloatKernel::Scalar,
                    (_, _, "nhwc") => FloatKernel::F32Nhwc,
                    _ => FloatKernel::F32Nchw,
                };
                assert_eq!(kernel, want, "{label}");
                let (_, differ, total) = compare(&label, &run(isa).unwrap(), &scalar);
                if !isa.is_scalar() {
                    stats.0 += differ;
                    stats.1 += total;
                }
            }
        }
    }
    println!(
        "{}x{}: {} of {} NEON mask values differ from scalar",
        M::NAME,
        P::NAME,
        stats.0,
        stats.1
    );
}

#[test]
fn float_masks_on_every_tier() {
    check_float::<f32, f32>(21);
    check_float::<f16, f16>(22);
    check_float::<f32, f16>(23);
    check_float::<f16, f32>(24);
    check_float::<f64, f64>(25);
}

/// `view.map` of the `dequant_3d!` closure, bits and strides.
fn reference<T: Copy + AsPrimitive<f32>, D: ndarray::Dimension>(
    view: ndarray::ArrayView<T, D>,
    q: Quantization,
) -> Array<f32, D> {
    view.map(|v| {
        let val: f32 = v.as_();
        (val - q.zero_point as f32) * q.scale
    })
}

fn check_dequant<T: IntCase + AsPrimitive<f32>>(seed: u64) {
    let mut rng = Rng(seed);
    let v: Vec<T> = (0..6 * 35 * 9).map(|_| T::make(&mut rng)).collect();
    let a = Array3::from_shape_vec((6, 35, 9), v).unwrap();
    let views = [
        a.view(),
        a.view().permuted_axes([1, 2, 0]),
        a.slice(ndarray::s![.., 1..30, ..]),
        a.slice(ndarray::s![..;-1, .., ..]),
    ];
    for q in [
        Quantization::new(0.0208, -115),
        Quantization::new(1.0, 0),
        Quantization::new(3e-4, 40000),
    ] {
        for (tier, isa) in tiers() {
            for (i, view) in views.iter().enumerate() {
                let got = dequant_array(*view, q, isa);
                let want = reference(*view, q);
                assert_eq!(got.strides(), want.strides(), "{tier} {} view {i}", T::NAME);
                let bits = |x: &Array3<f32>| x.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&got), bits(&want), "{tier} {} view {i}", T::NAME);
            }
            let flat = a.index_axis(Axis(0), 2);
            let got = dequant_array(flat.t(), q, isa);
            let want = reference(flat.t(), q);
            assert_eq!(got.strides(), want.strides());
            assert!(got
                .iter()
                .zip(&want)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
    }
}

#[test]
fn dequant_array_is_map_on_every_tier() {
    check_dequant::<i8>(31);
    check_dequant::<u8>(32);
    check_dequant::<i16>(33);
    check_dequant::<u16>(34);
    check_dequant::<i32>(35);
}

fn check_dequant_protos<T: IntCase + AsPrimitive<f32>>(seed: u64) {
    let mut rng = Rng(seed);
    for k in [32usize, 5] {
        let v: Vec<T> = (0..H * W * k).map(|_| T::make(&mut rng)).collect();
        let (nhwc, nchw) = proto_layouts(&v, k);
        let q = Quantization::new(0.0208, 9);
        for (layout, protos) in [
            ("nhwc", nhwc.view()),
            ("nchw", nchw.view().permuted_axes([1, 2, 0])),
        ] {
            let want: Vec<u32> = protos
                .iter()
                .map(|&s| {
                    let v: f32 = s.as_();
                    ((v - q.zero_point as f32) * q.scale).to_bits()
                })
                .collect();
            for (tier, isa) in tiers() {
                let mut got = vec![f32::NAN; want.len()];
                let vector = dequant_protos(isa, &protos, q, &mut got);
                let neon = !isa.is_scalar() && std::mem::size_of::<T>() <= 2;
                assert_eq!(vector, neon, "{tier} {} {layout}", T::NAME);
                if vector {
                    let got: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                    assert_eq!(got, want, "{tier} {} {layout} k {k}", T::NAME);
                }
            }
        }
    }
}

#[test]
fn proto_dequant_is_bit_exact_on_every_tier() {
    check_dequant_protos::<u8>(41);
    check_dequant_protos::<i8>(42);
    check_dequant_protos::<i16>(43);
    check_dequant_protos::<u16>(44);
    check_dequant_protos::<i32>(45);
}

#[test]
fn threshold_matches_scalar_on_every_tier() {
    let mut rng = Rng(51);
    let v: Vec<u8> = (0..41 * 29).map(|_| rng.int(0, 255) as u8).collect();
    let seg = Array3::from_shape_vec((41, 29, 1), v).unwrap();
    for t in [0u8, 1, 127, 128, 255] {
        let want = seg.index_axis(Axis(2), 0).map(|&x| u8::from(x >= t));
        for (tier, isa) in tiers() {
            let got = threshold_mask(isa, &seg.view(), t);
            assert_eq!(got.is_some(), !isa.is_scalar(), "{tier}");
            if let Some(got) = got {
                assert_eq!(got, want, "{tier} threshold {t}");
            }
        }
    }
}

#[test]
fn coefficient_rows_longer_than_the_stack_buffer() {
    let mut rng = Rng(61);
    let k = 300;
    let pv: Vec<i8> = (0..4 * 5 * k).map(|_| rng.int(-128, 127) as i8).collect();
    let protos = Array3::from_shape_vec((4, 5, k), pv).unwrap();
    let cv: Vec<i8> = (0..2 * k).map(|_| rng.int(-128, 127) as i8).collect();
    let coeff = Array2::from_shape_vec((2, k), cv).unwrap();
    let full = DetectBox {
        bbox: BoundingBox {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
        },
        score: 1.0,
        label: 0,
    };
    let det = vec![(full, 0), (full, 1)];
    let q = Quantization::new(0.001, 3);
    let scalar =
        decode_segdet_quant_on(Isa::SCALAR, det.clone(), coeff.view(), protos.view(), q, q)
            .unwrap();
    for (tier, isa) in tiers() {
        let got =
            decode_segdet_quant_on(isa, det.clone(), coeff.view(), protos.view(), q, q).unwrap();
        compare(tier, &got, &scalar);
    }
    let fp: Vec<f32> = protos.iter().map(|&v| f32::from(v) * 0.01).collect();
    let fprotos = Array3::from_shape_vec((4, 5, k), fp).unwrap();
    let fc = coeff.mapv(|v| f32::from(v) * 0.01);
    let scalar = decode_segdet_f32_on(Isa::SCALAR, det.clone(), fc.view(), fprotos.view()).unwrap();
    for (tier, isa) in tiers() {
        let got = decode_segdet_f32_on(isa, det.clone(), fc.view(), fprotos.view()).unwrap();
        compare(tier, &got, &scalar);
    }
}
