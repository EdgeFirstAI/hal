//! EdgeFirst HAL - Decoders
#![allow(clippy::excessive_precision)]
use ndarray::{Array, Array2, Array3, ArrayView, ArrayView1, ArrayView3, Dimension};
use num_traits::{AsPrimitive, Float, PrimInt, Signed};
use std::ops::{Add, Mul, Sub};
pub mod byte;
pub mod error;
pub mod float;
pub mod modelpack;
pub mod yolo;

mod decoder;
pub use decoder::*;

pub use error::Error;

use crate::{
    decoder::configs::QuantTuple, modelpack::modelpack_segmentation_to_mask,
    yolo::yolo_segmentation_to_mask,
};

pub trait BBoxTypeTrait {
    /// Converts the bbox into XYXY quantized format. The XYXY quantized values
    /// are scaled to the zero point and and doubled. Doubled values ensure that
    /// no rounding is needed when converting BBox formats that typically
    /// require dividing some of the inputs by 2.
    ///
    /// Generally, A should be a wider, signed, integer type than B. This
    /// ensures no over or under flow.
    fn to_xyxy_quant<A: PrimInt + 'static, B: AsPrimitive<A>>(input: &[B; 4], zp: A) -> [A; 4];

    /// Converts the bbox into XYXY float format.
    fn to_xyxy_float<A: Float + 'static, B: AsPrimitive<A>>(input: &[B; 4]) -> [A; 4];

    #[inline(always)]
    /// Converts the bbox into XYXY quantized format. The XYXY quantized values
    /// are scaled to the zero point and and doubled. Doubled values ensure that
    /// no rounding is needed when converting BBox formats that typically
    /// require dividing some of the inputs by 2.
    ///
    /// Generally, A should be a signed integer type wider than B. This
    /// ensures no overflow or underflow errors.
    fn ndarray_to_xyxy_quant<A: PrimInt + 'static, B: AsPrimitive<A>>(
        input: ArrayView1<B>,
        zp: A,
    ) -> [A; 4] {
        Self::to_xyxy_quant(&[input[0], input[1], input[2], input[3]], zp)
    }

    #[inline(always)]
    /// Converts the bbox into XYXY float format.
    fn ndarray_to_xyxy_float<A: Float + 'static, B: AsPrimitive<A>>(
        input: ArrayView1<B>,
    ) -> [A; 4] {
        Self::to_xyxy_float(&[input[0], input[1], input[2], input[3]])
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XYXY {}

impl BBoxTypeTrait for XYXY {
    fn to_xyxy_quant<A: PrimInt + 'static, B: AsPrimitive<A>>(input: &[B; 4], zp: A) -> [A; 4] {
        input.map(|b| (b.as_() - zp) << 1)
    }

    fn to_xyxy_float<A: Float + 'static, B: AsPrimitive<A>>(input: &[B; 4]) -> [A; 4] {
        input.map(|b| b.as_())
    }

    #[inline(always)]
    fn ndarray_to_xyxy_quant<A: PrimInt + 'static, B: AsPrimitive<A>>(
        input: ArrayView1<B>,
        zp: A,
    ) -> [A; 4] {
        [
            (input[0].as_() - zp) << 1,
            (input[1].as_() - zp) << 1,
            (input[2].as_() - zp) << 1,
            (input[3].as_() - zp) << 1,
        ]
    }

    #[inline(always)]
    fn ndarray_to_xyxy_float<A: Float + 'static, B: AsPrimitive<A>>(
        input: ArrayView1<B>,
    ) -> [A; 4] {
        [
            input[0].as_(),
            input[1].as_(),
            input[2].as_(),
            input[3].as_(),
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XYWH {}

impl BBoxTypeTrait for XYWH {
    #[inline(always)]
    fn to_xyxy_quant<A: PrimInt + 'static, B: AsPrimitive<A>>(input: &[B; 4], zp: A) -> [A; 4] {
        [
            ((input[0].as_() - zp) << 1) - (input[2].as_() - zp),
            ((input[1].as_() - zp) << 1) - (input[3].as_() - zp),
            ((input[0].as_() - zp) << 1) + (input[2].as_() - zp),
            ((input[1].as_() - zp) << 1) + (input[3].as_() - zp),
        ]
    }

    #[inline(always)]
    fn to_xyxy_float<A: Float + 'static, B: AsPrimitive<A>>(input: &[B; 4]) -> [A; 4] {
        let half = A::from(0.5).unwrap();
        [
            (input[0].as_()) - (input[2].as_() * half),
            (input[1].as_()) - (input[3].as_() * half),
            (input[0].as_()) + (input[2].as_() * half),
            (input[1].as_()) + (input[3].as_() * half),
        ]
    }

    #[inline(always)]
    fn ndarray_to_xyxy_quant<A: PrimInt + 'static, B: AsPrimitive<A>>(
        input: ArrayView1<B>,
        zp: A,
    ) -> [A; 4] {
        [
            ((input[0].as_() - zp) << 1) - (input[2].as_() - zp),
            ((input[1].as_() - zp) << 1) - (input[3].as_() - zp),
            ((input[0].as_() - zp) << 1) + (input[2].as_() - zp),
            ((input[1].as_() - zp) << 1) + (input[3].as_() - zp),
        ]
    }

    #[inline(always)]
    fn ndarray_to_xyxy_float<A: Float + 'static, B: AsPrimitive<A>>(
        input: ArrayView1<B>,
    ) -> [A; 4] {
        let half = A::from(0.5).unwrap();
        [
            (input[0].as_()) - (input[2].as_() * half),
            (input[1].as_()) - (input[3].as_() * half),
            (input[0].as_()) + (input[2].as_() * half),
            (input[1].as_()) + (input[3].as_() * half),
        ]
    }
}

#[derive(Debug, Copy, Clone)]
pub struct Quantization {
    pub scale: f32,
    pub zero_point: i32,
}

impl Quantization {
    pub fn new(scale: f32, zero_point: i32) -> Self {
        Self { scale, zero_point }
    }

    pub fn from_array<F: AsPrimitive<i32> + AsPrimitive<f32>>(value: [F; 2]) -> Self {
        Quantization {
            scale: value[0].as_(),
            zero_point: value[1].as_(),
        }
    }
}

impl From<QuantTuple> for Quantization {
    fn from(quant_tuple: QuantTuple) -> Quantization {
        Quantization {
            scale: quant_tuple.0,
            zero_point: quant_tuple.1,
        }
    }
}

impl<S, Z> From<(S, Z)> for Quantization
where
    S: AsPrimitive<f32>,
    Z: AsPrimitive<i32>,
{
    fn from((scale, zp): (S, Z)) -> Quantization {
        Self {
            scale: scale.as_(),
            zero_point: zp.as_(),
        }
    }
}

impl Default for Quantization {
    fn default() -> Self {
        Self {
            scale: 1.0,
            zero_point: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DetectBox {
    pub bbox: BoundingBox,
    /// model-specific score for this detection, higher implies more confidence
    pub score: f32,
    /// label index for this detection
    pub label: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BoundingBox {
    /// left-most normalized coordinate of the bounding box
    pub xmin: f32,
    /// top-most normalized coordinate of the bounding box
    pub ymin: f32,
    /// right-most normalized coordinate of the bounding box
    pub xmax: f32,
    /// bottom-most normalized coordinate of the bounding box
    pub ymax: f32,
}

impl BoundingBox {
    /// Transforms BoundingBox so that xmin <= xmax and ymin <= ymax
    pub fn to_canonical(&self) -> Self {
        let xmin = self.xmin.min(self.xmax);
        let xmax = self.xmin.max(self.xmax);
        let ymin = self.ymin.min(self.ymax);
        let ymax = self.ymin.max(self.ymax);
        BoundingBox {
            xmin,
            ymin,
            xmax,
            ymax,
        }
    }
}

impl From<BoundingBox> for [f32; 4] {
    fn from(b: BoundingBox) -> Self {
        [b.xmin, b.ymin, b.xmax, b.ymax]
    }
}

impl From<[f32; 4]> for BoundingBox {
    fn from(arr: [f32; 4]) -> Self {
        BoundingBox {
            xmin: arr[0],
            ymin: arr[1],
            xmax: arr[2],
            ymax: arr[3],
        }
    }
}

impl DetectBox {
    /// Check if one detect box is equal to another detect box, within the given
    /// delta
    pub fn equal_within_delta(&self, rhs: &DetectBox, delta: f32) -> bool {
        let eq_delta = |a: f32, b: f32| (a - b).abs() <= delta;
        self.label == rhs.label
            && eq_delta(self.score, rhs.score)
            && eq_delta(self.bbox.xmin, rhs.bbox.xmin)
            && eq_delta(self.bbox.ymin, rhs.bbox.ymin)
            && eq_delta(self.bbox.xmax, rhs.bbox.xmax)
            && eq_delta(self.bbox.ymax, rhs.bbox.ymax)
    }
}

// impl edgefirst_tracker::DetectionBox for DetectBox {
//     fn bbox(&self) -> [f32; 4] {
//         self.bbox.into()
//     }

//     fn score(&self) -> f32 {
//         self.score
//     }

//     fn label(&self) -> usize {
//         self.label
//     }
// }

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Segmentation {
    /// left-most normalized coordinate of the segmentation box
    pub xmin: f32,
    /// top-most normalized coordinate of the segmentation box
    pub ymin: f32,
    /// right-most normalized coordinate of the segmentation box
    pub xmax: f32,
    /// bottom-most normalized coordinate of the segmentation box
    pub ymax: f32,
    /// 3D segmentation array. If the last dimension is 1, values equal or above
    /// 128 are considered objects. Otherwise the object is the argmax index
    pub segmentation: Array3<u8>,
}

/// A quantized bounding box used to process boxes faster. The coordinates are
/// 2x the actual coordinates to prevent needing to round for XYWH boxes. The
/// coordinates should be shifted to the zero point already.
///
/// When choosing T, make sure to choose a type that is wider than the input
/// data to prevent overflow or underflow
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectBoxQuantized<
    BOX: Signed + PrimInt + AsPrimitive<f32>,
    SCORE: PrimInt + AsPrimitive<f32>,
> {
    pub bbox: BoundingBoxQuantized<BOX>,
    /// model-specific score for this detection, higher implies more
    /// confidence.
    pub score: SCORE,
    /// label index for this detect
    pub label: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BoundingBoxQuantized<T: Copy + Signed + Mul + Add + Sub + Ord + AsPrimitive<f32>> {
    /// 2x left-most coordinate of the bounding box. Should already be shifted
    /// to zero point
    pub xmin: T,
    /// 2x top-most coordinate of the bounding box. Should already be shifted to
    /// zero point
    pub ymin: T,
    /// 2x right-most coordinate of the bounding box. Should already be shifted
    /// to zero point
    pub xmax: T,
    /// 2x bottom-most coordinate of the bounding box. Should already be shifted
    /// to zero point
    pub ymax: T,
}

impl<T: Copy + Signed + Mul + Add + Sub + Ord + AsPrimitive<f32>> BoundingBoxQuantized<T> {
    pub fn to_array(&self) -> [T; 4] {
        [self.xmin, self.ymin, self.xmax, self.ymax]
    }

    pub fn from_array(arr: &[T; 4]) -> Self {
        Self {
            xmin: arr[0],
            ymin: arr[1],
            xmax: arr[2],
            ymax: arr[3],
        }
    }
}

/// Turns a DetectBoxQuantized into a DetectBox. The zero point is not used
/// for quant_boxes as the DetectBoxQuantized is already be shifted to
/// the zero points
pub fn dequant_detect_box<
    BOXES: Signed + PrimInt + AsPrimitive<f32>,
    SCORE: PrimInt + AsPrimitive<f32>,
>(
    detect: &DetectBoxQuantized<BOXES, SCORE>,
    quant_boxes: Quantization,
    quant_scores: Quantization,
) -> DetectBox {
    let scaled_zp = -quant_scores.scale * quant_scores.zero_point as f32;
    DetectBox {
        bbox: BoundingBox {
            xmin: quant_boxes.scale * detect.bbox.xmin.as_() * 0.5,
            ymin: quant_boxes.scale * detect.bbox.ymin.as_() * 0.5,
            xmax: quant_boxes.scale * detect.bbox.xmax.as_() * 0.5,
            ymax: quant_boxes.scale * detect.bbox.ymax.as_() * 0.5,
        },
        score: quant_scores.scale * detect.score.as_() + scaled_zp,
        label: detect.label,
    }
}

pub fn dequantize_ndarray<T: AsPrimitive<F>, D: Dimension, F: Float + 'static>(
    input: ArrayView<T, D>,
    quant: Quantization,
) -> Array<F, D>
where
    i32: num_traits::AsPrimitive<F>,
    f32: num_traits::AsPrimitive<F>,
{
    let zero_point = quant.zero_point.as_();
    let scale = quant.scale.as_();
    if zero_point != F::zero() {
        let scaled_zero = -zero_point * scale;
        input.mapv(|d| d.as_() * scale + scaled_zero)
    } else {
        input.mapv(|d| d.as_() * scale)
    }
}

pub fn dequantize_cpu<T: AsPrimitive<F>, F: Float + 'static>(
    input: &[T],
    quant: Quantization,
    output: &mut [F],
) where
    f32: num_traits::AsPrimitive<F>,
    i32: num_traits::AsPrimitive<F>,
{
    assert!(input.len() == output.len());
    let zero_point = quant.zero_point.as_();
    let scale = quant.scale.as_();
    if zero_point != F::zero() {
        let scaled_zero = -zero_point * scale; // scale * (d - zero_point) = d * scale - zero_point * scale
        input
            .iter()
            .zip(output)
            .for_each(|(d, deq)| *deq = d.as_() * scale + scaled_zero);
    } else {
        input
            .iter()
            .zip(output)
            .for_each(|(d, deq)| *deq = d.as_() * scale);
    }
}

pub fn dequantize_cpu_chunked<T: AsPrimitive<F>, F: Float + 'static>(
    input: &[T],
    quant: Quantization,
    output: &mut [F],
) where
    f32: num_traits::AsPrimitive<F>,
    i32: num_traits::AsPrimitive<F>,
{
    assert!(input.len() == output.len());
    let zero_point = quant.zero_point.as_();
    let scale = quant.scale.as_();
    if zero_point != F::zero() {
        let scaled_zero = -zero_point * scale; // scale * (d - zero_point) = d * scale - zero_point * scale

        for (d, deq) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
            unsafe {
                *deq.get_unchecked_mut(0) = d.get_unchecked(0).as_() * scale + scaled_zero;
                *deq.get_unchecked_mut(1) = d.get_unchecked(1).as_() * scale + scaled_zero;
                *deq.get_unchecked_mut(2) = d.get_unchecked(2).as_() * scale + scaled_zero;
                *deq.get_unchecked_mut(3) = d.get_unchecked(3).as_() * scale + scaled_zero;
            }
        }
        let rem = input.len() / 4 * 4;
        input[rem..]
            .iter()
            .zip(&mut output[rem..])
            .for_each(|(d, deq)| *deq = d.as_() * scale + scaled_zero);
    } else {
        for (d, deq) in input.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
            unsafe {
                *deq.get_unchecked_mut(0) = d.get_unchecked(0).as_() * scale;
                *deq.get_unchecked_mut(1) = d.get_unchecked(1).as_() * scale;
                *deq.get_unchecked_mut(2) = d.get_unchecked(2).as_() * scale;
                *deq.get_unchecked_mut(3) = d.get_unchecked(3).as_() * scale;
            }
        }
        let rem = input.len() / 4 * 4;
        input[rem..]
            .iter()
            .zip(&mut output[rem..])
            .for_each(|(d, deq)| *deq = d.as_() * scale);
    }
}

pub fn segmentation_to_mask(segmentation: ArrayView3<u8>) -> Array2<u8> {
    assert!(segmentation.shape()[2] > 0);
    if segmentation.shape()[2] == 1 {
        yolo_segmentation_to_mask(segmentation, 128)
    } else {
        modelpack_segmentation_to_mask(segmentation)
    }
}

fn arg_max<T: PartialOrd + Copy>(score: ArrayView1<T>) -> (T, usize) {
    score
        .iter()
        .enumerate()
        .fold((score[0], 0), |(max, arg_max), (ind, s)| {
            if max > *s { (max, arg_max) } else { (*s, ind) }
        })
}
#[cfg(test)]
mod tests {
    use ndarray::s;
    use ndarray_stats::DeviationExt;

    use crate::{
        modelpack::{ModelPackDetectionConfig, decode_modelpack_det, decode_modelpack_split},
        yolo::{decode_yolo_det, decode_yolo_f32, decode_yolo_segdet, decode_yolo_segdet_f32},
        *,
    };

    #[test]
    fn test_decoder_yolo_i8() {
        let score_threshold = 0.25;
        let iou_threshold = 0.7;
        let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
        let out = ndarray::Array2::from_shape_vec((84, 8400), out.to_vec()).unwrap();
        let quant = Quantization::new(0.0040811873, -123);
        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        decode_yolo_det(
            (out.view(), quant),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
        println!("output_boxes {output_boxes:?}");
        assert!(output_boxes[0].equal_within_delta(
            &DetectBox {
                bbox: BoundingBox {
                    xmin: 0.5285137,
                    ymin: 0.05305544,
                    xmax: 0.87541467,
                    ymax: 0.9998909,
                },
                score: 0.5591227,
                label: 0
            },
            1e-6
        ));

        assert!(output_boxes[1].equal_within_delta(
            &DetectBox {
                bbox: BoundingBox {
                    xmin: 0.130598,
                    ymin: 0.43260583,
                    xmax: 0.35098213,
                    ymax: 0.9958097,
                },
                score: 0.33057618,
                label: 75
            },
            1e-6
        ))
    }

    #[test]
    fn test_decoder_yolo_f32() {
        let score_threshold = 0.25;
        let iou_threshold = 0.7;
        let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
        let mut out_dequant = vec![0.0; 84 * 8400];

        let quant = Quantization::new(0.0040811873, -123);
        dequantize_cpu(out, quant, &mut out_dequant);
        let out = ndarray::Array2::from_shape_vec((84, 8400), out_dequant).unwrap();

        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        decode_yolo_f32(
            out.view(),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
        println!("output_boxes {output_boxes:?}");
        assert!(output_boxes[0].equal_within_delta(
            &DetectBox {
                bbox: BoundingBox {
                    xmin: 0.5285137,
                    ymin: 0.05305544,
                    xmax: 0.87541467,
                    ymax: 0.9998909,
                },
                score: 0.5591227,
                label: 0
            },
            1e-6
        ));

        assert!(output_boxes[1].equal_within_delta(
            &DetectBox {
                bbox: BoundingBox {
                    xmin: 0.130598,
                    ymin: 0.43260583,
                    xmax: 0.35098213,
                    ymax: 0.9958097,
                },
                score: 0.33057618,
                label: 75
            },
            1e-6
        ))
    }

    #[test]
    fn test_decoder_modelpack_u8() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let boxes = include_bytes!("../../../testdata/modelpack_boxes_1935x1x4.bin");
        let boxes = ndarray::Array2::from_shape_vec((1935, 4), boxes.to_vec()).unwrap();

        let scores = include_bytes!("../../../testdata/modelpack_scores_1935x1.bin");
        let scores = ndarray::Array2::from_shape_vec((1935, 1), scores.to_vec()).unwrap();

        let quant_boxes = Quantization::new(0.004656755365431309, 21);
        let quant_scores = Quantization::new(0.0019603664986789227, 0);

        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        decode_modelpack_det(
            (boxes.view(), quant_boxes),
            (scores.view(), quant_scores),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
        println!("output_boxes {output_boxes:?}");
        assert!(output_boxes[0].equal_within_delta(
            &DetectBox {
                bbox: BoundingBox {
                    xmin: 0.40513772,
                    ymin: 0.6379755,
                    xmax: 0.5122431,
                    ymax: 0.7730214,
                },
                score: 0.4861709,
                label: 0
            },
            1e-6
        ));
    }

    #[test]
    fn test_decoder_modelpack_split_u8() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let detect0 = include_bytes!("../../../testdata/modelpack_split_9x15x18.bin");
        let detect0 = ndarray::Array3::from_shape_vec((9, 15, 18), detect0.to_vec()).unwrap();
        let config0 = ModelPackDetectionConfig {
            anchors: vec![
                [0.36666667461395264, 0.31481480598449707],
                [0.38749998807907104, 0.4740740656852722],
                [0.5333333611488342, 0.644444465637207],
            ],
            quantization: Some(Quantization::new(0.08547406643629074, 174)),
        };

        let detect1 = include_bytes!("../../../testdata/modelpack_split_17x30x18.bin");
        let detect1 = ndarray::Array3::from_shape_vec((17, 30, 18), detect1.to_vec()).unwrap();
        let config1 = ModelPackDetectionConfig {
            anchors: vec![
                [0.13750000298023224, 0.2074074000120163],
                [0.2541666626930237, 0.21481481194496155],
                [0.23125000298023224, 0.35185185074806213],
            ],
            quantization: Some(Quantization::new(0.09929127991199493, 183)),
        };

        let mut output_boxes: Vec<_> = Vec::with_capacity(2);
        decode_modelpack_split(
            &[detect0.view(), detect1.view()],
            &[config0, config1],
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );

        println!("output_boxes {output_boxes:?}");
        assert!(output_boxes[0].equal_within_delta(
            &DetectBox {
                bbox: BoundingBox {
                    xmin: 0.43171933,
                    ymin: 0.68243736,
                    xmax: 0.5626645,
                    ymax: 0.808863,
                },
                score: 0.99240804,
                label: 0
            },
            1e-6
        ));
    }

    #[test]
    fn test_decoder_parse_config_modelpack_split_u8() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let detect0 = include_bytes!("../../../testdata/modelpack_split_9x15x18.bin");
        let detect0 = ndarray::Array4::from_shape_vec((1, 9, 15, 18), detect0.to_vec()).unwrap();

        let detect1 = include_bytes!("../../../testdata/modelpack_split_17x30x18.bin");
        let detect1 = ndarray::Array4::from_shape_vec((1, 17, 30, 18), detect1.to_vec()).unwrap();

        let decoder = DecoderBuilder::default()
            .with_config_yaml_str(
                include_str!("../../../testdata/modelpack_split.yaml").to_string(),
            )
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .build()
            .unwrap();

        let mut output_boxes: Vec<_> = Vec::with_capacity(10);
        let mut output_masks: Vec<_> = Vec::with_capacity(10);
        decoder
            .decode_quantized(
                &[
                    ArrayViewDQuantized::from(detect1.view().into_dyn()),
                    ArrayViewDQuantized::from(detect0.view().into_dyn()),
                ],
                &mut output_boxes,
                &mut output_masks,
            )
            .unwrap();

        println!("output_boxes {output_boxes:?}");
        assert!(output_boxes[0].equal_within_delta(
            &DetectBox {
                bbox: BoundingBox {
                    xmin: 0.43171933,
                    ymin: 0.68243736,
                    xmax: 0.5626645,
                    ymax: 0.808863,
                },
                score: 0.99240804,
                label: 0
            },
            1e-6
        ));
    }

    #[test]
    fn test_dequant_chunked() {
        let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
        let mut out_dequant = vec![0.0; 84 * 8400];
        let mut out_dequant_simd = vec![0.0; 84 * 8400];
        let quant = Quantization::new(0.0040811873, -123);
        dequantize_cpu(out, quant, &mut out_dequant);

        dequantize_cpu_chunked(out, quant, &mut out_dequant_simd);

        assert_eq!(out_dequant, out_dequant_simd);
    }

    #[test]
    fn test_decoder_masks() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let boxes = include_bytes!("../../../testdata/yolov8_boxes_116x8400.bin");
        let boxes = unsafe { std::slice::from_raw_parts(boxes.as_ptr() as *const i8, boxes.len()) };
        let boxes = ndarray::Array2::from_shape_vec((116, 8400), boxes.to_vec()).unwrap();
        let quant_boxes = Quantization::new(0.021287761628627777, 31);

        let protos = include_bytes!("../../../testdata/yolov8_protos_160x160x32.bin");
        let protos =
            unsafe { std::slice::from_raw_parts(protos.as_ptr() as *const i8, protos.len()) };
        let protos = ndarray::Array3::from_shape_vec((160, 160, 32), protos.to_vec()).unwrap();
        let quant_protos = Quantization::new(0.02491161972284317, -117);
        let protos = dequantize_ndarray(protos.view(), quant_protos);
        let seg = dequantize_ndarray(boxes.view(), quant_boxes);
        let mut output_boxes: Vec<_> = Vec::with_capacity(10);
        let mut output_masks: Vec<_> = Vec::with_capacity(10);
        decode_yolo_segdet_f32(
            seg.view(),
            protos.view(),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
            &mut output_masks,
        );
        assert_eq!(output_boxes.len(), 2);
        assert_eq!(output_boxes.len(), output_masks.len());

        for (b, m) in output_boxes.iter().zip(&output_masks) {
            assert!(b.bbox.xmin >= m.xmin);
            assert!(b.bbox.ymin >= m.ymin);
            assert!(b.bbox.xmax >= m.xmax);
            assert!(b.bbox.ymax >= m.ymax);
        }
        assert!(output_boxes[0].equal_within_delta(
            &DetectBox {
                bbox: BoundingBox {
                    xmin: 0.08515105,
                    ymin: 0.7131401,
                    xmax: 0.29802868,
                    ymax: 0.8195788,
                },
                score: 0.91537374,
                label: 23
            },
            1.0 / 160.0, // wider range because mask will expand the box
        ));

        assert!(output_boxes[1].equal_within_delta(
            &DetectBox {
                bbox: BoundingBox {
                    xmin: 0.59605736,
                    ymin: 0.25545314,
                    xmax: 0.93666154,
                    ymax: 0.72378385,
                },
                score: 0.91537374,
                label: 23
            },
            1.0 / 160.0, // wider range because mask will expand the box
        ));

        let full_mask = include_bytes!("../../../testdata/yolov8_mask_results.bin");
        let full_mask = ndarray::Array2::from_shape_vec((160, 160), full_mask.to_vec()).unwrap();

        let cropped_mask = full_mask.slice(ndarray::s![
            (output_masks[1].ymin * 160.0) as usize..(output_masks[1].ymax * 160.0) as usize,
            (output_masks[1].xmin * 160.0) as usize..(output_masks[1].xmax * 160.0) as usize,
        ]);

        assert_eq!(
            cropped_mask,
            segmentation_to_mask(output_masks[1].segmentation.view())
        );
    }

    #[test]
    fn test_decoder_masks_i8() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let boxes = include_bytes!("../../../testdata/yolov8_boxes_116x8400.bin");
        let boxes = unsafe { std::slice::from_raw_parts(boxes.as_ptr() as *const i8, boxes.len()) };
        let boxes = ndarray::Array2::from_shape_vec((116, 8400), boxes.to_vec()).unwrap();
        let quant_boxes = Quantization::new(0.021287761628627777, 31);

        let protos = include_bytes!("../../../testdata/yolov8_protos_160x160x32.bin");
        let protos =
            unsafe { std::slice::from_raw_parts(protos.as_ptr() as *const i8, protos.len()) };
        let protos = ndarray::Array3::from_shape_vec((160, 160, 32), protos.to_vec()).unwrap();
        let quant_protos = Quantization::new(0.02491161972284317, -117);
        let mut output_boxes: Vec<_> = Vec::with_capacity(500);
        let mut output_masks: Vec<_> = Vec::with_capacity(500);

        decode_yolo_segdet(
            (boxes.view(), quant_boxes),
            (protos.view(), quant_protos),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
            &mut output_masks,
        );

        let protos = dequantize_ndarray(protos.view(), quant_protos);
        let seg = dequantize_ndarray(boxes.view(), quant_boxes);
        let mut output_boxes_f32: Vec<_> = Vec::with_capacity(500);
        let mut output_masks_f32: Vec<_> = Vec::with_capacity(500);
        decode_yolo_segdet_f32(
            seg.view(),
            protos.view(),
            score_threshold,
            iou_threshold,
            &mut output_boxes_f32,
            &mut output_masks_f32,
        );

        assert_eq!(output_boxes.len(), output_boxes_f32.len());
        assert_eq!(output_masks.len(), output_masks_f32.len());

        for (b_i8, b_f32) in output_boxes.iter().zip(&output_boxes_f32) {
            assert!(
                b_i8.equal_within_delta(b_f32, 1e-6),
                "{b_i8:?} is not equal to {b_f32:?}"
            );
        }

        for (m_i8, m_f32) in output_masks.iter().zip(&output_masks_f32) {
            assert_eq!(
                [m_i8.xmin, m_i8.ymin, m_i8.xmax, m_i8.ymax],
                [m_f32.xmin, m_f32.ymin, m_f32.xmax, m_f32.ymax],
            );
            assert_eq!(m_i8.segmentation.shape(), m_f32.segmentation.shape());
            let mask_i8 = m_i8.segmentation.map(|x| *x as i32);
            let mask_f32 = m_f32.segmentation.map(|x| *x as i32);
            let diff = &mask_i8 - &mask_f32;
            assert!(
                !diff.iter().any(|x| x.abs() > 1),
                "Difference between mask i8 and mask f32 is greater than 1: {:#?}",
                diff
            );
            let mean_sq_err = mask_i8.mean_sq_err(&mask_f32).unwrap();
            assert!(
                mean_sq_err < 1e-2,
                "Mean Square Error between masks was greater than 1%: {:.2}%",
                mean_sq_err * 100.0
            );
        }
    }

    #[test]
    fn test_decoder_masks_config_mixed() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let boxes = include_bytes!("../../../testdata/yolov8_boxes_116x8400.bin");
        let boxes = unsafe { std::slice::from_raw_parts(boxes.as_ptr() as *const i8, boxes.len()) };
        let boxes: Vec<_> = boxes.iter().map(|x| *x as i16).collect();
        let boxes = ndarray::Array3::from_shape_vec((1, 116, 8400), boxes).unwrap();

        let quant_boxes = Quantization::new(0.021287761628627777, 31);

        let protos = include_bytes!("../../../testdata/yolov8_protos_160x160x32.bin");
        let protos =
            unsafe { std::slice::from_raw_parts(protos.as_ptr() as *const i8, protos.len()) };
        let protos = ndarray::Array4::from_shape_vec((1, 160, 160, 32), protos.to_vec()).unwrap();
        let quant_protos = Quantization::new(0.02491161972284317, -117);

        let decoder = DecoderBuilder::default()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    channels_first: false,
                    decoder: configs::DecoderType::Yolov8,
                    quantization: Some(QuantTuple(0.021287761628627777, 31)),
                    shape: vec![1, 4, 8400],
                },
                configs::Scores {
                    channels_first: false,
                    decoder: configs::DecoderType::Yolov8,
                    quantization: Some(QuantTuple(0.021287761628627777, 31)),
                    shape: vec![1, 80, 8400],
                },
                configs::MaskCoefficients {
                    channels_first: false,
                    decoder: configs::DecoderType::Yolov8,
                    quantization: Some(QuantTuple(0.021287761628627777, 31)),
                    shape: vec![1, 32, 8400],
                },
                configs::Protos {
                    channels_first: false,
                    decoder: configs::DecoderType::Yolov8,
                    quantization: Some(QuantTuple(0.02491161972284317, -117)),
                    shape: vec![1, 160, 160, 32],
                },
            )
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .build()
            .unwrap();

        let mut output_boxes: Vec<_> = Vec::with_capacity(500);
        let mut output_masks: Vec<_> = Vec::with_capacity(500);

        decoder
            .decode_quantized(
                &[
                    boxes.slice(s![.., ..4, ..]).into_dyn().into(),
                    boxes.slice(s![.., 4..84, ..]).into_dyn().into(),
                    boxes.slice(s![.., 84.., ..]).into_dyn().into(),
                    protos.view().into_dyn().into(),
                ],
                &mut output_boxes,
                &mut output_masks,
            )
            .unwrap();

        let protos = dequantize_ndarray(protos.view(), quant_protos);
        let seg = dequantize_ndarray(boxes.view(), quant_boxes);
        let mut output_boxes_f32: Vec<_> = Vec::with_capacity(500);
        let mut output_masks_f32: Vec<_> = Vec::with_capacity(500);
        decode_yolo_segdet_f32(
            seg.slice(s![0, .., ..]),
            protos.slice(s![0, .., .., ..]),
            score_threshold,
            iou_threshold,
            &mut output_boxes_f32,
            &mut output_masks_f32,
        );

        assert_eq!(output_boxes.len(), output_boxes_f32.len());
        assert_eq!(output_masks.len(), output_masks_f32.len());

        for (b_i8, b_f32) in output_boxes.iter().zip(&output_boxes_f32) {
            assert!(
                b_i8.equal_within_delta(b_f32, 1e-6),
                "{b_i8:?} is not equal to {b_f32:?}"
            );
        }

        for (m_i8, m_f32) in output_masks.iter().zip(&output_masks_f32) {
            assert_eq!(
                [m_i8.xmin, m_i8.ymin, m_i8.xmax, m_i8.ymax],
                [m_f32.xmin, m_f32.ymin, m_f32.xmax, m_f32.ymax],
            );
            assert_eq!(m_i8.segmentation.shape(), m_f32.segmentation.shape());
            let mask_i8 = m_i8.segmentation.map(|x| *x as i32);
            let mask_f32 = m_f32.segmentation.map(|x| *x as i32);
            let diff = &mask_i8 - &mask_f32;
            assert!(
                !diff.iter().any(|x| x.abs() > 1),
                "Difference between mask i8 and mask f32 is greater than 1: {:#?}",
                diff
            );
            let mean_sq_err = mask_i8.mean_sq_err(&mask_f32).unwrap();
            assert!(
                mean_sq_err < 1e-2,
                "Mean Square Error between masks was greater than 1%: {:.2}%",
                mean_sq_err * 100.0
            );
        }
    }
}
