// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

/*!
## EdgeFirst HAL - Decoders
This crate provides decoding utilities for YOLOobject detection and segmentation models, and ModelPack detection and segmentation models.
It supports both floating-point and quantized model outputs, allowing for efficient processing on edge devices. The crate includes functions
for efficient post-processing model outputs into usable detection boxes and segmentation masks, as well as utilities for dequantizing model outputs..

For general usage, use the `Decoder` struct which provides functions for decoding various model outputs based on the model configuration.
If you already know the model type and output formats, you can use the lower-level functions directly from the `yolo` and `modelpack` modules.


### Quick Example
```rust
# use edgefirst_decoder::{ DecoderBuilder, DecoderResult, configs };
# fn main() -> DecoderResult<()> {
// Create a decoder for a YOLOv8 model with quantized int8 output with 0.25 score threshold and 0.7 IOU threshold
let decoder = DecoderBuilder::new()
    .with_config_yolo_det(configs::Detection {
        anchors: None,
        decoder: configs::DecoderType::Ultralytics,
        quantization: Some(configs::QuantTuple(0.012345, 26)),
        shape: vec![1, 84, 8400],
        dshape: Vec::new(),
    })
    .with_score_threshold(0.25)
    .with_iou_threshold(0.7)
    .build()?;

// Get the model output from the model. Here we load it from a test data file for demonstration purposes.
let model_output: Vec<i8> = include_bytes!("../../../testdata/yolov8s_80_classes.bin")
    .iter()
    .map(|b| *b as i8)
    .collect();
let model_output_array = ndarray::Array3::from_shape_vec((1, 84, 8400), model_output)?;

// THe capacity is used to determine the maximum number of detections to decode.
let mut output_boxes: Vec<_> = Vec::with_capacity(10);
let mut output_masks: Vec<_> = Vec::with_capacity(10);

// Decode the quantized model output into detection boxes and segmentation masks
// Because this model is a detection-only model, the `output_masks` vector will remain empty.
decoder.decode_quantized(&[model_output_array.view().into()], &mut output_boxes, &mut output_masks)?;
# Ok(())
# }
```

# Overview

The primary components of this crate are:
- `Decoder`/`DecoderBuilder` struct: Provides high-level functions to decode model outputs based on the model configuration.
- `yolo` module: Contains functions specific to decoding YOLO model outputs.
- `modelpack` module: Contains functions specific to decoding ModelPack model outputs.

The `Decoder` supports both floating-point and quantized model outputs, allowing for efficient processing on edge devices.
It also supports mixed integer types for quantized outputs, such as when one output tensor is int8 and another is uint8.
When decoding quantized outputs, the appropriate quantization parameters must be provided for each output tensor.
If the integer types used in the model output is not supported by the decoder, the user can manually dequantize the model outputs using
the `dequantize` functions provided in this crate, and then use the floating-point decoding functions. However, it is recommended
to not dequantize the model outputs manually before passing them to the decoder, as the quantized decoder functions are optimized for performance.

The `yolo` and `modelpack` modules provide lower-level functions for decoding model outputs directly,
which can be used if the model type and output formats are known in advance.


*/
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use ndarray::{Array, Array2, Array3, ArrayView, ArrayView1, ArrayView3, Dimension};
use num_traits::{AsPrimitive, Float, PrimInt};

pub mod byte;
pub mod error;
pub mod float;
pub mod modelpack;
pub mod yolo;

mod decoder;
pub use decoder::*;

pub use error::{DecoderError, DecoderResult};

use crate::{
    decoder::configs::QuantTuple, modelpack::modelpack_segmentation_to_mask,
    yolo::yolo_segmentation_to_mask,
};

/// Trait to convert bounding box formats to XYXY float format
pub trait BBoxTypeTrait {
    /// Converts the bbox into XYXY float format.
    fn to_xyxy_float<A: Float + 'static, B: AsPrimitive<A>>(input: &[B; 4]) -> [A; 4];

    /// Converts the bbox into XYXY float format.
    fn to_xyxy_dequant<A: Float + 'static, B: AsPrimitive<A>>(
        input: &[B; 4],
        quant: Quantization,
    ) -> [A; 4]
    where
        f32: AsPrimitive<A>,
        i32: AsPrimitive<A>;

    /// Converts the bbox into XYXY float format.
    ///
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::{BBoxTypeTrait, XYWH};
    /// # use ndarray::array;
    /// let arr = array![10.0_f32, 20.0, 20.0, 20.0];
    /// let xyxy: [f32; 4] = XYWH::ndarray_to_xyxy_float(arr.view());
    /// assert_eq!(xyxy, [0.0_f32, 10.0, 20.0, 30.0]);
    /// ```
    #[inline(always)]
    fn ndarray_to_xyxy_float<A: Float + 'static, B: AsPrimitive<A>>(
        input: ArrayView1<B>,
    ) -> [A; 4] {
        Self::to_xyxy_float(&[input[0], input[1], input[2], input[3]])
    }

    #[inline(always)]
    /// Converts the bbox into XYXY float format.
    fn ndarray_to_xyxy_dequant<A: Float + 'static, B: AsPrimitive<A>>(
        input: ArrayView1<B>,
        quant: Quantization,
    ) -> [A; 4]
    where
        f32: AsPrimitive<A>,
        i32: AsPrimitive<A>,
    {
        Self::to_xyxy_dequant(&[input[0], input[1], input[2], input[3]], quant)
    }
}

/// Converts XYXY bounding boxes to XYXY
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XYXY {}

impl BBoxTypeTrait for XYXY {
    fn to_xyxy_float<A: Float + 'static, B: AsPrimitive<A>>(input: &[B; 4]) -> [A; 4] {
        input.map(|b| b.as_())
    }

    fn to_xyxy_dequant<A: Float + 'static, B: AsPrimitive<A>>(
        input: &[B; 4],
        quant: Quantization,
    ) -> [A; 4]
    where
        f32: AsPrimitive<A>,
        i32: AsPrimitive<A>,
    {
        let scale = quant.scale.as_();
        let zp = quant.zero_point.as_();
        input.map(|b| (b.as_() - zp) * scale)
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

/// Converts XYWH bounding boxes to XYXY. The XY values are the center of the
/// box
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XYWH {}

impl BBoxTypeTrait for XYWH {
    #[inline(always)]
    fn to_xyxy_float<A: Float + 'static, B: AsPrimitive<A>>(input: &[B; 4]) -> [A; 4] {
        let half = A::one() / (A::one() + A::one());
        [
            (input[0].as_()) - (input[2].as_() * half),
            (input[1].as_()) - (input[3].as_() * half),
            (input[0].as_()) + (input[2].as_() * half),
            (input[1].as_()) + (input[3].as_() * half),
        ]
    }

    #[inline(always)]
    fn to_xyxy_dequant<A: Float + 'static, B: AsPrimitive<A>>(
        input: &[B; 4],
        quant: Quantization,
    ) -> [A; 4]
    where
        f32: AsPrimitive<A>,
        i32: AsPrimitive<A>,
    {
        let scale = quant.scale.as_();
        let half_scale = (quant.scale * 0.5).as_();
        let zp = quant.zero_point.as_();
        let [x, y, w, h] = [
            (input[0].as_() - zp) * scale,
            (input[1].as_() - zp) * scale,
            (input[2].as_() - zp) * half_scale,
            (input[3].as_() - zp) * half_scale,
        ];

        [x - w, y - h, x + w, y + h]
    }

    #[inline(always)]
    fn ndarray_to_xyxy_float<A: Float + 'static, B: AsPrimitive<A>>(
        input: ArrayView1<B>,
    ) -> [A; 4] {
        let half = A::one() / (A::one() + A::one());
        [
            (input[0].as_()) - (input[2].as_() * half),
            (input[1].as_()) - (input[3].as_() * half),
            (input[0].as_()) + (input[2].as_() * half),
            (input[1].as_()) + (input[3].as_() * half),
        ]
    }
}

/// Describes the quantization parameters for a tensor
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantization {
    pub scale: f32,
    pub zero_point: i32,
}

impl Quantization {
    /// Creates a new Quantization struct
    /// # Examples
    /// ```
    /// # use edgefirst_decoder::Quantization;
    /// let quant = Quantization::new(0.1, -128);
    /// assert_eq!(quant.scale, 0.1);
    /// assert_eq!(quant.zero_point, -128);
    /// ```
    pub fn new(scale: f32, zero_point: i32) -> Self {
        Self { scale, zero_point }
    }
}

impl From<QuantTuple> for Quantization {
    /// Creates a new Quantization struct from a QuantTuple
    /// # Examples
    /// ```
    /// # use edgefirst_decoder::Quantization;
    /// # use edgefirst_decoder::configs::QuantTuple;
    /// let quant_tuple = QuantTuple(0.1_f32, -128_i32);
    /// let quant = Quantization::from(quant_tuple);
    /// assert_eq!(quant.scale, 0.1);
    /// assert_eq!(quant.zero_point, -128);
    /// ```
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
    /// Creates a new Quantization struct from a tuple
    /// # Examples
    /// ```
    /// # use edgefirst_decoder::Quantization;
    /// let quant = Quantization::from((0.1_f64, -128_i64));
    /// assert_eq!(quant.scale, 0.1);
    /// assert_eq!(quant.zero_point, -128);
    /// ```
    fn from((scale, zp): (S, Z)) -> Quantization {
        Self {
            scale: scale.as_(),
            zero_point: zp.as_(),
        }
    }
}

impl Default for Quantization {
    /// Creates a default Quantization struct with scale 1.0 and zero_point 0
    /// # Examples
    /// ```rust
    /// # use edgefirst_decoder::Quantization;
    /// let quant = Quantization::default();
    /// assert_eq!(quant.scale, 1.0);
    /// assert_eq!(quant.zero_point, 0);
    /// ```
    fn default() -> Self {
        Self {
            scale: 1.0,
            zero_point: 0,
        }
    }
}

/// A detection box with f32 bbox and score
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DetectBox {
    pub bbox: BoundingBox,
    /// model-specific score for this detection, higher implies more confidence
    pub score: f32,
    /// label index for this detection
    pub label: usize,
}

impl edgefirst_tracker::DetectionBox for DetectBox {
    fn bbox(&self) -> [f32; 4] {
        self.bbox.into()
    }

    fn score(&self) -> f32 {
        self.score
    }

    fn label(&self) -> usize {
        self.label
    }
}

/// A bounding box with f32 coordinates in XYXY format
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
    /// Creates a new BoundingBox from the given coordinates
    pub fn new(xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> Self {
        Self {
            xmin,
            ymin,
            xmax,
            ymax,
        }
    }

    /// Transforms BoundingBox so that `xmin <= xmax` and `ymin <= ymax`
    ///
    /// ```
    /// # use edgefirst_decoder::BoundingBox;
    /// let bbox = BoundingBox::new(0.8, 0.6, 0.4, 0.2);
    /// let canonical_bbox = bbox.to_canonical();
    /// assert_eq!(canonical_bbox, BoundingBox::new(0.4, 0.2, 0.8, 0.6));
    /// ```
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
    /// Converts a BoundingBox into an array of 4 f32 values in xmin, ymin,
    /// xmax, ymax order
    /// # Examples
    /// ```
    /// # use edgefirst_decoder::BoundingBox;
    /// let bbox = BoundingBox {
    ///     xmin: 0.1,
    ///     ymin: 0.2,
    ///     xmax: 0.3,
    ///     ymax: 0.4,
    /// };
    /// let arr: [f32; 4] = bbox.into();
    /// assert_eq!(arr, [0.1, 0.2, 0.3, 0.4]);
    /// ```
    fn from(b: BoundingBox) -> Self {
        [b.xmin, b.ymin, b.xmax, b.ymax]
    }
}

impl From<[f32; 4]> for BoundingBox {
    // Converts an array of 4 f32 values in xmin, ymin, xmax, ymax order into a
    // BoundingBox
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
    /// Returns true if one detect box is equal to another detect box, within
    /// the given `eps`
    ///
    /// # Examples
    /// ```
    /// # use edgefirst_decoder::DetectBox;
    /// let box1 = DetectBox {
    ///     bbox: edgefirst_decoder::BoundingBox {
    ///         xmin: 0.1,
    ///         ymin: 0.2,
    ///         xmax: 0.3,
    ///         ymax: 0.4,
    ///     },
    ///     score: 0.5,
    ///     label: 1,
    /// };
    /// let box2 = DetectBox {
    ///     bbox: edgefirst_decoder::BoundingBox {
    ///         xmin: 0.101,
    ///         ymin: 0.199,
    ///         xmax: 0.301,
    ///         ymax: 0.399,
    ///     },
    ///     score: 0.510,
    ///     label: 1,
    /// };
    /// assert!(box1.equal_within_delta(&box2, 0.011));
    /// ```
    pub fn equal_within_delta(&self, rhs: &DetectBox, eps: f32) -> bool {
        let eq_delta = |a: f32, b: f32| (a - b).abs() <= eps;
        self.label == rhs.label
            && eq_delta(self.score, rhs.score)
            && eq_delta(self.bbox.xmin, rhs.bbox.xmin)
            && eq_delta(self.bbox.ymin, rhs.bbox.ymin)
            && eq_delta(self.bbox.xmax, rhs.bbox.xmax)
            && eq_delta(self.bbox.ymax, rhs.bbox.ymax)
    }
}

/// A segmentation result with a segmentation mask, and a normalized bounding
/// box representing the area that the segmentation mask covers
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

/// Turns a DetectBoxQuantized into a DetectBox by dequantizing the score.
///
///  # Examples
/// ```
/// # use edgefirst_decoder::{BoundingBox, DetectBoxQuantized, Quantization, dequant_detect_box};
/// let quant = Quantization::new(0.1, -128);
/// let bbox = BoundingBox::new(0.1, 0.2, 0.3, 0.4);
/// let detect_quant = DetectBoxQuantized {
///     bbox,
///     score: 100_i8,
///     label: 1,
/// };
/// let detect = dequant_detect_box(&detect_quant, quant);
/// assert_eq!(detect.score, 0.1 * 100.0 + 12.8);
/// assert_eq!(detect.label, 1);
/// assert_eq!(detect.bbox, bbox);
/// ```
pub fn dequant_detect_box<SCORE: PrimInt + AsPrimitive<f32>>(
    detect: &DetectBoxQuantized<SCORE>,
    quant_scores: Quantization,
) -> DetectBox {
    let scaled_zp = -quant_scores.scale * quant_scores.zero_point as f32;
    DetectBox {
        bbox: detect.bbox,
        score: quant_scores.scale * detect.score.as_() + scaled_zp,
        label: detect.label,
    }
}
/// A detection box with a f32 bbox and quantized score
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectBoxQuantized<
    // BOX: Signed + PrimInt + AsPrimitive<f32>,
    SCORE: PrimInt + AsPrimitive<f32>,
> {
    // pub bbox: BoundingBoxQuantized<BOX>,
    pub bbox: BoundingBox,
    /// model-specific score for this detection, higher implies more
    /// confidence.
    pub score: SCORE,
    /// label index for this detect
    pub label: usize,
}

/// Dequantizes an ndarray from quantized values to f32 values using the given
/// quantization parameters
///
/// # Examples
/// ```
/// # use edgefirst_decoder::{dequantize_ndarray, Quantization};
/// let quant = Quantization::new(0.1, -128);
/// let input: Vec<i8> = vec![0, 127, -128, 64];
/// let input_array = ndarray::Array1::from(input);
/// let output_array: ndarray::Array1<f32> = dequantize_ndarray(input_array.view(), quant);
/// assert_eq!(output_array, ndarray::array![12.8, 25.5, 0.0, 19.2]);
/// ```
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

/// Dequantizes a slice from quantized values to float values using the given
/// quantization parameters
///
/// # Examples
/// ```
/// # use edgefirst_decoder::{dequantize_cpu, Quantization};
/// let quant = Quantization::new(0.1, -128);
/// let input: Vec<i8> = vec![0, 127, -128, 64];
/// let mut output: Vec<f32> = vec![0.0; input.len()];
/// dequantize_cpu(&input, quant, &mut output);
/// assert_eq!(output, vec![12.8, 25.5, 0.0, 19.2]);
/// ```
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

/// Dequantizes a slice from quantized values to float values using the given
/// quantization parameters, using chunked processing. This is around 5% faster
/// than `dequantize_cpu` for large slices.
///
/// # Examples
/// ```
/// # use edgefirst_decoder::{dequantize_cpu_chunked, Quantization};
/// let quant = Quantization::new(0.1, -128);
/// let input: Vec<i8> = vec![0, 127, -128, 64];
/// let mut output: Vec<f32> = vec![0.0; input.len()];
/// dequantize_cpu_chunked(&input, quant, &mut output);
/// assert_eq!(output, vec![12.8, 25.5, 0.0, 19.2]);
/// ```
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

    let input = input.as_chunks::<4>();
    let output = output.as_chunks_mut::<4>();

    if zero_point != F::zero() {
        let scaled_zero = -zero_point * scale; // scale * (d - zero_point) = d * scale - zero_point * scale

        input
            .0
            .iter()
            .zip(output.0)
            .for_each(|(d, deq)| *deq = d.map(|d| d.as_() * scale + scaled_zero));
        input
            .1
            .iter()
            .zip(output.1)
            .for_each(|(d, deq)| *deq = d.as_() * scale + scaled_zero);
    } else {
        input
            .0
            .iter()
            .zip(output.0)
            .for_each(|(d, deq)| *deq = d.map(|d| d.as_() * scale));
        input
            .1
            .iter()
            .zip(output.1)
            .for_each(|(d, deq)| *deq = d.as_() * scale);
    }
}

/// Converts a segmentation tensor into a 2D mask
/// If the last dimension of the segmentation tensor is 1, values equal or
/// above 128 are considered objects. Otherwise the object is the argmax index
/// # Examples
/// ```
/// # use edgefirst_decoder::segmentation_to_mask;
/// let segmentation =
///     ndarray::Array3::<u8>::from_shape_vec((2, 2, 1), vec![0, 255, 128, 127]).unwrap();
/// let mask = segmentation_to_mask(segmentation.view());
/// assert_eq!(mask, ndarray::array![[0, 1], [1, 0]]);
/// ```
pub fn segmentation_to_mask(segmentation: ArrayView3<u8>) -> Array2<u8> {
    assert!(segmentation.shape()[2] > 0);
    if segmentation.shape()[2] == 1 {
        yolo_segmentation_to_mask(segmentation, 128)
    } else {
        modelpack_segmentation_to_mask(segmentation)
    }
}

/// Returns the maximum value and its index from a 1D array
fn arg_max<T: PartialOrd + Copy>(score: ArrayView1<T>) -> (T, usize) {
    score
        .iter()
        .enumerate()
        .fold((score[0], 0), |(max, arg_max), (ind, s)| {
            if max > *s { (max, arg_max) } else { (*s, ind) }
        })
}
#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod decoder_tests {
    #![allow(clippy::excessive_precision)]
    use crate::{
        configs::{DecoderType, DimName, Protos},
        modelpack::{decode_modelpack_det, decode_modelpack_split_quant},
        yolo::{
            decode_yolo_det, decode_yolo_det_float, decode_yolo_segdet_float,
            decode_yolo_segdet_quant,
        },
        *,
    };
    use ndarray::{Array4, array, s};
    use ndarray_stats::DeviationExt;

    fn compare_outputs(
        boxes: (&[DetectBox], &[DetectBox]),
        masks: (&[Segmentation], &[Segmentation]),
    ) {
        let (boxes0, boxes1) = boxes;
        let (masks0, masks1) = masks;

        assert_eq!(boxes0.len(), boxes1.len());
        assert_eq!(masks0.len(), masks1.len());

        for (b_i8, b_f32) in boxes0.iter().zip(boxes1) {
            assert!(
                b_i8.equal_within_delta(b_f32, 1e-6),
                "{b_i8:?} is not equal to {b_f32:?}"
            );
        }

        for (m_i8, m_f32) in masks0.iter().zip(masks1) {
            assert_eq!(
                [m_i8.xmin, m_i8.ymin, m_i8.xmax, m_i8.ymax],
                [m_f32.xmin, m_f32.ymin, m_f32.xmax, m_f32.ymax],
            );
            assert_eq!(m_i8.segmentation.shape(), m_f32.segmentation.shape());
            let mask_i8 = m_i8.segmentation.map(|x| *x as i32);
            let mask_f32 = m_f32.segmentation.map(|x| *x as i32);
            let diff = &mask_i8 - &mask_f32;
            for x in 0..diff.shape()[0] {
                for y in 0..diff.shape()[1] {
                    for z in 0..diff.shape()[2] {
                        let val = diff[[x, y, z]];
                        assert!(
                            val.abs() <= 1,
                            "Difference between mask0 and mask1 is greater than 1 at ({}, {}, {}): {}",
                            x,
                            y,
                            z,
                            val
                        );
                    }
                }
            }
            let mean_sq_err = mask_i8.mean_sq_err(&mask_f32).unwrap();
            assert!(
                mean_sq_err < 1e-2,
                "Mean Square Error between masks was greater than 1%: {:.2}%",
                mean_sq_err * 100.0
            );
        }
    }

    #[test]
    fn test_decoder_modelpack() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let boxes = include_bytes!("../../../testdata/modelpack_boxes_1935x1x4.bin");
        let boxes = ndarray::Array4::from_shape_vec((1, 1935, 1, 4), boxes.to_vec()).unwrap();

        let scores = include_bytes!("../../../testdata/modelpack_scores_1935x1.bin");
        let scores = ndarray::Array3::from_shape_vec((1, 1935, 1), scores.to_vec()).unwrap();

        let quant_boxes = (0.004656755365431309, 21).into();
        let quant_scores = (0.0019603664986789227, 0).into();

        let decoder = DecoderBuilder::default()
            .with_config_modelpack_det(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: Some(quant_boxes),
                    shape: vec![1, 1935, 1, 4],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 1935),
                        (DimName::Padding, 1),
                        (DimName::NumFeatures, 4),
                    ],
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: Some(quant_scores),
                    shape: vec![1, 1935, 1],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 1935),
                        (DimName::NumClasses, 1),
                    ],
                },
            )
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .build()
            .unwrap();

        let quant_boxes = quant_boxes.into();
        let quant_scores = quant_scores.into();

        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        decode_modelpack_det(
            (boxes.slice(s![0, .., 0, ..]), quant_boxes),
            (scores.slice(s![0, .., ..]), quant_scores),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
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

        let mut output_boxes1 = Vec::with_capacity(50);
        let mut output_masks1 = Vec::with_capacity(50);
        decoder
            .decode_quantized(
                &[boxes.view().into(), scores.view().into()],
                &mut output_boxes1,
                &mut output_masks1,
            )
            .unwrap();

        let mut output_boxes_float = Vec::with_capacity(50);
        let mut output_masks_float = Vec::with_capacity(50);
        let boxes = dequantize_ndarray(boxes.view(), quant_boxes);
        let scores = dequantize_ndarray(scores.view(), quant_scores);

        decoder
            .decode_float::<f32>(
                &[boxes.view().into_dyn(), scores.view().into_dyn()],
                &mut output_boxes_float,
                &mut output_masks_float,
            )
            .unwrap();

        compare_outputs((&output_boxes, &output_boxes1), (&[], &output_masks1));
        compare_outputs(
            (&output_boxes, &output_boxes_float),
            (&[], &output_masks_float),
        );
    }

    #[test]
    fn test_decoder_modelpack_split_u8() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let detect0 = include_bytes!("../../../testdata/modelpack_split_9x15x18.bin");
        let detect0 = ndarray::Array4::from_shape_vec((1, 9, 15, 18), detect0.to_vec()).unwrap();

        let detect1 = include_bytes!("../../../testdata/modelpack_split_17x30x18.bin");
        let detect1 = ndarray::Array4::from_shape_vec((1, 17, 30, 18), detect1.to_vec()).unwrap();

        let quant0 = (0.08547406643629074, 174).into();
        let quant1 = (0.09929127991199493, 183).into();
        let anchors0 = vec![
            [0.36666667461395264, 0.31481480598449707],
            [0.38749998807907104, 0.4740740656852722],
            [0.5333333611488342, 0.644444465637207],
        ];
        let anchors1 = vec![
            [0.13750000298023224, 0.2074074000120163],
            [0.2541666626930237, 0.21481481194496155],
            [0.23125000298023224, 0.35185185074806213],
        ];

        let detect_config0 = configs::Detection {
            decoder: DecoderType::ModelPack,
            shape: vec![1, 9, 15, 18],
            anchors: Some(anchors0.clone()),
            quantization: Some(quant0),
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::Height, 9),
                (DimName::Width, 15),
                (DimName::NumAnchorsXFeatures, 18),
            ],
        };

        let detect_config1 = configs::Detection {
            decoder: DecoderType::ModelPack,
            shape: vec![1, 17, 30, 18],
            anchors: Some(anchors1.clone()),
            quantization: Some(quant1),
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::Height, 17),
                (DimName::Width, 30),
                (DimName::NumAnchorsXFeatures, 18),
            ],
        };

        let config0 = (&detect_config0).try_into().unwrap();
        let config1 = (&detect_config1).try_into().unwrap();

        let decoder = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![detect_config1, detect_config0])
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .build()
            .unwrap();

        let quant0 = quant0.into();
        let quant1 = quant1.into();

        let mut output_boxes: Vec<_> = Vec::with_capacity(10);
        decode_modelpack_split_quant(
            &[
                detect0.slice(s![0, .., .., ..]),
                detect1.slice(s![0, .., .., ..]),
            ],
            &[config0, config1],
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
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

        let mut output_boxes1: Vec<_> = Vec::with_capacity(10);
        let mut output_masks1: Vec<_> = Vec::with_capacity(10);
        decoder
            .decode_quantized(
                &[detect0.view().into(), detect1.view().into()],
                &mut output_boxes1,
                &mut output_masks1,
            )
            .unwrap();

        let mut output_boxes1_f32: Vec<_> = Vec::with_capacity(10);
        let mut output_masks1_f32: Vec<_> = Vec::with_capacity(10);

        let detect0 = dequantize_ndarray(detect0.view(), quant0);
        let detect1 = dequantize_ndarray(detect1.view(), quant1);
        decoder
            .decode_float::<f32>(
                &[detect0.view().into_dyn(), detect1.view().into_dyn()],
                &mut output_boxes1_f32,
                &mut output_masks1_f32,
            )
            .unwrap();

        compare_outputs((&output_boxes, &output_boxes1), (&[], &output_masks1));
        compare_outputs(
            (&output_boxes, &output_boxes1_f32),
            (&[], &output_masks1_f32),
        );
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
                    ArrayViewDQuantized::from(detect1.view()),
                    ArrayViewDQuantized::from(detect0.view()),
                ],
                &mut output_boxes,
                &mut output_masks,
            )
            .unwrap();
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
    fn test_modelpack_seg() {
        let out = include_bytes!("../../../testdata/modelpack_seg_2x160x160.bin");
        let out = ndarray::Array4::from_shape_vec((1, 2, 160, 160), out.to_vec()).unwrap();
        let quant = (1.0 / 255.0, 0).into();

        let decoder = DecoderBuilder::default()
            .with_config_modelpack_seg(configs::Segmentation {
                decoder: DecoderType::ModelPack,
                quantization: Some(quant),
                shape: vec![1, 2, 160, 160],
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumClasses, 2),
                    (DimName::Height, 160),
                    (DimName::Width, 160),
                ],
            })
            .build()
            .unwrap();
        let mut output_boxes: Vec<_> = Vec::with_capacity(10);
        let mut output_masks: Vec<_> = Vec::with_capacity(10);
        decoder
            .decode_quantized(&[out.view().into()], &mut output_boxes, &mut output_masks)
            .unwrap();

        let mut mask = out.slice(s![0, .., .., ..]);
        mask.swap_axes(0, 1);
        mask.swap_axes(1, 2);
        let mask = [Segmentation {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
            segmentation: mask.into_owned(),
        }];
        compare_outputs((&[], &output_boxes), (&mask, &output_masks));

        decoder
            .decode_float::<f32>(
                &[dequantize_ndarray(out.view(), quant.into())
                    .view()
                    .into_dyn()],
                &mut output_boxes,
                &mut output_masks,
            )
            .unwrap();

        // not expected for float decoder to have same values as quantized decoder, as
        // float decoder ensures the data fills 0-255, quantized decoder uses whatever
        // the model output. Thus the float output is the same as the quantized output
        // but scaled differently. However, it is expected that the mask after argmax
        // will be the same.
        compare_outputs((&[], &output_boxes), (&[], &[]));
        let mask0 = segmentation_to_mask(mask[0].segmentation.view());
        let mask1 = segmentation_to_mask(output_masks[0].segmentation.view());

        assert_eq!(mask0, mask1);
    }

    #[test]
    fn test_modelpack_segdet() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;

        let boxes = include_bytes!("../../../testdata/modelpack_boxes_1935x1x4.bin");
        let boxes = Array4::from_shape_vec((1, 1935, 1, 4), boxes.to_vec()).unwrap();

        let scores = include_bytes!("../../../testdata/modelpack_scores_1935x1.bin");
        let scores = Array3::from_shape_vec((1, 1935, 1), scores.to_vec()).unwrap();

        let seg = include_bytes!("../../../testdata/modelpack_seg_2x160x160.bin");
        let seg = Array4::from_shape_vec((1, 2, 160, 160), seg.to_vec()).unwrap();

        let quant_boxes = (0.004656755365431309, 21).into();
        let quant_scores = (0.0019603664986789227, 0).into();
        let quant_seg = (1.0 / 255.0, 0).into();

        let decoder = DecoderBuilder::default()
            .with_config_modelpack_segdet(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: Some(quant_boxes),
                    shape: vec![1, 1935, 1, 4],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 1935),
                        (DimName::Padding, 1),
                        (DimName::NumFeatures, 4),
                    ],
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: Some(quant_scores),
                    shape: vec![1, 1935, 1],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 1935),
                        (DimName::NumClasses, 1),
                    ],
                },
                configs::Segmentation {
                    decoder: DecoderType::ModelPack,
                    quantization: Some(quant_seg),
                    shape: vec![1, 2, 160, 160],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 2),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
                },
            )
            .with_iou_threshold(iou_threshold)
            .with_score_threshold(score_threshold)
            .build()
            .unwrap();
        let mut output_boxes: Vec<_> = Vec::with_capacity(10);
        let mut output_masks: Vec<_> = Vec::with_capacity(10);
        decoder
            .decode_quantized(
                &[scores.view().into(), boxes.view().into(), seg.view().into()],
                &mut output_boxes,
                &mut output_masks,
            )
            .unwrap();

        let mut mask = seg.slice(s![0, .., .., ..]);
        mask.swap_axes(0, 1);
        mask.swap_axes(1, 2);
        let mask = [Segmentation {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
            segmentation: mask.into_owned(),
        }];
        let correct_boxes = [DetectBox {
            bbox: BoundingBox {
                xmin: 0.40513772,
                ymin: 0.6379755,
                xmax: 0.5122431,
                ymax: 0.7730214,
            },
            score: 0.4861709,
            label: 0,
        }];
        compare_outputs((&correct_boxes, &output_boxes), (&mask, &output_masks));

        let scores = dequantize_ndarray(scores.view(), quant_scores.into());
        let boxes = dequantize_ndarray(boxes.view(), quant_boxes.into());
        let seg = dequantize_ndarray(seg.view(), quant_seg.into());
        decoder
            .decode_float::<f32>(
                &[
                    scores.view().into_dyn(),
                    boxes.view().into_dyn(),
                    seg.view().into_dyn(),
                ],
                &mut output_boxes,
                &mut output_masks,
            )
            .unwrap();

        // not expected for float segmentation decoder to have same values as quantized
        // segmentation decoder, as float decoder ensures the data fills 0-255,
        // quantized decoder uses whatever the model output. Thus the float
        // output is the same as the quantized output but scaled differently.
        // However, it is expected that the mask after argmax will be the same.
        compare_outputs((&correct_boxes, &output_boxes), (&[], &[]));
        let mask0 = segmentation_to_mask(mask[0].segmentation.view());
        let mask1 = segmentation_to_mask(output_masks[0].segmentation.view());

        assert_eq!(mask0, mask1);
    }

    #[test]
    fn test_modelpack_segdet_split() {
        let score_threshold = 0.8;
        let iou_threshold = 0.5;

        let seg = include_bytes!("../../../testdata/modelpack_seg_2x160x160.bin");
        let seg = ndarray::Array4::from_shape_vec((1, 2, 160, 160), seg.to_vec()).unwrap();

        let detect0 = include_bytes!("../../../testdata/modelpack_split_9x15x18.bin");
        let detect0 = ndarray::Array4::from_shape_vec((1, 9, 15, 18), detect0.to_vec()).unwrap();

        let detect1 = include_bytes!("../../../testdata/modelpack_split_17x30x18.bin");
        let detect1 = ndarray::Array4::from_shape_vec((1, 17, 30, 18), detect1.to_vec()).unwrap();

        let quant0 = (0.08547406643629074, 174).into();
        let quant1 = (0.09929127991199493, 183).into();
        let quant_seg = (1.0 / 255.0, 0).into();

        let anchors0 = vec![
            [0.36666667461395264, 0.31481480598449707],
            [0.38749998807907104, 0.4740740656852722],
            [0.5333333611488342, 0.644444465637207],
        ];
        let anchors1 = vec![
            [0.13750000298023224, 0.2074074000120163],
            [0.2541666626930237, 0.21481481194496155],
            [0.23125000298023224, 0.35185185074806213],
        ];

        let decoder = DecoderBuilder::default()
            .with_config_modelpack_segdet_split(
                vec![
                    configs::Detection {
                        decoder: DecoderType::ModelPack,
                        shape: vec![1, 17, 30, 18],
                        anchors: Some(anchors1),
                        quantization: Some(quant1),
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::Height, 17),
                            (DimName::Width, 30),
                            (DimName::NumAnchorsXFeatures, 18),
                        ],
                    },
                    configs::Detection {
                        decoder: DecoderType::ModelPack,
                        shape: vec![1, 9, 15, 18],
                        anchors: Some(anchors0),
                        quantization: Some(quant0),
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::Height, 9),
                            (DimName::Width, 15),
                            (DimName::NumAnchorsXFeatures, 18),
                        ],
                    },
                ],
                configs::Segmentation {
                    decoder: DecoderType::ModelPack,
                    quantization: Some(quant_seg),
                    shape: vec![1, 2, 160, 160],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 2),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
                },
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
                    detect0.view().into(),
                    detect1.view().into(),
                    seg.view().into(),
                ],
                &mut output_boxes,
                &mut output_masks,
            )
            .unwrap();

        let mut mask = seg.slice(s![0, .., .., ..]);
        mask.swap_axes(0, 1);
        mask.swap_axes(1, 2);
        let mask = [Segmentation {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
            segmentation: mask.into_owned(),
        }];
        let correct_boxes = [DetectBox {
            bbox: BoundingBox {
                xmin: 0.43171933,
                ymin: 0.68243736,
                xmax: 0.5626645,
                ymax: 0.808863,
            },
            score: 0.99240804,
            label: 0,
        }];
        println!("Output Boxes: {:?}", output_boxes);
        compare_outputs((&correct_boxes, &output_boxes), (&mask, &output_masks));

        let detect0 = dequantize_ndarray(detect0.view(), quant0.into());
        let detect1 = dequantize_ndarray(detect1.view(), quant1.into());
        let seg = dequantize_ndarray(seg.view(), quant_seg.into());
        decoder
            .decode_float::<f32>(
                &[
                    detect0.view().into_dyn(),
                    detect1.view().into_dyn(),
                    seg.view().into_dyn(),
                ],
                &mut output_boxes,
                &mut output_masks,
            )
            .unwrap();

        // not expected for float segmentation decoder to have same values as quantized
        // segmentation decoder, as float decoder ensures the data fills 0-255,
        // quantized decoder uses whatever the model output. Thus the float
        // output is the same as the quantized output but scaled differently.
        // However, it is expected that the mask after argmax will be the same.
        compare_outputs((&correct_boxes, &output_boxes), (&[], &[]));
        let mask0 = segmentation_to_mask(mask[0].segmentation.view());
        let mask1 = segmentation_to_mask(output_masks[0].segmentation.view());

        assert_eq!(mask0, mask1);
    }

    #[test]
    fn test_dequant_chunked() {
        let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
        let mut out =
            unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) }.to_vec();
        out.push(123); // make sure to test non multiple of 16 length

        let mut out_dequant = vec![0.0; 84 * 8400 + 1];
        let mut out_dequant_simd = vec![0.0; 84 * 8400 + 1];
        let quant = Quantization::new(0.0040811873, -123);
        dequantize_cpu(&out, quant, &mut out_dequant);

        dequantize_cpu_chunked(&out, quant, &mut out_dequant_simd);
        assert_eq!(out_dequant, out_dequant_simd);

        let quant = Quantization::new(0.0040811873, 0);
        dequantize_cpu(&out, quant, &mut out_dequant);

        dequantize_cpu_chunked(&out, quant, &mut out_dequant_simd);
        assert_eq!(out_dequant, out_dequant_simd);
    }

    #[test]
    fn test_decoder_yolo_det() {
        let score_threshold = 0.25;
        let iou_threshold = 0.7;
        let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
        let out = Array3::from_shape_vec((1, 84, 8400), out.to_vec()).unwrap();
        let quant = (0.0040811873, -123).into();

        let decoder = DecoderBuilder::default()
            .with_config_yolo_det(configs::Detection {
                decoder: DecoderType::Ultralytics,
                shape: vec![1, 84, 8400],
                anchors: None,
                quantization: Some(quant),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumFeatures, 84),
                    (DimName::NumBoxes, 8400),
                ],
            })
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .build()
            .unwrap();

        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        decode_yolo_det(
            (out.slice(s![0, .., ..]), quant.into()),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
        );
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
        ));

        let mut output_boxes1: Vec<_> = Vec::with_capacity(50);
        let mut output_masks1: Vec<_> = Vec::with_capacity(50);
        decoder
            .decode_quantized(&[out.view().into()], &mut output_boxes1, &mut output_masks1)
            .unwrap();

        let out = dequantize_ndarray(out.view(), quant.into());
        let mut output_boxes_f32: Vec<_> = Vec::with_capacity(50);
        let mut output_masks_f32: Vec<_> = Vec::with_capacity(50);
        decoder
            .decode_float::<f32>(
                &[out.view().into_dyn()],
                &mut output_boxes_f32,
                &mut output_masks_f32,
            )
            .unwrap();

        compare_outputs((&output_boxes, &output_boxes1), (&[], &output_masks1));
        compare_outputs((&output_boxes, &output_boxes_f32), (&[], &output_masks_f32));
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
        let protos = dequantize_ndarray::<_, _, f32>(protos.view(), quant_protos);
        let seg = dequantize_ndarray::<_, _, f32>(boxes.view(), quant_boxes);
        let mut output_boxes: Vec<_> = Vec::with_capacity(10);
        let mut output_masks: Vec<_> = Vec::with_capacity(10);
        decode_yolo_segdet_float(
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
        let boxes = ndarray::Array3::from_shape_vec((1, 116, 8400), boxes.to_vec()).unwrap();
        let quant_boxes = (0.021287761628627777, 31).into();

        let protos = include_bytes!("../../../testdata/yolov8_protos_160x160x32.bin");
        let protos =
            unsafe { std::slice::from_raw_parts(protos.as_ptr() as *const i8, protos.len()) };
        let protos = ndarray::Array4::from_shape_vec((1, 160, 160, 32), protos.to_vec()).unwrap();
        let quant_protos = (0.02491161972284317, -117).into();
        let mut output_boxes: Vec<_> = Vec::with_capacity(500);
        let mut output_masks: Vec<_> = Vec::with_capacity(500);

        let decoder = DecoderBuilder::default()
            .with_config_yolo_segdet(
                configs::Detection {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(quant_boxes),
                    shape: vec![1, 116, 8400],
                    anchors: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 116),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(quant_protos),
                    shape: vec![1, 160, 160, 32],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                        (DimName::NumProtos, 32),
                    ],
                },
            )
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .build()
            .unwrap();

        let quant_boxes = quant_boxes.into();
        let quant_protos = quant_protos.into();

        decode_yolo_segdet_quant(
            (boxes.slice(s![0, .., ..]), quant_boxes),
            (protos.slice(s![0, .., .., ..]), quant_protos),
            score_threshold,
            iou_threshold,
            &mut output_boxes,
            &mut output_masks,
        );

        let mut output_boxes1: Vec<_> = Vec::with_capacity(500);
        let mut output_masks1: Vec<_> = Vec::with_capacity(500);

        decoder
            .decode_quantized(
                &[boxes.view().into(), protos.view().into()],
                &mut output_boxes1,
                &mut output_masks1,
            )
            .unwrap();

        let protos = dequantize_ndarray::<_, _, f32>(protos.view(), quant_protos);
        let seg = dequantize_ndarray::<_, _, f32>(boxes.view(), quant_boxes);

        let mut output_boxes_f32: Vec<_> = Vec::with_capacity(500);
        let mut output_masks_f32: Vec<_> = Vec::with_capacity(500);
        decode_yolo_segdet_float(
            seg.slice(s![0, .., ..]),
            protos.slice(s![0, .., .., ..]),
            score_threshold,
            iou_threshold,
            &mut output_boxes_f32,
            &mut output_masks_f32,
        );

        let mut output_boxes1_f32: Vec<_> = Vec::with_capacity(500);
        let mut output_masks1_f32: Vec<_> = Vec::with_capacity(500);

        decoder
            .decode_float(
                &[seg.view().into_dyn(), protos.view().into_dyn()],
                &mut output_boxes1_f32,
                &mut output_masks1_f32,
            )
            .unwrap();

        compare_outputs(
            (&output_boxes, &output_boxes1),
            (&output_masks, &output_masks1),
        );

        compare_outputs(
            (&output_boxes, &output_boxes_f32),
            (&output_masks, &output_masks_f32),
        );

        compare_outputs(
            (&output_boxes_f32, &output_boxes1_f32),
            (&output_masks_f32, &output_masks1_f32),
        );
    }

    #[test]
    fn test_decoder_yolo_split() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let boxes = include_bytes!("../../../testdata/yolov8_boxes_116x8400.bin");
        let boxes = unsafe { std::slice::from_raw_parts(boxes.as_ptr() as *const i8, boxes.len()) };
        let boxes: Vec<_> = boxes.iter().map(|x| *x as i16 * 256).collect();
        let boxes = ndarray::Array3::from_shape_vec((1, 116, 8400), boxes).unwrap();

        let quant_boxes = Quantization::new(0.021287761628627777 / 256.0, 31 * 256);

        let decoder = DecoderBuilder::default()
            .with_config_yolo_split_det(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(QuantTuple(quant_boxes.scale, quant_boxes.zero_point)),
                    shape: vec![1, 4, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(QuantTuple(quant_boxes.scale, quant_boxes.zero_point)),
                    shape: vec![1, 80, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
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
                    boxes.slice(s![.., ..4, ..]).into(),
                    boxes.slice(s![.., 4..84, ..]).into(),
                ],
                &mut output_boxes,
                &mut output_masks,
            )
            .unwrap();

        let seg = dequantize_ndarray::<_, _, f32>(boxes.view(), quant_boxes);
        let mut output_boxes_f32: Vec<_> = Vec::with_capacity(500);
        decode_yolo_det_float(
            seg.slice(s![0, ..84, ..]),
            score_threshold,
            iou_threshold,
            &mut output_boxes_f32,
        );

        let mut output_boxes1: Vec<_> = Vec::with_capacity(500);
        let mut output_masks1: Vec<_> = Vec::with_capacity(500);
        decoder
            .decode_float(
                &[
                    seg.slice(s![.., ..4, ..]).into_dyn(),
                    seg.slice(s![.., 4..84, ..]).into_dyn(),
                ],
                &mut output_boxes1,
                &mut output_masks1,
            )
            .unwrap();
        compare_outputs((&output_boxes, &output_boxes_f32), (&output_masks, &[]));
        compare_outputs((&output_boxes_f32, &output_boxes1), (&[], &output_masks1));
    }

    #[test]
    fn test_decoder_masks_config_mixed() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let boxes = include_bytes!("../../../testdata/yolov8_boxes_116x8400.bin");
        let boxes = unsafe { std::slice::from_raw_parts(boxes.as_ptr() as *const i8, boxes.len()) };
        let boxes: Vec<_> = boxes.iter().map(|x| *x as i16 * 256).collect();
        let boxes = ndarray::Array3::from_shape_vec((1, 116, 8400), boxes).unwrap();

        let quant_boxes = Quantization::new(0.021287761628627777 / 256.0, 31 * 256);

        let protos = include_bytes!("../../../testdata/yolov8_protos_160x160x32.bin");
        let protos =
            unsafe { std::slice::from_raw_parts(protos.as_ptr() as *const i8, protos.len()) };
        let protos: Vec<_> = protos.to_vec();
        let protos = ndarray::Array4::from_shape_vec((1, 160, 160, 32), protos.to_vec()).unwrap();
        let quant_protos = Quantization::new(0.02491161972284317, -117);

        let decoder = DecoderBuilder::default()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(QuantTuple(quant_boxes.scale, quant_boxes.zero_point)),
                    shape: vec![1, 4, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(QuantTuple(quant_boxes.scale, quant_boxes.zero_point)),
                    shape: vec![1, 80, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(QuantTuple(quant_boxes.scale, quant_boxes.zero_point)),
                    shape: vec![1, 32, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(QuantTuple(quant_protos.scale, quant_protos.zero_point)),
                    shape: vec![1, 160, 160, 32],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                        (DimName::NumProtos, 32),
                    ],
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
                    boxes.slice(s![.., ..4, ..]).into(),
                    boxes.slice(s![.., 4..84, ..]).into(),
                    boxes.slice(s![.., 84.., ..]).into(),
                    protos.view().into(),
                ],
                &mut output_boxes,
                &mut output_masks,
            )
            .unwrap();

        let protos = dequantize_ndarray::<_, _, f32>(protos.view(), quant_protos);
        let seg = dequantize_ndarray::<_, _, f32>(boxes.view(), quant_boxes);
        let mut output_boxes_f32: Vec<_> = Vec::with_capacity(500);
        let mut output_masks_f32: Vec<_> = Vec::with_capacity(500);
        decode_yolo_segdet_float(
            seg.slice(s![0, .., ..]),
            protos.slice(s![0, .., .., ..]),
            score_threshold,
            iou_threshold,
            &mut output_boxes_f32,
            &mut output_masks_f32,
        );

        let mut output_boxes1: Vec<_> = Vec::with_capacity(500);
        let mut output_masks1: Vec<_> = Vec::with_capacity(500);

        decoder
            .decode_float(
                &[
                    seg.slice(s![.., ..4, ..]).into_dyn(),
                    seg.slice(s![.., 4..84, ..]).into_dyn(),
                    seg.slice(s![.., 84.., ..]).into_dyn(),
                    protos.view().into_dyn(),
                ],
                &mut output_boxes1,
                &mut output_masks1,
            )
            .unwrap();
        compare_outputs(
            (&output_boxes, &output_boxes_f32),
            (&output_masks, &output_masks_f32),
        );
        compare_outputs(
            (&output_boxes_f32, &output_boxes1),
            (&output_masks_f32, &output_masks1),
        );
    }

    #[test]
    fn test_decoder_masks_config_i32() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let boxes = include_bytes!("../../../testdata/yolov8_boxes_116x8400.bin");
        let boxes = unsafe { std::slice::from_raw_parts(boxes.as_ptr() as *const i8, boxes.len()) };
        let scale = 1 << 23;
        let boxes: Vec<_> = boxes.iter().map(|x| *x as i32 * scale).collect();
        let boxes = ndarray::Array3::from_shape_vec((1, 116, 8400), boxes).unwrap();

        let quant_boxes = Quantization::new(0.021287761628627777 / scale as f32, 31 * scale);

        let protos = include_bytes!("../../../testdata/yolov8_protos_160x160x32.bin");
        let protos =
            unsafe { std::slice::from_raw_parts(protos.as_ptr() as *const i8, protos.len()) };
        let protos: Vec<_> = protos.iter().map(|x| *x as i32 * scale).collect();
        let protos = ndarray::Array4::from_shape_vec((1, 160, 160, 32), protos.to_vec()).unwrap();
        let quant_protos = Quantization::new(0.02491161972284317 / scale as f32, -117 * scale);

        let decoder = DecoderBuilder::default()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(QuantTuple(quant_boxes.scale, quant_boxes.zero_point)),
                    shape: vec![1, 4, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(QuantTuple(quant_boxes.scale, quant_boxes.zero_point)),
                    shape: vec![1, 80, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(QuantTuple(quant_boxes.scale, quant_boxes.zero_point)),
                    shape: vec![1, 32, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(QuantTuple(quant_protos.scale, quant_protos.zero_point)),
                    shape: vec![1, 160, 160, 32],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                        (DimName::NumProtos, 32),
                    ],
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
                    boxes.slice(s![.., ..4, ..]).into(),
                    boxes.slice(s![.., 4..84, ..]).into(),
                    boxes.slice(s![.., 84.., ..]).into(),
                    protos.view().into(),
                ],
                &mut output_boxes,
                &mut output_masks,
            )
            .unwrap();

        let protos = dequantize_ndarray::<_, _, f32>(protos.view(), quant_protos);
        let seg = dequantize_ndarray::<_, _, f32>(boxes.view(), quant_boxes);
        let mut output_boxes_f32: Vec<_> = Vec::with_capacity(500);
        let mut output_masks_f32: Vec<Segmentation> = Vec::with_capacity(500);
        decode_yolo_segdet_float(
            seg.slice(s![0, .., ..]),
            protos.slice(s![0, .., .., ..]),
            score_threshold,
            iou_threshold,
            &mut output_boxes_f32,
            &mut output_masks_f32,
        );

        assert_eq!(output_boxes.len(), output_boxes_f32.len());
        assert_eq!(output_masks.len(), output_masks_f32.len());

        compare_outputs(
            (&output_boxes, &output_boxes_f32),
            (&output_masks, &output_masks_f32),
        );
    }

    #[test]
    fn test_ndarray_to_xyxy_float() {
        let arr = array![10.0_f32, 20.0, 20.0, 20.0];
        let xyxy: [f32; 4] = XYWH::ndarray_to_xyxy_float(arr.view());
        assert_eq!(xyxy, [0.0_f32, 10.0, 20.0, 30.0]);

        let arr = array![10.0_f32, 20.0, 20.0, 20.0];
        let xyxy: [f32; 4] = XYXY::ndarray_to_xyxy_float(arr.view());
        assert_eq!(xyxy, [10.0_f32, 20.0, 20.0, 20.0]);
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod decoder_tracked_tests {
    #![allow(clippy::excessive_precision)]
    use super::*;
    use crate::{
        DecoderBuilder,
        configs::{DecoderType, DimName, Protos},
    };
    use edgefirst_tracker::ByteTrackBuilder;
    use ndarray::{Array3, s};

    #[test]
    fn test_tracker_error() {
        let score_threshold = 0.15;
        let iou_threshold = 0.7;
        let quant = (0.0040811873, -123);
        let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
        let out = Array3::from_shape_vec((1, 84, 8400), out.to_vec()).unwrap();
        let out_float = dequantize_ndarray::<_, _, f32>(out.view(), quant.into());
        let mut decoder = DecoderBuilder::default()
            .with_config_yolo_det(configs::Detection {
                decoder: DecoderType::Ultralytics,
                shape: vec![1, 84, 8400],
                anchors: None,
                quantization: Some(quant.into()),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumFeatures, 84),
                    (DimName::NumBoxes, 8400),
                ],
            })
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .build()
            .unwrap();

        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        let mut output_masks: Vec<_> = Vec::with_capacity(50);
        let mut output_tracks: Vec<_> = Vec::with_capacity(50);
        let result = decoder.decode_quantized_tracked(
            &[out.view().into()],
            &mut output_boxes,
            &mut output_masks,
            &mut output_tracks,
            0,
        );
        assert!(matches!(result, Err(DecoderError::NoTracker)));

        let result = decoder.decode_float_tracked(
            &[out_float.view().into_dyn()],
            &mut output_boxes,
            &mut output_masks,
            &mut output_tracks,
            0,
        );
        assert!(matches!(result, Err(DecoderError::NoTracker)));
    }

    #[test]
    fn test_tracker_yolo_det() {
        let score_threshold = 0.15;
        let iou_threshold = 0.7;
        let quant = (0.0040811873, -123);
        let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
        let out = Array3::from_shape_vec((1, 84, 8400), out.to_vec()).unwrap();
        let out_float = dequantize_ndarray::<_, _, f32>(out.view(), quant.into());
        let mut decoder = DecoderBuilder::default()
            .with_config_yolo_det(configs::Detection {
                decoder: DecoderType::Ultralytics,
                shape: vec![1, 84, 8400],
                anchors: None,
                quantization: Some(quant.into()),
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumFeatures, 84),
                    (DimName::NumBoxes, 8400),
                ],
            })
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_tracker(ByteTrackBuilder::default().track_high_conf(0.5).build())
            .build()
            .unwrap();

        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        let mut output_masks: Vec<_> = Vec::with_capacity(50);
        let mut output_tracks: Vec<_> = Vec::with_capacity(50);
        decoder
            .decode_float_tracked(
                &[out_float.view().into_dyn()],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                0,
            )
            .unwrap();

        assert!(!output_tracks.is_empty());
        assert_eq!(output_boxes.len(), 1);
        assert_eq!(output_tracks.len(), 1);
        assert_eq!(output_masks.len(), 0);
        assert!(output_boxes[0].equal_within_delta(&output_tracks[0].last_box, 1e-6));

        let last_track = output_tracks[0];

        // Decode again
        decoder
            .decode_quantized_tracked(
                &[out.view().into()],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                10000,
            )
            .unwrap();
        assert!(!output_tracks.is_empty());
        assert_eq!(output_boxes.len(), 1);
        assert_eq!(output_tracks.len(), 1);
        assert_eq!(output_masks.len(), 0);
        assert!(output_boxes[0].equal_within_delta(&output_tracks[0].last_box, 1e-6));
        assert_eq!(output_tracks[0].uuid, last_track.uuid);
    }

    #[test]
    fn test_tracker_yolo_segdet() {
        let score_threshold = 0.3;
        let iou_threshold = 0.45;
        let boxes = include_bytes!("../../../testdata/yolov8_boxes_116x8400.bin");
        let boxes = unsafe { std::slice::from_raw_parts(boxes.as_ptr() as *const i8, boxes.len()) };
        let boxes = ndarray::Array3::from_shape_vec((1, 116, 8400), boxes.to_vec()).unwrap();
        let quant_boxes = (0.021287761628627777, 31).into();

        let protos = include_bytes!("../../../testdata/yolov8_protos_160x160x32.bin");
        let protos =
            unsafe { std::slice::from_raw_parts(protos.as_ptr() as *const i8, protos.len()) };
        let protos = ndarray::Array4::from_shape_vec((1, 160, 160, 32), protos.to_vec()).unwrap();
        let quant_protos = (0.02491161972284317, -117).into();

        let mut decoder = DecoderBuilder::default()
            .with_config_yolo_segdet(
                configs::Detection {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(quant_boxes),
                    shape: vec![1, 116, 8400],
                    anchors: None,
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 116),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(quant_protos),
                    shape: vec![1, 160, 160, 32],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                        (DimName::NumProtos, 32),
                    ],
                },
            )
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_tracker(ByteTrackBuilder::new().track_high_conf(0.5).build())
            .build()
            .unwrap();

        let quant_boxes = quant_boxes.into();
        let quant_protos = quant_protos.into();

        let mut output_boxes1: Vec<_> = Vec::with_capacity(500);
        let mut output_masks1: Vec<_> = Vec::with_capacity(500);
        let mut output_tracks1: Vec<_> = Vec::with_capacity(500);

        decoder
            .decode_quantized_tracked(
                &[boxes.view().into(), protos.view().into()],
                &mut output_boxes1,
                &mut output_masks1,
                &mut output_tracks1,
                1000,
            )
            .unwrap();

        let protos = dequantize_ndarray::<_, _, f32>(protos.view(), quant_protos);
        let boxes = dequantize_ndarray::<_, _, f32>(boxes.view(), quant_boxes);

        let mut output_boxes2: Vec<_> = Vec::with_capacity(500);
        let mut output_masks2: Vec<_> = Vec::with_capacity(500);
        let mut output_tracks2: Vec<_> = Vec::with_capacity(500);

        decoder
            .decode_float_tracked(
                &[boxes.view().into_dyn(), protos.view().into_dyn()],
                &mut output_boxes2,
                &mut output_masks2,
                &mut output_tracks2,
                1000,
            )
            .unwrap();

        assert_eq!(output_boxes1.len(), output_boxes2.len());
        assert_eq!(output_masks1.len(), output_masks2.len());
        assert_eq!(output_tracks1.len(), output_tracks2.len());

        assert_eq!(output_boxes1, output_boxes2);
        for (m1, m2) in output_masks1.iter().zip(output_masks2.iter()) {
            assert_eq!(m1.xmin, m2.xmin);
            assert_eq!(m1.ymin, m2.ymin);
            assert_eq!(m1.xmax, m2.xmax);
            assert_eq!(m1.ymax, m2.ymax);

            let m1 = segmentation_to_mask(m1.segmentation.view());
            let m2 = segmentation_to_mask(m2.segmentation.view());
            assert_eq!(m1, m2);
        }
        for (t1, t2) in output_tracks1.iter().zip(output_tracks2.iter()) {
            assert_eq!(t1.uuid, t2.uuid);
        }
    }

    #[test]
    fn test_tracker_yolo_split_det() {
        let score_threshold = 0.15;
        let iou_threshold = 0.7;
        let quant = (0.0040811873, -123);
        let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
        let out = Array3::from_shape_vec((1, 84, 8400), out.to_vec()).unwrap();
        let out_float = dequantize_ndarray::<_, _, f32>(out.view(), quant.into());
        let mut decoder = DecoderBuilder::default()
            .with_config_yolo_split_det(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(quant.into()),
                    shape: vec![1, 4, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(quant.into()),
                    shape: vec![1, 80, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
                },
            )
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_tracker(ByteTrackBuilder::default().track_high_conf(0.5).build())
            .build()
            .unwrap();

        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        let mut output_masks: Vec<_> = Vec::with_capacity(50);
        let mut output_tracks: Vec<_> = Vec::with_capacity(50);
        decoder
            .decode_float_tracked(
                &[
                    out_float.slice(s![.., ..4, ..]).view().into_dyn(),
                    out_float.slice(s![.., 4.., ..]).view().into_dyn(),
                ],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                0,
            )
            .unwrap();

        assert!(!output_tracks.is_empty());
        assert_eq!(output_boxes.len(), 1);
        assert_eq!(output_tracks.len(), 1);
        assert_eq!(output_masks.len(), 0);
        assert!(output_boxes[0].equal_within_delta(&output_tracks[0].last_box, 1e-6));

        let last_track = output_tracks[0];

        // Decode again
        decoder
            .decode_quantized_tracked(
                &[
                    out.slice(s![.., ..4, ..]).view().into(),
                    out.slice(s![.., 4.., ..]).view().into(),
                ],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                10000,
            )
            .unwrap();
        assert!(!output_tracks.is_empty());
        assert_eq!(output_boxes.len(), 1);
        assert_eq!(output_tracks.len(), 1);
        assert_eq!(output_masks.len(), 0);
        assert!(output_boxes[0].equal_within_delta(&output_tracks[0].last_box, 1e-6));
        assert_eq!(output_tracks[0].uuid, last_track.uuid);
    }

    #[test]
    fn test_tracker_yolo_split_segdet() {
        let score_threshold = 0.3;
        let iou_threshold = 0.45;
        let boxes = include_bytes!("../../../testdata/yolov8_boxes_116x8400.bin");
        let boxes = unsafe { std::slice::from_raw_parts(boxes.as_ptr() as *const i8, boxes.len()) };
        let boxes = ndarray::Array3::from_shape_vec((1, 116, 8400), boxes.to_vec()).unwrap();
        let quant_boxes = (0.021287761628627777, 31);
        let boxes_float = dequantize_ndarray::<_, _, f32>(boxes.view(), quant_boxes.into());

        let protos = include_bytes!("../../../testdata/yolov8_protos_160x160x32.bin");
        let protos =
            unsafe { std::slice::from_raw_parts(protos.as_ptr() as *const i8, protos.len()) };
        let protos = ndarray::Array4::from_shape_vec((1, 160, 160, 32), protos.to_vec()).unwrap();
        let quant_protos = (0.02491161972284317, -117);
        let protos_float = dequantize_ndarray::<_, _, f32>(protos.view(), quant_protos.into());

        let mut decoder = DecoderBuilder::default()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(quant_boxes.into()),
                    shape: vec![1, 4, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Scores {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(quant_boxes.into()),
                    shape: vec![1, 80, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 80),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::MaskCoefficients {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(quant_boxes.into()),
                    shape: vec![1, 32, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumProtos, 32),
                        (DimName::NumBoxes, 8400),
                    ],
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(quant_protos.into()),
                    shape: vec![1, 160, 160, 32],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                        (DimName::NumProtos, 32),
                    ],
                },
            )
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_tracker(ByteTrackBuilder::default().track_high_conf(0.5).build())
            .build()
            .unwrap();

        let mut output_boxes1: Vec<_> = Vec::with_capacity(500);
        let mut output_masks1: Vec<_> = Vec::with_capacity(500);
        let mut output_tracks1: Vec<_> = Vec::with_capacity(500);
        decoder
            .decode_float_tracked(
                &[
                    boxes_float.slice(s![.., ..4, ..]).view().into_dyn(),
                    boxes_float.slice(s![.., 4..84, ..]).view().into_dyn(),
                    boxes_float.slice(s![.., 84.., ..]).view().into_dyn(),
                    protos_float.view().into_dyn(),
                ],
                &mut output_boxes1,
                &mut output_masks1,
                &mut output_tracks1,
                0,
            )
            .unwrap();

        let mut output_boxes2: Vec<_> = Vec::with_capacity(500);
        let mut output_masks2: Vec<_> = Vec::with_capacity(500);
        let mut output_tracks2: Vec<_> = Vec::with_capacity(500);
        // Decode again
        decoder
            .decode_quantized_tracked(
                &[
                    boxes.slice(s![.., ..4, ..]).view().into(),
                    boxes.slice(s![.., 4..84, ..]).view().into(),
                    boxes.slice(s![.., 84.., ..]).view().into(),
                    protos.view().into(),
                ],
                &mut output_boxes2,
                &mut output_masks2,
                &mut output_tracks2,
                10000,
            )
            .unwrap();

        assert_eq!(output_boxes1, output_boxes2);
        for (m1, m2) in output_masks1.iter().zip(output_masks2.iter()) {
            assert_eq!(m1.xmin, m2.xmin);
            assert_eq!(m1.ymin, m2.ymin);
            assert_eq!(m1.xmax, m2.xmax);
            assert_eq!(m1.ymax, m2.ymax);

            let m1 = segmentation_to_mask(m1.segmentation.view());
            let m2 = segmentation_to_mask(m2.segmentation.view());
            assert_eq!(m1, m2);
        }
        for (t1, t2) in output_tracks1.iter().zip(output_tracks2.iter()) {
            assert_eq!(t1.uuid, t2.uuid);
        }
    }

    #[test]
    fn test_tracker_modelpack_det() {
        let score_threshold = 0.2;
        let iou_threshold = 0.45;

        let boxes = include_bytes!("../../../testdata/modelpack_boxes_1935x1x4.bin");
        let boxes = ndarray::Array4::from_shape_vec((1, 1935, 1, 4), boxes.to_vec()).unwrap();

        let scores = include_bytes!("../../../testdata/modelpack_scores_1935x1.bin");
        let scores = ndarray::Array3::from_shape_vec((1, 1935, 1), scores.to_vec()).unwrap();

        let quant_boxes = (0.004656755365431309, 21);
        let quant_scores = (0.0019603664986789227, 0);

        let boxes_float = dequantize_ndarray::<_, _, f32>(boxes.view(), quant_boxes.into());
        let scores_float = dequantize_ndarray::<_, _, f32>(scores.view(), quant_scores.into());

        let mut decoder = DecoderBuilder::default()
            .with_config_modelpack_det(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: Some(quant_boxes.into()),
                    shape: vec![1, 1935, 1, 4],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 1935),
                        (DimName::Padding, 1),
                        (DimName::NumFeatures, 4),
                    ],
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: Some(quant_scores.into()),
                    shape: vec![1, 1935, 1],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 1935),
                        (DimName::NumClasses, 1),
                    ],
                },
            )
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_tracker(ByteTrackBuilder::default().track_high_conf(0.4).build())
            .build()
            .unwrap();

        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        let mut output_masks: Vec<_> = Vec::with_capacity(50);
        let mut output_tracks: Vec<_> = Vec::with_capacity(50);
        decoder
            .decode_float_tracked(
                &[
                    boxes_float.view().into_dyn(),
                    scores_float.view().into_dyn(),
                ],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                0,
            )
            .unwrap();

        assert!(!output_tracks.is_empty());
        assert_eq!(output_boxes.len(), 1);
        assert_eq!(output_tracks.len(), 1);
        assert_eq!(output_masks.len(), 0);
        assert!(output_boxes[0].equal_within_delta(&output_tracks[0].last_box, 1e-6));

        let last_track = output_tracks[0];

        // Decode again
        decoder
            .decode_quantized_tracked(
                &[boxes.view().into(), scores.view().into()],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                10000,
            )
            .unwrap();
        assert!(!output_tracks.is_empty());
        assert_eq!(output_boxes.len(), 1);
        assert_eq!(output_tracks.len(), 1);
        assert_eq!(output_masks.len(), 0);
        assert!(output_boxes[0].equal_within_delta(&output_tracks[0].last_box, 1e-6));
        assert_eq!(output_tracks[0].uuid, last_track.uuid);
    }

    #[test]
    fn test_tracker_modelpack_det_split() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let detect0 = include_bytes!("../../../testdata/modelpack_split_9x15x18.bin");
        let detect0 = ndarray::Array4::from_shape_vec((1, 9, 15, 18), detect0.to_vec()).unwrap();
        let quant0 = (0.08547406643629074, 174);
        let detect0_float = dequantize_ndarray::<_, _, f32>(detect0.view(), quant0.into());

        let detect1 = include_bytes!("../../../testdata/modelpack_split_17x30x18.bin");
        let detect1 = ndarray::Array4::from_shape_vec((1, 17, 30, 18), detect1.to_vec()).unwrap();
        let quant1 = (0.09929127991199493, 183);
        let detect1_float = dequantize_ndarray::<_, _, f32>(detect1.view(), quant1.into());

        let anchors0 = vec![
            [0.36666667461395264, 0.31481480598449707],
            [0.38749998807907104, 0.4740740656852722],
            [0.5333333611488342, 0.644444465637207],
        ];
        let anchors1 = vec![
            [0.13750000298023224, 0.2074074000120163],
            [0.2541666626930237, 0.21481481194496155],
            [0.23125000298023224, 0.35185185074806213],
        ];

        let detect_config0 = configs::Detection {
            decoder: DecoderType::ModelPack,
            shape: vec![1, 9, 15, 18],
            anchors: Some(anchors0),
            quantization: Some(quant0.into()),
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::Height, 9),
                (DimName::Width, 15),
                (DimName::NumAnchorsXFeatures, 18),
            ],
        };

        let detect_config1 = configs::Detection {
            decoder: DecoderType::ModelPack,
            shape: vec![1, 17, 30, 18],
            anchors: Some(anchors1),
            quantization: Some(quant1.into()),
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::Height, 17),
                (DimName::Width, 30),
                (DimName::NumAnchorsXFeatures, 18),
            ],
        };

        let mut decoder = DecoderBuilder::default()
            .with_config_modelpack_det_split(vec![detect_config0, detect_config1])
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_tracker(ByteTrackBuilder::default().track_high_conf(0.4).build())
            .build()
            .unwrap();

        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        let mut output_masks: Vec<_> = Vec::with_capacity(50);
        let mut output_tracks: Vec<_> = Vec::with_capacity(50);
        decoder
            .decode_float_tracked(
                &[
                    detect0_float.view().into_dyn(),
                    detect1_float.view().into_dyn(),
                ],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                0,
            )
            .unwrap();

        assert!(!output_tracks.is_empty());
        assert_eq!(output_boxes.len(), 1);
        assert_eq!(output_tracks.len(), 1);
        assert_eq!(output_masks.len(), 0);
        assert!(output_boxes[0].equal_within_delta(&output_tracks[0].last_box, 1e-6));

        let last_track = output_tracks[0];

        // Decode again
        decoder
            .decode_quantized_tracked(
                &[detect0.view().into(), detect1.view().into()],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                10000,
            )
            .unwrap();
        assert!(!output_tracks.is_empty());
        assert_eq!(output_boxes.len(), 1);
        assert_eq!(output_tracks.len(), 1);
        assert_eq!(output_masks.len(), 0);
        assert!(output_boxes[0].equal_within_delta(&output_tracks[0].last_box, 1e-6));
        assert_eq!(output_tracks[0].uuid, last_track.uuid);
    }

    #[test]
    fn test_modelpack_segdet() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;

        let boxes = include_bytes!("../../../testdata/modelpack_boxes_1935x1x4.bin");
        let boxes = ndarray::Array4::from_shape_vec((1, 1935, 1, 4), boxes.to_vec()).unwrap();
        let quant_boxes = (0.004656755365431309, 21);
        let boxes_float = dequantize_ndarray::<_, _, f32>(boxes.view(), quant_boxes.into());

        let scores = include_bytes!("../../../testdata/modelpack_scores_1935x1.bin");
        let scores = ndarray::Array3::from_shape_vec((1, 1935, 1), scores.to_vec()).unwrap();
        let quant_scores = (0.0019603664986789227, 0);
        let scores_float = dequantize_ndarray::<_, _, f32>(scores.view(), quant_scores.into());

        let seg = include_bytes!("../../../testdata/modelpack_seg_2x160x160.bin");
        let seg = ndarray::Array4::from_shape_vec((1, 2, 160, 160), seg.to_vec()).unwrap();
        let quant_seg = (1.0 / 255.0, 0);
        let seg_float = dequantize_ndarray::<_, _, f32>(seg.view(), quant_seg.into());

        let mut decoder = DecoderBuilder::default()
            .with_config_modelpack_segdet(
                configs::Boxes {
                    decoder: DecoderType::ModelPack,
                    quantization: Some(quant_boxes.into()),
                    shape: vec![1, 1935, 1, 4],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 1935),
                        (DimName::Padding, 1),
                        (DimName::NumFeatures, 4),
                    ],
                },
                configs::Scores {
                    decoder: DecoderType::ModelPack,
                    quantization: Some(quant_scores.into()),
                    shape: vec![1, 1935, 1],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumBoxes, 1935),
                        (DimName::NumClasses, 1),
                    ],
                },
                configs::Segmentation {
                    decoder: DecoderType::ModelPack,
                    quantization: Some(quant_seg.into()),
                    shape: vec![1, 2, 160, 160],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 2),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
                },
            )
            .with_iou_threshold(iou_threshold)
            .with_score_threshold(score_threshold)
            .with_tracker(ByteTrackBuilder::default().track_high_conf(0.4).build())
            .build()
            .unwrap();

        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        let mut output_masks: Vec<_> = Vec::with_capacity(50);
        let mut output_tracks: Vec<_> = Vec::with_capacity(50);
        decoder
            .decode_float_tracked(
                &[
                    boxes_float.view().into_dyn(),
                    scores_float.view().into_dyn(),
                    seg_float.view().into_dyn(),
                ],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                0,
            )
            .unwrap();

        assert!(!output_tracks.is_empty());
        assert_eq!(output_boxes.len(), 1);
        assert_eq!(output_tracks.len(), 1);
        assert_eq!(output_masks.len(), 1);
        assert!(output_boxes[0].equal_within_delta(&output_tracks[0].last_box, 1e-6));

        let last_track = output_tracks[0];

        // Decode again
        decoder
            .decode_quantized_tracked(
                &[boxes.view().into(), scores.view().into(), seg.view().into()],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                10000,
            )
            .unwrap();
        assert!(!output_tracks.is_empty());
        assert_eq!(output_boxes.len(), 1);
        assert_eq!(output_tracks.len(), 1);
        assert_eq!(output_masks.len(), 1);
        assert!(output_boxes[0].equal_within_delta(&output_tracks[0].last_box, 1e-6));
        assert_eq!(output_tracks[0].uuid, last_track.uuid);
    }

    #[test]
    fn test_tracker_modelpack_segdet_split() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let detect0 = include_bytes!("../../../testdata/modelpack_split_9x15x18.bin");
        let detect0 = ndarray::Array4::from_shape_vec((1, 9, 15, 18), detect0.to_vec()).unwrap();
        let quant0 = (0.08547406643629074, 174);
        let detect0_float = dequantize_ndarray::<_, _, f32>(detect0.view(), quant0.into());

        let detect1 = include_bytes!("../../../testdata/modelpack_split_17x30x18.bin");
        let detect1 = ndarray::Array4::from_shape_vec((1, 17, 30, 18), detect1.to_vec()).unwrap();
        let quant1 = (0.09929127991199493, 183);
        let detect1_float = dequantize_ndarray::<_, _, f32>(detect1.view(), quant1.into());

        let seg = include_bytes!("../../../testdata/modelpack_seg_2x160x160.bin");
        let seg = ndarray::Array4::from_shape_vec((1, 2, 160, 160), seg.to_vec()).unwrap();
        let quant_seg = (1.0 / 255.0, 0);
        let seg_float = dequantize_ndarray::<_, _, f32>(seg.view(), quant_seg.into());

        let anchors0 = vec![
            [0.36666667461395264, 0.31481480598449707],
            [0.38749998807907104, 0.4740740656852722],
            [0.5333333611488342, 0.644444465637207],
        ];
        let anchors1 = vec![
            [0.13750000298023224, 0.2074074000120163],
            [0.2541666626930237, 0.21481481194496155],
            [0.23125000298023224, 0.35185185074806213],
        ];

        let detect_config0 = configs::Detection {
            decoder: DecoderType::ModelPack,
            shape: vec![1, 9, 15, 18],
            anchors: Some(anchors0),
            quantization: Some(quant0.into()),
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::Height, 9),
                (DimName::Width, 15),
                (DimName::NumAnchorsXFeatures, 18),
            ],
        };

        let detect_config1 = configs::Detection {
            decoder: DecoderType::ModelPack,
            shape: vec![1, 17, 30, 18],
            anchors: Some(anchors1),
            quantization: Some(quant1.into()),
            dshape: vec![
                (DimName::Batch, 1),
                (DimName::Height, 17),
                (DimName::Width, 30),
                (DimName::NumAnchorsXFeatures, 18),
            ],
        };

        let mut decoder = DecoderBuilder::default()
            .with_config_modelpack_segdet_split(
                vec![detect_config0, detect_config1],
                configs::Segmentation {
                    decoder: DecoderType::ModelPack,
                    quantization: Some(quant_seg.into()),
                    shape: vec![1, 2, 160, 160],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumClasses, 2),
                        (DimName::Height, 160),
                        (DimName::Width, 160),
                    ],
                },
            )
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_tracker(ByteTrackBuilder::default().track_high_conf(0.4).build())
            .build()
            .unwrap();

        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        let mut output_masks: Vec<_> = Vec::with_capacity(50);
        let mut output_tracks: Vec<_> = Vec::with_capacity(50);
        decoder
            .decode_float_tracked(
                &[
                    detect0_float.view().into_dyn(),
                    detect1_float.view().into_dyn(),
                    seg_float.view().into_dyn(),
                ],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                0,
            )
            .unwrap();

        assert!(!output_tracks.is_empty());
        assert_eq!(output_boxes.len(), 1);
        assert_eq!(output_tracks.len(), 1);
        assert_eq!(output_masks.len(), 1);
        assert!(output_boxes[0].equal_within_delta(&output_tracks[0].last_box, 1e-6));

        let last_track = output_tracks[0];

        // Decode again
        decoder
            .decode_quantized_tracked(
                &[
                    detect0.view().into(),
                    detect1.view().into(),
                    seg.view().into(),
                ],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                10000,
            )
            .unwrap();
        assert!(!output_tracks.is_empty());
        assert_eq!(output_boxes.len(), 1);
        assert_eq!(output_tracks.len(), 1);
        assert_eq!(output_masks.len(), 1);
        assert!(output_boxes[0].equal_within_delta(&output_tracks[0].last_box, 1e-6));
        assert_eq!(output_tracks[0].uuid, last_track.uuid);
    }

    #[test]
    fn test_tracker_modelpack_seg() {
        let out = include_bytes!("../../../testdata/modelpack_seg_2x160x160.bin");
        let out = ndarray::Array4::from_shape_vec((1, 2, 160, 160), out.to_vec()).unwrap();
        let quant = (1.0 / 255.0, 0);
        let out_float = dequantize_ndarray::<_, _, f32>(out.view(), quant.into());

        let mut decoder = DecoderBuilder::default()
            .with_config_modelpack_seg(configs::Segmentation {
                decoder: DecoderType::ModelPack,
                quantization: Some(quant.into()),
                shape: vec![1, 2, 160, 160],
                dshape: vec![
                    (DimName::Batch, 1),
                    (DimName::NumClasses, 2),
                    (DimName::Height, 160),
                    (DimName::Width, 160),
                ],
            })
            .with_tracker(ByteTrackBuilder::default().track_high_conf(0.4).build())
            .build()
            .unwrap();
        let mut output_boxes: Vec<_> = Vec::with_capacity(10);
        let mut output_masks: Vec<_> = Vec::with_capacity(10);
        let mut output_tracks: Vec<_> = Vec::with_capacity(10);
        decoder
            .decode_quantized_tracked(
                &[out.view().into()],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                100,
            )
            .unwrap();
        assert_eq!(output_boxes.len(), 0);
        assert_eq!(output_tracks.len(), 0);
        assert_eq!(output_masks.len(), 1);
        let mask = output_masks[0].clone();
        decoder
            .decode_float_tracked(
                &[out_float.view().into_dyn()],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
                1000,
            )
            .unwrap();
        assert_eq!(output_boxes.len(), 0);
        assert_eq!(output_tracks.len(), 0);
        assert_eq!(output_masks.len(), 1);

        // not expected for float decoder to have same values as quantized decoder, as
        // float decoder ensures the data fills 0-255, quantized decoder uses whatever
        // the model output. Thus the float output is the same as the quantized output
        // but scaled differently. However, it is expected that the mask after argmax
        // will be the same.
        let mask0 = segmentation_to_mask(mask.segmentation.view());
        let mask1 = segmentation_to_mask(output_masks[0].segmentation.view());

        assert_eq!(mask0, mask1);
    }
}
