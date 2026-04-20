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
```rust,no_run
use edgefirst_decoder::{DecoderBuilder, DecoderResult, configs::{self, DecoderVersion}};
use edgefirst_tensor::TensorDyn;

fn main() -> DecoderResult<()> {
    // Create a decoder for a YOLOv8 model with quantized int8 output
    let decoder = DecoderBuilder::new()
        .with_config_yolo_det(configs::Detection {
            anchors: None,
            decoder: configs::DecoderType::Ultralytics,
            quantization: Some(configs::QuantTuple(0.012345, 26)),
            shape: vec![1, 84, 8400],
            dshape: Vec::new(),
            normalized: Some(true),
        },
        Some(DecoderVersion::Yolov8))
        .with_score_threshold(0.25)
        .with_iou_threshold(0.7)
        .build()?;

    // Get the model output tensors from inference
    let model_output: Vec<TensorDyn> = vec![/* tensors from inference */];
    let tensor_refs: Vec<&TensorDyn> = model_output.iter().collect();

    let mut output_boxes = Vec::with_capacity(10);
    let mut output_masks = Vec::with_capacity(10);

    // Decode model output into detection boxes and segmentation masks
    decoder.decode(&tensor_refs, &mut output_boxes, &mut output_masks)?;
    Ok(())
}
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
pub mod schema;
pub mod yolo;

mod decoder;
pub use decoder::*;

pub use configs::{DecoderVersion, Nms};
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
    /// 3D segmentation array of shape `(H, W, C)`.
    ///
    /// For instance segmentation (e.g. YOLO): `C=1` — per-instance mask with
    /// continuous sigmoid confidence values quantized to u8 (0 = background,
    /// 255 = full confidence). Renderers typically threshold at 128 (sigmoid
    /// 0.5) or use smooth interpolation for anti-aliased edges.
    ///
    /// For semantic segmentation (e.g. ModelPack): `C=num_classes` — per-pixel
    /// class scores where the object class is the argmax index.
    pub segmentation: Array3<u8>,
}

/// Prototype tensor variants for fused decode+render pipelines.
///
/// Carries either raw quantized data (to skip CPU dequantization and let the
/// GPU shader dequantize) or dequantized f32 data (from float models or legacy
/// paths).
#[derive(Debug, Clone)]
pub enum ProtoTensor {
    /// Raw int8 protos with quantization parameters — skip CPU dequantization.
    /// The GPU fragment shader will dequantize per-texel using the scale and
    /// zero_point.
    Quantized {
        protos: Array3<i8>,
        quantization: Quantization,
    },
    /// Dequantized f32 protos (from float models or legacy path).
    Float(Array3<f32>),
}

impl ProtoTensor {
    /// Returns `true` if this is the quantized variant.
    pub fn is_quantized(&self) -> bool {
        matches!(self, ProtoTensor::Quantized { .. })
    }

    /// Returns the spatial dimensions `(height, width, num_protos)`.
    pub fn dim(&self) -> (usize, usize, usize) {
        match self {
            ProtoTensor::Quantized { protos, .. } => protos.dim(),
            ProtoTensor::Float(arr) => arr.dim(),
        }
    }

    /// Returns dequantized f32 protos. For the `Float` variant this is a
    /// no-copy reference; for `Quantized` it allocates and dequantizes.
    pub fn as_f32(&self) -> std::borrow::Cow<'_, Array3<f32>> {
        match self {
            ProtoTensor::Float(arr) => std::borrow::Cow::Borrowed(arr),
            ProtoTensor::Quantized {
                protos,
                quantization,
            } => {
                let scale = quantization.scale;
                let zp = quantization.zero_point as f32;
                std::borrow::Cow::Owned(protos.map(|&v| (v as f32 - zp) * scale))
            }
        }
    }
}

/// Raw prototype data for fused decode+render pipelines.
///
/// Holds post-NMS intermediate state before mask materialization, allowing the
/// renderer to compute `mask_coeff @ protos` directly (e.g. in a GPU fragment
/// shader) without materializing intermediate `Array3<u8>` masks.
#[derive(Debug, Clone)]
pub struct ProtoData {
    /// Mask coefficients per detection (each `Vec<f32>` has length `num_protos`).
    pub mask_coefficients: Vec<Vec<f32>>,
    /// Prototype tensor, shape `(proto_h, proto_w, num_protos)`.
    pub protos: ProtoTensor,
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
///
/// # Errors
///
/// Returns `DecoderError::InvalidShape` if the segmentation tensor has an
/// invalid shape.
///
/// # Examples
/// ```
/// # use edgefirst_decoder::segmentation_to_mask;
/// let segmentation =
///     ndarray::Array3::<u8>::from_shape_vec((2, 2, 1), vec![0, 255, 128, 127]).unwrap();
/// let mask = segmentation_to_mask(segmentation.view()).unwrap();
/// assert_eq!(mask, ndarray::array![[0, 1], [1, 0]]);
/// ```
pub fn segmentation_to_mask(segmentation: ArrayView3<u8>) -> Result<Array2<u8>, DecoderError> {
    if segmentation.shape()[2] == 0 {
        return Err(DecoderError::InvalidShape(
            "Segmentation tensor must have non-zero depth".to_string(),
        ));
    }
    if segmentation.shape()[2] == 1 {
        yolo_segmentation_to_mask(segmentation, 128)
    } else {
        Ok(modelpack_segmentation_to_mask(segmentation))
    }
}

/// Returns the maximum value and its index from a 1D array
fn arg_max<T: PartialOrd + Copy>(score: ArrayView1<T>) -> (T, usize) {
    score
        .iter()
        .enumerate()
        .fold((score[0], 0), |(max, arg_max), (ind, s)| {
            if max > *s {
                (max, arg_max)
            } else {
                (*s, ind)
            }
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
    use edgefirst_tensor::{Tensor, TensorMapTrait, TensorTrait};
    use ndarray::Dimension;
    use ndarray::{array, s, Array2, Array3, Array4, Axis};
    use ndarray_stats::DeviationExt;
    use num_traits::{AsPrimitive, PrimInt};

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

    // ─── Shared test data loaders ────────────────────────

    fn load_yolov8_boxes() -> Array3<i8> {
        let raw = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_boxes_116x8400.bin"
        ));
        let raw = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len()) };
        Array3::from_shape_vec((1, 116, 8400), raw.to_vec()).unwrap()
    }

    fn load_yolov8_protos() -> Array4<i8> {
        let raw = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_protos_160x160x32.bin"
        ));
        let raw = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len()) };
        Array4::from_shape_vec((1, 160, 160, 32), raw.to_vec()).unwrap()
    }

    fn load_yolov8s_det() -> Array3<i8> {
        let raw = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8s_80_classes.bin"
        ));
        let raw = unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i8, raw.len()) };
        Array3::from_shape_vec((1, 84, 8400), raw.to_vec()).unwrap()
    }

    #[test]
    fn test_decoder_modelpack() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let boxes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_boxes_1935x1x4.bin"
        ));
        let boxes = ndarray::Array4::from_shape_vec((1, 1935, 1, 4), boxes.to_vec()).unwrap();

        let scores = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_scores_1935x1.bin"
        ));
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
                        (DimName::BoxCoords, 4),
                    ],
                    normalized: Some(true),
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
        let detect0 = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_split_9x15x18.bin"
        ));
        let detect0 = ndarray::Array4::from_shape_vec((1, 9, 15, 18), detect0.to_vec()).unwrap();

        let detect1 = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_split_17x30x18.bin"
        ));
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
            normalized: Some(true),
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
            normalized: Some(true),
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
        let detect0 = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_split_9x15x18.bin"
        ));
        let detect0 = ndarray::Array4::from_shape_vec((1, 9, 15, 18), detect0.to_vec()).unwrap();

        let detect1 = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_split_17x30x18.bin"
        ));
        let detect1 = ndarray::Array4::from_shape_vec((1, 17, 30, 18), detect1.to_vec()).unwrap();

        let decoder = DecoderBuilder::default()
            .with_config_yaml_str(
                include_str!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/modelpack_split.yaml"
                ))
                .to_string(),
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
        let out = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_seg_2x160x160.bin"
        ));
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
        let mask0 = segmentation_to_mask(mask[0].segmentation.view()).unwrap();
        let mask1 = segmentation_to_mask(output_masks[0].segmentation.view()).unwrap();

        assert_eq!(mask0, mask1);
    }
    #[test]
    fn test_modelpack_seg_quant() {
        let out = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_seg_2x160x160.bin"
        ));
        let out_u8 = ndarray::Array4::from_shape_vec((1, 2, 160, 160), out.to_vec()).unwrap();
        let out_i8 = out_u8.mapv(|x| (x as i16 - 128) as i8);
        let out_u16 = out_u8.mapv(|x| (x as u16) << 8);
        let out_i16 = out_u8.mapv(|x| (((x as i32) << 8) - 32768) as i16);
        let out_u32 = out_u8.mapv(|x| (x as u32) << 24);
        let out_i32 = out_u8.mapv(|x| (((x as i64) << 24) - 2147483648) as i32);

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
        let mut output_masks_u8: Vec<_> = Vec::with_capacity(10);
        decoder
            .decode_quantized(
                &[out_u8.view().into()],
                &mut output_boxes,
                &mut output_masks_u8,
            )
            .unwrap();

        let mut output_masks_i8: Vec<_> = Vec::with_capacity(10);
        decoder
            .decode_quantized(
                &[out_i8.view().into()],
                &mut output_boxes,
                &mut output_masks_i8,
            )
            .unwrap();

        let mut output_masks_u16: Vec<_> = Vec::with_capacity(10);
        decoder
            .decode_quantized(
                &[out_u16.view().into()],
                &mut output_boxes,
                &mut output_masks_u16,
            )
            .unwrap();

        let mut output_masks_i16: Vec<_> = Vec::with_capacity(10);
        decoder
            .decode_quantized(
                &[out_i16.view().into()],
                &mut output_boxes,
                &mut output_masks_i16,
            )
            .unwrap();

        let mut output_masks_u32: Vec<_> = Vec::with_capacity(10);
        decoder
            .decode_quantized(
                &[out_u32.view().into()],
                &mut output_boxes,
                &mut output_masks_u32,
            )
            .unwrap();

        let mut output_masks_i32: Vec<_> = Vec::with_capacity(10);
        decoder
            .decode_quantized(
                &[out_i32.view().into()],
                &mut output_boxes,
                &mut output_masks_i32,
            )
            .unwrap();

        compare_outputs((&[], &output_boxes), (&[], &[]));
        let mask_u8 = segmentation_to_mask(output_masks_u8[0].segmentation.view()).unwrap();
        let mask_i8 = segmentation_to_mask(output_masks_i8[0].segmentation.view()).unwrap();
        let mask_u16 = segmentation_to_mask(output_masks_u16[0].segmentation.view()).unwrap();
        let mask_i16 = segmentation_to_mask(output_masks_i16[0].segmentation.view()).unwrap();
        let mask_u32 = segmentation_to_mask(output_masks_u32[0].segmentation.view()).unwrap();
        let mask_i32 = segmentation_to_mask(output_masks_i32[0].segmentation.view()).unwrap();
        assert_eq!(mask_u8, mask_i8);
        assert_eq!(mask_u8, mask_u16);
        assert_eq!(mask_u8, mask_i16);
        assert_eq!(mask_u8, mask_u32);
        assert_eq!(mask_u8, mask_i32);
    }

    #[test]
    fn test_modelpack_segdet() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;

        let boxes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_boxes_1935x1x4.bin"
        ));
        let boxes = Array4::from_shape_vec((1, 1935, 1, 4), boxes.to_vec()).unwrap();

        let scores = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_scores_1935x1.bin"
        ));
        let scores = Array3::from_shape_vec((1, 1935, 1), scores.to_vec()).unwrap();

        let seg = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_seg_2x160x160.bin"
        ));
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
                        (DimName::BoxCoords, 4),
                    ],
                    normalized: Some(true),
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
        let mask0 = segmentation_to_mask(mask[0].segmentation.view()).unwrap();
        let mask1 = segmentation_to_mask(output_masks[0].segmentation.view()).unwrap();

        assert_eq!(mask0, mask1);
    }

    #[test]
    fn test_modelpack_segdet_split() {
        let score_threshold = 0.8;
        let iou_threshold = 0.5;

        let seg = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_seg_2x160x160.bin"
        ));
        let seg = ndarray::Array4::from_shape_vec((1, 2, 160, 160), seg.to_vec()).unwrap();

        let detect0 = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_split_9x15x18.bin"
        ));
        let detect0 = ndarray::Array4::from_shape_vec((1, 9, 15, 18), detect0.to_vec()).unwrap();

        let detect1 = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/modelpack_split_17x30x18.bin"
        ));
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
                        normalized: Some(true),
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
                        normalized: Some(true),
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
        let mask0 = segmentation_to_mask(mask[0].segmentation.view()).unwrap();
        let mask1 = segmentation_to_mask(output_masks[0].segmentation.view()).unwrap();

        assert_eq!(mask0, mask1);
    }

    #[test]
    fn test_dequant_chunked() {
        let mut out = load_yolov8s_det().into_raw_vec_and_offset().0;
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
    fn test_dequant_ground_truth() {
        // Formula: output = (input - zero_point) * scale
        // Verify both dequantize_cpu and dequantize_cpu_chunked against hand-computed values.

        // Case 1: scale=0.1, zero_point=-128 (from doc example)
        let quant = Quantization::new(0.1, -128);
        let input: Vec<i8> = vec![0, 127, -128, 64];
        let mut output = vec![0.0f32; 4];
        let mut output_chunked = vec![0.0f32; 4];
        dequantize_cpu(&input, quant, &mut output);
        dequantize_cpu_chunked(&input, quant, &mut output_chunked);
        // (0 - (-128)) * 0.1 = 12.8
        // (127 - (-128)) * 0.1 = 25.5
        // (-128 - (-128)) * 0.1 = 0.0
        // (64 - (-128)) * 0.1 = 19.2
        let expected: Vec<f32> = vec![12.8, 25.5, 0.0, 19.2];
        for (i, (&out, &exp)) in output.iter().zip(expected.iter()).enumerate() {
            assert!((out - exp).abs() < 1e-5, "cpu[{i}]: {out} != {exp}");
        }
        for (i, (&out, &exp)) in output_chunked.iter().zip(expected.iter()).enumerate() {
            assert!((out - exp).abs() < 1e-5, "chunked[{i}]: {out} != {exp}");
        }

        // Case 2: scale=1.0, zero_point=0 (identity-like)
        let quant = Quantization::new(1.0, 0);
        dequantize_cpu(&input, quant, &mut output);
        dequantize_cpu_chunked(&input, quant, &mut output_chunked);
        let expected: Vec<f32> = vec![0.0, 127.0, -128.0, 64.0];
        assert_eq!(output, expected);
        assert_eq!(output_chunked, expected);

        // Case 3: scale=0.5, zero_point=0
        let quant = Quantization::new(0.5, 0);
        dequantize_cpu(&input, quant, &mut output);
        dequantize_cpu_chunked(&input, quant, &mut output_chunked);
        let expected: Vec<f32> = vec![0.0, 63.5, -64.0, 32.0];
        assert_eq!(output, expected);
        assert_eq!(output_chunked, expected);

        // Case 4: i8 min/max boundaries with typical quantization params
        let quant = Quantization::new(0.021287762, 31);
        let input: Vec<i8> = vec![-128, -1, 0, 1, 31, 127];
        let mut output = vec![0.0f32; 6];
        let mut output_chunked = vec![0.0f32; 6];
        dequantize_cpu(&input, quant, &mut output);
        dequantize_cpu_chunked(&input, quant, &mut output_chunked);
        for i in 0..6 {
            let expected = (input[i] as f32 - 31.0) * 0.021287762;
            assert!(
                (output[i] - expected).abs() < 1e-5,
                "cpu[{i}]: {} != {expected}",
                output[i]
            );
            assert!(
                (output_chunked[i] - expected).abs() < 1e-5,
                "chunked[{i}]: {} != {expected}",
                output_chunked[i]
            );
        }
    }

    #[test]
    fn test_decoder_yolo_det() {
        let score_threshold = 0.25;
        let iou_threshold = 0.7;
        let out = load_yolov8s_det();
        let quant = (0.0040811873, -123).into();

        let decoder = DecoderBuilder::default()
            .with_config_yolo_det(
                configs::Detection {
                    decoder: DecoderType::Ultralytics,
                    shape: vec![1, 84, 8400],
                    anchors: None,
                    quantization: Some(quant),
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::NumFeatures, 84),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
                },
                Some(DecoderVersion::Yolo11),
            )
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .build()
            .unwrap();

        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        decode_yolo_det(
            (out.slice(s![0, .., ..]), quant.into()),
            score_threshold,
            iou_threshold,
            Some(configs::Nms::ClassAgnostic),
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
        let boxes = load_yolov8_boxes();
        let quant_boxes = Quantization::new(0.021287761628627777, 31);

        let protos = load_yolov8_protos();
        let quant_protos = Quantization::new(0.02491161972284317, -117);
        let protos = dequantize_ndarray::<_, _, f32>(protos.view(), quant_protos);
        let seg = dequantize_ndarray::<_, _, f32>(boxes.view(), quant_boxes);
        let mut output_boxes: Vec<_> = Vec::with_capacity(10);
        let mut output_masks: Vec<_> = Vec::with_capacity(10);
        decode_yolo_segdet_float(
            seg.slice(s![0, .., ..]),
            protos.slice(s![0, .., .., ..]),
            score_threshold,
            iou_threshold,
            Some(configs::Nms::ClassAgnostic),
            &mut output_boxes,
            &mut output_masks,
        )
        .unwrap();
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

        let full_mask = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_mask_results.bin"
        ));
        let full_mask = ndarray::Array2::from_shape_vec((160, 160), full_mask.to_vec()).unwrap();

        let cropped_mask = full_mask.slice(ndarray::s![
            (output_masks[1].ymin * 160.0) as usize..(output_masks[1].ymax * 160.0) as usize,
            (output_masks[1].xmin * 160.0) as usize..(output_masks[1].xmax * 160.0) as usize,
        ]);

        assert_eq!(
            cropped_mask,
            segmentation_to_mask(output_masks[1].segmentation.view()).unwrap()
        );
    }

    /// Regression test: config-driven path with NCHW protos (no dshape).
    /// Simulates YOLOv8-seg ONNX outputs where protos are (1, 32, 160, 160)
    /// and the YAML config has no dshape field — the exact scenario from
    /// hal_mask_matmul_bug.md.
    #[test]
    fn test_decoder_masks_nchw_protos() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;

        // Load test data — boxes as [116, 8400]
        let boxes_2d = load_yolov8_boxes().slice_move(s![0, .., ..]);
        let quant_boxes = Quantization::new(0.021287761628627777, 31);

        // Load protos as HWC [160, 160, 32] (file layout) then dequantize
        let protos_hwc = load_yolov8_protos().slice_move(s![0, .., .., ..]);
        let quant_protos = Quantization::new(0.02491161972284317, -117);
        let protos_f32_hwc = dequantize_ndarray::<_, _, f32>(protos_hwc.view(), quant_protos);

        // ---- Reference: direct call with HWC protos (known working) ----
        let seg = dequantize_ndarray::<_, _, f32>(boxes_2d.view(), quant_boxes);
        let mut ref_boxes: Vec<_> = Vec::with_capacity(10);
        let mut ref_masks: Vec<_> = Vec::with_capacity(10);
        decode_yolo_segdet_float(
            seg.view(),
            protos_f32_hwc.view(),
            score_threshold,
            iou_threshold,
            Some(configs::Nms::ClassAgnostic),
            &mut ref_boxes,
            &mut ref_masks,
        )
        .unwrap();
        assert_eq!(ref_boxes.len(), 2);

        // ---- Config-driven path: NCHW protos, no dshape ----
        // Permute protos to NCHW [1, 32, 160, 160] as an ONNX model would output
        let protos_f32_chw = protos_f32_hwc.permuted_axes([2, 0, 1]); // [32, 160, 160]
        let protos_nchw = protos_f32_chw.insert_axis(ndarray::Axis(0)); // [1, 32, 160, 160]

        // Build boxes as [1, 116, 8400] f32
        let seg_3d = seg.insert_axis(ndarray::Axis(0)); // [1, 116, 8400]

        // Build decoder from config with no dshape on protos
        let decoder = DecoderBuilder::default()
            .with_config_yolo_segdet(
                configs::Detection {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: None,
                    shape: vec![1, 116, 8400],
                    dshape: vec![],
                    normalized: Some(true),
                    anchors: None,
                },
                configs::Protos {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: None,
                    shape: vec![1, 32, 160, 160],
                    dshape: vec![], // No dshape — simulates YAML without dshape
                },
                None, // decoder version
            )
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .build()
            .unwrap();

        let mut cfg_boxes: Vec<_> = Vec::with_capacity(10);
        let mut cfg_masks: Vec<_> = Vec::with_capacity(10);
        decoder
            .decode_float(
                &[seg_3d.view().into_dyn(), protos_nchw.view().into_dyn()],
                &mut cfg_boxes,
                &mut cfg_masks,
            )
            .unwrap();

        // Must produce the same number of detections
        assert_eq!(
            cfg_boxes.len(),
            ref_boxes.len(),
            "config path produced {} boxes, reference produced {}",
            cfg_boxes.len(),
            ref_boxes.len()
        );

        // Boxes must match
        for (i, (cb, rb)) in cfg_boxes.iter().zip(&ref_boxes).enumerate() {
            assert!(
                cb.equal_within_delta(rb, 0.01),
                "box {i} mismatch: config={cb:?}, reference={rb:?}"
            );
        }

        // Masks must match pixel-for-pixel
        for (i, (cm, rm)) in cfg_masks.iter().zip(&ref_masks).enumerate() {
            let cm_arr = segmentation_to_mask(cm.segmentation.view()).unwrap();
            let rm_arr = segmentation_to_mask(rm.segmentation.view()).unwrap();
            assert_eq!(
                cm_arr, rm_arr,
                "mask {i} pixel mismatch between config-driven and reference paths"
            );
        }
    }

    #[test]
    fn test_decoder_masks_i8() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let boxes = load_yolov8_boxes();
        let quant_boxes = (0.021287761628627777, 31).into();

        let protos = load_yolov8_protos();
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
                    normalized: Some(true),
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
                Some(DecoderVersion::Yolo11),
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
            Some(configs::Nms::ClassAgnostic),
            &mut output_boxes,
            &mut output_masks,
        )
        .unwrap();

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
            Some(configs::Nms::ClassAgnostic),
            &mut output_boxes_f32,
            &mut output_masks_f32,
        )
        .unwrap();

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
        let boxes = load_yolov8_boxes();
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
                        (DimName::BoxCoords, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
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
            Some(configs::Nms::ClassAgnostic),
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
        let boxes_raw = load_yolov8_boxes();
        let boxes: Vec<_> = boxes_raw.iter().map(|x| *x as i16 * 256).collect();
        let boxes = ndarray::Array3::from_shape_vec((1, 116, 8400), boxes).unwrap();

        let quant_boxes = (0.021287761628627777 / 256.0, 31 * 256);

        let protos = load_yolov8_protos();
        let quant_protos = (0.02491161972284317, -117);

        let decoder = build_yolo_split_segdet_decoder(
            score_threshold,
            iou_threshold,
            quant_boxes,
            quant_protos,
        );
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

        let protos = dequantize_ndarray::<_, _, f32>(protos.view(), quant_protos.into());
        let seg = dequantize_ndarray::<_, _, f32>(boxes.view(), quant_boxes.into());
        let mut output_boxes_f32: Vec<_> = Vec::with_capacity(500);
        let mut output_masks_f32: Vec<_> = Vec::with_capacity(500);
        decode_yolo_segdet_float(
            seg.slice(s![0, .., ..]),
            protos.slice(s![0, .., .., ..]),
            score_threshold,
            iou_threshold,
            Some(configs::Nms::ClassAgnostic),
            &mut output_boxes_f32,
            &mut output_masks_f32,
        )
        .unwrap();

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

    fn build_yolo_split_segdet_decoder(
        score_threshold: f32,
        iou_threshold: f32,
        quant_boxes: (f32, i32),
        quant_protos: (f32, i32),
    ) -> crate::Decoder {
        DecoderBuilder::default()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(quant_boxes.into()),
                    shape: vec![1, 4, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
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
            .build()
            .unwrap()
    }

    fn build_yolov8_seg_decoder(score_threshold: f32, iou_threshold: f32) -> crate::Decoder {
        let config_yaml = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_seg.yaml"
        ));
        DecoderBuilder::default()
            .with_config_yaml_str(config_yaml.to_string())
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .build()
            .unwrap()
    }
    #[test]
    fn test_decoder_masks_config_i32() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;
        let boxes_raw = load_yolov8_boxes();
        let scale = 1 << 23;
        let boxes: Vec<_> = boxes_raw.iter().map(|x| *x as i32 * scale).collect();
        let boxes = ndarray::Array3::from_shape_vec((1, 116, 8400), boxes).unwrap();

        let quant_boxes = (0.021287761628627777 / scale as f32, 31 * scale);

        let protos_raw = load_yolov8_protos();
        let protos: Vec<_> = protos_raw.iter().map(|x| *x as i32 * scale).collect();
        let protos = ndarray::Array4::from_shape_vec((1, 160, 160, 32), protos).unwrap();
        let quant_protos = (0.02491161972284317 / scale as f32, -117 * scale);

        let decoder = build_yolo_split_segdet_decoder(
            score_threshold,
            iou_threshold,
            quant_boxes,
            quant_protos,
        );

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

        let protos = dequantize_ndarray::<_, _, f32>(protos.view(), quant_protos.into());
        let seg = dequantize_ndarray::<_, _, f32>(boxes.view(), quant_boxes.into());
        let mut output_boxes_f32: Vec<_> = Vec::with_capacity(500);
        let mut output_masks_f32: Vec<Segmentation> = Vec::with_capacity(500);
        decode_yolo_segdet_float(
            seg.slice(s![0, .., ..]),
            protos.slice(s![0, .., .., ..]),
            score_threshold,
            iou_threshold,
            Some(configs::Nms::ClassAgnostic),
            &mut output_boxes_f32,
            &mut output_masks_f32,
        )
        .unwrap();

        assert_eq!(output_boxes.len(), output_boxes_f32.len());
        assert_eq!(output_masks.len(), output_masks_f32.len());

        compare_outputs(
            (&output_boxes, &output_boxes_f32),
            (&output_masks, &output_masks_f32),
        );
    }

    /// test running multiple decoders concurrently
    #[test]
    fn test_context_switch() {
        let yolo_det = || {
            let score_threshold = 0.25;
            let iou_threshold = 0.7;
            let out = load_yolov8s_det();
            let quant = (0.0040811873, -123).into();

            let decoder = DecoderBuilder::default()
                .with_config_yolo_det(
                    configs::Detection {
                        decoder: DecoderType::Ultralytics,
                        shape: vec![1, 84, 8400],
                        anchors: None,
                        quantization: Some(quant),
                        dshape: vec![
                            (DimName::Batch, 1),
                            (DimName::NumFeatures, 84),
                            (DimName::NumBoxes, 8400),
                        ],
                        normalized: None,
                    },
                    None,
                )
                .with_score_threshold(score_threshold)
                .with_iou_threshold(iou_threshold)
                .build()
                .unwrap();

            let mut output_boxes: Vec<_> = Vec::with_capacity(50);
            let mut output_masks: Vec<_> = Vec::with_capacity(50);

            for _ in 0..100 {
                decoder
                    .decode_quantized(&[out.view().into()], &mut output_boxes, &mut output_masks)
                    .unwrap();

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
                assert!(output_masks.is_empty());
            }
        };

        let modelpack_det_split = || {
            let score_threshold = 0.8;
            let iou_threshold = 0.5;

            let seg = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/modelpack_seg_2x160x160.bin"
            ));
            let seg = ndarray::Array4::from_shape_vec((1, 2, 160, 160), seg.to_vec()).unwrap();

            let detect0 = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/modelpack_split_9x15x18.bin"
            ));
            let detect0 =
                ndarray::Array4::from_shape_vec((1, 9, 15, 18), detect0.to_vec()).unwrap();

            let detect1 = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/modelpack_split_17x30x18.bin"
            ));
            let detect1 =
                ndarray::Array4::from_shape_vec((1, 17, 30, 18), detect1.to_vec()).unwrap();

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
                            normalized: None,
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
                            normalized: None,
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

            for _ in 0..100 {
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

                compare_outputs((&correct_boxes, &output_boxes), (&mask, &output_masks));
            }
        };

        let handles = vec![
            std::thread::spawn(yolo_det),
            std::thread::spawn(modelpack_det_split),
            std::thread::spawn(yolo_det),
            std::thread::spawn(modelpack_det_split),
            std::thread::spawn(yolo_det),
            std::thread::spawn(modelpack_det_split),
            std::thread::spawn(yolo_det),
            std::thread::spawn(modelpack_det_split),
        ];
        for handle in handles {
            handle.join().unwrap();
        }
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

    #[test]
    fn test_class_aware_nms_float() {
        use crate::float::nms_class_aware_float;

        // Create two overlapping boxes with different classes
        let boxes = vec![
            DetectBox {
                bbox: BoundingBox {
                    xmin: 0.0,
                    ymin: 0.0,
                    xmax: 0.5,
                    ymax: 0.5,
                },
                score: 0.9,
                label: 0, // class 0
            },
            DetectBox {
                bbox: BoundingBox {
                    xmin: 0.1,
                    ymin: 0.1,
                    xmax: 0.6,
                    ymax: 0.6,
                },
                score: 0.8,
                label: 1, // class 1 - different class
            },
        ];

        // Class-aware NMS should keep both boxes (different classes, IoU ~0.47 >
        // threshold 0.3)
        let result = nms_class_aware_float(0.3, boxes.clone());
        assert_eq!(
            result.len(),
            2,
            "Class-aware NMS should keep both boxes with different classes"
        );

        // Now test with same class - should suppress one
        let same_class_boxes = vec![
            DetectBox {
                bbox: BoundingBox {
                    xmin: 0.0,
                    ymin: 0.0,
                    xmax: 0.5,
                    ymax: 0.5,
                },
                score: 0.9,
                label: 0,
            },
            DetectBox {
                bbox: BoundingBox {
                    xmin: 0.1,
                    ymin: 0.1,
                    xmax: 0.6,
                    ymax: 0.6,
                },
                score: 0.8,
                label: 0, // same class
            },
        ];

        let result = nms_class_aware_float(0.3, same_class_boxes);
        assert_eq!(
            result.len(),
            1,
            "Class-aware NMS should suppress overlapping box with same class"
        );
        assert_eq!(result[0].label, 0);
        assert!((result[0].score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_class_agnostic_vs_aware_nms() {
        use crate::float::{nms_class_aware_float, nms_float};

        // Two overlapping boxes with different classes
        let boxes = vec![
            DetectBox {
                bbox: BoundingBox {
                    xmin: 0.0,
                    ymin: 0.0,
                    xmax: 0.5,
                    ymax: 0.5,
                },
                score: 0.9,
                label: 0,
            },
            DetectBox {
                bbox: BoundingBox {
                    xmin: 0.1,
                    ymin: 0.1,
                    xmax: 0.6,
                    ymax: 0.6,
                },
                score: 0.8,
                label: 1,
            },
        ];

        // Class-agnostic should suppress one (IoU ~0.47 > threshold 0.3)
        let agnostic_result = nms_float(0.3, boxes.clone());
        assert_eq!(
            agnostic_result.len(),
            1,
            "Class-agnostic NMS should suppress overlapping boxes"
        );

        // Class-aware should keep both (different classes)
        let aware_result = nms_class_aware_float(0.3, boxes);
        assert_eq!(
            aware_result.len(),
            2,
            "Class-aware NMS should keep boxes with different classes"
        );
    }

    #[test]
    fn test_class_aware_nms_int() {
        use crate::byte::nms_class_aware_int;

        // Create two overlapping boxes with different classes
        let boxes = vec![
            DetectBoxQuantized {
                bbox: BoundingBox {
                    xmin: 0.0,
                    ymin: 0.0,
                    xmax: 0.5,
                    ymax: 0.5,
                },
                score: 200_u8,
                label: 0,
            },
            DetectBoxQuantized {
                bbox: BoundingBox {
                    xmin: 0.1,
                    ymin: 0.1,
                    xmax: 0.6,
                    ymax: 0.6,
                },
                score: 180_u8,
                label: 1, // different class
            },
        ];

        // Should keep both (different classes)
        let result = nms_class_aware_int(0.5, boxes);
        assert_eq!(
            result.len(),
            2,
            "Class-aware NMS (int) should keep boxes with different classes"
        );
    }

    #[test]
    fn test_nms_enum_default() {
        // Test that Nms enum has the correct default
        let default_nms: configs::Nms = Default::default();
        assert_eq!(default_nms, configs::Nms::ClassAgnostic);
    }

    #[test]
    fn test_decoder_nms_mode() {
        // Test that decoder properly stores NMS mode
        let decoder = DecoderBuilder::default()
            .with_config_yolo_det(
                configs::Detection {
                    anchors: None,
                    decoder: DecoderType::Ultralytics,
                    quantization: None,
                    shape: vec![1, 84, 8400],
                    dshape: Vec::new(),
                    normalized: Some(true),
                },
                None,
            )
            .with_nms(Some(configs::Nms::ClassAware))
            .build()
            .unwrap();

        assert_eq!(decoder.nms, Some(configs::Nms::ClassAware));
    }

    #[test]
    fn test_decoder_nms_bypass() {
        // Test that decoder can be configured with nms=None (bypass)
        let decoder = DecoderBuilder::default()
            .with_config_yolo_det(
                configs::Detection {
                    anchors: None,
                    decoder: DecoderType::Ultralytics,
                    quantization: None,
                    shape: vec![1, 84, 8400],
                    dshape: Vec::new(),
                    normalized: Some(true),
                },
                None,
            )
            .with_nms(None)
            .build()
            .unwrap();

        assert_eq!(decoder.nms, None);
    }

    #[test]
    fn test_decoder_normalized_boxes_true() {
        // Test that normalized_boxes returns Some(true) when explicitly set
        let decoder = DecoderBuilder::default()
            .with_config_yolo_det(
                configs::Detection {
                    anchors: None,
                    decoder: DecoderType::Ultralytics,
                    quantization: None,
                    shape: vec![1, 84, 8400],
                    dshape: Vec::new(),
                    normalized: Some(true),
                },
                None,
            )
            .build()
            .unwrap();

        assert_eq!(decoder.normalized_boxes(), Some(true));
    }

    #[test]
    fn test_decoder_normalized_boxes_false() {
        // Test that normalized_boxes returns Some(false) when config specifies
        // unnormalized
        let decoder = DecoderBuilder::default()
            .with_config_yolo_det(
                configs::Detection {
                    anchors: None,
                    decoder: DecoderType::Ultralytics,
                    quantization: None,
                    shape: vec![1, 84, 8400],
                    dshape: Vec::new(),
                    normalized: Some(false),
                },
                None,
            )
            .build()
            .unwrap();

        assert_eq!(decoder.normalized_boxes(), Some(false));
    }

    #[test]
    fn test_decoder_normalized_boxes_unknown() {
        // Test that normalized_boxes returns None when not specified in config
        let decoder = DecoderBuilder::default()
            .with_config_yolo_det(
                configs::Detection {
                    anchors: None,
                    decoder: DecoderType::Ultralytics,
                    quantization: None,
                    shape: vec![1, 84, 8400],
                    dshape: Vec::new(),
                    normalized: None,
                },
                Some(DecoderVersion::Yolo11),
            )
            .build()
            .unwrap();

        assert_eq!(decoder.normalized_boxes(), None);
    }

    pub fn quantize_ndarray<T: PrimInt + 'static, D: Dimension, F: Float + AsPrimitive<T>>(
        input: ArrayView<F, D>,
        quant: Quantization,
    ) -> Array<T, D>
    where
        i32: num_traits::AsPrimitive<F>,
        f32: num_traits::AsPrimitive<F>,
    {
        let zero_point = quant.zero_point.as_();
        let div_scale = F::one() / quant.scale.as_();
        if zero_point != F::zero() {
            input.mapv(|d| (d * div_scale + zero_point).round().as_())
        } else {
            input.mapv(|d| (d * div_scale).round().as_())
        }
    }

    fn real_data_expected_boxes() -> [DetectBox; 2] {
        [
            DetectBox {
                bbox: BoundingBox {
                    xmin: 0.08515105,
                    ymin: 0.7131401,
                    xmax: 0.29802868,
                    ymax: 0.8195788,
                },
                score: 0.91537374,
                label: 23,
            },
            DetectBox {
                bbox: BoundingBox {
                    xmin: 0.59605736,
                    ymin: 0.25545314,
                    xmax: 0.93666154,
                    ymax: 0.72378385,
                },
                score: 0.91537374,
                label: 23,
            },
        ]
    }

    fn e2e_expected_boxes_quant() -> [DetectBox; 1] {
        [DetectBox {
            bbox: BoundingBox {
                xmin: 0.12549022,
                ymin: 0.12549022,
                xmax: 0.23529413,
                ymax: 0.23529413,
            },
            score: 0.98823535,
            label: 2,
        }]
    }

    fn e2e_expected_boxes_float() -> [DetectBox; 1] {
        [DetectBox {
            bbox: BoundingBox {
                xmin: 0.1234,
                ymin: 0.1234,
                xmax: 0.2345,
                ymax: 0.2345,
            },
            score: 0.9876,
            label: 2,
        }]
    }

    macro_rules! real_data_proto_test {
        ($name:ident, quantized, $layout:ident) => {
            #[test]
            fn $name() {
                let is_split = matches!(stringify!($layout), "split");

                let score_threshold = 0.45;
                let iou_threshold = 0.45;
                let quant_boxes = (0.021287762_f32, 31_i32);
                let quant_protos = (0.02491162_f32, -117_i32);

                let raw_boxes = include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/yolov8_boxes_116x8400.bin"
                ));
                let raw_boxes = unsafe {
                    std::slice::from_raw_parts(raw_boxes.as_ptr() as *const i8, raw_boxes.len())
                };
                let boxes_i8 =
                    ndarray::Array3::from_shape_vec((1, 116, 8400), raw_boxes.to_vec()).unwrap();

                let raw_protos = include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/yolov8_protos_160x160x32.bin"
                ));
                let raw_protos = unsafe {
                    std::slice::from_raw_parts(raw_protos.as_ptr() as *const i8, raw_protos.len())
                };
                let protos_i8 =
                    ndarray::Array4::from_shape_vec((1, 160, 160, 32), raw_protos.to_vec())
                        .unwrap();

                // Pre-split (unused for combined, but harmless)
                let mask_split = boxes_i8.slice(s![.., 84.., ..]).to_owned();
                let scores_split = boxes_i8.slice(s![.., 4..84, ..]).to_owned();
                let boxes_split = boxes_i8.slice(s![.., ..4, ..]).to_owned();
                let boxes_combined = boxes_i8;

                let decoder = if is_split {
                    build_yolo_split_segdet_decoder(
                        score_threshold,
                        iou_threshold,
                        quant_boxes,
                        quant_protos,
                    )
                } else {
                    build_yolov8_seg_decoder(score_threshold, iou_threshold)
                };

                let expected = real_data_expected_boxes();
                let mut output_boxes = Vec::with_capacity(50);

                let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> = if is_split {
                    vec![
                        boxes_split.view().into(),
                        scores_split.view().into(),
                        mask_split.view().into(),
                        protos_i8.view().into(),
                    ]
                } else {
                    vec![boxes_combined.view().into(), protos_i8.view().into()]
                };
                decoder
                    .decode_quantized_proto(&inputs, &mut output_boxes)
                    .unwrap();

                assert_eq!(output_boxes.len(), 2);
                assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
                assert!(output_boxes[1].equal_within_delta(&expected[1], 1.0 / 160.0));
            }
        };
        ($name:ident, float, $layout:ident) => {
            #[test]
            fn $name() {
                let is_split = matches!(stringify!($layout), "split");

                let score_threshold = 0.45;
                let iou_threshold = 0.45;
                let quant_boxes = (0.021287762_f32, 31_i32);
                let quant_protos = (0.02491162_f32, -117_i32);

                let raw_boxes = include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/yolov8_boxes_116x8400.bin"
                ));
                let raw_boxes = unsafe {
                    std::slice::from_raw_parts(raw_boxes.as_ptr() as *const i8, raw_boxes.len())
                };
                let boxes_i8 =
                    ndarray::Array3::from_shape_vec((1, 116, 8400), raw_boxes.to_vec()).unwrap();
                let boxes_f32: Array3<f32> =
                    dequantize_ndarray(boxes_i8.view(), quant_boxes.into());

                let raw_protos = include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/yolov8_protos_160x160x32.bin"
                ));
                let raw_protos = unsafe {
                    std::slice::from_raw_parts(raw_protos.as_ptr() as *const i8, raw_protos.len())
                };
                let protos_i8 =
                    ndarray::Array4::from_shape_vec((1, 160, 160, 32), raw_protos.to_vec())
                        .unwrap();
                let protos_f32: Array4<f32> =
                    dequantize_ndarray(protos_i8.view(), quant_protos.into());

                // Pre-split from dequantized data
                let mask_split = boxes_f32.slice(s![.., 84.., ..]).to_owned();
                let scores_split = boxes_f32.slice(s![.., 4..84, ..]).to_owned();
                let boxes_split = boxes_f32.slice(s![.., ..4, ..]).to_owned();
                let boxes_combined = boxes_f32;

                let decoder = if is_split {
                    build_yolo_split_segdet_decoder(
                        score_threshold,
                        iou_threshold,
                        quant_boxes,
                        quant_protos,
                    )
                } else {
                    build_yolov8_seg_decoder(score_threshold, iou_threshold)
                };

                let expected = real_data_expected_boxes();
                let mut output_boxes = Vec::with_capacity(50);

                let inputs = if is_split {
                    vec![
                        boxes_split.view().into_dyn(),
                        scores_split.view().into_dyn(),
                        mask_split.view().into_dyn(),
                        protos_f32.view().into_dyn(),
                    ]
                } else {
                    vec![
                        boxes_combined.view().into_dyn(),
                        protos_f32.view().into_dyn(),
                    ]
                };
                decoder
                    .decode_float_proto(&inputs, &mut output_boxes)
                    .unwrap();

                assert_eq!(output_boxes.len(), 2);
                assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
                assert!(output_boxes[1].equal_within_delta(&expected[1], 1.0 / 160.0));
            }
        };
    }

    real_data_proto_test!(test_decoder_segdet_proto, quantized, combined);
    real_data_proto_test!(test_decoder_segdet_proto_float, float, combined);
    real_data_proto_test!(test_decoder_segdet_split_proto, quantized, split);
    real_data_proto_test!(test_decoder_segdet_split_proto_float, float, split);

    const E2E_COMBINED_DET_CONFIG: &str = "
decoder_version: yolo26
outputs:
 - type: detection
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 6]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_features, 6]
   normalized: true
";

    const E2E_COMBINED_SEGDET_CONFIG: &str = "
decoder_version: yolo26
outputs:
 - type: detection
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 38]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_features, 38]
   normalized: true
 - type: protos
   decoder: ultralytics
   quantization: [0.0039215686274509803921568627451, 128]
   shape: [1, 160, 160, 32]
   dshape:
    - [batch, 1]
    - [height, 160]
    - [width, 160]
    - [num_protos, 32]
";

    const E2E_SPLIT_DET_CONFIG: &str = "
decoder_version: yolo26
outputs:
 - type: boxes
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 4]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [box_coords, 4]
   normalized: true
 - type: scores
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 1]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_classes, 1]
 - type: classes
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 1]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_classes, 1]
";

    const E2E_SPLIT_SEGDET_CONFIG: &str = "
decoder_version: yolo26
outputs:
 - type: boxes
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 4]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [box_coords, 4]
   normalized: true
 - type: scores
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 1]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_classes, 1]
 - type: classes
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 1]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_classes, 1]
 - type: mask_coefficients
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 32]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_protos, 32]
 - type: protos
   decoder: ultralytics
   quantization: [0.0039215686274509803921568627451, 128]
   shape: [1, 160, 160, 32]
   dshape:
    - [batch, 1]
    - [height, 160]
    - [width, 160]
    - [num_protos, 32]
";

    macro_rules! e2e_segdet_test {
        ($name:ident, quantized, $layout:ident, $output:ident) => {
            #[test]
            fn $name() {
                let is_split = matches!(stringify!($layout), "split");
                let is_proto = matches!(stringify!($output), "proto");

                let score_threshold = 0.45;
                let iou_threshold = 0.45;

                let mut boxes = Array2::zeros((10, 4));
                let mut scores = Array2::zeros((10, 1));
                let mut classes = Array2::zeros((10, 1));
                let mask = Array2::zeros((10, 32));
                let protos = Array3::<f64>::zeros((160, 160, 32));
                let protos = protos.insert_axis(Axis(0));
                let protos_quant = (1.0 / 255.0, 0.0);
                let protos: Array4<u8> = quantize_ndarray(protos.view(), protos_quant.into());

                boxes
                    .slice_mut(s![0, ..])
                    .assign(&array![0.1234, 0.1234, 0.2345, 0.2345]);
                scores.slice_mut(s![0, ..]).assign(&array![0.9876]);
                classes.slice_mut(s![0, ..]).assign(&array![2.0]);

                let detect_quant = (2.0 / 255.0, 0.0);

                let decoder = if is_split {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_SPLIT_SEGDET_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                } else {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_COMBINED_SEGDET_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                };

                let expected = e2e_expected_boxes_quant();
                let mut output_boxes = Vec::with_capacity(50);

                if is_split {
                    let boxes = boxes.insert_axis(Axis(0));
                    let scores = scores.insert_axis(Axis(0));
                    let classes = classes.insert_axis(Axis(0));
                    let mask = mask.insert_axis(Axis(0));

                    let boxes: Array3<u8> = quantize_ndarray(boxes.view(), detect_quant.into());
                    let scores: Array3<u8> = quantize_ndarray(scores.view(), detect_quant.into());
                    let classes: Array3<u8> = quantize_ndarray(classes.view(), detect_quant.into());
                    let mask: Array3<u8> = quantize_ndarray(mask.view(), detect_quant.into());

                    if is_proto {
                        let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> = vec![
                            boxes.view().into(),
                            scores.view().into(),
                            classes.view().into(),
                            mask.view().into(),
                            protos.view().into(),
                        ];
                        decoder
                            .decode_quantized_proto(&inputs, &mut output_boxes)
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
                    } else {
                        let mut output_masks = Vec::with_capacity(50);
                        let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> = vec![
                            boxes.view().into(),
                            scores.view().into(),
                            classes.view().into(),
                            mask.view().into(),
                            protos.view().into(),
                        ];
                        decoder
                            .decode_quantized(&inputs, &mut output_boxes, &mut output_masks)
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
                    }
                } else {
                    // Combined layout
                    let detect = ndarray::concatenate![
                        Axis(1),
                        boxes.view(),
                        scores.view(),
                        classes.view(),
                        mask.view()
                    ];
                    let detect = detect.insert_axis(Axis(0));
                    assert_eq!(detect.shape(), &[1, 10, 38]);
                    let detect: Array3<u8> = quantize_ndarray(detect.view(), detect_quant.into());

                    if is_proto {
                        let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> =
                            vec![detect.view().into(), protos.view().into()];
                        decoder
                            .decode_quantized_proto(&inputs, &mut output_boxes)
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
                    } else {
                        let mut output_masks = Vec::with_capacity(50);
                        let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> =
                            vec![detect.view().into(), protos.view().into()];
                        decoder
                            .decode_quantized(&inputs, &mut output_boxes, &mut output_masks)
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
                    }
                }
            }
        };
        ($name:ident, float, $layout:ident, $output:ident) => {
            #[test]
            fn $name() {
                let is_split = matches!(stringify!($layout), "split");
                let is_proto = matches!(stringify!($output), "proto");

                let score_threshold = 0.45;
                let iou_threshold = 0.45;

                let mut boxes = Array2::zeros((10, 4));
                let mut scores = Array2::zeros((10, 1));
                let mut classes = Array2::zeros((10, 1));
                let mask: Array2<f64> = Array2::zeros((10, 32));
                let protos = Array3::<f64>::zeros((160, 160, 32));
                let protos = protos.insert_axis(Axis(0));

                boxes
                    .slice_mut(s![0, ..])
                    .assign(&array![0.1234, 0.1234, 0.2345, 0.2345]);
                scores.slice_mut(s![0, ..]).assign(&array![0.9876]);
                classes.slice_mut(s![0, ..]).assign(&array![2.0]);

                let decoder = if is_split {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_SPLIT_SEGDET_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                } else {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_COMBINED_SEGDET_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                };

                let expected = e2e_expected_boxes_float();
                let mut output_boxes = Vec::with_capacity(50);

                if is_split {
                    let boxes = boxes.insert_axis(Axis(0));
                    let scores = scores.insert_axis(Axis(0));
                    let classes = classes.insert_axis(Axis(0));
                    let mask = mask.insert_axis(Axis(0));

                    if is_proto {
                        let inputs = vec![
                            boxes.view().into_dyn(),
                            scores.view().into_dyn(),
                            classes.view().into_dyn(),
                            mask.view().into_dyn(),
                            protos.view().into_dyn(),
                        ];
                        decoder
                            .decode_float_proto(&inputs, &mut output_boxes)
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
                    } else {
                        let mut output_masks = Vec::with_capacity(50);
                        let inputs = vec![
                            boxes.view().into_dyn(),
                            scores.view().into_dyn(),
                            classes.view().into_dyn(),
                            mask.view().into_dyn(),
                            protos.view().into_dyn(),
                        ];
                        decoder
                            .decode_float(&inputs, &mut output_boxes, &mut output_masks)
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
                    }
                } else {
                    // Combined layout
                    let detect = ndarray::concatenate![
                        Axis(1),
                        boxes.view(),
                        scores.view(),
                        classes.view(),
                        mask.view()
                    ];
                    let detect = detect.insert_axis(Axis(0));
                    assert_eq!(detect.shape(), &[1, 10, 38]);

                    if is_proto {
                        let inputs = vec![detect.view().into_dyn(), protos.view().into_dyn()];
                        decoder
                            .decode_float_proto(&inputs, &mut output_boxes)
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
                    } else {
                        let mut output_masks = Vec::with_capacity(50);
                        let inputs = vec![detect.view().into_dyn(), protos.view().into_dyn()];
                        decoder
                            .decode_float(&inputs, &mut output_boxes, &mut output_masks)
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
                    }
                }
            }
        };
    }

    e2e_segdet_test!(test_decoder_end_to_end_segdet, quantized, combined, masks);
    e2e_segdet_test!(test_decoder_end_to_end_segdet_float, float, combined, masks);
    e2e_segdet_test!(
        test_decoder_end_to_end_segdet_proto,
        quantized,
        combined,
        proto
    );
    e2e_segdet_test!(
        test_decoder_end_to_end_segdet_proto_float,
        float,
        combined,
        proto
    );
    e2e_segdet_test!(
        test_decoder_end_to_end_segdet_split,
        quantized,
        split,
        masks
    );
    e2e_segdet_test!(
        test_decoder_end_to_end_segdet_split_float,
        float,
        split,
        masks
    );
    e2e_segdet_test!(
        test_decoder_end_to_end_segdet_split_proto,
        quantized,
        split,
        proto
    );
    e2e_segdet_test!(
        test_decoder_end_to_end_segdet_split_proto_float,
        float,
        split,
        proto
    );

    macro_rules! e2e_det_test {
        ($name:ident, quantized, $layout:ident) => {
            #[test]
            fn $name() {
                let is_split = matches!(stringify!($layout), "split");

                let score_threshold = 0.45;
                let iou_threshold = 0.45;

                let mut boxes = Array3::zeros((1, 10, 4));
                let mut scores = Array3::zeros((1, 10, 1));
                let mut classes = Array3::zeros((1, 10, 1));

                boxes
                    .slice_mut(s![0, 0, ..])
                    .assign(&array![0.1234, 0.1234, 0.2345, 0.2345]);
                scores.slice_mut(s![0, 0, ..]).assign(&array![0.9876]);
                classes.slice_mut(s![0, 0, ..]).assign(&array![2.0]);

                let detect_quant = (2.0 / 255.0, 0_i32);

                let decoder = if is_split {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_SPLIT_DET_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                } else {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_COMBINED_DET_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                };

                let expected = e2e_expected_boxes_quant();
                let mut output_boxes = Vec::with_capacity(50);

                if is_split {
                    let boxes: Array<u8, _> = quantize_ndarray(boxes.view(), detect_quant.into());
                    let scores: Array<u8, _> = quantize_ndarray(scores.view(), detect_quant.into());
                    let classes: Array<u8, _> =
                        quantize_ndarray(classes.view(), detect_quant.into());
                    let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> = vec![
                        boxes.view().into(),
                        scores.view().into(),
                        classes.view().into(),
                    ];
                    decoder
                        .decode_quantized(&inputs, &mut output_boxes, &mut Vec::new())
                        .unwrap();
                } else {
                    let detect =
                        ndarray::concatenate![Axis(2), boxes.view(), scores.view(), classes.view()];
                    assert_eq!(detect.shape(), &[1, 10, 6]);
                    let detect: Array3<u8> = quantize_ndarray(detect.view(), detect_quant.into());
                    let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> =
                        vec![detect.view().into()];
                    decoder
                        .decode_quantized(&inputs, &mut output_boxes, &mut Vec::new())
                        .unwrap();
                }

                assert_eq!(output_boxes.len(), 1);
                assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
            }
        };
        ($name:ident, float, $layout:ident) => {
            #[test]
            fn $name() {
                let is_split = matches!(stringify!($layout), "split");

                let score_threshold = 0.45;
                let iou_threshold = 0.45;

                let mut boxes = Array3::zeros((1, 10, 4));
                let mut scores = Array3::zeros((1, 10, 1));
                let mut classes = Array3::zeros((1, 10, 1));

                boxes
                    .slice_mut(s![0, 0, ..])
                    .assign(&array![0.1234, 0.1234, 0.2345, 0.2345]);
                scores.slice_mut(s![0, 0, ..]).assign(&array![0.9876]);
                classes.slice_mut(s![0, 0, ..]).assign(&array![2.0]);

                let decoder = if is_split {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_SPLIT_DET_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                } else {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_COMBINED_DET_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                };

                let expected = e2e_expected_boxes_float();
                let mut output_boxes = Vec::with_capacity(50);

                if is_split {
                    let inputs = vec![
                        boxes.view().into_dyn(),
                        scores.view().into_dyn(),
                        classes.view().into_dyn(),
                    ];
                    decoder
                        .decode_float(&inputs, &mut output_boxes, &mut Vec::new())
                        .unwrap();
                } else {
                    let detect =
                        ndarray::concatenate![Axis(2), boxes.view(), scores.view(), classes.view()];
                    assert_eq!(detect.shape(), &[1, 10, 6]);
                    let inputs = vec![detect.view().into_dyn()];
                    decoder
                        .decode_float(&inputs, &mut output_boxes, &mut Vec::new())
                        .unwrap();
                }

                assert_eq!(output_boxes.len(), 1);
                assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
            }
        };
    }

    e2e_det_test!(test_decoder_end_to_end_combined_det, quantized, combined);
    e2e_det_test!(test_decoder_end_to_end_combined_det_float, float, combined);
    e2e_det_test!(test_decoder_end_to_end_split_det, quantized, split);
    e2e_det_test!(test_decoder_end_to_end_split_det_float, float, split);

    #[test]
    fn test_decode_tensor() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;

        let raw_boxes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_boxes_116x8400.bin"
        ));
        let raw_boxes =
            unsafe { std::slice::from_raw_parts(raw_boxes.as_ptr() as *const i8, raw_boxes.len()) };
        let boxes_i8: Tensor<i8> = Tensor::new(&[1, 116, 8400], None, None).unwrap();
        boxes_i8
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(raw_boxes);
        let boxes_i8 = boxes_i8.into();

        let raw_protos = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_protos_160x160x32.bin"
        ));
        let raw_protos = unsafe {
            std::slice::from_raw_parts(raw_protos.as_ptr() as *const i8, raw_protos.len())
        };
        let protos_i8: Tensor<i8> = Tensor::new(&[1, 160, 160, 32], None, None).unwrap();
        protos_i8
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(raw_protos);
        let protos_i8 = protos_i8.into();

        let decoder = build_yolov8_seg_decoder(score_threshold, iou_threshold);
        let expected = real_data_expected_boxes();
        let mut output_boxes = Vec::with_capacity(50);

        decoder
            .decode(&[&boxes_i8, &protos_i8], &mut output_boxes, &mut Vec::new())
            .unwrap();

        assert_eq!(output_boxes.len(), 2);
        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
        assert!(output_boxes[1].equal_within_delta(&expected[1], 1.0 / 160.0));
    }

    #[test]
    fn test_decode_tensor_f32() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;

        let quant_boxes = (0.021287762_f32, 31_i32);
        let quant_protos = (0.02491162_f32, -117_i32);
        let raw_boxes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_boxes_116x8400.bin"
        ));
        let raw_boxes =
            unsafe { std::slice::from_raw_parts(raw_boxes.as_ptr() as *const i8, raw_boxes.len()) };
        let mut raw_boxes_f32 = vec![0f32; raw_boxes.len()];
        dequantize_cpu(raw_boxes, quant_boxes.into(), &mut raw_boxes_f32);
        let boxes_f32: Tensor<f32> = Tensor::new(&[1, 116, 8400], None, None).unwrap();
        boxes_f32
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&raw_boxes_f32);
        let boxes_f32 = boxes_f32.into();

        let raw_protos = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_protos_160x160x32.bin"
        ));
        let raw_protos = unsafe {
            std::slice::from_raw_parts(raw_protos.as_ptr() as *const i8, raw_protos.len())
        };
        let mut raw_protos_f32 = vec![0f32; raw_protos.len()];
        dequantize_cpu(raw_protos, quant_protos.into(), &mut raw_protos_f32);
        let protos_f32: Tensor<f32> = Tensor::new(&[1, 160, 160, 32], None, None).unwrap();
        protos_f32
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&raw_protos_f32);
        let protos_f32 = protos_f32.into();

        let decoder = build_yolov8_seg_decoder(score_threshold, iou_threshold);

        let expected = real_data_expected_boxes();
        let mut output_boxes = Vec::with_capacity(50);

        decoder
            .decode(
                &[&boxes_f32, &protos_f32],
                &mut output_boxes,
                &mut Vec::new(),
            )
            .unwrap();

        assert_eq!(output_boxes.len(), 2);
        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
        assert!(output_boxes[1].equal_within_delta(&expected[1], 1.0 / 160.0));
    }

    #[test]
    fn test_decode_tensor_f64() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;

        let quant_boxes = (0.021287762_f32, 31_i32);
        let quant_protos = (0.02491162_f32, -117_i32);
        let raw_boxes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_boxes_116x8400.bin"
        ));
        let raw_boxes =
            unsafe { std::slice::from_raw_parts(raw_boxes.as_ptr() as *const i8, raw_boxes.len()) };
        let mut raw_boxes_f64 = vec![0f64; raw_boxes.len()];
        dequantize_cpu(raw_boxes, quant_boxes.into(), &mut raw_boxes_f64);
        let boxes_f64: Tensor<f64> = Tensor::new(&[1, 116, 8400], None, None).unwrap();
        boxes_f64
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&raw_boxes_f64);
        let boxes_f64 = boxes_f64.into();

        let raw_protos = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_protos_160x160x32.bin"
        ));
        let raw_protos = unsafe {
            std::slice::from_raw_parts(raw_protos.as_ptr() as *const i8, raw_protos.len())
        };
        let mut raw_protos_f64 = vec![0f64; raw_protos.len()];
        dequantize_cpu(raw_protos, quant_protos.into(), &mut raw_protos_f64);
        let protos_f64: Tensor<f64> = Tensor::new(&[1, 160, 160, 32], None, None).unwrap();
        protos_f64
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(&raw_protos_f64);
        let protos_f64 = protos_f64.into();

        let decoder = build_yolov8_seg_decoder(score_threshold, iou_threshold);

        let expected = real_data_expected_boxes();
        let mut output_boxes = Vec::with_capacity(50);

        decoder
            .decode(
                &[&boxes_f64, &protos_f64],
                &mut output_boxes,
                &mut Vec::new(),
            )
            .unwrap();

        assert_eq!(output_boxes.len(), 2);
        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
        assert!(output_boxes[1].equal_within_delta(&expected[1], 1.0 / 160.0));
    }

    #[test]
    fn test_decode_tensor_proto() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;

        let raw_boxes = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_boxes_116x8400.bin"
        ));
        let raw_boxes =
            unsafe { std::slice::from_raw_parts(raw_boxes.as_ptr() as *const i8, raw_boxes.len()) };
        let boxes_i8: Tensor<i8> = Tensor::new(&[1, 116, 8400], None, None).unwrap();
        boxes_i8
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(raw_boxes);
        let boxes_i8 = boxes_i8.into();

        let raw_protos = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_protos_160x160x32.bin"
        ));
        let raw_protos = unsafe {
            std::slice::from_raw_parts(raw_protos.as_ptr() as *const i8, raw_protos.len())
        };
        let protos_i8: Tensor<i8> = Tensor::new(&[1, 160, 160, 32], None, None).unwrap();
        protos_i8
            .map()
            .unwrap()
            .as_mut_slice()
            .copy_from_slice(raw_protos);
        let protos_i8 = protos_i8.into();

        let decoder = build_yolov8_seg_decoder(score_threshold, iou_threshold);

        let expected = real_data_expected_boxes();
        let mut output_boxes = Vec::with_capacity(50);

        let proto_data = decoder
            .decode_proto(&[&boxes_i8, &protos_i8], &mut output_boxes)
            .unwrap();

        assert_eq!(output_boxes.len(), 2);
        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
        assert!(output_boxes[1].equal_within_delta(&expected[1], 1.0 / 160.0));

        let proto_data = proto_data.expect("segmentation model should return ProtoData");
        assert_eq!(
            proto_data.mask_coefficients.len(),
            output_boxes.len(),
            "mask_coefficients count must match detection count"
        );
        for coeff in &proto_data.mask_coefficients {
            assert_eq!(
                coeff.len(),
                32,
                "each detection should have 32 mask coefficients"
            );
        }
    }
}

#[cfg(feature = "tracker")]
#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod decoder_tracked_tests {

    use edgefirst_tracker::{ByteTrackBuilder, Tracker};
    use ndarray::{array, s, Array, Array2, Array3, Array4, ArrayView, Axis, Dimension};
    use num_traits::{AsPrimitive, Float, PrimInt};
    use rand::{RngExt, SeedableRng};
    use rand_distr::StandardNormal;

    use crate::{
        configs::{self, DimName},
        dequantize_ndarray, BoundingBox, DecoderBuilder, DetectBox, Quantization,
    };

    pub fn quantize_ndarray<T: PrimInt + 'static, D: Dimension, F: Float + AsPrimitive<T>>(
        input: ArrayView<F, D>,
        quant: Quantization,
    ) -> Array<T, D>
    where
        i32: num_traits::AsPrimitive<F>,
        f32: num_traits::AsPrimitive<F>,
    {
        let zero_point = quant.zero_point.as_();
        let div_scale = F::one() / quant.scale.as_();
        if zero_point != F::zero() {
            input.mapv(|d| (d * div_scale + zero_point).round().as_())
        } else {
            input.mapv(|d| (d * div_scale).round().as_())
        }
    }

    #[test]
    fn test_decoder_tracked_random_jitter() {
        use crate::configs::{DecoderType, Nms};
        use crate::DecoderBuilder;

        let score_threshold = 0.25;
        let iou_threshold = 0.1;
        let out = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8s_80_classes.bin"
        ));
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
        let out = Array3::from_shape_vec((1, 84, 8400), out.to_vec()).unwrap();
        let quant = (0.0040811873, -123).into();

        let decoder = DecoderBuilder::default()
            .with_config_yolo_det(
                crate::configs::Detection {
                    decoder: DecoderType::Ultralytics,
                    shape: vec![1, 84, 8400],
                    anchors: None,
                    quantization: Some(quant),
                    dshape: vec![
                        (crate::configs::DimName::Batch, 1),
                        (crate::configs::DimName::NumFeatures, 84),
                        (crate::configs::DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
                },
                None,
            )
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_nms(Some(Nms::ClassAgnostic))
            .build()
            .unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(0xAB_BEEF); // fixed seed for reproducibility

        let expected_boxes = [
            crate::DetectBox {
                bbox: crate::BoundingBox {
                    xmin: 0.5285137,
                    ymin: 0.05305544,
                    xmax: 0.87541467,
                    ymax: 0.9998909,
                },
                score: 0.5591227,
                label: 0,
            },
            crate::DetectBox {
                bbox: crate::BoundingBox {
                    xmin: 0.130598,
                    ymin: 0.43260583,
                    xmax: 0.35098213,
                    ymax: 0.9958097,
                },
                score: 0.33057618,
                label: 75,
            },
        ];

        let mut tracker = ByteTrackBuilder::new()
            .track_update(0.1)
            .track_high_conf(0.3)
            .build();

        let mut output_boxes = Vec::with_capacity(50);
        let mut output_masks = Vec::with_capacity(50);
        let mut output_tracks = Vec::with_capacity(50);

        decoder
            .decode_tracked_quantized(
                &mut tracker,
                0,
                &[out.view().into()],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
            )
            .unwrap();

        assert_eq!(output_boxes.len(), 2);
        assert!(output_boxes[0].equal_within_delta(&expected_boxes[0], 1e-6));
        assert!(output_boxes[1].equal_within_delta(&expected_boxes[1], 1e-6));

        let mut last_boxes = output_boxes.clone();

        for i in 1..=100 {
            let mut out = out.clone();
            // introduce jitter into the XY coordinates to simulate movement and test tracking stability
            let mut x_values = out.slice_mut(s![0, 0, ..]);
            for x in x_values.iter_mut() {
                let r: f32 = rng.sample(StandardNormal);
                let r = r.clamp(-2.0, 2.0) / 2.0;
                *x = x.saturating_add((r * 1e-2 / quant.0) as i8);
            }

            let mut y_values = out.slice_mut(s![0, 1, ..]);
            for y in y_values.iter_mut() {
                let r: f32 = rng.sample(StandardNormal);
                let r = r.clamp(-2.0, 2.0) / 2.0;
                *y = y.saturating_add((r * 1e-2 / quant.0) as i8);
            }

            decoder
                .decode_tracked_quantized(
                    &mut tracker,
                    100_000_000 * i / 3, // simulate 33.333ms between frames
                    &[out.view().into()],
                    &mut output_boxes,
                    &mut output_masks,
                    &mut output_tracks,
                )
                .unwrap();

            assert_eq!(output_boxes.len(), 2);
            assert!(output_boxes[0].equal_within_delta(&expected_boxes[0], 5e-3));
            assert!(output_boxes[1].equal_within_delta(&expected_boxes[1], 5e-3));

            assert!(output_boxes[0].equal_within_delta(&last_boxes[0], 1e-3));
            assert!(output_boxes[1].equal_within_delta(&last_boxes[1], 1e-3));
            last_boxes = output_boxes.clone();
        }
    }

    // ─── Shared helpers for tracked decoder tests ────────────────────

    fn real_data_expected_boxes() -> [DetectBox; 2] {
        [
            DetectBox {
                bbox: BoundingBox {
                    xmin: 0.08515105,
                    ymin: 0.7131401,
                    xmax: 0.29802868,
                    ymax: 0.8195788,
                },
                score: 0.91537374,
                label: 23,
            },
            DetectBox {
                bbox: BoundingBox {
                    xmin: 0.59605736,
                    ymin: 0.25545314,
                    xmax: 0.93666154,
                    ymax: 0.72378385,
                },
                score: 0.91537374,
                label: 23,
            },
        ]
    }

    fn e2e_expected_boxes_quant() -> [DetectBox; 1] {
        [DetectBox {
            bbox: BoundingBox {
                xmin: 0.12549022,
                ymin: 0.12549022,
                xmax: 0.23529413,
                ymax: 0.23529413,
            },
            score: 0.98823535,
            label: 2,
        }]
    }

    fn e2e_expected_boxes_float() -> [DetectBox; 1] {
        [DetectBox {
            bbox: BoundingBox {
                xmin: 0.1234,
                ymin: 0.1234,
                xmax: 0.2345,
                ymax: 0.2345,
            },
            score: 0.9876,
            label: 2,
        }]
    }

    fn build_yolo_split_segdet_decoder(
        score_threshold: f32,
        iou_threshold: f32,
        quant_boxes: (f32, i32),
        quant_protos: (f32, i32),
    ) -> crate::Decoder {
        DecoderBuilder::default()
            .with_config_yolo_split_segdet(
                configs::Boxes {
                    decoder: configs::DecoderType::Ultralytics,
                    quantization: Some(quant_boxes.into()),
                    shape: vec![1, 4, 8400],
                    dshape: vec![
                        (DimName::Batch, 1),
                        (DimName::BoxCoords, 4),
                        (DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
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
            .build()
            .unwrap()
    }

    fn build_yolov8_seg_decoder(score_threshold: f32, iou_threshold: f32) -> crate::Decoder {
        let config_yaml = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8_seg.yaml"
        ));
        DecoderBuilder::default()
            .with_config_yaml_str(config_yaml.to_string())
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .build()
            .unwrap()
    }

    // ─── Real-data tracked test macro ───────────────────────────────
    //
    // Generates tests that load i8 binary test data from testdata/ and
    // exercise all (quant/float) × (combined/split) × (masks/proto)
    // decoder paths.

    macro_rules! real_data_tracked_test {
        ($name:ident, quantized, $layout:ident, $output:ident) => {
            #[test]
            fn $name() {
                let is_split = matches!(stringify!($layout), "split");
                let is_proto = matches!(stringify!($output), "proto");

                let score_threshold = 0.45;
                let iou_threshold = 0.45;
                let quant_boxes = (0.021287762_f32, 31_i32);
                let quant_protos = (0.02491162_f32, -117_i32);

                let raw_boxes = include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/yolov8_boxes_116x8400.bin"
                ));
                let raw_boxes = unsafe {
                    std::slice::from_raw_parts(raw_boxes.as_ptr() as *const i8, raw_boxes.len())
                };
                let boxes_i8 =
                    ndarray::Array3::from_shape_vec((1, 116, 8400), raw_boxes.to_vec()).unwrap();

                let raw_protos = include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/yolov8_protos_160x160x32.bin"
                ));
                let raw_protos = unsafe {
                    std::slice::from_raw_parts(raw_protos.as_ptr() as *const i8, raw_protos.len())
                };
                let protos_i8 =
                    ndarray::Array4::from_shape_vec((1, 160, 160, 32), raw_protos.to_vec())
                        .unwrap();

                // Pre-split (unused for combined, but harmless)
                let mask_split = boxes_i8.slice(s![.., 84.., ..]).to_owned();
                let mut scores_split = boxes_i8.slice(s![.., 4..84, ..]).to_owned();
                let boxes_split = boxes_i8.slice(s![.., ..4, ..]).to_owned();
                let mut boxes_combined = boxes_i8;

                let decoder = if is_split {
                    build_yolo_split_segdet_decoder(
                        score_threshold,
                        iou_threshold,
                        quant_boxes,
                        quant_protos,
                    )
                } else {
                    build_yolov8_seg_decoder(score_threshold, iou_threshold)
                };

                let expected = real_data_expected_boxes();
                let mut tracker = ByteTrackBuilder::new()
                    .track_update(0.1)
                    .track_high_conf(0.7)
                    .build();
                let mut output_boxes = Vec::with_capacity(50);
                let mut output_tracks = Vec::with_capacity(50);

                // Frame 1: decode
                if is_proto {
                    {
                        let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> = if is_split {
                            vec![
                                boxes_split.view().into(),
                                scores_split.view().into(),
                                mask_split.view().into(),
                                protos_i8.view().into(),
                            ]
                        } else {
                            vec![boxes_combined.view().into(), protos_i8.view().into()]
                        };
                        decoder
                            .decode_tracked_quantized_proto(
                                &mut tracker,
                                0,
                                &inputs,
                                &mut output_boxes,
                                &mut output_tracks,
                            )
                            .unwrap();
                    }
                    assert_eq!(output_boxes.len(), 2);
                    assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
                    assert!(output_boxes[1].equal_within_delta(&expected[1], 1.0 / 160.0));

                    // Zero scores for frame 2
                    if is_split {
                        for score in scores_split.iter_mut() {
                            *score = i8::MIN;
                        }
                    } else {
                        for score in boxes_combined.slice_mut(s![0, 4..84, ..]).iter_mut() {
                            *score = i8::MIN;
                        }
                    }

                    let proto_result = {
                        let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> = if is_split {
                            vec![
                                boxes_split.view().into(),
                                scores_split.view().into(),
                                mask_split.view().into(),
                                protos_i8.view().into(),
                            ]
                        } else {
                            vec![boxes_combined.view().into(), protos_i8.view().into()]
                        };
                        decoder
                            .decode_tracked_quantized_proto(
                                &mut tracker,
                                100_000_000 / 3,
                                &inputs,
                                &mut output_boxes,
                                &mut output_tracks,
                            )
                            .unwrap()
                    };
                    assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                    assert!(output_boxes[1].equal_within_delta(&expected[1], 1e-6));
                    assert!(proto_result.is_some_and(|x| x.mask_coefficients.is_empty()));
                } else {
                    let mut output_masks = Vec::with_capacity(50);
                    {
                        let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> = if is_split {
                            vec![
                                boxes_split.view().into(),
                                scores_split.view().into(),
                                mask_split.view().into(),
                                protos_i8.view().into(),
                            ]
                        } else {
                            vec![boxes_combined.view().into(), protos_i8.view().into()]
                        };
                        decoder
                            .decode_tracked_quantized(
                                &mut tracker,
                                0,
                                &inputs,
                                &mut output_boxes,
                                &mut output_masks,
                                &mut output_tracks,
                            )
                            .unwrap();
                    }
                    assert_eq!(output_boxes.len(), 2);
                    assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
                    assert!(output_boxes[1].equal_within_delta(&expected[1], 1.0 / 160.0));

                    if is_split {
                        for score in scores_split.iter_mut() {
                            *score = i8::MIN;
                        }
                    } else {
                        for score in boxes_combined.slice_mut(s![0, 4..84, ..]).iter_mut() {
                            *score = i8::MIN;
                        }
                    }

                    {
                        let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> = if is_split {
                            vec![
                                boxes_split.view().into(),
                                scores_split.view().into(),
                                mask_split.view().into(),
                                protos_i8.view().into(),
                            ]
                        } else {
                            vec![boxes_combined.view().into(), protos_i8.view().into()]
                        };
                        decoder
                            .decode_tracked_quantized(
                                &mut tracker,
                                100_000_000 / 3,
                                &inputs,
                                &mut output_boxes,
                                &mut output_masks,
                                &mut output_tracks,
                            )
                            .unwrap();
                    }
                    assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                    assert!(output_boxes[1].equal_within_delta(&expected[1], 1e-6));
                    assert!(output_masks.is_empty());
                }
            }
        };
        ($name:ident, float, $layout:ident, $output:ident) => {
            #[test]
            fn $name() {
                let is_split = matches!(stringify!($layout), "split");
                let is_proto = matches!(stringify!($output), "proto");

                let score_threshold = 0.45;
                let iou_threshold = 0.45;
                let quant_boxes = (0.021287762_f32, 31_i32);
                let quant_protos = (0.02491162_f32, -117_i32);

                let raw_boxes = include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/yolov8_boxes_116x8400.bin"
                ));
                let raw_boxes = unsafe {
                    std::slice::from_raw_parts(raw_boxes.as_ptr() as *const i8, raw_boxes.len())
                };
                let boxes_i8 =
                    ndarray::Array3::from_shape_vec((1, 116, 8400), raw_boxes.to_vec()).unwrap();
                let boxes_f32 = dequantize_ndarray(boxes_i8.view(), quant_boxes.into());

                let raw_protos = include_bytes!(concat!(
                    env!("CARGO_MANIFEST_DIR"),
                    "/../../testdata/yolov8_protos_160x160x32.bin"
                ));
                let raw_protos = unsafe {
                    std::slice::from_raw_parts(raw_protos.as_ptr() as *const i8, raw_protos.len())
                };
                let protos_i8 =
                    ndarray::Array4::from_shape_vec((1, 160, 160, 32), raw_protos.to_vec())
                        .unwrap();
                let protos_f32 = dequantize_ndarray(protos_i8.view(), quant_protos.into());

                // Pre-split from dequantized data
                let mask_split = boxes_f32.slice(s![.., 84.., ..]).to_owned();
                let mut scores_split = boxes_f32.slice(s![.., 4..84, ..]).to_owned();
                let boxes_split = boxes_f32.slice(s![.., ..4, ..]).to_owned();
                let mut boxes_combined = boxes_f32;

                let decoder = if is_split {
                    build_yolo_split_segdet_decoder(
                        score_threshold,
                        iou_threshold,
                        quant_boxes,
                        quant_protos,
                    )
                } else {
                    build_yolov8_seg_decoder(score_threshold, iou_threshold)
                };

                let expected = real_data_expected_boxes();
                let mut tracker = ByteTrackBuilder::new()
                    .track_update(0.1)
                    .track_high_conf(0.7)
                    .build();
                let mut output_boxes = Vec::with_capacity(50);
                let mut output_tracks = Vec::with_capacity(50);

                if is_proto {
                    {
                        let inputs = if is_split {
                            vec![
                                boxes_split.view().into_dyn(),
                                scores_split.view().into_dyn(),
                                mask_split.view().into_dyn(),
                                protos_f32.view().into_dyn(),
                            ]
                        } else {
                            vec![
                                boxes_combined.view().into_dyn(),
                                protos_f32.view().into_dyn(),
                            ]
                        };
                        decoder
                            .decode_tracked_float_proto(
                                &mut tracker,
                                0,
                                &inputs,
                                &mut output_boxes,
                                &mut output_tracks,
                            )
                            .unwrap();
                    }
                    assert_eq!(output_boxes.len(), 2);
                    assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
                    assert!(output_boxes[1].equal_within_delta(&expected[1], 1.0 / 160.0));

                    if is_split {
                        for score in scores_split.iter_mut() {
                            *score = 0.0;
                        }
                    } else {
                        for score in boxes_combined.slice_mut(s![0, 4..84, ..]).iter_mut() {
                            *score = 0.0;
                        }
                    }

                    let proto_result = {
                        let inputs = if is_split {
                            vec![
                                boxes_split.view().into_dyn(),
                                scores_split.view().into_dyn(),
                                mask_split.view().into_dyn(),
                                protos_f32.view().into_dyn(),
                            ]
                        } else {
                            vec![
                                boxes_combined.view().into_dyn(),
                                protos_f32.view().into_dyn(),
                            ]
                        };
                        decoder
                            .decode_tracked_float_proto(
                                &mut tracker,
                                100_000_000 / 3,
                                &inputs,
                                &mut output_boxes,
                                &mut output_tracks,
                            )
                            .unwrap()
                    };
                    assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                    assert!(output_boxes[1].equal_within_delta(&expected[1], 1e-6));
                    assert!(proto_result.is_some_and(|x| x.mask_coefficients.is_empty()));
                } else {
                    let mut output_masks = Vec::with_capacity(50);
                    {
                        let inputs = if is_split {
                            vec![
                                boxes_split.view().into_dyn(),
                                scores_split.view().into_dyn(),
                                mask_split.view().into_dyn(),
                                protos_f32.view().into_dyn(),
                            ]
                        } else {
                            vec![
                                boxes_combined.view().into_dyn(),
                                protos_f32.view().into_dyn(),
                            ]
                        };
                        decoder
                            .decode_tracked_float(
                                &mut tracker,
                                0,
                                &inputs,
                                &mut output_boxes,
                                &mut output_masks,
                                &mut output_tracks,
                            )
                            .unwrap();
                    }
                    assert_eq!(output_boxes.len(), 2);
                    assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));
                    assert!(output_boxes[1].equal_within_delta(&expected[1], 1.0 / 160.0));

                    if is_split {
                        for score in scores_split.iter_mut() {
                            *score = 0.0;
                        }
                    } else {
                        for score in boxes_combined.slice_mut(s![0, 4..84, ..]).iter_mut() {
                            *score = 0.0;
                        }
                    }

                    {
                        let inputs = if is_split {
                            vec![
                                boxes_split.view().into_dyn(),
                                scores_split.view().into_dyn(),
                                mask_split.view().into_dyn(),
                                protos_f32.view().into_dyn(),
                            ]
                        } else {
                            vec![
                                boxes_combined.view().into_dyn(),
                                protos_f32.view().into_dyn(),
                            ]
                        };
                        decoder
                            .decode_tracked_float(
                                &mut tracker,
                                100_000_000 / 3,
                                &inputs,
                                &mut output_boxes,
                                &mut output_masks,
                                &mut output_tracks,
                            )
                            .unwrap();
                    }
                    assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                    assert!(output_boxes[1].equal_within_delta(&expected[1], 1e-6));
                    assert!(output_masks.is_empty());
                }
            }
        };
    }

    real_data_tracked_test!(test_decoder_tracked_segdet, quantized, combined, masks);
    real_data_tracked_test!(test_decoder_tracked_segdet_float, float, combined, masks);
    real_data_tracked_test!(
        test_decoder_tracked_segdet_proto,
        quantized,
        combined,
        proto
    );
    real_data_tracked_test!(
        test_decoder_tracked_segdet_proto_float,
        float,
        combined,
        proto
    );
    real_data_tracked_test!(test_decoder_tracked_segdet_split, quantized, split, masks);
    real_data_tracked_test!(test_decoder_tracked_segdet_split_float, float, split, masks);
    real_data_tracked_test!(
        test_decoder_tracked_segdet_split_proto,
        quantized,
        split,
        proto
    );
    real_data_tracked_test!(
        test_decoder_tracked_segdet_split_proto_float,
        float,
        split,
        proto
    );

    // ─── End-to-end tracked test macro ──────────────────────────────
    //
    // Generates tests with synthetic data to exercise all tracked
    // decode paths without needing real model output files.

    const E2E_COMBINED_CONFIG: &str = "
decoder_version: yolo26
outputs:
 - type: detection
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 38]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_features, 38]
   normalized: true
 - type: protos
   decoder: ultralytics
   quantization: [0.0039215686274509803921568627451, 128]
   shape: [1, 160, 160, 32]
   dshape:
    - [batch, 1]
    - [height, 160]
    - [width, 160]
    - [num_protos, 32]
";

    const E2E_SPLIT_CONFIG: &str = "
decoder_version: yolo26
outputs:
 - type: boxes
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 4]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [box_coords, 4]
   normalized: true
 - type: scores
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 1]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_classes, 1]
 - type: classes
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 1]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_classes, 1]
 - type: mask_coefficients
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 32]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_protos, 32]
 - type: protos
   decoder: ultralytics
   quantization: [0.0039215686274509803921568627451, 128]
   shape: [1, 160, 160, 32]
   dshape:
    - [batch, 1]
    - [height, 160]
    - [width, 160]
    - [num_protos, 32]
";

    macro_rules! e2e_tracked_test {
        ($name:ident, quantized, $layout:ident, $output:ident) => {
            #[test]
            fn $name() {
                let is_split = matches!(stringify!($layout), "split");
                let is_proto = matches!(stringify!($output), "proto");

                let score_threshold = 0.45;
                let iou_threshold = 0.45;

                let mut boxes = Array2::zeros((10, 4));
                let mut scores = Array2::zeros((10, 1));
                let mut classes = Array2::zeros((10, 1));
                let mask = Array2::zeros((10, 32));
                let protos = Array3::<f64>::zeros((160, 160, 32));
                let protos = protos.insert_axis(Axis(0));
                let protos_quant = (1.0 / 255.0, 0.0);
                let protos: Array4<u8> = quantize_ndarray(protos.view(), protos_quant.into());

                boxes
                    .slice_mut(s![0, ..])
                    .assign(&array![0.1234, 0.1234, 0.2345, 0.2345]);
                scores.slice_mut(s![0, ..]).assign(&array![0.9876]);
                classes.slice_mut(s![0, ..]).assign(&array![2.0]);

                let detect_quant = (2.0 / 255.0, 0.0);

                let decoder = if is_split {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_SPLIT_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                } else {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_COMBINED_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                };

                let expected = e2e_expected_boxes_quant();
                let mut tracker = ByteTrackBuilder::new()
                    .track_update(0.1)
                    .track_high_conf(0.7)
                    .build();
                let mut output_boxes = Vec::with_capacity(50);
                let mut output_tracks = Vec::with_capacity(50);

                if is_split {
                    let boxes = boxes.insert_axis(Axis(0));
                    let scores = scores.insert_axis(Axis(0));
                    let classes = classes.insert_axis(Axis(0));
                    let mask = mask.insert_axis(Axis(0));

                    let boxes: Array3<u8> = quantize_ndarray(boxes.view(), detect_quant.into());
                    let mut scores: Array3<u8> =
                        quantize_ndarray(scores.view(), detect_quant.into());
                    let classes: Array3<u8> = quantize_ndarray(classes.view(), detect_quant.into());
                    let mask: Array3<u8> = quantize_ndarray(mask.view(), detect_quant.into());

                    if is_proto {
                        {
                            let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> = vec![
                                boxes.view().into(),
                                scores.view().into(),
                                classes.view().into(),
                                mask.view().into(),
                                protos.view().into(),
                            ];
                            decoder
                                .decode_tracked_quantized_proto(
                                    &mut tracker,
                                    0,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_tracks,
                                )
                                .unwrap();
                        }
                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in scores.slice_mut(s![.., .., ..]).iter_mut() {
                            *score = u8::MIN;
                        }
                        let proto_result = {
                            let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> = vec![
                                boxes.view().into(),
                                scores.view().into(),
                                classes.view().into(),
                                mask.view().into(),
                                protos.view().into(),
                            ];
                            decoder
                                .decode_tracked_quantized_proto(
                                    &mut tracker,
                                    100_000_000 / 3,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_tracks,
                                )
                                .unwrap()
                        };
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(proto_result.is_some_and(|x| x.mask_coefficients.is_empty()));
                    } else {
                        let mut output_masks = Vec::with_capacity(50);
                        {
                            let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> = vec![
                                boxes.view().into(),
                                scores.view().into(),
                                classes.view().into(),
                                mask.view().into(),
                                protos.view().into(),
                            ];
                            decoder
                                .decode_tracked_quantized(
                                    &mut tracker,
                                    0,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_masks,
                                    &mut output_tracks,
                                )
                                .unwrap();
                        }
                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in scores.slice_mut(s![.., .., ..]).iter_mut() {
                            *score = u8::MIN;
                        }
                        {
                            let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> = vec![
                                boxes.view().into(),
                                scores.view().into(),
                                classes.view().into(),
                                mask.view().into(),
                                protos.view().into(),
                            ];
                            decoder
                                .decode_tracked_quantized(
                                    &mut tracker,
                                    100_000_000 / 3,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_masks,
                                    &mut output_tracks,
                                )
                                .unwrap();
                        }
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(output_masks.is_empty());
                    }
                } else {
                    // Combined layout
                    let detect = ndarray::concatenate![
                        Axis(1),
                        boxes.view(),
                        scores.view(),
                        classes.view(),
                        mask.view()
                    ];
                    let detect = detect.insert_axis(Axis(0));
                    assert_eq!(detect.shape(), &[1, 10, 38]);
                    let mut detect: Array3<u8> =
                        quantize_ndarray(detect.view(), detect_quant.into());

                    if is_proto {
                        {
                            let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> =
                                vec![detect.view().into(), protos.view().into()];
                            decoder
                                .decode_tracked_quantized_proto(
                                    &mut tracker,
                                    0,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_tracks,
                                )
                                .unwrap();
                        }
                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in detect.slice_mut(s![.., .., 4]).iter_mut() {
                            *score = u8::MIN;
                        }
                        let proto_result = {
                            let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> =
                                vec![detect.view().into(), protos.view().into()];
                            decoder
                                .decode_tracked_quantized_proto(
                                    &mut tracker,
                                    100_000_000 / 3,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_tracks,
                                )
                                .unwrap()
                        };
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(proto_result.is_some_and(|x| x.mask_coefficients.is_empty()));
                    } else {
                        let mut output_masks = Vec::with_capacity(50);
                        {
                            let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> =
                                vec![detect.view().into(), protos.view().into()];
                            decoder
                                .decode_tracked_quantized(
                                    &mut tracker,
                                    0,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_masks,
                                    &mut output_tracks,
                                )
                                .unwrap();
                        }
                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in detect.slice_mut(s![.., .., 4]).iter_mut() {
                            *score = u8::MIN;
                        }
                        {
                            let inputs: Vec<crate::decoder::ArrayViewDQuantized<'_>> =
                                vec![detect.view().into(), protos.view().into()];
                            decoder
                                .decode_tracked_quantized(
                                    &mut tracker,
                                    100_000_000 / 3,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_masks,
                                    &mut output_tracks,
                                )
                                .unwrap();
                        }
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(output_masks.is_empty());
                    }
                }
            }
        };
        ($name:ident, float, $layout:ident, $output:ident) => {
            #[test]
            fn $name() {
                let is_split = matches!(stringify!($layout), "split");
                let is_proto = matches!(stringify!($output), "proto");

                let score_threshold = 0.45;
                let iou_threshold = 0.45;

                let mut boxes = Array2::zeros((10, 4));
                let mut scores = Array2::zeros((10, 1));
                let mut classes = Array2::zeros((10, 1));
                let mask: Array2<f64> = Array2::zeros((10, 32));
                let protos = Array3::<f64>::zeros((160, 160, 32));
                let protos = protos.insert_axis(Axis(0));

                boxes
                    .slice_mut(s![0, ..])
                    .assign(&array![0.1234, 0.1234, 0.2345, 0.2345]);
                scores.slice_mut(s![0, ..]).assign(&array![0.9876]);
                classes.slice_mut(s![0, ..]).assign(&array![2.0]);

                let decoder = if is_split {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_SPLIT_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                } else {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_COMBINED_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                };

                let expected = e2e_expected_boxes_float();
                let mut tracker = ByteTrackBuilder::new()
                    .track_update(0.1)
                    .track_high_conf(0.7)
                    .build();
                let mut output_boxes = Vec::with_capacity(50);
                let mut output_tracks = Vec::with_capacity(50);

                if is_split {
                    let boxes = boxes.insert_axis(Axis(0));
                    let mut scores = scores.insert_axis(Axis(0));
                    let classes = classes.insert_axis(Axis(0));
                    let mask = mask.insert_axis(Axis(0));

                    if is_proto {
                        {
                            let inputs = vec![
                                boxes.view().into_dyn(),
                                scores.view().into_dyn(),
                                classes.view().into_dyn(),
                                mask.view().into_dyn(),
                                protos.view().into_dyn(),
                            ];
                            decoder
                                .decode_tracked_float_proto(
                                    &mut tracker,
                                    0,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_tracks,
                                )
                                .unwrap();
                        }
                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in scores.slice_mut(s![.., .., ..]).iter_mut() {
                            *score = 0.0;
                        }
                        let proto_result = {
                            let inputs = vec![
                                boxes.view().into_dyn(),
                                scores.view().into_dyn(),
                                classes.view().into_dyn(),
                                mask.view().into_dyn(),
                                protos.view().into_dyn(),
                            ];
                            decoder
                                .decode_tracked_float_proto(
                                    &mut tracker,
                                    100_000_000 / 3,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_tracks,
                                )
                                .unwrap()
                        };
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(proto_result.is_some_and(|x| x.mask_coefficients.is_empty()));
                    } else {
                        let mut output_masks = Vec::with_capacity(50);
                        {
                            let inputs = vec![
                                boxes.view().into_dyn(),
                                scores.view().into_dyn(),
                                classes.view().into_dyn(),
                                mask.view().into_dyn(),
                                protos.view().into_dyn(),
                            ];
                            decoder
                                .decode_tracked_float(
                                    &mut tracker,
                                    0,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_masks,
                                    &mut output_tracks,
                                )
                                .unwrap();
                        }
                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in scores.slice_mut(s![.., .., ..]).iter_mut() {
                            *score = 0.0;
                        }
                        {
                            let inputs = vec![
                                boxes.view().into_dyn(),
                                scores.view().into_dyn(),
                                classes.view().into_dyn(),
                                mask.view().into_dyn(),
                                protos.view().into_dyn(),
                            ];
                            decoder
                                .decode_tracked_float(
                                    &mut tracker,
                                    100_000_000 / 3,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_masks,
                                    &mut output_tracks,
                                )
                                .unwrap();
                        }
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(output_masks.is_empty());
                    }
                } else {
                    // Combined layout
                    let detect = ndarray::concatenate![
                        Axis(1),
                        boxes.view(),
                        scores.view(),
                        classes.view(),
                        mask.view()
                    ];
                    let mut detect = detect.insert_axis(Axis(0));
                    assert_eq!(detect.shape(), &[1, 10, 38]);

                    if is_proto {
                        {
                            let inputs = vec![detect.view().into_dyn(), protos.view().into_dyn()];
                            decoder
                                .decode_tracked_float_proto(
                                    &mut tracker,
                                    0,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_tracks,
                                )
                                .unwrap();
                        }
                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in detect.slice_mut(s![.., .., 4]).iter_mut() {
                            *score = 0.0;
                        }
                        let proto_result = {
                            let inputs = vec![detect.view().into_dyn(), protos.view().into_dyn()];
                            decoder
                                .decode_tracked_float_proto(
                                    &mut tracker,
                                    100_000_000 / 3,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_tracks,
                                )
                                .unwrap()
                        };
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(proto_result.is_some_and(|x| x.mask_coefficients.is_empty()));
                    } else {
                        let mut output_masks = Vec::with_capacity(50);
                        {
                            let inputs = vec![detect.view().into_dyn(), protos.view().into_dyn()];
                            decoder
                                .decode_tracked_float(
                                    &mut tracker,
                                    0,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_masks,
                                    &mut output_tracks,
                                )
                                .unwrap();
                        }
                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in detect.slice_mut(s![.., .., 4]).iter_mut() {
                            *score = 0.0;
                        }
                        {
                            let inputs = vec![detect.view().into_dyn(), protos.view().into_dyn()];
                            decoder
                                .decode_tracked_float(
                                    &mut tracker,
                                    100_000_000 / 3,
                                    &inputs,
                                    &mut output_boxes,
                                    &mut output_masks,
                                    &mut output_tracks,
                                )
                                .unwrap();
                        }
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(output_masks.is_empty());
                    }
                }
            }
        };
    }

    e2e_tracked_test!(
        test_decoder_tracked_end_to_end_segdet,
        quantized,
        combined,
        masks
    );
    e2e_tracked_test!(
        test_decoder_tracked_end_to_end_segdet_float,
        float,
        combined,
        masks
    );
    e2e_tracked_test!(
        test_decoder_tracked_end_to_end_segdet_proto,
        quantized,
        combined,
        proto
    );
    e2e_tracked_test!(
        test_decoder_tracked_end_to_end_segdet_proto_float,
        float,
        combined,
        proto
    );
    e2e_tracked_test!(
        test_decoder_tracked_end_to_end_segdet_split,
        quantized,
        split,
        masks
    );
    e2e_tracked_test!(
        test_decoder_tracked_end_to_end_segdet_split_float,
        float,
        split,
        masks
    );
    e2e_tracked_test!(
        test_decoder_tracked_end_to_end_segdet_split_proto,
        quantized,
        split,
        proto
    );
    e2e_tracked_test!(
        test_decoder_tracked_end_to_end_segdet_split_proto_float,
        float,
        split,
        proto
    );

    // ─── End-to-end tracked TensorDyn test macro ────────────────────
    //
    // Same as e2e_tracked_test but wraps data in TensorDyn and exercises
    // the public decode_tracked / decode_proto_tracked API.

    macro_rules! e2e_tracked_tensor_test {
        ($name:ident, quantized, $layout:ident, $output:ident) => {
            #[test]
            fn $name() {
                use edgefirst_tensor::{Tensor, TensorMapTrait, TensorTrait};

                let is_split = matches!(stringify!($layout), "split");
                let is_proto = matches!(stringify!($output), "proto");

                let score_threshold = 0.45;
                let iou_threshold = 0.45;

                let mut boxes = Array2::zeros((10, 4));
                let mut scores = Array2::zeros((10, 1));
                let mut classes = Array2::zeros((10, 1));
                let mask = Array2::zeros((10, 32));
                let protos_f64 = Array3::<f64>::zeros((160, 160, 32));
                let protos_f64 = protos_f64.insert_axis(Axis(0));
                let protos_quant = (1.0 / 255.0, 0.0);
                let protos_u8: Array4<u8> =
                    quantize_ndarray(protos_f64.view(), protos_quant.into());

                boxes
                    .slice_mut(s![0, ..])
                    .assign(&array![0.1234, 0.1234, 0.2345, 0.2345]);
                scores.slice_mut(s![0, ..]).assign(&array![0.9876]);
                classes.slice_mut(s![0, ..]).assign(&array![2.0]);

                let detect_quant = (2.0 / 255.0, 0.0);

                let decoder = if is_split {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_SPLIT_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                } else {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_COMBINED_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                };

                // Helper to wrap a u8 slice into a TensorDyn
                let make_u8_tensor =
                    |shape: &[usize], data: &[u8]| -> edgefirst_tensor::TensorDyn {
                        let t = Tensor::<u8>::new(shape, None, None).unwrap();
                        t.map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
                        t.into()
                    };

                let expected = e2e_expected_boxes_quant();
                let mut tracker = ByteTrackBuilder::new()
                    .track_update(0.1)
                    .track_high_conf(0.7)
                    .build();
                let mut output_boxes = Vec::with_capacity(50);
                let mut output_tracks = Vec::with_capacity(50);

                let protos_td = make_u8_tensor(protos_u8.shape(), protos_u8.as_slice().unwrap());

                if is_split {
                    let boxes = boxes.insert_axis(Axis(0));
                    let scores = scores.insert_axis(Axis(0));
                    let classes = classes.insert_axis(Axis(0));
                    let mask = mask.insert_axis(Axis(0));

                    let boxes_q: Array3<u8> = quantize_ndarray(boxes.view(), detect_quant.into());
                    let mut scores_q: Array3<u8> =
                        quantize_ndarray(scores.view(), detect_quant.into());
                    let classes_q: Array3<u8> =
                        quantize_ndarray(classes.view(), detect_quant.into());
                    let mask_q: Array3<u8> = quantize_ndarray(mask.view(), detect_quant.into());

                    let boxes_td = make_u8_tensor(boxes_q.shape(), boxes_q.as_slice().unwrap());
                    let classes_td =
                        make_u8_tensor(classes_q.shape(), classes_q.as_slice().unwrap());
                    let mask_td = make_u8_tensor(mask_q.shape(), mask_q.as_slice().unwrap());

                    if is_proto {
                        let scores_td =
                            make_u8_tensor(scores_q.shape(), scores_q.as_slice().unwrap());
                        decoder
                            .decode_proto_tracked(
                                &mut tracker,
                                0,
                                &[&boxes_td, &scores_td, &classes_td, &mask_td, &protos_td],
                                &mut output_boxes,
                                &mut output_tracks,
                            )
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in scores_q.slice_mut(s![.., .., ..]).iter_mut() {
                            *score = u8::MIN;
                        }
                        let scores_td =
                            make_u8_tensor(scores_q.shape(), scores_q.as_slice().unwrap());
                        let proto_result = decoder
                            .decode_proto_tracked(
                                &mut tracker,
                                100_000_000 / 3,
                                &[&boxes_td, &scores_td, &classes_td, &mask_td, &protos_td],
                                &mut output_boxes,
                                &mut output_tracks,
                            )
                            .unwrap();
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(proto_result.is_some_and(|x| x.mask_coefficients.is_empty()));
                    } else {
                        let scores_td =
                            make_u8_tensor(scores_q.shape(), scores_q.as_slice().unwrap());
                        let mut output_masks = Vec::with_capacity(50);
                        decoder
                            .decode_tracked(
                                &mut tracker,
                                0,
                                &[&boxes_td, &scores_td, &classes_td, &mask_td, &protos_td],
                                &mut output_boxes,
                                &mut output_masks,
                                &mut output_tracks,
                            )
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in scores_q.slice_mut(s![.., .., ..]).iter_mut() {
                            *score = u8::MIN;
                        }
                        let scores_td =
                            make_u8_tensor(scores_q.shape(), scores_q.as_slice().unwrap());
                        decoder
                            .decode_tracked(
                                &mut tracker,
                                100_000_000 / 3,
                                &[&boxes_td, &scores_td, &classes_td, &mask_td, &protos_td],
                                &mut output_boxes,
                                &mut output_masks,
                                &mut output_tracks,
                            )
                            .unwrap();
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(output_masks.is_empty());
                    }
                } else {
                    // Combined layout
                    let detect = ndarray::concatenate![
                        Axis(1),
                        boxes.view(),
                        scores.view(),
                        classes.view(),
                        mask.view()
                    ];
                    let detect = detect.insert_axis(Axis(0));
                    assert_eq!(detect.shape(), &[1, 10, 38]);
                    // Ensure contiguous layout after concatenation for as_slice()
                    let detect =
                        Array3::from_shape_vec(detect.raw_dim(), detect.iter().copied().collect())
                            .unwrap();
                    let mut detect_q: Array3<u8> =
                        quantize_ndarray(detect.view(), detect_quant.into());

                    if is_proto {
                        let detect_td =
                            make_u8_tensor(detect_q.shape(), detect_q.as_slice().unwrap());
                        decoder
                            .decode_proto_tracked(
                                &mut tracker,
                                0,
                                &[&detect_td, &protos_td],
                                &mut output_boxes,
                                &mut output_tracks,
                            )
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in detect_q.slice_mut(s![.., .., 4]).iter_mut() {
                            *score = u8::MIN;
                        }
                        let detect_td =
                            make_u8_tensor(detect_q.shape(), detect_q.as_slice().unwrap());
                        let proto_result = decoder
                            .decode_proto_tracked(
                                &mut tracker,
                                100_000_000 / 3,
                                &[&detect_td, &protos_td],
                                &mut output_boxes,
                                &mut output_tracks,
                            )
                            .unwrap();
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(proto_result.is_some_and(|x| x.mask_coefficients.is_empty()));
                    } else {
                        let detect_td =
                            make_u8_tensor(detect_q.shape(), detect_q.as_slice().unwrap());
                        let mut output_masks = Vec::with_capacity(50);
                        decoder
                            .decode_tracked(
                                &mut tracker,
                                0,
                                &[&detect_td, &protos_td],
                                &mut output_boxes,
                                &mut output_masks,
                                &mut output_tracks,
                            )
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in detect_q.slice_mut(s![.., .., 4]).iter_mut() {
                            *score = u8::MIN;
                        }
                        let detect_td =
                            make_u8_tensor(detect_q.shape(), detect_q.as_slice().unwrap());
                        decoder
                            .decode_tracked(
                                &mut tracker,
                                100_000_000 / 3,
                                &[&detect_td, &protos_td],
                                &mut output_boxes,
                                &mut output_masks,
                                &mut output_tracks,
                            )
                            .unwrap();
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(output_masks.is_empty());
                    }
                }
            }
        };
        ($name:ident, float, $layout:ident, $output:ident) => {
            #[test]
            fn $name() {
                use edgefirst_tensor::{Tensor, TensorMapTrait, TensorTrait};

                let is_split = matches!(stringify!($layout), "split");
                let is_proto = matches!(stringify!($output), "proto");

                let score_threshold = 0.45;
                let iou_threshold = 0.45;

                let mut boxes = Array2::zeros((10, 4));
                let mut scores = Array2::zeros((10, 1));
                let mut classes = Array2::zeros((10, 1));
                let mask: Array2<f64> = Array2::zeros((10, 32));
                let protos = Array3::<f64>::zeros((160, 160, 32));
                let protos = protos.insert_axis(Axis(0));

                boxes
                    .slice_mut(s![0, ..])
                    .assign(&array![0.1234, 0.1234, 0.2345, 0.2345]);
                scores.slice_mut(s![0, ..]).assign(&array![0.9876]);
                classes.slice_mut(s![0, ..]).assign(&array![2.0]);

                let decoder = if is_split {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_SPLIT_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                } else {
                    DecoderBuilder::default()
                        .with_config_yaml_str(E2E_COMBINED_CONFIG.to_string())
                        .with_score_threshold(score_threshold)
                        .with_iou_threshold(iou_threshold)
                        .build()
                        .unwrap()
                };

                // Helper to wrap an f64 slice into a TensorDyn
                let make_f64_tensor =
                    |shape: &[usize], data: &[f64]| -> edgefirst_tensor::TensorDyn {
                        let t = Tensor::<f64>::new(shape, None, None).unwrap();
                        t.map().unwrap().as_mut_slice()[..data.len()].copy_from_slice(data);
                        t.into()
                    };

                let expected = e2e_expected_boxes_float();
                let mut tracker = ByteTrackBuilder::new()
                    .track_update(0.1)
                    .track_high_conf(0.7)
                    .build();
                let mut output_boxes = Vec::with_capacity(50);
                let mut output_tracks = Vec::with_capacity(50);

                let protos_td = make_f64_tensor(protos.shape(), protos.as_slice().unwrap());

                if is_split {
                    let boxes = boxes.insert_axis(Axis(0));
                    let mut scores = scores.insert_axis(Axis(0));
                    let classes = classes.insert_axis(Axis(0));
                    let mask = mask.insert_axis(Axis(0));

                    let boxes_td = make_f64_tensor(boxes.shape(), boxes.as_slice().unwrap());
                    let classes_td = make_f64_tensor(classes.shape(), classes.as_slice().unwrap());
                    let mask_td = make_f64_tensor(mask.shape(), mask.as_slice().unwrap());

                    if is_proto {
                        let scores_td = make_f64_tensor(scores.shape(), scores.as_slice().unwrap());
                        decoder
                            .decode_proto_tracked(
                                &mut tracker,
                                0,
                                &[&boxes_td, &scores_td, &classes_td, &mask_td, &protos_td],
                                &mut output_boxes,
                                &mut output_tracks,
                            )
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in scores.slice_mut(s![.., .., ..]).iter_mut() {
                            *score = 0.0;
                        }
                        let scores_td = make_f64_tensor(scores.shape(), scores.as_slice().unwrap());
                        let proto_result = decoder
                            .decode_proto_tracked(
                                &mut tracker,
                                100_000_000 / 3,
                                &[&boxes_td, &scores_td, &classes_td, &mask_td, &protos_td],
                                &mut output_boxes,
                                &mut output_tracks,
                            )
                            .unwrap();
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(proto_result.is_some_and(|x| x.mask_coefficients.is_empty()));
                    } else {
                        let scores_td = make_f64_tensor(scores.shape(), scores.as_slice().unwrap());
                        let mut output_masks = Vec::with_capacity(50);
                        decoder
                            .decode_tracked(
                                &mut tracker,
                                0,
                                &[&boxes_td, &scores_td, &classes_td, &mask_td, &protos_td],
                                &mut output_boxes,
                                &mut output_masks,
                                &mut output_tracks,
                            )
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in scores.slice_mut(s![.., .., ..]).iter_mut() {
                            *score = 0.0;
                        }
                        let scores_td = make_f64_tensor(scores.shape(), scores.as_slice().unwrap());
                        decoder
                            .decode_tracked(
                                &mut tracker,
                                100_000_000 / 3,
                                &[&boxes_td, &scores_td, &classes_td, &mask_td, &protos_td],
                                &mut output_boxes,
                                &mut output_masks,
                                &mut output_tracks,
                            )
                            .unwrap();
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(output_masks.is_empty());
                    }
                } else {
                    // Combined layout
                    let detect = ndarray::concatenate![
                        Axis(1),
                        boxes.view(),
                        scores.view(),
                        classes.view(),
                        mask.view()
                    ];
                    let detect = detect.insert_axis(Axis(0));
                    assert_eq!(detect.shape(), &[1, 10, 38]);
                    // Ensure contiguous layout after concatenation for as_slice()
                    let mut detect =
                        Array3::from_shape_vec(detect.raw_dim(), detect.iter().copied().collect())
                            .unwrap();

                    if is_proto {
                        let detect_td = make_f64_tensor(detect.shape(), detect.as_slice().unwrap());
                        decoder
                            .decode_proto_tracked(
                                &mut tracker,
                                0,
                                &[&detect_td, &protos_td],
                                &mut output_boxes,
                                &mut output_tracks,
                            )
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in detect.slice_mut(s![.., .., 4]).iter_mut() {
                            *score = 0.0;
                        }
                        let detect_td = make_f64_tensor(detect.shape(), detect.as_slice().unwrap());
                        let proto_result = decoder
                            .decode_proto_tracked(
                                &mut tracker,
                                100_000_000 / 3,
                                &[&detect_td, &protos_td],
                                &mut output_boxes,
                                &mut output_tracks,
                            )
                            .unwrap();
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(proto_result.is_some_and(|x| x.mask_coefficients.is_empty()));
                    } else {
                        let detect_td = make_f64_tensor(detect.shape(), detect.as_slice().unwrap());
                        let mut output_masks = Vec::with_capacity(50);
                        decoder
                            .decode_tracked(
                                &mut tracker,
                                0,
                                &[&detect_td, &protos_td],
                                &mut output_boxes,
                                &mut output_masks,
                                &mut output_tracks,
                            )
                            .unwrap();

                        assert_eq!(output_boxes.len(), 1);
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1.0 / 160.0));

                        for score in detect.slice_mut(s![.., .., 4]).iter_mut() {
                            *score = 0.0;
                        }
                        let detect_td = make_f64_tensor(detect.shape(), detect.as_slice().unwrap());
                        decoder
                            .decode_tracked(
                                &mut tracker,
                                100_000_000 / 3,
                                &[&detect_td, &protos_td],
                                &mut output_boxes,
                                &mut output_masks,
                                &mut output_tracks,
                            )
                            .unwrap();
                        assert!(output_boxes[0].equal_within_delta(&expected[0], 1e-6));
                        assert!(output_masks.is_empty());
                    }
                }
            }
        };
    }

    e2e_tracked_tensor_test!(
        test_decoder_tracked_tensor_end_to_end_segdet,
        quantized,
        combined,
        masks
    );
    e2e_tracked_tensor_test!(
        test_decoder_tracked_tensor_end_to_end_segdet_float,
        float,
        combined,
        masks
    );
    e2e_tracked_tensor_test!(
        test_decoder_tracked_tensor_end_to_end_segdet_proto,
        quantized,
        combined,
        proto
    );
    e2e_tracked_tensor_test!(
        test_decoder_tracked_tensor_end_to_end_segdet_proto_float,
        float,
        combined,
        proto
    );
    e2e_tracked_tensor_test!(
        test_decoder_tracked_tensor_end_to_end_segdet_split,
        quantized,
        split,
        masks
    );
    e2e_tracked_tensor_test!(
        test_decoder_tracked_tensor_end_to_end_segdet_split_float,
        float,
        split,
        masks
    );
    e2e_tracked_tensor_test!(
        test_decoder_tracked_tensor_end_to_end_segdet_split_proto,
        quantized,
        split,
        proto
    );
    e2e_tracked_tensor_test!(
        test_decoder_tracked_tensor_end_to_end_segdet_split_proto_float,
        float,
        split,
        proto
    );

    #[test]
    fn test_decoder_tracked_linear_motion() {
        use crate::configs::{DecoderType, Nms};
        use crate::DecoderBuilder;

        let score_threshold = 0.25;
        let iou_threshold = 0.1;
        let out = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/yolov8s_80_classes.bin"
        ));
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
        let mut out = Array3::from_shape_vec((1, 84, 8400), out.to_vec()).unwrap();
        let quant = (0.0040811873, -123).into();

        let decoder = DecoderBuilder::default()
            .with_config_yolo_det(
                crate::configs::Detection {
                    decoder: DecoderType::Ultralytics,
                    shape: vec![1, 84, 8400],
                    anchors: None,
                    quantization: Some(quant),
                    dshape: vec![
                        (crate::configs::DimName::Batch, 1),
                        (crate::configs::DimName::NumFeatures, 84),
                        (crate::configs::DimName::NumBoxes, 8400),
                    ],
                    normalized: Some(true),
                },
                None,
            )
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .with_nms(Some(Nms::ClassAgnostic))
            .build()
            .unwrap();

        let mut expected_boxes = [
            DetectBox {
                bbox: BoundingBox {
                    xmin: 0.5285137,
                    ymin: 0.05305544,
                    xmax: 0.87541467,
                    ymax: 0.9998909,
                },
                score: 0.5591227,
                label: 0,
            },
            DetectBox {
                bbox: BoundingBox {
                    xmin: 0.130598,
                    ymin: 0.43260583,
                    xmax: 0.35098213,
                    ymax: 0.9958097,
                },
                score: 0.33057618,
                label: 75,
            },
        ];

        let mut tracker = ByteTrackBuilder::new()
            .track_update(0.1)
            .track_high_conf(0.3)
            .build();

        let mut output_boxes = Vec::with_capacity(50);
        let mut output_masks = Vec::with_capacity(50);
        let mut output_tracks = Vec::with_capacity(50);

        decoder
            .decode_tracked_quantized(
                &mut tracker,
                0,
                &[out.view().into()],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
            )
            .unwrap();

        assert_eq!(output_boxes.len(), 2);
        assert!(output_boxes[0].equal_within_delta(&expected_boxes[0], 1e-6));
        assert!(output_boxes[1].equal_within_delta(&expected_boxes[1], 1e-6));

        for i in 1..=100 {
            let mut out = out.clone();
            // introduce linear movement into the XY coordinates
            let mut x_values = out.slice_mut(s![0, 0, ..]);
            for x in x_values.iter_mut() {
                *x = x.saturating_add((i as f32 * 1e-3 / quant.0).round() as i8);
            }

            decoder
                .decode_tracked_quantized(
                    &mut tracker,
                    100_000_000 * i / 3, // simulate 33.333ms between frames
                    &[out.view().into()],
                    &mut output_boxes,
                    &mut output_masks,
                    &mut output_tracks,
                )
                .unwrap();

            assert_eq!(output_boxes.len(), 2);
        }
        let tracks = tracker.get_active_tracks();
        let predicted_boxes: Vec<_> = tracks
            .iter()
            .map(|track| {
                let mut l = track.last_box;
                l.bbox = track.info.tracked_location.into();
                l
            })
            .collect();
        expected_boxes[0].bbox.xmin += 0.1; // compensate for linear movement
        expected_boxes[0].bbox.xmax += 0.1;
        expected_boxes[1].bbox.xmin += 0.1;
        expected_boxes[1].bbox.xmax += 0.1;

        assert!(predicted_boxes[0].equal_within_delta(&expected_boxes[0], 1e-3));
        assert!(predicted_boxes[1].equal_within_delta(&expected_boxes[1], 1e-3));

        // give the decoder a final frame with no detections to ensure tracks are properly predicting forward when detection is missing
        let mut scores_values = out.slice_mut(s![0, 4.., ..]);
        for score in scores_values.iter_mut() {
            *score = i8::MIN; // set all scores to minimum to simulate no detections
        }
        decoder
            .decode_tracked_quantized(
                &mut tracker,
                100_000_000 * 101 / 3,
                &[out.view().into()],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
            )
            .unwrap();
        expected_boxes[0].bbox.xmin += 0.001; // compensate for expected movement
        expected_boxes[0].bbox.xmax += 0.001;
        expected_boxes[1].bbox.xmin += 0.001;
        expected_boxes[1].bbox.xmax += 0.001;

        assert!(output_boxes[0].equal_within_delta(&expected_boxes[0], 1e-3));
        assert!(output_boxes[1].equal_within_delta(&expected_boxes[1], 1e-3));
    }

    #[test]
    fn test_decoder_tracked_end_to_end_float() {
        let score_threshold = 0.45;
        let iou_threshold = 0.45;

        let mut boxes = Array2::zeros((10, 4));
        let mut scores = Array2::zeros((10, 1));
        let mut classes = Array2::zeros((10, 1));

        boxes
            .slice_mut(s![0, ..,])
            .assign(&array![0.1234, 0.1234, 0.2345, 0.2345]);
        scores.slice_mut(s![0, ..]).assign(&array![0.9876]);
        classes.slice_mut(s![0, ..]).assign(&array![2.0]);

        let detect = ndarray::concatenate![Axis(1), boxes.view(), scores.view(), classes.view(),];
        let mut detect = detect.insert_axis(Axis(0));
        assert_eq!(detect.shape(), &[1, 10, 6]);
        let config = "
decoder_version: yolo26
outputs:
 - type: detection
   decoder: ultralytics
   quantization: [0.00784313725490196, 0]
   shape: [1, 10, 6]
   dshape:
    - [batch, 1]
    - [num_boxes, 10]
    - [num_features, 6]
   normalized: true
";

        let decoder = DecoderBuilder::default()
            .with_config_yaml_str(config.to_string())
            .with_score_threshold(score_threshold)
            .with_iou_threshold(iou_threshold)
            .build()
            .unwrap();

        let expected_boxes = [DetectBox {
            bbox: BoundingBox {
                xmin: 0.1234,
                ymin: 0.1234,
                xmax: 0.2345,
                ymax: 0.2345,
            },
            score: 0.9876,
            label: 2,
        }];

        let mut tracker = ByteTrackBuilder::new()
            .track_update(0.1)
            .track_high_conf(0.7)
            .build();

        let mut output_boxes = Vec::with_capacity(50);
        let mut output_masks = Vec::with_capacity(50);
        let mut output_tracks = Vec::with_capacity(50);

        decoder
            .decode_tracked_float(
                &mut tracker,
                0,
                &[detect.view().into_dyn()],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
            )
            .unwrap();

        assert_eq!(output_boxes.len(), 1);
        assert!(output_boxes[0].equal_within_delta(&expected_boxes[0], 1e-6));

        // give the decoder a final frame with no detections to ensure tracks are properly predicting forward when detection is missing

        for score in detect.slice_mut(s![.., .., 4]).iter_mut() {
            *score = 0.0; // set all scores to minimum to simulate no detections
        }

        decoder
            .decode_tracked_float(
                &mut tracker,
                100_000_000 / 3,
                &[detect.view().into_dyn()],
                &mut output_boxes,
                &mut output_masks,
                &mut output_tracks,
            )
            .unwrap();
        assert!(output_boxes[0].equal_within_delta(&expected_boxes[0], 1e-6));
    }
}
