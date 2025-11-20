// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use ndarray::{Array2, ArrayView2, ArrayView3};
use num_traits::{AsPrimitive, Float, PrimInt};

use crate::{
    BBoxTypeTrait, DecoderError, DetectBox, Quantization, XYWH, XYXY,
    byte::{nms_int, postprocess_boxes_quant, quantize_score_threshold},
    configs::Detection,
    dequant_detect_box,
    float::{nms_float, postprocess_boxes_float},
};

/// Configuration for ModelPack split detection decoder. The quantization is
/// ignored when decoding float models.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelPackDetectionConfig {
    pub anchors: Vec<[f32; 2]>,
    pub quantization: Option<Quantization>,
}

impl TryFrom<&Detection> for ModelPackDetectionConfig {
    type Error = DecoderError;

    fn try_from(value: &Detection) -> Result<Self, DecoderError> {
        Ok(Self {
            anchors: value.anchors.clone().ok_or_else(|| {
                DecoderError::InvalidConfig("ModelPack Split Detection missing anchors".to_string())
            })?,
            quantization: value.quantization.map(Quantization::from),
        })
    }
}

/// Decodes ModelPack detection outputs from quantized tensors.
///
/// The boxes are expected to be in XYXY format.
///
/// Expected shapes of inputs:
/// - boxes: (num_boxes, 4)
/// - scores: (num_boxes, num_classes)
///
/// # Panics
/// Panics if shapes don't match the expected dimensions.
pub fn decode_modelpack_det<
    BOX: PrimInt + AsPrimitive<f32> + Send + Sync,
    SCORE: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    boxes_tensor: (ArrayView2<BOX>, Quantization),
    scores_tensor: (ArrayView2<SCORE>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) where
    f32: AsPrimitive<SCORE>,
{
    impl_modelpack_quant::<XYXY, _, _>(
        boxes_tensor,
        scores_tensor,
        score_threshold,
        iou_threshold,
        output_boxes,
    )
}

/// Decodes ModelPack detection outputs from float tensors. The boxes
/// are expected to be in XYXY format.
///
/// Expected shapes of inputs:
/// - boxes: (num_boxes, 4)
/// - scores: (num_boxes, num_classes)
///
/// # Panics
/// Panics if shapes don't match the expected dimensions.
pub fn decode_modelpack_float<
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    SCORE: Float + AsPrimitive<f32> + Send + Sync,
>(
    boxes_tensor: ArrayView2<BOX>,
    scores_tensor: ArrayView2<SCORE>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) where
    f32: AsPrimitive<SCORE>,
{
    impl_modelpack_float::<XYXY, _, _>(
        boxes_tensor,
        scores_tensor,
        score_threshold,
        iou_threshold,
        output_boxes,
    )
}

/// Decodes ModelPack split detection outputs from quantized tensors. The boxes
/// are expected to be in XYWH format.
///
/// The `configs` must correspond to the `outputs` in order.
///
/// Expected shapes of inputs:
/// - outputs: (width, height, num_anchors * (5 + num_classes))
///
/// # Panics
/// Panics if shapes don't match the expected dimensions.
pub fn decode_modelpack_split_quant<D: AsPrimitive<f32>>(
    outputs: &[ArrayView3<D>],
    configs: &[ModelPackDetectionConfig],
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    impl_modelpack_split_quant::<XYWH, D>(
        outputs,
        configs,
        score_threshold,
        iou_threshold,
        output_boxes,
    )
}

/// Decodes ModelPack split detection outputs from float tensors. The boxes
/// are expected to be in XYWH format.
///
/// The `configs` must correspond to the `outputs` in order.
///
/// Expected shapes of inputs:
/// - outputs: (width, height, num_anchors * (5 + num_classes))
///
/// # Panics
/// Panics if shapes don't match the expected dimensions.
pub fn decode_modelpack_split_float<D: AsPrimitive<f32>>(
    outputs: &[ArrayView3<D>],
    configs: &[ModelPackDetectionConfig],
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    impl_modelpack_split_float::<XYWH, D>(
        outputs,
        configs,
        score_threshold,
        iou_threshold,
        output_boxes,
    );
}
/// Implementation of ModelPack detection decoding for quantized tensors.
///
/// Expected shapes of inputs:
/// - boxes: (num_boxes, 4)
/// - scores: (num_boxes, num_classes)
///
/// # Panics
/// Panics if shapes don't match the expected dimensions.
#[doc(hidden)]
pub fn impl_modelpack_quant<
    B: BBoxTypeTrait,
    BOX: PrimInt + AsPrimitive<f32> + Send + Sync,
    SCORE: PrimInt + AsPrimitive<f32> + Send + Sync,
>(
    boxes: (ArrayView2<BOX>, Quantization),
    scores: (ArrayView2<SCORE>, Quantization),
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) where
    f32: AsPrimitive<SCORE>,
{
    let (boxes_tensor, quant_boxes) = boxes;
    let (scores_tensor, quant_scores) = scores;
    let boxes = {
        let score_threshold = quantize_score_threshold(score_threshold, quant_boxes);
        postprocess_boxes_quant::<B, _, _>(
            score_threshold,
            boxes_tensor,
            scores_tensor,
            quant_boxes,
        )
    };
    let boxes = nms_int(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(dequant_detect_box(&b, quant_scores));
    }
}

/// Implementation of ModelPack detection decoding for float tensors.
///
/// Expected shapes of inputs:
/// - boxes: (num_boxes, 4)
/// - scores: (num_boxes, num_classes)
///
/// # Panics
/// Panics if shapes don't match the expected dimensions.
#[doc(hidden)]
pub fn impl_modelpack_float<
    B: BBoxTypeTrait,
    BOX: Float + AsPrimitive<f32> + Send + Sync,
    SCORE: Float + AsPrimitive<f32> + Send + Sync,
>(
    boxes_tensor: ArrayView2<BOX>,
    scores_tensor: ArrayView2<SCORE>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) where
    f32: AsPrimitive<SCORE>,
{
    let boxes =
        postprocess_boxes_float::<B, _, _>(score_threshold.as_(), boxes_tensor, scores_tensor);
    let boxes = nms_float(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(b);
    }
}

/// Implementation of ModelPack split detection decoding for quantized tensors.
///
/// Expected shapes of inputs:
/// - boxes: (num_boxes, 4)
/// - scores: (num_boxes, num_classes)
///
/// # Panics
/// Panics if shapes don't match the expected dimensions.
#[doc(hidden)]
pub fn impl_modelpack_split_quant<B: BBoxTypeTrait, D: AsPrimitive<f32>>(
    outputs: &[ArrayView3<D>],
    configs: &[ModelPackDetectionConfig],
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    let (boxes_tensor, scores_tensor) = postprocess_modelpack_split_quant(outputs, configs);
    let boxes = postprocess_boxes_float::<B, _, _>(
        score_threshold,
        boxes_tensor.view(),
        scores_tensor.view(),
    );
    let boxes = nms_float(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(b);
    }
}

/// Implementation of ModelPack split detection decoding for float tensors.
///
/// The `configs` must correspond to the `outputs` in order.
///
/// Expected shapes of inputs:
/// - outputs: (width, height, num_anchors * (5 + num_classes))
///
/// # Panics
/// Panics if shapes don't match the expected dimensions.
#[doc(hidden)]
pub fn impl_modelpack_split_float<B: BBoxTypeTrait, D: AsPrimitive<f32>>(
    outputs: &[ArrayView3<D>],
    configs: &[ModelPackDetectionConfig],
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    let (boxes_tensor, scores_tensor) = postprocess_modelpack_split_float(outputs, configs);
    let boxes = postprocess_boxes_float::<B, _, _>(
        score_threshold,
        boxes_tensor.view(),
        scores_tensor.view(),
    );
    let boxes = nms_float(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(b);
    }
}

/// Post processes ModelPack split detection into detection boxes,
/// filtering out any boxes below the score threshold. Returns the boxes and
/// scores tensors. Boxes are in XYWH format.
#[doc(hidden)]
pub fn postprocess_modelpack_split_quant<T: AsPrimitive<f32>>(
    outputs: &[ArrayView3<T>],
    config: &[ModelPackDetectionConfig],
) -> (Array2<f32>, Array2<f32>) {
    let mut total_capacity = 0;
    let mut nc = 0;
    for (p, detail) in outputs.iter().zip(config) {
        let shape = p.shape();
        let na = detail.anchors.len();
        nc = *shape
            .last()
            .expect("Shape must have at least one dimension")
            / na
            - 5;
        total_capacity += shape[0] * shape[1] * na;
    }
    let mut bboxes = Vec::with_capacity(total_capacity * 4);
    let mut bscores = Vec::with_capacity(total_capacity * nc);

    for (p, detail) in outputs.iter().zip(config) {
        let anchors = &detail.anchors;
        let na = detail.anchors.len();
        let shape = p.shape();
        assert_eq!(
            shape.iter().product::<usize>(),
            p.len(),
            "Shape product doesn't match tensor length"
        );
        let p_sigmoid = if let Some(quant) = &detail.quantization {
            let scaled_zero = -quant.zero_point as f32 * quant.scale;
            p.mapv(|x| fast_sigmoid_impl(x.as_() * quant.scale + scaled_zero))
        } else {
            p.mapv(|x| fast_sigmoid_impl(x.as_()))
        };
        let p_sigmoid = p_sigmoid.as_standard_layout();

        // Safe to unwrap since we ensured standard layout above
        let p = p_sigmoid
            .as_slice()
            .expect("Sigmoids are not in standard layout");
        let height = shape[0];
        let width = shape[1];

        let div_width = 1.0 / width as f32;
        let div_height = 1.0 / height as f32;

        let mut grid = Vec::with_capacity(height * width * na * 2);
        for y in 0..height {
            for x in 0..width {
                for _ in 0..na {
                    grid.push(x as f32 - 0.5);
                    grid.push(y as f32 - 0.5);
                }
            }
        }
        for ((p, g), anchor) in p
            .chunks_exact(nc + 5)
            .zip(grid.chunks_exact(2))
            .zip(anchors.iter().cycle())
        {
            let (x, y) = (p[0], p[1]);
            let x = (x * 2.0 + g[0]) * div_width;
            let y = (y * 2.0 + g[1]) * div_height;
            let (w, h) = (p[2], p[3]);
            let w = w * w * 4.0 * anchor[0];
            let h = h * h * 4.0 * anchor[1];

            bboxes.push(x);
            bboxes.push(y);
            bboxes.push(w);
            bboxes.push(h);

            if nc == 1 {
                bscores.push(p[4]);
            } else {
                let obj = p[4];
                let probs = p[5..].iter().map(|x| *x * obj);
                bscores.extend(probs);
            }
        }
    }
    // Safe to unwrap since we ensured lengths will match above

    debug_assert_eq!(bboxes.len() % 4, 0);
    debug_assert_eq!(bscores.len() % nc, 0);

    let bboxes = Array2::from_shape_vec((bboxes.len() / 4, 4), bboxes)
        .expect("Failed to create bboxes array");
    let bscores = Array2::from_shape_vec((bscores.len() / nc, nc), bscores)
        .expect("Failed to create bscores array");
    (bboxes, bscores)
}

/// Post processes ModelPack split detection into detection boxes,
/// filtering out any boxes below the score threshold. Returns the boxes and
/// scores tensors. Boxes are in XYWH format.
#[doc(hidden)]
pub fn postprocess_modelpack_split_float<T: AsPrimitive<f32>>(
    outputs: &[ArrayView3<T>],
    config: &[ModelPackDetectionConfig],
) -> (Array2<f32>, Array2<f32>) {
    let mut total_capacity = 0;
    let mut nc = 0;
    for (p, detail) in outputs.iter().zip(config) {
        let shape = p.shape();
        let na = detail.anchors.len();
        nc = *shape
            .last()
            .expect("Shape must have at least one dimension")
            / na
            - 5;
        total_capacity += shape[0] * shape[1] * na;
    }
    let mut bboxes = Vec::with_capacity(total_capacity * 4);
    let mut bscores = Vec::with_capacity(total_capacity * nc);

    for (p, detail) in outputs.iter().zip(config) {
        let anchors = &detail.anchors;
        let na = detail.anchors.len();
        let shape = p.shape();
        assert_eq!(
            shape.iter().product::<usize>(),
            p.len(),
            "Shape product doesn't match tensor length"
        );
        let p_sigmoid = p.mapv(|x| fast_sigmoid_impl(x.as_()));
        let p_sigmoid = p_sigmoid.as_standard_layout();

        // Safe to unwrap since we ensured standard layout above
        let p = p_sigmoid
            .as_slice()
            .expect("Sigmoids are not in standard layout");
        let height = shape[0];
        let width = shape[1];

        let div_width = 1.0 / width as f32;
        let div_height = 1.0 / height as f32;

        let mut grid = Vec::with_capacity(height * width * na * 2);
        for y in 0..height {
            for x in 0..width {
                for _ in 0..na {
                    grid.push(x as f32 - 0.5);
                    grid.push(y as f32 - 0.5);
                }
            }
        }
        for ((p, g), anchor) in p
            .chunks_exact(nc + 5)
            .zip(grid.chunks_exact(2))
            .zip(anchors.iter().cycle())
        {
            let (x, y) = (p[0], p[1]);
            let x = (x * 2.0 + g[0]) * div_width;
            let y = (y * 2.0 + g[1]) * div_height;
            let (w, h) = (p[2], p[3]);
            let w = w * w * 4.0 * anchor[0];
            let h = h * h * 4.0 * anchor[1];

            bboxes.push(x);
            bboxes.push(y);
            bboxes.push(w);
            bboxes.push(h);

            if nc == 1 {
                bscores.push(p[4]);
            } else {
                let obj = p[4];
                let probs = p[5..].iter().map(|x| *x * obj);
                bscores.extend(probs);
            }
        }
    }
    // Safe to unwrap since we ensured lengths will match above

    debug_assert_eq!(bboxes.len() % 4, 0);
    debug_assert_eq!(bscores.len() % nc, 0);

    let bboxes = Array2::from_shape_vec((bboxes.len() / 4, 4), bboxes)
        .expect("Failed to create bboxes array");
    let bscores = Array2::from_shape_vec((bscores.len() / nc, nc), bscores)
        .expect("Failed to create bscores array");
    (bboxes, bscores)
}

#[inline(always)]
fn fast_sigmoid_impl(f: f32) -> f32 {
    if f.abs() > 80.0 {
        f.signum() * 0.5 + 0.5
    } else {
        // these values are only valid for -88 < x < 88
        1.0 / (1.0 + fast_math::exp_raw(-f))
    }
}

/// Converts ModelPack segmentation into a 2D mask.
/// The input segmentation is expected to have shape (H, W, num_classes).
///
/// The output mask will have shape (H, W), with values `0..num_classes` based
/// on the argmax across the channels.
///
/// # Panics
/// Panics if the input tensor does not have more than one channel.
pub fn modelpack_segmentation_to_mask(segmentation: ArrayView3<u8>) -> Array2<u8> {
    use argminmax::ArgMinMax;
    assert!(
        segmentation.shape()[2] > 1,
        "Model Instance Segmentation should have shape (H, W, x) where x > 1"
    );
    let height = segmentation.shape()[0];
    let width = segmentation.shape()[1];
    let channels = segmentation.shape()[2];
    let segmentation = segmentation.as_standard_layout();
    // Safe to unwrap since we ensured standard layout above
    let seg = segmentation
        .as_slice()
        .expect("Segmentation is not in standard layout");
    let argmax = seg
        .chunks_exact(channels)
        .map(|x| x.argmax() as u8)
        .collect::<Vec<_>>();

    Array2::from_shape_vec((height, width), argmax).expect("Failed to create mask array")
}
