use ndarray::{Array2, ArrayView2, ArrayView3};
use num_traits::AsPrimitive;

use crate::{
    BBoxTypeTrait, DetectBox, Detection, Quantization,
    byte::{nms_i16, postprocess_boxes_8bit, quantize_score_threshold},
    dequant_detect_box,
    error::Result,
    float::{nms_f32, postprocess_boxes_f32},
};

pub struct ModelPackDetectionConfig<D: AsPrimitive<f32>> {
    pub anchors: Vec<[f32; 2]>,
    pub quantization: Option<Quantization<D>>,
}

impl From<&Detection> for ModelPackDetectionConfig<u8> {
    fn from(value: &Detection) -> Self {
        Self {
            anchors: value.anchors.clone().unwrap(),
            quantization: value.quantization.map(|x| x.into()),
        }
    }
}

impl From<&Detection> for ModelPackDetectionConfig<f32> {
    fn from(value: &Detection) -> Self {
        Self {
            anchors: value.anchors.clone().unwrap(),
            quantization: value.quantization.map(|x| x.into()),
        }
    }
}

pub fn decode_modelpack_i8<B: BBoxTypeTrait>(
    boxes_tensor: ArrayView2<i8>,
    scores_tensor: ArrayView2<i8>,
    quant_boxes: &Quantization<i8>,
    quant_scores: &Quantization<i8>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    let score_threshold = quantize_score_threshold(score_threshold, quant_scores);
    let boxes =
        postprocess_boxes_8bit::<B, _>(score_threshold, boxes_tensor, scores_tensor, quant_boxes);
    let boxes = nms_i16(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(dequant_detect_box(&b, quant_boxes, quant_scores));
    }
}

pub fn decode_modelpack_u8<B: BBoxTypeTrait>(
    boxes_tensor: ArrayView2<u8>,
    scores_tensor: ArrayView2<u8>,
    quant_boxes: &Quantization<u8>,
    quant_scores: &Quantization<u8>,
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    let score_threshold = quantize_score_threshold(score_threshold, quant_scores);
    let boxes =
        postprocess_boxes_8bit::<B, _>(score_threshold, boxes_tensor, scores_tensor, quant_boxes);
    let boxes = nms_i16(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(dequant_detect_box(&b, quant_boxes, quant_scores));
    }
}

pub fn decode_modelpack_split<B: BBoxTypeTrait, D: AsPrimitive<f32>>(
    outputs: &[ArrayView3<D>],
    configs: &[ModelPackDetectionConfig<D>],
    score_threshold: f32,
    iou_threshold: f32,
    output_boxes: &mut Vec<DetectBox>,
) {
    let (boxes_tensor, scores_tensor) = postprocess_modelpack_split(outputs, configs).unwrap();
    let boxes =
        postprocess_boxes_f32::<B>(score_threshold, boxes_tensor.view(), scores_tensor.view());
    let boxes = nms_f32(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(b);
    }
}

pub fn postprocess_modelpack_split<T: AsPrimitive<f32>>(
    outputs: &[ArrayView3<T>],
    config: &[ModelPackDetectionConfig<T>],
) -> Result<(Array2<f32>, Array2<f32>)> {
    let mut total_capacity = 0;
    let mut nc = 0;
    for (p, detail) in outputs.iter().zip(config) {
        let shape = p.shape();
        let na = detail.anchors.len();
        nc = *shape.last().unwrap() / na - 5;
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
            let scaled_zero = -quant.zero_point.as_() * quant.scale;
            p.mapv(|x| fast_sigmoid_impl(x.as_() * quant.scale + scaled_zero))
        } else {
            p.mapv(|x| fast_sigmoid_impl(x.as_()))
        };
        let p = p_sigmoid.as_slice().unwrap();
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

            let obj = p[4];
            let probs = p[5..].iter().map(|x| *x * obj);
            bscores.extend(probs);
        }
    }
    let bboxes = Array2::from_shape_vec((bboxes.len() / 4, 4), bboxes).unwrap();
    let bscores = Array2::from_shape_vec((bscores.len() / nc, nc), bscores).unwrap();
    Ok((bboxes, bscores))
}

#[inline]
pub fn fast_sigmoid(f: &mut f32) {
    *f = fast_sigmoid_impl(*f);
}

#[inline(always)]
pub fn fast_sigmoid_impl(f: f32) -> f32 {
    if f.abs() > 80.0 {
        f.signum() * 0.5 + 0.5
    } else {
        // these values are only valid for -88 < x < 88
        1.0 / (1.0 + fast_math::exp_raw(-f))
    }
}
