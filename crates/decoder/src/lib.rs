//! EdgeFirst HAL - Decoders
use std::ops::{Add, Mul, Sub};

use ndarray::{
    Array2, ArrayView2, Zip,
    parallel::prelude::{IntoParallelIterator, ParallelIterator as _},
    s,
};
use num_traits::AsPrimitive;
mod error;

#[derive(Debug, Copy, Clone)]
pub struct Quantization<T: AsPrimitive<f32>> {
    pub scale: f32,
    pub zero_point: T,
}

pub fn decode_boxes_and_nms_i8(
    score_threshold: f32,
    iou_threshold: f32,
    output: Array2<i8>,
    num_classes: usize,
    quant: &Quantization<i8>,
    output_boxes: &mut Vec<DetectBox>,
) {
    let score_threshold = (score_threshold / quant.scale + quant.zero_point as f32) as i8;
    let boxes_tensor = output.slice(s![..4, ..,]);
    let scores_tensor = output.slice(s![4..(num_classes + 4), ..,]);
    let boxes = decode_boxes_i8(
        score_threshold,
        scores_tensor,
        boxes_tensor,
        num_classes,
        quant,
    );
    let boxes = nms_i16(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.iter().take(len) {
        output_boxes.push(dequant_detect_box(b, quant));
    }
}

pub fn decode_boxes_and_nms_f32(
    score_threshold: f32,
    iou_threshold: f32,
    output: Array2<f32>,
    num_classes: usize,
    output_boxes: &mut Vec<DetectBox>,
) {
    let boxes_tensor = output.slice(s![..4, ..,]);
    let scores_tensor = output.slice(s![4..(num_classes + 4), ..,]);

    let boxes = decode_boxes_f32(score_threshold, scores_tensor, boxes_tensor, num_classes);

    let boxes = nms_f32(iou_threshold, boxes);
    let len = output_boxes.capacity().min(boxes.len());
    output_boxes.clear();
    for b in boxes.into_iter().take(len) {
        output_boxes.push(b);
    }
}

pub fn decode_boxes_f32(
    threshold: f32,
    scores: ArrayView2<f32>,
    boxes: ArrayView2<f32>,
    num_classes: usize,
) -> Vec<DetectBox> {
    assert_eq!(scores.len() / num_classes, boxes.len() / 4);
    Zip::from(scores.columns())
        .and(boxes.columns())
        .into_par_iter()
        .filter_map(|(score, bbox)| {
            let (score_, label) =
                score
                    .iter()
                    .enumerate()
                    .fold((score[0], 0), |(max, arg_max), (ind, s)| {
                        if max > *s { (max, arg_max) } else { (*s, ind) }
                    });
            if score_ < threshold {
                return None;
            }
            Some(DetectBox {
                label,
                score: score_,
                xmin: bbox[0] - bbox[2] * 0.5,
                ymin: bbox[1] - bbox[3] * 0.5,
                xmax: bbox[0] + bbox[2] * 0.5,
                ymax: bbox[1] + bbox[3] * 0.5,
            })
        })
        .collect()
}

pub fn decode_boxes_u8(
    threshold: u8,
    scores: ArrayView2<u8>,
    boxes: ArrayView2<u8>,
    num_classes: usize,
    quant: &Quantization<u8>,
) -> Vec<DetectBoxQuantized<i16>> {
    assert_eq!(scores.len() / num_classes, boxes.len() / 4);
    let zp = quant.zero_point as i16;
    Zip::from(scores.columns())
        .and(boxes.columns())
        .into_par_iter()
        .filter_map(|(score, bbox)| {
            let (score_, label) =
                score
                    .iter()
                    .enumerate()
                    .fold((score[0], 0), |(max, arg_max), (ind, s)| {
                        if max > *s { (max, arg_max) } else { (*s, ind) }
                    });
            if score_ < threshold {
                return None;
            }
            Some(DetectBoxQuantized::<i16> {
                label,
                score: score_ as i16,
                xmin: 2 * (bbox[0] as i16 - zp) - (bbox[2] as i16 - zp),
                ymin: 2 * (bbox[1] as i16 - zp) - (bbox[3] as i16 - zp),
                xmax: 2 * (bbox[0] as i16 - zp) + (bbox[2] as i16 - zp),
                ymax: 2 * (bbox[1] as i16 - zp) + (bbox[3] as i16 - zp),
            })
        })
        .collect()
}

pub fn decode_boxes_i8(
    threshold: i8,
    scores: ArrayView2<i8>,
    boxes: ArrayView2<i8>,
    num_classes: usize,
    quant: &Quantization<i8>,
) -> Vec<DetectBoxQuantized<i16>> {
    assert_eq!(scores.len() / num_classes, boxes.len() / 4);
    let zp = quant.zero_point as i16;
    Zip::from(scores.columns())
        .and(boxes.columns())
        .into_par_iter()
        .filter_map(|(score, bbox)| {
            let (score_, label) =
                score
                    .iter()
                    .enumerate()
                    .fold((score[0], 0), |(max, arg_max), (ind, s)| {
                        if max > *s { (max, arg_max) } else { (*s, ind) }
                    });
            if score_ < threshold {
                return None;
            }
            Some(DetectBoxQuantized::<i16> {
                label,
                score: score_ as i16,
                xmin: 2 * (bbox[0] as i16 - zp) - (bbox[2] as i16 - zp),
                ymin: 2 * (bbox[1] as i16 - zp) - (bbox[3] as i16 - zp),
                xmax: 2 * (bbox[0] as i16 - zp) + (bbox[2] as i16 - zp),
                ymax: 2 * (bbox[1] as i16 - zp) + (bbox[3] as i16 - zp),
            })
        })
        .collect()
}

pub fn nms_i16(iou: f32, mut boxes: Vec<DetectBoxQuantized<i16>>) -> Vec<DetectBoxQuantized<i16>> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.sort_by(|a, b| b.score.cmp(&a.score));
    let min_val = i16::MIN;
    // Outer loop over all boxes.
    for i in 0..boxes.len() {
        if boxes[i].score <= min_val {
            // this box was merged with a different box earlier
            continue;
        }
        for j in (i + 1)..boxes.len() {
            // Inner loop over boxes with lower score (later in the list).

            if boxes[j].score <= min_val {
                // this box was suppressed by different box earlier
                continue;
            }
            if jaccard_i16(&boxes[j], &boxes[i]) > iou {
                // suppress this box
                boxes[j].score = min_val;
            }
        }
    }

    // Filter out boxes that were suppressed.
    boxes.into_iter().filter(|b| b.score > min_val).collect()
}

pub fn nms_f32(iou: f32, mut boxes: Vec<DetectBox>) -> Vec<DetectBox> {
    // Boxes get sorted by score in descending order so we know based on the
    // index the scoring of the boxes and can skip parts of the loop.
    boxes.sort_by(|a, b| b.score.total_cmp(&a.score));
    // Outer loop over all boxes.
    for i in 0..boxes.len() {
        if boxes[i].score <= 0.0 {
            // this box was merged with a different box earlier
            continue;
        }
        for j in (i + 1)..boxes.len() {
            // Inner loop over boxes with lower score (later in the list).

            if boxes[j].score <= 0.0 {
                // this box was suppressed by different box earlier
                continue;
            }
            if jaccard_f32(&boxes[j], &boxes[i]) > iou {
                // max_box(boxes[j].bbox, &mut boxes[i].bbox);
                boxes[j].score = 0.0;
            }
        }
    }

    // Filter out boxes with a score of 0.0.
    boxes.into_iter().filter(|b| b.score > 0.0).collect()
}

fn jaccard_f32(a: &DetectBox, b: &DetectBox) -> f32 {
    let left = a.xmin.max(b.xmin);
    let top = a.ymin.max(b.ymin);
    let right = a.xmax.min(b.xmax);
    let bottom = a.ymax.min(b.ymax);

    let intersection = (right - left).max(0.0) * (bottom - top).max(0.0);
    let area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    let area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin);

    // need to make sure we are not dividing by zero
    let union = (area_a + area_b - intersection).max(0.0000001);

    intersection / union
}

fn jaccard_i16(a: &DetectBoxQuantized<i16>, b: &DetectBoxQuantized<i16>) -> f32 {
    let left = a.xmin.max(b.xmin);
    let top = a.ymin.max(b.ymin);
    let right = a.xmax.min(b.xmax);
    let bottom = a.ymax.min(b.ymax);

    let intersection = (right - left).max(0) as i32 * (bottom - top).max(0) as i32;
    let area_a = (a.xmax - a.xmin) as i32 * (a.ymax - a.ymin) as i32;
    let area_b = (b.xmax - b.xmin) as i32 * (b.ymax - b.ymin) as i32;

    // need to make sure we are not dividing by zero
    let union = area_a + area_b - intersection;
    if union <= 0 {
        return 0.0;
    }
    intersection as f32 / union as f32
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DetectBox {
    #[doc = " left-most normalized coordinate of the bounding box."]
    pub xmin: f32,
    #[doc = " top-most normalized coordinate of the bounding box."]
    pub ymin: f32,
    #[doc = " right-most normalized coordinate of the bounding box."]
    pub xmax: f32,
    #[doc = " bottom-most normalized coordinate of the bounding box."]
    pub ymax: f32,
    #[doc = " model-specific score for this detection, higher implies more confidence."]
    pub score: f32,
    #[doc = " label index for this detection, text representation can be retrived using\n @ref VAALContext::vaal_label()"]
    pub label: usize,
}

impl DetectBox {
    // Check if one detect box is equal to another detect box, within the given
    // delta
    pub fn equal_within_delta(&self, rhs: &DetectBox, delta: f32) -> bool {
        let eq_delta = |a: f32, b: f32| (a - b).abs() <= delta;
        self.label == rhs.label
            && eq_delta(self.score, rhs.score)
            && eq_delta(self.xmin, rhs.xmin)
            && eq_delta(self.ymin, rhs.ymin)
            && eq_delta(self.xmax, rhs.xmax)
            && eq_delta(self.ymax, rhs.ymax)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectBoxQuantized<T: Clone + Mul + Add + Sub + Ord + AsPrimitive<f32>> {
    #[doc = " 2x the left-most coordinate of the bounding box. Should already be scaled to zero point"]
    pub xmin: T,
    #[doc = " 2x top-most coordinate of the bounding box. Should already be scaled to zero point"]
    pub ymin: T,
    #[doc = " 2x right-most coordinate of the bounding box. Should already be scaled to zero point"]
    pub xmax: T,
    #[doc = " 2x bottom-most coordinate of the bounding box. Should already be scaled to zero point"]
    pub ymax: T,
    #[doc = " model-specific score for this detection, higher implies more confidence."]
    pub score: T,
    #[doc = " label index for this detection, text representation can be retrived using\n @ref VAALContext::vaal_label()"]
    pub label: usize,
}

pub fn dequant_detect_box<
    T: Clone + Mul + Add + Sub + Ord + AsPrimitive<f32>,
    Q: AsPrimitive<f32>,
>(
    detect: &DetectBoxQuantized<T>,
    quant: &Quantization<Q>,
) -> DetectBox {
    let scaled_zp = -quant.scale * quant.zero_point.as_();
    DetectBox {
        xmin: quant.scale * detect.xmin.as_() * 0.5,
        ymin: quant.scale * detect.ymin.as_() * 0.5,
        xmax: quant.scale * detect.xmax.as_() * 0.5,
        ymax: quant.scale * detect.ymax.as_() * 0.5,
        score: quant.scale * detect.score.as_() + scaled_zp,
        label: detect.label,
    }
}

pub fn dequantize_cpu<T: AsPrimitive<f32>>(
    quant: &Quantization<T>,
    input: &[T],
    output: &mut [f32],
) {
    assert!(input.len() == output.len());
    let zero_point = quant.zero_point.as_();
    let scale = quant.scale;
    if zero_point != 0.0 {
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

pub fn dequantize_cpu_chunked<T: AsPrimitive<f32>>(
    quant: &Quantization<T>,
    input: &[T],
    output: &mut [f32],
) {
    assert!(input.len() == output.len());
    let zero_point = quant.zero_point.as_();
    let scale = quant.scale;

    if zero_point != 0.0 {
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

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_decoder_i8() {
        let score_threshold = 0.25;
        let iou_threshold = 0.7;
        let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
        let out = ndarray::Array2::from_shape_vec((84, 8400), out.to_vec()).unwrap();
        let quant = Quantization::<i8> {
            scale: 0.0040811873,
            zero_point: -123i8,
        };
        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        decode_boxes_and_nms_i8(
            score_threshold,
            iou_threshold,
            out,
            80,
            &quant,
            &mut output_boxes,
        );
        println!("output_boxes {output_boxes:?}");
        assert!(output_boxes[0].equal_within_delta(
            &DetectBox {
                xmin: 0.5285137,
                ymin: 0.05305544,
                xmax: 0.87541467,
                ymax: 0.9998909,
                score: 0.5591227,
                label: 0
            },
            1e-6
        ));

        assert!(output_boxes[1].equal_within_delta(
            &DetectBox {
                xmin: 0.130598,
                ymin: 0.43260583,
                xmax: 0.35098213,
                ymax: 0.9958097,
                score: 0.33057618,
                label: 75
            },
            1e-6
        ))
    }

    #[test]
    fn test_decoder_f32() {
        let score_threshold = 0.25;
        let iou_threshold = 0.7;
        let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
        let mut out_dequant = vec![0.0; 84 * 8400];

        let quant = Quantization::<i8> {
            scale: 0.0040811873,
            zero_point: -123i8,
        };
        dequantize_cpu(&quant, out, &mut out_dequant);
        let out = ndarray::Array2::from_shape_vec((84, 8400), out_dequant).unwrap();

        let mut output_boxes: Vec<_> = Vec::with_capacity(50);
        decode_boxes_and_nms_f32(score_threshold, iou_threshold, out, 80, &mut output_boxes);
        println!("output_boxes {output_boxes:?}");
        assert!(output_boxes[0].equal_within_delta(
            &DetectBox {
                xmin: 0.5285137,
                ymin: 0.05305544,
                xmax: 0.87541467,
                ymax: 0.9998909,
                score: 0.5591227,
                label: 0
            },
            1e-6
        ));

        assert!(output_boxes[1].equal_within_delta(
            &DetectBox {
                xmin: 0.130598,
                ymin: 0.43260583,
                xmax: 0.35098213,
                ymax: 0.9958097,
                score: 0.33057618,
                label: 75
            },
            1e-6
        ))
    }

    #[test]
    fn test_dequant_chunked() {
        let out = include_bytes!("../../../testdata/yolov8s_80_classes.bin");
        let out = unsafe { std::slice::from_raw_parts(out.as_ptr() as *const i8, out.len()) };
        let mut out_dequant = vec![0.0; 84 * 8400];
        let mut out_dequant_simd = vec![0.0; 84 * 8400];
        let quant = Quantization::<i8> {
            scale: 0.0040811873,
            zero_point: -123i8,
        };
        dequantize_cpu(&quant, out, &mut out_dequant);

        dequantize_cpu_chunked(&quant, out, &mut out_dequant_simd);

        assert_eq!(out_dequant, out_dequant_simd);
    }
}
