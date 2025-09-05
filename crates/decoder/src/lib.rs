//! EdgeFirst HAL - Decoders
use std::ops::{Add, Mul, Sub};

use num_traits::AsPrimitive;

pub mod bits8;
mod error;
pub mod float;

pub use error::Error;

#[derive(Debug, Copy, Clone)]
pub struct Quantization<T: AsPrimitive<f32>> {
    pub scale: f32,
    pub zero_point: T,
}

#[derive(Debug, Copy, Clone)]
pub struct QuantizationF64<T: AsPrimitive<f64>> {
    pub scale: f64,
    pub zero_point: T,
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

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DetectBoxF64 {
    #[doc = " left-most normalized coordinate of the bounding box."]
    pub xmin: f64,
    #[doc = " top-most normalized coordinate of the bounding box."]
    pub ymin: f64,
    #[doc = " right-most normalized coordinate of the bounding box."]
    pub xmax: f64,
    #[doc = " bottom-most normalized coordinate of the bounding box."]
    pub ymax: f64,
    #[doc = " model-specific score for this detection, higher implies more confidence."]
    pub score: f64,
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

pub fn dequantize_cpu_f64<T: AsPrimitive<f64>>(
    quant: &QuantizationF64<T>,
    input: &[T],
    output: &mut [f64],
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

pub fn dequantize_cpu_chunked_f64<T: AsPrimitive<f64>>(
    quant: &QuantizationF64<T>,
    input: &[T],
    output: &mut [f64],
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
    use crate::{bits8::decode_i8, float::decode_f32, *};

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
        decode_i8(
            out.view(),
            80,
            &quant,
            score_threshold,
            iou_threshold,
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
        decode_f32(
            out.view(),
            80,
            score_threshold,
            iou_threshold,
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
