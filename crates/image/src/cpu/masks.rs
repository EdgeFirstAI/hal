// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::CPUProcessor;
use crate::Result;
use edgefirst_decoder::{DetectBox, Segmentation};
use ndarray::Axis;

impl CPUProcessor {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_modelpack_segmentation(
        &mut self,
        dst_w: usize,
        dst_h: usize,
        dst_rs: usize,
        dst_c: usize,
        dst_slice: &mut [u8],
        segmentation: &Segmentation,
        opacity: f32,
    ) -> Result<()> {
        use ndarray_stats::QuantileExt;

        let seg = &segmentation.segmentation;
        let [seg_height, seg_width, seg_classes] = *seg.shape() else {
            unreachable!("Array3 did not have [usize; 3] as shape");
        };
        let start_y = (dst_h as f32 * segmentation.ymin).round();
        let end_y = (dst_h as f32 * segmentation.ymax).round();
        let start_x = (dst_w as f32 * segmentation.xmin).round();
        let end_x = (dst_w as f32 * segmentation.xmax).round();

        let scale_x = (seg_width as f32 - 1.0) / ((end_x - start_x) - 1.0);
        let scale_y = (seg_height as f32 - 1.0) / ((end_y - start_y) - 1.0);

        let start_x_u = (start_x as usize).min(dst_w);
        let start_y_u = (start_y as usize).min(dst_h);
        let end_x_u = (end_x as usize).min(dst_w);
        let end_y_u = (end_y as usize).min(dst_h);

        let argmax = seg.map_axis(Axis(2), |r| r.argmax().unwrap());
        let get_value_at_nearest = |x: f32, y: f32| -> usize {
            let x = x.round() as usize;
            let y = y.round() as usize;
            argmax
                .get([y.min(seg_height - 1), x.min(seg_width - 1)])
                .copied()
                .unwrap_or(0)
        };

        for y in start_y_u..end_y_u {
            for x in start_x_u..end_x_u {
                let seg_x = (x as f32 - start_x) * scale_x;
                let seg_y = (y as f32 - start_y) * scale_y;
                let label = get_value_at_nearest(seg_x, seg_y);

                if label == seg_classes - 1 {
                    continue;
                }

                let color = self.colors[label % self.colors.len()];

                let alpha = if opacity == 1.0 {
                    color[3] as u16
                } else {
                    (color[3] as f32 * opacity).round() as u16
                };

                let dst_index = (y * dst_rs) + (x * dst_c);
                for c in 0..3 {
                    dst_slice[dst_index + c] = ((color[c] as u16 * alpha
                        + dst_slice[dst_index + c] as u16 * (255 - alpha))
                        / 255) as u8;
                }
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_yolo_segmentation(
        &mut self,
        dst_w: usize,
        dst_h: usize,
        dst_rs: usize,
        dst_c: usize,
        dst_slice: &mut [u8],
        segmentation: &Segmentation,
        class: usize,
        opacity: f32,
    ) -> Result<()> {
        let seg = &segmentation.segmentation;
        let [seg_height, seg_width, classes] = *seg.shape() else {
            unreachable!("Array3 did not have [usize;3] as shape");
        };
        debug_assert_eq!(classes, 1);

        let start_y = (dst_h as f32 * segmentation.ymin).round();
        let end_y = (dst_h as f32 * segmentation.ymax).round();
        let start_x = (dst_w as f32 * segmentation.xmin).round();
        let end_x = (dst_w as f32 * segmentation.xmax).round();

        let scale_x = (seg_width as f32 - 1.0) / ((end_x - start_x) - 1.0);
        let scale_y = (seg_height as f32 - 1.0) / ((end_y - start_y) - 1.0);

        let start_x_u = (start_x as usize).min(dst_w);
        let start_y_u = (start_y as usize).min(dst_h);
        let end_x_u = (end_x as usize).min(dst_w);
        let end_y_u = (end_y as usize).min(dst_h);

        for y in start_y_u..end_y_u {
            for x in start_x_u..end_x_u {
                let seg_x = ((x as f32 - start_x) * scale_x) as usize;
                let seg_y = ((y as f32 - start_y) * scale_y) as usize;
                let val = *seg.get([seg_y, seg_x, 0]).unwrap_or(&0);

                if val < 127 {
                    continue;
                }

                let color = self.colors[class % self.colors.len()];

                let alpha = if opacity == 1.0 {
                    color[3] as u16
                } else {
                    (color[3] as f32 * opacity).round() as u16
                };

                let dst_index = (y * dst_rs) + (x * dst_c);
                for c in 0..3 {
                    dst_slice[dst_index + c] = ((color[c] as u16 * alpha
                        + dst_slice[dst_index + c] as u16 * (255 - alpha))
                        / 255) as u8;
                }
            }
        }

        Ok(())
    }

    pub(super) fn render_box(
        &mut self,
        dst_w: usize,
        dst_h: usize,
        dst_rs: usize,
        dst_c: usize,
        dst_slice: &mut [u8],
        detect: &[DetectBox],
    ) -> Result<()> {
        const LINE_THICKNESS: usize = 3;

        for d in detect {
            use edgefirst_decoder::BoundingBox;

            let label = d.label;
            let [r, g, b, _] = self.colors[label % self.colors.len()];
            let bbox = d.bbox.to_canonical();
            let bbox = BoundingBox {
                xmin: bbox.xmin.clamp(0.0, 1.0),
                ymin: bbox.ymin.clamp(0.0, 1.0),
                xmax: bbox.xmax.clamp(0.0, 1.0),
                ymax: bbox.ymax.clamp(0.0, 1.0),
            };
            let inner = [
                ((dst_w - 1) as f32 * bbox.xmin - 0.5).round() as usize,
                ((dst_h - 1) as f32 * bbox.ymin - 0.5).round() as usize,
                ((dst_w - 1) as f32 * bbox.xmax + 0.5).round() as usize,
                ((dst_h - 1) as f32 * bbox.ymax + 0.5).round() as usize,
            ];

            let outer = [
                inner[0].saturating_sub(LINE_THICKNESS),
                inner[1].saturating_sub(LINE_THICKNESS),
                (inner[2] + LINE_THICKNESS).min(dst_w),
                (inner[3] + LINE_THICKNESS).min(dst_h),
            ];

            // top line
            for y in outer[1] + 1..=inner[1] {
                for x in outer[0] + 1..outer[2] {
                    let index = (y * dst_rs) + (x * dst_c);
                    dst_slice[index..(index + 3)].copy_from_slice(&[r, g, b]);
                }
            }

            // left and right lines
            for y in inner[1]..inner[3] {
                for x in outer[0] + 1..=inner[0] {
                    let index = (y * dst_rs) + (x * dst_c);
                    dst_slice[index..(index + 3)].copy_from_slice(&[r, g, b]);
                }

                for x in inner[2]..outer[2] {
                    let index = (y * dst_rs) + (x * dst_c);
                    dst_slice[index..(index + 3)].copy_from_slice(&[r, g, b]);
                }
            }

            // bottom line
            for y in inner[3]..outer[3] {
                for x in outer[0] + 1..outer[2] {
                    let index = (y * dst_rs) + (x * dst_c);
                    dst_slice[index..(index + 3)].copy_from_slice(&[r, g, b]);
                }
            }
        }
        Ok(())
    }

    /// Materialize segmentation masks from proto data into `Vec<Segmentation>`.
    ///
    /// This is the CPU-side decode step of the hybrid mask rendering path:
    /// call this to get pre-decoded masks, then pass them to
    /// [`draw_masks`](crate::ImageProcessorTrait::draw_masks) for GPU overlay.
    /// Benchmarks show this hybrid path (CPU decode + GL overlay) is faster
    /// than the fused GPU `draw_masks_proto` on all tested platforms.
    ///
    /// Optimized: fused dequantization + dot product avoids a 3.1MB f32
    /// allocation for the full proto tensor. Uses fast sigmoid approximation.
    pub fn materialize_segmentations(
        &self,
        detect: &[crate::DetectBox],
        proto_data: &crate::ProtoData,
    ) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
        use edgefirst_decoder::ProtoTensor;

        if detect.is_empty() || proto_data.mask_coefficients.is_empty() {
            return Ok(Vec::new());
        }

        // Extract proto tensor metadata for the fused kernel.
        let (proto_h, proto_w, num_protos) = match &proto_data.protos {
            ProtoTensor::Quantized { protos, .. } => {
                (protos.shape()[0], protos.shape()[1], protos.shape()[2])
            }
            ProtoTensor::Float(arr) => (arr.shape()[0], arr.shape()[1], arr.shape()[2]),
        };

        detect
            .iter()
            .zip(proto_data.mask_coefficients.iter())
            .map(|(det, coeff)| {
                // Clamp bbox to [0, 1]
                let xmin = det.bbox.xmin.clamp(0.0, 1.0);
                let ymin = det.bbox.ymin.clamp(0.0, 1.0);
                let xmax = det.bbox.xmax.clamp(0.0, 1.0);
                let ymax = det.bbox.ymax.clamp(0.0, 1.0);

                // Map to proto-space pixel coordinates (clamp to valid range)
                let x0 = ((xmin * proto_w as f32) as usize).min(proto_w.saturating_sub(1));
                let y0 = ((ymin * proto_h as f32) as usize).min(proto_h.saturating_sub(1));
                let x1 = ((xmax * proto_w as f32).ceil() as usize).min(proto_w);
                let y1 = ((ymax * proto_h as f32).ceil() as usize).min(proto_h);

                let roi_w = x1.saturating_sub(x0).max(1);
                let roi_h = y1.saturating_sub(y0).max(1);

                if coeff.len() != num_protos {
                    return Err(crate::Error::Internal(format!(
                        "mask coeff length {} != proto channels {num_protos}",
                        coeff.len()
                    )));
                }

                // Fused dequant + dot product + sigmoid, directly producing u8 mask.
                // Avoids allocating a full f32 proto tensor and the to_shape copy.
                let mask = match &proto_data.protos {
                    ProtoTensor::Quantized {
                        protos,
                        quantization,
                    } => {
                        let scale = quantization.scale;
                        let zp = quantization.zero_point as f32;
                        fused_dequant_dot_sigmoid_i8(
                            protos, coeff, scale, zp, y0, x0, roi_h, roi_w, num_protos,
                        )
                    }
                    ProtoTensor::Float(protos) => {
                        fused_dot_sigmoid_f32(protos, coeff, y0, x0, roi_h, roi_w, num_protos)
                    }
                };

                Ok(edgefirst_decoder::Segmentation {
                    xmin: x0 as f32 / proto_w as f32,
                    ymin: y0 as f32 / proto_h as f32,
                    xmax: x1 as f32 / proto_w as f32,
                    ymax: y1 as f32 / proto_h as f32,
                    segmentation: mask,
                })
            })
            .collect::<crate::Result<Vec<_>>>()
    }
}

/// Bilinear interpolation of proto values at `(px, py)` combined with dot
/// product against `coeff`. Returns the scalar accumulator before sigmoid.
///
/// Samples the four nearest proto texels, weights by bilinear coefficients,
/// and simultaneously computes the dot product with the mask coefficients.
#[inline]
pub(super) fn bilinear_dot(
    protos: &ndarray::Array3<f32>,
    coeff: &[f32],
    num_protos: usize,
    px: f32,
    py: f32,
    proto_w: usize,
    proto_h: usize,
) -> f32 {
    let x0 = (px.floor() as isize).clamp(0, proto_w as isize - 1) as usize;
    let y0 = (py.floor() as isize).clamp(0, proto_h as isize - 1) as usize;
    let x1 = (x0 + 1).min(proto_w - 1);
    let y1 = (y0 + 1).min(proto_h - 1);

    let fx = px - px.floor();
    let fy = py - py.floor();

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let mut acc = 0.0f32;
    for p in 0..num_protos {
        let val = w00 * protos[[y0, x0, p]]
            + w10 * protos[[y0, x1, p]]
            + w01 * protos[[y1, x0, p]]
            + w11 * protos[[y1, x1, p]];
        acc += coeff[p] * val;
    }
    acc
}

/// Fast sigmoid approximation: `1 / (1 + exp(-x))`.
///
/// Uses bit-manipulation for exp() (same approach as `fast_math::exp_raw`).
/// Max relative error < 1.1% for normal results, which is well within the
/// precision needed for mask thresholding and u8 quantization.
#[inline(always)]
fn fast_sigmoid(x: f32) -> f32 {
    if x >= 16.0 {
        return 1.0;
    }
    if x <= -16.0 {
        return 0.0;
    }
    // Fast exp(-x) via bit manipulation (Schraudolph's algorithm).
    // f32 bits: 2^23 * log2(e) * x + (127 << 23) approximates exp(x).
    const A: f32 = (1u32 << 23) as f32; // 8388608.0
    const B: f32 = A * std::f32::consts::LOG2_E; // A / ln(2)
    const C: u32 = 127 << 23; // exponent bias
    let neg_x = -x;
    let bits = (B * neg_x) as i32 + C as i32;
    let exp_neg_x = f32::from_bits(bits as u32);
    1.0 / (1.0 + exp_neg_x)
}

/// Fused dequantization + dot product + sigmoid for quantized (i8) protos.
///
/// For each pixel in the ROI, computes:
///   acc = sum_k(coeff[k] * (proto[y, x, k] as f32 - zp) * scale)
///   mask[y, x] = fast_sigmoid(acc) * 255
///
/// This avoids allocating a full f32 proto tensor (3.1MB for 160x160x32)
/// and the hidden `to_shape` copy on non-contiguous ROI slices.
#[allow(clippy::too_many_arguments)]
fn fused_dequant_dot_sigmoid_i8(
    protos: &ndarray::Array3<i8>,
    coeff: &[f32],
    scale: f32,
    zp: f32,
    y0: usize,
    x0: usize,
    roi_h: usize,
    roi_w: usize,
    num_protos: usize,
) -> ndarray::Array3<u8> {
    // Pre-scale coefficients: coeff[k] * scale, so the inner loop is
    // just fma: acc += scaled_coeff[k] * (proto_i8 - zp)
    let scaled_coeff: Vec<f32> = coeff.iter().map(|&c| c * scale).collect();
    // Pre-compute coeff_sum * (-zp * scale) offset:
    // sum_k(coeff[k] * (proto - zp) * scale) = sum_k(scaled_coeff[k] * proto) - zp * sum_k(scaled_coeff[k])
    // But since zp is a constant per-pixel term, factor it out:
    // acc = sum_k(scaled_coeff[k] * proto_i8_as_f32) - zp * sum_k(scaled_coeff[k])
    let zp_offset: f32 = zp * scaled_coeff.iter().sum::<f32>();

    let proto_stride_y = protos.strides()[0] as usize;
    let proto_stride_x = protos.strides()[1] as usize;
    let proto_stride_k = protos.strides()[2] as usize;
    let proto_ptr = protos.as_ptr();

    let mut mask = ndarray::Array3::<u8>::zeros((roi_h, roi_w, 1));

    for y in 0..roi_h {
        for x in 0..roi_w {
            // Base pointer for protos[y0+y, x0+x, 0]
            let base = (y0 + y) * proto_stride_y + (x0 + x) * proto_stride_x;

            let mut acc = 0.0f32;
            let mut k = 0;

            // Process 4 protos at a time for better ILP
            let chunks = num_protos / 4;
            for _ in 0..chunks {
                unsafe {
                    let p0 = *proto_ptr.add(base + k * proto_stride_k) as f32;
                    let p1 = *proto_ptr.add(base + (k + 1) * proto_stride_k) as f32;
                    let p2 = *proto_ptr.add(base + (k + 2) * proto_stride_k) as f32;
                    let p3 = *proto_ptr.add(base + (k + 3) * proto_stride_k) as f32;
                    acc += scaled_coeff[k] * p0
                        + scaled_coeff[k + 1] * p1
                        + scaled_coeff[k + 2] * p2
                        + scaled_coeff[k + 3] * p3;
                }
                k += 4;
            }
            // Remainder
            while k < num_protos {
                unsafe {
                    let p = *proto_ptr.add(base + k * proto_stride_k) as f32;
                    acc += scaled_coeff[k] * p;
                }
                k += 1;
            }

            acc -= zp_offset;
            let sigmoid = fast_sigmoid(acc);
            mask[[y, x, 0]] = (sigmoid * 255.0 + 0.5) as u8;
        }
    }
    mask
}

/// Fused dot product + sigmoid for f32 protos (no dequantization needed).
fn fused_dot_sigmoid_f32(
    protos: &ndarray::Array3<f32>,
    coeff: &[f32],
    y0: usize,
    x0: usize,
    roi_h: usize,
    roi_w: usize,
    num_protos: usize,
) -> ndarray::Array3<u8> {
    let proto_stride_y = protos.strides()[0] as usize;
    let proto_stride_x = protos.strides()[1] as usize;
    let proto_stride_k = protos.strides()[2] as usize;
    let proto_ptr = protos.as_ptr();

    let mut mask = ndarray::Array3::<u8>::zeros((roi_h, roi_w, 1));

    for y in 0..roi_h {
        for x in 0..roi_w {
            let base = (y0 + y) * proto_stride_y + (x0 + x) * proto_stride_x;

            let mut acc = 0.0f32;
            let mut k = 0;
            let chunks = num_protos / 4;
            for _ in 0..chunks {
                unsafe {
                    let p0 = *proto_ptr.add(base + k * proto_stride_k);
                    let p1 = *proto_ptr.add(base + (k + 1) * proto_stride_k);
                    let p2 = *proto_ptr.add(base + (k + 2) * proto_stride_k);
                    let p3 = *proto_ptr.add(base + (k + 3) * proto_stride_k);
                    acc +=
                        coeff[k] * p0 + coeff[k + 1] * p1 + coeff[k + 2] * p2 + coeff[k + 3] * p3;
                }
                k += 4;
            }
            while k < num_protos {
                unsafe {
                    let p = *proto_ptr.add(base + k * proto_stride_k);
                    acc += coeff[k] * p;
                }
                k += 1;
            }

            let sigmoid = fast_sigmoid(acc);
            mask[[y, x, 0]] = (sigmoid * 255.0 + 0.5) as u8;
        }
    }
    mask
}
