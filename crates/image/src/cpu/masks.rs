// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::CPUProcessor;
use crate::Result;
use edgefirst_decoder::{DetectBox, Segmentation};
use ndarray::Axis;

impl CPUProcessor {
    pub(super) fn render_modelpack_segmentation(
        &mut self,
        dst_w: usize,
        dst_h: usize,
        dst_rs: usize,
        dst_c: usize,
        dst_slice: &mut [u8],
        segmentation: &Segmentation,
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

                let alpha = color[3] as u16;

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

                let alpha = color[3] as u16;

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
    pub fn materialize_segmentations(
        &self,
        detect: &[crate::DetectBox],
        proto_data: &crate::ProtoData,
    ) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
        if detect.is_empty() || proto_data.mask_coefficients.is_empty() {
            return Ok(Vec::new());
        }

        let protos_cow = proto_data.protos.as_f32();
        let protos = protos_cow.as_ref();
        let proto_h = protos.shape()[0];
        let proto_w = protos.shape()[1];
        let num_protos = protos.shape()[2];

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

                // Extract proto ROI and compute mask_coeff @ protos
                let roi = protos.slice(ndarray::s![y0..y0 + roi_h, x0..x0 + roi_w, ..]);
                let coeff_arr = ndarray::Array2::from_shape_vec((1, num_protos), coeff.clone())
                    .map_err(|e| crate::Error::Internal(format!("mask coeff shape: {e}")))?;
                let protos_2d = roi
                    .to_shape((roi_h * roi_w, num_protos))
                    .map_err(|e| crate::Error::Internal(format!("proto reshape: {e}")))?
                    .reversed_axes();
                let mask = coeff_arr.dot(&protos_2d);
                let mask = mask
                    .into_shape_with_order((roi_h, roi_w, 1))
                    .map_err(|e| crate::Error::Internal(format!("mask reshape: {e}")))?
                    .mapv(|x: f32| {
                        let sigmoid = 1.0 / (1.0 + (-x).exp());
                        (sigmoid * 255.0).round() as u8
                    });

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

    /// Renders per-instance grayscale masks from raw prototype data at full
    /// output resolution. Used internally by [`decode_masks_atlas`] to generate
    /// per-detection mask crops that are then packed into the atlas.
    pub(super) fn render_masks_from_protos(
        &mut self,
        detect: &[crate::DetectBox],
        proto_data: crate::ProtoData,
        output_width: usize,
        output_height: usize,
    ) -> Result<Vec<crate::MaskResult>> {
        use crate::FunctionTimer;

        let _timer = FunctionTimer::new("CPUProcessor::render_masks_from_protos");

        if detect.is_empty() || proto_data.mask_coefficients.is_empty() {
            return Ok(Vec::new());
        }

        let protos_cow = proto_data.protos.as_f32();
        let protos = protos_cow.as_ref();
        let proto_h = protos.shape()[0];
        let proto_w = protos.shape()[1];
        let num_protos = protos.shape()[2];

        let mut results = Vec::with_capacity(detect.len());

        for (det, coeff) in detect.iter().zip(proto_data.mask_coefficients.iter()) {
            let start_x = (output_width as f32 * det.bbox.xmin).round() as usize;
            let start_y = (output_height as f32 * det.bbox.ymin).round() as usize;
            // Use span-based rounding to match the numpy reference convention.
            let bbox_w = ((det.bbox.xmax - det.bbox.xmin) * output_width as f32)
                .round()
                .max(1.0) as usize;
            let bbox_h = ((det.bbox.ymax - det.bbox.ymin) * output_height as f32)
                .round()
                .max(1.0) as usize;
            let bbox_w = bbox_w.min(output_width.saturating_sub(start_x));
            let bbox_h = bbox_h.min(output_height.saturating_sub(start_y));

            let mut pixels = vec![0u8; bbox_w * bbox_h];

            for row in 0..bbox_h {
                let y = start_y + row;
                for col in 0..bbox_w {
                    let x = start_x + col;
                    let px = (x as f32 / output_width as f32) * proto_w as f32 - 0.5;
                    let py = (y as f32 / output_height as f32) * proto_h as f32 - 0.5;
                    let acc = bilinear_dot(protos, coeff, num_protos, px, py, proto_w, proto_h);
                    let mask = 1.0 / (1.0 + (-acc).exp());
                    pixels[row * bbox_w + col] = if mask > 0.5 { 255 } else { 0 };
                }
            }

            results.push(crate::MaskResult {
                x: start_x,
                y: start_y,
                w: bbox_w,
                h: bbox_h,
                pixels,
            });
        }

        Ok(results)
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
