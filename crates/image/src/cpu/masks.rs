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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn render_box(
        &mut self,
        dst_w: usize,
        dst_h: usize,
        dst_rs: usize,
        dst_c: usize,
        dst_slice: &mut [u8],
        detect: &[DetectBox],
        color_mode: crate::ColorMode,
    ) -> Result<()> {
        const LINE_THICKNESS: usize = 3;

        for (idx, d) in detect.iter().enumerate() {
            use edgefirst_decoder::BoundingBox;

            let color_index = color_mode.index(idx, d.label);
            let [r, g, b, _] = self.colors[color_index % self.colors.len()];
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
    /// [`draw_decoded_masks`](crate::ImageProcessorTrait::draw_decoded_masks) for GPU overlay.
    /// Benchmarks show this hybrid path (CPU decode + GL overlay) is faster
    /// than the fused GPU `draw_proto_masks` on all tested platforms.
    ///
    /// Optimized: fused dequantization + dot product avoids a 3.1MB f32
    /// allocation for the full proto tensor. Uses fast sigmoid approximation.
    pub fn materialize_segmentations(
        &self,
        detect: &[crate::DetectBox],
        proto_data: &crate::ProtoData,
        letterbox: Option<[f32; 4]>,
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
            ProtoTensor::Float16(arr) => (arr.shape()[0], arr.shape()[1], arr.shape()[2]),
            ProtoTensor::Float(arr) => (arr.shape()[0], arr.shape()[1], arr.shape()[2]),
        };

        // Precompute inverse letterbox scale for output-coord conversion.
        let (lx0, inv_lw, ly0, inv_lh) = match letterbox {
            Some([lx0, ly0, lx1, ly1]) => {
                let lw = lx1 - lx0;
                let lh = ly1 - ly0;
                (
                    lx0,
                    if lw > 0.0 { 1.0 / lw } else { 1.0 },
                    ly0,
                    if lh > 0.0 { 1.0 / lh } else { 1.0 },
                )
            }
            None => (0.0_f32, 1.0_f32, 0.0_f32, 1.0_f32),
        };

        detect
            .iter()
            .zip(proto_data.mask_coefficients.iter())
            .map(|(det, coeff)| {
                // Canonicalise bbox (swap min/max if inverted) then clamp to [0,1].
                // Without canonicalisation a degenerate box (ymax < ymin) causes a
                // near-zero ROI height after saturating_sub, making the mask invisible.
                let bbox = det.bbox.to_canonical();
                let xmin = bbox.xmin.clamp(0.0, 1.0);
                let ymin = bbox.ymin.clamp(0.0, 1.0);
                let xmax = bbox.xmax.clamp(0.0, 1.0);
                let ymax = bbox.ymax.clamp(0.0, 1.0);

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
                //
                // For the `Float16` variant we currently widen lazily through
                // `as_f32()` (one allocation per call, bounded by proto size).
                // A dedicated fused f16 kernel would eliminate that allocation
                // but would roughly double the kernel code size — defer until
                // profiling shows the CPU mask path matters on fp16 engines.
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
                    ProtoTensor::Float16(_) => {
                        let widened = proto_data.protos.as_f32();
                        fused_dot_sigmoid_f32(&widened, coeff, y0, x0, roi_h, roi_w, num_protos)
                    }
                };

                // Convert proto-space normalised coords to output-image-normalised
                // coords by applying the inverse letterbox transform.  When no
                // letterbox was used (lx0=0, inv_lw=1, ly0=0, inv_lh=1) this is a
                // no-op and the coords are identical to the proto-normalised values.
                let seg_xmin = ((x0 as f32 / proto_w as f32) - lx0) * inv_lw;
                let seg_ymin = ((y0 as f32 / proto_h as f32) - ly0) * inv_lh;
                let seg_xmax = ((x1 as f32 / proto_w as f32) - lx0) * inv_lw;
                let seg_ymax = ((y1 as f32 / proto_h as f32) - ly0) * inv_lh;

                Ok(edgefirst_decoder::Segmentation {
                    xmin: seg_xmin.clamp(0.0, 1.0),
                    ymin: seg_ymin.clamp(0.0, 1.0),
                    xmax: seg_xmax.clamp(0.0, 1.0),
                    ymax: seg_ymax.clamp(0.0, 1.0),
                    segmentation: mask,
                })
            })
            .collect::<crate::Result<Vec<_>>>()
    }

    /// Produce per-detection masks at `(width, height)` pixel resolution by
    /// upsampling the full proto plane once then cropping per bbox. Each
    /// `det.bbox` is assumed to be in model-input normalized coordinates
    /// (the convention used by the decoder output); when `letterbox` is
    /// `Some`, `(width, height)` are original-content pixel dims and the
    /// inverse letterbox transform is applied to both the bbox (for the
    /// crop region and returned `Segmentation` metadata) and each output
    /// pixel (for proto-plane sampling). Mask values are binary
    /// `uint8 {0, 255}` after thresholding sigmoid > 0.5.
    ///
    /// Used by [`ImageProcessor::materialize_masks`] when the caller selects
    /// [`MaskResolution::Scaled`](crate::MaskResolution::Scaled).
    pub fn materialize_scaled_segmentations(
        &self,
        detect: &[crate::DetectBox],
        proto_data: &crate::ProtoData,
        letterbox: Option<[f32; 4]>,
        width: u32,
        height: u32,
    ) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
        use edgefirst_decoder::ProtoTensor;

        if detect.is_empty() || proto_data.mask_coefficients.is_empty() {
            return Ok(Vec::new());
        }
        if width == 0 || height == 0 {
            return Err(crate::Error::InvalidShape(
                "Scaled mask width/height must be positive".into(),
            ));
        }

        match &proto_data.protos {
            ProtoTensor::Float(protos) => scaled_segmentations_float(
                detect,
                &proto_data.mask_coefficients,
                protos,
                letterbox,
                width,
                height,
            ),
            ProtoTensor::Float16(_) => {
                // Lazy widen — the Cow::Owned allocation is the same size as
                // the f32 proto tensor we'd otherwise materialize ourselves.
                let widened = proto_data.protos.as_f32();
                scaled_segmentations_float(
                    detect,
                    &proto_data.mask_coefficients,
                    &widened,
                    letterbox,
                    width,
                    height,
                )
            }
            ProtoTensor::Quantized {
                protos,
                quantization,
            } => scaled_segmentations_quant_i8(
                detect,
                &proto_data.mask_coefficients,
                protos,
                *quantization,
                letterbox,
                width,
                height,
            ),
        }
    }
}

fn scaled_segmentations_float(
    detect: &[crate::DetectBox],
    mask_coefficients: &[Vec<f32>],
    protos: &ndarray::Array3<f32>,
    letterbox: Option<[f32; 4]>,
    width: u32,
    height: u32,
) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
    let (proto_h, proto_w, num_protos) = (protos.shape()[0], protos.shape()[1], protos.shape()[2]);

    // letterbox = [lx0, ly0, lx1, ly1] in model-input normalized coords.
    // Two uses for the same (lx0, lw) pair:
    //   * Forward (sampling): each output pixel maps to model-input normalized
    //     via model_norm = lx0 + out_norm * lw, then to proto pixels.
    //   * Inverse (bbox frame): det.bbox arrives in model-input normalized
    //     (decoder output); the crop region and returned Segmentation are in
    //     output-content normalized, obtained via (bbox - lx0) / lw.
    // When letterbox is None, (lx0=0, lw=1) makes both directions the identity.
    let (lx0, lw, ly0, lh) = match letterbox {
        Some([lx0, ly0, lx1, ly1]) => {
            let lw = (lx1 - lx0).max(f32::EPSILON);
            let lh = (ly1 - ly0).max(f32::EPSILON);
            (lx0, lw, ly0, lh)
        }
        None => (0.0_f32, 1.0_f32, 0.0_f32, 1.0_f32),
    };

    let out_w = width as usize;
    let out_h = height as usize;

    detect
        .iter()
        .zip(mask_coefficients.iter())
        .map(|(det, coeff)| {
            if coeff.len() != num_protos {
                return Err(crate::Error::Internal(format!(
                    "mask coeff length {} != proto channels {num_protos}",
                    coeff.len()
                )));
            }

            // Canonicalise, then inverse-letterbox into output-content
            // normalized space for bbox crop + Segmentation metadata.
            // Matches the end-of-pipeline transform in the Proto path
            // (see `materialize_segmentations`).
            let bbox = det.bbox.to_canonical();
            let xmin = ((bbox.xmin - lx0) / lw).clamp(0.0, 1.0);
            let ymin = ((bbox.ymin - ly0) / lh).clamp(0.0, 1.0);
            let xmax = ((bbox.xmax - lx0) / lw).clamp(0.0, 1.0);
            let ymax = ((bbox.ymax - ly0) / lh).clamp(0.0, 1.0);

            let px0 = (xmin * out_w as f32).round() as usize;
            let py0 = (ymin * out_h as f32).round() as usize;
            let px1 = ((xmax * out_w as f32).round() as usize).min(out_w);
            let py1 = ((ymax * out_h as f32).round() as usize).min(out_h);
            let bbox_w = px1.saturating_sub(px0).max(1);
            let bbox_h = py1.saturating_sub(py0).max(1);

            let mut tile = ndarray::Array3::<u8>::zeros((bbox_h, bbox_w, 1));

            // Map each output pixel within bbox back to proto-plane
            // coordinates. Center-of-pixel offset (-0.5) matches torch
            // align_corners=False, the Ultralytics retina convention.
            for yi in 0..bbox_h {
                let py = (py0 + yi) as f32;
                let model_y_norm = ly0 + (py + 0.5) / out_h as f32 * lh;
                let sample_y = model_y_norm * proto_h as f32 - 0.5;
                for xi in 0..bbox_w {
                    let px = (px0 + xi) as f32;
                    let model_x_norm = lx0 + (px + 0.5) / out_w as f32 * lw;
                    let sample_x = model_x_norm * proto_w as f32 - 0.5;
                    let acc = bilinear_dot(
                        protos, coeff, num_protos, sample_x, sample_y, proto_w, proto_h,
                    );
                    let sigmoid = fast_sigmoid(acc);
                    tile[[yi, xi, 0]] = if sigmoid > 0.5 { 255 } else { 0 };
                }
            }

            Ok(edgefirst_decoder::Segmentation {
                xmin,
                ymin,
                xmax,
                ymax,
                segmentation: tile,
            })
        })
        .collect::<crate::Result<Vec<_>>>()
}

/// Quantized i8 proto variant of [`scaled_segmentations_float`]. Dequantizes
/// inline during bilinear sample + dot product:
///   dequant = (i8 - zero_point) * scale
/// Factoring scale out of the dot product:
///   sigmoid(scale * Σ coef_k * (i8 - zp))
/// so we can run a single scale multiply on the accumulator.
#[allow(clippy::too_many_arguments)]
fn scaled_segmentations_quant_i8(
    detect: &[crate::DetectBox],
    mask_coefficients: &[Vec<f32>],
    protos: &ndarray::Array3<i8>,
    quant: edgefirst_decoder::Quantization,
    letterbox: Option<[f32; 4]>,
    width: u32,
    height: u32,
) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
    let (proto_h, proto_w, num_protos) = (protos.shape()[0], protos.shape()[1], protos.shape()[2]);

    // See scaled_segmentations_float for the forward / inverse letterbox
    // convention — det.bbox arrives in model-input normalized; crop region
    // and Segmentation metadata are output-content normalized via
    // (bbox - lx0) / lw; the sampling transform goes the other direction.
    let (lx0, lw, ly0, lh) = match letterbox {
        Some([lx0, ly0, lx1, ly1]) => {
            let lw = (lx1 - lx0).max(f32::EPSILON);
            let lh = (ly1 - ly0).max(f32::EPSILON);
            (lx0, lw, ly0, lh)
        }
        None => (0.0_f32, 1.0_f32, 0.0_f32, 1.0_f32),
    };

    let out_w = width as usize;
    let out_h = height as usize;
    let scale = quant.scale;
    let zp = quant.zero_point as f32;

    detect
        .iter()
        .zip(mask_coefficients.iter())
        .map(|(det, coeff)| {
            if coeff.len() != num_protos {
                return Err(crate::Error::Internal(format!(
                    "mask coeff length {} != proto channels {num_protos}",
                    coeff.len()
                )));
            }

            let bbox = det.bbox.to_canonical();
            let xmin = ((bbox.xmin - lx0) / lw).clamp(0.0, 1.0);
            let ymin = ((bbox.ymin - ly0) / lh).clamp(0.0, 1.0);
            let xmax = ((bbox.xmax - lx0) / lw).clamp(0.0, 1.0);
            let ymax = ((bbox.ymax - ly0) / lh).clamp(0.0, 1.0);

            let px0 = (xmin * out_w as f32).round() as usize;
            let py0 = (ymin * out_h as f32).round() as usize;
            let px1 = ((xmax * out_w as f32).round() as usize).min(out_w);
            let py1 = ((ymax * out_h as f32).round() as usize).min(out_h);
            let bbox_w = px1.saturating_sub(px0).max(1);
            let bbox_h = py1.saturating_sub(py0).max(1);

            let mut tile = ndarray::Array3::<u8>::zeros((bbox_h, bbox_w, 1));

            for yi in 0..bbox_h {
                let py = (py0 + yi) as f32;
                let model_y_norm = ly0 + (py + 0.5) / out_h as f32 * lh;
                let sample_y = model_y_norm * proto_h as f32 - 0.5;
                for xi in 0..bbox_w {
                    let px = (px0 + xi) as f32;
                    let model_x_norm = lx0 + (px + 0.5) / out_w as f32 * lw;
                    let sample_x = model_x_norm * proto_w as f32 - 0.5;
                    let acc = bilinear_dot_quant_i8(
                        protos, coeff, num_protos, sample_x, sample_y, proto_w, proto_h, zp,
                    );
                    let sigmoid = fast_sigmoid(scale * acc);
                    tile[[yi, xi, 0]] = if sigmoid > 0.5 { 255 } else { 0 };
                }
            }

            Ok(edgefirst_decoder::Segmentation {
                xmin,
                ymin,
                xmax,
                ymax,
                segmentation: tile,
            })
        })
        .collect::<crate::Result<Vec<_>>>()
}

/// Bilinear sample + zero-point-subtracting dot product over an i8 proto
/// tensor. Returns the scaled-but-not-yet-sigmoid accumulator; caller applies
/// the quantization scale and sigmoid.
#[inline]
#[allow(clippy::too_many_arguments)]
fn bilinear_dot_quant_i8(
    protos: &ndarray::Array3<i8>,
    coeff: &[f32],
    num_protos: usize,
    px: f32,
    py: f32,
    proto_w: usize,
    proto_h: usize,
    zp: f32,
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
        let v00 = protos[[y0, x0, p]] as f32 - zp;
        let v10 = protos[[y0, x1, p]] as f32 - zp;
        let v01 = protos[[y1, x0, p]] as f32 - zp;
        let v11 = protos[[y1, x1, p]] as f32 - zp;
        let val = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
        acc += coeff[p] * val;
    }
    acc
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
    debug_assert!(
        protos.strides().iter().all(|&s| s >= 0),
        "negative strides unsupported"
    );
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
                // SAFETY: bounds are guaranteed by ROI clamping in the caller:
                // y0+y < proto_h, x0+x < proto_w, k+3 < num_protos <= protos.shape()[2].
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
                // SAFETY: bounds are guaranteed by ROI clamping in the caller:
                // y0+y < proto_h, x0+x < proto_w, k < num_protos <= protos.shape()[2].
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
    debug_assert!(
        protos.strides().iter().all(|&s| s >= 0),
        "negative strides unsupported"
    );
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
                // SAFETY: bounds are guaranteed by ROI clamping in the caller:
                // y0+y < proto_h, x0+x < proto_w, k+3 < num_protos <= protos.shape()[2].
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
                // SAFETY: bounds are guaranteed by ROI clamping in the caller:
                // y0+y < proto_h, x0+x < proto_w, k < num_protos <= protos.shape()[2].
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

#[cfg(test)]
mod scaled_tests {
    use super::*;
    use edgefirst_decoder::{BoundingBox, DetectBox, ProtoData, ProtoTensor};
    use ndarray::Array3;

    fn make_cpu() -> CPUProcessor {
        CPUProcessor::new()
    }

    /// A synthetic proto plane where channel 0 is a centred gaussian peaking
    /// at 10.0 and tailing off to ~0.0 at the edges; other channels zero.
    /// Mask coefficients (1.0, 0, 0, ...) select only channel 0.
    fn synthetic_proto_data(proto_h: usize, proto_w: usize, num_protos: usize) -> ProtoData {
        let mut protos = Array3::<f32>::zeros((proto_h, proto_w, num_protos));
        let cy = (proto_h as f32 - 1.0) / 2.0;
        let cx = (proto_w as f32 - 1.0) / 2.0;
        for y in 0..proto_h {
            for x in 0..proto_w {
                let dy = (y as f32 - cy) / cy;
                let dx = (x as f32 - cx) / cx;
                protos[[y, x, 0]] = 10.0 * (-(dx * dx + dy * dy)).exp();
            }
        }
        let mut coeffs = vec![0.0_f32; num_protos];
        coeffs[0] = 1.0;
        ProtoData {
            mask_coefficients: vec![coeffs],
            protos: ProtoTensor::Float(protos),
        }
    }

    #[test]
    fn scaled_central_bbox_produces_foreground_blob() {
        let cpu = make_cpu();
        let proto_data = synthetic_proto_data(16, 16, 4);

        // bbox covers the centre 50% of the plane.
        let detect = vec![DetectBox {
            bbox: BoundingBox::new(0.25, 0.25, 0.75, 0.75),
            score: 0.9,
            label: 0,
        }];

        let out = cpu
            .materialize_scaled_segmentations(&detect, &proto_data, None, 64, 64)
            .expect("scaled mask rendering must succeed");

        assert_eq!(out.len(), 1);
        let seg = &out[0];
        assert_eq!(seg.segmentation.shape(), &[32, 32, 1]);
        let values: Vec<u8> = seg.segmentation.iter().copied().collect();
        assert!(
            values.iter().all(|&v| v == 0 || v == 255),
            "scaled mask must be binary {{0, 255}}, found other values"
        );
        let fg_count = values.iter().filter(|&&v| v == 255).count();
        let fg_frac = fg_count as f32 / values.len() as f32;
        assert!(
            fg_frac > 0.80,
            "expected >80% foreground for centred gaussian, got {fg_frac:.2}"
        );
    }

    /// Realistic 160×160 → 640×640 parity vs hand-rolled bilinear + sigmoid
    /// reference. Tolerate <0.5% pixel mismatch due to fast_sigmoid's ~1.1%
    /// approximation error near the decision boundary.
    #[test]
    fn scaled_realistic_160x160_to_640x640_matches_reference() {
        let mut rng_seed: u32 = 0xD00D_F00D;
        let next = |s: &mut u32| -> f32 {
            *s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (*s as f32) / (u32::MAX as f32) * 2.0 - 1.0
        };

        let proto_h = 160;
        let proto_w = 160;
        let num_protos = 32;
        let mut protos = Array3::<f32>::zeros((proto_h, proto_w, num_protos));
        for y in 0..proto_h {
            for x in 0..proto_w {
                for k in 0..num_protos {
                    protos[[y, x, k]] = next(&mut rng_seed) * 3.0;
                }
            }
        }
        let mut coeffs = vec![0.0_f32; num_protos];
        for c in &mut coeffs {
            *c = next(&mut rng_seed);
        }
        let proto_data = ProtoData {
            mask_coefficients: vec![coeffs.clone()],
            protos: ProtoTensor::Float(protos.clone()),
        };

        let detect = vec![DetectBox {
            bbox: BoundingBox::new(0.1, 0.2, 0.6, 0.9),
            score: 0.9,
            label: 0,
        }];

        let cpu = make_cpu();
        let out = cpu
            .materialize_scaled_segmentations(&detect, &proto_data, None, 640, 640)
            .unwrap();
        assert_eq!(out.len(), 1);
        let hal_tile = &out[0].segmentation;

        let px0 = (0.1_f32 * 640.0_f32).round() as usize;
        let py0 = (0.2_f32 * 640.0_f32).round() as usize;
        let px1 = (0.6_f32 * 640.0_f32).round() as usize;
        let py1 = (0.9_f32 * 640.0_f32).round() as usize;
        let bbox_h = py1 - py0;
        let bbox_w = px1 - px0;
        assert_eq!(hal_tile.shape(), &[bbox_h, bbox_w, 1]);

        let sx = proto_w as f32 / 640.0_f32;
        let sy = proto_h as f32 / 640.0_f32;
        let mut mismatches = 0_usize;
        for yi in 0..bbox_h {
            let py = (py0 + yi) as f32;
            let sy_coord = (py + 0.5) * sy - 0.5;
            for xi in 0..bbox_w {
                let px = (px0 + xi) as f32;
                let sx_coord = (px + 0.5) * sx - 0.5;
                let x0 = sx_coord.floor().clamp(0.0, proto_w as f32 - 1.0) as usize;
                let y0 = sy_coord.floor().clamp(0.0, proto_h as f32 - 1.0) as usize;
                let x1 = (x0 + 1).min(proto_w - 1);
                let y1 = (y0 + 1).min(proto_h - 1);
                let fx = sx_coord - sx_coord.floor();
                let fy = sy_coord - sy_coord.floor();
                let w00 = (1.0 - fx) * (1.0 - fy);
                let w10 = fx * (1.0 - fy);
                let w01 = (1.0 - fx) * fy;
                let w11 = fx * fy;
                let mut acc = 0.0_f32;
                for p in 0..num_protos {
                    let val = w00 * protos[[y0, x0, p]]
                        + w10 * protos[[y0, x1, p]]
                        + w01 * protos[[y1, x0, p]]
                        + w11 * protos[[y1, x1, p]];
                    acc += coeffs[p] * val;
                }
                let sigmoid = 1.0_f32 / (1.0 + (-acc).exp());
                let expected: u8 = if sigmoid > 0.5 { 255 } else { 0 };
                if hal_tile[[yi, xi, 0]] != expected {
                    mismatches += 1;
                }
            }
        }
        let total = bbox_h * bbox_w;
        let mismatch_rate = mismatches as f32 / total as f32;
        assert!(
            mismatch_rate < 0.005,
            "mismatch rate {mismatch_rate:.4} > 0.5% tolerance \
             ({mismatches}/{total} pixels)"
        );
    }

    /// Letterbox path: the inverse letterbox transform must shift the
    /// bbox sample into the proto plane correctly when (width, height)
    /// are original-content pixel dims. A centred gaussian on the proto
    /// plane, aligned with the content region via letterbox, should
    /// still produce a centred foreground blob in the original-content
    /// coordinate frame.
    #[test]
    fn scaled_letterbox_original_content_coords() {
        let cpu = make_cpu();
        let proto_data = synthetic_proto_data(16, 16, 4);

        // Bbox in *model-input* normalized coords — the frame used by the
        // decoder output and consumed by `materialize_scaled_segmentations`.
        // Under the letterbox below, this maps to output-content normalized
        // (0.25, 0.3, 0.75, 0.7) and centres on the proto-plane centre.
        let detect = vec![DetectBox {
            bbox: BoundingBox::new(0.25, 0.3875, 0.75, 0.6125),
            score: 0.9,
            label: 0,
        }];

        // Letterbox: content fills full width, centred 56.25% vertically.
        // Caller requests Scaled(640, 360) in original-content space.
        let letterbox = Some([0.0_f32, 0.21875, 1.0, 0.78125]);
        let out = cpu
            .materialize_scaled_segmentations(&detect, &proto_data, letterbox, 640, 360)
            .expect("letterbox scaled rendering must succeed");

        assert_eq!(out.len(), 1);
        let seg = &out[0];
        let bbox_w = (0.5_f32 * 640.0_f32).round() as usize; // 320
        let bbox_h = (0.4_f32 * 360.0_f32).round() as usize; // 144
        assert_eq!(seg.segmentation.shape(), &[bbox_h, bbox_w, 1]);
        let uniq: std::collections::BTreeSet<u8> = seg.segmentation.iter().copied().collect();
        assert!(
            uniq.iter().all(|&v| v == 0 || v == 255),
            "letterbox scaled mask must be binary {{0, 255}}, got {uniq:?}"
        );

        // Middle-25%-square of the bbox should be solidly foreground
        // (the proto gaussian is centred; the bbox is centred on the
        // original-content frame; letterbox preserves centre alignment).
        let cy = bbox_h / 2;
        let cx = bbox_w / 2;
        let patch_h = bbox_h / 4;
        let patch_w = bbox_w / 4;
        let mut fg = 0_usize;
        let mut total = 0_usize;
        for y in cy.saturating_sub(patch_h / 2)..=(cy + patch_h / 2).min(bbox_h - 1) {
            for x in cx.saturating_sub(patch_w / 2)..=(cx + patch_w / 2).min(bbox_w - 1) {
                if seg.segmentation[[y, x, 0]] == 255 {
                    fg += 1;
                }
                total += 1;
            }
        }
        let fg_frac = fg as f32 / total as f32;
        assert!(
            fg_frac > 0.95,
            "centre of letterboxed bbox should be >95% foreground, got {fg_frac:.2}"
        );
    }

    /// Quantized i8 protos must produce the same binary mask as a float
    /// equivalent, modulo quantization rounding error.
    #[test]
    fn scaled_quantized_proto_produces_same_result_as_float() {
        use edgefirst_decoder::Quantization;

        let proto_h = 32;
        let proto_w = 32;
        let num_protos = 8;
        let mut protos_f32 = Array3::<f32>::zeros((proto_h, proto_w, num_protos));
        for y in 0..proto_h {
            for x in 0..proto_w {
                for k in 0..num_protos {
                    let v = ((y + x + k * 3) as f32 * 0.05).sin() * 3.0;
                    protos_f32[[y, x, k]] = v;
                }
            }
        }
        let scale = 0.1_f32;
        let zp = 0_i32;
        let protos_i8 = protos_f32.mapv(|v| (v / scale).round().clamp(-127.0, 127.0) as i8);
        let coeffs: Vec<f32> = (0..num_protos).map(|k| (k as f32 - 3.5) * 0.3).collect();

        let pd_float = ProtoData {
            mask_coefficients: vec![coeffs.clone()],
            protos: ProtoTensor::Float(protos_f32.clone()),
        };
        let pd_quant = ProtoData {
            mask_coefficients: vec![coeffs.clone()],
            protos: ProtoTensor::Quantized {
                protos: protos_i8,
                quantization: Quantization {
                    scale,
                    zero_point: zp,
                },
            },
        };

        let detect = vec![DetectBox {
            bbox: BoundingBox::new(0.1, 0.1, 0.9, 0.9),
            score: 0.9,
            label: 0,
        }];

        let cpu = make_cpu();
        let out_f = cpu
            .materialize_scaled_segmentations(&detect, &pd_float, None, 320, 320)
            .unwrap();
        let out_q = cpu
            .materialize_scaled_segmentations(&detect, &pd_quant, None, 320, 320)
            .unwrap();
        let mismatches = out_f[0]
            .segmentation
            .iter()
            .zip(out_q[0].segmentation.iter())
            .filter(|(a, b)| a != b)
            .count();
        let total = out_f[0].segmentation.len();
        assert!(
            mismatches < total / 200,
            "quantized and float diverged at {mismatches}/{total} pixels (>0.5%)"
        );
    }
}
