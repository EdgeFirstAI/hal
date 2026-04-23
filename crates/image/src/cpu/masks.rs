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
        use edgefirst_tensor::{DType, TensorMapTrait, TensorTrait};

        if detect.is_empty() {
            return Ok(Vec::new());
        }
        let proto_shape = proto_data.protos.shape();
        if proto_shape.len() != 3 {
            return Err(crate::Error::InvalidShape(format!(
                "protos tensor must be rank-3, got {proto_shape:?}"
            )));
        }
        let (proto_h, proto_w, num_protos) = (proto_shape[0], proto_shape[1], proto_shape[2]);
        let coeff_shape = proto_data.mask_coefficients.shape();
        if coeff_shape.len() != 2 || coeff_shape[1] != num_protos {
            return Err(crate::Error::InvalidShape(format!(
                "mask_coefficients shape {coeff_shape:?} incompatible with protos \
                 {proto_shape:?} (expected [N, {num_protos}])"
            )));
        }
        if coeff_shape[0] == 0 {
            return Ok(Vec::new());
        }
        if coeff_shape[0] != detect.len() {
            return Err(crate::Error::Internal(format!(
                "mask_coefficients rows {} != detection count {}",
                coeff_shape[0],
                detect.len()
            )));
        }

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

        // Coefficients may be F32 (from quantized or f32 models) or F16
        // (from fp16 models). For the mask kernel we always need an f32
        // view (the multiply-accumulate is done in f32 for precision). Map
        // once and widen once if f16, outside the per-detection loop.
        let coeff_f32_storage: Vec<f32>;
        let coeff_f32_slice: &[f32] = match proto_data.mask_coefficients.dtype() {
            DType::F32 => {
                let t = proto_data
                    .mask_coefficients
                    .as_f32()
                    .expect("dtype matched F32");
                let m = t.map()?;
                coeff_f32_storage = m.as_slice().to_vec();
                &coeff_f32_storage[..]
            }
            DType::F16 => {
                let t = proto_data
                    .mask_coefficients
                    .as_f16()
                    .expect("dtype matched F16");
                let m = t.map()?;
                coeff_f32_storage = m.as_slice().iter().map(|v| v.to_f32()).collect();
                &coeff_f32_storage[..]
            }
            other => {
                return Err(crate::Error::InvalidShape(format!(
                    "mask_coefficients dtype {other:?} not supported; expected F32 or F16"
                )));
            }
        };

        let per_detection = |i: usize,
                             det: &crate::DetectBox|
         -> crate::Result<edgefirst_decoder::Segmentation> {
            let coeff = &coeff_f32_slice[i * num_protos..(i + 1) * num_protos];

            let bbox = det.bbox.to_canonical();
            let xmin = bbox.xmin.clamp(0.0, 1.0);
            let ymin = bbox.ymin.clamp(0.0, 1.0);
            let xmax = bbox.xmax.clamp(0.0, 1.0);
            let ymax = bbox.ymax.clamp(0.0, 1.0);
            let x0 = ((xmin * proto_w as f32) as usize).min(proto_w.saturating_sub(1));
            let y0 = ((ymin * proto_h as f32) as usize).min(proto_h.saturating_sub(1));
            let x1 = ((xmax * proto_w as f32).ceil() as usize).min(proto_w);
            let y1 = ((ymax * proto_h as f32).ceil() as usize).min(proto_h);
            let roi_w = x1.saturating_sub(x0).max(1);
            let roi_h = y1.saturating_sub(y0).max(1);

            let mask = match proto_data.protos.dtype() {
                DType::I8 => {
                    let t = proto_data.protos.as_i8().expect("dtype matched I8");
                    let quant = t.quantization().ok_or_else(|| {
                        crate::Error::InvalidShape("I8 protos require quantization metadata".into())
                    })?;
                    let m = t.map()?;
                    fused_dequant_dot_sigmoid_i8_slice(
                        m.as_slice(),
                        coeff,
                        quant,
                        proto_h,
                        proto_w,
                        y0,
                        x0,
                        roi_h,
                        roi_w,
                        num_protos,
                    )?
                }
                DType::F32 => {
                    let t = proto_data.protos.as_f32().expect("dtype matched F32");
                    let m = t.map()?;
                    fused_dot_sigmoid_f32_slice(
                        m.as_slice(),
                        coeff,
                        proto_h,
                        proto_w,
                        y0,
                        x0,
                        roi_h,
                        roi_w,
                        num_protos,
                    )
                }
                DType::F16 => {
                    let t = proto_data.protos.as_f16().expect("dtype matched F16");
                    let m = t.map()?;
                    fused_dot_sigmoid_f16_slice(
                        m.as_slice(),
                        coeff,
                        proto_h,
                        proto_w,
                        y0,
                        x0,
                        roi_h,
                        roi_w,
                        num_protos,
                    )
                }
                other => {
                    return Err(crate::Error::InvalidShape(format!(
                        "proto tensor dtype {other:?} not supported"
                    )));
                }
            };

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
        };

        detect
            .iter()
            .enumerate()
            .map(|(i, det)| per_detection(i, det))
            .collect()
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
        use edgefirst_tensor::{DType, TensorMapTrait, TensorTrait};

        if detect.is_empty() {
            return Ok(Vec::new());
        }
        if width == 0 || height == 0 {
            return Err(crate::Error::InvalidShape(
                "Scaled mask width/height must be positive".into(),
            ));
        }
        let proto_shape = proto_data.protos.shape();
        if proto_shape.len() != 3 {
            return Err(crate::Error::InvalidShape(format!(
                "protos tensor must be rank-3, got {proto_shape:?}"
            )));
        }
        let (proto_h, proto_w, num_protos) = (proto_shape[0], proto_shape[1], proto_shape[2]);
        let coeff_shape = proto_data.mask_coefficients.shape();
        if coeff_shape.len() != 2 || coeff_shape[1] != num_protos {
            return Err(crate::Error::InvalidShape(format!(
                "mask_coefficients shape {coeff_shape:?} incompatible with protos \
                 {proto_shape:?}"
            )));
        }
        if coeff_shape[0] == 0 {
            return Ok(Vec::new());
        }

        // Widen coefficients to f32 once for the scaled-sample inner loop.
        let coeff_f32: Vec<f32> = match proto_data.mask_coefficients.dtype() {
            DType::F32 => {
                let t = proto_data.mask_coefficients.as_f32().expect("F32");
                let m = t.map()?;
                m.as_slice().to_vec()
            }
            DType::F16 => {
                let t = proto_data.mask_coefficients.as_f16().expect("F16");
                let m = t.map()?;
                m.as_slice().iter().map(|v| v.to_f32()).collect()
            }
            other => {
                return Err(crate::Error::InvalidShape(format!(
                    "mask_coefficients dtype {other:?} not supported"
                )));
            }
        };

        match proto_data.protos.dtype() {
            DType::F32 => {
                let t = proto_data.protos.as_f32().expect("F32");
                let m = t.map()?;
                scaled_segmentations_f32_slice(
                    detect,
                    &coeff_f32,
                    m.as_slice(),
                    proto_h,
                    proto_w,
                    num_protos,
                    letterbox,
                    width,
                    height,
                )
            }
            DType::F16 => {
                let t = proto_data.protos.as_f16().expect("F16");
                let m = t.map()?;
                scaled_segmentations_f16_slice(
                    detect,
                    &coeff_f32,
                    m.as_slice(),
                    proto_h,
                    proto_w,
                    num_protos,
                    letterbox,
                    width,
                    height,
                )
            }
            DType::I8 => {
                let t = proto_data.protos.as_i8().expect("I8");
                let m = t.map()?;
                let quant = t.quantization().ok_or_else(|| {
                    crate::Error::InvalidShape("I8 protos require quantization metadata".into())
                })?;
                scaled_segmentations_i8_slice(
                    detect,
                    &coeff_f32,
                    m.as_slice(),
                    proto_h,
                    proto_w,
                    num_protos,
                    quant,
                    letterbox,
                    width,
                    height,
                )
            }
            other => Err(crate::Error::InvalidShape(format!(
                "proto tensor dtype {other:?} not supported"
            ))),
        }
    }
}

// =============================================================================
// Slice-native fused kernels.
//
// All kernels take row-major `[H, W, num_protos]` proto slices + `&[f32]`
// coefficients (widened once from the source dtype at the materialize entry
// point). Per-dtype variants exist for i8 (with on-the-fly dequant using a
// tensor-level `Quantization`), f32, and f16; f16 widens to f32 per-element
// at the FMA site via `half::f16::to_f32()`.
//
// On ARMv8.2-FP16 this compiles to `fcvt`; on Cortex-A53 and non-F16C x86 it
// becomes a soft-float helper. Stage 8 adds explicit intrinsic kernels
// gated by `#[cfg(target_feature = "fp16")]` / `+f16c`.
// =============================================================================

#[allow(clippy::too_many_arguments)]
fn fused_dequant_dot_sigmoid_i8_slice(
    protos: &[i8],
    coeff: &[f32],
    quant: &edgefirst_tensor::Quantization,
    _proto_h: usize,
    proto_w: usize,
    y0: usize,
    x0: usize,
    roi_h: usize,
    roi_w: usize,
    num_protos: usize,
) -> crate::Result<ndarray::Array3<u8>> {
    use edgefirst_tensor::QuantMode;
    let stride_y = proto_w * num_protos;
    // Precompute scaled coefficients + zp_offset on the stack. Supports
    // per-tensor and per-channel; per-channel with axis != channel dim is
    // rejected. num_protos ≤ 64 covers every real model.
    if num_protos > 64 {
        return Err(crate::Error::InvalidShape(format!(
            "num_protos {num_protos} exceeds kernel limit 64"
        )));
    }
    let mut scaled_coeff = [0.0_f32; 64];
    let zp_offset: f32;
    match quant.mode() {
        QuantMode::PerTensorSymmetric { scale } => {
            for k in 0..num_protos {
                scaled_coeff[k] = coeff[k] * scale;
            }
            zp_offset = 0.0;
        }
        QuantMode::PerTensor { scale, zero_point } => {
            for k in 0..num_protos {
                scaled_coeff[k] = coeff[k] * scale;
            }
            zp_offset = zero_point as f32 * scaled_coeff.iter().take(num_protos).sum::<f32>();
        }
        QuantMode::PerChannelSymmetric { scales, axis } => {
            if axis != 2 {
                return Err(crate::Error::InvalidShape(format!(
                    "per-channel quantization on axis {axis} not supported (expected 2)"
                )));
            }
            for k in 0..num_protos {
                scaled_coeff[k] = coeff[k] * scales[k];
            }
            zp_offset = 0.0;
        }
        QuantMode::PerChannel {
            scales,
            zero_points,
            axis,
        } => {
            if axis != 2 {
                return Err(crate::Error::InvalidShape(format!(
                    "per-channel quantization on axis {axis} not supported (expected 2)"
                )));
            }
            for k in 0..num_protos {
                scaled_coeff[k] = coeff[k] * scales[k];
            }
            zp_offset = (0..num_protos)
                .map(|k| scaled_coeff[k] * zero_points[k] as f32)
                .sum();
        }
    }

    let mut mask = ndarray::Array3::<u8>::zeros((roi_h, roi_w, 1));
    for y in 0..roi_h {
        for x in 0..roi_w {
            let base = (y0 + y) * stride_y + (x0 + x) * num_protos;
            let mut acc = 0.0_f32;
            let mut k = 0;
            let chunks = num_protos / 4;
            for _ in 0..chunks {
                let p0 = protos[base + k] as f32;
                let p1 = protos[base + k + 1] as f32;
                let p2 = protos[base + k + 2] as f32;
                let p3 = protos[base + k + 3] as f32;
                acc += scaled_coeff[k] * p0
                    + scaled_coeff[k + 1] * p1
                    + scaled_coeff[k + 2] * p2
                    + scaled_coeff[k + 3] * p3;
                k += 4;
            }
            while k < num_protos {
                acc += scaled_coeff[k] * protos[base + k] as f32;
                k += 1;
            }
            acc -= zp_offset;
            let sigmoid = fast_sigmoid(acc);
            mask[[y, x, 0]] = (sigmoid * 255.0 + 0.5) as u8;
        }
    }
    Ok(mask)
}

#[allow(clippy::too_many_arguments)]
fn fused_dot_sigmoid_f32_slice(
    protos: &[f32],
    coeff: &[f32],
    _proto_h: usize,
    proto_w: usize,
    y0: usize,
    x0: usize,
    roi_h: usize,
    roi_w: usize,
    num_protos: usize,
) -> ndarray::Array3<u8> {
    let stride_y = proto_w * num_protos;
    let mut mask = ndarray::Array3::<u8>::zeros((roi_h, roi_w, 1));
    for y in 0..roi_h {
        for x in 0..roi_w {
            let base = (y0 + y) * stride_y + (x0 + x) * num_protos;
            let mut acc = 0.0_f32;
            let mut k = 0;
            let chunks = num_protos / 4;
            for _ in 0..chunks {
                acc += coeff[k] * protos[base + k]
                    + coeff[k + 1] * protos[base + k + 1]
                    + coeff[k + 2] * protos[base + k + 2]
                    + coeff[k + 3] * protos[base + k + 3];
                k += 4;
            }
            while k < num_protos {
                acc += coeff[k] * protos[base + k];
                k += 1;
            }
            let sigmoid = fast_sigmoid(acc);
            mask[[y, x, 0]] = (sigmoid * 255.0 + 0.5) as u8;
        }
    }
    mask
}

/// Native-f16 fused kernel.
///
/// Three code paths, selected at compile time:
///
/// 1. **x86_64 + F16C + FMA** — explicit intrinsic kernel (`_mm256_cvtph_ps`
///    8-lane f16→f32 widening, `_mm256_fmadd_ps` FMA). Guaranteed to use
///    hardware f16 conversion, not LLVM's autovectorizer (which is unreliable
///    for this pattern per rust-lang/stdarch #1349).
///
/// 2. **aarch64 + FP16** — scalar `half::f16::to_f32()` at the FMA site.
///    LLVM lowers each `.to_f32()` to a single `fcvt` instruction when
///    `target-feature=+fp16` is active (e.g. `target-cpu=cortex-a78ae`).
///    The stable f16-typed NEON intrinsics (`vcvt_f32_f16`, `vld1q_f16`)
///    require nightly as of this commit; the scalar path is equally
///    efficient at this granularity.
///
/// 3. **Fallback (Cortex-A53, targets without FP16)** — same scalar code.
///    `half::f16::to_f32()` lowers to `__extendhfsf2` soft-float helper,
///    one call per proto load. Correctness-preserving; ~15 cycles/load
///    vs. ~3 cycles for the hardware path. Documented in
///    `docs/orin-build.md`.
#[allow(clippy::too_many_arguments)]
fn fused_dot_sigmoid_f16_slice(
    protos: &[half::f16],
    coeff: &[f32],
    proto_h: usize,
    proto_w: usize,
    y0: usize,
    x0: usize,
    roi_h: usize,
    roi_w: usize,
    num_protos: usize,
) -> ndarray::Array3<u8> {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "f16c",
        target_feature = "fma"
    ))]
    {
        // SAFETY: target-feature gates both `vcvtph2ps` and `vfmadd*ps`;
        // the caller's slice-bounds contract is identical to the scalar arm.
        unsafe {
            fused_dot_sigmoid_f16_slice_f16c(
                protos, coeff, proto_h, proto_w, y0, x0, roi_h, roi_w, num_protos,
            )
        }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "f16c",
        target_feature = "fma"
    )))]
    {
        let _ = proto_h;
        fused_dot_sigmoid_f16_slice_scalar(protos, coeff, proto_w, y0, x0, roi_h, roi_w, num_protos)
    }
}

/// Scalar native-f16 kernel. `half::f16::to_f32()` at the FMA site is
/// lowered to a single `fcvt` (aarch64+fp16) or a single `vcvtps_ps`
/// (x86_64+f16c) by LLVM, or to the soft-float helper `__extendhfsf2` on
/// targets without FP16 hardware. Loop unrolled by 4 to give the scheduler
/// room to overlap loads with FMAs.
#[allow(clippy::too_many_arguments, dead_code)]
fn fused_dot_sigmoid_f16_slice_scalar(
    protos: &[half::f16],
    coeff: &[f32],
    proto_w: usize,
    y0: usize,
    x0: usize,
    roi_h: usize,
    roi_w: usize,
    num_protos: usize,
) -> ndarray::Array3<u8> {
    let stride_y = proto_w * num_protos;
    let mut mask = ndarray::Array3::<u8>::zeros((roi_h, roi_w, 1));
    for y in 0..roi_h {
        for x in 0..roi_w {
            let base = (y0 + y) * stride_y + (x0 + x) * num_protos;
            let mut acc = 0.0_f32;
            let mut k = 0;
            let chunks = num_protos / 4;
            for _ in 0..chunks {
                let p0 = protos[base + k].to_f32();
                let p1 = protos[base + k + 1].to_f32();
                let p2 = protos[base + k + 2].to_f32();
                let p3 = protos[base + k + 3].to_f32();
                acc += coeff[k] * p0 + coeff[k + 1] * p1 + coeff[k + 2] * p2 + coeff[k + 3] * p3;
                k += 4;
            }
            while k < num_protos {
                acc += coeff[k] * protos[base + k].to_f32();
                k += 1;
            }
            let sigmoid = fast_sigmoid(acc);
            mask[[y, x, 0]] = (sigmoid * 255.0 + 0.5) as u8;
        }
    }
    mask
}

/// x86_64 F16C + FMA explicit intrinsic kernel. Processes 8 f16 lanes per
/// inner iteration via `_mm256_cvtph_ps` (8-lane f16→f32 widen) followed by
/// `_mm256_fmadd_ps` (8-lane fused multiply-add). Horizontal reduce at the
/// end of each pixel.
///
/// # Safety
///
/// Caller must ensure the target CPU supports F16C + FMA. The workspace's
/// `.cargo/config.toml` sets these target-features on `x86_64-unknown-linux-gnu`
/// and `x86_64-apple-darwin`, making this function statically callable on
/// those targets.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "f16c",
    target_feature = "fma"
))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "f16c,fma,avx")]
unsafe fn fused_dot_sigmoid_f16_slice_f16c(
    protos: &[half::f16],
    coeff: &[f32],
    _proto_h: usize,
    proto_w: usize,
    y0: usize,
    x0: usize,
    roi_h: usize,
    roi_w: usize,
    num_protos: usize,
) -> ndarray::Array3<u8> {
    use core::arch::x86_64::{
        _mm256_castps256_ps128, _mm256_cvtph_ps, _mm256_extractf128_ps, _mm256_fmadd_ps,
        _mm256_loadu_ps, _mm256_setzero_ps, _mm_add_ps, _mm_cvtss_f32, _mm_hadd_ps,
        _mm_loadu_si128,
    };

    let stride_y = proto_w * num_protos;
    let chunks8 = num_protos / 8;
    let tail = num_protos % 8;
    let mut mask = ndarray::Array3::<u8>::zeros((roi_h, roi_w, 1));

    for y in 0..roi_h {
        for x in 0..roi_w {
            let base = (y0 + y) * stride_y + (x0 + x) * num_protos;
            let mut acc_v = _mm256_setzero_ps();
            let mut k = 0;
            for _ in 0..chunks8 {
                // Load 8 f16 (128 bits / 16 bytes) via a byte-level cast.
                let p_ptr = protos
                    .as_ptr()
                    .add(base + k)
                    .cast::<core::arch::x86_64::__m128i>();
                let raw = _mm_loadu_si128(p_ptr);
                let widened = _mm256_cvtph_ps(raw);
                let coeffs_v = _mm256_loadu_ps(coeff.as_ptr().add(k));
                acc_v = _mm256_fmadd_ps(coeffs_v, widened, acc_v);
                k += 8;
            }
            // Horizontal reduce 8 → 1.
            let lo = _mm256_castps256_ps128(acc_v);
            let hi = _mm256_extractf128_ps::<1>(acc_v);
            let sum4 = _mm_add_ps(lo, hi);
            let sum2 = _mm_hadd_ps(sum4, sum4);
            let sum1 = _mm_hadd_ps(sum2, sum2);
            let mut acc = _mm_cvtss_f32(sum1);

            // Scalar tail for num_protos % 8 (≤ 7 items).
            while k < num_protos && k - chunks8 * 8 < tail {
                acc += coeff[k] * protos[base + k].to_f32();
                k += 1;
            }

            let sigmoid = fast_sigmoid(acc);
            mask[[y, x, 0]] = (sigmoid * 255.0 + 0.5) as u8;
        }
    }
    mask
}

// Scaled mask kernels — bilinear sample + sigmoid, binarized output.

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn bilinear_dot_slice<P: Copy>(
    protos: &[P],
    stride_y: usize,
    num_protos: usize,
    coeff: &[f32],
    px: f32,
    py: f32,
    proto_w: usize,
    proto_h: usize,
    load_f32: impl Fn(&P) -> f32,
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
    let b00 = y0 * stride_y + x0 * num_protos;
    let b10 = y0 * stride_y + x1 * num_protos;
    let b01 = y1 * stride_y + x0 * num_protos;
    let b11 = y1 * stride_y + x1 * num_protos;
    let mut acc = 0.0_f32;
    for p in 0..num_protos {
        let v00 = load_f32(&protos[b00 + p]);
        let v10 = load_f32(&protos[b10 + p]);
        let v01 = load_f32(&protos[b01 + p]);
        let v11 = load_f32(&protos[b11 + p]);
        let val = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
        acc += coeff[p] * val;
    }
    acc
}

#[allow(clippy::too_many_arguments)]
fn scaled_segmentations_f32_slice(
    detect: &[crate::DetectBox],
    coeff_all: &[f32],
    protos: &[f32],
    proto_h: usize,
    proto_w: usize,
    num_protos: usize,
    letterbox: Option<[f32; 4]>,
    width: u32,
    height: u32,
) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
    scaled_run(
        detect,
        coeff_all,
        protos,
        proto_h,
        proto_w,
        num_protos,
        letterbox,
        width,
        height,
        1.0,
        |p, _| *p,
    )
}

#[allow(clippy::too_many_arguments)]
fn scaled_segmentations_f16_slice(
    detect: &[crate::DetectBox],
    coeff_all: &[f32],
    protos: &[half::f16],
    proto_h: usize,
    proto_w: usize,
    num_protos: usize,
    letterbox: Option<[f32; 4]>,
    width: u32,
    height: u32,
) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
    scaled_run(
        detect,
        coeff_all,
        protos,
        proto_h,
        proto_w,
        num_protos,
        letterbox,
        width,
        height,
        1.0,
        |p: &half::f16, _| p.to_f32(),
    )
}

#[allow(clippy::too_many_arguments)]
fn scaled_segmentations_i8_slice(
    detect: &[crate::DetectBox],
    coeff_all: &[f32],
    protos: &[i8],
    proto_h: usize,
    proto_w: usize,
    num_protos: usize,
    quant: &edgefirst_tensor::Quantization,
    letterbox: Option<[f32; 4]>,
    width: u32,
    height: u32,
) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
    use edgefirst_tensor::QuantMode;
    // Only per-tensor quantization supported on the scaled-path CPU kernel
    // today. Per-channel fits naturally into a future extension (would need
    // to pass per-channel scaled coefficients into bilinear_dot_slice).
    let (scale, zp) = match quant.mode() {
        QuantMode::PerTensor { scale, zero_point } => (scale, zero_point as f32),
        QuantMode::PerTensorSymmetric { scale } => (scale, 0.0),
        QuantMode::PerChannel { axis, .. } | QuantMode::PerChannelSymmetric { axis, .. } => {
            return Err(crate::Error::InvalidShape(format!(
                "per-channel quantization (axis={axis}) on scaled seg path \
                 not yet supported"
            )));
        }
    };
    scaled_run(
        detect,
        coeff_all,
        protos,
        proto_h,
        proto_w,
        num_protos,
        letterbox,
        width,
        height,
        scale,
        move |p: &i8, _| *p as f32 - zp,
    )
}

#[allow(clippy::too_many_arguments)]
fn scaled_run<P: Copy>(
    detect: &[crate::DetectBox],
    coeff_all: &[f32],
    protos: &[P],
    proto_h: usize,
    proto_w: usize,
    num_protos: usize,
    letterbox: Option<[f32; 4]>,
    width: u32,
    height: u32,
    acc_scale: f32,
    load_f32: impl Fn(&P, f32) -> f32 + Copy,
) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
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
    let stride_y = proto_w * num_protos;

    detect
        .iter()
        .enumerate()
        .map(|(i, det)| {
            let coeff = &coeff_all[i * num_protos..(i + 1) * num_protos];
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
                    let acc = bilinear_dot_slice(
                        protos,
                        stride_y,
                        num_protos,
                        coeff,
                        sample_x,
                        sample_y,
                        proto_w,
                        proto_h,
                        |p: &P| load_f32(p, 0.0),
                    );
                    let sigmoid = fast_sigmoid(acc_scale * acc);
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
        .collect()
}

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
