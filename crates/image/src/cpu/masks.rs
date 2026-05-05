// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::CPUProcessor;
use crate::Result;
use edgefirst_decoder::{DetectBox, Segmentation};
use ndarray::Axis;
use rayon::prelude::*;

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

        let _span = tracing::trace_span!(
            "materialize_masks",
            mode = "proto",
            n_detections = detect.len(),
        )
        .entered();

        if detect.is_empty() {
            return Ok(Vec::new());
        }
        let proto_shape = proto_data.protos.shape();
        if proto_shape.len() != 3 {
            return Err(crate::Error::InvalidShape(format!(
                "protos tensor must be rank-3, got {proto_shape:?}"
            )));
        }
        // Interpret shape based on physical layout.
        let (proto_h, proto_w, num_protos) = match proto_data.layout {
            edgefirst_decoder::ProtoLayout::Nhwc => {
                (proto_shape[0], proto_shape[1], proto_shape[2])
            }
            edgefirst_decoder::ProtoLayout::Nchw => {
                (proto_shape[1], proto_shape[2], proto_shape[0])
            }
        };
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

        // Fast integer path: when both coefficients and protos are I8 with
        // per-tensor quantization, use the all-integer kernel (same dot
        // product infrastructure as the scaled path, but at proto resolution
        // without bilinear upsampling). Output is binary {0, 255}.
        // Falls through to the general f32 dequant path for per-channel
        // quantization or other unsupported modes.
        if proto_data.mask_coefficients.dtype() == DType::I8
            && proto_data.protos.dtype() == DType::I8
        {
            let coeff_t = proto_data
                .mask_coefficients
                .as_i8()
                .expect("I8 coefficients");
            let coeff_m = coeff_t.map()?;
            let coeff_quant = coeff_t.quantization().ok_or_else(|| {
                crate::Error::InvalidShape(
                    "I8 mask_coefficients require quantization metadata".into(),
                )
            })?;
            let proto_t = proto_data.protos.as_i8().expect("I8 protos");
            let proto_m = proto_t.map()?;
            let proto_quant = proto_t.quantization().ok_or_else(|| {
                crate::Error::InvalidShape("I8 protos require quantization metadata".into())
            })?;
            match proto_segmentations_i8_i8(
                detect,
                coeff_m.as_slice(),
                coeff_quant,
                proto_m.as_slice(),
                proto_quant,
                proto_h,
                proto_w,
                num_protos,
                lx0,
                inv_lw,
                ly0,
                inv_lh,
                proto_data.layout,
            ) {
                Ok(result) => return Ok(result),
                Err(crate::Error::NotSupported(_)) => {
                    // Fall through to the general f32 dequant path below for
                    // per-channel quantization and other unsupported modes.
                }
                Err(e) => return Err(e),
            }
        }

        // Coefficients may be F32 (from f32 models), F16 (from fp16 models),
        // or I8 (from quantized models — kept raw with quantization). For the
        // mask kernel we always need an f32 view (the multiply-accumulate is
        // done in f32 for precision). Map once and widen once outside the loop.
        // NCHW layout is only supported in the i8×i8 integer fast path above
        // with per-tensor quantization. Reject here for all other combinations.
        if proto_data.layout == edgefirst_decoder::ProtoLayout::Nchw {
            return Err(crate::Error::NotSupported(
                "NCHW proto layout requires I8 protos and coefficients with per-tensor quantization"
                    .into(),
            ));
        }
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
            DType::I8 => {
                let t = proto_data
                    .mask_coefficients
                    .as_i8()
                    .expect("dtype matched I8");
                let m = t.map()?;
                coeff_f32_storage = if let Some(q) = t.quantization() {
                    use edgefirst_tensor::QuantMode;
                    let (scale, zp) = match q.mode() {
                        QuantMode::PerTensor { scale, zero_point } => (scale, zero_point as f32),
                        QuantMode::PerTensorSymmetric { scale } => (scale, 0.0),
                        other => {
                            return Err(crate::Error::NotSupported(format!(
                                "I8 mask_coefficients quantization mode {other:?} not supported"
                            )));
                        }
                    };
                    m.as_slice()
                        .iter()
                        .map(|&v| (v as f32 - zp) * scale)
                        .collect()
                } else {
                    m.as_slice().iter().map(|&v| v as f32).collect()
                };
                &coeff_f32_storage[..]
            }
            other => {
                return Err(crate::Error::InvalidShape(format!(
                    "mask_coefficients dtype {other:?} not supported; expected F32, F16, or I8"
                )));
            }
        };

        // Hoist the proto tensor map() out of the per-detection loop so the
        // map-guard is acquired once. Then dispatch per-dtype via a helper
        // that runs the per-detection kernels in parallel across detections
        // via rayon. This restores the parallelism that PR #54 added and
        // PR #51 (EDGEAI-1244 f16 refactor) inadvertently removed.
        match proto_data.protos.dtype() {
            DType::I8 => {
                let t = proto_data.protos.as_i8().expect("dtype matched I8");
                let quant = t.quantization().ok_or_else(|| {
                    crate::Error::InvalidShape("I8 protos require quantization metadata".into())
                })?;
                let m = t.map()?;
                let protos_slice = m.as_slice();
                detect
                    .par_iter()
                    .enumerate()
                    .map(|(i, det)| {
                        let coeff = &coeff_f32_slice[i * num_protos..(i + 1) * num_protos];
                        let (x0, y0, x1, y1, roi_w, roi_h) =
                            bbox_to_proto_roi(det, proto_w, proto_h);
                        let mask = fused_dequant_dot_sign_i8_slice(
                            protos_slice,
                            coeff,
                            quant,
                            proto_h,
                            proto_w,
                            y0,
                            x0,
                            roi_h,
                            roi_w,
                            num_protos,
                        )?;
                        Ok(seg_from_roi(
                            mask, x0, y0, x1, y1, proto_w, proto_h, lx0, inv_lw, ly0, inv_lh,
                        ))
                    })
                    .collect()
            }
            DType::F32 => {
                let t = proto_data.protos.as_f32().expect("dtype matched F32");
                let m = t.map()?;
                let protos_slice = m.as_slice();
                detect
                    .par_iter()
                    .enumerate()
                    .map(|(i, det)| {
                        let coeff = &coeff_f32_slice[i * num_protos..(i + 1) * num_protos];
                        let (x0, y0, x1, y1, roi_w, roi_h) =
                            bbox_to_proto_roi(det, proto_w, proto_h);
                        let mask = fused_dot_sign_f32_slice(
                            protos_slice,
                            coeff,
                            proto_h,
                            proto_w,
                            y0,
                            x0,
                            roi_h,
                            roi_w,
                            num_protos,
                        );
                        Ok(seg_from_roi(
                            mask, x0, y0, x1, y1, proto_w, proto_h, lx0, inv_lw, ly0, inv_lh,
                        ))
                    })
                    .collect()
            }
            DType::F16 => {
                let t = proto_data.protos.as_f16().expect("dtype matched F16");
                let m = t.map()?;
                let protos_slice = m.as_slice();
                detect
                    .par_iter()
                    .enumerate()
                    .map(|(i, det)| {
                        let coeff = &coeff_f32_slice[i * num_protos..(i + 1) * num_protos];
                        let (x0, y0, x1, y1, roi_w, roi_h) =
                            bbox_to_proto_roi(det, proto_w, proto_h);
                        let mask = fused_dot_sign_f16_slice(
                            protos_slice,
                            coeff,
                            proto_h,
                            proto_w,
                            y0,
                            x0,
                            roi_h,
                            roi_w,
                            num_protos,
                        );
                        Ok(seg_from_roi(
                            mask, x0, y0, x1, y1, proto_w, proto_h, lx0, inv_lw, ly0, inv_lh,
                        ))
                    })
                    .collect()
            }
            other => Err(crate::Error::InvalidShape(format!(
                "proto tensor dtype {other:?} not supported"
            ))),
        }
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

        let _span = tracing::trace_span!(
            "materialize_masks",
            mode = "scaled",
            n_detections = detect.len(),
            width,
            height,
        )
        .entered();

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
        // Interpret shape based on physical layout.
        let (proto_h, proto_w, num_protos) = match proto_data.layout {
            edgefirst_decoder::ProtoLayout::Nhwc => {
                (proto_shape[0], proto_shape[1], proto_shape[2])
            }
            edgefirst_decoder::ProtoLayout::Nchw => {
                (proto_shape[1], proto_shape[2], proto_shape[0])
            }
        };
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
        if coeff_shape[0] != detect.len() {
            return Err(crate::Error::Internal(format!(
                "mask_coefficients rows {} != detection count {}",
                coeff_shape[0],
                detect.len()
            )));
        }

        // Fast integer path: when both coefficients and protos are I8, use
        // the all-integer kernel (i8×i8→i32 dot product, sign-shortcut
        // bilinear). No floating-point conversion at all.
        // Falls through to the general f32 dequant path for per-channel
        // quantization or other unsupported modes.
        if proto_data.mask_coefficients.dtype() == DType::I8
            && proto_data.protos.dtype() == DType::I8
        {
            let coeff_t = proto_data
                .mask_coefficients
                .as_i8()
                .expect("I8 coefficients");
            let coeff_m = coeff_t.map()?;
            let coeff_quant = coeff_t.quantization().ok_or_else(|| {
                crate::Error::InvalidShape(
                    "I8 mask_coefficients require quantization metadata".into(),
                )
            })?;
            let proto_t = proto_data.protos.as_i8().expect("I8 protos");
            let proto_m = proto_t.map()?;
            let proto_quant = proto_t.quantization().ok_or_else(|| {
                crate::Error::InvalidShape("I8 protos require quantization metadata".into())
            })?;
            match scaled_segmentations_i8_i8(
                detect,
                coeff_m.as_slice(),
                coeff_quant,
                proto_m.as_slice(),
                proto_quant,
                proto_h,
                proto_w,
                num_protos,
                letterbox,
                width,
                height,
                proto_data.layout,
            ) {
                Ok(result) => return Ok(result),
                Err(crate::Error::NotSupported(_)) => {
                    // Fall through to the general f32 dequant path below for
                    // per-channel quantization and other unsupported modes.
                }
                Err(e) => return Err(e),
            }
        }

        // Fallback: widen coefficients to f32 for the float-path kernels.
        // NCHW layout is only supported in the i8×i8 integer fast path above
        // with per-tensor quantization. Reject here for all other combinations.
        if proto_data.layout == edgefirst_decoder::ProtoLayout::Nchw {
            return Err(crate::Error::NotSupported(
                "NCHW proto layout requires I8 protos and coefficients with per-tensor quantization"
                    .into(),
            ));
        }
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
            DType::I8 => {
                // Dequantize I8 coefficients to f32 for the float proto path.
                let t = proto_data.mask_coefficients.as_i8().expect("I8");
                let m = t.map()?;
                let q = t.quantization().ok_or_else(|| {
                    crate::Error::InvalidShape(
                        "I8 mask_coefficients require quantization metadata".into(),
                    )
                })?;
                use edgefirst_tensor::QuantMode;
                let (scale, zp) = match q.mode() {
                    QuantMode::PerTensor { scale, zero_point } => (scale, zero_point as f32),
                    QuantMode::PerTensorSymmetric { scale } => (scale, 0.0),
                    _ => {
                        return Err(crate::Error::NotSupported(
                            "per-channel mask_coefficients not supported".into(),
                        ))
                    }
                };
                m.as_slice()
                    .iter()
                    .map(|&v| (v as f32 - zp) * scale)
                    .collect()
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

/// Map a detection bbox in normalised letterboxed coords to its ROI in
/// the proto plane (floor xmin/ymin, ceil xmax/ymax, clamp to plane bounds).
/// Returns `(x0, y0, x1, y1, roi_w, roi_h)` where roi_w/h are guaranteed ≥ 1.
fn bbox_to_proto_roi(
    det: &DetectBox,
    proto_w: usize,
    proto_h: usize,
) -> (usize, usize, usize, usize, usize, usize) {
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
    (x0, y0, x1, y1, roi_w, roi_h)
}

/// Build a `Segmentation` from a per-detection mask + the ROI bounds in
/// proto coords. Applies the inverse letterbox transform to express the
/// segmentation bbox in original-image-content normalised space.
#[allow(clippy::too_many_arguments)]
fn seg_from_roi(
    mask: ndarray::Array3<u8>,
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    proto_w: usize,
    proto_h: usize,
    lx0: f32,
    inv_lw: f32,
    ly0: f32,
    inv_lh: f32,
) -> edgefirst_decoder::Segmentation {
    let seg_xmin = ((x0 as f32 / proto_w as f32) - lx0) * inv_lw;
    let seg_ymin = ((y0 as f32 / proto_h as f32) - ly0) * inv_lh;
    let seg_xmax = ((x1 as f32 / proto_w as f32) - lx0) * inv_lw;
    let seg_ymax = ((y1 as f32 / proto_h as f32) - ly0) * inv_lh;
    edgefirst_decoder::Segmentation {
        xmin: seg_xmin.clamp(0.0, 1.0),
        ymin: seg_ymin.clamp(0.0, 1.0),
        xmax: seg_xmax.clamp(0.0, 1.0),
        ymax: seg_ymax.clamp(0.0, 1.0),
        segmentation: mask,
    }
}

// =============================================================================
// Integer-domain proto-resolution kernel: i8 coefficients × i8 protos → i32
// → sign threshold → binary {0, 255}.
//
// Reuses the same dot product infrastructure as the scaled path (NEON sdot on
// A55+, smull+sadalp on A53, scalar fallback on x86). Since proto-resolution
// produces masks at the native proto grid (~30×30 per ROI), there is no
// bilinear upsampling — just a direct sign threshold per pixel.
// =============================================================================

/// Proto-resolution mask materialization using integer-domain math.
///
/// For each detection, computes the i8×i8 dot product at every proto-ROI pixel,
/// applies the zero-point correction, and thresholds at sign(logit) → {0, 255}.
/// Supports both NHWC and NCHW proto layouts.
#[allow(clippy::too_many_arguments)]
fn proto_segmentations_i8_i8(
    detect: &[crate::DetectBox],
    coeff_all: &[i8],
    coeff_quant: &edgefirst_tensor::Quantization,
    protos: &[i8],
    proto_quant: &edgefirst_tensor::Quantization,
    proto_h: usize,
    proto_w: usize,
    num_protos: usize,
    lx0: f32,
    inv_lw: f32,
    ly0: f32,
    inv_lh: f32,
    layout: edgefirst_decoder::ProtoLayout,
) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
    use edgefirst_tensor::QuantMode;

    let _span = tracing::trace_span!(
        "mask_i8_fastpath",
        n = detect.len(),
        proto_h,
        proto_w,
        num_protos,
        ?layout,
    )
    .entered();

    let zp_c: i32 = match coeff_quant.mode() {
        QuantMode::PerTensor { zero_point, .. } => zero_point,
        QuantMode::PerTensorSymmetric { .. } => 0,
        _ => {
            return Err(crate::Error::NotSupported(
                "per-channel coeff quantization not supported on proto-res i8 path".into(),
            ))
        }
    };
    let zp_p: i32 = match proto_quant.mode() {
        QuantMode::PerTensor { zero_point, .. } => zero_point,
        QuantMode::PerTensorSymmetric { .. } => 0,
        _ => {
            return Err(crate::Error::NotSupported(
                "per-channel proto quantization not supported on proto-res i8 path".into(),
            ))
        }
    };

    let hw = proto_h * proto_w;

    // Precompute per-pixel proto sums for zero-point correction.
    let proto_sums: Vec<i32> = if zp_c != 0 {
        match layout {
            edgefirst_decoder::ProtoLayout::Nhwc => (0..hw)
                .map(|px_idx| {
                    let base = px_idx * num_protos;
                    protos[base..base + num_protos]
                        .iter()
                        .map(|&v| v as i32)
                        .sum()
                })
                .collect(),
            edgefirst_decoder::ProtoLayout::Nchw => {
                let mut sums = vec![0i32; hw];
                for c in 0..num_protos {
                    let plane = &protos[c * hw..];
                    for (px, s) in sums.iter_mut().enumerate() {
                        *s += plane[px] as i32;
                    }
                }
                sums
            }
        }
    } else {
        Vec::new()
    };

    #[cfg(target_arch = "aarch64")]
    let use_dotprod = std::arch::is_aarch64_feature_detected!("dotprod");

    detect
        .par_iter()
        .enumerate()
        .map(|(i, det)| {
            let coeff = &coeff_all[i * num_protos..(i + 1) * num_protos];
            let (x0, y0, x1, y1, roi_w, roi_h) = bbox_to_proto_roi(det, proto_w, proto_h);

            // Per-detection bias: zp_p·Σc_raw - N·zp_c·zp_p
            let coeff_sum: i32 = coeff.iter().map(|&c| c as i32).sum();
            let bias = zp_p * coeff_sum - (num_protos as i32) * zp_c * zp_p;

            let mut mask_buf = vec![0u8; roi_h * roi_w];

            match layout {
                edgefirst_decoder::ProtoLayout::Nhwc => {
                    let stride_y = proto_w * num_protos;
                    #[cfg(target_arch = "aarch64")]
                    {
                        if use_dotprod {
                            for ly in 0..roi_h {
                                let py = y0 + ly;
                                let row_base = py * stride_y + x0 * num_protos;
                                for lx in 0..roi_w {
                                    let pix_base = row_base + lx * num_protos;
                                    let proto_px = &protos[pix_base..pix_base + num_protos];
                                    let raw_dot = unsafe {
                                        dot_i8_neon_dotprod(
                                            coeff.as_ptr(),
                                            proto_px.as_ptr(),
                                            num_protos,
                                        )
                                    };
                                    let correction = if zp_c != 0 {
                                        zp_c * proto_sums[py * proto_w + x0 + lx]
                                    } else {
                                        0
                                    };
                                    let logit = raw_dot - correction - bias;
                                    if logit > 0 {
                                        mask_buf[ly * roi_w + lx] = 255;
                                    }
                                }
                            }
                        } else {
                            for ly in 0..roi_h {
                                let py = y0 + ly;
                                let row_base = py * stride_y + x0 * num_protos;
                                for lx in 0..roi_w {
                                    let pix_base = row_base + lx * num_protos;
                                    let proto_px = &protos[pix_base..pix_base + num_protos];
                                    let raw_dot = unsafe {
                                        dot_i8_neon_base(
                                            coeff.as_ptr(),
                                            proto_px.as_ptr(),
                                            num_protos,
                                        )
                                    };
                                    let correction = if zp_c != 0 {
                                        zp_c * proto_sums[py * proto_w + x0 + lx]
                                    } else {
                                        0
                                    };
                                    let logit = raw_dot - correction - bias;
                                    if logit > 0 {
                                        mask_buf[ly * roi_w + lx] = 255;
                                    }
                                }
                            }
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        for ly in 0..roi_h {
                            let py = y0 + ly;
                            let row_base = py * stride_y + x0 * num_protos;
                            for lx in 0..roi_w {
                                let pix_base = row_base + lx * num_protos;
                                let proto_px = &protos[pix_base..pix_base + num_protos];
                                let raw_dot = dot_i8_scalar(coeff, proto_px, num_protos);
                                let correction = if zp_c != 0 {
                                    zp_c * proto_sums[py * proto_w + x0 + lx]
                                } else {
                                    0
                                };
                                let logit = raw_dot - correction - bias;
                                if logit > 0 {
                                    mask_buf[ly * roi_w + lx] = 255;
                                }
                            }
                        }
                    }
                }
                edgefirst_decoder::ProtoLayout::Nchw => {
                    // Channel-major accumulation: for each channel, accumulate
                    // coeff[c] * proto[c, py, px] across the ROI. Each channel
                    // plane is contiguous, giving excellent sequential read access.
                    let mut accum = vec![0i32; roi_h * roi_w];
                    for c in 0..num_protos {
                        let plane = &protos[c * hw..];
                        let coeff_c = coeff[c] as i32;
                        for ly in 0..roi_h {
                            let py = y0 + ly;
                            let row_start = py * proto_w + x0;
                            let out_row_start = ly * roi_w;
                            for lx in 0..roi_w {
                                accum[out_row_start + lx] += coeff_c * plane[row_start + lx] as i32;
                            }
                        }
                    }
                    // Apply zero-point correction and threshold.
                    for ly in 0..roi_h {
                        let py = y0 + ly;
                        for lx in 0..roi_w {
                            let idx = ly * roi_w + lx;
                            let correction = if zp_c != 0 {
                                zp_c * proto_sums[py * proto_w + x0 + lx]
                            } else {
                                0
                            };
                            let logit = accum[idx] - correction - bias;
                            if logit > 0 {
                                mask_buf[idx] = 255;
                            }
                        }
                    }
                }
            }

            let mask = ndarray::Array3::from_shape_vec((roi_h, roi_w, 1), mask_buf)
                .expect("mask_buf length matches roi_h * roi_w");
            Ok(seg_from_roi(
                mask, x0, y0, x1, y1, proto_w, proto_h, lx0, inv_lw, ly0, inv_lh,
            ))
        })
        .collect()
}

// =============================================================================
// Sign-threshold proto-resolution kernels (f32/f16/i8 protos with f32 coeffs).
//
// These replace the sigmoid-computing kernels for the non-i8×i8 fallback paths.
// Since downstream always thresholds at > 127, computing sigmoid is wasteful;
// sign(dot) > 0 ⟺ sigmoid(dot) > 0.5 gives the same binary result.
// =============================================================================

/// f32 protos × f32 coefficients → sign threshold → binary {0, 255}.
#[allow(clippy::too_many_arguments)]
fn fused_dot_sign_f32_slice(
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
    let mut mask_buf = vec![0u8; roi_h * roi_w];
    for y in 0..roi_h {
        let row_base = (y0 + y) * stride_y + x0 * num_protos;
        let out_row = &mut mask_buf[y * roi_w..(y + 1) * roi_w];
        for (x, out_px) in out_row.iter_mut().enumerate() {
            let base = row_base + x * num_protos;
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
            if acc > 0.0 {
                *out_px = 255;
            }
        }
    }
    ndarray::Array3::from_shape_vec((roi_h, roi_w, 1), mask_buf)
        .expect("mask_buf length matches roi_h * roi_w")
}

/// f16 protos × f32 coefficients → sign threshold → binary {0, 255}.
///
/// Two code paths:
///
/// 1. **x86_64 + F16C + FMA** — explicit intrinsic kernel using
///    `_mm256_cvtph_ps` (8-lane f16→f32 widening) + `_mm256_fmadd_ps`.
///
/// 2. **Scalar fallback** — loop-unrolled by 4 with `half::f16::to_f32()`.
#[allow(clippy::too_many_arguments)]
fn fused_dot_sign_f16_slice(
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
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "f16c",
        target_feature = "fma"
    ))]
    {
        // SAFETY: target-feature gates guarantee F16C + FMA support.
        unsafe {
            fused_dot_sign_f16_slice_f16c(protos, coeff, proto_w, y0, x0, roi_h, roi_w, num_protos)
        }
    }
    #[cfg(not(all(
        target_arch = "x86_64",
        target_feature = "f16c",
        target_feature = "fma"
    )))]
    {
        fused_dot_sign_f16_slice_scalar(protos, coeff, proto_w, y0, x0, roi_h, roi_w, num_protos)
    }
}

/// Scalar f16 sign-threshold kernel — loop-unrolled by 4.
#[allow(clippy::too_many_arguments)]
fn fused_dot_sign_f16_slice_scalar(
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
    let mut mask_buf = vec![0u8; roi_h * roi_w];
    for y in 0..roi_h {
        let row_base = (y0 + y) * stride_y + x0 * num_protos;
        let out_row = &mut mask_buf[y * roi_w..(y + 1) * roi_w];
        for (x, out_px) in out_row.iter_mut().enumerate() {
            let base = row_base + x * num_protos;
            let mut acc = 0.0_f32;
            let mut k = 0;
            let chunks = num_protos / 4;
            for _ in 0..chunks {
                acc += coeff[k] * protos[base + k].to_f32()
                    + coeff[k + 1] * protos[base + k + 1].to_f32()
                    + coeff[k + 2] * protos[base + k + 2].to_f32()
                    + coeff[k + 3] * protos[base + k + 3].to_f32();
                k += 4;
            }
            while k < num_protos {
                acc += coeff[k] * protos[base + k].to_f32();
                k += 1;
            }
            if acc > 0.0 {
                *out_px = 255;
            }
        }
    }
    ndarray::Array3::from_shape_vec((roi_h, roi_w, 1), mask_buf)
        .expect("mask_buf length matches roi_h * roi_w")
}

/// x86_64 F16C + FMA intrinsic kernel for f16 sign-threshold.
///
/// Uses `_mm256_cvtph_ps` for hardware 8-lane f16→f32 widening and
/// `_mm256_fmadd_ps` for fused multiply-add. Only the sign of the
/// accumulated dot product is checked (no sigmoid needed).
///
/// # Safety
///
/// Caller must ensure the target CPU supports F16C + FMA.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "f16c",
    target_feature = "fma"
))]
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "f16c,fma,avx")]
unsafe fn fused_dot_sign_f16_slice_f16c(
    protos: &[half::f16],
    coeff: &[f32],
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
    let mut mask_buf = vec![0u8; roi_h * roi_w];

    for y in 0..roi_h {
        let row_base = (y0 + y) * stride_y + x0 * num_protos;
        let out_row = &mut mask_buf[y * roi_w..(y + 1) * roi_w];
        for (x, out_px) in out_row.iter_mut().enumerate() {
            let base = row_base + x * num_protos;
            let mut acc_v = _mm256_setzero_ps();
            let mut k = 0;
            for _ in 0..chunks8 {
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

            // Scalar tail for num_protos % 8.
            while k < num_protos {
                acc += coeff[k] * protos[base + k].to_f32();
                k += 1;
            }

            if acc > 0.0 {
                *out_px = 255;
            }
        }
    }
    ndarray::Array3::from_shape_vec((roi_h, roi_w, 1), mask_buf)
        .expect("mask_buf length matches roi_h * roi_w")
}

/// i8 protos (with quant) × f32 coefficients → sign threshold → binary {0, 255}.
/// Fallback for per-channel quant or mixed-dtype cases where the i8×i8 fast path
/// doesn't apply.
#[allow(clippy::too_many_arguments)]
fn fused_dequant_dot_sign_i8_slice(
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

    // Precompute scaled coefficients + zp_offset (same as the old sigmoid kernel).
    let mut stack_scratch = [0.0_f32; 64];
    let mut heap_scratch: Vec<f32>;
    let scaled_coeff: &mut [f32] = if num_protos <= stack_scratch.len() {
        &mut stack_scratch[..num_protos]
    } else {
        heap_scratch = vec![0.0_f32; num_protos];
        heap_scratch.as_mut_slice()
    };
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
                return Err(crate::Error::NotSupported(format!(
                    "per-channel quantization on axis {axis} not supported \
                     (only channel axis 2 is implemented on this kernel)"
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
                return Err(crate::Error::NotSupported(format!(
                    "per-channel quantization on axis {axis} not supported \
                     (only channel axis 2 is implemented on this kernel)"
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

    let mut mask_buf = vec![0u8; roi_h * roi_w];
    for y in 0..roi_h {
        let row_base = (y0 + y) * stride_y + (x0) * num_protos;
        let out_row = &mut mask_buf[y * roi_w..(y + 1) * roi_w];
        for (x, out_px) in out_row.iter_mut().enumerate() {
            let base = row_base + x * num_protos;
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
            if acc > zp_offset {
                *out_px = 255;
            }
        }
    }
    Ok(ndarray::Array3::from_shape_vec((roi_h, roi_w, 1), mask_buf)
        .expect("mask_buf length matches roi_h * roi_w"))
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
    // per-channel scaled coefficients in scaled_run's dot-product precompute).
    let (scale, zp) = match quant.mode() {
        QuantMode::PerTensor { scale, zero_point } => (scale, zero_point as f32),
        QuantMode::PerTensorSymmetric { scale } => (scale, 0.0),
        QuantMode::PerChannel { axis, .. } | QuantMode::PerChannelSymmetric { axis, .. } => {
            return Err(crate::Error::NotSupported(format!(
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

// =============================================================================
// Integer-domain kernel: i8 coefficients × i8 protos → i32 → sign threshold.
//
// Eliminates all f32 conversion by working directly with raw quantized values.
// The math:
//   sign(dot(dequant(coeff), dequant(proto)))
//   = sign(Σ (c_raw - zp_c) · (p_raw - zp_p))
//   = sign(Σ c_raw·p_raw - zp_c·Σp_raw - zp_p·Σc_raw + N·zp_c·zp_p)
//   = sign(sdot(c_raw, p_raw) - zp_c·proto_sum[pixel] - bias_per_det)
//
// where bias_per_det = zp_p·Σc_raw - N·zp_c·zp_p  (precomputed once per det).
// =============================================================================

/// Compute i8×i8 dot product (32 elements) → i32.
/// Platform-agnostic scalar fallback.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline(always)]
fn dot_i8_scalar(coeff: &[i8], proto: &[i8], n: usize) -> i32 {
    let mut acc: i32 = 0;
    let chunks = n / 4;
    let mut k = 0;
    for _ in 0..chunks {
        acc += coeff[k] as i32 * proto[k] as i32
            + coeff[k + 1] as i32 * proto[k + 1] as i32
            + coeff[k + 2] as i32 * proto[k + 2] as i32
            + coeff[k + 3] as i32 * proto[k + 3] as i32;
        k += 4;
    }
    while k < n {
        acc += coeff[k] as i32 * proto[k] as i32;
        k += 1;
    }
    acc
}

/// NEON i8×i8→i32 dot product using smull+sadalp (works on ALL aarch64, A53+).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_i8_neon_base(coeff: *const i8, proto: *const i8, n: usize) -> i32 {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_s32(0);
    let full_chunks = n / 16;
    let mut offset = 0usize;
    for _ in 0..full_chunks {
        let c = vld1q_s8(coeff.add(offset));
        let p = vld1q_s8(proto.add(offset));
        // Widening multiply + pairwise accumulate (all aarch64).
        let lo = vmull_s8(vget_low_s8(c), vget_low_s8(p));
        let hi = vmull_high_s8(c, p);
        acc = vpadalq_s16(acc, lo);
        acc = vpadalq_s16(acc, hi);
        offset += 16;
    }
    // Handle remaining elements (for num_protos=32, full_chunks=2, remainder=0)
    let remainder = n - offset;
    if remainder >= 8 {
        let c = vld1_s8(coeff.add(offset));
        let p = vld1_s8(proto.add(offset));
        let prod = vmull_s8(c, p);
        acc = vpadalq_s16(acc, prod);
        offset += 8;
    }
    let mut scalar_acc = vaddvq_s32(acc);
    while offset < n {
        scalar_acc += *coeff.add(offset) as i32 * *proto.add(offset) as i32;
        offset += 1;
    }
    scalar_acc
}

/// NEON i8×i8→i32 dot product using sdot (ARMv8.2-A dotprod, A55+).
/// Each `sdot` processes 16 i8 lanes → 4 i32 partial sums in one instruction,
/// replacing the 3-instruction smull+smull2+sadalp sequence.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn dot_i8_neon_dotprod(coeff: *const i8, proto: *const i8, n: usize) -> i32 {
    use std::arch::aarch64::*;
    let mut acc = vdupq_n_s32(0);
    let full_chunks = n / 16;
    let mut offset = 0usize;
    for _ in 0..full_chunks {
        let c = vld1q_s8(coeff.add(offset));
        let p = vld1q_s8(proto.add(offset));
        // Enable dotprod extension locally so the assembler accepts sdot
        // even when compiling for baseline aarch64 (A53). At runtime we only
        // reach this path when HWCAP confirms dotprod support.
        let result: int32x4_t;
        core::arch::asm!(
            ".arch_extension dotprod",
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc => result,
            a = in(vreg) c,
            b = in(vreg) p,
            options(pure, nomem, nostack),
        );
        acc = result;
        offset += 16;
    }
    let mut scalar_acc = vaddvq_s32(acc);
    // Tail: handle remainder (unlikely for num_protos=32, but correct)
    while offset < n {
        scalar_acc += *coeff.add(offset) as i32 * *proto.add(offset) as i32;
        offset += 1;
    }
    scalar_acc
}

/// Compute the logit grid using the dotprod (sdot) path.
/// Separated into its own function so the compiler inlines the sdot asm fully.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn compute_logits_dotprod(
    logits: &mut [i32],
    coeff: &[i8],
    protos: &[i8],
    proto_sums: &[i32],
    proto_w: usize,
    proto_x0: usize,
    proto_y0: usize,
    roi_w: usize,
    roi_h: usize,
    stride_y: usize,
    num_protos: usize,
    zp_c: i32,
    bias: i32,
) {
    for ly_idx in 0..roi_h {
        let py = proto_y0 + ly_idx;
        let row_base = py * stride_y + proto_x0 * num_protos;
        for lx_idx in 0..roi_w {
            let pix_base = row_base + lx_idx * num_protos;
            let proto_px = &protos[pix_base..pix_base + num_protos];
            let raw_dot =
                unsafe { dot_i8_neon_dotprod(coeff.as_ptr(), proto_px.as_ptr(), num_protos) };
            let correction = if zp_c != 0 {
                zp_c * proto_sums[py * proto_w + proto_x0 + lx_idx]
            } else {
                0
            };
            logits[ly_idx * roi_w + lx_idx] = raw_dot - correction - bias;
        }
    }
}

/// Compute the logit grid using the base NEON path (smull+sadalp).
/// Separated into its own function so the compiler inlines the NEON code fully.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn compute_logits_base(
    logits: &mut [i32],
    coeff: &[i8],
    protos: &[i8],
    proto_sums: &[i32],
    proto_w: usize,
    proto_x0: usize,
    proto_y0: usize,
    roi_w: usize,
    roi_h: usize,
    stride_y: usize,
    num_protos: usize,
    zp_c: i32,
    bias: i32,
) {
    for ly_idx in 0..roi_h {
        let py = proto_y0 + ly_idx;
        let row_base = py * stride_y + proto_x0 * num_protos;
        for lx_idx in 0..roi_w {
            let pix_base = row_base + lx_idx * num_protos;
            let proto_px = &protos[pix_base..pix_base + num_protos];
            let raw_dot =
                unsafe { dot_i8_neon_base(coeff.as_ptr(), proto_px.as_ptr(), num_protos) };
            let correction = if zp_c != 0 {
                zp_c * proto_sums[py * proto_w + proto_x0 + lx_idx]
            } else {
                0
            };
            logits[ly_idx * roi_w + lx_idx] = raw_dot - correction - bias;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn scaled_segmentations_i8_i8(
    detect: &[crate::DetectBox],
    coeff_all: &[i8],
    coeff_quant: &edgefirst_tensor::Quantization,
    protos: &[i8],
    proto_quant: &edgefirst_tensor::Quantization,
    proto_h: usize,
    proto_w: usize,
    num_protos: usize,
    letterbox: Option<[f32; 4]>,
    width: u32,
    height: u32,
    layout: edgefirst_decoder::ProtoLayout,
) -> crate::Result<Vec<edgefirst_decoder::Segmentation>> {
    use edgefirst_tensor::QuantMode;

    let _span = tracing::trace_span!(
        "mask_i8_fastpath",
        n = detect.len(),
        proto_h,
        proto_w,
        num_protos,
        width,
        height,
        ?layout,
    )
    .entered();

    let zp_c: i32 = match coeff_quant.mode() {
        QuantMode::PerTensor { zero_point, .. } => zero_point,
        QuantMode::PerTensorSymmetric { .. } => 0,
        _ => {
            return Err(crate::Error::NotSupported(
                "per-channel coeff quantization not supported".into(),
            ))
        }
    };
    let zp_p: i32 = match proto_quant.mode() {
        QuantMode::PerTensor { zero_point, .. } => zero_point,
        QuantMode::PerTensorSymmetric { .. } => 0,
        _ => {
            return Err(crate::Error::NotSupported(
                "per-channel proto quantization not supported".into(),
            ))
        }
    };

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
    let hw = proto_h * proto_w;

    // Precompute proto_sum for the entire proto tensor (zero-point correction).
    let proto_sums: Vec<i32> = if zp_c != 0 {
        match layout {
            edgefirst_decoder::ProtoLayout::Nhwc => (0..hw)
                .map(|px_idx| {
                    let base = px_idx * num_protos;
                    let mut s: i32 = 0;
                    for k in 0..num_protos {
                        s += protos[base + k] as i32;
                    }
                    s
                })
                .collect(),
            edgefirst_decoder::ProtoLayout::Nchw => {
                let mut sums = vec![0i32; hw];
                for c in 0..num_protos {
                    let plane = &protos[c * hw..];
                    for (px, s) in sums.iter_mut().enumerate() {
                        *s += plane[px] as i32;
                    }
                }
                sums
            }
        }
    } else {
        Vec::new()
    };

    // Detect dotprod support once, outside the hot loop.
    #[cfg(target_arch = "aarch64")]
    let use_dotprod = std::arch::is_aarch64_feature_detected!("dotprod");

    // For NHWC layout, stride for row navigation.
    let stride_y = proto_w * num_protos;

    detect
        .par_iter()
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

            // Map output bbox → proto ROI.
            let sample_x_at = |px: f32| -> f32 {
                let model_x_norm = lx0 + (px + 0.5) / out_w as f32 * lw;
                model_x_norm * proto_w as f32 - 0.5
            };
            let sample_y_at = |py: f32| -> f32 {
                let model_y_norm = ly0 + (py + 0.5) / out_h as f32 * lh;
                model_y_norm * proto_h as f32 - 0.5
            };
            let s_x_min = sample_x_at(px0 as f32);
            let s_x_max = sample_x_at((px1 as f32) - 1.0);
            let s_y_min = sample_y_at(py0 as f32);
            let s_y_max = sample_y_at((py1 as f32) - 1.0);
            let proto_x0 = (s_x_min.floor() as isize)
                .max(0)
                .min(proto_w.saturating_sub(1) as isize) as usize;
            let proto_x1 = ((s_x_max.ceil() as isize) + 1).max(0).min(proto_w as isize) as usize;
            let proto_y0 = (s_y_min.floor() as isize)
                .max(0)
                .min(proto_h.saturating_sub(1) as isize) as usize;
            let proto_y1 = ((s_y_max.ceil() as isize) + 1).max(0).min(proto_h as isize) as usize;
            let roi_w = proto_x1.saturating_sub(proto_x0).max(1);
            let roi_h = proto_y1.saturating_sub(proto_y0).max(1);

            // Per-detection bias.
            let coeff_sum: i32 = coeff.iter().map(|&c| c as i32).sum();
            let bias = zp_p * coeff_sum - (num_protos as i32) * zp_c * zp_p;

            // Step 2: Compute i32 logits at each proto-ROI pixel.
            let mut logits = vec![0_i32; roi_h * roi_w];
            match layout {
                edgefirst_decoder::ProtoLayout::Nhwc => {
                    #[cfg(target_arch = "aarch64")]
                    {
                        if use_dotprod {
                            compute_logits_dotprod(
                                &mut logits,
                                coeff,
                                protos,
                                &proto_sums,
                                proto_w,
                                proto_x0,
                                proto_y0,
                                roi_w,
                                roi_h,
                                stride_y,
                                num_protos,
                                zp_c,
                                bias,
                            );
                        } else {
                            compute_logits_base(
                                &mut logits,
                                coeff,
                                protos,
                                &proto_sums,
                                proto_w,
                                proto_x0,
                                proto_y0,
                                roi_w,
                                roi_h,
                                stride_y,
                                num_protos,
                                zp_c,
                                bias,
                            );
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        for ly_idx in 0..roi_h {
                            let py = proto_y0 + ly_idx;
                            let row_base = py * stride_y + proto_x0 * num_protos;
                            for lx_idx in 0..roi_w {
                                let pix_base = row_base + lx_idx * num_protos;
                                let proto_px = &protos[pix_base..pix_base + num_protos];
                                let raw_dot = dot_i8_scalar(coeff, proto_px, num_protos);
                                let correction = if zp_c != 0 {
                                    zp_c * proto_sums[py * proto_w + proto_x0 + lx_idx]
                                } else {
                                    0
                                };
                                logits[ly_idx * roi_w + lx_idx] = raw_dot - correction - bias;
                            }
                        }
                    }
                }
                edgefirst_decoder::ProtoLayout::Nchw => {
                    // Channel-major accumulation: contiguous reads per channel plane.
                    for c in 0..num_protos {
                        let plane = &protos[c * hw..];
                        let coeff_c = coeff[c] as i32;
                        for ly_idx in 0..roi_h {
                            let py = proto_y0 + ly_idx;
                            let row_start = py * proto_w + proto_x0;
                            let out_row_start = ly_idx * roi_w;
                            for lx_idx in 0..roi_w {
                                logits[out_row_start + lx_idx] +=
                                    coeff_c * plane[row_start + lx_idx] as i32;
                            }
                        }
                    }
                    // Apply zero-point correction and per-detection bias.
                    for ly_idx in 0..roi_h {
                        let py = proto_y0 + ly_idx;
                        for lx_idx in 0..roi_w {
                            let idx = ly_idx * roi_w + lx_idx;
                            let correction = if zp_c != 0 {
                                zp_c * proto_sums[py * proto_w + proto_x0 + lx_idx]
                            } else {
                                0
                            };
                            logits[idx] -= correction + bias;
                        }
                    }
                }
            }

            // Step 3: Bilinear upsample i32 logits → binary mask with
            // sign-shortcut (skip interpolation when all 4 neighbors agree).
            let roi_last_x = roi_w.saturating_sub(1);
            let roi_last_y = roi_h.saturating_sub(1);

            // X-coordinate LUT with fixed-point fraction (scale 1024).
            const FRAC_BITS: i32 = 10;
            const FRAC_SCALE: i32 = 1 << FRAC_BITS; // 1024
            let x_coords: Vec<(usize, usize, i32)> = (0..bbox_w)
                .map(|xi| {
                    let sample_x = sample_x_at((px0 + xi) as f32) - proto_x0 as f32;
                    let x_floor = sample_x.floor();
                    let x_lo = (x_floor as isize).max(0).min(roi_last_x as isize) as usize;
                    let x_hi = (x_lo + 1).min(roi_w - 1);
                    let x_frac = ((sample_x - x_floor).clamp(0.0, 1.0) * FRAC_SCALE as f32) as i32;
                    (x_lo, x_hi, x_frac)
                })
                .collect();

            let mut tile_buf = vec![0u8; bbox_h * bbox_w];
            for yi in 0..bbox_h {
                let sample_y = sample_y_at((py0 + yi) as f32) - proto_y0 as f32;
                let y_floor = sample_y.floor();
                let y_lo = (y_floor as isize).max(0).min(roi_last_y as isize) as usize;
                let y_hi = (y_lo + 1).min(roi_h - 1);
                let y_frac = ((sample_y - y_floor).clamp(0.0, 1.0) * FRAC_SCALE as f32) as i32;
                let y_frac_inv = FRAC_SCALE - y_frac;
                let row_lo = &logits[y_lo * roi_w..y_lo * roi_w + roi_w];
                let row_hi = &logits[y_hi * roi_w..y_hi * roi_w + roi_w];
                let out_row = &mut tile_buf[yi * bbox_w..(yi + 1) * bbox_w];

                for (xi, &(x_lo, x_hi, x_frac)) in x_coords.iter().enumerate() {
                    let tl = row_lo[x_lo];
                    let tr = row_lo[x_hi];
                    let bl = row_hi[x_lo];
                    let br = row_hi[x_hi];

                    // Sign-shortcut: if all 4 corners have the same sign,
                    // the bilinear interpolation (positive-weight combination)
                    // preserves that sign. Skip arithmetic for ~80% of pixels.
                    if (tl & tr & bl & br) < 0 {
                        // All negative → output 0 (already zero).
                        continue;
                    }
                    if tl > 0 && tr > 0 && bl > 0 && br > 0 {
                        // All strictly positive → output 255.
                        out_row[xi] = 255;
                        continue;
                    }

                    // Boundary pixel: fixed-point bilinear in i64.
                    let x_frac_inv = FRAC_SCALE - x_frac;
                    let l0 = tl as i64 * x_frac_inv as i64 + tr as i64 * x_frac as i64;
                    let l1 = bl as i64 * x_frac_inv as i64 + br as i64 * x_frac as i64;
                    let logit = l0 * y_frac_inv as i64 + l1 * y_frac as i64;
                    out_row[xi] = if logit > 0 { 255 } else { 0 };
                }
            }

            let tile = ndarray::Array3::from_shape_vec((bbox_h, bbox_w, 1), tile_buf)
                .expect("tile_buf length matches bbox_h * bbox_w");
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

#[allow(clippy::too_many_arguments)]
fn scaled_run<P: Copy + Sync>(
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
    load_f32: impl Fn(&P, f32) -> f32 + Copy + Sync,
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

    // Parallelise across detections. Each detection produces an
    // independent ndarray::Array3<u8> tile from a read-only proto slice +
    // its own coeff slice; no shared mutable state.
    //
    // Algorithm (restores the spirit of PR #54's batched-GEMM optimisation
    // that PR #51's f16 dispatch refactor inadvertently removed):
    //
    //   1. Map the output bbox back to a proto-plane ROI (with 1-px margin
    //      so the bilinear sampling at the output edges has neighbours).
    //   2. Precompute *f32 logits* at every proto pixel inside that ROI by
    //      doing a single K-wide dot product per proto pixel — once, not
    //      once per output pixel.
    //   3. For each output pixel, bilinear-interpolate the scalar f32 logit
    //      from the 4 surrounding proto-roi pixels, apply sigmoid, and
    //      threshold to {0, 255}.
    //
    // For typical YOLO-seg: proto_roi ~ 30×30 = 900 px × K=32 = 28.8K dot
    // ops vs the legacy "bilinear sample then dot at every output pixel"
    // which costs bbox_h × bbox_w × 4 × K = ~1.3M ops at 100×100 output
    // bbox. ~45× fewer FMAs at this size; the bilinear upsample of a
    // scalar plane (no inner K loop) is comparatively negligible.
    detect
        .par_iter()
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

            // Step 1 — proto-plane ROI for this detection's output bbox.
            // Map the four output bbox corners back to proto coords and
            // expand by 1 pixel in each direction so the bilinear sampler
            // at the bbox boundary has both neighbours.
            let sample_x_at = |px: f32| -> f32 {
                let model_x_norm = lx0 + (px + 0.5) / out_w as f32 * lw;
                model_x_norm * proto_w as f32 - 0.5
            };
            let sample_y_at = |py: f32| -> f32 {
                let model_y_norm = ly0 + (py + 0.5) / out_h as f32 * lh;
                model_y_norm * proto_h as f32 - 0.5
            };
            let s_x_min = sample_x_at(px0 as f32);
            let s_x_max = sample_x_at((px1 as f32) - 1.0);
            let s_y_min = sample_y_at(py0 as f32);
            let s_y_max = sample_y_at((py1 as f32) - 1.0);
            // Floor min, ceil max+1 to include both bilinear neighbours.
            // Start indices are used as direct bases into `protos`, so clamp
            // them to the last valid index, not to the exclusive upper bound.
            let proto_x0 = (s_x_min.floor() as isize)
                .max(0)
                .min(proto_w.saturating_sub(1) as isize) as usize;
            let proto_x1 = ((s_x_max.ceil() as isize) + 1).max(0).min(proto_w as isize) as usize;
            let proto_y0 = (s_y_min.floor() as isize)
                .max(0)
                .min(proto_h.saturating_sub(1) as isize) as usize;
            let proto_y1 = ((s_y_max.ceil() as isize) + 1).max(0).min(proto_h as isize) as usize;
            let roi_w = proto_x1.saturating_sub(proto_x0).max(1);
            let roi_h = proto_y1.saturating_sub(proto_y0).max(1);

            // Step 2 — precompute f32 logits at every proto-roi pixel.
            // logits[(py - proto_y0) * roi_w + (px - proto_x0)] = dot(coeff, proto[py, px, :])
            //
            // Since the final threshold is `logit > 0` (O1) and bilinear
            // interpolation is a positive-weight linear combination,
            // `acc_scale * interp(logits) > 0 ⟺ interp(logits) > 0` when
            // acc_scale > 0. We therefore skip the per-pixel `acc_scale *`
            // multiply entirely, storing raw dot products.
            if !acc_scale.is_finite() || acc_scale <= 0.0 {
                return Err(crate::Error::NotSupported(format!(
                    "acc_scale must be finite and positive for sign-threshold optimization (got {acc_scale})"
                )));
            }
            let _ = acc_scale; // Scale-invariant: only sign matters.
            let mut logits = vec![0.0_f32; roi_h * roi_w];
            for ly_idx in 0..roi_h {
                let py = proto_y0 + ly_idx;
                let row_base = py * stride_y + proto_x0 * num_protos;
                for lx_idx in 0..roi_w {
                    let pix_base = row_base + lx_idx * num_protos;
                    let mut acc = 0.0_f32;
                    // 4-wide unroll to help auto-vectorization.
                    let mut k = 0;
                    let chunks = num_protos / 4;
                    for _ in 0..chunks {
                        acc += coeff[k] * load_f32(&protos[pix_base + k], 0.0)
                            + coeff[k + 1] * load_f32(&protos[pix_base + k + 1], 0.0)
                            + coeff[k + 2] * load_f32(&protos[pix_base + k + 2], 0.0)
                            + coeff[k + 3] * load_f32(&protos[pix_base + k + 3], 0.0);
                        k += 4;
                    }
                    while k < num_protos {
                        acc += coeff[k] * load_f32(&protos[pix_base + k], 0.0);
                        k += 1;
                    }
                    logits[ly_idx * roi_w + lx_idx] = acc;
                }
            }

            // Step 3 — bilinear upsample logits → binary mask.
            //
            // O1: sigmoid(x) > 0.5 ⟺ x > 0 (sigmoid is strictly monotonic,
            // and acc_scale > 0 preserves sign). The sign threshold replaces
            // the old fast_sigmoid approximation, saving ~15 cycles/pixel.
            //
            // O5: Pre-compute bilinear sample coordinates. sample_x_at /
            // sample_y_at depend only on pixel index, not on logit values.
            // Building lookup tables avoids redundant float ops in the inner
            // loop (floor, clamp, isize cast per pixel).
            let roi_last_x = roi_w.saturating_sub(1);
            let roi_last_y = roi_h.saturating_sub(1);

            // X-coordinate LUT (shared across all rows).
            let x_coords: Vec<(u32, u32, f32)> = (0..bbox_w)
                .map(|xi| {
                    let sample_x = sample_x_at((px0 + xi) as f32) - proto_x0 as f32;
                    let x_floor = sample_x.floor();
                    let x_lo = (x_floor as isize).max(0).min(roi_last_x as isize) as u32;
                    let x_hi = (x_lo as usize + 1).min(roi_w - 1) as u32;
                    let x_frac = (sample_x - x_floor).clamp(0.0, 1.0);
                    (x_lo, x_hi, x_frac)
                })
                .collect();

            // Write the output tile through a contiguous slice to avoid
            // ndarray's per-element bounds checks + stride arithmetic.
            let mut tile_buf = vec![0u8; bbox_h * bbox_w];
            for yi in 0..bbox_h {
                let sample_y = sample_y_at((py0 + yi) as f32) - proto_y0 as f32;
                let y_floor = sample_y.floor();
                let y_lo = (y_floor as isize).max(0).min(roi_last_y as isize) as usize;
                let y_hi = (y_lo + 1).min(roi_h - 1);
                let y_frac = (sample_y - y_floor).clamp(0.0, 1.0);
                let row_lo = &logits[y_lo * roi_w..y_lo * roi_w + roi_w];
                let row_hi = &logits[y_hi * roi_w..y_hi * roi_w + roi_w];
                let out_row = &mut tile_buf[yi * bbox_w..(yi + 1) * bbox_w];
                for (xi, &(x_lo, x_hi, x_frac)) in x_coords.iter().enumerate() {
                    let (xl, xh) = (x_lo as usize, x_hi as usize);
                    let l0 = row_lo[xl] + (row_lo[xh] - row_lo[xl]) * x_frac;
                    let l1 = row_hi[xl] + (row_hi[xh] - row_hi[xl]) * x_frac;
                    let logit = l0 + (l1 - l0) * y_frac;
                    out_row[xi] = if logit > 0.0 { 255 } else { 0 };
                }
            }
            // Wrap into the expected Array3<u8> shape [bbox_h, bbox_w, 1].
            let tile = ndarray::Array3::from_shape_vec((bbox_h, bbox_w, 1), tile_buf)
                .expect("tile_buf length matches bbox_h * bbox_w");
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
