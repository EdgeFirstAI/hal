// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use ndarray::{s, Array3, ArrayViewD};
#[cfg(feature = "tracker")]
use ndarray::{ArrayView2, ArrayView3};
use ndarray_stats::QuantileExt;
use num_traits::{AsPrimitive, Float};

use super::configs::{self};
use super::{ArrayViewDQuantized, Decoder};
use crate::{
    dequantize_ndarray,
    modelpack::{
        decode_modelpack_det, decode_modelpack_float, decode_modelpack_split_float,
        ModelPackDetectionConfig,
    },
    yolo::{
        decode_yolo_det, decode_yolo_det_float, decode_yolo_segdet_float, decode_yolo_segdet_quant,
        decode_yolo_split_det_float, decode_yolo_split_det_quant, decode_yolo_split_segdet_float,
        impl_yolo_split_segdet_quant_get_boxes, impl_yolo_split_segdet_quant_process_masks,
    },
    DecoderError, DetectBox, ProtoData, Quantization, Segmentation, XYWH,
};

macro_rules! dequant_3d {
    ($tensor:expr, $config:expr, $quant:expr) => {{
        with_quantized!($tensor, t, {
            let t = Self::swap_axes_if_needed(t, $config.into());
            let t = t.slice(s![0, .., ..]);
            t.map(|v| {
                let val: f32 = v.as_();
                (val - $quant.zero_point as f32) * $quant.scale
            })
        })
    }};
}
macro_rules! dequant_4d {
    ($tensor:expr, $config:expr, $quant:expr) => {{
        with_quantized!($tensor, t, {
            let t = Self::swap_axes_if_needed(t, $config.into());
            let t = t.slice(s![0, .., .., ..]);
            t.map(|v| {
                let val: f32 = v.as_();
                (val - $quant.zero_point as f32) * $quant.scale
            })
        })
    }};
}

impl Decoder {
    pub(super) fn decode_modelpack_det_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError> {
        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&boxes.shape, outputs, &[])?;
        let (scores_tensor, _) =
            Self::find_outputs_with_shape_quantized(&scores.shape, outputs, &[ind])?;
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_scores = scores
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        with_quantized!(boxes_tensor, b, {
            with_quantized!(scores_tensor, s, {
                let boxes_tensor = Self::swap_axes_if_needed(b, boxes.into());
                let boxes_tensor = boxes_tensor.slice(s![0, .., 0, ..]);

                let scores_tensor = Self::swap_axes_if_needed(s, scores.into());
                let scores_tensor = scores_tensor.slice(s![0, .., ..]);
                decode_modelpack_det(
                    (boxes_tensor, quant_boxes),
                    (scores_tensor, quant_scores),
                    self.score_threshold,
                    self.iou_threshold,
                    output_boxes,
                );
            });
        });

        Ok(())
    }

    pub(super) fn decode_modelpack_seg_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        segmentation: &configs::Segmentation,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError> {
        let (seg, _) = Self::find_outputs_with_shape_quantized(&segmentation.shape, outputs, &[])?;

        macro_rules! modelpack_seg {
            ($seg:expr, $body:expr) => {{
                let seg = Self::swap_axes_if_needed($seg, segmentation.into());
                let seg = seg.slice(s![0, .., .., ..]);
                seg.mapv($body)
            }};
        }
        use ArrayViewDQuantized::*;
        let seg = match seg {
            UInt8(s) => {
                modelpack_seg!(s, |x| x)
            }
            Int8(s) => {
                modelpack_seg!(s, |x| (x as i16 + 128) as u8)
            }
            UInt16(s) => {
                modelpack_seg!(s, |x| (x >> 8) as u8)
            }
            Int16(s) => {
                modelpack_seg!(s, |x| ((x as i32 + 32768) >> 8) as u8)
            }
            UInt32(s) => {
                modelpack_seg!(s, |x| (x >> 24) as u8)
            }
            Int32(s) => {
                modelpack_seg!(s, |x| ((x as i64 + 2147483648) >> 24) as u8)
            }
        };

        output_masks.push(Segmentation {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
            segmentation: seg,
        });
        Ok(())
    }

    pub(super) fn decode_modelpack_det_split_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        detection: &[configs::Detection],
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError> {
        let new_detection = detection
            .iter()
            .map(|x| match &x.anchors {
                None => Err(DecoderError::InvalidConfig(
                    "ModelPack Split Detection missing anchors".to_string(),
                )),
                Some(a) => Ok(ModelPackDetectionConfig {
                    anchors: a.clone(),
                    quantization: None,
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let new_outputs = Self::match_outputs_to_detect_quantized(detection, outputs)?;

        macro_rules! dequant_output {
            ($det_tensor:expr, $detection:expr) => {{
                let det_tensor = Self::swap_axes_if_needed($det_tensor, $detection.into());
                let det_tensor = det_tensor.slice(s![0, .., .., ..]);
                if let Some(q) = $detection.quantization {
                    dequantize_ndarray(det_tensor, q.into())
                } else {
                    det_tensor.map(|x| *x as f32)
                }
            }};
        }

        let new_outputs = new_outputs
            .iter()
            .zip(detection)
            .map(|(det_tensor, detection)| {
                with_quantized!(det_tensor, d, dequant_output!(d, detection))
            })
            .collect::<Vec<_>>();

        let new_outputs_view = new_outputs
            .iter()
            .map(|d: &Array3<f32>| d.view())
            .collect::<Vec<_>>();
        decode_modelpack_split_float(
            &new_outputs_view,
            &new_detection,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    pub(super) fn decode_yolo_det_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Detection,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError> {
        let (boxes_tensor, _) =
            Self::find_outputs_with_shape_quantized(&boxes.shape, outputs, &[])?;
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        with_quantized!(boxes_tensor, b, {
            let boxes_tensor = Self::swap_axes_if_needed(b, boxes.into());
            let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
            decode_yolo_det(
                (boxes_tensor, quant_boxes),
                self.score_threshold,
                self.iou_threshold,
                self.nms,
                output_boxes,
            );
        });

        Ok(())
    }

    pub(super) fn decode_yolo_segdet_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Detection,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError> {
        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&boxes.shape, outputs, &[])?;
        let (protos_tensor, _) =
            Self::find_outputs_with_shape_quantized(&protos.shape, outputs, &[ind])?;

        let quant_boxes = boxes
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = protos
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        with_quantized!(boxes_tensor, b, {
            with_quantized!(protos_tensor, p, {
                let box_tensor = Self::swap_axes_if_needed(b, boxes.into());
                let box_tensor = box_tensor.slice(s![0, .., ..]);

                let protos_tensor = Self::swap_axes_if_needed(p, protos.into());
                let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
                let protos_tensor = Self::protos_to_hwc(protos_tensor, protos);
                decode_yolo_segdet_quant(
                    (box_tensor, quant_boxes),
                    (protos_tensor, quant_protos),
                    self.score_threshold,
                    self.iou_threshold,
                    self.nms,
                    output_boxes,
                    output_masks,
                )
            })
        })
    }

    pub(super) fn decode_yolo_split_det_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError> {
        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&boxes.shape, outputs, &[])?;
        let (scores_tensor, _) =
            Self::find_outputs_with_shape_quantized(&scores.shape, outputs, &[ind])?;
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_scores = scores
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        with_quantized!(boxes_tensor, b, {
            with_quantized!(scores_tensor, s, {
                let boxes_tensor = Self::swap_axes_if_needed(b, boxes.into());
                let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);

                let scores_tensor = Self::swap_axes_if_needed(s, scores.into());
                let scores_tensor = scores_tensor.slice(s![0, .., ..]);
                decode_yolo_split_det_quant(
                    (boxes_tensor, quant_boxes),
                    (scores_tensor, quant_scores),
                    self.score_threshold,
                    self.iou_threshold,
                    self.nms,
                    output_boxes,
                );
            });
        });

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_yolo_split_segdet_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError> {
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_scores = scores
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_masks = mask_coeff
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = protos
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        let mut skip = vec![];

        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&boxes.shape, outputs, &skip)?;
        skip.push(ind);

        let (scores_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&scores.shape, outputs, &skip)?;
        skip.push(ind);

        let (mask_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&mask_coeff.shape, outputs, &skip)?;
        skip.push(ind);

        let (protos_tensor, _) =
            Self::find_outputs_with_shape_quantized(&protos.shape, outputs, &skip)?;

        let boxes = with_quantized!(boxes_tensor, b, {
            with_quantized!(scores_tensor, s, {
                let boxes_tensor = Self::swap_axes_if_needed(b, boxes.into());
                let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
                let boxes_tensor = boxes_tensor.reversed_axes();

                let scores_tensor = Self::swap_axes_if_needed(s, scores.into());
                let scores_tensor = scores_tensor.slice(s![0, .., ..]);
                let scores_tensor = scores_tensor.reversed_axes();

                impl_yolo_split_segdet_quant_get_boxes::<XYWH, _, _>(
                    (boxes_tensor, quant_boxes),
                    (scores_tensor, quant_scores),
                    self.score_threshold,
                    self.iou_threshold,
                    self.nms,
                    output_boxes.capacity(),
                )
            })
        });

        with_quantized!(mask_tensor, m, {
            with_quantized!(protos_tensor, p, {
                let mask_tensor = Self::swap_axes_if_needed(m, mask_coeff.into());
                let mask_tensor = mask_tensor.slice(s![0, .., ..]);
                let mask_tensor = mask_tensor.reversed_axes();

                let protos_tensor = Self::swap_axes_if_needed(p, protos.into());
                let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
                let protos_tensor = Self::protos_to_hwc(protos_tensor, protos);

                impl_yolo_split_segdet_quant_process_masks::<_, _>(
                    boxes,
                    (mask_tensor, quant_masks),
                    (protos_tensor, quant_protos),
                    output_boxes,
                    output_masks,
                )
            })
        })
    }

    /// Decode 2-way split: combined detection [1,nc+4,N] + separate
    /// mask_coeff [1,32,N] + protos [1,H,W,32].
    pub(super) fn decode_yolo_segdet_2way_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        detection: &configs::Detection,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError> {
        let quant_det = detection
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_masks = mask_coeff
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = protos
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        let mut skip = vec![];

        let (det_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&detection.shape, outputs, &skip)?;
        skip.push(ind);

        let (mask_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&mask_coeff.shape, outputs, &skip)?;
        skip.push(ind);

        let (protos_tensor, _) =
            Self::find_outputs_with_shape_quantized(&protos.shape, outputs, &skip)?;

        // Phase 1: Slice combined detection into boxes[0:4] and scores[4:],
        // run NMS. Both slices share the detection tensor's quantization.
        let boxes = with_quantized!(det_tensor, d, {
            let det = Self::swap_axes_if_needed(d, detection.into());
            let det = det.slice(s![0, .., ..]);
            let boxes_view = det.slice(s![..4, ..]);
            let scores_view = det.slice(s![4.., ..]);
            impl_yolo_split_segdet_quant_get_boxes::<XYWH, _, _>(
                (boxes_view, quant_det),
                (scores_view, quant_det),
                self.score_threshold,
                self.iou_threshold,
                self.nms,
                output_boxes.capacity(),
            )
        });

        // Phase 2: Process masks with separate mask_coeff tensor.
        with_quantized!(mask_tensor, m, {
            with_quantized!(protos_tensor, p, {
                let mask_tensor = Self::swap_axes_if_needed(m, mask_coeff.into());
                let mask_tensor = mask_tensor.slice(s![0, .., ..]);

                let protos_tensor = Self::swap_axes_if_needed(p, protos.into());
                let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
                let protos_tensor = Self::protos_to_hwc(protos_tensor, protos);
                impl_yolo_split_segdet_quant_process_masks::<_, _>(
                    boxes,
                    (mask_tensor, quant_masks),
                    (protos_tensor, quant_protos),
                    output_boxes,
                    output_masks,
                )
            })
        })
    }

    /// Decode 2-way split (float): combined detection + separate mask_coeff +
    /// protos.
    pub(super) fn decode_yolo_segdet_2way_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        detection: &configs::Detection,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let mut skip = vec![];
        let (det_tensor, ind) = Self::find_outputs_with_shape(&detection.shape, outputs, &skip)?;
        let det_tensor = Self::swap_axes_if_needed(det_tensor, detection.into());
        let det_tensor = det_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (mask_tensor, ind) = Self::find_outputs_with_shape(&mask_coeff.shape, outputs, &skip)?;
        let mask_tensor = Self::swap_axes_if_needed(mask_tensor, mask_coeff.into());
        let mask_tensor = mask_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &skip)?;
        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        let protos_tensor = Self::protos_to_hwc(protos_tensor, protos);

        // Slice combined detection into boxes and scores
        let boxes_view = det_tensor.slice(s![..4, ..]);
        let scores_view = det_tensor.slice(s![4.., ..]);

        decode_yolo_split_segdet_float(
            boxes_view,
            scores_view,
            mask_tensor,
            protos_tensor,
            self.score_threshold,
            self.iou_threshold,
            self.nms,
            output_boxes,
            output_masks,
        )
    }

    pub(super) fn decode_modelpack_det_split_float<D>(
        &self,
        outputs: &[ArrayViewD<D>],
        detection: &[configs::Detection],
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError>
    where
        D: AsPrimitive<f32>,
    {
        let new_detection = detection
            .iter()
            .map(|x| match &x.anchors {
                None => Err(DecoderError::InvalidConfig(
                    "ModelPack Split Detection missing anchors".to_string(),
                )),
                Some(a) => Ok(ModelPackDetectionConfig {
                    anchors: a.clone(),
                    quantization: None,
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let new_outputs = Self::match_outputs_to_detect(detection, outputs)?;
        let new_outputs = new_outputs
            .into_iter()
            .map(|x| x.slice(s![0, .., .., ..]))
            .collect::<Vec<_>>();

        decode_modelpack_split_float(
            &new_outputs,
            &new_detection,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    pub(super) fn decode_modelpack_seg_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        segmentation: &configs::Segmentation,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + AsPrimitive<u8> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (seg, _) = Self::find_outputs_with_shape(&segmentation.shape, outputs, &[])?;

        let seg = Self::swap_axes_if_needed(seg, segmentation.into());
        let seg = seg.slice(s![0, .., .., ..]);
        let u8_max = 255.0_f32.as_();
        let max = *seg.max().unwrap_or(&u8_max);
        let min = *seg.min().unwrap_or(&0.0_f32.as_());
        let seg = seg.mapv(|x| ((x - min) / (max - min) * u8_max).as_());
        output_masks.push(Segmentation {
            xmin: 0.0,
            ymin: 0.0,
            xmax: 1.0,
            ymax: 1.0,
            segmentation: seg,
        });
        Ok(())
    }

    pub(super) fn decode_modelpack_det_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;

        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., 0, ..]);

        let (scores_tensor, _) = Self::find_outputs_with_shape(&scores.shape, outputs, &[ind])?;
        let scores_tensor = Self::swap_axes_if_needed(scores_tensor, scores.into());
        let scores_tensor = scores_tensor.slice(s![0, .., ..]);

        decode_modelpack_float(
            boxes_tensor,
            scores_tensor,
            self.score_threshold,
            self.iou_threshold,
            output_boxes,
        );
        Ok(())
    }

    pub(super) fn decode_yolo_det_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Detection,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (boxes_tensor, _) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;

        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        decode_yolo_det_float(
            boxes_tensor,
            self.score_threshold,
            self.iou_threshold,
            self.nms,
            output_boxes,
        );
        Ok(())
    }

    pub(super) fn decode_yolo_segdet_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Detection,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;

        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &[ind])?;

        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        let protos_tensor = Self::protos_to_hwc(protos_tensor, protos);
        decode_yolo_segdet_float(
            boxes_tensor,
            protos_tensor,
            self.score_threshold,
            self.iou_threshold,
            self.nms,
            output_boxes,
            output_masks,
        )
    }

    pub(super) fn decode_yolo_split_det_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);

        let (scores_tensor, _) = Self::find_outputs_with_shape(&scores.shape, outputs, &[ind])?;

        let scores_tensor = Self::swap_axes_if_needed(scores_tensor, scores.into());
        let scores_tensor = scores_tensor.slice(s![0, .., ..]);

        decode_yolo_split_det_float(
            boxes_tensor,
            scores_tensor,
            self.score_threshold,
            self.iou_threshold,
            self.nms,
            output_boxes,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_yolo_split_segdet_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let mut skip = vec![];
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &skip)?;

        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (scores_tensor, ind) = Self::find_outputs_with_shape(&scores.shape, outputs, &skip)?;

        let scores_tensor = Self::swap_axes_if_needed(scores_tensor, scores.into());
        let scores_tensor = scores_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (mask_tensor, ind) = Self::find_outputs_with_shape(&mask_coeff.shape, outputs, &skip)?;
        let mask_tensor = Self::swap_axes_if_needed(mask_tensor, mask_coeff.into());
        let mask_tensor = mask_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &skip)?;
        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        let protos_tensor = Self::protos_to_hwc(protos_tensor, protos);
        decode_yolo_split_segdet_float(
            boxes_tensor,
            scores_tensor,
            mask_tensor,
            protos_tensor,
            self.score_threshold,
            self.iou_threshold,
            self.nms,
            output_boxes,
            output_masks,
        )
    }

    /// Decodes end-to-end YOLO detection outputs (post-NMS from model).
    ///
    /// Input shape: (1, N, 6+) where columns are [x1, y1, x2, y2, conf, class,
    /// ...] Boxes are output directly from model (may be normalized or
    /// pixel coords depending on config).
    pub(super) fn decode_yolo_end_to_end_det_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes_config: &configs::Detection,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (det_tensor, _) = Self::find_outputs_with_shape(&boxes_config.shape, outputs, &[])?;
        let det_tensor = Self::swap_axes_if_needed(det_tensor, boxes_config.into());
        let det_tensor = det_tensor.slice(s![0, .., ..]);

        crate::yolo::decode_yolo_end_to_end_det_float(
            det_tensor,
            self.score_threshold,
            output_boxes,
        )?;
        Ok(())
    }

    /// Decodes end-to-end YOLO detection + segmentation outputs (post-NMS from
    /// model).
    ///
    /// Input shapes:
    /// - detection: (1, N, 6 + num_protos) where columns are [x1, y1, x2, y2,
    ///   conf, class, mask_coeff_0, ..., mask_coeff_31]
    /// - protos: (1, proto_height, proto_width, num_protos)
    pub(super) fn decode_yolo_end_to_end_segdet_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes_config: &configs::Detection,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        if outputs.len() < 2 {
            return Err(DecoderError::InvalidShape(
                "End-to-end segdet requires detection and protos outputs".to_string(),
            ));
        }

        let (det_tensor, det_ind) =
            Self::find_outputs_with_shape(&boxes_config.shape, outputs, &[])?;
        let det_tensor = Self::swap_axes_if_needed(det_tensor, boxes_config.into());
        let det_tensor = det_tensor.slice(s![0, .., ..]);

        let (protos_tensor, _) =
            Self::find_outputs_with_shape(&protos_config.shape, outputs, &[det_ind])?;
        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos_config.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        let protos_tensor = Self::protos_to_hwc(protos_tensor, protos_config);

        crate::yolo::decode_yolo_end_to_end_segdet_float(
            det_tensor,
            protos_tensor,
            self.score_threshold,
            output_boxes,
            output_masks,
        )?;
        Ok(())
    }

    /// Decodes monolithic end-to-end YOLO detection from quantized tensors.
    /// Dequantizes then delegates to the float decode path.
    pub(super) fn decode_yolo_end_to_end_det_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes_config: &configs::Detection,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError> {
        let (det_tensor, _) =
            Self::find_outputs_with_shape_quantized(&boxes_config.shape, outputs, &[])?;
        let quant = boxes_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        with_quantized!(det_tensor, d, {
            let d = Self::swap_axes_if_needed(d, boxes_config.into());
            let d = d.slice(s![0, .., ..]);
            let dequant = d.map(|v| {
                let val: f32 = v.as_();
                (val - quant.zero_point as f32) * quant.scale
            });
            crate::yolo::decode_yolo_end_to_end_det_float(
                dequant.view(),
                self.score_threshold,
                output_boxes,
            )?;
        });
        Ok(())
    }

    /// Decodes monolithic end-to-end YOLO seg detection from quantized tensors.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_yolo_end_to_end_segdet_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes_config: &configs::Detection,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError> {
        let (det_tensor, det_ind) =
            Self::find_outputs_with_shape_quantized(&boxes_config.shape, outputs, &[])?;
        let (protos_tensor, _) =
            Self::find_outputs_with_shape_quantized(&protos_config.shape, outputs, &[det_ind])?;

        let quant_det = boxes_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = protos_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        let dequant_d = dequant_3d!(det_tensor, boxes_config, quant_det);
        let dequant_p = dequant_4d!(protos_tensor, protos_config, quant_protos);

        crate::yolo::decode_yolo_end_to_end_segdet_float(
            dequant_d.view(),
            dequant_p.view(),
            self.score_threshold,
            output_boxes,
            output_masks,
        )?;
        Ok(())
    }

    /// Decodes split end-to-end YOLO detection from float tensors.
    pub(super) fn decode_yolo_split_end_to_end_det_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes_config: &configs::Boxes,
        scores_config: &configs::Scores,
        classes_config: &configs::Classes,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let mut skip = vec![];
        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape(&boxes_config.shape, outputs, &skip)?;
        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes_config.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (scores_tensor, ind) =
            Self::find_outputs_with_shape(&scores_config.shape, outputs, &skip)?;
        let scores_tensor = Self::swap_axes_if_needed(scores_tensor, scores_config.into());
        let scores_tensor = scores_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (classes_tensor, _) =
            Self::find_outputs_with_shape(&classes_config.shape, outputs, &skip)?;
        let classes_tensor = Self::swap_axes_if_needed(classes_tensor, classes_config.into());
        let classes_tensor = classes_tensor.slice(s![0, .., ..]);

        crate::yolo::decode_yolo_split_end_to_end_det_float(
            boxes_tensor,
            scores_tensor,
            classes_tensor,
            self.score_threshold,
            output_boxes,
        )?;
        Ok(())
    }

    /// Decodes split end-to-end YOLO seg detection from float tensors.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_yolo_split_end_to_end_segdet_float<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes_config: &configs::Boxes,
        scores_config: &configs::Scores,
        classes_config: &configs::Classes,
        mask_coeff_config: &configs::MaskCoefficients,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let mut skip = vec![];
        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape(&boxes_config.shape, outputs, &skip)?;
        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes_config.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (scores_tensor, ind) =
            Self::find_outputs_with_shape(&scores_config.shape, outputs, &skip)?;
        let scores_tensor = Self::swap_axes_if_needed(scores_tensor, scores_config.into());
        let scores_tensor = scores_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (classes_tensor, ind) =
            Self::find_outputs_with_shape(&classes_config.shape, outputs, &skip)?;
        let classes_tensor = Self::swap_axes_if_needed(classes_tensor, classes_config.into());
        let classes_tensor = classes_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (mask_tensor, ind) =
            Self::find_outputs_with_shape(&mask_coeff_config.shape, outputs, &skip)?;
        let mask_tensor = Self::swap_axes_if_needed(mask_tensor, mask_coeff_config.into());
        let mask_tensor = mask_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (protos_tensor, _) =
            Self::find_outputs_with_shape(&protos_config.shape, outputs, &skip)?;
        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos_config.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        let protos_tensor = Self::protos_to_hwc(protos_tensor, protos_config);

        crate::yolo::decode_yolo_split_end_to_end_segdet_float(
            boxes_tensor,
            scores_tensor,
            classes_tensor,
            mask_tensor,
            protos_tensor,
            self.score_threshold,
            output_boxes,
            output_masks,
        )?;
        Ok(())
    }

    /// Decodes split end-to-end YOLO detection from quantized tensors.
    /// Dequantizes each tensor then delegates to the float decode path.
    pub(super) fn decode_yolo_split_end_to_end_det_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes_config: &configs::Boxes,
        scores_config: &configs::Scores,
        classes_config: &configs::Classes,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<(), DecoderError> {
        let mut skip = vec![];
        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&boxes_config.shape, outputs, &skip)?;
        skip.push(ind);
        let (scores_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&scores_config.shape, outputs, &skip)?;
        skip.push(ind);
        let (classes_tensor, _) =
            Self::find_outputs_with_shape_quantized(&classes_config.shape, outputs, &skip)?;

        let quant_boxes = boxes_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_scores = scores_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_classes = classes_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        let dequant_b = dequant_3d!(boxes_tensor, boxes_config, quant_boxes);
        let dequant_s = dequant_3d!(scores_tensor, scores_config, quant_scores);
        let dequant_c = dequant_3d!(classes_tensor, classes_config, quant_classes);

        crate::yolo::decode_yolo_split_end_to_end_det_float(
            dequant_b.view(),
            dequant_s.view(),
            dequant_c.view(),
            self.score_threshold,
            output_boxes,
        )?;
        Ok(())
    }

    /// Decodes split end-to-end YOLO seg detection from quantized tensors.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_yolo_split_end_to_end_segdet_quantized(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes_config: &configs::Boxes,
        scores_config: &configs::Scores,
        classes_config: &configs::Classes,
        mask_coeff_config: &configs::MaskCoefficients,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
    ) -> Result<(), DecoderError> {
        let mut skip = vec![];
        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&boxes_config.shape, outputs, &skip)?;
        skip.push(ind);
        let (scores_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&scores_config.shape, outputs, &skip)?;
        skip.push(ind);
        let (classes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&classes_config.shape, outputs, &skip)?;
        skip.push(ind);
        let (mask_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&mask_coeff_config.shape, outputs, &skip)?;
        skip.push(ind);
        let (protos_tensor, _) =
            Self::find_outputs_with_shape_quantized(&protos_config.shape, outputs, &skip)?;

        let quant_boxes = boxes_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_scores = scores_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_classes = classes_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_masks = mask_coeff_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = protos_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        let dequant_b = dequant_3d!(boxes_tensor, boxes_config, quant_boxes);
        let dequant_s = dequant_3d!(scores_tensor, scores_config, quant_scores);
        let dequant_c = dequant_3d!(classes_tensor, classes_config, quant_classes);
        let dequant_m = dequant_3d!(mask_tensor, mask_coeff_config, quant_masks);
        let dequant_p = dequant_4d!(protos_tensor, protos_config, quant_protos);

        crate::yolo::decode_yolo_split_end_to_end_segdet_float(
            dequant_b.view(),
            dequant_s.view(),
            dequant_c.view(),
            dequant_m.view(),
            dequant_p.view(),
            self.score_threshold,
            output_boxes,
            output_masks,
        )?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Proto-extraction private helpers (mirror the non-proto variants)
    // ------------------------------------------------------------------

    pub(super) fn decode_yolo_segdet_quantized_proto(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Detection,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<ProtoData, DecoderError> {
        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&boxes.shape, outputs, &[])?;
        let (protos_tensor, _) =
            Self::find_outputs_with_shape_quantized(&protos.shape, outputs, &[ind])?;

        let quant_boxes = boxes
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = protos
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        let proto = with_quantized!(boxes_tensor, b, {
            with_quantized!(protos_tensor, p, {
                let box_tensor = Self::swap_axes_if_needed(b, boxes.into());
                let box_tensor = box_tensor.slice(s![0, .., ..]);

                let protos_tensor = Self::swap_axes_if_needed(p, protos.into());
                let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
                let protos_tensor = Self::protos_to_hwc(protos_tensor, protos);
                crate::yolo::impl_yolo_segdet_quant_proto::<XYWH, _, _>(
                    (box_tensor, quant_boxes),
                    (protos_tensor, quant_protos),
                    self.score_threshold,
                    self.iou_threshold,
                    self.nms,
                    output_boxes,
                )
            })
        });
        Ok(proto)
    }

    pub(super) fn decode_yolo_segdet_float_proto<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Detection,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<ProtoData, DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;
        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &[ind])?;
        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        let protos_tensor = Self::protos_to_hwc(protos_tensor, protos);

        Ok(crate::yolo::impl_yolo_segdet_float_proto::<XYWH, _, _>(
            boxes_tensor,
            protos_tensor,
            self.score_threshold,
            self.iou_threshold,
            self.nms,
            output_boxes,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_yolo_split_segdet_quantized_proto(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<ProtoData, DecoderError> {
        let quant_boxes = boxes
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_scores = scores
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_masks = mask_coeff
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = protos
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        let mut skip = vec![];

        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&boxes.shape, outputs, &skip)?;
        skip.push(ind);

        let (scores_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&scores.shape, outputs, &skip)?;
        skip.push(ind);

        let (mask_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&mask_coeff.shape, outputs, &skip)?;
        skip.push(ind);

        let (protos_tensor, _) =
            Self::find_outputs_with_shape_quantized(&protos.shape, outputs, &skip)?;

        // Phase 1: boxes + scores (2-level nesting, 36 paths).
        let det_indices = with_quantized!(boxes_tensor, b, {
            with_quantized!(scores_tensor, s, {
                let boxes_tensor = Self::swap_axes_if_needed(b, boxes.into());
                let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
                let boxes_tensor = boxes_tensor.reversed_axes();

                let scores_tensor = Self::swap_axes_if_needed(s, scores.into());
                let scores_tensor = scores_tensor.slice(s![0, .., ..]);
                let scores_tensor = scores_tensor.reversed_axes();

                impl_yolo_split_segdet_quant_get_boxes::<XYWH, _, _>(
                    (boxes_tensor, quant_boxes),
                    (scores_tensor, quant_scores),
                    self.score_threshold,
                    self.iou_threshold,
                    self.nms,
                    output_boxes.capacity(),
                )
            })
        });

        // Phase 2: masks + protos (2-level nesting, 36 paths).
        let proto = with_quantized!(mask_tensor, m, {
            with_quantized!(protos_tensor, p, {
                let mask_tensor = Self::swap_axes_if_needed(m, mask_coeff.into());
                let mask_tensor = mask_tensor.slice(s![0, .., ..]);
                let mask_tensor = mask_tensor.reversed_axes();

                let protos_tensor = Self::swap_axes_if_needed(p, protos.into());
                let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
                let protos_tensor = Self::protos_to_hwc(protos_tensor, protos);

                crate::yolo::extract_proto_data_quant(
                    det_indices,
                    mask_tensor,
                    quant_masks,
                    protos_tensor,
                    quant_protos,
                    output_boxes,
                )
            })
        });
        Ok(proto)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_yolo_split_segdet_float_proto<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<ProtoData, DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let mut skip = vec![];
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &skip)?;
        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (scores_tensor, ind) = Self::find_outputs_with_shape(&scores.shape, outputs, &skip)?;
        let scores_tensor = Self::swap_axes_if_needed(scores_tensor, scores.into());
        let scores_tensor = scores_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (mask_tensor, ind) = Self::find_outputs_with_shape(&mask_coeff.shape, outputs, &skip)?;
        let mask_tensor = Self::swap_axes_if_needed(mask_tensor, mask_coeff.into());
        let mask_tensor = mask_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &skip)?;
        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        let protos_tensor = Self::protos_to_hwc(protos_tensor, protos);

        Ok(crate::yolo::impl_yolo_split_segdet_float_proto::<
            XYWH,
            _,
            _,
            _,
            _,
        >(
            boxes_tensor,
            scores_tensor,
            mask_tensor,
            protos_tensor,
            self.score_threshold,
            self.iou_threshold,
            self.nms,
            output_boxes,
        ))
    }

    /// Decode 2-way split proto (quantized): combined detection + separate
    /// mask_coeff + protos → ProtoData.
    pub(super) fn decode_yolo_segdet_2way_quantized_proto(
        &self,
        outputs: &[ArrayViewDQuantized],
        detection: &configs::Detection,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<ProtoData, DecoderError> {
        let quant_det = detection
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_masks = mask_coeff
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = protos
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        let mut skip = vec![];

        let (det_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&detection.shape, outputs, &skip)?;
        skip.push(ind);

        let (mask_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&mask_coeff.shape, outputs, &skip)?;
        skip.push(ind);

        let (protos_tensor, _) =
            Self::find_outputs_with_shape_quantized(&protos.shape, outputs, &skip)?;

        // Phase 1: Slice detection into boxes + scores, run NMS.
        let det_indices = with_quantized!(det_tensor, d, {
            let det = Self::swap_axes_if_needed(d, detection.into());
            let det = det.slice(s![0, .., ..]);
            let boxes_view = det.slice(s![..4, ..]);
            let scores_view = det.slice(s![4.., ..]);
            impl_yolo_split_segdet_quant_get_boxes::<XYWH, _, _>(
                (boxes_view, quant_det),
                (scores_view, quant_det),
                self.score_threshold,
                self.iou_threshold,
                self.nms,
                output_boxes.capacity(),
            )
        });

        // Phase 2: Extract proto data from separate mask_coeff + protos.
        let proto = with_quantized!(mask_tensor, m, {
            with_quantized!(protos_tensor, p, {
                let mask_tensor = Self::swap_axes_if_needed(m, mask_coeff.into());
                let mask_tensor = mask_tensor.slice(s![0, .., ..]);
                let mask_tensor = mask_tensor.reversed_axes();

                let protos_tensor = Self::swap_axes_if_needed(p, protos.into());
                let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
                let protos_tensor = Self::protos_to_hwc(protos_tensor, protos);

                crate::yolo::extract_proto_data_quant(
                    det_indices,
                    mask_tensor,
                    quant_masks,
                    protos_tensor,
                    quant_protos,
                    output_boxes,
                )
            })
        });
        Ok(proto)
    }

    /// Decode 2-way split proto (float): combined detection + separate
    /// mask_coeff + protos → ProtoData.
    pub(super) fn decode_yolo_segdet_2way_float_proto<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        detection: &configs::Detection,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<ProtoData, DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let mut skip = vec![];
        let (det_tensor, ind) = Self::find_outputs_with_shape(&detection.shape, outputs, &skip)?;
        let det_tensor = Self::swap_axes_if_needed(det_tensor, detection.into());
        let det_tensor = det_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (mask_tensor, ind) = Self::find_outputs_with_shape(&mask_coeff.shape, outputs, &skip)?;
        let mask_tensor = Self::swap_axes_if_needed(mask_tensor, mask_coeff.into());
        let mask_tensor = mask_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &skip)?;
        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        let protos_tensor = Self::protos_to_hwc(protos_tensor, protos);

        // Slice combined detection into boxes and scores
        let boxes_view = det_tensor.slice(s![..4, ..]);
        let scores_view = det_tensor.slice(s![4.., ..]);

        Ok(crate::yolo::impl_yolo_split_segdet_float_proto::<
            XYWH,
            _,
            _,
            _,
            _,
        >(
            boxes_view,
            scores_view,
            mask_tensor,
            protos_tensor,
            self.score_threshold,
            self.iou_threshold,
            self.nms,
            output_boxes,
        ))
    }

    pub(super) fn decode_yolo_end_to_end_segdet_float_proto<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes_config: &configs::Detection,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<ProtoData, DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        if outputs.len() < 2 {
            return Err(DecoderError::InvalidShape(
                "End-to-end segdet requires detection and protos outputs".to_string(),
            ));
        }

        let (det_tensor, det_ind) =
            Self::find_outputs_with_shape(&boxes_config.shape, outputs, &[])?;
        let det_tensor = Self::swap_axes_if_needed(det_tensor, boxes_config.into());
        let det_tensor = det_tensor.slice(s![0, .., ..]);

        let (protos_tensor, _) =
            Self::find_outputs_with_shape(&protos_config.shape, outputs, &[det_ind])?;
        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos_config.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        let protos_tensor = Self::protos_to_hwc(protos_tensor, protos_config);

        crate::yolo::decode_yolo_end_to_end_segdet_float_proto(
            det_tensor,
            protos_tensor,
            self.score_threshold,
            output_boxes,
        )
    }

    pub(super) fn decode_yolo_end_to_end_segdet_quantized_proto(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes_config: &configs::Detection,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<ProtoData, DecoderError> {
        let (det_tensor, det_ind) =
            Self::find_outputs_with_shape_quantized(&boxes_config.shape, outputs, &[])?;
        let (protos_tensor, _) =
            Self::find_outputs_with_shape_quantized(&protos_config.shape, outputs, &[det_ind])?;

        let quant_det = boxes_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = protos_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        let dequant_d = dequant_3d!(det_tensor, boxes_config, quant_det);
        let dequant_p = dequant_4d!(protos_tensor, protos_config, quant_protos);

        let proto = crate::yolo::decode_yolo_end_to_end_segdet_float_proto(
            dequant_d.view(),
            dequant_p.view(),
            self.score_threshold,
            output_boxes,
        )?;
        Ok(proto)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_yolo_split_end_to_end_segdet_float_proto<T>(
        &self,
        outputs: &[ArrayViewD<T>],
        boxes_config: &configs::Boxes,
        scores_config: &configs::Scores,
        classes_config: &configs::Classes,
        mask_coeff_config: &configs::MaskCoefficients,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<ProtoData, DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        let mut skip = vec![];
        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape(&boxes_config.shape, outputs, &skip)?;
        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes_config.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (scores_tensor, ind) =
            Self::find_outputs_with_shape(&scores_config.shape, outputs, &skip)?;
        let scores_tensor = Self::swap_axes_if_needed(scores_tensor, scores_config.into());
        let scores_tensor = scores_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (classes_tensor, ind) =
            Self::find_outputs_with_shape(&classes_config.shape, outputs, &skip)?;
        let classes_tensor = Self::swap_axes_if_needed(classes_tensor, classes_config.into());
        let classes_tensor = classes_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (mask_tensor, ind) =
            Self::find_outputs_with_shape(&mask_coeff_config.shape, outputs, &skip)?;
        let mask_tensor = Self::swap_axes_if_needed(mask_tensor, mask_coeff_config.into());
        let mask_tensor = mask_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (protos_tensor, _) =
            Self::find_outputs_with_shape(&protos_config.shape, outputs, &skip)?;
        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos_config.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        let protos_tensor = Self::protos_to_hwc(protos_tensor, protos_config);

        crate::yolo::decode_yolo_split_end_to_end_segdet_float_proto(
            boxes_tensor,
            scores_tensor,
            classes_tensor,
            mask_tensor,
            protos_tensor,
            self.score_threshold,
            output_boxes,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_yolo_split_end_to_end_segdet_quantized_proto(
        &self,
        outputs: &[ArrayViewDQuantized],
        boxes_config: &configs::Boxes,
        scores_config: &configs::Scores,
        classes_config: &configs::Classes,
        mask_coeff_config: &configs::MaskCoefficients,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
    ) -> Result<ProtoData, DecoderError> {
        let mut skip = vec![];
        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&boxes_config.shape, outputs, &skip)?;
        skip.push(ind);
        let (scores_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&scores_config.shape, outputs, &skip)?;
        skip.push(ind);
        let (classes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&classes_config.shape, outputs, &skip)?;
        skip.push(ind);
        let (mask_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&mask_coeff_config.shape, outputs, &skip)?;
        skip.push(ind);
        let (protos_tensor, _) =
            Self::find_outputs_with_shape_quantized(&protos_config.shape, outputs, &skip)?;

        let quant_boxes = boxes_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_scores = scores_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_classes = classes_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_masks = mask_coeff_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = protos_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        let dequant_b = dequant_3d!(boxes_tensor, boxes_config, quant_boxes);
        let dequant_s = dequant_3d!(scores_tensor, scores_config, quant_scores);
        let dequant_c = dequant_3d!(classes_tensor, classes_config, quant_classes);
        let dequant_m = dequant_3d!(mask_tensor, mask_coeff_config, quant_masks);
        let dequant_p = dequant_4d!(protos_tensor, protos_config, quant_protos);

        crate::yolo::decode_yolo_split_end_to_end_segdet_float_proto(
            dequant_b.view(),
            dequant_s.view(),
            dequant_c.view(),
            dequant_m.view(),
            dequant_p.view(),
            self.score_threshold,
            output_boxes,
        )
    }
}

#[cfg(feature = "tracker")]
impl edgefirst_tracker::DetectionBox for DetectBox {
    fn bbox(&self) -> [f32; 4] {
        self.bbox.into()
    }
    fn score(&self) -> f32 {
        self.score
    }
    fn label(&self) -> usize {
        self.label
    }
}

#[cfg(feature = "tracker")]
macro_rules! process_tracked_yolo_segmentation {
    ($self:expr, $tracker:expr, $timestamp:expr, $outputs:expr, $boxes:expr, $protos:expr, $output_boxes:expr, $output_masks:expr, $output_tracks:expr, $mask_body:expr) => {{
        // 1. Find and extract quantized tensors
        let (boxes_tensor, ind) =
            Decoder::find_outputs_with_shape_quantized(&$boxes.shape, $outputs, &[])?;
        let (protos_tensor, _) =
            Decoder::find_outputs_with_shape_quantized(&$protos.shape, $outputs, &[ind])?;

        // 2. Prepare quantization parameters
        let quant_boxes = $boxes
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = $protos
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        // 3. Nested quantized processing
        let proto = with_quantized!(boxes_tensor, b, {
            with_quantized!(protos_tensor, p, {
                // Slice and reshape boxes
                let box_tensor = Decoder::swap_axes_if_needed(b, $boxes.into());
                let box_tensor = box_tensor.slice(s![0, .., ..]);

                // Slice and reshape protos
                let protos_tensor = Decoder::swap_axes_if_needed(p, $protos.into());
                let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
                let protos_tensor = Decoder::protos_to_hwc(protos_tensor, $protos);

                let num_protos = protos_tensor.dim().2;

                // Split boxes/scores/masks
                let (boxes_tensor, scores_tensor, mask_tensor) =
                    postprocess_yolo_seg(&box_tensor, num_protos);

                // Run NMS and box extraction
                let detect_boxes = impl_yolo_split_segdet_quant_get_boxes::<XYWH, _, _>(
                    (boxes_tensor, quant_boxes),
                    (scores_tensor, quant_boxes),
                    $self.score_threshold,
                    $self.iou_threshold,
                    $self.nms,
                    $output_boxes.capacity(),
                );

                // Update tracker logic
                let (new_boxes, old_boxes) = Decoder::update_tracker_yolo_segdet(
                    $tracker,
                    $timestamp,
                    detect_boxes,
                    $output_tracks,
                );

                // Extract mask data
                let mask_data = $mask_body(
                    new_boxes,
                    mask_tensor,
                    quant_boxes,
                    protos_tensor,
                    quant_protos,
                    $output_boxes,
                    $output_masks,
                );

                $output_boxes.extend(old_boxes);

                mask_data
            })
        });

        proto
    }};
}

#[cfg(feature = "tracker")]
macro_rules! process_tracked_yolo_segmentation_split {
    (
        $self:expr,
        $tracker:expr,
        $timestamp:expr,
        $outputs:expr,
        $boxes:expr,
        $scores:expr,
        $mask_coeff:expr,
        $protos:expr,
        $output_boxes:expr,
        $output_masks:expr,
        $output_tracks:expr,
        $mask_body:expr
    ) => {{
        let quant_boxes = $boxes
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_scores = $scores
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_masks = $mask_coeff
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = $protos
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        let mut skip = vec![];

        let (boxes_tensor, ind) =
            Decoder::find_outputs_with_shape_quantized(&$boxes.shape, $outputs, &skip)?;
        skip.push(ind);

        let (scores_tensor, ind) =
            Decoder::find_outputs_with_shape_quantized(&$scores.shape, $outputs, &skip)?;
        skip.push(ind);

        let (mask_tensor, ind) =
            Decoder::find_outputs_with_shape_quantized(&$mask_coeff.shape, $outputs, &skip)?;
        skip.push(ind);

        let (protos_tensor, _) =
            Decoder::find_outputs_with_shape_quantized(&$protos.shape, $outputs, &skip)?;

        let boxes = with_quantized!(boxes_tensor, b, {
            with_quantized!(scores_tensor, s, {
                let boxes_tensor = Decoder::swap_axes_if_needed(b, $boxes.into());
                let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
                let boxes_tensor = boxes_tensor.reversed_axes();

                let scores_tensor = Decoder::swap_axes_if_needed(s, $scores.into());
                let scores_tensor = scores_tensor.slice(s![0, .., ..]);
                let scores_tensor = scores_tensor.reversed_axes();

                impl_yolo_split_segdet_quant_get_boxes::<XYWH, _, _>(
                    (boxes_tensor, quant_boxes),
                    (scores_tensor, quant_scores),
                    $self.score_threshold,
                    $self.iou_threshold,
                    $self.nms,
                    $output_boxes.capacity(),
                )
            })
        });

        let (new_boxes, old_boxes) =
            Decoder::update_tracker_yolo_segdet($tracker, $timestamp, boxes, $output_tracks);

        let mask_data = with_quantized!(mask_tensor, m, {
            with_quantized!(protos_tensor, p, {
                let mask_tensor = Decoder::swap_axes_if_needed(m, $mask_coeff.into());
                let mask_tensor = mask_tensor.slice(s![0, .., ..]);
                let mask_tensor = mask_tensor.reversed_axes();

                let protos_tensor = Decoder::swap_axes_if_needed(p, $protos.into());
                let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
                let protos_tensor = Decoder::protos_to_hwc(protos_tensor, $protos);

                // Extract mask data
                $mask_body(
                    new_boxes,
                    mask_tensor,
                    quant_masks,
                    protos_tensor,
                    quant_protos,
                    $output_boxes,
                    $output_masks,
                )
            })
        });
        $output_boxes.extend(old_boxes);
        mask_data
    }};
}

#[cfg(feature = "tracker")]
use edgefirst_tracker::TrackInfo;

#[cfg(feature = "tracker")]
use crate::yolo::postprocess_yolo_seg;

#[cfg(feature = "tracker")]
impl Decoder {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_segdet_quantized<
        TR: edgefirst_tracker::Tracker<DetectBox>,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Detection,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<(), DecoderError> {
        process_tracked_yolo_segmentation!(
            self,
            tracker,
            timestamp,
            outputs,
            boxes,
            protos,
            output_boxes,
            output_masks,
            output_tracks,
            |new_boxes, mask_t, quant_masks, protos_t, quant_protos, output_boxes, output_masks| {
                impl_yolo_split_segdet_quant_process_masks::<_, _>(
                    new_boxes,
                    (mask_t, quant_masks),
                    (protos_t, quant_protos),
                    output_boxes,
                    output_masks,
                )
            }
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_split_segdet_quantized<
        TR: edgefirst_tracker::Tracker<DetectBox>,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<(), DecoderError> {
        process_tracked_yolo_segmentation_split!(
            self,
            tracker,
            timestamp,
            outputs,
            boxes,
            scores,
            mask_coeff,
            protos,
            output_boxes,
            output_masks,
            output_tracks,
            |new_boxes, mask_t, quant_masks, protos_t, quant_protos, output_boxes, output_masks| {
                impl_yolo_split_segdet_quant_process_masks::<_, _>(
                    new_boxes,
                    (mask_t, quant_masks),
                    (protos_t, quant_protos),
                    output_boxes,
                    output_masks,
                )
            }
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn process_tracked_yolo_segdet_float<T, TR: edgefirst_tracker::Tracker<DetectBox>, M>(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Detection,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo>,
        mask_fn: impl FnOnce(
            Vec<(DetectBox, usize)>,
            ArrayView2<T>,
            ArrayView3<T>,
            &mut Vec<DetectBox>,
            &mut Vec<Segmentation>,
        ) -> M,
    ) -> Result<M, DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        use crate::yolo::impl_yolo_segdet_get_boxes;

        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &[])?;

        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &[ind])?;

        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        let protos_tensor = Self::protos_to_hwc(protos_tensor, protos);

        let num_protos = protos_tensor.dim().2;
        let (boxes_tensor, scores_tensor, mask_tensor) =
            postprocess_yolo_seg(&boxes_tensor, num_protos);
        let boxes = impl_yolo_segdet_get_boxes::<XYWH, _, _>(
            boxes_tensor,
            scores_tensor,
            self.score_threshold,
            self.iou_threshold,
            self.nms,
            output_boxes.capacity(),
        );

        let (new_boxes, old_boxes) =
            Self::update_tracker_yolo_segdet(tracker, timestamp, boxes, output_tracks);

        let mask_data = mask_fn(
            new_boxes,
            mask_tensor,
            protos_tensor,
            output_boxes,
            output_masks,
        );

        output_boxes.extend(old_boxes);

        Ok(mask_data)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_segdet_float<TR: edgefirst_tracker::Tracker<DetectBox>, T>(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Detection,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        use crate::yolo::impl_yolo_split_segdet_process_masks;

        self.process_tracked_yolo_segdet_float(
            tracker,
            timestamp,
            outputs,
            boxes,
            protos,
            output_boxes,
            output_masks,
            output_tracks,
            impl_yolo_split_segdet_process_masks,
        )
        .flatten()?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn process_tracked_yolo_segdet_split_float<T, TR: edgefirst_tracker::Tracker<DetectBox>, M>(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo>,
        mask_fn: impl FnOnce(
            Vec<(DetectBox, usize)>,
            ArrayView2<T>,
            ArrayView3<T>,
            &mut Vec<DetectBox>,
            &mut Vec<Segmentation>,
        ) -> M,
    ) -> Result<M, DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        use crate::yolo::{impl_yolo_segdet_get_boxes, postprocess_yolo_split_segdet};

        let mut skip = vec![];
        let (boxes_tensor, ind) = Self::find_outputs_with_shape(&boxes.shape, outputs, &skip)?;

        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (scores_tensor, ind) = Self::find_outputs_with_shape(&scores.shape, outputs, &skip)?;

        let scores_tensor = Self::swap_axes_if_needed(scores_tensor, scores.into());
        let scores_tensor = scores_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (mask_tensor, ind) = Self::find_outputs_with_shape(&mask_coeff.shape, outputs, &skip)?;
        let mask_tensor = Self::swap_axes_if_needed(mask_tensor, mask_coeff.into());
        let mask_tensor = mask_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (protos_tensor, _) = Self::find_outputs_with_shape(&protos.shape, outputs, &skip)?;
        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        let protos_tensor = Self::protos_to_hwc(protos_tensor, protos);

        let (boxes_tensor, scores_tensor, mask_tensor) =
            postprocess_yolo_split_segdet(boxes_tensor, scores_tensor, mask_tensor);
        let boxes = impl_yolo_segdet_get_boxes::<XYWH, _, _>(
            boxes_tensor,
            scores_tensor,
            self.score_threshold,
            self.iou_threshold,
            self.nms,
            output_boxes.capacity(),
        );

        let (new_boxes, old_boxes) =
            Self::update_tracker_yolo_segdet(tracker, timestamp, boxes, output_tracks);

        let mask_data = mask_fn(
            new_boxes,
            mask_tensor,
            protos_tensor,
            output_boxes,
            output_masks,
        );

        output_boxes.extend(old_boxes);

        Ok(mask_data)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_split_segdet_float<
        TR: edgefirst_tracker::Tracker<DetectBox>,
        T,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        use crate::yolo::impl_yolo_split_segdet_process_masks;

        self.process_tracked_yolo_segdet_split_float(
            tracker,
            timestamp,
            outputs,
            boxes,
            scores,
            mask_coeff,
            protos,
            output_boxes,
            output_masks,
            output_tracks,
            impl_yolo_split_segdet_process_masks,
        )
        .flatten()
    }

    #[allow(clippy::too_many_arguments)]
    fn process_tracked_yolo_end_to_end_segdet_float<
        T,
        TR: edgefirst_tracker::Tracker<DetectBox>,
        M,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewD<T>],
        boxes_config: &configs::Detection,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo>,
        mask_fn: impl FnOnce(
            Vec<(DetectBox, usize)>,
            ArrayView2<T>,
            ArrayView3<T>,
            &mut Vec<DetectBox>,
            &mut Vec<Segmentation>,
        ) -> M,
    ) -> Result<M, DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        use crate::{
            yolo::impl_yolo_end_to_end_segdet_get_boxes, yolo::postprocess_yolo_end_to_end_segdet,
            XYXY,
        };

        if outputs.len() < 2 {
            return Err(DecoderError::InvalidShape(
                "End-to-end segdet requires detection and protos outputs".to_string(),
            ));
        }

        let (det_tensor, det_ind) =
            Self::find_outputs_with_shape(&boxes_config.shape, outputs, &[])?;
        let det_tensor = Self::swap_axes_if_needed(det_tensor, boxes_config.into());
        let det_tensor = det_tensor.slice(s![0, .., ..]);

        let (protos_tensor, _) =
            Self::find_outputs_with_shape(&protos_config.shape, outputs, &[det_ind])?;
        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos_config.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        let protos_tensor = Self::protos_to_hwc(protos_tensor, protos_config);

        let (boxes, scores, classes, mask_coeff) =
            postprocess_yolo_end_to_end_segdet(&det_tensor, protos_tensor.dim().2)?;
        let boxes = impl_yolo_end_to_end_segdet_get_boxes::<XYXY, _, _, _>(
            boxes,
            scores,
            classes,
            self.score_threshold,
            output_boxes.capacity(),
        );

        let (new_boxes, old_boxes) =
            Self::update_tracker_yolo_segdet(tracker, timestamp, boxes, output_tracks);

        // No NMS — model output is already post-NMS

        let mask_data = mask_fn(
            new_boxes,
            mask_coeff,
            protos_tensor,
            output_boxes,
            output_masks,
        );
        output_boxes.extend(old_boxes);
        Ok(mask_data)
    }

    /// Decodes end-to-end YOLO detection + segmentation outputs (post-NMS from
    /// model).
    ///
    /// Input shapes:
    /// - detection: (1, N, 6 + num_protos) where columns are [x1, y1, x2, y2,
    ///   conf, class, mask_coeff_0, ..., mask_coeff_31]
    /// - protos: (1, proto_height, proto_width, num_protos)
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_end_to_end_segdet_float<
        TR: edgefirst_tracker::Tracker<DetectBox>,
        T,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewD<T>],
        boxes_config: &configs::Detection,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        use crate::yolo::impl_yolo_split_segdet_process_masks;

        self.process_tracked_yolo_end_to_end_segdet_float(
            tracker,
            timestamp,
            outputs,
            boxes_config,
            protos_config,
            output_boxes,
            output_masks,
            output_tracks,
            impl_yolo_split_segdet_process_masks,
        )
        .flatten()
    }

    #[allow(clippy::too_many_arguments)]
    fn process_tracked_yolo_end_to_end_segdet_quantized<
        TR: edgefirst_tracker::Tracker<DetectBox>,
        M,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewDQuantized],
        boxes_config: &configs::Detection,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo>,
        mask_fn: impl FnOnce(
            Vec<(DetectBox, usize)>,
            ArrayView2<f32>,
            ArrayView3<f32>,
            &mut Vec<DetectBox>,
            &mut Vec<Segmentation>,
        ) -> M,
    ) -> Result<M, DecoderError> {
        use crate::{
            yolo::impl_yolo_end_to_end_segdet_get_boxes, yolo::postprocess_yolo_end_to_end_segdet,
            XYXY,
        };

        let (det_tensor, det_ind) =
            Self::find_outputs_with_shape_quantized(&boxes_config.shape, outputs, &[])?;
        let (protos_tensor, _) =
            Self::find_outputs_with_shape_quantized(&protos_config.shape, outputs, &[det_ind])?;

        let quant_det = boxes_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = protos_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        let dequant_d = dequant_3d!(det_tensor, boxes_config, quant_det);
        let dequant_p = dequant_4d!(protos_tensor, protos_config, quant_protos);

        let det_tensor = dequant_d.view();
        let protos_tensor = dequant_p.view();
        let (boxes, scores, classes, mask_coeff) =
            postprocess_yolo_end_to_end_segdet(&det_tensor, protos_tensor.shape()[2])?;
        let boxes = impl_yolo_end_to_end_segdet_get_boxes::<XYXY, _, _, _>(
            boxes,
            scores,
            classes,
            self.score_threshold,
            output_boxes.capacity(),
        );

        let (new_boxes, old_boxes) =
            Self::update_tracker_yolo_segdet(tracker, timestamp, boxes, output_tracks);

        // No NMS — model output is already post-NMS

        let mask_data = mask_fn(
            new_boxes,
            mask_coeff,
            protos_tensor,
            output_boxes,
            output_masks,
        );
        output_boxes.extend(old_boxes);

        Ok(mask_data)
    }

    /// Decodes monolithic end-to-end YOLO seg detection from quantized tensors.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_end_to_end_segdet_quantized<
        TR: edgefirst_tracker::Tracker<DetectBox>,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewDQuantized],
        boxes_config: &configs::Detection,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<(), DecoderError> {
        use crate::yolo::impl_yolo_split_segdet_process_masks;

        self.process_tracked_yolo_end_to_end_segdet_quantized(
            tracker,
            timestamp,
            outputs,
            boxes_config,
            protos_config,
            output_boxes,
            output_masks,
            output_tracks,
            |new_boxes, mask_coeff, protos_tensor, output_boxes, output_masks| {
                impl_yolo_split_segdet_process_masks::<_, _>(
                    new_boxes,
                    mask_coeff,
                    protos_tensor,
                    output_boxes,
                    output_masks,
                )
            },
        )
        .flatten()
    }

    #[allow(clippy::too_many_arguments)]
    fn process_tracked_yolo_split_end_to_end_segdet_float<
        T,
        TR: edgefirst_tracker::Tracker<DetectBox>,
        M,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewD<T>],
        boxes_config: &configs::Boxes,
        scores_config: &configs::Scores,
        classes_config: &configs::Classes,
        mask_coeff_config: &configs::MaskCoefficients,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo>,
        mask_fn: impl FnOnce(
            Vec<(DetectBox, usize)>,
            ArrayView2<T>,
            ArrayView3<T>,
            &mut Vec<DetectBox>,
            &mut Vec<Segmentation>,
        ) -> M,
    ) -> Result<M, DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        use crate::yolo::{
            impl_yolo_end_to_end_segdet_get_boxes, postprocess_yolo_split_end_to_end_segdet,
        };
        use crate::XYXY;

        let mut skip = vec![];
        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape(&boxes_config.shape, outputs, &skip)?;
        let boxes_tensor = Self::swap_axes_if_needed(boxes_tensor, boxes_config.into());
        let boxes_tensor = boxes_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (scores_tensor, ind) =
            Self::find_outputs_with_shape(&scores_config.shape, outputs, &skip)?;
        let scores_tensor = Self::swap_axes_if_needed(scores_tensor, scores_config.into());
        let scores_tensor = scores_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (classes_tensor, ind) =
            Self::find_outputs_with_shape(&classes_config.shape, outputs, &skip)?;
        let classes_tensor = Self::swap_axes_if_needed(classes_tensor, classes_config.into());
        let classes_tensor = classes_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (mask_tensor, ind) =
            Self::find_outputs_with_shape(&mask_coeff_config.shape, outputs, &skip)?;
        let mask_tensor = Self::swap_axes_if_needed(mask_tensor, mask_coeff_config.into());
        let mask_tensor = mask_tensor.slice(s![0, .., ..]);
        skip.push(ind);

        let (protos_tensor, _) =
            Self::find_outputs_with_shape(&protos_config.shape, outputs, &skip)?;
        let protos_tensor = Self::swap_axes_if_needed(protos_tensor, protos_config.into());
        let protos_tensor = protos_tensor.slice(s![0, .., .., ..]);
        let protos_tensor = Self::protos_to_hwc(protos_tensor, protos_config);

        let (boxes, scores, classes, mask_coeff) = postprocess_yolo_split_end_to_end_segdet(
            boxes_tensor,
            scores_tensor,
            &classes_tensor,
            mask_tensor,
        )?;
        let boxes = impl_yolo_end_to_end_segdet_get_boxes::<XYXY, _, _, _>(
            boxes,
            scores,
            classes,
            self.score_threshold,
            output_boxes.capacity(),
        );

        let (new_boxes, old_boxes) =
            Self::update_tracker_yolo_segdet(tracker, timestamp, boxes, output_tracks);

        let mask_data = mask_fn(
            new_boxes,
            mask_coeff,
            protos_tensor,
            output_boxes,
            output_masks,
        );

        output_boxes.extend(old_boxes);

        Ok(mask_data)
    }

    /// Decodes split end-to-end YOLO seg detection from float tensors.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_split_end_to_end_segdet_float<
        TR: edgefirst_tracker::Tracker<DetectBox>,
        T,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewD<T>],
        boxes_config: &configs::Boxes,
        scores_config: &configs::Scores,
        classes_config: &configs::Classes,
        mask_coeff_config: &configs::MaskCoefficients,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<(), DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        use crate::yolo::impl_yolo_split_segdet_process_masks;

        self.process_tracked_yolo_split_end_to_end_segdet_float(
            tracker,
            timestamp,
            outputs,
            boxes_config,
            scores_config,
            classes_config,
            mask_coeff_config,
            protos_config,
            output_boxes,
            output_masks,
            output_tracks,
            impl_yolo_split_segdet_process_masks,
        )
        .flatten()
    }

    #[allow(clippy::too_many_arguments)]
    fn process_tracked_yolo_split_end_to_end_segdet_quantized<
        TR: edgefirst_tracker::Tracker<DetectBox>,
        M,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewDQuantized],
        boxes_config: &configs::Boxes,
        scores_config: &configs::Scores,
        classes_config: &configs::Classes,
        mask_coeff_config: &configs::MaskCoefficients,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo>,
        mask_fn: impl FnOnce(
            Vec<(DetectBox, usize)>,
            ArrayView2<f32>,
            ArrayView3<f32>,
            &mut Vec<DetectBox>,
            &mut Vec<Segmentation>,
        ) -> M,
    ) -> Result<M, DecoderError> {
        use crate::yolo::{
            impl_yolo_end_to_end_segdet_get_boxes, postprocess_yolo_split_end_to_end_segdet,
        };
        use crate::XYXY;

        let mut skip = vec![];
        let (boxes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&boxes_config.shape, outputs, &skip)?;
        skip.push(ind);
        let (scores_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&scores_config.shape, outputs, &skip)?;
        skip.push(ind);
        let (classes_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&classes_config.shape, outputs, &skip)?;
        skip.push(ind);
        let (mask_tensor, ind) =
            Self::find_outputs_with_shape_quantized(&mask_coeff_config.shape, outputs, &skip)?;
        skip.push(ind);
        let (protos_tensor, _) =
            Self::find_outputs_with_shape_quantized(&protos_config.shape, outputs, &skip)?;

        let quant_boxes = boxes_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_scores = scores_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_classes = classes_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_masks = mask_coeff_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();
        let quant_protos = protos_config
            .quantization
            .map(Quantization::from)
            .unwrap_or_default();

        let dequant_b = dequant_3d!(boxes_tensor, boxes_config, quant_boxes);
        let dequant_s = dequant_3d!(scores_tensor, scores_config, quant_scores);
        let dequant_c = dequant_3d!(classes_tensor, classes_config, quant_classes);
        let dequant_m = dequant_3d!(mask_tensor, mask_coeff_config, quant_masks);
        let dequant_p = dequant_4d!(protos_tensor, protos_config, quant_protos);

        let dequant_b_view = dequant_b.view();
        let dequant_s_view = dequant_s.view();
        let dequant_c_view = dequant_c.view();
        let dequant_m_view = dequant_m.view();
        let (boxes, scores, classes, mask_coeff) = postprocess_yolo_split_end_to_end_segdet(
            dequant_b_view,
            dequant_s_view,
            &dequant_c_view,
            dequant_m_view,
        )?;
        let boxes = impl_yolo_end_to_end_segdet_get_boxes::<XYXY, _, _, _>(
            boxes,
            scores,
            classes,
            self.score_threshold,
            output_boxes.capacity(),
        );

        let (new_boxes, old_boxes) =
            Self::update_tracker_yolo_segdet(tracker, timestamp, boxes, output_tracks);

        let mask_data = mask_fn(
            new_boxes,
            mask_coeff,
            dequant_p.view(),
            output_boxes,
            output_masks,
        );

        output_boxes.extend(old_boxes);

        Ok(mask_data)
    }

    /// Decodes split end-to-end YOLO seg detection from quantized tensors.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_split_end_to_end_segdet_quantized<
        TR: edgefirst_tracker::Tracker<DetectBox>,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewDQuantized],
        boxes_config: &configs::Boxes,
        scores_config: &configs::Scores,
        classes_config: &configs::Classes,
        mask_coeff_config: &configs::MaskCoefficients,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_masks: &mut Vec<Segmentation>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<(), DecoderError> {
        use crate::yolo::impl_yolo_split_segdet_process_masks;

        self.process_tracked_yolo_split_end_to_end_segdet_quantized(
            tracker,
            timestamp,
            outputs,
            boxes_config,
            scores_config,
            classes_config,
            mask_coeff_config,
            protos_config,
            output_boxes,
            output_masks,
            output_tracks,
            impl_yolo_split_segdet_process_masks,
        )
        .flatten()
    }

    // ------------------------------------------------------------------
    // Proto-extraction private helpers (mirror the non-proto variants)
    // ------------------------------------------------------------------
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_segdet_quantized_proto<
        TR: edgefirst_tracker::Tracker<DetectBox>,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Detection,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<ProtoData, DecoderError> {
        use crate::yolo::extract_proto_data_quant;
        let mut output_masks: Vec<Segmentation> = Vec::new();
        let output_masks = &mut output_masks;

        let protos = process_tracked_yolo_segmentation!(
            self,
            tracker,
            timestamp,
            outputs,
            boxes,
            protos,
            output_boxes,
            output_masks,
            output_tracks,
            |new_boxes, mask_tensor, quant_masks, protos_tensor, quant_protos, output_boxes, _| {
                extract_proto_data_quant(
                    new_boxes,
                    mask_tensor,
                    quant_masks,
                    protos_tensor,
                    quant_protos,
                    output_boxes,
                )
            }
        );
        Ok(protos)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_split_segdet_quantized_proto<
        TR: edgefirst_tracker::Tracker<DetectBox>,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewDQuantized],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<ProtoData, DecoderError> {
        let mut output_masks: Vec<Segmentation> = Vec::new();
        let output_masks = &mut output_masks;

        let proto = process_tracked_yolo_segmentation_split!(
            self,
            tracker,
            timestamp,
            outputs,
            boxes,
            scores,
            mask_coeff,
            protos,
            output_boxes,
            output_masks,
            output_tracks,
            |new_boxes, mask_t, quant_masks, protos_t, quant_protos, output_boxes, _| {
                crate::yolo::extract_proto_data_quant(
                    new_boxes,
                    mask_t,
                    quant_masks,
                    protos_t,
                    quant_protos,
                    output_boxes,
                )
            }
        );
        Ok(proto)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_segdet_float_proto<
        TR: edgefirst_tracker::Tracker<DetectBox>,
        T,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Detection,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<ProtoData, DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        use crate::yolo::extract_proto_data_float;

        self.process_tracked_yolo_segdet_float(
            tracker,
            timestamp,
            outputs,
            boxes,
            protos,
            output_boxes,
            &mut Vec::new(),
            output_tracks,
            |boxes, masks, protos, out, _| extract_proto_data_float(boxes, masks, protos, out),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_split_segdet_float_proto<
        TR: edgefirst_tracker::Tracker<DetectBox>,
        T,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewD<T>],
        boxes: &configs::Boxes,
        scores: &configs::Scores,
        mask_coeff: &configs::MaskCoefficients,
        protos: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<ProtoData, DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        use crate::yolo::extract_proto_data_float;

        self.process_tracked_yolo_segdet_split_float(
            tracker,
            timestamp,
            outputs,
            boxes,
            scores,
            mask_coeff,
            protos,
            output_boxes,
            &mut Vec::new(),
            output_tracks,
            |boxes, masks, protos, out, _| extract_proto_data_float(boxes, masks, protos, out),
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_end_to_end_segdet_float_proto<
        TR: edgefirst_tracker::Tracker<DetectBox>,
        T,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewD<T>],
        boxes_config: &configs::Detection,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<ProtoData, DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        use crate::yolo::extract_proto_data_float;

        self.process_tracked_yolo_end_to_end_segdet_float(
            tracker,
            timestamp,
            outputs,
            boxes_config,
            protos_config,
            output_boxes,
            &mut Vec::new(),
            output_tracks,
            |new_boxes, mask_coeff, protos_tensor, output_boxes, _| {
                extract_proto_data_float(new_boxes, mask_coeff, protos_tensor, output_boxes)
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_end_to_end_segdet_quantized_proto<
        TR: edgefirst_tracker::Tracker<DetectBox>,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewDQuantized],
        boxes_config: &configs::Detection,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<ProtoData, DecoderError> {
        use crate::yolo::extract_proto_data_float;

        self.process_tracked_yolo_end_to_end_segdet_quantized(
            tracker,
            timestamp,
            outputs,
            boxes_config,
            protos_config,
            output_boxes,
            &mut Vec::new(),
            output_tracks,
            |new_boxes, mask_coeff, protos_tensor, output_boxes, _| {
                extract_proto_data_float(new_boxes, mask_coeff, protos_tensor, output_boxes)
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_split_end_to_end_segdet_float_proto<
        TR: edgefirst_tracker::Tracker<DetectBox>,
        T,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewD<T>],
        boxes_config: &configs::Boxes,
        scores_config: &configs::Scores,
        classes_config: &configs::Classes,
        mask_coeff_config: &configs::MaskCoefficients,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<ProtoData, DecoderError>
    where
        T: Float + AsPrimitive<f32> + Send + Sync + 'static,
        f32: AsPrimitive<T>,
    {
        use crate::yolo::extract_proto_data_float;
        self.process_tracked_yolo_split_end_to_end_segdet_float(
            tracker,
            timestamp,
            outputs,
            boxes_config,
            scores_config,
            classes_config,
            mask_coeff_config,
            protos_config,
            output_boxes,
            &mut Vec::new(),
            output_tracks,
            |new_boxes, mask_coeff, protos_tensor, output_boxes, _| {
                extract_proto_data_float(new_boxes, mask_coeff, protos_tensor, output_boxes)
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_tracked_yolo_split_end_to_end_segdet_quantized_proto<
        TR: edgefirst_tracker::Tracker<DetectBox>,
    >(
        &self,
        tracker: &mut TR,
        timestamp: u64,
        outputs: &[ArrayViewDQuantized],
        boxes_config: &configs::Boxes,
        scores_config: &configs::Scores,
        classes_config: &configs::Classes,
        mask_coeff_config: &configs::MaskCoefficients,
        protos_config: &configs::Protos,
        output_boxes: &mut Vec<DetectBox>,
        output_tracks: &mut Vec<TrackInfo>,
    ) -> Result<ProtoData, DecoderError> {
        use crate::yolo::extract_proto_data_float;

        self.process_tracked_yolo_split_end_to_end_segdet_quantized(
            tracker,
            timestamp,
            outputs,
            boxes_config,
            scores_config,
            classes_config,
            mask_coeff_config,
            protos_config,
            output_boxes,
            &mut Vec::new(),
            output_tracks,
            |new_boxes, mask_coeff, protos_tensor, output_boxes, _| {
                extract_proto_data_float(new_boxes, mask_coeff, protos_tensor, output_boxes)
            },
        )
    }

    pub(super) fn update_tracker_yolo_segdet<TR: edgefirst_tracker::Tracker<DetectBox>>(
        tracker: &mut TR,
        timestamp: u64,
        boxes: Vec<(DetectBox, usize)>,
        tracks: &mut Vec<TrackInfo>,
    ) -> (Vec<(DetectBox, usize)>, Vec<DetectBox>) {
        // custom tracking since we can only use live boxes for masks. We
        // can still output "old" boxes but they will not have masks.
        // let live_tracks = tracker.update(output_boxes, timestamp);
        let (new_boxes, boxes_indices): (Vec<_>, Vec<_>) = boxes.into_iter().unzip();
        let live_tracks = tracker.update(&new_boxes, timestamp);
        let old_tracks = tracker
            .get_active_tracks()
            .into_iter()
            .filter(|x| x.info.last_updated != timestamp)
            .collect::<Vec<_>>();

        let live_boxes = live_tracks
            .iter()
            .zip(new_boxes)
            .zip(boxes_indices)
            .filter_map(|((t, mut b), ind)| {
                if let Some(t) = t {
                    b.bbox = t.tracked_location.into();
                    Some((b, ind))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let old_boxes = old_tracks
            .iter()
            .map(|t| {
                let mut b = t.last_box;
                b.bbox = t.info.tracked_location.into();
                b
            })
            .collect::<Vec<_>>();

        tracks.clear();
        tracks.extend(live_tracks.into_iter().flatten());
        tracks.extend(old_tracks.into_iter().map(|x| x.info));
        (live_boxes, old_boxes)
    }

    pub(super) fn update_tracker<TR: edgefirst_tracker::Tracker<DetectBox>>(
        tracker: &mut TR,
        timestamp: u64,
        output_boxes: &mut Vec<DetectBox>,
        output_tracks: &mut Vec<TrackInfo>,
    ) {
        tracker.update(output_boxes, timestamp);
        let tracks = tracker.get_active_tracks();

        let (new_tracks, new_boxes): (Vec<_>, Vec<_>) = tracks
            .into_iter()
            .map(|t| {
                let mut box_ = t.last_box;
                box_.bbox = t.info.tracked_location.into();
                (t.info, box_)
            })
            .unzip();
        output_boxes.clear();
        output_boxes.extend(new_boxes);
        output_tracks.clear();
        output_tracks.extend(new_tracks);
    }
}
