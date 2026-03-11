// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::config::ConfigOutputRef;
use super::configs::{self, DecoderType, DimName};
use super::{ArrayViewDQuantized, Decoder};
use crate::DecoderError;
use ndarray::{ArrayView, ArrayViewD, Dimension};

impl Decoder {
    pub(super) fn match_outputs_to_detect<'a, 'b, T>(
        configs: &[configs::Detection],
        outputs: &'a [ArrayViewD<'b, T>],
    ) -> Result<Vec<&'a ArrayViewD<'b, T>>, DecoderError> {
        let mut new_output_order = Vec::new();
        for c in configs {
            let mut found = false;
            for o in outputs {
                if o.shape() == c.shape {
                    new_output_order.push(o);
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(DecoderError::InvalidShape(format!(
                    "Did not find output with shape {:?}",
                    c.shape
                )));
            }
        }
        Ok(new_output_order)
    }

    pub(super) fn find_outputs_with_shape<'a, 'b, T>(
        shape: &[usize],
        outputs: &'a [ArrayViewD<'b, T>],
        skip: &[usize],
    ) -> Result<(&'a ArrayViewD<'b, T>, usize), DecoderError> {
        for (ind, o) in outputs.iter().enumerate() {
            if skip.contains(&ind) {
                continue;
            }
            if o.shape() == shape {
                return Ok((o, ind));
            }
        }
        Err(DecoderError::InvalidShape(format!(
            "Did not find output with shape {:?}",
            shape
        )))
    }

    pub(super) fn find_outputs_with_shape_quantized<'a, 'b>(
        shape: &[usize],
        outputs: &'a [ArrayViewDQuantized<'b>],
        skip: &[usize],
    ) -> Result<(&'a ArrayViewDQuantized<'b>, usize), DecoderError> {
        for (ind, o) in outputs.iter().enumerate() {
            if skip.contains(&ind) {
                continue;
            }
            if o.shape() == shape {
                return Ok((o, ind));
            }
        }
        Err(DecoderError::InvalidShape(format!(
            "Did not find output with shape {:?}",
            shape
        )))
    }

    /// This is split detection, need to swap axes to batch, height, width,
    /// num_anchors_x_features,
    pub(super) fn modelpack_det_order(x: DimName) -> usize {
        match x {
            DimName::Batch => 0,
            DimName::NumBoxes => 1,
            DimName::Padding => 2,
            DimName::BoxCoords => 3,
            _ => 1000, // this should be unreachable
        }
    }

    // This is Ultralytics detection, need to swap axes to batch, num_features,
    // height, width
    pub(super) fn yolo_det_order(x: DimName) -> usize {
        match x {
            DimName::Batch => 0,
            DimName::NumFeatures => 1,
            DimName::NumBoxes => 2,
            _ => 1000, // this should be unreachable
        }
    }

    // This is modelpack boxes, need to swap axes to batch, num_boxes, padding,
    // box_coords
    pub(super) fn modelpack_boxes_order(x: DimName) -> usize {
        match x {
            DimName::Batch => 0,
            DimName::NumBoxes => 1,
            DimName::Padding => 2,
            DimName::BoxCoords => 3,
            _ => 1000, // this should be unreachable
        }
    }

    /// This is Ultralytics boxes, need to swap axes to batch, box_coords,
    /// num_boxes
    pub(super) fn yolo_boxes_order(x: DimName) -> usize {
        match x {
            DimName::Batch => 0,
            DimName::BoxCoords => 1,
            DimName::NumBoxes => 2,
            _ => 1000, // this should be unreachable
        }
    }

    /// This is modelpack scores, need to swap axes to batch, num_boxes,
    /// num_classes
    pub(super) fn modelpack_scores_order(x: DimName) -> usize {
        match x {
            DimName::Batch => 0,
            DimName::NumBoxes => 1,
            DimName::NumClasses => 2,
            _ => 1000, // this should be unreachable
        }
    }

    pub(super) fn yolo_scores_order(x: DimName) -> usize {
        match x {
            DimName::Batch => 0,
            DimName::NumClasses => 1,
            DimName::NumBoxes => 2,
            _ => 1000, // this should be unreachable
        }
    }

    /// This is modelpack segmentation, need to swap axes to batch, height,
    /// width, num_classes
    pub(super) fn modelpack_segmentation_order(x: DimName) -> usize {
        match x {
            DimName::Batch => 0,
            DimName::Height => 1,
            DimName::Width => 2,
            DimName::NumClasses => 3,
            _ => 1000, // this should be unreachable
        }
    }

    /// This is modelpack masks, need to swap axes to batch, height,
    /// width
    pub(super) fn modelpack_mask_order(x: DimName) -> usize {
        match x {
            DimName::Batch => 0,
            DimName::Height => 1,
            DimName::Width => 2,
            _ => 1000, // this should be unreachable
        }
    }

    /// This is yolo protos, need to swap axes to batch, height, width,
    /// num_protos
    pub(super) fn yolo_protos_order(x: DimName) -> usize {
        match x {
            DimName::Batch => 0,
            DimName::Height => 1,
            DimName::Width => 2,
            DimName::NumProtos => 3,
            _ => 1000, // this should be unreachable
        }
    }

    /// This is yolo mask coefficients, need to swap axes to batch, num_protos,
    /// num_boxes
    pub(super) fn yolo_maskcoefficients_order(x: DimName) -> usize {
        match x {
            DimName::Batch => 0,
            DimName::NumProtos => 1,
            DimName::NumBoxes => 2,
            _ => 1000, // this should be unreachable
        }
    }

    pub(super) fn get_order_fn(config: ConfigOutputRef) -> fn(DimName) -> usize {
        let decoder_type = config.decoder();
        match (config, decoder_type) {
            (ConfigOutputRef::Detection(_), DecoderType::ModelPack) => Self::modelpack_det_order,
            (ConfigOutputRef::Detection(_), DecoderType::Ultralytics) => Self::yolo_det_order,
            (ConfigOutputRef::Boxes(_), DecoderType::ModelPack) => Self::modelpack_boxes_order,
            (ConfigOutputRef::Boxes(_), DecoderType::Ultralytics) => Self::yolo_boxes_order,
            (ConfigOutputRef::Scores(_), DecoderType::ModelPack) => Self::modelpack_scores_order,
            (ConfigOutputRef::Scores(_), DecoderType::Ultralytics) => Self::yolo_scores_order,
            (ConfigOutputRef::Segmentation(_), _) => Self::modelpack_segmentation_order,
            (ConfigOutputRef::Mask(_), _) => Self::modelpack_mask_order,
            (ConfigOutputRef::Protos(_), _) => Self::yolo_protos_order,
            (ConfigOutputRef::MaskCoefficients(_), _) => Self::yolo_maskcoefficients_order,
            (ConfigOutputRef::Classes(_), _) => Self::yolo_scores_order,
        }
    }

    /// Ensure a 3D protos tensor is in HWC order.  When dshape is set,
    /// `swap_axes_if_needed` already reorders the 4D tensor to NHWC before
    /// the batch-dim slice, so the resulting 3D view is already HWC.
    ///
    /// When dshape is empty (no named dimensions), we detect the layout by
    /// comparing axis sizes: the channel/proto count (e.g. 32) is always
    /// smaller than the spatial dimensions (e.g. 160×160).  If axis 0 is
    /// the smallest, the tensor is CHW and needs permutation to HWC;
    /// otherwise it is already HWC.
    ///
    /// **Known limitations**: the heuristic fails when the channel count is
    /// not strictly the smallest dimension (e.g. protos shape `(32, 1, 1)`
    /// or `(160, 5, 5)`).  Set `dshape` in the config for reliable axis
    /// ordering in these edge cases.
    pub(super) fn protos_to_hwc<'a, T>(
        protos: ArrayView<'a, T, ndarray::Ix3>,
        config: &configs::Protos,
    ) -> ArrayView<'a, T, ndarray::Ix3> {
        if config.dshape.is_empty() {
            let (d0, d1, d2) = protos.dim();
            log::debug!(
                "protos_to_hwc: no dshape configured, using size heuristic on \
                 shape ({d0}, {d1}, {d2}); set dshape in config for reliable ordering"
            );
            if d0 < d1 && d0 < d2 {
                // CHW (from NCHW) → permute to HWC
                protos.permuted_axes([1, 2, 0])
            } else {
                // Already HWC (from NHWC) or ambiguous — keep as-is
                protos
            }
        } else {
            protos
        }
    }

    pub(super) fn swap_axes_if_needed<'a, T, D: Dimension>(
        array: &ArrayView<'a, T, D>,
        config: ConfigOutputRef,
    ) -> ArrayView<'a, T, D> {
        let mut array = array.clone();
        if config.dshape().is_empty() {
            return array;
        }
        let order_fn: fn(DimName) -> usize = Self::get_order_fn(config.clone());
        let mut current_order: Vec<usize> = config
            .dshape()
            .iter()
            .map(|x| order_fn(x.0))
            .collect::<Vec<_>>();

        assert_eq!(array.shape().len(), current_order.len());
        // do simple bubble sort as swap_axes is inexpensive and the
        // number of dimensions is small
        for i in 0..current_order.len() {
            let mut swapped = false;
            for j in 0..current_order.len() - 1 - i {
                if current_order[j] > current_order[j + 1] {
                    array.swap_axes(j, j + 1);
                    current_order.swap(j, j + 1);
                    swapped = true;
                }
            }
            if !swapped {
                break;
            }
        }
        array
    }

    pub(super) fn match_outputs_to_detect_quantized<'a, 'b>(
        configs: &[configs::Detection],
        outputs: &'a [ArrayViewDQuantized<'b>],
    ) -> Result<Vec<&'a ArrayViewDQuantized<'b>>, DecoderError> {
        let mut new_output_order = Vec::new();
        for c in configs {
            let mut found = false;
            for o in outputs {
                if o.shape() == c.shape {
                    new_output_order.push(o);
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(DecoderError::InvalidShape(format!(
                    "Did not find output with shape {:?}",
                    c.shape
                )));
            }
        }
        Ok(new_output_order)
    }
}
