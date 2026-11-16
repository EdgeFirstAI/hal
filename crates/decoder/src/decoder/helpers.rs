// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::config::ConfigOutputRef;
use super::configs::{self, DecoderType, DimName};
use super::{ArrayViewDQuantized, Decoder};
use crate::DecoderError;
use ndarray::{ArrayView, ArrayViewD, Dimension};

impl Decoder {
    /// The tensor of each detection head, found by shape in configured
    /// order, and the tensors' indices in `outputs`. Heads of one shape take
    /// successive tensors of that shape.
    pub(super) fn match_outputs_to_detect<'a, 'b, T>(
        configs: &[configs::Detection],
        outputs: &'a [ArrayViewD<'b, T>],
    ) -> Result<(Vec<&'a ArrayViewD<'b, T>>, Vec<usize>), DecoderError> {
        let mut matched = Vec::with_capacity(configs.len());
        let mut indices = Vec::with_capacity(configs.len());
        for c in configs {
            let (output, index) = Self::find_outputs_with_shape(&c.shape, outputs, &indices)?;
            matched.push(output);
            indices.push(index);
        }
        Ok((matched, indices))
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

    /// Returns the canonical logical-axis index for a given `(output role,
    /// decoder, dim name)` tuple: the position the axis occupies in the
    /// decoder's internal tensor view, e.g. index 1 for `(Boxes,
    /// Ultralytics, BoxCoords)` in `[batch, coords, anchors]`.
    ///
    /// Each `(role, decoder)` pair maps the dim names that have a
    /// canonical position to a distinct index in `[0, ndim)`. Dim names
    /// without a canonical position for the role (for example `Height`
    /// / `Width` on a flat-anchor detection output) return `None`.
    pub(super) fn canonical_axis_index(
        config: ConfigOutputRef<'_>,
        name: DimName,
    ) -> Option<usize> {
        use ConfigOutputRef::*;
        use DecoderType::*;
        use DimName::*;
        let decoder = config.decoder();
        match (&config, decoder) {
            // Ultralytics flat detection: [batch, num_features, num_boxes]
            (Detection(_), Ultralytics) => match name {
                Batch => Some(0),
                NumFeatures => Some(1),
                NumBoxes => Some(2),
                _ => None,
            },
            // ModelPack per-scale FPN detection: dshape is
            // [batch, height, width, num_anchors_x_features]; only the batch
            // axis has a canonical index.
            (Detection(_), ModelPack) => match name {
                Batch => Some(0),
                _ => None,
            },
            // Ultralytics boxes: [batch, box_coords, num_boxes]
            (Boxes(_), Ultralytics) => match name {
                Batch => Some(0),
                BoxCoords => Some(1),
                NumBoxes => Some(2),
                _ => None,
            },
            // ModelPack boxes: [batch, num_boxes, padding, box_coords]
            (Boxes(_), ModelPack) => match name {
                Batch => Some(0),
                NumBoxes => Some(1),
                Padding => Some(2),
                BoxCoords => Some(3),
                _ => None,
            },
            // Ultralytics scores / classes: [batch, num_classes, num_boxes]
            (Scores(_) | Classes(_), Ultralytics) => match name {
                Batch => Some(0),
                NumClasses => Some(1),
                NumBoxes => Some(2),
                _ => None,
            },
            // ModelPack scores / classes: [batch, num_boxes, num_classes]
            (Scores(_) | Classes(_), ModelPack) => match name {
                Batch => Some(0),
                NumBoxes => Some(1),
                NumClasses => Some(2),
                _ => None,
            },
            // The semantic decoder has no box-family layout.
            (Detection(_) | Boxes(_) | Scores(_) | Classes(_), Semantic) => None,
            // Segmentation (decoder-agnostic): [batch, height, width, num_classes]
            (Segmentation(_), _) => match name {
                Batch => Some(0),
                Height => Some(1),
                Width => Some(2),
                NumClasses => Some(3),
                _ => None,
            },
            // Mask (decoder-agnostic): [batch, height, width]
            (Mask(_), _) => match name {
                Batch => Some(0),
                Height => Some(1),
                Width => Some(2),
                _ => None,
            },
            // Protos (decoder-agnostic): [batch, height, width, num_protos]
            (Protos(_), _) => match name {
                Batch => Some(0),
                Height => Some(1),
                Width => Some(2),
                NumProtos => Some(3),
                _ => None,
            },
            // Mask coefficients (decoder-agnostic): [batch, num_protos, num_boxes]
            (MaskCoefficients(_), _) => match name {
                Batch => Some(0),
                NumProtos => Some(1),
                NumBoxes => Some(2),
                _ => None,
            },
        }
    }

    /// Permutes the tensor view into the decoder's canonical logical-axis
    /// order (as defined by [`Self::canonical_axis_index`]).
    ///
    /// The input view's strides must match the producer's physical memory
    /// layout. Only the stride tuple is permuted; no bytes move.
    ///
    /// Dim names without a canonical position for the current role (e.g.
    /// `Height` / `Width` on a flat-anchor detection output) sort to the
    /// tail and keep their relative order.
    pub(super) fn swap_axes_if_needed<'a, T, D: Dimension>(
        array: &ArrayView<'a, T, D>,
        config: ConfigOutputRef,
    ) -> ArrayView<'a, T, D> {
        let mut array = array.clone();
        if config.dshape().is_empty() {
            return array;
        }
        // Sentinel for axes with no canonical index for this role: they sort
        // to the end.
        const TAIL: usize = usize::MAX;
        let mut current_order: Vec<usize> = config
            .dshape()
            .iter()
            .map(|(name, _)| Self::canonical_axis_index(config.clone(), *name).unwrap_or(TAIL))
            .collect();

        assert_eq!(array.shape().len(), current_order.len());
        // Bubble sort with `swap_axes`, which permutes the stride tuple.
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

    /// [`match_outputs_to_detect`](Self::match_outputs_to_detect) for
    /// quantized outputs.
    pub(super) fn match_outputs_to_detect_quantized<'a, 'b>(
        configs: &[configs::Detection],
        outputs: &'a [ArrayViewDQuantized<'b>],
    ) -> Result<(Vec<&'a ArrayViewDQuantized<'b>>, Vec<usize>), DecoderError> {
        let mut matched = Vec::with_capacity(configs.len());
        let mut indices = Vec::with_capacity(configs.len());
        for c in configs {
            let (output, index) =
                Self::find_outputs_with_shape_quantized(&c.shape, outputs, &indices)?;
            matched.push(output);
            indices.push(index);
        }
        Ok((matched, indices))
    }

    /// Validate that an output's `shape` and `dshape` satisfy the HAL
    /// physical-order contract: when `dshape` is present, `shape[i]` and
    /// `dshape[i]` describe the same axis, and both are listed in
    /// physical memory order (outermost first, innermost last). An output
    /// whose `shape` is in the decoder's canonical order may omit `dshape`.
    ///
    /// Runs once per output at
    /// [`DecoderBuilder::build`](super::DecoderBuilder::build) and checks:
    ///
    /// 1. `dshape.len() == shape.len()` (when dshape is present).
    /// 2. Each `dshape[i].1` size matches `shape[i]`.
    /// 3. `Padding` axes have size 1.
    /// 4. `BoxCoords` axes have size 4 (xyxy/xywh convention).
    /// 5. No axis name appears twice in the dshape.
    ///
    /// The per-role required-dim-name checks are done by `verify_dshapes`
    /// in `DecoderBuilder`.
    pub(super) fn validate_output_layout(config: ConfigOutputRef<'_>) -> Result<(), DecoderError> {
        use ConfigOutputRef::*;
        let role_name = match config {
            Detection(_) => "detection",
            Mask(_) => "mask",
            Segmentation(_) => "segmentation",
            Protos(_) => "protos",
            Scores(_) => "scores",
            Boxes(_) => "boxes",
            MaskCoefficients(_) => "mask_coefficients",
            Classes(_) => "classes",
        };

        let dshape = config.dshape();
        let shape = match config {
            Detection(v) => &v.shape,
            Mask(v) => &v.shape,
            Segmentation(v) => &v.shape,
            Protos(v) => &v.shape,
            Scores(v) => &v.shape,
            Boxes(v) => &v.shape,
            MaskCoefficients(v) => &v.shape,
            Classes(v) => &v.shape,
        };

        crate::configs::validate_dshape(role_name, shape, dshape)
    }
}
