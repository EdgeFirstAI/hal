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

    /// Returns the canonical logical-axis index for a given `(output role,
    /// decoder, dim name)` tuple.
    ///
    /// The returned index is the position the axis must occupy in the
    /// decoder's internal tensor view — e.g. `(Boxes, Ultralytics,
    /// BoxCoords)` must be at index 1 because the ultralytics box kernels
    /// iterate `[batch, coords, anchors]`.
    ///
    /// Each `(role, decoder)` pair maps the dim names that have a
    /// canonical position to a distinct index in `[0, ndim)`. Dim names
    /// without a canonical position for the role — for example `Height`
    /// / `Width` on a flat-anchor detection output, or `NumAnchorsXFeatures`
    /// on a ModelPack per-scale FPN tensor — return `None`. The caller
    /// ([`Self::swap_axes_if_needed`]) maps `None` to a `usize::MAX`
    /// sentinel so those axes sort to the tail of the stride permutation;
    /// they keep their relative order among themselves but end up after
    /// all canonically-placed axes. This preserves the legacy behaviour
    /// that ModelPack per-scale FPN configs rely on.
    ///
    /// This table is the single source of truth for role → canonical order;
    /// it replaces the per-role `_order()` helpers that previously scattered
    /// the same information across eleven separate functions.
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
            // [batch, height, width, num_anchors_x_features] and none of
            // the non-batch axes have a canonical logical slot — they
            // drive the FPN decode in declared order, with the batch
            // axis bubbled to index 0 by swap_axes_if_needed and the
            // remainder falling to the TAIL sentinel so their relative
            // order is preserved.
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

    // `protos_to_hwc` was removed after the physical-order contract
    // landed. Under the contract, the 3D view entering a decode kernel
    // is always canonical HWC: either `swap_axes_if_needed` reordered
    // the 4D tensor to `[batch, height, width, num_protos]` using the
    // caller's dshape, or the caller omitted dshape and asserted the
    // declared shape was already canonical. The pre-2026-04 size
    // heuristic (permute when `d0 < d1 && d0 < d2`) was the workaround
    // that couldn't distinguish physically-NCHW bytes from
    // physically-NHWC bytes mislabelled as NCHW — the root cause of
    // the vertical-stripe mask bug on i.MX 8M Plus TFLite.

    /// Permutes the tensor view into the decoder's canonical logical-axis
    /// order (as defined by [`Self::canonical_axis_index`]).
    ///
    /// **Precondition**: the input view's strides must already match the
    /// producer's physical memory layout. That is guaranteed by the HAL
    /// contract — callers declare `shape` and `dshape` in physical memory
    /// order (outermost first), so the C-contiguous strides derived at
    /// wrap time are correct by construction.
    ///
    /// This function only reshuffles the stride tuple; it never moves
    /// bytes. If the precondition is violated (declared shape does not
    /// match physical layout), this function silently produces a view
    /// with strides that point into the wrong places — the bug we explicitly
    /// avoid by enforcing the contract at [`Decoder::validate_output_layout`].
    ///
    /// Dim names without a canonical position for the current role (e.g.
    /// `Height` / `Width` on a flat-anchor detection output) sort to the
    /// tail. This preserves the legacy "unreachable-axis stays at the
    /// back" behaviour that the per-scale modelpack FPN path relies on.
    pub(super) fn swap_axes_if_needed<'a, T, D: Dimension>(
        array: &ArrayView<'a, T, D>,
        config: ConfigOutputRef,
    ) -> ArrayView<'a, T, D> {
        let mut array = array.clone();
        if config.dshape().is_empty() {
            return array;
        }
        // Sentinel for axes with no canonical index for this role —
        // guarantees they sort to the end without affecting relative
        // order of canonical axes.
        const TAIL: usize = usize::MAX;
        let mut current_order: Vec<usize> = config
            .dshape()
            .iter()
            .map(|(name, _)| Self::canonical_axis_index(config.clone(), *name).unwrap_or(TAIL))
            .collect();

        assert_eq!(array.shape().len(), current_order.len());
        // Simple bubble sort: swap_axes is cheap (stride-tuple permutation
        // only, no byte movement) and ndim is small (≤ 4 in practice).
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

    /// Validate that an output's `shape` and `dshape` satisfy the HAL
    /// physical-order contract.
    ///
    /// The contract is: if `dshape` is present, `shape[i]` and
    /// `dshape[i]` describe the same axis, and both are listed in
    /// physical memory order (outermost first, innermost last).
    /// C-contiguous strides derived from `shape` are therefore correct
    /// by construction. Callers who already declare `shape` in the
    /// decoder's canonical order may omit `dshape` entirely.
    ///
    /// This validator runs once per output at
    /// [`DecoderBuilder::build`](super::DecoderBuilder::build) and enforces
    /// structural, role-agnostic rules:
    ///
    /// 1. `dshape.len() == shape.len()` (when dshape is present).
    /// 2. Each `dshape[i].1` size matches `shape[i]` — catches the common
    ///    mistake of declaring dshape in a different order than shape
    ///    (the exact failure mode that caused the TFLite stripe bug).
    /// 3. `Padding` axes must have size 1.
    /// 4. `BoxCoords` axes must have size 4 (xyxy/xywh convention).
    /// 5. Dshape roles are unique within the output (no axis name appears
    ///    twice), so every axis maps to a distinct canonical slot.
    ///
    /// Per-role *required-dim-name* checks (e.g. Ultralytics boxes must
    /// carry `Batch`, `BoxCoords`, and `NumBoxes`) are enforced
    /// separately by `verify_dshapes` at the role-specific verification
    /// step in `DecoderBuilder`. The two layers are intentional: this
    /// validator rejects malformed dshapes at build entry; role
    /// verification then checks role-specific semantic constraints.
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

        if dshape.is_empty() {
            // Omitted dshape → caller asserts shape is already canonical.
            return Ok(());
        }

        if dshape.len() != shape.len() {
            return Err(DecoderError::InvalidConfig(format!(
                "{role_name} output: dshape has {} entries but shape has {} entries \
                 — shape and dshape must describe the same axes in the same order",
                dshape.len(),
                shape.len()
            )));
        }

        for (i, (name, size)) in dshape.iter().enumerate() {
            if *size != shape[i] {
                return Err(DecoderError::InvalidConfig(format!(
                    "{role_name} output: dshape[{i}] = ({name}, {size}) does not \
                     match shape[{i}] = {} — declare shape and dshape in the \
                     same physical order (outermost axis first)",
                    shape[i]
                )));
            }
            // Role-agnostic size constraints on well-known axis roles.
            if *name == DimName::Padding && *size != 1 {
                return Err(DecoderError::InvalidConfig(format!(
                    "{role_name} output: `padding` axis must have size 1, got {size}"
                )));
            }
            if *name == DimName::BoxCoords && *size != 4 {
                return Err(DecoderError::InvalidConfig(format!(
                    "{role_name} output: `box_coords` axis must have size 4, got {size}"
                )));
            }
        }

        // Duplicate-axis check: duplicates map to the same canonical slot
        // and would break `swap_axes_if_needed` by producing a
        // non-permutation ordering.
        for i in 0..dshape.len() {
            for j in (i + 1)..dshape.len() {
                if dshape[i].0 == dshape[j].0 {
                    return Err(DecoderError::InvalidConfig(format!(
                        "{role_name} output: dshape axis `{}` appears at \
                         both index {i} and {j}",
                        dshape[i].0
                    )));
                }
            }
        }

        Ok(())
    }
}
