// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Shared vocabulary for describing detections and masks.
//!
//! These types say *what a model reported about a buffer*, as plain data with
//! no decoding and no drawing behaviour. They live here, beside
//! [`PixelFormat`](crate::PixelFormat), [`Region`](crate::Region) and
//! [`Colorimetry`](crate::Colorimetry), because `edgefirst-tensor` is this
//! project's datatype crate.
//!
//! # Why here and not in `edgefirst-decoder`
//!
//! `edgefirst-image` needs this vocabulary to draw — `draw_decoded_masks` and
//! `draw_proto_masks` take nothing else — but it has no business knowing how a
//! YOLO or ModelPack head is decoded. Defining it in the decoder made a
//! rendering crate depend on a model-postprocessing crate, which linked
//! `serde_json`, `serde_yaml_ng`, `unsafe-libyaml`, `ndarray-stats`,
//! `argminmax`, `rand`, `indexmap`, `hashbrown` and `itertools` into every
//! image build: a YAML parser, because drawing a rectangle needed
//! [`DetectBox`]. With the vocabulary here, `image` and `decoder` are siblings
//! over a common base instead of a stack.
//!
//! Nothing in this module adds a dependency to `edgefirst-tensor`: every field
//! is a primitive or a [`TensorDyn`].

use crate::TensorDyn;

/// A bounding box with f32 coordinates in XYXY format
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct BoundingBox {
    /// left-most normalized coordinate of the bounding box
    pub xmin: f32,
    /// top-most normalized coordinate of the bounding box
    pub ymin: f32,
    /// right-most normalized coordinate of the bounding box
    pub xmax: f32,
    /// bottom-most normalized coordinate of the bounding box
    pub ymax: f32,
}

impl BoundingBox {
    /// Creates a new BoundingBox from the given coordinates
    pub fn new(xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> Self {
        Self {
            xmin,
            ymin,
            xmax,
            ymax,
        }
    }

    /// Transforms BoundingBox so that `xmin <= xmax` and `ymin <= ymax`
    ///
    /// ```
    /// # use edgefirst_tensor::BoundingBox;
    /// let bbox = BoundingBox::new(0.8, 0.6, 0.4, 0.2);
    /// let canonical_bbox = bbox.to_canonical();
    /// assert_eq!(canonical_bbox, BoundingBox::new(0.4, 0.2, 0.8, 0.6));
    /// ```
    pub fn to_canonical(&self) -> Self {
        let xmin = self.xmin.min(self.xmax);
        let xmax = self.xmin.max(self.xmax);
        let ymin = self.ymin.min(self.ymax);
        let ymax = self.ymin.max(self.ymax);
        BoundingBox {
            xmin,
            ymin,
            xmax,
            ymax,
        }
    }
}

impl From<BoundingBox> for [f32; 4] {
    /// Converts a BoundingBox into an array of 4 f32 values in xmin, ymin,
    /// xmax, ymax order
    /// # Examples
    /// ```
    /// # use edgefirst_tensor::BoundingBox;
    /// let bbox = BoundingBox {
    ///     xmin: 0.1,
    ///     ymin: 0.2,
    ///     xmax: 0.3,
    ///     ymax: 0.4,
    /// };
    /// let arr: [f32; 4] = bbox.into();
    /// assert_eq!(arr, [0.1, 0.2, 0.3, 0.4]);
    /// ```
    fn from(b: BoundingBox) -> Self {
        [b.xmin, b.ymin, b.xmax, b.ymax]
    }
}

impl From<[f32; 4]> for BoundingBox {
    // Converts an array of 4 f32 values in xmin, ymin, xmax, ymax order into a
    // BoundingBox
    fn from(arr: [f32; 4]) -> Self {
        BoundingBox {
            xmin: arr[0],
            ymin: arr[1],
            xmax: arr[2],
            ymax: arr[3],
        }
    }
}

/// A detection box with f32 bbox and score
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DetectBox {
    pub bbox: BoundingBox,
    /// model-specific score for this detection, higher implies more confidence
    pub score: f32,
    /// label index for this detection
    pub label: usize,
}

/// A segmentation result paired with the normalized bounding box that
/// describes the spatial extent of `segmentation`.
///
/// `Segmentation` bounds are for rendering masks, and `DetectBox.bbox` for
/// IoU evaluation, drawing detection rectangles, etc. The mask region always
/// encloses the detection bbox.
///
/// The `xmin`/`ymin`/`xmax`/`ymax` fields describe the **mask region** — the
/// proto-grid-aligned crop the `segmentation` tensor was sliced from. They are
/// quantized to the proto grid step (`1/proto_height` and `1/proto_width`,
/// typically `1/160`), so the region's origin floors and its extent ceils
/// relative to the companion [`DetectBox`]'s `bbox`. The detection bbox itself
/// stays un-snapped (see EDGEAI-1304).
#[derive(Debug)]
pub struct Segmentation {
    /// Left-most normalized coordinate of the mask region (proto-grid
    /// floor of the companion `DetectBox.bbox.xmin`).
    pub xmin: f32,
    /// Top-most normalized coordinate of the mask region (proto-grid
    /// floor of the companion `DetectBox.bbox.ymin`).
    pub ymin: f32,
    /// Right-most normalized coordinate of the mask region (proto-grid
    /// ceil of the companion `DetectBox.bbox.xmax`).
    pub xmax: f32,
    /// Bottom-most normalized coordinate of the mask region (proto-grid
    /// ceil of the companion `DetectBox.bbox.ymax`).
    pub ymax: f32,
    /// Mask data, shape `[H, W, C]`, dtype `U8`.
    ///
    /// For instance segmentation (e.g. YOLO): `C=1` — per-instance mask with
    /// continuous sigmoid confidence values quantized to u8 (0 = background,
    /// 255 = full confidence). Renderers typically threshold at 128 (sigmoid
    /// 0.5) or use smooth interpolation for anti-aliased edges.
    ///
    /// For semantic segmentation (e.g. ModelPack): `C=num_classes` — per-pixel
    /// class scores where the object class is the argmax index.
    ///
    /// A [`TensorDyn`] rather than an `ndarray::Array3<u8>`, matching
    /// [`ProtoData`]. That is what keeps this crate free of an `ndarray`
    /// dependency; callers who want ndarray ergonomics already have
    /// [`TensorMapTrait::view`](crate::TensorMapTrait::view) behind the
    /// existing optional `ndarray` feature.
    pub segmentation: TensorDyn,
}

/// Memory layout of the prototype tensor within [`ProtoData`].
///
/// Models may output protos in either channel-last (NHWC) or channel-first
/// (NCHW) layout. The mask materialisation kernels dispatch on this field to
/// avoid a costly per-frame transpose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtoLayout {
    /// Channel-last: tensor shape is `[H, W, K]`, contiguous along K.
    /// This is the traditional layout produced by the `extract_proto_data`
    /// path after transposing from NCHW model outputs.
    Nhwc,
    /// Channel-first: tensor shape is `[K, H, W]`, each channel plane is
    /// contiguous. Skipping the NCHW→NHWC transpose saves ~3 ms per frame
    /// on Cortex-A53/A55 targets (819 KB for 32×160×160 protos).
    Nchw,
}

/// Raw prototype data for fused decode+render pipelines.
///
/// Holds post-NMS intermediate state before mask materialization, allowing the
/// renderer to compute `mask_coeff @ protos` directly (e.g. in a GPU fragment
/// shader) without materializing intermediate masks.
///
/// Both fields are carried as [`TensorDyn`] so downstream consumers (Rust, C
/// API, Python) get zero-copy typed access through the HAL's shared tensor
/// infrastructure. Dtype policy:
///
/// | Source model | protos dtype | mask_coefficients dtype | protos.quantization |
/// |---|---|---|---|
/// | int8 quantized | [`TensorDyn::I8`] | [`TensorDyn::I8`] (raw + quantization) | `Some(q)` |
/// | f32 | [`TensorDyn::F32`] | [`TensorDyn::F32`] | `None` |
/// | f16 (TensorRT fp16) | [`TensorDyn::F16`] | [`TensorDyn::F16`] | `None` |
/// | f64 (narrowed) | [`TensorDyn::F32`] | [`TensorDyn::F32`] | `None` |
///
/// Quantization metadata lives on the proto tensor itself via
/// [`Tensor::quantization`](crate::Tensor::quantization) — float tensors
/// cannot carry quantization (compile-time gated on the `IntegerType` sealed
/// trait).
///
/// `TensorDyn` is not `Clone`, so neither is `ProtoData`. Consumers that need
/// to share the proto buffer across threads should use `TensorDyn::clone_fd`
/// / `dmabuf_clone` to dup the backing fd.
#[derive(Debug)]
pub struct ProtoData {
    /// Per-detection mask coefficients, shape `[num_detections, num_protos]`.
    pub mask_coefficients: TensorDyn,
    /// Prototype tensor.
    ///
    /// - When `layout == ProtoLayout::Nhwc`: shape is `[proto_h, proto_w, num_protos]`.
    /// - When `layout == ProtoLayout::Nchw`: shape is `[num_protos, proto_h, proto_w]`.
    pub protos: TensorDyn,
    /// Physical memory layout of the `protos` tensor.
    pub layout: ProtoLayout,
}

/// Invert a letterbox: map a [`BoundingBox`] normalized over the model input
/// back to normalized-over-the-crop, given the content bounds
/// `[lx0, ly0, lx1, ly1]`. The box is canonicalised first, a degenerate
/// (zero-span) letterbox axis maps with unit scale (no divide-by-zero), and the
/// result is clamped to `[0, 1]`.
///
/// This is the single home for the inverse-letterbox transform; both
/// `edgefirst_decoder::tiling`'s detection lift and `edgefirst_image`'s mask
/// path call it, which is why it belongs beside the type it transforms rather
/// than in either consumer.
#[must_use]
#[inline]
pub fn unletter_norm(b: BoundingBox, lb: [f32; 4]) -> BoundingBox {
    let b = b.to_canonical();
    let [lx0, ly0, lx1, ly1] = lb;
    let inv_w = if lx1 > lx0 { 1.0 / (lx1 - lx0) } else { 1.0 };
    let inv_h = if ly1 > ly0 { 1.0 / (ly1 - ly0) } else { 1.0 };
    BoundingBox {
        xmin: ((b.xmin - lx0) * inv_w).clamp(0.0, 1.0),
        ymin: ((b.ymin - ly0) * inv_h).clamp(0.0, 1.0),
        xmax: ((b.xmax - lx0) * inv_w).clamp(0.0, 1.0),
        ymax: ((b.ymax - ly0) * inv_h).clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounding_box_is_c_compatible() {
        // The C API exposes this layout; `#[repr(C)]` and the field order are
        // load-bearing, not stylistic.
        assert_eq!(std::mem::size_of::<BoundingBox>(), 16);
        assert_eq!(std::mem::align_of::<BoundingBox>(), 4);
    }

    #[test]
    fn bounding_box_round_trips_through_an_array() {
        let bb = BoundingBox::new(0.1, 0.2, 0.3, 0.4);
        let arr: [f32; 4] = bb.into();
        assert_eq!(BoundingBox::from(arr), bb);
    }

    #[test]
    fn unletter_norm_is_identity_for_a_full_frame_letterbox() {
        let bb = BoundingBox::new(0.25, 0.25, 0.75, 0.75);
        assert_eq!(unletter_norm(bb, [0.0, 0.0, 1.0, 1.0]), bb);
    }

    #[test]
    fn unletter_norm_canonicalises_and_clamps() {
        // Inverted input must come back canonical (xmin <= xmax) and inside
        // [0,1] -- the contract `image`'s mask path relies on.
        let got = unletter_norm(
            BoundingBox::new(0.9, 0.9, 0.1, 0.1),
            [0.0, 0.125, 1.0, 0.875],
        );
        assert!(got.xmin <= got.xmax && got.ymin <= got.ymax);
        for v in [got.xmin, got.ymin, got.xmax, got.ymax] {
            assert!((0.0..=1.0).contains(&v), "{v} escaped [0,1]");
        }
    }

    #[test]
    fn unletter_norm_survives_a_degenerate_letterbox_axis() {
        // A zero-span axis must map with unit scale rather than dividing by
        // zero and producing NaN/inf.
        let got = unletter_norm(BoundingBox::new(0.2, 0.2, 0.8, 0.8), [0.5, 0.0, 0.5, 1.0]);
        for v in [got.xmin, got.ymin, got.xmax, got.ymax] {
            assert!(v.is_finite(), "degenerate axis produced {v}");
        }
    }
}
