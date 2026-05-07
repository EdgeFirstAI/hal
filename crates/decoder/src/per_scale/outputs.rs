// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Output buffer storage and zero-copy view types for the per-scale decoder.
//!
//! The decoder owns three flat output `Vec<F>`s (boxes, scores,
//! mask_coefs) and an optional `Array4<F>` for protos. They're
//! pre-allocated at first frame and overwritten in place each subsequent
//! frame. Callers receive `DecodedOutputsRef<'_>` views borrowed from
//! the decoder.

use super::kernels::dispatch::DstSliceMut;
use super::DecodeDtype;
use half::f16;
use ndarray::{Array4, ArrayView2, ArrayView4};

/// Owned output buffers, reused frame-to-frame.
#[derive(Debug)]
#[allow(dead_code)] // Wired by Task 21.
pub(crate) struct DecodedOutputBuffers {
    pub(crate) boxes: Buffer,
    pub(crate) scores: Buffer,
    pub(crate) mask_coefs: Option<Buffer>,
    pub(crate) protos: Option<ProtoStorage>,

    /// Per-frame transpose scratch for NCHW children. Populated lazily
    /// by `pipeline::run` when at least one level is declared NCHW
    /// (otherwise stays default-empty and costs nothing). Per-dtype
    /// fields each grow on first use to `max(h * w * c)` across all
    /// per-scale levels and roles for that dtype, then are reused for
    /// every subsequent frame.
    ///
    /// The fields are public to the pipeline orchestration. Callers
    /// must use the `ensure_*` methods to size them — direct mutation
    /// risks under-sized allocations and out-of-bounds writes inside
    /// the transpose helper.
    pub(crate) layout_scratch: LayoutScratch,
}

/// Per-dtype scratch for NCHW → NHWC transpose. Lazy allocation: the
/// common case where all per-scale children share one dtype (e.g. all
/// int8 from a TFLite or NPU export) populates exactly one field.
#[derive(Debug, Default)]
#[allow(dead_code)]
pub(crate) struct LayoutScratch {
    pub(crate) i8: Vec<i8>,
    pub(crate) u8: Vec<u8>,
    pub(crate) i16: Vec<i16>,
    pub(crate) u16: Vec<u16>,
    pub(crate) f16: Vec<f16>,
    pub(crate) f32: Vec<f32>,
}

impl LayoutScratch {
    /// Grow the i8 scratch to at least `n` elements; return the prefix
    /// of length `n`. Backed by `Vec::resize`, which keeps existing
    /// elements and zero-initializes any new tail.
    #[allow(dead_code)]
    pub(crate) fn ensure_i8(&mut self, n: usize) -> &mut [i8] {
        if self.i8.len() < n {
            self.i8.resize(n, 0);
        }
        &mut self.i8[..n]
    }

    #[allow(dead_code)]
    pub(crate) fn ensure_u8(&mut self, n: usize) -> &mut [u8] {
        if self.u8.len() < n {
            self.u8.resize(n, 0);
        }
        &mut self.u8[..n]
    }

    #[allow(dead_code)]
    pub(crate) fn ensure_i16(&mut self, n: usize) -> &mut [i16] {
        if self.i16.len() < n {
            self.i16.resize(n, 0);
        }
        &mut self.i16[..n]
    }

    #[allow(dead_code)]
    pub(crate) fn ensure_u16(&mut self, n: usize) -> &mut [u16] {
        if self.u16.len() < n {
            self.u16.resize(n, 0);
        }
        &mut self.u16[..n]
    }

    #[allow(dead_code)]
    pub(crate) fn ensure_f16(&mut self, n: usize) -> &mut [f16] {
        if self.f16.len() < n {
            self.f16.resize(n, f16::ZERO);
        }
        &mut self.f16[..n]
    }

    #[allow(dead_code)]
    pub(crate) fn ensure_f32(&mut self, n: usize) -> &mut [f32] {
        if self.f32.len() < n {
            self.f32.resize(n, 0.0);
        }
        &mut self.f32[..n]
    }
}

/// Dtype-tagged owned buffer.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum Buffer {
    F32(Vec<f32>),
    F16(Vec<f16>),
}

/// Dtype-tagged owned proto storage. Layout NHWC `[1, H_p, W_p, NM]`.
#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum ProtoStorage {
    F32(Array4<f32>),
    F16(Array4<f16>),
}

impl DecodedOutputBuffers {
    /// Allocate buffers sized for the plan.
    #[allow(dead_code)] // Wired by Task 21.
    pub(crate) fn new(
        out_dtype: DecodeDtype,
        total_anchors: usize,
        num_classes: usize,
        num_mask_coefs: usize,
        proto_shape: Option<&[usize]>,
    ) -> Self {
        let alloc_buf = |n: usize| match out_dtype {
            DecodeDtype::F32 => Buffer::F32(vec![0.0; n]),
            DecodeDtype::F16 => Buffer::F16(vec![f16::ZERO; n]),
        };
        let alloc_proto = |s: &[usize]| -> Option<ProtoStorage> {
            if s.len() != 4 {
                return None;
            }
            Some(match out_dtype {
                DecodeDtype::F32 => ProtoStorage::F32(Array4::zeros((s[0], s[1], s[2], s[3]))),
                DecodeDtype::F16 => {
                    ProtoStorage::F16(Array4::from_elem((s[0], s[1], s[2], s[3]), f16::ZERO))
                }
            })
        };
        Self {
            boxes: alloc_buf(4 * total_anchors),
            scores: alloc_buf(num_classes * total_anchors),
            mask_coefs: (num_mask_coefs > 0).then(|| alloc_buf(num_mask_coefs * total_anchors)),
            protos: proto_shape.and_then(alloc_proto),
            layout_scratch: LayoutScratch::default(),
        }
    }

    /// Get a mutable dispatch slice for the boxes buffer.
    #[allow(dead_code)]
    pub(crate) fn boxes_dst(&mut self) -> DstSliceMut<'_> {
        match &mut self.boxes {
            Buffer::F32(v) => DstSliceMut::F32(v),
            Buffer::F16(v) => DstSliceMut::F16(v),
        }
    }

    /// Mutable dispatch slice for one level's anchor range in the boxes buffer.
    /// `level_start` and `level_len` are in anchor units (×4 for the boxes buffer).
    #[allow(dead_code)]
    pub(crate) fn boxes_level_slice(
        &mut self,
        level_start: usize,
        level_len: usize,
    ) -> DstSliceMut<'_> {
        let start = level_start * 4;
        let end = start + level_len * 4;
        match &mut self.boxes {
            Buffer::F32(v) => DstSliceMut::F32(&mut v[start..end]),
            Buffer::F16(v) => DstSliceMut::F16(&mut v[start..end]),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn scores_level_slice(
        &mut self,
        level_start: usize,
        level_len: usize,
        num_classes: usize,
    ) -> DstSliceMut<'_> {
        let start = level_start * num_classes;
        let end = start + level_len * num_classes;
        match &mut self.scores {
            Buffer::F32(v) => DstSliceMut::F32(&mut v[start..end]),
            Buffer::F16(v) => DstSliceMut::F16(&mut v[start..end]),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn mask_coefs_level_slice(
        &mut self,
        level_start: usize,
        level_len: usize,
        num_mc: usize,
    ) -> Option<DstSliceMut<'_>> {
        let start = level_start * num_mc;
        let end = start + level_len * num_mc;
        match self.mask_coefs.as_mut()? {
            Buffer::F32(v) => Some(DstSliceMut::F32(&mut v[start..end])),
            Buffer::F16(v) => Some(DstSliceMut::F16(&mut v[start..end])),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn protos_dst(&mut self) -> Option<DstSliceMut<'_>> {
        match self.protos.as_mut()? {
            ProtoStorage::F32(a) => a.as_slice_mut().map(DstSliceMut::F32),
            ProtoStorage::F16(a) => a.as_slice_mut().map(DstSliceMut::F16),
        }
    }

    /// Snapshot the pre-NMS buffers as owned f32 ndarrays.
    ///
    /// Used by [`crate::Decoder::_testing_run_per_scale_pre_nms`].
    /// Widens f16 buffers to f32 if the decoder was built with
    /// [`crate::per_scale::DecodeDtype::F16`].
    #[doc(hidden)]
    pub(crate) fn snapshot_owned_f32(
        &self,
        total_anchors: usize,
        num_classes: usize,
        num_mc: usize,
    ) -> crate::per_scale::PreNmsCapture {
        let widen = |buf: &Buffer, n: usize, k: usize| -> ndarray::Array2<f32> {
            match buf {
                Buffer::F32(v) => ndarray::Array2::from_shape_vec((n, k), v[..n * k].to_vec())
                    .expect("f32 reshape"),
                Buffer::F16(v) => {
                    let widened: Vec<f32> = v[..n * k].iter().map(|h| h.to_f32()).collect();
                    ndarray::Array2::from_shape_vec((n, k), widened).expect("f16 reshape")
                }
            }
        };
        let boxes_xywh = widen(&self.boxes, total_anchors, 4);
        let scores = widen(&self.scores, total_anchors, num_classes);
        let mask_coefs = self
            .mask_coefs
            .as_ref()
            .map(|b| widen(b, total_anchors, num_mc));
        let protos = self.protos.as_ref().map(|p| match p {
            ProtoStorage::F32(arr) => arr.clone(),
            ProtoStorage::F16(arr) => arr.mapv(|h| h.to_f32()),
        });
        crate::per_scale::PreNmsCapture {
            boxes_xywh,
            scores,
            mask_coefs,
            protos,
        }
    }
}

// Free helpers that take a single field by mutable reference instead of
// `&mut DecodedOutputBuffers`. Used by the pipeline orchestrator when it
// needs to hold a destination slice and a scratch slice simultaneously
// (the borrow checker tracks disjoint field borrows on direct field
// accesses, but the `&mut self` methods above hold the entire struct).
//
// Logic is identical to the same-named methods on `DecodedOutputBuffers`.

/// `boxes_level_slice` for use with disjoint field borrows.
#[allow(dead_code)]
pub(crate) fn boxes_level_slice_of(
    buffer: &mut Buffer,
    level_start: usize,
    level_len: usize,
) -> DstSliceMut<'_> {
    let start = level_start * 4;
    let end = start + level_len * 4;
    match buffer {
        Buffer::F32(v) => DstSliceMut::F32(&mut v[start..end]),
        Buffer::F16(v) => DstSliceMut::F16(&mut v[start..end]),
    }
}

/// `scores_level_slice` for use with disjoint field borrows.
#[allow(dead_code)]
pub(crate) fn scores_level_slice_of(
    buffer: &mut Buffer,
    level_start: usize,
    level_len: usize,
    num_classes: usize,
) -> DstSliceMut<'_> {
    let start = level_start * num_classes;
    let end = start + level_len * num_classes;
    match buffer {
        Buffer::F32(v) => DstSliceMut::F32(&mut v[start..end]),
        Buffer::F16(v) => DstSliceMut::F16(&mut v[start..end]),
    }
}

/// `mask_coefs_level_slice` for use with disjoint field borrows.
#[allow(dead_code)]
pub(crate) fn mask_coefs_level_slice_of(
    buffer: &mut Buffer,
    level_start: usize,
    level_len: usize,
    num_mc: usize,
) -> DstSliceMut<'_> {
    let start = level_start * num_mc;
    let end = start + level_len * num_mc;
    match buffer {
        Buffer::F32(v) => DstSliceMut::F32(&mut v[start..end]),
        Buffer::F16(v) => DstSliceMut::F16(&mut v[start..end]),
    }
}

#[cfg(any())]
mod _doc_only {
    //! marker so the impl block above doesn't accidentally close prematurely.

    /// Snapshot the pre-NMS buffers as owned f32 ndarrays.
    ///
    /// Used by [`crate::Decoder::_testing_run_per_scale_pre_nms`].
    /// Widens f16 buffers to f32 if the decoder was built with
    /// [`crate::per_scale::DecodeDtype::F16`].
    #[doc(hidden)]
    pub(crate) fn snapshot_owned_f32(
        &self,
        total_anchors: usize,
        num_classes: usize,
        num_mc: usize,
    ) -> crate::per_scale::PreNmsCapture {
        let widen = |buf: &Buffer, n: usize, k: usize| -> ndarray::Array2<f32> {
            match buf {
                Buffer::F32(v) => ndarray::Array2::from_shape_vec((n, k), v[..n * k].to_vec())
                    .expect("f32 reshape"),
                Buffer::F16(v) => {
                    let widened: Vec<f32> = v[..n * k].iter().map(|h| h.to_f32()).collect();
                    ndarray::Array2::from_shape_vec((n, k), widened).expect("f16 reshape")
                }
            }
        };
        let boxes_xywh = widen(&self.boxes, total_anchors, 4);
        let scores = widen(&self.scores, total_anchors, num_classes);
        let mask_coefs = self
            .mask_coefs
            .as_ref()
            .map(|b| widen(b, total_anchors, num_mc));
        let protos = self.protos.as_ref().map(|p| match p {
            ProtoStorage::F32(arr) => arr.clone(),
            ProtoStorage::F16(arr) => arr.mapv(|h| h.to_f32()),
        });
        crate::per_scale::PreNmsCapture {
            boxes_xywh,
            scores,
            mask_coefs,
            protos,
        }
    }
}

/// Zero-copy view returned to the caller; borrows from the decoder's buffers.
#[derive(Debug)]
#[allow(dead_code)] // Consumed by Task 24's decoder integration.
pub(crate) struct DecodedOutputsRef<'a> {
    pub(crate) boxes: BufferRef<'a>,
    pub(crate) scores: BufferRef<'a>,
    pub(crate) mask_coefs: Option<BufferRef<'a>>,
    pub(crate) protos: Option<ProtosView<'a>>,
    pub(crate) total_anchors: usize,
    pub(crate) num_classes: usize,
    pub(crate) num_mask_coefs: usize,
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum BufferRef<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
}

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) enum ProtosView<'a> {
    F32(ArrayView4<'a, f32>),
    F16(ArrayView4<'a, f16>),
}

impl<'a> BufferRef<'a> {
    /// View as an `(N, cols)` 2-D array — used by the existing NMS path.
    #[allow(dead_code)]
    pub(crate) fn as_array_view2_f32(
        &self,
        rows: usize,
        cols: usize,
    ) -> Option<ArrayView2<'a, f32>> {
        if let BufferRef::F32(s) = self {
            ArrayView2::from_shape((rows, cols), s).ok()
        } else {
            None
        }
    }

    #[allow(dead_code)]
    pub(crate) fn as_array_view2_f16(
        &self,
        rows: usize,
        cols: usize,
    ) -> Option<ArrayView2<'a, f16>> {
        if let BufferRef::F16(s) = self {
            ArrayView2::from_shape((rows, cols), s).ok()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_alloc_f32_correct_sizes() {
        let b = DecodedOutputBuffers::new(DecodeDtype::F32, 8400, 80, 32, Some(&[1, 160, 160, 32]));
        match &b.boxes {
            Buffer::F32(v) => assert_eq!(v.len(), 4 * 8400),
            _ => panic!(),
        }
        match &b.scores {
            Buffer::F32(v) => assert_eq!(v.len(), 80 * 8400),
            _ => panic!(),
        }
        match b.mask_coefs.as_ref().unwrap() {
            Buffer::F32(v) => assert_eq!(v.len(), 32 * 8400),
            _ => panic!(),
        }
        match b.protos.as_ref().unwrap() {
            ProtoStorage::F32(a) => assert_eq!(a.shape(), &[1, 160, 160, 32]),
            _ => panic!(),
        }
    }

    #[test]
    fn buffers_alloc_f16_correct_sizes() {
        let b = DecodedOutputBuffers::new(DecodeDtype::F16, 100, 80, 0, None);
        match &b.boxes {
            Buffer::F16(v) => assert_eq!(v.len(), 400),
            _ => panic!(),
        }
        match &b.scores {
            Buffer::F16(v) => assert_eq!(v.len(), 8000),
            _ => panic!(),
        }
        assert!(b.mask_coefs.is_none());
        assert!(b.protos.is_none());
    }

    #[test]
    fn buffers_detection_only_no_mc() {
        let b = DecodedOutputBuffers::new(DecodeDtype::F32, 100, 80, 0, None);
        assert!(b.mask_coefs.is_none());
        assert!(b.protos.is_none());
    }

    #[test]
    fn boxes_level_slice_carves_disjoint_ranges() {
        let mut b = DecodedOutputBuffers::new(DecodeDtype::F32, 100, 80, 0, None);
        // First level (anchor_offset=0, h*w=64)
        let s1 = b.boxes_level_slice(0, 64);
        if let DstSliceMut::F32(v) = s1 {
            assert_eq!(v.len(), 4 * 64);
            v[0] = 1.0;
        } else {
            panic!()
        }
        // Second level (anchor_offset=64, h*w=36)
        let s2 = b.boxes_level_slice(64, 36);
        if let DstSliceMut::F32(v) = s2 {
            assert_eq!(v.len(), 4 * 36);
            v[0] = 2.0;
        } else {
            panic!()
        }
        // Verify both writes preserved
        match &b.boxes {
            Buffer::F32(v) => {
                assert!((v[0] - 1.0).abs() < 1e-9);
                assert!((v[256] - 2.0).abs() < 1e-9); // 4 * 64
            }
            _ => panic!(),
        }
    }

    #[test]
    fn buffer_ref_array_view2_f32() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let r = BufferRef::F32(&v);
        let a = r.as_array_view2_f32(2, 2).unwrap();
        assert_eq!(a[[0, 0]], 1.0);
        assert_eq!(a[[1, 1]], 4.0);
        assert!(r.as_array_view2_f16(2, 2).is_none()); // wrong dtype
    }
}
