// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! ModelPack decode on the NEON tiers: the split-detection rows (`prim::neon_modelpack`) and the
//! segmentation map conversions and argmax (`prim::neon_seg`), each bit-identical to the scalar
//! code in `modelpack.rs` and `decoder/postprocess.rs`.

use std::mem::MaybeUninit;

use ndarray::{Array2, Array3, ArrayView3};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::{ParallelSlice, ParallelSliceMut};

use super::{Isa, Ran, PAR_MIN_SCORES};

/// Anchor rows of one ModelPack split output `(height, width, na·(5 + classes))`, from
/// `first_row`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) struct SplitRows<'a> {
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) anchors: &'a [[f32; 2]],
    pub(crate) classes: usize,
    pub(crate) first_row: usize,
}

/// The split decode of whole rows `src` (`5 + classes` values each) into `boxes` and `scores`
/// on `isa`, every element written unless this returns [`Ran::Scalar`]; see
/// `prim::neon_modelpack::SplitIn`.
pub(crate) fn split_rows_kernel<T: 'static>(
    isa: Isa,
    src: &[T],
    dequant: Option<(f32, Option<f32>)>,
    rows: &SplitRows<'_>,
    boxes: &mut [MaybeUninit<f32>],
    scores: &mut [MaybeUninit<f32>],
) -> Ran {
    if isa.is_scalar() {
        return Ran::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use edgefirst_kernels::prim::neon_modelpack::{SplitGrid, SplitIn};
        let grid = SplitGrid {
            width: rows.width,
            height: rows.height,
            anchors: rows.anchors,
            classes: rows.classes,
            first_row: rows.first_row,
        };
        macro_rules! run {
            ($($u:ty),*) => {$(
                if let Some(s) = super::cast::<T, $u>(src) {
                    let ran = <$u>::modelpack_split_rows_neon(s, dequant, &grid, boxes, scores);
                    return if ran { Ran::Neon } else { Ran::Scalar };
                }
            )*};
        }
        run!(i8, u8, i16, u16, f32, half::f16);
        Ran::Scalar
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (src, dequant, rows, boxes, scores);
        Ran::Scalar
    }
}

/// The boxes and scores of a ModelPack split decode on a NEON tier, built one output at a time.
pub(crate) struct SplitDecode {
    isa: Isa,
    classes: Option<usize>,
    boxes: Vec<f32>,
    scores: Vec<f32>,
}

impl SplitDecode {
    /// An empty decode with room for `outputs`, each `((height, width, channels), anchors)`, or
    /// `None` on the scalar tier and when [`Self::push`] would refuse an output's class count,
    /// checked before anything is reserved.
    pub(crate) fn new(isa: Isa, outputs: &[((usize, usize, usize), usize)]) -> Option<Self> {
        if isa.is_scalar() {
            return None;
        }
        let (mut classes, mut rows) = (None, 0usize);
        for &((h, w, c), na) in outputs {
            if na == 0 || !c.is_multiple_of(na) || c / na < 6 {
                return None;
            }
            let nc = c / na - 5;
            if *classes.get_or_insert(nc) != nc {
                return None;
            }
            rows = rows.checked_add(h.checked_mul(w)?.checked_mul(na)?)?;
        }
        Some(Self {
            isa,
            classes: None,
            boxes: Vec::with_capacity(rows.checked_mul(4)?),
            scores: Vec::with_capacity(rows.checked_mul(classes.unwrap_or(0))?),
        })
    }

    /// Decodes every anchor row of `output` `(height, width, na·(5 + nc))` (dequantized as
    /// `x · scale + bias` first when given) after the outputs before it; ranges of rows run on
    /// the rayon pool when the output is large. `false` when it cannot run here (not standard
    /// layout, a type without a kernel, a class count that differs from earlier outputs or
    /// does not divide the channels); nothing is then added.
    pub(crate) fn push<T: Copy + Send + Sync + 'static>(
        &mut self,
        output: ArrayView3<'_, T>,
        anchors: &[[f32; 2]],
        dequant: Option<(f32, Option<f32>)>,
    ) -> bool {
        let (height, width, channels) = output.dim();
        let na = anchors.len();
        let Some(src) = output.as_slice() else {
            return false;
        };
        if na == 0 || !channels.is_multiple_of(na) || channels / na < 6 {
            return false;
        }
        let classes = channels / na - 5;
        if *self.classes.get_or_insert(classes) != classes {
            return false;
        }
        let rows = height * width * na;
        let row = classes + 5;
        let (Some(nb), Some(ns)) = (rows.checked_mul(4), rows.checked_mul(classes)) else {
            return false;
        };
        // Already in capacity when `new` was given every output.
        self.boxes.reserve(nb);
        self.scores.reserve(ns);
        let (Some(boxes), Some(scores)) = (
            self.boxes.spare_capacity_mut().get_mut(..nb),
            self.scores.spare_capacity_mut().get_mut(..ns),
        ) else {
            return false;
        };
        let layout = |first_row| SplitRows {
            width,
            height,
            anchors,
            classes,
            first_row,
        };
        let isa = self.isa;
        let ran = if src.len() < PAR_MIN_SCORES {
            split_rows_kernel(isa, src, dequant, &layout(0), boxes, scores) != Ran::Scalar
        } else {
            let piece = rows.div_ceil(2 * rayon::current_num_threads()).max(1);
            src.par_chunks(piece * row)
                .zip(boxes.par_chunks_mut(piece * 4))
                .zip(scores.par_chunks_mut(piece * classes))
                .enumerate()
                .all(|(k, ((s, b), c))| {
                    split_rows_kernel(isa, s, dequant, &layout(k * piece), b, c) != Ran::Scalar
                })
        };
        if !ran {
            return false;
        }
        // SAFETY: the kernels wrote every one of the `nb` and `ns` reserved elements.
        unsafe {
            self.boxes.set_len(self.boxes.len() + nb);
            self.scores.set_len(self.scores.len() + ns);
        }
        true
    }

    /// The boxes `(N, 4)` and scores `(N, nc)` of every output pushed.
    pub(crate) fn finish(self) -> Option<(Array2<f32>, Array2<f32>)> {
        let classes = self.classes?;
        let n = self.boxes.len() / 4;
        Some((
            Array2::from_shape_vec((n, 4), self.boxes).ok()?,
            Array2::from_shape_vec((n, classes), self.scores).ok()?,
        ))
    }
}

/// Whether `a` is planar `[C, H, W]` memory seen as `(H, W, C)` (a ModelPack NCHW output), not
/// standard layout.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn is_planar<S: ndarray::Data>(a: &ndarray::ArrayBase<S, ndarray::Ix3>) -> bool {
    let (h, w, _) = a.dim();
    !a.is_standard_layout() && a.strides() == [w as isize, 1, (h * w) as isize]
}

/// The standard-layout `(H, W, C)` array of planar `[C, H, W]` memory `src`, by the NEON
/// transpose.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn from_planar(src: &[u8], (h, w, c): (usize, usize, usize)) -> Option<Array3<u8>> {
    let mut out = vec![0u8; src.len()];
    edgefirst_kernels::convert::nchw_to_nhwc(src, h, w, c, &mut out);
    Array3::from_shape_vec((h, w, c), out).ok()
}

/// `a` in standard layout when it is planar, so the mask tensor copies it whole; otherwise `a`.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn standard_layout(a: Array3<u8>) -> Array3<u8> {
    if !is_planar(&a) {
        return a;
    }
    match a
        .as_slice_memory_order()
        .and_then(|s| from_planar(s, a.dim()))
    {
        Some(o) => o,
        None => a,
    }
}

/// The top byte of each integer score of `seg` (sign bit flipped for signed types), in
/// standard layout when `seg` is planar and otherwise as `map` lays it out (a planar `u8` map
/// is only transposed); `None` when the caller maps it (scalar tier, other layouts, a `u8` map
/// that is not planar).
pub(crate) fn seg_to_u8_array<T: 'static>(isa: Isa, seg: &ArrayView3<'_, T>) -> Option<Array3<u8>> {
    if isa.is_scalar() {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use super::map_array;
        use edgefirst_kernels::prim::neon_seg::SegToU8;
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<u8>() {
            if !is_planar(seg) {
                return None;
            }
            let src = super::cast::<T, u8>(seg.as_slice_memory_order()?)?;
            return from_planar(src, seg.dim());
        }
        macro_rules! run {
            ($($u:ty),*) => {$(
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$u>() {
                    return map_array(seg, |s, d| {
                        super::cast::<T, $u>(s)
                            .is_some_and(|s| <$u>::seg_to_u8_neon(s, d))
                    })
                    .map(standard_layout);
                }
            )*};
        }
        run!(i8, u16, i16, u32, i32);
        None
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = seg;
        None
    }
}

/// `((x − min) / (max − min) · 255) as u8` of each float score of `seg` over its minimum and
/// maximum (0 and 255 when one is NaN), in standard layout when `seg` is planar and otherwise
/// as `map` lays it out, with the tier that ran; `None` when the caller computes it (scalar
/// tier, `f64`, other layouts).
pub(crate) fn seg_normalize_array<T: 'static>(
    isa: Isa,
    seg: &ArrayView3<'_, T>,
) -> Option<(Array3<u8>, Ran)> {
    if isa.is_scalar() {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use super::map_array;
        use edgefirst_kernels::prim::neon_seg::{seg_normalize_f16_neon_fp16, SegNormalize};
        if isa.fp16() && std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::f16>() {
            let out = map_array(seg, |s, d| {
                // SAFETY: `Isa::of` saw FEAT_FP16.
                super::cast::<T, half::f16>(s)
                    .is_some_and(|s| unsafe { seg_normalize_f16_neon_fp16(s, d) })
            });
            if let Some(out) = out {
                return Some((standard_layout(out), Ran::NeonFp16));
            }
        }
        macro_rules! run {
            ($($u:ty),*) => {$(
                if std::any::TypeId::of::<T>() == std::any::TypeId::of::<$u>() {
                    let out = map_array(seg, |s, d| {
                        super::cast::<T, $u>(s)
                            .is_some_and(|s| <$u>::seg_normalize_neon(s, d))
                    });
                    return out.map(|o| (standard_layout(o), Ran::Neon));
                }
            )*};
        }
        run!(f32, half::f16);
        None
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = seg;
        None
    }
}

/// For each pixel of `seg` (`channels` scores per pixel, pixel-major), the index of its first
/// largest score as `u8`; `None` when the caller computes it (scalar tier).
pub(crate) fn channel_argmax(isa: Isa, seg: &[u8], channels: usize) -> Option<Vec<u8>> {
    if isa.is_scalar() || channels == 0 {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    {
        let n = seg.len() / channels;
        let mut out: Vec<u8> = Vec::with_capacity(n);
        let ran = edgefirst_kernels::prim::neon_seg::channel_argmax_u8_neon(
            seg,
            channels,
            out.spare_capacity_mut().get_mut(..n)?,
        );
        if !ran {
            return None;
        }
        // SAFETY: the kernel wrote all `n` elements, within the capacity.
        unsafe { out.set_len(n) };
        Some(out)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = seg;
        None
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
