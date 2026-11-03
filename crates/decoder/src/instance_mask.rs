// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! CPU tiers of the materialised instance masks (`Decoder::decode` with protos), the proto
//! dequantize of `decode_proto`, the quantized-output conversions and the mask threshold.
//!
//! The tier is [`Isa`], resolved per call from `EDGEFIRST_DECODER_FORCE_KERNEL`: decode paths
//! fail on a forced tier the CPU lacks, `segmentation_to_mask`'s threshold runs scalar. On NEON
//! tiers masks run `edgefirst_kernels::prim::neon_mask`: `i8`/`u8`/`i16`/`u16` coefficients and
//! protos (zero points in their type's range, at most `MAX_K` channels) take the exact integer
//! dot product (`sdot` when the tier has FEAT_DotProd), so every logit equals the scalar tier's;
//! `f32`/`f16` take the f32 FMA dot product. A fused NEON sigmoid rounds to `u8` within 1 of
//! the scalar tier.
//! 32-bit integers, `f64`, proto strides other than NHWC and NCHW, and the scalar tier run the
//! scalar code. Dequantize and threshold conversions are bit-identical on every tier.

use ndarray::{Array, Array2, ArrayView, ArrayView3, Dimension, ShapeBuilder};
use num_traits::AsPrimitive;

#[cfg(target_arch = "aarch64")]
use crate::neon_dispatch::cast;
pub(crate) use crate::neon_dispatch::Isa;
use crate::Quantization;

#[cfg(target_arch = "aarch64")]
mod neon;
#[cfg(target_arch = "aarch64")]
pub(crate) use neon::{FloatPlan, QuantPlan};

#[cfg(not(target_arch = "aarch64"))]
mod portable;
#[cfg(not(target_arch = "aarch64"))]
pub(crate) use portable::{FloatPlan, QuantPlan};

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;

/// Proto-grid cells `y0..y1` × `x0..x1` under a detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Crop {
    pub(crate) x0: usize,
    pub(crate) y0: usize,
    pub(crate) x1: usize,
    pub(crate) y1: usize,
}

/// The integer kernel a quantized mask call runs.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum QuantKernel {
    Scalar,
    Dot8Nhwc,
    Dot8NhwcDotprod,
    Dot8Nchw,
    WideNhwc,
    WideNchw,
}

/// The float kernel a float mask call runs.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FloatKernel {
    Scalar,
    F32Nhwc,
    F32Nchw,
}

/// How a proto tensor's `[H, W, K]` view lies in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Layout {
    /// Standard `[H, W, K]`.
    Nhwc,
    /// `[K, H, W]` memory seen through swapped strides.
    Nchw,
}

/// The proto memory and its layout, for the two layouts the NEON paths read.
pub(crate) fn proto_memory<'a, T>(protos: &'a ArrayView3<'_, T>) -> Option<(&'a [T], Layout)> {
    let (h, w, _) = protos.dim();
    if protos.is_standard_layout() {
        protos.as_slice().map(|s| (s, Layout::Nhwc))
    } else if protos.strides() == [w as isize, 1, (h * w) as isize] {
        protos.as_slice_memory_order().map(|s| (s, Layout::Nchw))
    } else {
        None
    }
}

/// The kernel a quantized mask call with these inputs runs on `isa`.
#[cfg(test)]
pub(crate) fn quant_kernel<MASK: 'static, PROTO: 'static>(
    isa: Isa,
    protos: &ArrayView3<'_, PROTO>,
    quant_masks: Quantization,
    quant_protos: Quantization,
) -> QuantKernel {
    QuantPlan::new::<MASK, PROTO>(isa, protos, quant_masks, quant_protos)
        .map_or(QuantKernel::Scalar, |p| p.kernel::<MASK>())
}

/// The kernel a float mask call with these protos runs on `isa`.
#[cfg(test)]
pub(crate) fn float_kernel<PROTO: 'static>(
    isa: Isa,
    protos: &ArrayView3<'_, PROTO>,
) -> FloatKernel {
    FloatPlan::new::<PROTO>(isa, protos).map_or(FloatKernel::Scalar, |p| p.kernel())
}

/// Whether [`dequant_slice`] has a NEON kernel for `T`.
fn neon_dequant_type<T: 'static>() -> bool {
    cfg!(target_arch = "aarch64")
        && [
            std::any::TypeId::of::<i8>(),
            std::any::TypeId::of::<u8>(),
            std::any::TypeId::of::<i16>(),
            std::any::TypeId::of::<u16>(),
        ]
        .contains(&std::any::TypeId::of::<T>())
}

/// `(x − zero_point) · scale` of contiguous `src` into `dst` on a NEON tier, bit for bit; false
/// when `T` or `isa` needs the scalar loop.
fn dequant_slice<T: 'static>(isa: Isa, src: &[T], q: Quantization, dst: &mut [f32]) -> bool {
    if isa.is_scalar() || src.len() != dst.len() {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use edgefirst_kernels::prim::neon_convert as nc;
        let q = edgefirst_kernels::Quant::new(q.scale, q.zero_point);
        // SAFETY (every arm): NEON is part of the aarch64 baseline.
        if let Some(s) = cast::<T, i8>(src) {
            unsafe { nc::dequant_i8_to_f32_exact_neon(s, q, dst) };
        } else if let Some(s) = cast::<T, u8>(src) {
            unsafe { nc::dequant_u8_to_f32_exact_neon(s, q, dst) };
        } else if let Some(s) = cast::<T, i16>(src) {
            unsafe { nc::dequant_i16_to_f32_exact_neon(s, q, dst) };
        } else if let Some(s) = cast::<T, u16>(src) {
            unsafe { nc::dequant_u16_to_f32_exact_neon(s, q, dst) };
        } else {
            return false;
        }
        true
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = q;
        false
    }
}

/// `(x − zero_point) · scale` of NCHW `src` into NHWC `dst` on a NEON tier, bit for bit; false
/// when `T` or `isa` needs the scalar loop.
fn dequant_nchw<T: 'static>(
    isa: Isa,
    src: &[T],
    (h, w, k): (usize, usize, usize),
    q: Quantization,
    dst: &mut [f32],
) -> bool {
    if isa.is_scalar() {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use edgefirst_kernels::prim::neon_mask as nm;
        let q = edgefirst_kernels::Quant::new(q.scale, q.zero_point);
        // SAFETY (every arm): NEON is part of the aarch64 baseline.
        if let Some(s) = cast::<T, i8>(src) {
            unsafe { nm::dequant_nchw_to_nhwc_i8_neon(s, h, w, k, q, dst) }
        } else if let Some(s) = cast::<T, u8>(src) {
            unsafe { nm::dequant_nchw_to_nhwc_u8_neon(s, h, w, k, q, dst) }
        } else if let Some(s) = cast::<T, i16>(src) {
            unsafe { nm::dequant_nchw_to_nhwc_i16_neon(s, h, w, k, q, dst) }
        } else if let Some(s) = cast::<T, u16>(src) {
            unsafe { nm::dequant_nchw_to_nhwc_u16_neon(s, h, w, k, q, dst) }
        } else {
            false
        }
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (src, h, w, k, q, dst);
        false
    }
}

/// `(x − zero_point) · scale` of every element of `view`, with `map`'s layout and bits.
pub(crate) fn dequant_array<T, D>(
    view: ArrayView<'_, T, D>,
    q: Quantization,
    isa: Isa,
) -> Array<f32, D>
where
    T: Copy + AsPrimitive<f32>,
    D: Dimension,
{
    let neon =
        !isa.is_scalar() && neon_dequant_type::<T>() && view.strides().iter().all(|&s| s >= 0);
    if let (true, Some(src)) = (neon, view.as_slice_memory_order()) {
        let mut out = vec![0f32; src.len()];
        if dequant_slice(isa, src, q, &mut out) {
            let mut strides = view.raw_dim();
            for (d, &s) in strides.slice_mut().iter_mut().zip(view.strides()) {
                *d = s as usize;
            }
            if let Ok(a) = Array::from_shape_vec(view.raw_dim().strides(strides), out) {
                return a;
            }
        }
    }
    view.map(|v| {
        let val: f32 = v.as_();
        (val - q.zero_point as f32) * q.scale
    })
}

/// `(x − zero_point) · scale` of `protos` into NHWC `dst` on a NEON tier, bit for bit; false
/// when the type, layout or `isa` needs the scalar copy.
pub(crate) fn dequant_protos<T: 'static>(
    isa: Isa,
    protos: &ArrayView3<'_, T>,
    q: Quantization,
    dst: &mut [f32],
) -> bool {
    match proto_memory(protos) {
        Some((src, Layout::Nhwc)) => dequant_slice(isa, src, q, dst),
        Some((src, Layout::Nchw)) => dequant_nchw(isa, src, protos.dim(), q, dst),
        None => false,
    }
}

/// `1` where an `(H, W, 1)` segmentation is at least `threshold`, else `0`, on a NEON tier;
/// `None` when `isa` or the layout needs the scalar map.
pub(crate) fn threshold_mask(
    isa: Isa,
    segmentation: &ArrayView3<'_, u8>,
    threshold: u8,
) -> Option<Array2<u8>> {
    if isa.is_scalar() {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    {
        let (h, w, _) = segmentation.dim();
        let src = segmentation.as_slice()?;
        let mut out = vec![0u8; src.len()];
        // SAFETY: NEON is part of the aarch64 baseline.
        let ok = unsafe {
            edgefirst_kernels::prim::neon_mask::mask_threshold_u8_neon(src, threshold, &mut out)
        };
        ok.then(|| Array2::from_shape_vec((h, w), out).ok())?
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (segmentation, threshold);
        None
    }
}
