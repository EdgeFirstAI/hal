// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The decoder's CPU tier and its typed dispatch to the shared NEON kernels in
//! `edgefirst_kernels::prim`, for the YOLO candidate filters, the generic dequantize, NMS,
//! ModelPack decode, semantic segmentation decode and the instance masks (`instance_mask`).
//!
//! [`Isa`] comes from `EDGEFIRST_DECODER_FORCE_KERNEL`: `Decoder::decode*` paths resolve it with
//! [`Isa::current`], which fails with `ForcedKernelUnavailable` for a forced tier the CPU lacks or
//! an unknown name, and pass it down; the public infallible helpers (`dequantize_cpu*`,
//! `dequantize_ndarray`, `postprocess_boxes_*`, `nms_*`, `jaccard_batch4`, `segmentation_to_mask`)
//! use [`Isa::current_or_scalar`], which runs the scalar code instead. On NEON tiers `i8`, `u8`,
//! `i16`, `u16`, `f32` and `f16` scores take `prim::neon_candidates` (argmax over class columns or
//! a contiguous row, and threshold positions), and `i8`/`u8`/`i16`/`u16` → `f32`/`f16` dequantize
//! takes the `scale_bias` kernels; every kernel equals the scalar code bit for bit. The `*_kernel`
//! functions report the kernel family that ran, [`Ran::Scalar`] meaning the caller runs its scalar
//! code. The candidate scans shared by the float and quantized filters also live here, and the NMS
//! (`nms`), ModelPack (`modelpack`) and semantic segmentation (`semantic`) dispatch in submodules.

use edgefirst_kernels::CpuFeatures;
use ndarray::{Array, ArrayView, ArrayView1, ArrayView2, Dimension, ShapeBuilder};
use num_traits::AsPrimitive;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use rayon::slice::ParallelSliceMut;
use std::mem::MaybeUninit;

use crate::per_scale::kernels::decoder_cpu_features;
use crate::DecoderResult;

mod modelpack;
mod nms;
pub(crate) mod semantic;

pub(crate) use modelpack::{channel_argmax, seg_normalize_array, seg_to_u8_array, SplitDecode};
pub(crate) use nms::{jaccard4_kernel, keep_unflagged, nms_dead, nms_isa, sort_by_score_desc};

/// Candidate filters over fewer score elements than this run on the calling thread: handing a
/// few hundred rows to rayon costs more CPU, in waking and then spinning workers, than filtering
/// them.
pub(crate) const PAR_MIN_SCORES: usize = 1 << 16;

/// [`PAR_MIN_SCORES`] of the row-major float candidate filters on a NEON tier, where a row
/// costs tens of nanoseconds and only a dense frame repays the pool.
pub(crate) const PAR_MIN_ROW_SCORES: usize = 1 << 19;

/// Score rows per rayon task in the row-major candidate scans: one row's argmax is too little
/// work to split further.
pub(crate) const ROWS_PER_TASK: usize = 64;

/// Anchor ranges of the parallel column argmax start on multiples of this, a whole number of
/// register tiles for every score type.
const COLUMN_PIECE_ALIGN: usize = 64;

/// The instruction set a decode, filter, dequantize or mask call runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Isa {
    neon: bool,
    fp16: bool,
    dotprod: bool,
    f16_nms: bool,
}

impl Isa {
    /// The scalar code.
    pub(crate) const SCALAR: Self = Self {
        neon: false,
        fp16: false,
        dotprod: false,
        f16_nms: false,
    };

    /// The best instruction set `f` allows on this CPU.
    pub(crate) fn of(f: &CpuFeatures) -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                neon: f.neon_baseline,
                fp16: f.fp16_usable(),
                dotprod: f.dotprod_usable(),
                f16_nms: false,
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            let _ = f;
            Self::SCALAR
        }
    }

    /// The tier `EDGEFIRST_DECODER_FORCE_KERNEL` selects; an unavailable or unknown forced tier
    /// is `ForcedKernelUnavailable`.
    pub(crate) fn current() -> DecoderResult<Self> {
        Ok(Self::of(&decoder_cpu_features()?))
    }

    /// [`Self::current`] for the public infallible helpers, which run the scalar code on an
    /// unavailable or unknown forced tier.
    pub(crate) fn current_or_scalar() -> Self {
        decoder_cpu_features().map_or(Self::SCALAR, |f| Self::of(&f))
    }

    /// Whether only the scalar code runs.
    pub(crate) fn is_scalar(self) -> bool {
        !self.neon
    }

    /// Whether the FP16 kernels may run.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) fn fp16(self) -> bool {
        self.fp16
    }

    /// `self` with the NMS overlap tests in FP16 when the FP16 kernels may run.
    pub(crate) fn with_f16_nms(self) -> Self {
        Self {
            f16_nms: self.fp16,
            ..self
        }
    }

    /// Whether NMS runs its overlap tests in FP16.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) fn f16_nms(self) -> bool {
        self.f16_nms
    }

    /// Whether the `sdot` kernels may run.
    #[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
    pub(crate) fn dotprod(self) -> bool {
        self.dotprod
    }
}

/// `x` as f32, exactly. An f16 takes one `FCVT` (half to single is part of the aarch64 baseline,
/// unlike FP16 arithmetic) in place of a call into `half`.
#[inline(always)]
pub(crate) fn widen<T: AsPrimitive<f32>>(x: T) -> f32 {
    #[cfg(target_arch = "aarch64")]
    if let Some(h) = cast_one::<T, half::f16>(x) {
        let out: f32;
        // SAFETY: a register-only conversion every aarch64 CPU has.
        unsafe {
            core::arch::asm!(
                "fmov {t:s}, {bits:w}",
                "fcvt {t:s}, {t:h}",
                bits = in(reg) u32::from(h.to_bits()),
                t = out(vreg) out,
                options(pure, nomem, nostack, preserves_flags)
            );
        }
        return out;
    }
    x.as_()
}

/// The kernel family a dispatch ran.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Ran {
    Scalar,
    Neon,
    NeonFp16,
}

/// Every tier this CPU can run, by its `EDGEFIRST_DECODER_FORCE_KERNEL` name.
#[cfg(test)]
pub(crate) fn tiers() -> Vec<(&'static str, Isa)> {
    ["scalar", "neon", "neon_fp16", "neon_dotprod"]
        .into_iter()
        .filter_map(|t| {
            crate::per_scale::kernels::decoder_cpu_features_from(Some(t))
                .ok()
                .map(|f| (t, Isa::of(&f)))
        })
        .collect()
}

/// `s` as a slice of `U` when `T` is `U`.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) fn cast<T: 'static, U: 'static>(s: &[T]) -> Option<&[U]> {
    (std::any::TypeId::of::<T>() == std::any::TypeId::of::<U>()).then(|| {
        // SAFETY: `T` and `U` are the same type.
        unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<U>(), s.len()) }
    })
}

/// `v` as a `U` when `T` is `U`.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) fn cast_one<T: Copy + 'static, U: Copy + 'static>(v: T) -> Option<U> {
    cast::<T, U>(std::slice::from_ref(&v)).and_then(|s| s.first().copied())
}

/// `cols` as slices of `U` when `T` is `U`.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn cast_cols<'a, 'b, T: 'static, U: 'static>(cols: &'a [&'b [T]]) -> Option<&'a [&'b [U]]> {
    (std::any::TypeId::of::<T>() == std::any::TypeId::of::<U>()).then(|| {
        // SAFETY: `T` and `U` are the same type, so `&[T]` and `&[U]` are too.
        unsafe { std::slice::from_raw_parts(cols.as_ptr().cast::<&'b [U]>(), cols.len()) }
    })
}

/// `s` as possibly uninitialized elements, for kernels that only write initialized values.
///
/// # Safety
/// Only initialized values may be written through the returned slice.
pub(crate) unsafe fn as_uninit<T>(s: &mut [T]) -> &mut [MaybeUninit<T>] {
    // SAFETY: `MaybeUninit<T>` has `T`'s layout; the caller writes only initialized values.
    unsafe { &mut *(s as *mut [T] as *mut [MaybeUninit<T>]) }
}

/// `dst` as a mutable slice of `U` when `T` is `U`.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
fn cast_mut<T: 'static, U: 'static>(dst: &mut [T]) -> Option<&mut [U]> {
    (std::any::TypeId::of::<T>() == std::any::TypeId::of::<U>()).then(|| {
        // SAFETY: `T` and `U` are the same type.
        unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr().cast::<U>(), dst.len()) }
    })
}

/// For each anchor `i` the class `c` with the largest `cols[c][i]` as the scalar fold picks it,
/// `keep(i, c)` in anchor order unless that score is below `threshold`.
pub(crate) fn column_argmax_kernel<T: Copy + 'static>(
    isa: Isa,
    cols: &[&[T]],
    threshold: T,
    keep: impl FnMut(usize, usize),
) -> Ran {
    if isa.is_scalar() {
        return Ran::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use edgefirst_kernels::prim::neon_candidates as nc;
        macro_rules! run {
            ($u:ty, $kernel:path, $tier:expr) => {
                if let (Some(c), Some(t)) = (cast_cols::<T, $u>(cols), cast_one::<T, $u>(threshold))
                {
                    // SAFETY: NEON is baseline; `Isa::of` saw FEAT_FP16 for the FP16 kernels.
                    let ran = unsafe { $kernel(c, t, keep) };
                    return if ran { $tier } else { Ran::Scalar };
                }
            };
        }
        run!(i8, nc::column_argmax_i8_neon, Ran::Neon);
        run!(u8, nc::column_argmax_u8_neon, Ran::Neon);
        run!(i16, nc::column_argmax_i16_neon, Ran::Neon);
        run!(u16, nc::column_argmax_u16_neon, Ran::Neon);
        run!(f32, nc::column_argmax_f32_neon, Ran::Neon);
        if isa.fp16() {
            run!(half::f16, nc::column_argmax_f16_neon_fp16, Ran::NeonFp16);
        }
        run!(half::f16, nc::column_argmax_f16_neon, Ran::Neon);
        Ran::Scalar
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (cols, threshold, keep);
        Ran::Scalar
    }
}

/// The position of the largest score of `row` as the scalar fold picks it, with the tier that
/// found it; `None` when the caller folds (scalar tier, other types, short rows, NaN).
#[inline(always)]
pub(crate) fn row_argmax_kernel<T: Copy + 'static>(isa: Isa, row: &[T]) -> Option<(usize, Ran)> {
    if isa.is_scalar() {
        return None;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use edgefirst_kernels::prim::neon_candidates as nc;
        macro_rules! run {
            ($u:ty, $kernel:path, $tier:expr) => {
                if let Some(r) = cast::<T, $u>(row) {
                    // SAFETY: as in `column_argmax_kernel`.
                    return unsafe { $kernel(r) }.map(|(_, i)| (i, $tier));
                }
            };
        }
        run!(i8, nc::row_argmax_i8_neon, Ran::Neon);
        run!(u8, nc::row_argmax_u8_neon, Ran::Neon);
        run!(i16, nc::row_argmax_i16_neon, Ran::Neon);
        run!(u16, nc::row_argmax_u16_neon, Ran::Neon);
        run!(f32, nc::row_argmax_f32_neon, Ran::Neon);
        if isa.fp16() {
            run!(half::f16, nc::row_argmax_f16_neon_fp16, Ran::NeonFp16);
        }
        run!(half::f16, nc::row_argmax_f16_neon, Ran::Neon);
        None
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = row;
        None
    }
}

/// `hit(i)` for each position of `src` at or above `threshold`, in order.
pub(crate) fn ge_indices_kernel<T: Copy + 'static>(
    isa: Isa,
    src: &[T],
    threshold: T,
    hit: impl FnMut(usize),
) -> Ran {
    if isa.is_scalar() {
        return Ran::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use edgefirst_kernels::prim::neon_candidates as nc;
        macro_rules! run {
            ($u:ty, $kernel:path, $tier:expr) => {
                if let (Some(s), Some(t)) = (cast::<T, $u>(src), cast_one::<T, $u>(threshold)) {
                    // SAFETY: as in `column_argmax_kernel`.
                    unsafe { $kernel(s, t, hit) };
                    return $tier;
                }
            };
        }
        run!(i8, nc::ge_indices_i8_neon, Ran::Neon);
        run!(u8, nc::ge_indices_u8_neon, Ran::Neon);
        run!(i16, nc::ge_indices_i16_neon, Ran::Neon);
        run!(u16, nc::ge_indices_u16_neon, Ran::Neon);
        run!(f32, nc::ge_indices_f32_neon, Ran::Neon);
        if isa.fp16() {
            run!(half::f16, nc::ge_indices_f16_neon_fp16, Ran::NeonFp16);
        }
        run!(half::f16, nc::ge_indices_f16_neon, Ran::Neon);
        Ran::Scalar
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (src, threshold, hit);
        Ran::Scalar
    }
}

/// `x · scale + bias` (`x · scale` without a bias) of `src` into `dst` as `F` arithmetic computes
/// it, bit for bit; every element of `dst` is written unless this returns [`Ran::Scalar`].
pub(crate) fn scale_bias<T: 'static, F: Copy + 'static>(
    isa: Isa,
    src: &[T],
    scale: F,
    bias: Option<F>,
    dst: &mut [MaybeUninit<F>],
) -> Ran {
    if isa.is_scalar() || src.len() != dst.len() {
        return Ran::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use edgefirst_kernels::prim::{neon_convert as nc, neon_fp16 as nf};
        macro_rules! run {
            ($s:ty, $d:ty, $kernel:path, $tier:expr) => {
                if let (Some(s), Some(sc), Some(d)) = (
                    cast::<T, $s>(src),
                    cast_one::<F, $d>(scale),
                    cast_mut::<MaybeUninit<F>, MaybeUninit<$d>>(dst),
                ) {
                    let b = bias.and_then(cast_one::<F, $d>);
                    // SAFETY: as in `column_argmax_kernel`.
                    unsafe { $kernel(s, sc, b, d) };
                    return $tier;
                }
            };
        }
        run!(i8, f32, nc::scale_bias_i8_to_f32_neon, Ran::Neon);
        run!(u8, f32, nc::scale_bias_u8_to_f32_neon, Ran::Neon);
        run!(i16, f32, nc::scale_bias_i16_to_f32_neon, Ran::Neon);
        run!(u16, f32, nc::scale_bias_u16_to_f32_neon, Ran::Neon);
        if isa.fp16() {
            run!(
                i8,
                half::f16,
                nf::scale_bias_i8_to_f16_neon_fp16,
                Ran::NeonFp16
            );
            run!(
                u8,
                half::f16,
                nf::scale_bias_u8_to_f16_neon_fp16,
                Ran::NeonFp16
            );
            run!(
                i16,
                half::f16,
                nf::scale_bias_i16_to_f16_neon_fp16,
                Ran::NeonFp16
            );
            run!(
                u16,
                half::f16,
                nf::scale_bias_u16_to_f16_neon_fp16,
                Ran::NeonFp16
            );
        }
        run!(i8, half::f16, nc::scale_bias_i8_to_f16_neon, Ran::Neon);
        run!(u8, half::f16, nc::scale_bias_u8_to_f16_neon, Ran::Neon);
        run!(i16, half::f16, nc::scale_bias_i16_to_f16_neon, Ran::Neon);
        run!(u16, half::f16, nc::scale_bias_u16_to_f16_neon, Ran::Neon);
        Ran::Scalar
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        let _ = (src, scale, bias, dst);
        Ran::Scalar
    }
}

/// [`scale_bias`] of every element of `view` into an array with `map`'s layout, written once;
/// `None` when the caller maps it.
pub(crate) fn scale_bias_array<T: 'static, F: Copy + 'static, D: Dimension>(
    isa: Isa,
    view: &ArrayView<'_, T, D>,
    scale: F,
    bias: Option<F>,
) -> Option<Array<F, D>> {
    if isa.is_scalar() {
        return None;
    }
    map_array(view, |src, dst| {
        scale_bias(isa, src, scale, bias, dst) != Ran::Scalar
    })
}

/// `kernel` of every element of `view` in memory order into an array with `map`'s layout,
/// written once; `None` when `view` has a negative stride or is not contiguous, or `kernel`
/// returns `false` (and then must have written nothing).
pub(crate) fn map_array<T, U, D: Dimension>(
    view: &ArrayView<'_, T, D>,
    kernel: impl FnOnce(&[T], &mut [MaybeUninit<U>]) -> bool,
) -> Option<Array<U, D>> {
    if view.strides().iter().any(|&s| s < 0) {
        return None;
    }
    let src = view.as_slice_memory_order()?;
    let n = src.len();
    let mut out: Vec<U> = Vec::with_capacity(n);
    if !kernel(src, out.spare_capacity_mut().get_mut(..n)?) {
        return None;
    }
    // SAFETY: `kernel` wrote all `n` elements, within the capacity.
    unsafe { out.set_len(n) };
    let mut strides = view.raw_dim();
    for (d, &s) in strides.slice_mut().iter_mut().zip(view.strides()) {
        *d = s as usize;
    }
    Array::from_shape_vec(view.raw_dim().strides(strides), out).ok()
}

/// The score columns as slices when every class column is contiguous (the transposed view of a
/// `[C, N]` output).
pub(crate) fn score_columns<'a, T>(scores: &'a ArrayView2<'_, T>) -> Option<Vec<&'a [T]>> {
    if scores.dim().0 > 1 && scores.strides()[0] != 1 {
        return None;
    }
    scores.columns().into_iter().map(|c| c.to_slice()).collect()
}

/// `(anchor, class)` of every anchor whose best class clears `threshold`, in anchor order, from
/// contiguous class columns; ranges of anchors run on the rayon pool when the scores are large.
/// `None` when `isa` has no kernel for `T`.
pub(crate) fn column_argmax<T: Copy + Send + Sync + 'static>(
    isa: Isa,
    cols: &[&[T]],
    threshold: T,
) -> Option<Vec<(usize, usize)>> {
    if isa.is_scalar() {
        return None;
    }
    let n = cols.first()?.len();
    let scan = |start: usize, end: usize| {
        let range: Option<Vec<&[T]>> = cols.iter().map(|c| c.get(start..end)).collect();
        let mut kept = Vec::new();
        let ran = column_argmax_kernel(isa, &range?, threshold, |i, c| {
            kept.push((start + i, c));
        });
        (ran != Ran::Scalar).then_some(kept)
    };
    if n * cols.len() < PAR_MIN_SCORES {
        return scan(0, n);
    }
    let piece = n
        .div_ceil(2 * rayon::current_num_threads())
        .next_multiple_of(COLUMN_PIECE_ALIGN);
    let pieces: Option<Vec<Vec<(usize, usize)>>> = (0..n.div_ceil(piece))
        .into_par_iter()
        .map(|k| scan(k * piece, ((k + 1) * piece).min(n)))
        .collect();
    pieces.map(|p| p.concat())
}

/// `(anchor, class)` of every score at or above `threshold`, ordered by anchor then class, from
/// contiguous class columns; `None` when `isa` has no kernel for `T`.
pub(crate) fn column_hits<T: Copy + Send + Sync + 'static>(
    isa: Isa,
    cols: &[&[T]],
    threshold: T,
) -> Option<Vec<(usize, usize)>> {
    if isa.is_scalar() {
        return None;
    }
    let scan = |(c, col): (usize, &&[T])| {
        let mut hits = Vec::new();
        let ran = ge_indices_kernel(isa, col, threshold, |i| hits.push((i, c)));
        (ran != Ran::Scalar).then_some(hits)
    };
    let n: usize = cols.iter().map(|c| c.len()).sum();
    let per_class: Option<Vec<Vec<(usize, usize)>>> = if n < PAR_MIN_SCORES {
        cols.iter().enumerate().map(scan).collect()
    } else {
        cols.par_iter().enumerate().map(scan).collect()
    };
    let mut hits = per_class?.concat();
    hits.par_sort_unstable();
    Some(hits)
}

/// [`crate::arg_max`] of a score row, on `isa` when the row is contiguous.
#[inline(always)]
pub(crate) fn row_arg_max<T: PartialOrd + Copy + 'static>(
    isa: Isa,
    row: ArrayView1<T>,
) -> (T, usize) {
    if let Some(r) = row.to_slice() {
        if let Some((i, _)) = row_argmax_kernel(isa, r) {
            if let Some(&s) = r.get(i) {
                return (s, i);
            }
        }
    }
    crate::arg_max(row)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
