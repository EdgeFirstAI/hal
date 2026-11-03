// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Semantic segmentation decode on the NEON tiers (`prim::neon_semantic`): score rows, resampling
//! rows and the per-channel dequantize, each equal to the scalar code in `semantic/` bit for bit
//! except the sigmoid confidence (within `2^-21` relative). Every function reports the kernel
//! that ran or whether it ran; on [`Ran::Scalar`] or `false` the caller runs its scalar code.

#![cfg_attr(not(target_arch = "aarch64"), allow(unused_variables))]

use super::{Isa, Ran};

/// How a row's class scores lie in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScoreRow {
    /// Pixel `x`'s class `c` at `c * cs + x`.
    Planar { cs: usize },
    /// Pixel `x`'s class `c` at `x * classes + c`.
    Interleaved,
}

impl ScoreRow {
    /// The layout of a row with class stride `cs` and pixel stride `xs`, when a kernel reads it.
    pub(crate) fn of(classes: usize, cs: usize, xs: usize) -> Option<Self> {
        if xs == 1 {
            Some(Self::Planar { cs })
        } else if cs == 1 && xs == classes {
            Some(Self::Interleaved)
        } else {
            None
        }
    }
}

/// Runs `$body` with `$s` as `src` cast to each score type that has a kernel, returning its
/// result; falls through when `T` is none of them.
#[cfg(target_arch = "aarch64")]
macro_rules! for_score_types {
    ($src:expr, |$s:ident: $u:ident| $body:expr) => {{
        use edgefirst_kernels::prim::neon_semantic::SemanticScore;
        for_score_types!(@one $src, $s, $u, $body, i8, u8, i16, u16, f32, half::f16);
    }};
    (@one $src:expr, $s:ident, $u:ident, $body:expr, $($t:ty),*) => {$(
        if let Some($s) = super::cast::<_, $t>($src) {
            #[allow(dead_code)]
            type $u = $t;
            return $body;
        }
    )*};
}

/// The class argmax of a row's pixels (`dst.len()` of them, first class on ties), from `src`
/// starting at the row's first score.
pub(crate) fn argmax_row<T: 'static>(
    isa: Isa,
    src: &[T],
    row: ScoreRow,
    classes: usize,
    dst: &mut [u16],
) -> Ran {
    if isa.is_scalar() {
        return Ran::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use edgefirst_kernels::prim::neon_semantic as ns;
        if isa.fp16() {
            if let Some(s) = super::cast::<T, half::f16>(src) {
                // SAFETY: `Isa::of` saw FEAT_FP16.
                let ran = unsafe {
                    match row {
                        ScoreRow::Planar { cs } => {
                            ns::argmax_planar_f16_neon_fp16(s, cs, classes, dst)
                        }
                        ScoreRow::Interleaved => {
                            ns::argmax_interleaved_f16_neon_fp16(s, classes, dst)
                        }
                    }
                };
                return if ran { Ran::NeonFp16 } else { Ran::Scalar };
            }
        }
        for_score_types!(src, |s: U| {
            let ran = match row {
                ScoreRow::Planar { cs } => U::argmax_planar_neon(s, cs, classes, dst),
                ScoreRow::Interleaved => U::argmax_interleaved_neon(s, classes, dst),
            };
            if ran {
                Ran::Neon
            } else {
                Ran::Scalar
            }
        });
    }
    Ran::Scalar
}

/// `1` where a score of the contiguous row `src` is above `cut`, else `0`.
pub(crate) fn threshold_row<T: 'static>(isa: Isa, src: &[T], cut: f32, dst: &mut [u16]) -> Ran {
    if isa.is_scalar() {
        return Ran::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    for_score_types!(src, |s: U| if U::threshold_neon(s, cut, dst) {
        Ran::Neon
    } else {
        Ran::Scalar
    });
    Ran::Scalar
}

/// The winner's softmax probability `1 / Σ_c e^(scale · (x_c − top))` of each pixel of a row.
pub(crate) fn softmax_row<T: 'static>(
    isa: Isa,
    src: &[T],
    row: ScoreRow,
    classes: usize,
    top: &[f32],
    scale: f32,
    dst: &mut [f32],
) -> Ran {
    if isa.is_scalar() {
        return Ran::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    for_score_types!(src, |s: U| {
        let ran = match row {
            ScoreRow::Planar { cs } => U::softmax_planar_neon(s, cs, classes, top, scale, dst),
            ScoreRow::Interleaved => U::softmax_interleaved_neon(s, classes, top, scale, dst),
        };
        if ran {
            Ran::Neon
        } else {
            Ran::Scalar
        }
    });
    Ran::Scalar
}

/// `max(p, 1 − p)` of `p = (x − zero_point) · scale` (its sigmoid when `logit`) for each score
/// of the contiguous row `src`.
pub(crate) fn binary_row<T: 'static>(
    isa: Isa,
    src: &[T],
    zero_point: f32,
    scale: f32,
    logit: bool,
    dst: &mut [f32],
) -> Ran {
    if isa.is_scalar() {
        return Ran::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    for_score_types!(src, |s: U| {
        if U::binary_confidence_neon(s, zero_point, scale, logit, dst) {
            Ran::Neon
        } else {
            Ran::Scalar
        }
    });
    Ran::Scalar
}

/// The sigmoid of each element of `src`.
pub(crate) fn sigmoid_row(isa: Isa, src: &[f32], dst: &mut [f32]) -> Ran {
    if isa.is_scalar() {
        return Ran::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    if edgefirst_kernels::prim::neon_semantic::sigmoid_neon(src, dst) {
        return Ran::Neon;
    }
    Ran::Scalar
}

/// Each class index of the contiguous row `src` as `u16`; [`Ran::Scalar`] also when a value is
/// negative or not below `limit`, `dst` then being unspecified.
pub(crate) fn class_index_row<T: 'static>(
    isa: Isa,
    src: &[T],
    limit: usize,
    dst: &mut [u16],
) -> Ran {
    if isa.is_scalar() {
        return Ran::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    {
        use edgefirst_kernels::prim::neon_semantic::ClassIndex;
        macro_rules! run {
            ($($t:ty),*) => {$(
                if let Some(s) = super::cast::<T, $t>(src) {
                    return if <$t>::class_index_neon(s, limit, dst) { Ran::Neon } else { Ran::Scalar };
                }
            )*};
        }
        run!(u8, u16, i32, u32, i64);
    }
    Ran::Scalar
}

/// The scores of columns `cols` of a row starting at `src`, column-major as f32:
/// `out[j * classes + c] = src[c * cs + cols[j] * xs]`.
pub(crate) fn gather_columns<T: 'static>(
    isa: Isa,
    src: &[T],
    (cs, xs, classes): (usize, usize, usize),
    cols: &[u32],
    out: &mut [f32],
) -> Ran {
    if isa.is_scalar() {
        return Ran::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    for_score_types!(src, |s: U| {
        if U::gather_columns_neon(s, cs, xs, classes, cols, out) {
            Ran::Neon
        } else {
            Ran::Scalar
        }
    });
    Ran::Scalar
}

/// The candidate classes of a boundary cell from its corners' class scores, pushed with their
/// corners as `Band::build` selects them.
#[cfg_attr(not(target_arch = "aarch64"), allow(clippy::ptr_arg))]
pub(crate) fn cell_candidates(
    isa: Isa,
    corners: [&[f32]; 4],
    classes: &mut Vec<u16>,
    out: &mut Vec<[f32; 4]>,
) -> Ran {
    if isa.is_scalar() {
        return Ran::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    {
        let [a, b, c, d] = corners;
        if edgefirst_kernels::prim::neon_semantic::cell_candidates_neon(a, b, c, d, classes, out) {
            return Ran::Neon;
        }
    }
    Ran::Scalar
}

/// The argmax of a boundary cell's pixels at weights `(fy, fx[i])` over the cell's candidate
/// `classes` and their `corners`, as `Band::argmax` computes it.
pub(crate) fn cell_argmax(
    isa: Isa,
    classes: &[u16],
    corners: &[[f32; 4]],
    fy: f32,
    fx: &[f32],
    dst: &mut [u16],
) -> Ran {
    if isa.is_scalar() {
        return Ran::Scalar;
    }
    #[cfg(target_arch = "aarch64")]
    if edgefirst_kernels::prim::neon_semantic::cell_argmax_neon(classes, corners, fy, fx, dst) {
        return Ran::Neon;
    }
    Ran::Scalar
}

/// Element strides of a score map: class `c`, row `y`, column `x` at `c·cs + y·ys + x·xs`.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) struct Planes {
    pub(crate) classes: usize,
    pub(crate) height: usize,
    pub(crate) width: usize,
    pub(crate) cs: usize,
    pub(crate) ys: usize,
    pub(crate) xs: usize,
}

/// The `(x − zero_point[c]) · scale[c]` of every score `grid` addresses, into the same positions
/// of `out` (laid out like `data`), when the class planes are contiguous rows or the pixels hold
/// contiguous classes; `false` when the caller dequantizes (scalar tier, other layouts or types).
pub(crate) fn dequantize_per_channel<T: 'static>(
    isa: Isa,
    data: &[T],
    grid: Planes,
    scales: &[f32],
    zero_points: &[i32],
    out: &mut [f32],
) -> bool {
    if isa.is_scalar() {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        macro_rules! run {
            ($($t:ty: $planar:path),*) => {$(
                if let Some(s) = super::cast::<T, $t>(data) {
                    return per_channel::<$t>(s, grid, scales, zero_points, out, $planar);
                }
            )*};
        }
        use edgefirst_kernels::prim::neon_convert as nc;
        run!(
            i8: nc::dequant_i8_to_f32_exact_neon,
            u8: nc::dequant_u8_to_f32_exact_neon,
            i16: nc::dequant_i16_to_f32_exact_neon,
            u16: nc::dequant_u16_to_f32_exact_neon
        );
    }
    false
}

/// Taps of one bilinear output axis: the lower and upper source positions and the upper weight.
#[cfg_attr(not(target_arch = "aarch64"), allow(dead_code))]
pub(crate) struct Taps<'a> {
    pub(crate) lo: &'a [u32],
    pub(crate) hi: &'a [u32],
    pub(crate) frac: &'a [f32],
}

/// Nearest resampling of a class map (rows of `width`) into `u8` (truncated) or `u16` classes
/// at rows `ys` and columns `xs`; `false` when the caller resamples (`dst` then unspecified).
pub(crate) fn nearest_classes<O: 'static>(
    isa: Isa,
    map: &[u16],
    width: usize,
    ys: &[u32],
    xs: &[u32],
    dst: &mut [O],
) -> bool {
    if isa.is_scalar() {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if let Some(d) = super::cast_mut::<O, u8>(dst) {
            return resample::nearest(map, width, ys, xs, d);
        }
        if let Some(d) = super::cast_mut::<O, u16>(dst) {
            return resample::nearest(map, width, ys, xs, d);
        }
    }
    false
}

/// [`nearest_classes`] of a confidence map into `f32` or `u8` (`(v.clamp(0, 1) · 255 + 0.5) as
/// u8`) confidence.
pub(crate) fn nearest_confidence<C: 'static>(
    isa: Isa,
    map: &[f32],
    width: usize,
    ys: &[u32],
    xs: &[u32],
    dst: &mut [C],
) -> bool {
    if isa.is_scalar() {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if let Some(d) = super::cast_mut::<C, f32>(dst) {
            return resample::nearest(map, width, ys, xs, d);
        }
        if let Some(d) = super::cast_mut::<C, u8>(dst) {
            return resample::nearest(map, width, ys, xs, d);
        }
    }
    false
}

/// Bilinear resampling of a confidence map (rows of `width`) into `f32` or `u8` confidence, in
/// the scalar `lerp2` order; `false` when the caller resamples (`dst` then unspecified).
pub(crate) fn bilinear_confidence<C: 'static>(
    isa: Isa,
    map: &[f32],
    width: usize,
    ys: Taps<'_>,
    xs: Taps<'_>,
    dst: &mut [C],
) -> bool {
    if isa.is_scalar() {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        let fill = |y: usize, row: &mut [f32]| {
            if let Some(src) = map.get(y * width..(y + 1) * width) {
                row.copy_from_slice(src);
            }
        };
        if map.len()
            < ys.hi
                .iter()
                .chain(ys.lo)
                .map(|&y| (y as usize + 1) * width)
                .max()
                .unwrap_or(0)
        {
            return false;
        }
        if let Some(d) = super::cast_mut::<C, f32>(dst) {
            return resample::bilinear(width, fill, ys, xs, d, resample::blend::<f32>);
        }
        if let Some(d) = super::cast_mut::<C, u8>(dst) {
            return resample::bilinear(width, fill, ys, xs, d, resample::blend::<u8>);
        }
    }
    false
}

/// Bilinear resampling of one score channel whose row `y` `fill(y, row)` writes as f32 (`width`
/// columns), as `u8` or `u16` classes `1` above `cut`, else `0`; `false` when the caller
/// resamples (`dst` then unspecified).
pub(crate) fn bilinear_threshold<O: 'static>(
    isa: Isa,
    width: usize,
    fill: impl Fn(usize, &mut [f32]) + Sync,
    ys: Taps<'_>,
    xs: Taps<'_>,
    cut: f32,
    dst: &mut [O],
) -> bool {
    if isa.is_scalar() {
        return false;
    }
    #[cfg(target_arch = "aarch64")]
    {
        if let Some(d) = super::cast_mut::<O, u8>(dst) {
            return resample::bilinear(width, fill, ys, xs, d, resample::above::<u8>(cut));
        }
        if let Some(d) = super::cast_mut::<O, u16>(dst) {
            return resample::bilinear(width, fill, ys, xs, d, resample::above::<u16>(cut));
        }
    }
    false
}

/// Classes per tiled row of the interleaved per-channel dequantize, at least.
#[cfg(target_arch = "aarch64")]
const TILED_VALUES: usize = 256;

/// [`dequantize_per_channel`] for one integer type: class-plane rows through the per-tensor
/// kernel `planar`, interleaved pixels through the per-channel kernel with tiled parameters.
#[cfg(target_arch = "aarch64")]
fn per_channel<T: edgefirst_kernels::prim::neon_convert::PerChannelIn>(
    data: &[T],
    Planes {
        classes,
        height,
        width,
        cs,
        ys,
        xs,
    }: Planes,
    scales: &[f32],
    zero_points: &[i32],
    out: &mut [f32],
    planar: unsafe fn(&[T], edgefirst_kernels::Quant, &mut [f32]),
) -> bool {
    if scales.len() != classes || zero_points.len() != classes || out.len() != data.len() {
        return false;
    }
    let span = |start: usize, len: usize| start.checked_add(len).filter(|&e| e <= data.len());
    if xs == 1 {
        // Whole planes at once when rows abut.
        let (rows, row_len, step) = if ys == width {
            (1, height * width, 0)
        } else {
            (height, width, ys)
        };
        for (c, (&scale, &zero_point)) in scales.iter().zip(zero_points).enumerate() {
            let q = edgefirst_kernels::Quant { scale, zero_point };
            for y in 0..rows {
                let start = c * cs + y * step;
                let Some(end) = span(start, row_len) else {
                    return false;
                };
                // SAFETY: NEON is part of the aarch64 baseline.
                unsafe { planar(&data[start..end], q, &mut out[start..end]) };
            }
        }
        return true;
    }
    if cs != 1 || xs != classes {
        return false;
    }
    let pixels = TILED_VALUES.div_ceil(classes).max(1);
    let zp: Vec<f32> = zero_points
        .iter()
        .map(|&z| z as f32)
        .cycle()
        .take(pixels * classes)
        .collect();
    let sc: Vec<f32> = scales
        .iter()
        .copied()
        .cycle()
        .take(pixels * classes)
        .collect();
    for y in 0..height {
        let start = y * ys;
        let Some(end) = span(start, width * classes) else {
            return false;
        };
        let rows = data[start..end].chunks(pixels * classes);
        for (s, d) in rows.zip(out[start..end].chunks_mut(pixels * classes)) {
            let n = s.len();
            T::dequant_per_channel_neon(s, &zp[..n], &sc[..n], d);
        }
    }
    true
}

/// Resampling rows on the NEON tiers: nearest gathers through `TBL` windows, rows that repeat
/// their predecessor's source row copied, and bilinear rows as a horizontal pass per source row
/// (`(1 − fx) · v_lo + fx · v_hi`) blended vertically, the scalar `lerp2` order.
#[cfg(target_arch = "aarch64")]
pub(crate) mod resample {
    use super::Taps;
    use edgefirst_kernels::prim::neon_semantic as ns;
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

    use rayon::slice::{ParallelSlice, ParallelSliceMut};

    /// Output rows per rayon task of the nearest resampling.
    const ROWS_PER_TASK: usize = 16;

    /// A nearest-resampling source element and output pairing with a kernel.
    pub(crate) trait Nearest<O>: Copy + Sync {
        /// The gather plan of `xs` over a source row of `width`.
        fn plan(xs: &[u32], width: usize) -> Option<ns::RowGather>;
        /// One output row; `scratch` holds `dst.len()` source elements.
        fn row(plan: &ns::RowGather, src: &[Self], scratch: &mut [Self], dst: &mut [O]) -> bool;
    }

    impl Nearest<u8> for u16 {
        fn plan(xs: &[u32], width: usize) -> Option<ns::RowGather> {
            ns::RowGather::u16_low_byte(xs, width)
        }
        fn row(plan: &ns::RowGather, src: &[u16], _: &mut [u16], dst: &mut [u8]) -> bool {
            ns::gather_u16_to_u8_neon(plan, src, dst)
        }
    }

    impl Nearest<u16> for u16 {
        fn plan(xs: &[u32], width: usize) -> Option<ns::RowGather> {
            ns::RowGather::u16(xs, width)
        }
        fn row(plan: &ns::RowGather, src: &[u16], _: &mut [u16], dst: &mut [u16]) -> bool {
            ns::gather_u16_neon(plan, src, dst)
        }
    }

    impl Nearest<f32> for f32 {
        fn plan(xs: &[u32], width: usize) -> Option<ns::RowGather> {
            ns::RowGather::f32(xs, width)
        }
        fn row(plan: &ns::RowGather, src: &[f32], _: &mut [f32], dst: &mut [f32]) -> bool {
            ns::gather_f32_neon(plan, src, dst)
        }
    }

    impl Nearest<u8> for f32 {
        fn plan(xs: &[u32], width: usize) -> Option<ns::RowGather> {
            ns::RowGather::f32(xs, width)
        }
        fn row(plan: &ns::RowGather, src: &[f32], scratch: &mut [f32], dst: &mut [u8]) -> bool {
            ns::gather_f32_neon(plan, src, scratch) && ns::confidence_to_u8_neon(scratch, dst)
        }
    }

    /// Nearest resampling of `map` (rows of `width`) at rows `ys` and columns `xs` into `dst`;
    /// `false` (with `dst` unspecified) when a position is outside the map.
    pub(crate) fn nearest<S: Nearest<O> + Default + Send, O: Copy + Send + Sync>(
        map: &[S],
        width: usize,
        ys: &[u32],
        xs: &[u32],
        dst: &mut [O],
    ) -> bool {
        let out_w = xs.len();
        let Some(plan) = S::plan(xs, width) else {
            return false;
        };
        if out_w == 0 || dst.len() != ys.len() * out_w {
            return false;
        }
        dst.par_chunks_mut(out_w * ROWS_PER_TASK)
            .zip(ys.par_chunks(ROWS_PER_TASK))
            .all(|(rows, sys)| {
                let mut scratch = vec![S::default(); out_w];
                let mut prev: Option<u32> = None;
                for (r, &sy) in sys.iter().enumerate() {
                    let (done, rest) = rows.split_at_mut(r * out_w);
                    let Some(out) = rest.get_mut(..out_w) else {
                        return false;
                    };
                    if prev == Some(sy) {
                        if let Some(last) =
                            done.len().checked_sub(out_w).and_then(|s| done.get(s..))
                        {
                            out.copy_from_slice(last);
                            continue;
                        }
                    }
                    let start = sy as usize * width;
                    let Some(src) = map.get(start..start + width) else {
                        return false;
                    };
                    if !S::row(&plan, src, &mut scratch, out) {
                        return false;
                    }
                    prev = Some(sy);
                }
                true
            })
    }

    /// A bilinear output row from two horizontally interpolated source rows.
    pub(crate) trait Blend: Copy + Default + Send + Sync {
        /// `wy0 · h0 + fy · h1` per element, converted to `Self`.
        fn blend(h0: &[f32], h1: &[f32], wy0: f32, fy: f32, dst: &mut [Self]) -> bool;
    }

    /// Bilinear confidence as `f32`.
    impl Blend for f32 {
        fn blend(h0: &[f32], h1: &[f32], wy0: f32, fy: f32, dst: &mut [f32]) -> bool {
            ns::blend_neon(h0, h1, wy0, fy, dst)
        }
    }

    /// Bilinear confidence as `u8`.
    impl Blend for u8 {
        fn blend(h0: &[f32], h1: &[f32], wy0: f32, fy: f32, dst: &mut [u8]) -> bool {
            ns::blend_confidence_u8_neon(h0, h1, wy0, fy, dst)
        }
    }

    /// Bilinear resampling of a source with `width` columns, whose row `y` `fill(y, row)` writes
    /// as f32, at output rows `(ys.lo, ys.hi, ys.frac)` and columns `xs`, each output row
    /// produced by `emit(h_lo, h_hi, wy0, fy, out)`; `false` (with `dst` unspecified) when a
    /// position is outside the source or a kernel declines.
    pub(crate) fn bilinear<O: Copy + Send + Sync>(
        width: usize,
        fill: impl Fn(usize, &mut [f32]) + Sync,
        ys: Taps<'_>,
        xs: Taps<'_>,
        dst: &mut [O],
        emit: impl Fn(&[f32], &[f32], f32, f32, &mut [O]) -> bool + Sync,
    ) -> bool {
        let out_w = xs.lo.len();
        let (Some(lo), Some(hi)) = (
            ns::RowGather::f32(xs.lo, width),
            ns::RowGather::f32(xs.hi, width),
        ) else {
            return false;
        };
        let rows = ys.lo.len();
        if out_w == 0
            || Some(dst.len()) != rows.checked_mul(out_w)
            || xs.frac.len() != out_w
            || ys.hi.len() != rows
            || ys.frac.len() != rows
        {
            return false;
        }
        let wx0: Vec<f32> = xs.frac.iter().map(|&f| 1.0 - f).collect();
        // Bands of output rows that read the same lower source row.
        let mut bands: Vec<(usize, &mut [O])> = Vec::new();
        let mut rest = dst;
        let mut start = 0;
        while start < ys.lo.len() {
            let first = ys.lo[start];
            let end = start + ys.lo[start..].iter().take_while(|&&l| l == first).count();
            let (band, tail) = rest.split_at_mut((end - start) * out_w);
            bands.push((start, band));
            rest = tail;
            start = end;
        }
        let horizontal =
            |y: usize, row: &mut [f32], a: &mut [f32], b: &mut [f32], h: &mut [f32]| {
                fill(y, row);
                ns::gather_f32_neon(&lo, row, a)
                    && ns::gather_f32_neon(&hi, row, b)
                    && ns::lerp_neon(a, b, &wx0, xs.frac, h)
            };
        // One source row and four output-width rows per worker, reused across its bands.
        let scratch = || {
            let row = || vec![0f32; out_w];
            (vec![0f32; width], [row(), row(), row(), row()])
        };
        bands
            .into_par_iter()
            .map_init(scratch, |(row, [a, b, h0, h1]), (first, band)| {
                let y0 = ys.lo[first] as usize;
                if !horizontal(y0, row, a, b, h0) {
                    return false;
                }
                let mut upper: Option<usize> = None;
                for (r, out) in band.chunks_mut(out_w).enumerate() {
                    let (y1, fy) = (ys.hi[first + r] as usize, ys.frac[first + r]);
                    let h = if y1 == y0 {
                        &*h0
                    } else {
                        if upper != Some(y1) {
                            if !horizontal(y1, row, a, b, h1) {
                                return false;
                            }
                            upper = Some(y1);
                        }
                        &*h1
                    };
                    if !emit(h0, h, 1.0 - fy, fy, out) {
                        return false;
                    }
                }
                true
            })
            .all(|ok| ok)
    }

    /// [`bilinear`] output rows as [`Blend`] values.
    pub(crate) fn blend<O: Blend>(
        h0: &[f32],
        h1: &[f32],
        wy0: f32,
        fy: f32,
        out: &mut [O],
    ) -> bool {
        O::blend(h0, h1, wy0, fy, out)
    }

    /// [`bilinear`] output rows as `1` above `cut`, else `0`.
    pub(crate) fn above<O: ns::ClassLane>(
        cut: f32,
    ) -> impl Fn(&[f32], &[f32], f32, f32, &mut [O]) -> bool + Sync {
        move |h0, h1, wy0, fy, out| ns::blend_threshold_neon(h0, h1, wy0, fy, cut, out)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;
