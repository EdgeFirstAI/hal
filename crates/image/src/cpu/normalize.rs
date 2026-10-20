// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Per-channel affine output mapping: normalization and quantization of a
//! finished 8-bit image into an f32, f16, u8 or i8 destination.
//!
//! Every element is `y = x * scale + offset` (unfused, so the NEON and scalar
//! kernels agree bit for bit). Integer destinations take
//! `clamp(floor(y + 0.5) + zero_point, lo, hi)` and store its low byte, the
//! same rounding the GL shaders use. A row's coefficients follow a 12-lane
//! pattern, which covers 1-, 3- and 4-channel packed rows (12 is a multiple of
//! each) and a planar plane (one channel, repeated).

use super::{guard_plane, logical_surface, tensor_row_stride};
use crate::options::OutputMap;
use crate::{Error, Rect, Result};
use edgefirst_tensor::{DType, PixelLayout, Tensor, TensorDyn, TensorMapTrait, TensorTrait};

const PERIOD: usize = 12;

/// Coefficients for each lane of a 12-element group.
#[derive(Clone, Copy)]
pub(crate) struct Pattern {
    pub(crate) scale: [f32; PERIOD],
    pub(crate) offset: [f32; PERIOD],
    pub(crate) zero_point: [i32; PERIOD],
    pub(crate) lo: [i32; PERIOD],
    pub(crate) hi: [i32; PERIOD],
}

impl Pattern {
    /// Packed rows interleave `channels` channels; a planar row is one channel.
    pub(crate) fn new(map: &OutputMap, channels: usize, plane: Option<usize>) -> Self {
        let ch = |i: usize| plane.unwrap_or(i % channels.max(1));
        let bounds = map.bounds.unwrap_or([(0.0, 255.0); 4]);
        Self {
            scale: std::array::from_fn(|i| map.scale[ch(i)]),
            offset: std::array::from_fn(|i| map.offset[ch(i)]),
            zero_point: std::array::from_fn(|i| map.zero_point[ch(i)] as i32),
            lo: std::array::from_fn(|i| bounds[ch(i)].0 as i32),
            hi: std::array::from_fn(|i| bounds[ch(i)].1 as i32),
        }
    }
}

/// `dst[i] = src[i] * scale + offset`.
pub(crate) fn affine_u8_to_f32(src: &[u8], dst: &mut [f32], p: &Pattern) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "aarch64")]
    let done = unsafe { neon::affine_f32(src, dst, p) };
    #[cfg(not(target_arch = "aarch64"))]
    let done = 0;
    // `done` is a whole number of 12-lane groups, so each group starts at lane 0.
    // Fixed-width groups vectorize whatever the caller's inlining; a variable-length chunk did not under fat LTO.
    let (src, dst) = (&src[done..], &mut dst[done..]);
    let mut groups = src.chunks_exact(PERIOD).zip(dst.chunks_exact_mut(PERIOD));
    for (s, d) in &mut groups {
        for l in 0..PERIOD {
            d[l] = s[l] as f32 * p.scale[l] + p.offset[l];
        }
    }
    let whole = src.len() / PERIOD * PERIOD;
    for (l, (&x, y)) in src[whole..].iter().zip(&mut dst[whole..]).enumerate() {
        *y = x as f32 * p.scale[l] + p.offset[l];
    }
}

/// As [`affine_u8_to_f32`], rounded to the nearest f16.
pub(crate) fn affine_u8_to_f16(
    src: &[u8],
    dst: &mut [half::f16],
    p: &Pattern,
    scratch: &mut Vec<f32>,
) {
    use half::slice::HalfFloatSliceExt as _;
    scratch.resize(src.len(), 0.0);
    affine_u8_to_f32(src, scratch, p);
    dst.convert_from_f32_slice(scratch);
}

/// `clamp(floor(src[i] * scale + offset + 0.5) + zero_point, lo, hi)`, stored
/// as its low byte (two's complement for i8).
pub(crate) fn affine_u8_to_int(src: &[u8], dst: &mut [u8], p: &Pattern) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "aarch64")]
    let done = unsafe { neon::affine_int(src, dst, p) };
    #[cfg(not(target_arch = "aarch64"))]
    let done = 0;
    // `done` is a whole number of 12-lane groups, so each group starts at lane 0.
    let (src, dst) = (&src[done..], &mut dst[done..]);
    let mut groups = src.chunks_exact(PERIOD).zip(dst.chunks_exact_mut(PERIOD));
    for (s, d) in &mut groups {
        for l in 0..PERIOD {
            d[l] = quantize_lane(s[l], p, l);
        }
    }
    let whole = src.len() / PERIOD * PERIOD;
    for (l, (&x, q)) in src[whole..].iter().zip(&mut dst[whole..]).enumerate() {
        *q = quantize_lane(x, p, l);
    }
}

/// One element of [`affine_u8_to_int`], in float arithmetic that vectorizes
/// (no `f32::floor` call, no saturating cast).
#[inline(always)]
fn quantize_lane(x: u8, p: &Pattern, l: usize) -> u8 {
    // Bounded first so the float floor below is exact and the cast cannot
    // overflow.
    let t = (x as f32 * p.scale[l] + p.offset[l] + 0.5).clamp(-1024.0, 1024.0);
    let r = (t + 12_582_912.0) - 12_582_912.0;
    let f = if r > t { r - 1.0 } else { r };
    let v = (f + p.zero_point[l] as f32).clamp(p.lo[l] as f32, p.hi[l] as f32);
    // SAFETY: `v` is finite and within [-128, 255], so it fits an i32.
    (unsafe { v.to_int_unchecked::<i32>() }) as u8
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{Pattern, PERIOD};
    use std::arch::aarch64::*;

    /// Widen 16 bytes to four f32x4.
    #[inline]
    #[target_feature(enable = "neon")]
    fn widen16(v: uint8x16_t) -> [float32x4_t; 4] {
        let lo = vmovl_u8(vget_low_u8(v));
        let hi = vmovl_high_u8(v);
        [
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(lo))),
            vcvtq_f32_u32(vmovl_high_u16(lo)),
            vcvtq_f32_u32(vmovl_u16(vget_low_u16(hi))),
            vcvtq_f32_u32(vmovl_high_u16(hi)),
        ]
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn coeffs(p: &Pattern) -> ([float32x4_t; 3], [float32x4_t; 3]) {
        // SAFETY: NEON is baseline on aarch64; the callers keep every
        // access inside the slices.
        unsafe {
            let s = p.scale.as_ptr();
            let o = p.offset.as_ptr();
            (
                [vld1q_f32(s), vld1q_f32(s.add(4)), vld1q_f32(s.add(8))],
                [vld1q_f32(o), vld1q_f32(o.add(4)), vld1q_f32(o.add(8))],
            )
        }
    }

    /// Processes whole 48-byte blocks; returns the count handled.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn affine_f32(src: &[u8], dst: &mut [f32], p: &Pattern) -> usize {
        // SAFETY: NEON is baseline on aarch64; the callers keep every
        // access inside the slices.
        unsafe {
            const _: () = assert!(48 % PERIOD == 0);
            let (sv, ov) = coeffs(p);
            let n = src.len() / 48 * 48;
            let (sp, dp) = (src.as_ptr(), dst.as_mut_ptr());
            let mut i = 0;
            while i < n {
                for b in 0..3 {
                    let f = widen16(vld1q_u8(sp.add(i + 16 * b)));
                    for (k, fk) in f.iter().enumerate() {
                        let j = b * 4 + k;
                        let y = vaddq_f32(vmulq_f32(*fk, sv[j % 3]), ov[j % 3]);
                        vst1q_f32(dp.add(i + 4 * j), y);
                    }
                }
                i += 48;
            }
            n
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    unsafe fn ints(v: &[i32; PERIOD]) -> [int32x4_t; 3] {
        // SAFETY: NEON is baseline on aarch64; the callers keep every
        // access inside the slices.
        unsafe {
            let p = v.as_ptr();
            [vld1q_s32(p), vld1q_s32(p.add(4)), vld1q_s32(p.add(8))]
        }
    }

    /// Processes whole 48-byte blocks; returns the count handled.
    #[target_feature(enable = "neon")]
    pub(super) unsafe fn affine_int(src: &[u8], dst: &mut [u8], p: &Pattern) -> usize {
        // SAFETY: NEON is baseline on aarch64; the callers keep every
        // access inside the slices.
        unsafe {
            let (sv, ov) = coeffs(p);
            let (zv, lov, hiv) = (ints(&p.zero_point), ints(&p.lo), ints(&p.hi));
            let half = vdupq_n_f32(0.5);
            let n = src.len() / 48 * 48;
            let (sp, dp) = (src.as_ptr(), dst.as_mut_ptr());
            let mut i = 0;
            while i < n {
                for b in 0..3 {
                    let f = widen16(vld1q_u8(sp.add(i + 16 * b)));
                    let q: [int32x4_t; 4] = std::array::from_fn(|k| {
                        let j = (b * 4 + k) % 3;
                        let y = vaddq_f32(vmulq_f32(f[k], sv[j]), ov[j]);
                        let q = vaddq_s32(vcvtmq_s32_f32(vaddq_f32(y, half)), zv[j]);
                        vmaxq_s32(vminq_s32(q, hiv[j]), lov[j])
                    });
                    // Bounded to [-128, 255]: the first narrow keeps the value,
                    // the second keeps the low byte.
                    let lo = vcombine_s16(vmovn_s32(q[0]), vmovn_s32(q[1]));
                    let hi = vcombine_s16(vmovn_s32(q[2]), vmovn_s32(q[3]));
                    let out = vreinterpretq_u8_s8(vcombine_s8(vmovn_s16(lo), vmovn_s16(hi)));
                    vst1q_u8(dp.add(i + 16 * b), out);
                }
                i += 48;
            }
            n
        }
    }
}

/// Map the finished 8-bit image `tmp` into `dst` (same format and size).
pub(super) fn write_mapped(tmp: &Tensor<u8>, dst: &mut TensorDyn, map: &OutputMap) -> Result<()> {
    let fmt = tmp.format().ok_or(Error::NotAnImage)?;
    let h = tmp.height().ok_or(Error::NotAnImage)?;
    let (rows, row_len) = logical_surface(tmp)?;
    let planar = fmt.layout() == PixelLayout::Planar;
    let channels = fmt.channels();
    let patterns: Vec<Pattern> = if planar {
        (0..channels)
            .map(|c| Pattern::new(map, 1, Some(c)))
            .collect()
    } else {
        vec![Pattern::new(map, channels, None)]
    };
    let pattern_for = |r: usize| &patterns[if planar { r / h } else { 0 }];

    let src_stride = tensor_row_stride(tmp);
    let src_map = tmp.map_read()?;
    guard_plane(
        src_map.as_slice().len(),
        src_stride,
        rows,
        row_len,
        "map src",
    )?;
    let src_rows = src_map.as_slice().chunks(src_stride).take(rows);

    let dtype = dst.dtype();
    let elem = dtype.size();
    let dst_stride_bytes = dst.effective_row_stride().ok_or(Error::NotAnImage)?;
    if !dst_stride_bytes.is_multiple_of(elem) {
        return Err(Error::InvalidShape(format!(
            "{dtype} destination row stride {dst_stride_bytes} is not a multiple of the \
             element size {elem}"
        )));
    }
    let dst_stride = dst_stride_bytes / elem;
    match dtype {
        DType::F32 => {
            let mut m = dst.as_typed_mut::<f32>().unwrap().map_mut()?;
            guard_plane(m.as_slice().len(), dst_stride, rows, row_len, "map f32 dst")?;
            for (r, (s, d)) in src_rows
                .zip(m.as_mut_slice().chunks_mut(dst_stride))
                .enumerate()
            {
                affine_u8_to_f32(&s[..row_len], &mut d[..row_len], pattern_for(r));
            }
        }
        DType::F16 => {
            let mut m = dst.as_typed_mut::<half::f16>().unwrap().map_mut()?;
            guard_plane(m.as_slice().len(), dst_stride, rows, row_len, "map f16 dst")?;
            let mut scratch = Vec::with_capacity(row_len);
            for (r, (s, d)) in src_rows
                .zip(m.as_mut_slice().chunks_mut(dst_stride))
                .enumerate()
            {
                affine_u8_to_f16(
                    &s[..row_len],
                    &mut d[..row_len],
                    pattern_for(r),
                    &mut scratch,
                );
            }
        }
        DType::U8 | DType::I8 => {
            let mut m = int_map(dst)?;
            guard_plane(m.as_slice().len(), dst_stride, rows, row_len, "map int dst")?;
            for (r, (s, d)) in src_rows
                .zip(m.as_mut_slice().chunks_mut(dst_stride))
                .enumerate()
            {
                affine_u8_to_int(&s[..row_len], &mut d[..row_len], pattern_for(r));
            }
        }
        other => return Err(Error::NotSupported(format!("convert to {other}"))),
    }
    Ok(())
}

/// The bytes of a u8 or i8 destination.
fn int_map(dst: &mut TensorDyn) -> Result<edgefirst_tensor::HostView<'static, u8>> {
    match dst.dtype() {
        DType::U8 => Ok(dst.as_typed_mut::<u8>().unwrap().map_mut()?),
        DType::I8 => {
            let t = dst.as_typed_mut::<i8>().unwrap();
            // SAFETY: Tensor<i8> and Tensor<u8> are layout-identical (same
            // element size, no T-dependent drop glue).
            let t = unsafe { &mut *(t as *mut Tensor<i8> as *mut Tensor<u8>) };
            Ok(t.map_mut()?)
        }
        other => Err(Error::Internal(format!("int_map on {other}"))),
    }
}

/// Write `values` (destination channel order, already mapped) to every
/// destination element outside `rect`.
pub(super) fn fill_outside(dst: &mut TensorDyn, rect: Rect, values: [f32; 4]) -> Result<()> {
    let fmt = dst.format().ok_or(Error::NotAnImage)?;
    let w = dst.width().ok_or(Error::NotAnImage)?;
    let h = dst.height().ok_or(Error::NotAnImage)?;
    let channels = fmt.channels();
    let planar = fmt.layout() == PixelLayout::Planar;
    if !matches!(fmt.layout(), PixelLayout::Packed | PixelLayout::Planar) {
        return Err(Error::NotSupported(format!(
            "output-domain letterbox pad on {fmt:?}"
        )));
    }
    let (rows, row_len) = if planar {
        (channels * h, w)
    } else {
        (h, w * channels)
    };
    let dtype = dst.dtype();
    let dst_stride = dst.effective_row_stride().ok_or(Error::NotAnImage)? / dtype.size();

    // Element ranges of row `r` that lie outside the content rectangle.
    let spans = |r: usize| -> [(usize, usize); 2] {
        let y = if planar { r % h } else { r };
        let px = if planar { 1 } else { channels };
        if y < rect.top || y >= rect.top + rect.height {
            [(0, row_len), (0, 0)]
        } else {
            [
                (0, rect.left * px),
                ((rect.left + rect.width) * px, row_len),
            ]
        }
    };
    // A planar row holds one channel; a packed row cycles through `channels` lanes.
    let lanes = if planar { 1 } else { channels };
    let row_values = |r: usize| if planar { [values[r / h]; 4] } else { values };
    // Whole-slice fills per row, so the loop stays a memset-like store whatever the caller's inlining.
    fn fill<T: Copy>(
        buf: &mut [T],
        stride: usize,
        rows: usize,
        lanes: usize,
        spans: impl Fn(usize) -> [(usize, usize); 2],
        row_values: impl Fn(usize) -> [T; 4],
    ) {
        for (r, row) in buf.chunks_mut(stride).take(rows).enumerate() {
            let v = row_values(r);
            for (a, b) in spans(r) {
                if lanes == 1 {
                    row[a..b].fill(v[0]);
                } else {
                    for px in row[a..b].chunks_exact_mut(lanes) {
                        px.copy_from_slice(&v[..lanes]);
                    }
                }
            }
        }
    }
    match dtype {
        DType::F32 => {
            let mut m = dst.as_typed_mut::<f32>().unwrap().map_mut()?;
            guard_plane(m.as_slice().len(), dst_stride, rows, row_len, "pad f32")?;
            fill(m.as_mut_slice(), dst_stride, rows, lanes, spans, row_values);
        }
        DType::F16 => {
            let mut m = dst.as_typed_mut::<half::f16>().unwrap().map_mut()?;
            guard_plane(m.as_slice().len(), dst_stride, rows, row_len, "pad f16")?;
            fill(m.as_mut_slice(), dst_stride, rows, lanes, spans, |r| {
                row_values(r).map(half::f16::from_f32)
            });
        }
        DType::U8 | DType::I8 => {
            let mut m = int_map(dst)?;
            guard_plane(m.as_slice().len(), dst_stride, rows, row_len, "pad int")?;
            fill(m.as_mut_slice(), dst_stride, rows, lanes, spans, |r| {
                row_values(r).map(|x| x as i32 as u8)
            });
        }
        other => return Err(Error::NotSupported(format!("convert to {other}"))),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::options::{Normalization, OutputMap};
    use edgefirst_tensor::PixelFormat;

    fn scalar_f32(src: &[u8], p: &Pattern) -> Vec<f32> {
        src.iter()
            .enumerate()
            .map(|(i, &x)| x as f32 * p.scale[i % PERIOD] + p.offset[i % PERIOD])
            .collect()
    }

    #[test]
    fn kernels_match_the_scalar_reference_for_every_length() {
        let map = OutputMap::resolve(
            &Normalization::imagenet(),
            DType::F32,
            PixelFormat::Rgb,
            None,
        )
        .unwrap();
        let p = Pattern::new(&map, 3, None);
        let src: Vec<u8> = (0..200).map(|i| (i * 37 % 256) as u8).collect();
        for len in [0, 1, 47, 48, 49, 96, 150, 200] {
            let mut out = vec![0.0; len];
            affine_u8_to_f32(&src[..len], &mut out, &p);
            assert_eq!(out, scalar_f32(&src[..len], &p), "len {len}");
        }
    }

    #[test]
    fn int_kernel_rounds_half_up_and_clamps_per_lane() {
        let mut p = Pattern {
            scale: [1.0; PERIOD],
            offset: [-127.5; PERIOD],
            zero_point: [0; PERIOD],
            lo: [-128; PERIOD],
            hi: [127; PERIOD],
        };
        // Lane 3 of each 4-channel pixel is a raw alpha byte.
        for l in (3..PERIOD).step_by(4) {
            (p.scale[l], p.offset[l], p.lo[l], p.hi[l]) = (1.0, 0.0, 0, 255);
        }
        let src: Vec<u8> = (0..=255).chain(0..=255).collect();
        let mut out = vec![0u8; src.len()];
        affine_u8_to_int(&src, &mut out, &p);
        for (i, (&x, &q)) in src.iter().zip(&out).enumerate() {
            let want = if i % PERIOD % 4 == 3 {
                x
            } else {
                (x as i32 - 127).clamp(-128, 127) as u8
            };
            assert_eq!(q, want, "element {i} x {x}");
        }
    }

    use crate::{CPUProcessor, ConvertOptions, ImageProcessorTrait, LetterboxPlacement};
    use edgefirst_tensor::{CpuAccess, Quantization, TensorMemory};

    fn image(w: usize, h: usize, fmt: PixelFormat, dtype: DType) -> TensorDyn {
        TensorDyn::image(
            w,
            h,
            fmt,
            dtype,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap()
    }

    fn source(w: usize, h: usize) -> TensorDyn {
        let src = image(w, h, PixelFormat::Rgba, DType::U8);
        {
            let mut m = src.as_u8().unwrap().map().unwrap();
            for (i, b) in m.as_mut_slice().iter_mut().enumerate() {
                *b = (i * 29 % 256) as u8;
            }
        }
        src
    }

    /// The u8 image a plain convert produces, as (element, channel) pairs.
    fn plain_u8(src: &TensorDyn, fmt: PixelFormat, opts: ConvertOptions) -> Vec<(u8, usize)> {
        let (w, h) = (src.width().unwrap() / 2, src.height().unwrap() / 2);
        let mut dst = image(w, h, fmt, DType::U8);
        let plain = ConvertOptions::new()
            .crop(opts.get_crop())
            .placement(opts.get_placement());
        CPUProcessor::default()
            .convert(src, &mut dst, Some(plain))
            .unwrap();
        let m = dst.as_u8().unwrap().map().unwrap();
        let c = fmt.channels();
        let planar = fmt.layout() == PixelLayout::Planar;
        m.as_slice()
            .iter()
            .enumerate()
            .map(|(i, &x)| (x, if planar { i / (w * h) } else { i % c }))
            .collect()
    }

    fn rgb_index(fmt: PixelFormat, c: usize) -> Option<usize> {
        match (fmt, c) {
            (_, 3) => None,
            (PixelFormat::Bgra, c) => Some(2 - c),
            (_, c) => Some(c),
        }
    }

    fn reference(n: &Normalization, fmt: PixelFormat, x: u8, c: usize) -> f64 {
        match rgb_index(fmt, c) {
            None => x as f64 / 255.0,
            Some(k) => (x as f64 - n.mean()[k] as f64) / n.std()[k] as f64,
        }
    }

    fn mapped(dst: &TensorDyn) -> Vec<f64> {
        match dst.dtype() {
            DType::F32 => dst
                .as_f32()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .iter()
                .map(|&v| v as f64)
                .collect(),
            DType::F16 => dst
                .as_f16()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .iter()
                .map(|v| v.to_f64())
                .collect(),
            DType::U8 => dst
                .as_u8()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .iter()
                .map(|&v| v as f64)
                .collect(),
            DType::I8 => dst
                .as_i8()
                .unwrap()
                .map()
                .unwrap()
                .as_slice()
                .iter()
                .map(|&v| v as f64)
                .collect(),
            _ => unreachable!(),
        }
    }

    #[test]
    fn normalization_matches_the_reference_for_every_dtype_and_layout() {
        let src = source(16, 12);
        let n = Normalization::imagenet();
        let q = Quantization::per_tensor(0.0187, -14);
        let formats = [
            PixelFormat::Rgb,
            PixelFormat::Rgba,
            PixelFormat::Bgra,
            PixelFormat::PlanarRgb,
            PixelFormat::Grey,
        ];
        for fmt in formats {
            let opts = ConvertOptions::new()
                .letterbox([114, 0, 255, 255])
                .normalization(n);
            let want = plain_u8(&src, fmt, opts);
            for dtype in [DType::F32, DType::F16, DType::U8, DType::I8] {
                let mut dst = image(8, 6, fmt, dtype);
                if matches!(dtype, DType::U8 | DType::I8) {
                    dst.set_quantization(q.clone()).unwrap();
                }
                CPUProcessor::default()
                    .convert(&src, &mut dst, Some(opts))
                    .unwrap();
                let got = mapped(&dst);
                assert_eq!(got.len(), want.len(), "{fmt:?} {dtype:?}");
                for (i, (&(x, c), g)) in want.iter().zip(&got).enumerate() {
                    let y = reference(&n, fmt, x, c);
                    let (lo, hi) = if dtype == DType::I8 {
                        (-128.0, 127.0)
                    } else {
                        (0.0, 255.0)
                    };
                    let (expect, tol) = match dtype {
                        DType::F32 => (y, 1e-5 * y.abs().max(1.0)),
                        DType::F16 => (y, 2e-3 * y.abs().max(1.0)),
                        // Alpha stays the raw byte for integer destinations.
                        _ if rgb_index(fmt, c).is_none() && dtype == DType::I8 => {
                            (x as i8 as f64, 0.0)
                        }
                        _ if rgb_index(fmt, c).is_none() => (x as f64, 0.0),
                        _ => ((y / 0.0187 - 14.0).round().clamp(lo, hi), 1.0),
                    };
                    assert!(
                        (g - expect).abs() <= tol,
                        "{fmt:?} {dtype:?} element {i}: x {x} c {c} got {g} want {expect}"
                    );
                }
            }
        }
    }

    #[test]
    fn default_options_keep_the_plain_mapping() {
        let src = source(16, 12);
        for dtype in [DType::U8, DType::I8, DType::F32] {
            let mut a = image(8, 6, PixelFormat::PlanarRgb, dtype);
            let mut b = image(8, 6, PixelFormat::PlanarRgb, dtype);
            CPUProcessor::default().convert(&src, &mut a, None).unwrap();
            // An explicit default normalization goes through the same path.
            let opts = ConvertOptions::new().normalization(Normalization::unsigned());
            CPUProcessor::default()
                .convert(&src, &mut b, Some(opts))
                .unwrap();
            assert_eq!(mapped(&a), mapped(&b), "{dtype:?}");
        }
    }

    #[test]
    fn top_left_letterbox_with_output_pad() {
        // 16x8 source into 8x8: content 8x4 at the top, output-domain pad below.
        let src = source(16, 8);
        let mut dst = image(8, 8, PixelFormat::PlanarRgb, DType::F32);
        let opts = ConvertOptions::new()
            .letterbox([114, 114, 114, 255])
            .placement(LetterboxPlacement::TopLeft)
            .pad_output([0.0; 4])
            .normalization(Normalization::imagenet());
        CPUProcessor::default()
            .convert(&src, &mut dst, Some(opts))
            .unwrap();
        let v = mapped(&dst);
        for c in 0..3 {
            let plane = &v[c * 64..(c + 1) * 64];
            assert!(plane[32..].iter().all(|&p| p == 0.0), "plane {c} pad rows");
            assert!(
                plane[..32].iter().any(|&p| p != 0.0),
                "plane {c} content rows"
            );
        }
    }

    #[test]
    fn integer_normalization_without_quantization_is_refused() {
        let src = source(16, 12);
        let mut dst = image(8, 6, PixelFormat::Rgb, DType::I8);
        let opts = ConvertOptions::new().normalization(Normalization::imagenet());
        let e = CPUProcessor::default().convert(&src, &mut dst, Some(opts));
        assert!(matches!(e, Err(Error::InvalidOptions(_))), "{e:?}");
    }
}
