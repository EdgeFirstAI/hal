// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{Error, Rect, Result};
use edgefirst_tensor::{Tensor, TensorMapTrait, TensorTrait};
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::ops::Shr;

use super::{CPUProcessor, ColorParams};

#[inline(always)]
pub(super) fn limit_to_full(l: u8) -> u8 {
    // Expand limited-range luma (16..=235, a 219-step swing) to full-range
    // (0..=255). Luma uses the 219 swing, NOT the 224 chroma swing — this must
    // match `colorimetry::yuv_to_rgb_coeffs` (255/219). Real decoded YUV (e.g.
    // JPEG → NV12) can carry luma below 16 or above 235, so clamp into the valid
    // limited range first to avoid u16 underflow on the `l - 16` term (and keep
    // the result within 0..=255).
    let l = (l as u16).clamp(16, 235);
    (((l - 16) * 255 + (235 - 16) / 2) / (235 - 16)) as u8
}

#[inline(always)]
pub(super) fn full_to_limit(l: u8) -> u8 {
    // Compress full-range luma (0..=255) into limited-range luma (16..=235,
    // the 219-step swing — luma max is 235, not the 240 chroma max).
    ((l as u16 * (235 - 16) + 255 / 2) / 255 + 16) as u8
}

/// Select the luma-decode mapping for grey/luma extraction. Limited-range
/// sources expand 16..=235 → 0..=255; full-range sources copy the byte as-is
/// (the luma channel is already the grey value).
#[inline(always)]
fn luma_mapper(full_range: bool) -> fn(u8) -> u8 {
    if full_range {
        |l| l
    } else {
        limit_to_full
    }
}

/// YUV↔RGB conversion mode for the `yuv` crate.
///
/// `Balanced` relies on Q15 rounded-doubling multiply-accumulate (`SQRDMLAH`,
/// Armv8.1 "rdm"). Cores without it — Cortex-A53/A35-class — emulate the op
/// with a `vqrdmulh`+`vqadd` pair per accumulate, roughly doubling the cost of
/// every conversion row. On those cores `Fast` (lower-precision integer
/// approximation, error ≤ ~2/255) is 2×+ faster and visually
/// indistinguishable; everywhere else `Balanced` keeps full precision.
#[inline]
fn yuv_mode() -> yuv::YuvConversionMode {
    #[cfg(target_arch = "aarch64")]
    {
        if !std::arch::is_aarch64_feature_detected!("rdm") {
            return yuv::YuvConversionMode::Fast;
        }
    }
    yuv::YuvConversionMode::Balanced
}

/// One row of a YUYV-destination convert, split into its whole macropixels and
/// the trailing unpaired pixel an odd width leaves over. See [`split_yuyv_row`].
type YuyvRowSplit<'s, 'd> = (&'s [u8], &'d mut [u8], Option<(&'s [u8], &'d mut [u8])>);

/// Split one row of a YUYV-destination convert into its whole macropixels and,
/// at an odd width, the trailing unpaired pixel.
///
/// YUYV packs two pixels into a 4-byte `[Y0, U, Y1, V]` macropixel, so an
/// odd-width row ends with a pixel that has no partner. Its destination is the
/// 2 bytes `[Y, U]` — a row is `width * 2` bytes, which leaves no room for the
/// trailing `V`, so that pixel's chroma is necessarily half-written whatever we
/// do. Encoding it anyway is still right: the alternative is leaving whatever
/// the destination held before, and a caller that did not pre-clear its buffer
/// then reads stale bytes as pixel data.
///
/// Returns `(paired_src, paired_dst, tail)`; `tail` is `None` at even widths.
fn split_yuyv_row<'s, 'd>(
    src: &'s [u8],
    dst: &'d mut [u8],
    src_bpp: usize,
) -> YuyvRowSplit<'s, 'd> {
    let pairs = dst.len() / 4;
    let (dst_pairs, dst_tail) = dst.split_at_mut(pairs * 4);
    let (src_pairs, src_tail) = src.split_at((pairs * 2 * src_bpp).min(src.len()));
    let tail = if dst_tail.len() >= 2 && src_tail.len() >= src_bpp {
        Some((src_tail, dst_tail))
    } else {
        None
    };
    (src_pairs, dst_pairs, tail)
}

/// Select the luma-encode mapping for grey→YUV. Full-range destinations keep
/// the grey value as Y directly; limited-range destinations compress it into
/// 16..=235.
#[inline(always)]
fn luma_encoder(full_range: bool) -> fn(u8) -> u8 {
    if full_range {
        |l| l
    } else {
        full_to_limit
    }
}

/// Fixed-point RGB→YUV coefficient table for the hand-rolled YUYV encoders,
/// resolved from the destination tensor's encoding (`cp.encoding`) and range
/// (`cp.range_kind`). All terms are `Q(BIAS)` fixed-point; `y_off`/`c_off` are
/// the post-shift integer offsets.
struct YuyvEncodeCoeffs {
    y_r: i32,
    y_g: i32,
    y_b: i32,
    u_r: i32,
    u_g: i32,
    u_b: i32,
    v_r: i32,
    v_g: i32,
    v_b: i32,
    y_off: i32,
    c_off: i32,
}

impl YuyvEncodeCoeffs {
    /// `BIAS` is Q20 fixed point — retained from the pre-refactor hand-coded
    /// tables to keep encoder output byte-identical.
    const BIAS: i32 = 20;
    const ROUND: i32 = 1 << (Self::BIAS - 1);
    const ROUND2: i32 = 1 << Self::BIAS;

    /// Build the table from the resolved `ColorParams`. The luma/chroma swings
    /// are full-range (255/255) or limited-range (219/224) per `cp.range_kind`;
    /// the `KR`/`KB` luma weights come from `cp.encoding` (BT.601 / 709 / 2020).
    fn from_params(cp: ColorParams) -> Self {
        // KR/KB luma weights and luma/chroma swings come from the canonical
        // source in `edgefirst_tensor::colorimetry`, shared with the in-shader
        // GL coefficients (see `crate::colorimetry::yuv_to_rgb_coeffs`).
        let w = cp.encoding.luma_weights();
        let (kr, kb) = (w.kr, w.kb);
        let kg = w.kg();
        let s = cp.range_kind.scaling();
        // Chroma is always centred on 128; the luma black level (`y_off`) and
        // the swings come from the resolved range.
        let (y_swing, c_swing, y_off, c_off) = (s.y_swing, s.c_swing, s.y_offset as i32, 128);
        let b = Self::BIAS;
        let yscale = (1_i64 << b) as f64 * y_swing / 255.0;
        let cscale = (1_i64 << b) as f64 * c_swing / 255.0;
        Self {
            y_r: (kr * yscale).round() as i32,
            y_g: (kg * yscale).round() as i32,
            y_b: (kb * yscale).round() as i32,
            u_r: (-kr / (kr + kg) / 2.0 * cscale).round() as i32,
            u_g: (-kg / (kr + kg) / 2.0 * cscale).round() as i32,
            u_b: (0.5 * cscale).ceil() as i32,
            v_r: (0.5 * cscale).ceil() as i32,
            v_g: (-kg / (kg + kb) / 2.0 * cscale).round() as i32,
            v_b: (-kb / (kg + kb) / 2.0 * cscale).round() as i32,
            y_off,
            c_off,
        }
    }

    /// Encode two adjacent RGB pixels into a YUYV macropixel `[Y0,U,Y1,V]`,
    /// matching the original subsampled-chroma averaging.
    #[inline(always)]
    fn encode_pair(&self, p0: [i32; 3], p1: [i32; 3]) -> [u8; 4] {
        let [r0, g0, b0] = p0;
        let [r1, g1, b1] = p1;
        let b = Self::BIAS;
        let y0 = ((self.y_r * r0 + self.y_g * g0 + self.y_b * b0 + Self::ROUND).shr(b) + self.y_off)
            as u8;
        let y1 = ((self.y_r * r1 + self.y_g * g1 + self.y_b * b1 + Self::ROUND).shr(b) + self.y_off)
            as u8;
        let u = ((self.u_r * r0
            + self.u_g * g0
            + self.u_b * b0
            + self.u_r * r1
            + self.u_g * g1
            + self.u_b * b1
            + Self::ROUND2)
            .shr(b + 1)
            + self.c_off) as u8;
        let v = ((self.v_r * r0
            + self.v_g * g0
            + self.v_b * b0
            + self.v_r * r1
            + self.v_g * g1
            + self.v_b * b1
            + Self::ROUND2)
            .shr(b + 1)
            + self.c_off) as u8;
        [y0, u, y1, v]
    }

    /// Encode a single RGB pixel into `[Y, U, Y, V]` (no chroma subsampling) —
    /// used for solid fill colors.
    #[inline(always)]
    fn encode_single(&self, rgb: [i32; 3]) -> [u8; 4] {
        let [r, g, b] = rgb;
        let bias = Self::BIAS;
        let y = (((self.y_r * r + self.y_g * g + self.y_b * b + Self::ROUND) >> bias) + self.y_off)
            as u8;
        let u = (((self.u_r * r + self.u_g * g + self.u_b * b + Self::ROUND) >> bias) + self.c_off)
            as u8;
        let v = (((self.v_r * r + self.v_g * g + self.v_b * b + Self::ROUND) >> bias) + self.c_off)
            as u8;
        [y, u, y, v]
    }
}

/// Scatter a packed `src_ch`-channel image into single-channel destination
/// planes, honouring **both** source and destination row strides — rows are
/// pitch-padded on DMA/IOSurface tensors (and `None`-memory tensors auto-select
/// DMA on i.MX), so a flat `as_slice()` read shears the image. `plane_src[p]`
/// selects the source channel copied into plane `p`, or `None` to fill that
/// plane with a constant `255` (the alpha plane of a planar-RGBA destination).
/// Each plane only touches the `w` logical bytes of every row; planes run
/// concurrently.
fn pack_to_planar(
    src: &Tensor<u8>,
    dst: &mut Tensor<u8>,
    src_ch: usize,
    plane_src: &[Option<usize>],
) -> Result<()> {
    let w = src.width().unwrap_or(0);
    let h = src.height().unwrap_or(0);
    let src_stride = super::tensor_row_stride(src);
    let dst_stride = super::tensor_row_stride(dst);
    let src_map = src.map_read()?;
    let src_bytes = src_map.as_slice();
    let mut dst_map = dst.map_mut()?;
    let dst_bytes = dst_map.as_mut_slice();

    // Validate the mapped buffers against the derived geometry before indexing,
    // so a malformed/untrusted tensor yields `InvalidShape` instead of a panic
    // (mirrors `planar_to_packed` / `split_semi_planar`).
    let src_row = w.checked_mul(src_ch).ok_or_else(|| {
        Error::InvalidShape(format!(
            "pack_to_planar src row overflow (w={w}, ch={src_ch})"
        ))
    })?;
    // Each destination plane is `h` rows of `dst_stride` bytes (the row pitch).
    let plane = dst_stride.checked_mul(h).ok_or_else(|| {
        Error::InvalidShape(format!(
            "pack_to_planar plane size overflow (stride={dst_stride}, h={h})"
        ))
    })?;
    let src_need = src_stride.checked_mul(h).ok_or_else(|| {
        Error::InvalidShape(format!(
            "pack_to_planar src size overflow (stride={src_stride}, h={h})"
        ))
    })?;
    let dst_need = plane.checked_mul(plane_src.len()).ok_or_else(|| {
        Error::InvalidShape(format!(
            "pack_to_planar dst size overflow (plane={plane}, planes={})",
            plane_src.len()
        ))
    })?;
    if src_row > src_stride || src_bytes.len() < src_need || dst_bytes.len() < dst_need {
        return Err(Error::InvalidShape(format!(
            "pack_to_planar geometry exceeds buffers: src {} (need {src_need}), dst {} (need \
             {dst_need}), row {src_row} vs stride {src_stride} (w={w}, h={h}, src_ch={src_ch})",
            src_bytes.len(),
            dst_bytes.len()
        )));
    }
    if plane == 0 {
        return Ok(()); // zero-height / empty image: nothing to scatter
    }

    // Fast path: identity colour mapping (R,G,B ← src channels 0,1,2) with a
    // 3- or 4-channel packed source, and either no alpha plane, a constant
    // alpha plane (RGB → PlanarRgba), or an alpha plane copied from src
    // channel 3 (RGBA → PlanarRgba). This covers every current caller. A single
    // NEON deinterleaving pass reads the packed source once and writes all
    // planes, replacing the per-plane scalar gather (which re-read the source
    // once per plane). Parallelism moves from per-plane to per-row-strip.
    let n_planes = plane_src.len();
    let identity_rgb = n_planes >= 3
        && plane_src[0] == Some(0)
        && plane_src[1] == Some(1)
        && plane_src[2] == Some(2);
    let alpha_from_src = n_planes == 4 && plane_src[3] == Some(3);
    let const_alpha = n_planes == 4 && plane_src[3].is_none();
    let fast = identity_rgb
        && (src_ch == 3 || src_ch == 4)
        && (n_planes == 3 || (alpha_from_src && src_ch == 4) || const_alpha);

    if fast {
        let mut planes = dst_bytes.chunks_mut(plane).take(n_planes);
        let rp = planes.next().unwrap();
        let gp = planes.next().unwrap();
        let bp = planes.next().unwrap();
        let ap = planes.next(); // Some(_) only when n_planes == 4
        let src_rows = &src_bytes[..h * src_stride];

        // Serial, one pass per row: the NEON `vld3`/`vld4` deinterleave reads
        // the packed source once and is memory-bandwidth-bound, so on Orin's
        // shared bus row-level rayon parallelism measured no faster than serial
        // for the model-preprocessing sizes (≤1080p) while adding scheduling
        // overhead on small frames. The rare arbitrary-mapping path below keeps
        // its per-plane parallelism.
        match ap {
            Some(ap) if alpha_from_src => {
                src_rows
                    .chunks(src_stride)
                    .zip(rp.chunks_mut(dst_stride))
                    .zip(gp.chunks_mut(dst_stride))
                    .zip(bp.chunks_mut(dst_stride))
                    .zip(ap.chunks_mut(dst_stride))
                    .for_each(|((((s, r), g), b), a)| {
                        super::simd::deinterleave_row(s, r, g, b, Some(a), w, src_ch);
                    });
            }
            other => {
                if let Some(ap) = other {
                    // Constant alpha plane (RGB → PlanarRgba): fill only the `w`
                    // logical bytes of each row, matching this function's
                    // contract and the slow-path `None` handling (leave per-row
                    // padding untouched rather than filling the whole plane).
                    for row in ap.chunks_mut(dst_stride).take(h) {
                        row[..w].fill(255);
                    }
                }
                src_rows
                    .chunks(src_stride)
                    .zip(rp.chunks_mut(dst_stride))
                    .zip(gp.chunks_mut(dst_stride))
                    .zip(bp.chunks_mut(dst_stride))
                    .for_each(|(((s, r), g), b)| {
                        super::simd::deinterleave_row(s, r, g, b, None, w, src_ch);
                    });
            }
        }
        return Ok(());
    }

    let plane_slices: Vec<&mut [u8]> = dst_bytes.chunks_mut(plane).take(plane_src.len()).collect();
    rayon::scope(|sc| {
        for (pb, &chan) in plane_slices.into_iter().zip(plane_src.iter()) {
            sc.spawn(move |_| match chan {
                Some(c) => {
                    for row in 0..h {
                        let s = &src_bytes[row * src_stride..row * src_stride + w * src_ch];
                        let d = &mut pb[row * dst_stride..row * dst_stride + w];
                        for x in 0..w {
                            d[x] = s[x * src_ch + c];
                        }
                    }
                }
                None => {
                    for row in 0..h {
                        pb[row * dst_stride..row * dst_stride + w].fill(255);
                    }
                }
            });
        }
    });
    Ok(())
}

impl CPUProcessor {
    /// Shared decode for every semi-planar (NV12/NV16/NV24) → packed
    /// conversion: wrap the already-resolved planes/strides in a
    /// `YuvBiPlanarImage` and run the format-specific `yuv` kernel. `decode`
    /// is a closure that forwards to the right `yuv::yuv_nvXX_to_rgb[a]` with
    /// the matrix/range bound from `ColorParams`; only the plane geometry
    /// (resolved by the `convert_nvXX` wrappers) differs between formats.
    #[allow(clippy::too_many_arguments)]
    fn semi_planar_decode<F>(
        y_plane: &[u8],
        uv_plane: &[u8],
        width: usize,
        height: usize,
        y_stride: usize,
        uv_stride: usize,
        dst: &mut Tensor<u8>,
        decode: F,
    ) -> Result<()>
    where
        F: FnOnce(
            &yuv::YuvBiPlanarImage<u8>,
            &mut [u8],
            u32,
        ) -> std::result::Result<(), yuv::YuvError>,
    {
        let src = yuv::YuvBiPlanarImage {
            y_plane,
            y_stride: y_stride as u32,
            uv_plane,
            uv_stride: uv_stride as u32,
            width: width as u32,
            height: height as u32,
        };
        let dst_stride = super::tensor_row_stride(dst) as u32;
        Ok(decode(&src, dst.map_mut()?.as_mut_slice(), dst_stride)?)
    }

    /// Resolve an NV12 (4:2:0) source's planes/strides and decode. The chroma
    /// plane is half-height with one `(Cb,Cr)` pair per two luma columns ⇒ the
    /// same row pitch as luma. A true-multiplane source reads the chroma
    /// plane's own stride (a raw tensor whose `effective_row_stride()` has no
    /// width fallback — default to even(width)); the contiguous buffer's two
    /// planes share the one stride.
    fn convert_nv12<F>(src: &Tensor<u8>, dst: &mut Tensor<u8>, decode: F) -> Result<()>
    where
        F: FnOnce(
            &yuv::YuvBiPlanarImage<u8>,
            &mut [u8],
            u32,
        ) -> std::result::Result<(), yuv::YuvError>,
    {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let stride = src
            .effective_row_stride()
            .unwrap_or(src_w.next_multiple_of(2));
        if src.is_multiplane() {
            let y_map = src.map_read()?;
            let uv_map = src.chroma().unwrap().map_read()?;
            let uv_stride = src
                .chroma()
                .unwrap()
                .effective_row_stride()
                .unwrap_or(src_w.next_multiple_of(2));
            Self::semi_planar_decode(
                y_map.as_slice(),
                uv_map.as_slice(),
                src_w,
                src_h,
                stride,
                uv_stride,
                dst,
                decode,
            )
        } else {
            let map = src.map_read()?;
            let (y_plane, uv_plane) = super::split_semi_planar(
                map.as_slice(),
                stride,
                src_h,
                src.format().expect("semi-planar source has a pixel format"),
            )?;
            Self::semi_planar_decode(y_plane, uv_plane, src_w, src_h, stride, stride, dst, decode)
        }
    }

    pub(super) fn convert_nv12_to_rgb(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        Self::convert_nv12(src, dst, |img, out, stride| {
            yuv::yuv_nv12_to_rgb(img, out, stride, cp.range, cp.matrix, yuv_mode())
        })
    }

    // NOTE: The `*_to_rgba` helpers below all accept BGRA destinations.
    // They always write pixels in RGBA channel order; for BGRA destinations the
    // caller applies an R<->B swizzle afterwards via `swizzle_rb_4chan`.
    pub(super) fn convert_nv12_to_rgba(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        Self::convert_nv12(src, dst, |img, out, stride| {
            yuv::yuv_nv12_to_rgba(img, out, stride, cp.range, cp.matrix, yuv_mode())
        })
    }

    pub(super) fn convert_nv12_to_grey(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        // NV12→GREY drops chroma and copies the luma plane (the first `src_h`
        // rows). Honour the source row stride so padded buffers and odd widths
        // are handled correctly, and the destination grey stride so we write a
        // tightly-packed [H, W] output.
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let src_stride = super::tensor_row_stride(src);
        let dst_stride = super::tensor_row_stride(dst);

        // Full-range luma is copied directly; limited-range luma is expanded.
        let luma = luma_mapper(cp.src_full_range);

        let src_map = src.map_read()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map_mut()?;
        let dst_bytes = dst_map.as_mut_slice();
        super::guard_plane(src_bytes.len(), src_stride, src_h, src_w, "nv12→grey src")?;
        super::guard_plane(dst_bytes.len(), dst_stride, src_h, src_w, "nv12→grey dst")?;

        for row in 0..src_h {
            let s = &src_bytes[row * src_stride..][..src_w];
            let d = &mut dst_bytes[row * dst_stride..][..src_w];
            let (s_chunks, s_rem) = s.as_chunks::<8>();
            let (d_chunks, d_rem) = d.as_chunks_mut::<8>();
            for (sc, dc) in s_chunks.iter().zip(d_chunks) {
                sc.iter().zip(dc).for_each(|(s, d)| *d = luma(*s));
            }
            for (s, d) in s_rem.iter().zip(d_rem) {
                *d = luma(*s);
            }
        }

        Ok(())
    }

    pub(super) fn convert_yuyv_to_rgb(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let src_rs = super::tensor_row_stride(src);
        let src = yuv::YuvPackedImage::<u8> {
            yuy: &src.map_read()?,
            yuy_stride: src_rs as u32,
            width: src_w as u32,
            height: src_h as u32,
        };

        let dst_rs = super::tensor_row_stride(dst);
        Ok(yuv::yuyv422_to_rgb(
            &src,
            dst.map_mut()?.as_mut_slice(),
            dst_rs as u32,
            cp.range,
            cp.matrix,
        )?)
    }

    pub(super) fn convert_yuyv_to_rgba(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let src_rs = super::tensor_row_stride(src);
        let src = yuv::YuvPackedImage::<u8> {
            yuy: &src.map_read()?,
            yuy_stride: src_rs as u32,
            width: src_w as u32,
            height: src_h as u32,
        };

        Ok(yuv::yuyv422_to_rgba(
            &src,
            dst.map_mut()?.as_mut_slice(),
            super::tensor_row_stride(dst) as u32,
            cp.range,
            cp.matrix,
        )?)
    }

    pub(super) fn convert_yuyv_to_8bps(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let mut tmp = Tensor::<u8>::image(
            src_w,
            src_h,
            edgefirst_tensor::PixelFormat::Rgb,
            Some(edgefirst_tensor::TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        Self::convert_yuyv_to_rgb(src, &mut tmp, cp)?;
        Self::convert_rgb_to_8bps(&tmp, dst)
    }

    pub(super) fn convert_yuyv_to_prgba(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let mut tmp = Tensor::<u8>::image(
            src_w,
            src_h,
            edgefirst_tensor::PixelFormat::Rgb,
            Some(edgefirst_tensor::TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        Self::convert_yuyv_to_rgb(src, &mut tmp, cp)?;
        Self::convert_rgb_to_prgba(&tmp, dst)
    }

    pub(super) fn convert_yuyv_to_grey(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        // YUYV→GREY keeps the luma samples and drops chroma. Honour the source
        // row stride so padded/odd-width buffers are not read across row
        // boundaries (a flat `as_chunks` over the whole map ignores stride and
        // reads pad bytes as luma — see EDGEAI stride-handling fix).
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let src_stride = super::tensor_row_stride(src);
        let dst_stride = super::tensor_row_stride(dst);
        let luma = luma_mapper(cp.src_full_range);

        // Each macropixel is 2 bytes/px; check the row width for overflow so a
        // malformed width can't wrap and slip past `guard_plane`.
        let src_row = src_w.checked_mul(2).ok_or_else(|| {
            Error::InvalidShape(format!("yuyv→grey src row overflow (w={src_w})"))
        })?;
        let src_map = src.map_read()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map_mut()?;
        let dst_bytes = dst_map.as_mut_slice();
        super::guard_plane(src_bytes.len(), src_stride, src_h, src_row, "yuyv→grey src")?;
        super::guard_plane(dst_bytes.len(), dst_stride, src_h, src_w, "yuyv→grey dst")?;

        // YUYV byte order per macropixel: [Y0, U, Y1, V] — luma at even offsets.
        for row in 0..src_h {
            let s = &src_bytes[row * src_stride..][..src_row];
            let d = &mut dst_bytes[row * dst_stride..][..src_w];
            for (x, dx) in d.iter_mut().enumerate() {
                *dx = luma(s[x * 2]);
            }
        }
        Ok(())
    }

    pub(super) fn convert_yuyv_to_nv16(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let src_rs = super::tensor_row_stride(src);
        let dst_w = dst.width().unwrap();
        let dst_stride = super::tensor_row_stride(dst);
        let dst_h = if dst.is_multiplane() {
            dst.shape()[0]
        } else {
            dst.shape()[0] / 2
        };
        let src_map = src.map_read()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map_mut()?;
        // Split at the stride-aligned luma plane boundary, not the tight one,
        // validating the destination holds the full combined plane first.
        let (y_plane, uv_plane) = super::split_semi_planar_mut(
            dst_map.as_mut_slice(),
            dst_stride,
            dst_h,
            edgefirst_tensor::PixelFormat::Nv16,
        )?;

        // YUYV byte order per two-pixel macropixel: [Y0, Cb, Y1, Cr].
        // The NV16 chroma row is `even(dst_w)` bytes wide (one (Cb,Cr) pair per
        // 2 luma columns, rounded up), so slice the UV row to the even width.
        let chroma_w = dst_w.next_multiple_of(2);
        for row in 0..src_h {
            let src_row = &src_bytes[row * src_rs..row * src_rs + src_w * 2];
            let y_row = &mut y_plane[row * dst_stride..row * dst_stride + dst_w];
            let uv_row = &mut uv_plane[row * dst_stride..row * dst_stride + chroma_w];
            let mut xi = 0usize;
            let mut si = 0usize;
            while xi + 1 < dst_w {
                y_row[xi] = src_row[si]; // Y0
                y_row[xi + 1] = src_row[si + 2]; // Y1
                uv_row[xi] = src_row[si + 1]; // Cb
                uv_row[xi + 1] = src_row[si + 3]; // Cr
                xi += 2;
                si += 4;
            }
            // Odd width: one trailing lone pixel. Write its Y and the full
            // (Cb,Cr) chroma pair so the even-width chroma row is fully
            // initialized; the lone pixel has no second-Y Cr in the source, so
            // replicate Cb when absent.
            if xi < dst_w && si + 1 < src_row.len() {
                y_row[xi] = src_row[si];
                uv_row[xi] = src_row[si + 1]; // Cb
                uv_row[xi + 1] = src_row.get(si + 3).copied().unwrap_or(src_row[si + 1]);
            }
        }
        Ok(())
    }

    pub(super) fn convert_vyuy_to_rgb(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let src_rs = super::tensor_row_stride(src);
        let src = yuv::YuvPackedImage::<u8> {
            yuy: &src.map_read()?,
            yuy_stride: src_rs as u32,
            width: src_w as u32,
            height: src_h as u32,
        };

        let dst_rs = super::tensor_row_stride(dst);
        Ok(yuv::vyuy422_to_rgb(
            &src,
            dst.map_mut()?.as_mut_slice(),
            dst_rs as u32,
            cp.range,
            cp.matrix,
        )?)
    }

    pub(super) fn convert_vyuy_to_rgba(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let src_rs = super::tensor_row_stride(src);
        let src = yuv::YuvPackedImage::<u8> {
            yuy: &src.map_read()?,
            yuy_stride: src_rs as u32,
            width: src_w as u32,
            height: src_h as u32,
        };

        Ok(yuv::vyuy422_to_rgba(
            &src,
            dst.map_mut()?.as_mut_slice(),
            super::tensor_row_stride(dst) as u32,
            cp.range,
            cp.matrix,
        )?)
    }

    pub(super) fn convert_vyuy_to_8bps(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let mut tmp = Tensor::<u8>::image(
            src_w,
            src_h,
            edgefirst_tensor::PixelFormat::Rgb,
            Some(edgefirst_tensor::TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        Self::convert_vyuy_to_rgb(src, &mut tmp, cp)?;
        Self::convert_rgb_to_8bps(&tmp, dst)
    }

    pub(super) fn convert_vyuy_to_prgba(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let mut tmp = Tensor::<u8>::image(
            src_w,
            src_h,
            edgefirst_tensor::PixelFormat::Rgb,
            Some(edgefirst_tensor::TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )?;
        Self::convert_vyuy_to_rgb(src, &mut tmp, cp)?;
        Self::convert_rgb_to_prgba(&tmp, dst)
    }

    pub(super) fn convert_vyuy_to_grey(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        // VYUY→GREY keeps the luma samples and drops chroma. Honour the source
        // row stride so padded/odd-width buffers are not read across row
        // boundaries (a flat `as_chunks` over the whole map ignores stride).
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let src_stride = super::tensor_row_stride(src);
        let dst_stride = super::tensor_row_stride(dst);
        let luma = luma_mapper(cp.src_full_range);

        // Each macropixel is 2 bytes/px; check the row width for overflow so a
        // malformed width can't wrap and slip past `guard_plane`.
        let src_row = src_w.checked_mul(2).ok_or_else(|| {
            Error::InvalidShape(format!("vyuy→grey src row overflow (w={src_w})"))
        })?;
        let src_map = src.map_read()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map_mut()?;
        let dst_bytes = dst_map.as_mut_slice();
        super::guard_plane(src_bytes.len(), src_stride, src_h, src_row, "vyuy→grey src")?;
        super::guard_plane(dst_bytes.len(), dst_stride, src_h, src_w, "vyuy→grey dst")?;

        // VYUY byte order per macropixel: [V, Y0, U, Y1] — luma at odd offsets.
        for row in 0..src_h {
            let s = &src_bytes[row * src_stride..][..src_row];
            let d = &mut dst_bytes[row * dst_stride..][..src_w];
            for (x, dx) in d.iter_mut().enumerate() {
                *dx = luma(s[x * 2 + 1]);
            }
        }
        Ok(())
    }

    pub(super) fn convert_vyuy_to_nv16(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let src_rs = super::tensor_row_stride(src);
        let dst_w = dst.width().unwrap();
        let dst_stride = super::tensor_row_stride(dst);
        let dst_h = if dst.is_multiplane() {
            dst.shape()[0]
        } else {
            dst.shape()[0] / 2
        };
        let src_map = src.map_read()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map_mut()?;
        // Split at the stride-aligned luma plane boundary, not the tight one,
        // validating the destination holds the full combined plane first.
        let (y_plane, uv_plane) = super::split_semi_planar_mut(
            dst_map.as_mut_slice(),
            dst_stride,
            dst_h,
            edgefirst_tensor::PixelFormat::Nv16,
        )?;

        // VYUY byte order per two-pixel macropixel: [V, Y0, U, Y1]. The NV16
        // chroma row is `even(dst_w)` bytes wide, so slice UV to the even width.
        let chroma_w = dst_w.next_multiple_of(2);
        for row in 0..src_h {
            let src_row = &src_bytes[row * src_rs..row * src_rs + src_w * 2];
            let y_row = &mut y_plane[row * dst_stride..row * dst_stride + dst_w];
            let uv_row = &mut uv_plane[row * dst_stride..row * dst_stride + chroma_w];
            let mut xi = 0usize;
            let mut si = 0usize;
            while xi + 1 < dst_w {
                y_row[xi] = src_row[si + 1]; // Y0
                y_row[xi + 1] = src_row[si + 3]; // Y1
                uv_row[xi] = src_row[si + 2]; // U (Cb)
                uv_row[xi + 1] = src_row[si]; // V (Cr)
                xi += 2;
                si += 4;
            }
            // Odd width: one trailing lone pixel — write Y and the full (Cb,Cr)
            // pair (both are present in this macropixel's V,Y0,U bytes) so the
            // even-width chroma row is fully initialized.
            if xi < dst_w && si + 2 < src_row.len() {
                y_row[xi] = src_row[si + 1];
                uv_row[xi] = src_row[si + 2]; // U (Cb)
                uv_row[xi + 1] = src_row[si]; // V (Cr)
            }
        }
        Ok(())
    }

    pub(super) fn convert_grey_to_rgb(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let src_rs = super::tensor_row_stride(src);
        let src = yuv::YuvGrayImage::<u8> {
            y_plane: &src.map_read()?,
            y_stride: src_rs as u32,
            width: src_w as u32,
            height: src_h as u32,
        };
        Ok(yuv::yuv400_to_rgb(
            &src,
            dst.map_mut()?.as_mut_slice(),
            super::tensor_row_stride(dst) as u32,
            yuv::YuvRange::Full,
            yuv::YuvStandardMatrix::Bt601,
        )?)
    }

    pub(super) fn convert_grey_to_rgba(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let src_rs = super::tensor_row_stride(src);
        let src = yuv::YuvGrayImage::<u8> {
            y_plane: &src.map_read()?,
            y_stride: src_rs as u32,
            width: src_w as u32,
            height: src_h as u32,
        };
        Ok(yuv::yuv400_to_rgba(
            &src,
            dst.map_mut()?.as_mut_slice(),
            super::tensor_row_stride(dst) as u32,
            yuv::YuvRange::Full,
            yuv::YuvStandardMatrix::Bt601,
        )?)
    }

    pub(super) fn convert_grey_to_8bps(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        // Grey broadcast into R, G, B planes.
        pack_to_planar(src, dst, 1, &[Some(0), Some(0), Some(0)])
    }

    pub(super) fn convert_grey_to_prgba(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        // Grey broadcast into R, G, B planes + constant alpha plane.
        pack_to_planar(src, dst, 1, &[Some(0), Some(0), Some(0), None])
    }

    pub(super) fn convert_grey_to_yuyv(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        // Full-range luma maps directly into Y; limited-range compresses it.
        let y_enc = luma_encoder(cp.dst_full_range);
        Self::for_each_row(src, dst, "grey→yuyv", |s, d| {
            let (s, d, tail) = split_yuyv_row(s, d, 1);
            for (s, d) in s
                .as_chunks::<2>()
                .0
                .iter()
                .zip(d.as_chunks_mut::<4>().0.iter_mut())
            {
                d[0] = y_enc(s[0]);
                d[1] = 128;

                d[2] = y_enc(s[1]);
                d[3] = 128;
            }
            if let Some((s, d)) = tail {
                d[0] = y_enc(s[0]);
                d[1] = 128;
            }
        })
    }

    pub(super) fn convert_grey_to_nv16(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let y_enc = luma_encoder(cp.dst_full_range);
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let src_stride = super::tensor_row_stride(src);
        let dst_stride = super::tensor_row_stride(dst);
        let src_map = src.map_read()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map_mut()?;
        let dst_bytes = dst_map.as_mut_slice();
        // NV16 luma plane: src_h rows, then UV plane: another src_h rows.
        // Validate the destination holds the full combined plane before splitting.
        let (y_plane, uv_plane) = super::split_semi_planar_mut(
            dst_bytes,
            dst_stride,
            src_h,
            edgefirst_tensor::PixelFormat::Nv16,
        )?;

        for row in 0..src_h {
            // Copy luma row, respecting source and destination strides.
            let src_row = &src_bytes[row * src_stride..row * src_stride + src_w];
            let y_row = &mut y_plane[row * dst_stride..row * dst_stride + src_w];
            for (s, d) in src_row.iter().zip(y_row.iter_mut()) {
                *d = y_enc(*s);
            }
            // UV row: neutral chroma (128 = no colour)
            let uv_row = &mut uv_plane[row * dst_stride..row * dst_stride + src_w];
            uv_row.fill(128);
        }

        Ok(())
    }

    pub(super) fn convert_rgba_to_rgb(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let src_rs = super::tensor_row_stride(src);
        let dst_rs = super::tensor_row_stride(dst);
        Ok(yuv::rgba_to_rgb(
            src.map_read()?.as_slice(),
            src_rs as u32,
            dst.map_mut()?.as_mut_slice(),
            dst_rs as u32,
            src_w as u32,
            src_h as u32,
        )?)
    }

    pub(super) fn convert_rgba_to_grey(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let dst_w = dst.width().unwrap();
        let dst_h = dst.height().unwrap();
        let dst_rs = super::tensor_row_stride(dst);
        let src_rs = super::tensor_row_stride(src);
        let mut dst = yuv::YuvGrayImageMut::<u8> {
            y_plane: yuv::BufferStoreMut::Borrowed(&mut dst.map_mut()?),
            y_stride: dst_rs as u32,
            width: dst_w as u32,
            height: dst_h as u32,
        };
        Ok(yuv::rgba_to_yuv400(
            &mut dst,
            src.map_read()?.as_slice(),
            src_rs as u32,
            yuv::YuvRange::Full,
            yuv::YuvStandardMatrix::Bt601,
        )?)
    }

    pub(super) fn convert_rgba_to_8bps(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        // RGBA → R, G, B planes (alpha dropped).
        pack_to_planar(src, dst, 4, &[Some(0), Some(1), Some(2)])
    }

    pub(super) fn convert_rgba_to_prgba(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        // RGBA → R, G, B, A planes.
        pack_to_planar(src, dst, 4, &[Some(0), Some(1), Some(2), Some(3)])
    }

    pub(super) fn convert_rgba_to_yuyv(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        // RGB→YUV coefficients resolved from the destination colorimetry.
        let c = YuyvEncodeCoeffs::from_params(cp);
        let process_rgba_to_yuyv = |s: &[u8; 8], d: &mut [u8; 4]| {
            let [r0, g0, b0, _, r1, g1, b1, _] = *s;
            *d = c.encode_pair(
                [r0 as i32, g0 as i32, b0 as i32],
                [r1 as i32, g1 as i32, b1 as i32],
            );
        };

        Self::for_each_row(src, dst, "rgba→yuyv", |src, dst| {
            let (src, dst, tail) = split_yuyv_row(src, dst, 4);
            let src = src.as_chunks::<{ 8 * 32 }>();
            let dst = dst.as_chunks_mut::<{ 4 * 32 }>();

            for (s, d) in src.0.iter().zip(dst.0.iter_mut()) {
                let s = s.as_chunks::<8>().0;
                let d = d.as_chunks_mut::<4>().0;
                for (s, d) in s.iter().zip(d.iter_mut()) {
                    process_rgba_to_yuyv(s, d);
                }
            }

            let s = src.1.as_chunks::<8>().0;
            let d = dst.1.as_chunks_mut::<4>().0;
            for (s, d) in s.iter().zip(d.iter_mut()) {
                process_rgba_to_yuyv(s, d);
            }

            if let Some((s, d)) = tail {
                let mut pair = [0u8; 4];
                process_rgba_to_yuyv(&[s[0], s[1], s[2], s[3], s[0], s[1], s[2], s[3]], &mut pair);
                d[0] = pair[0];
                d[1] = pair[1];
            }
        })
    }

    pub(super) fn convert_rgba_to_nv16(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let dst_w = dst.width().unwrap();
        let dst_h = if dst.is_multiplane() {
            dst.shape()[0]
        } else {
            dst.shape()[0] / 2
        };
        let src_rs = super::tensor_row_stride(src);
        let dst_stride = super::tensor_row_stride(dst);
        let mut dst_map = dst.map_mut()?;

        // Split at the stride-aligned luma plane boundary, not the tight one,
        // validating the destination holds the full combined plane first.
        let (y_plane, uv_plane) = super::split_semi_planar_mut(
            dst_map.as_mut_slice(),
            dst_stride,
            dst_h,
            edgefirst_tensor::PixelFormat::Nv16,
        )?;
        let mut bi_planar_image = yuv::YuvBiPlanarImageMut::<u8> {
            y_plane: yuv::BufferStoreMut::Borrowed(y_plane),
            y_stride: dst_stride as u32,
            uv_plane: yuv::BufferStoreMut::Borrowed(uv_plane),
            uv_stride: dst_stride as u32,
            width: dst_w as u32,
            height: dst_h as u32,
        };

        Ok(yuv::rgba_to_yuv_nv16(
            &mut bi_planar_image,
            src.map_read()?.as_slice(),
            src_rs as u32,
            cp.range,
            cp.matrix,
            yuv_mode(),
        )?)
    }

    pub(super) fn convert_rgb_to_rgba(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let src_rs = super::tensor_row_stride(src);
        let dst_rs = super::tensor_row_stride(dst);
        Ok(yuv::rgb_to_rgba(
            src.map_read()?.as_slice(),
            src_rs as u32,
            dst.map_mut()?.as_mut_slice(),
            dst_rs as u32,
            src_w as u32,
            src_h as u32,
        )?)
    }

    pub(super) fn convert_rgb_to_grey(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let dst_w = dst.width().unwrap();
        let dst_h = dst.height().unwrap();
        let dst_rs = super::tensor_row_stride(dst);
        let src_rs = super::tensor_row_stride(src);
        let mut dst = yuv::YuvGrayImageMut::<u8> {
            y_plane: yuv::BufferStoreMut::Borrowed(&mut dst.map_mut()?),
            y_stride: dst_rs as u32,
            width: dst_w as u32,
            height: dst_h as u32,
        };
        Ok(yuv::rgb_to_yuv400(
            &mut dst,
            src.map_read()?.as_slice(),
            src_rs as u32,
            yuv::YuvRange::Full,
            yuv::YuvStandardMatrix::Bt601,
        )?)
    }

    pub(super) fn convert_rgb_to_8bps(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        // RGB → R, G, B planes.
        pack_to_planar(src, dst, 3, &[Some(0), Some(1), Some(2)])
    }

    pub(super) fn convert_rgb_to_prgba(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        // RGB → R, G, B planes + constant alpha plane.
        pack_to_planar(src, dst, 3, &[Some(0), Some(1), Some(2), None])
    }

    pub(super) fn convert_rgb_to_yuyv(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        // RGB→YUV coefficients resolved from the destination colorimetry.
        let c = YuyvEncodeCoeffs::from_params(cp);
        let process_rgb_to_yuyv = |s: &[u8; 6], d: &mut [u8; 4]| {
            let [r0, g0, b0, r1, g1, b1] = *s;
            *d = c.encode_pair(
                [r0 as i32, g0 as i32, b0 as i32],
                [r1 as i32, g1 as i32, b1 as i32],
            );
        };

        Self::for_each_row(src, dst, "rgb→yuyv", |src, dst| {
            let (src, dst, tail) = split_yuyv_row(src, dst, 3);
            let src = src.as_chunks::<{ 6 * 32 }>();
            let dst = dst.as_chunks_mut::<{ 4 * 32 }>();
            for (s, d) in src.0.iter().zip(dst.0.iter_mut()) {
                let s = s.as_chunks::<6>().0;
                let d = d.as_chunks_mut::<4>().0;
                for (s, d) in s.iter().zip(d.iter_mut()) {
                    process_rgb_to_yuyv(s, d);
                }
            }

            let s = src.1.as_chunks::<6>().0;
            let d = dst.1.as_chunks_mut::<4>().0;
            for (s, d) in s.iter().zip(d.iter_mut()) {
                process_rgb_to_yuyv(s, d);
            }

            if let Some((s, d)) = tail {
                let mut pair = [0u8; 4];
                process_rgb_to_yuyv(&[s[0], s[1], s[2], s[0], s[1], s[2]], &mut pair);
                d[0] = pair[0];
                d[1] = pair[1];
            }
        })
    }

    pub(super) fn convert_rgb_to_nv16(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let dst_w = dst.width().unwrap();
        let dst_h = if dst.is_multiplane() {
            dst.shape()[0]
        } else {
            dst.shape()[0] / 2
        };
        let src_rs = super::tensor_row_stride(src);
        let dst_stride = super::tensor_row_stride(dst);
        let mut dst_map = dst.map_mut()?;

        // Split at the stride-aligned luma plane boundary, not the tight one,
        // validating the destination holds the full combined plane first.
        let (y_plane, uv_plane) = super::split_semi_planar_mut(
            dst_map.as_mut_slice(),
            dst_stride,
            dst_h,
            edgefirst_tensor::PixelFormat::Nv16,
        )?;
        let mut bi_planar_image = yuv::YuvBiPlanarImageMut::<u8> {
            y_plane: yuv::BufferStoreMut::Borrowed(y_plane),
            y_stride: dst_stride as u32,
            uv_plane: yuv::BufferStoreMut::Borrowed(uv_plane),
            uv_stride: dst_stride as u32,
            width: dst_w as u32,
            height: dst_h as u32,
        };

        Ok(yuv::rgb_to_yuv_nv16(
            &mut bi_planar_image,
            src.map_read()?.as_slice(),
            src_rs as u32,
            cp.range,
            cp.matrix,
            yuv_mode(),
        )?)
    }

    /// Run `f(src_row, dst_row)` over the logical rows of a source/destination
    /// pair, each row clipped to its own pixel bytes.
    ///
    /// The row-confined form of "map both tensors and walk the slices": that
    /// flat walk is only correct when both sides are tightly packed, and it
    /// silently mis-places every row of a padded destination — or, for a
    /// `Tensor::view()` destination, packs the whole output into the head of the
    /// parent buffer and overwrites pixels beside the view.
    fn for_each_row(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        what: &str,
        mut f: impl FnMut(&[u8], &mut [u8]),
    ) -> Result<()> {
        let (src_rows, src_row_bytes) = super::logical_surface(src)?;
        let (dst_rows, dst_row_bytes) = super::logical_surface(dst)?;
        if src_rows != dst_rows {
            return Err(Error::InvalidShape(format!(
                "{what} row-count mismatch: {src_rows} source rows vs {dst_rows} destination rows"
            )));
        }
        let src_stride = super::tensor_row_stride(src);
        let dst_stride = super::tensor_row_stride(dst);
        let src_map = src.map_read()?;
        let mut dst_map = dst.map_mut()?;
        let (s, d) = (src_map.as_slice(), dst_map.as_mut_slice());
        super::guard_plane(s.len(), src_stride, src_rows, src_row_bytes, what)?;
        super::guard_plane(d.len(), dst_stride, dst_rows, dst_row_bytes, what)?;
        for (s, d) in super::packed_row_pairs(
            s,
            src_stride,
            src_row_bytes,
            d,
            dst_stride,
            dst_row_bytes,
            dst_rows,
        ) {
            f(s, d);
        }
        Ok(())
    }

    /// Row-wise copy between two same-format images. Each side is walked at its
    /// own row pitch, so a padded — or `view()`-derived — destination lands its
    /// rows at the parent pitch instead of packing them into the buffer head.
    pub(super) fn copy_image(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let (src_rows, src_row_bytes) = super::logical_surface(src)?;
        let (dst_rows, dst_row_bytes) = super::logical_surface(dst)?;
        if (src_rows, src_row_bytes) != (dst_rows, dst_row_bytes) {
            return Err(Error::InvalidShape(format!(
                "copy_image source/destination geometry mismatch: \
                 {src_rows}x{src_row_bytes} vs {dst_rows}x{dst_row_bytes} bytes"
            )));
        }
        Self::for_each_row(src, dst, "copy_image", |s, d| d.copy_from_slice(s))
    }

    /// Swap R and B channels in-place for an interleaved 4-channel image.
    ///
    /// Confined to each row's logical bytes: past them lies stride padding or,
    /// for a `view()` destination, the parent image's neighbouring pixels, which
    /// a whole-buffer swizzle would silently recolour.
    pub(super) fn swizzle_rb_4chan(dst: &mut Tensor<u8>) -> Result<()> {
        let (rows, row_bytes) = super::logical_surface(dst)?;
        let stride = super::tensor_row_stride(dst);
        let mut map = dst.map_mut()?;
        let buf = map.as_mut_slice();
        super::guard_plane(buf.len(), stride, rows, row_bytes, "swizzle dst")?;
        for row in buf.chunks_mut(stride).take(rows) {
            for chunk in row[..row_bytes].chunks_exact_mut(4) {
                chunk.swap(0, 2);
            }
        }
        Ok(())
    }

    /// Resolve an NV16 (4:2:2) source's planes/strides and decode. The UV plane
    /// is full-height with one `(Cb,Cr)` pair per two luma columns ⇒ `width`
    /// bytes per chroma row, i.e. the SAME pitch as luma; both planes use the
    /// buffer's (possibly even-padded) row stride (the logical width would
    /// corrupt every row past the first for an odd width where stride > width).
    fn convert_nv16<F>(src: &Tensor<u8>, dst: &mut Tensor<u8>, decode: F) -> Result<()>
    where
        F: FnOnce(
            &yuv::YuvBiPlanarImage<u8>,
            &mut [u8],
            u32,
        ) -> std::result::Result<(), yuv::YuvError>,
    {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let stride = src
            .effective_row_stride()
            .unwrap_or(src_w.next_multiple_of(2));
        if src.is_multiplane() {
            let y_map = src.map_read()?;
            let uv_map = src.chroma().unwrap().map_read()?;
            Self::semi_planar_decode(
                y_map.as_slice(),
                uv_map.as_slice(),
                src_w,
                src_h,
                stride,
                stride,
                dst,
                decode,
            )
        } else {
            let map = src.map_read()?;
            let (y_plane, uv_plane) = super::split_semi_planar(
                map.as_slice(),
                stride,
                src_h,
                src.format().expect("semi-planar source has a pixel format"),
            )?;
            Self::semi_planar_decode(y_plane, uv_plane, src_w, src_h, stride, stride, dst, decode)
        }
    }

    pub(super) fn convert_nv16_to_rgb(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        Self::convert_nv16(src, dst, |img, out, stride| {
            yuv::yuv_nv16_to_rgb(img, out, stride, cp.range, cp.matrix, yuv_mode())
        })
    }

    pub(super) fn convert_nv16_to_rgba(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        Self::convert_nv16(src, dst, |img, out, stride| {
            yuv::yuv_nv16_to_rgba(img, out, stride, cp.range, cp.matrix, yuv_mode())
        })
    }

    /// Resolve an NV24 (4:4:4 semi-planar) source's planes/strides and decode.
    /// The contiguous layout is `[3H, W]`: the Y plane (H rows) then the
    /// full-resolution interleaved UV plane (2H rows of W ⇒ `2*W` bytes per
    /// chroma row), so the UV stride is twice the luma stride. Handles
    /// true-multiplane (separate Y / CbCr buffers) as well as the contiguous
    /// buffer so NV24 is not silently mis-sliced when chroma is its own tensor.
    fn convert_nv24<F>(src: &Tensor<u8>, dst: &mut Tensor<u8>, decode: F) -> Result<()>
    where
        F: FnOnce(
            &yuv::YuvBiPlanarImage<u8>,
            &mut [u8],
            u32,
        ) -> std::result::Result<(), yuv::YuvError>,
    {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let stride = src
            .effective_row_stride()
            .unwrap_or(src_w.next_multiple_of(2));
        let uv_stride = stride * 2;
        if src.is_multiplane() {
            let y_map = src.map_read()?;
            let uv_map = src.chroma().unwrap().map_read()?;
            Self::semi_planar_decode(
                y_map.as_slice(),
                uv_map.as_slice(),
                src_w,
                src_h,
                stride,
                uv_stride,
                dst,
                decode,
            )
        } else {
            let map = src.map_read()?;
            let (y_plane, uv_plane) = super::split_semi_planar(
                map.as_slice(),
                stride,
                src_h,
                src.format().expect("semi-planar source has a pixel format"),
            )?;
            Self::semi_planar_decode(
                y_plane, uv_plane, src_w, src_h, stride, uv_stride, dst, decode,
            )
        }
    }

    pub(super) fn convert_nv24_to_rgb(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        Self::convert_nv24(src, dst, |img, out, stride| {
            yuv::yuv_nv24_to_rgb(img, out, stride, cp.range, cp.matrix, yuv_mode())
        })
    }

    pub(super) fn convert_nv24_to_rgba(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        Self::convert_nv24(src, dst, |img, out, stride| {
            yuv::yuv_nv24_to_rgba(img, out, stride, cp.range, cp.matrix, yuv_mode())
        })
    }

    /// NV24 → GREY: drop chroma, copy the luma plane honouring its row stride.
    pub(super) fn convert_nv24_to_grey(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src_w = src.width().unwrap();
        // The luma plane height: for a true-multiplane NV24 the (luma) tensor's
        // shape[0] is already the logical height; for the contiguous combined
        // buffer the shape is [3H, W] so divide by three. Computing this before
        // the multiplane check (as the previous code did) truncated multiplane
        // output to one third of its rows.
        let src_h = if src.is_multiplane() {
            src.shape()[0]
        } else {
            src.shape()[0] / 3
        };
        let src_stride = super::tensor_row_stride(src);
        let dst_stride = super::tensor_row_stride(dst);

        // Full-range luma is copied directly; limited-range luma is expanded.
        let luma = luma_mapper(cp.src_full_range);

        let src_map = src.map_read()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map_mut()?;
        let dst_bytes = dst_map.as_mut_slice();
        super::guard_plane(src_bytes.len(), src_stride, src_h, src_w, "nv24→grey src")?;
        super::guard_plane(dst_bytes.len(), dst_stride, src_h, src_w, "nv24→grey dst")?;
        for row in 0..src_h {
            let s = &src_bytes[row * src_stride..][..src_w];
            let d = &mut dst_bytes[row * dst_stride..][..src_w];
            for (s, d) in s.iter().zip(d) {
                *d = luma(*s);
            }
        }
        Ok(())
    }

    /// Copy the sub-rectangle `region` out of a semi-planar (NV12/NV16/NV24)
    /// source into `dst`, a `region.width × region.height` tensor of the *same*
    /// pixel format.
    ///
    /// This is the extraction step of the crop-sized pre-resize intermediate:
    /// `Tensor::view` only supports packed layouts, so a sub-rectangle of an
    /// NV source cannot be expressed as a strided view — its two planes sit at
    /// different offsets and subsample independently. Copying `region`'s luma
    /// and chroma rows into a small NV tensor lets the *unmodified* format
    /// converters decode just the crop.
    ///
    /// The copy is byte-exact by construction as long as `region`'s origin sits
    /// on a chroma sample boundary — even `left` for NV12/NV16, even `top` for
    /// NV12 — because a luma pixel's chroma sample is then found at the same
    /// relative index in the extracted plane as in the frame. The caller
    /// (`pre_resize_region`) guarantees that alignment; this function
    /// re-validates it rather than trusting it, and validates `region` against
    /// both buffers so a malformed tensor yields `InvalidShape`, not a panic.
    pub(super) fn extract_nv_region(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        fmt: edgefirst_tensor::PixelFormat,
        region: Rect,
    ) -> Result<()> {
        use edgefirst_tensor::PixelFormat::{Nv12, Nv16, Nv24};

        let src_w = src.width().unwrap_or(0);
        let src_h = src.height().unwrap_or(0);
        let (w, h) = (region.width, region.height);
        if region.left + w > src_w || region.top + h > src_h {
            return Err(Error::InvalidShape(format!(
                "nv region extract out of bounds: {region:?} (source {src_w}x{src_h})"
            )));
        }

        // Plane geometry, mirroring `convert_nv12`/`convert_nv16`/`convert_nv24`.
        // `chroma_div` maps a luma row to its chroma row; `chroma_x` converts a
        // luma-column offset into a chroma byte offset.
        let y_stride = src
            .effective_row_stride()
            .unwrap_or(src_w.next_multiple_of(2));
        let (uv_stride, chroma_div, chroma_x, chroma_row_bytes) = match fmt {
            Nv12 => {
                let uv = if src.is_multiplane() {
                    src.chroma()
                        .unwrap()
                        .effective_row_stride()
                        .unwrap_or(src_w.next_multiple_of(2))
                } else {
                    y_stride
                };
                (uv, 2usize, region.left, w.div_ceil(2) * 2)
            }
            Nv16 => (y_stride, 1usize, region.left, w.div_ceil(2) * 2),
            Nv24 => (y_stride * 2, 1usize, region.left * 2, w * 2),
            other => {
                return Err(Error::NotSupported(format!(
                    "nv region extract from {other}"
                )));
            }
        };
        // Chroma-boundary alignment (see the doc comment): without it the
        // extracted chroma plane would be offset by half a sample against the
        // luma and the decode would not match the frame's.
        let misaligned = match fmt {
            Nv12 => !region.left.is_multiple_of(2) || !region.top.is_multiple_of(2),
            Nv16 => !region.left.is_multiple_of(2),
            _ => false,
        };
        if misaligned {
            return Err(Error::InvalidShape(format!(
                "nv region extract needs a chroma-aligned origin for {fmt}: {region:?}"
            )));
        }

        let src_map = src.map_read()?;
        let chroma_map = if src.is_multiplane() {
            Some(src.chroma().unwrap().map_read()?)
        } else {
            None
        };
        let (src_y, src_uv): (&[u8], &[u8]) = if let Some(cm) = &chroma_map {
            (src_map.as_slice(), cm.as_slice())
        } else {
            super::split_semi_planar(src_map.as_slice(), y_stride, src_h, fmt)?
        };

        if dst.is_multiplane() {
            return Err(Error::InvalidShape(
                "nv region extract destination must be a contiguous single-plane tensor".into(),
            ));
        }
        let dst_stride = super::tensor_row_stride(dst);
        let dst_uv_stride = if fmt == Nv24 {
            dst_stride * 2
        } else {
            dst_stride
        };
        if dst_stride < w || dst_uv_stride < chroma_row_bytes {
            return Err(Error::InvalidShape(format!(
                "nv region extract destination stride {dst_stride} too small for width {w}"
            )));
        }
        let mut dst_map = dst.map_mut()?;
        let (dst_y, dst_uv) =
            super::split_semi_planar_mut(dst_map.as_mut_slice(), dst_stride, h, fmt)?;

        let chroma_rows = h.div_ceil(chroma_div);
        super::guard_plane(
            src_y.len(),
            y_stride,
            region.top + h,
            region.left + w,
            "nv extract src luma",
        )?;
        super::guard_plane(
            src_uv.len(),
            uv_stride,
            region.top / chroma_div + chroma_rows,
            chroma_x + chroma_row_bytes,
            "nv extract src chroma",
        )?;
        super::guard_plane(dst_y.len(), dst_stride, h, w, "nv extract dst luma")?;
        super::guard_plane(
            dst_uv.len(),
            dst_uv_stride,
            chroma_rows,
            chroma_row_bytes,
            "nv extract dst chroma",
        )?;

        for i in 0..h {
            let s = (region.top + i) * y_stride + region.left;
            dst_y[i * dst_stride..i * dst_stride + w].copy_from_slice(&src_y[s..s + w]);
        }
        for j in 0..chroma_rows {
            let s = (region.top / chroma_div + j) * uv_stride + chroma_x;
            dst_uv[j * dst_uv_stride..j * dst_uv_stride + chroma_row_bytes]
                .copy_from_slice(&src_uv[s..s + chroma_row_bytes]);
        }
        Ok(())
    }

    /// Strip-fused NV12/NV16/NV24 → PlanarRgb/PlanarRgba for the no-resize
    /// case. Decodes the YUV source into packed RGB one cache-resident row
    /// strip at a time (into the reused [`Self::nv_strip_scratch`]) and
    /// NEON-deinterleaves each strip straight into the destination planes, so
    /// the full-size packed-RGB intermediate never round-trips through DRAM and
    /// is not reallocated per frame. The strip height keeps a `width × 3`
    /// strip resident in L2 between the YUV decode and the deinterleave.
    ///
    /// `region` (source pixels) selects a sub-rectangle to decode: `None`
    /// decodes the whole source (the original whole-frame hot path). `Some(r)`
    /// decodes only `r`, writing the dense `r.width × r.height` result — the
    /// caller (the gate in `cpu/mod.rs`) guarantees `r` is scale-identity with
    /// the destination and chroma-aligned for `src_fmt`; this function itself
    /// only validates `r` against the source bounds.
    ///
    /// Geometry mirrors `convert_nv12`/`convert_nv16`/`convert_nv24` (contiguous
    /// and multiplane sources). NV12 (4:2:0) advances the chroma plane by half
    /// the luma rows; the strip height is even so each strip starts on an even
    /// luma row (relative to `region.top`, which the gate guarantees is even
    /// for NV12). The destination is validated against the derived plane sizes
    /// (untrusted dims → `InvalidShape`, not a panic), like the other helpers.
    pub(super) fn convert_nv_to_planar_fused(
        &mut self,
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        src_fmt: edgefirst_tensor::PixelFormat,
        dst_fmt: edgefirst_tensor::PixelFormat,
        cp: ColorParams,
        region: Option<Rect>,
    ) -> Result<()> {
        use edgefirst_tensor::PixelFormat::{Nv12, Nv16, Nv24, PlanarRgb, PlanarRgba};

        /// Strip height (rows). Even (NV12 4:2:0 needs an even strip start) and
        /// sized so one packed-RGB strip stays in L2 across realistic widths
        /// (32 × 1920 × 3 ≈ 180 KiB).
        const STRIP_ROWS: usize = 32;

        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let has_alpha = dst_fmt == PlanarRgba;
        debug_assert!(matches!(dst_fmt, PlanarRgb | PlanarRgba));

        // `region` in source pixels; `None` is the whole-frame case (left=0,
        // top=0, out dims == source dims), preserving the original behaviour
        // exactly.
        let (region_left, region_top, out_w, out_h) = match region {
            Some(r) => (r.left, r.top, r.width, r.height),
            None => (0, 0, src_w, src_h),
        };
        if region_left.checked_add(out_w).is_none_or(|e| e > src_w)
            || region_top.checked_add(out_h).is_none_or(|e| e > src_h)
        {
            return Err(Error::InvalidShape(format!(
                "fused nv→planar region out of bounds: {region:?} (source {src_w}x{src_h})"
            )));
        }

        // ---- source plane geometry (mirrors convert_nv12/nv16/nv24) ----
        let y_stride = src
            .effective_row_stride()
            .unwrap_or(src_w.next_multiple_of(2));
        // `chroma_div` maps a luma row index to its chroma row index: NV12
        // (4:2:0) subsamples chroma vertically by two; NV16/NV24 do not.
        // `chroma_x` converts a luma-column offset to the matching chroma
        // byte offset: NV12/NV16 pack one U+V byte pair per two luma columns
        // (1 byte/column); NV24 carries a full-resolution U+V pair per luma
        // column (2 bytes/column).
        let (uv_stride, chroma_div, chroma_x) = match src_fmt {
            Nv12 => {
                let uv = if src.is_multiplane() {
                    src.chroma()
                        .unwrap()
                        .effective_row_stride()
                        .unwrap_or(src_w.next_multiple_of(2))
                } else {
                    y_stride
                };
                (uv, 2usize, region_left)
            }
            Nv16 => (y_stride, 1usize, region_left),
            Nv24 => (y_stride * 2, 1usize, region_left * 2),
            other => return Err(Error::NotSupported(format!("fused {other} → planar"))),
        };

        let src_map = src.map_read()?;
        let chroma_map = if src.is_multiplane() {
            Some(src.chroma().unwrap().map_read()?)
        } else {
            None
        };
        let (y_plane, uv_plane): (&[u8], &[u8]) = if let Some(cm) = &chroma_map {
            (src_map.as_slice(), cm.as_slice())
        } else {
            super::split_semi_planar(src_map.as_slice(), y_stride, src_h, src_fmt)?
        };

        // ---- destination plane geometry + validation ----
        let dst_stride = super::tensor_row_stride(dst);
        let n_planes = if has_alpha { 4 } else { 3 };
        let mut dst_map = dst.map_mut()?;
        let dst_bytes = dst_map.as_mut_slice();
        let plane = dst_stride.checked_mul(out_h).ok_or_else(|| {
            Error::InvalidShape(format!(
                "fused nv→planar plane overflow (stride={dst_stride}, h={out_h})"
            ))
        })?;
        let dst_need = plane.checked_mul(n_planes).ok_or_else(|| {
            Error::InvalidShape(format!(
                "fused nv→planar dst overflow (plane={plane}, planes={n_planes})"
            ))
        })?;
        if dst_stride < out_w || dst_bytes.len() < dst_need {
            return Err(Error::InvalidShape(format!(
                "fused nv→planar dst too small: {} bytes, need {dst_need} (stride={dst_stride} >= w={out_w}, planes={n_planes})",
                dst_bytes.len()
            )));
        }
        if out_w == 0 || out_h == 0 {
            return Ok(());
        }

        let mut planes = dst_bytes.chunks_mut(plane);
        let rp = planes.next().unwrap();
        let gp = planes.next().unwrap();
        let bp = planes.next().unwrap();
        if has_alpha {
            // NV sources carry no alpha; PlanarRgba gets a constant 255 plane.
            planes.next().unwrap().fill(255);
        }

        // ---- strip loop: decode into the cached scratch, deinterleave out ----
        let mut scratch = std::mem::take(&mut self.nv_strip_scratch);
        let need = STRIP_ROWS.saturating_mul(out_w).saturating_mul(3);
        if scratch.len() < need {
            scratch.resize(need, 0);
        }

        // Chroma bytes per row for `out_w` luma columns: NV12/NV16 pack one
        // U+V byte pair per two luma columns (1 byte/column on average), so
        // an odd `out_w` still needs the *whole* trailing byte pair for the
        // column pair it's the first half of — round up to the next even
        // count, not down. NV24 carries a full-resolution U+V pair per luma
        // column (2 bytes/column, always exact). The gate requires an even
        // `region_left`, so this rounding never reads past the row's own
        // stride: the source region is always validated to fit within
        // `src_w <= uv_stride` (both `uv_stride` and `region_left` are even
        // for a chroma-subsampled format, so an odd `out_w` — the only case
        // that adds the extra byte — keeps `region_left + out_w` odd and
        // therefore strictly less than the even `uv_stride`, leaving room
        // for the pad byte).
        let chroma_row_bytes = if src_fmt == Nv24 {
            out_w * 2
        } else {
            out_w.div_ceil(2) * 2
        };

        // See the `nv_strip_y_pack`/`nv_strip_uv_pack` field docs: a nonzero
        // column offset always needs packing. Row-aligned reads
        // (`region_left == 0` — left-edge crops and the whole frame) slice
        // at row boundaries with the parent stride, so every stride-sized
        // chunk the `yuv` crate walks is a real, fully-owned source row and
        // the read can stay zero-copy — EXCEPT the NV12 odd-height,
        // non-flush-bottom case: the crate's 4:2:0 odd-last-row handling
        // takes `chunks_exact(2*stride).remainder()` / `uv.chunks_exact(
        // stride).last()`, which are only the region's own last rows when
        // the slice holds EXACTLY `height` rows. A row-aligned slice runs to
        // the plane's end, which is exact only when the region is flush with
        // the source's bottom edge; otherwise the remainder is empty or the
        // wrong row entirely (caught by
        // `left_edge_crop_zero_copy_arm_is_fused_and_correct`). Even heights
        // never reach that handling (the paired loop is bounded by the
        // destination zip), and NV16/NV24 have no vertical subsampling and
        // therefore no remainder path.
        let flush_bottom = region_top + out_h == src_h;
        let needs_pack =
            region_left != 0 || (src_fmt == Nv12 && !out_h.is_multiple_of(2) && !flush_bottom);
        let mut y_pack = std::mem::take(&mut self.nv_strip_y_pack);
        let mut uv_pack = std::mem::take(&mut self.nv_strip_uv_pack);
        if needs_pack {
            let y_need = STRIP_ROWS.saturating_mul(out_w);
            if y_pack.len() < y_need {
                y_pack.resize(y_need, 0);
            }
            let uv_need = STRIP_ROWS
                .div_ceil(chroma_div)
                .saturating_mul(chroma_row_bytes);
            if uv_pack.len() < uv_need {
                uv_pack.resize(uv_need, 0);
            }
        }

        let mut r0 = 0usize;
        let mut result = Ok(());
        while r0 < out_h {
            let sh = STRIP_ROWS.min(out_h - r0);
            let src_row = region_top + r0;

            // `img_y`/`img_uv` always start at column 0 of a real source row,
            // so the crate's internal `stride`-sized chunking never reads
            // past a row it doesn't own: a column-shifted region is packed to
            // `stride == width` first (see the `nv_strip_y_pack` field doc),
            // while a row-aligned region (`region_left == 0`) slices the
            // parent planes directly at its first row and stays zero-copy.
            let (img_y, img_y_stride, img_uv, img_uv_stride): (&[u8], u32, &[u8], u32) =
                if needs_pack {
                    let chroma_rows = sh.div_ceil(chroma_div);
                    for i in 0..sh {
                        let s_off = (src_row + i) * y_stride + region_left;
                        y_pack[i * out_w..i * out_w + out_w]
                            .copy_from_slice(&y_plane[s_off..s_off + out_w]);
                    }
                    for j in 0..chroma_rows {
                        let chroma_row = src_row / chroma_div + j;
                        let s_off = chroma_row * uv_stride + chroma_x;
                        uv_pack[j * chroma_row_bytes..j * chroma_row_bytes + chroma_row_bytes]
                            .copy_from_slice(&uv_plane[s_off..s_off + chroma_row_bytes]);
                    }
                    (
                        &y_pack[..sh * out_w],
                        out_w as u32,
                        &uv_pack[..chroma_rows * chroma_row_bytes],
                        chroma_row_bytes as u32,
                    )
                } else {
                    // `region_left`/`chroma_x` are always 0 here (row-aligned
                    // region or whole frame), so this is exactly the original
                    // zero-copy read, offset to the region's first row.
                    let yoff = src_row * y_stride + region_left;
                    let uvoff = (src_row / chroma_div) * uv_stride + chroma_x;
                    (
                        &y_plane[yoff..],
                        y_stride as u32,
                        &uv_plane[uvoff..],
                        uv_stride as u32,
                    )
                };
            let img = yuv::YuvBiPlanarImage {
                y_plane: img_y,
                y_stride: img_y_stride,
                uv_plane: img_uv,
                uv_stride: img_uv_stride,
                width: out_w as u32,
                height: sh as u32,
            };
            let rgb_stride = (out_w * 3) as u32;
            {
                let rgb = &mut scratch[..sh * out_w * 3];
                let decode = match src_fmt {
                    Nv12 => {
                        yuv::yuv_nv12_to_rgb(&img, rgb, rgb_stride, cp.range, cp.matrix, yuv_mode())
                    }
                    Nv16 => {
                        yuv::yuv_nv16_to_rgb(&img, rgb, rgb_stride, cp.range, cp.matrix, yuv_mode())
                    }
                    Nv24 => {
                        yuv::yuv_nv24_to_rgb(&img, rgb, rgb_stride, cp.range, cp.matrix, yuv_mode())
                    }
                    _ => unreachable!(),
                };
                if let Err(e) = decode {
                    result = Err(e.into());
                    break;
                }
            }
            // The strip's packed RGB is now hot in the scratch; scatter each row
            // into the destination planes at its frame-row offset.
            for i in 0..sh {
                let s = &scratch[i * out_w * 3..i * out_w * 3 + out_w * 3];
                let roff = (r0 + i) * dst_stride;
                super::simd::deinterleave_row(
                    s,
                    &mut rp[roff..roff + out_w],
                    &mut gp[roff..roff + out_w],
                    &mut bp[roff..roff + out_w],
                    None,
                    out_w,
                    3,
                );
            }
            r0 += sh;
        }
        self.nv_strip_scratch = scratch;
        self.nv_strip_y_pack = y_pack;
        self.nv_strip_uv_pack = uv_pack;
        result
    }

    /// Read a planar `[C, H, W]` source into a packed interleaved destination,
    /// honouring both the source row stride and the destination row stride.
    /// Colour planes 0..3 map to destination channels R, G, B. When the
    /// destination has a fourth channel it is taken from source plane 3 if the
    /// source has one (`PlanarRgba`), otherwise filled with 255 (`PlanarRgb`).
    ///
    /// Plane offsets are derived from `height * row_stride` and rows are walked
    /// individually, so strided / pitch-aligned sources (DMA, `create_image`)
    /// are not mis-sliced or read across their per-row padding — a flat
    /// `mapped_len / channels` split reads pad bytes as pixels on padded buffers.
    fn planar_to_packed(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        src_planes: usize,
        dst_ch: usize,
    ) -> Result<()> {
        let w = src.width().unwrap();
        let h = src.height().unwrap();
        let src_stride = super::tensor_row_stride(src);
        let dst_stride = super::tensor_row_stride(dst);
        let has_alpha_plane = dst_ch == 4 && src_planes >= 4;
        // Planes actually read: R/G/B always, plus the alpha plane when the
        // destination has one and the source supplies it.
        let planes_read = if has_alpha_plane { 4 } else { 3 };

        let src_map = src.map_read()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map_mut()?;
        let dst_bytes = dst_map.as_mut_slice();

        // Validate the buffers against the derived geometry before indexing.
        // Like `split_semi_planar`, an imported tensor may carry a stride/shape
        // that exceeds its actual allocation (untrusted input), so use checked
        // arithmetic and return `InvalidShape` instead of panicking with an
        // out-of-bounds slice. `src_stride >= w`, so a plane spans at most
        // `h * src_stride` bytes and the last row's `w` bytes stay in-plane.
        let plane_stride = src_stride.checked_mul(h).ok_or_else(|| {
            Error::InvalidShape(format!(
                "planar plane size overflow (stride={src_stride}, h={h})"
            ))
        })?;
        let src_need = plane_stride.checked_mul(planes_read).ok_or_else(|| {
            Error::InvalidShape(format!(
                "planar source size overflow (plane_stride={plane_stride}, planes={planes_read})"
            ))
        })?;
        if src_bytes.len() < src_need {
            return Err(Error::InvalidShape(format!(
                "planar source has {} bytes but needs {src_need} (stride={src_stride}, h={h}, planes={planes_read})",
                src_bytes.len()
            )));
        }
        let dst_row = w.checked_mul(dst_ch).ok_or_else(|| {
            Error::InvalidShape(format!("packed dst row overflow (w={w}, ch={dst_ch})"))
        })?;
        let dst_need = dst_stride.checked_mul(h).ok_or_else(|| {
            Error::InvalidShape(format!(
                "packed dst size overflow (stride={dst_stride}, h={h})"
            ))
        })?;
        if dst_stride < dst_row || dst_bytes.len() < dst_need {
            return Err(Error::InvalidShape(format!(
                "packed dst has stride={dst_stride}, {} bytes but needs stride>={dst_row} and {dst_need} bytes (w={w}, h={h}, ch={dst_ch})",
                dst_bytes.len()
            )));
        }

        dst_bytes
            .par_chunks_mut(dst_stride)
            .take(h)
            .enumerate()
            .for_each(|(row, d)| {
                let off = row * src_stride;
                let r = &src_bytes[off..][..w];
                let g = &src_bytes[plane_stride + off..][..w];
                let b = &src_bytes[2 * plane_stride + off..][..w];
                for x in 0..w {
                    let p = &mut d[x * dst_ch..][..dst_ch];
                    p[0] = r[x];
                    p[1] = g[x];
                    p[2] = b[x];
                    if dst_ch == 4 {
                        p[3] = if has_alpha_plane {
                            src_bytes[3 * plane_stride + off + x]
                        } else {
                            255
                        };
                    }
                }
            });
        Ok(())
    }

    pub(super) fn convert_8bps_to_rgb(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        Self::planar_to_packed(src, dst, 3, 3)
    }

    pub(super) fn convert_8bps_to_rgba(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        Self::planar_to_packed(src, dst, 3, 4)
    }

    pub(super) fn convert_prgba_to_rgb(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        Self::planar_to_packed(src, dst, 4, 3)
    }

    pub(super) fn convert_prgba_to_rgba(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        Self::planar_to_packed(src, dst, 4, 4)
    }

    pub(super) fn rgba_to_rgb(rgba: [u8; 4]) -> [u8; 3] {
        let [r, g, b, _] = rgba;
        [r, g, b]
    }

    pub(super) fn rgba_to_grey(rgba: [u8; 4]) -> [u8; 1] {
        const BIAS: i32 = 20;
        // Conventional BT.601 luma weights (Rec.601 is the standard luma basis
        // for RGB→grayscale; full-range, no 16/235 expansion).
        const KR: f64 = 0.299f64;
        const KB: f64 = 0.114f64;
        const KG: f64 = 1.0 - KR - KB;
        const Y_R: i32 = (KR * (255 << BIAS) as f64 / 255.0).round() as i32;
        const Y_G: i32 = (KG * (255 << BIAS) as f64 / 255.0).round() as i32;
        const Y_B: i32 = (KB * (255 << BIAS) as f64 / 255.0).round() as i32;

        const ROUND: i32 = 1 << (BIAS - 1);

        let [r, g, b, _] = rgba;
        let y = ((Y_R * r as i32 + Y_G * g as i32 + Y_B * b as i32 + ROUND) >> BIAS) as u8;
        [y]
    }

    pub(super) fn rgba_to_yuyv(rgba: [u8; 4], cp: ColorParams) -> [u8; 4] {
        let [r, g, b, _] = rgba;
        YuyvEncodeCoeffs::from_params(cp).encode_single([r as i32, g as i32, b as i32])
    }
}
