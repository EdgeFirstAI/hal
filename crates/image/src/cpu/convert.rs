// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{Error, Result};
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
    let src_map = src.map()?;
    let src_bytes = src_map.as_slice();
    let mut dst_map = dst.map()?;
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
        Ok(decode(&src, dst.map()?.as_mut_slice(), dst_stride)?)
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
            let y_map = src.map()?;
            let uv_map = src.chroma().unwrap().map()?;
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
            let map = src.map()?;
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
            yuv::yuv_nv12_to_rgb(
                img,
                out,
                stride,
                cp.range,
                cp.matrix,
                yuv::YuvConversionMode::Balanced,
            )
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
            yuv::yuv_nv12_to_rgba(
                img,
                out,
                stride,
                cp.range,
                cp.matrix,
                yuv::YuvConversionMode::Balanced,
            )
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

        let src_map = src.map()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map()?;
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
            yuy: &src.map()?,
            yuy_stride: src_rs as u32,
            width: src_w as u32,
            height: src_h as u32,
        };

        let dst_w = dst.width().unwrap();
        Ok(yuv::yuyv422_to_rgb(
            &src,
            dst.map()?.as_mut_slice(),
            dst_w as u32 * 3,
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
            yuy: &src.map()?,
            yuy_stride: src_rs as u32,
            width: src_w as u32,
            height: src_h as u32,
        };

        Ok(yuv::yuyv422_to_rgba(
            &src,
            dst.map()?.as_mut_slice(),
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
        let src_map = src.map()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map()?;
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
        let src_map = src.map()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map()?;
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
            yuy: &src.map()?,
            yuy_stride: src_rs as u32,
            width: src_w as u32,
            height: src_h as u32,
        };

        let dst_w = dst.width().unwrap();
        Ok(yuv::vyuy422_to_rgb(
            &src,
            dst.map()?.as_mut_slice(),
            dst_w as u32 * 3,
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
            yuy: &src.map()?,
            yuy_stride: src_rs as u32,
            width: src_w as u32,
            height: src_h as u32,
        };

        Ok(yuv::vyuy422_to_rgba(
            &src,
            dst.map()?.as_mut_slice(),
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
        let src_map = src.map()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map()?;
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
        let src_map = src.map()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map()?;
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
            y_plane: &src.map()?,
            y_stride: src_rs as u32,
            width: src_w as u32,
            height: src_h as u32,
        };
        Ok(yuv::yuv400_to_rgb(
            &src,
            dst.map()?.as_mut_slice(),
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
            y_plane: &src.map()?,
            y_stride: src_rs as u32,
            width: src_w as u32,
            height: src_h as u32,
        };
        Ok(yuv::yuv400_to_rgba(
            &src,
            dst.map()?.as_mut_slice(),
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
        let src = src.map()?;
        let src = src.as_slice();

        let mut dst = dst.map()?;
        let dst = dst.as_mut_slice();
        for (s, d) in src
            .as_chunks::<2>()
            .0
            .iter()
            .zip(dst.as_chunks_mut::<4>().0.iter_mut())
        {
            d[0] = y_enc(s[0]);
            d[1] = 128;

            d[2] = y_enc(s[1]);
            d[3] = 128;
        }
        Ok(())
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
        let src_map = src.map()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map()?;
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
        Ok(yuv::rgba_to_rgb(
            src.map()?.as_slice(),
            (src_w * 4) as u32,
            dst.map()?.as_mut_slice(),
            (dst.width().unwrap() * 3) as u32,
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
            y_plane: yuv::BufferStoreMut::Borrowed(&mut dst.map()?),
            y_stride: dst_rs as u32,
            width: dst_w as u32,
            height: dst_h as u32,
        };
        Ok(yuv::rgba_to_yuv400(
            &mut dst,
            src.map()?.as_slice(),
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
        let src = src.map()?;
        let src = src.as_slice();

        let mut dst = dst.map()?;
        let dst = dst.as_mut_slice();

        // RGB→YUV coefficients resolved from the destination colorimetry.
        let c = YuyvEncodeCoeffs::from_params(cp);
        let process_rgba_to_yuyv = |s: &[u8; 8], d: &mut [u8; 4]| {
            let [r0, g0, b0, _, r1, g1, b1, _] = *s;
            *d = c.encode_pair(
                [r0 as i32, g0 as i32, b0 as i32],
                [r1 as i32, g1 as i32, b1 as i32],
            );
        };

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

        Ok(())
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
        let mut dst_map = dst.map()?;

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
            src.map()?.as_slice(),
            src_rs as u32,
            cp.range,
            cp.matrix,
            yuv::YuvConversionMode::Balanced,
        )?)
    }

    pub(super) fn convert_rgb_to_rgba(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        Ok(yuv::rgb_to_rgba(
            src.map()?.as_slice(),
            (src_w * 3) as u32,
            dst.map()?.as_mut_slice(),
            (dst.width().unwrap() * 4) as u32,
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
            y_plane: yuv::BufferStoreMut::Borrowed(&mut dst.map()?),
            y_stride: dst_rs as u32,
            width: dst_w as u32,
            height: dst_h as u32,
        };
        Ok(yuv::rgb_to_yuv400(
            &mut dst,
            src.map()?.as_slice(),
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
        let src = src.map()?;
        let src = src.as_slice();

        let mut dst = dst.map()?;
        let dst = dst.as_mut_slice();

        // RGB→YUV coefficients resolved from the destination colorimetry.
        let c = YuyvEncodeCoeffs::from_params(cp);
        let process_rgb_to_yuyv = |s: &[u8; 6], d: &mut [u8; 4]| {
            let [r0, g0, b0, r1, g1, b1] = *s;
            *d = c.encode_pair(
                [r0 as i32, g0 as i32, b0 as i32],
                [r1 as i32, g1 as i32, b1 as i32],
            );
        };

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

        Ok(())
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
        let mut dst_map = dst.map()?;

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
            src.map()?.as_slice(),
            src_rs as u32,
            cp.range,
            cp.matrix,
            yuv::YuvConversionMode::Balanced,
        )?)
    }

    pub(super) fn copy_image(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src_map = src.map()?;
        let mut dst_map = dst.map()?;
        let (s, d) = (src_map.as_slice(), dst_map.as_mut_slice());
        // Guard the length before `copy_from_slice` (which panics on mismatch),
        // for parity with `prepare_dst_base_cpu`.
        if s.len() != d.len() {
            return Err(Error::InvalidShape(format!(
                "copy_image source/destination size mismatch: {} vs {} bytes",
                s.len(),
                d.len()
            )));
        }
        d.copy_from_slice(s);
        Ok(())
    }

    /// Swap R and B channels in-place for an interleaved 4-channel image.
    pub(super) fn swizzle_rb_4chan(dst: &mut Tensor<u8>) -> Result<()> {
        let mut map = dst.map()?;
        let buf = map.as_mut_slice();
        for chunk in buf.chunks_exact_mut(4) {
            chunk.swap(0, 2);
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
            let y_map = src.map()?;
            let uv_map = src.chroma().unwrap().map()?;
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
            let map = src.map()?;
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
            yuv::yuv_nv16_to_rgb(
                img,
                out,
                stride,
                cp.range,
                cp.matrix,
                yuv::YuvConversionMode::Balanced,
            )
        })
    }

    pub(super) fn convert_nv16_to_rgba(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        Self::convert_nv16(src, dst, |img, out, stride| {
            yuv::yuv_nv16_to_rgba(
                img,
                out,
                stride,
                cp.range,
                cp.matrix,
                yuv::YuvConversionMode::Balanced,
            )
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
            let y_map = src.map()?;
            let uv_map = src.chroma().unwrap().map()?;
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
            let map = src.map()?;
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
            yuv::yuv_nv24_to_rgb(
                img,
                out,
                stride,
                cp.range,
                cp.matrix,
                yuv::YuvConversionMode::Balanced,
            )
        })
    }

    pub(super) fn convert_nv24_to_rgba(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        Self::convert_nv24(src, dst, |img, out, stride| {
            yuv::yuv_nv24_to_rgba(
                img,
                out,
                stride,
                cp.range,
                cp.matrix,
                yuv::YuvConversionMode::Balanced,
            )
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

        let src_map = src.map()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map()?;
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

        let src_map = src.map()?;
        let src_bytes = src_map.as_slice();
        let mut dst_map = dst.map()?;
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
        // BT.601 luma coefficients (interim 601-full colorimetry stop-gap).
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
