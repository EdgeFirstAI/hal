// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::Result;
use edgefirst_tensor::{Tensor, TensorMapTrait, TensorTrait};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
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
/// resolved from the destination tensor's matrix + range. All terms are
/// `Q(BIAS)` fixed-point; `y_off`/`c_off` are the post-shift integer offsets.
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
    /// `BIAS` matches the original hand-rolled tables (Q20 fixed point).
    const BIAS: i32 = 20;
    const ROUND: i32 = 1 << (Self::BIAS - 1);
    const ROUND2: i32 = 1 << Self::BIAS;

    /// Build the table from the resolved `ColorParams`. The luma/chroma swings
    /// are full-range (255/255) or limited-range (219/224) per `cp.range`; the
    /// `KR`/`KB` luma weights come from `cp.matrix` (BT.601 / 709 / 2020).
    fn from_params(cp: ColorParams) -> Self {
        // KR/KB luma weights per standard matrix.
        let (kr, kb) = match cp.matrix {
            yuv::YuvStandardMatrix::Bt601 => (0.299_f64, 0.114_f64),
            yuv::YuvStandardMatrix::Bt2020 => (0.2627_f64, 0.0593_f64),
            // BT.709 and any future/unknown matrix default to 709 weights.
            _ => (0.2126_f64, 0.0722_f64),
        };
        let kg = 1.0 - kr - kb;
        let full = matches!(cp.range, yuv::YuvRange::Full);
        // Luma swing / chroma swing and offsets for the selected range.
        let (y_swing, c_swing, y_off, c_off) = if full {
            (255.0_f64, 255.0_f64, 0, 128)
        } else {
            (219.0_f64, 224.0_f64, 16, 128)
        };
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

impl CPUProcessor {
    pub(super) fn convert_nv12_to_rgb(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src_w = src.width().unwrap();
        if src.is_multiplane() {
            let y_map = src.map()?;
            let uv_map = src.chroma().unwrap().map()?;
            let src_h = src.shape()[0]; // multiplane: luma shape is [H, W]
            Self::nv12_to_rgb_kernel(y_map.as_slice(), uv_map.as_slice(), src_w, src_h, dst, cp)
        } else {
            let map = src.map()?;
            // contiguous NV12: shape is [H*3/2, W], so height = shape[0] * 2 / 3
            let src_h = src.shape()[0] * 2 / 3;
            let (y_plane, uv_plane) = map.as_slice().split_at(src_w * src_h);
            Self::nv12_to_rgb_kernel(y_plane, uv_plane, src_w, src_h, dst, cp)
        }
    }

    fn nv12_to_rgb_kernel(
        y_plane: &[u8],
        uv_plane: &[u8],
        width: usize,
        height: usize,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src = yuv::YuvBiPlanarImage {
            y_plane,
            y_stride: width as u32,
            uv_plane,
            uv_stride: width as u32,
            width: width as u32,
            height: height as u32,
        };

        Ok(yuv::yuv_nv12_to_rgb(
            &src,
            dst.map()?.as_mut_slice(),
            super::tensor_row_stride(dst) as u32,
            cp.range,
            cp.matrix,
            yuv::YuvConversionMode::Balanced,
        )?)
    }

    // NOTE: The `*_to_rgba` helpers below all accept BGRA destinations.
    // They always write pixels in RGBA channel order; for BGRA destinations the
    // caller applies an R<->B swizzle afterwards via `swizzle_rb_4chan`.
    pub(super) fn convert_nv12_to_rgba(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src_w = src.width().unwrap();
        if src.is_multiplane() {
            let y_map = src.map()?;
            let uv_map = src.chroma().unwrap().map()?;
            let src_h = src.shape()[0];
            Self::nv12_to_rgba_kernel(y_map.as_slice(), uv_map.as_slice(), src_w, src_h, dst, cp)
        } else {
            let map = src.map()?;
            let src_h = src.shape()[0] * 2 / 3;
            let (y_plane, uv_plane) = map.as_slice().split_at(src_w * src_h);
            Self::nv12_to_rgba_kernel(y_plane, uv_plane, src_w, src_h, dst, cp)
        }
    }

    fn nv12_to_rgba_kernel(
        y_plane: &[u8],
        uv_plane: &[u8],
        width: usize,
        height: usize,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src = yuv::YuvBiPlanarImage {
            y_plane,
            y_stride: width as u32,
            uv_plane,
            uv_stride: width as u32,
            width: width as u32,
            height: height as u32,
        };

        Ok(yuv::yuv_nv12_to_rgba(
            &src,
            dst.map()?.as_mut_slice(),
            super::tensor_row_stride(dst) as u32,
            cp.range,
            cp.matrix,
            yuv::YuvConversionMode::Balanced,
        )?)
    }

    pub(super) fn convert_nv12_to_grey(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src_w = src.width().unwrap();
        let src_h = if src.is_multiplane() {
            src.shape()[0]
        } else {
            src.shape()[0] * 2 / 3
        };

        let src_map = src.map()?;
        let y_len = src_w * src_h;
        let y_slice = &src_map.as_slice()[..y_len];

        // Full-range luma is copied directly; limited-range luma is expanded.
        let luma = luma_mapper(cp.src_full_range);

        let mut dst_map = dst.map()?;
        let src_chunks = y_slice.as_chunks::<8>();
        let dst_chunks = dst_map.as_chunks_mut::<8>();
        for (s, d) in src_chunks.0.iter().zip(dst_chunks.0) {
            s.iter().zip(d).for_each(|(s, d)| *d = luma(*s));
        }

        for (s, d) in src_chunks.1.iter().zip(dst_chunks.1) {
            *d = luma(*s);
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
        let luma = luma_mapper(cp.src_full_range);
        let src_map = src.map()?;
        let mut dst_map = dst.map()?;
        let src_chunks = src_map.as_chunks::<16>();
        let dst_chunks = dst_map.as_chunks_mut::<8>();
        for (s, d) in src_chunks.0.iter().zip(dst_chunks.0) {
            s.iter().step_by(2).zip(d).for_each(|(s, d)| *d = luma(*s));
        }

        for (s, d) in src_chunks.1.iter().step_by(2).zip(dst_chunks.1) {
            *d = luma(*s);
        }

        Ok(())
    }

    pub(super) fn convert_yuyv_to_nv16(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src_map = src.map()?;
        let mut dst_map = dst.map()?;

        let src_chunks = src_map.as_chunks::<2>().0;
        let dst_w = dst.width().unwrap();
        let dst_h = if dst.is_multiplane() {
            dst.shape()[0]
        } else {
            dst.shape()[0] / 2
        };
        let (y_plane, uv_plane) = dst_map.split_at_mut(dst_w * dst_h);

        for ((s, y), uv) in src_chunks.iter().zip(y_plane).zip(uv_plane) {
            *y = s[0];
            *uv = s[1];
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
        let luma = luma_mapper(cp.src_full_range);
        let src_map = src.map()?;
        let mut dst_map = dst.map()?;
        // VYUY byte order: [V, Y0, U, Y1] — Y at offsets 1, 3
        let src_chunks = src_map.as_chunks::<16>();
        let dst_chunks = dst_map.as_chunks_mut::<8>();
        for (s, d) in src_chunks.0.iter().zip(dst_chunks.0) {
            for (di, si) in (1..16).step_by(2).enumerate() {
                d[di] = luma(s[si]);
            }
        }

        for (di, si) in (1..src_chunks.1.len()).step_by(2).enumerate() {
            dst_chunks.1[di] = luma(src_chunks.1[si]);
        }

        Ok(())
    }

    pub(super) fn convert_vyuy_to_nv16(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src_map = src.map()?;
        let mut dst_map = dst.map()?;

        // VYUY byte order: [V, Y0, U, Y1] — per 4-byte macropixel
        let src_chunks = src_map.as_chunks::<4>().0;
        let dst_w = dst.width().unwrap();
        let dst_h = if dst.is_multiplane() {
            dst.shape()[0]
        } else {
            dst.shape()[0] / 2
        };
        let (y_plane, uv_plane) = dst_map.split_at_mut(dst_w * dst_h);
        let y_pairs = y_plane.as_chunks_mut::<2>().0;
        let uv_pairs = uv_plane.as_chunks_mut::<2>().0;

        for ((s, y), uv) in src_chunks.iter().zip(y_pairs).zip(uv_pairs) {
            y[0] = s[1]; // Y0
            y[1] = s[3]; // Y1
            uv[0] = s[2]; // U
            uv[1] = s[0]; // V
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
            yuv::YuvStandardMatrix::Bt709,
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
            yuv::YuvStandardMatrix::Bt709,
        )?)
    }

    pub(super) fn convert_grey_to_8bps(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src = src.map()?;
        let src = src.as_slice();

        let mut dst_map = dst.map()?;
        let dst_ = dst_map.as_mut_slice();

        let plane_size = src.len();
        let (dst0, dst1) = dst_.split_at_mut(plane_size);
        let (dst1, dst2) = dst1.split_at_mut(plane_size);

        rayon::scope(|s| {
            s.spawn(|_| dst0.copy_from_slice(src));
            s.spawn(|_| dst1.copy_from_slice(src));
            s.spawn(|_| dst2.copy_from_slice(src));
        });
        Ok(())
    }

    pub(super) fn convert_grey_to_prgba(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src = src.map()?;
        let src = src.as_slice();

        let mut dst_map = dst.map()?;
        let dst_ = dst_map.as_mut_slice();

        let plane_size = src.len();
        let (dst0, dst1) = dst_.split_at_mut(plane_size);
        let (dst1, dst2) = dst1.split_at_mut(plane_size);
        let (dst2, dst3) = dst2.split_at_mut(plane_size);
        rayon::scope(|s| {
            s.spawn(|_| dst0.copy_from_slice(src));
            s.spawn(|_| dst1.copy_from_slice(src));
            s.spawn(|_| dst2.copy_from_slice(src));
            s.spawn(|_| dst3.fill(255));
        });
        Ok(())
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
        let src = src.map()?;
        let src = src.as_slice();

        let mut dst = dst.map()?;
        let dst = dst.as_mut_slice();

        for (s, d) in src.iter().zip(dst[0..src.len()].iter_mut()) {
            *d = y_enc(*s);
        }
        dst[src.len()..].fill(128);

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
            yuv::YuvStandardMatrix::Bt709,
        )?)
    }

    pub(super) fn convert_rgba_to_8bps(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src = src.map()?;
        let src = src.as_slice();
        let src = src.as_chunks::<4>().0;

        let mut dst_map = dst.map()?;
        let dst_ = dst_map.as_mut_slice();

        let plane_size = src.len();
        let (dst0, dst1) = dst_.split_at_mut(plane_size);
        let (dst1, dst2) = dst1.split_at_mut(plane_size);

        src.par_iter()
            .zip_eq(dst0)
            .zip_eq(dst1)
            .zip_eq(dst2)
            .for_each(|(((s, d0), d1), d2)| {
                *d0 = s[0];
                *d1 = s[1];
                *d2 = s[2];
            });
        Ok(())
    }

    pub(super) fn convert_rgba_to_prgba(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src = src.map()?;
        let src = src.as_slice();
        let src = src.as_chunks::<4>().0;

        let mut dst_map = dst.map()?;
        let dst_ = dst_map.as_mut_slice();

        let plane_size = src.len();
        let (dst0, dst1) = dst_.split_at_mut(plane_size);
        let (dst1, dst2) = dst1.split_at_mut(plane_size);
        let (dst2, dst3) = dst2.split_at_mut(plane_size);

        src.par_iter()
            .zip_eq(dst0)
            .zip_eq(dst1)
            .zip_eq(dst2)
            .zip_eq(dst3)
            .for_each(|((((s, d0), d1), d2), d3)| {
                *d0 = s[0];
                *d1 = s[1];
                *d2 = s[2];
                *d3 = s[3];
            });
        Ok(())
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
        let mut dst_map = dst.map()?;

        let (y_plane, uv_plane) = dst_map.split_at_mut(dst_w * dst_h);
        let mut bi_planar_image = yuv::YuvBiPlanarImageMut::<u8> {
            y_plane: yuv::BufferStoreMut::Borrowed(y_plane),
            y_stride: dst_w as u32,
            uv_plane: yuv::BufferStoreMut::Borrowed(uv_plane),
            uv_stride: dst_w as u32,
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
            yuv::YuvStandardMatrix::Bt709,
        )?)
    }

    pub(super) fn convert_rgb_to_8bps(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src = src.map()?;
        let src = src.as_slice();
        let src = src.as_chunks::<3>().0;

        let mut dst_map = dst.map()?;
        let dst_ = dst_map.as_mut_slice();

        let plane_size = src.len();
        let (dst0, dst1) = dst_.split_at_mut(plane_size);
        let (dst1, dst2) = dst1.split_at_mut(plane_size);

        src.par_iter()
            .zip_eq(dst0)
            .zip_eq(dst1)
            .zip_eq(dst2)
            .for_each(|(((s, d0), d1), d2)| {
                *d0 = s[0];
                *d1 = s[1];
                *d2 = s[2];
            });
        Ok(())
    }

    pub(super) fn convert_rgb_to_prgba(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src = src.map()?;
        let src = src.as_slice();
        let src = src.as_chunks::<3>().0;

        let mut dst_map = dst.map()?;
        let dst_ = dst_map.as_mut_slice();

        let plane_size = src.len();
        let (dst0, dst1) = dst_.split_at_mut(plane_size);
        let (dst1, dst2) = dst1.split_at_mut(plane_size);
        let (dst2, dst3) = dst2.split_at_mut(plane_size);

        rayon::scope(|s| {
            s.spawn(|_| {
                src.par_iter()
                    .zip_eq(dst0)
                    .zip_eq(dst1)
                    .zip_eq(dst2)
                    .for_each(|(((s, d0), d1), d2)| {
                        *d0 = s[0];
                        *d1 = s[1];
                        *d2 = s[2];
                    })
            });
            s.spawn(|_| dst3.fill(255));
        });
        Ok(())
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
        let mut dst_map = dst.map()?;

        let (y_plane, uv_plane) = dst_map.split_at_mut(dst_w * dst_h);
        let mut bi_planar_image = yuv::YuvBiPlanarImageMut::<u8> {
            y_plane: yuv::BufferStoreMut::Borrowed(y_plane),
            y_stride: dst_w as u32,
            uv_plane: yuv::BufferStoreMut::Borrowed(uv_plane),
            uv_stride: dst_w as u32,
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
        dst.map()?.copy_from_slice(&src.map()?);
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

    pub(super) fn convert_nv16_to_rgb(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src_w = src.width().unwrap();
        if src.is_multiplane() {
            let y_map = src.map()?;
            let uv_map = src.chroma().unwrap().map()?;
            let src_h = src.shape()[0];
            Self::nv16_to_rgb_kernel(y_map.as_slice(), uv_map.as_slice(), src_w, src_h, dst, cp)
        } else {
            let map = src.map()?;
            let src_h = src.shape()[0] / 2;
            let (y_plane, uv_plane) = map.as_slice().split_at(src_w * src_h);
            Self::nv16_to_rgb_kernel(y_plane, uv_plane, src_w, src_h, dst, cp)
        }
    }

    fn nv16_to_rgb_kernel(
        y_plane: &[u8],
        uv_plane: &[u8],
        width: usize,
        height: usize,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src = yuv::YuvBiPlanarImage {
            y_plane,
            y_stride: width as u32,
            uv_plane,
            uv_stride: width as u32,
            width: width as u32,
            height: height as u32,
        };

        Ok(yuv::yuv_nv16_to_rgb(
            &src,
            dst.map()?.as_mut_slice(),
            super::tensor_row_stride(dst) as u32,
            cp.range,
            cp.matrix,
            yuv::YuvConversionMode::Balanced,
        )?)
    }

    pub(super) fn convert_nv16_to_rgba(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src_w = src.width().unwrap();
        if src.is_multiplane() {
            let y_map = src.map()?;
            let uv_map = src.chroma().unwrap().map()?;
            let src_h = src.shape()[0];
            Self::nv16_to_rgba_kernel(y_map.as_slice(), uv_map.as_slice(), src_w, src_h, dst, cp)
        } else {
            let map = src.map()?;
            let src_h = src.shape()[0] / 2;
            let (y_plane, uv_plane) = map.as_slice().split_at(src_w * src_h);
            Self::nv16_to_rgba_kernel(y_plane, uv_plane, src_w, src_h, dst, cp)
        }
    }

    fn nv16_to_rgba_kernel(
        y_plane: &[u8],
        uv_plane: &[u8],
        width: usize,
        height: usize,
        dst: &mut Tensor<u8>,
        cp: ColorParams,
    ) -> Result<()> {
        let src = yuv::YuvBiPlanarImage {
            y_plane,
            y_stride: width as u32,
            uv_plane,
            uv_stride: width as u32,
            width: width as u32,
            height: height as u32,
        };

        Ok(yuv::yuv_nv16_to_rgba(
            &src,
            dst.map()?.as_mut_slice(),
            super::tensor_row_stride(dst) as u32,
            cp.range,
            cp.matrix,
            yuv::YuvConversionMode::Balanced,
        )?)
    }

    pub(super) fn convert_8bps_to_rgb(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src_map = src.map()?;
        let src_ = src_map.as_slice();

        // Planar [C, H, W] — plane size = H*W = total/3
        let plane_size = src_.len() / 3;
        let (src0, src1) = src_.split_at(plane_size);
        let (src1, src2) = src1.split_at(plane_size);

        let mut dst_map = dst.map()?;
        let dst_ = dst_map.as_mut_slice();

        src0.par_iter()
            .zip_eq(src1)
            .zip_eq(src2)
            .zip_eq(dst_.as_chunks_mut::<3>().0.par_iter_mut())
            .for_each(|(((s0, s1), s2), d)| {
                d[0] = *s0;
                d[1] = *s1;
                d[2] = *s2;
            });
        Ok(())
    }

    pub(super) fn convert_8bps_to_rgba(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src_map = src.map()?;
        let src_ = src_map.as_slice();

        let plane_size = src_.len() / 3;
        let (src0, src1) = src_.split_at(plane_size);
        let (src1, src2) = src1.split_at(plane_size);

        let mut dst_map = dst.map()?;
        let dst_ = dst_map.as_mut_slice();

        src0.par_iter()
            .zip_eq(src1)
            .zip_eq(src2)
            .zip_eq(dst_.as_chunks_mut::<4>().0.par_iter_mut())
            .for_each(|(((s0, s1), s2), d)| {
                d[0] = *s0;
                d[1] = *s1;
                d[2] = *s2;
                d[3] = 255;
            });
        Ok(())
    }

    pub(super) fn convert_prgba_to_rgb(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src_map = src.map()?;
        let src_ = src_map.as_slice();

        let plane_size = src_.len() / 4;
        let (src0, src1) = src_.split_at(plane_size);
        let (src1, src2) = src1.split_at(plane_size);
        let (src2, _src3) = src2.split_at(plane_size);

        let mut dst_map = dst.map()?;
        let dst_ = dst_map.as_mut_slice();

        src0.par_iter()
            .zip_eq(src1)
            .zip_eq(src2)
            .zip_eq(dst_.as_chunks_mut::<3>().0.par_iter_mut())
            .for_each(|(((s0, s1), s2), d)| {
                d[0] = *s0;
                d[1] = *s1;
                d[2] = *s2;
            });
        Ok(())
    }

    pub(super) fn convert_prgba_to_rgba(src: &Tensor<u8>, dst: &mut Tensor<u8>) -> Result<()> {
        let src_map = src.map()?;
        let src_ = src_map.as_slice();

        let plane_size = src_.len() / 4;
        let (src0, src1) = src_.split_at(plane_size);
        let (src1, src2) = src1.split_at(plane_size);
        let (src2, src3) = src2.split_at(plane_size);

        let mut dst_map = dst.map()?;
        let dst_ = dst_map.as_mut_slice();

        src0.par_iter()
            .zip_eq(src1)
            .zip_eq(src2)
            .zip_eq(src3)
            .zip_eq(dst_.as_chunks_mut::<4>().0.par_iter_mut())
            .for_each(|((((s0, s1), s2), s3), d)| {
                d[0] = *s0;
                d[1] = *s1;
                d[2] = *s2;
                d[3] = *s3;
            });
        Ok(())
    }

    pub(super) fn rgba_to_rgb(rgba: [u8; 4]) -> [u8; 3] {
        let [r, g, b, _] = rgba;
        [r, g, b]
    }

    pub(super) fn rgba_to_grey(rgba: [u8; 4]) -> [u8; 1] {
        const BIAS: i32 = 20;
        const KR: f64 = 0.2126f64;
        const KB: f64 = 0.0722f64;
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
