// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    Result, TensorImage, TensorImageDst, BGRA, GREY, NV12, NV16, PLANAR_RGB, PLANAR_RGBA, RGB,
    RGBA, VYUY, YUYV,
};
use edgefirst_tensor::{TensorMapTrait, TensorTrait};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::ops::Shr;

use super::CPUProcessor;

#[inline(always)]
pub(super) fn limit_to_full(l: u8) -> u8 {
    (((l as u16 - 16) * 255 + (240 - 16) / 2) / (240 - 16)) as u8
}

#[inline(always)]
pub(super) fn full_to_limit(l: u8) -> u8 {
    ((l as u16 * (240 - 16) + 255 / 2) / 255 + 16) as u8
}

impl CPUProcessor {
    pub(super) fn convert_nv12_to_rgb(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), NV12);
        assert_eq!(dst.fourcc(), RGB);

        if src.is_multiplane() {
            let y_map = src.tensor.map()?;
            let uv_map = src.chroma.as_ref().unwrap().map()?;
            Self::nv12_to_rgb_kernel(
                y_map.as_slice(),
                uv_map.as_slice(),
                src.width(),
                src.height(),
                dst,
            )
        } else {
            let map = src.tensor.map()?;
            let y_stride = src.width();
            let (y_plane, uv_plane) = map.as_slice().split_at(y_stride * src.height());
            Self::nv12_to_rgb_kernel(y_plane, uv_plane, src.width(), src.height(), dst)
        }
    }

    fn nv12_to_rgb_kernel(
        y_plane: &[u8],
        uv_plane: &[u8],
        width: usize,
        height: usize,
        dst: &mut TensorImage,
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
            dst.tensor.map()?.as_mut_slice(),
            dst.row_stride() as u32,
            yuv::YuvRange::Limited,
            yuv::YuvStandardMatrix::Bt709,
            yuv::YuvConversionMode::Balanced,
        )?)
    }

    // NOTE: The `*_to_rgba` helpers below all accept BGRA destinations
    // (`assert!(matches!(dst.fourcc(), RGBA | BGRA))`). They always write
    // pixels in RGBA channel order; for BGRA destinations the caller applies
    // an R↔B swizzle afterwards via `swizzle_rb_4chan`.
    pub(super) fn convert_nv12_to_rgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), NV12);
        assert!(matches!(dst.fourcc(), RGBA | BGRA));

        if src.is_multiplane() {
            let y_map = src.tensor.map()?;
            let uv_map = src.chroma.as_ref().unwrap().map()?;
            Self::nv12_to_rgba_kernel(
                y_map.as_slice(),
                uv_map.as_slice(),
                src.width(),
                src.height(),
                dst,
            )
        } else {
            let map = src.tensor.map()?;
            let y_stride = src.width();
            let (y_plane, uv_plane) = map.as_slice().split_at(y_stride * src.height());
            Self::nv12_to_rgba_kernel(y_plane, uv_plane, src.width(), src.height(), dst)
        }
    }

    fn nv12_to_rgba_kernel(
        y_plane: &[u8],
        uv_plane: &[u8],
        width: usize,
        height: usize,
        dst: &mut TensorImage,
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
            dst.tensor.map()?.as_mut_slice(),
            dst.row_stride() as u32,
            yuv::YuvRange::Limited,
            yuv::YuvStandardMatrix::Bt709,
            yuv::YuvConversionMode::Balanced,
        )?)
    }

    pub(super) fn convert_nv12_to_grey(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), NV12);
        assert_eq!(dst.fourcc(), GREY);

        // For both multiplane and contiguous, we only need the Y plane.
        // For multiplane, the luma tensor IS the Y plane directly.
        // For contiguous, Y is the first H*W bytes.
        let src_map = src.tensor.map()?;
        let y_len = src.width() * src.height();
        let y_slice = &src_map.as_slice()[..y_len];

        let mut dst_map = dst.tensor.map()?;
        let src_chunks = y_slice.as_chunks::<8>();
        let dst_chunks = dst_map.as_chunks_mut::<8>();
        for (s, d) in src_chunks.0.iter().zip(dst_chunks.0) {
            s.iter().zip(d).for_each(|(s, d)| *d = limit_to_full(*s));
        }

        for (s, d) in src_chunks.1.iter().zip(dst_chunks.1) {
            *d = limit_to_full(*s);
        }

        Ok(())
    }

    pub(super) fn convert_yuyv_to_rgb(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), YUYV);
        assert_eq!(dst.fourcc(), RGB);
        let src = yuv::YuvPackedImage::<u8> {
            yuy: &src.tensor.map()?,
            yuy_stride: src.row_stride() as u32, // we assume packed yuyv
            width: src.width() as u32,
            height: src.height() as u32,
        };

        Ok(yuv::yuyv422_to_rgb(
            &src,
            dst.tensor.map()?.as_mut_slice(),
            dst.width() as u32 * 3,
            yuv::YuvRange::Limited,
            yuv::YuvStandardMatrix::Bt709,
        )?)
    }

    pub(super) fn convert_yuyv_to_rgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), YUYV);
        assert!(matches!(dst.fourcc(), RGBA | BGRA));
        let src = yuv::YuvPackedImage::<u8> {
            yuy: &src.tensor.map()?,
            yuy_stride: src.row_stride() as u32, // we assume packed yuyv
            width: src.width() as u32,
            height: src.height() as u32,
        };

        Ok(yuv::yuyv422_to_rgba(
            &src,
            dst.tensor.map()?.as_mut_slice(),
            dst.row_stride() as u32,
            yuv::YuvRange::Limited,
            yuv::YuvStandardMatrix::Bt709,
        )?)
    }

    pub(super) fn convert_yuyv_to_8bps(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), YUYV);
        assert_eq!(dst.fourcc(), PLANAR_RGB);
        let mut tmp = TensorImage::new(src.width(), src.height(), RGB, None)?;
        Self::convert_yuyv_to_rgb(src, &mut tmp)?;
        Self::convert_rgb_to_8bps(&tmp, dst)
    }

    pub(super) fn convert_yuyv_to_prgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), YUYV);
        assert_eq!(dst.fourcc(), PLANAR_RGBA);
        let mut tmp = TensorImage::new(src.width(), src.height(), RGB, None)?;
        Self::convert_yuyv_to_rgb(src, &mut tmp)?;
        Self::convert_rgb_to_prgba(&tmp, dst)
    }

    pub(super) fn convert_yuyv_to_grey(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), YUYV);
        assert_eq!(dst.fourcc(), GREY);
        let src_map = src.tensor.map()?;
        let mut dst_map = dst.tensor.map()?;
        let src_chunks = src_map.as_chunks::<16>();
        let dst_chunks = dst_map.as_chunks_mut::<8>();
        for (s, d) in src_chunks.0.iter().zip(dst_chunks.0) {
            s.iter()
                .step_by(2)
                .zip(d)
                .for_each(|(s, d)| *d = limit_to_full(*s));
        }

        for (s, d) in src_chunks.1.iter().step_by(2).zip(dst_chunks.1) {
            *d = limit_to_full(*s);
        }

        Ok(())
    }

    pub(super) fn convert_yuyv_to_nv16(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), YUYV);
        assert_eq!(dst.fourcc(), NV16);
        let src_map = src.tensor.map()?;
        let mut dst_map = dst.tensor.map()?;

        let src_chunks = src_map.as_chunks::<2>().0;
        let (y_plane, uv_plane) = dst_map.split_at_mut(dst.row_stride() * dst.height());

        for ((s, y), uv) in src_chunks.iter().zip(y_plane).zip(uv_plane) {
            *y = s[0];
            *uv = s[1];
        }
        Ok(())
    }

    pub(super) fn convert_vyuy_to_rgb(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), VYUY);
        assert_eq!(dst.fourcc(), RGB);
        let src = yuv::YuvPackedImage::<u8> {
            yuy: &src.tensor.map()?,
            yuy_stride: src.row_stride() as u32,
            width: src.width() as u32,
            height: src.height() as u32,
        };

        Ok(yuv::vyuy422_to_rgb(
            &src,
            dst.tensor.map()?.as_mut_slice(),
            dst.width() as u32 * 3,
            yuv::YuvRange::Limited,
            yuv::YuvStandardMatrix::Bt709,
        )?)
    }

    pub(super) fn convert_vyuy_to_rgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), VYUY);
        assert!(matches!(dst.fourcc(), RGBA | BGRA));
        let src = yuv::YuvPackedImage::<u8> {
            yuy: &src.tensor.map()?,
            yuy_stride: src.row_stride() as u32,
            width: src.width() as u32,
            height: src.height() as u32,
        };

        Ok(yuv::vyuy422_to_rgba(
            &src,
            dst.tensor.map()?.as_mut_slice(),
            dst.row_stride() as u32,
            yuv::YuvRange::Limited,
            yuv::YuvStandardMatrix::Bt709,
        )?)
    }

    pub(super) fn convert_vyuy_to_8bps(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), VYUY);
        assert_eq!(dst.fourcc(), PLANAR_RGB);
        let mut tmp = TensorImage::new(src.width(), src.height(), RGB, None)?;
        Self::convert_vyuy_to_rgb(src, &mut tmp)?;
        Self::convert_rgb_to_8bps(&tmp, dst)
    }

    pub(super) fn convert_vyuy_to_prgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), VYUY);
        assert_eq!(dst.fourcc(), PLANAR_RGBA);
        let mut tmp = TensorImage::new(src.width(), src.height(), RGB, None)?;
        Self::convert_vyuy_to_rgb(src, &mut tmp)?;
        Self::convert_rgb_to_prgba(&tmp, dst)
    }

    pub(super) fn convert_vyuy_to_grey(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), VYUY);
        assert_eq!(dst.fourcc(), GREY);
        let src_map = src.tensor.map()?;
        let mut dst_map = dst.tensor.map()?;
        // VYUY byte order: [V, Y0, U, Y1] — Y at offsets 1, 3
        let src_chunks = src_map.as_chunks::<16>();
        let dst_chunks = dst_map.as_chunks_mut::<8>();
        for (s, d) in src_chunks.0.iter().zip(dst_chunks.0) {
            for (di, si) in (1..16).step_by(2).enumerate() {
                d[di] = limit_to_full(s[si]);
            }
        }

        for (di, si) in (1..src_chunks.1.len()).step_by(2).enumerate() {
            dst_chunks.1[di] = limit_to_full(src_chunks.1[si]);
        }

        Ok(())
    }

    pub(super) fn convert_vyuy_to_nv16(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), VYUY);
        assert_eq!(dst.fourcc(), NV16);
        let src_map = src.tensor.map()?;
        let mut dst_map = dst.tensor.map()?;

        // VYUY byte order: [V, Y0, U, Y1] — per 4-byte macropixel
        let src_chunks = src_map.as_chunks::<4>().0;
        let (y_plane, uv_plane) = dst_map.split_at_mut(dst.row_stride() * dst.height());
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

    pub(super) fn convert_grey_to_rgb(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), GREY);
        assert_eq!(dst.fourcc(), RGB);
        let src = yuv::YuvGrayImage::<u8> {
            y_plane: &src.tensor.map()?,
            y_stride: src.row_stride() as u32, // we assume packed Y
            width: src.width() as u32,
            height: src.height() as u32,
        };
        Ok(yuv::yuv400_to_rgb(
            &src,
            dst.tensor.map()?.as_mut_slice(),
            dst.row_stride() as u32,
            yuv::YuvRange::Full,
            yuv::YuvStandardMatrix::Bt709,
        )?)
    }

    pub(super) fn convert_grey_to_rgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), GREY);
        assert!(matches!(dst.fourcc(), RGBA | BGRA));
        let src = yuv::YuvGrayImage::<u8> {
            y_plane: &src.tensor.map()?,
            y_stride: src.row_stride() as u32,
            width: src.width() as u32,
            height: src.height() as u32,
        };
        Ok(yuv::yuv400_to_rgba(
            &src,
            dst.tensor.map()?.as_mut_slice(),
            dst.row_stride() as u32,
            yuv::YuvRange::Full,
            yuv::YuvStandardMatrix::Bt709,
        )?)
    }

    pub(super) fn convert_grey_to_8bps(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), GREY);
        assert_eq!(dst.fourcc(), PLANAR_RGB);

        let src = src.tensor().map()?;
        let src = src.as_slice();

        let mut dst_map = dst.tensor().map()?;
        let dst_ = dst_map.as_mut_slice();

        let (dst0, dst1) = dst_.split_at_mut(dst.width() * dst.height());
        let (dst1, dst2) = dst1.split_at_mut(dst.width() * dst.height());

        rayon::scope(|s| {
            s.spawn(|_| dst0.copy_from_slice(src));
            s.spawn(|_| dst1.copy_from_slice(src));
            s.spawn(|_| dst2.copy_from_slice(src));
        });
        Ok(())
    }

    pub(super) fn convert_grey_to_prgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), GREY);
        assert_eq!(dst.fourcc(), PLANAR_RGBA);

        let src = src.tensor().map()?;
        let src = src.as_slice();

        let mut dst_map = dst.tensor().map()?;
        let dst_ = dst_map.as_mut_slice();

        let (dst0, dst1) = dst_.split_at_mut(dst.width() * dst.height());
        let (dst1, dst2) = dst1.split_at_mut(dst.width() * dst.height());
        let (dst2, dst3) = dst2.split_at_mut(dst.width() * dst.height());
        rayon::scope(|s| {
            s.spawn(|_| dst0.copy_from_slice(src));
            s.spawn(|_| dst1.copy_from_slice(src));
            s.spawn(|_| dst2.copy_from_slice(src));
            s.spawn(|_| dst3.fill(255));
        });
        Ok(())
    }

    pub(super) fn convert_grey_to_yuyv(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), GREY);
        assert_eq!(dst.fourcc(), YUYV);

        let src = src.tensor().map()?;
        let src = src.as_slice();

        let mut dst = dst.tensor().map()?;
        let dst = dst.as_mut_slice();
        for (s, d) in src
            .as_chunks::<2>()
            .0
            .iter()
            .zip(dst.as_chunks_mut::<4>().0.iter_mut())
        {
            d[0] = full_to_limit(s[0]);
            d[1] = 128;

            d[2] = full_to_limit(s[1]);
            d[3] = 128;
        }
        Ok(())
    }

    pub(super) fn convert_grey_to_nv16(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), GREY);
        assert_eq!(dst.fourcc(), NV16);

        let src = src.tensor().map()?;
        let src = src.as_slice();

        let mut dst = dst.tensor().map()?;
        let dst = dst.as_mut_slice();

        for (s, d) in src.iter().zip(dst[0..src.len()].iter_mut()) {
            *d = full_to_limit(*s);
        }
        dst[src.len()..].fill(128);

        Ok(())
    }

    pub(super) fn convert_rgba_to_rgb(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), RGBA);
        assert_eq!(dst.fourcc(), RGB);

        Ok(yuv::rgba_to_rgb(
            src.tensor.map()?.as_slice(),
            (src.width() * src.channels()) as u32,
            dst.tensor.map()?.as_mut_slice(),
            (dst.width() * dst.channels()) as u32,
            src.width() as u32,
            src.height() as u32,
        )?)
    }

    pub(super) fn convert_rgba_to_grey(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), RGBA);
        assert_eq!(dst.fourcc(), GREY);

        let mut dst = yuv::YuvGrayImageMut::<u8> {
            y_plane: yuv::BufferStoreMut::Borrowed(&mut dst.tensor.map()?),
            y_stride: dst.row_stride() as u32,
            width: dst.width() as u32,
            height: dst.height() as u32,
        };
        Ok(yuv::rgba_to_yuv400(
            &mut dst,
            src.tensor.map()?.as_slice(),
            src.row_stride() as u32,
            yuv::YuvRange::Full,
            yuv::YuvStandardMatrix::Bt709,
        )?)
    }

    pub(super) fn convert_rgba_to_8bps(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), RGBA);
        assert_eq!(dst.fourcc(), PLANAR_RGB);

        let src = src.tensor().map()?;
        let src = src.as_slice();
        let src = src.as_chunks::<4>().0;

        let mut dst_map = dst.tensor().map()?;
        let dst_ = dst_map.as_mut_slice();

        let (dst0, dst1) = dst_.split_at_mut(dst.width() * dst.height());
        let (dst1, dst2) = dst1.split_at_mut(dst.width() * dst.height());

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

    pub(super) fn convert_rgba_to_prgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), RGBA);
        assert_eq!(dst.fourcc(), PLANAR_RGBA);

        let src = src.tensor().map()?;
        let src = src.as_slice();
        let src = src.as_chunks::<4>().0;

        let mut dst_map = dst.tensor().map()?;
        let dst_ = dst_map.as_mut_slice();

        let (dst0, dst1) = dst_.split_at_mut(dst.width() * dst.height());
        let (dst1, dst2) = dst1.split_at_mut(dst.width() * dst.height());
        let (dst2, dst3) = dst2.split_at_mut(dst.width() * dst.height());

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

    pub(super) fn convert_rgba_to_yuyv(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), RGBA);
        assert_eq!(dst.fourcc(), YUYV);

        let src = src.tensor().map()?;
        let src = src.as_slice();

        let mut dst = dst.tensor().map()?;
        let dst = dst.as_mut_slice();

        // compute quantized Bt.709 limited range RGB to YUV matrix
        const KR: f64 = 0.2126f64;
        const KB: f64 = 0.0722f64;
        const KG: f64 = 1.0 - KR - KB;
        const BIAS: i32 = 20;

        const Y_R: i32 = (KR * (219 << BIAS) as f64 / 255.0).round() as i32;
        const Y_G: i32 = (KG * (219 << BIAS) as f64 / 255.0).round() as i32;
        const Y_B: i32 = (KB * (219 << BIAS) as f64 / 255.0).round() as i32;

        const U_R: i32 = (-KR / (KR + KG) / 2.0 * (224 << BIAS) as f64 / 255.0).round() as i32;
        const U_G: i32 = (-KG / (KR + KG) / 2.0 * (224 << BIAS) as f64 / 255.0).round() as i32;
        const U_B: i32 = (0.5_f64 * (224 << BIAS) as f64 / 255.0).ceil() as i32;

        const V_R: i32 = (0.5_f64 * (224 << BIAS) as f64 / 255.0).ceil() as i32;
        const V_G: i32 = (-KG / (KG + KB) / 2.0 * (224 << BIAS) as f64 / 255.0).round() as i32;
        const V_B: i32 = (-KB / (KG + KB) / 2.0 * (224 << BIAS) as f64 / 255.0).round() as i32;
        const ROUND: i32 = 1 << (BIAS - 1);
        const ROUND2: i32 = 1 << BIAS;
        let process_rgba_to_yuyv = |s: &[u8; 8], d: &mut [u8; 4]| {
            let [r0, g0, b0, _, r1, g1, b1, _] = *s;
            let r0 = r0 as i32;
            let g0 = g0 as i32;
            let b0 = b0 as i32;
            let r1 = r1 as i32;
            let g1 = g1 as i32;
            let b1 = b1 as i32;
            d[0] = ((Y_R * r0 + Y_G * g0 + Y_B * b0 + ROUND).shr(BIAS) + 16) as u8;
            d[1] = ((U_R * r0 + U_G * g0 + U_B * b0 + U_R * r1 + U_G * g1 + U_B * b1 + ROUND2)
                .shr(BIAS + 1)
                + 128) as u8;
            d[2] = ((Y_R * r1 + Y_G * g1 + Y_B * b1 + ROUND).shr(BIAS) + 16) as u8;
            d[3] = ((V_R * r0 + V_G * g0 + V_B * b0 + V_R * r1 + V_G * g1 + V_B * b1 + ROUND2)
                .shr(BIAS + 1)
                + 128) as u8;
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

    pub(super) fn convert_rgba_to_nv16(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), RGBA);
        assert_eq!(dst.fourcc(), NV16);

        let mut dst_map = dst.tensor().map()?;

        let (y_plane, uv_plane) = dst_map.split_at_mut(dst.width() * dst.height());
        let mut bi_planar_image = yuv::YuvBiPlanarImageMut::<u8> {
            y_plane: yuv::BufferStoreMut::Borrowed(y_plane),
            y_stride: dst.width() as u32,
            uv_plane: yuv::BufferStoreMut::Borrowed(uv_plane),
            uv_stride: dst.width() as u32,
            width: dst.width() as u32,
            height: dst.height() as u32,
        };

        Ok(yuv::rgba_to_yuv_nv16(
            &mut bi_planar_image,
            src.tensor.map()?.as_slice(),
            src.row_stride() as u32,
            yuv::YuvRange::Limited,
            yuv::YuvStandardMatrix::Bt709,
            yuv::YuvConversionMode::Balanced,
        )?)
    }

    pub(super) fn convert_rgb_to_rgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), RGB);
        assert!(matches!(dst.fourcc(), RGBA | BGRA));

        Ok(yuv::rgb_to_rgba(
            src.tensor.map()?.as_slice(),
            (src.width() * src.channels()) as u32,
            dst.tensor.map()?.as_mut_slice(),
            (dst.width() * dst.channels()) as u32,
            src.width() as u32,
            src.height() as u32,
        )?)
    }

    pub(super) fn convert_rgb_to_grey(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), RGB);
        assert_eq!(dst.fourcc(), GREY);

        let mut dst = yuv::YuvGrayImageMut::<u8> {
            y_plane: yuv::BufferStoreMut::Borrowed(&mut dst.tensor.map()?),
            y_stride: dst.row_stride() as u32,
            width: dst.width() as u32,
            height: dst.height() as u32,
        };
        Ok(yuv::rgb_to_yuv400(
            &mut dst,
            src.tensor.map()?.as_slice(),
            src.row_stride() as u32,
            yuv::YuvRange::Full,
            yuv::YuvStandardMatrix::Bt709,
        )?)
    }

    pub(super) fn convert_rgb_to_8bps(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), RGB);
        assert_eq!(dst.fourcc(), PLANAR_RGB);

        let src = src.tensor().map()?;
        let src = src.as_slice();
        let src = src.as_chunks::<3>().0;

        let mut dst_map = dst.tensor().map()?;
        let dst_ = dst_map.as_mut_slice();

        let (dst0, dst1) = dst_.split_at_mut(dst.width() * dst.height());
        let (dst1, dst2) = dst1.split_at_mut(dst.width() * dst.height());

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

    pub(super) fn convert_rgb_to_prgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), RGB);
        assert_eq!(dst.fourcc(), PLANAR_RGBA);

        let src = src.tensor().map()?;
        let src = src.as_slice();
        let src = src.as_chunks::<3>().0;

        let mut dst_map = dst.tensor().map()?;
        let dst_ = dst_map.as_mut_slice();

        let (dst0, dst1) = dst_.split_at_mut(dst.width() * dst.height());
        let (dst1, dst2) = dst1.split_at_mut(dst.width() * dst.height());
        let (dst2, dst3) = dst2.split_at_mut(dst.width() * dst.height());

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

    pub(super) fn convert_rgb_to_yuyv(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), RGB);
        assert_eq!(dst.fourcc(), YUYV);

        let src = src.tensor().map()?;
        let src = src.as_slice();

        let mut dst = dst.tensor().map()?;
        let dst = dst.as_mut_slice();

        // compute quantized Bt.709 limited range RGB to YUV matrix
        const BIAS: i32 = 20;
        const KR: f64 = 0.2126f64;
        const KB: f64 = 0.0722f64;
        const KG: f64 = 1.0 - KR - KB;
        const Y_R: i32 = (KR * (219 << BIAS) as f64 / 255.0).round() as i32;
        const Y_G: i32 = (KG * (219 << BIAS) as f64 / 255.0).round() as i32;
        const Y_B: i32 = (KB * (219 << BIAS) as f64 / 255.0).round() as i32;

        const U_R: i32 = (-KR / (KR + KG) / 2.0 * (224 << BIAS) as f64 / 255.0).round() as i32;
        const U_G: i32 = (-KG / (KR + KG) / 2.0 * (224 << BIAS) as f64 / 255.0).round() as i32;
        const U_B: i32 = (0.5_f64 * (224 << BIAS) as f64 / 255.0).ceil() as i32;

        const V_R: i32 = (0.5_f64 * (224 << BIAS) as f64 / 255.0).ceil() as i32;
        const V_G: i32 = (-KG / (KG + KB) / 2.0 * (224 << BIAS) as f64 / 255.0).round() as i32;
        const V_B: i32 = (-KB / (KG + KB) / 2.0 * (224 << BIAS) as f64 / 255.0).round() as i32;
        const ROUND: i32 = 1 << (BIAS - 1);
        const ROUND2: i32 = 1 << BIAS;
        let process_rgb_to_yuyv = |s: &[u8; 6], d: &mut [u8; 4]| {
            let [r0, g0, b0, r1, g1, b1] = *s;
            let r0 = r0 as i32;
            let g0 = g0 as i32;
            let b0 = b0 as i32;
            let r1 = r1 as i32;
            let g1 = g1 as i32;
            let b1 = b1 as i32;
            d[0] = ((Y_R * r0 + Y_G * g0 + Y_B * b0 + ROUND).shr(BIAS) + 16) as u8;
            d[1] = ((U_R * r0 + U_G * g0 + U_B * b0 + U_R * r1 + U_G * g1 + U_B * b1 + ROUND2)
                .shr(BIAS + 1)
                + 128) as u8;
            d[2] = ((Y_R * r1 + Y_G * g1 + Y_B * b1 + ROUND).shr(BIAS) + 16) as u8;
            d[3] = ((V_R * r0 + V_G * g0 + V_B * b0 + V_R * r1 + V_G * g1 + V_B * b1 + ROUND2)
                .shr(BIAS + 1)
                + 128) as u8;
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

    pub(super) fn convert_rgb_to_nv16(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), RGB);
        assert_eq!(dst.fourcc(), NV16);

        let mut dst_map = dst.tensor().map()?;

        let (y_plane, uv_plane) = dst_map.split_at_mut(dst.width() * dst.height());
        let mut bi_planar_image = yuv::YuvBiPlanarImageMut::<u8> {
            y_plane: yuv::BufferStoreMut::Borrowed(y_plane),
            y_stride: dst.width() as u32,
            uv_plane: yuv::BufferStoreMut::Borrowed(uv_plane),
            uv_stride: dst.width() as u32,
            width: dst.width() as u32,
            height: dst.height() as u32,
        };

        Ok(yuv::rgb_to_yuv_nv16(
            &mut bi_planar_image,
            src.tensor.map()?.as_slice(),
            src.row_stride() as u32,
            yuv::YuvRange::Limited,
            yuv::YuvStandardMatrix::Bt709,
            yuv::YuvConversionMode::Balanced,
        )?)
    }

    pub(super) fn copy_image(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), dst.fourcc());
        dst.tensor().map()?.copy_from_slice(&src.tensor().map()?);
        Ok(())
    }

    /// Swap R and B channels in-place for an interleaved 4-channel image.
    pub(super) fn swizzle_rb_4chan(dst: &mut TensorImage) -> Result<()> {
        let mut map = dst.tensor().map()?;
        let buf = map.as_mut_slice();
        for chunk in buf.chunks_exact_mut(4) {
            chunk.swap(0, 2);
        }
        Ok(())
    }

    pub(super) fn convert_nv16_to_rgb(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), NV16);
        assert_eq!(dst.fourcc(), RGB);

        if src.is_multiplane() {
            let y_map = src.tensor.map()?;
            let uv_map = src.chroma.as_ref().unwrap().map()?;
            Self::nv16_to_rgb_kernel(
                y_map.as_slice(),
                uv_map.as_slice(),
                src.width(),
                src.height(),
                dst,
            )
        } else {
            let map = src.tensor.map()?;
            let y_stride = src.width();
            let (y_plane, uv_plane) = map.as_slice().split_at(y_stride * src.height());
            Self::nv16_to_rgb_kernel(y_plane, uv_plane, src.width(), src.height(), dst)
        }
    }

    fn nv16_to_rgb_kernel(
        y_plane: &[u8],
        uv_plane: &[u8],
        width: usize,
        height: usize,
        dst: &mut TensorImage,
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
            dst.tensor.map()?.as_mut_slice(),
            dst.row_stride() as u32,
            yuv::YuvRange::Limited,
            yuv::YuvStandardMatrix::Bt709,
            yuv::YuvConversionMode::Balanced,
        )?)
    }

    pub(super) fn convert_nv16_to_rgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), NV16);
        assert!(matches!(dst.fourcc(), RGBA | BGRA));

        if src.is_multiplane() {
            let y_map = src.tensor.map()?;
            let uv_map = src.chroma.as_ref().unwrap().map()?;
            Self::nv16_to_rgba_kernel(
                y_map.as_slice(),
                uv_map.as_slice(),
                src.width(),
                src.height(),
                dst,
            )
        } else {
            let map = src.tensor.map()?;
            let y_stride = src.width();
            let (y_plane, uv_plane) = map.as_slice().split_at(y_stride * src.height());
            Self::nv16_to_rgba_kernel(y_plane, uv_plane, src.width(), src.height(), dst)
        }
    }

    fn nv16_to_rgba_kernel(
        y_plane: &[u8],
        uv_plane: &[u8],
        width: usize,
        height: usize,
        dst: &mut TensorImage,
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
            dst.tensor.map()?.as_mut_slice(),
            dst.row_stride() as u32,
            yuv::YuvRange::Limited,
            yuv::YuvStandardMatrix::Bt709,
            yuv::YuvConversionMode::Balanced,
        )?)
    }

    pub(super) fn convert_8bps_to_rgb(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), PLANAR_RGB);
        assert_eq!(dst.fourcc(), RGB);

        let src_map = src.tensor().map()?;
        let src_ = src_map.as_slice();

        let (src0, src1) = src_.split_at(src.width() * src.height());
        let (src1, src2) = src1.split_at(src.width() * src.height());

        let mut dst_map = dst.tensor().map()?;
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

    pub(super) fn convert_8bps_to_rgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), PLANAR_RGB);
        assert!(matches!(dst.fourcc(), RGBA | BGRA));

        let src_map = src.tensor().map()?;
        let src_ = src_map.as_slice();

        let (src0, src1) = src_.split_at(src.width() * src.height());
        let (src1, src2) = src1.split_at(src.width() * src.height());

        let mut dst_map = dst.tensor().map()?;
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

    pub(super) fn convert_prgba_to_rgb(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), PLANAR_RGBA);
        assert_eq!(dst.fourcc(), RGB);

        let src_map = src.tensor().map()?;
        let src_ = src_map.as_slice();

        let (src0, src1) = src_.split_at(src.width() * src.height());
        let (src1, src2) = src1.split_at(src.width() * src.height());
        let (src2, _src3) = src2.split_at(src.width() * src.height());

        let mut dst_map = dst.tensor().map()?;
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

    pub(super) fn convert_prgba_to_rgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), PLANAR_RGBA);
        assert!(matches!(dst.fourcc(), RGBA | BGRA));

        let src_map = src.tensor().map()?;
        let src_ = src_map.as_slice();

        let (src0, src1) = src_.split_at(src.width() * src.height());
        let (src1, src2) = src1.split_at(src.width() * src.height());
        let (src2, src3) = src2.split_at(src.width() * src.height());

        let mut dst_map = dst.tensor().map()?;
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

    /// Generic RGB to PLANAR_RGB conversion that works with any TensorImageDst.
    pub(super) fn convert_rgb_to_planar_rgb_generic<D: TensorImageDst>(
        src: &TensorImage,
        dst: &mut D,
    ) -> Result<()> {
        assert_eq!(src.fourcc(), RGB);
        assert_eq!(dst.fourcc(), PLANAR_RGB);

        let src = src.tensor().map()?;
        let src = src.as_slice();
        let src = src.as_chunks::<3>().0;

        let mut dst_map = dst.tensor_mut().map()?;
        let dst_ = dst_map.as_mut_slice();

        let (dst0, dst1) = dst_.split_at_mut(dst.width() * dst.height());
        let (dst1, dst2) = dst1.split_at_mut(dst.width() * dst.height());

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

    /// Generic RGBA to PLANAR_RGB conversion that works with any
    /// TensorImageDst.
    pub(super) fn convert_rgba_to_planar_rgb_generic<D: TensorImageDst>(
        src: &TensorImage,
        dst: &mut D,
    ) -> Result<()> {
        assert_eq!(src.fourcc(), RGBA);
        assert_eq!(dst.fourcc(), PLANAR_RGB);

        let src = src.tensor().map()?;
        let src = src.as_slice();
        let src = src.as_chunks::<4>().0;

        let mut dst_map = dst.tensor_mut().map()?;
        let dst_ = dst_map.as_mut_slice();

        let (dst0, dst1) = dst_.split_at_mut(dst.width() * dst.height());
        let (dst1, dst2) = dst1.split_at_mut(dst.width() * dst.height());

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

    pub(super) fn rgba_to_yuyv(rgba: [u8; 4]) -> [u8; 4] {
        const KR: f64 = 0.2126f64;
        const KB: f64 = 0.0722f64;
        const KG: f64 = 1.0 - KR - KB;
        const BIAS: i32 = 20;

        const Y_R: i32 = (KR * (219 << BIAS) as f64 / 255.0).round() as i32;
        const Y_G: i32 = (KG * (219 << BIAS) as f64 / 255.0).round() as i32;
        const Y_B: i32 = (KB * (219 << BIAS) as f64 / 255.0).round() as i32;

        const U_R: i32 = (-KR / (KR + KG) / 2.0 * (224 << BIAS) as f64 / 255.0).round() as i32;
        const U_G: i32 = (-KG / (KR + KG) / 2.0 * (224 << BIAS) as f64 / 255.0).round() as i32;
        const U_B: i32 = (0.5_f64 * (224 << BIAS) as f64 / 255.0).ceil() as i32;

        const V_R: i32 = (0.5_f64 * (224 << BIAS) as f64 / 255.0).ceil() as i32;
        const V_G: i32 = (-KG / (KG + KB) / 2.0 * (224 << BIAS) as f64 / 255.0).round() as i32;
        const V_B: i32 = (-KB / (KG + KB) / 2.0 * (224 << BIAS) as f64 / 255.0).round() as i32;
        const ROUND: i32 = 1 << (BIAS - 1);

        let [r, g, b, _] = rgba;
        let r = r as i32;
        let g = g as i32;
        let b = b as i32;
        let y = (((Y_R * r + Y_G * g + Y_B * b + ROUND) >> BIAS) + 16) as u8;
        let u = (((U_R * r + U_G * g + U_B * b + ROUND) >> BIAS) + 128) as u8;
        let v = (((V_R * r + V_G * g + V_B * b + ROUND) >> BIAS) + 128) as u8;

        [y, u, y, v]
    }
}
