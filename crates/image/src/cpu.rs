// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    Crop, Error, Flip, FunctionTimer, GREY, ImageConverterTrait, NV12, NV16, PLANAR_RGB,
    PLANAR_RGBA, RGB, RGBA, Rect, Result, Rotation, TensorImage, YUYV,
};
use edgefirst_tensor::{TensorMapTrait, TensorTrait};
use four_char_code::FourCharCode;
use ndarray::{ArrayView3, ArrayViewMut3, Axis};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};
use std::ops::Shr;

/// CPUConverter implements the ImageConverter trait using the fallback CPU
/// implementation for image processing.
#[derive(Debug, Clone)]
pub struct CPUConverter {
    resizer: fast_image_resize::Resizer,
    options: fast_image_resize::ResizeOptions,
}

unsafe impl Send for CPUConverter {}
unsafe impl Sync for CPUConverter {}

#[inline(always)]
fn limit_to_full(l: u8) -> u8 {
    (((l as u16 - 16) * 255 + (240 - 16) / 2) / (240 - 16)) as u8
}

#[inline(always)]
fn full_to_limit(l: u8) -> u8 {
    ((l as u16 * (240 - 16) + 255 / 2) / 255 + 16) as u8
}

impl Default for CPUConverter {
    fn default() -> Self {
        Self::new_bilinear()
    }
}

impl CPUConverter {
    /// Creates a new CPUConverter with bilinear resizing.
    pub fn new() -> Self {
        Self::new_bilinear()
    }

    /// Creates a new CPUConverter with bilinear resizing.
    fn new_bilinear() -> Self {
        let resizer = fast_image_resize::Resizer::new();
        let options = fast_image_resize::ResizeOptions::new()
            .resize_alg(fast_image_resize::ResizeAlg::Convolution(
                fast_image_resize::FilterType::Bilinear,
            ))
            .use_alpha(false);

        log::debug!("CPUConverter created");
        Self { resizer, options }
    }

    /// Creates a new CPUConverter with nearest neighbor resizing.
    pub fn new_nearest() -> Self {
        let resizer = fast_image_resize::Resizer::new();
        let options = fast_image_resize::ResizeOptions::new()
            .resize_alg(fast_image_resize::ResizeAlg::Nearest)
            .use_alpha(false);
        log::debug!("CPUConverter created");
        Self { resizer, options }
    }

    pub(crate) fn flip_rotate_ndarray(
        src_map: &[u8],
        dst_map: &mut [u8],
        dst: &TensorImage,
        rotation: Rotation,
        flip: Flip,
    ) -> Result<(), crate::Error> {
        let mut dst_view =
            ArrayViewMut3::from_shape((dst.height(), dst.width(), dst.channels()), dst_map)?;
        let mut src_view = match rotation {
            Rotation::None | Rotation::Rotate180 => {
                ArrayView3::from_shape((dst.height(), dst.width(), dst.channels()), src_map)?
            }
            Rotation::Clockwise90 | Rotation::CounterClockwise90 => {
                ArrayView3::from_shape((dst.width(), dst.height(), dst.channels()), src_map)?
            }
        };

        match flip {
            Flip::None => {}
            Flip::Vertical => {
                src_view.invert_axis(Axis(0));
            }
            Flip::Horizontal => {
                src_view.invert_axis(Axis(1));
            }
        }

        match rotation {
            Rotation::None => {}
            Rotation::Clockwise90 => {
                src_view.swap_axes(0, 1);
                src_view.invert_axis(Axis(1));
            }
            Rotation::Rotate180 => {
                src_view.invert_axis(Axis(0));
                src_view.invert_axis(Axis(1));
            }
            Rotation::CounterClockwise90 => {
                src_view.swap_axes(0, 1);
                src_view.invert_axis(Axis(0));
            }
        }

        dst_view.assign(&src_view);

        Ok(())
    }

    fn convert_nv12_to_rgb(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), NV12);
        assert_eq!(dst.fourcc(), RGB);
        let map = src.tensor.map()?;
        let y_stride = src.width() as u32;
        let uv_stride = src.width() as u32;
        let slices = map.as_slice().split_at(y_stride as usize * src.height());

        let src = yuv::YuvBiPlanarImage {
            y_plane: slices.0,
            y_stride,
            uv_plane: slices.1,
            uv_stride,
            width: src.width() as u32,
            height: src.height() as u32,
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

    fn convert_nv12_to_rgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), NV12);
        assert_eq!(dst.fourcc(), RGBA);
        let map = src.tensor.map()?;
        let y_stride = src.width() as u32;
        let uv_stride = src.width() as u32;
        let slices = map.as_slice().split_at(y_stride as usize * src.height());

        let src = yuv::YuvBiPlanarImage {
            y_plane: slices.0,
            y_stride,
            uv_plane: slices.1,
            uv_stride,
            width: src.width() as u32,
            height: src.height() as u32,
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

    fn convert_nv12_to_grey(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), NV12);
        assert_eq!(dst.fourcc(), GREY);
        let src_map = src.tensor.map()?;
        let mut dst_map = dst.tensor.map()?;
        let y_stride = src.width() as u32;
        let y_slice = src_map
            .as_slice()
            .split_at(y_stride as usize * src.height())
            .0;
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

    fn convert_yuyv_to_rgb(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_yuyv_to_rgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), YUYV);
        assert_eq!(dst.fourcc(), RGBA);
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

    fn convert_yuyv_to_8bps(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), YUYV);
        assert_eq!(dst.fourcc(), PLANAR_RGB);
        let mut tmp = TensorImage::new(src.width(), src.height(), RGB, None)?;
        Self::convert_yuyv_to_rgb(src, &mut tmp)?;
        Self::convert_rgb_to_8bps(&tmp, dst)
    }

    fn convert_yuyv_to_prgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), YUYV);
        assert_eq!(dst.fourcc(), PLANAR_RGBA);
        let mut tmp = TensorImage::new(src.width(), src.height(), RGB, None)?;
        Self::convert_yuyv_to_rgb(src, &mut tmp)?;
        Self::convert_rgb_to_prgba(&tmp, dst)
    }

    fn convert_yuyv_to_grey(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_yuyv_to_nv16(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_grey_to_rgb(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_grey_to_rgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), GREY);
        assert_eq!(dst.fourcc(), RGBA);
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

    fn convert_grey_to_8bps(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_grey_to_prgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_grey_to_yuyv(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_grey_to_nv16(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_rgba_to_rgb(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_rgba_to_grey(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_rgba_to_8bps(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_rgba_to_prgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_rgba_to_yuyv(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_rgba_to_nv16(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_rgb_to_rgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), RGB);
        assert_eq!(dst.fourcc(), RGBA);

        Ok(yuv::rgb_to_rgba(
            src.tensor.map()?.as_slice(),
            (src.width() * src.channels()) as u32,
            dst.tensor.map()?.as_mut_slice(),
            (dst.width() * dst.channels()) as u32,
            src.width() as u32,
            src.height() as u32,
        )?)
    }

    fn convert_rgb_to_grey(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_rgb_to_8bps(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_rgb_to_prgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_rgb_to_yuyv(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_rgb_to_nv16(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn copy_image(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), dst.fourcc());
        dst.tensor().map()?.copy_from_slice(&src.tensor().map()?);
        Ok(())
    }

    fn convert_nv16_to_rgb(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), NV16);
        assert_eq!(dst.fourcc(), RGB);
        let map = src.tensor.map()?;
        let y_stride = src.width() as u32;
        let uv_stride = src.width() as u32;
        let slices = map.as_slice().split_at(y_stride as usize * src.height());

        let src = yuv::YuvBiPlanarImage {
            y_plane: slices.0,
            y_stride,
            uv_plane: slices.1,
            uv_stride,
            width: src.width() as u32,
            height: src.height() as u32,
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

    fn convert_nv16_to_rgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), NV16);
        assert_eq!(dst.fourcc(), RGBA);
        let map = src.tensor.map()?;
        let y_stride = src.width() as u32;
        let uv_stride = src.width() as u32;
        let slices = map.as_slice().split_at(y_stride as usize * src.height());

        let src = yuv::YuvBiPlanarImage {
            y_plane: slices.0,
            y_stride,
            uv_plane: slices.1,
            uv_stride,
            width: src.width() as u32,
            height: src.height() as u32,
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

    fn convert_8bps_to_rgb(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_8bps_to_rgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), PLANAR_RGB);
        assert_eq!(dst.fourcc(), RGBA);

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

    fn convert_prgba_to_rgb(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_prgba_to_rgba(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), PLANAR_RGBA);
        assert_eq!(dst.fourcc(), RGBA);

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

    pub(crate) fn support_conversion(src: FourCharCode, dst: FourCharCode) -> bool {
        matches!(
            (src, dst),
            (NV12, RGB)
                | (NV12, RGBA)
                | (NV12, GREY)
                | (NV16, RGB)
                | (NV16, RGBA)
                | (YUYV, RGB)
                | (YUYV, RGBA)
                | (YUYV, GREY)
                | (YUYV, YUYV)
                | (YUYV, PLANAR_RGB)
                | (YUYV, PLANAR_RGBA)
                | (YUYV, NV16)
                | (RGBA, RGB)
                | (RGBA, RGBA)
                | (RGBA, GREY)
                | (RGBA, YUYV)
                | (RGBA, PLANAR_RGB)
                | (RGBA, PLANAR_RGBA)
                | (RGBA, NV16)
                | (RGB, RGB)
                | (RGB, RGBA)
                | (RGB, GREY)
                | (RGB, YUYV)
                | (RGB, PLANAR_RGB)
                | (RGB, PLANAR_RGBA)
                | (RGB, NV16)
                | (GREY, RGB)
                | (GREY, RGBA)
                | (GREY, GREY)
                | (GREY, YUYV)
                | (GREY, PLANAR_RGB)
                | (GREY, PLANAR_RGBA)
                | (GREY, NV16)
        )
    }

    pub(crate) fn convert_format(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        // shapes should be equal
        let _timer = FunctionTimer::new(format!(
            "ImageConverter::convert_format {} to {}",
            src.fourcc().display(),
            dst.fourcc().display()
        ));
        assert_eq!(src.height(), dst.height());
        assert_eq!(src.width(), dst.width());

        match (src.fourcc(), dst.fourcc()) {
            (NV12, RGB) => Self::convert_nv12_to_rgb(src, dst),
            (NV12, RGBA) => Self::convert_nv12_to_rgba(src, dst),
            (NV12, GREY) => Self::convert_nv12_to_grey(src, dst),
            (YUYV, RGB) => Self::convert_yuyv_to_rgb(src, dst),
            (YUYV, RGBA) => Self::convert_yuyv_to_rgba(src, dst),
            (YUYV, GREY) => Self::convert_yuyv_to_grey(src, dst),
            (YUYV, YUYV) => Self::copy_image(src, dst),
            (YUYV, PLANAR_RGB) => Self::convert_yuyv_to_8bps(src, dst),
            (YUYV, PLANAR_RGBA) => Self::convert_yuyv_to_prgba(src, dst),
            (YUYV, NV16) => Self::convert_yuyv_to_nv16(src, dst),
            (RGBA, RGB) => Self::convert_rgba_to_rgb(src, dst),
            (RGBA, RGBA) => Self::copy_image(src, dst),
            (RGBA, GREY) => Self::convert_rgba_to_grey(src, dst),
            (RGBA, YUYV) => Self::convert_rgba_to_yuyv(src, dst),
            (RGBA, PLANAR_RGB) => Self::convert_rgba_to_8bps(src, dst),
            (RGBA, PLANAR_RGBA) => Self::convert_rgba_to_prgba(src, dst),
            (RGBA, NV16) => Self::convert_rgba_to_nv16(src, dst),
            (RGB, RGB) => Self::copy_image(src, dst),
            (RGB, RGBA) => Self::convert_rgb_to_rgba(src, dst),
            (RGB, GREY) => Self::convert_rgb_to_grey(src, dst),
            (RGB, YUYV) => Self::convert_rgb_to_yuyv(src, dst),
            (RGB, PLANAR_RGB) => Self::convert_rgb_to_8bps(src, dst),
            (RGB, PLANAR_RGBA) => Self::convert_rgb_to_prgba(src, dst),
            (RGB, NV16) => Self::convert_rgb_to_nv16(src, dst),
            (GREY, RGB) => Self::convert_grey_to_rgb(src, dst),
            (GREY, RGBA) => Self::convert_grey_to_rgba(src, dst),
            (GREY, GREY) => Self::copy_image(src, dst),
            (GREY, YUYV) => Self::convert_grey_to_yuyv(src, dst),
            (GREY, PLANAR_RGB) => Self::convert_grey_to_8bps(src, dst),
            (GREY, PLANAR_RGBA) => Self::convert_grey_to_prgba(src, dst),
            (GREY, NV16) => Self::convert_grey_to_nv16(src, dst),

            // the following converts are added for use in testing
            (NV16, RGB) => Self::convert_nv16_to_rgb(src, dst),
            (NV16, RGBA) => Self::convert_nv16_to_rgba(src, dst),
            (PLANAR_RGB, RGB) => Self::convert_8bps_to_rgb(src, dst),
            (PLANAR_RGB, RGBA) => Self::convert_8bps_to_rgba(src, dst),
            (PLANAR_RGBA, RGB) => Self::convert_prgba_to_rgb(src, dst),
            (PLANAR_RGBA, RGBA) => Self::convert_prgba_to_rgba(src, dst),
            (s, d) => Err(Error::NotSupported(format!(
                "Conversion from {} to {}",
                s.display(),
                d.display()
            ))),
        }
    }

    /// The src and dest img should be in RGB/RGBA/grey format for correct
    /// output. If the format is not 1, 3, or 4 bits per pixel, and error will
    /// be returned. The src and dest img must have the same fourcc,
    /// otherwise the function will panic.
    fn resize_flip_rotate(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        let _timer = FunctionTimer::new(format!(
            "ImageConverter::resize_flip_rotate {}x{} to {}x{} {}",
            src.width(),
            src.height(),
            dst.width(),
            dst.height(),
            dst.fourcc().display()
        ));
        assert_eq!(src.fourcc(), dst.fourcc());

        let src_type = match src.channels() {
            1 => fast_image_resize::PixelType::U8,
            3 => fast_image_resize::PixelType::U8x3,
            4 => fast_image_resize::PixelType::U8x4,
            _ => {
                return Err(Error::NotImplemented(
                    "Unsupported source image format".to_string(),
                ));
            }
        };

        let mut src_map = src.tensor().map()?;

        let mut dst_map = dst.tensor().map()?;

        let options = if let Some(crop) = crop.src_rect {
            self.options.crop(
                crop.left as f64 / src.width() as f64,
                crop.top as f64 / src.height() as f64,
                crop.width as f64 / src.width() as f64,
                crop.height as f64 / src.height() as f64,
            )
        } else {
            self.options
        };

        let mut dst_rect = crop.dst_rect.unwrap_or_else(|| Rect {
            left: 0,
            top: 0,
            width: dst.width(),
            height: dst.height(),
        });

        // adjust crop box for rotation/flip
        Self::adjust_dest_rect_for_rotate_flip(&mut dst_rect, dst, rotation, flip);

        let needs_resize = src.width() != dst.width()
            || src.height() != dst.height()
            || crop.src_rect.is_some_and(|crop| {
                crop != Rect {
                    left: 0,
                    top: 0,
                    width: src.width(),
                    height: src.height(),
                }
            })
            || crop.dst_rect.is_some_and(|crop| {
                crop != Rect {
                    left: 0,
                    top: 0,
                    width: dst.width(),
                    height: dst.height(),
                }
            });

        if needs_resize {
            let src_view = fast_image_resize::images::Image::from_slice_u8(
                src.width() as u32,
                src.height() as u32,
                &mut src_map,
                src_type,
            )?;

            match (rotation, flip) {
                (Rotation::None, Flip::None) => {
                    let mut dst_view = fast_image_resize::images::Image::from_slice_u8(
                        dst.width() as u32,
                        dst.height() as u32,
                        &mut dst_map,
                        src_type,
                    )?;

                    let mut dst_view = fast_image_resize::images::CroppedImageMut::new(
                        &mut dst_view,
                        dst_rect.left as u32,
                        dst_rect.top as u32,
                        dst_rect.width as u32,
                        dst_rect.height as u32,
                    )?;

                    self.resizer.resize(&src_view, &mut dst_view, &options)?;
                }
                (Rotation::Clockwise90, _) | (Rotation::CounterClockwise90, _) => {
                    let mut tmp = vec![0; dst.row_stride() * dst.height()];
                    let mut tmp_view = fast_image_resize::images::Image::from_slice_u8(
                        dst.height() as u32,
                        dst.width() as u32,
                        &mut tmp,
                        src_type,
                    )?;

                    let mut tmp_view = fast_image_resize::images::CroppedImageMut::new(
                        &mut tmp_view,
                        dst_rect.left as u32,
                        dst_rect.top as u32,
                        dst_rect.width as u32,
                        dst_rect.height as u32,
                    )?;

                    self.resizer.resize(&src_view, &mut tmp_view, &options)?;
                    Self::flip_rotate_ndarray(&tmp, &mut dst_map, dst, rotation, flip)?;
                }
                (Rotation::None, _) | (Rotation::Rotate180, _) => {
                    let mut tmp = vec![0; dst.row_stride() * dst.height()];
                    let mut tmp_view = fast_image_resize::images::Image::from_slice_u8(
                        dst.width() as u32,
                        dst.height() as u32,
                        &mut tmp,
                        src_type,
                    )?;

                    let mut tmp_view = fast_image_resize::images::CroppedImageMut::new(
                        &mut tmp_view,
                        dst_rect.left as u32,
                        dst_rect.top as u32,
                        dst_rect.width as u32,
                        dst_rect.height as u32,
                    )?;

                    self.resizer.resize(&src_view, &mut tmp_view, &options)?;
                    Self::flip_rotate_ndarray(&tmp, &mut dst_map, dst, rotation, flip)?;
                }
            }
        } else {
            Self::flip_rotate_ndarray(&src_map, &mut dst_map, dst, rotation, flip)?;
        }
        Ok(())
    }

    fn adjust_dest_rect_for_rotate_flip(
        crop: &mut Rect,
        dst: &TensorImage,
        rot: Rotation,
        flip: Flip,
    ) {
        match rot {
            Rotation::None => {}
            Rotation::Clockwise90 => {
                *crop = Rect {
                    left: crop.top,
                    top: dst.width() - crop.left - crop.width,
                    width: crop.height,
                    height: crop.width,
                }
            }
            Rotation::Rotate180 => {
                *crop = Rect {
                    left: dst.width() - crop.left - crop.width,
                    top: dst.height() - crop.top - crop.height,
                    width: crop.width,
                    height: crop.height,
                }
            }
            Rotation::CounterClockwise90 => {
                *crop = Rect {
                    left: dst.height() - crop.top - crop.height,
                    top: crop.left,
                    width: crop.height,
                    height: crop.width,
                }
            }
        }

        match flip {
            Flip::None => {}
            Flip::Vertical => crop.top = dst.height() - crop.top - crop.height,
            Flip::Horizontal => crop.left = dst.width() - crop.left - crop.width,
        }
    }

    pub(crate) fn fill_image_outside_crop(
        dst: &mut TensorImage,
        rgba: [u8; 4],
        crop: Rect,
    ) -> Result<()> {
        let dst_fourcc = dst.fourcc();
        let mut dst_map = dst.tensor().map()?;
        let dst = (dst_map.as_mut_slice(), dst.width(), dst.height());
        match dst_fourcc {
            RGBA => Self::fill_image_outside_crop_(dst, rgba, crop),
            RGB => Self::fill_image_outside_crop_(dst, Self::rgba_to_rgb(rgba), crop),
            GREY => Self::fill_image_outside_crop_(dst, Self::rgba_to_grey(rgba), crop),
            YUYV => Self::fill_image_outside_crop_(
                (dst.0, dst.1 / 2, dst.2),
                Self::rgba_to_yuyv(rgba),
                Rect::new(crop.left / 2, crop.top, crop.width.div_ceil(2), crop.height),
            ),
            PLANAR_RGB => Self::fill_image_outside_crop_planar(dst, Self::rgba_to_rgb(rgba), crop),
            PLANAR_RGBA => Self::fill_image_outside_crop_planar(dst, rgba, crop),
            NV16 => {
                let yuyv = Self::rgba_to_yuyv(rgba);
                Self::fill_image_outside_crop_yuv_semiplanar(dst, yuyv[0], [yuyv[1], yuyv[3]], crop)
            }
            _ => Err(Error::Internal(format!(
                "Found unexpected destination {}",
                dst_fourcc.display()
            ))),
        }
    }

    fn fill_image_outside_crop_<const N: usize>(
        (dst, dst_width, _dst_height): (&mut [u8], usize, usize),
        pix: [u8; N],
        crop: Rect,
    ) -> Result<()> {
        use rayon::{
            iter::{IntoParallelRefMutIterator, ParallelIterator},
            prelude::ParallelSliceMut,
        };

        let s = dst.as_chunks_mut::<N>().0;
        // calculate the top/bottom
        let top_offset = (0, (crop.top * dst_width + crop.left));
        let bottom_offset = (
            ((crop.top + crop.height) * dst_width + crop.left).min(s.len()),
            s.len(),
        );

        s[top_offset.0..top_offset.1]
            .par_iter_mut()
            .for_each(|x| *x = pix);

        s[bottom_offset.0..bottom_offset.1]
            .par_iter_mut()
            .for_each(|x| *x = pix);

        if dst_width == crop.width {
            return Ok(());
        }

        // the middle part has a stride as well
        let middle_stride = dst_width - crop.width;
        let middle_offset = (
            (crop.top * dst_width + crop.left + crop.width),
            ((crop.top + crop.height) * dst_width + crop.left + crop.width).min(s.len()),
        );

        s[middle_offset.0..middle_offset.1]
            .par_chunks_exact_mut(dst_width)
            .for_each(|row| {
                for p in &mut row[0..middle_stride] {
                    *p = pix;
                }
            });

        Ok(())
    }

    fn fill_image_outside_crop_planar<const N: usize>(
        (dst, dst_width, dst_height): (&mut [u8], usize, usize),
        pix: [u8; N],
        crop: Rect,
    ) -> Result<()> {
        use rayon::{
            iter::{IntoParallelRefMutIterator, ParallelIterator},
            prelude::ParallelSliceMut,
        };

        // map.as_mut_slice().splitn_mut(n, pred)
        let s_rem = dst;

        s_rem
            .par_chunks_exact_mut(dst_height * dst_width)
            .zip(pix)
            .for_each(|(s, p)| {
                let top_offset = (0, (crop.top * dst_width + crop.left));
                let bottom_offset = (
                    ((crop.top + crop.height) * dst_width + crop.left).min(s.len()),
                    s.len(),
                );

                s[top_offset.0..top_offset.1]
                    .par_iter_mut()
                    .for_each(|x| *x = p);

                s[bottom_offset.0..bottom_offset.1]
                    .par_iter_mut()
                    .for_each(|x| *x = p);

                if dst_width == crop.width {
                    return;
                }

                // the middle part has a stride as well
                let middle_stride = dst_width - crop.width;
                let middle_offset = (
                    (crop.top * dst_width + crop.left + crop.width),
                    ((crop.top + crop.height) * dst_width + crop.left + crop.width).min(s.len()),
                );

                s[middle_offset.0..middle_offset.1]
                    .par_chunks_exact_mut(dst_width)
                    .for_each(|row| {
                        for x in &mut row[0..middle_stride] {
                            *x = p;
                        }
                    });
            });
        Ok(())
    }

    fn fill_image_outside_crop_yuv_semiplanar(
        (dst, dst_width, dst_height): (&mut [u8], usize, usize),
        y: u8,
        uv: [u8; 2],
        mut crop: Rect,
    ) -> Result<()> {
        let (y_plane, uv_plane) = dst.split_at_mut(dst_width * dst_height);
        Self::fill_image_outside_crop_::<1>((y_plane, dst_width, dst_height), [y], crop)?;
        crop.left /= 2;
        crop.width /= 2;
        Self::fill_image_outside_crop_::<2>((uv_plane, dst_width / 2, dst_height), uv, crop)?;
        Ok(())
    }

    fn rgba_to_rgb(rgba: [u8; 4]) -> [u8; 3] {
        let [r, g, b, _] = rgba;
        [r, g, b]
    }

    fn rgba_to_grey(rgba: [u8; 4]) -> [u8; 1] {
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

    fn rgba_to_yuyv(rgba: [u8; 4]) -> [u8; 4] {
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

impl ImageConverterTrait for CPUConverter {
    fn convert(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        crop.check_crop(src, dst)?;
        // supported destinations and srcs:
        let intermediate = match (src.fourcc(), dst.fourcc()) {
            (NV12, RGB) => RGB,
            (NV12, RGBA) => RGBA,
            (NV12, GREY) => GREY,
            (NV12, YUYV) => RGBA, // RGBA intermediary for YUYV dest resize/convert/rotation/flip
            (NV12, NV16) => RGBA, // RGBA intermediary for YUYV dest resize/convert/rotation/flip
            (NV12, PLANAR_RGB) => RGB,
            (NV12, PLANAR_RGBA) => RGBA,
            (YUYV, RGB) => RGB,
            (YUYV, RGBA) => RGBA,
            (YUYV, GREY) => GREY,
            (YUYV, YUYV) => RGBA, // RGBA intermediary for YUYV dest resize/convert/rotation/flip
            (YUYV, PLANAR_RGB) => RGB,
            (YUYV, PLANAR_RGBA) => RGBA,
            (YUYV, NV16) => RGBA,
            (RGBA, RGB) => RGBA,
            (RGBA, RGBA) => RGBA,
            (RGBA, GREY) => GREY,
            (RGBA, YUYV) => RGBA, // RGBA intermediary for YUYV dest resize/convert/rotation/flip
            (RGBA, PLANAR_RGB) => RGBA,
            (RGBA, PLANAR_RGBA) => RGBA,
            (RGBA, NV16) => RGBA,
            (RGB, RGB) => RGB,
            (RGB, RGBA) => RGB,
            (RGB, GREY) => GREY,
            (RGB, YUYV) => RGB, // RGB intermediary for YUYV dest resize/convert/rotation/flip
            (RGB, PLANAR_RGB) => RGB,
            (RGB, PLANAR_RGBA) => RGB,
            (RGB, NV16) => RGB,
            (GREY, RGB) => RGB,
            (GREY, RGBA) => RGBA,
            (GREY, GREY) => GREY,
            (GREY, YUYV) => GREY,
            (GREY, PLANAR_RGB) => GREY,
            (GREY, PLANAR_RGBA) => GREY,
            (GREY, NV16) => GREY,
            (s, d) => {
                return Err(Error::NotSupported(format!(
                    "Conversion from {} to {}",
                    s.display(),
                    d.display()
                )));
            }
        };

        // let crop = crop.src_rect;

        let need_resize_flip_rotation = rotation != Rotation::None
            || flip != Flip::None
            || src.width() != dst.width()
            || src.height() != dst.height()
            || crop.src_rect.is_some_and(|crop| {
                crop != Rect {
                    left: 0,
                    top: 0,
                    width: src.width(),
                    height: src.height(),
                }
            })
            || crop.dst_rect.is_some_and(|crop| {
                crop != Rect {
                    left: 0,
                    top: 0,
                    width: dst.width(),
                    height: dst.height(),
                }
            });

        // check if a direct conversion can be done
        if !need_resize_flip_rotation && Self::support_conversion(src.fourcc(), dst.fourcc()) {
            return Self::convert_format(src, dst);
        };

        // any extra checks
        if dst.fourcc() == YUYV && !dst.width().is_multiple_of(2) {
            return Err(Error::NotSupported(format!(
                "{} destination must have width divisible by 2",
                dst.fourcc().display(),
            )));
        }

        // create tmp buffer
        let mut tmp_buffer;
        let tmp;
        if intermediate != src.fourcc() {
            tmp_buffer = TensorImage::new(
                src.width(),
                src.height(),
                intermediate,
                Some(edgefirst_tensor::TensorMemory::Mem),
            )?;

            Self::convert_format(src, &mut tmp_buffer)?;
            tmp = &tmp_buffer;
        } else {
            tmp = src;
        }

        // format must be RGB/RGBA/GREY
        matches!(tmp.fourcc(), RGB | RGBA | GREY);
        if tmp.fourcc() == dst.fourcc() {
            self.resize_flip_rotate(tmp, dst, rotation, flip, crop)?;
        } else if !need_resize_flip_rotation {
            Self::convert_format(tmp, dst)?;
        } else {
            let mut tmp2 = TensorImage::new(
                dst.width(),
                dst.height(),
                tmp.fourcc(),
                Some(edgefirst_tensor::TensorMemory::Mem),
            )?;
            if crop.dst_rect.is_some_and(|crop| {
                crop != Rect {
                    left: 0,
                    top: 0,
                    width: dst.width(),
                    height: dst.height(),
                }
            }) && crop.dst_color.is_none()
            {
                // convert the dst into tmp2 when there is a dst crop
                // TODO: this could be optimized by changing convert_format to take a
                // destination crop?

                Self::convert_format(dst, &mut tmp2)?;
            }
            self.resize_flip_rotate(tmp, &mut tmp2, rotation, flip, crop)?;
            Self::convert_format(&tmp2, dst)?;
        }
        if let Some(dst_rect) = crop.dst_rect
            && dst_rect
                != (Rect {
                    left: 0,
                    top: 0,
                    width: dst.width(),
                    height: dst.height(),
                })
            && let Some(dst_color) = crop.dst_color
        {
            Self::fill_image_outside_crop(dst, dst_color, dst_rect)?;
        }

        Ok(())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod cpu_tests {

    use super::*;
    use crate::{CPUConverter, Rotation};
    use edgefirst_tensor::{TensorMapTrait, TensorMemory};
    use g2d_sys::RGBA;
    use image::buffer::ConvertBuffer;

    macro_rules! function {
        () => {{
            fn f() {}
            fn type_name_of<T>(_: T) -> &'static str {
                std::any::type_name::<T>()
            }
            let name = type_name_of(f);

            // Find and cut the rest of the path
            match &name[..name.len() - 3].rfind(':') {
                Some(pos) => &name[pos + 1..name.len() - 3],
                None => &name[..name.len() - 3],
            }
        }};
    }

    fn compare_images_convert_to_grey(
        img1: &TensorImage,
        img2: &TensorImage,
        threshold: f64,
        name: &str,
    ) {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");

        let mut img_rgb1 = TensorImage::new(img1.width(), img1.height(), RGBA, None).unwrap();
        let mut img_rgb2 = TensorImage::new(img1.width(), img1.height(), RGBA, None).unwrap();
        CPUConverter::convert_format(img1, &mut img_rgb1).unwrap();
        CPUConverter::convert_format(img2, &mut img_rgb2).unwrap();

        let image1 = image::RgbaImage::from_vec(
            img_rgb1.width() as u32,
            img_rgb1.height() as u32,
            img_rgb1.tensor().map().unwrap().to_vec(),
        )
        .unwrap();

        let image2 = image::RgbaImage::from_vec(
            img_rgb2.width() as u32,
            img_rgb2.height() as u32,
            img_rgb2.tensor().map().unwrap().to_vec(),
        )
        .unwrap();

        let similarity = image_compare::gray_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &image1.convert(),
            &image2.convert(),
        )
        .expect("Image Comparison failed");
        if similarity.score < threshold {
            // image1.save(format!("{name}_1.png"));
            // image2.save(format!("{name}_2.png"));
            similarity
                .image
                .to_color_map()
                .save(format!("{name}.png"))
                .unwrap();
            panic!(
                "{name}: converted image and target image have similarity score too low: {} < {}",
                similarity.score, threshold
            )
        }
    }

    fn compare_images_convert_to_rgb(
        img1: &TensorImage,
        img2: &TensorImage,
        threshold: f64,
        name: &str,
    ) {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");

        let mut img_rgb1 = TensorImage::new(img1.width(), img1.height(), RGB, None).unwrap();
        let mut img_rgb2 = TensorImage::new(img1.width(), img1.height(), RGB, None).unwrap();
        CPUConverter::convert_format(img1, &mut img_rgb1).unwrap();
        CPUConverter::convert_format(img2, &mut img_rgb2).unwrap();

        let image1 = image::RgbImage::from_vec(
            img_rgb1.width() as u32,
            img_rgb1.height() as u32,
            img_rgb1.tensor().map().unwrap().to_vec(),
        )
        .unwrap();

        let image2 = image::RgbImage::from_vec(
            img_rgb2.width() as u32,
            img_rgb2.height() as u32,
            img_rgb2.tensor().map().unwrap().to_vec(),
        )
        .unwrap();

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &image1,
            &image2,
        )
        .expect("Image Comparison failed");
        if similarity.score < threshold {
            // image1.save(format!("{name}_1.png"));
            // image2.save(format!("{name}_2.png"));
            similarity
                .image
                .to_color_map()
                .save(format!("{name}.png"))
                .unwrap();
            panic!(
                "{name}: converted image and target image have similarity score too low: {} < {}",
                similarity.score, threshold
            )
        }
    }

    fn load_bytes_to_tensor(
        width: usize,
        height: usize,
        fourcc: FourCharCode,
        memory: Option<TensorMemory>,
        bytes: &[u8],
    ) -> Result<TensorImage, Error> {
        log::debug!("Current function is {}", function!());
        let src = TensorImage::new(width, height, fourcc, memory)?;
        src.tensor().map()?.as_mut_slice()[0..bytes.len()].copy_from_slice(bytes);
        Ok(src)
    }

    macro_rules! generate_conversion_tests {
        (
        $src_fmt:ident,  $src_file:expr, $dst_fmt:ident, $dst_file:expr
    ) => {{
            // Load source
            let src = load_bytes_to_tensor(
                1280,
                720,
                $src_fmt,
                None,
                include_bytes!(concat!("../../../testdata/", $src_file)),
            )?;

            // Load destination reference
            let dst = load_bytes_to_tensor(
                1280,
                720,
                $dst_fmt,
                None,
                include_bytes!(concat!("../../../testdata/", $dst_file)),
            )?;

            let mut converter = CPUConverter::default();

            let mut converted = TensorImage::new(src.width(), src.height(), dst.fourcc(), None)?;

            converter.convert(
                &src,
                &mut converted,
                Rotation::None,
                Flip::None,
                Crop::default(),
            )?;

            compare_images_convert_to_rgb(&dst, &converted, 0.99, function!());

            Ok(())
        }};
    }

    macro_rules! generate_conversion_tests_greyscale {
        (
        $src_fmt:ident,  $src_file:expr, $dst_fmt:ident, $dst_file:expr
    ) => {{
            // Load source
            let src = load_bytes_to_tensor(
                1280,
                720,
                $src_fmt,
                None,
                include_bytes!(concat!("../../../testdata/", $src_file)),
            )?;

            // Load destination reference
            let dst = load_bytes_to_tensor(
                1280,
                720,
                $dst_fmt,
                None,
                include_bytes!(concat!("../../../testdata/", $dst_file)),
            )?;

            let mut converter = CPUConverter::default();

            let mut converted = TensorImage::new(src.width(), src.height(), dst.fourcc(), None)?;

            converter.convert(
                &src,
                &mut converted,
                Rotation::None,
                Flip::None,
                Crop::default(),
            )?;

            compare_images_convert_to_grey(&dst, &converted, 0.99, function!());

            Ok(())
        }};
    }

    // let mut dsts = [yuyv, rgb, rgba, grey, nv16, planar_rgb, planar_rgba];

    #[test]
    fn test_cpu_yuyv_to_yuyv() -> Result<()> {
        generate_conversion_tests!(YUYV, "camera720p.yuyv", YUYV, "camera720p.yuyv")
    }

    #[test]
    fn test_cpu_yuyv_to_rgb() -> Result<()> {
        generate_conversion_tests!(YUYV, "camera720p.yuyv", RGB, "camera720p.rgb")
    }

    #[test]
    fn test_cpu_yuyv_to_rgba() -> Result<()> {
        generate_conversion_tests!(YUYV, "camera720p.yuyv", RGBA, "camera720p.rgba")
    }

    #[test]
    fn test_cpu_yuyv_to_grey() -> Result<()> {
        generate_conversion_tests!(YUYV, "camera720p.yuyv", GREY, "camera720p.y800")
    }

    #[test]
    fn test_cpu_yuyv_to_nv16() -> Result<()> {
        generate_conversion_tests!(YUYV, "camera720p.yuyv", NV16, "camera720p.nv16")
    }

    #[test]
    fn test_cpu_yuyv_to_planar_rgb() -> Result<()> {
        generate_conversion_tests!(YUYV, "camera720p.yuyv", PLANAR_RGB, "camera720p.8bps")
    }

    #[test]
    fn test_cpu_yuyv_to_planar_rgba() -> Result<()> {
        generate_conversion_tests!(YUYV, "camera720p.yuyv", PLANAR_RGBA, "camera720p.8bpa")
    }

    #[test]
    fn test_cpu_rgb_to_yuyv() -> Result<()> {
        generate_conversion_tests!(RGB, "camera720p.rgb", YUYV, "camera720p.yuyv")
    }

    #[test]
    fn test_cpu_rgb_to_rgb() -> Result<()> {
        generate_conversion_tests!(RGB, "camera720p.rgb", RGB, "camera720p.rgb")
    }

    #[test]
    fn test_cpu_rgb_to_rgba() -> Result<()> {
        generate_conversion_tests!(RGB, "camera720p.rgb", RGBA, "camera720p.rgba")
    }

    #[test]
    fn test_cpu_rgb_to_grey() -> Result<()> {
        generate_conversion_tests!(RGB, "camera720p.rgb", GREY, "camera720p.y800")
    }

    #[test]
    fn test_cpu_rgb_to_nv16() -> Result<()> {
        generate_conversion_tests!(RGB, "camera720p.rgb", NV16, "camera720p.nv16")
    }

    #[test]
    fn test_cpu_rgb_to_planar_rgb() -> Result<()> {
        generate_conversion_tests!(RGB, "camera720p.rgb", PLANAR_RGB, "camera720p.8bps")
    }

    #[test]
    fn test_cpu_rgb_to_planar_rgba() -> Result<()> {
        generate_conversion_tests!(RGB, "camera720p.rgb", PLANAR_RGBA, "camera720p.8bpa")
    }

    #[test]
    fn test_cpu_rgba_to_yuyv() -> Result<()> {
        generate_conversion_tests!(RGBA, "camera720p.rgba", YUYV, "camera720p.yuyv")
    }

    #[test]
    fn test_cpu_rgba_to_rgb() -> Result<()> {
        generate_conversion_tests!(RGBA, "camera720p.rgba", RGB, "camera720p.rgb")
    }

    #[test]
    fn test_cpu_rgba_to_rgba() -> Result<()> {
        generate_conversion_tests!(RGBA, "camera720p.rgba", RGBA, "camera720p.rgba")
    }

    #[test]
    fn test_cpu_rgba_to_grey() -> Result<()> {
        generate_conversion_tests!(RGBA, "camera720p.rgba", GREY, "camera720p.y800")
    }

    #[test]
    fn test_cpu_rgba_to_nv16() -> Result<()> {
        generate_conversion_tests!(RGBA, "camera720p.rgba", NV16, "camera720p.nv16")
    }

    #[test]
    fn test_cpu_rgba_to_planar_rgb() -> Result<()> {
        generate_conversion_tests!(RGBA, "camera720p.rgba", PLANAR_RGB, "camera720p.8bps")
    }

    #[test]
    fn test_cpu_rgba_to_planar_rgba() -> Result<()> {
        generate_conversion_tests!(RGBA, "camera720p.rgba", PLANAR_RGBA, "camera720p.8bpa")
    }

    #[test]
    fn test_cpu_nv12_to_rgb() -> Result<()> {
        generate_conversion_tests!(NV12, "camera720p.nv12", RGB, "camera720p.rgb")
    }

    #[test]
    fn test_cpu_nv12_to_yuyv() -> Result<()> {
        generate_conversion_tests!(NV12, "camera720p.nv12", YUYV, "camera720p.yuyv")
    }

    #[test]
    fn test_cpu_nv12_to_rgba() -> Result<()> {
        generate_conversion_tests!(NV12, "camera720p.nv12", RGBA, "camera720p.rgba")
    }

    #[test]
    fn test_cpu_nv12_to_grey() -> Result<()> {
        generate_conversion_tests!(NV12, "camera720p.nv12", GREY, "camera720p.y800")
    }

    #[test]
    fn test_cpu_nv12_to_nv16() -> Result<()> {
        generate_conversion_tests!(NV12, "camera720p.nv12", NV16, "camera720p.nv16")
    }

    #[test]
    fn test_cpu_nv12_to_planar_rgb() -> Result<()> {
        generate_conversion_tests!(NV12, "camera720p.nv12", PLANAR_RGB, "camera720p.8bps")
    }

    #[test]
    fn test_cpu_nv12_to_planar_rgba() -> Result<()> {
        generate_conversion_tests!(NV12, "camera720p.nv12", PLANAR_RGBA, "camera720p.8bpa")
    }

    #[test]
    fn test_cpu_grey_to_yuyv() -> Result<()> {
        generate_conversion_tests_greyscale!(GREY, "camera720p.y800", YUYV, "camera720p.yuyv")
    }

    #[test]
    fn test_cpu_grey_to_rgb() -> Result<()> {
        generate_conversion_tests_greyscale!(GREY, "camera720p.y800", RGB, "camera720p.rgb")
    }

    #[test]
    fn test_cpu_grey_to_rgba() -> Result<()> {
        generate_conversion_tests_greyscale!(GREY, "camera720p.y800", RGBA, "camera720p.rgba")
    }

    #[test]
    fn test_cpu_grey_to_grey() -> Result<()> {
        generate_conversion_tests_greyscale!(GREY, "camera720p.y800", GREY, "camera720p.y800")
    }

    #[test]
    fn test_cpu_grey_to_nv16() -> Result<()> {
        generate_conversion_tests_greyscale!(GREY, "camera720p.y800", NV16, "camera720p.nv16")
    }

    #[test]
    fn test_cpu_grey_to_planar_rgb() -> Result<()> {
        generate_conversion_tests_greyscale!(GREY, "camera720p.y800", PLANAR_RGB, "camera720p.8bps")
    }

    #[test]
    fn test_cpu_grey_to_planar_rgba() -> Result<()> {
        generate_conversion_tests_greyscale!(
            GREY,
            "camera720p.y800",
            PLANAR_RGBA,
            "camera720p.8bpa"
        )
    }

    #[test]
    fn test_cpu_nearest() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(2, 1, RGB, None, &[0, 0, 0, 255, 255, 255])?;

        let mut converter = CPUConverter::new_nearest();

        let mut converted = TensorImage::new(4, 1, RGB, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::None,
            Flip::None,
            Crop::default(),
        )?;

        assert_eq!(
            &converted.tensor().map()?.as_slice(),
            &[0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 255, 255]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_rotate_cw() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(
            2,
            2,
            RGBA,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUConverter::default();

        let mut converted = TensorImage::new(4, 4, RGBA, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::Clockwise90,
            Flip::None,
            Crop::default(),
        )?;

        assert_eq!(&converted.tensor().map()?.as_slice()[0..4], &[2, 2, 2, 255]);
        assert_eq!(
            &converted.tensor().map()?.as_slice()[12..16],
            &[0, 0, 0, 255]
        );
        assert_eq!(
            &converted.tensor().map()?.as_slice()[48..52],
            &[3, 3, 3, 255]
        );

        assert_eq!(
            &converted.tensor().map()?.as_slice()[60..64],
            &[1, 1, 1, 255]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_rotate_ccw() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(
            2,
            2,
            RGBA,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUConverter::default();

        let mut converted = TensorImage::new(4, 4, RGBA, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::CounterClockwise90,
            Flip::None,
            Crop::default(),
        )?;

        assert_eq!(&converted.tensor().map()?.as_slice()[0..4], &[1, 1, 1, 255]);
        assert_eq!(
            &converted.tensor().map()?.as_slice()[12..16],
            &[3, 3, 3, 255]
        );
        assert_eq!(
            &converted.tensor().map()?.as_slice()[48..52],
            &[0, 0, 0, 255]
        );

        assert_eq!(
            &converted.tensor().map()?.as_slice()[60..64],
            &[2, 2, 2, 255]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_rotate_180() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(
            2,
            2,
            RGBA,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUConverter::default();

        let mut converted = TensorImage::new(4, 4, RGBA, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::Rotate180,
            Flip::None,
            Crop::default(),
        )?;

        assert_eq!(&converted.tensor().map()?.as_slice()[0..4], &[3, 3, 3, 255]);
        assert_eq!(
            &converted.tensor().map()?.as_slice()[12..16],
            &[2, 2, 2, 255]
        );
        assert_eq!(
            &converted.tensor().map()?.as_slice()[48..52],
            &[1, 1, 1, 255]
        );

        assert_eq!(
            &converted.tensor().map()?.as_slice()[60..64],
            &[0, 0, 0, 255]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_flip_v() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(
            2,
            2,
            RGBA,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUConverter::default();

        let mut converted = TensorImage::new(4, 4, RGBA, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::None,
            Flip::Vertical,
            Crop::default(),
        )?;

        assert_eq!(&converted.tensor().map()?.as_slice()[0..4], &[2, 2, 2, 255]);
        assert_eq!(
            &converted.tensor().map()?.as_slice()[12..16],
            &[3, 3, 3, 255]
        );
        assert_eq!(
            &converted.tensor().map()?.as_slice()[48..52],
            &[0, 0, 0, 255]
        );

        assert_eq!(
            &converted.tensor().map()?.as_slice()[60..64],
            &[1, 1, 1, 255]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_flip_h() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(
            2,
            2,
            RGBA,
            None,
            &[0, 0, 0, 255, 1, 1, 1, 255, 2, 2, 2, 255, 3, 3, 3, 255],
        )?;

        let mut converter = CPUConverter::default();

        let mut converted = TensorImage::new(4, 4, RGBA, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::None,
            Flip::Horizontal,
            Crop::default(),
        )?;

        assert_eq!(&converted.tensor().map()?.as_slice()[0..4], &[1, 1, 1, 255]);
        assert_eq!(
            &converted.tensor().map()?.as_slice()[12..16],
            &[0, 0, 0, 255]
        );
        assert_eq!(
            &converted.tensor().map()?.as_slice()[48..52],
            &[3, 3, 3, 255]
        );

        assert_eq!(
            &converted.tensor().map()?.as_slice()[60..64],
            &[2, 2, 2, 255]
        );

        Ok(())
    }

    #[test]
    fn test_cpu_src_crop() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(2, 2, GREY, None, &[1, 2, 3, 4])?;

        let mut converter = CPUConverter::default();

        let mut converted = TensorImage::new(2, 2, RGBA, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::None,
            Flip::None,
            Crop::new().with_src_rect(Some(Rect::new(0, 0, 1, 1))),
        )?;

        assert_eq!(
            converted.tensor().map()?.as_slice(),
            &[1, 1, 1, 255, 1, 1, 1, 255, 1, 1, 1, 255, 1, 1, 1, 255]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_dst_crop() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(2, 2, GREY, None, &[2, 4, 6, 8])?;

        let mut converter = CPUConverter::default();

        let mut converted =
            load_bytes_to_tensor(2, 2, YUYV, None, &[200, 128, 200, 128, 200, 128, 200, 128])?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::None,
            Flip::None,
            Crop::new().with_dst_rect(Some(Rect::new(0, 0, 2, 1))),
        )?;

        assert_eq!(
            converted.tensor().map()?.as_slice(),
            &[20, 128, 21, 128, 200, 128, 200, 128]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_fill_rgba() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(1, 1, RGBA, None, &[3, 3, 3, 255])?;

        let mut converter = CPUConverter::default();

        let mut converted = TensorImage::new(2, 2, RGBA, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::None,
            Flip::None,
            Crop {
                src_rect: None,
                dst_rect: Some(Rect {
                    left: 1,
                    top: 1,
                    width: 1,
                    height: 1,
                }),
                dst_color: Some([255, 0, 0, 255]),
            },
        )?;

        assert_eq!(
            converted.tensor().map()?.as_slice(),
            &[255, 0, 0, 255, 255, 0, 0, 255, 255, 0, 0, 255, 3, 3, 3, 255]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_fill_yuyv() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(2, 1, RGBA, None, &[3, 3, 3, 255, 3, 3, 3, 255])?;

        let mut converter = CPUConverter::default();

        let mut converted = TensorImage::new(2, 3, YUYV, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::None,
            Flip::None,
            Crop {
                src_rect: None,
                dst_rect: Some(Rect {
                    left: 0,
                    top: 1,
                    width: 2,
                    height: 1,
                }),
                dst_color: Some([255, 0, 0, 255]),
            },
        )?;

        assert_eq!(
            converted.tensor().map()?.as_slice(),
            &[63, 102, 63, 240, 19, 128, 19, 128, 63, 102, 63, 240]
        );
        Ok(())
    }

    #[test]
    fn test_cpu_fill_grey() -> Result<()> {
        // Load source
        let src = load_bytes_to_tensor(2, 1, RGBA, None, &[3, 3, 3, 255, 3, 3, 3, 255])?;

        let mut converter = CPUConverter::default();

        let mut converted = TensorImage::new(2, 3, GREY, None)?;

        converter.convert(
            &src,
            &mut converted,
            Rotation::None,
            Flip::None,
            Crop {
                src_rect: None,
                dst_rect: Some(Rect {
                    left: 0,
                    top: 1,
                    width: 2,
                    height: 1,
                }),
                dst_color: Some([200, 200, 200, 255]),
            },
        )?;

        assert_eq!(
            converted.tensor().map()?.as_slice(),
            &[200, 200, 3, 3, 200, 200]
        );
        Ok(())
    }
}
