// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    Crop, Error, Flip, FunctionTimer, Rect, Result, Rotation, TensorImage, TensorImageDst, GREY,
    NV16, PLANAR_RGB, PLANAR_RGBA, RGB, RGBA, YUYV,
};
use edgefirst_tensor::{TensorMapTrait, TensorTrait};
use ndarray::{ArrayView3, ArrayViewMut3, Axis};
use rayon::iter::IndexedParallelIterator;

use super::CPUProcessor;

impl CPUProcessor {
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

    pub(super) fn resize_flip_rotate(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        let _timer = FunctionTimer::new(format!(
            "ImageProcessor::resize_flip_rotate {}x{} to {}x{} {}",
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
                crop.left as f64,
                crop.top as f64,
                crop.width as f64,
                crop.height as f64,
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

    pub(super) fn adjust_dest_rect_for_rotate_flip(
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

    /// Fills the area outside a crop rectangle with the specified color.
    pub fn fill_image_outside_crop(dst: &mut TensorImage, rgba: [u8; 4], crop: Rect) -> Result<()> {
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

    /// Generic fill for TensorImageDst types.
    pub(crate) fn fill_image_outside_crop_generic<D: TensorImageDst>(
        dst: &mut D,
        rgba: [u8; 4],
        crop: Rect,
    ) -> Result<()> {
        let dst_fourcc = dst.fourcc();
        let dst_width = dst.width();
        let dst_height = dst.height();
        let mut dst_map = dst.tensor_mut().map()?;
        let dst = (dst_map.as_mut_slice(), dst_width, dst_height);
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

    pub(super) fn fill_image_outside_crop_<const N: usize>(
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

    pub(super) fn fill_image_outside_crop_planar<const N: usize>(
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

    pub(super) fn fill_image_outside_crop_yuv_semiplanar(
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
}
