use crate::{Error, ImageConverterTrait, Rect, Result, Rotation, TensorImage};
use edgefirst_tensor::TensorTrait;
use ndarray::{ArrayView3, ArrayViewMut3, Axis};

/// CPUConverter implements the ImageConverter trait using the fallback CPU
/// implementation for image processing.
pub struct CPUConverter {
    resizer: fast_image_resize::Resizer,
    options: fast_image_resize::ResizeOptions,
}

impl CPUConverter {
    pub fn new() -> Result<Self> {
        let resizer = fast_image_resize::Resizer::new();
        let options = fast_image_resize::ResizeOptions::new()
            .resize_alg(fast_image_resize::ResizeAlg::Convolution(
                fast_image_resize::FilterType::Hamming,
            ))
            .use_alpha(false);
        Ok(Self { resizer, options })
    }

    pub fn new_nearest() -> Result<Self> {
        let resizer = fast_image_resize::Resizer::new();
        let options = fast_image_resize::ResizeOptions::new()
            .resize_alg(fast_image_resize::ResizeAlg::Nearest)
            .use_alpha(false);
        Ok(Self { resizer, options })
    }

    fn rotate_ndarray(
        &self,
        src_map: &[u8],
        dst_map: &mut [u8],
        dst: &TensorImage,
        rotation: Rotation,
    ) -> Result<(), crate::Error> {
        match rotation {
            Rotation::None => {
                dst_map.copy_from_slice(src_map);
            }
            Rotation::Rotate90Clockwise => {
                let mut src_view =
                    ArrayView3::from_shape((dst.width(), dst.height(), dst.channels()), src_map)
                        .expect("rotate src shape incorrect");
                let mut dst_view =
                    ArrayViewMut3::from_shape((dst.height(), dst.width(), dst.channels()), dst_map)
                        .expect("rotate dst shape incorrect");
                src_view.swap_axes(0, 1);
                src_view.invert_axis(Axis(1));
                src_view
                    .iter()
                    .zip(dst_view.iter_mut())
                    .for_each(|(s, d)| *d = *s);
            }
            Rotation::Rotate180 => {
                let mut src_view =
                    ArrayView3::from_shape((dst.height(), dst.width(), dst.channels()), src_map)
                        .expect("rotate src shape incorrect");
                let mut dst_view =
                    ArrayViewMut3::from_shape((dst.height(), dst.width(), dst.channels()), dst_map)
                        .expect("rotate dst shape incorrect");
                src_view.invert_axis(Axis(0));
                src_view.invert_axis(Axis(1));
                src_view
                    .iter()
                    .zip(dst_view.iter_mut())
                    .for_each(|(s, d)| *d = *s);
            }
            Rotation::Rotate90CounterClockwise => {
                let mut src_view =
                    ArrayView3::from_shape((dst.width(), dst.height(), dst.channels()), src_map)
                        .expect("rotate src shape incorrect");
                let mut dst_view =
                    ArrayViewMut3::from_shape((dst.height(), dst.width(), dst.channels()), dst_map)
                        .expect("rotate dst shape incorrect");
                src_view.swap_axes(0, 1);
                src_view.invert_axis(Axis(0));
                src_view
                    .iter()
                    .zip(dst_view.iter_mut())
                    .for_each(|(s, d)| *d = *s);
            }
        }
        Ok(())
    }
}

impl ImageConverterTrait for CPUConverter {
    fn convert(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: Rotation,
        crop: Option<Rect>,
    ) -> Result<()> {
        // if rotation != Rotation::None {
        //     return Err(Error::NotImplemented(
        //         "Rotation is not supported in CPUConverter".to_string(),
        //     ));
        // }

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

        let dst_type = match dst.channels() {
            1 => fast_image_resize::PixelType::U8,
            3 => fast_image_resize::PixelType::U8x3,
            4 => fast_image_resize::PixelType::U8x4,
            _ => {
                return Err(Error::NotImplemented(
                    "Unsupported destination image format".to_string(),
                ));
            }
        };

        let mut src_map = src.tensor().map()?;

        let mut dst_map = dst.tensor().map()?;

        let options = if let Some(crop) = crop {
            self.options.crop(
                crop.left as f64,
                crop.top as f64,
                crop.width as f64,
                crop.height as f64,
            )
        } else {
            self.options
        };

        let needs_resize = src.width() != dst.width()
            || src.height() != dst.height()
            || crop.is_some_and(|crop| {
                crop != Rect {
                    left: 0,
                    top: 0,
                    width: src.width(),
                    height: src.height(),
                }
            });

        if needs_resize {
            let src_view = fast_image_resize::images::Image::from_slice_u8(
                src.width() as u32,
                src.height() as u32,
                &mut src_map,
                src_type,
            )?;
            match rotation {
                Rotation::None => {
                    let mut dst_view = fast_image_resize::images::Image::from_slice_u8(
                        dst.width() as u32,
                        dst.height() as u32,
                        &mut dst_map,
                        dst_type,
                    )?;
                    self.resizer.resize(&src_view, &mut dst_view, &options)?;
                }
                Rotation::Rotate90Clockwise | Rotation::Rotate90CounterClockwise => {
                    let mut tmp = vec![0; dst.row_stride() * dst.height()];
                    let mut tmp_view = fast_image_resize::images::Image::from_slice_u8(
                        dst.height() as u32,
                        dst.width() as u32,
                        &mut tmp,
                        dst_type,
                    )?;
                    self.resizer.resize(&src_view, &mut tmp_view, &options)?;
                    self.rotate_ndarray(&tmp, &mut dst_map, dst, rotation)?;
                }
                Rotation::Rotate180 => {
                    let mut tmp = vec![0; dst.row_stride() * dst.height()];
                    let mut tmp_view = fast_image_resize::images::Image::from_slice_u8(
                        dst.width() as u32,
                        dst.height() as u32,
                        &mut tmp,
                        dst_type,
                    )?;
                    self.resizer.resize(&src_view, &mut tmp_view, &options)?;
                    self.rotate_ndarray(&tmp, &mut dst_map, dst, rotation)?;
                }
            }
        } else {
            self.rotate_ndarray(&src_map, &mut dst_map, dst, rotation)?;
        }

        Ok(())
    }
}
