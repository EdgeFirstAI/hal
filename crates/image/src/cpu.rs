use crate::{
    Error, ImageConverterTrait, NV12, RGB, RGBA, Rect, Result, Rotation, TensorImage, YUYV,
};
use edgefirst_tensor::{TensorMapTrait, TensorTrait};
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

    fn convert_nv12_to_rgb(&self, src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), YUYV);
        assert_eq!(dst.fourcc(), RGB);
        // let src = yuv::YuvPackedImage::<u8> {
        //     yuy: &src.tensor.map()?,
        //     yuy_stride: src.width() as u32 * 2, // we assume packed yuyv
        //     width: src.width() as u32,
        //     height: src.height() as u32,
        // };
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
            dst.width() as u32 * 3,
            yuv::YuvRange::Limited,
            yuv::YuvStandardMatrix::Bt709,
            yuv::YuvConversionMode::Balanced,
        )?)
    }

    fn convert_nv12_to_rgba(&self, src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), YUYV);
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
            dst.width() as u32 * 4,
            yuv::YuvRange::Limited,
            yuv::YuvStandardMatrix::Bt709,
            yuv::YuvConversionMode::Balanced,
        )?)
    }

    fn convert_yuyv_to_rgb(&self, src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), YUYV);
        assert_eq!(dst.fourcc(), RGB);
        let src = yuv::YuvPackedImage::<u8> {
            yuy: &src.tensor.map()?,
            yuy_stride: src.width() as u32 * 2, // we assume packed yuyv
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

    fn convert_yuyv_to_rgba(&self, src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), YUYV);
        assert_eq!(dst.fourcc(), RGBA);
        let src = yuv::YuvPackedImage::<u8> {
            yuy: &src.tensor.map()?,
            yuy_stride: src.width() as u32 * 2, // we assume packed yuyv
            width: src.width() as u32,
            height: src.height() as u32,
        };

        Ok(yuv::yuyv422_to_rgba(
            &src,
            dst.tensor.map()?.as_mut_slice(),
            dst.width() as u32 * 4,
            yuv::YuvRange::Limited,
            yuv::YuvStandardMatrix::Bt709,
        )?)
    }

    fn convert_rgba_to_rgb(&self, src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_rgb_to_rgba(&self, src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn resize_and_rotate(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: Rotation,
        crop: Option<Rect>,
    ) -> Result<()> {
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
                        src_type,
                    )?;
                    self.resizer.resize(&src_view, &mut dst_view, &options)?;
                }
                Rotation::Rotate90Clockwise | Rotation::Rotate90CounterClockwise => {
                    let mut tmp = vec![0; dst.row_stride() * dst.height()];
                    let mut tmp_view = fast_image_resize::images::Image::from_slice_u8(
                        dst.height() as u32,
                        dst.width() as u32,
                        &mut tmp,
                        src_type,
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
                        src_type,
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

impl ImageConverterTrait for CPUConverter {
    fn convert(
        &mut self,
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: Rotation,
        crop: Option<Rect>,
    ) -> Result<()> {
        let mut src = src;
        let mut tmp;

        // YUV conversions need to happen at the start
        match src.fourcc() {
            YUYV => {
                // when there is no crop, no rotation, and width/height is the same, we can
                // directly convert into the dest
                if crop.is_none()
                    && rotation == Rotation::None
                    && dst.width() == src.width()
                    && dst.height() == src.height()
                {
                    match dst.fourcc() {
                        RGB => return self.convert_yuyv_to_rgb(src, dst),
                        RGBA => return self.convert_yuyv_to_rgba(src, dst),
                        _ => {
                            return Err(Error::NotSupported(
                                "YUYV destination not supported".to_string(),
                            ));
                        }
                    }
                } else {
                    // otherwise we convert into a temporary buffer that will be used later for
                    // resize/rotate
                    tmp = TensorImage::new(
                        src.width(),
                        src.height(),
                        dst.fourcc(),
                        Some(edgefirst_tensor::TensorMemory::Mem),
                    )?;
                    match dst.fourcc() {
                        RGB => self.convert_yuyv_to_rgb(src, &mut tmp)?,
                        RGBA => self.convert_yuyv_to_rgba(src, &mut tmp)?,
                        _ => {
                            return Err(Error::NotSupported(
                                "YUYV destination not supported".to_string(),
                            ));
                        }
                    }
                    src = &tmp;
                }
            }
            NV12 => {
                // when there is no crop, no rotation, and width/height is the same, we can
                // directly convert into the dest
                if crop.is_none()
                    && rotation == Rotation::None
                    && dst.width() == src.width()
                    && dst.height() == src.height()
                {
                    match dst.fourcc() {
                        RGB => return self.convert_nv12_to_rgb(src, dst),
                        RGBA => return self.convert_nv12_to_rgba(src, dst),
                        _ => {
                            return Err(Error::NotSupported(
                                "destination format not supported".to_string(),
                            ));
                        }
                    }
                } else {
                    // otherwise we convert into a temporary buffer that will be used later for
                    // resize/rotate
                    tmp = TensorImage::new(
                        src.width(),
                        src.height(),
                        dst.fourcc(),
                        Some(edgefirst_tensor::TensorMemory::Mem),
                    )?;
                    match dst.fourcc() {
                        RGB => self.convert_nv12_to_rgb(src, &mut tmp)?,
                        RGBA => self.convert_nv12_to_rgba(src, &mut tmp)?,
                        _ => {
                            return Err(Error::NotSupported(
                                "destination format not supported".to_string(),
                            ));
                        }
                    }
                    src = &tmp;
                }
            }
            RGB | RGBA => {
                if src.fourcc() != dst.fourcc()
                    && src.width() * src.height() < dst.width() * dst.height()
                {
                    // we do the RGB/RGBA conversion early only when enlarging the image
                    tmp = TensorImage::new(
                        src.width(),
                        src.height(),
                        dst.fourcc(),
                        Some(edgefirst_tensor::TensorMemory::Mem),
                    )?;
                    match dst.fourcc() {
                        RGB => self.convert_rgba_to_rgb(src, &mut tmp)?,
                        RGBA => self.convert_rgb_to_rgba(src, &mut tmp)?,
                        _ => {
                            return Err(Error::NotSupported(
                                "destination format not supported".to_string(),
                            ));
                        }
                    }
                    src = &tmp;
                }
            }
            _ => {
                return Err(Error::NotSupported("unknown format".to_string()));
            }
        }

        matches!(src.fourcc(), RGB | RGBA);
        if src.fourcc() == dst.fourcc() {
            self.resize_and_rotate(dst, src, rotation, crop)?;
        } else {
            let mut tmp2 = TensorImage::new(
                dst.width(),
                dst.height(),
                src.fourcc(),
                Some(edgefirst_tensor::TensorMemory::Mem),
            )?;
            self.resize_and_rotate(&mut tmp2, src, rotation, crop)?;
            match dst.fourcc() {
                RGB => self.convert_rgba_to_rgb(&tmp2, dst)?,
                RGBA => self.convert_rgb_to_rgba(&tmp2, dst)?,
                _ => unreachable!(),
            }
        }

        Ok(())
    }
}
