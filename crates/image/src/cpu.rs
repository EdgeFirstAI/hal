use crate::{
    Error, GREY, ImageConverterTrait, NV12, RGB, RGBA, Rect, Result, Rotation, TensorImage, YUYV,
};
use edgefirst_tensor::{TensorMapTrait, TensorTrait};
use ndarray::{ArrayView3, ArrayViewMut3, Axis};

/// CPUConverter implements the ImageConverter trait using the fallback CPU
/// implementation for image processing.
pub struct CPUConverter {
    resizer: fast_image_resize::Resizer,
    options: fast_image_resize::ResizeOptions,
}

#[inline(always)]
fn limit_to_full(l: u8) -> u8 {
    (((l as u16 - 16) * 255) / (240 - 16)) as u8
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
            Rotation::Clockwise90 => {
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
            Rotation::CounterClockwise90 => {
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

    fn convert_nv12_to_rgba(&self, src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_nv12_to_grey(&self, src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_yuyv_to_rgba(&self, src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_yuyv_to_grey(&self, src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_grey_to_rgb(&self, src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_grey_to_rgba(&self, src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_rgba_to_grey(&self, src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    fn convert_rgb_to_grey(&self, src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
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

    /// The src and dest img should be in RGB/RGBA/grey format for correct
    /// output. If the format is not 1, 3, or 4 bits per pixel, and error will
    /// be returned. The src and dest img must have the same fourcc,
    /// otherwise the function will panic.
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
                Rotation::Clockwise90 | Rotation::CounterClockwise90 => {
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

        src: &TensorImage,
        dst: &mut TensorImage,
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
                        GREY => self.convert_yuyv_to_grey(src, dst)?,
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
                        GREY => self.convert_yuyv_to_grey(src, &mut tmp)?,
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
                        GREY => return self.convert_nv12_to_grey(src, dst),
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
                        GREY => self.convert_nv12_to_grey(src, &mut tmp)?,
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
                // we do the RGB/RGBA conversion early only when enlarging the image
                // This is faster than doing it later

                // we always do Greyscale conversion early
                if src.fourcc() != dst.fourcc()
                    && (src.width() * src.height() < dst.width() * dst.height()
                        || dst.fourcc() == GREY)
                {
                    tmp = TensorImage::new(
                        src.width(),
                        src.height(),
                        dst.fourcc(),
                        Some(edgefirst_tensor::TensorMemory::Mem),
                    )?;
                    match (src.fourcc(), dst.fourcc()) {
                        (RGBA, RGB) => self.convert_rgba_to_rgb(src, &mut tmp)?,
                        (RGBA, GREY) => self.convert_rgba_to_grey(src, &mut tmp)?,
                        (RGB, RGBA) => self.convert_rgb_to_rgba(src, &mut tmp)?,
                        (RGB, GREY) => self.convert_rgb_to_grey(src, &mut tmp)?,
                        (GREY, RGB) => self.convert_grey_to_rgb(src, &mut tmp)?,
                        (GREY, RGBA) => self.convert_grey_to_rgba(src, &mut tmp)?,
                        (RGBA, RGBA) | (RGB, RGB) | (GREY, GREY) => {} // this is unreachable
                        _ => {
                            return Err(Error::NotSupported(
                                "destination format not supported".to_string(),
                            ));
                        }
                    }
                    src = &tmp;
                }
            }
            GREY => {
                // we never convert away from Greyscale early
            }
            _ => {
                return Err(Error::NotSupported("unknown format".to_string()));
            }
        }

        matches!(src.fourcc(), RGB | RGBA | GREY);
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
            match (src.fourcc(), dst.fourcc()) {
                (RGBA, RGB) => self.convert_rgba_to_rgb(&tmp2, dst)?,
                (RGBA, GREY) => self.convert_rgba_to_grey(&tmp2, dst)?,
                (RGB, RGBA) => self.convert_rgb_to_rgba(&tmp2, dst)?,
                (RGB, GREY) => self.convert_rgb_to_grey(&tmp2, dst)?,
                (GREY, RGB) => self.convert_grey_to_rgb(&tmp2, dst)?,
                (GREY, RGBA) => self.convert_grey_to_rgba(&tmp2, dst)?,
                (RGBA, RGBA) | (RGB, RGB) | (GREY, GREY) => {} // this is unreachable
                _ => unreachable!(),
            }
        }

        Ok(())
    }
}
