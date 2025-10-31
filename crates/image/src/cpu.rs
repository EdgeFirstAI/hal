use std::time::Instant;

use crate::{
    _8BPS, Crop, Error, Flip, GREY, ImageConverterTrait, NV12, RGB, RGBA, Rect, Result, Rotation,
    TensorImage, YUYV,
};
use edgefirst_tensor::{TensorMapTrait, TensorTrait};
use four_char_code::FourCharCode;
use ndarray::{ArrayView3, ArrayViewMut3, Axis};

/// CPUConverter implements the ImageConverter trait using the fallback CPU
/// implementation for image processing.
pub struct CPUConverter {
    resizer: fast_image_resize::Resizer,
    options: fast_image_resize::ResizeOptions,
}

unsafe impl Send for CPUConverter {}
unsafe impl Sync for CPUConverter {}

#[inline(always)]
fn limit_to_full(l: u8) -> u8 {
    (((l as u16 - 16) * 255) / (240 - 16)) as u8
}

#[inline(always)]
fn full_to_limit(l: u8) -> u8 {
    ((l as u16 * (240 - 16)) / 255 + 16) as u8
}

impl CPUConverter {
    pub fn new() -> Result<Self> {
        let resizer = fast_image_resize::Resizer::new();
        let options = fast_image_resize::ResizeOptions::new()
            .resize_alg(fast_image_resize::ResizeAlg::Interpolation(
                fast_image_resize::FilterType::Bilinear,
            ))
            .use_alpha(false);

        log::debug!("CPUConverter created");
        Ok(Self { resizer, options })
    }

    pub fn new_nearest() -> Result<Self> {
        let resizer = fast_image_resize::Resizer::new();
        let options = fast_image_resize::ResizeOptions::new()
            .resize_alg(fast_image_resize::ResizeAlg::Nearest)
            .use_alpha(false);
        Ok(Self { resizer, options })
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

    fn convert_nv12_to_yuyv(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), NV12);
        assert_eq!(dst.fourcc(), YUYV);
        Err(Error::NotImplemented("NV12 to YUYV".to_string()))
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

    fn convert_rgba_to_yuyv(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), RGBA);
        assert_eq!(dst.fourcc(), YUYV);

        let src = src.tensor().map()?;
        let src = src.as_slice();

        let mut dst = dst.tensor().map()?;
        let dst = dst.as_mut_slice();

        // compute quantized Bt.709 limited range RGB to YUV matrix
        const BIAS: i32 = 20;
        const Y_R: i32 = (0.2126_f64 * ((235 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const Y_G: i32 = (0.7152_f64 * ((235 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const Y_B: i32 = (((235 - 16) * (1 << BIAS)) as f64 / 255.0).ceil() as i32 - Y_R - Y_G;
        const U_R: i32 = (-0.114572_f64 * ((240 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const U_B: i32 = (0.5_f64 * ((240 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const U_G: i32 = -U_R - U_B;
        const V_R: i32 = (0.5_f64 * ((240 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const V_B: i32 = (-0.045847_f64 * ((240 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const V_G: i32 = -V_R - V_B;

        let process_rgba_to_yuyv = |s: &[u8; 8], d: &mut [u8; 4]| {
            let [r0, g0, b0, _, r1, g1, b1, _] = *s;
            d[0] = (((Y_R * r0 as i32 + Y_G * g0 as i32 + Y_B * b0 as i32) >> BIAS) + 16) as u8;
            d[1] = ((((U_R * (r0 as i32) + U_G * (g0 as i32) + U_B * (b0 as i32)) >> BIAS)
                + ((U_R * (r1 as i32) + U_G * (g1 as i32) + U_B * (b1 as i32)) >> BIAS))
                / 2
                + 128) as u8;
            d[2] = ((Y_R * r1 as i32 + Y_G * g1 as i32 + Y_B * b1 as i32) >> BIAS) as u8 + 16;
            d[3] = ((((V_R * (r0 as i32) + V_G * (g0 as i32) + V_B * (b0 as i32)) >> BIAS)
                + ((V_R * (r1 as i32) + V_G * (g1 as i32) + V_B * (b1 as i32)) >> BIAS))
                / 2
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

    fn convert_rgb_to_yuyv(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), RGB);
        assert_eq!(dst.fourcc(), YUYV);

        let src = src.tensor().map()?;
        let src = src.as_slice();

        let mut dst = dst.tensor().map()?;
        let dst = dst.as_mut_slice();

        // compute quantized Bt.709 limited range RGB to YUV matrix
        const BIAS: i32 = 20;
        const Y_R: i32 = (0.2126_f64 * ((235 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const Y_G: i32 = (0.7152_f64 * ((235 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const Y_B: i32 = (((235 - 16) * (1 << BIAS)) as f64 / 255.0).ceil() as i32 - Y_R - Y_G;
        const U_R: i32 = (-0.114572_f64 * ((240 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const U_B: i32 = (0.5_f64 * ((240 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const U_G: i32 = -U_R - U_B;
        const V_R: i32 = (0.5_f64 * ((240 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const V_B: i32 = (-0.045847_f64 * ((240 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const V_G: i32 = -V_R - V_B;

        let process_rgb_to_yuyv = |s: &[u8; 6], d: &mut [u8; 4]| {
            let [r0, g0, b0, r1, g1, b1] = *s;
            d[0] = (((Y_R * r0 as i32 + Y_G * g0 as i32 + Y_B * b0 as i32) >> BIAS) + 16) as u8;
            d[1] = ((((U_R * (r0 as i32) + U_G * (g0 as i32) + U_B * (b0 as i32)) >> BIAS)
                + ((U_R * (r1 as i32) + U_G * (g1 as i32) + U_B * (b1 as i32)) >> BIAS))
                / 2
                + 128) as u8;
            d[2] = ((Y_R * r1 as i32 + Y_G * g1 as i32 + Y_B * b1 as i32) >> BIAS) as u8 + 16;
            d[3] = ((((V_R * (r0 as i32) + V_G * (g0 as i32) + V_B * (b0 as i32)) >> BIAS)
                + ((V_R * (r1 as i32) + V_G * (g1 as i32) + V_B * (b1 as i32)) >> BIAS))
                / 2
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

    fn copy_image(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        assert_eq!(src.fourcc(), dst.fourcc());
        dst.tensor().map()?.copy_from_slice(&src.tensor().map()?);
        Ok(())
    }

    pub(crate) fn support_conversion(src: FourCharCode, dst: FourCharCode) -> bool {
        matches!(
            (src, dst),
            (NV12, RGB)
                | (NV12, RGBA)
                | (NV12, GREY)
                | (NV12, YUYV)
                | (YUYV, RGB)
                | (YUYV, RGBA)
                | (YUYV, GREY)
                | (YUYV, YUYV)
                | (RGBA, RGB)
                | (RGBA, RGBA)
                | (RGBA, GREY)
                | (RGBA, YUYV)
                | (RGB, RGB)
                | (RGB, RGBA)
                | (RGB, GREY)
                | (RGB, YUYV)
                | (GREY, RGB)
                | (GREY, RGBA)
                | (GREY, GREY)
                | (GREY, YUYV)
        )
    }

    pub(crate) fn convert_format(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        // shapes should be equal
        assert_eq!(src.height(), dst.height());
        assert_eq!(src.width(), dst.width());

        match (src.fourcc(), dst.fourcc()) {
            (NV12, RGB) => Self::convert_nv12_to_rgb(src, dst),
            (NV12, RGBA) => Self::convert_nv12_to_rgba(src, dst),
            (NV12, GREY) => Self::convert_nv12_to_grey(src, dst),
            (NV12, YUYV) => Self::convert_nv12_to_yuyv(src, dst),
            (YUYV, RGB) => Self::convert_yuyv_to_rgb(src, dst),
            (YUYV, RGBA) => Self::convert_yuyv_to_rgba(src, dst),
            (YUYV, GREY) => Self::convert_yuyv_to_grey(src, dst),
            (YUYV, YUYV) => Self::copy_image(src, dst),
            (RGBA, RGB) => Self::convert_rgba_to_rgb(src, dst),
            (RGBA, RGBA) => Self::copy_image(src, dst),
            (RGBA, GREY) => Self::convert_rgba_to_grey(src, dst),
            (RGBA, YUYV) => Self::convert_rgba_to_yuyv(src, dst),
            (RGBA, _8BPS) => todo!(),
            (RGB, RGB) => Self::copy_image(src, dst),
            (RGB, RGBA) => Self::convert_rgb_to_rgba(src, dst),
            (RGB, GREY) => Self::convert_rgb_to_grey(src, dst),
            (RGB, YUYV) => Self::convert_rgb_to_yuyv(src, dst),
            (RGB, _8BPS) => todo!(),
            (GREY, RGB) => Self::convert_grey_to_rgb(src, dst),
            (GREY, RGBA) => Self::convert_grey_to_rgba(src, dst),
            (GREY, GREY) => Self::copy_image(src, dst),
            (GREY, YUYV) => Self::convert_grey_to_yuyv(src, dst),
            (GREY, _8BPS) => todo!(),
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
        dst: &mut TensorImage,
        src: &TensorImage,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
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
                    width: src.width(),
                    height: src.height(),
                }
            });

        if needs_resize {
            let start = Instant::now();
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
                    log::debug!("resize takes {:?}", start.elapsed());
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
                    log::debug!("resize takes {:?}", start.elapsed());
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
                    log::debug!("resize takes {:?}", start.elapsed());
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
        match dst.fourcc() {
            RGBA => Self::fill_image_outside_crop_(dst, rgba, crop),
            RGB => Self::fill_image_outside_crop_(dst, Self::rgba_to_rgb(rgba), crop),
            GREY => Self::fill_image_outside_crop_(dst, Self::rgba_to_grey(rgba), crop),
            YUYV => Self::fill_image_outside_crop_(dst, Self::rgba_to_yuyv(rgba), crop),
            _ => Err(Error::Internal(format!(
                "Found unexpected destination {}",
                dst.fourcc.display()
            ))),
        }
    }

    fn fill_image_outside_crop_<const N: usize>(
        dst: &mut TensorImage,
        pix: [u8; N],
        crop: Rect,
    ) -> Result<()> {
        use rayon::{
            iter::{IntoParallelRefMutIterator, ParallelIterator},
            prelude::ParallelSliceMut,
        };
        let mut map = dst.tensor.map()?;
        let s = map.as_mut_slice().as_chunks_mut::<N>().0;
        // calculate the top/bottom
        let top_offset = (0, (crop.top * dst.width() + crop.left));
        let bottom_offset = (
            ((crop.top + crop.height) * dst.width() + crop.left).min(s.len()),
            s.len(),
        );

        s[top_offset.0..top_offset.1]
            .par_iter_mut()
            .for_each(|x| *x = pix);
        // for p in &mut s[top_offset.0..top_offset.1] {
        //     *p = pix;
        // }

        s[bottom_offset.0..bottom_offset.1]
            .par_iter_mut()
            .for_each(|x| *x = pix);
        // for p in &mut s[bottom_offset.0..bottom_offset.1] {
        //     *p = pix;
        // }

        if dst.width() == crop.width {
            return Ok(());
        }

        // the middle part has a stride as well
        let middle_stride = dst.width() - crop.width;
        let middle_offset = (
            (crop.top * dst.width() + crop.left + crop.width),
            ((crop.top + crop.height) * dst.width() + crop.left + crop.width).min(s.len()),
        );

        s[middle_offset.0..middle_offset.1]
            .par_chunks_exact_mut(dst.width())
            .for_each(|row| {
                for p in &mut row[0..middle_stride] {
                    *p = pix;
                }
            });

        // let middle =
        // s[middle_offset.0..middle_offset.1].chunks_exact_mut(dst.width());
        // for row in middle {
        //     for p in &mut row[0..middle_stride] {
        //         *p = pix;
        //     }
        // }

        Ok(())
    }

    fn rgba_to_rgb(rgba: [u8; 4]) -> [u8; 3] {
        let [r, g, b, _] = rgba;
        [r, g, b]
    }

    fn rgba_to_grey(rgba: [u8; 4]) -> [u8; 1] {
        const BIAS: usize = 20;
        const Y_R: i32 = (0.2126_f64 * (255 * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const Y_G: i32 = (0.7152_f64 * (255 * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const Y_B: i32 = ((255 * (1 << BIAS)) as f64 / 255.0).ceil() as i32 - Y_R - Y_G;
        let [r, g, b, _] = rgba;
        let y = ((Y_R * r as i32 + Y_G * g as i32 + Y_B * b as i32) >> BIAS) as u8;
        [y]
    }

    fn rgba_to_yuyv(rgba: [u8; 4]) -> [u8; 4] {
        // compute quantized Bt.709 limited range RGB to YUV matrix
        const BIAS: i32 = 20;
        const Y_R: i32 = (0.2126_f64 * ((235 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const Y_G: i32 = (0.7152_f64 * ((235 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const Y_B: i32 = (((235 - 16) * (1 << BIAS)) as f64 / 255.0).ceil() as i32 - Y_R - Y_G;
        const U_R: i32 = (-0.114572_f64 * ((240 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const U_B: i32 = (0.5_f64 * ((240 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const U_G: i32 = -U_R - U_B;
        const V_R: i32 = (0.5_f64 * ((240 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const V_B: i32 = (-0.045847_f64 * ((240 - 16) * (1 << BIAS)) as f64 / 255.0).round() as i32;
        const V_G: i32 = -V_R - V_B;

        let [r, g, b, _] = rgba;
        let y = (((Y_R * r as i32 + Y_G * g as i32 + Y_B * b as i32) >> BIAS) + 16) as u8;
        let u = (((U_R * (r as i32) + U_G * (g as i32) + U_B * (b as i32)) >> BIAS) + 128) as u8;
        let v = (((V_R * (r as i32) + V_G * (g as i32) + V_B * (b as i32)) >> BIAS) + 128) as u8;

        [y, u, y, v]
    }
    // fn paint_mk_onto_rgba(
    //     src: &TensorImage,
    //     dst: &mut TensorImage,
    //     rotation: Rotation,
    //     flip: Flip,
    //     crop: Crop,
    // ) {
    //     crop.check_crop(src, dst)?;
    // }
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
            (NV12, YUYV) => RGB, // RGB intermediary for YUYV dest resize/convert/rotation/flip
            (YUYV, RGB) => RGB,
            (YUYV, RGBA) => RGBA,
            (YUYV, GREY) => GREY,
            (YUYV, YUYV) => RGB, // RGB intermediary for YUYV dest resize/convert/rotation/flip
            (RGBA, RGB) => RGB,
            (RGBA, RGBA) => RGBA,
            (RGBA, GREY) => GREY,
            (RGBA, YUYV) => RGBA, // RGB intermediary for YUYV dest resize/convert/rotation/flip
            (RGBA, _8BPS) => RGBA,
            (RGB, RGB) => RGB,
            (RGB, RGBA) => RGB,
            (RGB, GREY) => GREY,
            (RGB, YUYV) => RGB, // RGB intermediary for YUYV dest resize/convert/rotation/flip
            (RGB, _8BPS) => RGB,
            (GREY, RGB) => RGB,
            (GREY, RGBA) => RGBA,
            (GREY, GREY) => GREY,
            (GREY, YUYV) => GREY,
            (GREY, _8BPS) => GREY,
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
                    width: src.width(),
                    height: src.height(),
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
            self.resize_flip_rotate(dst, tmp, rotation, flip, crop)?;
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
                    width: src.width(),
                    height: src.height(),
                }
            }) && crop.dst_color.is_none()
            {
                // convert the dst into tmp2 when there is a dst crop
                // TODO: this could be optimized by changing convert_format to take a
                // destination crop?

                Self::convert_format(dst, &mut tmp2)?;
            }
            self.resize_flip_rotate(&mut tmp2, tmp, rotation, flip, crop)?;
            Self::convert_format(&tmp2, dst)?;
        }

        if let Some(dst_rect) = crop.dst_rect
            && dst_rect
                != (Rect {
                    left: 0,
                    top: 0,
                    width: src.width(),
                    height: src.height(),
                })
            && let Some(dst_color) = crop.dst_color
        {
            Self::fill_image_outside_crop(dst, dst_color, dst_rect)?;
        }

        Ok(())
    }
}
