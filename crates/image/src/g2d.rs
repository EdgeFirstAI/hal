// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "linux")]

use crate::{
    CPUConverter, Crop, Error, Flip, ImageConverterTrait, RGB, RGBA, Result, Rotation, TensorImage,
    YUYV,
};
use edgefirst_tensor::Tensor;
use g2d_sys::{G2D, G2DFormat, G2DPhysical, G2DSurface};
use std::{os::fd::AsRawFd, time::Instant};

/// G2DConverter implements the ImageConverter trait using the NXP G2D
/// library for hardware-accelerated image processing on i.MX platforms.
#[derive(Debug)]
pub struct G2DConverter {
    g2d: G2D,
}

unsafe impl Send for G2DConverter {}
unsafe impl Sync for G2DConverter {}

impl G2DConverter {
    /// Creates a new G2DConverter instance.
    pub fn new() -> Result<Self> {
        let mut g2d = G2D::new("libg2d.so.2")?;
        g2d.set_bt709_colorspace()?;

        log::debug!("G2DConverter created with version {:?}", g2d.version());
        Ok(Self { g2d })
    }

    /// Returns the G2D library version as defined by G2D_VERSION in the shared
    /// library.
    pub fn version(&self) -> g2d_sys::Version {
        self.g2d.version()
    }

    fn convert_(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        let mut src_surface: G2DSurface = src.try_into()?;
        let mut dst_surface: G2DSurface = dst.try_into()?;

        src_surface.rot = match flip {
            Flip::None => g2d_sys::g2d_rotation_G2D_ROTATION_0,
            Flip::Vertical => g2d_sys::g2d_rotation_G2D_FLIP_V,
            Flip::Horizontal => g2d_sys::g2d_rotation_G2D_FLIP_H,
        };

        dst_surface.rot = match rotation {
            Rotation::None => g2d_sys::g2d_rotation_G2D_ROTATION_0,
            Rotation::Clockwise90 => g2d_sys::g2d_rotation_G2D_ROTATION_90,
            Rotation::Rotate180 => g2d_sys::g2d_rotation_G2D_ROTATION_180,
            Rotation::CounterClockwise90 => g2d_sys::g2d_rotation_G2D_ROTATION_270,
        };

        if let Some(crop_rect) = crop.src_rect {
            src_surface.left = crop_rect.left as i32;
            src_surface.top = crop_rect.top as i32;
            src_surface.right = (crop_rect.left + crop_rect.width) as i32;
            src_surface.bottom = (crop_rect.top + crop_rect.height) as i32;
        }

        // need to clear before assigning the crop
        let needs_clear = if let Some(dst_rect) = crop.dst_rect
            && (dst_rect.left != 0
                || dst_rect.top != 0
                || dst_rect.width != dst.width()
                || dst_rect.height != dst.height())
            && crop.dst_color.is_some()
        {
            true
        } else {
            false
        };

        let mut cleared = false;
        if needs_clear
            && dst.fourcc != RGB
            && let Some(dst_rect) = crop.dst_rect
            && dst_rect.width * dst_rect.height < dst.width() * dst.height() / 2
            && let Some(dst_color) = crop.dst_color
        {
            let start = Instant::now();
            self.g2d.clear(&mut dst_surface, dst_color)?;
            log::trace!("clear takes {:?}", start.elapsed());
            cleared = true;
        }

        if let Some(crop_rect) = crop.dst_rect {
            dst_surface.planes[0] += ((crop_rect.top * dst_surface.width as usize + crop_rect.left)
                * dst.channels()) as u64;

            dst_surface.right = crop_rect.width as i32;
            dst_surface.bottom = crop_rect.height as i32;
            dst_surface.width = crop_rect.width as i32;
            dst_surface.height = crop_rect.height as i32;

            // right: img.width() as i32,
            // bottom: img.height() as i32,
            // stride: img.width() as i32,
            // width: img.width() as i32,
            // height: img.height() as i32,

            // dst_surface.left = crop_rect.left as i32;
            // dst_surface.top = crop_rect.top as i32;
            // dst_surface.right = (crop_rect.left + crop_rect.width) as i32;
            // dst_surface.bottom = (crop_rect.top + crop_rect.height) as i32;
        }

        log::trace!("Blitting from {src_surface:?} to {dst_surface:?}");
        self.g2d.blit(&src_surface, &dst_surface)?;

        if needs_clear
            && !cleared
            && let Some(dst_color) = crop.dst_color
            && let Some(dst_rect) = crop.dst_rect
        {
            let start = Instant::now();
            CPUConverter::fill_image_outside_crop(dst, dst_color, dst_rect)?;
            log::trace!("clear takes {:?}", start.elapsed());
        }

        Ok(())
    }
}

impl ImageConverterTrait for G2DConverter {
    /// Converts the source image to the destination image using G2D.
    ///
    /// # Arguments
    ///
    /// * `dst` - The destination image to be converted to.
    /// * `src` - The source image to convert from.
    /// * `rotation` - The rotation to apply to the destination image (after
    ///   crop if specified).
    /// * `crop` - An optional rectangle specifying the area to crop from the
    ///   source image.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure of the conversion.
    fn convert(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        crop.check_crop(src, dst)?;
        match (src.fourcc(), dst.fourcc()) {
            (RGBA, RGBA) => {}
            (RGBA, YUYV) => {}
            (RGBA, RGB) => {}
            (YUYV, RGBA) => {}
            (YUYV, YUYV) => {}
            (YUYV, RGB) => {}
            // (YUYV, NV12) => {}
            // (NV12, RGBA) => {}
            // (NV12, YUYV) => {}
            // (NV12, RGB) => {}
            (s, d) => {
                return Err(Error::NotSupported(format!(
                    "G2D does not support {} to {} conversion",
                    s.display(),
                    d.display()
                )));
            }
        }
        // if dst.fourcc() == RGB
        //     && crop.dst_rect.is_some_and(|crop| {
        //         crop.left != 0
        //             || crop.top != 0
        //             || crop.width != dst.width()
        //             || crop.height != dst.height()
        //     })
        // {
        //     return Err(Error::NotSupported(
        //         "G2D does not support conversion to RGB with destination
        // crop".to_string(),     ));
        // }
        self.convert_(src, dst, rotation, flip, crop)
    }
}

impl TryFrom<&TensorImage> for G2DSurface {
    type Error = Error;

    fn try_from(img: &TensorImage) -> Result<Self, Self::Error> {
        let phys: G2DPhysical = match img.tensor() {
            Tensor::Shm(t) => t.as_raw_fd(),
            Tensor::Dma(t) => t.as_raw_fd(),
            _ => {
                return Err(Error::NotImplemented(
                    "g2d only supports Shm or Dma memory".to_string(),
                ));
            }
        }
        .try_into()?;

        Ok(Self {
            planes: [phys.address(), 0, 0],
            format: G2DFormat::try_from(img.fourcc())?.format(),
            left: 0,
            top: 0,
            right: img.width() as i32,
            bottom: img.height() as i32,
            stride: img.width() as i32,
            width: img.width() as i32,
            height: img.height() as i32,
            blendfunc: 0,
            clrcolor: 0,
            rot: 0,
            global_alpha: 0,
        })
    }
}

impl TryFrom<&mut TensorImage> for G2DSurface {
    type Error = Error;

    fn try_from(img: &mut TensorImage) -> Result<Self, Self::Error> {
        let phys: G2DPhysical = match img.tensor() {
            Tensor::Shm(t) => t.as_raw_fd(),
            Tensor::Dma(t) => t.as_raw_fd(),
            _ => {
                return Err(Error::NotImplemented(
                    "g2d only supports Shm or Dma memory".to_string(),
                ));
            }
        }
        .try_into()?;

        Ok(Self {
            planes: [phys.address(), 0, 0],
            format: G2DFormat::try_from(img.fourcc())?.format(),
            left: 0,
            top: 0,
            right: img.width() as i32,
            bottom: img.height() as i32,
            stride: img.width() as i32,
            width: img.width() as i32,
            height: img.height() as i32,
            blendfunc: 0,
            clrcolor: 0,
            rot: 0,
            global_alpha: 0,
        })
    }
}

#[cfg(feature = "g2d_test_formats")]
#[cfg(test)]
mod g2d_tests {
    use super::*;
    use crate::{
        CPUConverter, Flip, G2DConverter, GREY, ImageConverterTrait, RGB, RGBA, Rect, Rotation,
        TensorImage, YUYV,
    };
    use edgefirst_tensor::{TensorMapTrait, TensorMemory, TensorTrait};
    use four_char_code::FourCharCode;
    use image::buffer::ConvertBuffer;

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_formats_no_resize() {
        for i in [RGBA, YUYV, RGB, GREY] {
            for o in [RGBA, YUYV, RGB, GREY] {
                let res = test_g2d_format_no_resize_(i, o);
                if let Err(e) = res {
                    println!("{} to {} failed: {e:?}", i.display(), o.display());
                } else {
                    println!("{} to {} success", i.display(), o.display());
                }
            }
        }
    }

    fn test_g2d_format_no_resize_(
        g2d_in_fmt: FourCharCode,
        g2d_out_fmt: FourCharCode,
    ) -> Result<(), crate::Error> {
        let dst_width = 1280;
        let dst_height = 720;
        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load_jpeg(&file, Some(RGB), None)?;

        let mut src2 = TensorImage::new(1280, 720, g2d_in_fmt, Some(TensorMemory::Dma))?;

        let mut cpu_converter = CPUConverter::new()?;

        cpu_converter.convert(&src, &mut src2, Rotation::None, Flip::None, Crop::no_crop())?;

        let mut g2d_dst =
            TensorImage::new(dst_width, dst_height, g2d_out_fmt, Some(TensorMemory::Dma))?;
        let mut g2d_converter = G2DConverter::new()?;
        g2d_converter.convert_(
            &src2,
            &mut g2d_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGB, None)?;
        cpu_converter.convert(
            &g2d_dst,
            &mut cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;

        compare_images(
            &src,
            &cpu_dst,
            0.98,
            &format!("{}_to_{}", g2d_in_fmt.display(), g2d_out_fmt.display()),
        )
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_formats_with_resize() {
        for i in [RGBA, YUYV, RGB, GREY] {
            for o in [RGBA, YUYV, RGB, GREY] {
                let res = test_g2d_format_with_resize_(i, o);
                if let Err(e) = res {
                    println!("{} to {} failed: {e:?}", i.display(), o.display());
                } else {
                    println!("{} to {} success", i.display(), o.display());
                }
            }
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_formats_with_resize_dst_crop() {
        for i in [RGBA, YUYV, RGB, GREY] {
            for o in [RGBA, YUYV, RGB, GREY] {
                let res = test_g2d_format_with_resize_dst_crop(i, o);
                if let Err(e) = res {
                    println!("{} to {} failed: {e:?}", i.display(), o.display());
                } else {
                    println!("{} to {} success", i.display(), o.display());
                }
            }
        }
    }

    fn test_g2d_format_with_resize_(
        g2d_in_fmt: FourCharCode,
        g2d_out_fmt: FourCharCode,
    ) -> Result<(), crate::Error> {
        let dst_width = 600;
        let dst_height = 400;
        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load_jpeg(&file, Some(RGB), None)?;

        let mut cpu_converter = CPUConverter::new()?;

        let mut reference = TensorImage::new(dst_width, dst_height, RGB, Some(TensorMemory::Dma))?;
        cpu_converter.convert(
            &src,
            &mut reference,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;

        let mut src2 = TensorImage::new(1280, 720, g2d_in_fmt, Some(TensorMemory::Dma))?;
        cpu_converter.convert(&src, &mut src2, Rotation::None, Flip::None, Crop::no_crop())?;

        let mut g2d_dst =
            TensorImage::new(dst_width, dst_height, g2d_out_fmt, Some(TensorMemory::Dma))?;
        let mut g2d_converter = G2DConverter::new()?;
        g2d_converter.convert_(
            &src2,
            &mut g2d_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGB, None)?;
        cpu_converter.convert(
            &g2d_dst,
            &mut cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;

        compare_images(
            &reference,
            &cpu_dst,
            0.98,
            &format!(
                "{}_to_{}_resized",
                g2d_in_fmt.display(),
                g2d_out_fmt.display()
            ),
        )
    }

    fn test_g2d_format_with_resize_dst_crop(
        g2d_in_fmt: FourCharCode,
        g2d_out_fmt: FourCharCode,
    ) -> Result<(), crate::Error> {
        let dst_width = 600;
        let dst_height = 400;
        let crop = Crop {
            src_rect: None,
            dst_rect: Some(Rect {
                top: 100,
                left: 100,
                height: 100,
                width: 200,
            }),
            dst_color: None,
        };
        let file = include_bytes!("../../../testdata/zidane.jpg").to_vec();
        let src = TensorImage::load_jpeg(&file, Some(RGB), None)?;

        let mut cpu_converter = CPUConverter::new()?;

        let mut reference = TensorImage::new(dst_width, dst_height, RGB, Some(TensorMemory::Dma))?;
        reference.tensor.map().unwrap().as_mut_slice().fill(128);
        cpu_converter.convert(&src, &mut reference, Rotation::None, Flip::None, crop)?;

        let mut src2 = TensorImage::new(1280, 720, g2d_in_fmt, Some(TensorMemory::Dma))?;
        cpu_converter.convert(&src, &mut src2, Rotation::None, Flip::None, Crop::no_crop())?;

        let mut g2d_dst =
            TensorImage::new(dst_width, dst_height, g2d_out_fmt, Some(TensorMemory::Dma))?;
        g2d_dst.tensor.map().unwrap().as_mut_slice().fill(128);
        let mut g2d_converter = G2DConverter::new()?;
        g2d_converter.convert_(&src2, &mut g2d_dst, Rotation::None, Flip::None, crop)?;

        let mut cpu_dst = TensorImage::new(dst_width, dst_height, RGB, None)?;
        cpu_converter.convert(
            &g2d_dst,
            &mut cpu_dst,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;

        compare_images(
            &reference,
            &cpu_dst,
            0.98,
            &format!(
                "{}_to_{}_resized_dst_crop",
                g2d_in_fmt.display(),
                g2d_out_fmt.display()
            ),
        )
    }

    fn compare_images(
        img1: &TensorImage,
        img2: &TensorImage,
        threshold: f64,
        name: &str,
    ) -> Result<(), crate::Error> {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");
        assert_eq!(img1.fourcc(), img2.fourcc(), "FourCC differ");
        assert!(
            matches!(img1.fourcc(), RGB | RGBA),
            "FourCC must be RGB or RGBA for comparison"
        );
        let image1 = match img1.fourcc() {
            RGB => image::RgbImage::from_vec(
                img1.width() as u32,
                img1.height() as u32,
                img1.tensor().map().unwrap().to_vec(),
            )
            .unwrap(),
            RGBA => image::RgbaImage::from_vec(
                img1.width() as u32,
                img1.height() as u32,
                img1.tensor().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),

            _ => unreachable!(),
        };

        let image2 = match img2.fourcc() {
            RGB => image::RgbImage::from_vec(
                img2.width() as u32,
                img2.height() as u32,
                img2.tensor().map().unwrap().to_vec(),
            )
            .unwrap(),
            RGBA => image::RgbaImage::from_vec(
                img2.width() as u32,
                img2.height() as u32,
                img2.tensor().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),

            _ => unreachable!(),
        };

        let similarity = image_compare::rgb_similarity_structure(
            &image_compare::Algorithm::RootMeanSquared,
            &image1,
            &image2,
        )
        .expect("Image Comparison failed");

        if similarity.score < threshold {
            image1.save(format!("{name}_1.png")).unwrap();
            image2.save(format!("{name}_2.png")).unwrap();
            // similarity
            //     .image
            //     .to_color_map()
            //     .save(format!("{name}.png"))
            //     .unwrap();
            return Err(Error::Internal(format!(
                "{name}: converted image and target image have similarity score too low: {} < {}",
                similarity.score, threshold
            )));
        }
        Ok(())
    }
}
