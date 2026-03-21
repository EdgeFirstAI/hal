// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

#![cfg(target_os = "linux")]

use crate::{CPUProcessor, Crop, Error, Flip, ImageProcessorTrait, Result, Rotation};
use edgefirst_tensor::{DType, PixelFormat, Tensor, TensorDyn, TensorMapTrait, TensorTrait};
use four_char_code::FourCharCode;
use g2d_sys::{G2DFormat, G2DPhysical, G2DSurface, G2D};
use std::{os::fd::AsRawFd, time::Instant};

/// Convert a PixelFormat to the G2D-compatible FourCharCode.
fn pixelfmt_to_fourcc(fmt: PixelFormat) -> FourCharCode {
    use four_char_code::four_char_code;
    match fmt {
        PixelFormat::Rgb => four_char_code!("RGB "),
        PixelFormat::Rgba => four_char_code!("RGBA"),
        PixelFormat::Bgra => four_char_code!("BGRA"),
        PixelFormat::Grey => four_char_code!("Y800"),
        PixelFormat::Yuyv => four_char_code!("YUYV"),
        PixelFormat::Vyuy => four_char_code!("VYUY"),
        PixelFormat::Nv12 => four_char_code!("NV12"),
        PixelFormat::Nv16 => four_char_code!("NV16"),
        // Planar formats have no standard FourCC; use RGBA as fallback
        _ => four_char_code!("RGBA"),
    }
}

/// G2DConverter implements the ImageProcessor trait using the NXP G2D
/// library for hardware-accelerated image processing on i.MX platforms.
#[derive(Debug)]
pub struct G2DProcessor {
    g2d: G2D,
}

unsafe impl Send for G2DProcessor {}
unsafe impl Sync for G2DProcessor {}

impl G2DProcessor {
    /// Creates a new G2DConverter instance.
    pub fn new() -> Result<Self> {
        let mut g2d = G2D::new("libg2d.so.2")?;
        g2d.set_bt709_colorspace()?;

        log::debug!("G2DConverter created with version {:?}", g2d.version());
        Ok(Self { g2d })
    }

    /// Returns the G2D library version as defined by _G2D_VERSION in the shared
    /// library.
    pub fn version(&self) -> g2d_sys::Version {
        self.g2d.version()
    }

    fn convert_impl(
        &mut self,
        src_dyn: &TensorDyn,
        dst_dyn: &mut TensorDyn,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        if log::log_enabled!(log::Level::Trace) {
            log::trace!(
                "G2D convert: {:?}({:?}/{:?}) → {:?}({:?}/{:?})",
                src_dyn.format(),
                src_dyn.dtype(),
                src_dyn.memory(),
                dst_dyn.format(),
                dst_dyn.dtype(),
                dst_dyn.memory(),
            );
        }

        if src_dyn.dtype() != DType::U8 {
            return Err(Error::NotSupported(
                "G2D only supports u8 source tensors".to_string(),
            ));
        }
        let is_int8_dst = dst_dyn.dtype() == DType::I8;
        if dst_dyn.dtype() != DType::U8 && !is_int8_dst {
            return Err(Error::NotSupported(
                "G2D only supports u8 or i8 destination tensors".to_string(),
            ));
        }

        let src_fmt = src_dyn.format().ok_or(Error::NotAnImage)?;
        let dst_fmt = dst_dyn.format().ok_or(Error::NotAnImage)?;

        // Validate supported format pairs
        use PixelFormat::*;
        match (src_fmt, dst_fmt) {
            (Rgba, Rgba) => {}
            (Rgba, Yuyv) => {}
            (Rgba, Rgb) => {}
            (Yuyv, Rgba) => {}
            (Yuyv, Yuyv) => {}
            (Yuyv, Rgb) => {}
            // VYUY: i.MX8MP G2D hardware rejects VYUY blits (only YUYV/UYVY
            // among packed YUV 4:2:2). ImageProcessor falls through to CPU.
            (Nv12, Rgba) => {}
            (Nv12, Yuyv) => {}
            (Nv12, Rgb) => {}
            (Rgba, Bgra) => {}
            (Yuyv, Bgra) => {}
            (Nv12, Bgra) => {}
            (Bgra, Bgra) => {}
            (s, d) => {
                return Err(Error::NotSupported(format!(
                    "G2D does not support {} to {} conversion",
                    s, d
                )));
            }
        }

        crop.check_crop_dyn(src_dyn, dst_dyn)?;

        let src = src_dyn.as_u8().unwrap();
        // For i8 destinations, reinterpret as u8 for G2D (same byte layout).
        // The XOR 0x80 post-pass is applied after the blit completes.
        let dst = if is_int8_dst {
            // SAFETY: Tensor<i8> and Tensor<u8> have identical memory layout.
            // The T parameter only affects PhantomData<T> (zero-sized) in
            // TensorStorage variants and the typed view from map(). The chroma
            // field (Option<Box<Tensor<T>>>) is also layout-identical. This
            // reinterpreted reference is used only for shape/fd access and the
            // G2D blit (which operates on raw DMA bytes). It does not outlive
            // dst_dyn and is never stored.
            let i8_tensor = dst_dyn.as_i8_mut().unwrap();
            unsafe { &mut *(i8_tensor as *mut Tensor<i8> as *mut Tensor<u8>) }
        } else {
            dst_dyn.as_u8_mut().unwrap()
        };

        let mut src_surface = tensor_to_g2d_surface(src)?;
        let mut dst_surface = tensor_to_g2d_surface(dst)?;

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

        let dst_w = dst.width().unwrap();
        let dst_h = dst.height().unwrap();

        // Clear the destination with the letterbox color before blitting the
        // image into the sub-region.
        //
        // g2d_clear does not support 3-byte-per-pixel formats (RGB888, BGR888).
        // For those formats, fall back to CPU fill after the blit.
        let needs_clear = crop.dst_color.is_some()
            && crop.dst_rect.is_some_and(|dst_rect| {
                dst_rect.left != 0
                    || dst_rect.top != 0
                    || dst_rect.width != dst_w
                    || dst_rect.height != dst_h
            });

        if needs_clear && dst_fmt != Rgb {
            if let Some(dst_color) = crop.dst_color {
                let start = Instant::now();
                self.g2d.clear(&mut dst_surface, dst_color)?;
                log::trace!("g2d clear takes {:?}", start.elapsed());
            }
        }

        if let Some(crop_rect) = crop.dst_rect {
            dst_surface.planes[0] += ((crop_rect.top * dst_surface.width as usize + crop_rect.left)
                * dst_fmt.channels()) as u64;

            dst_surface.right = crop_rect.width as i32;
            dst_surface.bottom = crop_rect.height as i32;
            dst_surface.width = crop_rect.width as i32;
            dst_surface.height = crop_rect.height as i32;
        }

        log::trace!("G2D blit: {src_fmt}→{dst_fmt} int8={is_int8_dst}");
        self.g2d.blit(&src_surface, &dst_surface)?;
        self.g2d.finish()?;
        log::trace!("G2D blit complete");

        // CPU fallback for RGB888 (unsupported by g2d_clear)
        if needs_clear && dst_fmt == Rgb {
            if let (Some(dst_color), Some(dst_rect)) = (crop.dst_color, crop.dst_rect) {
                let start = Instant::now();
                CPUProcessor::fill_image_outside_crop_u8(dst, dst_color, dst_rect)?;
                log::trace!("cpu fill takes {:?}", start.elapsed());
            }
        }

        // Apply XOR 0x80 for int8 output (u8→i8 bias conversion).
        // map() triggers DMA_BUF_SYNC_START (cache invalidation) so CPU reads
        // the G2D-written data correctly. The map drop triggers DMA_BUF_SYNC_END
        // (cache flush) so downstream DMA consumers see the XOR'd data.
        if is_int8_dst {
            let start = Instant::now();
            let mut map = dst.map()?;
            let data = map.as_mut_slice();
            for byte in data.iter_mut() {
                *byte ^= 0x80;
            }
            log::trace!("g2d int8 XOR 0x80 post-pass takes {:?}", start.elapsed());
        }

        Ok(())
    }
}

impl ImageProcessorTrait for G2DProcessor {
    fn convert(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        self.convert_impl(src, dst, rotation, flip, crop)
    }

    fn draw_masks(
        &mut self,
        _dst: &mut TensorDyn,
        _detect: &[crate::DetectBox],
        _segmentation: &[crate::Segmentation],
    ) -> Result<()> {
        Err(Error::NotImplemented(
            "G2D does not support drawing detection or segmentation overlays".to_string(),
        ))
    }

    fn draw_masks_proto(
        &mut self,
        _dst: &mut TensorDyn,
        _detect: &[crate::DetectBox],
        _proto_data: &crate::ProtoData,
    ) -> Result<()> {
        Err(Error::NotImplemented(
            "G2D does not support drawing detection or segmentation overlays".to_string(),
        ))
    }

    fn decode_masks_atlas(
        &mut self,
        _detect: &[crate::DetectBox],
        _proto_data: crate::ProtoData,
        _output_width: usize,
        _output_height: usize,
    ) -> Result<(Vec<u8>, Vec<crate::MaskRegion>)> {
        Err(Error::NotImplemented(
            "G2D does not support decoding mask atlas".to_string(),
        ))
    }

    fn set_class_colors(&mut self, _: &[[u8; 4]]) -> Result<()> {
        Err(Error::NotImplemented(
            "G2D does not support setting colors for rendering detection or segmentation overlays"
                .to_string(),
        ))
    }
}

/// Build a `G2DSurface` from a `Tensor<u8>` that carries pixel-format metadata.
///
/// The tensor must be backed by DMA memory and must have a pixel format set.
fn tensor_to_g2d_surface(img: &Tensor<u8>) -> Result<G2DSurface> {
    let fmt = img.format().ok_or(Error::NotAnImage)?;
    let dma = img
        .as_dma()
        .ok_or_else(|| Error::NotImplemented("g2d only supports Dma memory".to_string()))?;
    let phys: G2DPhysical = dma.fd.as_raw_fd().try_into()?;

    // NV12 is a two-plane format: Y plane followed by interleaved UV plane.
    // planes[0] = Y plane start, planes[1] = UV plane start (Y size = width * height)
    let base_addr = phys.address();
    let planes = if fmt == PixelFormat::Nv12 {
        if img.is_multiplane() {
            // Multiplane: UV in separate DMA-BUF, get its physical address
            let chroma = img.chroma().unwrap();
            let chroma_dma = chroma.as_dma().ok_or_else(|| {
                Error::NotImplemented("g2d multiplane chroma must be DMA-backed".to_string())
            })?;
            let uv_phys: G2DPhysical = chroma_dma.fd.as_raw_fd().try_into()?;
            [base_addr, uv_phys.address(), 0]
        } else {
            let w = img.width().unwrap();
            let h = img.height().unwrap();
            let uv_offset = (w * h) as u64;
            [base_addr, base_addr + uv_offset, 0]
        }
    } else {
        [base_addr, 0, 0]
    };

    let w = img.width().unwrap();
    let h = img.height().unwrap();
    let fourcc = pixelfmt_to_fourcc(fmt);

    Ok(G2DSurface {
        planes,
        format: G2DFormat::try_from(fourcc)?.format(),
        left: 0,
        top: 0,
        right: w as i32,
        bottom: h as i32,
        stride: w as i32,
        width: w as i32,
        height: h as i32,
        blendfunc: 0,
        clrcolor: 0,
        rot: 0,
        global_alpha: 0,
    })
}

#[cfg(feature = "g2d_test_formats")]
#[cfg(test)]
mod g2d_tests {
    use super::*;
    use crate::{CPUProcessor, Flip, G2DProcessor, ImageProcessorTrait, Rect, Rotation};
    use edgefirst_tensor::{
        is_dma_available, DType, PixelFormat, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
    };
    use image::buffer::ConvertBuffer;

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_formats_no_resize() {
        for i in [
            PixelFormat::Rgba,
            PixelFormat::Yuyv,
            PixelFormat::Rgb,
            PixelFormat::Grey,
            PixelFormat::Nv12,
        ] {
            for o in [
                PixelFormat::Rgba,
                PixelFormat::Yuyv,
                PixelFormat::Rgb,
                PixelFormat::Grey,
            ] {
                let res = test_g2d_format_no_resize_(i, o);
                if let Err(e) = res {
                    println!("{i} to {o} failed: {e:?}");
                } else {
                    println!("{i} to {o} success");
                }
            }
        }
    }

    fn test_g2d_format_no_resize_(
        g2d_in_fmt: PixelFormat,
        g2d_out_fmt: PixelFormat,
    ) -> Result<(), crate::Error> {
        let dst_width = 1280;
        let dst_height = 720;
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgb), None)?;

        // Create DMA buffer for G2D input
        let mut src2 = TensorDyn::image(1280, 720, g2d_in_fmt, DType::U8, Some(TensorMemory::Dma))?;

        let mut cpu_converter = CPUProcessor::new();

        // For PixelFormat::Nv12 input, load from file since CPU doesn't support PixelFormat::Rgb→PixelFormat::Nv12
        if g2d_in_fmt == PixelFormat::Nv12 {
            let nv12_bytes = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/zidane.nv12"
            ));
            src2.as_u8()
                .unwrap()
                .map()?
                .as_mut_slice()
                .copy_from_slice(nv12_bytes);
        } else {
            cpu_converter.convert(&src, &mut src2, Rotation::None, Flip::None, Crop::no_crop())?;
        }

        let mut g2d_dst = TensorDyn::image(
            dst_width,
            dst_height,
            g2d_out_fmt,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let mut g2d_converter = G2DProcessor::new()?;
        let src2_dyn = src2;
        let mut g2d_dst_dyn = g2d_dst;
        g2d_converter.convert(
            &src2_dyn,
            &mut g2d_dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        g2d_dst = {
            let mut __t = g2d_dst_dyn.into_u8().unwrap();
            __t.set_format(g2d_out_fmt)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        let mut cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgb, DType::U8, None)?;
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
            &format!("{g2d_in_fmt}_to_{g2d_out_fmt}"),
        )
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_formats_with_resize() {
        for i in [
            PixelFormat::Rgba,
            PixelFormat::Yuyv,
            PixelFormat::Rgb,
            PixelFormat::Grey,
            PixelFormat::Nv12,
        ] {
            for o in [
                PixelFormat::Rgba,
                PixelFormat::Yuyv,
                PixelFormat::Rgb,
                PixelFormat::Grey,
            ] {
                let res = test_g2d_format_with_resize_(i, o);
                if let Err(e) = res {
                    println!("{i} to {o} failed: {e:?}");
                } else {
                    println!("{i} to {o} success");
                }
            }
        }
    }

    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_formats_with_resize_dst_crop() {
        for i in [
            PixelFormat::Rgba,
            PixelFormat::Yuyv,
            PixelFormat::Rgb,
            PixelFormat::Grey,
            PixelFormat::Nv12,
        ] {
            for o in [
                PixelFormat::Rgba,
                PixelFormat::Yuyv,
                PixelFormat::Rgb,
                PixelFormat::Grey,
            ] {
                let res = test_g2d_format_with_resize_dst_crop(i, o);
                if let Err(e) = res {
                    println!("{i} to {o} failed: {e:?}");
                } else {
                    println!("{i} to {o} success");
                }
            }
        }
    }

    fn test_g2d_format_with_resize_(
        g2d_in_fmt: PixelFormat,
        g2d_out_fmt: PixelFormat,
    ) -> Result<(), crate::Error> {
        let dst_width = 600;
        let dst_height = 400;
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgb), None)?;

        let mut cpu_converter = CPUProcessor::new();

        let mut reference = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        cpu_converter.convert(
            &src,
            &mut reference,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;

        // Create DMA buffer for G2D input
        let mut src2 = TensorDyn::image(1280, 720, g2d_in_fmt, DType::U8, Some(TensorMemory::Dma))?;

        // For PixelFormat::Nv12 input, load from file since CPU doesn't support PixelFormat::Rgb→PixelFormat::Nv12
        if g2d_in_fmt == PixelFormat::Nv12 {
            let nv12_bytes = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/zidane.nv12"
            ));
            src2.as_u8()
                .unwrap()
                .map()?
                .as_mut_slice()
                .copy_from_slice(nv12_bytes);
        } else {
            cpu_converter.convert(&src, &mut src2, Rotation::None, Flip::None, Crop::no_crop())?;
        }

        let mut g2d_dst = TensorDyn::image(
            dst_width,
            dst_height,
            g2d_out_fmt,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let mut g2d_converter = G2DProcessor::new()?;
        let src2_dyn = src2;
        let mut g2d_dst_dyn = g2d_dst;
        g2d_converter.convert(
            &src2_dyn,
            &mut g2d_dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        g2d_dst = {
            let mut __t = g2d_dst_dyn.into_u8().unwrap();
            __t.set_format(g2d_out_fmt)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        let mut cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgb, DType::U8, None)?;
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
            &format!("{g2d_in_fmt}_to_{g2d_out_fmt}_resized"),
        )
    }

    fn test_g2d_format_with_resize_dst_crop(
        g2d_in_fmt: PixelFormat,
        g2d_out_fmt: PixelFormat,
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
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgb), None)?;

        let mut cpu_converter = CPUProcessor::new();

        let mut reference = TensorDyn::image(
            dst_width,
            dst_height,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        reference
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(128);
        cpu_converter.convert(&src, &mut reference, Rotation::None, Flip::None, crop)?;

        // Create DMA buffer for G2D input
        let mut src2 = TensorDyn::image(1280, 720, g2d_in_fmt, DType::U8, Some(TensorMemory::Dma))?;

        // For PixelFormat::Nv12 input, load from file since CPU doesn't support PixelFormat::Rgb→PixelFormat::Nv12
        if g2d_in_fmt == PixelFormat::Nv12 {
            let nv12_bytes = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/zidane.nv12"
            ));
            src2.as_u8()
                .unwrap()
                .map()?
                .as_mut_slice()
                .copy_from_slice(nv12_bytes);
        } else {
            cpu_converter.convert(&src, &mut src2, Rotation::None, Flip::None, Crop::no_crop())?;
        }

        let mut g2d_dst = TensorDyn::image(
            dst_width,
            dst_height,
            g2d_out_fmt,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        g2d_dst
            .as_u8()
            .unwrap()
            .map()
            .unwrap()
            .as_mut_slice()
            .fill(128);
        let mut g2d_converter = G2DProcessor::new()?;
        let src2_dyn = src2;
        let mut g2d_dst_dyn = g2d_dst;
        g2d_converter.convert(
            &src2_dyn,
            &mut g2d_dst_dyn,
            Rotation::None,
            Flip::None,
            crop,
        )?;
        g2d_dst = {
            let mut __t = g2d_dst_dyn.into_u8().unwrap();
            __t.set_format(g2d_out_fmt)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        let mut cpu_dst =
            TensorDyn::image(dst_width, dst_height, PixelFormat::Rgb, DType::U8, None)?;
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
            &format!("{g2d_in_fmt}_to_{g2d_out_fmt}_resized_dst_crop"),
        )
    }

    fn compare_images(
        img1: &TensorDyn,
        img2: &TensorDyn,
        threshold: f64,
        name: &str,
    ) -> Result<(), crate::Error> {
        assert_eq!(img1.height(), img2.height(), "Heights differ");
        assert_eq!(img1.width(), img2.width(), "Widths differ");
        assert_eq!(
            img1.format().unwrap(),
            img2.format().unwrap(),
            "PixelFormat differ"
        );
        assert!(
            matches!(img1.format().unwrap(), PixelFormat::Rgb | PixelFormat::Rgba),
            "format must be Rgb or Rgba for comparison"
        );
        let image1 = match img1.format().unwrap() {
            PixelFormat::Rgb => image::RgbImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                img1.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap(),
            PixelFormat::Rgba => image::RgbaImage::from_vec(
                img1.width().unwrap() as u32,
                img1.height().unwrap() as u32,
                img1.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap()
            .convert(),

            _ => unreachable!(),
        };

        let image2 = match img2.format().unwrap() {
            PixelFormat::Rgb => image::RgbImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                img2.as_u8().unwrap().map().unwrap().to_vec(),
            )
            .unwrap(),
            PixelFormat::Rgba => image::RgbaImage::from_vec(
                img2.width().unwrap() as u32,
                img2.height().unwrap() as u32,
                img2.as_u8().unwrap().map().unwrap().to_vec(),
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
            return Err(Error::Internal(format!(
                "{name}: converted image and target image have similarity score too low: {} < {}",
                similarity.score, threshold
            )));
        }
        Ok(())
    }

    // =========================================================================
    // PixelFormat::Nv12 Reference Validation Tests
    // These tests compare G2D PixelFormat::Nv12 conversions against ffmpeg-generated references
    // =========================================================================

    fn load_raw_image(
        width: usize,
        height: usize,
        format: PixelFormat,
        memory: Option<TensorMemory>,
        bytes: &[u8],
    ) -> Result<TensorDyn, crate::Error> {
        let img = TensorDyn::image(width, height, format, DType::U8, memory)?;
        let mut map = img.as_u8().unwrap().map()?;
        map.as_mut_slice()[..bytes.len()].copy_from_slice(bytes);
        Ok(img)
    }

    /// Test G2D PixelFormat::Nv12→PixelFormat::Rgba conversion against ffmpeg reference
    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_nv12_to_rgba_reference() -> Result<(), crate::Error> {
        if !is_dma_available() {
            return Ok(());
        }
        // Load PixelFormat::Nv12 source
        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.nv12"
            )),
        )?;

        // Load PixelFormat::Rgba reference (ffmpeg-generated)
        let reference = load_raw_image(
            1280,
            720,
            PixelFormat::Rgba,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )),
        )?;

        // Convert using G2D
        let mut dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let mut g2d = G2DProcessor::new()?;
        let src_dyn = src;
        let mut dst_dyn = dst;
        g2d.convert(
            &src_dyn,
            &mut dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        dst = {
            let mut __t = dst_dyn.into_u8().unwrap();
            __t.set_format(PixelFormat::Rgba)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        // Copy to CPU for comparison
        let cpu_dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None)?;
        cpu_dst
            .as_u8()
            .unwrap()
            .map()?
            .as_mut_slice()
            .copy_from_slice(dst.as_u8().unwrap().map()?.as_slice());

        compare_images(&reference, &cpu_dst, 0.98, "g2d_nv12_to_rgba_reference")
    }

    /// Test G2D PixelFormat::Nv12→PixelFormat::Rgb conversion against ffmpeg reference
    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_nv12_to_rgb_reference() -> Result<(), crate::Error> {
        if !is_dma_available() {
            return Ok(());
        }
        // Load PixelFormat::Nv12 source
        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Nv12,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.nv12"
            )),
        )?;

        // Load PixelFormat::Rgb reference (ffmpeg-generated)
        let reference = load_raw_image(
            1280,
            720,
            PixelFormat::Rgb,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgb"
            )),
        )?;

        // Convert using G2D
        let mut dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let mut g2d = G2DProcessor::new()?;
        let src_dyn = src;
        let mut dst_dyn = dst;
        g2d.convert(
            &src_dyn,
            &mut dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        dst = {
            let mut __t = dst_dyn.into_u8().unwrap();
            __t.set_format(PixelFormat::Rgb)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        // Copy to CPU for comparison
        let cpu_dst = TensorDyn::image(1280, 720, PixelFormat::Rgb, DType::U8, None)?;
        cpu_dst
            .as_u8()
            .unwrap()
            .map()?
            .as_mut_slice()
            .copy_from_slice(dst.as_u8().unwrap().map()?.as_slice());

        compare_images(&reference, &cpu_dst, 0.98, "g2d_nv12_to_rgb_reference")
    }

    /// Test G2D PixelFormat::Yuyv→PixelFormat::Rgba conversion against ffmpeg reference
    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_yuyv_to_rgba_reference() -> Result<(), crate::Error> {
        if !is_dma_available() {
            return Ok(());
        }
        // Load PixelFormat::Yuyv source
        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )?;

        // Load PixelFormat::Rgba reference (ffmpeg-generated)
        let reference = load_raw_image(
            1280,
            720,
            PixelFormat::Rgba,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgba"
            )),
        )?;

        // Convert using G2D
        let mut dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let mut g2d = G2DProcessor::new()?;
        let src_dyn = src;
        let mut dst_dyn = dst;
        g2d.convert(
            &src_dyn,
            &mut dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        dst = {
            let mut __t = dst_dyn.into_u8().unwrap();
            __t.set_format(PixelFormat::Rgba)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        // Copy to CPU for comparison
        let cpu_dst = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None)?;
        cpu_dst
            .as_u8()
            .unwrap()
            .map()?
            .as_mut_slice()
            .copy_from_slice(dst.as_u8().unwrap().map()?.as_slice());

        compare_images(&reference, &cpu_dst, 0.98, "g2d_yuyv_to_rgba_reference")
    }

    /// Test G2D PixelFormat::Yuyv→PixelFormat::Rgb conversion against ffmpeg reference
    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_yuyv_to_rgb_reference() -> Result<(), crate::Error> {
        if !is_dma_available() {
            return Ok(());
        }
        // Load PixelFormat::Yuyv source
        let src = load_raw_image(
            1280,
            720,
            PixelFormat::Yuyv,
            Some(TensorMemory::Dma),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.yuyv"
            )),
        )?;

        // Load PixelFormat::Rgb reference (ffmpeg-generated)
        let reference = load_raw_image(
            1280,
            720,
            PixelFormat::Rgb,
            None,
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/camera720p.rgb"
            )),
        )?;

        // Convert using G2D
        let mut dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let mut g2d = G2DProcessor::new()?;
        let src_dyn = src;
        let mut dst_dyn = dst;
        g2d.convert(
            &src_dyn,
            &mut dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        dst = {
            let mut __t = dst_dyn.into_u8().unwrap();
            __t.set_format(PixelFormat::Rgb)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        // Copy to CPU for comparison
        let cpu_dst = TensorDyn::image(1280, 720, PixelFormat::Rgb, DType::U8, None)?;
        cpu_dst
            .as_u8()
            .unwrap()
            .map()?
            .as_mut_slice()
            .copy_from_slice(dst.as_u8().unwrap().map()?.as_slice());

        compare_images(&reference, &cpu_dst, 0.98, "g2d_yuyv_to_rgb_reference")
    }

    /// Test G2D native PixelFormat::Bgra conversion for all supported source formats.
    /// Compares G2D src→PixelFormat::Bgra against G2D src→PixelFormat::Rgba by verifying R↔B swap.
    #[test]
    #[cfg(target_os = "linux")]
    fn test_g2d_bgra_no_resize() {
        for src_fmt in [
            PixelFormat::Rgba,
            PixelFormat::Yuyv,
            PixelFormat::Nv12,
            PixelFormat::Bgra,
        ] {
            test_g2d_bgra_no_resize_(src_fmt).unwrap_or_else(|e| {
                panic!("{src_fmt} to PixelFormat::Bgra failed: {e:?}");
            });
        }
    }

    fn test_g2d_bgra_no_resize_(g2d_in_fmt: PixelFormat) -> Result<(), crate::Error> {
        let file = include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../testdata/zidane.jpg"
        ))
        .to_vec();
        let src = crate::load_image(&file, Some(PixelFormat::Rgb), None)?;

        // Create DMA buffer for G2D input
        let mut src2 = TensorDyn::image(1280, 720, g2d_in_fmt, DType::U8, Some(TensorMemory::Dma))?;
        let mut cpu_converter = CPUProcessor::new();

        if g2d_in_fmt == PixelFormat::Nv12 {
            let nv12_bytes = include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../testdata/zidane.nv12"
            ));
            src2.as_u8()
                .unwrap()
                .map()?
                .as_mut_slice()
                .copy_from_slice(nv12_bytes);
        } else {
            cpu_converter.convert(&src, &mut src2, Rotation::None, Flip::None, Crop::no_crop())?;
        }

        let mut g2d = G2DProcessor::new()?;

        // Convert to PixelFormat::Bgra via G2D
        let mut bgra_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Bgra,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let src2_dyn = src2;
        let mut bgra_dst_dyn = bgra_dst;
        g2d.convert(
            &src2_dyn,
            &mut bgra_dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        bgra_dst = {
            let mut __t = bgra_dst_dyn.into_u8().unwrap();
            __t.set_format(PixelFormat::Bgra)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        // Reconstruct src2 from dyn for PixelFormat::Rgba conversion
        let src2 = {
            let mut __t = src2_dyn.into_u8().unwrap();
            __t.set_format(g2d_in_fmt)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        // Convert to PixelFormat::Rgba via G2D as reference
        let mut rgba_dst = TensorDyn::image(
            1280,
            720,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::Dma),
        )?;
        let src2_dyn2 = src2;
        let mut rgba_dst_dyn = rgba_dst;
        g2d.convert(
            &src2_dyn2,
            &mut rgba_dst_dyn,
            Rotation::None,
            Flip::None,
            Crop::no_crop(),
        )?;
        rgba_dst = {
            let mut __t = rgba_dst_dyn.into_u8().unwrap();
            __t.set_format(PixelFormat::Rgba)
                .map_err(|e| crate::Error::Internal(e.to_string()))?;
            TensorDyn::from(__t)
        };

        // Copy both to CPU memory for comparison
        let bgra_cpu = TensorDyn::image(1280, 720, PixelFormat::Bgra, DType::U8, None)?;
        bgra_cpu
            .as_u8()
            .unwrap()
            .map()?
            .as_mut_slice()
            .copy_from_slice(bgra_dst.as_u8().unwrap().map()?.as_slice());

        let rgba_cpu = TensorDyn::image(1280, 720, PixelFormat::Rgba, DType::U8, None)?;
        rgba_cpu
            .as_u8()
            .unwrap()
            .map()?
            .as_mut_slice()
            .copy_from_slice(rgba_dst.as_u8().unwrap().map()?.as_slice());

        // Verify PixelFormat::Bgra output has R↔B swapped vs PixelFormat::Rgba output
        let bgra_map = bgra_cpu.as_u8().unwrap().map()?;
        let rgba_map = rgba_cpu.as_u8().unwrap().map()?;
        let bgra_buf = bgra_map.as_slice();
        let rgba_buf = rgba_map.as_slice();

        assert_eq!(bgra_buf.len(), rgba_buf.len());
        for (i, (bc, rc)) in bgra_buf
            .chunks_exact(4)
            .zip(rgba_buf.chunks_exact(4))
            .enumerate()
        {
            assert_eq!(
                bc[0], rc[2],
                "{g2d_in_fmt} to PixelFormat::Bgra: pixel {i} B mismatch",
            );
            assert_eq!(
                bc[1], rc[1],
                "{g2d_in_fmt} to PixelFormat::Bgra: pixel {i} G mismatch",
            );
            assert_eq!(
                bc[2], rc[0],
                "{g2d_in_fmt} to PixelFormat::Bgra: pixel {i} R mismatch",
            );
            assert_eq!(
                bc[3], rc[3],
                "{g2d_in_fmt} to PixelFormat::Bgra: pixel {i} A mismatch",
            );
        }
        Ok(())
    }
}
