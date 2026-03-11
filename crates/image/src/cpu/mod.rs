// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{
    fourcc_is_int8, fourcc_uint8_equivalent, Crop, Error, Flip, FunctionTimer, ImageProcessorTrait,
    Rect, Result, Rotation, TensorImage, TensorImageDst, TensorImageRef, BGRA, GREY, NV12, NV16,
    PLANAR_RGB, PLANAR_RGBA, RGB, RGBA, VYUY, YUYV,
};
use edgefirst_decoder::{DetectBox, ProtoData, Segmentation};
use edgefirst_tensor::{TensorMapTrait, TensorTrait};
use four_char_code::FourCharCode;

mod convert;
mod masks;
mod resize;
mod tests;

use masks::bilinear_dot;

/// CPUConverter implements the ImageProcessor trait using the fallback CPU
/// implementation for image processing.
#[derive(Debug, Clone)]
pub struct CPUProcessor {
    resizer: fast_image_resize::Resizer,
    options: fast_image_resize::ResizeOptions,
    colors: [[u8; 4]; 20],
}

unsafe impl Send for CPUProcessor {}
unsafe impl Sync for CPUProcessor {}

impl Default for CPUProcessor {
    fn default() -> Self {
        Self::new_bilinear()
    }
}

impl CPUProcessor {
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
        Self {
            resizer,
            options,
            colors: crate::DEFAULT_COLORS_U8,
        }
    }

    /// Creates a new CPUConverter with nearest neighbor resizing.
    pub fn new_nearest() -> Self {
        let resizer = fast_image_resize::Resizer::new();
        let options = fast_image_resize::ResizeOptions::new()
            .resize_alg(fast_image_resize::ResizeAlg::Nearest)
            .use_alpha(false);
        log::debug!("CPUConverter created");
        Self {
            resizer,
            options,
            colors: crate::DEFAULT_COLORS_U8,
        }
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
                | (VYUY, RGB)
                | (VYUY, RGBA)
                | (VYUY, GREY)
                | (VYUY, VYUY)
                | (VYUY, PLANAR_RGB)
                | (VYUY, PLANAR_RGBA)
                | (VYUY, NV16)
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
                | (NV12, BGRA)
                | (YUYV, BGRA)
                | (VYUY, BGRA)
                | (RGBA, BGRA)
                | (RGB, BGRA)
                | (GREY, BGRA)
                | (BGRA, BGRA)
        )
    }

    pub(crate) fn convert_format(src: &TensorImage, dst: &mut TensorImage) -> Result<()> {
        // shapes should be equal
        let _timer = FunctionTimer::new(format!(
            "ImageProcessor::convert_format {} to {}",
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
            (VYUY, RGB) => Self::convert_vyuy_to_rgb(src, dst),
            (VYUY, RGBA) => Self::convert_vyuy_to_rgba(src, dst),
            (VYUY, GREY) => Self::convert_vyuy_to_grey(src, dst),
            (VYUY, VYUY) => Self::copy_image(src, dst),
            (VYUY, PLANAR_RGB) => Self::convert_vyuy_to_8bps(src, dst),
            (VYUY, PLANAR_RGBA) => Self::convert_vyuy_to_prgba(src, dst),
            (VYUY, NV16) => Self::convert_vyuy_to_nv16(src, dst),
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

            // BGRA destination: convert to RGBA layout, then swap R and B
            (BGRA, BGRA) => Self::copy_image(src, dst),
            (NV12, BGRA) => {
                Self::convert_nv12_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }
            (NV16, BGRA) => {
                Self::convert_nv16_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }
            (YUYV, BGRA) => {
                Self::convert_yuyv_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }
            (VYUY, BGRA) => {
                Self::convert_vyuy_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }
            (RGBA, BGRA) => {
                dst.tensor().map()?.copy_from_slice(&src.tensor().map()?);
                Self::swizzle_rb_4chan(dst)
            }
            (RGB, BGRA) => {
                Self::convert_rgb_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }
            (GREY, BGRA) => {
                Self::convert_grey_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }
            (PLANAR_RGB, BGRA) => {
                Self::convert_8bps_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }
            (PLANAR_RGBA, BGRA) => {
                Self::convert_prgba_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }

            (s, d) => Err(Error::NotSupported(format!(
                "Conversion from {} to {}",
                s.display(),
                d.display()
            ))),
        }
    }

    /// Generic copy for same-format images that works with any TensorImageDst.
    fn copy_image_generic<D: TensorImageDst>(src: &TensorImage, dst: &mut D) -> Result<()> {
        assert_eq!(src.fourcc(), dst.fourcc());
        dst.tensor_mut()
            .map()?
            .copy_from_slice(&src.tensor().map()?);
        Ok(())
    }

    /// Format conversion that writes to a generic TensorImageDst.
    /// Supports common zero-copy preprocessing cases.
    pub(crate) fn convert_format_generic<D: TensorImageDst>(
        src: &TensorImage,
        dst: &mut D,
    ) -> Result<()> {
        let _timer = FunctionTimer::new(format!(
            "ImageProcessor::convert_format_generic {} to {}",
            src.fourcc().display(),
            dst.fourcc().display()
        ));
        assert_eq!(src.height(), dst.height());
        assert_eq!(src.width(), dst.width());

        match (src.fourcc(), dst.fourcc()) {
            (RGB, PLANAR_RGB) => Self::convert_rgb_to_planar_rgb_generic(src, dst),
            (RGBA, PLANAR_RGB) => Self::convert_rgba_to_planar_rgb_generic(src, dst),
            (f1, f2) if f1 == f2 => Self::copy_image_generic(src, dst),
            (s, d) => Err(Error::NotSupported(format!(
                "Generic conversion from {} to {} not supported",
                s.display(),
                d.display()
            ))),
        }
    }
}

impl ImageProcessorTrait for CPUProcessor {
    fn convert(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImage,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        // Int8 formats: convert directly into dst as uint8 (layouts are
        // identical), then XOR 0x80 in-place. Avoids a temporary allocation.
        if fourcc_is_int8(dst.fourcc()) {
            let int8_fourcc = dst.fourcc();
            dst.set_fourcc(fourcc_uint8_equivalent(int8_fourcc));
            if let Err(e) = self.convert(src, dst, rotation, flip, crop) {
                dst.set_fourcc(int8_fourcc);
                return Err(e);
            }
            dst.set_fourcc(int8_fourcc);
            let mut dst_map = dst.tensor().map()?;
            for byte in dst_map.iter_mut() {
                *byte ^= 0x80;
            }
            return Ok(());
        }

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
            (VYUY, RGB) => RGB,
            (VYUY, RGBA) => RGBA,
            (VYUY, GREY) => GREY,
            (VYUY, VYUY) => RGBA, // RGBA intermediary for VYUY dest resize/convert/rotation/flip
            (VYUY, PLANAR_RGB) => RGB,
            (VYUY, PLANAR_RGBA) => RGBA,
            (VYUY, NV16) => RGBA,
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
            (NV12, BGRA) => RGBA,
            (YUYV, BGRA) => RGBA,
            (VYUY, BGRA) => RGBA,
            (RGBA, BGRA) => RGBA,
            (RGB, BGRA) => RGB,
            (GREY, BGRA) => GREY,
            (BGRA, BGRA) => BGRA,
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
        debug_assert!(matches!(tmp.fourcc(), RGB | RGBA | GREY));
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
        if let (Some(dst_rect), Some(dst_color)) = (crop.dst_rect, crop.dst_color) {
            let full_rect = Rect {
                left: 0,
                top: 0,
                width: dst.width(),
                height: dst.height(),
            };
            if dst_rect != full_rect {
                Self::fill_image_outside_crop(dst, dst_color, dst_rect)?;
            }
        }

        Ok(())
    }

    fn convert_ref(
        &mut self,
        src: &TensorImage,
        dst: &mut TensorImageRef<'_>,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        crop.check_crop_ref(src, dst)?;

        // Determine intermediate format needed for conversion
        let intermediate = match (src.fourcc(), dst.fourcc()) {
            (NV12, RGB) => RGB,
            (NV12, RGBA) => RGBA,
            (NV12, GREY) => GREY,
            (NV12, PLANAR_RGB) => RGB,
            (NV12, PLANAR_RGBA) => RGBA,
            (YUYV, RGB) => RGB,
            (YUYV, RGBA) => RGBA,
            (YUYV, GREY) => GREY,
            (YUYV, PLANAR_RGB) => RGB,
            (YUYV, PLANAR_RGBA) => RGBA,
            (VYUY, RGB) => RGB,
            (VYUY, RGBA) => RGBA,
            (VYUY, GREY) => GREY,
            (VYUY, PLANAR_RGB) => RGB,
            (VYUY, PLANAR_RGBA) => RGBA,
            (RGBA, RGB) => RGBA,
            (RGBA, RGBA) => RGBA,
            (RGBA, GREY) => GREY,
            (RGBA, PLANAR_RGB) => RGBA,
            (RGBA, PLANAR_RGBA) => RGBA,
            (RGB, RGB) => RGB,
            (RGB, RGBA) => RGB,
            (RGB, GREY) => GREY,
            (RGB, PLANAR_RGB) => RGB,
            (RGB, PLANAR_RGBA) => RGB,
            (GREY, RGB) => RGB,
            (GREY, RGBA) => RGBA,
            (GREY, GREY) => GREY,
            (GREY, PLANAR_RGB) => GREY,
            (GREY, PLANAR_RGBA) => GREY,
            (s, d) => {
                return Err(Error::NotSupported(format!(
                    "Conversion from {} to {}",
                    s.display(),
                    d.display()
                )));
            }
        };

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

        // Simple case: no resize/flip/rotation needed
        if !need_resize_flip_rotation {
            // Try direct generic conversion (zero-copy path)
            if let Ok(()) = Self::convert_format_generic(src, dst) {
                return Ok(());
            }
        }

        // Complex case: need intermediate buffers
        // First, convert source to intermediate format if needed
        let mut tmp_buffer;
        let tmp: &TensorImage;
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

        // Process resize/flip/rotation if needed
        if need_resize_flip_rotation {
            // Create intermediate buffer for resize output
            let mut tmp2 = TensorImage::new(
                dst.width(),
                dst.height(),
                tmp.fourcc(),
                Some(edgefirst_tensor::TensorMemory::Mem),
            )?;
            self.resize_flip_rotate(tmp, &mut tmp2, rotation, flip, crop)?;

            // Final conversion to destination (zero-copy into dst)
            Self::convert_format_generic(&tmp2, dst)?;
        } else {
            // Direct conversion (already checked above, but handle edge cases)
            Self::convert_format_generic(tmp, dst)?;
        }

        // Handle destination crop fill if needed
        if let (Some(dst_rect), Some(dst_color)) = (crop.dst_rect, crop.dst_color) {
            let full_rect = Rect {
                left: 0,
                top: 0,
                width: dst.width(),
                height: dst.height(),
            };
            if dst_rect != full_rect {
                Self::fill_image_outside_crop_generic(dst, dst_color, dst_rect)?;
            }
        }

        Ok(())
    }

    fn draw_masks(
        &mut self,
        dst: &mut TensorImage,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
    ) -> Result<()> {
        if !matches!(dst.fourcc(), RGBA | RGB) {
            return Err(crate::Error::NotSupported(
                "CPU image rendering only supports RGBA or RGB images".to_string(),
            ));
        }

        let _timer = FunctionTimer::new("CPUProcessor::draw_masks");

        let mut map = dst.tensor.map()?;
        let dst_slice = map.as_mut_slice();

        self.render_box(dst, dst_slice, detect)?;

        if segmentation.is_empty() {
            return Ok(());
        }

        // Semantic segmentation (e.g. ModelPack) has C > 1 (multi-class),
        // instance segmentation (e.g. YOLO) has C = 1 (binary per-instance).
        let is_semantic = segmentation[0].segmentation.shape()[2] > 1;

        if is_semantic {
            self.render_modelpack_segmentation(dst, dst_slice, &segmentation[0])?;
        } else {
            for (seg, detect) in segmentation.iter().zip(detect) {
                self.render_yolo_segmentation(dst, dst_slice, seg, detect.label)?;
            }
        }

        Ok(())
    }

    fn draw_masks_proto(
        &mut self,
        dst: &mut TensorImage,
        detect: &[DetectBox],
        proto_data: &ProtoData,
    ) -> Result<()> {
        if !matches!(dst.fourcc(), RGBA | RGB) {
            return Err(crate::Error::NotSupported(
                "CPU image rendering only supports RGBA or RGB images".to_string(),
            ));
        }

        let _timer = FunctionTimer::new("CPUProcessor::draw_masks_proto");

        let mut map = dst.tensor.map()?;
        let dst_slice = map.as_mut_slice();

        self.render_box(dst, dst_slice, detect)?;

        if detect.is_empty() || proto_data.mask_coefficients.is_empty() {
            return Ok(());
        }

        let protos_cow = proto_data.protos.as_f32();
        let protos = protos_cow.as_ref();
        let proto_h = protos.shape()[0];
        let proto_w = protos.shape()[1];
        let num_protos = protos.shape()[2];
        let dst_w = dst.width();
        let dst_h = dst.height();
        let row_stride = dst.row_stride();
        let channels = dst.channels();

        for (det, coeff) in detect.iter().zip(proto_data.mask_coefficients.iter()) {
            let color = self.colors[det.label % self.colors.len()];
            let alpha = color[3] as u16;

            // Pixel bounds of the detection in dst image space
            let start_x = (dst_w as f32 * det.bbox.xmin).round() as usize;
            let start_y = (dst_h as f32 * det.bbox.ymin).round() as usize;
            let end_x = ((dst_w as f32 * det.bbox.xmax).round() as usize).min(dst_w);
            let end_y = ((dst_h as f32 * det.bbox.ymax).round() as usize).min(dst_h);

            for y in start_y..end_y {
                for x in start_x..end_x {
                    // Map pixel (x, y) to proto space
                    let px = (x as f32 / dst_w as f32) * proto_w as f32 - 0.5;
                    let py = (y as f32 / dst_h as f32) * proto_h as f32 - 0.5;

                    // Bilinear interpolation + dot product
                    let acc = bilinear_dot(protos, coeff, num_protos, px, py, proto_w, proto_h);

                    // Sigmoid threshold
                    let mask = 1.0 / (1.0 + (-acc).exp());
                    if mask < 0.5 {
                        continue;
                    }

                    // Alpha blend
                    let dst_index = y * row_stride + x * channels;
                    for c in 0..3 {
                        dst_slice[dst_index + c] = ((color[c] as u16 * alpha
                            + dst_slice[dst_index + c] as u16 * (255 - alpha))
                            / 255) as u8;
                    }
                }
            }
        }

        Ok(())
    }

    fn decode_masks_atlas(
        &mut self,
        detect: &[crate::DetectBox],
        proto_data: crate::ProtoData,
        output_width: usize,
        output_height: usize,
    ) -> Result<(Vec<u8>, Vec<crate::MaskRegion>)> {
        use crate::FunctionTimer;

        let _timer = FunctionTimer::new("CPUProcessor::decode_masks_atlas");

        let padding = 4usize;

        // Render per-detection masks via existing path
        let mask_results =
            self.render_masks_from_protos(detect, proto_data, output_width, output_height)?;

        // Pack into compact atlas: each strip is padded bbox height
        let ow = output_width as i32;
        let oh = output_height as i32;
        let pad = padding as i32;

        let mut regions = Vec::with_capacity(mask_results.len());
        let mut atlas_y = 0usize;

        // Pre-compute regions
        for mr in &mask_results {
            let bx = mr.x as i32;
            let by = mr.y as i32;
            let bw = mr.w as i32;
            let bh = mr.h as i32;
            let padded_x = (bx - pad).max(0);
            let padded_y = (by - pad).max(0);
            let padded_w = ((bx + bw + pad).min(ow) - padded_x).max(1);
            let padded_h = ((by + bh + pad).min(oh) - padded_y).max(1);
            regions.push(crate::MaskRegion {
                atlas_y_offset: atlas_y,
                padded_x: padded_x as usize,
                padded_y: padded_y as usize,
                padded_w: padded_w as usize,
                padded_h: padded_h as usize,
                bbox_x: mr.x,
                bbox_y: mr.y,
                bbox_w: mr.w,
                bbox_h: mr.h,
            });
            atlas_y += padded_h as usize;
        }

        let atlas_height = atlas_y;
        let mut atlas = vec![0u8; output_width * atlas_height];

        for (mr, region) in mask_results.iter().zip(regions.iter()) {
            // Copy mask pixels into the atlas at the correct position
            for row in 0..mr.h {
                let dst_row = region.atlas_y_offset + (mr.y - region.padded_y) + row;
                let dst_start = dst_row * output_width + mr.x;
                let src_start = row * mr.w;
                if dst_start + mr.w <= atlas.len() && src_start + mr.w <= mr.pixels.len() {
                    atlas[dst_start..dst_start + mr.w]
                        .copy_from_slice(&mr.pixels[src_start..src_start + mr.w]);
                }
            }
        }

        Ok((atlas, regions))
    }

    fn set_class_colors(&mut self, colors: &[[u8; 4]]) -> Result<()> {
        for (c, new_c) in self.colors.iter_mut().zip(colors.iter()) {
            *c = *new_c;
        }
        Ok(())
    }
}
