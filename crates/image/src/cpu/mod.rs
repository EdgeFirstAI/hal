// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use crate::{Crop, Error, Flip, FunctionTimer, ImageProcessorTrait, Rect, Result, Rotation};
use edgefirst_decoder::{DetectBox, ProtoData, Segmentation};
use edgefirst_tensor::{
    DType, PixelFormat, Tensor, TensorDyn, TensorMapTrait, TensorMemory, TensorTrait,
};

mod convert;
mod masks;
mod resize;
mod tests;

// bilinear_dot removed — masks.rs now uses slice-native bilinear_dot_slice
// closure-based kernel, invoked through the local dtype dispatch below.

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

/// Write the base layer of `dst` before mask rendering.
///
/// This is the terminal fallback: on CPU we have no 2D hardware, so a
/// direct buffer write is the appropriate primitive. The invariant is that
/// every call to the CPU draw_* entry points fully initialises dst — we
/// never rely on "whatever was in the buffer" from the caller.
///
/// - `background == Some(bg)` → byte-for-byte copy bg → dst (after shape /
///   format validation).
/// - `background == None` → fill dst with 0x00 (transparent black).
fn prepare_dst_base_cpu(dst: &mut TensorDyn, background: Option<&TensorDyn>) -> Result<()> {
    match background {
        Some(bg) => {
            if bg.shape() != dst.shape() {
                return Err(Error::InvalidShape(
                    "background shape does not match dst".into(),
                ));
            }
            if bg.format() != dst.format() {
                return Err(Error::InvalidShape(
                    "background pixel format does not match dst".into(),
                ));
            }
            let bg_u8 = bg.as_u8().ok_or(Error::NotAnImage)?;
            let dst_u8 = dst.as_u8_mut().ok_or(Error::NotAnImage)?;
            let bg_map = bg_u8.map()?;
            let mut dst_map = dst_u8.map()?;
            let bg_slice = bg_map.as_slice();
            let dst_slice = dst_map.as_mut_slice();
            if bg_slice.len() != dst_slice.len() {
                return Err(Error::InvalidShape(
                    "background buffer size does not match dst".into(),
                ));
            }
            dst_slice.copy_from_slice(bg_slice);
        }
        None => {
            let dst_u8 = dst.as_u8_mut().ok_or(Error::NotAnImage)?;
            let mut dst_map = dst_u8.map()?;
            dst_map.as_mut_slice().fill(0);
        }
    }
    Ok(())
}

/// Compute row stride for a packed-format Tensor<u8> image given its format.
fn row_stride_for(width: usize, fmt: PixelFormat) -> usize {
    use edgefirst_tensor::PixelLayout;
    match fmt.layout() {
        PixelLayout::Packed => width * fmt.channels(),
        PixelLayout::Planar | PixelLayout::SemiPlanar => width,
        _ => width, // fallback for non-exhaustive
    }
}

/// Apply XOR 0x80 bias to color channels only, preserving alpha.
///
/// Matches GL int8 shader behavior: `vec4(int8_bias(c.rgb), c.a)`.
/// For formats without alpha, XORs every byte (fast path).
pub(crate) fn apply_int8_xor_bias(data: &mut [u8], fmt: PixelFormat) {
    use edgefirst_tensor::PixelLayout;
    if !fmt.has_alpha() {
        for b in data.iter_mut() {
            *b ^= 0x80;
        }
    } else if fmt.layout() == PixelLayout::Planar {
        // Planar with alpha (e.g. PlanarRgba): XOR color planes, skip alpha plane.
        let channels = fmt.channels();
        let plane_size = data.len() / channels;
        for b in data[..plane_size * (channels - 1)].iter_mut() {
            *b ^= 0x80;
        }
    } else {
        // Packed with alpha (Rgba, Bgra): XOR color bytes, skip alpha byte.
        let channels = fmt.channels();
        for pixel in data.chunks_exact_mut(channels) {
            for b in &mut pixel[..channels - 1] {
                *b ^= 0x80;
            }
        }
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

    pub(crate) fn support_conversion_pf(src: PixelFormat, dst: PixelFormat) -> bool {
        use PixelFormat::*;
        matches!(
            (src, dst),
            (Nv12, Rgb)
                | (Nv12, Rgba)
                | (Nv12, Grey)
                | (Nv16, Rgb)
                | (Nv16, Rgba)
                | (Nv16, Bgra)
                | (Yuyv, Rgb)
                | (Yuyv, Rgba)
                | (Yuyv, Grey)
                | (Yuyv, Yuyv)
                | (Yuyv, PlanarRgb)
                | (Yuyv, PlanarRgba)
                | (Yuyv, Nv16)
                | (Vyuy, Rgb)
                | (Vyuy, Rgba)
                | (Vyuy, Grey)
                | (Vyuy, Vyuy)
                | (Vyuy, PlanarRgb)
                | (Vyuy, PlanarRgba)
                | (Vyuy, Nv16)
                | (Rgba, Rgb)
                | (Rgba, Rgba)
                | (Rgba, Grey)
                | (Rgba, Yuyv)
                | (Rgba, PlanarRgb)
                | (Rgba, PlanarRgba)
                | (Rgba, Nv16)
                | (Rgb, Rgb)
                | (Rgb, Rgba)
                | (Rgb, Grey)
                | (Rgb, Yuyv)
                | (Rgb, PlanarRgb)
                | (Rgb, PlanarRgba)
                | (Rgb, Nv16)
                | (Grey, Rgb)
                | (Grey, Rgba)
                | (Grey, Grey)
                | (Grey, Yuyv)
                | (Grey, PlanarRgb)
                | (Grey, PlanarRgba)
                | (Grey, Nv16)
                | (Nv12, Bgra)
                | (Yuyv, Bgra)
                | (Vyuy, Bgra)
                | (Rgba, Bgra)
                | (Rgb, Bgra)
                | (Grey, Bgra)
                | (Bgra, Bgra)
                | (PlanarRgb, Rgb)
                | (PlanarRgb, Rgba)
                | (PlanarRgba, Rgb)
                | (PlanarRgba, Rgba)
                | (PlanarRgb, Bgra)
                | (PlanarRgba, Bgra)
        )
    }

    /// Format conversion dispatch for Tensor<u8> with PixelFormat metadata.
    pub(crate) fn convert_format_pf(
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        src_fmt: PixelFormat,
        dst_fmt: PixelFormat,
    ) -> Result<()> {
        let _timer = FunctionTimer::new(format!(
            "ImageProcessor::convert_format {} to {}",
            src_fmt, dst_fmt,
        ));

        use PixelFormat::*;
        match (src_fmt, dst_fmt) {
            (Nv12, Rgb) => Self::convert_nv12_to_rgb(src, dst),
            (Nv12, Rgba) => Self::convert_nv12_to_rgba(src, dst),
            (Nv12, Grey) => Self::convert_nv12_to_grey(src, dst),
            (Yuyv, Rgb) => Self::convert_yuyv_to_rgb(src, dst),
            (Yuyv, Rgba) => Self::convert_yuyv_to_rgba(src, dst),
            (Yuyv, Grey) => Self::convert_yuyv_to_grey(src, dst),
            (Yuyv, Yuyv) => Self::copy_image(src, dst),
            (Yuyv, PlanarRgb) => Self::convert_yuyv_to_8bps(src, dst),
            (Yuyv, PlanarRgba) => Self::convert_yuyv_to_prgba(src, dst),
            (Yuyv, Nv16) => Self::convert_yuyv_to_nv16(src, dst),
            (Vyuy, Rgb) => Self::convert_vyuy_to_rgb(src, dst),
            (Vyuy, Rgba) => Self::convert_vyuy_to_rgba(src, dst),
            (Vyuy, Grey) => Self::convert_vyuy_to_grey(src, dst),
            (Vyuy, Vyuy) => Self::copy_image(src, dst),
            (Vyuy, PlanarRgb) => Self::convert_vyuy_to_8bps(src, dst),
            (Vyuy, PlanarRgba) => Self::convert_vyuy_to_prgba(src, dst),
            (Vyuy, Nv16) => Self::convert_vyuy_to_nv16(src, dst),
            (Rgba, Rgb) => Self::convert_rgba_to_rgb(src, dst),
            (Rgba, Rgba) => Self::copy_image(src, dst),
            (Rgba, Grey) => Self::convert_rgba_to_grey(src, dst),
            (Rgba, Yuyv) => Self::convert_rgba_to_yuyv(src, dst),
            (Rgba, PlanarRgb) => Self::convert_rgba_to_8bps(src, dst),
            (Rgba, PlanarRgba) => Self::convert_rgba_to_prgba(src, dst),
            (Rgba, Nv16) => Self::convert_rgba_to_nv16(src, dst),
            (Rgb, Rgb) => Self::copy_image(src, dst),
            (Rgb, Rgba) => Self::convert_rgb_to_rgba(src, dst),
            (Rgb, Grey) => Self::convert_rgb_to_grey(src, dst),
            (Rgb, Yuyv) => Self::convert_rgb_to_yuyv(src, dst),
            (Rgb, PlanarRgb) => Self::convert_rgb_to_8bps(src, dst),
            (Rgb, PlanarRgba) => Self::convert_rgb_to_prgba(src, dst),
            (Rgb, Nv16) => Self::convert_rgb_to_nv16(src, dst),
            (Grey, Rgb) => Self::convert_grey_to_rgb(src, dst),
            (Grey, Rgba) => Self::convert_grey_to_rgba(src, dst),
            (Grey, Grey) => Self::copy_image(src, dst),
            (Grey, Yuyv) => Self::convert_grey_to_yuyv(src, dst),
            (Grey, PlanarRgb) => Self::convert_grey_to_8bps(src, dst),
            (Grey, PlanarRgba) => Self::convert_grey_to_prgba(src, dst),
            (Grey, Nv16) => Self::convert_grey_to_nv16(src, dst),

            // the following converts are added for use in testing
            (Nv16, Rgb) => Self::convert_nv16_to_rgb(src, dst),
            (Nv16, Rgba) => Self::convert_nv16_to_rgba(src, dst),
            (PlanarRgb, Rgb) => Self::convert_8bps_to_rgb(src, dst),
            (PlanarRgb, Rgba) => Self::convert_8bps_to_rgba(src, dst),
            (PlanarRgba, Rgb) => Self::convert_prgba_to_rgb(src, dst),
            (PlanarRgba, Rgba) => Self::convert_prgba_to_rgba(src, dst),

            // BGRA destination: convert to RGBA layout, then swap R and B
            (Bgra, Bgra) => Self::copy_image(src, dst),
            (Nv12, Bgra) => {
                Self::convert_nv12_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }
            (Nv16, Bgra) => {
                Self::convert_nv16_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }
            (Yuyv, Bgra) => {
                Self::convert_yuyv_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }
            (Vyuy, Bgra) => {
                Self::convert_vyuy_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }
            (Rgba, Bgra) => {
                dst.map()?.copy_from_slice(&src.map()?);
                Self::swizzle_rb_4chan(dst)
            }
            (Rgb, Bgra) => {
                Self::convert_rgb_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }
            (Grey, Bgra) => {
                Self::convert_grey_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }
            (PlanarRgb, Bgra) => {
                Self::convert_8bps_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }
            (PlanarRgba, Bgra) => {
                Self::convert_prgba_to_rgba(src, dst)?;
                Self::swizzle_rb_4chan(dst)
            }

            (s, d) => Err(Error::NotSupported(format!("Conversion from {s} to {d}",))),
        }
    }

    /// Tensor<u8>-based fill_image_outside_crop.
    pub(crate) fn fill_image_outside_crop_u8(
        dst: &mut Tensor<u8>,
        rgba: [u8; 4],
        crop: Rect,
    ) -> Result<()> {
        let dst_fmt = dst.format().unwrap();
        let dst_w = dst.width().unwrap();
        let dst_h = dst.height().unwrap();
        let mut dst_map = dst.map()?;
        let dst_tup = (dst_map.as_mut_slice(), dst_w, dst_h);
        Self::fill_outside_crop_dispatch(dst_tup, dst_fmt, rgba, crop)
    }

    /// Common fill dispatch by format.
    fn fill_outside_crop_dispatch(
        dst: (&mut [u8], usize, usize),
        fmt: PixelFormat,
        rgba: [u8; 4],
        crop: Rect,
    ) -> Result<()> {
        use PixelFormat::*;
        match fmt {
            Rgba | Bgra => Self::fill_image_outside_crop_(dst, rgba, crop),
            Rgb => Self::fill_image_outside_crop_(dst, Self::rgba_to_rgb(rgba), crop),
            Grey => Self::fill_image_outside_crop_(dst, Self::rgba_to_grey(rgba), crop),
            Yuyv => Self::fill_image_outside_crop_(
                (dst.0, dst.1 / 2, dst.2),
                Self::rgba_to_yuyv(rgba),
                Rect::new(crop.left / 2, crop.top, crop.width.div_ceil(2), crop.height),
            ),
            PlanarRgb => Self::fill_image_outside_crop_planar(dst, Self::rgba_to_rgb(rgba), crop),
            PlanarRgba => Self::fill_image_outside_crop_planar(dst, rgba, crop),
            Nv16 => {
                let yuyv = Self::rgba_to_yuyv(rgba);
                Self::fill_image_outside_crop_yuv_semiplanar(dst, yuyv[0], [yuyv[1], yuyv[3]], crop)
            }
            _ => Err(Error::Internal(format!(
                "Found unexpected destination {fmt}",
            ))),
        }
    }
}

impl ImageProcessorTrait for CPUProcessor {
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

    fn draw_decoded_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
        overlay: crate::MaskOverlay<'_>,
    ) -> Result<()> {
        // CPU is the terminal fallback — it must always produce the full
        // output, never assume the caller cleared dst. Every call writes
        // the base layer first (bg copy or zero fill) and then the masks.
        prepare_dst_base_cpu(dst, overlay.background)?;
        let dst = dst.as_u8_mut().ok_or(Error::NotAnImage)?;
        self.draw_decoded_masks_impl(
            dst,
            detect,
            segmentation,
            overlay.opacity,
            overlay.color_mode,
        )
    }

    fn draw_proto_masks(
        &mut self,
        dst: &mut TensorDyn,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        overlay: crate::MaskOverlay<'_>,
    ) -> Result<()> {
        prepare_dst_base_cpu(dst, overlay.background)?;
        let dst = dst.as_u8_mut().ok_or(Error::NotAnImage)?;
        self.draw_proto_masks_impl(
            dst,
            detect,
            proto_data,
            overlay.opacity,
            overlay.letterbox,
            overlay.color_mode,
        )
    }

    fn set_class_colors(&mut self, colors: &[[u8; 4]]) -> Result<()> {
        for (c, new_c) in self.colors.iter_mut().zip(colors.iter()) {
            *c = *new_c;
        }
        Ok(())
    }
}

// Internal methods — dtype-aware dispatch layer.
impl CPUProcessor {
    /// Top-level conversion dispatcher: handles dtype combinations.
    pub(crate) fn convert_impl(
        &mut self,
        src: &TensorDyn,
        dst: &mut TensorDyn,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        let src_fmt = src.format().ok_or(Error::NotAnImage)?;
        let dst_fmt = dst.format().ok_or(Error::NotAnImage)?;

        match (src.dtype(), dst.dtype()) {
            (DType::U8, DType::U8) => {
                let src = src.as_u8().unwrap();
                let dst = dst.as_u8_mut().unwrap();
                self.convert_u8(src, dst, src_fmt, dst_fmt, rotation, flip, crop)
            }
            (DType::U8, DType::I8) => {
                // Int8 output: reinterpret the i8 destination as u8 (layout-
                // identical), convert directly into it, then XOR 0x80 in-place.
                let src_u8 = src.as_u8().unwrap();
                let dst_i8 = dst.as_i8_mut().unwrap();
                // SAFETY: Tensor<i8> and Tensor<u8> are layout-identical
                // (same element size, no T-dependent drop glue). Same
                // rationale as gl::processor::tensor_i8_as_u8_mut.
                let dst_u8 = unsafe { &mut *(dst_i8 as *mut Tensor<i8> as *mut Tensor<u8>) };
                self.convert_u8(src_u8, dst_u8, src_fmt, dst_fmt, rotation, flip, crop)?;
                // Apply XOR 0x80 bias in-place (u8 → i8 conversion)
                let mut map = dst_u8.map()?;
                apply_int8_xor_bias(map.as_mut_slice(), dst_fmt);
                Ok(())
            }
            (s, d) => Err(Error::NotSupported(format!("dtype {s} -> {d}",))),
        }
    }

    /// U8-to-U8 conversion: the full format conversion + resize pipeline.
    #[allow(clippy::too_many_arguments)]
    fn convert_u8(
        &mut self,
        src: &Tensor<u8>,
        dst: &mut Tensor<u8>,
        src_fmt: PixelFormat,
        dst_fmt: PixelFormat,
        rotation: Rotation,
        flip: Flip,
        crop: Crop,
    ) -> Result<()> {
        use PixelFormat::*;

        let src_w = src.width().unwrap();
        let src_h = src.height().unwrap();
        let dst_w = dst.width().unwrap();
        let dst_h = dst.height().unwrap();

        crop.check_crop_dims(src_w, src_h, dst_w, dst_h)?;

        // Determine intermediate format for the resize step
        let intermediate = match (src_fmt, dst_fmt) {
            (Nv12, Rgb) => Rgb,
            (Nv12, Rgba) => Rgba,
            (Nv12, Grey) => Grey,
            (Nv12, Yuyv) => Rgba,
            (Nv12, Nv16) => Rgba,
            (Nv12, PlanarRgb) => Rgb,
            (Nv12, PlanarRgba) => Rgba,
            (Yuyv, Rgb) => Rgb,
            (Yuyv, Rgba) => Rgba,
            (Yuyv, Grey) => Grey,
            (Yuyv, Yuyv) => Rgba,
            (Yuyv, PlanarRgb) => Rgb,
            (Yuyv, PlanarRgba) => Rgba,
            (Yuyv, Nv16) => Rgba,
            (Vyuy, Rgb) => Rgb,
            (Vyuy, Rgba) => Rgba,
            (Vyuy, Grey) => Grey,
            (Vyuy, Vyuy) => Rgba,
            (Vyuy, PlanarRgb) => Rgb,
            (Vyuy, PlanarRgba) => Rgba,
            (Vyuy, Nv16) => Rgba,
            (Rgba, Rgb) => Rgba,
            (Rgba, Rgba) => Rgba,
            (Rgba, Grey) => Grey,
            (Rgba, Yuyv) => Rgba,
            (Rgba, PlanarRgb) => Rgba,
            (Rgba, PlanarRgba) => Rgba,
            (Rgba, Nv16) => Rgba,
            (Rgb, Rgb) => Rgb,
            (Rgb, Rgba) => Rgb,
            (Rgb, Grey) => Grey,
            (Rgb, Yuyv) => Rgb,
            (Rgb, PlanarRgb) => Rgb,
            (Rgb, PlanarRgba) => Rgb,
            (Rgb, Nv16) => Rgb,
            (Grey, Rgb) => Rgb,
            (Grey, Rgba) => Rgba,
            (Grey, Grey) => Grey,
            (Grey, Yuyv) => Grey,
            (Grey, PlanarRgb) => Grey,
            (Grey, PlanarRgba) => Grey,
            (Grey, Nv16) => Grey,
            (Nv12, Bgra) => Rgba,
            (Yuyv, Bgra) => Rgba,
            (Vyuy, Bgra) => Rgba,
            (Rgba, Bgra) => Rgba,
            (Rgb, Bgra) => Rgb,
            (Grey, Bgra) => Grey,
            (Bgra, Bgra) => Bgra,
            (Nv16, Rgb) => Rgb,
            (Nv16, Rgba) => Rgba,
            (Nv16, Bgra) => Rgba,
            (PlanarRgb, Rgb) => Rgb,
            (PlanarRgb, Rgba) => Rgb,
            (PlanarRgb, Bgra) => Rgb,
            (PlanarRgba, Rgb) => Rgba,
            (PlanarRgba, Rgba) => Rgba,
            (PlanarRgba, Bgra) => Rgba,
            (s, d) => {
                return Err(Error::NotSupported(format!("Conversion from {s} to {d}",)));
            }
        };

        let need_resize_flip_rotation = rotation != Rotation::None
            || flip != Flip::None
            || src_w != dst_w
            || src_h != dst_h
            || crop.src_rect.is_some_and(|c| {
                c != Rect {
                    left: 0,
                    top: 0,
                    width: src_w,
                    height: src_h,
                }
            })
            || crop.dst_rect.is_some_and(|c| {
                c != Rect {
                    left: 0,
                    top: 0,
                    width: dst_w,
                    height: dst_h,
                }
            });

        // check if a direct conversion can be done
        if !need_resize_flip_rotation && Self::support_conversion_pf(src_fmt, dst_fmt) {
            return Self::convert_format_pf(src, dst, src_fmt, dst_fmt);
        }

        // any extra checks
        if dst_fmt == Yuyv && !dst_w.is_multiple_of(2) {
            return Err(Error::NotSupported(format!(
                "{} destination must have width divisible by 2",
                dst_fmt,
            )));
        }

        // create tmp buffer
        let mut tmp_buffer;
        let tmp;
        let tmp_fmt;
        if intermediate != src_fmt {
            tmp_buffer = Tensor::<u8>::image(src_w, src_h, intermediate, Some(TensorMemory::Mem))?;

            Self::convert_format_pf(src, &mut tmp_buffer, src_fmt, intermediate)?;
            tmp = &tmp_buffer;
            tmp_fmt = intermediate;
        } else {
            tmp = src;
            tmp_fmt = src_fmt;
        }

        // format must be RGB/RGBA/GREY
        debug_assert!(matches!(tmp_fmt, Rgb | Rgba | Grey));
        if tmp_fmt == dst_fmt {
            self.resize_flip_rotate_pf(tmp, dst, dst_fmt, rotation, flip, crop)?;
        } else if !need_resize_flip_rotation {
            Self::convert_format_pf(tmp, dst, tmp_fmt, dst_fmt)?;
        } else {
            let mut tmp2 = Tensor::<u8>::image(dst_w, dst_h, tmp_fmt, Some(TensorMemory::Mem))?;
            if crop.dst_rect.is_some_and(|c| {
                c != Rect {
                    left: 0,
                    top: 0,
                    width: dst_w,
                    height: dst_h,
                }
            }) && crop.dst_color.is_none()
            {
                Self::convert_format_pf(dst, &mut tmp2, dst_fmt, tmp_fmt)?;
            }
            self.resize_flip_rotate_pf(tmp, &mut tmp2, tmp_fmt, rotation, flip, crop)?;
            Self::convert_format_pf(&tmp2, dst, tmp_fmt, dst_fmt)?;
        }
        if let (Some(dst_rect), Some(dst_color)) = (crop.dst_rect, crop.dst_color) {
            let full_rect = Rect {
                left: 0,
                top: 0,
                width: dst_w,
                height: dst_h,
            };
            if dst_rect != full_rect {
                Self::fill_image_outside_crop_u8(dst, dst_color, dst_rect)?;
            }
        }

        Ok(())
    }

    fn draw_decoded_masks_impl(
        &mut self,
        dst: &mut Tensor<u8>,
        detect: &[DetectBox],
        segmentation: &[Segmentation],
        opacity: f32,
        color_mode: crate::ColorMode,
    ) -> Result<()> {
        let dst_fmt = dst.format().ok_or(Error::NotAnImage)?;
        if !matches!(dst_fmt, PixelFormat::Rgba | PixelFormat::Rgb) {
            return Err(crate::Error::NotSupported(
                "CPU image rendering only supports RGBA or RGB images".to_string(),
            ));
        }

        let _timer = FunctionTimer::new("CPUProcessor::draw_decoded_masks");

        let dst_w = dst.width().unwrap();
        let dst_h = dst.height().unwrap();
        let dst_rs = row_stride_for(dst_w, dst_fmt);
        let dst_c = dst_fmt.channels();

        let mut map = dst.map()?;
        let dst_slice = map.as_mut_slice();

        self.render_box(dst_w, dst_h, dst_rs, dst_c, dst_slice, detect, color_mode)?;

        if segmentation.is_empty() {
            return Ok(());
        }

        // Semantic segmentation (e.g. ModelPack) has C > 1 (multi-class),
        // instance segmentation (e.g. YOLO) has C = 1 (binary per-instance).
        let is_semantic = segmentation[0].segmentation.shape()[2] > 1;

        if is_semantic {
            self.render_modelpack_segmentation(
                dst_w,
                dst_h,
                dst_rs,
                dst_c,
                dst_slice,
                &segmentation[0],
                opacity,
            )?;
        } else {
            for (idx, (seg, det)) in segmentation.iter().zip(detect).enumerate() {
                let color_index = color_mode.index(idx, det.label);
                self.render_yolo_segmentation(
                    dst_w,
                    dst_h,
                    dst_rs,
                    dst_c,
                    dst_slice,
                    seg,
                    color_index,
                    opacity,
                )?;
            }
        }

        Ok(())
    }

    fn draw_proto_masks_impl(
        &mut self,
        dst: &mut Tensor<u8>,
        detect: &[DetectBox],
        proto_data: &ProtoData,
        opacity: f32,
        letterbox: Option<[f32; 4]>,
        color_mode: crate::ColorMode,
    ) -> Result<()> {
        let dst_fmt = dst.format().ok_or(Error::NotAnImage)?;
        if !matches!(dst_fmt, PixelFormat::Rgba | PixelFormat::Rgb) {
            return Err(crate::Error::NotSupported(
                "CPU image rendering only supports RGBA or RGB images".to_string(),
            ));
        }

        let _timer = FunctionTimer::new("CPUProcessor::draw_proto_masks");

        let dst_w = dst.width().unwrap();
        let dst_h = dst.height().unwrap();
        let dst_rs = row_stride_for(dst_w, dst_fmt);
        let channels = dst_fmt.channels();

        let mut map = dst.map()?;
        let dst_slice = map.as_mut_slice();

        self.render_box(
            dst_w, dst_h, dst_rs, channels, dst_slice, detect, color_mode,
        )?;

        if detect.is_empty() {
            return Ok(());
        }
        let proto_shape = proto_data.protos.shape();
        if proto_shape.len() != 3 {
            return Err(Error::InvalidShape(format!(
                "protos tensor must be rank-3, got {proto_shape:?}"
            )));
        }
        let proto_h = proto_shape[0];
        let proto_w = proto_shape[1];
        let num_protos = proto_shape[2];
        let coeff_shape = proto_data.mask_coefficients.shape();
        if coeff_shape.len() != 2 || coeff_shape[1] != num_protos || coeff_shape[0] == 0 {
            return Ok(());
        }

        // Widen coefficients to f32 once; shape [N, num_protos].
        let coeff_f32: Vec<f32> = match proto_data.mask_coefficients.dtype() {
            DType::F32 => {
                let t = proto_data.mask_coefficients.as_f32().expect("F32");
                let m = t.map()?;
                m.as_slice().to_vec()
            }
            DType::F16 => {
                let t = proto_data.mask_coefficients.as_f16().expect("F16");
                let m = t.map()?;
                m.as_slice().iter().map(|v| v.to_f32()).collect()
            }
            other => {
                return Err(Error::InvalidShape(format!(
                    "mask_coefficients dtype {other:?} not supported"
                )));
            }
        };

        // Precompute letterbox scale/offset for output-pixel → proto-pixel mapping.
        let (lx0, lx_range, ly0, ly_range) = match letterbox {
            Some([lx0, ly0, lx1, ly1]) => (lx0, lx1 - lx0, ly0, ly1 - ly0),
            None => (0.0_f32, 1.0_f32, 0.0_f32, 1.0_f32),
        };

        // Per-dtype dispatch. Map protos once, call the inner draw loop
        // with a dtype-specialized loader closure.
        match proto_data.protos.dtype() {
            DType::F32 => {
                let t = proto_data.protos.as_f32().expect("F32");
                let m = t.map()?;
                self.draw_proto_masks_inner(
                    dst_slice,
                    dst_w,
                    dst_h,
                    dst_rs,
                    channels,
                    detect,
                    m.as_slice(),
                    &coeff_f32,
                    proto_h,
                    proto_w,
                    num_protos,
                    opacity,
                    (lx0, lx_range, ly0, ly_range),
                    color_mode,
                    0.0_f32,
                    |p: &f32, _| *p,
                );
            }
            DType::F16 => {
                let t = proto_data.protos.as_f16().expect("F16");
                let m = t.map()?;
                self.draw_proto_masks_inner(
                    dst_slice,
                    dst_w,
                    dst_h,
                    dst_rs,
                    channels,
                    detect,
                    m.as_slice(),
                    &coeff_f32,
                    proto_h,
                    proto_w,
                    num_protos,
                    opacity,
                    (lx0, lx_range, ly0, ly_range),
                    color_mode,
                    0.0_f32,
                    |p: &half::f16, _| p.to_f32(),
                );
            }
            DType::I8 => {
                use edgefirst_tensor::QuantMode;
                let t = proto_data.protos.as_i8().expect("I8");
                let m = t.map()?;
                let quant = t.quantization().ok_or_else(|| {
                    Error::InvalidShape("I8 protos require quantization metadata".into())
                })?;
                let (scale, zp) = match quant.mode() {
                    QuantMode::PerTensor { scale, zero_point } => (scale, zero_point as f32),
                    QuantMode::PerTensorSymmetric { scale } => (scale, 0.0),
                    QuantMode::PerChannel { axis, .. }
                    | QuantMode::PerChannelSymmetric { axis, .. } => {
                        return Err(Error::InvalidShape(format!(
                            "per-channel quantization (axis={axis}) in draw_proto_masks \
                             CPU path not yet supported"
                        )));
                    }
                };
                self.draw_proto_masks_inner(
                    dst_slice,
                    dst_w,
                    dst_h,
                    dst_rs,
                    channels,
                    detect,
                    m.as_slice(),
                    &coeff_f32,
                    proto_h,
                    proto_w,
                    num_protos,
                    opacity,
                    (lx0, lx_range, ly0, ly_range),
                    color_mode,
                    scale,
                    move |p: &i8, _| (*p as f32) - zp,
                );
            }
            other => {
                return Err(Error::InvalidShape(format!(
                    "proto tensor dtype {other:?} not supported"
                )));
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn draw_proto_masks_inner<P: Copy>(
        &self,
        dst_slice: &mut [u8],
        dst_w: usize,
        dst_h: usize,
        dst_rs: usize,
        channels: usize,
        detect: &[DetectBox],
        protos: &[P],
        coeff_all_f32: &[f32],
        proto_h: usize,
        proto_w: usize,
        num_protos: usize,
        opacity: f32,
        letterbox_xy: (f32, f32, f32, f32),
        color_mode: crate::ColorMode,
        acc_scale: f32,
        load_f32: impl Fn(&P, f32) -> f32 + Copy,
    ) {
        let (lx0, lx_range, ly0, ly_range) = letterbox_xy;
        let stride_y = proto_w * num_protos;
        for (idx, det) in detect.iter().enumerate() {
            let coeff = &coeff_all_f32[idx * num_protos..(idx + 1) * num_protos];
            let color_index = color_mode.index(idx, det.label);
            let color = self.colors[color_index % self.colors.len()];
            let alpha = if opacity == 1.0 {
                color[3] as u16
            } else {
                (color[3] as f32 * opacity).round() as u16
            };

            let start_x = (dst_w as f32 * det.bbox.xmin).round() as usize;
            let start_y = (dst_h as f32 * det.bbox.ymin).round() as usize;
            let end_x = ((dst_w as f32 * det.bbox.xmax).round() as usize).min(dst_w);
            let end_y = ((dst_h as f32 * det.bbox.ymax).round() as usize).min(dst_h);

            for y in start_y..end_y {
                for x in start_x..end_x {
                    let px = (lx0 + (x as f32 / dst_w as f32) * lx_range) * proto_w as f32 - 0.5;
                    let py = (ly0 + (y as f32 / dst_h as f32) * ly_range) * proto_h as f32 - 0.5;

                    // Bilinear interpolation with per-load widening. Inline
                    // bilinear-sample since bilinear_dot_slice takes a
                    // different closure shape (no `zp` arg).
                    let x0 = (px.floor() as isize).clamp(0, proto_w as isize - 1) as usize;
                    let y0 = (py.floor() as isize).clamp(0, proto_h as isize - 1) as usize;
                    let x1 = (x0 + 1).min(proto_w - 1);
                    let y1 = (y0 + 1).min(proto_h - 1);
                    let fx = px - px.floor();
                    let fy = py - py.floor();
                    let w00 = (1.0 - fx) * (1.0 - fy);
                    let w10 = fx * (1.0 - fy);
                    let w01 = (1.0 - fx) * fy;
                    let w11 = fx * fy;
                    let b00 = y0 * stride_y + x0 * num_protos;
                    let b10 = y0 * stride_y + x1 * num_protos;
                    let b01 = y1 * stride_y + x0 * num_protos;
                    let b11 = y1 * stride_y + x1 * num_protos;
                    let mut acc = 0.0_f32;
                    for p in 0..num_protos {
                        let v00 = load_f32(&protos[b00 + p], 0.0);
                        let v10 = load_f32(&protos[b10 + p], 0.0);
                        let v01 = load_f32(&protos[b01 + p], 0.0);
                        let v11 = load_f32(&protos[b11 + p], 0.0);
                        let val = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
                        acc += coeff[p] * val;
                    }
                    let final_acc = if acc_scale == 0.0 {
                        acc
                    } else {
                        acc_scale * acc
                    };
                    // Pass-through: acc_scale=0.0 means "no scaling" (f32/f16
                    // native); non-zero means "apply scale once" (i8 with
                    // per-tensor quant).
                    let mask = 1.0 / (1.0 + (-final_acc).exp());
                    if mask < 0.5 {
                        continue;
                    }
                    let dst_index = y * dst_rs + x * channels;
                    for c in 0..3 {
                        dst_slice[dst_index + c] = ((color[c] as u16 * alpha
                            + dst_slice[dst_index + c] as u16 * (255 - alpha))
                            / 255) as u8;
                    }
                }
            }
        }
    }
}
