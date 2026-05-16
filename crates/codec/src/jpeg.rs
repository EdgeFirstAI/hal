// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! JPEG decoding into pre-allocated tensors (Phase 1: zune-jpeg shim).

use crate::error::CodecError;
use crate::options::{DecodeOptions, ImageInfo};
use crate::pixel::ImagePixel;
use edgefirst_tensor::{PixelFormat, Tensor, TensorTrait};
use zune_jpeg::zune_core::colorspace::ColorSpace;
use zune_jpeg::zune_core::options::DecoderOptions;
use zune_jpeg::JpegDecoder;

/// Map a [`PixelFormat`] to zune's [`ColorSpace`].
fn pixelfmt_to_colorspace(fmt: PixelFormat) -> Option<ColorSpace> {
    match fmt {
        PixelFormat::Rgb => Some(ColorSpace::RGB),
        PixelFormat::Rgba => Some(ColorSpace::RGBA),
        PixelFormat::Grey => Some(ColorSpace::Luma),
        PixelFormat::Bgra => Some(ColorSpace::BGRA),
        _ => None,
    }
}

/// Map zune's [`ColorSpace`] back to a [`PixelFormat`].
fn colorspace_to_pixelfmt(cs: ColorSpace) -> Option<PixelFormat> {
    match cs {
        ColorSpace::RGB => Some(PixelFormat::Rgb),
        ColorSpace::RGBA => Some(PixelFormat::Rgba),
        ColorSpace::Luma => Some(PixelFormat::Grey),
        ColorSpace::BGRA => Some(PixelFormat::Bgra),
        _ => None,
    }
}

/// Read EXIF orientation tag and return (rotation_degrees, flip_horizontal).
fn read_exif_orientation(exif_data: &[u8]) -> (u16, bool) {
    let reader = exif::Reader::new();
    let Ok(exif) = reader.read_raw(exif_data.to_vec()) else {
        return (0, false);
    };
    let Some(orient) = exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY) else {
        return (0, false);
    };
    match orient.value.get_uint(0).unwrap_or(1) {
        1 => (0, false),
        2 => (0, true),
        3 => (180, false),
        4 => (180, true),
        5 => (270, true),
        6 => (90, false),
        7 => (90, true),
        8 => (270, false),
        _ => (0, false),
    }
}

/// Rotate pixel data 90° clockwise in-place (using scratch buffer).
fn rotate_90_cw(data: &mut [u8], w: usize, h: usize, channels: usize, scratch: &mut Vec<u8>) {
    let new_w = h;
    scratch.resize(data.len(), 0);
    for y in 0..h {
        for x in 0..w {
            let src_off = (y * w + x) * channels;
            let dst_x = h - 1 - y;
            let dst_y = x;
            let dst_off = (dst_y * new_w + dst_x) * channels;
            scratch[dst_off..dst_off + channels]
                .copy_from_slice(&data[src_off..src_off + channels]);
        }
    }
    data[..scratch.len()].copy_from_slice(&scratch[..]);
}

/// Rotate pixel data 180° in-place.
fn rotate_180(data: &mut [u8], w: usize, h: usize, channels: usize) {
    let total_pixels = w * h;
    for i in 0..total_pixels / 2 {
        let j = total_pixels - 1 - i;
        let a = i * channels;
        let b = j * channels;
        for c in 0..channels {
            data.swap(a + c, b + c);
        }
    }
}

/// Rotate pixel data 270° clockwise (= 90° CCW) in-place.
fn rotate_270_cw(data: &mut [u8], w: usize, h: usize, channels: usize, scratch: &mut Vec<u8>) {
    let new_w = h;
    scratch.resize(data.len(), 0);
    for y in 0..h {
        for x in 0..w {
            let src_off = (y * w + x) * channels;
            let dst_x = y;
            let dst_y = w - 1 - x;
            let dst_off = (dst_y * new_w + dst_x) * channels;
            scratch[dst_off..dst_off + channels]
                .copy_from_slice(&data[src_off..src_off + channels]);
        }
    }
    data[..scratch.len()].copy_from_slice(&scratch[..]);
}

/// Flip pixel data horizontally in-place.
fn flip_horizontal(data: &mut [u8], w: usize, h: usize, channels: usize) {
    for y in 0..h {
        let row_start = y * w * channels;
        for x in 0..w / 2 {
            let left = row_start + x * channels;
            let right = row_start + (w - 1 - x) * channels;
            for c in 0..channels {
                data.swap(left + c, right + c);
            }
        }
    }
}

/// Decode a JPEG image from `data` into the pre-allocated tensor `dst`.
///
/// The tensor must have at least `width × height` pixel capacity. Decoded
/// rows are written at the tensor's `effective_row_stride()` offsets.
pub(crate) fn decode_jpeg_into<T: ImagePixel>(
    data: &[u8],
    dst: &mut Tensor<T>,
    opts: &DecodeOptions,
    scratch: &mut Vec<u8>,
) -> crate::Result<ImageInfo> {
    // Determine output format
    let dest_fmt = opts.format.unwrap_or(PixelFormat::Rgb);
    let colour = pixelfmt_to_colorspace(dest_fmt).ok_or(CodecError::UnsupportedFormat(dest_fmt))?;

    let mut zune_opts = DecoderOptions::default().jpeg_set_out_colorspace(colour);
    if opts.scale_denom > 1 {
        // zune-jpeg doesn't expose IDCT scaling, so we ignore scale_denom
        // for now. Phase 2 custom decoder will support this.
        log::debug!(
            "JPEG IDCT scaling not supported in Phase 1 shim, ignoring scale_denom={}",
            opts.scale_denom
        );
    }
    let _ = &mut zune_opts;

    let mut decoder = JpegDecoder::new_with_options(
        zune_jpeg::zune_core::bytestream::ZCursor::new(data),
        zune_opts,
    );
    decoder.decode_headers()?;

    let info = decoder
        .info()
        .ok_or_else(|| CodecError::InvalidData("JPEG: no header info".into()))?;
    let output_cs = decoder
        .output_colorspace()
        .ok_or_else(|| CodecError::InvalidData("JPEG: no output colorspace".into()))?;
    let output_fmt = colorspace_to_pixelfmt(output_cs).ok_or_else(|| {
        CodecError::InvalidData(format!("JPEG: unsupported output colorspace {output_cs:?}"))
    })?;

    let mut img_w = info.width as usize;
    let mut img_h = info.height as usize;

    // Read EXIF orientation
    let (rotation_deg, flip_h) = if opts.apply_exif {
        decoder
            .exif()
            .map(|x| read_exif_orientation(x))
            .unwrap_or((0, false))
    } else {
        (0, false)
    };

    // After rotation, dimensions may swap
    let (final_w, final_h) = match rotation_deg {
        90 | 270 => (img_h, img_w),
        _ => (img_w, img_h),
    };

    // Validate tensor capacity
    let tensor_w = dst
        .width()
        .unwrap_or_else(|| dst.shape().get(1).copied().unwrap_or(0));
    let tensor_h = dst
        .height()
        .unwrap_or_else(|| dst.shape().first().copied().unwrap_or(0));
    if final_w > tensor_w || final_h > tensor_h {
        return Err(CodecError::InsufficientCapacity {
            image: (final_w, final_h),
            tensor: (tensor_w, tensor_h),
        });
    }

    // Validate that T is a supported pixel type (all ImagePixel types are valid)
    // No dtype check needed — any ImagePixel impl works via from_u8().

    let channels = output_fmt.channels();

    // Decode into scratch buffer (contiguous)
    let decoded_size = img_w * img_h * channels;
    scratch.resize(decoded_size, 0);
    decoder.decode_into(scratch)?;

    // Apply EXIF transformations on the contiguous buffer
    if flip_h {
        flip_horizontal(scratch, img_w, img_h, channels);
    }
    match rotation_deg {
        90 => {
            rotate_90_cw(scratch, img_w, img_h, channels, &mut Vec::new());
            std::mem::swap(&mut img_w, &mut img_h);
        }
        180 => rotate_180(scratch, img_w, img_h, channels),
        270 => {
            rotate_270_cw(scratch, img_w, img_h, channels, &mut Vec::new());
            std::mem::swap(&mut img_w, &mut img_h);
        }
        _ => {}
    }

    // Write decoded rows into the tensor buffer at stride offsets.
    // The tensor keeps its original shape — ImageInfo reports the actual
    // decoded dimensions. Callers use ImageInfo.width/height (e.g. via
    // Crop) when passing the tensor to convert().
    let elem_size = std::mem::size_of::<T>();
    let dst_stride = dst
        .effective_row_stride()
        .unwrap_or(tensor_w * channels * elem_size);
    let src_stride = img_w * channels;

    // Write rows into tensor at stride offsets
    {
        let mut map = dst.map()?;
        let dst_bytes: &mut [T] = &mut map;

        if T::dtype() == edgefirst_tensor::DType::U8 {
            // Fast path: u8 → u8, direct byte copy
            // SAFETY: T is u8, so &mut [T] and &mut [u8] are layout-identical
            let dst_u8: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut u8, dst_bytes.len())
            };
            let dst_stride_bytes = dst_stride;
            for y in 0..img_h {
                let s = y * src_stride;
                let d = y * dst_stride_bytes;
                dst_u8[d..d + src_stride].copy_from_slice(&scratch[s..s + src_stride]);
            }
        } else if T::dtype() == edgefirst_tensor::DType::I8 {
            // Fast path: u8 → i8, copy + XOR 0x80 per byte
            // SAFETY: T is i8 (1 byte), layout-identical to u8
            let dst_u8: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut u8, dst_bytes.len())
            };
            for y in 0..img_h {
                let s = y * src_stride;
                let d = y * dst_stride;
                dst_u8[d..d + src_stride].copy_from_slice(&scratch[s..s + src_stride]);
                // XOR sign-bit flip in-place
                for b in &mut dst_u8[d..d + src_stride] {
                    *b ^= 0x80;
                }
            }
        } else {
            // Generic conversion path: u8 → T (u16, i16, f32)
            let dst_stride_elems = dst_stride / elem_size;
            for y in 0..img_h {
                let s = y * src_stride;
                let d = y * dst_stride_elems;
                for x in 0..src_stride {
                    dst_bytes[d + x] = T::from_u8(scratch[s + x]);
                }
            }
        }
    }

    Ok(ImageInfo {
        width: img_w,
        height: img_h,
        format: output_fmt,
        row_stride: dst_stride,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colorspace_roundtrip() {
        for fmt in [
            PixelFormat::Rgb,
            PixelFormat::Rgba,
            PixelFormat::Grey,
            PixelFormat::Bgra,
        ] {
            let cs = pixelfmt_to_colorspace(fmt).unwrap();
            assert_eq!(colorspace_to_pixelfmt(cs), Some(fmt));
        }
    }

    #[test]
    fn unsupported_format() {
        assert!(pixelfmt_to_colorspace(PixelFormat::Nv12).is_none());
    }

    #[test]
    fn exif_default() {
        assert_eq!(read_exif_orientation(&[]), (0, false));
    }
}
