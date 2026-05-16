// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! PNG decoding into pre-allocated tensors (Phase 1: zune-png shim).

use crate::error::CodecError;
use crate::options::{DecodeOptions, ImageInfo};
use crate::pixel::ImagePixel;
use edgefirst_tensor::{PixelFormat, Tensor, TensorTrait};
use zune_png::zune_core::colorspace::ColorSpace;
use zune_png::zune_core::options::DecoderOptions;
use zune_png::PngDecoder;

/// Map zune's [`ColorSpace`] to a [`PixelFormat`] and whether LumaA stripping
/// is needed.
fn colorspace_to_pixelfmt(cs: ColorSpace) -> Option<(PixelFormat, bool)> {
    match cs {
        ColorSpace::Luma => Some((PixelFormat::Grey, false)),
        ColorSpace::LumaA => Some((PixelFormat::Grey, true)),
        ColorSpace::RGB => Some((PixelFormat::Rgb, false)),
        ColorSpace::RGBA => Some((PixelFormat::Rgba, false)),
        _ => None,
    }
}

/// Decode a PNG image from `data` into the pre-allocated tensor `dst`.
pub(crate) fn decode_png_into<T: ImagePixel>(
    data: &[u8],
    dst: &mut Tensor<T>,
    opts: &DecodeOptions,
    scratch: &mut Vec<u8>,
) -> crate::Result<ImageInfo> {
    let dest_fmt = opts.format.unwrap_or(PixelFormat::Rgb);

    let zune_opts = DecoderOptions::default()
        .png_set_add_alpha_channel(false)
        .png_set_decode_animated(false);
    let mut decoder = PngDecoder::new_with_options(
        zune_png::zune_core::bytestream::ZCursor::new(data),
        zune_opts,
    );
    decoder.decode_headers()?;

    let info = decoder
        .info()
        .ok_or_else(|| CodecError::InvalidData("PNG: no header info".into()))?;
    let img_w = info.width;
    let img_h = info.height;

    let decoder_cs = decoder
        .colorspace()
        .ok_or_else(|| CodecError::InvalidData("PNG: no colorspace".into()))?;
    let (decoded_fmt, strip_luma_alpha) = colorspace_to_pixelfmt(decoder_cs).ok_or_else(|| {
        CodecError::InvalidData(format!("PNG: unsupported colorspace {decoder_cs:?}"))
    })?;

    // Validate tensor capacity
    let tensor_w = dst
        .width()
        .unwrap_or_else(|| dst.shape().get(1).copied().unwrap_or(0));
    let tensor_h = dst
        .height()
        .unwrap_or_else(|| dst.shape().first().copied().unwrap_or(0));
    if img_w > tensor_w || img_h > tensor_h {
        return Err(CodecError::InsufficientCapacity {
            image: (img_w, img_h),
            tensor: (tensor_w, tensor_h),
        });
    }

    // Validate dtype
    if T::dtype() != edgefirst_tensor::DType::U8 && T::dtype() != edgefirst_tensor::DType::F32 {
        return Err(CodecError::UnsupportedDtype(T::dtype()));
    }

    // Decode into scratch
    let decode_channels = if strip_luma_alpha {
        2
    } else {
        decoded_fmt.channels()
    };
    let decoded_size = img_w * img_h * decode_channels;
    scratch.resize(decoded_size, 0);
    decoder.decode_into(scratch)?;

    // Strip LumaA → Grey if needed
    let (src_pixels, src_channels) = if strip_luma_alpha {
        // Compact in-place: take every other byte (luma, skip alpha)
        for (write, i) in (0..decoded_size).step_by(2).enumerate() {
            scratch[write] = scratch[i];
        }
        (&scratch[..img_w * img_h], 1usize)
    } else {
        (&scratch[..decoded_size], decoded_fmt.channels())
    };

    // Format conversion if needed (decoded_fmt → dest_fmt)
    // For Phase 1, we support a subset of conversions:
    //   Grey → Grey, RGB → RGB, RGBA → RGBA (identity)
    //   RGB → RGBA (add alpha=255), RGBA → RGB (strip alpha)
    //   Grey → RGB (broadcast), RGB → Grey (luminance)
    let final_channels = dest_fmt.channels();
    let needs_conversion = decoded_fmt != dest_fmt;

    // Allocate conversion scratch if needed
    let converted: Vec<u8>;
    let final_pixels: &[u8] = if needs_conversion {
        let conv_size = img_w * img_h * final_channels;
        converted = convert_pixels(src_pixels, src_channels, final_channels, img_w * img_h);
        if converted.len() != conv_size {
            return Err(CodecError::UnsupportedFormat(dest_fmt));
        }
        &converted
    } else {
        src_pixels
    };

    // Write decoded rows into tensor at stride offsets.
    // The tensor keeps its original shape — ImageInfo reports actual dims.
    let elem_size = std::mem::size_of::<T>();
    let dst_stride = dst
        .effective_row_stride()
        .unwrap_or(tensor_w * final_channels * elem_size);
    let src_stride = img_w * final_channels;

    {
        let mut map = dst.map()?;
        let dst_elems: &mut [T] = &mut map;

        if T::dtype() == edgefirst_tensor::DType::U8 {
            let dst_u8: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(dst_elems.as_mut_ptr() as *mut u8, dst_elems.len())
            };
            for y in 0..img_h {
                let s = y * src_stride;
                let d = y * dst_stride;
                dst_u8[d..d + src_stride].copy_from_slice(&final_pixels[s..s + src_stride]);
            }
        } else {
            let dst_stride_elems = dst_stride / elem_size;
            for y in 0..img_h {
                let s = y * src_stride;
                let d = y * dst_stride_elems;
                for x in 0..src_stride {
                    dst_elems[d + x] = T::from_u8(final_pixels[s + x]);
                }
            }
        }
    }

    Ok(ImageInfo {
        width: img_w,
        height: img_h,
        format: dest_fmt,
        row_stride: dst_stride,
    })
}

/// Simple pixel format conversion.
fn convert_pixels(src: &[u8], src_ch: usize, dst_ch: usize, pixel_count: usize) -> Vec<u8> {
    let mut out = vec![0u8; pixel_count * dst_ch];
    match (src_ch, dst_ch) {
        // RGB → RGBA: add alpha=255
        (3, 4) => {
            for i in 0..pixel_count {
                out[i * 4] = src[i * 3];
                out[i * 4 + 1] = src[i * 3 + 1];
                out[i * 4 + 2] = src[i * 3 + 2];
                out[i * 4 + 3] = 255;
            }
        }
        // RGBA → RGB: strip alpha
        (4, 3) => {
            for i in 0..pixel_count {
                out[i * 3] = src[i * 4];
                out[i * 3 + 1] = src[i * 4 + 1];
                out[i * 3 + 2] = src[i * 4 + 2];
            }
        }
        // Grey → RGB: broadcast
        (1, 3) => {
            for i in 0..pixel_count {
                out[i * 3] = src[i];
                out[i * 3 + 1] = src[i];
                out[i * 3 + 2] = src[i];
            }
        }
        // Grey → RGBA: broadcast + alpha
        (1, 4) => {
            for i in 0..pixel_count {
                out[i * 4] = src[i];
                out[i * 4 + 1] = src[i];
                out[i * 4 + 2] = src[i];
                out[i * 4 + 3] = 255;
            }
        }
        // RGB → Grey: BT.601 luminance
        (3, 1) => {
            for i in 0..pixel_count {
                let r = src[i * 3] as u32;
                let g = src[i * 3 + 1] as u32;
                let b = src[i * 3 + 2] as u32;
                out[i] = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
            }
        }
        // RGBA → Grey: BT.601 luminance (ignore alpha)
        (4, 1) => {
            for i in 0..pixel_count {
                let r = src[i * 4] as u32;
                let g = src[i * 4 + 1] as u32;
                let b = src[i * 4 + 2] as u32;
                out[i] = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
            }
        }
        _ => {
            // Unsupported: return empty (caller checks length)
            return Vec::new();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_to_rgba() {
        let src = [10, 20, 30, 40, 50, 60];
        let out = convert_pixels(&src, 3, 4, 2);
        assert_eq!(out, [10, 20, 30, 255, 40, 50, 60, 255]);
    }

    #[test]
    fn rgba_to_rgb() {
        let src = [10, 20, 30, 255, 40, 50, 60, 128];
        let out = convert_pixels(&src, 4, 3, 2);
        assert_eq!(out, [10, 20, 30, 40, 50, 60]);
    }

    #[test]
    fn grey_to_rgb() {
        let src = [100, 200];
        let out = convert_pixels(&src, 1, 3, 2);
        assert_eq!(out, [100, 100, 100, 200, 200, 200]);
    }

    #[test]
    fn unsupported_conversion() {
        let src = [1, 2, 3];
        let out = convert_pixels(&src, 3, 5, 1);
        assert!(out.is_empty());
    }
}
