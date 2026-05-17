// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! PNG decoding into pre-allocated tensors via zune-png.

use crate::error::CodecError;
use crate::exif::{apply_exif_u8, read_exif_orientation, rotated_dims};
use crate::options::{DecodeOptions, ImageInfo};
use crate::pixel::ImagePixel;
use edgefirst_tensor::{PixelFormat, Tensor, TensorTrait};
use zune_png::zune_core::colorspace::ColorSpace;
use zune_png::zune_core::options::DecoderOptions;
use zune_png::zune_core::result::DecodingResult;
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

/// Parse PNG headers and return image dimensions/format without decoding pixels.
///
/// The returned `format` matches `opts.format` when set (defaulting to RGB),
/// regardless of the source colorspace — the decoder converts at decode time.
/// For `apply_exif=true` and a PNG with a 90°/270° eXIf chunk, the returned
/// `width` and `height` reflect the post-rotation layout.
pub fn peek_png_info(data: &[u8], opts: &DecodeOptions) -> crate::Result<ImageInfo> {
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
    let width = info.width;
    let height = info.height;
    let exif_bytes = info.exif.clone();

    let decoder_cs = decoder
        .colorspace()
        .ok_or_else(|| CodecError::InvalidData("PNG: no colorspace".into()))?;
    let _ = colorspace_to_pixelfmt(decoder_cs).ok_or_else(|| {
        CodecError::InvalidData(format!("PNG: unsupported colorspace {decoder_cs:?}"))
    })?;

    let dest_fmt = opts.format.unwrap_or(PixelFormat::Rgb);

    let (rotation_deg, _flip_h) = if opts.apply_exif {
        exif_bytes
            .as_deref()
            .map(read_exif_orientation)
            .unwrap_or((0, false))
    } else {
        (0, false)
    };

    let (final_w, final_h) = rotated_dims(width, height, rotation_deg);
    let channels = dest_fmt.channels();
    Ok(ImageInfo {
        width: final_w,
        height: final_h,
        format: dest_fmt,
        row_stride: final_w * channels,
    })
}

/// Decode a PNG image from `data` into the pre-allocated tensor `dst`.
pub(crate) fn decode_png_into<T: ImagePixel>(
    data: &[u8],
    dst: &mut Tensor<T>,
    opts: &DecodeOptions,
    scratch: &mut Vec<u8>,
    rot_scratch: &mut Vec<u8>,
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
    let exif_bytes = info.exif.clone();

    let decoder_cs = decoder
        .colorspace()
        .ok_or_else(|| CodecError::InvalidData("PNG: no colorspace".into()))?;
    let (decoded_fmt, strip_luma_alpha) = colorspace_to_pixelfmt(decoder_cs).ok_or_else(|| {
        CodecError::InvalidData(format!("PNG: unsupported colorspace {decoder_cs:?}"))
    })?;

    // For types that can benefit from native 16-bit PNG decode (u16, i16, f32),
    // use decode() → DecodingResult which preserves the full bit depth.
    // For u8/i8, use decode_into(&mut [u8]) as before. EXIF rotation is only
    // applied on the u8 path; the native-u16 path ignores EXIF orientation.
    let use_native_u16 = matches!(
        T::dtype(),
        edgefirst_tensor::DType::U16 | edgefirst_tensor::DType::I16 | edgefirst_tensor::DType::F32
    );

    let (rotation_deg, flip_h) = if !use_native_u16 && opts.apply_exif {
        exif_bytes
            .as_deref()
            .map(read_exif_orientation)
            .unwrap_or((0, false))
    } else {
        (0, false)
    };

    // Validate tensor capacity against POST-rotation dimensions.
    let (final_w_check, final_h_check) = rotated_dims(img_w, img_h, rotation_deg);
    let tensor_w = dst
        .width()
        .unwrap_or_else(|| dst.shape().get(1).copied().unwrap_or(0));
    let tensor_h = dst
        .height()
        .unwrap_or_else(|| dst.shape().first().copied().unwrap_or(0));
    if final_w_check > tensor_w || final_h_check > tensor_h {
        return Err(CodecError::InsufficientCapacity {
            image: (final_w_check, final_h_check),
            tensor: (tensor_w, tensor_h),
        });
    }

    if use_native_u16 {
        decode_png_via_decoding_result(
            decoder,
            dst,
            opts,
            dest_fmt,
            decoded_fmt,
            strip_luma_alpha,
            img_w,
            img_h,
        )
    } else {
        decode_png_via_u8(
            decoder,
            dst,
            opts,
            scratch,
            rot_scratch,
            dest_fmt,
            decoded_fmt,
            strip_luma_alpha,
            img_w,
            img_h,
            rotation_deg,
            flip_h,
        )
    }
}

/// Decode PNG using `decode()` → `DecodingResult` to preserve native bit depth.
/// If the PNG is 16-bit, we get `Vec<u16>` directly from zune-png.
/// If 8-bit, we get `Vec<u8>` and upscale via `from_u8`.
#[allow(clippy::too_many_arguments)]
fn decode_png_via_decoding_result<T: ImagePixel>(
    mut decoder: PngDecoder<zune_png::zune_core::bytestream::ZCursor<&[u8]>>,
    dst: &mut Tensor<T>,
    _opts: &DecodeOptions,
    dest_fmt: PixelFormat,
    decoded_fmt: PixelFormat,
    strip_luma_alpha: bool,
    img_w: usize,
    img_h: usize,
) -> crate::Result<ImageInfo> {
    let result = decoder.decode()?;

    let final_channels = dest_fmt.channels();
    let elem_size = std::mem::size_of::<T>();
    let tensor_w = dst
        .width()
        .unwrap_or_else(|| dst.shape().get(1).copied().unwrap_or(0));
    let dst_stride = dst
        .effective_row_stride()
        .unwrap_or(tensor_w * final_channels * elem_size);
    let dst_stride_elems = dst_stride / elem_size;
    let src_stride = img_w * final_channels;

    match result {
        DecodingResult::U16(raw_u16) => {
            // Native 16-bit PNG: use from_u16 for best precision
            let decode_channels = if strip_luma_alpha {
                2
            } else {
                decoded_fmt.channels()
            };

            // Strip LumaA if needed
            let src_u16: Vec<u16> = if strip_luma_alpha {
                raw_u16.iter().step_by(2).copied().collect()
            } else {
                raw_u16
            };

            // Format conversion if needed
            let final_pixels = if decoded_fmt != dest_fmt {
                convert_pixels_u16(
                    &src_u16,
                    if strip_luma_alpha { 1 } else { decode_channels },
                    final_channels,
                    img_w * img_h,
                )
            } else {
                src_u16
            };

            // Write rows into tensor
            let mut map = dst.map()?;
            let dst_elems: &mut [T] = &mut map;
            for y in 0..img_h {
                let s = y * src_stride;
                let d = y * dst_stride_elems;
                for x in 0..src_stride {
                    dst_elems[d + x] = T::from_u16(final_pixels[s + x]);
                }
            }
        }
        DecodingResult::U8(raw_u8) => {
            // 8-bit PNG: use from_u8 (same as the u8 path but into wider types)
            let decode_channels = if strip_luma_alpha {
                2
            } else {
                decoded_fmt.channels()
            };

            let src_u8: Vec<u8> = if strip_luma_alpha {
                raw_u8.iter().step_by(2).copied().collect()
            } else {
                raw_u8
            };

            let final_pixels = if decoded_fmt != dest_fmt {
                convert_pixels(
                    &src_u8,
                    if strip_luma_alpha { 1 } else { decode_channels },
                    final_channels,
                    img_w * img_h,
                )
            } else {
                src_u8
            };

            if final_pixels.len() != img_w * img_h * final_channels {
                return Err(CodecError::UnsupportedFormat(dest_fmt));
            }

            let mut map = dst.map()?;
            let dst_elems: &mut [T] = &mut map;
            for y in 0..img_h {
                let s = y * src_stride;
                let d = y * dst_stride_elems;
                for x in 0..src_stride {
                    dst_elems[d + x] = T::from_u8(final_pixels[s + x]);
                }
            }
        }
        DecodingResult::F32(raw_f32) => {
            // F32 PNG (rare but possible)
            let decode_channels = if strip_luma_alpha {
                2
            } else {
                decoded_fmt.channels()
            };

            let src_f32: Vec<f32> = if strip_luma_alpha {
                raw_f32.iter().step_by(2).copied().collect()
            } else {
                raw_f32
            };

            // Convert f32 [0,1] → u16 [0,65535] for the from_u16 path
            let as_u16: Vec<u16> = src_f32
                .iter()
                .map(|&v| (v.clamp(0.0, 1.0) * 65535.0) as u16)
                .collect();

            let final_pixels = if decoded_fmt != dest_fmt {
                convert_pixels_u16(
                    &as_u16,
                    if strip_luma_alpha { 1 } else { decode_channels },
                    final_channels,
                    img_w * img_h,
                )
            } else {
                as_u16
            };

            let mut map = dst.map()?;
            let dst_elems: &mut [T] = &mut map;
            for y in 0..img_h {
                let s = y * src_stride;
                let d = y * dst_stride_elems;
                for x in 0..src_stride {
                    dst_elems[d + x] = T::from_u16(final_pixels[s + x]);
                }
            }
        }
        _ => {
            return Err(CodecError::InvalidData(
                "PNG: unsupported decoded pixel format from zune".into(),
            ));
        }
    }

    Ok(ImageInfo {
        width: img_w,
        height: img_h,
        format: dest_fmt,
        row_stride: dst_stride,
    })
}

/// Decode PNG via the `decode_into(&mut [u8])` path for u8/i8 targets.
///
/// `rotation_deg` and `flip_h` come from the eXIf chunk and are applied to
/// the post-conversion pixel buffer; `img_w`/`img_h` reflect raw decode dims
/// and become post-rotation dims after `apply_exif_u8`.
#[allow(clippy::too_many_arguments)]
fn decode_png_via_u8<T: ImagePixel>(
    mut decoder: PngDecoder<zune_png::zune_core::bytestream::ZCursor<&[u8]>>,
    dst: &mut Tensor<T>,
    _opts: &DecodeOptions,
    scratch: &mut Vec<u8>,
    rot_scratch: &mut Vec<u8>,
    dest_fmt: PixelFormat,
    decoded_fmt: PixelFormat,
    strip_luma_alpha: bool,
    img_w: usize,
    img_h: usize,
    rotation_deg: u16,
    flip_h: bool,
) -> crate::Result<ImageInfo> {
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
        for (write, i) in (0..decoded_size).step_by(2).enumerate() {
            scratch[write] = scratch[i];
        }
        (&scratch[..img_w * img_h], 1usize)
    } else {
        (&scratch[..decoded_size], decoded_fmt.channels())
    };

    // Format conversion if needed. Always materialise an owned buffer when
    // EXIF rotation is requested so apply_exif_u8 can mutate it in place.
    let final_channels = dest_fmt.channels();
    let needs_conversion = decoded_fmt != dest_fmt;
    let needs_rotation = flip_h || rotation_deg != 0;

    let owned_pixels: Option<Vec<u8>> = if needs_conversion {
        let conv_size = img_w * img_h * final_channels;
        let c = convert_pixels(src_pixels, src_channels, final_channels, img_w * img_h);
        if c.len() != conv_size {
            return Err(CodecError::UnsupportedFormat(dest_fmt));
        }
        Some(c)
    } else if needs_rotation {
        Some(src_pixels.to_vec())
    } else {
        None
    };

    // Apply EXIF rotation/flip in place on the owned buffer.
    let mut img_w = img_w;
    let mut img_h = img_h;
    let owned_pixels = if let Some(mut buf) = owned_pixels {
        if needs_rotation {
            apply_exif_u8(
                &mut buf,
                img_w * final_channels,
                &mut img_w,
                &mut img_h,
                final_channels,
                rotation_deg,
                flip_h,
                rot_scratch,
            );
        }
        Some(buf)
    } else {
        None
    };
    let final_pixels: &[u8] = owned_pixels.as_deref().unwrap_or(src_pixels);

    // Write decoded rows into tensor at stride offsets
    let elem_size = std::mem::size_of::<T>();
    let tensor_w = dst
        .width()
        .unwrap_or_else(|| dst.shape().get(1).copied().unwrap_or(0));
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
        } else if T::dtype() == edgefirst_tensor::DType::I8 {
            // Fast path: copy + XOR 0x80
            let dst_u8: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(dst_elems.as_mut_ptr() as *mut u8, dst_elems.len())
            };
            for y in 0..img_h {
                let s = y * src_stride;
                let d = y * dst_stride;
                dst_u8[d..d + src_stride].copy_from_slice(&final_pixels[s..s + src_stride]);
                for b in &mut dst_u8[d..d + src_stride] {
                    *b ^= 0x80;
                }
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

/// Simple pixel format conversion (u8 path).
fn convert_pixels(src: &[u8], src_ch: usize, dst_ch: usize, pixel_count: usize) -> Vec<u8> {
    let mut out = vec![0u8; pixel_count * dst_ch];
    match (src_ch, dst_ch) {
        (3, 4) => {
            for i in 0..pixel_count {
                out[i * 4] = src[i * 3];
                out[i * 4 + 1] = src[i * 3 + 1];
                out[i * 4 + 2] = src[i * 3 + 2];
                out[i * 4 + 3] = 255;
            }
        }
        (4, 3) => {
            for i in 0..pixel_count {
                out[i * 3] = src[i * 4];
                out[i * 3 + 1] = src[i * 4 + 1];
                out[i * 3 + 2] = src[i * 4 + 2];
            }
        }
        (1, 3) => {
            for i in 0..pixel_count {
                out[i * 3] = src[i];
                out[i * 3 + 1] = src[i];
                out[i * 3 + 2] = src[i];
            }
        }
        (1, 4) => {
            for i in 0..pixel_count {
                out[i * 4] = src[i];
                out[i * 4 + 1] = src[i];
                out[i * 4 + 2] = src[i];
                out[i * 4 + 3] = 255;
            }
        }
        (3, 1) => {
            for i in 0..pixel_count {
                let r = src[i * 3] as u32;
                let g = src[i * 3 + 1] as u32;
                let b = src[i * 3 + 2] as u32;
                out[i] = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
            }
        }
        (4, 1) => {
            for i in 0..pixel_count {
                let r = src[i * 4] as u32;
                let g = src[i * 4 + 1] as u32;
                let b = src[i * 4 + 2] as u32;
                out[i] = ((r * 77 + g * 150 + b * 29) >> 8) as u8;
            }
        }
        _ => {
            return Vec::new();
        }
    }
    out
}

/// Pixel format conversion for u16 data (16-bit PNG path).
fn convert_pixels_u16(src: &[u16], src_ch: usize, dst_ch: usize, pixel_count: usize) -> Vec<u16> {
    let mut out = vec![0u16; pixel_count * dst_ch];
    match (src_ch, dst_ch) {
        (3, 4) => {
            for i in 0..pixel_count {
                out[i * 4] = src[i * 3];
                out[i * 4 + 1] = src[i * 3 + 1];
                out[i * 4 + 2] = src[i * 3 + 2];
                out[i * 4 + 3] = 65535;
            }
        }
        (4, 3) => {
            for i in 0..pixel_count {
                out[i * 3] = src[i * 4];
                out[i * 3 + 1] = src[i * 4 + 1];
                out[i * 3 + 2] = src[i * 4 + 2];
            }
        }
        (1, 3) => {
            for i in 0..pixel_count {
                out[i * 3] = src[i];
                out[i * 3 + 1] = src[i];
                out[i * 3 + 2] = src[i];
            }
        }
        (1, 4) => {
            for i in 0..pixel_count {
                out[i * 4] = src[i];
                out[i * 4 + 1] = src[i];
                out[i * 4 + 2] = src[i];
                out[i * 4 + 3] = 65535;
            }
        }
        (3, 1) => {
            for i in 0..pixel_count {
                let r = src[i * 3] as u32;
                let g = src[i * 3 + 1] as u32;
                let b = src[i * 3 + 2] as u32;
                out[i] = ((r * 77 + g * 150 + b * 29) >> 8) as u16;
            }
        }
        (4, 1) => {
            for i in 0..pixel_count {
                let r = src[i * 4] as u32;
                let g = src[i * 4 + 1] as u32;
                let b = src[i * 4 + 2] as u32;
                out[i] = ((r * 77 + g * 150 + b * 29) >> 8) as u16;
            }
        }
        _ => {
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

    #[test]
    fn u16_rgb_to_rgba() {
        let src: Vec<u16> = vec![1000, 2000, 3000, 4000, 5000, 6000];
        let out = convert_pixels_u16(&src, 3, 4, 2);
        assert_eq!(out, [1000, 2000, 3000, 65535, 4000, 5000, 6000, 65535]);
    }

    #[test]
    fn u16_grey_to_rgb() {
        let src: Vec<u16> = vec![10000, 50000];
        let out = convert_pixels_u16(&src, 1, 3, 2);
        assert_eq!(out, [10000, 10000, 10000, 50000, 50000, 50000]);
    }
}
