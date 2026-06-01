// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! PNG decoding into pre-allocated tensors via zune-png.
//!
//! PNG is decoded to its native format (Luma/LumaA → `Grey`, RGB → `Rgb`,
//! RGBA → `Rgba`); the decoder configures the destination tensor's dimensions
//! and format accordingly and reports EXIF orientation. It never colour-
//! converts or rotates — use `ImageProcessor::convert()` for that.

use crate::error::CodecError;
use crate::exif::read_exif_orientation;
use crate::options::ImageInfo;
use crate::pixel::ImagePixel;
use edgefirst_tensor::{PixelFormat, Tensor, TensorTrait};
use zune_png::zune_core::colorspace::ColorSpace;
use zune_png::zune_core::options::DecoderOptions;
use zune_png::zune_core::result::DecodingResult;
use zune_png::PngDecoder;

/// Map zune's [`ColorSpace`] to the native [`PixelFormat`] and whether LumaA
/// alpha stripping (→ `Grey`) is needed.
fn colorspace_to_pixelfmt(cs: ColorSpace) -> Option<(PixelFormat, bool)> {
    match cs {
        ColorSpace::Luma => Some((PixelFormat::Grey, false)),
        ColorSpace::LumaA => Some((PixelFormat::Grey, true)),
        ColorSpace::RGB => Some((PixelFormat::Rgb, false)),
        ColorSpace::RGBA => Some((PixelFormat::Rgba, false)),
        _ => None,
    }
}

fn zune_options() -> DecoderOptions {
    DecoderOptions::default()
        .png_set_add_alpha_channel(false)
        .png_set_decode_animated(false)
}

/// Map a `configure_image` error to the codec error type.
fn config_err(
    e: edgefirst_tensor::Error,
    img_w: usize,
    img_h: usize,
    cap: (usize, usize),
) -> CodecError {
    match e {
        edgefirst_tensor::Error::InsufficientCapacity { .. } => CodecError::InsufficientCapacity {
            image: (img_w, img_h),
            tensor: cap,
        },
        other => CodecError::Tensor(other),
    }
}

/// Parse PNG headers and return native dimensions, format, and EXIF
/// orientation without decoding pixels. The codec does not rotate.
pub fn peek_png_info(data: &[u8]) -> crate::Result<ImageInfo> {
    let mut decoder = PngDecoder::new_with_options(
        zune_png::zune_core::bytestream::ZCursor::new(data),
        zune_options(),
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
    let (format, _strip) = colorspace_to_pixelfmt(decoder_cs).ok_or_else(|| {
        CodecError::InvalidData(format!("PNG: unsupported colorspace {decoder_cs:?}"))
    })?;

    let (rotation_degrees, flip_horizontal) = exif_bytes
        .as_deref()
        .map(read_exif_orientation)
        .unwrap_or((0, false));

    Ok(ImageInfo {
        width,
        height,
        format,
        row_stride: width * format.channels(),
        rotation_degrees,
        flip_horizontal,
    })
}

/// Decode a PNG image from `data` into the pre-allocated tensor `dst`,
/// configuring its dimensions and native format.
pub(crate) fn decode_png_into<T: ImagePixel>(
    data: &[u8],
    dst: &mut Tensor<T>,
    scratch: &mut Vec<u8>,
) -> crate::Result<ImageInfo> {
    let _span = tracing::trace_span!(
        "codec.decode_png",
        dtype = std::any::type_name::<T>(),
        n_bytes = data.len(),
    )
    .entered();

    let mut decoder = PngDecoder::new_with_options(
        zune_png::zune_core::bytestream::ZCursor::new(data),
        zune_options(),
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
    let (format, strip_luma_alpha) = colorspace_to_pixelfmt(decoder_cs).ok_or_else(|| {
        CodecError::InvalidData(format!("PNG: unsupported colorspace {decoder_cs:?}"))
    })?;

    let (rotation_degrees, flip_horizontal) = exif_bytes
        .as_deref()
        .map(read_exif_orientation)
        .unwrap_or((0, false));

    let cap = (dst.width().unwrap_or(0), dst.height().unwrap_or(0));
    dst.configure_image(img_w, img_h, format)
        .map_err(|e| config_err(e, img_w, img_h, cap))?;

    let channels = format.channels();
    let elem_size = std::mem::size_of::<T>();
    let dst_stride = dst
        .effective_row_stride()
        .unwrap_or(img_w * channels * elem_size);

    // Native-bit-depth path for wide types; u8 path otherwise.
    let use_native_u16 = matches!(
        T::dtype(),
        edgefirst_tensor::DType::U16 | edgefirst_tensor::DType::I16 | edgefirst_tensor::DType::F32
    );

    if use_native_u16 {
        decode_png_wide::<T>(
            decoder,
            dst,
            format,
            strip_luma_alpha,
            img_w,
            img_h,
            dst_stride,
        )?;
    } else {
        decode_png_u8::<T>(
            decoder,
            dst,
            scratch,
            format,
            strip_luma_alpha,
            img_w,
            img_h,
            dst_stride,
        )?;
    }

    Ok(ImageInfo {
        width: img_w,
        height: img_h,
        format,
        row_stride: dst_stride,
        rotation_degrees,
        flip_horizontal,
    })
}

/// u8/i8 path: decode into `scratch`, strip LumaA if needed, write rows.
#[allow(clippy::too_many_arguments)]
fn decode_png_u8<T: ImagePixel>(
    mut decoder: PngDecoder<zune_png::zune_core::bytestream::ZCursor<&[u8]>>,
    dst: &mut Tensor<T>,
    scratch: &mut Vec<u8>,
    format: PixelFormat,
    strip_luma_alpha: bool,
    img_w: usize,
    img_h: usize,
    dst_stride: usize,
) -> crate::Result<()> {
    let decode_channels = if strip_luma_alpha {
        2
    } else {
        format.channels()
    };
    let decoded_size = img_w * img_h * decode_channels;
    scratch.resize(decoded_size, 0);
    {
        let _s = tracing::trace_span!("codec.decode_png.zune_decode", path = "u8").entered();
        decoder.decode_into(scratch)?;
    }

    let channels = format.channels();
    // Strip LumaA → Grey in place (keep every other byte).
    let src: &[u8] = if strip_luma_alpha {
        for (write, i) in (0..decoded_size).step_by(2).enumerate() {
            scratch[write] = scratch[i];
        }
        &scratch[..img_w * img_h]
    } else {
        &scratch[..decoded_size]
    };
    let src_stride = img_w * channels;

    let mut map = dst.map()?;
    let dst_elems: &mut [T] = &mut map;
    match T::dtype() {
        edgefirst_tensor::DType::U8 => {
            // SAFETY: T is u8 — layout-identical reinterpret.
            let d: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(dst_elems.as_mut_ptr() as *mut u8, dst_elems.len())
            };
            for y in 0..img_h {
                let s = y * src_stride;
                let o = y * dst_stride;
                d[o..o + src_stride].copy_from_slice(&src[s..s + src_stride]);
            }
        }
        edgefirst_tensor::DType::I8 => {
            // SAFETY: T is i8 — layout-identical reinterpret; copy + bias.
            let d: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(dst_elems.as_mut_ptr() as *mut u8, dst_elems.len())
            };
            for y in 0..img_h {
                let s = y * src_stride;
                let o = y * dst_stride;
                d[o..o + src_stride].copy_from_slice(&src[s..s + src_stride]);
                for b in &mut d[o..o + src_stride] {
                    *b ^= 0x80;
                }
            }
        }
        _ => {
            let dse = dst_stride / elem_or_one::<T>();
            for y in 0..img_h {
                let s = y * src_stride;
                let o = y * dse;
                for x in 0..src_stride {
                    dst_elems[o + x] = T::from_u8(src[s + x]);
                }
            }
        }
    }
    Ok(())
}

/// Native-bit-depth path for u16/i16/f32 targets.
#[allow(clippy::too_many_arguments)]
fn decode_png_wide<T: ImagePixel>(
    mut decoder: PngDecoder<zune_png::zune_core::bytestream::ZCursor<&[u8]>>,
    dst: &mut Tensor<T>,
    format: PixelFormat,
    strip_luma_alpha: bool,
    img_w: usize,
    img_h: usize,
    dst_stride: usize,
) -> crate::Result<()> {
    let result = {
        let _s =
            tracing::trace_span!("codec.decode_png.zune_decode", path = "native_u16").entered();
        decoder.decode()?
    };
    let channels = format.channels();
    let src_stride = img_w * channels;
    let dse = dst_stride / elem_or_one::<T>();

    let mut map = dst.map()?;
    let dst_elems: &mut [T] = &mut map;

    match result {
        DecodingResult::U16(raw) => {
            let src: Vec<u16> = if strip_luma_alpha {
                raw.iter().step_by(2).copied().collect()
            } else {
                raw
            };
            for y in 0..img_h {
                let s = y * src_stride;
                let o = y * dse;
                for x in 0..src_stride {
                    dst_elems[o + x] = T::from_u16(src[s + x]);
                }
            }
        }
        DecodingResult::U8(raw) => {
            let src: Vec<u8> = if strip_luma_alpha {
                raw.iter().step_by(2).copied().collect()
            } else {
                raw
            };
            for y in 0..img_h {
                let s = y * src_stride;
                let o = y * dse;
                for x in 0..src_stride {
                    dst_elems[o + x] = T::from_u8(src[s + x]);
                }
            }
        }
        DecodingResult::F32(raw) => {
            let src: Vec<u16> = raw
                .iter()
                .map(|&v| (v.clamp(0.0, 1.0) * 65535.0) as u16)
                .collect();
            let src: Vec<u16> = if strip_luma_alpha {
                src.iter().step_by(2).copied().collect()
            } else {
                src
            };
            for y in 0..img_h {
                let s = y * src_stride;
                let o = y * dse;
                for x in 0..src_stride {
                    dst_elems[o + x] = T::from_u16(src[s + x]);
                }
            }
        }
        _ => {
            return Err(CodecError::InvalidData(
                "PNG: unsupported decoded pixel format from zune".into(),
            ));
        }
    }
    Ok(())
}

#[inline]
fn elem_or_one<T>() -> usize {
    std::mem::size_of::<T>().max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colorspace_mapping() {
        assert_eq!(
            colorspace_to_pixelfmt(ColorSpace::Luma),
            Some((PixelFormat::Grey, false))
        );
        assert_eq!(
            colorspace_to_pixelfmt(ColorSpace::LumaA),
            Some((PixelFormat::Grey, true))
        );
        assert_eq!(
            colorspace_to_pixelfmt(ColorSpace::RGB),
            Some((PixelFormat::Rgb, false))
        );
        assert_eq!(
            colorspace_to_pixelfmt(ColorSpace::RGBA),
            Some((PixelFormat::Rgba, false))
        );
    }
}
