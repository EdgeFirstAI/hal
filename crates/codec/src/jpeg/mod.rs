// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Custom JPEG decoder with zero-allocation hot loop, strided output, and
//! NEON SIMD kernels.
//!
//! This module replaces the previous `zune-jpeg` wrapper with a from-scratch
//! baseline JPEG decoder optimised for:
//! - Reusable decoder state (zero allocations after initialisation)
//! - Direct output into strided tensor buffers (GPU-aligned pitch)
//! - NEON SIMD kernels for IDCT, color conversion, and chroma upsampling

pub mod bitstream;
pub mod color;
pub mod huffman;
pub mod idct;
pub mod markers;
pub mod mcu;
pub mod types;
pub mod upsample;

use crate::error::CodecError;
use crate::options::{DecodeOptions, ImageInfo};
use crate::pixel::ImagePixel;
use edgefirst_tensor::{PixelFormat, Tensor, TensorTrait};

/// Reusable JPEG decoder state.
///
/// Holds scratch buffers that grow to the high-water mark and are reused
/// across decode calls. After the first frame at a given resolution,
/// subsequent decodes perform zero heap allocations.
pub struct JpegDecoderState {
    /// MCU scratch buffers (component buffers, chroma rows, output row).
    mcu_scratch: Option<mcu::McuScratch>,
    /// EXIF rotation scratch buffer.
    exif_scratch: Vec<u8>,
}

impl JpegDecoderState {
    pub fn new() -> Self {
        Self {
            mcu_scratch: None,
            exif_scratch: Vec::new(),
        }
    }
}

impl Default for JpegDecoderState {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a [`PixelFormat`] to an output format supported by the JPEG decoder.
fn validate_output_format(fmt: PixelFormat) -> crate::Result<PixelFormat> {
    match fmt {
        PixelFormat::Rgb | PixelFormat::Rgba | PixelFormat::Grey | PixelFormat::Bgra => Ok(fmt),
        _ => Err(CodecError::UnsupportedFormat(fmt)),
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

/// Decode a JPEG image from `data` into the pre-allocated tensor `dst`.
///
/// The tensor must have at least `width × height` pixel capacity. Decoded
/// rows are written at the tensor's `effective_row_stride()` offsets.
pub fn decode_jpeg_into<T: ImagePixel>(
    data: &[u8],
    dst: &mut Tensor<T>,
    opts: &DecodeOptions,
    state: &mut JpegDecoderState,
) -> crate::Result<ImageInfo> {
    // Parse all marker segments
    let headers = markers::parse_markers(data)?;

    let hdr = &headers.header;
    let mut img_w = hdr.width as usize;
    let mut img_h = hdr.height as usize;

    // Determine output format
    let dest_fmt = opts.format.unwrap_or(PixelFormat::Rgb);
    let output_fmt = validate_output_format(dest_fmt)?;

    // Handle greyscale JPEGs
    let output_fmt = if hdr.components.len() == 1 && output_fmt != PixelFormat::Grey {
        // Greyscale JPEG: we can still output RGB/RGBA by expanding
        output_fmt
    } else {
        output_fmt
    };

    // Read EXIF orientation
    let (rotation_deg, flip_h) = if opts.apply_exif {
        headers
            .exif_data
            .as_deref()
            .map(read_exif_orientation)
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

    let channels = output_fmt.channels();
    let elem_size = std::mem::size_of::<T>();
    let dst_stride = dst
        .effective_row_stride()
        .unwrap_or(tensor_w * channels * elem_size);

    // Initialise or reuse MCU scratch buffers
    match &mut state.mcu_scratch {
        Some(scratch) => scratch.ensure_capacity(&headers),
        None => state.mcu_scratch = Some(mcu::McuScratch::new(&headers)),
    }
    let mcu_scratch = state.mcu_scratch.as_mut().unwrap();

    // Map the tensor buffer
    let mut map = dst.map()?;
    let dst_bytes: &mut [T] = &mut map;

    // We need to decode into u8 first, then convert to the target type.
    // For u8 targets, we can decode directly. For other types, use a
    // temporary u8 buffer then convert row-by-row.
    if T::dtype() == edgefirst_tensor::DType::U8 {
        // Fast path: decode directly into tensor buffer
        // SAFETY: T is u8, layout-identical
        let dst_u8: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut u8, dst_bytes.len())
        };

        mcu::decode_image(data, &headers, mcu_scratch, dst_u8, dst_stride, output_fmt)?;

        // Apply EXIF transformations
        if flip_h || rotation_deg != 0 {
            apply_exif_u8(
                dst_u8,
                dst_stride,
                &mut img_w,
                &mut img_h,
                channels,
                rotation_deg,
                flip_h,
                &mut state.exif_scratch,
            );
        }
    } else {
        // Generic path: decode into temporary u8 buffer, then convert
        let temp_stride = img_w * channels;
        let temp_size = temp_stride * img_h;
        state.exif_scratch.resize(temp_size, 0);

        mcu::decode_image(
            data,
            &headers,
            mcu_scratch,
            &mut state.exif_scratch,
            temp_stride,
            output_fmt,
        )?;

        // Apply EXIF transformations on the temp buffer
        if flip_h || rotation_deg != 0 {
            let mut extra_scratch = Vec::new();
            apply_exif_u8(
                &mut state.exif_scratch,
                temp_stride,
                &mut img_w,
                &mut img_h,
                channels,
                rotation_deg,
                flip_h,
                &mut extra_scratch,
            );
        }

        // Convert u8 → T and write to tensor at stride offsets
        let src_stride = img_w * channels;
        let dst_stride_elems = dst_stride / elem_size;

        if T::dtype() == edgefirst_tensor::DType::I8 {
            // Fast path: copy + XOR 0x80
            let dst_u8: &mut [u8] = unsafe {
                std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut u8, dst_bytes.len())
            };
            for y in 0..img_h {
                let s = y * src_stride;
                let d = y * dst_stride;
                dst_u8[d..d + src_stride].copy_from_slice(&state.exif_scratch[s..s + src_stride]);
                for b in &mut dst_u8[d..d + src_stride] {
                    *b ^= 0x80;
                }
            }
        } else {
            for y in 0..img_h {
                let s = y * src_stride;
                let d = y * dst_stride_elems;
                for x in 0..src_stride {
                    dst_bytes[d + x] = T::from_u8(state.exif_scratch[s + x]);
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

/// Apply EXIF rotation/flip to a contiguous u8 pixel buffer.
#[allow(clippy::too_many_arguments)]
fn apply_exif_u8(
    data: &mut [u8],
    stride: usize,
    w: &mut usize,
    h: &mut usize,
    channels: usize,
    rotation_deg: u16,
    flip_h: bool,
    scratch: &mut Vec<u8>,
) {
    let img_w = *w;
    let img_h = *h;

    if flip_h {
        for y in 0..img_h {
            let row_start = y * stride;
            for x in 0..img_w / 2 {
                let left = row_start + x * channels;
                let right = row_start + (img_w - 1 - x) * channels;
                for c in 0..channels {
                    data.swap(left + c, right + c);
                }
            }
        }
    }

    match rotation_deg {
        90 => {
            let src_stride = img_w * channels;
            let new_w = img_h;
            let new_h = img_w;
            scratch.resize(new_w * new_h * channels, 0);
            for y in 0..img_h {
                for x in 0..img_w {
                    let src_off = y * src_stride + x * channels;
                    let dst_x = img_h - 1 - y;
                    let dst_y = x;
                    let dst_off = dst_y * new_w * channels + dst_x * channels;
                    scratch[dst_off..dst_off + channels]
                        .copy_from_slice(&data[src_off..src_off + channels]);
                }
            }
            data[..scratch.len()].copy_from_slice(scratch);
            *w = new_w;
            *h = new_h;
        }
        180 => {
            let src_stride = img_w * channels;
            let total = img_w * img_h;
            for i in 0..total / 2 {
                let j = total - 1 - i;
                let a_y = i / img_w;
                let a_x = i % img_w;
                let b_y = j / img_w;
                let b_x = j % img_w;
                let a = a_y * src_stride + a_x * channels;
                let b = b_y * src_stride + b_x * channels;
                for c in 0..channels {
                    data.swap(a + c, b + c);
                }
            }
        }
        270 => {
            let src_stride = img_w * channels;
            let new_w = img_h;
            let new_h = img_w;
            scratch.resize(new_w * new_h * channels, 0);
            for y in 0..img_h {
                for x in 0..img_w {
                    let src_off = y * src_stride + x * channels;
                    let dst_x = y;
                    let dst_y = img_w - 1 - x;
                    let dst_off = dst_y * new_w * channels + dst_x * channels;
                    scratch[dst_off..dst_off + channels]
                        .copy_from_slice(&data[src_off..src_off + channels]);
                }
            }
            data[..scratch.len()].copy_from_slice(scratch);
            *w = new_w;
            *h = new_h;
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exif_orientation_default() {
        assert_eq!(read_exif_orientation(&[]), (0, false));
    }
}
