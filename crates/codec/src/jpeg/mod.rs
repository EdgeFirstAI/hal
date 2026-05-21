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
pub mod convert;
pub mod huffman;
pub mod idct;
pub mod markers;
pub mod mcu;
pub mod types;
pub mod upsample;

use crate::error::CodecError;
use crate::exif::{apply_exif_u8, read_exif_orientation, rotated_dims};
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
    /// Pre-rotation pixel scratch (also reused as u8 staging for non-u8 outputs).
    exif_scratch: Vec<u8>,
    /// Post-rotation pixel scratch used by `apply_exif_u8` for 90°/270°.
    /// Reused across decodes so EXIF-rotated workloads don't re-allocate
    /// a multi-megabyte buffer per frame.
    rot_scratch: Vec<u8>,
}

impl JpegDecoderState {
    pub fn new() -> Self {
        Self {
            mcu_scratch: None,
            exif_scratch: Vec::new(),
            rot_scratch: Vec::new(),
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
        PixelFormat::Rgb
        | PixelFormat::Rgba
        | PixelFormat::Grey
        | PixelFormat::Bgra
        | PixelFormat::Nv12 => Ok(fmt),
        _ => Err(CodecError::UnsupportedFormat(fmt)),
    }
}

/// Parse JPEG headers and return image dimensions/format without decoding pixels.
///
/// For `apply_exif=true` and a JPEG with a 90°/270° EXIF orientation tag,
/// the returned `width` and `height` reflect the **post-rotation** layout —
/// they match exactly what a subsequent `decode_jpeg_into` call would write.
pub fn peek_jpeg_info(data: &[u8], opts: &DecodeOptions) -> crate::Result<ImageInfo> {
    let headers = markers::parse_markers(data)?;
    let hdr = &headers.header;
    let img_w = hdr.width as usize;
    let img_h = hdr.height as usize;

    let dest_fmt = opts.format.unwrap_or(PixelFormat::Rgb);
    let output_fmt = validate_output_format(dest_fmt)?;

    if hdr.components.len() == 1 && output_fmt == PixelFormat::Nv12 {
        return Err(CodecError::InvalidData(
            "cannot decode greyscale JPEG to NV12".into(),
        ));
    }
    if output_fmt == PixelFormat::Nv12 && (!img_w.is_multiple_of(2) || !img_h.is_multiple_of(2)) {
        return Err(CodecError::InvalidData(format!(
            "NV12 requires even dimensions; got {img_w}×{img_h}"
        )));
    }

    let (rotation_deg, _flip_h) = if opts.apply_exif && output_fmt != PixelFormat::Nv12 {
        headers
            .exif_data
            .as_deref()
            .map(read_exif_orientation)
            .unwrap_or((0, false))
    } else {
        (0, false)
    };

    let (final_w, final_h) = rotated_dims(img_w, img_h, rotation_deg);
    let channels = output_fmt.channels();
    Ok(ImageInfo {
        width: final_w,
        height: final_h,
        format: output_fmt,
        row_stride: final_w * channels,
    })
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
    let _span = tracing::trace_span!(
        "codec.decode_jpeg",
        dtype = std::any::type_name::<T>(),
        n_bytes = data.len(),
    )
    .entered();
    // Parse all marker segments
    let headers = {
        let _s = tracing::trace_span!("codec.decode_jpeg.parse_markers").entered();
        markers::parse_markers(data)?
    };

    let hdr = &headers.header;
    let mut img_w = hdr.width as usize;
    let mut img_h = hdr.height as usize;

    // Determine output format
    let dest_fmt = opts.format.unwrap_or(PixelFormat::Rgb);
    let output_fmt = validate_output_format(dest_fmt)?;

    // Handle greyscale JPEGs
    let output_fmt = if hdr.components.len() == 1 && output_fmt == PixelFormat::Nv12 {
        // Greyscale JPEG cannot produce NV12 with chroma
        return Err(CodecError::InvalidData(
            "cannot decode greyscale JPEG to NV12".into(),
        ));
    } else if hdr.components.len() == 1 && output_fmt != PixelFormat::Grey {
        // Greyscale JPEG: we can still output RGB/RGBA by expanding
        output_fmt
    } else {
        output_fmt
    };

    // NV12 requires even width/height by definition (chroma plane stores
    // one Cb/Cr pair per 2×2 luma block). Reject odd dimensions up front
    // rather than silently truncate or overflow the row stride.
    if output_fmt == PixelFormat::Nv12 && (!img_w.is_multiple_of(2) || !img_h.is_multiple_of(2)) {
        return Err(CodecError::InvalidData(format!(
            "NV12 requires even dimensions; got {img_w}×{img_h}"
        )));
    }

    // Read EXIF orientation (NV12 output does not support EXIF rotation)
    let (rotation_deg, flip_h) = if opts.apply_exif && output_fmt != PixelFormat::Nv12 {
        headers
            .exif_data
            .as_deref()
            .map(read_exif_orientation)
            .unwrap_or((0, false))
    } else {
        (0, false)
    };

    // After rotation, dimensions may swap
    let (final_w, final_h) = rotated_dims(img_w, img_h, rotation_deg);

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

        if flip_h || rotation_deg != 0 {
            // EXIF rotation/flip path: decode at native (pre-rotation) stride
            // into the codec's scratch buffer, rotate in-place there, then
            // stride-copy into the (possibly pitch-padded) destination. We
            // cannot decode directly into `dst_u8` when rotation swaps dims
            // because native rows would overflow the rotated tensor stride.
            let native_stride = img_w * channels;
            state.exif_scratch.resize(native_stride * img_h, 0);
            {
                let _s = tracing::trace_span!("codec.decode_jpeg.mcu_loop").entered();
                mcu::decode_image(
                    data,
                    &headers,
                    mcu_scratch,
                    &mut state.exif_scratch,
                    native_stride,
                    output_fmt,
                )?;
            }
            {
                let _s =
                    tracing::trace_span!("codec.decode_jpeg.apply_exif", rotation_deg, flip_h,)
                        .entered();
                apply_exif_u8(
                    &mut state.exif_scratch,
                    native_stride,
                    &mut img_w,
                    &mut img_h,
                    channels,
                    rotation_deg,
                    flip_h,
                    &mut state.rot_scratch,
                );
            }
            // exif_scratch now holds post-rotation pixels at stride
            // `img_w * channels`. Copy into the tensor at `dst_stride`,
            // which honors any pitch padding.
            let final_native_stride = img_w * channels;
            for y in 0..img_h {
                let src_off = y * final_native_stride;
                let dst_off = y * dst_stride;
                dst_u8[dst_off..dst_off + final_native_stride]
                    .copy_from_slice(&state.exif_scratch[src_off..src_off + final_native_stride]);
            }
        } else {
            let _s = tracing::trace_span!("codec.decode_jpeg.mcu_loop").entered();
            mcu::decode_image(data, &headers, mcu_scratch, dst_u8, dst_stride, output_fmt)?;
        }
    } else {
        // Generic path: decode into temporary u8 buffer, then convert
        let temp_stride = img_w * channels;
        let temp_size = temp_stride * img_h;
        state.exif_scratch.resize(temp_size, 0);

        {
            let _s = tracing::trace_span!("codec.decode_jpeg.mcu_loop").entered();
            mcu::decode_image(
                data,
                &headers,
                mcu_scratch,
                &mut state.exif_scratch,
                temp_stride,
                output_fmt,
            )?;
        }

        // Apply EXIF transformations on the temp buffer
        if flip_h || rotation_deg != 0 {
            let _s = tracing::trace_span!("codec.decode_jpeg.apply_exif", rotation_deg, flip_h,)
                .entered();
            apply_exif_u8(
                &mut state.exif_scratch,
                temp_stride,
                &mut img_w,
                &mut img_h,
                channels,
                rotation_deg,
                flip_h,
                &mut state.rot_scratch,
            );
        }

        // Convert u8 → T and write to tensor at stride offsets
        let _s = tracing::trace_span!(
            "codec.decode_jpeg.type_convert",
            dtype = std::any::type_name::<T>(),
        )
        .entered();
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
        } else if T::dtype() == edgefirst_tensor::DType::F32 {
            // SIMD-vectorized u8 → f32 normalisation
            let dst_f32: &mut [f32] = unsafe {
                std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut f32, dst_bytes.len())
            };
            for y in 0..img_h {
                let s = y * src_stride;
                let d = y * dst_stride_elems;
                convert::convert_u8_to_f32(
                    &state.exif_scratch[s..s + src_stride],
                    &mut dst_f32[d..d + src_stride],
                );
            }
        } else if T::dtype() == edgefirst_tensor::DType::U16 {
            // SIMD-vectorized u8 → u16 scaling (×257)
            let dst_u16: &mut [u16] = unsafe {
                std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut u16, dst_bytes.len())
            };
            for y in 0..img_h {
                let s = y * src_stride;
                let d = y * dst_stride_elems;
                convert::convert_u8_to_u16(
                    &state.exif_scratch[s..s + src_stride],
                    &mut dst_u16[d..d + src_stride],
                );
            }
        } else if T::dtype() == edgefirst_tensor::DType::I16 {
            // SIMD-vectorized u8 → i16 (×257 XOR 0x8000)
            let dst_i16: &mut [i16] = unsafe {
                std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut i16, dst_bytes.len())
            };
            for y in 0..img_h {
                let s = y * src_stride;
                let d = y * dst_stride_elems;
                convert::convert_u8_to_i16(
                    &state.exif_scratch[s..s + src_stride],
                    &mut dst_i16[d..d + src_stride],
                );
            }
        } else {
            // Fallback for any other type
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
