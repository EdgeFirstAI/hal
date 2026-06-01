// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Custom baseline JPEG decoder with a zero-allocation hot loop and strided
//! native output.
//!
//! The decoder emits the source's native format only — `Grey` for greyscale
//! (1-component) JPEGs, `Nv12` (4:2:0, with chroma downsampled from any source
//! subsampling) for colour (3-component) JPEGs. It never converts to RGB and
//! never rotates: colour and geometry are applied downstream by
//! `ImageProcessor::convert()`. EXIF orientation is reported in [`ImageInfo`].

pub mod bitstream;
pub mod huffman;
pub mod idct;
pub mod markers;
pub mod mcu;
pub mod types;

/// Optional V4L2 hardware JPEG-decoder backend (Linux only, `v4l2` feature).
#[cfg(all(target_os = "linux", feature = "v4l2"))]
mod v4l2;

use crate::error::{CodecError, UnsupportedFeature};
use crate::exif::read_exif_orientation;
use crate::options::ImageInfo;
use crate::pixel::ImagePixel;
use edgefirst_tensor::{PixelFormat, Tensor, TensorTrait};

/// Reusable JPEG decoder state.
///
/// Holds scratch buffers that grow to the high-water mark and are reused across
/// decode calls. After the first frame at a given resolution, subsequent
/// decodes perform zero heap allocations.
pub struct JpegDecoderState {
    /// MCU scratch buffers (per-component IDCT output for one MCU row band).
    mcu_scratch: Option<mcu::McuScratch>,
    /// V4L2 hardware decoder, lazily probed on first decode. Probed at most
    /// once; a ready context is reused (and amortises per-image setup) across
    /// decodes.
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    v4l2: v4l2::V4l2Probe,
}

impl JpegDecoderState {
    pub fn new() -> Self {
        Self {
            mcu_scratch: None,
            #[cfg(all(target_os = "linux", feature = "v4l2"))]
            v4l2: v4l2::V4l2Probe::default(),
        }
    }
}

impl Default for JpegDecoderState {
    fn default() -> Self {
        Self::new()
    }
}

/// The codec's native output format for a JPEG: `Grey` for 1-component
/// (greyscale) images, `Nv12` for 3-component (YCbCr) images.
fn native_format(headers: &markers::JpegHeaders) -> crate::Result<PixelFormat> {
    match headers.header.components.len() {
        1 => Ok(PixelFormat::Grey),
        3 => Ok(PixelFormat::Nv12),
        n => Err(CodecError::Unsupported(
            UnsupportedFeature::JpegComponentCount {
                components: n as u8,
            },
        )),
    }
}

/// Native luma/primary-plane row stride in bytes for a freshly decoded image
/// (NV12 luma and GREY are both 1 byte/pixel).
fn native_row_stride(width: usize) -> usize {
    width
}

/// Parse JPEG headers and return native dimensions, format, and EXIF
/// orientation without decoding pixels. The codec does not rotate, so `width`
/// and `height` are the source's true dimensions and the orientation is
/// reported for the caller to apply via `convert()`.
pub fn peek_jpeg_info(data: &[u8]) -> crate::Result<ImageInfo> {
    let headers = markers::parse_markers(data)?;
    let img_w = headers.header.width as usize;
    let img_h = headers.header.height as usize;
    let format = native_format(&headers)?;
    let (rotation_degrees, flip_horizontal) = headers
        .exif_data
        .as_deref()
        .map(read_exif_orientation)
        .unwrap_or((0, false));
    Ok(ImageInfo {
        width: img_w,
        height: img_h,
        format,
        row_stride: native_row_stride(img_w),
        rotation_degrees,
        flip_horizontal,
    })
}

/// Decode a JPEG image from `data` into `dst`, configuring `dst`'s dimensions
/// and pixel format to the decoded native format.
///
/// The destination must be a `u8` tensor with enough underlying capacity for
/// the decoded image; smaller allocations yield
/// [`CodecError::InsufficientCapacity`]. Decoded rows are written at the
/// tensor's `effective_row_stride()` offsets. The returned [`ImageInfo`]
/// reports the EXIF orientation for the caller to apply downstream.
pub fn decode_jpeg_into<T: ImagePixel>(
    data: &[u8],
    dst: &mut Tensor<T>,
    state: &mut JpegDecoderState,
) -> crate::Result<ImageInfo> {
    let _span = tracing::trace_span!(
        "codec.decode_jpeg",
        dtype = std::any::type_name::<T>(),
        n_bytes = data.len(),
    )
    .entered();

    let headers = {
        let _s = tracing::trace_span!("codec.decode_jpeg.parse_markers").entered();
        markers::parse_markers(data)?
    };
    let img_w = headers.header.width as usize;
    let img_h = headers.header.height as usize;
    let output_fmt = native_format(&headers)?;

    // NV12 supports odd dimensions. Odd *height* gives a `H + ceil(H/2)`
    // combined-plane height (`PixelFormat::image_shape`). Odd *width* rounds the
    // tensor's buffer width up to even (also via `image_shape`), so the MCU
    // writer's `ceil(width/2)` chroma columns are byte-aligned; the reported
    // `ImageInfo.width` below stays the true odd value for a downstream crop.

    // Native NV12/GREY are u8 formats; reject non-u8 destinations.
    if T::dtype() != edgefirst_tensor::DType::U8 {
        return Err(CodecError::UnsupportedDtype(T::dtype()));
    }

    // EXIF orientation is reported, never applied.
    let (rotation_degrees, flip_horizontal) = headers
        .exif_data
        .as_deref()
        .map(read_exif_orientation)
        .unwrap_or((0, false));

    // Configure the destination tensor to the decoded image (or error if its
    // allocation is too small).
    let cap_w = dst.width().unwrap_or(0);
    let cap_h = dst.height().unwrap_or(0);
    dst.configure_image(img_w, img_h, output_fmt)
        .map_err(|e| match e {
            edgefirst_tensor::Error::InsufficientCapacity { .. } => {
                CodecError::InsufficientCapacity {
                    image: (img_w, img_h),
                    tensor: (cap_w, cap_h),
                }
            }
            other => CodecError::Tensor(other),
        })?;
    let dst_stride = dst
        .effective_row_stride()
        .unwrap_or(native_row_stride(img_w));

    // Try the V4L2 hardware decoder first when available; a probed-but-failing
    // device resets itself and falls through to the CPU decoder transparently.
    #[cfg(all(target_os = "linux", feature = "v4l2"))]
    {
        match state
            .v4l2
            .try_decode::<T>(data, &headers, dst, output_fmt, img_w, img_h, dst_stride)
        {
            Ok(Some(mut info)) => {
                info.rotation_degrees = rotation_degrees;
                info.flip_horizontal = flip_horizontal;
                return Ok(info);
            }
            Ok(None) => {}
            Err(v4l2::V4l2Decode::Fallback(why)) => {
                log::debug!("v4l2 jpeg decode fell back to cpu: {why}");
            }
            Err(v4l2::V4l2Decode::Fatal(e)) => return Err(e),
        }
    }

    // CPU decode: MCU loop writes NV12/GREY u8 directly into the tensor.
    match &mut state.mcu_scratch {
        Some(scratch) => scratch.ensure_capacity(&headers),
        None => state.mcu_scratch = Some(mcu::McuScratch::new(&headers)),
    }
    let mcu_scratch = state.mcu_scratch.as_mut().unwrap();

    {
        let mut map = dst.map()?;
        let dst_bytes: &mut [T] = &mut map;
        // SAFETY: T is u8 (checked above) — layout-identical reinterpret.
        let dst_u8: &mut [u8] = unsafe {
            std::slice::from_raw_parts_mut(dst_bytes.as_mut_ptr() as *mut u8, dst_bytes.len())
        };
        let _s = tracing::trace_span!("codec.decode_jpeg.mcu_loop").entered();
        mcu::decode_image(data, &headers, mcu_scratch, dst_u8, dst_stride, output_fmt)?;
    }

    Ok(ImageInfo {
        width: img_w,
        height: img_h,
        format: output_fmt,
        row_stride: dst_stride,
        rotation_degrees,
        flip_horizontal,
    })
}
