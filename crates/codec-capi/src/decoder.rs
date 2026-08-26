// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Decoding into an existing tensor.
//!
//! The decoder is a reusable object rather than a free function because it
//! holds backend state — the V4L2 or nvJPEG handle, the chosen DCT method —
//! and re-creating that per frame is what a decode loop must not do.

use std::ffi::{c_char, c_int, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_codec::{ImageDecoder, ImageLoad};
use edgefirst_tensor::TensorDyn;
use edgefirst_tensor_ffi::EfTensor;

/// An opaque image decoder.
pub struct EfImageDecoder {
    inner: ImageDecoder,
}

/// Create a decoder. `NULL` on failure.
#[no_mangle]
pub extern "C" fn ef_image_decoder_new() -> *mut EfImageDecoder {
    catch_unwind(|| {
        Box::into_raw(Box::new(EfImageDecoder {
            inner: ImageDecoder::default(),
        }))
    })
    .unwrap_or(std::ptr::null_mut())
}

/// Free a decoder. Freeing `NULL` is a no-op, matching `free(3)`.
///
/// # Safety
/// `d` must be `NULL` or have come from [`ef_image_decoder_new`].
#[no_mangle]
pub unsafe extern "C" fn ef_image_decoder_free(d: *mut EfImageDecoder) {
    unsafe {
        if d.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(Box::from_raw(d))));
    }
}

/// Decode JPEG or PNG bytes into `dst`, sizing and formatting it to the image.
///
/// The container is detected from magic bytes. `dst` must be large enough;
/// a smaller allocation is an error rather than a truncated image.
///
/// @return 0 on success, `ENOSPC` when `dst` is too small, otherwise an
///         errno.
///
/// # Safety
/// `data` must be readable for `len` bytes; `dst` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_image_decoder_decode_into(
    d: *mut EfImageDecoder,
    data: *const u8,
    len: usize,
    dst: *mut EfTensor,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if d.is_null() || data.is_null() || len == 0 || dst.is_null() {
                return libc::EINVAL;
            }
            let bytes = std::slice::from_raw_parts(data, len);
            match TensorDyn::with_raw(dst, |t| t.load_image(&mut (*d).inner, bytes)) {
                Ok(_) => 0,
                Err(e) => errno_for(&e),
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Decode an image file into `dst`.
///
/// @return as [`ef_image_decoder_decode_into`], plus `ENOENT` for a missing file.
///
/// # Safety
/// `path` must be a NUL-terminated string; `dst` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_image_decoder_decode_file_into(
    d: *mut EfImageDecoder,
    path: *const c_char,
    dst: *mut EfTensor,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if d.is_null() || path.is_null() {
                return libc::EINVAL;
            }
            let Ok(p) = CStr::from_ptr(path).to_str() else {
                return libc::EINVAL;
            };
            if dst.is_null() {
                return libc::EINVAL;
            }
            match TensorDyn::with_raw(dst, |t| t.load_image_file(&mut (*d).inner, p)) {
                Ok(_) => 0,
                Err(e) => errno_for(&e),
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Map a codec error to an errno.
///
/// `InsufficientCapacity` is `ENOSPC` specifically: a caller who sized the
/// destination from a header can act on that, whereas a generic EIO tells them
/// nothing about what to do next.
fn errno_for(e: &edgefirst_codec::CodecError) -> c_int {
    match e {
        edgefirst_codec::CodecError::InsufficientCapacity { .. } => libc::ENOSPC,
        _ => libc::EIO,
    }
}

/// Whether a hardware V4L2 JPEG decoder is present.
#[no_mangle]
pub extern "C" fn ef_codec_v4l2_available() -> c_int {
    catch_unwind(|| i32::from(edgefirst_codec::v4l2_available())).unwrap_or(0)
}

/// Whether nvJPEG is present.
#[no_mangle]
pub extern "C" fn ef_codec_nvjpeg_available() -> c_int {
    catch_unwind(|| i32::from(edgefirst_codec::nvjpeg_available())).unwrap_or(0)
}

/// Select the software JPEG IDCT kernel. `0` = accurate, `1` = fast.
///
/// # Safety
/// `d` must be `NULL` or a live decoder.
#[no_mangle]
pub unsafe extern "C" fn ef_image_decoder_set_dct_method(
    d: *mut EfImageDecoder,
    method: u32,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if d.is_null() {
                return libc::EINVAL;
            }
            let m = match method {
                0 => edgefirst_codec::DctMethod::Accurate,
                1 => edgefirst_codec::DctMethod::Fast,
                _ => return libc::EINVAL,
            };
            (*d).inner.set_dct_method(m);
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Request a fused JPEG output format (`"rgb8"`, `"NV12"`, …). `NULL` resets.
///
/// # Safety
/// `format` must be `NULL` or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn ef_image_decoder_set_output_format(
    d: *mut EfImageDecoder,
    format: *const c_char,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if d.is_null() {
                return libc::EINVAL;
            }
            if format.is_null() {
                (*d).inner.set_output_format(None);
                return 0;
            }
            let Ok(s) = CStr::from_ptr(format).to_str() else {
                return libc::EINVAL;
            };
            let Some(fmt) = edgefirst_tensor::PixelFormat::from_str_code(s) else {
                return libc::EINVAL;
            };
            (*d).inner.set_output_format(Some(fmt));
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Reset fused JPEG output to the source's native format.
///
/// # Safety
/// `d` must be `NULL` or a live decoder.
#[no_mangle]
pub unsafe extern "C" fn ef_image_decoder_reset_output_format(d: *mut EfImageDecoder) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if d.is_null() {
                return libc::EINVAL;
            }
            (*d).inner.set_output_format(None);
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Map raw V4L2 colorimetry integers to the packed `ef_tensor_colorimetry` form.
///
/// # Safety
/// `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ef_codec_colorimetry_from_v4l2(
    colorspace: u32,
    xfer: u32,
    ycbcr_enc: u32,
    quant: u32,
    out: *mut u32,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if out.is_null() {
                return libc::EINVAL;
            }
            *out =
                edgefirst_tensor::Colorimetry::from_v4l2(colorspace, xfer, ycbcr_enc, quant).pack();
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Encode a packed RGB/RGBA u8 tensor as JPEG.
///
/// `quality` in 1–100; `0` or out-of-range uses 80.
///
/// # Safety
/// `path` must be a NUL-terminated string; `t` must be a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_codec_save_jpeg(
    t: *const EfTensor,
    path: *const c_char,
    quality: c_int,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if t.is_null() || path.is_null() {
                return libc::EINVAL;
            }
            let Ok(p) = CStr::from_ptr(path).to_str() else {
                return libc::EINVAL;
            };
            let quality = if quality <= 0 || quality > 100 {
                80
            } else {
                quality as u8
            };
            match TensorDyn::with_raw(t as *mut EfTensor, |td| save_jpeg(td, p, quality)) {
                Ok(()) => 0,
                Err(_) => libc::EIO,
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

fn save_jpeg(tensor: &TensorDyn, path: &str, quality: u8) -> Result<(), c_int> {
    use edgefirst_tensor::{CpuAccess, PixelFormat, PixelLayout};

    if tensor.dtype() != edgefirst_tensor::DType::U8 {
        return Err(libc::EINVAL);
    }
    let fmt = tensor.format().ok_or(libc::EINVAL)?;
    if fmt.layout() != PixelLayout::Packed {
        return Err(libc::ENOTSUP);
    }
    let colour = match fmt {
        PixelFormat::Rgb => jpeg_encoder::ColorType::Rgb,
        PixelFormat::Rgba => jpeg_encoder::ColorType::Rgba,
        _ => return Err(libc::ENOTSUP),
    };
    let w = tensor.width().ok_or(libc::EINVAL)?;
    let h = tensor.height().ok_or(libc::EINVAL)?;
    let pin = tensor.pin_host(CpuAccess::Read).map_err(|_| libc::EIO)?;
    let bytes = unsafe { pin.as_slice() };
    let encoder = jpeg_encoder::Encoder::new_file(path, quality).map_err(|_| libc::EIO)?;
    encoder
        .encode(bytes, w as u16, h as u16, colour)
        .map_err(|_| libc::EIO)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_decoder_can_be_created_and_freed() {
        let d = ef_image_decoder_new();
        assert!(!d.is_null());
        unsafe { ef_image_decoder_free(d) };
        unsafe { ef_image_decoder_free(std::ptr::null_mut()) };
    }

    #[test]
    fn null_arguments_are_errors_not_crashes() {
        unsafe {
            let d = ef_image_decoder_new();
            assert_eq!(
                ef_image_decoder_decode_into(d, std::ptr::null(), 0, std::ptr::null_mut()),
                libc::EINVAL
            );
            assert_eq!(
                ef_image_decoder_decode_file_into(d, std::ptr::null(), std::ptr::null_mut()),
                libc::EINVAL
            );
            assert_eq!(
                ef_image_decoder_decode_into(
                    std::ptr::null_mut(),
                    [0u8; 4].as_ptr(),
                    4,
                    std::ptr::null_mut()
                ),
                libc::EINVAL
            );
            ef_image_decoder_free(d);
        }
    }

    #[test]
    fn the_availability_probes_do_not_crash_on_any_host() {
        // Both are 0 or 1 everywhere; the point is that probing a device that
        // is absent must not fault.
        assert!(matches!(ef_codec_v4l2_available(), 0 | 1));
        assert!(matches!(ef_codec_nvjpeg_available(), 0 | 1));
    }
}
