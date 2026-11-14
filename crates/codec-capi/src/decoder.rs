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
/// @return 0 on success, `ENOSPC` when `dst` is too small, `EACCES` when the
///         CPU decoder must write a `dst` not declared CPU-writable,
///         `EBADMSG` for data that is not a decodable image, `ENOTSUP` for a
///         feature or destination format the decoder does not support,
///         otherwise an errno, with the reason in `ef_codec_last_error`.
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
        crate::error::guard(|| {
            if d.is_null() || dst.is_null() {
                return crate::error::invalid("the decoder or the destination tensor is NULL");
            }
            if data.is_null() || len == 0 {
                return crate::error::invalid("the image data is empty");
            }
            let bytes = std::slice::from_raw_parts(data, len);
            match TensorDyn::with_raw(dst, |t| t.load_image(&mut (*d).inner, bytes)) {
                Ok(_) => 0,
                Err(e) => crate::error::fail(&e),
            }
        })
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
        crate::error::guard(|| {
            if d.is_null() || path.is_null() {
                return crate::error::invalid("the decoder or the path is NULL");
            }
            let Ok(p) = CStr::from_ptr(path).to_str() else {
                return crate::error::invalid("the path is not UTF-8");
            };
            if dst.is_null() {
                return crate::error::invalid("the destination tensor is NULL");
            }
            match TensorDyn::with_raw(dst, |t| t.load_image_file(&mut (*d).inner, p)) {
                Ok(_) => 0,
                Err(e) => crate::error::fail(&e),
            }
        })
    }
}

/// `EACCES` for a tensor whose declared CPU access does not cover the work,
/// else `EIO`.
fn tensor_errno(e: &edgefirst_tensor::Error) -> c_int {
    if e.cpu_access_denied().is_some() {
        return libc::EACCES;
    }
    libc::EIO
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

/// Selects the engine that decodes JPEG, an `ef_codec_engine`:
/// `EF_CODEC_ENGINE_AUTO` (nvJPEG, V4L2, then the CPU, where present), or
/// one engine used alone (`EF_CODEC_ENGINE_CPU`, `EF_CODEC_ENGINE_V4L2`,
/// `EF_CODEC_ENGINE_NVJPEG`). PNG is decoded on the CPU under `AUTO` and
/// `CPU` only.
///
/// @return 0, or `EINVAL`.
///
/// # Safety
/// `d` must be a live decoder.
#[no_mangle]
pub unsafe extern "C" fn ef_image_decoder_set_engine(d: *mut EfImageDecoder, engine: u32) -> c_int {
    crate::error::guard(|| {
        let Some(d) = (unsafe { d.as_mut() }) else {
            return crate::error::invalid("the decoder is NULL");
        };
        match crate::engine::engine_arg(engine) {
            Ok(engine) => {
                d.inner.set_engine(engine);
                0
            }
            Err(rc) => rc,
        }
    })
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
        crate::error::guard(|| {
            if d.is_null() {
                return crate::error::invalid("the decoder is NULL");
            }
            let m = match method {
                0 => edgefirst_codec::DctMethod::Accurate,
                1 => edgefirst_codec::DctMethod::Fast,
                _ => return crate::error::invalid(&format!("DCT method {method} is not 0 or 1")),
            };
            (*d).inner.set_dct_method(m);
            0
        })
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
        crate::error::guard(|| {
            if d.is_null() {
                return crate::error::invalid("the decoder is NULL");
            }
            if format.is_null() {
                (*d).inner.set_output_format(None);
                return 0;
            }
            let Ok(s) = CStr::from_ptr(format).to_str() else {
                return crate::error::invalid("the pixel format is not UTF-8");
            };
            let Some(fmt) = edgefirst_tensor::PixelFormat::from_str_code(s) else {
                return crate::error::invalid(&format!("{s:?} is not a pixel format"));
            };
            (*d).inner.set_output_format(Some(fmt));
            0
        })
    }
}

/// Reset fused JPEG output to the source's native format.
///
/// # Safety
/// `d` must be `NULL` or a live decoder.
#[no_mangle]
pub unsafe extern "C" fn ef_image_decoder_reset_output_format(d: *mut EfImageDecoder) -> c_int {
    unsafe {
        crate::error::guard(|| {
            if d.is_null() {
                return crate::error::invalid("the decoder is NULL");
            }
            (*d).inner.set_output_format(None);
            0
        })
    }
}

/// Map raw V4L2 colorimetry integers to the packed `ef_tensor_colorimetry` form.
///
/// `is_rgb_or_hsv` is non-zero when the negotiated format's pixel encoding
/// is RGB or HSV. Per `V4L2_MAP_QUANTIZATION_DEFAULT`, a `DEFAULT` (0)
/// `quant` is full range for an RGB/HSV format, whatever the colorspace, and
/// for `V4L2_COLORSPACE_JPEG`; limited for a YUV format in any other
/// recognised colorspace; and unset (range byte 0) for a YUV format in a
/// `DEFAULT` or unrecognised colorspace, so the at-use height heuristic
/// decides.
///
/// # Safety
/// `out` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ef_codec_colorimetry_from_v4l2(
    colorspace: u32,
    xfer: u32,
    ycbcr_enc: u32,
    quant: u32,
    is_rgb_or_hsv: c_int,
    out: *mut u32,
) -> c_int {
    unsafe {
        crate::error::guard(|| {
            if out.is_null() {
                return crate::error::invalid("out is NULL");
            }
            *out = edgefirst_tensor::Colorimetry::from_v4l2(
                colorspace,
                xfer,
                ycbcr_enc,
                quant,
                is_rgb_or_hsv != 0,
            )
            .pack();
            0
        })
    }
}

/// Encode a packed RGB/RGBA u8 tensor as JPEG.
///
/// `quality` in 1–100; `0` or out-of-range uses 80.
///
/// @return 0 on success, `EACCES` when `t` is not declared CPU-readable,
///         otherwise an errno.
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
        crate::error::guard(|| {
            if t.is_null() || path.is_null() {
                return crate::error::invalid("the tensor or the path is NULL");
            }
            let Ok(p) = CStr::from_ptr(path).to_str() else {
                return crate::error::invalid("the path is not UTF-8");
            };
            let quality = if quality <= 0 || quality > 100 {
                80
            } else {
                quality as u8
            };
            match TensorDyn::with_raw(t as *mut EfTensor, |td| save_jpeg(td, p, quality)) {
                Ok(()) => 0,
                Err(e) => e,
            }
        })
    }
}

/// Writes `tensor` as a JPEG file; a failure records its reason.
fn save_jpeg(tensor: &TensorDyn, path: &str, quality: u8) -> Result<(), c_int> {
    use crate::error::refuse;
    use edgefirst_tensor::{CpuAccess, PixelFormat};

    if tensor.dtype() != edgefirst_tensor::DType::U8 {
        return Err(refuse(
            libc::EINVAL,
            &format!("JPEG is written from u8 tensors, not {:?}", tensor.dtype()),
        ));
    }
    let Some(fmt) = tensor.format() else {
        return Err(refuse(libc::EINVAL, "the tensor is not an image"));
    };
    let colour = match fmt {
        PixelFormat::Rgb => jpeg_encoder::ColorType::Rgb,
        PixelFormat::Rgba => jpeg_encoder::ColorType::Rgba,
        _ => {
            return Err(refuse(
                libc::ENOTSUP,
                &format!("JPEG is written from RGB or RGBA images, not {fmt:?}"),
            ))
        }
    };
    let (Some(w), Some(h)) = (tensor.width(), tensor.height()) else {
        return Err(refuse(libc::EINVAL, "the image has no width or height"));
    };
    let (Ok(w16), Ok(h16)) = (u16::try_from(w), u16::try_from(h)) else {
        return Err(refuse(
            libc::ENOTSUP,
            &format!("a {w}x{h} image is larger than JPEG allows (65535x65535)"),
        ));
    };
    let pin = tensor
        .pin_host(CpuAccess::Read)
        .map_err(|e| refuse(tensor_errno(&e), &e.to_string()))?;
    let bytes = unsafe { pin.as_slice() };
    let encoder = jpeg_encoder::Encoder::new_file(path, quality)
        .map_err(|e| refuse(libc::EIO, &format!("{path}: {e}")))?;
    encoder
        .encode(bytes, w16, h16, colour)
        .map_err(|e| refuse(libc::EIO, &format!("{path}: {e}")))
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

    fn last_error() -> String {
        let p = crate::error::ef_codec_last_error();
        assert!(!p.is_null(), "a failed call left no message");
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned()
    }

    #[test]
    fn every_refusal_says_why() {
        use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorMemory};
        unsafe {
            let d = ef_image_decoder_new();
            let dst = TensorDyn::image(
                8,
                8,
                PixelFormat::Grey,
                DType::U8,
                Some(TensorMemory::Mem),
                CpuAccess::ReadWrite,
            )
            .unwrap()
            .into_raw();
            assert_eq!(
                ef_image_decoder_decode_into(d, [0u8].as_ptr(), 0, dst),
                libc::EINVAL
            );
            assert!(last_error().contains("empty"));
            assert_eq!(ef_image_decoder_set_dct_method(d, 9), libc::EINVAL);
            assert!(last_error().contains("DCT method 9"));
            let bogus = c"not-a-format";
            assert_eq!(
                ef_image_decoder_set_output_format(d, bogus.as_ptr()),
                libc::EINVAL
            );
            assert!(last_error().contains("not-a-format"));
            let path = c"never-written.jpg";
            assert_eq!(ef_codec_save_jpeg(dst, path.as_ptr(), 80), libc::ENOTSUP);
            assert!(last_error().contains("RGB or RGBA"));
            let mut out = 0;
            assert_eq!(ef_codec_colorimetry_from_v4l2(0, 0, 0, 0, 0, &mut out), 0);
            assert!(crate::error::ef_codec_last_error().is_null());
            edgefirst_tensor_ffi::ef_tensor_free(dst);
            ef_image_decoder_free(d);
        }
    }

    /// Decoding into a destination not declared CPU-writable, and saving one
    /// not declared CPU-readable, are `EACCES`, as a tensor map is.
    #[test]
    fn a_cpu_access_refusal_is_eacces() {
        use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorMemory};
        let jpeg = include_bytes!("../../../testdata/coco_grey_odd.jpg");
        let image = |access| {
            TensorDyn::image(
                640,
                480,
                PixelFormat::Grey,
                DType::U8,
                Some(TensorMemory::Mem),
                access,
            )
            .unwrap()
            .into_raw()
        };
        // Both saves are refused before the file is created.
        let path = std::ffi::CString::new("never-written.jpg").unwrap();
        unsafe {
            let d = ef_image_decoder_new();
            let read_only = image(CpuAccess::Read);
            assert_eq!(
                ef_image_decoder_decode_into(d, jpeg.as_ptr(), jpeg.len(), read_only),
                libc::EACCES
            );
            let write_only = image(CpuAccess::Write);
            assert_eq!(
                ef_image_decoder_decode_into(d, jpeg.as_ptr(), jpeg.len(), write_only),
                0
            );
            assert_eq!(
                ef_codec_save_jpeg(write_only, path.as_ptr(), 80),
                libc::ENOTSUP,
                "a GREY image is refused before any access is checked"
            );
            edgefirst_tensor_ffi::ef_tensor_free(read_only);
            edgefirst_tensor_ffi::ef_tensor_free(write_only);
            ef_image_decoder_free(d);
        }
        let rgb = TensorDyn::image(
            8,
            8,
            PixelFormat::Rgb,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::Write,
        )
        .unwrap()
        .into_raw();
        unsafe {
            assert_eq!(ef_codec_save_jpeg(rgb, path.as_ptr(), 80), libc::EACCES);
            edgefirst_tensor_ffi::ef_tensor_free(rgb);
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
