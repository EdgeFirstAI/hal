// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Still-image encoding from a tensor.

use std::ffi::c_int;

use edgefirst_codec::ImageEncoder;
use edgefirst_tensor::TensorDyn;
use edgefirst_tensor_ffi::EfTensor;

use crate::engine::{codec_arg, engine_arg};
use crate::error::{fail, guard, guard_new, invalid};

/// An opaque still-image encoder.
pub struct EfImageEncoder {
    inner: ImageEncoder,
    /// The last encoded file, reused by the next encode.
    out: Vec<u8>,
}

/// Creates an encoder for `codec`, an `ef_codec`. Only `EF_CODEC_JPEG` is
/// encoded; the quality starts at 85 and the engine at
/// `EF_CODEC_ENGINE_AUTO`.
///
/// Under `EF_CODEC_ENGINE_AUTO` a V4L2 JPEG encoder takes DMA-BUF sources it
/// can read and the CPU takes the rest. The CPU encodes `u8` NV12, NV16,
/// YUYV, GREY, RGB and RGBA images at any row stride.
///
/// @return the encoder, freed with [`ef_image_encoder_free`]; `NULL` with
///         `errno` `ENOTSUP` for a codec it does not encode or `EINVAL` for
///         a value outside `ef_codec`.
#[no_mangle]
pub extern "C" fn ef_image_encoder_new(codec: u32) -> *mut EfImageEncoder {
    guard_new(|| {
        let codec = codec_arg(codec)?;
        let inner = ImageEncoder::new(codec).map_err(|e| fail(&e))?;
        Ok(EfImageEncoder {
            inner,
            out: Vec::new(),
        })
    })
}

/// Frees an encoder. Freeing `NULL` is a no-op.
///
/// # Safety
/// `e` must be `NULL` or have come from [`ef_image_encoder_new`].
#[no_mangle]
pub unsafe extern "C" fn ef_image_encoder_free(e: *mut EfImageEncoder) {
    if !e.is_null() {
        drop(unsafe { Box::from_raw(e) });
    }
}

/// Selects the engine, an `ef_codec_engine`: `EF_CODEC_ENGINE_AUTO`, or
/// one engine used alone (`EF_CODEC_ENGINE_CPU`, `EF_CODEC_ENGINE_V4L2`).
///
/// @return 0, or `EINVAL`.
///
/// # Safety
/// `e` must be a live encoder.
#[no_mangle]
pub unsafe extern "C" fn ef_image_encoder_set_engine(e: *mut EfImageEncoder, engine: u32) -> c_int {
    guard(|| {
        let Some(e) = (unsafe { e.as_mut() }) else {
            return invalid("the encoder is NULL");
        };
        match engine_arg(engine) {
            Ok(engine) => {
                e.inner.set_engine(engine);
                0
            }
            Err(rc) => rc,
        }
    })
}

/// Sets the JPEG quality, 1 (smallest) to 100 (best).
///
/// @return 0, or `EINVAL` outside 1 to 100.
///
/// # Safety
/// `e` must be a live encoder.
#[no_mangle]
pub unsafe extern "C" fn ef_image_encoder_set_quality(
    e: *mut EfImageEncoder,
    quality: c_int,
) -> c_int {
    guard(|| {
        let Some(e) = (unsafe { e.as_mut() }) else {
            return invalid("the encoder is NULL");
        };
        let Ok(quality) = u8::try_from(quality) else {
            return invalid(&format!("JPEG quality {quality} is not within 1 to 100"));
        };
        match e.inner.set_quality(quality) {
            Ok(()) => 0,
            Err(err) => fail(&err),
        }
    })
}

/// Encodes `src` and points `*data` and `*len` at the file.
///
/// The bytes belong to the encoder and stay valid until its next encode or
/// until it is freed; the next encode reuses the buffer.
///
/// @return 0; `EACCES` when the CPU must read a `src` not declared
///         CPU-readable; `ENOTSUP` for a source format or element type no
///         engine takes; `EINVAL` for a bad argument or source (for example
///         an odd-width YUYV image); `ENODEV` when the engine selected by
///         name is not present; otherwise an errno, with the reason in
///         [`ef_codec_last_error`](crate::error::ef_codec_last_error).
///
/// # Safety
/// `e` must be a live encoder, `src` a live tensor handle, and `data` and
/// `len` writable.
#[no_mangle]
pub unsafe extern "C" fn ef_image_encoder_encode(
    e: *mut EfImageEncoder,
    src: *const EfTensor,
    data: *mut *const u8,
    len: *mut usize,
) -> c_int {
    guard(|| {
        if data.is_null() || len.is_null() {
            return invalid("data and len must not be NULL");
        }
        unsafe {
            *data = std::ptr::null();
            *len = 0;
        }
        let Some(e) = (unsafe { e.as_mut() }) else {
            return invalid("the encoder is NULL");
        };
        if src.is_null() {
            return invalid("the source tensor is NULL");
        }
        let EfImageEncoder { inner, out } = e;
        let result =
            unsafe { TensorDyn::with_raw(src as *mut EfTensor, |t| inner.encode_into(t, out)) };
        match result {
            Ok(()) => {
                unsafe {
                    *data = out.as_ptr();
                    *len = out.len();
                }
                0
            }
            Err(err) => fail(&err),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{EfCodec, EfCodecEngine};
    use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorMemory};

    fn grey(width: usize, height: usize, access: CpuAccess) -> *mut EfTensor {
        let t = TensorDyn::image(
            width,
            height,
            PixelFormat::Grey,
            DType::U8,
            Some(TensorMemory::Mem),
            access,
        )
        .unwrap();
        t.into_raw()
    }

    #[test]
    fn only_jpeg_has_an_encoder() {
        let png = ef_image_encoder_new(EfCodec::Png as u32);
        assert!(png.is_null());
        assert_eq!(errno::errno().0, libc::ENOTSUP);
        assert!(ef_image_encoder_new(99).is_null());
        assert_eq!(errno::errno().0, libc::EINVAL);
    }

    #[test]
    fn a_grey_image_encodes_to_a_jpeg_on_the_cpu() {
        unsafe {
            let e = ef_image_encoder_new(EfCodec::Jpeg as u32);
            assert!(!e.is_null());
            assert_eq!(ef_image_encoder_set_engine(e, EfCodecEngine::Cpu as u32), 0);
            assert_eq!(ef_image_encoder_set_quality(e, 0), libc::EINVAL);
            assert_eq!(ef_image_encoder_set_quality(e, 101), libc::EINVAL);
            assert_eq!(ef_image_encoder_set_quality(e, 90), 0);
            assert_eq!(ef_image_encoder_set_engine(e, 42), libc::EINVAL);
            let src = grey(64, 48, CpuAccess::ReadWrite);
            let mut data = std::ptr::null();
            let mut len = 0;
            assert_eq!(ef_image_encoder_encode(e, src, &mut data, &mut len), 0);
            let file = std::slice::from_raw_parts(data, len);
            assert_eq!(&file[..2], [0xff, 0xd8]);
            let info = edgefirst_codec::peek_info(file).unwrap();
            assert_eq!((info.width, info.height), (64, 48));
            edgefirst_tensor_ffi::ef_tensor_free(src);
            ef_image_encoder_free(e);
        }
    }

    #[test]
    fn a_source_the_cpu_may_not_read_is_eacces() {
        unsafe {
            let e = ef_image_encoder_new(EfCodec::Jpeg as u32);
            assert_eq!(ef_image_encoder_set_engine(e, EfCodecEngine::Cpu as u32), 0);
            let src = grey(16, 16, CpuAccess::Write);
            let mut data = std::ptr::null();
            let mut len = 0;
            assert_eq!(
                ef_image_encoder_encode(e, src, &mut data, &mut len),
                libc::EACCES
            );
            assert!(data.is_null());
            assert_eq!(len, 0);
            assert!(!crate::error::ef_codec_last_error().is_null());
            edgefirst_tensor_ffi::ef_tensor_free(src);
            ef_image_encoder_free(e);
        }
    }
}
