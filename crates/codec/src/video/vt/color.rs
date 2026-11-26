// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Colorimetry to and from CoreVideo colour attachments.

use edgefirst_tensor::{ColorEncoding, ColorRange, ColorSpace, ColorTransfer, Colorimetry};

use super::ffi::*;
use super::{FOURCC_420F, FOURCC_420V, FOURCC_YUVF, FOURCC_YUVS};
use crate::{CodecError, Result};

/// The CoreVideo attachment values for a colorimetry; an unset axis is
/// `None`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CoreVideoColour {
    pub(crate) primaries: Option<CFStringRef>,
    pub(crate) transfer: Option<CFStringRef>,
    pub(crate) matrix: Option<CFStringRef>,
}

/// The CoreVideo primaries, transfer function and matrix of `c`.
///
/// # Errors
///
/// [`CodecError::InvalidInput`] naming the axis and value when a set axis
/// has no CoreVideo value on this platform.
pub(crate) fn to_corevideo(c: &Colorimetry) -> Result<CoreVideoColour> {
    Ok(CoreVideoColour {
        primaries: corevideo_value("colour primaries", c.space, cv_primaries)?,
        transfer: corevideo_value("transfer function", c.transfer, cv_transfer)?,
        matrix: corevideo_value("YCbCr matrix", c.encoding, cv_matrix)?,
    })
}

/// The CoreVideo value `map` gives the `axis` value `value`, or `None` when
/// `value` is unset.
///
/// # Errors
///
/// [`CodecError::InvalidInput`] naming `axis` and `value` when `map` gives
/// none.
fn corevideo_value<T: Copy + std::fmt::Debug>(
    axis: &str,
    value: Option<T>,
    map: impl Fn(T) -> Option<CFStringRef>,
) -> Result<Option<CFStringRef>> {
    value
        .map(|v| {
            map(v).ok_or_else(|| {
                CodecError::InvalidInput(format!(
                    "the source's {axis} {v:?} has no CoreVideo value on this platform"
                ))
            })
        })
        .transpose()
}

/// The range of CoreVideo pixel format `fourcc`: `420v` and `yuvs` limited,
/// `420f` and `yuvf` full, and `None` for any other.
pub(crate) fn surface_range(fourcc: u32) -> Option<ColorRange> {
    match fourcc {
        FOURCC_420V | FOURCC_YUVS => Some(ColorRange::Limited),
        FOURCC_420F | FOURCC_YUVF => Some(ColorRange::Full),
        _ => None,
    }
}

/// The colorimetry `buffer`'s colour attachments describe, with the range
/// [`surface_range`] gives `pixel_format`.
///
/// # Safety
///
/// `buffer` is a live `CVImageBuffer`.
pub(crate) unsafe fn from_attachments(buffer: CVImageBufferRef, pixel_format: u32) -> Colorimetry {
    let attachment = |key: CFStringRef| {
        // SAFETY: the caller guarantees `buffer`, and `key` is a CoreVideo
        // constant. The attachment is borrowed (+0), not released.
        unsafe { CVBufferGetAttachment(buffer, key, std::ptr::null_mut()) }
    };
    let is = |value: CFTypeRef, constant: Option<CFStringRef>| {
        // SAFETY: `value` is a live attachment of `buffer` and `constant` a
        // CoreVideo constant.
        !value.is_null() && constant.is_some_and(|k| unsafe { CFEqual(value, k) } != 0)
    };
    // SAFETY: reads CoreVideo's key constants.
    let (primaries_key, transfer_key, matrix_key) = unsafe {
        (
            kCVImageBufferColorPrimariesKey,
            kCVImageBufferTransferFunctionKey,
            kCVImageBufferYCbCrMatrixKey,
        )
    };
    let value = attachment(primaries_key);
    let space = [ColorSpace::Bt709, ColorSpace::Smpte170m, ColorSpace::Bt2020]
        .into_iter()
        .find(|&s| is(value, cv_primaries(s)));
    let value = attachment(transfer_key);
    let transfer = [
        ColorTransfer::Bt709,
        ColorTransfer::Srgb,
        ColorTransfer::Pq,
        ColorTransfer::Hlg,
        ColorTransfer::Linear,
    ]
    .into_iter()
    .find(|&t| is(value, cv_transfer(t)));
    let value = attachment(matrix_key);
    let encoding = [
        ColorEncoding::Bt601,
        ColorEncoding::Bt709,
        ColorEncoding::Bt2020,
    ]
    .into_iter()
    .find(|&e| is(value, cv_matrix(e)));
    Colorimetry {
        space,
        transfer,
        encoding,
        range: surface_range(pixel_format),
    }
}

fn cv_primaries(space: ColorSpace) -> Option<CFStringRef> {
    // SAFETY: reads immutable constants CoreVideo defines.
    unsafe {
        match space {
            ColorSpace::Bt709 | ColorSpace::Srgb => Some(kCVImageBufferColorPrimaries_ITU_R_709_2),
            ColorSpace::Smpte170m => Some(kCVImageBufferColorPrimaries_SMPTE_C),
            ColorSpace::Bt2020 => Some(kCVImageBufferColorPrimaries_ITU_R_2020),
            _ => None,
        }
    }
}

fn cv_transfer(transfer: ColorTransfer) -> Option<CFStringRef> {
    // SAFETY: reads immutable constants CoreVideo defines.
    unsafe {
        match transfer {
            ColorTransfer::Bt709 => Some(kCVImageBufferTransferFunction_ITU_R_709_2),
            ColorTransfer::Srgb => Some(kCVImageBufferTransferFunction_sRGB),
            ColorTransfer::Pq => Some(kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ),
            ColorTransfer::Hlg => Some(kCVImageBufferTransferFunction_ITU_R_2100_HLG),
            #[cfg(target_os = "macos")]
            ColorTransfer::Linear => Some(kCVImageBufferTransferFunction_Linear),
            _ => None,
        }
    }
}

fn cv_matrix(encoding: ColorEncoding) -> Option<CFStringRef> {
    // SAFETY: reads immutable constants CoreVideo defines.
    unsafe {
        match encoding {
            ColorEncoding::Bt601 => Some(kCVImageBufferYCbCrMatrix_ITU_R_601_4),
            ColorEncoding::Bt709 => Some(kCVImageBufferYCbCrMatrix_ITU_R_709_2),
            ColorEncoding::Bt2020 => Some(kCVImageBufferYCbCrMatrix_ITU_R_2020),
            _ => None,
        }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;

    fn surface_buffer(fourcc: u32) -> CfOwned {
        // SAFETY: the dictionaries hold live CF objects, `buffer` is a valid
        // out-pointer, and `CVPixelBufferCreate` returns +1.
        unsafe {
            let surface = cf_dictionary(&[]).unwrap();
            let attributes =
                cf_dictionary(&[(kCVPixelBufferIOSurfacePropertiesKey, surface.as_ptr())]).unwrap();
            let mut buffer = std::ptr::null_mut();
            let status = CVPixelBufferCreate(
                std::ptr::null(),
                64,
                64,
                fourcc,
                attributes.as_ptr(),
                &mut buffer,
            );
            assert_eq!(status, kCVReturnSuccess);
            CfOwned::from_create(buffer).unwrap()
        }
    }

    fn attach(buffer: &CfOwned, colour: &CoreVideoColour) {
        // SAFETY: reads CoreVideo's key constants.
        let keys = unsafe {
            [
                (kCVImageBufferColorPrimariesKey, colour.primaries),
                (kCVImageBufferTransferFunctionKey, colour.transfer),
                (kCVImageBufferYCbCrMatrixKey, colour.matrix),
            ]
        };
        for (key, value) in keys {
            if let Some(value) = value {
                // SAFETY: `buffer` is a live pixel buffer, and the key and
                // value are CoreVideo constants.
                unsafe {
                    CVBufferSetAttachment(
                        buffer.as_ptr(),
                        key,
                        value,
                        kCVAttachmentMode_ShouldPropagate,
                    )
                };
            }
        }
    }

    fn read(buffer: &CfOwned, fourcc: u32) -> Colorimetry {
        // SAFETY: `buffer` is a live pixel buffer.
        unsafe { from_attachments(buffer.as_ptr(), fourcc) }
    }

    #[test]
    fn colorimetry_round_trips_through_corevideo_attachments() {
        let cases = [
            (
                Colorimetry {
                    space: Some(ColorSpace::Bt709),
                    transfer: Some(ColorTransfer::Bt709),
                    encoding: Some(ColorEncoding::Bt709),
                    range: Some(ColorRange::Limited),
                },
                FOURCC_420V,
            ),
            (
                Colorimetry {
                    space: Some(ColorSpace::Smpte170m),
                    transfer: Some(ColorTransfer::Bt709),
                    encoding: Some(ColorEncoding::Bt601),
                    range: Some(ColorRange::Full),
                },
                FOURCC_420F,
            ),
        ];
        for (c, fourcc) in cases {
            let buffer = surface_buffer(fourcc);
            attach(&buffer, &to_corevideo(&c).unwrap());
            assert_eq!(read(&buffer, fourcc), c);
        }
    }

    #[test]
    fn every_mapped_value_reads_back() {
        let cases = [
            (
                ColorSpace::Bt2020,
                ColorTransfer::Srgb,
                ColorEncoding::Bt2020,
            ),
            (ColorSpace::Bt709, ColorTransfer::Pq, ColorEncoding::Bt709),
            (ColorSpace::Bt709, ColorTransfer::Hlg, ColorEncoding::Bt601),
            (
                ColorSpace::Bt709,
                ColorTransfer::Linear,
                ColorEncoding::Bt601,
            ),
        ];
        for (space, transfer, encoding) in cases {
            let c = Colorimetry {
                space: Some(space),
                transfer: Some(transfer),
                encoding: Some(encoding),
                range: Some(ColorRange::Limited),
            };
            let buffer = surface_buffer(FOURCC_420V);
            attach(&buffer, &to_corevideo(&c).unwrap());
            assert_eq!(read(&buffer, FOURCC_420V), c);
        }
        let srgb = Colorimetry {
            space: Some(ColorSpace::Srgb),
            ..Colorimetry::default()
        };
        assert_eq!(
            to_corevideo(&srgb).unwrap().primaries,
            to_corevideo(&Colorimetry {
                space: Some(ColorSpace::Bt709),
                ..Colorimetry::default()
            })
            .unwrap()
            .primaries
        );
    }

    #[test]
    fn a_set_axis_without_a_corevideo_value_is_refused() {
        let unmapped = |_: ColorTransfer| None;
        let e = corevideo_value("transfer function", Some(ColorTransfer::Linear), unmapped)
            .unwrap_err();
        assert!(
            matches!(&e, CodecError::InvalidInput(why)
                if why.contains("transfer function") && why.contains("Linear")),
            "{e}"
        );
        assert_eq!(
            corevideo_value("transfer function", None, unmapped).unwrap(),
            None
        );
        assert!(
            corevideo_value("transfer function", Some(ColorTransfer::Pq), cv_transfer)
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn every_colour_tag_has_a_corevideo_value_on_macos() {
        for space in [
            ColorSpace::Bt709,
            ColorSpace::Bt2020,
            ColorSpace::Srgb,
            ColorSpace::Smpte170m,
        ] {
            for transfer in [
                ColorTransfer::Bt709,
                ColorTransfer::Srgb,
                ColorTransfer::Pq,
                ColorTransfer::Hlg,
                ColorTransfer::Linear,
            ] {
                for encoding in [
                    ColorEncoding::Bt601,
                    ColorEncoding::Bt709,
                    ColorEncoding::Bt2020,
                ] {
                    let c = Colorimetry {
                        space: Some(space),
                        transfer: Some(transfer),
                        encoding: Some(encoding),
                        range: None,
                    };
                    let cv = to_corevideo(&c).unwrap_or_else(|e| panic!("{c:?}: {e}"));
                    assert!(cv.primaries.is_some() && cv.transfer.is_some() && cv.matrix.is_some());
                }
            }
        }
    }

    #[test]
    fn an_untagged_buffer_reads_only_its_range() {
        let buffer = surface_buffer(FOURCC_420V);
        assert_eq!(
            read(&buffer, FOURCC_420V),
            Colorimetry {
                range: Some(ColorRange::Limited),
                ..Colorimetry::default()
            }
        );
        let buffer = surface_buffer(FOURCC_420F);
        assert_eq!(
            read(&buffer, FOURCC_420F),
            Colorimetry {
                range: Some(ColorRange::Full),
                ..Colorimetry::default()
            }
        );
    }

    #[test]
    fn corevideo_formats_give_their_range() {
        for (fourcc, range) in [
            (FOURCC_420V, Some(ColorRange::Limited)),
            (FOURCC_YUVS, Some(ColorRange::Limited)),
            (FOURCC_420F, Some(ColorRange::Full)),
            (FOURCC_YUVF, Some(ColorRange::Full)),
            (u32::from_be_bytes(*b"BGRA"), None),
            (u32::from_be_bytes(*b"L008"), None),
        ] {
            assert_eq!(surface_range(fourcc), range, "{:?}", fourcc.to_be_bytes());
        }
    }
}
