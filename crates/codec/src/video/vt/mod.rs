// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! H.264 through Apple VideoToolbox, on macOS and iOS.

pub(crate) mod color;
pub(crate) mod decoder;
pub(crate) mod encoder;
pub(crate) mod ffi;

use edgefirst_tensor::PixelFormat;

use crate::{Capability, Codec, CodecError, Direction, Engine, Transfer};

/// The CoreVideo pixel format of video-range bi-planar 4:2:0 (NV12).
pub(crate) const FOURCC_420V: u32 = u32::from_be_bytes(*b"420v");
/// The CoreVideo pixel format of full-range bi-planar 4:2:0 (NV12).
pub(crate) const FOURCC_420F: u32 = u32::from_be_bytes(*b"420f");
/// The CoreVideo pixel format of video-range packed 4:2:2 ordered Y0 Cb Y1
/// Cr (YUYV).
pub(crate) const FOURCC_YUVS: u32 = u32::from_be_bytes(*b"yuvs");
/// The CoreVideo pixel format of full-range packed 4:2:2 ordered Y0 Cb Y1
/// Cr (YUYV).
pub(crate) const FOURCC_YUVF: u32 = u32::from_be_bytes(*b"yuvf");

/// An H.264 encoder VideoToolbox lists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ListedEncoder {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) hardware: bool,
}

/// The H.264 encoders VideoToolbox lists, in its order.
pub(crate) fn h264_encoders() -> Vec<ListedEncoder> {
    let mut list = std::ptr::null();
    // SAFETY: a valid out-pointer; on success `list` holds a +1 reference.
    let list = match unsafe { ffi::VTCopyVideoEncoderList(std::ptr::null(), &mut list) } {
        0 => unsafe { ffi::CfOwned::from_create(list) },
        _ => None,
    };
    let Some(list) = list else {
        return Vec::new();
    };
    // SAFETY: `list` is a live `CFArray` of `CFDictionary` entries, each
    // read at +0 while `list` holds it.
    unsafe {
        (0..ffi::CFArrayGetCount(list.as_ptr()))
            .filter_map(|i| {
                let entry = ffi::CFArrayGetValueAtIndex(list.as_ptr(), i);
                let value = |key| ffi::CFDictionaryGetValue(entry, key);
                let codec = value(ffi::kVTVideoEncoderList_CodecType);
                let mut fourcc = 0i32;
                if codec.is_null()
                    || ffi::CFNumberGetValue(
                        codec,
                        ffi::kCFNumberSInt32Type,
                        (&mut fourcc as *mut i32).cast(),
                    ) == 0
                    || fourcc as u32 != ffi::kCMVideoCodecType_H264
                {
                    return None;
                }
                let id = ffi::string_of(value(ffi::kVTVideoEncoderList_EncoderID))?;
                let name = ffi::string_of(value(ffi::kVTVideoEncoderList_DisplayName))
                    .unwrap_or_else(|| id.clone());
                Some(ListedEncoder {
                    id,
                    name,
                    hardware: listed_hardware(entry),
                })
            })
            .collect()
    }
}

/// Whether the encoder list entry `entry` is a hardware encoder.
///
/// # Safety
///
/// `entry` is a live encoder list dictionary.
#[cfg(target_os = "macos")]
unsafe fn listed_hardware(entry: ffi::CFDictionaryRef) -> bool {
    // SAFETY: `entry` is a live dictionary and the key a VideoToolbox
    // constant; the value is read at +0.
    unsafe {
        let value =
            ffi::CFDictionaryGetValue(entry, ffi::kVTVideoEncoderList_IsHardwareAccelerated);
        !value.is_null()
            && ffi::CFGetTypeID(value) == ffi::CFBooleanGetTypeID()
            && ffi::CFBooleanGetValue(value) != 0
    }
}

/// Whether the encoder list entry `entry` is a hardware encoder: on iOS, a
/// device encodes in hardware and the simulator does not.
///
/// # Safety
///
/// `entry` is a live encoder list dictionary.
#[cfg(target_os = "ios")]
unsafe fn listed_hardware(_entry: ffi::CFDictionaryRef) -> bool {
    cfg!(not(target_abi = "sim"))
}

/// Adds the VideoToolbox H.264 decoder and encoders to `out`.
pub(crate) fn capabilities(out: &mut Vec<Capability>) {
    // SAFETY: takes a codec type and has no other preconditions.
    let hardware = unsafe { ffi::VTIsHardwareDecodeSupported(ffi::kCMVideoCodecType_H264) } != 0;
    out.push(Capability {
        engine: Engine::VideoToolbox,
        device: "VideoToolbox H.264 decoder".into(),
        codec: Codec::H264,
        direction: Direction::Decode,
        hardware,
        formats: vec![PixelFormat::Nv12],
        transfer: Transfer::ZeroCopy,
    });
    out.extend(h264_encoders().into_iter().map(|e| Capability {
        engine: Engine::VideoToolbox,
        device: e.name,
        codec: Codec::H264,
        direction: Direction::Encode,
        hardware: e.hardware,
        formats: vec![PixelFormat::Nv12, PixelFormat::Yuyv],
        transfer: if e.hardware {
            Transfer::ZeroCopy
        } else {
            Transfer::CpuCopy
        },
    }));
}

/// The offset of a chroma plane at address `chroma` from a luma plane at
/// address `luma`, and the length from the start of the luma plane to the
/// end of the chroma plane, for planes of `pitches` bytes per row and `rows`
/// rows.
///
/// Returns `None` when either address is null, the chroma plane starts
/// before the luma plane ends, or the length exceeds `isize::MAX`.
pub(crate) fn chroma_offset(
    luma: usize,
    chroma: usize,
    pitches: [usize; 2],
    rows: [usize; 2],
) -> Option<(usize, usize)> {
    if luma == 0 || chroma == 0 {
        return None;
    }
    let at = chroma.checked_sub(luma)?;
    if at < pitches[0].checked_mul(rows[0])? {
        return None;
    }
    let len = at.checked_add(pitches[1].checked_mul(rows[1])?)?;
    (len <= isize::MAX as usize).then_some((at, len))
}

/// The error for `op` failing with `status`.
pub(crate) fn status(op: &str, status: i32) -> CodecError {
    CodecError::Device {
        op: op.to_owned(),
        source: std::io::Error::other(format!("OSStatus {status}")),
    }
}

#[cfg(all(test, target_os = "macos"))]
mod tests;
