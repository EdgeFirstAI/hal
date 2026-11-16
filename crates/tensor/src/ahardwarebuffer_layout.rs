// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Platform-neutral AHardwareBuffer layout logic: the format-mapping table and
//! the descriptor geometry math used by
//! [`ahardwarebuffer`](crate::ahardwarebuffer). Nothing here touches FFI.

use crate::{
    error::{Error, Result},
    BufferIdentity, DType, IdentityKind, PixelFormat,
};

/// `AHardwareBuffer_Desc` from `<android/hardware_buffer.h>` (40 bytes, no
/// padding). `stride` is filled by `AHardwareBuffer_allocate`/`_describe`, in
/// pixels.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct AHardwareBufferDesc {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) layers: u32,
    pub(crate) format: u32,
    pub(crate) usage: u64,
    pub(crate) stride: u32,
    pub(crate) rfu0: u32,
    pub(crate) rfu1: u64,
}

// AHardwareBuffer_UsageFlags CPU bits: enum values within their masks (NEVER=0,
// RARELY=2, OFTEN=3).
pub(crate) const USAGE_CPU_READ_OFTEN: u64 = 0x3;
pub(crate) const USAGE_CPU_READ_MASK: u64 = 0xF;
pub(crate) const USAGE_CPU_WRITE_OFTEN: u64 = 0x30;
pub(crate) const USAGE_CPU_WRITE_MASK: u64 = 0xF0;

/// Classify a GPU vendor string (the `ro.hardware.egl` system property) into
/// its native tile-compression scheme. Emulation stacks (`angle`, `emulation`,
/// `swiftshader`) and unknown vendors return `None`.
pub(crate) fn scheme_for_egl_vendor(vendor: &str) -> Option<crate::CompressionScheme> {
    use crate::CompressionScheme::*;
    let v = vendor.trim().to_ascii_lowercase();
    if v.is_empty() || v.contains("emulation") || v.contains("angle") || v.contains("swiftshader") {
        return None;
    }
    if v.contains("adreno") {
        Some(Ubwc)
    } else if v.contains("mali") || v.contains("immortalis") {
        Some(Afbc)
    } else if v.contains("powervr") {
        Some(Pvric)
    } else if v.contains("xclipse") {
        Some(Dcc)
    } else {
        None
    }
}

/// Whether `(format, dtype)` is eligible for a vendor tile-compressed
/// allocation: RGBA8888 `u8`/`i8`.
pub(crate) fn compression_eligible(format: PixelFormat, dtype: DType) -> bool {
    matches!(format, PixelFormat::Rgba) && matches!(dtype, DType::U8 | DType::I8)
}

// AHardwareBuffer_Format values (subset used by the HAL).
/// RGBA 8:8:8:8 UNORM — API 26+.
pub(crate) const FORMAT_R8G8B8A8_UNORM: u32 = 1;
/// RGBA 16:16:16:16 half-float — API 26+. The F16 NCHW render target.
pub(crate) const FORMAT_R16G16B16A16_FLOAT: u32 = 0x16;
/// Opaque byte buffer (`width` = length, `height = layers = 1`) — API 26+.
/// The layout NNAPI uses for tensor blobs; the byte-bag allocation format.
pub(crate) const FORMAT_BLOB: u32 = 0x21;

/// Bytes-per-element for a known AHardwareBuffer pixel format, or `None`
/// for formats the HAL cannot CPU-map linearly (multi-planar YUV etc.).
pub(crate) fn format_bpe(format: u32) -> Option<usize> {
    match format {
        FORMAT_R8G8B8A8_UNORM => Some(4),
        FORMAT_R16G16B16A16_FLOAT => Some(8),
        FORMAT_BLOB => Some(1),
        _ => None,
    }
}

/// Byte pitch + allocation size derived from a (post-allocation)
/// descriptor. BLOB buffers are `width` bytes with a single "row"; image
/// formats use the allocator-chosen `stride` (pixels) × bytes-per-element.
/// `None` when the format has no linear CPU layout or the size overflows.
pub(crate) fn desc_layout(desc: &AHardwareBufferDesc) -> Option<(usize, usize)> {
    let bpe = format_bpe(desc.format)?;
    if desc.format == FORMAT_BLOB {
        let len = desc.width as usize;
        return Some((len, len));
    }
    let bytes_per_row = (desc.stride as usize).checked_mul(bpe)?;
    let buf_size = bytes_per_row.checked_mul(desc.height as usize)?;
    Some((bytes_per_row, buf_size))
}

/// AHardwareBuffer format + bytes-per-element mapping for image-formatted
/// buffers, keyed on `(PixelFormat, DType)`.
///
/// Returns `None` for every other combination, including
/// `Grey`/`Nv12`/`Nv16`/`Nv24`, `Bgra` and `Yuyv` u8.
pub(crate) fn image_format_and_bpe(format: PixelFormat, dtype: DType) -> Option<(u32, usize)> {
    match (format, dtype) {
        // I8 shares the U8 layout.
        (PixelFormat::Rgba, DType::U8 | DType::I8) => Some((FORMAT_R8G8B8A8_UNORM, 4)),
        // Packed RGB u8/i8: the `[H, W, 3]` byte stream in an RGBA8888 surface
        // sized `(W*3/4, H)` via `packed_rgb888_layout`.
        (PixelFormat::Rgb, DType::U8 | DType::I8) => Some((FORMAT_R8G8B8A8_UNORM, 4)),
        // F16: RGBA16F, as a packed RGBA image and as the 4-elements-per-pixel
        // packing of planar `[C, H, W]` f16 streams (surface sized via
        // `packed_rgba16f_layout`).
        (PixelFormat::Rgba | PixelFormat::PlanarRgb | PixelFormat::PlanarRgba, DType::F16) => {
            Some((FORMAT_R16G16B16A16_FLOAT, 8))
        }
        _ => None,
    }
}

/// Byte footprint of `shape` for element type `T`, with overflow-checked
/// arithmetic; an overflow is an error.
pub(crate) fn checked_shape_bytes<T>(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .and_then(|n| n.checked_mul(std::mem::size_of::<T>()))
        .ok_or_else(|| {
            Error::InvalidShape(format!(
                "shape footprint overflows usize (shape={shape:?}, sizeof T={})",
                std::mem::size_of::<T>()
            ))
        })
}

/// Choose an AHardwareBuffer's identity from the parts the FFI supplies: the
/// system id when there is one, the buffer pointer otherwise.
pub(crate) fn identity_from_parts(id: Option<u64>, ptr: u64) -> BufferIdentity {
    match id {
        // API 31+: the system's id for the allocation.
        Some(id) => BufferIdentity::derived(IdentityKind::AHardwareBufferId, id),
        // Before API 31: the buffer pointer.
        None => BufferIdentity::derived(IdentityKind::AHardwareBufferPtr, ptr),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pre_api_31_buffers_do_not_share_an_identity() {
        assert_ne!(
            identity_from_parts(None, 0x1000).id(),
            identity_from_parts(None, 0x2000).id(),
            "pre-31 buffers must be distinguished by their pointer"
        );
    }

    #[test]
    fn the_system_id_is_preferred_over_the_pointer_when_present() {
        let a = identity_from_parts(Some(7), 0xDEAD);
        assert_eq!(a.kind(), IdentityKind::AHardwareBufferId);
        assert_ne!(a.id(), identity_from_parts(Some(8), 0xDEAD).id());
    }

    fn desc(width: u32, height: u32, format: u32, stride: u32) -> AHardwareBufferDesc {
        AHardwareBufferDesc {
            width,
            height,
            layers: 1,
            format,
            usage: 0,
            stride,
            rfu0: 0,
            rfu1: 0,
        }
    }

    #[test]
    fn egl_vendor_classifier_table() {
        use crate::CompressionScheme::*;
        assert_eq!(scheme_for_egl_vendor("adreno"), Some(Ubwc));
        assert_eq!(scheme_for_egl_vendor("Adreno735"), Some(Ubwc));
        assert_eq!(scheme_for_egl_vendor("mali"), Some(Afbc));
        assert_eq!(scheme_for_egl_vendor("immortalis-g925"), Some(Afbc));
        assert_eq!(scheme_for_egl_vendor("powervr"), Some(Pvric));
        assert_eq!(scheme_for_egl_vendor("xclipse"), Some(Dcc));
        assert_eq!(scheme_for_egl_vendor("emulation"), None);
        assert_eq!(scheme_for_egl_vendor("angle"), None);
        assert_eq!(scheme_for_egl_vendor("swiftshader"), None);
        assert_eq!(scheme_for_egl_vendor(""), None);
        assert_eq!(scheme_for_egl_vendor("llvmpipe"), None);
    }

    #[test]
    fn egl_vendor_emulation_marker_wins_over_a_vendor_name() {
        assert_eq!(scheme_for_egl_vendor("emulation-mali"), None);
        assert_eq!(scheme_for_egl_vendor("angle-adreno"), None);
        assert_eq!(scheme_for_egl_vendor("swiftshader_mali"), None);
        assert_eq!(scheme_for_egl_vendor("Emulation-PowerVR"), None);
        assert_eq!(
            scheme_for_egl_vendor("  Mali  "),
            Some(crate::CompressionScheme::Afbc)
        );
        assert_eq!(scheme_for_egl_vendor("   "), None);
    }

    #[test]
    fn compression_eligibility_table() {
        assert!(compression_eligible(PixelFormat::Rgba, DType::U8));
        assert!(compression_eligible(PixelFormat::Rgba, DType::I8));
        assert!(!compression_eligible(PixelFormat::Rgba, DType::F16));
        assert!(!compression_eligible(PixelFormat::Rgb, DType::U8));
        assert!(!compression_eligible(PixelFormat::Nv12, DType::U8));
        assert!(!compression_eligible(PixelFormat::PlanarRgb, DType::U8));
    }

    #[test]
    fn desc_matches_ndk_layout() {
        assert_eq!(std::mem::size_of::<AHardwareBufferDesc>(), 40);
        assert_eq!(std::mem::align_of::<AHardwareBufferDesc>(), 8);
    }

    #[test]
    fn format_table_is_stable() {
        assert_eq!(
            image_format_and_bpe(PixelFormat::Rgba, DType::U8),
            Some((1, 4))
        );
        assert_eq!(
            image_format_and_bpe(PixelFormat::Rgb, DType::U8),
            Some((1, 4))
        );
        assert_eq!(
            image_format_and_bpe(PixelFormat::Rgba, DType::F16),
            Some((0x16, 8))
        );
        assert_eq!(
            image_format_and_bpe(PixelFormat::PlanarRgb, DType::F16),
            Some((0x16, 8))
        );
        assert_eq!(
            image_format_and_bpe(PixelFormat::PlanarRgba, DType::F16),
            Some((0x16, 8))
        );
        assert_eq!(image_format_and_bpe(PixelFormat::Grey, DType::U8), None);
        assert_eq!(image_format_and_bpe(PixelFormat::Nv12, DType::U8), None);
        assert_eq!(image_format_and_bpe(PixelFormat::Bgra, DType::U8), None);
        assert_eq!(image_format_and_bpe(PixelFormat::Yuyv, DType::U8), None);
        assert_eq!(
            image_format_and_bpe(PixelFormat::PlanarRgb, DType::U8),
            None
        );
        assert_eq!(image_format_and_bpe(PixelFormat::Rgba, DType::F32), None);
    }

    #[test]
    fn blob_layout_is_single_row() {
        // BLOB: width is the byte length and the whole allocation is one row.
        let (row, size) = desc_layout(&desc(4096, 1, FORMAT_BLOB, 0)).unwrap();
        assert_eq!((row, size), (4096, 4096));
    }

    #[test]
    fn image_layout_uses_allocator_stride() {
        // 640×480 RGBA8 with a 704-pixel stride.
        let (row, size) = desc_layout(&desc(640, 480, FORMAT_R8G8B8A8_UNORM, 704)).unwrap();
        assert_eq!(row, 704 * 4);
        assert_eq!(size, 704 * 4 * 480);
        // RGBA16F: 8 bytes per texel.
        let (row, size) = desc_layout(&desc(160, 1920, FORMAT_R16G16B16A16_FLOAT, 160)).unwrap();
        assert_eq!(row, 160 * 8);
        assert_eq!(size, 160 * 8 * 1920);
    }

    #[test]
    fn desc_layout_rejects_unknown_and_overflow() {
        // Unknown format (e.g. a YUV camera format) has no linear layout.
        assert!(desc_layout(&desc(64, 64, 0x23, 64)).is_none());
        assert!(desc_layout(&desc(
            u32::MAX,
            u32::MAX,
            FORMAT_R16G16B16A16_FLOAT,
            u32::MAX
        ))
        .is_none());
    }

    #[test]
    fn checked_shape_bytes_accepts_and_rejects() {
        assert_eq!(
            checked_shape_bytes::<u8>(&[480, 640, 4]).unwrap(),
            1_228_800
        );
        assert_eq!(checked_shape_bytes::<u16>(&[2, 3]).unwrap(), 12);
        // Empty product = 1 element (scalar convention).
        assert_eq!(checked_shape_bytes::<u8>(&[]).unwrap(), 1);
        assert!(checked_shape_bytes::<u8>(&[usize::MAX, 2]).is_err());
        assert!(checked_shape_bytes::<u64>(&[usize::MAX / 4]).is_err());
    }
}
