// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Platform-neutral AHardwareBuffer layout logic — the pure half of
//! [`ahardwarebuffer`](crate::ahardwarebuffer), split out so it compiles
//! and unit-tests on every host.
//!
//! No CI lane can *execute* `cfg(target_os = "android")` code (the
//! Android lane is compile + link-validation only), so anything living in
//! the FFI module is invisible to regression tests until the Device Farm
//! coverage lane exists. The format-mapping table and the descriptor
//! geometry/overflow math are exactly the logic that must not drift — the
//! macOS analog of the table caused a silent R↔B swap during bring-up
//! (see `iosurface.rs`) — so they live here, cfg-free, with host tests.
//! The Android module re-exports them; nothing here touches FFI.

use crate::{
    error::{Error, Result},
    DType, PixelFormat,
};

/// `AHardwareBuffer_Desc` from `<android/hardware_buffer.h>`. Layout must
/// match the NDK header exactly (40 bytes, no padding); `stride` is
/// filled by `AHardwareBuffer_allocate`/`_describe` (in pixels, not
/// bytes). Defined here so the geometry math below is host-testable; the
/// Android module uses it in the FFI signatures.
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
/// buffers, keyed on `(PixelFormat, DType)`. **This function is the single
/// source of truth for the mapping** — the tensor allocation side and the
/// image crate's Android GL import both read it (via
/// `image_ahardwarebuffer_layout`); keep the two layers in sync by not
/// duplicating this table (the same rule that prevents the macOS R↔B
/// drift documented in `iosurface.rs`).
///
/// Combinations not listed have no zero-copy path on Android today and
/// fall back to SHM/Mem + CPU conversion:
///
/// * `Grey`/`Nv12`/`Nv16`/`Nv24` u8 — the single-channel
///   `AHARDWAREBUFFER_FORMAT_R8_UNORM` requires API 29; the HAL floor is
///   26. Zero-copy Grey/NV on 29+ is a planned follow-up together with
///   the external-OES YUV sampling path.
/// * `Bgra` u8 — AHardwareBuffer has no BGRA format; mapping it to RGBA
///   would silently swap R↔B (the exact macOS footgun).
/// * `Yuyv` u8 — no packed-4:2:2 AHardwareBuffer format exists.
pub(crate) fn image_format_and_bpe(format: PixelFormat, dtype: DType) -> Option<(u32, usize)> {
    match (format, dtype) {
        (PixelFormat::Rgba, DType::U8) => Some((FORMAT_R8G8B8A8_UNORM, 4)),
        // The F16 zero-copy path: RGBA16F, both as a packed RGBA image and
        // as the 4-elements-per-pixel packing of planar `[C, H, W]` f16
        // streams (surface sized via `packed_rgba16f_layout`).
        (PixelFormat::Rgba | PixelFormat::PlanarRgb | PixelFormat::PlanarRgba, DType::F16) => {
            Some((FORMAT_R16G16B16A16_FLOAT, 8))
        }
        _ => None,
    }
}

/// Byte footprint of `shape` for element type `T`, with overflow-checked
/// arithmetic — a shape whose element product (or its byte size) wraps
/// `usize` must be rejected, not allowed to slip past a capacity check and
/// produce an out-of-bounds map later.
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

#[cfg(test)]
mod tests {
    use super::*;

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
    fn desc_matches_ndk_layout() {
        // `AHardwareBuffer_Desc` is 40 bytes in the NDK header; a size or
        // alignment drift here silently corrupts every allocate/describe.
        assert_eq!(std::mem::size_of::<AHardwareBufferDesc>(), 40);
        assert_eq!(std::mem::align_of::<AHardwareBufferDesc>(), 8);
    }

    #[test]
    fn format_table_is_stable() {
        // The (PixelFormat, DType) → AHardwareBuffer format table — the
        // R↔B/format-drift guard. Values from <android/hardware_buffer.h>.
        assert_eq!(
            image_format_and_bpe(PixelFormat::Rgba, DType::U8),
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
        // No-mapping cases: R8 needs API 29, BGRA/YUYV have no AHB format,
        // planar u8 has no packing.
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
        // BLOB: width = byte length, stride is meaningless (gralloc may
        // report anything); the whole allocation is one "row".
        let (row, size) = desc_layout(&desc(4096, 1, FORMAT_BLOB, 0)).unwrap();
        assert_eq!((row, size), (4096, 4096));
    }

    #[test]
    fn image_layout_uses_allocator_stride() {
        // 640×480 RGBA8 with a gralloc-padded 704-pixel stride: the row
        // pitch and allocation must follow the stride, never width×bpe.
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
        // stride×bpe×height overflowing usize must yield None, not wrap.
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
        // Element-product overflow and byte-size overflow both reject.
        assert!(checked_shape_bytes::<u8>(&[usize::MAX, 2]).is_err());
        assert!(checked_shape_bytes::<u64>(&[usize::MAX / 4]).is_err());
    }
}
