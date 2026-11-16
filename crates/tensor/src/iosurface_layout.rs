// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `(PixelFormat, DType) → (FourCC, bytes-per-element)` for IOSurface.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use crate::{DType, PixelFormat};

/// IOSurface FourCC + bytes-per-element mapping for image-formatted IOSurfaces,
/// keyed on `(PixelFormat, DType)`. The image crate's macOS GL backend reads it
/// when constructing the EGL pbuffer attribute list.
///
/// FourCC codes follow Apple's CoreVideo `kCVPixelFormatType_*` constants. The
/// only float entry is RGBA16F.
///
/// Returns `None` when the (format, dtype) pair has no IOSurface FourCC mapping
/// (planar u8, any f32, etc.).
pub fn image_iosurface_layout(format: PixelFormat, dtype: DType) -> Option<(u32, usize)> {
    match (format, dtype) {
        // YUYV is 4:2:2 packed (2 bytes/pixel): FourCC '2C08'
        // (kCVPixelFormatType_TwoComponent8), sampled as GL_RG.
        (PixelFormat::Yuyv, DType::U8) => Some((u32::from_be_bytes(*b"2C08"), 2)),
        // The FourCC matches the in-memory byte order: 'RGBA' for Rgba tensors,
        // 'BGRA' for Bgra. I8 shares the U8 layout.
        (PixelFormat::Rgba, DType::U8 | DType::I8) => Some((u32::from_be_bytes(*b"RGBA"), 4)),
        (PixelFormat::Bgra, DType::U8) => Some((u32::from_be_bytes(*b"BGRA"), 4)),
        // Packed RGB u8/i8: the `[H, W, 3]` byte stream in an RGBA8888 surface
        // sized `(W*3/4, H)` via `packed_rgb888_layout`.
        (PixelFormat::Rgb, DType::U8 | DType::I8) => Some((u32::from_be_bytes(*b"RGBA"), 4)),
        // Single-channel 8-bit (`L008` = kCVPixelFormatType_OneComponent8),
        // sampled as `GL_RED`: GREY images, and the whole contiguous `[total_h,
        // W]` buffer of the semi-planar YUV formats (NV12/NV16/NV24).
        (
            PixelFormat::Grey | PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24,
            DType::U8,
        ) => Some((u32::from_be_bytes(*b"L008"), 1)),
        // F16: RGBA16F, FourCC 'RGhA' (kCVPixelFormatType_64RGBAHalf), 8 bytes
        // per pixel. For Rgba, one RGBA16F pixel is one image pixel. For
        // PlanarRgb / PlanarRgba, each RGBA16F pixel holds 4 contiguous
        // half-floats of the planar `[C, H, W]` byte stream and the IOSurface
        // is sized `(W/4, C*H)`.
        (PixelFormat::Rgba | PixelFormat::PlanarRgb | PixelFormat::PlanarRgba, DType::F16) => {
            Some((u32::from_be_bytes(*b"RGhA"), 8))
        }
        _ => None,
    }
}
