// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `(PixelFormat, DType) → (FourCC, bytes-per-element)` for IOSurface.
//!
//! Split out of [`crate::iosurface`] so the table compiles under the
//! `dynamic` backend. `iosurface.rs` is `static`-only (it owns
//! `IoSurfaceTensor` storage); the image crate's macOS GL import still
//! needs this mapping when it talks to IOSurface-backed handles minted
//! by `libedgefirst_tensor`.

#![cfg(any(target_os = "macos", target_os = "ios"))]

use crate::{DType, PixelFormat};

/// IOSurface FourCC + bytes-per-element mapping for image-formatted
/// IOSurfaces, keyed on `(PixelFormat, DType)`. The GL backend's
/// `EGL_ANGLE_iosurface_client_buffer` import requires the IOSurface
/// pixel format to match the GL internal format / type combination —
/// ANGLE validates `IOSurfaceGetBytesPerElement` against the requested
/// `EGL_TEXTURE_INTERNAL_FORMAT_ANGLE` and rejects mismatches with
/// `EGL_BAD_ATTRIBUTE`. **This function is the single source of truth
/// for the `(PixelFormat, DType) → (FourCC, bpe)` mapping** — the image
/// crate's macOS GL backend reads it via [`image_iosurface_layout`]
/// when constructing the EGL pbuffer attribute list. Keep the two
/// layers in sync by not duplicating this table.
///
/// FourCC codes follow Apple's CoreVideo `kCVPixelFormatType_*`
/// constants because ANGLE's Metal backend recognizes those for
/// `EGL_TEXTURE_INTERNAL_FORMAT_ANGLE` mapping.
///
/// **ANGLE float-format constraint** (verified against
/// `EGL_ANGLE_iosurface_client_buffer.txt`): the extension's accepted
/// `(type, internal_format)` allowlist contains exactly **one** float
/// entry — `GL_HALF_FLOAT + GL_RGBA` (RGBA16F). There is no
/// `GL_FLOAT` entry, no single-channel float, no RGBA32F. R32F and
/// R16F single-channel bindings produce `EGL_BAD_ATTRIBUTE` at
/// `eglCreatePbufferFromClientBuffer` time even though the
/// extension-presence query (`GL_EXT_color_buffer_float` /
/// `_half_float`) reports them as available. Until the spec changes
/// our only viable float path is RGBA16F + 4-element pixel packing.
///
/// Combinations not listed are not supported by the GL backend on
/// macOS; callers fall back to SHM/Mem and a CPU code path.
///
/// Returns `None` when the (format, dtype) pair does not have a
/// defined IOSurface FourCC mapping in HAL (NV12, U8 planar, etc).
pub fn image_iosurface_layout(format: PixelFormat, dtype: DType) -> Option<(u32, usize)> {
    match (format, dtype) {
        // YUYV is 4:2:2 packed (2 bytes/pixel); sampled as GL_RG via
        // FourCC '2C08' (kCVPixelFormatType_TwoComponent8).
        (PixelFormat::Yuyv, DType::U8) => Some((u32::from_be_bytes(*b"2C08"), 2)),
        // The FourCC matches the in-memory byte order: 'RGBA' for Rgba
        // tensors, 'BGRA' for Bgra. ANGLE supports both via
        // `EGL_TEXTURE_INTERNAL_FORMAT_ANGLE = GL_RGBA` / `GL_BGRA_EXT`
        // and produces the matching shader output. Mapping both to
        // 'BGRA' would put the IOSurface bytes in BGRA order, which is
        // wrong for the Rgba contract.
        // I8 shares the U8 layout for the packed RGB/RGBA arms: INT8 is a
        // per-byte `^0x80` bias applied in the shader, not a format change.
        (PixelFormat::Rgba, DType::U8 | DType::I8) => Some((u32::from_be_bytes(*b"RGBA"), 4)),
        (PixelFormat::Bgra, DType::U8) => Some((u32::from_be_bytes(*b"BGRA"), 4)),
        // Packed RGB u8/i8 (the INT8 NPU input layout): no 3-channel
        // IOSurface format exists, so the tight `[H, W, 3]` byte stream
        // lives in an RGBA8888 surface sized `(W*3/4, H)` via
        // `packed_rgb888_layout` — the same texel-packing trick as the
        // planar-F16 arm below and the identical representation the
        // Android AHardwareBuffer side uses. The GL engine's two-pass
        // packed-RGB shader renders into it zero-copy (the pbuffer bind
        // carries explicit geometry, so the FourCC only fixes the byte
        // layout). Historically this combination fell through to a
        // generic 'L008' byte-bag that happened to bind the same way;
        // this is the designed replacement.
        (PixelFormat::Rgb, DType::U8 | DType::I8) => Some((u32::from_be_bytes(*b"RGBA"), 4)),
        // Single-channel 8-bit (`L008` = kCVPixelFormatType_OneComponent8),
        // sampled as `GL_RED`. Used for GREY images and as the raw byte plane
        // for the semi-planar YUV formats (NV12/NV16/NV24): the GPU binds the
        // whole contiguous `[total_h, W]` buffer as one R8 texture and the
        // YUV→RGB shader computes the luma/chroma texel positions itself
        // (portable across ANGLE/Metal, Mali/EGL, and embedded GLES).
        (
            PixelFormat::Grey | PixelFormat::Nv12 | PixelFormat::Nv16 | PixelFormat::Nv24,
            DType::U8,
        ) => Some((u32::from_be_bytes(*b"L008"), 1)),
        // ── F16 IOSurface for zero-copy preprocessing (CoreML / ANE) ──
        // The only ANGLE-supported float (type, internal_format) pair
        // is `(GL_HALF_FLOAT, GL_RGBA)` = RGBA16F, FourCC 'RGhA'
        // (kCVPixelFormatType_64RGBAHalf), 8 bytes per pixel.
        //
        // For Rgba destinations: 1 RGBA16F pixel = 1 image pixel of 4
        // half-floats.
        //
        // For PlanarRgb / PlanarRgba destinations: we pack 4 contiguous
        // half-floats of the planar `[C, H, W]` byte stream into each
        // RGBA16F pixel. The IOSurface is then sized `(W/4, C*H)` —
        // see `new_image` for the geometry. The byte layout is
        // identical to a (nonexistent) R16F `(W, C*H)` surface so ORT
        // can consume the locked base address as `&[f16]` with shape
        // `[1, C, H, W]` without any rearrangement.
        (PixelFormat::Rgba | PixelFormat::PlanarRgb | PixelFormat::PlanarRgba, DType::F16) => {
            Some((u32::from_be_bytes(*b"RGhA"), 8))
        }
        _ => None,
    }
}
