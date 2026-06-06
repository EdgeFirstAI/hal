// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Portable `PixelFormat` → DRM FourCC mapping.
//!
//! Uses the platform-neutral [`drm_fourcc`] crate (the same `DrmFourcc` enum
//! `gbm` re-exports) so this mapping — and the shader/format code that depends
//! on it — carries no `gbm`/Linux coupling. `gbm` is a buffer *allocator* and
//! is needed only where real DMA-BUF/DRM buffers or GBM EGL displays are
//! created (`gl::context`), not for naming a FourCC.

use crate::Error;
use drm_fourcc::DrmFourcc;
use edgefirst_tensor::PixelFormat;

/// Map a [`PixelFormat`] to its DRM FourCC code.
///
/// The returned [`DrmFourcc`] is a `#[repr(u32)]` industry-standard FourCC
/// constant; at the EGL boundary it is consumed as its raw `u32` value
/// (`EGL_LINUX_DRM_FOURCC`). This is a pure, platform-neutral mapping.
pub(super) fn pixel_format_to_drm(fmt: PixelFormat) -> Result<DrmFourcc, Error> {
    match fmt {
        PixelFormat::Rgba => Ok(DrmFourcc::Abgr8888),
        PixelFormat::Bgra => Ok(DrmFourcc::Argb8888),
        PixelFormat::Yuyv => Ok(DrmFourcc::Yuyv),
        PixelFormat::Vyuy => Ok(DrmFourcc::Vyuy),
        PixelFormat::Rgb => Ok(DrmFourcc::Bgr888),
        PixelFormat::Grey => Ok(DrmFourcc::R8),
        PixelFormat::Nv12 => Ok(DrmFourcc::Nv12),
        PixelFormat::PlanarRgb => Ok(DrmFourcc::R8),
        _ => Err(Error::NotSupported(format!(
            "PixelFormat {fmt:?} has no DRM format mapping"
        ))),
    }
}
