// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `PixelFormat` → DRM FourCC mapping through the [`drm_fourcc`] crate.

use crate::Error;
use drm_fourcc::DrmFourcc;
use edgefirst_tensor::PixelFormat;

/// Map a [`PixelFormat`] to its DRM FourCC code. At the EGL boundary the
/// [`DrmFourcc`] is consumed as its raw `u32` value (`EGL_LINUX_DRM_FOURCC`).
pub(super) fn pixel_format_to_drm(fmt: PixelFormat) -> Result<DrmFourcc, Error> {
    match fmt {
        PixelFormat::Rgba => Ok(DrmFourcc::Abgr8888),
        PixelFormat::Bgra => Ok(DrmFourcc::Argb8888),
        PixelFormat::Yuyv => Ok(DrmFourcc::Yuyv),
        PixelFormat::Vyuy => Ok(DrmFourcc::Vyuy),
        PixelFormat::Rgb => Ok(DrmFourcc::Bgr888),
        PixelFormat::Grey => Ok(DrmFourcc::R8),
        PixelFormat::Nv12 => Ok(DrmFourcc::Nv12),
        PixelFormat::PlanarRgb | PixelFormat::PlanarRgba => Ok(DrmFourcc::R8),
        _ => Err(Error::NotSupported(format!(
            "PixelFormat {fmt:?} has no DRM format mapping"
        ))),
    }
}
