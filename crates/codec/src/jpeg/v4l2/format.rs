// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Classify the driver-chosen CAPTURE pixel format.
//!
//! The codec emits native `NV12`/`GREY`; the hardware decode path copies the
//! decoded planes straight into the tensor (no colour conversion). 4:2:0 JPEGs
//! decode to `NV12`/`NV12M` (Y + CbCr planes copied through) and greyscale to
//! `GREY` (Y copied through). 4:4:4 (`YUV3`) capture is reported but the native
//! NV12 write for it falls back to the CPU decoder.

use super::ioctl;

/// A CAPTURE pixel layout the hardware may produce.
pub(crate) enum CapKind {
    /// Packed Y/Cb/Cr 4:4:4 (`V4L2_PIX_FMT_YUV24`). One plane.
    Yuv444Packed,
    /// 8-bit luma only (`V4L2_PIX_FMT_GREY`). One plane.
    Grey,
    /// Y plane + interleaved Cb/Cr at 4:2:0 (`V4L2_PIX_FMT_NV12`/`NV12M`).
    Nv12,
}

/// Classify a CAPTURE FourCC, or `None` if we cannot consume it.
pub(crate) fn classify(fourcc: u32) -> Option<CapKind> {
    match fourcc {
        ioctl::V4L2_PIX_FMT_YUV24 => Some(CapKind::Yuv444Packed),
        ioctl::V4L2_PIX_FMT_GREY => Some(CapKind::Grey),
        ioctl::V4L2_PIX_FMT_NV12 | ioctl::V4L2_PIX_FMT_NV12M => Some(CapKind::Nv12),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_maps_known_capture_formats() {
        assert!(matches!(
            classify(ioctl::V4L2_PIX_FMT_YUV24),
            Some(CapKind::Yuv444Packed)
        ));
        assert!(matches!(
            classify(ioctl::V4L2_PIX_FMT_GREY),
            Some(CapKind::Grey)
        ));
        assert!(matches!(
            classify(ioctl::V4L2_PIX_FMT_NV12),
            Some(CapKind::Nv12)
        ));
        assert!(matches!(
            classify(ioctl::V4L2_PIX_FMT_NV12M),
            Some(CapKind::Nv12)
        ));
        assert!(classify(ioctl::V4L2_PIX_FMT_YUYV).is_none());
    }
}
