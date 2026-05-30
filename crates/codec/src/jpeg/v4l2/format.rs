// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Convert a hardware CAPTURE row into a packed-u8 destination row, reusing
//! the existing planar colour kernels in [`crate::jpeg::color`].
//!
//! The driver chooses the raw CAPTURE format from the JPEG's subsampling, so
//! we dispatch on the FourCC reported by `G_FMT`. Supported layouts are packed
//! 4:4:4 (`YUV3`), `GREY`, and 4:2:0 `NV12`/`NV12M` (the common output for
//! colour JPEGs on this hardware). Anything else is reported unsupported and
//! the caller falls back to the CPU decoder rather than emitting wrong pixels.

use super::ioctl;
use crate::jpeg::color::{self, ColorConvertFn, GreyCopyFn};
use edgefirst_tensor::PixelFormat;

/// A CAPTURE pixel layout this backend can convert.
pub(crate) enum CapKind {
    /// Packed Y/Cb/Cr 4:4:4, 3 bytes/pixel (`V4L2_PIX_FMT_YUV24`). One plane.
    Yuv444Packed,
    /// 8-bit luma only (`V4L2_PIX_FMT_GREY`). One plane.
    Grey,
    /// Y plane + interleaved Cb/Cr at 4:2:0 (`V4L2_PIX_FMT_NV12`/`NV12M`). Two
    /// logical planes (one buffer for NV12, two for NV12M).
    Nv12,
}

/// Classify a CAPTURE FourCC, or `None` if we cannot convert it.
pub(crate) fn classify(fourcc: u32) -> Option<CapKind> {
    match fourcc {
        ioctl::V4L2_PIX_FMT_YUV24 => Some(CapKind::Yuv444Packed),
        ioctl::V4L2_PIX_FMT_GREY => Some(CapKind::Grey),
        ioctl::V4L2_PIX_FMT_NV12 | ioctl::V4L2_PIX_FMT_NV12M => Some(CapKind::Nv12),
        _ => None,
    }
}

/// How to write the destination pixels for the requested output format.
pub(crate) enum OutPlan {
    /// YCbCr → packed colour (RGB/RGBA/BGRA).
    Color(ColorConvertFn),
    /// Luma → greyscale.
    Grey(GreyCopyFn),
}

/// Select the destination plan for `fmt`, or `None` if unsupported on the
/// hardware path (e.g. NV12 output, which the zero-copy path handles).
pub(crate) fn plan_output(fmt: PixelFormat) -> Option<OutPlan> {
    match fmt {
        PixelFormat::Rgb | PixelFormat::Rgba | PixelFormat::Bgra => {
            color::select_color_convert(fmt).map(OutPlan::Color)
        }
        PixelFormat::Grey => Some(OutPlan::Grey(color::select_grey_copy())),
        _ => None,
    }
}

/// Reusable per-row Y/Cb/Cr scratch (full-width, after any chroma upsample).
pub(crate) struct RowScratch {
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
}

impl RowScratch {
    pub fn new() -> Self {
        Self {
            y: Vec::new(),
            cb: Vec::new(),
            cr: Vec::new(),
        }
    }

    fn ensure(&mut self, width: usize) {
        if self.y.len() < width {
            self.y.resize(width, 0);
            self.cb.resize(width, 0);
            self.cr.resize(width, 0);
        }
    }
}

/// Convert one CAPTURE row into `out` (packed destination pixels for `width`
/// pixels).
///
/// - `luma`: the row's primary plane — packed YCbCr for `Yuv444Packed`, the Y
///   plane for `Grey`/`Nv12`.
/// - `chroma`: the interleaved Cb/Cr row for `Nv12` (the chroma row for this
///   output line, i.e. source row `y / 2`); `None` for the single-plane kinds.
///
/// For greyscale sources feeding a colour output, neutral chroma (128) is
/// supplied so the BT.601 kernel yields `R=G=B=Y`; for any source feeding a
/// grey output, only the Y component is taken.
pub(crate) fn convert_row(
    kind: &CapKind,
    plan: &OutPlan,
    luma: &[u8],
    chroma: Option<&[u8]>,
    width: usize,
    scratch: &mut RowScratch,
    out: &mut [u8],
) {
    // Fast path: any luma-only conversion to grey output — no de-interleave.
    if let OutPlan::Grey(g) = plan {
        match kind {
            CapKind::Grey | CapKind::Nv12 => return g(&luma[..width], out, width),
            CapKind::Yuv444Packed => {
                scratch.ensure(width);
                for (x, y) in scratch.y[..width].iter_mut().enumerate() {
                    *y = luma[3 * x];
                }
                return g(&scratch.y[..width], out, width);
            }
        }
    }

    scratch.ensure(width);
    match kind {
        CapKind::Yuv444Packed => {
            for x in 0..width {
                scratch.y[x] = luma[3 * x];
                scratch.cb[x] = luma[3 * x + 1];
                scratch.cr[x] = luma[3 * x + 2];
            }
        }
        CapKind::Grey => {
            scratch.y[..width].copy_from_slice(&luma[..width]);
            for x in 0..width {
                scratch.cb[x] = 128;
                scratch.cr[x] = 128;
            }
        }
        CapKind::Nv12 => {
            // 4:2:0: one interleaved Cb/Cr pair per 2×2 luma block. Nearest
            // upsample horizontally (the chroma row is already chosen as y/2).
            // The index addresses three planes at different strides (luma at x,
            // chroma at x/2), so a range loop is clearer than a zip.
            let cbcr = chroma.unwrap_or(&[]);
            #[allow(clippy::needless_range_loop)]
            for x in 0..width {
                scratch.y[x] = luma[x];
                let c = (x / 2) * 2;
                scratch.cb[x] = cbcr[c];
                scratch.cr[x] = cbcr[c + 1];
            }
        }
    }

    if let OutPlan::Color(f) = plan {
        f(
            &scratch.y[..width],
            &scratch.cb[..width],
            &scratch.cr[..width],
            out,
            width,
        );
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
        // Formats not yet handled on the hardware path fall back to CPU.
        assert!(classify(ioctl::V4L2_PIX_FMT_YUYV).is_none());
    }

    #[test]
    fn grey_to_grey_is_passthrough() {
        let mut sc = RowScratch::new();
        let src = [10u8, 20, 30, 40];
        let mut out = [0u8; 4];
        let plan = plan_output(PixelFormat::Grey).unwrap();
        convert_row(&CapKind::Grey, &plan, &src, None, 4, &mut sc, &mut out);
        assert_eq!(out, [10, 20, 30, 40]);
    }

    #[test]
    fn grey_to_rgb_uses_neutral_chroma() {
        // Neutral chroma (128) must yield R==G==B==Y through the BT.601 kernel.
        let mut sc = RowScratch::new();
        let src = [200u8];
        let mut out = [0u8; 3];
        let plan = plan_output(PixelFormat::Rgb).unwrap();
        convert_row(&CapKind::Grey, &plan, &src, None, 1, &mut sc, &mut out);
        for c in out {
            assert!((c as i32 - 200).abs() <= 2, "expected ~200, got {out:?}");
        }
    }

    #[test]
    fn yuv444_deinterleaves_then_converts() {
        // A grey pixel encoded as YUV (Y=128, Cb=Cr=128) → ~128 in all channels.
        let mut sc = RowScratch::new();
        let src = [128u8, 128, 128];
        let mut out = [0u8; 3];
        let plan = plan_output(PixelFormat::Rgb).unwrap();
        convert_row(
            &CapKind::Yuv444Packed,
            &plan,
            &src,
            None,
            1,
            &mut sc,
            &mut out,
        );
        for c in out {
            assert!((c as i32 - 128).abs() <= 2, "expected ~128, got {out:?}");
        }
    }

    #[test]
    fn nv12_neutral_chroma_to_rgb_is_grey() {
        // Two luma pixels sharing one neutral chroma pair → R==G==B==Y.
        let mut sc = RowScratch::new();
        let luma = [90u8, 160];
        let chroma = [128u8, 128];
        let mut out = [0u8; 6];
        let plan = plan_output(PixelFormat::Rgb).unwrap();
        convert_row(
            &CapKind::Nv12,
            &plan,
            &luma,
            Some(&chroma),
            2,
            &mut sc,
            &mut out,
        );
        assert!((out[0] as i32 - 90).abs() <= 2 && (out[3] as i32 - 160).abs() <= 2);
    }

    #[test]
    fn nv12_to_grey_takes_luma() {
        let mut sc = RowScratch::new();
        let luma = [11u8, 22, 33, 44];
        let chroma = [128u8, 128, 128, 128];
        let mut out = [0u8; 4];
        let plan = plan_output(PixelFormat::Grey).unwrap();
        convert_row(
            &CapKind::Nv12,
            &plan,
            &luma,
            Some(&chroma),
            4,
            &mut sc,
            &mut out,
        );
        assert_eq!(out, [11, 22, 33, 44]);
    }
}
