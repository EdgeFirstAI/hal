// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Resolve a tensor's colorimetry to concrete conversion parameters.
//!
//! `convert()` calls [`effective_colorimetry`] at use-time to fill only the
//! axes it needs, per industry convention, WITHOUT mutating the tensor.

use edgefirst_tensor::{
    ColorEncoding, ColorRange, ColorSpace, ColorTransfer, Colorimetry, TensorDyn,
};

/// Height (in lines) at/above which HD conventions (BT.709) apply; below it,
/// SD conventions (BT.601). Matches Rec.709(HD)/Rec.601(SD) + libcamera.
const HD_THRESHOLD: usize = 720;

/// Return a fully-resolved `Colorimetry` (every axis `Some`): the tensor's
/// specified axes, with only the missing ones filled by the height heuristic.
/// Pure — does NOT mutate the tensor.
pub(crate) fn effective_colorimetry(t: &TensorDyn) -> Colorimetry {
    let hd = t.height().map(|h| h >= HD_THRESHOLD).unwrap_or(true);
    let src = t.colorimetry().unwrap_or_default();
    Colorimetry {
        space: src.space.or(Some(if hd {
            ColorSpace::Bt709
        } else {
            ColorSpace::Smpte170m
        })),
        transfer: src.transfer.or(Some(ColorTransfer::Bt709)),
        encoding: src.encoding.or(Some(if hd {
            ColorEncoding::Bt709
        } else {
            ColorEncoding::Bt601
        })),
        range: src.range.or(Some(ColorRange::Limited)),
    }
}

/// Map a resolved `ColorEncoding` to the `yuv` crate matrix.
pub(crate) fn yuv_matrix(e: ColorEncoding) -> yuv::YuvStandardMatrix {
    match e {
        ColorEncoding::Bt601 => yuv::YuvStandardMatrix::Bt601,
        ColorEncoding::Bt709 => yuv::YuvStandardMatrix::Bt709,
        ColorEncoding::Bt2020 => yuv::YuvStandardMatrix::Bt2020,
        // Future encodings added in a later HAL version default to BT.709.
        _ => yuv::YuvStandardMatrix::Bt709,
    }
}

/// Map a resolved `ColorRange` to the `yuv` crate range.
pub(crate) fn yuv_range(r: ColorRange) -> yuv::YuvRange {
    match r {
        ColorRange::Full => yuv::YuvRange::Full,
        ColorRange::Limited => yuv::YuvRange::Limited,
        // Future range variants default to Limited (broadcast convention).
        _ => yuv::YuvRange::Limited,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use edgefirst_tensor::{ColorEncoding, ColorRange, Colorimetry, DType, PixelFormat, TensorDyn};

    #[test]
    fn heuristic_hd_is_709_limited_sd_is_601_limited() {
        let hd = TensorDyn::image(1280, 720, PixelFormat::Nv12, DType::U8, None).unwrap();
        let r = effective_colorimetry(&hd);
        assert_eq!(r.encoding, Some(ColorEncoding::Bt709));
        assert_eq!(r.range, Some(ColorRange::Limited));
        let sd = TensorDyn::image(640, 480, PixelFormat::Nv12, DType::U8, None).unwrap();
        let r = effective_colorimetry(&sd);
        assert_eq!(r.encoding, Some(ColorEncoding::Bt601));
        assert_eq!(r.range, Some(ColorRange::Limited));
    }

    #[test]
    fn specified_axes_win_only_missing_filled() {
        let mut t = TensorDyn::image(1280, 720, PixelFormat::Nv12, DType::U8, None).unwrap();
        t.set_colorimetry(Some(Colorimetry::default().with_range(ColorRange::Full)));
        let r = effective_colorimetry(&t);
        assert_eq!(r.range, Some(ColorRange::Full)); // kept
        assert_eq!(r.encoding, Some(ColorEncoding::Bt709)); // filled by heuristic (HD)
    }

    #[test]
    fn resolver_does_not_mutate_tensor() {
        let t = TensorDyn::image(1280, 720, PixelFormat::Nv12, DType::U8, None).unwrap();
        let _ = effective_colorimetry(&t);
        assert_eq!(t.colorimetry(), None); // unchanged
    }

    #[test]
    fn encoding_maps_to_yuv_matrix() {
        assert!(matches!(
            yuv_matrix(ColorEncoding::Bt601),
            yuv::YuvStandardMatrix::Bt601
        ));
        assert!(matches!(
            yuv_matrix(ColorEncoding::Bt709),
            yuv::YuvStandardMatrix::Bt709
        ));
        assert!(matches!(yuv_range(ColorRange::Full), yuv::YuvRange::Full));
        assert!(matches!(
            yuv_range(ColorRange::Limited),
            yuv::YuvRange::Limited
        ));
    }
}
