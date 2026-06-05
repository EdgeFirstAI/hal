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
    resolve_colorimetry(t.colorimetry(), t.height())
}

/// Core resolver shared by `effective_colorimetry` and any caller that already
/// has the raw `(colorimetry, height)` pair (e.g. a `Tensor<u8>` fill target).
/// Fills only the missing axes by the height heuristic; never mutates.
pub(crate) fn resolve_colorimetry(
    colorimetry: Option<Colorimetry>,
    height: Option<usize>,
) -> Colorimetry {
    let hd = height.map(|h| h >= HD_THRESHOLD).unwrap_or(true);
    let src = colorimetry.unwrap_or_default();
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

/// YUV→RGB shader coefficients resolved from a tensor's colorimetry.
///
/// `y_offset` / `y_scale` normalise the luma for the sample range; the four
/// chroma cross-terms (`c_vr`, `c_ug`, `c_vg`, `c_ub`) encode the
/// `ColorEncoding` matrix folded with the range's chroma scaling. These feed
/// the shared GLSL YUV→RGB fragment shaders (see `gl::shaders_common`) as
/// `float` uniforms — identical on every GL backend (macOS ANGLE + Linux
/// GLES); only the texture import differs by platform.
///
/// The coefficients derive from the BT matrix luma weights `(kr, kb)`:
/// ```text
///   c_vr = 2*(1-kr) * c_scale
///   c_ub = 2*(1-kb) * c_scale
///   c_ug = (2*(1-kb)*kb / kg) * c_scale,  kg = 1-kr-kb
///   c_vg = (2*(1-kr)*kr / kg) * c_scale
/// ```
/// For limited range the luma gain is 255/219 and chroma scale 255/224; for
/// full range both are 1.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct YuvToRgbCoeffs {
    pub y_offset: f32,
    pub y_scale: f32,
    pub c_vr: f32,
    pub c_ug: f32,
    pub c_vg: f32,
    pub c_ub: f32,
}

/// Compute the in-shader YUV→RGB coefficients for a resolved
/// `(ColorEncoding, ColorRange)`. Pure; platform-neutral.
pub(crate) fn yuv_to_rgb_coeffs(encoding: ColorEncoding, range: ColorRange) -> YuvToRgbCoeffs {
    // Luma weights (kr, kb) and range swings come from the canonical source in
    // `edgefirst_tensor::colorimetry` so the GL coefficients stay in lockstep
    // with the CPU encoders. The arithmetic below is kept in `f32` (the GL
    // uniform type) for bit-stable shader output.
    let w = encoding.luma_weights();
    let (kr, kb) = (w.kr as f32, w.kb as f32);
    let kg = 1.0 - kr - kb;

    // Limited: luma spans 16..235 (gain 255/219), chroma spans 16..240 (gain
    // 255/224). Full: both unity.
    let s = range.scaling();
    let y_offset = s.y_offset as f32 / 255.0;
    let y_scale = 255.0 / s.y_swing as f32;
    let c_scale = 255.0 / s.c_swing as f32;

    YuvToRgbCoeffs {
        y_offset,
        y_scale,
        c_vr: 2.0 * (1.0 - kr) * c_scale,
        c_ub: 2.0 * (1.0 - kb) * c_scale,
        c_ug: (2.0 * (1.0 - kb) * kb / kg) * c_scale,
        c_vg: (2.0 * (1.0 - kr) * kr / kg) * c_scale,
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
    fn yuv_to_rgb_coeffs_match_reference_values() {
        // Reference YCbCr→RGB cross-terms (well-known broadcast constants). The
        // shader applies `r = y' + c_vr*v'`, `b = y' + c_ub*u'`, etc.
        fn close(a: f32, b: f32, what: &str) {
            assert!((a - b).abs() < 2e-3, "{what}: {a} vs expected {b}");
        }

        // BT.601 limited range — the canonical SD decode.
        let c = yuv_to_rgb_coeffs(ColorEncoding::Bt601, ColorRange::Limited);
        close(c.y_offset, 16.0 / 255.0, "601L y_offset");
        close(c.y_scale, 255.0 / 219.0, "601L y_scale");
        close(c.c_vr, 1.596, "601L c_vr");
        close(c.c_ub, 2.017, "601L c_ub");
        close(c.c_ug, 0.391, "601L c_ug");
        close(c.c_vg, 0.813, "601L c_vg");

        // BT.709 limited range — the canonical HD decode.
        let c = yuv_to_rgb_coeffs(ColorEncoding::Bt709, ColorRange::Limited);
        close(c.c_vr, 1.793, "709L c_vr");
        close(c.c_ub, 2.113, "709L c_ub");
        close(c.c_ug, 0.213, "709L c_ug");
        close(c.c_vg, 0.533, "709L c_vg");

        // Full range zeroes the luma offset and unity-scales luma; the chroma
        // cross-terms drop the 255/224 gain.
        let c = yuv_to_rgb_coeffs(ColorEncoding::Bt601, ColorRange::Full);
        close(c.y_offset, 0.0, "601F y_offset");
        close(c.y_scale, 1.0, "601F y_scale");
        close(c.c_vr, 2.0 * (1.0 - 0.299), "601F c_vr");
        close(c.c_ub, 2.0 * (1.0 - 0.114), "601F c_ub");
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
