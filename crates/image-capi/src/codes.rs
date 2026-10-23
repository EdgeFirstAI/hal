// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The image option vocabularies, as named C enumerators.
//!
//! Each enumerator's value is asserted against the `code()` of the
//! `edgefirst_image` vocabulary it names, at compile time, so the header
//! cannot drift from the Rust declaration. The functions that take one keep
//! a plain `uint32_t` parameter: an out-of-range value is rejected with
//! `EINVAL` rather than transmuted into a Rust enum, which would be
//! undefined behaviour for a value no variant names.

use edgefirst_image::{
    ColorMode, ComputeBackend, FitMode, Flip, Interpolation, LetterboxPlacement,
    NormalizationPreset, PadSpace, Rotation,
};

/// The preprocessing vocabularies `ef_preprocess_spec` carries, declared once
/// in header-only `edgefirst/detect.h` so `decoder.h` and `image.h` share
/// them; asserted against this library's codes below.
pub use edgefirst_decoder_abi::{EfFit, EfInterpolation, EfLetterboxPlacement, EfPadSpace};

/// A quarter-turn rotation, for `ef_image_processor_convert` and friends.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfRotation {
    None = 0,
    Clockwise90 = 1,
    Rotate180 = 2,
    CounterClockwise90 = 3,
}

/// A mirror, for `ef_image_processor_convert` and friends.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfFlip {
    None = 0,
    /// Mirror top to bottom.
    Vertical = 1,
    /// Mirror left to right.
    Horizontal = 2,
}

/// How a mask's palette colour is chosen, for the draw functions.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfColorMode {
    /// By class label.
    Class = 0,
    /// By detection index.
    Instance = 1,
    /// By track ID.
    Track = 2,
}

/// The backend `ef_image_processor_new_with_backend` forces.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfComputeBackend {
    /// Auto-detect, with the fallback chain.
    Auto = 0,
    /// CPU only.
    Cpu = 1,
    /// G2D only.
    G2d = 2,
    /// OpenGL only.
    OpenGl = 3,
}

/// A named normalization, `(x - mean) / std` on 0-255 values, for
/// `ef_convert_options_set_normalization_preset`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfNormalization {
    /// `x / 255`, giving `[0, 1]`.
    Unsigned = 0,
    /// `x / 127.5 - 1`, giving `[-1, 1]`.
    Signed = 1,
    /// `x`, giving `[0, 255]`.
    Raw = 2,
    /// ImageNet mean and standard deviation.
    Imagenet = 3,
}

const _: () = {
    assert!(EfRotation::None as u32 == Rotation::None.code());
    assert!(EfRotation::Clockwise90 as u32 == Rotation::Clockwise90.code());
    assert!(EfRotation::Rotate180 as u32 == Rotation::Rotate180.code());
    assert!(EfRotation::CounterClockwise90 as u32 == Rotation::CounterClockwise90.code());

    assert!(EfFlip::None as u32 == Flip::None.code());
    assert!(EfFlip::Vertical as u32 == Flip::Vertical.code());
    assert!(EfFlip::Horizontal as u32 == Flip::Horizontal.code());

    assert!(EfColorMode::Class as u32 == ColorMode::Class.code());
    assert!(EfColorMode::Instance as u32 == ColorMode::Instance.code());
    assert!(EfColorMode::Track as u32 == ColorMode::Track.code());

    assert!(EfFit::Stretch as u32 == FitMode::Stretch.code());
    assert!(EfFit::Letterbox as u32 == FitMode::Letterbox.code());

    assert!(EfComputeBackend::Auto as u32 == ComputeBackend::Auto.code());
    assert!(EfComputeBackend::Cpu as u32 == ComputeBackend::Cpu.code());
    assert!(EfComputeBackend::G2d as u32 == ComputeBackend::G2d.code());
    assert!(EfComputeBackend::OpenGl as u32 == ComputeBackend::OpenGl.code());

    assert!(EfLetterboxPlacement::Center as u32 == LetterboxPlacement::Center.code());
    assert!(EfLetterboxPlacement::TopLeft as u32 == LetterboxPlacement::TopLeft.code());

    assert!(EfNormalization::Unsigned as u32 == NormalizationPreset::Unsigned.code());
    assert!(EfNormalization::Signed as u32 == NormalizationPreset::Signed.code());
    assert!(EfNormalization::Raw as u32 == NormalizationPreset::Raw.code());
    assert!(EfNormalization::Imagenet as u32 == NormalizationPreset::Imagenet.code());

    assert!(EfPadSpace::Pixel as u32 == PadSpace::Pixel.code());
    assert!(EfPadSpace::Normalized as u32 == PadSpace::Normalized.code());

    assert!(EfInterpolation::Bilinear as u32 == Interpolation::Bilinear.code());
    assert!(EfInterpolation::Nearest as u32 == Interpolation::Nearest.code());
    assert!(EfInterpolation::Bicubic as u32 == Interpolation::Bicubic.code());
    assert!(EfInterpolation::Area as u32 == Interpolation::Area.code());
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The C enumerator for a variant, spelled the way cbindgen's
    /// `ScreamingSnakeCase` spells the matching `Ef*` variant (which shares
    /// the Rust variant's name): `CounterClockwise90` ->
    /// `COUNTER_CLOCKWISE90`.
    fn enumerator(prefix: &str, v: impl std::fmt::Debug) -> String {
        let mut out = format!("{prefix}_");
        for (i, c) in format!("{v:?}").chars().enumerate() {
            if c.is_ascii_uppercase() && i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_uppercase());
        }
        out
    }

    /// `(C enumerator name, code)` for every enumerator the header must
    /// declare, built from the Rust vocabularies rather than restated.
    fn expected_enumerators() -> Vec<(String, u32)> {
        let mut out = Vec::new();
        out.extend(
            Rotation::all()
                .iter()
                .map(|&v| (enumerator("EF_ROTATION", v), v.code())),
        );
        out.extend(
            Flip::all()
                .iter()
                .map(|&v| (enumerator("EF_FLIP", v), v.code())),
        );
        out.extend(
            ColorMode::all()
                .iter()
                .map(|&v| (enumerator("EF_COLOR_MODE", v), v.code())),
        );
        out.extend(
            FitMode::all()
                .iter()
                .map(|&v| (enumerator("EF_FIT", v), v.code())),
        );
        out.extend(
            ComputeBackend::all()
                .iter()
                .map(|&v| (enumerator("EF_COMPUTE_BACKEND", v), v.code())),
        );
        out.extend(
            LetterboxPlacement::all()
                .iter()
                .map(|&v| (enumerator("EF_LETTERBOX_PLACEMENT", v), v.code())),
        );
        out.extend(
            NormalizationPreset::all()
                .iter()
                .map(|&v| (enumerator("EF_NORMALIZATION", v), v.code())),
        );
        out.extend(
            PadSpace::all()
                .iter()
                .map(|&v| (enumerator("EF_PAD_SPACE", v), v.code())),
        );
        out.extend(
            Interpolation::all()
                .iter()
                .map(|&v| (enumerator("EF_INTERPOLATION", v), v.code())),
        );
        out
    }

    #[test]
    fn every_vocabulary_variant_has_a_c_enumerator() {
        // The const block proves the listed values agree; this proves the
        // lists are complete.
        assert_eq!(Rotation::all().len(), 4, "add the EfRotation enumerator");
        assert_eq!(Flip::all().len(), 3, "add the EfFlip enumerator");
        assert_eq!(ColorMode::all().len(), 3, "add the EfColorMode enumerator");
        assert_eq!(FitMode::all().len(), 2, "add the EfFit enumerator");
        assert_eq!(
            ComputeBackend::all().len(),
            4,
            "add the EfComputeBackend enumerator"
        );
        assert_eq!(
            LetterboxPlacement::all().len(),
            2,
            "add the EfLetterboxPlacement enumerator"
        );
        assert_eq!(
            NormalizationPreset::all().len(),
            4,
            "add the EfNormalization enumerator"
        );
        assert_eq!(PadSpace::all().len(), 2, "add the EfPadSpace enumerator");
        assert_eq!(
            Interpolation::all().len(),
            4,
            "add the EfInterpolation enumerator"
        );
    }

    /// `image.h` and the `detect.h` it includes: the preprocessing
    /// vocabularies are declared in the latter.
    fn headers_text() -> String {
        let image = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/include/edgefirst/image.h"
        ))
        .expect("cbindgen wrote include/edgefirst/image.h");
        let detect = include_str!("../../decoder-abi/include/edgefirst/detect.h");
        image + detect
    }

    #[test]
    fn the_header_declares_every_option_code() {
        let header = headers_text();
        for (name, code) in expected_enumerators() {
            let line = format!("{name} = {code},");
            assert!(
                header.contains(&line),
                "image.h (with detect.h) does not declare `{line}`"
            );
        }
        for ty in [
            "ef_rotation",
            "ef_flip",
            "ef_color_mode",
            "ef_fit",
            "ef_compute_backend",
            "ef_letterbox_placement",
            "ef_normalization",
            "ef_pad_space",
            "ef_interpolation",
        ] {
            assert!(
                header.contains(&format!("enum {ty}")),
                "image.h (with detect.h) does not declare `enum {ty}`"
            );
        }
    }

    #[test]
    fn the_shared_vocabularies_are_declared_once() {
        let image = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/include/edgefirst/image.h"
        ))
        .expect("cbindgen wrote include/edgefirst/image.h");
        for ty in [
            "enum ef_fit",
            "enum ef_letterbox_placement",
            "enum ef_pad_space",
            "enum ef_interpolation",
            "struct ef_preprocess_spec {",
        ] {
            assert!(
                !image.contains(ty),
                "`{ty}` belongs to detect.h; image.h must not redeclare it"
            );
        }
    }

    #[test]
    fn unknown_option_codes_decode_to_nothing() {
        assert_eq!(Rotation::from_code(4), None);
        assert_eq!(Flip::from_code(3), None);
        assert_eq!(ColorMode::from_code(3), None);
        assert_eq!(FitMode::from_code(2), None);
        assert_eq!(ComputeBackend::from_code(4), None);
        assert_eq!(LetterboxPlacement::from_code(2), None);
        assert_eq!(NormalizationPreset::from_code(4), None);
        assert_eq!(PadSpace::from_code(2), None);
        assert_eq!(Interpolation::from_code(4), None);
        assert_eq!(Rotation::from_code(u32::MAX), None);
    }
}
