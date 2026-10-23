// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Preprocessing values shared by the image and decoder crates: how pixels
//! are normalized, how a source is fit into a model input, where the image
//! content lands ([`letterbox_rect`], [`ContentRegion`]), and a model's whole
//! recipe as one value ([`PreprocessSpec`]).
//!
//! They live here, beside [`unletter_norm`](crate::unletter_norm), so the
//! image crate (which applies them) and the decoder crate (which reads them
//! from model metadata) share one definition without depending on each
//! other. Only plain values and pure functions belong in this module.

use crate::{BoundingBox, Error, Region, Result};

crate::ef_vocabulary! {
    /// Where a letterboxed image sits inside the destination. The code is the
    /// one the C `EF_LETTERBOX_PLACEMENT_*` enumerators and the Python
    /// `LetterboxPlacement` enum use; the string is the `edgefirst.json` name.
    #[derive(Default)]
    #[non_exhaustive]
    pub enum LetterboxPlacement {
        /// Centred, padding split between both sides (the default). On odd
        /// slack the extra pixel goes to the right or bottom.
        #[default]
        Center = 0, "center", CENTER,
        /// Anchored at the top-left corner, padding on the right and bottom.
        TopLeft = 1, "topleft", TOP_LEFT,
    }
    #[doc(hidden)]
    pub mod letterbox_placement_code;
}

crate::ef_vocabulary! {
    /// A named [`Normalization`]. The code is the one the C
    /// `EF_NORMALIZATION_*` enumerators and the Python `NormalizationPreset`
    /// enum use; the names match the `edgefirst.json` presets.
    #[derive(Default)]
    #[non_exhaustive]
    pub enum NormalizationPreset {
        /// `x / 255`, giving `[0, 1]` (the default).
        #[default]
        Unsigned = 0, "unsigned", UNSIGNED,
        /// `x / 127.5 - 1`, giving `[-1, 1]`.
        Signed = 1, "signed", SIGNED,
        /// `x`, giving `[0, 255]`.
        Raw = 2, "raw", RAW,
        /// ImageNet mean and standard deviation.
        Imagenet = 3, "imagenet", IMAGENET,
    }
    #[doc(hidden)]
    pub mod normalization_code;
}

crate::ef_vocabulary! {
    /// How a source is fit into the destination, without the letterbox pad
    /// colour. The code is the one the C `EF_FIT_*` enumerators and the
    /// Python `Fit` enum use.
    #[derive(Default)]
    #[non_exhaustive]
    pub enum FitMode {
        /// Scale each axis independently to fill the destination (the
        /// default).
        #[default]
        Stretch = 0, "stretch", STRETCH,
        /// Keep the source aspect ratio and pad the rest.
        Letterbox = 1, "letterbox", LETTERBOX,
    }
    #[doc(hidden)]
    pub mod fit_code;
}

crate::ef_vocabulary! {
    /// Resampling filter. The code is the one the C `EF_INTERPOLATION_*`
    /// enumerators use; the string is the `edgefirst.json` and Python name.
    ///
    /// The image crate runs [`Nearest`](Self::Nearest) and
    /// [`Bilinear`](Self::Bilinear) (half-pixel centres, two taps per axis at
    /// every scale, matching OpenCV `INTER_LINEAR`). [`Bicubic`](Self::Bicubic)
    /// and [`Area`](Self::Area) describe reference pipelines; asking the image
    /// crate for them logs a warning and runs [`supported`](Self::supported).
    #[derive(Default)]
    #[non_exhaustive]
    pub enum Interpolation {
        /// Bilinear (the default).
        #[default]
        Bilinear = 0, "bilinear", BILINEAR,
        /// Nearest neighbour.
        Nearest = 1, "nearest", NEAREST,
        /// Bicubic.
        Bicubic = 2, "bicubic", BICUBIC,
        /// Pixel-area averaging.
        Area = 3, "area", AREA,
    }
    #[doc(hidden)]
    pub mod interpolation_code;
}

crate::ef_vocabulary! {
    /// The space a letterbox pad value is expressed in. The code is the one
    /// the C `EF_PAD_SPACE_*` enumerators use; the string is the
    /// `edgefirst.json` name.
    #[derive(Default)]
    #[non_exhaustive]
    pub enum PadSpace {
        /// A 0–255 pixel value, normalized like the image (the default).
        #[default]
        Pixel = 0, "pixel", PIXEL,
        /// A value written as-is into the normalized destination, e.g. 0.0 to
        /// pad with the mean colour after ImageNet normalization.
        Normalized = 1, "normalized", NORMALIZED,
    }
    #[doc(hidden)]
    pub mod pad_space_code;
}

impl Interpolation {
    /// The filter the image crate runs for this one: itself for
    /// [`Nearest`](Self::Nearest) and [`Bilinear`](Self::Bilinear), else
    /// [`Bilinear`](Self::Bilinear).
    pub const fn supported(self) -> Self {
        match self {
            Self::Nearest => Self::Nearest,
            Self::Bilinear | Self::Bicubic | Self::Area => Self::Bilinear,
        }
    }
}

macro_rules! serde_by_name {
    ($($ty:ident),+) => {$(
        impl serde::Serialize for $ty {
            fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
                s.serialize_str(self.as_str())
            }
        }

        impl<'de> serde::Deserialize<'de> for $ty {
            fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
                let s = <std::borrow::Cow<'de, str> as serde::Deserialize>::deserialize(d)?;
                Self::from_str_code(&s).ok_or_else(|| {
                    let names: Vec<&str> = Self::all().iter().map(|v| v.as_str()).collect();
                    serde::de::Error::custom(format!(
                        "unknown {} `{s}`, expected one of {names:?}",
                        stringify!($ty)
                    ))
                })
            }
        }
    )+};
}

serde_by_name!(
    LetterboxPlacement,
    NormalizationPreset,
    FitMode,
    Interpolation,
    PadSpace
);

const IMAGENET_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const IMAGENET_STD: [f32; 3] = [58.395, 57.12, 57.375];

/// Per-channel `(x - mean) / std` on 0–255 pixel values, in RGB order.
///
/// It applies to the colour channels of the destination; an alpha channel
/// always maps as [`NormalizationPreset::Unsigned`]. A `Grey` destination uses
/// the first channel's coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normalization {
    mean: [f32; 3],
    std: [f32; 3],
}

impl Default for Normalization {
    fn default() -> Self {
        Self::preset(NormalizationPreset::Unsigned)
    }
}

impl From<NormalizationPreset> for Normalization {
    fn from(p: NormalizationPreset) -> Self {
        Self::preset(p)
    }
}

impl Normalization {
    /// The coefficients of a named preset.
    pub const fn preset(p: NormalizationPreset) -> Self {
        let (mean, std) = match p {
            NormalizationPreset::Unsigned => ([0.0; 3], [255.0; 3]),
            NormalizationPreset::Signed => ([127.5; 3], [127.5; 3]),
            NormalizationPreset::Raw => ([0.0; 3], [1.0; 3]),
            NormalizationPreset::Imagenet => (IMAGENET_MEAN, IMAGENET_STD),
        };
        Self { mean, std }
    }

    /// `x / 255`.
    pub const fn unsigned() -> Self {
        Self::preset(NormalizationPreset::Unsigned)
    }

    /// `x / 127.5 - 1`.
    pub const fn signed() -> Self {
        Self::preset(NormalizationPreset::Signed)
    }

    /// `x` unchanged.
    pub const fn raw() -> Self {
        Self::preset(NormalizationPreset::Raw)
    }

    /// ImageNet mean `[123.675, 116.28, 103.53]`, std `[58.395, 57.12, 57.375]`.
    pub const fn imagenet() -> Self {
        Self::preset(NormalizationPreset::Imagenet)
    }

    /// Explicit coefficients on the 0–255 scale: one value (every channel)
    /// or three (RGB).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] for another count, a non-finite `mean`, or
    /// a `std` that is not finite and positive.
    ///
    /// ```
    /// use edgefirst_tensor::Normalization;
    ///
    /// assert_eq!(Normalization::mean_std(&[0.0], &[255.0]).unwrap(), Normalization::unsigned());
    /// assert!(Normalization::mean_std(&[0.0], &[0.0]).is_err());
    /// assert!(Normalization::mean_std(&[0.0], &[-1.0]).is_err());
    /// ```
    pub fn mean_std(mean: &[f32], std: &[f32]) -> Result<Self> {
        let count = |name: &str, v: &[f32]| {
            broadcast_rgb(v).ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "normalization {name} needs 1 or 3 values, got {}",
                    v.len()
                ))
            })
        };
        let mean = count("mean", mean)?;
        let std = count("std", std)?;
        if mean.iter().any(|m| !m.is_finite()) {
            return Err(Error::InvalidArgument(format!(
                "normalization mean {mean:?} must be finite"
            )));
        }
        if std.iter().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err(Error::InvalidArgument(format!(
                "normalization std {std:?} must be finite and positive"
            )));
        }
        Ok(Self { mean, std })
    }

    /// Coefficients on the 0–1 scale, as Hugging Face and torchvision write
    /// them; scaled by `scale` (usually 255) onto the 0–255 scale in f64.
    ///
    /// # Errors
    ///
    /// As for [`mean_std`](Self::mean_std).
    ///
    /// ```
    /// use edgefirst_tensor::Normalization;
    ///
    /// let n = Normalization::unit_mean_std(&[0.5], &[0.5], 255.0).unwrap();
    /// assert_eq!(n, Normalization::signed());
    /// ```
    pub fn unit_mean_std(mean: &[f32], std: &[f32], scale: f64) -> Result<Self> {
        let scaled =
            |v: &[f32]| -> Vec<f32> { v.iter().map(|&x| (f64::from(x) * scale) as f32).collect() };
        Self::mean_std(&scaled(mean), &scaled(std))
    }

    /// The per-channel mean, RGB order, on the 0–255 scale.
    pub fn mean(&self) -> [f32; 3] {
        self.mean
    }

    /// The per-channel standard deviation, RGB order, on the 0–255 scale.
    pub fn std(&self) -> [f32; 3] {
        self.std
    }

    /// The preset with exactly these coefficients, if any.
    pub fn as_preset(&self) -> Option<NormalizationPreset> {
        NormalizationPreset::all()
            .iter()
            .copied()
            .find(|&p| Self::preset(p) == *self)
    }
}

/// One value repeated for every RGB channel, or three values as given;
/// `None` for any other count.
///
/// ```
/// use edgefirst_tensor::broadcast_rgb;
///
/// assert_eq!(broadcast_rgb(&[1]), Some([1, 1, 1]));
/// assert_eq!(broadcast_rgb(&[1, 2]), None);
/// ```
pub fn broadcast_rgb<T: Copy>(v: &[T]) -> Option<[T; 3]> {
    match *v {
        [x] => Some([x; 3]),
        [r, g, b] => Some([r, g, b]),
        _ => None,
    }
}

/// How a source is fit into the requested destination shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Fit {
    /// Stretch the source (or source crop) to fill the whole destination.
    #[default]
    Stretch,
    /// Keep the source aspect ratio and pad the rest of the destination with
    /// the RGBA pixel `pad` (e.g. `[114, 114, 114, 255]` for YOLO-style
    /// preprocessing). Where the image sits is a [`LetterboxPlacement`].
    Letterbox {
        /// The RGBA pad pixel.
        pad: [u8; 4],
    },
}

/// The rectangle an `src_w`×`src_h` source occupies when letterboxed into a
/// `dst_w`×`dst_h` destination: the larger axis fills, the other is scaled
/// to keep the aspect ratio (rounded, at least one pixel), and `placement`
/// positions it. A zero-sized source fills the destination.
///
/// This is the one letterbox computation: `convert` paints this rectangle,
/// and [`ContentRegion`]s are built from it.
///
/// ```
/// use edgefirst_tensor::{letterbox_rect, LetterboxPlacement, Region};
///
/// let r = letterbox_rect(1280, 720, 640, 640, LetterboxPlacement::Center);
/// assert_eq!(r, Region::new(0, 140, 640, 360));
/// let r = letterbox_rect(1280, 720, 640, 640, LetterboxPlacement::TopLeft);
/// assert_eq!(r, Region::new(0, 0, 640, 360));
/// ```
pub fn letterbox_rect(
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
    placement: LetterboxPlacement,
) -> Region {
    if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
        return Region::new(0, 0, dst_w, dst_h);
    }
    let src_aspect = src_w as f64 / src_h as f64;
    let dst_aspect = dst_w as f64 / dst_h as f64;
    let (w, h) = if src_aspect > dst_aspect {
        (dst_w, ((dst_w as f64 / src_aspect).round() as usize).max(1))
    } else {
        (((dst_h as f64 * src_aspect).round() as usize).max(1), dst_h)
    };
    let (x, y) = match placement {
        LetterboxPlacement::Center => (dst_w.saturating_sub(w) / 2, dst_h.saturating_sub(h) / 2),
        LetterboxPlacement::TopLeft => (0, 0),
    };
    Region::new(x, y, w, h)
}

/// Where the image content sits in a destination: in pixels, and as
/// `[xmin, ymin, xmax, ymax]` normalized by the destination size, the form
/// [`unletter_norm`](crate::unletter_norm) and the semantic decoder take.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContentRegion {
    /// The content rectangle in destination pixels.
    pub pixels: Region,
    /// The content rectangle normalized by the destination size.
    pub normalized: [f32; 4],
}

impl ContentRegion {
    /// The region of `pixels` in a `dst_w`×`dst_h` destination.
    pub fn new(pixels: Region, dst_w: usize, dst_h: usize) -> Self {
        let (w, h) = (dst_w.max(1) as f32, dst_h.max(1) as f32);
        Self {
            pixels,
            normalized: [
                pixels.x as f32 / w,
                pixels.y as f32 / h,
                pixels.x.saturating_add(pixels.width) as f32 / w,
                pixels.y.saturating_add(pixels.height) as f32 / h,
            ],
        }
    }

    /// Content that fills a `dst_w`×`dst_h` destination.
    pub fn full(dst_w: usize, dst_h: usize) -> Self {
        Self::new(Region::new(0, 0, dst_w, dst_h), dst_w, dst_h)
    }

    /// Whether the content fills the destination.
    pub fn is_full(&self) -> bool {
        self.normalized == [0.0, 0.0, 1.0, 1.0]
    }

    /// The normalized content rectangle when it does not fill the
    /// destination (a letterbox), else `None`.
    pub fn letterboxed(&self) -> Option<[f32; 4]> {
        (!self.is_full()).then_some(self.normalized)
    }

    /// Map a box normalized to the destination back to the content, i.e. to
    /// the source image (see [`unletter_norm`](crate::unletter_norm)).
    pub fn unletter(&self, b: BoundingBox) -> BoundingBox {
        crate::unletter_norm(b, self.normalized)
    }
}

/// Something that knows where it puts the image content in a destination,
/// such as the image crate's `ConvertOptions` or a [`PreprocessSpec`]. The
/// decoder takes it to map its output back to the source image with the same
/// geometry the convert used.
pub trait ContentRegionSource {
    /// The content region for an `src_w`×`src_h` source converted into a
    /// `dst_w`×`dst_h` destination.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] when the geometry cannot be resolved, e.g.
    /// a source crop outside the source.
    fn content_region(
        &self,
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Result<ContentRegion>;
}

/// A model's input preprocessing as plain values: what the image crate needs
/// to build a convert, produced by the decoder from model metadata.
///
/// The default is the documented `edgefirst.json` default: unsigned
/// normalization, a centred letterbox padded with pixel 114, bilinear.
///
/// ```
/// use edgefirst_tensor::{FitMode, LetterboxPlacement, PreprocessSpec};
///
/// let spec = PreprocessSpec::default();
/// assert_eq!(spec.fit, FitMode::Letterbox);
/// assert_eq!(spec.placement, LetterboxPlacement::Center);
/// assert_eq!(spec.pad, [114.0; 3]);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PreprocessSpec {
    /// Pixel normalization.
    pub normalization: Normalization,
    /// Stretch or letterbox.
    pub fit: FitMode,
    /// Letterbox placement; ignored by a stretch.
    pub placement: LetterboxPlacement,
    /// Letterbox pad per RGB channel, in the space `pad_space` names.
    pub pad: [f32; 3],
    /// The space `pad` is expressed in.
    pub pad_space: PadSpace,
    /// Resampling filter of the reference pipeline.
    pub interpolation: Interpolation,
    /// Model input `(width, height)`, when known.
    pub size: Option<(usize, usize)>,
}

impl Default for PreprocessSpec {
    fn default() -> Self {
        Self {
            normalization: Normalization::unsigned(),
            fit: FitMode::Letterbox,
            placement: LetterboxPlacement::Center,
            pad: [114.0; 3],
            pad_space: PadSpace::Pixel,
            interpolation: Interpolation::Bilinear,
            size: None,
        }
    }
}

impl PreprocessSpec {
    /// The default spec; set the public fields to change it.
    pub fn new() -> Self {
        Self::default()
    }
}

impl ContentRegionSource for PreprocessSpec {
    fn content_region(
        &self,
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Result<ContentRegion> {
        Ok(match self.fit {
            FitMode::Stretch => ContentRegion::full(dst_w, dst_h),
            FitMode::Letterbox => ContentRegion::new(
                letterbox_rect(src_w, src_h, dst_w, dst_h, self.placement),
                dst_w,
                dst_h,
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_rect_rounds_and_places() {
        let c = LetterboxPlacement::Center;
        let t = LetterboxPlacement::TopLeft;
        // 641x479 -> width fills, height round(640 * 479 / 641) = 478.
        assert_eq!(
            letterbox_rect(641, 479, 640, 640, c),
            Region::new(0, 81, 640, 478)
        );
        assert_eq!(
            letterbox_rect(641, 479, 640, 640, t),
            Region::new(0, 0, 640, 478)
        );
        assert_eq!(
            letterbox_rect(479, 641, 640, 640, c),
            Region::new(81, 0, 478, 640)
        );
        assert_eq!(
            letterbox_rect(1, 3, 320, 256, c),
            Region::new(117, 0, 85, 256)
        );
        assert_eq!(
            letterbox_rect(3, 1, 320, 256, c),
            Region::new(0, 74, 320, 107)
        );
        assert_eq!(
            letterbox_rect(640, 640, 320, 320, c),
            Region::new(0, 0, 320, 320)
        );
        assert_eq!(letterbox_rect(0, 5, 32, 16, c), Region::new(0, 0, 32, 16));
        // A sliver keeps at least one pixel.
        assert_eq!(
            letterbox_rect(10000, 1, 64, 64, t),
            Region::new(0, 0, 64, 1)
        );
    }

    #[test]
    fn content_region_normalizes_pixels() {
        let r = ContentRegion::new(Region::new(0, 140, 640, 360), 640, 640);
        assert_eq!(r.normalized, [0.0, 0.21875, 1.0, 0.78125]);
        assert!(!r.is_full());
        assert_eq!(r.letterboxed(), Some(r.normalized));
        assert!(ContentRegion::full(32, 16).is_full());
        assert_eq!(ContentRegion::full(32, 16).letterboxed(), None);
    }

    #[test]
    fn interpolation_codes_keep_c_values() {
        assert_eq!(Interpolation::Bilinear.code(), 0);
        assert_eq!(Interpolation::Nearest.code(), 1);
        assert_eq!(Interpolation::Bicubic.supported(), Interpolation::Bilinear);
        assert_eq!(Interpolation::Area.supported(), Interpolation::Bilinear);
        assert_eq!(Interpolation::Nearest.supported(), Interpolation::Nearest);
    }

    #[test]
    fn vocab_serde_round_trips() {
        for &p in LetterboxPlacement::all() {
            let s = serde_json::to_string(&p).unwrap();
            assert_eq!(serde_json::from_str::<LetterboxPlacement>(&s).unwrap(), p);
        }
        assert_eq!(
            serde_json::to_string(&PadSpace::Normalized).unwrap(),
            "\"normalized\""
        );
        assert_eq!(
            serde_json::to_string(&LetterboxPlacement::TopLeft).unwrap(),
            "\"topleft\""
        );
        assert!(serde_json::from_str::<LetterboxPlacement>("\"top_left\"").is_err());
        let e = serde_json::from_str::<Interpolation>("\"lanczos\"").unwrap_err();
        assert!(e.to_string().contains("bicubic"), "{e}");
    }

    #[test]
    fn spec_content_region_follows_fit() {
        let mut spec = PreprocessSpec::new();
        let lb = spec.content_region(1280, 720, 640, 640).unwrap();
        assert_eq!(lb.pixels, Region::new(0, 140, 640, 360));
        spec.fit = FitMode::Stretch;
        assert!(spec.content_region(1280, 720, 640, 640).unwrap().is_full());
    }

    #[test]
    fn normalization_finds_its_preset() {
        assert_eq!(
            Normalization::imagenet().as_preset(),
            Some(NormalizationPreset::Imagenet)
        );
        let custom = Normalization::mean_std(&[1.0], &[2.0]).unwrap();
        assert_eq!(custom.as_preset(), None);
        assert!(Normalization::mean_std(&[0.0, 0.0], &[1.0]).is_err());
        assert!(Normalization::mean_std(&[f32::NAN], &[1.0]).is_err());
    }
}
