// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Convert options: geometry (rotation, flip, crop, letterbox placement and
//! pad) and the pixel value mapping (normalization, output quantization).

use crate::{Crop, Error, Flip, Region, Result, Rotation};
use edgefirst_tensor::{DType, PixelFormat, QuantMode, Quantization};

edgefirst_tensor::ef_vocabulary! {
    /// Where a letterboxed image sits inside the destination. The code is the
    /// one the C `EF_LETTERBOX_PLACEMENT_*` enumerators and the Python
    /// `LetterboxPlacement` enum use.
    #[derive(Default)]
    pub enum LetterboxPlacement {
        /// Centred, padding split between both sides (the default).
        #[default]
        Center = 0, "center", CENTER,
        /// Anchored at the top-left corner, padding on the right and bottom.
        TopLeft = 1, "top_left", TOP_LEFT,
    }
    #[doc(hidden)]
    pub mod letterbox_placement_code;
}

edgefirst_tensor::ef_vocabulary! {
    /// A named [`Normalization`]. The code is the one the C
    /// `EF_NORMALIZATION_*` enumerators and the Python `Normalization` enum
    /// use; the names match the `edgefirst.json` presets.
    #[derive(Default)]
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

    /// Explicit coefficients: one value (every channel) or three (RGB).
    /// `std` must be finite and non-zero, `mean` finite.
    pub fn mean_std(mean: &[f32], std: &[f32]) -> Result<Self> {
        let mean = broadcast3("mean", mean)?;
        let std = broadcast3("std", std)?;
        if mean.iter().any(|m| !m.is_finite()) {
            return Err(Error::InvalidOptions(format!(
                "normalization mean {mean:?} must be finite"
            )));
        }
        if std.iter().any(|s| !s.is_finite() || *s == 0.0) {
            return Err(Error::InvalidOptions(format!(
                "normalization std {std:?} must be finite and non-zero"
            )));
        }
        Ok(Self { mean, std })
    }

    /// The per-channel mean, RGB order.
    pub fn mean(&self) -> [f32; 3] {
        self.mean
    }

    /// The per-channel standard deviation, RGB order.
    pub fn std(&self) -> [f32; 3] {
        self.std
    }
}

fn broadcast3(name: &str, v: &[f32]) -> Result<[f32; 3]> {
    match v {
        [x] => Ok([*x; 3]),
        [r, g, b] => Ok([*r, *g, *b]),
        _ => Err(Error::InvalidOptions(format!(
            "normalization {name} needs 1 or 3 values, got {}",
            v.len()
        ))),
    }
}

/// Options for [`convert`](crate::ImageProcessorTrait::convert). The default
/// converts the whole source, stretched, with no rotation or flip, and maps
/// pixels as [`Normalization::unsigned`] for float destinations.
///
/// ```
/// # use edgefirst_image::{ConvertOptions, LetterboxPlacement, Normalization, Rotation};
/// let opts = ConvertOptions::new()
///     .letterbox([0, 0, 0, 255])
///     .placement(LetterboxPlacement::TopLeft)
///     .pad_output([0.0; 4])
///     .normalization(Normalization::imagenet())
///     .rotation(Rotation::None);
/// # let _ = opts;
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ConvertOptions {
    rotation: Rotation,
    flip: Flip,
    crop: Crop,
    placement: LetterboxPlacement,
    pad_output: Option<[f32; 4]>,
    normalization: Normalization,
}

impl ConvertOptions {
    /// The default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Rotate the output, in 90° steps.
    pub fn rotation(mut self, rotation: Rotation) -> Self {
        self.rotation = rotation;
        self
    }

    /// Flip the output.
    pub fn flip(mut self, flip: Flip) -> Self {
        self.flip = flip;
        self
    }

    /// Set the source region and fit together.
    pub fn crop(mut self, crop: Crop) -> Self {
        self.crop = crop;
        self
    }

    /// Sample only `source` of the input; `None` samples all of it.
    pub fn source(mut self, source: Option<Region>) -> Self {
        self.crop.source = source;
        self
    }

    /// Letterbox: keep the source aspect ratio and pad the rest with the
    /// RGBA pixel `pad`, which is mapped like the image.
    pub fn letterbox(mut self, pad: [u8; 4]) -> Self {
        self.crop.fit = crate::Fit::Letterbox { pad };
        self
    }

    /// Where a letterboxed image sits. Ignored without a letterbox fit.
    pub fn placement(mut self, placement: LetterboxPlacement) -> Self {
        self.placement = placement;
        self
    }

    /// Pad with these values as written to the destination (after
    /// normalization; quantized for integer destinations), instead of mapping
    /// the pixel pad. Ignored without a letterbox fit.
    pub fn pad_output(mut self, pad: [f32; 4]) -> Self {
        self.pad_output = Some(pad);
        self
    }

    /// How pixel values map to the destination.
    pub fn normalization(mut self, normalization: impl Into<Normalization>) -> Self {
        self.normalization = normalization.into();
        self
    }

    /// The output rotation.
    pub fn get_rotation(&self) -> Rotation {
        self.rotation
    }

    /// The output flip.
    pub fn get_flip(&self) -> Flip {
        self.flip
    }

    /// The source region and fit.
    pub fn get_crop(&self) -> Crop {
        self.crop
    }

    /// The letterbox placement.
    pub fn get_placement(&self) -> LetterboxPlacement {
        self.placement
    }

    /// The output-domain pad, when set.
    pub fn get_pad_output(&self) -> Option<[f32; 4]> {
        self.pad_output
    }

    /// The pixel value mapping.
    pub fn get_normalization(&self) -> Normalization {
        self.normalization
    }
}

#[cfg(feature = "decode")]
impl ConvertOptions {
    /// Apply a model's `input.preprocessing` from `edgefirst.json`:
    /// normalization, letterbox fit, placement and pad. `resize: stretch`
    /// clears a letterbox; with no `resize` and no `letterbox` the fit is left
    /// as it is. `interpolation` is a processor setting
    /// ([`ImageProcessor::set_interpolation`](crate::ImageProcessor)) and is
    /// not read here.
    pub fn with_preprocessing(
        mut self,
        pre: &edgefirst_decoder::schema::Preprocessing,
    ) -> Result<Self> {
        use edgefirst_decoder::schema::{LetterboxPlacement as P, PadSpace, ResizeMode};
        if let Some(n) = &pre.normalization {
            let (mean, std) = n
                .mean_std(3)
                .map_err(|e| Error::InvalidOptions(e.to_string()))?;
            self.normalization = Normalization::mean_std(&mean, &std)?;
        }
        let letterbox = match pre.resize {
            Some(ResizeMode::Stretch) => false,
            Some(ResizeMode::Letterbox) => true,
            Some(other) => {
                return Err(Error::NotSupported(format!("resize mode {other:?}")));
            }
            None => pre.letterbox.is_some(),
        };
        if pre.resize == Some(ResizeMode::Stretch) {
            self.crop.fit = crate::Fit::Stretch;
        }
        if !letterbox {
            return Ok(self);
        }
        let lb = pre.letterbox.clone().unwrap_or_default();
        self.placement = match lb.placement {
            None | Some(P::Center) => LetterboxPlacement::Center,
            Some(P::TopLeft) => LetterboxPlacement::TopLeft,
            Some(other) => {
                return Err(Error::NotSupported(format!(
                    "letterbox placement {other:?}"
                )));
            }
        };
        let pad: [f32; 3] = match lb.pad.as_deref() {
            None => [114.0; 3],
            Some(p) => broadcast3("letterbox pad", p)?,
        };
        match lb.pad_space {
            None | Some(PadSpace::Pixel) => {
                let [r, g, b] = pad.map(|v| v.round().clamp(0.0, 255.0) as u8);
                self.crop.fit = crate::Fit::Letterbox {
                    pad: [r, g, b, 255],
                };
                self.pad_output = None;
            }
            Some(PadSpace::Normalized) => {
                self.crop.fit = crate::Fit::Letterbox {
                    pad: [0, 0, 0, 255],
                };
                self.pad_output = Some([pad[0], pad[1], pad[2], 1.0]);
            }
            Some(other) => {
                return Err(Error::NotSupported(format!(
                    "letterbox pad space {other:?}"
                )));
            }
        }
        Ok(self)
    }
}

/// The output mapping a convert must apply beyond the plain one, or `None`
/// when the plain mapping is exact. `letterboxed` says whether the resolved
/// crop pads the destination.
pub(crate) fn output_map_for(
    options: &ConvertOptions,
    dst: &edgefirst_tensor::TensorDyn,
    letterboxed: bool,
) -> Result<Option<(OutputMap, Option<[f32; 4]>)>> {
    let fmt = dst.format().ok_or(Error::NotAnImage)?;
    let mappable = matches!(
        fmt.layout(),
        edgefirst_tensor::PixelLayout::Packed | edgefirst_tensor::PixelLayout::Planar
    ) && !fmt.is_yuv();
    let pad_output = options.pad_output.filter(|_| letterboxed);
    if !mappable {
        if options.normalization != Normalization::default() || pad_output.is_some() {
            return Err(Error::NotSupported(format!(
                "normalization and output-domain padding do not apply to {fmt:?}"
            )));
        }
        return Ok(None);
    }
    let map = OutputMap::resolve(&options.normalization, dst.dtype(), fmt, dst.quantization())?;
    if map.is_default && pad_output.is_none() {
        return Ok(None);
    }
    Ok(Some((map, pad_output)))
}

/// How a destination element is produced from an 8-bit source sample `x`,
/// per destination channel `c`:
///
/// - float destinations: `x * scale[c] + offset[c]`;
/// - integer destinations: `clamp(floor(x * scale[c] + offset[c] + 0.5) +
///   zero_point[c], lo[c], hi[c])`, stored as its low byte.
///
/// Colour channels carry the normalization (and quantization). Alpha maps to
/// `x / 255` for float destinations and stays the raw byte for integer ones,
/// as a plain convert always has.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OutputMap {
    /// Per destination channel, in the destination's channel order.
    pub(crate) scale: [f32; 4],
    pub(crate) offset: [f32; 4],
    pub(crate) zero_point: [f32; 4],
    /// `Some([(lo, hi); 4])` for integer destinations.
    pub(crate) bounds: Option<[(f32, f32); 4]>,
    /// Integer destinations: the quantization scale per channel, which an
    /// output-domain pad is divided by.
    pub(crate) quant_scale: [f32; 4],
    /// The mapping a plain convert has always applied for this dtype.
    pub(crate) is_default: bool,
}

impl OutputMap {
    /// Resolve the mapping for a destination of `dtype` and `format`, with
    /// optional quantization. Integer destinations without quantization use
    /// scale 1/255 and zero point 0 (u8) or -128 (i8), which reproduces a
    /// plain u8 copy and the i8 `x ^ 0x80` bias.
    pub(crate) fn resolve(
        normalization: &Normalization,
        dtype: DType,
        format: PixelFormat,
        quantization: Option<&Quantization>,
    ) -> Result<Self> {
        let plain = *normalization == Normalization::default();
        let order = channel_order(format);
        let (lo, hi, implicit_zp) = match dtype {
            DType::F32 | DType::F16 => {
                let (scale, offset) = channel_affine(normalization, order, [1.0; 4]);
                return Ok(Self {
                    scale,
                    offset,
                    zero_point: [0.0; 4],
                    bounds: None,
                    quant_scale: [1.0; 4],
                    is_default: plain,
                });
            }
            DType::U8 => (0.0, 255.0, 0),
            DType::I8 => (-128.0, 127.0, -128),
            other => {
                return Err(Error::NotSupported(format!(
                    "convert does not write {other:?} destinations"
                )))
            }
        };
        let (qscale, qzp) = match quantization {
            None if plain => ([1.0 / 255.0; 4], [implicit_zp as f32; 4]),
            None => {
                return Err(Error::InvalidOptions(format!(
                    "a {dtype:?} destination needs quantization to apply a \
                     non-default normalization"
                )))
            }
            Some(q) => quant_per_channel(q, format, format.channels())?,
        };
        let (mut scale, mut offset) = channel_affine(normalization, order, qscale);
        let mut zero_point = qzp;
        let mut quant_scale = qscale;
        let mut bounds = [(lo, hi); 4];
        for (c, &src) in order.iter().enumerate() {
            if src == 3 {
                (scale[c], offset[c], zero_point[c], quant_scale[c]) = (1.0, 0.0, 0.0, 1.0);
                bounds[c] = (0.0, 255.0);
            }
        }
        let is_default = plain
            && qscale.iter().all(|s| *s == 1.0 / 255.0)
            && qzp.iter().all(|z| *z == implicit_zp as f32);
        Ok(Self {
            scale,
            offset,
            zero_point,
            bounds: Some(bounds),
            quant_scale,
            is_default,
        })
    }

    /// Map an 8-bit sample on channel `c`.
    pub(crate) fn apply(&self, c: usize, x: u8) -> f32 {
        let y = x as f32 * self.scale[c] + self.offset[c];
        match self.bounds {
            Some(b) => ((y + 0.5).floor() + self.zero_point[c]).clamp(b[c].0, b[c].1),
            None => y,
        }
    }

    /// The destination value for an output-domain value `v` on channel `c`.
    pub(crate) fn apply_output(&self, c: usize, v: f32) -> f32 {
        match self.bounds {
            Some(b) => {
                ((v / self.quant_scale[c] + 0.5).floor() + self.zero_point[c]).clamp(b[c].0, b[c].1)
            }
            None => v,
        }
    }

    /// The 8-bit sample that maps to exactly `target` on integer channel `c`,
    /// preferring the one whose unrounded value sits farthest from a rounding
    /// tie; `None` when no sample reaches it.
    #[cfg_attr(not(feature = "opengl"), allow(dead_code))]
    pub(crate) fn sample_for(&self, c: usize, target: f32) -> Option<u8> {
        self.bounds?;
        (0..=255u8)
            .filter(|&x| self.apply(c, x) == target)
            .min_by(|&a, &b| {
                let off = |x: u8| {
                    let y = x as f32 * self.scale[c] + self.offset[c];
                    ((y - y.floor()) - 0.5).abs()
                };
                off(b).total_cmp(&off(a))
            })
    }

    /// The destination values of the letterbox pad: `pad_output` as given,
    /// else the pixel `pad` mapped like the image.
    pub(crate) fn pad_values(&self, pad: [u8; 4], pad_output: Option<[f32; 4]>) -> [f32; 4] {
        std::array::from_fn(|c| match pad_output {
            Some(p) => self.apply_output(c, p[c]),
            None => self.apply(c, pad[c]),
        })
    }
}

/// The destination's colour channel order as indices into RGB, with `3` for
/// alpha.
fn channel_order(format: PixelFormat) -> [usize; 4] {
    match format {
        PixelFormat::Bgra => [2, 1, 0, 3],
        PixelFormat::Grey => [0, 3, 3, 3],
        _ => [0, 1, 2, 3],
    }
}

/// Fold normalization and the quantization scale into one per-channel affine
/// on 8-bit samples, in the destination's channel order. Alpha maps as
/// `x / 255` before quantization.
fn channel_affine(n: &Normalization, order: [usize; 4], qscale: [f32; 4]) -> ([f32; 4], [f32; 4]) {
    let mut scale = [0.0; 4];
    let mut offset = [0.0; 4];
    for (c, &src) in order.iter().enumerate() {
        let (mean, std) = if src == 3 {
            (0.0, 255.0)
        } else {
            (n.mean[src] as f64, n.std[src] as f64)
        };
        let qs = qscale[c] as f64;
        scale[c] = (1.0 / (std * qs)) as f32;
        offset[c] = (-mean / (std * qs)) as f32;
    }
    (scale, offset)
}

/// Per destination channel scale and zero point.
fn quant_per_channel(
    q: &Quantization,
    format: PixelFormat,
    channels: usize,
) -> Result<([f32; 4], [f32; 4])> {
    let per_tensor = |s: f32, z: i32| ([s; 4], [z as f32; 4]);
    let (s, z) = match q.mode() {
        QuantMode::PerTensorSymmetric { scale } => per_tensor(scale, 0),
        QuantMode::PerTensor { scale, zero_point } => per_tensor(scale, zero_point),
        QuantMode::PerChannelSymmetric { scales, .. } if scales.len() == channels => {
            let mut s = [scales[0]; 4];
            s[..channels].copy_from_slice(scales);
            (s, [0.0; 4])
        }
        QuantMode::PerChannel {
            scales,
            zero_points,
            ..
        } if scales.len() == channels => {
            let mut s = [scales[0]; 4];
            let mut z = [zero_points[0] as f32; 4];
            s[..channels].copy_from_slice(scales);
            for (d, zp) in z.iter_mut().zip(zero_points) {
                *d = *zp as f32;
            }
            (s, z)
        }
        _ => {
            return Err(Error::InvalidOptions(format!(
                "{format:?} destination quantization must be per-tensor or \
                 have one scale per channel ({channels})"
            )))
        }
    };
    if s.iter().any(|v| !v.is_finite() || *v <= 0.0) {
        return Err(Error::InvalidOptions(format!(
            "destination quantization scale {s:?} must be positive and finite"
        )));
    }
    Ok((s, z))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_float_map_is_divide_by_255() {
        let m = OutputMap::resolve(
            &Normalization::default(),
            DType::F32,
            PixelFormat::Rgb,
            None,
        )
        .unwrap();
        assert!(m.is_default);
        for x in [0u8, 1, 128, 255] {
            assert_eq!(m.apply(0, x), x as f32 / 255.0);
        }
    }

    #[test]
    fn implicit_int_quantization_reproduces_plain_converts() {
        let u = OutputMap::resolve(&Normalization::default(), DType::U8, PixelFormat::Rgb, None)
            .unwrap();
        let i = OutputMap::resolve(&Normalization::default(), DType::I8, PixelFormat::Rgb, None)
            .unwrap();
        assert!(u.is_default && i.is_default);
        for x in 0..=255u8 {
            assert_eq!(u.apply(1, x), x as f32);
            assert_eq!(i.apply(1, x) as i8 as u8, x ^ 0x80);
        }
    }

    #[test]
    fn imagenet_matches_reference_in_bgra_order() {
        let n = Normalization::imagenet();
        let m = OutputMap::resolve(&n, DType::F32, PixelFormat::Bgra, None).unwrap();
        assert!(!m.is_default);
        let want_b = (200.0 - 103.53) / 57.375;
        assert!((m.apply(0, 200) - want_b).abs() < 1e-5);
        assert!((m.apply(3, 255) - 1.0).abs() < 1e-6, "alpha stays unsigned");
    }

    #[test]
    fn int_normalization_needs_quantization() {
        let e = OutputMap::resolve(&Normalization::signed(), DType::I8, PixelFormat::Rgb, None);
        assert!(matches!(e, Err(Error::InvalidOptions(_))));
        let q = Quantization::per_tensor(1.0 / 128.0, 0);
        let m = OutputMap::resolve(
            &Normalization::signed(),
            DType::I8,
            PixelFormat::Rgb,
            Some(&q),
        )
        .unwrap();
        assert_eq!(m.apply(0, 0), -128.0);
        assert_eq!(m.apply(0, 255), 127.0, "128 clamps to the i8 range");
        assert_eq!(m.apply(0, 191), 64.0);
    }

    #[test]
    fn mean_std_broadcasts_and_validates() {
        assert_eq!(
            Normalization::mean_std(&[0.0], &[255.0]).unwrap(),
            Normalization::unsigned()
        );
        assert!(Normalization::mean_std(&[0.0, 0.0], &[1.0]).is_err());
        assert!(Normalization::mean_std(&[0.0], &[0.0]).is_err());
        assert!(Normalization::mean_std(&[f32::NAN], &[1.0]).is_err());
    }

    #[test]
    fn presets_parse_from_their_names() {
        for &p in NormalizationPreset::all() {
            assert_eq!(NormalizationPreset::from_str_code(p.as_str()), Some(p));
        }
        assert_eq!(
            LetterboxPlacement::from_str_code("top_left"),
            Some(LetterboxPlacement::TopLeft)
        );
    }

    #[cfg(feature = "decode")]
    #[test]
    fn segformer_preprocessing_maps_to_options() {
        let pre: edgefirst_decoder::schema::Preprocessing = serde_json::from_str(
            r#"{"normalization": "imagenet", "resize": "letterbox",
                "letterbox": {"placement": "top_left", "pad": [0.0], "pad_space": "normalized"}}"#,
        )
        .unwrap();
        let o = ConvertOptions::new().with_preprocessing(&pre).unwrap();
        assert_eq!(o.get_normalization(), Normalization::imagenet());
        assert_eq!(o.get_placement(), LetterboxPlacement::TopLeft);
        assert_eq!(o.get_pad_output(), Some([0.0, 0.0, 0.0, 1.0]));
        assert!(matches!(o.get_crop().fit, crate::Fit::Letterbox { .. }));

        let pre: edgefirst_decoder::schema::Preprocessing =
            serde_json::from_str(r#"{"resize": "letterbox"}"#).unwrap();
        let o = ConvertOptions::new().with_preprocessing(&pre).unwrap();
        assert_eq!(
            o.get_crop().fit,
            crate::Fit::Letterbox {
                pad: [114, 114, 114, 255]
            }
        );
        assert_eq!(o.get_normalization(), Normalization::unsigned());
    }

    #[test]
    fn an_output_pad_has_an_equivalent_sample_when_reachable() {
        let q = Quantization::per_tensor(0.0187, -14);
        let m = OutputMap::resolve(
            &Normalization::imagenet(),
            DType::I8,
            PixelFormat::Rgb,
            Some(&q),
        )
        .unwrap();
        let pad = m.pad_values([0; 4], Some([0.0; 4]));
        for (c, &want) in pad.iter().enumerate().take(3) {
            let x = m.sample_for(c, want).expect("0.0 is reachable");
            assert_eq!(m.apply(c, x), want);
        }
        // A coarse quantization skips values: 10.0 / 0.5 is not reachable at
        // scale 1/255 per step 0.5.
        let q = Quantization::per_tensor(1.0 / 512.0, 0);
        let m = OutputMap::resolve(
            &Normalization::unsigned(),
            DType::U8,
            PixelFormat::Rgb,
            Some(&q),
        )
        .unwrap();
        assert_eq!(m.sample_for(0, 1.0), None);
    }
}
