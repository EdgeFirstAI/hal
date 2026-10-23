// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Convert options: geometry (rotation, flip, crop, letterbox placement and
//! pad), resampling, and the pixel value mapping (normalization, output
//! quantization).

use crate::{Crop, Error, Flip, Region, Result, Rotation};
use edgefirst_tensor::{
    ContentRegion, ContentRegionSource, DType, FitMode, Interpolation, LetterboxPlacement,
    Normalization, PadSpace, PixelFormat, PreprocessSpec, QuantMode, Quantization,
};

/// Options for [`convert`](crate::ImageProcessorTrait::convert). The default
/// converts the whole source, stretched, with no rotation or flip, resamples
/// with the processor's interpolation, and maps pixels as
/// [`Normalization::unsigned`] for float destinations.
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
    interpolation: Option<Interpolation>,
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

    /// Resample with `interpolation` when this convert resizes, instead of
    /// the processor's setting (see
    /// [`ImageProcessor::set_interpolation`](crate::ImageProcessor::set_interpolation)).
    /// [`Interpolation::Bicubic`] and [`Interpolation::Area`] run as
    /// [`Interpolation::supported`], with a warning logged once per mode.
    /// `EDGEFIRST_INTERPOLATION`, when set, takes precedence.
    pub fn interpolation(mut self, interpolation: Interpolation) -> Self {
        self.interpolation = Some(interpolation);
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

    /// The per-convert interpolation, or `None` for the processor's.
    pub fn get_interpolation(&self) -> Option<Interpolation> {
        self.interpolation
    }

    /// These options with the per-convert interpolation set to `mode`, or
    /// cleared by `None`.
    pub(crate) fn with_interpolation(mut self, mode: Option<Interpolation>) -> Self {
        self.interpolation = mode;
        self
    }
}

impl ConvertOptions {
    /// Options that apply a model's preprocessing: its normalization, fit,
    /// letterbox placement, pad and interpolation.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidOptions`] for a pixel-space pad outside 0–255 or a
    /// non-finite pad.
    ///
    /// ```
    /// use edgefirst_image::{ConvertOptions, LetterboxPlacement, PadSpace, PreprocessSpec};
    ///
    /// let mut spec = PreprocessSpec::new();
    /// spec.placement = LetterboxPlacement::TopLeft;
    /// spec.pad = [0.0; 3];
    /// spec.pad_space = PadSpace::Normalized;
    /// let opts = ConvertOptions::from_spec(&spec)?;
    /// assert_eq!(opts.get_placement(), LetterboxPlacement::TopLeft);
    /// assert_eq!(opts.get_pad_output(), Some([0.0, 0.0, 0.0, 1.0]));
    /// assert_eq!(opts.get_interpolation(), Some(spec.interpolation));
    /// # Ok::<(), edgefirst_image::Error>(())
    /// ```
    pub fn from_spec(spec: &PreprocessSpec) -> Result<Self> {
        Self::new().with_spec(spec)
    }

    /// These options with a model's preprocessing applied; rotation, flip
    /// and the source crop are kept. See [`from_spec`](Self::from_spec).
    ///
    /// # Errors
    ///
    /// As [`from_spec`](Self::from_spec).
    pub fn with_spec(mut self, spec: &PreprocessSpec) -> Result<Self> {
        if spec.pad.iter().any(|v| !v.is_finite()) {
            return Err(Error::InvalidOptions(format!(
                "letterbox pad {:?} must be finite",
                spec.pad
            )));
        }
        self.normalization = spec.normalization;
        self.interpolation = Some(spec.interpolation);
        self.pad_output = None;
        // Comparisons, not a match: an unknown fit stretches and an unknown pad space is pixel.
        self.crop.fit = if spec.fit != FitMode::Letterbox {
            crate::Fit::Stretch
        } else if spec.pad_space == PadSpace::Normalized {
            let [r, g, b] = spec.pad;
            self.pad_output = Some([r, g, b, 1.0]);
            crate::Fit::Letterbox {
                pad: [0, 0, 0, 255],
            }
        } else {
            if spec.pad.iter().any(|v| !(0.0..=255.0).contains(v)) {
                return Err(Error::InvalidOptions(format!(
                    "pixel letterbox pad {:?} must be within 0-255",
                    spec.pad
                )));
            }
            let [r, g, b] = spec.pad.map(|v| v.round() as u8);
            crate::Fit::Letterbox {
                pad: [r, g, b, 255],
            }
        };
        if spec.fit == FitMode::Letterbox {
            self.placement = spec.placement;
        }
        Ok(self)
    }

    /// Where `convert` puts the image content when converting an
    /// `src_w`×`src_h` source into a `dst_w`×`dst_h` destination with these
    /// options: the letterbox rectangle, or the whole destination for a
    /// stretch. It is computed by the same code the convert paints with, so
    /// mapping results back through it is exact.
    ///
    /// # Errors
    ///
    /// [`Error::CropInvalid`] when the source crop falls outside the source.
    ///
    /// ```
    /// use edgefirst_image::{ConvertOptions, Region};
    ///
    /// let opts = ConvertOptions::new().letterbox([114, 114, 114, 255]);
    /// let r = opts.content_region(1280, 720, 640, 640)?;
    /// assert_eq!(r.pixels, Region::new(0, 140, 640, 360));
    /// assert_eq!(r.normalized, [0.0, 0.21875, 1.0, 0.78125]);
    /// # Ok::<(), edgefirst_image::Error>(())
    /// ```
    pub fn content_region(
        &self,
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Result<ContentRegion> {
        let resolved = self
            .crop
            .resolve(src_w, src_h, dst_w, dst_h, self.placement)?;
        Ok(match resolved.dst_rect {
            Some(r) => {
                ContentRegion::new(Region::new(r.left, r.top, r.width, r.height), dst_w, dst_h)
            }
            None => ContentRegion::full(dst_w, dst_h),
        })
    }
}

impl ContentRegionSource for ConvertOptions {
    fn content_region(
        &self,
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> edgefirst_tensor::Result<ContentRegion> {
        ConvertOptions::content_region(self, src_w, src_h, dst_w, dst_h)
            .map_err(|e| edgefirst_tensor::Error::InvalidArgument(e.to_string()))
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
            (n.mean()[src] as f64, n.std()[src] as f64)
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
