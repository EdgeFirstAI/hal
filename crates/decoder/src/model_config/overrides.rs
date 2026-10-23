// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Caller overrides and their string grammar, shared by the Rust, Python and
//! C surfaces and by command lines.

use std::fmt;
use std::str::FromStr;

use edgefirst_tensor::{LetterboxPlacement, Normalization, NormalizationPreset};

use super::{ModelConfigError, ModelTask};
use crate::schema::DecodeMode;

fn bad(what: &str, value: &str, grammar: &str) -> ModelConfigError {
    ModelConfigError::Override(format!("{what} `{value}` is not one of {grammar}"))
}

const NORMALIZE_GRAMMAR: &str = "unsigned | signed | raw | imagenet | R,G,B/R,G,B";
const RESIZE_GRAMMAR: &str = "stretch | letterbox | letterbox-center | letterbox-topleft";
const TASK_GRAMMAR: &str = "detect | segment | semantic";
const DECODE_GRAMMAR: &str = "accurate | fast";

/// A `normalize` override: a preset name or `R,G,B/R,G,B`, a per-channel
/// mean and standard deviation in the 0–1 domain as Hugging Face and
/// torchvision write them.
///
/// ```
/// use edgefirst_decoder::model_config::NormalizeOverride;
/// use edgefirst_tensor::NormalizationPreset;
///
/// let n: NormalizeOverride = "0.485,0.456,0.406/0.229,0.224,0.225".parse().unwrap();
/// assert_eq!(n.to_string(), "0.485,0.456,0.406/0.229,0.224,0.225");
/// assert_eq!(
///     "unsigned".parse::<NormalizeOverride>().unwrap(),
///     NormalizeOverride::Preset(NormalizationPreset::Unsigned)
/// );
/// assert!("0,0,0/0,1,1".parse::<NormalizeOverride>().is_err());
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum NormalizeOverride {
    /// A named preset.
    Preset(NormalizationPreset),
    /// Explicit mean and standard deviation, 0–1 domain.
    MeanStd {
        /// Per-channel mean.
        mean: [f32; 3],
        /// Per-channel standard deviation.
        std: [f32; 3],
    },
}

impl NormalizeOverride {
    /// The normalization on 0–255 values.
    ///
    /// # Errors
    ///
    /// [`ModelConfigError::Override`] for a non-finite mean or a standard
    /// deviation that is not finite and positive.
    pub fn to_normalization(&self) -> Result<Normalization, ModelConfigError> {
        match self {
            Self::Preset(p) => Ok(Normalization::preset(*p)),
            Self::MeanStd { mean, std } => Normalization::unit_mean_std(mean, std, 255.0)
                .map_err(|e| ModelConfigError::Override(format!("normalize: {e}"))),
        }
    }
}

fn triple(s: &str) -> Option<[f32; 3]> {
    let mut it = s.split(',').map(|v| v.trim().parse::<f32>().ok());
    let t = [it.next()??, it.next()??, it.next()??];
    it.next().is_none().then_some(t)
}

impl FromStr for NormalizeOverride {
    type Err = ModelConfigError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        if let Some(p) = NormalizationPreset::from_str_code(s) {
            return Ok(Self::Preset(p));
        }
        let (mean, std) = s
            .split_once('/')
            .and_then(|(m, d)| Some((triple(m)?, triple(d)?)))
            .ok_or_else(|| bad("normalize", s, NORMALIZE_GRAMMAR))?;
        let n = Self::MeanStd { mean, std };
        n.to_normalization()?;
        Ok(n)
    }
}

impl fmt::Display for NormalizeOverride {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Preset(p) => f.write_str(p.as_str()),
            Self::MeanStd { mean, std } => write!(
                f,
                "{},{},{}/{},{},{}",
                mean[0], mean[1], mean[2], std[0], std[1], std[2]
            ),
        }
    }
}

/// A `resize` override: `stretch`, or `letterbox` with an optional
/// placement (`letterbox-center`, `letterbox-topleft`). A bare `letterbox`
/// follows the model family's convention; an explicit placement keeps the
/// family's pad.
///
/// ```
/// use edgefirst_decoder::model_config::ResizeOverride;
/// use edgefirst_tensor::LetterboxPlacement;
///
/// let r: ResizeOverride = "letterbox-topleft".parse().unwrap();
/// assert_eq!(r, ResizeOverride::Letterbox(Some(LetterboxPlacement::TopLeft)));
/// assert_eq!(ResizeOverride::Letterbox(None).to_string(), "letterbox");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResizeOverride {
    /// Scale each axis to the input size.
    Stretch,
    /// Letterbox, at the placement given or the family's.
    Letterbox(Option<LetterboxPlacement>),
}

impl FromStr for ResizeOverride {
    type Err = ModelConfigError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let s = s.trim();
        match s {
            "stretch" => Ok(Self::Stretch),
            "letterbox" => Ok(Self::Letterbox(None)),
            _ => s
                .strip_prefix("letterbox-")
                .and_then(LetterboxPlacement::from_str_code)
                .map(|p| Self::Letterbox(Some(p)))
                .ok_or_else(|| bad("resize", s, RESIZE_GRAMMAR)),
        }
    }
}

impl fmt::Display for ResizeOverride {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stretch => f.write_str("stretch"),
            Self::Letterbox(None) => f.write_str("letterbox"),
            Self::Letterbox(Some(p)) => write!(f, "letterbox-{}", p.as_str()),
        }
    }
}

impl FromStr for ModelTask {
    type Err = ModelConfigError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::from_str_code(s.trim()).ok_or_else(|| bad("task", s, TASK_GRAMMAR))
    }
}

impl fmt::Display for ModelTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

fn parse_labels(value: &str) -> Result<Vec<String>, ModelConfigError> {
    let labels: Vec<String> = value.split(',').map(|l| l.trim().to_owned()).collect();
    if labels.iter().any(String::is_empty) {
        return Err(ModelConfigError::Override(format!(
            "labels `{value}` has an empty name; expected comma-separated class names"
        )));
    }
    Ok(labels)
}

/// Values that replace what discovery found, applied after every source and
/// recorded as [`ConfigSource::Override`](super::ConfigSource::Override).
///
/// ```
/// use edgefirst_decoder::model_config::{ModelOverrides, ResizeOverride};
/// use edgefirst_tensor::LetterboxPlacement;
///
/// let mut o = ModelOverrides::default();
/// o.set("resize", "letterbox-topleft").unwrap();
/// o.set("labels", "background, person").unwrap();
/// assert_eq!(o.resize, Some(ResizeOverride::Letterbox(Some(LetterboxPlacement::TopLeft))));
/// assert_eq!(o.labels.as_deref(), Some(&["background".to_string(), "person".to_string()][..]));
/// assert!(o.set("resize", "fill").is_err());
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct ModelOverrides {
    /// What the outputs decode to.
    pub task: Option<ModelTask>,
    /// Input normalization.
    pub normalize: Option<NormalizeOverride>,
    /// Resize mode.
    pub resize: Option<ResizeOverride>,
    /// Semantic decode mode.
    pub decode: Option<DecodeMode>,
    /// Class names, in channel order.
    pub labels: Option<Vec<String>>,
}

impl ModelOverrides {
    /// No overrides.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether no override is set.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Sets the override named `key` (`task`, `normalize`, `resize`,
    /// `decode` or `labels`) from its string form. `labels` is a
    /// comma-separated list of non-empty names.
    ///
    /// # Errors
    ///
    /// [`ModelConfigError::Override`] for an unknown key or a value outside
    /// the key's grammar.
    pub fn set(&mut self, key: &str, value: &str) -> Result<(), ModelConfigError> {
        match key.trim() {
            "task" => self.task = Some(value.parse()?),
            "normalize" => self.normalize = Some(value.parse()?),
            "resize" => self.resize = Some(value.parse()?),
            "decode" => {
                let v = value.trim();
                self.decode = Some(
                    DecodeMode::from_str_code(v).ok_or_else(|| bad("decode", v, DECODE_GRAMMAR))?,
                )
            }
            "labels" => self.labels = Some(parse_labels(value)?),
            other => {
                return Err(ModelConfigError::Override(format!(
                    "unknown override `{other}`; expected task, normalize, resize, decode or labels"
                )))
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_grammar_round_trips() {
        for s in [
            "imagenet",
            "unsigned",
            "signed",
            "raw",
            "0.485,0.456,0.406/0.229,0.224,0.225",
            "0.5,0.5,0.5/0.5,0.5,0.5",
        ] {
            let n: NormalizeOverride = s.parse().unwrap();
            assert_eq!(n.to_string(), s);
        }
        for s in [
            "stretch",
            "letterbox",
            "letterbox-center",
            "letterbox-topleft",
        ] {
            assert_eq!(s.parse::<ResizeOverride>().unwrap().to_string(), s);
        }
        for s in ["detect", "segment", "semantic"] {
            assert_eq!(s.parse::<ModelTask>().unwrap().to_string(), s);
        }
        let mut o = ModelOverrides::new();
        for s in ["accurate", "fast"] {
            o.set("decode", s).unwrap();
            assert_eq!(o.decode.unwrap().as_str(), s);
        }
    }

    #[test]
    fn normalize_mean_std_is_scaled_to_255() {
        let n: NormalizeOverride = "0.5,0.5,0.5/0.5,0.5,0.5".parse().unwrap();
        assert_eq!(n.to_normalization().unwrap(), Normalization::signed());
    }

    #[test]
    fn bad_override_lists_accepted_forms() {
        let e = "fill".parse::<ResizeOverride>().unwrap_err().to_string();
        assert!(e.contains("letterbox-topleft"), "{e}");
        assert!("letterbox-top_left".parse::<ResizeOverride>().is_err());
        assert!("unit".parse::<NormalizeOverride>().is_err());
        let e = "0.1,0.2/0.3"
            .parse::<NormalizeOverride>()
            .unwrap_err()
            .to_string();
        assert!(e.contains("R,G,B/R,G,B"), "{e}");
        assert!("1,2,3,4/1,2,3".parse::<NormalizeOverride>().is_err());
        for bad_std in [
            "0,0,0/0,1,1",
            "0,0,0/-1,1,1",
            "nan,0,0/1,1,1",
            "0,0,0/inf,1,1",
        ] {
            assert!(bad_std.parse::<NormalizeOverride>().is_err(), "{bad_std}");
        }
        assert!("classify".parse::<ModelTask>().is_err());
        assert!(ModelOverrides::new().set("decode", "slow").is_err());
    }

    #[test]
    fn set_by_key() {
        let mut o = ModelOverrides::new();
        assert!(o.is_empty());
        o.set("task", "semantic").unwrap();
        o.set("normalize", "imagenet").unwrap();
        o.set("decode", " fast ").unwrap();
        o.set("labels", "a, b").unwrap();
        assert_eq!(o.task, Some(ModelTask::Semantic));
        assert_eq!(o.decode, Some(DecodeMode::Fast));
        assert_eq!(
            o.labels.as_deref(),
            Some(&["a".to_string(), "b".to_string()][..])
        );
        assert!(o.set("colour", "x").is_err());
        for empty in ["", "a,,b", "a, "] {
            assert!(o.set("labels", empty).is_err(), "{empty:?}");
        }
    }
}
