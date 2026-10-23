// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Where each resolved value came from, and the model family it implies.

use std::fmt;

edgefirst_tensor::ef_vocabulary! {
    /// The source that supplied a resolved value. The code is the one the C
    /// `EF_CONFIG_SOURCE_*` enumerators and the Python `ConfigSource` enum
    /// use.
    #[non_exhaustive]
    pub enum ConfigSource {
        /// `edgefirst.json` embedded in the model file.
        EmbeddedEdgefirst = 0, "embedded_edgefirst", EMBEDDED_EDGEFIRST,
        /// `edgefirst.json` (and `labels.txt`) beside the model file.
        SidecarEdgefirst = 1, "sidecar_edgefirst", SIDECAR_EDGEFIRST,
        /// Hugging Face `config.json` / `preprocessor_config.json` embedded
        /// in the model file.
        EmbeddedHf = 2, "embedded_hf", EMBEDDED_HF,
        /// Hugging Face `config.json` / `preprocessor_config.json` beside the
        /// model file.
        SidecarHf = 3, "sidecar_hf", SIDECAR_HF,
        /// Ultralytics export metadata, through
        /// [`infer_ultralytics_schema`](crate::infer_ultralytics_schema).
        UltralyticsMetadata = 4, "ultralytics_metadata", ULTRALYTICS_METADATA,
        /// Guessed from the model's input and output names and shapes.
        Heuristic = 5, "heuristic", HEURISTIC,
        /// The documented default.
        Default = 6, "default", DEFAULT,
        /// A caller override ([`ModelOverrides`](super::ModelOverrides)).
        Override = 7, "override", OVERRIDE,
    }
    #[doc(hidden)]
    pub mod config_source_code;
}

edgefirst_tensor::ef_vocabulary! {
    /// The family a model's configuration identifies. It selects the
    /// convention a bare `letterbox` override follows. The code is the one
    /// the C `EF_MODEL_FAMILY_*` enumerators and the Python `ModelFamily`
    /// enum use.
    #[non_exhaustive]
    pub enum ModelFamily {
        /// Described by an `edgefirst.json`.
        EdgeFirst = 0, "edgefirst", EDGEFIRST,
        /// An Ultralytics export, described by its metadata.
        Ultralytics = 1, "ultralytics", ULTRALYTICS,
        /// A Hugging Face export, described by its configs or recognised by
        /// its input and output names.
        HuggingFace = 2, "huggingface", HUGGINGFACE,
        /// Configured only by overrides.
        Unknown = 3, "unknown", UNKNOWN,
    }
    #[doc(hidden)]
    pub mod model_family_code;
}

edgefirst_tensor::ef_vocabulary! {
    /// What a model's outputs decode to. The code is the one the C
    /// `EF_MODEL_TASK_*` enumerators and the Python `ModelTask` enum use;
    /// the string is the `task` override grammar.
    #[non_exhaustive]
    pub enum ModelTask {
        /// Bounding boxes.
        Detect = 0, "detect", DETECT,
        /// Bounding boxes with instance masks.
        Segment = 1, "segment", SEGMENT,
        /// A per-pixel class map.
        Semantic = 2, "semantic", SEMANTIC,
    }
    #[doc(hidden)]
    pub mod model_task_code;
}

edgefirst_tensor::ef_vocabulary! {
    /// A resolved value whose source is recorded. The code is the one the C
    /// `EF_CONFIG_FIELD_*` enumerators and the Python `ConfigField` enum use;
    /// the string is the name [`ResolvedModelConfig::describe`] prints.
    ///
    /// [`ResolvedModelConfig::describe`]: super::ResolvedModelConfig::describe
    #[derive(PartialOrd, Ord)]
    #[non_exhaustive]
    pub enum ConfigField {
        /// Input normalization.
        Normalization = 0, "normalize", NORMALIZATION,
        /// Resize mode.
        Resize = 1, "resize", RESIZE,
        /// Letterbox placement.
        Placement = 2, "placement", PLACEMENT,
        /// Letterbox pad value and the space it is expressed in.
        Pad = 3, "pad", PAD,
        /// Resampling filter.
        Interpolation = 4, "interpolation", INTERPOLATION,
        /// Model task.
        Task = 5, "task", TASK,
        /// Semantic decode mode.
        DecodeMode = 6, "decode", DECODE_MODE,
        /// Class labels.
        Labels = 7, "labels", LABELS,
        /// Model input size.
        Size = 8, "size", SIZE,
    }
    #[doc(hidden)]
    pub mod config_field_code;
}

/// The source of one resolved value, with an optional detail naming the file
/// or the reason for a guess.
///
/// Its [`Display`](fmt::Display) form is the bracketed text of
/// [`ResolvedModelConfig::describe`](super::ResolvedModelConfig::describe):
///
/// ```
/// use edgefirst_decoder::model_config::{ConfigSource, Provenance};
///
/// let p = Provenance::new(ConfigSource::Heuristic, Some("SegFormer default"));
/// assert_eq!(p.to_string(), "guess: SegFormer default");
/// assert_eq!(Provenance::new(ConfigSource::Override, None).to_string(), "override");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Provenance {
    /// The source that supplied the value.
    pub source: ConfigSource,
    /// The file or field it was read from, or the reason for a guess.
    pub detail: Option<String>,
}

impl Provenance {
    /// A provenance record.
    pub fn new(source: ConfigSource, detail: Option<&str>) -> Self {
        Self {
            source,
            detail: detail.map(str::to_owned),
        }
    }
}

impl fmt::Display for Provenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let detail = self.detail.as_deref();
        match (self.source, detail) {
            (ConfigSource::Heuristic, Some(d)) => write!(f, "guess: {d}"),
            (ConfigSource::Heuristic, None) => f.write_str("guess"),
            (ConfigSource::Default, Some(d)) => write!(f, "default: {d}"),
            (ConfigSource::Default, None) => f.write_str("default"),
            (ConfigSource::Override, _) => f.write_str("override"),
            (ConfigSource::UltralyticsMetadata, _) => f.write_str("ultralytics metadata"),
            (_, Some(d)) => f.write_str(d),
            (ConfigSource::EmbeddedEdgefirst, None) => f.write_str("edgefirst.json"),
            (ConfigSource::SidecarEdgefirst, None) => f.write_str("sidecar edgefirst.json"),
            (ConfigSource::EmbeddedHf, None) => f.write_str("embedded hugging face config"),
            (ConfigSource::SidecarHf, None) => f.write_str("sidecar hugging face config"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabularies_round_trip_codes_and_strings() {
        for &s in ConfigSource::all() {
            assert_eq!(ConfigSource::from_code(s.code()), Some(s));
            assert_eq!(ConfigSource::from_str_code(s.as_str()), Some(s));
        }
        for &m in ModelFamily::all() {
            assert_eq!(ModelFamily::from_str_code(m.as_str()), Some(m));
        }
        for &t in ModelTask::all() {
            assert_eq!(ModelTask::from_str_code(t.as_str()), Some(t));
        }
        for (i, &c) in ConfigField::all().iter().enumerate() {
            assert_eq!(c.code() as usize, i, "fields describe in code order");
            assert_eq!(ConfigField::from_str_code(c.as_str()), Some(c));
        }
    }

    #[test]
    fn provenance_displays_bracket_text() {
        let p = |s, d| Provenance::new(s, d).to_string();
        assert_eq!(
            p(ConfigSource::SidecarHf, Some("preprocessor_config.json")),
            "preprocessor_config.json"
        );
        assert_eq!(
            p(ConfigSource::Heuristic, Some("SegFormer default")),
            "guess: SegFormer default"
        );
        assert_eq!(p(ConfigSource::Override, None), "override");
        assert_eq!(p(ConfigSource::Default, None), "default");
        assert_eq!(p(ConfigSource::EmbeddedEdgefirst, None), "edgefirst.json");
        assert_eq!(
            p(ConfigSource::UltralyticsMetadata, None),
            "ultralytics metadata"
        );
    }
}
