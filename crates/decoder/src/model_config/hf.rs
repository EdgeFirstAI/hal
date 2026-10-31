// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Hugging Face `config.json` and `preprocessor_config.json`.

use serde_json::{Map, Value};

use super::container::{trim_labels, usable_labels};
use super::error::invalid;
use super::ModelConfigError;
use edgefirst_tensor::{Interpolation, Normalization};

/// The outcome of checking a file that may be a Hugging Face config.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum HfCheck<T> {
    /// A Hugging Face config.
    Valid(T),
    /// Not a Hugging Face config, with the reason.
    NotHf(&'static str),
}

/// What `config.json` says about the model.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HfModel {
    pub model_type: String,
    pub architectures: Vec<String>,
    /// Class names in channel order, from `id2label` or `label2id`.
    pub labels: Option<Vec<String>>,
    pub num_labels: Option<usize>,
    /// Unusable class names, which are ignored.
    pub notes: Vec<String>,
}

/// What `preprocessor_config.json` says about the input.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct HfPreprocessing {
    pub processor_type: Option<String>,
    /// Fixed resize target `(width, height)`.
    pub size: Option<(usize, usize)>,
    pub normalization: Option<Normalization>,
    pub interpolation: Option<Interpolation>,
    /// Pads right and bottom (`do_pad`).
    pub pad: bool,
    pub reduce_labels: Option<bool>,
    /// Deviations from the reference to report.
    pub notes: Vec<String>,
}

/// The decoder an architecture maps to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HfDecoder {
    Semantic,
}

/// The decoder for a Hugging Face architecture class name.
pub(crate) fn decoder_for_architecture(arch: &str) -> Option<HfDecoder> {
    arch.ends_with("ForSemanticSegmentation")
        .then_some(HfDecoder::Semantic)
}

fn object(text: &str) -> Option<Map<String, Value>> {
    match serde_json::from_str(text) {
        Ok(Value::Object(m)) => Some(m),
        _ => None,
    }
}

fn string_array(v: &Value) -> Option<Vec<String>> {
    v.as_array()?
        .iter()
        .map(|s| s.as_str().map(str::to_owned))
        .collect()
}

/// Parses `config.json`; `file` names it in errors.
pub(crate) fn parse_hf_config(
    text: &str,
    file: &str,
) -> Result<HfCheck<HfModel>, ModelConfigError> {
    let Some(m) = object(text) else {
        return Ok(HfCheck::NotHf("not a JSON object"));
    };
    let Some(model_type) = m.get("model_type").and_then(Value::as_str) else {
        return Ok(HfCheck::NotHf("no string `model_type`"));
    };
    let architectures = m.get("architectures").and_then(string_array);
    let has_version = m.get("transformers_version").is_some_and(Value::is_string);
    let architectures = match (architectures, has_version) {
        (Some(a), _) => a,
        (None, true) if m.get("architectures").is_some_and(|v| !v.is_null()) => {
            return Err(invalid(file, "architectures", "is not an array of strings"));
        }
        (None, true) => Vec::new(),
        (None, false) => {
            return Ok(HfCheck::NotHf(
                "no `architectures` list or `transformers_version`",
            ))
        }
    };
    let mut notes = Vec::new();
    let labels = match (m.get("id2label"), m.get("label2id")) {
        (Some(v), _) if !v.is_null() => usable_labels(id2label(file, v), &mut notes),
        (_, Some(v)) if !v.is_null() => usable_labels(label2id(file, v), &mut notes),
        _ => None,
    };
    let num_labels = match m.get("num_labels") {
        None | Some(Value::Null) => None,
        Some(v) => Some(
            v.as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| invalid(file, "num_labels", "is not a non-negative integer"))?,
        ),
    };
    let labels = match (labels, num_labels) {
        (Some(l), Some(n)) if l.len() != n => usable_labels(
            Err(invalid(
                file,
                "num_labels",
                format!("is {n} but id2label has {} classes", l.len()),
            )),
            &mut notes,
        ),
        (labels, _) => labels,
    };
    Ok(HfCheck::Valid(HfModel {
        model_type: model_type.to_owned(),
        architectures,
        labels,
        num_labels,
        notes,
    }))
}

/// Class names indexed `0..n`, from pairs of (index, name), trimmed.
fn dense(
    file: &str,
    field: &str,
    pairs: impl Iterator<Item = Option<(usize, String)>>,
) -> Result<Vec<String>, ModelConfigError> {
    let mut slots: Vec<Option<String>> = Vec::new();
    for pair in pairs {
        let (i, name) = pair.ok_or_else(|| invalid(file, field, "maps a non-integer index"))?;
        if i >= 1 << 16 {
            return Err(invalid(file, field, format!("has index {i} beyond 65535")));
        }
        if slots.len() <= i {
            slots.resize(i + 1, None);
        }
        if slots[i].replace(name).is_some() {
            return Err(invalid(file, field, format!("repeats index {i}")));
        }
    }
    let names = slots
        .into_iter()
        .enumerate()
        .map(|(i, s)| s.ok_or_else(|| invalid(file, field, format!("has no class {i}"))))
        .collect::<Result<_, _>>()?;
    trim_labels(file, field, names)
}

fn id2label(file: &str, v: &Value) -> Result<Vec<String>, ModelConfigError> {
    let m = v
        .as_object()
        .ok_or_else(|| invalid(file, "id2label", "is not an object"))?;
    dense(
        file,
        "id2label",
        m.iter()
            .map(|(k, v)| Some((k.parse().ok()?, v.as_str()?.to_owned()))),
    )
}

fn label2id(file: &str, v: &Value) -> Result<Vec<String>, ModelConfigError> {
    let m = v
        .as_object()
        .ok_or_else(|| invalid(file, "label2id", "is not an object"))?;
    dense(
        file,
        "label2id",
        m.iter()
            .map(|(k, v)| Some((usize::try_from(v.as_u64()?).ok()?, k.clone()))),
    )
}

/// PIL resampling codes.
fn resample(
    file: &str,
    v: &Value,
    notes: &mut Vec<String>,
) -> Result<Interpolation, ModelConfigError> {
    let code = v
        .as_u64()
        .ok_or_else(|| invalid(file, "resample", "is not a PIL resampling code"))?;
    let (mode, name) = match code {
        0 => return Ok(Interpolation::Nearest),
        2 => return Ok(Interpolation::Bilinear),
        3 => return Ok(Interpolation::Bicubic),
        1 => (Interpolation::Bicubic, "lanczos"),
        4 => (Interpolation::Area, "box"),
        5 => (Interpolation::Bicubic, "hamming"),
        _ => {
            return Err(invalid(
                file,
                "resample",
                format!("is {code}; PIL codes are 0 to 5"),
            ))
        }
    };
    notes.push(format!(
        "{file}: resample={name} has no equivalent; using {}",
        mode.as_str()
    ));
    Ok(mode)
}

fn size(file: &str, v: &Value) -> Result<(usize, usize), ModelConfigError> {
    let dim = |v: &Value, field: &str| {
        v.as_u64()
            .filter(|&n| n > 0)
            .map(|n| n as usize)
            .ok_or_else(|| invalid(file, field, "is not a positive integer"))
    };
    if let Some(m) = v.as_object() {
        for field in ["shortest_edge", "longest_edge"] {
            if m.contains_key(field) {
                return Err(ModelConfigError::NotSupported {
                    field: format!("size.{field}"),
                    reason: "an aspect-preserving resize to a variable size needs a \
                             dynamic-shape model"
                        .into(),
                });
            }
        }
        let h = m
            .get("height")
            .ok_or_else(|| invalid(file, "size", "has no `height`"))?;
        let w = m
            .get("width")
            .ok_or_else(|| invalid(file, "size", "has no `width`"))?;
        return Ok((dim(w, "size.width")?, dim(h, "size.height")?));
    }
    let n = dim(v, "size")?;
    Ok((n, n))
}

fn mean_std(file: &str, field: &str, v: &Value) -> Result<Vec<f64>, ModelConfigError> {
    let values: Option<Vec<f64>> = match v {
        Value::Number(n) => n.as_f64().map(|x| vec![x]),
        Value::Array(a) => a.iter().map(Value::as_f64).collect(),
        _ => None,
    };
    match values {
        Some(v) if v.len() == 1 || v.len() == 3 => Ok(v),
        Some(v) => Err(invalid(
            file,
            field,
            format!("has {} values; expected 1 or 3", v.len()),
        )),
        None => Err(invalid(file, field, "is not a number or a list of numbers")),
    }
}

fn flag(m: &Map<String, Value>, key: &str, default: bool) -> bool {
    m.get(key).and_then(Value::as_bool).unwrap_or(default)
}

/// Parses `preprocessor_config.json`. Without a processor type it is
/// accepted only beside a validated `config.json` (`config_validated`) and
/// with normalization or size fields.
pub(crate) fn parse_hf_preprocessor(
    text: &str,
    file: &str,
    config_validated: bool,
) -> Result<HfCheck<HfPreprocessing>, ModelConfigError> {
    let Some(m) = object(text) else {
        return Ok(HfCheck::NotHf("not a JSON object"));
    };
    let processor_type = [
        "image_processor_type",
        "feature_extractor_type",
        "processor_class",
    ]
    .iter()
    .find_map(|k| m.get(*k).and_then(Value::as_str))
    .map(str::to_owned);
    let has_fields = ["image_mean", "image_std", "size"]
        .iter()
        .any(|k| m.contains_key(*k));
    if processor_type.is_none() && !(config_validated && has_fields) {
        return Ok(HfCheck::NotHf("no image processor type"));
    }
    let mut pre = HfPreprocessing {
        processor_type,
        ..Default::default()
    };
    if flag(&m, "keep_aspect_ratio", false) {
        return Err(ModelConfigError::NotSupported {
            field: "keep_aspect_ratio".into(),
            reason: "an aspect-preserving resize to a multiple of a size needs a \
                     dynamic-shape model"
                .into(),
        });
    }
    if flag(&m, "do_resize", true) {
        if let Some(v) = m.get("size").filter(|v| !v.is_null()) {
            pre.size = Some(size(file, v)?);
        }
    }
    if let Some(v) = m.get("resample").filter(|v| !v.is_null()) {
        pre.interpolation = Some(resample(file, v, &mut pre.notes)?);
    }
    let rescale = if flag(&m, "do_rescale", true) {
        match m.get("rescale_factor") {
            None | Some(Value::Null) => 1.0 / 255.0,
            Some(v) => v
                .as_f64()
                .filter(|f| f.is_finite() && *f > 0.0)
                .ok_or_else(|| invalid(file, "rescale_factor", "is not a positive number"))?,
        }
    } else {
        1.0
    };
    // HF computes (x * rescale - mean) / std; on 0-255 input that is
    // (x - mean / rescale) / (std / rescale).
    let (mean, std) = if flag(&m, "do_normalize", true) {
        match (m.get("image_mean"), m.get("image_std")) {
            (Some(mean), Some(std)) => (
                mean_std(file, "image_mean", mean)?,
                mean_std(file, "image_std", std)?,
            ),
            _ => (Vec::new(), Vec::new()),
        }
    } else {
        (vec![0.0], vec![1.0])
    };
    if !mean.is_empty() {
        let mean: Vec<f32> = mean.iter().map(|m| (m / rescale) as f32).collect();
        let std: Vec<f32> = std.iter().map(|s| (s / rescale) as f32).collect();
        pre.normalization = Some(
            Normalization::mean_std(&mean, &std)
                .map_err(|e| invalid(file, "image_std", e.to_string()))?,
        );
    }
    pre.pad = flag(&m, "do_pad", false);
    pre.reduce_labels = ["do_reduce_labels", "reduce_labels"]
        .iter()
        .find_map(|k| m.get(*k).and_then(Value::as_bool));
    if pre.size.is_some() && pre.interpolation != Some(Interpolation::Nearest) {
        pre.notes.push(format!(
            "{file}: the reference resizes with PIL, which antialiases when \
             downscaling; this resize does not"
        ));
    }
    Ok(HfCheck::Valid(pre))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(model: &str, file: &str) -> String {
        crate::model_config::read_testdata(&format!("hf/{model}/{file}"))
    }

    fn valid<T>(c: HfCheck<T>) -> T {
        match c {
            HfCheck::Valid(t) => t,
            HfCheck::NotHf(why) => panic!("not HF: {why}"),
        }
    }

    fn assert_close(a: [f32; 3], b: [f32; 3]) {
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-3, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn segformer_b0_config_and_preprocessor() {
        let m = valid(
            parse_hf_config(&fixture("segformer-512", "config.json"), "config.json").unwrap(),
        );
        assert_eq!(m.model_type, "segformer");
        assert_eq!(
            decoder_for_architecture(&m.architectures[0]),
            Some(HfDecoder::Semantic)
        );
        let labels = m.labels.unwrap();
        assert_eq!(labels.len(), 150);
        assert_eq!(labels[0], "class_0");

        let p = valid(
            parse_hf_preprocessor(
                &fixture("segformer-512", "preprocessor_config.json"),
                "preprocessor_config.json",
                true,
            )
            .unwrap(),
        );
        assert_eq!(p.size, Some((512, 512)));
        assert_eq!(p.interpolation, Some(Interpolation::Bilinear));
        assert_eq!(p.reduce_labels, Some(true));
        let n = p.normalization.unwrap();
        assert_close(n.mean(), Normalization::imagenet().mean());
        assert_close(n.std(), Normalization::imagenet().std());
        assert!(p.notes.iter().any(|n| n.contains("antialiases")));
    }

    #[test]
    fn upernet_size_dict_and_rescale() {
        let p = valid(
            parse_hf_preprocessor(
                &fixture("upernet-512", "preprocessor_config.json"),
                "preprocessor_config.json",
                false,
            )
            .unwrap(),
        );
        assert_eq!(p.processor_type.as_deref(), Some("SegformerImageProcessor"));
        assert_eq!(p.size, Some((512, 512)));
        assert_eq!(p.reduce_labels, Some(false));
        assert_close(
            p.normalization.unwrap().mean(),
            Normalization::imagenet().mean(),
        );
        let m =
            valid(parse_hf_config(&fixture("upernet-512", "config.json"), "config.json").unwrap());
        assert_eq!(m.architectures, ["UperNetForSemanticSegmentation"]);
    }

    #[test]
    fn dpt_half_mean_std() {
        let p = valid(
            parse_hf_preprocessor(
                &fixture("dpt-480", "preprocessor_config.json"),
                "preprocessor_config.json",
                true,
            )
            .unwrap(),
        );
        assert_eq!(p.size, Some((480, 480)));
        assert_eq!(p.normalization.unwrap(), Normalization::signed());
    }

    /// Older exports: a feature extractor, an integer `size`, `reduce_labels`,
    /// no `rescale_factor`, and a config with no `transformers_version`.
    #[test]
    fn legacy_feature_extractor_keys() {
        let p = valid(
            parse_hf_preprocessor(
                r#"{"feature_extractor_type": "SegformerFeatureExtractor",
                    "do_resize": true, "size": 512, "resample": 2,
                    "do_normalize": true, "reduce_labels": true,
                    "image_mean": [0.485, 0.456, 0.406],
                    "image_std": [0.229, 0.224, 0.225]}"#,
                "preprocessor_config.json",
                true,
            )
            .unwrap(),
        );
        assert_eq!(p.size, Some((512, 512)));
        assert_eq!(p.reduce_labels, Some(true));
        assert_eq!(p.interpolation, Some(Interpolation::Bilinear));
        assert_close(
            p.normalization.unwrap().mean(),
            Normalization::imagenet().mean(),
        );
        let m = valid(
            parse_hf_config(
                r#"{"architectures": ["UperNetForSemanticSegmentation"],
                    "model_type": "upernet", "transformers_version": null,
                    "id2label": {"0": "a", "1": "b"}}"#,
                "config.json",
            )
            .unwrap(),
        );
        assert_eq!(m.labels.unwrap(), ["a", "b"]);
    }

    #[test]
    fn legacy_segformer_fixture_trims_labels() {
        let text = fixture("segformer-legacy-512", "config.json");
        let labels = valid(parse_hf_config(&text, "config.json").unwrap())
            .labels
            .unwrap();
        assert_eq!(labels.len(), 150);
        assert_eq!(labels[7], "bed");
        assert_eq!(labels[33], "chest of drawers");

        let mut v: Value = serde_json::from_str(&text).unwrap();
        v.as_object_mut().unwrap().remove("id2label");
        let from_label2id = valid(parse_hf_config(&v.to_string(), "config.json").unwrap());
        assert_eq!(from_label2id.labels.unwrap(), labels);

        let p = valid(
            parse_hf_preprocessor(
                &fixture("segformer-legacy-512", "preprocessor_config.json"),
                "preprocessor_config.json",
                true,
            )
            .unwrap(),
        );
        assert_eq!(p.size, Some((512, 512)));
        assert_eq!(p.reduce_labels, Some(true));
    }

    #[test]
    fn untrimmed_label_collision_and_empty_name() {
        let base =
            r#""model_type": "segformer", "architectures": ["SegformerForSemanticSegmentation"]"#;
        let m = valid(
            parse_hf_config(
                &format!(r#"{{{base}, "id2label": {{"0": "a", "1": "  "}}}}"#),
                "config.json",
            )
            .unwrap(),
        );
        assert_eq!(m.labels, None);
        assert_eq!(
            m.notes,
            [
                "config.json: field `id2label[1]` is empty after trimming whitespace; its class \
              names are ignored"
            ]
        );
        let m = valid(
            parse_hf_config(
                &format!(r#"{{{base}, "label2id": {{"bed": 0, "bed ": 1}}}}"#),
                "config.json",
            )
            .unwrap(),
        );
        assert_eq!(m.labels.unwrap(), ["bed", "bed"]);
    }

    #[test]
    fn foreign_config_json_is_skipped() {
        for text in [
            r#"{"name": "x"}"#,
            r#"{"model_type": "x"}"#,
            r#"[1, 2]"#,
            "not json",
            r#"{"model_type": 3, "architectures": ["A"]}"#,
        ] {
            assert!(
                matches!(
                    parse_hf_config(text, "config.json").unwrap(),
                    HfCheck::NotHf(_)
                ),
                "{text}"
            );
        }
        let pre = r#"{"image_mean": [0.5], "image_std": [0.5]}"#;
        assert!(matches!(
            parse_hf_preprocessor(pre, "p", false).unwrap(),
            HfCheck::NotHf(_)
        ));
        assert!(matches!(
            parse_hf_preprocessor(pre, "p", true).unwrap(),
            HfCheck::Valid(_)
        ));
    }

    #[test]
    fn bad_id2label_is_noted_and_ignored() {
        let field_of = |text: &str| {
            let m = valid(parse_hf_config(text, "config.json").unwrap());
            assert_eq!(m.labels, None, "{text}");
            let [note] = m.notes.as_slice() else {
                panic!("{:?}", m.notes)
            };
            note.split('`').nth(1).unwrap_or_default().to_owned()
        };
        let base =
            r#""model_type": "segformer", "architectures": ["SegformerForSemanticSegmentation"]"#;
        assert_eq!(
            field_of(&format!(
                r#"{{{base}, "id2label": {{"0": "a", "2": "c"}}}}"#
            )),
            "id2label"
        );
        assert_eq!(
            field_of(&format!(r#"{{{base}, "id2label": {{"x": "a"}}}}"#)),
            "id2label"
        );
        assert_eq!(
            field_of(&format!(
                r#"{{{base}, "id2label": {{"0": "a"}}, "num_labels": 2}}"#
            )),
            "num_labels"
        );
        let m = valid(
            parse_hf_config(
                &format!(r#"{{{base}, "label2id": {{"b": 1, "a": 0}}}}"#),
                "c",
            )
            .unwrap(),
        );
        assert_eq!(m.labels.unwrap(), ["a", "b"]);
    }

    #[test]
    fn wrong_length_mean_names_field() {
        let e = parse_hf_preprocessor(
            r#"{"image_processor_type": "X", "image_mean": [0.1, 0.2], "image_std": [0.5]}"#,
            "preprocessor_config.json",
            true,
        )
        .unwrap_err();
        assert!(
            matches!(&e, ModelConfigError::InvalidField { field, .. } if field == "image_mean"),
            "{e}"
        );
    }

    #[test]
    fn variable_size_resizes_are_not_supported() {
        for (text, field) in [
            (
                r#"{"image_processor_type": "X", "size": {"shortest_edge": 224}}"#,
                "size.shortest_edge",
            ),
            (
                r#"{"image_processor_type": "X", "keep_aspect_ratio": true}"#,
                "keep_aspect_ratio",
            ),
        ] {
            match parse_hf_preprocessor(text, "p", true) {
                Err(ModelConfigError::NotSupported { field: f, .. }) => assert_eq!(f, field),
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn resample_lanczos_maps_with_a_note() {
        let p = valid(
            parse_hf_preprocessor(r#"{"image_processor_type": "X", "resample": 1}"#, "p", true)
                .unwrap(),
        );
        assert_eq!(p.interpolation, Some(Interpolation::Bicubic));
        assert!(p.notes.iter().any(|n| n.contains("lanczos")));
        assert!(parse_hf_preprocessor(
            r#"{"image_processor_type": "X", "resample": 9}"#,
            "p",
            true
        )
        .is_err());
    }

    #[test]
    fn rescale_and_normalize_flags() {
        let p = |t: &str| {
            valid(parse_hf_preprocessor(t, "p", true).unwrap())
                .normalization
                .unwrap()
        };
        assert_eq!(
            p(r#"{"image_processor_type": "X", "do_normalize": false}"#),
            Normalization::unsigned()
        );
        assert_eq!(
            p(r#"{"image_processor_type": "X", "do_normalize": false, "do_rescale": false}"#),
            Normalization::raw()
        );
    }

    #[test]
    fn detr_is_not_a_semantic_architecture() {
        assert_eq!(decoder_for_architecture("DetrForObjectDetection"), None);
    }
}
