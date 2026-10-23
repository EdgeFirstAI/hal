// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Resolution of every source into one [`ResolvedModelConfig`].

use std::collections::BTreeMap;
use std::path::Path;

use edgefirst_tensor::{
    broadcast_rgb, FitMode, Interpolation, LetterboxPlacement, Normalization, NormalizationPreset,
    PadSpace, PreprocessSpec,
};
use serde_json::Value;

use super::container::{parse_labels, trim_labels, usable_labels};
use super::error::invalid;
use super::heuristics::{hf_family_name, hf_semantic_by_names, semantic_schema, SemanticExport};
use super::hf::{decoder_for_architecture, parse_hf_config, parse_hf_preprocessor, HfCheck};
use super::{
    ConfigField, ConfigSource, ModelConfigError, ModelFamily, ModelFiles, ModelOverrides,
    ModelTask, NormalizeOverride, Provenance, ResizeOverride,
};
use crate::configs::DimName;
use crate::infer::{infer_ultralytics_schema, InferError, InferredSchema, ModelSignals};
use crate::schema::{
    self, DecodeMode, DecoderKind, LogicalOutput, LogicalType, NormalizationConfig, SchemaV2,
};

/// Everything discovery reads: the files in the model container, the files
/// beside it, and the model's tensors as the runtime reports them.
///
/// ```
/// use edgefirst_decoder::model_config::{ModelFiles, ModelSources};
///
/// let sources = ModelSources::new().with_sidecar(ModelFiles::default());
/// assert!(sources.signals.is_none());
/// ```
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ModelSources {
    /// Files embedded in the model.
    pub embedded: ModelFiles,
    /// Files in the model's directory.
    pub sidecar: ModelFiles,
    /// Input and output tensors and runtime metadata. Needed to build a
    /// schema from a Hugging Face config, by Ultralytics inference and by
    /// the name heuristics.
    pub signals: Option<ModelSignals>,
}

impl ModelSources {
    /// No sources.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// With the files embedded in the model.
    #[must_use]
    pub fn with_embedded(mut self, files: ModelFiles) -> Self {
        self.embedded = files;
        self
    }

    /// With the files in the model's directory.
    #[must_use]
    pub fn with_sidecar(mut self, files: ModelFiles) -> Self {
        self.sidecar = files;
        self
    }

    /// With the model's tensors and runtime metadata.
    #[must_use]
    pub fn with_signals(mut self, signals: ModelSignals) -> Self {
        self.signals = Some(signals);
        self
    }
}

/// A model's configuration as resolved from every source.
///
/// [`preprocessing`](Self::preprocessing) and
/// [`decode_mode`](Self::decode_mode) are authoritative; `schema` carries a
/// copy of them for consumers that read the schema alone.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ResolvedModelConfig {
    /// The output schema, with `input.preprocessing` filled with the
    /// resolved values when the schema has an input.
    pub schema: SchemaV2,
    /// Class names in channel order, with outer whitespace trimmed; empty
    /// when no source names them.
    pub labels: Vec<String>,
    /// The family the configuration identifies.
    pub family: ModelFamily,
    /// What the outputs decode to.
    pub task: ModelTask,
    /// The semantic decode mode, for semantic models.
    pub decode_mode: Option<DecodeMode>,
    /// The input preprocessing of the reference pipeline. Its
    /// `interpolation` may be one the image crate runs as bilinear
    /// ([`Interpolation::supported`]); a note says so.
    pub preprocessing: PreprocessSpec,
    /// Where each value came from.
    pub provenance: BTreeMap<ConfigField, Provenance>,
    /// A one-line summary of the model.
    pub description: String,
    /// Deviations and skipped files worth logging.
    pub notes: Vec<String>,
    /// Hugging Face `do_reduce_labels`: the reference dataset drops its
    /// background class and shifts the rest down. Exposed, not applied.
    pub reduce_labels: Option<bool>,
}

impl ResolvedModelConfig {
    /// Where `field` came from.
    #[must_use]
    pub fn source(&self, field: ConfigField) -> Option<&Provenance> {
        self.provenance.get(&field)
    }

    /// One `key=value [source]` line per resolved value, for logging. Lines
    /// whose key is an override (`normalize`, `resize`, `task`, `decode`)
    /// print the value in the override grammar, e.g.
    /// `resize=letterbox-topleft [preprocessor_config.json]` or
    /// `normalize=imagenet [guess: SegFormer default]`.
    #[must_use]
    pub fn describe(&self) -> Vec<String> {
        let p = &self.preprocessing;
        let list = |v: &[f32; 3]| v.map(|x| x.to_string()).join(",");
        let value = |field: ConfigField| -> Option<String> {
            Some(match field {
                ConfigField::Normalization => normalize_override(&p.normalization).to_string(),
                ConfigField::Resize => resize_override(p).to_string(),
                ConfigField::Placement => p.placement.as_str().to_owned(),
                ConfigField::Pad => format!("{} {}", list(&p.pad), p.pad_space.as_str()),
                ConfigField::Interpolation => p.interpolation.as_str().to_owned(),
                ConfigField::Task => self.task.as_str().to_owned(),
                ConfigField::DecodeMode => self.decode_mode?.as_str().to_owned(),
                ConfigField::Labels => match self.labels.len() {
                    0 => "unknown".to_owned(),
                    n => format!("{n} classes"),
                },
                ConfigField::Size => {
                    let (w, h) = p.size?;
                    format!("{w}x{h}")
                }
            })
        };
        ConfigField::all()
            .iter()
            .filter_map(|&f| {
                let v = value(f)?;
                let prov = self
                    .provenance
                    .get(&f)
                    .map(|p| format!(" [{p}]"))
                    .unwrap_or_default();
                Some(format!("{}={v}{prov}", f.as_str()))
            })
            .collect()
    }
}

/// `n` as a `normalize` override: its preset, else mean/std on 0–1.
fn normalize_override(n: &Normalization) -> NormalizeOverride {
    match n.as_preset() {
        Some(p) => NormalizeOverride::Preset(p),
        None => {
            let unit = |v: [f32; 3]| v.map(|x| (f64::from(x) / 255.0) as f32);
            NormalizeOverride::MeanStd {
                mean: unit(n.mean()),
                std: unit(n.std()),
            }
        }
    }
}

/// `p`'s fit as a `resize` override, with its placement when letterboxed.
fn resize_override(p: &PreprocessSpec) -> ResizeOverride {
    match p.fit {
        FitMode::Letterbox => ResizeOverride::Letterbox(Some(p.placement)),
        _ => ResizeOverride::Stretch,
    }
}

/// A [`LogicalOutput`] with every field unset.
pub(crate) fn empty_logical() -> LogicalOutput {
    LogicalOutput {
        name: None,
        type_: None,
        shape: Vec::new(),
        dshape: Vec::new(),
        decoder: None,
        encoding: None,
        score_format: None,
        normalized: None,
        anchors: None,
        stride: None,
        dtype: None,
        quantization: None,
        outputs: Vec::new(),
        activation_applied: None,
        activation_required: None,
        semantic: Default::default(),
    }
}

/// The Ultralytics preprocessing, which is also the documented default:
/// unsigned, centred letterbox, pixel pad 114, bilinear.
fn ultralytics_convention() -> PreprocessSpec {
    PreprocessSpec::new()
}

/// The Hugging Face image processor defaults: ImageNet, stretch, bilinear,
/// and for a letterbox a top-left placement padded with normalized zeros.
fn hf_convention() -> PreprocessSpec {
    let mut s = PreprocessSpec::new();
    s.normalization = Normalization::imagenet();
    s.fit = FitMode::Stretch;
    s.placement = LetterboxPlacement::TopLeft;
    s.pad = [0.0; 3];
    s.pad_space = PadSpace::Normalized;
    s
}

type Slot<T> = Option<(T, Provenance)>;

fn fill<T>(slot: &mut Slot<T>, value: T, prov: &Provenance) {
    if slot.is_none() {
        *slot = Some((value, prov.clone()));
    }
}

fn take<T>(
    provenance: &mut BTreeMap<ConfigField, Provenance>,
    field: ConfigField,
    slot: Slot<T>,
    default: T,
) -> T {
    match slot {
        Some((v, p)) => {
            provenance.insert(field, p);
            v
        }
        None => default,
    }
}

/// Names that appear more than once ignoring case, as `"bed" at 7, 9`, or
/// `None` when every name is distinct.
fn repeated_labels(labels: &[String]) -> Option<String> {
    let mut seen: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, name) in labels.iter().enumerate() {
        seen.entry(name.to_lowercase()).or_default().push(i);
    }
    let repeats: Vec<String> = seen
        .values()
        .filter(|at| at.len() > 1)
        .map(|at| {
            let index: Vec<String> = at.iter().map(ToString::to_string).collect();
            format!("{:?} at {}", labels[at[0]], index.join(", "))
        })
        .collect();
    (!repeats.is_empty()).then(|| repeats.join("; "))
}

fn at(source: ConfigSource, detail: impl Into<String>) -> Provenance {
    Provenance::new(source, Some(&detail.into()))
}

#[derive(Default)]
struct State {
    schema: Option<(SchemaV2, Provenance)>,
    family: Option<ModelFamily>,
    /// Class-name candidates, highest priority first.
    labels: Vec<(Vec<String>, Provenance)>,
    normalization: Slot<Normalization>,
    fit: Slot<FitMode>,
    placement: Slot<LetterboxPlacement>,
    pad: Slot<([f32; 3], PadSpace)>,
    interpolation: Slot<Interpolation>,
    size: Option<(usize, usize)>,
    num_classes: Option<usize>,
    hf_stride: Option<usize>,
    declared_placement: bool,
    reduce_labels: Option<bool>,
    description: Option<String>,
    infer_error: Option<InferError>,
    notes: Vec<String>,
}

impl State {
    fn set_schema(&mut self, schema: SchemaV2, prov: Provenance, family: ModelFamily) {
        if self.schema.is_none() {
            self.schema = Some((schema, prov));
            self.family = Some(family);
        }
    }

    /// Takes a schema laid out from the model's tensors.
    fn adopt_export(
        &mut self,
        export: SemanticExport,
        prov: Provenance,
        family: ModelFamily,
        description: String,
    ) {
        if self.schema.is_none() {
            self.description = Some(description);
            self.size = Some(export.size);
            self.num_classes = Some(export.num_classes);
            self.hf_stride = Some(export.stride);
            self.set_schema(export.schema, prov, family);
        }
    }

    /// Fills the preprocessing slots that `fields` names from `spec`.
    fn fill_spec(&mut self, spec: &PreprocessSpec, prov: &Provenance, fields: &[ConfigField]) {
        for field in fields {
            match field {
                ConfigField::Normalization => {
                    fill(&mut self.normalization, spec.normalization, prov)
                }
                ConfigField::Resize => fill(&mut self.fit, spec.fit, prov),
                ConfigField::Placement => fill(&mut self.placement, spec.placement, prov),
                ConfigField::Pad => fill(&mut self.pad, (spec.pad, spec.pad_space), prov),
                ConfigField::Interpolation => {
                    fill(&mut self.interpolation, spec.interpolation, prov)
                }
                _ => {}
            }
        }
    }

    fn apply_schema_preprocessing(
        &mut self,
        pre: &schema::Preprocessing,
        file: &str,
        prov: &Provenance,
    ) -> Result<(), ModelConfigError> {
        if let Some(n) = &pre.normalization {
            let (mean, std) = n.mean_std(3)?;
            let n = Normalization::mean_std(&mean, &std)
                .map_err(|e| invalid(file, "input.preprocessing.normalization", e.to_string()))?;
            fill(&mut self.normalization, n, prov);
        }
        let fit = pre
            .resize
            .or_else(|| pre.letterbox.as_ref().map(|_| FitMode::Letterbox));
        if let Some(fit) = fit {
            fill(&mut self.fit, fit, prov);
        }
        if let Some(lb) = &pre.letterbox {
            if let Some(p) = lb.placement {
                if self.placement.is_none() {
                    self.declared_placement = true;
                }
                fill(&mut self.placement, p, prov);
            }
            if let Some(pad) = &lb.pad {
                let pad = broadcast_rgb(pad)
                    .ok_or_else(|| invalid(file, "letterbox.pad", "needs 1 or 3 values"))?;
                fill(&mut self.pad, (pad, lb.pad_space.unwrap_or_default()), prov);
            }
        }
        if let Some(i) = pre.interpolation {
            fill(&mut self.interpolation, i, prov);
        }
        Ok(())
    }

    /// Steps 1 and 2: `edgefirst.json` and `labels.txt`.
    fn edgefirst(
        &mut self,
        files: &ModelFiles,
        source: ConfigSource,
    ) -> Result<(), ModelConfigError> {
        if let Some(f) = &files.edgefirst_json {
            let here = place(source, &f.name);
            let prov = at(source, here.as_str());
            let value: Value = serde_json::from_str(&f.text)
                .map_err(|e| invalid(&f.name, "(document)", e.to_string()))?;
            let validation = value.get("validation");
            let legacy_normalization = validation
                .and_then(|v| v.get("normalization"))
                .and_then(Value::as_str)
                .map(str::to_owned);
            let legacy_resize = validation
                .and_then(|v| v.get("preprocessing"))
                .and_then(Value::as_str)
                .map(str::to_owned);
            let classes: Option<Vec<String>> = value
                .pointer("/dataset/classes")
                .and_then(Value::as_array)
                .and_then(|a| a.iter().map(|c| c.as_str().map(str::to_owned)).collect());
            let classes = classes.and_then(|c| {
                usable_labels(trim_labels(&f.name, "dataset.classes", c), &mut self.notes)
            });
            let name = value
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_owned();
            let schema = SchemaV2::from_json_value(value)?;
            if let Some(pre) = schema.input.as_ref().and_then(|i| i.preprocessing.as_ref()) {
                self.apply_schema_preprocessing(pre, &f.name, &prov)?;
            }
            if let Some(n) = legacy_normalization {
                let preset = NormalizationPreset::from_str_code(&n).ok_or_else(|| {
                    invalid(
                        &f.name,
                        "validation.normalization",
                        format!("`{n}` is not a preset"),
                    )
                })?;
                let p = at(source, format!("{here} validation.normalization"));
                fill(&mut self.normalization, Normalization::preset(preset), &p);
            }
            if let Some(r) = legacy_resize {
                let p = at(source, format!("{here} validation.preprocessing"));
                match r.as_str() {
                    "letterbox" => fill(&mut self.fit, FitMode::Letterbox, &p),
                    "resize" | "stretch" => fill(&mut self.fit, FitMode::Stretch, &p),
                    other => self.notes.push(format!(
                        "{}: validation.preprocessing `{other}` ignored",
                        f.name
                    )),
                }
            }
            if let Some(classes) = classes.filter(|c| !c.is_empty()) {
                self.labels.push((classes, prov.clone()));
            }
            if schema.outputs.is_empty() {
                self.notes.push(format!(
                    "{here} declares no outputs; the schema comes from a later source"
                ));
            } else if self.schema.is_none() {
                self.description = Some(format!("EdgeFirst model {name}").trim_end().to_owned());
                self.set_schema(schema, prov, ModelFamily::EdgeFirst);
            }
        }
        if let Some(f) = &files.labels_txt {
            let labels = parse_labels(&f.text);
            if !labels.is_empty() {
                self.labels
                    .push((labels, at(source, place(source, &f.name))));
            }
        }
        Ok(())
    }

    /// Steps 3 and 4: Hugging Face `config.json` and `preprocessor_config.json`.
    fn hugging_face(
        &mut self,
        files: &ModelFiles,
        source: ConfigSource,
        signals: Option<&ModelSignals>,
    ) -> Result<(), ModelConfigError> {
        let mut model = None;
        if let Some(f) = &files.hf_config {
            match parse_hf_config(&f.text, &f.name)? {
                HfCheck::Valid(m) => model = Some((m, at(source, place(source, &f.name)))),
                HfCheck::NotHf(why) => self.skipped(source, &f.name, "Hugging Face config", why),
            }
        }
        let mut pre = None;
        if let Some(f) = &files.hf_preprocessor {
            match parse_hf_preprocessor(&f.text, &f.name, model.is_some())? {
                HfCheck::Valid(p) => pre = Some((p, at(source, place(source, &f.name)))),
                HfCheck::NotHf(why) => {
                    self.skipped(source, &f.name, "Hugging Face image processor config", why)
                }
            }
        }
        if let Some((mut m, prov)) = model {
            self.notes.append(&mut m.notes);
            let file = files
                .hf_config
                .as_ref()
                .map_or("config.json", |f| f.name.as_str());
            if self.schema.is_none() {
                let Some(arch) = m
                    .architectures
                    .iter()
                    .find(|a| decoder_for_architecture(a).is_some())
                else {
                    return Err(ModelConfigError::UnsupportedArchitecture(m.model_type));
                };
                let signals = signals.ok_or_else(|| ModelConfigError::Unresolved {
                    missing: "the model's input and output tensors, to lay out the outputs a \
                              Hugging Face config describes"
                        .into(),
                    overrides: "the model signals".into(),
                })?;
                let export = semantic_schema(signals)?;
                let classes = export.num_classes;
                if let Some(n) = m.num_labels.filter(|&n| m.labels.is_none() && n != classes) {
                    let note = format!("{file}: num_labels is {n} but the output has {classes}");
                    log::warn!("{note}");
                    self.notes.push(note);
                }
                let description = format!(
                    "Hugging Face {} semantic segmentation ({arch}), {classes} classes",
                    m.model_type
                );
                self.adopt_export(export, prov.clone(), ModelFamily::HuggingFace, description);
            }
            if let Some(labels) = m.labels {
                self.labels.push((labels, prov));
            }
        }
        if let Some((p, prov)) = pre {
            if let Some(n) = p.normalization {
                fill(&mut self.normalization, n, &prov);
            }
            if let Some(i) = p.interpolation {
                fill(&mut self.interpolation, i, &prov);
            }
            if p.pad {
                self.fill_spec(
                    &hf_convention(),
                    &prov,
                    &[ConfigField::Placement, ConfigField::Pad],
                );
                fill(&mut self.fit, FitMode::Letterbox, &prov);
            } else if p.size.is_some() {
                fill(&mut self.fit, FitMode::Stretch, &prov);
            }
            if let (Some(want), Some(have)) = (p.size, self.size) {
                if want != have {
                    self.notes.push(format!(
                        "{prov}: the reference resizes to {}x{}; the model input is {}x{}",
                        want.0, want.1, have.0, have.1
                    ));
                }
            }
            self.reduce_labels = self.reduce_labels.or(p.reduce_labels);
            self.notes.extend(p.notes);
        }
        Ok(())
    }

    fn skipped(&mut self, source: ConfigSource, name: &str, what: &str, why: &str) {
        let note = format!("{} present but not a {what} ({why})", place(source, name));
        log::debug!("{note}");
        self.notes.push(note);
    }

    /// Step 5: Ultralytics metadata.
    fn ultralytics(&mut self, signals: &ModelSignals) {
        match self.infer_ultralytics(signals) {
            Ok(inferred) => {
                let prov = Provenance::new(ConfigSource::UltralyticsMetadata, None);
                self.fill_ultralytics_labels(inferred.labels, &prov);
                self.description = Some(inferred.description);
                self.set_schema(inferred.schema, prov, ModelFamily::Ultralytics);
                let conv = Provenance::new(ConfigSource::Default, Some("Ultralytics convention"));
                self.fill_spec(&ultralytics_convention(), &conv, PREPROCESSING);
            }
            Err(InferError::NotUltralytics(why)) => log::debug!("not Ultralytics: {why}"),
            Err(e) => self.infer_error = Some(e),
        }
    }

    /// Ultralytics class names for a schema another source supplied.
    fn ultralytics_labels(&mut self, signals: &ModelSignals) {
        if let Ok(inferred) = self.infer_ultralytics(signals) {
            let prov = Provenance::new(ConfigSource::UltralyticsMetadata, None);
            self.fill_ultralytics_labels(inferred.labels, &prov);
        }
    }

    /// Infers the Ultralytics schema, laying it out from the output's class
    /// count and ignoring `names` when the two disagree.
    fn infer_ultralytics(&mut self, signals: &ModelSignals) -> Result<InferredSchema, InferError> {
        match infer_ultralytics_schema(signals) {
            Err(InferError::ClassCountMismatch { expected, found }) if found > 0 => {
                let note = format!(
                    "Ultralytics metadata: names lists {expected} classes but the output has \
                     {found}; its class names are ignored"
                );
                log::warn!("{note}");
                self.notes.push(note);
                let names: Vec<String> = (0..found).map(|i| format!("{i}: 'class{i}'")).collect();
                let mut signals = signals.clone();
                signals
                    .metadata
                    .insert("names".into(), format!("{{{}}}", names.join(", ")));
                infer_ultralytics_schema(&signals).map(|mut inferred| {
                    inferred.labels.clear();
                    inferred
                })
            }
            r => r,
        }
    }

    fn fill_ultralytics_labels(&mut self, labels: Vec<String>, prov: &Provenance) {
        if labels.is_empty() {
            return;
        }
        let trimmed = trim_labels("Ultralytics metadata", "names", labels);
        if let Some(labels) = usable_labels(trimmed, &mut self.notes) {
            self.labels.push((labels, prov.clone()));
        }
    }

    /// Step 6: a Hugging Face export recognised by its tensor names.
    fn heuristics(&mut self, signals: &ModelSignals) {
        if let Some(export) = hf_semantic_by_names(signals) {
            let prov = Provenance::new(ConfigSource::Heuristic, Some("pixel_values/logits"));
            let description = format!(
                "Hugging Face semantic segmentation export (recognised by tensor names), {} \
                 classes",
                export.num_classes
            );
            self.adopt_export(export, prov, ModelFamily::HuggingFace, description);
        }
    }

    /// The defaults of a Hugging Face family's image processor.
    fn hf_family_defaults(&mut self) {
        let detail = hf_family_name(self.hf_stride.unwrap_or(0));
        let guess = Provenance::new(ConfigSource::Heuristic, Some(detail));
        self.fill_spec(
            &hf_convention(),
            &guess,
            &[
                ConfigField::Normalization,
                ConfigField::Resize,
                ConfigField::Interpolation,
            ],
        );
    }

    /// Step 7: the documented defaults.
    fn defaults(&mut self) {
        let d = Provenance::new(ConfigSource::Default, None);
        self.fill_spec(&PreprocessSpec::new(), &d, PREPROCESSING);
    }
}

/// Every preprocessing slot.
const PREPROCESSING: &[ConfigField] = &[
    ConfigField::Normalization,
    ConfigField::Resize,
    ConfigField::Placement,
    ConfigField::Pad,
    ConfigField::Interpolation,
];

/// Where a file was found, as provenance detail: its name beside the model,
/// `embedded <name>` inside it.
fn place(source: ConfigSource, name: &str) -> String {
    match source {
        ConfigSource::EmbeddedEdgefirst | ConfigSource::EmbeddedHf => format!("embedded {name}"),
        _ => name.to_owned(),
    }
}

/// The `(width, height)` of a schema's input, when its layout resolves.
fn input_size(schema: &SchemaV2) -> Option<(usize, usize)> {
    let input = schema.input.as_ref()?;
    if !input.dshape.is_empty() {
        let dim = |n| {
            input
                .dshape
                .iter()
                .position(|(d, _)| *d == n)
                .and_then(|i| input.shape.get(i).copied())
        };
        return Some((dim(DimName::Width)?, dim(DimName::Height)?));
    }
    match (input.shape.as_slice(), input.channels()?) {
        ([_, c, h, w], ch) if *c == ch => Some((*w, *h)),
        ([_, h, w, _], _) => Some((*w, *h)),
        _ => None,
    }
}

fn task_of(schema: &SchemaV2) -> ModelTask {
    let has = |t: LogicalType| schema.outputs.iter().any(|o| o.type_ == Some(t));
    let semantic = has(LogicalType::Segmentation) || has(LogicalType::Masks);
    let boxes =
        has(LogicalType::Boxes) || has(LogicalType::Detection) || has(LogicalType::Detections);
    if semantic && !boxes {
        ModelTask::Semantic
    } else if has(LogicalType::Protos) || has(LogicalType::MaskCoefs) {
        ModelTask::Segment
    } else {
        ModelTask::Detect
    }
}

fn semantic_outputs(schema: &mut SchemaV2) -> impl Iterator<Item = &mut LogicalOutput> {
    schema.outputs.iter_mut().filter(|o| {
        matches!(
            o.type_,
            Some(LogicalType::Segmentation | LogicalType::Masks)
        )
    })
}

/// The class count the outputs state or imply; `None` for layouts that carry
/// none, such as end-to-end detections and class maps.
fn class_count(schema: &SchemaV2) -> Option<usize> {
    let dim = |dshape: &[(DimName, usize)], name: DimName| {
        dshape.iter().find(|(d, _)| *d == name).map(|(_, n)| *n)
    };
    let protos = schema
        .outputs
        .iter()
        .filter(|o| o.type_ == Some(LogicalType::Protos))
        .find_map(|o| dim(&o.dshape, DimName::NumProtos))
        .unwrap_or(0);
    schema.outputs.iter().find_map(|o| {
        let dshape = match (o.dshape.as_slice(), o.outputs.first()) {
            ([], Some(child)) => child.dshape.as_slice(),
            (d, _) => d,
        };
        let classes = dim(dshape, DimName::NumClasses);
        match o.type_? {
            LogicalType::Segmentation | LogicalType::Scores => classes,
            LogicalType::Detection => classes.or_else(|| match o.decoder? {
                DecoderKind::Ultralytics => dim(dshape, DimName::NumFeatures)?
                    .checked_sub(4 + protos)
                    .filter(|&n| n > 0),
                DecoderKind::ModelPack => {
                    let anchors = o.anchors.as_ref()?.len();
                    let features = dim(dshape, DimName::NumAnchorsXFeatures)?;
                    (anchors > 0 && features % anchors == 0)
                        .then(|| features / anchors)?
                        .checked_sub(5)
                        .filter(|&n| n > 0)
                }
                DecoderKind::Semantic => None,
            }),
            _ => None,
        }
    })
}

/// The letterbox `(placement, pad)` a bare `letterbox` override means for
/// a family.
fn family_letterbox(
    family: ModelFamily,
    declared: bool,
    current: (LetterboxPlacement, ([f32; 3], PadSpace)),
) -> (LetterboxPlacement, ([f32; 3], PadSpace), &'static str) {
    let of = |s: PreprocessSpec| (s.placement, (s.pad, s.pad_space));
    let (placement, pad, why) = match family {
        ModelFamily::Ultralytics => {
            let (p, d) = of(ultralytics_convention());
            (p, d, "Ultralytics convention")
        }
        ModelFamily::HuggingFace => {
            let (p, d) = of(hf_convention());
            (p, d, "Hugging Face convention")
        }
        ModelFamily::EdgeFirst if declared => (current.0, current.1, "as edgefirst.json declares"),
        ModelFamily::EdgeFirst => (
            LetterboxPlacement::Center,
            current.1,
            "edgefirst.json declares no placement",
        ),
        _ => {
            let (p, d) = of(ultralytics_convention());
            (p, d, "convention unknown")
        }
    };
    (placement, pad, why)
}

/// Resolves a configuration from already-read sources.
///
/// When the signals carry no metadata, the embedded files' metadata stands
/// in for it, so Ultralytics inference sees the ONNX props or the TFLite
/// `metadata.json` the container holds.
///
/// # Errors
///
/// Any [`ModelConfigError`]: a malformed recognised file, an unsupported
/// field or architecture, an override that cannot apply, or nothing to
/// build a schema from ([`ModelConfigError::Unresolved`]).
pub fn resolve_sources(
    sources: ModelSources,
    overrides: &ModelOverrides,
) -> Result<ResolvedModelConfig, ModelConfigError> {
    let ModelSources {
        embedded,
        sidecar,
        signals,
    } = sources;
    let signals = signals.map(|mut s| {
        if s.metadata.is_empty() {
            s.metadata = embedded.metadata.clone();
        }
        s
    });
    let mut st = State::default();
    st.edgefirst(&embedded, ConfigSource::EmbeddedEdgefirst)?;
    st.edgefirst(&sidecar, ConfigSource::SidecarEdgefirst)?;
    st.hugging_face(&embedded, ConfigSource::EmbeddedHf, signals.as_ref())?;
    st.hugging_face(&sidecar, ConfigSource::SidecarHf, signals.as_ref())?;
    if let Some(s) = &signals {
        if st.schema.is_none() {
            st.ultralytics(s);
        } else {
            st.ultralytics_labels(s);
        }
        if st.schema.is_none() {
            st.heuristics(s);
        }
    }
    let ov = Provenance::new(ConfigSource::Override, None);
    let mut task_overridden = false;
    if st.schema.is_none() && overrides.task == Some(ModelTask::Semantic) {
        if let Some(s) = &signals {
            let export = semantic_schema(s)?;
            let description = format!(
                "semantic segmentation (task override), {} classes",
                export.num_classes
            );
            st.adopt_export(export, ov.clone(), ModelFamily::Unknown, description);
            task_overridden = true;
        }
    }
    let Some((mut schema, schema_prov)) = st.schema.take() else {
        if let Some(e) = st.infer_error {
            return Err(ModelConfigError::Infer(e));
        }
        return Err(ModelConfigError::Unresolved {
            missing: "a model configuration: no edgefirst.json, Hugging Face config or \
                      Ultralytics metadata was found, and the input and output tensors do not \
                      match a Hugging Face segmentation export"
                .into(),
            overrides: "an edgefirst.json beside the model, or task=semantic with the model \
                        signals"
                .into(),
        });
    };
    let family = st.family.unwrap_or(ModelFamily::Unknown);
    if family == ModelFamily::HuggingFace {
        st.hf_family_defaults();
    }
    st.defaults();
    let size = st.size.or_else(|| input_size(&schema));
    let classes = st.num_classes.or_else(|| class_count(&schema));

    let d = PreprocessSpec::new();
    let mut provenance = BTreeMap::new();
    let p = &mut provenance;
    let mut normalization = take(
        p,
        ConfigField::Normalization,
        st.normalization,
        d.normalization,
    );
    let mut fit = take(p, ConfigField::Resize, st.fit, d.fit);
    let mut placement = take(p, ConfigField::Placement, st.placement, d.placement);
    let mut pad = take(p, ConfigField::Pad, st.pad, (d.pad, d.pad_space));
    let interpolation = take(
        p,
        ConfigField::Interpolation,
        st.interpolation,
        d.interpolation,
    );
    let mut labels = Vec::new();
    let mut notes = st.notes;
    if size.is_some() {
        let detail = Some("model input");
        provenance.insert(
            ConfigField::Size,
            Provenance::new(schema_prov.source, detail),
        );
    }

    let mut task = task_of(&schema);
    if let Some(want) = overrides.task {
        if want != task && !(task == ModelTask::Segment && want == ModelTask::Detect) {
            return Err(ModelConfigError::Override(format!(
                "task={want} does not apply to a {task} model"
            )));
        }
        task = want;
    }
    let task_prov = if task_overridden || overrides.task.is_some() {
        ov.clone()
    } else {
        schema_prov.clone()
    };
    provenance.insert(ConfigField::Task, task_prov);
    if let Some(n) = overrides.normalize {
        normalization = n.to_normalization()?;
        provenance.insert(ConfigField::Normalization, ov.clone());
    }
    if let Some(r) = overrides.resize {
        match r {
            ResizeOverride::Letterbox(want) => {
                let (p, d, why) = family_letterbox(family, st.declared_placement, (placement, pad));
                fit = FitMode::Letterbox;
                pad = d;
                placement = want.unwrap_or(p);
                notes.push(format!(
                    "resize={r} resolved to {} with {} pad {:?} ({why})",
                    ResizeOverride::Letterbox(Some(placement)),
                    pad.1.as_str(),
                    pad.0
                ));
                provenance.insert(ConfigField::Placement, ov.clone());
                provenance.insert(ConfigField::Pad, ov.clone());
            }
            _ => fit = FitMode::Stretch,
        }
        provenance.insert(ConfigField::Resize, ov.clone());
    }
    let mut decode_mode = None;
    if task == ModelTask::Semantic {
        let declared = semantic_outputs(&mut schema).find_map(|o| o.semantic.decode_mode);
        let modelpack =
            semantic_outputs(&mut schema).any(|o| o.decoder == Some(DecoderKind::ModelPack));
        decode_mode = Some(declared.unwrap_or(if modelpack {
            DecodeMode::Fast
        } else {
            DecodeMode::Accurate
        }));
        let p = if declared.is_some() {
            schema_prov.clone()
        } else {
            Provenance::new(ConfigSource::Default, None)
        };
        provenance.insert(ConfigField::DecodeMode, p);
    }
    if let Some(mode) = overrides.decode {
        if task != ModelTask::Semantic {
            return Err(ModelConfigError::Override(format!(
                "decode applies to semantic models, not a {task} model"
            )));
        }
        decode_mode = Some(mode);
        for o in semantic_outputs(&mut schema) {
            o.semantic.decode_mode = Some(mode);
        }
        provenance.insert(ConfigField::DecodeMode, ov.clone());
    }
    if let Some(l) = &overrides.labels {
        if let Some(i) = l.iter().position(|n| n.trim().is_empty()) {
            return Err(ModelConfigError::Override(format!(
                "labels has an empty name at {i}"
            )));
        }
        if let Some(n) = classes.filter(|&n| n != l.len()) {
            return Err(ModelConfigError::Override(format!(
                "labels names {} classes but the model has {n}",
                l.len()
            )));
        }
        labels = l.iter().map(|n| n.trim().to_owned()).collect();
        provenance.insert(ConfigField::Labels, ov);
    } else {
        for (candidate, prov) in st.labels {
            if let Some(n) = classes.filter(|&n| n != candidate.len()) {
                let note = format!(
                    "labels [{prov}] name {} classes but the model has {n}; they are ignored",
                    candidate.len()
                );
                log::warn!("{note}");
                notes.push(note);
            } else {
                labels = candidate;
                provenance.insert(ConfigField::Labels, prov);
                break;
            }
        }
    }
    if labels.is_empty() {
        let note = "labels unknown; supply them with the labels override";
        log::warn!("{note}");
        notes.push(note.into());
    }
    if let Some(repeats) = repeated_labels(&labels) {
        let note = format!(
            "labels [{}] repeat names, so matching classes by name is ambiguous: {repeats}",
            provenance
                .get(&ConfigField::Labels)
                .map(ToString::to_string)
                .unwrap_or_default(),
        );
        log::warn!("{note}");
        notes.push(note);
    }

    let mut preprocessing = PreprocessSpec::new();
    preprocessing.normalization = normalization;
    preprocessing.fit = fit;
    preprocessing.placement = placement;
    preprocessing.pad = pad.0;
    preprocessing.pad_space = pad.1;
    preprocessing.interpolation = interpolation;
    preprocessing.size = size;
    if interpolation.supported() != interpolation {
        notes.push(format!(
            "interpolation={} is not implemented by the image crate, which runs {}",
            interpolation.as_str(),
            interpolation.supported().as_str()
        ));
    }
    if let Some(input) = schema.input.as_mut() {
        input.preprocessing = Some(to_schema_preprocessing(&preprocessing));
    }
    let description = st
        .description
        .unwrap_or_else(|| format!("{} {} model", family.as_str(), task.as_str()));
    Ok(ResolvedModelConfig {
        schema,
        labels,
        family,
        task,
        decode_mode,
        preprocessing,
        provenance,
        description,
        notes,
        reduce_labels: st.reduce_labels,
    })
}

fn to_schema_preprocessing(p: &PreprocessSpec) -> schema::Preprocessing {
    let normalization = match p.normalization.as_preset() {
        Some(preset) => NormalizationConfig::Preset(preset),
        None => NormalizationConfig::MeanStd {
            mean: p.normalization.mean().to_vec(),
            std: p.normalization.std().to_vec(),
        },
    };
    let letterbox = schema::Letterbox {
        placement: Some(p.placement),
        pad: Some(p.pad.to_vec()),
        pad_space: Some(p.pad_space),
    };
    schema::Preprocessing {
        normalization: Some(normalization),
        resize: Some(p.fit),
        interpolation: Some(p.interpolation),
        letterbox: (p.fit == FitMode::Letterbox).then_some(letterbox),
    }
}

/// Resolves the configuration of the model file at `path`: what it embeds,
/// the files in its directory, and `signals` when given.
///
/// # Errors
///
/// [`ModelConfigError::Io`] when the model or a sidecar cannot be read, and
/// every error of [`resolve_sources`].
///
/// ```no_run
/// use edgefirst_decoder::model_config::{resolve_model_config, ModelOverrides};
///
/// let config = resolve_model_config("segformer.onnx", None, &ModelOverrides::default())?;
/// for line in config.describe() {
///     println!("{line}");
/// }
/// # Ok::<(), edgefirst_decoder::model_config::ModelConfigError>(())
/// ```
pub fn resolve_model_config(
    path: impl AsRef<Path>,
    signals: Option<ModelSignals>,
    overrides: &ModelOverrides,
) -> Result<ResolvedModelConfig, ModelConfigError> {
    let path = path.as_ref();
    let embedded = super::read_embedded(path)?;
    let dir = path
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let sidecar = super::read_sidecars(dir)?;
    let mut sources = ModelSources::new()
        .with_embedded(embedded)
        .with_sidecar(sidecar);
    sources.signals = signals;
    resolve_sources(sources, overrides)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_config::heuristics::tests::{segformer_signals, tensor};
    use crate::model_config::ConfigFile;
    use crate::{ModelSource, TensorInfo};

    fn file(name: &str, text: &str) -> Option<ConfigFile> {
        Some(ConfigFile {
            name: name.into(),
            text: text.into(),
        })
    }

    fn fixture(model: &str, name: &str) -> String {
        let path = format!(
            "{}/../../testdata/hf/{model}/{name}",
            env!("CARGO_MANIFEST_DIR")
        );
        std::fs::read_to_string(path).unwrap()
    }

    fn hf_sidecar(model: &str) -> ModelFiles {
        ModelFiles {
            hf_config: file("config.json", &fixture(model, "config.json")),
            hf_preprocessor: file(
                "preprocessor_config.json",
                &fixture(model, "preprocessor_config.json"),
            ),
            ..Default::default()
        }
    }

    const SEGFORMER_EF: &str = r#"{
      "schema_version": 2,
      "dataset": {"classes": ["bg", "fg"]},
      "input": {"shape": [1, 3, 512, 512],
                "dshape": [{"batch": 1}, {"num_features": 3}, {"height": 512}, {"width": 512}],
                "preprocessing": {"letterbox": {"placement": "topleft", "pad": [0], "pad_space": "normalized"}}},
      "outputs": [{"name": "logits", "type": "segmentation", "decoder": "semantic",
                   "shape": [1, 2, 128, 128],
                   "dshape": [{"batch": 1}, {"num_classes": 2}, {"height": 128}, {"width": 128}],
                   "dtype": "float32", "stride": 4}]
    }"#;

    fn sources(
        embedded: ModelFiles,
        sidecar: ModelFiles,
        signals: Option<ModelSignals>,
    ) -> ModelSources {
        ModelSources {
            embedded,
            sidecar,
            signals,
        }
    }

    fn resolve(s: ModelSources) -> ResolvedModelConfig {
        resolve_sources(s, &ModelOverrides::default()).unwrap()
    }

    #[test]
    fn edgefirst_wins_over_hf_field_by_field() {
        let embedded = ModelFiles {
            edgefirst_json: file("edgefirst.json", SEGFORMER_EF),
            ..Default::default()
        };
        let c = resolve(sources(
            embedded,
            hf_sidecar("segformer-512"),
            Some(segformer_signals(2)),
        ));
        assert_eq!(c.family, ModelFamily::EdgeFirst);
        assert_eq!(c.labels, ["bg", "fg"]);
        assert_eq!(c.preprocessing.placement, LetterboxPlacement::TopLeft);
        assert_eq!(c.preprocessing.fit, FitMode::Letterbox);
        assert_eq!(c.preprocessing.pad_space, PadSpace::Normalized);
        // edgefirst.json declares no normalization: the HF processor fills it.
        assert_eq!(c.preprocessing.normalization, Normalization::imagenet());
        assert_eq!(
            c.source(ConfigField::Normalization).unwrap().source,
            ConfigSource::SidecarHf
        );
        assert_eq!(
            c.source(ConfigField::Placement).unwrap().to_string(),
            "embedded edgefirst.json"
        );
        assert_eq!(c.task, ModelTask::Semantic);
    }

    #[test]
    fn embedded_beats_sidecar() {
        let embedded = ModelFiles {
            labels_txt: file("labels.txt", "a\nb\n"),
            ..Default::default()
        };
        let sidecar = ModelFiles {
            edgefirst_json: file("edgefirst.json", SEGFORMER_EF),
            labels_txt: file("labels.txt", "x\ny\n"),
            ..Default::default()
        };
        let c = resolve(sources(embedded, sidecar, None));
        assert_eq!(c.labels, ["a", "b"]);
        assert_eq!(
            c.source(ConfigField::Labels).unwrap().source,
            ConfigSource::EmbeddedEdgefirst
        );
        assert_eq!(
            c.source(ConfigField::Task).unwrap().source,
            ConfigSource::SidecarEdgefirst
        );
    }

    #[test]
    fn overrides_apply_last() {
        let sources = sources(
            ModelFiles::default(),
            hf_sidecar("segformer-512"),
            Some(segformer_signals(150)),
        );
        let mut o = ModelOverrides::default();
        o.set("normalize", "signed").unwrap();
        o.set("resize", "letterbox").unwrap();
        o.set("decode", "fast").unwrap();
        let c = resolve_sources(sources, &o).unwrap();
        assert_eq!(c.preprocessing.normalization, Normalization::signed());
        assert_eq!(c.preprocessing.placement, LetterboxPlacement::TopLeft);
        assert_eq!(c.preprocessing.pad, [0.0; 3]);
        assert_eq!(c.decode_mode, Some(DecodeMode::Fast));
        assert_eq!(
            c.schema.outputs[0].semantic.decode_mode,
            Some(DecodeMode::Fast)
        );
        for f in [
            ConfigField::Normalization,
            ConfigField::Resize,
            ConfigField::DecodeMode,
        ] {
            assert_eq!(c.source(f).unwrap().source, ConfigSource::Override, "{f:?}");
        }
        assert!(c
            .notes
            .iter()
            .any(|n| n.contains("Hugging Face convention")));
    }

    #[test]
    fn validation_fallback_fills_unset_fields() {
        let mut v: Value = serde_json::from_str(SEGFORMER_EF).unwrap();
        v["input"].as_object_mut().unwrap().remove("preprocessing");
        v["validation"] = serde_json::json!({"normalization": "signed", "preprocessing": "resize"});
        let c = resolve(sources(
            ModelFiles {
                edgefirst_json: file("edgefirst.json", &v.to_string()),
                ..Default::default()
            },
            ModelFiles::default(),
            None,
        ));
        assert_eq!(c.preprocessing.normalization, Normalization::signed());
        assert_eq!(c.preprocessing.fit, FitMode::Stretch);
        assert_eq!(
            c.source(ConfigField::Resize).unwrap().to_string(),
            "embedded edgefirst.json validation.preprocessing"
        );
        assert_eq!(
            c.source(ConfigField::Placement).unwrap().source,
            ConfigSource::Default
        );
    }

    #[test]
    fn segformer_config_resolves_semantic_schema() {
        let c = resolve(sources(
            ModelFiles::default(),
            hf_sidecar("segformer-512"),
            Some(segformer_signals(150)),
        ));
        assert_eq!(c.family, ModelFamily::HuggingFace);
        assert_eq!(c.task, ModelTask::Semantic);
        assert_eq!(c.labels.len(), 150);
        assert_eq!(c.preprocessing.fit, FitMode::Stretch);
        assert_eq!(c.preprocessing.size, Some((512, 512)));
        assert_eq!(c.reduce_labels, Some(true));
        c.schema.validate().unwrap();
        crate::semantic::SemanticConfig::from_logical(&c.schema.outputs[0]).unwrap();
        let pre = c.schema.input.unwrap().preprocessing.unwrap();
        assert_eq!(pre.resize, Some(FitMode::Stretch));
        assert!(c.description.contains("SegformerForSemanticSegmentation"));
    }

    #[test]
    fn heuristic_segformer_dev_io() {
        let c = resolve(sources(
            ModelFiles::default(),
            ModelFiles::default(),
            Some(segformer_signals(150)),
        ));
        assert_eq!(c.family, ModelFamily::HuggingFace);
        for f in [
            ConfigField::Normalization,
            ConfigField::Resize,
            ConfigField::Interpolation,
        ] {
            let p = c.source(f).unwrap();
            assert_eq!(p.source, ConfigSource::Heuristic, "{f:?}");
            assert_eq!(p.to_string(), "guess: SegFormer default");
        }
        assert_eq!(c.preprocessing.normalization, Normalization::imagenet());
        assert!(c.labels.is_empty());
        assert!(c.notes.iter().any(|n| n.contains("labels unknown")));
    }

    #[test]
    fn bare_letterbox_follows_family() {
        let mut o = ModelOverrides::default();
        o.set("resize", "letterbox").unwrap();
        // Unknown family: a semantic schema built from the task override.
        o.set("task", "semantic").unwrap();
        let mut s = segformer_signals(19);
        s.outputs[0].name = "out".into();
        let c =
            resolve_sources(sources(Default::default(), Default::default(), Some(s)), &o).unwrap();
        assert_eq!(c.family, ModelFamily::Unknown);
        assert_eq!(c.preprocessing.placement, LetterboxPlacement::Center);
        assert_eq!(
            (c.preprocessing.pad, c.preprocessing.pad_space),
            ([114.0; 3], PadSpace::Pixel)
        );

        let mut o = ModelOverrides::default();
        o.set("resize", "letterbox-center").unwrap();
        let c = resolve_sources(
            sources(
                Default::default(),
                Default::default(),
                Some(segformer_signals(19)),
            ),
            &o,
        )
        .unwrap();
        assert_eq!(c.preprocessing.placement, LetterboxPlacement::Center);
        assert_eq!(
            c.preprocessing.pad_space,
            PadSpace::Normalized,
            "keeps the HF pad"
        );
    }

    fn ultralytics_signals() -> ModelSignals {
        let mut metadata = BTreeMap::new();
        metadata.insert("author".into(), "Ultralytics".into());
        metadata.insert("task".into(), "detect".into());
        metadata.insert("names".into(), "{0: 'person', 1: 'car'}".into());
        ModelSignals {
            source: ModelSource::Onnx,
            inputs: vec![tensor("images", &[1, 3, 640, 640])],
            outputs: vec![TensorInfo {
                name: "output0".into(),
                shape: vec![1, 6, 8400],
                dtype: schema::DType::Float32,
                quantization: None,
            }],
            metadata,
        }
    }

    #[test]
    fn ultralytics_signals_resolve() {
        let c = resolve(sources(
            Default::default(),
            Default::default(),
            Some(ultralytics_signals()),
        ));
        assert_eq!(c.family, ModelFamily::Ultralytics);
        assert_eq!(c.task, ModelTask::Detect);
        assert_eq!(c.labels, ["person", "car"]);
        assert_eq!(c.preprocessing.pad, [114.0; 3]);
        assert_eq!(
            c.source(ConfigField::Labels).unwrap().to_string(),
            "ultralytics metadata"
        );
        assert_eq!(
            c.source(ConfigField::Pad).unwrap().to_string(),
            "default: Ultralytics convention"
        );

        // Metadata embedded in the container reaches inference.
        let mut s = ultralytics_signals();
        let embedded = ModelFiles {
            metadata: std::mem::take(&mut s.metadata),
            ..Default::default()
        };
        let c = resolve(sources(embedded, Default::default(), Some(s)));
        assert_eq!(c.family, ModelFamily::Ultralytics);

        let mut o = ModelOverrides::default();
        o.set("decode", "fast").unwrap();
        let e = resolve_sources(
            sources(
                Default::default(),
                Default::default(),
                Some(ultralytics_signals()),
            ),
            &o,
        );
        assert!(matches!(e, Err(ModelConfigError::Override(_))));
    }

    #[test]
    fn ultralytics_names_label_an_edgefirst_schema() {
        let ef = r#"{"schema_version": 2,
          "outputs": [{"name": "output0", "type": "detection", "decoder": "ultralytics",
                       "shape": [1, 6, 8400], "dtype": "float32"}]}"#;
        let embedded = ModelFiles {
            edgefirst_json: file("edgefirst.json", ef),
            ..Default::default()
        };
        let c = resolve(sources(
            embedded,
            Default::default(),
            Some(ultralytics_signals()),
        ));
        assert_eq!(c.family, ModelFamily::EdgeFirst);
        assert_eq!(c.labels, ["person", "car"]);
        assert_eq!(
            c.source(ConfigField::Labels).unwrap().source,
            ConfigSource::UltralyticsMetadata
        );
    }

    #[test]
    fn unresolved_names_missing_piece_and_overrides() {
        let mut s = segformer_signals(19);
        s.inputs[0].name = "input".into();
        let e = resolve_sources(
            sources(Default::default(), Default::default(), Some(s)),
            &ModelOverrides::default(),
        )
        .unwrap_err();
        let text = e.to_string();
        assert!(matches!(e, ModelConfigError::Unresolved { .. }), "{text}");
        assert!(text.contains("task=semantic"), "{text}");

        let e = resolve_sources(
            sources(Default::default(), hf_sidecar("segformer-512"), None),
            &ModelOverrides::default(),
        )
        .unwrap_err();
        assert!(matches!(e, ModelConfigError::Unresolved { .. }), "{e}");
    }

    #[test]
    fn hf_labels_that_miscount_the_output_are_ignored() {
        let c = resolve(sources(
            Default::default(),
            hf_sidecar("segformer-512"),
            Some(segformer_signals(19)),
        ));
        assert!(c.labels.is_empty());
        assert!(
            c.notes
                .iter()
                .any(|n| n.contains("name 150 classes but the model has 19")),
            "{:?}",
            c.notes
        );
    }

    #[test]
    fn foreign_config_json_is_noted_and_skipped() {
        let sidecar = ModelFiles {
            hf_config: file("config.json", r#"{"learning_rate": 0.1}"#),
            ..Default::default()
        };
        let c = resolve(sources(
            Default::default(),
            sidecar,
            Some(segformer_signals(19)),
        ));
        assert!(c
            .notes
            .iter()
            .any(|n| n.contains("config.json present but not a Hugging Face config")));
        assert_eq!(
            c.source(ConfigField::Task).unwrap().source,
            ConfigSource::Heuristic
        );
    }

    #[test]
    fn task_override_conflict_is_error() {
        let mut o = ModelOverrides::default();
        o.set("task", "detect").unwrap();
        let e = resolve_sources(
            sources(
                Default::default(),
                Default::default(),
                Some(segformer_signals(19)),
            ),
            &o,
        );
        assert!(matches!(e, Err(ModelConfigError::Override(_))));
        let mut o = ModelOverrides::default();
        o.set("labels", "a,b").unwrap();
        let e = resolve_sources(
            sources(
                Default::default(),
                Default::default(),
                Some(segformer_signals(19)),
            ),
            &o,
        );
        assert!(
            matches!(e, Err(ModelConfigError::Override(_))),
            "label count is checked"
        );
    }

    #[test]
    fn describe_lines_match_format() {
        let c = resolve(sources(
            Default::default(),
            hf_sidecar("segformer-512"),
            Some(segformer_signals(150)),
        ));
        let lines = c.describe();
        assert_eq!(
            lines[0], "normalize=imagenet [preprocessor_config.json]",
            "{lines:?}"
        );
        assert_eq!(lines[1], "resize=stretch [preprocessor_config.json]");
        assert!(
            lines.contains(&"size=512x512 [model input]".to_string()),
            "{lines:?}"
        );
        assert!(
            lines.contains(&"task=semantic [config.json]".to_string()),
            "{lines:?}"
        );
        assert!(lines.contains(&"labels=150 classes [config.json]".to_string()));
        assert!(lines.iter().any(|l| l.starts_with("decode=accurate")));

        let mut o = ModelOverrides::default();
        o.set("resize", "letterbox-topleft").unwrap();
        let c = resolve_sources(
            sources(
                Default::default(),
                Default::default(),
                Some(segformer_signals(150)),
            ),
            &o,
        )
        .unwrap();
        let lines = c.describe();
        assert!(
            lines.contains(&"placement=topleft [override]".to_string()),
            "{lines:?}"
        );
        assert!(lines.contains(&"normalize=imagenet [guess: SegFormer default]".to_string()));
        assert!(lines.contains(&"resize=letterbox-topleft [override]".to_string()));
    }

    /// Every override key `describe()` prints parses back to the same value.
    #[test]
    fn describe_round_trips_through_overrides() {
        let mut sidecar = hf_sidecar("segformer-512");
        sidecar.hf_preprocessor = file(
            "preprocessor_config.json",
            r#"{"image_processor_type": "SegformerImageProcessor", "do_pad": true,
                "image_mean": [0.25, 0.5, 0.75], "image_std": [0.5, 0.25, 0.125]}"#,
        );
        let c = resolve(sources(
            Default::default(),
            sidecar,
            Some(segformer_signals(150)),
        ));
        let mut o = ModelOverrides::new();
        for line in c.describe() {
            let (key, rest) = line.split_once('=').unwrap();
            let value = rest.split(" [").next().unwrap();
            if ["normalize", "resize", "task", "decode"].contains(&key) {
                o.set(key, value).unwrap_or_else(|e| panic!("{line}: {e}"));
            }
        }
        let again = resolve_sources(
            sources(
                Default::default(),
                hf_sidecar("segformer-512"),
                Some(segformer_signals(150)),
            ),
            &o,
        )
        .unwrap();
        assert_eq!(again.preprocessing.fit, c.preprocessing.fit);
        assert_eq!(again.preprocessing.placement, c.preprocessing.placement);
        let (a, b) = (
            again.preprocessing.normalization,
            c.preprocessing.normalization,
        );
        for (x, y) in a
            .mean()
            .iter()
            .chain(&a.std())
            .zip(b.mean().iter().chain(&b.std()))
        {
            assert!((x - y).abs() < 1e-3, "{a:?} vs {b:?}");
        }
    }

    /// An `edgefirst.json` with no outputs supplies preprocessing but not the
    /// schema.
    #[test]
    fn edgefirst_without_outputs_leaves_the_schema_to_hf() {
        let sidecar = ModelFiles {
            edgefirst_json: file(
                "edgefirst.json",
                r#"{"schema_version": 2, "input": {"shape": [1, 3, 512, 512], "preprocessing": {"normalization": "signed"}}}"#,
            ),
            ..hf_sidecar("segformer-512")
        };
        let c = resolve(sources(
            Default::default(),
            sidecar,
            Some(segformer_signals(150)),
        ));
        assert_eq!(c.task, ModelTask::Semantic);
        assert_eq!(c.family, ModelFamily::HuggingFace);
        assert_eq!(c.preprocessing.normalization, Normalization::signed());
        assert!(
            c.notes.iter().any(|n| n.contains("declares no outputs")),
            "{:?}",
            c.notes
        );
        let empty = ModelFiles {
            edgefirst_json: file("edgefirst.json", "{}"),
            ..hf_sidecar("segformer-512")
        };
        let c = resolve(sources(
            Default::default(),
            empty,
            Some(segformer_signals(150)),
        ));
        assert_eq!(c.task, ModelTask::Semantic);
    }

    /// Discovered labels that miscount the classes are noted and the next
    /// source is used; an override that miscounts is an error.
    #[test]
    fn labels_txt_count_mismatch_falls_back() {
        let sidecar = ModelFiles {
            labels_txt: file("labels.txt", "a\nb\n"),
            ..hf_sidecar("segformer-512")
        };
        let c = resolve(sources(
            Default::default(),
            sidecar,
            Some(segformer_signals(150)),
        ));
        assert_eq!(c.labels.len(), 150);
        assert_eq!(
            c.provenance[&ConfigField::Labels].source,
            ConfigSource::SidecarHf
        );
        assert!(
            c.notes
                .iter()
                .any(|n| n.contains("name 2 classes but the model has 150; they are ignored")),
            "{:?}",
            c.notes
        );
        let mut o = ModelOverrides::new();
        o.labels = Some(vec!["a".into(), " ".into()]);
        assert!(matches!(
            resolve_sources(
                sources(
                    Default::default(),
                    hf_sidecar("segformer-512"),
                    Some(segformer_signals(150))
                ),
                &o
            ),
            Err(ModelConfigError::Override(_))
        ));
    }

    #[test]
    fn legacy_segformer_resolves_trimmed_labels() {
        let c = resolve(sources(
            Default::default(),
            hf_sidecar("segformer-legacy-512"),
            Some(segformer_signals(150)),
        ));
        assert_eq!(c.labels.len(), 150);
        assert_eq!(c.labels[7], "bed");
        assert_eq!(c.labels[33], "chest of drawers");
        assert_eq!(c.preprocessing.fit, FitMode::Stretch);
        assert_eq!(c.preprocessing.size, Some((512, 512)));
        assert_eq!(c.reduce_labels, Some(true));
        assert!(
            !c.notes.iter().any(|n| n.contains("repeat")),
            "{:?}",
            c.notes
        );
    }

    fn edgefirst_with_classes(classes: &str) -> ModelSources {
        let ef = SEGFORMER_EF.replace(r#"["bg", "fg"]"#, classes);
        let embedded = ModelFiles {
            edgefirst_json: file("edgefirst.json", &ef),
            ..Default::default()
        };
        sources(embedded, Default::default(), None)
    }

    fn ultralytics_with_names(names: &str) -> ModelSources {
        let mut s = ultralytics_signals();
        s.metadata.insert("names".into(), names.into());
        sources(Default::default(), Default::default(), Some(s))
    }

    /// Asserts that the labels were ignored with a note naming `field`.
    fn assert_ignored_labels(c: &ResolvedModelConfig, field: &str) {
        let ignored = format!("field `{field}` is empty after trimming whitespace; its class");
        assert!(
            c.notes.iter().any(|n| n.contains(&ignored)),
            "{:?}",
            c.notes
        );
        assert!(
            c.notes.iter().any(|n| n.contains("labels unknown")),
            "{:?}",
            c.notes
        );
        assert!(c.labels.is_empty());
    }

    #[test]
    fn labels_are_trimmed_from_every_source() {
        let none = ModelOverrides::default();
        let c = resolve(edgefirst_with_classes(r#"[" bg", "fg\t"]"#));
        assert_eq!(c.labels, ["bg", "fg"]);
        let c = resolve(edgefirst_with_classes(r#"["bg", " "]"#));
        assert_ignored_labels(&c, "dataset.classes[1]");

        let c = resolve(ultralytics_with_names("{0: ' person', 1: 'car '}"));
        assert_eq!(c.labels, ["person", "car"]);
        let c = resolve(ultralytics_with_names("{0: ' ', 1: 'car'}"));
        assert_ignored_labels(&c, "names[0]");

        let mut s = edgefirst_with_classes(r#"["bg", " "]"#);
        s.sidecar.labels_txt = file("labels.txt", "a\nb\n");
        let c = resolve_sources(s, &none).unwrap();
        assert_eq!(c.labels, ["a", "b"]);

        let mut o = ModelOverrides::new();
        o.labels = Some(vec![" a".into(), "b c ".into()]);
        let c = resolve_sources(ultralytics_with_names("{0: 'x', 1: 'y'}"), &o).unwrap();
        assert_eq!(c.labels, ["a", "b c"]);
    }

    #[test]
    fn miscounted_edgefirst_classes_fall_back_to_labels_txt() {
        let mut s = edgefirst_with_classes(r#"["bg"]"#);
        s.sidecar.labels_txt = file("labels.txt", "a\nb\n");
        let c = resolve(s);
        assert_eq!(c.labels, ["a", "b"]);
        assert!(
            c.notes
                .iter()
                .any(|n| n.contains("name 1 classes but the model has 2")),
            "{:?}",
            c.notes
        );
    }

    #[test]
    fn miscounted_ultralytics_names_are_ignored() {
        let c = resolve(ultralytics_with_names("{0: 'person'}"));
        assert_eq!(c.family, ModelFamily::Ultralytics);
        assert!(c.labels.is_empty());
        assert!(
            c.notes
                .iter()
                .any(|n| n.contains("names lists 1 classes but the output has 2")),
            "{:?}",
            c.notes
        );
        let mut o = ModelOverrides::new();
        o.labels = Some(vec!["a".into(), "b".into()]);
        let c = resolve_sources(ultralytics_with_names("{0: 'person'}"), &o).unwrap();
        assert_eq!(c.labels, ["a", "b"]);
    }

    /// An `edgefirst.json` naming three classes for `outputs`, beside a
    /// two-line `labels.txt`.
    fn detection_with_classes(outputs: &str) -> ResolvedModelConfig {
        let ef = format!(
            r#"{{"schema_version": 2, "dataset": {{"classes": ["a", "b", "c"]}},
                 "outputs": {outputs}}}"#
        );
        let embedded = ModelFiles {
            edgefirst_json: file("edgefirst.json", &ef),
            ..Default::default()
        };
        let sidecar = ModelFiles {
            labels_txt: file("labels.txt", "x\ny\n"),
            ..Default::default()
        };
        resolve(sources(embedded, sidecar, None))
    }

    #[test]
    fn detection_labels_fall_back_on_a_class_count_mismatch() {
        let ultralytics = r#"[{"name": "output0", "type": "detection", "decoder": "ultralytics",
            "shape": [1, 6, 8400],
            "dshape": [{"batch": 1}, {"num_features": 6}, {"num_boxes": 8400}]}]"#;
        let segdet = r#"[{"name": "output0", "type": "detection", "decoder": "ultralytics",
            "shape": [1, 38, 8400],
            "dshape": [{"batch": 1}, {"num_features": 38}, {"num_boxes": 8400}]},
            {"name": "protos", "type": "protos", "decoder": "ultralytics",
            "shape": [1, 32, 160, 160],
            "dshape": [{"batch": 1}, {"num_protos": 32}, {"height": 160}, {"width": 160}]}]"#;
        let modelpack = r#"[{"name": "det", "type": "detection", "decoder": "modelpack",
            "anchors": [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
            "shape": [1, 40, 40, 21],
            "dshape": [{"batch": 1}, {"height": 40}, {"width": 40},
                       {"num_anchors_x_features": 21}]}]"#;
        let scores = r#"[{"name": "boxes", "type": "boxes", "decoder": "ultralytics",
            "shape": [1, 4, 8400],
            "dshape": [{"batch": 1}, {"box_coords": 4}, {"num_boxes": 8400}]},
            {"name": "scores", "type": "scores", "decoder": "ultralytics",
            "shape": [1, 2, 8400],
            "dshape": [{"batch": 1}, {"num_classes": 2}, {"num_boxes": 8400}]}]"#;
        for outputs in [ultralytics, segdet, modelpack, scores] {
            let c = detection_with_classes(outputs);
            assert_eq!(c.labels, ["x", "y"], "{outputs}");
            assert!(
                c.notes
                    .iter()
                    .any(|n| n.contains("name 3 classes but the model has 2; they are ignored")),
                "{outputs}: {:?}",
                c.notes
            );
        }
    }

    /// End-to-end detections carry no class count, so the first source wins.
    #[test]
    fn end_to_end_detection_labels_are_not_counted() {
        let c = detection_with_classes(
            r#"[{"name": "output0", "type": "detections", "decoder": "ultralytics",
                "shape": [1, 300, 6],
                "dshape": [{"batch": 1}, {"num_boxes": 300}, {"num_features": 6}]}]"#,
        );
        assert_eq!(c.labels, ["a", "b", "c"]);
    }

    #[test]
    fn repeated_labels_are_noted() {
        let c = resolve(ultralytics_with_names("{0: 'bed', 1: 'Bed '}"));
        assert_eq!(c.labels, ["bed", "Bed"]);
        assert!(
            c.notes
                .iter()
                .any(|n| n.contains("ultralytics metadata") && n.contains(r#""bed" at 0, 1"#)),
            "{:?}",
            c.notes
        );
    }

    /// A reference size that differs in one axis is noted.
    #[test]
    fn width_only_size_mismatch_is_noted() {
        let mut sidecar = hf_sidecar("segformer-512");
        sidecar.hf_preprocessor = file(
            "preprocessor_config.json",
            r#"{"image_processor_type": "SegformerImageProcessor", "size": {"height": 512, "width": 256}}"#,
        );
        let c = resolve(sources(
            Default::default(),
            sidecar,
            Some(segformer_signals(150)),
        ));
        assert!(
            c.notes.iter().any(|n| n.contains("resizes to 256x512")),
            "{:?}",
            c.notes
        );
    }

    #[test]
    fn bicubic_plan_falls_back_with_note() {
        let sidecar = ModelFiles {
            hf_config: file("config.json", &fixture("segformer-512", "config.json")),
            hf_preprocessor: file(
                "preprocessor_config.json",
                r#"{"image_processor_type": "SegformerImageProcessor", "resample": 3,
                    "image_mean": [0.5], "image_std": [0.5], "size": {"height": 512, "width": 512}}"#,
            ),
            ..Default::default()
        };
        let c = resolve(sources(
            Default::default(),
            sidecar,
            Some(segformer_signals(150)),
        ));
        assert_eq!(c.preprocessing.interpolation, Interpolation::Bicubic);
        assert!(c.notes.iter().any(|n| n.contains("interpolation=bicubic")));
    }

    #[test]
    fn resolve_model_config_reads_sidecars() {
        let dir = std::env::temp_dir().join(format!("ef-resolve-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("edgefirst.json"), SEGFORMER_EF).unwrap();
        let model = dir.join("model.bin");
        std::fs::write(&model, [0u8; 64]).unwrap();
        let c = resolve_model_config(&model, None, &ModelOverrides::default());
        std::fs::remove_dir_all(&dir).unwrap();
        let c = c.unwrap();
        assert_eq!(
            c.source(ConfigField::Task).unwrap().source,
            ConfigSource::SidecarEdgefirst
        );
        assert_eq!(c.preprocessing.size, Some((512, 512)));
    }
}
