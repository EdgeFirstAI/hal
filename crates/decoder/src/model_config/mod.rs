// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Model configuration discovery.
//!
//! Finds what a model needs to be run and decoded — its output schema, class
//! labels and input preprocessing — from whatever the model carries, and
//! records where each value came from.
//!
//! # Sources
//!
//! Sources are read in this order; each fills only what the ones before it
//! left unset, so `edgefirst.json` wins over a Hugging Face config field by
//! field:
//!
//! 1. `edgefirst.json` embedded in the model (its `input.preprocessing`, then
//!    the legacy `validation.normalization` / `validation.preprocessing`);
//! 2. `edgefirst.json` and `labels.txt` beside the model;
//! 3. Hugging Face `config.json` + `preprocessor_config.json` embedded in the
//!    model;
//! 4. the same files beside the model (the `optimum` export layout);
//! 5. Ultralytics export metadata ([`infer_ultralytics_schema`]);
//! 6. Hugging Face input/output names (`pixel_values` → `logits`), as guesses;
//! 7. the documented defaults.
//!
//! [`ModelOverrides`] apply last.
//!
//! # Embedding conventions
//!
//! | Container | Where | Names |
//! |---|---|---|
//! | ONNX | `ModelProto.metadata_props` | `edgefirst` (legacy `edgefirst.json`), `labels`, `hf_config`, `hf_preprocessor_config` |
//! | TFLite, HEF, DVM, TensorRT, any file | appended ZIP (TFLite associated files) | `edgefirst.json`, `labels.txt`, `config.json`, `preprocessor_config.json` |
//! | any | the model's directory | as for ZIP |
//!
//! ONNX models are read by walking the top-level protobuf fields and seeking
//! past the graph; ZIP trailers need the `model-files` feature (off by
//! default). A `config.json` is used only when it looks like a Hugging Face
//! config (a string `model_type` and `architectures` or
//! `transformers_version`); any other `config.json` is skipped and noted.
//!
//! # Class labels
//!
//! Every label source — `edgefirst.json` `dataset.classes`, `labels.txt` and
//! the ONNX `labels` key, Hugging Face `id2label` / `label2id`, Ultralytics
//! `names` and the `labels` override — has leading and trailing whitespace
//! removed from each name ([`str::trim`]), so ADE20K's `"bed "` resolves to
//! `"bed"`. Whitespace inside a name (`"chest of drawers"`) is kept.
//! Discovered class names never fail resolution: a source whose names are
//! unusable (a name empty after trimming, a malformed `id2label`, or a count
//! that disagrees with `num_labels` or the output) is ignored with a warning
//! and a note naming the source and the index (`id2label[7]`), and the next
//! source supplies the labels. The count is checked against the class count
//! the decoder builder derives from the outputs, or the segmentation
//! channels of a semantic model, so an earlier source that miscounts yields
//! to a later one that fits; end-to-end detections and class maps carry no
//! count, so there the first source wins. A ModelPack segmentation output
//! has one channel more than its classes: the last is background and has no
//! label, so a multitask model's masks and boxes share class indices. Ultralytics `names` that disagree with the output are
//! ignored and the schema is laid out from the count a `labels` override
//! gives, or else from the one output whose class count fits; when several
//! could, the layout is ambiguous and resolution fails, naming the override
//! and a sidecar `edgefirst.json` as the ways to settle it.
//! When no source fits, the labels are empty with a warning.
//! `labels.txt` skips blank lines, so it cannot name an empty class.
//! Names that repeat ignoring case add a note, since they make matching
//! classes by name ambiguous. The `labels` override is checked strictly.
//!
//! [`infer_ultralytics_schema`]: crate::infer_ultralytics_schema

mod container;
mod error;
mod heuristics;
mod hf;
mod onnx;
mod overrides;
mod provenance;
mod resolve;
#[cfg(feature = "model-files")]
mod ziptrailer;

/// Reads `rel` from the workspace `testdata/` tree. `EDGEFIRST_TESTDATA_DIR`
/// wins when set: the manifest-relative fallback is baked in at compile time,
/// so a cross-compiled test binary run on a board cannot reach it.
#[cfg(test)]
fn read_testdata(rel: &str) -> String {
    let root = std::env::var("EDGEFIRST_TESTDATA_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| {
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
        });
    let path = root.join(rel);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

pub use container::{
    read_embedded, read_embedded_bytes, read_sidecars, ConfigFile, ModelFiles, EDGEFIRST_JSON,
    HF_CONFIG, HF_PREPROCESSOR, LABELS_TXT, ONNX_EDGEFIRST, ONNX_HF_CONFIG, ONNX_HF_PREPROCESSOR,
    ONNX_LABELS,
};
pub use error::ModelConfigError;
pub use overrides::{ModelOverrides, NormalizeOverride, ResizeOverride};
pub use provenance::{
    config_field_code, config_source_code, model_family_code, model_task_code, ConfigField,
    ConfigSource, ModelFamily, ModelTask, Provenance,
};
pub use resolve::{resolve_model_config, resolve_sources, ModelSources, ResolvedModelConfig};
