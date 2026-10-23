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
