// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! [`ModelConfigError`], returned by model configuration discovery.

use std::fmt;
use std::path::PathBuf;

use crate::{DecoderError, InferError};

/// Why a model configuration could not be read or resolved.
#[derive(Debug)]
#[non_exhaustive]
pub enum ModelConfigError {
    /// A file could not be read.
    Io {
        /// The file.
        path: PathBuf,
        /// The underlying error.
        source: std::io::Error,
    },
    /// A model container is malformed where its metadata is stored.
    Container(String),
    /// A recognised configuration file has a malformed field.
    InvalidField {
        /// The file, e.g. `config.json`.
        file: String,
        /// The field, e.g. `id2label`.
        field: String,
        /// What is wrong with it.
        reason: String,
    },
    /// A recognised field asks for something the HAL cannot do.
    NotSupported {
        /// The field, e.g. `size.shortest_edge`.
        field: String,
        /// Why it is not supported.
        reason: String,
    },
    /// A Hugging Face architecture with no decoder mapping, named by its
    /// `model_type`.
    UnsupportedArchitecture(String),
    /// An override is malformed or cannot apply to this model.
    Override(String),
    /// Nothing supplied a required piece of the configuration.
    Unresolved {
        /// What is missing.
        missing: String,
        /// The overrides or files that would supply it.
        overrides: String,
    },
    /// Ultralytics metadata was found but could not be interpreted.
    Infer(InferError),
    /// A configuration was found but is not a valid schema.
    Schema(DecoderError),
}

impl fmt::Display for ModelConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "cannot read {}: {source}", path.display()),
            Self::Container(s) => write!(f, "malformed model container: {s}"),
            Self::InvalidField {
                file,
                field,
                reason,
            } => write!(f, "{file}: field `{field}` {reason}"),
            Self::NotSupported { field, reason } => {
                write!(f, "`{field}` is not supported: {reason}")
            }
            Self::UnsupportedArchitecture(t) => write!(
                f,
                "Hugging Face model_type `{t}` has no decoder; supported: *ForSemanticSegmentation"
            ),
            Self::Override(s) => write!(f, "invalid override: {s}"),
            Self::Unresolved { missing, overrides } => {
                write!(f, "missing {missing}; supply {overrides}")
            }
            Self::Infer(e) => write!(f, "Ultralytics metadata: {e}"),
            Self::Schema(e) => write!(f, "invalid model schema: {e}"),
        }
    }
}

impl std::error::Error for ModelConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Infer(e) => Some(e),
            Self::Schema(e) => Some(e),
            _ => None,
        }
    }
}

impl From<InferError> for ModelConfigError {
    fn from(e: InferError) -> Self {
        Self::Infer(e)
    }
}

impl From<DecoderError> for ModelConfigError {
    fn from(e: DecoderError) -> Self {
        Self::Schema(e)
    }
}

pub(crate) fn invalid(file: &str, field: &str, reason: impl Into<String>) -> ModelConfigError {
    ModelConfigError::InvalidField {
        file: file.to_owned(),
        field: field.to_owned(),
        reason: reason.into(),
    }
}
