// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use core::fmt;

pub type DecoderResult<T, E = DecoderError> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum DecoderError {
    /// An internal error occurred
    Internal(String),
    /// An operation was requested that is not supported
    NotSupported(String),
    /// An invalid tensor shape was given
    InvalidShape(String),
    /// An error occurred while parsing YAML
    Yaml(serde_yaml::Error),
    /// An error occurred while parsing YAML
    Json(serde_json::Error),
    /// Attmpted to use build a decoder without configuration
    NoConfig,
    /// The provide decoder configuration was invalid
    InvalidConfig(String),
    /// An error occurred with ndarray shape operations
    NDArrayShape(ndarray::ShapeError),
}

impl fmt::Display for DecoderError {
    /// Formats the error for display
    /// # Arguments
    /// * `f` - The formatter to write to
    /// # Returns
    /// A result indicating success or failure
    /// # Examples
    /// ```rust
    /// use edgefirst_decoder::DecoderError;
    /// let err = DecoderError::InvalidConfig("The config was invalid".to_string());
    /// println!("{}", err);
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DecoderError {}

impl From<serde_yaml::Error> for DecoderError {
    fn from(err: serde_yaml::Error) -> Self {
        DecoderError::Yaml(err)
    }
}

impl From<serde_json::Error> for DecoderError {
    fn from(err: serde_json::Error) -> Self {
        DecoderError::Json(err)
    }
}

impl From<ndarray::ShapeError> for DecoderError {
    fn from(err: ndarray::ShapeError) -> Self {
        DecoderError::NDArrayShape(err)
    }
}
