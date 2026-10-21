// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! [`DecoderError`] and the [`DecoderResult`] alias returned across the crate.

use core::fmt;

/// `Result` alias defaulting the error type to [`DecoderError`].
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
    /// Schema-declared per-scale child dtype != bound tensor dtype at run time.
    DtypeMismatch {
        expected: edgefirst_tensor::DType,
        actual: edgefirst_tensor::DType,
        role: &'static str,
        level: usize,
    },
    /// Integer tensor bound to a role that requires dequantization, but the
    /// tensor carries no `Quantization` metadata. The upstream inference layer
    /// must call `tensor.set_quantization(...)` before invoking the decoder,
    /// or callers may use `per_scale::apply_schema_quant()` as a fallback.
    QuantMissing {
        dtype: edgefirst_tensor::DType,
        role: &'static str,
        level: usize,
    },
    /// Internal logic bug — a dispatch variant was constructed but the
    /// concrete kernel wasn't matched. Indicates a missing arm in the
    /// dispatch enum's `run()` impl.
    KernelDispatchUnreachable(String),
    /// `EDGEFIRST_DECODER_FORCE_KERNEL` requested a tier whose features
    /// the running CPU doesn't support.
    ForcedKernelUnavailable {
        tier: &'static str,
        missing_feature: &'static str,
    },
    /// A caller tensor refused the operation, e.g. a CPU access its declared
    /// [`CpuAccess`](edgefirst_tensor::CpuAccess) does not cover.
    Tensor(edgefirst_tensor::Error),
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
        match self {
            DecoderError::Tensor(e) => write!(f, "Tensor({e})"),
            _ => write!(f, "{self:?}"),
        }
    }
}

impl std::error::Error for DecoderError {}

impl DecoderError {
    /// The [`CpuAccessDenied`](edgefirst_tensor::Error::CpuAccessDenied)
    /// refusal this error carries, if any.
    pub fn cpu_access_denied(&self) -> Option<&edgefirst_tensor::Error> {
        match self {
            DecoderError::Tensor(e) => e.cpu_access_denied(),
            _ => None,
        }
    }

    /// A failed map: a `CpuAccessDenied` refusal stays typed as
    /// [`DecoderError::Tensor`], any other error goes through `other`.
    pub(crate) fn from_map(
        e: edgefirst_tensor::Error,
        other: impl FnOnce(edgefirst_tensor::Error) -> DecoderError,
    ) -> DecoderError {
        match e {
            e @ edgefirst_tensor::Error::CpuAccessDenied { .. } => DecoderError::Tensor(e),
            e => other(e),
        }
    }
}

impl From<edgefirst_tensor::Error> for DecoderError {
    fn from(err: edgefirst_tensor::Error) -> Self {
        DecoderError::Tensor(err)
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_error_display() {
        let e = DecoderError::Internal("something broke".to_string());
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("Internal") && msg.contains("something broke"),
            "unexpected Internal message: {msg}"
        );

        let e = DecoderError::NotSupported("yolov99".to_string());
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("NotSupported") && msg.contains("yolov99"),
            "unexpected NotSupported message: {msg}"
        );

        let e = DecoderError::InvalidShape("expected 3D".to_string());
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("InvalidShape") && msg.contains("expected 3D"),
            "unexpected InvalidShape message: {msg}"
        );

        let e = DecoderError::NoConfig;
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("NoConfig"),
            "unexpected NoConfig message: {msg}"
        );

        let e = DecoderError::InvalidConfig("missing field".to_string());
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("InvalidConfig") && msg.contains("missing field"),
            "unexpected InvalidConfig message: {msg}"
        );
    }
}
