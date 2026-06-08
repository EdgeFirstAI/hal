// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    IoError(std::io::Error),
    #[cfg(unix)]
    NixError(nix::Error),
    NotImplemented(String),
    InvalidSize(usize),
    ShapeMismatch(String),
    #[cfg(target_os = "linux")]
    UnknownDeviceType(u64, u64),
    InvalidMemoryType(String),
    /// The GL context backing a PBO tensor has been destroyed.
    PboDisconnected,
    /// The PBO buffer is currently mapped and cannot be used for GL operations.
    PboMapped,
    #[cfg(feature = "ndarray")]
    NdArrayError(ndarray::ShapeError),
    InvalidShape(String),
    InvalidArgument(String),
    InvalidOperation(String),
    /// Structured quantization-invariant failure. Round-trippable through
    /// the C and Python boundaries so callers can diagnose which field
    /// failed without parsing strings.
    QuantizationInvalid {
        /// Which invariant failed: `"scale.len"`, `"zero_point.len"`,
        /// `"axis"`, `"per_channel_requires_axis"`,
        /// `"per_tensor_redundant_axis"`, `"dtype_is_integer"`.
        field: &'static str,
        /// What the validator expected, e.g. `"length matches scale (48)"`.
        expected: String,
        /// What was observed, e.g. `"length 32"`.
        got: String,
    },
    /// A capacity-aware operation needs more bytes than the tensor's
    /// underlying allocation provides.
    InsufficientCapacity {
        /// Bytes the requested layout needs.
        needed: usize,
        /// Bytes the allocation provides.
        capacity: usize,
    },
    /// A [`crate::view`]-style sub-region extends past the parent's bounds.
    /// `view`/`batch` reject rather than clamp.
    RegionOutOfBounds {
        /// The offending region.
        region: crate::Region,
        /// The parent frame `(width, height)` in pixels.
        bounds: (usize, usize),
    },
    /// A `batch(n)` index is `>=` the tensor's leading batch dimension `N`.
    BatchIndexOutOfBounds {
        /// The requested element index.
        index: usize,
        /// The tensor's leading dimension `N`.
        batch: usize,
    },
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IoError(err)
    }
}
#[cfg(unix)]
impl From<nix::Error> for Error {
    fn from(err: nix::Error) -> Self {
        Error::NixError(err)
    }
}

#[cfg(feature = "ndarray")]
impl From<ndarray::ShapeError> for Error {
    fn from(err: ndarray::ShapeError) -> Self {
        Error::NdArrayError(err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InsufficientCapacity { needed, capacity } => write!(
                f,
                "insufficient tensor capacity: need {needed} bytes, have {capacity}"
            ),
            Error::RegionOutOfBounds { region, bounds } => write!(
                f,
                "region {region:?} out of bounds for {}x{} frame",
                bounds.0, bounds.1
            ),
            Error::BatchIndexOutOfBounds { index, batch } => write!(
                f,
                "batch index {index} out of bounds for batch size {batch}"
            ),
            _ => write!(f, "{self:?}"),
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let e = Error::InvalidSize(0);
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("InvalidSize"),
            "unexpected InvalidSize message: {msg}"
        );

        let e = Error::NotImplemented("foo".to_string());
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("NotImplemented") && msg.contains("foo"),
            "unexpected NotImplemented message: {msg}"
        );

        let e = Error::ShapeMismatch("expected 3, got 4".to_string());
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("ShapeMismatch") && msg.contains("expected 3"),
            "unexpected ShapeMismatch message: {msg}"
        );

        let e = Error::InvalidMemoryType("dma".to_string());
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("InvalidMemoryType") && msg.contains("dma"),
            "unexpected InvalidMemoryType message: {msg}"
        );

        let e = Error::PboDisconnected;
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("PboDisconnected"),
            "unexpected PboDisconnected message: {msg}"
        );

        let e = Error::PboMapped;
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("PboMapped"),
            "unexpected PboMapped message: {msg}"
        );

        let e = Error::InvalidShape("bad shape".to_string());
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("InvalidShape") && msg.contains("bad shape"),
            "unexpected InvalidShape message: {msg}"
        );

        let e = Error::InvalidArgument("negative".to_string());
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("InvalidArgument") && msg.contains("negative"),
            "unexpected InvalidArgument message: {msg}"
        );

        let e = Error::InvalidOperation("read-only".to_string());
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("InvalidOperation") && msg.contains("read-only"),
            "unexpected InvalidOperation message: {msg}"
        );

        let e = Error::IoError(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "file missing",
        ));
        let msg = e.to_string();
        assert!(!msg.is_empty());
        assert!(
            msg.contains("IoError") && msg.contains("file missing"),
            "unexpected IoError message: {msg}"
        );
    }

    #[test]
    fn insufficient_capacity_message() {
        let e = Error::InsufficientCapacity {
            needed: 100,
            capacity: 64,
        };
        let msg = format!("{e}");
        assert!(
            msg.contains("100") && msg.contains("64"),
            "unexpected message: {msg}"
        );
    }
}
