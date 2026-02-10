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
    #[cfg(feature = "ndarray")]
    NdArrayError(ndarray::ShapeError),
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
        write!(f, "{self:?}")
    }
}

impl std::error::Error for Error {}
