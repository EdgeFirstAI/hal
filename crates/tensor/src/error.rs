// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use std::fmt::Display;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    IoError(std::io::Error),
    #[cfg(target_os = "linux")]
    NixError(nix::Error),
    NotImplemented(String),
    InvalidSize(usize),
    ShapeMismatch(String),
    UnknownDeviceType(u64, u64),
    InvalidMemoryType(String),
    #[cfg(feature = "ndarray")]
    NdArrayError(ndarray::ShapeError),
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&format!("{:?}", self))
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IoError(err)
    }
}
#[cfg(target_os = "linux")]
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
