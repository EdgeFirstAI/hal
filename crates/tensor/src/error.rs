pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    IoError(std::io::Error),
    NixError(nix::Error),
    NotImplemented(String),
    InvalidSize(usize),
    ShapeMismatch(String),
    UnknownDeviceType(u64, u64),
    InvalidMemoryType(String),
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IoError(err)
    }
}

impl From<nix::Error> for Error {
    fn from(err: nix::Error) -> Self {
        Error::NixError(err)
    }
}
