pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Tensor(edgefirst_tensor::Error),
    NotImplemented(String),
    NotSupported(String),
    InvalidShape(String),
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<edgefirst_tensor::Error> for Error {
    fn from(err: edgefirst_tensor::Error) -> Self {
        Error::Tensor(err)
    }
}
