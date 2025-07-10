pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    IoError(std::io::Error),
    NixError(nix::Error),
    NotImplemented(String),
    InvalidSize(usize),
    ShapeVolumeMismatch,
    UnknownDeviceType(u64, u64),
    UnsupportedOperation(String),
    InvalidMemoryType(String),
    ResizeError(fast_image_resize::ResizeError),
    ImageBufferError(fast_image_resize::ImageBufferError),
    JpegDecodeError(zune_jpeg::errors::DecodeErrors),
    JpegEncodeError(jpeg_encoder::EncodingError),
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

impl From<fast_image_resize::ResizeError> for Error {
    fn from(err: fast_image_resize::ResizeError) -> Self {
        Error::ResizeError(err)
    }
}

impl From<fast_image_resize::ImageBufferError> for Error {
    fn from(err: fast_image_resize::ImageBufferError) -> Self {
        Error::ImageBufferError(err)
    }
}

impl From<zune_jpeg::errors::DecodeErrors> for Error {
    fn from(err: zune_jpeg::errors::DecodeErrors) -> Self {
        Error::JpegDecodeError(err)
    }
}

impl From<jpeg_encoder::EncodingError> for Error {
    fn from(err: jpeg_encoder::EncodingError) -> Self {
        Error::JpegEncodeError(err)
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::IoError(e) => write!(f, "IO error: {}", e),
            Error::NixError(e) => write!(f, "Nix error: {}", e),
            Error::NotImplemented(msg) => write!(f, "Not implemented: {}", msg),
            Error::InvalidSize(size) => write!(f, "Invalid size: {}", size),
            Error::ShapeVolumeMismatch => write!(f, "Shape volume mismatch"),
            Error::UnknownDeviceType(major, minor) => {
                write!(f, "Unknown device type: {}:{}", major, minor)
            }
            Error::UnsupportedOperation(op) => write!(f, "Unsupported operation: {}", op),
            Error::InvalidMemoryType(mem_type) => write!(f, "Invalid memory type: {}", mem_type),
            Error::ResizeError(e) => write!(f, "{}", e),
            Error::ImageBufferError(e) => write!(f, "{}", e),
            Error::JpegDecodeError(e) => write!(f, "{}", e),
            Error::JpegEncodeError(e) => write!(f, "{}", e),
        }
    }
}
