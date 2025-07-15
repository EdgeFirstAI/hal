pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    IoError(std::io::Error),
    LibraryError(libloading::Error),
    EncodingError(jpeg_encoder::EncodingError),
    DecodingError(zune_jpeg::errors::DecodeErrors),
    ImageBufferError(fast_image_resize::ImageBufferError),
    ResizeError(fast_image_resize::ResizeError),
    G2DError(g2d_sys::Error),
    TensorError(tensor::Error),
    NotImplemented(String),
    InvalidShape(String),
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::IoError(err)
    }
}

impl From<libloading::Error> for Error {
    fn from(err: libloading::Error) -> Self {
        Error::LibraryError(err)
    }
}

impl From<jpeg_encoder::EncodingError> for Error {
    fn from(err: jpeg_encoder::EncodingError) -> Self {
        Error::EncodingError(err)
    }
}

impl From<zune_jpeg::errors::DecodeErrors> for Error {
    fn from(err: zune_jpeg::errors::DecodeErrors) -> Self {
        Error::DecodingError(err)
    }
}

impl From<fast_image_resize::ImageBufferError> for Error {
    fn from(err: fast_image_resize::ImageBufferError) -> Self {
        Error::ImageBufferError(err)
    }
}

impl From<fast_image_resize::ResizeError> for Error {
    fn from(err: fast_image_resize::ResizeError) -> Self {
        Error::ResizeError(err)
    }
}

impl From<g2d_sys::Error> for Error {
    fn from(err: g2d_sys::Error) -> Self {
        Error::G2DError(err)
    }
}

impl From<tensor::Error> for Error {
    fn from(err: tensor::Error) -> Self {
        Error::TensorError(err)
    }
}
