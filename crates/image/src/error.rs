pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    IoError(std::io::Error),
    LibraryError(libloading::Error),
    EncodingError(jpeg_encoder::EncodingError),
    DecodingError(zune_jpeg::errors::DecodeErrors),
    ImageBufferError(fast_image_resize::ImageBufferError),
    ResizeError(fast_image_resize::ResizeError),
    TransposeError(fast_transpose::TransposeError),
    #[cfg(target_os = "linux")]
    G2DError(g2d_sys::Error),
    TensorError(edgefirst_tensor::Error),
    NotImplemented(String),
    NotSupported(String),
    InvalidShape(String),
    #[cfg(target_os = "linux")]
    EGLError(khronos_egl::Error),
    #[cfg(target_os = "linux")]
    EGLLoadError(khronos_egl::LoadError<libloading::Error>),
    #[cfg(target_os = "linux")]
    InvalidFdError(gbm::InvalidFdError),
    #[cfg(target_os = "linux")]
    FrontBufferError(gbm::FrontBufferError),
    GlError(String),
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

impl From<fast_transpose::TransposeError> for Error {
    fn from(err: fast_transpose::TransposeError) -> Self {
        Error::TransposeError(err)
    }
}

#[cfg(target_os = "linux")]
impl From<g2d_sys::Error> for Error {
    fn from(err: g2d_sys::Error) -> Self {
        Error::G2DError(err)
    }
}

impl From<edgefirst_tensor::Error> for Error {
    fn from(err: edgefirst_tensor::Error) -> Self {
        Error::TensorError(err)
    }
}

#[cfg(target_os = "linux")]
impl From<khronos_egl::Error> for Error {
    fn from(err: khronos_egl::Error) -> Self {
        Error::EGLError(err)
    }
}

#[cfg(target_os = "linux")]
impl From<khronos_egl::LoadError<libloading::Error>> for Error {
    fn from(err: khronos_egl::LoadError<libloading::Error>) -> Self {
        Error::EGLLoadError(err)
    }
}

#[cfg(target_os = "linux")]
impl From<gbm::InvalidFdError> for Error {
    fn from(err: gbm::InvalidFdError) -> Self {
        Error::InvalidFdError(err)
    }
}
#[cfg(target_os = "linux")]
impl From<gbm::FrontBufferError> for Error {
    fn from(err: gbm::FrontBufferError) -> Self {
        Error::FrontBufferError(err)
    }
}
