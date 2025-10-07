pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    NotFound(String),
    Library(libloading::Error),
    JpegEncoding(jpeg_encoder::EncodingError),
    JpegDecoding(zune_jpeg::errors::DecodeErrors),
    PngDecoding(zune_png::error::PngDecodeErrors),
    ResizeImageBuffer(fast_image_resize::ImageBufferError),
    Resize(fast_image_resize::ResizeError),
    Yuv(yuv::YuvError),
    #[cfg(target_os = "linux")]
    G2D(g2d_sys::Error),
    Tensor(edgefirst_tensor::Error),
    NotImplemented(String),
    NotSupported(String),
    InvalidShape(String),
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    EGL(khronos_egl::Error),
    GLVersion(String),
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    EGLLoad(khronos_egl::LoadError<libloading::Error>),
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    GbmInvalidFd(gbm::InvalidFdError),
    #[cfg(target_os = "linux")]
    #[cfg(feature = "opengl")]
    GbmFrontBuffer(gbm::FrontBufferError),
    OpenGl(String),
    Internal(String),
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

impl From<libloading::Error> for Error {
    fn from(err: libloading::Error) -> Self {
        Error::Library(err)
    }
}

impl From<jpeg_encoder::EncodingError> for Error {
    fn from(err: jpeg_encoder::EncodingError) -> Self {
        Error::JpegEncoding(err)
    }
}

impl From<zune_jpeg::errors::DecodeErrors> for Error {
    fn from(err: zune_jpeg::errors::DecodeErrors) -> Self {
        Error::JpegDecoding(err)
    }
}

impl From<zune_png::error::PngDecodeErrors> for Error {
    fn from(err: zune_png::error::PngDecodeErrors) -> Self {
        Error::PngDecoding(err)
    }
}

impl From<fast_image_resize::ImageBufferError> for Error {
    fn from(err: fast_image_resize::ImageBufferError) -> Self {
        Error::ResizeImageBuffer(err)
    }
}

impl From<fast_image_resize::ResizeError> for Error {
    fn from(err: fast_image_resize::ResizeError) -> Self {
        Error::Resize(err)
    }
}

impl From<yuv::YuvError> for Error {
    fn from(err: yuv::YuvError) -> Self {
        Error::Yuv(err)
    }
}

#[cfg(target_os = "linux")]
impl From<g2d_sys::Error> for Error {
    fn from(err: g2d_sys::Error) -> Self {
        Error::G2D(err)
    }
}

impl From<edgefirst_tensor::Error> for Error {
    fn from(err: edgefirst_tensor::Error) -> Self {
        Error::Tensor(err)
    }
}

#[cfg(target_os = "linux")]
#[cfg(feature = "opengl")]
impl From<khronos_egl::Error> for Error {
    fn from(err: khronos_egl::Error) -> Self {
        Error::EGL(err)
    }
}

#[cfg(target_os = "linux")]
#[cfg(feature = "opengl")]
impl From<khronos_egl::LoadError<libloading::Error>> for Error {
    fn from(err: khronos_egl::LoadError<libloading::Error>) -> Self {
        Error::EGLLoad(err)
    }
}

#[cfg(target_os = "linux")]
#[cfg(feature = "opengl")]
impl From<gbm::InvalidFdError> for Error {
    fn from(err: gbm::InvalidFdError) -> Self {
        Error::GbmInvalidFd(err)
    }
}

#[cfg(target_os = "linux")]
#[cfg(feature = "opengl")]
impl From<gbm::FrontBufferError> for Error {
    fn from(err: gbm::FrontBufferError) -> Self {
        Error::GbmFrontBuffer(err)
    }
}

impl From<ndarray::ShapeError> for Error {
    fn from(err: ndarray::ShapeError) -> Self {
        Error::Internal(format!("{err}"))
    }
}
