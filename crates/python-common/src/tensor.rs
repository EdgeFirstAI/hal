// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use edgefirst_tensor::{
    self as tensor, DType, PixelFormat, Region, TensorDyn, TensorMapTrait, TensorMemory,
    TensorTrait,
};
#[cfg(any(not(Py_LIMITED_API), Py_3_11))]
use pyo3::ffi::Py_buffer;
use pyo3::{exceptions::PyBufferError, prelude::*};

use std::ffi::c_void;
#[cfg(any(not(Py_LIMITED_API), Py_3_11))]
use std::ffi::{c_int, CString};
#[cfg(unix)]
use std::os::fd::{IntoRawFd, RawFd};

use std::fmt::{self, Display};

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum Error {
    Tensor(tensor::Error),
    UnsupportedMemoryType(String),
    UnsupportedDataType(String),
    HostView(String),
    Format(String),
    Io(std::io::Error),
    /// A CPU access refusal reported by the image crate.
    #[cfg(feature = "image")]
    Image(edgefirst_image::Error),
    /// A codec failure, raised as its `edgefirst.codec` exception.
    #[cfg(feature = "codec")]
    Codec(edgefirst_codec::CodecError),
    /// A Python exception raised underneath, passed through unchanged.
    Py(PyErr),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Tensor(e) => write!(f, "Tensor error: {e}"),
            Error::UnsupportedMemoryType(msg) => write!(f, "Invalid memory type: {msg}"),
            Error::UnsupportedDataType(msg) => write!(f, "Invalid data type: {msg}"),
            Error::HostView(msg) => write!(f, "Tensor map error: {msg}"),
            Error::Format(msg) => write!(f, "Format error: {msg}"),
            Error::Io(e) => write!(f, "IO error: {e:?}"),
            #[cfg(feature = "image")]
            Error::Image(e) => write!(f, "{e}"),
            #[cfg(feature = "codec")]
            Error::Codec(e) => write!(f, "{e}"),
            Error::Py(e) => write!(f, "{e}"),
        }
    }
}

/// `PermissionError` (an `OSError`) when `denied`, else `RuntimeError`.
pub(crate) fn runtime_or_permission_error(denied: bool, msg: String) -> PyErr {
    if denied {
        pyo3::exceptions::PyPermissionError::new_err(msg)
    } else {
        pyo3::exceptions::PyRuntimeError::new_err(msg)
    }
}

impl From<tensor::Error> for Error {
    fn from(err: tensor::Error) -> Self {
        Error::Tensor(err)
    }
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err)
    }
}

#[cfg(feature = "image")]
impl From<edgefirst_image::Error> for Error {
    fn from(err: edgefirst_image::Error) -> Self {
        if err.cpu_access_denied().is_some() {
            return Error::Image(err);
        }
        Error::Format(format!("{err:?}"))
    }
}

#[cfg(feature = "image")]
impl From<crate::image::Error> for Error {
    fn from(err: crate::image::Error) -> Self {
        match err {
            crate::image::Error::Image(e) if e.cpu_access_denied().is_some() => Error::Image(e),
            crate::image::Error::Tensor(e) if e.cpu_access_denied().is_some() => Error::Tensor(e),
            crate::image::Error::Py(e) => Error::Py(e),
            err => Error::Format(format!("{err}")),
        }
    }
}

impl Error {
    /// Whether this is a CPU access a tensor's declaration refuses.
    pub(crate) fn is_cpu_access_denied(&self) -> bool {
        match self {
            Error::Tensor(e) => e.cpu_access_denied().is_some(),
            #[cfg(feature = "image")]
            Error::Image(e) => e.cpu_access_denied().is_some(),
            #[cfg(feature = "codec")]
            Error::Codec(e) => e.cpu_access_denied().is_some(),
            _ => false,
        }
    }
}

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        match err {
            Error::Py(e) => e,
            #[cfg(feature = "codec")]
            Error::Codec(e) => crate::codec::to_pyerr(e),
            other => runtime_or_permission_error(other.is_cpu_access_denied(), other.to_string()),
        }
    }
}

#[cfg(feature = "codec")]
impl From<edgefirst_codec::CodecError> for Error {
    fn from(err: edgefirst_codec::CodecError) -> Self {
        Error::Codec(err)
    }
}

#[cfg(feature = "codec")]
use std::cell::RefCell;
#[cfg(feature = "codec")]
thread_local! {
    static DECODER: RefCell<edgefirst_codec::ImageDecoder> =
        RefCell::new(edgefirst_codec::ImageDecoder::new());
}

/// IDCT kernel selection for the software JPEG decoder.
///
/// :attr:`Accurate` is the default: the `islow`-class Loeffler IDCT.
/// :attr:`Fast` selects the AAN `ifast`-class kernel, which is faster and less
/// accurate. Fast is advisory: paths without a fast kernel (non-NEON tiers,
/// the V4L2/nvJPEG hardware decoders, and PNG) use their accurate path.
/// Applies to the thread-local decoder used by :meth:`Tensor.decode_image` /
/// :meth:`Tensor.decode_image_file`; set via :func:`set_dct_method`. Each
/// thread has its own decoder state.
#[cfg(feature = "codec")]
#[pyo3::pyclass(name = "DctMethod", eq, eq_int, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PyDctMethod {
    /// Accurate `islow`-class IDCT (default).
    #[default]
    Accurate = 0,
    /// Fast AAN `ifast`-class IDCT (opt-in).
    Fast = 1,
}

#[cfg(feature = "codec")]
impl From<PyDctMethod> for edgefirst_codec::DctMethod {
    fn from(m: PyDctMethod) -> Self {
        match m {
            PyDctMethod::Accurate => edgefirst_codec::DctMethod::Accurate,
            PyDctMethod::Fast => edgefirst_codec::DctMethod::Fast,
        }
    }
}

/// Select the software JPEG IDCT kernel class for the thread-local decoder
/// used by :meth:`Tensor.decode_image` / :meth:`Tensor.decode_image_file`.
/// Affects only the calling thread.
///
/// The default is :attr:`DctMethod.Accurate`. ``EDGEFIRST_CODEC_DCT=fast`` in
/// the environment sets the default of a new thread to
/// :attr:`DctMethod.Fast`.
///
/// Args:
///     method: IDCT kernel class.
#[cfg(feature = "codec")]
#[pyo3::pyfunction]
pub fn set_dct_method(method: PyDctMethod) {
    DECODER.with(|cell| cell.borrow_mut().set_dct_method(method.into()));
}

/// Request a fused JPEG decode output format for the thread-local decoder
/// used by :meth:`Tensor.decode_image` / :meth:`Tensor.decode_image_file`.
/// Affects only the calling thread. PNG decodes are unaffected.
///
/// The software JPEG decoder converts colour and downsamples chroma at the
/// MCU write stage. The V4L2 and nvJPEG decoders are bypassed when the
/// resolved output differs from the native format.
///
/// - ``PixelFormat.RGB``: 4:4:4 colour JPEGs decode to interleaved RGB.
///   Other sources decode to their native format.
/// - ``PixelFormat.NV12``: colour JPEGs decode to NV12, downsampling
///   chroma at the write stage (2×2 average for 4:4:4, vertical for 4:2:2).
/// - Any other format is ignored; the decode produces the native format.
/// - ``None`` (default): native format (``Nv12``/``Nv16``/``Nv24``/``Grey``).
///
/// Args:
///     format: Requested fused output format, or ``None`` to restore native
///         output.
#[cfg(feature = "codec")]
#[pyo3::pyfunction]
#[pyo3(signature = (format=None))]
pub fn set_output_format(format: Option<PyPixelFormat>) {
    let fmt = format.map(edgefirst_tensor::PixelFormat::from);
    DECODER.with(|cell| cell.borrow_mut().set_output_format(fmt));
}

/// True when a V4L2 hardware JPEG decoder (e.g. the i.MX ``mxc-jpeg``
/// block) is present and not disabled via ``EDGEFIRST_DISABLE_V4L2``.
///
/// Opens and closes the device once. Always ``False`` on platforms without
/// V4L2 hardware decode support.
#[cfg(feature = "codec")]
#[pyo3::pyfunction]
pub fn is_v4l2_available() -> bool {
    edgefirst_codec::v4l2_available()
}

/// Shared decode body for ``Tensor.decode_image`` and the cross-package
/// ``decode_into`` free function.
#[cfg(feature = "codec")]
fn decode_image_into(dst: &mut TensorDyn, data: &[u8]) -> Result<PyImageInfo> {
    use edgefirst_codec::ImageLoad;
    DECODER.with(|cell| {
        let mut decoder = cell.borrow_mut();
        let info = dst.load_image(&mut decoder, data)?;
        Ok(PyImageInfo {
            width: info.width,
            height: info.height,
            format: PyPixelFormat::try_from(info.format)
                .map_err(|e| Error::Format(e.to_string()))?,
            row_stride: info.row_stride,
            rotation_degrees: info.rotation_degrees,
            flip_horizontal: info.flip_horizontal,
        })
    })
}

/// Copy the format, shape and colorimetry of a decode from `tensor`, the
/// `TensorDyn` [`crate::interop::reconstruct`] built for the detached region,
/// onto `dst`.
#[cfg(feature = "codec")]
fn write_back_decode(dst: &mut TensorDyn, tensor: &TensorDyn, info: &PyImageInfo) -> Result<()> {
    let fmt: PixelFormat = info.format.into();
    dst.configure_image(info.width, info.height, fmt)?;
    dst.set_colorimetry(tensor.colorimetry());
    Ok(())
}

/// Decode image bytes (JPEG/PNG) into any object implementing the
/// ``__edgefirst_tensor__`` capsule protocol, such as a destination created by
/// another ``edgefirst.*`` extension (e.g.
/// ``edgefirst.image.ImageProcessor.create_image()``).
///
/// Decode semantics (native format, EXIF handling, ``set_output_format`` /
/// ``set_dct_method``) are those of :meth:`Tensor.decode_image`.
///
/// After the decode, ``tensor``'s format, dimensions and colorimetry are
/// updated through ``configure_image()`` and ``colorimetry``. When ``tensor``
/// does not implement ``configure_image()`` or exposes ``colorimetry``
/// read-only, that update is skipped without raising. The returned
/// :class:`ImageInfo` always describes the decode.
///
/// Args:
///     tensor: Destination tensor, from this or another ``edgefirst.*``
///         package (anything implementing ``__edgefirst_tensor__``).
///     data: Raw JPEG or PNG bytes.
#[cfg(feature = "codec")]
#[pyo3::pyfunction]
pub fn decode_into(tensor: &Bound<'_, PyAny>, data: &[u8]) -> PyResult<PyImageInfo> {
    let py = tensor.py();
    let arg = crate::interop::TensorArg::extract_mut(tensor, None)?;
    // Copy the input while the GIL is held.
    let data = data.to_vec();
    let (info, colorimetry) = if arg.can_detach() {
        // Decode into a `TensorDyn` reconstructed from `tensor`'s descriptor, with
        // the GIL released.
        let mut raw = arg.into_raw_access()?;
        let (info, raw) = py
            .detach(move || decode_image_into(raw.as_mut(), &data).map(|info| (info, raw)))
            .map_err(PyErr::from)?;
        (info, raw.as_ref().colorimetry())
    } else {
        // A PBO-backed `tensor` is decoded in place with the GIL held.
        let mut arg = arg;
        let info = decode_image_into(arg.as_mut(), &data).map_err(PyErr::from)?;
        (info, arg.as_ref().colorimetry())
    };

    // Copy the decoded format, shape and colorimetry onto `tensor`. A
    // destination that rejects either update is logged and skipped.
    if let Err(e) = tensor.call_method1("configure_image", (info.width, info.height, info.format)) {
        log::warn!(
            "decode_into: destination rejected configure_image({}, {}, {:?}) after a \
             successful decode ({e}); its format/shape may now read stale until the \
             caller reconfigures it explicitly",
            info.width,
            info.height,
            info.format,
        );
    }
    let colorimetry: Option<crate::colorimetry::PyColorimetry> = colorimetry.map(Into::into);
    if let Err(e) = tensor.setattr("colorimetry", colorimetry) {
        log::warn!(
            "decode_into: destination rejected the colorimetry write-back after a \
             successful decode ({e}); downstream colour conversion may fall back to an \
             undefined colorimetry until the caller sets it explicitly"
        );
    }

    Ok(info)
}

/// Decode an image file (JPEG/PNG) into any object implementing the
/// ``__edgefirst_tensor__`` capsule protocol. See :func:`decode_into`.
///
/// Args:
///     tensor: Destination tensor, from this or another ``edgefirst.*``
///         package (anything implementing ``__edgefirst_tensor__``).
///     filename: Path to the image file.
#[cfg(feature = "codec")]
#[pyo3::pyfunction]
pub fn decode_file_into(tensor: &Bound<'_, PyAny>, filename: &str) -> PyResult<PyImageInfo> {
    let data = std::fs::read(filename).map_err(Error::Io)?;
    decode_into(tensor, &data)
}

/// Metadata returned by ``decode_image`` / ``decode_image_file``.
#[cfg(feature = "codec")]
#[pyclass(
    name = "ImageInfo",
    get_all,
    skip_from_py_object,
    module = "edgefirst.tensor"
)]
#[derive(Debug, Clone)]
pub struct PyImageInfo {
    /// Decoded image width in pixels.
    pub width: usize,
    /// Decoded image height in pixels.
    pub height: usize,
    /// Native pixel format of the decoded data.
    pub format: PyPixelFormat,
    /// Row stride in bytes used for writing.
    pub row_stride: usize,
    /// Clockwise rotation in degrees reported by EXIF orientation (0/90/180/270).
    /// The decode itself never rotates; the decoded dimensions are unrotated.
    pub rotation_degrees: u16,
    /// Horizontal flip reported by EXIF orientation. The decode never flips.
    pub flip_horizontal: bool,
}

#[pymethods]
#[cfg(feature = "codec")]
impl PyImageInfo {
    fn __repr__(&self) -> String {
        format!(
            "ImageInfo(width={}, height={}, format={:?}, row_stride={}, rotation_degrees={}, flip_horizontal={})",
            self.width,
            self.height,
            self.format,
            self.row_stride,
            self.rotation_degrees,
            self.flip_horizontal
        )
    }
}

// Discriminants are the codes of `edgefirst_tensor::TensorMemory`.
#[pyclass(
    name = "TensorMemory",
    skip_from_py_object,
    module = "edgefirst.tensor"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms, non_camel_case_types)]
pub enum PyTensorMemory {
    /// Regular system memory allocation. Available everywhere.
    MEM = 0,
    /// POSIX Shared Memory allocation. Suitable for inter-process
    /// communication, but not for hardware acceleration. Nameable on every
    /// platform; only allocatable on unix.
    SHM = 1,
    /// Platform-native zero-copy GPU buffer: DMA-BUF on Linux,
    /// IOSurface on macOS.
    DMABUF = 2,
    /// Apple IOSurface. No backend produces or accepts it; macOS and iOS
    /// allocate and report `DMABUF`.
    IOSURFACE = 3,
    /// GPU Pixel Buffer Object (PBO) allocation. Used for zero-copy GPU
    /// upload/readback on platforms without DMA-buf support.
    PBO = 4,
    /// CUDA device memory. No backend produces or accepts it.
    CUDA = 5,
    /// Windows D3D11 texture. No backend produces or accepts it; Windows
    /// allocates and reports `DMABUF`.
    D3D11_TEXTURE = 6,
}

// Compile-time checks that each discriminant equals `TensorMemory::code()`.
const _: () = assert!(PyTensorMemory::MEM as u32 == TensorMemory::Mem.code());
const _: () = assert!(PyTensorMemory::SHM as u32 == TensorMemory::Shm.code());
const _: () = assert!(PyTensorMemory::DMABUF as u32 == TensorMemory::DmaBuf.code());
const _: () = assert!(PyTensorMemory::IOSURFACE as u32 == TensorMemory::IoSurface.code());
const _: () = assert!(PyTensorMemory::PBO as u32 == TensorMemory::Pbo.code());
const _: () = assert!(PyTensorMemory::CUDA as u32 == TensorMemory::Cuda.code());
const _: () = assert!(PyTensorMemory::D3D11_TEXTURE as u32 == TensorMemory::D3d11Texture.code());

impl PyTensorMemory {
    /// Reconstruct from the `__int__()` discriminant of a sibling package's
    /// copy of this enum.
    fn from_discriminant(v: i64) -> Option<Self> {
        u32::try_from(v)
            .ok()
            .and_then(TensorMemory::from_code)
            .map(Into::into)
    }
}

/// Extracts a native instance, or a sibling `edgefirst.*` package's
/// `TensorMemory` through its `__int__()` discriminant.
impl<'a, 'py> FromPyObject<'a, 'py> for PyTensorMemory {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        extract_eq_int_enum(obj, "TensorMemory", Self::from_discriminant)
    }
}

#[pymethods]
impl PyTensorMemory {
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        eq_int_richcmp(*self, other, false, "TensorMemory", Self::from_discriminant)
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        eq_int_richcmp(*self, other, true, "TensorMemory", Self::from_discriminant)
    }

    /// Returns the discriminant as the hash.
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl From<PyTensorMemory> for TensorMemory {
    fn from(value: PyTensorMemory) -> Self {
        match value {
            PyTensorMemory::MEM => TensorMemory::Mem,
            PyTensorMemory::SHM => TensorMemory::Shm,
            PyTensorMemory::DMABUF => TensorMemory::DmaBuf,
            PyTensorMemory::IOSURFACE => TensorMemory::IoSurface,
            PyTensorMemory::PBO => TensorMemory::Pbo,
            PyTensorMemory::CUDA => TensorMemory::Cuda,
            PyTensorMemory::D3D11_TEXTURE => TensorMemory::D3d11Texture,
        }
    }
}

impl From<TensorMemory> for PyTensorMemory {
    fn from(value: TensorMemory) -> Self {
        match value {
            TensorMemory::Mem => PyTensorMemory::MEM,
            TensorMemory::Shm => PyTensorMemory::SHM,
            TensorMemory::DmaBuf => PyTensorMemory::DMABUF,
            TensorMemory::IoSurface => PyTensorMemory::IOSURFACE,
            TensorMemory::Pbo => PyTensorMemory::PBO,
            TensorMemory::Cuda => PyTensorMemory::CUDA,
            TensorMemory::D3d11Texture => PyTensorMemory::D3D11_TEXTURE,
            other => {
                log::warn!("TensorMemory {other:?} has no Python counterpart, reporting MEM");
                PyTensorMemory::MEM
            }
        }
    }
}

/// Extract a `#[pyclass(eq, eq_int)]` value enum (`TensorMemory`,
/// `PixelFormat`, the colorimetry axis enums) from a native instance or from a
/// sibling `edgefirst.*` package's copy of the same enum.
///
/// The native downcast is tried first. Otherwise, when `type(obj).__name__`
/// equals `type_name`, the value is rebuilt from the object's `__int__()`
/// discriminant. A bare Python `int` is not accepted.
pub(crate) fn extract_eq_int_enum<'a, 'py, T, F>(
    obj: Borrowed<'a, 'py, PyAny>,
    type_name: &str,
    from_discriminant: F,
) -> PyResult<T>
where
    T: pyo3::PyClass + Copy,
    F: FnOnce(i64) -> Option<T>,
{
    if let Ok(guard) = obj.extract::<pyo3::PyClassGuard<'_, T>>() {
        return Ok(*guard);
    }
    let is_same_type = obj
        .get_type()
        .name()
        .map(|n| n == type_name)
        .unwrap_or(false);
    if is_same_type {
        if let Ok(v) = obj.call_method0("__int__").and_then(|r| r.extract::<i64>()) {
            return from_discriminant(v).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown {type_name} discriminant {v}"
                ))
            });
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "expected a {type_name}"
    )))
}

/// Cross-package `__eq__`/`__ne__` for the value enums covered by
/// [`extract_eq_int_enum`].
///
/// `other` compares equal when it extracts to the same variant through
/// [`extract_eq_int_enum`], or when it is a Python `int` equal to the
/// discriminant. Any other type returns `NotImplemented`. `invert` selects
/// `__ne__` over `__eq__`.
pub(crate) fn eq_int_richcmp<T, F>(
    this: T,
    other: &Bound<'_, PyAny>,
    invert: bool,
    type_name: &str,
    from_discriminant: F,
) -> PyResult<Py<PyAny>>
where
    T: pyo3::PyClass + PartialEq + Copy,
    F: Fn(i64) -> Option<T> + Copy,
{
    let py = other.py();
    if let Ok(v) = extract_eq_int_enum(other.as_borrowed(), type_name, from_discriminant) {
        return pyo3::IntoPyObjectExt::into_py_any((this == v) != invert, py);
    }
    if let Ok(v) = other.extract::<i64>() {
        let eq = from_discriminant(v).is_some_and(|other_val| this == other_val);
        return pyo3::IntoPyObjectExt::into_py_any(eq != invert, py);
    }
    Ok(py.NotImplemented())
}

/// Parse a Python dtype string (e.g. "float32", "uint8") into a `DType`.
pub(crate) fn parse_dtype(dtype: &str) -> Result<DType> {
    match dtype {
        "uint8" => Ok(DType::U8),
        "int8" => Ok(DType::I8),
        "uint16" => Ok(DType::U16),
        "int16" => Ok(DType::I16),
        "uint32" => Ok(DType::U32),
        "int32" => Ok(DType::I32),
        "uint64" => Ok(DType::U64),
        "int64" => Ok(DType::I64),
        "float16" => Ok(DType::F16),
        "float32" => Ok(DType::F32),
        "float64" => Ok(DType::F64),
        _ => Err(Error::UnsupportedDataType(dtype.to_string())),
    }
}

/// Parse a Python CPU-access string into a `CpuAccess` declaration.
pub(crate) fn parse_cpu_access(access: &str) -> Result<tensor::CpuAccess> {
    tensor::CpuAccess::from_str_code(access).ok_or_else(|| {
        Error::Format(format!(
            "access must be one of none|read|write|readwrite, got {access:?}"
        ))
    })
}

/// Parse a Python compression-request string into a [`Compression`]
/// request (`None` input = no request).
#[cfg(feature = "image")]
pub(crate) fn parse_compression(compression: Option<&str>) -> Result<Option<tensor::Compression>> {
    use tensor::{Compression, CompressionScheme};
    match compression {
        None => Ok(None),
        Some("any") => Ok(Some(Compression::Any)),
        Some("ubwc") => Ok(Some(Compression::Scheme(CompressionScheme::Ubwc))),
        Some("afbc") => Ok(Some(Compression::Scheme(CompressionScheme::Afbc))),
        Some("pvric") => Ok(Some(Compression::Scheme(CompressionScheme::Pvric))),
        Some("dcc") => Ok(Some(Compression::Scheme(CompressionScheme::Dcc))),
        Some(other) => Err(Error::Format(format!(
            "compression must be one of any|ubwc|afbc|pvric|dcc, got {other:?}"
        ))),
    }
}

/// Convert a `DType` to a Python dtype string.
fn dtype_to_str(dtype: DType) -> &'static str {
    match dtype {
        DType::U8 => "uint8",
        DType::I8 => "int8",
        DType::U16 => "uint16",
        DType::I16 => "int16",
        DType::U32 => "uint32",
        DType::I32 => "int32",
        DType::U64 => "uint64",
        DType::I64 => "int64",
        DType::F16 => "float16",
        DType::F32 => "float32",
        DType::F64 => "float64",
        _ => "unknown",
    }
}

// ─── Type-erased HostView ──────────────────────────────────────────────────
// Dispatches per dtype for typed pointers, format strings and per-element
// operations.

pub enum TensorMapT {
    TensorU8(tensor::HostView<'static, u8>),
    TensorI8(tensor::HostView<'static, i8>),
    TensorU16(tensor::HostView<'static, u16>),
    TensorI16(tensor::HostView<'static, i16>),
    TensorU32(tensor::HostView<'static, u32>),
    TensorI32(tensor::HostView<'static, i32>),
    TensorU64(tensor::HostView<'static, u64>),
    TensorI64(tensor::HostView<'static, i64>),
    TensorF16(tensor::HostView<'static, half::f16>),
    TensorF32(tensor::HostView<'static, f32>),
    TensorF64(tensor::HostView<'static, f64>),
}

/// Dispatch a method call across all TensorMapT variants.
macro_rules! map_dispatch {
    ($self:expr, $method:ident $(, $arg:expr)*) => {
        match $self {
            TensorMapT::TensorU8(m) => m.$method($($arg),*),
            TensorMapT::TensorI8(m) => m.$method($($arg),*),
            TensorMapT::TensorU16(m) => m.$method($($arg),*),
            TensorMapT::TensorI16(m) => m.$method($($arg),*),
            TensorMapT::TensorU32(m) => m.$method($($arg),*),
            TensorMapT::TensorI32(m) => m.$method($($arg),*),
            TensorMapT::TensorU64(m) => m.$method($($arg),*),
            TensorMapT::TensorI64(m) => m.$method($($arg),*),
            TensorMapT::TensorF16(m) => m.$method($($arg),*),
            TensorMapT::TensorF32(m) => m.$method($($arg),*),
            TensorMapT::TensorF64(m) => m.$method($($arg),*),
        }
    };
}

impl TensorMapT {
    pub fn unmap(&mut self) {
        map_dispatch!(self, unmap);
    }

    /// Whether the underlying mapping permits writes.
    pub fn is_writable(&self) -> bool {
        map_dispatch!(self, is_writable)
    }

    pub fn shape(&self) -> &[usize] {
        map_dispatch!(self, shape)
    }

    pub fn size(&self) -> usize {
        map_dispatch!(self, size)
    }

    /// Bytes the mapping spans. For a strided image this is every
    /// padded row, which is more than [`size`](Self::size)'s logical bytes.
    pub fn mapped_bytes(&self) -> usize {
        match self {
            TensorMapT::TensorU8(m) => std::mem::size_of_val(m.as_slice()),
            TensorMapT::TensorI8(m) => std::mem::size_of_val(m.as_slice()),
            TensorMapT::TensorU16(m) => std::mem::size_of_val(m.as_slice()),
            TensorMapT::TensorI16(m) => std::mem::size_of_val(m.as_slice()),
            TensorMapT::TensorU32(m) => std::mem::size_of_val(m.as_slice()),
            TensorMapT::TensorI32(m) => std::mem::size_of_val(m.as_slice()),
            TensorMapT::TensorU64(m) => std::mem::size_of_val(m.as_slice()),
            TensorMapT::TensorI64(m) => std::mem::size_of_val(m.as_slice()),
            TensorMapT::TensorF16(m) => std::mem::size_of_val(m.as_slice()),
            TensorMapT::TensorF32(m) => std::mem::size_of_val(m.as_slice()),
            TensorMapT::TensorF64(m) => std::mem::size_of_val(m.as_slice()),
        }
    }

    pub fn element_size(&self) -> usize {
        match self {
            TensorMapT::TensorU8(_) => std::mem::size_of::<u8>(),
            TensorMapT::TensorI8(_) => std::mem::size_of::<i8>(),
            TensorMapT::TensorU16(_) => std::mem::size_of::<u16>(),
            TensorMapT::TensorI16(_) => std::mem::size_of::<i16>(),
            TensorMapT::TensorU32(_) => std::mem::size_of::<u32>(),
            TensorMapT::TensorI32(_) => std::mem::size_of::<i32>(),
            TensorMapT::TensorU64(_) => std::mem::size_of::<u64>(),
            TensorMapT::TensorI64(_) => std::mem::size_of::<i64>(),
            TensorMapT::TensorF16(_) => std::mem::size_of::<half::f16>(),
            TensorMapT::TensorF32(_) => std::mem::size_of::<f32>(),
            TensorMapT::TensorF64(_) => std::mem::size_of::<f64>(),
        }
    }

    pub fn get_value_at(&self, index: usize, py: Python) -> PyResult<Py<PyAny>> {
        if index >= self.size() {
            return Err(PyBufferError::new_err("Index out of bounds"));
        }
        match self {
            TensorMapT::TensorU8(m) => Ok(m.as_slice()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorI8(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorU16(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorI16(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorU32(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorI32(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorU64(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorI64(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorF16(m) => Ok(half::f16::to_f32(m.as_ref()[index])
                .into_pyobject(py)?
                .into()),
            TensorMapT::TensorF32(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
            TensorMapT::TensorF64(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
        }
    }

    pub fn set_value_at(&mut self, index: usize, value: Py<PyAny>, py: Python) -> PyResult<()> {
        if index >= self.size() {
            return Err(PyBufferError::new_err("Index out of bounds"));
        }
        match self {
            TensorMapT::TensorU8(m) => m.as_mut()[index] = value.extract::<u8>(py)?,
            TensorMapT::TensorI8(m) => m.as_mut()[index] = value.extract::<i8>(py)?,
            TensorMapT::TensorU16(m) => m.as_mut()[index] = value.extract::<u16>(py)?,
            TensorMapT::TensorI16(m) => m.as_mut()[index] = value.extract::<i16>(py)?,
            TensorMapT::TensorU32(m) => m.as_mut()[index] = value.extract::<u32>(py)?,
            TensorMapT::TensorI32(m) => m.as_mut()[index] = value.extract::<i32>(py)?,
            TensorMapT::TensorU64(m) => m.as_mut()[index] = value.extract::<u64>(py)?,
            TensorMapT::TensorI64(m) => m.as_mut()[index] = value.extract::<i64>(py)?,
            TensorMapT::TensorF16(m) => {
                m.as_mut()[index] = half::f16::from_f32(value.extract::<f32>(py)?)
            }
            TensorMapT::TensorF32(m) => m.as_mut()[index] = value.extract::<f32>(py)?,
            TensorMapT::TensorF64(m) => m.as_mut()[index] = value.extract::<f64>(py)?,
        }
        Ok(())
    }

    /// Get a raw pointer to the mapped data (for Python buffer protocol).
    fn data_ptr(&self) -> *mut c_void {
        match self {
            TensorMapT::TensorU8(m) => m.as_slice().as_ptr() as *mut c_void,
            TensorMapT::TensorI8(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorU16(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorI16(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorU32(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorI32(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorU64(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorI64(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorF16(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorF32(m) => m.as_ref().as_ptr() as *mut c_void,
            TensorMapT::TensorF64(m) => m.as_ref().as_ptr() as *mut c_void,
        }
    }

    /// Get the struct format character for Python buffer protocol.
    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    fn format_str(&self) -> &'static str {
        match self {
            TensorMapT::TensorU8(_) => "B",
            TensorMapT::TensorI8(_) => "b",
            TensorMapT::TensorU16(_) => "H",
            TensorMapT::TensorI16(_) => "h",
            TensorMapT::TensorU32(_) => "I",
            TensorMapT::TensorI32(_) => "i",
            TensorMapT::TensorU64(_) => "Q",
            TensorMapT::TensorI64(_) => "q",
            TensorMapT::TensorF16(_) => "e",
            TensorMapT::TensorF32(_) => "f",
            TensorMapT::TensorF64(_) => "d",
        }
    }

    fn dtype_name(&self) -> &'static str {
        match self {
            TensorMapT::TensorU8(_) => "uint8",
            TensorMapT::TensorI8(_) => "int8",
            TensorMapT::TensorU16(_) => "uint16",
            TensorMapT::TensorI16(_) => "int16",
            TensorMapT::TensorU32(_) => "uint32",
            TensorMapT::TensorI32(_) => "int32",
            TensorMapT::TensorU64(_) => "uint64",
            TensorMapT::TensorI64(_) => "int64",
            TensorMapT::TensorF16(_) => "float16",
            TensorMapT::TensorF32(_) => "float32",
            TensorMapT::TensorF64(_) => "float64",
        }
    }
}

/// Which of the two mapping calls [`map_tensor_dyn`] dispatches to.
#[derive(Clone, Copy)]
enum MapWait {
    /// `map_with`: wait for a GPU copy the map depends on to finish.
    Block,
    /// `try_map_with`: returns `WouldBlock` instead of waiting.
    Fail,
}

/// Map a `TensorDyn` to a `TensorMapT`.
fn map_tensor_dyn(
    t: &TensorDyn,
    access: tensor::CpuAccess,
    wait: MapWait,
) -> tensor::Result<TensorMapT> {
    // Maps with `map_with` or `try_map_with` according to `wait`.
    macro_rules! lens {
        ($ty:ty, $variant:ident) => {{
            let typed = t.as_typed::<$ty>().expect("dtype checked");
            match wait {
                MapWait::Block => typed.map_with(access),
                MapWait::Fail => typed.try_map_with(access),
            }
            .map(TensorMapT::$variant)
        }};
    }
    match t.dtype() {
        tensor::DType::U8 => lens!(u8, TensorU8),
        tensor::DType::I8 => lens!(i8, TensorI8),
        tensor::DType::U16 => lens!(u16, TensorU16),
        tensor::DType::I16 => lens!(i16, TensorI16),
        tensor::DType::U32 => lens!(u32, TensorU32),
        tensor::DType::I32 => lens!(i32, TensorI32),
        tensor::DType::U64 => lens!(u64, TensorU64),
        tensor::DType::I64 => lens!(i64, TensorI64),
        tensor::DType::F16 => lens!(half::f16, TensorF16),
        tensor::DType::F32 => lens!(f32, TensorF32),
        tensor::DType::F64 => lens!(f64, TensorF64),
        _ => Err(tensor::Error::InvalidArgument(
            "unsupported dtype for tensor mapping".to_string(),
        )),
    }
}

/// Translate a failed `MapWait::Fail` mapping into a Python exception:
/// `BlockingIOError` for `WouldBlock`, the tensor error's exception otherwise.
fn would_block_err(err: tensor::Error) -> PyErr {
    match &err {
        tensor::Error::IoError(io) if io.kind() == std::io::ErrorKind::WouldBlock => {
            pyo3::exceptions::PyBlockingIOError::new_err(format!("try_map: {io}"))
        }
        _ => Error::from(err).into(),
    }
}

/// Check a texture-import `shape` against the geometry read off the texture.
///
/// Accepts the allocation shape or the addressing shape the format gives
/// those dimensions.
#[cfg(target_os = "windows")]
fn check_d3d11_shape(
    geometry: tensor::Result<(usize, usize)>,
    format: PixelFormat,
    shape: &[usize],
    what: &str,
) -> PyResult<(usize, usize)> {
    let (width, height) = geometry.map_err(Error::from)?;
    let allocation = format.allocation_shape(width, height);
    let addressing = format.addressing_shape(width, height);
    if [&allocation, &addressing]
        .into_iter()
        .flatten()
        .any(|s| s.as_slice() == shape)
    {
        return Ok((width, height));
    }
    Err(pyo3::exceptions::PyValueError::new_err(format!(
        "{what}: shape {shape:?} describes neither the allocation shape {allocation:?} nor \
         the addressing shape {addressing:?} of this {width}x{height} {format:?} texture"
    )))
}

// ─── numpy → tensor copy ────────────────────────────────────────────────────

/// Byte threshold above which copies are parallelized via rayon.
const PARALLEL_THRESHOLD_BYTES: usize = 256 * 1024;

/// Type-matched copy from a numpy array into a `TensorDyn`.
///
/// Downcasts the numpy array to the concrete element type matching the
/// tensor's dtype, then copies via the typed `HostView` slice.
///
/// Copy strategy:
/// 1. Fully contiguous: a single `copy_from_slice`.
/// 2. Strided with contiguous inner rows: one memcpy per row, iterating over
///    the outer dimensions.
/// 3. Fully strided (e.g. a transposed view): a contiguous source is
///    materialized via `np.ascontiguousarray()` and then copied.
///
/// All three paths use `rayon` parallel iteration at 256 KiB and above.
///
/// Returns an error on dtype mismatch or element-count mismatch.
fn copy_numpy_to_tensor_dyn(src: &Bound<'_, pyo3::types::PyAny>, tensor: &TensorDyn) -> Result<()> {
    macro_rules! lens {
        ($ty:ty) => {
            copy_numpy_typed::<$ty>(src, tensor.as_typed::<$ty>().expect("dtype checked"))
        };
    }
    match tensor.dtype() {
        tensor::DType::U8 => lens!(u8),
        tensor::DType::I8 => lens!(i8),
        tensor::DType::U16 => lens!(u16),
        tensor::DType::I16 => lens!(i16),
        tensor::DType::U32 => lens!(u32),
        tensor::DType::I32 => lens!(i32),
        tensor::DType::U64 => lens!(u64),
        tensor::DType::I64 => lens!(i64),
        tensor::DType::F16 => lens!(half::f16),
        tensor::DType::F32 => lens!(f32),
        tensor::DType::F64 => lens!(f64),
        _ => Err(Error::UnsupportedDataType(format!(
            "tensor dtype {:?} not supported for from_numpy",
            tensor.dtype()
        ))),
    }
}

fn copy_numpy_typed<
    T: numpy::Element
        + edgefirst_tensor::Element
        + num_traits::Num
        + Copy
        + Clone
        + std::fmt::Debug
        + Send
        + Sync,
>(
    src: &Bound<'_, pyo3::types::PyAny>,
    tensor: &tensor::Tensor<T>,
) -> Result<()> {
    use numpy::{PyArrayMethods, PyUntypedArrayMethods};

    let py = src.py();
    let arr = src
        .cast::<numpy::PyArrayDyn<T>>()
        .map_err(|_| Error::Format("numpy dtype does not match tensor dtype".to_string()))?;

    let readonly = arr.readonly();
    let src_view = readonly.as_array();

    let tensor_len = tensor.len();
    if src_view.len() != tensor_len {
        return Err(Error::Format(format!(
            "element count mismatch: numpy array has {} elements but tensor has {tensor_len}",
            src_view.len()
        )));
    }

    let mut map = tensor.map_for_write()?;
    let dst = map.as_mut_slice();
    let dst_len = dst.len();
    let nbytes = tensor_len * std::mem::size_of::<T>();
    let parallel = nbytes >= PARALLEL_THRESHOLD_BYTES;
    // Minimum chunk: 4 KiB worth of elements.
    let min_chunk = (4096 / std::mem::size_of::<T>()).max(1);

    // A destination map longer than the logical element count has row padding:
    // copy row by row, skipping the padding bytes in the destination.
    if dst_len > tensor_len {
        return copy_numpy_padded(
            py,
            (arr, &readonly, &src_view),
            tensor,
            dst,
            (dst_len, tensor_len, parallel),
        );
    }

    if arr.is_c_contiguous() {
        copy_numpy_c_contiguous(
            py, &readonly, &src_view, dst, tensor_len, min_chunk, parallel,
        );
        return Ok(());
    }

    copy_numpy_noncontiguous(py, src, dst, tensor_len, min_chunk, parallel, &src_view)
}

fn copy_numpy_padded<
    T: numpy::Element
        + edgefirst_tensor::Element
        + num_traits::Num
        + Copy
        + Clone
        + std::fmt::Debug
        + Send
        + Sync,
>(
    py: Python<'_>,
    src: (
        &Bound<'_, numpy::PyArrayDyn<T>>,
        &numpy::PyReadonlyArrayDyn<'_, T>,
        &ndarray::ArrayViewD<'_, T>,
    ),
    tensor: &tensor::Tensor<T>,
    dst: &mut [T],
    padded: (usize, usize, bool),
) -> Result<()> {
    use numpy::PyUntypedArrayMethods;

    let (arr, readonly, src_view) = src;
    let (dst_len, tensor_len, parallel) = padded;

    let elem_sz = std::mem::size_of::<T>();
    let stride_bytes = tensor.effective_row_stride().ok_or_else(|| {
        Error::Format(format!(
            "destination buffer is padded ({dst_len} elems > {tensor_len} logical) \
             but tensor has no effective_row_stride"
        ))
    })?;
    if elem_sz == 0 || stride_bytes == 0 {
        return Ok(());
    }
    // A strided map spans `rows * stride` bytes; `rows` is derived from the
    // mapping.
    if !stride_bytes.is_multiple_of(elem_sz) {
        return Err(Error::Format(format!(
            "stride-padded copy: row pitch {stride_bytes} B is not a whole number of \
             {elem_sz}-byte elements, so rows cannot be addressed in this dtype"
        )));
    }
    let dst_stride_elems = stride_bytes / elem_sz;
    let height = dst_len / dst_stride_elems;
    let row_elems = tensor_len.checked_div(height).unwrap_or(0);

    if height == 0
        || dst_stride_elems * height != dst_len
        || row_elems * height != tensor_len
        || row_elems > dst_stride_elems
    {
        return Err(Error::Format(format!(
            "stride-padded copy: inconsistent dimensions: \
             dst_len={dst_len}, tensor_len={tensor_len}, rows={height}, \
             dst_stride_elems={dst_stride_elems}, row_elems={row_elems}"
        )));
    }

    if arr.is_c_contiguous() {
        if let Ok(src_slice) = readonly.as_slice() {
            py.detach(|| {
                if parallel {
                    use rayon::prelude::*;
                    dst.par_chunks_mut(dst_stride_elems)
                        .zip(src_slice.par_chunks(row_elems))
                        .for_each(|(d, s)| d[..row_elems].copy_from_slice(s));
                } else {
                    for row in 0..height {
                        let s = row * row_elems;
                        let d = row * dst_stride_elems;
                        dst[d..d + row_elems].copy_from_slice(&src_slice[s..s + row_elems]);
                    }
                }
            });
        } else {
            py.detach(|| {
                let mut it = src_view.iter();
                for row in 0..height {
                    let d = row * dst_stride_elems;
                    for col in 0..row_elems {
                        dst[d + col] = *it.next().unwrap();
                    }
                }
            });
        }
    } else {
        py.detach(|| {
            let mut it = src_view.iter();
            for row in 0..height {
                let d = row * dst_stride_elems;
                for col in 0..row_elems {
                    dst[d + col] = *it.next().unwrap();
                }
            }
        });
    }
    Ok(())
}

fn copy_numpy_c_contiguous<T: numpy::Element + Copy + Send + Sync>(
    py: Python<'_>,
    readonly: &numpy::PyReadonlyArrayDyn<'_, T>,
    src_view: &ndarray::ArrayViewD<'_, T>,
    dst: &mut [T],
    tensor_len: usize,
    min_chunk: usize,
    parallel: bool,
) {
    if let Ok(src_slice) = readonly.as_slice() {
        // Path 1: fully contiguous — single memcpy.
        py.detach(|| {
            if parallel {
                use rayon::prelude::*;
                let chunk = (tensor_len / rayon::current_num_threads()).max(min_chunk);
                dst.par_chunks_mut(chunk)
                    .zip(src_slice.par_chunks(chunk))
                    .for_each(|(d, s): (&mut [T], &[T])| d.copy_from_slice(s));
            } else {
                dst.copy_from_slice(src_slice);
            }
        });
    } else {
        // C-contiguous but `as_slice()` returned `None`: element-wise copy.
        py.detach(|| {
            for (d, &s) in dst.iter_mut().zip(src_view.iter()) {
                *d = s;
            }
        });
    }
}

fn copy_numpy_noncontiguous<
    T: numpy::Element
        + edgefirst_tensor::Element
        + num_traits::Num
        + Copy
        + Clone
        + std::fmt::Debug
        + Send
        + Sync,
>(
    py: Python<'_>,
    src: &Bound<'_, pyo3::types::PyAny>,
    dst: &mut [T],
    tensor_len: usize,
    min_chunk: usize,
    parallel: bool,
    src_view: &ndarray::ArrayViewD<'_, T>,
) -> Result<()> {
    // Non-contiguous: find the longest contiguous inner dimension.
    // Walk inward from the last axis: if stride[i] == product of
    // shape[i+1..] (in elements), that axis and all inner axes form
    // a contiguous row.
    let shape = src_view.shape();
    let strides = src_view.strides(); // in elements (ndarray convention)
    let ndim = shape.len();

    let mut contig_elems: usize = 1;
    let mut contig_dims: usize = 0;
    for i in (0..ndim).rev() {
        // Size-1 dims count as contiguous.
        if strides[i] == contig_elems as isize || shape[i] <= 1 {
            contig_elems *= shape[i];
            contig_dims += 1;
        } else {
            break;
        }
    }

    if contig_elems > 1 && contig_elems < tensor_len {
        copy_numpy_strided_rows(
            py,
            src_view,
            dst,
            (tensor_len, contig_elems, contig_dims, ndim),
            (shape, strides),
            parallel,
        );
        return Ok(());
    }

    copy_numpy_via_contiguous_array(py, src, dst, tensor_len, min_chunk, parallel)
}

fn copy_numpy_strided_rows<T: Copy + Send + Sync>(
    py: Python<'_>,
    src_view: &ndarray::ArrayViewD<'_, T>,
    dst: &mut [T],
    contig: (usize, usize, usize, usize),
    layout: (&[usize], &[isize]),
    parallel: bool,
) {
    let (tensor_len, contig_elems, contig_dims, ndim) = contig;
    let (shape, strides) = layout;
    // Path 2: strided outer, contiguous inner rows. Row byte offsets are
    // computed from the strides.
    let n_rows = tensor_len / contig_elems;
    let row_len = contig_elems;
    let elem_size = std::mem::size_of::<T>() as isize;

    // Outer dimensions are those not part of the contiguous tail.
    let outer_ndim = ndim - contig_dims;
    let outer_shape = &shape[..outer_ndim];
    let outer_strides = &strides[..outer_ndim];

    // Compute the signed byte offset for each row by decomposing
    // the row index into a multi-index over the outer dimensions
    // and taking the dot product with their byte strides.
    let mut row_offsets: Vec<isize> = Vec::with_capacity(n_rows);
    for row_idx in 0..n_rows {
        let mut remaining = row_idx;
        let mut byte_off: isize = 0;
        for dim in (0..outer_ndim).rev() {
            let coord = remaining % outer_shape[dim];
            remaining /= outer_shape[dim];
            byte_off += coord as isize * outer_strides[dim] * elem_size;
        }
        row_offsets.push(byte_off);
    }

    // The base address is stored as `usize` and rebuilt into a pointer inside
    // each task.
    let base_addr = src_view.as_ptr() as usize;

    py.detach(|| {
        let copy_row = |dst_row: &mut [T], byte_off: isize| unsafe {
            let src_ptr = (base_addr as *const u8).offset(byte_off) as *const T;
            let src_row = std::slice::from_raw_parts(src_ptr, row_len);
            dst_row.copy_from_slice(src_row);
        };

        if parallel {
            use rayon::prelude::*;
            dst.par_chunks_mut(row_len)
                .zip(row_offsets.par_iter())
                .for_each(|(dst_row, &off)| copy_row(dst_row, off));
        } else {
            for (dst_row, &off) in dst.chunks_mut(row_len).zip(row_offsets.iter()) {
                copy_row(dst_row, off);
            }
        }
    });
}

fn copy_numpy_via_contiguous_array<T: numpy::Element + Copy + Send + Sync>(
    py: Python<'_>,
    src: &Bound<'_, pyo3::types::PyAny>,
    dst: &mut [T],
    tensor_len: usize,
    min_chunk: usize,
    parallel: bool,
) -> Result<()> {
    use numpy::PyArrayMethods;

    // Path 3: fully strided (`contig_elems == 1`). Materialize a contiguous
    // source via `np.ascontiguousarray()`, then copy as in path 1.
    let np = py
        .import("numpy")
        .map_err(|e| Error::Format(format!("failed to import numpy: {e}")))?;
    let contig_obj = np
        .call_method1("ascontiguousarray", (src,))
        .map_err(|e| Error::Format(format!("np.ascontiguousarray failed: {e}")))?;
    let contig_arr = contig_obj
        .cast::<numpy::PyArrayDyn<T>>()
        .map_err(|_| Error::Format("np.ascontiguousarray returned the wrong dtype".to_string()))?;
    let contig_readonly = contig_arr.readonly();
    let contig_slice = contig_readonly.as_slice().map_err(|_| {
        Error::Format("np.ascontiguousarray result is not contiguous (unexpected)".to_string())
    })?;

    py.detach(|| {
        if parallel {
            use rayon::prelude::*;
            let chunk = (tensor_len / rayon::current_num_threads()).max(min_chunk);
            dst.par_chunks_mut(chunk)
                .zip(contig_slice.par_chunks(chunk))
                .for_each(|(d, s): (&mut [T], &[T])| d.copy_from_slice(s));
        } else {
            dst.copy_from_slice(contig_slice);
        }
    });
    Ok(())
}

// ─── PyTensor ───────────────────────────────────────────────────────────────

/// A typed N-dimensional buffer. `edgefirst.codec.Frame` subclasses it.
#[pyclass(name = "Tensor", str, subclass, module = "edgefirst.tensor")]
pub struct PyTensor(pub(crate) TensorDyn);

impl Display for PyTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Tensor(dtype={}, shape={:?}, memory={:?})",
            dtype_to_str(self.0.dtype()),
            self.0.shape(),
            self.0.memory(),
        )
    }
}

#[pymethods]
impl PyTensor {
    #[new]
    #[pyo3(signature = (shape, dtype = "float32", mem = None, name = None))]
    fn __init__(
        shape: Vec<usize>,
        dtype: &str,
        mem: Option<PyTensorMemory>,
        name: Option<&str>,
    ) -> Result<Self> {
        let dt = parse_dtype(dtype)?;
        let memory = mem.map(Into::into);
        let tensor = TensorDyn::new(&shape, dt, memory, name)?;
        Ok(PyTensor(tensor))
    }

    /// Import an existing buffer as a tensor, without copying.
    ///
    /// Platforms:
    ///     Linux, macOS.
    ///
    /// The buffer type is detected. On Linux a `dma_buf` fd imports as
    /// `TensorMemory.DMABUF` and a tmpfs fd (`/dev/shm` or `memfd`) as
    /// `TensorMemory.SHM`, decided by filesystem magic; any other filesystem
    /// raises `RuntimeError`. On macOS the fd is always imported as SHM.
    /// `tensor.memory` reports the detected type.
    ///
    /// The fd is duplicated; the caller retains ownership of the original and
    /// must close it.
    #[cfg(unix)]
    #[staticmethod]
    #[pyo3(signature = (fd, shape, dtype = "float32", name = None))]
    fn from_fd(fd: RawFd, shape: Vec<usize>, dtype: &str, name: Option<&str>) -> Result<Self> {
        use std::os::fd::BorrowedFd;
        if fd < 0 {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "Invalid file descriptor",
            )));
        }
        let dt = parse_dtype(dtype)?;
        // Duplicate the fd; the caller keeps the original.
        let borrowed = unsafe { BorrowedFd::borrow_raw(fd) };
        let fd = borrowed.try_clone_to_owned()?;
        let tensor = TensorDyn::from_fd(fd, &shape, dt, name)?;
        Ok(PyTensor(tensor))
    }

    /// Wrap an externally-allocated IOSurface as a Tensor.
    ///
    /// Platforms:
    ///     macOS.
    ///
    /// `surface_ref` is an `IOSurfaceRef` cast to `int`, e.g. obtained via
    /// `ctypes` from a CoreVideo, AVFoundation or VideoToolbox handle, or via
    /// `IOSurfaceLookup(id)`. The surface is retained for the tensor's
    /// lifetime; the caller keeps its own reference.
    ///
    /// Raises RuntimeError on non-macOS platforms.
    #[cfg(target_os = "macos")]
    #[staticmethod]
    #[pyo3(signature = (surface_ref, shape, dtype = "uint8", name = None))]
    fn from_iosurface(
        surface_ref: usize,
        shape: Vec<usize>,
        dtype: &str,
        name: Option<&str>,
    ) -> Result<Self> {
        if surface_ref == 0 {
            return Err(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "surface_ref must be a non-zero IOSurfaceRef",
            )));
        }
        let dt = parse_dtype(dtype)?;
        let tensor = unsafe {
            TensorDyn::from_iosurface(surface_ref as *mut std::ffi::c_void, &shape, dt, name)?
        };
        Ok(PyTensor(tensor))
    }

    #[getter]
    fn dtype(&self) -> String {
        dtype_to_str(self.0.dtype()).to_string()
    }

    #[getter]
    fn size(&self) -> usize {
        self.0.size()
    }

    #[getter]
    fn memory(&self) -> PyTensorMemory {
        self.0.memory().into()
    }

    /// The vendor tile-compression scheme recorded at allocation
    /// (``"ubwc"``/``"afbc"``/``"pvric"``/``"dcc"``), or ``None`` for a
    /// linear layout. A compressed tensor has no meaningful linear row
    /// stride and CPU maps are best-effort.
    #[getter]
    fn compression(&self) -> Option<&'static str> {
        use edgefirst_tensor::CompressionScheme;
        match self.0.compression()? {
            CompressionScheme::Ubwc => Some("ubwc"),
            CompressionScheme::Afbc => Some("afbc"),
            CompressionScheme::Pvric => Some("pvric"),
            CompressionScheme::Dcc => Some("dcc"),
            _ => None,
        }
    }

    /// Whether the memory is known to be physically contiguous:
    /// ``"contiguous"`` for a DMA-BUF allocated from the CMA heap,
    /// ``"non_contiguous"`` for one from the system heap, and ``"unknown"``
    /// for an imported fd and every other kind of memory.
    #[getter]
    fn contiguity(&self) -> &'static str {
        self.0.contiguity().as_str()
    }

    /// The CPU access declared at allocation: ``"none"``, ``"read"``,
    /// ``"write"`` or ``"readwrite"``.
    #[getter]
    fn declared_cpu_access(&self) -> &'static str {
        self.0.cpu_access().as_str()
    }

    #[getter]
    fn name(&self) -> String {
        self.0.name()
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.0.shape().to_vec()
    }

    /// A duplicate of the file descriptor backing the tensor's memory.
    ///
    /// Platforms:
    ///     Linux, macOS.
    ///
    /// The caller owns the returned descriptor and must close it.
    #[cfg(unix)]
    #[getter]
    fn fd(&self) -> Result<RawFd> {
        let owned = self.0.clone_fd()?;
        Ok(owned.into_raw_fd())
    }

    fn reshape(&mut self, shape: Vec<usize>) -> Result<()> {
        Ok(self.0.reshape(&shape)?)
    }

    /// Zero-copy rectangular sub-region view of an image tensor.
    ///
    /// ``region`` is in pixels of the image's leading frame. The returned view
    /// shares the parent's buffer and addresses the sub-rectangle by offset and
    /// the parent's row pitch. ``convert(src, dst.view(region), ...)`` renders
    /// into that sub-rectangle. The parent must be a packed-format image tensor.
    ///
    /// The view supports :meth:`map`, :meth:`from_numpy` and ``fd`` like any
    /// other tensor.
    #[pyo3(signature = (region))]
    fn view(&self, region: PyRegion) -> Result<Self> {
        let view = self.0.view(region.into())?;
        Ok(PyTensor(view))
    }

    /// Borrow batch element ``n`` of a batched tensor as a zero-copy view.
    ///
    /// A batched tensor prepends ``N`` as the leading dimension over the
    /// per-element image layout (``[N, H, W, C]`` packed or ``[N, C, H, W]``
    /// planar). ``batch(n)`` returns element ``n``: the contiguous per-element
    /// region at byte offset ``n * element_size``, sharing the parent's buffer.
    /// ``batch(0)`` on a tensor with ``N == 1`` is equivalent to the whole
    /// tensor.
    #[pyo3(signature = (n))]
    fn batch(&self, n: usize) -> Result<Self> {
        let view = self.0.batch(n)?;
        Ok(PyTensor(view))
    }

    /// Attach pixel format metadata to this tensor.
    ///
    /// Validates that the tensor's shape is compatible with the format's
    /// layout (packed, planar, or semi-planar).
    fn set_format(&mut self, format: PyPixelFormat) -> Result<()> {
        use edgefirst_tensor::PixelFormat;
        let fmt: PixelFormat = format.into();
        Ok(self.0.set_format(fmt)?)
    }

    /// Set this tensor's logical dimensions and pixel format to a decoded
    /// image, reusing the existing allocation.
    ///
    /// The logical shape can shrink or grow within the tensor's allocated
    /// capacity.
    ///
    /// Raises if the allocation cannot hold ``width``x``height`` in
    /// ``format``, or if the dimensions are invalid for the format.
    fn configure_image(
        &mut self,
        width: usize,
        height: usize,
        format: PyPixelFormat,
    ) -> Result<()> {
        use edgefirst_tensor::PixelFormat;
        let fmt: PixelFormat = format.into();
        Ok(self.0.configure_image(width, height, fmt)?)
    }

    /// Clone the DMA-BUF file descriptor backing this tensor.
    ///
    /// Platforms:
    ///     Linux.
    ///
    /// Returns a new file descriptor that the caller must close.
    ///
    /// Raises RuntimeError if the tensor is not DMA-backed or if the
    /// fd clone syscall fails.
    #[cfg(target_os = "linux")]
    fn dmabuf_clone(&self) -> Result<RawFd> {
        let owned = self.0.dmabuf_clone()?;
        Ok(owned.into_raw_fd())
    }

    /// IOSurfaceID for cross-process surface sharing.
    ///
    /// Platforms:
    ///     macOS.
    ///
    /// Returns None when the tensor is not IOSurface-backed. The ID is
    /// a 32-bit handle stable for the lifetime of the IOSurface; it
    /// can be passed across process boundaries and recovered via
    /// `IOSurfaceLookup(id)`.
    #[cfg(target_os = "macos")]
    #[getter]
    fn iosurface_id(&self) -> Option<u32> {
        self.0.iosurface_id()
    }

    /// Borrowed `IOSurfaceRef` as an integer.
    ///
    /// Platforms:
    ///     macOS.
    ///
    /// Passed to native macOS APIs that take an IOSurfaceRef (CIImage,
    /// AVSampleBufferDisplayLayer, CVPixelBufferCreateWithIOSurface). Wrap
    /// with `ctypes.c_void_p(...)` before passing to a ctypes-bound C
    /// function. The pointer's lifetime is tied to this tensor; do not call
    /// CFRelease on it.
    ///
    /// Returns None when the tensor is not IOSurface-backed.
    #[cfg(target_os = "macos")]
    #[getter]
    fn iosurface_ref(&self) -> Option<usize> {
        self.0.iosurface_ref().map(|p| p as usize)
    }

    /// Borrowed ``ID3D11Texture2D*`` as an integer.
    ///
    /// Platforms:
    ///     Windows.
    ///
    /// Wrap with ``ctypes.c_void_p`` for native callers; the pointer lives
    /// as long as this tensor. ``None`` when the tensor is not a texture.
    #[cfg(target_os = "windows")]
    fn d3d11_texture(&self) -> Option<usize> {
        self.0.d3d11_texture().map(|p| p as usize)
    }

    /// The texture's DXGI format and geometry.
    ///
    /// Platforms:
    ///     Windows.
    ///
    /// ``None`` when the tensor is not a texture.
    #[cfg(target_os = "windows")]
    fn d3d11_layout(&self) -> Option<PyD3d11Layout> {
        self.0.d3d11_layout().map(PyD3d11Layout::from)
    }

    /// Duplicate the NT handle of the D3D11 texture backing this tensor.
    ///
    /// Platforms:
    ///     Windows.
    ///
    /// Returns:
    ///     An NT handle the caller owns and must close with ``CloseHandle``.
    ///
    /// Raises:
    ///     RuntimeError: If the tensor is not a D3D11 texture, or the handle
    ///         could not be duplicated.
    #[cfg(target_os = "windows")]
    fn d3d11_shared_handle(&self) -> Result<usize> {
        use std::os::windows::io::IntoRawHandle;
        Ok(self.0.d3d11_shared_handle()?.into_raw_handle() as usize)
    }

    /// Fence handle and value a GPU consumer waits on before reading.
    ///
    /// Platforms:
    ///     Windows.
    ///
    /// Returns:
    ///     ``(fence_handle, value)``, the handle owned by the caller and
    ///     closed with ``CloseHandle``; ``None`` when no GPU write has been
    ///     recorded.
    #[cfg(target_os = "windows")]
    fn gpu_completion(&self) -> Result<Option<(usize, u64)>> {
        use std::os::windows::io::IntoRawHandle;
        Ok(self
            .0
            .gpu_completion()?
            .map(|c| (c.fence.into_raw_handle() as usize, c.value)))
    }

    /// The fence value of the newest recorded GPU write, or 0.
    ///
    /// Platforms:
    ///     Windows.
    ///
    /// The same number :meth:`gpu_completion` returns as ``value``. 0 when no
    /// write has been recorded, or when the tensor is not a texture.
    #[cfg(target_os = "windows")]
    #[getter]
    fn gpu_write_value(&self) -> u64 {
        self.0.gpu_write_value()
    }

    /// Record that GPU work writing this texture completes at ``value``.
    ///
    /// Platforms:
    ///     Windows.
    ///
    /// The recorded value is the maximum of the values set.
    ///
    /// Raises:
    ///     RuntimeError: If the tensor is not a D3D11 texture.
    #[cfg(target_os = "windows")]
    fn set_gpu_write(&self, value: u64) -> Result<()> {
        Ok(self.0.set_gpu_write(value)?)
    }

    /// Wrap a texture created on the HAL device.
    ///
    /// Platforms:
    ///     Windows.
    ///
    /// The width and height are read from the texture's own description.
    /// ``shape`` must be either the allocation shape or the addressing shape
    /// ``format`` gives those dimensions.
    ///
    /// A semi-planar texture (``Nv12``, ``Nv16``, ``Nv24``) carries the
    /// image's row stride as its width. It is accepted only when that width is
    /// even and equal to the staging row pitch the driver gives a texture of
    /// that width.
    ///
    /// Raises:
    ///     ValueError: If ``shape`` is neither shape of this texture; the
    ///         message names both.
    ///     RuntimeError: If ``texture`` is not a live texture on the HAL
    ///         device holding a ``format`` image.
    #[cfg(target_os = "windows")]
    #[staticmethod]
    #[pyo3(signature = (texture, shape, dtype, format, access = "readwrite", name = None))]
    fn from_d3d11_texture(
        texture: usize,
        shape: Vec<usize>,
        dtype: &str,
        format: PyPixelFormat,
        access: &str,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let fmt: PixelFormat = format.into();
        let texture = texture as *mut c_void;
        // SAFETY: `texture` is a live `ID3D11Texture2D*` per the caller's
        // contract; the helper takes and drops its own reference.
        let geometry =
            unsafe { edgefirst_tensor::d3d11_texture_geometry(texture, fmt, Some(&shape)) };
        let (width, height) = check_d3d11_shape(geometry, fmt, &shape, "from_d3d11_texture")?;
        // SAFETY: `texture` is a live texture holding this image; the
        // constructor takes its own reference.
        let tensor = unsafe {
            TensorDyn::from_d3d11_texture(
                texture,
                width,
                height,
                fmt,
                parse_dtype(dtype)?,
                parse_cpu_access(access)?,
                name,
            )
        }
        .map_err(Error::from)?;
        Ok(PyTensor(tensor))
    }

    /// Open a shared texture on the HAL device, waiting on ``completion``
    /// first.
    ///
    /// Platforms:
    ///     Windows.
    ///
    /// ``handle`` and ``completion``'s fence handle stay the caller's; the
    /// tensor duplicates what it keeps. The width and height are read from the
    /// opened texture's own description. ``shape`` must be either the
    /// allocation shape or the addressing shape ``format`` gives those
    /// dimensions. The semi-planar width rule of :meth:`from_d3d11_texture`
    /// applies.
    ///
    /// Args:
    ///     completion: ``(fence_handle, value)`` from
    ///         :meth:`gpu_completion` on the producing side, waited on
    ///         before the texture is read.
    ///
    /// Raises:
    ///     ValueError: If ``shape`` is neither shape of this texture; the
    ///         message names both.
    ///     RuntimeError: If ``handle`` does not open a texture holding a
    ///         ``format`` image, or the fence wait fails.
    #[cfg(target_os = "windows")]
    #[staticmethod]
    #[pyo3(signature = (handle, shape, dtype, format, access = "readwrite", completion = None, name = None))]
    fn from_d3d11_shared_handle(
        handle: usize,
        shape: Vec<usize>,
        dtype: &str,
        format: PyPixelFormat,
        access: &str,
        completion: Option<(usize, u64)>,
        name: Option<&str>,
    ) -> PyResult<Self> {
        let fmt: PixelFormat = format.into();
        let handle = handle as *mut c_void;
        // SAFETY: `handle` is an NT shared handle of a D3D11 texture valid in
        // this process per the caller's contract; the helper drops the texture it
        // opens before returning.
        let geometry =
            unsafe { edgefirst_tensor::d3d11_shared_handle_geometry(handle, fmt, Some(&shape)) };
        let (width, height) = check_d3d11_shape(geometry, fmt, &shape, "from_d3d11_shared_handle")?;
        let completion = completion.map(|(fence, value)| (fence as *mut c_void, value));
        // SAFETY: `handle` is as above and `completion`'s handle is a shared
        // fence handle per the caller's contract; the constructor duplicates what
        // it keeps.
        let tensor = unsafe {
            TensorDyn::from_d3d11_shared_handle(
                handle,
                width,
                height,
                fmt,
                parse_dtype(dtype)?,
                parse_cpu_access(access)?,
                completion,
                name,
            )
        }
        .map_err(Error::from)?;
        Ok(PyTensor(tensor))
    }

    /// Producer half of the cross-package tensor protocol.
    ///
    /// Returns a ``PyCapsule`` named ``edgefirst_tensor_v2`` wrapping a
    /// ``TensorDesc`` and this tensor's quantization metadata.
    ///
    /// Host memory is addressable through the descriptor only when the tensor
    /// has been pinned; the capsule holds the pin for as long as the consumer
    /// holds the capsule.
    ///
    /// The call has no side effects and may be repeated with the same or a
    /// different ``access``.
    ///
    /// Args:
    ///     access: ``None`` (default) requests no pin; the descriptor carries
    ///         shape, format and the native handle. ``"read"``, ``"write"`` or
    ///         ``"readwrite"`` pins host memory with the matching access and
    ///         fills in the descriptor's address.
    ///
    /// Raises:
    ///     ValueError: if ``access`` is not one of the values above.
    ///     NotImplementedError: if the tensor's chroma plane is a separate
    ///         tensor (two-buffer NV12, chroma at an offset, a decoded video
    ///         frame), which cannot be passed to another edgefirst package.
    #[pyo3(signature = (access=None))]
    fn __edgefirst_tensor__<'py>(
        &self,
        py: Python<'py>,
        access: Option<&str>,
    ) -> PyResult<Bound<'py, pyo3::types::PyCapsule>> {
        use edgefirst_tensor::CpuAccess;

        crate::interop::exportable(&self.0)?;

        // Pin only for the access the caller asked for.
        let pin = match access.map(|s| (s, CpuAccess::from_str_code(s))) {
            None => None,
            Some((_, Some(a))) if a != CpuAccess::None => {
                Some(self.0.pin_host(a).map_err(Error::from)?)
            }
            // Any other string, including "none", raises `ValueError`.
            Some((other, _)) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "access must be None, \"read\", \"write\" or \"readwrite\", got {other:?}"
                )));
            }
        };
        let desc = self.0.descriptor_pinned(pin.as_ref());

        // The capsule owns the descriptor, the pin, the PBO keepalive and an
        // `Arc` of the quantization that `quant` borrows; dropping the capsule
        // releases them.
        let quant_keepalive = self.0.quantization().cloned().map(std::sync::Arc::new);
        let payload = crate::interop::TensorCapsulePayload {
            desc,
            quant: quant_keepalive
                .as_deref()
                .map_or_else(crate::interop::QuantDesc::absent, |q| {
                    crate::interop::QuantDesc::borrowing(q)
                }),
            // A `view()`'s byte offset into the parent buffer.
            plane_offset: self.0.plane_offset().unwrap_or(0) as u64,
            pin,
            pbo_keepalive: self.0.pbo_keepalive(),
            quant_keepalive,
        };
        pyo3::types::PyCapsule::new_with_value(py, payload, c"edgefirst_tensor_v2")
            .map_err(|e| Error::from(std::io::Error::other(e.to_string())).into())
    }

    /// Pin a stable host address for this tensor's data.
    ///
    /// The returned :class:`HostPin` carries no borrow of the tensor. The
    /// address stays valid until the pin is released.
    ///
    /// The pin gives no coherency guarantee: bracket CPU access with
    /// :meth:`cpu_access`.
    ///
    /// Args:
    ///     access: ``"read"``, ``"write"`` or ``"readwrite"``.
    ///
    /// Raises:
    ///     RuntimeError: if the backend cannot pin. PBO and AHardwareBuffer
    ///         have no host address outside their map/lock; use :meth:`map`
    ///         there.
    fn pin_host(slf: &Bound<'_, Self>, access: &str) -> Result<PyHostPin> {
        let pin = slf.borrow().0.pin_host(parse_cpu_access(access)?)?;
        Ok(PyHostPin {
            pin: Some(pin),
            _owner: Some(slf.clone().into_any().unbind()),
        })
    }

    /// Bracket CPU access to this tensor for coherency.
    ///
    /// Use as a context manager::
    ///
    ///     pin = t.pin_host("readwrite")
    ///     with t.cpu_access("write"):
    ///         ctypes.memset(pin.ptr, 0, pin.len)
    ///
    /// A no-op on backends that need no maintenance (``Mem``, ``Shm``). The
    /// same direction is applied on exit.
    ///
    /// Args:
    ///     access: ``"read"``, ``"write"`` or ``"readwrite"``.
    fn cpu_access(slf: Py<Self>, access: &str) -> Result<PyCpuAccessGuard> {
        Ok(PyCpuAccessGuard {
            tensor: slf,
            access: parse_cpu_access(access)?,
            active: false,
        })
    }

    /// Map the tensor for CPU access, returning a buffer-protocol view.
    ///
    /// Use as a context manager::
    ///
    ///     with tensor.map("read") as view:
    ///         frame = np.asarray(view)     # shape, dtype and strides carried
    ///
    /// The map owns its coherency bracket in both directions.
    ///
    /// `access` selects the direction of that bracket. ``"read"`` skips the
    /// cache writeback on unmap and, on macOS, takes the read-only IOSurface
    /// lock. A ``"read"`` view is advertised to the buffer protocol as
    /// read-only, so ``np.asarray`` of it is not writable.
    ///
    /// Args:
    ///     access: ``"read"``, ``"write"`` or ``"readwrite"`` (default).
    ///         ``"none"`` is rejected.
    #[pyo3(signature = (access=None))]
    fn map(slf: &Bound<'_, Self>, access: Option<&str>) -> Result<PyTensorMap> {
        let this = slf.borrow();
        let access = parse_cpu_access(access.unwrap_or("readwrite"))?;
        if access == tensor::CpuAccess::None {
            return Err(Error::Format(
                "map: access=\"none\" is not a mapping; a mapped view is CPU \
                 access by definition. Use \"read\", \"write\" or \"readwrite\""
                    .to_owned(),
            ));
        }
        // Capture the physical row pitch of an image tensor for the buffer
        // protocol's strides.
        let row_stride = this.0.format().and(this.0.effective_row_stride());
        let layout = this.0.format().map(|f| f.layout());
        let mapped = map_tensor_dyn(&this.0, access, MapWait::Block)?;
        Ok(PyTensorMap {
            readonly: !mapped.is_writable(),
            mapped: Some(mapped),
            row_stride,
            layout,
            _owner: Some(slf.clone().into_any().unbind()),
        })
    }

    /// Non-blocking :meth:`map`.
    ///
    /// Raises ``BlockingIOError`` while a GPU copy the map depends on is in
    /// flight; identical to :meth:`map` on backings without such a copy.
    ///
    /// A caller that retries after ``BlockingIOError`` must yield or sleep
    /// between attempts.
    ///
    /// Args:
    ///     access: as :meth:`map`.
    ///
    /// Raises:
    ///     BlockingIOError: While a GPU copy the map depends on is in
    ///         flight.
    #[pyo3(signature = (access = None))]
    fn try_map(slf: &Bound<'_, Self>, access: Option<&str>) -> PyResult<PyTensorMap> {
        let this = slf.borrow();
        let access = parse_cpu_access(access.unwrap_or("readwrite"))?;
        if access == tensor::CpuAccess::None {
            return Err(Error::Format(
                "try_map: access=\"none\" is not a mapping; a mapped view is CPU \
                 access by definition. Use \"read\", \"write\" or \"readwrite\""
                    .to_owned(),
            )
            .into());
        }
        let row_stride = this.0.format().and(this.0.effective_row_stride());
        let layout = this.0.format().map(|f| f.layout());
        let mapped = map_tensor_dyn(&this.0, access, MapWait::Fail).map_err(would_block_err)?;
        Ok(PyTensorMap {
            readonly: !mapped.is_writable(),
            mapped: Some(mapped),
            row_stride,
            layout,
            _owner: Some(slf.clone().into_any().unbind()),
        })
    }

    /// Attempt a zero-copy CUDA device-pointer mapping.
    ///
    /// Returns a ``CudaMap`` context manager exposing ``device_ptr`` and
    /// ``size``, or ``None`` if CUDA is unavailable for this tensor (libcudart
    /// not found, or the tensor was not registered with CUDA).
    ///
    /// Fall back to ``map()`` when it returns ``None``::
    ///
    ///     cm = tensor.cuda_map()
    ///     if cm is not None:
    ///         with cm as m:
    ///             trt_set_input_address(m.device_ptr)   # zero-copy GPU
    ///     else:
    ///         with tensor.map() as host:
    ///             run_cpu_path(host)                    # fallback
    ///
    /// On Windows the mapping of a D3D11 texture tensor is tight rows of
    /// ``width * bytes_per_texel``, and ``size`` is their sum, independent of
    /// :attr:`row_stride`. A consumer must not stride the device pointer by
    /// :attr:`row_stride`.
    ///
    /// A semi-planar texture (NV12, NV16, NV24) is as wide as its own pitch;
    /// its mapping is :attr:`row_stride` times the combined height.
    fn cuda_map(slf: Bound<'_, Self>) -> Option<PyCudaMap> {
        let owner: Py<PyTensor> = slf.clone().unbind();
        // SAFETY: the mapping borrows the tensor's storage; `_owner` keeps the
        // `PyTensor` alive and is dropped after `map`.
        let borrowed = slf.borrow();
        let map = borrowed.0.cuda_map()?;
        let map_static: edgefirst_tensor::CudaMap<'static> = unsafe { std::mem::transmute(map) };
        Some(PyCudaMap {
            map: Some(map_static),
            _owner: owner,
        })
    }

    /// Writable counterpart of :meth:`cuda_map`.
    ///
    /// Changes are written back to the tensor when the mapping is released.
    /// Returns ``None`` on the same terms as :meth:`cuda_map`. For a Windows
    /// D3D11 texture the release copies the device buffer back into the
    /// texture.
    ///
    /// On Windows the mapping of a D3D11 texture tensor is tight rows of
    /// ``width * bytes_per_texel``, and ``size`` is their sum, independent of
    /// :attr:`row_stride`. A consumer must not stride the device pointer by
    /// :attr:`row_stride`.
    ///
    /// A semi-planar texture (NV12, NV16, NV24) is as wide as its own pitch;
    /// its mapping is :attr:`row_stride` times the combined height.
    fn cuda_map_mut(slf: Bound<'_, Self>) -> Option<PyCudaMap> {
        let owner: Py<PyTensor> = slf.clone().unbind();
        // SAFETY: the mapping borrows the tensor's storage; `_owner` keeps the
        // `PyTensor` alive and is dropped after `map`.
        let borrowed = slf.borrow();
        let map = borrowed.0.cuda_map_mut()?;
        let map_static: edgefirst_tensor::CudaMap<'static> = unsafe { std::mem::transmute(map) };
        Some(PyCudaMap {
            map: Some(map_static),
            _owner: owner,
        })
    }

    // ── Image-specific methods ──────────────────────────────────────────

    /// Create an image tensor with the given dimensions and pixel format.
    ///
    /// `access` declares CPU access (`"readwrite"`, the default, `"read"`,
    /// `"write"` for whole-buffer writes, or `"none"`); hardware access is
    /// always implied. A CPU map outside the declaration raises, as does HAL
    /// CPU work it does not cover. On Linux a `"none"` or `"write"` DMA-BUF
    /// comes from the uncached CMA heap where available.
    ///
    /// `contiguous=True` requires physically contiguous memory: a DMA-BUF
    /// from the Linux CMA heap, or an error naming the heap (see
    /// :attr:`contiguity`).
    #[staticmethod]
    #[pyo3(signature = (width, height, format, mem = None, access = "readwrite", contiguous = false))]
    fn image(
        width: usize,
        height: usize,
        format: PyPixelFormat,
        mem: Option<PyTensorMemory>,
        access: &str,
        contiguous: bool,
    ) -> Result<Self> {
        use edgefirst_tensor::PixelFormat;
        let fmt: PixelFormat = format.into();
        let memory = mem.map(Into::into);
        if contiguous {
            let desc = edgefirst_tensor::ImageDesc::new(width, height, fmt, DType::U8)
                .with_memory(memory)
                .with_access(parse_cpu_access(access)?)
                .with_contiguous(true);
            return Ok(PyTensor(TensorDyn::image_desc(&desc)?));
        }
        let tensor = TensorDyn::image(
            width,
            height,
            fmt,
            DType::U8,
            memory,
            parse_cpu_access(access)?,
        )?;
        Ok(PyTensor(tensor))
    }

    /// Parse the header of a JPEG/PNG byte string and return its native
    /// dimensions, pixel format, and EXIF orientation without decoding
    /// pixels.
    ///
    /// The reported dimensions are unrotated; ``rotation_degrees`` /
    /// ``flip_horizontal`` report the EXIF orientation.
    #[cfg(feature = "codec")]
    #[staticmethod]
    fn peek_image_info(data: &[u8]) -> Result<PyImageInfo> {
        use edgefirst_codec::peek_info;
        let info = peek_info(data)?;
        Ok(PyImageInfo {
            width: info.width,
            height: info.height,
            format: PyPixelFormat::try_from(info.format)
                .map_err(|e| Error::Format(e.to_string()))?,
            row_stride: info.row_stride,
            rotation_degrees: info.rotation_degrees,
            flip_horizontal: info.flip_horizontal,
        })
    }

    /// Parse the header of an image file and return its native dimensions,
    /// pixel format, and EXIF orientation without decoding pixels.
    #[cfg(feature = "codec")]
    #[staticmethod]
    fn peek_image_info_file(filename: &str) -> Result<PyImageInfo> {
        let data = std::fs::read(filename)?;
        Self::peek_image_info(&data)
    }

    /// Save this image tensor as a JPEG file.
    #[cfg(feature = "image")]
    #[pyo3(signature = (filename, quality=80))]
    fn save_jpeg(&self, filename: &str, quality: u8) -> Result<()> {
        edgefirst_image::save_jpeg(&self.0, filename, quality)?;
        Ok(())
    }

    /// Decode image bytes (JPEG/PNG) directly into this pre-allocated tensor.
    ///
    /// The image is decoded in its native pixel format (JPEG → ``Nv12`` for
    /// colour / ``Grey`` for greyscale; PNG → ``Rgb`` / ``Rgba`` / ``Grey``)
    /// and the tensor's dimensions and format are configured by the decoder
    /// to match. The decode never rotates or flips.
    ///
    /// Returns an ``ImageInfo`` with the native ``width``, ``height``,
    /// ``format``, ``row_stride``, and the EXIF ``rotation_degrees`` /
    /// ``flip_horizontal``. The tensor must have sufficient capacity for the
    /// decoded image (it is reconfigured within that capacity).
    ///
    /// :func:`set_output_format` selects a fused ``Rgb``/``Nv12`` output for a
    /// JPEG source.
    ///
    /// Args:
    ///     data: Raw JPEG or PNG bytes.
    #[cfg(feature = "codec")]
    fn decode_image<'py>(mut self_: PyRefMut<'py, Self>, data: &[u8]) -> Result<PyImageInfo> {
        let py = self_.py();
        // Copy the input while the GIL is held.
        let data = data.to_vec();
        if self_.0.memory() == TensorMemory::Pbo {
            // A PBO-backed tensor is decoded in place with the GIL held.
            return decode_image_into(&mut self_.0, &data);
        }
        // Decode into a `TensorDyn` reconstructed over the same backing memory,
        // with the GIL released.
        let (mut tensor, _pin) =
            crate::interop::reconstruct(&self_.0, crate::interop::write_pin_access(&self_.0))
                .map_err(Error::Py)?;
        let self_bound: Py<Self> = self_.into_pyobject(py).unwrap().unbind();
        let (info, tensor, _pin) = py.detach(move || {
            decode_image_into(&mut tensor, &data).map(|info| (info, tensor, _pin))
        })?;
        write_back_decode(&mut self_bound.bind(py).borrow_mut().0, &tensor, &info)?;
        Ok(info)
    }

    /// Decode an image file (JPEG/PNG) directly into this pre-allocated tensor.
    ///
    /// Decodes in the source's native pixel format and configures the
    /// tensor's dimensions and format to match. Returns an ``ImageInfo``
    /// with the native ``width``, ``height``, ``format``, ``row_stride``,
    /// and the EXIF ``rotation_degrees`` / ``flip_horizontal``.
    ///
    /// Args:
    ///     filename: Path to the image file.
    #[cfg(feature = "codec")]
    fn decode_image_file<'py>(
        mut self_: PyRefMut<'py, Self>,
        filename: &str,
    ) -> Result<PyImageInfo> {
        let py = self_.py();
        // Copy the filename while the GIL is held.
        let filename = filename.to_string();
        if self_.0.memory() == TensorMemory::Pbo {
            // A PBO-backed tensor is decoded in place with the GIL held.
            let data = std::fs::read(&filename).map_err(Error::Io)?;
            return decode_image_into(&mut self_.0, &data);
        }
        // Decode into a `TensorDyn` reconstructed over the same backing memory,
        // with the GIL released.
        let (mut tensor, _pin) =
            crate::interop::reconstruct(&self_.0, crate::interop::write_pin_access(&self_.0))
                .map_err(Error::Py)?;
        let self_bound: Py<Self> = self_.into_pyobject(py).unwrap().unbind();
        let (info, tensor, _pin) = py.detach(move || {
            let data = std::fs::read(&filename).map_err(Error::Io)?;
            decode_image_into(&mut tensor, &data).map(|info| (info, tensor, _pin))
        })?;
        write_back_decode(&mut self_bound.bind(py).borrow_mut().0, &tensor, &info)?;
        Ok(info)
    }

    /// Pixel format of this tensor (None if not an image tensor).
    #[getter]
    fn format(&self) -> Option<PyPixelFormat> {
        self.0
            .format()
            .and_then(|f| PyPixelFormat::try_from(f).ok())
    }

    /// Image width in pixels (None if not an image tensor).
    #[getter]
    fn width(&self) -> Option<usize> {
        self.0.width()
    }

    /// Image height in pixels (None if not an image tensor).
    #[getter]
    fn height(&self) -> Option<usize> {
        self.0.height()
    }

    /// Effective row stride in bytes, or ``None`` if unknown.
    ///
    /// For images allocated via ``ImageProcessor.create_image``, this
    /// includes any DMA pitch-alignment padding applied to the row.
    /// Returns the explicit stride when set, otherwise computes
    /// ``width × channels × sizeof(element)`` from the pixel format.
    /// Returns ``None`` for non-image tensors without a pixel format.
    #[getter]
    fn row_stride(&self) -> Option<usize> {
        self.0.format().and(self.0.effective_row_stride())
    }

    /// Whether this image uses a planar pixel layout.
    #[getter]
    fn is_planar(&self) -> bool {
        use edgefirst_tensor::PixelLayout;
        self.0
            .format()
            .map(|f| f.layout() == PixelLayout::Planar)
            .unwrap_or(false)
    }

    /// Normalize image data and write to a numpy array.
    ///
    /// ``normalization`` is a ``NormalizationPreset`` (or its name):
    /// ``Unsigned`` (``x / 255``), ``Signed`` (``(x - zp) / 127.5``, or
    /// ``x - zp`` into ``int8``) or ``Raw``. ``None`` is ``Raw`` for a
    /// ``uint8`` destination and ``Signed`` otherwise.
    #[cfg(feature = "image")]
    #[pyo3(signature = (dst, normalization=None, zero_point=None))]
    fn normalize_to_numpy(
        &self,
        dst: crate::image::ImageDest3,
        normalization: Option<crate::image::NumpyNormalizationArg>,
        zero_point: Option<i64>,
    ) -> Result<()> {
        let normalization = crate::image::NumpyNormalizationArg::resolve(normalization)?;
        Ok(crate::image::normalize_tensor_to_numpy(
            &self.0,
            dst,
            normalization,
            zero_point,
        )?)
    }

    /// Copy data from a numpy array into this tensor.
    ///
    /// Accepts any numpy dtype as long as it matches the tensor's dtype.
    /// The total element count must match. Both contiguous and
    /// non-contiguous (strided) arrays are supported. Large copies
    /// (≥256 KiB) are parallelized automatically.
    ///
    /// Raises ``RuntimeError`` on dtype mismatch or element-count mismatch.
    #[allow(clippy::wrong_self_convention)]
    fn from_numpy(&mut self, src: &Bound<'_, pyo3::types::PyAny>) -> Result<()> {
        copy_numpy_to_tensor_dyn(src, &self.0)
    }

    /// Quantization metadata, or ``None`` for float tensors and
    /// integer tensors without quantization attached.
    #[getter]
    fn quantization(&self) -> Option<PyQuantization> {
        self.0.quantization().cloned().map(PyQuantization)
    }

    /// Attach per-tensor asymmetric quantization. Integer tensors only.
    fn set_quantization_per_tensor(&mut self, scale: f32, zero_point: i32) -> Result<()> {
        let q = edgefirst_tensor::Quantization::per_tensor(scale, zero_point);
        Ok(self.0.set_quantization(q)?)
    }

    /// Attach per-tensor symmetric quantization. Integer tensors only.
    fn set_quantization_per_tensor_symmetric(&mut self, scale: f32) -> Result<()> {
        let q = edgefirst_tensor::Quantization::per_tensor_symmetric(scale);
        Ok(self.0.set_quantization(q)?)
    }

    /// Attach per-channel asymmetric quantization. Integer tensors only.
    /// Raises on ``len(scales) != len(zero_points)`` or invalid ``axis``.
    fn set_quantization_per_channel(
        &mut self,
        scales: Vec<f32>,
        zero_points: Vec<i32>,
        axis: usize,
    ) -> Result<()> {
        let q = edgefirst_tensor::Quantization::per_channel(scales, zero_points, axis)?;
        Ok(self.0.set_quantization(q)?)
    }

    /// Attach per-channel symmetric quantization. Integer tensors only.
    fn set_quantization_per_channel_symmetric(
        &mut self,
        scales: Vec<f32>,
        axis: usize,
    ) -> Result<()> {
        let q = edgefirst_tensor::Quantization::per_channel_symmetric(scales, axis)?;
        Ok(self.0.set_quantization(q)?)
    }

    /// Remove any quantization metadata from this tensor.
    fn clear_quantization(&mut self) {
        self.0.clear_quantization();
    }

    /// Colorimetry metadata, or ``None`` when undefined.
    ///
    /// ``ImageProcessor.convert`` tags an untagged YUV destination with the
    /// colorimetry it encoded. Set to ``None`` to clear.
    #[getter]
    fn colorimetry(&self) -> Option<crate::colorimetry::PyColorimetry> {
        self.0.colorimetry().map(Into::into)
    }

    #[setter]
    fn set_colorimetry(&mut self, colorimetry: Option<crate::colorimetry::PyColorimetry>) {
        self.0.set_colorimetry(colorimetry.map(Into::into));
    }
}

// ─── PyQuantization ─────────────────────────────────────────────────────────

/// Quantization parameters for an integer tensor.
///
/// Four modes are supported, matching the EdgeFirst model metadata spec:
/// per-tensor symmetric, per-tensor asymmetric, per-channel symmetric, and
/// per-channel asymmetric. Construct via the ``per_tensor`` /
/// ``per_tensor_symmetric`` / ``per_channel`` / ``per_channel_symmetric``
/// static methods.
#[pyclass(
    name = "Quantization",
    str,
    from_py_object,
    module = "edgefirst.tensor"
)]
#[derive(Clone)]
pub struct PyQuantization(pub(crate) edgefirst_tensor::Quantization);

impl Display for PyQuantization {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_per_channel() {
            write!(
                f,
                "Quantization(per_channel, num_scales={}, symmetric={}, axis={:?})",
                self.0.scale().len(),
                self.0.is_symmetric(),
                self.0.axis(),
            )
        } else {
            write!(
                f,
                "Quantization(per_tensor, scale={:?}, symmetric={})",
                self.0.scale(),
                self.0.is_symmetric(),
            )
        }
    }
}

#[pymethods]
impl PyQuantization {
    /// Construct per-tensor asymmetric quantization: single scale + zero point.
    #[staticmethod]
    fn per_tensor(scale: f32, zero_point: i32) -> Self {
        Self(edgefirst_tensor::Quantization::per_tensor(
            scale, zero_point,
        ))
    }

    /// Construct per-tensor symmetric quantization: single scale, zero point = 0.
    #[staticmethod]
    fn per_tensor_symmetric(scale: f32) -> Self {
        Self(edgefirst_tensor::Quantization::per_tensor_symmetric(scale))
    }

    /// Construct per-channel asymmetric quantization. Raises on length
    /// mismatch between ``scales`` and ``zero_points``, or empty inputs.
    #[staticmethod]
    fn per_channel(scales: Vec<f32>, zero_points: Vec<i32>, axis: usize) -> Result<Self> {
        Ok(Self(edgefirst_tensor::Quantization::per_channel(
            scales,
            zero_points,
            axis,
        )?))
    }

    /// Construct per-channel symmetric quantization. Raises on empty scales.
    #[staticmethod]
    fn per_channel_symmetric(scales: Vec<f32>, axis: usize) -> Result<Self> {
        Ok(Self(edgefirst_tensor::Quantization::per_channel_symmetric(
            scales, axis,
        )?))
    }

    /// List of scale factors (length 1 for per-tensor, N for per-channel).
    #[getter]
    fn scale(&self) -> Vec<f32> {
        self.0.scale().to_vec()
    }

    /// List of zero points, or ``None`` for symmetric quantization.
    #[getter]
    fn zero_point(&self) -> Option<Vec<i32>> {
        self.0.zero_point().map(<[i32]>::to_vec)
    }

    /// Channel axis for per-channel quantization, or ``None`` for per-tensor.
    #[getter]
    fn axis(&self) -> Option<usize> {
        self.0.axis()
    }

    #[getter]
    fn is_per_tensor(&self) -> bool {
        self.0.is_per_tensor()
    }

    #[getter]
    fn is_per_channel(&self) -> bool {
        self.0.is_per_channel()
    }

    #[getter]
    fn is_symmetric(&self) -> bool {
        self.0.is_symmetric()
    }
}

/// DXGI format and texture geometry behind a texture tensor.
///
/// Platforms:
///     Windows.
///
/// The dimensions are the texture's, not the image's: a packed row folds
/// several pixels or several planes into one texel.
#[cfg(target_os = "windows")]
#[pyclass(
    name = "D3d11Layout",
    frozen,
    get_all,
    skip_from_py_object,
    module = "edgefirst.tensor"
)]
#[derive(Debug, Clone, Copy)]
pub struct PyD3d11Layout {
    /// The texture's `DXGI_FORMAT` as an integer.
    pub dxgi_format: u32,
    /// Texture width in texels.
    pub texture_width: u32,
    /// Texture height in rows.
    pub texture_height: u32,
    /// Bytes in one texel.
    pub bytes_per_texel: u32,
    /// The GL internal format ANGLE accepts for this texture.
    pub gl_internal_format: u32,
}

#[cfg(target_os = "windows")]
impl From<edgefirst_tensor::d3d11_layout::D3d11ImageLayout> for PyD3d11Layout {
    fn from(layout: edgefirst_tensor::d3d11_layout::D3d11ImageLayout) -> Self {
        PyD3d11Layout {
            dxgi_format: layout.dxgi_format,
            texture_width: layout.texture_width as u32,
            texture_height: layout.texture_height as u32,
            bytes_per_texel: layout.bytes_per_texel as u32,
            gl_internal_format: layout.gl_internal_format,
        }
    }
}

#[cfg(target_os = "windows")]
#[pymethods]
impl PyD3d11Layout {
    fn __repr__(&self) -> String {
        format!(
            "D3d11Layout(dxgi_format={}, texture_width={}, texture_height={}, \
             bytes_per_texel={}, gl_internal_format={})",
            self.dxgi_format,
            self.texture_width,
            self.texture_height,
            self.bytes_per_texel,
            self.gl_internal_format
        )
    }
}

// ─── PyCudaMap ──────────────────────────────────────────────────────────────

/// Scoped zero-copy CUDA device-pointer mapping for a tensor (e.g. a TensorRT
/// input). Use as a context manager; the mapping is released on exit. Obtain
/// via Tensor.cuda_map().
#[pyclass(name = "CudaMap", module = "edgefirst.tensor")]
pub struct PyCudaMap {
    // `map` is declared before `_owner` and is dropped first.
    map: Option<edgefirst_tensor::CudaMap<'static>>,
    _owner: Py<PyTensor>,
}

// SAFETY: a CUDA device pointer is process-global and usable from any
// thread; `Py<PyTensor>` is `Send + Sync`.
unsafe impl Send for PyCudaMap {}
unsafe impl Sync for PyCudaMap {}

#[pymethods]
impl PyCudaMap {
    /// Raw CUDA device pointer (as an integer) to the mapped buffer.
    ///
    /// Pass to TensorRT ``setInputTensorAddress``, cupy, or pycuda for
    /// zero-copy GPU input. Returns 0 if the mapping has been released.
    #[getter]
    fn device_ptr(&self) -> usize {
        self.map.as_ref().map_or(0, |m| m.device_ptr() as usize)
    }

    /// Length of the mapping in bytes. Returns 0 if released.
    #[getter]
    fn size(&self) -> usize {
        self.map.as_ref().map_or(0, |m| m.len())
    }

    fn __len__(&self) -> usize {
        self.size()
    }

    /// Release the CUDA mapping (idempotent). Called on ``with`` exit; may also
    /// be called explicitly.
    fn release(&mut self) {
        self.map = None; // drops the CudaMap guard → unmaps
    }

    fn __enter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    fn __exit__(&mut self, _exc_type: Py<PyAny>, _exc_value: Py<PyAny>, _traceback: Py<PyAny>) {
        self.release();
    }
}

// ─── PyHostPin / PyCpuAccess ────────────────────────────────────────────────

/// A stable host address for a tensor's data, valid until released.
///
/// Carries no borrow of the tensor.
///
/// The address gives no coherency guarantee. Bracket CPU access with
/// :meth:`Tensor.cpu_access`.
#[pyclass(name = "HostPin", module = "edgefirst.tensor")]
pub struct PyHostPin {
    pin: Option<edgefirst_tensor::HostPin<'static>>,
    /// The tensor object the pin came from, kept alive while the pin is and
    /// released after it.
    pub(crate) _owner: Option<Py<PyAny>>,
}

#[pymethods]
impl PyHostPin {
    /// The pinned address as an integer, for ``ctypes`` / delegate handoff.
    #[getter]
    fn ptr(&self) -> Result<usize> {
        match self.pin.as_ref() {
            Some(p) => Ok(p.as_mut_ptr() as usize),
            None => Err(Error::HostView("pin has been released".into())),
        }
    }

    /// Usable length in bytes.
    ///
    /// The tensor's logical length, not the allocation's capacity. For a
    /// stride-padded image :meth:`Tensor.map` exposes the padded extent.
    #[getter]
    fn len(&self) -> usize {
        self.pin.as_ref().map_or(0, |p| p.len())
    }

    /// Alignment of the pinned address in bytes.
    #[getter]
    fn alignment(&self) -> usize {
        self.pin.as_ref().map_or(0, |p| p.alignment())
    }

    /// Release the pin. Idempotent.
    fn release(&mut self) {
        self.pin = None;
        self._owner = None;
    }

    fn __enter__(slf: Bound<'_, Self>) -> Bound<'_, Self> {
        slf
    }

    fn __exit__(&mut self, _t: Py<PyAny>, _v: Py<PyAny>, _tb: Py<PyAny>) {
        self.release();
    }

    fn __repr__(&self) -> String {
        match self.pin.as_ref() {
            Some(p) => format!(
                "HostPin(ptr=0x{:x}, len={})",
                p.as_mut_ptr() as usize,
                p.len()
            ),
            None => "HostPin(released)".to_owned(),
        }
    }
}

/// Coherency bracket returned by :meth:`Tensor.cpu_access`.
///
/// The direction given on entry is applied again on exit.
#[pyclass(name = "CpuAccessGuard", module = "edgefirst.tensor")]
pub struct PyCpuAccessGuard {
    /// The tensor, kept alive for the bracket's duration.
    tensor: Py<PyTensor>,
    access: edgefirst_tensor::CpuAccess,
    active: bool,
}

#[pymethods]
impl PyCpuAccessGuard {
    fn __enter__(mut slf: PyRefMut<'_, Self>) -> Result<PyRefMut<'_, Self>> {
        let access = slf.access;
        {
            let py = slf.py();
            let tensor = slf.tensor.borrow(py);
            tensor.0.sync_for_cpu(access)?;
        }
        slf.active = true;
        Ok(slf)
    }

    fn __exit__(
        mut slf: PyRefMut<'_, Self>,
        _t: Py<PyAny>,
        _v: Py<PyAny>,
        _tb: Py<PyAny>,
    ) -> Result<()> {
        if !slf.active {
            return Ok(());
        }
        slf.active = false;
        let access = slf.access;
        let py = slf.py();
        let tensor = slf.tensor.borrow(py);
        tensor.0.sync_for_device(access)?;
        Ok(())
    }
}

// ─── PyTensorMap ────────────────────────────────────────────────────────────

#[pyclass(name = "HostView", module = "edgefirst.tensor")]
pub struct PyTensorMap {
    pub(crate) mapped: Option<TensorMapT>,
    /// True when the map was taken read-only; `__getbuffer__` advertises the
    /// view as read-only.
    pub(crate) readonly: bool,
    /// Physical row pitch in bytes for image tensors, captured from
    /// `effective_row_stride()` at map time. `Some` for any tensor with a pixel
    /// format, `None` otherwise. `__getbuffer__` applies the padded stride only
    /// when it exceeds the tight stride.
    #[cfg_attr(all(Py_LIMITED_API, not(Py_3_11)), allow(dead_code))]
    pub(crate) row_stride: Option<usize>,
    /// Pixel layout at map time, which says which dimension `row_stride`
    /// pads: `[H, W, C]` and `[H*k, W]` pad dim 0, planar `[C, H, W]` pads
    /// dim 1 and makes each plane `row_stride * H` bytes.
    pub(crate) layout: Option<tensor::PixelLayout>,
    /// The tensor object the map came from, kept alive while the map is and
    /// released after it.
    pub(crate) _owner: Option<Py<PyAny>>,
}

unsafe impl Send for PyTensorMap {}
unsafe impl Sync for PyTensorMap {}

impl PyTensorMap {
    /// Byte strides and byte length of the buffer `mapped` exposes, in the
    /// stride convention the descriptor, the blob and the C API share
    /// (`edgefirst_tensor::protocol::c_byte_strides`). A padded image spans
    /// its padded rows; a tight buffer keeps C-contiguous strides.
    fn buffer_geometry(&self, mapped: &TensorMapT) -> (Vec<isize>, usize) {
        let shape = mapped.shape();
        let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        let itemsize = mapped.element_size() as i64;
        let tight = tensor::protocol::c_byte_strides(&dims, itemsize, self.layout, None);
        let tight_len = mapped.size();
        // The padded strides apply when they differ from the tight ones and the
        // rows they describe lie inside the mapping.
        let padded = self
            .row_stride
            .map(|rs| tensor::protocol::c_byte_strides(&dims, itemsize, self.layout, Some(rs)))
            .filter(|p| *p != tight);
        match (padded, shape.first()) {
            (Some(p), Some(&outer)) if p[0] as usize * outer <= mapped.mapped_bytes() => {
                let len = p[0] as usize * outer;
                (p.into_iter().map(|s| s as isize).collect(), len)
            }
            _ => (tight.into_iter().map(|s| s as isize).collect(), tight_len),
        }
    }
}

#[pymethods]
impl PyTensorMap {
    fn unmap(&mut self) {
        if let Some(map) = &mut self.mapped {
            map.unmap();
            self.mapped = None;
        }
        self._owner = None;
    }

    fn __repr__(&self) -> String {
        match &self.mapped {
            Some(m) => format!("HostView(dtype={}, shape={:?})", m.dtype_name(), m.shape(),),
            None => "Unmapped Tensor".to_string(),
        }
    }

    fn __len__(&self) -> usize {
        if let Some(map) = &self.mapped {
            map.shape().iter().product()
        } else {
            0
        }
    }

    fn __getitem__(&self, index: usize, py: Python) -> PyResult<Py<PyAny>> {
        if let Some(map) = &self.mapped {
            map.get_value_at(index, py)
        } else {
            Err(PyBufferError::new_err("Buffer not mapped"))
        }
    }

    fn __setitem__(&mut self, index: usize, value: Py<PyAny>, py: Python) -> PyResult<()> {
        if self.readonly {
            return Err(PyBufferError::new_err(
                "this view is read-only: it was obtained with access=\"read\", \
                 whose coherency bracket discards writes on release. Re-map \
                 with access=\"write\" or access=\"readwrite\" to modify it.",
            ));
        }
        if let Some(map) = &mut self.mapped {
            map.set_value_at(index, value, py)
        } else {
            Err(PyBufferError::new_err("Buffer not mapped"))
        }
    }

    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let slf2 = slf.borrow();

        if slf2.mapped.is_none() {
            return Err(PyBufferError::new_err("Buffer not mapped"));
        }

        // PEP 3118: a request for a writable buffer on a read-only view is
        // refused.
        const PY_BUF_WRITABLE: c_int = 0x0001;
        if slf2.readonly && (flags & PY_BUF_WRITABLE) != 0 {
            return Err(PyBufferError::new_err(
                "this view is read-only: the tensor was mapped with \
                 access=\"read\", whose unmap skips the cache writeback. \
                 Re-map with access=\"readwrite\" (or \"write\") to obtain \
                 a writable buffer",
            ));
        }

        let mapped = slf2.mapped.as_ref().unwrap();
        let shape: Vec<isize> = mapped.shape().iter().map(|&s| s as isize).collect();
        let ndim = shape.len();

        // A row-padded image backing is exposed with its padded strides.
        let (strides, buf_len) = slf2.buffer_geometry(mapped);
        let buf_len = buf_len as isize;
        let itemsize = mapped.element_size() as isize;

        // Box the shape and strides; `__releasebuffer__` frees them using the
        // `ndim` stored in `view.internal`.
        let mut shape = shape.into_boxed_slice();
        let mut strides = strides.into_boxed_slice();

        let ptr = mapped.data_ptr();
        let format = CString::new(mapped.format_str()).unwrap();

        unsafe {
            (*view).buf = ptr;
            (*view).len = buf_len;
            (*view).itemsize = itemsize;
            (*view).readonly = if slf2.readonly { 1 } else { 0 };

            (*view).format = format.into_raw(); // dropped in __releasebuffer__

            (*view).ndim = ndim as i32;
            (*view).shape = shape.as_mut_ptr();
            (*view).strides = strides.as_mut_ptr();
            // `internal` holds ndim, read by `__releasebuffer__`.
            (*view).internal = ndim as *mut c_void;
            std::mem::forget(shape); // dropped in __releasebuffer__
            std::mem::forget(strides); // dropped in __releasebuffer__

            (*view).suboffsets = std::ptr::null_mut();

            (*view).obj = slf.into_ptr();
        }

        Ok(())
    }

    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    unsafe fn __releasebuffer__(&mut self, view: *mut Py_buffer) {
        drop(unsafe { CString::from_raw((*view).format) });
        let ndim = unsafe { (*view).internal } as usize;
        if ndim > 0 {
            // Reconstruct the boxed slices with the correct length.
            drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut((*view).shape, ndim)) });
            drop(unsafe {
                Box::from_raw(std::ptr::slice_from_raw_parts_mut((*view).strides, ndim))
            });
        }
    }

    fn __enter__(slf: Bound<'_, Self>) -> PyResult<Bound<'_, Self>> {
        if slf.borrow().mapped.is_none() {
            return Err(PyBufferError::new_err("Buffer not mapped"));
        }

        Ok(slf)
    }

    fn __exit__(&mut self, _exc_type: Py<PyAny>, _exc_value: Py<PyAny>, _traceback: Py<PyAny>) {
        self.mapped = None; // Release the mapped buffer
        self._owner = None;
    }

    /// Zero-copy NumPy array over the mapped buffer, with the same row-padding
    /// strides as ``memoryview(self)``.
    fn numpy(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if slf.borrow().mapped.is_none() {
            return Err(PyBufferError::new_err("Buffer not mapped"));
        }

        #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
        {
            let np = py.import("numpy")?;
            return Ok(np.getattr("asarray")?.call1((&slf,))?.unbind());
        }

        #[cfg(all(Py_LIMITED_API, not(Py_3_11)))]
        {
            use pyo3::ffi::PyMemoryView_FromMemory;
            use std::os::raw::c_char;
            let borrow = slf.borrow();
            let mapped = borrow.mapped.as_ref().unwrap();
            let shape: Vec<usize> = mapped.shape().to_vec();
            let (strides, buf_len) = borrow.buffer_geometry(mapped);
            let ptr = mapped.data_ptr() as *mut c_char;
            let flag = if borrow.readonly {
                0x100 // PyBUF_READ
            } else {
                0x200 // PyBUF_WRITE
            };
            let mem = unsafe { PyMemoryView_FromMemory(ptr, buf_len as isize, flag) };
            let mv = unsafe { Bound::<PyAny>::from_owned_ptr_or_err(py, mem)? };
            let np = py.import("numpy")?;
            let dtype = mapped.dtype_name();
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("shape", shape)?;
            kwargs.set_item("dtype", dtype)?;
            kwargs.set_item("buffer", mv)?;
            kwargs.set_item("strides", strides)?;
            Ok(np.getattr("ndarray")?.call((), Some(&kwargs))?.unbind())
        }
    }
}

// ─── PyPixelFormat / PyRegion ──────────────────────────────────────────────

/// Pixel format for image tensors.
///
/// Each variant maps directly to an `edgefirst_tensor::PixelFormat` value.
#[pyclass(name = "PixelFormat", skip_from_py_object, module = "edgefirst.tensor")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyPixelFormat {
    Rgb = 1,
    Rgba = 2,
    Bgra = 3,
    Grey = 4,
    Yuyv = 5,
    Vyuy = 6,
    Nv12 = 7,
    Nv16 = 8,
    Nv24 = 11,
    PlanarRgb = 9,
    PlanarRgba = 10,
}

// Compile-time checks that each discriminant equals `PixelFormat::code()`.
const _: () = assert!(PyPixelFormat::Rgb as u32 == PixelFormat::Rgb.code());
const _: () = assert!(PyPixelFormat::Rgba as u32 == PixelFormat::Rgba.code());
const _: () = assert!(PyPixelFormat::Bgra as u32 == PixelFormat::Bgra.code());
const _: () = assert!(PyPixelFormat::Grey as u32 == PixelFormat::Grey.code());
const _: () = assert!(PyPixelFormat::Yuyv as u32 == PixelFormat::Yuyv.code());
const _: () = assert!(PyPixelFormat::Vyuy as u32 == PixelFormat::Vyuy.code());
const _: () = assert!(PyPixelFormat::Nv12 as u32 == PixelFormat::Nv12.code());
const _: () = assert!(PyPixelFormat::Nv16 as u32 == PixelFormat::Nv16.code());
const _: () = assert!(PyPixelFormat::PlanarRgb as u32 == PixelFormat::PlanarRgb.code());
const _: () = assert!(PyPixelFormat::PlanarRgba as u32 == PixelFormat::PlanarRgba.code());
const _: () = assert!(PyPixelFormat::Nv24 as u32 == PixelFormat::Nv24.code());

#[pymethods]
impl PyPixelFormat {
    #[new]
    pub fn new(name: &str) -> Result<Self> {
        Self::try_from(name)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        eq_int_richcmp(*self, other, false, "PixelFormat", Self::from_discriminant)
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        eq_int_richcmp(*self, other, true, "PixelFormat", Self::from_discriminant)
    }

    /// Returns the discriminant as the hash.
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl PyPixelFormat {
    /// Reconstruct from the `__int__()` discriminant of a sibling package's
    /// copy of this enum.
    fn from_discriminant(v: i64) -> Option<Self> {
        u32::try_from(v)
            .ok()
            .and_then(PixelFormat::from_code)
            .and_then(|f| PyPixelFormat::try_from(f).ok())
    }
}

/// Extracts a native instance, or a sibling `edgefirst.*` package's
/// `PixelFormat` through its `__int__()` discriminant.
impl<'a, 'py> FromPyObject<'a, 'py> for PyPixelFormat {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        extract_eq_int_enum(obj, "PixelFormat", Self::from_discriminant)
    }
}

impl TryFrom<&str> for PyPixelFormat {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value.to_uppercase().as_str() {
            "YUYV" => Ok(PyPixelFormat::Yuyv),
            "VYUY" => Ok(PyPixelFormat::Vyuy),
            "RGBA" => Ok(PyPixelFormat::Rgba),
            "BGRA" => Ok(PyPixelFormat::Bgra),
            "RGB" | "RGB " => Ok(PyPixelFormat::Rgb),
            "NV12" => Ok(PyPixelFormat::Nv12),
            "NV16" => Ok(PyPixelFormat::Nv16),
            "NV24" => Ok(PyPixelFormat::Nv24),
            "Y800" | "GREY" | "GRAY" => Ok(PyPixelFormat::Grey),
            "8BPS" | "PLANAR_RGB" | "PLANARRGB" => Ok(PyPixelFormat::PlanarRgb),
            "PLANAR_RGBA" | "PLANARRGBA" => Ok(PyPixelFormat::PlanarRgba),
            _ => Err(Error::Format(value.to_string())),
        }
    }
}

impl From<PyPixelFormat> for PixelFormat {
    fn from(val: PyPixelFormat) -> Self {
        match val {
            PyPixelFormat::Rgb => PixelFormat::Rgb,
            PyPixelFormat::Rgba => PixelFormat::Rgba,
            PyPixelFormat::Bgra => PixelFormat::Bgra,
            PyPixelFormat::Grey => PixelFormat::Grey,
            PyPixelFormat::Yuyv => PixelFormat::Yuyv,
            PyPixelFormat::Vyuy => PixelFormat::Vyuy,
            PyPixelFormat::Nv12 => PixelFormat::Nv12,
            PyPixelFormat::Nv16 => PixelFormat::Nv16,
            PyPixelFormat::Nv24 => PixelFormat::Nv24,
            PyPixelFormat::PlanarRgb => PixelFormat::PlanarRgb,
            PyPixelFormat::PlanarRgba => PixelFormat::PlanarRgba,
        }
    }
}

impl TryFrom<PixelFormat> for PyPixelFormat {
    type Error = Error;

    fn try_from(val: PixelFormat) -> Result<Self, Self::Error> {
        match val {
            PixelFormat::Rgb => Ok(PyPixelFormat::Rgb),
            PixelFormat::Rgba => Ok(PyPixelFormat::Rgba),
            PixelFormat::Bgra => Ok(PyPixelFormat::Bgra),
            PixelFormat::Grey => Ok(PyPixelFormat::Grey),
            PixelFormat::Yuyv => Ok(PyPixelFormat::Yuyv),
            PixelFormat::Vyuy => Ok(PyPixelFormat::Vyuy),
            PixelFormat::Nv12 => Ok(PyPixelFormat::Nv12),
            PixelFormat::Nv16 => Ok(PyPixelFormat::Nv16),
            PixelFormat::Nv24 => Ok(PyPixelFormat::Nv24),
            PixelFormat::PlanarRgb => Ok(PyPixelFormat::PlanarRgb),
            PixelFormat::PlanarRgba => Ok(PyPixelFormat::PlanarRgba),
            _ => Err(Error::Format(format!("unsupported pixel format: {val:?}"))),
        }
    }
}

#[pyclass(name = "Region", skip_from_py_object, module = "edgefirst.tensor")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyRegion {
    #[pyo3(get, set)]
    pub x: usize,
    #[pyo3(get, set)]
    pub y: usize,
    #[pyo3(get, set)]
    pub width: usize,
    #[pyo3(get, set)]
    pub height: usize,
}

#[pymethods]
impl PyRegion {
    #[new]
    pub fn new(x: usize, y: usize, width: usize, height: usize) -> PyRegion {
        PyRegion {
            x,
            y,
            width,
            height,
        }
    }

    /// Equality against a native `Region` or a sibling package's `Region` with
    /// the same `x`/`y`/`width`/`height`.
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        match other.as_borrowed().extract::<PyRegion>() {
            Ok(v) => pyo3::IntoPyObjectExt::into_py_any(*self == v, py),
            Err(_) => Ok(py.NotImplemented()),
        }
    }

    fn __ne__(&self, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = other.py();
        match other.as_borrowed().extract::<PyRegion>() {
            Ok(v) => pyo3::IntoPyObjectExt::into_py_any(*self != v, py),
            Err(_) => Ok(py.NotImplemented()),
        }
    }

    /// Hash of `x`, `y`, `width` and `height`, mixed in that order with rotate,
    /// xor and wrapping multiply by the FxHash constant.
    fn __hash__(&self) -> u64 {
        const K: u64 = 0x517c_c1b7_2722_0a95;
        let mut h: u64 = 0;
        for v in [self.x, self.y, self.width, self.height] {
            h = (h.rotate_left(5) ^ v as u64).wrapping_mul(K);
        }
        h
    }
}

/// Extracts a native instance, or an object whose type is named `Region`
/// through its `x`/`y`/`width`/`height` attributes.
impl<'a, 'py> FromPyObject<'a, 'py> for PyRegion {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> Result<Self, Self::Error> {
        if let Ok(guard) = obj.extract::<pyo3::PyClassGuard<'_, Self>>() {
            return Ok(*guard);
        }
        let is_region = obj
            .get_type()
            .name()
            .map(|n| n == "Region")
            .unwrap_or(false);
        if !is_region {
            return Err(pyo3::exceptions::PyTypeError::new_err("expected a Region"));
        }
        Ok(PyRegion {
            x: obj.getattr("x")?.extract()?,
            y: obj.getattr("y")?.extract()?,
            width: obj.getattr("width")?.extract()?,
            height: obj.getattr("height")?.extract()?,
        })
    }
}

impl From<PyRegion> for Region {
    fn from(val: PyRegion) -> Self {
        Region {
            x: val.x,
            y: val.y,
            width: val.width,
            height: val.height,
        }
    }
}
