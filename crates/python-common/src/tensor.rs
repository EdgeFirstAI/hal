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
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            // Display, not Debug: variants with a hand-written Display arm
            // (UnknownBufferType's hex magic, InsufficientCapacity,
            // RegionOutOfBounds, BatchIndexOutOfBounds) render legibly, and
            // the rest fall back to Debug inside tensor::Error's own Display.
            Error::Tensor(e) => write!(f, "Tensor error: {e}"),
            Error::UnsupportedMemoryType(msg) => write!(f, "Invalid memory type: {msg}"),
            Error::UnsupportedDataType(msg) => write!(f, "Invalid data type: {msg}"),
            Error::HostView(msg) => write!(f, "Tensor map error: {msg}"),
            Error::Format(msg) => write!(f, "Format error: {msg}"),
            Error::Io(e) => write!(f, "IO error: {e:?}"),
        }
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
        Error::Format(format!("{err:?}"))
    }
}

#[cfg(feature = "image")]
impl From<crate::image::Error> for Error {
    fn from(err: crate::image::Error) -> Self {
        Error::Format(format!("{err}"))
    }
}

impl From<Error> for PyErr {
    fn from(err: Error) -> PyErr {
        pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
    }
}

#[cfg(feature = "codec")]
impl From<edgefirst_codec::CodecError> for Error {
    fn from(err: edgefirst_codec::CodecError) -> Self {
        Error::Format(format!("{err}"))
    }
}

#[cfg(feature = "codec")]
use std::cell::RefCell;
#[cfg(feature = "codec")]
thread_local! {
    static DECODER: RefCell<edgefirst_codec::ImageDecoder> =
        RefCell::new(edgefirst_codec::ImageDecoder::new());
}

/// IDCT accuracy/speed selection for the software JPEG decoder.
///
/// :attr:`Accurate` is the default: the `islow`-class Loeffler IDCT,
/// bit-comparable to libjpeg-turbo's default. :attr:`Fast` opts into the
/// AAN `ifast`-class kernel — roughly an eighth of the multiplies, at a
/// small, bounded pixel accuracy cost. Fast is advisory: paths without a
/// fast kernel (non-NEON tiers, the V4L2/nvJPEG hardware decoders, and PNG)
/// use their normal accurate path. Applies to the thread-local decoder used
/// by :meth:`Tensor.decode_image` / :meth:`Tensor.decode_image_file`; set via
/// :func:`set_dct_method`. Each thread has its own decoder state, so this
/// must be set on every thread that decodes images.
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
/// This only affects the calling thread; call it on every thread that
/// decodes images if you want a non-default setting everywhere.
///
/// Accurate by default. See :class:`DctMethod` for the accuracy/speed
/// tradeoff. ``EDGEFIRST_CODEC_DCT=fast`` in the environment flips a
/// **new** thread's default for A/B runs.
///
/// Args:
///     method: IDCT kernel class.
#[cfg(feature = "codec")]
#[pyo3::pyfunction]
pub fn set_dct_method(method: PyDctMethod) {
    DECODER.with(|cell| cell.borrow_mut().set_dct_method(method.into()));
}

/// Request a fused JPEG decode output format instead of the source's
/// native format, for the thread-local decoder used by
/// :meth:`Tensor.decode_image` / :meth:`Tensor.decode_image_file`. Only
/// affects the calling thread. PNG decodes are unaffected.
///
/// This is a **pure CPU, single-pass** path inside the software JPEG
/// decoder — colour conversion / chroma downsample happens at the MCU
/// write stage. It is **not** a GPU hybrid or nvJPEG path; V4L2/nvJPEG are
/// bypassed whenever the resolved output differs from native.
///
/// - ``PixelFormat.RGB``: 4:4:4 colour JPEGs decode straight to interleaved
///   RGB. Other sources fall back to native.
/// - ``PixelFormat.NV12``: colour JPEGs decode to NV12, downsampling
///   chroma at the write stage (2×2 average for 4:4:4, vertical for 4:2:2).
/// - Any other format is ignored; the decode falls back to native.
/// - ``None`` (default): native format (``Nv12``/``Nv16``/``Nv24``/``Grey``).
///
/// Callers still run ``ImageProcessor.convert()`` on the result for
/// model-input preprocessing (letterbox, resize, EXIF orientation). With
/// fused RGB that convert step is typically a pure resize.
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
/// block) is present and not opted out via ``EDGEFIRST_DISABLE_V4L2``.
///
/// Opens and drops the device once; the decode path re-probes lazily and
/// keeps its own context. Always ``False`` on platforms without V4L2
/// hardware decode support. Useful for benchmarks and callers that must
/// fail fast instead of silently falling back to the CPU decoder.
#[cfg(feature = "codec")]
#[pyo3::pyfunction]
pub fn is_v4l2_available() -> bool {
    edgefirst_codec::v4l2_available()
}

/// Shared decode body for ``Tensor.decode_image`` and the cross-package
/// ``decode_into`` free function -- both must run exactly this code, or the
/// two entry points silently drift apart.
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

/// Copy a decode's resulting format/shape/colorimetry from `tensor` --
/// the independent `TensorDyn` [`crate::interop::reconstruct`] built for the
/// detached region -- back onto `dst`, the real `PyTensor`'s own `TensorDyn`.
///
/// Needed because `dst`/`tensor` are no longer the same value: `reconstruct`
/// aliases the same backing memory (so `decode_image_into`'s pixel writes
/// above already landed on the real allocation) but keeps its own
/// `shape_cache` etc. (see `RawTensorAccess`'s docs), so the metadata
/// `load_image` set on `tensor` -- not `dst` -- needs writing back
/// explicitly. `Tensor.decode_image`/`decode_image_file`'s counterpart to
/// `decode_into`'s Foreign write-back, but going through `TensorDyn`
/// directly rather than a generic `call_method1`/`setattr`: `dst` is known
/// statically to be a `PyTensor` from this crate, so there is no
/// third-party destination to degrade gracefully for.
#[cfg(feature = "codec")]
fn write_back_decode(dst: &mut TensorDyn, tensor: &TensorDyn, info: &PyImageInfo) -> Result<()> {
    let fmt: PixelFormat = info.format.into();
    dst.configure_image(info.width, info.height, fmt)?;
    dst.set_colorimetry(tensor.colorimetry());
    Ok(())
}

/// Decode image bytes (JPEG/PNG) into any object implementing the
/// ``__edgefirst_tensor__`` capsule protocol -- the cross-package
/// counterpart of :meth:`Tensor.decode_image` for destinations created by
/// another ``edgefirst.*`` extension, e.g.
/// ``edgefirst.image.ImageProcessor.create_image()``.
///
/// Decode semantics (native format, EXIF handling, ``set_output_format`` /
/// ``set_dct_method`` hooks) are identical to :meth:`Tensor.decode_image`;
/// this is a thin wrapper that resolves ``tensor`` through the interop
/// protocol and shares the exact same decode path.
///
/// For a foreign ``tensor`` (a different ``edgefirst.*`` package than this
/// one), the pixels are always written correctly -- that part cannot fail.
/// ``tensor``'s own format, dimensions and colorimetry are then updated to
/// match the decode on a **best-effort** basis: if ``tensor`` does not
/// implement ``configure_image()``, or exposes ``colorimetry`` read-only,
/// that update is silently skipped (only a ``log::warn`` on the Rust side
/// notes it -- nothing raises). The returned :class:`ImageInfo` always
/// describes the decode accurately regardless, so a caller that needs the
/// decoded format/dimensions reliably should read them from the return
/// value, not from ``tensor`` itself, when ``tensor`` may not be a
/// ``PyTensor`` from this crate.
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
    // Copy while the GIL is still held -- see `PyTensor::decode_image`.
    let data = data.to_vec();
    let (info, colorimetry) = if arg.can_detach() {
        // `into_raw_access` decodes into an independent `TensorDyn`
        // reconstructed from `tensor`'s own descriptor (see
        // `RawTensorAccess`'s docs) -- never `tensor` itself, for the
        // native path just as much as the foreign one. Every Python guard
        // is released by this point, so the decode itself can run with the
        // GIL released.
        let mut raw = arg.into_raw_access()?;
        let (info, raw) = py
            .detach(move || decode_image_into(raw.as_mut(), &data).map(|info| (info, raw)))
            .map_err(PyErr::from)?;
        (info, raw.as_ref().colorimetry())
    } else {
        // A GL-PBO-backed `tensor`: `TensorArg::can_detach` (see its docs)
        // cannot reconstruct an independent `TensorDyn` for it, so this
        // call keeps the GIL held for its whole duration, exactly this
        // crate's behaviour before GIL release existed for tensors at all.
        // (Always the native path here: a PBO-backed `Foreign` extraction
        // already fails upstream, at `TensorArg::extract_mut`, for the same
        // reason `can_detach` would refuse it -- `import_descriptor` has no
        // PBO arm either way.)
        let mut arg = arg;
        let info = decode_image_into(arg.as_mut(), &data).map_err(PyErr::from)?;
        (info, arg.as_ref().colorimetry())
    };

    // `configure_image` (format + logical shape) and colorimetry the decode
    // determined must be copied back onto `tensor` explicitly, to leave it
    // in the same state a direct in-place decode would -- every downstream
    // `convert()` call resolves `tensor` afresh via its own
    // `__edgefirst_tensor__` capsule (or, for a same-module `PyTensor`,
    // reads its fields directly), so anything left only on the disposable
    // reconstructed copy is invisible to it either way. (For the
    // non-detached branch above, `tensor` and the decoded value are the
    // same object, so this only re-asserts values it already has --
    // harmless, and simpler than special-casing it away.)
    //
    // Best-effort: the pixels are already written zero-copy into the real
    // allocation by this point (a `HOST`-kind reconstruction aliases the
    // producer's own pinned address; `DMABUF`/`IOSURFACE` alias the same
    // fd/surface), regardless of what happens below -- that part cannot
    // fail here. INTEROP.md's `EdgeFirstTensorExportable` requires only
    // `__edgefirst_tensor__`, so a conforming third-party destination need
    // not expose `configure_image` or a settable `colorimetry`; failing the
    // whole decode over metadata write-back after the actual write already
    // succeeded would be a failure-after-success, the worst shape for a
    // caller to handle. A `PyTensor` from this crate (the common case,
    // native or foreign) always accepts both calls, so this only degrades
    // for a non-`PyTensor` destination.
    //
    // This re-enters Python (`call_method1`/`setattr`); on the detached
    // branch it runs after `py.detach` has returned and the GIL is held
    // again, and on the non-detached branch the GIL was never released.
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
    // `decode_into` releases the GIL for the decode itself; the file read
    // ahead of it is comparatively small next to CPU JPEG/PNG decode, so
    // it stays a plain synchronous read rather than its own detach.
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

// No `eq`/`eq_int` here: those auto-generate `__richcmp__`, which would
// collide with the hand-written `__eq__`/`__ne__` below (needed for the
// cross-package fallback -- see `eq_int_richcmp`). `__int__()` is still
// generated unconditionally for every simple enum, `eq`/`eq_int` or not.
//
// Discriminants are the shared codes from `edgefirst_tensor`'s
// `TensorMemory` (declared via `ef_vocabulary!`), not a locally-numbered
// list -- this is the fix for the bug this vocabulary work closes: before,
// this enum had NO explicit discriminants and a `#[cfg(unix)]` variant in
// the middle, so `MEM` was `3` on unix and `2` elsewhere, while the C ABI's
// `PBO` was `3` -- bridging the two surfaces by raw integer silently handed
// back `PBO` where `MEM` was asked for. `SHM` is no longer `#[cfg(unix)]`:
// the code is defined on every platform (only `TensorMemory::is_available`
// answers the runtime question), so a tensor recorded on Linux and read
// back on macOS can still be *named* here even though it cannot be
// materialised.
#[pyclass(
    name = "TensorMemory",
    skip_from_py_object,
    module = "edgefirst.tensor"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum PyTensorMemory {
    /// Regular system memory allocation. Available everywhere.
    MEM = 0,
    /// POSIX Shared Memory allocation. Suitable for inter-process
    /// communication, but not for hardware acceleration. Nameable on every
    /// platform; only allocatable on unix.
    SHM = 1,
    /// Platform-native zero-copy GPU buffer: DMA-BUF on Linux,
    /// IOSurface on macOS. Same Python name on both platforms.
    DMABUF = 2,
    /// Apple IOSurface, named specifically rather than through the
    /// portable `DMABUF` spelling. No backend produces or accepts it yet --
    /// macOS/iOS allocate and report `DMABUF`.
    IOSURFACE = 3,
    /// GPU Pixel Buffer Object (PBO) allocation. Used for zero-copy GPU
    /// upload/readback on platforms without DMA-buf support.
    PBO = 4,
    /// CUDA device memory. No backend produces or accepts it yet.
    CUDA = 5,
}

// The literals above are copied from `TensorMemory::code()`, not computed
// from it -- PyO3 requires a literal in a `#[pyclass]` enum's discriminant
// position, so `MEM = TensorMemory::Mem.code()` does not parse. Without
// something enforcing the link, that copy is exactly the kind of
// hand-synced parallel table this vocabulary work exists to remove; it
// would just be a smaller one. These assertions are that enforcement: a
// mismatch on either side is a BUILD failure, naming the failed assertion,
// not a test someone has to remember to run. `TensorMemory::code()` is
// `const fn` for exactly this purpose (see `ef_vocabulary!`'s doc comment).
const _: () = assert!(PyTensorMemory::MEM as u32 == TensorMemory::Mem.code());
const _: () = assert!(PyTensorMemory::SHM as u32 == TensorMemory::Shm.code());
const _: () = assert!(PyTensorMemory::DMABUF as u32 == TensorMemory::DmaBuf.code());
const _: () = assert!(PyTensorMemory::IOSURFACE as u32 == TensorMemory::IoSurface.code());
const _: () = assert!(PyTensorMemory::PBO as u32 == TensorMemory::Pbo.code());
const _: () = assert!(PyTensorMemory::CUDA as u32 == TensorMemory::Cuda.code());

impl PyTensorMemory {
    /// Reconstruct from the `__int__()` discriminant of a sibling package's
    /// copy of this enum. Used by the cross-package `FromPyObject` fallback
    /// below -- see `extract_eq_int_enum`.
    ///
    /// Delegates to `TensorMemory::from_code` rather than a hand-written
    /// `if v == Self::X as i64` chain: that chain is the parallel table
    /// this vocabulary work removes. `PyTensorMemory`'s discriminants are
    /// the same shared codes by construction (see the enum's doc comment
    /// above), so a code that resolves on the Rust side always resolves
    /// here too.
    fn from_discriminant(v: i64) -> Option<Self> {
        u32::try_from(v)
            .ok()
            .and_then(TensorMemory::from_code)
            .map(Into::into)
    }
}

/// A sibling `edgefirst.*` package's `TensorMemory` names the same variant
/// but is a distinct PyO3 type object (see `interop.rs`'s module docs), so
/// `#[pyclass(from_py_object)]`'s auto-derived native-only extraction would
/// reject it. Try the native downcast first, then fall back to the
/// `__int__()` discriminant a sibling package's copy also exposes (via
/// `eq_int`).
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

    /// Equal objects must hash equal: this is the same discriminant
    /// `__eq__`/`__ne__` compare (and, since a plain int compares equal
    /// when its value matches, the same integer Python's own `hash(int)`
    /// returns -- `int.__hash__` is the identity function for values this
    /// small).
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
        }
    }
}

impl From<TensorMemory> for PyTensorMemory {
    fn from(value: TensorMemory) -> Self {
        match value {
            TensorMemory::Mem => PyTensorMemory::MEM,
            TensorMemory::Shm => PyTensorMemory::SHM,
            TensorMemory::DmaBuf => PyTensorMemory::DMABUF,
            // Task 4 left this arm as a placeholder mapping to `MEM` with a
            // warning, pending this task's decision: now that `SHM` is
            // defined on every platform and `IoSurface`/`Cuda` each have a
            // same-named, same-coded Python variant, every one of them maps
            // straight across -- no more folding or fallback needed.
            TensorMemory::IoSurface => PyTensorMemory::IOSURFACE,
            TensorMemory::Pbo => PyTensorMemory::PBO,
            TensorMemory::Cuda => PyTensorMemory::CUDA,
            // `TensorMemory` is `#[non_exhaustive]`, so rustc requires this
            // arm outside its defining crate. Reached only by a variant
            // added upstream without a decision made here -- which is what
            // the warning is for, since this signature cannot fail.
            other => {
                log::warn!("TensorMemory {other:?} has no Python counterpart, reporting MEM");
                PyTensorMemory::MEM
            }
        }
    }
}

/// Cross-package fallback shared by every `#[pyclass(eq, eq_int)]` value
/// enum (`TensorMemory`, `PixelFormat`, the colorimetry axis enums): try the
/// native downcast first (same-package instance -- zero-cost, the common
/// case), then fall back to a sibling package's copy of the same enum.
///
/// A sibling package's enum names the same variant but is a distinct PyO3
/// type object -- each `edgefirst.*` extension statically links its own copy
/// of these bindings (see `interop.rs`'s module docs) -- so the downcast
/// always fails for it. `eq_int` gives every such enum an `__int__()` (but
/// deliberately *not* an `__index__()`, so `Bound::extract::<uN>()` does not
/// silently accept these values where an ordinary integer was expected);
/// call it explicitly and reconstruct locally from the discriminant.
///
/// The `__int__()` fallback is gated on `type(obj).__name__` matching
/// `type_name` first. Without this gate, *any* object with an `__int__()`
/// would be accepted here -- including an unrelated `eq_int` enum that
/// happens to share a discriminant, e.g. `PixelFormat.Rgb` and
/// `TensorMemory.SHM` are both `1`. That is a type-confusion bug, not a
/// looser acceptance policy: it let a `TensorMemory` silently pass as a
/// `PixelFormat` (and compare/hash equal to it). Every `edgefirst.*`
/// package's copy of the SAME type shares the same `#[pyclass(name = ...)]`,
/// so the name check still accepts a sibling package's copy while rejecting
/// a same-shaped but unrelated type. It also means a bare Python `int` is no
/// longer accepted here (`type(1).__name__ == "int"`) -- unlike equality
/// (see `eq_int_richcmp`), argument extraction never accepted a bare int as
/// a matter of policy; this closes an accidental hole where it used to.
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

/// Shared cross-package `__eq__`/`__ne__` for the value enums covered by
/// [`extract_eq_int_enum`] above. `#[pyclass(eq, eq_int)]`'s auto-generated
/// `__richcmp__` resolves `other` by native identity or a bare int only, so
/// a sibling package's copy of the same enum compares unequal -- silently,
/// never an error, which is worse: `tensor.memory != TensorMemory.DMABUF`
/// returns `True` for values that ARE equal. This reuses the exact
/// `FromPyObject` fallback (name-gated -- see `extract_eq_int_enum`) so
/// `==`/`!=` agree with what the enum already accepts as a value.
///
/// A plain `int` still compares by discriminant exactly as `eq_int` did
/// (`TensorMemory.MEM == 0`), but only a genuine `int`: extracted via
/// `Bound::extract::<i64>()`, which goes through `__index__` -- an unrelated
/// `eq_int` enum has `__int__()` but deliberately no `__index__()` (see
/// `extract_eq_int_enum`'s docs), so it cannot slip through this branch
/// either. A genuinely unrelated type gets `NotImplemented` so Python's
/// normal fallback (and thus `!=`'s identity default) applies. `invert`
/// selects `__ne__` over `__eq__`.
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
    match access {
        "none" => Ok(tensor::CpuAccess::None),
        "read" => Ok(tensor::CpuAccess::Read),
        "write" => Ok(tensor::CpuAccess::Write),
        "readwrite" => Ok(tensor::CpuAccess::ReadWrite),
        _ => Err(Error::Format(format!(
            "access must be one of none|read|write|readwrite, got {access:?}"
        ))),
    }
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
// Needed for Python buffer protocol — must dispatch per dtype to get typed
// pointers, format strings, and per-element operations.

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
    ///
    /// Read from the mapping itself rather than from the access string the
    /// caller passed, so the buffer protocol's `readonly` flag cannot drift
    /// from the bracket actually taken -- if the direction ever stopped
    /// reaching `map_with`, this would report writable and the read-only
    /// tests would fail, instead of agreeing with a request that was never
    /// honoured.
    pub fn is_writable(&self) -> bool {
        map_dispatch!(self, is_writable)
    }

    pub fn shape(&self) -> &[usize] {
        map_dispatch!(self, shape)
    }

    pub fn size(&self) -> usize {
        map_dispatch!(self, size)
    }

    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
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
            TensorMapT::TensorU8(m) => Ok(m.as_ref()[index].into_pyobject(py)?.into()),
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
            TensorMapT::TensorU8(m) => m.as_ref().as_ptr() as *mut c_void,
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

/// Map a `TensorDyn` to a `TensorMapT`.
fn map_tensor_dyn(t: &TensorDyn, access: tensor::CpuAccess) -> tensor::Result<TensorMapT> {
    // `map_with`, not `map()`: `map()` is `map_with(ReadWrite)`, and on a
    // non-coherent backing a ReadWrite bracket pays a full-buffer cache
    // writeback on unmap even when the caller only read. Routing the
    // direction through means a reader skips it (and on macOS takes the
    // read-only IOSurface lock, which skips the unlock flush).
    macro_rules! lens {
        ($ty:ty, $variant:ident) => {
            t.as_typed::<$ty>()
                .expect("dtype checked")
                .map_with(access)
                .map(TensorMapT::$variant)
        };
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

// ─── numpy → tensor copy ────────────────────────────────────────────────────

/// Type-matched copy from a numpy array into a `TensorDyn`.
///
/// Downcasts the numpy array to the concrete element type matching the
/// tensor's dtype, then copies via the typed `HostView` slice.
///
/// Copy strategy (selected automatically):
/// 1. **Fully contiguous** → single `copy_from_slice` (memcpy).
/// 2. **Strided with contiguous inner rows** → one memcpy per row,
///    iterating over outer dimensions.
/// 3. **Fully strided** (e.g. transposed view, every-other-element) →
///    materialize a contiguous source via `np.ascontiguousarray()`
///    before the destination memcpy. numpy's vectorized strided→contig
///    pass is dramatically faster than ndarray's stride-respecting
///    iterator for fully strided arrays — on a `(1, 116, 8400)` f32
///    transposed view (typical HailoRT output) this is ≈12× faster
///    than per-element iteration.
///
/// All three paths use `rayon` parallel iteration when ≥ 256 KiB.
///
/// Raises on dtype mismatch or element-count mismatch.
fn copy_numpy_to_tensor_dyn(src: &Bound<'_, pyo3::types::PyAny>, tensor: &TensorDyn) -> Result<()> {
    use numpy::{PyArrayMethods, PyUntypedArrayMethods};

    /// Byte threshold above which copies are parallelized via rayon.
    const PARALLEL_THRESHOLD_BYTES: usize = 256 * 1024;

    fn copy_typed<
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

        let mut map = tensor.map()?;
        let dst = map.as_mut_slice();
        let dst_len = dst.len();
        let nbytes = tensor_len * std::mem::size_of::<T>();
        let parallel = nbytes >= PARALLEL_THRESHOLD_BYTES;
        // Minimum chunk: 4 KiB worth of elements (scales with element size).
        let min_chunk = (4096 / std::mem::size_of::<T>()).max(1);

        // Destination-side stride padding (STRIDES_BUG.md): when
        // `create_image` allocates a DMA-BUF or PBO buffer with GPU
        // pitch alignment padding, `map()` exposes the full padded
        // buffer (`stride × height` bytes) but the logical element
        // count from shape is smaller (`width × channels × height`).
        // A flat `copy_from_slice` would panic on the length
        // mismatch. Detect this and copy row-by-row, placing
        // `row_elems` logical pixels per row and skipping the
        // padding bytes in the destination.
        if dst_len > tensor_len {
            let elem_sz = std::mem::size_of::<T>();
            let stride_bytes = tensor.effective_row_stride().ok_or_else(|| {
                Error::Format(format!(
                    "destination buffer is padded ({dst_len} elems > {tensor_len} logical) \
                     but tensor has no effective_row_stride"
                ))
            })?;
            let height = tensor.height().ok_or_else(|| {
                Error::Format("destination buffer is padded but tensor has no height".to_string())
            })?;
            if height == 0 || elem_sz == 0 {
                return Ok(());
            }
            let dst_stride_elems = stride_bytes / elem_sz;
            let row_elems = tensor_len / height;

            if dst_stride_elems * height != dst_len || row_elems * height != tensor_len {
                return Err(Error::Format(format!(
                    "stride-padded copy: inconsistent dimensions: \
                     dst_len={dst_len}, tensor_len={tensor_len}, height={height}, \
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
            return Ok(());
        }

        if arr.is_c_contiguous() {
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
                // C-contiguous but as_slice() failed (e.g., misaligned buffer).
                // Fall back to element-wise copy.
                py.detach(|| {
                    for (d, &s) in dst.iter_mut().zip(src_view.iter()) {
                        *d = s;
                    }
                });
            }
        } else {
            // Non-contiguous: find the longest contiguous inner dimension.
            // Walk inward from the last axis: if stride[i] == product of
            // shape[i+1..] (in elements), that axis and all inner axes form
            // a contiguous row we can memcpy.
            let shape = src_view.shape();
            let strides = src_view.strides(); // in elements (ndarray convention)
            let ndim = shape.len();

            let mut contig_elems: usize = 1;
            let mut contig_dims: usize = 0;
            for i in (0..ndim).rev() {
                // Size-1 dims are always contiguous regardless of stride.
                if strides[i] == contig_elems as isize || shape[i] <= 1 {
                    contig_elems *= shape[i];
                    contig_dims += 1;
                } else {
                    break;
                }
            }

            if contig_elems > 1 && contig_elems < tensor_len {
                // Path 2: strided outer, contiguous inner rows.
                // Compute row byte-offsets from strides in O(n_rows) —
                // no element-level iteration needed.
                let n_rows = tensor_len / contig_elems;
                let row_len = contig_elems;
                let elem_size = std::mem::size_of::<T>() as isize;

                // Outer dimensions are those NOT part of the contiguous tail.
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

                // Store base as usize for Send+Sync safety in rayon closures.
                // The pointer is reconstructed inside each task. This is safe
                // because the numpy readonly guard pins the source buffer for
                // our entire scope.
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
            } else {
                // Path 3: fully strided (contig_elems == 1) — e.g. a
                // transposed view of a contiguous backing buffer such as
                // the (1, channels, anchors) output that HailoRT returns
                // as a (0, 2, 1) transpose of (1, anchors, channels).
                //
                // Element-wise iteration over a strided ndarray view is
                // dramatically slower than a contiguous memcpy because
                // every load incurs stride arithmetic and breaks
                // vectorization. Measurements on rpi5-hailo with a
                // (1, 116, 8400) f32 view showed 27 ms/call for the old
                // per-element path versus 6.5 ms/call when the caller
                // pre-applied np.ascontiguousarray().
                //
                // Materialize a contiguous source via
                // np.ascontiguousarray() — its strided→contig pass runs
                // in vectorized C and is much faster than ndarray's
                // stride-respecting iter() — then fall back to the
                // Path 1 memcpy. The intermediate buffer is owned by
                // numpy and freed when contig_obj is dropped.
                let np = py
                    .import("numpy")
                    .map_err(|e| Error::Format(format!("failed to import numpy: {e}")))?;
                let contig_obj = np
                    .call_method1("ascontiguousarray", (src,))
                    .map_err(|e| Error::Format(format!("np.ascontiguousarray failed: {e}")))?;
                let contig_arr = contig_obj.cast::<numpy::PyArrayDyn<T>>().map_err(|_| {
                    Error::Format("np.ascontiguousarray returned the wrong dtype".to_string())
                })?;
                let contig_readonly = contig_arr.readonly();
                let contig_slice = contig_readonly.as_slice().map_err(|_| {
                    Error::Format(
                        "np.ascontiguousarray result is not contiguous (unexpected)".to_string(),
                    )
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
            }
        }

        Ok(())
    }

    macro_rules! lens {
        ($ty:ty) => {
            copy_typed::<$ty>(src, tensor.as_typed::<$ty>().expect("dtype checked"))
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

// ─── PyTensor ───────────────────────────────────────────────────────────────

#[pyclass(name = "Tensor", str, module = "edgefirst.tensor")]
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
        let memory = mem.map(|x| x.into());
        let tensor = TensorDyn::new(&shape, dt, memory, name)?;
        Ok(PyTensor(tensor))
    }

    /// Import an existing buffer as a tensor, without copying.
    ///
    /// The buffer type is detected, not chosen. On Linux a `dma_buf` fd
    /// imports as `TensorMemory.DMABUF` and a tmpfs fd (`/dev/shm` or `memfd`)
    /// as `TensorMemory.SHM`, decided by filesystem magic; any other
    /// filesystem raises `RuntimeError` rather than silently falling back to
    /// shared memory. On macOS the fd is always imported as SHM.
    ///
    /// The fd is `dup()`'d immediately — the caller retains ownership of the
    /// original and must close it.
    ///
    /// Check `tensor.memory` if you require zero-copy; a successful import
    /// is not by itself proof of DMA backing.
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
        // Dup the fd — caller retains ownership of the original.
        let borrowed = unsafe { BorrowedFd::borrow_raw(fd) };
        let fd = borrowed.try_clone_to_owned()?;
        let tensor = TensorDyn::from_fd(fd, &shape, dt, name)?;
        Ok(PyTensor(tensor))
    }

    /// Wrap an externally-allocated IOSurface as a Tensor (macOS only).
    ///
    /// `surface_ref` is an `IOSurfaceRef` cast to `int` — typically
    /// obtained via `ctypes` from a CoreVideo / AVFoundation /
    /// VideoToolbox handle, or via `IOSurfaceLookup(id)` to recover a
    /// surface received over XPC. The surface is retained for the
    /// tensor's lifetime; the caller keeps its own reference.
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

    #[getter]
    fn name(&self) -> String {
        self.0.name()
    }

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.0.shape().to_vec()
    }

    #[cfg(unix)]
    #[getter]
    fn fd(&self) -> Result<RawFd> {
        let owned = self.0.clone_fd()?;
        Ok(owned.into_raw_fd())
    }

    fn reshape(&mut self, shape: Vec<usize>) -> Result<()> {
        Ok(self.0.reshape(&shape)?)
    }

    /// Zero-copy rectangular sub-region view of an image tensor — the
    /// destination/source **crop** primitive.
    ///
    /// ``region`` is in pixels of the image's leading frame. The returned view
    /// shares the parent's buffer (and ``BufferIdentity``) with **no copy**,
    /// addressing the sub-rectangle by offset + the parent's row pitch.
    /// ``convert(src, dst.view(region), ...)`` renders into that sub-rectangle.
    /// The parent must be a packed-format image tensor.
    ///
    /// The view exposes the same surface as any other tensor —
    /// :meth:`map` (NumPy buffer protocol), :meth:`from_numpy`, ``fd``.
    #[pyo3(signature = (region))]
    fn view(&self, region: PyRegion) -> Result<Self> {
        let view = self.0.view(region.into())?;
        Ok(PyTensor(view))
    }

    /// Borrow batch element ``n`` of a batched tensor as a zero-copy view.
    ///
    /// A batched tensor prepends ``N`` as the leading dimension over the
    /// per-element image layout (``[N, H, W, C]`` packed or ``[N, C, H, W]``
    /// planar). ``batch(n)`` returns element ``n`` — the contiguous per-element
    /// region at byte offset ``n * element_size``, sharing the parent's buffer.
    /// ``batch(0)`` on a tensor with ``N == 1`` is equivalent to the whole
    /// tensor. This is the destination primitive for assembling a batch into a
    /// single buffer.
    #[pyo3(signature = (n))]
    fn batch(&self, n: usize) -> Result<Self> {
        let view = self.0.batch(n)?;
        Ok(PyTensor(view))
    }

    /// Attach pixel format metadata to this tensor.
    ///
    /// Validates that the tensor's shape is compatible with the format's
    /// layout (packed, planar, or semi-planar). This enables
    /// `from_fd()` tensors to be used as image conversion destinations.
    fn set_format(&mut self, format: PyPixelFormat) -> Result<()> {
        use edgefirst_tensor::PixelFormat;
        let fmt: PixelFormat = format.into();
        Ok(self.0.set_format(fmt)?)
    }

    /// Set this tensor's logical dimensions and pixel format to a decoded
    /// image, reusing the existing allocation.
    ///
    /// Unlike :meth:`set_format` (which only validates the *current* shape
    /// against a format) and :meth:`reshape` (which requires the same
    /// element count), this can shrink or grow the logical shape within the
    /// tensor's allocated capacity -- exactly what a JPEG/PNG decode does to
    /// its destination. This is the Python-visible counterpart of the
    /// decoder's own `Tensor::configure_image`, used by the cross-package
    /// `decode_into`/`decode_file_into` write-back
    /// (`crates/python-common/src/tensor.rs`) to leave a foreign
    /// destination in the same state a same-module decode would.
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
    /// Returns a new file descriptor that the caller must close.
    ///
    /// Raises RuntimeError if the tensor is not DMA-backed or if the
    /// fd clone syscall fails.
    #[cfg(target_os = "linux")]
    fn dmabuf_clone(&self) -> Result<RawFd> {
        let owned = self.0.dmabuf_clone()?;
        Ok(owned.into_raw_fd())
    }

    /// IOSurfaceID for cross-process surface sharing (macOS only).
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

    /// Borrowed `IOSurfaceRef` as an integer (macOS only).
    ///
    /// Use this to hand the surface off to native macOS APIs that take
    /// an IOSurfaceRef directly (CIImage, AVSampleBufferDisplayLayer,
    /// CVPixelBufferCreateWithIOSurface). Wrap with `ctypes.c_void_p(...)`
    /// before passing to a ctypes-bound C function. The pointer's
    /// lifetime is tied to this tensor — do not call CFRelease on it.
    ///
    /// Returns None when the tensor is not IOSurface-backed.
    #[cfg(target_os = "macos")]
    #[getter]
    fn iosurface_ref(&self) -> Option<usize> {
        self.0.iosurface_ref().map(|p| p as usize)
    }

    /// Producer half of the cross-package tensor protocol.
    ///
    /// Returns a ``PyCapsule`` named ``edgefirst_tensor_v1`` wrapping an
    /// ``TensorDesc``. Consumers in *other* EdgeFirst packages read this
    /// rather than type-checking, because each extension module statically
    /// links its own copy of the bindings and therefore has its own PyO3 type
    /// objects — ``isinstance`` across packages would always fail.
    ///
    /// Host memory is only addressable through the descriptor when the tensor
    /// has been pinned; the capsule then holds the pin alive for as long as the
    /// consumer holds the capsule.
    ///
    /// A consumer may call this more than once per operation — for example to
    /// retry with a different ``access`` after an ``access=None`` descriptor
    /// turns out to need a host address after all. Implementations (including
    /// third-party producers) must therefore be side-effect free: repeated
    /// calls, with the same or different ``access``, must be safe and must not
    /// accumulate state.
    ///
    /// Args:
    ///     access: ``None`` (default) requests no pin — the descriptor still
    ///         carries shape, format and the native handle, which is all a
    ///         zero-copy consumer needs. ``"read"``, ``"write"`` or
    ///         ``"readwrite"`` pins host memory with the matching access and
    ///         fills in the descriptor's address.
    ///
    /// Raises:
    ///     ValueError: if ``access`` is not one of the values above.
    #[pyo3(signature = (access=None))]
    fn __edgefirst_tensor__<'py>(
        &self,
        py: Python<'py>,
        access: Option<&str>,
    ) -> PyResult<Bound<'py, pyo3::types::PyCapsule>> {
        use edgefirst_tensor::CpuAccess;

        // Only pin the access the caller actually asked for: `convert(src,
        // dst)` needs just the dma-buf handle from `access=None` on both
        // sides, and a read-only source must not be silently upgraded to
        // ReadWrite.
        let pin = match access {
            None => None,
            Some("read") => Some(self.0.pin_host(CpuAccess::Read).map_err(Error::from)?),
            Some("write") => Some(self.0.pin_host(CpuAccess::Write).map_err(Error::from)?),
            Some("readwrite") => Some(self.0.pin_host(CpuAccess::ReadWrite).map_err(Error::from)?),
            // A bad access string is an invalid argument value, not a tensor
            // error, so raise ValueError directly instead of routing through
            // Error's blanket PyRuntimeError conversion.
            Some(other) => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "access must be None, \"read\", \"write\" or \"readwrite\", got {other:?}"
                )));
            }
        };
        let desc = self.0.descriptor_pinned(pin.as_ref());

        // The capsule owns the descriptor, the pin, and (for a PBO-backed
        // tensor) the vtable keepalive: dropping the capsule releases all
        // three, which is what keeps `desc.ptr` valid for exactly as long as
        // the consumer can see it. `pbo_keepalive()` is `None` for every
        // other kind, so this is unconditional rather than gated on
        // `desc.kind`. `TensorCapsulePayload` is `#[repr(C)]` because this
        // crosses an `.so` boundary -- see its doc comment in `interop.rs`.
        let payload = crate::interop::TensorCapsulePayload {
            desc,
            pin,
            pbo_keepalive: self.0.pbo_keepalive(),
        };
        pyo3::types::PyCapsule::new_with_value(py, payload, c"edgefirst_tensor_v1")
            .map_err(|e| Error::from(std::io::Error::other(e.to_string())).into())
    }

    /// Pin a stable host address for this tensor's data.
    ///
    /// The returned :class:`HostPin` carries no borrow of the tensor, so you
    /// can pin once at init and still pass the tensor to
    /// ``ImageProcessor.convert()`` every frame — the borrow conflict reported
    /// in issue #134. The address stays valid until the pin is released.
    ///
    /// It is **not** a coherency guarantee: bracket CPU access with
    /// :meth:`cpu_access`.
    ///
    /// Args:
    ///     access: ``"read"``, ``"write"`` or ``"readwrite"``.
    ///
    /// Raises:
    ///     RuntimeError: if the backend cannot pin. PBO and AHardwareBuffer
    ///         have no host address outside their map/lock, so use
    ///         :meth:`map` there; the error says so.
    fn pin_host(&self, access: &str) -> Result<PyHostPin> {
        Ok(PyHostPin {
            pin: Some(self.0.pin_host(parse_cpu_access(access)?)?),
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
    /// A no-op on backends that owe no maintenance (``Mem``, ``Shm``), so
    /// portable code can bracket unconditionally. The direction is replayed on
    /// exit: a mismatched pair skips half the cache maintenance.
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
    /// The map owns its coherency bracket in both directions -- unlike
    /// :meth:`pin_host`, which is deliberately decoupled and pairs with
    /// :meth:`cpu_access`.
    ///
    /// `access` selects the direction of that bracket, and it is worth
    /// setting: the default ``"readwrite"`` pays a full-buffer cache
    /// writeback on unmap, which a reader does not need. On a non-coherent
    /// backing (Arm DMA-BUF) that is real per-frame cost; ``"read"`` skips
    /// it, and on macOS takes the read-only IOSurface lock, skipping the
    /// unlock flush. A ``"read"`` view is advertised to the buffer protocol
    /// as read-only, so ``np.asarray`` of it is not writable.
    ///
    /// Args:
    ///     access: ``"read"``, ``"write"`` or ``"readwrite"`` (default).
    ///         ``"none"`` is rejected -- a mapping is CPU access by
    ///         definition.
    #[pyo3(signature = (access=None))]
    fn map(&self, access: Option<&str>) -> Result<PyTensorMap> {
        let access = parse_cpu_access(access.unwrap_or("readwrite"))?;
        if access == tensor::CpuAccess::None {
            return Err(Error::Format(
                "map: access=\"none\" is not a mapping; a mapped view is CPU \
                 access by definition. Use \"read\", \"write\" or \"readwrite\""
                    .to_owned(),
            ));
        }
        // Capture the physical row pitch so the buffer protocol can expose a
        // padded (DMA / GPU pitch-aligned) backing as a correctly-strided view.
        // Only image tensors (pixel format set) have a row pitch: a formatless
        // DMA tensor's plane-0 stride is the whole allocation, and treating
        // that as strides[0] makes numpy `.fill` walk off the mapping.
        let row_stride = self.0.format().and(self.0.effective_row_stride());
        let mapped = map_tensor_dyn(&self.0, access)?;
        Ok(PyTensorMap {
            // Derived from the mapping, not from `access`: see
            // `TensorMapT::is_writable`.
            readonly: !mapped.is_writable(),
            mapped: Some(mapped),
            row_stride,
        })
    }

    /// Attempt a zero-copy CUDA device-pointer mapping.
    ///
    /// Returns a ``CudaMap`` context manager exposing ``device_ptr`` and
    /// ``size``, or ``None`` if CUDA is unavailable for this tensor (libcudart
    /// not found, or the tensor was not registered with CUDA). Fast-fails to
    /// ``None`` without GL-thread routing.
    ///
    /// The recommended pattern is to try ``cuda_map()`` first and fall back to
    /// ``map()`` when it returns ``None``::
    ///
    ///     cm = tensor.cuda_map()
    ///     if cm is not None:
    ///         with cm as m:
    ///             trt_set_input_address(m.device_ptr)   # zero-copy GPU
    ///     else:
    ///         with tensor.map() as host:
    ///             run_cpu_path(host)                    # fallback
    fn cuda_map(slf: Bound<'_, Self>) -> Option<PyCudaMap> {
        let owner: Py<PyTensor> = slf.clone().unbind();
        // SAFETY: The CudaMap borrows into the tensor's storage. We extend
        // the lifetime to 'static and keep the PyTensor alive via `_owner`.
        // `PyCudaMap.map` is declared before `_owner`, so Rust drops the
        // CudaMap guard before decrementing the tensor ref-count. Callers
        // must not reshape the tensor while a CudaMap is live.
        let borrowed = slf.borrow();
        let map = borrowed.0.cuda_map()?;
        let map_static: edgefirst_tensor::CudaMap<'static> = unsafe { std::mem::transmute(map) };
        Some(PyCudaMap {
            map: Some(map_static),
            _owner: owner,
        })
    }

    // ── Image-specific methods ──────────────────────────────────────────

    /// Create an image tensor with the given dimensions and pixel format.
    ///
    /// `access` declares CPU access (`"none"`, `"read"`, `"write"`,
    /// `"readwrite"`); hardware access is always implied. Scripts that
    /// `map()` or `numpy()` the tensor should pass `access="readwrite"`.
    #[staticmethod]
    #[pyo3(signature = (width, height, format, mem = None, access = "none"))]
    fn image(
        width: usize,
        height: usize,
        format: PyPixelFormat,
        mem: Option<PyTensorMemory>,
        access: &str,
    ) -> Result<Self> {
        use edgefirst_tensor::PixelFormat;
        let fmt: PixelFormat = format.into();
        let memory = mem.map(|x| x.into());
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
    /// pixels. Use this to allocate a tensor at the right size before
    /// calling ``decode_image``.
    ///
    /// The reported dimensions are unrotated (the decode never applies EXIF
    /// rotation); ``rotation_degrees`` / ``flip_horizontal`` report the EXIF
    /// orientation so callers can apply it themselves if desired.
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
    ///
    /// Gated on the `image` feature: JPEG encoding lives in edgefirst-image,
    /// and pulling it in unconditionally would make `edgefirst.tensor` and
    /// `edgefirst.codec` link the OpenGL stack they never use.
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
    /// to match. The decode never rotates or flips; if you need RGB, decode
    /// then call ``ImageProcessor.convert(...)``.
    ///
    /// Returns an ``ImageInfo`` with the native ``width``, ``height``,
    /// ``format``, ``row_stride``, and the EXIF ``rotation_degrees`` /
    /// ``flip_horizontal``. The tensor must have sufficient capacity for the
    /// decoded image (it is reconfigured within that capacity).
    ///
    /// This is the preferred API for real-time pipelines: allocate once via
    /// ``ImageProcessor.create_image()``, then call ``decode_image()`` in
    /// the main loop to avoid per-frame allocations.
    ///
    /// Call :func:`set_output_format` beforehand to opt a JPEG source into a
    /// fused ``Rgb``/``Nv12`` output instead, computed in the same decode
    /// pass.
    ///
    /// Args:
    ///     data: Raw JPEG or PNG bytes.
    #[cfg(feature = "codec")]
    fn decode_image<'py>(mut self_: PyRefMut<'py, Self>, data: &[u8]) -> Result<PyImageInfo> {
        let py = self_.py();
        // Copy while the GIL is still held -- `data` borrows a Python
        // `bytes` object tied to this call's GIL token and cannot itself
        // cross `py.detach`. The decode this feeds is a software JPEG/PNG
        // decode, orders of magnitude more expensive than one linear copy
        // of the (compressed, therefore smaller-than-the-output) input.
        let data = data.to_vec();
        if self_.0.memory() == TensorMemory::Pbo {
            // A GL-PBO-backed tensor: `crate::interop::TensorArg::
            // can_detach`'s docs explain why `reconstruct` below cannot
            // build an independent value for it, so this call keeps the
            // GIL held for its whole duration, exactly this crate's
            // behaviour before GIL release existed for tensors at all.
            return decode_image_into(&mut self_.0, &data);
        }
        // Reconstruct an independent `TensorDyn` aliasing the same backing
        // memory as `self_`'s own, rather than a raw pointer straight at
        // the live `PyTensor`'s field -- see `crate::interop::
        // RawTensorAccess`'s docs for why the latter is unsound to
        // dereference once `self_`'s borrow is dropped below. Called
        // directly (not via `TensorArg::into_raw_access`) so `self_bound`
        // below keeps a *typed* handle rather than one folded into an
        // opaque `Py<PyAny>` keepalive -- the decode's resulting
        // format/shape/colorimetry need writing back onto the real object
        // once `py.detach` returns (see `write_back_decode`).
        let (mut tensor, _pin) =
            crate::interop::reconstruct(&self_.0, tensor::CpuAccess::ReadWrite)
                .map_err(|e| Error::Format(e.to_string()))?;
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
        // Copy while the GIL is still held -- see `decode_image`.
        let filename = filename.to_string();
        if self_.0.memory() == TensorMemory::Pbo {
            // See `decode_image`: a GL-PBO-backed tensor keeps the GIL held.
            let data = std::fs::read(&filename).map_err(Error::Io)?;
            return decode_image_into(&mut self_.0, &data);
        }
        // See `decode_image` for why this reconstructs rather than going
        // through `TensorArg::into_raw_access`.
        let (mut tensor, _pin) =
            crate::interop::reconstruct(&self_.0, tensor::CpuAccess::ReadWrite)
                .map_err(|e| Error::Format(e.to_string()))?;
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
    /// reflects any DMA pitch-alignment padding applied to the row.
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
    /// Normalize into a numpy array.
    ///
    /// Gated on the `image` feature: normalization lives in the image module,
    /// and gating it keeps `edgefirst.tensor` and `edgefirst.codec` from
    /// linking the OpenGL stack they never use.
    #[cfg(feature = "image")]
    #[pyo3(signature = (dst, normalization=crate::image::Normalization::DEFAULT, zero_point=None))]
    fn normalize_to_numpy(
        &self,
        dst: crate::image::ImageDest3,
        normalization: crate::image::Normalization,
        zero_point: Option<i64>,
    ) -> Result<()> {
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
    /// Faithful: never auto-filled. Set to ``None`` to clear.
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

// ─── PyCudaMap ──────────────────────────────────────────────────────────────

/// Scoped zero-copy CUDA device-pointer mapping for a tensor (e.g. a TensorRT
/// input). Use as a context manager; the mapping is released on exit so the
/// GPU buffer can be reused by the next convert(). Obtain via Tensor.cuda_map().
#[pyclass(name = "CudaMap", module = "edgefirst.tensor")]
pub struct PyCudaMap {
    // FIELD ORDER IS LOAD-BEARING: `map` must be declared before `_owner` so
    // that Rust drops `map` (the CudaMap guard) before `_owner` (the PyTensor
    // ref-count). This preserves the invariant that the CUDA mapping is
    // released before the tensor can be freed.
    map: Option<edgefirst_tensor::CudaMap<'static>>,
    _owner: Py<PyTensor>,
}

// SAFETY: CudaMap holds a *mut c_void CUDA device pointer. CUDA device
// pointers are process-global and may be used from any thread, so Send+Sync
// are sound here. The owning PyTensor Py<> handle is also Send+Sync.
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

    /// Release the CUDA mapping (idempotent). Called automatically on ``with``
    /// exit; may also be called explicitly when early release is needed.
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
/// Carries **no borrow of the tensor**, so a caller can pin once at init and
/// still pass the tensor to ``ImageProcessor.convert()`` every frame — the
/// borrow conflict reported in issue #134.
///
/// The address is not a coherency guarantee. Bracket CPU access with
/// :meth:`Tensor.cpu_access`, which is a no-op where the backend owes nothing.
#[pyclass(name = "HostPin", module = "edgefirst.tensor")]
pub struct PyHostPin {
    pin: Option<edgefirst_tensor::HostPin<'static>>,
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
    /// The tensor's **logical** length, not the allocation's capacity — a
    /// backend may round up, and reporting that would let a caller read
    /// padding as data. For a stride-padded image use :meth:`Tensor.map`,
    /// which exposes the padded extent.
    #[getter]
    fn len(&self) -> usize {
        self.pin.as_ref().map_or(0, |p| p.len())
    }

    /// Alignment of the pinned address in bytes.
    ///
    /// TFLite requires 64-byte alignment unless the caller opts out, and
    /// upstream warns that opting out can crash in ``Invoke()``.
    #[getter]
    fn alignment(&self) -> usize {
        self.pin.as_ref().map_or(0, |p| p.alignment())
    }

    /// Release the pin. Idempotent.
    fn release(&mut self) {
        self.pin = None;
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
/// One guard parameterised by direction, rather than a class per direction:
/// ``CpuAccess`` has four variants and only the caller knows which applies.
/// The direction is replayed on exit because a mismatched pair skips half the
/// cache maintenance — a read-only bracket lets the kernel skip the writeback,
/// a write-only one skips the invalidate.
#[pyclass(name = "CpuAccessGuard", module = "edgefirst.tensor")]
pub struct PyCpuAccessGuard {
    /// A Python reference, not a clone: `TensorDyn` is not `Clone`, and the
    /// guard must keep the tensor alive for the bracket's duration anyway.
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
    /// True when the map was taken read-only, so `__getbuffer__` must
    /// advertise the view as such. A writable view over a read-only map
    /// would let a consumer write bytes that the unmap deliberately does
    /// not flush -- silently discarded on a non-coherent backing.
    pub(crate) readonly: bool,
    /// Physical row pitch in bytes for image tensors, captured from
    /// `effective_row_stride()` at map time. `Some` for **any** image tensor
    /// that has a pixel format set (including DMA, IOSurface, and
    /// self-allocated semi-planar tensors whose stride is always
    /// 64-byte-aligned). `None` only for non-image tensors or tensors without
    /// a pixel format. The `__getbuffer__` impl applies the padded stride only
    /// when `rs > strides[0]` (tight buffers pass through unchanged).
    ///
    /// Read **only** by `__getbuffer__`, which is itself gated on
    /// `any(not(Py_LIMITED_API), Py_3_11)` — the buffer protocol is not part
    /// of the stable ABI before 3.11. The field stays unconditional so the
    /// construction sites need no cfg, and dead_code is silenced in exactly
    /// the configuration that compiles the reader out. That configuration is
    /// not hypothetical: `.github/workflows/release.yml` builds
    /// `--features abi3-py38`, so `-D warnings` fails there without this.
    #[cfg_attr(all(Py_LIMITED_API, not(Py_3_11)), allow(dead_code))]
    pub(crate) row_stride: Option<usize>,
}

unsafe impl Send for PyTensorMap {}
unsafe impl Sync for PyTensorMap {}

#[pymethods]
impl PyTensorMap {
    fn unmap(&mut self) {
        if let Some(map) = &mut self.mapped {
            map.unmap();
            self.mapped = None;
        }
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
        // Checked before reaching the map: `set_value_at` asserts on a
        // read-only view, and that panic crosses the FFI boundary as a
        // `PanicException`, which derives from `BaseException` and so escapes
        // an ordinary `except Exception` handler.
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

        // PEP 3118: a consumer that asks for a writable buffer must be
        // refused rather than handed a read-only one. Silently downgrading
        // is the dangerous outcome here -- the caller would write bytes that
        // this map's unmap deliberately does not flush, so on a non-coherent
        // backing they would be discarded without a diagnostic.
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

        // Compute C-contiguous strides: strides[i] = itemsize * product(shape[i+1..])
        let itemsize = mapped.element_size() as isize;
        let mut strides = vec![0isize; ndim];
        if ndim > 0 {
            strides[ndim - 1] = itemsize;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }

        // Default (tight / contiguous) byte length.
        let mut buf_len = mapped.size() as isize;

        // Row-padded image backing (DMA / GPU pitch alignment): `map().as_slice()`
        // exposes the full padded buffer, so the rows sit at `row_stride` bytes,
        // wider than the tight outer stride. Expose that pitch as the outer
        // (row) stride and widen `len` to span the padded buffer, so a consumer
        // (`np.asarray(memoryview(map))`) reads the logical pixels zero-copy
        // instead of a sheared contiguous reinterpretation. Tight buffers
        // (`row_stride == strides[0]`, or non-image `None`) are unchanged.
        if ndim > 0 {
            if let Some(rs) = slf2.row_stride {
                let rs = rs as isize;
                // A row pitch cannot be the whole mapping: that is the
                // formatless plane-0 fallback (capacity as stride), not padding.
                if rs > strides[0] && rs < buf_len {
                    strides[0] = rs;
                    buf_len = rs * shape[0];
                }
            }
        }

        // Box both arrays together so we can recover the length in __releasebuffer__.
        // Store (shape_ptr, strides_ptr, ndim) using view.internal.
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
            // Store ndim in internal so __releasebuffer__ can reconstruct the slices.
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
    }

    /// Zero-copy NumPy array over the mapped buffer, honouring DMA row
    /// padding the same way ``memoryview(self)`` / ``__getbuffer__`` does.
    ///
    /// On the abi3 wheel ``__getbuffer__`` is compiled out, so this path
    /// constructs the strided array itself.
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
            let itemsize = mapped.element_size();
            let ndim = shape.len();
            let mut strides = vec![0isize; ndim];
            if ndim > 0 {
                strides[ndim - 1] = itemsize as isize;
                for i in (0..ndim - 1).rev() {
                    strides[i] = strides[i + 1] * shape[i + 1] as isize;
                }
            }
            let mut buf_len = mapped.size();
            if ndim > 0 {
                if let Some(rs) = borrow.row_stride {
                    if (rs as isize) > strides[0] && rs < buf_len {
                        strides[0] = rs as isize;
                        buf_len = rs * shape[0];
                    }
                }
            }
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

// ---------------------------------------------------------------------------
// Moved from image.rs: both wrap edgefirst_tensor types (PixelFormat, Region),
// so they are tensor-level concepts. Living in image.rs forced tensor.rs to
// depend on image.rs, which meant every extension module linked the OpenGL
// stack whether it used it or not.
// ---------------------------------------------------------------------------

/// Pixel format for image tensors.
///
/// Each variant maps directly to an `edgefirst_tensor::PixelFormat` value.
// No `eq`/`eq_int` -- see `PyTensorMemory`'s comment above; `__eq__`/`__ne__`
// are hand-written below instead, for the cross-package fallback.
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

// Same reasoning as `PyTensorMemory`'s const assertions above: these
// literals are copied from `PixelFormat::code()`, and this is what stops
// that copy drifting silently -- a mismatch fails the build.
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

    /// See `PyTensorMemory::__hash__` -- same discriminant-is-the-hash story.
    fn __hash__(&self) -> isize {
        *self as isize
    }
}

impl PyPixelFormat {
    /// Reconstruct from the `__int__()` discriminant of a sibling package's
    /// copy of this enum. See `extract_eq_int_enum`.
    ///
    /// Delegates to `PixelFormat::from_code` -- see `PyTensorMemory::
    /// from_discriminant`'s doc comment for why a hand-written match here
    /// is the parallel table this vocabulary work removes. Every code
    /// `PixelFormat::from_code` accepts today has a same-named Python
    /// variant (`TryFrom<PixelFormat> for PyPixelFormat` below is total
    /// over the current set), so this only returns `None` for a code that
    /// is unassigned on the Rust side too, or -- once `PixelFormat` grows a
    /// variant upstream without a matching Python addition -- one that
    /// resolves on the Rust side but has no Python counterpart yet.
    fn from_discriminant(v: i64) -> Option<Self> {
        u32::try_from(v)
            .ok()
            .and_then(PixelFormat::from_code)
            .and_then(|f| PyPixelFormat::try_from(f).ok())
    }
}

/// See `PyTensorMemory`'s `FromPyObject` impl -- same cross-package story.
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

// No `eq` here -- see `PyTensorMemory`'s comment above; `__eq__`/`__ne__`
// are hand-written below instead, for the cross-package fallback.
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

    /// Cross-package equality: `#[pyclass(eq)]`'s auto-generated `__eq__`
    /// resolves `other` by native identity only, so a sibling package's
    /// `Region` with identical `x`/`y`/`width`/`height` compared unequal --
    /// silently, never an error. Reuses this type's own `FromPyObject`
    /// (native downcast first, then the four field getters) so `==`/`!=`
    /// agree with what `Region` already accepts as a value.
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

    /// Equal objects must hash equal, and `__eq__` above compares exactly
    /// `x`/`y`/`width`/`height` -- so the hash mixes exactly those four
    /// fields, in this fixed order, and nothing else.
    ///
    /// Deliberately NOT routed through `std::hash::Hash`/`DefaultHasher`:
    /// every `edgefirst.*` extension module must compute the identical
    /// hash for the identical value (each is a separately compiled copy of
    /// this same source), and while `DefaultHasher`'s *seed* is fixed
    /// (`SipHasher13::new_with_keys(0, 0)`), its *algorithm* is documented
    /// as unspecified across std/compiler versions -- a future split
    /// toolchain build could silently reintroduce the exact bug this hash
    /// exists to prevent. Mixing the fields here with primitive,
    /// unconditionally well-defined operations (rotate/xor/wrapping-mul)
    /// removes that cross-build assumption instead of merely documenting
    /// it. The constant is FxHash's.
    fn __hash__(&self) -> u64 {
        const K: u64 = 0x517c_c1b7_2722_0a95;
        let mut h: u64 = 0;
        for v in [self.x, self.y, self.width, self.height] {
            h = (h.rotate_left(5) ^ v as u64).wrapping_mul(K);
        }
        h
    }
}

/// `Region` is a plain value struct, not an `eq_int` enum, so the fallback
/// here reads the four fields back via `getattr` instead of a discriminant
/// -- a sibling package's `Region` exposes the same `x`/`y`/`width`/`height`
/// properties (see `PyTensorMemory`'s `FromPyObject` impl for the identity
/// story this is working around).
///
/// Gated on `type(obj).__name__ == "Region"` first. Without this gate, the
/// getattr fallback is pure structural duck typing: *any* object exposing
/// `x`/`y`/`width`/`height` -- not just a sibling package's `Region` -- would
/// be accepted and compare equal. Every `edgefirst.*` package's copy of
/// `Region` shares the same `#[pyclass(name = "Region")]`, so the check
/// still accepts a sibling package's copy while rejecting an unrelated
/// same-shaped type.
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
