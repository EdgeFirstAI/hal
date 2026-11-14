// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `edgefirst.codec`: engines and capabilities, still-image encoding, and
//! H.264 and MJPEG video, with tensors as the pictures.

use std::sync::{Arc, Mutex, Weak};

use edgefirst_codec::video::{
    self, Bitrate, DecodeEvent, DecodedFrame, DecoderConfig, EncoderConfig, FrameOptions,
    OutputPool,
};
use edgefirst_codec::{Codec, CodecError as RsCodecError, Direction, Engine, Transfer};
use edgefirst_tensor::{PixelFormat, TensorDyn};
use pyo3::create_exception;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::colorimetry::PyColorimetry;
use crate::interop::{RawKeepalive, TensorArg};
use crate::tensor::{PyPixelFormat, PyRegion, PyTensor};

create_exception!(
    edgefirst.codec,
    CodecError,
    PyRuntimeError,
    "A codec failure. Each kind below is a subclass; a refused CPU access is a PermissionError and a missing file a FileNotFoundError instead."
);
create_exception!(
    edgefirst.codec,
    NoDeviceError,
    CodecError,
    "No engine or device here for the request."
);
create_exception!(
    edgefirst.codec,
    UnsupportedError,
    CodecError,
    "The engine exists, but not for this codec, format or feature."
);
create_exception!(
    edgefirst.codec,
    PoolExhaustedError,
    CodecError,
    "Every frame of the decoder's pool is held: release frames and decode the same access unit again."
);
create_exception!(
    edgefirst.codec,
    CodecTimeoutError,
    CodecError,
    "The device did not finish in time."
);
create_exception!(
    edgefirst.codec,
    InvalidConfigError,
    CodecError,
    "A configuration the engine cannot open, or a provided pool it cannot use."
);
create_exception!(
    edgefirst.codec,
    InvalidInputError,
    CodecError,
    "A bad argument or source picture."
);
create_exception!(
    edgefirst.codec,
    InvalidDataError,
    CodecError,
    "Compressed data the decoder cannot parse."
);

/// The Python exception for a codec error.
pub(crate) fn to_pyerr(e: RsCodecError) -> PyErr {
    if e.cpu_access_denied().is_some() {
        return pyo3::exceptions::PyPermissionError::new_err(e.to_string());
    }
    let msg = e.to_string();
    match e {
        RsCodecError::Io(io) => PyErr::from(io),
        RsCodecError::NoDevice(_) => NoDeviceError::new_err(msg),
        RsCodecError::Unsupported(_)
        | RsCodecError::UnsupportedFormat(_)
        | RsCodecError::UnsupportedDtype(_) => UnsupportedError::new_err(msg),
        RsCodecError::PoolExhausted { .. } => PoolExhaustedError::new_err(msg),
        RsCodecError::Timeout(_) => CodecTimeoutError::new_err(msg),
        RsCodecError::InvalidConfig(_) => InvalidConfigError::new_err(msg),
        RsCodecError::InvalidInput(_) => InvalidInputError::new_err(msg),
        RsCodecError::InvalidData(_) => InvalidDataError::new_err(msg),
        _ => CodecError::new_err(msg),
    }
}

fn unknown(what: &str, value: impl std::fmt::Debug) -> PyErr {
    PyValueError::new_err(format!("{value:?} has no {what} in this version"))
}

/// A compressed image or video format.
#[pyclass(name = "Codec", eq, eq_int, from_py_object, module = "edgefirst.codec")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyCodec {
    /// JPEG still images.
    Jpeg = 0,
    /// PNG still images.
    Png = 1,
    /// H.264, as an Annex B byte stream.
    H264 = 2,
    /// HEVC. No engine runs it yet.
    Hevc = 3,
    /// Motion JPEG: a stream of JPEG pictures.
    Mjpeg = 4,
    /// VP8. No engine runs it yet.
    Vp8 = 5,
    /// VP9. No engine runs it yet.
    Vp9 = 6,
    /// AV1. No engine runs it yet.
    Av1 = 7,
}

impl From<PyCodec> for Codec {
    fn from(c: PyCodec) -> Self {
        match c {
            PyCodec::Jpeg => Codec::Jpeg,
            PyCodec::Png => Codec::Png,
            PyCodec::H264 => Codec::H264,
            PyCodec::Hevc => Codec::Hevc,
            PyCodec::Mjpeg => Codec::Mjpeg,
            PyCodec::Vp8 => Codec::Vp8,
            PyCodec::Vp9 => Codec::Vp9,
            PyCodec::Av1 => Codec::Av1,
        }
    }
}

impl TryFrom<Codec> for PyCodec {
    type Error = PyErr;
    fn try_from(c: Codec) -> PyResult<Self> {
        Ok(match c {
            Codec::Jpeg => PyCodec::Jpeg,
            Codec::Png => PyCodec::Png,
            Codec::H264 => PyCodec::H264,
            Codec::Hevc => PyCodec::Hevc,
            Codec::Mjpeg => PyCodec::Mjpeg,
            Codec::Vp8 => PyCodec::Vp8,
            Codec::Vp9 => PyCodec::Vp9,
            Codec::Av1 => PyCodec::Av1,
            other => return Err(unknown("Codec", other)),
        })
    }
}

#[pymethods]
impl PyCodec {
    /// The codec's name, for example ``"H.264"``.
    fn __str__(&self) -> String {
        Codec::from(*self).to_string()
    }
}

/// What runs a codec.
#[pyclass(
    name = "Engine",
    eq,
    eq_int,
    from_py_object,
    module = "edgefirst.codec"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyEngine {
    /// The first engine that can do the request, in the platform's order.
    Auto = 0,
    /// Linux V4L2 memory-to-memory devices.
    V4l2 = 1,
    /// Windows Media Foundation transforms.
    MediaFoundation = 2,
    /// Apple VideoToolbox. No backend runs it yet.
    VideoToolbox = 3,
    /// NVIDIA nvJPEG (JPEG decoding on Linux).
    NvJpeg = 4,
    /// The CPU, for still images only.
    Cpu = 5,
}

impl From<PyEngine> for Engine {
    fn from(e: PyEngine) -> Self {
        match e {
            PyEngine::Auto => Engine::Auto,
            PyEngine::V4l2 => Engine::V4l2,
            PyEngine::MediaFoundation => Engine::MediaFoundation,
            PyEngine::VideoToolbox => Engine::VideoToolbox,
            PyEngine::NvJpeg => Engine::NvJpeg,
            PyEngine::Cpu => Engine::Cpu,
        }
    }
}

impl TryFrom<Engine> for PyEngine {
    type Error = PyErr;
    fn try_from(e: Engine) -> PyResult<Self> {
        Ok(match e {
            Engine::Auto => PyEngine::Auto,
            Engine::V4l2 => PyEngine::V4l2,
            Engine::MediaFoundation => PyEngine::MediaFoundation,
            Engine::VideoToolbox => PyEngine::VideoToolbox,
            Engine::NvJpeg => PyEngine::NvJpeg,
            Engine::Cpu => PyEngine::Cpu,
            other => return Err(unknown("Engine", other)),
        })
    }
}

#[pymethods]
impl PyEngine {
    /// The engine's name, as ``EDGEFIRST_CODEC_ENGINE`` spells it, for
    /// example ``"media-foundation"``.
    fn __str__(&self) -> String {
        Engine::from(*self).to_string()
    }

    /// The engine a name stands for, as ``EDGEFIRST_CODEC_ENGINE`` spells
    /// it (``"v4l2"``, ``"mf"``, ``"cpu"``, ...). Raises ``ValueError`` for an
    /// unknown name.
    #[staticmethod]
    fn from_name(name: &str) -> PyResult<Self> {
        let engine: Engine = name
            .parse()
            .map_err(|e: RsCodecError| PyValueError::new_err(e.to_string()))?;
        engine.try_into()
    }
}

/// Encoding or decoding.
#[pyclass(
    name = "Direction",
    eq,
    eq_int,
    from_py_object,
    module = "edgefirst.codec"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyDirection {
    /// Images to compressed data.
    Encode = 0,
    /// Compressed data to images.
    Decode = 1,
}

impl From<Direction> for PyDirection {
    fn from(d: Direction) -> Self {
        match d {
            Direction::Encode => PyDirection::Encode,
            Direction::Decode => PyDirection::Decode,
        }
    }
}

/// How pictures move between tensors and the engine.
#[pyclass(
    name = "Transfer",
    eq,
    eq_int,
    from_py_object,
    module = "edgefirst.codec"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyTransfer {
    /// The engine reads or writes the tensors' own memory.
    ZeroCopy = 0,
    /// One copy on the GPU, with no CPU access to the pixels.
    GpuCopy = 1,
    /// The pixels pass through the CPU.
    CpuCopy = 2,
}

impl TryFrom<Transfer> for PyTransfer {
    type Error = PyErr;
    fn try_from(t: Transfer) -> PyResult<Self> {
        Ok(match t {
            Transfer::ZeroCopy => PyTransfer::ZeroCopy,
            Transfer::GpuCopy => PyTransfer::GpuCopy,
            Transfer::CpuCopy => PyTransfer::CpuCopy,
            other => return Err(unknown("Transfer", other)),
        })
    }
}

/// H.264 profile (ITU-T H.264 Annex A).
#[pyclass(
    name = "H264Profile",
    eq,
    eq_int,
    from_py_object,
    module = "edgefirst.codec"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyH264Profile {
    Baseline = 0,
    ConstrainedBaseline = 1,
    Main = 2,
    High = 3,
}

impl From<PyH264Profile> for video::H264Profile {
    fn from(p: PyH264Profile) -> Self {
        match p {
            PyH264Profile::Baseline => video::H264Profile::Baseline,
            PyH264Profile::ConstrainedBaseline => video::H264Profile::ConstrainedBaseline,
            PyH264Profile::Main => video::H264Profile::Main,
            PyH264Profile::High => video::H264Profile::High,
        }
    }
}

/// H.264 level (ITU-T H.264 Table A-1).
#[pyclass(
    name = "H264Level",
    eq,
    eq_int,
    from_py_object,
    module = "edgefirst.codec"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum PyH264Level {
    L1_0 = 0,
    L1b = 1,
    L1_1 = 2,
    L1_2 = 3,
    L1_3 = 4,
    L2_0 = 5,
    L2_1 = 6,
    L2_2 = 7,
    L3_0 = 8,
    L3_1 = 9,
    L3_2 = 10,
    L4_0 = 11,
    L4_1 = 12,
    L4_2 = 13,
    L5_0 = 14,
    L5_1 = 15,
    L5_2 = 16,
    L6_0 = 17,
    L6_1 = 18,
    L6_2 = 19,
}

impl From<PyH264Level> for video::H264Level {
    fn from(l: PyH264Level) -> Self {
        use video::H264Level as L;
        match l {
            PyH264Level::L1_0 => L::L1_0,
            PyH264Level::L1b => L::L1b,
            PyH264Level::L1_1 => L::L1_1,
            PyH264Level::L1_2 => L::L1_2,
            PyH264Level::L1_3 => L::L1_3,
            PyH264Level::L2_0 => L::L2_0,
            PyH264Level::L2_1 => L::L2_1,
            PyH264Level::L2_2 => L::L2_2,
            PyH264Level::L3_0 => L::L3_0,
            PyH264Level::L3_1 => L::L3_1,
            PyH264Level::L3_2 => L::L3_2,
            PyH264Level::L4_0 => L::L4_0,
            PyH264Level::L4_1 => L::L4_1,
            PyH264Level::L4_2 => L::L4_2,
            PyH264Level::L5_0 => L::L5_0,
            PyH264Level::L5_1 => L::L5_1,
            PyH264Level::L5_2 => L::L5_2,
            PyH264Level::L6_0 => L::L6_0,
            PyH264Level::L6_1 => L::L6_1,
            PyH264Level::L6_2 => L::L6_2,
        }
    }
}

/// Whose memory a decoder writes its pictures to.
#[pyclass(
    name = "OutputMemory",
    eq,
    eq_int,
    from_py_object,
    module = "edgefirst.codec"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyOutputMemory {
    /// The pool's buffers (allocated by the decoder or provided by the
    /// caller), which the device imports or the decoder copies into.
    Imported = 0,
    /// The device's own buffers, exported because it would not import the
    /// pool's. CPU reads of these may be uncached and slow.
    Exported = 1,
}

/// A JPEG quality, refused outside 1 to 100 as the codec refuses it.
fn jpeg_quality(quality: i64) -> PyResult<u8> {
    u8::try_from(quality)
        .ok()
        .filter(|q| (1..=100).contains(q))
        .ok_or_else(|| {
            InvalidConfigError::new_err(format!("JPEG quality {quality} is not within 1 to 100"))
        })
}

fn pixel_format(f: PixelFormat) -> PyResult<PyPixelFormat> {
    PyPixelFormat::try_from(f).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// What one engine does with one codec in one direction.
#[pyclass(
    name = "Capability",
    frozen,
    get_all,
    skip_from_py_object,
    module = "edgefirst.codec"
)]
#[derive(Debug, Clone)]
pub struct PyCapability {
    /// The engine.
    engine: PyEngine,
    /// The device within the engine: a device node, a transform's name, or
    /// ``"cpu"``.
    device: String,
    /// The compressed format.
    codec: PyCodec,
    /// Encoding or decoding.
    direction: PyDirection,
    /// Whether the work runs on dedicated hardware rather than the CPU.
    hardware: bool,
    /// The pixel formats an encoder takes or a decoder produces.
    formats: Vec<PyPixelFormat>,
    /// How pictures move in the best case.
    transfer: PyTransfer,
}

#[pymethods]
impl PyCapability {
    fn __repr__(&self) -> String {
        format!(
            "Capability(engine={:?}, device={:?}, codec={:?}, direction={:?}, hardware={}, formats={:?}, transfer={:?})",
            self.engine, self.device, self.codec, self.direction, self.hardware, self.formats, self.transfer
        )
    }
}

/// What the codec engines on this machine can do, engine by engine, in the
/// order ``Engine.Auto`` tries them.
///
/// Probing opens no stream, but it opens device nodes and enumerates (on
/// Windows, activates) transforms, so it is not free: call it once.
#[pyfunction]
pub fn capabilities(py: Python<'_>) -> PyResult<Vec<PyCapability>> {
    py.detach(edgefirst_codec::capabilities)
        .into_iter()
        // A capability this version of the bindings cannot name is skipped.
        .filter_map(|c| {
            let item = (|| -> PyResult<PyCapability> {
                Ok(PyCapability {
                    engine: c.engine.try_into()?,
                    device: c.device.clone(),
                    codec: c.codec.try_into()?,
                    direction: c.direction.into(),
                    hardware: c.hardware,
                    formats: c
                        .formats
                        .iter()
                        .filter_map(|&f| pixel_format(f).ok())
                        .collect(),
                    transfer: c.transfer.try_into()?,
                })
            })();
            item.ok()
        })
        .map(Ok)
        .collect()
}

/// Runs `body` on `src` with the GIL released where the tensor can be
/// resolved independently of its Python object, and with it held otherwise.
fn with_source<R: Send>(
    py: Python<'_>,
    src: &Bound<'_, PyAny>,
    body: impl FnOnce(&TensorDyn) -> R + Send,
) -> PyResult<R> {
    let arg = TensorArg::extract(src, None)?;
    if arg.can_detach() {
        let raw = arg.into_raw_access()?;
        Ok(py.detach(move || body(raw.as_ref())))
    } else {
        Ok(body(arg.as_ref()))
    }
}

/// Encodes tensors as still images: JPEG.
///
/// Under ``Engine.Auto`` a V4L2 JPEG encoder takes DMA-BUF sources it can
/// read and the CPU takes the rest. The CPU encodes ``uint8`` NV12, NV16,
/// YUYV, GREY, RGB and RGBA images at any row stride.
#[pyclass(name = "ImageEncoder", module = "edgefirst.codec")]
pub struct PyImageEncoder {
    /// Behind a lock because a V4L2 encoder is not `Sync`.
    inner: Mutex<ImageEncoderState>,
}

struct ImageEncoderState {
    encoder: edgefirst_codec::ImageEncoder,
    /// The last file, reused by the next encode.
    out: Vec<u8>,
}

#[pymethods]
impl PyImageEncoder {
    /// Args:
    ///     codec: The format to write. Only ``Codec.Jpeg`` is encoded.
    ///     quality: JPEG quality, 1 (smallest) to 100 (best); 85 when omitted.
    ///     engine: ``Engine.Auto``, ``Engine.Cpu`` or ``Engine.V4l2``.
    #[new]
    #[pyo3(signature = (codec = PyCodec::Jpeg, *, quality = None, engine = PyEngine::Auto))]
    fn new(codec: PyCodec, quality: Option<i64>, engine: PyEngine) -> PyResult<Self> {
        let mut inner = edgefirst_codec::ImageEncoder::new(codec.into()).map_err(to_pyerr)?;
        if let Some(q) = quality {
            inner.set_quality(jpeg_quality(q)?).map_err(to_pyerr)?;
        }
        inner.set_engine(engine.into());
        Ok(Self {
            inner: Mutex::new(ImageEncoderState {
                encoder: inner,
                out: Vec::new(),
            }),
        })
    }

    /// JPEG quality, 1 (smallest) to 100 (best).
    #[getter]
    fn quality(&self) -> u8 {
        locked(&self.inner).encoder.quality()
    }

    #[setter]
    fn set_quality(&self, quality: i64) -> PyResult<()> {
        locked(&self.inner)
            .encoder
            .set_quality(jpeg_quality(quality)?)
            .map_err(to_pyerr)
    }

    /// The engine selected: ``Engine.Auto`` or one engine used alone.
    #[getter]
    fn engine(&self) -> PyResult<PyEngine> {
        locked(&self.inner).encoder.engine().try_into()
    }

    #[setter]
    fn set_engine(&self, engine: PyEngine) {
        locked(&self.inner).encoder.set_engine(engine.into());
    }

    /// Encodes ``src``, an image tensor from any ``edgefirst`` package, and
    /// returns the file's bytes.
    ///
    /// Raises:
    ///     PermissionError: the CPU must read a tensor not declared readable.
    ///     UnsupportedError: no engine takes the source's format or type.
    ///     InvalidInputError: a bad source, for example an odd-width YUYV image.
    ///     NoDeviceError: the engine selected by name is not present.
    fn encode<'py>(
        &self,
        py: Python<'py>,
        src: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let inner = &self.inner;
        let file = with_source(py, src, |t| {
            let mut state = locked(inner);
            let ImageEncoderState { encoder, out } = &mut *state;
            encoder.encode_into(t, out).map(|()| std::mem::take(out))
        })?
        .map_err(to_pyerr)?;
        let bytes = PyBytes::new(py, &file);
        // The buffer goes back for the next encode to reuse.
        locked(inner).out = file;
        Ok(bytes)
    }

    fn __repr__(&self) -> String {
        let state = locked(&self.inner);
        format!(
            "ImageEncoder(quality={}, engine={})",
            state.encoder.quality(),
            state.encoder.engine()
        )
    }
}

/// One encoded access unit.
#[pyclass(
    name = "EncodedFrame",
    frozen,
    get_all,
    skip_from_py_object,
    module = "edgefirst.codec"
)]
pub struct PyEncodedFrame {
    /// The access unit: Annex B NAL units for H.264, a JPEG file for MJPEG.
    data: Py<PyBytes>,
    /// Whether it is a key frame (an IDR picture for H.264).
    keyframe: bool,
    /// The ``pts`` of the source frame it encodes.
    pts: u64,
}

impl PyEncodedFrame {
    fn new(py: Python<'_>, f: video::EncodedFrame) -> Self {
        Self {
            data: PyBytes::new(py, &f.data).unbind(),
            keyframe: f.keyframe,
            pts: f.pts,
        }
    }
}

#[pymethods]
impl PyEncodedFrame {
    fn __len__(&self, py: Python<'_>) -> usize {
        self.data.bind(py).as_bytes().len()
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!(
            "EncodedFrame(len={}, keyframe={}, pts={})",
            self.data.bind(py).as_bytes().len(),
            self.keyframe,
            self.pts
        )
    }
}

/// What an open encoder runs on and how sources reach it.
#[pyclass(
    name = "EncoderInfo",
    frozen,
    get_all,
    skip_from_py_object,
    module = "edgefirst.codec"
)]
pub struct PyEncoderInfo {
    /// The engine.
    engine: PyEngine,
    /// The device: a device node or a transform's name.
    device: String,
    /// Whether it encodes on dedicated hardware.
    hardware: bool,
    /// How sources reach it, for a source in the memory it prefers.
    transfer: PyTransfer,
}

#[pymethods]
impl PyEncoderInfo {
    fn __repr__(&self) -> String {
        format!(
            "EncoderInfo(engine={:?}, device={:?}, hardware={}, transfer={:?})",
            self.engine, self.device, self.hardware, self.transfer
        )
    }
}

/// Encodes image tensors into an H.264 or MJPEG stream on hardware.
///
/// Platforms:
///     Linux: V4L2 memory-to-memory encoders; DMA-BUF sources are read in
///     place.
///     Windows: Media Foundation; D3D11 sources are read in place.
///     Elsewhere opening fails with ``NoDeviceError``.
#[pyclass(name = "VideoEncoder", module = "edgefirst.codec")]
pub struct PyVideoEncoder {
    inner: Mutex<video::VideoEncoder>,
}

fn locked<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

#[pymethods]
impl PyVideoEncoder {
    /// Opens an encoder.
    ///
    /// Args:
    ///     codec: ``Codec.H264`` or ``Codec.Mjpeg``.
    ///     width: Picture width in pixels.
    ///     height: Picture height in pixels.
    ///     format: The sources' pixel format, for example ``PixelFormat.Nv12``.
    ///     frame_rate: Frames per second, for rate control.
    ///     engine: ``Engine.Auto`` or one engine used alone.
    ///     bitrate: Target bits per second; chosen from size and rate when
    ///         omitted.
    ///     gop: Frames from one key frame to the next; the device's default
    ///         when omitted.
    ///     repeat_headers: Repeat SPS and PPS before every key frame.
    ///     profile: H.264 profile; High when omitted.
    ///     level: H.264 level; the lowest that covers the stream when omitted.
    ///     quality: MJPEG quality, 1 to 100; 85 when omitted.
    ///
    /// Raises:
    ///     NoDeviceError: there is no encoder.
    ///     UnsupportedError: no engine here encodes the codec or format.
    ///     InvalidConfigError: the device refuses the configuration.
    #[new]
    #[pyo3(signature = (
        codec, width, height, format, frame_rate, *,
        engine = PyEngine::Auto, bitrate = None, gop = None, repeat_headers = true,
        profile = None, level = None, quality = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        codec: PyCodec,
        width: u32,
        height: u32,
        format: PyPixelFormat,
        frame_rate: f64,
        engine: PyEngine,
        bitrate: Option<u32>,
        gop: Option<u32>,
        repeat_headers: bool,
        profile: Option<PyH264Profile>,
        level: Option<PyH264Level>,
        quality: Option<i64>,
    ) -> PyResult<Self> {
        let mut config = EncoderConfig::new(codec.into(), width, height, format.into(), frame_rate)
            .with_engine(engine.into());
        config.bitrate = bitrate.map_or(Bitrate::Auto, Bitrate::Bps);
        config.gop = gop;
        config.repeat_headers = repeat_headers;
        config.profile = profile.map(Into::into);
        config.level = level.map(Into::into);
        config.quality = quality.map(jpeg_quality).transpose()?;
        let inner = py
            .detach(|| video::VideoEncoder::new(config))
            .map_err(to_pyerr)?;
        Ok(Self {
            inner: Mutex::new(inner),
        })
    }

    /// Encodes one picture.
    ///
    /// Returns the next access unit, or ``None`` while the encoder holds
    /// pictures back; :meth:`flush` returns the rest at the end.
    ///
    /// Args:
    ///     src: An image tensor from any ``edgefirst`` package, of the
    ///         configured format and size (or larger, with ``crop``).
    ///     pts: A timestamp returned unchanged on the matching access unit.
    ///     keyframe: Encode this picture as a key frame.
    ///     crop: The part of ``src`` to encode, exactly the configured size.
    #[pyo3(signature = (src, pts, *, keyframe = false, crop = None))]
    fn encode(
        &self,
        py: Python<'_>,
        src: &Bound<'_, PyAny>,
        pts: u64,
        keyframe: bool,
        crop: Option<PyRegion>,
    ) -> PyResult<Option<PyEncodedFrame>> {
        let opts = FrameOptions {
            pts,
            crop: crop.map(Into::into),
            keyframe,
        };
        let inner = &self.inner;
        let frame = with_source(py, src, |t| locked(inner).encode(t, &opts))?.map_err(to_pyerr)?;
        Ok(frame.map(|f| PyEncodedFrame::new(py, f)))
    }

    /// Ends the stream and returns the access units still held.
    fn flush(&self, py: Python<'_>) -> PyResult<Vec<PyEncodedFrame>> {
        let inner = &self.inner;
        let frames = py.detach(|| locked(inner).flush()).map_err(to_pyerr)?;
        Ok(frames
            .into_iter()
            .map(|f| PyEncodedFrame::new(py, f))
            .collect())
    }

    /// The engine, device and transfer of this encoder.
    #[getter]
    fn info(&self) -> PyResult<PyEncoderInfo> {
        let info = locked(&self.inner).info();
        Ok(PyEncoderInfo {
            engine: info.engine.try_into()?,
            device: info.device,
            hardware: info.hardware,
            transfer: info.transfer.try_into()?,
        })
    }

    fn __repr__(&self) -> String {
        format!("VideoEncoder(device={:?})", locked(&self.inner).device())
    }
}

/// The stream a decoder produces, as of a ``StreamInfo`` event.
#[pyclass(
    name = "StreamInfo",
    frozen,
    get_all,
    skip_from_py_object,
    module = "edgefirst.codec"
)]
pub struct PyStreamInfo {
    /// Visible width in pixels.
    width: usize,
    /// Visible height in pixels.
    height: usize,
    /// The visible rectangle within the coded picture.
    crop: PyRegion,
    /// Coded (macroblock-aligned) width.
    coded_width: usize,
    /// Coded height.
    coded_height: usize,
    /// The frames' pixel format.
    format: PyPixelFormat,
    /// The stream's colorimetry, where it says.
    colorimetry: Option<PyColorimetry>,
    /// Buffers in the pool.
    buffers: usize,
    /// Whose memory the frames are.
    memory: PyOutputMemory,
    /// The engine decoding.
    engine: PyEngine,
    /// How pictures reach the frames.
    transfer: PyTransfer,
}

impl PyStreamInfo {
    fn new(s: &video::StreamInfo) -> PyResult<Self> {
        Ok(Self {
            width: s.width,
            height: s.height,
            crop: PyRegion {
                x: s.crop.x,
                y: s.crop.y,
                width: s.crop.width,
                height: s.crop.height,
            },
            coded_width: s.coded_width,
            coded_height: s.coded_height,
            format: pixel_format(s.format)?,
            colorimetry: s.colorimetry.map(PyColorimetry::from),
            buffers: s.buffers,
            memory: match s.memory {
                video::OutputMemory::Imported => PyOutputMemory::Imported,
                video::OutputMemory::Exported => PyOutputMemory::Exported,
            },
            engine: s.engine.try_into()?,
            transfer: s.transfer.try_into()?,
        })
    }
}

#[pymethods]
impl PyStreamInfo {
    fn __repr__(&self) -> String {
        format!(
            "StreamInfo(width={}, height={}, format={:?}, buffers={}, memory={:?}, engine={:?}, transfer={:?})",
            self.width,
            self.height,
            self.format,
            self.buffers,
            self.memory,
            self.engine,
            self.transfer
        )
    }
}

/// Frames Python has let go of, given back to the decoder at its next call.
///
/// A frame's buffer must not go back while its tensor's handle is still
/// referenced, or the decoder's CPU writes into that buffer are refused. A
/// frame's own fields are dropped before its base tensor's, so the frame is
/// parked here and released by the decoder while it holds the GIL, which
/// the frame's deallocation also holds.
type Returned = Mutex<Vec<DecodedFrame>>;

/// A decoded picture: a :class:`Tensor` whose buffer belongs to its
/// decoder's pool.
///
/// The buffer goes back to the decoder once this frame is no longer
/// referenced. Hold frames only as long as their pictures are needed: the
/// pool is bounded, and a decoder whose every buffer is held raises
/// ``PoolExhaustedError``. A ``view()`` or ``batch()`` taken from a frame
/// does not keep the buffer.
#[pyclass(name = "Frame", extends = PyTensor, module = "edgefirst.codec")]
pub struct PyFrame {
    frame: Option<DecodedFrame>,
    returned: Weak<Returned>,
    /// Keeps a reconstructed base tensor's host mapping alive.
    _pin: Option<edgefirst_tensor::HostPin<'static>>,
}

impl Drop for PyFrame {
    fn drop(&mut self) {
        let Some(frame) = self.frame.take() else {
            return;
        };
        match self.returned.upgrade() {
            Some(returned) => locked(&returned).push(frame),
            None => drop(frame),
        }
    }
}

impl PyFrame {
    fn into_py(
        py: Python<'_>,
        frame: DecodedFrame,
        returned: &Arc<Returned>,
    ) -> PyResult<Py<PyAny>> {
        // Sharing the handle keeps everything the decoder set on the
        // tensor, a separate chroma plane included. Where tensors are not
        // shared handles, the memory is reconstructed from its descriptor.
        let (base, pin) = match frame.tensor().share() {
            Ok(shared) => (shared, None),
            Err(_) => {
                crate::interop::reconstruct(frame.tensor(), edgefirst_tensor::CpuAccess::Read)?
            }
        };
        let init = PyClassInitializer::from(PyTensor(base)).add_subclass(PyFrame {
            frame: Some(frame),
            returned: Arc::downgrade(returned),
            _pin: pin,
        });
        Ok(Py::new(py, init)?.into_any())
    }

    fn inner(&self) -> &DecodedFrame {
        self.frame
            .as_ref()
            .expect("a frame holds its picture until dropped")
    }
}

#[pymethods]
impl PyFrame {
    /// The ``pts`` passed with the access unit of this picture.
    #[getter]
    fn pts(&self) -> u64 {
        self.inner().pts()
    }

    /// Whether the decoder reported the picture damaged (missing
    /// references, for example after a lost access unit).
    #[getter]
    fn corrupted(&self) -> bool {
        self.inner().corrupted()
    }

    fn __repr__(slf: PyRef<'_, Self>) -> String {
        let t = &slf.as_super().0;
        let dim = |d: Option<usize>| d.map_or_else(|| "None".to_owned(), |d| d.to_string());
        let format = t
            .format()
            .map_or_else(|| "None".to_owned(), |f| format!("{f:?}"));
        format!(
            "Frame(pts={}, width={}, height={}, format={format}, memory={:?})",
            slf.inner().pts(),
            dim(t.width()),
            dim(t.height()),
            t.memory()
        )
    }
}

/// Decodes an H.264 or MJPEG stream on hardware into frames whose pictures
/// are tensors.
///
/// Platforms:
///     Linux: V4L2 memory-to-memory decoders, into DMA-BUF frames.
///     Windows: Media Foundation (Microsoft's H.264 decoder with DXVA, and
///     MJPEG decoders), into D3D11 frames.
///     Elsewhere opening fails with ``NoDeviceError``.
#[pyclass(name = "VideoDecoder", module = "edgefirst.codec")]
pub struct PyVideoDecoder {
    inner: Mutex<video::VideoDecoder>,
    returned: Arc<Returned>,
    /// What keeps caller-provided pool tensors valid; after `inner`, so it
    /// is dropped after the decoder that uses them.
    _pool: Vec<RawKeepalive>,
}

impl PyVideoDecoder {
    /// Gives back the buffers of frames Python has released, then runs
    /// `body` on the decoder with the GIL released and turns its events
    /// into Python objects.
    fn run(
        &self,
        py: Python<'_>,
        body: impl FnOnce(&mut video::VideoDecoder) -> edgefirst_codec::Result<Vec<DecodeEvent>> + Send,
    ) -> PyResult<Vec<Py<PyAny>>> {
        drop(std::mem::take(&mut *locked(&self.returned)));
        let inner = &self.inner;
        let events = py.detach(|| body(&mut locked(inner))).map_err(to_pyerr)?;
        events
            .into_iter()
            .map(|event| match event {
                DecodeEvent::Frame(frame) => PyFrame::into_py(py, frame, &self.returned),
                DecodeEvent::Reconfigured(info) => {
                    Ok(Py::new(py, PyStreamInfo::new(&info)?)?.into_any())
                }
            })
            .collect()
    }
}

#[pymethods]
impl PyVideoDecoder {
    /// Opens a decoder.
    ///
    /// Args:
    ///     codec: ``Codec.H264`` or ``Codec.Mjpeg``.
    ///     engine: ``Engine.Auto`` or one engine used alone.
    ///     pool: The frames the caller may hold at once (4 when omitted), or
    ///         a sequence of tensors to decode into, each a different
    ///         DMA-BUF buffer (an NV12 D3D11 image of the stream's visible
    ///         size on Windows). The decoder writes those tensors: do not
    ///         use them while it is open.
    ///
    /// Raises:
    ///     NoDeviceError: there is no decoder.
    ///     UnsupportedError: no engine here decodes the codec.
    ///     InvalidConfigError: a provided pool that is empty, holds one
    ///         buffer twice or is not in DMA-BUF memory.
    #[new]
    #[pyo3(signature = (codec, *, engine = PyEngine::Auto, pool = None))]
    fn new(
        py: Python<'_>,
        codec: PyCodec,
        engine: PyEngine,
        pool: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut keepalive = Vec::new();
        let output = match pool {
            None => OutputPool::Allocate { count: 4 },
            Some(p) => match p.extract::<usize>() {
                Ok(count) => OutputPool::Allocate { count },
                Err(_) => {
                    let items = p.try_iter()?.collect::<PyResult<Vec<_>>>()?;
                    // The same object twice is refused here, before resolving
                    // it twice; distinct objects over one buffer are refused
                    // by the decoder, which compares buffer identities.
                    for (i, item) in items.iter().enumerate() {
                        if let Some(first) = items[..i].iter().position(|o| o.is(item)) {
                            return Err(InvalidConfigError::new_err(format!(
                                "provided tensors {first} and {i} are the same buffer; each pool buffer must be distinct"
                            )));
                        }
                    }
                    let mut tensors = Vec::new();
                    for item in items {
                        let raw = TensorArg::extract_mut(&item, None)?.into_raw_access()?;
                        let (tensor, keep) = raw.into_owned();
                        tensors.push(tensor);
                        keepalive.push(keep);
                    }
                    OutputPool::Provided(tensors)
                }
            },
        };
        let config = DecoderConfig::new(codec.into())
            .with_engine(engine.into())
            .with_output(output);
        let inner = py
            .detach(|| video::VideoDecoder::new(config))
            .map_err(to_pyerr)?;
        Ok(Self {
            inner: Mutex::new(inner),
            returned: Arc::default(),
            _pool: keepalive,
        })
    }

    /// Decodes one access unit (H.264 in Annex B, or one JPEG picture for
    /// MJPEG) whose picture is tagged with ``pts``.
    ///
    /// Returns, in order, the events ready so far: :class:`StreamInfo` when
    /// the stream's format becomes known or changes, and :class:`Frame` for
    /// each picture, in presentation order. The list may be empty, and
    /// frames may belong to earlier access units.
    ///
    /// Raises:
    ///     PoolExhaustedError: every buffer is held; release frames and
    ///         decode the same access unit again.
    ///     InvalidDataError: data the decoder cannot parse.
    fn decode(&self, py: Python<'_>, data: &[u8], pts: u64) -> PyResult<Vec<Py<PyAny>>> {
        let data = data.to_vec();
        self.run(py, move |d| d.decode(&data, pts))
    }

    /// Ends the stream and returns the remaining events.
    fn flush(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.run(py, |d| d.flush())
    }

    /// The stream as of the last ``StreamInfo`` event, or ``None`` before
    /// the first.
    #[getter]
    fn stream_info(&self) -> PyResult<Option<PyStreamInfo>> {
        locked(&self.inner)
            .stream_info()
            .map(PyStreamInfo::new)
            .transpose()
    }

    /// The engine decoding.
    #[getter]
    fn engine(&self) -> PyResult<PyEngine> {
        locked(&self.inner).engine().try_into()
    }

    /// The device decoding: a device node or a transform's name.
    #[getter]
    fn device(&self) -> String {
        locked(&self.inner).device()
    }

    fn __repr__(&self) -> String {
        format!("VideoDecoder(device={:?})", locked(&self.inner).device())
    }
}

/// Splits an H.264 Annex B byte stream into access units, each ready for
/// :meth:`VideoDecoder.decode`. Bytes before the first start code are
/// skipped.
#[pyfunction]
pub fn access_units<'py>(py: Python<'py>, stream: &[u8]) -> Vec<Bound<'py, PyBytes>> {
    video::access_units(stream)
        .map(|unit| PyBytes::new(py, unit))
        .collect()
}

/// Adds the codec classes, functions and exceptions to `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<PyCodec>()?;
    m.add_class::<PyEngine>()?;
    m.add_class::<PyDirection>()?;
    m.add_class::<PyTransfer>()?;
    m.add_class::<PyH264Profile>()?;
    m.add_class::<PyH264Level>()?;
    m.add_class::<PyOutputMemory>()?;
    m.add_class::<PyCapability>()?;
    m.add_class::<PyImageEncoder>()?;
    m.add_class::<PyEncodedFrame>()?;
    m.add_class::<PyEncoderInfo>()?;
    m.add_class::<PyVideoEncoder>()?;
    m.add_class::<PyStreamInfo>()?;
    m.add_class::<PyFrame>()?;
    m.add_class::<PyVideoDecoder>()?;
    m.add_function(wrap_pyfunction!(capabilities, m)?)?;
    m.add_function(wrap_pyfunction!(access_units, m)?)?;
    m.add("CodecError", py.get_type::<CodecError>())?;
    m.add("NoDeviceError", py.get_type::<NoDeviceError>())?;
    m.add("UnsupportedError", py.get_type::<UnsupportedError>())?;
    m.add("PoolExhaustedError", py.get_type::<PoolExhaustedError>())?;
    m.add("CodecTimeoutError", py.get_type::<CodecTimeoutError>())?;
    m.add("InvalidConfigError", py.get_type::<InvalidConfigError>())?;
    m.add("InvalidInputError", py.get_type::<InvalidInputError>())?;
    m.add("InvalidDataError", py.get_type::<InvalidDataError>())?;
    Ok(())
}
