"""Type stubs for ``edgefirst.codec``.

Split from the pre-0.29 monolithic ``edgefirst_hal.pyi``; assignment
mirrors the pymodule registrations in crates/python-codec/src/lib.rs.
"""

import enum
from collections.abc import Sequence
from typing import NoReturn

# Re-exported from `edgefirst.tensor` (a hard dependency of this package), so
# `edgefirst.codec.PixelFormat` names the same enum a caller already has. The
# redundant `as` form is what marks it a public re-export to type checkers.
# TensorMemory/Region/Colorimetry and the colour axis enums are registered
# directly in this module too (see crates/python-codec/src/lib.rs), so a
# decode destination's memory backend or colorimetry can be named without
# also depending on edgefirst.tensor.
from edgefirst.tensor import ColorEncoding as ColorEncoding
from edgefirst.tensor import Colorimetry as Colorimetry
from edgefirst.tensor import ColorRange as ColorRange
from edgefirst.tensor import ColorSpace as ColorSpace
from edgefirst.tensor import ColorTransfer as ColorTransfer

# The cross-package capsule protocol -- see crates/python-common/INTEROP.md.
# `decode_into`/`decode_file_into` accept any object implementing it, not
# just this package's own `Tensor`.
from edgefirst.tensor import EdgeFirstTensorExportable as EdgeFirstTensorExportable
from edgefirst.tensor import PixelFormat as PixelFormat
from edgefirst.tensor import Region as Region
from edgefirst.tensor import Tensor as _BaseTensor
from edgefirst.tensor import TensorMemory as TensorMemory

# "dynamic" in every wheel; "static" only in the G13 comparison build.
__tensor_backend__: str

def version() -> str:
    """Return the HAL version string (matches ``Cargo.toml``)."""

def is_v4l2_available() -> bool:
    """True when a V4L2 hardware JPEG decoder is present and not opted out.

    Checks for a device such as the i.MX ``mxc-jpeg`` block, honouring
    ``EDGEFIRST_DISABLE_V4L2``. Opens and drops the device once;
    :meth:`Tensor.decode_image` re-probes lazily and keeps its own context.
    Always ``False`` on platforms without V4L2 hardware decode support.
    Useful for benchmarks and callers that must fail fast instead of
    silently falling back to the CPU decoder.
    """

class DctMethod(enum.Enum):
    """IDCT accuracy/speed selection for the software JPEG decoder.

    :attr:`Accurate` is the default: the ``islow``-class Loeffler IDCT,
    bit-comparable to libjpeg-turbo's default. :attr:`Fast` opts into the
    AAN ``ifast``-class kernel — roughly an eighth of the multiplies, at a
    small, bounded pixel accuracy cost. Fast is advisory: paths without a
    fast kernel (non-NEON tiers, the V4L2/nvJPEG hardware decoders, and
    PNG) use their normal accurate path. Applies to the thread-local decoder
    used by :meth:`Tensor.decode_image` / :meth:`Tensor.decode_image_file`;
    set via :func:`set_dct_method`. Each thread has its own decoder state,
    so this must be set on every thread that decodes images.
    """

    Accurate: DctMethod
    """Accurate ``islow``-class IDCT (default)."""
    Fast: DctMethod
    """Fast AAN ``ifast``-class IDCT (opt-in)."""

def set_dct_method(method: DctMethod) -> None:
    """Select the software JPEG IDCT kernel class for the thread-local
    decoder used by :meth:`Tensor.decode_image` / :meth:`Tensor.decode_image_file`.
    This only affects the calling thread; call it on every thread that
    decodes images if you want a non-default setting everywhere.

    Accurate by default. See :class:`DctMethod` for the accuracy/speed
    tradeoff. ``EDGEFIRST_CODEC_DCT=fast`` in the environment flips a
    **new** thread's default for A/B runs.

    Args:
        method: IDCT kernel class.
    """

def set_output_format(format: PixelFormat | None = None) -> None:
    """Request a fused JPEG decode output format instead of the source's
    native format, for the thread-local decoder used by
    :meth:`Tensor.decode_image` / :meth:`Tensor.decode_image_file`. Only
    affects the calling thread. PNG decodes are unaffected.

    This is a **pure CPU, single-pass** path inside the software JPEG
    decoder — colour conversion / chroma downsample happens at the MCU
    write stage. It is **not** a GPU hybrid or nvJPEG path; V4L2/nvJPEG are
    bypassed whenever the resolved output differs from native.

    - ``PixelFormat.RGB``: 4:4:4 colour JPEGs decode straight to
      interleaved RGB. Other sources fall back to native.
    - ``PixelFormat.NV12``: colour JPEGs decode to NV12, downsampling
      chroma at the write stage (2x2 average for 4:4:4, vertical for
      4:2:2).
    - Any other format is ignored; the decode falls back to native.
    - ``None`` (default): native format (``Nv12``/``Nv16``/``Nv24``/``Grey``).

    Callers still run ``ImageProcessor.convert()`` on the result for
    model-input preprocessing (letterbox, resize, EXIF orientation). With
    fused RGB that convert step is typically a pure resize.

    Args:
        format: Requested fused output format, or ``None`` to restore
            native output.
    """

class ImageInfo:
    """Metadata returned by ``Tensor.decode_image()`` / ``Tensor.decode_image_file()``.

    Describes the actual decoded image dimensions and format, which may
    be smaller than the tensor's allocated capacity.
    """

    width: int
    """Decoded image width in pixels."""
    height: int
    """Decoded image height in pixels."""
    format: PixelFormat
    """Native pixel format of the decoded data."""
    row_stride: int
    """Row stride in bytes used for writing."""
    rotation_degrees: int
    """Clockwise rotation in degrees from EXIF orientation (0/90/180/270).

    The decode itself never rotates; the reported dimensions are unrotated."""
    flip_horizontal: bool
    """Horizontal flip from EXIF orientation. The decode never flips."""

class Tensor(_BaseTensor):
    """A tensor, as seen by the codec module.

    Carries the whole ``edgefirst.tensor.Tensor`` surface — this is the same
    Rust type with the codec feature compiled in — plus the decode entry
    points declared below. Inheriting rather than redeclaring is what keeps
    ``Tensor.decode_image(data).shape()`` type-checking; the previous stub
    declared only the four decode methods and left the other 32 invisible.

    At runtime ``edgefirst.codec.Tensor`` and ``edgefirst.tensor.Tensor`` are
    nonetheless distinct type objects (each extension module caches its own),
    so cross-package handoff goes through the ``__edgefirst_tensor__``
    capsule rather than an isinstance check.
    """

    @staticmethod
    def peek_image_info(data: bytes) -> ImageInfo:
        """Parse the header of JPEG/PNG bytes without decoding pixels.

        Returns the native image dimensions, pixel format, and EXIF
        orientation. Use this to allocate a tensor at the right size before
        calling ``decode_image``. The reported dimensions are unrotated.

        Args:
            data: Raw JPEG or PNG bytes.

        Returns:
            ImageInfo with native width, height, format, row_stride, and the
            EXIF rotation_degrees / flip_horizontal.
        """

    @staticmethod
    def peek_image_info_file(filename: str) -> ImageInfo:
        """Parse the header of an image file without decoding pixels.

        Args:
            filename: Path to the image file.

        Returns:
            ImageInfo with native width, height, format, row_stride, and the
            EXIF rotation_degrees / flip_horizontal.
        """

    def decode_image(self, data: bytes) -> ImageInfo:
        """Decode image bytes (JPEG/PNG) directly into this pre-allocated tensor.

        The image is decoded in its native pixel format (JPEG -> ``Nv12`` for
        colour / ``Grey`` for greyscale; PNG -> ``Rgb`` / ``Rgba`` / ``Grey``)
        and the tensor's dimensions and format are configured by the decoder
        to match. The decode never rotates or flips; if you need RGB, decode
        then call ``ImageProcessor.convert(...)``.

        This is the preferred API for real-time pipelines: allocate a tensor
        once via ``ImageProcessor.create_image()``, then call ``decode_image()``
        in the main loop to avoid per-frame allocations.

        The tensor must have sufficient capacity for the decoded image; it is
        reconfigured (dims + format) within that capacity.

        Call :func:`set_output_format` beforehand to opt a JPEG source into a
        fused ``Rgb``/``Nv12`` output instead, computed in the same decode
        pass.

        Args:
            data: Raw JPEG or PNG bytes.

        Returns:
            ImageInfo with native width, height, format, row_stride, and the
            EXIF rotation_degrees / flip_horizontal.

        Raises:
            InvalidDataError: The data is not a decodable image.
            UnsupportedError: An image feature or destination type the
                decoder does not support.
            CodecError: The tensor is too small for the image.
            PermissionError: If the tensor's declared CPU access does not
                cover the CPU decode's write.
        """

    def decode_image_file(self, filename: str) -> ImageInfo:
        """Decode an image file (JPEG/PNG) directly into this pre-allocated tensor.

        Convenience wrapper around ``decode_image()`` that reads from a file
        path. Decodes in the source's native pixel format and configures the
        tensor's dimensions and format to match.

        Args:
            filename: Path to the image file.

        Returns:
            ImageInfo with native width, height, format, row_stride, and the
            EXIF rotation_degrees / flip_horizontal.

        Raises:
            InvalidDataError: The data is not a decodable image.
            UnsupportedError: An image feature or destination type the
                decoder does not support.
            CodecError: The tensor is too small for the image.
            PermissionError: If the tensor's declared CPU access does not
                cover the CPU decode's write.
        """

def decode_into(tensor: EdgeFirstTensorExportable, data: bytes) -> ImageInfo:
    """Decode image bytes (JPEG/PNG) into any object implementing the
    ``__edgefirst_tensor__`` capsule protocol.

    The cross-package counterpart of :meth:`Tensor.decode_image`, for a
    destination allocated by another ``edgefirst.*`` extension -- e.g. a
    DMA/PBO-backed ``edgefirst.image.Tensor`` from
    ``ImageProcessor.create_image()``. Decode semantics are identical to
    :meth:`Tensor.decode_image`.

    For a foreign ``tensor``, the pixels are always written correctly --
    that part cannot fail. ``tensor``'s own format, dimensions and
    colorimetry are then updated to match the decode on a **best-effort**
    basis: if ``tensor`` does not implement ``configure_image()``, or
    exposes ``colorimetry`` read-only, that update is silently skipped (a
    warning is logged on the Rust side; nothing raises). The returned
    :class:`ImageInfo` always describes the decode accurately regardless, so
    a caller that needs the decoded format/dimensions reliably should read
    them from the return value, not from ``tensor`` itself, when ``tensor``
    may not be a ``Tensor`` from this crate.

    Args:
        tensor: Destination tensor, from this or another ``edgefirst.*``
            package.
        data: Raw JPEG or PNG bytes.

    Returns:
        ImageInfo with native width, height, format, row_stride, and the
        EXIF rotation_degrees / flip_horizontal.

    Raises:
        InvalidDataError: The data is not a decodable image.
        UnsupportedError: An image feature or destination type the decoder
            does not support.
        CodecError: The tensor is too small for the image.
        PermissionError: If the tensor's declared CPU access does not cover
            the CPU decode's write.
        TypeError: If ``tensor`` does not implement the
            ``__edgefirst_tensor__`` protocol.
    """

def decode_file_into(tensor: EdgeFirstTensorExportable, filename: str) -> ImageInfo:
    """Decode an image file (JPEG/PNG) into any object implementing the
    ``__edgefirst_tensor__`` capsule protocol. See :func:`decode_into`.

    Args:
        tensor: Destination tensor, from this or another ``edgefirst.*``
            package.
        filename: Path to the image file.

    Returns:
        ImageInfo with native width, height, format, row_stride, and the
        EXIF rotation_degrees / flip_horizontal.

    Raises:
        InvalidDataError: The data is not a decodable image.
        UnsupportedError: An image feature or destination type the decoder
            does not support.
        CodecError: The tensor is too small for the image.
        PermissionError: If the tensor's declared CPU access does not cover
            the CPU decode's write.
        TypeError: If ``tensor`` does not implement the
            ``__edgefirst_tensor__`` protocol.
    """

# --- Engines and capabilities ----------------------------------------------

class CodecError(RuntimeError):
    """A codec failure. Each kind below is a subclass.

    A refused CPU access is raised as ``PermissionError`` and a missing file
    as ``FileNotFoundError`` instead.
    """

class NoDeviceError(CodecError):
    """No engine or device here for the request."""

class UnsupportedError(CodecError):
    """The engine exists, but not for this codec, format or feature."""

class PoolExhaustedError(CodecError):
    """Every frame of the decoder's pool is held: release frames and decode
    the same access unit again."""

class CodecTimeoutError(CodecError):
    """The device did not finish in time."""

class InvalidConfigError(CodecError):
    """A configuration the engine cannot open, or a provided pool it cannot
    use."""

class InvalidInputError(CodecError):
    """A bad argument or source picture."""

class InvalidDataError(CodecError):
    """Compressed data the decoder cannot parse."""

class Codec(enum.Enum):
    """A compressed image or video format. ``str()`` gives its name, for
    example ``"H.264"``."""

    Jpeg: Codec
    """JPEG still images."""
    Png: Codec
    """PNG still images."""
    H264: Codec
    """H.264, as an Annex B byte stream."""
    Hevc: Codec
    """HEVC. No engine runs it yet."""
    Mjpeg: Codec
    """Motion JPEG: a stream of JPEG pictures."""
    Vp8: Codec
    """VP8. No engine runs it yet."""
    Vp9: Codec
    """VP9. No engine runs it yet."""
    Av1: Codec
    """AV1. No engine runs it yet."""

class Engine(enum.Enum):
    """What runs a codec. ``str()`` gives the name ``EDGEFIRST_CODEC_ENGINE``
    uses, for example ``"media-foundation"``.

    ``Engine.Auto`` takes the first engine that can do the request; the
    environment variable ``EDGEFIRST_CODEC_ENGINE`` (engine names,
    comma-separated) narrows and orders that choice. An engine named
    explicitly is used alone.
    """

    Auto: Engine
    """The first engine that can do the request, in the platform's order."""
    V4l2: Engine
    """Linux V4L2 memory-to-memory devices."""
    MediaFoundation: Engine
    """Windows Media Foundation transforms."""
    VideoToolbox: Engine
    """Apple VideoToolbox. No backend runs it yet."""
    NvJpeg: Engine
    """NVIDIA nvJPEG (JPEG decoding on Linux)."""
    Cpu: Engine
    """The CPU, for still images only."""

    @staticmethod
    def from_name(name: str) -> Engine:
        """The engine a name stands for, as ``EDGEFIRST_CODEC_ENGINE`` spells
        it (``"v4l2"``, ``"mf"``, ``"cpu"``, ...).

        Raises:
            ValueError: An unknown name.
        """

class Direction(enum.Enum):
    """Encoding or decoding."""

    Encode: Direction
    """Images to compressed data."""
    Decode: Direction
    """Compressed data to images."""

class Transfer(enum.Enum):
    """How pictures move between tensors and the engine."""

    ZeroCopy: Transfer
    """The engine reads or writes the tensors' own memory."""
    GpuCopy: Transfer
    """One copy on the GPU, with no CPU access to the pixels."""
    CpuCopy: Transfer
    """The pixels pass through the CPU."""

class Capability:
    """What one engine does with one codec in one direction."""

    engine: Engine
    """The engine."""
    device: str
    """The device within the engine: a device node, a transform's name, or
    ``"cpu"``."""
    codec: Codec
    """The compressed format."""
    direction: Direction
    """Encoding or decoding."""
    hardware: bool
    """Whether the work runs on dedicated hardware rather than the CPU."""
    formats: list[PixelFormat]
    """The pixel formats an encoder takes or a decoder produces."""
    transfer: Transfer
    """How pictures move in the best case."""

def capabilities() -> list[Capability]:
    """What the codec engines on this machine can do, engine by engine, in
    the order ``Engine.Auto`` tries them.

    Probing opens no stream, but it opens device nodes and enumerates (on
    Windows, activates) transforms, so it is not free: call it once.
    """

# --- Still-image encoding --------------------------------------------------

class ImageEncoder:
    """Encodes tensors as still images: JPEG.

    Under ``Engine.Auto`` a V4L2 JPEG encoder takes DMA-BUF sources it can
    read and the CPU takes the rest. The CPU encodes ``uint8`` NV12, NV16,
    YUYV, GREY, RGB and RGBA images at any row stride.
    """

    quality: int
    """JPEG quality, 1 (smallest) to 100 (best). Setting a value outside that
    range raises ``InvalidConfigError``."""
    engine: Engine
    """The engine selected: ``Engine.Auto`` or one engine used alone."""

    def __init__(
        self,
        codec: Codec = ...,
        *,
        quality: int | None = None,
        engine: Engine = ...,
    ) -> None:
        """
        Args:
            codec: The format to write. Only ``Codec.Jpeg`` is encoded.
            quality: JPEG quality, 1 (smallest) to 100 (best); 85 when
                omitted.
            engine: ``Engine.Auto``, ``Engine.Cpu`` or ``Engine.V4l2``.

        Raises:
            UnsupportedError: A codec with no encoder.
            InvalidConfigError: A quality outside 1 to 100.
        """

    def encode(self, src: EdgeFirstTensorExportable) -> bytes:
        """Encodes ``src``, an image tensor from any ``edgefirst`` package,
        and returns the file's bytes.

        Raises:
            PermissionError: The CPU must read a tensor not declared readable.
            UnsupportedError: No engine takes the source's format or type.
            InvalidInputError: A bad source, for example an odd-width YUYV
                image.
            NoDeviceError: The engine selected by name is not present.
        """

# --- Video -------------------------------------------------------------------

class H264Profile(enum.Enum):
    """H.264 profile (ITU-T H.264 Annex A)."""

    Baseline: H264Profile
    ConstrainedBaseline: H264Profile
    Main: H264Profile
    High: H264Profile

class H264Level(enum.Enum):
    """H.264 level (ITU-T H.264 Table A-1): ``L4_1`` is level 4.1."""

    L1_0: H264Level
    L1b: H264Level
    L1_1: H264Level
    L1_2: H264Level
    L1_3: H264Level
    L2_0: H264Level
    L2_1: H264Level
    L2_2: H264Level
    L3_0: H264Level
    L3_1: H264Level
    L3_2: H264Level
    L4_0: H264Level
    L4_1: H264Level
    L4_2: H264Level
    L5_0: H264Level
    L5_1: H264Level
    L5_2: H264Level
    L6_0: H264Level
    L6_1: H264Level
    L6_2: H264Level

class EncodedFrame:
    """One encoded access unit. ``len()`` gives its size in bytes."""

    data: bytes
    """The access unit: Annex B NAL units for H.264, a JPEG file for MJPEG."""
    keyframe: bool
    """Whether it is a key frame (an IDR picture for H.264)."""
    pts: int
    """The ``pts`` of the source picture it encodes."""

    def __len__(self) -> int: ...

class EncoderInfo:
    """What an open encoder runs on and how sources reach it."""

    engine: Engine
    """The engine."""
    device: str
    """The device: a device node or a transform's name."""
    hardware: bool
    """Whether it encodes on dedicated hardware."""
    transfer: Transfer
    """How sources reach it, for a source in the memory it prefers."""

class VideoEncoder:
    """Encodes image tensors into an H.264 or MJPEG stream on hardware.

    There is no software video encoder: where no engine runs the codec,
    opening fails.

    Platforms:
        Linux: V4L2 memory-to-memory encoders; DMA-BUF sources are read in
        place.
        Windows: Media Foundation; D3D11 sources are read in place.
        Elsewhere opening fails with ``NoDeviceError``.
    """

    def __init__(
        self,
        codec: Codec,
        width: int,
        height: int,
        format: PixelFormat,
        frame_rate: float,
        *,
        engine: Engine = ...,
        bitrate: int | None = None,
        gop: int | None = None,
        repeat_headers: bool = True,
        profile: H264Profile | None = None,
        level: H264Level | None = None,
        quality: int | None = None,
    ) -> None:
        """Opens an encoder.

        Args:
            codec: ``Codec.H264`` or ``Codec.Mjpeg``.
            width: Picture width in pixels.
            height: Picture height in pixels.
            format: The sources' pixel format, for example
                ``PixelFormat.Nv12``.
            frame_rate: Frames per second, for rate control.
            engine: ``Engine.Auto`` or one engine used alone.
            bitrate: Target bits per second; chosen from size and rate when
                omitted.
            gop: Frames from one key frame to the next; the device's default
                when omitted.
            repeat_headers: Repeat SPS and PPS before every key frame.
            profile: H.264 profile; High when omitted.
            level: H.264 level; the lowest that covers the stream when
                omitted.
            quality: MJPEG quality, 1 to 100; 85 when omitted.

        Raises:
            NoDeviceError: There is no encoder.
            UnsupportedError: No engine here encodes the codec or format.
            InvalidConfigError: The device refuses the configuration.
        """

    def encode(
        self,
        src: EdgeFirstTensorExportable,
        pts: int,
        *,
        keyframe: bool = False,
        crop: Region | None = None,
    ) -> EncodedFrame | None:
        """Encodes one picture.

        Returns the next access unit, or ``None`` while the encoder holds
        pictures back; :meth:`flush` returns the rest at the end.

        Args:
            src: An image tensor from any ``edgefirst`` package, of the
                configured format and size (or larger, with ``crop``).
            pts: A timestamp returned unchanged on the matching access unit.
            keyframe: Encode this picture as a key frame.
            crop: The part of ``src`` to encode, exactly the configured size.

        Raises:
            InvalidInputError: A source of the wrong format or size.
            PermissionError: The engine must read the source on the CPU and
                it is not declared readable.
        """

    def flush(self) -> list[EncodedFrame]:
        """Ends the stream and returns the access units still held."""

    @property
    def info(self) -> EncoderInfo:
        """The engine, device and transfer of this encoder."""

class OutputMemory(enum.Enum):
    """Whose memory a decoder writes its pictures to."""

    Imported: OutputMemory
    """The pool's buffers (allocated by the decoder or provided by the
    caller), which the device imports or the decoder copies into."""
    Exported: OutputMemory
    """The device's own buffers, exported because it would not import the
    pool's. CPU reads of these may be uncached and slow."""

class StreamInfo:
    """The stream a decoder produces. :meth:`VideoDecoder.decode` returns one
    when the stream's format becomes known or changes."""

    width: int
    """Visible width in pixels."""
    height: int
    """Visible height in pixels."""
    crop: Region
    """The visible rectangle within the coded picture."""
    coded_width: int
    """Coded (macroblock-aligned) width."""
    coded_height: int
    """Coded height."""
    format: PixelFormat
    """The frames' pixel format."""
    colorimetry: Colorimetry | None
    """The stream's colorimetry, where it says."""
    buffers: int
    """Buffers in the pool."""
    memory: OutputMemory
    """Whose memory the frames are."""
    engine: Engine
    """The engine decoding."""
    transfer: Transfer
    """How pictures reach the frames."""

class Frame(Tensor):
    """A decoded picture: a :class:`Tensor` whose buffer belongs to its
    decoder's pool.

    The buffer goes back to the decoder once the frame is no longer
    referenced. Hold frames only as long as their pictures are needed: the
    pool is bounded, and a decoder whose every buffer is held raises
    ``PoolExhaustedError``. A map or pin of a frame keeps the buffer until it
    is released; a ``view()`` or ``batch()`` does not.

    A frame is read-only: its buffer and metadata belong to the decoder.
    Changing its shape, format, quantization or colorimetry, or decoding or
    writing into it, raises ``TypeError``, and a writable CPU map raises
    ``RuntimeError``. Copy the picture to change it.
    """

    @property
    def pts(self) -> int:
        """The ``pts`` passed with the access unit of this picture."""

    @property
    def corrupted(self) -> bool:
        """Whether the decoder reported the picture damaged (missing
        references, for example after a lost access unit)."""

    @property
    def colorimetry(self) -> Colorimetry | None:  # type: ignore[override]
        """The stream's colorimetry for this picture. Read-only on a frame."""

    # The changes a Tensor allows and a Frame refuses, each raising TypeError.
    def reshape(self, *args: object, **kwargs: object) -> NoReturn:
        """Raises ``TypeError``: a frame is read-only."""
    def set_format(self, *args: object, **kwargs: object) -> NoReturn:
        """Raises ``TypeError``: a frame is read-only."""
    def configure_image(self, *args: object, **kwargs: object) -> NoReturn:
        """Raises ``TypeError``: a frame is read-only."""
    def from_numpy(self, *args: object, **kwargs: object) -> NoReturn:
        """Raises ``TypeError``: a frame is read-only."""
    def decode_image(self, *args: object, **kwargs: object) -> NoReturn:
        """Raises ``TypeError``: a frame is read-only."""
    def decode_image_file(self, *args: object, **kwargs: object) -> NoReturn:
        """Raises ``TypeError``: a frame is read-only."""
    def set_quantization_per_tensor(self, *args: object, **kwargs: object) -> NoReturn:
        """Raises ``TypeError``: a frame is read-only."""
    def set_quantization_per_tensor_symmetric(
        self, *args: object, **kwargs: object
    ) -> NoReturn:
        """Raises ``TypeError``: a frame is read-only."""
    def set_quantization_per_channel(self, *args: object, **kwargs: object) -> NoReturn:
        """Raises ``TypeError``: a frame is read-only."""
    def set_quantization_per_channel_symmetric(
        self, *args: object, **kwargs: object
    ) -> NoReturn:
        """Raises ``TypeError``: a frame is read-only."""
    def clear_quantization(self) -> NoReturn:
        """Raises ``TypeError``: a frame is read-only."""

class VideoDecoder:
    """Decodes an H.264 or MJPEG stream on hardware into frames whose
    pictures are tensors.

    There is no software video decoder: where no engine runs the codec,
    opening fails.

    Platforms:
        Linux: V4L2 memory-to-memory decoders, into DMA-BUF frames.
        Windows: Media Foundation (Microsoft's H.264 decoder with DXVA, and
        MJPEG decoders), into D3D11 frames.
        Elsewhere opening fails with ``NoDeviceError``.
    """

    def __init__(
        self,
        codec: Codec,
        *,
        engine: Engine = ...,
        pool: int | Sequence[EdgeFirstTensorExportable] | None = None,
    ) -> None:
        """Opens a decoder.

        Args:
            codec: ``Codec.H264`` or ``Codec.Mjpeg``.
            engine: ``Engine.Auto`` or one engine used alone.
            pool: The frames the caller may hold at once (4 when omitted), or
                a sequence of tensors to decode into, each a different
                DMA-BUF buffer (an NV12 D3D11 image of the stream's visible
                size on Windows). The decoder writes those tensors: do not
                use them while it is open.
                A V4L2 device takes a bounded number of buffers (31 on the
                i.MX 95's Wave6, its own included); a pool beyond that
                raises ``InvalidConfigError`` from :meth:`decode` once the
                stream's format is known.

        Raises:
            NoDeviceError: There is no decoder.
            UnsupportedError: No engine here decodes the codec.
            InvalidConfigError: A provided pool that is empty, holds one
                buffer twice or is not in DMA-BUF memory.
        """

    def decode(self, data: bytes, pts: int) -> list[Frame | StreamInfo]:
        """Decodes one access unit (H.264 in Annex B, or one JPEG picture for
        MJPEG) whose picture is tagged with ``pts``.

        Returns, in order, the events ready so far: :class:`StreamInfo` when
        the stream's format becomes known or changes, and :class:`Frame` for
        each picture, in presentation order. The list may be empty, and
        frames may belong to earlier access units.

        Raises:
            PoolExhaustedError: Every buffer is held; release frames and
                decode the same access unit again.
            InvalidDataError: Data the decoder cannot parse.
        """

    def flush(self) -> list[Frame | StreamInfo]:
        """Ends the stream and returns the remaining events."""

    @property
    def stream_info(self) -> StreamInfo | None:
        """The stream as of the last :class:`StreamInfo` event, or ``None``
        before the first."""

    @property
    def engine(self) -> Engine:
        """The engine decoding."""

    @property
    def device(self) -> str:
        """The device decoding: a device node or a transform's name."""

def access_units(stream: bytes) -> list[bytes]:
    """Splits an H.264 Annex B byte stream into access units, each ready for
    :meth:`VideoDecoder.decode`. Bytes before the first start code are
    skipped."""
