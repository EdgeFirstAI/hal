"""Type stubs for ``edgefirst.codec``.

Split from the pre-0.29 monolithic ``edgefirst_hal.pyi``; assignment
mirrors the pymodule registrations in crates/python-codec/src/lib.rs.
"""

import enum

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
            RuntimeError: If the tensor is too small or the dtype is unsupported.
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
            RuntimeError: If the tensor is too small or the dtype is unsupported.
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
        RuntimeError: If the tensor is too small or the dtype is unsupported.
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
        RuntimeError: If the tensor is too small or the dtype is unsupported.
        TypeError: If ``tensor`` does not implement the
            ``__edgefirst_tensor__`` protocol.
    """
