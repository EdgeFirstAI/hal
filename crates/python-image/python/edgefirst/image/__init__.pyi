"""Type stubs for ``edgefirst.image``."""

import enum
import sys

# Public re-exports from `edgefirst.tensor`. TensorMemory, Colorimetry and the
# colour axis enums are also registered in this module.
from typing import Protocol, Sequence

import numpy as np
import numpy.typing as npt
from edgefirst.tensor import ColorEncoding as ColorEncoding
from edgefirst.tensor import Colorimetry as Colorimetry
from edgefirst.tensor import ColorRange as ColorRange
from edgefirst.tensor import ColorSpace as ColorSpace
from edgefirst.tensor import ColorTransfer as ColorTransfer

# DType is a type alias.
from edgefirst.tensor import DType

# The cross-package capsule protocol accepted by the convert and drawing
# methods.
from edgefirst.tensor import EdgeFirstTensorExportable as EdgeFirstTensorExportable
from edgefirst.tensor import PixelFormat as PixelFormat
from edgefirst.tensor import Region as Region
from edgefirst.tensor import Tensor as Tensor
from edgefirst.tensor import TensorMemory as TensorMemory
from typing_extensions import TypeAlias

class EdgeFirstProtoDataExportable(Protocol):
    """Structural type for mask-prototype data via ``__edgefirst_protodata__``."""

    def __edgefirst_protodata__(self) -> tuple[object, object, str]: ...

# "dynamic" in every wheel; "static" only in the comparison build.
__tensor_backend__: str

def version() -> str:
    """Return the HAL version string (matches ``Cargo.toml``)."""

class ColorMode(enum.Enum):
    """Controls how mask colors are assigned to detections."""

    Class: ColorMode
    """Color chosen by class label (default)"""
    Instance: ColorMode
    """Color chosen by detection index (unique color per instance)"""
    Track: ColorMode
    """Color chosen by track ID (use with object tracking)"""

class MaskResolution:
    """Controls the resolution and coordinate frame of masks produced by
    :meth:`ImageProcessor.materialize_masks`.

    Construct via classmethods:

    - ``MaskResolution.Proto()`` — per-detection tiles at proto-plane
      resolution (default). Mask values are continuous sigmoid
      ``uint8 [0, 255]``.
    - ``MaskResolution.Scaled(width, height)`` — per-detection tiles at
      caller-specified pixel resolution, produced by upsampling the full
      proto plane once (edge-clamp bilinear) and cropping by bbox
      after sigmoid. Mask values are binary ``uint8 {0, 255}`` —
      interchangeable with the continuous sigmoid output via the same
      ``> 127`` test. If a ``content_region`` is also passed to
      ``materialize_masks``, ``(width, height)`` are interpreted as
      original-content pixel dims and the inverse letterbox transform is
      applied during the upsample.
    """

    @classmethod
    def Proto(cls) -> MaskResolution:
        """Per-detection tile at proto-plane resolution (default)."""

    @classmethod
    def Scaled(cls, width: int, height: int) -> MaskResolution:
        """Per-detection tile at ``(width, height)`` pixel resolution."""

class Flip(enum.Enum):
    """A mirror applied during a convert.

    The integer values are the C ``EF_FLIP_*`` codes: ``NoFlip`` is 0,
    ``Vertical`` 1 and ``Horizontal`` 2.
    """

    NoFlip: Flip
    """No flip"""
    Vertical: Flip
    """Flip the image vertically (top to bottom)"""
    Horizontal: Flip
    """Flip the image horizontally (left to right)"""

class Rotation(enum.Enum):
    Rotate0: Rotation
    """No rotation"""

    Clockwise90: Rotation
    """Rotate the image 90 degrees clockwise"""

    Rotate180: Rotation
    """Rotate the image 180 degrees"""

    CounterClockwise90: Rotation
    """Rotate the image 90 degrees counter-clockwise"""

    @staticmethod
    def degrees_clockwise(angle: int) -> Rotation:
        """Get the Rotation enum variant corresponding to the specified angle in degrees clockwise. Valid angles are 0, 90, 180, and 270."""

class EglDisplayKind(enum.Enum):
    """Identifies the type of EGL display used for headless OpenGL ES rendering.

    The HAL creates a surfaceless GLES 3.0 context and renders exclusively
    through FBOs. No window or PBuffer surface is created.

    Display types (probed in priority order):
        PlatformDevice: EGL device enumeration via EGL_EXT_device_enumeration.
            Headless, with no compositor.
        Gbm: Direct GPU access via DRM render node (/dev/dri/renderD128).
            No compositor required.
        Default: Uses eglGetDisplay(EGL_DEFAULT_DISPLAY). Connects to
            Wayland compositor or X server if available. May block on
            headless systems.
    """

    Gbm: EglDisplayKind
    PlatformDevice: EglDisplayKind
    Default: EglDisplayKind

if sys.platform == "linux":
    class EglDisplayInfo:
        """A validated, available EGL display discovered by probe_egl_displays()."""

        @property
        def kind(self) -> EglDisplayKind:
            """The type of EGL display."""

        @property
        def description(self) -> str:
            """Human-readable description for logging/diagnostics."""

    def probe_egl_displays() -> list[EglDisplayInfo]:
        """Probe for available EGL displays supporting headless OpenGL ES 3.0.

        Linux only. Returns displays in priority order (PlatformDevice, GBM,
        Default). Each display is validated with eglInitialize and checked
        for the required extensions (EGL_KHR_surfaceless_context,
        EGL_KHR_no_config_context). An empty list means OpenGL is not
        available on this system.

        Raises:
            RuntimeError: If libEGL.so.1 cannot be loaded.
        """

def align_width_for_gpu_pitch(width: int, bpp: int) -> int:
    """Round ``width`` up so that ``width * bpp`` satisfies the GPU DMA-BUF
    pitch alignment requirement (64 bytes).

    Use when allocating a DMA-BUF that will be imported as an EGLImage by
    the GL backend.

    Widths that are already aligned are returned unchanged.
    Returns ``width`` unchanged if ``bpp == 0``, ``width == 0``, or if the
    rounded value would overflow.

    Args:
        width: Image width in pixels.
        bpp: Bytes per pixel for the primary plane (4 for RGBA8/BGRA8,
            3 for RGB888, 1 for Grey/NV12-luma).

    Returns:
        Aligned width in pixels (always ``>= width``).
    """

def align_width_for_pixel_format(
    width: int,
    format: PixelFormat,
    dtype: DType = "uint8",
) -> int:
    """Convenience wrapper that derives bytes-per-pixel from a pixel format
    and dtype, then calls :func:`align_width_for_gpu_pitch`.

    Args:
        width: Image width in pixels.
        format: Pixel format (e.g. ``PixelFormat.Rgba``).
        dtype: Element data type as a string — same set accepted by
            :meth:`ImageProcessor.create_image` (``"uint8"``, ``"int8"``,
            ``"uint16"``, ``"float16"``, ``"float32"``, …).

    Returns:
        Aligned width in pixels (always ``>= width``).
    """

def gpu_dma_buf_pitch_alignment_bytes() -> int:
    """Required DMA-BUF row pitch alignment in bytes for GL backend imports
    (64). Each row pitch of an externally allocated DMA-BUF must be a
    multiple of this value.
    """

class LetterboxPlacement(enum.Enum):
    """Where a letterboxed image sits. The integer values are the C
    ``EF_LETTERBOX_PLACEMENT_*`` codes."""

    Center: LetterboxPlacement
    """Centred, padding split between both sides (the default)."""
    TopLeft: LetterboxPlacement
    """Anchored at the top-left corner, padding on the right and bottom."""

class Fit(enum.Enum):
    """How a source is fit into the destination. The integer values are the
    C ``EF_FIT_*`` codes."""

    Stretch: Fit
    """Scale each axis to fill the destination."""

    Letterbox: Fit
    """Preserve aspect ratio and pad the rest."""

class NormalizationPreset(enum.Enum):
    """A named convert normalization, ``(x - mean) / std`` on 0-255 values.
    The integer values are the C ``EF_NORMALIZATION_*`` codes; the names match
    the ``edgefirst.json`` presets."""

    Unsigned: NormalizationPreset
    """``x / 255``, giving ``[0, 1]`` (the default)."""
    Signed: NormalizationPreset
    """``x / 127.5 - 1``, giving ``[-1, 1]``."""
    Raw: NormalizationPreset
    """``x``, giving ``[0, 255]``."""
    Imagenet: NormalizationPreset
    """ImageNet mean ``(123.675, 116.28, 103.53)`` and std ``(58.395, 57.12, 57.375)``."""

NormalizationArg: TypeAlias = (
    NormalizationPreset | str | int | tuple[Sequence[float], Sequence[float]]
)
"""A :class:`NormalizationPreset`, its name or integer code, or
``(mean, std)``."""

PlacementArg: TypeAlias = LetterboxPlacement | str | int
"""A :class:`LetterboxPlacement`, its name (``"center"``, ``"topleft"``) or
its integer code."""

class Interpolation(enum.Enum):
    """Resampling filter for resizing converts. The integer values are the C
    ``EF_INTERPOLATION_*`` codes; the names (``"bilinear"``, ``"nearest"``,
    ``"bicubic"``, ``"area"``) are accepted wherever an interpolation is."""

    Bilinear: Interpolation
    """Bilinear with half-pixel centres (the default)."""
    Nearest: Interpolation
    """Nearest neighbour."""
    Bicubic: Interpolation
    """Bicubic; not implemented, runs as bilinear with a warning."""
    Area: Interpolation
    """Pixel-area averaging; not implemented, runs as bilinear with a
    warning."""

InterpolationArg: TypeAlias = Interpolation | str | int
"""An :class:`Interpolation`, its name (``"bilinear"``, ``"nearest"``,
``"bicubic"``, ``"area"``) or its integer code."""

class PadSpace(enum.Enum):
    """The space a letterbox pad value is expressed in. The integer values
    are the C ``EF_PAD_SPACE_*`` codes."""

    Pixel: PadSpace
    """A 0-255 pixel value, normalized like the image (the default)."""
    Normalized: PadSpace
    """A value written as-is into the normalized destination."""

class PreprocessSpecLike(Protocol):
    """Anything with the attributes of ``edgefirst.decoder.PreprocessSpec``,
    as read by :meth:`ConvertOptions.from_spec`. Every attribute is required:
    ``normalization`` (or, when it is missing or ``None``, ``mean`` and
    ``std``), ``fit``, ``placement``, ``pad``, ``pad_space``,
    ``interpolation`` and ``size`` (``None`` when unknown). Enum-valued
    attributes take a name, an enum member or an integer code."""

    @property
    def normalization(self) -> NormalizationArg | None: ...
    @property
    def fit(self) -> Fit | str | int: ...
    @property
    def placement(self) -> PlacementArg: ...
    @property
    def pad(self) -> Sequence[float]: ...
    @property
    def pad_space(self) -> PadSpace | str | int: ...
    @property
    def interpolation(self) -> InterpolationArg: ...
    @property
    def size(self) -> tuple[int, int] | None: ...

class ContentRegion:
    """Where the image content sits in a convert destination, in pixels and
    normalized by the destination size. Returned by
    :meth:`ConvertOptions.content_region`; pass it as ``content_region=`` to
    the drawing methods or ``edgefirst.decoder.Decoder.decode_semantic``."""

    @property
    def x(self) -> int:
        """Left edge in destination pixels."""
    @property
    def y(self) -> int:
        """Top edge in destination pixels."""
    @property
    def width(self) -> int:
        """Width in destination pixels."""
    @property
    def height(self) -> int:
        """Height in destination pixels."""
    @property
    def pixels(self) -> Region:
        """The rectangle in destination pixels."""
    @property
    def normalized(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` normalized by the destination size."""
    @property
    def is_full(self) -> bool:
        """Whether the content fills the destination (a stretch)."""

ContentRegionArg: TypeAlias = (
    ContentRegion | tuple[float, float, float, float] | list[float]
)
"""A ``content_region=`` argument: a :class:`ContentRegion`, any object with
a ``normalized`` attribute, or normalized ``(xmin, ymin, xmax, ymax)``."""

class ConvertOptions:
    """Options for :meth:`ImageProcessor.convert`. Every method returns a new
    object, so one can be built once and reused.

    Example::

        opts = (ConvertOptions()
                .letterbox((0, 0, 0, 255))
                .placement(LetterboxPlacement.TopLeft)
                .pad_output((0.0, 0.0, 0.0, 0.0))
                .normalization(NormalizationPreset.Imagenet))
        proc.convert(src, dst, options=opts)
    """

    def __init__(
        self,
        rotation: Rotation | None = None,
        flip: Flip | None = None,
        source: Region | None = None,
        letterbox: tuple[int, int, int, int] | None = None,
        placement: PlacementArg | None = None,
        pad_output: tuple[float, float, float, float] | None = None,
        normalization: NormalizationArg | None = None,
        interpolation: InterpolationArg | None = None,
    ) -> None:
        """``interpolation`` resamples this convert when it resizes, instead
        of the processor's setting (see :meth:`interpolation`)."""
    def rotation(self, rotation: Rotation) -> ConvertOptions:
        """Rotate the output, in 90° steps."""
    def flip(self, flip: Flip) -> ConvertOptions:
        """Flip the output."""
    def source(self, source: Region | None) -> ConvertOptions:
        """Sample only ``source`` of the input; ``None`` samples all of it."""
    def letterbox(self, pad: tuple[int, int, int, int]) -> ConvertOptions:
        """Letterbox, padding with the RGBA pixel ``pad``, mapped like the image."""
    def placement(self, placement: PlacementArg) -> ConvertOptions:
        """Where a letterboxed image sits."""
    def pad_output(self, pad: tuple[float, float, float, float]) -> ConvertOptions:
        """Pad with these values as written to the destination."""
    def normalization(self, normalization: NormalizationArg) -> ConvertOptions:
        """How pixel values map to the destination."""
    def interpolation(self, interpolation: InterpolationArg) -> ConvertOptions:
        """Resample with this interpolation when the convert resizes, instead
        of the processor's (:meth:`ImageProcessor.set_interpolation`).
        Bicubic and area run as bilinear, with a warning logged once per
        mode. ``EDGEFIRST_INTERPOLATION``, when set, takes precedence."""
    @staticmethod
    def from_spec(spec: PreprocessSpecLike) -> ConvertOptions:
        """Options that apply a model's preprocessing: normalization, fit,
        letterbox placement, pad and interpolation. ``spec`` is an
        ``edgefirst.decoder.PreprocessSpec`` (e.g. from
        ``ResolvedModelConfig.preprocessing``) or any object with all of
        its attributes (see :class:`PreprocessSpecLike`).

        Raises:
            TypeError: ``spec`` lacks one of the attributes (e.g. ``None`` or
                a ``dict``), or one has the wrong type.
            ValueError: A pixel-space pad outside 0-255, a non-finite pad, an
                unknown name, or a size that is not positive.
        """
    def with_spec(self, spec: PreprocessSpecLike) -> ConvertOptions:
        """These options with a model's preprocessing applied (see
        :meth:`from_spec`); rotation, flip and the source crop are kept."""
    def content_region(
        self, src_w: int, src_h: int, dst_w: int, dst_h: int
    ) -> ContentRegion:
        """Where ``convert`` puts the image content when converting a
        ``src_w`` x ``src_h`` source into a ``dst_w`` x ``dst_h``
        destination with these options: the letterbox rectangle, or the
        whole destination for a stretch.

        Raises:
            ValueError: A zero size, or the source crop falls outside the
                source.
        """

class ImageProcessor:
    """Convert images between different formats, with optional rotation, flipping, and cropping."""

    def __init__(
        self,
        egl_display: EglDisplayKind | None = None,
        interpolation: InterpolationArg | None = None,
    ) -> None:
        """Create an ImageProcessor with optional EGL display override.

        Args:
            egl_display: Force OpenGL to use this display type instead of
                auto-detecting. Use probe_egl_displays() to discover
                available displays. Ignored if EDGEFIRST_DISABLE_GL=1.
            interpolation: How convert() resamples when it resizes, for
                converts whose options do not choose their own:
                "bilinear" (the default), "nearest", "bicubic" or "area"
                (the last two run as bilinear with a warning). The
                EDGEFIRST_INTERPOLATION environment variable overrides it.
        """

    def draw_decoded_masks(
        self,
        dst: EdgeFirstTensorExportable,
        bbox: npt.NDArray[np.float32],
        scores: npt.NDArray[np.float32],
        classes: npt.NDArray[np.uintp],
        seg: list[npt.NDArray[np.uint8]] = [],
        background: EdgeFirstTensorExportable | None = None,
        opacity: float = 1.0,
        content_region: ContentRegionArg | None = None,
        color_mode: ColorMode = ColorMode.Class,
    ) -> None:
        """
        Draw detection boxes and optional segmentation masks onto ``dst``.

        This method draws pre-decoded results; ``Decoder.draw_onto`` decodes
        and draws.

        ``dst`` is always fully overwritten. If ``background`` is provided the
        output is ``background + masks``; otherwise ``dst`` is cleared to
        transparent (``0x00000000``) before masks are drawn.

        On Windows the destination's ``gpu_completion()`` reflects this draw
        afterwards.

        A ``BGRA`` ``background=`` onto a GPU-backed destination raises.

        Args:
            dst: Output image tensor (always fully written by this call). Must
                be ``RGBA`` or ``RGB`` for the CPU backend, or
                ``RGBA``/``BGRA``/``RGB`` for OpenGL (``BGRA`` only without a
                ``background=``).
            bbox: ``(N, 4)`` float32 array of normalized bounding boxes in
                ``[x1, y1, x2, y2]`` format with values in ``[0, 1]``.
            scores: ``(N,)`` float32 array of confidence scores.
            classes: ``(N,)`` uintp array of class indices.
            seg: Optional list of ``uint8`` mask arrays from ``decode()``.
                Each element has shape ``(H, W, C)``. When ``C=1`` (instance
                segmentation, e.g. YOLO), one mask per detection is expected
                (``len(seg) <= len(bbox)``). When ``C > 1`` (semantic
                segmentation, e.g. ModelPack), a single mask covers all classes.
                When empty and ``background`` is ``None``, ``dst`` is cleared to
                transparent.
            background: Optional compositing source. When provided, ``dst`` is
                written as ``background + masks``. When ``None``, ``dst`` is
                cleared to ``0x00000000`` before masks are drawn. Must have the
                same dimensions and format as ``dst``, and must reference a
                distinct underlying buffer; aliasing ``dst`` (directly or
                via two ``Tensor`` wrappers over the same dmabuf fd) raises
                ``RuntimeError``.
            opacity: Scales the alpha of rendered mask and box colors.
                ``1.0`` (default) preserves the class color's alpha unchanged;
                ``0.5`` makes overlays semi-transparent. Clamped to [0, 1].
            content_region: Where the image content sits in the model input
                (:meth:`ConvertOptions.content_region`, or normalized
                ``(x0, y0, x1, y1)``), for mapping model-space boxes back to
                image space. ``None`` (default) means the image fills the
                input.
            color_mode: How to assign colors to detections.
                ``ColorMode.Class`` (default) colors by class label,
                ``ColorMode.Instance`` colors by detection index.

        Raises:
            RuntimeError: If ``dst`` format is unsupported by the active
                backend, if ``background`` aliases ``dst``, if ``bbox``,
                ``scores``, or ``classes`` lengths do not match, or if
                ``bbox`` shape is not ``(N, 4)``.
            PermissionError: If a CPU draw needs access the tensors'
                declarations do not cover (``dst`` ``"readwrite"``,
                ``background`` readable).
        """

    def materialize_masks(
        self,
        bbox: npt.NDArray[np.float32],
        scores: npt.NDArray[np.float32],
        classes: npt.NDArray[np.uintp],
        proto_data: EdgeFirstProtoDataExportable,
        content_region: ContentRegionArg | None = None,
        resolution: MaskResolution | None = None,
    ) -> list[npt.NDArray[np.uint8]]:
        """Materialize per-instance segmentation masks from prototype data.

        Computes ``mask_coeff @ protos`` with sigmoid activation for each
        detection. The ``resolution`` parameter selects the output shape
        and value encoding:

        - ``MaskResolution.Proto()`` (default when ``resolution`` is
          ``None``): returns per-detection tiles at proto-plane resolution
          with continuous sigmoid values in ``uint8 [0, 255]``.
        - ``MaskResolution.Scaled(width, height)``: returns per-detection
          tiles at ``(width, height)`` pixel resolution with binary
          ``uint8 {0, 255}`` values (sigmoid > 0.5 threshold). The upsample
          happens on the full proto plane with edge-clamp bilinear
          sampling.

        The returned masks can be inspected for analytics, IoU computation,
        or export, and also passed to :meth:`draw_decoded_masks` for
        GPU-interpolated rendering.

        Args:
            bbox: ``(N, 4)`` float32 array of normalized bounding boxes.
            scores: ``(N,)`` float32 array of confidence scores.
            classes: ``(N,)`` uintp array of class indices.
            proto_data: Prototype data from :meth:`Decoder.decode_proto`,
                from this or another ``edgefirst.*`` package.
            content_region: Where the image content sits in the model input
                (:meth:`ConvertOptions.content_region`, or normalized
                ``(x0, y0, x1, y1)``). When set with
                ``MaskResolution.Scaled``, ``(width, height)`` are
                interpreted as original-content pixel dims and the inverse
                letterbox transform is applied during the upsample.
            resolution: Output resolution and encoding. Defaults to
                :meth:`MaskResolution.Proto` when ``None``.

        Returns:
            List of ``(H, W, 1)`` uint8 arrays. Shape and encoding depend
            on ``resolution`` — see above.
        """

    def draw_proto_masks(
        self,
        dst: EdgeFirstTensorExportable,
        bbox: npt.NDArray[np.float32],
        scores: npt.NDArray[np.float32],
        classes: npt.NDArray[np.uintp],
        proto_data: EdgeFirstProtoDataExportable,
        background: EdgeFirstTensorExportable | None = None,
        opacity: float = 1.0,
        content_region: ContentRegionArg | None = None,
        color_mode: ColorMode = ColorMode.Class,
    ) -> None:
        """Draw prototype masks onto ``dst`` without materialising per-instance
        mask arrays in Python. ``proto_data`` may come from another
        ``edgefirst.*`` package via the ``__edgefirst_protodata__`` capsule.

        On Windows the destination's ``gpu_completion()`` reflects this draw
        afterwards. The same ``BGRA`` ``background=`` restriction as
        :meth:`draw_decoded_masks` applies.

        Raises:
            PermissionError: As :meth:`draw_decoded_masks`.
        """

    def create_image(
        self,
        width: int,
        height: int,
        format: PixelFormat = PixelFormat.Rgba,
        dtype: DType = "uint8",
        access: str = "readwrite",
        compression: str | None = None,
        contiguous: bool = False,
    ) -> Tensor:
        """Create an image tensor with the processor's optimal memory backend.

        Selects the backing storage from the hardware capabilities, in the
        order DMA-buf, PBO (GPU buffer), system memory.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            format: Pixel format (default: ``PixelFormat.Rgba``).
            dtype: Element data type string (default: ``"uint8"``).
                Supported values: ``"uint8"``, ``"int8"``, ``"uint16"``,
                ``"int16"``, ``"uint32"``, ``"int32"``, ``"uint64"``,
                ``"int64"``, ``"float16"``, ``"float32"``, ``"float64"``.
                PBO backing covers ``"uint8"``/``"int8"`` and, on GL backends
                advertising float-texture support, ``"float16"`` and
                ``"float32"`` (for GPU float model-input paths); other types
                still prefer DMA-buf when available and otherwise use system
                memory.
            access: Declared CPU access: ``"readwrite"`` (default),
                ``"read"``, ``"write"`` (whole-buffer writes only), or
                ``"none"`` (hardware-only, required for tile compression).
                Hardware (GPU/NPU/G2D/compositor) access is always implied.
                A CPU map outside the declaration raises ``PermissionError``,
                as does HAL CPU work (a CPU convert or draw) it does not cover.
                On Linux a ``"none"`` or ``"write"`` DMA-BUF image comes from
                the uncached CMA heap where available. What this processor's
                own ``convert()`` may do to the image as a destination on the
                CPU is added, so ``Tensor.declared_cpu_access`` may be wider
                than ``access``: ``"write"`` when it may fill the whole image
                (CPU backend, host memory, a format or float dtype the GPU
                does not render, or an ``Nv12`` image), ``"readwrite"`` when a pass reads back what
                it wrote or writes only part of it (an ``"int8"`` image through
                the CPU or G2D, a ``Bgra`` image through the CPU, a G2D ``Rgb``
                letterbox border). Nothing is added for use as a convert
                source (declare ``"read"``) or for a CPU mask draw (declare
                ``"readwrite"``). On Android a ``"none"`` image lets gralloc
                keep a tiled or compressed allocation. When the GPU declines
                the source of a convert into a ``"none"`` image, the CPU
                fallback raises ``PermissionError`` naming the GPU's reason.
            compression: Tile-compression request: ``None`` (default,
                linear), ``"any"`` (device's native scheme when eligible,
                linear otherwise), or a specific scheme (``"ubwc"``,
                ``"afbc"``, ``"pvric"``, ``"dcc"``; allocation fails unless
                the device's native scheme matches). Requires
                ``access="none"``. ``Tensor.compression`` reports the
                outcome.
            contiguous: Require physically contiguous memory: a DMA-BUF from
                the Linux CMA heap, or an error naming the heap.
                ``Tensor.contiguity`` reports the outcome.

        Returns:
            A new image ``Tensor`` backed by the optimal memory type.

        Example: a display canvas the GPU renders and a Wayland compositor
        shows (linux-dmabuf), and an overlay the CPU writes::

            proc = ImageProcessor()
            frame = proc.create_image(1920, 1080, PixelFormat.Rgba, access="none")
            canvas = proc.create_image(1920, 1080, PixelFormat.Rgba, access="none")
            proc.set_class_colors([(255, 96, 0, 160)])  # translucent boxes
            proc.set_box_line_width(2)
            proc.convert(camera, frame)
            proc.draw_decoded_masks(canvas, boxes, scores, classes, background=frame)
            fd, stride = canvas.fd, canvas.row_stride  # to zwp_linux_dmabuf_v1

            overlay = proc.create_image(640, 64, PixelFormat.Rgba, access="write")
            with overlay.map("write") as m:  # map("read") raises PermissionError
                ...  # write every byte
        """

    def convert(
        self,
        src: EdgeFirstTensorExportable,
        dst: EdgeFirstTensorExportable,
        rotation: Rotation | None = None,
        flip: Flip | None = None,
        source: Region | None = None,
        letterbox: tuple[int, int, int, int] | None = None,
        *,
        options: ConvertOptions | None = None,
        letterbox_placement: PlacementArg | None = None,
        pad_output: tuple[float, float, float, float] | None = None,
        normalization: NormalizationArg | None = None,
    ) -> None:
        """Convert ``src`` into ``dst``, scaling, converting colour, rotating
        and flipping as needed, and wait for the GPU before returning.

        Args:
            src: Source image, from this or any other ``edgefirst.*``
                package (see ``crates/python-common/INTEROP.md``).
            dst: Destination image, written in place.
            rotation: Applied to the converted image.
            flip: Applied to the converted image.
            source: Sub-region of ``src`` to read, in source pixels.
                ``None`` reads the whole image.
            letterbox: ``(r, g, b, a)`` pad colour. Given, the source is
                fitted into the destination preserving aspect ratio and the
                remainder is padded; ``None`` stretches.
            options: A :class:`ConvertOptions` to start from; every other
                argument given here is applied on top of it.
            letterbox_placement: Where a letterboxed image sits.
            pad_output: Pad with these values as written to ``dst`` (after
                normalization, quantized for integer destinations) instead of
                mapping ``letterbox`` like the image.
            normalization: How pixel values map to ``dst``: a
                :class:`NormalizationPreset`, its name, or ``(mean, std)`` with
                one or three values each, as ``(x - mean) / std`` on 0-255
                values. Float destinations default to ``x / 255``. An integer
                destination with a non-default normalization must carry
                quantization (``Tensor.set_quantization``), which the result
                is quantized with.

        On Windows the destination's ``gpu_completion()`` reflects this
        convert afterwards.

        Raises:
            PermissionError: If a tensor's declared CPU access does not cover
                the CPU work (``src`` ``"read"``; ``dst`` ``"write"``, or
                ``"readwrite"`` for a view or a read-back pass), including
                when the GPU declined and the CPU fallback was refused.
        """

    def convert_deferred(
        self,
        src: EdgeFirstTensorExportable,
        dst: EdgeFirstTensorExportable,
        rotation: Rotation | None = None,
        flip: Flip | None = None,
        source: Region | None = None,
        letterbox: tuple[int, int, int, int] | None = None,
        *,
        options: ConvertOptions | None = None,
        letterbox_placement: PlacementArg | None = None,
        pad_output: tuple[float, float, float, float] | None = None,
        normalization: NormalizationArg | None = None,
    ) -> None:
        """Convert without waiting for the GPU.

        Same arguments as :meth:`convert`, but the OpenGL backend skips the
        per-call ``glFinish()``. Render N model inputs by looping this over
        ``dst.batch(n)`` / ``dst.view(region)`` row-bands of one batched
        destination, then call :meth:`flush` once. A deferred destination is
        not safe to read (or ``cuda_map``) until :meth:`flush` returns.
        Non-GL backends complete synchronously and :meth:`flush` is a no-op.

        On Windows the destination's ``gpu_completion()`` reflects this
        convert afterwards; the value covers the queued render.

        Raises:
            PermissionError: As :meth:`convert`.
        """

    def flush(self) -> None:
        """Complete all deferred converts since the last flush with a single
        GPU sync.

        After this returns, every destination written by
        :meth:`convert_deferred` is finished and safe to read back or
        ``cuda_map``. Non-GL backends return immediately.
        """

    def convert_with_fence(
        self,
        src: EdgeFirstTensorExportable,
        dst: EdgeFirstTensorExportable,
        rotation: Rotation | None = None,
        flip: Flip | None = None,
        source: Region | None = None,
        letterbox: tuple[int, int, int, int] | None = None,
        *,
        options: ConvertOptions | None = None,
        letterbox_placement: PlacementArg | None = None,
        pad_output: tuple[float, float, float, float] | None = None,
        normalization: NormalizationArg | None = None,
    ) -> int | None:
        """Convert without blocking on the GPU; returns a completion handle.

        Platforms:
            Linux, macOS, iOS, Android (sync-file descriptor) and Windows
            (event handle). Declared and callable on every platform.

        Takes the same ``src``/``dst``/``rotation``/``flip``/``source``/
        ``letterbox`` arguments as :meth:`convert`.

        Returns:
            A sync-file descriptor on Linux and Android, an event handle
            (as an integer) on Windows, or ``None`` when the convert
            completed synchronously (no native fence on this display). The
            caller owns the handle and closes it (``os.close`` for the fd,
            ``ctypes.windll.kernel32.CloseHandle`` for the Windows handle).

        On Windows the destination's ``gpu_completion()`` reflects this
        convert afterwards.

        Raises:
            PermissionError: As :meth:`convert`.
        """

    def alloc_tile_batch(
        self,
        n: int,
        cfg: TilingConfig,
        format: PixelFormat = PixelFormat.Rgba,
        dtype: DType = "uint8",
        memory: TensorMemory | None = None,
        access: str = "readwrite",
    ) -> Tensor:
        """Allocate the tall packed batch destination ``[tile_w, n*tile_h]``.

        A single parent that stacks ``n`` tiles vertically. Uses the same
        DMA-pitch alignment and memory selection as :meth:`create_image`.

        Args:
            n: Number of tiles to stack vertically.
            cfg: Tiling configuration (supplies ``tile_w``/``tile_h``).
            format: Pixel format (default: ``PixelFormat.Rgba``).
            dtype: Element data type string (default: ``"uint8"``).
            memory: Force a specific backing memory kind, or ``None`` to
                auto-select.
            access: Declared CPU access (``"readwrite"``/``"read"``/
                ``"write"``/``"none"``). Default ``"readwrite"``; widened as
                for :meth:`create_image`, with any predicted CPU write added
                as ``"readwrite"``.

        Returns:
            A new batched image ``Tensor`` sized ``[tile_w, n*tile_h]``.
        """

    def plan_tiles(
        self,
        width: int,
        height: int,
        cfg: TilingConfig,
    ) -> list[TilePlacement]:
        """Compute the tile grid and per-tile placements for a frame.

        Runs without touching the GPU.

        Args:
            width: Full-frame width in pixels.
            height: Full-frame height in pixels.
            cfg: Tiling configuration.

        Returns:
            The per-tile :class:`TilePlacement` list in tile-index order.
        """

    def tile_into(
        self,
        src: EdgeFirstTensorExportable,
        dst_batched: EdgeFirstTensorExportable,
        cfg: TilingConfig,
    ) -> list[TilePlacement]:
        """Render every tile of ``src`` into a tall packed parent.

        Issues one deferred convert per tile into ``dst_batched`` (from
        :meth:`alloc_tile_batch`), then a single ``flush``.

        Args:
            src: Whole-frame source image.
            dst_batched: Tall packed destination from :meth:`alloc_tile_batch`.
            cfg: Tiling configuration.

        Returns:
            The per-tile :class:`TilePlacement` list in tile-index order.

        Raises:
            PermissionError: As :meth:`convert`.
        """

    def tile_one(
        self,
        src: EdgeFirstTensorExportable,
        dst_slot: EdgeFirstTensorExportable,
        placement: TilePlacement,
        cfg: TilingConfig,
    ) -> None:
        """Render exactly one tile of ``src`` into a single destination slot.

        Deferred: call :meth:`flush` to complete it. ``placement`` (from
        :meth:`plan_tiles`) carries the geometry.

        Args:
            src: Whole-frame source image.
            dst_slot: A single model-input sized destination.
            placement: The tile's placement from :meth:`plan_tiles`.
            cfg: Tiling configuration.

        Raises:
            PermissionError: As :meth:`convert`.
        """

    if sys.platform == "linux":
        def import_image(
            self,
            fd: int,
            width: int,
            height: int,
            format: PixelFormat,
            dtype: DType = "uint8",
            stride: int | None = None,
            offset: int | None = None,
            chroma_fd: int | None = None,
            chroma_stride: int | None = None,
            chroma_offset: int | None = None,
            colorimetry: Colorimetry | None = None,
        ) -> Tensor:
            """Import an external DMA-BUF image.

            The fd is ``dup()``'d immediately — the caller retains ownership.
            The GPU renders directly into this buffer via EGL DMA-BUF import.

            For multiplane NV12/NV16, pass ``chroma_fd`` for the UV plane.

            Args:
                fd: DMA-BUF file descriptor (caller retains ownership).
                width: Image width in pixels.
                height: Image height in pixels.
                format: Pixel format of the buffer.
                dtype: Element data type string (default: ``"uint8"``).
                stride: Row stride in bytes for the image plane
                    (default: ``None`` = tightly packed).
                offset: Byte offset within the DMA-BUF where image data starts
                    (default: ``None`` = 0).
                chroma_fd: DMA-BUF fd for the UV chroma plane, for multiplane
                    NV12/NV16 (default: ``None`` = single-plane).
                chroma_stride: Row stride in bytes for the chroma plane
                    (default: ``None`` = tightly packed).
                chroma_offset: Byte offset within the chroma DMA-BUF where
                    data starts (default: ``None`` = 0).
                colorimetry: Colour signalling (matrix/range/primaries) from the
                    producer, used by ``convert()`` to pick the YUV→RGB matrix
                    and range (default: ``None`` = resolved by heuristic).

            Returns:
                A new image ``Tensor`` backed by the external DMA-BUF(s).

            Raises:
                RuntimeError: If a file descriptor is invalid, dimensions are zero,
                    the format layout is unsupported, NV12 height is odd, the
                    fd dup syscall fails, or stride is smaller than the minimum.
            """

    def set_class_colors(self, colors: list[tuple[int, int, int, int]]) -> None:
        """Set the colours used to render segmentation masks and boxes by
        class label.

        The palette holds 20 entries and a class index wraps around it. Only
        the leading entries this call supplies are replaced; the rest keep
        the defaults, and anything past the twentieth is ignored.

        The alpha channel sets the coverage of both the mask fill and the box
        outline: each is blended over the image with ``a / 255`` scaled by
        the draw's ``opacity``, so ``255`` draws opaque boxes and lower
        values translucent ones.

        Args:
            colors: ``(r, g, b, a)`` tuples, indexed by class label.

        Raises:
            RuntimeError: If the backend rejects the palette.
        """

    def set_box_line_width(self, width: int) -> None:
        """Set the thickness of the box outlines drawn by
        :meth:`draw_decoded_masks` and :meth:`draw_proto_masks`, in image
        pixels.

        The outline is drawn outside the box edges. The default is 3.

        Args:
            width: Outline thickness in pixels, at least 1.

        Raises:
            ValueError: If ``width`` is 0.
            OverflowError: If ``width`` is negative or does not fit in 32 bits.
            RuntimeError: If the backend rejects the width.
        """

    def set_interpolation(self, mode: InterpolationArg) -> None:
        """Sets how convert() resamples when it resizes, for converts whose
        options do not choose their own (``ConvertOptions(interpolation=...)``,
        which :meth:`ConvertOptions.from_spec` sets from a model's spec).

        Accepts "bilinear" (the default), "nearest", "bicubic" or "area", an
        :class:`Interpolation`, or its integer code. Bicubic and area are not
        implemented: they run bilinear, with a warning logged once per mode.
        Applies to every backend; the EDGEFIRST_INTERPOLATION environment
        variable overrides it.

        Args:
            mode: Interpolation mode.
        """

    def set_int8_interpolation(self, mode: str) -> None:
        """Sets the interpolation mode for int8 proto textures.

        Accepts "nearest", "bilinear", or "twopass". Default is "bilinear".
        Only affects rendering of quantized (int8) proto segmentation masks.
        A no-op where no OpenGL backend is available.

        Args:
            mode: Interpolation mode string.
        """

class TilingConfig:
    """Static tiling configuration for one model. Independent of frame size."""

    def __init__(
        self,
        tile_w: int,
        tile_h: int,
        overlap: float = 0.2,
        options: ConvertOptions | None = None,
    ) -> None:
        """Create a tiling config.

        Args:
            tile_w: Tile width = model input width.
            tile_h: Tile height = model input height.
            overlap: Minimum overlap ratio between adjacent tiles, in
                ``[0.0, 1.0)`` (default ``0.2``). The realized overlap is
                redistributed evenly and is always ``>=`` this.
            options: The :class:`ConvertOptions` every tile is converted
                with (fit, letterbox placement and pad, normalization,
                interpolation), e.g. from :meth:`ConvertOptions.from_spec`.
                Each tile replaces
                the source crop with its own; options that rotate or flip are
                refused when tiles are planned. ``None`` stretches each tile
                with the default pixel mapping.
        """

    def with_overlap(self, overlap: float) -> TilingConfig:
        """Return a copy with the minimum overlap ratio set (chainable)."""

    def with_options(self, options: ConvertOptions) -> TilingConfig:
        """Return a copy with the per-tile convert options set (chainable)."""

    @property
    def tile_w(self) -> int: ...
    @property
    def tile_h(self) -> int: ...
    @property
    def overlap(self) -> float: ...
    @property
    def options(self) -> ConvertOptions:
        """The options every tile is converted with."""

class TileSpec:
    """One tile's native-frame crop rectangle and its grid coordinates."""

    @property
    def source(self) -> Region:
        """Native crop in full-frame pixels."""

    @property
    def index(self) -> int:
        """Row-major flat tile index, ``0..count``."""

    @property
    def row(self) -> int:
        """Grid row."""

    @property
    def col(self) -> int:
        """Grid column."""

class TilePlacement:
    """How one tile was cut from the full frame and fed to the model.
    Produced by the tiling grid and consumed by
    :func:`edgefirst.decoder.lift_tile_boxes` and
    :class:`edgefirst.decoder.TiledFrameAccumulator`. All fields are
    full-frame pixels except ``content_region`` (the normalized image
    content of a letterboxed tile on the model input).
    """

    def __init__(
        self,
        index: int,
        count: int,
        origin: tuple[float, float],
        crop_size: tuple[float, float],
        frame_dims: tuple[float, float],
        content_region: ContentRegionArg | None = None,
    ) -> None:
        """Create a tile placement.

        Args:
            index: Tile index within the frame grid, ``0..count``.
            count: Total tiles for this frame.
            origin: Native crop origin ``(ox, oy)`` in full-frame pixels.
            crop_size: Native crop size ``(cw, ch)`` in full-frame pixels.
            frame_dims: Full-frame dimensions ``(frame_w, frame_h)`` in pixels.
            content_region: The image content on the model input as
                normalized ``(xmin, ymin, xmax, ymax)``, or ``None`` when the
                crop was stretched to fill it.
        """

    @property
    def index(self) -> int: ...
    @property
    def count(self) -> int: ...
    @property
    def origin(self) -> tuple[float, float]: ...
    @property
    def crop_size(self) -> tuple[float, float]: ...
    @property
    def content_region(self) -> tuple[float, float, float, float] | None: ...
    @property
    def frame_dims(self) -> tuple[float, float]: ...

def tile_grid(
    frame_w: int,
    frame_h: int,
    tile_w: int,
    tile_h: int,
    overlap: float = 0.2,
) -> list[TileSpec]:
    """Uniform overlapping EvenDist tile grid covering a ``frame_w``×``frame_h``
    frame.

    Row-major. Every tile is full-size unless the frame is smaller than the
    tile on an axis, in which case that axis yields a single whole-frame crop.

    Note:
        Argument order here is width-first. The Rust
        (``edgefirst_image::tile_grid``) and C (``hal_tile_grid``) APIs are
        height-first: ``(frame_h, frame_w, tile_h, tile_w)``.

    Args:
        frame_w: Full-frame width in pixels.
        frame_h: Full-frame height in pixels.
        tile_w: Tile width in pixels.
        tile_h: Tile height in pixels.
        overlap: Minimum overlap ratio in ``[0.0, 1.0)`` (default ``0.2``).

    Returns:
        The tile specs in row-major order.

    Raises:
        RuntimeError: If ``overlap`` is not in ``[0.0, 1.0)`` or a tile
            dimension is zero.
    """

# Remove `Protocol` from the module's public names.
del Protocol
