"""Type stubs for ``edgefirst.image``.

Split from the pre-0.29 monolithic ``edgefirst_hal.pyi``; assignment
mirrors the pymodule registrations in crates/python-image/src/lib.rs.
"""

import enum
import sys

# Public re-exports from `edgefirst.tensor`, a hard dependency of this package
# (see pyproject.toml). The redundant `as` form is what marks a name a public
# re-export to type checkers. TensorMemory/Colorimetry and the colour axis
# enums are registered directly in this module too (see
# crates/python-image/src/lib.rs), so an image tensor with an explicit memory
# backend or colorimetry override can be built using only edgefirst.image
# imports.
from typing import Protocol

import numpy as np
import numpy.typing as npt
from edgefirst.tensor import ColorEncoding as ColorEncoding
from edgefirst.tensor import Colorimetry as Colorimetry
from edgefirst.tensor import ColorRange as ColorRange
from edgefirst.tensor import ColorSpace as ColorSpace
from edgefirst.tensor import ColorTransfer as ColorTransfer

# DType is a type alias, not a runtime class — nothing to re-export.
from edgefirst.tensor import DType

# Image-only: no decoder or tracker dependency. Drawing takes boxes+masks+
# proto data via the capsule protocol; fused decode+draw lives on
# ``Decoder.draw_onto``.
from edgefirst.tensor import EdgeFirstTensorExportable as EdgeFirstTensorExportable
from edgefirst.tensor import PixelFormat as PixelFormat
from edgefirst.tensor import Region as Region
from edgefirst.tensor import Tensor as Tensor
from edgefirst.tensor import TensorMemory as TensorMemory

class EdgeFirstProtoDataExportable(Protocol):
    """Structural type for mask-prototype data via ``__edgefirst_protodata__``."""

    def __edgefirst_protodata__(self) -> tuple[object, object, str]: ...

# `Protocol` must not remain bound: stub-parity treats every top-level name
# as this package's public surface.
del Protocol

def version() -> str:
    """Return the HAL version string (matches ``Cargo.toml``)."""

class Normalization(enum.Enum):
    DEFAULT: Normalization
    """Default normalization
    | Output Type  | Default |
    |--------------|---------|
    | `np.uint8`   | RAW     |
    | `np.int8`    | SIGNED  |
    | `np.float16` | SIGNED  |
    | `np.float32` | SIGNED  |
    | `np.float64` | SIGNED  |
    """

    SIGNED: Normalization
    """Signed normalization
    | Output Type  | SIGNED Normalization  | Default Zero Point (zp) |
    |--------------|-----------------------|-------------------------|
    | `np.uint8`   | Not supported         | -                       |
    | `np.int8`    | (value - zp)          | 128                     |
    | `np.float16` | (value - zp) / 127.5) | 127.5                   |
    | `np.float32` | (value - zp) / 127.5) | 127.5                   |
    | `np.float64` | (value - zp) / 127.5) | 127.5                   |
    """
    UNSIGNED: Normalization
    """Unsigned normalization
    | Output Type  | UNSIGNED Normalization |
    |--------------|------------------------|
    | `np.uint8`   | value                  |
    | `np.int8`    | Not supported          |
    | `np.float16` | value / 255.0          |
    | `np.float32` | value / 255.0          |
    | `np.float64` | value / 255.0          |
    """
    RAW: Normalization
    """Raw normalization
    | Output Type  | RAW Normalization |
    |--------------|-------------------|
    | `np.uint8`   | Not supported     |
    | `np.int8`    | Not supported     |
    | `np.float16` | value             |
    | `np.float32` | value             |
    | `np.float64` | value             |
    """

class ColorMode(enum.Enum):
    """Controls how mask colors are assigned to detections."""

    Class: ColorMode
    """Color chosen by class label (default, correct for semantic segmentation)"""
    Instance: ColorMode
    """Color chosen by detection index (unique color per instance)"""
    Track: ColorMode
    """Color chosen by track ID (use with object tracking)"""

class MaskResolution:
    """Controls the resolution and coordinate frame of masks produced by
    :meth:`ImageProcessor.materialize_masks`.

    Construct via classmethods:

    - ``MaskResolution.Proto()`` — per-detection tiles at proto-plane
      resolution (historical default). Mask values are continuous sigmoid
      ``uint8 [0, 255]``.
    - ``MaskResolution.Scaled(width, height)`` — per-detection tiles at
      caller-specified pixel resolution, produced by upsampling the full
      proto plane once (correct edge-clamp bilinear) and cropping by bbox
      after sigmoid. Mask values are binary ``uint8 {0, 255}`` —
      interchangeable with the continuous sigmoid output via the same
      ``> 127`` test. If a ``letterbox`` is also passed to
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
    NoFlip: Flip
    """No flip"""
    Horizontal: Flip
    """Flip the image horizontally"""
    Vertical: Flip
    """Flip the image vertically"""

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
            Headless/compositor-free with zero external deps. Works on NVIDIA
            and newer Vivante drivers.
        Gbm: Direct GPU access via DRM render node (/dev/dri/renderD128).
            No compositor required. Needed on ARM Mali and older Vivante.
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
    pitch alignment requirement (currently 64 bytes).

    Use when allocating a DMA-BUF that will later be imported as an
    EGLImage by HAL's GL backend (or by any GLES driver that requires
    64-byte aligned pitches — currently Mali Valhall on i.MX 95).

    Pre-aligned widths (640, 1280, 1920, 3008, 3840, …) round-trip
    unchanged. Misaligned widths are bumped up to the next valid value.
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
    (currently 64). External callers that need to allocate their own
    DMA-BUFs should size them so each row pitch is a multiple of this value.
    """

class ImageProcessor:
    """Convert images between different formats, with optional rotation, flipping, and cropping."""

    def __init__(self, egl_display: EglDisplayKind | None = None) -> None:
        """Create an ImageProcessor with optional EGL display override.

        Args:
            egl_display: Force OpenGL to use this display type instead of
                auto-detecting. Use probe_egl_displays() to discover
                available displays. Ignored if EDGEFIRST_DISABLE_GL=1.
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
        letterbox: tuple[float, float, float, float] | None = None,
        color_mode: ColorMode = ColorMode.Class,
    ) -> None:
        """
        Draw detection boxes and optional segmentation masks onto ``dst``.

        This method draws pre-decoded results. For fused decode+draw, use
        ``Decoder.draw_onto`` on the decoder wheel.

        This function **always fully overwrites** ``dst`` — its prior contents
        are discarded.  If ``background`` is provided the output is
        ``background + masks``; otherwise ``dst`` is cleared to transparent
        (``0x00000000``) before masks are drawn.

        .. note::

            **Migrating from v0.16.3 or earlier:** if you previously loaded an
            image into ``dst`` before calling this function, you must now pass
            that image via ``background=`` instead.

        Args:
            dst: Output image tensor (always fully written by this call). Must
                be ``RGBA`` or ``RGB`` for the CPU backend, or
                ``RGBA``/``BGRA``/``RGB`` for OpenGL.
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
                **distinct underlying buffer** — aliasing ``dst`` (directly or
                via two ``Tensor`` wrappers over the same dmabuf fd) raises
                ``RuntimeError``.
            opacity: Scales the alpha of rendered mask and box colors.
                ``1.0`` (default) preserves the class color's alpha unchanged;
                ``0.5`` makes overlays semi-transparent. Clamped to [0, 1].
            letterbox: Optional ``(x0, y0, x1, y1)`` letterbox region in
                normalized coordinates for mapping model-space boxes back to
                image space. ``None`` (default) means no letterbox correction.
            color_mode: How to assign colors to detections.
                ``ColorMode.Class`` (default) colors by class label,
                ``ColorMode.Instance`` colors by detection index.

        Raises:
            RuntimeError: If ``dst`` format is unsupported by the active
                backend, if ``background`` aliases ``dst``, if ``bbox``,
                ``scores``, or ``classes`` lengths do not match, or if
                ``bbox`` shape is not ``(N, 4)``.
        """

    def materialize_masks(
        self,
        bbox: npt.NDArray[np.float32],
        scores: npt.NDArray[np.float32],
        classes: npt.NDArray[np.uintp],
        proto_data: EdgeFirstProtoDataExportable,
        letterbox: tuple[float, float, float, float] | None = None,
        resolution: MaskResolution | None = None,
    ) -> list[npt.NDArray[np.uint8]]:
        """Materialize per-instance segmentation masks from prototype data.

        Computes ``mask_coeff @ protos`` with sigmoid activation for each
        detection. The ``resolution`` parameter selects the output shape
        and value encoding:

        - ``MaskResolution.Proto()`` (default when ``resolution`` is
          ``None``): returns per-detection tiles at proto-plane resolution
          with **continuous sigmoid** values in ``uint8 [0, 255]``.
        - ``MaskResolution.Scaled(width, height)``: returns per-detection
          tiles at ``(width, height)`` pixel resolution with **binary**
          ``uint8 {0, 255}`` values (sigmoid > 0.5 threshold). The upsample
          happens on the full proto plane with edge-clamp bilinear
          sampling — correct by construction, unlike per-tile resize.
          Drop-in replacement for the continuous output under the caller's
          ``> 127`` threshold.

        The returned masks can be inspected for analytics, IoU computation,
        or export, and also passed to :meth:`draw_decoded_masks` for
        GPU-interpolated rendering.

        .. note::

            Calling ``materialize_masks`` + ``draw_decoded_masks`` separately
            prevents the HAL from using its internal fused optimization. For
            render-only use cases, prefer ``Decoder.draw_onto``.

        Args:
            bbox: ``(N, 4)`` float32 array of normalized bounding boxes.
            scores: ``(N,)`` float32 array of confidence scores.
            classes: ``(N,)`` uintp array of class indices.
            proto_data: Prototype data from :meth:`Decoder.decode_proto`,
                from this or another ``edgefirst.*`` package.
            letterbox: Optional ``(x0, y0, x1, y1)`` letterbox region in
                normalized model-input coordinates. When set with
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
        letterbox: tuple[float, float, float, float] | None = None,
        color_mode: ColorMode = ColorMode.Class,
    ) -> None:
        """Draw prototype masks onto ``dst`` without materialising per-instance
        mask arrays in Python. ``proto_data`` may come from another
        ``edgefirst.*`` package via the ``__edgefirst_protodata__`` capsule.
        """

    def create_image(
        self,
        width: int,
        height: int,
        format: PixelFormat = PixelFormat.Rgba,
        dtype: DType = "uint8",
        access: str = "none",
        compression: str | None = None,
    ) -> Tensor:
        """Create an image tensor with the processor's optimal memory backend.

        Selects the best available backing storage based on hardware capabilities:
        DMA-buf > PBO (GPU buffer) > system memory. Images created this way
        benefit from zero-copy GPU paths when used with this processor's
        ``convert()``.

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
            access: Declared CPU access — ``"none"`` (default), ``"read"``,
                ``"write"``, or ``"readwrite"``. Hardware (GPU/NPU) access is
                always implied; declare CPU access only when the script will
                touch the pixels directly (``map()``, ``numpy()``, buffer
                protocol). The strict ``"none"`` default keeps hardware
                pipelines eligible for vendor tile compression; undeclared
                CPU access still works best-effort but is logged and counted.
            compression: Tile-compression request — ``None`` (default,
                linear), ``"any"`` (device's native scheme when eligible,
                counted linear fallback), or a specific scheme (``"ubwc"``,
                ``"afbc"``, ``"pvric"``, ``"dcc"`` — allocation fails unless
                the device's native scheme matches). Requires
                ``access="none"``. Read the outcome via
                ``Tensor.compression``.

        Returns:
            A new image ``Tensor`` backed by the optimal memory type.
        """

    def alloc_tile_batch(
        self,
        n: int,
        cfg: TilingConfig,
        format: PixelFormat = PixelFormat.Rgba,
        dtype: DType = "uint8",
        memory: TensorMemory | None = None,
        access: str = "none",
    ) -> Tensor:
        """Allocate the tall packed batch destination ``[tile_w, n*tile_h]``.

        A single GL-importable parent that stacks ``n`` tiles vertically.
        Uses the same DMA-pitch alignment and ``Dma > Pbo > Mem`` selection
        as :meth:`create_image`. Caller-owned for pool reuse.

        Args:
            n: Number of tiles to stack vertically.
            cfg: Tiling configuration (supplies ``tile_w``/``tile_h``).
            format: Pixel format (default: ``PixelFormat.Rgba``).
            dtype: Element data type string (default: ``"uint8"``).
            memory: Force a specific backing memory kind, or ``None`` to
                auto-select.
            access: Declared CPU access (``"none"``/``"read"``/``"write"``/
                ``"readwrite"``). Default ``"none"``; pass ``"readwrite"``
                when the batch will be CPU-mapped.

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

        Runs **without touching the GPU**. Call once per frame to size pools
        and drive a tile stream.

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
        """

    def tile_one(
        self,
        src: EdgeFirstTensorExportable,
        dst_slot: EdgeFirstTensorExportable,
        placement: TilePlacement,
        cfg: TilingConfig,
    ) -> None:
        """Render exactly one tile of ``src`` into a single destination slot.

        Deferred — call :meth:`flush` on your own cadence so tiles overlap
        with inference. All geometry rides in ``placement`` (from
        :meth:`plan_tiles`).

        Args:
            src: Whole-frame source image.
            dst_slot: A single model-input sized destination.
            placement: The tile's placement from :meth:`plan_tiles`.
            cfg: Tiling configuration.
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

    def set_int8_interpolation(self, mode: str) -> None:
        """Sets the interpolation mode for int8 proto textures.

        Accepts "nearest", "bilinear", or "twopass". Default is "bilinear".
        Only affects rendering of quantized (int8) proto segmentation masks.
        A no-op where no OpenGL backend is available.

        Args:
            mode: Interpolation mode string.
        """

class Fit(enum.Enum):
    """How a tile crop is fit into the model input."""

    Stretch: Fit
    """Stretch the crop to fill the model input (the hot path; identity for
    the full-square tiles the grid produces)."""

    Letterbox: Fit
    """Preserve aspect ratio and pad with the :class:`TilingConfig` ``pad``
    colour."""

class TilingConfig:
    """Static tiling configuration for one model. Independent of frame size."""

    def __init__(
        self,
        tile_w: int,
        tile_h: int,
        overlap: float = 0.2,
        fit: Fit = Fit.Stretch,
        pad: tuple[int, int, int, int] = (114, 114, 114, 255),
    ) -> None:
        """Create a tiling config.

        Args:
            tile_w: Tile width = model input width.
            tile_h: Tile height = model input height.
            overlap: Minimum overlap ratio between adjacent tiles, in
                ``[0.0, 1.0)`` (default ``0.2``). The realized overlap is
                redistributed evenly and is always ``>=`` this.
            fit: How a tile crop is fit into the model input
                (default ``Fit.Stretch``).
            pad: Letterbox pad colour (RGBA), used only when
                ``fit == Fit.Letterbox`` (default ``(114, 114, 114, 255)``).
        """

    def with_overlap(self, overlap: float) -> TilingConfig:
        """Return a copy with the minimum overlap ratio set (chainable)."""

    def with_fit(self, fit: Fit) -> TilingConfig:
        """Return a copy with the fit mode set (chainable). ``Fit.Letterbox``
        uses the configured pad."""

    @property
    def tile_w(self) -> int: ...
    @property
    def tile_h(self) -> int: ...
    @property
    def overlap(self) -> float: ...
    @property
    def fit(self) -> Fit: ...
    @property
    def pad(self) -> tuple[int, int, int, int]: ...

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
    """How one tile was cut from the full frame and fed to the model — the
    shared input/output contract produced by the tiling grid and consumed by
    :func:`edgefirst.decoder.lift_tile_boxes` and
    :class:`edgefirst.decoder.TiledFrameAccumulator`. All fields are
    full-frame **pixels** except ``letterbox`` (normalized model-input bounds).
    """

    def __init__(
        self,
        index: int,
        count: int,
        origin: tuple[float, float],
        crop_size: tuple[float, float],
        frame_dims: tuple[float, float],
        letterbox: tuple[float, float, float, float] | None = None,
    ) -> None:
        """Create a tile placement.

        Args:
            index: Tile index within the frame grid, ``0..count``.
            count: Total tiles for this frame (the streaming fan-in fence).
            origin: Native crop origin ``(ox, oy)`` in full-frame pixels.
            crop_size: Native crop size ``(cw, ch)`` in full-frame pixels.
            frame_dims: Full-frame dimensions ``(frame_w, frame_h)`` in pixels.
            letterbox: Normalized letterbox content bounds
                ``(lx0, ly0, lx1, ly1)`` on the model input, or ``None`` when
                the crop was stretched to fill it.
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
    def letterbox(self) -> tuple[float, float, float, float] | None: ...
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
        Argument order here is width-first, following Python imaging
        conventions. The Rust (``edgefirst_image::tile_grid``) and C
        (``hal_tile_grid``) APIs are height-first:
        ``(frame_h, frame_w, tile_h, tile_w)``. Mind the swap when porting.

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
