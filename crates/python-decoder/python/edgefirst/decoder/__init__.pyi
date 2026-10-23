"""Type stubs for ``edgefirst.decoder``.

Split from the pre-0.29 monolithic ``edgefirst_hal.pyi``; assignment
mirrors the pymodule registrations in crates/python-decoder/src/lib.rs.
"""

import dataclasses
import enum
import os
from typing import Any, NamedTuple, Protocol, Sequence

import numpy as np
import numpy.typing as npt

# Re-exported from `edgefirst.tensor` (a hard dependency of this package). The
# redundant `as` form is what marks it a public re-export to type checkers.
# TensorMemory/PixelFormat/Region/Colorimetry and the colour axis enums are
# registered directly in this module too (see
# crates/python-decoder/src/lib.rs), so callers building a value tensor for
# draw_masks()/materialize_masks() can name them without also depending on
# edgefirst.tensor.
from edgefirst.tensor import ColorEncoding as ColorEncoding
from edgefirst.tensor import Colorimetry as Colorimetry
from edgefirst.tensor import ColorRange as ColorRange
from edgefirst.tensor import ColorSpace as ColorSpace
from edgefirst.tensor import ColorTransfer as ColorTransfer

# The cross-package capsule protocol -- see crates/python-common/INTEROP.md.
# `decode`/`decode_proto`/`decode_tracked` accept any object implementing
# this, not just `edgefirst.tensor.Tensor`.
from edgefirst.tensor import EdgeFirstTensorExportable as EdgeFirstTensorExportable
from edgefirst.tensor import PixelFormat as PixelFormat
from edgefirst.tensor import Region as Region
from edgefirst.tensor import Tensor as Tensor
from edgefirst.tensor import TensorMemory as TensorMemory
from typing_extensions import TypeAlias

# "dynamic" in every wheel; "static" only in the G13 comparison build.
__tensor_backend__: str

def version() -> str:
    """Return the HAL version string (matches ``Cargo.toml``)."""

class EdgeFirstDecoderExportable(Protocol):
    """Structural type for anything that can hand a :class:`Decoder` across
    an ``edgefirst.*`` package boundary via the ``__edgefirst_decoder__``
    capsule protocol. See ``crates/python-common/INTEROP.md`` in the
    `hal <https://github.com/EdgeFirstAI/hal>`_ repository.
    """

    def __edgefirst_decoder__(self) -> object: ...

class EdgeFirstProtoDataExportable(Protocol):
    """Structural type for anything that can hand mask-prototype data
    across an ``edgefirst.*`` package boundary via the
    ``__edgefirst_protodata__`` capsule protocol. See
    ``crates/python-common/INTEROP.md`` in the
    `hal <https://github.com/EdgeFirstAI/hal>`_ repository.
    """

    def __edgefirst_protodata__(self) -> tuple[object, object, str]: ...

class Nms(enum.Enum):
    """Non-Maximum Suppression mode for object detection.

    Auto: Uses the model config's mode, falling back to ClassAware.
    ClassAware: Only suppresses boxes of the same class (default).
    ClassAgnostic: Suppresses all boxes based on IoU regardless of class.

    The integer values are the C ``EF_NMS_*`` codes: ``Auto`` is 1,
    ``ClassAware`` 2 and ``ClassAgnostic`` 3. Code 0 is NMS off, which Python
    spells ``None``.
    """

    Auto: Nms
    ClassAware: Nms
    ClassAgnostic: Nms

class HasNormalizedRegion(Protocol):
    """Anything with a normalized ``(xmin, ymin, xmax, ymax)`` content
    region, such as ``edgefirst.image.ContentRegion``."""

    @property
    def normalized(self) -> tuple[float, float, float, float]: ...

ContentRegionLike: TypeAlias = (
    tuple[float, float, float, float] | list[float] | HasNormalizedRegion
)
"""A ``content_region=`` argument: normalized ``(xmin, ymin, xmax, ymax)``
or an object with a ``normalized`` attribute."""

DetectionOutput: TypeAlias = tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.uintp]
]
"""Detection output type alias.
A tuple containing:
- boxes: A NumPy array of shape (N, 4) containing the bounding boxes in (x1, y1, x2, y2) format.
- scores: A NumPy array of shape (N,) containing the confidence scores for each bounding box.
- class_ids: A NumPy array of shape (N,) containing the class IDs for each bounding box.
"""

SegDetOutput: TypeAlias = tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.uintp],
    list[npt.NDArray[np.uint8]],
]
"""
Segmentation and Detection output type alias.
A tuple containing:
- boxes: A NumPy array of shape (N, 4) containing the bounding boxes in (x1, y1, x2, y2) format.
- scores: A NumPy array of shape (N,) containing the confidence scores for each bounding box.
- class_ids: A NumPy array of shape (N,) containing the class IDs for each bounding box.
- masks: A list of NumPy arrays containing per-detection segmentation masks.
  The exact shape depends on the method:

  - ``decode()``: shape ``(H, W, C)`` at prototype resolution. For instance
    segmentation models (e.g. YOLO) ``C=1`` — a binary per-instance mask
    (threshold at 128). For semantic segmentation models (e.g. ModelPack)
    ``C=num_classes`` — per-pixel class scores (use ``argmax`` over ``C``
    to get the class index).
"""

SegDetTrackedOutput: TypeAlias = tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.uintp],
    list[npt.NDArray[np.uint8]],
    list[object],
]
"""
Segmentation and Detection output type alias with tracking.
A tuple containing:
- boxes: A NumPy array of shape (N, 4) containing the bounding boxes in (x1, y1, x2, y2) format.
- scores: A NumPy array of shape (N,) containing the confidence scores for each bounding box.
- class_ids: A NumPy array of shape (N,) containing the class IDs for each bounding box.
- masks: A list of NumPy arrays containing per-detection segmentation masks.
  The exact shape depends on the method:
- tracks: Per-detection track objects from ``tracker.update``.
"""

class DecoderType(enum.Enum):
    """Decoder type — selects the post-processing algorithm family.

    - ``Ultralytics``: Ultralytics YOLO models (YOLOv5, YOLOv8, YOLO11, YOLO26)
    - ``ModelPack``: ModelPack models
    - ``Semantic``: generic semantic segmentation, decoded by
      :meth:`Decoder.decode_semantic`
    """

    Ultralytics: DecoderType
    ModelPack: DecoderType
    Semantic: DecoderType

class DecodeMode(enum.Enum):
    """Semantic segmentation decode mode.

    Accurate: Upsample the scores of every class, then take the argmax.
    Fast: Take the argmax at the model's output resolution, then upsample the
        class map with nearest-neighbour sampling.
    """

    Accurate: DecodeMode
    Fast: DecodeMode

class DecoderVersion(enum.Enum):
    """Decoder version for Ultralytics models.

    Specifies the YOLO architecture version, which determines the decoding strategy:

    - ``Yolov5``, ``Yolov8``, ``Yolo11``: Traditional models requiring external NMS.
    - ``Yolo26``: End-to-end models with NMS embedded in the model architecture.
      When set, the decoder uses end-to-end model types regardless of the ``nms`` setting.
    """

    Yolov5: DecoderVersion
    """YOLOv5 - anchor-based decoder, requires external NMS."""
    Yolov8: DecoderVersion
    """YOLOv8 - anchor-free DFL decoder, requires external NMS."""
    Yolo11: DecoderVersion
    """YOLO11 - anchor-free DFL decoder, requires external NMS."""
    Yolo26: DecoderVersion
    """YOLO26 - end-to-end model with embedded NMS (one-to-one matching heads)."""

class DimName(enum.Enum):
    """Named dimension for model output tensors.

    Used with ``dshape`` to give semantic meaning to each dimension,
    enabling the decoder to validate and interpret the tensor layout.
    """

    Batch: DimName
    """Batch dimension (typically 1)."""
    Height: DimName
    """Spatial height."""
    Width: DimName
    """Spatial width."""
    NumClasses: DimName
    """Number of object classes."""
    NumFeatures: DimName
    """Number of features per box (e.g. 4 box coords + N class scores)."""
    NumBoxes: DimName
    """Number of candidate boxes / anchors."""
    NumProtos: DimName
    """Number of segmentation prototype channels."""
    NumAnchorsXFeatures: DimName
    """Product of anchors and features (ModelPack split format)."""
    Padding: DimName
    """Padding dimension."""
    BoxCoords: DimName
    """Box coordinate dimension (typically 4)."""
    NumQueries: DimName
    """Query axis of mask-classification outputs."""

class Output:
    """A model output configuration for programmatic decoder setup.

    Use the static factory methods (``detection``, ``boxes``, ``scores``, etc.)
    to create outputs, then pass them to ``Decoder.new_from_outputs()``.

    **Shape specification** — provide one of:

    - ``shape``: anonymous integer dimensions, e.g. ``[1, 25200, 85]``
    - ``dshape``: named dimensions, e.g. ``[(DimName.Batch, 1), (DimName.NumFeatures, 85), ...]``

    If ``dshape`` is provided, ``shape`` is derived automatically.

    Example::

        # Anonymous shape:
        Output.detection(shape=[1, 25200, 85])

        # Named shape (preferred):
        Output.detection(dshape=[(DimName.Batch, 1),
                                 (DimName.NumFeatures, 85),
                                 (DimName.NumBoxes, 25200)])

        # With quantization and chaining:
        Output.detection(shape=[1, 84, 8400]).with_quantization(scale=0.004, zero_point=-123)
    """

    @staticmethod
    def detection(
        shape: list[int] | None = None,
        dshape: list[tuple[DimName, int]] | None = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a detection output (combined boxes + scores in one tensor).

        Expected ``DimName`` values: ``Batch``, ``NumFeatures``, ``NumBoxes``.

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """

    @staticmethod
    def boxes(
        shape: list[int] | None = None,
        dshape: list[tuple[DimName, int]] | None = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a boxes-only output (split detection format).

        Expected ``DimName`` values: ``Batch``, ``BoxCoords``, ``NumBoxes``.

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """

    @staticmethod
    def scores(
        shape: list[int] | None = None,
        dshape: list[tuple[DimName, int]] | None = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a scores-only output (split detection format).

        Expected ``DimName`` values: ``Batch``, ``NumClasses``, ``NumBoxes``.

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """

    @staticmethod
    def protos(
        shape: list[int] | None = None,
        dshape: list[tuple[DimName, int]] | None = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a protos output (segmentation prototype tensor).

        Expected ``DimName`` values: ``Batch``, ``NumProtos``, ``Height``, ``Width``.

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """

    @staticmethod
    def segmentation(
        shape: list[int] | None = None,
        dshape: list[tuple[DimName, int]] | None = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a segmentation output.

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """

    @staticmethod
    def mask_coefficients(
        shape: list[int] | None = None,
        dshape: list[tuple[DimName, int]] | None = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a mask coefficients output.

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """

    @staticmethod
    def mask(
        shape: list[int] | None = None,
        dshape: list[tuple[DimName, int]] | None = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a mask output.

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """

    @staticmethod
    def classes(
        shape: list[int] | None = None,
        dshape: list[tuple[DimName, int]] | None = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a classes output (class label indices for end-to-end split models).

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """

    def with_quantization(self, scale: float, zero_point: int) -> Output:
        """Set quantization parameters for this output.

        Returns self for method chaining.

        Args:
            scale: Quantization scale factor.
            zero_point: Quantization zero point.
        """

    def with_anchors(self, anchors: list[tuple[float, float]]) -> Output:
        """Set anchors for this output (detection outputs only).

        Returns self for method chaining.

        Args:
            anchors: List of (width, height) anchor pairs.

        Raises:
            ValueError: If called on a non-detection output.
        """

    def with_normalized(self, normalized: bool) -> Output:
        """Set the normalized flag for this output (detection/boxes outputs only).

        Returns self for method chaining.

        Args:
            normalized: True if box coordinates are in [0,1] range.

        Raises:
            ValueError: If called on an unsupported output type.
        """

class ProtoData:
    """Opaque prototype data from a segmentation model's decode step.

    Holds raw mask coefficients and prototype tensors. Pass to
    :meth:`ImageProcessor.materialize_masks` to compute per-instance masks
    for analytics or export, or use :meth:`Decoder.draw_onto` for
    fused GPU rendering instead.

    For detection-only models, :meth:`Decoder.decode_proto` returns ``None``
    instead of a ``ProtoData`` instance.
    """

    @property
    def layout(self) -> str:
        """Physical memory layout of the prototype tensor.

        Returns ``"nhwc"`` when protos shape is ``(H, W, K)`` or ``"nchw"``
        when shape is ``(K, H, W)``. Use this to interpret the tensor returned
        by :meth:`take_protos`.
        """

    def take_protos(self) -> Tensor | None:
        """Take ownership of the prototype masks tensor.

        Returns a Tensor whose shape depends on :attr:`layout`:

        - ``"nhwc"``: shape is ``(H, W, num_protos)``
        - ``"nchw"``: shape is ``(num_protos, H, W)``

        For quantized models the returned tensor carries quantization metadata
        accessible via the ``quantization`` property.

        Consumes the proto data's ``protos`` field — subsequent calls
        return ``None``.
        """

    def take_mask_coefficients(self) -> Tensor | None:
        """Take ownership of the per-detection mask coefficients tensor.

        Returns a Tensor with shape ``(num_detections, num_protos)``.

        Consumes the proto data's ``mask_coefficients`` field — subsequent
        calls return ``None``.
        """

    def __edgefirst_protodata__(
        self,
    ) -> tuple[object, object, str]:
        """Producer half of the cross-package ``ProtoData`` protocol.

        Composes the existing ``__edgefirst_tensor__`` capsule protocol
        rather than describing its own layout: returns the
        ``mask_coefficients`` and ``protos`` tensors as
        ``edgefirst_tensor_v2`` capsules, plus the prototype layout as a
        string (``"nhwc"`` or ``"nchw"``). See
        :class:`EdgeFirstProtoDataExportable` and
        ``crates/python-common/INTEROP.md``.
        """

class Decoder:
    def __init__(
        self,
        config: dict,
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        nms: Nms | None = Nms.Auto,
        input_dims: tuple[int, int] | None = None,
        multi_label: bool | None = None,
    ) -> None:
        """
        Create a new Decoder instance from a dictionary configuration describing the model outputs.

        Args:
            config: Model output configuration dictionary.
            score_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            nms: NMS mode - Nms.Auto (default; the config's mode, else
                Nms.ClassAware), Nms.ClassAgnostic, Nms.ClassAware, or None to
                bypass NMS.
            input_dims: Optional ``(width, height)`` model input, used to
                normalize pixel-space boxes when the config has no input shape.
            multi_label: Emit one box per class above ``score_threshold``
                instead of one per anchor (validation decode). None (default)
                uses the config's ``nms_multi_label``, else False.
                :meth:`decode_tracked` always uses one label per box.
        """

    @staticmethod
    def new_from_json_str(
        json_str: str,
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        nms: Nms | None = Nms.Auto,
        input_dims: tuple[int, int] | None = None,
        multi_label: bool | None = None,
    ) -> Decoder:
        """
        Create a new Decoder instance from a JSON configuration string describing the model outputs.

        Args:
            json_str: JSON configuration string.
            score_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            nms: NMS mode - Nms.Auto (default; the config's mode, else
                Nms.ClassAware), Nms.ClassAgnostic, Nms.ClassAware, or None to
                bypass NMS.
            input_dims: Optional ``(width, height)`` model input, used to
                normalize pixel-space boxes when the config has no input shape.
            multi_label: Emit one box per class above ``score_threshold``
                instead of one per anchor (validation decode). None (default)
                uses the config's ``nms_multi_label``, else False.
                :meth:`decode_tracked` always uses one label per box.
        """

    @staticmethod
    def new_from_yaml_str(
        yaml_str: str,
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        nms: Nms | None = Nms.Auto,
        input_dims: tuple[int, int] | None = None,
        multi_label: bool | None = None,
    ) -> Decoder:
        """
        Create a new Decoder instance from a YAML configuration string describing the model outputs.

        Args:
            yaml_str: YAML configuration string.
            score_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            nms: NMS mode - Nms.Auto (default; the config's mode, else
                Nms.ClassAware), Nms.ClassAgnostic, Nms.ClassAware, or None to
                bypass NMS.
            input_dims: Optional ``(width, height)`` model input, used to
                normalize pixel-space boxes when the config has no input shape.
            multi_label: Emit one box per class above ``score_threshold``
                instead of one per anchor (validation decode). None (default)
                uses the config's ``nms_multi_label``, else False.
                :meth:`decode_tracked` always uses one label per box.
        """

    @staticmethod
    def new_from_outputs(
        outputs: list[Output],
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        nms: Nms | None = Nms.Auto,
        decoder_version: DecoderVersion | None = None,
        input_dims: tuple[int, int] | None = None,
        multi_label: bool | None = None,
    ) -> Decoder:
        """Create a new Decoder from a list of Output objects.

        This provides a Pythonic way to configure the decoder programmatically
        without JSON/YAML configuration strings or dictionaries.

        The default thresholds (0.25 / 0.45) are tuned for typical YOLO models.
        The dict/JSON/YAML constructors use lower defaults (0.1 / 0.7) for
        backward compatibility.

        Example::

            decoder = Decoder.new_from_outputs(
                outputs=[
                    Output.detection(shape=[1, 84, 8400])
                        .with_quantization(scale=0.004, zero_point=-123)
                ],
                score_threshold=0.25,
                iou_threshold=0.45,
            )

        Args:
            outputs: List of Output objects describing the model outputs.
            score_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            nms: NMS mode - Nms.Auto (default; resolves to Nms.ClassAware, since
                outputs carry no config-level mode), Nms.ClassAgnostic,
                Nms.ClassAware, or None to bypass NMS.
            decoder_version: Optional decoder version for Ultralytics models.
                Set to DecoderVersion.Yolo26 for end-to-end models.
            input_dims: Optional ``(width, height)`` model input, used to
                normalize pixel-space boxes when the config has no input shape.
            multi_label: Emit one box per class above ``score_threshold``
                instead of one per anchor (validation decode). None (default)
                leaves it off, since outputs carry no ``nms_multi_label``.
                :meth:`decode_tracked` always uses one label per box.
        """

    def decode(
        self, model_output: list[EdgeFirstTensorExportable], max_boxes: int = 100
    ) -> SegDetOutput:
        """
        Decode model outputs into detection and segmentation results.

        Accepts HAL Tensor objects directly from model inference -- from
        this or any other ``edgefirst.*`` package, via the
        ``__edgefirst_tensor__`` capsule protocol (see
        ``crates/python-common/INTEROP.md``). Quantization parameters must
        be specified in the Decoder configuration when the tensors contain
        quantized data.

        Masks are returned at prototype resolution as 3D arrays of shape
        ``(H, W, C)``. For instance segmentation models (e.g. YOLO) ``C=1``
        -- a binary per-instance mask (threshold at 128). For semantic
        segmentation models (e.g. ModelPack) ``C=num_classes`` -- per-pixel
        class scores (use ``argmax`` over the last axis).

        Args:
            model_output: List of HAL Tensor objects from model inference,
                from this or another ``edgefirst.*`` package.
            max_boxes: Maximum number of detections to return (default: 100).
                Effective limit is ``min(max_boxes, decoder.max_det)``.

        Raises:
            PermissionError: If an output tensor is not declared
                CPU-readable.
        """

    def decode_semantic(
        self,
        model_output: list[EdgeFirstTensorExportable],
        dst: EdgeFirstTensorExportable,
        confidence: EdgeFirstTensorExportable | None = None,
        *,
        mode: DecodeMode | None = None,
        content_region: ContentRegionLike | None = None,
    ) -> None:
        """Decode the model's semantic segmentation output into a class map.

        Each pixel of ``dst`` is mapped through the content region of the
        model input, so letterbox padding is cropped and ``dst`` covers the
        original image at any resolution.

        Args:
            model_output: The model's output tensors, from this or another
                ``edgefirst.*`` package. The semantic output is found by
                name, by shape, or by its position when all outputs are
                passed.
            dst: ``(H, W)`` class map to write, ``uint8`` (up to 256 classes)
                or ``uint16``.
            confidence: Optional ``(H, W)`` ``float32`` or ``uint8`` map that
                receives the winning class's probability (``uint8`` scaled to
                255). Computed at the model's resolution and upsampled, so it
                is approximate and meant for overlays.
            mode: ``DecodeMode.Accurate`` or ``DecodeMode.Fast``; ``None`` uses
                the model's ``decode_mode``.
            content_region: Where the image content sits in the model input:
                normalized ``(xmin, ymin, xmax, ymax)``, or an object with a
                ``normalized`` attribute such as the ``ContentRegion`` from
                ``edgefirst.image.ConvertOptions.content_region``. ``None``
                when the image fills the input.

        Raises:
            RuntimeError: The model has no semantic output, or the tensors do
                not match it.
            PermissionError: An output tensor is not declared CPU-readable,
                or ``dst``/``confidence`` not CPU-writable.
        """

    @property
    def semantic_classes(self) -> int | None:
        """Classes of the semantic output: the channel count, ``2`` for one
        thresholded channel, ``0`` for a class map, ``None`` without one."""

    @property
    def semantic_mode(self) -> DecodeMode | None:
        """Default decode mode of the semantic output, ``None`` without one."""

    def decode_proto(
        self, model_output: list[EdgeFirstTensorExportable], max_boxes: int = 100
    ) -> tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.uintp],
        ProtoData | None,
    ]:
        """Decode model outputs into detection boxes and optional prototype data.

        For segmentation models, returns a :class:`ProtoData` instance that can
        be passed to :meth:`ImageProcessor.materialize_masks` to compute
        per-instance masks for analytics, export, or IoU computation.

        For detection-only models, returns ``None`` for proto_data but still
        populates detection boxes.

        .. note::

            Calling ``decode_proto`` + ``materialize_masks`` +
            ``draw_decoded_masks`` separately prevents the HAL from using its
            internal fused optimization. For render-only use cases, prefer
            :meth:`Decoder.draw_onto` which is 1.6--27x faster on
            tested platforms.

        Args:
            model_output: List of HAL Tensor objects from model inference,
                from this or another ``edgefirst.*`` package.
            max_boxes: Pre-allocation hint (default: 100). The actual output
                count is bounded by ``decoder.max_det`` (default: 300).
                The returned ``ProtoData.mask_coefficients`` always matches
                the detection count.

        Returns:
            ``(boxes, scores, classes, proto_data)`` where ``proto_data`` is
            ``None`` for detection-only models.

        Raises:
            PermissionError: If an output tensor is not declared
                CPU-readable.
        """

    def decode_tracked(
        self,
        tracker: object,
        timestamp: int,
        model_output: list[EdgeFirstTensorExportable],
        max_boxes: int = 100,
    ) -> SegDetTrackedOutput:
        """
        Decode model outputs into detection and segmentation results with tracking.

        Accepts HAL Tensor objects directly from model inference. Quantization
        parameters must be specified in the Decoder configuration when the
        tensors contain quantized data.

        Always decodes one label per box, regardless of :attr:`multi_label`:
        the tracker matches on IoU only, so per-class duplicates of one anchor
        would become phantom tracks. Logs a warning once per decoder when the
        decoder has multi-label enabled.

        Masks are returned at prototype resolution as 3D arrays of shape
        ``(H, W, C)``. For instance segmentation models (e.g. YOLO) ``C=1``
        -- a binary per-instance mask (threshold at 128). For semantic
        segmentation models (e.g. ModelPack) ``C=num_classes`` -- per-pixel
        class scores (use ``argmax`` over the last axis).

        Args:
            tracker: ByteTrack tracker instance.
            timestamp: Frame timestamp in nanoseconds.
            model_output: List of HAL Tensor objects from model inference,
                from this or another ``edgefirst.*`` package.
            max_boxes: Maximum number of detections to return (default: 100).
                Effective limit is ``min(max_boxes, decoder.max_det)``.

        Raises:
            PermissionError: If an output tensor is not declared
                CPU-readable.
        """

    @property
    def score_threshold(self) -> float:
        """
        Score threshold used when decoding detections with the `decode` method.
        Decoded detections will have a score equal to or higher than this threshold.
        """

    @score_threshold.setter
    def score_threshold(self, value: float): ...
    @property
    def iou_threshold(self) -> float:
        """
        IOU threshold used when decoding detections with the `decode` method.
        Detections with IOU equal to or higher than this threshold will be suppressed during non-maximum suppression.
        """

    @iou_threshold.setter
    def iou_threshold(self, value: float): ...
    @property
    def nms(self) -> Nms | None:
        """
        NMS mode used when decoding detections with the `decode` method.
        Returns Nms.ClassAgnostic, Nms.ClassAware, or None if NMS is bypassed.
        """

    @property
    def multi_label(self) -> bool:
        """
        Whether `decode` emits one box per class above the score threshold
        instead of one per anchor. `decode_tracked` ignores it and always uses
        one label per box.
        """

    @property
    def pre_nms_top_k(self) -> int:
        """
        Maximum candidates fed into NMS after score filtering; ``0`` means
        no limit. Uses O(N) partial sort to cap O(N²) NMS cost.

        Reading it returns the cap `decode` uses. Unset, that is 300, or
        30000 (Ultralytics' ``max_nms``) when `multi_label` is on, and
        `decode_tracked` (always argmax) uses 300. Assigning an ``int`` makes
        the cap explicit: every decode, tracked included, uses it as given,
        even if it equals a default. Assigning ``None`` returns to the
        defaults.

        .. warning::

           The default of 300 is tuned for **deployment**
           (``score_threshold >= 0.25``) where few anchors pass the score
           filter.  For **COCO mAP evaluation** (``score_threshold = 0.001``),
           set this to ``0`` (no limit) to avoid discarding ~74% of valid
           candidates before NMS, which causes **~9 pp box mAP loss**. Under
           argmax decode the anchor count (8400 for 640×640 YOLO models)
           also suffices; multi-label decode emits up to anchors × classes
           candidates.

           Deployment::

               decoder.score_threshold = 0.25
               # decoder.pre_nms_top_k = 300  (default, appropriate)

           COCO mAP evaluation::

               decoder.score_threshold = 0.001
               decoder.pre_nms_top_k = 0      # no limit
               decoder.max_det = 300

           Post-processing latency scales with candidate count.  At deployment
           thresholds the cost difference is negligible; at validation
           thresholds it is measurable but necessary for correct recall.
        """

    @pre_nms_top_k.setter
    def pre_nms_top_k(self, value: int | None): ...
    @property
    def max_det(self) -> int:
        """
        Maximum detections returned after NMS. Default: 300.
        """

    @max_det.setter
    def max_det(self, value: int): ...
    @property
    def normalized_boxes(self) -> bool | None:
        """
        Whether decoded bounding boxes are normalized to the [0, 1] range.
        Returns True if normalized, False if pixel coordinates, or None if
        unknown.

        Segmentation decoders (combined, split, and two-way) and the
        per-scale path divide by :attr:`input_dims` before returning, so
        they report ``True`` once ``input_dims`` is known. Detection-only,
        end-to-end YOLO, and ModelPack decoders report the raw schema
        annotation; if one of those returns ``False`` and you need
        ``[0, 1]`` boxes, divide by :attr:`input_dims` yourself. Never
        re-normalize when this returns ``True`` — dividing already-normalized
        coordinates collapses every detection to roughly zero.
        """

    def draw_onto(
        self,
        processor: object,
        model_output: list[EdgeFirstTensorExportable],
        dst: EdgeFirstTensorExportable,
        background: EdgeFirstTensorExportable | None = None,
        opacity: float = 1.0,
        content_region: ContentRegionLike | None = None,
        color_mode: object | None = None,
    ) -> DetectionOutput:
        """Decode and draw onto an ``edgefirst.image.ImageProcessor``.

        Prefers ``draw_proto_masks`` when the model yields prototype data;
        otherwise ``draw_decoded_masks``. This extension does not link image.
        """

    @property
    def input_dims(self) -> tuple[int, int] | None:
        """
        Model input dimensions ``(width, height)``, or ``None`` when unknown.

        Set through the ``input_dims`` constructor argument, or taken from a
        v2 schema's ``input.shape`` / ``input.dshape``. The decode paths
        listed under :attr:`normalized_boxes` use it to convert pixel-space
        boxes to ``[0, 1]`` before mask cropping. When it is ``None`` those
        paths skip the division, and pixel-space boxes will trip the
        prototype-mask safety guard.
        """

    def __edgefirst_decoder__(self) -> object:
        """Producer half of the cross-package decoder protocol.

        Returns a ``PyCapsule`` named ``edgefirst_decoder_v1``. The capsule
        borrows this decoder: it is valid only for the duration of the call
        it is passed into and must not be stored. See
        :class:`EdgeFirstDecoderExportable` and
        ``crates/python-common/INTEROP.md`` -- in particular the layout-guard
        caveat, since a ``Decoder`` cannot be decomposed the way a
        :class:`ProtoData` or a tensor can.
        """

class MatchMetric(enum.Enum):
    Iou: MatchMetric
    Ios: MatchMetric

class MergeMode(enum.Enum):
    """What the tiled-detection merge emits for a group of overlapping boxes.

    ``KeepBest`` (default) keeps the group's highest-scoring box exactly as
    decoded and drops the boxes it matched. ``Union`` replaces the group with
    its enclosing union carrying the max score -- the original GREEDYNMM
    merge, opt-in because it measured about 0.05 AP50 worse on every frame
    of the Ocean Cleanup ADIS 4K validation (TOP2-836).
    """

    KeepBest: MergeMode
    Union: MergeMode

class MergeConfig:
    def __init__(
        self,
        metric: MatchMetric = ...,
        threshold: float = 0.5,
        class_agnostic: bool = False,
        max_det: int = 300,
        score_threshold: float = 0.0,
        mode: MergeMode = ...,
    ) -> None: ...
    @property
    def metric(self) -> MatchMetric: ...
    @property
    def threshold(self) -> float: ...
    @property
    def class_agnostic(self) -> bool: ...
    @property
    def max_det(self) -> int: ...
    @property
    def score_threshold(self) -> float: ...
    @property
    def mode(self) -> MergeMode: ...

class TiledFrameAccumulator:
    def __init__(
        self,
        frame_dims: tuple[float, float],
        tiles_total: int,
        cfg: MergeConfig,
        est_per_tile: int = 16,
    ) -> None: ...
    def push_tile(
        self,
        bbox: npt.NDArray[np.float32],
        scores: npt.NDArray[np.float32],
        classes: npt.NDArray[np.uintp],
        placement: object,
    ) -> bool: ...
    def is_complete(self) -> bool: ...
    def remaining(self) -> int: ...
    def finalize(self) -> DetectionOutput: ...
    def finalize_normalized(self) -> DetectionOutput: ...

def lift_tile_boxes(
    bbox: npt.NDArray[np.float32],
    scores: npt.NDArray[np.float32],
    classes: npt.NDArray[np.uintp],
    placement: object,
) -> DetectionOutput: ...
def merge_tiled_detections(
    bbox: npt.NDArray[np.float32],
    scores: npt.NDArray[np.float32],
    classes: npt.NDArray[np.uintp],
    cfg: MergeConfig,
) -> DetectionOutput: ...

class InferredSchema(NamedTuple):
    """What :func:`infer_ultralytics_schema` derived from a model's signals.

    A ``NamedTuple``, so it unpacks positionally like a plain tuple while
    giving the three fields names.
    """

    schema: dict[str, Any]
    labels: list[str]
    description: str

def infer_ultralytics_schema(
    source: str,
    inputs: list[
        tuple[str, list[int | str], str]
        | tuple[str, list[int | str], str, tuple[list[float], list[int]] | None]
    ],
    outputs: list[
        tuple[str, list[int | str], str]
        | tuple[str, list[int | str], str, tuple[list[float], list[int]] | None]
    ],
    metadata: dict[str, str],
) -> InferredSchema:
    """Infer an Ultralytics YOLO schema from raw model I/O signals.

    Detection and segmentation are both supported, for Ultralytics v8/v11
    pre-NMS heads and YOLO26 end-to-end heads alike. This lets a caller build
    a decoder for an Ultralytics export straight from the shapes/dtypes/
    metadata its inference runtime reports, without a hand-written
    ``edgefirst.json``.

    Args:
        source: Container format the signals were read from: ``"onnx"``,
            ``"tflite"``, or ``"coreml"``. CoreML resolves pixel-space box
            coordinates the same way ONNX does, but only for the no-NMS
            anchor-grid export; a ``nms=True`` CoreML export produces
            Apple's NMS-pipeline artifact, which is refused rather than
            resolved. ``"other"`` is accepted but rejected by inference:
            whether boxes are pixel-space or ``[0, 1]`` follows the exporter
            and is not derivable from shapes, so an uncharacterized
            container is refused rather than guessed at.
        inputs: Input tensors as ``(name, shape, dtype)``, or with a 4th
            ``quantization`` element -- an input's quantization is accepted
            for symmetry with ``outputs`` and ignored.
        outputs: Output tensors as ``(name, shape, dtype)`` for an
            unquantized tensor, or ``(name, shape, dtype, quantization)``
            where ``quantization`` is ``(scales, zero_points)`` or ``None``.
            Only per-tensor quantization (one scale, one zero point) is
            usable -- the decoder consumes per-tensor only, so more than one
            scale is rejected rather than turned into a schema that cannot
            build.
        metadata: Raw model metadata key/values, passed through verbatim
            (ONNX ``metadata_props``, the TFLite ``metadata.json``
            envelope under whichever key it was captured, or CoreML's own
            metadata alongside Ultralytics' props).

    Note:
        Supported dtype strings are the tensor dtype names ``"i8"``,
        ``"u8"``, ``"i16"``, ``"u16"``, ``"i32"``, ``"u32"``, ``"f16"``,
        ``"f32"`` and ``"i64"``. The older spellings ``"int8"``, ``"uint8"``,
        ``"int16"``, ``"uint16"``, ``"int32"``, ``"uint32"``, ``"float16"``,
        ``"float32"`` and ``"int64"`` are still accepted.

        Shapes must be concrete. A model exported with ``dynamic=True``
        reports a symbolic dimension (``'batch'`` from ONNX, ``-1`` from
        TFLite); those are accepted as arguments and refused with a
        ``ValueError`` naming the tensor and axis, since the layout rules
        need real sizes.

    Returns:
        An :class:`InferredSchema` of ``(schema, labels, description)``: the
        inferred ``edgefirst.json`` schema v2 document as a dict ready for
        ``Decoder(schema)``, class names in index order, and a
        human-readable summary (e.g. ``"Ultralytics YOLOv8/11 detect, 80
        classes"``). It unpacks positionally as well.

    Raises:
        ValueError: the signals carry no recognizable Ultralytics schema
            (bad/missing metadata, an unsupported task, a class-count
            mismatch, an unrecognized output layout, or per-channel
            quantization), or a dtype/source string isn't one of the values
            listed above.
    """

class ConfigSource(enum.Enum):
    """The source that supplied a resolved value. The integer values are the
    C ``EF_CONFIG_SOURCE_*`` codes."""

    EmbeddedEdgefirst: ConfigSource
    """``edgefirst.json`` embedded in the model file."""
    SidecarEdgefirst: ConfigSource
    """``edgefirst.json`` (and ``labels.txt``) beside the model file."""
    EmbeddedHf: ConfigSource
    """Hugging Face configs embedded in the model file."""
    SidecarHf: ConfigSource
    """Hugging Face configs beside the model file."""
    UltralyticsMetadata: ConfigSource
    """Ultralytics export metadata."""
    Heuristic: ConfigSource
    """Guessed from the model's input and output names and shapes."""
    Default: ConfigSource
    """The documented default."""
    Override: ConfigSource
    """A caller override (:class:`ModelOverrides`)."""

class ModelFamily(enum.Enum):
    """The family a model's configuration identifies; it selects the
    convention a bare ``letterbox`` resize override follows. The integer
    values are the C ``EF_MODEL_FAMILY_*`` codes."""

    EdgeFirst: ModelFamily
    Ultralytics: ModelFamily
    HuggingFace: ModelFamily
    Unknown: ModelFamily

class ModelTask(enum.Enum):
    """What a model's outputs decode to. The integer values are the C
    ``EF_MODEL_TASK_*`` codes."""

    Detect: ModelTask
    Segment: ModelTask
    Semantic: ModelTask

class ConfigField(enum.Enum):
    """A resolved value whose source is recorded. The integer values are the
    C ``EF_CONFIG_FIELD_*`` codes."""

    Normalization: ConfigField
    Resize: ConfigField
    Placement: ConfigField
    Pad: ConfigField
    Interpolation: ConfigField
    Task: ConfigField
    DecodeMode: ConfigField
    Labels: ConfigField
    Size: ConfigField

class ModelSignals(NamedTuple):
    """A model's inputs, outputs and metadata, as a runtime reports them.

    Passed to :func:`resolve_model_config`. The fields take the same forms
    as :func:`infer_ultralytics_schema`'s arguments.
    """

    source: str
    inputs: Sequence[tuple[Any, ...]]
    outputs: Sequence[tuple[Any, ...]]
    metadata: dict[str, str] | None = None

@dataclasses.dataclass(frozen=True)
class PreprocessSpec:
    """A model's input preprocessing as plain values, consumed by
    ``edgefirst.image.ConvertOptions.from_spec``."""

    normalization: tuple[tuple[float, float, float], tuple[float, float, float]] = ...
    """``(mean, std)``, each an RGB triple on the 0-255 scale."""
    fit: str = "letterbox"
    """``"stretch"`` or ``"letterbox"``."""
    placement: str = "center"
    """``"center"`` or ``"topleft"``."""
    pad: tuple[float, float, float] = (114.0, 114.0, 114.0)
    """Letterbox pad per RGB channel, in the space :attr:`pad_space` names."""
    pad_space: str = "pixel"
    """``"pixel"`` or ``"normalized"``."""
    interpolation: str = "bilinear"
    """``"bilinear"``, ``"nearest"``, ``"bicubic"`` or ``"area"``."""
    size: tuple[int, int] | None = None
    """Model input ``(width, height)``, when known."""
    @property
    def mean(self) -> tuple[float, float, float]:
        """The per-channel mean, RGB order, on the 0-255 scale."""
    @property
    def std(self) -> tuple[float, float, float]:
        """The per-channel standard deviation, RGB order, on the 0-255 scale."""

class ModelOverrides:
    """Values that replace what discovery found, applied last and recorded
    as :attr:`ConfigSource.Override`.

    Grammar (shared with the C API and command lines):

    - ``task``: ``"detect" | "segment" | "semantic"`` or a :class:`ModelTask`
    - ``normalize``: ``"unsigned" | "signed" | "raw" | "imagenet" |
      "R,G,B/R,G,B"`` (mean/std in the 0-1 domain)
    - ``resize``: ``"stretch" | "letterbox" | "letterbox-center" |
      "letterbox-topleft"``; bare ``"letterbox"`` follows the model family
    - ``decode``: ``"accurate" | "fast"`` or a :class:`DecodeMode`
    - ``labels``: a list of non-empty class names or one comma-separated
      string. Resolution raises :class:`LabelCountMismatchError` when their
      number differs from a class count the model's outputs state;
      end-to-end detections and class maps state none, so any number is
      accepted there and the caller checks it.

    :meth:`ResolvedModelConfig.describe` prints these keys in the same
    grammar.

    Raises:
        OverrideError: A value outside its grammar; the message lists the
            accepted forms.
    """

    def __init__(
        self,
        task: str | ModelTask | None = None,
        normalize: str | None = None,
        resize: str | None = None,
        decode: str | DecodeMode | None = None,
        labels: list[str] | str | None = None,
    ) -> None: ...
    def set(self, key: str, value: str) -> None:
        """Set the override ``key`` (``task``, ``normalize``, ``resize``,
        ``decode`` or ``labels``) from its string form.

        Raises:
            OverrideError: An unknown key or a value outside its grammar.
        """
    @property
    def task(self) -> str | None: ...
    @property
    def normalize(self) -> str | None: ...
    @property
    def resize(self) -> str | None: ...
    @property
    def decode(self) -> str | None: ...
    @property
    def labels(self) -> list[str] | None: ...
    @property
    def is_empty(self) -> bool:
        """Whether no override is set."""

class ResolvedModelConfig:
    """A model's configuration as resolved from every source, with where
    each value came from. Returned by :func:`resolve_model_config`."""

    @property
    def schema(self) -> dict[str, Any]:
        """The output schema as an ``edgefirst.json`` schema v2 dict, ready
        for ``Decoder(schema)``."""
    @property
    def labels(self) -> list[str]:
        """Class names in channel order; empty when no source names them.

        Leading and trailing whitespace is trimmed from every source
        (ADE20K's ``"bed "`` is ``"bed"``). A source with unusable names, such
        as one empty after trimming, is ignored with a note naming the source
        and index, and names repeated ignoring case add a note."""
    @property
    def family(self) -> ModelFamily: ...
    @property
    def task(self) -> ModelTask: ...
    @property
    def decode_mode(self) -> DecodeMode | None:
        """The semantic decode mode; ``None`` for other tasks."""
    @property
    def description(self) -> str:
        """A one-line summary of the model."""
    @property
    def notes(self) -> list[str]:
        """Deviations and skipped files worth logging."""
    @property
    def reduce_labels(self) -> bool | None:
        """Hugging Face ``do_reduce_labels``, exposed but not applied."""
    @property
    def provenance(self) -> dict[ConfigField, tuple[ConfigSource, str | None]]:
        """Where each value came from, with a detail naming the file or the
        reason for a guess."""
    @property
    def preprocessing(self) -> PreprocessSpec:
        """The model's input preprocessing, for
        ``edgefirst.image.ConvertOptions.from_spec``. Its interpolation is
        the reference pipeline's; ``edgefirst.image`` runs bicubic and area
        as bilinear, and :attr:`notes` says so."""
    def source(self, field: ConfigField) -> tuple[ConfigSource, str | None] | None:
        """Where ``field`` came from, or ``None``."""
    def describe(self) -> list[str]:
        """One ``key=value [source]`` line per resolved value, e.g.
        ``"resize=letterbox-topleft [preprocessor_config.json]"``. Override
        keys print in the :class:`ModelOverrides` grammar."""

def resolve_model_config(
    path: str | os.PathLike[str],
    signals: ModelSignals | None = None,
    overrides: ModelOverrides | None = None,
) -> ResolvedModelConfig:
    """Resolve the configuration of the model file at ``path``.

    Sources, each filling only what earlier ones left unset: an embedded
    ``edgefirst.json``, one beside the model (with ``labels.txt``), embedded
    then sidecar Hugging Face ``config.json`` + ``preprocessor_config.json``,
    Ultralytics metadata in ``signals``, Hugging Face input/output names in
    ``signals``, then the documented defaults. ``overrides`` apply last.
    The file reads run with the GIL released.

    Args:
        path: The model file.
        signals: The model's inputs, outputs and metadata; needed to build
            a schema from Hugging Face configs, Ultralytics metadata or
            input/output names.
        overrides: Caller overrides.

    Raises:
        OSError: The model or a file beside it cannot be read, or the path
            is a directory.
        UnresolvedConfigError: Nothing supplies a required piece; the
            message names it and the override that supplies it.
        UnsupportedModelError: The configuration asks for something the
            library cannot run.
        OverrideError: An override cannot apply.
        LabelCountMismatchError: The ``labels`` override names a different
            number of classes than the model has (an ``OverrideError``).
        ModelConfigError: A configuration file is malformed. All four are
            ``ValueError`` subclasses.
    """

class ModelConfigError(ValueError):
    """A model configuration that is malformed or cannot be resolved."""

class OverrideError(ModelConfigError):
    """An override outside its grammar or one that does not apply."""

class LabelCountMismatchError(OverrideError):
    """A ``labels`` override naming a different number of classes than the
    model has: ``expected`` is the model's class count, ``found`` the number
    of names given."""

    expected: int
    found: int
    def __init__(self, message: str, expected: int, found: int) -> None: ...

class UnresolvedConfigError(ModelConfigError):
    """Nothing supplies a required piece of the configuration."""

class UnsupportedModelError(ModelConfigError, NotImplementedError):
    """A configuration that asks for something the library cannot run."""
