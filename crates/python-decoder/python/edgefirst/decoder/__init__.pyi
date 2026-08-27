"""Type stubs for ``edgefirst.decoder``.

Split from the pre-0.29 monolithic ``edgefirst_hal.pyi``; assignment
mirrors the pymodule registrations in crates/python-decoder/src/lib.rs.
"""

import enum
from typing import Protocol

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

    ClassAgnostic: Suppresses all boxes based on IoU regardless of class.
    ClassAware: Only suppresses boxes of the same class.
    """

    ClassAgnostic: Nms
    ClassAware: Nms

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
    """

    Ultralytics: DecoderType
    ModelPack: DecoderType

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
        ``edgefirst_tensor_v1`` capsules, plus the prototype layout as a
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
        nms: Nms | None = Nms.ClassAgnostic,
    ) -> None:
        """
        Create a new Decoder instance from a dictionary configuration describing the model outputs.

        Args:
            config: Model output configuration dictionary.
            score_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            nms: NMS mode - Nms.ClassAgnostic (default), Nms.ClassAware, or None to bypass NMS.
        """

    @staticmethod
    def new_from_json_str(
        json_str: str,
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        nms: Nms | None = Nms.ClassAgnostic,
    ) -> Decoder:
        """
        Create a new Decoder instance from a JSON configuration string describing the model outputs.

        Args:
            json_str: JSON configuration string.
            score_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            nms: NMS mode - Nms.ClassAgnostic (default), Nms.ClassAware, or None to bypass NMS.
        """

    @staticmethod
    def new_from_yaml_str(
        yaml_str: str,
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        nms: Nms | None = Nms.ClassAgnostic,
    ) -> Decoder:
        """
        Create a new Decoder instance from a YAML configuration string describing the model outputs.

        Args:
            yaml_str: YAML configuration string.
            score_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            nms: NMS mode - Nms.ClassAgnostic (default), Nms.ClassAware, or None to bypass NMS.
        """

    @staticmethod
    def new_from_outputs(
        outputs: list[Output],
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        nms: Nms | None = Nms.ClassAgnostic,
        decoder_version: DecoderVersion | None = None,
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
            nms: NMS mode - Nms.ClassAgnostic (default), Nms.ClassAware, or None to bypass NMS.
            decoder_version: Optional decoder version for Ultralytics models.
                Set to DecoderVersion.Yolo26 for end-to-end models.
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
        """

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
    def pre_nms_top_k(self) -> int:
        """
        Maximum candidates fed into NMS after score filtering.
        Uses O(N) partial sort to cap O(N²) NMS cost. Default: 300.

        .. warning::

           The default of 300 is tuned for **deployment**
           (``score_threshold >= 0.25``) where few anchors pass the score
           filter.  For **COCO mAP evaluation** (``score_threshold = 0.001``),
           set this to the total anchor count (8400 for 640×640 YOLO models)
           or to ``0`` (no limit) to avoid discarding ~74% of valid
           candidates before NMS, which causes **~9 pp box mAP loss**.

           Deployment::

               decoder.score_threshold = 0.25
               # decoder.pre_nms_top_k = 300  (default, appropriate)

           COCO mAP evaluation::

               decoder.score_threshold = 0.001
               decoder.pre_nms_top_k = 8400   # all anchors
               decoder.max_det = 300

           Post-processing latency scales with candidate count.  At deployment
           thresholds the cost difference is negligible; at validation
           thresholds it is measurable but necessary for correct recall.
        """

    @pre_nms_top_k.setter
    def pre_nms_top_k(self, value: int): ...
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
        letterbox: tuple[float, float, float, float] | None = None,
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

class MergeConfig:
    def __init__(
        self,
        metric: MatchMetric = ...,
        threshold: float = 0.5,
        class_agnostic: bool = False,
        max_det: int = 300,
        score_threshold: float = 0.0,
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
