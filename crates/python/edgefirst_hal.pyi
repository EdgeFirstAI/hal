import numpy as np
import numpy.typing as npt
from typing import Literal, Optional, Union, Tuple, List
import enum
import sys

"""EdgeFirst HAL Python bindings."""

def version() -> str:
    """Return the HAL version string (matches ``Cargo.toml``)."""

def build_info() -> str:
    """Return a human-readable build configuration string."""

def is_dma_available() -> bool:
    """True when Linux DMA-BUF heap allocation is available.

    macOS callers should use :func:`is_iosurface_available` or the
    portable :func:`is_gpu_buffer_available` instead.
    """

def is_iosurface_available() -> bool:
    """True when macOS IOSurface allocation is available.

    Always returns ``False`` on non-macOS platforms.
    """

def is_gpu_buffer_available() -> bool:
    """True when a platform-native GPU-coherent buffer kind is available.

    Dispatches to :func:`is_dma_available` on Linux and
    :func:`is_iosurface_available` on macOS. Use this when you only care
    whether ``TensorMemory.DMA`` will succeed without caring which
    primitive backs it.
    """

def is_shm_available() -> bool:
    """True when POSIX shared memory allocation is available (Unix only)."""

def is_cuda_available() -> bool:
    """True when libcudart is loaded and all CUDA interop symbols resolved.

    Checks whether zero-copy CUDA tensor mapping is available on this system.
    Use this to gate CUDA-specific code paths before calling
    :meth:`Tensor.cuda_map`. The result is cached after the first call.
    """

class Nms(enum.Enum):
    """Non-Maximum Suppression mode for object detection.

    ClassAgnostic: Suppresses all boxes based on IoU regardless of class.
    ClassAware: Only suppresses boxes of the same class.
    """

    ClassAgnostic: Nms
    ClassAware: Nms

DetectionOutput = Tuple[
    npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.uintp]
]
"""Detection output type alias.
A tuple containing:
- boxes: A NumPy array of shape (N, 4) containing the bounding boxes in (x1, y1, x2, y2) format.
- scores: A NumPy array of shape (N,) containing the confidence scores for each bounding box.
- class_ids: A NumPy array of shape (N,) containing the class IDs for each bounding box.
"""

SegDetOutput = Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.uintp],
    List[npt.NDArray[np.uint8]],
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

SegDetTrackedOutput = Tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
    npt.NDArray[np.uintp],
    List[npt.NDArray[np.uint8]],
    List[TrackInfo],
]
"""
Segmentation and Detection output type alias with tracking.
A tuple containing:
- boxes: A NumPy array of shape (N, 4) containing the bounding boxes in (x1, y1, x2, y2) format.
- scores: A NumPy array of shape (N,) containing the confidence scores for each bounding box.
- class_ids: A NumPy array of shape (N,) containing the class IDs for each bounding box.
- masks: A list of NumPy arrays containing per-detection segmentation masks.
  The exact shape depends on the method:
- tracks: A list of TrackInfo objects containing tracking information for each detection.
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
        shape: Optional[List[int]] = None,
        dshape: Optional[List[Tuple[DimName, int]]] = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a detection output (combined boxes + scores in one tensor).

        Expected ``DimName`` values: ``Batch``, ``NumFeatures``, ``NumBoxes``.

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """
        ...

    @staticmethod
    def boxes(
        shape: Optional[List[int]] = None,
        dshape: Optional[List[Tuple[DimName, int]]] = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a boxes-only output (split detection format).

        Expected ``DimName`` values: ``Batch``, ``BoxCoords``, ``NumBoxes``.

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """
        ...

    @staticmethod
    def scores(
        shape: Optional[List[int]] = None,
        dshape: Optional[List[Tuple[DimName, int]]] = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a scores-only output (split detection format).

        Expected ``DimName`` values: ``Batch``, ``NumClasses``, ``NumBoxes``.

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """
        ...

    @staticmethod
    def protos(
        shape: Optional[List[int]] = None,
        dshape: Optional[List[Tuple[DimName, int]]] = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a protos output (segmentation prototype tensor).

        Expected ``DimName`` values: ``Batch``, ``NumProtos``, ``Height``, ``Width``.

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """
        ...

    @staticmethod
    def segmentation(
        shape: Optional[List[int]] = None,
        dshape: Optional[List[Tuple[DimName, int]]] = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a segmentation output.

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """
        ...

    @staticmethod
    def mask_coefficients(
        shape: Optional[List[int]] = None,
        dshape: Optional[List[Tuple[DimName, int]]] = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a mask coefficients output.

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """
        ...

    @staticmethod
    def mask(
        shape: Optional[List[int]] = None,
        dshape: Optional[List[Tuple[DimName, int]]] = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a mask output.

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """
        ...

    @staticmethod
    def classes(
        shape: Optional[List[int]] = None,
        dshape: Optional[List[Tuple[DimName, int]]] = None,
        decoder: DecoderType = DecoderType.Ultralytics,
    ) -> Output:
        """Create a classes output (class label indices for end-to-end split models).

        Args:
            shape: Anonymous integer dimensions (mutually exclusive with dshape).
            dshape: Named dimensions (mutually exclusive with shape).
            decoder: Decoder type (default: Ultralytics).
        """
        ...

    def with_quantization(self, scale: float, zero_point: int) -> Output:
        """Set quantization parameters for this output.

        Returns self for method chaining.

        Args:
            scale: Quantization scale factor.
            zero_point: Quantization zero point.
        """
        ...

    def with_anchors(self, anchors: List[Tuple[float, float]]) -> Output:
        """Set anchors for this output (detection outputs only).

        Returns self for method chaining.

        Args:
            anchors: List of (width, height) anchor pairs.

        Raises:
            ValueError: If called on a non-detection output.
        """
        ...

    def with_normalized(self, normalized: bool) -> Output:
        """Set the normalized flag for this output (detection/boxes outputs only).

        Returns self for method chaining.

        Args:
            normalized: True if box coordinates are in [0,1] range.

        Raises:
            ValueError: If called on an unsupported output type.
        """
        ...

class ProtoData:
    """Opaque prototype data from a segmentation model's decode step.

    Holds raw mask coefficients and prototype tensors. Pass to
    :meth:`ImageProcessor.materialize_masks` to compute per-instance masks
    for analytics or export, or use :meth:`ImageProcessor.draw_masks` for
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
        ...

    def take_protos(self) -> Optional[Tensor]:
        """Take ownership of the prototype masks tensor.

        Returns a Tensor whose shape depends on :attr:`layout`:

        - ``"nhwc"``: shape is ``(H, W, num_protos)``
        - ``"nchw"``: shape is ``(num_protos, H, W)``

        For quantized models the returned tensor carries quantization metadata
        accessible via the ``quantization`` property.

        Consumes the proto data's ``protos`` field — subsequent calls
        return ``None``.
        """
        ...

    def take_mask_coefficients(self) -> Optional[Tensor]:
        """Take ownership of the per-detection mask coefficients tensor.

        Returns a Tensor with shape ``(num_detections, num_protos)``.

        Consumes the proto data's ``mask_coefficients`` field — subsequent
        calls return ``None``.
        """
        ...

class Quantization:
    """Quantization parameters for an integer tensor.

    Four modes, matching the EdgeFirst model metadata spec:
    per-tensor/per-channel × symmetric/asymmetric.
    """

    @staticmethod
    def per_tensor(scale: float, zero_point: int) -> Quantization: ...
    @staticmethod
    def per_tensor_symmetric(scale: float) -> Quantization: ...
    @staticmethod
    def per_channel(
        scales: list[float], zero_points: list[int], axis: int
    ) -> Quantization: ...
    @staticmethod
    def per_channel_symmetric(scales: list[float], axis: int) -> Quantization: ...
    @property
    def scale(self) -> list[float]: ...
    @property
    def zero_point(self) -> Optional[list[int]]: ...
    @property
    def axis(self) -> Optional[int]: ...
    @property
    def is_per_tensor(self) -> bool: ...
    @property
    def is_per_channel(self) -> bool: ...
    @property
    def is_symmetric(self) -> bool: ...

class Decoder:
    def __init__(
        self,
        config: dict,
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        nms: Optional[Nms] = Nms.ClassAgnostic,
    ) -> None:
        """
        Create a new Decoder instance from a dictionary configuration describing the model outputs.

        Args:
            config: Model output configuration dictionary.
            score_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            nms: NMS mode - Nms.ClassAgnostic (default), Nms.ClassAware, or None to bypass NMS.
        """
        ...

    @staticmethod
    def new_from_json_str(
        json_str: str,
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        nms: Optional[Nms] = Nms.ClassAgnostic,
    ) -> Decoder:
        """
        Create a new Decoder instance from a JSON configuration string describing the model outputs.

        Args:
            json_str: JSON configuration string.
            score_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            nms: NMS mode - Nms.ClassAgnostic (default), Nms.ClassAware, or None to bypass NMS.
        """
        ...

    @staticmethod
    def new_from_yaml_str(
        yaml_str: str,
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        nms: Optional[Nms] = Nms.ClassAgnostic,
    ) -> Decoder:
        """
        Create a new Decoder instance from a YAML configuration string describing the model outputs.

        Args:
            yaml_str: YAML configuration string.
            score_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            nms: NMS mode - Nms.ClassAgnostic (default), Nms.ClassAware, or None to bypass NMS.
        """
        ...

    @staticmethod
    def new_from_outputs(
        outputs: List[Output],
        score_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        nms: Optional[Nms] = Nms.ClassAgnostic,
        decoder_version: Optional[DecoderVersion] = None,
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
        ...

    def decode(self, model_output: List[Tensor], max_boxes: int = 100) -> SegDetOutput:
        """
        Decode model outputs into detection and segmentation results.

        Accepts HAL Tensor objects directly from model inference. Quantization
        parameters must be specified in the Decoder configuration when the
        tensors contain quantized data.

        Masks are returned at prototype resolution as 3D arrays of shape
        ``(H, W, C)``. For instance segmentation models (e.g. YOLO) ``C=1``
        -- a binary per-instance mask (threshold at 128). For semantic
        segmentation models (e.g. ModelPack) ``C=num_classes`` -- per-pixel
        class scores (use ``argmax`` over the last axis).

        Args:
            model_output: List of HAL Tensor objects from model inference.
            max_boxes: Maximum number of detections to return (default: 100).
                Effective limit is ``min(max_boxes, decoder.max_det)``.
        """
        ...

    def decode_proto(
        self, model_output: List[Tensor], max_boxes: int = 100
    ) -> Tuple[
        npt.NDArray[np.float32],
        npt.NDArray[np.float32],
        npt.NDArray[np.uintp],
        Optional[ProtoData],
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
            :meth:`ImageProcessor.draw_masks` which is 1.6--27x faster on
            tested platforms.

        Args:
            model_output: List of HAL Tensor objects from model inference.
            max_boxes: Pre-allocation hint (default: 100). The actual output
                count is bounded by ``decoder.max_det`` (default: 300).
                The returned ``ProtoData.mask_coefficients`` always matches
                the detection count.

        Returns:
            ``(boxes, scores, classes, proto_data)`` where ``proto_data`` is
            ``None`` for detection-only models.
        """
        ...

    def decode_tracked(
        self,
        tracker: ByteTrack,
        timestamp: int,
        model_output: List[Tensor],
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
            model_output: List of HAL Tensor objects from model inference.
            max_boxes: Maximum number of detections to return (default: 100).
                Effective limit is ``min(max_boxes, decoder.max_det)``.
        """
        ...

    @property
    def score_threshold(self) -> float:
        """
        Score threshold used when decoding detections with the `decode` method.
        Decoded detections will have a score equal to or higher than this threshold.
        """
        ...

    @score_threshold.setter
    def score_threshold(self, value: float): ...
    @property
    def iou_threshold(self) -> float:
        """
        IOU threshold used when decoding detections with the `decode` method.
        Detections with IOU equal to or higher than this threshold will be suppressed during non-maximum suppression.
        """
        ...

    @iou_threshold.setter
    def iou_threshold(self, value: float): ...
    @property
    def nms(self) -> Optional[Nms]:
        """
        NMS mode used when decoding detections with the `decode` method.
        Returns Nms.ClassAgnostic, Nms.ClassAware, or None if NMS is bypassed.
        """
        ...

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
        ...

    @pre_nms_top_k.setter
    def pre_nms_top_k(self, value: int): ...
    @property
    def max_det(self) -> int:
        """
        Maximum detections returned after NMS. Default: 300.
        """
        ...

    @max_det.setter
    def max_det(self, value: int): ...
    @property
    def normalized_boxes(self) -> Optional[bool]:
        """
        Whether decoded bounding boxes are normalized to [0, 1] range.
        Returns True if normalized, False if pixel coordinates, or None if unknown.
        """
        ...

class TensorMemory(enum.Enum):
    if sys.platform == "linux":
        DMA: TensorMemory
        """
        Direct Memory Access (DMA) allocation. Incurs additional overhead for memory reading/writing with the CPU. 
        Allows for hardware acceleration when supported
        """

        SHM: TensorMemory
        """
        POSIX Shared Memory allocation. Suitable for inter-process
        communication, but not suitable for hardware acceleration.
        """
    PBO: TensorMemory
    """
    GPU Pixel Buffer Object (PBO) allocation. Used for zero-copy GPU
    upload/readback on platforms without DMA-buf support.
    """
    MEM: TensorMemory
    """Regular system memory allocation"""

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

class Tensor:
    if sys.platform == "linux":
        def __init__(
            self,
            shape: list[int],
            dtype: Literal[
                "int8",
                "uint8",
                "int16",
                "uint16",
                "int32",
                "uint32",
                "int64",
                "uint64",
                "float32",
                "float64",
            ] = "float32",
            mem: None | TensorMemory = None,
            name: None | str = None,
        ) -> None: ...
        """
        Create a new tensor with the given shape, memory type, and optional
        name. If no name is given, a random name will be generated. If no
        memory type is given, the best available memory type will be chosen
        based on the platform and environment variables.
            
        On Linux platforms, the order of preference is: DMA -> SHM -> MEM.
        On non-Linux platforms, only MEM is available.
        
        # Environment Variables
        - `EDGEFIRST_TENSOR_FORCE_MEM`: If set to a non-zero and non-false
        value, forces the use of regular system memory allocation
        (`TensorMemory.MEM`) regardless of platform capabilities.
        """

        @staticmethod
        def from_fd(
            fd: int,
            shape: list[int],
            dtype: Literal[
                "int8",
                "uint8",
                "int16",
                "uint16",
                "int32",
                "uint32",
                "int64",
                "uint64",
                "float32",
                "float64",
            ] = "float32",
            name: None | str = None,
        ) -> Tensor: ...
        """
        Create a new tensor using the given file descriptor, shape, and optional
        name. If no name is given, a random name will be generated.

        Inspects the file descriptor to determine the appropriate tensor type
        (DMA or SHM) based on the device major and minor numbers.

        The fd is ``dup()``'d immediately — the caller retains ownership
        of the original fd and must close it when done.
        """

        @property
        def fd(self) -> int: ...
        """Gets a duplicate of the file descriptor associated with the tensor's memory. The caller will be responsible for closing the file descriptor."""

    else:
        def __init__(
            self,
            shape: list[int],
            dtype: Literal[
                "int8",
                "uint8",
                "int16",
                "uint16",
                "int32",
                "uint32",
                "int64",
                "uint64",
                "float32",
                "float64",
            ] = "float32",
            mem: None | TensorMemory = None,
            name: None | str = None,
        ) -> None: ...
        """
        Create a new tensor with the given shape, memory type, and optional
        name. If no name is given, a random name will be generated. If no
        memory type is given, the best available memory type will be chosen
        based on the platform and environment variables.

        On Linux platforms, the order of preference is: DMA -> SHM -> MEM.
        On non-Linux platforms, only MEM is available.

        # Environment Variables
        - `EDGEFIRST_TENSOR_FORCE_MEM`: If set to a non-zero and non-false
        value, forces the use of regular system memory allocation
        (`TensorMemory.MEM`) regardless of platform capabilities.
        """

    @property
    def dtype(
        self,
    ) -> Literal[
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "int64",
        "uint64",
        "float32",
        "float64",
    ]: ...
    """The data type of the tensor."""

    @property
    def size(self) -> int: ...
    """The size of the tensor in bytes."""

    @property
    def memory(self) -> TensorMemory: ...
    """The memory type of the tensor."""

    @property
    def name(self) -> str: ...
    """The name of the tensor."""

    @property
    def shape(self) -> list[int]: ...
    """The shape of the tensor."""

    def reshape(self, shape: list[int]) -> None: ...
    """Reshape the tensor to the given shape. The total number of elements must remain the same."""

    def view(self, region: Region) -> Tensor:
        """Zero-copy rectangular sub-region view — the source/destination crop.

        ``region`` is in pixels of the image's leading frame. The view shares
        the parent's buffer (and ``BufferIdentity``) with no copy, addressing
        the sub-rectangle by offset + the parent's row pitch.
        ``convert(src, dst.view(region), ...)`` renders into that sub-rectangle.
        The parent must be a packed-format image tensor.

        Args:
            region: Sub-rectangle (pixels) into the parent image.

        Returns:
            A new ``Tensor`` viewing the requested sub-rectangle.
        """

    def batch(self, n: int) -> Tensor:
        """Borrow batch element ``n`` of a batched tensor as a zero-copy view.

        A batched tensor prepends ``N`` as the leading dimension over the
        per-element image layout (``[N, H, W, C]`` packed or ``[N, C, H, W]``
        planar). ``batch(n)`` returns element ``n`` — the contiguous per-element
        region at byte offset ``n * element_size``, sharing the parent's buffer.
        ``batch(0)`` on a tensor with ``N == 1`` is equivalent to the whole
        tensor.

        Args:
            n: Batch element index (``0 <= n < N``).

        Returns:
            A new ``Tensor`` viewing element ``n``.
        """

    def set_format(self, format: PixelFormat) -> None:
        """Attach pixel format metadata to this tensor.

        Validates that the tensor's shape is compatible with the format's
        layout (packed, planar, or semi-planar). This enables ``from_fd()``
        tensors to be used as image conversion destinations.

        Args:
            format: Pixel format to attach.

        Raises:
            RuntimeError: If the tensor shape doesn't match the format layout.
        """
        ...

    if sys.platform == "linux":
        def dmabuf_clone(self) -> int:
            """Clone the DMA-BUF file descriptor backing this tensor.

            Returns a new file descriptor that the caller must close.

            Returns:
                A new file descriptor (int) that the caller must close.

            Raises:
                RuntimeError: If the tensor is not DMA-backed or fd clone fails.
            """
            ...

    if sys.platform == "darwin":
        @staticmethod
        def from_iosurface(
            surface_ref: int,
            shape: list[int],
            dtype: Literal[
                "int8",
                "uint8",
                "int16",
                "uint16",
                "int32",
                "uint32",
                "int64",
                "uint64",
                "float32",
                "float64",
            ] = "uint8",
            name: None | str = None,
        ) -> Tensor:
            """Wrap an externally-allocated IOSurface as a Tensor (macOS only).

            ``surface_ref`` is an ``IOSurfaceRef`` cast to ``int`` — typically
            obtained via ``ctypes`` from a CoreVideo / AVFoundation /
            VideoToolbox handle, or via ``IOSurfaceLookup(id)`` to recover a
            surface received over XPC. The surface is retained for the
            tensor's lifetime; the caller keeps its own reference and must
            release it independently.

            Args:
                surface_ref: ``IOSurfaceRef`` as ``int`` (non-zero).
                shape: Tensor shape. The product of all dimensions times
                    the element size must fit within
                    ``IOSurfaceGetAllocSize(surface_ref)``; a mismatched
                    shape raises ``RuntimeError`` instead of risking
                    out-of-bounds access at map time.
                dtype: Element type; defaults to ``"uint8"`` for image data.
                name: Optional tensor name for debugging.

            Returns:
                A new ``Tensor`` reporting ``TensorMemory.DMA``.

            Raises:
                RuntimeError: If ``surface_ref`` is null, the import
                    fails, or the requested shape exceeds the surface's
                    allocated size.
            """
            ...

        @property
        def iosurface_id(self) -> int | None:
            """``IOSurfaceID`` for cross-process surface sharing (macOS only).

            Returns ``None`` when the tensor is not IOSurface-backed. The ID
            is a 32-bit handle stable for the lifetime of the IOSurface; it
            can be passed across process boundaries (Mach port, XPC) and
            recovered via ``IOSurfaceLookup(id)``.
            """

        @property
        def iosurface_ref(self) -> int | None:
            """Borrowed ``IOSurfaceRef`` as an ``int`` (macOS only).

            Hand this off to native macOS APIs that take an ``IOSurfaceRef``
            directly (``CIImage``, ``AVSampleBufferDisplayLayer``,
            ``CVPixelBufferCreateWithIOSurface``). The integer value is a
            raw pointer — wrap it with ``ctypes.c_void_p(...)`` before
            passing to a ctypes-bound function.

            The pointer's lifetime is tied to this tensor — the HAL holds
            the only retain count. If the surface must outlive this
            tensor, call ``CFRetain`` (via ctypes) on the pointer and
            pair it with a matching ``CFRelease``. Do *not* call
            ``CFRelease`` on the borrowed pointer without first
            ``CFRetain``-ing — that would drop HAL's retain and produce
            a use-after-free.

            Returns ``None`` when the tensor is not IOSurface-backed.

            Example — hand the surface to ``CIImage``::

                import ctypes
                from edgefirst_hal import Tensor, TensorMemory

                # Create the tensor (or import an existing IOSurface).
                t = Tensor.image(1280, 720, "rgba", mem=TensorMemory.DMA)

                # Wrap the raw IOSurfaceRef for ctypes handoff.
                surf_ptr = ctypes.c_void_p(t.iosurface_ref)

                # `ci_image_with_iosurface` is whatever native API you
                # bound via ctypes; the IOSurface stays valid while `t`
                # is alive.
                # ci_image_with_iosurface(surf_ptr)
            """

    def map(self) -> TensorMap:
        """Map the tensor's memory for direct read/write access.

        Returns a ``TensorMap`` context manager that exposes the raw buffer.
        Use with a ``with`` statement to ensure the mapping is released.

        Example — write a numpy array into a tensor::

            import numpy as np
            from edgefirst_hal import Tensor

            tensor = Tensor([480, 640, 3], dtype="float32")
            data = np.random.rand(480, 640, 3).astype(np.float32)

            with tensor.map() as m:
                # np.asarray(memoryview(m)) honours the buffer-protocol strides,
                # so padded (DMA/GPU) tensors map correctly without shearing.
                dst = np.asarray(memoryview(m))
                dst[:] = data

        Example — read tensor data as numpy::

            with tensor.map() as m:
                arr = np.asarray(memoryview(m))
                print(arr.mean())

        .. tip::

            For bulk numpy-to-tensor copies, prefer :meth:`from_numpy` which
            validates dtypes and handles the mapping internally.

        Raises:
            BufferError: If the tensor is already mapped or has been unmapped.
        """
        ...

    def cuda_map(self) -> Optional[CudaMap]:
        """Attempt a zero-copy CUDA device-pointer mapping.

        Returns a :class:`CudaMap` context manager, or ``None`` if CUDA is
        unavailable for this tensor (libcudart not found, or the tensor was
        not registered with CUDA). Fast-fails to ``None`` without GL-thread
        routing.

        The recommended pattern is to try ``cuda_map()`` first and fall back
        to ``map()`` when it returns ``None``::

            cm = tensor.cuda_map()
            if cm is not None:
                with cm as m:
                    trt_set_input_address(m.device_ptr)   # zero-copy GPU
            else:
                with tensor.map() as host:
                    run_cpu_path(host)                    # CPU fallback

        Returns:
            A :class:`CudaMap` context manager, or ``None``.
        """
        ...

    @staticmethod
    def image(
        width: int,
        height: int,
        format: PixelFormat,
        mem: TensorMemory | None = None,
    ) -> Tensor:
        """Create an image tensor with the given dimensions and pixel format.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            format: Pixel format for the image data.
            mem: Optional memory type override. If None, the best available
                memory type is chosen automatically.
        """
        ...

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
        ...

    @staticmethod
    def peek_image_info_file(filename: str) -> ImageInfo:
        """Parse the header of an image file without decoding pixels.

        Args:
            filename: Path to the image file.

        Returns:
            ImageInfo with native width, height, format, row_stride, and the
            EXIF rotation_degrees / flip_horizontal.
        """
        ...

    def save_jpeg(self, filename: str, quality: int = 80) -> None:
        """Save this image tensor as a JPEG file.

        The tensor must have an image pixel format (e.g. RGB, RGBA).

        Args:
            filename: Output file path.
            quality: JPEG quality (1-100, default 80).
        """
        ...

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

        Args:
            data: Raw JPEG or PNG bytes.

        Returns:
            ImageInfo with native width, height, format, row_stride, and the
            EXIF rotation_degrees / flip_horizontal.

        Raises:
            RuntimeError: If the tensor is too small or the dtype is unsupported.
        """
        ...

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
        ...

    def normalize_to_numpy(
        self,
        dst: npt.NDArray[np.uint8]
        | npt.NDArray[np.int8]
        | npt.NDArray[np.float16]
        | npt.NDArray[np.float32]
        | npt.NDArray[np.float64],
        normalization: "Normalization" = ...,
        zero_point: None | int = None,
    ) -> None:
        """Normalize image data and write to a numpy array.

        The optional ``zero_point`` parameter specifies the zero point for
        signed normalization. RGBA images are converted to RGB by dropping
        the alpha channel.

        Args:
            dst: Destination numpy array.
            normalization: Normalization mode (default: DEFAULT).
            zero_point: Optional zero point for signed normalization.
        """
        ...

    def from_numpy(self, src: npt.NDArray) -> None:
        """Copy data from a numpy array into this tensor.

        Accepts any numpy dtype as long as it matches the tensor's dtype.
        The total number of elements must match. Both contiguous and
        non-contiguous (strided) arrays are supported:

        - **Contiguous arrays** use a direct memcpy (fastest).
        - **Non-contiguous arrays** (slices, transposes) are copied
          element-wise via the array's stride metadata.
        - **Large copies** (≥256 KiB) are parallelized automatically.

        Example::

            import numpy as np
            from edgefirst_hal import Tensor

            # float32 model output → float32 tensor
            tensor = Tensor([1, 10, 6], dtype="float32")
            output = model.run(input_data)  # returns np.float32 array
            tensor.from_numpy(output.reshape(1, 10, 6))

            # uint8 image → uint8 tensor
            tensor = Tensor([480, 640, 3], dtype="uint8")
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            tensor.from_numpy(image)

            # Non-contiguous slice — works without ascontiguousarray()
            big = np.zeros((1000, 1000), dtype=np.float32)
            tensor = Tensor([500, 500], dtype="float32")
            tensor.from_numpy(big[:500, :500])

        Args:
            src: Source numpy array. The dtype must match the tensor's
                dtype (e.g. ``float32`` tensor requires ``np.float32``).
                Total element count must match. Contiguous and strided
                layouts are both accepted.

        Raises:
            RuntimeError: If the numpy dtype does not match the tensor
                dtype, or the element count differs.
        """
        ...

    @property
    def format(self) -> PixelFormat | None:
        """Pixel format of this tensor (None if not an image tensor)."""
        ...

    @property
    def width(self) -> int | None:
        """Image width in pixels (None if not an image tensor)."""
        ...

    @property
    def height(self) -> int | None:
        """Image height in pixels (None if not an image tensor)."""
        ...

    @property
    def row_stride(self) -> int | None:
        """Physical row pitch in bytes, or ``None`` for tightly packed tensors.

        Set for every image tensor allocated via :meth:`image` or configured
        via ``configure_image`` (DMA, IOSurface, and self-allocated semi-planar
        tensors always carry a 64-byte-aligned stride). ``None`` only for
        non-image tensors or raw tensors without a pixel format.

        Use :meth:`effective_row_stride` when you need a non-``None`` fallback
        equal to the minimum tight stride.
        """
        ...

    @property
    def is_planar(self) -> bool:
        """Whether this image uses a planar pixel layout."""
        ...

    @property
    def quantization(self) -> Optional[Quantization]:
        """Quantization metadata, or ``None`` for float tensors and
        unquantized integer tensors."""
        ...

    @property
    def colorimetry(self) -> Colorimetry | None:
        """Colour signalling (matrix/range/primaries), or ``None`` if undefined.

        Set automatically by the codec on decode (JPEG → JFIF/BT.601-full,
        PNG → sRGB) and carried through ``convert()`` to pick the YUV→RGB
        matrix and range. ``None`` is never auto-filled; consumers resolve
        missing axes via an SD/HD height heuristic at use time.
        """
        ...

    @colorimetry.setter
    def colorimetry(self, value: Colorimetry | None) -> None: ...
    def set_quantization_per_tensor(self, scale: float, zero_point: int) -> None:
        """Attach per-tensor asymmetric quantization. Integer tensors only."""
        ...

    def set_quantization_per_tensor_symmetric(self, scale: float) -> None:
        """Attach per-tensor symmetric quantization. Integer tensors only."""
        ...

    def set_quantization_per_channel(
        self, scales: list[float], zero_points: list[int], axis: int
    ) -> None:
        """Attach per-channel asymmetric quantization. Integer tensors only.
        Raises on length mismatch or invalid axis."""
        ...

    def set_quantization_per_channel_symmetric(
        self, scales: list[float], axis: int
    ) -> None:
        """Attach per-channel symmetric quantization. Integer tensors only."""
        ...

    def clear_quantization(self) -> None:
        """Remove any quantization metadata from this tensor."""
        ...

class TensorMap:
    def unmap(self) -> None: ...
    def numpy(self) -> memoryview: ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> object: ...
    def __setitem__(self, index: int, value: object) -> None: ...
    def __getbuffer__(self, view, _flags) -> None: ...
    def __releasebuffer__(self, view) -> None: ...
    def __enter__(self) -> TensorMap: ...
    def __exit__(self, _exc_type, _exc_value, _traceback) -> None: ...

class CudaMap:
    """Scoped zero-copy CUDA device-pointer mapping for a tensor.

    Obtain via :meth:`Tensor.cuda_map`. Use as a context manager; the
    mapping is released on ``__exit__`` so the GPU buffer can be reused
    by the next ``convert()`` call.

    Example::

        cm = tensor.cuda_map()
        if cm is not None:
            with cm as m:
                trt_set_input_address(m.device_ptr)   # zero-copy GPU input
        else:
            with tensor.map() as host:
                run_cpu_path(host)                    # CPU fallback
    """

    @property
    def device_ptr(self) -> int:
        """Raw CUDA device pointer as an integer.

        Pass to TensorRT ``setInputTensorAddress``, cupy, or pycuda for
        zero-copy GPU input. Returns ``0`` if the mapping has been released.
        """
        ...

    @property
    def size(self) -> int:
        """Length of the mapping in bytes. Returns ``0`` if released."""
        ...

    def __len__(self) -> int: ...
    def release(self) -> None:
        """Release the CUDA mapping (idempotent).

        Called automatically on ``with`` exit. May also be called explicitly
        when early release is needed before the ``with`` block exits.
        """
        ...

    def __enter__(self) -> CudaMap: ...
    def __exit__(self, _exc_type, _exc_value, _traceback) -> None: ...

class ColorSpace(enum.Enum):
    """Colour primaries (the chromaticities of the RGB primaries)."""

    Bt709: ColorSpace
    Bt2020: ColorSpace
    Srgb: ColorSpace
    Smpte170m: ColorSpace

class ColorTransfer(enum.Enum):
    """Transfer function (opto-electronic / gamma)."""

    Bt709: ColorTransfer
    Srgb: ColorTransfer
    Pq: ColorTransfer
    Hlg: ColorTransfer
    Linear: ColorTransfer

class ColorEncoding(enum.Enum):
    """YCbCr encoding matrix — selects the YUV↔RGB coefficients."""

    Bt601: ColorEncoding
    Bt709: ColorEncoding
    Bt2020: ColorEncoding

class ColorRange(enum.Enum):
    """Quantization range of the luma/chroma samples."""

    Full: ColorRange
    """Full range (0–255), e.g. JFIF/JPEG."""
    Limited: ColorRange
    """Limited / studio range (luma 16–235), e.g. broadcast video."""

class Colorimetry:
    """Four-axis colour signalling (primaries / transfer / matrix / range).

    Each axis is independently optional; ``None`` means "undefined" and is
    resolved at use time by an SD/HD height heuristic. Carried on image
    ``Tensor`` objects and consumed by ``convert()`` to select the exact
    YUV→RGB matrix and range.
    """

    def __init__(
        self,
        space: ColorSpace | None = None,
        transfer: ColorTransfer | None = None,
        encoding: ColorEncoding | None = None,
        range: ColorRange | None = None,
    ) -> None: ...
    @staticmethod
    def from_v4l2(
        colorspace: int, xfer: int, ycbcr_enc: int, quant: int
    ) -> Colorimetry:
        """Build from the four raw V4L2 colorimetry integers.

        A ``DEFAULT`` (0) ``ycbcr_enc``/``quant`` is resolved from the
        colorspace (e.g. ``V4L2_COLORSPACE_JPEG`` → BT.601 full-range) per the
        kernel ``V4L2_MAP_*_DEFAULT`` rules; an unrecognised value maps to
        ``None``.
        """
        ...

    @property
    def space(self) -> ColorSpace | None:
        """Colour primaries, or ``None`` if undefined."""
        ...

    @property
    def transfer(self) -> ColorTransfer | None:
        """Transfer function, or ``None`` if undefined."""
        ...

    @property
    def encoding(self) -> ColorEncoding | None:
        """YCbCr encoding matrix, or ``None`` if undefined."""
        ...

    @property
    def range(self) -> ColorRange | None:
        """Quantization range, or ``None`` if undefined."""
        ...

class PixelFormat(enum.Enum):
    """Pixel format for image tensors."""

    Rgb: PixelFormat
    """Packed RGB [H, W, 3]"""

    Rgba: PixelFormat
    """Packed RGBA [H, W, 4]"""

    Bgra: PixelFormat
    """Packed BGRA [H, W, 4]. Destination-only format for
    Cairo/Wayland compositing (ARGB32 on little-endian)."""

    Grey: PixelFormat
    """Grayscale [H, W, 1]"""

    Yuyv: PixelFormat
    """Packed YUV 4:2:2, YUYV byte order [H, W, 2]"""

    Vyuy: PixelFormat
    """Packed YUV 4:2:2, VYUY byte order [H, W, 2]"""

    Nv12: PixelFormat
    """Semi-planar YUV 4:2:0 [H*3/2, W]"""

    Nv16: PixelFormat
    """Semi-planar YUV 4:2:2 [H*2, W]"""

    Nv24: PixelFormat
    """Semi-planar YUV 4:4:4 [H*3, W] (full chroma). Emitted by the JPEG
    decoder for 4:4:4 sources."""

    PlanarRgb: PixelFormat
    """Planar RGB, channels-first [3, H, W]"""

    PlanarRgba: PixelFormat
    """Planar RGBA, channels-first [4, H, W]"""

    def __init__(self, name: str) -> None:
        """Create a PixelFormat from a string name (e.g. 'RGBA', 'NV12', 'GREY')."""
        ...

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
    def Proto(cls) -> "MaskResolution":
        """Per-detection tile at proto-plane resolution (default)."""
        ...

    @classmethod
    def Scaled(cls, width: int, height: int) -> "MaskResolution":
        """Per-detection tile at ``(width, height)`` pixel resolution."""
        ...

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
        ...

class Region:
    """A rectangular sub-region (pixels) defined by its top-left corner (x, y)
    and dimensions (width, height). Used for ``Tensor.view(region)`` and the
    source crop of ``convert``."""

    def __init__(self, x: int, y: int, width: int, height: int): ...
    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...

class EglDisplayKind:
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

class EglDisplayInfo:
    """A validated, available EGL display discovered by probe_egl_displays()."""

    @property
    def kind(self) -> EglDisplayKind:
        """The type of EGL display."""
        ...

    @property
    def description(self) -> str:
        """Human-readable description for logging/diagnostics."""
        ...

def probe_egl_displays() -> list[EglDisplayInfo]:
    """Probe for available EGL displays supporting headless OpenGL ES 3.0.

    Returns displays in priority order (PlatformDevice, GBM, Default).
    Each display is validated with eglInitialize and checked for required
    extensions (EGL_KHR_surfaceless_context, EGL_KHR_no_config_context).
    An empty list means OpenGL is not available on this system.

    Raises:
        RuntimeError: If libEGL.so.1 cannot be loaded.
    """
    ...

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
    ...

def align_width_for_pixel_format(
    width: int,
    format: PixelFormat,
    dtype: str = "uint8",
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
    ...

def gpu_dma_buf_pitch_alignment_bytes() -> int:
    """Required DMA-BUF row pitch alignment in bytes for GL backend imports
    (currently 64). External callers that need to allocate their own
    DMA-BUFs should size them so each row pitch is a multiple of this value.
    """
    ...

class ImageProcessor:
    """Convert images between different formats, with optional rotation, flipping, and cropping."""

    def __init__(self, egl_display: EglDisplayKind | None = None) -> None:
        """Create an ImageProcessor with optional EGL display override.

        Args:
            egl_display: Force OpenGL to use this display type instead of
                auto-detecting. Use probe_egl_displays() to discover
                available displays. Ignored if EDGEFIRST_DISABLE_GL=1.
        """
        ...

    def draw_decoded_masks(
        self,
        dst: Tensor,
        bbox: npt.NDArray[np.float32],
        scores: npt.NDArray[np.float32],
        classes: npt.NDArray[np.uintp],
        seg: List[npt.NDArray[np.uint8]] = [],
        background: Optional[Tensor] = None,
        opacity: float = 1.0,
        letterbox: Optional[Tuple[float, float, float, float]] = None,
        color_mode: ColorMode = ColorMode.Class,
    ) -> None:
        """
        Draw detection boxes and optional segmentation masks onto ``dst``.

        This method draws pre-decoded results. For the fused decode+draw path
        (recommended for most use cases), use ``draw_masks()`` instead.

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
        ...

    def materialize_masks(
        self,
        bbox: npt.NDArray[np.float32],
        scores: npt.NDArray[np.float32],
        classes: npt.NDArray[np.uintp],
        proto_data: ProtoData,
        letterbox: Optional[Tuple[float, float, float, float]] = None,
        resolution: Optional[MaskResolution] = None,
    ) -> List[npt.NDArray[np.uint8]]:
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
            render-only use cases, prefer :meth:`draw_masks` which is 1.6--27x
            faster on tested platforms.

        Args:
            bbox: ``(N, 4)`` float32 array of normalized bounding boxes.
            scores: ``(N,)`` float32 array of confidence scores.
            classes: ``(N,)`` uintp array of class indices.
            proto_data: Prototype data from :meth:`Decoder.decode_proto`.
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
        ...

    def draw_masks(
        self,
        decoder: Decoder,
        model_output: List[Tensor],
        dst: Tensor,
        tracker: Optional[ByteTrack] = None,
        timestamp: Optional[int] = None,
        background: Optional[Tensor] = None,
        opacity: float = 1.0,
        letterbox: Optional[Tuple[float, float, float, float]] = None,
        color_mode: ColorMode = ColorMode.Class,
    ) -> Union[
        DetectionOutput,
        Tuple[
            npt.NDArray[np.float32],
            npt.NDArray[np.float32],
            npt.NDArray[np.uintp],
            List[TrackInfo],
        ],
    ]:
        """
        Decode model outputs and draw colored masks directly onto the
        destination image in a single fused call. This is the fastest path
        for visualization -- masks never leave Rust/GPU, eliminating the
        Python round-trip overhead of ``decode()`` + ``draw_decoded_masks()``.

        For segmentation models, prototype data is passed directly to the
        renderer which evaluates the mask at every output pixel. For
        detection-only models, this falls back to the standard drawing path.

        When ``tracker`` is provided, object tracking is performed and the
        return value includes a ``tracks`` list:
        ``(boxes, scores, classes, tracks)``. If ``timestamp`` is omitted,
        the current system time in nanoseconds is used.

        Without a tracker the return value is ``(boxes, scores, classes)``.

        This function **always fully overwrites** ``dst`` — its prior contents
        are discarded.  If ``background`` is provided the output is
        ``background + masks``; otherwise ``dst`` is cleared to transparent
        (``0x00000000``) before masks are drawn.

        .. note::

            **Migrating from v0.16.3 or earlier:** if you previously loaded an
            image into ``dst`` before calling this function, you must now pass
            that image via ``background=`` instead.

        Args:
            decoder: Decoder instance for interpreting model outputs.
            model_output: List of HAL Tensor objects from model inference.
            dst: Output image tensor (always fully written by this call). Must
                be ``RGBA`` or ``RGB`` for the CPU backend, or
                ``RGBA``/``BGRA``/``RGB`` for OpenGL backend.
            tracker: Optional ByteTrack tracker for object tracking.
            timestamp: Optional frame timestamp in nanoseconds. Only used
                when ``tracker`` is provided. Defaults to current system time.
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
            RuntimeError: If ``dst`` format is unsupported by the active backend,
                or if ``background`` aliases ``dst``;
                or if the ImageProcessor uses G2D (mask rendering not supported).
        """
        ...

    def set_class_colors(self, colors: List[List[int]]) -> None:
        """
        Sets the colors used for rendering boxes and masks. The first 20 colors
        will be set. Each color should be a list of 4 values (0-255 inclusive) representing RGBA.
        """
        ...

    def convert(
        self,
        src: Tensor,
        dst: Tensor,
        rotation: Rotation = Rotation.Rotate0,
        flip: Flip = Flip.NoFlip,
        source: Region | None = None,
        letterbox: List[np.uint8] | None = None,
    ) -> None:
        """
        Convert the source image to the destination image format, with optional
        rotation, flipping, and source crop.

        ``source`` selects a sub-rectangle of the input to sample (whole image if
        ``None``). The destination shape is the placement — use ``dst.view(region)``
        / ``dst.batch(n)`` to render into a sub-region. ``letterbox`` is a resize
        mode: preserve the source aspect ratio and pad the remainder with the given
        RGBA colour (stretch-to-fill when ``None``).
        """
        ...

    def convert_deferred(
        self,
        src: Tensor,
        dst: Tensor,
        rotation: Rotation = Rotation.Rotate0,
        flip: Flip = Flip.NoFlip,
        source: Region | None = None,
        letterbox: List[np.uint8] | None = None,
    ) -> None:
        """
        Convert without waiting for the GPU — the batch-preprocessing primitive.

        Same as ``convert`` but the OpenGL backend skips the per-call ``glFinish``.
        Render N model inputs by looping this over ``dst.batch(n)`` / ``dst.view(region)``
        row-bands of one batched destination, then call ``flush()`` once: the
        backend imports the destination a single time and renders each tile as a
        ``glViewport`` band, syncing once at flush. A deferred destination is not
        safe to read (or ``cuda_map``) until ``flush`` returns. Non-GL backends
        complete synchronously and ``flush`` is a no-op.
        """
        ...

    def flush(self) -> None:
        """
        Complete all deferred converts since the last flush with a single GPU sync.

        After this returns, every destination written by ``convert_deferred`` is
        finished and safe to read back or ``cuda_map``. Non-GL backends return
        immediately.
        """
        ...

    def create_image(
        self,
        width: int,
        height: int,
        format: PixelFormat = PixelFormat.Rgba,
        dtype: str = "uint8",
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

        Returns:
            A new image ``Tensor`` backed by the optimal memory type.
        """
        ...

    if sys.platform == "linux":
        def import_image(
            self,
            fd: int,
            width: int,
            height: int,
            format: PixelFormat,
            dtype: str = "uint8",
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
            ...

    def set_int8_interpolation(self, mode: str) -> None:
        """Sets the interpolation mode for int8 proto textures.

        Accepts "nearest", "bilinear", or "twopass". Default is "bilinear".
        Only affects rendering of quantized (int8) proto segmentation masks.

        Args:
            mode: Interpolation mode string.
        """

class TrackInfo:
    def __init__(
        self,
        uuid: str,
        tracked_location: Tuple[float, float, float, float],
        count: int,
        created: int,
        last_updated: int,
    ) -> None:
        """Information about a single tracked object."""
        ...

    @property
    def uuid(self) -> str:
        """The unique identifier for the tracked object."""
        ...

    @property
    def tracked_location(self) -> Tuple[float, float, float, float]:
        """The current bounding box of the tracked object in (x1, y1, x2, y2) format."""
        ...

    @property
    def count(self) -> int:
        """The number of consecutive frames the object has been tracked for."""
        ...

    @property
    def created(self) -> int:
        """The timestamp (in nanoseconds) when the track was created."""
        ...

    @property
    def last_updated(self) -> int:
        """The timestamp (in nanoseconds) when the track was last updated."""
        ...

class ActiveTrackInfo:
    def __init__(
        self,
        info: TrackInfo,
        bbox: Tuple[float, float, float, float],
        score: float,
        label: int,
    ) -> None:
        """Information about an actively tracked object."""
        ...

    @property
    def info(self) -> TrackInfo:
        """The information about the tracked object."""
        ...

    @property
    def last_box(self) -> Tuple[Tuple[float, float, float, float], float, int]:
        """The last bounding box, score, and class ID of the tracked object."""
        ...

class ByteTrack:
    def __init__(
        self, high_conf=0.7, iou=0.25, update=0.25, lifespan_ns=500_000_000
    ) -> None:
        """Create a new ByteTrack tracker."""
        ...

    def update(
        self,
        boxes: npt.NDArray[np.float32],
        scores: npt.NDArray[np.float32],
        labels: npt.NDArray[np.uintp],
        timestamp_ns: int,
    ) -> List[Optional[TrackInfo]]:
        """
        Update the tracker with new detections for the current timestamp.
        Returns a list of tracks corresponding to the input after the update.
        """
        ...

    def get_active_tracks(self) -> List[ActiveTrackInfo]:
        """
        Get the list of currently active tracks.
        """
        ...

class Tracing:
    """Trace capture context manager for Perfetto/Chrome JSON output.

    Records internal HAL tracing spans (decode sub-steps, mask materialization,
    proto extraction, etc.) to a Chrome JSON file viewable at
    https://ui.perfetto.dev/.

    Only one trace session per process lifetime is supported.  The tracing
    spans are always compiled into the library but have near-zero overhead
    (a single atomic load) until a session is started via this API.

    The ``tracing`` feature is enabled by default in all builds.  It can be
    removed with ``--no-default-features`` if the capture infrastructure is
    not needed (span sites remain compiled at near-zero overhead but cannot
    be activated for capture).

    Usage as context manager (recommended):

    .. code-block:: python

        import edgefirst_hal as hal

        with hal.Tracing("/tmp/trace.json"):
            # ... inference pipeline ...
            pass
        # trace file is flushed and closed on __exit__

    Usage with explicit start/stop:

    .. code-block:: python

        guard = hal.Tracing("/tmp/trace.json")
        guard.start()
        # ... inference pipeline ...
        guard.stop()  # flushes trace file

    The resulting JSON file can be dragged into https://ui.perfetto.dev/ to
    visualize the timeline of decode and mask operations with per-span metadata
    (detection counts, proto dimensions, layout, etc.).
    """

    def __init__(self, path: str) -> None:
        """Create a tracing session targeting the given output file path.

        Args:
            path: File path for the Chrome JSON trace output. The file is
                created on :meth:`start` (or ``__enter__``).
        """
        ...

    def start(self) -> None:
        """Start trace capture.

        Installs a process-wide tracing subscriber and begins recording
        spans to the configured file.

        Only one trace session per process lifetime is supported. Once
        started and stopped, subsequent calls will raise RuntimeError.

        Raises:
            RuntimeError: If a trace session is already active, was
                previously started and stopped (only one session per
                process lifetime), or if tracing support was not compiled in.
        """
        ...

    def stop(self) -> None:
        """Stop trace capture and flush the trace file.

        After this call the trace file is complete and ready for viewing.
        No-op if not currently active.
        """
        ...

    def __enter__(self) -> "Tracing": ...
    def __exit__(self, *args: object) -> bool: ...
