import numpy as np
import numpy.typing as npt
from typing import Literal, Optional, Union, Tuple, List
import enum
import sys

"""EdgeFirst HAL Python bindings."""

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
  - ``decode_masks()``: shape ``(H, W)`` at the requested output resolution.
    Binary ``uint8`` where 255 = mask presence.
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

    def decode(self, model_output: List[np.ndarray], max_boxes=100) -> SegDetOutput:
        """
        Decode model outputs into detection and segmentation results. When giving quantized
        tensors as input, the quantization parameters must be specified in the Decoder configuration.

        The accepted integer types are `np.uint8`, `np.int8`, `np.uint16`, `np.int16`, `np.uint32`, and `np.int32`.
        Integer types can be mixed and matched across the different model outputs.

        The accepted floating point types are `np.float16`, `np.float32` and `np.float64`. All outputs must be
        the same floating point type.

        Masks are returned at prototype resolution as 3D arrays of shape
        ``(H, W, C)``. For instance segmentation models (e.g. YOLO) ``C=1``
        — a binary per-instance mask (threshold at 128). For semantic
        segmentation models (e.g. ModelPack) ``C=num_classes`` — per-pixel
        class scores (use ``argmax`` over the last axis). Use
        ``decode_masks()`` to get upsampled 2D binary masks.
        """
        ...

    def draw_masks(
        self,
        model_output: List[np.ndarray],
        processor: ImageProcessor,
        dst: TensorImage,
        max_boxes: int = 100,
    ) -> DetectionOutput:
        """
        Decode model outputs and draw colored masks directly onto the
        destination image in a single fused call. This is the fastest path
        for visualization — masks never leave Rust/GPU, eliminating the
        Python round-trip overhead of ``decode()`` + ``processor.draw_masks()``.

        For segmentation models, prototype data is passed directly to the
        renderer which evaluates the mask at every output pixel. For
        detection-only models, this falls back to the standard drawing path.

        Returns ``(boxes, scores, classes)`` — no mask arrays are returned.
        Use ``decode_masks()`` if you need the raw mask pixels.

        Args:
            model_output: List of model output tensors (same types as ``decode``).
            processor: ImageProcessor instance for drawing.
            dst: Destination TensorImage to draw onto. Must be ``RGBA`` or
                ``RGB`` for CPU backend, or ``RGBA``/``BGRA``/``RGB`` for
                OpenGL backend.
            max_boxes: Maximum number of detections to return (default: 100).

        Raises:
            RuntimeError: If ``dst`` format is unsupported by the active backend,
                or if the ImageProcessor uses G2D (mask rendering not supported).
        """
        ...

    def decode_masks(
        self,
        model_output: List[np.ndarray],
        processor: ImageProcessor,
        output_width: int = 640,
        output_height: int = 640,
        max_boxes: int = 100,
    ) -> SegDetOutput:
        """
        Decode model outputs and return per-detection binary masks.

        Internally uses GPU atlas rendering when available (OpenGL), then
        splits the atlas into individual per-detection mask arrays. Each mask
        is a binary ``uint8`` array of shape ``(bbox_h, bbox_w)`` covering
        the detection's bounding box region, where ``255`` = mask presence
        and ``0`` = background. Threshold at ``> 127`` for boolean masks.

        Use this method when you need the raw mask pixels for downstream
        processing (tracking, area measurement, custom compositing). For
        direct overlay onto a display frame, prefer ``draw_masks()`` which
        avoids the Python round-trip.

        Args:
            model_output: List of model output tensors (same types as ``decode``).
            processor: ImageProcessor instance for GPU-accelerated mask rendering.
            output_width: Coordinate-space width for bounding box interpretation
                (default: 640). This is **not** the per-mask array width — each
                returned mask is sized to its detection's bounding box.
            output_height: Coordinate-space height for bounding box interpretation
                (default: 640). Typically matches the model input resolution.
            max_boxes: Maximum number of detections to return (default: 100).

        Returns:
            ``(boxes, scores, classes, masks)`` where masks is a list of
            ``ndarray[bbox_h, bbox_w]`` of ``uint8`` — one per detection.

        Raises:
            RuntimeError: If the ImageProcessor uses G2D (mask rendering not
                supported on G2D). Use an OpenGL or CPU processor instead.
        """
        ...

    @staticmethod
    def decode_yolo_det(
        model_output: Union[
            npt.NDArray[np.uint8],
            npt.NDArray[np.int8],
            npt.NDArray[np.float32],
            npt.NDArray[np.float64],
        ],
        quant_boxes: Tuple[float, int] = (1.0, 0),
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        nms: Optional[Nms] = Nms.ClassAgnostic,
        max_boxes: int = 100,
    ) -> DetectionOutput:
        """
        Decode YOLO outputs into detection results. When giving float tensors as input, the quantization
        parameters will be ignored.

        The accepted types are `np.uint8`, `np.int8`, `np.float32` and `np.float64`. All outputs must be
        the same type.

        Args:
            model_output: YOLO model output tensor.
            quant_boxes: Quantization parameters (scale, zero_point) for boxes.
            score_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            nms: NMS mode - Nms.ClassAgnostic (default), Nms.ClassAware, or None to bypass NMS.
            max_boxes: Maximum number of boxes to return.
        """
        ...

    @staticmethod
    def decode_yolo_segdet(
        boxes: Union[
            npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32]
        ],
        protos: Union[
            npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32]
        ],
        quant_boxes: Tuple[float, int] = (1.0, 0),
        quant_protos: Tuple[float, int] = (1.0, 0),
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        nms: Optional[Nms] = Nms.ClassAgnostic,
        max_boxes: int = 100,
    ) -> SegDetOutput:
        """
        Decode YOLO outputs into detection segmentation results. When giving float tensors as input, the quantization
        parameters will be ignored.

        The accepted types are `np.uint8`, `np.int8`, `np.float32` and `np.float64`. All outputs must be
        the same type.

        Args:
            boxes: YOLO box output tensor.
            protos: YOLO proto output tensor.
            quant_boxes: Quantization parameters (scale, zero_point) for boxes.
            quant_protos: Quantization parameters (scale, zero_point) for protos.
            score_threshold: Minimum confidence score for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            nms: NMS mode - Nms.ClassAgnostic (default), Nms.ClassAware, or None to bypass NMS.
            max_boxes: Maximum number of boxes to return.
        """
        ...

    @staticmethod
    def decode_modelpack_det(
        boxes: Union[
            npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32]
        ],
        scores: Union[
            npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32]
        ],
        quant_boxes: Tuple[float, int] = (1.0, 0),
        quant_scores: Tuple[float, int] = (1.0, 0),
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        max_boxes: int = 100,
    ) -> DetectionOutput:
        """
        Decode ModelPack outputs into detection results. When giving float tensors as input, the quantization
        parameters will be ignored.

        The accepted types are `np.uint8`, `np.int8`, `np.float32` and `np.float64`. All outputs must be
        the same type.
        """
        ...

    @staticmethod
    def decode_modelpack_det_split(
        boxes: List[
            Union[npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32]]
        ],
        anchors: List[List[List[float]]],
        quant: List[Tuple[float, int]] = [],
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        max_boxes: int = 100,
    ) -> DetectionOutput:
        """
        Decode ModelPack outputs into detection results. When giving float tensors as input, the quantization
        parameters will be ignored.

        The accepted types are `np.uint8`, `np.int8`, `np.float32` and `np.float64`. All outputs must be
        the same type.
        """
        ...

    @staticmethod
    def dequantize(
        quantized: Union[
            npt.NDArray[np.uint8],
            npt.NDArray[np.int8],
            npt.NDArray[np.uint16],
            npt.NDArray[np.int16],
            npt.NDArray[np.uint32],
            npt.NDArray[np.int32],
        ],
        quant_boxes: Tuple[float, int],
        dequant_into: Union[npt.NDArray[np.float32], npt.NDArray[np.float64]],
    ) -> None:
        """
        Dequantize a quantized tensor into a floating point tensor.
        The destination tensor must have the same shape as the input tensor.
        """
        ...

    @staticmethod
    def segmentation_to_mask(
        segmentation: npt.NDArray[np.uint8],
    ) -> npt.NDArray[np.uint8]:
        """
        Converts a 3D segmentation tensor into a 2D mask.

        Raises:
            ValueError: If the segmentation tensor has an invalid shape.
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

        This will take ownership of the file descriptor, and the file descriptor will 
        be closed when the tensor is dropped.
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

    def map(self) -> TensorMap: ...
    """Returns a mapped view of the tensor data for direct access."""

class TensorMap:
    def unmap(self) -> None: ...
    def view(self) -> memoryview: ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> object: ...
    def __setitem__(self, index: int, value: object) -> None: ...
    def __getbuffer__(self, view, _flags) -> None: ...
    def __releasebuffer__(self, view) -> None: ...
    def __enter__(self) -> TensorMap: ...
    def __exit__(self, _exc_type, _exc_value, _traceback) -> None: ...

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

class TensorImage:
    def __init__(
        self,
        width: int,
        height: int,
        format: PixelFormat = PixelFormat.Rgba,
        mem: None | TensorMemory = None,
    ) -> None:
        """
        Create a new TensorImage with the specified width, height, and pixel format.
        The optional `mem` parameter can be used to specify the type of memory allocation for the image.
        """
        ...

    if sys.platform == "linux":
        @staticmethod
        def from_fd(
            fd: int,
            shape: list[int],
            format: PixelFormat,
        ) -> TensorImage:
            """
            Load an image from a file descriptor, inspecting the file descriptor to determine
            the appropriate tensor type (DMA or SHM) based on the device major and minor numbers.

            The ``shape`` must match the pixel format. Most formats use a 3D shape
            ``[height, width, channels]`` (interleaved) or ``[channels, height, width]``
            (planar). The semi-planar formats NV12 and NV16 use a 2D shape because
            their Y and UV planes have different heights:

            ==================  ==================  ====================================
            Format               Shape               Description
            ==================  ==================  ====================================
            PixelFormat.Rgb      [H, W, 3]           3-channel interleaved
            PixelFormat.Rgba     [H, W, 4]           4-channel interleaved
            PixelFormat.Grey     [H, W, 1]           Single-channel grayscale
            PixelFormat.Yuyv     [H, W, 2]           YUV 4:2:2 interleaved
            PixelFormat.PlanarRgb   [3, H, W]        Channels-first (3 planes)
            PixelFormat.PlanarRgba  [4, H, W]        Channels-first (4 planes)
            PixelFormat.Nv12     [H * 3 // 2, W]     Semi-planar YUV 4:2:0 (2D)
            PixelFormat.Nv16     [H * 2, W]          Semi-planar YUV 4:2:2 (2D)
            ==================  ==================  ====================================

            For example, a 1080p NV12 frame has 1080 Y rows plus 540 UV rows,
            giving shape ``[1620, 1920]``.

            The ``format`` parameter specifies the pixel format of the image data.

            This will take ownership of the file descriptor, and the file descriptor will
            be closed when the tensor is dropped.
            """
            ...

    @staticmethod
    def load_from_bytes(
        data: bytes,
        format: None | PixelFormat = PixelFormat.Rgba,
        mem: None | TensorMemory = None,
    ) -> TensorImage:
        """
        Load a JPEG or PNG image from a bytes object.
        The `format` parameter can be used to specify the destination pixel format of the image data.
        The optional `mem` parameter can be used to specify the type of memory allocation for the image.
        """
        ...

    @staticmethod
    def load(
        filename: str,
        format: None | PixelFormat = PixelFormat.Rgba,
        mem: None | TensorMemory = None,
    ) -> TensorImage:
        """
        Load a JPEG or PNG image from disk. The `format` parameter can be used to specify the destination pixel format of the image data.
        The optional `mem` parameter can be used to specify the type of memory allocation for the image.
        """
        ...

    def save_jpeg(self, filename: str, quality: int = 80) -> None:
        """Save the image as a JPEG file to disk with the specified quality (1-100). The image must be RGB or RGBA format."""
        ...

    def normalize_to_numpy(
        self,
        dst: npt.NDArray[np.uint8]
        | npt.NDArray[np.int8]
        | npt.NDArray[np.float32]
        | npt.NDArray[np.float64],
        normalization: Normalization = Normalization.DEFAULT,
        zero_point: None | int = None,
    ) -> None:
        """
        Normalize the image data into a NumPy array with the specified data type and normalization method.
        The optional `zero_point` parameter can be used to specify the zero point for signed normalization.
        This will also convert RGBA images to RGB by dropping the alpha channel.
        """
        ...

    def copy_from_numpy(self, src: npt.NDArray[np.uint8]) -> None:
        """Copy data from a NumPy array into the image. The shape and data type of the NumPy array must match the image's format."""
        ...

    def map(self) -> TensorMap:
        """Returns a mapped view of the image data for direct access."""
        ...

    @property
    def format(self) -> PixelFormat:
        """The pixel format of the image."""
        ...

    @property
    def width(self) -> int:
        """The width of the image in pixels."""
        ...

    @property
    def height(self) -> int:
        """The height of the image in pixels."""
        ...

    @property
    def is_planar(self) -> bool:
        """If the image format is planar."""
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

class Rect:
    """A crop rectangle defined by its top-left corner (left, top) and its dimensions (width, height)."""

    def __init__(self, left: int, top: int, width: int, height: int): ...
    @property
    def left(self) -> int: ...
    @property
    def top(self) -> int: ...
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
    def draw_masks(
        self,
        dst: TensorImage,
        bbox: npt.NDArray[np.float32],
        scores: npt.NDArray[np.float32],
        classes: npt.NDArray[np.uintp],
        seg: List[npt.NDArray[np.uint8]] = [],
    ) -> None:
        """
        Draw detection boxes and optional segmentation masks onto ``dst``.

        This method draws pre-decoded results. For the fused decode+draw path
        (recommended for most use cases), use ``Decoder.draw_masks()`` instead.

        Args:
            dst: Destination image. Must be ``RGBA`` or ``RGB`` for CPU backend,
                or ``RGBA``/``BGRA``/``RGB`` for OpenGL.
            bbox: ``(N, 4)`` float32 array of normalized bounding boxes in
                ``[x1, y1, x2, y2]`` format with values in ``[0, 1]``.
            scores: ``(N,)`` float32 array of confidence scores.
            classes: ``(N,)`` uintp array of class indices.
            seg: Optional list of ``uint8`` mask arrays from ``decode()``.
                Each element has shape ``(H, W, C)``. When ``C=1`` (instance
                segmentation, e.g. YOLO), one mask per detection is expected
                (``len(seg) <= len(bbox)``). When ``C > 1`` (semantic
                segmentation, e.g. ModelPack), a single mask covers all classes.

        Raises:
            RuntimeError: If ``dst`` format is unsupported by the active backend.
            ValueError: If ``bbox``, ``scores``, ``classes`` lengths do not match,
                or if ``bbox`` shape is not ``(N, 4)``.
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
        src: TensorImage,
        dst: TensorImage,
        rotation: Rotation = Rotation.Rotate0,
        flip: Flip = Flip.NoFlip,
        src_crop: Rect | None = None,
        dst_crop: Rect | None = None,
        dst_color: List[np.uint8] | None = None,
    ) -> None:
        """
        Convert the source image to the destination image format, with optional rotation, flipping, cropping.
        The fill color can be used for areas outside the destination crop. The fill color is provided as RGBA values.
        """
        ...

    def create_image(
        self,
        width: int,
        height: int,
        format: PixelFormat = PixelFormat.Rgba,
    ) -> TensorImage:
        """Create an image with the processor's optimal memory backend.

        Selects the best available backing storage based on hardware capabilities:
        DMA-buf > PBO (GPU buffer) > system memory. Images created this way benefit
        from zero-copy GPU paths when used with this processor's ``convert()``.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            format: Pixel format (default: ``PixelFormat.Rgba``).

        Returns:
            A new ``TensorImage`` backed by the optimal memory type.
        """
        ...

    def set_int8_interpolation(self, mode: str) -> None:
        """Sets the interpolation mode for int8 proto textures.

        Accepts "nearest", "bilinear", or "twopass". Default is "bilinear".
        Only affects rendering of quantized (int8) proto segmentation masks.

        Args:
            mode: Interpolation mode string.
        """
        ...
