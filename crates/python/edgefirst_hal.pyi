import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, List
import enum

"""EdgeFirst HAL Python bindings."""
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
- masks: A list of NumPy arrays, each of shape (H, W, ...) containing detected segmentation mask.
"""


class Decoder:
    # [pyo3(signature = (config, score_threshold=0.1, iou_threshold=0.7))]
    def __init__(
        self, config: dict, score_threshold: float = 0.1, iou_threshold: float = 0.7
    ) -> None:
        """
        Create a new Decoder instance from a dictionary configuration describing the model outputs.
        """
        ...

    # [pyo3(signature = (json_str, score_threshold=0.1, iou_threshold=0.7))]
    @staticmethod
    def new_from_json_str(
        json_str: str, score_threshold: float = 0.1, iou_threshold: float = 0.7
    ) -> Decoder:
        """
        Create a new Decoder instance from a JSON configuration string describing the model outputs.
        """
        ...

    # [pyo3(signature = (yaml_str, score_threshold=0.1, iou_threshold=0.7))]
    @staticmethod
    def new_from_yaml_str(
        yaml_str: str, score_threshold: float = 0.1, iou_threshold=0.7
    ) -> Decoder:
        """
        Create a new Decoder instance from a YAML configuration string describing the model outputs.
        """
        ...

    # [pyo3(signature = (model_output, max_boxes=100))]
    def decode(self, model_output: List[np.ndarray], max_boxes=100) -> SegDetOutput:
        """
        Decode model outputs into detection and segmentation results. When giving quantized
        tensors as input, the quantization parameters must be specified in the Decoder configuration.

        The accepted integer types are `np.uint8`, `np.int8`, `np.uint16`, `np.int16`, `np.uint32`, and `np.int32`.
        Integer types can be mixed and matched across the different model outputs.

        The accepted floating point types are `np.float16`, `np.float32` and `np.float64`. All outputs must be
        the same floating point type.
        """
        ...

    # [pyo3(signature = (boxes, quant_boxes=(1.0, 0), score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]

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
        max_boxes: int = 100,
    ) -> DetectionOutput:
        """
        Decode YOLO outputs into detection results. When giving float tensors as input, the quantization
        parameters will be ignored.

        The accepted types are `np.uint8`, `np.int8`, `np.float32` and `np.float64`. All outputs must be
        the same type.
        """
        ...

    # [pyo3(signature = (boxes, protos, quant_boxes=(1.0, 0), quant_protos=(1.0, 0), score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
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
        max_boxes: int = 100,
    ) -> SegDetOutput:
        """
        Decode YOLO outputs into detection segmentation results. When giving float tensors as input, the quantization
        parameters will be ignored.

        The accepted types are `np.uint8`, `np.int8`, `np.float32` and `np.float64`. All outputs must be
        the same type.
        """
        ...

    # [pyo3(signature = (boxes, scores, quant_boxes=(1.0, 0), quant_scores=(1.0, 0), score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
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

    # [pyo3(signature = (boxes, anchors, quant=Vec::new(), score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
    @staticmethod
    def decode_modelpack_det_split(
        boxes: List[
            Union[npt.NDArray[np.uint8],
                  npt.NDArray[np.int8], npt.NDArray[np.float32]]
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

    # [pyo3(signature = (quantized, quant_boxes, dequant_into))]
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

    # [pyo3(signature = (segmentation))]
    @staticmethod
    def segmentation_to_mask(segmentation: npt.NDArray[np.uint8]) -> None:
        """Converts a 3D segmentation tensor into a 2D mask."""
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


class FourCC(enum.Enum):
    YUYV: FourCC
    """YUYV format (YUV 4:2:2)"""

    RGBA: FourCC
    """RGBA format (Red, Green, Blue, Alpha)"""

    RGB: FourCC
    """RGB format (Red, Green, Blue)"""

    NV12: FourCC
    """NV12 format (YUV 4:2:0)"""

    NV16: FourCC
    """NV16 format (YUV 4:2:2)"""

    GREY: FourCC
    """Greyscale format"""

    PLANAR_RGB: FourCC
    """Planar RGB format (Red plane, Green plane, Blue plane)"""

    PLANAR_RGBA: FourCC
    """Planar RGBA format (Red plane, Green plane, Blue plane, Alpha plane)"""

    def __init__(self, fourcc: str) -> None: ...


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


class TensorMap:
    def unmap(self) -> None: ...
    def view(self) -> object: ...
    def __repr__(self) -> str: ...
    def __len__(self) -> int: ...
    def __getitem__(self) -> object: ...
    def __setitem__(self, index: int, value: object) -> None: ...
    def __getbuffer__(self, view, _flags) -> None: ...
    def __releasebuffer__(self, view) -> None: ...
    def __enter__(self) -> TensorMap: ...
    def __exit__(self) -> None: ...


class TensorMemory(enum.Enum):
    import sys

    if sys.platform == "linux":
        DMA: TensorMemory
        """
        Direct Memory Access (DMA) allocation. Incurs additional overhead for memory reading/writing with the CPU. 
        Allows for hardware acceleration when suppported
        """

        SHM: TensorMemory
        """
        POSIX Shared Memory allocation. Suitable for inter-process
        communication, but not suitable for hardware acceleration.
        """
    MEM: TensorMemory
    """Regular system memory allocation"""


class TensorImage:
    # [pyo3(signature = (width, height, fourcc = FourCC::RGB))]
    def __init__(
        self,
        width: int,
        height: int,
        fourcc: FourCC = FourCC.RGBA,
        mem: None | TensorMemory = None,
    ) -> None:
        """
        Create a new TensorImage with the specified width, height, and pixel format.
        The optional `mem` parameter can be used to specify the type of memory allocation for the image.
        """
        ...

    # [pyo3(signature = (data, fourcc = None))]
    @staticmethod
    def load_from_bytes(
        data: bytes,
        fourcc: None | FourCC = FourCC.RGBA,
        mem: None | TensorMemory = None,
    ) -> TensorImage:
        """
        Load a JPEG or PNG image from a bytes object.
        The `fourcc` parameter can be used to specify the destination pixel format of the image data.
        The optional `mem` parameter can be used to specify the type of memory allocation for the image.
        """
        ...

    # [pyo3(signature = (filename, fourcc = None))]
    @staticmethod
    def load(
        filename: str,
        fourcc: None | FourCC = FourCC.RGBA,
        mem: None | TensorMemory = None,
    ) -> TensorImage:
        """
        Load a JPEG or PNG image from disk. The `fourcc` parameter can be used to specify the destination pixel format of the image data.
        The optional `mem` parameter can be used to specify the type of memory allocation for the image.
        """
        ...

    # [pyo3(signature = (filename, quality=80))]
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
    def format(self) -> FourCC:
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


class ImageProcessor:
    """Convert images between different formats, with optional rotation, flipping, and cropping."""

    def __init__(self) -> None: ...

    # [pyo3(signature = (dst, bbox, scores, classes, seg=vec![]))]
    def render_to_image(
            self,
            dst: TensorImage,
            bbox: npt.NDArray[np.float32],
            scores: npt.NDArray[np.float32],
            classes: npt.NDArray[np.uintp],
            seg: List[npt.NDArray[np.uint8]] = [],
    ) -> None:
        """
        Render detection and segmentation results onto the destination image.
        The `bbox`, `scores`, and `classes` parameters should correspond to the outputs of a detection model.
        The optional `seg` parameter can be used to provide segmentation masks to render.
        """
        ...

    # [pyo3(signature = (src, dst, rotation = PyRotation::Rotate0, flip = PyFlip::NoFlip, src_crop = None, dst_crop = None, dst_color = None))]

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
