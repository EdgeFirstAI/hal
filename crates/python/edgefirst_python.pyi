import numpy as np
import numpy.typing as npt
from typing import Union, Tuple, List
import enum


DetectionOutput = Tuple[npt.NDArray[np.float32],
                        npt.NDArray[np.float32], npt.NDArray[np.uintp]]
SegDetOutput = Tuple[npt.NDArray[np.float32],
                     npt.NDArray[np.float32], npt.NDArray[np.uintp], List[npt.NDArray[np.uint8]]]


class Decoder:
    # [pyo3(signature = (config, score_threshold=0.1, iou_threshold=0.7))]
    def __init__(
        self,
        config: dict,
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7
    ) -> None:
        ...

    # [pyo3(signature = (json_str, score_threshold=0.1, iou_threshold=0.7))]
    @staticmethod
    def new_from_json_str(
        json_str: str,
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7
    ) -> Decoder:
        ...

    # [pyo3(signature = (yaml_str, score_threshold=0.1, iou_threshold=0.7))]
    @staticmethod
    def new_from_yaml_str(
        yaml_str: str,
        score_threshold: float = 0.1,
        iou_threshold=0.7
    ) -> Decoder:
        ...

    # [pyo3(signature = (model_output, max_boxes=100))]
    def decode(
        self,
        model_output: List[np.ndarray],
        max_boxes=100
    ) -> SegDetOutput:
        ...

    # [pyo3(signature = (boxes, quant_boxes=(1.0, 0), score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
    @staticmethod
    def decode_yolo(
        model_output: Union[npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32], npt.NDArray[np.float64]],
        quant_boxes: Tuple[float, int] = (1.0, 0),
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        max_boxes: int = 100
    ) -> DetectionOutput:
        ...

    # [pyo3(signature = (boxes, protos, quant_boxes=(1.0, 0), quant_protos=(1.0, 0), score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
    @staticmethod
    def decode_yolo_segdet(
        boxes: Union[npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32]],
        protos: Union[npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32]],
        quant_boxes: Tuple[float, int] = (1.0, 0),
        quant_protos: Tuple[float, int] = (1.0, 0),
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        max_boxes: int = 100
    ) -> SegDetOutput:
        ...

    # [pyo3(signature = (boxes, scores, quant_boxes=(1.0, 0), quant_scores=(1.0, 0), score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
    @staticmethod
    def decode_modelpack_det(
        boxes: Union[npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32]],
        scores: Union[npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32]],
        quant_boxes: Tuple[float, int] = (1.0, 0),
        quant_scores: Tuple[float, int] = (1.0, 0),
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        max_boxes: int = 100
    ) -> DetectionOutput:
        ...

    # [pyo3(signature = (boxes, anchors, quant=Vec::new(), score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
    @staticmethod
    def decode_modelpack_det_split(
        boxes: List[Union[npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32]]],
        anchors: List[List[List[float]]],
        quant: List[Tuple[float, int]] = [],
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        max_boxes: int = 100
    ) -> DetectionOutput:
        ...

    # [pyo3(signature = (quantized, quant_boxes, dequant_into))]
    @staticmethod
    def dequantize(
        quantized: Union[npt.NDArray[np.uint8], npt.NDArray[np.int8]],
        quant_boxes: Tuple[float, int],
        dequant_into: Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]
    ) -> None:
        ...

    # [pyo3(signature = (segmentation))]
    @staticmethod
    def segmentation_to_mask(segmentation: npt.NDArray[np.uint8]) -> None:
        ...

    @property
    def score_threshold(self) -> float:
        ...

    @score_threshold.setter
    def score_threshold(self, value: float):
        ...

    @property
    def iou_threshold(self) -> float:
        ...

    @iou_threshold.setter
    def iou_threshold(self, value: float):
        ...


class FourCC(enum.Enum):
    YUYV: FourCC
    RGBA: FourCC
    RGB: FourCC
    NV12: FourCC
    GREY: FourCC

    def __init__(self, fourcc: str) -> None:
        ...


class TensorMemory(enum.Enum):
    import sys
    if sys.platform == 'linux':
        DMA: TensorMemory
        SHM: TensorMemory
    MEM: TensorMemory


class TensorImage:
    # [pyo3(signature = (width, height, fourcc = FourCC::RGB))]
    def __init__(self, width: int, height: int, fourcc: FourCC = FourCC.RGB, mem: Union[None, TensorMemory] = None) -> None:
        ...

    # [pyo3(signature = (data, fourcc = None))]
    @staticmethod
    def load_from_bytes(data: bytes, fourcc: Union[None, FourCC] = FourCC.RGB, mem: Union[None, TensorMemory] = None) -> TensorImage:
        ...

    # [pyo3(signature = (filename, fourcc = None))]
    @staticmethod
    def load(filename: str, fourcc: Union[None, FourCC] = FourCC.RGB, mem: Union[None, TensorMemory] = None) -> TensorImage:
        ...

    # [pyo3(signature = (filename, quality=80))]
    def save_jpeg(self, filename:  str, quality: int = 80) -> None:
        ...

    def to_numpy(self) -> npt.NDArray[np.uint8]:
        ...

    def copy_into_numpy(self, dst: Union[npt.NDArray[np.uint8], npt.NDArray[np.int8]]) -> None:
        ...

    def copy_from_numpy(self, src: npt.NDArray[np.uint8]) -> None:
        ...

    @property
    def format(self) -> FourCC:
        ...

    @property
    def width(self) -> int:
        ...

    @property
    def height(self) -> int:
        ...

    @property
    def is_planar(self) -> bool:
        ...


class Flip(enum.Enum):
    NoFlip: Flip
    Horizontal: Flip
    Vertical: Flip


class Rotation(enum.Enum):
    Rotate0: Rotation
    Clockwise90: Rotation
    Rotate180: Rotation
    CounterClockwise90: Rotation

    @staticmethod
    def degrees_clockwise(angle: int) -> Rotation:
        ...


class Rect:

    def __init__(self, left: int, top: int, width: int, height: int):
        ...

    @property
    def left(self) -> int:
        ...

    @property
    def top(self) -> int:
        ...

    @property
    def width(self) -> int:
        ...

    @property
    def height(self) -> int:
        ...


class ImageConverter:
    def __init__(self) -> None:
        ...

    # [pyo3(signature = (src, dst, rotation = Rotation::Rotate0, flip = Flip::NoFlip, src_crop = None, dst_crop = None))]
    def convert(self, src: TensorImage, dst: TensorImage, rotation: Rotation = Rotation.Rotate0, flip: Flip = Flip.NoFlip, src_crop: Rect | None = None, dst_crop: Rect | None = None) -> None:
        ...
