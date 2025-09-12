from enum import Enum
import numpy as np
import numpy.typing as npt
from typing import Union


class BBoxType:
    XYXY: BBoxType
    XYWH: BBoxType

    @property
    def value(self) -> int: ...


class Decoder:
    # [pyo3(signature = (json_str, score_threshold=0.1, iou_threshold=0.7))]
    @staticmethod
    def new_from_json_str(json_str: str, score_threshold: float = 0.1, iou_threshold=0.7) -> Decoder:
        ...

    # [pyo3(signature = (yaml_str, score_threshold=0.1, iou_threshold=0.7))]
    @staticmethod
    def new_from_yaml_str(yaml_str: str, score_threshold: float = 0.1, iou_threshold=0.7) -> Decoder:
        ...

    # [pyo3(signature = (model_output, max_boxes=100))]
    def decode(model_output: list[np.ndarray], max_boxes=100):
        ...

    # [pyo3(signature = (output, num_classes, scale=1.0, zero_point=0, score_threshold=0.1, iou_threshold=0.7, max_boxes=100, bbox_type=PyBBoxType::Xywh))]

    @staticmethod
    def decode_yolo(model_output: Union[npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32], npt.NDArray[np.float64]], num_classes: int, scale: float = 1.0, zero_point: int = 0, score_threshold: float = 0.1, iou_threshold: float = 0.7, max_boxes: int = 100, bbox_type=BBoxType.XYXY) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.uintp]]:
        ...

    # [pyo3(signature = (output, num_classes, scale, zero_point, score_threshold=0.1, iou_threshold=0.7, max_boxes=100, bbox_type=PyBBoxType::Xywh))]
    @staticmethod
    def decode_yolo_i8(model_output: npt.NDArray[np.int8], num_classes: int, scale: float, zero_point: int, score_threshold: float = 0.1, iou_threshold: float = 0.7, max_boxes: int = 100, bbox_type=BBoxType.XYXY) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.uintp]]:
        ...

    # [pyo3(signature = (output, num_classes, scale, zero_point, score_threshold=0.1, iou_threshold=0.7, max_boxes=100, bbox_type=PyBBoxType::Xywh))]
    @staticmethod
    def decode_yolo_u8(model_output: npt.NDArray[np.uint8], num_classes: int, scale: float, zero_point: int, score_threshold: float = 0.1, iou_threshold: float = 0.7, max_boxes: int = 100, bbox_type=BBoxType.XYXY) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.uintp]]:
        ...

    # [pyo3(signature = (output, num_classes, score_threshold=0.1, iou_threshold=0.7, max_boxes=100, bbox_type=PyBBoxType::Xywh))]
    @staticmethod
    def decode_yolo_f32(model_output: npt.NDArray[np.float32], num_classes: int, score_threshold: float = 0.1, iou_threshold: float = 0.7, max_boxes: int = 100, bbox_type=BBoxType.XYXY) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.uintp]]:
        ...

    # [pyo3(signature = (output, num_classes, score_threshold=0.1, iou_threshold=0.7, max_boxes=100, bbox_type=PyBBoxType::Xywh))]
    @staticmethod
    def decode_yolo_f64(model_output: npt.NDArray[np.float64], num_classes: int, score_threshold: float = 0.1, iou_threshold: float = 0.7, max_boxes: int = 100, bbox_type=BBoxType.XYXY) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.uintp]]:
        ...

    @staticmethod
    def dequantize(quantized: Union[npt.NDArray[np.uint8], npt.NDArray[np.int8]], scale: float, zero_point: int, dequant_into: Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]) -> None:
        ...
