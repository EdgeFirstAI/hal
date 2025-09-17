import numpy as np
import numpy.typing as npt
from typing import Union, Tuple


DetectionOutput = Tuple[npt.NDArray[np.float32],
                        npt.NDArray[np.float32], npt.NDArray[np.uintp]]
SegDetOutput = Tuple[npt.NDArray[np.float32],
                     npt.NDArray[np.float32], npt.NDArray[np.uintp], list[npt.NDArray[np.uint8]]]


class Decoder:
    # [pyo3(signature = (json_str, score_threshold=0.1, iou_threshold=0.7))]
    @staticmethod
    def new_from_json_str(
        json_str: str,
        score_threshold: float = 0.1,
        iou_threshold=0.7
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
        model_output: list[np.ndarray],
        max_boxes=100
    ) -> SegDetOutput:
        ...

    # [pyo3(signature = (boxes, quant_boxes=(1.0, 0), score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
    @staticmethod
    def decode_yolo(
        model_output: Union[npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32], npt.NDArray[np.float64]],
        quant_boxes: tuple[float, int] = (1.0, 0),
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        max_boxes: int = 100
    ) -> DetectionOutput:
        ...

    # [pyo3(signature = (boxes, protos, quant_boxes=(1.0, 0), quant_protos=(1.0, 0), score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]

    def decode_yolo_segdet(
        boxes: Union[npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32]],
        protos: Union[npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32]],
        quant_boxes: tuple[float, int] = (1.0, 0),
        quant_protos: tuple[float, int] = (1.0, 0),
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
        quant_boxes: tuple[float, int] = (1.0, 0),
        quant_scores: tuple[float, int] = (1.0, 0),
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        max_boxes: int = 100
    ) -> DetectionOutput:
        ...

    # [pyo3(signature = (boxes, anchors, quant=Vec::new(), score_threshold=0.1, iou_threshold=0.7, max_boxes=100))]
    @staticmethod
    def decode_modelpack_det_split(
        boxes: list[Union[npt.NDArray[np.uint8], npt.NDArray[np.int8], npt.NDArray[np.float32]]],
        anchors: list[list[list[int]]],
        quant: list[tuple[float, int]] = [],
        score_threshold: float = 0.1,
        iou_threshold: float = 0.7,
        max_boxes: int = 100
    ) -> DetectionOutput:
        ...

    # [pyo3(signature = (quantized, quant_boxes, dequant_into))]
    @staticmethod
    def dequantize(
        quantized: Union[npt.NDArray[np.uint8], npt.NDArray[np.int8]],
        quant_boxes: tuple[float, int],
        dequant_into: Union[npt.NDArray[np.float32], npt.NDArray[np.float64]]
    ) -> None:
        ...

    # [pyo3(signature = (segmentation))]
    @staticmethod
    def segmentation_to_mask(segmentation: npt.NDArray[np.uint8]) -> None:
        ...
