"""Type stubs for ``edgefirst.kernels``.

Compute-kernel plans over numpy arrays. Build a plan once, then call ``run``
with any strided arrays (negative strides are rejected). ``run`` releases the
GIL. A quantization of ``None`` means identity (scale 1, zero point 0).

Float tensors ignore their quantization in every plan: a ``float16`` or
``float32`` input or output may pass any quant, ``(0.0, 0)`` included.

``run`` raises ``ValueError`` when the input and output overlap in memory,
including interleaved views of one array (``a[..., ::2]`` and ``a[..., 1::2]``):
each view's memory span runs from its first element to one past its last.
"""

from collections.abc import Sequence
from typing import Literal

import numpy
import numpy.typing as npt
from typing_extensions import TypeAlias

DTypeLike = npt.DTypeLike
Quant: TypeAlias = tuple[float, int] | None
Level: TypeAlias = tuple[int, int, float]
BoxFormat: TypeAlias = Literal["xywh", "xyxy"]

def version() -> str:
    """Return the HAL version string (matches ``Cargo.toml``)."""

class SoftmaxI8Plan:
    """int8 softmax over the last axis, matching TFLite's int8 ``SOFTMAX``.

    The output must be quantized with scale 1/256 and zero point -128, else
    ``NotImplementedError`` is raised.
    """

    def __init__(
        self,
        input_scale: float,
        input_zero_point: int,
        output_scale: float,
        output_zero_point: int,
        beta: float = 1.0,
    ) -> None: ...
    @property
    def tier(self) -> str:
        """Kernel tier in use: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``."""
    def run(
        self,
        input: npt.NDArray[numpy.int8],
        output: npt.NDArray[numpy.int8],
        *,
        threads: int | None = None,
    ) -> None:
        """Softmax over the last axis; ``output`` has ``input``'s shape.

        ``threads`` of ``None`` or 1 runs on the calling thread; larger values
        split the work across that many threads.
        """

class SoftmaxI16Plan:
    """int16 softmax over the last axis, matching TFLite's int16 ``SOFTMAX``."""

    def __init__(
        self,
        input_scale: float,
        input_zero_point: int,
        output_scale: float,
        output_zero_point: int,
        beta: float = 1.0,
    ) -> None: ...
    @property
    def tier(self) -> str:
        """Kernel tier in use: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``."""
    def run(
        self,
        input: npt.NDArray[numpy.int16],
        output: npt.NDArray[numpy.int16],
        *,
        threads: int | None = None,
    ) -> None:
        """Softmax over the last axis; ``output`` has ``input``'s shape."""

class SigmoidPlan:
    """Elementwise sigmoid for int8, uint8, int16, uint16, float16 and float32."""

    def __init__(
        self,
        input_dtype: DTypeLike,
        input_quant: Quant,
        output_dtype: DTypeLike,
        output_quant: Quant,
    ) -> None: ...
    @property
    def tier(self) -> str:
        """Kernel tier in use: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``."""
    def run(
        self,
        input: npt.NDArray[numpy.generic],
        output: npt.NDArray[numpy.generic],
        *,
        threads: int | None = None,
    ) -> None:
        """Apply sigmoid elementwise; ``output`` has ``input``'s shape."""

class DflPlan:
    """Distribution focal loss: ``[anchors, 4 * reg_max]`` logits to ``[anchors, 4]`` LTRB distances."""

    def __init__(
        self,
        input_dtype: DTypeLike,
        input_quant: Quant,
        output_dtype: DTypeLike,
        output_quant: Quant,
        reg_max: int,
    ) -> None: ...
    @property
    def tier(self) -> str:
        """Kernel tier in use: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``."""
    def run(
        self,
        input: npt.NDArray[numpy.generic],
        output: npt.NDArray[numpy.generic],
        *,
        threads: int | None = None,
    ) -> None:
        """Decode logits into distances in grid units."""

class Dist2BboxPlan:
    """LTRB grid-unit distances to boxes in pixels over a multi-level anchor axis.

    ``levels`` holds ``(anchors, grid_w, stride)`` per FPN level, in anchor-axis order.
    """

    def __init__(
        self,
        input_dtype: DTypeLike,
        input_quant: Quant,
        output_dtype: DTypeLike,
        output_quant: Quant,
        levels: Sequence[Level],
        box_format: BoxFormat = "xywh",
    ) -> None: ...
    @property
    def tier(self) -> str:
        """Kernel tier in use: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``."""
    def run(
        self,
        input: npt.NDArray[numpy.generic],
        output: npt.NDArray[numpy.generic],
        *,
        threads: int | None = None,
    ) -> None:
        """Decode ``[anchors, 4]`` distances into ``[anchors, 4]`` boxes in pixels."""

class DflBoxPlan:
    """Fused DFL and dist2bbox: ``[anchors, 4 * reg_max]`` logits to ``[anchors, 4]`` boxes in pixels."""

    def __init__(
        self,
        input_dtype: DTypeLike,
        input_quant: Quant,
        output_dtype: DTypeLike,
        output_quant: Quant,
        reg_max: int,
        levels: Sequence[Level],
        box_format: BoxFormat = "xywh",
    ) -> None: ...
    @property
    def tier(self) -> str:
        """Kernel tier in use: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``."""
    def run(
        self,
        input: npt.NDArray[numpy.generic],
        output: npt.NDArray[numpy.generic],
        *,
        threads: int | None = None,
    ) -> None:
        """Decode logits into boxes in pixels."""

class LutI8Plan:
    """256-entry lookup table over int8 or uint8 data."""

    @staticmethod
    def from_table(dtype: DTypeLike, table: npt.NDArray[numpy.uint8]) -> LutI8Plan:
        """A plan applying ``table`` (256 uint8 entries, indexed by the input's raw byte)."""
    @staticmethod
    def sigmoid(
        dtype: DTypeLike,
        input_quant: tuple[float, int],
        output_quant: tuple[float, int],
    ) -> LutI8Plan:
        """The TFLite int8/uint8 ``LOGISTIC`` table for the given quantization."""
    @property
    def tier(self) -> str:
        """Kernel tier in use: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``."""
    def run(
        self,
        input: npt.NDArray[numpy.generic],
        output: npt.NDArray[numpy.generic],
        *,
        threads: int | None = None,
    ) -> None:
        """Map ``input`` through the table into ``output`` (same shape and dtype)."""
