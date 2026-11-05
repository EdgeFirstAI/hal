"""Type stubs for ``edgefirst.kernels``.

Compute-kernel plans over numpy arrays. Build a plan once, then call ``run``
with any strided arrays (negative strides are rejected). ``run`` releases the
GIL. A quantization of ``None`` means identity (scale 1, zero point 0).

Float tensors ignore their quantization in every plan: a ``float16`` or
``float32`` input or output may pass any quant, ``(0.0, 0)`` included.

``run`` raises ``ValueError`` when the output overlaps an input in memory,
including interleaved views of one array (``a[..., ::2]`` and ``a[..., 1::2]``):
each view's memory span runs from its first element to one past its last.
Inputs may share memory with each other.

The int8 TFLite plans take quantizations as ``(scale, zero_point)`` tuples, a
fused ``activation`` and the ``kernels`` whose roundings to reproduce.
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
QuantI8: TypeAlias = tuple[float, int]
Activation: TypeAlias = Literal["none", "relu", "relu_n1_to_1", "relu6"]
TfliteKernels: TypeAlias = Literal["neon_ruy", "neon_gemmlowp", "ruy", "reference"]
"""Which TFLite int8 kernels to reproduce where they round differently.

``neon_ruy``: int8 ``ADD`` and ``MUL`` in whole 16-element NEON blocks and matrix
products through ruy, as every aarch64 build (such as the NXP BSP runtimes) runs
them. ``neon_gemmlowp``: NEON blocks and matrix products through gemmlowp, which
rounds as the reference kernels, as x86 builds with SSE4.1 (such as the LiteRT
wheels) run them. ``ruy``: scalar arithmetic and ruy, for builds without NEON.
``reference``: the reference kernels (``BuiltinRefOpResolver``).
"""

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

class BinaryI8Plan:
    """int8 ``ADD``, ``SUB`` or ``MUL``, matching TFLite's int8 kernels bit for bit.

    ``activation`` is the fused activation, applied as TFLite's output clamp.
    """

    def __init__(
        self,
        op: Literal["add", "sub", "mul"],
        lhs_quant: QuantI8,
        rhs_quant: QuantI8,
        output_quant: QuantI8,
        activation: Activation = "none",
        kernels: TfliteKernels = "neon_ruy",
    ) -> None: ...
    @property
    def tier(self) -> str:
        """Kernel tier in use: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``."""
    def table_for_rhs(self, rhs: int, length: int) -> npt.NDArray[numpy.uint8] | None:
        """The output for every ``lhs`` byte with ``rhs`` fixed, for a call over ``length`` elements.

        Indexed by the ``lhs`` byte as ``LutI8Plan.from_table`` takes it.
        ``None`` when such a call gives some elements TFLite's NEON-block
        arithmetic and others its scalar code, and the two disagree.
        """
    def run(
        self,
        lhs: npt.NDArray[numpy.int8],
        rhs: npt.NDArray[numpy.int8],
        output: npt.NDArray[numpy.int8],
        *,
        threads: int | None = None,
    ) -> None:
        """``output = lhs op rhs`` elementwise; all three have one shape.

        A broadcast input is a zero-stride view (``numpy.broadcast_to``).
        """

class MatMulI8Plan:
    """int8 matrix multiplication over the last two axes with TFLite's requantization.

    ``out[..., m, n] = clamp(zo + R[n](sum_k (lhs - zl)(rhs - zr) + bias[n]))``
    where ``R`` is ``requant[n]`` (or ``requant[0]`` for every column): a
    ``(multiplier, shift)`` pair as TFLite's ``QuantizeMultiplier`` gives it.
    """

    def __init__(
        self,
        lhs_zero_point: int,
        rhs_zero_point: int,
        output_quant: QuantI8,
        requant: Sequence[tuple[int, int]],
        bias: Sequence[int] | None = None,
        activation: Activation = "none",
        kernels: TfliteKernels = "neon_ruy",
    ) -> None: ...
    @staticmethod
    def tflite_batch_matmul(
        lhs_quant: QuantI8,
        rhs_quant: QuantI8,
        output_quant: QuantI8,
        kernels: TfliteKernels = "neon_ruy",
    ) -> MatMulI8Plan:
        """TFLite's int8 ``BATCH_MATMUL``; ``adj_x`` / ``adj_y`` are transposed views."""
    @staticmethod
    def tflite_conv(
        input_quant: QuantI8,
        filter_scales: Sequence[float],
        output_quant: QuantI8,
        bias: Sequence[int] | None = None,
        activation: Activation = "none",
        kernels: TfliteKernels = "neon_ruy",
    ) -> MatMulI8Plan:
        """A TFLite int8 1x1 ``CONV_2D``: ``lhs`` is the input ``[pixels, in_channels]``,
        ``rhs`` the filter as ``[in_channels, out_channels]`` (a transposed view of
        TFLite's ``[out, 1, 1, in]``), one filter scale per output channel or one for all.
        """
    @staticmethod
    def check_depth(k: int) -> None:
        """Raise ``NotImplementedError`` when a reduction axis of ``k`` elements is too long."""
    @property
    def tier(self) -> str:
        """Kernel tier in use: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``."""
    def run(
        self,
        lhs: npt.NDArray[numpy.int8],
        rhs: npt.NDArray[numpy.int8],
        output: npt.NDArray[numpy.int8],
        *,
        threads: int | None = None,
    ) -> None:
        """``[..., M, K] @ [..., K, N] -> [..., M, N]``, all int8 of one rank.

        The batch axes must match; a broadcast operand is a zero-stride view.
        """

class SoftmaxDotI8Plan:
    """int8 softmax over the last axis fused with a product by one weight column.

    Stands for ``SoftmaxI8Plan`` then ``projection`` with ``weights`` as its
    ``[n, 1]`` right-hand side (a YOLO DFL head: ``SOFTMAX`` then a 1x1
    ``CONV_2D`` onto one channel), without rounding the probabilities to 8 bits:
    within one LSB of the float result, closer to it than TFLite's pair.
    ``projection`` must have one requantization, at most one bias, and the
    probabilities' zero point as its ``lhs_zero_point``.
    """

    def __init__(
        self,
        input_quant: QuantI8,
        weights: npt.NDArray[numpy.int8] | Sequence[int],
        projection: MatMulI8Plan,
        beta: float = 1.0,
        probs_quant: QuantI8 = (1 / 256, -128),
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
        """``[..., n]`` logits (unit-stride rows) to ``[..., 1]`` outputs."""

class AttentionI8Plan:
    """Fused int8 attention ``softmax(q' @ k.T) @ v`` per head.

    Replaces TFLite's ``MUL`` (on Q) -> ``BATCH_MATMUL`` -> ``SOFTMAX`` ->
    ``BATCH_MATMUL`` chain with its int8 roundings. ``q_mul`` is the optional
    scalar ``MUL`` on Q: its 256-entry output table (``BinaryI8Plan.table_for_rhs``)
    and its output quantization. ``probs_quant`` must be ``(1/256, -128)``.

    The plan caches the packed scratch the kernels need and rebuilds it when the
    shapes change; a ``run`` that finds it in use by another thread allocates
    its own for that call.
    """

    def __init__(
        self,
        q_quant: QuantI8,
        k_quant: QuantI8,
        v_quant: QuantI8,
        scores_quant: QuantI8,
        probs_quant: QuantI8,
        output_quant: QuantI8,
        beta: float = 1.0,
        q_mul: tuple[npt.NDArray[numpy.uint8], QuantI8] | None = None,
        kernels: TfliteKernels = "neon_ruy",
    ) -> None: ...
    @staticmethod
    def check_shape(tokens: int, dk: int, dv: int) -> None:
        """Raise ``NotImplementedError`` when the kernels cannot take these tokens and head dims."""
    @property
    def tier(self) -> str:
        """Kernel tier in use: ``scalar``, ``neon``, ``neon_fp16`` or ``neon_dotprod``."""
    def run(
        self,
        q: npt.NDArray[numpy.int8],
        k: npt.NDArray[numpy.int8],
        v: npt.NDArray[numpy.int8],
        output: npt.NDArray[numpy.int8],
        *,
        threads: int | None = None,
    ) -> None:
        """Attention of ``q`` ``[heads, queries, dk]`` over ``k`` ``[heads, tokens, dk]``
        and ``v`` ``[heads, tokens, dv]`` into ``output`` ``[heads, queries, dv]``.

        Any non-negative strides: head slices of one token-major array work
        without a copy. The inputs may share memory; the output may not.
        """
