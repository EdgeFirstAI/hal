"""Type stubs for ``edgefirst.tensor``.

Split from the pre-0.29 monolithic ``edgefirst_hal.pyi``; assignment
mirrors the pymodule registrations in crates/python-tensor/src/lib.rs.
"""

import enum
import sys
from typing import Literal, Protocol

import numpy.typing as npt
from typing_extensions import Self, TypeAlias

"""EdgeFirst HAL Python bindings."""

def version() -> str:
    """Return the HAL version string (matches ``Cargo.toml``)."""

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
    whether ``TensorMemory.DMABUF`` will succeed without caring which
    primitive backs it.
    """

def is_shm_available() -> bool:
    """True when POSIX shared memory allocation is available (Linux and macOS)."""

def is_cuda_available() -> bool:
    """True when libcudart is loaded and all CUDA interop symbols resolved.

    Checks whether zero-copy CUDA tensor mapping is available on this system.
    Use this to gate CUDA-specific code paths before calling
    :meth:`Tensor.cuda_map`. The result is cached after the first call.
    """

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
    def zero_point(self) -> list[int] | None: ...
    @property
    def axis(self) -> int | None: ...
    @property
    def is_per_tensor(self) -> bool: ...
    @property
    def is_per_channel(self) -> bool: ...
    @property
    def is_symmetric(self) -> bool: ...

class TensorMemory(enum.Enum):
    """Every member is defined on every platform -- this is a namespace of
    codes, not a list of what the current platform can allocate. A tensor
    recorded on one platform names the same member when read back on
    another, even where that platform cannot materialise it.
    ``TensorMemory.is_available()`` (not shown here -- see the Rust
    ``TensorMemory::is_available`` docs) answers the runtime question.
    """

    MEM: TensorMemory
    """Regular system memory allocation. Available everywhere."""

    SHM: TensorMemory
    """
    POSIX Shared Memory allocation. Suitable for inter-process
    communication, but not suitable for hardware acceleration. Nameable on
    every platform; only allocatable on unix.
    """

    DMABUF: TensorMemory
    """
    Platform-native zero-copy GPU buffer: DMA-BUF on Linux, IOSurface on
    macOS. The name is the same on both. CPU reads and writes cost more
    than system memory, but the GPU and other hardware blocks can use the
    buffer without a copy.
    """

    IOSURFACE: TensorMemory
    """
    Apple IOSurface, named specifically rather than through the portable
    ``DMABUF`` spelling. No backend produces or accepts it yet -- macOS/iOS
    allocate and report ``DMABUF``.
    """

    PBO: TensorMemory
    """
    GPU Pixel Buffer Object (PBO) allocation. Used for zero-copy GPU
    upload/readback on platforms without DMA-buf support.
    """

    CUDA: TensorMemory
    """CUDA device memory. No backend produces or accepts it yet."""

class EdgeFirstTensorExportable(Protocol):
    """Structural type for anything that can hand a tensor across an
    ``edgefirst.*`` package boundary via the ``__edgefirst_tensor__``
    capsule protocol.

    See ``crates/python-common/INTEROP.md`` in the
    `hal <https://github.com/EdgeFirstAI/hal>`_ repository for the full
    protocol. Consumers should duck type
    (``hasattr(obj, "__edgefirst_tensor__")``) rather than ``isinstance``
    against this protocol or against a concrete ``Tensor`` type -- every
    ``edgefirst.*`` package registers its own ``Tensor`` type object (see
    `PyO3 #1444 <https://github.com/PyO3/pyo3/issues/1444>`_).
    """

    def __edgefirst_tensor__(self, access: str | None = None) -> object: ...

DType: TypeAlias = Literal[
    "int8",
    "uint8",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
    "float16",
    "float32",
    "float64",
]
"""Element type names accepted wherever the API takes a ``dtype`` string."""

class Tensor:
    def __init__(
        self,
        shape: list[int],
        dtype: DType = "float32",
        mem: None | TensorMemory = None,
        name: None | str = None,
    ) -> None:
        """Create a new tensor with the given shape, memory type, and optional
        name. If no name is given, a random name is generated. If no memory
        type is given, the best available one is chosen for the platform.

        The order of preference is DMABUF -> SHM -> MEM on Linux and
        DMABUF (IOSurface) -> SHM -> MEM on macOS; Windows falls back to MEM.
        ``Tensor(...)`` does not probe the GPU backend — use
        :meth:`ImageProcessor.create_image` for anything you intend to pass
        to :meth:`ImageProcessor.convert`.

        Environment variables:
            ``EDGEFIRST_TENSOR_FORCE_MEM``: when set to a non-zero,
            non-false value, forces ``TensorMemory.MEM`` regardless of
            platform capabilities.
        """

    if sys.platform != "win32":
        @staticmethod
        def from_fd(
            fd: int,
            shape: list[int],
            dtype: DType = "float32",
            name: None | str = None,
        ) -> Tensor:
            """Import an existing buffer as a tensor, without copying. If no
            name is given, a random name will be generated.

            The buffer type is **detected, not chosen**. On Linux it is
            determined by the file descriptor's filesystem magic:

            =========================================  =========================
            File descriptor                            ``Tensor.memory``
            =========================================  =========================
            ``dma_buf`` (``DMA_BUF_MAGIC``)            ``TensorMemory.DMABUF``
            ``tmpfs`` — ``/dev/shm`` and ``memfd``     ``TensorMemory.SHM``
            anything else                              raises (see below)
            =========================================  =========================

            On macOS the fd is always imported as ``TensorMemory.SHM``.

            Both supported types are identified positively; an unrecognized
            filesystem raises rather than silently falling back to shared
            memory. That fallback would not fail loudly — a DMA-BUF is
            mmap-able, so it would import as a working tensor that merely
            isn't DMA, and the lost zero-copy would only surface later as
            ``ImageProcessor.import_image`` refusing the buffer.

            The fd is ``dup()``'d immediately — the caller retains ownership
            of the original fd and must close it when done.

            Raises:
                RuntimeError: The fd could not be imported. Most commonly its
                    buffer type could not be determined because it is neither
                    a DMA-BUF nor tmpfs-backed (a regular file, a pipe or
                    socket, or a ``MFD_HUGETLB`` memfd) — the message reports
                    the observed filesystem magic, e.g. ``Tensor error:
                    UnknownBufferType: fd is on an unrecognized filesystem
                    (magic 0x50495045)``. Also raised for a negative ``fd``,
                    an unsupported ``dtype``, a shape larger than the buffer,
                    or a failed syscall.

            Note:
                Check ``tensor.memory`` if you require zero-copy — a
                successful import is not by itself proof of DMA backing.
            """

        @property
        def fd(self) -> int:
            """A duplicate of the file descriptor backing the tensor's memory.

            The caller owns the returned descriptor and must close it.
            """

    @property
    def dtype(self) -> DType:
        """The data type of the tensor."""

    @property
    def size(self) -> int:
        """The size of the tensor in bytes."""

    @property
    def compression(self) -> str | None:
        """Vendor tile-compression scheme recorded at allocation.

        ``"ubwc"``/``"afbc"``/``"pvric"``/``"dcc"``, or ``None`` for a
        linear layout. A compressed tensor has no meaningful linear row
        stride and CPU maps are best-effort.
        """

    @property
    def memory(self) -> TensorMemory:
        """The memory type of the tensor."""

    @property
    def name(self) -> str:
        """The name of the tensor."""

    @property
    def shape(self) -> list[int]:
        """The shape of the tensor. A property, not a method."""

    def reshape(self, shape: list[int]) -> None:
        """Reshape the tensor. The total element count must stay the same."""

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

    def configure_image(self, width: int, height: int, format: PixelFormat) -> None:
        """Set this tensor's logical dimensions and pixel format to a decoded
        image, reusing the existing allocation.

        Unlike :meth:`set_format` (which only validates the *current* shape
        against a format) and :meth:`reshape` (which requires the same
        element count), this can shrink or grow the logical shape within the
        tensor's allocated capacity -- exactly what a JPEG/PNG decode does to
        its destination. ``edgefirst.codec``'s cross-package
        ``decode_into``/``decode_file_into`` write-back uses this to leave a
        foreign destination in the same state a same-module decode would.

        Args:
            width: Decoded image width in pixels.
            height: Decoded image height in pixels.
            format: Native pixel format of the decoded data.

        Raises:
            RuntimeError: If the allocation cannot hold ``width``x``height``
                in ``format``, or the dimensions are invalid for the format.
        """

    if sys.platform == "linux":
        def dmabuf_clone(self) -> int:
            """Clone the DMA-BUF file descriptor backing this tensor.

            Returns a new file descriptor that the caller must close.

            Returns:
                A new file descriptor (int) that the caller must close.

            Raises:
                RuntimeError: If the tensor is not DMA-backed or fd clone fails.
            """

    if sys.platform == "darwin":
        @staticmethod
        def from_iosurface(
            surface_ref: int,
            shape: list[int],
            dtype: DType = "uint8",
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
                A new ``Tensor`` reporting ``TensorMemory.DMABUF``.

            Raises:
                RuntimeError: If ``surface_ref`` is null, the import
                    fails, or the requested shape exceeds the surface's
                    allocated size.
            """

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
                from edgefirst.tensor import PixelFormat, Tensor, TensorMemory

                # Create the tensor (or import an existing IOSurface).
                t = Tensor.image(
                    1280, 720, PixelFormat.Rgba, mem=TensorMemory.DMABUF
                )

                # Wrap the raw IOSurfaceRef for ctypes handoff.
                surf_ptr = ctypes.c_void_p(t.iosurface_ref)

                # `ci_image_with_iosurface` is whatever native API you
                # bound via ctypes; the IOSurface stays valid while `t`
                # is alive.
                # ci_image_with_iosurface(surf_ptr)
            """

    def map(
        self, access: Literal["read", "write", "readwrite"] = "readwrite"
    ) -> HostView:
        """Map the tensor's memory for direct CPU access.

        Returns a ``HostView`` context manager that exposes the raw buffer.
        Use with a ``with`` statement to ensure the mapping is released.

        The map owns its cache-coherency bracket in both directions, unlike
        :meth:`pin_host`, which is decoupled and pairs with
        :meth:`cpu_access`.

        ``access`` selects that bracket's direction and is worth setting.
        The default ``"readwrite"`` pays a full-buffer cache writeback when
        the map is released; a reader does not need it. On a non-coherent
        DMA-BUF backing that is a per-frame cost, and on macOS ``"read"``
        takes the read-only IOSurface lock, skipping the unlock flush::

            with tensor.map("read") as view:
                frame = np.asarray(view)   # read-only, zero-copy

        A ``"read"`` view is advertised to the buffer protocol as read-only,
        so ``np.asarray`` of it is not writable and asking for a writable
        buffer raises ``BufferError`` rather than silently handing back
        writes that the release would discard.

        Example — write a numpy array into a tensor::

            import numpy as np
            from edgefirst.tensor import Tensor

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

    def cuda_map(self) -> CudaMap | None:
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

    @staticmethod
    def image(
        width: int,
        height: int,
        format: PixelFormat,
        mem: TensorMemory | None = None,
        access: str = "none",
    ) -> Tensor:
        """Create an image tensor with the given dimensions and pixel format.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            format: Pixel format for the image data.
            mem: Optional memory type override. If None, the best available
                memory type is chosen automatically.
            access: Declared CPU access — ``"none"`` (default), ``"read"``,
                ``"write"``, or ``"readwrite"``. Hardware access is always
                implied; pass ``"readwrite"`` (or the precise direction) when
                the script will ``map()`` or ``numpy()`` the tensor.
        """

    # `save_jpeg` and `normalize_to_numpy` are NOT on this class. They were
    # inherited from the pre-0.29 monolithic stub and belong to the encoders
    # and normalisers that ship with `edgefirst.codec` and `edgefirst.image`
    # respectively — importing `edgefirst.tensor` alone links neither.

    def __edgefirst_tensor__(self, access: str | None = None) -> object:
        """Producer half of the cross-package tensor protocol.

        Returns a ``PyCapsule`` named ``edgefirst_tensor_v1``. Consumers in
        other ``edgefirst.*`` packages read this instead of type-checking --
        see :class:`EdgeFirstTensorExportable` and
        ``crates/python-common/INTEROP.md``.

        May be called more than once per operation (e.g. a caller retrying
        with a different ``access`` after an ``access=None`` descriptor
        turns out to need a host address); implementations must be
        side-effect free.

        Args:
            access: ``None`` (default) requests no pin -- the descriptor
                still carries shape, format and the native handle, which is
                all a zero-copy consumer needs. ``"read"``, ``"write"`` or
                ``"readwrite"`` pins host memory with the matching access
                and fills in the descriptor's address.

        Raises:
            ValueError: if ``access`` is not one of the values above.
        """

    def pin_host(self, access: Literal["read", "write", "readwrite"]) -> HostPin:
        """Pin a stable host address that outlives every map guard.

        The returned :class:`HostPin` borrows nothing, so the tensor stays
        free to be mutated — that is the point. Hand the address to an
        external runtime and keep converting into the tensor underneath it.

        Args:
            access: The CPU access to pin for. ``"none"`` is not valid: a
                pinned mapping is CPU access by definition.

        Raises:
            RuntimeError: on a backend that cannot separate addressing from
                coherency (an OpenGL PBO, an Android AHardwareBuffer). Use
                :meth:`map` there instead.
        """

    def cpu_access(
        self, access: Literal["read", "write", "readwrite"]
    ) -> CpuAccessGuard:
        """Bracket CPU access to this tensor for cache coherency.

        A no-op on coherent backends, so portable code can bracket
        unconditionally::

            with tensor.cpu_access("read"):
                ...  # read through a pin or a raw address
        """

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
            from edgefirst.tensor import Tensor

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

    @property
    def format(self) -> PixelFormat | None:
        """Pixel format of this tensor (None if not an image tensor)."""

    @property
    def width(self) -> int | None:
        """Image width in pixels (None if not an image tensor)."""

    @property
    def height(self) -> int | None:
        """Image height in pixels (None if not an image tensor)."""

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

    @property
    def is_planar(self) -> bool:
        """Whether this image uses a planar pixel layout."""

    @property
    def quantization(self) -> Quantization | None:
        """Quantization metadata, or ``None`` for float tensors and
        unquantized integer tensors."""

    @property
    def colorimetry(self) -> Colorimetry | None:
        """Colour signalling (matrix/range/primaries), or ``None`` if undefined.

        Set automatically by the codec on decode (JPEG → JFIF/BT.601-full,
        PNG → sRGB) and carried through ``convert()`` to pick the YUV→RGB
        matrix and range. ``None`` is never auto-filled; consumers resolve
        missing axes via an SD/HD height heuristic at use time.
        """

    @colorimetry.setter
    def colorimetry(self, value: Colorimetry | None) -> None: ...
    def set_quantization_per_tensor(self, scale: float, zero_point: int) -> None:
        """Attach per-tensor asymmetric quantization. Integer tensors only."""

    def set_quantization_per_tensor_symmetric(self, scale: float) -> None:
        """Attach per-tensor symmetric quantization. Integer tensors only."""

    def set_quantization_per_channel(
        self, scales: list[float], zero_points: list[int], axis: int
    ) -> None:
        """Attach per-channel asymmetric quantization. Integer tensors only.
        Raises on length mismatch or invalid axis."""

    def set_quantization_per_channel_symmetric(
        self, scales: list[float], axis: int
    ) -> None:
        """Attach per-channel symmetric quantization. Integer tensors only."""

    def clear_quantization(self) -> None:
        """Remove any quantization metadata from this tensor."""

class HostView:
    """Mapped-memory guard returned by :meth:`Tensor.map`.

    Not constructible from Python — obtain instances from ``Tensor.map()``.
    Replaced the former ``TensorMap`` in 0.29: one view type now covers every
    backend, so the mapped extent no longer depends on which one you have.
    """

    def unmap(self) -> None: ...
    def numpy(self) -> memoryview: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> object: ...
    def __setitem__(self, index: int, value: object) -> None: ...
    def __getbuffer__(self, view, _flags) -> None: ...
    def __releasebuffer__(self, view) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, _exc_type, _exc_value, _traceback) -> None: ...

class HostPin:
    """A stable host address for a tensor's data, valid until released.

    Unlike :class:`HostView` this carries **no borrow of the tensor**, so the
    address survives every map guard and stays put across calls that mutate
    the tensor — including ``ImageProcessor.convert()`` rendering into it.
    That is what lets a pinned buffer back an external runtime's tensor (a
    TFLite custom allocation, an ONNX Runtime external tensor) while a frame
    loop keeps writing it.

    Not constructible from Python — obtain instances from
    ``Tensor.pin_host()``. Usable as a context manager, which releases on
    exit; reading :attr:`ptr` after release raises ``RuntimeError``.

    The address is not a coherency guarantee: bracket CPU access with
    :meth:`Tensor.cpu_access`, which is a no-op where nothing is owed.
    """

    @property
    def ptr(self) -> int:
        """The pinned host address as an integer."""

    @property
    def len(self) -> int:
        """The tensor's logical byte length, offset-adjusted for views.

        For a stride-padded image this is less than ``row_stride * height``;
        use :meth:`Tensor.map` when you need the padded extent.
        """

    @property
    def alignment(self) -> int:
        """Alignment of the pinned address in bytes.

        TFLite requires 64 (``kDefaultTensorAlignment``) for a custom
        allocation, so check rather than assume.
        """

    def release(self) -> None:
        """Release the pin. Any consumer still holding the address is now
        holding a dangling pointer."""

    def __enter__(self) -> Self: ...
    def __exit__(self, _exc_type, _exc_value, _traceback) -> None: ...

class CpuAccessGuard:
    """Coherency bracket returned by :meth:`Tensor.cpu_access`.

    Entering makes the buffer coherent for the CPU, leaving releases it back
    to the device. A no-op on ``mem``/``shm``; a cache maintenance ioctl on a
    Linux DMA-BUF and a lock/unlock pair on a macOS IOSurface.
    """

    def __enter__(self) -> Self: ...
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

    @property
    def size(self) -> int:
        """Length of the mapping in bytes. Returns ``0`` if released."""

    def __len__(self) -> int: ...
    def release(self) -> None:
        """Release the CUDA mapping (idempotent).

        Called automatically on ``with`` exit. May also be called explicitly
        when early release is needed before the ``with`` block exits.
        """

    def __enter__(self) -> Self: ...
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

    @property
    def space(self) -> ColorSpace | None:
        """Colour primaries, or ``None`` if undefined."""

    @property
    def transfer(self) -> ColorTransfer | None:
        """Transfer function, or ``None`` if undefined."""

    @property
    def encoding(self) -> ColorEncoding | None:
        """YCbCr encoding matrix, or ``None`` if undefined."""

    @property
    def range(self) -> ColorRange | None:
        """Quantization range, or ``None`` if undefined."""

def build_info() -> str:
    """Return a human-readable build configuration string."""

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

        import edgefirst.tensor as hal

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

    def stop(self) -> None:
        """Stop trace capture and flush the trace file.

        After this call the trace file is complete and ready for viewing.
        No-op if not currently active.
        """

    def __enter__(self) -> Self: ...
    def __exit__(self, *args: object) -> bool: ...

# ---------------------------------------------------------------------------
# SAHI-style tiled inference
# ---------------------------------------------------------------------------

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

class Region:
    """A rectangular sub-region (pixels) defined by its top-left corner (x, y)
    and dimensions (width, height). Used for ``Tensor.view(region)`` and the
    source crop of ``convert``."""

    def __init__(self, x: int, y: int, width: int, height: int): ...

    x: int
    y: int
    width: int
    height: int
