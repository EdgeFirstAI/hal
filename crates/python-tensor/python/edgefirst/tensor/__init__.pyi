"""Type stubs for ``edgefirst.tensor``."""

import enum
import sys
from typing import Literal, Protocol

import numpy.typing as npt
from typing_extensions import Self, TypeAlias

"""EdgeFirst HAL Python bindings."""

# "dynamic" in every wheel; "static" only in the comparison build.
__tensor_backend__: str

def version() -> str:
    """Return the HAL version string (matches ``Cargo.toml``)."""

def is_dma_available() -> bool:
    """True when Linux DMA-BUF heap allocation is available.

    Declared and answered on every platform; ``True`` only on Linux. Use
    :func:`is_gpu_buffer_available` for the portable question.
    """

def is_iosurface_available() -> bool:
    """True when macOS IOSurface allocation is available.

    Declared and answered on every platform; ``True`` only on macOS and iOS.
    Use :func:`is_gpu_buffer_available` for the portable question.
    """

def is_gpu_buffer_available() -> bool:
    """True when a platform-native GPU-coherent buffer kind is available.

    A DMA-BUF on Linux, an IOSurface on macOS and iOS, an AHardwareBuffer on
    Android, a D3D11 texture on Windows. Declared and answered on every
    platform. Reports whether ``TensorMemory.DMABUF`` allocation can succeed,
    whichever primitive backs it.
    """

def is_shm_available() -> bool:
    """True when POSIX shared memory allocation is available.

    Declared and answered on every platform; ``True`` only where ``/dev/shm``
    is writable, so not on Windows.
    """

def is_cuda_available() -> bool:
    """True when libcudart is loaded and all CUDA interop symbols resolved.

    Declared and answered on every platform; ``True`` only where a CUDA
    runtime loaded.
    """

if sys.platform == "win32":
    def d3d11_device() -> int:
        """Borrowed ``ID3D11Device*`` of the process device, as an integer.

        Platforms:
            Windows.

        The device is created on the first call. Wrap with
        ``ctypes.c_void_p`` for native callers; no reference is
        transferred, so never ``Release`` this pointer.

        This is the device inside ``libedgefirst_tensor``, on which the
        tensors this wheel allocates live.

        Raises:
            RuntimeError: If no device could be created.
        """

    def d3d11_use_external_device(ptr: int) -> None:
        """Install a host-owned D3D11 device before the HAL's first use.

        Platforms:
            Windows.

        Tensors the HAL allocates then live on the caller's device. The
        reference stays the caller's. Must run before anything that creates
        the device as a side effect: :func:`d3d11_device`,
        :func:`is_gpu_buffer_available`, or any texture allocation.

        The pointer is installed into ``libedgefirst_tensor``.

        Raises:
            RuntimeError: If the device is already initialized, or ``ptr``
                is not a live ``ID3D11Device``.
        """

    class D3d11Layout:
        """DXGI format and texture geometry behind a texture tensor.

        Platforms:
            Windows.

        The dimensions are the texture's, not the image's: a packed row
        folds several pixels or several planes into one texel. An NV12
        image of even width and height is a native two-plane texture
        (``dxgi_format`` 103, ``DXGI_FORMAT_NV12``) whose dimensions are the
        luma grid; its chroma is the texture's second plane.
        """

        dxgi_format: int
        texture_width: int
        texture_height: int
        bytes_per_texel: int
        gl_internal_format: int

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
    """Every member is defined on every platform. A tensor recorded on one
    platform names the same member when read back on another, even where
    that platform cannot materialise it.
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
    Apple IOSurface. No backend produces or accepts it; macOS and iOS
    allocate and report ``DMABUF``.
    """

    PBO: TensorMemory
    """
    GPU Pixel Buffer Object (PBO) allocation. Used for zero-copy GPU
    upload/readback on platforms without DMA-buf support.
    """

    CUDA: TensorMemory
    """CUDA device memory. No backend produces or accepts it."""

    D3D11_TEXTURE: TensorMemory
    """
    Windows D3D11 texture. No backend produces or accepts it; Windows
    allocates and reports ``DMABUF``.
    """

class EdgeFirstTensorExportable(Protocol):
    """Structural type for anything that can hand a tensor across an
    ``edgefirst.*`` package boundary via the ``__edgefirst_tensor__``
    capsule protocol.

    See ``crates/python-common/INTEROP.md`` in the
    `hal <https://github.com/EdgeFirstAI/hal>`_ repository for the full
    protocol. Each ``edgefirst.*`` package registers its own ``Tensor`` type
    object, so consumers duck type
    (``hasattr(obj, "__edgefirst_tensor__")``) and do not use ``isinstance``.

    A tensor whose chroma plane is a separate tensor (two-buffer NV12, chroma
    at an offset, a decoded video frame) cannot cross a package boundary: its
    ``__edgefirst_tensor__`` raises ``NotImplementedError``.
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

            Platforms:
                Linux, macOS.

            The buffer type is detected. On Linux it is determined by the
            file descriptor's filesystem magic:

            =========================================  =========================
            File descriptor                            ``Tensor.memory``
            =========================================  =========================
            ``dma_buf`` (``DMA_BUF_MAGIC``)            ``TensorMemory.DMABUF``
            ``tmpfs`` — ``/dev/shm`` and ``memfd``     ``TensorMemory.SHM``
            anything else                              raises (see below)
            =========================================  =========================

            On macOS the fd is always imported as ``TensorMemory.SHM``.

            The fd is ``dup()``'d immediately — the caller retains ownership
            of the original fd and must close it when done.

            Raises:
                RuntimeError: The fd could not be imported: its buffer type is
                    neither a DMA-BUF nor tmpfs-backed (a regular file, a pipe
                    or socket, or a ``MFD_HUGETLB`` memfd; the message reports
                    the observed filesystem magic), ``fd`` is negative,
                    ``dtype`` is unsupported, the shape is larger than the
                    buffer, or a syscall failed.

            Note:
                ``tensor.memory`` reports the detected type.
            """

        @property
        def fd(self) -> int:
            """A duplicate of the file descriptor backing the tensor's memory.

            Platforms:
                Linux, macOS.

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
    def contiguity(self) -> str:
        """Whether the memory is known to be physically contiguous.

        ``"contiguous"`` for a DMA-BUF allocated from the CMA heap,
        ``"non_contiguous"`` for one from the system heap, and ``"unknown"``
        for an imported fd and every other kind of memory.
        """

    @property
    def declared_cpu_access(self) -> str:
        """The CPU access declared for this tensor.

        ``"none"``, ``"read"``, ``"write"`` or ``"readwrite"``: the maps,
        pins and syncs this tensor accepts. Set at allocation; a tensor
        imported from another package carries its producer's declaration.
        See :meth:`cpu_access` for the coherency bracket.
        """

    @property
    def memory(self) -> TensorMemory:
        """The memory type of the tensor."""

    @property
    def name(self) -> str:
        """The name of the tensor."""

    @property
    def shape(self) -> list[int]:
        """The shape of the tensor."""

    def reshape(self, shape: list[int]) -> None:
        """Reshape the tensor. The total element count must stay the same."""

    def view(self, region: Region) -> Tensor:
        """Zero-copy rectangular sub-region view of an image tensor.

        ``region`` is in pixels of the image's leading frame. The view shares
        the parent's buffer and addresses the sub-rectangle by offset and the
        parent's row pitch.
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
        layout (packed, planar, or semi-planar).

        Args:
            format: Pixel format to attach.

        Raises:
            RuntimeError: If the tensor shape doesn't match the format layout.
        """

    def configure_image(self, width: int, height: int, format: PixelFormat) -> None:
        """Set this tensor's logical dimensions and pixel format to a decoded
        image, reusing the existing allocation.

        The logical shape can shrink or grow within the tensor's allocated
        capacity.

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

            Platforms:
                Linux.

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
            """Wrap an externally-allocated IOSurface as a Tensor.

            Platforms:
                macOS.

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
                    shape raises ``RuntimeError``.
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
            """``IOSurfaceID`` for cross-process surface sharing.

            Platforms:
                macOS.

            Returns ``None`` when the tensor is not IOSurface-backed. The ID
            is a 32-bit handle stable for the lifetime of the IOSurface; it
            can be passed across process boundaries (Mach port, XPC) and
            recovered via ``IOSurfaceLookup(id)``.
            """

        @property
        def iosurface_ref(self) -> int | None:
            """Borrowed ``IOSurfaceRef`` as an ``int``.

            Platforms:
                macOS.

            Hand this off to native macOS APIs that take an ``IOSurfaceRef``
            directly (``CIImage``, ``AVSampleBufferDisplayLayer``,
            ``CVPixelBufferCreateWithIOSurface``). The integer value is a
            raw pointer — wrap it with ``ctypes.c_void_p(...)`` before
            passing to a ctypes-bound function.

            The pointer's lifetime is tied to this tensor. If the surface
            must outlive this tensor, call ``CFRetain`` (via ctypes) on the
            pointer and pair it with a matching ``CFRelease``. Do not call
            ``CFRelease`` on the borrowed pointer without a prior
            ``CFRetain``.

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

    if sys.platform == "win32":
        def d3d11_texture(self) -> int | None:
            """Borrowed ``ID3D11Texture2D*`` as an integer.

            Platforms:
                Windows.

            Wrap with ``ctypes.c_void_p`` for native callers; the pointer
            lives as long as this tensor. ``None`` when the tensor is not a
            texture.
            """

        def d3d11_layout(self) -> D3d11Layout | None:
            """The texture's DXGI format and geometry.

            Platforms:
                Windows.

            ``None`` when the tensor is not a texture.
            """

        def d3d11_shared_handle(self) -> int:
            """Duplicate the NT handle of the D3D11 texture backing this tensor.

            Platforms:
                Windows.

            Returns:
                An NT handle the caller owns and must close with
                ``CloseHandle``.

            Raises:
                RuntimeError: If the tensor is not a D3D11 texture, or the
                    handle could not be duplicated.
            """

        def gpu_completion(self) -> tuple[int, int] | None:
            """Fence handle and value a GPU consumer waits on before reading.

            Platforms:
                Windows.

            Returns:
                ``(fence_handle, value)``, the handle owned by the caller
                and closed with ``CloseHandle``; ``None`` when no GPU write
                has been recorded.
            """

        @property
        def gpu_write_value(self) -> int:
            """The fence value of the newest recorded GPU write, or 0.

            Platforms:
                Windows.

            The same number :meth:`gpu_completion` returns as ``value``. 0
            when no write has been recorded, or when the tensor is not a
            texture.
            """

        def set_gpu_write(self, value: int) -> None:
            """Record that GPU work writing this texture completes at ``value``.

            Platforms:
                Windows.

            The recorded value is the maximum of the values set.

            Raises:
                RuntimeError: If the tensor is not a D3D11 texture.
            """

        @staticmethod
        def from_d3d11_texture(
            texture: int,
            shape: list[int],
            dtype: DType,
            format: PixelFormat,
            access: str = "readwrite",
            name: str | None = None,
        ) -> Tensor:
            """Wrap a texture created on the HAL device.

            Platforms:
                Windows.

            The width and height are read from the texture's own
            description. ``shape`` must be either the allocation shape or
            the addressing shape ``format`` gives those dimensions.

            A semi-planar texture (``Nv12``, ``Nv16``, ``Nv24``) carries the
            image's row stride as its width. It is accepted only when that
            width is even and equal to the staging row pitch the driver
            gives a texture of that width.

            Raises:
                ValueError: If ``shape`` is neither shape of this texture;
                    the message names both.
                RuntimeError: If ``texture`` is not a live texture on the
                    HAL device holding a ``format`` image.
            """

        @staticmethod
        def from_d3d11_shared_handle(
            handle: int,
            shape: list[int],
            dtype: DType,
            format: PixelFormat,
            access: str = "readwrite",
            completion: tuple[int, int] | None = None,
            name: str | None = None,
        ) -> Tensor:
            """Open a shared texture on the HAL device, waiting on ``completion`` first.

            Platforms:
                Windows.

            ``handle`` and ``completion``'s fence handle stay the caller's;
            the tensor duplicates what it keeps. The width and height are
            read from the opened texture's own description. ``shape`` must
            be either the allocation shape or the addressing shape
            ``format`` gives those dimensions. The semi-planar width rule of
            :meth:`from_d3d11_texture` applies.

            Args:
                completion: ``(fence_handle, value)`` from
                    :meth:`gpu_completion` on the producing side, waited on
                    before the texture is read.

            Raises:
                ValueError: If ``shape`` is neither shape of this texture;
                    the message names both.
                RuntimeError: If ``handle`` does not open a texture holding
                    a ``format`` image, or the fence wait fails.
            """

    def map(
        self, access: Literal["read", "write", "readwrite"] = "readwrite"
    ) -> HostView:
        """Map the tensor's memory for direct CPU access.

        Returns a ``HostView`` context manager that exposes the raw buffer.
        Use with a ``with`` statement to ensure the mapping is released.

        The map owns its cache-coherency bracket in both directions.

        ``access`` selects that bracket's direction. ``"read"`` skips the
        cache writeback when the map is released and, on macOS, takes the
        read-only IOSurface lock::

            with tensor.map("read") as view:
                frame = np.asarray(view)   # read-only, zero-copy

        A ``"read"`` view is advertised to the buffer protocol as read-only:
        ``np.asarray`` of it is not writable and asking for a writable
        buffer raises ``BufferError``.

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
            PermissionError: If ``access`` is outside the tensor's declared
                CPU access (the default ``"readwrite"`` on a ``"read"``- or
                ``"write"``-declared tensor included).
        """

    def try_map(
        self, access: Literal["read", "write", "readwrite"] = "readwrite"
    ) -> HostView:
        """Non-blocking :meth:`map`.

        Raises ``BlockingIOError`` while a GPU copy the map depends on is in
        flight; identical to :meth:`map` on backings without such a copy.

        A caller that retries after ``BlockingIOError`` must yield or sleep
        between attempts.

        Args:
            access: as :meth:`map`.

        Raises:
            BlockingIOError: While a GPU copy the map depends on is in
                flight.
            PermissionError: As :meth:`map`.
        """

    def cuda_map(self) -> CudaMap | None:
        """Attempt a zero-copy CUDA device-pointer mapping.

        Returns a :class:`CudaMap` context manager, or ``None`` if CUDA is
        unavailable for this tensor (libcudart not found, or the tensor was
        not registered with CUDA).

        Fall back to ``map()`` when it returns ``None``::

            cm = tensor.cuda_map()
            if cm is not None:
                with cm as m:
                    trt_set_input_address(m.device_ptr)   # zero-copy GPU
            else:
                with tensor.map() as host:
                    run_cpu_path(host)                    # CPU fallback

        On Windows the mapping of a D3D11 texture tensor is tight rows of
        ``width * bytes_per_texel``, and ``size`` is their sum, independent of
        :attr:`row_stride`. A consumer must not stride the device pointer by
        :attr:`row_stride`.

        A semi-planar texture (NV12, NV16, NV24) is as wide as its own pitch;
        its mapping is :attr:`row_stride` times the combined height.

        Returns:
            A :class:`CudaMap` context manager, or ``None``.
        """

    def cuda_map_mut(self) -> CudaMap | None:
        """Writable counterpart of :meth:`cuda_map`.

        Changes are written back to the tensor when the mapping is released.
        Returns ``None`` on the same terms as :meth:`cuda_map`. For a Windows
        D3D11 texture the release copies the device buffer back into the
        texture.

        On Windows the mapping of a D3D11 texture tensor is tight rows of
        ``width * bytes_per_texel``, and ``size`` is their sum, independent of
        :attr:`row_stride`. A consumer must not stride the device pointer by
        :attr:`row_stride`.

        A semi-planar texture (NV12, NV16, NV24) is as wide as its own pitch;
        its mapping is :attr:`row_stride` times the combined height.
        """

    @staticmethod
    def image(
        width: int,
        height: int,
        format: PixelFormat,
        mem: TensorMemory | None = None,
        access: str = "readwrite",
        contiguous: bool = False,
    ) -> Tensor:
        """Create an image tensor with the given dimensions and pixel format.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            format: Pixel format for the image data.
            mem: Optional memory type override. If None, the best available
                memory type is chosen automatically.
            access: Declared CPU access: ``"readwrite"`` (default),
                ``"read"``, ``"write"`` (whole-buffer writes only), or
                ``"none"`` (hardware-only). Hardware access is always
                implied. A CPU map, pin or sync outside the declaration
                raises ``PermissionError``, as does HAL CPU work (a CPU
                convert or draw, a decode) the declaration does not cover.
                ``"readwrite"`` and ``"read"`` use cached memory; on Linux a
                ``"none"`` or ``"write"`` DMA-BUF comes from the uncached CMA
                heap where available.
            contiguous: Require physically contiguous memory: a DMA-BUF from
                the Linux CMA heap, or an error naming the heap. ``mem``
                must be ``None`` or ``TensorMemory.DMABUF``.
        """

    def __edgefirst_tensor__(self, access: str | None = None) -> object:
        """Producer half of the cross-package tensor protocol.

        Returns a ``PyCapsule`` named ``edgefirst_tensor_v2``. See
        :class:`EdgeFirstTensorExportable` and
        ``crates/python-common/INTEROP.md``.

        The call has no side effects and may be repeated with the same or a
        different ``access``.

        Args:
            access: ``None`` (default) requests no pin; the descriptor
                carries shape, format and the native handle. ``"read"``,
                ``"write"`` or ``"readwrite"`` pins host memory with the
                matching access and fills in the descriptor's address.

        Raises:
            ValueError: if ``access`` is not one of the values above.
            PermissionError: if ``access`` pins outside the tensor's
                declared CPU access (see :attr:`declared_cpu_access`).
            NotImplementedError: if the tensor's chroma plane is a separate
                tensor (two-buffer NV12, chroma at an offset, a decoded video
                frame), which cannot be passed to another edgefirst package.
        """

    def pin_host(self, access: Literal["read", "write", "readwrite"]) -> HostPin:
        """Pin a stable host address that outlives every map guard.

        The returned :class:`HostPin` carries no borrow of the tensor: the
        tensor can still be converted into while the address is held.

        Args:
            access: The CPU access to pin for. ``"none"`` is not valid.

        Raises:
            RuntimeError: on a backend that cannot separate addressing from
                coherency (an OpenGL PBO, an Android AHardwareBuffer). Use
                :meth:`map` there instead.
            PermissionError: If ``access`` is outside the tensor's declared
                CPU access (see :attr:`declared_cpu_access`).
        """

    def cpu_access(
        self, access: Literal["read", "write", "readwrite"]
    ) -> CpuAccessGuard:
        """Bracket CPU access to this tensor for cache coherency.

        A no-op on coherent backends::

            with tensor.cpu_access("read"):
                ...  # read through a pin or a raw address

        Raises:
            PermissionError: If ``access`` is outside the tensor's declared
                CPU access (see :attr:`declared_cpu_access`).
        """

    def from_numpy(self, src: npt.NDArray) -> None:
        """Copy data from a numpy array into this tensor.

        Accepts any numpy dtype as long as it matches the tensor's dtype.
        The total number of elements must match. Both contiguous and
        non-contiguous (strided) arrays are supported:

        - Contiguous arrays use a direct memcpy.
        - Non-contiguous arrays (slices, transposes) are copied
          element-wise via the array's stride metadata.
        - Large copies (≥256 KiB) are parallelized automatically.

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
            PermissionError: If the tensor's declared CPU access has no
                write direction.
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
        matrix and range. ``None`` means undefined; consumers resolve missing
        axes via an SD/HD height heuristic at use time, and
        ``ImageProcessor.convert`` tags an untagged YUV destination with the
        colorimetry it encoded.
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

    Carries no borrow of the tensor: the address stays valid across calls
    that mutate the tensor, including ``ImageProcessor.convert()`` rendering
    into it.

    Not constructible from Python — obtain instances from
    ``Tensor.pin_host()``. Usable as a context manager, which releases on
    exit; reading :attr:`ptr` after release raises ``RuntimeError``.

    The address gives no coherency guarantee: bracket CPU access with
    :meth:`Tensor.cpu_access`.
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
        """Alignment of the pinned address in bytes."""

    def release(self) -> None:
        """Release the pin. The address is invalid afterwards."""

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
    mapping is released on ``__exit__``.

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

        Called automatically on ``with`` exit. May also be called explicitly.
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
        colorspace: int, xfer: int, ycbcr_enc: int, quant: int, is_rgb_or_hsv: bool
    ) -> Colorimetry:
        """Build from the four raw V4L2 colorimetry integers.

        ``is_rgb_or_hsv`` is whether the negotiated format's pixel encoding
        is RGB or HSV. A ``DEFAULT`` (0) ``ycbcr_enc``/``quant`` is resolved
        per the kernel ``V4L2_MAP_*_DEFAULT`` rules: ``quant`` is full range
        for an RGB/HSV format, whatever the colorspace, and for
        ``V4L2_COLORSPACE_JPEG``; limited for a YUV format in any other
        recognised colorspace; and ``None`` for a YUV format in a
        ``DEFAULT`` or unrecognised colorspace. ``ycbcr_enc`` follows the
        colorspace (e.g. ``V4L2_COLORSPACE_JPEG`` → BT.601) and is likewise
        ``None`` for a ``DEFAULT`` or unrecognised one. An unrecognised value
        maps to ``None``.
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

    Only one trace session per process lifetime is supported.

    Usage as context manager:

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
# Pixel formats and regions
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
