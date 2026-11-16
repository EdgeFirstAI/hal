// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Layout-only C ABI contract for the EdgeFirst tensor surface.
//!
//! Plain `#[repr(C)]` values and the integer vocabularies, declared once and
//! imported by every `-capi` crate.
//!
//! This crate has no dependencies.

/// One plane's location, mirroring `TensorPlane` on the wire.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EfTensorPlane {
    /// dma-buf fd, IOSurface id, or -1 when host memory.
    pub handle: i64,
    /// Byte offset of this plane within the handle.
    pub offset: u64,
    /// Bytes per line.
    pub stride: u64,
    /// Plane extent in bytes.
    pub size: u64,
    /// Valid payload bytes.
    pub used: u64,
    /// DRM format modifier; 0 = linear.
    pub modifier: u64,
}

/// Element type of a tensor's addressing grid.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfDtype {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    U64 = 6,
    I64 = 7,
    F16 = 8,
    F32 = 9,
    F64 = 10,
}

/// Backing store for a tensor.
///
/// Every kind is declared on every platform; whether one can be allocated is a
/// runtime question.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfStorageKind {
    Mem = 0,
    Shm = 1,
    DmaBuf = 2,
    IoSurface = 3,
    Pbo = 4,
    Cuda = 5,
    /// Windows `ID3D11Texture2D`, named specifically. A tensor's own
    /// storage kind reports `DmaBuf` on Windows; this is the code the
    /// cross-package descriptor carries for that backing.
    D3d11Texture = 6,
}

/// CPU access: a tensor's declaration, or the direction of a map window.
///
/// Mirrors `edgefirst_tensor::CpuAccess`; the values are the wire codes.
/// `None` names the no-CPU-access declaration and is never a valid map
/// direction.
///
/// The declaration, fixed when an image is allocated, covers all CPU access
/// to it, the library's own included, and is enforced: `ef_tensor_map`,
/// `ef_tensor_sync_*` and the copies fail with `EACCES` outside it, and a
/// library CPU pass (a CPU convert or draw, a decode) that would need more
/// fails instead of running. Hardware (GPU, NPU, G2D, a compositor) access is
/// always allowed.
///
/// - `EF_CPU_ACCESS_NONE`: no CPU access. On Linux a DMA-BUF comes from the
///   uncached CMA heap where available.
/// - `EF_CPU_ACCESS_READ`: CPU reads; cached memory.
/// - `EF_CPU_ACCESS_WRITE`: CPU writes that cover the whole buffer (decode
///   targets, CPU-drawn overlays). On Linux a DMA-BUF comes from the uncached
///   CMA heap where available, and CPU writes are write-combined.
/// - `EF_CPU_ACCESS_READ_WRITE`: both; cached memory. The default.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfCpuAccess {
    None = 0,
    Read = 1,
    Write = 2,
    ReadWrite = 3,
}

/// Compression request, as `ef_tensor_image_desc_set_compression` takes it and
/// [`EfImageDescView::compression`] reports it.
///
/// Distinct from [`EfCompression`], which names the scheme an allocation
/// resolved to. No C setter can request `Scheme`; the view reports it for a
/// request made through the Rust API.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EfCompressionRequest {
    /// No compression request: allocate linear.
    #[default]
    None = 0,
    /// Any scheme the platform offers, linear when the format is not eligible.
    Any = 1,
    /// A specific vendor scheme, not further decodable through the view.
    Scheme = 2,
}

/// Vendor tile-compression scheme in force for a tensor's allocation.
///
/// A result, not a request: [`EfImageDescView::compression`] carries what an
/// allocation asked for, and this names the scheme the allocator resolved to.
/// `None` means a linear layout.
///
/// Mirrors `edgefirst_tensor::CompressionScheme`'s variants, plus the `None`
/// the Rust side spells as `Option::None`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EfCompression {
    /// Linear layout: no vendor tile compression.
    #[default]
    None = 0,
    /// Qualcomm Adreno Universal Bandwidth Compression.
    Ubwc = 1,
    /// Arm Mali/Immortalis Framebuffer Compression.
    Afbc = 2,
    /// Imagination PowerVR Image Compression.
    Pvric = 3,
    /// Samsung Xclipse (AMD RDNA) Delta Color Compression.
    Dcc = 4,
}

/// Whether a tensor's memory is known to be physically contiguous
/// (`ef_tensor_contiguity`).
///
/// Mirrors `edgefirst_tensor::Contiguity`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EfContiguity {
    /// Not known: an imported fd, or memory not allocated from a DMA heap.
    #[default]
    Unknown = 0,
    /// Allocated from the CMA heap.
    Contiguous = 1,
    /// Allocated from the system heap.
    NonContiguous = 2,
}

/// Which kind of failure the calling thread's last failing `ef_tensor_*` call
/// was. The companion to `ef_tensor_last_error_message`.
///
/// A class, not a one-to-one mirror of `edgefirst_tensor::Error`: several Rust
/// variants collapse into one here. `Unspecified` is reported when a failure
/// path recorded only a message, and every such path resets the class to it.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EfErrorClass {
    /// No class recorded for this failure; the message is all there is.
    #[default]
    Unspecified = 0,
    /// A caller argument was malformed, out of range, or not recognized.
    InvalidArgument = 1,
    /// A shape was rejected: wrong rank, wrong element count, or not a
    /// shape the format admits.
    InvalidShape = 2,
    /// A shape or window did not fit the allocation behind it.
    InsufficientCapacity = 3,
    /// An index was outside the leading (batch) dimension.
    BatchIndexOutOfBounds = 4,
    /// A spatial region did not fit inside its parent frame.
    RegionOutOfBounds = 5,
    /// The operation is not available for this tensor's backing, on this
    /// platform, or in this build.
    NotSupported = 6,
    /// The operation is legal but not permitted right now: a live map, a
    /// shared handle, a lock another holder owns.
    InvalidOperation = 7,
    /// An allocation failed, or a syscall backing one did.
    AllocationFailed = 8,
    /// A quantization payload was rejected.
    QuantizationInvalid = 9,
}

/// Flattened, `#[repr(C)]` view of an image-request descriptor's fields.
///
/// `ef_tensor_image_desc` is opaque; this is the scalar block
/// `ef_tensor_image_desc_get` fills.
///
/// `memory` and `compression` are each a value plus an explicit presence flag.
/// `compression` is an [`EfCompressionRequest`]: `Any`, or `Scheme` for a
/// specific vendor scheme, which this view does not decode further.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EfImageDescView {
    /// Requested width in pixels.
    pub width: u64,
    /// Requested height in pixels.
    pub height: u64,
    /// The requested pixel format's wire code (`PixelFormat::code()`).
    pub format: u32,
    /// `ef_dtype`.
    pub dtype: u32,
    /// `ef_cpu_access`.
    pub access: u32,
    /// `ef_storage_kind`, meaningful only when `has_memory != 0`.
    pub memory: u32,
    /// Non-zero when a specific memory backing was requested (`None` on the
    /// Rust side auto-selects).
    pub has_memory: u32,
    /// `ef_compression_request`, meaningful only when `has_compression != 0`.
    pub compression: u32,
    /// Non-zero when a compression request was made.
    pub has_compression: u32,
}

/// Parent-region snapshot for a tensor that is a `view`/`batch` sub-region.
///
/// The scalar block `ef_tensor_view_origin` fills. The other fields are
/// meaningful only when `has_origin` is non-zero. Mirrors
/// `edgefirst_tensor::ViewOrigin`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EfViewOrigin {
    /// Logical width of the root parent image, in pixels.
    pub parent_width: u64,
    /// Logical height of the root parent image, in pixels.
    pub parent_height: u64,
    /// The parent's row stride in bytes.
    pub parent_row_stride: u64,
    /// This view's top-left x origin within the root parent, in pixels.
    pub x: u64,
    /// This view's top-left y origin within the root parent, in pixels.
    pub y: u64,
    /// Non-zero when this tensor is a view/batch sub-region; 0 for a whole
    /// tensor, in which case the other fields are all zero and unused.
    pub has_origin: u32,
}

/// The D3D11 texture behind a Windows texture tensor.
///
/// The scalar block `ef_tensor_d3d11_layout` fills. Mirrors
/// `edgefirst_tensor::d3d11_layout::D3d11ImageLayout`.
///
/// The dimensions are the texture's, in texels and rows, not the image's in
/// pixels: a combined-plane semi-planar image is one texture whose row count
/// covers both planes, and a YUYV image is one texel per two pixels. Read the
/// image's own dimensions from `ef_tensor_shape`.
///
/// An NV12 image of even width and height is a native two-plane texture,
/// `dxgi_format` 103 (`DXGI_FORMAT_NV12`): `texture_width` x `texture_height`
/// is the luma grid, the image's own size, and the chroma plane is a second
/// plane of the same texture at half each dimension, which GL and video APIs
/// address by plane index. A CPU map of it is laid out as luma rows then
/// chroma rows at `ef_tensor_row_stride`.
///
/// Every other semi-planar image (`nv16`, `nv24`, and `nv12` at an odd size)
/// is the combined-plane `R8` texture, whose `texture_width` is the driver's
/// row pitch: at least `even(width)`, and possibly more. It is the pitch the
/// combined plane's rows are spaced by and the width a sampler addresses the
/// texture at. `ef_tensor_row_stride` carries the same number.
///
/// Passed by value; its size and field offsets are fixed.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EfD3d11Layout {
    /// `DXGI_FORMAT` the texture was created with.
    pub dxgi_format: u32,
    /// Texture width in texels.
    pub texture_width: u32,
    /// Texture height in rows.
    pub texture_height: u32,
    /// Bytes per texel of `dxgi_format`.
    pub bytes_per_texel: u32,
    /// The GL internal format an importer binds this texture as; 0 when the
    /// format has no GL equivalent.
    pub gl_internal_format: u32,
}

/// Presence and shape summary of a tensor's quantization metadata.
///
/// The first half of the two-call idiom `ef_tensor_quantization_info` /
/// `ef_tensor_quantization_get` use: this scalar block reports whether
/// quantization is attached and how many entries the second call's buffers
/// need.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct EfQuantizationInfo {
    /// Channel axis for per-channel quantization; `-1` for per-tensor (no
    /// axis). Meaningful only when `has_quantization != 0`.
    pub axis: i32,
    /// Number of entries in the `scale`/`zero_point` arrays (1 for
    /// per-tensor). Meaningful only when `has_quantization != 0`.
    pub count: u32,
    /// Non-zero when this tensor carries quantization metadata; 0 when it
    /// does not, in which case `axis` and `count` are both zeroed and
    /// unused.
    pub has_quantization: u32,
}

/// A mapped CPU window over a tensor's bytes.
///
/// Passed by value; its size is fixed. The pointer is valid from
/// `ef_tensor_map` until the matching `ef_tensor_unmap`; writing through it is
/// allowed only when the map was taken with a writable access.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfTensorView {
    pub ptr: *mut u8,
    pub len: usize,
}

/// A retain or release call on a client-owned callback context.
///
/// Must be safe to call from any thread and must not unwind.
pub type EfClientRefFn = unsafe extern "C" fn(ctx: *const core::ffi::c_void);

/// Client-owned state a library entry point calls back into. Passed by value;
/// its size and field offsets are fixed.
///
/// # What `retain`/`release` govern
///
/// They extend the life of the callback channel only. They never transfer
/// ownership of the GL buffer, the CUDA resource, or anything else the context
/// addresses:
///
/// * The producing side's own destructor stays the sole caller of the real
///   `delete_buffer`.
/// * A library that reconstructed operations from this struct treats
///   `delete_buffer` as a no-op: it borrows the resource.
///
/// # Ownership at an entry point
///
/// An entry point that stores this struct calls `retain(ctx)` itself and calls
/// `release(ctx)` exactly once when the object holding it is destroyed. The
/// caller keeps its own count and may drop it at any time after the call
/// returns.
///
/// `ctx` is opaque to the library: only the producing copy of the code
/// dereferences it.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct EfClientState {
    /// Opaque to every caller except this struct's own function pointers.
    pub ctx: *const core::ffi::c_void,
    /// Add one reference to the callback channel. Never NULL in a valid
    /// state; an entry point receiving NULL must fail with `EINVAL`.
    pub retain: Option<unsafe extern "C" fn(ctx: *const core::ffi::c_void)>,
    /// Drop one reference to the callback channel. Never NULL in a valid
    /// state; an entry point receiving NULL must fail with `EINVAL`.
    pub release: Option<unsafe extern "C" fn(ctx: *const core::ffi::c_void)>,
}

/// Map a PBO for CPU access. `0` on success; `-1` when the GL context is
/// gone; any other non-zero value a generic failure.
pub type EfPboMapFn = unsafe extern "C" fn(
    ctx: *const core::ffi::c_void,
    buffer_id: u32,
    size: usize,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
) -> core::ffi::c_int;

/// Unmap a PBO previously mapped by an [`EfPboMapFn`]. Same return codes.
pub type EfPboUnmapFn =
    unsafe extern "C" fn(ctx: *const core::ffi::c_void, buffer_id: u32) -> core::ffi::c_int;

/// Map a PBO for CPU access. `0` on success; `-1` when the GL context is
/// gone; any other non-zero value a generic failure.
///
/// May be NULL where an entry point takes one; such an entry point refuses a
/// NULL.
pub type EfPboMapFnNullable = Option<
    unsafe extern "C" fn(
        ctx: *const core::ffi::c_void,
        buffer_id: u32,
        size: usize,
        out_ptr: *mut *mut u8,
        out_len: *mut usize,
    ) -> core::ffi::c_int,
>;

/// Unmap a PBO previously mapped by an `ef_pbo_map_fn`. Same return codes.
///
/// May be NULL where an entry point takes one; such an entry point refuses a
/// NULL.
pub type EfPboUnmapFnNullable =
    Option<unsafe extern "C" fn(ctx: *const core::ffi::c_void, buffer_id: u32) -> core::ffi::c_int>;

/// Map a CUDA graphics resource to a device pointer. `0` on success.
pub type EfCudaMapFn = unsafe extern "C" fn(
    ctx: *const core::ffi::c_void,
    resource: *mut core::ffi::c_void,
    out_ptr: *mut *mut core::ffi::c_void,
    out_len: *mut usize,
) -> core::ffi::c_int;

/// Unmap a resource previously mapped by an [`EfCudaMapFn`].
pub type EfCudaUnmapFn =
    unsafe extern "C" fn(ctx: *const core::ffi::c_void, resource: *mut core::ffi::c_void);

/// Unregister a CUDA graphics resource.
pub type EfCudaUnregisterFn =
    unsafe extern "C" fn(ctx: *const core::ffi::c_void, resource: *mut core::ffi::c_void);

/// Map a CUDA graphics resource to a device pointer. `0` on success.
///
/// May be NULL where an entry point takes one; such an entry point refuses a
/// NULL.
pub type EfCudaMapFnNullable = Option<
    unsafe extern "C" fn(
        ctx: *const core::ffi::c_void,
        resource: *mut core::ffi::c_void,
        out_ptr: *mut *mut core::ffi::c_void,
        out_len: *mut usize,
    ) -> core::ffi::c_int,
>;

/// Unmap a resource previously mapped by an `ef_cuda_map_fn`.
///
/// May be NULL where an entry point takes one; such an entry point refuses a
/// NULL.
pub type EfCudaUnmapFnNullable =
    Option<unsafe extern "C" fn(ctx: *const core::ffi::c_void, resource: *mut core::ffi::c_void)>;

/// Unregister a CUDA graphics resource.
///
/// May be NULL where an entry point takes one; such an entry point refuses a
/// NULL.
pub type EfCudaUnregisterFnNullable =
    Option<unsafe extern "C" fn(ctx: *const core::ffi::c_void, resource: *mut core::ffi::c_void)>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_layout_is_the_48_byte_scalar_block_the_blob_assumes() {
        // The blob's 56-byte plane record is this 48-byte scalar block plus
        // two 4-byte sequence counts.
        assert_eq!(std::mem::size_of::<EfTensorPlane>(), 48);
        assert_eq!(std::mem::align_of::<EfTensorPlane>(), 8);
    }

    #[test]
    fn vocabulary_enums_are_u32() {
        assert_eq!(std::mem::size_of::<EfDtype>(), 4);
        assert_eq!(std::mem::size_of::<EfStorageKind>(), 4);
        assert_eq!(std::mem::size_of::<EfCompressionRequest>(), 4);
    }

    #[test]
    fn tensor_view_is_two_pointers_wide_and_frozen() {
        assert_eq!(std::mem::size_of::<EfTensorView>(), 16);
        assert_eq!(std::mem::align_of::<EfTensorView>(), 8);
    }

    #[test]
    fn image_desc_view_layout_is_pinned() {
        // Two `u64`s and seven `u32`s (28 bytes) pad out to 48 bytes at align 8.
        assert_eq!(std::mem::size_of::<EfImageDescView>(), 48);
        assert_eq!(std::mem::align_of::<EfImageDescView>(), 8);
    }

    #[test]
    fn view_origin_layout_is_pinned() {
        // Five `u64`s plus a trailing `u32` pad out to 48 at align 8.
        assert_eq!(std::mem::size_of::<EfViewOrigin>(), 48);
        assert_eq!(std::mem::align_of::<EfViewOrigin>(), 8);
    }

    #[test]
    fn error_class_is_u32_and_unspecified_is_zero() {
        assert_eq!(std::mem::size_of::<EfErrorClass>(), 4);
        assert_eq!(EfErrorClass::Unspecified as u32, 0);
        assert_eq!(EfErrorClass::default(), EfErrorClass::Unspecified);
    }

    #[test]
    fn compression_is_u32_and_none_is_zero() {
        assert_eq!(std::mem::size_of::<EfCompression>(), 4);
        assert_eq!(std::mem::size_of::<EfContiguity>(), 4);
        assert_eq!(EfCompression::None as u32, 0);
        assert_eq!(EfCompression::default(), EfCompression::None);
    }

    #[test]
    fn cpu_access_is_u32() {
        assert_eq!(std::mem::size_of::<EfCpuAccess>(), 4);
    }

    #[test]
    fn d3d11_layout_is_pinned() {
        // Five 4-byte fields: 20 bytes, 4-byte aligned.
        assert_eq!(std::mem::size_of::<EfD3d11Layout>(), 20);
        assert_eq!(std::mem::align_of::<EfD3d11Layout>(), 4);
    }

    #[test]
    fn quantization_info_layout_is_pinned() {
        // Three 4-byte fields: 12 bytes, 4-byte aligned.
        assert_eq!(std::mem::size_of::<EfQuantizationInfo>(), 12);
        assert_eq!(std::mem::align_of::<EfQuantizationInfo>(), 4);
    }

    #[test]
    fn a_nullable_pbo_op_is_one_pointer_and_matches_its_bare_alias() {
        // The nullable aliases are the size of a plain function pointer.
        assert_eq!(
            core::mem::size_of::<EfPboMapFnNullable>(),
            core::mem::size_of::<EfPboMapFn>()
        );
        assert_eq!(
            core::mem::size_of::<EfPboUnmapFnNullable>(),
            core::mem::size_of::<EfPboUnmapFn>()
        );
        assert_eq!(
            core::mem::size_of::<EfPboMapFnNullable>(),
            core::mem::size_of::<*const core::ffi::c_void>()
        );
        // `Some(f)` of the bare alias is the nullable alias: these assignments compile.
        unsafe extern "C" fn m(
            _: *const core::ffi::c_void,
            _: u32,
            _: usize,
            _: *mut *mut u8,
            _: *mut usize,
        ) -> core::ffi::c_int {
            0
        }
        unsafe extern "C" fn u(_: *const core::ffi::c_void, _: u32) -> core::ffi::c_int {
            0
        }
        let f: EfPboMapFn = m;
        let _: EfPboMapFnNullable = Some(f);
        let g: EfPboUnmapFn = u;
        let _: EfPboUnmapFnNullable = Some(g);
    }

    #[test]
    fn client_state_layout_is_pinned() {
        // Three pointer-sized members at align 8.
        assert_eq!(std::mem::size_of::<EfClientState>(), 24);
        assert_eq!(std::mem::align_of::<EfClientState>(), 8);
        assert_eq!(std::mem::offset_of!(EfClientState, ctx), 0);
        assert_eq!(std::mem::offset_of!(EfClientState, retain), 8);
        assert_eq!(std::mem::offset_of!(EfClientState, release), 16);
    }

    #[test]
    fn a_nullable_client_ref_fn_is_pointer_sized() {
        // `Option<extern "C" fn(..)>` is pointer-sized.
        assert_eq!(
            std::mem::size_of::<Option<EfClientRefFn>>(),
            std::mem::size_of::<*const core::ffi::c_void>()
        );
        // The field's type is `Option<EfClientRefFn>`: this assignment compiles.
        let _: Option<EfClientRefFn> = EfClientState {
            ctx: core::ptr::null(),
            retain: None,
            release: None,
        }
        .retain;
    }

    #[test]
    fn this_crate_has_no_dependencies_and_never_may() {
        // The `[dependencies]` table is empty.
        let manifest = include_str!("../Cargo.toml");
        assert!(
            !manifest.contains("[dependencies."),
            "dependency sub-tables are dependencies too"
        );
        let deps = manifest
            .split("[dependencies]")
            .nth(1)
            .expect("manifest has a [dependencies] table");
        let table_body = deps.split("\n[").next().unwrap_or(deps);
        let non_comment_lines: Vec<&str> = table_body
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect();
        assert!(
            non_comment_lines.is_empty(),
            "edgefirst-tensor-abi grew dependencies: {non_comment_lines:?}"
        );
    }
}
