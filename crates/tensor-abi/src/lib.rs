// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Layout-only C ABI contract for the EdgeFirst tensor surface.
//!
//! Plain `#[repr(C)]` values and the integer vocabularies, declared exactly
//! once. Every `-capi` crate imports these instead of carrying an "identical
//! copy" — r2's duplication was safe but unverified-by-construction; a single
//! declaration makes divergence impossible instead of merely absent.
//!
//! This crate has **no dependencies** and must never gain one. The vocabulary
//! values are asserted against `edgefirst-tensor`'s authority from the crates
//! that depend on both (see `edgefirst-tensor-capi/src/codes.rs`) — this
//! crate is emitted surface, never a second authority.

/// One plane's location, mirroring `TensorPlane` on the wire.
///
/// The route by which one library consumes a tensor another minted: the Rust
/// types are not shared across `.so` boundaries, so the planes are the
/// interface. Emitted into `edgefirst/tensor.h` as `ef_tensor_plane` by
/// cbindgen.
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
/// Every kind is declared on every platform. Whether one can be *materialised*
/// here is a runtime question — an IOSurface is a meaningful thing to name on
/// Linux even though nothing will allocate one — and platform-gating the
/// vocabulary would make the same integer mean different things per target.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfStorageKind {
    Mem = 0,
    Shm = 1,
    DmaBuf = 2,
    IoSurface = 3,
    Pbo = 4,
    Cuda = 5,
}

/// CPU access direction for a map window.
///
/// Mirrors `edgefirst_tensor::CpuAccess`'s semantics; the values are the wire
/// codes. `None` names the no-CPU-access declaration and is never a valid map
/// direction.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfCpuAccess {
    None = 0,
    Read = 1,
    Write = 2,
    ReadWrite = 3,
}

/// Vendor tile-compression scheme actually in force for a tensor's
/// allocation.
///
/// A *result*, not a request: [`EfImageDescView::compression`] carries what
/// an allocation asked for ("any scheme" / "a specific one"), whereas this
/// names the scheme the allocator actually resolved to. `None` is a real
/// enumerator rather than a presence flag because "linear" is the answer for
/// almost every allocation on almost every platform -- there is no absent
/// case to distinguish from it, unlike `EfViewOrigin`/`EfQuantizationInfo`
/// where every field value is legitimate.
///
/// Mirrors `edgefirst_tensor::CompressionScheme`'s variants, plus the
/// `None` the Rust side spells as `Option::None`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EfCompression {
    /// Linear layout -- no vendor tile compression.
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

/// Which *kind* of failure the calling thread's last failing
/// `ef_tensor_*` call was.
///
/// The companion to `ef_tensor_last_error_message`, and the reason it
/// exists: an entry point that reports failure by returning `NULL` has no
/// errno to carry a class, so a Rust consumer rebuilding a typed
/// `edgefirst_tensor::Error` from it had nothing to go on but the advisory
/// message -- which the message's own contract says must never be parsed.
/// `ef_tensor_batch` really did match on a fragment of a `Display` string
/// for exactly this reason.
///
/// A **class**, not a one-to-one mirror of `edgefirst_tensor::Error`. Its
/// variants are the distinctions a caller acts on differently; several Rust
/// variants deliberately collapse into one here, and adding a Rust variant
/// does not oblige adding one here. `Unspecified` is the honest answer when
/// a failure path recorded only a message, and is what every such path
/// resets this to -- a stale class from an earlier failure would be read as
/// this call's, which is the confident-falsehood shape this whole mechanism
/// exists to remove.
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
    /// The operation is legal but not permitted right now -- a live map, a
    /// shared handle, a lock another holder owns.
    InvalidOperation = 7,
    /// An allocation failed, or a syscall backing one did.
    AllocationFailed = 8,
    /// A quantization payload was rejected.
    QuantizationInvalid = 9,
}

/// Flattened, `#[repr(C)]` view of an image-request descriptor's fields.
///
/// `ef_tensor_image_desc` is opaque (handles are opaque both ways: a
/// receiving library never dereferences, sizes, or copies one). This is the
/// scalar block `ef_tensor_image_desc_get` fills instead -- the same shape as
/// `ef_tensor_plane`, which lets one library read a tensor it did not mint
/// without touching the other's private layout.
///
/// `memory` and `compression` are each a value plus an explicit presence
/// flag rather than a sentinel: every code in `ef_storage_kind` (0..=5) is a
/// real value, so there is no unused number to repurpose as "no request"
/// without colliding with `ef_storage_kind`'s `MEM == 0`. `compression` is 1
/// for "any scheme" and 2 for "a specific vendor scheme", the latter not
/// further decodable through this view -- no `ef_tensor_image_desc_set_*`
/// entry point can request one, so this view has never needed to carry more
/// detail than "present, and it's a specific one."
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
    /// 1 = any scheme the platform offers; 2 = a specific vendor scheme.
    /// Meaningful only when `has_compression != 0`.
    pub compression: u32,
    /// Non-zero when a compression request was made.
    pub has_compression: u32,
}

/// Parent-region snapshot for a tensor that is a `view`/`batch` sub-region.
///
/// The scalar block `ef_tensor_view_origin` fills, the same shape as
/// `ef_tensor_plane` and `ef_tensor_image_desc_view` -- one library reading a
/// tensor it did not mint. `has_origin` is a presence flag rather than a
/// sentinel value because every field is a legitimate 0 for a view pinned at
/// the parent's top-left corner; the other fields are meaningful only when
/// it is non-zero. Mirrors `edgefirst_tensor::ViewOrigin`.
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
/// The scalar block `ef_tensor_d3d11_layout` fills, the same shape as
/// `ef_tensor_plane` and `ef_tensor_view_origin` -- one library reading a
/// tensor it did not mint. Mirrors `edgefirst_tensor::d3d11_layout::
/// D3d11ImageLayout`.
///
/// The dimensions are the texture's, in texels and rows, not the image's in
/// pixels: a semi-planar image is one texture whose row count covers both
/// planes, and a YUYV image is one texel per two pixels, so
/// `texture_width`/`texture_height` do not match `ef_tensor_shape` for
/// either. Read the image's own dimensions from the shape.
///
/// For a semi-planar format (`nv12`, `nv16`, `nv24`) `texture_width` is the
/// driver's row pitch -- at least `even(width)`, and on a discrete adapter
/// commonly more (128 bytes on NVIDIA, so a 64-wide NV12 image is a 128-wide
/// texture). It is the pitch the combined plane's rows are spaced by and the
/// width a sampler must address the texture at. Never derive it from the
/// image width: read this field, or `ef_tensor_row_stride`, which carries the
/// same number.
///
/// By-value and frozen forever: a consumer bakes this size and these offsets
/// into its call sites, so the struct evolves by a suffixed successor, never
/// by an in-place edit. `d3d11_layout_is_pinned` and the C golden in
/// `tensor-capi/tests/c/test_layout_goldens.c` hold both sides to it.
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

/// Presence/shape summary of a tensor's quantization metadata.
///
/// The first half of the two-call idiom `ef_tensor_quantization_info` /
/// `ef_tensor_quantization_get` use: this scalar block tells the caller
/// *whether* quantization is attached and *how big* a buffer the second call
/// needs, without allocating on either side of the boundary --
/// `Quantization` itself is variable-length (an axis plus per-axis `scales`
/// and `zero_points`), so unlike `ef_tensor_plane`/`ef_tensor_view_origin`
/// this view cannot carry the payload itself.
///
/// `has_quantization` is a presence flag, not a sentinel, for the same
/// reason `EfViewOrigin::has_origin` is: axis `0` and a scale of `0.0` are
/// both legitimate values, so there is no unused bit pattern to repurpose as
/// "absent" without colliding with a real one.
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
/// By-value with no version field and no reserved tail, so its size
/// is baked into consumer call sites, so it evolves by a suffixed successor
/// (`ef_tensor_view2` + new entry points), never in place. The pointer is
/// valid from `ef_tensor_map` until the matching `ef_tensor_unmap`; writing
/// through it is allowed only when the map was taken with a writable access.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EfTensorView {
    pub ptr: *mut u8,
    pub len: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plane_layout_is_the_48_byte_scalar_block_the_blob_assumes() {
        // The blob's 56-byte plane record is this 48-byte scalar block plus
        // two 4-byte sequence counts (spec: Blob format, [r2] As built).
        assert_eq!(std::mem::size_of::<EfTensorPlane>(), 48);
        assert_eq!(std::mem::align_of::<EfTensorPlane>(), 8);
    }

    #[test]
    fn vocabulary_enums_are_u32() {
        assert_eq!(std::mem::size_of::<EfDtype>(), 4);
        assert_eq!(std::mem::size_of::<EfStorageKind>(), 4);
    }

    #[test]
    fn tensor_view_is_two_pointers_wide_and_frozen() {
        assert_eq!(std::mem::size_of::<EfTensorView>(), 16);
        assert_eq!(std::mem::align_of::<EfTensorView>(), 8);
    }

    #[test]
    fn image_desc_view_layout_is_pinned() {
        // Two `u64`s force 8-byte alignment, so the seven trailing `u32`s
        // (28 bytes) pad out to a 48-byte struct. Pinned so a field reorder
        // or an added field is a deliberate, visible change here.
        assert_eq!(std::mem::size_of::<EfImageDescView>(), 48);
        assert_eq!(std::mem::align_of::<EfImageDescView>(), 8);
    }

    #[test]
    fn view_origin_layout_is_pinned() {
        // Five `u64`s plus a trailing `u32` pad out to 48 at align 8. Same
        // freeze as the C golden: a field reorder is a new suffixed struct.
        assert_eq!(std::mem::size_of::<EfViewOrigin>(), 48);
        assert_eq!(std::mem::align_of::<EfViewOrigin>(), 8);
    }

    #[test]
    fn error_class_is_u32_and_unspecified_is_zero() {
        assert_eq!(std::mem::size_of::<EfErrorClass>(), 4);
        // 0 must be the "nothing recorded" answer: a zeroed or never-set
        // slot has to read as "no class", never as a real classification a
        // consumer would then act on.
        assert_eq!(EfErrorClass::Unspecified as u32, 0);
        assert_eq!(EfErrorClass::default(), EfErrorClass::Unspecified);
    }

    #[test]
    fn compression_is_u32_and_none_is_zero() {
        assert_eq!(std::mem::size_of::<EfCompression>(), 4);
        // `ef_tensor_compression` returns this by value, and a zeroed/failed
        // read must read as "linear", never as a scheme.
        assert_eq!(EfCompression::None as u32, 0);
        assert_eq!(EfCompression::default(), EfCompression::None);
    }

    #[test]
    fn cpu_access_is_u32() {
        assert_eq!(std::mem::size_of::<EfCpuAccess>(), 4);
    }

    #[test]
    fn d3d11_layout_is_pinned() {
        // Five 4-byte fields, no `u64` to force wider alignment -- 20 bytes,
        // 4-byte aligned, and the field order the C golden asserts offsets
        // for. A widened field or a reorder is a new suffixed struct.
        assert_eq!(std::mem::size_of::<EfD3d11Layout>(), 20);
        assert_eq!(std::mem::align_of::<EfD3d11Layout>(), 4);
    }

    #[test]
    fn quantization_info_layout_is_pinned() {
        // Three 4-byte fields, no `u64` to force wider alignment -- 12 bytes,
        // 4-byte aligned. Pinned so a field reorder or an added field is a
        // deliberate, visible change here.
        assert_eq!(std::mem::size_of::<EfQuantizationInfo>(), 12);
        assert_eq!(std::mem::align_of::<EfQuantizationInfo>(), 4);
    }

    #[test]
    fn this_crate_has_no_dependencies_and_never_may() {
        // Load-bearing for the whole design: tensor-ffi -> tensor-abi must never
        // drag an implementation into the dynamic backend (spec r3). The
        // [dependencies] table is empty and this test keeps it that way.
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
