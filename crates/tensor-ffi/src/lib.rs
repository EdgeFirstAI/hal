// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Declarations-only bindings to `libedgefirst_tensor` — the Rust "header".
//!
//! This crate contains no code, only undefined symbols. It is one side of a
//! two-sided contract whose other side is `edgefirst-tensor-capi`'s exports;
//! `tensor-capi/tests/check_abi.rs` asserts every declaration here is a real
//! export there, because a declaration without an export is a load-time
//! failure waiting in every consumer. Linking is the consumer's decision:
//! there is deliberately no `#[link]` attribute here.
//!
//! Every declaration is transcribed from the generated
//! `crates/tensor-capi/include/edgefirst/tensor.h`, which is the authority:
//! when a signature here and the header disagree, the header wins and this
//! file is wrong.

use std::ffi::{c_char, c_int};

pub use edgefirst_tensor_abi::{
    EfCompression, EfCpuAccess, EfDtype, EfErrorClass, EfImageDescView, EfQuantizationInfo,
    EfStorageKind, EfTensorPlane, EfTensorView, EfViewOrigin,
};

/// Opaque tensor handle (never dereferenced, sized, or copied).
#[repr(C)]
pub struct EfTensor {
    _opaque: [u8; 0],
}

/// Opaque builder handle.
#[repr(C)]
pub struct EfTensorBuilder {
    _opaque: [u8; 0],
}

/// Opaque image-request handle.
#[repr(C)]
pub struct EfTensorImageDesc {
    _opaque: [u8; 0],
}

/// Declares the extern functions and records their names in [`DECLARED`],
/// so the conformance test cannot drift from the declarations.
macro_rules! declare_abi {
    ($(pub fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?;)*) => {
        extern "C" {
            $(pub fn $name($($arg: $ty),*) $(-> $ret)?;)*
        }
        /// Every symbol this crate declares, for the check-abi gate.
        pub const DECLARED: &[&str] = &[$(stringify!($name)),*];
    };
}

declare_abi! {
    pub fn ef_tensor_abi_version() -> u32;

    pub fn ef_tensor_builder_new() -> *mut EfTensorBuilder;
    pub fn ef_tensor_builder_free(b: *mut EfTensorBuilder);
    pub fn ef_tensor_builder_error(b: *const EfTensorBuilder) -> c_int;
    pub fn ef_tensor_builder_dtype(b: *mut EfTensorBuilder, dtype: u32) -> c_int;
    pub fn ef_tensor_builder_shape(
        b: *mut EfTensorBuilder,
        dims: *const u64,
        ndim: u32,
    ) -> c_int;
    pub fn ef_tensor_builder_strides(
        b: *mut EfTensorBuilder,
        strides: *const i64,
        ndim: u32,
    ) -> c_int;
    pub fn ef_tensor_builder_storage(b: *mut EfTensorBuilder, kind: u32) -> c_int;
    pub fn ef_tensor_builder_add_plane(
        b: *mut EfTensorBuilder,
        handle: i64,
        offset: u64,
        stride: u64,
        size: u64,
        used: u64,
        modifier: u64,
    ) -> c_int;
    pub fn ef_tensor_builder_format(b: *mut EfTensorBuilder, f: *const c_char) -> c_int;
    pub fn ef_tensor_builder_colorimetry(
        b: *mut EfTensorBuilder,
        space: *const c_char,
        transfer: *const c_char,
        encoding: *const c_char,
        range: *const c_char,
    ) -> c_int;
    pub fn ef_tensor_builder_quantization(
        b: *mut EfTensorBuilder,
        axis: i32,
        scales: *const f32,
        zps: *const i32,
        n: u32,
    ) -> c_int;
    pub fn ef_tensor_builder_fence(b: *mut EfTensorBuilder, fd: c_int) -> c_int;
    pub fn ef_tensor_builder_alloc(b: *mut EfTensorBuilder) -> *mut EfTensor;
    pub fn ef_tensor_builder_wrap(b: *mut EfTensorBuilder) -> *mut EfTensor;

    pub fn ef_tensor_image_desc_new(
        width: usize,
        height: usize,
        format: *const c_char,
        dtype: u32,
    ) -> *mut EfTensorImageDesc;
    pub fn ef_tensor_image_desc_free(d: *mut EfTensorImageDesc);
    pub fn ef_tensor_image_desc_set_memory(d: *mut EfTensorImageDesc, kind: u32) -> c_int;
    pub fn ef_tensor_image_desc_set_access(d: *mut EfTensorImageDesc, access: u32) -> c_int;
    pub fn ef_tensor_image_desc_set_compression(
        d: *mut EfTensorImageDesc,
        compression: u32,
    ) -> c_int;
    pub fn ef_tensor_image_desc_get(
        d: *const EfTensorImageDesc,
        out: *mut EfImageDescView,
    ) -> c_int;
    pub fn ef_tensor_image_desc_alloc(d: *const EfTensorImageDesc) -> *mut EfTensor;

    pub fn ef_tensor_image_alloc(
        width: usize,
        height: usize,
        format: *const c_char,
        dtype: u32,
        has_memory: c_int,
        memory: u32,
        access: u32,
    ) -> *mut EfTensor;
    pub fn ef_tensor_image_with_stride_alloc(
        width: usize,
        height: usize,
        format: *const c_char,
        dtype: u32,
        row_stride_bytes: usize,
        has_memory: c_int,
        memory: u32,
        access: u32,
    ) -> *mut EfTensor;
    pub fn ef_tensor_batch(t: *const EfTensor, n: u64) -> *mut EfTensor;
    pub fn ef_tensor_wrap_host(
        ptr: *mut u8,
        capacity: usize,
        dtype: u32,
        dims: *const u64,
        ndim: u32,
    ) -> *mut EfTensor;
    pub fn ef_tensor_from_iosurface_id(
        id: u32,
        dtype: u32,
        dims: *const u64,
        ndim: u32,
    ) -> *mut EfTensor;
    pub fn ef_tensor_view_region(
        t: *const EfTensor,
        x: u64,
        y: u64,
        width: u64,
        height: u64,
    ) -> *mut EfTensor;
    pub fn ef_tensor_from_planes(
        luma: *mut EfTensor,
        chroma: *mut EfTensor,
        format: *const c_char,
    ) -> *mut EfTensor;

    pub fn ef_tensor_last_error_message() -> *const c_char;
    pub fn ef_tensor_last_error_class() -> u32;
    pub fn ef_tensor_map(t: *mut EfTensor, access: u32, out: *mut EfTensorView) -> c_int;
    pub fn ef_tensor_unmap(t: *mut EfTensor) -> c_int;
    pub fn ef_tensor_sync_for_cpu(t: *const EfTensor, access: u32) -> c_int;
    pub fn ef_tensor_sync_for_device(t: *const EfTensor, access: u32) -> c_int;
    pub fn ef_tensor_copy_to(t: *mut EfTensor, out: *mut u8, cap: usize) -> i64;

    pub fn ef_tensor_set_format(t: *mut EfTensor, format: *const c_char) -> c_int;
    pub fn ef_tensor_set_row_stride(t: *mut EfTensor, stride: usize) -> c_int;
    pub fn ef_tensor_set_row_stride_unchecked(t: *mut EfTensor, stride: usize) -> c_int;
    pub fn ef_tensor_set_dtype(t: *mut EfTensor, dtype: u32) -> c_int;
    pub fn ef_tensor_reshape(t: *mut EfTensor, dims: *const u64, ndim: u32) -> c_int;
    pub fn ef_tensor_set_logical_shape(
        t: *mut EfTensor,
        dims: *const u64,
        ndim: u32,
    ) -> c_int;
    pub fn ef_tensor_clone_fd(t: *const EfTensor) -> c_int;
    pub fn ef_tensor_set_plane_offset(t: *mut EfTensor, offset: usize) -> c_int;
    pub fn ef_tensor_plane_offset(t: *const EfTensor) -> i64;
    pub fn ef_tensor_configure_image(
        t: *mut EfTensor,
        width: usize,
        height: usize,
        format: *const c_char,
    ) -> c_int;

    pub fn ef_tensor_quantization_info(
        t: *const EfTensor,
        out: *mut EfQuantizationInfo,
    ) -> c_int;
    pub fn ef_tensor_quantization_get(
        t: *const EfTensor,
        scales: *mut f32,
        zps: *mut i32,
        n: u32,
    ) -> c_int;
    pub fn ef_tensor_quantization_set(
        t: *mut EfTensor,
        axis: i32,
        scales: *const f32,
        zps: *const i32,
        n: u32,
    ) -> c_int;
    pub fn ef_tensor_quantization_clear(t: *mut EfTensor) -> c_int;

    pub fn ef_tensor_export(
        t: *const EfTensor,
        blob_out: *mut u8,
        blob_cap: usize,
        blob_len: *mut usize,
        fds_out: *mut c_int,
        fds_cap: usize,
        fds_len: *mut usize,
    ) -> c_int;
    pub fn ef_tensor_import(
        blob_in: *const u8,
        blob_len: usize,
        fds_in: *const c_int,
        fds_len: usize,
    ) -> *mut EfTensor;

    pub fn ef_tensor_ndim(t: *const EfTensor) -> u32;
    pub fn ef_tensor_shape(t: *const EfTensor) -> *const u64;
    pub fn ef_tensor_strides(t: *const EfTensor) -> *const i64;
    pub fn ef_tensor_dtype(t: *const EfTensor) -> u32;
    pub fn ef_tensor_storage_kind(t: *const EfTensor) -> u32;
    pub fn ef_tensor_plane_count(t: *const EfTensor) -> u32;
    pub fn ef_tensor_format(t: *const EfTensor) -> *const c_char;
    pub fn ef_tensor_plane_at(
        t: *const EfTensor,
        index: u32,
        out: *mut EfTensorPlane,
    ) -> c_int;
    pub fn ef_tensor_capacity_bytes(t: *const EfTensor) -> i64;
    pub fn ef_tensor_row_stride(t: *const EfTensor) -> i64;
    pub fn ef_tensor_compression(t: *const EfTensor) -> u32;
    pub fn ef_tensor_colorimetry(t: *const EfTensor) -> u32;
    pub fn ef_tensor_set_colorimetry(t: *mut EfTensor, packed: u32) -> c_int;
    pub fn ef_tensor_view_origin(t: *const EfTensor, out: *mut EfViewOrigin) -> c_int;

    pub fn ef_tensor_new(dtype: u32, dims: *const u64, ndim: u32) -> *mut EfTensor;
    pub fn ef_tensor_free(t: *mut EfTensor);
    pub fn ef_tensor_retain(t: *mut EfTensor) -> c_int;

    pub fn ef_is_cuda_available() -> c_int;
    pub fn ef_is_dma_available() -> c_int;
    pub fn ef_is_gpu_buffer_available() -> c_int;
    pub fn ef_is_iosurface_available() -> c_int;
    pub fn ef_is_shm_available() -> c_int;
    pub fn ef_platform_compression_support(format: *const c_char, dtype: u32) -> c_int;
    pub fn ef_compression_fallback_count() -> u64;
    pub fn ef_unplanned_cpu_access_count() -> u64;

    pub fn ef_tensor_cuda_map(t: *const EfTensor) -> *mut std::ffi::c_void;
    pub fn ef_tensor_cuda_device_ptr(
        map: *const std::ffi::c_void,
        out_size: *mut usize,
    ) -> *mut std::ffi::c_void;
    pub fn ef_tensor_cuda_unmap(map: *mut std::ffi::c_void);

    pub fn ef_tensor_from_hardware_buffer(
        dtype: u32,
        buffer: *mut std::ffi::c_void,
        dims: *const u64,
        ndim: u32,
        name: *const c_char,
    ) -> *mut EfTensor;
    pub fn ef_tensor_hardware_buffer_ptr(t: *const EfTensor) -> *mut std::ffi::c_void;
    pub fn ef_tensor_hardware_buffer_physical_dims(
        t: *const EfTensor,
        width: *mut usize,
        height: *mut usize,
    ) -> c_int;
    pub fn ef_tensor_iosurface_ref(t: *const EfTensor) -> *mut std::ffi::c_void;
    pub fn ef_tensor_name(t: *const EfTensor) -> *mut c_char;

    pub fn ef_start_tracing(path: *const c_char) -> c_int;
    pub fn ef_stop_tracing();
    pub fn ef_is_tracing_active() -> c_int;
    pub fn ef_log_init_file(stream: *mut std::ffi::c_void, max_level: u32) -> c_int;
    pub fn ef_log_init_callback(
        cb: Option<
            unsafe extern "C" fn(u32, *const c_char, *const c_char, *mut std::ffi::c_void),
        >,
        userdata: *mut std::ffi::c_void,
        max_level: u32,
    ) -> c_int;
}

#[cfg(test)]
mod tests {
    use super::DECLARED;

    /// Pinned against `include/edgefirst/tensor.h`: 57 function declarations
    /// (`grep -cE '^[A-Za-z].*ef_.*\(' include/edgefirst/tensor.h` from
    /// `crates/tensor-capi`). History, since this number moves for reasons
    /// that look identical at a glance but are not: it was 38, then 32 when
    /// the detection-list family (`ef_detect_box_list_{new,push,len,get,
    /// data,free}`) moved OUT to `edgefirst-decoder-capi`, its one
    /// implementation home; then 37 when the `ef_tensor_image_desc_{new,
    /// free,set_memory,set_access,set_compression}` family moved IN from
    /// `edgefirst-image-capi`, this being `ImageDesc`'s one implementation
    /// home instead; then 38 again with `ef_tensor_image_desc_get` added --
    /// the scalar getter that lets a foreign library read a request without
    /// dereferencing the opaque handle; then 41 with `ef_tensor_colorimetry`,
    /// `ef_tensor_set_colorimetry`, and `ef_tensor_view_origin` added -- the
    /// three genuinely-new primitives `docs/superpowers/plans/
    /// PRIMITIVE-INVENTORY.md` found the dynamic backend needs beyond what
    /// was already exported; now 54 with the task-15 primitive families that
    /// close the dynamic backend's stub gaps with live callers: 4 metadata
    /// mutators (`ef_tensor_set_format`, `_set_row_stride`,
    /// `_set_plane_offset`, `_configure_image` -- family 1 + 3, `import_image`
    /// and JPEG decode-into-pool), 4 quantization primitives
    /// (`ef_tensor_quantization_{info,get,set,clear}` -- family 2, the
    /// decoder's int8 pipeline, `info`/`get` splitting the read into the
    /// two-call idiom `Quantization`'s variable length requires), 4
    /// construction/view primitives (`ef_tensor_image_alloc`,
    /// `_image_with_stride_alloc`, `_view_region`, `_image_desc_alloc` --
    /// family 4, image tiling and the V4L2 JPEG decoder's destination
    /// allocation), and 1 consuming constructor (`ef_tensor_from_planes` --
    /// family 5, the two-fd import path); then 55 with `ef_tensor_plane_offset`
    /// added -- family 1's setter had no reader at all
    /// (`ef_tensor_plane_at`'s plane-0 offset is `plane_table`'s
    /// intra-buffer layout, always 0 for plane 0, a different quantity
    /// entirely), found while proving the setter through the dynamic
    /// backend and fixed in the same task rather than left as a second,
    /// undiscovered gap. See `.superpowers/sdd/
    /// 2026-08-24-single-tensor-home/task-15-report.md` for the derivation
    /// of each. Then 56 with `ef_tensor_set_row_stride_unchecked` added --
    /// task 17's predicate-divergence audit found `Tensor::
    /// set_row_stride_unchecked` (the raw, format-independent stride setter
    /// a multiplane chroma sub-tensor needs, since `from_planes` requires it
    /// to carry no format) had a `dynamic`-side stub that only ever
    /// panicked; the earlier 55 primitives covered every *formatted* stride
    /// path but never this formatless one. See `.superpowers/sdd/
    /// 2026-08-24-single-tensor-home/task-17-report.md`. Briefly 57 with
    /// `ef_tensor_is_native` added -- task 9's own G3 harness (the
    /// two-library JPEG-into-host-memory-tensor test) found that a `-capi`
    /// leaf which had just switched to linking this library dynamically had
    /// no safe way to tell "a destination handle this library itself
    /// minted, now reachable through a real `ef_tensor_*` call" apart from
    /// "a destination from a still-static sibling's private, structurally
    /// different `EfTensorImpl` layout" -- the two invariant-2 transition
    /// guards on `ef_tensor_map`/`_unmap` (own-mint only) meant every
    /// caller-supplied host-memory destination fell through to the
    /// fd-based foreign-re-import path, which cannot work at all without a
    /// shareable handle. See `.superpowers/sdd/
    /// 2026-08-24-single-tensor-home/task-9-report.md`.
    /// **Back to 56** with `ef_tensor_is_native` deleted -- task 10 (this
    /// track's "delete the transition vtable") removed the vtable dispatch
    /// machinery `is_native` existed to distinguish a foreign handle from.
    /// All four sibling `-capi` leaves now link `libedgefirst_tensor.so`
    /// dynamically, so every handle in the process is the same layout from
    /// the same allocator; there is no more "foreign, structurally
    /// different" handle for a caller to tell apart, and the own-mint-only
    /// restriction on `ef_tensor_retain`/`_map`/`_unmap`/`_copy_to` is gone
    /// with it. See `.superpowers/sdd/2026-08-24-single-tensor-home/
    /// task-10-report.md`.
    /// `check_abi.rs`'s `the_ffi_declaration_count_matches_the_header`
    /// re-derives this count from the header on every run, so this constant
    /// cannot drift silently -- either both change together or that test
    /// fails.
    /// **Now 64** with task P2a's eight additions -- the primitives the
    /// dynamic backend needs to satisfy `edgefirst-python-common`'s
    /// `TensorDyn` call sites, which had never been compiled against this
    /// backend before: `ef_tensor_sync_for_cpu`/`_sync_for_device` (the
    /// standalone cache-maintenance bracket, `Tensor.cpu_access()`'s
    /// context manager -- distinct from `ef_tensor_map`, which establishes
    /// an address and the coherency window together), `ef_tensor_batch`
    /// (the leading-dimension view, distinct from `ef_tensor_view_region`'s
    /// spatial crop), and the three read-back accessors
    /// `ef_tensor_capacity_bytes`, `ef_tensor_row_stride` and
    /// `ef_tensor_compression` that `TensorDyn::descriptor_pinned` needs
    /// and no existing primitive could answer (`ef_tensor_plane_at` reports
    /// *effective* pitch and per-plane geometry, neither of which is the
    /// recorded stride or the allocation's byte count). See
    /// `.superpowers/sdd/2026-08-25-python-single-tensor-home/
    /// task-P2a-report.md`. The last two are `TensorDyn::
    /// import_descriptor`'s remaining two arms: `ef_tensor_wrap_host`
    /// (`kind::HOST` -- alias a producer's host pointer, carrying its real
    /// capacity so the import is not clamped to today's shape) and
    /// `ef_tensor_from_iosurface_id` (`kind::IOSURFACE`, declared on every
    /// platform and refused at runtime off Apple, so this library's symbol
    /// set does not vary by target). `kind::DMABUF` needed nothing new --
    /// `ef_tensor_builder_*` already covers it via `TensorDyn::from_fd` --
    /// and `kind::PBO` needed nothing at all, being in-process Rust state
    /// with no wire form.
    /// **Now 65** with `ef_tensor_set_dtype` (task P2b) -- retag a handle's
    /// element type without touching its bytes, refusing a width change.
    /// Under `static` the dtype lives in the Rust type, so
    /// `TensorDyn::from(Tensor<i8>)` after a layout-identical transmute is
    /// free; under `dynamic` it lives in the C handle, so the transmute
    /// changed a `PhantomData` and nothing else. That is what made
    /// `create_image(dtype="int8")` report `uint8` -- the only one of P2b's
    /// four regressions that returned a *wrong answer* rather than a
    /// refusal.
    /// **Now 67** with P2b's remaining two: `ef_tensor_reshape`, and
    /// `ef_tensor_clone_fd` -- the latter not derivable from
    /// `ef_tensor_plane_at`, whose *native handle* is a dma-buf fd or an
    /// IOSurface id and `-1` for everything else, so deriving "clone this
    /// tensor's fd" from it refused SHM-backed tensors, which do have one.
    ///
    /// A third, `ef_tensor_set_logical_shape`, was written and removed
    /// before commit: nothing in this workspace calls it, and the defect
    /// that motivated it turned out to be elsewhere (`Tensor<T>` never
    /// overrides `TensorTrait::set_logical_shape`, so the capacity-aware
    /// storage implementations are unreachable through it on *both*
    /// backends -- see task P2b's report). Adding ABI surface for a caller
    /// that does not exist is how an entry point drifts unwatched.
    /// **Now 68** with `ef_tensor_last_error_class` (task P2c): the class
    /// of the calling thread's last failure, beside the advisory message.
    ///
    /// It exists because an entry point that reports failure by returning
    /// `NULL` has no errno to carry a class, so a consumer rebuilding a
    /// typed error from one had only the message -- whose own contract says
    /// never to parse it. `ef_tensor_batch`'s wrapper was parsing it anyway.
    ///
    /// **One new symbol, not an out-param on each of the twelve
    /// NULL-returning entries.** An out-param would change those twelve
    /// signatures, which is not additive at all: an existing caller passing
    /// the old argument count to a function expecting one more is undefined
    /// behaviour, and the loader cannot see it. This follows the ABI's own
    /// `dlerror` convention instead -- the same thread-local, the same
    /// lifetime rules, one accessor -- and covers every NULL-returning
    /// entry at once. See task P2c's report for the enumeration.
    /// **Now 90** after the leftover monolith ports: capability probes,
    /// CUDA map, AHardwareBuffer/IOSurface, tracing, logging, and
    /// `ef_tensor_name`.
    const HEADER_DECLARATION_COUNT: usize = 90;

    #[test]
    fn declared_matches_the_header_derived_count() {
        assert_eq!(DECLARED.len(), HEADER_DECLARATION_COUNT);
    }
}
