// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Declarations-only bindings to `libedgefirst_tensor`.
//!
//! This crate contains no code, only undefined symbols, and carries no
//! `#[link]` attribute: linking is the consumer's decision.
//!
//! Every declaration is transcribed from the generated
//! `crates/tensor-capi/include/edgefirst/tensor.h`.

use std::ffi::{c_char, c_int};

pub use edgefirst_tensor_abi::{
    EfClientRefFn, EfClientState, EfCompression, EfCompressionRequest, EfContiguity, EfCpuAccess,
    EfCudaMapFn, EfCudaMapFnNullable, EfCudaUnmapFn, EfCudaUnmapFnNullable, EfCudaUnregisterFn,
    EfCudaUnregisterFnNullable, EfD3d11Layout, EfDtype, EfErrorClass, EfImageDescView, EfPboMapFn,
    EfPboMapFnNullable, EfPboUnmapFn, EfPboUnmapFnNullable, EfQuantizationInfo, EfStorageKind,
    EfTensorPlane, EfTensorView, EfViewOrigin,
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

/// Declares the extern functions and records their names in [`DECLARED`].
macro_rules! declare_abi {
    ($(pub fn $name:ident($($arg:ident: $ty:ty),* $(,)?) $(-> $ret:ty)?;)*) => {
        extern "C" {
            $(pub fn $name($($arg: $ty),*) $(-> $ret)?;)*
        }
        /// Every symbol this crate declares.
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
    pub fn ef_tensor_image_desc_set_contiguous(d: *mut EfTensorImageDesc, contiguous: c_int)
        -> c_int;
    pub fn ef_tensor_image_desc_contiguous(d: *const EfTensorImageDesc) -> c_int;
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
    pub fn ef_tensor_wrap_pbo(
        state: EfClientState,
        buffer_id: u32,
        size: usize,
        dtype: u32,
        dims: *const u64,
        ndim: u32,
        map_fn: EfPboMapFnNullable,
        unmap_fn: EfPboUnmapFnNullable,
    ) -> *mut EfTensor;
    pub fn ef_tensor_pbo_id(t: *const EfTensor, out_id: *mut u32) -> c_int;
    pub fn ef_tensor_pbo_is_mapped(t: *const EfTensor) -> c_int;
    pub fn ef_tensor_pbo_vtable(t: *const EfTensor) -> *const std::ffi::c_void;
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
    pub fn ef_tensor_is_multiplane(t: *const EfTensor) -> c_int;
    pub fn ef_tensor_chroma(t: *const EfTensor) -> *mut EfTensor;

    pub fn ef_tensor_last_error_message() -> *const c_char;
    pub fn ef_tensor_last_error_class() -> u32;
    pub fn ef_tensor_map(t: *mut EfTensor, access: u32, out: *mut EfTensorView) -> c_int;
    pub fn ef_tensor_try_map(t: *mut EfTensor, access: u32, out: *mut EfTensorView) -> c_int;
    pub fn ef_tensor_unmap(t: *mut EfTensor) -> c_int;
    pub fn ef_tensor_with_cpu_mappings_excluded(
        f: Option<unsafe extern "C" fn(ctx: *mut std::ffi::c_void)>,
        ctx: *mut std::ffi::c_void,
    );
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
    pub fn ef_tensor_cpu_access(t: *const EfTensor) -> u32;
    pub fn ef_tensor_restrict_cpu_access(t: *mut EfTensor, access: u32) -> c_int;
    pub fn ef_tensor_contiguity(t: *const EfTensor) -> u32;
    pub fn ef_tensor_colorimetry(t: *const EfTensor) -> u32;
    pub fn ef_tensor_set_colorimetry(t: *mut EfTensor, packed: u32) -> c_int;
    pub fn ef_tensor_view_origin(t: *const EfTensor, out: *mut EfViewOrigin) -> c_int;
    pub fn ef_tensor_is_view(t: *const EfTensor) -> c_int;
    pub fn ef_tensor_mark_view(t: *mut EfTensor) -> c_int;

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

    pub fn ef_tensor_cuda_map(t: *const EfTensor) -> *mut std::ffi::c_void;
    pub fn ef_tensor_cuda_map_mut(t: *const EfTensor) -> *mut std::ffi::c_void;
    pub fn ef_tensor_cuda_device_ptr(
        map: *const std::ffi::c_void,
        out_size: *mut usize,
    ) -> *mut std::ffi::c_void;
    pub fn ef_tensor_cuda_unmap(map: *mut std::ffi::c_void);
    pub fn ef_tensor_cuda_attach(
        t: *mut EfTensor,
        state: EfClientState,
        resource: *mut std::ffi::c_void,
        size: usize,
        map_fn: EfCudaMapFnNullable,
        unmap_fn: EfCudaUnmapFnNullable,
        unregister_fn: EfCudaUnregisterFnNullable,
    ) -> c_int;
    pub fn ef_tensor_cuda_attached(t: *const EfTensor) -> c_int;

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

    pub fn ef_d3d11_device() -> *mut std::ffi::c_void;
    pub fn ef_d3d11_use_external_device(device: *mut std::ffi::c_void) -> c_int;
    pub fn ef_tensor_d3d11_texture(t: *const EfTensor) -> *mut std::ffi::c_void;
    pub fn ef_tensor_d3d11_layout(t: *const EfTensor, out: *mut EfD3d11Layout) -> c_int;
    pub fn ef_tensor_d3d11_shared_handle(t: *const EfTensor) -> *mut std::ffi::c_void;
    pub fn ef_tensor_gpu_completion(
        t: *const EfTensor,
        fence: *mut *mut std::ffi::c_void,
        value: *mut u64,
    ) -> c_int;
    pub fn ef_tensor_gpu_write_value(t: *const EfTensor) -> u64;
    pub fn ef_tensor_set_gpu_write(t: *mut EfTensor, value: u64) -> c_int;
    pub fn ef_tensor_from_d3d11_texture(
        texture: *mut std::ffi::c_void,
        dtype: u32,
        dims: *const u64,
        ndim: u32,
        format: *const c_char,
        access: u32,
        name: *const c_char,
    ) -> *mut EfTensor;
    pub fn ef_tensor_from_d3d11_shared_handle(
        handle: *mut std::ffi::c_void,
        dtype: u32,
        dims: *const u64,
        ndim: u32,
        format: *const c_char,
        access: u32,
        fence: *mut std::ffi::c_void,
        fence_value: u64,
        name: *const c_char,
    ) -> *mut EfTensor;

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

    /// The number of function declarations in `include/edgefirst/tensor.h`
    /// (`grep -cE '^[A-Za-z].*ef_.*\(' include/edgefirst/tensor.h` from
    /// `crates/tensor-capi`).
    const HEADER_DECLARATION_COUNT: usize = 117;

    #[test]
    fn declared_matches_the_header_derived_count() {
        assert_eq!(DECLARED.len(), HEADER_DECLARATION_COUNT);
    }
}
