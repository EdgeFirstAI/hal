// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//! Windows D3D11 texture backing: the process device, adapter selection and
//! the texture storage. Windows-only; the layout table lives in
//! `crate::d3d11_layout`.

pub mod adapter;
pub(crate) mod com;
pub mod device;
// Compiled for both backends.
pub mod geometry;
#[cfg(not(edgefirst_tensor_dynamic))]
pub(crate) mod texture;

pub use adapter::{
    enumerate_dxgi_adapters, parse_adapter_selection, resolve_adapter, AdapterSelection,
    DxgiAdapter, ADAPTER_ENV, ADAPTER_ENV_ALIAS,
};

pub use device::{device, use_external_device, D3d11Device, GpuCompletion};
pub use geometry::{d3d11_shared_handle_geometry, d3d11_texture_geometry};

/// The `ID3D11Device*` the tensors this build allocates live on.
///
/// On the `static` backend this is [`device`]'s pointer. On the `dynamic`
/// backend it is the device of `libedgefirst_tensor`, read through
/// `ef_d3d11_device`.
///
/// Borrowed: no reference is transferred, and the caller must not `Release`
/// it.
#[cfg(not(edgefirst_tensor_dynamic))]
pub fn backend_device() -> crate::Result<*mut std::ffi::c_void> {
    device().map(|d| d.raw())
}

/// Installs a host-owned `ID3D11Device*` into the copy of this crate that
/// allocates.
///
/// On the `static` backend this is [`use_external_device`]. On the `dynamic`
/// backend the call goes through `ef_d3d11_use_external_device`; the library
/// publishes the device through the rendezvous and this copy adopts it.
///
/// # Safety
///
/// As [`use_external_device`]: `ptr` must be a live `ID3D11Device*` and must
/// stay live until the first device use takes the reference.
#[cfg(not(edgefirst_tensor_dynamic))]
pub unsafe fn backend_use_external_device(ptr: *mut std::ffi::c_void) -> crate::Result<()> {
    // SAFETY: the caller carries the contract `use_external_device` states.
    unsafe { use_external_device(ptr) }
}

/// The `dynamic` backend's [`backend_device`]: the library's device.
#[cfg(edgefirst_tensor_dynamic)]
pub fn backend_device() -> crate::Result<*mut std::ffi::c_void> {
    // SAFETY: no arguments; the export returns a borrowed pointer or NULL
    // with the reason recorded on this thread.
    let p = unsafe { edgefirst_tensor_ffi::ef_d3d11_device() };
    if p.is_null() {
        return Err(crate::tensor_dyn::ffi_error(|m| {
            crate::Error::IoError(std::io::Error::other(m))
        }));
    }
    Ok(p)
}

/// The `dynamic` backend's [`backend_use_external_device`].
///
/// # Safety
///
/// As [`use_external_device`].
#[cfg(edgefirst_tensor_dynamic)]
pub unsafe fn backend_use_external_device(ptr: *mut std::ffi::c_void) -> crate::Result<()> {
    // SAFETY: the caller carries the contract `ef_d3d11_use_external_device`
    // states.
    let rc = unsafe { edgefirst_tensor_ffi::ef_d3d11_use_external_device(ptr) };
    if rc != 0 {
        return Err(crate::tensor_dyn::ffi_error(crate::Error::InvalidArgument));
    }
    Ok(())
}
