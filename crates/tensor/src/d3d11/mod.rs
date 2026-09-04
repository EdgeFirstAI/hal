// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//! Windows D3D11 texture backing: the process device, adapter selection and
//! the texture storage. Windows-only; the layout table lives in
//! `crate::d3d11_layout` so every platform tests it.
//!
//! The module is gated once, on its declaration in `lib.rs`.

pub mod adapter;
pub(crate) mod com;
pub mod device;
// Both backends: a constructor handed a raw texture reads its geometry
// here rather than inferring it from a caller's shape, and three of the
// four surfaces that do so build `dynamic`. See the module's own comment.
pub mod geometry;
// The texture storage is reachable only through the `static` backend's
// `TensorStorage`; `dynamic` forwards every tensor call to the C ABI and never
// constructs one, so compiling it there would be dead code.
#[cfg(feature = "static")]
pub(crate) mod texture;

pub use adapter::{
    enumerate_dxgi_adapters, parse_adapter_selection, resolve_adapter, AdapterSelection,
    DxgiAdapter, ADAPTER_ENV, ADAPTER_ENV_ALIAS,
};

pub use device::{device, use_external_device, D3d11Device, GpuCompletion};
pub use geometry::{d3d11_shared_handle_geometry, d3d11_texture_geometry};

/// The `ID3D11Device*` the tensors *this build* allocates actually live on.
///
/// [`device`] answers for this copy of the crate. On the `dynamic` backend
/// that copy allocates nothing: every tensor is created inside
/// `libedgefirst_tensor`, whose own copy owns the process device, so the
/// honest answer to "which device are my textures on" comes from asking that
/// library through `ef_d3d11_device`. On `static` the two are one object.
///
/// **Borrowed**: no reference is transferred, and the caller must never
/// `Release` it.
#[cfg(feature = "static")]
pub fn backend_device() -> crate::Result<*mut std::ffi::c_void> {
    device().map(|d| d.raw())
}

/// Installs a host-owned `ID3D11Device*` into the copy of this crate that
/// allocates.
///
/// [`use_external_device`] installs into this copy's own slot. On the
/// `dynamic` backend that slot is never read: the library allocates, and it
/// consults its own slot and the rendezvous. A host that installed here and
/// then allocated would silently get the library's own freshly created
/// device instead, with no error to say so. Routing the call through
/// `ef_d3d11_use_external_device` puts the pointer where the allocation reads
/// it; the library then publishes it through the rendezvous, so this copy
/// adopts it as well.
///
/// # Safety
///
/// As [`use_external_device`]: `ptr` must be a live `ID3D11Device*` and must
/// stay live until the first device use takes the reference.
#[cfg(feature = "static")]
pub unsafe fn backend_use_external_device(ptr: *mut std::ffi::c_void) -> crate::Result<()> {
    // SAFETY: the caller carries this function's own contract, which is the
    // one `use_external_device` states.
    unsafe { use_external_device(ptr) }
}

/// The `dynamic` backend's [`backend_device`]: the library's device, not this
/// copy's.
#[cfg(feature = "dynamic")]
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
#[cfg(feature = "dynamic")]
pub unsafe fn backend_use_external_device(ptr: *mut std::ffi::c_void) -> crate::Result<()> {
    // SAFETY: the caller carries the contract `ef_d3d11_use_external_device`
    // states, which is the one `use_external_device` states.
    let rc = unsafe { edgefirst_tensor_ffi::ef_d3d11_use_external_device(ptr) };
    if rc != 0 {
        return Err(crate::tensor_dyn::ffi_error(crate::Error::InvalidArgument));
    }
    Ok(())
}
