// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Windows D3D11 texture tensor surface.
//!
//! Every export here exists on every platform and refuses with `ENOTSUP` off
//! Windows, the same rule `ef_tensor_from_iosurface_id` and the
//! AHardwareBuffer family follow: this library's symbol set is identical
//! everywhere, so "does this build have it" is never a link-time question
//! for a consumer.
//!
//! # Shapes and geometry
//!
//! The two constructors take `dims` as the tensor's shape -- exactly what
//! `ef_tensor_shape` reports back -- and derive `width`/`height` from the
//! texture itself rather than from `dims`. The shape is then checked
//! against the texture: a mismatch is `EINVAL` rather than a silent
//! reinterpretation, which matters because a semi-planar allocation shape
//! and an addressing shape are both rank 2 and nothing in the numbers
//! distinguishes them.
//!
//! # Panic-path bookkeeping
//!
//! Every export here is shielded, and every one installs the quiet panic hook
//! before its `catch_unwind` -- through [`shield_int`] where the C return type
//! is `int`, and by calling [`ensure_hook_installed`] directly where it is
//! not: a pointer, or [`ef_tensor_gpu_write_value`]'s `uint64_t`. Without the
//! hook a caught panic writes no thread-local, so a consumer reading
//! `ef_tensor_last_error_class` after a `NULL` return gets a class some
//! earlier failure left behind and reports it as this call's. The
//! off-Windows arms sit inside the shield too, so a caller sees the same
//! `errno`-and-message contract on every platform.

use std::ffi::{c_char, c_int, c_void};
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor_abi::EfD3d11Layout;

use crate::handle::EfTensor;
use crate::last_error::{ensure_hook_installed, set_errno, set_last_error, shield_int};

#[cfg(target_os = "windows")]
use crate::handle::{into_handle, read_dims, tensor_of};
#[cfg(target_os = "windows")]
use crate::last_error::{class_of, set_last_error_classified};

/// Platforms: Windows.
///
/// The process `ID3D11Device*`, created on first call. **Borrowed**: no
/// reference is transferred, so a caller that keeps the pointer past this
/// tensor library's lifetime must `AddRef` it, and must never `Release` the
/// one handed back here.
///
/// Every texture tensor this library allocates lives on this device, so a
/// consumer that wants to render into one creates its own resources on the
/// same device rather than opening a shared handle.
///
/// @retval `NULL` with `errno` set: `EIO` when no device could be created
///         (no adapter, or `D3D11CreateDevice` failed --
///         `ef_tensor_last_error_message` carries the reason), `ENOTSUP`
///         off Windows.
#[no_mangle]
pub extern "C" fn ef_d3d11_device() -> *mut c_void {
    ensure_hook_installed();
    catch_unwind(|| {
        #[cfg(target_os = "windows")]
        {
            match edgefirst_tensor::d3d11::device() {
                Ok(d) => d.raw(),
                Err(e) => {
                    set_last_error(&format!("d3d11_device: {e}"));
                    set_errno(libc::EIO);
                    std::ptr::null_mut()
                }
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            set_last_error("d3d11_device: D3D11 is Windows only");
            set_errno(libc::ENOTSUP);
            std::ptr::null_mut()
        }
    })
    .unwrap_or_else(|_| {
        // An unwind must not leave a stale errno behind the NULL it returns.
        set_errno(libc::EINVAL);
        std::ptr::null_mut()
    })
}

/// Platforms: Windows.
///
/// Adopt a host-owned `ID3D11Device*` as the process device, so tensors this
/// library allocates share the caller's device instead of creating a second
/// one. The reference stays the caller's; this library takes its own when it
/// first uses the device.
///
/// Must be called before anything that creates the device as a side effect
/// -- `ef_d3d11_device`, `ef_is_gpu_buffer_available`, or any texture
/// allocation. Once the device exists it cannot be replaced.
///
/// @return 0 on success, `EINVAL` for `NULL` or a pointer that is not a live
///         `ID3D11Device`, `EBUSY` when the device is already initialized,
///         `ENOTSUP` off Windows.
///
/// # Safety
/// `device` must be `NULL` or a live `ID3D11Device *`.
#[no_mangle]
pub unsafe extern "C" fn ef_d3d11_use_external_device(device: *mut c_void) -> c_int {
    shield_int(|| {
        #[cfg(target_os = "windows")]
        {
            // SAFETY: this entry point's own contract is that `device` is
            // NULL or a live `ID3D11Device *`; the caller keeps the reference
            // until the library takes one on first use.
            match unsafe { edgefirst_tensor::d3d11::use_external_device(device) } {
                Ok(()) => 0,
                Err(e) => {
                    let rc = match &e {
                        edgefirst_tensor::Error::InvalidOperation(_) => libc::EBUSY,
                        _ => libc::EINVAL,
                    };
                    note_error(&e, "d3d11_use_external_device");
                    set_errno(rc);
                    rc
                }
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = device;
            set_last_error("d3d11_use_external_device: D3D11 is Windows only");
            set_errno(libc::ENOTSUP);
            libc::ENOTSUP
        }
    })
}

/// Record a real backend error with its class, prefixed by the entry point
/// that produced it, and set the matching `errno`.
///
/// Returns that errno so an `int`-returning entry point can `return
/// refuse(..)` and a pointer-returning one can ignore it.
///
/// [`crate::map::errno_for`] does the translation rather than a blanket
/// `ENOTSUP`: "this tensor is not a D3D11 texture" really is `ENOTSUP` (it
/// arrives as `Error::NotImplemented`, which `errno_for` already maps that
/// way), but a fence duplication that failed on a handle limit is not, and
/// collapsing the two costs the caller the distinction.
///
/// Windows-only, because the off-platform arms have no
/// [`edgefirst_tensor::Error`] to classify -- they never reach the backend
/// at all -- and so call [`set_last_error`] directly.
#[cfg(target_os = "windows")]
fn refuse(e: &edgefirst_tensor::Error, what: &str) -> c_int {
    note_error(e, what);
    let rc = crate::map::errno_for(e);
    set_errno(rc);
    rc
}

/// [`refuse`] without the errno: for the one entry point that chooses its
/// own code.
///
/// `ef_d3d11_use_external_device` answers `EBUSY` for "the device is already
/// initialized", which arrives as `Error::InvalidOperation` and which
/// [`crate::map::errno_for`] maps to `EACCES` -- correct for the map path
/// that conversion was written for, wrong here.
#[cfg(target_os = "windows")]
fn note_error(e: &edgefirst_tensor::Error, what: &str) {
    set_last_error_classified(class_of(e), &format!("{what}: {e}"));
}

/// Platforms: Windows.
///
/// The `ID3D11Texture2D*` backing this tensor. **Borrowed**: valid while the
/// tensor lives, and never `Release`d by the caller. Bind it, copy from it,
/// or `QueryInterface` it; to hand it to another device or process use
/// [`ef_tensor_d3d11_shared_handle`] instead.
///
/// @retval `NULL` with `errno` set: `EINVAL` for a `NULL` tensor, `ENOTSUP`
///         for a tensor that is not a D3D11 texture and off Windows.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_d3d11_texture(t: *const EfTensor) -> *mut c_void {
    ensure_hook_installed();
    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(target_os = "windows")]
        {
            let Some(inner) = tensor_of(t) else {
                set_last_error("d3d11_texture: could not resolve handle");
                set_errno(libc::EINVAL);
                return std::ptr::null_mut();
            };
            match inner.d3d11_texture() {
                Some(p) => p,
                None => {
                    set_last_error("d3d11_texture: not a D3D11 texture tensor");
                    set_errno(libc::ENOTSUP);
                    std::ptr::null_mut()
                }
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = t;
            set_last_error("d3d11_texture: D3D11 is Windows only");
            set_errno(libc::ENOTSUP);
            std::ptr::null_mut()
        }
    }))
    .unwrap_or_else(|_| {
        // An unwind must not leave a stale errno behind the NULL it returns.
        set_errno(libc::EINVAL);
        std::ptr::null_mut()
    })
}

/// Platforms: Windows.
///
/// The texture geometry the HAL chose for this image: DXGI format, texel
/// dimensions, bytes per texel, and the matching GL internal format. The
/// dimensions are the *texture's*, not the image's -- see `ef_d3d11_layout`
/// for why the two differ for semi-planar and packed-YUV formats.
///
/// `out` is written only on success.
///
/// @return 0 on success, `EINVAL` for a `NULL` tensor or `out`, `ENOTSUP`
///         for a tensor that is not a D3D11 texture and off Windows.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `out` must be writable for one
/// `ef_d3d11_layout`.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_d3d11_layout(
    t: *const EfTensor,
    out: *mut EfD3d11Layout,
) -> c_int {
    shield_int(|| {
        #[cfg(target_os = "windows")]
        {
            if out.is_null() {
                set_last_error("d3d11_layout: null out");
                set_errno(libc::EINVAL);
                return libc::EINVAL;
            }
            let Some(inner) = tensor_of(t) else {
                set_last_error("d3d11_layout: could not resolve handle");
                set_errno(libc::EINVAL);
                return libc::EINVAL;
            };
            let Some(l) = inner.d3d11_layout() else {
                set_last_error("d3d11_layout: not a D3D11 texture tensor");
                set_errno(libc::ENOTSUP);
                return libc::ENOTSUP;
            };
            // SAFETY: `out` is non-null and the caller contracts it is
            // writable for one `EfD3d11Layout`.
            unsafe {
                *out = EfD3d11Layout {
                    dxgi_format: l.dxgi_format,
                    texture_width: l.texture_width as u32,
                    texture_height: l.texture_height as u32,
                    bytes_per_texel: l.bytes_per_texel as u32,
                    gl_internal_format: l.gl_internal_format,
                };
            }
            0
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = (t, out);
            set_last_error("d3d11_layout: D3D11 is Windows only");
            set_errno(libc::ENOTSUP);
            libc::ENOTSUP
        }
    })
}

/// Platforms: Windows.
///
/// An **owned** duplicate of the texture's NT shared handle; close it with
/// `CloseHandle`. Open it on a D3D12 device, another D3D11 device, in CUDA,
/// or duplicate it into another process. Closing it does not affect the
/// tensor, and the tensor's own handle outlives this duplicate.
///
/// @retval `NULL` with `errno` set: `EINVAL` for a `NULL` tensor, `ENOTSUP`
///         for a tensor that is not a D3D11 texture and off Windows, or
///         another errno translated from the backend's error -- a
///         duplication that failed for its own reason keeps that reason.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_d3d11_shared_handle(t: *const EfTensor) -> *mut c_void {
    ensure_hook_installed();
    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::io::IntoRawHandle;
            let Some(inner) = tensor_of(t) else {
                set_last_error("d3d11_shared_handle: could not resolve handle");
                set_errno(libc::EINVAL);
                return std::ptr::null_mut();
            };
            match inner.d3d11_shared_handle() {
                Ok(h) => h.into_raw_handle().cast::<c_void>(),
                Err(e) => {
                    refuse(&e, "d3d11_shared_handle");
                    std::ptr::null_mut()
                }
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = t;
            set_last_error("d3d11_shared_handle: D3D11 is Windows only");
            set_errno(libc::ENOTSUP);
            std::ptr::null_mut()
        }
    }))
    .unwrap_or_else(|_| {
        // An unwind must not leave a stale errno behind the NULL it returns.
        set_errno(libc::EINVAL);
        std::ptr::null_mut()
    })
}

/// Platforms: Windows.
///
/// The fence a GPU consumer waits on before reading this texture, and the
/// timeline value to wait for.
///
/// `*fence` and `*value` are cleared to `NULL`/0 before anything else, so
/// whatever the caller had in those variables can never be mistaken for an
/// answer: "no write has been recorded" is return 0 with `*fence == NULL`,
/// and every failure path leaves them cleared too.
///
/// When a write *is* recorded, `*fence` is an **owned** duplicate of the
/// shared fence's NT handle; close it with `CloseHandle`. Open it with
/// `ID3D12Device::OpenSharedHandle`, `ID3D11Device5::OpenSharedFence`, or
/// `cudaImportExternalSemaphore`, and wait for `*value`.
///
/// @return 0 on success, whether or not a write is recorded; `EINVAL` for a
///         `NULL` tensor, `fence` or `value`; `ENOTSUP` for a tensor that is
///         not a D3D11 texture and off Windows; or another errno translated
///         from the backend's error.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `fence` and `value` must be writable.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_gpu_completion(
    t: *const EfTensor,
    fence: *mut *mut c_void,
    value: *mut u64,
) -> c_int {
    shield_int(|| {
        if fence.is_null() || value.is_null() {
            set_last_error("gpu_completion: null fence or value");
            set_errno(libc::EINVAL);
            return libc::EINVAL;
        }
        // SAFETY: both are non-null and the caller contracts they are
        // writable. Cleared before any other work so every failure path
        // leaves the caller's locals in the documented "absent" state.
        unsafe {
            *fence = std::ptr::null_mut();
            *value = 0;
        }
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::io::IntoRawHandle;
            let Some(inner) = tensor_of(t) else {
                set_last_error("gpu_completion: could not resolve handle");
                set_errno(libc::EINVAL);
                return libc::EINVAL;
            };
            match inner.gpu_completion() {
                Ok(None) => 0,
                Ok(Some(c)) => {
                    // SAFETY: as above.
                    unsafe {
                        *fence = c.fence.into_raw_handle().cast::<c_void>();
                        *value = c.value;
                    }
                    0
                }
                Err(e) => refuse(&e, "gpu_completion"),
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = t;
            set_last_error("gpu_completion: D3D11 is Windows only");
            set_errno(libc::ENOTSUP);
            libc::ENOTSUP
        }
    })
}

/// Platforms: Windows.
///
/// The fence value of the newest GPU write recorded on this tensor, or 0 when
/// there is none: the `*value` [`ef_tensor_gpu_completion`] reports, without
/// the duplicated fence handle. For a consumer that already holds the process
/// fence -- one an earlier `ef_tensor_gpu_completion` handed it -- and needs
/// only the value to wait for, so a query costs no `DuplicateHandle` and no
/// `CloseHandle`.
///
/// @return the recorded value, or 0 with `errno` set: `EINVAL` for a `NULL`
///         tensor, `ENOTSUP` for a tensor that is not a D3D11 texture and
///         off Windows. A texture with no recorded write answers 0 and
///         leaves `errno` alone.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_gpu_write_value(t: *const EfTensor) -> u64 {
    ensure_hook_installed();
    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(target_os = "windows")]
        {
            let Some(inner) = tensor_of(t) else {
                set_last_error("gpu_write_value: could not resolve handle");
                set_errno(libc::EINVAL);
                return 0;
            };
            if inner.d3d11_texture().is_none() {
                set_last_error("gpu_write_value: not a D3D11 texture tensor");
                set_errno(libc::ENOTSUP);
                return 0;
            }
            inner.gpu_write_value()
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = t;
            set_last_error("gpu_write_value: D3D11 is Windows only");
            set_errno(libc::ENOTSUP);
            0
        }
    }))
    // A caught panic is a failure too, and 0 on its own is the "no write
    // recorded" answer: without an `errno` beside it a caller cannot tell
    // the two apart. `EINVAL` matches the other unresolvable-input path.
    .unwrap_or_else(|_| {
        set_errno(libc::EINVAL);
        0
    })
}

/// Platforms: Windows.
///
/// Record that the GPU work writing this texture completes at `value` of the
/// process device's shared fence -- the value a later
/// [`ef_tensor_gpu_completion`] hands a consumer. Monotonic: an older value
/// never displaces a newer one.
///
/// The value is a monotonic maximum, recorded into an atomic: it is safe to
/// call while consumers hold the same handle and are reading the tensor,
/// which is the situation it exists for -- a producer that has just queued
/// work publishes the fence value on a tensor it has already shared. Unlike
/// the geometry setters, this needs no exclusive access. `ef_tensor *` rather
/// than `const ef_tensor *` only because it changes what the tensor reports.
///
/// @return 0 on success, `EINVAL` for a `NULL` tensor, `ENOTSUP` for a
///         tensor that is not a D3D11 texture and off Windows, or another
///         errno translated from the backend's error.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_set_gpu_write(t: *mut EfTensor, value: u64) -> c_int {
    shield_int(|| {
        #[cfg(target_os = "windows")]
        {
            let Some(inner) = tensor_of(t) else {
                set_last_error("set_gpu_write: could not resolve handle");
                set_errno(libc::EINVAL);
                return libc::EINVAL;
            };
            match inner.set_gpu_write(value) {
                Ok(()) => 0,
                Err(e) => refuse(&e, "set_gpu_write"),
            }
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = (t, value);
            set_last_error("set_gpu_write: D3D11 is Windows only");
            set_errno(libc::ENOTSUP);
            libc::ENOTSUP
        }
    })
}

/// Platforms: Windows.
///
/// Wrap an existing `ID3D11Texture2D` as a tensor. The texture must live on
/// the process device ([`ef_d3d11_device`]) and match the layout the HAL
/// would have chosen for `format` at the texture's own dimensions.
/// Ownership stays with the caller: this takes its own reference, and the
/// tensor releases it when freed.
///
/// `dims`/`ndim` describe the tensor's grid: the allocation shape --
/// `[height, width, channels]` packed, `[channels, height, width]` planar,
/// `[combined_height, width]` semi-planar -- or the (smaller) addressing
/// shape, which names the same texture. The width and height are read off
/// the texture description, never derived from `dims`; `dims` is then
/// checked against them and anything else is `EINVAL`, so a shape cannot
/// silently reinterpret the texture. The tensor that comes back always
/// carries the allocation shape, so `ef_tensor_shape` echoes `dims` back
/// only for that spelling.
///
/// `format` is the wire descriptor (`"NV12"`, `"rgba8"`), the same
/// vocabulary every other entry point takes; `dtype` is an `EF_DTYPE_*`
/// code; `access` is an `EF_CPU_ACCESS_*` code, and decides whether a CPU
/// staging texture is created alongside -- `EF_CPU_ACCESS_NONE` included,
/// which is what a caller that will only touch the texture from the GPU
/// asks for and costs no staging texture at all. `name` may be `NULL`.
///
/// A semi-planar texture (`"NV12"`, `"NV16"`, `"NV24"`) carries the image's
/// row stride as its width, so it is accepted only when that width is even
/// and equal to the staging row pitch the driver gives a texture of that
/// width; anything else is `EINVAL` with both numbers in the message. The
/// HAL allocates its own that way. A D3D12 or CUDA producer that needs one
/// should allocate through the HAL, or match the pitch.
///
/// @retval a new tensor the caller must free with `ef_tensor_free`.
/// @retval `NULL` with `errno` set: `EINVAL` for a `NULL`/unshaped argument,
///         an unknown format/dtype/access code, or a `dims` the texture does
///         not have; the errno the backend's own failure maps to when the
///         import itself fails; `ENOTSUP` off Windows.
///         `ef_tensor_last_error_message` carries the reason.
///
/// # Safety
/// `texture` must be a live `ID3D11Texture2D *` created on the process
/// device; `dims` must point to `ndim` readable `uint64_t`; `format` and
/// `name` must be `NULL` or NUL-terminated.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_from_d3d11_texture(
    texture: *mut c_void,
    dtype: u32,
    dims: *const u64,
    ndim: u32,
    format: *const c_char,
    access: u32,
    name: *const c_char,
) -> *mut EfTensor {
    // The quiet hook, before the catch, and around the off-Windows arm too:
    // see the module's panic-path note.
    ensure_hook_installed();
    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(target_os = "windows")]
        {
            // SAFETY: the caller contracts these pointers.
            let Some(args) = (unsafe {
                ImportArgs::read(
                    dims,
                    ndim,
                    format,
                    dtype,
                    access,
                    name,
                    "from_d3d11_texture",
                )
            }) else {
                return std::ptr::null_mut();
            };
            if texture.is_null() {
                set_last_error("from_d3d11_texture: null texture");
                set_errno(libc::EINVAL);
                return std::ptr::null_mut();
            }
            // SAFETY: the caller contracts `texture` is a live
            // `ID3D11Texture2D` on the process device.
            // The shape goes in, not just out: a semi-planar texture is as
            // wide as its row pitch, so `dims` is where its image width comes
            // from. `check_geometry` still checks the rest.
            let geometry = unsafe {
                edgefirst_tensor::d3d11_texture_geometry(texture, args.format, Some(&args.shape))
            };
            let Some((width, height)) = args.check_geometry(geometry, "from_d3d11_texture") else {
                return std::ptr::null_mut();
            };
            // SAFETY: as above; the constructor takes its own reference.
            let built = unsafe {
                edgefirst_tensor::TensorDyn::from_d3d11_texture(
                    texture,
                    width,
                    height,
                    args.format,
                    args.dtype,
                    args.access,
                    args.name.as_deref(),
                )
            };
            finish(built, "from_d3d11_texture")
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = (texture, dtype, dims, ndim, format, access, name);
            set_last_error("from_d3d11_texture: D3D11 is Windows only");
            set_errno(libc::ENOTSUP);
            std::ptr::null_mut()
        }
    }))
    .unwrap_or_else(|_| {
        // An unwind must not leave a stale errno behind the NULL it returns.
        set_errno(libc::EINVAL);
        std::ptr::null_mut()
    })
}

/// Platforms: Windows.
///
/// Open a shared D3D11 texture by its NT handle on the process device --
/// the consumer half of [`ef_tensor_d3d11_shared_handle`], and the route by
/// which a texture crosses a process boundary.
///
/// `handle` stays owned by the caller: this opens its own texture from it,
/// so the caller still closes the handle with `CloseHandle`. `fence` is the
/// same: when non-`NULL` it names a shared fence whose value `fence_value`
/// is waited for on the process device's immediate context, so a same-device
/// reader needs no further ordering; the handle is duplicated, never
/// consumed. Pass `NULL`/0 when there is nothing to wait for.
///
/// `dims`, `format`, `dtype`, `access` and `name` follow
/// [`ef_tensor_from_d3d11_texture`] exactly, geometry check included.
///
/// @retval a new tensor the caller must free with `ef_tensor_free`.
/// @retval `NULL` with `errno` set, as [`ef_tensor_from_d3d11_texture`].
///
/// # Safety
/// `handle` must be an NT shared handle of a D3D11 texture, valid in this
/// process, and `fence` `NULL` or a shared-fence NT handle; `dims` must
/// point to `ndim` readable `uint64_t`; `format` and `name` must be `NULL`
/// or NUL-terminated.
#[no_mangle]
#[allow(clippy::too_many_arguments)] // one image description, spelled out
pub unsafe extern "C" fn ef_tensor_from_d3d11_shared_handle(
    handle: *mut c_void,
    dtype: u32,
    dims: *const u64,
    ndim: u32,
    format: *const c_char,
    access: u32,
    fence: *mut c_void,
    fence_value: u64,
    name: *const c_char,
) -> *mut EfTensor {
    // As `ef_tensor_from_d3d11_texture`: see the module's panic-path note.
    ensure_hook_installed();
    catch_unwind(AssertUnwindSafe(|| {
        #[cfg(target_os = "windows")]
        {
            // SAFETY: the caller contracts these pointers.
            let Some(args) = (unsafe {
                ImportArgs::read(
                    dims,
                    ndim,
                    format,
                    dtype,
                    access,
                    name,
                    "from_d3d11_shared_handle",
                )
            }) else {
                return std::ptr::null_mut();
            };
            if handle.is_null() {
                set_last_error("from_d3d11_shared_handle: null handle");
                set_errno(libc::EINVAL);
                return std::ptr::null_mut();
            }
            // SAFETY: the caller contracts `handle` is a live NT shared
            // handle of a D3D11 texture; this opens and drops its own
            // texture from it.
            // As above: `dims` carries the semi-planar image width.
            let geometry = unsafe {
                edgefirst_tensor::d3d11_shared_handle_geometry(
                    handle,
                    args.format,
                    Some(&args.shape),
                )
            };
            let Some((width, height)) = args.check_geometry(geometry, "from_d3d11_shared_handle")
            else {
                return std::ptr::null_mut();
            };
            let completion = (!fence.is_null()).then_some((fence, fence_value));
            // SAFETY: as above; the constructor duplicates what it keeps.
            let built = unsafe {
                edgefirst_tensor::TensorDyn::from_d3d11_shared_handle(
                    handle,
                    width,
                    height,
                    args.format,
                    args.dtype,
                    args.access,
                    completion,
                    args.name.as_deref(),
                )
            };
            finish(built, "from_d3d11_shared_handle")
        }
        #[cfg(not(target_os = "windows"))]
        {
            let _ = (
                handle,
                dtype,
                dims,
                ndim,
                format,
                access,
                fence,
                fence_value,
                name,
            );
            set_last_error("from_d3d11_shared_handle: D3D11 is Windows only");
            set_errno(libc::ENOTSUP);
            std::ptr::null_mut()
        }
    }))
    .unwrap_or_else(|_| {
        // An unwind must not leave a stale errno behind the NULL it returns.
        set_errno(libc::EINVAL);
        std::ptr::null_mut()
    })
}

/// The arguments the two constructors share, decoded once.
///
/// Written once rather than twice: the pair differ only in which geometry
/// helper reads the texture, and two hand-written copies of "decode the
/// vocabulary codes, then check the shape" are exactly where the two would
/// drift into accepting different arguments.
#[cfg(target_os = "windows")]
struct ImportArgs {
    shape: Vec<usize>,
    format: edgefirst_tensor::PixelFormat,
    dtype: edgefirst_tensor::DType,
    access: edgefirst_tensor::CpuAccess,
    name: Option<String>,
}

#[cfg(target_os = "windows")]
impl ImportArgs {
    /// Decode `dims`/`format`/`dtype`/`access`/`name`, setting the last
    /// error and `errno` and returning `None` on the first bad one.
    ///
    /// # Safety
    /// `dims` must be `NULL` or point to `ndim` readable `uint64_t`;
    /// `format` and `name` must be `NULL` or NUL-terminated.
    unsafe fn read(
        dims: *const u64,
        ndim: u32,
        format: *const c_char,
        dtype: u32,
        access: u32,
        name: *const c_char,
        what: &str,
    ) -> Option<Self> {
        // SAFETY: the caller contracts `dims`.
        let shape = unsafe { read_dims(dims, ndim, what) }.or_else(|| {
            set_errno(libc::EINVAL);
            None
        })?;
        if format.is_null() {
            set_last_error(&format!("{what}: null format"));
            set_errno(libc::EINVAL);
            return None;
        }
        // SAFETY: the caller contracts `format` is NUL-terminated.
        let format_str = unsafe { std::ffi::CStr::from_ptr(format) }.to_str().ok();
        let Some(fmt) = format_str.and_then(edgefirst_tensor::PixelFormat::from_str_code) else {
            set_last_error_classified(
                edgefirst_tensor_abi::EfErrorClass::InvalidArgument,
                &format!("{what}: unknown pixel format {format_str:?}"),
            );
            set_errno(libc::EINVAL);
            return None;
        };
        let Some(dt) = edgefirst_tensor::DType::from_code(dtype) else {
            set_last_error(&format!("{what}: unknown dtype code {dtype}"));
            set_errno(libc::EINVAL);
            return None;
        };
        // The declaration decoder, not the map-direction one: these
        // constructors ask what CPU access to provision, and
        // `EF_CPU_ACCESS_NONE` -- a texture the caller will only touch from
        // the GPU -- is a first-class answer that costs no staging texture.
        let Some(acc) = crate::codes::declared_cpu_access_from_code(access) else {
            set_last_error(&format!("{what}: unknown access code {access}"));
            set_errno(libc::EINVAL);
            return None;
        };
        let name = if name.is_null() {
            None
        } else {
            // SAFETY: the caller contracts `name` is NUL-terminated.
            unsafe { std::ffi::CStr::from_ptr(name) }
                .to_str()
                .ok()
                .map(str::to_owned)
        };
        Some(ImportArgs {
            shape,
            format: fmt,
            dtype: dt,
            access: acc,
            name,
        })
    }

    /// Check the caller's `dims` against the geometry read off the texture.
    ///
    /// Either the allocation shape or the addressing shape is accepted: the
    /// two are different grids over the same texture, both legitimate
    /// answers to "what shape is this", and refusing one would make the
    /// export unusable from whichever surface reports the other.
    fn check_geometry(
        &self,
        geometry: edgefirst_tensor::Result<(usize, usize)>,
        what: &str,
    ) -> Option<(usize, usize)> {
        let (width, height) = match geometry {
            Ok(g) => g,
            Err(e) => {
                refuse(&e, what);
                return None;
            }
        };
        let allocation = self.format.allocation_shape(width, height);
        let addressing = self.format.addressing_shape(width, height);
        let matches = [&allocation, &addressing]
            .into_iter()
            .flatten()
            .any(|s| s.as_slice() == self.shape.as_slice());
        if !matches {
            set_last_error_classified(
                edgefirst_tensor_abi::EfErrorClass::InvalidShape,
                &format!(
                    "{what}: dims {:?} describe neither the allocation shape {:?} nor the \
                     addressing shape {:?} of this {}x{} {:?} texture",
                    self.shape, allocation, addressing, width, height, self.format
                ),
            );
            set_errno(libc::EINVAL);
            return None;
        }
        Some((width, height))
    }
}

/// Wrap a constructed tensor in a handle, or record the failure.
#[cfg(target_os = "windows")]
fn finish(
    built: edgefirst_tensor::Result<edgefirst_tensor::TensorDyn>,
    what: &str,
) -> *mut EfTensor {
    match built {
        Ok(t) => into_handle(t),
        Err(e) => {
            refuse(&e, what);
            std::ptr::null_mut()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(target_os = "windows")]
    use crate::codes::EfStorageKind;
    use crate::codes::{EfCpuAccess, EfDtype};
    use edgefirst_tensor_abi::EfTensorView;

    /// Wire codes, spelled by name. The C tests reach these through the
    /// header's enumerators; a Rust test in this crate reaches the same
    /// values through `codes.rs`, whose compile-time assertions pin them to
    /// the Rust vocabulary -- so neither side hard-codes an integer.
    const U8: u32 = EfDtype::U8 as u32;
    const READ: u32 = EfCpuAccess::Read as u32;
    const READ_WRITE: u32 = EfCpuAccess::ReadWrite as u32;
    /// Gated with its users: only the Windows tests allocate a texture, so
    /// off Windows this is an unused constant and the C leaves' clippy runs
    /// with `-D warnings`.
    #[cfg(target_os = "windows")]
    const DMA_BUF: u32 = EfStorageKind::DmaBuf as u32;

    /// The calling thread's `errno`, for the entry points whose only failure
    /// channel is a `NULL` return. Read immediately after the call under
    /// test, as a C caller would.
    fn current_errno() -> i32 {
        errno::errno().0
    }

    /// Clear `errno` so a later read cannot pick up a value some earlier
    /// call left behind and report it as this one's.
    fn clear_errno() {
        errno::set_errno(errno::Errno(0));
    }

    /// What a `NULL` tensor costs: `EINVAL` where the handle is resolved
    /// (Windows), `ENOTSUP` where the whole family refuses before ever
    /// looking at it.
    fn null_tensor_errno() -> i32 {
        if cfg!(target_os = "windows") {
            libc::EINVAL
        } else {
            libc::ENOTSUP
        }
    }

    /// Every export refuses a `NULL` handle rather than dereferencing it --
    /// asserted on every platform, since the whole point of this family is
    /// that it is declared everywhere.
    #[test]
    fn every_export_refuses_a_null_tensor() {
        let mut layout = EfD3d11Layout::default();

        clear_errno();
        assert!(unsafe { ef_tensor_d3d11_texture(std::ptr::null()) }.is_null());
        assert_eq!(current_errno(), null_tensor_errno());

        assert_ne!(
            unsafe { ef_tensor_d3d11_layout(std::ptr::null(), &mut layout) },
            0
        );

        clear_errno();
        assert!(unsafe { ef_tensor_d3d11_shared_handle(std::ptr::null()) }.is_null());
        assert_eq!(current_errno(), null_tensor_errno());

        let mut fence = std::ptr::null_mut();
        let mut value = 0u64;
        assert_ne!(
            unsafe { ef_tensor_gpu_completion(std::ptr::null(), &mut fence, &mut value) },
            0
        );
        assert_ne!(
            unsafe { ef_tensor_set_gpu_write(std::ptr::null_mut(), 1) },
            0
        );

        clear_errno();
        assert_eq!(unsafe { ef_tensor_gpu_write_value(std::ptr::null()) }, 0);
        assert_eq!(current_errno(), null_tensor_errno());
    }

    /// `gpu_completion` clears the caller's out-parameters before doing
    /// anything else, so a failing call cannot leave stale values that read
    /// as a recorded write.
    #[test]
    fn gpu_completion_clears_its_out_parameters_before_failing() {
        let mut fence = std::ptr::dangling_mut::<c_void>();
        let mut value = 7u64;
        assert_ne!(
            unsafe { ef_tensor_gpu_completion(std::ptr::null(), &mut fence, &mut value) },
            0
        );
        assert!(fence.is_null());
        assert_eq!(value, 0);
    }

    /// A null `fence`/`value` is `EINVAL`, not a write through a null
    /// pointer.
    #[test]
    fn gpu_completion_rejects_null_out_parameters() {
        let mut value = 0u64;
        assert_eq!(
            unsafe { ef_tensor_gpu_completion(std::ptr::null(), std::ptr::null_mut(), &mut value) },
            libc::EINVAL
        );
    }

    /// A tensor that is not a D3D11 texture refuses the accessors rather
    /// than answering for some other backing, and says `ENOTSUP` rather than
    /// returning a bare `NULL`. Runs everywhere: off Windows the same calls
    /// take the platform arm, which answers `ENOTSUP` too.
    #[test]
    fn a_host_memory_tensor_is_not_a_d3d11_texture() {
        let dims = [4u64, 4, 4];
        let t = unsafe { crate::handle::ef_tensor_new(U8, dims.as_ptr(), 3) };
        assert!(!t.is_null());

        clear_errno();
        assert!(unsafe { ef_tensor_d3d11_texture(t) }.is_null());
        assert_eq!(current_errno(), libc::ENOTSUP);

        let mut layout = EfD3d11Layout::default();
        assert_eq!(
            unsafe { ef_tensor_d3d11_layout(t, &mut layout) },
            libc::ENOTSUP
        );

        clear_errno();
        assert!(unsafe { ef_tensor_d3d11_shared_handle(t) }.is_null());
        assert_eq!(current_errno(), libc::ENOTSUP);

        assert_eq!(unsafe { ef_tensor_set_gpu_write(t, 1) }, libc::ENOTSUP);

        clear_errno();
        assert_eq!(unsafe { ef_tensor_gpu_write_value(t) }, 0);
        assert_eq!(current_errno(), libc::ENOTSUP);
        unsafe { crate::handle::ef_tensor_free(t) };
    }

    /// `ef_tensor_try_map` behaves as `ef_tensor_map` on a backing with no
    /// GPU copy to wait for -- the contract that lets a caller use it
    /// unconditionally.
    #[test]
    fn try_map_succeeds_on_host_memory() {
        let dims = [4u64];
        let t = unsafe { crate::handle::ef_tensor_new(U8, dims.as_ptr(), 1) };
        let mut view = EfTensorView {
            ptr: std::ptr::null_mut(),
            len: 0,
        };
        assert_eq!(
            unsafe { crate::map::ef_tensor_try_map(t, READ_WRITE, &mut view) },
            0
        );
        assert_eq!(view.len, 4);
        assert_eq!(unsafe { crate::map::ef_tensor_unmap(t) }, 0);
        unsafe { crate::handle::ef_tensor_free(t) };
    }

    /// A non-null pointer the constructors under test never dereference:
    /// every case in `the_constructors_reject_bad_arguments` fails while
    /// decoding `dims`/`format`/`dtype`/`access`, before either constructor
    /// looks at the texture or the handle. Non-null on purpose -- a null one
    /// would be rejected by the null check instead, testing a different arm.
    fn not_reached() -> *mut c_void {
        std::ptr::dangling_mut::<c_void>()
    }

    /// What a malformed argument costs: `EINVAL` where the arguments are
    /// actually decoded (Windows), `ENOTSUP` where the constructor refuses
    /// before reading any of them.
    fn bad_argument_errno() -> i32 {
        if cfg!(target_os = "windows") {
            libc::EINVAL
        } else {
            libc::ENOTSUP
        }
    }

    /// The bad-argument arms of the two constructors, which are reached on
    /// every platform before any texture is touched.
    #[test]
    fn the_constructors_reject_bad_arguments() {
        let dims = [32u64, 64, 4];
        let rgba = std::ffi::CString::new("rgba8").unwrap();
        let bogus = std::ffi::CString::new("no-such-format").unwrap();

        // Null dims.
        clear_errno();
        assert!(unsafe {
            ef_tensor_from_d3d11_texture(
                not_reached(),
                U8,
                std::ptr::null(),
                0,
                rgba.as_ptr(),
                READ,
                std::ptr::null(),
            )
        }
        .is_null());
        assert_eq!(current_errno(), bad_argument_errno());

        // Unknown format name.
        clear_errno();
        assert!(unsafe {
            ef_tensor_from_d3d11_texture(
                not_reached(),
                U8,
                dims.as_ptr(),
                3,
                bogus.as_ptr(),
                READ,
                std::ptr::null(),
            )
        }
        .is_null());
        assert_eq!(current_errno(), bad_argument_errno());

        // Unknown dtype code.
        clear_errno();
        assert!(unsafe {
            ef_tensor_from_d3d11_texture(
                not_reached(),
                999,
                dims.as_ptr(),
                3,
                rgba.as_ptr(),
                READ,
                std::ptr::null(),
            )
        }
        .is_null());
        assert_eq!(current_errno(), bad_argument_errno());

        // An access code outside the vocabulary.
        clear_errno();
        assert!(unsafe {
            ef_tensor_from_d3d11_shared_handle(
                not_reached(),
                U8,
                dims.as_ptr(),
                3,
                rgba.as_ptr(),
                99,
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
            )
        }
        .is_null());
        assert_eq!(current_errno(), bad_argument_errno());
    }

    /// `EF_CPU_ACCESS_NONE` is accepted by both constructors: it is the
    /// declaration a caller makes when it will only touch the texture from
    /// the GPU, and the default the Python constructors document. What
    /// refuses it is the *map*, which needs a direction.
    #[cfg(target_os = "windows")]
    #[test]
    fn the_constructors_accept_no_cpu_access_and_the_map_still_refuses_it() {
        let Some(src) = texture_tensor("rgba8", 32, 16, READ_WRITE) else {
            return;
        };
        let texture = unsafe { ef_tensor_d3d11_texture(src) };
        assert!(!texture.is_null());
        let dims: [u64; 3] = [16, 32, 4];
        let rgba = std::ffi::CString::new("rgba8").unwrap();

        clear_errno();
        let wrapped = unsafe {
            ef_tensor_from_d3d11_texture(
                texture,
                U8,
                dims.as_ptr(),
                3,
                rgba.as_ptr(),
                EfCpuAccess::None as u32,
                std::ptr::null(),
            )
        };
        assert!(
            !wrapped.is_null(),
            "EF_CPU_ACCESS_NONE is a declaration, not a map direction (errno {})",
            current_errno()
        );

        let handle = unsafe { ef_tensor_d3d11_shared_handle(src) };
        assert!(!handle.is_null());
        let opened = unsafe {
            ef_tensor_from_d3d11_shared_handle(
                handle,
                U8,
                dims.as_ptr(),
                3,
                rgba.as_ptr(),
                EfCpuAccess::None as u32,
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
            )
        };
        assert!(!opened.is_null(), "errno {}", current_errno());

        // And the map still refuses the same code, because there is no
        // direction to map in.
        let mut view = EfTensorView {
            ptr: std::ptr::null_mut(),
            len: 0,
        };
        clear_errno();
        assert_ne!(
            unsafe { crate::map::ef_tensor_map(wrapped, EfCpuAccess::None as u32, &mut view) },
            0
        );

        unsafe {
            crate::handle::ef_tensor_free(opened);
            crate::handle::ef_tensor_free(wrapped);
            crate::handle::ef_tensor_free(src);
        }
        close_handle(handle);
    }

    /// The Windows path end to end: allocate a texture tensor, read its
    /// layout and texture, share it, record and read a completion, and wrap
    /// the shared handle back into a second tensor that sees the same bytes.
    ///
    /// The C leaf test (`tests/c/test_d3d11.c`) asserts the same contract
    /// through the generated header; this one runs on a host with no C
    /// toolchain, which is every Windows developer box in this project.
    #[cfg(target_os = "windows")]
    #[test]
    fn a_texture_tensor_round_trips_through_the_exports() {
        let Some(t) = texture_tensor("rgba8", 64, 32, READ_WRITE) else {
            return;
        };
        assert_eq!(unsafe { crate::handle::ef_tensor_storage_kind(t) }, DMA_BUF);

        let mut layout = EfD3d11Layout::default();
        assert_eq!(unsafe { ef_tensor_d3d11_layout(t, &mut layout) }, 0);
        assert_eq!(layout.texture_width, 64);
        assert_eq!(layout.texture_height, 32);
        assert_eq!(layout.bytes_per_texel, 4);

        let texture = unsafe { ef_tensor_d3d11_texture(t) };
        assert!(!texture.is_null());

        // Fill through the map window so the re-wrapped tensor has
        // something recognisable to read back.
        let mut view = EfTensorView {
            ptr: std::ptr::null_mut(),
            len: 0,
        };
        assert_eq!(
            unsafe { crate::map::ef_tensor_map(t, READ_WRITE, &mut view) },
            0
        );
        // SAFETY: the map window is outstanding and writable.
        unsafe { std::ptr::write_bytes(view.ptr, 0x3C, view.len) };
        assert_eq!(unsafe { crate::map::ef_tensor_unmap(t) }, 0);

        // No write recorded yet, then one, and the fence comes back owned.
        //
        // The `ef_tensor_d3d11_texture` reads bracketing the recording are
        // the point: `ef_tensor_set_gpu_write` needs only a shared
        // reference, so a producer can publish a fence value on a tensor
        // consumers already hold and are reading. Under the exclusive-borrow
        // route this used to take, this call sequence was the one the
        // contract forbade.
        let mut fence = std::ptr::dangling_mut::<c_void>();
        let mut value = 1u64;
        assert_eq!(
            unsafe { ef_tensor_gpu_completion(t, &mut fence, &mut value) },
            0
        );
        assert!(fence.is_null() && value == 0);
        assert_eq!(unsafe { ef_tensor_gpu_write_value(t) }, 0);

        let borrowed_before = unsafe { ef_tensor_d3d11_texture(t) };
        assert_eq!(unsafe { ef_tensor_set_gpu_write(t, 42) }, 0);
        let borrowed_after = unsafe { ef_tensor_d3d11_texture(t) };
        assert_eq!(borrowed_before, borrowed_after);
        assert_eq!(
            unsafe { ef_tensor_gpu_completion(t, &mut fence, &mut value) },
            0
        );
        assert!(!fence.is_null());
        assert_eq!(value, 42);
        close_handle(fence);
        assert_eq!(unsafe { ef_tensor_gpu_write_value(t) }, value);

        // Monotonic: an older value never displaces the newer one.
        assert_eq!(unsafe { ef_tensor_set_gpu_write(t, 7) }, 0);
        assert_eq!(
            unsafe { ef_tensor_gpu_completion(t, &mut fence, &mut value) },
            0
        );
        assert_eq!(value, 42);
        close_handle(fence);

        let shared = unsafe { ef_tensor_d3d11_shared_handle(t) };
        assert!(!shared.is_null());
        let dims = [32u64, 64, 4];
        let rgba = std::ffi::CString::new("rgba8").unwrap();
        let name = std::ffi::CString::new("again").unwrap();
        let again = unsafe {
            ef_tensor_from_d3d11_shared_handle(
                shared,
                U8,
                dims.as_ptr(),
                3,
                rgba.as_ptr(),
                READ,
                std::ptr::null_mut(),
                0,
                name.as_ptr(),
            )
        };
        assert!(!again.is_null(), "reopening the shared handle failed");
        assert_eq!(
            unsafe { crate::map::ef_tensor_map(again, READ, &mut view) },
            0
        );
        // SAFETY: the map window is outstanding.
        assert_eq!(unsafe { *view.ptr }, 0x3C);
        assert_eq!(unsafe { crate::map::ef_tensor_unmap(again) }, 0);
        unsafe { crate::handle::ef_tensor_free(again) };

        // The borrowed texture wraps back into a tensor of the same shape.
        let wrapped = unsafe {
            ef_tensor_from_d3d11_texture(
                texture,
                U8,
                dims.as_ptr(),
                3,
                rgba.as_ptr(),
                READ,
                std::ptr::null(),
            )
        };
        assert!(!wrapped.is_null(), "re-wrapping the texture failed");
        // SAFETY: `wrapped` is live and rank 3.
        let shape =
            unsafe { std::slice::from_raw_parts(crate::handle::ef_tensor_shape(wrapped), 3) };
        assert_eq!(shape, &[32, 64, 4]);
        unsafe { crate::handle::ef_tensor_free(wrapped) };

        // A shape the texture does not have is refused, not reinterpreted.
        let wrong = [16u64, 16, 4];
        clear_errno();
        assert!(unsafe {
            ef_tensor_from_d3d11_texture(
                texture,
                U8,
                wrong.as_ptr(),
                3,
                rgba.as_ptr(),
                READ,
                std::ptr::null(),
            )
        }
        .is_null());
        assert_eq!(current_errno(), libc::EINVAL);

        // `try_map` either succeeds or asks to be retried; it never stalls
        // and never reports some third thing.
        let rc = unsafe { crate::map::ef_tensor_try_map(t, READ, &mut view) };
        assert!(rc == 0 || rc == libc::EAGAIN, "try_map returned {rc}");
        if rc == 0 {
            assert_eq!(unsafe { crate::map::ef_tensor_unmap(t) }, 0);
        }

        close_handle(shared);
        unsafe { crate::handle::ef_tensor_free(t) };
    }

    /// Both spellings of `dims` name the same texture, on a format where
    /// they genuinely differ.
    ///
    /// NV12 is the discriminating case: at 64x48 its allocation shape is
    /// `[72, 64]` (the combined luma + chroma plane height) and its
    /// addressing shape is `[48, 64]` (the luma grid) -- both rank 2, so
    /// nothing but the numbers tells them apart. RGBA cannot test this at
    /// all: its two shapes are identical. A combined height no image height
    /// produces is refused rather than rounded to a nearby one.
    #[cfg(target_os = "windows")]
    #[test]
    fn either_shape_spelling_names_the_same_nv12_texture() {
        let Some(t) = texture_tensor("NV12", 64, 48, READ) else {
            return;
        };
        let shared = unsafe { ef_tensor_d3d11_shared_handle(t) };
        assert!(!shared.is_null());
        let nv12 = std::ffi::CString::new("NV12").unwrap();

        let open = |dims: &[u64]| unsafe {
            ef_tensor_from_d3d11_shared_handle(
                shared,
                U8,
                dims.as_ptr(),
                dims.len() as u32,
                nv12.as_ptr(),
                READ,
                std::ptr::null_mut(),
                0,
                std::ptr::null(),
            )
        };

        let allocation = open(&[72, 64]);
        assert!(!allocation.is_null(), "the allocation shape was refused");
        unsafe { crate::handle::ef_tensor_free(allocation) };

        let addressing = open(&[48, 64]);
        assert!(!addressing.is_null(), "the addressing shape was refused");
        unsafe { crate::handle::ef_tensor_free(addressing) };

        clear_errno();
        let neither = open(&[47, 64]);
        assert!(
            neither.is_null(),
            "a shape that is neither spelling must be refused, not reinterpreted"
        );
        assert_eq!(current_errno(), libc::EINVAL);

        close_handle(shared);
        unsafe { crate::handle::ef_tensor_free(t) };
    }

    /// Allocate a texture-backed image tensor through the image-desc
    /// builder, or `None` with a printed reason when this host has no D3D11
    /// device. Shared by the two Windows tests so neither repeats the
    /// builder dance.
    #[cfg(target_os = "windows")]
    fn texture_tensor(
        format: &str,
        width: usize,
        height: usize,
        access: u32,
    ) -> Option<*mut EfTensor> {
        if crate::probe::ef_is_gpu_buffer_available() == 0 {
            eprintln!("SKIP: no D3D11 device on this host");
            return None;
        }
        let f = std::ffi::CString::new(format).unwrap();
        let d = unsafe { crate::desc::ef_tensor_image_desc_new(width, height, f.as_ptr(), U8) };
        assert!(!d.is_null());
        assert_eq!(
            unsafe { crate::desc::ef_tensor_image_desc_set_memory(d, DMA_BUF) },
            0
        );
        assert_eq!(
            unsafe { crate::desc::ef_tensor_image_desc_set_access(d, access) },
            0
        );
        let t = unsafe { crate::desc::ef_tensor_image_desc_alloc(d) };
        unsafe { crate::desc::ef_tensor_image_desc_free(d) };
        assert!(!t.is_null(), "{format} {width}x{height} allocation failed");
        Some(t)
    }

    /// `CloseHandle` without pulling the `windows` crate into this leaf:
    /// the handles crossing this ABI are plain `void *`, and the test needs
    /// exactly one Win32 call to give the owned ones back.
    #[cfg(target_os = "windows")]
    fn close_handle(h: *mut c_void) {
        extern "system" {
            fn CloseHandle(h: *mut c_void) -> i32;
        }
        // SAFETY: `h` is an owned NT handle this test received from an
        // export documented to transfer ownership.
        assert_ne!(unsafe { CloseHandle(h) }, 0);
    }
}
