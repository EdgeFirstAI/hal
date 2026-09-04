// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Serialization: a `(blob, handles)` pair.
//!
//! A struct with an `fd` field is meaningless across a process boundary. File
//! descriptors travel out of band — `SCM_RIGHTS` on a unix socket, or
//! `pidfd_getfd` — and the receiver is handed *different numbers* than the
//! sender used. So export is a pair, exactly as Wayland, D-Bus and
//! `CameraFrame` are: a self-describing byte blob, plus the handles it refers
//! to by index.
//!
//! Use the opaque handle in-process and this across processes. Using the
//! blob in-process would force a `dup` per plane per call; passing the
//! handle itself (any EdgeFirst library reads it directly -- there is only
//! one implementation) avoids that entirely.

use std::ffi::c_int;
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor::{blob, TensorMemory};

use crate::handle::EfTensor;

/// Serialize a tensor into a caller-provided blob buffer and handle table.
///
/// Follows the standard two-call C pattern: pass `blob_cap`/`fds_cap` of 0 (or
/// `NULL` buffers) to learn the sizes required, then call again with buffers
/// that large. `blob_len` and `fds_len` are always written when non-`NULL`, so
/// a caller learns the requirement even from a failed call.
///
/// A `NULL` `fds_out` is a zero-capacity handle table, not an error: it is
/// refused only when the export actually produced handles. An export that
/// produces none — every inline export, and every export on a platform that
/// shares by handle value inside the blob rather than out of band, which is
/// all of them on Windows — therefore succeeds with `fds_out, fds_cap` of
/// `NULL, 0` and reports `*fds_len == 0`.
///
/// The transport mode is chosen from the tensor's storage: a backing with a
/// shareable handle is exported by reference, and one without — `mem`, `pbo` —
/// is inlined, because there is nothing to refer to. A caller that needs bytes
/// from a shareable tensor (sending over a network, where a handle is
/// meaningless off-host) should copy into a `mem` tensor first; a mode-selecting
/// variant can be appended later without breaking this signature.
///
/// @return 0 on success, `ENOSPC` when a buffer is too small (with the
///         required lengths written) -- a `NULL` `fds_out` on an export that
///         produced handles is that case, not a pointer refusal -- `EINVAL`
///         on a null tensor, a null `blob_len` or a null `fds_len`, `EIO` if
///         the tensor cannot be serialized.
///
/// # Safety
/// `blob` must be writable for `blob_cap` bytes and `fds` for `fds_cap` ints.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_export(
    t: *const EfTensor,
    blob_out: *mut u8,
    blob_cap: usize,
    blob_len: *mut usize,
    fds_out: *mut c_int,
    fds_cap: usize,
    fds_len: *mut usize,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if t.is_null() || blob_len.is_null() || fds_len.is_null() {
                return libc::EINVAL;
            }
            let Some(inner) = crate::handle::tensor_of(t) else {
                return libc::EINVAL;
            };
            // `mem` and `pbo` have no shareable handle at all, so transmitting one
            // means copying its bytes. Everything else refers.
            let mode = match inner.memory() {
                TensorMemory::Mem | TensorMemory::Pbo => blob::TransportMode::Inline,
                _ => blob::TransportMode::Reference,
            };
            let Ok((bytes, fds)) = blob::export(inner, mode) else {
                return libc::EIO;
            };

            // Written before the capacity check, so a caller that deliberately
            // passes zero capacity learns the requirement rather than an error
            // with no size.
            *blob_len = bytes.len();
            *fds_len = fds.len();

            // A null handle table is zero capacity rather than a separate
            // refusal, so an export with no handles is served by the one
            // `fds_cap < fds.len()` check instead of failing on the pointer.
            let fds_cap = if fds_out.is_null() { 0 } else { fds_cap };
            if blob_out.is_null() || blob_cap < bytes.len() || fds_cap < fds.len() {
                return libc::ENOSPC;
            }
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), blob_out, bytes.len());
            for (i, fd) in fds.iter().enumerate() {
                *fds_out.add(i) = *fd as c_int;
            }
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Reconstruct a tensor from a blob and its handle table.
///
/// **Import dups** every handle it retains, so the result is independent and
/// the sender may close its own copies as soon as this returns. There is no
/// keepalive protocol.
///
/// `blob` is untrusted — it may have arrived from another library, another
/// process, or a network — so every length and index inside it is validated
/// against the buffer and against `fds_len` before use.
///
/// @return a new tensor the caller must free with `ef_tensor_free`, or `NULL`.
///
/// # Safety
/// `blob` must be readable for `blob_len` bytes and `fds` for `fds_len` ints.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_import(
    blob_in: *const u8,
    blob_len: usize,
    fds_in: *const c_int,
    fds_len: usize,
) -> *mut EfTensor {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if blob_in.is_null() || blob_len == 0 {
                return std::ptr::null_mut();
            }
            if fds_in.is_null() && fds_len != 0 {
                return std::ptr::null_mut();
            }
            let bytes = std::slice::from_raw_parts(blob_in, blob_len);
            let fds: &[i32] = if fds_len == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(fds_in, fds_len)
            };
            match blob::import(bytes, fds) {
                Ok(t) => crate::handle::into_handle(t),
                Err(_) => std::ptr::null_mut(),
            }
        }))
        .unwrap_or(std::ptr::null_mut())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mem_tensor() -> *mut EfTensor {
        let dims = [3u64, 8];
        let t = unsafe { crate::handle::ef_tensor_new(0, dims.as_ptr(), 2) };
        assert!(!t.is_null());
        t
    }

    #[test]
    fn a_zero_capacity_call_reports_the_sizes_required() {
        // The standard two-call C pattern: ask, allocate, ask again.
        let t = mem_tensor();
        let (mut bl, mut fl) = (0usize, 0usize);
        let rc = unsafe {
            ef_tensor_export(
                t,
                std::ptr::null_mut(),
                0,
                &mut bl,
                std::ptr::null_mut(),
                0,
                &mut fl,
            )
        };
        assert_eq!(rc, libc::ENOSPC);
        assert!(
            bl > 0,
            "the caller must learn the blob size from a failed call"
        );
        unsafe { crate::handle::ef_tensor_free(t) };
    }

    #[test]
    fn a_two_call_export_then_import_round_trips() {
        let t = mem_tensor();
        let (mut bl, mut fl) = (0usize, 0usize);
        unsafe {
            ef_tensor_export(
                t,
                std::ptr::null_mut(),
                0,
                &mut bl,
                std::ptr::null_mut(),
                0,
                &mut fl,
            );
            let mut buf = vec![0u8; bl];
            let mut fds = vec![0 as c_int; fl.max(1)];
            let rc = ef_tensor_export(
                t,
                buf.as_mut_ptr(),
                buf.len(),
                &mut bl,
                fds.as_mut_ptr(),
                fds.len(),
                &mut fl,
            );
            assert_eq!(rc, 0);

            let got = ef_tensor_import(buf.as_ptr(), bl, fds.as_ptr(), fl);
            assert!(!got.is_null(), "a blob this library just wrote must import");
            assert_eq!(crate::handle::ef_tensor_ndim(got), 2);
            crate::handle::ef_tensor_free(got);
            crate::handle::ef_tensor_free(t);
        }
    }

    #[test]
    fn an_export_with_no_handles_accepts_a_null_handle_table() {
        // An inline export produces no handles, and neither does any export
        // on Windows, where the shareable handle travels inside the blob.
        // `NULL, 0` for the table is the ordinary call in that case and must
        // not be refused for a table there was nothing to write into.
        let t = mem_tensor();
        let (mut bl, mut fl) = (0usize, 0usize);
        unsafe {
            ef_tensor_export(
                t,
                std::ptr::null_mut(),
                0,
                &mut bl,
                std::ptr::null_mut(),
                0,
                &mut fl,
            );
            let mut buf = vec![0u8; bl];
            let rc = ef_tensor_export(
                t,
                buf.as_mut_ptr(),
                buf.len(),
                &mut bl,
                std::ptr::null_mut(),
                0,
                &mut fl,
            );
            assert_eq!(rc, 0, "a handleless export must accept a null fds_out");
            assert_eq!(fl, 0, "an inline export carries no handles");

            let got = ef_tensor_import(buf.as_ptr(), bl, std::ptr::null(), 0);
            assert!(!got.is_null(), "the blob that export wrote must import");
            crate::handle::ef_tensor_free(got);
            crate::handle::ef_tensor_free(t);
        }
    }

    #[test]
    fn a_buffer_one_byte_too_small_is_refused_not_truncated() {
        let t = mem_tensor();
        let (mut bl, mut fl) = (0usize, 0usize);
        unsafe {
            ef_tensor_export(
                t,
                std::ptr::null_mut(),
                0,
                &mut bl,
                std::ptr::null_mut(),
                0,
                &mut fl,
            );
            let mut buf = vec![0u8; bl - 1];
            let mut fds = vec![0 as c_int; 4];
            let rc = ef_tensor_export(
                t,
                buf.as_mut_ptr(),
                buf.len(),
                &mut bl,
                fds.as_mut_ptr(),
                fds.len(),
                &mut fl,
            );
            assert_eq!(
                rc,
                libc::ENOSPC,
                "a short buffer must not be partially filled"
            );
            crate::handle::ef_tensor_free(t);
        }
    }

    #[test]
    fn null_arguments_are_errors_not_crashes() {
        let (mut bl, mut fl) = (0usize, 0usize);
        unsafe {
            assert_eq!(
                ef_tensor_export(
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    0,
                    &mut bl,
                    std::ptr::null_mut(),
                    0,
                    &mut fl
                ),
                libc::EINVAL
            );
            let t = mem_tensor();
            assert_eq!(
                ef_tensor_export(
                    t,
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    0,
                    &mut fl
                ),
                libc::EINVAL,
                "a null out-length has nowhere to report the requirement"
            );
            crate::handle::ef_tensor_free(t);
            assert!(ef_tensor_import(std::ptr::null(), 0, std::ptr::null(), 0).is_null());
        }
    }

    #[test]
    fn a_garbage_blob_is_refused_not_trusted() {
        let junk = [0xABu8; 128];
        assert!(
            unsafe { ef_tensor_import(junk.as_ptr(), junk.len(), std::ptr::null(), 0) }.is_null()
        );
    }
}
