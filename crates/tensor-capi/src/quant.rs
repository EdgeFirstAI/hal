// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Quantization on a live handle: `ef_tensor_quantization_info`,
//! `ef_tensor_quantization_get`, `ef_tensor_quantization_set`,
//! `ef_tensor_quantization_clear`.
//!
//! `Quantization` (`edgefirst_tensor::Quantization`) is variable-length --
//! an axis plus per-axis `scales` and `zero_points` -- so unlike
//! `ef_tensor_plane`/`ef_tensor_view_origin` it does not fit a single fixed
//! `#[repr(C)]` scalar block a getter can fill in one call. Reading it is
//! therefore the standard two-call idiom: [`ef_tensor_quantization_info`]
//! reports whether quantization is attached and how many entries it has,
//! then [`ef_tensor_quantization_get`] fills caller-provided `scales`/`zps`
//! buffers sized from that count.
//!
//! `has_quantization` in [`EfQuantizationInfo`] is a presence flag, not a
//! sentinel -- axis `0` and a scale of `0.0` are both legitimate values, the
//! same reasoning as `EfViewOrigin::has_origin`
//! (`edgefirst-tensor-abi/src/lib.rs`).
//!
//! `axis` on the wire is `-1` for per-tensor (no axis) and `>= 0` for
//! per-channel, matching `ef_tensor_builder_quantization`'s own encoding
//! (minus that function's `-2` "clear" sentinel -- here clearing is its own
//! entry point, [`ef_tensor_quantization_clear`], not a magic axis value).
//! `zero_point` is always readable as a full array: a symmetric
//! quantization (`Quantization`'s `zero_point` field is `None`) is
//! dequantized identically to an explicit all-zero array
//! (`scale * (q - 0)`), so [`ef_tensor_quantization_get`] fills `zps` with
//! zeros in that case rather than exposing a second "has zero point" flag
//! nothing downstream needs.
//!
//! # Concurrency
//!
//! [`ef_tensor_quantization_info`]/[`ef_tensor_quantization_get`] are
//! read-only accessors (shared `&EfTensorImpl`, the same shape as
//! `ef_tensor_colorimetry`/`ef_tensor_view_origin`) and safe to call from
//! any thread holding a valid reference, concurrently with each other or
//! with any other read-only accessor.
//!
//! [`ef_tensor_quantization_set`]/[`ef_tensor_quantization_clear`] mutate
//! `inner` directly through [`crate::handle::imp_mut`] and carry the same
//! narrower constraint `mutate.rs`'s setters do: **not safe to call
//! concurrently with any other `tensor-capi` call on the same handle from
//! another thread.** See `mutate.rs`'s module docs for the full reasoning
//! and why every real caller (quantization attached once at model load,
//! read many times per frame afterward -- `edgefirst-decoder`'s
//! `per_scale/helper.rs`/`pipeline.rs`) fits that shape.

use std::ffi::c_int;
use std::panic::{catch_unwind, AssertUnwindSafe};

use edgefirst_tensor::Quantization;
use edgefirst_tensor_abi::EfQuantizationInfo;

use crate::handle::{imp_mut, tensor_of, EfTensor};
use crate::last_error::set_last_error;
use crate::map::errno_for;

/// Report whether a live handle carries quantization metadata, and how many
/// `scale`/`zero_point` entries it has.
///
/// The first half of the two-call idiom; see this module's docs.
///
/// @retval 0 success (`out` is always fully written, whether or not
///         quantization is present).
/// @retval EINVAL `t` or `out` is `NULL`.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `out` must be writable for one
/// `ef_quantization_info`.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_quantization_info(
    t: *const EfTensor,
    out: *mut EfQuantizationInfo,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if t.is_null() || out.is_null() {
                set_last_error("quantization_info: null tensor or out");
                return libc::EINVAL;
            }
            let Some(inner) = tensor_of(t) else {
                set_last_error("quantization_info: could not resolve handle");
                return libc::EINVAL;
            };
            *out = match inner.quantization() {
                Some(q) => EfQuantizationInfo {
                    axis: q.axis().map(|a| a as i32).unwrap_or(-1),
                    count: q.scale().len() as u32,
                    has_quantization: 1,
                },
                None => EfQuantizationInfo::default(),
            };
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Fill caller-provided buffers with a live handle's quantization scales and
/// zero-points. The second half of the two-call idiom; `n` must equal the
/// `count` [`ef_tensor_quantization_info`] reported.
///
/// @retval 0 success; `scales[0..n]` is filled, and `zps[0..n]` too when
///         `zps` is non-`NULL` (zero-filled for a symmetric quantization).
/// @retval EINVAL `t` or `scales` is `NULL`, this tensor has no quantization
///         attached, or `n` does not match its actual entry count.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `scales` must point to `n` writable
/// `float`s; `zps` must be `NULL` or point to `n` writable `int`s.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_quantization_get(
    t: *const EfTensor,
    scales: *mut f32,
    zps: *mut i32,
    n: u32,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if t.is_null() || scales.is_null() {
                set_last_error("quantization_get: null tensor or scales buffer");
                return libc::EINVAL;
            }
            let Some(inner) = tensor_of(t) else {
                set_last_error("quantization_get: could not resolve handle");
                return libc::EINVAL;
            };
            let Some(q) = inner.quantization() else {
                set_last_error("quantization_get: this tensor has no quantization attached");
                return libc::EINVAL;
            };
            let scale = q.scale();
            if scale.len() != n as usize {
                set_last_error(&format!(
                    "quantization_get: n={n} does not match this tensor's entry count \
                 ({}) -- call ef_tensor_quantization_info first",
                    scale.len()
                ));
                return libc::EINVAL;
            }
            let out_scales = std::slice::from_raw_parts_mut(scales, n as usize);
            out_scales.copy_from_slice(scale);
            if !zps.is_null() {
                let out_zps = std::slice::from_raw_parts_mut(zps, n as usize);
                match q.zero_point() {
                    Some(z) => out_zps.copy_from_slice(z),
                    // Symmetric: dequantizes identically to an explicit all-zero
                    // array (scale * (q - 0)) -- see this module's docs.
                    None => out_zps.fill(0),
                }
            }
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Attach quantization metadata to a live handle. `axis` is `-1` for
/// per-tensor (`n` must be 1) or `>= 0` for per-channel; `zps` may be `NULL`
/// for symmetric quantization (zero-point implicitly 0).
///
/// Only meaningful for an integer-dtype tensor; a float tensor is refused,
/// matching [`edgefirst_tensor::TensorDyn::set_quantization`]'s own
/// `QuantizationInvalid { field: "dtype_is_integer", .. }` refusal.
///
/// @retval 0 success.
/// @retval EINVAL `t` or `scales` is `NULL`, `n == 0`, `axis` is `< -1`,
///         `axis == -1` with `n != 1`, or the backend rejects the resulting
///         `Quantization` (dtype not integer, or `axis`/`n` incompatible
///         with the tensor's shape).
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread -- see this module's docs.
///
/// # Safety
/// `t` must be `NULL` or a live handle; `scales` must point to `n` readable
/// `float`s; `zps` must be `NULL` or point to `n` readable `int`s.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_quantization_set(
    t: *mut EfTensor,
    axis: i32,
    scales: *const f32,
    zps: *const i32,
    n: u32,
) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if t.is_null() || scales.is_null() || n == 0 {
                set_last_error("quantization_set: null tensor/scales, or n == 0");
                return libc::EINVAL;
            }
            if axis == -1 && n != 1 {
                set_last_error("quantization_set: axis=-1 (per-tensor) requires n == 1");
                return libc::EINVAL;
            }
            if axis < -1 {
                set_last_error(
                    "quantization_set: axis must be -1 (per-tensor) or >= 0 (per-channel)",
                );
                return libc::EINVAL;
            }
            let scale_vec = std::slice::from_raw_parts(scales, n as usize).to_vec();
            let zp_vec =
                (!zps.is_null()).then(|| std::slice::from_raw_parts(zps, n as usize).to_vec());
            let q = match (axis, zp_vec) {
                (-1, None) => Quantization::per_tensor_symmetric(scale_vec[0]),
                (-1, Some(z)) => Quantization::per_tensor(scale_vec[0], z[0]),
                (a, None) => match Quantization::per_channel_symmetric(scale_vec, a as usize) {
                    Ok(q) => q,
                    Err(e) => {
                        let errno = errno_for(&e);
                        set_last_error(&format!("quantization_set: {e}"));
                        return errno;
                    }
                },
                (a, Some(z)) => match Quantization::per_channel(scale_vec, z, a as usize) {
                    Ok(q) => q,
                    Err(e) => {
                        let errno = errno_for(&e);
                        set_last_error(&format!("quantization_set: {e}"));
                        return errno;
                    }
                },
            };
            let Some(imp) = imp_mut(t) else {
                set_last_error("quantization_set: could not resolve handle");
                return libc::EINVAL;
            };
            match imp.inner.set_quantization(q) {
                Ok(()) => 0,
                Err(e) => {
                    let errno = errno_for(&e);
                    set_last_error(&format!("quantization_set: {e}"));
                    errno
                }
            }
        }))
        .unwrap_or(libc::EINVAL)
    }
}

/// Clear any quantization metadata on a live handle. A no-op if none is
/// attached.
///
/// @retval 0 success.
/// @retval EINVAL `t` is `NULL`.
///
/// @warning Not safe to call concurrently with any other `tensor-capi` call
/// on the same handle from another thread -- see this module's docs.
///
/// # Safety
/// `t` must be `NULL` or a live handle.
#[no_mangle]
pub unsafe extern "C" fn ef_tensor_quantization_clear(t: *mut EfTensor) -> c_int {
    unsafe {
        catch_unwind(AssertUnwindSafe(|| {
            if t.is_null() {
                set_last_error("quantization_clear: null tensor");
                return libc::EINVAL;
            }
            let Some(imp) = imp_mut(t) else {
                set_last_error("quantization_clear: could not resolve handle");
                return libc::EINVAL;
            };
            imp.inner.clear_quantization();
            0
        }))
        .unwrap_or(libc::EINVAL)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handle::{ef_tensor_free, ef_tensor_new};

    fn t() -> *mut EfTensor {
        let dims = [4u64, 4];
        unsafe { ef_tensor_new(0, dims.as_ptr(), 2) } // U8
    }

    #[test]
    fn info_reports_absence_by_flag_on_a_fresh_tensor() {
        let h = t();
        let mut info = EfQuantizationInfo::default();
        assert_eq!(unsafe { ef_tensor_quantization_info(h, &mut info) }, 0);
        assert_eq!(info.has_quantization, 0);
        unsafe { ef_tensor_free(h) };
    }

    #[test]
    fn set_then_info_then_get_round_trips_per_tensor_asymmetric() {
        let h = t();
        let scales = [0.5f32];
        let zps = [3i32];
        assert_eq!(
            unsafe { ef_tensor_quantization_set(h, -1, scales.as_ptr(), zps.as_ptr(), 1) },
            0
        );

        let mut info = EfQuantizationInfo::default();
        assert_eq!(unsafe { ef_tensor_quantization_info(h, &mut info) }, 0);
        assert_eq!(info.has_quantization, 1);
        assert_eq!(info.axis, -1);
        assert_eq!(info.count, 1);

        let mut out_scale = [0f32; 1];
        let mut out_zp = [0i32; 1];
        assert_eq!(
            unsafe {
                ef_tensor_quantization_get(h, out_scale.as_mut_ptr(), out_zp.as_mut_ptr(), 1)
            },
            0
        );
        assert_eq!(out_scale, [0.5]);
        assert_eq!(out_zp, [3]);
        unsafe { ef_tensor_free(h) };
    }

    #[test]
    fn set_per_channel_symmetric_reads_back_zero_filled_zero_points() {
        let h = t();
        // Per-channel over axis 1 (width=4), matching the tensor's own shape.
        let scales = [0.1f32, 0.2, 0.3, 0.4];
        assert_eq!(
            unsafe { ef_tensor_quantization_set(h, 1, scales.as_ptr(), std::ptr::null(), 4) },
            0
        );
        let mut info = EfQuantizationInfo::default();
        assert_eq!(unsafe { ef_tensor_quantization_info(h, &mut info) }, 0);
        assert_eq!(info.axis, 1);
        assert_eq!(info.count, 4);

        let mut out_scale = [0f32; 4];
        let mut out_zp = [7i32; 4]; // sentinel to prove it gets zero-filled
        assert_eq!(
            unsafe {
                ef_tensor_quantization_get(h, out_scale.as_mut_ptr(), out_zp.as_mut_ptr(), 4)
            },
            0
        );
        assert_eq!(out_scale, scales);
        assert_eq!(out_zp, [0, 0, 0, 0]);
        unsafe { ef_tensor_free(h) };
    }

    #[test]
    fn clear_removes_previously_set_quantization() {
        let h = t();
        let scales = [1.0f32];
        unsafe { ef_tensor_quantization_set(h, -1, scales.as_ptr(), std::ptr::null(), 1) };
        assert_eq!(unsafe { ef_tensor_quantization_clear(h) }, 0);
        let mut info = EfQuantizationInfo::default();
        assert_eq!(unsafe { ef_tensor_quantization_info(h, &mut info) }, 0);
        assert_eq!(info.has_quantization, 0);
        unsafe { ef_tensor_free(h) };
    }

    #[test]
    fn get_rejects_a_count_mismatch() {
        let h = t();
        let scales = [1.0f32];
        unsafe { ef_tensor_quantization_set(h, -1, scales.as_ptr(), std::ptr::null(), 1) };
        let mut out = [0f32; 2];
        assert_eq!(
            unsafe { ef_tensor_quantization_get(h, out.as_mut_ptr(), std::ptr::null_mut(), 2) },
            libc::EINVAL
        );
        unsafe { ef_tensor_free(h) };
    }

    #[test]
    fn get_rejects_a_tensor_with_no_quantization() {
        let h = t();
        let mut out = [0f32; 1];
        assert_eq!(
            unsafe { ef_tensor_quantization_get(h, out.as_mut_ptr(), std::ptr::null_mut(), 1) },
            libc::EINVAL
        );
        unsafe { ef_tensor_free(h) };
    }

    #[test]
    fn set_on_a_float_tensor_is_rejected() {
        let dims = [4u64, 4];
        let h =
            unsafe { ef_tensor_new(edgefirst_tensor_abi::EfDtype::F32 as u32, dims.as_ptr(), 2) };
        let scales = [1.0f32];
        assert_eq!(
            unsafe { ef_tensor_quantization_set(h, -1, scales.as_ptr(), std::ptr::null(), 1) },
            libc::EINVAL
        );
        unsafe { ef_tensor_free(h) };
    }

    #[test]
    fn every_entry_point_rejects_a_null_handle() {
        let mut info = EfQuantizationInfo::default();
        let scales = [1.0f32];
        let mut out = [0f32; 1];
        assert_eq!(
            unsafe { ef_tensor_quantization_info(std::ptr::null(), &mut info) },
            libc::EINVAL
        );
        assert_eq!(
            unsafe {
                ef_tensor_quantization_get(
                    std::ptr::null(),
                    out.as_mut_ptr(),
                    std::ptr::null_mut(),
                    1,
                )
            },
            libc::EINVAL
        );
        assert_eq!(
            unsafe {
                ef_tensor_quantization_set(
                    std::ptr::null_mut(),
                    -1,
                    scales.as_ptr(),
                    std::ptr::null(),
                    1,
                )
            },
            libc::EINVAL
        );
        assert_eq!(
            unsafe { ef_tensor_quantization_clear(std::ptr::null_mut()) },
            libc::EINVAL
        );
    }
}
