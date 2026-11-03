// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The plan entry points: `new` / `run` / `free` per operation.

use std::ffi::c_int;
use std::ptr;

use edgefirst_kernels as k;

use crate::error::{ffi, CError};
use crate::pool::CPool;
use crate::types::*;
use crate::view::{dtype_of, span, tensor_mut, tensor_ref};

fn box_format(raw: u32) -> Result<k::BoxFormat, CError> {
    EfKernelBoxFormat::from_raw(raw).ok_or(CError::Invalid("unknown box format value"))
}

/// Copies `n` levels from `ptr`.
///
/// # Safety
/// `ptr` must point to `n` readable levels when `n > 0`.
unsafe fn levels(ptr: *const EfKernelLevel, n: usize) -> Result<Vec<k::Level>, CError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(CError::Invalid("levels is NULL"));
    }
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| CError::NoMemory("cannot allocate levels"))?;
    // SAFETY: `ptr` is non-null and the caller guarantees `n` elements.
    v.extend(
        unsafe { std::slice::from_raw_parts(ptr, n) }
            .iter()
            .map(|l| k::Level::from(*l)),
    );
    Ok(v)
}

/// Sets `*out` to NULL when `out` is non-NULL, so every failure leaves it NULL.
///
/// # Safety
/// `out` must be NULL or writable.
unsafe fn clear_out<W>(out: *mut *mut W) {
    if !out.is_null() {
        // SAFETY: `out` is non-null and writable.
        unsafe { *out = ptr::null_mut() };
    }
}

/// Validates the pointers, builds the plan with `build` and hands it out boxed.
///
/// # Safety
/// `params` must be NULL or point to a valid `T`; `out` must be NULL or writable.
unsafe fn make<T, W>(
    params: *const T,
    out: *mut *mut W,
    build: impl FnOnce(&T) -> Result<W, CError>,
) -> Result<(), CError> {
    // SAFETY: forwarded from this function's contract.
    unsafe { clear_out(out) };
    if params.is_null() || out.is_null() {
        return Err(CError::Invalid("params and out must not be NULL"));
    }
    // SAFETY: `params` is non-null and valid.
    let plan = build(unsafe { &*params })?;
    // SAFETY: as above.
    unsafe { *out = Box::into_raw(Box::new(plan)) };
    Ok(())
}

/// Validates the pointers, converts the views and the pool, and calls `f`.
///
/// # Safety
/// Every non-NULL pointer must satisfy the contract of its C type.
pub(crate) unsafe fn run_plan<P>(
    plan: *const P,
    input: *const EfKernelView,
    output: *const EfKernelView,
    pool: *const EfKernelPool,
    f: impl FnOnce(&P, k::TensorRef<'_>, k::TensorMut<'_>, Option<&dyn k::Pool>) -> k::Result<()>,
) -> Result<(), CError> {
    if plan.is_null() || input.is_null() || output.is_null() {
        return Err(CError::Invalid("plan, input and output must not be NULL"));
    }
    // SAFETY: all three are non-null and valid per the C contract; `pool` is checked below.
    let (plan, input, output) = unsafe { (&*plan, &*input, &*output) };
    let pool = if pool.is_null() {
        None
    } else {
        // SAFETY: non-null and valid per the C contract.
        Some(CPool::new(unsafe { &*pool })?)
    };
    // SAFETY: the C contract makes the shape and strides arrays valid for the call.
    if unsafe { span(input)?.overlaps(&span(output)?) } {
        return Err(CError::Invalid("input and output views overlap"));
    }
    // SAFETY: the C contract makes the views valid for the call, and they do not overlap.
    let (input, output) = unsafe { (tensor_ref(input)?, tensor_mut(output)?) };
    f(
        plan,
        input,
        output,
        pool.as_ref().map(|p| p as &dyn k::Pool),
    )?;
    if pool.as_ref().is_some_and(CPool::panicked) {
        return Err(CError::Invalid("internal error"));
    }
    Ok(())
}

/// Frees a plan made by one of the `new` functions.
///
/// # Safety
/// `plan` must be NULL or a pointer from the matching `new` that is not used again.
unsafe fn free_plan<P>(plan: *mut P) {
    if plan.is_null() {
        return;
    }
    // A panic must not unwind into C; a plan's drop frees memory only, so one is not expected.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: the caller owns `plan`, which came from `Box::into_raw`.
        drop(unsafe { Box::from_raw(plan) });
    }));
}

/// Creates a plan for an int8 softmax over the last axis.
/// `beta` scales the logits; the output must be quantized (1/256, -128).
///
/// @param params Plan parameters; read during the call only.
/// @param out Receives the new plan, or NULL on failure.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown dtype, invalid
///         parameters) or ENOTSUP (valid but not handled), and a message for
///         ef_kernels_last_error().
///
/// # Safety
/// `params` must point to a valid params struct (and any array it points to) and
/// `out` to writable storage for one pointer.
#[no_mangle]
pub unsafe extern "C" fn ef_softmax_i8_plan_new(
    params: *const EfSoftmaxI8Params,
    out: *mut *mut EfSoftmaxI8Plan,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            make(params, out, |p| {
                Ok(EfSoftmaxI8Plan(k::SoftmaxI8Plan::new(
                    k::SoftmaxI8Params {
                        input: p.input.into(),
                        output: p.output.into(),
                        beta: p.beta,
                    },
                )?))
            })
        }
    })
}

/// Runs an int8 softmax over the last axis. Safe to call concurrently on one plan from several threads.
///
/// @param plan A plan from the matching `_plan_new`.
/// @param input Input tensor; must stay readable for the call.
/// @param output Output tensor; must stay writable for the call. Views whose memory spans
///        overlap the input's, interleaved views of one buffer included, are EINVAL.
/// @param pool Optional thread pool; NULL runs on the calling thread.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
///         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
///         ENOTSUP, and a message for ef_kernels_last_error().
///
/// # Safety
/// All pointers must be NULL or valid, and the views must address memory the caller owns.
#[no_mangle]
pub unsafe extern "C" fn ef_softmax_i8_plan_run(
    plan: *const EfSoftmaxI8Plan,
    input: *const EfKernelView,
    output: *const EfKernelView,
    pool: *const EfKernelPool,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe { run_plan(plan, input, output, pool, |p, i, o, pl| p.0.run(i, o, pl)) }
    })
}

/// Frees a plan. NULL is a no-op.
///
/// @param plan A plan from the matching `_plan_new`, not used afterwards.
///
/// # Safety
/// `plan` must be NULL or an unfreed pointer from the matching `_plan_new`.
#[no_mangle]
pub unsafe extern "C" fn ef_softmax_i8_plan_free(plan: *mut EfSoftmaxI8Plan) {
    // SAFETY: forwarded from this function's contract.
    unsafe { free_plan(plan) }
}

/// Creates a plan for an int16 softmax over the last axis.
///
/// @param params Plan parameters; read during the call only.
/// @param out Receives the new plan, or NULL on failure.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown dtype, invalid
///         parameters) or ENOTSUP (valid but not handled), and a message for
///         ef_kernels_last_error().
///
/// # Safety
/// `params` must point to a valid params struct (and any array it points to) and
/// `out` to writable storage for one pointer.
#[no_mangle]
pub unsafe extern "C" fn ef_softmax_i16_plan_new(
    params: *const EfSoftmaxI16Params,
    out: *mut *mut EfSoftmaxI16Plan,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            make(params, out, |p| {
                Ok(EfSoftmaxI16Plan(k::SoftmaxI16Plan::new(
                    k::SoftmaxI16Params {
                        input: p.input.into(),
                        output: p.output.into(),
                        beta: p.beta,
                    },
                )?))
            })
        }
    })
}

/// Runs an int16 softmax over the last axis. Safe to call concurrently on one plan from several threads.
///
/// @param plan A plan from the matching `_plan_new`.
/// @param input Input tensor; must stay readable for the call.
/// @param output Output tensor; must stay writable for the call. Views whose memory spans
///        overlap the input's, interleaved views of one buffer included, are EINVAL.
/// @param pool Optional thread pool; NULL runs on the calling thread.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
///         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
///         ENOTSUP, and a message for ef_kernels_last_error().
///
/// # Safety
/// All pointers must be NULL or valid, and the views must address memory the caller owns.
#[no_mangle]
pub unsafe extern "C" fn ef_softmax_i16_plan_run(
    plan: *const EfSoftmaxI16Plan,
    input: *const EfKernelView,
    output: *const EfKernelView,
    pool: *const EfKernelPool,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe { run_plan(plan, input, output, pool, |p, i, o, pl| p.0.run(i, o, pl)) }
    })
}

/// Frees a plan. NULL is a no-op.
///
/// @param plan A plan from the matching `_plan_new`, not used afterwards.
///
/// # Safety
/// `plan` must be NULL or an unfreed pointer from the matching `_plan_new`.
#[no_mangle]
pub unsafe extern "C" fn ef_softmax_i16_plan_free(plan: *mut EfSoftmaxI16Plan) {
    // SAFETY: forwarded from this function's contract.
    unsafe { free_plan(plan) }
}

/// Creates a plan for an elementwise sigmoid.
///
/// @param params Plan parameters; read during the call only.
/// @param out Receives the new plan, or NULL on failure.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown dtype, invalid
///         parameters) or ENOTSUP (valid but not handled), and a message for
///         ef_kernels_last_error().
///
/// # Safety
/// `params` must point to a valid params struct (and any array it points to) and
/// `out` to writable storage for one pointer.
#[no_mangle]
pub unsafe extern "C" fn ef_sigmoid_plan_new(
    params: *const EfSigmoidParams,
    out: *mut *mut EfSigmoidPlan,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            make(params, out, |p| {
                Ok(EfSigmoidPlan(k::SigmoidPlan::new(k::SigmoidParams {
                    input_dtype: dtype_of(p.input_dtype)?,
                    input: p.input.into(),
                    output_dtype: dtype_of(p.output_dtype)?,
                    output: p.output.into(),
                })?))
            })
        }
    })
}

/// Runs an elementwise sigmoid. Safe to call concurrently on one plan from several threads.
///
/// @param plan A plan from the matching `_plan_new`.
/// @param input Input tensor; must stay readable for the call.
/// @param output Output tensor; must stay writable for the call. Views whose memory spans
///        overlap the input's, interleaved views of one buffer included, are EINVAL.
/// @param pool Optional thread pool; NULL runs on the calling thread.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
///         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
///         ENOTSUP, and a message for ef_kernels_last_error().
///
/// # Safety
/// All pointers must be NULL or valid, and the views must address memory the caller owns.
#[no_mangle]
pub unsafe extern "C" fn ef_sigmoid_plan_run(
    plan: *const EfSigmoidPlan,
    input: *const EfKernelView,
    output: *const EfKernelView,
    pool: *const EfKernelPool,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe { run_plan(plan, input, output, pool, |p, i, o, pl| p.0.run(i, o, pl)) }
    })
}

/// Frees a plan. NULL is a no-op.
///
/// @param plan A plan from the matching `_plan_new`, not used afterwards.
///
/// # Safety
/// `plan` must be NULL or an unfreed pointer from the matching `_plan_new`.
#[no_mangle]
pub unsafe extern "C" fn ef_sigmoid_plan_free(plan: *mut EfSigmoidPlan) {
    // SAFETY: forwarded from this function's contract.
    unsafe { free_plan(plan) }
}

/// Creates a plan for a distribution-focal-loss decode.
///
/// @param params Plan parameters; read during the call only.
/// @param out Receives the new plan, or NULL on failure.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown dtype, invalid
///         parameters) or ENOTSUP (valid but not handled), and a message for
///         ef_kernels_last_error().
///
/// # Safety
/// `params` must point to a valid params struct (and any array it points to) and
/// `out` to writable storage for one pointer.
#[no_mangle]
pub unsafe extern "C" fn ef_dfl_plan_new(
    params: *const EfDflParams,
    out: *mut *mut EfDflPlan,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            make(params, out, |p| {
                Ok(EfDflPlan(k::DflPlan::new(k::DflParams {
                    input_dtype: dtype_of(p.input_dtype)?,
                    input: p.input.into(),
                    output_dtype: dtype_of(p.output_dtype)?,
                    output: p.output.into(),
                    reg_max: p.reg_max,
                })?))
            })
        }
    })
}

/// Runs a distribution-focal-loss decode. Safe to call concurrently on one plan from several threads.
///
/// @param plan A plan from the matching `_plan_new`.
/// @param input Input tensor; must stay readable for the call.
/// @param output Output tensor; must stay writable for the call. Views whose memory spans
///        overlap the input's, interleaved views of one buffer included, are EINVAL.
/// @param pool Optional thread pool; NULL runs on the calling thread.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
///         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
///         ENOTSUP, and a message for ef_kernels_last_error().
///
/// # Safety
/// All pointers must be NULL or valid, and the views must address memory the caller owns.
#[no_mangle]
pub unsafe extern "C" fn ef_dfl_plan_run(
    plan: *const EfDflPlan,
    input: *const EfKernelView,
    output: *const EfKernelView,
    pool: *const EfKernelPool,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe { run_plan(plan, input, output, pool, |p, i, o, pl| p.0.run(i, o, pl)) }
    })
}

/// Frees a plan. NULL is a no-op.
///
/// @param plan A plan from the matching `_plan_new`, not used afterwards.
///
/// # Safety
/// `plan` must be NULL or an unfreed pointer from the matching `_plan_new`.
#[no_mangle]
pub unsafe extern "C" fn ef_dfl_plan_free(plan: *mut EfDflPlan) {
    // SAFETY: forwarded from this function's contract.
    unsafe { free_plan(plan) }
}

/// Creates a plan for a distance-to-box decode.
///
/// @param params Plan parameters; read during the call only.
/// @param out Receives the new plan, or NULL on failure.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown dtype, invalid
///         parameters) or ENOTSUP (valid but not handled), and a message for
///         ef_kernels_last_error().
///
/// # Safety
/// `params` must point to a valid params struct (and any array it points to) and
/// `out` to writable storage for one pointer.
#[no_mangle]
pub unsafe extern "C" fn ef_dist2bbox_plan_new(
    params: *const EfDist2bboxParams,
    out: *mut *mut EfDist2bboxPlan,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            make(params, out, |p| {
                Ok(EfDist2bboxPlan(k::Dist2BboxPlan::new(
                    k::Dist2BboxParams {
                        input_dtype: dtype_of(p.input_dtype)?,
                        input: p.input.into(),
                        output_dtype: dtype_of(p.output_dtype)?,
                        output: p.output.into(),
                        levels: levels(p.levels, p.n_levels)?,
                        format: box_format(p.box_format)?,
                    },
                )?))
            })
        }
    })
}

/// Runs a distance-to-box decode. Safe to call concurrently on one plan from several threads.
///
/// @param plan A plan from the matching `_plan_new`.
/// @param input Input tensor; must stay readable for the call.
/// @param output Output tensor; must stay writable for the call. Views whose memory spans
///        overlap the input's, interleaved views of one buffer included, are EINVAL.
/// @param pool Optional thread pool; NULL runs on the calling thread.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
///         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
///         ENOTSUP, and a message for ef_kernels_last_error().
///
/// # Safety
/// All pointers must be NULL or valid, and the views must address memory the caller owns.
#[no_mangle]
pub unsafe extern "C" fn ef_dist2bbox_plan_run(
    plan: *const EfDist2bboxPlan,
    input: *const EfKernelView,
    output: *const EfKernelView,
    pool: *const EfKernelPool,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe { run_plan(plan, input, output, pool, |p, i, o, pl| p.0.run(i, o, pl)) }
    })
}

/// Frees a plan. NULL is a no-op.
///
/// @param plan A plan from the matching `_plan_new`, not used afterwards.
///
/// # Safety
/// `plan` must be NULL or an unfreed pointer from the matching `_plan_new`.
#[no_mangle]
pub unsafe extern "C" fn ef_dist2bbox_plan_free(plan: *mut EfDist2bboxPlan) {
    // SAFETY: forwarded from this function's contract.
    unsafe { free_plan(plan) }
}

/// Creates a plan for the fused DFL plus distance-to-box decode.
///
/// @param params Plan parameters; read during the call only.
/// @param out Receives the new plan, or NULL on failure.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown dtype, invalid
///         parameters) or ENOTSUP (valid but not handled), and a message for
///         ef_kernels_last_error().
///
/// # Safety
/// `params` must point to a valid params struct (and any array it points to) and
/// `out` to writable storage for one pointer.
#[no_mangle]
pub unsafe extern "C" fn ef_dfl_box_plan_new(
    params: *const EfDflBoxParams,
    out: *mut *mut EfDflBoxPlan,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            make(params, out, |p| {
                Ok(EfDflBoxPlan(k::DflBoxPlan::new(k::DflBoxParams {
                    input_dtype: dtype_of(p.input_dtype)?,
                    input: p.input.into(),
                    output_dtype: dtype_of(p.output_dtype)?,
                    output: p.output.into(),
                    reg_max: p.reg_max,
                    levels: levels(p.levels, p.n_levels)?,
                    format: box_format(p.box_format)?,
                })?))
            })
        }
    })
}

/// Runs the fused DFL plus distance-to-box decode. Safe to call concurrently on one plan from several threads.
///
/// @param plan A plan from the matching `_plan_new`.
/// @param input Input tensor; must stay readable for the call.
/// @param output Output tensor; must stay writable for the call. Views whose memory spans
///        overlap the input's, interleaved views of one buffer included, are EINVAL.
/// @param pool Optional thread pool; NULL runs on the calling thread.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
///         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
///         ENOTSUP, and a message for ef_kernels_last_error().
///
/// # Safety
/// All pointers must be NULL or valid, and the views must address memory the caller owns.
#[no_mangle]
pub unsafe extern "C" fn ef_dfl_box_plan_run(
    plan: *const EfDflBoxPlan,
    input: *const EfKernelView,
    output: *const EfKernelView,
    pool: *const EfKernelPool,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe { run_plan(plan, input, output, pool, |p, i, o, pl| p.0.run(i, o, pl)) }
    })
}

/// Frees a plan. NULL is a no-op.
///
/// @param plan A plan from the matching `_plan_new`, not used afterwards.
///
/// # Safety
/// `plan` must be NULL or an unfreed pointer from the matching `_plan_new`.
#[no_mangle]
pub unsafe extern "C" fn ef_dfl_box_plan_free(plan: *mut EfDflBoxPlan) {
    // SAFETY: forwarded from this function's contract.
    unsafe { free_plan(plan) }
}

/// Creates a plan that maps every 8-bit element through a 256-entry table.
///
/// @param dtype `EF_KERNEL_DTYPE_I8` or `EF_KERNEL_DTYPE_U8`.
/// @param table 256 bytes, indexed by the element reinterpreted as `uint8_t`.
/// @param out Receives the new plan, or NULL on failure.
/// @return 0 on success; -1 with errno EINVAL or ENOTSUP, and a message for
///         ef_kernels_last_error().
///
/// # Safety
/// `table` must be NULL or point to 256 readable bytes; `out` must be NULL or writable.
#[no_mangle]
pub unsafe extern "C" fn ef_lut_i8_plan_new_table(
    dtype: u32,
    table: *const u8,
    out: *mut *mut EfLutI8Plan,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe { clear_out(out) };
        if table.is_null() {
            return Err(CError::Invalid("table must not be NULL"));
        }
        // SAFETY: `table` is non-null and the caller guarantees 256 bytes.
        let table = unsafe { std::slice::from_raw_parts(table, 256) };
        let mut t = [0u8; 256];
        t.copy_from_slice(table);
        let dtype = dtype_of(dtype)?;
        // SAFETY: forwarded from this function's contract.
        unsafe {
            make(&raw const t, out, |t| {
                Ok(EfLutI8Plan(k::LutI8Plan::from_table(dtype, *t)?))
            })
        }
    })
}

/// Creates a plan that applies a quantized sigmoid through a table built from the quantization.
///
/// @param dtype `EF_KERNEL_DTYPE_I8` or `EF_KERNEL_DTYPE_U8`.
/// @param input Input quantization.
/// @param output Output quantization.
/// @param out Receives the new plan, or NULL on failure.
/// @return 0 on success; -1 with errno EINVAL or ENOTSUP, and a message for
///         ef_kernels_last_error().
///
/// # Safety
/// `out` must be NULL or writable.
#[no_mangle]
pub unsafe extern "C" fn ef_lut_i8_plan_new_sigmoid(
    dtype: u32,
    input: EfKernelQuant,
    output: EfKernelQuant,
    out: *mut *mut EfLutI8Plan,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe { clear_out(out) };
        let dtype = dtype_of(dtype)?;
        // SAFETY: forwarded from this function's contract.
        unsafe {
            make(&raw const dtype, out, |d| {
                Ok(EfLutI8Plan(k::LutI8Plan::sigmoid(
                    *d,
                    input.into(),
                    output.into(),
                )?))
            })
        }
    })
}

/// Runs a table plan. Safe to call concurrently on one plan from several threads.
///
/// @param plan A plan from `ef_lut_i8_plan_new_table` or `ef_lut_i8_plan_new_sigmoid`.
/// @param input Input tensor of the plan's dtype.
/// @param output Output tensor of the same dtype and shape. Views whose memory spans
///        overlap the input's, interleaved views of one buffer included, are EINVAL.
/// @param pool Optional thread pool; NULL runs on the calling thread.
/// @return 0 on success; -1 with errno EINVAL or ENOTSUP, and a message for
///         ef_kernels_last_error().
///
/// # Safety
/// All pointers must be NULL or valid, and the views must address memory the caller owns.
#[no_mangle]
pub unsafe extern "C" fn ef_lut_i8_plan_run(
    plan: *const EfLutI8Plan,
    input: *const EfKernelView,
    output: *const EfKernelView,
    pool: *const EfKernelPool,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe { run_plan(plan, input, output, pool, |p, i, o, pl| p.0.run(i, o, pl)) }
    })
}

/// Frees a table plan. NULL is a no-op.
///
/// @param plan A plan from a `ef_lut_i8_plan_new_*` function, not used afterwards.
///
/// # Safety
/// `plan` must be NULL or an unfreed pointer from a `ef_lut_i8_plan_new_*` function.
#[no_mangle]
pub unsafe extern "C" fn ef_lut_i8_plan_free(plan: *mut EfLutI8Plan) {
    // SAFETY: forwarded from this function's contract.
    unsafe { free_plan(plan) }
}
