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
    // SAFETY: forwarded from this function's contract.
    unsafe {
        run_plan_n(plan, [input], output, pool, |p, [i], o, pl| {
            Ok(f(p, i?, o, pl)?)
        })
    }
}

/// [`run_plan`] for `N` inputs: rejects an output whose memory span overlaps any input's
/// (inputs may overlap each other), converts the views and the pool, and calls `f` with each
/// input's conversion result, before anything is written.
///
/// # Safety
/// Every non-NULL pointer must satisfy the contract of its C type.
pub(crate) unsafe fn run_plan_n<P, const N: usize>(
    plan: *const P,
    inputs: [*const EfKernelView; N],
    output: *const EfKernelView,
    pool: *const EfKernelPool,
    f: impl FnOnce(
        &P,
        [Result<k::TensorRef<'_>, CError>; N],
        k::TensorMut<'_>,
        Option<&dyn k::Pool>,
    ) -> Result<(), CError>,
) -> Result<(), CError> {
    if plan.is_null() || output.is_null() || inputs.iter().any(|i| i.is_null()) {
        return Err(CError::Invalid("plan, inputs and output must not be NULL"));
    }
    // SAFETY: all are non-null and valid per the C contract; `pool` is checked below.
    let (plan, output) = unsafe { (&*plan, &*output) };
    // SAFETY: as above.
    let inputs: [&EfKernelView; N] = std::array::from_fn(|i| unsafe { &*inputs[i] });
    let pool = if pool.is_null() {
        None
    } else {
        // SAFETY: non-null and valid per the C contract.
        Some(CPool::new(unsafe { &*pool })?)
    };
    // SAFETY: the C contract makes the shape and strides arrays valid for the call.
    let out_span = unsafe { span(output)? };
    for i in inputs {
        // SAFETY: as above.
        if unsafe { span(i)? }.overlaps(&out_span) {
            return Err(CError::Invalid("input and output views overlap"));
        }
    }
    // SAFETY: the C contract makes the views valid for the call, and the output overlaps no
    // input.
    let output = unsafe { tensor_mut(output)? };
    // SAFETY: as above; inputs are only read, so they may share memory.
    let inputs = inputs.map(|i| unsafe { tensor_ref(i) });
    f(
        plan,
        inputs,
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

fn tflite_kernels(raw: u32) -> Result<k::TfliteKernels, CError> {
    EfTfliteKernels::from_raw(raw).ok_or(CError::Invalid("unknown tflite kernels value"))
}

fn activation(raw: u32) -> Result<k::Activation, CError> {
    EfKernelActivation::from_raw(raw).ok_or(CError::Invalid("unknown activation value"))
}

/// Copies `n` elements from `ptr`, converted by `map`.
///
/// # Safety
/// `ptr` must point to `n` readable elements when `n > 0`.
unsafe fn copy_array<T: Copy, U>(
    ptr: *const T,
    n: usize,
    null: &'static str,
    map: impl Fn(T) -> U,
) -> Result<Vec<U>, CError> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if ptr.is_null() {
        return Err(CError::Invalid(null));
    }
    let mut v = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| CError::NoMemory("cannot allocate a parameter array"))?;
    // SAFETY: `ptr` is non-null and the caller guarantees `n` elements.
    v.extend(
        unsafe { std::slice::from_raw_parts(ptr, n) }
            .iter()
            .map(|&x| map(x)),
    );
    Ok(v)
}

/// The kernels params of a C matmul params struct, its arrays copied.
///
/// # Safety
/// The struct's array pointers must satisfy its documented contract.
unsafe fn matmul_params(p: &EfMatmulI8Params) -> Result<k::MatMulI8Params, CError> {
    Ok(k::MatMulI8Params {
        lhs_zero_point: p.lhs_zero_point,
        rhs_zero_point: p.rhs_zero_point,
        output: p.output.into(),
        // SAFETY: forwarded from this function's contract.
        requant: unsafe {
            copy_array(p.requant, p.n_requant, "requant is NULL", k::Requant::from)?
        },
        // SAFETY: as above.
        bias: unsafe { copy_array(p.bias, p.n_bias, "bias is NULL", |b| b)? },
        activation: activation(p.activation)?,
        kernels: tflite_kernels(p.kernels)?,
    })
}

/// Writes a requantization made by `make` to `out`.
///
/// # Safety
/// `out` must be NULL or writable.
unsafe fn requant_out(
    out: *mut EfKernelRequant,
    make: impl FnOnce() -> k::Result<k::Requant>,
) -> c_int {
    ffi(|| {
        if out.is_null() {
            return Err(CError::Invalid("out must not be NULL"));
        }
        let r = make()?;
        // SAFETY: `out` is non-null and writable.
        unsafe { *out = r.into() };
        Ok(())
    })
}

/// Computes the requantization of TFLite's int8 `BATCH_MATMUL` (and per-tensor
/// `FULLY_CONNECTED`): `(lhs_scale * rhs_scale)` in float, then divided by `output_scale` in
/// double, as `QuantizeMultiplier`.
///
/// @param lhs_scale Left operand scale.
/// @param rhs_scale Right operand scale.
/// @param output_scale Output scale.
/// @param out Receives the requantization.
/// @return 0 on success; -1 with errno EINVAL (NULL `out`, a multiplier that is not finite
///         and positive), and a message for ef_kernels_last_error().
///
/// # Safety
/// `out` must be NULL or writable.
#[no_mangle]
pub unsafe extern "C" fn ef_kernel_requant_tflite_batch_matmul(
    lhs_scale: f32,
    rhs_scale: f32,
    output_scale: f32,
    out: *mut EfKernelRequant,
) -> c_int {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        requant_out(out, || {
            k::Requant::tflite_batch_matmul(lhs_scale, rhs_scale, output_scale)
        })
    }
}

/// Computes the requantization of one output channel of TFLite's int8 `CONV_2D` and
/// `DEPTHWISE_CONV_2D`: the whole product `input_scale * filter_scale / output_scale` in
/// double, as `QuantizeMultiplier`.
///
/// @param input_scale Input scale.
/// @param filter_scale The channel's filter scale.
/// @param output_scale Output scale.
/// @param out Receives the requantization.
/// @return 0 on success; -1 with errno EINVAL (NULL `out`, a multiplier that is not finite
///         and positive), and a message for ef_kernels_last_error().
///
/// # Safety
/// `out` must be NULL or writable.
#[no_mangle]
pub unsafe extern "C" fn ef_kernel_requant_tflite_conv(
    input_scale: f32,
    filter_scale: f32,
    output_scale: f32,
    out: *mut EfKernelRequant,
) -> c_int {
    // SAFETY: forwarded from this function's contract.
    unsafe {
        requant_out(out, || {
            k::Requant::tflite_conv(input_scale, filter_scale, output_scale)
        })
    }
}

/// Creates a plan for an int8 `ADD`, `SUB` or `MUL` reproducing the TFLite kernels `kernels`
/// names, bit for bit.
///
/// @param params Plan parameters; read during the call only.
/// @param out Receives the new plan, or NULL on failure.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown enum value, invalid
///         parameters) or ENOTSUP (scale ratios or an activation range TFLite does not take),
///         and a message for ef_kernels_last_error().
///
/// # Safety
/// `params` must point to a valid params struct and `out` to writable storage for one
/// pointer.
#[no_mangle]
pub unsafe extern "C" fn ef_binary_i8_plan_new(
    params: *const EfBinaryI8Params,
    out: *mut *mut EfBinaryI8Plan,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            make(params, out, |p| {
                Ok(EfBinaryI8Plan(k::BinaryI8Plan::new(k::BinaryI8Params {
                    op: EfBinaryOp::from_raw(p.op)
                        .ok_or(CError::Invalid("unknown binary op value"))?,
                    lhs: p.lhs.into(),
                    rhs: p.rhs.into(),
                    output: p.output.into(),
                    activation: activation(p.activation)?,
                    kernels: tflite_kernels(p.kernels)?,
                })?))
            })
        }
    })
}

/// Runs an int8 binary op elementwise: all three int8 with the same shape (any rank); a
/// broadcast input is a view with zero strides. Safe to call concurrently on one plan from
/// several threads.
///
/// @param plan A plan from `ef_binary_i8_plan_new`.
/// @param lhs Left input; must stay readable for the call.
/// @param rhs Right input; must stay readable for the call. It may overlap `lhs`.
/// @param output Output tensor; must stay writable for the call. Views whose memory spans
///        overlap either input's, interleaved views of one buffer included, are EINVAL.
/// @param pool Optional thread pool; NULL runs on the calling thread.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
///         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
///         ENOTSUP, and a message for ef_kernels_last_error().
///
/// # Safety
/// All pointers must be NULL or valid, and the views must address memory the caller owns.
#[no_mangle]
pub unsafe extern "C" fn ef_binary_i8_plan_run(
    plan: *const EfBinaryI8Plan,
    lhs: *const EfKernelView,
    rhs: *const EfKernelView,
    output: *const EfKernelView,
    pool: *const EfKernelPool,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            run_plan_n(plan, [lhs, rhs], output, pool, |p, [l, r], o, pl| {
                Ok(p.0.run(l?, r?, o, pl)?)
            })
        }
    })
}

/// Fills `table` with the plan's output for every `lhs` value with the right operand fixed at
/// `rhs`, indexed by the left value's raw byte, valid for every element of a call over `len`
/// elements; it can feed `ef_lut_i8_plan_new_table` or an `ef_attention_q_mul`.
///
/// @param plan A plan from `ef_binary_i8_plan_new`.
/// @param rhs The constant right operand.
/// @param len Elements of the call the table stands for.
/// @param table Receives 256 bytes.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer) or ENOTSUP (the call would give
///         some elements TFLite's NEON-block arithmetic and others its scalar code, and the
///         two disagree for this `rhs`), and a message for ef_kernels_last_error().
///
/// # Safety
/// `plan` must be NULL or a live plan; `table` must be NULL or point to 256 writable bytes.
#[no_mangle]
pub unsafe extern "C" fn ef_binary_i8_plan_table_for_rhs(
    plan: *const EfBinaryI8Plan,
    rhs: i8,
    len: usize,
    table: *mut u8,
) -> c_int {
    ffi(|| {
        if plan.is_null() || table.is_null() {
            return Err(CError::Invalid("plan and table must not be NULL"));
        }
        // SAFETY: `plan` is non-null and live per the contract.
        let t = unsafe { &*plan }.0.table_for_rhs(rhs, len).ok_or_else(|| {
            CError::Kernel(k::KernelError::Unsupported(
                "no single table: the NEON-block and scalar results differ for this rhs".into(),
            ))
        })?;
        // SAFETY: `table` is non-null and points to 256 writable bytes.
        unsafe { std::ptr::copy_nonoverlapping(t.as_ptr(), table, 256) };
        Ok(())
    })
}

/// Frees a plan. NULL is a no-op.
///
/// @param plan A plan from `ef_binary_i8_plan_new`, not used afterwards.
///
/// # Safety
/// `plan` must be NULL or an unfreed pointer from `ef_binary_i8_plan_new`.
#[no_mangle]
pub unsafe extern "C" fn ef_binary_i8_plan_free(plan: *mut EfBinaryI8Plan) {
    // SAFETY: forwarded from this function's contract.
    unsafe { free_plan(plan) }
}

/// Creates a plan for an int8 matrix multiplication over the last two axes, with TFLite's
/// integer requantization (`BATCH_MATMUL`, `FULLY_CONNECTED`, 1x1 `CONV_2D`).
///
/// @param params Plan parameters; read during the call only, arrays included.
/// @param out Receives the new plan, or NULL on failure.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, NULL array with a nonzero count,
///         unknown enum value, invalid parameters), ENOTSUP (ruy rounding of a multiplier of
///         0.5 or more, an empty activation range) or ENOMEM, and a message for
///         ef_kernels_last_error().
///
/// # Safety
/// `params` must point to a valid params struct (and the arrays it points to) and `out` to
/// writable storage for one pointer.
#[no_mangle]
pub unsafe extern "C" fn ef_matmul_i8_plan_new(
    params: *const EfMatmulI8Params,
    out: *mut *mut EfMatmulI8Plan,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            make(params, out, |p| {
                Ok(EfMatmulI8Plan(k::MatMulI8Plan::new(matmul_params(p)?)?))
            })
        }
    })
}

/// Runs `out = lhs * rhs` over the last two axes: `[.., M, K] x [.., K, N] -> [.., M, N]`, all
/// int8 and of the same rank (2 to 6); the other axes are batch axes (a broadcast operand has
/// zero batch strides) and transposed operands are views with swapped strides. Safe to call
/// concurrently on one plan from several threads.
///
/// @param plan A plan from `ef_matmul_i8_plan_new`.
/// @param lhs Left operand; must stay readable for the call.
/// @param rhs Right operand; must stay readable for the call. It may overlap `lhs`.
/// @param output Output tensor; must stay writable for the call. Views whose memory spans
///        overlap either operand's are EINVAL.
/// @param pool Optional thread pool; NULL runs on the calling thread.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
///         misaligned data, overlapping views, shapes that do not chain, a requantization or
///         bias count that is not one per column, a pool task that panicked) or ENOTSUP (a
///         reduction axis above 4096), and a message for ef_kernels_last_error().
///
/// # Safety
/// All pointers must be NULL or valid, and the views must address memory the caller owns.
#[no_mangle]
pub unsafe extern "C" fn ef_matmul_i8_plan_run(
    plan: *const EfMatmulI8Plan,
    lhs: *const EfKernelView,
    rhs: *const EfKernelView,
    output: *const EfKernelView,
    pool: *const EfKernelPool,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            run_plan_n(plan, [lhs, rhs], output, pool, |p, [l, r], o, pl| {
                Ok(p.0.run(l?, r?, o, pl)?)
            })
        }
    })
}

/// Whether the matmul kernels take a reduction axis of `k` elements.
///
/// @param k Reduction axis length.
/// @return 0 when they do; -1 with errno ENOTSUP past 4096, and a message for
///         ef_kernels_last_error().
#[no_mangle]
pub extern "C" fn ef_matmul_i8_check_depth(k: usize) -> c_int {
    ffi(|| Ok(k::MatMulI8Plan::check_depth(k)?))
}

/// Frees a plan. NULL is a no-op.
///
/// @param plan A plan from `ef_matmul_i8_plan_new`, not used afterwards.
///
/// # Safety
/// `plan` must be NULL or an unfreed pointer from `ef_matmul_i8_plan_new`.
#[no_mangle]
pub unsafe extern "C" fn ef_matmul_i8_plan_free(plan: *mut EfMatmulI8Plan) {
    // SAFETY: forwarded from this function's contract.
    unsafe { free_plan(plan) }
}

/// Creates a plan for an int8 softmax over the last axis followed by a product with one weight
/// column (a YOLO DFL head) in one pass. The probabilities are not rounded to 8 bits: the
/// result is within a few LSB of the float model and not TFLite's pair bit for bit.
///
/// @param params Plan parameters; read during the call only, arrays included.
/// @param out Receives the new plan, or NULL on failure.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, NULL array with a nonzero count,
///         no weights, a projection with more than one requantization or bias or another left
///         zero point than the softmax output's, invalid parameters), ENOTSUP or ENOMEM, and a
///         message for ef_kernels_last_error().
///
/// # Safety
/// `params` must point to a valid params struct (and the arrays it points to) and `out` to
/// writable storage for one pointer.
#[no_mangle]
pub unsafe extern "C" fn ef_softmax_dot_i8_plan_new(
    params: *const EfSoftmaxDotI8Params,
    out: *mut *mut EfSoftmaxDotI8Plan,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            make(params, out, |p| {
                Ok(EfSoftmaxDotI8Plan(k::SoftmaxDotI8Plan::new(
                    k::SoftmaxDotI8Params {
                        softmax: k::SoftmaxI8Params {
                            input: p.softmax.input.into(),
                            output: p.softmax.output.into(),
                            beta: p.softmax.beta,
                        },
                        weights: copy_array(p.weights, p.n_weights, "weights is NULL", |w| w)?,
                        projection: matmul_params(&p.projection)?,
                    },
                )?))
            })
        }
    })
}

/// Runs the softmax-then-dot of `input` (`[.., n]`, `n` the number of weights, unit stride
/// along the last axis) into `output` (`[.., 1]`), both int8. Safe to call concurrently on one
/// plan from several threads.
///
/// @param plan A plan from `ef_softmax_dot_i8_plan_new`.
/// @param input Input logits; must stay readable for the call.
/// @param output Output tensor; must stay writable for the call. Views whose memory spans
///        overlap the input's are EINVAL.
/// @param pool Optional thread pool; NULL runs on the calling thread.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
///         misaligned data, overlapping views, shape mismatch, a pool task that panicked) or
///         ENOTSUP (an input row axis that is not unit-stride), and a message for
///         ef_kernels_last_error().
///
/// # Safety
/// All pointers must be NULL or valid, and the views must address memory the caller owns.
#[no_mangle]
pub unsafe extern "C" fn ef_softmax_dot_i8_plan_run(
    plan: *const EfSoftmaxDotI8Plan,
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
/// @param plan A plan from `ef_softmax_dot_i8_plan_new`, not used afterwards.
///
/// # Safety
/// `plan` must be NULL or an unfreed pointer from `ef_softmax_dot_i8_plan_new`.
#[no_mangle]
pub unsafe extern "C" fn ef_softmax_dot_i8_plan_free(plan: *mut EfSoftmaxDotI8Plan) {
    // SAFETY: forwarded from this function's contract.
    unsafe { free_plan(plan) }
}

/// Creates a plan for fused int8 attention, `softmax(q' * k^T) * v` per head with `q'` Q
/// through the optional `q_mul`, reproducing TFLite's `[MUL] -> BATCH_MATMUL -> SOFTMAX ->
/// BATCH_MATMUL` chain.
///
/// @param params Plan parameters; read during the call only, `q_mul` included.
/// @param out Receives the new plan, or NULL on failure.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, unknown enum value, invalid
///         scales or zero points) or ENOTSUP (probabilities not quantized (1/256, -128), a
///         requantization the rounding cannot apply), and a message for
///         ef_kernels_last_error().
///
/// # Safety
/// `params` must point to a valid params struct (`q_mul` NULL or valid) and `out` to writable
/// storage for one pointer.
#[no_mangle]
pub unsafe extern "C" fn ef_attention_i8_plan_new(
    params: *const EfAttentionI8Params,
    out: *mut *mut EfAttentionI8Plan,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            make(params, out, |p| {
                let q_mul = if p.q_mul.is_null() {
                    None
                } else {
                    // SAFETY: non-null and valid per the contract.
                    let m = &*p.q_mul;
                    Some(k::AttentionQMul {
                        table: m.table,
                        output: m.output.into(),
                    })
                };
                Ok(EfAttentionI8Plan(k::AttentionI8Plan::new(
                    k::AttentionI8Params {
                        q: p.q.into(),
                        q_mul,
                        k: p.k.into(),
                        v: p.v.into(),
                        scores: p.scores.into(),
                        beta: p.beta,
                        probs: p.probs.into(),
                        output: p.output.into(),
                        kernels: tflite_kernels(p.kernels)?,
                    },
                )?))
            })
        }
    })
}

/// Whether the attention kernels take `tokens` key/value tokens per head, `dk`-long queries
/// and keys and `dv`-long values.
///
/// @param tokens Key/value tokens per head.
/// @param dk Query and key length.
/// @param dv Value length.
/// @return 0 when they do; -1 with errno ENOTSUP past 2048 tokens or a head dim of 256, and a
///         message for ef_kernels_last_error().
#[no_mangle]
pub extern "C" fn ef_attention_i8_check_shape(tokens: usize, dk: usize, dv: usize) -> c_int {
    ffi(|| Ok(k::AttentionI8Plan::check_shape(tokens, dk, dv)?))
}

/// Creates the packing scratch `plan` needs for one call shape: `heads` heads of `queries`
/// queries attending over `tokens` key/value tokens, `dk`-long queries and keys and `dv`-long
/// values. A scratch serves any number of calls of that shape on that plan (or on a plan with
/// the same kernel), one call at a time.
///
/// @param plan A plan from `ef_attention_i8_plan_new`; read during the call only.
/// @param heads Heads.
/// @param queries Queries per head.
/// @param tokens Key/value tokens per head.
/// @param dk Query and key length.
/// @param dv Value length.
/// @param out Receives the new scratch, or NULL on failure.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, sizes that overflow) or ENOTSUP
///         (a shape `ef_attention_i8_check_shape` refuses, an allocation that failed), and a
///         message for ef_kernels_last_error().
///
/// # Safety
/// `plan` must be NULL or a live plan; `out` must be NULL or writable.
#[no_mangle]
pub unsafe extern "C" fn ef_attention_scratch_new(
    plan: *const EfAttentionI8Plan,
    heads: usize,
    queries: usize,
    tokens: usize,
    dk: usize,
    dv: usize,
    out: *mut *mut EfAttentionScratch,
) -> c_int {
    ffi(|| {
        // SAFETY: forwarded from this function's contract.
        unsafe {
            make(plan, out, |p| {
                Ok(EfAttentionScratch(
                    p.0.scratch(heads, queries, tokens, dk, dv)?,
                ))
            })
        }
    })
}

/// Frees a scratch. NULL is a no-op.
///
/// @param scratch A scratch from `ef_attention_scratch_new`, not used afterwards.
///
/// # Safety
/// `scratch` must be NULL or an unfreed pointer from `ef_attention_scratch_new`.
#[no_mangle]
pub unsafe extern "C" fn ef_attention_scratch_free(scratch: *mut EfAttentionScratch) {
    // SAFETY: forwarded from this function's contract.
    unsafe { free_plan(scratch) }
}

/// Runs attention of `q` `[heads, queries, dk]` over `k` `[heads, tokens, dk]` and `v`
/// `[heads, tokens, dv]` into `output` `[heads, queries, dv]`, all int8 views with any
/// strides (head slices of one token-major buffer included). Safe to call concurrently on
/// one plan from several threads, each call with its own scratch.
///
/// @param plan A plan from `ef_attention_i8_plan_new`.
/// @param q Queries; must stay readable for the call.
/// @param k Keys; must stay readable for the call. Inputs may overlap each other.
/// @param v Values; must stay readable for the call.
/// @param output Output tensor; must stay writable for the call. Views whose memory spans
///        overlap any input's are EINVAL.
/// @param scratch A scratch from `ef_attention_scratch_new` for this shape, used by no other
///        call during this one. A scratch built for another shape, or by a plan with another
///        kernel, is EINVAL and nothing is written.
/// @param pool Optional thread pool; NULL runs on the calling thread.
/// @return 0 on success; -1 with errno EINVAL (NULL pointer, bad dtype, rank or view,
///         misaligned data, overlapping views, shape mismatch, a mismatched scratch, a pool
///         task that panicked) or ENOTSUP, and a message for ef_kernels_last_error().
///
/// # Safety
/// All pointers must be NULL or valid, the views must address memory the caller owns, and no
/// other call may use `scratch` concurrently.
#[no_mangle]
pub unsafe extern "C" fn ef_attention_i8_plan_run(
    plan: *const EfAttentionI8Plan,
    q: *const EfKernelView,
    k: *const EfKernelView,
    v: *const EfKernelView,
    output: *const EfKernelView,
    scratch: *mut EfAttentionScratch,
    pool: *const EfKernelPool,
) -> c_int {
    ffi(|| {
        if scratch.is_null() {
            return Err(CError::Invalid("scratch must not be NULL"));
        }
        // SAFETY: forwarded from this function's contract.
        unsafe {
            run_plan_n(plan, [q, k, v], output, pool, |p, [q, k, v], o, pl| {
                // SAFETY: non-null, live, and used by this call alone per the contract.
                let s = &mut (*scratch).0;
                Ok(p.0.run(q?, k?, v?, o, s, pl)?)
            })
        }
    })
}

/// Frees a plan. NULL is a no-op.
///
/// @param plan A plan from `ef_attention_i8_plan_new`, not used afterwards.
///
/// # Safety
/// `plan` must be NULL or an unfreed pointer from `ef_attention_i8_plan_new`.
#[no_mangle]
pub unsafe extern "C" fn ef_attention_i8_plan_free(plan: *mut EfAttentionI8Plan) {
    // SAFETY: forwarded from this function's contract.
    unsafe { free_plan(plan) }
}
