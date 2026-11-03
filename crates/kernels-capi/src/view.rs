// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Conversion of C views into the kernels crate's type-erased tensors.

use edgefirst_kernels::{ByteSpan, DType, TensorMut, TensorRef, View, ViewMut, MAX_RANK};
use half::f16;

use crate::error::CError;
use crate::types::{EfKernelDtype, EfKernelView};

/// The dtype of `v`, or an error for an unknown value.
pub fn dtype_of(raw: u32) -> Result<DType, CError> {
    EfKernelDtype::from_raw(raw).ok_or(CError::Invalid("unknown dtype value"))
}

/// Borrows the shape and optional strides of `v`.
///
/// # Safety
/// `v.shape` must point to `v.ndim` readable `usize`s, and `v.strides`, when non-null, to
/// `v.ndim` readable `isize`s.
unsafe fn dims<'a>(v: &EfKernelView) -> Result<(&'a [usize], Option<&'a [isize]>), CError> {
    if v.ndim > MAX_RANK {
        return Err(CError::Invalid("view rank exceeds 6"));
    }
    if v.ndim == 0 {
        return Ok((&[], None));
    }
    if v.shape.is_null() {
        return Err(CError::Invalid("view shape is NULL"));
    }
    // SAFETY: shape is non-null and the caller guarantees `ndim` elements.
    let shape = unsafe { std::slice::from_raw_parts(v.shape, v.ndim) };
    let strides = if v.strides.is_null() {
        None
    } else {
        // SAFETY: strides is non-null and the caller guarantees `ndim` elements.
        Some(unsafe { std::slice::from_raw_parts(v.strides, v.ndim) })
    };
    Ok((shape, strides))
}

macro_rules! convert {
    ($v:expr, $p:expr, $view:ident, $tensor:ident) => {{
        let v: &EfKernelView = $v;
        let dtype = dtype_of(v.dtype)?;
        // SAFETY: the caller of the entry point guarantees the shape and strides arrays.
        let (shape, strides) = unsafe { dims(v)? };
        // SAFETY: the caller guarantees `data` addresses every element the view reaches.
        unsafe {
            Ok(match dtype {
                DType::I8 => $tensor::I8($view::from_raw_parts($p.cast(), shape, strides)?),
                DType::U8 => $tensor::U8($view::from_raw_parts($p.cast(), shape, strides)?),
                DType::I16 => $tensor::I16($view::from_raw_parts($p.cast(), shape, strides)?),
                DType::U16 => $tensor::U16($view::from_raw_parts($p.cast(), shape, strides)?),
                DType::F16 => $tensor::F16($view::from_raw_parts($p.cast(), shape, strides)?),
                DType::F32 => $tensor::F32($view::from_raw_parts($p.cast(), shape, strides)?),
            })
        }
    }};
}

/// The bytes `v` reaches, from its data pointer and validated layout; reads no element.
///
/// # Safety
/// `v`'s shape and strides pointers must satisfy the contract documented on `ef_kernel_view`.
pub unsafe fn span(v: &EfKernelView) -> Result<ByteSpan, CError> {
    let dtype = dtype_of(v.dtype)?;
    // SAFETY: the caller of the entry point guarantees the shape and strides arrays.
    let (shape, strides) = unsafe { dims(v)? };
    let p = v.data.cast_const();
    Ok(match dtype {
        DType::I8 | DType::U8 => ByteSpan::new(p.cast::<u8>(), shape, strides)?,
        DType::I16 | DType::U16 | DType::F16 => ByteSpan::new(p.cast::<u16>(), shape, strides)?,
        DType::F32 => ByteSpan::new(p.cast::<f32>(), shape, strides)?,
    })
}

/// An immutable tensor over `v`.
///
/// # Safety
/// `v`'s pointers must satisfy the contract documented on `ef_kernel_view`.
pub unsafe fn tensor_ref<'a>(v: &EfKernelView) -> Result<TensorRef<'a>, CError> {
    convert!(v, v.data.cast_const(), View, TensorRef)
}

/// A mutable tensor over `v`.
///
/// # Safety
/// `v`'s pointers must satisfy the contract documented on `ef_kernel_view`, and nothing else
/// may access the elements during the call.
pub unsafe fn tensor_mut<'a>(v: &EfKernelView) -> Result<TensorMut<'a>, CError> {
    convert!(v, v.data, ViewMut, TensorMut)
}

// f16 is `repr(transparent)` over `u16`, which is what C callers hold.
const _: () = assert!(std::mem::size_of::<f16>() == 2);
