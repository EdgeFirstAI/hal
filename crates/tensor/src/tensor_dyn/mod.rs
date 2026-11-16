// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `TensorDyn`: the main tensor type, and the one every FFI surface calls
//! `Tensor`.
//!
//! Two backends present an identical public API. `static` compiles the
//! implementation in. `dynamic` calls into `libedgefirst_tensor.so`; the
//! `-capi` leaves link against it.
//!
//! `derived` holds the methods expressible over the primitive API and is
//! compiled into both backends.

#[cfg(not(edgefirst_tensor_dynamic))]
mod static_backend;
#[cfg(not(edgefirst_tensor_dynamic))]
pub use static_backend::{BorrowedTensor, Raw, TensorDyn};

#[cfg(edgefirst_tensor_dynamic)]
mod dynamic_backend;
#[cfg(edgefirst_tensor_dynamic)]
pub use dynamic_backend::{BorrowedTensor, Raw, TensorDyn};
// The C library's last error message, used by `dynamic_tensor.rs`.
#[cfg(edgefirst_tensor_dynamic)]
pub(crate) use dynamic_backend::ffi_last_error;
// The typed error rebuilt from the C library's last error, used by
// `crate::d3d11`.
#[cfg(all(edgefirst_tensor_dynamic, target_os = "windows"))]
pub(crate) use dynamic_backend::ffi_error;

// `dynamic`'s `Tensor<T>`: the typed lens with no storage of its own.
#[cfg(edgefirst_tensor_dynamic)]
pub(crate) mod dynamic_tensor;
#[cfg(edgefirst_tensor_dynamic)]
pub use dynamic_tensor::Tensor;

mod derived;
