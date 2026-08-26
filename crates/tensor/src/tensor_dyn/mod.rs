// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `TensorDyn` — the main tensor type, and the one every FFI surface calls
//! `Tensor`.
//!
//! Two backends present an identical public API. `static` compiles the
//! implementation in and is what every Rust consumer uses. `dynamic` calls
//! into `libedgefirst_tensor.so` instead of embedding a private copy, and
//! exists so the four `-capi` leaves can link against it. `dynamic`'s
//! `TensorDyn` is not `#[repr(transparent)]` over the bare C handle -- it
//! also caches a shape and a [`crate::BufferIdentity`] the ABI has no
//! primitive to re-derive on every call cheaply, see
//! `dynamic_backend.rs`'s module docs -- but `dynamic_tensor::Tensor<T>` IS
//! `#[repr(transparent)]` over `TensorDyn` itself (whatever its fields),
//! which is what makes the lens cast in `lens.rs` sound for this backend.
//!
//! `derived` holds every method expressible over the primitive API. It is
//! compiled into BOTH backends, written once, against the public surface
//! only — the technique that keeps the ABI near a dozen entry points instead
//! of eighty.

#[cfg(feature = "static")]
mod static_backend;
#[cfg(feature = "static")]
pub use static_backend::{Raw, TensorDyn};

#[cfg(feature = "dynamic")]
mod dynamic_backend;
#[cfg(feature = "dynamic")]
pub use dynamic_backend::{Raw, TensorDyn};
// `dynamic_tensor.rs` (a sibling module, not a child of `dynamic_backend`)
// needs this to enrich its own `ef_tensor_from_planes` error with the C
// side's advisory detail, the same way `dynamic_backend.rs`'s own
// constructors already do.
#[cfg(feature = "dynamic")]
pub(crate) use dynamic_backend::ffi_last_error;

// `dynamic`'s `Tensor<T>` -- the typed lens with no storage of its own.
// `static`'s `Tensor<T>` lives directly in `lib.rs` (it predates this
// module and owns real per-backend storage, unlike this one).
#[cfg(feature = "dynamic")]
pub(crate) mod dynamic_tensor;
#[cfg(feature = "dynamic")]
pub use dynamic_tensor::Tensor;

mod derived;
