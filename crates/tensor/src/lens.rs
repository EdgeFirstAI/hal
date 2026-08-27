// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `Tensor<T>` as a typed lens over [`TensorDyn`].
//!
//! `TensorDyn` is the main type — the one every FFI surface (C, Python, and
//! any future binding) calls `Tensor`. Rust code that needs the element type
//! known at compile time opens a lens onto it with
//! [`TensorDyn::as_typed`]/[`TensorDyn::as_typed_mut`] instead of matching a
//! constructor; `Tensor<T>` itself owns no storage of its own to lose — it
//! borrows the handle's.
//!
//! Rust-only. The lens exists because Rust can express "this handle's
//! element type is `f32`" as a static fact; C and Python have no such
//! concept and always operate on the erased `TensorDyn` directly. Both
//! backends implement [`as_typed`](TensorDyn::as_typed)/
//! [`as_typed_mut`](TensorDyn::as_typed_mut) -- `static_lens` below for the
//! enum-of-`Tensor<T>` backend, `dynamic_lens` for the opaque-handle one --
//! with different validity arguments for the same cast; see each module's
//! docs.

use crate::sealed::Sealed;
use crate::DType;
use half::f16;
use num_traits::Num;
use std::fmt;

/// An element type a `Tensor<T>` lens may be opened for.
///
/// Sealed: the eleven variants of [`TensorDyn`] are the only element types a
/// tensor can carry, and a downstream impl would let
/// [`as_typed`](TensorDyn::as_typed) name a dtype no tensor can hold.
pub trait Element: Sealed + Num + Clone + fmt::Debug + Send + Sync + 'static {
    /// The runtime dtype this element type corresponds to.
    const DTYPE: DType;
}

macro_rules! element {
    ($($t:ty => $d:ident),* $(,)?) => {$(
        impl Element for $t { const DTYPE: DType = DType::$d; }
    )*};
}
element! {
    u8 => U8, i8 => I8, u16 => U16, i16 => I16, u32 => U32, i32 => I32,
    u64 => U64, i64 => I64, f16 => F16, f32 => F32, f64 => F64,
}

#[cfg(feature = "static")]
mod static_lens {
    //! The lens accessors, for the backend where `TensorDyn` really is an
    //! enum over eleven `Tensor<T>` values.
    //!
    //! Implementation note: this deliberately does **not** reinterpret
    //! `TensorDyn` as `Tensor<T>` via a raw pointer cast. `TensorDyn` is a
    //! tagged-union `enum`, and Rust gives no layout guarantee that would
    //! make `&TensorDyn as *const Tensor<T>` sound for any variant --
    //! that's a promise only `#[repr(transparent)]` over a *single* field
    //! makes, and `TensorDyn` has eleven. Instead this matches the active
    //! variant (as `as_u8`/`as_f32`/etc. in `static_backend.rs` already do)
    //! and uses `std::any::Any::downcast_ref`, which performs the exact
    //! `TypeId` check that proves `T` is the concrete element type -- safe,
    //! no `unsafe` block, and it costs nothing `as_u8()` wasn't already
    //! paying.
    use super::Element;
    use crate::{Tensor, TensorDyn};
    use std::any::Any;

    impl TensorDyn {
        /// Borrow this tensor as a statically-typed `Tensor<T>`, or `None`
        /// when the element type does not match.
        pub fn as_typed<T: Element>(&self) -> Option<&Tensor<T>> {
            let any: &dyn Any = match self {
                TensorDyn::U8(t) => t,
                TensorDyn::I8(t) => t,
                TensorDyn::U16(t) => t,
                TensorDyn::I16(t) => t,
                TensorDyn::U32(t) => t,
                TensorDyn::I32(t) => t,
                TensorDyn::U64(t) => t,
                TensorDyn::I64(t) => t,
                TensorDyn::F16(t) => t,
                TensorDyn::F32(t) => t,
                TensorDyn::F64(t) => t,
            };
            any.downcast_ref::<Tensor<T>>()
        }

        /// Mutably borrow this tensor as a statically-typed `Tensor<T>`, or
        /// `None` when the element type does not match.
        pub fn as_typed_mut<T: Element>(&mut self) -> Option<&mut Tensor<T>> {
            let any: &mut dyn Any = match self {
                TensorDyn::U8(t) => t,
                TensorDyn::I8(t) => t,
                TensorDyn::U16(t) => t,
                TensorDyn::I16(t) => t,
                TensorDyn::U32(t) => t,
                TensorDyn::I32(t) => t,
                TensorDyn::U64(t) => t,
                TensorDyn::I64(t) => t,
                TensorDyn::F16(t) => t,
                TensorDyn::F32(t) => t,
                TensorDyn::F64(t) => t,
            };
            any.downcast_mut::<Tensor<T>>()
        }
    }
}

#[cfg(feature = "dynamic")]
mod dynamic_lens {
    //! The lens accessors, for the backend where `TensorDyn` really is one
    //! handle.
    //!
    //! Implementation note: unlike `static_lens`, this one *does*
    //! reinterpret `&TensorDyn`/`&mut TensorDyn` as `&Tensor<T>`/`&mut
    //! Tensor<T>` via a pointer cast, and that is sound here even though it
    //! is UB for the `static` enum. `dynamic`'s `Tensor<T>`
    //! (`tensor_dyn/dynamic_tensor.rs`) is `#[repr(transparent)]` with
    //! `TensorDyn` as its only nonzero-sized field, so it has identical
    //! layout to a bare `TensorDyn` value regardless of what `TensorDyn`'s
    //! own fields are (a handle plus two cached facts, per
    //! `dynamic_backend.rs`'s module docs -- not a bare pointer, but that
    //! does not matter here: `#[repr(transparent)]` guarantees identical
    //! layout to whatever the single field's layout is, not specifically
    //! to a pointer's). What makes the cast safe rather than merely
    //! well-typed is the dtype check below: it proves `T` is the caller's
    //! actual element type before the reinterpreted reference is handed
    //! out, exactly as `static_lens`'s `TypeId` check does for its
    //! `downcast_ref`.
    use super::Element;
    use crate::{Tensor, TensorDyn};

    impl TensorDyn {
        /// Borrow this tensor as a statically-typed `Tensor<T>`, or `None`
        /// when the element type does not match.
        pub fn as_typed<T: Element>(&self) -> Option<&Tensor<T>> {
            if self.dtype() != T::DTYPE {
                return None;
            }
            // SAFETY: `Tensor<T>` is `#[repr(transparent)]` over `TensorDyn`
            // (see module docs), so a `&TensorDyn` and a `&Tensor<T>` share
            // layout; the dtype check above proves `T` matches what this
            // handle actually holds.
            Some(unsafe { &*(self as *const TensorDyn as *const Tensor<T>) })
        }

        /// Mutably borrow this tensor as a statically-typed `Tensor<T>`, or
        /// `None` when the element type does not match.
        pub fn as_typed_mut<T: Element>(&mut self) -> Option<&mut Tensor<T>> {
            if self.dtype() != T::DTYPE {
                return None;
            }
            // SAFETY: see `as_typed` above; `&mut self` gives exclusive
            // access, preserved through the cast.
            Some(unsafe { &mut *(self as *mut TensorDyn as *mut Tensor<T>) })
        }
    }
}

#[cfg(all(test, feature = "static"))]
mod tests {
    use crate::{DType, Tensor, TensorDyn};

    #[test]
    fn as_typed_matches_only_the_right_dtype() {
        let t: Tensor<f32> = Tensor::new(&[4], None, None).expect("alloc");
        let d: TensorDyn = t.into();
        assert_eq!(d.dtype(), DType::F32);
        assert!(d.as_typed::<f32>().is_some(), "f32 lens must open");
        assert!(
            d.as_typed::<u8>().is_none(),
            "u8 lens must refuse an f32 tensor"
        );
    }

    #[test]
    fn as_typed_mut_is_exclusive_and_writes_through() {
        use crate::{TensorMapTrait, TensorTrait};
        let t: Tensor<u8> = Tensor::new(&[4], None, None).expect("alloc");
        let mut d: TensorDyn = t.into();
        {
            // `map_write` takes &self and returns HostView<'_, u8>; the
            // exclusivity here comes from `as_typed_mut`'s &mut self, not
            // from the map. There is no HostViewMut in this codebase.
            let lens = d.as_typed_mut::<u8>().expect("u8 lens");
            lens.map_write().expect("map").as_mut_slice()[0] = 42;
        }
        let lens = d.as_typed::<u8>().expect("lens");
        assert_eq!(lens.map_read().expect("map").as_slice()[0], 42);
    }
}
