// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `Tensor<T>` as a typed lens over [`TensorDyn`].
//!
//! Rust code that needs the element type known at compile time opens a lens
//! onto a `TensorDyn` with [`TensorDyn::as_typed`]/[`TensorDyn::as_typed_mut`].
//! Both backends implement them: `static_lens` for the enum-of-`Tensor<T>`
//! backend, `dynamic_lens` for the opaque-handle one.

use crate::sealed::Sealed;
use crate::DType;
use half::f16;
use num_traits::Num;
use std::fmt;

/// An element type a `Tensor<T>` lens may be opened for. Sealed: the eleven
/// variants of [`TensorDyn`] are the only implementors.
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

#[cfg(not(edgefirst_tensor_dynamic))]
mod static_lens {
    //! The lens accessors, for the backend where `TensorDyn` is an enum over
    //! eleven `Tensor<T>` values. They match the active variant and use
    //! `std::any::Any::downcast_ref` to check that `T` is its element type.
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

#[cfg(edgefirst_tensor_dynamic)]
mod dynamic_lens {
    //! The lens accessors, for the backend where `TensorDyn` is one handle.
    //! After checking the handle's dtype against `T`, they reinterpret
    //! `&TensorDyn` as `&Tensor<T>`, which is `#[repr(transparent)]` over
    //! `TensorDyn`.
    use super::Element;
    use crate::{Tensor, TensorDyn};

    impl TensorDyn {
        /// Borrow this tensor as a statically-typed `Tensor<T>`, or `None`
        /// when the element type does not match.
        pub fn as_typed<T: Element>(&self) -> Option<&Tensor<T>> {
            if self.dtype() != T::DTYPE {
                return None;
            }
            // SAFETY: `Tensor<T>` is `#[repr(transparent)]` over `TensorDyn`,
            // so a `&TensorDyn` and a `&Tensor<T>` share layout; the dtype
            // check above proves `T` matches what this handle holds.
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

#[cfg(all(test, not(edgefirst_tensor_dynamic)))]
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
            let lens = d.as_typed_mut::<u8>().expect("u8 lens");
            lens.map_write().expect("map").as_mut_slice()[0] = 42;
        }
        let lens = d.as_typed::<u8>().expect("lens");
        assert_eq!(lens.map_read().expect("map").as_slice()[0], 42);
    }
}
