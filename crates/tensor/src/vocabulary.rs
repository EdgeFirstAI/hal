// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! [`ef_vocabulary!`]: one declaration per shared vocabulary.
//!
//! `TensorMemory`, `PixelFormat` and the other vocabularies carry the same
//! numbering in the Rust API, the wire protocol, the Python bindings and the C
//! header. The macro takes explicit discriminants in a single declaration and
//! generates `code()` and the wire module's `const`s from it; a `cfg`-gated
//! variant keeps its number reserved when it is compiled out.

/// Declare a vocabulary once, emit every surface from it.
///
/// Discriminants are mandatory and explicit. A `cfg`-gated variant keeps its
/// number reserved when absent.
///
/// Emits, for a vocabulary `V` with wire module `m`:
/// - `pub enum V` with explicit `#[repr(u32)]` discriminants taken verbatim
///   from the declaration.
/// - `V::code(self) -> u32` (`const fn`) / `V::from_code(u32) -> Option<V>`,
///   the wire numbering. `from_code` returns `None` for an unassigned code.
/// - `V::as_str(self) -> &'static str` / `V::from_str_code(&str) -> Option<V>`,
///   the wire string form.
/// - `V::all() -> &'static [V]`, every variant in declaration order.
/// - `pub mod m { pub const CONST_NAME: u32 = ...; }`, one `const` per
///   variant. The `mod` clause accepts its own `$(#[meta])*` prefix (e.g.
///   `#[doc(hidden)]`), as the `enum` clause does.
///
/// The wire `const`'s identifier is a separate, explicit field per variant.
#[macro_export]
macro_rules! ef_vocabulary {
    (
        $(#[$meta:meta])*
        $vis:vis enum $name:ident {
            $( $(#[$vmeta:meta])* $variant:ident = $code:expr, $text:literal, $const:ident ),+ $(,)?
        }
        $(#[$mmeta:meta])*
        $modvis:vis mod $modname:ident;
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        #[repr(u32)]
        $vis enum $name {
            $( $(#[$vmeta])* $variant = $code ),+
        }

        impl $name {
            /// The shared code for this variant. Identical on every surface.
            pub const fn code(self) -> u32 { self as u32 }

            /// The variant for `code`, or `None` for an unassigned code.
            pub fn from_code(code: u32) -> Option<Self> {
                match code { $( $code => Some(Self::$variant), )+ _ => None }
            }

            /// The wire string form.
            pub fn as_str(self) -> &'static str {
                match self { $( Self::$variant => $text, )+ }
            }

            /// The variant for a wire string, or `None` for an unknown one.
            pub fn from_str_code(s: &str) -> Option<Self> {
                match s { $( $text => Some(Self::$variant), )+ _ => None }
            }

            /// Every variant, in declaration order.
            pub fn all() -> &'static [Self] { &[ $( Self::$variant ),+ ] }
        }

        /// Wire codes for [`
        #[doc = stringify!($name)]
        /// `], one `const` per variant, identical to `.code()` for the same
        /// variant.
        $(#[$mmeta])*
        $modvis mod $modname {
            $( pub const $const: u32 = $code; )+
        }
    };
}

/// Demo vocabulary exercising [`ef_vocabulary!`]. `Beta`'s code (7) is
/// non-contiguous with `Alpha`'s (0).
#[doc(hidden)]
pub mod vocabulary_demo {
    crate::ef_vocabulary! {
        pub enum Demo {
            Alpha = 0, "alpha", ALPHA,
            Beta = 7, "beta", BETA,
        }
        pub mod demo_code;
    }
}
