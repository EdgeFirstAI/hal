// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! One declaration per shared vocabulary, instead of four.
//!
//! `TensorMemory`, `PixelFormat`, and friends each need to agree across the
//! Rust API, the wire protocol, the Python bindings, and the C header. Before
//! this macro, each surface carried its own hand-written numbering of the
//! same variants, and they only agreed by coincidence. `PyTensorMemory` is
//! the live example: its Python enum has no explicit discriminants and a
//! `#[cfg(unix)]` variant sits in the middle of the list, so every variant
//! declared after it silently shifts by one on a non-unix build. Bridging
//! Python and C by integer then hands back `Pbo` where `Mem` was requested,
//! with no error at any layer.
//!
//! [`ef_vocabulary!`] closes that gap structurally: discriminants are
//! mandatory and explicit in the single declaration, so a `cfg`-gated
//! variant keeps its number reserved when it is compiled out, rather than
//! renumbering everything after it. `code()` and the wire module's `const`s
//! are generated from that one declaration, so the Rust enum, the wire
//! encoding, and every other consumer read the same numbers by construction
//! -- there is no second table to fall out of sync.

/// Declare a vocabulary once, emit every surface from it.
///
/// Discriminants are MANDATORY and explicit. A `cfg`-gated variant therefore
/// keeps its number reserved when absent, instead of shifting every variant
/// after it -- which is how `PyTensorMemory` ended up platform-dependent.
///
/// Emits, for a vocabulary `V` with wire module `m`:
/// - `pub enum V` with explicit `#[repr(u32)]` discriminants taken verbatim
///   from the declaration.
/// - `V::code(self) -> u32` (`const fn`) / `V::from_code(u32) -> Option<V>`,
///   the wire numbering. `from_code` returns `None` for an unassigned code
///   rather than guessing at the nearest variant. `code` being `const`
///   lets a consumer that cannot call it directly (a `#[pyclass]`
///   discriminant, which PyO3 requires to be a literal) instead pin its
///   own literal to it with `const _: () = assert!(...)`.
/// - `V::as_str(self) -> &'static str` / `V::from_str_code(&str) -> Option<V>`,
///   the wire string form (not `Debug`, which is free to change).
/// - `V::all() -> &'static [V]`, every variant in declaration order, so
///   cross-surface consistency tests can be exhaustive instead of a sample.
/// - `pub mod m { pub const CONST_NAME: u32 = ...; }`, one `const` per
///   variant, for `const`-only consumers (cbindgen headers, match-free FFI
///   code). This is the shape `protocol::dtype` / `protocol::format` /
///   `protocol::kind` already use by hand. The `mod` clause accepts its own
///   `$(#[meta])*` prefix (e.g. `#[doc(hidden)]`), the same as the `enum`
///   clause does -- `m` has to be `pub` for a *different* module (typically
///   `protocol`) to `pub use` it back out under its real, documented name,
///   so `#[doc(hidden)]` is how a caller marks it as emission plumbing
///   rather than a second, undocumented public API surface for the same
///   vocabulary.
///
/// The wire `const`'s identifier is a separate, explicit field per variant
/// rather than derived from the enum variant name. `DType`'s variants (`U8`,
/// `I8`, ...) happen to already be valid `SCREAMING_SNAKE_CASE`, but
/// `PixelFormat`'s do not -- `PlanarRgb` is the enum variant,
/// `protocol::format::PLANAR_RGB` is the existing wire constant, and no
/// mechanical transform of one produces the other without either an extra
/// dependency for identifier-pasting-with-case-conversion or a hidden
/// word-boundary guessing rule. Spelling both out keeps the same "explicit,
/// not inferred" rule this macro already applies to discriminants.
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
            ///
            /// `const`, deliberately: a downstream binding that cannot
            /// write `Variant = OtherEnum::Variant.code()` in a `#[pyclass]`
            /// discriminant position (PyO3 requires a literal there) can
            /// still pin that literal to this value with a
            /// `const _: () = assert!(...)` -- a build-time check instead
            /// of a hand-synced comment. See `PyTensorMemory`/`PyPixelFormat`
            /// in `crates/python-common/src/tensor.rs` for the consumer.
            pub const fn code(self) -> u32 { self as u32 }

            /// Reject an unknown code rather than guessing at one.
            pub fn from_code(code: u32) -> Option<Self> {
                match code { $( $code => Some(Self::$variant), )+ _ => None }
            }

            /// The wire string form -- stable, unlike `Debug`.
            pub fn as_str(self) -> &'static str {
                match self { $( Self::$variant => $text, )+ }
            }

            /// Reject an unknown string rather than guessing at one.
            pub fn from_str_code(s: &str) -> Option<Self> {
                match s { $( $text => Some(Self::$variant), )+ _ => None }
            }

            /// Every variant, for exhaustive cross-surface tests.
            pub fn all() -> &'static [Self] { &[ $( Self::$variant ),+ ] }
        }

        /// Wire codes for [`
        #[doc = stringify!($name)]
        /// `], one `const` per variant and identical to `.code()` for the
        /// same variant -- both are generated from the same declared
        /// literal here, so a `const`-only consumer (cbindgen, FFI) and
        /// `code()` cannot independently drift apart.
        $(#[$mmeta])*
        $modvis mod $modname {
            $( pub const $const: u32 = $code; )+
        }
    };
}

/// Demo vocabulary exercising [`ef_vocabulary!`] end to end.
///
/// `Beta`'s code (7) is deliberately non-contiguous with `Alpha`'s (0): it
/// proves the discriminant comes from the declaration, not from position in
/// the list. This module carries no real vocabulary -- [`crate::DType`] is
/// converted onto the macro directly (as a proof the macro reproduces
/// today's values unchanged); `PixelFormat` and `TensorMemory` follow in
/// later tasks.
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
