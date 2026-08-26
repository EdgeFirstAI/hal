// This module is machine-generated FFI (gl_generator) plus a re-export; don't
// lint generated bindings.
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]
#![allow(dead_code)]
#![allow(clippy::all)]
// gl_generator's own codegen predates `unsafe_op_in_unsafe_fn` (a
// consumer's own `[lints.rust] unsafe_op_in_unsafe_fn = "deny"` -- see
// each -capi leaf's Cargo.toml, task 10/hygiene-9 -- propagates onto this
// path-dependency crate too, since it's a workspace member, not an
// external one). Found the hard way: task 12's first genuinely-fresh
// aarch64 cross-build (an empty target dir hits this generated file for
// the first time; every prior local build silently reused an already-
// compiled, pre-deny artifact from the shared target/ dir and never
// re-triggered it) produced 700+ near-identical errors here, none of them
// this crate's own code to fix.
#![allow(unsafe_op_in_unsafe_fn)]

include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));

// Re-export the generated GL type aliases (GLenum, GLuint, …).
pub use types::*;
