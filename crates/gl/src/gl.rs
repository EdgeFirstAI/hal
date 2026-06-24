// This module is machine-generated FFI (gl_generator) plus a re-export; don't
// lint generated bindings.
#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(improper_ctypes)]
#![allow(dead_code)]
#![allow(clippy::all)]

include!(concat!(env!("OUT_DIR"), "/gl_bindings.rs"));

// Re-export the generated GL type aliases (GLenum, GLuint, …).
pub use types::*;
