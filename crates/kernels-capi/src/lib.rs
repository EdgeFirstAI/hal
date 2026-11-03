// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The C API for `edgefirst-kernels`: quantized softmax, sigmoid, table lookup and DFL decode.
//!
//! Plans are built once with `ef_<op>_plan_new` and run any number of times, from any number
//! of threads, with `ef_<op>_plan_run`. Tensors are caller-owned `ef_kernel_view`s, so this
//! library links no sibling library.

pub mod error;
pub mod plans;
pub mod pool;
pub mod types;
pub mod view;

pub use error::ef_kernels_last_error;
pub use plans::*;
pub use types::*;

/// ABI version of this library's C surface.
///
/// It is bumped by any change to the layout of a C type, including a field added to any
/// params struct.
#[no_mangle]
pub extern "C" fn ef_kernels_abi_version() -> u32 {
    1
}

#[cfg(test)]
mod tests;
