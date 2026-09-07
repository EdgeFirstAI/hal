// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The integer vocabularies, as named C enumerators.
//!
//! Without these a C caller writes `ef_tensor_builder_dtype(b, 0)` and hopes.
//! Worse, the bare integer is exactly how this repo previously ended up with
//! `TensorMemory.MEM == 3` in Python colliding with
//! `HAL_TENSOR_MEMORY_PBO == 3` in C — three surfaces each numbering the same
//! vocabulary independently.
//!
//! So each enumerator's value is asserted against the Rust vocabulary **at
//! compile time**. A renumbering on either side is a build failure here, not a
//! silent misinterpretation at a library boundary. This is the same mechanism
//! used for the Python bindings' discriminants.

use edgefirst_tensor::{CompressionScheme, DType, TensorMemory};
pub use edgefirst_tensor_abi::{EfCompression, EfCpuAccess, EfDtype, EfStorageKind};

/// Wire code -> validated *map direction*. `None` (0) and unknown codes are
/// not mappable directions: the wire rule says validate the integer, never
/// materialize an enum from it.
///
/// For the constructors, which take `access` as a *declaration* of what CPU
/// access to provision rather than as a direction to map in, use
/// [`declared_cpu_access_from_code`]: `EF_CPU_ACCESS_NONE` is a legitimate
/// answer there and a nonsensical one here.
pub fn cpu_access_from_code(code: u32) -> Option<edgefirst_tensor::CpuAccess> {
    match code {
        1 => Some(edgefirst_tensor::CpuAccess::Read),
        2 => Some(edgefirst_tensor::CpuAccess::Write),
        3 => Some(edgefirst_tensor::CpuAccess::ReadWrite),
        _ => None,
    }
}

/// Wire code -> validated Rust access, for an allocation or wrap request.
///
/// The same mapping `ef_tensor_image_desc_set_access` uses, `None` (0)
/// included: "no CPU access at all" is what a caller wrapping a texture it
/// will only ever touch from the GPU asks for, and it is what the Python
/// constructors default to. Refusing it would cost that caller a staging
/// texture it never reads.
pub fn declared_cpu_access_from_code(code: u32) -> Option<edgefirst_tensor::CpuAccess> {
    match code {
        0 => Some(edgefirst_tensor::CpuAccess::None),
        _ => cpu_access_from_code(code),
    }
}

/// Rust scheme -> wire code, with `None` (linear) as code 0.
///
/// The `Option` is the whole reason this is a function rather than a bare
/// `.code()` call: `CompressionScheme` has no variant for "linear", so the
/// absent case has to be folded into the same integer space here. Every
/// present case delegates to `.code()`, which is generated from the same
/// declaration the `EfCompression` enumerators are asserted against below --
/// there is no second table to drift.
pub fn compression_code(scheme: Option<edgefirst_tensor::CompressionScheme>) -> u32 {
    scheme
        .map(|s| s.code())
        .unwrap_or(EfCompression::None as u32)
}

/// Compile-time proof that the C enumerators and the Rust vocabulary agree.
///
/// Deliberately exhaustive rather than spot-checked: a mapping is only safe if
/// *every* variant lines up, and the one that gets renumbered is never the one
/// someone thought to sample.
const _: () = {
    assert!(EfDtype::U8 as u32 == DType::U8.code());
    assert!(EfDtype::I8 as u32 == DType::I8.code());
    assert!(EfDtype::U16 as u32 == DType::U16.code());
    assert!(EfDtype::I16 as u32 == DType::I16.code());
    assert!(EfDtype::U32 as u32 == DType::U32.code());
    assert!(EfDtype::I32 as u32 == DType::I32.code());
    assert!(EfDtype::U64 as u32 == DType::U64.code());
    assert!(EfDtype::I64 as u32 == DType::I64.code());
    assert!(EfDtype::F16 as u32 == DType::F16.code());
    assert!(EfDtype::F32 as u32 == DType::F32.code());
    assert!(EfDtype::F64 as u32 == DType::F64.code());

    assert!(EfStorageKind::Mem as u32 == TensorMemory::Mem.code());
    assert!(EfStorageKind::Shm as u32 == TensorMemory::Shm.code());
    assert!(EfStorageKind::DmaBuf as u32 == TensorMemory::DmaBuf.code());
    assert!(EfStorageKind::IoSurface as u32 == TensorMemory::IoSurface.code());
    assert!(EfStorageKind::Pbo as u32 == TensorMemory::Pbo.code());
    assert!(EfStorageKind::Cuda as u32 == TensorMemory::Cuda.code());

    assert!(EfCompression::Ubwc as u32 == CompressionScheme::Ubwc.code());
    assert!(EfCompression::Afbc as u32 == CompressionScheme::Afbc.code());
    assert!(EfCompression::Pvric as u32 == CompressionScheme::Pvric.code());
    assert!(EfCompression::Dcc as u32 == CompressionScheme::Dcc.code());
    // 0 is reserved for "linear", the state `CompressionScheme` spells
    // `Option::None` and therefore has no variant (hence no `.code()`) for.
    // That 0 stays unassigned on the Rust side is asserted at runtime, in
    // `zero_is_reserved_for_linear_on_both_sides` -- `from_code` is not a
    // `const fn`, so it cannot be checked here.
    assert!(EfCompression::None as u32 == 0);

    assert!(EfCpuAccess::None as u32 == 0);
    assert!(EfCpuAccess::Read as u32 == 1);
    assert!(EfCpuAccess::Write as u32 == 2);
    assert!(EfCpuAccess::ReadWrite as u32 == 3);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_dtype_has_a_c_enumerator() {
        // The const block above proves the values agree; this proves the set is
        // complete. Adding a Rust variant without a C one would otherwise be
        // invisible -- the assertions only cover what is listed.
        assert_eq!(
            DType::all().len(),
            11,
            "a DType was added or removed; add the matching EfDtype enumerator \
             and its compile-time assertion"
        );
        assert_eq!(
            TensorMemory::all().len(),
            6,
            "a TensorMemory was added or removed; add the matching \
             EfStorageKind enumerator and its compile-time assertion"
        );
    }

    #[test]
    fn every_compression_scheme_has_a_c_enumerator() {
        // Same shape as `every_dtype_has_a_c_enumerator`: the const block
        // above proves the values agree, this proves the set is complete.
        // Without it, adding a `CompressionScheme` variant is invisible --
        // the assertions only cover what is listed there.
        assert_eq!(
            CompressionScheme::all().len(),
            4,
            "a CompressionScheme was added or removed; add the matching \
             EfCompression enumerator and its compile-time assertion"
        );
    }

    #[test]
    fn zero_is_reserved_for_linear_on_both_sides() {
        // The half of the `EfCompression::None == 0` assertion that the
        // const block above cannot make (`from_code` is not `const`): that
        // no `CompressionScheme` variant claims code 0. A scheme declared
        // as 0 would round-trip as "linear" and decode garbage.
        assert_eq!(CompressionScheme::from_code(0), None);
    }

    #[test]
    fn every_compression_scheme_round_trips_its_wire_code() {
        for &scheme in CompressionScheme::all() {
            let code = compression_code(Some(scheme));
            assert_eq!(
                CompressionScheme::from_code(code),
                Some(scheme),
                "{scheme:?} did not survive the wire code {code}"
            );
        }
        assert_eq!(compression_code(None), 0);
    }

    #[test]
    fn an_unknown_compression_code_maps_to_no_scheme() {
        // `from_code` has no variant to return for a code this build does
        // not know, so it answers `None`. What a *consumer* should do with
        // that is a separate decision, made and justified where the
        // consumer is -- `TensorDyn::compression` (`dynamic_backend.rs`),
        // which logs the unrecognised code rather than passing it off as a
        // linear layout in silence.
        assert_eq!(CompressionScheme::from_code(0), None);
        assert_eq!(CompressionScheme::from_code(99), None);
        assert_eq!(CompressionScheme::from_code(u32::MAX), None);
    }

    #[test]
    fn cpu_access_from_code_round_trips_the_mappable_codes() {
        use edgefirst_tensor::CpuAccess;

        let read = cpu_access_from_code(1).expect("1 is Read");
        assert_eq!(read, CpuAccess::Read);
        assert!(read.reads() && !read.writes());

        let write = cpu_access_from_code(2).expect("2 is Write");
        assert_eq!(write, CpuAccess::Write);
        assert!(!write.reads() && write.writes());

        let read_write = cpu_access_from_code(3).expect("3 is ReadWrite");
        assert_eq!(read_write, CpuAccess::ReadWrite);
        assert!(read_write.reads() && read_write.writes());
    }

    #[test]
    fn cpu_access_from_code_rejects_none_and_unknown_codes() {
        assert_eq!(cpu_access_from_code(0), None);
        assert_eq!(cpu_access_from_code(4), None);
        assert_eq!(cpu_access_from_code(u32::MAX), None);
    }
}
