// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The integer vocabularies, as named C enumerators.
//!
//! Each enumerator's value is asserted against the Rust vocabulary at compile
//! time.

use edgefirst_tensor::{CompressionScheme, Contiguity, CpuAccess, DType, TensorMemory};
pub use edgefirst_tensor_abi::{
    EfCompression, EfCompressionRequest, EfContiguity, EfCpuAccess, EfDtype, EfStorageKind,
};

/// Wire code -> validated map direction. Returns `None` for
/// `EF_CPU_ACCESS_NONE` (0) and for unknown codes.
///
/// For a declaration of CPU access, where `EF_CPU_ACCESS_NONE` is valid, use
/// [`declared_cpu_access_from_code`].
pub fn cpu_access_from_code(code: u32) -> Option<edgefirst_tensor::CpuAccess> {
    CpuAccess::from_code(code).filter(|a| *a != CpuAccess::None)
}

/// Wire code -> validated Rust access, for an allocation or wrap request.
/// `EF_CPU_ACCESS_NONE` (0) is accepted.
pub fn declared_cpu_access_from_code(code: u32) -> Option<edgefirst_tensor::CpuAccess> {
    CpuAccess::from_code(code)
}

/// Rust scheme -> wire code, with `None` (linear) as code 0.
pub fn compression_code(scheme: Option<edgefirst_tensor::CompressionScheme>) -> u32 {
    scheme
        .map(|s| s.code())
        .unwrap_or(EfCompression::None as u32)
}

/// Compile-time proof that the C enumerators and the Rust vocabulary agree,
/// for every variant.
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
    assert!(EfStorageKind::D3d11Texture as u32 == TensorMemory::D3d11Texture.code());

    assert!(EfCompression::Ubwc as u32 == CompressionScheme::Ubwc.code());
    assert!(EfCompression::Afbc as u32 == CompressionScheme::Afbc.code());
    assert!(EfCompression::Pvric as u32 == CompressionScheme::Pvric.code());
    assert!(EfCompression::Dcc as u32 == CompressionScheme::Dcc.code());
    // 0 is reserved for "linear", which `CompressionScheme` spells
    // `Option::None`.
    assert!(EfCompression::None as u32 == 0);

    assert!(EfContiguity::Unknown as u32 == Contiguity::Unknown.code());
    assert!(EfContiguity::Contiguous as u32 == Contiguity::Contiguous.code());
    assert!(EfContiguity::NonContiguous as u32 == Contiguity::NonContiguous.code());

    assert!(EfCpuAccess::None as u32 == CpuAccess::None.code());
    assert!(EfCpuAccess::Read as u32 == CpuAccess::Read.code());
    assert!(EfCpuAccess::Write as u32 == CpuAccess::Write.code());
    assert!(EfCpuAccess::ReadWrite as u32 == CpuAccess::ReadWrite.code());
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_dtype_has_a_c_enumerator() {
        assert_eq!(
            DType::all().len(),
            11,
            "a DType was added or removed; add the matching EfDtype enumerator \
             and its compile-time assertion"
        );
        assert_eq!(
            TensorMemory::all().len(),
            7,
            "a TensorMemory was added or removed; add the matching \
             EfStorageKind enumerator and its compile-time assertion"
        );
    }

    #[test]
    fn every_cpu_access_has_a_c_enumerator() {
        assert_eq!(
            CpuAccess::all().len(),
            4,
            "a CpuAccess was added or removed; add the matching EfCpuAccess \
             enumerator and its compile-time assertion"
        );
        for &a in CpuAccess::all() {
            assert_eq!(declared_cpu_access_from_code(a.code()), Some(a));
        }
        assert_eq!(cpu_access_from_code(CpuAccess::None.code()), None);
    }

    #[test]
    fn every_compression_scheme_has_a_c_enumerator() {
        assert_eq!(
            CompressionScheme::all().len(),
            4,
            "a CompressionScheme was added or removed; add the matching \
             EfCompression enumerator and its compile-time assertion"
        );
    }

    #[test]
    fn zero_is_reserved_for_linear_on_both_sides() {
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
        assert_eq!(CompressionScheme::from_code(0), None);
        assert_eq!(CompressionScheme::from_code(99), None);
        assert_eq!(CompressionScheme::from_code(u32::MAX), None);
    }

    #[test]
    fn cpu_access_from_code_round_trips_the_mappable_codes() {
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
