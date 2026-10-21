// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The mapped CPU-access direction.
//!
//! `map()` owns its coherency bracket in both directions, and the direction
//! decides how much cache maintenance the release performs. The direction is
//! not observable from the data -- the bracket is a side effect -- so what is
//! asserted here is the property a binding can actually build on:
//! [`TensorMapTrait::is_writable`] reports the direction the mapping was
//! taken with. The Python buffer protocol's `readonly` flag is derived from
//! it, so a regression here surfaces as a writable view over a read-only map.

use edgefirst_tensor::{
    CpuAccess, DType, Error, ImageDesc, PixelFormat, Tensor, TensorDyn, TensorMapTrait,
    TensorMemory, TensorTrait,
};

fn image() -> Tensor<u8> {
    Tensor::<u8>::image(
        16,
        8,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc")
}

#[test]
fn map_with_reports_the_direction_it_was_taken_with() {
    let t = image();
    assert!(
        !t.map_with(CpuAccess::Read).expect("read map").is_writable(),
        "a read map must not report itself writable"
    );
    assert!(t
        .map_with(CpuAccess::Write)
        .expect("write map")
        .is_writable());
    assert!(t
        .map_with(CpuAccess::ReadWrite)
        .expect("rw map")
        .is_writable());
}

#[test]
fn plain_map_is_readwrite() {
    // `map()` is documented as `map_with(ReadWrite)`. Bindings default to it,
    // so this is the behaviour existing callers depend on.
    assert!(image().map().expect("map").is_writable());
}

#[test]
fn map_read_matches_map_with_read() {
    assert!(!image().map_read().expect("map_read").is_writable());
    assert!(image().map_write().expect("map_write").is_writable());
}

#[test]
fn a_read_map_still_reads() {
    // The direction restricts writes, not reads -- a read-only map is only
    // useful if it can actually be read.
    let t = image();
    {
        let mut m = t.map_with(CpuAccess::ReadWrite).expect("rw");
        m.as_mut_slice()[0] = 0x5A;
    }
    let m = t.map_with(CpuAccess::Read).expect("read");
    assert_eq!(m.as_slice()[0], 0x5A);
}

#[test]
fn none_is_not_a_mappable_direction() {
    assert!(
        image().map_with(CpuAccess::None).is_err(),
        "CpuAccess::None is not a direction a mapping can be taken in"
    );
}

// The two tests below are the only coverage of `assert_map_writable`
// (crates/tensor/src/lib.rs), which since the per-backend map types collapsed
// into `HostView` is the single point enforcing the read-only contract for all
// six backends. Without them, deleting that assertion leaves the whole suite
// green: every other test here checks that `is_writable()` *reports* the
// direction, not that acting against it is refused. On Android the assertion is
// what stands between a `map_read()` misuse and writing through a read-only
// AHardwareBuffer lock, which the NDK documents as undefined behaviour.

#[test]
#[should_panic(expected = "read-only")]
fn as_mut_slice_on_a_read_map_is_refused() {
    let t = image();
    let mut m = t.map_with(CpuAccess::Read).expect("read");
    let _ = m.as_mut_slice();
}

#[test]
#[should_panic(expected = "read-only")]
fn deref_mut_on_a_read_map_is_refused() {
    let t = image();
    let mut m = t.map_with(CpuAccess::Read).expect("read");
    m[0] = 0xFF;
}

fn declared(access: CpuAccess) -> TensorDyn {
    TensorDyn::image(
        16,
        8,
        PixelFormat::Rgb,
        DType::U8,
        Some(TensorMemory::Mem),
        access,
    )
    .expect("alloc")
}

fn assert_denied<T>(r: edgefirst_tensor::Result<T>, declared_as: CpuAccess, requested: CpuAccess) {
    match r {
        Err(Error::CpuAccessDenied {
            declared: d,
            requested: q,
        }) => {
            assert_eq!(d, declared_as);
            assert_eq!(q, requested);
        }
        Err(e) => {
            panic!("{requested:?} on a {declared_as:?} tensor: expected CpuAccessDenied, got {e:?}")
        }
        Ok(_) => panic!("{requested:?} on a {declared_as:?} tensor must be refused"),
    }
}

// Runs on both backends (map_access is a dynamic test target), so the error
// kind is the same whichever side refuses.
#[test]
fn a_map_outside_the_declaration_is_refused_in_both_directions() {
    for (decl, req) in [
        (CpuAccess::Write, CpuAccess::Read),
        (CpuAccess::None, CpuAccess::Read),
        (CpuAccess::Read, CpuAccess::Write),
        (CpuAccess::None, CpuAccess::Write),
        (CpuAccess::Read, CpuAccess::ReadWrite),
        (CpuAccess::Write, CpuAccess::ReadWrite),
        (CpuAccess::None, CpuAccess::ReadWrite),
    ] {
        let t = declared(decl);
        assert_eq!(t.cpu_access(), decl);
        assert_denied(t.map_bytes(req), decl, req);
        let typed = t.as_u8().expect("u8 image");
        assert_denied(typed.map_with(req), decl, req);
    }
}

#[test]
fn a_pin_or_sync_outside_the_declaration_is_refused() {
    let t = declared(CpuAccess::Read);
    assert_denied(
        t.pin_host(CpuAccess::Write),
        CpuAccess::Read,
        CpuAccess::Write,
    );
    assert_denied(
        t.sync_for_cpu(CpuAccess::ReadWrite),
        CpuAccess::Read,
        CpuAccess::ReadWrite,
    );
    assert!(t.pin_host(CpuAccess::Read).is_ok());
    assert!(t.sync_for_cpu(CpuAccess::Read).is_ok());
    assert!(t.sync_for_device(CpuAccess::Read).is_ok());
}

#[test]
fn covered_maps_succeed() {
    for (decl, req) in [
        (CpuAccess::Read, CpuAccess::Read),
        (CpuAccess::Write, CpuAccess::Write),
        (CpuAccess::ReadWrite, CpuAccess::Read),
        (CpuAccess::ReadWrite, CpuAccess::Write),
        (CpuAccess::ReadWrite, CpuAccess::ReadWrite),
    ] {
        assert!(declared(decl).map_bytes(req).is_ok(), "{req:?} on {decl:?}");
    }
}

#[test]
fn the_default_declaration_is_readwrite() {
    assert_eq!(CpuAccess::default(), CpuAccess::ReadWrite);
    let desc = ImageDesc::new(16, 8, PixelFormat::Rgb, DType::U8);
    assert_eq!(desc.access(), CpuAccess::ReadWrite);
    let t = TensorDyn::image_desc(&desc.with_memory(Some(TensorMemory::Mem))).expect("alloc");
    assert_eq!(t.cpu_access(), CpuAccess::ReadWrite);
    assert_eq!(
        Tensor::<u8>::new(&[4], Some(TensorMemory::Mem), None)
            .expect("alloc")
            .cpu_access(),
        CpuAccess::ReadWrite
    );
}

// An alias imported from a descriptor keeps its producer's declaration, so
// the refusal does not depend on which handle the CPU maps through.
#[test]
fn an_imported_alias_keeps_the_producer_declaration() {
    let t = declared(CpuAccess::Read);
    let pin = t.pin_host(CpuAccess::Read).expect("read pin");
    let desc = t.descriptor_pinned(Some(&pin));
    assert_eq!(
        edgefirst_tensor::protocol::flags::cpu_access(desc.flags),
        Some(CpuAccess::Read)
    );
    let alias = TensorDyn::import_descriptor(&desc).expect("import");
    assert_eq!(alias.cpu_access(), CpuAccess::Read);
    assert_denied(
        alias.map_bytes(CpuAccess::Write),
        CpuAccess::Read,
        CpuAccess::Write,
    );
    assert!(alias.map_bytes(CpuAccess::Read).is_ok());
}

#[test]
fn restricting_never_widens() {
    let mut t = declared(CpuAccess::Read);
    t.restrict_cpu_access(CpuAccess::ReadWrite);
    assert_eq!(t.cpu_access(), CpuAccess::Read);
    t.restrict_cpu_access(CpuAccess::Write);
    assert_eq!(t.cpu_access(), CpuAccess::None);
}

#[test]
fn map_for_write_takes_write_on_write_only_and_refuses_read_and_none() {
    let write_only = declared(CpuAccess::Write);
    let typed = write_only.as_u8().expect("u8 image");
    assert_eq!(typed.write_access(), CpuAccess::Write);
    typed.map_for_write().expect("write map on a Write tensor");
    assert_eq!(
        declared(CpuAccess::ReadWrite)
            .as_u8()
            .expect("u8 image")
            .write_access(),
        CpuAccess::ReadWrite
    );
    let read_only = declared(CpuAccess::Read);
    assert_denied(
        read_only.as_u8().expect("u8 image").map_for_write(),
        CpuAccess::Read,
        CpuAccess::ReadWrite,
    );
    let none = declared(CpuAccess::None);
    assert_denied(
        none.as_u8().expect("u8 image").map_for_write(),
        CpuAccess::None,
        CpuAccess::Write,
    );
}

// A write of part of a buffer must invalidate, so it needs `ReadWrite`.
#[test]
fn a_partial_write_map_needs_readwrite() {
    let write_only = declared(CpuAccess::Write);
    assert_denied(
        write_only
            .as_u8()
            .expect("u8 image")
            .map_for_partial_write(),
        CpuAccess::Write,
        CpuAccess::ReadWrite,
    );
    let rw = declared(CpuAccess::ReadWrite);
    assert!(rw
        .as_u8()
        .expect("u8 image")
        .map_for_partial_write()
        .expect("partial write on ReadWrite")
        .is_writable());
}

#[test]
fn a_view_is_written_readwrite() {
    let region = || edgefirst_tensor::Region::new(4, 2, 8, 4);
    let parent = declared(CpuAccess::Write);
    let view = parent.view(region()).expect("view");
    let typed = view.as_u8().expect("u8 view");
    assert_eq!(typed.write_access(), CpuAccess::ReadWrite);
    assert_denied(
        typed.map_for_write(),
        CpuAccess::Write,
        CpuAccess::ReadWrite,
    );
    let rw = declared(CpuAccess::ReadWrite);
    let rw_view = rw.view(region()).expect("view");
    rw_view
        .as_u8()
        .expect("u8 view")
        .map_for_write()
        .expect("write map on a ReadWrite view");
}

#[test]
fn restricting_none_stays_none_and_narrows_chroma() {
    let mut t = declared(CpuAccess::None);
    for access in CpuAccess::all() {
        t.restrict_cpu_access(*access);
        assert_eq!(t.cpu_access(), CpuAccess::None);
    }
    let (w, h) = (16usize, 8usize);
    let plane = |shape: &[usize]| Tensor::<u8>::new(shape, Some(TensorMemory::Mem), None);
    let luma = plane(&[h, w]).expect("luma");
    let chroma = plane(&[h / 2, w]).expect("chroma");
    let mut nv12 = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12).expect("nv12");
    assert!(nv12.is_multiplane());
    nv12.restrict_cpu_access(CpuAccess::Read);
    assert_eq!(nv12.cpu_access(), CpuAccess::Read);
    let chroma = nv12.chroma().expect("chroma plane");
    assert_eq!(chroma.cpu_access(), CpuAccess::Read);
    assert_denied(
        chroma.map_with(CpuAccess::Write),
        CpuAccess::Read,
        CpuAccess::Write,
    );
}

#[test]
fn flags_round_trip_every_cpu_access() {
    use edgefirst_tensor::protocol::flags;
    for access in CpuAccess::all() {
        assert_eq!(
            flags::cpu_access(flags::from_cpu_access(*access)),
            Some(*access)
        );
    }
    assert_eq!(flags::cpu_access(0), None);
    assert_eq!(flags::cpu_access(flags::CPU_READ), None);
}

// A descriptor without `CPU_ACCESS_DECLARED` leaves the importer's default.
#[test]
fn an_undeclared_descriptor_imports_readwrite() {
    use edgefirst_tensor::protocol::flags;
    let t = declared(CpuAccess::Read);
    let pin = t.pin_host(CpuAccess::Read).expect("read pin");
    let mut desc = t.descriptor_pinned(Some(&pin));
    desc.flags &= !(flags::CPU_ACCESS_DECLARED | flags::CPU_READ | flags::CPU_WRITE);
    let alias = TensorDyn::import_descriptor(&desc).expect("import");
    assert_eq!(alias.cpu_access(), CpuAccess::ReadWrite);
}
