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

use edgefirst_tensor::{CpuAccess, PixelFormat, Tensor, TensorMapTrait, TensorMemory, TensorTrait};

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
