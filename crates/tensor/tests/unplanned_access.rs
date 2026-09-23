// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `unplanned_cpu_access_count` counts exactly the maps that exceed a
//! buffer's declared [`CpuAccess`].
//!
//! The counter is process-global, so this binary holds a single test: no
//! other test in the process can map a tensor between its reads. Only the
//! `static` backend produces the telemetry; under `dynamic` the count is
//! always zero by design.

#![cfg(feature = "static")]

use edgefirst_tensor::{
    unplanned_cpu_access_count, CpuAccess, PixelFormat, Tensor, TensorMemory, TensorTrait,
};

fn image(access: CpuAccess) -> Tensor<u8> {
    Tensor::<u8>::image(100, 8, PixelFormat::Rgb, Some(TensorMemory::Mem), access)
        .expect("Mem RGB 100x8")
}

#[test]
fn only_maps_beyond_the_declared_access_are_counted() {
    let start = unplanned_cpu_access_count();

    // Declared ReadWrite: every direction is planned.
    let rw = image(CpuAccess::ReadWrite);
    drop(rw.map_read().unwrap());
    drop(rw.map_write().unwrap());
    drop(rw.map_mut().unwrap());
    assert_eq!(unplanned_cpu_access_count(), start);

    // Declared Read: reading is planned, writing is not -- and each such map
    // counts, not only the first (the warning is once per buffer, the count
    // is not).
    let ro = image(CpuAccess::Read);
    drop(ro.map_read().unwrap());
    assert_eq!(unplanned_cpu_access_count(), start);
    drop(ro.map_mut().unwrap());
    assert_eq!(unplanned_cpu_access_count(), start + 1);
    drop(ro.map_write().unwrap());
    assert_eq!(unplanned_cpu_access_count(), start + 2);

    // Declared None: any CPU map is unplanned.
    let hw = image(CpuAccess::None);
    drop(hw.map_read().unwrap());
    assert_eq!(unplanned_cpu_access_count(), start + 3);
}
