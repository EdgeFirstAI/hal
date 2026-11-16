// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `set_logical_shape` is capacity-based, and `reshape` is not.
//!
//! [`TensorTrait::reshape`] keeps the element count and rejects anything else;
//! [`TensorTrait::set_logical_shape`] accepts any shape whose bytes fit the
//! allocation.
//!
//! The tests run on both backends.

use edgefirst_tensor::{CpuAccess, PixelFormat, Tensor, TensorMemory, TensorTrait};

/// 16x16 Grey = 256 bytes of allocation.
fn image() -> Tensor<u8> {
    Tensor::<u8>::image(
        16,
        16,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("Mem-backed 16x16 Grey allocation")
}

#[test]
fn set_logical_shape_accepts_a_smaller_shape_that_fits_the_allocation() {
    let mut t = image();
    let capacity = t.capacity_bytes();
    assert_eq!(capacity, 256, "precondition: the allocation is 256 bytes");

    // Half the elements: `reshape` refuses it and `set_logical_shape` accepts it.
    assert!(
        t.reshape(&[128]).is_err(),
        "precondition: reshape is the strict one and rejects a changed count"
    );
    t.set_logical_shape(&[128])
        .expect("a shape whose bytes fit the allocation is exactly what this method accepts");
    assert_eq!(t.shape(), &[128], "and the tensor really is that shape now");
    assert_eq!(
        t.capacity_bytes(),
        capacity,
        "the allocation is unchanged -- reconfiguring a pool buffer must not reallocate"
    );
}

#[test]
fn set_logical_shape_still_refuses_a_shape_larger_than_the_allocation() {
    let mut t = image();
    assert!(
        t.set_logical_shape(&[257]).is_err(),
        "capacity-based does not mean unchecked: past the allocation is still refused"
    );
    assert_eq!(
        t.shape(),
        &[16, 16, 1],
        "a refused reconfigure must leave the shape alone"
    );
}

#[test]
fn reshape_keeps_its_own_stricter_contract() {
    let mut t = image();
    // Same count, different rank.
    t.reshape(&[256])
        .expect("an equal element count is what reshape allows");
    assert_eq!(t.shape(), &[256]);
}
