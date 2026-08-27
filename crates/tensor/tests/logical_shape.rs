// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! `set_logical_shape` is capacity-based, and `reshape` is not.
//!
//! Two methods with deliberately different contracts.
//! [`TensorTrait::reshape`] keeps the element count and rejects anything
//! else; [`TensorTrait::set_logical_shape`] accepts any shape whose bytes
//! fit the allocation, which is what lets an oversized reusable pool buffer
//! be reconfigured to a smaller image without reallocating.
//!
//! Every storage type implements the capacity-based one
//! (`mem.rs`, `dma.rs`, `shm.rs`, `pbo.rs`, `iosurface.rs`,
//! `ahardwarebuffer.rs`), and `Tensor::configure_image` reaches them by
//! calling `self.storage.set_logical_shape(..)` directly. `Tensor<T>`
//! itself, though, did not override the trait method for most of this
//! crate's life -- it inherited the default, whose body is
//! `self.reshape(shape)`. So a caller going through `Tensor<T>` got
//! `reshape`'s strict rule under a name that promises the opposite, on
//! **both** backends.
//!
//! Not cfg-gated: this is exactly the kind of defect that is identical on
//! both backends and therefore invisible to G13, which compares them
//! against each other. It has to be asserted directly.

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

    // Half the elements: a different count, so `reshape` must refuse it --
    // and `set_logical_shape` must not, because it fits.
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
    // Same count, different rank: reshape's own job, unaffected by any of
    // the above.
    t.reshape(&[256])
        .expect("an equal element count is what reshape allows");
    assert_eq!(t.shape(), &[256]);
}
