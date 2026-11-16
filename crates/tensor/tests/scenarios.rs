// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Aliasing scenarios for `TensorDyn`'s raw-handle borrow shapes, run under
//! Miri with both aliasing models (`scripts/miri.sh` runs this file with
//! Stacked Borrows, then again with `MIRIFLAGS=-Zmiri-tree-borrows`).
//!
//! The file runs against the static backend (`tensor_dyn/static_backend.rs`).
use edgefirst_tensor::{Tensor, TensorDyn, TensorTrait};

#[test]
fn with_raw_reborrow_is_exclusive_for_its_whole_window() {
    let t: Tensor<u8> = Tensor::new(&[8], None, None).expect("alloc");
    let d: TensorDyn = t.into();
    let raw = TensorDyn::into_raw(d);
    // SAFETY: `raw` is live and unaliased for this call.
    unsafe {
        TensorDyn::with_raw(raw, |a| {
            let lens = a.as_typed_mut::<u8>().expect("u8");
            lens.map_write().expect("map")[0] = 1;
        });
        drop(TensorDyn::from_raw(raw));
    }
}

/// Two `&mut` reach one allocation; `scripts/miri.sh` runs it with `--ignored` and expects failure.
#[test]
#[ignore = "diagnostic: fails under Stacked Borrows, passes under Tree Borrows; run by scripts/miri.sh"]
fn unwrap_then_use_aliases_the_same_tensor() {
    let t: Tensor<u8> = Tensor::new(&[8], None, None).expect("alloc");
    let d: TensorDyn = t.into();
    let raw = TensorDyn::into_raw(d);
    // SAFETY: none holds; two `&mut` reach the same allocation.
    unsafe {
        let a = &mut *(raw as *mut TensorDyn); // borrow taken...
        let b = &mut *(raw as *mut TensorDyn); // ...and a second, aliasing it
        let _ = a.dtype();
        let _ = b.dtype();
        drop(TensorDyn::from_raw(raw));
    }
}

#[test]
fn borrow_guard_leaves_the_handle_live_and_writable() {
    let t: Tensor<u8> = Tensor::new(&[8], None, None).expect("alloc");
    let raw = TensorDyn::into_raw(t.into());
    // SAFETY: `raw` is live and unaliased until `from_raw` retakes it.
    unsafe {
        {
            let mut g = TensorDyn::borrow_raw(raw);
            assert_eq!(g.shape(), &[8]);
            let _ = g.quantization();
            g.as_typed_mut::<u8>()
                .expect("u8")
                .map_write()
                .expect("map")[0] = 7;
        }
        TensorDyn::with_raw(raw, |t| {
            t.as_typed_mut::<u8>()
                .expect("u8")
                .map_write()
                .expect("map")[1] = 9;
        });
        let owned = TensorDyn::from_raw(raw);
        let lens = owned.as_typed::<u8>().expect("u8");
        lens.map_write().expect("writable after the guards dropped")[2] = 1;
        let m = lens.map_read().expect("map");
        assert_eq!(&m[..3], &[7, 9, 1]);
    }
}

#[test]
fn borrowing_a_handle_does_not_grow_the_heap() {
    let t: Tensor<u8> = Tensor::new(&[2, 4], None, None).expect("alloc");
    let raw = TensorDyn::into_raw(t.into());
    let borrow = || {
        // SAFETY: `raw` is live and unaliased until `from_raw` retakes it.
        unsafe {
            let g = TensorDyn::borrow_raw(raw);
            let _ = g.quantization();
            TensorDyn::with_raw(raw, |t| t.shape().len())
        }
    };
    let n = if cfg!(miri) { 20 } else { 1000 };
    for _ in 0..10 {
        borrow();
    }
    let before = counting::live_bytes();
    for _ in 0..n {
        borrow();
    }
    let grown = counting::live_bytes() - before;
    assert!(grown < 1024, "heap grew {grown} bytes over {n} borrows");
    // SAFETY: `raw` came from `into_raw` above and is retaken once.
    drop(unsafe { TensorDyn::from_raw(raw) });
}

/// A global allocator counting the bytes live on the current thread.
mod counting {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static LIVE: Cell<isize> = const { Cell::new(0) };
    }

    struct Counting;

    // SAFETY: forwards every call to `System`; the counter is a const thread local.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            let _ = LIVE.try_with(|c| c.set(c.get() + l.size() as isize));
            unsafe { System.alloc(l) }
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            let _ = LIVE.try_with(|c| c.set(c.get() - l.size() as isize));
            unsafe { System.dealloc(p, l) }
        }
    }

    #[global_allocator]
    static ALLOC: Counting = Counting;

    pub fn live_bytes() -> isize {
        LIVE.with(Cell::get)
    }
}
