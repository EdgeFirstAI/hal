// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! A global allocator for the leak tests: counts the bytes live on the
//! current thread.

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

/// Bytes the live heap grows by over `n` calls of `f`, after a warm-up.
///
/// Runs on a one-thread rayon pool so the decoder's parallel work allocates
/// and frees on the counted thread instead of the global pool's workers.
pub(crate) fn growth_over(n: usize, f: impl FnMut()) -> isize {
    struct AssertSend<F>(F);
    // SAFETY: `install` blocks the caller until the closure returns, so the
    // raw handles it captures are never used from two threads at once.
    unsafe impl<F> Send for AssertSend<F> {}
    impl<F> AssertSend<F> {
        fn into_inner(self) -> F {
            self.0
        }
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("one-thread pool");
    let f = AssertSend(f);
    pool.install(move || {
        let mut f = f.into_inner();
        for _ in 0..50 {
            f();
        }
        let before = LIVE.with(Cell::get);
        for _ in 0..n {
            f();
        }
        LIVE.with(Cell::get) - before
    })
}
