// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Adapter from a C `ef_kernel_pool` to the kernels crate's `Pool`.

use std::ffi::c_void;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};

use edgefirst_kernels::Pool;

use crate::error::CError;
use crate::types::EfKernelPool;

type RunFn =
    unsafe extern "C" fn(*mut c_void, usize, unsafe extern "C" fn(*mut c_void, usize), *mut c_void);

pub struct CPool {
    ctx: *mut c_void,
    run: RunFn,
    panicked: AtomicBool,
}

// SAFETY: the C contract requires `run` and `ctx` to be usable from any thread.
unsafe impl Sync for CPool {}

impl CPool {
    pub fn new(p: &EfKernelPool) -> Result<Self, CError> {
        let run = p.run.ok_or(CError::Invalid("pool run callback is NULL"))?;
        Ok(Self {
            ctx: p.ctx,
            run,
            panicked: AtomicBool::new(false),
        })
    }

    /// Whether a task panicked during any `run` on this adapter; its block was left unfinished.
    pub fn panicked(&self) -> bool {
        self.panicked.load(Ordering::Acquire)
    }
}

/// What the trampoline's `arg` points at for the duration of one `run`.
struct Call<'a> {
    task: &'a (dyn Fn(usize) + Sync),
    panicked: &'a AtomicBool,
}

unsafe extern "C" fn trampoline(arg: *mut c_void, index: usize) {
    // SAFETY: `arg` is the address of the `Call` that `CPool::run` keeps alive across the call.
    let call = unsafe { &*arg.cast::<Call<'_>>() };
    // A panic must not unwind into C; it is recorded so `plan_run` reports the unfinished block.
    if catch_unwind(AssertUnwindSafe(|| (call.task)(index))).is_err() {
        call.panicked.store(true, Ordering::Release);
    }
}

impl Pool for CPool {
    fn run(&self, n_tasks: usize, task: &(dyn Fn(usize) + Sync)) {
        let call = Call {
            task,
            panicked: &self.panicked,
        };
        // SAFETY: `run` is the caller's callback; `call` outlives the call it blocks on.
        unsafe {
            (self.run)(
                self.ctx,
                n_tasks,
                trampoline,
                std::ptr::from_ref(&call).cast_mut().cast(),
            );
        }
    }
}
