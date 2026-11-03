// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Threading abstraction: callers supply a pool, kernels split work into blocks.

use core::ops::Range;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Runs `n_tasks` independent tasks and returns after all have finished.
///
/// Kernels tolerate a pool that repeats, skips or invents task indices: each block still runs exactly once.
/// Tasks keep staging buffers of up to 32 KiB on the stack, so worker threads need about 64 KiB
/// of stack headroom beyond the pool's own use.
pub trait Pool: Sync {
    fn run(&self, n_tasks: usize, task: &(dyn Fn(usize) + Sync));
}

/// A [`Pool`] that spawns scoped threads per call.
pub struct ScopedThreads {
    threads: NonZeroUsize,
}

impl ScopedThreads {
    pub fn new(threads: NonZeroUsize) -> Self {
        Self { threads }
    }
}

impl Pool for ScopedThreads {
    fn run(&self, n_tasks: usize, task: &(dyn Fn(usize) + Sync)) {
        let workers = self.threads.get().min(n_tasks);
        if workers <= 1 {
            (0..n_tasks).for_each(task);
            return;
        }
        let next = AtomicUsize::new(0);
        let drain = || loop {
            let i = next.fetch_add(1, Ordering::Relaxed);
            if i >= n_tasks {
                break;
            }
            task(i);
        };
        std::thread::scope(|s| {
            for _ in 1..workers {
                // A failed spawn is tolerated: the calling thread drains the remaining tasks.
                let _ = std::thread::Builder::new().spawn_scoped(s, drain);
            }
            drain();
        });
    }
}

const MAX_BLOCKS: usize = 64;

/// Splits `0..n_items` into disjoint non-empty ranges and calls `f` exactly once per range, whatever `pool` does.
pub(crate) fn for_each_block(
    n_items: usize,
    pool: Option<&dyn Pool>,
    f: &(dyn Fn(Range<usize>) + Sync),
) {
    if n_items == 0 {
        return;
    }
    let Some(pool) = pool else {
        f(0..n_items);
        return;
    };
    let tasks = n_items.min(MAX_BLOCKS);
    let (base, rem) = (n_items / tasks, n_items % tasks);
    let claimed = [const { AtomicBool::new(false) }; MAX_BLOCKS];
    let run_block = |i: usize| {
        if i < tasks
            && claimed
                .get(i)
                .is_some_and(|c| !c.swap(true, Ordering::AcqRel))
        {
            let start = i * base + i.min(rem);
            f(start..start + base + usize::from(i < rem));
        }
    };
    // A safe `Pool` cannot keep `run_block` past `run`, so its blocks are finished and visible here.
    pool.run(tasks, &run_block);
    // Blocks the pool skipped run here; claimed ones are skipped.
    (0..tasks).for_each(run_block);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::num::NonZeroUsize;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    #[test]
    fn scoped_threads_runs_every_task_once() {
        let pool = ScopedThreads::new(NonZeroUsize::new(3).unwrap());
        for n in [0usize, 1, 7] {
            let counts: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            pool.run(n, &|i| {
                counts[i].fetch_add(1, Ordering::SeqCst);
            });
            assert!(
                counts.iter().all(|c| c.load(Ordering::SeqCst) == 1),
                "n={n}"
            );
        }
    }

    struct Serial;
    impl Pool for Serial {
        fn run(&self, n_tasks: usize, task: &(dyn Fn(usize) + Sync)) {
            for i in 0..n_tasks {
                task(i);
            }
        }
    }

    /// Calls every index twice, from two threads at once.
    struct Twice;
    impl Pool for Twice {
        fn run(&self, n_tasks: usize, task: &(dyn Fn(usize) + Sync)) {
            std::thread::scope(|s| {
                s.spawn(|| (0..n_tasks).for_each(task));
                (0..n_tasks).rev().for_each(task);
            });
        }
    }

    /// Calls out-of-range indices and only every other valid one.
    struct OutOfRange;
    impl Pool for OutOfRange {
        fn run(&self, n_tasks: usize, task: &(dyn Fn(usize) + Sync)) {
            for i in [usize::MAX, n_tasks, n_tasks + 1, usize::MAX / 2] {
                task(i);
            }
            (0..n_tasks).step_by(2).for_each(task);
        }
    }

    /// Calls nothing.
    struct Idle;
    impl Pool for Idle {
        fn run(&self, _: usize, _: &(dyn Fn(usize) + Sync)) {}
    }

    #[test]
    fn for_each_block_covers_range_exactly() {
        let scoped = ScopedThreads::new(NonZeroUsize::new(4).unwrap());
        let pools: [Option<&dyn Pool>; 6] = [
            None,
            Some(&Serial),
            Some(&scoped),
            Some(&Twice),
            Some(&OutOfRange),
            Some(&Idle),
        ];
        for pool in pools {
            for n in [0usize, 1, 5, 63, 64, 65, 1000] {
                let seen = Mutex::new(Vec::new());
                for_each_block(n, pool, &|r| {
                    assert!(r.start < r.end, "empty range");
                    seen.lock().unwrap().push(r);
                });
                let mut v = seen.into_inner().unwrap();
                v.sort_by_key(|r| r.start);
                let mut next = 0;
                for r in v {
                    assert_eq!(r.start, next);
                    next = r.end;
                }
                assert_eq!(next, n);
            }
        }
    }
}
