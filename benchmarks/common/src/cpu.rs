// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Process + system CPU load sampling for benchmark runs (Linux `/proc`).

use std::fs;
use std::time::Instant;

/// Snapshot of process + system CPU counters.
#[derive(Debug, Clone)]
pub struct CpuSnapshot {
    wall: Instant,
    /// utime + stime from `/proc/self/stat` (jiffies).
    process_jiffies: u64,
    /// Per-CPU (user+nice+system+irq+softirq+steal, idle, iowait) totals.
    /// `cores[i] = (busy, idle_like)` where idle_like = idle + iowait.
    cores: Vec<(u64, u64)>,
}

#[derive(Debug, Clone, Default)]
pub struct CpuStats {
    /// Process CPU time / wall × 100 (may exceed 100% on multi-core).
    pub process_pct: f64,
    /// System-wide non-idle fraction × 100 over the interval.
    pub system_pct: f64,
    /// Busiest single core's non-idle fraction × 100 over the interval.
    pub peak_core_pct: f64,
    /// Max per-sample peak_core seen during the run (if sampled).
    pub peak_core_pct_max: f64,
}

impl CpuSnapshot {
    pub fn now() -> Self {
        Self {
            wall: Instant::now(),
            process_jiffies: read_process_jiffies(),
            cores: read_per_core_busy_idle(),
        }
    }

    pub fn delta_since(&self, start: &Self) -> CpuStats {
        let wall_s = self.wall.duration_since(start.wall).as_secs_f64();
        let hz = proc_hz();
        let proc_s = (self.process_jiffies.saturating_sub(start.process_jiffies)) as f64 / hz;
        let process_pct = if wall_s > 0.0 {
            100.0 * proc_s / wall_s
        } else {
            0.0
        };

        let mut busy_sum = 0u64;
        let mut total_sum = 0u64;
        let mut peak_core = 0.0f64;
        let n = start.cores.len().min(self.cores.len());
        for i in 0..n {
            let (b0, i0) = start.cores[i];
            let (b1, i1) = self.cores[i];
            let db = b1.saturating_sub(b0);
            let di = i1.saturating_sub(i0);
            let dt = db + di;
            busy_sum += db;
            total_sum += dt;
            if dt > 0 {
                let pct = 100.0 * (db as f64) / (dt as f64);
                if pct > peak_core {
                    peak_core = pct;
                }
            }
        }
        let system_pct = if total_sum > 0 {
            100.0 * (busy_sum as f64) / (total_sum as f64)
        } else {
            0.0
        };

        CpuStats {
            process_pct,
            system_pct,
            peak_core_pct: peak_core,
            peak_core_pct_max: peak_core,
        }
    }
}

/// Running peak-core tracker: call `sample()` periodically during the hot loop.
#[derive(Debug, Default)]
pub struct CpuPeakTracker {
    last: Option<CpuSnapshot>,
    peak_core_pct_max: f64,
}

impl CpuPeakTracker {
    pub fn start() -> Self {
        Self {
            last: Some(CpuSnapshot::now()),
            peak_core_pct_max: 0.0,
        }
    }

    pub fn sample(&mut self) {
        let now = CpuSnapshot::now();
        if let Some(prev) = &self.last {
            let d = now.delta_since(prev);
            if d.peak_core_pct > self.peak_core_pct_max {
                self.peak_core_pct_max = d.peak_core_pct;
            }
        }
        self.last = Some(now);
    }

    pub fn peak_core_pct_max(&self) -> f64 {
        self.peak_core_pct_max
    }
}

/// Advisory scheduling hint for boards with no `taskset`-equivalent (macOS):
/// bias the calling thread toward the performance cores and away from being
/// pre-empted by lower-QoS background work. Not a hard pin — the scheduler
/// can still migrate the thread — but it's the best substitute available on
/// Apple Silicon, where unpinned P-core/E-core migration between rounds is a
/// measured source of extra spread (see BENCHMARKS.md's mbp-m2-max
/// discussion). Call once, before the timed loop. No-op on non-macOS.
#[cfg(target_os = "macos")]
pub fn pin_qos() {
    // QOS_CLASS_USER_INTERACTIVE = 0x21, from Apple's <pthread/qos.h>; no
    // `libc`-crate binding exists for this Apple-specific API, so it's
    // declared directly.
    #[allow(non_camel_case_types)]
    type qos_class_t = u32;
    const QOS_CLASS_USER_INTERACTIVE: qos_class_t = 0x21;
    extern "C" {
        fn pthread_set_qos_class_self_np(qos_class: qos_class_t, relative_priority: i32) -> i32;
    }
    // SAFETY: FFI call with no pointer arguments; a nonzero return (e.g. the
    // thread already has an explicit QoS class) is advisory-only and safe to
    // ignore — this is a best-effort scheduling hint, not a correctness
    // requirement.
    unsafe {
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
}

#[cfg(not(target_os = "macos"))]
pub fn pin_qos() {}

fn proc_hz() -> f64 {
    // POSIX clocks per second; 100 on virtually every Linux embedded target.
    unsafe { libc::sysconf(libc::_SC_CLK_TCK) as f64 }.max(1.0)
}

fn read_process_jiffies() -> u64 {
    let Ok(stat) = fs::read_to_string("/proc/self/stat") else {
        return 0;
    };
    // comm can contain spaces/parens — find the trailing ") " then split fields.
    let Some(idx) = stat.rfind(") ") else {
        return 0;
    };
    let fields: Vec<&str> = stat[idx + 2..].split_whitespace().collect();
    // After comm: state(0) ppid(1) ... utime(11) stime(12) — 0-based in this slice
    let utime: u64 = fields.get(11).and_then(|s| s.parse().ok()).unwrap_or(0);
    let stime: u64 = fields.get(12).and_then(|s| s.parse().ok()).unwrap_or(0);
    utime + stime
}

fn read_per_core_busy_idle() -> Vec<(u64, u64)> {
    let Ok(stat) = fs::read_to_string("/proc/stat") else {
        return Vec::new();
    };
    let mut cores = Vec::new();
    for line in stat.lines() {
        if !line.starts_with("cpu") || line.starts_with("cpu ") {
            continue; // skip aggregate "cpu " line; keep cpu0, cpu1, …
        }
        let mut parts = line.split_whitespace();
        let Some(name) = parts.next() else { continue };
        if name == "cpu"
            || !name
                .as_bytes()
                .get(3)
                .copied()
                .is_some_and(|c| c.is_ascii_digit())
        {
            continue;
        }
        // user nice system idle iowait irq softirq steal guest guest_nice
        let nums: Vec<u64> = parts.filter_map(|s| s.parse().ok()).collect();
        if nums.len() < 5 {
            continue;
        }
        let idle = nums[3];
        let iowait = nums[4];
        let idle_like = idle + iowait;
        let total: u64 = nums.iter().take(8).sum();
        let busy = total.saturating_sub(idle_like);
        cores.push((busy, idle_like));
    }
    cores
}
