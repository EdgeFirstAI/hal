// SPDX-FileCopyrightText: Copyright 2025-2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Lightweight micro-benchmark harness for EdgeFirst crates.
//!
//! Runs all benchmarks sequentially in a single process with no forking,
//! making it safe for GPU hardware that cannot survive `fork()` (EGL, G2D,
//! Vulkan).
//!
//! # Usage
//!
//! ```rust,no_run
//! use edgefirst_bench::{run_bench, format_throughput};
//!
//! let result = run_bench("my_operation", 5, 100, || {
//!     // ... code to benchmark ...
//! });
//! result.print_summary();
//! // or with throughput:
//! result.print_summary_with_throughput(1920 * 1080 * 4);
//! ```

use std::time::{Duration, Instant};

/// Result of a single benchmark run, holding per-iteration timings.
pub struct BenchResult {
    pub name: String,
    pub iterations: usize,
    pub times: Vec<Duration>,
}

impl BenchResult {
    /// Minimum observed duration.
    pub fn min(&self) -> Duration {
        self.times.iter().copied().min().unwrap_or(Duration::ZERO)
    }

    /// Maximum observed duration.
    pub fn max(&self) -> Duration {
        self.times.iter().copied().max().unwrap_or(Duration::ZERO)
    }

    /// Arithmetic mean of all durations.
    pub fn mean(&self) -> Duration {
        if self.times.is_empty() {
            return Duration::ZERO;
        }
        let total: Duration = self.times.iter().sum();
        total / self.times.len() as u32
    }

    /// Median duration (50th percentile).
    pub fn median(&self) -> Duration {
        self.percentile(50.0)
    }

    /// Compute a percentile (0.0–100.0) from the sorted timing data.
    pub fn percentile(&self, p: f64) -> Duration {
        if self.times.is_empty() {
            return Duration::ZERO;
        }
        let mut sorted = self.times.clone();
        sorted.sort();
        let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        let idx = idx.min(sorted.len() - 1);
        sorted[idx]
    }

    /// Print a single-line summary with aligned columns.
    pub fn print_summary(&self) {
        println!(
            "  {:50} median={:>10.1?}  mean={:>10.1?}  min={:>10.1?}  max={:>10.1?}  p95={:>10.1?}  p99={:>10.1?}  (n={})",
            self.name,
            self.median(),
            self.mean(),
            self.min(),
            self.max(),
            self.percentile(95.0),
            self.percentile(99.0),
            self.iterations,
        );
    }

    /// Print a single-line summary with throughput in MiB/s or GiB/s.
    pub fn print_summary_with_throughput(&self, bytes: u64) {
        println!(
            "  {:50} median={:>10.1?}  mean={:>10.1?}  min={:>10.1?}  max={:>10.1?}  p95={:>10.1?}  {}  (n={})",
            self.name,
            self.median(),
            self.mean(),
            self.min(),
            self.max(),
            self.percentile(95.0),
            format_throughput(bytes, self.median()),
            self.iterations,
        );
    }
}

/// Format throughput as human-readable string (MiB/s or GiB/s).
pub fn format_throughput(bytes: u64, duration: Duration) -> String {
    if duration.is_zero() {
        return "[N/A]".to_string();
    }
    let bytes_per_sec = bytes as f64 / duration.as_secs_f64();
    let gib = 1024.0 * 1024.0 * 1024.0;
    let mib = 1024.0 * 1024.0;
    if bytes_per_sec >= gib {
        format!("[{:.1} GiB/s]", bytes_per_sec / gib)
    } else {
        format!("[{:.1} MiB/s]", bytes_per_sec / mib)
    }
}

/// Run a micro-benchmark.
///
/// Executes `warmup` unmeasured iterations of `f`, followed by `iterations`
/// measured iterations. Returns a [`BenchResult`] with per-iteration timings.
pub fn run_bench<F>(name: &str, warmup: usize, iterations: usize, mut f: F) -> BenchResult
where
    F: FnMut(),
{
    // Warm-up phase (unmeasured)
    for _ in 0..warmup {
        f();
    }

    // Measured phase
    let mut times = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        times.push(start.elapsed());
    }

    BenchResult {
        name: name.to_string(),
        iterations,
        times,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_result_statistics() {
        let result = BenchResult {
            name: "test".to_string(),
            iterations: 5,
            times: vec![
                Duration::from_micros(100),
                Duration::from_micros(200),
                Duration::from_micros(300),
                Duration::from_micros(400),
                Duration::from_micros(500),
            ],
        };

        assert_eq!(result.min(), Duration::from_micros(100));
        assert_eq!(result.max(), Duration::from_micros(500));
        assert_eq!(result.mean(), Duration::from_micros(300));
        assert_eq!(result.median(), Duration::from_micros(300));
    }

    #[test]
    fn test_bench_result_empty() {
        let result = BenchResult {
            name: "empty".to_string(),
            iterations: 0,
            times: vec![],
        };

        assert_eq!(result.min(), Duration::ZERO);
        assert_eq!(result.max(), Duration::ZERO);
        assert_eq!(result.mean(), Duration::ZERO);
        assert_eq!(result.median(), Duration::ZERO);
    }

    #[test]
    fn test_run_bench() {
        let mut counter = 0u32;
        let result = run_bench("increment", 2, 5, || {
            counter += 1;
        });

        // 2 warmup + 5 measured = 7 total calls
        assert_eq!(counter, 7);
        assert_eq!(result.iterations, 5);
        assert_eq!(result.times.len(), 5);
        assert_eq!(result.name, "increment");
    }

    #[test]
    fn test_percentile_single_element() {
        let result = BenchResult {
            name: "single".to_string(),
            iterations: 1,
            times: vec![Duration::from_micros(42)],
        };

        assert_eq!(result.percentile(0.0), Duration::from_micros(42));
        assert_eq!(result.percentile(50.0), Duration::from_micros(42));
        assert_eq!(result.percentile(100.0), Duration::from_micros(42));
    }

    #[test]
    fn test_format_throughput_zero_duration() {
        assert_eq!(format_throughput(1024, Duration::ZERO), "[N/A]");
    }

    #[test]
    fn test_format_throughput_gib() {
        // 2 GiB in 1 second → should report ~2.0 GiB/s
        let result = format_throughput(2 * 1024 * 1024 * 1024, Duration::from_secs(1));
        assert!(result.contains("GiB/s"), "expected GiB/s, got: {result}");
        assert!(result.contains("2.0"), "expected 2.0, got: {result}");
    }

    #[test]
    fn test_format_throughput_mib() {
        // 100 MiB in 1 second → should report ~100.0 MiB/s
        let result = format_throughput(100 * 1024 * 1024, Duration::from_secs(1));
        assert!(result.contains("MiB/s"), "expected MiB/s, got: {result}");
        assert!(result.contains("100.0"), "expected 100.0, got: {result}");
    }
}
