// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Simple micro-benchmark timing harness for GPU probe measurements.

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
            "  {:40} median={:>10.1?}  mean={:>10.1?}  min={:>10.1?}  max={:>10.1?}  p95={:>10.1?}  p99={:>10.1?}  (n={})",
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
}
