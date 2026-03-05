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

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

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

    /// Serialize this result to a [`serde_json::Value`].
    ///
    /// All timing values are expressed in microseconds (integer).
    ///
    /// # Returns
    ///
    /// A JSON object with fields: `name`, `median_us`, `mean_us`, `min_us`,
    /// `max_us`, `p95_us`, `p99_us`, `iterations`.
    pub fn to_json(&self) -> serde_json::Value {
        // as_micros() returns u128; JSON numbers top out at u64 (serde_json
        // does not implement PartialEq<u128>).  Benchmark durations never
        // exceed u64::MAX microseconds (~584,542 years), so the cast is safe.
        serde_json::json!({
            "name":       self.name,
            "median_us":  self.median().as_micros() as u64,
            "mean_us":    self.mean().as_micros() as u64,
            "min_us":     self.min().as_micros() as u64,
            "max_us":     self.max().as_micros() as u64,
            "p95_us":     self.percentile(95.0).as_micros() as u64,
            "p99_us":     self.percentile(99.0).as_micros() as u64,
            "iterations": self.iterations,
        })
    }
}

/// Collects benchmark results and optionally writes them to a JSON file.
///
/// # Usage
///
/// ```rust,no_run
/// use edgefirst_bench::{run_bench, BenchSuite};
///
/// let mut suite = BenchSuite::from_args();
/// let result = run_bench("my_op", 5, 100, || {});
/// result.print_summary();
/// suite.record(&result);
/// suite.finish();
/// ```
///
/// Pass `--json <path>` on the command line to write results to a file.
pub struct BenchSuite {
    /// Path to write JSON output, or `None` if not requested.
    json_path: Option<String>,
    /// Collected results in insertion order.
    results: Vec<serde_json::Value>,
}

impl BenchSuite {
    /// Build a `BenchSuite` by scanning `std::env::args()` for `--json <path>`.
    ///
    /// If `--json` is absent the suite is a no-op: `record` and `finish` do
    /// nothing observable beyond collecting data in memory.
    pub fn from_args() -> Self {
        let args: Vec<String> = std::env::args().collect();
        let json_path = args
            .windows(2)
            .find(|w| w[0] == "--json")
            .map(|w| w[1].clone());
        Self {
            json_path,
            results: Vec::new(),
        }
    }

    /// Add a benchmark result to the suite.
    pub fn record(&mut self, result: &BenchResult) {
        self.results.push(result.to_json());
    }

    /// Write collected results to the JSON file specified by `--json`, if any.
    ///
    /// The output format is:
    /// ```json
    /// {
    ///   "timestamp": "unix:1709553600",
    ///   "benchmarks": [ { "name": "...", "median_us": 123, ... } ]
    /// }
    /// ```
    ///
    /// Uses [`std::time::SystemTime`] for the timestamp; no external
    /// time-keeping dependencies are required.
    ///
    /// # Panics
    ///
    /// Panics if the file cannot be created or written.  Benchmark runners are
    /// short-lived processes — a hard failure is preferable to silently losing
    /// data.
    pub fn finish(&self) {
        let Some(path) = &self.json_path else {
            return;
        };

        let unix_secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();

        let output = serde_json::json!({
            "timestamp":  format!("unix:{unix_secs}"),
            "benchmarks": self.results,
        });

        let json_str =
            serde_json::to_string_pretty(&output).expect("failed to serialise benchmark results");
        std::fs::write(path, json_str).unwrap_or_else(|e| {
            panic!("failed to write benchmark JSON to {path}: {e}");
        });
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

    #[test]
    fn test_to_json_fields_present() {
        let result = BenchResult {
            name: "json_test".to_string(),
            iterations: 5,
            times: vec![
                Duration::from_micros(100),
                Duration::from_micros(200),
                Duration::from_micros(300),
                Duration::from_micros(400),
                Duration::from_micros(500),
            ],
        };

        let json = result.to_json();

        assert_eq!(json["name"], "json_test");
        assert_eq!(json["iterations"], 5);
        // median of [100,200,300,400,500] µs = 300 µs
        assert_eq!(json["median_us"], 300u64);
        // mean = 300 µs
        assert_eq!(json["mean_us"], 300u64);
        assert_eq!(json["min_us"], 100u64);
        assert_eq!(json["max_us"], 500u64);
        // p95 index = round(0.95 * 4) = 4 → 500 µs
        assert_eq!(json["p95_us"], 500u64);
        // p99 index = round(0.99 * 4) = 4 → 500 µs
        assert_eq!(json["p99_us"], 500u64);
    }

    #[test]
    fn test_to_json_empty() {
        let result = BenchResult {
            name: "empty".to_string(),
            iterations: 0,
            times: vec![],
        };

        let json = result.to_json();

        assert_eq!(json["name"], "empty");
        assert_eq!(json["iterations"], 0);
        assert_eq!(json["median_us"], 0u64);
        assert_eq!(json["min_us"], 0u64);
        assert_eq!(json["max_us"], 0u64);
    }

    #[test]
    fn test_bench_suite_record_and_finish_no_path() {
        // When no --json flag is present, finish() is a no-op.
        let mut suite = BenchSuite {
            json_path: None,
            results: Vec::new(),
        };
        let result = BenchResult {
            name: "noop".to_string(),
            iterations: 1,
            times: vec![Duration::from_micros(42)],
        };
        suite.record(&result);
        assert_eq!(suite.results.len(), 1);
        // finish() must not panic or create any file
        suite.finish();
    }

    #[test]
    fn test_bench_suite_finish_writes_file() {
        use std::fs;

        let tmp_path = std::env::temp_dir().join("edgefirst_bench_test_output.json");
        let path_str = tmp_path.to_string_lossy().to_string();

        let mut suite = BenchSuite {
            json_path: Some(path_str.clone()),
            results: Vec::new(),
        };

        let result = BenchResult {
            name: "write_test".to_string(),
            iterations: 3,
            times: vec![
                Duration::from_micros(10),
                Duration::from_micros(20),
                Duration::from_micros(30),
            ],
        };
        suite.record(&result);
        suite.finish();

        let raw = fs::read_to_string(&tmp_path).expect("JSON file should have been written");
        let parsed: serde_json::Value =
            serde_json::from_str(&raw).expect("output must be valid JSON");

        assert!(
            parsed["timestamp"]
                .as_str()
                .unwrap_or("")
                .starts_with("unix:"),
            "timestamp should start with 'unix:'"
        );

        let benchmarks = parsed["benchmarks"].as_array().expect("benchmarks array");
        assert_eq!(benchmarks.len(), 1);
        assert_eq!(benchmarks[0]["name"], "write_test");
        assert_eq!(benchmarks[0]["iterations"], 3);

        // Clean up
        let _ = fs::remove_file(&tmp_path);
    }
}
