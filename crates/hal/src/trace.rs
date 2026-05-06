// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Trace capture for performance analysis.
//!
//! Provides a simple start/stop API for capturing [`tracing`]-based spans
//! emitted by HAL crates into Chrome JSON trace files viewable at
//! <https://ui.perfetto.dev/>.
//!
//! # Design
//!
//! The HAL library crates (`edgefirst-decoder`, `edgefirst-image`) emit
//! [`tracing::trace_span!`] spans on hot paths. These have near-zero overhead
//! when no subscriber is active (a single relaxed atomic load per span site).
//!
//! This module installs a **process-wide subscriber** consisting of a Chrome
//! trace layer writing spans to a JSON file for Perfetto. Existing `log::*`
//! output (via `env_logger`) continues independently to stderr.
//!
//! The subscriber is installed once on the first call to [`start_tracing`].
//! Only one trace capture session is supported per process lifetime (this is
//! a limitation of Rust's global subscriber model and is acceptable for
//! profiling workflows where a single trace per run is the norm).
//!
//! # Usage from Rust
//!
//! ```no_run
//! # #[cfg(feature = "tracing")]
//! # {
//! use edgefirst_hal::trace::{start_tracing, stop_tracing};
//!
//! start_tracing("/tmp/trace.json").expect("start tracing");
//! // ... run inference pipeline ...
//! stop_tracing(); // flushes and closes the trace file
//! # }
//! ```
//!
//! # Usage from Python
//!
//! ```python
//! import edgefirst_hal as hal
//!
//! with hal.Tracing("/tmp/trace.json"):
//!     # ... run inference ...
//!     pass
//! # trace file is flushed on __exit__
//! ```
//!
//! # Usage from C
//!
//! ```c
//! #include "edgefirst_hal.h"
//! hal_start_tracing("/tmp/trace.json");
//! // ... run inference ...
//! hal_stop_tracing(); // flushes trace file
//! ```

use std::sync::Mutex;

use tracing_chrome::FlushGuard;
use tracing_subscriber::prelude::*;

/// Global flush guard for the active trace session.
static GUARD: Mutex<Option<FlushGuard>> = Mutex::new(None);

/// Errors from tracing operations.
#[derive(Debug)]
pub enum TracingError {
    /// A trace capture session is already active.
    AlreadyActive,
    /// Failed to install the global subscriber (another was already set).
    SubscriberInstallFailed(String),
}

impl std::fmt::Display for TracingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AlreadyActive => write!(f, "trace capture already active"),
            Self::SubscriberInstallFailed(e) => {
                write!(f, "failed to install tracing subscriber: {e}")
            }
        }
    }
}

impl std::error::Error for TracingError {}

/// Start trace capture, writing Chrome JSON to `path`.
///
/// Installs a global tracing subscriber (chrome layer only) on first call.
/// The trace file is created immediately. All `tracing::trace_span!` spans
/// emitted by HAL crates will be recorded until [`stop_tracing`] is called.
///
/// Only one session per process lifetime is supported.
///
/// # Errors
///
/// Returns [`TracingError::AlreadyActive`] if already capturing.
/// Returns [`TracingError::SubscriberInstallFailed`] if another subscriber
/// was previously installed by user code.
pub fn start_tracing(path: &str) -> Result<(), TracingError> {
    let mut lock = GUARD.lock().unwrap();
    if lock.is_some() {
        return Err(TracingError::AlreadyActive);
    }

    // Build chrome layer writing to the specified file.
    let (chrome_layer, guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file(path)
        .include_args(true)
        .build();

    // Install only the chrome layer. Existing log::* output continues through
    // env_logger to stderr independently — no conflict.
    let subscriber = tracing_subscriber::registry().with(chrome_layer);

    tracing::subscriber::set_global_default(subscriber)
        .map_err(|e| TracingError::SubscriberInstallFailed(e.to_string()))?;

    *lock = Some(guard);
    Ok(())
}

/// Stop trace capture, flushing all buffered spans to the output file.
///
/// No-op if no session is active. After this call the trace file is complete
/// and can be loaded into <https://ui.perfetto.dev/>.
pub fn stop_tracing() {
    let mut lock = GUARD.lock().unwrap();
    // Dropping the FlushGuard flushes remaining spans and closes the file.
    lock.take();
}

/// Returns `true` if a trace capture session is currently active.
pub fn is_tracing_active() -> bool {
    GUARD.lock().unwrap().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    // Single test because the global subscriber is per-process lifetime.
    #[test]
    fn test_trace_lifecycle() {
        let dir = std::env::temp_dir();
        let path = dir.join("hal_test_trace_lifecycle.json");
        let path_str = path.to_str().unwrap();

        // Clean up any previous test artifact
        let _ = std::fs::remove_file(&path);

        assert!(!is_tracing_active());

        // First start should succeed
        start_tracing(path_str).expect("start_tracing should succeed");
        assert!(is_tracing_active());

        // Second start while active should fail with AlreadyActive
        let err = start_tracing(path_str).unwrap_err();
        assert!(
            matches!(err, TracingError::AlreadyActive),
            "expected AlreadyActive, got: {err:?}"
        );

        // Emit a span to ensure the file gets content
        {
            let _span = tracing::trace_span!("test_span", key = "value").entered();
        }

        // Stop should deactivate
        stop_tracing();
        assert!(!is_tracing_active());

        // Trace file should exist with content
        assert!(Path::new(path_str).exists());
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(!content.is_empty(), "trace file should not be empty");

        // Third start fails because global subscriber is already installed
        let err = start_tracing(path_str).unwrap_err();
        assert!(
            matches!(err, TracingError::SubscriberInstallFailed(_)),
            "expected SubscriberInstallFailed, got: {err:?}"
        );

        // Clean up
        let _ = std::fs::remove_file(&path);
    }
}
