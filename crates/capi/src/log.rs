// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! C-compatible logging API.
//!
//! Provides two ways to initialise HAL logging from C code:
//!
//! - **File-based** (`hal_log_init_file`): writes formatted log lines to a
//!   `FILE*` (typically `stderr`).
//! - **Callback-based** (`hal_log_init_callback`): invokes a user-provided
//!   C function for each log record, allowing integration with GStreamer,
//!   syslog, or other logging frameworks.
//!
//! Only one call to either init function is honoured per process; subsequent
//! calls return `-1` with `errno = EALREADY`.

use libc::{c_char, c_int, c_void};
use std::ffi::CString;
use std::sync::Once;

/// Log severity level.
///
/// Maps 1:1 to Rust `log::Level`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalLogLevel {
    /// Errors — unrecoverable or unexpected failures.
    Error = 1,
    /// Warnings — degraded but recoverable situations.
    Warn = 2,
    /// Informational messages.
    Info = 3,
    /// Debug-level detail.
    Debug = 4,
    /// Fine-grained trace output.
    Trace = 5,
}

impl HalLogLevel {
    fn from_log_level(level: log::Level) -> Self {
        match level {
            log::Level::Error => HalLogLevel::Error,
            log::Level::Warn => HalLogLevel::Warn,
            log::Level::Info => HalLogLevel::Info,
            log::Level::Debug => HalLogLevel::Debug,
            log::Level::Trace => HalLogLevel::Trace,
        }
    }

    fn to_log_level_filter(self) -> log::LevelFilter {
        match self {
            HalLogLevel::Error => log::LevelFilter::Error,
            HalLogLevel::Warn => log::LevelFilter::Warn,
            HalLogLevel::Info => log::LevelFilter::Info,
            HalLogLevel::Debug => log::LevelFilter::Debug,
            HalLogLevel::Trace => log::LevelFilter::Trace,
        }
    }
}

/// Callback function type for log messages.
///
/// @param level     Severity level of the message
/// @param target    Module path that produced the message (null-terminated)
/// @param message   Log message text (null-terminated)
/// @param userdata  Opaque pointer passed to `hal_log_init_callback`
pub type HalLogCallback = Option<
    unsafe extern "C" fn(
        level: HalLogLevel,
        target: *const c_char,
        message: *const c_char,
        userdata: *mut c_void,
    ),
>;

// ── Internal helpers ────────────────────────────────────────────────────

/// Convert a Rust string to a CString, replacing interior null bytes with
/// the Unicode replacement character so the message is never silently dropped.
fn to_cstring(s: &str) -> CString {
    CString::new(s.replace('\0', "\u{FFFD}")).expect("no nulls after replace")
}

// ── Internal logger implementation ──────────────────────────────────────

enum LoggerKind {
    File {
        stream: *mut libc::FILE,
    },
    Callback {
        cb: unsafe extern "C" fn(HalLogLevel, *const c_char, *const c_char, *mut c_void),
        userdata: *mut c_void,
    },
}

// SAFETY: The FILE* and callback+userdata are expected to be thread-safe by
// the caller's contract (documented in the public API).
unsafe impl Send for LoggerKind {}
unsafe impl Sync for LoggerKind {}

struct HalLogger {
    kind: LoggerKind,
    max_level: log::LevelFilter,
}

// SAFETY: HalLogger delegates to LoggerKind which is Send+Sync.
unsafe impl Send for HalLogger {}
unsafe impl Sync for HalLogger {}

impl log::Log for HalLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= self.max_level
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let level = HalLogLevel::from_log_level(record.level());

        match &self.kind {
            LoggerKind::File { stream } => {
                let msg = format!(
                    "[{}] {}: {}\n",
                    record.level(),
                    record.target(),
                    record.args()
                );
                let c_msg = to_cstring(&msg);
                unsafe {
                    libc::fputs(c_msg.as_ptr(), *stream);
                    libc::fflush(*stream);
                }
            }
            LoggerKind::Callback { cb, userdata } => {
                let target = to_cstring(record.target());
                let message = to_cstring(&format!("{}", record.args()));
                unsafe {
                    cb(level, target.as_ptr(), message.as_ptr(), *userdata);
                }
            }
        }
    }

    fn flush(&self) {
        if let LoggerKind::File { stream } = &self.kind {
            unsafe {
                libc::fflush(*stream);
            }
        }
    }
}

/// Guard ensuring at-most-once initialisation.
static INIT: Once = Once::new();

/// Install and activate the logger. Returns 0 on success, -1 if already set.
///
/// Uses `Box::leak` to create a `&'static HalLogger` and registers it with
/// `log::set_logger`. The `Once` guard prevents double initialisation.
/// If `log::set_logger` fails (another crate already set a global logger),
/// the leaked allocation is harmless and we return an error.
fn install_logger(logger: HalLogger) -> c_int {
    let mut result: c_int = -1;
    INIT.call_once(|| {
        let max_level = logger.max_level;
        let leaked: &'static HalLogger = Box::leak(Box::new(logger));
        if log::set_logger(leaked).is_err() {
            // Another crate already called log::set_logger — we leak one
            // small allocation but never panic across FFI.
            return;
        }
        log::set_max_level(max_level);
        result = 0;
    });
    if result != 0 {
        errno::set_errno(errno::Errno(libc::EALREADY));
    }
    result
}

// ── Public C API ────────────────────────────────────────────────────────

/// Initialise HAL logging to a `FILE*` stream.
///
/// Writes `[LEVEL] target: message` lines to the given stream, typically
/// `stderr`. Only the first successful call takes effect; subsequent calls
/// return `-1` with `errno = EALREADY`.
///
/// **Thread safety**: The `stream` must remain valid for the lifetime of the
/// process. On glibc/musl, `stderr` and other standard streams are internally
/// locked per-call, so individual lines will not be corrupted. However, two
/// log lines from different threads may interleave. If fully ordered output
/// is required, use `hal_log_init_callback` with an application-side lock.
///
/// @param stream     Open `FILE*` to write log output to (e.g. `stderr`)
/// @param max_level  Maximum log level to emit (messages above this are dropped)
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL:  `stream` is NULL
/// - EALREADY: logging has already been initialised
///
/// @par Example
/// @code{.c}
/// hal_log_init_file(stderr, HAL_LOG_LEVEL_DEBUG);
/// @endcode
#[no_mangle]
pub unsafe extern "C" fn hal_log_init_file(
    stream: *mut libc::FILE,
    max_level: HalLogLevel,
) -> c_int {
    if stream.is_null() {
        errno::set_errno(errno::Errno(libc::EINVAL));
        return -1;
    }
    install_logger(HalLogger {
        kind: LoggerKind::File { stream },
        max_level: max_level.to_log_level_filter(),
    })
}

/// Initialise HAL logging with a user-provided callback.
///
/// Each log record is forwarded to `cb` with the level, target module name,
/// formatted message, and the opaque `userdata` pointer. This allows routing
/// HAL log output into GStreamer, syslog, or any other logging framework.
///
/// Only the first successful call takes effect; subsequent calls return `-1`
/// with `errno = EALREADY`.
///
/// @param cb         Callback function invoked for each log message
/// @param userdata   Opaque pointer forwarded to the callback (may be NULL)
/// @param max_level  Maximum log level to emit
/// @return 0 on success, -1 on error
/// @par Errors (errno):
/// - EINVAL:  `cb` is NULL
/// - EALREADY: logging has already been initialised
///
/// @par Example
/// @code{.c}
/// void my_logger(hal_log_level level, const char* target,
///                const char* message, void* userdata) {
///     fprintf(stderr, "[%d] %s: %s\n", level, target, message);
/// }
/// hal_log_init_callback(my_logger, NULL, HAL_LOG_LEVEL_TRACE);
/// @endcode
#[no_mangle]
pub unsafe extern "C" fn hal_log_init_callback(
    cb: HalLogCallback,
    userdata: *mut c_void,
    max_level: HalLogLevel,
) -> c_int {
    let cb = match cb {
        Some(f) => f,
        None => {
            errno::set_errno(errno::Errno(libc::EINVAL));
            return -1;
        }
    };
    install_logger(HalLogger {
        kind: LoggerKind::Callback { cb, userdata },
        max_level: max_level.to_log_level_filter(),
    })
}
