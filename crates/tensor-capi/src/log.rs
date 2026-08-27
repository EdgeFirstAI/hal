// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Process-wide logging from C: a `FILE*` or a callback.

use std::ffi::{c_char, c_int, c_void, CString};
use std::sync::Once;

/// Log severity. Maps 1:1 to Rust `log::Level`.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EfLogLevel {
    Error = 1,
    Warn = 2,
    Info = 3,
    Debug = 4,
    Trace = 5,
}

impl EfLogLevel {
    fn from_code(code: u32) -> Option<Self> {
        match code {
            1 => Some(Self::Error),
            2 => Some(Self::Warn),
            3 => Some(Self::Info),
            4 => Some(Self::Debug),
            5 => Some(Self::Trace),
            _ => None,
        }
    }

    fn from_log_level(level: log::Level) -> Self {
        match level {
            log::Level::Error => Self::Error,
            log::Level::Warn => Self::Warn,
            log::Level::Info => Self::Info,
            log::Level::Debug => Self::Debug,
            log::Level::Trace => Self::Trace,
        }
    }

    fn to_log_level_filter(self) -> log::LevelFilter {
        match self {
            Self::Error => log::LevelFilter::Error,
            Self::Warn => log::LevelFilter::Warn,
            Self::Info => log::LevelFilter::Info,
            Self::Debug => log::LevelFilter::Debug,
            Self::Trace => log::LevelFilter::Trace,
        }
    }
}

/// Callback invoked for each log record.
pub type EfLogCallback = Option<
    unsafe extern "C" fn(
        level: EfLogLevel,
        target: *const c_char,
        message: *const c_char,
        userdata: *mut c_void,
    ),
>;

fn to_cstring(s: &str) -> CString {
    CString::new(s.replace('\0', "\u{FFFD}")).unwrap_or_default()
}

enum LoggerKind {
    File {
        stream: *mut libc::FILE,
    },
    Callback {
        cb: unsafe extern "C" fn(EfLogLevel, *const c_char, *const c_char, *mut c_void),
        userdata: *mut c_void,
    },
}

// SAFETY: the caller keeps the FILE* / callback valid for the process life.
unsafe impl Send for LoggerKind {}
unsafe impl Sync for LoggerKind {}

struct EfLogger {
    kind: LoggerKind,
    max_level: log::LevelFilter,
}

unsafe impl Send for EfLogger {}
unsafe impl Sync for EfLogger {}

impl log::Log for EfLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= self.max_level
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }
        let level = EfLogLevel::from_log_level(record.level());
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

static INIT: Once = Once::new();

fn install_logger(logger: EfLogger) -> c_int {
    let mut result: c_int = -1;
    INIT.call_once(|| {
        let max_level = logger.max_level;
        let leaked: &'static EfLogger = Box::leak(Box::new(logger));
        if log::set_logger(leaked).is_err() {
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

/// Initialise logging to a `FILE*`. First successful call wins (`EALREADY` after).
///
/// # Safety
/// `stream` must remain valid for the process lifetime.
#[no_mangle]
pub unsafe extern "C" fn ef_log_init_file(stream: *mut libc::FILE, max_level: u32) -> c_int {
    if stream.is_null() {
        errno::set_errno(errno::Errno(libc::EINVAL));
        return -1;
    }
    let Some(level) = EfLogLevel::from_code(max_level) else {
        errno::set_errno(errno::Errno(libc::EINVAL));
        return -1;
    };
    install_logger(EfLogger {
        kind: LoggerKind::File { stream },
        max_level: level.to_log_level_filter(),
    })
}

/// Initialise logging with a callback. First successful call wins.
///
/// # Safety
/// `cb` must remain valid for the process lifetime.
#[no_mangle]
pub unsafe extern "C" fn ef_log_init_callback(
    cb: EfLogCallback,
    userdata: *mut c_void,
    max_level: u32,
) -> c_int {
    let Some(cb) = cb else {
        errno::set_errno(errno::Errno(libc::EINVAL));
        return -1;
    };
    let Some(level) = EfLogLevel::from_code(max_level) else {
        errno::set_errno(errno::Errno(libc::EINVAL));
        return -1;
    };
    install_logger(EfLogger {
        kind: LoggerKind::Callback { cb, userdata },
        max_level: level.to_log_level_filter(),
    })
}
