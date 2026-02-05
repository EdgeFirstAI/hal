// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Error handling utilities for the C API.
//!
//! This module provides errno-based error reporting following POSIX conventions.

use errno::{Errno, set_errno};

/// Set errno and return -1 for functions returning int
#[inline]
pub fn set_error(code: i32) -> i32 {
    set_errno(Errno(code));
    -1
}

/// Set errno and return null for functions returning pointers
#[inline]
pub fn set_error_null<T>(code: i32) -> *mut T {
    set_errno(Errno(code));
    std::ptr::null_mut()
}

/// Check if a pointer is null, set EINVAL and return -1 if so
#[macro_export]
macro_rules! check_null {
    ($ptr:expr) => {
        if $ptr.is_null() {
            return $crate::error::set_error(libc::EINVAL);
        }
    };
    ($ptr:expr, $($rest:expr),+) => {
        $crate::check_null!($ptr);
        $crate::check_null!($($rest),+);
    };
}

/// Check if a pointer is null, set EINVAL and return NULL if so
#[macro_export]
macro_rules! check_null_ret_null {
    ($ptr:expr) => {
        if $ptr.is_null() {
            return $crate::error::set_error_null(libc::EINVAL);
        }
    };
    ($ptr:expr, $($rest:expr),+) => {
        $crate::check_null_ret_null!($ptr);
        $crate::check_null_ret_null!($($rest),+);
    };
}

/// Convert a Rust Result to a C return value, setting errno on error
#[macro_export]
macro_rules! try_or_errno {
    ($expr:expr, $errno:expr) => {
        match $expr {
            Ok(val) => val,
            Err(_) => return $crate::error::set_error($errno),
        }
    };
}

/// Convert a Rust Result to a C pointer, setting errno and returning NULL on error
#[macro_export]
macro_rules! try_or_null {
    ($expr:expr, $errno:expr) => {
        match $expr {
            Ok(val) => val,
            Err(_) => return $crate::error::set_error_null($errno),
        }
    };
}

/// Convert a C string to a Rust &str, returning the provided default if NULL or invalid UTF-8
#[inline]
pub unsafe fn c_str_to_str(ptr: *const libc::c_char, default: &str) -> &str {
    if ptr.is_null() {
        return default;
    }
    match std::ffi::CStr::from_ptr(ptr).to_str() {
        Ok(s) => s,
        Err(_) => default,
    }
}

/// Convert a C string to an Option<&str>, returning None if NULL
#[inline]
pub unsafe fn c_str_to_option<'a>(ptr: *const libc::c_char) -> Option<&'a str> {
    if ptr.is_null() {
        return None;
    }
    std::ffi::CStr::from_ptr(ptr).to_str().ok()
}

/// Allocate a C string from a Rust string, returning NULL on allocation failure
#[inline]
pub fn str_to_c_string(s: &str) -> *mut libc::c_char {
    match std::ffi::CString::new(s) {
        Ok(cstr) => cstr.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_error() {
        let result = set_error(libc::EINVAL);
        assert_eq!(result, -1);
        assert_eq!(errno::errno().0, libc::EINVAL);
    }

    #[test]
    fn test_set_error_null() {
        let result: *mut i32 = set_error_null(libc::ENOMEM);
        assert!(result.is_null());
        assert_eq!(errno::errno().0, libc::ENOMEM);
    }
}
