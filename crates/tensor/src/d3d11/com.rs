// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//! COM error mapping and UTF-16 encoding shared by the D3D11 modules.

/// Turns a `windows::core::Result` into this crate's error, with the call
/// name and the HRESULT attached.
pub(crate) fn hr<T>(what: &'static str, result: windows::core::Result<T>) -> crate::Result<T> {
    result.map_err(|e| {
        crate::Error::IoError(std::io::Error::other(format!(
            "{what}: {message} (HRESULT 0x{code:08X})",
            message = e.message(),
            code = e.code().0 as u32,
        )))
    })
}

/// Encodes `name` as UTF-16 with a trailing NUL.
pub(crate) fn wide(name: &str) -> Vec<u16> {
    name.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hr_keeps_context_and_code() {
        let err = windows::core::Error::from_hresult(windows::Win32::Foundation::E_INVALIDARG);
        let result: crate::Result<()> = hr("create texture", Err(err));
        let message = result.unwrap_err().to_string();
        assert!(message.contains("create texture"), "{message}");
        assert!(message.contains("0x80070057"), "{message}");

        assert_eq!(hr("x", Ok(7)).unwrap(), 7);
    }

    #[test]
    fn wide_is_nul_terminated() {
        assert_eq!(wide("abc"), vec![97, 98, 99, 0]);
    }
}
