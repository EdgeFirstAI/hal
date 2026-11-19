// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Logging for the conversions the auto chain moves past a backend for.

#![cfg_attr(not(any(target_os = "linux", feature = "opengl")), allow(dead_code))]

use crate::Error;
use edgefirst_tensor::{PixelFormat, TensorDyn};
use std::collections::HashSet;

/// The most shapes [`FallbackLog`] holds; it clears them when full.
const CAPACITY: usize = 1024;

/// A declining backend, the source format and size, and the destination format and size.
type Shape = (
    &'static str,
    Option<PixelFormat>,
    (usize, usize),
    Option<PixelFormat>,
    (usize, usize),
);

/// The conversion shapes a backend has declined, so each logs at info once and at debug after.
#[derive(Debug, Default)]
pub(crate) struct FallbackLog {
    seen: HashSet<Shape>,
}

impl FallbackLog {
    /// Logs that `backend` declined converting `src` into `dst` for `reason`: at info the first
    /// time for the shape, at debug afterwards. Returns the level it logged at.
    pub(crate) fn note(
        &mut self,
        backend: &'static str,
        src: &TensorDyn,
        dst: &TensorDyn,
        reason: &Error,
    ) -> log::Level {
        let size = |t: &TensorDyn| (t.width().unwrap_or(0), t.height().unwrap_or(0));
        let (src_fmt, (sw, sh), dst_fmt, (dw, dh)) =
            (src.format(), size(src), dst.format(), size(dst));
        let shape = (backend, src_fmt, (sw, sh), dst_fmt, (dw, dh));
        if self.seen.len() >= CAPACITY && !self.seen.contains(&shape) {
            self.seen.clear();
        }
        let level = if self.seen.insert(shape) {
            log::Level::Info
        } else {
            log::Level::Debug
        };
        let name =
            |f: Option<PixelFormat>| f.map_or_else(|| "tensor".to_owned(), |f| format!("{f:?}"));
        log::log!(
            level,
            "convert: {backend} declined {} {sw}x{sh} -> {} {dw}x{dh}: {reason}",
            name(src_fmt),
            name(dst_fmt),
        );
        level
    }

    /// The number of shapes held.
    #[cfg(test)]
    fn seen_len(&self) -> usize {
        self.seen.len()
    }
}

#[cfg(test)]
mod tests {
    use super::FallbackLog;
    use crate::Error;
    use edgefirst_tensor::{CpuAccess, DType, PixelFormat, TensorDyn, TensorMemory};

    fn image(w: usize, h: usize, fmt: PixelFormat) -> TensorDyn {
        TensorDyn::image(
            w,
            h,
            fmt,
            DType::U8,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .unwrap()
    }

    #[test]
    fn fallback_log_is_info_once_per_shape_then_debug() {
        let mut log = FallbackLog::default();
        let reason = Error::NotSupported("declined".into());
        let src = image(64, 48, PixelFormat::Rgba);
        let dst = image(64, 48, PixelFormat::Nv12);
        let other = image(32, 24, PixelFormat::Nv12);
        assert_eq!(log.note("opengl", &src, &dst, &reason), log::Level::Info);
        assert_eq!(log.note("opengl", &src, &dst, &reason), log::Level::Debug);
        assert_eq!(log.note("g2d", &src, &dst, &reason), log::Level::Info);
        assert_eq!(log.note("opengl", &src, &other, &reason), log::Level::Info);
        assert_eq!(log.note("opengl", &src, &other, &reason), log::Level::Debug);

        for w in 1..=1025 {
            log.note("opengl", &src, &image(w, 1, PixelFormat::Grey), &reason);
            assert!(
                log.seen_len() <= 1024,
                "{} shapes after {w}",
                log.seen_len()
            );
        }
    }
}
