// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Platform-specific helpers for the OpenGL backend.
//!
//! Linux uses GBM + EGL + DMA-BUF; macOS uses ANGLE + IOSurface. The two
//! platforms have very different EGL bring-up and buffer import flows —
//! the macOS path goes through `eglGetPlatformDisplayEXT` (ANGLE Metal
//! display) and `eglCreatePbufferFromClientBuffer` (IOSurface import),
//! while Linux goes through the standard EGL display path and
//! `eglCreateImageKHR` (DMA-BUF import).
//!
//! The Linux platform helpers (display bring-up, GBM device handling,
//! DMA-BUF import) live in [`super::context`] and [`super::dma_import`].
//! Those modules predate this directory and are not yet reorganised. The
//! macOS helpers live in [`macos`] and are called directly by
//! [`super::macos_processor`].
//!
//! There is no cross-platform trait. An earlier draft of this branch
//! defined a `GlPlatform` trait with `import_buffer` / `bind_as_*`
//! methods, but the trait was never wired through `processor.rs` and
//! `macos_processor.rs` bypassed it entirely. The trait has been removed;
//! a future cross-platform refactor can introduce one once Linux and
//! macOS actually share a processor implementation.

#[cfg(target_os = "macos")]
pub(super) mod macos;
