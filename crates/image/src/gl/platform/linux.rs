// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Linux implementation of [`GlPlatform`]: EGL display bring-up via
//! GBM/PlatformDevice/Default and DMA-BUF buffer import.
//!
//! Pure delegation — the actual machinery predates this seam and lives in
//! [`super::super::context`] (display/context bring-up, transfer-backend
//! probe) and [`super::super::dma_import`] (DMA-BUF attribute assembly).
//! This file only binds it to the cross-platform contract; it must stay
//! free of logic so the platform trait remains the single source of truth
//! for *what* a platform provides, with `context.rs`/`dma_import.rs`
//! owning *how* Linux provides it.

use super::super::context::GlContext;
use super::super::resources::EglImage;
use super::super::EglDisplayKind;
use super::GlPlatform;

/// Marker type: Linux EGL + DMA-BUF platform. Stateless — all state
/// lives in the [`GlContext`] created by [`GlPlatform::init_display`].
pub(crate) struct LinuxEgl;

impl GlPlatform for LinuxEgl {
    type Display = GlContext;
    type Import = EglImage;

    fn init_display(kind: Option<EglDisplayKind>) -> crate::Result<GlContext> {
        GlContext::new(kind)
    }
}
