// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::resources::EglImage;

/// Selects which EGLImage cache to use.
#[derive(Debug)]
pub(super) enum CacheKind {
    Src,
    Dst,
}

/// A cached EGLImage with a weak reference to the source tensor's guard.
pub(super) struct CachedEglImage {
    pub(super) egl_image: EglImage,
    /// Weak reference to the source Tensor's BufferIdentity guard.
    pub(super) guard: std::sync::Weak<()>,
    /// Optional GL renderbuffer backed by this EGLImage (used by direct RGB path).
    pub(super) renderbuffer: Option<u32>,
    /// Monotonic access counter for LRU eviction.
    pub(super) last_used: u64,
}

/// EGLImage cache owned by GLProcessorST.
///
/// Uses a HashMap with a monotonic counter for LRU eviction: each access
/// updates the entry's `last_used` timestamp, and eviction removes the entry
/// with the smallest `last_used` value.
pub(super) struct EglImageCache {
    pub(super) entries: std::collections::HashMap<u64, CachedEglImage>,
    pub(super) capacity: usize,
    pub(super) hits: u64,
    pub(super) misses: u64,
    /// Monotonic counter incremented on each access for LRU tracking.
    pub(super) access_counter: u64,
}

impl EglImageCache {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            entries: std::collections::HashMap::with_capacity(capacity),
            capacity,
            hits: 0,
            misses: 0,
            access_counter: 0,
        }
    }

    /// Allocate a new LRU timestamp.
    pub(super) fn next_timestamp(&mut self) -> u64 {
        self.access_counter += 1;
        self.access_counter
    }

    /// Evict the least recently used entry.
    pub(super) fn evict_lru(&mut self) {
        if let Some((&evict_id, _)) = self.entries.iter().min_by_key(|(_, entry)| entry.last_used) {
            if let Some(evicted) = self.entries.remove(&evict_id) {
                if let Some(rbo) = evicted.renderbuffer {
                    unsafe { gls::gl::DeleteRenderbuffers(1, &rbo) };
                }
            }
        }
    }

    /// Sweep dead entries (tensor dropped, Weak is dead).
    pub(super) fn sweep(&mut self) {
        let before = self.entries.len();
        self.entries.retain(|_id, entry| {
            let alive = entry.guard.upgrade().is_some();
            if !alive {
                if let Some(rbo) = entry.renderbuffer {
                    unsafe { gls::gl::DeleteRenderbuffers(1, &rbo) };
                }
            }
            alive
        });
        let swept = before - self.entries.len();
        if swept > 0 {
            log::debug!("EglImageCache: swept {swept} dead entries");
        }
    }
}

impl Drop for EglImageCache {
    fn drop(&mut self) {
        for entry in self.entries.values() {
            if let Some(rbo) = entry.renderbuffer {
                unsafe { gls::gl::DeleteRenderbuffers(1, &rbo) };
            }
        }
        log::debug!(
            "EglImageCache stats: {} hits, {} misses, {} entries remaining",
            self.hits,
            self.misses,
            self.entries.len()
        );
    }
}
