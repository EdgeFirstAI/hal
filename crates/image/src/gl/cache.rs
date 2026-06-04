// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::resources::EglImage;

/// Selects which EGLImage cache to use.
#[derive(Debug, PartialEq)]
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
/// Cache key: `(luma_id, chroma_id, plane_offset)`. For single-plane images,
/// `chroma_id` is `None`. `plane_offset` distinguishes sub-region views that
/// share one buffer (same identities) but start at different byte offsets —
/// without it, N offset-distinct views of one DMA-BUF collide on the first
/// (offset-0) EGLImage and every view renders/samples the base region.
pub(super) type EglCacheKey = (u64, Option<u64>, usize);

pub(super) struct EglImageCache {
    pub(super) entries: std::collections::HashMap<EglCacheKey, CachedEglImage>,
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

    /// Evict the least recently used entry. Returns `true` if an entry was evicted.
    pub(super) fn evict_lru(&mut self) -> bool {
        if let Some((&evict_id, _)) = self.entries.iter().min_by_key(|(_, entry)| entry.last_used) {
            let evicted = self.entries.remove(&evict_id).expect("key just found");
            if let Some(rbo) = evicted.renderbuffer {
                unsafe { gls::gl::DeleteRenderbuffers(1, &rbo) };
            }
            return true;
        }
        false
    }

    /// Sweep dead entries (tensor dropped, Weak is dead).
    /// Returns `true` if any entries were removed.
    pub(super) fn sweep(&mut self) -> bool {
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
        swept > 0
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

#[cfg(test)]
mod tests {
    use super::EglCacheKey;
    use std::collections::HashMap;

    #[test]
    fn cache_key_distinguishes_plane_offset() {
        // Root-cause regression guard (no GPU needed): two sub-region views of
        // one DMA-BUF share buffer identities but start at different byte
        // offsets. The cache key must keep them as DISTINCT entries — otherwise
        // the first (offset-0) EGLImage is reused for every offset and the GPU
        // renders/samples the base region.
        let mut map: HashMap<EglCacheKey, u32> = HashMap::new();
        let base: EglCacheKey = (0xABCD, None, 0);
        let at_offset: EglCacheKey = (0xABCD, None, 4096);
        map.insert(base, 1);
        map.insert(at_offset, 2);
        assert_eq!(map.len(), 2, "offset-distinct views must not collide");
        assert_eq!(map.get(&base), Some(&1));
        assert_eq!(map.get(&at_offset), Some(&2));

        // Identical keys still collide (a genuine cache hit), as before.
        map.insert(base, 3);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&base), Some(&3));
    }
}
