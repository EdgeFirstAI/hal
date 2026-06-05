// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use super::resources::EglImage;
use edgefirst_tensor::{PixelFormat, Tensor, TensorTrait};

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
/// Identity + geometry that uniquely determine an imported EGLImage.
///
/// `luma_id` / `chroma_id` are the buffer identities; `plane_offset`
/// distinguishes sub-region views that share one buffer but start at different
/// byte offsets (without it, N offset-distinct views of one DMA-BUF collide on
/// the first EGLImage and every view renders/samples the base region).
///
/// `width` / `height` / `row_stride` / `format` capture the geometry the
/// EGLImage was imported with. A pooled buffer reused at a different size via
/// `Tensor::configure_image` (e.g. a 128-wide pool decoding a 96-wide image)
/// keeps the same identities and `plane_offset`, but needs a fresh import: the
/// import's pitch/dimensions/fourcc all derive from these fields. Omitting them
/// reuses the stale-geometry EGLImage and the GPU samples the buffer at the
/// wrong pitch — deterministically wrong single-threaded, nondeterministic in
/// parallel, correct only on a heap source (which never takes this path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct EglCacheKey {
    pub(super) luma_id: u64,
    pub(super) chroma_id: Option<u64>,
    pub(super) plane_offset: usize,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) row_stride: usize,
    pub(super) format: PixelFormat,
}

impl EglCacheKey {
    /// Build the cache key from a tensor and the format it will be imported as.
    /// Every construction site MUST go through this so the key used to insert
    /// an EGLImage matches the key used to look it up and to gate the texture
    /// binding-skip.
    pub(super) fn from_tensor<T>(img: &Tensor<T>, format: PixelFormat) -> Self
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync,
    {
        Self {
            luma_id: img.buffer_identity().id(),
            chroma_id: img.chroma().map(|t| t.buffer_identity().id()),
            plane_offset: img.plane_offset().unwrap_or(0),
            width: img.width().unwrap_or(0),
            height: img.height().unwrap_or(0),
            row_stride: img.effective_row_stride().unwrap_or(0),
            format,
        }
    }
}

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
    use edgefirst_tensor::PixelFormat;
    use std::collections::HashMap;

    fn key(
        luma_id: u64,
        plane_offset: usize,
        width: usize,
        height: usize,
        row_stride: usize,
        format: PixelFormat,
    ) -> EglCacheKey {
        EglCacheKey {
            luma_id,
            chroma_id: None,
            plane_offset,
            width,
            height,
            row_stride,
            format,
        }
    }

    #[test]
    fn cache_key_distinguishes_plane_offset() {
        // Root-cause regression guard (no GPU needed): two sub-region views of
        // one DMA-BUF share buffer identities but start at different byte
        // offsets. The cache key must keep them as DISTINCT entries — otherwise
        // the first (offset-0) EGLImage is reused for every offset and the GPU
        // renders/samples the base region.
        let mut map: HashMap<EglCacheKey, u32> = HashMap::new();
        let base = key(0xABCD, 0, 64, 64, 64, PixelFormat::Grey);
        let at_offset = key(0xABCD, 4096, 64, 64, 64, PixelFormat::Grey);
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

    #[test]
    fn cache_key_distinguishes_geometry() {
        // Root-cause regression guard for the pool-recycle bug: ONE buffer
        // (same luma_id + offset) reconfigured to different geometry via
        // `configure_image` must produce DISTINCT keys — otherwise the EGLImage
        // imported for the first geometry is reused at the wrong pitch.
        let mut map: HashMap<EglCacheKey, u32> = HashMap::new();
        let g0 = key(0xBEEF, 0, 128, 96, 128, PixelFormat::Grey);
        let g1 = key(0xBEEF, 0, 96, 128, 96, PixelFormat::Grey); // different w/h/stride
        let g2 = key(0xBEEF, 0, 128, 96, 128, PixelFormat::Nv12); // different format
        map.insert(g0, 1);
        map.insert(g1, 2);
        map.insert(g2, 3);
        assert_eq!(
            map.len(),
            3,
            "geometry/format-distinct reuses must not collide"
        );

        // Same identity + same geometry is still a genuine hit.
        let g0_again = key(0xBEEF, 0, 128, 96, 128, PixelFormat::Grey);
        map.insert(g0_again, 4);
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&g0), Some(&4));
    }

    #[test]
    fn cache_key_distinguishes_stride() {
        // A stride-only change (same identity/offset/w/h/format, different
        // row_stride — e.g. a padded vs tight pool buffer) must be a DISTINCT
        // key. Guards against dropping `row_stride` from the key, which would
        // reintroduce the wrong-pitch stale read on a re-padded pool.
        let mut map: HashMap<EglCacheKey, u32> = HashMap::new();
        let tight = key(0xBEEF, 0, 128, 96, 128, PixelFormat::Grey);
        let padded = key(0xBEEF, 0, 128, 96, 256, PixelFormat::Grey); // stride differs
        map.insert(tight, 1);
        map.insert(padded, 2);
        assert_eq!(map.len(), 2, "stride-distinct imports must not collide");
        assert_eq!(map.get(&tight), Some(&1));
        assert_eq!(map.get(&padded), Some(&2));
    }
}
