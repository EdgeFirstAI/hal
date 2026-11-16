// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use edgefirst_tensor::{PixelFormat, Tensor, TensorTrait};

/// Selects which import cache to use.
#[derive(Debug, PartialEq)]
pub(super) enum CacheKind {
    Src,
    Dst,
}

/// A cached buffer import, retained until evicted by the LRU bound.
///
/// Generic over the platform's owned import object `I` — an `EglImage`
/// (DMA-BUF) on Linux, an IOSurface-backed EGL pbuffer on macOS. Dropping
/// the entry drops `I`, which releases the platform object. The entry
/// outlives the tensor it was built from.
pub(super) struct CachedImport<I> {
    pub(super) import: I,
    /// Optional GL renderbuffer backed by this import (used by direct RGB path).
    pub(super) renderbuffer: Option<u32>,
    /// Monotonic access counter for LRU eviction.
    pub(super) last_used: u64,
}

/// Per-processor zero-copy telemetry: how convert sources reached the GPU
/// and how often a zero-copy opportunity was declined into a copy path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConvertStats {
    /// Sources fed zero-copy: the tensor's Dma buffer became the sampled
    /// texture's storage (EGLImage attach / IOSurface pbuffer bind).
    pub src_imports: u64,
    /// Sources fed from a PBO (GL-internal copy, no CPU visit).
    pub src_pbo_uploads: u64,
    /// Sources fed by CPU map + TexImage upload — the copy path.
    pub src_uploads: u64,
    /// Times a zero-copy source feed was attempted and declined into a
    /// copy path (import/attach failure).
    pub zero_copy_declines: u64,
    /// Times a zero-copy destination import was attempted and declined into
    /// the mapped-texture readback (render to a texture, read back through
    /// `map()` at the destination's offset and pitch).
    pub dst_import_fallbacks: u64,
}

/// Identity + geometry that uniquely determine an imported GPU buffer
/// (an EGLImage over a DMA-BUF on Linux; an EGL pbuffer over an IOSurface
/// on macOS).
///
/// `luma_id` / `chroma_id` are the buffer identities.
///
/// A destination `view()`/`batch()` sub-region is keyed on its parent
/// geometry (`view_origin`) with `plane_offset = 0`: every sibling view of
/// one buffer shares a single key. A source, and a non-view tensor carrying
/// a byte offset, is keyed by its own `plane_offset`.
///
/// `width` / `height` / `row_stride` / `format` are the geometry the import
/// was made with (the parent's for a destination view).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) struct BufferImportKey {
    pub(super) luma_id: u64,
    pub(super) chroma_id: Option<u64>,
    pub(super) plane_offset: usize,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) row_stride: usize,
    pub(super) format: PixelFormat,
}

impl BufferImportKey {
    /// Build the cache key from a tensor and the format it will be imported as.
    pub(super) fn from_tensor<T>(img: &Tensor<T>, format: PixelFormat, for_dst: bool) -> Self
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        // A destination view keys on its parent's geometry and row stride at
        // offset 0. A source view or a whole tensor keys on its own geometry
        // and plane offset.
        let view_origin = if for_dst { img.view_origin() } else { None };
        let (width, height, row_stride, plane_offset) = match view_origin {
            Some(vo) => (vo.parent_width, vo.parent_height, vo.parent_row_stride, 0),
            None => (
                img.width().unwrap_or(0),
                img.height().unwrap_or(0),
                img.effective_row_stride().unwrap_or(0),
                img.plane_offset().unwrap_or(0),
            ),
        };
        Self {
            luma_id: img.buffer_identity().id(),
            chroma_id: img.chroma().map(|t| t.buffer_identity().id()),
            plane_offset,
            width,
            height,
            row_stride,
            format,
        }
    }
}

/// Snapshot of one EGLImage cache's hit/miss counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    /// High-water mark of `entries`. Saturates at `capacity`.
    pub peak_entries: usize,
    /// Entries evicted to stay within `capacity`.
    pub evictions: u64,
}

/// Combined snapshot of every EGLImage cache on the GL processor
/// (source, destination, the NV R8 source cache, and its chroma-plane view).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlCacheStats {
    pub src: CacheStats,
    pub dst: CacheStats,
    pub nv_r8: CacheStats,
    pub nv_uv: CacheStats,
}

impl GlCacheStats {
    /// Total imports performed (cache misses) across all caches.
    pub fn total_misses(&self) -> u64 {
        self.src.misses + self.dst.misses + self.nv_r8.misses + self.nv_uv.misses
    }

    /// Peak entries summed across all caches. An upper bound on the number
    /// of buffers pinned at once: entries in different caches can name the
    /// same buffer.
    pub fn total_peak_entries(&self) -> usize {
        self.src.peak_entries
            + self.dst.peak_entries
            + self.nv_r8.peak_entries
            + self.nv_uv.peak_entries
    }

    /// Evictions across all caches.
    pub fn total_evictions(&self) -> u64 {
        self.src.evictions + self.dst.evictions + self.nv_r8.evictions + self.nv_uv.evictions
    }
}

/// Buffer-import cache owned by the GL processor: a HashMap with a monotonic
/// access counter for LRU eviction, bounded by `capacity`.
pub(super) struct ImportCache<I> {
    pub(super) entries: std::collections::HashMap<BufferImportKey, CachedImport<I>>,
    pub(super) capacity: usize,
    pub(super) hits: u64,
    pub(super) misses: u64,
    /// Monotonic counter incremented on each access for LRU tracking.
    pub(super) access_counter: u64,
    /// High-water mark of `entries.len()`. See [`CacheStats::peak_entries`].
    peak_entries: usize,
    /// Count of entries evicted to stay within `capacity`. See
    /// [`CacheStats::evictions`].
    evictions: u64,
}

impl<I> ImportCache<I> {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            entries: std::collections::HashMap::with_capacity(capacity),
            capacity,
            hits: 0,
            misses: 0,
            access_counter: 0,
            peak_entries: 0,
            evictions: 0,
        }
    }

    /// Snapshot the hit/miss counters.
    pub(super) fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.entries.len(),
            peak_entries: self.peak_entries,
            evictions: self.evictions,
        }
    }

    /// Insert a freshly created import, evicting the least recently used
    /// entry first if the cache is at capacity, and update the high-water
    /// mark.
    pub(super) fn insert(&mut self, key: BufferImportKey, import: I, renderbuffer: Option<u32>) {
        if self.entries.len() >= self.capacity {
            self.evict_lru();
        }
        let last_used = self.next_timestamp();
        self.entries.insert(
            key,
            CachedImport {
                import,
                renderbuffer,
                last_used,
            },
        );
        self.peak_entries = self.peak_entries.max(self.entries.len());
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
            self.evictions += 1;
            if let Some(rbo) = evicted.renderbuffer {
                unsafe { edgefirst_gl::gl::DeleteRenderbuffers(1, &rbo) };
            }
            return true;
        }
        false
    }
}

impl<I> Drop for ImportCache<I> {
    fn drop(&mut self) {
        for entry in self.entries.values() {
            if let Some(rbo) = entry.renderbuffer {
                unsafe { edgefirst_gl::gl::DeleteRenderbuffers(1, &rbo) };
            }
        }
        log::debug!(
            "ImportCache stats: {} hits, {} misses, {} entries remaining, \
             peak {}/{}, {} evictions",
            self.hits,
            self.misses,
            self.entries.len(),
            self.peak_entries,
            self.capacity,
            self.evictions
        );
    }
}

#[cfg(test)]
mod tests {
    use super::BufferImportKey;
    use edgefirst_tensor::PixelFormat;
    use std::collections::HashMap;

    fn key(
        luma_id: u64,
        plane_offset: usize,
        width: usize,
        height: usize,
        row_stride: usize,
        format: PixelFormat,
    ) -> BufferImportKey {
        BufferImportKey {
            luma_id,
            chroma_id: None,
            plane_offset,
            width,
            height,
            row_stride,
            format,
        }
    }

    /// Source views whose offsets share a 64-byte-aligned base but differ in remainder key apart.
    #[test]
    fn source_keys_distinguish_offsets_that_share_an_aligned_base() {
        let a = key(1, 2048, 16, 16, 256, PixelFormat::Rgba);
        let b = key(1, 2064, 16, 16, 256, PixelFormat::Rgba);
        let c = key(1, 2080, 16, 16, 256, PixelFormat::Rgba);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
        let mut map: HashMap<BufferImportKey, u32> = HashMap::new();
        map.insert(a, 1);
        map.insert(b, 2);
        map.insert(c, 3);
        assert_eq!(map.len(), 3, "three remainders, three imports");
    }

    /// `peak_entries` and `evictions` track capacity pressure, and eviction follows LRU order.
    #[test]
    fn peak_and_evictions_report_capacity_pressure() {
        use super::ImportCache;
        let k = |id| key(id, 0, 64, 64, 64, PixelFormat::Grey);
        let mut cache: ImportCache<u32> = ImportCache::new(2);

        cache.insert(k(1), 10, None);
        cache.insert(k(2), 20, None);
        let full = cache.stats();
        assert_eq!((full.entries, full.peak_entries), (2, 2));
        assert_eq!(full.evictions, 0, "a cache within capacity must not evict");

        // A third buffer at capacity evicts the LRU entry, k(1).
        cache.insert(k(3), 30, None);
        let pressed = cache.stats();
        assert_eq!(pressed.entries, 2, "capacity is enforced");
        assert_eq!(
            pressed.evictions, 1,
            "an over-capacity insert must record an eviction — this is the \
             signal that separates a workload that fits from one that thrashes, \
             since peak_entries saturates at capacity in both cases"
        );
        assert_eq!(pressed.peak_entries, 2, "peak saturates at capacity");
        assert!(!cache.entries.contains_key(&k(1)), "LRU entry was evicted");
        assert!(cache.entries.contains_key(&k(3)), "new entry was inserted");
    }

    #[test]
    fn cache_key_distinguishes_foreign_plane_offset() {
        // Two non-view imports at different plane offsets are distinct entries.
        let mut map: HashMap<BufferImportKey, u32> = HashMap::new();
        let base = key(0xABCD, 0, 64, 64, 64, PixelFormat::Grey);
        let at_offset = key(0xABCD, 4096, 64, 64, 64, PixelFormat::Grey);
        map.insert(base, 1);
        map.insert(at_offset, 2);
        assert_eq!(
            map.len(),
            2,
            "offset-distinct foreign imports must not collide"
        );
        assert_eq!(map.get(&base), Some(&1));
        assert_eq!(map.get(&at_offset), Some(&2));

        // Identical keys collide.
        map.insert(base, 3);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&base), Some(&3));
    }

    #[test]
    fn cache_key_collapses_sibling_views() {
        // Sibling destination views of one buffer key identically to the whole parent.
        use edgefirst_tensor::{Region, Tensor, TensorMemory};
        let parent = Tensor::<u8>::image(
            64,
            64,
            PixelFormat::Rgba,
            Some(TensorMemory::Mem),
            edgefirst_tensor::CpuAccess::ReadWrite,
        )
        .unwrap();
        let a = parent.view(Region::new(0, 0, 32, 32)).unwrap();
        let b = parent.view(Region::new(0, 32, 32, 32)).unwrap();
        // Destinations (`for_dst = true`) collapse onto the parent key.
        let ka = BufferImportKey::from_tensor(&a, PixelFormat::Rgba, true);
        let kb = BufferImportKey::from_tensor(&b, PixelFormat::Rgba, true);
        let kp = BufferImportKey::from_tensor(&parent, PixelFormat::Rgba, true);
        assert_eq!(
            ka, kb,
            "sibling dst views collapse to one parent-keyed import"
        );
        assert_eq!(ka, kp, "a dst view keys identically to its whole parent");
        assert_eq!(
            ka.plane_offset, 0,
            "a dst view contributes no offset to the key"
        );
        assert_eq!((ka.width, ka.height), (64, 64), "keyed on parent geometry");

        // Sources (`for_dst = false`) key on their own region.
        let sa = BufferImportKey::from_tensor(&a, PixelFormat::Rgba, false);
        let sb = BufferImportKey::from_tensor(&b, PixelFormat::Rgba, false);
        assert_ne!(sa, sb, "source views key on their own region (no collapse)");
        assert_eq!(
            (sa.width, sa.height),
            (32, 32),
            "a source view keys on its own dimensions"
        );
    }

    #[test]
    fn cache_key_distinguishes_geometry() {
        // One buffer reconfigured to different geometry produces distinct keys.
        let mut map: HashMap<BufferImportKey, u32> = HashMap::new();
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

        // The same identity and geometry is a hit.
        let g0_again = key(0xBEEF, 0, 128, 96, 128, PixelFormat::Grey);
        map.insert(g0_again, 4);
        assert_eq!(map.len(), 3);
        assert_eq!(map.get(&g0), Some(&4));
    }

    #[test]
    fn cache_key_distinguishes_stride() {
        // A stride-only change is a distinct key.
        let mut map: HashMap<BufferImportKey, u32> = HashMap::new();
        let tight = key(0xBEEF, 0, 128, 96, 128, PixelFormat::Grey);
        let padded = key(0xBEEF, 0, 128, 96, 256, PixelFormat::Grey); // stride differs
        map.insert(tight, 1);
        map.insert(padded, 2);
        assert_eq!(map.len(), 2, "stride-distinct imports must not collide");
        assert_eq!(map.get(&tight), Some(&1));
        assert_eq!(map.get(&padded), Some(&2));
    }
}
