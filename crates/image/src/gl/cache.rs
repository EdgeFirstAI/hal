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
/// the entry drops `I`, which releases the platform object.
///
/// **`import` is the anchor, and it is load-bearing for correctness, not
/// only for speed.** The entry deliberately outlives the tensor it was built
/// from: a cross-package `convert()` constructs a fresh tensor over the
/// producer's buffer, uses it, and drops it within the call, so an entry tied
/// to that tensor's lifetime would be created and destroyed every frame and
/// never hit (measured: 0 hits / 50 misses, against 49/1 for a native
/// source). Outliving the tensor is safe because the platform import holds
/// its own driver-side reference to the buffer, so a cached entry keeps the
/// underlying object alive even after every userspace handle is closed — the
/// system key it is identified by (a dma-buf `(st_dev, st_ino)`, an
/// `IOSurfaceID`) therefore cannot be recycled onto a *different* buffer
/// while the entry lives, which is what makes a stale key impossible rather
/// than merely unlikely. Verified on hardware, both directions: on Linux the
/// buffer's inode is still listed in `/sys/kernel/debug/dma_buf/bufinfo`
/// after the tensor drops (Vivante/imx8mp, Mali/imx95, V3D/rpi5) and
/// disappears the moment the cache drops; on macOS the `IOSurfaceID` still
/// resolves after the tensor drops and stops resolving once the cache drops.
///
/// The consequence for any future refactor: the cache must keep owning `I`.
/// Storing only the raw handle (`Platform::import_handle`) would keep the
/// key-matching behaviour and silently lose the anchor.
///
/// Only DMA-BUF/IOSurface-backed tensors ever reach here — both
/// `Platform::import_buffer*` implementations reject anything else — so the
/// process-local identity kinds that genuinely could be recycled
/// (`IdentityKind::HostPtr`, `Pbo`) are never cached.
pub(super) struct CachedImport<I> {
    pub(super) import: I,
    /// Optional GL renderbuffer backed by this import (used by direct RGB path).
    pub(super) renderbuffer: Option<u32>,
    /// Monotonic access counter for LRU eviction.
    pub(super) last_used: u64,
}

/// Per-processor zero-copy telemetry: how convert sources reached the GPU
/// and how often a zero-copy opportunity was declined into a copy path.
///
/// This is deliberately SEPARATE from [`GlCacheStats`]: that struct is the
/// steady-state import-cache equality gate (tests assert exact equality on
/// it), while these counters change on every convert. `src_uploads` staying
/// at 0 across a Dma-source workload is the "no silent zero-copy drop"
/// assertion callers (and the on-device validation harness) can make —
/// EGLImage miss counts alone are blind to uploads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ConvertStats {
    /// Sources fed zero-copy: the tensor's Dma buffer became the sampled
    /// texture's storage (EGLImage attach / IOSurface pbuffer bind).
    pub src_imports: u64,
    /// Sources fed from a PBO (GL-internal copy, no CPU visit).
    pub src_pbo_uploads: u64,
    /// Sources fed by CPU map + TexImage upload — the copy path.
    pub src_uploads: u64,
    /// Times a zero-copy source feed was attempted and DECLINED into a
    /// copy path (import/attach failure). A nonzero value with Dma
    /// sources means the platform/driver refused the fast path.
    pub zero_copy_declines: u64,
}

/// Buffer-import cache owned by the GL processor.
///
/// Uses a HashMap with a monotonic counter for LRU eviction: each access
/// updates the entry's `last_used` timestamp, and eviction removes the entry
/// with the smallest `last_used` value. Recency is the *only* eviction
/// signal — entries are retained past the lifetime of the tensor that
/// produced them, which is what lets a re-imported buffer hit (see
/// [`CachedImport`]), so `capacity` is the sole bound.
/// Identity + geometry that uniquely determine an imported GPU buffer
/// (an EGLImage over a DMA-BUF on Linux; an EGL pbuffer over an IOSurface
/// on macOS — the key fields are platform-neutral).
///
/// `luma_id` / `chroma_id` are the buffer identities.
///
/// A `view()`/`batch()` sub-region is a `glViewport`/`glScissor` ROI into its
/// parent, **not** a distinct import: [`from_tensor`](BufferImportKey::from_tensor)
/// keys such a tensor on its **parent** geometry (`view_origin`) with
/// `plane_offset = 0`, so every sibling view of one buffer collapses to a single
/// EGLImage and the per-tile offset becomes render state (the viewport). The
/// remaining `plane_offset` use is a non-view tensor that carries a genuine
/// foreign/multi-plane byte offset (e.g. an externally-imported buffer whose data
/// starts past the fd origin); those still key distinctly.
///
/// `width` / `height` / `row_stride` / `format` capture the geometry the
/// EGLImage was imported with — the **parent's** for a view. A pooled buffer
/// reused at a different size via `Tensor::configure_image` (e.g. a 128-wide pool
/// decoding a 96-wide image) keeps the same identities but needs a fresh import:
/// the import's pitch/dimensions/fourcc all derive from these fields. Omitting
/// them reuses the stale-geometry EGLImage and the GPU samples the buffer at the
/// wrong pitch — deterministically wrong single-threaded, nondeterministic in
/// parallel, correct only on a heap source (which never takes this path).
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
    /// Every construction site MUST go through this so the key used to insert
    /// an EGLImage matches the key used to look it up and to gate the texture
    /// binding-skip.
    pub(super) fn from_tensor<T>(img: &Tensor<T>, format: PixelFormat, for_dst: bool) -> Self
    where
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync + edgefirst_tensor::Element,
    {
        // A DESTINATION view()/batch() sub-region keys on its PARENT so all
        // siblings of one buffer collapse to a single import; the view's offset is
        // the viewport, not a key. It keys on the parent's `row_stride` (from
        // `view_origin`), NOT the view's own `effective_row_stride` — a single-row
        // view sets a tight stride for map-span safety, which would otherwise
        // mis-key it apart from its multi-row siblings. A SOURCE view (or a whole
        // tensor) keys on its OWN geometry + any genuine foreign/multi-plane
        // plane_offset — a source view imports its own region (it is sampled, not
        // rendered into), so it must NOT collapse onto the parent key.
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
///
/// The counters themselves have always existed (logged at `Drop`); this
/// snapshot makes them **assertable**: steady-state tests capture stats after
/// warmup and after an N-frame loop and require `misses` to stay flat — any
/// increase means a convert re-imported a buffer it should have found cached,
/// which is the cache-behavior equality gate for GL refactors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
    pub entries: usize,
    /// High-water mark of `entries`.
    ///
    /// Since entries are retained past the lifetime of the tensor that
    /// produced them (the retention that `CachedImport` documents), each one
    /// pins its buffer, so
    /// occupancy — not hit rate — is what says how much memory a workload
    /// holds. Below `capacity` this is the workload's true working-set size,
    /// and the number to size `EDGEFIRST_EGL_CACHE_CAPACITY` against.
    ///
    /// It **saturates at `capacity`**, so on its own it cannot tell a
    /// workload that exactly fits from one that is thrashing — both report
    /// `peak_entries == capacity`. Read it together with [`evictions`], which
    /// is what distinguishes them.
    ///
    /// [`evictions`]: Self::evictions
    pub peak_entries: usize,
    /// Entries evicted to stay within `capacity`.
    ///
    /// The signal that `capacity` is too small for the workload: every
    /// eviction discards an import that is likely to be asked for again, so
    /// a steady-state pipeline should reach zero. Nonzero-and-climbing with
    /// `peak_entries == capacity` is thrashing — raise
    /// `EDGEFIRST_EGL_CACHE_CAPACITY`, at the cost of pinning more buffers.
    pub evictions: u64,
}

/// Combined snapshot of every EGLImage cache on the GL processor
/// (source, destination, and the Path-B NV R8 source cache).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GlCacheStats {
    pub src: CacheStats,
    pub dst: CacheStats,
    pub nv_r8: CacheStats,
}

impl GlCacheStats {
    /// Total imports performed (cache misses) across all caches — the number
    /// steady-state loops assert stays flat.
    pub fn total_misses(&self) -> u64 {
        self.src.misses + self.dst.misses + self.nv_r8.misses
    }

    /// Peak entries summed across all caches — an UPPER bound on how many
    /// buffers were pinned at once, not an exact count: `src` and `nv_r8`
    /// routinely name the *same* source buffer (one NV12 tensor imported
    /// two ways), and two entries over one buffer pin that buffer once.
    /// Use the per-cache peaks to size capacity; use this to bound memory.
    pub fn total_peak_entries(&self) -> usize {
        self.src.peak_entries + self.dst.peak_entries + self.nv_r8.peak_entries
    }

    /// Evictions across all caches — zero in a correctly sized steady state.
    pub fn total_evictions(&self) -> u64 {
        self.src.evictions + self.dst.evictions + self.nv_r8.evictions
    }
}

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

    /// Snapshot the hit/miss counters for steady-state assertions.
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
    /// entry first if the cache is at capacity.
    ///
    /// Every insert site goes through here so that the capacity bound and the
    /// high-water mark cannot drift apart from each other — a site that
    /// inserted directly would silently raise the real peak above the number
    /// [`stats`](Self::stats) reports, which is the number capacity is tuned
    /// against.
    ///
    /// Eviction happens *after* the caller has successfully created the
    /// import: a failed import must not cost the cache a live entry.
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
        // peak/evictions are the two numbers the capacity story rests on, and
        // they were previously reachable only through the API -- so a field
        // report of "GPU memory pressure" could not be diagnosed from a log.
        // `peak < capacity` with zero evictions means correctly sized;
        // evictions climbing means the producer's pool is deeper than the bound.
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

    /// The two numbers Task 7 (capacity sizing) reads, and the LRU order they
    /// describe. Runs without a GL context: `evict_lru` only touches GL for an
    /// entry that carries a renderbuffer, and these carry none.
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

        // Third distinct buffer at capacity: the LRU entry (k(1)) goes.
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
        // `plane_offset` no longer distinguishes view()/batch() sub-regions —
        // those key on their parent and collapse (see
        // `cache_key_collapses_sibling_views`). It survives ONLY for a non-view
        // tensor carrying a genuine foreign/multi-plane byte offset (e.g. an
        // externally-imported buffer whose data starts past the fd origin); two
        // such imports at different offsets must remain DISTINCT entries.
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

        // Identical keys still collide (a genuine cache hit), as before.
        map.insert(base, 3);
        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&base), Some(&3));
    }

    #[test]
    fn cache_key_collapses_sibling_views() {
        // The batch-engine pivot: a view()/batch() sub-region is a
        // glViewport/scissor ROI into its parent, so sibling views of one buffer
        // MUST produce the SAME cache key (one shared EGLImage import) and key
        // identically to the whole parent. The per-tile offset is render state,
        // never part of the key.
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

        // SOURCES (`for_dst = false`) key on their OWN region — a source view is
        // imported and SAMPLED, not rendered into, so two source views of one
        // parent must NOT collapse (they'd alias and sample the wrong region).
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
        // Root-cause regression guard for the pool-recycle bug: ONE buffer
        // (same luma_id + offset) reconfigured to different geometry via
        // `configure_image` must produce DISTINCT keys — otherwise the EGLImage
        // imported for the first geometry is reused at the wrong pitch.
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
