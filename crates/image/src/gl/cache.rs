// SPDX-FileCopyrightText: Copyright 2025 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

use edgefirst_tensor::{PixelFormat, Tensor, TensorTrait};

/// Selects which import cache to use.
#[derive(Debug, PartialEq)]
pub(super) enum CacheKind {
    Src,
    Dst,
}

/// A cached buffer import with a weak reference to the source tensor's guard.
///
/// Generic over the platform's owned import object `I` — an `EglImage`
/// (DMA-BUF) on Linux, an IOSurface-backed EGL pbuffer on macOS. Dropping
/// the entry drops `I`, which releases the platform object.
pub(super) struct CachedImport<I> {
    pub(super) import: I,
    /// Weak reference to the source Tensor's BufferIdentity guard.
    pub(super) guard: std::sync::Weak<()>,
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
/// with the smallest `last_used` value.
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
        T: num_traits::Num + Clone + std::fmt::Debug + Send + Sync,
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
}

pub(super) struct ImportCache<I> {
    pub(super) entries: std::collections::HashMap<BufferImportKey, CachedImport<I>>,
    pub(super) capacity: usize,
    pub(super) hits: u64,
    pub(super) misses: u64,
    /// Monotonic counter incremented on each access for LRU tracking.
    pub(super) access_counter: u64,
}

impl<I> ImportCache<I> {
    pub(super) fn new(capacity: usize) -> Self {
        Self {
            entries: std::collections::HashMap::with_capacity(capacity),
            capacity,
            hits: 0,
            misses: 0,
            access_counter: 0,
        }
    }

    /// Snapshot the hit/miss counters for steady-state assertions.
    pub(super) fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            entries: self.entries.len(),
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
                unsafe { edgefirst_gl::gl::DeleteRenderbuffers(1, &rbo) };
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
                    unsafe { edgefirst_gl::gl::DeleteRenderbuffers(1, &rbo) };
                }
            }
            alive
        });
        let swept = before - self.entries.len();
        if swept > 0 {
            log::debug!("ImportCache: swept {swept} dead entries");
        }
        swept > 0
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
            "ImportCache stats: {} hits, {} misses, {} entries remaining",
            self.hits,
            self.misses,
            self.entries.len()
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
