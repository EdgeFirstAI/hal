// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Platform-neutral AHardwareBuffer layout logic — the pure half of
//! [`ahardwarebuffer`](crate::ahardwarebuffer), split out so it compiles
//! and unit-tests on every host.
//!
//! No CI lane can *execute* `cfg(target_os = "android")` code (the
//! Android lane is compile + link-validation only), so anything living in
//! the FFI module is invisible to regression tests until the Device Farm
//! coverage lane exists. The format-mapping table and the descriptor
//! geometry/overflow math are exactly the logic that must not drift — the
//! macOS analog of the table caused a silent R↔B swap during bring-up
//! (see `iosurface.rs`) — so they live here, cfg-free, with host tests.
//! The Android module re-exports them; nothing here touches FFI.

use crate::{
    error::{Error, Result},
    DType, PixelFormat,
};

/// `AHardwareBuffer_Desc` from `<android/hardware_buffer.h>`. Layout must
/// match the NDK header exactly (40 bytes, no padding); `stride` is
/// filled by `AHardwareBuffer_allocate`/`_describe` (in pixels, not
/// bytes). Defined here so the geometry math below is host-testable; the
/// Android module uses it in the FFI signatures.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct AHardwareBufferDesc {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) layers: u32,
    pub(crate) format: u32,
    pub(crate) usage: u64,
    pub(crate) stride: u32,
    pub(crate) rfu0: u32,
    pub(crate) rfu1: u64,
}

// AHardwareBuffer_UsageFlags CPU bits (subset used by the HAL). The CPU
// flags are enum VALUES within their masks (NEVER=0, RARELY=2, OFTEN=3),
// not independent bits — lock() must replay allocated values. Kept here
// (host-compiled) so the lock-usage decision below is unit-testable; the
// FFI module re-uses these.
pub(crate) const USAGE_CPU_READ_OFTEN: u64 = 0x3;
pub(crate) const USAGE_CPU_READ_MASK: u64 = 0xF;
pub(crate) const USAGE_CPU_WRITE_OFTEN: u64 = 0x30;
pub(crate) const USAGE_CPU_WRITE_MASK: u64 = 0xF0;

/// Outcome of matching a map request against the allocation-time CPU
/// usage — the pure half of `AHardwareBufferMap::new_inner`'s lock logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LockDecision {
    /// Buffer has no CPU usage at all (`CpuAccess::None`): locking with
    /// undeclared bits is undefined behaviour per the NDK contract, so
    /// the best-effort map contract resolves to a deterministic refusal.
    Refuse,
    /// The declared usage covers the requested direction: lock with the
    /// declared VALUE bits masked to the requested direction (read-only
    /// maps skip the writeback; write-only maps can go write-combined).
    Covered(u64),
    /// Requested beyond the declaration: replay the full declared flags
    /// (the pre-CpuAccess behavior — correct cache maintenance for what
    /// the buffer CAN do), counted as unplanned by the caller.
    Unplanned(u64),
}

/// Decide the `AHardwareBuffer_lock` usage bits for a map request of
/// direction `(reads, writes)` against the allocation's CPU usage bits.
pub(crate) fn lock_usage_for(cpu_usage: u64, reads: bool, writes: bool) -> LockDecision {
    let declared = cpu_usage & (USAGE_CPU_READ_MASK | USAGE_CPU_WRITE_MASK);
    if declared == 0 {
        return LockDecision::Refuse;
    }
    let declared_reads = cpu_usage & USAGE_CPU_READ_MASK != 0;
    let declared_writes = cpu_usage & USAGE_CPU_WRITE_MASK != 0;
    if (!reads || declared_reads) && (!writes || declared_writes) {
        let mut bits = 0;
        if reads {
            bits |= cpu_usage & USAGE_CPU_READ_MASK;
        }
        if writes {
            bits |= cpu_usage & USAGE_CPU_WRITE_MASK;
        }
        LockDecision::Covered(bits)
    } else {
        LockDecision::Unplanned(declared)
    }
}

/// Classify a GPU vendor string (the `ro.hardware.egl` system property)
/// into its native tile-compression scheme. Pure so the mapping is
/// host-testable; the Android module feeds it the live property value.
///
/// Emulation stacks (`angle`, `emulation`, `swiftshader`) and unknown
/// vendors return `None` — a scheme is recorded only on positive
/// identification (there is deliberately no `Unknown` variant).
pub(crate) fn scheme_for_egl_vendor(vendor: &str) -> Option<crate::CompressionScheme> {
    use crate::CompressionScheme::*;
    let v = vendor.trim().to_ascii_lowercase();
    if v.is_empty() || v.contains("emulation") || v.contains("angle") || v.contains("swiftshader") {
        return None;
    }
    if v.contains("adreno") {
        Some(Ubwc)
    } else if v.contains("mali") || v.contains("immortalis") {
        Some(Afbc)
    } else if v.contains("powervr") {
        Some(Pvric)
    } else if v.contains("xclipse") {
        Some(Dcc)
    } else {
        None
    }
}

/// Whether `(format, dtype)` is eligible for a vendor tile-compressed
/// allocation. Conservative by design: RGBA8888 `u8`/`i8` initially —
/// the layout every vendor compresses and the one QNN names in its UBWC
/// data formats. Growing this table is a per-format, per-device
/// validation exercise (Device Farm), not a code change alone.
pub(crate) fn compression_eligible(format: PixelFormat, dtype: DType) -> bool {
    matches!(format, PixelFormat::Rgba) && matches!(dtype, DType::U8 | DType::I8)
}

/// Identity-intern policy — the pure half of the Android
/// `AHardwareBuffer_getId` interning (host-tested; the FFI module owns
/// the `dlsym` lookup and the process-wide table).
///
/// Maps a stable system buffer key (the 64-bit `AHardwareBuffer_getId`)
/// to the HAL's own `BufferIdentity` parts. Reuse requires the recorded
/// guard to still be LIVE: a live guard means some tensor still holds
/// its acquire-reference on the underlying buffer, so the system cannot
/// have recycled that id (the ABA case is defused by construction —
/// guards are held only by tensors, never by caches, which hold weaks).
/// Dead entries are pruned on every insert, bounding the table to the
/// number of live buffers.
pub(crate) struct IdentityInternTable {
    map: std::collections::HashMap<u64, (u64, std::sync::Weak<()>)>,
}

impl IdentityInternTable {
    pub(crate) fn new() -> Self {
        Self {
            map: std::collections::HashMap::new(),
        }
    }

    /// Resolve `key` to identity parts: a live recorded entry is reused
    /// verbatim; otherwise `mint` supplies fresh parts which are
    /// recorded (after pruning dead entries). Returns `(id, guard,
    /// reused)`.
    pub(crate) fn resolve(
        &mut self,
        key: u64,
        mint: impl FnOnce() -> (u64, std::sync::Arc<()>),
    ) -> (u64, std::sync::Arc<()>, bool) {
        if let Some((id, weak)) = self.map.get(&key) {
            if let Some(guard) = weak.upgrade() {
                return (*id, guard, true);
            }
        }
        self.map.retain(|_, (_, weak)| weak.strong_count() > 0);
        let (id, guard) = mint();
        self.map
            .insert(key, (id, std::sync::Arc::downgrade(&guard)));
        (id, guard, false)
    }

    /// Number of recorded entries (test observability).
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }
}

// AHardwareBuffer_Format values (subset used by the HAL).
/// RGBA 8:8:8:8 UNORM — API 26+.
pub(crate) const FORMAT_R8G8B8A8_UNORM: u32 = 1;
/// RGBA 16:16:16:16 half-float — API 26+. The F16 NCHW render target.
pub(crate) const FORMAT_R16G16B16A16_FLOAT: u32 = 0x16;
/// Opaque byte buffer (`width` = length, `height = layers = 1`) — API 26+.
/// The layout NNAPI uses for tensor blobs; the byte-bag allocation format.
pub(crate) const FORMAT_BLOB: u32 = 0x21;

/// Bytes-per-element for a known AHardwareBuffer pixel format, or `None`
/// for formats the HAL cannot CPU-map linearly (multi-planar YUV etc.).
pub(crate) fn format_bpe(format: u32) -> Option<usize> {
    match format {
        FORMAT_R8G8B8A8_UNORM => Some(4),
        FORMAT_R16G16B16A16_FLOAT => Some(8),
        FORMAT_BLOB => Some(1),
        _ => None,
    }
}

/// Byte pitch + allocation size derived from a (post-allocation)
/// descriptor. BLOB buffers are `width` bytes with a single "row"; image
/// formats use the allocator-chosen `stride` (pixels) × bytes-per-element.
/// `None` when the format has no linear CPU layout or the size overflows.
pub(crate) fn desc_layout(desc: &AHardwareBufferDesc) -> Option<(usize, usize)> {
    let bpe = format_bpe(desc.format)?;
    if desc.format == FORMAT_BLOB {
        let len = desc.width as usize;
        return Some((len, len));
    }
    let bytes_per_row = (desc.stride as usize).checked_mul(bpe)?;
    let buf_size = bytes_per_row.checked_mul(desc.height as usize)?;
    Some((bytes_per_row, buf_size))
}

/// AHardwareBuffer format + bytes-per-element mapping for image-formatted
/// buffers, keyed on `(PixelFormat, DType)`. **This function is the single
/// source of truth for the mapping** — the tensor allocation side and the
/// image crate's Android GL import both read it (via
/// `image_ahardwarebuffer_layout`); keep the two layers in sync by not
/// duplicating this table (the same rule that prevents the macOS R↔B
/// drift documented in `iosurface.rs`).
///
/// Combinations not listed have no zero-copy path on Android today and
/// fall back to SHM/Mem + CPU conversion:
///
/// * `Grey`/`Nv12`/`Nv16`/`Nv24` u8 — the single-channel
///   `AHARDWAREBUFFER_FORMAT_R8_UNORM` requires API 29; the HAL floor is
///   26. Zero-copy Grey/NV on 29+ is a planned follow-up together with
///   the external-OES YUV sampling path.
/// * `Bgra` u8 — AHardwareBuffer has no BGRA format; mapping it to RGBA
///   would silently swap R↔B (the exact macOS footgun).
/// * `Yuyv` u8 — no packed-4:2:2 AHardwareBuffer format exists.
pub(crate) fn image_format_and_bpe(format: PixelFormat, dtype: DType) -> Option<(u32, usize)> {
    match (format, dtype) {
        // I8 shares the U8 layout everywhere below: INT8 is a per-byte
        // `^0x80` bias applied in the shader, not a format change — the
        // buffer bytes ARE the signed model input.
        (PixelFormat::Rgba, DType::U8 | DType::I8) => Some((FORMAT_R8G8B8A8_UNORM, 4)),
        // Packed RGB u8/i8 (the INT8 NPU input layout): no 3-channel
        // renderable format exists, so the tight `[H, W, 3]` byte stream
        // lives in an RGBA8888 surface sized `(W*3/4, H)` via
        // `packed_rgb888_layout` — the same texel-packing trick as the
        // planar-F16 arm below, and the geometry the GL engine's two-pass
        // packed-RGB shader already renders.
        (PixelFormat::Rgb, DType::U8 | DType::I8) => Some((FORMAT_R8G8B8A8_UNORM, 4)),
        // The F16 zero-copy path: RGBA16F, both as a packed RGBA image and
        // as the 4-elements-per-pixel packing of planar `[C, H, W]` f16
        // streams (surface sized via `packed_rgba16f_layout`).
        (PixelFormat::Rgba | PixelFormat::PlanarRgb | PixelFormat::PlanarRgba, DType::F16) => {
            Some((FORMAT_R16G16B16A16_FLOAT, 8))
        }
        _ => None,
    }
}

/// Byte footprint of `shape` for element type `T`, with overflow-checked
/// arithmetic — a shape whose element product (or its byte size) wraps
/// `usize` must be rejected, not allowed to slip past a capacity check and
/// produce an out-of-bounds map later.
pub(crate) fn checked_shape_bytes<T>(shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .and_then(|n| n.checked_mul(std::mem::size_of::<T>()))
        .ok_or_else(|| {
            Error::InvalidShape(format!(
                "shape footprint overflows usize (shape={shape:?}, sizeof T={})",
                std::mem::size_of::<T>()
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(width: u32, height: u32, format: u32, stride: u32) -> AHardwareBufferDesc {
        AHardwareBufferDesc {
            width,
            height,
            layers: 1,
            format,
            usage: 0,
            stride,
            rfu0: 0,
            rfu1: 0,
        }
    }

    #[test]
    fn identity_intern_reuses_live_and_remints_dead() {
        use std::sync::Arc;
        let mut table = IdentityInternTable::new();
        let mint = |id: u64| move || (id, Arc::new(()));

        // First sight of key 42 mints.
        let (id1, guard1, reused) = table.resolve(42, mint(100));
        assert_eq!((id1, reused), (100, false));

        // Same key while the guard lives → the SAME identity (the
        // CameraX re-wrap hit that makes the EGL import cache work).
        let (id2, _guard2, reused) = table.resolve(42, mint(101));
        assert_eq!((id2, reused), (100, true));

        // Guard dead (all tensors dropped) → fresh identity; a fresh id
        // is always safe even if the system recycled the key.
        drop(guard1);
        drop(_guard2);
        let (id3, _guard3, reused) = table.resolve(42, mint(102));
        assert_eq!((id3, reused), (102, false));
    }

    #[test]
    fn identity_intern_prunes_dead_entries_on_insert() {
        use std::sync::Arc;
        let mut table = IdentityInternTable::new();
        // Dead entries never accumulate: every insert prunes, so a
        // wrap-drop churn loop holds the table at one entry…
        for key in 0..8u64 {
            let (_, guard, _) = table.resolve(key, || (key + 100, Arc::new(())));
            drop(guard);
            assert_eq!(table.len(), 1);
        }
        // …while live entries are all retained.
        let guards: Vec<_> = (100..104u64)
            .map(|key| table.resolve(key, || (key, Arc::new(()))).1)
            .collect();
        assert_eq!(table.len(), 4);
        drop(guards);
        let (_, _live, _) = table.resolve(999, || (999, Arc::new(())));
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn egl_vendor_classifier_table() {
        use crate::CompressionScheme::*;
        // Real ro.hardware.egl values seen in the wild.
        assert_eq!(scheme_for_egl_vendor("adreno"), Some(Ubwc));
        assert_eq!(scheme_for_egl_vendor("Adreno735"), Some(Ubwc));
        assert_eq!(scheme_for_egl_vendor("mali"), Some(Afbc));
        assert_eq!(scheme_for_egl_vendor("immortalis-g925"), Some(Afbc));
        assert_eq!(scheme_for_egl_vendor("powervr"), Some(Pvric));
        assert_eq!(scheme_for_egl_vendor("xclipse"), Some(Dcc));
        // Emulation stacks and unknown vendors record linear.
        assert_eq!(scheme_for_egl_vendor("emulation"), None);
        assert_eq!(scheme_for_egl_vendor("angle"), None);
        assert_eq!(scheme_for_egl_vendor("swiftshader"), None);
        assert_eq!(scheme_for_egl_vendor(""), None);
        assert_eq!(scheme_for_egl_vendor("llvmpipe"), None);
    }

    #[test]
    fn compression_eligibility_table() {
        // Initial table: RGBA8888 u8/i8 only.
        assert!(compression_eligible(PixelFormat::Rgba, DType::U8));
        assert!(compression_eligible(PixelFormat::Rgba, DType::I8));
        assert!(!compression_eligible(PixelFormat::Rgba, DType::F16));
        assert!(!compression_eligible(PixelFormat::Rgb, DType::U8));
        assert!(!compression_eligible(PixelFormat::Nv12, DType::U8));
        assert!(!compression_eligible(PixelFormat::PlanarRgb, DType::U8));
    }

    #[test]
    fn lock_usage_decision_table() {
        use LockDecision::*;
        const R: u64 = USAGE_CPU_READ_OFTEN;
        const W: u64 = USAGE_CPU_WRITE_OFTEN;
        const GPU: u64 = 0x300; // non-CPU bits must be ignored

        // CpuAccess::None allocations refuse every direction.
        assert_eq!(lock_usage_for(GPU, true, false), Refuse);
        assert_eq!(lock_usage_for(GPU, false, true), Refuse);
        assert_eq!(lock_usage_for(0, true, true), Refuse);

        // Covered requests lock with the declared VALUES masked to the
        // requested direction.
        assert_eq!(lock_usage_for(R | W | GPU, true, false), Covered(R));
        assert_eq!(lock_usage_for(R | W | GPU, false, true), Covered(W));
        assert_eq!(lock_usage_for(R | W | GPU, true, true), Covered(R | W));
        assert_eq!(lock_usage_for(R | GPU, true, false), Covered(R));
        assert_eq!(lock_usage_for(W | GPU, false, true), Covered(W));

        // RARELY (=2 / =0x20) values replay exactly, never upgraded to
        // OFTEN — the lock must be a subset of the allocation.
        assert_eq!(lock_usage_for(0x2 | GPU, true, false), Covered(0x2));
        assert_eq!(lock_usage_for(0x20 | GPU, false, true), Covered(0x20));

        // Beyond-declaration requests replay the full declared flags
        // (pre-CpuAccess behavior) and are flagged Unplanned.
        assert_eq!(lock_usage_for(R | GPU, false, true), Unplanned(R));
        assert_eq!(lock_usage_for(R | GPU, true, true), Unplanned(R));
        assert_eq!(lock_usage_for(W | GPU, true, false), Unplanned(W));
        assert_eq!(lock_usage_for(W | GPU, true, true), Unplanned(W));
    }

    #[test]
    fn desc_matches_ndk_layout() {
        // `AHardwareBuffer_Desc` is 40 bytes in the NDK header; a size or
        // alignment drift here silently corrupts every allocate/describe.
        assert_eq!(std::mem::size_of::<AHardwareBufferDesc>(), 40);
        assert_eq!(std::mem::align_of::<AHardwareBufferDesc>(), 8);
    }

    #[test]
    fn format_table_is_stable() {
        // The (PixelFormat, DType) → AHardwareBuffer format table — the
        // R↔B/format-drift guard. Values from <android/hardware_buffer.h>.
        assert_eq!(
            image_format_and_bpe(PixelFormat::Rgba, DType::U8),
            Some((1, 4))
        );
        // Packed RGB u8 rides an RGBA8888 surface at (W*3/4, H) — the
        // format is RGBA even though the pixel format is Rgb.
        assert_eq!(
            image_format_and_bpe(PixelFormat::Rgb, DType::U8),
            Some((1, 4))
        );
        assert_eq!(
            image_format_and_bpe(PixelFormat::Rgba, DType::F16),
            Some((0x16, 8))
        );
        assert_eq!(
            image_format_and_bpe(PixelFormat::PlanarRgb, DType::F16),
            Some((0x16, 8))
        );
        assert_eq!(
            image_format_and_bpe(PixelFormat::PlanarRgba, DType::F16),
            Some((0x16, 8))
        );
        // No-mapping cases: R8 needs API 29, BGRA/YUYV have no AHB format,
        // planar u8 has no packing.
        assert_eq!(image_format_and_bpe(PixelFormat::Grey, DType::U8), None);
        assert_eq!(image_format_and_bpe(PixelFormat::Nv12, DType::U8), None);
        assert_eq!(image_format_and_bpe(PixelFormat::Bgra, DType::U8), None);
        assert_eq!(image_format_and_bpe(PixelFormat::Yuyv, DType::U8), None);
        assert_eq!(
            image_format_and_bpe(PixelFormat::PlanarRgb, DType::U8),
            None
        );
        assert_eq!(image_format_and_bpe(PixelFormat::Rgba, DType::F32), None);
    }

    #[test]
    fn blob_layout_is_single_row() {
        // BLOB: width = byte length, stride is meaningless (gralloc may
        // report anything); the whole allocation is one "row".
        let (row, size) = desc_layout(&desc(4096, 1, FORMAT_BLOB, 0)).unwrap();
        assert_eq!((row, size), (4096, 4096));
    }

    #[test]
    fn image_layout_uses_allocator_stride() {
        // 640×480 RGBA8 with a gralloc-padded 704-pixel stride: the row
        // pitch and allocation must follow the stride, never width×bpe.
        let (row, size) = desc_layout(&desc(640, 480, FORMAT_R8G8B8A8_UNORM, 704)).unwrap();
        assert_eq!(row, 704 * 4);
        assert_eq!(size, 704 * 4 * 480);
        // RGBA16F: 8 bytes per texel.
        let (row, size) = desc_layout(&desc(160, 1920, FORMAT_R16G16B16A16_FLOAT, 160)).unwrap();
        assert_eq!(row, 160 * 8);
        assert_eq!(size, 160 * 8 * 1920);
    }

    #[test]
    fn desc_layout_rejects_unknown_and_overflow() {
        // Unknown format (e.g. a YUV camera format) has no linear layout.
        assert!(desc_layout(&desc(64, 64, 0x23, 64)).is_none());
        // stride×bpe×height overflowing usize must yield None, not wrap.
        assert!(desc_layout(&desc(
            u32::MAX,
            u32::MAX,
            FORMAT_R16G16B16A16_FLOAT,
            u32::MAX
        ))
        .is_none());
    }

    #[test]
    fn checked_shape_bytes_accepts_and_rejects() {
        assert_eq!(
            checked_shape_bytes::<u8>(&[480, 640, 4]).unwrap(),
            1_228_800
        );
        assert_eq!(checked_shape_bytes::<u16>(&[2, 3]).unwrap(), 12);
        // Empty product = 1 element (scalar convention).
        assert_eq!(checked_shape_bytes::<u8>(&[]).unwrap(), 1);
        // Element-product overflow and byte-size overflow both reject.
        assert!(checked_shape_bytes::<u8>(&[usize::MAX, 2]).is_err());
        assert!(checked_shape_bytes::<u64>(&[usize::MAX / 4]).is_err());
    }
}
