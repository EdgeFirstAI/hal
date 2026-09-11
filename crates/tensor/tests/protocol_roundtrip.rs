//! Consumer side of the cross-package tensor protocol:
//! [`TensorDyn::import_descriptor`] rebuilding a usable, *aliasing* tensor
//! from an [`TensorDesc`](edgefirst_tensor::TensorDesc).
//!
//! The aliasing tests are the ones that matter here: a consumer that copies
//! instead of aliasing would still pass a naive shape/dtype assertion while
//! defeating the entire point of the protocol. Every roundtrip test below
//! writes through the producer handle and reads back through the imported
//! one to prove the two see the same physical bytes.

use edgefirst_tensor::{
    Colorimetry, CpuAccess, Error, PixelFormat, Tensor, TensorDyn, TensorMapTrait, TensorMemory,
    TensorTrait,
};

/// Host has no usable dma-heap. The dynamic backend wraps the OS error in
/// `io::ErrorKind::Other` (`image_alloc: IoError(Os { kind: PermissionDenied })`),
/// so matching only `e.kind()` panics on GitHub-hosted runners.
#[cfg(target_os = "linux")]
fn platform_resource_absent(err: &Error) -> bool {
    match err {
        Error::IoError(e) => {
            matches!(
                e.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied
            ) || e.to_string().contains("Permission denied")
                || e.to_string().contains("No such file or directory")
        }
        Error::NotImplemented(msg) => {
            msg.contains("Permission denied") || msg.contains("errno 13") || msg.contains("errno 2")
        }
        _ => false,
    }
}

#[test]
fn host_roundtrip_sees_the_same_bytes() {
    let t = Tensor::<u8>::image(
        64,
        32,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    {
        let mut m = t.map_write().unwrap();
        m.as_mut_slice()[0] = 0xAB;
    }

    // `pin_host` must be taken before `t` moves into the TensorDyn wrapper.
    let pin = t.pin_host(CpuAccess::ReadWrite).unwrap();
    let dyn_t = TensorDyn::from(t);
    let desc = dyn_t.descriptor_pinned(Some(&pin));

    let imported = TensorDyn::import_descriptor(&desc).unwrap();
    assert_eq!(imported.shape(), dyn_t.shape());
    assert_eq!(imported.dtype(), dyn_t.dtype());
    assert_eq!(imported.format(), dyn_t.format());

    let m = imported.as_u8().unwrap().map_read().unwrap();
    assert_eq!(m.as_slice()[0], 0xAB, "import must alias, not copy");
}

#[test]
fn planar_format_round_trips_through_import() {
    // PlanarRgb.to_fourcc() == 0 -- the same sentinel a non-image tensor
    // uses -- so this only round-trips if import_descriptor prefers
    // format over fourcc, per format_from_code(desc.format) with a
    // fourcc fallback only when format is NONE.
    let t = Tensor::<u8>::image(
        640,
        480,
        PixelFormat::PlanarRgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    let pin = t.pin_host(CpuAccess::ReadWrite).unwrap();
    let dyn_t = TensorDyn::from(t);
    let desc = dyn_t.descriptor_pinned(Some(&pin));
    assert_eq!(
        desc.fourcc, 0,
        "precondition: fourcc cannot express PlanarRgb"
    );
    assert_ne!(
        desc.format, 0,
        "precondition: format carries the real format"
    );

    let imported = TensorDyn::import_descriptor(&desc).unwrap();
    assert_eq!(
        imported.format(),
        Some(PixelFormat::PlanarRgb),
        "must round-trip via format, not the fourcc==0 sentinel"
    );
    assert_eq!(imported.shape(), dyn_t.shape());
}

#[test]
fn rejects_unknown_version() {
    let t = Tensor::<u8>::new(&[4], None, None).unwrap();
    let mut desc = TensorDyn::from(t).descriptor();
    desc.version = 9999;
    let err = TensorDyn::import_descriptor(&desc).unwrap_err();
    assert!(
        matches!(err, Error::NotImplemented(_)),
        "expected NotImplemented, got {err:?}"
    );
}

#[test]
fn rejects_unknown_dtype_code() {
    let t = Tensor::<u8>::new(&[4], None, None).unwrap();
    let mut desc = TensorDyn::from(t).descriptor();
    desc.dtype = 9999;
    let err = TensorDyn::import_descriptor(&desc).unwrap_err();
    assert!(
        matches!(err, Error::NotImplemented(_)),
        "expected NotImplemented, got {err:?}"
    );
}

#[test]
fn colorimetry_round_trips_through_import() {
    // Task 10b, defect A: `descriptor_pinned` packs colorimetry, but until
    // fixed `import_descriptor` never unpacked it back onto the imported
    // tensor -- a decode through the cross-package path left the caller
    // unable to see BT.601/full-range JFIF tagging, silently degrading
    // downstream colour conversion accuracy.
    let t = Tensor::<u8>::image(
        16,
        16,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    let pin = t.pin_host(CpuAccess::ReadWrite).unwrap();
    let dyn_t = TensorDyn::from(t).with_colorimetry(Colorimetry::jfif());
    let desc = dyn_t.descriptor_pinned(Some(&pin));
    assert_ne!(desc.colorimetry, 0, "precondition: producer tagged jfif");

    let imported = TensorDyn::import_descriptor(&desc).unwrap();
    assert_eq!(
        imported.colorimetry(),
        Some(Colorimetry::jfif()),
        "import must unpack desc.colorimetry onto the reconstructed tensor"
    );
}

#[test]
fn undefined_colorimetry_imports_as_none() {
    let t = Tensor::<u8>::image(
        16,
        16,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    let pin = t.pin_host(CpuAccess::ReadWrite).unwrap();
    let desc = TensorDyn::from(t).descriptor_pinned(Some(&pin));
    assert_eq!(desc.colorimetry, 0);
    let imported = TensorDyn::import_descriptor(&desc).unwrap();
    assert_eq!(imported.colorimetry(), None);
}

#[test]
fn host_import_inherits_the_producers_capacity_headroom() {
    // Task 10b, defect B. NV12 at an odd width pads the row stride (even,
    // then 64-byte aligned) beyond shape.product(): a real allocation gap
    // between `capacity_bytes()` and the logical shape's byte size, exactly
    // like a decoder's MCU-aligned write headroom. Before the fix, a `HOST`
    // import had no way to learn about that gap -- the reconstructed tensor's
    // capacity was clamped to exactly the declared shape, so writing the
    // producer's own padded stride into it would have run out of bounds.
    let t = Tensor::<u8>::image(
        295,
        175,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    let producer_capacity = t.capacity_bytes();
    let producer_shape_bytes = t.size();
    assert!(
        producer_capacity > producer_shape_bytes,
        "precondition: NV12 at an odd width must pad the allocation beyond \
         the logical shape (capacity={producer_capacity}, shape_bytes={producer_shape_bytes})"
    );

    let pin = t.pin_host(CpuAccess::ReadWrite).unwrap();
    let dyn_t = TensorDyn::from(t);
    let desc = dyn_t.descriptor_pinned(Some(&pin));
    assert_eq!(desc.capacity, producer_capacity as u64);

    let imported = TensorDyn::import_descriptor(&desc).unwrap();
    assert_eq!(
        imported.capacity_bytes(),
        producer_capacity,
        "the imported alias must inherit the producer's real headroom, not \
         just the declared shape's byte size"
    );

    // Reconfiguring within that headroom (the decode-time `configure_image`
    // pool-reuse path) must succeed -- this is exactly what previously
    // returned `Error::InsufficientCapacity` even though the producer's real
    // allocation had room, because the import discarded the padding.
    let mut imported = imported;
    imported
        .configure_image(295, 175, PixelFormat::Nv12)
        .expect("configure_image must fit within the producer's real capacity");
}

#[test]
fn host_import_preserves_the_producers_row_stride_for_pool_reuse() {
    // Task 10b, defect B (the other half): `configure_image`'s pool-reuse
    // logic only *prefers* keeping a stride when one is already recorded on
    // the tensor (`prior_stride`). Capacity headroom alone is not enough --
    // without restoring the producer's stride on import, a reconfigure to a
    // smaller image recomputes a tighter, un-padded stride instead of
    // preserving the producer's real (wider) allocation pitch, corrupting
    // row addressing for any consumer that reads rows at the reported
    // stride. This mirrors `test_decode_image_from_bytes` in the Python
    // suite (a 1920-wide pool tensor decoding a 1280-wide image must still
    // report row_stride=1920, not 1280).
    let t = Tensor::<u8>::image(
        1920,
        1080,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    assert_eq!(
        t.row_stride(),
        Some(1920),
        "precondition: semi-planar always records it"
    );

    let pin = t.pin_host(CpuAccess::ReadWrite).unwrap();
    let dyn_t = TensorDyn::from(t);
    let desc = dyn_t.descriptor_pinned(Some(&pin));

    let mut imported = TensorDyn::import_descriptor(&desc).unwrap();
    imported
        .configure_image(1280, 720, PixelFormat::Nv12)
        .expect("fits within capacity");
    assert_eq!(
        imported.effective_row_stride(),
        Some(1920),
        "must preserve the producer's pool-sized pitch, not recompute a \
         tighter one that only fits the smaller image"
    );
}

#[test]
fn host_import_rejects_a_row_stride_that_would_overflow_capacity() {
    // `set_row_stride` validates only a minimum (`stride >= min_stride`) --
    // it does no size validation by design, since it is pure layout
    // metadata for callers who never touch a real allocation (see
    // `strides_follow_row_stride_not_shape` in `tests/protocol.rs`). A
    // descriptor crossing the capsule boundary is untrusted input, though:
    // import_descriptor must guard `stride_bytes * rows <= capacity` itself
    // rather than trusting an oversized stride just because `set_row_stride`
    // accepted it.
    let t = Tensor::<u8>::image(
        16,
        16,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    let pin = t.pin_host(CpuAccess::ReadWrite).unwrap();
    let dyn_t = TensorDyn::from(t);
    let mut desc = dyn_t.descriptor_pinned(Some(&pin));
    // Claim a row pitch that, times the row count, wildly exceeds the real
    // 16×16×3-byte capacity -- as if a corrupt or hostile producer sent it.
    let malicious_stride = 1_000_000i64;
    desc.strides[0] = malicious_stride;

    let imported = TensorDyn::import_descriptor(&desc).unwrap();
    assert_ne!(
        imported.row_stride(),
        Some(malicious_stride as usize),
        "an oversized stride must not be trusted just because set_row_stride's \
         own validation is a minimum-only check"
    );
}

#[test]
fn host_import_without_pin_fails() {
    // A bare (unpinned) descriptor is descriptive-only: ptr is null. Importing
    // it must fail loudly rather than dereference a null address.
    let t = Tensor::<u8>::new(&[4], Some(TensorMemory::Mem), None).unwrap();
    let desc = TensorDyn::from(t).descriptor();
    let err = TensorDyn::import_descriptor(&desc).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// dma-buf (Linux only): not testable on this platform.
//
// This is a real, load-bearing gap, not an oversight -- dma-buf import
// (TensorDyn::import_descriptor's kind::DMABUF arm) has no coverage on any
// non-Linux CI lane. libtest swallows println! for a passing test, so this
// says so on stderr where it cannot be missed.
// ---------------------------------------------------------------------------
#[cfg(not(target_os = "linux"))]
#[test]
fn dmabuf_import_has_no_coverage_off_linux() {
    use std::io::Write;
    let _ = writeln!(
        std::io::stderr(),
        "SKIPPED: TensorDyn::import_descriptor's kind::DMABUF arm is untested on \
         this platform (dma-buf is Linux-only). No aliasing coverage for the \
         dma-buf import path outside a Linux run."
    );
}

#[cfg(target_os = "linux")]
#[test]
fn dmabuf_roundtrip_sees_the_same_bytes() {
    let t = match Tensor::<u8>::image(
        64,
        32,
        PixelFormat::Rgb,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    ) {
        Ok(t) => t,
        // No DMA-BUF heap on this platform (e.g. Jetson Orin, which uses
        // nvmap and ships with CONFIG_DMABUF_HEAPS unset) -- a capability
        // gap, not a regression. Two error kinds mean "no heap here":
        // `NotFound`, when the device node does not exist at all, and
        // `PermissionDenied`, when it exists but this user cannot open it
        // (a stock x86 desktop ships /dev/dma_heap/system as 0600 root).
        // A node you cannot open is exactly as unavailable as one that is
        // absent, and panicking on it reports a host-configuration fact as
        // a code failure -- which then has to be re-diagnosed on every
        // on-target run. Any other error kind still fails this test below
        // via the match's final arm.
        Err(e) if platform_resource_absent(&e) => {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "SKIPPED: dmabuf_roundtrip_sees_the_same_bytes -- this platform has no \
                 DMA-BUF heap (Tensor::image(.., TensorMemory::DmaBuf, ..) returned \
                 NotFound); dma-buf allocation is unavailable here, not broken."
            );
            return;
        }
        Err(e) => panic!("alloc dma: {e:?}"),
    };
    {
        let mut m = t.map_write().unwrap();
        m.as_mut_slice()[0] = 0xEF;
    }
    let dyn_t = TensorDyn::from(t);
    let desc = dyn_t.descriptor(); // handle carries the dma-buf fd; no pin needed
    assert_eq!(desc.kind, edgefirst_tensor::tensor_kind::DMABUF);

    let imported = TensorDyn::import_descriptor(&desc).unwrap();
    assert_eq!(imported.shape(), dyn_t.shape());
    let m = imported.as_u8().unwrap().map_read().unwrap();
    assert_eq!(m.as_slice()[0], 0xEF, "dma-buf import must alias, not copy");
}

#[cfg(target_os = "linux")]
#[test]
fn imported_dmabuf_with_a_recorded_stride_is_still_cpu_mappable() {
    // Task 10b follow-up (the flagship-pipeline regression): `configure_image`
    // (called from inside a JPEG decode, `crates/codec/src/jpeg/mod.rs`)
    // unconditionally records a row pitch for any `Dma`-backed destination,
    // self-allocated or not. An imported DMA-BUF crossing the capsule
    // protocol -- `decode_file_into` writing into an `edgefirst.image`
    // `ImageProcessor.create_image()` destination from `edgefirst.codec`,
    // the documented cross-package pipeline -- used to fail the moment the
    // decoder's MCU writer tried to `map_write()` it, because `Tensor::map`
    // rejected *any* strided imported DMA-BUF outright. Fixed by bounds-
    // checking a strided imported DMA-BUF against its real (`fstat`-derived)
    // `buf_size` instead of rejecting it unconditionally -- the same check
    // a self-allocated Dma tensor already relied on for its own soundness.
    //
    // This reproduces the mechanism directly (`from_fd` + `set_row_stride`,
    // mirroring what `configure_image` does), without going through the
    // Python packages: mint a self-allocated Dma tensor, import an *aliased*
    // copy of its own fd (so `is_imported = true`), record a stride on the
    // import the way `configure_image` would, and confirm the write through
    // it both succeeds and is visible through the original owner.
    let owner = match Tensor::<u8>::image(
        64,
        32,
        PixelFormat::Rgb,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    ) {
        Ok(t) => t,
        Err(e) if platform_resource_absent(&e) => {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "SKIPPED: imported_dmabuf_with_a_recorded_stride_is_still_cpu_mappable -- \
                 this platform has no DMA-BUF heap; dma-buf allocation is unavailable \
                 here, not broken."
            );
            return;
        }
        Err(e) => panic!("alloc dma: {e:?}"),
    };

    let fd = TensorTrait::clone_fd(&owner).expect("clone dma-buf fd");
    let mut imported = Tensor::<u8>::from_fd(fd, &[32, 64, 3], None).expect("import via from_fd");
    imported.set_format(PixelFormat::Rgb).expect("set_format");
    // What `configure_image` would compute for this exact width/format:
    // width * channels * elem, already a whole number and (here) already
    // 64-aligned -- the ordinary case this regression broke.
    imported
        .set_row_stride(64 * 3)
        .expect("set_row_stride: a HAL-computed pitch for the import's own format/width");

    {
        let mut m = imported.map_write().expect(
            "a HAL-computed, capacity-bounded stride must not be rejected on an imported \
             DMA-BUF -- that is exactly the write path a cross-package decode takes",
        );
        m.as_mut_slice()[0] = 0xAB;
    }

    let m = owner
        .map_read()
        .expect("read back through the owning tensor");
    assert_eq!(
        m.as_slice()[0],
        0xAB,
        "write through the imported, now-strided alias must be visible to the owner"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn dmabuf_import_preserves_the_producers_row_stride_for_pool_reuse() {
    // Task 10b follow-up (confirmed on rpi5-hailo, not just anticipated):
    // `Tensor::image()` auto-selects `Dma` when a DMA-BUF heap is available
    // (the common case on Linux), so the exact `HOST` pool-reuse scenario
    // `host_import_preserves_the_producers_row_stride_for_pool_reuse` covers
    // also happens routinely on `Dma`. Before extending the stride-restore
    // gate past `HOST`, a decode into an oversized DMA-BUF-backed
    // destination crossing the capsule protocol recomputed a tighter stride
    // for the smaller decoded image instead of keeping the pool's true
    // (wider) pitch -- silent misalignment for any GPU consumer reading at
    // the buffer's real physical stride. Mirrors the `HOST` test exactly,
    // for `TensorMemory::DmaBuf`.
    let t = match Tensor::<u8>::image(
        1920,
        1080,
        PixelFormat::Nv12,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    ) {
        Ok(t) => t,
        Err(e) if platform_resource_absent(&e) => {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "SKIPPED: dmabuf_import_preserves_the_producers_row_stride_for_pool_reuse -- \
                 this platform has no DMA-BUF heap; dma-buf allocation is unavailable \
                 here, not broken."
            );
            return;
        }
        Err(e) => panic!("alloc dma: {e:?}"),
    };
    assert_eq!(
        t.row_stride(),
        Some(1920),
        "precondition: semi-planar always records it"
    );

    let dyn_t = TensorDyn::from(t);
    let desc = dyn_t.descriptor(); // handle carries the dma-buf fd; no pin needed

    let mut imported = TensorDyn::import_descriptor(&desc).unwrap();
    imported
        .configure_image(1280, 720, PixelFormat::Nv12)
        .expect("fits within capacity");
    assert_eq!(
        imported.effective_row_stride(),
        Some(1920),
        "must preserve the producer's pool-sized pitch, not recompute a tighter one \
         that only fits the smaller image"
    );
}

// ---------------------------------------------------------------------------
// IOSurface (macOS/iOS): TensorMemory::DmaBuf is IOSurface-backed there, so
// this platform's own "Dma" arm gets real aliasing coverage.
// ---------------------------------------------------------------------------
#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
fn iosurface_roundtrip_sees_the_same_bytes() {
    let t = Tensor::<u8>::image(
        64,
        32,
        PixelFormat::Rgb,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    {
        let mut m = t.map_write().unwrap();
        m.as_mut_slice()[0] = 0xCD;
    }
    let dyn_t = TensorDyn::from(t);
    let desc = dyn_t.descriptor(); // handle carries the IOSurfaceID; no pin needed
    assert_eq!(desc.kind, edgefirst_tensor::tensor_kind::IOSURFACE);

    let imported = TensorDyn::import_descriptor(&desc).unwrap();
    assert_eq!(imported.shape(), dyn_t.shape());
    // `restore_imported_row_stride` lists the kinds whose pitch is a
    // property of the shared buffer itself, and since issue #161 an
    // IOSurface is one of them. This used to assert the opposite, guarding
    // a Linux DMA-BUF CPU-map restriction that has since been lifted at the
    // map site (`imported_dmabuf_with_a_recorded_stride_is_still_cpu_mappable`
    // pins that). The stride must now survive the round trip, and the map
    // below must still alias with it recorded. At 64 RGB texels the pitch
    // equals the tight row, so this only proves the restore ran; the
    // padded-pitch case is
    // `descriptor_import_restores_the_iosurface_pitch_onto_a_view` in the
    // `iosurface` module.
    assert_eq!(
        imported.row_stride(),
        Some(desc.strides()[0] as usize),
        "an IOSurface import must carry the producer's pitch \
         (restore_imported_row_stride)"
    );
    let m = imported.as_u8().unwrap().map_read().unwrap();
    assert_eq!(
        m.as_slice()[0],
        0xCD,
        "IOSurface import must alias, not copy"
    );
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
fn iosurface_import_rejects_dead_id() {
    let t = Tensor::<u8>::image(
        16,
        16,
        PixelFormat::Grey,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    let mut desc = TensorDyn::from(t).descriptor();
    desc.handle = 0x7FFF_FFFF; // implausible id: no live surface should hold it
    let err = TensorDyn::import_descriptor(&desc).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}

#[test]
fn import_refuses_a_descriptor_advertising_an_unwaitable_fence() {
    // `sync` is reserved in this build, so a producer that sets
    // SYNC_PRESENT is describing a fence nothing here can wait on. Importing
    // anyway would alias memory the producer's device may still be writing,
    // and the resulting corruption would be timing-dependent -- the worst
    // kind to diagnose. Refusing is the only honest option while the field
    // is reserved; this test is what stops a future change from quietly
    // relaxing it into "ignore the flag".
    let t = Tensor::<u8>::image(
        16,
        8,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    // `pin_host` before the move, so the descriptor carries a real address --
    // otherwise the import fails for an unrelated reason and this test would
    // pass without ever reaching the SYNC_PRESENT check.
    let pin = t.pin_host(CpuAccess::ReadWrite).unwrap();
    let dyn_t = TensorDyn::from(t);
    let mut desc = dyn_t.descriptor_pinned(Some(&pin));

    // Baseline: the untouched descriptor imports.
    TensorDyn::import_descriptor(&desc).expect("clean descriptor imports");

    desc.flags |= edgefirst_tensor::protocol::flags::SYNC_PRESENT;
    let err = TensorDyn::import_descriptor(&desc)
        .expect_err("a descriptor advertising SYNC_PRESENT must be refused");
    let msg = format!("{err}");
    assert!(
        msg.contains("SYNC_PRESENT"),
        "the error must name what it refused, got: {msg}"
    );
}

/// `D3D11_TEXTURE` is the one kind allowed to set `SYNC_PRESENT`, and it
/// carries the fence's NT handle in `ptr`. A descriptor that advertises a
/// fence value with a null `ptr` therefore names a completion nobody can wait
/// on, which is the same in-flight-write hazard rather than a missing
/// feature. Refused on every platform: the check is in the shared validation,
/// not in the Windows construction arm, so a non-Windows consumer refuses it
/// too instead of reporting the kind as unimplemented.
#[test]
fn import_refuses_a_d3d11_completion_with_no_fence_handle() {
    let t = Tensor::<u8>::image(
        16,
        8,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let pin = t.pin_host(CpuAccess::ReadWrite).unwrap();
    let dyn_t = TensorDyn::from(t);
    let mut desc = dyn_t.descriptor_pinned(Some(&pin));
    desc.kind = edgefirst_tensor::protocol::kind::D3D11_TEXTURE;
    desc.flags |= edgefirst_tensor::protocol::flags::SYNC_PRESENT;
    desc.sync = 7;
    desc.ptr = edgefirst_tensor::protocol::SendPtr(std::ptr::null_mut());
    let err = TensorDyn::import_descriptor(&desc)
        .expect_err("a completion with no fence to read it on must be refused");
    let msg = format!("{err}");
    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
    assert!(
        msg.contains("fence handle"),
        "the error must name what it refused, got: {msg}"
    );
}

// --- A PBO's descriptor pointer, on both backends -------------------------

/// Mock [`edgefirst_tensor::PboOps`] over a `Vec<u8>`, so this file can
/// build a PBO-backed tensor with no GL context. A sibling of the one in
/// `dynamic_primitives.rs`, reimplemented here for the same reason: an
/// integration test can only reach the crate's public API, and this file
/// runs on BOTH backends where that one is `dynamic`-only.
struct MockPboOps {
    storage: std::sync::Mutex<Vec<u8>>,
}

// SAFETY: the returned pointer addresses a `Vec<u8>` allocated once in
// `new` and never resized, so it stays valid for as long as the
// `MockPboOps` does -- which outlives every mapping handed out, since the
// tensor holds an `Arc` to it.
unsafe impl edgefirst_tensor::PboOps for MockPboOps {
    fn map_buffer(
        &self,
        _buffer_id: u32,
        size: usize,
    ) -> edgefirst_tensor::Result<edgefirst_tensor::PboMapping> {
        let storage = self.storage.lock().expect("lock");
        assert_eq!(storage.len(), size, "mock PBO size mismatch");
        Ok(edgefirst_tensor::PboMapping {
            ptr: storage.as_ptr() as *mut u8,
            size,
        })
    }

    fn unmap_buffer(&self, _buffer_id: u32) -> edgefirst_tensor::Result<()> {
        Ok(())
    }

    fn delete_buffer(&self, _buffer_id: u32) {}
}

fn pbo_backed(buffer_id: u32, bytes: usize, shape: &[usize]) -> TensorDyn {
    let ops = std::sync::Arc::new(MockPboOps {
        storage: std::sync::Mutex::new(vec![0u8; bytes]),
    }) as std::sync::Arc<dyn edgefirst_tensor::PboOps>;
    let pbo = edgefirst_tensor::PboTensor::<u8>::from_pbo(buffer_id, bytes, shape, None, ops)
        .expect("PboTensor::from_pbo");
    Tensor::<u8>::from_pbo(pbo)
        .expect("Tensor::from_pbo")
        .into()
}

/// A PBO-backed tensor's descriptor carries the PBO vtable pointer in
/// `ptr`, never a host address.
///
/// `TensorDesc::ptr` is filled from a pin *before* `pbo_vtable_ptr`
/// (`protocol.rs`'s `from_parts`), a precedence that is only safe because
/// the two are mutually exclusive: a PBO refuses `pin_host` on both
/// backends, so no PBO descriptor can carry a pin. That makes this a real
/// invariant rather than a restatement of the code -- swap the two arms and
/// nothing changes today, but the day a PBO acquires a pin the consumer
/// would open a host address as a vtable. Pinning the *value* is what
/// catches that; pinning the ordering would not.
///
/// The consumer side is the reason it matters: `import_descriptor`'s
/// `kind::PBO` arm reads `ptr` as an `EfClientState` vtable and calls
/// through it. A host address there is a call through whatever bytes the
/// producer's pixels happen to spell.
#[test]
fn a_pbo_descriptor_carries_the_vtable_pointer_not_a_host_address() {
    let t = pbo_backed(77, 4 * 4, &[4, 4, 1]);
    let vtable = t
        .pbo_vtable_ptr()
        .expect("a PBO-backed tensor has a client-state vtable");
    assert!(!vtable.is_null(), "the vtable pointer must be real");

    let d = t.descriptor();
    assert_eq!(
        d.kind,
        edgefirst_tensor::tensor_kind::PBO,
        "a PBO reports the PBO kind"
    );
    assert_eq!(
        d.ptr.0 as *const std::ffi::c_void, vtable,
        "`ptr` is the PBO vtable, which is what `import_descriptor`'s PBO arm \
         calls through -- a host address here would be called as a vtable"
    );

    // The same descriptor, taken the way a capsule takes it: an explicit
    // (absent) pin. `descriptor_pinned(None)` is the `access=None` form the
    // converter uses, and it must reach the same answer.
    let d = t.descriptor_pinned(None);
    assert_eq!(
        d.ptr.0 as *const std::ffi::c_void, vtable,
        "the pinned-descriptor entry point agrees with the plain one when \
         there is no pin to prefer"
    );

    // And the precondition the precedence rests on: a PBO has no pin to
    // take, on either backend, so `pin` can never win the `or_else` above.
    assert!(
        matches!(
            t.pin_host(CpuAccess::Read),
            Err(edgefirst_tensor::Error::NotImplemented(_))
        ),
        "a PBO must refuse pin_host, or `ptr` could carry a host address"
    );
}

/// A PBO `view()` keeps its parent's pitch across the descriptor.
///
/// The sibling of `host_import_preserves_the_producers_row_stride_for_pool_reuse`
/// and its DMA-BUF twin, for the kind that was left out of
/// `restore_imported_row_stride`'s allowlist. A sub-view's recorded stride
/// is the PARENT's pitch -- that is how its rows are spaced -- so a rebuild
/// that recomputed a tight pitch from the window's own width put every row
/// after the first on the wrong columns. Silent: the shape, the dtype, the
/// buffer id and the offset all survive, and only the pixels are wrong.
///
/// A GL buffer has no pitch of its own to fall back on (it is bytes), so
/// unlike a D3D11 texture the producer's stride is the only description
/// there is. The exclusion was correct before Stage B for a reason that
/// stopped being true: `import_descriptor` had no `kind::PBO` arm at all,
/// so no PBO descriptor ever reached that function.
///
/// The narrow window is load-bearing: a full-width band's tight pitch and
/// its parent's pitch are the same number, so it passes either way.
#[test]
fn pbo_import_preserves_a_views_parent_pitch() {
    const W: usize = 64;
    const H: usize = 64;
    const SIDE: usize = 16;
    const X0: usize = 8;
    const Y0: usize = 8;

    let mut parent = pbo_backed(88, W * H * 3, &[H, W, 3]);
    parent
        .set_format(PixelFormat::Rgb)
        .expect("a packed RGB image");
    let parent_pitch = parent
        .effective_row_stride()
        .expect("a formatted image reports a pitch");
    assert_eq!(parent_pitch, W * 3, "precondition: the parent is tight");

    let view = parent
        .view(edgefirst_tensor::Region::new(X0, Y0, SIDE, SIDE))
        .expect("view a narrow window");
    assert_eq!(
        view.row_stride(),
        Some(parent_pitch),
        "precondition: a multi-row view records the parent's pitch, not its own \
         tight row"
    );

    let imported = TensorDyn::import_descriptor(&view.descriptor())
        .expect("a PBO descriptor imports through its client-state vtable");
    assert_eq!(
        imported.effective_row_stride(),
        Some(parent_pitch),
        "the rebuilt view must step rows by the parent pitch ({parent_pitch}); \
         {} is the window's own tight row, which shears every row after the \
         first (issue #162)",
        SIDE * 3
    );
}

/// `pin_host` is refused on a PBO by BOTH backends, with the same message.
///
/// `glMapBufferRange`'s address is valid only until `glUnmapBuffer`, so a
/// pin -- which promises an address that outlives the call -- has nothing
/// to hand back. The static backend refuses in `Tensor::pin_host`'s
/// `TensorStorage::Pbo` arm; the dynamic backend cannot forward to it,
/// because `ef_tensor_map` is `map`, not `pin_host`, and the library cannot
/// tell the two callers apart -- so the refusal is repeated in
/// `dynamic_backend.rs` and both arms read the same
/// `crate::pbo::PIN_HOST_PBO_REFUSAL`. Without the repeat, a PBO pinned on
/// one backend and refused on the other from the identical call, which is
/// precisely the divergence G13 cannot see.
///
/// This file, not `dynamic_primitives.rs`: only a test that COMPILES on
/// both backends can make a claim about both, and this one is in
/// `DYNAMIC_TEST_TARGETS` as well as the default static run.
///
/// The host half is not padding: it proves the refusal is keyed on the
/// storage kind rather than a `pin_host` that simply never works here.
#[test]
fn pin_host_is_refused_by_a_pbo_on_both_backends() {
    let host: TensorDyn = Tensor::<u8>::new(&[4, 4], Some(TensorMemory::Mem), None)
        .expect("host allocation")
        .into();
    host.pin_host(CpuAccess::Read)
        .expect("host memory has an address that outlives the call");

    let pbo_t = pbo_backed(9, 16, &[4, 4, 1]);
    match pbo_t.pin_host(CpuAccess::Read) {
        Err(Error::NotImplemented(msg)) => assert!(
            msg.contains("PBO"),
            "the refusal must name the backing it applies to, got: {msg}"
        ),
        other => panic!(
            "a PBO has no host address outside its own map guard, so pin_host \
             must refuse it on this backend exactly as it does on the other, got: {other:?}"
        ),
    }
}
