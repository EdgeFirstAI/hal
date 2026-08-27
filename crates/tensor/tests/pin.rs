//! Persistent host pinning (issue #134).

use edgefirst_tensor::{CpuAccess, Error, PixelFormat, Tensor, TensorMemory, TensorTrait};

// Used only by the DMA tests below. Importing these unconditionally makes them
// dead on macOS, which fails that lane under `-D warnings`.
#[cfg(target_os = "linux")]
use edgefirst_tensor::Region;
#[cfg(unix)]
use edgefirst_tensor::TensorMapTrait;

/// Skip marker that survives libtest's output capture. See `skip` below.
#[cfg(unix)]
fn skip_unix(why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "SKIPPED: {why}");
}

#[test]
fn pin_yields_a_stable_usable_address() {
    let t = Tensor::<u8>::image(
        64,
        32,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let pin = t.pin_host(CpuAccess::ReadWrite).expect("pin");
    assert!(!pin.as_ptr().is_null());
    assert_eq!(pin.len(), 64 * 32 * 3);
    assert!(!pin.is_empty());
}

#[test]
fn pin_does_not_borrow_the_tensor() {
    // The #134 blocker in its purest form: a guard that borrowed the tensor
    // could not coexist with the `&mut` that ImageProcessor::convert takes on
    // its destination every frame. Taking that &mut while a pin is live must
    // COMPILE -- that is the whole property, and it is checked at build time.
    let mut t = Tensor::<u8>::image(
        32,
        32,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let pin = t.pin_host(CpuAccess::ReadWrite).expect("pin");

    let borrowed: &mut Tensor<u8> = &mut t; // <- would not compile if the
    borrowed.set_logical_shape(&[3072]).unwrap(); //    pin borrowed the tensor

    assert!(
        !pin.as_ptr().is_null(),
        "pin remains usable after the mutation"
    );
    assert_eq!(pin.len(), 32 * 32 * 3);
}

#[test]
fn pin_survives_writes_through_it() {
    let t = Tensor::<u8>::image(
        8,
        8,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let pin = t.pin_host(CpuAccess::ReadWrite).expect("pin");
    unsafe {
        std::ptr::write_bytes(pin.as_mut_ptr(), 0xAB, pin.len());
        assert!(pin.as_slice().iter().all(|b| *b == 0xAB));
    }
}

#[test]
fn pin_outlives_the_tensor_via_its_keepalive() {
    let pin = {
        let t = Tensor::<u8>::image(
            16,
            16,
            PixelFormat::Grey,
            Some(TensorMemory::Mem),
            CpuAccess::ReadWrite,
        )
        .expect("alloc");
        let p = t.pin_host(CpuAccess::ReadWrite).expect("pin");
        unsafe { std::ptr::write_bytes(p.as_mut_ptr(), 0x5A, p.len()) };
        p
    }; // tensor dropped here; the Arc keepalive must hold the allocation
    unsafe { assert!(pin.as_slice().iter().all(|b| *b == 0x5A)) };
}

#[test]
fn cpu_access_none_is_an_error_not_a_silent_downgrade() {
    let t = Tensor::<u8>::image(
        8,
        8,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    match t.pin_host(CpuAccess::None) {
        Err(Error::InvalidArgument(m)) => assert!(m.contains("CpuAccess::None")),
        other => panic!("expected InvalidArgument, got {other:?}"),
    }
}

#[test]
fn alignment_is_queryable_for_tflite() {
    // TFLite requires 64-byte alignment unless the caller opts out.
    let t = Tensor::<u8>::image(
        64,
        64,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let pin = t.pin_host(CpuAccess::ReadWrite).expect("pin");
    assert!(pin.alignment() > 0);
    assert!(pin.alignment().is_power_of_two());
}

// --- sync API -------------------------------------------------------------

#[test]
fn sync_is_a_documented_noop_on_coherent_backends() {
    // `sync_for_cpu`'s rustdoc table says Mem and Shm are no-ops and NOTHING
    // ELSE is: DMA-BUF issues an ioctl, IOSurface locks/unlocks, and
    // AHardwareBuffer and PBO report NotImplemented because their platform
    // fuses addressing with coherency. This asserts the no-op half of that
    // table for BOTH rows -- Shm is claimed by the docs and was previously
    // untested, so the claim rested on reading the match arm.
    for memory in [
        TensorMemory::Mem,
        #[cfg(unix)]
        TensorMemory::Shm,
    ] {
        let t = Tensor::<u8>::image(8, 8, PixelFormat::Grey, Some(memory), CpuAccess::ReadWrite)
            .unwrap_or_else(|e| panic!("alloc {memory:?}: {e}"));
        for access in [CpuAccess::Read, CpuAccess::Write, CpuAccess::ReadWrite] {
            t.sync_for_cpu(access)
                .unwrap_or_else(|e| panic!("{memory:?} sync_for_cpu({access:?}) must no-op: {e}"));
            t.sync_for_device(access).unwrap_or_else(|e| {
                panic!("{memory:?} sync_for_device({access:?}) must no-op: {e}")
            });
        }
    }
}

#[test]
fn sync_rejects_an_access_that_describes_nothing() {
    let t = Tensor::<u8>::image(
        8,
        8,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    assert!(matches!(
        t.sync_for_cpu(CpuAccess::None),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        t.sync_for_device(CpuAccess::None),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn pin_then_bracket_then_mutate_is_the_intended_frame_loop() {
    // The whole #134 shape in one test: pin once, then each "frame" bracket a
    // CPU read and still take &mut on the tensor.
    let mut t = Tensor::<u8>::image(
        16,
        16,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let pin = t.pin_host(CpuAccess::ReadWrite).expect("pin");
    for frame in 0u8..4 {
        t.sync_for_cpu(CpuAccess::Write).unwrap();
        unsafe { std::ptr::write_bytes(pin.as_mut_ptr(), frame, pin.len()) };
        t.sync_for_device(CpuAccess::Write).unwrap();

        let borrowed: &mut Tensor<u8> = &mut t; // stands in for convert(&mut dst)
        borrowed.set_logical_shape(&[256]).unwrap();

        unsafe { assert!(pin.as_slice().iter().all(|b| *b == frame)) };
    }
}

// --- DMA (Linux) ----------------------------------------------------------

/// Report a skip so it survives to the log.
///
/// libtest captures `println!`/`eprintln!` and discards it for **passing**
/// tests, so a test that skips and returns is indistinguishable from one that
/// did the work — it just prints `... ok`. Writing to `std::io::stderr()`
/// directly bypasses that capture, which is the same trick the unit tests in
/// `lib.rs` use for their skip warnings. Without this a board with no DMA heap
/// silently reports a full green pin suite.
#[cfg(target_os = "linux")]
fn skip(why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "SKIPPED (no DMA): {why}");
}

#[cfg(target_os = "linux")]
#[test]
fn dma_pin_of_a_subview_is_offset_adjusted() {
    // Requirement 2 of #134: a pin must hand back the *view's* address, not the
    // raw mmap base. A freshly allocated tensor has mmap_offset == 0, so it
    // cannot show this -- only a subview can. Proven by writing through the
    // pin and reading the bytes back through the PARENT at the same offset.
    if !edgefirst_tensor::is_dma_available() {
        skip("no DMA heap on this host");
        return;
    }
    // `view()` is the public crop primitive; it delegates to the same offset
    // machinery. A bottom-half view at x=0 avoids the documented bottom-right
    // multi-row limitation while still producing a non-zero mmap_offset.
    let parent = Tensor::<u8>::image(
        64,
        64,
        PixelFormat::Rgb,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .expect("alloc parent");
    {
        let mut g = parent.map().expect("zero parent");
        g.as_mut_slice().fill(0u8);
    }
    let stride = parent.effective_row_stride().expect("stride");
    let split = 32 * stride; // byte offset of the view's first row

    let child = parent.view(Region::new(0, 32, 64, 32)).expect("view");
    let pin = child.pin_host(CpuAccess::ReadWrite).expect("pin view");

    child.sync_for_cpu(CpuAccess::Write).expect("sync start");
    unsafe { std::ptr::write_bytes(pin.as_mut_ptr(), 0xC3, split) };
    child.sync_for_device(CpuAccess::Write).expect("sync end");

    let guard = parent.map().expect("map parent");
    let bytes = guard.as_slice();
    assert!(
        bytes[..split].iter().all(|b| *b == 0),
        "pin wrote before the view's offset -- it handed back the raw mmap base"
    );
    assert!(
        bytes[split..split * 2].iter().all(|b| *b == 0xC3),
        "pin did not write at the view's offset"
    );
}

#[cfg(target_os = "linux")]
#[test]
fn dma_pin_survives_guard_drops() {
    // Issue #134 criterion 1: the address must stay valid after every map
    // guard has dropped. Skips where no DMA heap is available.
    if !edgefirst_tensor::is_dma_available() {
        skip("no DMA heap on this host");
        return;
    }
    let t = Tensor::<u8>::image(
        64,
        64,
        PixelFormat::Rgb,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let pin = t.pin_host(CpuAccess::ReadWrite).expect("pin DMA");
    assert!(!pin.as_ptr().is_null());

    // Take and drop a conventional map guard; the pin must be unaffected.
    {
        let _guard = t.map().expect("map");
    }
    t.sync_for_cpu(CpuAccess::Write).expect("sync start");
    unsafe { std::ptr::write_bytes(pin.as_mut_ptr(), 0x7E, 64) };
    t.sync_for_device(CpuAccess::Write).expect("sync end");

    t.sync_for_cpu(CpuAccess::Read).expect("sync start");
    unsafe { assert!(pin.as_slice()[..64].iter().all(|b| *b == 0x7E)) };
    t.sync_for_device(CpuAccess::Read).expect("sync end");
}

#[cfg(target_os = "linux")]
#[test]
fn dma_sync_brackets_round_trip_in_every_direction() {
    if !edgefirst_tensor::is_dma_available() {
        skip("no DMA heap on this host");
        return;
    }
    let t = Tensor::<u8>::image(
        32,
        32,
        PixelFormat::Grey,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    // The direction passed to sync_for_device MUST match sync_for_cpu; the
    // kernel skips a writeback or an invalidate based on it.
    for access in [CpuAccess::Read, CpuAccess::Write, CpuAccess::ReadWrite] {
        t.sync_for_cpu(access).expect("start");
        t.sync_for_device(access).expect("end");
    }
}

// --- sync must never silently claim success -------------------------------

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
fn iosurface_pin_is_stable_and_sync_brackets_lock_unlock() {
    // IOSurface is the reference implementation this whole design generalises:
    // it already separates the surface's ADDRESS (IOSurfaceGetBaseAddress,
    // stable for the surface's life) from the CPU ACCESS WINDOW
    // (IOSurfaceLock/Unlock). So the pin takes the address and the sync calls
    // take the lock, rather than fusing both into a map guard.
    //
    // A raw tensor, not an image: IOSurface rounds its row pitch up to 64
    // bytes, and a contiguous write through the pin would not respect that
    // stride. Pitch handling belongs to the map guard.
    let t = Tensor::<u8>::new(&[1024], Some(TensorMemory::DmaBuf), None).expect("alloc iosurface");

    let pin = t.pin_host(CpuAccess::ReadWrite).expect("pin iosurface");
    let addr = pin.as_ptr();
    assert!(!addr.is_null());

    // A conventional guard takes and releases its own lock; the pinned address
    // must be unaffected -- that is the property map() could not offer.
    {
        let _g = t.map().expect("map");
    }
    assert_eq!(pin.as_ptr(), addr, "pin address moved when a guard dropped");

    // Write bracket. The lock options are derived from `access`, and Apple
    // requires the SAME options at unlock -- which is exactly why the API
    // makes the caller name the direction on both halves.
    t.sync_for_cpu(CpuAccess::Write).expect("IOSurfaceLock");
    unsafe { std::ptr::write_bytes(pin.as_mut_ptr(), 0x5E, pin.len()) };
    t.sync_for_device(CpuAccess::Write)
        .expect("IOSurfaceUnlock");

    // Read bracket, which takes the read-only lock (it skips the cache flush
    // at unlock, so a mismatched pairing here would be a real bug).
    t.sync_for_cpu(CpuAccess::Read)
        .expect("IOSurfaceLock read-only");
    unsafe { assert!(pin.as_slice().iter().all(|b| *b == 0x5E)) };
    t.sync_for_device(CpuAccess::Read)
        .expect("IOSurfaceUnlock read-only");
}

// --- SHM ------------------------------------------------------------------

#[cfg(unix)]
#[test]
fn shm_pin_is_stable_across_guard_drops() {
    // SHM is plain CPU memory behind an fd: the mapping can outlive a guard,
    // and sync owes nothing. Skips where allocation is unavailable (Android's
    // bionic has no POSIX shared memory; see shm.rs module docs).
    if !edgefirst_tensor::is_shm_available() {
        skip_unix("SHM allocation unavailable on this platform");
        return;
    }
    let t = Tensor::<u8>::image(
        32,
        32,
        PixelFormat::Grey,
        Some(TensorMemory::Shm),
        CpuAccess::ReadWrite,
    )
    .expect("alloc shm");

    let pin = t.pin_host(CpuAccess::ReadWrite).expect("pin shm");
    let addr = pin.as_ptr();
    assert_eq!(pin.len(), 32 * 32);

    // A conventional guard comes and goes; the pinned address must not move.
    {
        let _g = t.map().expect("map");
    }
    assert_eq!(pin.as_ptr(), addr, "pin address moved when a guard dropped");

    t.sync_for_cpu(CpuAccess::Write)
        .expect("shm sync is a no-op");
    unsafe { std::ptr::write_bytes(pin.as_mut_ptr(), 0x3C, pin.len()) };
    t.sync_for_device(CpuAccess::Write)
        .expect("shm sync is a no-op");

    // Read back through a fresh guard: the pin and the guard must address the
    // same bytes, which is what makes the pin a real view of the tensor.
    let g = t.map().expect("remap");
    assert!(
        g.as_slice().iter().all(|b| *b == 0x3C),
        "pin wrote somewhere the tensor cannot see"
    );
}

// --- AHardwareBuffer (Android) --------------------------------------------

#[cfg(target_os = "android")]
#[test]
fn ahardwarebuffer_reports_why_it_cannot_pin() {
    // Not a TODO: the NDK exposes allocate/acquire/release/describe/lock/
    // unlock and nothing else, so there is no CPU address outside the lock. A
    // "persistent" address would mean holding AHardwareBuffer_lock for the
    // pin's lifetime, which both blocks the GPU from the buffer and makes any
    // further lock undefined -- the NDK does not document refcounted nesting.
    //
    // The error must explain that, because "not implemented" would invite
    // someone to implement it unsafely.
    // An IMAGE, not a raw byte-bag: the emulator's allocator rejects
    // FORMAT_BLOB (AHardwareBuffer_allocate rc=0x80000000 for 1024x1), while
    // image formats allocate fine. Skip if even that is unavailable, so a
    // device without a usable gralloc reports a skip rather than a failure.
    let t = match Tensor::<u8>::image(
        64,
        64,
        PixelFormat::Rgba,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    ) {
        Ok(t) => t,
        Err(e) => {
            skip_unix(&format!("AHardwareBuffer allocation unavailable: {e}"));
            return;
        }
    };

    match t.pin_host(CpuAccess::ReadWrite) {
        Err(Error::NotImplemented(m)) => {
            assert!(
                m.contains("AHardwareBuffer_lock"),
                "must name the constraint: {m}"
            );
            assert!(m.contains("map()"), "must point at the supported API: {m}");
        }
        other => panic!("expected an explained NotImplemented, got {other:?}"),
    }
    // Sync is unavailable for the same reason, and must not silently succeed.
    assert!(matches!(
        t.sync_for_cpu(CpuAccess::Read),
        Err(Error::NotImplemented(_))
    ));
    assert!(matches!(
        t.sync_for_device(CpuAccess::Read),
        Err(Error::NotImplemented(_))
    ));

    // The supported path must still work.
    let g = t.map().expect("map must work where pin cannot");
    assert!(!g.as_slice().is_empty(), "map must expose the buffer");
}

// --- re-layout while pinned -----------------------------------------------

#[test]
fn relayout_while_pinned_is_safe_and_the_pin_is_a_snapshot() {
    // Plan 2b Task 3 assumed a live pin had to BLOCK re-layout, on the theory
    // that a remap would move the address underneath it. No backend does that:
    // set_logical_shape only rewrites `shape` after a capacity check, and each
    // pin owns its own mapping (Mem an Arc over a never-resized allocation,
    // SHM/DMA their own mmap, IOSurface the retained surface). So a pin stays
    // valid across re-layout; what it does NOT do is follow the new shape.
    //
    // That snapshot semantic is the thing worth pinning down, because the
    // alternative reading -- "len() tracks the tensor" -- would be a silent
    // buffer-overrun waiting to happen if it were ever true.
    let mut t = Tensor::<u8>::image(
        64,
        64,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let pin = t.pin_host(CpuAccess::ReadWrite).expect("pin");
    let addr = pin.as_ptr();
    let len = pin.len();
    assert_eq!(len, 64 * 64);
    unsafe { std::ptr::write_bytes(pin.as_mut_ptr(), 0xD4, len) };

    // Shrink the logical shape. Allowed: it fits inside the same allocation.
    t.configure_image(32, 32, PixelFormat::Grey)
        .expect("re-layout within capacity");

    // The pin is unmoved and still readable for its original extent.
    assert_eq!(pin.as_ptr(), addr, "re-layout moved a pinned address");
    assert_eq!(
        pin.len(),
        len,
        "pin length must be a snapshot, not a tracker"
    );
    unsafe { assert!(pin.as_slice().iter().all(|b| *b == 0xD4)) };

    // Growing beyond capacity is refused, which is what keeps the snapshot
    // inside the allocation in every case.
    assert!(t.set_logical_shape(&[64 * 64 * 4]).is_err());
}

#[cfg(target_os = "linux")]
#[test]
fn dma_relayout_while_pinned_does_not_move_the_mapping() {
    // The Mem case is the easy one -- its base pointer is trivially stable.
    // DMA is where the "a remap moves the address" theory would have to hold
    // if it held anywhere, since the pin owns a real mmap of a heap buffer.
    if !edgefirst_tensor::is_dma_available() {
        skip("no DMA heap on this host");
        return;
    }
    let mut t = Tensor::<u8>::image(
        64,
        64,
        PixelFormat::Grey,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .expect("alloc dma");
    let pin = t.pin_host(CpuAccess::ReadWrite).expect("pin dma");
    let addr = pin.as_ptr();
    let len = pin.len();

    t.sync_for_cpu(CpuAccess::Write).expect("sync start");
    unsafe { std::ptr::write_bytes(pin.as_mut_ptr(), 0xB7, len) };
    t.sync_for_device(CpuAccess::Write).expect("sync end");

    t.configure_image(32, 32, PixelFormat::Grey)
        .expect("re-layout within capacity");

    assert_eq!(pin.as_ptr(), addr, "re-layout moved a pinned DMA mapping");
    assert_eq!(pin.len(), len, "pin length must be a snapshot");
    t.sync_for_cpu(CpuAccess::Read).expect("sync start");
    unsafe { assert!(pin.as_slice().iter().all(|b| *b == 0xB7)) };
    t.sync_for_device(CpuAccess::Read).expect("sync end");
}
