//! Persistent host pinning.

use edgefirst_tensor::{CpuAccess, Error, PixelFormat, Tensor, TensorMemory, TensorTrait};

// Used only by the DMA tests below.
#[cfg(target_os = "linux")]
use edgefirst_tensor::Region;
#[cfg(unix)]
use edgefirst_tensor::TensorMapTrait;

/// Allocation policy: a failure the platform probe did not predict fails the test.
#[cfg(target_os = "android")]
#[path = "support/outcome.rs"]
#[allow(dead_code)]
mod outcome;

/// Writes a skip marker directly to stderr.
#[cfg(unix)]
fn skip_unix(why: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {why}");
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
    // Taking `&mut` on the tensor while a pin is live compiles.
    let mut t = Tensor::<u8>::image(
        32,
        32,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let pin = t.pin_host(CpuAccess::ReadWrite).expect("pin");

    let borrowed: &mut Tensor<u8> = &mut t;
    borrowed.set_logical_shape(&[3072]).unwrap();

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
    }; // tensor dropped here
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

#[test]
fn alignment_of_a_float_pin_covers_the_element_and_divides_the_address() {
    let t = Tensor::<f32>::new(&[256], Some(TensorMemory::Mem), None).expect("alloc");
    let pin = t.pin_host(CpuAccess::Read).expect("pin");
    assert!(pin.alignment() >= std::mem::align_of::<f32>());
    assert_eq!(pin.as_ptr() as usize % pin.alignment(), 0);
    assert_eq!(pin.len(), 256 * 4);
    assert!(!pin.is_empty());
}

// --- sync API -------------------------------------------------------------

#[test]
fn sync_is_a_documented_noop_on_coherent_backends() {
    // Mem and Shm sync calls are no-ops.
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
    // Pins once, then brackets a CPU read each frame while taking `&mut` on
    // the tensor.
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

/// The DMA-heap skip policy: a skip under `HAL_TEST_REQUIRE_DMA=1` fails.
#[cfg(target_os = "linux")]
#[path = "support/dma_require.rs"]
mod dma_require;

#[cfg(target_os = "linux")]
#[test]
fn dma_pin_of_a_subview_is_offset_adjusted() {
    // A pin of a subview hands back the view's address: written through the
    // pin, read back through the parent at the same offset.
    if !dma_require::available_or_skip(
        edgefirst_tensor::is_dma_available(),
        "dma_pin_of_a_subview_is_offset_adjusted",
    ) {
        return;
    }
    // A bottom-half view at x=0 has a non-zero mmap offset.
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
    // The address stays valid after every map guard has dropped.
    if !dma_require::available_or_skip(
        edgefirst_tensor::is_dma_available(),
        "dma_pin_survives_guard_drops",
    ) {
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

    // Takes and drops a conventional map guard.
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
    if !dma_require::available_or_skip(
        edgefirst_tensor::is_dma_available(),
        "dma_sync_brackets_round_trip_in_every_direction",
    ) {
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
    // The direction passed to `sync_for_device` matches the one passed to `sync_for_cpu`.
    for access in [CpuAccess::Read, CpuAccess::Write, CpuAccess::ReadWrite] {
        t.sync_for_cpu(access).expect("start");
        t.sync_for_device(access).expect("end");
    }
}

// --- IOSurface (macOS/iOS) ------------------------------------------------

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
fn iosurface_pin_is_stable_and_sync_brackets_lock_unlock() {
    // A raw tensor: the pin takes the surface's address and the sync calls take the lock.
    let t = Tensor::<u8>::new(&[1024], Some(TensorMemory::DmaBuf), None).expect("alloc iosurface");

    let pin = t.pin_host(CpuAccess::ReadWrite).expect("pin iosurface");
    let addr = pin.as_ptr();
    assert!(!addr.is_null());

    // A conventional guard takes and releases its own lock.
    {
        let _g = t.map().expect("map");
    }
    assert_eq!(pin.as_ptr(), addr, "pin address moved when a guard dropped");

    // Write bracket.
    t.sync_for_cpu(CpuAccess::Write).expect("IOSurfaceLock");
    unsafe { std::ptr::write_bytes(pin.as_mut_ptr(), 0x5E, pin.len()) };
    t.sync_for_device(CpuAccess::Write)
        .expect("IOSurfaceUnlock");

    // Read bracket, which takes the read-only lock.
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
    // Skips where SHM allocation is unavailable.
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

    // A conventional guard comes and goes.
    {
        let _g = t.map().expect("map");
    }
    assert_eq!(pin.as_ptr(), addr, "pin address moved when a guard dropped");

    t.sync_for_cpu(CpuAccess::Write)
        .expect("shm sync is a no-op");
    unsafe { std::ptr::write_bytes(pin.as_mut_ptr(), 0x3C, pin.len()) };
    t.sync_for_device(CpuAccess::Write)
        .expect("shm sync is a no-op");

    // Reads back through a fresh guard.
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
    // An image tensor; skips when no AHardwareBuffer can be allocated.
    let expect = if edgefirst_tensor::is_ahardwarebuffer_available() {
        outcome::Expect::Succeeds
    } else {
        outcome::Expect::MayFail("the AHardwareBuffer probe failed on this device")
    };
    let Some(t) = outcome::allocated(
        Tensor::<u8>::image(
            64,
            64,
            PixelFormat::Rgba,
            Some(TensorMemory::DmaBuf),
            CpuAccess::ReadWrite,
        ),
        "AHardwareBuffer allocation",
        expect,
    ) else {
        return;
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
    // Sync reports `NotImplemented` too.
    assert!(matches!(
        t.sync_for_cpu(CpuAccess::Read),
        Err(Error::NotImplemented(_))
    ));
    assert!(matches!(
        t.sync_for_device(CpuAccess::Read),
        Err(Error::NotImplemented(_))
    ));

    // A map still works.
    let g = t.map().expect("map must work where pin cannot");
    assert!(!g.as_slice().is_empty(), "map must expose the buffer");
}

// --- re-layout while pinned -----------------------------------------------

#[test]
fn relayout_while_pinned_is_safe_and_the_pin_is_a_snapshot() {
    // A pin stays valid across re-layout and does not follow the new shape.
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

    // Shrinks the logical shape within the same allocation.
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

    // Growing beyond capacity is refused.
    assert!(t.set_logical_shape(&[64 * 64 * 4]).is_err());
}

#[cfg(target_os = "linux")]
#[test]
fn dma_relayout_while_pinned_does_not_move_the_mapping() {
    if !dma_require::available_or_skip(
        edgefirst_tensor::is_dma_available(),
        "dma_relayout_while_pinned_does_not_move_the_mapping",
    ) {
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
