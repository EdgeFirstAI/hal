//! Consumer side of the cross-package tensor protocol:
//! [`TensorDyn::import_descriptor`] rebuilding a usable, aliasing tensor from
//! an [`TensorDesc`](edgefirst_tensor::TensorDesc).
//!
//! Every roundtrip test writes through the producer handle and reads back
//! through the imported one.

use edgefirst_tensor::{
    Colorimetry, CpuAccess, Error, PixelFormat, Tensor, TensorDyn, TensorMapTrait, TensorMemory,
    TensorTrait,
};

/// The DMA-heap skip policy: a skip under `HAL_TEST_REQUIRE_DMA=1` fails.
#[cfg(target_os = "linux")]
#[path = "support/dma_require.rs"]
mod dma_require;

/// `true` when `what` should run: a usable DMA heap, or a skip reported
/// (and refused under `HAL_TEST_REQUIRE_DMA=1`).
#[cfg(target_os = "linux")]
fn dma_or_skip(what: &str) -> bool {
    dma_require::available_or_skip(edgefirst_tensor::is_dma_available(), what)
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

    // The pin is taken before `t` moves into the `TensorDyn` wrapper.
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
    // `PlanarRgb.to_fourcc()` is 0, the value a non-image tensor carries.
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

/// A version-1 descriptor is refused.
#[test]
fn rejects_a_version_1_descriptor() {
    let t = Tensor::<u8>::new(&[4], Some(TensorMemory::Mem), None).unwrap();
    let mut desc = TensorDyn::from(t).descriptor();
    desc.version = 1;
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
    // `descriptor_pinned` packs colorimetry and `import_descriptor` unpacks it.
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
    // NV12 at an odd width pads the row stride beyond `shape.product()`.
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

    // Reconfigures within that headroom.
    let mut imported = imported;
    imported
        .configure_image(295, 175, PixelFormat::Nv12)
        .expect("configure_image must fit within the producer's real capacity");
}

#[test]
fn host_import_preserves_the_producers_row_stride_for_pool_reuse() {
    // The import restores the producer's stride, and a reconfigure to a
    // smaller image keeps it.
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
    // `import_descriptor` checks `stride_bytes * rows <= capacity`.
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
    // A row pitch that, times the row count, exceeds the 16x16x3-byte capacity.
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
    // An unpinned descriptor has a null `ptr`.
    let t = Tensor::<u8>::new(&[4], Some(TensorMemory::Mem), None).unwrap();
    let desc = TensorDyn::from(t).descriptor();
    let err = TensorDyn::import_descriptor(&desc).unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("descriptive-only capsule")),
        "expected the no-address refusal naming the producer-side fix, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// dma-buf (Linux only): not testable on this platform.
// ---------------------------------------------------------------------------
#[cfg(not(target_os = "linux"))]
#[test]
fn dmabuf_import_has_no_coverage_off_linux() {
    use std::io::Write;
    let _ = writeln!(
        std::io::stderr(),
        "WARNING: SKIPPED: TensorDyn::import_descriptor's kind::DMABUF arm is untested on \
         this platform (dma-buf is Linux-only). No aliasing coverage for the \
         dma-buf import path outside a Linux run."
    );
}

/// A `Shm` producer reports `kind::SHM`; the import aliases the pinned address as host memory.
#[cfg(unix)]
#[test]
fn shm_descriptor_reports_shm_and_imports_as_host_memory() {
    let t = Tensor::<u8>::new(&[16, 16], Some(TensorMemory::Shm), None).expect("shm allocation");
    t.map_write().unwrap().as_mut_slice()[0] = 0x3C;
    let pin = t.pin_host(CpuAccess::ReadWrite).unwrap();
    let dyn_t = TensorDyn::from(t);
    let desc = dyn_t.descriptor_pinned(Some(&pin));
    assert_eq!(desc.kind, edgefirst_tensor::tensor_kind::SHM);
    assert_eq!(desc.kind, TensorMemory::Shm.code());
    assert_ne!(desc.kind, edgefirst_tensor::tensor_kind::DMABUF);
    assert!(desc.is_host());

    let imported = TensorDyn::import_descriptor(&desc).unwrap();
    let m = imported.as_u8().unwrap().map_read().unwrap();
    assert_eq!(m.as_slice()[0], 0x3C, "import must alias, not copy");
}

#[cfg(target_os = "linux")]
#[test]
fn dmabuf_roundtrip_sees_the_same_bytes() {
    if !dma_or_skip("dmabuf_roundtrip_sees_the_same_bytes") {
        return;
    }
    let t = Tensor::<u8>::image(
        64,
        32,
        PixelFormat::Rgb,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .expect("a usable DMA heap allocates");
    {
        let mut m = t.map_write().unwrap();
        m.as_mut_slice()[0] = 0xEF;
    }
    let dyn_t = TensorDyn::from(t);
    let desc = dyn_t.descriptor(); // handle carries the dma-buf fd
    assert_eq!(desc.kind, edgefirst_tensor::tensor_kind::DMABUF);
    assert_eq!(desc.kind, TensorMemory::DmaBuf.code());
    assert_eq!(desc.kind, 2, "dma-buf is code 2 on every surface");

    let imported = TensorDyn::import_descriptor(&desc).unwrap();
    assert_eq!(imported.shape(), dyn_t.shape());
    let m = imported.as_u8().unwrap().map_read().unwrap();
    assert_eq!(m.as_slice()[0], 0xEF, "dma-buf import must alias, not copy");
}

#[cfg(target_os = "linux")]
#[test]
fn imported_dmabuf_with_a_recorded_stride_is_still_cpu_mappable() {
    // Imports an aliased copy of a self-allocated Dma tensor's fd, records a
    // stride on the import, and writes through it.
    if !dma_or_skip("imported_dmabuf_with_a_recorded_stride_is_still_cpu_mappable") {
        return;
    }
    let owner = Tensor::<u8>::image(
        64,
        32,
        PixelFormat::Rgb,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .expect("a usable DMA heap allocates");

    let fd = TensorTrait::clone_fd(&owner).expect("clone dma-buf fd");
    let mut imported = Tensor::<u8>::from_fd(fd, &[32, 64, 3], None).expect("import via from_fd");
    imported.set_format(PixelFormat::Rgb).expect("set_format");
    // width * channels * elem.
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
    if !dma_or_skip("dmabuf_import_preserves_the_producers_row_stride_for_pool_reuse") {
        return;
    }
    let t = Tensor::<u8>::image(
        1920,
        1080,
        PixelFormat::Nv12,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .expect("a usable DMA heap allocates");
    assert_eq!(
        t.row_stride(),
        Some(1920),
        "precondition: semi-planar always records it"
    );

    let dyn_t = TensorDyn::from(t);
    let desc = dyn_t.descriptor(); // handle carries the dma-buf fd

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
// IOSurface (macOS/iOS).
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
    let desc = dyn_t.descriptor(); // handle carries the IOSurfaceID
    assert_eq!(desc.kind, edgefirst_tensor::tensor_kind::IOSURFACE);

    let imported = TensorDyn::import_descriptor(&desc).unwrap();
    assert_eq!(imported.shape(), dyn_t.shape());
    // The stride survives the round trip.
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
    desc.handle = 0x7FFF_FFFF; // an id no live surface holds
    let err = TensorDyn::import_descriptor(&desc).unwrap_err();
    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "expected InvalidArgument, got {err:?}"
    );
}

#[test]
fn import_refuses_a_descriptor_advertising_an_unwaitable_fence() {
    // `sync` is reserved in this build: a descriptor with `SYNC_PRESENT` is refused.
    let t = Tensor::<u8>::image(
        16,
        8,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    // The pin is taken before the move: the descriptor carries a real address.
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

/// A `D3D11_TEXTURE` descriptor with a fence value and a null `ptr` is refused on every platform.
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

/// Mock [`edgefirst_tensor::PboOps`] over a `Vec<u8>`.
struct MockPboOps {
    storage: std::sync::Mutex<Vec<u8>>,
}

// SAFETY: the returned pointer addresses a `Vec<u8>` allocated once in `new`
// and never resized, and the tensor holds an `Arc` to the `MockPboOps`.
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

/// A PBO-backed tensor's descriptor carries the PBO vtable pointer in `ptr`, not a host address.
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

    // The same descriptor through `descriptor_pinned(None)`.
    let d = t.descriptor_pinned(None);
    assert_eq!(
        d.ptr.0 as *const std::ffi::c_void, vtable,
        "the pinned-descriptor entry point agrees with the plain one when \
         there is no pin to prefer"
    );

    // A PBO has no pin to take.
    assert!(
        matches!(
            t.pin_host(CpuAccess::Read),
            Err(edgefirst_tensor::Error::NotImplemented(_))
        ),
        "a PBO must refuse pin_host, or `ptr` could carry a host address"
    );
}

/// A PBO `view()` keeps its parent's pitch across the descriptor.
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
        imported.pbo_id(),
        Some(88),
        "the import names the producer's GL buffer"
    );
    assert_eq!(
        imported.effective_row_stride(),
        Some(parent_pitch),
        "the rebuilt view must step rows by the parent pitch ({parent_pitch}); \
         {} is the window's own tight row, which shears every row after the \
         first",
        SIDE * 3
    );
}

/// `pin_host` works on host memory and is refused on a PBO, on both backends.
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

/// The descriptor's buffer id is carried verbatim, and only a negative one means no buffer.
#[test]
fn a_pbo_descriptor_carries_any_buffer_id_and_refuses_a_missing_one() {
    let t = pbo_backed(0, 16, &[4, 4, 1]);
    let imported = TensorDyn::import_descriptor(&t.descriptor())
        .expect("buffer id 0 is a descriptor value like any other");
    assert_eq!(imported.pbo_id(), Some(0));

    let mut desc = t.descriptor();
    desc.handle = -1;
    let err = TensorDyn::import_descriptor(&desc).unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m == "PBO descriptor carries no buffer id"),
        "got {err:?}"
    );
}

/// A `Shm` tensor's descriptor relabelled `kind::DMABUF` around a dup of its fd.
#[cfg(target_os = "linux")]
fn shm_relabelled_as_dmabuf(
    first: u8,
) -> (
    TensorDyn,
    std::os::fd::OwnedFd,
    edgefirst_tensor::TensorDesc,
) {
    use std::os::fd::AsRawFd;
    let t = Tensor::<u8>::new(&[16, 16], Some(TensorMemory::Shm), None).expect("shm allocation");
    t.map_write().expect("map shm").as_mut_slice()[0] = first;
    let t = TensorDyn::from(t);
    let fd = t.clone_fd().expect("shm fd");
    let mut desc = t.descriptor();
    desc.kind = edgefirst_tensor::tensor_kind::DMABUF;
    desc.handle = fd.as_raw_fd() as i64;
    (t, fd, desc)
}

#[cfg(target_os = "linux")]
#[test]
fn a_dmabuf_descriptor_imports_a_dup_of_its_fd() {
    let (_src, fd, desc) = shm_relabelled_as_dmabuf(0x5A);
    let imported = TensorDyn::import_descriptor(&desc).expect("the dma-buf arm imports the fd");
    // The import holds its own dup of the fd.
    drop(fd);
    assert_eq!(
        imported.memory(),
        TensorMemory::Shm,
        "from_fd classifies the tmpfs fd it was handed"
    );
    let m = imported.as_u8().unwrap().map_read().unwrap();
    assert_eq!(m.as_slice()[0], 0x5A, "the import aliases the producer");
}

#[cfg(target_os = "linux")]
#[test]
fn a_dmabuf_descriptor_with_a_negative_handle_carries_no_fd() {
    let (_src, _fd, mut desc) = shm_relabelled_as_dmabuf(0);
    desc.handle = -1;
    let err = TensorDyn::import_descriptor(&desc).unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m == "dma-buf descriptor carries no fd"),
        "got {err:?}"
    );
}

/// Handle `0` is an fd number like any other; only a negative handle is absent.
#[cfg(target_os = "linux")]
#[test]
fn a_dmabuf_descriptor_naming_fd_zero_is_not_refused_as_absent() {
    let (_src, _fd, mut desc) = shm_relabelled_as_dmabuf(0);
    desc.handle = 0;
    if let Err(err) = TensorDyn::import_descriptor(&desc) {
        assert!(
            !err.to_string().contains("carries no fd"),
            "fd 0 is a descriptor, got {err:?}"
        );
    }
}
