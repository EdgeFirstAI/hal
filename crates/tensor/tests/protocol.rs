//! The cross-package protocol contract.
//!
//! Layout here is not an implementation detail: independently-built packages
//! read this struct, so a silent change breaks consumers compiled against an
//! older version. Size, alignment and field meaning are pinned.

use edgefirst_tensor::{
    format_from_code, tensor_dtype as dtype, tensor_format, tensor_kind, Colorimetry, CpuAccess,
    PixelFormat, Tensor, TensorDesc, TensorDyn, TensorMemory, ABI_VERSION,
};
// Only used by the Linux-only `descriptor_carries_dmabuf_fd_as_handle` below
// -- importing it unconditionally warns as unused on every other platform.
#[cfg(target_os = "linux")]
use edgefirst_tensor::Error;

#[test]
fn descriptor_describes_the_tensor() {
    let t = Tensor::<u8>::image(
        64,
        32,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let d = TensorDyn::from(t).descriptor();
    assert_eq!(d.version, ABI_VERSION);
    assert_eq!(d.kind, tensor_kind::HOST, "Mem backend reports HOST");
    assert_eq!(d.ndim, 3);
    assert_eq!(d.shape(), &[32u64, 64, 3]);
    assert_eq!(d.len(), 32 * 64 * 3);
    assert!(!d.is_empty());
}

#[test]
fn contiguous_strides_are_row_major_in_bytes() {
    let t = Tensor::<u8>::image(
        64,
        32,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let d = TensorDyn::from(t).descriptor();
    // shape [32, 64, 3] -> strides [192, 3, 1]. u8, so bytes and elements
    // coincide -- which is exactly why this case alone cannot tell the two
    // representations apart. The f32 case below is the discriminating one.
    assert_eq!(d.strides(), &[192i64, 3, 1]);

    // Same shape, 4-byte elements: every stride scales by sizeof(f32).
    // Under the old ELEMENT strides this was [192, 3, 1] again.
    let t = Tensor::<f32>::new(&[32, 64, 3], Some(TensorMemory::Mem), None).expect("alloc f32");
    let d = TensorDyn::from(t).descriptor();
    assert_eq!(
        d.strides(),
        &[768i64, 12, 4],
        "strides are bytes: element strides would be [192, 3, 1] regardless of dtype"
    );
}

#[test]
fn descriptor_layout_is_pinned() {
    // The cross-package contract, and the reason this number is asserted
    // rather than merely observed: two independently-compiled `.so` files
    // read each other's payloads at their own `TensorDesc`, so this size
    // IS the wire format.
    //
    // If it changes, the capsule name must move with it, in the same
    // commit -- `edgefirst_tensor_v1` -> `_v2`, see INTEROP.md's Versioning
    // section. NOT `ABI_VERSION`: that field is checked by
    // `TensorDyn::import_descriptor`, but only after the payload has been
    // copied out at this build's size, so it cannot gate a mismatch it is
    // itself read through. The capsule name is checked by
    // `PyCapsule::pointer_checked` before any byte is touched, which is why
    // it is the gate. (DLPack landed on the same split: a version field
    // inside `DLManagedTensorVersioned` *and* a distinct
    // `dltensor_versioned` capsule name, because the field alone cannot be
    // trusted until the layout is already known.)
    //
    // Two traps worth stating explicitly, both of which this assertion
    // exists to catch:
    //
    // 1. Size staying the same is NOT proof a change was ABI-safe. A field
    //    added into existing padding -- as `dtype` was, into the 4-byte
    //    hole after `ndim` -- leaves this number untouched while changing
    //    what every byte means. Once a layout has shipped, its padding is
    //    not spare room: an older producer leaves those bytes zero and a
    //    newer consumer decodes a plausible-looking wrong answer (there,
    //    every tensor as `U8`). Claiming padding needs a name bump exactly
    //    like appending does.
    //
    // 2. The payload's *representation* counts, not just this struct's
    //    fields. `TensorCapsulePayload` in `python-common` is `#[repr(C)]`
    //    for that reason; it was briefly a bare Rust tuple, whose field
    //    order and padding the language does not specify -- fine within one
    //    compilation, not a guarantee across two `.so` files.
    //
    // The struct is now hole-free: `flags` claimed the 4 bytes that used to
    // sit between `colorimetry` and the 8-aligned `capacity`, which is why
    // adding it cost nothing while `sync` cost 8. That was deliberate --
    // both were added before the first release, when growth is free. After
    // release it is not: see the name-bump rule above.
    assert_eq!(
        std::mem::size_of::<TensorDesc>(),
        192,
        "descriptor size is ABI"
    );
    assert_eq!(std::mem::align_of::<TensorDesc>(), 8);
}

#[test]
fn unknown_major_is_detectable() {
    // A consumer's rejection path: it must be able to tell a version it does
    // not know from one it does, before reading any other field.
    let t = Tensor::<u8>::new(&[4], None, None).expect("alloc");
    let mut d = TensorDyn::from(t).descriptor();
    assert_eq!(d.version, ABI_VERSION);
    d.version = 999;
    assert_ne!(d.version, ABI_VERSION, "consumers reject on this check");
}

#[test]
fn bare_descriptor_carries_no_address() {
    // Descriptive only: a consumer learns shape/format/kind but cannot reach
    // the bytes. This is deliberate -- an unpinned host address is not durable.
    let t = Tensor::<u8>::image(
        16,
        16,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let d = TensorDyn::from(t).descriptor();
    assert!(
        d.ptr.is_null(),
        "unpinned descriptor must not hand out an address"
    );
}

#[test]
fn pinned_descriptor_reaches_the_bytes() {
    // The Plan 1 unblock: with a pin, a consumer in another package can read
    // the producer's buffer through the descriptor alone.
    let t = Tensor::<u8>::image(
        16,
        16,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let pin = t.pin_host(CpuAccess::ReadWrite).expect("pin");
    unsafe { std::ptr::write_bytes(pin.as_mut_ptr(), 0x3C, pin.len()) };

    let dyn_t = TensorDyn::from(t);
    let d = dyn_t.descriptor_pinned(Some(&pin));
    assert!(!d.ptr.is_null());
    assert_eq!(d.len(), 16 * 16);

    // Read it back the way a foreign consumer would: address + length only.
    let seen = unsafe { std::slice::from_raw_parts(d.ptr.0 as *const u8, d.len() as usize) };
    assert!(
        seen.iter().all(|b| *b == 0x3C),
        "consumer sees the producer's bytes"
    );
}

#[test]
fn descriptor_reports_the_element_type() {
    // `strides` is in BYTES, so a consumer can address the buffer without
    // knowing the element size -- but it still needs `dtype` to interpret
    // what it finds there. Two tensors of the same shape and different
    // element type must not produce identical descriptors.
    let f = Tensor::<f32>::new(&[4], Some(TensorMemory::Mem), None).expect("alloc f32");
    let u = Tensor::<u8>::new(&[4], Some(TensorMemory::Mem), None).expect("alloc u8");
    let df = TensorDyn::from(f).descriptor();
    let du = TensorDyn::from(u).descriptor();

    assert_eq!(df.dtype, dtype::F32);
    assert_eq!(du.dtype, dtype::U8);
    assert_ne!(
        df.dtype, du.dtype,
        "f32 and u8 tensors must not produce identical descriptors"
    );
}

#[test]
fn dtype_codes_are_abi() {
    // These numbers cross a package boundary; a consumer compiled against an
    // older version reads them by value. Reordering is a breaking change, so
    // pin every one rather than trusting declaration order to stay put.
    assert_eq!(
        [
            dtype::U8,
            dtype::I8,
            dtype::U16,
            dtype::I16,
            dtype::U32,
            dtype::I32,
            dtype::U64,
            dtype::I64,
            dtype::F16,
            dtype::F32,
            dtype::F64,
        ],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    );
}

#[test]
fn kind_codes_are_abi() {
    // Same ABI concern as dtype_codes_are_abi, for the backing-store kind.
    assert_eq!(
        [
            tensor_kind::HOST,
            tensor_kind::DMABUF,
            tensor_kind::IOSURFACE,
            tensor_kind::PBO,
        ],
        [0, 1, 2, 3]
    );
}

#[test]
fn strides_follow_row_stride_not_shape() {
    // A pitch-aligned image carries padding between rows that the shape
    // alone cannot express: strides[0] must come from row_stride, not from
    // width * channels. `set_row_stride` performs no buffer-size validation
    // (see its doc on `Tensor::set_row_stride`) -- it is pure layout
    // metadata -- so a padded pitch is reproducible on every platform
    // without needing a backend (IOSurface, DMA-BUF) that actually pads the
    // allocation.
    let mut t = Tensor::<u8>::image(
        100,
        4,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let natural = 100 * 3; // width * channels * sizeof(u8), what the bug computed
    let padded = 320; // > natural; u8, so this byte pitch is also an element count
    t.set_row_stride(padded).expect("set_row_stride");

    // Precondition: without this, the buggy and fixed computations produce
    // the same number and the test below cannot distinguish them.
    assert_ne!(
        t.row_stride(),
        Some(natural),
        "row_stride must differ from width*channels for this test to be discriminating"
    );

    let d = TensorDyn::from(t).descriptor();
    assert_eq!(
        d.strides()[0],
        padded as i64,
        "row stride must be reported in bytes, from row_stride not shape"
    );
}

#[test]
fn descriptor_handle_defaults_to_unused() {
    // HOST (Mem) tensors have no native handle to hand a consumer.
    let t = Tensor::<u8>::image(
        16,
        16,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let d = TensorDyn::from(t).descriptor();
    assert_eq!(d.handle, -1, "HOST backend carries no native handle");
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[test]
fn descriptor_carries_iosurface_id_as_handle() {
    // On macOS/iOS, TensorMemory::DmaBuf is IOSurface-backed; the consumer
    // re-imports zero-copy from the surface id carried in `handle`.
    let t = Tensor::<u8>::image(
        16,
        16,
        PixelFormat::Grey,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .expect("alloc dma");
    let dyn_t = TensorDyn::from(t);
    let expected = dyn_t.iosurface_id().expect("iosurface id");
    let d = dyn_t.descriptor();
    assert_eq!(d.handle, expected as i64);
}

#[cfg(target_os = "linux")]
#[test]
fn descriptor_carries_dmabuf_fd_as_handle() {
    use std::os::fd::AsRawFd;
    // On Linux, TensorMemory::DmaBuf is a dma-buf fd; the consumer re-imports
    // zero-copy from the fd carried in `handle`.
    let t = match Tensor::<u8>::image(
        16,
        16,
        PixelFormat::Grey,
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
        Err(Error::IoError(e))
            if matches!(
                e.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied
            ) =>
        {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "SKIP: descriptor_carries_dmabuf_fd_as_handle -- this platform has no \
                 DMA-BUF heap (Tensor::image(.., TensorMemory::DmaBuf, ..) returned \
                 NotFound); dma-buf allocation is unavailable here, not broken."
            );
            return;
        }
        Err(e) => panic!("alloc dma: {e:?}"),
    };
    let dyn_t = TensorDyn::from(t);
    let expected_fd = dyn_t.dmabuf().expect("dmabuf fd").as_raw_fd();
    let d = dyn_t.descriptor();
    assert_eq!(d.handle, expected_fd as i64);
}

#[test]
fn descriptor_colorimetry_defaults_to_zero() {
    let t = Tensor::<u8>::image(
        16,
        16,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let d = TensorDyn::from(t).descriptor();
    assert_eq!(d.colorimetry, 0, "undefined colorimetry packs to 0");
}

#[test]
fn planar_row_stride_does_not_clobber_the_plane_stride() {
    // PlanarRgb.allocation_shape(640, 480) == [3, 480, 640]: dimension 0 is the
    // channel/plane count, dimension 1 is the true row dimension. A padded
    // per-row pitch must land on strides[1], not strides[0] -- overwriting
    // strides[0] corrupts the plane stride while leaving the real row
    // stride at strides[1] unfixed, and PlanarRgb@Dma is a live path.
    let mut t = Tensor::<u8>::image(
        640,
        480,
        PixelFormat::PlanarRgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let natural_row = 640; // width * sizeof(u8), Planar's minimum row stride
    let natural_plane = 480 * 640; // the packed plane stride, in bytes (u8)
    let padded_row = 704; // > natural_row; u8, so bytes and elements coincide
    t.set_row_stride(padded_row).expect("set_row_stride");

    let d = TensorDyn::from(t).descriptor();
    assert_eq!(d.shape(), &[3u64, 480, 640]);
    assert_eq!(
        d.strides()[1],
        padded_row as i64,
        "the true row dimension must carry the padded pitch"
    );
    assert_eq!(
        d.strides()[0],
        natural_plane as i64,
        "the plane-count dimension must not be clobbered by the row pitch"
    );
    assert_ne!(natural_row, padded_row, "precondition: pitch actually pads");
}

#[test]
fn planar_format_round_trips_through_the_descriptor() {
    // fourcc alone cannot represent PlanarRgb: PlanarRgb.to_fourcc() == 0,
    // the same sentinel a non-image tensor uses, so a consumer reading only
    // `fourcc` would import this as a formatless rank-3 tensor and misread
    // the layout. `format` exists precisely to make this identifiable.
    let t = Tensor::<u8>::image(
        640,
        480,
        PixelFormat::PlanarRgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let d = TensorDyn::from(t).descriptor();

    assert_eq!(d.fourcc, 0, "precondition: fourcc cannot express PlanarRgb");
    assert_eq!(d.format, tensor_format::PLANAR_RGB);
    assert_ne!(
        d.format, 0,
        "PlanarRgb must not be indistinguishable from \"no format\""
    );
    assert_eq!(
        format_from_code(d.format),
        Some(PixelFormat::PlanarRgb),
        "must round-trip back to the original PixelFormat"
    );
}

#[test]
fn descriptor_carries_colorimetry() {
    let t = Tensor::<u8>::image(
        16,
        16,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let dyn_t = TensorDyn::from(t).with_colorimetry(Colorimetry::jfif());
    let d = dyn_t.descriptor();
    assert_eq!(d.colorimetry, Colorimetry::jfif().pack());
    assert_ne!(d.colorimetry, 0);
}

#[test]
fn a_pitch_that_is_not_a_whole_number_of_elements_is_exact() {
    // f32 (esz=4): `set_row_stride` validates only stride >= the packed
    // minimum, not alignment to sizeof(T), so a caller can legally record a
    // pitch that is not a whole number of elements.
    //
    // Under the old ELEMENT strides this value was unrepresentable: `rs / esz`
    // would truncate 54 -> 13 in a release build, so the code fell back to the
    // packed stride and reported a pitch the buffer does not have. Byte strides
    // delete the failure mode -- 54 is simply 54.
    let mut t = Tensor::<f32>::image(
        4,
        4,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let misaligned_bytes = 54; // >= min_stride (48), and 54 % 4 != 0
    t.set_row_stride(misaligned_bytes).expect("set_row_stride");

    let d = TensorDyn::from(t).descriptor();
    assert_eq!(
        d.strides()[0],
        misaligned_bytes as i64,
        "a byte pitch is carried exactly, whatever its remainder mod sizeof(T)"
    );
    // The innermost stride is one f32: proof the whole array is bytes, not
    // elements. Under element strides this was 1.
    assert_eq!(
        d.strides()[2],
        4,
        "innermost stride is one element expressed in bytes"
    );
}

#[test]
fn flags_and_sync_are_reserved_and_clear() {
    // Both fields exist so the layout is final before first release, but no
    // producer in this build sets either. A consumer that waited on `sync`
    // without first checking `SYNC_PRESENT` would be waiting on garbage, so
    // the invariant asserted here is "clear unless explicitly set", not
    // merely "the fields exist".
    let t = Tensor::<u8>::image(
        32,
        16,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let d = TensorDyn::from(t).descriptor();
    assert_eq!(d.flags, 0, "no producer in this build sets flags");
    assert_eq!(d.sync, 0, "sync is reserved and must stay clear");
    assert_eq!(
        d.flags & edgefirst_tensor::protocol::flags::SYNC_PRESENT,
        0,
        "SYNC_PRESENT must not be advertised while sync is unset"
    );
}
