//! The cross-package protocol contract: the size, alignment and field meaning
//! of `TensorDesc`.

use edgefirst_tensor::{
    format_from_code, tensor_dtype as dtype, tensor_format, tensor_kind, Colorimetry, CpuAccess,
    PixelFormat, Tensor, TensorDesc, TensorDyn, TensorMemory, ABI_VERSION,
};

/// The DMA-heap skip policy: a skip under `HAL_TEST_REQUIRE_DMA=1` fails.
#[cfg(target_os = "linux")]
#[path = "support/dma_require.rs"]
mod dma_require;

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
    assert_eq!(d.kind, tensor_kind::MEM, "Mem backend reports MEM");
    assert_eq!(d.kind, TensorMemory::Mem.code());
    assert!(d.is_host());
    assert_eq!(d.ndim, 3);
    assert_eq!(d.shape(), &[32u64, 64, 3]);
    assert_eq!(d.len(), 32 * 64 * 3);
    assert!(!d.is_empty());
}

#[test]
fn a_descriptor_with_a_zero_extent_is_empty() {
    let t = Tensor::<u8>::new(&[4, 4], Some(TensorMemory::Mem), None).expect("alloc");
    let mut d = TensorDyn::from(t).descriptor();
    d.shape[0] = 0;
    assert_eq!(d.len(), 0);
    assert!(
        d.is_empty(),
        "a zero-length dimension addresses no elements"
    );
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
    // shape [32, 64, 3] -> strides [192, 3, 1] for u8.
    assert_eq!(d.strides(), &[192i64, 3, 1]);

    // Same shape, 4-byte elements: every stride scales by sizeof(f32).
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
    // The size of `TensorDesc` is the wire format. The struct is hole-free:
    // `flags` occupies the 4 bytes between `colorimetry` and the 8-aligned
    // `capacity`.
    assert_eq!(
        std::mem::size_of::<TensorDesc>(),
        192,
        "descriptor size is ABI"
    );
    assert_eq!(std::mem::align_of::<TensorDesc>(), 8);
}

#[test]
fn unknown_major_is_detectable() {
    let t = Tensor::<u8>::new(&[4], None, None).expect("alloc");
    let mut d = TensorDyn::from(t).descriptor();
    assert_eq!(d.version, ABI_VERSION);
    d.version = 999;
    assert_ne!(d.version, ABI_VERSION, "consumers reject on this check");
}

#[test]
fn bare_descriptor_carries_no_address() {
    // An unpinned descriptor carries shape, format and kind but no address.
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
    // With a pin, the descriptor carries the buffer's address.
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

    // Reads it back through the address and length only.
    let seen = unsafe { std::slice::from_raw_parts(d.ptr.0 as *const u8, d.len() as usize) };
    assert!(
        seen.iter().all(|b| *b == 0x3C),
        "consumer sees the producer's bytes"
    );
}

#[test]
fn descriptor_reports_the_element_type() {
    // Two tensors of the same shape and different element type produce
    // different descriptors.
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
    // Every code is pinned by value.
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
    // The `TensorMemory` codes (ABI version 2).
    assert_eq!(
        [
            tensor_kind::MEM,
            tensor_kind::SHM,
            tensor_kind::DMABUF,
            tensor_kind::IOSURFACE,
            tensor_kind::PBO,
            tensor_kind::CUDA,
            tensor_kind::D3D11_TEXTURE,
        ],
        [0, 1, 2, 3, 4, 5, 6]
    );
    assert_eq!(ABI_VERSION, 2);
}

#[test]
fn only_mem_and_shm_kinds_are_host() {
    let t = Tensor::<u8>::new(&[4], Some(TensorMemory::Mem), None).expect("alloc");
    let mut d = TensorDyn::from(t).descriptor();
    for &m in TensorMemory::all() {
        d.kind = m.code();
        assert_eq!(
            d.is_host(),
            matches!(m, TensorMemory::Mem | TensorMemory::Shm),
            "{m:?}"
        );
    }
}

#[test]
fn strides_follow_row_stride_not_shape() {
    // A padded row pitch: `strides[0]` comes from `row_stride`, not from width * channels.
    let mut t = Tensor::<u8>::image(
        100,
        4,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let natural = 100 * 3; // width * channels * sizeof(u8)
    let padded = 320; // > natural
    t.set_row_stride(padded).expect("set_row_stride");

    // Precondition: the padded stride differs from the natural one.
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
    // Host (Mem) tensors have no native handle.
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
    // On macOS/iOS, `TensorMemory::DmaBuf` is IOSurface-backed and `handle`
    // carries the surface id.
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
    // On Linux, `TensorMemory::DmaBuf` is a dma-buf and `handle` carries the fd.
    if !dma_require::available_or_skip(
        edgefirst_tensor::is_dma_available(),
        "descriptor_carries_dmabuf_fd_as_handle",
    ) {
        return;
    }
    let t = Tensor::<u8>::image(
        16,
        16,
        PixelFormat::Grey,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .expect("a usable DMA heap allocates");
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
    // `PlanarRgb.allocation_shape(640, 480)` is `[3, 480, 640]`. A padded row
    // pitch is `strides[1]`, and the plane stride `strides[0]` is `H * pitch`.
    let mut t = Tensor::<u8>::image(
        640,
        480,
        PixelFormat::PlanarRgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let natural_row = 640; // width * sizeof(u8), Planar's minimum row stride
    let padded_row = 704; // > natural_row
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
        480 * padded_row as i64,
        "the plane stride is H rows of the padded pitch, not the row pitch itself \
         and not the tight H * W"
    );
    assert_ne!(natural_row, padded_row, "precondition: pitch actually pads");
}

#[test]
fn planar_format_round_trips_through_the_descriptor() {
    // `PlanarRgb.to_fourcc()` is 0, the value a non-image tensor carries;
    // `format` identifies it.
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
    // f32 (esz=4) with a recorded pitch that is not a whole number of elements.
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
    // The innermost stride is one f32, in bytes.
    assert_eq!(
        d.strides()[2],
        4,
        "innermost stride is one element expressed in bytes"
    );
}

#[test]
fn flags_and_sync_are_reserved_and_clear() {
    // A host producer sets only the CPU access report in `flags`, and never `sync`.
    let t = Tensor::<u8>::image(
        32,
        16,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("alloc");
    let d = TensorDyn::from(t).descriptor();
    assert_eq!(
        d.flags,
        edgefirst_tensor::protocol::flags::from_cpu_access(CpuAccess::ReadWrite),
        "a host producer reports only its CPU access in flags"
    );
    assert_eq!(d.sync, 0, "sync is reserved and must stay clear");
    assert_eq!(
        d.flags & edgefirst_tensor::protocol::flags::SYNC_PRESENT,
        0,
        "SYNC_PRESENT must not be advertised while sync is unset"
    );
}
