// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0
//! `Tensor::image(.., TensorMemory::DmaBuf, ..)` on Windows: the public
//! surface wired onto the D3D11 texture storage.
#![cfg(all(target_os = "windows", feature = "static"))]

use edgefirst_tensor::blob::TransportMode;
use edgefirst_tensor::{
    CpuAccess, DType, Error, PixelFormat, Tensor, TensorDyn, TensorMapTrait, TensorMemory,
    TensorTrait,
};
use std::os::windows::io::AsRawHandle;

#[test]
fn image_with_dmabuf_on_windows_is_a_texture_tensor() {
    if !edgefirst_tensor::is_gpu_buffer_available() {
        eprintln!("no D3D11 device on this box -- skipping");
        return;
    }
    let t = Tensor::<u8>::image(
        640,
        480,
        PixelFormat::Rgba,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    assert_eq!(t.memory(), TensorMemory::DmaBuf);
    assert!(t.d3d11_texture().is_some());
    assert_eq!(t.d3d11_layout().unwrap().texture_width, 640);
    assert_eq!(t.d3d11_layout().unwrap().texture_height, 480);
    let stride = t.row_stride().unwrap_or(640 * 4);
    assert!(stride >= 640 * 4);
    {
        let mut m = t.map().unwrap();
        m.as_mut_slice()[..4].copy_from_slice(&[1, 2, 3, 4]);
    }
    assert_eq!(&t.map().unwrap().as_slice()[..4], &[1, 2, 3, 4]);
}

#[test]
fn nv12_odd_height_allocates_the_combined_plane() {
    let t = Tensor::<u8>::image(
        640,
        481,
        PixelFormat::Nv12,
        Some(TensorMemory::DmaBuf),
        CpuAccess::Read,
    )
    .unwrap();
    assert_eq!(
        t.d3d11_layout().unwrap().texture_height,
        PixelFormat::Nv12.allocation_shape(640, 481).unwrap()[0]
    );
}

#[test]
fn unsupported_format_is_a_loud_error_not_a_silent_downgrade() {
    let err = Tensor::<u8>::image(
        64,
        64,
        PixelFormat::Vyuy,
        Some(TensorMemory::DmaBuf),
        CpuAccess::None,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("no zero-copy D3D11 texture layout"),
        "unexpected error: {err}"
    );
}

#[test]
fn try_map_returns_would_block_or_succeeds_and_other_backings_alias_map() {
    let t = Tensor::<u8>::image(
        1920,
        1080,
        PixelFormat::Rgba,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    match t.try_map() {
        Ok(_) => {}
        Err(Error::IoError(e)) => assert_eq!(e.kind(), std::io::ErrorKind::WouldBlock),
        Err(e) => panic!("unexpected {e}"),
    }
    match t.try_map_with(CpuAccess::Read) {
        Ok(_) => {}
        Err(Error::IoError(e)) => assert_eq!(e.kind(), std::io::ErrorKind::WouldBlock),
        Err(e) => panic!("unexpected {e}"),
    }
    let m = Tensor::<u8>::new(&[16], None, None).unwrap();
    assert!(m.try_map().is_ok());
}

/// A texture tensor's non-blocking map makes progress: retried, it hands back
/// the bytes a preceding CPU write put there.
#[test]
fn try_map_makes_progress_and_sees_the_written_bytes() {
    let t = Tensor::<u8>::image(
        64,
        64,
        PixelFormat::Rgba,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    {
        let mut m = t.map_write().unwrap();
        m.as_mut_slice()[..4].copy_from_slice(&[9, 8, 7, 6]);
    }
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
    let mapped = loop {
        match t.try_map_with(CpuAccess::Read) {
            Ok(v) => break v,
            Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Err(e) => panic!("unexpected {e}"),
        }
        assert!(
            std::time::Instant::now() < deadline,
            "try_map_with never succeeded"
        );
        // On WARP the threads that finish the staging copy are CPU threads
        // competing with this one for the same cores, so a tight retry starves
        // the very copy it waits for.
        std::thread::yield_now();
    };
    assert_eq!(&mapped.as_slice()[..4], &[9, 8, 7, 6]);
}

#[test]
fn tensordyn_exposes_the_same_surface() {
    let t = TensorDyn::image(
        64,
        32,
        PixelFormat::Bgra,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::None,
    )
    .unwrap();
    assert!(t.d3d11_texture().is_some());
    assert!(t.gpu_completion().unwrap().is_none());
    let h = t.d3d11_shared_handle().unwrap();
    // SAFETY: `h` is a shared NT handle this process owns and keeps alive for
    // the rest of the test; the constructor duplicates what it keeps.
    let again = unsafe {
        TensorDyn::from_d3d11_shared_handle(
            h.as_raw_handle() as _,
            64,
            32,
            PixelFormat::Bgra,
            DType::U8,
            CpuAccess::None,
            None,
            None,
        )
    }
    .unwrap();
    assert_eq!(again.memory(), TensorMemory::DmaBuf);
    assert!(again.d3d11_texture().is_some());
}

/// The pointer constructor adopts a texture the HAL allocated, and a recorded
/// fence value surfaces through `gpu_completion`.
#[test]
fn from_d3d11_texture_wraps_and_gpu_write_is_recorded() {
    let t = Tensor::<u8>::image(
        32,
        16,
        PixelFormat::Rgba,
        Some(TensorMemory::DmaBuf),
        CpuAccess::Read,
    )
    .unwrap();
    let ptr = t.d3d11_texture().unwrap();
    // SAFETY: `ptr` borrows `t`'s live texture, which the HAL device created
    // and `t` keeps alive across this call.
    let wrapped = unsafe {
        Tensor::<u8>::from_d3d11_texture(ptr, 32, 16, PixelFormat::Rgba, CpuAccess::Read, None)
    }
    .unwrap();
    assert_eq!(wrapped.memory(), TensorMemory::DmaBuf);
    assert_eq!(wrapped.d3d11_texture(), Some(ptr));
    // A wrapped texture keys on the texture pointer, so it aliases the
    // original; the same texture reopened from its NT handle would not.
    assert_eq!(
        wrapped.buffer_identity().id(),
        t.buffer_identity().id(),
        "a pointer-wrapped texture shares the original's identity"
    );
    // The exact key, not just "the two agree": this is the formula
    // `dynamic_primitives.rs`'s
    // `a_texture_handles_identity_names_its_texture_not_its_recyclable_handle_address`
    // pins for the dynamic backend, and `edgefirst-image`'s `texture_of`
    // recomputes before it will import. Asserting it on both sides is what
    // stops the backends drifting apart -- when they did, the image crate's
    // EGLImage cache served dropped textures.
    assert_eq!(
        t.buffer_identity().kind(),
        edgefirst_tensor::IdentityKind::D3d11Texture,
        "a texture tensor is identified by its texture"
    );
    assert_eq!(
        t.buffer_identity().id(),
        ((edgefirst_tensor::IdentityKind::D3d11Texture as u64) << 56) ^ (ptr as usize as u64),
        "the static backend's key is the ID3D11Texture2D pointer, the same value \
         the dynamic backend reads back through ef_tensor_d3d11_texture"
    );
    assert!(wrapped.gpu_completion().unwrap().is_none());
    wrapped.set_gpu_write(7).unwrap();
    let completion = wrapped.gpu_completion().unwrap().expect("a recorded write");
    assert_eq!(completion.value, 7);
}

/// Two handle imports of *different* textures with the same geometry get
/// different identities even when the caller's handle values are recycled: the
/// tensor keys on the `ID3D11Texture2D` its own open produced, and a live COM
/// object's address cannot be handed to another object.
///
/// The first import's source handle is closed before the second is opened,
/// which is what makes Windows free to hand the same numeric value back.
#[test]
fn handle_imports_of_different_textures_do_not_share_an_identity() {
    let open = |t: &TensorDyn| {
        let h = t.d3d11_shared_handle().unwrap();
        // SAFETY: `h` is a shared NT handle this process owns for the length
        // of this closure; the constructor duplicates what it keeps.
        let imported = unsafe {
            TensorDyn::from_d3d11_shared_handle(
                h.as_raw_handle() as _,
                32,
                16,
                PixelFormat::Rgba,
                DType::U8,
                CpuAccess::None,
                None,
                None,
            )
        }
        .unwrap();
        drop(h);
        imported
    };
    let make = || {
        TensorDyn::image(
            32,
            16,
            PixelFormat::Rgba,
            DType::U8,
            Some(TensorMemory::DmaBuf),
            CpuAccess::None,
        )
        .unwrap()
    };
    let (a, b) = (make(), make());
    let (ia, ib) = (open(&a), open(&b));
    assert_ne!(
        ia.buffer_identity().id(),
        ib.buffer_identity().id(),
        "two imports of different textures must not collide on a recycled handle value"
    );
}

/// A write-only map has to publish the whole texture, so a window shorter than
/// the backing is refused rather than silently overwriting the rows outside
/// it. `ReadWrite` is the same window with the refresh, and it is accepted.
#[test]
fn write_only_map_of_a_narrowed_window_is_refused_and_names_readwrite() {
    let mut t = Tensor::<u8>::image(
        64,
        16,
        PixelFormat::Rgba,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    assert!(t.map_write().is_ok(), "the whole tensor is writable");
    t.set_logical_shape(&[4, 64, 4]).unwrap();
    let err = t.map_write().unwrap_err();
    assert!(
        matches!(&err, Error::InvalidArgument(m) if m.contains("CpuAccess::ReadWrite")),
        "unexpected error: {err}"
    );
    assert!(t.map_mut().is_ok(), "the read-write window is accepted");
}

/// `gpu_write_value` is the recorded value alone, with no duplicated fence
/// handle: 0 before any write, then the number `gpu_completion` carries, on
/// the typed tensor and on `TensorDyn` alike. A host tensor answers 0 rather
/// than an error.
#[test]
fn d3d11_gpu_write_value_matches_the_completion_value() {
    let t = Tensor::<u8>::image(
        32,
        16,
        PixelFormat::Rgba,
        Some(TensorMemory::DmaBuf),
        CpuAccess::None,
    )
    .unwrap();
    assert_eq!(t.gpu_write_value(), 0);
    t.set_gpu_write(11).unwrap();
    let completion = t.gpu_completion().unwrap().expect("a recorded write");
    assert_eq!(t.gpu_write_value(), completion.value);

    let t = TensorDyn::from(t);
    assert_eq!(t.gpu_write_value(), 11);
    t.set_gpu_write(12).unwrap();
    let completion = t.gpu_completion().unwrap().expect("a recorded write");
    assert_eq!(t.gpu_write_value(), completion.value);

    let host = Tensor::<u8>::new(&[16], None, None).unwrap();
    assert_eq!(host.gpu_write_value(), 0);
    assert_eq!(TensorDyn::from(host).gpu_write_value(), 0);
}

/// The Windows accessors are `None`/`NotImplemented` for a tensor that is not
/// a texture, rather than panicking or lying.
#[test]
fn non_texture_tensors_report_no_d3d11_backing() {
    let m = Tensor::<u8>::new(&[16], None, None).unwrap();
    assert!(m.d3d11_texture().is_none());
    assert!(m.d3d11_layout().is_none());
    assert!(matches!(
        m.d3d11_shared_handle(),
        Err(Error::NotImplemented(_))
    ));
    assert!(matches!(m.gpu_completion(), Err(Error::NotImplemented(_))));
}

/// `Tensor::new` with an explicit `DmaBuf` request says why a texture tensor
/// cannot serve it instead of silently allocating host memory.
#[test]
fn plain_new_with_dmabuf_names_the_image_constructor() {
    let err = Tensor::<u8>::new(&[16], Some(TensorMemory::DmaBuf), None).unwrap_err();
    assert!(
        err.to_string().contains("Tensor::image"),
        "unexpected error: {err}"
    );
}

/// A texture tensor's descriptor names the texture's own NT handle, the
/// device fence and the value of the last recorded write, and importing it
/// rebuilds a tensor over the same texture.
#[test]
fn descriptor_round_trips_a_texture_with_its_completion() {
    let t = TensorDyn::image(
        32,
        16,
        PixelFormat::Rgba,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    {
        let mut m = t.map_bytes(CpuAccess::ReadWrite).unwrap();
        m.as_mut_slice()[..4].copy_from_slice(&[9, 8, 7, 6]);
    }
    let v = edgefirst_tensor::d3d11::device()
        .unwrap()
        .signal()
        .expect("fence");
    t.set_gpu_write(v).unwrap();
    let desc = t.descriptor_pinned(None);
    assert_eq!(desc.kind, edgefirst_tensor::protocol::kind::D3D11_TEXTURE);
    assert!(desc.flags & edgefirst_tensor::protocol::flags::SYNC_PRESENT != 0);
    assert_eq!(desc.sync, v);
    let imported = TensorDyn::import_descriptor(&desc).unwrap();
    assert_eq!(imported.memory(), TensorMemory::DmaBuf);
    assert_eq!(
        &imported.map_bytes(CpuAccess::Read).unwrap().as_slice()[..4],
        &[9, 8, 7, 6]
    );

    // A pinned capsule (Python's `access="read"`) pins a texture tensor too.
    // `ptr` is the fence under this kind, so the pin's host address must not
    // displace it -- a consumer would open that address as a fence.
    let pin = t.pin_host(CpuAccess::Read).unwrap();
    assert!(!pin.as_mut_ptr().is_null(), "a texture tensor does pin");
    let pinned = t.descriptor_pinned(Some(&pin));
    assert_eq!(pinned.ptr.0, desc.ptr.0);
    assert_ne!(pinned.ptr.0, pin.as_mut_ptr());
    assert!(TensorDyn::import_descriptor(&pinned).is_ok());
}

/// A semi-planar texture is as wide as its padded row pitch, so the import's
/// image width can only come from the descriptor's shape. Both spellings of
/// that shape are accepted and the producer's own is what the consumer sees.
#[test]
fn descriptor_round_trips_a_semi_planar_texture() {
    let mut t = TensorDyn::image(
        640,
        481,
        PixelFormat::Nv12,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    let imported = TensorDyn::import_descriptor(&t.descriptor_pinned(None)).unwrap();
    assert_eq!(imported.width(), Some(640));
    assert_eq!(imported.height(), Some(481));
    assert_eq!(imported.width(), t.width());
    assert_eq!(imported.height(), t.height());
    assert_eq!(imported.shape(), t.shape());
    assert_eq!(imported.row_stride(), t.row_stride());

    // The addressing spelling of the same image: the producer's logical shape
    // is restored onto the import, not the allocation shape it opens at.
    t.set_logical_shape(&[481, 640]).unwrap();
    let imported = TensorDyn::import_descriptor(&t.descriptor_pinned(None)).unwrap();
    assert_eq!(imported.shape(), &[481, 640]);
    assert_eq!(imported.row_stride(), t.row_stride());
}

/// Copies `src`'s tight rows into a destination whose rows are `dst_stride`
/// bytes apart, so a texture tensor whose backing pitch exceeds its image row
/// is written row by row rather than as one flat block.
fn copy_rows(dst: &mut [u8], src: &[u8], row_bytes: usize, dst_stride: usize) {
    for (i, row) in src.chunks(row_bytes).enumerate() {
        dst[i * dst_stride..i * dst_stride + row.len()].copy_from_slice(row);
    }
}

/// CUDA reads a texture tensor's pixels through the external-memory import
/// attached at allocation, and a writable CUDA map publishes its buffer back
/// into the texture when the guard drops.
#[test]
fn cuda_map_matches_the_cpu_map_and_map_mut_writes_back() {
    if !edgefirst_tensor::is_cuda_available() {
        eprintln!("SKIP: no CUDA runtime");
        return;
    }
    let device = edgefirst_tensor::d3d11::device().unwrap();
    if device.is_warp() {
        // `cudaD3D11GetDevice` has no ordinal for the WARP adapter, so the
        // import fails and the tensor is CUDA-less by design.
        eprintln!("SKIP: WARP adapter has no CUDA device");
        return;
    }
    let t = Tensor::<u8>::image(
        256,
        128,
        PixelFormat::Rgba,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    let pattern: Vec<u8> = (0..256 * 128 * 4).map(|i| (i * 7 + 13) as u8).collect();
    {
        let mut m = t.map().unwrap();
        copy_rows(
            m.as_mut_slice(),
            &pattern,
            256 * 4,
            t.row_stride().unwrap_or(256 * 4),
        );
    }
    let v = device.signal().unwrap();
    // A CPU write needs no fence, but the map must tolerate a recorded value.
    t.set_gpu_write(v).unwrap();
    let cm = t.cuda_map().expect("CUDA handle attached at allocation");
    assert_eq!(cm.len(), 256 * 128 * 4);
    let mut host = vec![0u8; cm.len()];
    // SAFETY: `host` holds `cm.len()` writable bytes and `cm.device_ptr()` is
    // the live CUDA buffer of exactly that size for as long as `cm` lives.
    assert!(unsafe {
        edgefirst_tensor::memcpy_device_to_host(
            host.as_mut_ptr().cast(),
            cm.device_ptr(),
            host.len(),
        )
    });
    assert_eq!(host, pattern);
    drop(cm);
    {
        let cm = t.cuda_map_mut().unwrap();
        let ones = vec![1u8; cm.len()];
        // SAFETY: mirror of the read above -- `ones` holds `cm.len()` readable
        // bytes and the device pointer is live for the guard's lifetime.
        assert!(unsafe {
            edgefirst_tensor::memcpy_host_to_device(
                cm.device_ptr(),
                ones.as_ptr().cast(),
                ones.len(),
            )
        });
    }
    // Every row, not just the first: a release that copies the wrong row count
    // or the wrong row length leaves the tail of the image untouched.
    let m = t.map_read().unwrap();
    let stride = t.row_stride().unwrap_or(256 * 4);
    for row in 0..128 {
        let at = row * stride;
        assert!(
            m.as_slice()[at..at + 256 * 4].iter().all(|&b| b == 1),
            "row {row} was not written back"
        );
    }
}

/// A geometry whose D3D11 allocation the driver pads still imports: the
/// declared external-memory size has to cover that padding, and 640x480 is one
/// of the shapes whose padding-free byte count the driver rejects (the brief's
/// 256x128 is one it accepts, so only this test covers the fallback).
#[test]
fn cuda_map_covers_a_texture_whose_allocation_the_driver_padded() {
    if !edgefirst_tensor::is_cuda_available() {
        eprintln!("SKIP: no CUDA runtime");
        return;
    }
    if edgefirst_tensor::d3d11::device().unwrap().is_warp() {
        eprintln!("SKIP: WARP adapter has no CUDA device");
        return;
    }
    let (w, h) = (640usize, 480usize);
    let t = Tensor::<u8>::image(
        w,
        h,
        PixelFormat::Rgba,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    let pattern: Vec<u8> = (0..w * h * 4).map(|i| (i % 251) as u8).collect();
    {
        let mut m = t.map().unwrap();
        copy_rows(
            m.as_mut_slice(),
            &pattern,
            w * 4,
            t.row_stride().unwrap_or(w * 4),
        );
    }
    let cm = t.cuda_map().expect("the padded declaration imports");
    assert_eq!(cm.len(), w * h * 4);
    let mut host = vec![0u8; cm.len()];
    // SAFETY: `host` holds `cm.len()` writable bytes and `cm.device_ptr()` is
    // the live CUDA buffer of exactly that size while `cm` lives.
    assert!(unsafe {
        edgefirst_tensor::memcpy_device_to_host(
            host.as_mut_ptr().cast(),
            cm.device_ptr(),
            host.len(),
        )
    });
    assert_eq!(host, pattern);
}

/// The semi-planar arm: NV12 is one R8 texture of the combined plane, whose
/// texture width *is* the row stride, so the CUDA mapping and the CPU map are
/// laid out identically and the whole plane round-trips byte for byte.
#[test]
fn cuda_map_reads_the_nv12_combined_plane() {
    if !edgefirst_tensor::is_cuda_available() {
        eprintln!("SKIP: no CUDA runtime");
        return;
    }
    if edgefirst_tensor::d3d11::device().unwrap().is_warp() {
        eprintln!("SKIP: WARP adapter has no CUDA device");
        return;
    }
    let (w, h) = (640usize, 480usize);
    let t = Tensor::<u8>::image(
        w,
        h,
        PixelFormat::Nv12,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    // The texture's own row, which for a semi-planar backing is also the row
    // a CPU map sees -- the allocator widens the texture to the driver's pitch
    // precisely so the two are one number, and `row_stride()` is only recorded
    // when it exceeds the natural stride.
    let layout = t.d3d11_layout().unwrap();
    let stride = layout.tight_row_bytes();
    assert_eq!(t.row_stride().unwrap_or(stride), stride);
    let rows = layout.texture_height;
    assert_eq!(rows, PixelFormat::Nv12.allocation_shape(w, h).unwrap()[0]);
    let pattern: Vec<u8> = (0..stride * rows).map(|i| (i % 253) as u8).collect();
    {
        let mut m = t.map().unwrap();
        copy_rows(m.as_mut_slice(), &pattern, stride, stride);
    }
    let cm = t.cuda_map().expect("the combined plane imports");
    assert_eq!(
        cm.len(),
        stride * rows,
        "the mapping is the whole combined plane at the texture's own row"
    );
    let mut host = vec![0u8; cm.len()];
    // SAFETY: `host` holds `cm.len()` writable bytes and `cm.device_ptr()` is
    // the live CUDA buffer of exactly that size while `cm` lives.
    assert!(unsafe {
        edgefirst_tensor::memcpy_device_to_host(
            host.as_mut_ptr().cast(),
            cm.device_ptr(),
            host.len(),
        )
    });
    assert_eq!(host, pattern);
}

/// Reference-mode blob transport of a texture, in one process: there is no fd
/// table to carry an NT handle, so the handle values ride in the plane record
/// and the header's pid says whose handle table they belong to.
#[test]
fn blob_export_import_shares_a_texture_in_process() {
    let t = TensorDyn::image(
        32,
        16,
        PixelFormat::Rgba,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    {
        let mut m = t.map_bytes(CpuAccess::ReadWrite).unwrap();
        m.as_mut_slice()[..2].copy_from_slice(&[3, 4]);
    }
    let (blob, fds) = edgefirst_tensor::blob::export(&t, TransportMode::Reference).unwrap();
    assert!(
        fds.is_empty(),
        "an NT handle does not travel in an fd table"
    );

    let view = edgefirst_tensor::blob::BlobView::parse(&blob).unwrap();
    assert_eq!(view.header().pid, std::process::id());
    let planes = view.planes().unwrap();
    assert!(planes
        .iter()
        .all(|p| p.handle_bytes.len() == edgefirst_tensor::blob::D3D11_HANDLE_BYTES));

    let back = edgefirst_tensor::blob::import(&blob, &[]).unwrap();
    assert_eq!(back.memory(), TensorMemory::DmaBuf);
    assert_eq!(back.format(), Some(PixelFormat::Rgba));
    assert_eq!(back.shape(), t.shape());
    assert_eq!(
        &back.map_bytes(CpuAccess::Read).unwrap().as_slice()[..2],
        &[3, 4]
    );
}

/// A texture import keeps the pitch its own device reports, not the producer's
/// that the blob happens to carry: the two are facts about different drivers,
/// and `from_d3d11_shared_handle` has already recorded the local one. The
/// descriptor path makes the same exclusion (`restore_imported_row_stride` is
/// `HOST | DMABUF` only), so a blob stride saying otherwise must not move it.
#[test]
fn blob_import_keeps_the_local_pitch_not_the_blobs_stride() {
    let (w, h) = (37usize, 16usize);
    let t = TensorDyn::image(
        w,
        h,
        PixelFormat::Rgba,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    let (blob, _) = edgefirst_tensor::blob::export(&t, TransportMode::Reference).unwrap();

    // What a fresh open of this very texture records for itself -- the answer
    // the import has to match, whatever the blob says.
    let handle = t.d3d11_shared_handle().unwrap();
    // SAFETY: `handle` is a live shared NT handle of this process's texture
    // and `t` holds the texture alive across the call.
    let direct = unsafe {
        TensorDyn::from_d3d11_shared_handle(
            handle.as_raw_handle(),
            w,
            h,
            PixelFormat::Rgba,
            DType::U8,
            CpuAccess::ReadWrite,
            None,
            None,
        )
    }
    .unwrap();

    // A plausible but wrong pitch in the blob's plane 0 record, whose `stride`
    // is the third of its six scalars.
    let view = edgefirst_tensor::blob::BlobView::parse(&blob).unwrap();
    let regions = edgefirst_tensor::blob::region_offsets(&view.header()).unwrap();
    let carried = view.planes().unwrap()[0].stride;
    let bogus = carried + 64;
    let at = regions.planes + 16;
    let mut tampered = blob.clone();
    tampered[at..at + 8].copy_from_slice(&bogus.to_le_bytes());

    let back = edgefirst_tensor::blob::import(&tampered, &[]).unwrap();
    assert_eq!(
        back.row_stride(),
        direct.row_stride(),
        "the import keeps the pitch this device reports for the texture"
    );
    // Through `effective_row_stride`, which is a real number on every adapter:
    // WARP pads nothing at this width, so `row_stride()` is `None` there and
    // comparing it alone would prove nothing.
    assert_eq!(
        back.effective_row_stride(),
        direct.effective_row_stride(),
        "and reports the same row as a fresh open of the same texture"
    );
    assert_ne!(
        back.effective_row_stride(),
        Some(bogus as usize),
        "the blob's stride field must not reach the imported tensor"
    );
}

/// Removes the blob file on every path out of the parent, panic included.
struct TempBlob(std::path::PathBuf);

impl Drop for TempBlob {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The cross-process half: the child re-runs this test binary with
/// `EF_D3D11_CHILD` naming the blob file. Its D3D11 device is its own, so the
/// import can only work by duplicating both NT handles out of the exporting
/// process the header names and opening the texture on the child's device.
///
/// The parent records a GPU write before exporting, so the child's import
/// opens the *fence* out of the parent too and waits on it -- which is only
/// reachable because `signal()` submits the context it signalled on. The
/// child then re-signals its own fence behind that wait, so what it reports
/// is a value on its own timeline rather than the parent's.
#[test]
fn blob_import_from_another_process_duplicates_the_handles() {
    if let Ok(path) = std::env::var("EF_D3D11_CHILD") {
        return child_imports_the_exported_blob(&path);
    }
    let t = TensorDyn::image(
        32,
        16,
        PixelFormat::Rgba,
        DType::U8,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .unwrap();
    {
        let mut m = t.map_bytes(CpuAccess::ReadWrite).unwrap();
        m.as_mut_slice()[..2].copy_from_slice(&[5, 6]);
    }
    let value = edgefirst_tensor::d3d11::device()
        .unwrap()
        .signal()
        .expect("fence");
    t.set_gpu_write(value).unwrap();

    let (blob, _) = edgefirst_tensor::blob::export(&t, TransportMode::Reference).unwrap();
    let file =
        TempBlob(std::env::temp_dir().join(format!("ef-d3d11-blob-{}.bin", std::process::id())));
    std::fs::write(&file.0, &blob).unwrap();
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "blob_import_from_another_process_duplicates_the_handles",
            "--exact",
            "--nocapture",
        ])
        .env("EF_D3D11_CHILD", &file.0)
        .status()
        .unwrap();
    // Export borrows: the handle values in the blob are this process's, so
    // the exporting tensor has to outlive the child that reads them.
    drop(t);
    assert!(
        status.success(),
        "child could not import the exported texture"
    );
}

fn child_imports_the_exported_blob(path: &str) {
    let blob = std::fs::read(path).expect("the parent wrote the blob");
    let t = edgefirst_tensor::blob::import(&blob, &[]).expect("import in the child");
    assert_eq!(t.memory(), TensorMemory::DmaBuf);

    // The parent's fence handle was duplicated out of it and reopened here, so
    // the import could wait on the parent's value -- and then record a value
    // of its *own* device's timeline, which is the only one the fence this
    // completion names ever reaches. The two numbers are unrelated: asserting
    // equality would pin the cross-process hang the local re-signal removes.
    let completion = t
        .gpu_completion()
        .expect("gpu_completion")
        .expect("the blob carried a completion, so the import recorded a local one");
    assert_ne!(
        completion.value, 0,
        "the import recorded a value on this process's fence"
    );

    // Reachable on the fence it came back with, which is the pairing every
    // consumer relies on: the parent's value on the child's fence would never
    // arrive.
    let d = edgefirst_tensor::d3d11::device().unwrap();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    while d.completed_value() < completion.value {
        assert!(
            std::time::Instant::now() < deadline,
            "the recorded value {} never completed on this process's fence (at {})",
            completion.value,
            d.completed_value()
        );
        std::thread::yield_now();
    }

    assert_eq!(
        &t.map_bytes(CpuAccess::Read).unwrap().as_slice()[..2],
        &[5, 6]
    );

    // With CUDA present the same ordering has to hold for the device path: a
    // completion on a foreign timeline made this wait forever.
    if edgefirst_tensor::is_cuda_available() {
        match t.cuda_map() {
            Some(m) => assert!(
                !m.device_ptr().is_null(),
                "the CUDA map has a device pointer"
            ),
            None => eprintln!("no CUDA registration on the imported texture -- skipping"),
        }
    }
}
