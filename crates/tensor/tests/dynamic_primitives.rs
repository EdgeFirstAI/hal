// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Tests that the `dynamic` backend's Rust wrappers reach working
//! `ef_tensor_*` primitives.
//!
//! # Running this file
//!
//! The test binary links against `libedgefirst_tensor`; build it first:
//!
//! ```sh
//! make test-tensor-dynamic
//! ```

#![cfg(edgefirst_tensor_dynamic)]

use edgefirst_tensor::{
    CpuAccess, DType, Error, PixelFormat, Quantization, Region, Tensor, TensorDyn, TensorMapTrait,
    TensorMemory, TensorTrait,
};

/// Allocation policy: a failure the host's probe did not predict fails the test.
#[path = "support/outcome.rs"]
#[allow(dead_code)]
mod outcome;

/// The DMA-heap skip policy: a skip under `HAL_TEST_REQUIRE_DMA=1` fails.
#[path = "support/dma_require.rs"]
mod dma_require;

/// `true` when `what` should run: a usable DMA heap, or a skip reported
/// (and refused under `HAL_TEST_REQUIRE_DMA=1`).
fn dma_or_skip(what: &str) -> bool {
    dma_require::available_or_skip(edgefirst_tensor::is_dma_available(), what)
}

/// Reports a test skip by writing `SKIPPED: {reason}` directly to stderr.
#[cfg(target_os = "windows")]
fn report_skip(reason: &str) {
    use std::io::Write;
    let _ = writeln!(&mut std::io::stderr(), "WARNING: SKIPPED: {reason}");
}

/// Allocates a bare (formatless) `Tensor<u8>` of the given shape.
fn bare_u8(shape: &[usize]) -> Tensor<u8> {
    Tensor::<u8>::new(shape, None, None).expect("bare tensor allocation")
}

// --- set_format / set_row_stride / set_plane_offset -------------

#[test]
fn set_format_set_row_stride_and_set_plane_offset_take_effect() {
    // NV12 combined-plane shape: [H + H/2, W].
    let (w, h) = (64usize, 48usize);
    let mut t: TensorDyn = bare_u8(&[h + h / 2, w]).into();

    // Before `set_format` there is no pixel format.
    assert_eq!(t.format(), None);

    t.set_format(PixelFormat::Nv12)
        .expect("set_format must succeed against a real ef_tensor_set_format primitive");
    assert_eq!(t.format(), Some(PixelFormat::Nv12));

    t.set_row_stride(128)
        .expect("set_row_stride must succeed against a real ef_tensor_set_row_stride primitive");
    assert_eq!(
        t.effective_row_stride(),
        Some(128),
        "the padded stride must be visible back through the handle"
    );

    assert_eq!(t.plane_offset(), None);
    t.set_plane_offset(4096);
    assert_eq!(
        t.plane_offset(),
        Some(4096),
        "set_plane_offset must be visible back through the handle, not a silent no-op"
    );
}

// --- quantization -------------------------------------------

#[test]
fn quantization_reads_back_metadata_set_by_another_producer() {
    // SAFETY: every call below either takes a value or a pointer to local
    // data valid for the call's duration; the resulting handle is freed
    // exactly once, via `TensorDyn::from_raw`'s `Drop`.
    let handle = unsafe {
        let b = edgefirst_tensor_ffi::ef_tensor_builder_new();
        assert!(!b.is_null());
        let dims = [4u64, 4];
        assert_eq!(
            edgefirst_tensor_ffi::ef_tensor_builder_dtype(b, DType::U8.code()),
            0
        );
        assert_eq!(
            edgefirst_tensor_ffi::ef_tensor_builder_shape(b, dims.as_ptr(), 2),
            0
        );
        assert_eq!(edgefirst_tensor_ffi::ef_tensor_builder_storage(b, 0), 0); // Mem
        let scales = [0.1f32, 0.2, 0.3, 0.4];
        let zps = [1i32, -2, 3, -4];
        assert_eq!(
            edgefirst_tensor_ffi::ef_tensor_builder_quantization(
                b,
                1,
                scales.as_ptr(),
                zps.as_ptr(),
                4
            ),
            0
        );
        let raw = edgefirst_tensor_ffi::ef_tensor_builder_alloc(b);
        edgefirst_tensor_ffi::ef_tensor_builder_free(b);
        assert!(!raw.is_null(), "producer-side allocation must succeed");
        TensorDyn::from_raw(raw)
    };

    let typed = handle
        .as_typed::<u8>()
        .expect("the handle's dtype must be u8");
    let q = typed.quantization().expect(
        "quantization set by another producer via ef_tensor_builder_quantization must \
                  be readable back through Tensor::quantization",
    );
    assert_eq!(q.axis(), Some(1));
    assert_eq!(q.scale(), &[0.1, 0.2, 0.3, 0.4]);
    assert_eq!(q.zero_point(), Some(&[1, -2, 3, -4][..]));

    // `set_quantization` and `clear_quantization` on a self-allocated tensor.
    let mut own = bare_u8(&[4, 4]);
    assert_eq!(own.quantization(), None);
    let per_tensor = Quantization::per_tensor(0.5, -7);
    own.set_quantization(per_tensor.clone()).expect(
        "set_quantization must succeed against a real ef_tensor_quantization_set primitive",
    );
    assert_eq!(own.quantization(), Some(&per_tensor));
    own.clear_quantization();
    assert_eq!(
        own.quantization(),
        None,
        "clear_quantization must be visible back through the handle"
    );
}

// --- configure_image -----------------------------------------

#[test]
fn configure_image_reconfigures_an_oversized_pool_tensor() {
    // An oversized NV12 pool buffer (128x128), reconfigured to a smaller Grey frame.
    let mut pool = bare_u8(&[128 + 128 / 2, 128]);
    pool.configure_image(64, 64, PixelFormat::Grey)
        .expect("configure_image must succeed against a real ef_tensor_configure_image primitive");
    assert_eq!(pool.format(), Some(PixelFormat::Grey));
    assert_eq!(pool.width(), Some(64));
    assert_eq!(pool.height(), Some(64));
}

// --- image / image_desc / image_with_stride / view --------------

#[test]
fn image_allocates_a_real_tensor() {
    let img = Tensor::<u8>::image(64, 48, PixelFormat::Grey, None, CpuAccess::None)
        .expect("image must succeed against a real ef_tensor_image_alloc primitive");
    assert_eq!(img.format(), Some(PixelFormat::Grey));
    assert_eq!(img.width(), Some(64));
    assert_eq!(img.height(), Some(48));
}

#[test]
fn image_desc_allocates_from_a_declarative_request() {
    let desc = edgefirst_tensor::ImageDesc::new(64, 48, PixelFormat::Rgb, DType::U8);
    let img = Tensor::<u8>::image_desc(&desc)
        .expect("image_desc must succeed against a real ef_tensor_image_desc_alloc primitive");
    assert_eq!(img.format(), Some(PixelFormat::Rgb));
    assert_eq!(img.width(), Some(64));
    assert_eq!(img.height(), Some(48));
}

#[test]
fn view_region_is_a_real_zero_copy_window_and_is_bounds_checked() {
    // A byte-level round-trip on a `Mem`-backed parent.
    let parent = Tensor::<u8>::image(
        8,
        8,
        PixelFormat::Grey,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("parent image allocation");
    {
        let mut m = parent.map_with(CpuAccess::ReadWrite).expect("map parent");
        m.as_mut_slice().fill(0);
    }

    // Top-left corner: the view's first byte is the parent's first byte.
    let view = parent
        .view(Region::new(0, 0, 4, 4))
        .expect("view must succeed against a real ef_tensor_view_region primitive");
    {
        let mut m = view.map_with(CpuAccess::ReadWrite).expect("map view");
        m.as_mut_slice()[0] = 0xAB;
    }

    {
        let m = parent.map_with(CpuAccess::Read).expect("re-map parent");
        assert_eq!(
            m.as_slice()[0],
            0xAB,
            "a write through the view must be visible through the parent -- \
             ef_tensor_view_region must share the parent's real storage, not a fresh copy"
        );
    }

    let out_of_bounds = parent.view(Region::new(0, 0, 1000, 1000));
    assert!(
        matches!(out_of_bounds, Err(Error::RegionOutOfBounds { .. })),
        "an out-of-bounds region must be rejected, not silently clamped: got {out_of_bounds:?}"
    );
}

// --- from_planes -----------------------------------------------

#[test]
fn from_planes_combines_two_independently_allocated_tensors() {
    // NV12: chroma height is luma height / 2, same width.
    let (w, h) = (64usize, 48usize);
    let luma = bare_u8(&[h, w]);
    let chroma = bare_u8(&[h / 2, w]);

    let combined = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)
        .expect("from_planes must succeed against a real ef_tensor_from_planes primitive");
    assert_eq!(combined.format(), Some(PixelFormat::Nv12));
}

#[test]
fn from_planes_is_genuinely_multiplane_and_chroma_is_independently_writable() {
    let (w, h) = (64usize, 48usize);
    let luma = bare_u8(&[h, w]);
    let mut chroma = bare_u8(&[h / 2, w]);
    chroma.set_row_stride_unchecked(w + 16);
    let combined = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)
        .expect("from_planes must succeed");
    assert!(
        combined.is_multiplane(),
        "from_planes stores chroma on the library tensor -- is_multiplane() must be true \
         even for Mem, and after from_raw"
    );
    assert_eq!(
        combined.chroma().and_then(|c| c.effective_row_stride()),
        Some(w + 16),
        "chroma stride set before from_planes must read back through chroma()"
    );

    let raw = TensorDyn::from(combined).into_raw();
    let wrapped = unsafe { TensorDyn::from_raw(raw) };
    assert!(
        wrapped.is_multiplane(),
        "from_raw after from_planes must still report is_multiplane"
    );
    let chroma = wrapped
        .chroma()
        .expect("from_raw must yield a working chroma() handle");
    assert_eq!(chroma.effective_row_stride(), Some(w + 16));

    let owned = wrapped
        .chroma_dyn()
        .expect("chroma_dyn must yield the chroma plane, as under the static backend");
    assert_eq!(owned.effective_row_stride(), Some(w + 16));
    assert_eq!(owned.shape(), &[h / 2, w]);
}

#[test]
fn dma_multiplane_from_planes_makes_is_multiplane_and_chroma_honest() {
    if !dma_or_skip("dma_multiplane_from_planes_makes_is_multiplane_and_chroma_honest") {
        return;
    }
    let (w, h) = (64usize, 48usize);
    let luma =
        Tensor::<u8>::new(&[h, w], Some(TensorMemory::DmaBuf), None).expect("dma luma allocation");
    let mut chroma = Tensor::<u8>::new(&[h / 2, w], Some(TensorMemory::DmaBuf), None)
        .expect("dma chroma allocation");
    chroma.set_row_stride_unchecked(w + 16);

    let combined = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)
        .expect("from_planes must succeed against a real ef_tensor_from_planes primitive");

    assert!(
        combined.is_multiplane(),
        "a DMA-backed from_planes tensor is genuinely two allocations -- is_multiplane() must \
         say so, not silently take the combined-plane path a real caller would misread chroma \
         bytes through"
    );

    assert_eq!(
        combined.chroma().and_then(|c| c.effective_row_stride()),
        Some(w + 16),
        "the stride set on chroma before from_planes must read back through chroma()"
    );

    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let combined = TensorDyn::from(combined);
        let luma_fd = combined
            .dmabuf()
            .expect("a DMA-backed TensorDyn lends its DMA-BUF");
        let chroma = combined
            .chroma_dyn()
            .expect("chroma_dyn on a DMA multiplane tensor");
        let chroma_fd = chroma
            .dmabuf()
            .expect("the chroma plane lends its own DMA-BUF");
        assert_ne!(
            luma_fd.as_raw_fd(),
            chroma_fd.as_raw_fd(),
            "the two planes are separate allocations"
        );
    }
}

// --- BufferIdentity for DMA-backed handles ---------------------------------

#[test]
fn dma_buffer_identity_is_derived_from_the_inode_not_the_handle_address() {
    if !dma_or_skip("dma_buffer_identity_is_derived_from_the_inode_not_the_handle_address") {
        return;
    }
    let t = Tensor::<u8>::new(&[4096], Some(TensorMemory::DmaBuf), None)
        .expect("dma tensor allocation");
    assert_eq!(
        t.buffer_identity().kind(),
        edgefirst_tensor::IdentityKind::DmaBuf,
        "a DMA-BUF handle's identity must be inode-derived, not the process-local HostPtr \
         fallback -- see TensorDyn::derive_identity's doc comment"
    );
}

/// A D3D11 texture handle's identity names its texture, across create-and-drop cycles.
#[cfg(target_os = "windows")]
#[test]
fn a_texture_handles_identity_names_its_texture_not_its_recyclable_handle_address() {
    use std::collections::HashSet;

    const ITERATIONS: usize = 200;

    // `Some(DmaBuf)` is the Windows texture request.
    if !edgefirst_tensor::is_gpu_buffer_available() {
        report_skip(
            "a_texture_handles_identity_names_its_texture_not_its_recyclable_handle_address \
             - no D3D11 device on this host",
        );
        return;
    }

    // The id `BufferIdentity::derived` builds: the kind's discriminant in the
    // high bits, the key below it.
    let expected_id =
        |tex: usize| ((edgefirst_tensor::IdentityKind::D3d11Texture as u64) << 56) ^ (tex as u64);

    let mut ids = HashSet::new();
    let mut recycled = 0usize;
    for i in 0..ITERATIONS {
        // One pair per iteration, both dropped at the end of it.
        let mut pair = Vec::new();
        for format in [PixelFormat::Rgba, PixelFormat::Rgb] {
            let t = Tensor::<u8>::image(
                64,
                48,
                format,
                Some(TensorMemory::DmaBuf),
                CpuAccess::ReadWrite,
            )
            .expect("image allocation");
            let tex = t.d3d11_texture().expect("a texture-backed handle") as usize;
            let identity = t.buffer_identity();
            assert_eq!(
                identity.kind(),
                edgefirst_tensor::IdentityKind::D3d11Texture,
                "iteration {i} {format:?}: a texture handle's identity must name its \
                 ID3D11Texture2D, not the process-local HostPtr fallback over the handle's \
                 own address -- an EGLImage cache keyed on that address serves the previous \
                 texture's image and the convert leaves the destination blank"
            );
            assert_eq!(
                identity.id(),
                expected_id(tex),
                "iteration {i} {format:?}: the identity must be the texture pointer keyed as \
                 D3d11Texture, the same value the static backend's tex_key derives, so one \
                 texture has one identity whichever backend names it"
            );
            if !ids.insert(identity.id()) {
                recycled += 1;
            }
            pair.push(t);
        }
        // Distinct live buffers have distinct identities.
        assert_ne!(
            pair[0].buffer_identity().id(),
            pair[1].buffer_identity().id(),
            "iteration {i}: two simultaneously live textures must not share an identity"
        );
    }
    eprintln!(
        "{ITERATIONS} texture pairs: {} distinct identities over {} tensors, \
         {recycled} landed on an address a dropped texture had used",
        ids.len(),
        ITERATIONS * 2
    );
}

#[test]
fn two_distinct_views_of_one_dma_parent_share_one_identity_and_do_not_collide_with_an_unrelated_buffer(
) {
    // Checks parent<->view_a, parent<->view_b and view_a<->view_b, plus an
    // unrelated buffer.
    if !dma_or_skip(
        "two_distinct_views_of_one_dma_parent_share_one_identity_and_do_not_collide_with_an_unrelated_buffer",
    ) {
        return;
    }
    let (w, h) = (64usize, 32usize);
    let parent = Tensor::<u8>::image(
        w,
        h,
        PixelFormat::Grey,
        Some(TensorMemory::DmaBuf),
        CpuAccess::ReadWrite,
    )
    .expect("dma parent image allocation");
    let view_a = parent
        .view(Region::new(0, 0, w, h / 2))
        .expect("view_a must succeed");
    let view_b = parent
        .view(Region::new(0, h / 2, w, h / 2))
        .expect("view_b must succeed");

    let parent_id = parent.buffer_identity().id();
    let a_id = view_a.buffer_identity().id();
    let b_id = view_b.buffer_identity().id();
    assert_eq!(parent_id, a_id, "view_a must share its parent's identity");
    assert_eq!(parent_id, b_id, "view_b must share its parent's identity");
    assert_eq!(
        a_id, b_id,
        "two sibling views of ONE parent must resolve to the SAME identity as each other, not \
         just each individually matching the parent -- a per-view address-derived identity \
         would fail exactly this check while still passing a parent-only comparison"
    );
    assert!(
        parent.aliases(&view_a) && parent.aliases(&view_b) && view_a.aliases(&view_b),
        "aliases() must agree with buffer_identity() for every pair"
    );

    let unrelated = Tensor::<u8>::new(&[w * h], Some(TensorMemory::DmaBuf), None)
        .expect("unrelated dma allocation");
    assert_ne!(
        parent_id,
        unrelated.buffer_identity().id(),
        "a genuinely different DMA-BUF must not collide with the parent's identity"
    );
}

// --- PBO / CUDA / typed-downcast primitive family --------------------------
//
// `Tensor::pbo_id`/`from_pbo`/`set_cuda_handle`, `TensorDyn::as_u8`/`as_i8`/
// `as_f32`/`as_f16`/`as_i16`/`into_u8`/... under `dynamic`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Mock [`edgefirst_tensor::PboOps`] backed by a `Vec<u8>`. Counts calls.
struct MockPboOps {
    storage: Mutex<Vec<u8>>,
    maps: AtomicUsize,
    unmaps: AtomicUsize,
    deletes: AtomicUsize,
}

impl MockPboOps {
    fn new(size: usize) -> Arc<Self> {
        Arc::new(Self {
            storage: Mutex::new(vec![0u8; size]),
            maps: AtomicUsize::new(0),
            unmaps: AtomicUsize::new(0),
            deletes: AtomicUsize::new(0),
        })
    }

    /// How many times `delete_buffer` has fired.
    fn delete_count(&self) -> usize {
        self.deletes.load(Ordering::Acquire)
    }
}

// SAFETY: the returned pointer addresses a `Vec<u8>` allocated once in `new`
// and never resized, and the tensor holds an `Arc` to the `MockPboOps`.
unsafe impl edgefirst_tensor::PboOps for MockPboOps {
    fn map_buffer(
        &self,
        _buffer_id: u32,
        size: usize,
    ) -> edgefirst_tensor::Result<edgefirst_tensor::PboMapping> {
        self.maps.fetch_add(1, Ordering::AcqRel);
        let storage = self.storage.lock().expect("lock");
        assert_eq!(storage.len(), size, "mock PBO size mismatch");
        Ok(edgefirst_tensor::PboMapping {
            ptr: storage.as_ptr() as *mut u8,
            size,
        })
    }

    fn unmap_buffer(&self, _buffer_id: u32) -> edgefirst_tensor::Result<()> {
        self.unmaps.fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    fn delete_buffer(&self, _buffer_id: u32) {
        self.deletes.fetch_add(1, Ordering::AcqRel);
    }
}

/// `Tensor::from_pbo` round-trips: `memory()`, `pbo_id()`, `pbo_is_mapped()` and a CPU map.
#[test]
fn dynamic_pbo_wraps_and_reads_back_through_the_typed_tensor() {
    let ops = MockPboOps::new(16);
    let pbo = edgefirst_tensor::PboTensor::<u8>::from_pbo(7, 16, &[4, 4, 1], Some("t18_pbo"), ops)
        .expect("PboTensor::from_pbo");
    let mut tensor = Tensor::<u8>::from_pbo(pbo).expect("Tensor::from_pbo must allocate cleanly");

    assert_eq!(
        TensorTrait::memory(&tensor),
        TensorMemory::Pbo,
        "a PBO-backed dynamic tensor must report TensorMemory::Pbo, which the library's \
         own TensorStorage::Pbo answers with no override on this side"
    );
    assert_eq!(
        tensor.pbo_id(),
        Some(7),
        "pbo_id must preserve the real GL buffer id"
    );
    assert_eq!(
        tensor.pbo_is_mapped(),
        Some(false),
        "a freshly wrapped PBO holds no CPU mapping"
    );

    tensor
        .set_format(PixelFormat::Grey)
        .expect("set_format on a PBO-backed tensor (real edgefirst-image call shape)");

    {
        let mut map = tensor.map().expect("map the PBO through the mock ops");
        assert_eq!(
            tensor.pbo_is_mapped(),
            Some(true),
            "pbo_is_mapped must see the outstanding map -- this is the fact bind_dst \
             refuses a convert on"
        );
        map.as_mut_slice().fill(0xEE);
    }
    assert_eq!(
        tensor.pbo_is_mapped(),
        Some(false),
        "...and must go back to false once the map drops, or every later convert \
         to this tensor would be refused"
    );
    // A second map reads the write back from the mock's storage.
    let map = tensor.map().expect("second map");
    assert!(
        map.as_slice().iter().all(|&b| b == 0xEE),
        "the byte written through the first PBO map must be visible through a fresh one"
    );
    drop(map);

    let dyn_tensor: TensorDyn = tensor.into();
    assert_eq!(
        dyn_tensor.pbo_id(),
        Some(7),
        "the erased TensorDyn answers the same buffer id"
    );
}

/// A client callback channel for `from_client_pbo`: counts retains and
/// releases and maps a `Vec<u8>`.
mod client_pbo {
    use std::ffi::{c_int, c_void};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    pub static RETAINS: AtomicUsize = AtomicUsize::new(0);
    pub static RELEASES: AtomicUsize = AtomicUsize::new(0);
    pub static STORAGE: Mutex<Vec<u8>> = Mutex::new(Vec::new());

    pub unsafe extern "C" fn retain(_ctx: *const c_void) {
        RETAINS.fetch_add(1, Ordering::AcqRel);
    }
    pub unsafe extern "C" fn release(_ctx: *const c_void) {
        RELEASES.fetch_add(1, Ordering::AcqRel);
    }
    pub unsafe extern "C" fn map(
        _ctx: *const c_void,
        _id: u32,
        size: usize,
        out_ptr: *mut *mut u8,
        out_len: *mut usize,
    ) -> c_int {
        let mut s = STORAGE.lock().expect("lock");
        s.resize(size, 0);
        unsafe {
            *out_ptr = s.as_mut_ptr();
            *out_len = size;
        }
        0
    }
    pub unsafe extern "C" fn unmap(_ctx: *const c_void, _id: u32) -> c_int {
        0
    }

    /// Non-NULL and never dereferenced.
    pub static CTX: u8 = 0;
}

#[test]
fn dynamic_from_client_pbo_crosses_the_abi_and_maps_through_the_clients_callback() {
    use edgefirst_tensor::{EfClientState, TensorMapTrait};
    use std::sync::atomic::Ordering;

    client_pbo::RETAINS.store(0, Ordering::Release);
    client_pbo::RELEASES.store(0, Ordering::Release);

    let state = EfClientState {
        ctx: std::ptr::addr_of!(client_pbo::CTX) as *const std::ffi::c_void,
        retain: Some(client_pbo::retain),
        release: Some(client_pbo::release),
    };

    // SAFETY: `state`, `map` and `unmap` are the module above's own valid
    // channel; `CTX` is a `'static` the callbacks never dereference.
    let tensor = unsafe {
        TensorDyn::from_client_pbo(
            state,
            client_pbo::map,
            client_pbo::unmap,
            11,
            64,
            &[64],
            DType::U8,
        )
    }
    .expect("from_client_pbo must cross ef_tensor_wrap_pbo");

    assert_eq!(
        tensor.memory(),
        TensorMemory::Pbo,
        "the library must report PBO storage, not the host memory it would          have allocated if it ignored the channel"
    );
    assert_eq!(
        client_pbo::RETAINS.load(Ordering::Acquire),
        1,
        "the library retains the channel exactly once"
    );

    {
        let mut map = tensor
            .map_bytes(edgefirst_tensor::CpuAccess::ReadWrite)
            .expect("map a wrapped PBO through the client's map_fn");
        map.as_mut_slice()[0] = 0xC3;
    }
    assert_eq!(
        client_pbo::STORAGE.lock().expect("lock")[0],
        0xC3,
        "the write landed in this side's own buffer, so the map really went          out through map_fn rather than into library-allocated host memory"
    );

    drop(tensor);
    assert_eq!(
        client_pbo::RELEASES.load(Ordering::Acquire),
        1,
        "the channel is released exactly once, when the handle is freed"
    );
}

/// A NULL `ctx`, `retain` or `release` is refused with `Error::InvalidArgument`.
#[test]
fn dynamic_from_client_pbo_refuses_an_incomplete_channel() {
    use edgefirst_tensor::EfClientState;

    let good = EfClientState {
        ctx: std::ptr::addr_of!(client_pbo::CTX) as *const std::ffi::c_void,
        retain: Some(client_pbo::retain),
        release: Some(client_pbo::release),
    };
    let broken = [
        EfClientState {
            ctx: std::ptr::null(),
            ..good
        },
        EfClientState {
            retain: None,
            ..good
        },
        EfClientState {
            release: None,
            ..good
        },
    ];
    for state in broken {
        // SAFETY: the op functions are valid; only the state is malformed.
        let r = unsafe {
            TensorDyn::from_client_pbo(
                state,
                client_pbo::map,
                client_pbo::unmap,
                11,
                64,
                &[64],
                DType::U8,
            )
        };
        let Err(err) = r else {
            panic!("an incomplete client state must be refused across the ABI");
        };
        assert!(
            matches!(err, Error::InvalidArgument(_)),
            "expected the same InvalidArgument the static backend returns, \
             got {err:?}"
        );
    }
}

/// Malformed geometry is refused with `Error::InvalidArgument`.
#[test]
fn dynamic_from_client_pbo_refuses_malformed_geometry_with_the_same_variant() {
    use edgefirst_tensor::EfClientState;

    let state = EfClientState {
        ctx: std::ptr::addr_of!(client_pbo::CTX) as *const std::ffi::c_void,
        retain: Some(client_pbo::retain),
        release: Some(client_pbo::release),
    };
    // An empty shape is `ndim == 0` at the ABI.
    let shape: Vec<usize> = Vec::new();
    // SAFETY: the channel is the module's own valid one; only the geometry is malformed.
    let r = unsafe {
        TensorDyn::from_client_pbo(
            state,
            client_pbo::map,
            client_pbo::unmap,
            11,
            64,
            &shape,
            DType::U8,
        )
    };
    let Err(err) = r else {
        panic!("zero ndim must be refused across the ABI");
    };
    assert!(
        matches!(err, Error::InvalidArgument(_)),
        "zero ndim: expected InvalidArgument, got {err:?}"
    );
}

/// Mock [`edgefirst_tensor::CudaGlOps`] that never calls into libcudart.
struct MockCudaGlOps;
impl edgefirst_tensor::CudaGlOps for MockCudaGlOps {
    fn map(&self, _resource: *mut std::ffi::c_void) -> Option<(*mut std::ffi::c_void, usize)> {
        None
    }
    fn unmap(&self, _resource: *mut std::ffi::c_void) {}
    fn unregister(&self, _resource: *mut std::ffi::c_void) {}
}

/// `set_cuda_handle`/`cuda` read back `None` before attachment and the registration after.
#[test]
fn dynamic_set_cuda_handle_is_read_back_through_cuda() {
    let mut tensor = bare_u8(&[4, 4]);
    assert!(
        !tensor.is_cuda_attached(),
        "a tensor with no attached CUDA handle must report none"
    );

    let handle = edgefirst_tensor::CudaHandle::new_gl(
        std::ptr::null_mut(),
        16,
        Arc::new(MockCudaGlOps) as Arc<dyn edgefirst_tensor::CudaGlOps>,
    );
    tensor.set_cuda_handle(handle);

    assert!(
        tensor.is_cuda_attached(),
        "set_cuda_handle must make is_cuda_attached() true after ef_tensor_cuda_attach"
    );
}

/// The `as_*`/`into_*` downcasts match the dtype and reject a mismatch.
#[test]
fn dynamic_typed_downcast_family_matches_dtype_and_rejects_mismatch() {
    let t: Tensor<f32> = Tensor::new(&[4], None, None).expect("f32 alloc");
    let d: TensorDyn = t.into();
    assert_eq!(d.dtype(), DType::F32);
    assert!(d.as_f32().is_some(), "f32 lens must open on an f32 tensor");
    assert!(d.as_u8().is_none(), "u8 lens must refuse an f32 tensor");
    assert!(d.as_i8().is_none(), "i8 lens must refuse an f32 tensor");
    assert!(d.as_f16().is_none(), "f16 lens must refuse an f32 tensor");
    assert!(d.as_i16().is_none(), "i16 lens must refuse an f32 tensor");

    // The consuming form hands the original `TensorDyn` back in `Err`.
    let d = match d.into_u8() {
        Ok(_) => panic!("into_u8 on an f32 tensor must not succeed"),
        Err(d) => d,
    };
    let f32_tensor = d
        .into_f32()
        .expect("into_f32 must succeed on the f32 tensor it started as");
    assert_eq!(f32_tensor.shape(), &[4]);
}

/// `Tensor::from_pbo` allocates no host memory, and a view of it reads the GL buffer.
#[test]
fn from_pbo_allocates_no_host_memory_for_the_metadata_handle() {
    let (w, h, c) = (3840usize, 2160usize, 4usize); // 4K, RGBA16F
    let bytes = w * h * c * std::mem::size_of::<half::f16>();
    assert_eq!(
        bytes, 66_355_200,
        "sanity check on the 4K RGBA16F byte count"
    );

    // The GL allocation is larger than the shape product.
    let allocated = bytes + 4096;
    let ops = MockPboOps::new(allocated);
    let pbo = edgefirst_tensor::PboTensor::<half::f16>::from_pbo(
        41,
        allocated,
        &[h, w, c],
        Some("4k_rgba16f"),
        ops,
    )
    .expect("PboTensor::from_pbo");
    let tensor = Tensor::<half::f16>::from_pbo(pbo).expect("Tensor::from_pbo");
    let mut erased: TensorDyn = tensor.into();

    assert_eq!(
        erased.memory(),
        TensorMemory::Pbo,
        "the handle IS the PBO now -- a Mem companion would report Mem once \
         memory()'s override is gone"
    );
    assert_eq!(
        erased.capacity_bytes(),
        allocated,
        "capacity comes from the GL allocation, which is larger than the shape \
         product -- not from a shape-sized host buffer"
    );
    // Stamps each byte with its offset modulo 251.
    {
        let mut mapped = erased
            .map_bytes(CpuAccess::ReadWrite)
            .expect("a PBO-backed handle maps through the client's own map callback");
        for (i, b) in mapped.as_mut_slice().iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
    }

    // A derived handle is still the PBO.
    erased
        .set_format(PixelFormat::Rgba)
        .expect("a 4K RGBA tensor takes the packed RGBA format");
    // A view at a non-zero origin.
    const VX: usize = 16;
    const VY: usize = 8;
    let pitch = erased
        .effective_row_stride()
        .expect("a formatted 4K RGBA16F tensor reports a pitch");
    let expect_offset = VY * pitch + VX * 4 * std::mem::size_of::<half::f16>();
    let view = erased
        .view(Region::new(VX, VY, 64, 64))
        .expect("view a 64x64 window of the PBO at a non-zero origin");
    assert_eq!(view.memory(), TensorMemory::Pbo, "a view of a PBO is a PBO");
    assert_eq!(
        view.pbo_id(),
        Some(41),
        "...and it names the same GL buffer, which is what a cross-package \
         re-import of a view needs"
    );
    assert_eq!(
        view.plane_offset(),
        Some(expect_offset),
        "the view's window starts {expect_offset} bytes into the GL buffer; a \
         `None`/0 here is a view that names the parent's origin"
    );
    let mapped = view
        .map_bytes(CpuAccess::Read)
        .expect("a PBO view maps through the same client callback");
    let seen = mapped.as_slice();
    assert_eq!(
        seen[0],
        (expect_offset % 251) as u8,
        "the view's first mapped byte must be the GL buffer's byte at \
         {expect_offset}, not its byte 0 -- read live, inside the guard"
    );
    // Checks the whole first row.
    for (x, &b) in seen
        .iter()
        .take(64 * 4 * std::mem::size_of::<half::f16>())
        .enumerate()
    {
        assert_eq!(
            b,
            ((expect_offset + x) % 251) as u8,
            "view byte {x} must be the GL buffer's byte at {}",
            expect_offset + x
        );
    }
}

// --- The primitives `edgefirst-python-common` needs ------------------------

/// `size` is the logical footprint and `capacity_bytes` is the allocation.
#[test]
fn capacity_bytes_reports_the_allocation_not_the_logical_size() {
    let t = Tensor::<u8>::image(
        295,
        175,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("Mem-backed NV12 allocation");
    let dyn_t: TensorDyn = t.into();

    let logical = dyn_t.size();
    let allocated = dyn_t.capacity_bytes();
    assert_eq!(
        logical,
        dyn_t.shape().iter().product::<usize>(),
        "size() is the shape's byte footprint for a u8 tensor"
    );
    assert!(
        allocated > logical,
        "an odd-width NV12 image pads its rows past the logical shape; \
         capacity_bytes must report the padded allocation ({allocated}) not \
         the logical size ({logical})"
    );
}

/// `row_stride` is the recorded pitch; `effective_row_stride` substitutes a computed one.
#[test]
fn row_stride_is_the_recorded_pitch_and_effective_row_stride_is_not() {
    let mut t: TensorDyn = bare_u8(&[48, 64, 1]).into();
    t.set_format(PixelFormat::Grey).expect("set_format");

    assert_eq!(
        t.row_stride(),
        None,
        "a freshly-allocated tight tensor has no recorded row stride"
    );
    assert_eq!(
        t.effective_row_stride(),
        Some(64),
        "...but its effective pitch is still the computed one -- if row_stride \
         forwarded to this, a tight tensor's descriptor would claim 64 is required"
    );

    t.set_row_stride(128).expect("set_row_stride");
    assert_eq!(t.row_stride(), Some(128), "the recorded pitch reads back");
    assert_eq!(
        t.effective_row_stride(),
        Some(128),
        "and so does the effective one"
    );
}

/// `compression` reports linear for an ordinary allocation.
#[test]
fn compression_reports_linear_for_an_ordinary_allocation() {
    let t: TensorDyn = bare_u8(&[4, 4]).into();
    assert_eq!(t.compression(), None);
}

/// The sync bracket succeeds on host memory and is refused on a PBO.
#[test]
fn sync_brackets_succeed_on_host_memory_and_are_refused_by_a_pbo() {
    let host: TensorDyn = bare_u8(&[4, 4]).into();
    host.sync_for_cpu(CpuAccess::ReadWrite)
        .expect("host memory is coherent");
    host.sync_for_device(CpuAccess::ReadWrite)
        .expect("and so is the release");

    let ops = MockPboOps::new(16);
    let pbo = edgefirst_tensor::PboTensor::<u8>::from_pbo(9, 16, &[4, 4, 1], None, ops)
        .expect("PboTensor::from_pbo");
    let pbo_t: TensorDyn = Tensor::<u8>::from_pbo(pbo)
        .expect("Tensor::from_pbo")
        .into();
    match pbo_t.sync_for_cpu(CpuAccess::Read) {
        Err(Error::NotImplemented(msg)) => assert!(
            msg.contains("PBO") || msg.contains("glMapBufferRange"),
            "the refusal must name the reason, got: {msg}"
        ),
        other => panic!("a PBO has no sync window independent of its map, got: {other:?}"),
    }
}

/// `CpuAccess::None` is refused by the sync bracket.
#[test]
fn sync_refuses_the_non_directional_access() {
    let t: TensorDyn = bare_u8(&[4, 4]).into();
    assert!(matches!(
        t.sync_for_cpu(CpuAccess::None),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        t.sync_for_device(CpuAccess::None),
        Err(Error::InvalidArgument(_))
    ));
}

/// A batched `[N, H, W, C]` image keeps its format and reports one element's width and height.
#[test]
fn batched_image_geometry_describes_one_element() {
    let img = Tensor::<u8>::image(
        5,
        12,
        PixelFormat::Rgba,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("Mem RGBA 5x12");
    let mut t: TensorDyn = img.into();
    t.set_logical_shape(&[4, 3, 5, 4])
        .expect("[4, 3, 5, 4] is the same bytes as [12, 5, 4]");
    assert_eq!(t.format(), Some(PixelFormat::Rgba));
    assert_eq!(t.width(), Some(5));
    assert_eq!(t.height(), Some(3));

    t.set_logical_shape(&[240]).expect("flatten to bytes");
    assert_eq!(t.width(), None, "a flat shape is not an image");
    assert_eq!(t.height(), None);
}

#[test]
fn batch_is_a_zero_copy_window_on_the_leading_dimension() {
    let parent = bare_u8(&[4, 2, 3]);
    let parent_dyn: TensorDyn = parent.into();

    let element = parent_dyn.batch(2).expect("batch(2) of a 4-element batch");
    assert_eq!(element.shape(), &[2, 3], "the leading dimension is dropped");
    // Sharing is checked by the write below.

    {
        let mut m = element
            .as_u8()
            .expect("as_u8")
            .map_with(CpuAccess::ReadWrite)
            .expect("map the batch element");
        m.as_mut_slice().fill(0x5A);
    }
    let m = parent_dyn
        .as_u8()
        .expect("as_u8")
        .map_with(CpuAccess::Read)
        .expect("map the parent");
    let bytes = m.as_slice();
    // Element 2 of a [4, 2, 3] u8 tensor occupies bytes [12, 18).
    assert!(
        bytes[12..18].iter().all(|&b| b == 0x5A),
        "the write through batch(2) must land in the parent's own buffer"
    );
    assert!(
        bytes[..12].iter().all(|&b| b == 0),
        "and must not touch the elements before it"
    );
}

/// An index past the leading dimension is refused with `BatchIndexOutOfBounds`.
#[test]
fn batch_rejects_an_index_past_the_leading_dimension() {
    let t: TensorDyn = bare_u8(&[4, 2, 3]).into();
    match t.batch(4) {
        Err(Error::BatchIndexOutOfBounds { index: 4, batch: 4 }) => {}
        other => {
            panic!("index 4 of a 4-element batch must be refused as out of bounds, got: {other:?}")
        }
    }
    match t.batch(usize::MAX) {
        Err(Error::BatchIndexOutOfBounds { index, batch: 4 }) if index == usize::MAX => {}
        other => panic!("a wild index must be refused as out of bounds, got: {other:?}"),
    }
}

/// `.batch()` on a tensor that is not batched reports the shape refusal, not an index error.
#[test]
fn batch_on_a_non_batched_tensor_names_that_and_not_the_index() {
    let img = Tensor::<u8>::image(
        640,
        480,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("Mem-backed NV12 allocation");
    let dyn_img: TensorDyn = img.into();

    match dyn_img.batch(0) {
        Err(Error::InvalidShape(msg)) => assert!(
            msg.contains("not batched"),
            "the refusal must carry the producing side's own diagnosis, got: {msg}"
        ),
        Err(Error::BatchIndexOutOfBounds { index, batch }) => panic!(
            "index {index} IS within a leading dimension of {batch} -- reporting it as \
             out of bounds is a false diagnosis, not merely an unhelpful one"
        ),
        other => panic!("expected the not-batched shape refusal, got: {other:?}"),
    }
}

/// `ef_tensor_batch`'s two refusals arrive as two different typed errors.
#[test]
fn the_two_batch_refusals_arrive_as_two_different_typed_errors() {
    // Out of range on a batched tensor.
    let batched: TensorDyn = bare_u8(&[4, 2, 3]).into();
    match batched.batch(9) {
        Err(Error::BatchIndexOutOfBounds { index: 9, batch: 4 }) => {}
        other => panic!("an out-of-range index must keep its own typed variant, got: {other:?}"),
    }

    // Not batched at all.
    let img = Tensor::<u8>::image(
        640,
        480,
        PixelFormat::Nv12,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("Mem-backed NV12 allocation");
    let dyn_img: TensorDyn = img.into();
    match dyn_img.batch(0) {
        Err(Error::InvalidShape(msg)) => assert!(
            msg.contains("not batched"),
            "the refusal must carry the producing side's own diagnosis, got: {msg}"
        ),
        other => panic!("expected the not-batched shape refusal, got: {other:?}"),
    }
}

/// `view_region` reports an out-of-bounds crop as such, and other refusals as other errors.
#[test]
fn view_region_distinguishes_out_of_bounds_from_every_other_refusal() {
    let img = Tensor::<u8>::image(
        64,
        48,
        PixelFormat::Rgb,
        Some(TensorMemory::Mem),
        CpuAccess::ReadWrite,
    )
    .expect("Mem-backed RGB allocation");
    let dyn_img: TensorDyn = img.into();
    match dyn_img.view(Region::new(0, 0, 4096, 4096)) {
        Err(Error::RegionOutOfBounds { bounds, .. }) => assert_eq!(
            bounds,
            (64, 48),
            "the reported frame must be the tensor's real one"
        ),
        other => panic!("a region past the frame is out of bounds, got: {other:?}"),
    }

    // A formatless tensor has no frame to crop.
    let bare: TensorDyn = bare_u8(&[8, 8]).into();
    if let Err(Error::RegionOutOfBounds { bounds, .. }) = bare.view(Region::new(0, 0, 2, 2)) {
        panic!(
            "a tensor with no pixel format has no frame; reporting bounds {bounds:?} \
             invents one"
        );
    }
}

/// `dmabuf_clone` refuses a non-DMA tensor and returns an independently owned fd for a DMA one.
#[cfg(target_os = "linux")]
#[test]
fn dmabuf_clone_refuses_host_memory_and_dups_a_real_dma_fd() {
    let host: TensorDyn = bare_u8(&[4, 4]).into();
    match host.dmabuf_clone() {
        Err(Error::NotImplemented(msg)) => assert!(
            msg.contains("dmabuf_clone requires DMA-backed tensor") && msg.contains("Mem"),
            "the refusal must name the operation and the actual backing, got: {msg}"
        ),
        other => panic!("host memory has no dma-buf fd, got: {other:?}"),
    }

    if !dma_or_skip("dmabuf_clone_refuses_host_memory_and_dups_a_real_dma_fd") {
        return;
    }
    let dma = Tensor::<u8>::new(&[64, 64], Some(TensorMemory::DmaBuf), None)
        .expect("a usable DMA heap allocates");
    let dma_dyn: TensorDyn = dma.into();
    let borrowed = { dma_dyn.plane0().expect("plane 0").handle };
    let cloned = dma_dyn
        .dmabuf_clone()
        .expect("a DMA-backed tensor clones its fd");
    use std::os::fd::AsRawFd;
    assert_ne!(
        cloned.as_raw_fd() as i64,
        borrowed,
        "dmabuf_clone must dup, so the caller can close its copy independently"
    );
}

/// `cuda_map` returns `None` with no handle attached and a guard once one is.
#[test]
fn cuda_map_is_none_without_a_handle_and_maps_with_one() {
    /// A `CudaGlOps` whose `map` hands back a real address.
    struct MappingCudaGlOps(Mutex<Vec<u8>>);
    impl edgefirst_tensor::CudaGlOps for MappingCudaGlOps {
        fn map(&self, _resource: *mut std::ffi::c_void) -> Option<(*mut std::ffi::c_void, usize)> {
            let mut g = self.0.lock().expect("lock");
            let len = g.len();
            Some((g.as_mut_ptr() as *mut std::ffi::c_void, len))
        }
        fn unmap(&self, _resource: *mut std::ffi::c_void) {}
        fn unregister(&self, _resource: *mut std::ffi::c_void) {}
    }

    let mut t = bare_u8(&[4, 4]);
    assert!(
        t.cuda_map().is_none(),
        "a tensor with no CUDA handle must fast-fail to None"
    );

    t.set_cuda_handle(edgefirst_tensor::CudaHandle::new_gl(
        std::ptr::null_mut(),
        16,
        Arc::new(MappingCudaGlOps(Mutex::new(vec![0u8; 16])))
            as Arc<dyn edgefirst_tensor::CudaGlOps>,
    ));
    let dyn_t: TensorDyn = t.into();
    let raw = dyn_t.into_raw();
    let wrapped = unsafe { TensorDyn::from_raw(raw) };
    assert!(
        wrapped.is_cuda_attached(),
        "CUDA attach must survive from_handle/from_raw"
    );
    let map = wrapped
        .cuda_map()
        .expect("an attached CUDA handle must produce a map guard after wrap");
    assert_eq!(map.len(), 16);
    assert!(!map.device_ptr().is_null());
}

/// A PBO whose GL buffer is larger than its shape product reports the buffer's size.
#[test]
fn an_oversized_pbos_capacity_is_the_gl_buffers_not_the_metadata_handles() {
    // 4x4 u8 = 16 logical bytes; the GL buffer is padded to 24.
    const LOGICAL: usize = 16;
    const ALLOCATED: usize = 24;
    let ops = MockPboOps::new(ALLOCATED);
    let pbo = edgefirst_tensor::PboTensor::<u8>::from_pbo(21, ALLOCATED, &[4, 4, 1], None, ops)
        .expect("PboTensor::from_pbo permits size > shape.product()");
    let t: TensorDyn = Tensor::<u8>::from_pbo(pbo)
        .expect("Tensor::from_pbo")
        .into();

    assert_eq!(
        t.size(),
        LOGICAL,
        "the logical footprint is the shape product"
    );
    assert_eq!(
        t.capacity_bytes(),
        ALLOCATED,
        "capacity_bytes must report the GL buffer's real size, not the shape \
         product"
    );
    assert_eq!(
        t.descriptor().capacity,
        ALLOCATED as u64,
        "...and the descriptor must carry it across the package boundary, or a \
         consumer re-importing this PBO maps only part of the buffer"
    );
}

/// The PBO accessors resolve through the dtype downcast, and report `None` without a PBO.
#[test]
fn the_pbo_accessors_resolve_through_the_dtype_downcast() {
    let plain: TensorDyn = bare_u8(&[4, 4]).into();
    assert_eq!(plain.pbo_id(), None);
    assert!(plain.pbo_vtable_ptr().is_none());
    assert!(plain.pbo_keepalive().is_none());

    let ops = MockPboOps::new(16);
    let pbo = edgefirst_tensor::PboTensor::<u8>::from_pbo(11, 16, &[4, 4, 1], None, ops)
        .expect("PboTensor::from_pbo");
    let pbo_t: TensorDyn = Tensor::<u8>::from_pbo(pbo)
        .expect("Tensor::from_pbo")
        .into();
    assert_eq!(
        pbo_t.pbo_id(),
        Some(11),
        "the downcast must find the real GL buffer id, not a default"
    );
    assert!(
        pbo_t.pbo_vtable_ptr().is_some_and(|p| !p.is_null()),
        "a PBO-backed tensor must build a real cross-cdylib ops vtable"
    );
    assert!(
        pbo_t.pbo_keepalive().is_some(),
        "and a keepalive holding the PboHandle that vtable addresses"
    );

    // A non-u8 PBO.
    let f32_ops = MockPboOps::new(64);
    let f32_pbo = edgefirst_tensor::PboTensor::<f32>::from_pbo(12, 64, &[4, 4, 1], None, f32_ops)
        .expect("PboTensor::<f32>::from_pbo");
    let f32_t: TensorDyn = Tensor::<f32>::from_pbo(f32_pbo)
        .expect("Tensor::from_pbo")
        .into();
    assert_eq!(
        f32_t.pbo_id(),
        Some(12),
        "the dtype downcast must handle every element type, not just u8"
    );
}

/// A `Tensor<i8>` transmuted from a `Tensor<u8>` reports `I8` once erased to `TensorDyn`.
#[test]
fn a_retagged_typed_lens_erases_to_its_own_dtype_not_the_handles() {
    let t_u8 = bare_u8(&[4, 4]);
    assert_eq!(TensorTrait::memory(&t_u8), TensorMemory::Mem);
    // SAFETY: `Tensor<u8>` and `Tensor<i8>` are layout-identical on both backends.
    let t_i8: Tensor<i8> = unsafe { std::mem::transmute(t_u8) };
    let erased: TensorDyn = t_i8.into();
    assert_eq!(
        erased.dtype(),
        DType::I8,
        "erasing a Tensor<i8> must yield an I8 tensor whatever dtype the \
         underlying handle was minted with -- a wrong dtype propagates \
         silently into quantization and inference rather than stopping \
         anything"
    );
}

// --- Behaviour of methods both backends declare ----------------------------

/// `reshape` changes the shape, and the cached `shape()` follows.
#[test]
fn reshape_changes_the_shape_and_the_cache_follows() {
    let mut t: TensorDyn = bare_u8(&[4, 6]).into();
    t.reshape(&[2, 12]).expect("equal element count");
    assert_eq!(
        t.shape(),
        &[2, 12],
        "shape() serves a cache; it must be refreshed"
    );
    assert_eq!(t.size(), 24);

    match t.reshape(&[5, 5]) {
        Err(Error::ShapeMismatch(_)) => {}
        other => panic!("a different element count must be refused: {other:?}"),
    }
    assert_eq!(
        t.shape(),
        &[2, 12],
        "a refused reshape must leave the shape alone"
    );
}

/// `clone_fd` returns a working fd for an SHM-backed tensor.
#[cfg(unix)]
#[test]
fn clone_fd_works_for_shm_not_only_dma() {
    const SHAPE: [usize; 2] = [64, 64];
    let expect = if edgefirst_tensor::is_shm_available() {
        outcome::Expect::Succeeds
    } else {
        outcome::Expect::MayFail("no usable /dev/shm on this host")
    };
    let Some(shm) = outcome::allocated(
        Tensor::<u8>::new(&SHAPE, Some(TensorMemory::Shm), None),
        "SHM allocation",
        expect,
    ) else {
        return;
    };

    // A marker written through the tensor.
    {
        let mut m = shm
            .map_with(CpuAccess::ReadWrite)
            .expect("map the SHM tensor");
        m.as_mut_slice().fill(0x9C);
    }
    let dyn_shm: TensorDyn = shm.into();
    assert_eq!(
        TensorTrait::memory(dyn_shm.as_u8().unwrap()),
        TensorMemory::Shm
    );

    let cloned = dyn_shm
        .clone_fd()
        .expect("an SHM-backed tensor has a real fd and must clone it");

    // Re-imports the cloned fd and reads the marker back.
    let reimported =
        Tensor::<u8>::from_fd(cloned, &SHAPE, None).expect("the cloned fd must import as a tensor");
    let m = reimported
        .map_with(CpuAccess::Read)
        .expect("map the re-imported tensor");
    assert!(
        m.as_slice().iter().all(|&b| b == 0x9C),
        "the cloned fd must address the SAME buffer -- a dup of an unrelated \
         descriptor would also be >= 0 and would read anything but this"
    );
}

/// Mapping a PBO-backed tensor through the type-erased handle reads the GL buffer.
#[test]
fn mapping_a_pbo_backed_tensor_reads_the_gl_buffer() {
    let ops = MockPboOps::new(16);
    let pbo = edgefirst_tensor::PboTensor::<u8>::from_pbo(31, 16, &[4, 4, 1], None, ops)
        .expect("PboTensor::from_pbo");
    let tensor = Tensor::<u8>::from_pbo(pbo).expect("Tensor::from_pbo");
    // Writes a marker through the typed tensor's map.
    {
        let mut m = tensor.map().expect("typed PBO map");
        m.as_mut_slice().fill(0xC3);
    }
    let dyn_t: TensorDyn = tensor.into();

    // Reads it back through the type-erased handle.
    let bytes = dyn_t
        .map_bytes(CpuAccess::Read)
        .expect("a PBO-backed tensor must be mappable through TensorDyn");
    assert_eq!(bytes.as_slice().len(), 16);
    assert!(
        bytes.as_slice().iter().all(|&b| b == 0xC3),
        "the erased map must read the GL buffer, not the companion host allocation"
    );
}

/// A retagged PBO tensor still resolves its buffer id, vtable, keepalive and map.
#[test]
fn a_retagged_pbo_still_resolves_its_buffer_id_vtable_and_map() {
    let ops = MockPboOps::new(16);
    let pbo = edgefirst_tensor::PboTensor::<u8>::from_pbo(41, 16, &[4, 4, 1], None, ops)
        .expect("PboTensor::from_pbo -- allocated as u8, as the GL backend does");
    let as_u8 = Tensor::<u8>::from_pbo(pbo).expect("Tensor::from_pbo");
    {
        let mut m = as_u8.map().expect("typed PBO map before retagging");
        m.as_mut_slice().fill(0x7E);
    }

    // SAFETY: `Tensor<u8>` and `Tensor<i8>` are layout-identical.
    let as_i8: Tensor<i8> = unsafe { std::mem::transmute(as_u8) };
    assert_eq!(
        as_i8.pbo_id(),
        Some(41),
        "pbo_id must find the stored PboTensor whatever element type it was created with"
    );
    assert_eq!(
        as_i8.pbo_is_mapped(),
        Some(false),
        "...and so must pbo_is_mapped, which gates every convert into this tensor"
    );

    let erased: TensorDyn = as_i8.into();
    assert_eq!(erased.dtype(), DType::I8, "the handle's dtype is retagged");
    assert_eq!(
        erased.pbo_id(),
        Some(41),
        "...and the PBO accessors must still resolve -- a None here makes the \
         descriptor carry no buffer id, which a cross-package re-import rejects"
    );
    assert!(erased.pbo_vtable_ptr().is_some_and(|p| !p.is_null()));
    assert!(erased.pbo_keepalive().is_some());
    assert_eq!(
        erased.descriptor().handle,
        41,
        "the descriptor must carry the real GL buffer id"
    );

    let bytes = erased
        .map_bytes(CpuAccess::Read)
        .expect("a retagged PBO tensor must still map through the erased handle");
    assert!(
        bytes.as_slice().iter().all(|&b| b == 0x7E),
        "and must still read the GL buffer"
    );
}

/// `memory()` and `dmabuf()` answer for a DMA-backed tensor; a Mem-backed one answers no.
#[cfg(target_os = "linux")]
#[test]
fn a_dma_tensor_answers_both_questions_the_gpu_import_sites_ask() {
    if !dma_or_skip("a_dma_tensor_answers_both_questions_the_gpu_import_sites_ask") {
        return;
    }
    let dma = Tensor::<u8>::new(&[64, 64], Some(TensorMemory::DmaBuf), None)
        .expect("a usable DMA heap allocates");

    // The capability probe.
    assert_eq!(
        TensorTrait::memory(&dma),
        TensorMemory::DmaBuf,
        "a DMA-backed tensor must report DmaBuf -- the probe that decides whether the \
         G2D hardware path is even attempted"
    );

    // The fd accessor.
    use std::os::fd::AsRawFd;
    let fd = dma
        .dmabuf()
        .expect("a DMA-backed tensor must yield its fd -- this is what the EGLImage import needs");
    assert!(fd.as_raw_fd() >= 0, "and it must be a real descriptor");

    // A Mem-backed tensor answers no to both.
    let host = bare_u8(&[64, 64]);
    assert_eq!(TensorTrait::memory(&host), TensorMemory::Mem);
    assert!(host.dmabuf().is_err());
}

/// The three geometry mutators leave a PBO-backed tensor's shape, id and map coherent.
#[test]
fn geometry_mutators_keep_a_pbo_backed_tensor_coherent() {
    const SENTINEL: u8 = 0x5C;

    fn pbo_tensor(bytes: usize, shape: &[usize]) -> TensorDyn {
        let ops = MockPboOps::new(bytes);
        let pbo = edgefirst_tensor::PboTensor::<u8>::from_pbo(51, bytes, shape, None, ops)
            .expect("PboTensor::from_pbo");
        let t: TensorDyn = Tensor::<u8>::from_pbo(pbo)
            .expect("Tensor::from_pbo")
            .into();
        // Stamps the GL buffer with a sentinel.
        let mut map = t.map_bytes(CpuAccess::ReadWrite).expect("stamp the PBO");
        map.as_mut_slice().fill(SENTINEL);
        drop(map);
        t
    }

    /// Asserts the live mapping spans exactly `expect` bytes of the PBO's own buffer.
    fn assert_maps_pbo_bytes(t: &TensorDyn, expect: usize, what: &str) {
        let map = t
            .map_bytes(CpuAccess::Read)
            .expect("a PBO-backed tensor must be mappable");
        assert_eq!(
            map.as_slice().len(),
            expect,
            "{what}: the map's extent must follow the new shape"
        );
        assert!(
            map.as_slice().iter().all(|&b| b == SENTINEL),
            "{what}: the map must read the GL buffer, not a zeroed placeholder"
        );
    }

    // --- reshape: same element count, different rank -----------------
    let mut t = pbo_tensor(16, &[4, 4, 1]);
    t.reshape(&[2, 8]).expect("equal element count");
    assert_eq!(t.shape(), &[2, 8], "the handle's shape follows");
    assert_eq!(t.pbo_id(), Some(51), "reshape keeps the same GL buffer");
    assert_maps_pbo_bytes(&t, 16, "reshape");

    // --- set_logical_shape: fewer elements, still fits ---------------
    let mut t = pbo_tensor(16, &[4, 4, 1]);
    t.set_logical_shape(&[2, 4])
        .expect("8 bytes fits a 16-byte GL buffer");
    assert_eq!(t.shape(), &[2, 4]);
    assert_eq!(
        t.pbo_id(),
        Some(51),
        "set_logical_shape keeps the same GL buffer"
    );
    assert_maps_pbo_bytes(&t, 8, "set_logical_shape");

    // --- reshape after a view: refused, window unchanged --------------
    let mut t = {
        let ops = MockPboOps::new(64);
        let pbo = edgefirst_tensor::PboTensor::<u8>::from_pbo(52, 64, &[8, 8, 1], None, ops)
            .expect("PboTensor::from_pbo");
        let mut t: TensorDyn = Tensor::<u8>::from_pbo(pbo)
            .expect("Tensor::from_pbo")
            .into();
        // Stamps each byte with its offset modulo 251.
        {
            let mut map = t.map_bytes(CpuAccess::ReadWrite).expect("stamp the PBO");
            for (i, b) in map.as_mut_slice().iter_mut().enumerate() {
                *b = (i % 251) as u8;
            }
        }
        t.set_format(PixelFormat::Grey).expect("set_format");
        t.view(Region::new(2, 3, 4, 4)).expect("view at (2, 3)")
    };
    const VIEW_OFFSET: usize = 3 * 8 + 2;
    assert_eq!(
        t.plane_offset(),
        Some(VIEW_OFFSET),
        "precondition: the view starts {VIEW_OFFSET} bytes into the GL buffer"
    );
    assert!(
        matches!(t.reshape(&[4, 4]), Err(Error::ShapeMismatch(_))),
        "a PBO sub-view covers less than its parent's allocation, which \
         `PboTensor::reshape`'s equal-size rule refuses"
    );
    assert_eq!(
        t.plane_offset(),
        Some(VIEW_OFFSET),
        "the refused reshape must leave the window exactly as it was"
    );
    {
        let map = t
            .map_bytes(CpuAccess::Read)
            .expect("the view still maps after the refusal");
        assert_eq!(
            map.as_slice()[0],
            (VIEW_OFFSET % 251) as u8,
            "...and the storage agrees: the map still starts at the view's \
             own byte {VIEW_OFFSET}, read live inside the guard"
        );
    }

    // --- configure_image: the pool-reuse path ------------------------
    let mut t = pbo_tensor(64, &[8, 8, 1]);
    t.configure_image(4, 4, PixelFormat::Grey)
        .expect("a smaller image fits the 64-byte buffer");
    assert_eq!(t.shape().iter().product::<usize>(), 16);
    assert_eq!(
        t.pbo_id(),
        Some(51),
        "configure_image is the decode-into-a-pool path; it must reconfigure the \
         SAME GL buffer, or the next import reads a different one"
    );
    assert_maps_pbo_bytes(&t, 16, "configure_image");
}

/// A PBO view reads its own region, and the GL buffer outlives the parent tensor.
#[test]
fn a_pbo_view_reads_its_own_region_and_outlives_its_parent() {
    const W: usize = 32;
    const H: usize = 32;
    const SIDE: usize = 8;
    const X0: usize = 8;
    const Y0: usize = 8;

    let ops = MockPboOps::new(W * H);
    let pbo = edgefirst_tensor::PboTensor::<u8>::from_pbo(
        51,
        W * H,
        &[H, W, 1],
        None,
        ops.clone() as std::sync::Arc<dyn edgefirst_tensor::PboOps>,
    )
    .expect("PboTensor::from_pbo");
    let mut parent_dyn: TensorDyn = Tensor::<u8>::from_pbo(pbo)
        .expect("Tensor::from_pbo")
        .into();
    parent_dyn
        .set_format(PixelFormat::Grey)
        .expect("set_format");
    {
        let mut m = parent_dyn
            .map_bytes(CpuAccess::ReadWrite)
            .expect("map parent");
        let s = m.as_mut_slice();
        for y in 0..H {
            for x in 0..W {
                s[y * W + x] = ((y * 4 + x) & 0xff) as u8;
            }
        }
    }

    let view = parent_dyn
        .view(Region::new(X0, Y0, SIDE, SIDE))
        .expect("view a PBO-backed tensor");
    assert_eq!(
        view.memory(),
        TensorMemory::Pbo,
        "a view of a PBO must stay PBO-backed"
    );
    assert_eq!(
        view.pbo_id(),
        Some(51),
        "the view shares the parent's GL buffer"
    );
    // The view shares its parent's vtable address.
    assert_eq!(
        view.pbo_vtable_ptr(),
        parent_dyn.pbo_vtable_ptr(),
        "a view reports its parent's PBO vtable"
    );

    // The parent drops first.
    drop(parent_dyn);
    assert_eq!(
        ops.delete_count(),
        0,
        "the GL buffer must survive its parent tensor while a view holds it"
    );

    // The view's map is indexed by the parent pitch.
    let stride = view
        .effective_row_stride()
        .expect("a formatted view reports a pitch");
    assert_eq!(
        stride, W,
        "a multi-row view's rows are spaced by the PARENT pitch, not its own \
         tight row"
    );
    let m = view.map_bytes(CpuAccess::Read).expect("map the view");
    let s = m.as_slice();
    for y in 0..SIDE {
        for x in 0..SIDE {
            assert_eq!(
                s[y * stride + x],
                (((y + Y0) * 4 + (x + X0)) & 0xff) as u8,
                "view pixel ({x}, {y}) must be parent ({}, {})",
                x + X0,
                y + Y0
            );
        }
    }
    drop(m);
    drop(view);
    assert_eq!(
        ops.delete_count(),
        1,
        "the GL buffer is deleted exactly once, when the last holder goes"
    );
}

#[test]
fn mapping_exclusion_runs_the_closure_in_the_library_and_returns_its_value() {
    let mut calls = 0;
    let out = edgefirst_tensor::pin::with_cpu_mappings_excluded(|| {
        calls += 1;
        42
    });
    assert_eq!((calls, out), (1, 42));
}

#[test]
fn mapping_exclusion_propagates_a_panic_without_crossing_the_c_frame() {
    let caught = std::panic::catch_unwind(|| {
        edgefirst_tensor::pin::with_cpu_mappings_excluded(|| panic!("inside the exclusion"))
    });
    let payload = caught.expect_err("the closure's panic must reach the caller");
    assert_eq!(
        payload.downcast_ref::<&str>(),
        Some(&"inside the exclusion")
    );
    // The lock was released: a second exclusion runs.
    assert_eq!(edgefirst_tensor::pin::with_cpu_mappings_excluded(|| 7), 7);
}
