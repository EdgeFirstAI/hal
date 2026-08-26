// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Proof that the five primitive families task 15 added
//! (`.superpowers/sdd/2026-08-24-single-tensor-home/task-15-brief.md`) are
//! real, working `ef_tensor_*` calls reached through the `dynamic` backend's
//! own Rust wrappers (`crates/tensor/src/tensor_dyn/dynamic_backend.rs`,
//! `dynamic_tensor.rs`) -- not just that the C primitives work in isolation
//! (`crates/tensor-capi/src/{mutate,quant,image,desc}.rs` already prove
//! that).
//!
//! Every test here would **fail** against the stub each primitive replaced:
//! family 1/3/4/5 stubs all returned `Err(Error::NotImplemented(..))` (or,
//! for `TensorDyn::set_plane_offset`, panicked via `unimplemented!()`), and
//! family 2's `quantization()` stub panicked unconditionally. A test that
//! only checks `.is_ok()`/a successful read is therefore sufficient proof:
//! it cannot pass against any of those stubs.
//!
//! # Running this file
//!
//! This crate's `dynamic` feature deliberately links no `ef_tensor_*`
//! implementation of its own (`edgefirst-tensor-ffi`'s declarations carry no
//! `#[link]` attribute -- see that crate's module docs); linking is
//! normally the *consumer*'s decision (each `-capi` leaf's own build). This
//! crate has no other consumer to make that decision on its own tests'
//! behalf, so the opt-in `dynamic-test-link` feature (never enabled by a
//! production consumer -- see its doc comment in `Cargo.toml`) makes
//! `build.rs` link this test binary against `libedgefirst_tensor.so`
//! directly.
//!
//! ```sh
//! export TMPDIR=/home/sebastien/.cache/hal-tmp   # avoid a near-full /tmp
//! # 1. Build the producer first (the real implementation lives here):
//! cargo build --manifest-path crates/tensor-capi/Cargo.toml --target-dir target
//! # 2. Run just this integration test target (not the whole crate's test
//! #    suite -- the other files under tests/ are static-only and do not
//! #    compile under `dynamic`, a pre-existing gap this task does not
//! #    close; see the task report):
//! cargo test -p edgefirst-tensor --no-default-features \
//!     --features dynamic,dynamic-test-link,ndarray --test dynamic_primitives
//! ```

#![cfg(feature = "dynamic")]

use edgefirst_tensor::{
    CpuAccess, DType, Error, PixelFormat, Quantization, Region, Tensor, TensorDyn, TensorMapTrait,
    TensorMemory, TensorTrait,
};

/// Allocate a bare (formatless) `TensorDyn` of the given shape/dtype code,
/// the same way `ef_tensor_new` does -- the smallest live handle every test
/// below builds on before attaching format/quantization metadata.
fn bare_u8(shape: &[usize]) -> Tensor<u8> {
    Tensor::<u8>::new(shape, None, None).expect("bare tensor allocation")
}

// --- Family 1: set_format / set_row_stride / set_plane_offset -------------
//
// Real caller: `edgefirst-image`'s `import_image` (single-plane path), which
// calls these on the type-erased `TensorDyn` it gets back from
// `TensorDyn::from_fd` -- not on a typed `Tensor<T>` (which does not even
// have a `set_format` under `dynamic`; only `TensorDyn` does, matching this
// real call site). The old stubs returned `Err(NotImplemented)` for
// `set_format`/`set_row_stride` and panicked for `set_plane_offset`.

#[test]
fn family1_set_format_set_row_stride_and_set_plane_offset_take_effect() {
    // NV12 combined-plane shape: [H + H/2, W].
    let (w, h) = (64usize, 48usize);
    let mut t: TensorDyn = bare_u8(&[h + h / 2, w]).into();

    // Before `set_format`, there is no pixel format at all.
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

    // `set_plane_offset` returns `()` on `static`'s own signature, so the
    // only way to prove it took effect (rather than silently no-op'ing,
    // which is exactly what the pre-primitive stub did before it was
    // converted to `unimplemented!()`) is reading it back through
    // `plane_offset()` -- itself a small fix found while proving this test
    // (see the task report): the pre-existing reader derived its answer
    // from `ef_tensor_plane_at`'s plane-0 offset, which is always 0 for
    // plane 0 by construction (a different quantity than the DMA-BUF-level
    // offset this setter writes), so it silently reported the wrong value.
    assert_eq!(t.plane_offset(), None);
    t.set_plane_offset(4096);
    assert_eq!(
        t.plane_offset(),
        Some(4096),
        "set_plane_offset must be visible back through the handle, not a silent no-op"
    );
}

// --- Family 2: quantization -------------------------------------------
//
// Real caller: `edgefirst-decoder`'s int8 per-scale pipeline. The old stub
// panicked unconditionally on `quantization()`. This test's whole point,
// per the task brief: read back quantization attached by a **different
// producer** via the raw `ef_tensor_builder_quantization` setter -- a
// round-trip through a tensor this backend allocated *and quantized itself*
// would also pass against a stub that just echoed back its own in-memory
// state, so it proves nothing about the real cross-ABI primitive.

#[test]
fn family2_quantization_reads_back_metadata_set_by_another_producer() {
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
                                                                              // Per-channel asymmetric, axis 1 (matches the 4-wide dimension) --
                                                                              // set through the pre-alloc *builder* path, the "another producer"
                                                                              // the task brief names, never through this crate's own
                                                                              // `Tensor::set_quantization`.
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
                  be readable back through Tensor::quantization -- the old stub panicked here \
                  unconditionally",
    );
    assert_eq!(q.axis(), Some(1));
    assert_eq!(q.scale(), &[0.1, 0.2, 0.3, 0.4]);
    assert_eq!(q.zero_point(), Some(&[1, -2, 3, -4][..]));

    // Round-trip through this crate's own `set_quantization`/`clear_quantization`
    // too, on a fresh, self-allocated tensor -- proving those primitives
    // independently of the cross-producer read above.
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

// --- Family 3: configure_image -----------------------------------------
//
// Real caller: `edgefirst-codec`'s JPEG decode-into-pool path. The old stub
// returned `Err(NotImplemented)`.

#[test]
fn family3_configure_image_reconfigures_an_oversized_pool_tensor() {
    // Oversized NV12 "pool" buffer (128x128), reconfigured down to a
    // smaller Grey (mono8) frame -- exactly the pool-reuse shape a JPEG
    // decoder targets.
    let mut pool = bare_u8(&[128 + 128 / 2, 128]);
    pool.configure_image(64, 64, PixelFormat::Grey)
        .expect("configure_image must succeed against a real ef_tensor_configure_image primitive");
    assert_eq!(pool.format(), Some(PixelFormat::Grey));
    assert_eq!(pool.width(), Some(64));
    assert_eq!(pool.height(), Some(64));
}

// --- Family 4: image / image_desc / image_with_stride / view --------------
//
// Real callers: `edgefirst-codec`'s V4L2 JPEG decoder (`Tensor::image`) and
// `edgefirst-image`'s tiled multi-slot convert path (`TensorDyn::view`). The
// old stubs all returned `Err(NotImplemented)`.

#[test]
fn family4_image_allocates_a_real_tensor() {
    let img = Tensor::<u8>::image(64, 48, PixelFormat::Grey, None, CpuAccess::None)
        .expect("image must succeed against a real ef_tensor_image_alloc primitive");
    assert_eq!(img.format(), Some(PixelFormat::Grey));
    assert_eq!(img.width(), Some(64));
    assert_eq!(img.height(), Some(48));
}

#[test]
fn family4_image_desc_allocates_from_a_declarative_request() {
    let desc = edgefirst_tensor::ImageDesc::new(64, 48, PixelFormat::Rgb, DType::U8);
    let img = Tensor::<u8>::image_desc(&desc)
        .expect("image_desc must succeed against a real ef_tensor_image_desc_alloc primitive");
    assert_eq!(img.format(), Some(PixelFormat::Rgb));
    assert_eq!(img.width(), Some(64));
    assert_eq!(img.height(), Some(48));
}

#[test]
fn family4_view_region_is_a_real_zero_copy_window_and_is_bounds_checked() {
    // A byte-level round-trip, not an identity-equality check -- this test
    // predates task 17's `BufferIdentity` fix (`TensorDyn::derive_identity`)
    // and used a `Mem`-backed parent specifically because, at the time it
    // was written, `dynamic`'s identity was derived locally per Rust-side
    // `TensorDyn` value from the handle's own address, so `parent.aliases(&view)`
    // could not observe DMA-BUF sharing at all -- see task 15's report and
    // task 17's report for the fix and
    // `two_distinct_views_of_one_dma_parent_share_one_identity_and_do_not_collide_with_an_unrelated_buffer`
    // below for the now-passing identity-equality proof on a DMA-backed
    // parent. Kept as a `Mem`-backed byte-level check regardless: it proves
    // `ef_tensor_view_region` shares the parent's real storage independent
    // of identity or memory kind, and needs no `is_dma_available()` guard to
    // run.
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

    // Top-left corner, so the view's first byte and the parent's first byte
    // are the exact same address -- no row-stride arithmetic needed to
    // state the proof precisely.
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

// --- Family 5: from_planes -----------------------------------------------
//
// Real caller: `edgefirst-image`'s two-fd import path. The old stub
// returned `Err(NotImplemented)` and consumed nothing.

#[test]
fn family5_from_planes_combines_two_independently_allocated_tensors() {
    // NV12: chroma height is luma height / 2, same width.
    let (w, h) = (64usize, 48usize);
    let luma = bare_u8(&[h, w]);
    let chroma = bare_u8(&[h / 2, w]);

    let combined = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)
        .expect("from_planes must succeed against a real ef_tensor_from_planes primitive");
    assert_eq!(combined.format(), Some(PixelFormat::Nv12));
}

#[test]
fn family5_from_planes_is_genuinely_multiplane_and_chroma_is_independently_writable() {
    // Task 17: before this fix, `is_multiplane()`/`chroma()`/`chroma_mut()`
    // unconditionally reported "no chroma", even for the tensor
    // `from_planes` had just built -- a real caller (`edgefirst-image`'s
    // `import_image`) reads chroma back through exactly this path to apply
    // its own stride/offset, and CPU convert reads chroma bytes through it.
    // Not DMA-backed (see `bare_u8`): `TensorDyn::multiplane_chroma`
    // degrades gracefully to `None` for a non-fd-backed chroma plane (see
    // its own doc comment) rather than failing the whole combine, so this
    // test intentionally does NOT assert `is_multiplane()` here -- that
    // needs an fd-backed pair, covered by
    // `dma_multiplane_from_planes_makes_is_multiplane_and_chroma_honest`
    // below, gated on DMA availability. This test instead proves the
    // *degraded* path stays exactly what it was before task 17 (still
    // correct, just not upgraded) so both branches of `from_planes`'s
    // shadow logic are exercised.
    let (w, h) = (64usize, 48usize);
    let luma = bare_u8(&[h, w]);
    let chroma = bare_u8(&[h / 2, w]);
    let combined = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)
        .expect("from_planes must succeed");
    assert!(
        !combined.is_multiplane(),
        "a Mem-backed chroma plane has no fd to shadow -- must degrade to the pre-task-17 \
         answer, not panic or silently misreport a `Some`"
    );
}

#[test]
fn dma_multiplane_from_planes_makes_is_multiplane_and_chroma_honest() {
    if !edgefirst_tensor::is_dma_available() {
        eprintln!(
            "SKIPPED: dma_multiplane_from_planes_makes_is_multiplane_and_chroma_honest - \
                    DMA not available"
        );
        return;
    }
    let (w, h) = (64usize, 48usize);
    let luma =
        Tensor::<u8>::new(&[h, w], Some(TensorMemory::DmaBuf), None).expect("dma luma allocation");
    let chroma = Tensor::<u8>::new(&[h / 2, w], Some(TensorMemory::DmaBuf), None)
        .expect("dma chroma allocation");

    let mut combined = Tensor::<u8>::from_planes(luma, chroma, PixelFormat::Nv12)
        .expect("from_planes must succeed against a real ef_tensor_from_planes primitive");

    assert!(
        combined.is_multiplane(),
        "a DMA-backed from_planes tensor is genuinely two allocations -- is_multiplane() must \
         say so, not silently take the combined-plane path a real caller would misread chroma \
         bytes through"
    );

    // The caller-facing scenario `import_image`'s multiplane path exercises:
    // set the chroma sub-tensor's own stride via `chroma_mut()` (task 17's
    // `set_row_stride_unchecked` primitive) and read it back through
    // `chroma()`, proving both the shadow and the new primitive round-trip
    // correctly together, not just in isolation.
    {
        let chroma_ref = combined
            .chroma_mut()
            .expect("chroma_mut() must return the shadow this from_planes call built");
        chroma_ref.set_row_stride_unchecked(w + 16);
    }
    assert_eq!(
        combined.chroma().and_then(|c| c.effective_row_stride()),
        Some(w + 16),
        "the stride set through chroma_mut() must read back through chroma() -- this is the \
         exact round trip edgefirst-image's import_image relies on"
    );
}

// --- Task 17: BufferIdentity for DMA-backed handles ------------------------
//
// Before this fix, every `dynamic` handle's identity was derived from its
// own process address (`IdentityKind::HostPtr`), regardless of backing --
// including DMA-BUF tensors, which `edgefirst-image`'s GL import cache
// (`gl/cache.rs`) keys a cached EGLImage on, and is documented as safe to
// outlive the tensor that produced it specifically because a system-level
// identity (an inode) cannot be recycled onto a different buffer while any
// reference is alive. A process address has no such guarantee: this
// library's own allocator can reuse a just-freed handle's address for an
// unrelated buffer, and cache code keyed on that address would silently
// serve the wrong buffer's stale texture. See `TensorDyn::derive_identity`'s
// doc comment and task 17's report.

#[test]
fn dma_buffer_identity_is_derived_from_the_inode_not_the_handle_address() {
    if !edgefirst_tensor::is_dma_available() {
        eprintln!(
            "SKIPPED: dma_buffer_identity_is_derived_from_the_inode_not_the_handle_address \
                    - DMA not available"
        );
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

#[test]
fn two_distinct_views_of_one_dma_parent_share_one_identity_and_do_not_collide_with_an_unrelated_buffer(
) {
    // The brief's own bar for this test: "two distinct views of one parent
    // must not collide. A test that only checks a view against its parent
    // is weaker and will pass for the wrong reason." So this checks THREE
    // relationships, not one: parent<->view_a, parent<->view_b, and
    // view_a<->view_b directly -- plus a genuinely unrelated buffer, which
    // must NOT share their identity (the other half of "not colliding":
    // sharing an identity with something that is not actually the same
    // buffer would be the ABA hazard this fix closes).
    if !edgefirst_tensor::is_dma_available() {
        eprintln!("SKIPPED: two_distinct_views_of_one_dma_parent... - DMA not available");
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

// --- Task 18: PBO / CUDA / typed-downcast primitive family ----------------
//
// `Tensor::as_pbo`/`from_pbo`/`set_cuda_handle`, `TensorDyn::as_u8`/`as_i8`/
// `as_f32`/`as_f16`/`as_i16`/`into_u8`/... did not exist at all under
// `dynamic` before this task -- not stubs, absent (the exact gap that blocked
// `edgefirst-image-capi`'s own flip, task 9's report). Every test below
// would fail to *compile* against the code these methods replaced, which is
// the strongest form of "would fail against its absence" available.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Mock [`edgefirst_tensor::PboOps`] backed by a `Vec<u8>` instead of a real
/// GL buffer -- the same shape as `pbo.rs`'s own private `MockPboOps` test
/// double, reimplemented here because this integration test can only reach
/// the crate's public API. Counts calls so the test can assert the real
/// `PboOps` methods were actually invoked, not bypassed.
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
}

// SAFETY: the returned pointer addresses a `Vec<u8>` allocated once in `new`
// and never resized, so it stays valid for as long as the `MockPboOps` does
// -- which outlives every mapping handed out, since the tensor holds an
// `Arc` to it (mirrors `pbo.rs`'s own `MockPboOps` safety argument).
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

/// `Tensor::from_pbo`/`as_pbo` genuinely round-trip a PBO-backed tensor
/// through the `dynamic` backend: `memory()` reports `Pbo` (not whatever the
/// metadata-only real handle underneath happens to be), `as_pbo()` recovers
/// the exact `buffer_id`, and a CPU map through the recovered `PboTensor`
/// actually reaches the mock's own backing storage -- proving the map/unmap
/// calls are real `PboOps` dispatch, not a no-op stub.
#[test]
fn dynamic_pbo_wraps_and_reads_back_through_the_typed_tensor() {
    let ops = MockPboOps::new(16);
    let pbo = edgefirst_tensor::PboTensor::<u8>::from_pbo(7, 16, &[4, 4, 1], Some("t18_pbo"), ops)
        .expect("PboTensor::from_pbo");
    let mut tensor = Tensor::<u8>::from_pbo(pbo).expect("Tensor::from_pbo must allocate cleanly");

    assert_eq!(
        TensorTrait::memory(&tensor),
        TensorMemory::Pbo,
        "a PBO-backed dynamic tensor must report TensorMemory::Pbo, not the metadata handle's \
         own (host) storage kind"
    );

    tensor
        .set_format(PixelFormat::Grey)
        .expect("set_format on a PBO-backed tensor (real edgefirst-image call shape)");

    {
        let pbo_ref = tensor
            .as_pbo()
            .expect("as_pbo must recover the wrapped PboTensor");
        assert_eq!(
            pbo_ref.buffer_id(),
            7,
            "as_pbo must preserve the real GL buffer id"
        );
        let mut map = pbo_ref.map().expect("PboTensor::map through the mock ops");
        map.as_mut_slice().fill(0xEE);
    }
    // A second map proves the write above genuinely reached the mock's own
    // backing storage through PboOps::map_buffer/unmap_buffer -- not just a
    // local copy the first map handed out.
    let map = tensor
        .as_pbo()
        .expect("as_pbo must still resolve after the write")
        .map()
        .expect("second PboTensor::map");
    assert!(
        map.as_slice().iter().all(|&b| b == 0xEE),
        "the byte written through the first PBO map must be visible through a fresh one"
    );

    let dyn_tensor: TensorDyn = tensor.into();
    assert!(
        dyn_tensor
            .as_u8()
            .expect("as_u8 on the erased PBO tensor")
            .as_pbo()
            .is_some(),
        "as_pbo must still resolve after erasing back to TensorDyn and downcasting again"
    );
}

/// Mock [`edgefirst_tensor::CudaGlOps`] that never actually calls into
/// libcudart -- `set_cuda_handle`/`cuda()` are pure in-process Rust state
/// (see `TensorDyn::cuda`'s own doc comment), so proving the attach/read-back
/// round trip needs no CUDA-capable hardware, only that the handle is really
/// stored and really recovered.
struct MockCudaGlOps;
impl edgefirst_tensor::CudaGlOps for MockCudaGlOps {
    fn map(&self, _resource: *mut std::ffi::c_void) -> Option<(*mut std::ffi::c_void, usize)> {
        None
    }
    fn unmap(&self, _resource: *mut std::ffi::c_void) {}
    fn unregister(&self, _resource: *mut std::ffi::c_void) {}
}

/// `Tensor::set_cuda_handle`/`cuda` genuinely attach and read back a
/// [`edgefirst_tensor::CudaHandle`] on the `dynamic` backend: `None` before
/// attachment, `Some` with the same registration afterward.
#[test]
fn dynamic_set_cuda_handle_is_read_back_through_cuda() {
    let mut tensor = bare_u8(&[4, 4]);
    assert!(
        tensor.cuda().is_none(),
        "a tensor with no attached CUDA handle must report None"
    );

    let handle = edgefirst_tensor::CudaHandle::new_gl(
        std::ptr::null_mut(),
        16,
        Arc::new(MockCudaGlOps) as Arc<dyn edgefirst_tensor::CudaGlOps>,
    );
    tensor.set_cuda_handle(handle);

    assert!(
        tensor.cuda().is_some(),
        "set_cuda_handle must make cuda() return Some -- this was unconditionally None before \
         this task, and set_cuda_handle did not exist at all"
    );
}

/// The `as_u8`/`as_i8`/`as_f32`/`as_f16`/`as_i16`/`into_u8` family
/// (`TensorDyn`, built on the existing `as_typed`/`as_typed_mut` lens) did
/// not exist at all under `dynamic` before this task -- found missing when
/// `image-capi` was first built against `dynamic` with `opengl` enabled
/// (`gl/processor/mod.rs` calls `as_f32`/`as_f16`/`as_i8`/`as_i16`/`as_u8`
/// directly on a `TensorDyn`). Matches the dtype, rejects a mismatch, and
/// the consuming `into_*` form returns the original value on mismatch
/// rather than dropping it.
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

    // Consuming form: wrong dtype hands the original TensorDyn back in Err,
    // rather than silently dropping it (mirrors static's own
    // `downcast_methods!`-generated `into_*` contract exactly).
    let d = match d.into_u8() {
        Ok(_) => panic!("into_u8 on an f32 tensor must not succeed"),
        Err(d) => d,
    };
    let f32_tensor = d
        .into_f32()
        .expect("into_f32 must succeed on the f32 tensor it started as");
    assert_eq!(f32_tensor.shape(), &[4]);
}

/// Task-18 review, F32: `Tensor::from_pbo`'s metadata handle
/// (`dynamic_tensor.rs::from_pbo`, `TensorDyn::new(&pbo.shape, T::DTYPE,
/// Some(TensorMemory::Mem), None)`) is NOT metadata-only. `TensorDyn::new`
/// for `TensorMemory::Mem` can only drive `ef_tensor_builder_alloc`
/// (`tensor-capi/src/builder.rs`), which always allocates real host memory
/// sized to the full shape it is given -- there is no way through today's
/// ABI to mint a handle that carries a shape without backing it byte for
/// byte. This test performs exactly that same construction call directly
/// (not through `from_pbo`, since the `map_pin` guard added in this same
/// task correctly refuses to map a PBO-backed `TensorDyn` directly -- see
/// its own doc comment) and proves, by successfully mapping back the FULL
/// byte count, that the allocation is real and full-sized, not a small
/// placeholder.
///
/// For a realistic 4K RGBA16F PBO (3840x2160x4, `f16`) this is
/// **66,355,200 bytes (~63.3 MiB) of real host RAM**, allocated purely to
/// carry shape/dtype/format metadata for a buffer whose real data already
/// lives on the GPU -- exactly the host copy the PBO path exists to avoid.
///
/// This is a confirmed, unresolved finding, not something this task fixes:
/// no `ef_tensor_*` primitive today can decouple a handle's logical shape
/// from its backing size (`ef_tensor_builder_wrap` requires a real fd of at
/// least the shape's byte count; `TensorDyn::reshape` has no primitive at
/// all -- see its own doc comment in `dynamic_backend.rs`), and
/// `tensor-capi`/`tensor-ffi` -- where a genuinely metadata-only primitive
/// would have to live -- were mid-edit by a concurrent implementer
/// throughout this task's own review round, making landing one here both
/// out of scope and unsafe to attempt. Follow-up: a primitive along the
/// lines of `ef_tensor_builder_alloc_metadata` (or a `wrap`-style adopt of a
/// zero/near-zero-size in-process placeholder buffer) that records shape/
/// dtype/format without a matching allocation, paired with the existing
/// `map_pin` refusal (already in place) so nothing can be tricked into
/// reading past a metadata-only handle's real, tiny backing.
#[test]
fn from_pbo_metadata_handle_allocation_cost_is_the_full_pbo_byte_count() {
    let (w, h, c) = (3840usize, 2160usize, 4usize); // 4K, RGBA16F
    let expected_bytes = w * h * c * std::mem::size_of::<half::f16>();
    assert_eq!(
        expected_bytes, 66_355_200,
        "sanity check on the 4K RGBA16F byte count itself"
    );

    // Exactly the call `Tensor::<f16>::from_pbo` makes internally.
    let metadata_handle = TensorDyn::new(&[h, w, c], DType::F16, Some(TensorMemory::Mem), None)
        .expect("the same allocation from_pbo performs for a real PBO of this shape");

    let mapped = metadata_handle
        .map_bytes(CpuAccess::Read)
        .expect("a genuine TensorMemory::Mem handle must be host-mappable");
    assert_eq!(
        mapped.len(),
        expected_bytes,
        "from_pbo's metadata handle allocates the FULL pbo byte count as real host memory -- \
         confirmed by task-18's review (F32); not a small metadata-only footprint, and not yet \
         fixable without a new ef_tensor_* primitive (see this test's own doc comment)"
    );
}

// --- Task P2a: the primitives `edgefirst-python-common` needs --------------
//
// `crates/python-common/src/tensor.rs` is the first consumer to call
// `TensorDyn` at the Rust level under `dynamic`; every earlier one reached
// it through the `ef_tensor_*` C ABI instead. Nine of its call sites had no
// implementation on this backend at all. The tests below cover the ones
// whose answer this backend cannot derive locally -- each would fail
// against the "method does not exist" state that preceded it, and each is
// written to fail against a *plausible wrong* implementation too, which is
// the part that matters: `capacity_bytes` returning the logical size, or
// `row_stride` returning the effective pitch, both look right until a
// padded allocation crosses a package boundary.

/// `size` is the logical footprint and `capacity_bytes` is the allocation.
///
/// The two are equal for most tensors, which is why this uses an odd-width
/// NV12 image: its rows pad to a 64-byte pitch, so the allocation really is
/// larger than the shape implies. A `capacity_bytes` that simply returned
/// `size` -- which is exactly what `TensorTrait::capacity_bytes`'s default
/// does, and what this backend silently inherited before P2a -- fails here.
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

/// `row_stride` is the *recorded* pitch (`None` when tight);
/// `effective_row_stride` substitutes a computed one. The distinction is
/// what keeps a tight tensor's descriptor from claiming a pitch is
/// required.
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
///
/// **This does not prove the accessor is wired end to end**, and the name
/// says only what it establishes. No `CompressionScheme` is reachable off
/// Android, so this assertion holds equally against a hardcoded `None`,
/// against `ef_tensor_compression` always returning 0, and against the FFI
/// call being deleted outright. What it does catch is a wire table that
/// mis-maps code 0 -- reporting a vendor tile scheme for an ordinary linear
/// buffer, which every consumer would then decode as tiled. That mutation
/// does turn this red.
///
/// The genuinely unexercised path is the other direction: a real scheme
/// crossing the boundary. Closing it needs an Android target (or a fake
/// allocator behind `ef_tensor_compression`), and neither exists here. Named
/// as a gap rather than counted as coverage.
#[test]
fn compression_reports_linear_for_an_ordinary_allocation() {
    let t: TensorDyn = bare_u8(&[4, 4]).into();
    assert_eq!(t.compression(), None);
}

/// The standalone sync bracket really reaches the library's per-storage
/// logic: it succeeds on coherent host memory and is *refused* on a PBO,
/// whose coherency window is inseparable from its map. A local no-op
/// implementation -- the plausible wrong answer, since host memory needs no
/// maintenance -- would return `Ok` for the PBO too.
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

/// `CpuAccess::None` is a declaration, not a direction, and is refused at
/// the Rust boundary before any ABI call -- matching `map_pin`'s own rule.
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

/// `batch(n)` is a real zero-copy window on the leading dimension: a write
/// through the element is visible in the parent, and the two share one
/// `BufferIdentity`. An implementation that allocated a fresh tensor of the
/// right shape -- the plausible wrong answer -- passes a shape assertion
/// and fails both of these.
#[test]
fn batch_is_a_zero_copy_window_on_the_leading_dimension() {
    let parent = bare_u8(&[4, 2, 3]);
    let parent_dyn: TensorDyn = parent.into();

    let element = parent_dyn.batch(2).expect("batch(2) of a 4-element batch");
    assert_eq!(element.shape(), &[2, 3], "the leading dimension is dropped");
    // Sharing is proved by the write below, not by `buffer_identity`: for a
    // `Mem`-backed handle this backend derives identity from the handle's
    // own address (there is no system-level key to use -- see
    // `TensorDyn::derive_identity`), so a view and its parent legitimately
    // report different ids. `two_distinct_views_of_one_dma_parent_...`
    // above covers the DMA case, where the inode makes them agree.

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

/// An index past the leading dimension is refused with the *right* error,
/// not clamped onto a neighbouring element.
///
/// Asserts the variant, not `.is_err()`. A bare `.is_err()` holds against a
/// wrapper that returns a fixed error for everything, or the wrong variant
/// for every input -- which is exactly the defect the non-batched case
/// below caught.
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

/// `.batch()` on a tensor that is not batched at all reports **that**, and
/// not a false statement about the index.
///
/// The most ordinary possible mistake -- calling `.batch(0)` on an ordinary
/// image -- and the one this backend previously answered with "batch index
/// 0 out of bounds for batch size 720", which is not merely unhelpful but
/// untrue: 0 is in range. `ef_tensor_batch` returns `NULL` for four
/// distinct conditions and sets its own last-error message for each; the
/// wrapper used to discard all four.
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

/// `ef_tensor_batch`'s two refusals arrive as two different typed errors,
/// carried by the ABI's error class rather than by matching a string.
///
/// This replaces a test that pinned a fragment of
/// `BatchIndexOutOfBounds`'s `Display`: the wrapper used to recognise that
/// refusal by searching the advisory message for it, because the C ABI
/// carried a refusal's *kind* nowhere else. `ef_tensor_last_error_class`
/// (task P2c) carries it structurally now, and the fragment pin is gone
/// along with the coupling it guarded.
#[test]
fn the_two_batch_refusals_arrive_as_two_different_typed_errors() {
    // Out of range on a genuinely batched tensor: keeps its own variant,
    // with the real numbers.
    let batched: TensorDyn = bare_u8(&[4, 2, 3]).into();
    match batched.batch(9) {
        Err(Error::BatchIndexOutOfBounds { index: 9, batch: 4 }) => {}
        other => panic!("an out-of-range index must keep its own typed variant, got: {other:?}"),
    }

    // Not batched at all -- a different refusal, which must NOT arrive as
    // an index complaint about an index that was fine.
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

/// `view_region` reports an out-of-bounds crop as out-of-bounds, and any
/// *other* refusal as what it actually was.
///
/// It used to answer `RegionOutOfBounds` for every `NULL`, fabricating the
/// `bounds` from `width()`/`height()` -- which for a tensor carrying no
/// pixel format are `None`, rendered as `(0, 0)`. "Region does not fit in a
/// 0x0 frame", for a tensor that has no frame at all, is the same
/// confident falsehood `batch` had. Found by P2c's sweep of NULL-returning
/// entries rather than by a failing test.
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

    // A formatless tensor has no frame to crop. Whatever the producing side
    // calls that, it must not be dressed up as a bounds failure against an
    // invented frame.
    let bare: TensorDyn = bare_u8(&[8, 8]).into();
    if let Err(Error::RegionOutOfBounds { bounds, .. }) = bare.view(Region::new(0, 0, 2, 2)) {
        panic!(
            "a tensor with no pixel format has no frame; reporting bounds {bounds:?} \
             invents one"
        );
    }
}

/// `dmabuf_clone` refuses a tensor that is not DMA-backed rather than
/// handing back some other fd, and returns a real, independently-owned fd
/// when it is.
#[test]
fn dmabuf_clone_refuses_host_memory_and_dups_a_real_dma_fd() {
    let host: TensorDyn = bare_u8(&[4, 4]).into();
    match host.dmabuf_clone() {
        // The exact refusal, not merely "something mentioning DMA": without
        // the memory-kind check, `clone_fd` still fails on the absent fd and
        // its own message also contains "DMA", so a looser assertion here is
        // a gate that cannot fail -- confirmed by inducing exactly that. The
        // check is what makes the diagnosis name the operation and the real
        // backing, instead of a missing file descriptor.
        Err(Error::NotImplemented(msg)) => assert!(
            msg.contains("dmabuf_clone requires DMA-backed tensor") && msg.contains("Mem"),
            "the refusal must name the operation and the actual backing, got: {msg}"
        ),
        other => panic!("host memory has no dma-buf fd, got: {other:?}"),
    }

    let dma = match Tensor::<u8>::new(&[64, 64], Some(TensorMemory::DmaBuf), None) {
        Ok(t) => t,
        // Same host-capability skip the protocol round-trip tests use: an
        // absent or unopenable /dev/dma_heap is a platform fact, not a
        // failure of this code.
        Err(Error::IoError(e))
            if matches!(
                e.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied
            ) =>
        {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "SKIP: dmabuf_clone_refuses_host_memory_and_dups_a_real_dma_fd -- \
                 no usable dma-buf heap on this host"
            );
            return;
        }
        Err(e) => panic!("unexpected dma-buf allocation failure: {e:?}"),
    };
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

/// `cuda_map` fast-fails to `None` with no handle attached, and produces a
/// guard once one is. Needs no CUDA hardware: `CudaHandle` is in-process
/// Rust state routed through a caller-supplied `CudaGlOps` (see
/// `TensorDyn::cuda`'s doc comment), so a mock that hands back a real
/// address proves the whole path.
#[test]
fn cuda_map_is_none_without_a_handle_and_maps_with_one() {
    /// Unlike `MockCudaGlOps` above (which returns `None` from `map`,
    /// enough to test attach/read-back), this one hands back a real
    /// address so `cuda_map` can produce an actual guard.
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
    let map = dyn_t
        .cuda_map()
        .expect("an attached CUDA handle must produce a map guard");
    assert_eq!(map.len(), 16);
    assert!(!map.device_ptr().is_null());
}

/// A PBO whose GL buffer is larger than its shape product reports the
/// buffer's size, not the companion metadata handle's.
///
/// `PboTensor::from_pbo` explicitly permits `size > shape.product()` for a
/// pitch-aligned PBO, while `Tensor::from_pbo` sizes the companion `Mem`
/// handle to the shape product exactly. Reading the companion understates
/// the real allocation, which clamps a `kind::PBO` descriptor's `capacity`
/// and leaves a consumer's `from_pbo_import` mapping only part of the
/// buffer -- the same pool-reuse breakage already fixed for `kind::HOST`.
///
/// Every production caller happens to size a PBO to match its shape today
/// (`edgefirst-image`'s `gl/threaded.rs`), so nothing else in the tree can
/// catch a regression here; this test deliberately does not.
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
        "capacity_bytes must report the GL buffer's real size, not the companion \
         metadata handle's shape-sized allocation"
    );
    assert_eq!(
        t.descriptor().capacity,
        ALLOCATED as u64,
        "...and the descriptor must carry it across the package boundary, or a \
         consumer re-importing this PBO maps only part of the buffer"
    );
}

/// `pbo_id`, `pbo_vtable_ptr` and `pbo_keepalive` all reach the erased
/// `PboTensor<T>` through the dtype downcast, and all report `None` for a
/// tensor that has no PBO. The keepalive is what a cross-package capsule
/// holds alongside the vtable address, so a `None` here would dangle that
/// address the moment the producing tensor dropped.
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

    // A non-u8 PBO proves the downcast really dispatches on dtype rather
    // than always trying `PboTensor<u8>`.
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

/// A `Tensor<T>` whose handle was minted for a *different* element type --
/// the layout-identical `transmute` `edgefirst-image` performs to hand back
/// an i8 view of a u8 PBO or DMA buffer -- must report `T`'s dtype once
/// erased back to `TensorDyn`, not the handle's original one.
///
/// Under `static` this is free: `TensorDyn::from(Tensor<i8>)` picks the
/// `I8` variant from the Rust type, which the transmute has already
/// changed. Under `dynamic` the element type lives in the C handle, not in
/// the type parameter, so the transmute changes a `PhantomData` and nothing
/// else -- `create_image(dtype="int8")` reported `uint8`.
#[test]
fn a_retagged_typed_lens_erases_to_its_own_dtype_not_the_handles() {
    let t_u8 = bare_u8(&[4, 4]);
    assert_eq!(TensorTrait::memory(&t_u8), TensorMemory::Mem);
    // SAFETY: mirrors `edgefirst-image`'s own i8-over-a-u8-buffer transmute
    // (`crates/image/src/lib.rs`); `Tensor<u8>` and `Tensor<i8>` are
    // layout-identical on both backends.
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

// --- Task P2b: behaviour, not merely a compiling surface ------------------
//
// Four regressions were found by running the Python suite against
// dynamic-linked wheels, and a fifth by sweeping the static backend's
// per-storage dispatch points afterwards. All five methods *existed* on
// both backends and `cargo check` was clean; they differed at runtime.

/// `reshape` really reshapes, and reports the new geometry immediately.
///
/// Was `Err(NotImplemented("no ef_tensor_reshape primitive exists in the
/// dynamic backend"))`. The cache assertion is the load-bearing half:
/// `shape()` serves a cached `Vec`, not the handle, so a reshape that
/// forgot to refresh it would succeed and then keep reporting the old
/// geometry -- a wrong answer rather than a refusal.
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

/// `clone_fd` works for every backing that has a file descriptor, not just
/// DMA-BUF.
///
/// It used to derive the fd from plane 0's *native handle*, which is a
/// dma-buf fd on Linux and `-1` for every other backing -- so SHM-backed
/// tensors were refused with "this tensor has no native fd (not
/// DMA-backed)", while the static backend clones theirs without complaint.
#[test]
fn clone_fd_works_for_shm_not_only_dma() {
    const SHAPE: [usize; 2] = [64, 64];
    let shm = match Tensor::<u8>::new(&SHAPE, Some(TensorMemory::Shm), None) {
        Ok(t) => t,
        // Only a genuine platform absence is a skip. Any other error is this
        // code failing and must not be swallowed -- the same discrimination
        // the DMA tests above apply, rather than a blanket catch that turns
        // every failure into a silent pass.
        Err(Error::IoError(e))
            if matches!(
                e.kind(),
                std::io::ErrorKind::NotFound | std::io::ErrorKind::PermissionDenied
            ) =>
        {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "SKIP: clone_fd_works_for_shm_not_only_dma -- no usable /dev/shm here"
            );
            return;
        }
        Err(e) => panic!("unexpected SHM allocation failure: {e:?}"),
    };

    // A marker written through the tensor, so the fd can be checked against
    // the bytes rather than against `>= 0`.
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

    // Assert the BYTES, not that the call returned something non-negative.
    // The plumbing under test is a `c_int` that is either an fd or a negated
    // errno, and a `>= 0` check passes for a dup of any unrelated descriptor
    // that happens to be open -- which is precisely the failure mode worth
    // catching. Re-importing through the public constructor and reading the
    // marker back is what makes "the right fd" unambiguous.
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

/// Mapping a PBO-backed tensor through the type-erased handle reads the GL
/// buffer, rather than refusing.
///
/// It used to refuse ("use `Tensor::as_pbo().map()` instead"). That was a
/// defensible half-truth -- mapping the *companion* handle would hand back
/// unrelated host bytes -- but the static backend simply dispatches
/// (`TensorStorage::Pbo(t) => t.map_with(..)`), and Python's
/// `normalize_to_numpy()` maps whatever `convert()` returned, which on a GL
/// machine is a PBO. Every GPU conversion result was unreadable.
///
/// Asserts the bytes, not just that the call succeeded: a map that returned
/// the companion `Mem` allocation would also be `Ok`, and would be exactly
/// the silent wrongness the refusal was protecting against.
#[test]
fn mapping_a_pbo_backed_tensor_reads_the_gl_buffer() {
    let ops = MockPboOps::new(16);
    let pbo = edgefirst_tensor::PboTensor::<u8>::from_pbo(31, 16, &[4, 4, 1], None, ops)
        .expect("PboTensor::from_pbo");
    let tensor = Tensor::<u8>::from_pbo(pbo).expect("Tensor::from_pbo");
    // Write a marker through the PBO's own typed path...
    {
        let pbo_ref = tensor.as_pbo().expect("as_pbo");
        let mut m = pbo_ref.map().expect("typed PBO map");
        m.as_mut_slice().fill(0xC3);
    }
    let dyn_t: TensorDyn = tensor.into();

    // ...and read it back through the type-erased handle, which is the path
    // `normalize_to_numpy()` takes.
    let bytes = dyn_t
        .map_bytes(CpuAccess::Read)
        .expect("a PBO-backed tensor must be mappable through TensorDyn");
    assert_eq!(bytes.as_slice().len(), 16);
    assert!(
        bytes.as_slice().iter().all(|&b| b == 0xC3),
        "the erased map must read the GL buffer, not the companion host allocation"
    );
}

/// A retagged PBO tensor still resolves its buffer id, vtable, keepalive
/// and map -- the second-order effect of retagging the handle's dtype.
///
/// `edgefirst-image` allocates an int8 PBO as `u8` and hands it back as an
/// `i8` tensor. `From<Tensor<T>> for TensorDyn` retags the *handle* so the
/// dtype is honest, but the `PboTensor<u8>` behind `TensorDyn::pbo` sits
/// behind a real `Any` vtable that no transmute of the enclosing
/// `Tensor<T>` touches. Every accessor that picked its downcast target from
/// `dtype()` therefore looked for a `PboTensor<i8>`, found nothing, and
/// reported the tensor as having no PBO at all -- `pbo_id()` was `None`, so
/// its descriptor carried no buffer id and a cross-package re-import died
/// with `InvalidArgument("PBO descriptor carries no buffer id")`.
#[test]
fn a_retagged_pbo_still_resolves_its_buffer_id_vtable_and_map() {
    let ops = MockPboOps::new(16);
    let pbo = edgefirst_tensor::PboTensor::<u8>::from_pbo(41, 16, &[4, 4, 1], None, ops)
        .expect("PboTensor::from_pbo -- allocated as u8, as the GL backend does");
    let as_u8 = Tensor::<u8>::from_pbo(pbo).expect("Tensor::from_pbo");
    {
        let p = as_u8.as_pbo().expect("as_pbo before retagging");
        let mut m = p.map().expect("typed PBO map");
        m.as_mut_slice().fill(0x7E);
    }

    // The int8 hand-back: `Tensor<u8>` -> `Tensor<i8>` by layout-identical
    // transmute, exactly as `crates/image/src/lib.rs` does it.
    // SAFETY: same rationale as that call site.
    let as_i8: Tensor<i8> = unsafe { std::mem::transmute(as_u8) };
    assert!(
        as_i8.as_pbo().is_some(),
        "as_pbo must find the stored PboTensor whatever element type it was created with"
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

/// The two accessors every GPU/G2D import site depends on answer correctly
/// for a genuinely DMA-backed tensor.
///
/// Those sites -- `crates/image`'s EGLImage import, its Path-B R8 import,
/// `import_buffer_packed`, and four G2D sites -- used to reach the fd via
/// `Tensor::as_dma()`, a downcast to `static`-backend-internal storage that
/// this backend returned `None` for **unconditionally**. Each one handled
/// that `None` by declining and falling back to a slower path, so a
/// DMA-backed tensor silently lost its zero-copy hardware route with
/// nothing failing and nothing logged. Reviewed as F3 on task P2b.
///
/// They now ask the two questions they actually have: "is this DMA-backed"
/// (`memory()`) and "give me the fd" (`dmabuf()`). Both are implemented on
/// both backends with the same signature. This pins them together, because
/// a caller needs both to hold before it can import.
#[test]
fn a_dma_tensor_answers_both_questions_the_gpu_import_sites_ask() {
    let dma = match Tensor::<u8>::new(&[64, 64], Some(TensorMemory::DmaBuf), None) {
        Ok(t) => t,
        Err(e) => {
            use std::io::Write;
            let _ = writeln!(
                std::io::stderr(),
                "SKIP: a_dma_tensor_answers_both_questions_the_gpu_import_sites_ask -- \
                 no usable dma-buf heap here ({e:?})"
            );
            return;
        }
    };

    // The capability probe (`g2d.rs`'s two call sites).
    assert_eq!(
        TensorTrait::memory(&dma),
        TensorMemory::DmaBuf,
        "a DMA-backed tensor must report DmaBuf -- the probe that decides whether the \
         G2D hardware path is even attempted"
    );

    // The fd accessor (the four import sites).
    use std::os::fd::AsRawFd;
    let fd = dma
        .dmabuf()
        .expect("a DMA-backed tensor must yield its fd -- this is what the EGLImage import needs");
    assert!(fd.as_raw_fd() >= 0, "and it must be a real descriptor");

    // A Mem-backed tensor must still answer no to both, or the probe would
    // wave through a tensor the hardware path cannot use.
    let host = bare_u8(&[64, 64]);
    assert_eq!(TensorTrait::memory(&host), TensorMemory::Mem);
    assert!(host.dmabuf().is_err());
}

/// Geometry changes reach the wrapped `PboTensor`, not just the companion
/// handle.
///
/// A PBO-backed `TensorDyn` carries geometry in **two** places: the
/// `ef_tensor_*` handle (which serves `shape()`) and the `PboTensor` behind
/// `TensorDyn::pbo` (which serves `as_pbo()`, and which `edgefirst-image`
/// reads for the GL buffer's own geometry). A mutator that updates only the
/// first leaves the two disagreeing -- `shape()` says one thing and
/// `as_pbo().shape` another. Reviewed as F6 on task P2b.
///
/// The byte length coincides for `reshape`, since it preserves the element
/// count, so a map still returns the right span and nothing errors. That is
/// what makes it the "stops erroring but returns something subtly
/// different" case rather than a loud one.
///
/// Covers all three mutators that change geometry the `PboTensor` also
/// carries. `set_format`, `set_row_stride`, `set_colorimetry`,
/// `set_quantization` and `set_dtype` are **not** here on purpose: a
/// `PboTensor` holds no parallel copy of any of those, and
/// `set_plane_offset` explicitly skips PBO storage on the static side too
/// (`lib.rs`'s `_ => {}`).
#[test]
fn geometry_mutators_keep_the_wrapped_pbo_in_step() {
    fn pbo_tensor(bytes: usize, shape: &[usize]) -> TensorDyn {
        let ops = MockPboOps::new(bytes);
        let pbo = edgefirst_tensor::PboTensor::<u8>::from_pbo(51, bytes, shape, None, ops)
            .expect("PboTensor::from_pbo");
        Tensor::<u8>::from_pbo(pbo)
            .expect("Tensor::from_pbo")
            .into()
    }

    // --- reshape: same element count, different rank -----------------
    let mut t = pbo_tensor(16, &[4, 4, 1]);
    t.reshape(&[2, 8]).expect("equal element count");
    assert_eq!(t.shape(), &[2, 8], "the handle's shape follows");
    assert_eq!(
        t.as_u8().expect("as_u8").as_pbo().expect("as_pbo").shape,
        vec![2, 8],
        "and so must the wrapped PboTensor's -- edgefirst-image reads the GL buffer's \
         geometry off as_pbo(), so a stale shape here is a wrong answer that never errors"
    );

    // --- set_logical_shape: fewer elements, still fits ---------------
    let mut t = pbo_tensor(16, &[4, 4, 1]);
    t.set_logical_shape(&[2, 4])
        .expect("8 bytes fits a 16-byte GL buffer");
    assert_eq!(t.shape(), &[2, 4]);
    assert_eq!(
        t.as_u8().expect("as_u8").as_pbo().expect("as_pbo").shape,
        vec![2, 4],
        "the capacity-based reconfigure must reach the PboTensor too"
    );

    // --- configure_image: the pool-reuse path ------------------------
    let mut t = pbo_tensor(64, &[8, 8, 1]);
    t.configure_image(4, 4, PixelFormat::Grey)
        .expect("a smaller image fits the 64-byte buffer");
    assert_eq!(
        t.as_u8().expect("as_u8").as_pbo().expect("as_pbo").shape,
        t.shape().to_vec(),
        "configure_image is the decode-into-a-pool path; the two geometries must agree \
         after it, or the next GL import reads the old one"
    );
}
