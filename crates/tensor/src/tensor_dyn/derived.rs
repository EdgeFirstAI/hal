// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Methods expressible purely over the primitive `TensorDyn` API, written
//! once and compiled into both the `static` and `dynamic` backends.
//!
//! [`TensorDyn::import_descriptor`] is the first resident: the consumer
//! half of the cross-package tensor protocol. It was `static`-only until
//! `edgefirst-python-common` -- the first consumer to call `TensorDyn` at
//! the Rust level under `dynamic` -- needed it, and almost all of it is
//! backend-independent. Only the per-kind *construction* switch names
//! constructors and therefore differs; that stays in each backend as
//! `import_storage`, and the validation before it and the metadata restore
//! after it live here, once. Writing the whole function twice was the
//! alternative and it is the shape this branch exists to delete: the
//! stride-restore reasoning below is forty lines of load-bearing
//! commentary that a second copy would eventually contradict.

use crate::{DType, Result, TensorDesc, TensorDyn};

impl TensorDyn {
    /// Rebuild a tensor addressing the memory an [`crate::TensorDesc`]
    /// describes — the consumer half of the cross-package tensor protocol
    /// (see [`crate::protocol`]'s module docs).
    ///
    /// **Borrow semantics.** The result *aliases* the producer's memory; it
    /// is valid only while the producer keeps that memory alive. For the
    /// capsule protocol, that is the capsule's own lifetime: the capsule's
    /// keepalive holds the producing tensor (and, for a `HOST` descriptor,
    /// its pin) for as long as the capsule itself is alive. A `DMABUF`
    /// import is the one exception with its own, narrower guarantee — this
    /// function `dup`s the fd, so the imported tensor's fd stays valid even
    /// after the producer's tensor drops (closing its own copy); the
    /// underlying dma-buf allocation itself is still only alive as long as
    /// *some* fd or GPU import references it. `D3D11_TEXTURE` behaves the
    /// same way for the same reason: the import duplicates the texture and
    /// fence NT handles it keeps, so the result outlives the producer's
    /// tensor while the texture itself lives as long as some handle or view
    /// references it. Both of the descriptor's handle values must be valid
    /// **in this process** — the protocol duplicates, it does not reach into
    /// another process's handle table (the blob transport does; see
    /// [`crate::blob::import`]). A consumer needing to survive past the
    /// capsule for any other kind (`HOST`, `IOSURFACE`) must take its own
    /// reference explicitly — this function does not.
    ///
    /// # Errors
    ///
    /// * [`Error::NotImplemented`](crate::Error::NotImplemented) — an ABI
    ///   major this build does not understand ([`desc.version`](crate::TensorDesc::version)),
    ///   a dtype/kind code this build does not recognise, or a kind this
    ///   platform cannot import (e.g. dma-buf off Linux).
    /// * [`Error::InvalidArgument`](crate::Error::InvalidArgument) — a
    ///   malformed descriptor: a missing fd/surface id/host address for the
    ///   kind it claims to be, or (macOS/iOS) a surface id that no longer
    ///   resolves to a live `IOSurface`.
    pub fn import_descriptor(desc: &TensorDesc) -> Result<Self> {
        let (dtype, shape) = validate_descriptor(desc)?;
        let mut t = Self::import_storage(desc, &shape, dtype)?;
        restore_descriptor_metadata(&mut t, desc, &shape)?;
        Ok(t)
    }
}

/// Refuse a descriptor this build cannot honour, and decode the two facts
/// every `import_storage` arm needs from one that it can.
///
/// Separate from the construction switch so the refusals apply identically
/// on both backends -- a descriptor is untrusted cross-package input, and
/// "which producers do we refuse" is exactly the decision that must not
/// have two implementations.
fn validate_descriptor(desc: &TensorDesc) -> Result<(DType, Vec<usize>)> {
    if desc.version != crate::ABI_VERSION {
        return Err(crate::Error::NotImplemented(format!(
            "tensor interop ABI v{} (this build understands v{})",
            desc.version,
            crate::ABI_VERSION
        )));
    }

    if desc.ndim as usize > crate::TensorDesc::MAX_NDIM {
        return Err(crate::Error::NotImplemented(format!(
            "descriptor declares rank {}, but this build's descriptor \
             carries only {} shape slots; importing would address a \
             prefix of a larger allocation and silently read wrong data",
            desc.ndim,
            crate::TensorDesc::MAX_NDIM
        )));
    }
    // A producer advertising a fence we cannot wait on is a correctness
    // problem, not a compatibility nicety: importing anyway would alias
    // memory whose contents are still being written by the producer's
    // device, and the corruption would be timing-dependent.
    //
    // `D3D11_TEXTURE` is the one kind with a wait path: its descriptor names
    // an `ID3D11Fence` plus a value, and `from_d3d11_shared_handle` issues
    // the GPU-side wait before the import is usable. Every other kind still
    // has nowhere to wait, so a producer advertising a fence is refused.
    if desc.flags & crate::protocol::flags::SYNC_PRESENT != 0
        && desc.kind != crate::protocol::kind::D3D11_TEXTURE
    {
        return Err(crate::Error::NotImplemented(
            "descriptor advertises SYNC_PRESENT, but waiting on a \
             producer fence is not implemented in this build; importing \
             would alias memory with device work still in flight"
                .to_owned(),
        ));
    }
    // The same refusal for a `D3D11_TEXTURE` descriptor that advertises a
    // fence value with no fence to read it on: `ptr` is where that kind
    // carries the fence handle, so a null one leaves a completion nobody can
    // wait on, which is the in-flight-write hazard again rather than a
    // missing feature.
    if desc.flags & crate::protocol::flags::SYNC_PRESENT != 0
        && desc.kind == crate::protocol::kind::D3D11_TEXTURE
        && desc.ptr.is_null()
    {
        return Err(crate::Error::InvalidArgument(
            "D3D11_TEXTURE descriptor advertises SYNC_PRESENT but carries no \
             fence handle in `ptr`; the completion it names cannot be waited on"
                .to_owned(),
        ));
    }
    let dtype = crate::protocol::dtype_to_dtype(desc.dtype).ok_or_else(|| {
        crate::Error::NotImplemented(format!("tensor interop dtype code {}", desc.dtype))
    })?;
    let shape: Vec<usize> = desc.shape().iter().map(|d| *d as usize).collect();
    Ok((dtype, shape))
}

/// Restore the producer's format, row pitch and colorimetry onto a
/// freshly-imported tensor.
///
/// Runs after `import_storage` on both backends, over the public setters
/// (`set_format`, `set_row_stride`, `set_colorimetry`) both already have --
/// which is what makes it shareable at all.
fn restore_descriptor_metadata(
    t: &mut TensorDyn,
    desc: &TensorDesc,
    shape: &[usize],
) -> Result<()> {
    // `format` is the HAL-aware format code; it can represent every
    // `PixelFormat` (including Planar, which has no FourCC). Fall back
    // to `fourcc` only when the producer left `format` at NONE, for
    // third-party/DRM-only producers that never set it.
    let format = if desc.format != crate::protocol::format::NONE {
        crate::protocol::format_from_code(desc.format)
    } else if desc.fourcc != 0 {
        crate::PixelFormat::from_fourcc(desc.fourcc)
    } else {
        None
    };
    if let Some(fmt) = format {
        t.set_format(fmt)?;
        restore_imported_row_stride(t, desc, shape, fmt);
    }

    // Colorimetry does not participate in `set_format`'s validation, so
    // this can be unconditional: `0` unpacks to all-`None`, matching a
    // freshly-imported tensor's default and costing nothing extra.
    t.set_colorimetry(if desc.colorimetry == 0 {
        None
    } else {
        Some(crate::Colorimetry::unpack(desc.colorimetry))
    });

    restore_d3d11_logical_shape(t, desc, shape)?;

    Ok(())
}

/// Give a `kind::D3D11_TEXTURE` import the shape its producer had.
///
/// The import opens the texture at the format's *allocation* shape, because
/// that is what `from_d3d11_shared_handle` builds from width and height. A
/// producer that called `set_logical_shape` was carrying the *addressing*
/// shape instead (`[h, w]` for a semi-planar image rather than
/// `[combined_h, w]`), and the descriptor faithfully reported it; without
/// this the consumer would see a shape its producer did not have. The
/// import arm has already checked `shape` is one of those two spellings of
/// the geometry the texture itself reports, so nothing untrusted reaches
/// `set_logical_shape` here.
///
/// After the format restore, not before: `set_format` validates the shape it
/// finds, and an addressing shape is not one it accepts for a semi-planar
/// format (`[481, 640]` is an unreachable combined-plane height).
///
/// No other kind needs this. Every one of them is imported at exactly the
/// shape the descriptor carries.
fn restore_d3d11_logical_shape(
    t: &mut TensorDyn,
    desc: &TensorDesc,
    shape: &[usize],
) -> Result<()> {
    if desc.kind != crate::protocol::kind::D3D11_TEXTURE || t.shape() == shape {
        return Ok(());
    }
    t.set_logical_shape(shape)
}

/// Restore the producer's physical row pitch (the row dimension
/// convention documented on `protocol::from_parts`) so a decode
/// that reconfigures this import within its allocation --
/// `configure_image`'s pool-reuse path -- keeps writing at the
/// producer's real stride instead of recomputing a tighter one
/// that only fits today's logical shape. Without this, `capacity`
/// alone is not enough: `configure_image` only *prefers* a prior
/// stride when one is recorded, so a fresh import with no stride
/// still picks a freshly-computed (and possibly narrower) pitch.
///
/// `HOST` and `DMABUF`. Historical note: restoring a stride for an
/// imported `DMABUF` used to be actively harmful -- `Tensor::map`'s
/// strided path rejected any imported (non-self-allocated) DMA-BUF
/// outright, so recording one here would have made a foreign
/// DMA-BUF's CPU map fail with `InvalidOperation`. That was fixed
/// at the map site instead (a strided imported DMA-BUF is now
/// accepted there, bounds-checked against `buf_size` exactly like
/// a self-allocated one is -- `fstat`-derived kernel truth when
/// the kernel reports a usable size, the declared logical size in
/// the rare fallback, and either way an out-of-range mmap is still
/// rejected by the kernel), which is what makes restoring it here
/// safe: the same argument that justified loosening `Tensor::map`
/// applies verbatim to the value being restored here -- it is
/// bounds-checked again by `set_row_stride`'s caller
/// (`configure_image`, later) and by `Tensor::map` itself at write
/// time, so an oversized or malformed producer-reported stride
/// cannot cause an OOB write even though it is trusted without
/// independent verification.
/// Confirmed load-bearing, not just anticipated: a pool-sized
/// (oversized) DMA-BUF-backed destination decoding a smaller
/// image lost its real pitch without this, recomputing a
/// tighter one instead -- silent misalignment for any GPU
/// consumer reading at the true (wider) physical pitch.
///
/// `IOSURFACE` is not included: nothing has reported this gap for
/// it, and unlike `DMABUF` its CPU-mapping was never restricted by
/// `is_imported` in the first place, so there is no known-broken
/// case pulling it in yet. The same argument would apply if one
/// surfaces.
fn restore_imported_row_stride(
    t: &mut TensorDyn,
    desc: &TensorDesc,
    shape: &[usize],
    fmt: crate::PixelFormat,
) {
    if !matches!(
        desc.kind,
        crate::protocol::kind::HOST | crate::protocol::kind::DMABUF
    ) {
        return;
    }
    let row_dim = match fmt.layout() {
        crate::PixelLayout::Planar if desc.ndim as usize >= 3 => 1,
        _ => 0,
    };
    if (desc.ndim as usize) < 2 {
        return;
    }
    // `strides` is signed to leave room for a future flip
    // (negative stride); nothing produces one today, and
    // `set_row_stride` has no way to express it, so a
    // negative value here is simply skipped rather than
    // reinterpreted as a huge unsigned pitch.
    let Some(&stride_bytes) = desc.strides().get(row_dim) else {
        return;
    };
    let Ok(stride_bytes) = usize::try_from(stride_bytes) else {
        return;
    };
    // `set_row_stride` enforces only a minimum
    // (`stride >= min_stride`); it explicitly does no
    // size validation, by design (it is pure layout
    // metadata, reused by `strides_follow_row_stride
    // _not_shape`-style tests that never touch a real
    // allocation). Import is not that case: `desc` is
    // an untrusted cross-package payload, and an
    // oversized stride here would otherwise be
    // caught only downstream, by whichever consumer
    // happens to multiply it out first
    // (`Tensor::map`'s strided path, or
    // `configure_image`'s own capacity check). The
    // multiply below is checked -- the descriptor's
    // stride is untrusted input, so it must not wrap
    // into a small-looking value that passes a
    // capacity comparison it should fail.
    //
    // `desc.strides()` is in bytes, so there is no
    // element->byte conversion here (and no second
    // multiply that could overflow) any more.
    let rows: usize = match fmt.layout() {
        crate::PixelLayout::Planar if shape.len() >= 2 => shape[0].saturating_mul(shape[1]),
        _ => shape.first().copied().unwrap_or(0),
    };
    apply_imported_row_stride(t, fmt, stride_bytes, rows);
}

fn apply_imported_row_stride(
    t: &mut TensorDyn,
    fmt: crate::PixelFormat,
    stride_bytes: usize,
    rows: usize,
) {
    let capacity = t.capacity_bytes();
    match stride_bytes.checked_mul(rows) {
        Some(needed) if needed <= capacity => {
            if let Err(e) = t.set_row_stride(stride_bytes) {
                log::warn!(
                    "import_descriptor: producer row_stride \
                         {stride_bytes} bytes rejected for {fmt:?} \
                         ({e}); keeping the freshly-imported tight \
                         pitch instead"
                );
            }
        }
        Some(needed) => {
            log::warn!(
                "import_descriptor: producer row_stride \
                     {stride_bytes} bytes × {rows} rows needs {needed} \
                     bytes, exceeding this import's capacity \
                     {capacity}; keeping the freshly-imported tight \
                     pitch instead of trusting an oversized descriptor"
            );
        }
        None => {
            log::warn!(
                "import_descriptor: producer row_stride \
                     {stride_bytes} bytes × {rows} rows overflows \
                     usize computing bytes needed; keeping the \
                     freshly-imported tight pitch instead of trusting \
                     a malformed descriptor"
            );
        }
    }
}
