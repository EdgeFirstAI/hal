// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! Methods expressible purely over the primitive `TensorDyn` API, compiled into
//! both the `static` and `dynamic` backends.
//!
//! [`TensorDyn::import_descriptor`] is the consumer half of the cross-package
//! tensor protocol. The per-kind construction switch is each backend's
//! `import_storage`; the validation before it and the metadata restore after it
//! live here.

use crate::{DType, Result, TensorDesc, TensorDyn};

impl TensorDyn {
    /// Rebuild a tensor addressing the memory an [`crate::TensorDesc`]
    /// describes: the consumer half of the cross-package tensor protocol (see
    /// [`crate::protocol`]'s module docs).
    ///
    /// The result aliases the producer's memory and is valid only while the
    /// producer keeps that memory alive; for the capsule protocol that is the
    /// capsule's lifetime. A `DMABUF` import `dup`s the fd and a
    /// `D3D11_TEXTURE` import duplicates the texture and fence NT handles, so
    /// those results stay valid after the producer's tensor drops. Both handle
    /// values of a `D3D11_TEXTURE` descriptor must be valid in this process.
    /// For any other kind (`HOST`, `IOSURFACE`) a consumer that must outlive
    /// the capsule takes its own reference.
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

/// Refuse a descriptor this build cannot honour, and decode the `dtype` and
/// `shape` of one that it can.
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
    // A fence is accepted only on a `D3D11_TEXTURE` descriptor, whose import
    // waits on it.
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
    // A `D3D11_TEXTURE` descriptor with a fence value needs the fence handle in
    // `ptr`.
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
fn restore_descriptor_metadata(
    t: &mut TensorDyn,
    desc: &TensorDesc,
    shape: &[usize],
) -> Result<()> {
    // `format` is the HAL format code; `fourcc` is used only when `format` is
    // NONE.
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

    t.set_colorimetry(if desc.colorimetry == 0 {
        None
    } else {
        Some(crate::Colorimetry::unpack(desc.colorimetry))
    });

    restore_d3d11_logical_shape(t, desc, shape)?;

    // Narrow the declared CPU access to the producer's.
    if let Some(access) = crate::protocol::flags::cpu_access(desc.flags) {
        t.restrict_cpu_access(access);
    }
    if desc.flags & crate::protocol::flags::VIEW != 0 {
        t.mark_view();
    }

    Ok(())
}

/// Give a `kind::D3D11_TEXTURE` import the shape its producer had: the import
/// opens the texture at the format's allocation shape, and this narrows it to
/// the descriptor's shape with `set_logical_shape`. Runs after the format
/// restore.
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

/// Restore the producer's physical row pitch (the row dimension convention
/// documented on `protocol::from_parts`) onto an import of kind `MEM`, `SHM`,
/// `DMABUF`, `IOSURFACE` or `PBO`. A stride whose `stride * rows` exceeds the
/// import's capacity is logged and not restored.
fn restore_imported_row_stride(
    t: &mut TensorDyn,
    desc: &TensorDesc,
    shape: &[usize],
    fmt: crate::PixelFormat,
) {
    if !matches!(
        desc.kind,
        crate::protocol::kind::MEM
            | crate::protocol::kind::SHM
            | crate::protocol::kind::DMABUF
            | crate::protocol::kind::IOSURFACE
            | crate::protocol::kind::PBO
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
    // A negative stride is skipped.
    let Some(&stride_bytes) = desc.strides().get(row_dim) else {
        return;
    };
    let Ok(stride_bytes) = usize::try_from(stride_bytes) else {
        return;
    };
    // Rows the stride applies to; `desc.strides()` is in bytes.
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
