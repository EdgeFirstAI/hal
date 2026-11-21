// SPDX-FileCopyrightText: Copyright 2026 Au-Zone Technologies
// SPDX-License-Identifier: Apache-2.0

//! The dynamic backend: a newtype over the opaque C handle that calls into
//! `libedgefirst_tensor.so`.
//!
//! A primitive method is a direct `ef_tensor_*` call; a derived method is built
//! from the primitives and lives in `derived.rs`. The struct also caches the
//! shape and a locally derived [`crate::BufferIdentity`].

use std::fmt;
use std::ptr::NonNull;

use edgefirst_tensor_ffi::EfTensor;

use crate::{
    BufferIdentity, Colorimetry, CpuAccess, DType, Error, IdentityKind, PixelFormat, PixelLayout,
    Region, Result, TensorMemory, ViewOrigin,
};

/// The tensor handle and the values cached alongside it.
pub struct TensorDyn {
    handle: Handle,
    shape_cache: Vec<usize>,
    identity: BufferIdentity,
    /// Quantization, read through `ef_tensor_quantization_{info,get}` on the
    /// first [`Self::quantization`] call and cached. Replaced by
    /// [`Self::set_quantization`] and [`Self::clear_quantization`]. Each
    /// `TensorDyn` wrapping the same handle has its own cache.
    quantization_cache: std::sync::OnceLock<Option<crate::Quantization>>,
    /// Retained lens over the library-owned chroma plane, obtained from
    /// `ef_tensor_chroma` on the first [`Self::chroma`] call.
    chroma_cache: std::sync::OnceLock<Option<Box<TensorDyn>>>,
    /// The duplicated texture NT handle [`Self::descriptor_pinned`] puts in a
    /// [`crate::protocol::kind::D3D11_TEXTURE`] descriptor, closed when this
    /// value drops.
    #[cfg(target_os = "windows")]
    descriptor_texture_handle: std::sync::OnceLock<Option<std::os::windows::io::OwnedHandle>>,
}

// SAFETY: the C handle is reference-counted and its entry points may be called
// from any thread holding a valid reference. `shape_cache` and `identity` are
// owned `Send + Sync` data, and the caches are `OnceLock`s over `Send + Sync`
// contents.
unsafe impl Send for TensorDyn {}
unsafe impl Sync for TensorDyn {}

/// The C handle, released on drop only when this value owns a reference.
struct Handle {
    ptr: NonNull<EfTensor>,
    owned: bool,
}

impl Handle {
    fn as_ptr(&self) -> *mut EfTensor {
        self.ptr.as_ptr()
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        if self.owned {
            // SAFETY: an owned handle carries one reference, taken at construction.
            unsafe { edgefirst_tensor_ffi::ef_tensor_free(self.ptr.as_ptr()) }
        }
    }
}

/// A [`TensorDyn`] borrowed from a raw handle the caller keeps owning,
/// returned by [`TensorDyn::borrow_raw`].
///
/// Dropping it releases the Rust-side caches but never the handle.
pub struct BorrowedTensor<'a> {
    tensor: TensorDyn,
    _handle: std::marker::PhantomData<&'a mut EfTensor>,
}

impl std::ops::Deref for BorrowedTensor<'_> {
    type Target = TensorDyn;
    fn deref(&self) -> &TensorDyn {
        &self.tensor
    }
}

impl std::ops::DerefMut for BorrowedTensor<'_> {
    fn deref_mut(&mut self) -> &mut TensorDyn {
        &mut self.tensor
    }
}

impl fmt::Debug for BorrowedTensor<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&self.tensor, f)
    }
}

/// The raw C handle.
pub type Raw = *mut EfTensor;

impl TensorDyn {
    /// Wrap an already-live handle, deriving the cached facts fresh.
    fn from_handle(handle: NonNull<EfTensor>) -> Self {
        Self::wrap(handle, true)
    }

    /// Wrap a live handle, freeing it on drop only when `owned`.
    fn wrap(handle: NonNull<EfTensor>, owned: bool) -> Self {
        let shape_cache = Self::query_shape(handle);
        let identity = Self::derive_identity(handle);
        TensorDyn {
            handle: Handle { ptr: handle, owned },
            shape_cache,
            identity,
            quantization_cache: std::sync::OnceLock::new(),
            chroma_cache: std::sync::OnceLock::new(),
            #[cfg(target_os = "windows")]
            descriptor_texture_handle: std::sync::OnceLock::new(),
        }
    }

    /// Derive this handle's [`BufferIdentity`].
    ///
    /// On Windows a texture-backed handle is keyed on the `ID3D11Texture2D*`
    /// returned by `ef_tensor_d3d11_texture`. On Unix a DMA-BUF-backed handle
    /// is keyed on the `(st_dev, st_ino)` of its plane 0 fd. Every other
    /// storage kind, and an fd that cannot be `fstat`-ed, is keyed on the
    /// handle's own address as `HostPtr`.
    fn derive_identity(handle: NonNull<EfTensor>) -> BufferIdentity {
        #[cfg(target_os = "windows")]
        {
            // SAFETY: `handle` is live for the duration of this call.
            let code = unsafe { edgefirst_tensor_ffi::ef_tensor_storage_kind(handle.as_ptr()) };
            if TensorMemory::from_code(code) == Some(TensorMemory::DmaBuf) {
                // SAFETY: `handle` is live for the duration of this call; the
                // pointer is borrowed and carries no reference of its own.
                let tex = unsafe { edgefirst_tensor_ffi::ef_tensor_d3d11_texture(handle.as_ptr()) };
                if !tex.is_null() {
                    return BufferIdentity::derived(
                        IdentityKind::D3d11Texture,
                        tex as usize as u64,
                    );
                }
            }
        }
        #[cfg(unix)]
        {
            // SAFETY: `handle` is live for the duration of this call.
            let code = unsafe { edgefirst_tensor_ffi::ef_tensor_storage_kind(handle.as_ptr()) };
            if TensorMemory::from_code(code) == Some(TensorMemory::DmaBuf) {
                let mut plane = edgefirst_tensor_ffi::EfTensorPlane::default();
                // SAFETY: `handle` is live; `plane` is a valid local out-param.
                let rc = unsafe {
                    edgefirst_tensor_ffi::ef_tensor_plane_at(handle.as_ptr(), 0, &mut plane)
                };
                if rc == 0 && plane.handle >= 0 {
                    // SAFETY: `plane.handle` is a valid fd owned by this
                    // handle for at least the duration of this borrow.
                    let fd = unsafe { std::os::fd::BorrowedFd::borrow_raw(plane.handle as i32) };
                    if let Ok(stat) = nix::sys::stat::fstat(fd) {
                        return identity_from_stat(&stat);
                    }
                }
            }
        }
        BufferIdentity::derived(IdentityKind::HostPtr, handle.as_ptr() as u64)
    }

    /// Whether this handle is a two-allocation semi-planar tensor.
    pub fn is_multiplane(&self) -> bool {
        // SAFETY: `self.handle` is live.
        unsafe { edgefirst_tensor_ffi::ef_tensor_is_multiplane(self.handle.as_ptr()) != 0 }
    }

    /// Borrowed lens over the chroma plane, or `None` if this is not
    /// multiplane.
    pub fn chroma(&self) -> Option<&TensorDyn> {
        self.chroma_cache
            .get_or_init(|| {
                // SAFETY: `self.handle` is live. A non-NULL result carries
                // one retained reference that the cached lens owns.
                let p = unsafe { edgefirst_tensor_ffi::ef_tensor_chroma(self.handle.as_ptr()) };
                std::ptr::NonNull::new(p).map(|h| Box::new(unsafe { Self::from_raw(h.as_ptr()) }))
            })
            .as_deref()
    }

    /// Owned chroma plane as its own `TensorDyn`, or `None` if this is not
    /// a two-allocation semi-planar tensor.
    pub fn chroma_dyn(&self) -> Option<TensorDyn> {
        // SAFETY: `self.handle` is live. A non-NULL result carries one
        // retained reference that the returned tensor owns.
        let p = unsafe { edgefirst_tensor_ffi::ef_tensor_chroma(self.handle.as_ptr()) };
        std::ptr::NonNull::new(p).map(|h| unsafe { Self::from_raw(h.as_ptr()) })
    }

    /// Read the handle's current shape via `ef_tensor_ndim`/`ef_tensor_shape`.
    fn query_shape(handle: NonNull<EfTensor>) -> Vec<usize> {
        // SAFETY: `handle` is a live handle for the duration of this call.
        unsafe {
            let ndim = edgefirst_tensor_ffi::ef_tensor_ndim(handle.as_ptr());
            let ptr = edgefirst_tensor_ffi::ef_tensor_shape(handle.as_ptr());
            if ptr.is_null() || ndim == 0 {
                return Vec::new();
            }
            std::slice::from_raw_parts(ptr, ndim as usize)
                .iter()
                .map(|&d| d as usize)
                .collect()
        }
    }

    /// The handle, still owned by this value and valid while it lives.
    ///
    /// For lending the tensor to C code that does not keep it. A borrower
    /// that keeps it takes its own reference with `ef_tensor_retain`; that
    /// reference makes the handle shared, so writable CPU maps of it are
    /// refused until it is released. Changing the tensor's shape or format
    /// through the handle leaves this value's cached copies stale.
    pub fn as_raw(&self) -> Raw {
        self.handle.as_ptr()
    }

    /// A second owner of the same tensor: one more reference to this handle.
    ///
    /// Both values read and write the same buffer and metadata, and the
    /// tensor lives until both are dropped. While both are alive the handle
    /// is shared, so writable CPU maps through either are refused.
    ///
    /// # Safety
    /// The handle's metadata (shape, format, row stride, plane offset,
    /// quantization) is not synchronized between owners. While both are
    /// alive, no metadata change through one may overlap any use of the
    /// other, and neither may rely on the other leaving the metadata alone:
    /// each keeps its own cached copy.
    ///
    /// # Errors
    /// `NotImplemented` when the library refuses the reference.
    pub unsafe fn share(&self) -> Result<TensorDyn> {
        // SAFETY: `self.handle` is live.
        let rc = unsafe { edgefirst_tensor_ffi::ef_tensor_retain(self.handle.as_ptr()) };
        if rc != 0 {
            return Err(Error::NotImplemented(format!(
                "ef_tensor_retain failed: errno {rc}"
            )));
        }
        // SAFETY: the reference taken above is owned by the new value.
        Ok(unsafe { Self::from_raw(self.handle.as_ptr()) })
    }

    /// Give up ownership, yielding the handle. The caches are dropped.
    pub fn into_raw(self) -> Raw {
        let TensorDyn { mut handle, .. } = self;
        handle.owned = false;
        handle.as_ptr()
    }

    /// # Safety
    /// `p` must be a live handle carrying a reference the caller owns.
    pub unsafe fn from_raw(p: Raw) -> TensorDyn {
        Self::from_handle(unsafe { NonNull::new_unchecked(p) })
    }

    /// Borrow a `TensorDyn` from a raw handle the caller keeps owning.
    ///
    /// The guard holds no reference of its own, so a writable map through it
    /// is not refused as shared; dropping it releases only its caches.
    ///
    /// # Safety
    /// `p` must be a live handle from an EdgeFirst library that outlives
    /// `'a`, and the `TensorDyn` must not be moved out of the guard.
    pub unsafe fn borrow_raw<'a>(p: Raw) -> BorrowedTensor<'a> {
        BorrowedTensor {
            tensor: Self::wrap(unsafe { NonNull::new_unchecked(p) }, false),
            _handle: std::marker::PhantomData,
        }
    }

    /// Borrow a `TensorDyn` from a raw handle for the duration of `f`.
    ///
    /// # Safety
    /// Same as [`Self::borrow_raw`].
    pub unsafe fn with_raw<R>(p: Raw, f: impl FnOnce(&mut TensorDyn) -> R) -> R {
        let mut t = unsafe { Self::borrow_raw(p) };
        f(&mut t)
    }

    // --- Primitives: each is a direct `ef_tensor_*` call.

    /// Create a type-erased tensor with the given shape and element type.
    ///
    /// Allocates through the builder
    /// (`ef_tensor_builder_{new,dtype,shape,storage,alloc}`); `memory` selects
    /// the backing. `name` has no effect.
    pub fn new(
        shape: &[usize],
        dtype: DType,
        memory: Option<TensorMemory>,
        _name: Option<&str>,
    ) -> Result<Self> {
        // SAFETY: every call below either takes a value or a pointer to data
        // this function owns for the call's duration.
        unsafe {
            let b = edgefirst_tensor_ffi::ef_tensor_builder_new();
            if b.is_null() {
                return Err(Error::NotImplemented(
                    "ef_tensor_builder_new: allocation failed".into(),
                ));
            }
            let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
            edgefirst_tensor_ffi::ef_tensor_builder_dtype(b, dtype.code());
            edgefirst_tensor_ffi::ef_tensor_builder_shape(b, dims.as_ptr(), dims.len() as u32);
            if let Some(mem) = memory {
                edgefirst_tensor_ffi::ef_tensor_builder_storage(b, mem.code());
            }
            let handle = edgefirst_tensor_ffi::ef_tensor_builder_alloc(b);
            let result = match NonNull::new(handle) {
                Some(h) => Ok(Self::from_handle(h)),
                None => Err(builder_error(
                    edgefirst_tensor_ffi::ef_tensor_builder_error(b),
                )),
            };
            edgefirst_tensor_ffi::ef_tensor_builder_free(b);
            result
        }
    }

    /// Import an existing buffer as a type-erased tensor, taking ownership of
    /// its file descriptor. No bytes are copied.
    ///
    /// Adopts the fd through the builder
    /// (`ef_tensor_builder_{new,dtype,shape,add_plane,wrap}`).
    #[cfg(unix)]
    pub fn from_fd(
        fd: std::os::fd::OwnedFd,
        shape: &[usize],
        dtype: DType,
        _name: Option<&str>,
    ) -> Result<Self> {
        use std::os::fd::{FromRawFd, IntoRawFd};
        // SAFETY: every call below either takes a value or a pointer to data
        // this function owns for the call's duration; `raw_fd` is reclaimed
        // into an `OwnedFd` on every path that does not hand it to `wrap`.
        unsafe {
            let b = edgefirst_tensor_ffi::ef_tensor_builder_new();
            if b.is_null() {
                return Err(Error::NotImplemented(
                    "ef_tensor_builder_new: allocation failed".into(),
                ));
            }
            let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
            edgefirst_tensor_ffi::ef_tensor_builder_dtype(b, dtype.code());
            edgefirst_tensor_ffi::ef_tensor_builder_shape(b, dims.as_ptr(), dims.len() as u32);
            let raw_fd = fd.into_raw_fd();
            edgefirst_tensor_ffi::ef_tensor_builder_add_plane(b, raw_fd as i64, 0, 0, 0, 0, 0);
            let handle = edgefirst_tensor_ffi::ef_tensor_builder_wrap(b);
            let result = match NonNull::new(handle) {
                Some(h) => Ok(Self::from_handle(h)),
                None => {
                    // Close the fd `wrap` did not adopt.
                    drop(std::os::fd::OwnedFd::from_raw_fd(raw_fd));
                    let errno = edgefirst_tensor_ffi::ef_tensor_builder_error(b);
                    Err(Error::NotImplemented(format!(
                        "ef_tensor_builder_wrap failed: errno {errno}"
                    )))
                }
            };
            edgefirst_tensor_ffi::ef_tensor_builder_free(b);
            result
        }
    }

    /// Return the tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape_cache
    }

    /// Reshape this tensor. Total element count must remain the same.
    ///
    /// Drives `ef_tensor_reshape`.
    pub fn reshape(&mut self, shape: &[usize]) -> Result<()> {
        self.reshape_impl(shape, false)
    }

    /// Set the logical shape to any shape whose bytes fit the allocation,
    /// without [`reshape`](Self::reshape)'s equal-count constraint. Drives
    /// `ef_tensor_set_logical_shape`.
    pub fn set_logical_shape(&mut self, shape: &[usize]) -> Result<()> {
        self.reshape_impl(shape, true)
    }

    /// The shared body of [`reshape`](Self::reshape) and
    /// [`set_logical_shape`](Self::set_logical_shape): marshals the dims and
    /// refreshes the shape cache.
    fn reshape_impl(&mut self, shape: &[usize], by_capacity: bool) -> Result<()> {
        let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        // SAFETY: `self.handle` is live; `dims` is a live local for the call.
        let rc = unsafe {
            if by_capacity {
                edgefirst_tensor_ffi::ef_tensor_set_logical_shape(
                    self.handle.as_ptr(),
                    dims.as_ptr(),
                    dims.len() as u32,
                )
            } else {
                edgefirst_tensor_ffi::ef_tensor_reshape(
                    self.handle.as_ptr(),
                    dims.as_ptr(),
                    dims.len() as u32,
                )
            }
        };
        if rc != 0 {
            let msg = ffi_last_error();
            return Err(if rc == libc::ERANGE {
                Error::ShapeMismatch(msg)
            } else {
                Error::InvalidShape(msg)
            });
        }
        self.shape_cache = Self::query_shape(self.handle.ptr);
        Ok(())
    }

    /// Return the tensor name.
    ///
    /// Always empty on this backend.
    pub fn name(&self) -> String {
        String::new()
    }

    /// Return the memory allocation type reported by `ef_tensor_storage_kind`.
    pub fn memory(&self) -> TensorMemory {
        // SAFETY: `self.handle` is a live handle for as long as `self` exists.
        let code = unsafe { edgefirst_tensor_ffi::ef_tensor_storage_kind(self.handle.as_ptr()) };
        TensorMemory::from_code(code).unwrap_or(TensorMemory::Mem)
    }

    /// Clone the file descriptor backing this tensor, for any storage kind that
    /// has one. Drives `ef_tensor_clone_fd`.
    #[cfg(unix)]
    pub fn clone_fd(&self) -> Result<std::os::fd::OwnedFd> {
        use std::os::fd::FromRawFd;
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let rc = unsafe { edgefirst_tensor_ffi::ef_tensor_clone_fd(self.handle.as_ptr()) };
        if rc < 0 {
            let msg = ffi_last_error();
            return Err(if -rc == libc::ENOTSUP {
                Error::NotImplemented(msg)
            } else {
                Error::IoError(std::io::Error::other(format!(
                    "clone_fd: {msg} (errno {})",
                    -rc
                )))
            });
        }
        // SAFETY: `ef_tensor_clone_fd` returned a freshly-`dup`'d fd whose
        // ownership it handed to this caller.
        Ok(unsafe { std::os::fd::OwnedFd::from_raw_fd(rc) })
    }

    /// Return the [`BufferIdentity`] of the underlying allocation, derived once
    /// at construction.
    pub fn buffer_identity(&self) -> &BufferIdentity {
        &self.identity
    }

    /// The shared core of [`map_bytes`](Self::map_bytes) and
    /// [`pin_host`](Self::pin_host): `ef_tensor_map` plus a retained
    /// [`MapKeepalive`], giving a `'static` pin that shares ownership of the
    /// handle.
    pub(crate) fn map_pin(&self, access: CpuAccess) -> Result<crate::pin::HostPin<'static>> {
        self.map_pin_with(access, false)
    }

    /// [`map_pin`](Self::map_pin), with the choice of `ef_tensor_map` or
    /// `ef_tensor_try_map`.
    pub(crate) fn map_pin_with(
        &self,
        access: CpuAccess,
        non_blocking: bool,
    ) -> Result<crate::pin::HostPin<'static>> {
        self.cpu_access().require_direction(access, "map_with")?;
        let code = access.code();
        let mut view = edgefirst_tensor_ffi::EfTensorView {
            ptr: std::ptr::null_mut(),
            len: 0,
        };
        // SAFETY: `self.handle` is live; `view` is a valid local out-param.
        let rc = unsafe {
            if non_blocking {
                edgefirst_tensor_ffi::ef_tensor_try_map(self.handle.as_ptr(), code, &mut view)
            } else {
                edgefirst_tensor_ffi::ef_tensor_map(self.handle.as_ptr(), code, &mut view)
            }
        };
        if rc == libc::EAGAIN && non_blocking {
            // The GPU copy this map depends on is in flight: report
            // `WouldBlock`.
            return Err(Error::IoError(std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                ffi_last_error(),
            )));
        }
        if rc != 0 {
            let what = if non_blocking {
                "ef_tensor_try_map"
            } else {
                "ef_tensor_map"
            };
            return Err(Error::InvalidOperation(format!(
                "{what} failed: errno {rc}"
            )));
        }
        // Take the reference `MapKeepalive::drop` releases.
        // SAFETY: `self.handle` is live.
        let retain_rc = unsafe { edgefirst_tensor_ffi::ef_tensor_retain(self.handle.as_ptr()) };
        if retain_rc != 0 {
            // SAFETY: undoes the successful map above.
            unsafe { edgefirst_tensor_ffi::ef_tensor_unmap(self.handle.as_ptr()) };
            return Err(Error::NotImplemented(format!(
                "ef_tensor_retain failed: errno {retain_rc}"
            )));
        }
        let keepalive: std::sync::Arc<dyn Send + Sync> =
            std::sync::Arc::new(MapKeepalive(self.handle.as_ptr()));
        Ok(crate::pin::HostPin::new(keepalive, view.ptr, view.len))
    }

    /// Map this tensor's whole extent for CPU access, type-erased to raw bytes.
    /// Returns a `'static` view.
    pub fn map_bytes(&self, access: CpuAccess) -> Result<crate::view::HostView<'static, u8>> {
        let pin = self.map_pin(access)?;
        let len = pin.len();
        Ok(crate::view::HostView::new(pin, vec![len], None, access))
    }

    /// Non-blocking [`map_bytes`](Self::map_bytes).
    ///
    /// Drives `ef_tensor_try_map`, whose `EAGAIN` becomes
    /// `Error::IoError(WouldBlock)`. Only a Windows D3D11 texture can return
    /// it; every other backing behaves as [`map_bytes`](Self::map_bytes).
    pub fn try_map_bytes(&self, access: CpuAccess) -> Result<crate::view::HostView<'static, u8>> {
        let pin = self.map_pin_with(access, true)?;
        let len = pin.len();
        Ok(crate::view::HostView::new(pin, vec![len], None, access))
    }

    /// Pin a stable host address for this tensor's data. Returns a `'static`
    /// pin built on [`map_pin`](Self::map_pin).
    ///
    /// # Errors
    ///
    /// [`Error::NotImplemented`] with [`crate::pbo::PIN_HOST_PBO_REFUSAL`] for
    /// a PBO-backed tensor.
    pub fn pin_host(&self, access: CpuAccess) -> Result<crate::pin::HostPin<'static>> {
        if self.memory() == TensorMemory::Pbo {
            return Err(Error::NotImplemented(
                crate::pbo::PIN_HOST_PBO_REFUSAL.to_owned(),
            ));
        }
        self.map_pin(access)
    }

    /// This tensor's plane-0 geometry and native handle (`ef_tensor_plane_at`),
    /// or `None` for an invalid handle or one with no planes.
    pub fn plane0(&self) -> Option<edgefirst_tensor_ffi::EfTensorPlane> {
        let mut plane = edgefirst_tensor_ffi::EfTensorPlane::default();
        // SAFETY: `self.handle` is live; `plane` is a valid local out-param.
        let rc = unsafe {
            edgefirst_tensor_ffi::ef_tensor_plane_at(self.handle.as_ptr(), 0, &mut plane)
        };
        (rc == 0).then_some(plane)
    }

    /// Effective row stride: the padded byte pitch a stride-aware CPU reader
    /// must honor, taken from plane 0's stride.
    pub fn effective_row_stride(&self) -> Option<usize> {
        self.plane0().map(|p| p.stride as usize)
    }

    /// Image width in pixels (`None` if not an image tensor), computed from
    /// `shape()` and `format()`.
    pub fn width(&self) -> Option<usize> {
        let (fmt, dims) = self.image_dims()?;
        match fmt.layout() {
            PixelLayout::Packed => dims.get(1).copied(),
            PixelLayout::Planar => dims.get(2).copied(),
            PixelLayout::SemiPlanar => dims.get(1).copied(),
        }
    }

    /// The format and the trailing shape dims that describe one image, allowing
    /// one leading batch `N`; `None` when unformatted or when the shape has any
    /// other rank.
    fn image_dims(&self) -> Option<(PixelFormat, &[usize])> {
        let fmt = self.format()?;
        let shape = self.shape();
        let rank = match fmt.layout() {
            PixelLayout::SemiPlanar => 2,
            PixelLayout::Packed | PixelLayout::Planar => 3,
        };
        let batch_dims = shape.len().checked_sub(rank).filter(|&n| n <= 1)?;
        Some((fmt, &shape[batch_dims..]))
    }

    /// Image height in pixels (`None` if not an image tensor), computed from
    /// `shape()` and `format()`.
    pub fn height(&self) -> Option<usize> {
        let (fmt, dims) = self.image_dims()?;
        match fmt.layout() {
            PixelLayout::Packed => dims.first().copied(),
            PixelLayout::Planar => dims.get(1).copied(),
            PixelLayout::SemiPlanar => {
                if self.is_multiplane() {
                    dims.first().copied()
                } else {
                    match fmt {
                        PixelFormat::Nv12 => dims.first().map(|h| h * 2 / 3),
                        PixelFormat::Nv16 => dims.first().map(|h| h / 2),
                        PixelFormat::Nv24 => dims.first().map(|h| h / 3),
                        _ => None,
                    }
                }
            }
        }
    }

    /// True if `self` and `other` reference the same underlying buffer:
    /// identity equality, then (Linux DMA tensors only) a same-fd-number
    /// compare via plane 0's native handle.
    pub fn aliases(&self, other: &Self) -> bool {
        if self.buffer_identity().id() == other.buffer_identity().id() {
            return true;
        }
        if self.memory() != other.memory() {
            return false;
        }
        if self.memory() == TensorMemory::DmaBuf {
            if let (Some(a), Some(b)) = (self.plane0(), other.plane0()) {
                if a.handle >= 0 && a.handle == b.handle {
                    return true;
                }
            }
        }
        false
    }

    // --- `colorimetry`, `set_colorimetry`, `view_origin`.

    /// Colorimetry metadata. `None` = undefined; an image convert tags an untagged YUV destination.
    pub fn colorimetry(&self) -> Option<Colorimetry> {
        // SAFETY: `self.handle` is live.
        let packed = unsafe { edgefirst_tensor_ffi::ef_tensor_colorimetry(self.handle.as_ptr()) };
        if packed == 0 {
            None
        } else {
            Some(Colorimetry::unpack(packed))
        }
    }

    /// Attach/clear colorimetry metadata.
    pub fn set_colorimetry(&mut self, c: Option<Colorimetry>) {
        let packed = c.map(|c| c.pack()).unwrap_or(0);
        // SAFETY: `self.handle` is live and was minted by this library.
        unsafe { edgefirst_tensor_ffi::ef_tensor_set_colorimetry(self.handle.as_ptr(), packed) };
    }

    /// Builder-style [`Self::set_colorimetry`].
    pub fn with_colorimetry(mut self, c: Colorimetry) -> Self {
        self.set_colorimetry(Some(c));
        self
    }

    /// The parent-image snapshot if this tensor is a `view`/`batch`
    /// sub-region; `None` for a whole tensor. See [`ViewOrigin`].
    pub fn view_origin(&self) -> Option<ViewOrigin> {
        let mut out = edgefirst_tensor_ffi::EfViewOrigin::default();
        // SAFETY: `self.handle` is live; `out` is a valid local out-param.
        let rc =
            unsafe { edgefirst_tensor_ffi::ef_tensor_view_origin(self.handle.as_ptr(), &mut out) };
        if rc != 0 || out.has_origin == 0 {
            return None;
        }
        Some(ViewOrigin {
            parent_width: out.parent_width as usize,
            parent_height: out.parent_height as usize,
            parent_row_stride: out.parent_row_stride as usize,
            x: out.x as usize,
            y: out.y as usize,
        })
    }

    /// Whether this tensor addresses part of a larger buffer (see
    /// [`crate::Tensor::is_view`]). Drives `ef_tensor_is_view`.
    pub fn is_view(&self) -> bool {
        // SAFETY: `self.handle` is live for as long as `self` exists.
        unsafe { edgefirst_tensor_ffi::ef_tensor_is_view(self.handle.as_ptr()) != 0 }
    }

    /// Record that this tensor addresses part of a larger buffer (see
    /// [`crate::Tensor::mark_view`]). Drives `ef_tensor_mark_view`.
    pub fn mark_view(&mut self) {
        // SAFETY: `self.handle` is live; a valid handle cannot be refused.
        let _ = unsafe { edgefirst_tensor_ffi::ef_tensor_mark_view(self.handle.as_ptr()) };
    }

    /// Return the runtime element type discriminant (`ef_tensor_dtype`).
    pub fn dtype(&self) -> DType {
        // SAFETY: `self.handle` is live.
        let code = unsafe { edgefirst_tensor_ffi::ef_tensor_dtype(self.handle.as_ptr()) };
        DType::from_code(code).unwrap_or(DType::U8)
    }

    /// Return the pixel format (`None` if not an image tensor), read from
    /// `ef_tensor_format`.
    pub fn format(&self) -> Option<PixelFormat> {
        // SAFETY: `self.handle` is live; the returned pointer is borrowed
        // for as long as the handle lives, which outlives this call.
        let ptr = unsafe { edgefirst_tensor_ffi::ef_tensor_format(self.handle.as_ptr()) };
        if ptr.is_null() {
            return None;
        }
        // SAFETY: `ef_tensor_format` returns a NUL-terminated C string.
        let s = unsafe { std::ffi::CStr::from_ptr(ptr) }.to_str().ok()?;
        if s.is_empty() {
            None
        } else {
            PixelFormat::from_str_code(s)
        }
    }

    /// Allocate an image tensor with the given geometry, memory backing, and
    /// CPU access declaration. Drives `ef_tensor_image_alloc`.
    pub fn image(
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        memory: Option<TensorMemory>,
        access: CpuAccess,
    ) -> Result<Self> {
        let c_format = format_cstring(format)?;
        let (has_memory, memory_code) = memory_code(memory);
        // SAFETY: `c_format` is a valid NUL-terminated string for the
        // call's duration.
        let handle = unsafe {
            edgefirst_tensor_ffi::ef_tensor_image_alloc(
                width,
                height,
                c_format.as_ptr(),
                dtype.code(),
                has_memory,
                memory_code,
                access.code(),
            )
        };
        match NonNull::new(handle) {
            Some(h) => Ok(Self::from_handle(h)),
            // The library's recorded error class, or `NotImplemented` when it
            // recorded none.
            None => Err(ffi_error(Error::NotImplemented)),
        }
    }

    /// Allocate an image tensor from a full [`crate::ImageDesc`] request,
    /// including any compression negotiation. Drives
    /// `ef_tensor_image_desc_alloc`.
    pub fn image_desc(desc: &crate::ImageDesc) -> Result<Self> {
        let c_format = format_cstring(desc.format())?;
        // SAFETY: every call below either takes a value or a pointer to
        // data this function owns for the call's duration.
        unsafe {
            let d = edgefirst_tensor_ffi::ef_tensor_image_desc_new(
                desc.width(),
                desc.height(),
                c_format.as_ptr(),
                desc.dtype().code(),
            );
            if d.is_null() {
                return Err(Error::NotImplemented(
                    "ef_tensor_image_desc_new: allocation failed".into(),
                ));
            }
            if let Some(m) = desc.memory() {
                edgefirst_tensor_ffi::ef_tensor_image_desc_set_memory(d, m.code());
            }
            edgefirst_tensor_ffi::ef_tensor_image_desc_set_access(d, desc.access().code());
            if desc.compression().is_some() {
                // `Compression::Any` and `Compression::Scheme(_)` both request
                // any scheme.
                edgefirst_tensor_ffi::ef_tensor_image_desc_set_compression(
                    d,
                    edgefirst_tensor_ffi::EfCompressionRequest::Any as u32,
                );
            }
            if desc.contiguous() {
                edgefirst_tensor_ffi::ef_tensor_image_desc_set_contiguous(d, 1);
            }
            let handle = edgefirst_tensor_ffi::ef_tensor_image_desc_alloc(d);
            let result = match NonNull::new(handle) {
                Some(h) => Ok(Self::from_handle(h)),
                None => Err(ffi_error(Error::NotImplemented)),
            };
            edgefirst_tensor_ffi::ef_tensor_image_desc_free(d);
            result
        }
    }

    /// Allocate an image tensor with a caller-supplied row stride. Drives
    /// `ef_tensor_image_with_stride_alloc`.
    pub fn image_with_stride(
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        row_stride_bytes: usize,
        memory: Option<TensorMemory>,
        access: CpuAccess,
    ) -> Result<Self> {
        let c_format = format_cstring(format)?;
        let (has_memory, memory_code) = memory_code(memory);
        // SAFETY: `c_format` is a valid NUL-terminated string for the
        // call's duration.
        let handle = unsafe {
            edgefirst_tensor_ffi::ef_tensor_image_with_stride_alloc(
                width,
                height,
                c_format.as_ptr(),
                dtype.code(),
                row_stride_bytes,
                has_memory,
                memory_code,
                access.code(),
            )
        };
        match NonNull::new(handle) {
            Some(h) => Ok(Self::from_handle(h)),
            None => Err(ffi_error(Error::NotImplemented)),
        }
    }

    /// Borrow a rectangular spatial sub-region as a zero-copy view sharing this
    /// tensor's allocation and [`BufferIdentity`]. Drives
    /// `ef_tensor_view_region`.
    pub fn view(&self, region: Region) -> Result<TensorDyn> {
        // SAFETY: `self.handle` is a live handle for the call's duration.
        let handle = unsafe {
            edgefirst_tensor_ffi::ef_tensor_view_region(
                self.handle.as_ptr(),
                region.x as u64,
                region.y as u64,
                region.width as u64,
                region.height as u64,
            )
        };
        match NonNull::new(handle) {
            Some(h) => Ok(Self::from_handle(h)),
            None => {
                if ffi_last_error_class() == edgefirst_tensor_ffi::EfErrorClass::RegionOutOfBounds {
                    Err(Error::RegionOutOfBounds {
                        region,
                        bounds: (self.width().unwrap_or(0), self.height().unwrap_or(0)),
                    })
                } else {
                    Err(ffi_error(Error::InvalidShape))
                }
            }
        }
    }

    /// Attach pixel format metadata to this tensor, validating shape
    /// compatibility. Drives `ef_tensor_set_format`.
    ///
    /// Must not be called while another thread has a call in flight on the same
    /// underlying handle.
    pub fn set_format(&mut self, format: PixelFormat) -> Result<()> {
        let c_format = format_cstring(format)?;
        // SAFETY: `self.handle` is a live handle; `c_format` is valid for
        // the call's duration.
        let rc = unsafe {
            edgefirst_tensor_ffi::ef_tensor_set_format(self.handle.as_ptr(), c_format.as_ptr())
        };
        if rc != 0 {
            return Err(Error::InvalidArgument(format!(
                "ef_tensor_set_format failed: {}",
                ffi_last_error()
            )));
        }
        Ok(())
    }

    /// Builder-style variant of [`Self::set_format`].
    pub fn with_format(mut self, format: PixelFormat) -> Result<Self> {
        self.set_format(format)?;
        Ok(self)
    }

    /// Set the row stride in bytes. The format must be set first. Drives
    /// `ef_tensor_set_row_stride`. Same concurrency rule as
    /// [`Self::set_format`].
    pub fn set_row_stride(&mut self, stride: usize) -> Result<()> {
        // SAFETY: `self.handle` is a live handle.
        let rc =
            unsafe { edgefirst_tensor_ffi::ef_tensor_set_row_stride(self.handle.as_ptr(), stride) };
        if rc != 0 {
            return Err(Error::InvalidArgument(format!(
                "ef_tensor_set_row_stride failed: {}",
                ffi_last_error()
            )));
        }
        Ok(())
    }

    /// Set the row stride in bytes without format validation. Drives
    /// `ef_tensor_set_row_stride_unchecked`. Unlike [`Self::set_row_stride`],
    /// needs no format on this tensor first. Same concurrency rule as
    /// [`Self::set_format`].
    pub fn set_row_stride_unchecked(&mut self, stride: usize) {
        // SAFETY: `self.handle` is a live handle.
        unsafe {
            edgefirst_tensor_ffi::ef_tensor_set_row_stride_unchecked(self.handle.as_ptr(), stride)
        };
    }

    /// Retag this tensor's element type, keeping its bytes untouched. Drives
    /// `ef_tensor_set_dtype`.
    pub fn set_dtype(&mut self, dtype: DType) -> Result<()> {
        if self.dtype() == dtype {
            return Ok(());
        }
        // SAFETY: `self.handle` is live and own-mint.
        let rc = unsafe {
            edgefirst_tensor_ffi::ef_tensor_set_dtype(self.handle.as_ptr(), dtype.code())
        };
        if rc == 0 {
            return Ok(());
        }
        Err(Error::InvalidArgument(format!(
            "ef_tensor_set_dtype failed: {} (errno {rc})",
            ffi_last_error()
        )))
    }

    /// Set the byte offset within the DMA-BUF where image data starts. Drives
    /// `ef_tensor_set_plane_offset`. Same concurrency rule as
    /// [`Self::set_format`].
    pub fn set_plane_offset(&mut self, offset: usize) {
        // SAFETY: `self.handle` is a live handle.
        unsafe { edgefirst_tensor_ffi::ef_tensor_set_plane_offset(self.handle.as_ptr(), offset) };
    }

    /// Byte offset within the DMA-BUF where image data starts (`None` = 0).
    /// Drives `ef_tensor_plane_offset`.
    pub fn plane_offset(&self) -> Option<usize> {
        // SAFETY: `self.handle` is live.
        let raw = unsafe { edgefirst_tensor_ffi::ef_tensor_plane_offset(self.handle.as_ptr()) };
        (raw >= 0).then_some(raw as usize)
    }

    /// Set this tensor's logical dimensions and pixel format to a decoded
    /// image, reusing the existing allocation. Drives
    /// `ef_tensor_configure_image`. Same concurrency rule as
    /// [`Self::set_format`].
    pub fn configure_image(
        &mut self,
        width: usize,
        height: usize,
        format: PixelFormat,
    ) -> Result<()> {
        let c_format = format_cstring(format)?;
        // SAFETY: `self.handle` is a live handle; `c_format` is valid for
        // the call's duration.
        let rc = unsafe {
            edgefirst_tensor_ffi::ef_tensor_configure_image(
                self.handle.as_ptr(),
                width,
                height,
                c_format.as_ptr(),
            )
        };
        if rc != 0 {
            return Err(Error::InvalidArgument(format!(
                "ef_tensor_configure_image failed: {}",
                ffi_last_error()
            )));
        }
        self.shape_cache = Self::query_shape(self.handle.ptr);
        Ok(())
    }

    // --- Quantization.

    /// Quantization metadata for this tensor (`None` for a tensor that carries
    /// none, or a float-dtype tensor). Read through
    /// `ef_tensor_quantization_info`/`ef_tensor_quantization_get` on the first
    /// call and cached in [`Self::quantization_cache`].
    pub fn quantization(&self) -> Option<&crate::Quantization> {
        self.quantization_cache
            .get_or_init(|| self.fetch_quantization())
            .as_ref()
    }

    /// Fetch the quantization through `ef_tensor_quantization_{info,get}`,
    /// reconstructing the mode the producer attached (symmetric or not,
    /// per-tensor or per-channel). An all-zero `zero_point` array reads back as
    /// symmetric.
    fn fetch_quantization(&self) -> Option<crate::Quantization> {
        let mut info = edgefirst_tensor_ffi::EfQuantizationInfo::default();
        // SAFETY: `self.handle` is live; `info` is a valid local out-param.
        let rc = unsafe {
            edgefirst_tensor_ffi::ef_tensor_quantization_info(self.handle.as_ptr(), &mut info)
        };
        if rc != 0 || info.has_quantization == 0 {
            return None;
        }
        let n = info.count as usize;
        let mut scales = vec![0f32; n];
        let mut zps = vec![0i32; n];
        // SAFETY: `self.handle` is live; `scales`/`zps` are sized to `n`,
        // the count `ef_tensor_quantization_info` just reported.
        let rc = unsafe {
            edgefirst_tensor_ffi::ef_tensor_quantization_get(
                self.handle.as_ptr(),
                scales.as_mut_ptr(),
                zps.as_mut_ptr(),
                n as u32,
            )
        };
        if rc != 0 {
            return None;
        }
        let symmetric = zps.iter().all(|&z| z == 0);
        if info.axis == -1 {
            Some(if symmetric {
                crate::Quantization::per_tensor_symmetric(scales[0])
            } else {
                crate::Quantization::per_tensor(scales[0], zps[0])
            })
        } else {
            let axis = info.axis as usize;
            let q = if symmetric {
                crate::Quantization::per_channel_symmetric(scales, axis)
            } else {
                crate::Quantization::per_channel(scales, zps, axis)
            };
            q.ok()
        }
    }

    /// Attach quantization metadata to this tensor. Drives
    /// `ef_tensor_quantization_set`, then refreshes
    /// [`Self::quantization_cache`].
    ///
    /// # Errors
    ///
    /// [`Error::InvalidArgument`] if the underlying call fails, including for a
    /// float-dtype tensor.
    pub fn set_quantization(&mut self, q: crate::Quantization) -> Result<()> {
        let axis = q.axis().map(|a| a as i32).unwrap_or(-1);
        let scales = q.scale();
        let n = scales.len() as u32;
        let zps_owned = q.zero_point().map(|z| z.to_vec());
        let zps_ptr = zps_owned
            .as_ref()
            .map(|z| z.as_ptr())
            .unwrap_or(std::ptr::null());
        // SAFETY: `self.handle` is live; `scales` points to `n` elements;
        // `zps_ptr` is either null or points to `n` elements from the same
        // `Quantization`.
        let rc = unsafe {
            edgefirst_tensor_ffi::ef_tensor_quantization_set(
                self.handle.as_ptr(),
                axis,
                scales.as_ptr(),
                zps_ptr,
                n,
            )
        };
        if rc != 0 {
            return Err(Error::InvalidArgument(format!(
                "ef_tensor_quantization_set failed: {}",
                ffi_last_error()
            )));
        }
        self.quantization_cache = std::sync::OnceLock::new();
        let _ = self.quantization_cache.set(Some(q));
        Ok(())
    }

    /// Builder-style variant of [`Self::set_quantization`].
    pub fn with_quantization(mut self, q: crate::Quantization) -> Result<Self> {
        self.set_quantization(q)?;
        Ok(self)
    }

    /// Clear any quantization metadata on this tensor. Drives
    /// `ef_tensor_quantization_clear`, then resets
    /// [`Self::quantization_cache`].
    pub fn clear_quantization(&mut self) {
        // SAFETY: `self.handle` is live.
        unsafe { edgefirst_tensor_ffi::ef_tensor_quantization_clear(self.handle.as_ptr()) };
        self.quantization_cache = std::sync::OnceLock::new();
        let _ = self.quantization_cache.set(None);
    }

    // --- The `edgefirst-python-common` surface.

    /// Total logical size of this tensor in bytes: the shape's element count
    /// times the element size. The allocation may be larger; see
    /// [`Self::capacity_bytes`].
    pub fn size(&self) -> usize {
        self.shape().iter().product::<usize>() * self.dtype().size()
    }

    /// Bytes of the underlying allocation, which is `>=` [`Self::size`]. Drives
    /// `ef_tensor_capacity_bytes`.
    pub fn capacity_bytes(&self) -> usize {
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let n = unsafe { edgefirst_tensor_ffi::ef_tensor_capacity_bytes(self.handle.as_ptr()) };
        // A negative result (invalid handle) falls back to the logical size.
        usize::try_from(n).unwrap_or_else(|_| self.size())
    }

    /// The recorded row stride in bytes; `None` when the tensor is tightly
    /// packed.
    ///
    /// Drives `ef_tensor_row_stride`. Unlike [`Self::effective_row_stride`], it
    /// computes no pitch when none is recorded.
    pub fn row_stride(&self) -> Option<usize> {
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let n = unsafe { edgefirst_tensor_ffi::ef_tensor_row_stride(self.handle.as_ptr()) };
        usize::try_from(n).ok()
    }

    /// The vendor tile-compression scheme recorded at allocation; `None` for a
    /// linear layout.
    ///
    /// Drives `ef_tensor_compression`. An unrecognised code is logged and
    /// reported as `None`.
    pub fn compression(&self) -> Option<crate::CompressionScheme> {
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let code = unsafe { edgefirst_tensor_ffi::ef_tensor_compression(self.handle.as_ptr()) };
        // Code 0 is the linear layout and is not logged.
        let scheme = crate::CompressionScheme::from_code(code);
        if scheme.is_none() && code != edgefirst_tensor_ffi::EfCompression::None as u32 {
            log::warn!(
                "ef_tensor_compression returned scheme code {code}, which this build of \
                 edgefirst-tensor does not recognise; reporting a linear layout. The \
                 loaded libedgefirst_tensor.so is newer than this consumer."
            );
        }
        scheme
    }

    /// The CPU access declared at allocation (see
    /// [`crate::Tensor::cpu_access`]). Drives `ef_tensor_cpu_access`; a code
    /// this build does not recognise is `None`.
    pub fn cpu_access(&self) -> CpuAccess {
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let code = unsafe { edgefirst_tensor_ffi::ef_tensor_cpu_access(self.handle.as_ptr()) };
        CpuAccess::from_code(code).unwrap_or(CpuAccess::None)
    }

    /// Narrow the declared CPU access (see
    /// [`crate::Tensor::restrict_cpu_access`]). Drives
    /// `ef_tensor_restrict_cpu_access`.
    pub fn restrict_cpu_access(&mut self, access: CpuAccess) {
        // SAFETY: `self.handle` is live; a valid handle and code cannot be refused.
        let _ = unsafe {
            edgefirst_tensor_ffi::ef_tensor_restrict_cpu_access(self.handle.as_ptr(), access.code())
        };
    }

    /// Whether the memory is known to be physically contiguous (see
    /// [`crate::Tensor::contiguity`]). Drives `ef_tensor_contiguity`; a code
    /// this build does not recognise is `Unknown`.
    pub fn contiguity(&self) -> crate::Contiguity {
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let code = unsafe { edgefirst_tensor_ffi::ef_tensor_contiguity(self.handle.as_ptr()) };
        crate::Contiguity::from_code(code).unwrap_or(crate::Contiguity::Unknown)
    }

    /// Acquire the buffer for CPU access. See [`crate::Tensor::sync_for_cpu`].
    /// Drives `ef_tensor_sync_for_cpu`.
    pub fn sync_for_cpu(&self, access: CpuAccess) -> Result<()> {
        self.sync(access, true)
    }

    /// Release the buffer back to the device. See
    /// [`crate::Tensor::sync_for_device`].
    ///
    /// Drives `ef_tensor_sync_for_device`; `access` must match the one that
    /// opened the bracket.
    pub fn sync_for_device(&self, access: CpuAccess) -> Result<()> {
        self.sync(access, false)
    }

    /// Shared body of the two sync brackets: translates the access code and the
    /// errno.
    fn sync(&self, access: CpuAccess, to_cpu: bool) -> Result<()> {
        let what = if to_cpu {
            "sync_for_cpu"
        } else {
            "sync_for_device"
        };
        self.cpu_access().require_direction(access, what)?;
        let code = access.code();
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let rc = unsafe {
            if to_cpu {
                edgefirst_tensor_ffi::ef_tensor_sync_for_cpu(self.handle.as_ptr(), code)
            } else {
                edgefirst_tensor_ffi::ef_tensor_sync_for_device(self.handle.as_ptr(), code)
            }
        };
        if rc == 0 {
            return Ok(());
        }
        let msg = ffi_last_error();
        if rc == libc::ENOTSUP {
            return Err(Error::NotImplemented(msg));
        }
        Err(Error::InvalidOperation(format!(
            "{what}: {msg} (errno {rc})"
        )))
    }

    /// Borrow batch element `n` of a batched tensor (leading `N` dimension) as
    /// a zero-copy view sharing this tensor's allocation. See
    /// [`crate::Tensor::batch`]. Drives `ef_tensor_batch`.
    pub fn batch(&self, n: usize) -> Result<TensorDyn> {
        // SAFETY: `self.handle` is a live handle for the call's duration.
        let handle =
            unsafe { edgefirst_tensor_ffi::ef_tensor_batch(self.handle.as_ptr(), n as u64) };
        match NonNull::new(handle) {
            Some(h) => Ok(Self::from_handle(h)),
            None => {
                let is_index = ffi_last_error_class()
                    == edgefirst_tensor_ffi::EfErrorClass::BatchIndexOutOfBounds;
                Err(batch_error(n, self.shape().first().copied(), is_index))
            }
        }
    }

    /// Borrow the DMA-BUF file descriptor backing this tensor (Linux only).
    ///
    /// # Errors
    ///
    /// [`Error::NotImplemented`] if this tensor is not DMA-backed, has no
    /// plane 0, or that plane carries no native fd.
    #[cfg(target_os = "linux")]
    pub fn dmabuf(&self) -> Result<std::os::fd::BorrowedFd<'_>> {
        if self.memory() != TensorMemory::DmaBuf {
            return Err(Error::NotImplemented(format!(
                "dmabuf requires DMA-backed tensor, got {:?}",
                self.memory()
            )));
        }
        let plane = self
            .plane0()
            .ok_or_else(|| Error::NotImplemented("dmabuf: this tensor has no plane 0".into()))?;
        if plane.handle < 0 {
            return Err(Error::NotImplemented(
                "dmabuf: this tensor has no native fd (not DMA-backed)".into(),
            ));
        }
        // SAFETY: `plane.handle` is a valid fd owned by the live handle,
        // which outlives the `&self` borrow.
        Ok(unsafe { std::os::fd::BorrowedFd::borrow_raw(plane.handle as i32) })
    }

    /// Clone the DMA-BUF file descriptor backing this tensor (Linux only).
    ///
    /// # Errors
    ///
    /// * [`Error::NotImplemented`] if the tensor is not DMA-backed
    /// * [`Error::IoError`] if the `dup` syscall fails
    #[cfg(target_os = "linux")]
    pub fn dmabuf_clone(&self) -> Result<std::os::fd::OwnedFd> {
        if self.memory() != TensorMemory::DmaBuf {
            return Err(Error::NotImplemented(format!(
                "dmabuf_clone requires DMA-backed tensor, got {:?}",
                self.memory()
            )));
        }
        self.clone_fd()
    }

    /// Borrow the raw `IOSurfaceRef` backing this handle (macOS/iOS).
    ///
    /// Drives `ef_tensor_iosurface_ref`. `None` when the producing library
    /// has no IOSurface (wrong backing, or not Apple).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn iosurface_ref(&self) -> Option<*mut std::ffi::c_void> {
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let p = unsafe { edgefirst_tensor_ffi::ef_tensor_iosurface_ref(self.handle.as_ptr()) };
        if p.is_null() {
            None
        } else {
            Some(p)
        }
    }

    /// Physical IOSurface dimensions in texels, independent of logical shape,
    /// read with `IOSurfaceGetWidth`/`IOSurfaceGetHeight`. `None` when
    /// [`Self::iosurface_ref`] is `None`.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn iosurface_physical_dims(&self) -> Option<(usize, usize)> {
        let surface = self.iosurface_ref()?;
        // SAFETY: `surface` is a live IOSurfaceRef owned by the producing
        // library for at least as long as `self` (the handle) exists.
        unsafe { Some((IOSurfaceGetWidth(surface), IOSurfaceGetHeight(surface))) }
    }

    /// The borrowed `ID3D11Texture2D*` backing this tensor (Windows only).
    /// `None` when the tensor is not texture-backed.
    ///
    /// Drives `ef_tensor_d3d11_texture`. The pointer is valid while the tensor
    /// lives and carries no reference of its own.
    #[cfg(target_os = "windows")]
    pub fn d3d11_texture(&self) -> Option<*mut std::ffi::c_void> {
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let p = unsafe { edgefirst_tensor_ffi::ef_tensor_d3d11_texture(self.handle.as_ptr()) };
        (!p.is_null()).then_some(p)
    }

    /// The texture geometry the HAL chose for this image (Windows only). `None`
    /// when the tensor is not texture-backed. Drives `ef_tensor_d3d11_layout`.
    #[cfg(target_os = "windows")]
    pub fn d3d11_layout(&self) -> Option<crate::d3d11_layout::D3d11ImageLayout> {
        let mut out = edgefirst_tensor_ffi::EfD3d11Layout::default();
        // SAFETY: `self.handle` is live; `out` is a valid local out-param.
        let rc =
            unsafe { edgefirst_tensor_ffi::ef_tensor_d3d11_layout(self.handle.as_ptr(), &mut out) };
        (rc == 0).then_some(crate::d3d11_layout::D3d11ImageLayout {
            dxgi_format: out.dxgi_format,
            texture_width: out.texture_width as usize,
            texture_height: out.texture_height as usize,
            bytes_per_texel: out.bytes_per_texel as usize,
            gl_internal_format: out.gl_internal_format,
        })
    }

    /// A duplicated NT handle the caller owns (Windows only). Drives
    /// `ef_tensor_d3d11_shared_handle`.
    #[cfg(target_os = "windows")]
    pub fn d3d11_shared_handle(&self) -> Result<std::os::windows::io::OwnedHandle> {
        use std::os::windows::io::FromRawHandle;
        // SAFETY: `self.handle` is live for as long as `self` exists.
        let h =
            unsafe { edgefirst_tensor_ffi::ef_tensor_d3d11_shared_handle(self.handle.as_ptr()) };
        if h.is_null() {
            return Err(ffi_error(Error::NotImplemented));
        }
        // SAFETY: the export transferred ownership of a live NT handle.
        Ok(unsafe { std::os::windows::io::OwnedHandle::from_raw_handle(h) })
    }

    /// The fence handle plus value a GPU consumer waits on before reading this
    /// texture (Windows only), or `None` when no GPU write has been recorded.
    ///
    /// Drives `ef_tensor_gpu_completion`. The returned fence handle is an owned
    /// duplicate.
    #[cfg(target_os = "windows")]
    pub fn gpu_completion(&self) -> Result<Option<crate::d3d11::GpuCompletion>> {
        use std::os::windows::io::FromRawHandle;
        let mut fence = std::ptr::null_mut();
        let mut value = 0u64;
        // SAFETY: `self.handle` is live; both out-params are valid locals.
        let rc = unsafe {
            edgefirst_tensor_ffi::ef_tensor_gpu_completion(
                self.handle.as_ptr(),
                &mut fence,
                &mut value,
            )
        };
        if rc != 0 {
            return Err(ffi_error(Error::NotImplemented));
        }
        if fence.is_null() {
            return Ok(None);
        }
        // SAFETY: the export transferred ownership of a live NT handle.
        let fence = unsafe { std::os::windows::io::OwnedHandle::from_raw_handle(fence) };
        Ok(Some(crate::d3d11::GpuCompletion { fence, value }))
    }

    /// The fence value of the newest GPU write recorded on this tensor, or 0
    /// when there is none; every platform, 0 off Windows. See
    /// [`Tensor::gpu_write_value`](crate::Tensor::gpu_write_value).
    ///
    /// Drives `ef_tensor_gpu_write_value` for a `DmaBuf`-kind handle; any other
    /// backing returns 0 without the call.
    pub fn gpu_write_value(&self) -> u64 {
        if self.memory() != TensorMemory::DmaBuf {
            return 0;
        }
        // SAFETY: `self.handle` is live for as long as `self` exists.
        unsafe { edgefirst_tensor_ffi::ef_tensor_gpu_write_value(self.handle.as_ptr()) }
    }

    /// Record that GPU work writing this texture completes at `value` of the
    /// process device's fence (Windows only).
    ///
    /// Drives `ef_tensor_set_gpu_write`, which records a monotonic maximum.
    #[cfg(target_os = "windows")]
    pub fn set_gpu_write(&self, value: u64) -> Result<()> {
        // SAFETY: `self.handle` is live; the export records a monotonic
        // maximum into an atomic, so a shared borrow is all it needs.
        let rc =
            unsafe { edgefirst_tensor_ffi::ef_tensor_set_gpu_write(self.handle.as_ptr(), value) };
        if rc != 0 {
            return Err(ffi_error(Error::NotImplemented));
        }
        Ok(())
    }

    /// Always `None`: the registration lives in the library. Use
    /// [`Self::is_cuda_attached`] or [`Self::cuda_map`].
    #[deprecated(
        since = "0.32.0",
        note = "On the dynamic backend use is_cuda_attached() or cuda_map() — the CudaHandle is not borrowable through this lens."
    )]
    pub fn cuda(&self) -> Option<&crate::cuda::CudaHandle> {
        None
    }

    /// Whether the library tensor carries a CUDA registration.
    pub fn is_cuda_attached(&self) -> bool {
        // SAFETY: `self.handle` is live.
        unsafe { edgefirst_tensor_ffi::ef_tensor_cuda_attached(self.handle.as_ptr()) != 0 }
    }

    /// Attach a GL-buffer CUDA registration on the library tensor.
    pub fn set_cuda_handle(&mut self, h: crate::cuda::CudaHandle) {
        match h.into_client_parts() {
            Ok(parts) => {
                // SAFETY: `parts` keeps the channel alive for the retain
                // `ef_tensor_cuda_attach` takes; the function pointers match
                // `CudaHandle::into_client_parts`.
                let rc = unsafe {
                    edgefirst_tensor_ffi::ef_tensor_cuda_attach(
                        self.handle.as_ptr(),
                        parts.state,
                        parts.resource,
                        parts.size,
                        Some(parts.map_fn),
                        Some(parts.unmap_fn),
                        Some(parts.unregister_fn),
                    )
                };
                if rc != 0 {
                    log::debug!("ef_tensor_cuda_attach failed: {rc}");
                }
            }
            Err(e) => log::debug!("set_cuda_handle: {e}"),
        }
    }

    /// The C library's mapping of this tensor. `mk` is `ef_tensor_cuda_map`
    /// or `ef_tensor_cuda_map_mut`; both return an opaque map that retains
    /// the tensor and is released by `ef_tensor_cuda_unmap`.
    fn ffi_cuda_map(
        &self,
        mk: unsafe extern "C" fn(*const EfTensor) -> *mut std::ffi::c_void,
    ) -> Option<crate::cuda::CudaMap<'_>> {
        // SAFETY: `self.handle` is this tensor's live C handle.
        let map = unsafe { mk(self.handle.as_ptr()) };
        if map.is_null() {
            return None;
        }
        // SAFETY: `map` is the live, non-NULL map `mk` just returned, with no
        // other owner, so the guard may release it exactly once.
        unsafe { crate::cuda::CudaMap::from_ffi(map) }
    }

    /// Fast-fail CUDA map through the library (`ef_tensor_cuda_map`).
    pub fn cuda_map(&self) -> Option<crate::cuda::CudaMap<'_>> {
        self.ffi_cuda_map(edgefirst_tensor_ffi::ef_tensor_cuda_map)
    }

    /// Writable counterpart of [`cuda_map`](Self::cuda_map).
    pub fn cuda_map_mut(&self) -> Option<crate::cuda::CudaMap<'_>> {
        self.ffi_cuda_map(edgefirst_tensor_ffi::ef_tensor_cuda_map_mut)
    }

    /// GL buffer ID for this PBO; `None` when the tensor is not PBO-backed.
    /// Drives `ef_tensor_pbo_id`.
    pub fn pbo_id(&self) -> Option<u32> {
        let mut id: u32 = 0;
        // SAFETY: `self.handle` is live; `id` is a valid out-param.
        let rc = unsafe { edgefirst_tensor_ffi::ef_tensor_pbo_id(self.handle.as_ptr(), &mut id) };
        (rc == 0).then_some(id)
    }

    /// Whether this PBO holds (or is establishing) a CPU mapping; `None` when
    /// the tensor is not PBO-backed.
    ///
    /// Drives `ef_tensor_pbo_is_mapped`, which returns `1` for mapped, `0` for
    /// not mapped and a negative value for a tensor that is not a PBO.
    pub fn pbo_is_mapped(&self) -> Option<bool> {
        // SAFETY: `self.handle` is live for the call.
        let rc = unsafe { edgefirst_tensor_ffi::ef_tensor_pbo_is_mapped(self.handle.as_ptr()) };
        match rc {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }

    /// The C-ABI `PboOpsVtable` address for this PBO, for cross-cdylib export
    /// via [`crate::TensorDesc::ptr`]; `None` when not PBO-backed.
    ///
    /// Drives `ef_tensor_pbo_vtable`. The address is valid while this handle
    /// lives; [`Self::pbo_keepalive`] holds it.
    pub fn pbo_vtable_ptr(&self) -> Option<*const std::ffi::c_void> {
        // SAFETY: `self.handle` is live for the call.
        let p = unsafe { edgefirst_tensor_ffi::ef_tensor_pbo_vtable(self.handle.as_ptr()) };
        (!p.is_null()).then_some(p)
    }

    /// A keepalive holding [`Self::pbo_vtable_ptr`]'s address valid: a retained
    /// reference on this tensor's own handle.
    ///
    /// `None` when the tensor is not PBO-backed, or when the retain fails.
    pub fn pbo_keepalive(&self) -> Option<std::sync::Arc<dyn Send + Sync>> {
        self.pbo_id()?;
        // SAFETY: `self.handle` is live for the call.
        let rc = unsafe { edgefirst_tensor_ffi::ef_tensor_retain(self.handle.as_ptr()) };
        if rc != 0 {
            return None;
        }
        Some(std::sync::Arc::new(HandleKeepalive(self.handle.as_ptr())))
    }

    /// Wrap a producer's host pointer as a type-erased tensor without copying,
    /// aliasing rather than owning it. This is the consumer half of the capsule
    /// protocol's host-memory kinds ([`crate::protocol::kind::MEM`],
    /// [`crate::protocol::kind::SHM`]). Drives `ef_tensor_wrap_host`.
    ///
    /// # Safety
    ///
    /// `ptr` must be non-null, aligned to `dtype`, and valid for
    /// `max(capacity_bytes, shape.product() * dtype.size())` bytes for as long
    /// as the returned tensor, and every view or map sharing its backing, is
    /// used.
    pub unsafe fn from_raw_host_with_capacity(
        ptr: *mut u8,
        shape: &[usize],
        capacity_bytes: usize,
        dtype: DType,
        _name: Option<&str>,
    ) -> Result<Self> {
        let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        // SAFETY: the caller upholds `ptr`'s validity (this function's own
        // contract); `dims` is a live local for the call's duration.
        let handle = unsafe {
            edgefirst_tensor_ffi::ef_tensor_wrap_host(
                ptr,
                capacity_bytes,
                dtype.code(),
                dims.as_ptr(),
                dims.len() as u32,
            )
        };
        match NonNull::new(handle) {
            Some(h) => Ok(Self::from_handle(h)),
            None => Err(Error::InvalidArgument(format!(
                "ef_tensor_wrap_host failed: {}",
                ffi_last_error()
            ))),
        }
    }

    /// Wrap a live `IOSurfaceRef` (macOS/iOS). Looks up the surface's
    /// `IOSurfaceID` and drives [`Self::from_iosurface_id`].
    ///
    /// # Safety
    ///
    /// `surface_ref` must be a valid live `IOSurfaceRef`. `shape` must match
    /// the IOSurface's pixel dimensions and chosen element type.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub unsafe fn from_iosurface(
        surface_ref: *mut std::ffi::c_void,
        shape: &[usize],
        dtype: DType,
        name: Option<&str>,
    ) -> Result<Self> {
        if surface_ref.is_null() {
            return Err(Error::InvalidArgument(
                "from_iosurface: surface_ref is null".into(),
            ));
        }
        // SAFETY: caller guarantees `surface_ref` is a live IOSurfaceRef.
        let id = unsafe { IOSurfaceGetID(surface_ref) };
        Self::from_iosurface_id(id, shape, dtype, name)
    }

    /// Wraps an image IOSurface, reading its pixel format and planes.
    ///
    /// `420v` and `420f` give NV12 as a multiplane tensor: luma and chroma
    /// each wrap `surface` at their plane's offset and row stride. `yuvs` gives
    /// YUYV and `BGRA` gives BGRA, as one tensor at the surface's row stride.
    /// The surface is retained for the tensor's lifetime. Colorimetry is not
    /// set.
    ///
    /// # Errors
    ///
    /// `InvalidArgument` for a null surface or any other pixel format;
    /// `InvalidShape` when a plane does not hold its image, or an NV12 surface
    /// has an odd width or height.
    ///
    /// # Safety
    ///
    /// `surface` must be a live `IOSurfaceRef`.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub unsafe fn from_planar_iosurface(
        surface: *mut std::ffi::c_void,
        name: Option<&str>,
    ) -> Result<Self> {
        use crate::tensor_dyn::derived::{planar_iosurface_parts, PlanarParts};

        // SAFETY: the caller guarantees `surface` is a live IOSurfaceRef.
        let parts = unsafe { planar_iosurface_parts(surface)? };
        match parts {
            PlanarParts::Nv12 { luma, chroma } => {
                // SAFETY: as above; each wrap takes its own retain.
                let (mut y, mut uv) = unsafe {
                    (
                        Self::from_iosurface(surface, &[luma.height, luma.width], DType::U8, name)?,
                        Self::from_iosurface(
                            surface,
                            &[chroma.height, luma.width],
                            DType::U8,
                            name,
                        )?,
                    )
                };
                y.set_plane_offset(luma.offset);
                y.set_row_stride_unchecked(luma.bytes_per_row);
                uv.set_plane_offset(chroma.offset);
                uv.set_row_stride_unchecked(chroma.bytes_per_row);
                let typed = |t: Self| {
                    t.into_u8()
                        .map_err(|_| Error::InvalidArgument("plane is not a u8 tensor".into()))
                };
                crate::Tensor::<u8>::from_planes(typed(y)?, typed(uv)?, PixelFormat::Nv12)
                    .map(Self::from)
            }
            PlanarParts::Packed {
                format,
                width,
                height,
                bytes_per_row,
            } => {
                // SAFETY: as above.
                let mut t = unsafe {
                    Self::from_iosurface(
                        surface,
                        &[height, width, format.channels()],
                        DType::U8,
                        name,
                    )?
                };
                t.set_format(format)?;
                t.set_row_stride_unchecked(bytes_per_row);
                Ok(t)
            }
        }
    }

    /// IOSurfaceID for cross-process sharing. `None` when this handle has
    /// no IOSurface (wrong backing, or `IOSurfaceGetID` returned 0).
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn iosurface_id(&self) -> Option<u32> {
        let surface = self.iosurface_ref()?;
        let id = unsafe { IOSurfaceGetID(surface) };
        if id == 0 {
            None
        } else {
            Some(id)
        }
    }

    /// Wrap a live IOSurface named by its cross-process `IOSurfaceID`
    /// (macOS/iOS only). This is the consumer half of the capsule protocol's
    /// [`crate::protocol::kind::IOSURFACE`].
    ///
    /// Drives `ef_tensor_from_iosurface_id`, which looks up, checks and retains
    /// the surface inside the library.
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub fn from_iosurface_id(
        id: u32,
        shape: &[usize],
        dtype: DType,
        _name: Option<&str>,
    ) -> Result<Self> {
        let dims: Vec<u64> = shape.iter().map(|&d| d as u64).collect();
        // SAFETY: `dims` is a live local for the call's duration.
        let handle = unsafe {
            edgefirst_tensor_ffi::ef_tensor_from_iosurface_id(
                id,
                dtype.code(),
                dims.as_ptr(),
                dims.len() as u32,
            )
        };
        match NonNull::new(handle) {
            Some(h) => Ok(Self::from_handle(h)),
            None => Err(Error::InvalidArgument(format!(
                "ef_tensor_from_iosurface_id failed: {}",
                ffi_last_error()
            ))),
        }
    }

    /// Wrap an existing `ID3D11Texture2D` as a type-erased tensor (Windows
    /// only). See
    /// [`Tensor::from_d3d11_texture`](crate::Tensor::from_d3d11_texture).
    ///
    /// Drives `ef_tensor_from_d3d11_texture`, passing `width`/`height` as the
    /// allocation shape.
    ///
    /// # Safety
    ///
    /// `texture` must be null or a live `ID3D11Texture2D` created on the HAL
    /// device ([`crate::d3d11::device()`]).
    #[cfg(target_os = "windows")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_d3d11_texture(
        texture: *mut std::ffi::c_void,
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        access: CpuAccess,
        name: Option<&str>,
    ) -> Result<Self> {
        let dims = allocation_dims(format, width, height, "from_d3d11_texture")?;
        let format = format_cstring(format)?;
        let name = name_cstring(name)?;
        // SAFETY: the caller guarantees `texture`; `dims`, `format` and
        // `name` are live locals for the call's duration.
        let handle = unsafe {
            edgefirst_tensor_ffi::ef_tensor_from_d3d11_texture(
                texture,
                dtype.code(),
                dims.as_ptr(),
                dims.len() as u32,
                format.as_ptr(),
                access.code(),
                name.as_ref().map_or(std::ptr::null(), |n| n.as_ptr()),
            )
        };
        match NonNull::new(handle) {
            Some(h) => Ok(Self::from_handle(h)),
            None => Err(ffi_error(Error::InvalidArgument)),
        }
    }

    /// Open a shared texture by its NT handle as a type-erased tensor
    /// (Windows only). See
    /// [`Tensor::from_d3d11_shared_handle`](crate::Tensor::from_d3d11_shared_handle).
    ///
    /// Drives `ef_tensor_from_d3d11_shared_handle`. Both handles stay the
    /// caller's: the export duplicates what it keeps.
    ///
    /// # Safety
    ///
    /// `handle` must be an NT shared handle of a D3D11 texture, valid in
    /// this process, and `completion`'s handle a shared fence handle.
    #[cfg(target_os = "windows")]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_d3d11_shared_handle(
        handle: std::os::windows::io::RawHandle,
        width: usize,
        height: usize,
        format: PixelFormat,
        dtype: DType,
        access: CpuAccess,
        completion: Option<(std::os::windows::io::RawHandle, u64)>,
        name: Option<&str>,
    ) -> Result<Self> {
        let dims = allocation_dims(format, width, height, "from_d3d11_shared_handle")?;
        let format = format_cstring(format)?;
        let name = name_cstring(name)?;
        let (fence, fence_value) = completion.unwrap_or((std::ptr::null_mut(), 0));
        // SAFETY: the caller guarantees both handles; `dims`, `format` and
        // `name` are live locals for the call's duration.
        let tensor = unsafe {
            edgefirst_tensor_ffi::ef_tensor_from_d3d11_shared_handle(
                handle,
                dtype.code(),
                dims.as_ptr(),
                dims.len() as u32,
                format.as_ptr(),
                access.code(),
                fence,
                fence_value,
                name.as_ref().map_or(std::ptr::null(), |n| n.as_ptr()),
            )
        };
        match NonNull::new(tensor) {
            Some(h) => Ok(Self::from_handle(h)),
            None => Err(ffi_error(Error::InvalidArgument)),
        }
    }

    /// Build a PBO-backed tensor from a client's callback channel. Drives
    /// `ef_tensor_wrap_pbo`.
    ///
    /// # Safety
    /// `state`, `map_fn` and `unmap_fn` must satisfy
    /// [`crate::client_state_pbo_ops`]'s contract.
    pub unsafe fn from_client_pbo(
        state: crate::EfClientState,
        map_fn: crate::EfPboMapFn,
        unmap_fn: crate::EfPboUnmapFn,
        buffer_id: u32,
        size: usize,
        shape: &[usize],
        dtype: DType,
    ) -> Result<TensorDyn> {
        let dims: Vec<u64> = shape.iter().map(|d| *d as u64).collect();
        // SAFETY: `dims` is a live local for the call; the state and op
        // pointers carry this function's own contract into the library.
        let handle = unsafe {
            edgefirst_tensor_ffi::ef_tensor_wrap_pbo(
                state,
                buffer_id,
                size,
                dtype.code(),
                dims.as_ptr(),
                dims.len() as u32,
                Some(map_fn),
                Some(unmap_fn),
            )
        };
        match NonNull::new(handle) {
            Some(h) => Ok(Self::from_handle(h)),
            // The library's recorded error class, or `InvalidArgument` when it
            // recorded none.
            None => Err(ffi_error(Error::InvalidArgument)),
        }
    }

    /// The per-kind construction switch [`TensorDyn::import_descriptor`]
    /// (`derived.rs`) drives, once the descriptor has been validated and its
    /// `dtype`/`shape` decoded.
    pub(crate) fn import_storage(
        desc: &crate::TensorDesc,
        shape: &[usize],
        dtype: DType,
    ) -> Result<Self> {
        match desc.kind {
            #[cfg(target_os = "linux")]
            crate::protocol::kind::DMABUF => {
                let fd = crate::protocol::dup_descriptor_fd(desc)?;
                Self::from_fd(fd, shape, dtype, None)
            }
            #[cfg(not(target_os = "linux"))]
            crate::protocol::kind::DMABUF => {
                Err(Error::NotImplemented("dma-buf import off Linux".into()))
            }
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            crate::protocol::kind::IOSURFACE => {
                let id = crate::protocol::descriptor_surface_id(desc)?;
                Self::from_iosurface_id(id, shape, dtype, None)
            }
            #[cfg(not(any(target_os = "macos", target_os = "ios")))]
            crate::protocol::kind::IOSURFACE => Err(Error::NotImplemented(
                "IOSurface import off Apple platforms".into(),
            )),
            crate::protocol::kind::MEM | crate::protocol::kind::SHM => {
                crate::protocol::check_descriptor_host_ptr(desc)?;
                // SAFETY: the caller guarantees the producer's keepalive
                // outlives the returned tensor, and `desc.capacity` comes from
                // the same producer as `ptr`/`shape`.
                unsafe {
                    Self::from_raw_host_with_capacity(
                        desc.ptr.0,
                        shape,
                        desc.capacity as usize,
                        dtype,
                        None,
                    )
                }
            }
            crate::protocol::kind::PBO => {
                let buffer_id = crate::protocol::descriptor_pbo_buffer_id(desc)?;
                // SAFETY: `desc.ptr` carries a `PboOpsVtable` address under
                // `kind::PBO`, held live for this call by the producer's
                // capsule keepalive.
                let parts = unsafe { crate::read_pbo_vtable_parts(desc.ptr.0 as *const _) }?;
                // SAFETY: the parts came from a table this crate built.
                unsafe {
                    Self::from_client_pbo(
                        parts.state,
                        parts.map_fn,
                        parts.unmap_fn,
                        buffer_id,
                        desc.capacity as usize,
                        shape,
                        dtype,
                    )
                }
            }
            #[cfg(target_os = "windows")]
            crate::protocol::kind::D3D11_TEXTURE => {
                let (tex, completion) = crate::protocol::descriptor_d3d11_handles(desc)?;
                // SAFETY: the caller guarantees the producer's keepalive
                // outlives the returned tensor, so both handles are live for
                // this call. An undeclared CPU access imports as `ReadWrite`.
                unsafe {
                    let (format, width, height) =
                        crate::protocol::descriptor_d3d11_geometry(desc, tex, shape)?;
                    Self::from_d3d11_shared_handle(
                        tex,
                        width,
                        height,
                        format,
                        dtype,
                        crate::protocol::flags::cpu_access(desc.flags)
                            .unwrap_or(CpuAccess::ReadWrite),
                        completion,
                        None,
                    )
                }
            }
            #[cfg(not(target_os = "windows"))]
            crate::protocol::kind::D3D11_TEXTURE => Err(Error::NotImplemented(
                "D3D11 texture import off Windows".into(),
            )),
            k => Err(Error::NotImplemented(format!(
                "tensor interop kind {k} cannot be imported by this build"
            ))),
        }
    }

    /// Descriptor for the cross-package tensor protocol; see
    /// [`crate::protocol`]'s module docs.
    pub fn descriptor(&self) -> crate::TensorDesc {
        self.descriptor_pinned(None)
    }

    /// Descriptor carrying `pin`'s host address when one is given, built by
    /// [`crate::protocol::from_parts`]. The native handle is the PBO's GL
    /// buffer id, the D3D11 texture's NT handle, or plane 0's handle.
    pub fn descriptor_pinned(&self, pin: Option<&crate::HostPin<'_>>) -> crate::TensorDesc {
        let memory = self.memory();
        let handle: i64 = match memory {
            TensorMemory::Pbo => self.pbo_id().map(|id| id as i64).unwrap_or(-1),
            // The duplicate kept in `descriptor_texture_handle`.
            #[cfg(target_os = "windows")]
            TensorMemory::DmaBuf => self.descriptor_texture_handle_value(),
            _ => self.plane0().map(|p| p.handle).unwrap_or(-1),
        };
        // The fence handle is the process device's, borrowed; the fence value
        // comes from `ef_tensor_gpu_write_value`.
        #[cfg(target_os = "windows")]
        let (fence_handle, sync) = match memory {
            TensorMemory::DmaBuf => (
                crate::d3d11::device()
                    .map(|d| d.exported_fence_handle_value())
                    .unwrap_or(0),
                Some(self.gpu_write_value()).filter(|v| *v != 0),
            ),
            _ => (0, None),
        };
        #[cfg(not(target_os = "windows"))]
        let (fence_handle, sync) = (0, None);
        crate::protocol::from_parts(crate::protocol::DescParts {
            dims: self.shape(),
            memory,
            dtype: self.dtype(),
            fourcc: self.format().map(|f| f.to_fourcc()).unwrap_or(0),
            format: self.format(),
            row_stride: self.row_stride(),
            handle,
            colorimetry: self.colorimetry().map(|c| c.pack()).unwrap_or(0),
            capacity: self.capacity_bytes() as u64,
            pin,
            pbo_vtable_ptr: self.pbo_vtable_ptr(),
            fence_handle,
            sync,
            cpu_access: self.cpu_access(),
            view: self.is_view(),
        })
    }

    /// The value of the texture handle this tensor keeps for its
    /// descriptors, duplicating it on first use; `-1` when this tensor has
    /// no D3D11 texture behind it.
    #[cfg(target_os = "windows")]
    fn descriptor_texture_handle_value(&self) -> i64 {
        use std::os::windows::io::AsRawHandle;
        self.descriptor_texture_handle
            .get_or_init(|| self.d3d11_shared_handle().ok())
            .as_ref()
            .map(|h| h.as_raw_handle() as usize as i64)
            .unwrap_or(-1)
    }
}

/// Turn `ef_tensor_batch`'s `NULL` into the error [`crate::Tensor::batch`]
/// would have returned.
///
/// An out-of-range index with a known leading dimension becomes
/// [`Error::BatchIndexOutOfBounds`] built from `n` and `leading`. Every other
/// failure goes through [`ffi_error`].
fn batch_error(n: usize, leading: Option<usize>, class_is_index: bool) -> Error {
    match leading {
        Some(batch) if class_is_index => Error::BatchIndexOutOfBounds { index: n, batch },
        _ => ffi_error(Error::InvalidShape),
    }
}

/// Turn a builder's sticky errno into a typed [`Error`] whose message is
/// [`ffi_last_error`]'s text.
fn builder_error(errno: std::ffi::c_int) -> Error {
    let msg = ffi_last_error();
    // The inverse of `errno_for` in `tensor-capi/src/map.rs`.
    match errno {
        libc::ENOTSUP => Error::NotImplemented(msg),
        libc::ENOMEM => Error::IoError(std::io::Error::from_raw_os_error(libc::ENOMEM)),
        libc::ERANGE => Error::ShapeMismatch(msg),
        libc::EACCES => Error::InvalidOperation(msg),
        libc::EIO => Error::IoError(std::io::Error::other(msg)),
        libc::EINVAL => Error::InvalidArgument(msg),
        other => Error::InvalidArgument(format!("{msg} (errno {other})")),
    }
}

/// Derive a [`BufferIdentity`] from an fd's `(st_dev, st_ino)`.
#[cfg(unix)]
#[allow(clippy::unnecessary_cast)]
fn identity_from_stat(stat: &nix::sys::stat::FileStat) -> BufferIdentity {
    let key = ((stat.st_dev as u64) << 32) ^ (stat.st_ino as u64);
    BufferIdentity::derived(IdentityKind::DmaBuf, key)
}

/// `NUL`-terminate a [`PixelFormat`]'s wire string for a C call.
///
/// # Errors
///
/// [`Error::InvalidArgument`] if the wire string contains a NUL byte.
fn format_cstring(format: PixelFormat) -> Result<std::ffi::CString> {
    std::ffi::CString::new(format.as_str())
        .map_err(|e| Error::InvalidArgument(format!("pixel format string contains a NUL: {e}")))
}

/// `NUL`-terminate an optional debug name for a C call. `None` stays `None`.
///
/// # Errors
///
/// [`Error::InvalidArgument`] if the name contains a NUL byte.
#[cfg(target_os = "windows")]
fn name_cstring(name: Option<&str>) -> Result<Option<std::ffi::CString>> {
    name.map(std::ffi::CString::new)
        .transpose()
        .map_err(|e| Error::InvalidArgument(format!("name contains a NUL: {e}")))
}

/// `width`x`height` as the `uint64_t[]` shape a D3D11 constructor takes.
///
/// # Errors
///
/// [`Error::InvalidArgument`] for a `width`/`height` the format has no
/// allocation shape for (a zero dimension, or an odd one for a subsampled
/// format).
#[cfg(target_os = "windows")]
fn allocation_dims(
    format: PixelFormat,
    width: usize,
    height: usize,
    what: &str,
) -> Result<Vec<u64>> {
    format
        .allocation_shape(width, height)
        .map(|s| s.iter().map(|&d| d as u64).collect())
        .ok_or_else(|| {
            Error::InvalidArgument(format!(
                "{what}: invalid dimensions {width}x{height} for format {format:?}"
            ))
        })
}

/// Encode `Option<TensorMemory>` as the `(has_memory, memory)` pair every
/// `ef_tensor_image_*` constructor takes; `(0, 0)` requests auto-selection.
fn memory_code(memory: Option<TensorMemory>) -> (std::ffi::c_int, u32) {
    match memory {
        Some(m) => (1, m.code()),
        None => (0, 0),
    }
}

/// Read the calling thread's most recent `tensor-capi` failure message. The
/// text is advisory, may be empty, and is valid until this thread's next call.
pub(crate) fn ffi_last_error() -> String {
    // SAFETY: `ef_tensor_last_error_message` always returns a valid,
    // NUL-terminated string (possibly empty), read immediately and copied
    // out before any further `tensor-capi` call on this thread.
    unsafe {
        let ptr = edgefirst_tensor_ffi::ef_tensor_last_error_message();
        if ptr.is_null() {
            return String::new();
        }
        std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

/// Read the calling thread's most recent `tensor-capi` failure class.
fn ffi_last_error_class() -> edgefirst_tensor_ffi::EfErrorClass {
    use edgefirst_tensor_ffi::EfErrorClass as C;
    // SAFETY: a plain thread-local read on the producing side; no pointers.
    let code = unsafe { edgefirst_tensor_ffi::ef_tensor_last_error_class() };
    const CLASSES: [C; 9] = [
        C::InvalidArgument,
        C::InvalidShape,
        C::InsufficientCapacity,
        C::BatchIndexOutOfBounds,
        C::RegionOutOfBounds,
        C::NotSupported,
        C::InvalidOperation,
        C::AllocationFailed,
        C::QuantizationInvalid,
    ];
    // A code outside `CLASSES` reads as `Unspecified`.
    CLASSES
        .into_iter()
        .find(|c| *c as u32 == code)
        .unwrap_or(C::Unspecified)
}

/// Rebuild a typed [`Error`] from the class and message the last failing
/// `ef_tensor_*` call recorded.
///
/// The message is the producing side's own. An `Unspecified` class yields
/// `fallback`. The classes whose variants carry structured fields
/// (`RegionOutOfBounds`, `BatchIndexOutOfBounds`, `InsufficientCapacity`) map
/// to [`Error::ShapeMismatch`].
pub(crate) fn ffi_error(fallback: fn(String) -> Error) -> Error {
    use edgefirst_tensor_ffi::EfErrorClass as C;
    let msg = ffi_last_error();
    match ffi_last_error_class() {
        C::InvalidArgument => Error::InvalidArgument(msg),
        C::InvalidShape => Error::InvalidShape(msg),
        C::InsufficientCapacity | C::BatchIndexOutOfBounds | C::RegionOutOfBounds => {
            Error::ShapeMismatch(msg)
        }
        C::NotSupported => Error::NotImplemented(msg),
        C::InvalidOperation => Error::InvalidOperation(msg),
        C::AllocationFailed => Error::IoError(std::io::Error::other(msg)),
        C::QuantizationInvalid => Error::InvalidArgument(msg),
        C::Unspecified => fallback(msg),
    }
}

/// Holds one `ef_tensor_retain` on a handle for a keepalive's lifetime, keeping
/// an address borrowed from that handle valid.
struct HandleKeepalive(*mut EfTensor);

// SAFETY: an `ef_tensor` handle is refcounted and documented as safe to
// cross threads (`ef_tensor_retain`); this holds one reference and does
// nothing else with the pointer.
unsafe impl Send for HandleKeepalive {}
unsafe impl Sync for HandleKeepalive {}

impl Drop for HandleKeepalive {
    fn drop(&mut self) {
        // SAFETY: releases exactly the reference `pbo_keepalive` took.
        unsafe { edgefirst_tensor_ffi::ef_tensor_free(self.0) }
    }
}

/// The keepalive behind [`TensorDyn::map_bytes`]'s pin. Holds its own
/// `ef_tensor_retain`'d reference and, on drop, unmaps and then releases it.
struct MapKeepalive(*mut EfTensor);

// SAFETY: the referenced handle is `libedgefirst_tensor`'s own C ABI object,
// designed to cross threads; ownership is tracked by refcount
// (`ef_tensor_retain`/`ef_tensor_free`), not by Rust's aliasing rules.
unsafe impl Send for MapKeepalive {}
unsafe impl Sync for MapKeepalive {}

impl Drop for MapKeepalive {
    fn drop(&mut self) {
        // SAFETY: `self.0` was retained when the pin was built and is released
        // here, after the unmap.
        unsafe {
            edgefirst_tensor_ffi::ef_tensor_unmap(self.0);
            edgefirst_tensor_ffi::ef_tensor_free(self.0);
        }
    }
}

/// Prints `dtype` and `shape` only.
impl fmt::Debug for TensorDyn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorDyn")
            .field("dtype", &self.dtype())
            .field("shape", &self.shape())
            .finish()
    }
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    fn IOSurfaceGetWidth(surface: *mut std::ffi::c_void) -> usize;
    fn IOSurfaceGetHeight(surface: *mut std::ffi::c_void) -> usize;
    fn IOSurfaceGetID(surface: *mut std::ffi::c_void) -> u32;
}
